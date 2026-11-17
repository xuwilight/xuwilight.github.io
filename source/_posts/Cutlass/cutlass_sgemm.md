---
title: Cutlass 学习笔记（五）Cutlass SGEMM 代码分析
date: 2025-08-19 12:00:00
tags: [CUTLASS, GEMM]
categories: [Cutlass 学习笔记,Cutlass]
description: 用 DefaultGemmUniversal、DefaultMmaCore、MmaSimt 和 MmaPipelined 拆解 CUTLASS 的 SIMT SGEMM，从主机侧模板实例化一路下探到线程级蛇形 FFMA，并汇总几种 SGEMM 实现方式与常用优化手段。
---

本文分析 CUTLASS 的 SIMT SGEMM，也就是走 CUDA Core 常规 FFMA 乘加的那条路径。它靠三级分块（线程块、warp、线程各一块）和软件流水隐藏访存延迟，核心是标量 FFMA 寄存器累加。整条调用链从主机侧的 GemmUniversalAdapter 进入 kernel::GemmUniversal，再经 DefaultGemmUniversal、DefaultMma、DefaultMmaCore 一层层定型，最后落到 MmaPipelined 的双缓冲主循环，最内层由 warp::MmaSimt 调 thread::Mma 发出 FFMA。

下面先给出精简后的调用示例，再自顶向下逐层拆解，最后汇总几种实现方式和优化手段。

```cpp
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

// (A * B)^T = B^T * A^T
template <class TA, class TB, class TC>
void cutlass_sgemm_tt(int M, int N, int K, TC alpha, TA const *A, int lda,
                      TB const *B, int ldb, TC beta, TC *C, int ldc) {

    using cutlass_simt_sgemm_128x64_8x2_tt_align1_base =
        typename cutlass::gemm::kernel::DefaultGemmUniversal<
            float, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1, // transposed B operand
            float, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1, // transposed A operand
            float, cutlass::layout::RowMajor,
            float,
            cutlass::arch::OpClassSimt,
            cutlass::arch::Sm50,
            cutlass::gemm::GemmShape<128, 64, 8>,
            cutlass::gemm::GemmShape<64, 32, 8>,
            cutlass::gemm::GemmShape<1, 1, 1>,
            cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
            2,
            cutlass::arch::OpMultiplyAdd>::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<cutlass_simt_sgemm_128x64_8x2_tt_align1_base>;

    Gemm gemm;

    long long int batch_stride_A = static_cast<long long int>(M) * K;
    long long int batch_stride_B = static_cast<long long int>(K) * N;
    long long int batch_stride_C = static_cast<long long int>(M) * N;
    long long int batch_stride_D = static_cast<long long int>(M) * N;

    Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        1,
        {alpha, beta},
        A, B, C, C,
        batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_D,
        lda, ldb, ldc, ldc);

    gemm(args);
}

// A, B, C are device pointers
extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K) {
    cutlass_sgemm_tt(M, N, K, 1.0f, A, K, B, N, 0.0f, C, M);
    cudaDeviceSynchronize();
}

// nvcc cutlass_sgemm.cu -o cutlass -arch=sm_80 -O3 -I ../../cutlass/include --expt-relaxed-constexpr
```

在 A100 上跑 8192×4096×6144 的性能：

cublas time = 22.026697 ms, TFLPOS = 18.718961, mfu = 0.959947

cutlass time = 22.918276 ms, TFLPOS = 17.990746, mfu = 0.922602

## 主机侧实例化

SGEMM 通过上面这组模板参数创建。内层名字 128x64_8x2_tt_align1 拆开看，block 大小是 128×64，blockK 是 8、有 2 个 stage；tt 表示 A 和 B 都要转置，转置后原矩阵在内存里是 row-major；align1 表示 A 和 B 都按 1 元素对齐。

因为 solve 里按 cutlass_sgemm_tt(M, N, K, 1.0f, A, K, B, N, 0.0f, C, M) 的顺序传参，第一个矩阵操作数是 B、第二个是 A，要计算的原始问题是 $C = A^T B^T$ 形式的 TT GEMM。利用 $(AB)^T = B^T A^T$，模板里就先写 B 再写 A。B 转置后在内存里是 column-major，对应第一个 layout::ColumnMajor；A 转置后同样是 column-major，对应第二个；C 转置后从 column-major 变成 row-major，对应 RowMajor。

GemmShape<128, 64, 8> 是 CTA 处理的数据大小，GemmShape<64, 32, 8> 是一个 warp 处理的数据大小，因此一个 CTA 有 4 个 warp、128 个线程。GemmIdentityThreadblockSwizzle<8> 表示 CTA swizzle 的大小是 8×8。

kernel 用的是 GemmUniversalAdapter，按要求完成初始化即可。启动时经过 gemm_universal_adapter.h 和 gemm_universal.h，最后在 gemm_universal_base.h 里 launch。

## DefaultGemmUniversal

DefaultGemmUniversal 在 default_gemm_universal.h 中包了一层 kernel::GemmUniversal，具体实现在 include/cutlass/gemm/kernel/gemm_universal.h 中，这里实现了最外层的计算流程，也就是 mma 加 epilogue。

另外，在 default_gemm_universal.h 中还定义了 DefaultGemm，DefaultGemm 由 DefaultMma 和 RegularEpilogue 组成。include/cutlass/gemm/threadblock/default_mma.h 中定义了 DefaultMma。

```cpp
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 2, Operator, false, SharedMemoryClearOption::kNone,
                  GatherA, GatherB, PermuteALayout, PermuteBLayout> {

  static_assert(platform::is_same<LayoutC, layout::RowMajor>::value
             || platform::is_same<LayoutC, layout::AffineRankN<2>>::value,
             "simt epilogue must be row major");

  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      arch::OpClassSimt, 2, Operator>;

  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA,
          GatherA, PermuteALayout>;

  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB,
          GatherB, PermuteBLayout>;

  using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      LayoutC, typename MmaCore::MmaPolicy>;
};
```

这个特化对应 Stages 等于 2 的情况。MmaCore 封装了从 shared memory 加载数据并计算的流程，IteratorA 和 IteratorB 负责从 global memory 高效读取 tile。三者共同组成 ThreadblockMma，也就是实现 2-stage pipeline 的 MmaPipelined。

## DefaultMmaCore

DefaultMmaCore 按 A、B 的主序组合有四个特化，这里只看 A、B 都是 column-major 的那个，定义在 include/cutlass/gemm/threadblock/default_mma_core_simt.h。

```cpp
template <typename Shape_, typename WarpShape_,
          typename ElementA_, typename ElementB_, typename ElementC_,
          typename LayoutC_, typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, ElementA_,
                      layout::ColumnMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, 2, Operator_> {
  using Shape = Shape_;
  ...
  static int const PartitionsK = Shape::kK / WarpShape::kK; // 8 / 8 = 1

  using Operator = Operator_;

  // 每个 CTA 里 warp 的排布
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    PartitionsK
  >; // 2 2 1

  static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value; // 32
  static int const kThreads = WarpCount::kCount * kWarpSize; // 4 * 32 = 128
  static int const kElementsPerAccess = 1;
```

这段定义了 warp 大小、线程总数，以及单次访问的元素数。SIMT 的 warp 是 32 个线程，WarpCount 的 M、N、K 分别是 2、2、1，所以 kCount 为 4，每 CTA 共 128 个线程。

接着定义写 shared memory 的迭代器。为了让寄存器到 smem 的加载高效，smem 一般沿 M 或 N 方向连续存放，所以这里 SmemLayoutA 取 ColumnMajor、SmemLayoutB 取 RowMajor。

这跟 SIMT 计算 GEMM 的方式有关。SIMT 以外积（outer product）的形式累加 $C \mathrel{+}= A \times B$，某个 k 步要取 A 的一列和 B 的一行。A 用列主序、B 用行主序，同一个 warp 的线程读取的数据在 smem 里就是连续或等步长的，能减少 bank conflict。

```cpp
  using SmemLayoutA = layout::ColumnMajor;
  using SmemLayoutB = layout::RowMajor;

  using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kM, Shape::kK>,
    kThreads, kElementsPerAccess
  >;

  using SmemIteratorA = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA,
    1, IteratorThreadMapA
  >;

  using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kN>,
    kThreads, kElementsPerAccess
  >;

  // B 从 gmem 连续读、写 smem 时旋转 90 度，避免 bank conflict
  using SmemThreadMapB = transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapB>;

  using SmemIteratorB = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB,
    0, SmemThreadMapB
  >;
```

### 线程映射与 PitchLinear

IteratorThreadMapA/B 决定每个线程搬运 tile 里的哪些元素，是最抽象的一层。CUTLASS 不直接用 Row/Column 描述，而用 PitchLinear 的 (C, S) 坐标：C 是连续维（stride 为 1），S 是跨步维（stride 为 leading dimension）。对 column-major 矩阵，C 是行、S 是列；对 row-major 矩阵，C 是列、S 是行。一套模板就能覆盖两种主序。

PitchLinearStripminedThreadMap 把二维 tile 平铺给一维排列的线程。tile 比线程数大时要分多次（strip-mine）搬运：先沿连续维 C 铺满线程，再折到跨步维 S。它按连续维长度和线程总数的关系分两种分支，分别给出 initial_offset 和 Delta 步长。

B 的线程映射外面多套了一层 TransposePitchLinearThreadMapSimt。B 从 global memory 读入时希望线程沿行方向连续读以满足合并访问，但写进 smem 时按 SmemLayoutB 的 row-major 摆放，若沿用原始坐标写入会产生 bank conflict。这层适配器把下层算出的 (C, S) 交换成 (S, C) 再交给下游，相当于把写 smem 的访问模式旋转 90 度，同时保留 gmem 读取的连续性。

这层转置要求每个线程只处理标量（kElementsPerAccess == 1），否则连续的 float4 会被拆成 4 个分散的 float，破坏向量化，所以它只用于 SIMT 标量场景。

PitchLinearStripminedThreadMap 定义在 include/cutlass/transform/pitch_linear_thread_map.h 中。以 A 为例，Shape 是 (M=128, K=8)，kElementsPerAccess=1，于是 ShapeVec = PitchLinearShape<128, 8>。kThreads 为 128，满足 Threads >= ShapeVec::kContiguous，走短连续维分支：Iterations = (1, 8)，Delta = (1, 1)，StorageShape = (128, 8)。initial_offset 把线程按 $((tid \bmod 128),\ tid / 128)$ 映射，128 个线程正好铺满一行的 128 个元素。

```cpp
  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id) {
    return TensorCoord(
      (thread_id % Detail::ShapeVec::kContiguous) * kElementsPerAccess,
      thread_id / Detail::ShapeVec::kContiguous);
  }
```

SmemIteratorA 就是这个 map 的包装，构造时传入 advance rank 1，表示主循环沿 K 维（rank 1）前进。B 的 Shape 是 (K=8, N=128)，ShapeVec = PitchLinearShape<8, 128>。连续维只有 8、小于 128 个线程，走另一分支：Iterations = (1, 8)，Delta = (1, 16)，StorageShape = (8, 128)。线程沿 8 宽的连续维排开，每 16 个换一组跨步，128 个线程覆盖整块。

接下来是 warp 级的线程布局。

```cpp
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>(); // 8
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM; // 4
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM; // 8
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN; // 8
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1; // 2
  static const int numElementsA = 128 / sizeof_bits<ElementA>::value; // 4
  static const int numElementsB = 128 / sizeof_bits<ElementB>::value; // 4
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM); // 4
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN); // 4

  static int const kPaddingN = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementB>::value); // 4

  using LaneMmaShape = cutlass::gemm::GemmShape<LaneM, LaneN, 1>;
  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // 8×4
      cutlass::layout::RowMajorInterleaved<LaneLayout>,
      LaneMmaShape
  >;
```

LaneM、LaneN 是 warp 内单条 SIMT 指令在 M、N 方向一次处理几个元素，取 128 bit 的位宽和 ThreadTile 的较小值，float 下就是 4，对应一次 128-bit 的 shared load。LaneMmaShape 把这两个粒度合成线程级 mma 形状 (4, 4, 1)。

MmaSimtPolicy 的第二个参数是 RowMajorInterleaved<LaneLayout>，描述 32 个 lane 到 warp 内 (WarpNumThreadsM, WarpNumThreadsN) 二维网格的排布。这里 M、N 各 8、4 个线程，lane 并不按自然行主序铺：RowMajorInterleaved<K> 把行号拆成 row / K 和 row % K 两部分，坐标映射为

```cpp
offset = (row / K) * stride + column * K + (row % K)
```

也就是每隔 K 行一组交错，让物理上相邻的 lane 落在不同的行块里。配合 LaneM 的 128-bit 加载，相邻线程取到的数据落在不同的寄存器 bank 上，减少寄存器堆的读冲突；交错步长 LaneLayout 在 ThreadTileM、ThreadTileN 都大于 4 时取 2，否则取 1。

最后用这些组件拼出 MmaWarpSimt 和 MmaPolicy。MmaPolicy 的第二个 MatrixShape 给 B 的 smem 加 kPaddingN 的 skew，用来压 bank conflict。

```cpp
  using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
      WarpShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Policy>;

  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<0, 0>,
    MatrixShape<0, kPaddingN>, // skew for B matrix to avoid SMEM bank conflicts
    WarpCount::kK
  >;
```

### MmaSimt

warp 级算子 mma::MmaSimt 内部定义线程级的 thread::Mma。它的形状是 warp tile 在 M、N 方向各除以 warp 内的线程数，K 方向取 LaneMmaShape 的 1，也就是每个线程负责 (8, 8, 1) 的一小块。

```cpp
  using ThreadMma = thread::Mma<
    GemmShape<
      Shape::kM / Policy::WarpShape::kRow,     // 64 / 8 = 8
      Shape::kN / Policy::WarpShape::kColumn,  // 32 / 4 = 8
      Policy::LaneMmaShape::kK>,               // 1
    ElementA, ThreadLayoutA, ElementB, ThreadLayoutB,
    ElementC, LayoutC, arch::OpMultiplyAdd, dp4a_type
  >;
```

IteratorA、IteratorC 负责把寄存器里的 fragment 按这个线程形状切分和访问（B 的结构对称，从略）。

```cpp
  using IteratorA = MmaSimtTileIterator<
    MatrixShape<Shape::kM, Policy::LaneMmaShape::kK>,
    Operand::kA, ElementA, LayoutA, Policy, PartitionsK, Shape::kK
  >;
  using FragmentA = typename IteratorA::Fragment;
  using TransformedFragmentA = FragmentA;

  ...
  using IteratorC = MmaSimtTileIterator<
    MatrixShape<Shape::kM, Shape::kN>,
    Operand::kC, ElementC, LayoutC, Policy
  >;
  using FragmentC = typename ThreadMma::FragmentC;
```

MmaSimtTileIterator 以 A 为例，取该线程在 M、N 方向的 ThreadShape（(8, 1)），再按 LaneMmaShape 的 M=4 把行方向切成 2 次加载，fragment 共 8 个元素。因为 MmaSimtPolicy 用了 RowMajorInterleaved，构造时线程 id 要经 lane_layout.inverse 换算回真实坐标。

```cpp
  using ThreadShape = MatrixShape<
    Shape::kRow / Policy::WarpShape::kRow, // 64 / 8 = 8
    Shape::kColumn                          // LaneMmaShape::kK = 1
  >;
  using Iterations = MatrixShape<
    ThreadShape::kRow / Policy::LaneMmaShape::kM, // 8 / 4 = 2
    ThreadShape::kColumn                          // 1
  >;
  using Fragment = Array<Element, ThreadShape::kCount>; // 8

private:
  cutlass::TensorRef<Array<Element, Policy::LaneMmaShape::kM>, layout::ColumnMajor> ref_; // 4

  MmaSimtTileIterator(TensorRef ref, int lane_id) {
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();
    MatrixCoord lane_offset = lane_layout.inverse(lane_id) *
      MatrixCoord(Policy::LaneMmaShape::kM, 0);
    ref.add_coord_offset(lane_offset);
    ref_.reset(
      reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM> *>(ref.data()),
      ref.stride(0) / Policy::LaneMmaShape::kM);
  }
```

load 从 smem 搬运 fragment。每次按 LaneMmaShape 的 4 个元素走一条 128-bit 的 shared load，8 个元素分两次取，正好对上 LaneM=4 的向量化宽度。

```cpp
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {
    Array<Element, Policy::LaneMmaShape::kM> *dst_ptr =
      reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM> *>(&frag);
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kColumn; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < Iterations::kRow; ++m) {
        auto ptr = ref_.data() + ref_.offset({m * Policy::WarpShape::kRow, k}) + pointer_offset / Policy::LaneMmaShape::kM;
        arch::shared_load(dst_ptr[m + k * Iterations::kRow], ptr);
      }
    }
  }
```

线程级 mma 最终落到 include/cutlass/gemm/thread/mma_sm50.h 的 MmaGeneric，代码最内层发出的就是这条标量 FFMA（指令形状 1×1×1）。

```cpp
template <typename Shape_, typename ElementA_, typename LayoutA_,
          typename ElementB_, typename LayoutB_,
          typename ElementC_, typename LayoutC_, typename Operator_>
struct MmaGeneric {
  using Shape = Shape_;
  using FragmentA = Array<ElementA_, Shape::kMK>;
  using FragmentB = Array<ElementB_, Shape::kKN>;
  using FragmentC = Array<ElementC_, Shape::kMN>;

  using MmaOp = arch::Mma<
    gemm::GemmShape<1,1,1>, 1,
    ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_, LayoutC_, Operator_>;

  static bool const kMultipleOf2 = ((Shape::kM % 2 == 0) && (Shape::kN % 2 == 0));
  static bool const kAllFp32 = platform::is_same<ElementA_, float>::value &&
      platform::is_same<ElementB_, float>::value &&
      platform::is_same<ElementC_, float>::value;

  void operator()(FragmentC &D, FragmentA const &A, FragmentB const &B, FragmentC const &C) {
    TensorRef<ElementA_ const, LayoutA_> a_ref(
      reinterpret_cast<ElementA_ const *>(&A), LayoutA_::packed({Shape::kM, Shape::kK}));
    TensorRef<ElementB_ const, LayoutB_> b_ref(
      reinterpret_cast<ElementB_ const *>(&B), LayoutB_::packed({Shape::kK, Shape::kN}));
    TensorRef<ElementC_, LayoutC_> d_ref(
      reinterpret_cast<ElementC_ *>(&D), LayoutC_::packed(make_Coord(Shape::kM, Shape::kN)));
    MmaOp mma_op;
    D = C;

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Shape::kK; ++k) {
      #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 860)
      if constexpr (kMultipleOf2 && kAllFp32) {
        // 2x2 zigzag，每个 2x2 小块内按 TL、BL、BR、TR 连发 4 条 FFMA
        for (int n = 0; n < Shape::kN; n += 2) {
          for (int m = 0; m < Shape::kM; m += 2) {
            int m_serpentine = (n % 4) ? (Shape::kM - 2 - m) : m;
            Array<ElementC_,1> d; Array<ElementA_,1> a; Array<ElementB_,1> b;

            // top-left
            { MatrixCoord mn(m_serpentine, n), mk(m_serpentine, k), kn(k, n);
              d[0]=d_ref.at(mn); a[0]=a_ref.at(mk); b[0]=b_ref.at(kn);
              mma_op(d,a,b,d); d_ref.at(mn)=d[0]; }
            // bottom-left
            { MatrixCoord mn(m_serpentine+1, n), mk(m_serpentine+1, k), kn(k, n);
              d[0]=d_ref.at(mn); a[0]=a_ref.at(mk); b[0]=b_ref.at(kn);
              mma_op(d,a,b,d); d_ref.at(mn)=d[0]; }
            // bottom-right
            { MatrixCoord mn(m_serpentine+1, n+1), mk(m_serpentine+1, k), kn(k, n+1);
              d[0]=d_ref.at(mn); a[0]=a_ref.at(mk); b[0]=b_ref.at(kn);
              mma_op(d,a,b,d); d_ref.at(mn)=d[0]; }
            // top-right
            { MatrixCoord mn(m_serpentine, n+1), mk(m_serpentine, k), kn(k, n+1);
              d[0]=d_ref.at(mn); a[0]=a_ref.at(mk); b[0]=b_ref.at(kn);
              mma_op(d,a,b,d); d_ref.at(mn)=d[0]; }
          }
        }
      } else
      #endif
      {
        // 标量 fallback，逐列蛇形
        for (int n = 0; n < Shape::kN; ++n) {
          for (int m = 0; m < Shape::kM; ++m) {
            int m_serpentine = (n % 2) ? (Shape::kM - 1 - m) : m;
            MatrixCoord mn(m_serpentine, n), mk(m_serpentine, k), kn(k, n);
            Array<ElementC_,1> d; Array<ElementA_,1> a; Array<ElementB_,1> b;
            d[0]=d_ref.at(mn); a[0]=a_ref.at(mk); b[0]=b_ref.at(kn);
            mma_op(d,a,b,d); d_ref.at(mn)=d[0];
          }
        }
      }
    }
  }
};
```

这段 MmaGeneric 是单线程内的矩阵乘，A、B、C、D 都是寄存器里的 Fragment，用 TensorRef 包成二维视图后按 (m, k)、(k, n)、(m, n) 坐标取数，每次调用 arch::Mma 发出一条标量 FFMA。外层 k 循环遍历规约维，所以它算的是每个线程负责的那一小块 C 的累加。

内层有一处蛇形索引，偶数列 m 从小到大、奇数列 m 从大到小，遍历方向在相邻列之间折返。这样一列结束时行号停在边界，下一列从同一端接着走，相邻两次访问的寄存器位置连续，减少寄存器堆访问的跳变。

Ampere（`__CUDA_ARCH__ >= 860`）且全部为 FP32 时走 2×2 展开分支，N、M 都以 2 为步长，每个 2×2 小块内按左上、左下、右下、右上四个点各发一条 FFMA。这个顺序让相邻两条指令复用同一个源操作数：左上和左下共用 B、左下和右下共用 A、右下和右上共用 B，右上又和下一轮的左上共用 A。GPU 的 operand collector 可以把刚读过的操作数留在缓存里，连续指令重用同一源操作数时省掉一次寄存器堆读，提升 FP32 FFMA 的吞吐。

## MmaPipelined

单线程的计算看完，接着看线程块级的流水是怎么组织的。MmaPipelined 实现在 include/cutlass/gemm/threadblock/mma_pipelined.h。

```cpp
class MmaPipelined : public MmaBase<Shape_, Policy_, 2> {
public:
  using Base = MmaBase<Shape_, Policy_, 2>;
  using Shape = Shape_;
  using IteratorA = IteratorA_;
  using IteratorB = IteratorB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using Policy = Policy_;
  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;
```

类里先声明访问 global memory 和 shared memory 的迭代器。构造函数从上一层接收 thread_idx、warp_idx、lane_idx 等参数，初始化 smem 迭代器并定位当前 warp 在 CTA tile 里的位置。

```cpp
  MmaPipelined(
    typename Base::SharedStorage &shared_storage,
    int thread_idx, int warp_idx, int lane_idx,
    TransformA transform_A = TransformA(), TransformB transform_B = TransformB()
  ):
    Base(shared_storage, thread_idx, warp_idx, lane_idx),
    smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
    smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
    transform_A_(transform_A), transform_B_(transform_B),
    smem_write_stage_idx(0)
  {
    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    this->warp_tile_iterator_A_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset({Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }
```

MmaPipelined 继承自 MmaBase，实现在 include/cutlass/gemm/threadblock/mma_base.h。MmaBase 算出每个 warp 的处理范围、stage 数，并建出带 padding 的多级 smem buffer。

```cpp
  using WarpGemm = typename Policy::Operator::Shape;

  using WarpCount = GemmShape<Shape::kM / WarpGemm::kM,  // 128 / 64 = 2
                              Shape::kN / WarpGemm::kN,  // 64 / 32 = 2
                              Shape::kK / WarpGemm::kK>; // 8 / 8 = 1

  static int const kWarpGemmIterations = (WarpGemm::kK / Operator::Policy::MmaShape::kK); // 1
  static int const kStages = Stages; // 2

  class SharedStorage {
   public:
    using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow,
                               Shape::kK * kStages + Policy::SmemPaddingA::kColumn>;
    using ShapeB = MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow,
                               Shape::kN + Policy::SmemPaddingB::kColumn>;
    ...
  };

  typename Operator::IteratorA warp_tile_iterator_A_;
  typename Operator::IteratorB warp_tile_iterator_B_;
```

warp_tile_iterator_A_、warp_tile_iterator_B_ 负责每个 warp 从 smem 中定位自己的 tile，内部配合 LaneShape 和 RowInterleave 完成 lane 级寻址。

两级流水的入口是 operator()，依次调用 prologue 预取前几级、gmem_wait 同步、再进入 gemm_iters 主循环。

```cpp
  void operator()(int gemm_k_iterations, FragmentC &accum,
                  IteratorA iterator_A, IteratorB iterator_B,
                  FragmentC const &src_accum) {
    prologue(iterator_A, iterator_B, gemm_k_iterations);
    gmem_wait();
    accum = src_accum;
    gemm_iters(gemm_k_iterations, accum, iterator_A, iterator_B);
  }
```

prologue 先把第一块 A、B 的 global fragment 搬进 smem，再推进写 stage，对应双缓冲里预取的那一级。

```cpp
  void prologue(IteratorA &iterator_A, IteratorB &iterator_B, int &gemm_k_iterations) {
    FragmentA tb_frag_A;
    tb_frag_A.clear();
    iterator_A.load(tb_frag_A);
    ++iterator_A;

    FragmentB tb_frag_B;
    tb_frag_B.clear();
    iterator_B.load(tb_frag_B);
    ++iterator_B;

    this->smem_iterator_A_.store(transform_A_(tb_frag_A));
    this->smem_iterator_B_.store(transform_B_(tb_frag_B));
    advance_smem_write_stage();
  }

  void gmem_wait() { __syncthreads(); }
```

gemm_iters 是主循环。warp fragment 开两组，按 warp_mma_k % 2 交替读写，实现 smem 读与计算的重叠；每轮开始时预取下一块 global tile 到 tb_frag，并在最后一个 kgroup 把当前 tb_frag 写回 smem、同步、推进 stage，让 global 搬运和 warp 计算流水起来。

```cpp
  void gemm_iters(int gemm_k_iterations, FragmentC &accum,
                  IteratorA &iterator_A, IteratorB &iterator_B) {
    using WarpFragmentA = typename Operator::FragmentA;
    using WarpFragmentB = typename Operator::FragmentB;

    // 两组 warp fragment，交替读写以重叠 smem load 与 mma
    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B[2];
    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
    ++this->warp_tile_iterator_A_;
    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);
    ++this->warp_tile_iterator_B_;

    FragmentA tb_frag_A;
    FragmentB tb_frag_B;
    iterator_A.clear_mask(gemm_k_iterations <= 1);
    iterator_B.clear_mask(gemm_k_iterations <= 1);

    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
        if (warp_mma_k == Base::kWarpGemmIterations - 1) {
          this->smem_iterator_A_.store(transform_A_(tb_frag_A));
          this->smem_iterator_B_.store(transform_B_(tb_frag_B));
          gmem_wait();
          advance_smem_stages();
        }

        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);
        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        if (warp_mma_k == 0) {
          tb_frag_A.clear();
          iterator_A.load(tb_frag_A);
          ++iterator_A;
          tb_frag_B.clear();
          iterator_B.load(tb_frag_B);
          ++iterator_B;
          iterator_A.clear_mask(gemm_k_iterations <= 2);
          iterator_B.clear_mask(gemm_k_iterations <= 2);
        }

        warp_mma(accum, warp_frag_A[warp_mma_k % 2], warp_frag_B[warp_mma_k % 2], accum);
      }
    }
  }
```

## 调用流程

把前面几层接起来，一次 SGEMM 的实际调用链是这样的。

主机侧 gemm(args) 进入 GemmUniversalAdapter::operator()，先 initialize 处理参数、查 smem 用量和工作区，再 run：走到 gemm_universal_base.h 计算 grid 维度、把 kernel 函数指针和参数打包后 launch。grid 的维度由问题大小和 CTA tile 决定，kernel::GemmUniversal::operator() 里再按 GemmIdentityThreadblockSwizzle 把 block idx 换算成当前的 CTA tile 坐标（做 CTA rasterization），据此给 IteratorA/B 定位起始 global tile。

进入 threadblock 层，MmaPipelined::operator() 跑 prologue、gmem_wait、gemm_iters 主循环。主循环里每个 warp 通过 warp_tile_iterator 从 smem 取自己那块 fragment，调 warp::MmaSimt::operator()，它在内部按 lane 把 fragment 切好，交给 thread::Mma，最终由 MmaGeneric 发出标量 FFMA 累加到寄存器。K 方向迭代结束后，kernel 调 epilogue 把寄存器里的 accum 乘 alpha、加 beta·C，按 LayoutC 写回 global。

整条链可以概括成：主机模板定型 → launch kernel → CTA 认领 tile → warp 取 smem fragment → 线程发 FFMA 累加 → epilogue 写回。分块和流水都在前半段把访存隐藏掉，后半段只剩密集的 FFMA。

## 几种 SGEMM 实现方式

CUTLASS 里 float 的 SIMT SGEMM 不止本文这一条路径，按用的搬运机制和所在的层大致分四种。

第一种是本文这条 sm50 两级流水，走 DefaultGemm → DefaultMma → MmaPipelined。default_gemm.h 里 SIMT 且 ArchTag 不是 Sm80 的那个特化会把 ArchTag 强制成 arch::Sm50，固定 2 个 stage，global 到 smem 用普通的 load/store 加 __syncthreads 同步。它兼容性最好，任何架构都能退到这条。

第二种是 Sm80 多级流水，同样是 DefaultGemm 出来，但当 ArchTag 是 Sm80 且 stage 数可配时改走 MmaMultistage。global 到 smem 换成 cp.async 异步拷贝、多级 buffer，省掉寄存器中转，重叠度更高。

第三种是不经过 threadblock 层、直接用 warp 级 MmaSimt 手搓，example 20_simt_canonical 就是这种。它自己定义 warp tile 和 fragment，把 MmaSimt 当成积木拼，适合教学或需要精细控制的小 kernel。

第四种是 3.x 的 collective-builder 路径。CollectiveBuilder 对 OpClassSimt、float 只有 SM100 一个特化（sm100_simt_builder.inl），用的是 SM100_2x1x1_F32F32F32F32 这个 cute atom，mainloop 走 cp.async，并限定 TileShape_K 必须是 16。SM80、SM90 在 3.x 里没有 collective 层的 FP32 SIMT，它们的 float collective 走的是 TF32 tensor-op。



## 优化手段

散在各层里的优化手段主要有以下几种：

分块与映射层面。三级分块（CTA 128×64、warp 64×32、线程 8×8）把大矩阵切成能塞进寄存器和 smem 的小块；warp 线程数按 simt_get_warp_threads_m 取成近似方形（8×4），让每个线程的 tile 接近方阵，FFMA 的数据复用更均衡；PitchLinear 的 (C, S) 抽象用一套模板执行 row/column 两种主序；TransposePitchLinearThreadMapSimt 把写 smem 的访问旋转 90 度，兼顾 gmem 合并读和 smem 低冲突写。

shared memory 层面。SmemLayoutA 列主序、SmemLayoutB 行主序，配合外积累加让 warp 内读取连续或等步长；MmaPolicy 给 B 加 kPaddingN 的 skew、simt_transpose_padding 算出 padding，专门错开 bank；MmaSimtTileIterator 每次走一条 128-bit 的 ld.shared.v4.u32（arch::shared_load），把 4 个 float 一次搬进寄存器。

寄存器层面。thread::Mma 的蛇形索引让相邻列的遍历方向折返，寄存器位置连续；Ampere 上的 2×2 zigzag 让相邻几条 FFMA 复用同一源操作数（TL→BL 复用 B、BL→BR 复用 A、BR→TR 复用 B），省掉寄存器堆读；RowMajorInterleaved 把 lane 错开到不同行块，减少寄存器 bank 冲突。

流水与边界层面。两级软件流水用双缓冲的 warp fragment [2] 和 smem 双 stage，把 smem 读和 FFMA 重叠；Sm80 的 cp.async 多级流水进一步把 global 搬运也藏进去；clear_mask 在主循环末尾关闭越界读取，处理 K 不是 tile 整数倍的尾巴；CUTLASS_PRAGMA_UNROLL 把最内层循环完全展开，让编译器排好 FFMA 的发射顺序。

再往上是 CTA 级。GemmIdentityThreadblockSwizzle 把相邻 CTA 映射到 L2 上相邻的 tile，提高缓存命中；这些手段叠在一起，才让 SIMT 的标量 FFMA 也能跑到接近 cublas 的 MFU。
