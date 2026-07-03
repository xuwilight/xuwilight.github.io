---
title: Cutlass 学习笔记（四）Cutlass SGEMM 代码分析
date: 2025-08-18 12:00:00
tags: [CUTLASS, GEMM]
categories: [Cutlass 学习笔记,Cutlass]
description: 介绍了 cutlass 中 sgemm的基本实现和计算流程。
---

# cutlass sgemm 代码

```cpp
#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include "utils.h"

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

// (A * B)^T = B^T * A^T
template <class TA, class TB, class TC>
void cutlass_sgemm_tt(int M, int N, int K, TC alpha, TA const *A, int lda, TB const *B, int ldb, TC beta, TC *C, int ldc)
{

    using cutlass_simt_sgemm_128x64_8x2_tt_align1_base =
        typename cutlass::gemm::kernel::DefaultGemmUniversal<
            float, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1, // transposed B operand
            float, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1, // transposed A operand
            float, cutlass::layout::RowMajor,
            float,
            cutlass::arch::OpClassSimt,
            cutlass::arch::Sm50,
            cutlass::gemm::GemmShape<128, 64, 8>,
            cutlass::gemm::GemmShape<64, 32, 8>,
            cutlass::gemm::GemmShape<1, 1, 1>,

            cutlass::epilogue::thread::LinearCombination<
                float,
                1,
                float,
                float>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
            2,
            cutlass::arch::OpMultiplyAdd>::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<cutlass_simt_sgemm_128x64_8x2_tt_align1_base>;

    Gemm gemm;

    long long int batch_stride_A = static_cast<long long int>(M) * K;
    long long int batch_stride_B = static_cast<long long int>(K) * N;
    long long int batch_stride_C = static_cast<long long int>(M) * N;
    long long int batch_stride_D = static_cast<long long int>(M) * N;

    Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,                        // 模式：标准 GEMM
        {M, N, K},                                                      // Problem Size
        1,                                                              // Batch Count (设为 1 表示非 Batched)
        {alpha, beta},                                                  // Epilogue 参数
        A, B, C, C,                                                     // 指针
        batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_D, // Batch Strides
        lda, ldb, ldc, ldc                                              // Leading Dimension Strides
    );

    gemm(args);
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K)
{
    cutlass_sgemm_tt(M, N, K, 1.0f, A, K, B, N, 0.0f, C, M);
    cudaDeviceSynchronize();
}

// nvcc cutlass_sgemm.cu -o cutlass -lcuda -lcublas -arch=sm_80 -O3 -I ../../cutlass-3.8/include  --ptxas-options=-v --expt-relaxed-constexpr  && ./cutlass
int main()
{
    srand(1234);

    // leetGPU:  A is M×N, B is N×K, C is M×K
    // our impl: A is M×K, B is K×N, C is M×N

    int M = 4096, N = 4096, K = 4096;

    // M = 67, N = 67, K = 67;
    // M = 63, N = 31, K = 64;
    // M = 3, N = 3, K = 3;
    // M = 1027,N = 1027,K = 1027;
    M = 8192, N = 4096, K = 6144; // performance case

    using T = float;

    thrust::host_vector<T> h_A(M * K);
    thrust::host_vector<T> h_B(N * K);
    thrust::host_vector<float> h_C(M * N);
    thrust::host_vector<float> h_C1(M * N);
    thrust::host_vector<float> mma_res(M * N);
    thrust::host_vector<float> ref_res(M * N);

    for (int i = 0; i < M * K; ++i)
    {
        h_A[i] = static_cast<T>(rand() % 9 * 1.0 / 10);
        // printf(" %f ", h_A[i]);
    }
    // printf("\n");
    for (int i = 0; i < N * K; ++i)
    {
        h_B[i] = static_cast<T>(rand() % 9 * 1.0 / 10);
        // printf(" %f ", h_B[i]);
    }
    // printf("\n");

    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<float> d_C = h_C;
    thrust::device_vector<float> d_C1 = h_C1;

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;
    // C is column-major
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A.data().get(), K, d_B.data().get(), N, &beta, d_C1.data().get(), M);
    ref_res = d_C1;

    solve(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    mma_res = d_C;

    test_gemm(ref_res.data(), mma_res.data(), M, N, K);

    int benchmark = 1;
    if (benchmark)
    {
        float flops = 2.0 * M * N * K;
        float h100 = 19.5e12;

        std::function<void()> cublas_func = [&]()
        {
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A.data().get(), K, d_B.data().get(), N, &beta, d_C1.data().get(), M);
        };

        std::function<void()> custom_func = [&]()
        {
            solve(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
        };

        run_benchmark(cublas_func, "cublas", flops, h100);
        run_benchmark(custom_func, "mma", flops, h100);
    }
    return 0;
}
```

在 A100 上性能：

cublas time = 22.026697 ms, TFLPOS = 18.718961, mfu = 0.959947

mma time = 22.918276 ms, TFLPOS = 17.990746, mfu = 0.922602

# HOST 侧

cutlass 上 sgemm 通过下面的模板参数创建。128x64_8x2_tt_align1 表示 block 的大小是 128×64，8×2 表示 blockK 的大小是 8，有 2 个 stage。tt 表示 A 和 B 都需要转置，也就是 A 和 B 都是 row-major。align1 表示 A 和 B 都是 1 字节对齐。

因为 ，所以实际传参的时候第一个位置是矩阵 B 的参数，转置后 row-major 也变成了 column-major。同理 C 转置后从 column-major 变成了 row-major。

```cpp
    using cutlass_simt_sgemm_128x64_8x2_tt_align1_base =
        typename cutlass::gemm::kernel::DefaultGemmUniversal<
            float, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1, // transposed B operand
            float, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1, // transposed A operand
            float, cutlass::layout::RowMajor,
            float,
            cutlass::arch::OpClassSimt,
            cutlass::arch::Sm50,
            cutlass::gemm::GemmShape<128, 64, 8>,
            cutlass::gemm::GemmShape<64, 32, 8>,
            cutlass::gemm::GemmShape<1, 1, 1>,

            cutlass::epilogue::thread::LinearCombination<
                float,
                1,
                float,
                float>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
            2,
            cutlass::arch::OpMultiplyAdd>::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<cutlass_simt_sgemm_128x64_8x2_tt_align1_base>;

    Gemm gemm;

    long long int batch_stride_A = static_cast<long long int>(M) * K;
    long long int batch_stride_B = static_cast<long long int>(K) * N;
    long long int batch_stride_C = static_cast<long long int>(M) * N;
    long long int batch_stride_D = static_cast<long long int>(M) * N;

    Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,                        // 模式：标准 GEMM
        {M, N, K},                                                      // Problem Size
        1,                                                              // Batch Count (设为 1 表示非 Batched)
        {alpha, beta},                                                  // Epilogue 参数
        A, B, C, C,                                                     // 指针
        batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_D, // Batch Strides
        lda, ldb, ldc, ldc                                              // Leading Dimension Strides
    );

    gemm(args);
```

GemmShape<128, 64, 8>是 CTA 处理的数据大小，GemmShape<64, 32, 8>是一个 warp 处理的数据大小。所以这里会有 4 个 warp，128 个线程。

GemmIdentityThreadblockSwizzle<8>表示 CTA swizzle 的大小是 8×8。

因为 gemm kernel 用的是 GemmUniversalAdapter，所以要按照上面的方式进行初始化。启动时会经过 gemm_universal_adapter.h 到 gemm_universal.h，最后在 gemm_universal_base.h 中启动。

# DefaultGemmUniversal

再看 DefaultGemmUniversal，在 default_gemm_universal.h 中包了层 kernel::GemmUniversal，具体实现在 cutlass-3.8/include/cutlass/gemm/kernel/gemm_universal.h 中。这里实现了最外层的计算流程，也就是 mma epilogue。

另外，在 default_gemm_universal.h 中还定义了 DefaultGemm，DefaultGemm 由 DefaultMma 和 RegularEpilogue 组成。

cutlass-3.8/include/cutlass/gemm/threadblock/default_mma.h 中定义了 DefaultMma。

```cpp
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 2, Operator, false, SharedMemoryClearOption::kNone,
                  GatherA, GatherB, PermuteALayout, PermuteBLayout> {

  static_assert(platform::is_same<LayoutC, layout::RowMajor>::value
             || platform::is_same<LayoutC, layout::AffineRankN<2>>::value,
             "simt epilogue must be row major");

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      arch::OpClassSimt, 2, Operator>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA,
          GatherA, PermuteALayout>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB,
          GatherB, PermuteBLayout>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      LayoutC, typename MmaCore::MmaPolicy>;
};
```

这里 MmaCore 包括了从 shared memory 中加载数据并计算的流程。IteratorA 和 IteratorB 是用于高效读取 gemm 数据的迭代器。他们共同组成了 ThreadblockMma，这个实现了具体的 2-stage pipeline。

# DefaultMmaCore

根据 A 和 B 不同的主序有不同的实现，具体在 cutlass-3.8/include/cutlass/gemm/threadblock/default_mma_core_simt.h 中。函数如下：

```cpp
/// Partial specialization:
///
///   A: column-major
///   B: column-major
///   Operator: simt class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, ElementA_,
                      layout::ColumnMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, 2, Operator_
                     > {
  using Shape = Shape_;
...
  static int const PartitionsK = Shape::kK / WarpShape::kK; // 8 / 8 = 1

  /// Default Operator
  using Operator = Operator_;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    PartitionsK
  >; // 2 2 1

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value; // 4

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize; // 4 * 32 = 128

  static int const kElementsPerAccess = 1;    
```

前面定义了 warp 的大小和线程数量，以及一个线程处理多少数据。

下面定义了加载共享内存的迭代器。在计算时，为了能高效加载共享内存数据到寄存器，一般都会把 smem 定义成在 bM 或 bN 方向连续的 layout。所以 SmemLayoutA = layout::ColumnMajor，SmemLayoutB = layout::RowMajor;

这与 SIMT 核心计算 GEMM 的方式有关（通常是外积 Outer Product 形式的累加）。

在计算

C+=A×B

的某个 k 步时，我们需要加载 A 的一列（或者列的一部分）和 B 的一行（或者行的一部分）。

让 A 列主序、B 行主序，可以使得在 Shared Memory 中，同一个 Warp 的线程读取的数据在物理地址上是连续的（或者步长固定的），从而减少 Bank Conflict（存储体冲突），提高读取效率。

```cpp
  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajor;
  using SmemLayoutB = layout::RowMajor;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kM, Shape::kK>,
    kThreads,
    kElementsPerAccess
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kM, Shape::kK>, 
    ElementA,
    SmemLayoutA,
    1,
    IteratorThreadMapA
  >;

  /// ThreadMap of iterator B
  using IteratorThreadMapB =  transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kN>,
    kThreads,
    kElementsPerAccess
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapB = transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapB>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kK, Shape::kN>, 
    ElementB,
    SmemLayoutB,
    0,
    SmemThreadMapB
  >;
```

#### 3. 线程映射 (Thread Map)

这是最抽象的部分，它定义了“**哪个线程负责搬运 Tile 中的哪一部分数据**”。    

```cpp
using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
  layout::PitchLinearShape,
  kThreads,
  kElementsPerAccess
>;
    
```

PitchLinear: CUTLASS 内部为了通用性，不直接用 Row/Column，而是用 Contiguous（连续维）和 Strided（跨步维）来描述。

对于 ColumnMajor，M 是连续维，K 是跨步维。

Stripmined: “条带化挖掘”。意思是如果 Tile 很大（例如 128x8），而线程数有限（例如 32），线程需要循环多次（Strip-mine）才能把整个 Tile 的数据搬完。

作用：IteratorThreadMapA 计算出了一个映射表：线程 ID tid 应该读写 Smem 中的哪些坐标 $(m, k)$。

#### 4. SMEM 迭代器 (Smem Iterator)

```cpp
using SmemIteratorA = transform::threadblock::RegularTileIterator<
  MatrixShape, 
  ElementA,
  SmemLayoutA,
  1,
  IteratorThreadMapA
>;
    
```

含义：这是一个 C++ 类，封装了指针运算。

作用：Kernel 运行时，利用这个 Iterator，线程就可以方便地执行 iterator.load() 或 iterator.store()，它会自动根据 SmemLayoutA 计算物理地址偏移，并根据 IteratorThreadMapA 决定当前线程该访问哪个位置。

这里的 1 是 AdvanceRank，表示迭代器在主循环中前进一步时，是沿着 K 维（Rank 1）前进。

#### 5. B 矩阵的特殊处理 (Transpose)

这里有一段非常有意思的代码：    

```cpp
// ThreadMap of iterator B
using IteratorThreadMapB =  transform::PitchLinearStripminedThreadMap<
  layout::PitchLinearShape,
  kThreads,
  kElementsPerAccess
>;

// Transpose the ThreadMap of iterator A (原文注释这里可能有误，实际上是 Transpose B 的 Map)
using SmemThreadMapB = transform::TransposePitchLinearThreadMapSimt;
    
```

为什么 B 需要 TransposeThreadMap？

通常从 Global Memory 加载数据时，为了合并访问（Coalescing），我们希望线程读取连续的地址。

但是，我们将 B 存入 Shared Memory 时定义为了 SmemLayoutB = RowMajor。

如果不做变换，线程 ID 与数据坐标的映射可能导致：从 Global Memory 读取是高效的，但写入 Shared Memory 时产生严重的 Bank Conflict；或者反之。

TransposePitchLinearThreadMapSimt 的作用是旋转线程的视窗。它调整了线程 ID 到 $(k, n)$ 坐标的映射关系，使得：

1. 加载阶段：线程能顺着 Global Memory 的连续维读取（满足 Coalescing）。
1. 存储阶段：线程写入 Shared Memory 时，配合 SmemLayoutB，也能尽量避免冲突。

#### 总结：这些定义在干什么？

这段代码在构建 GEMM 主循环（Mainloop）中的 **“流水线”结构**：

1. 数据容器：它决定了 A 和 B 的 Tile 在 GPU Shared Memory 里怎么摆放（A 竖着放，B 横着放），以便计算核心（CUDA Core）取数最快。
1. 搬运工指南：它定义了 SmemIterator，告诉每个线程：“你负责搬运这块 128x8 矩阵中的哪几个点”。
1. 性能优化：通过 TransposePitchLinearThreadMapSimt，它悄悄地转换了 B 矩阵的线程分配方式，以平衡 Global Memory 读取效率和 Shared Memory 写入效率。

**什么是 PitchLinear？**

这是一个非常专业且深入的问题。要理解这两个类，必须先理解 CUTLASS 中最核心的抽象概念之一：**PitchLinear**。

我将分三部分为您解释：什么是 PitchLinear，PitchLinearStripminedThreadMap 是怎么分配工作的，以及 Transpose 版本起到了什么魔改作用。

#### 1. 什么是 PitchLinear (跨步线性布局)？

在 CUDA 编程中，我们经常纠结于 Row-Major（行主序）和 Column-Major（列主序）。

Row-Major: 同一行的元素在内存中连续（Stride=1），跨行需要跳跃（Stride=LeadingDimension）。

Column-Major: 同一列的元素在内存中连续，跨列需要跳跃。

为了写出通用的模板代码，CUTLASS 引入了 **PitchLinear** 坐标系 $(C, S)$，屏蔽了行列的区别：

Contiguous (C, 连续维)：内存地址连续变化的那一维（Stride = 1）。

Strided (S, 跨步维)：内存地址跳跃变化的那一维（Stride = LD）。

**映射关系：**

如果是 Column-Major 矩阵：$C$ 对应 Row Index，$S$ 对应 Column Index。

如果是 Row-Major 矩阵：$C$ 对应 Column Index，$S$ 对应 Row Index。

**一句话总结**：在 PitchLinear 视角下，数据永远是“沿着 C 轴连续存储，沿着 S 轴堆叠”的。

#### 2. PitchLinearStripminedThreadMap 的工作原理

这个类的名字拆解开来就是它的功能：

PitchLinear: 在 $(C, S)$ 坐标系下工作。

Stripmined: “条带化挖掘/循环分块”。意思是如果一个 Tile 太大，线程一次搬不完，就需要分多次（Iterations）搬运，每次搬一个条带。

ThreadMap: 决定“我是几号线程，我该从哪里开始搬”。

#### 核心逻辑分析

代码中有两个分支（通过 platform::conditional 实现），取决于**连续维的长度**和**线程总数**的关系。

假设我们有一个 Tile 形状为 $(C, S)$，有 $T$ 个线程。

**情况 A：连续维很短（****Threads >= Contiguous****）**

场景：比如 Tile 是 $8 \times 128$（$C=8, S=128$），线程数 32。

布局：32 个线程足够把宽度为 8 的一行直接铺满，甚至还能铺好几行。

行为：线程会先填满 C 维，然后折行到 S 维。

代码对应：

initial_offset:

coord.c = tid % C (在行内对齐)

coord.s = tid / C (换行)

Delta (步长):

下次循环时，线程只需在 S 维方向跳跃。

**情况 B：连续维很长（****Threads < Contiguous****）**

场景：比如 Tile 是 $128 \times 8$（$C=128, S=8$），线程数 32。

布局：32 个线程排成一排，也填不满长度为 128 的 C 维。

行为：所有线程都沿着 C 维排开。搬完当前位置后，下次循环所有线程沿着 C 维向右跳跃（Strip-mine）。

代码对应：

initial_offset:

coord.c = tid (直接按 ID 排列)

coord.s = 0 (都在第 0 行)

Delta (步长):

contiguous = kThreads (下次循环，向右跳过线程总数个位置)

strided = 0 (不换行，直到这一行搬完)

**总结 ****initial_offset**** 函数：**

它实现了标准的**光栅扫描顺序（Raster Scan Order）**。如果把 Tile 看作一个 2D 图像，线程 ID 就是像素点索引，从左到右，从上到下填充。

#### 3. TransposePitchLinearThreadMapSimt 的工作原理

这个类是一个**适配器（Adapter）**。它包裹了上面的 PitchLinearStripminedThreadMap，但是把坐标系**旋转了**。

#### 为什么要转置？

在您之前的 GEMM 代码中，B 矩阵是 RowMajor 的，但在 Global Memory 读取时，我们希望利用内存合并（Coalescing）。

在 Global Memory 中，B 矩阵沿着 Row（C 维）连续。

但是，为了计算方便，我们在 Shared Memory 中可能希望把 B 矩阵“竖起来”或者以特定的方式存取。

这个类做的就是：**逻辑上的欺骗**。

#### 代码逻辑解释

```cpp
static TensorCoord initial_offset(int thread_id) {
    // 1. 先调用底层 Map 计算出原始坐标 (Original_C, Original_S)
    TensorCoord coord = ThreadMap::initial_offset(thread_id);

    // 2. 强行交换 C 和 S
    return TensorCoord(
        coord.strided(),    // 新的 C = 旧的 S
        coord.contiguous()  // 新的 S = 旧的 C
    );
}
    
```

原始行为：线程 ID 0 映射到 $(0, 0)$，ID 1 映射到 $(1, 0)$（假设沿着 C 轴走）。

转置后行为：线程 ID 0 映射到 $(0, 0)$，ID 1 映射到 $(0, 1)$（变成了沿着 S 轴走）。

#### 这里的 Constraint (静态断言)

注意代码中的断言：    

```cpp
static_assert(kElementsPerAccess == 1, "Simt transpose requires elements per access to be 1");
static_assert(Iterations::kStrided == 1, "...");
    
```

  这意味着这种转置非常“昂贵”或者说“特定”。它要求每个线程只处理**标量（Scalar）**，不能向量化。

这是因为：如果在内存连续方向上做转置，原本连续的 float4 就会变成 4 个分散的 float，这会破坏向量化加载的条件。所以这个类专门用于 **SIMT**（单指令多线程，即标量指令）场景，而不是 Tensor Core 场景。

### 综合应用图解

回到您之前的代码：

SmemIteratorB 使用了 TransposePitchLinearThreadMapSimt。

1. 物理内存（Global Memory B）：Row-Major。数据在行方向（Row）连续。
1. ThreadMapB (原始)：基于 PitchLinear，它会安排线程沿着 Row 方向（C 维）连续读取。这很好，满足 Global Memory 合并访问的要求。
1. TransposeThreadMap (转置后)：

虽然线程是从 Global Memory 顺着 Row 读进来的（物理上连续）。

但是当线程通过这个 Map 计算“我要把数据写到 Shared Memory 哪个位置”时，坐标被交换了。

线程读取的 Row 0, Col 1 的数据，被报告为 C=1, S=0 (假设)，经过转置可能变成了 C=0, S=1。

**最终效果**：

线程顺着 Global Memory 的“毛”摸（读取效率高），然后把数据“旋转 90 度”塞进 Shared Memory。配合 SmemLayoutB，这通常是为了让 Shared Memory 中的数据排列满足后续计算核心（Core）的读取模式，或者为了避免 Shared Memory 的 Bank Conflict。

**总结**：

PitchLinear：统一了行/列主序的数学表达（只有连续维和跨步维）。

StripminedThreadMap：把二维矩阵平铺给一堆线性排列的线程，决定谁搬哪里。

Transpose...Map：在搬运过程中，把坐标 $(x, y)$ 变成 $(y, x)$，实现数据的隐式转置。

上面一堆是 Gemini 3 pro 的生成结果。

```cpp
  using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kM, Shape::kK>, // 128, 8
    kThreads, // 128
    kElementsPerAccess // 1
  >;
```

PitchLinearStripminedThreadMap 定义在 cutlass-3.8/include/cutlass/transform/pitch_linear_thread_map.h 中。

在 PitchLinearStripminedThreadMap 中，首先会判断 Threads >= Detail::ShapeVec::kContiguous，其中 using ShapeVec = layout::PitchLinearShape<Shape::kContiguous / kElementsPerAccess,Shape::kStrided>，对于 A 来说 Detail::ShapeVec::kContiguous = 128。

using ShapeVec = layout::PitchLinearShape<128,8>

如果 Threads >= Detail::ShapeVec::kContiguous，using Iterations = layout::PitchLinearShape<1，8>

using Delta = layout::PitchLinearShape<1,1>

using StorageShape = layout::PitchLinearShape<128,8>

```cpp
  /// Maps thread ID to a coordinate offset within the tensor's logical coordinate space
  /// (in units of Elements)
  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id) {
    return TensorCoord(
      (thread_id % Detail::ShapeVec::kContiguous) * kElementsPerAccess, 
      thread_id / Detail::ShapeVec::kContiguous);
  }
```

简而言之，IteratorThreadMapA 会把 128 个线程映射到 block 中一行的 128 个元素上。

SmemIteratorA 

```cpp
  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kM, Shape::kK>,  // 128,8
    ElementA, //float
    SmemLayoutA, // column major
    1, // advance rank
    IteratorThreadMapA
  >;
```

对于 IteratorThreadMapB。

using ShapeVec = layout::PitchLinearShape<8,128>

using Iterations = layout::PitchLinearShape<1，8>

using Delta = layout::PitchLinearShape<1,16>

using StorageShape = layout::PitchLinearShape<8,128>

```cpp
  /// ThreadMap of iterator B
  using IteratorThreadMapB =  transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kN>,
    kThreads,
    kElementsPerAccess
  >;
```

IteratorThreadMapB 会把 128 个线程以 16 行 8 列为一组映射到 128 个元素上。

```cpp
  // Define the warp-level op
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>(); // 8
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM; // 4
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM; // 8
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN; // 8
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1; // 2
  static const int numElementsA = 128 / sizeof_bits<ElementA>::value; // 4
  static const int numElementsB = 128 / sizeof_bits<ElementB>::value; // 4
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM); // 4
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN); // 4

  static int const kPaddingN = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementB>::value); // 4
  
  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;
  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape 8×4
      cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout 
      LaneMmaShape
  >;
```

这里有个 LaneMmaShape，还有下面的 MmaSimtPolicy 里的 RowMajorInterleaved，具体不知道干啥的，下面是 Gemini 的解释。

这是一个非常硬核的问题，涉及到 CUTLASS 针对 **SIMT 核心**（即使用 CUDA Cores 而非 Tensor Cores 进行矩阵乘法）的底层线程映射策略。

这段代码主要是在定义 **Warp 内部的线程布局（Thread Map）**。简单来说，就是把 Warp 里的 32 个线程排列成一个 $8 \times 4$ 的二维网格，但不是按自然顺序排，而是按“交错（Interleaved）”的方式排，为了优化寄存器读写。

下面我逐一解释 LaneM/LaneN 和 RowMajorInterleaved。

#### 1. LaneM 和 LaneN 是什么？

这两个变量定义了 **SIMT 指令级并行的粒度**（Instruction/Vectorization Granularity）。    

```cpp
static const int numElementsA = 128 / sizeof_bits::value; // float 是 4
static const int numElementsB = 128 / sizeof_bits::value; // float 是 4

// 如果 ThreadTile 足够大，取 4；否则取 ThreadTile 大小
static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM); 
static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN); 
    
```

物理含义：GPU 硬件非常喜欢 128-bit 的访问和操作（即 float4）。

LaneM = 4 意味着在 M 维度上，我们逻辑上希望一次处理 4 个元素。

LaneN = 4 意味着在 N 维度上，我们也希望按 4 个元素对齐。

作用：告诉编译器和 CUTLASS 模板，接下来的数学运算（MMA）虽然是标量指令（SIMT），但请尽量按照 4 个一组（Vector-like）的方式去组织循环和寄存器布局。这通常是为了匹配 LDS.128（Shared Memory Load 128-bit）加载进来的数据结构。

**简单总结**：LaneM/N 就是“向量化处理的长度”，对于 float 通常是 4。

#### 2. RowMajorInterleaved<LaneLayout> 是什么？

这是最关键的部分。它定义了 **Warp 中的 32 个线程是如何映射到 $8 \times 4$ 的逻辑网格上的**。

#### A. 为什么要这个东西？

在一个普通的 Warp 中，线程 ID 是 $0 \dots 31$。

我们需要把它们映射到一个 WarpNumThreadsM=8 行, WarpNumThreadsN=4 列的网格中。

**普通的 RowMajor（行主序）映射是这样的：**

（假设 4 列，连续的线程填充一行）    

```text
Row 0:  T0,  T1,  T2,  T3
Row 1:  T4,  T5,  T6,  T7
Row 2:  T8,  T9,  T10, T11
...
Row 7:  T28, T29, T30, T31
    
```

  **但是，****RowMajorInterleaved<2>**** (LaneLayout=2) 是这样的：**

“Interleaved” 意思是**行是交错的**。Interleaved<2> 表示每隔 2 行，线程 ID 才是连续增加的。

它看起来像这样（重新排列了线程的位置）：    

```text
Row 0:  T0,  T1,  T2,  T3   <-- 这一组是 Warp 的前 4 个线程
Row 1:  T8,  T9,  T10, T11  <-- 注意！跳过了 T4-T7
Row 2:  T4,  T5,  T6,  T7   <-- T4-T7 跑到了这里
Row 3:  T12, T13, T14, T15
...
    
```

  *(注：具体的交错模式取决于 CUTLASS 内部实现，可能是按行交错，也可能是按列内部交错，但核心思想是****打乱相邻线程的 2D 空间位置****)*

#### B. 为什么 SIMT Kernel 要把线程排得这么乱？

这主要是为了 **寄存器重用（Register Reuse）** 和 **减少 Bank Conflict**。

在做矩阵乘法 $C += A \times B$ 时，SIMT 核心通常采用 **外积（Outer Product）** 的方式。

每个线程持有一部分 A 和一部分 B。

为了减少从 Shared Memory 读取数据的次数，我们希望相邻的线程共享相同的数据。

例如：

如果 T0 和 T1 在同一行，它们可能共享相同的 A 数据。

如果 T0 和 T8 在同一列，它们可能共享相同的 B 数据。

通过 RowMajorInterleaved，CUTLASS 试图构造一种特殊的布局，使得：

1. 数据局部性：配合 LaneM=4，使得一个 float4 读进来后，能够刚好分发给物理上相邻（或逻辑上需要协作）的一组线程。
1. 指令流水线：让编译器生成的指令序列在访问寄存器文件（Register File）时，能利用 GPU 硬件的 Bank 结构，避免所有线程同时访问同一个 Bank 导致排队延迟。

#### 3. 综合解读

结合您提供的代码：    

```cpp
using Policy = cutlass::gemm::warp::MmaSimtPolicy<
    cutlass::MatrixShape, // 8x4 线程网格
    cutlass::layout::RowMajorInterleaved,       // 布局策略
    LaneMmaShape
>;
    
```

  这段定义的含义是：

“我们要创建一个 SIMT 矩阵乘法策略。

1. Warp 里的 32 个线程被逻辑视为 $8 \times 4$ 的网格。
1. 但是！请不要按自然顺序排列线程。请使用 RowMajorInterleaved<2> 这种交错布局来分配线程 ID 到网格坐标。
1. 这样做的目的是为了配合 LaneM=4 的向量化数据加载，使得每个线程持有的数据片段在寄存器中是对齐的，从而榨干 CUDA Core 的性能。”

**一句话总结**：这是 CUTLASS 为了在没有 Tensor Core 的显卡（或者 float/double 任务）上把 CUDA Core 的性能优化到极致，而设计的一种**复杂的、非线性的线程排布方式**。

最后会定义 MmaSimt 和 MmaPolicy。其中会对 B 做 padding 防止 bank conflicts。

```cpp
  using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
      WarpShape,    /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
      ElementA,     /// Data type of A elements
      SmemLayoutA,  /// Layout of A matrix (concept: MatrixLayout)
      ElementB,     /// Data type of B elements
      SmemLayoutB,  /// Layout of B matrix (concept: MatrixLayout)
      ElementC,     /// Element type of C matrix
      LayoutC,      /// Layout of C matrix (concept: MatrixLayout)
      Policy        /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
  >;

  /// Policy used to define MmaPipelined 
  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<0, 0>,
    MatrixShape<0, kPaddingN>, // skew for B matrix to avoid SMEM bank conflicts
    WarpCount::kK
  >;
```

### MmaSimt

这里会定义 thread::Mma

```cpp
  /// Thread-level matrix multiply accumulate operator
  using ThreadMma = thread::Mma<
    GemmShape<
      Shape::kM / Policy::WarpShape::kRow, // 64 / 8 = 8
      Shape::kN / Policy::WarpShape::kColumn, // 32 / 4 = 8
      Policy::LaneMmaShape::kK>, // 1
    ElementA,
    ThreadLayoutA,
    ElementB,
    ThreadLayoutB,
    ElementC,
    LayoutC,
    arch::OpMultiplyAdd,
    dp4a_type
  >;
```

然后会定义 IteratorA IteratorB 和 IteratorC 用来处理寄存器。

```cpp
  /// Iterates over the A operand in memory
  using IteratorA = MmaSimtTileIterator<
    MatrixShape<Shape::kM, Policy::LaneMmaShape::kK>,
    Operand::kA,
    ElementA,
    LayoutA,
    Policy,
    PartitionsK,
    Shape::kK
  >;

  /// Storage for A tile
  using FragmentA = typename IteratorA::Fragment;

  /// Storage for transformed A tile
  using TransformedFragmentA = FragmentA;

  /// Iterates over the B operand in memory
  using IteratorB = MmaSimtTileIterator<
    MatrixShape<Policy::LaneMmaShape::kK, Shape::kN>,
    Operand::kB,
    ElementB,
    LayoutB,
    Policy,
    PartitionsK,
    Shape::kK
  >;

  /// Storage for B tile
  using FragmentB = typename IteratorB::Fragment;

  /// Storage for transformed A tile
  using TransformedFragmentB = FragmentB;

  /// Iterates over the C operand in memory
  using IteratorC = MmaSimtTileIterator<
    MatrixShape<Shape::kM, Shape::kN>,
    Operand::kC,
    ElementC,
    LayoutC,
    Policy
  >;

  /// Storage for C tile
  using FragmentC = typename ThreadMma::FragmentC;
```

在 MmaSimtTileIterator 中，以 A 为例，会创建下面这些

```cpp
  /// Thread-level shape of a fragment
  using ThreadShape = MatrixShape<
    Shape::kRow / Policy::WarpShape::kRow, // 64 / 8 = 8
    Shape::kColumn // LaneMmaShape::kK = 1
  >;

  static_assert(!(ThreadShape::kRow % Policy::LaneMmaShape::kM), 
    "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

  /// Number of individual loads
  using Iterations = MatrixShape<
    ThreadShape::kRow / Policy::LaneMmaShape::kM,  // 8 / 4 = 2
    ThreadShape::kColumn // 1
  >;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, ThreadShape::kCount>; // 8

private:

  /// Internal reference
  cutlass::TensorRef<Array<Element, Policy::LaneMmaShape::kM>, layout::ColumnMajor> ref_;  // 4
  

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator(
    TensorRef ref, 
    int lane_id
  ) {

    // compute offset based on thread ID and lane layout
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout(); // 初始化时由于设置了 RowMajorInterleaved，所以这里线程的 id 不是按顺序的

    MatrixCoord lane_offset = lane_layout.inverse(lane_id) * 
      MatrixCoord(Policy::LaneMmaShape::kM, 0);

    ref.add_coord_offset(lane_offset);

    ref_.reset(
      reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM> *>(ref.data()),
      ref.stride(0) / Policy::LaneMmaShape::kM);
  }
  
```

此外还有两个函数 load 和 store。从下面的代码中可以发现从 smem 加载和保存时，一个线程不是连续加载 8 个元素，而是 4 个 4 个加载的。

```cpp
  /// Loads a fragment from memory at the location pointed to by the iterator. (vector loads)
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {
    Array<Element, Policy::LaneMmaShape::kM> *dst_ptr = 
      reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kColumn; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < Iterations::kRow; ++m) {  // 0,1

        // This logic has been replaced with calls to inline PTX to guarantee vectorization.
        #if 0
        dst_ptr[m + k * Iterations::kRow] = 
          *(ref_.data() + ref_.offset({m * Policy::WarpShape::kRow, k}) + pointer_offset / Policy::LaneMmaShape::kM);
        #endif

        auto ptr = ref_.data() + ref_.offset({m * Policy::WarpShape::kRow, k}) + pointer_offset / Policy::LaneMmaShape::kM; // 0*64 1*64
        arch::shared_load(dst_ptr[m + k * Iterations::kRow], ptr); // 一个线程不是加载连续的 8 个元素，而是 4 个 4 个加载的。
      }
    }
  }
  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

 /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) const {
    
    Array<Element, Policy::LaneMmaShape::kM> const *src_ptr = 
      reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kN; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < Iterations::kM; ++m) {
        *(ref_.data() + ref_.offset(m * Policy::WarpShape::kM, k) + pointer_offset / Policy::LaneMmaShape::kM) = 
          src_ptr[m + k * Iterations::kM];
      }
    }
  }

  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }
```

最后进入到 cutlass-3.8/include/cutlass/gemm/thread/mma_sm50.h 计算每个线程的 mma。代码如下：

```cpp
/// Gemplate that handles all packed matrix layouts
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename ElementA_,
  /// Layout of A matrix (concept: layout::MapFunc)
  typename LayoutA_,
  /// Data type of B elements
  typename ElementB_,
  /// Layout of B matrix (concept: layout::MapFunc)
  typename LayoutB_,
  /// Element type of C matrix
  typename ElementC_,
  /// Layout of C matrix (concept: layout::MapFunc)
  typename LayoutC_,
  /// Operator used to compute GEMM
  typename Operator_
>
struct MmaGeneric {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// Data type of operand A
  using ElementA = ElementA_;

  /// Layout of A matrix (concept: layout::MapFunc)
  using LayoutA = LayoutA_;

  /// Data type of operand B
  using ElementB = ElementB_;

  /// Layout of B matrix (concept: layout::MapFunc)
  using LayoutB = LayoutB_;

  /// Element type of operand C
  using ElementC = ElementC_;

  /// Layout of C matrix (concept: layout::MapFunc)
  using LayoutC = LayoutC_;

  /// Underlying mathematical operator
  using Operator = Operator_;

  /// A operand storage
  using FragmentA = Array<ElementA, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<ElementB, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<ElementC, Shape::kMN>;

  /// Instruction
  using MmaOp = arch::Mma<
    gemm::GemmShape<1,1,1>,
    1,
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    Operator>;

  static bool const kMultipleOf2 = ((Shape::kM % 2 == 0) && (Shape::kN % 2 == 0));

  static bool const kAllFp32 = platform::is_same<ElementA, float>::value &&
      platform::is_same<ElementB, float>::value &&
      platform::is_same<ElementC, float>::value;
  //
  // Methods
  //

  /// Computes a matrix product D = A * B + C
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC & D,
    FragmentA const & A,
    FragmentB const & B,
    FragmentC const & C) {

    TensorRef<ElementA const, LayoutA> a_ref(
      reinterpret_cast<ElementA const *>(&A), LayoutA::packed({Shape::kM, Shape::kK}));

    TensorRef<ElementB const, LayoutB> b_ref(
      reinterpret_cast<ElementB const *>(&B), LayoutB::packed({Shape::kK, Shape::kN}));

    TensorRef<ElementC, LayoutC> d_ref(
      reinterpret_cast<ElementC *>(&D), LayoutC::packed(make_Coord(Shape::kM, Shape::kN)));

    MmaOp mma_op;

    // Copy accumulators
    D = C;

    // Compute matrix product
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Shape::kK; ++k) {
      #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 860)
      if constexpr (kMultipleOf2 && kAllFp32) {
        //2x2 zigzag - m and n loops to increment by 2. Inner loop to process 4 multiply-adds in a 2x2 tile.
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Shape::kN; n+=2) {
  
          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < Shape::kM; m+=2) {
  
            int m_serpentine = (n % 4) ? (Shape::kM - 2 - m) : m;

            //top-left element in 2x2 tile
            {
              MatrixCoord mn(m_serpentine, n);
              MatrixCoord mk(m_serpentine, k);
              MatrixCoord kn(k, n);
              Array<ElementC, 1> d;
              Array<ElementA, 1> a;
              Array<ElementB, 1> b;
              d[0] = d_ref.at(mn);
              a[0] = a_ref.at(mk);
              b[0] = b_ref.at(kn);
              mma_op(d, a, b, d);
              d_ref.at(mn) = d[0];
            }
  
            //bottom-left element in 2x2 tile
            {
              MatrixCoord mn(m_serpentine+1, n);
              MatrixCoord mk(m_serpentine+1, k);
              MatrixCoord kn(k, n);
              Array<ElementC, 1> d;
              Array<ElementA, 1> a;
              Array<ElementB, 1> b;
              d[0] = d_ref.at(mn);
              a[0] = a_ref.at(mk);
              b[0] = b_ref.at(kn);
              mma_op(d, a, b, d);
              d_ref.at(mn) = d[0];
            }
  
            //bottom-right element in 2x2 tile
            {
              MatrixCoord mn(m_serpentine+1, n+1);
              MatrixCoord mk(m_serpentine+1, k);
              MatrixCoord kn(k, n+1);
              Array<ElementC, 1> d;
              Array<ElementA, 1> a;
              Array<ElementB, 1> b;
              d[0] = d_ref.at(mn);
              a[0] = a_ref.at(mk);
              b[0] = b_ref.at(kn);
              mma_op(d, a, b, d);
              d_ref.at(mn) = d[0];
            }
  
            //top-right element in 2x2 tile
            {
              MatrixCoord mn(m_serpentine, n+1);
              MatrixCoord mk(m_serpentine, k);
              MatrixCoord kn(k, n+1);
              Array<ElementC, 1> d;
              Array<ElementA, 1> a;
              Array<ElementB, 1> b;
              d[0] = d_ref.at(mn);
              a[0] = a_ref.at(mk);
              b[0] = b_ref.at(kn);
              mma_op(d, a, b, d);
              d_ref.at(mn) = d[0];
            }
          }
        }
      } else 
      #endif
      {
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Shape::kN; ++n) {
  
          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < Shape::kM; ++m) {
  
            int m_serpentine = (n % 2) ? (Shape::kM - 1 - m) : m;
  
            MatrixCoord mn(m_serpentine, n);
            MatrixCoord mk(m_serpentine, k);
            MatrixCoord kn(k, n);
  
            Array<ElementC, 1> d;
            Array<ElementA, 1> a;
            Array<ElementB, 1> b;
  
            d[0] = d_ref.at(mn);
            a[0] = a_ref.at(mk);
            b[0] = b_ref.at(kn);
  
            mma_op(d, a, b, d);
  
            d_ref.at(mn) = d[0];
          }
        }
      }
    }
  }
};
```

下面是 Gemini 的解释：

这段代码是 CUTLASS 中用于 **SIMT（单指令多线程，即使用 CUDA Cores 而非 Tensor Cores）** 架构下的 **线程级（Thread-level）矩阵乘法** 实现。

它定义了一个通用的 MmaGeneric 结构体，用于在单个线程内部，计算它所负责的那一部分数据的乘加运算（FFMA: $D = A \times B + C$）。

这段代码之所以写得如此复杂（包含 m_serpentine 蛇形遍历和 2x2 展开），主要是为了 **指令级并行（ILP）优化** 和 **最大化寄存器操作数重用（Register Operand Reuse）**。

下面我为您详细拆解这段代码的逻辑和背后的硬件原理。

### 1. 代码上下文：数据在哪里？

```cpp
// A, B, C, D 都是 Fragment，本质上是寄存器数组
TensorRef a_ref(...);
TensorRef b_ref(...);
TensorRef d_ref(...);
    
```

  在进入这段代码前，数据已经被加载到了线程的 **寄存器（Registers）** 中。

CUTLASS 使用 TensorRef 将这些扁平的寄存器数组（Array）抽象为 2D 矩阵，以便使用坐标 $(m, k)$ 或 $(k, n)$ 来访问。

### 2. 核心逻辑一：蛇形遍历 (Serpentine / Zigzag)

代码中出现了一个非常奇怪的索引计算：    

```cpp
int m_serpentine = (n % 2) ? (Shape::kM - 1 - m) : m;
    
```

  或者在 Ampere 优化块中：    

```cpp
int m_serpentine = (n % 4) ? (Shape::kM - 2 - m) : m;
    
```

  **这是什么意思？**

当 n 是偶数列时，m 从 0 增加到 M（从上往下）。

当 n 是奇数列时，m 从 M 减小到 0（从下往上）。

**为什么要这么做？**

这是为了减少 **指令间的依赖延迟** 和 **寄存器访问冲突**。

在 GPU 的流水线中，如果连续两条指令使用完全不同的寄存器索引，或者使用了容易导致 **Bank Conflict（寄存器堆冲突）** 的模式，流水线可能会停顿。

蛇形遍历保证了当列号 n 切换时，行号 m 保持在附近（例如从 $(M-1, 0)$ 跳到 $(M-1, 1)$），这使得数据访问在时间局部性上更平滑。

### 3. 核心逻辑二：Ampere (SM86+) 的 2x2 展开优化

这是这段代码最精彩的部分。如果检测到是 Ampere 架构（RTX 30 系及以上）且是 FP32 运算，代码会执行一个特殊的 2x2 循环展开：    

```cpp
// 2x2 zigzag ...
for (int n = 0; n < Shape::kN; n+=2) {
  for (int m = 0; m < Shape::kM; m+=2) {
    // ... 依次计算 2x2 块内的四个点 ...
  }
}
    
```

  它手动展开了 4 次 FFMA（乘加）操作，顺序是：**左上 -> 左下 -> 右下 -> 右上**。

#### 为什么是这个顺序？（为了“操作数重用”）

让我们看看这四个点需要的操作数（假设当前计算的是 $C += A \times B$）：

1. Top-Left (0, 0):

需要 $A[0]$ 和 $B[0]$。

计算 $D[0][0] += A[0] * B[0]$

1. Bottom-Left (1, 0):

需要 $A[1]$ 和 $B[0]$。

优化点：$B[0]$ 刚刚在上一条指令用过！

计算 $D[1][0] += A[1] * B[0]$

1. Bottom-Right (1, 1):

需要 $A[1]$ 和 $B[1]$。

优化点：$A[1]$ 刚刚在上一条指令用过！

计算 $D[1][1] += A[1] * B[1]$

1. Top-Right (0, 1):

需要 $A[0]$ 和 $B[1]$。

优化点：$B[1]$ 刚刚在上一条指令用过！

循环衔接：下一轮循环又要算 Top-Left，需要 $A[0]$，而这里刚好用的是 $A[0]$！

#### 硬件原理：Operand Collector

现代 NVIDIA GPU（如 Ampere）的核心中有一个组件叫 **Operand Collector**。

从寄存器堆（Register File）读取数据是耗能且有带宽限制的。

如果连续两条指令复用了同一个源操作数（例如上面的 $B[0]$），Operand Collector 可以直接缓存并重用该值，而不需要再次从庞大的寄存器堆中读取。

这种 U 型（Horseshoe） 或 环形 的计算顺序，最大限度地利用了这种硬件特性，减少了寄存器读取压力，从而提升了 FFMA 指令的吞吐量。

### 4. 总结：为什么这么实现？

这段代码并不是简单的数学公式翻译，而是**针对 GPU 微架构的汇编级优化**的 C++ 实现。

1. 通用性 (Generic): 通过 Template 和 TensorRef，同一套代码可以支持任意形状和 Layout 的 Thread-level GEMM。
1. 蛇形遍历 (Serpentine): 使得寄存器访问模式更加平滑，减少 Bank Conflict。
2. 2x2 展开与重用: 在 Ampere 架构上，通过特定的计算顺序（左上-左下-右下-右上），构建了一条操作数重用链。这减少了寄存器文件的读次数，直接提升了 SIMT 核心做 FP32 乘法的效率。

这就是为什么 CUTLASS 的性能通常能媲美甚至超越手写 CUDA Assembly 的原因——它在 C++ 模板层面就把硬件的脾气摸透了。

## MmaPipelined

每一步的计算过程看完了，下面看 pipeline 是怎么计算的。

cutlass-3.8/include/cutlass/gemm/threadblock/mma_pipelined.h

```cpp
class MmaPipelined : public MmaBase<Shape_, Policy_, 2> {
public:

  ///< Base class
  using Base = MmaBase<Shape_, Policy_, 2>;

  using Shape = Shape_;             ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using IteratorA = IteratorA_;     ///< Iterates over tiles of A operand in global memory
  using IteratorB = IteratorB_;     ///< Iterates over tiles of B operand in global memory
  using ElementC = ElementC_;       ///< Data type of accumulator matrix
  using LayoutC = LayoutC_;         ///< Layout of accumulator matrix
  using Policy = Policy_;           ///< Policy describing tuning details

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  using TransformA = TransformA_;
  using TransformB = TransformB_;
```

前面会创建一些访问 global memory 和 shared memory 的迭代器。

这个类的 init 代码如下：会在上一层把 thread_idx，warp_idx，lane_idx 和其他一些参数传进来。

```cpp
  /// Construct from tensor references
  CUTLASS_DEVICE
  MmaPipelined(
    typename Base::SharedStorage &shared_storage,       ///< Shared storage needed for internal use by threadblock-scoped GEMM
    int thread_idx,                                     ///< ID within the threadblock
    int warp_idx,                                       ///< ID of warp
    int lane_idx,                                       ///< ID of each thread within a warp
    TransformA transform_A = TransformA(),              ///< transformation applied to A fragment
    TransformB transform_B = TransformB()               ///< transformation applied to B fragment
  ):
    Base(shared_storage, thread_idx, warp_idx, lane_idx),
    smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
    smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
    transform_A_(transform_A),
    transform_B_(transform_B),
    smem_write_stage_idx(0)
  {

    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM; // 计算 warp 的 idx，这里在 M 方向上是连续的
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM; // 

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset({Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }
```

然后会首先 init 一个 mma_base 的类。在 cutlass-3.8/include/cutlass/gemm/threadblock/mma_base.h 文件里。

mma_base 里主要是计算每个 warp 的处理范围。

```cpp
  using WarpGemm = typename Policy::Operator::Shape;

  /// Shape describing the number of warps filling the CTA
  using WarpCount = GemmShape<Shape::kM / WarpGemm::kM,  // 128 / 64 = 2
                              Shape::kN / WarpGemm::kN,  // 64 / 32 = 2
                              Shape::kK / WarpGemm::kK>; // 8 / 8 = 1

  /// Number of warp-level GEMM oeprations
  static int const kWarpGemmIterations =
      (WarpGemm::kK / Operator::Policy::MmaShape::kK);  // 1

  /// Number of stages
  static int const kStages = Stages; // 2
```

此外，这里面会创建 shared memory 的 buffer，这里会根据 padding 的大小创建。还有 warp_tile_iterator_A_和 warp_tile_iterator_B_，这两个是在计算时每个 warp 和 shared memory 的对应关系。这里会有一些 LaneShape 和 RowInterleave 的操作。

```cpp
  /// Shared storage object needed by threadblock-scoped GEMM
  class SharedStorage {
   public:
    //
    // Type definitions
    //

    /// Shape of the A matrix operand in shared memory
    using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow,
                               Shape::kK * kStages +
                                   Policy::SmemPaddingA::kColumn>;

    /// Shape of the B matrix operand in shared memory
    using ShapeB =
        MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow,
                    Shape::kN + Policy::SmemPaddingB::kColumn>;
                    
  /// Iterator to load a warp-scoped tile of A operand from shared memory
  typename Operator::IteratorA warp_tile_iterator_A_;

  /// Iterator to load a warp-scoped tile of B operand from shared memory
  typename Operator::IteratorB warp_tile_iterator_B_;
```

回到 pipeline。调用顺序如下：

```cpp
  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
    int gemm_k_iterations,                            ///< number of iterations of the mainloop
    FragmentC &accum,                                 ///< destination accumulator tile
    IteratorA iterator_A,                             ///< iterator over A operand in global memory
    IteratorB iterator_B,                             ///< iterator over B operand in global memory
    FragmentC const &src_accum)                       ///< source accumulator tile
  {
    // Prologue
    prologue(iterator_A, iterator_B, gemm_k_iterations);

    // Wait until we have at least one completed global fetch stage
    gmem_wait();

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    // Perform the MAC-iterations
    gemm_iters(gemm_k_iterations, accum, iterator_A, iterator_B);
  }
```

prologue

```cpp
  /// GEMM prologue.  Bootstrap the global->shared memory pipeline by fetching
  /// the global fragments needed by the first kStages-1 threadblock mainloop iterations
  CUTLASS_DEVICE
  void prologue(
    IteratorA &iterator_A,      ///< [in|out] iterator over A operand in global memory
    IteratorB &iterator_B,      ///< [in|out] iterator over B operand in global memory
    int &gemm_k_iterations)     ///< [in|out] number of threadblock mainloop iterations remaining
  {
    // The last kblock is loaded in the prolog

    // Load A fragment from global A
    FragmentA tb_frag_A;
    tb_frag_A.clear();
    iterator_A.load(tb_frag_A);
    ++iterator_A;

    // Load B fragment from global B
    FragmentB tb_frag_B;
    tb_frag_B.clear();
    iterator_B.load(tb_frag_B);
    ++iterator_B;

    // Store A and B fragments to shared
    this->smem_iterator_A_.store(transform_A_(tb_frag_A));
    this->smem_iterator_B_.store(transform_B_(tb_frag_B));

    // Advance write stage
    advance_smem_write_stage();
  }

  /// Wait until we have at least one completed global fetch stage
  CUTLASS_DEVICE
  void gmem_wait()
  {
    __syncthreads();
  }
```

gemm_iters

```cpp
  /// Perform the specified number of threadblock mainloop iterations of matrix
  /// multiply-accumulate.  Assumes prologue has been initiated.
  CUTLASS_DEVICE
  void gemm_iters(
    int gemm_k_iterations,        ///< number of threadblock mainloop iterations
    FragmentC &accum,             ///< [in|out] accumulator tile
    IteratorA &iterator_A,        ///< [in|out] iterator over A operand in global memory
    IteratorB &iterator_B)        ///< [in|out] iterator over B operand in global memory
  {
    using WarpFragmentA = typename Operator::FragmentA;
    using WarpFragmentB = typename Operator::FragmentB;

    // Pair of fragments used to overlap shared memory loads and math instructions
    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B[2];

    // Load A fragment from shared A
    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
    ++this->warp_tile_iterator_A_;

    // Load B fragment from shared B
    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);
    ++this->warp_tile_iterator_B_;

    // Pair of fragments used to overlap global memory loads and math instructions;
    FragmentA tb_frag_A;
    FragmentB tb_frag_B;

    // Avoid reading out of bounds
    iterator_A.clear_mask(gemm_k_iterations <= 1);
    iterator_B.clear_mask(gemm_k_iterations <= 1);

    //
    // Mainloop
    //

    // Note: The main loop does not support Base::kWarpGemmIterations == 2.
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      //
      // Loop over GEMM K dimension
      //

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {

        // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group
        // as the case may be.

        if (warp_mma_k == Base::kWarpGemmIterations - 1) {

          // Write fragments to shared memory
          this->smem_iterator_A_.store(transform_A_(tb_frag_A));

          this->smem_iterator_B_.store(transform_B_(tb_frag_B));

          // Wait until we have at least one completed global fetch stage
          gmem_wait();

          // Advance smem read and write stages
          advance_smem_stages();
        }

        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);

        this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        if (warp_mma_k == 0) {

          // Load fragment from global A
          tb_frag_A.clear();
          iterator_A.load(tb_frag_A);
          ++iterator_A;

          // Load fragment from global B
          tb_frag_B.clear();
          iterator_B.load(tb_frag_B);
          ++iterator_B;

          // Avoid reading out of bounds if this was the last loop iteration
          iterator_A.clear_mask(gemm_k_iterations <= 2);
          iterator_B.clear_mask(gemm_k_iterations <= 2);
        }

        warp_mma(
          accum,
          warp_frag_A[warp_mma_k % 2],
          warp_frag_B[warp_mma_k % 2],
          accum);
      }
    }

  }
```
