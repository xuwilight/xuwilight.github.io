---
title: Cutlass 学习笔记（八）SM90 GEMM TMA Warp Specialized
date: 2025-08-22 12:00:00
tags: [CUTLASS, GEMM, TMA, WarpSpecialized]
categories: [Cutlass 学习笔记,Cutlass]
description: 使用 TMA 加载数据并通过 Warp Specialized 方式实现 GEMM，producer warp 负责加载数据，consumer warp 负责计算。
---

## 代码实现

sm90_gemm_tma_warpspecialized 使用 tma 加载数据，与上一个不同的是 kernel 的实现使用了 warp specialized 的写法。也就是一部分 warp 只加载数据，称为 producer，一部分 warp 只计算数据，称为 comuser。

基本代码如下，C 矩阵使用了 half，不然性能会很慢。其余跟 tma 差不多，就是 schedule 和 dispatchpolicy 不一样。

```python
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

#include "../utils.h"

using namespace cute;

/**
 * nvcc sm90_tma_ws_gemm.cu -arch=sm_90a -I ../../../include/ -I ../../../tools/util/include/ -lcuda -lcublas -o sm90_tma_ws_gemm --expt-relaxed-constexpr && ./sm90_tma_ws_gemm
 */

template <class TA, class TB, class TC>
void sm90_tma_ws_gemm(int M, int N, int K, TC alpha, TA const *A, int lda, TB const *B, int ldb, TC beta, TC *C, int ldc)
{
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using ArchTag = cutlass::arch::Sm90;
    using OpClass = cutlass::arch::OpClassTensorOp;
    using TileShape = Shape<_128, _128, _64>;
    using ClusterShape = Shape<_1, _1, _1>;
    using MmaSchedule = cutlass::gemm::KernelTmaWarpSpecialized;  // deciding kernel is KernelTmaWarpSpecialized, KernelTmaWarpSpecializedPingpong or KernelTmaWarpSpecializedCooperative

    constexpr int stage = 3;
    using MmaDispatchPolicy = cutlass::gemm::MainloopSm90TmaGmmaWarpSpecialized<stage, ClusterShape, MmaSchedule>;  // gemm is warpspecialized
    using TMACOPY = SM90_TMA_LOAD;
    using AtomLayoutMNK = Layout<Shape<_1, _1, _1>>;

    // using TiledMMA = decltype(make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{}));
    using TiledMMA = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<TA, TB, TC, TileShape, GMMA::Major::K, GMMA::Major::K>(), AtomLayoutMNK{}));

    // using SmemLayoutAtomA = decltype(detail::ss_smem_selector<GMMA::Major::K, TA, decltype(cute::get<0>(TileShape{})), decltype(cute::get<2>(TileShape{}))>());
    // using SmemLayoutAtomB = decltype(detail::ss_smem_selector<GMMA::Major::K, TA, decltype(cute::get<1>(TileShape{})), decltype(cute::get<2>(TileShape{}))>());

    using SmemLayoutAtomA = GMMA::Layout_K_SW128_Atom<TA>;
    using SmemLayoutAtomB = GMMA::Layout_K_SW128_Atom<TB>;
    using SmemCopyAtomA = void;
    using SmemCopyAtomB = void;
    using Stride_A = cutlass::detail::TagToStrideA_t<LayoutA>;
    using Stride_B = cutlass::detail::TagToStrideB_t<LayoutB>;
    using Stride_C = cutlass::detail::TagToStrideC_t<LayoutC>;

    using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
        MmaDispatchPolicy,
        TileShape, TA, Stride_A, TB, Stride_B,
        TiledMMA,
        TMACOPY, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,
        TMACOPY, SmemLayoutAtomB, SmemCopyAtomB, cute::identity>;


    // build epilogue
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized; // NoSmemWarpSpecialized, TmaWarpSpecialized or TmaWarpSpecializedCooperative

    constexpr int AlignmentC  = 16;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        TC, TC,
        TC, LayoutC, AlignmentC,
        TC, LayoutC, AlignmentC,
        EpilogueSchedule
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    Gemm gemm;

    cutlass::KernelHardwareInfo kernel_hw_info;
    kernel_hw_info.device_id = 0;
    kernel_hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(kernel_hw_info.device_id);
    // cutlass::KernelHardwareInfo kernel_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<Gemm::GemmKernel>(device_id);

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {A, stride_A, B, stride_B},
        {{alpha, beta}, C, stride_C, C, stride_C},
        kernel_hw_info};

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    gemm.can_implement(arguments);
    gemm.initialize(arguments, workspace.get());
    gemm.run();
}

int main()
{
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;

    using TA = half_t;
    using TB = half_t;
    using TC = half_t;

    TC alpha = static_cast<TC>(1.0f);
    TC beta = static_cast<TC>(0.0f);

    std::vector<TA> h_A(M * K);
    std::vector<TB> h_B(N * K);
    std::vector<TC> cutlass_res(M * N);
    std::vector<TC> cublas_res(M * N);

    for (int i = 0; i < M * K; ++i)
    {
        h_A[i] = static_cast<TA>(rand() % 9 * 1.0 / 10);
    }

    for (int i = 0; i < N * K; ++i)
    {
        h_B[i] = static_cast<TB>(rand() % 9 * 1.0 / 10);
    }

    // Allocate device Matrices
    TA *d_A;
    TB *d_B;
    TC *C_cutlass;
    TC *C_cublas;

    cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(TA) * M * K);
    cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(TB) * N * K);
    cudaMalloc(reinterpret_cast<void **>(&C_cutlass), sizeof(TC) * M * N);
    cudaMalloc(reinterpret_cast<void **>(&C_cublas), sizeof(TC) * M * N);

    cudaMemcpy(d_A, h_A.data(), sizeof(TA) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeof(TB) * N * K, cudaMemcpyHostToDevice);

    sm90_tma_ws_gemm(M, N, K, alpha, d_A, M, d_B, K, beta, C_cutlass, N);
    cudaDeviceSynchronize();

    cublasHandle_t handle;
    cublasCreate(&handle);
    // cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, d_A, CUDA_R_16F, K, d_B, CUDA_R_16F, K, &beta, C_cublas, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    // C is column-major
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
                reinterpret_cast<const __half *>(&alpha),
                reinterpret_cast<__half *>(d_A), K,
                reinterpret_cast<__half *>(d_B), K,
                reinterpret_cast<const __half *>(&beta),
                reinterpret_cast<__half *>(C_cublas), N);
    cudaDeviceSynchronize();

    cudaMemcpy(cutlass_res.data(), C_cutlass, sizeof(TC) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(cublas_res.data(), C_cublas, sizeof(TC) * M * N, cudaMemcpyDeviceToHost);

    test_gemm(cublas_res.data(), cutlass_res.data(), M, N, K);

    int benchmark = 1;
    if (benchmark)
    {
        float flops = 2.0 * M * N * K;
        float h100 = 989e12;

        std::function<void()> cublas_func = [&]()
        {
            cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
                        reinterpret_cast<const __half *>(&alpha),
                        reinterpret_cast<__half *>(d_A), K,
                        reinterpret_cast<__half *>(d_B), K,
                        reinterpret_cast<const __half *>(&beta),
                        reinterpret_cast<__half *>(C_cublas), N);
        };

        std::function<void()> custom_func = [&]()
        {
            sm90_tma_ws_gemm(M, N, K, alpha, d_A, M, d_B, K, beta, C_cutlass, M);
        };

        run_benchmark(cublas_func, "cublas", flops, h100);
        run_benchmark(custom_func, "mma", flops, h100);
    }
}
```

### mainloop 部分

相比 tma gemm 多了个 MmaSchedule，可以等于 KernelTmaWarpSpecialized，KernelTmaWarpSpecializedPingpong 和 KernelTmaWarpSpecializedCooperative。用于区分具体是哪个类型的 WS kernel。

```python
using MmaSchedule = cutlass::gemm::KernelTmaWarpSpecialized;  // deciding kernel is KernelTmaWarpSpecialized, KernelTmaWarpSpecializedPingpong or KernelTmaWarpSpecializedCooperative
```

MmaDispatchPolicy 使用 MainloopSm90TmaGmmaWarpSpecialized，可以实例化对应的 tma ws kernel。

```python
using MmaDispatchPolicy = cutlass::gemm::MainloopSm90TmaGmmaWarpSpecialized<stage, ClusterShape, MmaSchedule>;  // gemm is warpspecialized
```

其余的构建 CollectiveMma 的部分和之前相同。

### Epilogue 部分

epilogue 有三种，NoSmemWarpSpecialized, TmaWarpSpecialized 和 TmaWarpSpecializedCooperative，这里使用 TmaWarpSpecialized。

```python
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized; // NoSmemWarpSpecialized, TmaWarpSpecialized or TmaWarpSpecializedCooperative
```

因为构建 epilogue 的逻辑有点杂乱，因此这里先试用 CollectiveBuilder 构建。

```python
    // build epilogue
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized; // NoSmemWarpSpecialized, TmaWarpSpecialized or TmaWarpSpecializedCooperative

    constexpr int AlignmentC  = 16;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        TC, TC,
        TC, LayoutC, AlignmentC,
        TC, LayoutC, AlignmentC,
        EpilogueSchedule
        >::CollectiveOp;
```

epilogue 具体的流程如下：

CollectiveBuilder 会根据 EpilogueSchedule 进入 Tma warp-specialized builder。

在这里会先计算 ElementD，EpilogueTile_MN 和 DispatchPolicy 

```python
  using ElementD = cute::conditional_t<cute::is_void_v<ElementD_>,
                     fusion::get_element_aux_t<FusionOperation>, ElementD_>;
  using EpilogueTile_MN =
    decltype(detail::sm90_compute_tile_shape_or_override<ElementD, EpilogueTileType, Schedule, TileShape_MNK>());
  using DispatchPolicy =
    decltype(detail::sm90_get_tma_dispatch_policy<TileShape_MNK,EpilogueTile_MN,ElementC,ElementD,Schedule>());
```

通过调试，EpilogueTile_MN: (_64,_32)，不知道为啥要这样算。

然后对于 DispatchPolicy，是按照 Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmem, DelayTmaStore>{};构建的，其中计算逻辑如下：EpiTiles = 8，FragmentSize = 16，ReuseSmem = true，DelayTmaStore = false，StagesD = 2，StagesC = 4。为啥要这样设置参数呢。

```python
// Returns the parameterized dispatch policy for the TMA epilogue
template<class TileShapeMNK, class EpilogueTileMN, class ElementC, class ElementD, class Schedule>
constexpr auto
sm90_get_tma_dispatch_policy() {
  using namespace cute;

  constexpr int EpiTiles = size(shape_div(take<0,2>(TileShapeMNK{}), EpilogueTileMN{}));
  constexpr int FragmentSize = size(EpilogueTileMN{}) / (detail::sm90_is_cooperative_v<Schedule> ? 256 : 128);
  // 8b residuals load fast and consume little smem, so the perf cost of waiting on stores to finish outweighs the cost of extra allocation
  constexpr bool ReuseSmem = (sizeof_bits_v<ElementC> == sizeof_bits_v<ElementD>) && (sizeof_bits_v<ElementD> > 8);
  // TMA store delay performs worse with residual loads and compilicates tensormap updates for Ptr-Array GEMMs
  constexpr bool DelayTmaStore = is_void_v<ElementC> && !detail::sm90_is_ptr_array_tma_v<Schedule>;
  constexpr int StagesD = cute::min(EpiTiles, 2);
  constexpr int StagesC = ReuseSmem ? cute::max(cute::min(EpiTiles, 4), StagesD+1)
                                    : cute::min(EpiTiles, 4);

  if constexpr (detail::sm90_is_ptr_array_tma_v<Schedule>) {
      return Sm90PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmem, 
                                            DelayTmaStore, Schedule::NumEpilogueWarpGroups>{};
  } 
  else {
    return Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmem, DelayTmaStore>{};
  }
}
```

有了上面的参数后，再通过 Sm90TmaBuilderImpl 构建 OP。Sm90TmaBuilderImpl 代码如下：

前面设置 C 和 D 相关的参数。CopyOpS2G 使用 TMA 把计算结果从共享内存拷贝到 gmem。CopyAtomC 使用 stmatrix 指令把数据从累计寄存器中拷贝到共享内存。

```python
// Helper for building TMA warp-specialized collective epilogues, specialized by
// the fusion operation performed and the dispatch policy to use.
template <
  class TileShape_MNK,
  class EpilogueTile_MN,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD_,
  class GmemLayoutTagD,
  int AlignmentD,
  class FusionOpOrCallbacks,
  class DispatchPolicy
>
struct Sm90TmaBuilderImpl {
  // C/D should meet TMA alignment requirement if not void
  static_assert(detail::is_aligned<ElementC_, AlignmentC, ElementD_, AlignmentD>(),
                "C/D Should meet TMA alignment requirement\n");
  // Passing void D disables destination store + smem allocation
  using ElementD = cute::conditional_t<cute::is_void_v<ElementD_>,
                     fusion::get_element_aux_t<FusionOpOrCallbacks>, ElementD_>;

  // Passing void C disables source load + smem allocation
  using ElementC = cute::conditional_t<cute::is_void_v<ElementC_>,ElementD,ElementC_>; // prevents void ref breakages
  using GmemLayoutTagC = cute::conditional_t<cute::is_void_v<ElementC_>,GmemLayoutTagD,GmemLayoutTagC_>;

  using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;
  
  using UnderlyingGmemStrideTypeC = cute::remove_pointer_t<GmemStrideTypeC>;
  using UnderlyingGmemStrideTypeD = cute::remove_pointer_t<GmemStrideTypeD>;

  using CopyOpS2G = cute::conditional_t<detail::is_im2col_mode<GmemLayoutTagD>,
      SM90_TMA_STORE_IM2COL,
      SM90_TMA_STORE
    >;
  using CopyOpG2S = cute::conditional_t<detail::is_im2col_mode<GmemLayoutTagC>,
      SM90_TMA_LOAD_IM2COL,
      SM90_TMA_LOAD
    >;

  // Get the smallest tiled copy we can use to retile the accumulators
  // using CopyAtomC = Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>;
  using CopyAtomC = cute::conditional_t<
    size<1>(EpilogueTile_MN{}) % 16 == 0,
    Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>,
    cute::conditional_t<
      size<1>(EpilogueTile_MN{}) % 8 == 0,
      Copy_Atom<SM90_U32x2_STSM_N, cutlass::half_t>,
      void
    >
  >;
  static_assert(!cute::is_same_v<CopyAtomC, void>, "CopyAtomC can't be void, divisiblity check for EpilogueTile_MN failed");
  // Get register to register tiled copy that happen before shared memory store.
  // Apply void as no register transform op needed currently.
  using CopyOpR2R = void;

  // TMA builder allows for passing callbacks directly, which is either a fusion::FusionCallbacks
  // instance or a direct visitor implementation, e.g. fusion::Sm90LinearCombination
  using FusionCallbacks = 
    typename CallbacksBuilder<
      DispatchPolicy,
      FusionOpOrCallbacks,
      TileShape_MNK,
      EpilogueTile_MN,
      ElementAccumulator
    >::Callbacks;

  using CollectiveOp = cutlass::epilogue::collective::CollectiveEpilogue<
      DispatchPolicy,
      TileShape_MNK,
      EpilogueTile_MN,
      ElementC_, // Need to pass void through to expose via GemmUniversal
      GmemStrideTypeC,
      ElementD_,
      GmemStrideTypeD,
      FusionCallbacks,
      CopyOpG2S,
      decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<UnderlyingGmemStrideTypeC, ElementC, EpilogueTile_MN>()),
      decltype(detail::sm90_get_smem_load_op_for_source<UnderlyingGmemStrideTypeC, ElementC, EpilogueTile_MN>()),
      CopyOpS2G,
      decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<UnderlyingGmemStrideTypeD, ElementD, EpilogueTile_MN>()),
      decltype(detail::sm90_get_smem_store_op_for_accumulator<UnderlyingGmemStrideTypeD, ElementD, EpilogueTile_MN>()),
      CopyAtomC,
      CopyOpR2R
    >;
};
```

### FusionCallbacks

上面还要设置个 FusionCallbacks。

```python
  using FusionCallbacks = 
    typename CallbacksBuilder<
      DispatchPolicy,
      FusionOpOrCallbacks,
      TileShape_MNK,
      EpilogueTile_MN,
      ElementAccumulator
    >::Callbacks;
```

这里需要一个 FusionOpOrCallbacks，但是前面没有传进来，应该是个 None。应该就是类似 LinearCombination 的操作。

什么是 FusionCallbacks？

简单来说，FusionCallbacks 定义了 “在矩阵乘法（GEMM）计算出结果之后，写入 Global Memory 之前，对数据做了什么操作”。

以下是详细的分析：

1. 核心作用：解耦“数据搬运”与“数学计算”

在 SM90 的 CollectiveEpilogue 设计中，CUTLASS 将 Epilogue 拆分为两个主要部分：

CollectiveEpilogue (数据搬运者):

负责利用 TMA (Tensor Memory Accelerator) 高效地从 Global Memory 加载数据（如矩阵 C 或 Bias 向量）到 Shared Memory。负责将最终结果从寄存器写回 Global Memory (矩阵 D)。处理复杂的 Pipeline 同步和 Tiling 逻辑。

FusionCallbacks (数学计算者):

这就是参数 FusionCallbacks 的意义。它负责在寄存器层面处理数据。它接收 GEMM 的累加器结果（Accumulators）。它接收从 CollectiveEpilogue 加载进来的源数据（Source, 如 C 矩阵、Bias）。它执行具体的算子融合（Fusion），例如：alpha * Acc + beta * C + Bias，或者 ReLU(...)，或者 Cast(fp32 -> fp16)。它将处理后的结果返回给 CollectiveEpilogue 用于存储。

FusionOpOrCallbacks 通常是以下两种情况之一：

预定义的简单操作: 例如 cutlass::epilogue::fusion::Sm90LinearCombination（标准的

）。

Epilogue Visitor Tree (EVT): 这是 CUTLASS 3.x 引入的强大功能。它允许用户定义一个操作树（DAG），例如 RELU( ADD( BIAS( ACC ), C ) )。CallbacksBuilder 会把这个树编译成一个高效的 C++ 类。

3. 具体执行流程

当 Epilogue 运行时，FusionCallbacks 会被“回调”（Invoked），流程如下：

Load: CollectiveEpilogue 把矩阵 C 的一个 Tile 加载到 Shared Memory，然后通过 ldmatrix 加载到寄存器。

Call: CollectiveEpilogue 调用 FusionCallbacks 对象的接口（通常是 operator() 或 visit()）。

输入：GEMM 累加器片段 (Register Fragment)。输入：C 矩阵片段 / Bias 片段 (Register Fragment)。

Compute: FusionCallbacks 内部执行指令（如 HFMA, HADD, MAX 等）：

Result = alpha * Accumulator + beta * C + BiasResult = max(Result, 0) (如果融合了 ReLU)

Return: 计算结果被交还给 CollectiveEpilogue。

Store: CollectiveEpilogue 将结果写入 Shared Memory，最终通过 TMA 存入 Global Memory。

4. 为什么叫 "Callbacks"？

之所以命名为 Callbacks，是因为它采用了 Visitor 模式。

Epilogue 的主循环（Main Loop）控制着迭代过程（遍历 M, N 维度的 Tiles），它是通用的。当它拿到数据后，它需要“回调”特定业务逻辑的代码来处理数据。这个特定的业务逻辑就是 FusionCallbacks。

总结

FusionCallbacks 参数的意思是：Epilogue 融合算子的具体实现类。

输入: 它由 FusionOpOrCallbacks（用户定义的算子描述，如 LinearCombination 或 EVT 树）通过 CallbacksBuilder 生成。

功能: 执行缩放（Scaling）、加偏置（Bias Add）、激活函数（Activation）、数据类型转换（Casting）等操作。

位置: 位于 GEMM 计算之后，Global Memory 存储之前，运行在寄存器（Register）层面。

在 CollectiveBuilder 里面 FusionOpOrCallbacks 默认是 cutlass::epilogue::fusion::LinearCombination<ElementD,ElementCompute,ElementC,ElementCompute>。

最终 FusionCallbacks 类型如下：

```cpp
class "cutlass::epilogue::fusion::FusionCallbacks<
    cutlass::epilogue::Sm90TmaWarpSpecialized<4, 2, 16, true, false>, 
    cutlass::epilogue::fusion::LinearCombination<cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::FloatRoundStyle::round_to_nearest>, 
    cute::tuple<cute::_128, cute::_128, cute::_64>, 
    cute::tuple<cute::C<64>, cute::C<32>>
>"
```

在来具体看下 FusionCallbacks 的定义，由于 fusionop 是 LinearCombination，所以会实例化下面的代码：

```cpp
template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinearCombination<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90LinearCombination<typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementSource, ElementScalar, RoundStyle> {

  using Impl = Sm90LinearCombination<typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinearCombination<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        {    // ternary op : beta * C + (alpha * acc)
          {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
          {},                   // leaf args : C
          {                     // binary op : alpha * acc
            {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
            {},                     // leaf args : acc
            {}                  // binary args : multiplies
          },                    // end binary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};
```

可以看到这个代码继承了 Sm90LinearCombination。

```cpp
template <class NodeOp, class... ChildOps>
using Sm90EVT = Sm90TreeVisitor<NodeOp, ChildOps...>;

// D = alpha * acc + beta * C
template<
  class ElementOutput,
  class ElementCompute,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinearCombination =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + (alpha * acc)
    Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>, // beta
    Sm90SrcFetch<ElementSource>, // C
    Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc
      Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>, // alpha
      Sm90AccFetch // acc
    >
  >;
```

Sm90LinearCombination 是一个模板别名，它定义了一个用于线性组合的 epilogue 操作。具体来说，它实现了 D = alpha * acc + beta * C 的计算。

这里使用了 CUTLASS 的表达式模板（Expression Template）技术，将计算表示为一棵树。这棵树描述了如何组合不同的操作和数据源。

最外层是 Sm90EVT，它表示一个表达式模板节点，该节点执行 homogeneous_multiply_add 操作（即 beta * C + (alpha * acc)）。

第一个子节点是 Sm90ScalarBroadcast，表示标量 beta，它将标量广播到整个张量。

第二个子节点是 Sm90SrcFetch，表示从源张量 C 中获取数据。

第三个子节点是另一个 Sm90EVT，表示 alpha * acc 这个乘法操作。

这个内层 Sm90EVT 的第一个子节点是 Sm90ScalarBroadcast，表示标量 alpha。

第二个子节点是 Sm90AccFetch，表示从累加器（即 GEMM 的结果）中获取数据。

这样，整个表达式就构建了 beta * C + alpha * acc 的计算图。

```cpp
      homogeneous_multiply_add (beta*C + alpha*acc)
      /         |          \
beta_broadcast  C_fetch   multiplies (alpha*acc)
                            /       \
                 alpha_broadcast   acc_fetch
```

所以 epilogue 在计算时会按照这个图来计算。

最后就是把所有参数传到 cutlass::epilogue::collective::CollectiveEpilogue 里来实例化 epilogue。

```python
  using CollectiveOp = cutlass::epilogue::collective::CollectiveEpilogue<
      DispatchPolicy,
      TileShape_MNK,
      EpilogueTile_MN,
      ElementC_, // Need to pass void through to expose via GemmUniversal
      GmemStrideTypeC,
      ElementD_,
      GmemStrideTypeD,
      FusionCallbacks,
      CopyOpG2S,
      decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<UnderlyingGmemStrideTypeC, ElementC, EpilogueTile_MN>()),
      decltype(detail::sm90_get_smem_load_op_for_source<UnderlyingGmemStrideTypeC, ElementC, EpilogueTile_MN>()),
      CopyOpS2G,
      decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<UnderlyingGmemStrideTypeD, ElementD, EpilogueTile_MN>()),
      decltype(detail::sm90_get_smem_store_op_for_accumulator<UnderlyingGmemStrideTypeD, ElementD, EpilogueTile_MN>()),
      CopyAtomC,
      CopyOpR2R
    >;
```

sm90_get_epilogue_smem_swizzle_layout_atom 就是用来获取 epilogue 处理 C 和 D 时合适的 smem swizzle layout，通过 UnderlyingGmemStrideTypeC 判断是行主序还是列主序，根据 EpilogueTile_MN 的大小选择合适的 swizzle atom。

sm90_get_smem_load_op_for_source 用来选择合适的 ldmatrix 大小来加载 smem 中的数据。

sm90_get_smem_store_op_for_accumulator 用来选择合适的 stmatrix 大小来保存寄存器到共享内存中。

最后创建 gemm。

```python
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    Gemm gemm;
```

### 代码启动

先是确定参数，

```python
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {A, stride_A, B, stride_B},
        {{alpha, beta}, C, stride_C, C, stride_C},
        kernel_hw_info};

    cutlass::gemm::GemmUniversalMode mode{}; //maintained here for backward compatibility
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
```

然后是 get_workspace_size。这次还是 0。

然后是 can_implement，判断能否满足实现条件。

然后是 initialize 初始化参数。这里会调用各个文件的 to_underlying_arguments 函数。

kernel 的 to_underlying_arguments 函数。

```python
  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {

    (void) workspace;
    // 对于 gemm 会直接 return args.problem_shape
    auto problem_shape_mnkl = cutlass::conv::detail::get_problem_shape_MNKL_helper<CollectiveMainloop>(args.problem_shape, cute::conditional_t<IsConvProblemShape, cute::true_type, cute::false_type>{});
    auto transformed_problem_shape = cutlass::conv::detail::get_transformed_problem_shape_MNKL(args.problem_shape);

    auto swapped_problem_shape = problem_shape_mnkl;
    if constexpr (detail::Has_SwapAB_v<CollectiveMainloop>) {
      // swap M/N
      get<0>(swapped_problem_shape) = get<1>(problem_shape_mnkl);
      get<1>(swapped_problem_shape) = get<0>(problem_shape_mnkl);
    }
    return {
      swapped_problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(transformed_problem_shape, args.epilogue, workspace)
    };
  }
```

然后调用 CollectiveMainloop::to_underlying_arguments，主要是创建 tma 描述符。

```python
  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    auto ptr_A = reinterpret_cast<InternalElementA const*>(args.ptr_A);
    auto ptr_B = reinterpret_cast<InternalElementB const*>(args.ptr_B);

    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,L), args.dB));

    typename Params::TMA_A tma_load_a = make_tma_copy_A_sm90(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{});
    typename Params::TMA_B tma_load_b = make_tma_copy_B_sm90(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{});
    uint32_t transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t transaction_bytes_nk = TmaTransactionBytesNK;
    uint32_t transaction_bytes = transaction_bytes_mk + transaction_bytes_nk;

    return {
      tma_load_a,
      tma_load_b,
      transaction_bytes,
      transaction_bytes_mk,
      transaction_bytes_nk
    };
  }
```

然后是 epilogue 的，也是创建 tma_load_c，gmen to smem 的 tma 和 tma_store_d，smem2gmem 的 tma。以及 FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, workspace)。

```python
  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    uint32_t transaction_bytes = TmaTransactionBytes;
    typename Params::TMA_C tma_load_c{};
    if constexpr (is_source_supported) {
      Tensor tensor_c = make_tensor(make_gmem_ptr<TmaElementC const>(args.ptr_C), make_layout(make_shape(M,N,L), args.dC));
      tma_load_c = make_tma_copy_C_sm90(
          CopyOpG2S{},
          tensor_c,
          take<0,2>(SmemLayoutC{}),
          EpilogueTile{});
    }

    typename Params::TMA_D tma_store_d{};
    if constexpr (is_destination_supported) {
      Tensor tensor_d = make_tensor(make_gmem_ptr<TmaElementD>(args.ptr_D), make_layout(make_shape(M,N,L), args.dD));
      tma_store_d = make_tma_copy_C_sm90(
          CopyOpS2G{},
          tensor_d,
          take<0,2>(SmemLayoutD{}),
          EpilogueTile{});
    }

    return {
      FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, workspace),
      tma_load_c,
      tma_store_d,
      transaction_bytes
    };
  }
```

最后就是 run 了。

```python
    dim3 const block = GemmKernel::get_block_shape(); // dim3(MaxThreadsPerBlock, 1, 1);
    dim3 const grid = get_grid_shape(params);
```

首先获取 block 和 grid。get_block_shape 返回的是 dim3(MaxThreadsPerBlock, 1, 1)，其中 MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{})) + (NumLoadWarpGroups * NumThreadsPerWarpGroup);，一个 tiledmma 有 128 个线程，再加上一个 warpgroup 一共 256 个线程。NumLoadWarpGroups 在这里等于 1 。

然后是 dim3 const grid = get_grid_shape(params)，会进入到 TileScheduler::get_tiled_cta_shape_mnl 函数。这里因为在初始化 kernel 的时候没有指定 tilescheduler，所以会统一初始化成 PersistentTileSchedulerSm90。

```python
  static dim3
  get_grid_shape(Params const& params) {
    auto cluster_shape = ClusterShape{};
    auto tile_shape = TileShape{};
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    return TileScheduler::get_tiled_cta_shape_mnl(
        problem_shape_MNKL, tile_shape, cluster_shape);
  }
```

在 PersistentTileSchedulerSm90 中，get_tiled_cta_shape_mnl 定义如下，其中 Params 是 PersistentTileSchedulerSm90Params。

```python
  // Given the inputs, computes the total number of output blocks over which this problem will compute. 
  // Note that this is only the logical size of our grid, not the physical grid we will actually launch.
  template<class ProblemShapeMNKL, class BlockShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_tiled_cta_shape_mnl(ProblemShapeMNKL problem_shape_mnkl, BlockShape cta_shape, ClusterShape cluster_shape) {
    auto cta_m = cute::size(cute::ceil_div(cute::shape<0>(problem_shape_mnkl), cute::shape<0>(cta_shape)));
    auto cta_n = cute::size(cute::ceil_div(cute::shape<1>(problem_shape_mnkl), cute::shape<1>(cta_shape)));

    return Params::get_tiled_cta_shape_mnl(
      to_gemm_coord(problem_shape_mnkl),
      to_gemm_coord(cluster_shape),
      cta_m, cta_n
    );
  }
```

反正最后返回的也是普通的 grid（32×32），没有经过 swizzle。

最后是 kernel 的启动，cutlass 中 kernel 的启动有许多条件的判断，具体如下：

![](/assets/sm90_gemm_tma_ws/image.png)

### pipeline 定义

启动后会进入到 sm90_gemm_tma_warpspecialized.hpp 的 operator()(Params const& params, char* smem_buf)来执行 kernel。

因为这个 kernel 有两个 warpgroup，所以第一个 warpgroup 是生产者，第二个 warpgroup 是消费者。其中生产者又被分为 MainloopEpilogue，Warp1，Warp2，Warp3。

然后先用一个线程把 tma 描述符从 host 加载到 device 上。

然后定义 MainloopPipeline = typename CollectiveMainloop::MainloopPipeline 和每个线程的角色，其中，using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;

```cpp
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = NumThreadsPerWarpGroup;
    mainloop_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, mainloop_pipeline_params, ClusterShape{});
```

然后是定义 EpiLoadPipeline，其中 using LoadPipeline = cutlass::PipelineTransactionAsync<StagesC>; 看起来是 PipelineTmaAsync 的简化版本，没有 cluster 上的操作，也没使用 arrive_expect，应该是通用的异步传输 pipeline。

```python
    // Epilogue Load pipeline
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
    epi_load_pipeline_params.consumer_arv_count = NumThreadsPerWarpGroup;
    if constexpr (CollectiveEpilogue::RequiresTransactionBytes) {
      epi_load_pipeline_params.transaction_bytes = params.epilogue.tma_transaction_bytes;
    }
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);
```

然后还有个 EpiStorePipeline。

```cpp
    // Epilogue Store pipeline
    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);


//StorePipeline
  using StorePipeline = cute::conditional_t<ReuseSmemC,
                          cutlass::PipelineTmaStore<StagesC, StagesD-1>,
                          cutlass::PipelineTmaStore<StagesD>>;
```

然后还要初始化一些 PipelineState。using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>; using LoadPipelineState = cutlass::PipelineState<StagesC>;

上面是 consumer 的 pipeline state，用于跟踪 consumer 的状态，下面是 producer 的 pipeline state，用于跟踪 producer 的状态。

```cpp
    // Initialize starting pipeline states for the collectives
    // Epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    // For the DMA Load (producer) we start with an opposite phase
    // i.e., we skip all waits since we know that the buffer is indeed empty
    PipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
    PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();
```

然后实例化 collective_mainloop 和 collective_epilogue。

```cpp
    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);
```

### SharedStorage

此外，对于 SharedStorage 的定义如下，可以看到 MainloopTensorStorage 和 EpilogueTensorStorage 是同一块内存，这是因为 mainloop 计算完成后才会进行 epilogue 的处理。然后还有 pipeline 内存的大小。

```cpp
  // Kernel level shared memory storage
  struct SharedStorage {
    // Mainloop and epilogue don't use smem concurrently since kernel is non-persistent, so we can use a union
    union TensorStorage {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    } tensors;

    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
    } pipelines;
  };
```

对于 mainloop，TensorStorage 包含 smem_A 和 smem_B 的大小。PipelineStorage 包含 stage 个 full_barrier_和 empty_barrier_。full_barrier_用于跟踪 tma，empty_barrier_用于跟踪 wgmma。

```cpp
  struct SharedStorage
  {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;
  
// PipelineStorage
  struct SharedStorage {
    FullBarrier full_barrier_[Stages];
    EmptyBarrier empty_barrier_[Stages];
  };
```

对于 epilogue，需要判断有没有 C 矩阵，因为会有类似 D = a * AB + b * C 的操作。会根据 is_source_supported 和 ReuseSmemC 申请具体大 smem 大小。LoadPipeline::SharedStorage 也包括 full_barrier_和 empty_barrier_。

```cpp
  struct CollectiveStorageWithC {
    alignas(SmemAlignmentC) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
    alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

  union CollectiveStorageWithoutC {
    cute::array<SmemElementC, 0> smem_C;
    alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

  union CollectiveStorageReuseC {
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

  struct SharedStorage {
    struct TensorStorage {
      using CollectiveStorage = cute::conditional_t<not is_source_supported, CollectiveStorageWithoutC,
                                  cute::conditional_t<ReuseSmemC, CollectiveStorageReuseC, CollectiveStorageWithC>>;
      CollectiveStorage collective;

      using FusionStorage = typename FusionCallbacks::SharedStorage;
      FusionStorage thread;
    } tensors;

    using PipelineStorage = typename LoadPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;
```

StorePipeline 因为不需要 mbarrier 跟踪，所以不需要 smem 空间。

后面会执行 collective_mainloop.load_init，其实就是对 A 和 B 进行分区。

```cpp
auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);

// in collective_mainloop
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K,L));                            // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K,L));                            // (n,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});        // (BLK_N,BLK_K,n,k,l)

    return cute::make_tuple(gA_mkl, gB_nkl);
  }
```

然后获取当前 threadblock 需要处理的坐标，auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

然后就使用 if else 区分 producer 和 consumer，并执行不同的代码。

对于 producer，只有 producer 中属于 MainloopEpilogue 的 warp 才会执行代码。

```cpp
    if (warp_group_role == WarpGroupRole::Producer) {
      if (producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
        // Ensure that the prefetched kernel does not touch
        // unflushed global memory prior to this instruction
        cutlass::arch::wait_on_dependent_grids();
        collective_mainloop.load(
          params.mainloop,
          mainloop_pipeline,
          mainloop_pipe_producer_state,
          load_inputs,
          blk_coord,
          k_tile_iter, k_tile_count,
          lane_idx,
          block_rank_in_cluster,
          shared_storage.tensors.mainloop
        );
        // Update starting mainloop pipeline state for the pipeline drain
        mainloop_pipe_producer_state.advance(k_tile_count);
        // Make sure mainloop consumer has been waited upon before issuing epilogue load
        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);

        if (collective_epilogue.is_producer_load_needed()) {
          // Ensure warp is converged before issuing epilogue loads
          __syncwarp();
          epi_load_pipe_producer_state = collective_epilogue.load(
            epi_load_pipeline,
            epi_load_pipe_producer_state,
            problem_shape_MNKL,
            blk_shape,
            blk_coord,
            tiled_mma,
            lane_idx,
            shared_storage.tensors.epilogue
          );
          collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
        }
      } 
    }
```

会先执行 collective_mainloop.load 函数。传入的参数分别是：params.mainloop，mainloop 的参数。mainloop_pipeline，mainloop 用的 pipeline。mainloop_pipe_producer_state，跟踪 producer 的 state。load_inputs，load_init 的返回结果，分块后的矩阵 A 和 B。和其他参数。

## Producer

### mainloop 逻辑

load，load 的逻辑比较简单，就是用一个线程不停的使用 tma 加载数据。前面是用 tma 对矩阵进行分区的代码，用于确定 tma 需要加载的 gmem 和 smem 之间的关系。

```cpp
  template <
    class TensorA, class TensorB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    int lane_predicate = cute::elect_one_sync(); // 选一个线程

    if (lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)

      //
      // Prepare the TMA loads for A and B
      //

      constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());
      uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

      Tensor gA_mkl = get<0>(load_inputs);
      Tensor gB_nkl = get<1>(load_inputs);

      auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
      auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
      Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)
      Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)

      // Applies the mapping from block_tma_a
      Tensor tAgA = block_tma_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

      uint16_t mcast_mask_a = 0;
      uint16_t mcast_mask_b = 0;

      // Issue TmaLoads
      // Maps the tile -> block, value
      if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int n = 0; n < size<1>(block_layout); ++n) {
          mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));
        }
      }

      if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int m = 0; m < size<0>(block_layout); ++m) {
          mcast_mask_b |= (uint16_t(1) << block_layout(m,cluster_local_block_id.y,Int<0>{}));
        }
      }

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        ++k_tile_iter;

        // Advance smem_pipe_write
        ++smem_pipe_write;
      }
    }
  }
```

mainloop 里会先等待 pipeline.producer_acquire(smem_pipe_write)，smem_pipe_write 就是 mainloop_pipe_producer_state，记录了跟踪的某个 mbarrier 的相位等信息。

producer_acquire 的具体实现如下，在 producer_acquire 里会先等待当前 stage 的 empty_barrier_ptr_完成，empty_barrier_ptr_跟踪的是 consumer，完成表示 consumer 对第 stage 处的 smem buffer 已经消费完成，可以写入数据了。

然后用 leader 线程调用 full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);，full_barrier_ptr_[stage]跟踪的是第 stage 个 buffer 的 tma 状态，arrive_and_expect_tx 是 mbarrier 的一个函数，表示线程已经到达此处并且设置了需要传输的数据量 transaction_bytes。后面需要传输这么多数据量才能通过 wait。

```cpp
  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase) {
    empty_barrier_ptr_[stage].wait(phase);

    if (params_.is_leader) {
      full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);
    }
    #ifndef NDEBUG
    if (params_.role == ThreadCategory::Consumer || params_.role == ThreadCategory::NonParticipant) {
      asm volatile ("brkpt;\n" ::);
    }

    // Most likely you have elected more than one leader
    if (params_.is_leader && (threadIdx.x % 32 != 0)) {
      asm volatile ("brkpt;\n" ::);
    }
    #endif
  }
```

然后再通过 producer_get_barrier 获取当前 stage 的 full_barrier_ptr_，配合 tma 进行数据加载。加载完后更新 smem_pipe_write 和 k_tile_iter。这就是 load 的功能。

load 完成后会运行 mainloop_pipe_producer_state.advance(k_tile_count); 把 mainloop_pipe_producer_state 前进 k_tile_count 次，说是为了防止 pipeline 早退出，具体是啥原理呢。

然后就是 collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);，说是 Make sure mainloop consumer has been waited upon before issuing epilogue load。

load_tail 代码如下，只有一个 producer_tail，producer_tail 的代码如下，看起来是要等待所有 stage 的 consumer 完成。

```cpp
  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      /* 这有助于避免 cluster 中的块提前退出，等待所有阶段被释放（所有消费者解锁），或者如果该阶段从未使用过，则将仅获取该阶段，因为该阶段仍从 make_ Producer_start_state 反转
       */
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    detail::pipeline_check_is_producer(params_.role);
    for (int count = 0; count < Stages; ++count) {
      empty_barrier_ptr_[state.index()].wait(state.phase());
      ++state;
    }
  }

```

### epilogue 逻辑

然后是 collective_epilogue.load，主要是加载矩阵 C，如果是 D = a * AB + b * C 这种操作，因此需要先判断是否需要加载 C。

```cpp
        if (collective_epilogue.is_producer_load_needed()) {
          // Ensure warp is converged before issuing epilogue loads
          __syncwarp();
          epi_load_pipe_producer_state = collective_epilogue.load(
            epi_load_pipeline,
            epi_load_pipe_producer_state,
            problem_shape_MNKL,
            blk_shape,
            blk_coord,
            tiled_mma,
            lane_idx,
            shared_storage.tensors.epilogue
          );
          collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
        }
```

is_producer_load_needed 会调用 fusion_callbacks.is_producer_load_needed();

从前面的分析中知道，fusion_callbacks 继承了 Sm90LinearCombination，而 Sm90LinearCombination 又是一个 EVT，所以最终会调用到 beta * C + Z （homogeneous_multiply_add）的 Sm90TreeVisitor 的 is_producer_load_needed。代码如下：

```cpp
  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    auto const& scale_op = get<0>(Impl::ops);
    auto const& added_op = get<2>(Impl::ops);
    if constexpr (detail::IsScalarBroadcast<InputScaleOp>::value && not is_void_v<ElementSource>) {
      return (get<2>(scale_op.params_ptr->dScalar[0]) != 0 && scale_op.params_ptr->scalar_ptrs[0] != nullptr) ||
              is_C_load_needed() ||
              added_op.is_producer_load_needed();
    }
    else {
      return is_C_load_needed() || added_op.is_producer_load_needed();
    }
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    auto const& scale_op = get<0>(Impl::ops);
    auto const& src_op = get<1>(Impl::ops);
    auto const& added_op = get<2>(Impl::ops);
    return (not scale_op.is_zero() && src_op.is_C_load_needed()) || added_op.is_C_load_needed();
  }
```

上面的代码中 scale_op 就是 beta，added_op 是 alpha * acc 的结果。因为 get<2>(scale_op.params_ptr->dScalar[0]) == 0，scale_op.params_ptr->scalar_ptrs[0] is nullptr。is_C_load_needed() 也是 False。

added_op.is_producer_load_needed() 就是 alpha * acc 的 Sm90TreeVisitor 的 is_producer_load_needed。因为 Sm90Compute<multiplies, ElementCompute, ElementCompute, RoundStyle>没有特化的 EVT 模板，所以会实例化普通的 EVT 模板，然后调用 Sm90Compute 里的 is_producer_load_needed，因为是对 acc 的处理，所以会直接返回 false。

因此 collective_epilogue.is_producer_load_needed()就是 FALSE，就不需要 epilogue 加载矩阵 C。

虽然不需要 collective_epilogue.load，但是还是可以看一下 collective_epilogue.load 的逻辑。

```cpp
    // Represent the full source tensor, slice to get the tile this CTA is currently responsible for
    Tensor mC_mn = params.tma_load_c.get_tma_tensor(make_shape(M,N,L));                                //       (M,N,L)
    Tensor mC = coalesce(mC_mn, take<0,2>(CtaTileMNK{}));
    Tensor gC = local_tile(mC, take<0,2>(CtaTileMNK{}), coord_shape);                                  // (CTA_M,CTA_N)

    // Apply epilogue subtile, get matching smem tensor
    auto ptr_sC = shared_tensors.collective.smem_C.begin();
    Tensor gC_epi = flat_divide(gC, EpilogueTile{});                             // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor sC_epi = make_tensor(make_smem_ptr(ptr_sC), SmemLayoutC{});           //      (EPI_TILE_M,EPI_TILE_N,PIPE_C)

    // Prepare the thread(b)lock's (G)mem to (S)mem TMA tiled copy (bGS_)
    ThrCopy thrblk_g2s = params.tma_load_c.get_slice(Int<0>{});
    Tensor bGS_gC = thrblk_g2s.partition_S(gC_epi);                                    // (G2S,G2S_M,G2S_N,EPI_M,EPI_N)
    Tensor bGS_sC = thrblk_g2s.partition_D(sC_epi);                                    // (G2S,G2S_M,G2S_N,PIPE_C)
```

load 的前面主要还是使用 cute 的形状运算，建立 tma 在 gmem 和 smem 上的拷贝关系。

```cpp
    // Get the fusion callbacks for the producer load warp
    auto pld_args = cutlass::epilogue::fusion::detail::ProducerLoadArgs(
                      problem_shape_mnkl,
                      CtaTileMNK{},
                      tile_coord_mnkl,
                      tiled_mma,
                      EpilogueTile{},
                      thread_idx
                    );
    auto pld_callbacks = fusion_callbacks.get_producer_load_callbacks(pld_args);
    bool is_C_load_needed = is_source_supported && fusion_callbacks.is_C_load_needed();
```

这段是通过 get_producer_load_callbacks 获取 fusion_callbacks 函数。get_producer_load_callbacks 会调用到 sm90_visitor_tma_warpspecialized.hpp 中的 get_producer_load_callbacks。

这个函数负责收集和组合多个操作的生产者加载回调。

```cpp
  // Producer load callbacks factory
  // All operations must redefine this, but most can just dispatch to the base impl
  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return transform_apply(ops,
      [&] (auto& op) CUTLASS_LAMBDA_FUNC_INLINE {
        return op.get_producer_load_callbacks(args);
      },
      [] (auto&&... callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
        auto callbacks_tuple = cute::make_tuple(callbacks...);
        return ProducerLoadCallbacksImpl<decltype(callbacks_tuple)>{callbacks_tuple};
      }
    );
  }
```

1. transform_apply 是一个高阶函数，它接受一个元组（或类似元组的结构）ops，一个转换函数和一个合并函数。

具体来说，这里：

第一个 lambda 表达式对 ops 中的每一个元素（每个 op）调用 get_producer_load_callbacks(args)，从而将每个操作转换为对应的回调对象。

第二个 lambda 表达式将转换后得到的多个回调对象（callbacks...）打包成一个元组，然后用这个元组构造一个 ProducerLoadCallbacksImpl 对象。

1. 第一个 lambda：

[&] (auto& op) CUTLASS_LAMBDA_FUNC_INLINE {

return op.get_producer_load_callbacks(args);

}

这个 lambda 捕获了外部变量 args，对每个 op 调用其 get_producer_load_callbacks 方法，并返回结果。

1. 第二个 lambda：

[] (auto&&... callbacks) CUTLASS_LAMBDA_FUNC_INLINE {

auto callbacks_tuple = cute::make_tuple(callbacks...);

return ProducerLoadCallbacksImpl<decltype(callbacks_tuple)>{callbacks_tuple};

}

这个 lambda 接受一系列回调对象，将它们打包成一个元组，然后用这个元组构造一个 ProducerLoadCallbacksImpl 对象。

1. 最终返回的是 ProducerLoadCallbacksImpl 对象，它包含了所有操作的生产者加载回调。

这个函数的设计目的是将多个操作（可能是一个操作集合）的生产者加载回调合并成一个统一的回调接口。这样，在后续的处理中，可以通过这个统一的接口来调用各个操作的回调。

ProducerLoadCallbacksImpl 类如下所示。

```cpp
//
// Producer load callbacks, called by the epilogue load warp.
// Operations usually only define this if TMA load is needed. Most operations will reuse this empy implementation
// Load callbacks are responsible for issuing corresponding mbarrier expect-tx ops for any TMA loads issued, but
// are not responsible for issuing the producer_commit barrier arrival, which is issued by the collective instead
// If this is non-empty, is_producer_load_needed must be true.
//
template <class CallbacksTuple>
struct ProducerLoadCallbacksImpl {
  // Callbacks can store non-persistent variables (e.g. tensors) or copies of persistent variables
  CallbacksTuple callbacks_tuple;

  // Before entry of the subtile load loop
  CUTLASS_DEVICE void
  begin() {
    for_each(callbacks_tuple,
      [&] (auto& callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
        callbacks.begin();
      }
    );
  }

  // Entry of the subtile load loop. Aux loads usually performed here
  // Upon entry the producer acquire of the current subtile lock has completed.
  // Upon exit all TMA loads for this subtile must have been issued, with corresponding expect-tx operations
  CUTLASS_DEVICE void
  step(uint64_t* full_mbarrier_ptr, int epi_m, int epi_n, int load_iteration, bool issue_tma_load) {
    for_each(callbacks_tuple,
      [&] (auto& callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
        callbacks.step(full_mbarrier_ptr, epi_m, epi_n, load_iteration, issue_tma_load);
      }
    );
  }

  // Exit of the subtile load loop.
  CUTLASS_DEVICE void
  end() {
    for_each(callbacks_tuple,
      [] (auto& callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
        callbacks.end();
      }
    );
  }
};
```

所以在 epilogue load 里 get_producer_load_callbacks 会返回嵌套的 ProducerLoadCallbacksImpl 对象。这些回调由 Epilogue 加载 Warp 调用，只有需要 TMA 加载的操作才需要定义这些回调，大多数操作可以重用这个空实现。*空操作，会被编译器优化掉*

```cpp
    // Pre-loop fusion callback entry point
    pld_callbacks.begin();

    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < size<3>(gC_epi); ++epi_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < size<2>(gC_epi); ++epi_m) {
        if (subtile_idx != -1 && (epi_n * static_cast<int>(size<2>(gC_epi)) + epi_m) != subtile_idx) {
          continue;
        }
        // Acquire the lock for this stage
        constexpr uint16_t mcast_mask = 0;
        uint64_t* tma_barrier = load_pipeline.producer_get_barrier(load_pipe_producer_state);  // 获取跟踪 tma 的 mbarrier。
        load_pipeline.producer_acquire(load_pipe_producer_state);  // 等待 consumer 完成。

        // Loop fusion callback entry point
        pld_callbacks.step(tma_barrier, epi_m, epi_n, load_pipe_producer_state.count(), issue_tma_load);

        // Execute the TMA load for C if needed
        if (issue_tma_load && is_C_load_needed) {
          copy(params.tma_load_c.with(*tma_barrier, mcast_mask),
              bGS_gC(_,_,_,epi_m,epi_n), bGS_sC(_,_,_,load_pipe_producer_state.index()));
          load_pipeline.producer_expect_transaction(load_pipe_producer_state);
        }

        // Commit TMA loads for this stage and release the lock
        load_pipeline.producer_commit(load_pipe_producer_state);
        ++load_pipe_producer_state;
      }
    }

    // Post-loop fusion callback entry point
    pld_callbacks.end();
```

后面再运行循环前会先运行 pld_callbacks.begin();，但是看起来这个 begin 很多都是空操作，只有 row broadcast 或 col broadcast 的时候才会用到。

然后就是在 K 方向上 for 循环

加载数据前会先用 producer_acquire 等待 consumer 完成。

pld_callbacks.step() 看起来是用 tma 加载一些额外的 Tensor，一般用不到。

下面是用 tma 加载 Tensor C，并使用 producer_expect_transaction 设置传输的数据量。

因为这里没有使用 arrive-expect-tx，所以还需要用 producer_commit 函数设置下 arrive，表示线程已经到达了此处。

循环结束后调用 pld_callbacks.end(); 看起来会保存一些 scale 之类的额外参数，如果有的话。

collective_epilogue.load 后是 collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);，一直等待所有 consumer 完成。

```cpp
  CUTLASS_DEVICE auto
  load_tail(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_producer_state) {
    bool issue_tma_load = cute::elect_one_sync();
    if (issue_tma_load) {
      load_pipeline.producer_tail(load_pipe_producer_state);
    }

    return load_pipe_producer_state;
  }
```

## Consumer

### mianloop 逻辑

如果是 consumer warpgroup 的话。会先通过 partition_fragment_C 获取累加器 Tensor。然后调用 collective_mainloop.mma 。

```cpp
      Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));                 // (MMA,MMA_M,MMA_N)

      collective_mainloop.mma(
        mainloop_pipeline,
        mainloop_pipe_consumer_state,
        accumulators,
        k_tile_count,
        warp_group_thread_idx,
        shared_storage.tensors.mainloop,
        params.mainloop
      );
```

在 collective_mainloop.mma 里，前面也是先对 tensor 分块。K_PIPE_MMAS = 1 表示 prologue_mma_count = 1。

在 prologue 阶段，会先等待一个 stage 的 tma 完成，然后计算 gemm。

```cpp
    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    assert(k_tile_count >= 1);
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    warpgroup_fence_operand(accum);
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token); // 等待第一个 stage 的 tma 完成。

      int read_stage = smem_pipe_read.index();
      warpgroup_arrive();  // wgmma.fence.sync.aligned
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }

      warpgroup_commit_batch();

      ++smem_pipe_read;
    }
```

prologue 结束后就是正式的 for 循环。

```cpp
    warpgroup_fence_operand(accum);
    // Mainloop GMMAs
    k_tile_count -= prologue_mma_count;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count)
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();
      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      // (V,M,K) x (V,N,K) => (V,M,N)
      cute::gemm(tiled_mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), accum);
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(accum);

      // UNLOCK smem_pipe_release, done _computing_ on it
      pipeline.consumer_release(smem_pipe_release);

      // Advance smem_pipe_read and smem_pipe_release
      ++smem_pipe_read;
      ++smem_pipe_release;
    }

    warpgroup_fence_operand(accum);
  }
```

for 循环里先等待下一个 stage 的 tma 完成，然后提交 wgmma，此时已经有两个 wgmma batch 了，然后通过 warpgroup_wait 等待第一个完成，完成后调用 consumer_release，表示跟踪 wgmma 的 mbarrier 已经 arrive 了，说明第一个 stage 的 buffer 消费完成，可以用 tma 加载了。

此时 producer 里的 load 函数会通过 wait 的等待加载数据到这个 stage 的 buffer。

mma tail 就是等待所有的 wgmma 完成，然后让所有的 stage 都 arrive。

```cpp
  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }
```

### epilogue 逻辑

consumer 的 mma 完成后开始进行 collective_epilogue.store。

```cpp
      // Epilogue and write to gD
      auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next] =
      collective_epilogue.store(
        epi_load_pipeline,
        epi_load_pipe_consumer_state,
        epi_store_pipeline,
        epi_store_pipe_producer_state,
        problem_shape_MNKL,
        blk_shape,
        blk_coord,
        accumulators,
        tiled_mma,
        warp_group_thread_idx,
        shared_storage.tensors.epilogue
      );
```

store 函数前面是复杂的处理 Tensor C 和 D 的逻辑。然后是获取 cst_callbacks。

大致就是加载 C（如果需要），处理一些 element 的操作，然后保存到 D 上。细节暂时没看懂。

最后在用 collective_epilogue.store_tail 确保所有 store 结束。

性能测试：

cublas time = 0.180039 ms, TFLPOS = 763.385407, mfu = 0.771876

mma time = 0.233156 ms, TFLPOS = 589.470840, mfu = 0.596027
