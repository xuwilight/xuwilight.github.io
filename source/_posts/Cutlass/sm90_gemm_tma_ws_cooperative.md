---
title: SM90 GEMM TMA WS Cooperative
date: 2025-08-15 12:00:00
tags: [CUTLASS, GEMM, TMA, WarpSpecialized, Cooperative, SM90, GPU]
categories: [Cutlass 学习笔记]
description: ws_cooperative 也是一个 persistent kernel，与 pingpong 的区别是存在两个消费者 warp groups 协作处理同一块 output tile，方法是将 output tile 沿着 M 维度分割成两半。这允许使用更大的 output tile，因为每个消费者 warp group 的寄存器压力降低了，从而可以提高性能。
---

ws_cooperative 也是一个 persistent kernel，与 pingpong 的区别是存在两个消费者 warp groups 协作处理同一块 output tile，方法是将 output tile 沿着 M 维度分割成两半。这允许使用更大的 output tile，因为每个消费者 warp group 的寄存器压力降低了，从而可以提高性能。

代码实现如下：

```cpp
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

#include "../utils.h"

using namespace cute;

/**
 * nvcc sm90_tma_ws_cooperative_gemm.cu -arch=sm_90a -I ../../../include/ -I ../../../tools/util/include/ -lcuda -lcublas -o sm90_tma_ws_cooperative_gemm --expt-relaxed-constexpr && ./sm90_tma_ws_cooperative_gemm
 */

template <class TA, class TB, class TC>
void sm90_tma_ws_cooperative_gemm(int M, int N, int K, TC alpha, TA const *A, int lda, TB const *B, int ldb, TC beta, TC *C, int ldc)
{
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using ArchTag = cutlass::arch::Sm90;
    using OpClass = cutlass::arch::OpClassTensorOp;
    using TileShape = Shape<_256, _128, _64>;
    using ClusterShape = Shape<_1, _1, _1>;
    using MmaSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  // KernelTmaWarpSpecialized, KernelTmaWarpSpecializedPingpong or KernelTmaWarpSpecializedCooperative

    constexpr int stage = 3;
    using MmaDispatchPolicy = cutlass::gemm::MainloopSm90TmaGmmaWarpSpecialized<stage, ClusterShape, MmaSchedule>;  // gemm is warpspecialized
    using TMACOPY = SM90_TMA_LOAD;
    using AtomLayoutMNK = Layout<Shape<_2, _1, _1>>;

    // using TiledMMA = decltype(make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{}));
    using TiledMMA = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<TA, TB, TC, TileShape, GMMA::Major::K, GMMA::Major::K>(), AtomLayoutMNK{}));

    // using SmemLayoutAtomA = decltype(detail::ss_smem_selector<GMMA::Major::K, TA, decltype(cute::get<0>(TileShape{})), decltype(cute::get<2>(TileShape{}))>());
    // using SmemLayoutAtomB = decltype(detail::ss_smem_selector<GMMA::Major::K, TA, decltype(cute::get<1>(TileShape{})), decltype(cute::get<2>(TileShape{}))>());

    using SmemLayoutAtomA = GMMA::Layout_K_SW128_Atom<TA>;
    using SmemLayoutAtomB = GMMA::Layout_K_SW128_Atom<TB>;
    using SmemCopyAtomA = void;
    using SmemCopyAtomB = void;
    using Stride_A = cutlass::detail::TagToStrideA_t<LayoutA>;
    using Stride_B = cutlass::detail::TagToStrideB_t<LayoutB>;
    using Stride_C = cutlass::detail::TagToStrideC_t<LayoutC>;

    using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
        MmaDispatchPolicy,
        TileShape, TA, Stride_A, TB, Stride_B,
        TiledMMA,
        TMACOPY, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,
        TMACOPY, SmemLayoutAtomB, SmemCopyAtomB, cute::identity>;


    // build epilogue
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative; // NoSmemWarpSpecialized, TmaWarpSpecialized or TmaWarpSpecializedCooperative

    constexpr int AlignmentC  = 16;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        TC, TC,
        TC, LayoutC, AlignmentC,
        TC, LayoutC, AlignmentC,
        EpilogueSchedule
        >::CollectiveOp;

    
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    Gemm gemm;

    cutlass::KernelHardwareInfo kernel_hw_info;
    kernel_hw_info.device_id = 0;
    kernel_hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(kernel_hw_info.device_id);
    // cutlass::KernelHardwareInfo kernel_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<Gemm::GemmKernel>(device_id);

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {A, stride_A, B, stride_B},
        {{alpha, beta}, C, stride_C, C, stride_C},
        kernel_hw_info};

    arguments.scheduler.max_swizzle_size = 2;

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

    using TA = half_t;
    using TB = half_t;
    using TC = half_t;

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

    // Allocate device Matrices
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

    sm90_tma_ws_cooperative_gemm(M, N, K, alpha, d_A, M, d_B, K, beta, C_cutlass, N);
    cudaDeviceSynchronize();

    cublasHandle_t handle;
    cublasCreate(&handle);
    // cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, d_A, CUDA_R_16F, K, d_B, CUDA_R_16F, K, &beta, C_cublas, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    // C is column-major
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
                reinterpret_cast<const __half *>(&alpha),
                reinterpret_cast<__half *>(d_A), K,
                reinterpret_cast<__half *>(d_B), K,
                reinterpret_cast<const __half *>(&beta),
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
                        reinterpret_cast<const __half *>(&alpha),
                        reinterpret_cast<__half *>(d_A), K,
                        reinterpret_cast<__half *>(d_B), K,
                        reinterpret_cast<const __half *>(&beta),
                        reinterpret_cast<__half *>(C_cublas), N);
        };

        std::function<void()> custom_func = [&]()
        {
            sm90_tma_ws_cooperative_gemm(M, N, K, alpha, d_A, M, d_B, K, beta, C_cutlass, M);
        };

        run_benchmark(cublas_func, "cublas", flops, h100);
        run_benchmark(custom_func, "mma", flops, h100);
    }
}
```

和 pingpong 的区别是：

1. using TileShape = Shape<_256, _128, _64>; tileshape 一般比 pingpong 大，因为 pingpong 是两个 warpgroup 负责两个 tile，而 cooperative 是两个 warpgroup 共同负责一个 tile。
1. using MmaSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
1. using AtomLayoutMNK = Layout<Shape<_2, _1, _1>>; cooperative 的 tiledmma 的线程数量必须是 256，所以 atomlayout 需要是 2。
1. using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

其余的和 pingpong 的实现一致。

参数设置和启动跟之前一样，下面主要看 sm90_gemm_tma_warpspecialized_cooperative.hpp 的实现。

前面是一堆 using，声明类型。

下面定义一些变量

```cpp
  // Warp specialization thread count per threadblock
  static constexpr uint32_t NumSchedThreads        = NumThreadsPerWarp;      // 1 warp       
  static constexpr uint32_t NumMMAThreads          = size(TiledMma{});       // 8 warps
  static constexpr uint32_t NumMainloopLoadThreads = NumThreadsPerWarp;      // 1 warp
  static constexpr uint32_t NumEpilogueLoadThreads = NumThreadsPerWarp;      // 1 warp for C

  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;
  static constexpr bool IsGdcEnabled = cutlass::arch::IsGdcGloballyEnabled;

  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = NumMMAThreads / NumThreadsPerWarpGroup;
  static constexpr uint32_t MaxThreadsPerBlock = NumMMAThreads + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumFixupBarriers = NumMmaWarpGroups;
  static constexpr uint32_t NumProducerThreads = CollectiveMainloop::NumProducerThreadEvents;
  static constexpr bool     IsMainloopAuxiliaryLoadNeeded = detail::HasAuxiliaryLoad_v<typename CollectiveMainloop::DispatchPolicy>;
```

这里主要有一个 warp32 个线程，一个 tiledmma 有 8 个 warp，256 个线程。IsSchedDynamicPersistent 在 sm90 上是 FALSE。一个 warpgroup 加载，一个 threadblock 有 256+128 个线程。每个 sm 上最小的 block 数量是 1。

然后设置一下寄存器的数量。对于 128*128 来说，一个线程需要 256*128*2/4/256 = 64 个寄存器。

```cpp
  /// Register requirement for Load and Math WGs
  static constexpr int RegsPerThread =
    size<0>(TileShape{}) * size<1>(TileShape{}) / NumMMAThreads *
    sizeof(ElementAccumulator) / sizeof(uint32_t);
  static constexpr bool HeavyRegisterPressure = RegsPerThread >= 208;
  static constexpr uint32_t LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;
  static constexpr uint32_t MmaRegisterRequirement = !HeavyRegisterPressure ? 232 : 240;
```

然后是 operator()。

在 operator 里，会现有两个 assert

```cpp
    // Preconditions
    static_assert(NumMMAThreads == 256, "Cooperative kernel must have TiledMMA operating using 256 threads.");
    static_assert(size<0>(TileShape{}) >= 128,
        "Cooperative kernel requires Tile Size to be greater than or equal to 128 along the M-dimension.");
```

后面的也没啥特殊的，跟 sm90_gemm_tma_ws 差不多，区别就是 ws 的 tiledmma 只有一个 warpgroup，cooperative 的 tiledmma 有两个 warpgroup。
