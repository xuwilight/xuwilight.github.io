---
title: SM90 GEMM TMA WS PingPong
date: 2025-08-15 12:00:00
tags: [CUTLASS, GEMM, TMA, WarpSpecialized, PingPong, SM90, GPU]
categories: [Cutlass 学习笔记]
description: sm90_gemm_tma_ws_pingpong，大的来了 sm90_gemm_tma_ws_pingpong 和 sm90_gemm_tma_ws 的实现代码完全一样，只需要 using MmaSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;就行。
---

sm90_gemm_tma_ws_pingpong，大的来了

sm90_gemm_tma_ws_pingpong 和 sm90_gemm_tma_ws 的实现代码完全一样，只需要 using MmaSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;就行。

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

// nvcc -O3 -DNDEBUG --expt-relaxed-constexpr sm90_tma_ws_pingpong_gemm.cu -arch=sm_90a -I ../../../include/ -I ../../../tools/util/include/ -lcuda -lcublas -o sm90_tma_ws_pingpong_gemm && ./sm90_tma_ws_pingpong_gemm

template <class TA, class TB, class TC>
void sm90_tma_ws_pingpong_gemm(int M, int N, int K, TC alpha, TA const *A, int lda, TB const *B, int ldb, TC beta, TC *C, int ldc)
{

    // // method 1 use CollectiveBuilder
    // using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    //     cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    //     cute::Shape<cute::_128, cute::_256, cute::_64>,
    //     cute::Shape<cute::_1, cute::_2, cute::_1>,
    //     cutlass::epilogue::collective::EpilogueTileAuto,
    //     TC, TC,
    //     TC, cutlass::layout::ColumnMajor, 8,
    //     TC, cutlass::layout::ColumnMajor, 8,
    //     cutlass::epilogue::TmaWarpSpecialized,
    //     cutlass::epilogue::fusion::LinearCombination<
    //         TC,
    //         TC,
    //         TC,
    //         TC>>::CollectiveOp;

    // using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    //     cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    //     TC, cutlass::layout::RowMajor, 8,
    //     TC, cutlass::layout::ColumnMajor, 8,
    //     TC,
    //     cute::Shape<cute::_128, cute::_256, cute::_64>,
    //     cute::Shape<cute::_1, cute::_2, cute::_1>,
    //     cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    //     cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;

    // method 2
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;

    using ArchTag = cutlass::arch::Sm90;
    using OpClass = cutlass::arch::OpClassTensorOp;
    using TileShape = Shape<_128, _256, _64>;
    using ClusterShape = Shape<_1, _2, _1>;
    using MmaSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong; // KernelTmaWarpSpecialized, KernelTmaWarpSpecializedPingpong or KernelTmaWarpSpecializedCooperative

    constexpr int stage = 4;
    using MmaDispatchPolicy = cutlass::gemm::MainloopSm90TmaGmmaWarpSpecialized<stage, ClusterShape, MmaSchedule>; // gemm is warpspecialized
    using TMACOPY = typename std::conditional<cute::size(ClusterShape{}) == 1,
                                              cute::SM90_TMA_LOAD,
                                              cute::SM90_TMA_LOAD_MULTICAST>::type;
    using AtomLayoutMNK = Layout<Shape<_1, _1, _1>>;

    // using TiledMMA = decltype(make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{}));
    using TiledMMA = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<TA, TB, TC, TileShape, GMMA::Major::K, GMMA::Major::K>(), AtomLayoutMNK{}));

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
    // constexpr int AlignmentC = 8;
    using EpilogueTileMN = Shape<_64, _32>;
    constexpr int ep_stage_C = 4;
    constexpr int ep_stage_D = 2;
    constexpr int FragmentSize = 16;
    constexpr bool ReuseSmem = true;
    constexpr bool DelayTmaStore = false;
    using EpilogueDispatchPolicy = cutlass::epilogue::Sm90TmaWarpSpecialized<ep_stage_C, ep_stage_D, FragmentSize, ReuseSmem, DelayTmaStore>;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized; // NoSmemWarpSpecialized, TmaWarpSpecialized or TmaWarpSpecializedCooperative

    using FusionOp = cutlass::epilogue::fusion::LinearCombination<TC, TC, TC, TC>;
    using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, FusionOp, TileShape, EpilogueTileMN>;

    using CopyOpG2S = SM90_TMA_LOAD;
    using CopyOpS2G = SM90_TMA_STORE;
    using CopyOpS2R = SM75_U16x8_LDSM_T;
    using CopyOpR2S = SM90_U16x8_STSM_T;
    using CopyAtomC = Copy_Atom<SM90_U32x4_STSM_N, TC>;
    using SmemLayoutAtomC = GMMA::Layout_MN_SW128_Atom<TC>;

    using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
        EpilogueDispatchPolicy,
        TileShape, EpilogueTileMN,
        TC, Stride_C, TC, Stride_C,
        FusionCallbacks,
        CopyOpG2S, SmemLayoutAtomC,
        CopyOpS2R, CopyOpS2G, SmemLayoutAtomC,
        CopyOpR2S, CopyAtomC, void>;

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

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {A, stride_A, B, stride_B},
        {{alpha, beta}, C, stride_C, C, stride_C},
        kernel_hw_info};

    // arguments.scheduler.max_swizzle_size = 1;

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

    sm90_tma_ws_pingpong_gemm(M, N, K, alpha, d_A, M, d_B, K, beta, C_cutlass, N);
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
        int64_t flops = 2 * (int64_t(M) * N * K + M * N);
        float h100 = 989e12;

        auto cublas_func = [&]()
        {
            cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
                        reinterpret_cast<const __half *>(&alpha),
                        reinterpret_cast<__half *>(d_A), K,
                        reinterpret_cast<__half *>(d_B), K,
                        reinterpret_cast<const __half *>(&beta),
                        reinterpret_cast<__half *>(C_cublas), N);
        };

        auto custom_func = [&]()
        {
            sm90_tma_ws_pingpong_gemm(M, N, K, alpha, d_A, M, d_B, K, beta, C_cutlass, M);
        };

        run_benchmark(cublas_func, "cublas", flops, h100);
        run_benchmark(custom_func, "mma", flops, h100);
    }
}
```

cutlass 中有两种 kernel 的构建方式，一种是使用 CollectiveBuilder。这种方法会自动根据 shape 大小，数据类型等选择合适的 tiledmma，stage 等。

```cpp
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        cute::Shape<cute::_128, cute::_256, cute::_64>,
        cute::Shape<cute::_1, cute::_2, cute::_1>,
        cutlass::epilogue::collective::EpilogueTileAuto,
        TC, TC,
        TC, cutlass::layout::ColumnMajor, 8,
        TC, cutlass::layout::ColumnMajor, 8,
        cutlass::epilogue::TmaWarpSpecialized,
        cutlass::epilogue::fusion::LinearCombination<
            TC,
            TC,
            TC,
            TC>>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TC, cutlass::layout::RowMajor, 8,
        TC, cutlass::layout::ColumnMajor, 8,
        TC,
        cute::Shape<cute::_128, cute::_256, cute::_64>,
        cute::Shape<cute::_1, cute::_2, cute::_1>,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
```

另一种就是手动设置各种参数，然后创建 CollectiveEpilogue 和 CollectiveMma，这种灵活性比较高，但是需要了解各个参数的意思。

```cpp
    // method 2
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;

    using ArchTag = cutlass::arch::Sm90;
    using OpClass = cutlass::arch::OpClassTensorOp;
    using TileShape = Shape<_128, _256, _64>;
    using ClusterShape = Shape<_1, _2, _1>;
    using MmaSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong; // KernelTmaWarpSpecialized, KernelTmaWarpSpecializedPingpong or KernelTmaWarpSpecializedCooperative

    constexpr int stage = 4;
    using MmaDispatchPolicy = cutlass::gemm::MainloopSm90TmaGmmaWarpSpecialized<stage, ClusterShape, MmaSchedule>; // gemm is warpspecialized
    using TMACOPY = typename std::conditional<cute::size(ClusterShape{}) == 1,
                                              cute::SM90_TMA_LOAD,
                                              cute::SM90_TMA_LOAD_MULTICAST>::type;
    using AtomLayoutMNK = Layout<Shape<_1, _1, _1>>;

    // using TiledMMA = decltype(make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{}));
    using TiledMMA = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<TA, TB, TC, TileShape, GMMA::Major::K, GMMA::Major::K>(), AtomLayoutMNK{}));

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
    // constexpr int AlignmentC = 8;
    using EpilogueTileMN = Shape<_64, _32>;
    constexpr int ep_stage_C = 4;
    constexpr int ep_stage_D = 2;
    constexpr int FragmentSize = 16;
    constexpr bool ReuseSmem = true;
    constexpr bool DelayTmaStore = false;
    using EpilogueDispatchPolicy = cutlass::epilogue::Sm90TmaWarpSpecialized<ep_stage_C, ep_stage_D, FragmentSize, ReuseSmem, DelayTmaStore>;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized; // NoSmemWarpSpecialized, TmaWarpSpecialized or TmaWarpSpecializedCooperative

    using FusionOp = cutlass::epilogue::fusion::LinearCombination<TC, TC, TC, TC>;
    using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, FusionOp, TileShape, EpilogueTileMN>;

    using CopyOpG2S = SM90_TMA_LOAD;
    using CopyOpS2G = SM90_TMA_STORE;
    using CopyOpS2R = SM75_U16x8_LDSM_T;
    using CopyOpR2S = SM90_U16x8_STSM_T;
    using CopyAtomC = Copy_Atom<SM90_U32x4_STSM_N, TC>;
    using SmemLayoutAtomC = GMMA::Layout_MN_SW128_Atom<TC>;

    using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
        EpilogueDispatchPolicy,
        TileShape, EpilogueTileMN,
        TC, Stride_C, TC, Stride_C,
        FusionCallbacks,
        CopyOpG2S, SmemLayoutAtomC,
        CopyOpS2R, CopyOpS2G, SmemLayoutAtomC,
        CopyOpR2S, CopyAtomC, void>;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue>;
```

此外编译的时候需要加上-O3 -DNDEBUG 来获取最佳性能。

```cpp
nvcc -O3 -DNDEBUG --expt-relaxed-constexpr sm90_tma_ws_pingpong_gemm.cu -arch=sm_90a -I ../../../include/ -I ../../../tools/util/include/ -lcuda -lcublas -o sm90_tma_ws_pingpong_gemm && ./sm90_tma_ws_pingpong_gemm
```

上面实现的是 profiler 中 cutlass3x_sm90_tensorop_gemm_f16_f16_f16_f16_f16_128x256x64_1x2x1_0_tnn_align8_warpspecialized_pingpong_epi_tma。虽然模板参数完全一样，但是跑出来的性能有点差别。

在 m = n = k = 4096 的情况下，cutlass_profiler 是 882TFLOPS，使用 CollectiveBuilder 的 kernel 是 808TFLOPS，自己设置参数的 kernel 性能是 767TFLOPS。差异可能与 host 端实现有关系。

下面直接看代码启动部分。

### 代码启动

```cpp
    typename Gemm::Arguments arguments{
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
```

首先还是 get_workspace_size，pingpong 这里在调用到 kernel 的 get_workspace_size 时不是直接返回 0，而是会调用 CollectiveEpilogue::get_workspace_size 和 TileScheduler::template get_workspace_size，但是这两个其实也是 0。

can_implement 没啥区别。

initialize 会把 args 转换成 params。这一步主要是通过 to_underlying_arguments 函数实现。

对于 pingpong params 类和普通 ws 有些不同，普通的 ws 只有 ProblemShapeMNKL，MainloopParams 和 EpilogueParams 这三个参数。而 pingpong 有下面的这么多。所以 to_underlying_arguments 需要处理这么多参数。

```cpp
  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
  };
```

pingpong 的 to_underlying_arguments 代码。会获取一个 sm_count 和 max_active_clusters。除了 CollectiveMainloop::to_underlying_arguments 和 CollectiveEpilogue::to_underlying_arguments，还有一个 TileScheduler::to_underlying_arguments。

```cpp
  template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
  static Params
  to_underlying_arguments(
      ProblemShapeMNKL problem_shape_mnkl,
      TileShape tile_shape,
      ClusterShape cluster_shape,
      [[maybe_unused]] KernelHardwareInfo const& hw_info,
      Arguments const& arguments,
      [[maybe_unused]] void* workspace=nullptr,
      [[maybe_unused]] const uint32_t epilogue_subtile = 1,
      [[maybe_unused]] uint32_t ktile_start_alignment_count = 1u) {

    // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);

    Params params;
    params.initialize(
      problem_blocks,
      to_gemm_coord(cluster_shape),
      hw_info,
      arguments.max_swizzle_size,
      arguments.raster_order
    );

    return params;
  }
```

因为 ws 不会调用 to_underlying_arguments，所以也就不会运行 params.initialize，那么通过 get_tiled_cta_shape_mnl 返回的就是普通的 grid。pingpong 调用 params.initialize，返回 swizzle 后的 grid。

params.initialize 的代码

```cpp
  // Version of initialize that takes in as input the number of CTAs in the M and N and L dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  void
  initialize(
    dim3 problem_blocks,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int max_swizzle_size,
    RasterOrderOptions raster_order_option
  ) {

    CUTLASS_UNUSED(hw_info);

    // Round up to nearest multiple of swizzle_size along each mode
    auto log_swizzle_size = get_log_swizzle_size(problem_blocks.x, problem_blocks.y, max_swizzle_size);
    auto problem_blocks_m = round_up(problem_blocks.x, (1 << log_swizzle_size) * cluster_shape.m());
    auto problem_blocks_n = round_up(problem_blocks.y, (1 << log_swizzle_size) * cluster_shape.n());

    problem_tiles_m_ = problem_blocks_m / cluster_shape.m();
    problem_tiles_n_ = problem_blocks_n / cluster_shape.n();
    problem_tiles_l_ = problem_blocks.z;
    cluster_shape_m_ = cluster_shape.m();
    cluster_shape_n_ = cluster_shape.n();

    RasterOrder raster_order = get_rasterization_order(
      problem_blocks_m,
      problem_blocks_n,
      raster_order_option
    );

    //
    // Set members
    //

    blocks_per_problem_ = problem_blocks_m * problem_blocks_n * problem_blocks.z;
    log_swizzle_size_ = log_swizzle_size;
    raster_order_ = raster_order;
    divmod_batch_ = FastDivmodU64(problem_blocks_m * problem_blocks_n);

    if (raster_order == RasterOrder::AlongN) {
      divmod_cluster_shape_major_ = FastDivmodU64Pow2(cluster_shape.n());
      divmod_cluster_shape_minor_ = FastDivmodU64Pow2(cluster_shape.m());
      divmod_cluster_blk_major_ = FastDivmodU64(problem_blocks_n / cluster_shape.n());
    }
    else {
      divmod_cluster_shape_major_ = FastDivmodU64Pow2(cluster_shape.m());
      divmod_cluster_shape_minor_ = FastDivmodU64Pow2(cluster_shape.n());
      divmod_cluster_blk_major_ = FastDivmodU64(problem_blocks_m / cluster_shape.m());
    }
  }
```

这里因为 log_swizzle_size = 0，所以这里 problem_tiles_m_，problem_tiles_n_还是 32，32。

然后通过 get_rasterization_order 获取 raster_order，返回的是 RasterOrder::AlongN;

```cpp
  CUTLASS_HOST_DEVICE
  static RasterOrder
  get_rasterization_order(
    uint32_t tiles_m,
    uint32_t tiles_n,
    RasterOrderOptions raster_order_option
  ) {

    if (raster_order_option == RasterOrderOptions::Heuristic) {
      if (tiles_n > tiles_m) {
        return RasterOrder::AlongM;
      }
      else {
        return RasterOrder::AlongN;
      }
    }
    else {
      switch (raster_order_option) {
        case RasterOrderOptions::AlongN:
          return RasterOrder::AlongN;
          break;
        default:
          return RasterOrder::AlongM;
      }
    }
  }
```

FastDivmodU64 和 FastDivmodU64Pow2 是为了优化 GPU 核函数（Kernel）中整数除法和取模运算 而设计的辅助类（Helper Classes）。在 GPU 编程中，硬件层面的整数除法（/）和取模（%）指令通常比乘法（*）、加法（+）或位运算（>>, &）要慢得多（延迟更高）。这两个类的核心目的是：在 CPU（Host 端）预先计算好一些“魔数”（Magic Numbers）或位移量，以便在 GPU（Device 端）执行时，用快速的乘法和位运算来替代昂贵的除法指令。

参数初始化完成后会调用 run 进行运行。

在 run 函数里面首先会调用 get_block_shape 和 get_grid_shape 获取 kernel 的 size。

对于 get_block_shape 就是直接返回 dim3(MaxThreadsPerBlock, 1, 1); 其中 MaxThreadsPerBlock = NumMMAThreads * NumMmaWarpGroups + (NumLoadWarpGroups * NumThreadsPerWarpGroup); 也就是 3 个 warpgroup，384 个线程。

然后对于 get_grid_shape，会调用 TileScheduler::get_grid_shape，而之前的 kernel 都是直接调用 TileScheduler::get_tiled_cta_shape_mnl。

```cpp
  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    // Given device SM count, set grid size s.t. we do not launch more thread blocks than we can run concurrently
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN ? TileScheduler::RasterOrderOptions::AlongN : TileScheduler::RasterOrderOptions::AlongM;
    return TileScheduler::get_grid_shape(params.scheduler, params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, args);
  }
```

对于 pingpong 类型的 kernel，因为是 persistent 的，所以 grid 的大小和硬件的 sm 数量一致。又因为默认是沿着 N 方向，所以最终 grid = (1, 132, 1)。而且 swizzle_size 的大小并不会影响这个结果。

### kernel 运行

在 pingpong 的 kernel 内，有一些限制，比如一共有 3 个 warpgroup，一个是 producer，另外两个是 consumer。所以线程总数需要是 384 个。

static_assert(MaxThreadsPerBlock == 384, "Pingpong kernel must have 384 threads in total.");

然后还有对寄存器的设置。如果每个线程使用的寄存器大于 208 个，producer 会使用 24 个，consumer 会使用 240 个。

```cpp
  /// Register requirement for Load and Math WGs
  static constexpr int RegsPerThread =
    (size<0>(TileShape{}) * size<1>(TileShape{}) * sizeof(ElementAccumulator))
    / (NumMMAThreads * sizeof(uint32_t));
  static constexpr bool HeavyRegisterPressure = RegsPerThread >= 208;
  static constexpr uint32_t LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;
  static constexpr uint32_t MmaRegisterRequirement = !HeavyRegisterPressure ? 232 : 240;
```

在 operator()(Params const& params, char* smem_buf)函数内开始运行。

```cpp
    enum class WarpGroupRole {
      Producer = 0,
      Consumer0 = 1,
      Consumer1 = 2
    };
    enum class ProducerWarpRole {
      Mainloop = 0,
      Warp1 = 1,
      Epilogue = 2,
      MainloopAux = 3
    };
```

首先会确定每个 warpgroup 和 warp 的角色。

然后会定义一下 TileScheduler pipeline。对于 sm90 的 PersistentTileSchedulerSm90 都是 StaticPersistentTileScheduler，也就是 IsDynamicPersistent = false。

```cpp
    TileSchedulerPipeline scheduler_pipeline(shared_storage.scheduler.pipeline(), scheduler_pipeline_params);
    TileSchedulerPipelineState scheduler_pipe_consumer_state;

    TileSchedulerThrottlePipeline scheduler_throttle_pipeline(shared_storage.scheduler.throttle_pipeline(), scheduler_throttle_pipeline_params);
    TileSchedulerThrottlePipelineState scheduler_pipe_throttle_consumer_state;
    TileSchedulerThrottlePipelineState scheduler_pipe_throttle_producer_state = cutlass::make_producer_start_state<TileSchedulerThrottlePipeline>();
```

但是 TileSchedulerPipeline 和 TileSchedulerThrottlePipeline 都是 PipelineEmpty，何意味？

然后是创建 Mainloop Load pipeline。并对线程进行分类。只有 warpgroup 是 producer 而且 warp 是 Mainloop 或者 MainloopAux 的才是 producer。

```cpp
    if (warp_group_role == WarpGroupRole::Producer && (producer_warp_role == ProducerWarpRole::Mainloop 
        || producer_warp_role == ProducerWarpRole::MainloopAux)) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
```

然后是创建 Epilogue Load pipeline。

```cpp
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Epilogue) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
```

然后还有 Epilogue Store pipeline。

此外 pingpong 还有 LoadWarpOrderBarrier 和 MathWarpGroupOrderBarrier，暂时不知道干啥的。

```cpp
    typename LoadWarpOrderBarrier::Params params_load_order_barrier;
    params_load_order_barrier.group_id = producer_warp_role == ProducerWarpRole::Mainloop ? 0 : 1;
    params_load_order_barrier.group_size = NumThreadsPerWarp;
    LoadWarpOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, params_load_order_barrier);

    typename MathWarpGroupOrderBarrier::Params params_math_wg_order_barrier;
    // DMA Load WG will not participate in these Ordered Barrier syncs
    params_math_wg_order_barrier.group_id = canonical_warp_group_idx() - static_cast<int>(WarpGroupRole::Consumer0);
    params_math_wg_order_barrier.group_size = NumThreadsPerWarpGroup; // Number of threads / participants in a group
    MathWarpGroupOrderBarrier math_wg_order_barrier(shared_storage.pipelines.math_wg_order, params_math_wg_order_barrier);
```

然后实例化 collective_mainloop 和 collective_epilogue。

```cpp
    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);
```

后面会执行 collective_mainloop.load_init，就是对 A 和 B 进行分区。

下面会对 consumer1 的 pipeline state 进行 advance。没看懂为啥

```cpp
    // Get pipeline stage increments from tensor shapes
    auto k_tile_count = size<3>(gA_mkl);
    auto c_tile_count = CollectiveEpilogue::get_load_pipe_increment(blk_shape);
    auto d_tile_count = CollectiveEpilogue::get_store_pipe_increment(blk_shape);

    TileScheduler scheduler{params.scheduler};
    if constexpr (IsSchedDynamicPersistent) {
      scheduler.set_data_ptr(shared_storage.scheduler.data());
    }

    if (warp_group_role == WarpGroupRole::Consumer1) {

      if constexpr (not IsSchedDynamicPersistent) {
        // Advance 2nd Math WG to the next work tile for the startup
        scheduler.advance_to_next_work();
      }

      // Advance 2nd Math WG pipeline states to the end of 1st Math WG
      mainloop_pipe_consumer_state.advance(k_tile_count);
      epi_load_pipe_consumer_state.advance(c_tile_count);
      epi_store_pipe_producer_state.advance(d_tile_count);
    }
```

然后是初始化 work_tile_info。

```cpp
auto work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});
```

这个 initial_work_tile_info 会调用到 include/cutlass/gemm/kernel/static_tile_scheduler.hpp 里的 initial_work_tile_info，其中 ClusterShape{}在这里是<1,1,1>。

```cpp
  // Returns the initial work tile info that will be computed over
  template <class ClusterShape>
  CUTLASS_DEVICE
  WorkTileInfo
  initial_work_tile_info(ClusterShape cluster_shape) {
    return get_current_work();
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work() const {
    return get_current_work_for_linear_idx(current_work_linear_idx_);
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work_for_linear_idx(uint64_t linear_idx) const {
    if (linear_idx >= scheduler_params.blocks_per_problem_) {
      return WorkTileInfo::invalid_work_tile();
    }

    // Map worker's linear index into the CTA tiled problem shape to the corresponding MNL indices
    uint64_t work_idx_l, remainder;
    scheduler_params.divmod_batch_(work_idx_l, remainder, linear_idx);

    uint64_t blk_per_grid_dim = scheduler_params.divmod_cluster_shape_minor_.divide(remainder);

    auto [work_idx_m, work_idx_n] = Subclass::get_work_idx_m_and_n(blk_per_grid_dim,
                                                         scheduler_params.divmod_cluster_shape_major_,
                                                         scheduler_params.divmod_cluster_shape_minor_,
                                                         scheduler_params.divmod_cluster_blk_major_,
                                                         scheduler_params.log_swizzle_size_,
                                                         scheduler_params.raster_order_);

    return {work_idx_m, work_idx_n, static_cast<int32_t>(work_idx_l), true};
  }
```

在前面初始化 TileScheduler scheduler{params.scheduler};的时候会计算 current_work_linear_idx_的值。对于 persistent kernel，大小是 1-132。但是打印出来不是的？

知道当前 threadblock 的 linear_idx 后，会使用 scheduler_params.divmod_batch_(work_idx_l, remainder, linear_idx);计算当前 linear_idx 在整个 grid 中的位置。

前面知道 divmod_batch_是 GPU 上的快速除法实现。除数是 problem_blocks_m * problem_blocks_n，在这里就是 32*32 = 1024。

work_idx_l 是商，remainder 是余数。

然后会逐层调用到 get_work_idx_m_and_n。这个函数的实现代码如下：

```cpp
  // get work_idx_m, work_idx_n from blk_per_grid_dim while applying swizzle
  static CUTLASS_DEVICE
  cute::tuple<int32_t, int32_t>
  get_work_idx_m_and_n(
      uint64_t blk_per_grid_dim,
      FastDivmodU64Pow2 const& divmod_cluster_shape_major,
      FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
      FastDivmodU64 const& divmod_cluster_blk_major,
      int32_t log_swizzle_size,
      RasterOrder raster_order) {
    auto [cta_m_in_cluster, cta_n_in_cluster, _] = cute::block_id_in_cluster();
    return get_work_idx_m_and_n(
      blk_per_grid_dim,
      divmod_cluster_shape_major,
      divmod_cluster_shape_minor,
      divmod_cluster_blk_major,
      log_swizzle_size,
      raster_order,
      cta_m_in_cluster,
      cta_n_in_cluster
    );
  }

  static CUTLASS_DEVICE
  cute::tuple<int32_t, int32_t>
  get_work_idx_m_and_n(
      uint64_t blk_per_grid_dim,
      FastDivmodU64Pow2 const& divmod_cluster_shape_major,
      FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
      FastDivmodU64 const& divmod_cluster_blk_major,
      int32_t log_swizzle_size,
      RasterOrder raster_order,
      uint64_t cta_m_in_cluster,
      uint64_t cta_n_in_cluster) {

    uint64_t cluster_id, cluster_major_offset = 0, cluster_minor_offset = 0;
    divmod_cluster_shape_major(cluster_id, cluster_major_offset, blk_per_grid_dim);

    if (raster_order == RasterOrder::AlongN) {
      cluster_minor_offset = cta_m_in_cluster;
    }
    else {
      cluster_minor_offset = cta_n_in_cluster;
    }

    uint64_t cluster_idx_minor, cluster_idx_major;

    uint64_t cluster_idx_minor_div_swizzle, extra, offset;

    offset = cluster_id & ((1 << log_swizzle_size) - 1);
    extra = cluster_id >> log_swizzle_size;

    divmod_cluster_blk_major(cluster_idx_minor_div_swizzle, cluster_idx_major, extra);

    cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << log_swizzle_size) + offset;

    auto minor_work_idx = static_cast<int32_t>(cluster_idx_minor * divmod_cluster_shape_minor.divisor +
                                               cluster_minor_offset);
    auto major_work_idx = static_cast<int32_t>(cluster_idx_major * divmod_cluster_shape_major.divisor +
                                               cluster_major_offset);

    if (raster_order == RasterOrder::AlongN) {
      return {minor_work_idx, major_work_idx};
    }
    else {
      return {major_work_idx, minor_work_idx};
    }

  }
```

具体没看太明白，大概就是告诉当前 thread block 需要处理哪一块的数据。通过打印相关变量可以看到 blockIdx.x 始终是 0，blockIdx.y 在 0-131 之间，因为是 alongN 的。work_tile_info.M_idx, work_tile_info.N_idx 在 0-31 范围内变化，work_tile_info.L_idx 始终是 0。

printf("--%d--%d--%d--%d--%d\n", blockIdx.x, blockIdx.y, work_tile_info.M_idx, work_tile_info.N_idx, work_tile_info.L_idx);

后面就是 producer 和 consumer 单独处理了。

在 producer 里会使用 cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();减少寄存器的使用。

producer_warp_role == ProducerWarpRole::Warp1 是用来处理 IsSchedDynamicPersistent 的，因为 sm90 都是 static 的，所以这个 warp 不会执行。

然后对于 producer_warp_role == ProducerWarpRole::Mainloop。有一个 while (work_tile_info.is_valid()) {} 循环。会把 work_tile_info 的 index 转换成矩阵乘里的坐标。scheduler_throttle_pipeline 没看明白是干啥的。

```cpp
      // Mainloop Producer Warp
      if (producer_warp_role == ProducerWarpRole::Mainloop) {
        // Ensure that the prefetched kernel does not touch
        // unflushed global memory prior to this instruction
        cutlass::arch::wait_on_dependent_grids();
        bool do_load_order_arrive = true;
        bool requires_clc_query = true;
        while (work_tile_info.is_valid()) {
          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

          auto k_tile_iter  = cute::make_coord_iterator(shape<3>(gA_mkl));

          if (requires_clc_query) {
            scheduler_throttle_pipeline.producer_acquire(scheduler_pipe_throttle_producer_state);
            scheduler_throttle_pipeline.producer_commit(scheduler_pipe_throttle_producer_state);
            ++scheduler_pipe_throttle_producer_state;
          }
```

然后就是 collective_mainloop.load，不停的加载当前 M，N 在 K 方向上的数据。

```cpp
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
          // Update starting pipeline state for the next tile
          mainloop_pipe_producer_state.advance(k_tile_count);
```

没看懂下面这段是干啥的。

```cpp
          // Signal for the epilogue load warp to begin
          if (do_load_order_arrive) {
            load_order_barrier.arrive();
            do_load_order_arrive = false;
          }
```

当前 tile 计算完成后会获取下一个要计算的 tile。

```cpp
          // Get next work tile
          scheduler.advance_to_next_work();
          work_tile_info = scheduler.get_current_work();
          
  CUTLASS_DEVICE
  void
  advance_to_next_work(uint32_t advance_count = 1) {
    current_work_linear_idx_ += total_grid_size_ * uint64_t(advance_count);
  }
```

然后循环结束后会调用 collective_mainloop.load_tail，就是等待 consumer 的结束。此外如果是 IsSchedDynamicPersistent 还会 scheduler.fetch_next_work。

producer 的 mainloop 就结束了。

然后对于 collective_mainloop.load_auxiliary，好像只有 fp8 才会有这个。应该是加载一些额外的参数。

然后是 producer_warp_role == ProducerWarpRole::Epilogue && collective_epilogue.is_producer_load_needed()，这个是加载矩阵 C 的话，如果没有就不会加载。

对于(warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1)。

首先会通过 cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();增加每个线程寄存器的数量。

然后还是一个 while (work_tile_info.is_valid()) {}循环，循环处理 tile。

下面是 pingpong 的关键：

```cpp
        // Order two Math WG's MMA one after the other, helps hide Epilogue
        math_wg_order_barrier.wait();

        collective_mainloop.mma(
          mainloop_pipeline,
          mainloop_pipe_consumer_state,
          accumulators,
          k_tile_count,
          warp_group_thread_idx,
          shared_storage.tensors.mainloop,
          params.mainloop
        );

        // Cue for next Math WG's MMA to start
        math_wg_order_barrier.arrive();
```

通过 math_wg_order_barrier.wait();让一个 warpgroup 的 wgmma 在另一个 wgmma 之后开始。在 mma 结束后执行 math_wg_order_barrier.arrive();此时下一个 wgmma 会继续执行 wgmma。

```cpp
        // Make sure the math instructions are done and free buffers before entering the epilogue
        collective_mainloop.mma_tail(
          mainloop_pipeline,
          mainloop_pipe_consumer_state,
          k_tile_count
        );
        // Update starting mainloop pipeline state for the next tile
        mainloop_pipe_consumer_state.advance(k_tile_count * NumMmaWarpGroups);

        #ifdef CUTLASS_ENABLE_GDC_FOR_SM90
        if (scheduler.is_last_tile(work_tile_info, NumMmaWarpGroups)) {
          // Hint on an early release of global memory resources.
          // The timing of calling this function only influences performance,
          // not functional correctness.
          cutlass::arch::launch_dependent_grids();

        }
        #endif
```

然后第一个 wgmma 执行 mma_tail 等待所有的 wgmma 完成。

epilogue 也是一样，第一个 wgmma 先执行 epilogue，第二个等前一个完成后执行。

```cpp
        // Order two Math WG's Epilogue one after the other
        math_wg_order_barrier.wait();

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

        // TMA store pipeline wait is only visible to TMA-issuing warp, so for multiple-consumer kernels
        // we need to wait for all TMA stores to complete before issuing consumer order barrier arrives
        // to ensure next math consumer doesn't overwrite smem of in-flight TMA stores of current consumer.
        auto [epi_load_pipe_consumer_state_next_, epi_store_pipe_producer_state_next_] =
        collective_epilogue.store_tail(
          epi_load_pipeline,
          epi_load_pipe_consumer_state_next,
          epi_store_pipeline,
          epi_store_pipe_producer_state_next
        );

        // Update starting load/store pipeline states for the next tile
        // state has already been incremented by 1 tile in collective calls, advance once again for ping pong
        epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next_;
        epi_store_pipe_producer_state = epi_store_pipe_producer_state_next_;
        epi_load_pipe_consumer_state.advance(c_tile_count);
        epi_store_pipe_producer_state.advance(d_tile_count);

        // Cue for next Math WG's Epilogue to start
        math_wg_order_barrier.arrive();
```

最后更新 work_tile_info。

```cpp
        // Get next work tile
        scheduler.advance_to_next_work(NumMmaWarpGroups);
        work_tile_info = scheduler.get_current_work();
```

感觉 pingpong 最关键的有两点：

1. tilescheduler 的调度。
1. math_wg_order_barrier 的实现。

针对第一点，如前所示，persistent kernel 在启动的时候 grid 大小是 sm 的数量，也就是 132。然后在实例化 TileScheduler scheduler{params.scheduler};的时候会创建一个变量 current_work_linear_idx_，这个变量记录了当前 thread block 的 idx。

然后在通过 auto work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});获取每个 thread block 需要处理的初始的 tile 位置。

```cpp
  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work_for_linear_idx(uint64_t linear_idx) const {
    if (linear_idx >= scheduler_params.blocks_per_problem_) {
      return WorkTileInfo::invalid_work_tile();
    }

    // Map worker's linear index into the CTA tiled problem shape to the corresponding MNL indices
    uint64_t work_idx_l, remainder;
    scheduler_params.divmod_batch_(work_idx_l, remainder, linear_idx);
    uint64_t blk_per_grid_dim = scheduler_params.divmod_cluster_shape_minor_.divide(remainder);

    auto [work_idx_m, work_idx_n] = Subclass::get_work_idx_m_and_n(blk_per_grid_dim,
                                                         scheduler_params.divmod_cluster_shape_major_,
                                                         scheduler_params.divmod_cluster_shape_minor_,
                                                         scheduler_params.divmod_cluster_blk_major_,
                                                         scheduler_params.log_swizzle_size_,
                                                         scheduler_params.raster_order_);

    return {work_idx_m, work_idx_n, static_cast<int32_t>(work_idx_l), true};
  }
```

在上面的代码中，linear_idx 就是每个 CTA 的 index，这个值会增加，但是需要始终小于 blocks_per_problem_，也就是矩阵 C 的分块大小。

scheduler_params.divmod_batch_是拿 linear_idx / blocks_per_problem_，得到商和余数。因为 linear_idx 始终小于 blocks_per_problem_，所以 work_idx_l 始终是 0，remainder 始终等于 linear_idx ，为啥这样写呢。

uint64_t blk_per_grid_dim = scheduler_params.divmod_cluster_shape_minor_.divide(remainder);没看明白啥意思，看起来是把 remainder 右移 x 位，其中 x 是根据 cluster dim 获得的。如果 cluster_dim = 1，x 等于 0。所以 blk_per_grid_dim = remainder。

然后会调用 get_work_idx_m_and_n 函数，其中 blk_per_grid_dim 是 reminder，也就是 linear_idx，divmod_cluster_shape_major_的除数是 cluster_n，minor 是 cluster_m，divmod_cluster_blk_major_的除数是 problem_blocks_n / cluster_shape.n()，也就是 problem_blocks_n。log_swizzle_size_ = 1。raster_order_是 alongN。

```cpp
    auto [work_idx_m, work_idx_n] = Subclass::get_work_idx_m_and_n(blk_per_grid_dim,
                                                         scheduler_params.divmod_cluster_shape_major_,
                                                         scheduler_params.divmod_cluster_shape_minor_,
                                                         scheduler_params.divmod_cluster_blk_major_,
                                                         scheduler_params.log_swizzle_size_,
                                                         scheduler_params.raster_order_);
```

具体是

```cpp
  static CUTLASS_DEVICE
  cute::tuple<int32_t, int32_t>
  get_work_idx_m_and_n(
      uint64_t blk_per_grid_dim,
      FastDivmodU64Pow2 const& divmod_cluster_shape_major,
      FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
      FastDivmodU64 const& divmod_cluster_blk_major,
      int32_t log_swizzle_size,
      RasterOrder raster_order,
      uint64_t cta_m_in_cluster,
      uint64_t cta_n_in_cluster) {

    uint64_t cluster_id, cluster_major_offset = 0, cluster_minor_offset = 0;
    divmod_cluster_shape_major(cluster_id, cluster_major_offset, blk_per_grid_dim);
  
      if (threadIdx.x == 0) printf("--%lld--%lld--%lld--%lld--%lld\n", cluster_id, cluster_major_offset, blk_per_grid_dim, cta_m_in_cluster, cta_n_in_cluster);

    if (raster_order == RasterOrder::AlongN) {
      cluster_minor_offset = cta_m_in_cluster;
    }
    else {
      cluster_minor_offset = cta_n_in_cluster;
    }

    uint64_t cluster_idx_minor, cluster_idx_major;

    uint64_t cluster_idx_minor_div_swizzle, extra, offset;

    offset = cluster_id & ((1 << log_swizzle_size) - 1);
    extra = cluster_id >> log_swizzle_size;

    divmod_cluster_blk_major(cluster_idx_minor_div_swizzle, cluster_idx_major, extra);

    cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << log_swizzle_size) + offset;

    auto minor_work_idx = static_cast<int32_t>(cluster_idx_minor * divmod_cluster_shape_minor.divisor +
                                               cluster_minor_offset);
    auto major_work_idx = static_cast<int32_t>(cluster_idx_major * divmod_cluster_shape_major.divisor +
                                               cluster_major_offset);

    if (raster_order == RasterOrder::AlongN) {
      return {minor_work_idx, major_work_idx};
    }
    else {
      return {major_work_idx, minor_work_idx};
    }

  }
```

max_swizzle_size = 1

cta swizzle 的核心是下面的代码，base 是 swizzle block 的大小，0 表示没有 swizzle，1 表示 2*2 大小的 swizzle block，2 表示 4*4 大小的 swizzle block。

```cpp
class CTASwizzle:
    def __init__(self, base: int = 0):
        self._base = base

    def get_grid_dim(self, tiled_shape: Dim):
        tile = 1 << (self._base)
        return (tiled_shape.m * tile, (tiled_shape.n + tile - 1) // tile)

    def cta_swizzle1(self, x, y):
        new_x = x >> self._base
        new_y = (y << self._base) + (x & ((1 << self._base) - 1))
        return int(new_x), int(new_y)
```

首先会对原始的 problem shape 用 get_grid_dim 做一个变化，比如 32*32 大小 base=1，就会变成 64*16 大小。kernel 在实际运行时 grid 的大小就是 64*16，64 是 x 方向。

然后通过 cta_swizzle1 计算实际的 block idx 对应的逻辑上的 problem shape，从而达到 swizzle 的目的。

一个 tile 计算完成后，对应的 threadblock 会前进 total_grid_size_大小，计算该位置需要处理的 tile，从而达到 persistent。

load_order_barrier 和 math_wg_order_barrier 的实现。

这两个变量是 cutlass 中的 cutlass::OrderedSequenceBarrier 类。这个类基于 mbarrier 实现了序列化的同步。

```cpp
template<int SequenceDepth_, int SequenceLength_>
class OrderedSequenceBarrier {
public:
  static constexpr int SequenceDepth = SequenceDepth_;
  static constexpr int SequenceLength = SequenceLength_;
  ...
```

SequenceDepth: 流水线的深度（Stage 数量）。SequenceLength: group 的数量。这个类中创建了 Barrier barrier_[SequenceDepth][SequenceLength];个 mbarrier。通过 PipelineState<SequenceDepth> stage_;来跟踪每个 stage 的状态。

在初始化 stage 的过程中，如果 group_id = 0，phase 会初始化为 1，这样可以直接通过第一个 wait()。

有一个私有函数 get_barrier_for_current_stage，可以根据 group_id 获取对应 stage 的 mbarrier。

```cpp
  CUTLASS_DEVICE
  Barrier& get_barrier_for_current_stage(int group_id) {
    return barrier_ptr_[stage_.index() * Length + group_id];
  }
```

三个成员函数 wait，arrive 和 advance。wait 会等待当前 group_id 的当前 stage 完成。arrive 会对下一个 group_id 执行 arrive，表示线程已经到达了。

```cpp
  // Wait on a stage to be unlocked
  CUTLASS_DEVICE
  void wait() {
    get_barrier_for_current_stage(params_.group_id).wait(stage_.phase());
  }

  // Signal completion of Stage and move to the next stage
  // (group_id) signals to (group_id+1)
  CUTLASS_DEVICE
  void arrive() {
    int signalling_id = (params_.group_id + 1) % Length;
    get_barrier_for_current_stage(signalling_id).arrive();
    ++stage_;
  }

  CUTLASS_DEVICE
  void advance() {
    ++stage_;
  }
```

对于 LoadWarpOrderBarrier，producer == mainloop 是 group_id 0，其余的是 1。因此在 producer 的 mainloop 完成前 epilogue 会一直在 load_order_barrier.wait();处等待。当 mainloop 的 load 完成后会执行 load_order_barrier.arrive();此时 epilogue 会执行 wait 后的代码，也就是对当前 tile 执行 epilogue，而 mainloop 会 load 下一个 tile。感觉这里只需要一个 mbarrier 跟踪 epilogue 就行了。但是为啥只执行一个 arrive 和 wait 呢。

```cpp
    using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;
    
    typename LoadWarpOrderBarrier::Params params_load_order_barrier;
    params_load_order_barrier.group_id = producer_warp_role == ProducerWarpRole::Mainloop ? 0 : 1;
    params_load_order_barrier.group_size = NumThreadsPerWarp;
    LoadWarpOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, params_load_order_barrier);
```

对于 MathWarpGroupOrderBarrier，因为需要跟踪 mainloop 和 epilogue 两个 stage，所以 depth = 2，又因为有两个 warpgroup，所以 group = 2。

```cpp
  // Order Sequence barrier with two stages: one for Mainloop and one for Epilogue
  static constexpr uint32_t StagesPerMathWarpGroup = 2;
  using MathWarpGroupOrderBarrier = cutlass::OrderedSequenceBarrier<
    StagesPerMathWarpGroup, NumMmaWarpGroups>;

    typename MathWarpGroupOrderBarrier::Params params_math_wg_order_barrier;
    // DMA Load WG will not participate in these Ordered Barrier syncs
    params_math_wg_order_barrier.group_id = canonical_warp_group_idx() - static_cast<int>(WarpGroupRole::Consumer0);
    params_math_wg_order_barrier.group_size = NumThreadsPerWarpGroup; // Number of threads / participants in a group
    MathWarpGroupOrderBarrier math_wg_order_barrier(shared_storage.pipelines.math_wg_order, params_math_wg_order_barrier);
```

math_wg_order_barrier 分别在 mma 前后 wait 和 arrive，在 store 前后 wait 和 arrive。

当进行计算时，分别在 mma 和 store 前后有 wait 和 arrive。

```cpp
        // Order two Math WG's MMA one after the other, helps hide Epilogue
        math_wg_order_barrier.wait();

        collective_mainloop.mma(
          mainloop_pipeline,
          mainloop_pipe_consumer_state,
          accumulators,
          k_tile_count,
          warp_group_thread_idx,
          shared_storage.tensors.mainloop,
          params.mainloop
        );

        // Cue for next Math WG's MMA to start
        math_wg_order_barrier.arrive();
        
        // Order two Math WG's Epilogue one after the other
        math_wg_order_barrier.wait();
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
        // Cue for next Math WG's Epilogue to start
        math_wg_order_barrier.arrive();
```

warpgroup1 和 warpgroup2 的 state 的初始状态为 warpgroup1：index=0，phase=1，warpgroup2：index=0，phase=0。

math_wg_order_barrier 的 wait 和 arrive 都是对当前的 index 对应的 mbarrier 进行操作，但是 arrive 处理的 group id 是下一个而且处理完后 stage 会+1，即 index+=1，当 index = stage 时，phase ^=1。

对于第一个 warpgroup1，在 wait 的时候 index=0，phase=1，所以会对 mma 对应的 mbarrier1 执行 wait，因为 phase=1，所以直接通过。

对于第二个 warpgroup2，在 wait 的时候 index=0，phase=0，所以会对 mma 对应的 mbarrier2 执行 wait，因为 phase=0，所以会卡住。

warpgroup1 执行 mma 后会执行 arrive，此时 state1 的 index=0，phase=1，所以会对 mma 对应的 mbarrier2 执行 arrive，arrive 后 mbarrier2 的内部 phase 变成 1，与 state2 的 phase 相位相同，所以 warpgroup2 会通过 wait，执行 mma。

而且 warpgroup1 的 state1 会+=1，此时 state1 的 index=1，因为 index 小于 stage 的大小，所以 phase 还是 1。

warpgroup1 继续执行后面的 wait，因为此时 state1 的 index=1，phase=1，所以会对 store 对应的 mbarrier1 执行 wait，因为相位不同，所以会直接执行 store。此时 wg1 的 store 会和 wg2 的 mma 进行 overlap。

warpgroup1 store 完成后会执行 arrive，此时 state1 的 index=1，phase=1，所以会对 store 对应的 mbarrier2 执行 arrive。执行完后 stage+=1，此时 index=0，phase=0。

warpgroup2 执行完 mma 后会执行 arrive，此时 state2 的 index=0，phase=0，所以会对 mma 对应的 mbarrier1 执行 arrive。执行完后 state2 +=1，此时 state2 的 index=1，小于 stage，phase 还是 0。

warpgroup2 继续执行 store 前的 wait，因为此时 state2 的 index=1，phase=0，所以会对 store 对应的 mbarrier2 执行 wait，如果 wg1 的 store 后的 arrive 还没执行的话就会卡住，等待 wg1 的 arrive 执行完成 wg2 就会执行 store。

warpgroup1 会继续执行下一个 tile 的 mma，因为前面 wg2 已经对 mma 的 mbarrier1 arrive 了，所以可以直接运算。此时 wg1 的 mma 会和 wg2 的 store 进行 overlap。

通过这种方法两个 warpgroup 实现了 pingpong 的状态。

```cpp
// if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) printf("--producer--%d--%d--%d--%d--%d\n", blockIdx.x, blockIdx.y, m_coord, n_coord, k_tile_count);
// if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 128) printf("--consumer1--%d--%d--%d--%d--%d\n", blockIdx.x, blockIdx.y, m_coord, n_coord, k_tile_count);
// if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 256) printf("--consumer2--%d--%d--%d--%d--%d\n", blockIdx.x, blockIdx.y, m_coord, n_coord, k_tile_count);

--producer--0--0--0--0--64
--consumer1 mma--0--0--0--0--64
--producer--0--0--4--2--64
--consumer2 mma--0--0--4--2--64
--consumer1 store--0--0--0--0--64
--producer--0--0--8--4--64
--consumer1 mma--0--0--8--4--64
--consumer2 store--0--0--4--2--64
--producer--0--0--12--6--64
--consumer2 mma--0--0--12--6--64
--consumer1 store--0--0--8--4--64
--producer--0--0--16--8--64
--consumer1 mma--0--0--16--8--64
--consumer2 store--0--0--12--6--64
--producer--0--0--20--10--64
--consumer2 mma--0--0--20--10--64
--consumer1 store--0--0--16--8--64
--producer--0--0--24--12--64
--consumer1 mma--0--0--24--12--64
--consumer2 store--0--0--20--10--64
--producer--0--0--28--14--64
--consumer2 mma--0--0--28--14--64
--consumer1 store--0--0--24--12--64
--consumer2 store--0--0--28--14--64
```

SharedStorage 

```cpp
  // Kernel level shared memory storage
  struct SharedStorage {
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage; // 64, 4 stage * (full_barrier + empty barrier)
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage; // 64
      using MathWarpGroupOrderBarrierStorage = MathWarpGroupOrderBarrierSharedStorage; // 32, (mma + store) * 2 warpgroup

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) MathWarpGroupOrderBarrierStorage math_wg_order;
      alignas(16) typename LoadWarpOrderBarrier::SharedStorage load_order; // 16
    } pipelines;
    
    alignas(16) TileSchedulerStorage scheduler; // 1, align 16

    struct TensorStorage : cute::aligned_struct<128, _1> {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      EpilogueTensorStorage epilogue; // 16896
      MainloopTensorStorage mainloop; // 196608 = (128 + 256) * 64 * 4stage * 2
    } tensors; // 213504
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage); // 214016
```

```cpp
  struct SharedStorage {
    struct TensorStorage {
      using CollectiveStorage = cute::conditional_t<not is_source_supported, CollectiveStorageWithoutC,
                                  cute::conditional_t<ReuseSmemC, CollectiveStorageReuseC, CollectiveStorageWithC>>;
      CollectiveStorage collective; // 16384

      using FusionStorage = typename FusionCallbacks::SharedStorage; // 1
      FusionStorage thread;
    } tensors;

    using PipelineStorage = typename LoadPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
```

sA  = Sw<3,4,3> o smem_ptr[16b](https://unset) o ((_8,_16),(_64,_1),(_1,_4)):((_64,_512),(_1,_0),(_0,_8192))

sC = Sw<2,4,3> o smem_ptr[16b](https://unset) o ((_8,_8),(_32,_1),(_1,_4)):((_32,_256),(_1,_0),(_0,_2048))

sD = Sw<2,4,3> o smem_ptr[16b](https://unset) o ((_8,_8),(_32,_1),(_1,_4)):((_32,_256),(_1,_0),(_0,_2048))

因为是 reuse C 所以大小就是 16384
