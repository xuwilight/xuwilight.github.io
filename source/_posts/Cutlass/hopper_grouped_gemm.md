---
title: Hopper Grouped GEMM
date: 2025-08-15 12:00:00
tags: [CUTLASS, Hopper, Grouped GEMM, GPU]
categories: [Cutlass 学习笔记]
description: 常用于 moe。假如有 e 个专家，input 的 shape 是[m * e, k]，每个专家的权重是[n, k]，总的权重是 [n * e, k]，所以 x*wT 后的输出 shape 是[m * e, n] ```cpp class "cutlass::gemm::kernel::GemmUniversal<cutlass::gemm::GroupProblemShape<cute::...
---

常用于 moe。假如有 e 个专家，input 的 shape 是[m * e, k]，每个专家的权重是[n, k]，总的权重是 [n * e, k]，所以 x*wT 后的输出 shape 是[m * e, n]

```cpp
class "cutlass::gemm::kernel::GemmUniversal<cutlass::gemm::GroupProblemShape<cute::tuple<int32_t, int32_t, int32_t>>, cutlass::gemm::collective::CollectiveMma<cutlass::gemm::MainloopSm90ArrayTmaGmmaWarpSpecialized<3, cute::tuple<cute::_2, cute::_1, cute::_1>, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>, cute::tuple<cute::_128, cute::_128, cute::_128>, cutlass::bfloat16_t, cute::tuple<int64_t, cute::C<1>, cute::C<0>> *, cutlass::bfloat16_t, cute::tuple<int64_t, cute::C<1>, cute::C<0>> *, cute::TiledMMA<cute::MMA_Atom<cute::SM90::GMMA::MMA_64x128x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::K, cute::SM90::GMMA::Major::K, cute::SM90::GMMA::ScaleIn::One, cute::SM90::GMMA::ScaleIn::One>>, cute::Layout<cute::tuple<cute::_1, cute::_1, cute::_1>, cute::tuple<cute::C<0>, cute::C<0>, cute::C<0>>>, cute::tuple<cute::Underscore, cute::Underscore, cute::Underscore>>, cute::SM90_TMA_LOAD, cute::ComposedLayout<cute::Swizzle<3, 4, 3>, cute::smem_ptr_flag_bits<16>, cute::Layout<cute::tuple<cute::_8, cute::_64>, cute::tuple<cute::_64, cute::_1>>>, void, cute::identity, cute::SM90_TMA_LOAD_MULTICAST, cute::ComposedLayout<cute::Swizzle<3, 4, 3>, cute::smem_ptr_flag_bits<16>, cute::Layout<cute::tuple<cute::_8, cute::_64>, cute::tuple<cute::_64, cute::_1>>>, void, cute::identity>, cutlass::epilogue::collective::CollectiveEpilogue<cutlass::epilogue::Sm90PtrArrayTmaWarpSpecialized<4, 2, 16, true, false, 2>, cute::tuple<cute::_128, cute::_128, cute::_128>, cute::tuple<cute::C<64>, cute::C<32>>, cutlass::bfloat16_t, cute::tuple<int64_t, cute::C<1>, cute::C<0>> *, cutlass::bfloat16_t, cute::tuple<int64_t, cute::C<1>, cute::C<0>> *, cutlass::epilogue::fusion::FusionCallbacks<cutlass::epilogue::Sm90PtrArrayTmaWarpSpecialized<4, 2, 16, true, false, 2>, cutlass::epilogue::fusion::LinearCombination<cutlass::bfloat16_t, float, cutlass::bfloat16_t, float, cutlass::FloatRoundStyle::round_to_nearest>, cute::tuple<cute::_128, cute::_128, cute::_128>, cute::tuple<cute::C<64>, cute::C<32>>>, cute::SM90_TMA_LOAD, cute::ComposedLayout<cute::Swizzle<2, 4, 3>, cute::smem_ptr_flag_bits<16>, cute::Layout<cute::tuple<cute::C<8>, cute::C<32>>, cute::tuple<cute::_32, cute::_1>>>, cute::SM75_U32x4_LDSM_N, cute::SM90_TMA_STORE, cute::ComposedLayout<cute::Swizzle<2, 4, 3>, cute::smem_ptr_flag_bits<16>, cute::Layout<cute::tuple<cute::C<8>, cute::C<32>>, cute::tuple<cute::_32, cute::_1>>>, cute::SM90_U32x4_STSM_N, cute::Copy_Atom<cute::SM90_U32x4_STSM_N, cutlass::half_t>, void>, void, void>"
```

模板定义，主要通过 cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong 和 cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong 来实例化 grouped gemm。

```cpp
// Different configs for pingpong/cooperative
struct CooperativeConfig {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using TileShape           = Shape<_256,_128,_128>;
  using ClusterShape        = Shape<_1,_2,_1>;
};

struct PingpongConfig {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape           = Shape<_128,_128,_128>;
  using ClusterShape        = Shape<_2,_1,_1>;
};

template <typename ScheduleConfig>
struct GemmGivenSchedule {
  using TileShape           = typename ScheduleConfig::TileShape;                   // Threadblock-level tile size
  using ClusterShape        = typename ScheduleConfig::ClusterShape;                // Shape of the threadblocks in a cluster
  using KernelSchedule      = typename ScheduleConfig::KernelSchedule;              // Kernel to launch
  using EpilogueSchedule    = typename ScheduleConfig::EpilogueSchedule;            // Epilogue to launch

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentC,
    ElementC, LayoutC *, AlignmentC,
    EpilogueSchedule,
    cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA *, AlignmentA,
    ElementB, LayoutB *, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

传入参数。

problem_sizes 是{m, n, k}数组。

std::vector<ElementA *> ptr_A_host(options.groups);

using StrideA = typename Gemm::GemmKernel::InternalStrideA;

stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));

```cpp
    fusion_args.alpha = options.alpha;
    fusion_args.beta = options.beta;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    // Single alpha and beta for all groups
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
  
  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };
  
    arguments = typename GemmT::Arguments {
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {options.groups, problem_sizes.get(), options.problem_sizes_host.data()},
      {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
      {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
      kernel_hw_info
    };
```

测试结果 

```cpp
// if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) printf("--producer--%d--%d--%d--%d--%d--%d\n", curr_batch, blockIdx.x, blockIdx.y, m_coord, n_coord, work_k_tile_count);
// if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 128) printf("--consumer1--%d--%d--%d--%d--%d--%d\n", curr_batch, blockIdx.x, blockIdx.y, m_coord, n_coord, work_k_tile_count);
// if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 256) printf("--consumer2--%d--%d--%d--%d--%d--%d\n", curr_batch, blockIdx.x, blockIdx.y, m_coord, n_coord, work_k_tile_count);
TileShape = cute::Shape<_128, _128, _128>;

blockM = 46, 44, 44, 46
blockN = 16, 16, 16, 16
blockK = 22, 22, 22, 22
num_tiles = 736, 704, 704, 736
sm_count = 132

expect_y: torch.Size([22775, 2048]) expect_dx:torch.Size([22775, 2816]) expect_dw:torch.Size([8192, 2816])
grouped_input: input_m_list:[5827, 5557, 5512, 5879] k=2816 t=False grouped_weight: input_n_list:[2048, 2048, 2048, 2048] k=2816 t=True
--producer--0--0--0--0--0--22   // group idx = 0, tile = 0
--consumer1--0--0--0--0--0--22
--producer--0--0--0--40--2--22  // group idx = 0, tile = 132 * 1 = 2 * 46 + 40
--consumer2--0--0--0--40--2--22
--producer--0--0--0--34--5--22  // group idx = 0, tile = 132 * 2 = 5 * 46 + 34
--consumer1--0--0--0--34--5--22
--producer--0--0--0--28--8--22  // group idx = 0, tile = 132 * 3 = 8 * 46 + 28
--consumer2--0--0--0--28--8--22
--producer--0--0--0--22--11--22  // group idx = 0, tile = 132 * 4 = 11 * 46 + 22
--consumer1--0--0--0--22--11--22
--producer--0--0--0--16--14--22  // group idx = 0, tile = 132 * 5 = 14 * 46 + 16
--consumer2--0--0--0--16--14--22
--consumer1--1--0--0--12--1--22
--producer--1--0--0--12--1--22  // group idx = 1, tile = 132 * 6 = 736 + 1 * 44 + 12
--producer--1--0--0--12--4--22  // group idx = 1, tile = 132 * 7 = 736 + 4 * 44 + 12
--consumer2--1--0--0--12--4--22
--producer--1--0--0--12--7--22
--consumer1--1--0--0--12--7--22
--producer--1--0--0--12--10--22
--consumer2--1--0--0--12--10--22
--producer--1--0--0--12--13--22
--consumer1--1--0--0--12--13--22
--consumer2--2--0--0--12--0--22
--producer--2--0--0--12--0--22
--producer--2--0--0--12--3--22  // group idx = 2, tile = 132 * 12 = 736 + 704 + 3 * 44 + 12
--consumer1--2--0--0--12--3--22
--producer--2--0--0--12--6--22
--consumer2--2--0--0--12--6--22
--producer--2--0--0--12--9--22
--consumer1--2--0--0--12--9--22
--producer--2--0--0--12--12--22
--consumer2--2--0--0--12--12--22
--producer--2--0--0--12--15--22
--consumer1--2--0--0--12--15--22
--consumer2--3--0--0--8--2--22
--producer--3--0--0--8--2--22
--producer--3--0--0--2--5--22
--consumer1--3--0--0--2--5--22
--producer--3--0--0--42--7--22
--consumer2--3--0--0--42--7--22
--producer--3--0--0--36--10--22
--consumer1--3--0--0--36--10--22
--producer--3--0--0--30--13--22
--consumer2--3--0--0--30--13--22
```
