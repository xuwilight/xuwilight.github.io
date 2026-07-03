---
title: Cutlass 学习笔记（七）SM90 GEMM TMA
date: 2025-08-21 12:00:00
tags: [CUTLASS, GEMM, TMA, SM90]
categories: [Cutlass 学习笔记,Cutlass]
description: sm90_gemm_tma 函数使用 tma 异步加载数据，是一个 multistage 函数。 代码如下所示，基本功能是使用 tma 加载数据，使用 wgmma 进行计算。没有进行 warp specification，也不是 persistent kernel。只是在一个 threadblock 中对不同的 blockK 的加载和计算进行 overlap。
---

sm90_gemm_tma 函数使用 tma 异步加载数据，是一个 multistage 函数。

代码如下所示，基本功能是使用 tma 加载数据，使用 wgmma 进行计算。没有进行 warp specification，也不是 persistent kernel。只是在一个 threadblock 中对不同的 blockK 的加载和计算进行 overlap。

```cpp
/**
 * just use tma to load gemm. In a threadblock, the loading and gemm are overlapped in the blockK dimension.
 * nvcc sm90_tma_gemm.cu -arch=sm_90a -I ../../../include/ -I ../../../tools/util/include/ -lcuda -lcublas -o sm90_tma_gemm --expt-relaxed-constexpr && ./sm90_tma_gemm
 */

template <class TA, class TB, class TC>
void sm90_tma_gemm(int M, int N, int K, TC alpha, TA const *A, int lda, TB const *B, int ldb, TC beta, TC *C, int ldc)
{
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using ArchTag = cutlass::arch::Sm90;
    using OpClass = cutlass::arch::OpClassTensorOp;
    using TileShape = Shape<_128, _128, _64>;
    using ClusterShape = Shape<_1, _1, _1>;

    constexpr int stage = 3;
    constexpr int mma_stage = 1;
    using DispatchPolicy = cutlass::gemm::MainloopSm90TmaGmma<stage, ClusterShape, mma_stage>; // KernelTma
    using TMACOPY = SM90_TMA_LOAD;
    using AtomLayoutMNK = Layout<Shape<_1, _1, _1>>;

    // using TiledMMA = decltype(make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{}));
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
        DispatchPolicy,
        TileShape, TA, Stride_A, TB, Stride_B,
        TiledMMA,
        TMACOPY, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,
        TMACOPY, SmemLayoutAtomB, SmemCopyAtomB, cute::identity>;

    static constexpr int FragmentSize = 4; // 128bit / 32bit
    using ThreadOp = cutlass::epilogue::thread::LinearCombination<TC, FragmentSize>;
    using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<TC, Stride_C, Stride_C, ThreadOp, cutlass::gemm::EpilogueDefault>;

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

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    gemm.can_implement(arguments);
    gemm.initialize(arguments, workspace.get());
    gemm.run();
}
```

TMA gemm 的 CollectiveMma 的实例化如下，当 DispatchPolicy 是 MainloopSm90TmaGmma 时就会实例化 tma gemm。

```cpp
    constexpr int stage = 3;
    using DispatchPolicy = cutlass::gemm::MainloopSm90TmaGmma<stage>;
    using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
        DispatchPolicy,
        TileShape,
        TA, Stride_A,
        TB, Stride_B,
        TiledMMA,
        TMACOPY,
        SmemLayoutAtomA,
        SmemCopyAtomA,
        cute::identity,
        TMACOPY,
        SmemLayoutAtomB,
        SmemCopyAtomB,
        cute::identity>;
```

TiledMMA 可以直接自己指定 WGMMA Atom，也可以使用 ss_op_selector 根据 shape 的大小进行选择。因为这里 tiled shape 是<128,128,64>，所以会选择 64*128*16 大小的 wgmma，测试下来比 64*64*16 的 wgmma 快一些。

```cpp
    using AtomLayoutMNK = Layout<Shape<_1, _1, _1>>;
    // using TiledMMA = decltype(make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{}));
    using TiledMMA = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
                                                       TA, TB, TC, TileShape, GMMA::Major::K, GMMA::Major::K>(),
                                                   AtomLayoutMNK{}));
```

using TMACOPY = SM90_TMA_LOAD; TMACOPY 就是 SM90_TMA_LOAD，表示使用 cp.async.bulk.tensor 进行加载数据。

using SmemLayoutAtomA = GMMA::Layout_K_SW128_Atom<TA>; 共享内存的 layout 使用 128B swizzle 的 pattern。也可以根据 ss_smem_selector 来选择。

SmemCopyAtomA 是设置从 smem 拷贝到 rmem 的指令，如 ldmatrix。因为这里 wgmma 直接从 smem 中读取数据，所以就不需要设置了。

然后是 CollectiveEpilogue 的选择。

cutlass 中定义了几个与 TMA 相关的 epilogue schedule，但是只能适用于 WS 的 GEMM。所以不是 ws 的 tma gemm 只能使用 NoSmemWarpSpecialized schedule，也就是 default epilogue。应该也可以自己定义。

```cpp
    static constexpr int FragmentSize = 4; // 128bit / 32bit
    using ThreadOp = cutlass::epilogue::thread::LinearCombination<TC, FragmentSize>;
    using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<TC, Stride_C, Stride_C, ThreadOp, cutlass::gemm::EpilogueDefault>;
```

然后组成 GemmUniversal 和 Gemm。

```cpp
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

最后设置参数，运行。

```cpp
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {A, stride_A, B, stride_B},
        {{1.0f, 0.0f}, C, stride_C, C, stride_C},
        kernel_hw_info};

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    gemm.can_implement(arguments);
    gemm.initialize(arguments, workspace.get());
    gemm.run();
```

下面从 GemmUniversalAdapter 开始分析。

GemmUniversalAdapter 的模板参数是 GemmKernel。前面一堆类型别名，然后有一个私有变量 Params params_;

can_implement 调用的是 GemmKernel::can_implement(args)，用于判断能否实现。

如果不是 kGemmSplitKParallel，get_workspace_size 返回的是 0。

get_grid_shape 函数返回 kernel launch grid shape。调用的是 GemmKernel::get_grid_shape(params)，GemmKernel 调用的是 TileScheduler::get_tiled_cta_shape_mnl(problem_shape_MNKL, tile_shape, cluster_shape);

因为 sm90 的 TileScheduler 都会实例化成 PersistentTileSchedulerSm90，所以最后调用的是 StaticPersistentTileScheduler 类的 get_tiled_cta_shape_mnl，在这里会计算 cta_m 和 cta_n，然后用 Params 类的 get_tiled_cta_shape_mnl 计算。

Params 类又是 PersistentTileSchedulerSm90Params，如果没有 cluster_shape，返回的是 m，n 和一个 batch 维度，一般是 1，不知道干啥的。

然后是 maximum_active_blocks 函数，用于计算每个 SM 上最大的 thread block 数量。当使用的共享内存大于静态共享内存时，需要动态的申请。

然后是 initialize 函数，用于根据 arguments 初始化 params。初始化的时候 initialize_workspace 直接返回的 success。然后通过 GemmKernel::to_underlying_arguments(args, workspace);将参数转化为 params。这个 to_underlying_arguments 会逐层调用 GemmKernel, CollectiveMMA 和 CollectiveEpilogue 的 to_underlying_arguments。然后根据使用的 smem 的大小设置 smem 的空间。

update 函数跟 initialize 函数差不多。

最后是 run 函数。

get_block_shape 返回一个 dim3，记录一个 threadblock 有多少个线程，其实就是 tiledmma 的线程数。

get_grid_shape 会调用到 GemmKernel 的 get_grid_shape，然后会调用 TileScheduler::get_tiled_cta_shape_mnl，因为 TileScheduler 是 PersistentTileSchedulerSm90，所以最终会调用到 StaticPersistentTileScheduler 中的 get_tiled_cta_shape_mnl。

貌似没用到 threadblock swizzle？应该是跟 kernel 的实现方式有关系，如果是 persistent kernel 就会用到。

参数什么的设置完成后就会调用 launch 函数启动 kernel。

```python
/// Generic CUTLASS kernel template.
template <typename Operator>
CUTLASS_GLOBAL
#ifdef __CUDACC__
// Enclosing this in __CUDACC__ suppresses MSVC warnings.
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
#endif // __CUDACC__
void device_kernel(CUTLASS_GRID_CONSTANT typename Operator::Params const params)
{
  // Dynamic shared memory base pointer
  extern __shared__ char smem[];
  Operator op;
  op(params, smem);
  cutlass::arch::synclog_print();

}
```

因为 gemm tma 这个 kernel 不是 warp specification 的，所以逻辑比较简单，和 cute 中 gemm tma 代码逻辑相似。把矩阵根据 blockIdx 进行分块后调用 collective_mma 进行循环计算。

```python
    // Perform the collective scoped MMA
    CollectiveMainloop collective_mma;
    collective_mma(
      gA, params.mainloop.tma_load_a,
      gB, params.mainloop.tma_load_b,
      accumulators,
      k_tile_iter, k_tile_count,
      thread_idx,
      block_rank_in_cluster,
      smem_buf,
      params.mainloop
    );
```

传进去的参数：gA 和 gB 是当前 threadblock 需要处理的 A 和 B 的大小。tma_load_a 和 tma_load_b 是分别加载 A 和 B 的 tiledcopy。累加器是每个线程对应的矩阵 C 的寄存器。

在 sm90_mma_tma_gmma_ss 中会先根据 stage 的数量启动 tma 进行异步拷贝。然后在启动 K_PIPE_MMAS 个 wgmma 作为 prologue。K_PIPE_MMAS 需要小于 Stage。

然后主循环中有两种实现方式，假设 stage=3，mma_async_num = 2，

一种是先等待第 3 个 stage 的 tma 加载完成，然后计算第三个 stage 的 wgmma，此时有 3 个 wgmma 在 in-flight，然后等待第 1 个 stage 的 wgmma 完成，然后启动 tma 加载数据到第 1 个 stage。

cutlass 默认是第一种。但是这种有个问题就是在主循环中提交 wgmma 时已经没有运行的 tma 了，然后下一个循环开始又要等到上一个提交的 tma 完成，等于是串行的了。而且如果 stage=3，mma_stage 必须小于 stage -1，也不知道为啥。

另一种是先等待第一个 wgmma 完成，然后启动 tma 加载数据到第一个 stage，然后启动第三个 stage 的 wgmma。修改后性能一样，但是用 cute 实现的就会快很多，不知道为啥。

mainloop 计算完成后直接进入 epilogue 部分保存计算结果。

性能测试：

cublas time = 0.180211 ms, TFLPOS = 762.654658, mfu = 0.771137

mma time = 0.282080 ms, TFLPOS = 487.234646, mfu = 0.492654
