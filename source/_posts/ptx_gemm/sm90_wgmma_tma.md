---
title: SM90 GEMM
date: 2026-05-28 10:00:00
tags: [CUDA, GEMM, TMA, WGMMA, SM90, GPU]
categories: [GEMM 优化]
description: 使用 TMA 和 WGMMA 在 Hopper 架构（H200）上实现一个高性能的矩阵乘，涵盖 TMA 描述符创建、mbarrier 流水线同步、WGMMA 描述符与计算、stmatrix epilogue 等完整流程，达到 98% cublas 性能。
---

# sm90_wgmma_tma

本文将使用 TMA 和 WGMMA 在 Hopper 架构（H200）上实现一个高性能的矩阵乘。

## 在 host 侧创建 TMA 描述符

block 的大小如下：

```cpp
    constexpr int blockM = 128;
    constexpr int blockN = 128;
    constexpr int blockK = 64;
```

因为我们使用 128B swizzle，且数据类型是 half，所以 copy box 的大小设置为 8×64。

这里需要注意 shape vector 的顺序。在创建描述符时，连续维度始终在内层，所以对于矩阵 A 来说，因为 A 是 K-major（K 维连续），所以 `gA_shape = [K, M]`；矩阵 B 是 N-major（N 维连续），所以 `gB_shape = [N, K]`。此外，copy box 的大小一般取 swizzle pattern 的大小，且宽度不能超过 swizzle pattern 的宽度，否则会报错。

```cpp
    // create tma desc
    std::vector<int> gA_shape = {K, M}; // stride = {1, K}
    std::vector<int> gB_shape = {N, K}; // stride = {1, N}

    // tma copy box is 64×8 for half
    std::vector<int> sA_shape = {64, 8}; // stride = {1, 64}
    std::vector<int> sB_shape = {64, 8}; // stride = {1, 64}

    auto smem_swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
    auto tmaA_desc = make_gemm_tma_desc<T, 2>(const_cast<half *>(A), gA_shape, sA_shape, smem_swizzle);
    auto tmaB_desc = make_gemm_tma_desc<T, 2>(const_cast<half *>(B), gB_shape, sB_shape, smem_swizzle);
```

创建描述符的函数如下：

```cpp
// make tma desc
template <class T, uint32_t RANK = 2>
CUtensorMap make_gemm_tma_desc(void *gmem_tensor_ptr, std::vector<int> &gmem_shape, std::vector<int> &smem_shape, CUtensorMapSwizzle swizzle)
{
    CUtensorMap tensor_map{};

    uint64_t gmem_prob_shape[5] = {1, 1, 1, 1, 1};
    uint64_t gmem_prob_stride[5] = {0, 0, 0, 0, 0};
    uint32_t smem_box_shape[5] = {1, 1, 1, 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    gmem_prob_shape[0] = gmem_shape[0];
    gmem_prob_stride[0] = 1;
    smem_box_shape[0] = smem_shape[0];

    for (int i = 1; i < RANK; ++i)
    {
        gmem_prob_shape[i] = gmem_shape[i];
        gmem_prob_stride[i] = gmem_prob_stride[i - 1] * gmem_shape[i - 1];
        smem_box_shape[i] = smem_shape[i];
    }

    for (int i = 0; i < RANK; ++i)
    {
        gmem_prob_stride[i] *= sizeof(T);
    }

    auto smem_swizzle = swizzle;
    auto tma_format = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    auto tma_interleave = CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE;
    auto tma_l2Promotion = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    auto tma_oobFill = CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    // Create the tensor descriptor.
    CUresult result = cuTensorMapEncodeTiled(
        &tensor_map, // CUtensorMap *tensorMap,
        tma_format,
        RANK,                 // cuuint32_t tensorRank,
        gmem_tensor_ptr,      // void *globalAddress,
        gmem_prob_shape,      // const cuuint64_t *globalDim,
        gmem_prob_stride + 1, // const cuuint64_t *globalStrides,
        smem_box_shape,       // const cuuint32_t *boxDim,
        smem_box_stride,      // const cuuint32_t *elementStrides,
        tma_interleave,       // Interleave patterns can be used to accelerate loading of values that are less than 4 bytes long.
        smem_swizzle,         // Swizzling can be used to avoid shared memory bank conflicts.
        tma_l2Promotion,      // L2 Promotion can be used to widen the effect of a cache-policy to a wider set of L2 cache lines.
        tma_oobFill           // Any element that is outside of bounds will be set to zero by the TMA transfer.
    );

    return tensor_map;
}
```

描述符创建完成后，计算需要的共享内存大小，并通过 `const __grid_constant__ tmaA tma_a` 的方式把描述符传到 device 上。此外还有其他方法可以将 TMA 描述符传到 device 上，详见 https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-tma-to-transfer-multi-dimensional-arrays。

下面设置 thread block 的大小。因为使用 WGMMA 至少需要一个 warpgroup（128 个线程），所以设置 `num_threads = 128`。此外，流水线设置为 3 级。

```cpp
    constexpr int num_threads = 128; // one warpgroup
    dim3 block(num_threads);
    dim3 grid(num_blockM, num_blockN);
    int smem_size = int(sizeof(T) * ((blockM + blockN) * blockK * numPipe));
    auto kernel_fptr = wgmma_tma_kernel<blockM, blockN, blockK, num_threads, T, TC, numPipe, base,
                                        decltype(tmaA_desc), decltype(tmaB_desc)>;
    cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel_fptr<<<grid, block, smem_size>>>(A, B, C, M, N, K, alpha, beta, tmaA_desc, tmaB_desc);
```

接下来进入 kernel 内部。

## 申请共享内存，创建 wgmma_desc

在 kernel 最开始会获取一些基本信息，比如 `warp_idx` 和 `lane_predicate`。

```cpp
    int warp_idx = canonical_warp_idx_sync();
    int lane_predicate = elect_one_sync();
```

`canonical_warp_idx_sync` 函数基本等价于 `threadIdx.x / 32`，这种写法看起来更专业，貌似对编译器优化也更友好。

```cpp
__device__ int canonical_warp_idx_sync()
{
    return __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
}
```

`lane_predicate` 用于选出一个活跃线程。因为 TMA 等指令只需要一个线程就能启动，也可以写成 `threadIdx.x == 0` 这样的判断，但如果 0 号线程恰好是 inactive 的，就会有潜在问题。`elect_one_sync` 从活跃线程中选出一个，更稳妥。

```cpp
__device__ uint32_t elect_one_sync()
{
    uint32_t pred = 0;
    uint32_t laneid = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "     elect.sync %%rx|%%px, %2;\n"
        "@%%px mov.s32 %1, 1;\n"
        "     mov.s32 %0, %%rx;\n"
        "}\n"
        : "+r"(laneid), "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}
```

下面是 thread block swizzle，具体原理参考其他文章，主要是可以提高 L2 命中率。

```cpp
    // thread block swizzle
    int ox = blockIdx.x;
    int oy = blockIdx.y;
    int y = (oy << base) + (ox & ((1 << base) - 1));
    int x = (ox >> base);
```

接下来申请共享内存。128B swizzle 需要 128 字节对齐，mbarrier 需要 8 字节对齐。

```cpp
    alignas(128) extern __shared__ T shared_memory[];
    T *sA = shared_memory;
    T *sB = shared_memory + bM * bK * NumPipe;

    // init mbarrier
    __shared__ alignas(8) uint64_t producer_mbar[NumPipe];
    __shared__ alignas(8) uint64_t consumer_mbar[NumPipe];
```

其余的就是定义 gC 在当前 CTA 的偏移，以及一些会经常用到的变量。

```cpp
    // auto gA = A + x * bM * K;          // A is K-major
    // auto gB = B + y * bN;              // B is N-major
    auto gC = C + x * bM * N + y * bN; // C is N-major

    constexpr int num_box_row_a = bM / 8;
    constexpr int num_box_col_a = bK / 64;
    constexpr int num_box_row_b = bK / 8;
    constexpr int num_box_col_b = bN / 64;

    constexpr int m_size = bM / 64;
    constexpr int n_size = bN / 128;
    constexpr int k_size = bK / 16;
```

下面创建 WGMMA 描述符并申请累加寄存器。因为 128 个线程要处理 128×128 个数据，所以每个线程需要处理 128 个 half，也就是需要 64 个 32 位寄存器。

```cpp
    uint32_t reg_c[64] = {0};
    auto wgmma_desc_a = make_wgmma_desc(sA, 1 /*swizzle type*/, 64 /*sbo*/, 1 /*lbo*/);   // 128B swizzle
    auto wgmma_desc_b = make_wgmma_desc(sB, 1 /*swizzle type*/, 64 /*sbo*/, 512 /*lbo*/); // 128B swizzle
```

`make_wgmma_desc` 的定义如下：

```cpp
__device__ __forceinline__ GmmaDescriptor make_wgmma_desc(void *smem_ptr, int siwzzle_type, int sbo, int lbo)
{
    GmmaDescriptor desc;
    desc.bitfield.layout_type_ = siwzzle_type;
    desc.bitfield.start_address_ = static_cast<uint16_t>(cast_smem_ptr_to_uint(smem_ptr) >> 4);
    desc.bitfield.base_offset_ = 0;
    desc.bitfield.stride_byte_offset_ = sbo;
    desc.bitfield.leading_byte_offset_ = lbo;
    return desc;
}
```

对于 WGMMA 描述符，主要需要确定 sbo 和 lbo 的值。

对于 A 来说，sA 大小是 128×64，一个 128B swizzle pattern 的大小是 8×64，所以 sA 需要 16×1 个 swizzle pattern。因为 sA 是 K-major，所以 sbo 是 M 维度上的间隔，两个 swizzle pattern 相差 8×64，所以 `sbo = 8×64/8 = 64`。lbo 是 K 维的 stride，默认为 1。

sA 的 layout 如下图所示，图中 sA 已经被 recast 成了 uint128 大小，只截取了前 16 行。

<!-- TODO: 原图 ../images/128BSW_desc.png 缺失，待补充 -->
![k-major 128B swizzle tiling 128×64 占位符](/assets/ptx_gemm/128BSW_desc.png)

对于 B 来说，sB 的大小是 128×64，且 B 是 N-major，所以一个 128B swizzle pattern 的大小是 64×8，sB 需要 2×8 个 swizzle pattern。在这里我们让 swizzle pattern 在 K 维上连续（实测性能会高一点）。因为 B 是 N-major，所以 sbo 是 K 维上的间隔，即 `8×64/8 = 64`。lbo 是 N 维上的间隔，而 WGMMA 在 N 上的大小是 128，包含 2 个 swizzle pattern，所以 `lbo = 8×64×8/8 = 512`。

sB 的 layout 如下图所示，图中数据类型被 recast 成了 uint128，只截取了前 32 列。

<!-- TODO: 原图 ../images/mn_128BSW_desc.png 缺失，待补充 -->
![mn-major 128B swizzle tiling 128×64 占位符](/assets/ptx_gemm/mn_128BSW_desc.png)

然后计算一个 stage 中 TMA 需要传输多少字节，也就是 sA 和 sB 的总大小。

```cpp
constexpr int tma_transaction_bytes = (bM * bK + bN * bK) * sizeof(T);
```

## 初始化 mbarrier

下面开始对 mbarrier 进行初始化。这里每个 stage 都有两个 mbarrier：一个用来跟踪 TMA 的状态（`producer_mbar`），一个用来跟踪 WGMMA 的状态（`consumer_mbar`）。

```cpp
#pragma unroll
    for (int pipe = 0; pipe < NumPipe; ++pipe)
    {
        if (warp_idx == 0 && lane_predicate == 1)
        {
            mbarrier_init(&producer_mbar[pipe], 1);
            mbarrier_init(&consumer_mbar[pipe], 128);
        }
    }
    __syncthreads();
```

mbarrier 的初始化只需要一个线程就可以完成。对于 TMA 来说，只有一个线程参与，所以初始化时 arrive count 是 1；而 WGMMA 需要所有 128 个线程参与，所以 arrive count 是 128。

这里需要加上 `__syncthreads`，防止在一个线程初始化时，其他线程就执行到后面的代码。

## 启动 TMA 进行 prefetch

对于多级流水线来说，开始需要执行 prefetch，提前启动 TMA 获取所有 stage 的数据。

```cpp
    // prefetch
#pragma unroll
    for (int pipe = 0; pipe < NumPipe; ++pipe)
    {
        auto tile_sA = sA + pipe * bM * bK;
        auto tile_sB = sB + pipe * bN * bK;

        if (warp_idx == 0 && lane_predicate == 1)
        {
            arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            tma_copy_a(&tma_a, &producer_mbar[pipe], tile_sA, num_box_row_a, num_box_col_a, k_tile * bK, x * bM);
            tma_copy_b(&tma_b, &producer_mbar[pipe], tile_sB, num_box_row_b, num_box_col_b, y * bN, k_tile * bK);
        }
        ++k_tile;
    }
```

这里 `tile_sA` 和 `tile_sB` 是每个 stage 的共享内存起点。对于跟踪 TMA 指令的 mbarrier，需要执行 `arrive_and_expect_tx`，这个指令不仅说明 TMA 已经 arrive 到执行地点了，而且还可以设置 TMA 需要传输的数据量。

一般的 mbarrier 在 arrive 的线程数量达到设置的 arrive count 后就会切换 phase。而如果设置了 `tma_transaction_bytes`，则需要 arrive count 和传输的数据量同时归零才会切换 phase。

因此在这里执行完 `arrive_and_expect_tx` 后，mbarrier 的 arrive count 已经是 0 了，但 TMA 传输尚未执行，所以 mbarrier 还不会切换 phase。

下面是 `tma_copy_a`。使用 TMA 进行 copy 时，global 的起始地址已经记录在 TMA 描述符里了，所以这里只需要传入共享内存的地址和 global memory 的坐标即可。

```cpp
__device__ __forceinline__ static void
tma_copy_a(void const *desc_ptr, uint64_t *mbar_ptr, half *smem_ptr, int row, int col, int crd0_start, int crd1_start)
{
    auto cache_hint = CacheHintSm90::EVICT_NORMAL;

#pragma unroll
    for (int i = 0; i < row; ++i)
    {
        int crd1 = crd1_start + i * 8; // tma box [8, 64]
#pragma unroll
        for (int j = 0; j < col; ++j)
        {
            int crd0 = crd0_start + j * 64;
            int offset = (j * row + i) * 8 * 64;
            tma_load_2d(desc_ptr, mbar_ptr, static_cast<uint64_t>(cache_hint), smem_ptr + offset, crd0, crd1);
        }
    }
}
```

从前面知道，gA 是 K-major，所以 crd0 就是 K 方向上的坐标，crd1 就是 M 方向的坐标，即 `crd0_start = k_tile * bK`，`crd1_start = x * bM`。

对于 sA，需要 16×1 个 TMA copy box，所以 `row = 16`，`col = 1`。因为 copy box 的大小是 8×64，所以在 row 维度上前进一步 crd1 加 8，在 col 维度上前进一步 crd0 加 64（虽然这里 j 始终等于 0）。

然后计算每个 TMA copy box 对应的 sA offset。这里我们把 TMA copy box 设置为在列方向上连续，所以 `offset = (j * row + i) * 8 * 64`。

确定了 sA 的 offset 和坐标后，就可以直接执行 TMA 指令进行 copy 了。

同理，对于 `tma_copy_b`，因为 gB 是 N-major，所以在 N 方向上是 crd0，在 K 方向上是 crd1，即 `crd0_start = y * bN`，`crd1_start = k_tile * bK`。

```cpp
__device__ __forceinline__ static void
tma_copy_b(void const *desc_ptr, uint64_t *mbar_ptr, half *smem_ptr, int row, int col, int crd0_start, int crd1_start)
{
    auto cache_hint = CacheHintSm90::EVICT_NORMAL;

#pragma unroll
    for (int i = 0; i < row; ++i)
    {
        int crd1 = crd1_start + i * 8;
#pragma unroll
        for (int j = 0; j < col; ++j)
        {
            int crd0 = crd0_start + j * 64;
            int offset = (i + j * row) * 8 * 64;
            tma_load_2d(desc_ptr, mbar_ptr, static_cast<uint64_t>(cache_hint), smem_ptr + offset, crd0, crd1);
        }
    }
}
```

对于 sB，因为 copy box 的大小是 64×8，需要 2×8 个 TMA copy box。这里我们把数据连续方向视为 col，所以 `row = 8`，`col = 2`。在 row 维度上前进一步 crd1 加 8，在 col 维度上前进一步 crd0 加 64。

因为我们把 TMA copy box 设置为在 row 方向上连续，所以 sB 的 `offset = (i + j * row) * 8 * 64`。

此时 TMA 已经在异步拷贝数据了。这里每个 stage 的 sA 和 sB 共用同一个 `producer_mbar[pipe]`。

## 创建 PipelineState

这里创建一个简单的 `PipelineState` 类来同步 mbarrier 的状态。

```cpp
template <uint32_t Stages_>
struct PipelineState
{
    static constexpr uint32_t Stages = Stages_;
    int index_ = 0;
    uint32_t phase_ = 0;
    __device__ void operator++()
    {
        ++index_;
        if (index_ == Stages)
        {
            index_ = 0;
            phase_ ^= 1;
        }
    }
};
```

```cpp
PipelineState<NumPipe> read_state;
PipelineState<NumPipe> write_state;
```

实现参考了 CUTLASS 的代码并根据需要进行了简化。每个 `PipelineState` 都有 index 和 phase：index 用来确定是哪一个 stage，phase 用来判断当前 pipeline 的 phase 和 mbarrier 是否一致。当 index 等于 stage 时，index 归零，切换 phase。

## 启动 wgmma 计算

前面三个 stage 的 TMA 已经开始拷贝了。为了实现异步流水线，我们这里先启动两个异步的 WGMMA 指令进行计算。

```cpp
#pragma unroll
    for (int pipe = 0; pipe < NumPipe - 1; ++pipe)
    {
        mbarrier_wait(&producer_mbar[pipe], read_state.phase_);
        ++read_state;

        warpgroup_arrive();
        gemm(m_size, n_size, k_size, wgmma_desc_a, wgmma_desc_b, reg_c, pipe);
        warpgroup_commit_batch();
        mbarrier_arrive(&consumer_mbar[pipe]);
    }
```

计算前需要等待前两个 stage 的 TMA 拷贝完成，因此使用 `mbarrier_wait` 来判断跟踪 TMA 的 mbarrier 的相位和 `read_state` 的相位是否一致。

从前面知道，当 mbarrier 的 arrive count 和 `tma_transaction_bytes` 同时归零时，mbarrier 内部的 phase 就会切换。

所以当传输没有完成时，mbarrier 的 phase 和 `read_state.phase_` 都是 0，线程就会卡在 `mbarrier_wait` 这里。当 TMA 传输完成时，mbarrier 的 phase 切换为 1，线程就会通过 `mbarrier_wait`。

此时对 `read_state` 自增，`read_state` 内部的 index 变成 1，表示第 0 个 stage 的拷贝已经完成。因为 index 还小于 stage，所以 `read_state` 的 phase 还是 0，这样可以继续判断下一个 stage 的 TMA 是否完成。

当线程通过 `mbarrier_wait` 时，表示对应 pipe 的 TMA 已经完成，数据就绪，就可以执行 WGMMA 了。

WGMMA 的执行比较套路：先执行 `warpgroup_arrive` 确保寄存器处于可用状态，然后调用 `gemm` 函数。

```cpp
__device__ __forceinline__ static void
gemm(int M, int N, int K, GmmaDescriptor &desc_a, GmmaDescriptor &desc_b, uint32_t *reg_c, int stage)
{
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
#pragma unroll
        for (int j = 0; j < N; ++j)
        {
#pragma unroll
            for (int k = 0; k < K; ++k)
            {
                // int is = (j & 1) ? M - 1 - i : i; // Serpentine coordinate
                int offset_a = i * 512 + k * 2 + stage * 1024;
                int offset_b = j * 512 + k * 128 + stage * 1024; // j is always 0
                auto desc_a_offset = gemm_desc_offset(desc_a, offset_a);
                auto desc_b_offset = gemm_desc_offset(desc_b, offset_b);
                fma(desc_a_offset.desc_, desc_b_offset.desc_, reg_c + i * 32);
            }
        }
    }
}
```

这里使用的是 `m64n128k16` 大小的 WGMMA 指令，所以对于 sA，一次可以计算 64×16 大小的数据，也就是需要在 M 维上循环 2 次，在 K 维上循环 4 次。对于 sB，一次可以计算 16×128 大小的数据，也就是在 N 维上一次就行，在 K 维上需要 4 次。

这里需要注意的是，虽然我们在前面设置了 WGMMA 描述符，但每次循环需要修改描述符里的 smem 地址，不然计算的就是同一片数据了。

对于 sA，一共有 2×4 块 WGMMA。M 维第二块和第一块的起始地址差了 `64×8×8/8 = 512`。K 维第二块和第一块相差了 `2×8/8 = 2`。stage 间的地址相差了 `128×64/8 = 1024`。所以 `offset_a = i×512 + k×2 + stage×1024`，这里除以 8 是因为 WGMMA 和 TMA 的指令在计算 offset 相关信息时都是以 128bit 为单位，所以对于 half 类型来说就是 `128/16 = 8`。

对于 sB，一共需要 1×4 块 WGMMA，所以 N 维上的地址相差了 `8×64×8/8 = 512`。K 维上第一块和第二块相差了 `8×64×2/8 = 128`，stage 间相差 `128×64/8 = 1024`，所以 `offset_b = j×512 + k×128 + stage×1024`。

可以通过 `gemm_desc_offset` 设置 offset。

```cpp
__device__ __forceinline__ static GmmaDescriptor gemm_desc_offset(GmmaDescriptor &desc_, int offset)
{
    GmmaDescriptor ret;
    ret.reg32_[0] = desc_.reg32_[0] + uint32_t(offset);
    ret.reg32_[1] = desc_.reg32_[1];
    return ret;
}

union GmmaDescriptor
{
    __device__ constexpr GmmaDescriptor() noexcept : desc_(0) {}
    __device__ constexpr GmmaDescriptor(uint64_t desc) noexcept : desc_(desc) {}
    __device__ constexpr GmmaDescriptor(GmmaDescriptor const &t) noexcept : desc_(t.desc_) {}
    __device__ constexpr GmmaDescriptor(GmmaDescriptor &&t) noexcept : desc_(t.desc_) {}

    __device__ constexpr GmmaDescriptor &operator=(GmmaDescriptor const &t) noexcept
    {
        desc_ = t.desc_;
        return *this;
    }

    __device__ constexpr GmmaDescriptor &operator=(GmmaDescriptor &&t) noexcept
    {
        desc_ = t.desc_;
        return *this;
    }

    uint64_t desc_;
    uint32_t reg32_[2];
    uint16_t reg16_[4];

    struct
    {
        uint16_t start_address_ : 14, : 2;       // 14 bits [0,14), 2 bits unused
        uint16_t leading_byte_offset_ : 14, : 2; // 14 bits [0,14), 2 bits unused
        uint16_t stride_byte_offset_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
        uint8_t : 1, base_offset_ : 3, : 4;      // 1 bit unused, 3 bits [1,4), 4 bits unused
        uint8_t : 6, layout_type_ : 2;           // 6 bits unused, 2 bits [6,8)
    } bitfield;

    // Decay to a uint64_t
    __device__ constexpr
    operator uint64_t() const noexcept { return desc_; }
};
```

因为 `GmmaDescriptor` 是 64 位的联合体，里面包含两个 32 位变量 `reg32_`，且地址在低位，所以对第一个 `reg32_` 加上 offset 就行，一般也不会越界。

WGMMA 描述符确定后，需要确定累加器寄存器的偏移。因为累加器使用的是 half 类型，所以对于一个 `m64n128k16` 大小的 WGMMA 指令，需要 64 个 32 位寄存器。又因为在 M 维上需要计算两次，在 N 维上只需要计算 1 次，所以我们把寄存器根据 M 维的 index 分成两份。

最后按下面这样调用 WGMMA 指令就可以计算了：

```cpp
fma(desc_a_offset.desc_, desc_b_offset.desc_, reg_c + i * 32);
```

WGMMA 指令如下，可以看到需要 A 和 B 的描述符以及 32 个寄存器。

```cpp
// GMMA 64x128x16 F16+=F16*F16
template <int scale_D = 1, int scaleA = 1, int scaleB = 1, int tnspA = 0, int tnspB = 1>
__device__ static void
fma(uint64_t const &desc_a, uint64_t const &desc_b, uint32_t *c)
{
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        " %32,"
        " %33,"
        " p,   %35, %36, %37, %38;\n"
        "}\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]),
          "+r"(c[4]), "+r"(c[5]), "+r"(c[6]), "+r"(c[7]),
          "+r"(c[8]), "+r"(c[9]), "+r"(c[10]), "+r"(c[11]),
          "+r"(c[12]), "+r"(c[13]), "+r"(c[14]), "+r"(c[15]),
          "+r"(c[16]), "+r"(c[17]), "+r"(c[18]), "+r"(c[19]),
          "+r"(c[20]), "+r"(c[21]), "+r"(c[22]), "+r"(c[23]),
          "+r"(c[24]), "+r"(c[25]), "+r"(c[26]), "+r"(c[27]),
          "+r"(c[28]), "+r"(c[29]), "+r"(c[30]), "+r"(c[31])
        : "l"(desc_a),
          "l"(desc_b),
          "r"(int32_t(scale_D)), "n"(int32_t(scaleA)), "n"(int32_t(scaleB)), "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
}
```

除此之外还有 5 个变量需要设置：`scale_D`、`scaleA`、`scaleB`、`tnspA` 和 `tnspB`。

`scale_D` 表示是否需要累加器 D，取值为 0 或 1，等于 0 时按照 `D = A×B` 计算，等于 1 时按照 `D = A×B + D` 计算。

`scaleA`、`scaleB` 表示是否对矩阵 A 和 B 进行取反，取值为 -1 和 1，1 不取反，-1 取反（即 `-1 × A`）。

`tnspA`、`tnspB` 表示是否对矩阵 A 和 B 进行转置操作，取值为 0 和 1，0 表示不转置，1 表示转置。

`scale_D`、`scaleA`、`scaleB` 比较好理解，全部设为 1 即可。

`tnspA` 和 `tnspB` 的设置可以结合 `ldmatrix` 指令来理解。对于 A 来说，因为 A 是 row-major，从共享内存中读取的数据不需要转置，所以 `tnspA = 0`，类似 `ldmatrix` 对 A 的读取方式。对于 B 来说，因为数据在 N 方向上连续，从共享内存读取后需要转置保存到寄存器中，所以 `tnspB = 1`，类似 `ldmatrix.trans` 对 B 的读取方式。

## Mainloop

一个 stage 的 WGMMA 启动计算后，使用 `warpgroup_commit_batch` 把这些 WGMMA 设为一个 group，然后执行 `mbarrier_arrive` 表示线程已经执行了 WGMMA。

之后进入主循环。此时有两个 stage 的 WGMMA 在执行，第 3 个 stage 的 TMA 在拷贝。

```cpp
    while (k_tile <= k_tile_count)
    {
        warpgroup_wait<1>();
        if (warp_idx == 0 && lane_predicate == 1 && k_tile < k_tile_count)
        {
            int pipe = write_state.index_;
            auto tile_sA = sA + pipe * bM * bK;
            auto tile_sB = sB + pipe * bN * bK;

            mbarrier_wait(&consumer_mbar[pipe], write_state.phase_);
            arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            tma_copy_a(&tma_a, &producer_mbar[pipe], tile_sA, num_box_row_a, num_box_col_a, k_tile * bK, x * bM);
            tma_copy_b(&tma_b, &producer_mbar[pipe], tile_sB, num_box_row_b, num_box_col_b, y * bN, k_tile * bK);
            ++write_state;
        }
        int read_pipe = read_state.index_;
        mbarrier_wait(&producer_mbar[read_pipe], read_state.phase_);

        warpgroup_arrive();
        gemm(m_size, n_size, k_size, wgmma_desc_a, wgmma_desc_b, reg_c, read_pipe);
        warpgroup_commit_batch();
        mbarrier_arrive(&consumer_mbar[read_pipe]);
        ++read_state;
        ++k_tile;
    }
```

`warpgroup_wait<1>()` 表示最多允许只有 1 个 group 的 WGMMA 计算没有完成。通过这个限制，我们可以确保第一个 stage 的 WGMMA 计算一定是完成的，第二个可以继续在后台计算。

第一个 stage 的 WGMMA 计算完成后，说明对应的 smem 可以拷贝数据了，因此可以使用 TMA 进行写数据。

此时 `write_state.index_ = 0`，然后调用 `mbarrier_wait` 等待 `consumer_mbar[pipe]` 完成。这里只是象征性地等一下，因为前面在执行完 WGMMA 后已经 arrive 了，所以执行到这里的时候 `consumer_mbar` 已经切换 phase 了。这里主要是通过 `warpgroup_wait<1>()` 来确保 WGMMA 的完成。

然后继续通过 `arrive_and_expect_tx` 设置新的 TMA copy bytes，并拷贝数据到 `write_state` 的 stage 里。

与此同时，通过 `mbarrier_wait(&producer_mbar[read_pipe], read_state.phase_)` 等待第三个 stage 的 TMA 拷贝完成，然后执行对应 stage 上的 WGMMA 计算。

此时第 1 个 stage 的 TMA 在拷贝，第 2、3 个 stage 的 WGMMA 在计算。然后下一个循环会等待第 2 个 WGMMA 完成后用 TMA 写数据，第 1 个 TMA 数据写完后计算，这样就形成了异步流水线。

## Epilogue

最后等待所有的 WGMMA 完成，此时计算结果已经全部累加到寄存器上。

因为累加器是 half 类型的，所以可以通过 `stmatrix` 指令保存到共享内存中，后续直接从共享内存保存到 global memory，可以确保合并访问。

这里没有实现 `beta × C`。

```cpp
warpgroup_wait<0>();
stmatrix_copy(reg_c, shared_memory);
__syncthreads();
```

```cpp
__device__ __forceinline__ static void
stmatrix_copy(uint32_t *frag, half *smem_dst)
{
    int rep = 8;
    int tid = threadIdx.x;
    int warp_idx = canonical_warp_idx_sync();
    int row = tid % 16;
    int col = (tid % 32) / 16;

#pragma unroll
    for (int i = 0; i < rep; ++i)
    {
        uint32_t a0 = frag[i * 4 + 0];
        uint32_t a1 = frag[i * 4 + 1];
        uint32_t a2 = frag[i * 4 + 2];
        uint32_t a3 = frag[i * 4 + 3];

        uint32_t a4 = frag[i * 4 + 0 + 32];
        uint32_t a5 = frag[i * 4 + 1 + 32];
        uint32_t a6 = frag[i * 4 + 2 + 32];
        uint32_t a7 = frag[i * 4 + 3 + 32];
        // int offset = row * 128 + (col + i * 2) * 8 + warp_idx * 16 * 128;
        int local_row = row * 128;
        int local_col_sw = (col + i * 2) ^ (row % 8);
        int offset = local_row + local_col_sw * 8 + warp_idx * 16 * 128;

        stmatrix_atom(a0, a1, a2, a3, smem_dst + offset);
        stmatrix_atom(a4, a5, a6, a7, smem_dst + offset + 8192);
    }
}
```

`stmatrix_copy` 函数如上所示。`stmatrix` 指令是 warp 级别的指令，一个 stmatrix 可以处理 16×16 个 half。一个 warpgroup 有 4 个 warp，可以处理 64 行的数据，所以 M 维上需要 stmatrix 执行两次，N 维上需要执行 8 次。

保存到共享内存的时候可以执行一下 swizzle，防止 bank conflicts。

最终，再把 smem 里的数据保存到 global memory 中。这里重用 shared memory 的目的是在保存到 global memory 时实现合并访问。

```cpp
    int nbN = bN / 8; // float4 = 8 half
    int row_base = threadIdx.x / nbN;
    int col = threadIdx.x % nbN;
    int col_sw = row_base ^ col;
#pragma unroll
    for (int i = 0; i < 16; ++i) // 128 / (128tid / 16)
    {
        int row = row_base + i * 8;
        float4 smem_vec = reinterpret_cast<float4 *>(shared_memory + row * bN)[col_sw];
        reinterpret_cast<float4 *>(gC + row * N)[col] = smem_vec;
    }
```

## 性能测试

矩阵大小 M = N = K = 4096，A 大小 M×K，B 大小 K×N，C 大小 M×N，A、B、C 都是 row-major。测试平台 H200，TFLOPS = 989e12。

cublas 实现如下。计算方式按照 $C^T = B^T \times A^T$ 进行，这样 A、B、C 都是 column-major，性能更好。

```cpp
    cublasHandle_t handle;
    cublasCreate(&handle);
    const __half alpha = 1.0f, beta = 0.0f;
    // C is column-major, CT = BT × AT
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                 reinterpret_cast<__half *>(d_B.data().get()), CUDA_R_16F, N,
                 reinterpret_cast<__half *>(d_A.data().get()), CUDA_R_16F, K,
                 &beta,
                 reinterpret_cast<__half *>(d_C1.data().get()), CUDA_R_16F, N,
                 CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

测试结果：

```cpp
cublas time = 0.188571 ms, TFLPOS = 728.845011, mfu = 0.736951
 wgmma time = 0.191733 ms, TFLPOS = 716.823129, mfu = 0.724796
```

可以看到，实现的 GEMM 可以达到 98% 的 cublas 性能。
