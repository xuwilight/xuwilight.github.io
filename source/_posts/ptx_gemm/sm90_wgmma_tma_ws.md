---
title: SM90 GEMM —— Warp Specialization
date: 2026-06-01 10:00:00
tags: [CUDA, GEMM, TMA, WGMMA, SM90, GPU]
categories: [GEMM 优化]
description: 本文将使用 Warp Specialization（WS）技术，结合 TMA 和 WGMMA 在 Hopper 架构（H200）上实现一个高性能的矩阵乘。

---

# sm90_wgmma_tma_ws

Warp Specialization 是 Hopper 架构引入的一种编程模式，允许不同的 warpgroup 执行不同的任务。在本文中，我们将使用两个 warpgroup：一个作为 Producer 负责数据加载，另一个作为 Consumer 负责矩阵计算。

## 核心思想

在 `sm90_wgmma_tma` 中，TMA 加载和 WGMMA 计算在同一个 warpgroup 中通过流水线交替执行。而在 Warp Specialization 模式下，我们将这两个任务分离到不同的 warpgroup 中：

- **Producer Warpgroup**：专门负责 TMA 数据加载
- **Consumer Warpgroup**：专门负责 WGMMA 矩阵计算

这种分离有以下优势：

1. **更好的并行性**：加载和计算可以真正并行执行，无需交替等待
2. **更高的寄存器利用率**：Producer 可以释放寄存器给 Consumer 使用
3. **更大的 tile 尺寸**：使用更大的 N 维度（256 vs 128），提高计算强度

## Host 侧配置

block 的大小如下：

```cpp
    constexpr int blockM = 128;
    constexpr int blockN = 256;
    constexpr int blockK = 64;
    constexpr int numPipe = 4;
```

与 `sm90_wgmma_tma` 相比，blockN 从 128 增加到 256，流水线深度从 3 增加到 4。

TMA 描述符的创建有所不同。对于 A 矩阵，copy box 大小设置为 64×128：

```cpp
    // create tma desc
    std::vector<int> gA_shape = {K, M}; // stride = {1, K}
    std::vector<int> gB_shape = {N, K}; // stride = {1, N}

    // tma copy box
    std::vector<int> sA_shape = {64, 128}; // stride = {1, 64}
    std::vector<int> sB_shape = {64, 64};  // stride = {1, 64}

    auto smem_swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
    auto tmaA_desc = make_gemm_tma_desc<T, 2>(const_cast<half *>(A), gA_shape, sA_shape, smem_swizzle);
    auto tmaB_desc = make_gemm_tma_desc<T, 2>(const_cast<half *>(B), gB_shape, sB_shape, smem_swizzle);
```

线程数设置为 256，即两个 warpgroup：

```cpp
    constexpr int num_threads = 256; // two warpgroups
    dim3 block(num_threads);
    dim3 grid(num_blockM, num_blockN);
    int smem_size = int(sizeof(T) * ((blockM + blockN) * blockK * numPipe));
```

## Warpgroup 角色划分

在 kernel 中，首先获取 warpgroup 的索引并划分角色：

```cpp
    int warp_idx = canonical_warp_idx_sync();
    int lane_predicate = elect_one_sync();
    int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);

    enum class WarpGroupRole
    {
        Producer = 0,
        Consumer = 1,
    };
    auto warp_group_role = WarpGroupRole(warpgroup_idx);
```

`warpgroup_idx` 通过 `threadIdx.x / 128` 计算：第一个 128 个线程为 Producer（`warpgroup_idx = 0`），第二个 128 个线程为 Consumer（`warpgroup_idx = 1`）。

## 寄存器动态分配

Warp Specialization 的一个关键优化是寄存器的动态分配。Producer 不需要大量寄存器，可以释放给 Consumer 使用。

```cpp
    static constexpr uint32_t LoadRegisterRequirement = 40;
    static constexpr uint32_t MmaRegisterRequirement = 232;
```

Producer 释放寄存器：

```cpp
    if (warp_group_role == WarpGroupRole::Producer)
    {
        warpgroup_reg_dealloc<LoadRegisterRequirement>();
        // ...
    }
```

Consumer 分配更多寄存器：

```cpp
    else if (warp_group_role == WarpGroupRole::Consumer)
    {
        warpgroup_reg_alloc<MmaRegisterRequirement>();
        // ...
    }
```

`setmaxnreg` 指令的定义如下：

```cpp
template <uint32_t RegCount>
__device__ void warpgroup_reg_dealloc()
{
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
__device__ void warpgroup_reg_alloc()
{
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}
```

`setmaxnreg.dec` 减少该 warpgroup 可用的最大寄存器数量，`setmaxnreg.inc` 增加可用寄存器数量。通过这种方式，Consumer 可以获得更多寄存器来存储更大的累加器。

## WGMMA 指令配置

由于 blockN 增加到 256，WGMMA 指令从 `m64n128k16` 升级到 `m64n256k16`：

```cpp
// GMMA 64x256x16 F16+=F16*F16
template <int scale_D = 1, int scaleA = 1, int scaleB = 1, int tnspA = 0, int tnspB = 1>
__device__ static void
fma(uint64_t const &desc_a, uint64_t const &desc_b, uint32_t *c)
{
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f16.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " p,    %67,  %68,  %69,  %70;\n"
        "}\n"
        // ... 输入输出操作数
    );
}
```

`m64n256k16` 指令需要 64 个 32 位寄存器来存储累加器（128 个 half）。因为 blockM = 128，需要在 M 维上执行 2 次，所以总共需要 128 个 32 位寄存器：

```cpp
    uint32_t reg_c[128] = {0};
```

WGMMA 描述符的设置：

```cpp
    auto wgmma_desc_a = make_wgmma_desc(sA, 1 /*swizzle type*/, 64 /*sbo*/, 1 /*lbo*/);
    auto wgmma_desc_b = make_wgmma_desc(sB, 1 /*swizzle type*/, 64 /*sbo*/, 512 /*lbo*/);
```

注意 B 矩阵的 `lbo` 仍然是 512，因为 `m64n256k16` 在 N 维上一次处理 256 个元素，包含 4 个 swizzle pattern（每个 64×8），所以 `lbo = 64×8×8/8 = 512`。

`gemm` 函数中的 offset 计算：

```cpp
    int offset_a = i * 512 + k * 2 + stage * 1024;
    int offset_b = j * 512 + k * 128 + stage * 2048;
```

B 矩阵的 stage offset 从 1024 变为 2048，因为 sB 的大小从 128×64 变为 256×64。

## TMA 拷贝函数

由于 TMA copy box 大小变化，拷贝函数也需要相应调整。

对于 A 矩阵，copy box 为 64×128：

```cpp
__device__ __forceinline__ static void
tma_copy_a(void const *desc_ptr, uint64_t *mbar_ptr, half *smem_ptr, int row, int col, int crd0_start, int crd1_start)
{
    auto cache_hint = CacheHintSm90::EVICT_NORMAL;

#pragma unroll
    for (int i = 0; i < row; ++i)
    {
        int crd1 = crd1_start + i * 128; // tma box [128, 64]
#pragma unroll
        for (int j = 0; j < col; ++j)
        {
            int crd0 = crd0_start + j * 64;
            int offset = (j * row + i) * 128 * 64;
            tma_load_2d(desc_ptr, mbar_ptr, static_cast<uint64_t>(cache_hint), smem_ptr + offset, crd0, crd1);
        }
    }
}
```

对于 sA（128×64），需要 1×1 个 copy box，`num_box_row_a = bM / 128 = 1`，`num_box_col_a = bK / 64 = 1`。

对于 B 矩阵，copy box 为 64×64：

```cpp
__device__ __forceinline__ static void
tma_copy_b(void const *desc_ptr, uint64_t *mbar_ptr, half *smem_ptr, int row, int col, int crd0_start, int crd1_start)
{
    auto cache_hint = CacheHintSm90::EVICT_NORMAL;

#pragma unroll
    for (int i = 0; i < row; ++i)
    {
        int crd1 = crd1_start + i * 64;
#pragma unroll
        for (int j = 0; j < col; ++j)
        {
            int crd0 = crd0_start + j * 64;
            int offset = (i + j * row) * 64 * 64;
            tma_load_2d(desc_ptr, mbar_ptr, static_cast<uint64_t>(cache_hint), smem_ptr + offset, crd0, crd1);
        }
    }
}
```

对于 sB（256×64），需要 1×4 个 copy box，`num_box_row_b = bK / 64 = 1`，`num_box_col_b = bN / 64 = 4`。

## Producer 流程

Producer warpgroup 专门负责 TMA 数据加载：

```cpp
    if (warp_group_role == WarpGroupRole::Producer)
    {
        warpgroup_reg_dealloc<LoadRegisterRequirement>();

        if (warp_idx == 0 && lane_predicate == 1)
        {
            int k_tile = 0;
            for (; k_tile_count > 0; --k_tile_count)
            {
                int pipe = write_state.index_;
                auto tile_sA = sA + pipe * bM * bK;
                auto tile_sB = sB + pipe * bN * bK;

                mbarrier_wait(&consumer_mbar[pipe], write_state.phase_);
                arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
                tma_copy_a(&tma_a, &producer_mbar[pipe], tile_sA, num_box_row_a, num_box_col_a, k_tile * bK, x * bM);
                tma_copy_b(&tma_b, &producer_mbar[pipe], tile_sB, num_box_row_b, num_box_col_b, y * bN, k_tile * bK);
                ++k_tile;
                ++write_state;
            }
        }
    }
```

Producer 的工作流程：

1. 等待 `consumer_mbar[pipe]`，确保 Consumer 已经完成该 stage 的计算
2. 设置 `expect_tx` 并启动 TMA 拷贝
3. 更新 `write_state`

注意 `write_state.phase_` 初始化为 1：

```cpp
    PipelineState<NumPipe> read_state;
    PipelineState<NumPipe> write_state;
    write_state.phase_ = 1;
```

这是因为 Consumer 在初始化时还没有执行任何 arrive 操作，`consumer_mbar` 的 phase 仍为 0。将 `write_state.phase_` 设为 1 可以确保 Producer 在第一次 wait 时会阻塞，直到 Consumer arrive 后 phase 切换为 1 才放行。

## Consumer 流程

Consumer warpgroup 专门负责 WGMMA 计算：

```cpp
    else if (warp_group_role == WarpGroupRole::Consumer)
    {
        warpgroup_reg_alloc<MmaRegisterRequirement>();

        uint32_t reg_c[128] = {0};
        auto wgmma_desc_a = make_wgmma_desc(sA, 1, 64, 1);
        auto wgmma_desc_b = make_wgmma_desc(sB, 1, 64, 512);

        auto read_release_state = read_state;

        // Prologue: 第一个 GMMA
        int pipe = read_state.index_;
        mbarrier_wait(&producer_mbar[pipe], read_state.phase_);
        warpgroup_arrive();
        gemm(m_size, n_size, k_size, wgmma_desc_a, wgmma_desc_b, reg_c, pipe);
        warpgroup_commit_batch();
        ++read_state;
        k_tile_count -= 1;

        // 主循环
        for (; k_tile_count > 0; --k_tile_count)
        {
            int read_pipe = read_state.index_;
            mbarrier_wait(&producer_mbar[read_pipe], read_state.phase_);
            warpgroup_arrive();
            gemm(m_size, n_size, k_size, wgmma_desc_a, wgmma_desc_b, reg_c, read_pipe);
            warpgroup_commit_batch();
            warpgroup_wait<1>();

            mbarrier_arrive(&consumer_mbar[read_release_state.index_]);
            ++read_state;
            ++read_release_state;
        }

        // 等待所有 GMMA 完成
        warpgroup_wait<0>();
        mbarrier_arrive(&consumer_mbar[read_release_state.index_]);
        // ...
    }
```

Consumer 使用了两个 PipelineState：

- `read_state`：跟踪当前读取的 stage
- `read_release_state`：跟踪需要释放的 stage（延迟一个 stage）

这种设计确保了：

1. Consumer 启动一个 GMMA 后不立即释放 barrier
2. 在下一次迭代时，先等待 `warpgroup_wait<1>()` 确保 GMMA 完成，然后再 arrive `consumer_mbar`
3. 这样 Producer 可以在 Consumer 真正完成计算后再写入数据

## Epilogue

计算完成后，Consumer 需要将结果写回 global memory。

首先使用 `stmatrix` 将累加器保存到共享内存：

```cpp
    warpgroup_wait<0>();
    mbarrier_arrive(&consumer_mbar[read_release_state.index_]);
    stmatrix_copy(reg_c, shared_memory, 1);
    bar_sync(128, 0);
```

`stmatrix_copy` 函数增加了 `warpgroup_idx` 参数：

```cpp
__device__ __forceinline__ static void
stmatrix_copy(uint32_t *frag, half *smem_dst, int warpgroup_idx)
{
    int rep = 16;
    int tid = threadIdx.x;
    int warp_idx = canonical_warp_idx_sync() - 4 * warpgroup_idx;
    int row = tid % 16;
    int col = (tid % 32) / 16;

#pragma unroll
    for (int i = 0; i < rep; ++i)
    {
        // ... 读取 128 个寄存器
        int local_row = row * 256;
        int local_col_sw = (col + i * 2) ^ (row % 8);
        int offset = local_row + local_col_sw * 8 + warp_idx * 16 * 256;

        stmatrix_atom(a0, a1, a2, a3, smem_dst + offset);
        stmatrix_atom(a4, a5, a6, a7, smem_dst + offset + 8192 * 2);
    }
}
```

由于累加器有 128 个 32 位寄存器，需要执行 16 次 `stmatrix` 操作（每次处理 16×16 个 half）。

然后通过 `bar_sync` 进行 warpgroup 内部同步（只有 Consumer warpgroup 的 128 个线程参与）：

```cpp
__device__ static void bar_sync(uint32_t num_threads, uint32_t barrier_id)
{
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}
```

最后将数据从共享内存写回 global memory：

```cpp
    auto gC = C + x * bM * N + y * bN;
    store<bM, bN, T>(gC, shared_memory, N, 1);
```

`store` 函数：

```cpp
template <int bM, int bN, class T>
__device__ __forceinline__ static void
store(T *gC, T *shared_memory, int N, int warpgroup_idx)
{
    int tid = threadIdx.x;
    int nbN = bN / 8; // float4 = 8 half, 32
    int row_base = (tid - 128 * warpgroup_idx) / nbN;
    int col = (tid - 128 * warpgroup_idx) % nbN;

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        int col_sw1 = row_base ^ col;
        int col_sw2 = (row_base + 4) ^ col;
        int row1 = row_base + i * 8;
        int row2 = row_base + 4 + i * 8;
        reinterpret_cast<float4 *>(gC + row1 * N)[col] = reinterpret_cast<float4 *>(shared_memory + row1 * bN)[col_sw1];
        reinterpret_cast<float4 *>(gC + row2 * N)[col] = reinterpret_cast<float4 *>(shared_memory + row2 * bN)[col_sw2];
    }
}
```

注意由于只有 128 个线程处理 128×256 的数据，每个线程需要处理 16 行。Swizzle 需要特殊处理以避免 bank conflicts。

## 性能测试

矩阵大小 M = N = K = 4096。测试平台 H200，TFLOPS = 989e12。

测试结果：

```cpp
cublas time = 0.188563 ms, TFLPOS = 728.876016, mfu = 0.736983
 mma time = 0.183942 ms, TFLPOS = 747.184723, mfu = 0.755495
```

可以看到，Warp Specialization 版本的性能略优于 cublas，实现了超过 100% 的 cublas 性能。相比非 WS 版本的 98% MFU，WS 版本达到了约 102% 的 MFU。

性能提升的原因：

1. **真正的并行**：加载和计算完全并行，无需交替等待
2. **更大的 tile**：blockN = 256 提高了计算强度
3. **更好的寄存器利用**：Consumer 获得更多寄存器

## 总结

Warp Specialization 模式的关键点：

| 特性 | sm90_wgmma_tma | sm90_wgmma_tma_ws |
|:------:|:----------------:|:-------------------:|
| 线程数 | 128（1 warpgroup） | 256（2 warpgroups） |
| blockN | 128 | 256 |
| WGMMA | m64n128k16 | m64n256k16 |
| 流水线深度 | 3 | 4 |
| 加载/计算 | 同一 warpgroup 交替 | 分离到不同 warpgroup |
| 寄存器分配 | 静态 | 动态（setmaxnreg） |
| MFU | ~98% | ~102% |
