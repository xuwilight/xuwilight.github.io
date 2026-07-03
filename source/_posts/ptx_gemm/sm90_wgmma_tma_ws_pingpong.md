---
title: SM90 GEMM 实现（三）—— Persistent Pingpong
date: 2026-01-20 12:00:00
tags: [CUDA, GEMM, TMA, WGMMA, SM90, GPU]
categories: [GEMM 性能优化]
description: 本文进一步实现 persistent pingpong kernel 版本。主要改进是将原本的 grid-level 并行改为 persistent 模式，并引入两个 consumer warpgroup 进行 pingpong 式交替计算，以更好地隐藏流水线启动开销，提高 SM 利用率。

---

# sm90_wgmma_tma_ws_pingpong

本文基于 [sm90_wgmma_tma_ws](./sm90_wgmma_tma_ws.md) 中介绍的 Warp Specialization 矩阵乘，进一步实现 **persistent pingpong kernel** 版本。主要改进是将原本的 **grid-level 并行** 改为 **persistent 模式**，并引入两个 consumer warpgroup 进行 pingpong 式交替计算，以更好地隐藏流水线启动开销，提高 SM 利用率。

## Persistent Kernel 与 Pingpong 调度

### 什么是 Persistent Kernel？

在传统的 CUDA 编程模型中，每个 thread block 通常只处理一个工作单元（tile），当 block 完成后就退出。

Persistent Kernel 是一种 CUDA 内核设计模式，其核心思想是：内核启动后不会立即退出，而是在 SM（Streaming Multiprocessor）上持续运行，通过一个全局工作调度器（TileScheduler）动态获取任务（如矩阵乘的 tile），直到所有任务处理完毕。通常，内核启动的线程块数量等于 SM 数量，每个线程块通过循环从 TileScheduler 中领取未处理的 tile 进行计算。

**Persistent kernel 的优点：**

1. **减少 kernel 启动开销**：传统内核每次计算一个 tile 就需要重新启动，而 persistent kernel 只需启动一次，减少了 kernel 启动的开销。
2. **提高 SM 利用率**：通过让线程块持续处理多个 tile，可以更好地隐藏流水线延迟，提高 SM 的计算资源利用率。
3. **负载均衡**：所有 SM 持续工作直到所有 tile 处理完成，避免了部分 SM 提前空闲的情况。
4. **更好的流水线重叠**：在 persistent 模式下，producer 可以提前加载下一个 tile 的数据，与当前 tile 的计算更好地重叠。

### 什么是 Pingpong 调度？

**Pingpong** 在这里特指使用两个 consumer warpgroup 交替执行计算和存储操作：

- 当 Consumer0 在执行 WGMMA 计算时，Consumer1 可以将之前计算的结果写入全局内存
- 当 Consumer1 在执行 WGMMA 计算时，Consumer0 可以将之前计算的结果写入全局内存

**Pingpong 调度的优点：**

1. **更好的计算-存储重叠**：当一个 consumer 计算时，另一个 consumer 可以同时进行存储操作，最大化硬件利用率
2. **隐藏存储延迟**：存储操作通常有较高的延迟，通过交替执行可以隐藏这部分延迟
3. **减少空闲时间**：避免了计算单元等待存储完成的时间，减少了流水线气泡

## 主要改进点

与 Warp Specialization 版本相比，persistent pingpong 版本引入了以下关键改进：

1. **Persistent 执行模式**：block 持续运行，处理多个 tile
2. **三个 warpgroup**：Producer、Consumer0、Consumer1（共 384 个线程）
3. **TileScheduler**：动态分配 tile 给各个 block
4. **OrderedSequenceBarrier**：协调两个 consumer 的执行顺序
5. **Pingpong 流水线**：两个 consumer 交替进行计算和存储

## Host 侧设置

### 线程数与 grid 配置

线程数增加到 **384**，对应三个 warpgroup：

```cpp
    constexpr int num_threads = 128 * 3; // one producer warpgroup, two consumer warpgroup
```

grid 大小不再是 tile 的数量，而是设置为 **SM 的数量**（对于 H100/H200 为 132）。这样每个 SM 上运行一个 persistent block，最大化硬件利用率。

```cpp
    constexpr int sm_count = 132; // H100 132 sm
    dim3 block(num_threads);
    dim3 grid(sm_count);
```

### 共享内存配置

在 persistent pingpong 实现中，由于两个 consumer warpgroup 交替使用共享内存，epilogue 区域无法像之前那样复用。当一个 consumer 正在计算时，它的结果需要存储在共享内存中，而另一个 consumer 可能正在使用共享内存进行其他操作。因此，我们需要额外的共享内存区域来暂存计算结果。

**参数配置：**

```cpp
    constexpr int blockM = 128;
    constexpr int blockN = 256;
    constexpr int blockK = 64;
    constexpr int numPipe = 4;

    constexpr int epM = 128;
    constexpr int epN = 128;
```

**共享内存布局：**

- `sA`：存储 A 矩阵的 tile，大小为 `blockM * blockK * numPipe`
- `sB`：存储 B 矩阵的 tile，大小为 `blockN * blockK * numPipe`
- `sC`：暂存计算结果，大小为 `epM * epN`

**共享内存大小计算：**

```
smem_size = sizeof(half) * ((blockM + blockN) * blockK * numPipe + epM * epN)
          = 2 * ((128 + 256) * 64 * 4 + 128 * 128)
          = 2 * (384 * 256 + 16384)
          = 2 * (98304 + 16384)
          = 2 * 114688
          = 229376 字节 ≈ 224KB
```

由于一个 SM 上只有一个 persistent thread block，这个共享内存大小（约 224KB）接近 SM 的硬件限制（228KB），最大化利用了共享内存资源。

### base 参数

代码中引入了 `base = 2` 参数，用于 tile 坐标的变换，实现更细粒度的负载均衡：

```cpp
    constexpr int base = 2;
```

在 kernel 中，tile 坐标变换如下：

```cpp
    auto x = work_tile_info.M_idx;
    auto y = work_tile_info.N_idx;
    y = (y << base) + (x & ((1 << base) - 1));  // y = y * 4 + (x % 4)
    x = (x >> base);                             // x = x / 4
```

这种变换将原本的 tile 网格重新映射，使得相邻的 tile 被分配给不同的 SM，从而改善缓存利用率。

## TileScheduler —— 动态任务分配

### TileScheduler 类

`TileScheduler` 负责在 persistent 模式下为每个 block 分配工作 tile。它的核心思想是让每个 block 从全局线性索引开始，每次前进 `step_size`（等于 SM 数量），确保不同 block 处理不同的 tile。

```cpp
class TileScheduler
{
private:
    uint64_t current_work_linear_idx_;  // 当前工作的线性索引，指向待处理的 tile
    uint64_t total_grid_size_;          // 总共需要处理的 tile 数量
    uint64_t step_size_;                // 步长，等于 SM 数量，确保不同 block 处理不同 tile
    int32_t grid_dim_m_;                // M 方向的 tile 总数（用于将线性索引转换为二维坐标）

public:
    // 用于描述一个工作单元（tile）的信息
    struct WorkTileInfo
    {
        int32_t M_idx = 0;              // 在 M 方向的 tile 索引
        int32_t N_idx = 0;              // 在 N 方向的 tile 索引
        bool is_valid_tile = false;     // 该 tile 是否有效（是否还有剩余工作）

        // 判断当前 tile 是否有效
        __device__ __forceinline__ bool is_valid() const
        {
            return is_valid_tile;
        }

        // 返回一个无效的 tile，表示没有更多工作
        __device__ __forceinline__ static WorkTileInfo invalid_work_tile()
        {
            return {-1, -1, false};
        }
    };

    // 构造函数：初始化调度器
    // grid_dim_m: M 方向的 tile 总数
    // total_grid_size: 总共需要处理的 tile 数量
    // sm_count: SM 数量，用于计算步长
    __device__ __forceinline__ TileScheduler(int32_t grid_dim_m, uint64_t total_grid_size, uint64_t sm_count = 132)
        : total_grid_size_(total_grid_size),
          step_size_(sm_count),         // 步长 = SM 数量
          grid_dim_m_(grid_dim_m)
    {
        // 每个 block 从 blockIdx.x 开始，作为初始线性索引
        current_work_linear_idx_ = blockIdx.x;
    }

    // 获取当前分配的工作 tile
    // 将线性索引转换为二维坐标 (M_idx, N_idx)
    // 线性索引按行主序展开：index = n * grid_dim_m + m
    __device__ __forceinline__ WorkTileInfo get_current_work() const
    {
        // 如果线性索引超出范围，说明没有更多工作
        if (current_work_linear_idx_ >= total_grid_size_)
        {
            return WorkTileInfo::invalid_work_tile();
        }

        // 将线性索引转换为二维坐标
        int32_t m = static_cast<int32_t>(current_work_linear_idx_ % grid_dim_m_);  // M 方向索引
        int32_t n = static_cast<int32_t>(current_work_linear_idx_ / grid_dim_m_);  // N 方向索引

        return {m, n, true};
    }

    // 前进到下一个工作单元
    // advance_count: 前进的 tile 数量（乘以 step_size_）
    // 例如：advance_to_next_work(2) 表示前进 2 * step_size 个 tile
    __device__ __forceinline__ void advance_to_next_work(uint32_t advance_count = 1)
    {
        current_work_linear_idx_ += step_size_ * uint64_t(advance_count);
    }
};
```

**关键点：**

- `step_size_` 等于 SM 数量（132），确保不同 block 处理不同的 tile
- 线性索引按行主序转换为二维坐标：(M_idx, N_idx)
- `advance_to_next_work(n)` 前进 n × step_size 个 tile

### 两个 Consumer 的 Tile 分配

每个 block 有两个 consumer，它们的 tile 分配如下：

- **Consumer0**：从 `blockIdx.x` 开始，每次前进 `2 * step_size`
- **Consumer1**：从 `blockIdx.x + step_size` 开始，每次前进 `2 * step_size`

```cpp
    // Consumer1 初始化时跳过一个 step_size
    if (warp_group_role == WarpGroupRole::Consumer1)
    {
        scheduler.advance_to_next_work();
        read_state.advance(num_k_tiles);
    }
```

在循环末尾：

```cpp
    scheduler.advance_to_next_work(2);  // 前进 2 * step_size
```

这种分配方式确保两个 consumer 处理不同的 tile，同时保持负载均衡。

### 分配示例：8×8 个 tile，5 个 SM

假设有 M = 8, N = 8 共 64 个 tile，使用 5 个 SM 运行。每个 SM 上的 block 有两个 consumer（Consumer0 和 Consumer1）。

线性索引按**行主序**展开：`tile_id = n × 8 + m`（M 方向连续，即同一列内 tile_id 连续）。

初始分配（第一轮）：

| M\N | N=0 | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| M=0 | SM0:C0 | SM3:C1 | SM1:C1 | SM4:C0 | SM2:C0 | SM0:C1 | SM3:C0 | SM1:C0 |
| M=1 | SM1:C0 | SM4:C0 | SM2:C0 | SM0:C1 | SM3:C1 | SM1:C1 | SM4:C1 | SM2:C1 |
| M=2 | SM2:C1 | SM0:C0 | SM3:C0 | SM1:C0 | SM4:C1 | SM2:C1 | SM0:C0 | SM3:C0 |
| M=3 | SM3:C0 | SM1:C1 | SM4:C1 | SM2:C1 | SM0:C0 | SM3:C0 | SM1:C0 | SM4:C0 |
| M=4 | SM4:C1 | SM2:C1 | SM0:C0 | SM3:C0 | SM1:C1 | SM4:C1 | SM2:C1 | SM0:C0 |
| M=5 | SM0:C1 | SM3:C0 | SM1:C0 | SM4:C0 | SM2:C1 | SM0:C1 | SM3:C1 | SM1:C1 |
| M=6 | SM1:C1 | SM4:C1 | SM2:C1 | SM0:C1 | SM3:C0 | SM1:C0 | SM4:C0 | SM2:C0 |
| M=7 | SM2:C0 | SM0:C1 | SM3:C1 | SM1:C1 | SM4:C0 | SM2:C0 | SM0:C1 | SM3:C1 |

**说明：**

- 表格中的 "SMx:Cy" 表示该 tile 由 SM x 的 Consumer y 处理
- Consumer0 从 `blockIdx.x` 开始，每次前进 `2 × step_size`（即 10）
- Consumer1 从 `blockIdx.x + step_size` 开始，每次前进 `2 × step_size`（即 10）
- 这种分配方式确保两个 consumer 处理不同的 tile，同时所有 SM 均匀分配工作

**分配规律：**

- Consumer0 处理的 tile 索引满足：`tile_id % 10 ∈ {0, 1, 2, 3, 4}`
- Consumer1 处理的 tile 索引满足：`tile_id % 10 ∈ {5, 6, 7, 8, 9}`

## OrderedSequenceBarrier —— 顺序同步

### 为什么需要 OrderedSequenceBarrier？

在 pingpong 模式下，两个 consumer 需要交替执行计算和存储操作。正确的执行顺序应该是：

1. Consumer0 执行计算，Consumer1 等待
2. Consumer0 完成计算后通知 Consumer1，Consumer0 开始存储，Consumer1 开始计算
3. Consumer1 完成计算后通知 Consumer0，Consumer1 开始存储，Consumer0 开始下一个 tile 的计算
4. 如此循环往复

`OrderedSequenceBarrier` 提供了这种顺序保证，确保两个 consumer 按照正确的顺序交替执行。

### OrderedSequenceBarrier 实现

```cpp
template <int SequenceDepth_, int SequenceLength_>
class OrderedSequenceBarrier
{
public:
    static constexpr int Depth = SequenceDepth_;      // 阶段数量：2（计算和存储）
    static constexpr int Length = SequenceLength_;    // consumer 数量：2
    using BarrierStorage = uint64_t[Depth * Length];  // 存储类型：4 个 mbarrier

private:
    uint64_t *barrier_base_ptr_;  // mbarrier 数组的基地址
    PipelineState<Depth> stage_;  // 当前阶段状态（包含 index 和 phase）
    int group_id_;                 // 本 consumer 的编号（0 或 1）

public:
    // 构造函数：初始化 barrier
    // smem_ptr: 共享内存中的 barrier 数组指针
    // group_id: 本 consumer 的编号（Consumer0 传入 0，Consumer1 传入 1）
    __device__ OrderedSequenceBarrier(uint64_t *smem_ptr, int group_id)
        : barrier_base_ptr_(smem_ptr), group_id_(group_id)
    {
        // 获取当前 warp 索引和 lane 状态
        int warp_idx = canonical_warp_idx_sync();
        int lane_predicate = elect_one_sync();

        // 只让 warp 0 的一个线程初始化所有 mbarrier
        if (warp_idx == 0 && lane_predicate)
        {
            for (int i = 0; i < Depth * Length; ++i)
            {
                // 初始化每个 mbarrier，arrive_count = 128（一个 warpgroup 有 128 线程）
                mbarrier_init(&barrier_base_ptr_[i], 128);
            }
        }
        // 等待所有线程完成 mbarrier 初始化
        __syncthreads();

        // Consumer0 初始 phase 设为 1，可以立即开始执行
        // Consumer1 保持默认 phase = 0，需要等待 Consumer0 通知
        if (group_id == 0)
        {
            stage_.set(0, 1);
        }
    }

    // 等待轮到本 consumer 执行
    // 阻塞直到收到上一个 consumer 的 arrive 通知
    __device__ __forceinline__ void wait()
    {
        mbarrier_wait(get_my_barrier_ptr(), stage_.phase_);
    }

    // 通知下一个 consumer 可以开始执行
    // 1. 计算下一个 consumer 的编号：(group_id + 1) % Length
    // 2. 向下一个 consumer 的 barrier 发送 arrive 信号
    // 3. 前进到下一个阶段（index++，可能翻转 phase）
    __device__ __forceinline__ void arrive()
    {
        int next_group_id = (group_id_ + 1) % Length;
        uint64_t *target_barrier = &barrier_base_ptr_[stage_.index_ * Length + next_group_id];
        mbarrier_arrive(target_barrier);
        ++stage_;
    }

private:
    // 获取本 consumer 当前阶段对应的 barrier 指针
    // barrier 索引 = stage_index * Length + group_id
    __device__ __forceinline__ uint64_t *get_my_barrier_ptr()
    {
        return &barrier_base_ptr_[stage_.index_ * Length + group_id_];
    }
};
```

**工作原理：**

- `Depth = 2`：两个阶段（计算和存储）
- `Length = 2`：两个 consumer
- 共 4 个 mbarrier：`barrier[stage][group_id]`
- Consumer0 初始 phase 为 1，可以立即执行；Consumer1 初始 phase 为 0，需要等待
- `arrive()` 通知下一个 consumer，并前进到下一个阶段

### Consumer 协调时序表

| 时间点 | Consumer0 | Consumer1 | 说明 |
|--------|-----------|-----------|------|
| t0 | wait() → 计算 | wait()（阻塞） | Consumer0 开始计算 |
| t1 | arrive() → wait() → 存储 | wait() 返回 → 计算 | Consumer0 通知 Consumer1 |
| t2 | wait() 返回 → 计算 | arrive() → wait() → 存储 | Consumer1 通知 Consumer0 |
| t3 | arrive() → wait() → 存储 | wait() 返回 → 计算 | 循环继续 |

### 在 kernel 中的使用

```cpp
    // consumer pingpong mbarrier
    __shared__ alignas(8) uint64_t pingpong_mbar[2 * 2];                                  // mma and store two stages, two consumers
    OrderedSequenceBarrier<2, 2> math_wg_order_barrier(pingpong_mbar, warpgroup_idx - 1); // pingpong barrier
```

Consumer0 的 `group_id = 0`，Consumer1 的 `group_id = 1`。

## 三个 warpgroup 的分工

### 角色定义

```cpp
    enum class WarpGroupRole
    {
        Producer = 0,
        Consumer0 = 1,
        Consumer1 = 2
    };
    auto warp_group_role = WarpGroupRole(warpgroup_idx);
```

- **Producer（warpgroup 0）**：只负责 TMA 加载
- **Consumer0（warpgroup 1）**：第一个计算 warpgroup
- **Consumer1（warpgroup 2）**：第二个计算 warpgroup

### 共享内存布局

```cpp
    alignas(128) extern __shared__ T shared_memory[];
    T *sA = shared_memory;
    T *sB = sA + bM * bK * NumPipe;
    T *sC = sB + bN * bK * NumPipe;
```

### Barrier 配置

```cpp
    // producer 和 consumer 之间的同步 barrier
    __shared__ alignas(8) uint64_t producer_mbar[NumPipe];
    __shared__ alignas(8) uint64_t consumer_mbar[NumPipe];

    // consumer pingpong barrier
    __shared__ alignas(8) uint64_t pingpong_mbar[2 * 2];
    OrderedSequenceBarrier<2, 2> math_wg_order_barrier(pingpong_mbar, warpgroup_idx - 1);
```

### Producer 工作循环

Producer 运行在 persistent 循环中，持续为当前 tile 加载数据：

```cpp
    if (warp_group_role == WarpGroupRole::Producer)
    {
        warpgroup_reg_dealloc<LoadRegisterRequirement>();

        while (work_tile_info.is_valid())
        {
            if (warp_idx == 0 && lane_predicate == 1)
            {
                auto x = work_tile_info.M_idx;
                auto y = work_tile_info.N_idx;
                y = (y << base) + (x & ((1 << base) - 1));
                x = (x >> base);

                auto k_tile_count = num_k_tiles;
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
            scheduler.advance_to_next_work();
            work_tile_info = scheduler.get_current_work();
        }
    }
```

### Consumer 的 pingpong 循环

两个 consumer 的运行模式基本相同，但通过 `OrderedSequenceBarrier` 协调顺序：

```cpp
    else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1)
    {
        warpgroup_reg_alloc<MmaRegisterRequirement>();

        auto wgmma_desc_a = make_wgmma_desc(sA, 1 /*swizzle type*/, 64 /*sbo*/, 1 /*lbo*/);
        auto wgmma_desc_b = make_wgmma_desc(sB, 1 /*swizzle type*/, 64 /*sbo*/, 512 /*lbo*/);

        while (work_tile_info.is_valid())
        {
            // 坐标变换
            auto x = work_tile_info.M_idx;
            auto y = work_tile_info.N_idx;
            y = (y << base) + (x & ((1 << base) - 1));
            x = (x >> base);

            uint32_t reg_c[128] = {0};

            // 1. 等待轮到本 consumer 执行计算
            math_wg_order_barrier.wait();

            // 2. 执行 WGMMA 计算
            auto read_release_state = read_state;
            int pipe = read_state.index_;
            mbarrier_wait(&producer_mbar[pipe], read_state.phase_);
            warpgroup_arrive();
            gemm(m_size, n_size, k_size, wgmma_desc_a, wgmma_desc_b, reg_c, pipe);
            warpgroup_commit_batch();
            ++read_state;
            k_tile_count -= 1;

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

            // 3. 通知下一个 consumer 可以开始计算
            math_wg_order_barrier.arrive();

            // 4. 等待 WGMMA 完成，释放 pipeline stage
            warpgroup_wait<0>();
            mbarrier_arrive(&consumer_mbar[read_release_state.index_]);
            read_state.advance(num_k_tiles);

            // 5. 等待轮到本 consumer 执行存储
            math_wg_order_barrier.wait();

            // 6. 分两部分存储结果
            auto gC = C + x * bM * N + y * bN;
            stmatrix_copy_tile128(reg_c, sC, warpgroup_idx, 0);
            bar_sync(128, 0);
            store_tile128<bM, bN, T>(gC, sC, N, warpgroup_idx, 0);

            bar_sync(128, 1);
            stmatrix_copy_tile128(reg_c, sC, warpgroup_idx, 1);
            bar_sync(128, 2);
            store_tile128<bM, bN, T>(gC + 128, sC, N, warpgroup_idx, 1);

            // 7. 通知下一个 consumer 可以开始存储
            math_wg_order_barrier.arrive();

            // 8. 获取下一个 tile
            scheduler.advance_to_next_work(2);
            work_tile_info = scheduler.get_current_work();
        }
    }
```

## Epilogue 的分块存储

由于 blockN = 256，而每个 WGMMA 操作产生 128 列的结果，所以存储分为两部分：

```cpp
    // 第一部分：前 128 列
    stmatrix_copy_tile128(reg_c, sC, warpgroup_idx, 0);
    bar_sync(128, 0);
    store_tile128<bM, bN, T>(gC, sC, N, warpgroup_idx, 0);

    // 第二部分：后 128 列
    bar_sync(128, 1);
    stmatrix_copy_tile128(reg_c, sC, warpgroup_idx, 1);
    bar_sync(128, 2);
    store_tile128<bM, bN, T>(gC + 128, sC, N, warpgroup_idx, 1);
```

**存储流程：**

1. `stmatrix_copy_tile128`：将寄存器中的结果通过 stmatrix 指令写入共享内存
2. `bar_sync`：warpgroup 内部同步，确保写入完成
3. `store_tile128`：将共享内存中的结果写入全局内存

`stmatrix_copy_tile128` 和 `store_tile128` 是专门为 128 列设计的版本，相比之前的全 256 列版本更简单。

## PipelineState 的增强

`PipelineState` 增加了 `set` 和 `advance` 方法以支持 persistent 模式下的一次性多步前进：

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

    __device__ __forceinline__ void set(int idx, uint32_t p)
    {
        index_ = idx;
        phase_ = p;
    }

    __device__ __forceinline__ PipelineState &advance(int num_iterations)
    {
        // 跨越 stage 边界时翻转 phase
        if ((num_iterations < Stages) && (index_ + num_iterations) >= Stages)
        {
            phase_ ^= 1;
        }
        if ((num_iterations >= Stages) && (((index_ + num_iterations) / Stages) % 2) == 1)
        {
            phase_ ^= 1;
        }
        index_ = (index_ + num_iterations) % Stages;
        return *this;
    }
};
```

`advance` 方法用于 Consumer1 初始化时跳过 Consumer0 已处理的流水线状态：

```cpp
    if (warp_group_role == WarpGroupRole::Consumer1)
    {
        scheduler.advance_to_next_work();
        read_state.advance(num_k_tiles); // 跳过 Consumer0 已处理的 K tiles
    }
```

## 性能测试

测试条件：M = N = K = 4096，H200（峰值 TFLOPS = 989e12）。

测试结果：

```
cublas time = 0.188707 ms, TFLOPS = 728.32, MFU = 0.736
wgmma time = 0.175286 ms, TFLOPS = 784.08, MFU = 0.796
```

与 Warp Specialization 版本相比，persistent pingpong 版本达到了 **784 TFLOPS**，相当于理论峰值的 **79.6%**，比 cublas 高出约 7.7%。

## 总结

本文在 Warp Specialization 的基础上，通过 persistent pingpong 模式实现了更高效的 GEMM 内核。主要改进包括：

1. **Persistent 执行模式**：block 持续处理多个 tile，减少 kernel 启动开销
2. **Pingpong 流水线**：两个 consumer 交替进行计算和存储，更好地重叠操作
3. **动态任务调度**：通过 `TileScheduler` 实现负载均衡
4. **顺序同步机制**：通过 `OrderedSequenceBarrier` 确保正确的执行顺序
5. **分块存储优化**：将 256 列分为两个 128 列分别存储
6. **base 参数**：通过坐标变换改善缓存利用率

这些改进使得内核在 H200 上达到了 **784 TFLOPS**，相当于理论峰值的 **79.6%**，相比 Warp Specialization 版本又有显著提升。

从 multi-stage → warp specialization → persistent pingpong 的演进，展示了如何通过逐步深入的优化，将 GEMM 性能不断提升到接近硬件极限的水平。
