---
title: CUDA C++ 笔记（七）第10章——C++ Language Extensions（三）
date: 2024-06-25 20:00:00
tags: [CUDA, Async Copy, TMA, GPU]
categories: [CUDA C++ Programming Guide]
description: 异步数据拷贝：memcpy_async API、cuda::barrier、cuda::pipeline、TMA（Tensor Memory Accelerator）等异步拷贝机制的学习笔记
---

# 10.27 异步数据拷贝

## 10.27.1 memcpy_async API

**功能**：

- 支持在设备代码中显式管理从源地址（src）到目标地址（dst）的数据异步拷贝。
- 结合同步原语（如 `cuda::barrier` 和 `cuda::pipeline`）完成同步操作。

**API 的同步方式**：

- `cuda::memcpy_async`：
  - 使用 `cuda::barrier` 或 `cuda::pipeline` 进行同步。
- `cooperative_groups::memcpy_async`：
  - 使用 `cooperative_groups::wait` 进行同步。

**硬件支持**：

- **计算能力要求**：
  - `memcpy_async` 需要 compute capability >= 7.0。
  - 在 compute capability >= 8.0 的设备上，从全局内存（global memory）到共享内存（shared memory）的 `memcpy_async` 操作可以利用硬件加速，即从 Ampere 架构开始。

> 注意：`cuda::memcpy_async` 提供了好多版本，在不同的头文件中 `cuda/barrier`、`cuda/pipeline` 和 `cooperative_groups/memcpy_async.h`，这些版本对应着完全不同的使用方法和同步方式。

memcpy_async API 文档：
[https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html](https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html)

## 10.27.2 拷贝与计算模式——通过共享内存分阶段处理数据

CUDA 应用程序通常采用以下 **拷贝与计算模式**：

1. **从全局内存获取数据**；
2. **将数据存储到共享内存**；
3. **对共享内存中的数据进行计算，并在必要时将结果写回全局内存**。

以下部分展示了如何在没有和使用 `memcpy_async` 的情况下实现这种模式：

1. **无 `memcpy_async` 的实现**：
   - 提供了一个示例，其中数据传输与计算不能重叠，并使用中间寄存器来拷贝数据。
2. **使用 `memcpy_async` 的实现**：
   - 通过引入 `memcpy_async` 和 `cuda::memcpy_async` API，直接将数据从全局内存拷贝到共享内存，无需使用中间寄存器，从而优化了上述示例。

异步数据拷贝的不同方式：

1. **使用 `cuda::barrier` 的异步数据拷贝**：
   - 结合协作组（cooperative groups）和屏障（barrier）机制进行拷贝。
2. **单阶段异步数据拷贝**：
   - 使用 `cuda::pipeline`，通过单阶段流水线完成数据拷贝。
3. **多阶段异步数据拷贝**：
   - 使用 `cuda::pipeline`，通过多阶段流水线处理大规模数据的拷贝与计算。

## 10.27.3 Without memcpy_async

```cpp
#include <cooperative_groups.h>
__device__ void compute(int* global_out, int const* shared_in) {
    // Computes using all values of current batch from shared memory.
    // Stores this thread's result back to global memory.
}

__global__ void without_memcpy_async(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  assert(size == batch_sz * grid.size()); // Exposition: input size fits batch_sz * grid_size

  extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

  size_t local_idx = block.thread_rank();

  for (size_t batch = 0; batch < batch_sz; ++batch) {
    // Compute the index of the current batch for this block in global memory:
    size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
    size_t global_idx = block_batch_idx + threadIdx.x;
    shared[local_idx] = global_in[global_idx];

    block.sync(); // Wait for all copies to complete

    compute(global_out + block_batch_idx, shared); // Compute and write result to global memory

    block.sync(); // Wait for compute using shared memory to finish
  }
}
```

## 10.27.4 With memcpy_async

传统的全局内存到共享内存的拷贝方式：

```cpp
shared[local_idx] = global_in[global_idx];
```

被以下异步拷贝操作替代：

```cpp
cooperative_groups::memcpy_async(group, shared, global_in + batch_idx, sizeof(int) * block.size());
```

**memcpy_async 的特性**

1. **异步拷贝**：
   - 数据从全局内存 `global_in + batch_idx` 开始，拷贝 `sizeof(int) * block.size()` 字节到共享内存 `shared`。
   - 拷贝操作由另一线程模拟完成，与当前线程异步执行。
   - 当前线程需要通过 `cooperative_groups::wait(group)` 等待拷贝完成。
2. **数据一致性**：
   - 在拷贝完成之前，若修改全局内存数据或读取/写入共享内存数据，会导致 **数据竞争（data race）**。
3. **硬件加速**：
   - 在计算能力 >= 8.0 的设备上，`memcpy_async` 通过硬件加速，直接从全局内存传输到共享内存，避免了 **中间寄存器** 的使用，进一步提高效率。

```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

__device__ void compute(int* global_out, int const* shared_in);

__global__ void with_memcpy_async(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  assert(size == batch_sz * grid.size()); // Exposition: input size fits batch_sz * grid_size

  extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

  for (size_t batch = 0; batch < batch_sz; ++batch) {
    size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
    // Whole thread-group cooperatively copies whole batch to shared memory:
    cooperative_groups::memcpy_async(block, shared, global_in + block_batch_idx, sizeof(int) * block.size());

    cooperative_groups::wait(block); // Joins all threads, waits for all copies to complete

    compute(global_out + block_batch_idx, shared);

    block.sync();
  }
}
```

## 10.27.5 Asynchronous Data Copies using cuda::barrier

`cuda::memcpy_async` 针对 `cuda::barrier` 的重载允许通过屏障同步异步数据传输。这种重载的操作机制如下：

1. **拷贝操作执行机制**：拷贝操作被模拟为由绑定到屏障的另一个线程执行。
2. **计数管理**：
   - 在创建异步拷贝操作时，当前阶段的期望计数（expected count）增加。
   - 在拷贝操作完成时，期望计数减少。
3. **屏障阶段推进**：屏障的阶段（phase）仅在以下条件满足时推进：
   - 所有参与屏障的线程均已到达。
   - 当前阶段绑定的所有 `memcpy_async` 操作已完成。

以下示例展示了一个线程块范围内的屏障（block-wide barrier），其中所有线程块的线程均参与同步操作，并用 `barrier arrive_and_wait` 替代了等待操作，同时提供与之前示例相同的功能。

> 注意：这里所有线程都参与了 `cuda::memcpy_async` 这个操作，我尝试了下只在第一个线程发起 `cuda::memcpy_async`，结果是不对的，后面介绍的 TMA 只需要一个线程触发 `cuda::memcpy_async` 即可。

```cpp
#include <cooperative_groups.h>
#include <cuda/barrier>
__device__ void compute(int* global_out, int const* shared_in);

__global__ void with_barrier(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

  extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

  // Create a synchronization object (C++20 barrier)
  __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
  if (block.thread_rank() == 0) {
    init(&barrier, block.size()); // Friend function initializes barrier
  }
  block.sync();

  for (size_t batch = 0; batch < batch_sz; ++batch) {
    size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
    cuda::memcpy_async(block, shared, global_in + block_batch_idx, sizeof(int) * block.size(), barrier);

    barrier.arrive_and_wait(); // Waits for all copies to complete

    compute(global_out + block_batch_idx, shared);

    block.sync();
  }
}
```

## 10.27.6 Performance Guidance for memcpy_async

> 注：原文中的 "Warp Entanglement 没看懂" 部分已略去。

1. **对齐（Alignment）**

   - **对齐的重要性**：
     - `cp.async` 支持每次异步拷贝 4、8 或 16 字节的数据。
     - 如果 `memcpy_async` 的大小是这些字节的倍数，且指针对齐到相应的边界，则 `memcpy_async` 可完全利用硬件加速。
   - **最佳性能对齐**：
     - 全局内存和共享内存的对齐要求为 128 字节，以确保最佳性能。
   - **运行时对齐检查的影响**：
     - 若指针对齐属性不明确（如对齐到 1 或 2 字节），需要运行时检查对齐情况，这会增加代码大小并引入运行时开销。
2. **可平凡复制类型（Trivially Copyable）**

   - **限制**：
     - 如果 `memcpy_async` 的指针类型不是可平凡复制（TriviallyCopyable）类型，则需要调用拷贝构造函数，无法使用 `cp.async` 指令进行加速。
   - **建议**：
     - 确保传递的指针指向 TriviallyCopyable 类型的数据（如 `int`、`float`），避免性能损失。

# 10.28 Asynchronous Data Copies using cuda::pipeline

## Pipeline 的结构和机制

1. **双端 N 阶段队列**：
   - `cuda::pipeline` 是一个 **双端队列**，包含 N 个阶段（stage）。
   - 数据按照 **先进先出（FIFO）** 顺序处理。
2. **两个指针**：
   - **Head**：指向队列的最早阶段（正在被消费者消费）。
   - **Tail**：指向队列的最新阶段（正在被生产者填充）。
3. **阶段管理**：
   - 管道的阶段可以被生产者（producer）填充任务，也可以被消费者（consumer）处理任务。

## 10.28.1 Single-Stage Asynchronous Data Copies using cuda::pipeline

这里演示了单 stage 的例子：

```cpp
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

__device__ void compute(int* global_out, int const* shared_in);
__global__ void with_single_stage(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

    constexpr size_t stages_count = 1; // Pipeline with one stage
    // One batch must fit in shared memory:
    extern __shared__ int shared[];  // block.size() * sizeof(int) bytes

    // Allocate shared storage for a single stage cuda::pipeline:
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    // Each thread processes `batch_sz` elements.
    // Compute offset of the batch `batch` of this thread block in global memory:
    auto block_batch = [&](size_t batch) -> int {
      return block.group_index().x * block.size() + grid.size() * batch;
    };

    for (size_t batch = 0; batch < batch_sz; ++batch) {
        size_t global_idx = block_batch(batch);

        // Collectively acquire the pipeline head stage from all producer threads:
        pipeline.producer_acquire();

        // Submit async copies to the pipeline's head stage to be
        // computed in the next loop iteration
        cuda::memcpy_async(block, shared, global_in + global_idx, sizeof(int) * block.size(), pipeline);
        // Collectively commit (advance) the pipeline's head stage
        pipeline.producer_commit();

        // Collectively wait for the operations committed to the
        // previous `compute` stage to complete:
        pipeline.consumer_wait();

        // Computation overlapped with the memcpy_async of the "copy" stage:
        compute(global_out + global_idx, shared);

        // Collectively release the stage resources
        pipeline.consumer_release();
    }
}
```

**pipeline 操作分为以下几个步骤：**

1. **获取阶段**：
   - 通过 `pipeline.producer_acquire()` 获取管道的一个阶段，用于提交任务。
2. **提交异步拷贝**：
   - 使用 `cuda::memcpy_async` 将数据从全局内存拷贝到共享内存。
   - 异步操作被绑定到当前管道阶段。
3. **提交阶段**：
   - 通过 `pipeline.producer_commit()` 提交当前阶段的任务，准备下一阶段。
4. **等待完成**：
   - 使用 `pipeline.consumer_wait()` 等待异步任务完成，确保数据传输结束。
5. **执行计算**：
   - 调用 `compute()` 在共享内存中的数据上执行计算。
6. **释放阶段**：
   - 使用 `pipeline.consumer_release()` 释放当前阶段资源，为下一个阶段做准备。

## 10.28.2 Multi-Stage Asynchronous Data Copies using cuda::pipeline

在之前关于 `cooperative_groups::wait` 和 `cuda::barrier` 的例子中，内核线程会立即等待数据传输到共享内存完成。这避免了将数据从全局内存传输到寄存器，但并未通过 **拷贝与计算重叠** 隐藏 `memcpy_async` 操作的延迟。

### MultiStage-V1

为了解决这个问题，以下示例中我们使用了 CUDA 的流水线（pipeline）功能。该功能提供了一种管理 `memcpy_async` 批次序列的机制，使 CUDA 内核能够将内存传输与计算重叠。以下示例实现了一个两阶段的流水线，能够将数据传输与计算重叠。具体过程如下：

1. 初始化 pipeline shared state
2. 启动流水线，即启动第一个 batch 的 `memcpy_async`
3. 遍历所有 batch：
   - 为下一个批次调度 `memcpy_async`
   - 等待上一个批次的 `memcpy_async` 完成（线程阻塞）
   - 这就实现了对上一个批次的计算与下一个批次的异步内存复制操作重叠
4. 最后，对最后一个 batch 执行计算操作

需要注意的是，为了与 `cuda::pipeline` 兼容，这里使用了 `cuda/pipeline` 头文件中的 `cuda::memcpy_async`。

```cpp
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

__device__ void compute(int* global_out, int const* shared_in);
__global__ void with_staging(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

    constexpr size_t stages_count = 2; // Pipeline with two stages
    // Two batches must fit in shared memory:
    extern __shared__ int shared[];  // stages_count * block.size() * sizeof(int) bytes
    size_t shared_offset[stages_count] = { 0, block.size() }; // Offsets to each batch

    // Allocate shared storage for a two-stage cuda::pipeline:
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    // Each thread processes `batch_sz` elements.
    // Compute offset of the batch `batch` of this thread block in global memory:
    auto block_batch = [&](size_t batch) -> int {
      return block.group_index().x * block.size() + grid.size() * batch;
    };

    // Initialize first pipeline stage by submitting a `memcpy_async` to fetch a whole batch for the block:
    if (batch_sz == 0) return;
    pipeline.producer_acquire();
    cuda::memcpy_async(block, shared + shared_offset[0], global_in + block_batch(0), sizeof(int) * block.size(), pipeline);
    pipeline.producer_commit();

    // Pipelined copy/compute:
    for (size_t batch = 1; batch < batch_sz; ++batch) {
        // Stage indices for the compute and copy stages:
        size_t compute_stage_idx = (batch - 1) % 2;
        size_t copy_stage_idx = batch % 2;

        size_t global_idx = block_batch(batch);

        // Collectively acquire the pipeline head stage from all producer threads:
        pipeline.producer_acquire();

        // Submit async copies to the pipeline's head stage to be
        // computed in the next loop iteration
        cuda::memcpy_async(block, shared + shared_offset[copy_stage_idx], global_in + global_idx, sizeof(int) * block.size(), pipeline);
        // Collectively commit (advance) the pipeline's head stage
        pipeline.producer_commit();

        // Collectively wait for the operations commited to the
        // previous `compute` stage to complete:
        pipeline.consumer_wait();

        // Computation overlapped with the memcpy_async of the "copy" stage:
        compute(global_out + global_idx, shared + shared_offset[compute_stage_idx]);

        // Collectively release the stage resources
        pipeline.consumer_release();
    }

    // Compute the data fetch by the last iteration
    pipeline.consumer_wait();
    compute(global_out + block_batch(batch_sz-1), shared + shared_offset[(batch_sz - 1) % 2]);
    pipeline.consumer_release();
}
```

流水线（pipeline）对象是一个带有头部和尾部的双端队列，用于以先进先出的（FIFO）顺序处理任务。生产者线程将任务提交到流水线的头部，而消费者线程从流水线的尾部拉取任务。在上面的例子中，所有线程既是生产者又是消费者。线程首先将 `memcpy_async` 操作提交到流水线中以获取下一批数据，同时等待上一批 `memcpy_async` 操作完成。

提交任务到 pipeline 步骤包括：

1. 生产者线程集体通过 `pipeline.producer_acquire()` 获取流水线头部。
2. 向流水线头部提交 `memcpy_async` 操作。
3. 生产者线程集体提交（推进）流水线头部，通过调用 `pipeline.producer_commit()`。

使用已提交的阶段的步骤包括：

1. 集体等待阶段完成，例如使用 `pipeline.consumer_wait()` 等待尾部（最旧的）阶段。
2. 集体释放阶段，通过调用 `pipeline.consumer_release()`。

`cuda::pipeline_shared_state<scope, count>` 封装了有限资源，使得流水线能够处理最多 `count` 个并发阶段。如果所有资源都已被占用，`pipeline.producer_acquire()` 会阻塞生产者线程，直到消费者线程释放下一流水线阶段的资源。

### MultiStage-V2（代码精简）

这个示例可以通过将循环的前序部分（prolog）和后序部分（epilog）与循环主体合并，改写为更简洁的形式，如下所示：

```cpp
template <size_t stages_count = 2 /* Pipeline with stages_count stages */>
__global__ void with_staging_unified(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

    extern __shared__ int shared[]; // stages_count * block.size() * sizeof(int) bytes
    size_t shared_offset[stages_count];
    for (int s = 0; s < stages_count; ++s) shared_offset[s] = s * block.size();

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    auto block_batch = [&](size_t batch) -> int {
        return block.group_index().x * block.size() + grid.size() * batch;
    };

    // compute_batch: next batch to process
    // fetch_batch:  next batch to fetch from global memory
    for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < batch_sz; ++compute_batch) {
        // The outer loop iterates over the computation of the batches
        for (; fetch_batch < batch_sz && fetch_batch < (compute_batch + stages_count); ++fetch_batch) {
            // This inner loop iterates over the memory transfers, making sure that the pipeline is always full
            pipeline.producer_acquire();
            size_t shared_idx = fetch_batch % stages_count;
            size_t batch_idx = fetch_batch;
            size_t block_batch_idx = block_batch(batch_idx);
            cuda::memcpy_async(block, shared + shared_offset[shared_idx], global_in + block_batch_idx, sizeof(int) * block.size(), pipeline);
            pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        int shared_idx = compute_batch % stages_count;
        int batch_idx = compute_batch;
        compute(global_out + block_batch(batch_idx), shared + shared_offset[shared_idx]);
        pipeline.consumer_release();
    }
}
```

### MultiStage-V3（线程角色划分）

上述使用的 `pipeline<thread_scope_block>` 原语非常灵活，并支持以下两项特性（尽管前面的示例未使用这些特性）：

1. 块内的任意子集线程都可以参与流水线。
2. 在参与的线程中，任意子集线程都可以是 **生产者**、**消费者** 或两者兼而有之。

在下面的示例中，线程编号为"偶数"的线程为生产者，而其他线程为消费者：

```cpp
__device__ void compute(int* global_out, int shared_in);

template <size_t stages_count = 2>
__global__ void with_specialized_staging_unified(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();

    // In this example, threads with "even" thread rank are producers, while threads with "odd" thread rank are consumers:
    const cuda::pipeline_role thread_role
      = block.thread_rank() % 2 == 0 ? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;

    // Each thread block only has half of its threads as producers:
    auto producer_threads = block.size() / 2;

    // Map adjacent even and odd threads to the same id:
    const int thread_idx = block.thread_rank() / 2;

    auto elements_per_batch = size / batch_sz;
    auto elements_per_batch_per_block = elements_per_batch / grid.group_dim().x;

    extern __shared__ int shared[]; // stages_count * elements_per_batch_per_block * sizeof(int) bytes
    size_t shared_offset[stages_count];
    for (int s = 0; s < stages_count; ++s) shared_offset[s] = s * elements_per_batch_per_block;

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    cuda::pipeline pipeline = cuda::make_pipeline(block, &shared_state, thread_role);

    // Each thread block processes `batch_sz` batches.
    // Compute offset of the batch `batch` of this thread block in global memory:
    auto block_batch = [&](size_t batch) -> int {
      return elements_per_batch * batch + elements_per_batch_per_block * blockIdx.x;
    };

    for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < batch_sz; ++compute_batch) {
        // The outer loop iterates over the computation of the batches
        for (; fetch_batch < batch_sz && fetch_batch < (compute_batch + stages_count); ++fetch_batch) {
            // This inner loop iterates over the memory transfers, making sure that the pipeline is always full
            if (thread_role == cuda::pipeline_role::producer) {
                // Only the producer threads schedule asynchronous memcpys:
                pipeline.producer_acquire();
                size_t shared_idx = fetch_batch % stages_count;
                size_t batch_idx = fetch_batch;
                size_t global_batch_idx = block_batch(batch_idx) + thread_idx;
                size_t shared_batch_idx = shared_offset[shared_idx] + thread_idx;
                cuda::memcpy_async(shared + shared_batch_idx, global_in + global_batch_idx, sizeof(int), pipeline);
                pipeline.producer_commit();
            }
        }
        if (thread_role == cuda::pipeline_role::consumer) {
            // Only the consumer threads compute:
            pipeline.consumer_wait();
            size_t shared_idx = compute_batch % stages_count;
            size_t global_batch_idx = block_batch(compute_batch) + thread_idx;
            size_t shared_batch_idx = shared_offset[shared_idx] + thread_idx;
            compute(global_out + global_batch_idx, *(shared + shared_batch_idx));
            pipeline.consumer_release();
        }
    }
}
```

### MultiStage-V4（性能优化）

流水线会在共享内存中存储和使用一组屏障（barrier）进行同步，但如果块内的所有线程都参与流水线，这实际上是没有必要的。

对于块内所有线程都参与流水线的特定情况，我们可以通过将 `pipeline<thread_scope_thread>` 与 `__syncthreads()` 结合使用，达到比 `pipeline<thread_scope_block>` 更优的性能。

> **注意**：
>
> 1. pipeline 的 thread scope 变成 `thread_scope_thread`，意味着每个线程不会和其他线程同步。
> 2. 这里的 `cuda::memcpy_async` 改成了单线程粒度。

```cpp
template<size_t stages_count>
__global__ void with_staging_scope_thread(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    auto thread = cooperative_groups::this_thread();
    assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

    extern __shared__ int shared[]; // stages_count * block.size() * sizeof(int) bytes
    size_t shared_offset[stages_count];
    for (int s = 0; s < stages_count; ++s) shared_offset[s] = s * block.size();

    // No pipeline::shared_state needed
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    auto block_batch = [&](size_t batch) -> int {
        return block.group_index().x * block.size() + grid.size() * batch;
    };

    for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < batch_sz; ++compute_batch) {
        for (; fetch_batch < batch_sz && fetch_batch < (compute_batch + stages_count); ++fetch_batch) {
            pipeline.producer_acquire();
            size_t shared_idx = fetch_batch % stages_count;
            size_t batch_idx = fetch_batch;
            // Each thread fetches its own data:
            size_t thread_batch_idx = block_batch(batch_idx) + threadIdx.x;
            // The copy is performed by a single `thread` and the size of the batch is now that of a single element:
            cuda::memcpy_async(thread, shared + shared_offset[shared_idx] + threadIdx.x, global_in + thread_batch_idx, sizeof(int), pipeline);
            pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        block.sync(); // __syncthreads: All memcpy_async of all threads in the block for this stage have completed here
        int shared_idx = compute_batch % stages_count;
        int batch_idx = compute_batch;
        compute(global_out + block_batch(batch_idx), shared + shared_offset[shared_idx]);
        pipeline.consumer_release();
    }
}
```

## 10.28.3 Pipeline Interface

The pipeline interface requires:

- at least CUDA 11.0
- at least ISO C++ 2011 compatibility, e.g., to be compiled with `-std=c++11`
- `#include <cuda/pipeline>`

## 10.28.4 Pipeline Primitives Interface

C 风格接口，不重要，已忽略。

# 10.29 Asynchronous Data Copies using the Tensor Memory Accelerator (TMA)

许多应用程序需要在全局内存和共享内存之间移动大量数据。通常，这些数据在全局内存中以多维数组的形式存储，并具有非连续的访问模式。为了减少全局内存的使用，这些数组的子块（sub-tiles）会被复制到共享内存中以供计算使用。然而，这种加载和存储操作涉及复杂的地址计算，容易出错且重复繁琐。为了解决这一问题，**计算能力 9.0** 引入了 **张量内存加速器（Tensor Memory Accelerator, TMA）**，其主要目标是为多维数组在全局内存和共享内存之间提供高效的数据传输机制。

1. **命名**

   - "Tensor memory accelerator (TMA)" 是一个广义术语，用于描述本节提到的功能。
   - 为了保持前向兼容性并减少与 PTX ISA 的差异，TMA 操作被称为 **批量异步复制（bulk-asynchronous copies）** 或 **批量张量异步复制（bulk tensor asynchronous copies）**，具体取决于所使用的复制类型。
   - "bulk" 一词用来区分这些操作与前面章节描述的异步内存操作。

2. **维度支持**

   TMA 支持从一维到五维的数组复制。

   **一维数组的批量异步复制**：编程模型较简单，仅需设备端的指针和大小参数即可完成。

   **多维数组的批量张量异步复制**：需要一个 **张量映射（tensor map）** 来描述全局内存和共享内存中多维数组的布局。张量映射通常由主机端通过 **cuTensorMapEncode API** 创建，然后作为常量内核参数（用 **`__grid_constant__`** 标注）从主机传输到设备。在设备端，张量映射可用于在全局和共享内存之间高效地复制数据块。

3. **源地址和目标地址**

   批量异步复制的源和目标地址可以是共享内存或全局内存。支持以下操作：

   - 从全局内存到共享内存的数据读取。
   - 从共享内存到全局内存的数据写入。
   - 在同一集群的不同块之间，从共享内存到分布式共享内存的数据复制。
   - 在集群中，可以使用多播（multicast）功能将数据从全局内存传输到集群中多个块的共享内存。多播功能针对 `sm_90a` 架构进行了优化，在其他架构上可能性能显著降低，因此建议仅在 `sm_90a` 上使用。

4. **异步特性**

   TMA 的数据传输是异步的：发起线程可以在硬件异步复制数据时继续进行计算。是否真正异步取决于硬件实现，未来可能会有所变化。

   **完成机制**：
   - 从全局内存读取到共享内存时，块中的任意线程都可以通过等待共享内存屏障（Shared Memory Barrier）来检查数据是否可读。
   - 从共享内存写入到全局内存或分布式共享内存时，只有发起线程可以通过异步组完成机制（Bulk async-group based completion mechanism）检查操作是否完成。

   TMA 相关的 PTX 指令见：9.7.10.24.6 Data Movement and Conversion Instructions: `cp.async.bulk`

## 10.29.1 Using TMA to transfer one-dimensional arrays

主要步骤如下：

1. 初始化 shared memory barrier
2. 启动从全局内存到共享内存的批量异步复制操作。
3. arrive and wait shared memory barrier
4. 操作这块内存（compute）
5. 添加 async proxy fence，确保对共享内存的操作被 async proxy 可见（TMA 是被 async proxy 执行的，所以需要保证 async proxy 对我们的共享内存操作可见）
6. 启动从共享内存缓冲区到全局内存的批量异步复制操作。
7. 在内核末尾等待批量异步复制完成共享内存的读取操作。

```cpp
#include <cuda/barrier>
#include <cuda/ptx>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

static constexpr size_t buf_len = 1024;
__global__ void add_one_kernel(int* data, size_t offset)
{
  // Shared memory buffer. The destination shared memory buffer of
  // a bulk operations should be 16 byte aligned.
  __shared__ alignas(16) int smem_data[buf_len];

  // 1. a) Initialize shared memory barrier with the number of threads participating in the barrier.
  //    b) Make initialized barrier visible in async proxy.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);              // a)
    ptx::fence_proxy_async(ptx::space_shared);   // b)
  }
  __syncthreads();

  // 2. Initiate TMA transfer to copy global to shared memory.
  if (threadIdx.x == 0) {
    // 3a. cuda::memcpy_async arrives on the barrier and communicates
    //     how many bytes are expected to come in (the transaction count)
    cuda::memcpy_async(
        smem_data,
        data + offset,
        cuda::aligned_size_t<16>(sizeof(smem_data)),
        bar
    );
  }
  // 3b. All threads arrive on the barrier
  barrier::arrival_token token = bar.arrive();

  // 3c. Wait for the data to have arrived.
  bar.wait(std::move(token));

  // 4. Compute saxpy and write back to shared memory
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
    smem_data[i] += 1;
  }

  // 5. Wait for shared memory writes to be visible to TMA engine.
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // 6. Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    ptx::cp_async_bulk(
        ptx::space_global,
        ptx::space_shared,
        data + offset, smem_data, sizeof(smem_data));
    // 7. Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    ptx::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }
}
```

**要点**：

1. **Barrier Initialization**

   屏障被初始化为块中参与线程的数量。因此，只有当所有线程到达屏障时，屏障才会翻转。共享内存屏障的详细描述见 **Asynchronous Data Copies using cuda::barrier**。为了使初始化的屏障对后续的批量异步复制可见，使用 `fence.proxy.async.shared::cta` 指令。该指令确保后续的批量异步复制操作能够正确处理已初始化的屏障。

2. **TMA Read**（重点：屏障只有在所有线程到达并且所有字节到达后才会翻转）

   批量异步复制指令将硬件指向全局内存的一大块数据，并将其复制到共享内存中，同时在完成读取后更新共享内存屏障的事务计数。通常，尽量减少批量复制的次数并增加每次复制的数据量有助于实现最佳性能。由于硬件可以异步执行复制操作，无需将数据拆分为更小的块。

   发起批量异步复制操作的线程通过 `mbarrier.expect_tx` 到达屏障，该操作由 `cuda::memcpy_async` 自动执行。这不仅通知屏障线程已到达，还指明了预期的字节数（tx/transactions）。只有一个线程需要更新预期事务计数。如果多个线程更新事务计数，则屏障的预期事务为所有更新值的总和。屏障只有在所有线程到达且所有字节到达后才会翻转。一旦屏障翻转，线程和后续批量异步复制都可以安全地从共享内存读取字节。更多关于屏障事务计算的信息可参考 **PTX ISA**。

3. **Barrier Wait**

   通过 `mbarrier.try_wait` 等待屏障翻转。该方法可能返回 `true`（表示等待结束）或 `false`（表示可能超时）。循环通过重试机制等待屏障完成。

4. **SMEM Write and Sync**

   缓冲区值的递增涉及对共享内存的读写操作。为了使写入对后续的批量异步复制可见，使用 `fence.proxy.async.shared::cta` 指令。该指令将写入操作排序在共享内存的后续读取之前。每个线程首先通过 `fence.proxy.async.shared::cta` 将对共享内存对象的写入排序到异步代理中，然后所有线程的这些操作会在线程 0 调用 `__syncthreads()` 前完成排序。

5. **TMA Write and Sync**

   从共享内存写入全局内存的操作由单个线程发起。写入完成的跟踪不依赖共享内存屏障，而是通过线程局部机制实现。多个写入可以被批量到一个所谓的 **批量异步组（bulk async-group）**。之后，线程可以等待该组中的所有操作完成共享内存读取（如上述代码）或完成全局内存写入，从而使写入对发起线程可见。更多信息可参考 **PTX ISA** 中的 `cp.async.bulk.wait_group` 文档。需要注意的是，批量异步和非批量异步复制指令具有不同的异步组，例如 `cp.async.wait_group` 和 `cp.async.bulk.wait_group`。

   如下就是 TMA write 的同步机制：

   ```cpp
   // 7. Wait for TMA transfer to have finished reading shared memory.
   // Create a "bulk async-group" out of the previous bulk copy operation.
   ptx::cp_async_bulk_commit_group();
   // Wait for the group to have completed reading from shared memory.
   ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
   ```

## 10.29.2 Using TMA to transfer multi-dimensional arrays

1. **一维与多维 Tensor Map 的差异**

   - **一维情况**：
     - 不需要额外的数据结构，直接通过指针和大小即可描述数组的布局。
   - **多维情况**：
     - 需要一个 **Tensor Map** 来描述多维数组的维度、步幅和数据布局。
     - 必须在主机端创建 Tensor Map，并将其传递到 CUDA 内核。

2. **Driver API 与 Tensor Map 创建**

   **Driver API 使用**：

   - Tensor Map 是通过 CUDA Driver API 的 `cuTensorMapEncodeTiled` 函数创建的。
   - 可以通过两种方式调用 Driver API：
     1. 直接链接 CUDA Driver 库（`-lcuda`）。
     2. 使用 `cudaGetDriverEntryPoint` 动态获取函数指针。
   - 文档示例代码展示了如何获取 `cuTensorMapEncodeTiled` 函数指针，并调用它来创建 Tensor Map。

   **Tensor Map 的主要参数**：

   1. **维度信息**：数组的维数（rank）以及每个维度的大小。
   2. **stride**：从当前行到下一行的字节数，必须是 16 的倍数。
   3. **共享内存块大小（box_size）**：用于接收 TMA 传输的 shared memory 大小。
   4. **元素间距（elem_stride）**：用于处理复数等特殊情况，比如只加载实部。
   5. **性能优化选项**：
      - **Interleave**：加速小于 4 字节值的加载。
      - **Swizzle**：避免共享内存 bank conflict。
      - **L2 缓存提升（L2 Promotion）**：扩大缓存策略的影响范围。
      - **越界填充（OOB Fill）**：处理越界元素，设置为 0。
   6. **示例代码解读**：
      - 示例代码定义了一个二维数组（GMEM_HEIGHT x GMEM_WIDTH），以行优先的方式存储，创建了一个 Tensor Map 来描述其布局。
      - 使用 `cuTensorMapEncodeTiled` 初始化了 Tensor Map，并设置了优化参数（如无交错、无置换、无 L2 缓存提升）。

   示例代码：

   ```cpp
    CUtensorMap tensor_map{};
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    uint32_t elem_stride[rank] = {1, 1};

    // Get a function pointer to the cuTensorMapEncodeTiled driver API.
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

    // Create the tensor descriptor.
    CUresult res = cuTensorMapEncodeTiled(
      &tensor_map,                // CUtensorMap *tensorMap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
      rank,                       // cuuint32_t tensorRank,
      tensor_ptr,                 // void *globalAddress,
      size,                       // const cuuint64_t *globalDim,
      stride,                     // const cuuint64_t *globalStrides,
      box_size,                   // const cuuint32_t *boxDim,
      elem_stride,                // const cuuint32_t *elementStrides,
      // Interleave patterns can be used to accelerate loading of values that
      // are less than 4 bytes long.
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      // Swizzling can be used to avoid shared memory bank conflicts.
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      // L2 Promotion can be used to widen the effect of a cache-policy to a wider
      // set of L2 cache lines.
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      // Any element that is outside of bounds will be set to zero by the TMA transfer.
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
   ```

3. **Tensor Map 的主机到设备传输**

   有三种方式将 Tensor Map 传递到设备端，推荐的方式是作为内核参数传递：

   **推荐方法：通过 `__grid_constant__` 参数传递**：

   - 将 Tensor Map 作为常量参数传递给 CUDA 内核。
   - 这种方式简单高效，但可能会引发 GCC 编译器的 ABI 警告，可忽略。

   推荐的方式：

   ```cpp
   #include <cuda.h>

   __global__ void kernel(const __grid_constant__ CUtensorMap tensor_map)
   {
      // Use tensor_map here.
   }
   int main() {
     CUtensorMap map;
     // [ ..Initialize map.. ]
     kernel<<<1, 1>>>(map);
   }
   ```

### 样例代码

```cpp
#include <cuda.h>          // CUtensorMap
#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map, int x, int y) {
  // The destination shared memory buffer of a bulk tensor operation should be
  // 128 byte aligned.
  __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

  // Initialize shared memory barrier with the number of threads participating in the barrier.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  if (threadIdx.x == 0) {
    // Initialize barrier. All `blockDim.x` threads in block participate.
    init(&bar, blockDim.x);
    // Make initialized barrier visible in async proxy.
    cde::fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    // Initiate bulk tensor copy.
    cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &tensor_map, x, y, bar);
    // Arrive on the barrier and tell how many bytes are expected to come in.
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  } else {
    // Other threads just arrive.
    token = bar.arrive();
  }
  // Wait for the data to have arrived.
  bar.wait(std::move(token));

  // Symbolically modify a value in shared memory.
  smem_buffer[0][threadIdx.x] += threadIdx.x;

  // Wait for shared memory writes to be visible to TMA engine.
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, x, y, &smem_buffer);
    // Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    cde::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    cde::cp_async_bulk_wait_group_read<0>();
  }

  // Destroy barrier. This invalidates the memory region of the barrier. If
  // further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (threadIdx.x == 0) {
    (&bar)->~barrier();
  }
}
```

# 10.30 Encoding a Tensor Map on Device

## 10.30.1 Device-side Encoding and Modification of a Tensor Map

1. **设备端编码与修改 Tensor Map**

   **目的**：
   允许在设备端动态调整 Tensor Map 的布局，用于更灵活的数据传输需求。

   **步骤**：
   1. **模板创建**：
      - 在主机端使用 Driver API (`cuTensorMapEncodeTiled`) 创建一个 Tensor Map 模板，作为基础配置。
   2. **设备端修改**：
      - 在设备端，将模板复制到共享内存。
      - 使用 CUDA PTX 指令（如 `tensormap_replace_*`）修改共享内存中的 Tensor Map，调整其参数（如地址、维度、步幅等）。
   3. **写回全局内存**：
      - 使用 `cuda::ptx::tensormap_copy_fenceproxy` 将修改后的 Tensor Map 从共享内存写回全局内存，并进行必要的内存屏障操作（fencing）。

   **代码解读**：

   - 在 `encode_tensor_map` 内核中：
     - 模板通过 **`__grid_constant__`** 参数传递给内核。
     - 使用 `tensormap_replace_*` 修改模板的字段，例如 rank、box_dim、stride 等。
     - 最后通过 `tensormap_copy_fenceproxy` 将修改后的 Tensor Map 写回全局内存。

   **注意事项**：
   - 修改操作仅支持 tiled 类型的 Tensor Map。
   - 对于 `sm_90a` 架构，可以直接使用共享内存的零初始化值作为初始模板。

   ```cpp
   #include <cuda/ptx>

   namespace ptx = cuda::ptx;

   // launch with 1 warp.
   __launch_bounds__(32)
   __global__ void encode_tensor_map(const __grid_constant__ CUtensorMap template_tensor_map, tensormap_params p, CUtensorMap* out) {
      __shared__ alignas(128) CUtensorMap smem_tmap;
      if (threadIdx.x == 0) {
         // Copy template to shared memory:
         smem_tmap = template_tensor_map;

         const auto space_shared = ptx::space_shared;
         ptx::tensormap_replace_global_address(space_shared, &smem_tmap, p.global_address);
         // For field .rank, the operand new_val must be ones less than the desired
         // tensor rank as this field uses zero-based numbering.
         ptx::tensormap_replace_rank(space_shared, &smem_tmap, p.rank - 1);

         // Set box dimensions:
         if (0 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.box_dim[0]); }
         if (1 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.box_dim[1]); }
         if (2 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.box_dim[2]); }
         if (3 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.box_dim[3]); }
         if (4 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<4>{}, p.box_dim[4]); }
         // Set global dimensions:
         if (0 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<0>{}, (uint32_t) p.global_dim[0]); }
         if (1 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<1>{}, (uint32_t) p.global_dim[1]); }
         if (2 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<2>{}, (uint32_t) p.global_dim[2]); }
         if (3 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<3>{}, (uint32_t) p.global_dim[3]); }
         if (4 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<4>{}, (uint32_t) p.global_dim[4]); }
         // Set global stride:
         if (1 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.global_stride[0]); }
         if (2 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.global_stride[1]); }
         if (3 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.global_stride[2]); }
         if (4 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.global_stride[3]); }
         // Set element stride:
         if (0 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.element_stride[0]); }
         if (1 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.element_stride[1]); }
         if (2 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.element_stride[2]); }
         if (3 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.element_stride[3]); }
         if (4 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<4>{}, p.element_stride[4]); }

         // These constants are documented in this table:
         // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensormap-new-val-validity
         auto u8_elem_type = ptx::n32_t<0>{};
         ptx::tensormap_replace_elemtype(space_shared, &smem_tmap, u8_elem_type);
         auto no_interleave = ptx::n32_t<0>{};
         ptx::tensormap_replace_interleave_layout(space_shared, &smem_tmap, no_interleave);
         auto no_swizzle = ptx::n32_t<0>{};
         ptx::tensormap_replace_swizzle_mode(space_shared, &smem_tmap, no_swizzle);
         auto zero_fill = ptx::n32_t<0>{};
         ptx::tensormap_replace_fill_mode(space_shared, &smem_tmap, zero_fill);
      }
      // Synchronize the modifications with other threads in warp
      __syncwarp();
      // Copy the tensor map to global memory collectively with threads in the warp.
      // In addition: make the updated tensor map visible to other threads on device that
      // for use with cp.async.bulk.
      ptx::n32_t<128> bytes_128;
      ptx::tensormap_cp_fenceproxy(ptx::sem_release, ptx::scope_gpu, out, &smem_tmap, bytes_128);
   }
   ```

## 10.30.2 Usage of a Modified Tensor Map

2. **使用修改后的 Tensor Map**

   在设备内核中使用存储在全局内存中的 Tensor Map 时，需要建立 **release-acquire** 同步模式，以确保数据一致性。

   **步骤**：
   1. **release 部分**：
      - 修改 Tensor Map 后，通过 `cuda::ptx::tensormap_cp_fenceproxy` 完成 release。
   2. **acquire 部分**：
      - 在另一个内核中，使用 `cuda::ptx::fence_proxy_tensormap_generic` 完成 acquire。
      - 需要指定作用域（如 `.gpu` 或 `.sys`），确保数据一致性。

   **代码解读**：
   - `consume_tensor_map` 内核：
     - 使用 `fence_proxy_tensormap_generic` 获取全局内存中 Tensor Map 的最新状态。
     - 初始化共享内存和屏障，并使用 `ptx::cp_async_bulk_tensor` 进行异步数据传输。
     - 打印传输结果以验证功能。

   **注意事项**：
   - release 和 acquire 操作必须在同一块内执行，不支持跨块或跨网格同步。

   ```cpp
   // Consumer of tensor map in global memory:
   __global__ void consume_tensor_map(CUtensorMap* tensor_map) {
     // Fence acquire tensor map:
     ptx::n32_t<128> size_bytes;
     ptx::fence_proxy_tensormap_generic(ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes);
     // Safe to use tensor_map after fence..

     __shared__ uint64_t bar;
     __shared__ alignas(128) char smem_buf[4][128];

     if (threadIdx.x == 0) {
       // Initialize barrier
       ptx::mbarrier_init(&bar, 1);
       // Make barrier init visible in async proxy, i.e., to TMA engine
       ptx::fence_proxy_async(ptx::space_shared);
       // Issue TMA request
       ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global, smem_buf, tensor_map, {0, 0}, &bar);

       // Arrive on barrier. Expect 4 * 128 bytes.
       ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, sizeof(smem_buf));
     }
     const int parity = 0;
     // Wait for load to have completed
     while (!ptx::mbarrier_try_wait_parity(&bar, parity)) {}

     // print items:
     printf("Got:\n\n");
     for (int j = 0; j < 4; ++j) {
       for (int i = 0; i < 128; ++i) {
         printf("%3d ", smem_buf[j][i]);
         if (i % 32 == 31) { printf("\n"); };
       }
       printf("\n");
     }
   }
   ```

## 10.30.3 Creating a Template Tensor Map Value Using the Driver API

**目的**：
在主机端创建一个最小化的 Tensor Map 模板，后续在设备端动态修改。

**步骤**：
- 使用 `cuTensorMapEncodeTiled` 定义初始的 Tensor Map，配置必要参数（如维度、步幅、类型等）。
- 将模板返回给设备端。

```cpp
CUtensorMap make_tensormap_template() {
 CUtensorMap template_tensor_map{};
 auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

 uint32_t dims_32          = 16;
 uint64_t dims_strides_64  = 16;
 uint32_t elem_strides     = 1;

 // Create the tensor descriptor.
 CUresult res = cuTensorMapEncodeTiled(
   &template_tensor_map, // CUtensorMap *tensorMap,
   CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
   1,                    // cuuint32_t tensorRank,
   nullptr,              // void *globalAddress,
   &dims_strides_64,     // const cuuint64_t *globalDim,
   &dims_strides_64,     // const cuuint64_t *globalStrides,
   &dims_32,             // const cuuint32_t *boxDim,
   &elem_strides,        // const cuuint32_t *elementStrides,
   CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
   CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
   CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
   CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

 CU_CHECK(res);
 return template_tensor_map;
}
```

# FAQ

## TMA 和原有异步拷贝的主要区别

无论是常规的 Shared Memory 拷贝，还是 Ampere 架构下的 Shared Memory 异步拷贝，在拷贝大块的显存时，都会拆分成若干个很小的显存块，利用循环、多线程方式完成多个小显存块拷贝。每次拷贝均要计算显存的起始地址，这种寻址操作是不能被异步拷贝重叠的，并且运算指令随着小显存块的增多而线性增加。显式计算地址的原因主要是地址不连续，比如在矩阵乘中，对 Global Memory 进行分块，并将每个小块加载到 Shared Memory 中，显存块中不同行的地址是不连续的，需要手动计算。所以 Ampere 及其以前的架构，是无法减少这种频繁的地址计算操作。为了解决这个问题，Hopper 架构引入了 TMA 功能。TMA 支持以下几个功能：

1. **大块（bulk）异步显存拷贝**：使用 `cuda::memcpy_async` 接口。这个类似 CPU 上的 `memcpy`，支持一整块的显存拷贝，可以减少拷贝指令数量。
2. **多维度显存块拷贝**：这个特性主要支持不连续的多段显存块拷贝。在实际使用中，需要区分一维度显存块拷贝和多维度显存块拷贝。多维度显存块拷贝需要在 Host 端调用 `cuTensorMapEncode` 的 API，计算显存块之间的地址映射关系，然后通过带有 `__grid_constant__` 注释的 `CUtensorMap` 类型参数传递给 Kernel 函数中，调用 TMA 的异步拷贝接口完成多维度的拷贝。
3. **支持从 Shared Memory 异步拷贝到 Global Memory**：Ampere 架构只支持从 Global Memory 异步拷贝到 Shared Memory，而在 Hopper 架构上更进一步支持反向的拷贝操作，提升 Kernel 的在不同存储结构上的读写性能。

## async proxy 是什么？

[https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#async-proxy](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#async-proxy)

`cp{.reduce}.async.bulk` 操作是在异步代理（async proxy）中执行的。

在多个代理（proxy）之间访问相同的内存位置时，需要进行跨代理（cross-proxy）同步操作。对于异步代理，应该使用 `fence.proxy.async` 来同步通用代理（generic proxy）和异步代理之间的内存。

`cp{.reduce}.async.bulk` 操作完成后，紧接着会有一个隐式的通用代理与异步代理之间的屏障（generic-async proxy fence）。因此，异步操作的结果在完成后会立即对通用代理可见。为了等待 `cp{.reduce}.async.bulk` 指令的完成，必须使用 **异步组（async-group）** 或 **屏障（mbarrier）** 的完成机制。

## `__syncthreads` 和 `ptx::fence_proxy_async(ptx::space_shared)` 区别？

TMA 代码中：

```cpp
ptx::fence_proxy_async(ptx::space_shared);
__syncthreads();
```

这两个操作的功能不一样：

- `__syncthreads()` 确保了所有线程都执行了 `ptx::fence_proxy_async(ptx::space_shared)` fence 指令。
- 而 `ptx::fence_proxy_async(ptx::space_shared)` 确保了每个线程对 shared memory 的操作对 TMA engine（也即 async proxy）可见。

## cuda::barrier 和 cuda::pipeline 的区别

`cuda::Pipeline` 和 `cuda::barrier` 都是 CUDA 中的同步机制，但适用场景和功能侧重点不同：

1. `cuda::Pipeline` 主要用于 **任务的多阶段管理**，它是一种生产者-消费者模型的实现。Pipeline 提供了分阶段执行任务的能力，适合需要同时进行数据传输和计算的复杂场景。例如，在处理大规模数据时，可以通过 Pipeline 将数据传输、计算和结果输出划分为多个阶段，按照 FIFO 顺序依次完成，同时支持资源复用。
2. `cuda::barrier` 是一种 **简单同步工具**，用于确保线程在同一屏障点处保持一致。Barrier 的核心作用是同步所有线程，确保它们在继续执行后续任务之前完成当前工作，适合需要一次性同步的简单任务，比如全局内存和共享内存之间的单次数据传输。

两者的核心区别在于应用的复杂程度：`cuda::Pipeline` 更灵活且适用于复杂场景，而 `cuda::barrier` 简单高效，适合直接同步需求。

## 为什么 TMA load 和 store，一个用 mbarrier，一个用 fence？

`mbarrier` 能跟踪异步操作的完成，并通过事务字节数和阶段控制提供细粒度的同步。它不仅同步线程，还与硬件加载状态密切绑定，避免线程与硬件状态不一致的问题。与 `mbarrier` 不同，`fence` 不需要跟踪事务大小或硬件状态。它只需要确保内存操作的顺序，这对 TMA 存储的需求来说已经足够。
