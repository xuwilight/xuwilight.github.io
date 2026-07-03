---
title: CUDA C++ 笔记（十七）——异步数据复制 Asynchronous Data Copies
date: 2024-08-12 20:00:00
tags: [CUDA, CUDA C++]
categories: [CUDA C++ Programming Guide]
description: 介绍 memcpy_async API、复制与计算模式、使用 cuda::barrier 和 cuda::pipeline 的异步数据复制、单阶段与多阶段流水线、性能指南以及流水线原语接口。
---

# 异步数据复制 Asynchronous Data Copies

## Asynchronous Data Copies

CUDA 11 引入了 memcpy_async API 的异步数据操作，允许设备代码显式管理数据的异步复制。memcpy_async 功能使 CUDA 内核能够将计算与数据移动重叠。

### memcpy_async API

memcpy_async API 在 `cuda/barrier`、`cuda/pipeline` 和 `cooperative_groups/memcpy_async.h` 头文件中提供。

`cuda::memcpy_async` API 与 `cuda::barrier` 和 `cuda::pipeline` 同步原语配合使用，而 `cooperative_groups::memcpy_async` 使用 `cooperative_groups::wait` 进行同步。

这些 API 的语义非常相似：将对象从 `src` 复制到 `dst`，如同由另一个线程执行，复制完成后，可以通过 `cuda::pipeline`、`cuda::barrier` 或 `cooperative_groups::wait` 进行同步。

libcudacxx API 文档中提供了 `cuda::barrier` 和 `cuda::pipeline` 的 `cuda::memcpy_async` 重载的完整 API 文档以及一些示例。

`cooperative_groups::memcpy_async` 的 API 文档在"协作组"部分提供。

使用 `cuda::barrier` 和 `cuda::pipeline` 的 memcpy_async API 需要计算能力 7.0 或更高版本。在计算能力 8.0 或更高的设备上，从全局内存到共享内存的 memcpy_async 操作可以受益于硬件加速。

### Copy and Compute Pattern - Staging Data Through Shared Memory

CUDA 应用程序通常采用复制和计算模式：

1. 从全局内存中获取数据，
2. 将数据存储到共享内存，以及
3. 对共享内存数据执行计算，并可能将结果写回全局内存。

以下章节分别演示了如何在不使用和使用 memcpy_async 功能的情况下表达此模式：

1. 不使用 memcpy_async 的示例引入了一个计算与数据移动不重叠的示例，并使用中间寄存器复制数据。
2. 使用 memcpy_async 的示例改进了前面的示例，引入了 `memcpy_async` 和 `cuda::memcpy_async` API，可以直接将数据从全局内存复制到共享内存，而无需使用中间寄存器。
3. 使用 `cuda::barrier` 的异步数据复制展示了使用协作组和屏障的 memcpy 复制。
4. 使用 `cuda::pipeline` 的单阶段异步数据复制展示了使用单阶段流水线的 memcpy 复制。
5. 使用 `cuda::pipeline` 进行多阶段异步数据复制展示了具有多阶段流水线的 memcpy。

### Without memcpy_async

如果没有 memcpy_async，复制和计算模式的复制阶段将表示为 `shared[local_idx] = global[global_idx]`。此全局到共享内存的复制扩展为从全局内存读取到寄存器，然后从寄存器写入共享内存。

当此模式出现在迭代算法中时，每个线程块都需要在 `shared[local_idx] = global[global_idx]` 赋值后进行同步，以确保所有对共享内存的写入操作在计算阶段开始之前均已完成。线程块也需要在计算阶段结束后再次进行同步，以防止在所有线程完成计算之前覆盖共享内存。以下代码片段演示了此模式。

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

### With memcpy_async

使用 memcpy_async 后，从全局内存分配共享内存的操作 `shared[local_idx] = global_in[global_idx];` 将被从协作组执行的异步复制操作取代。

```cpp
cooperative_groups::memcpy_async(group, shared, global_in + batch_idx, sizeof(int) * block.size());
```

`cooperative_groups::memcpy_async` API 会将 `sizeof(int) * block.size()` 个字节的数据从全局内存（从 `global_in + batch_idx` 开始）复制到共享数据中。此操作如同由另一个线程执行，在复制完成后，该线程会与当前线程对 `cooperative_groups::wait` 的调用同步。在复制操作完成之前，修改全局数据或读写共享数据都会引发数据争用。

在计算能力为 8.0 或更高的设备上，memcpy_async 从全局到共享内存的传输可以受益于硬件加速，从而避免通过中间寄存器传输数据。

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

### Asynchronous Data Copies using cuda::barrier

`cuda::barrier` 的 `cuda::memcpy_async` 重载支持使用屏障同步异步数据传输。此重载执行复制操作，如同由绑定到屏障的另一个线程执行一样：在创建时递增当前阶段的预期计数，并在复制操作完成时递减该计数。这样，只有当所有参与屏障的线程都已到达，并且所有绑定到屏障当前阶段的 memcpy_async 都已完成时，屏障的阶段才会推进。以下示例使用一个块级屏障（所有块线程都参与），并将等待操作与屏障 `arrive_and_wait` 交换，同时提供与上一个示例相同的功能：

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

### Performance Guidance for memcpy_async

对于计算能力 8.x 的设备，流水线机制在同一个 CUDA Warp 中的 CUDA 线程之间共享。这种共享会导致 memcpy_async 的批次在 Warp 内纠缠在一起，在某些情况下会影响性能。

本节重点介绍 Warp 纠缠对提交、等待和到达操作的影响。请参阅"流水线接口"和"流水线原语接口"以了解各个操作的概述。

#### Alignment

在计算能力 8.0 的设备上，`cp.async` 系列指令允许将数据从全局内存异步复制到共享内存。这些指令支持一次复制 4、8 和 16 个字节。如果提供给 memcpy_async 的大小是 4、8 或 16 的倍数，并且传递给 memcpy_async 的两个指针都对齐到 4、8 或 16 的对齐边界，则 memcpy_async 可以使用纯异步内存操作来实现。

此外，为了在使用 memcpy_async API 时获得最佳性能，共享内存和全局内存都需要对齐到 128 字节。

对于指向对齐要求为 1 或 2 的类型值的指针，通常无法证明这些指针始终对齐到更高的对齐边界。确定 `cp.async` 指令是否可用必须推迟到运行时。执行此类运行时对齐检查会增加代码大小并增加运行时开销。

`cuda::aligned_size_t<size_t Align>(size_t size)` 可用于证明传递给 memcpy_async 的两个指针均已对齐到 `Align` 对齐边界，并且 `size` 是 `Align` 的倍数。只需将其作为参数传递即可，memcpy_async API 需要 Shape 即可：

```cpp
cuda::memcpy_async(group, dst, src, cuda::aligned_size_t<16>(N * block.size()), pipeline);
```

如果证明不正确，则行为未定义。

#### Trivially Copyable

在计算能力 8.0 的设备上，`cp.async` 系列指令允许将数据从全局内存异步复制到共享内存。如果传递给 memcpy_async 的指针类型不指向 TriviallyCopyable 类型，则需要调用每个输出元素的复制构造函数，并且这些指令不能用于加速 memcpy_async。

#### Warp Entanglement - Commit

memcpy_async 批处理的序列在 Warp 中共享。提交操作会进行合并，使得所有调用提交操作的收敛线程的序列都会增加一次。如果 Warp 完全收敛，则序列增加 1；如果 Warp 完全发散，则序列增加 32。

设 PB 为 Warp 共享流水线的实际批处理序列。

```
PB = {BP0, BP1, BP2, …, BPL}
```

令 TB 为线程感知的批次序列，假设该序列仅通过该线程调用提交操作而递增。

```
TB = {BT0, BT1, BT2, …, BTL}
```

`pipeline::producer_commit()` 的返回值来自线程感知的批次序列。

线程感知序列中的索引始终与实际 Warp 共享序列中相等或更大的索引对齐。只有当所有提交操作都从收敛线程调用时，序列才相等。

```
BTn ≡ BPm，其中 n <= m
```

例如，当 Warp 完全发散时：

Warp 共享流水线的实际序列为：`PB = {0, 1, 2, 3, …, 31}` (`PL=31`)。

此 Warp 中每个线程的感知序列如下：

线程 0：`TB = {0}` (`TL=0`)

线程 1：`TB = {0}` (`TL=0`)

…

线程 31：`TB = {0}` (`TL=0`)

总之，不要在条件语句里提交 commit。

建议通过聚合线程执行提交和到达操作：

通过保持线程感知的批次序列与实际序列一致，避免过度等待，并尽量减少对屏障对象的更新。

当这些操作之前的代码使线程发散时，应在调用提交或到达操作之前通过 `__syncwarp` 重新聚合 Warp。

## Asynchronous Data Copies using cuda::pipeline

CUDA 提供了 `cuda::pipeline` 同步对象来管理和将异步数据移动与计算重叠。

`cuda::pipeline` 的 API 文档位于 libcudacxx API 中。管道对象是一个双端 N 级队列，具有头部和尾部，用于按先进先出 (FIFO) 顺序处理工作。管道对象具有以下成员函数来管理管道的各个阶段。

### Single-Stage Asynchronous Data Copies using cuda::pipeline

在之前的示例中，我们展示了如何使用 cooperative_groups 和 `cuda::barrier` 进行异步数据传输。在本节中，我们将使用单阶段的 `cuda::pipeline` API 来调度异步复制。稍后我们将扩展此示例，以展示多阶段重叠计算和复制。

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

### Multi-Stage Asynchronous Data Copies using cuda::pipeline

在前面使用 `cooperative_groups::wait` 和 `cuda::barrier` 的示例中，内核线程会立即等待数据传输到共享内存完成。这避免了将数据从全局内存传输到寄存器，但不会通过重叠计算来隐藏 memcpy_async 操作的延迟。

为此，我们在以下示例中使用了 CUDA 流水线功能。它提供了一种管理 memcpy_async 批次序列的机制，使 CUDA 内核能够将内存传输与计算重叠。以下示例实现了一个将数据传输与计算重叠的两级流水线。它：

1. 初始化流水线共享状态（详见下文）
2. 通过为第一个批次调度 memcpy_async 来启动流水线。
3. 循环遍历所有批次：它会为下一个批次调度 memcpy_async，在上一个批次的 memcpy_async 完成后阻塞所有线程，然后将上一个批次的计算与下一个批次的内存异步复制重叠。
4. 最后，它会通过执行最后一个批次的计算来耗尽流水线。

请注意，为了与 `cuda::pipeline` 互操作，此处使用了 `cuda/pipeline` 头文件中的 `cuda::memcpy_async`。

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

管道对象是一个双端队列，有头有尾，用于按先进先出 (FIFO) 的顺序处理工作。生产者线程将工作提交到管道的头部，而消费者线程则从管道的尾部获取工作。在上面的示例中，所有线程既是生产者线程，又是消费者线程。这些线程首先提交 memcpy_async 操作以获取下一批数据，同时等待上一批 memcpy_async 操作完成。

将工作提交到管道阶段包括：

1. 使用 `pipeline.producer_acquire()` 从一组生产者线程集体获取管道头部。
2. 将 memcpy_async 操作提交到管道头部。
3. 使用 `pipeline.producer_commit()` 集体提交（推进）管道头部。

使用先前提交的阶段包括：

1. 集体等待该阶段完成，例如，使用 `pipeline.consumer_wait()` 等待尾部（最旧的）阶段。
2. 使用 `pipeline.consumer_release()` 集体释放阶段。
3. `cuda::pipeline_shared_state<scope, count>` 封装了允许流水线处理最多 `count` 个并发阶段的有限资源。如果所有资源都已使用，`pipeline.producer_acquire()` 会阻塞生产者线程，直到下一个流水线阶段的资源被消费者线程释放。

此示例可以更简洁地编写，将循环的序言和结尾与循环本身合并，如下所示：

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

上面使用的 `pipeline<thread_scope_block>` 原语非常灵活，并且支持上述示例中未使用的两个特性：块中任意数量的线程子集都可以参与管道；参与的线程中，任意子集都可以是生产者、消费者，或者两者兼而有之。在以下示例中，线程等级为"偶数"的线程为生产者，而其他线程为消费者：

```cpp
__device__ void compute(int* global_out, int shared_in);

template <size_t stages_count = 2>
__global__ void with_specialized_staging_unified(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();

    // In this example, threads with "even" thread rank are producers, while threads with "odd" thread rank are consumers:
    const cuda::pipeline_role thread_role
      = block.thread_rank() % 2 == 0? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;

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

例如，当所有线程同时作为生产者和消费者时，管道会执行一些优化，但通常情况下，支持所有这些特性的成本无法完全消除。例如，管道会在共享内存中存储并使用一组屏障进行同步，但如果块中的所有线程都参与管道，则同步并非必需。

对于块中的所有线程都参与管道的特殊情况，我们可以通过将 `pipeline<thread_scope_thread>` 与 `__syncthreads()` 结合使用来比 `pipeline<thread_scope_block>` 做得更好：

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

如果计算操作仅读取与当前线程位于同一个 Warp 中的其他线程写入的共享内存，则 `__syncwarp()` 就足够了。

### Pipeline Interface

libcudacxx API 文档中提供了 `cuda::memcpy_async` 的完整 API 文档以及一些示例。

流水线接口要求至少 CUDA 11.0，至少兼容 ISO C++ 2011（例如，使用 `-std=c++11` 进行编译），以及 `#include <cuda/pipeline>`。

对于类 C 接口，在编译时不使用 ISO C++ 2011 兼容性，请参阅流水线原语接口。

### Pipeline Primitives Interface

流水线原语是 memcpy_async 功能的类似 C 语言的接口。可以通过包含 `<cuda_pipeline.h>` 头文件来使用流水线原语接口。在编译时不使用 ISO C++ 2011 兼容性，请包含 `<cuda_pipeline_primitives.h>` 头文件。

#### memcpy_async 原语

```cpp
void __pipeline_memcpy_async(void* __restrict__ dst_shared,
                             const void* __restrict__ src_global,
                             size_t size_and_align,
                             size_t zfill=0);
```

请求提交以下操作进行异步执行：

```cpp
size_t i = 0;
for (; i < size_and_align - zfill; ++i) ((char*)dst_shared)[i] = ((char*)src_global)[i]; /* copy */
for (; i < size_and_align; ++i) ((char*)dst_shared)[i] = 0; /* zero-fill */
```

要求：

1. `dst_shared` 必须是指向 memcpy_async 共享内存目标的指针。
2. `src_global` 必须是指向 memcpy_async 的全局内存源的指针。
3. `size_and_align` 必须是 4、8 或 16。
4. `zfill <= size_and_align`。
5. `size_and_align` 必须是 `dst_shared` 和 `src_global` 的对齐方式。

任何线程在等待 memcpy_async 操作完成之前修改源内存或观察目标内存都会导致竞争条件。在提交 memcpy_async 操作和等待其完成之间，以下任何操作都会引发竞争条件：

1. 从 `dst_shared` 加载。
2. 存储到 `dst_shared` 或 `src_global`。
3. 对 `dst_shared` 或 `src_global` 应用原子更新。

#### 提交原语

```cpp
void __pipeline_commit();
```

将已提交的 memcpy_async 操作作为当前批次提交到流水线。

#### 等待原语

```cpp
void __pipeline_wait_prior(size_t N);
```

令 `{0, 1, 2, ..., L}` 为与给定线程调用 `__pipeline_commit()` 关联的索引序列。等待至少 `L-N` 个批次的完成（包括 `L-N` 个批次）。

#### 到达屏障原语

```cpp
void __pipeline_arrive_on(__mbarrier_t* bar);
```

`bar` 指向共享内存中的屏障。

将屏障到达计数加一，当在此调用之前排序的所有 memcpy_async 操作都完成后，到达计数减一，因此到达计数的净效应为零。用户有责任确保到达计数的增量不超过 `__mbarrier_maximum_count()`。
