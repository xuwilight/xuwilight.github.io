---
title: CUDA C++ 笔记（十六）——异步屏障 Asynchronous Barrier
date: 2024-08-08 20:00:00
tags: [CUDA, CUDA C++]
categories: [CUDA C++ Programming Guide]
description: 介绍 cuda::barrier 的使用模式，包括简单同步、时间分割的五阶段同步、空间分区（Warp 特化）、提前退出、完成函数以及 mbarrier 原语接口。
---

# 异步屏障 Asynchronous Barrier

## 10.26. Asynchronous Barrier

NVIDIA C++ 标准库引入了 std::barrier 的 GPU 实现。除了 std::barrier 的实现之外，该库还提供了扩展，允许用户指定屏障对象的范围。屏障 API 的范围在"线程范围"下有文档说明。计算能力 8.0 或更高的设备为屏障操作提供硬件加速，并将这些屏障与 memcpy_async 功能集成。在计算能力低于 8.0 但从 7.0 开始的设备上，这些屏障无需硬件加速即可使用。

nvcuda::experimental::awbarrier 已弃用，取而代之的是 cuda::barrier。

### Simple Synchronization Pattern

没有到达/等待障碍，使用 `__syncthreads()`（同步块中的所有线程）或 `group.sync()`（在使用协作组时）实现同步。

```cpp
#include <cooperative_groups.h>

__global__ void simple_sync(int iteration_count) {
    auto block = cooperative_groups::this_thread_block();

    for (int i = 0; i < iteration_count; ++i) {
        /* code before arrive */
        block.sync(); /* wait for all threads to arrive here */
        /* code after wait */
    }
}
```

线程在同步点 (`block.sync()`) 处被阻塞，直到所有线程都到达同步点。此外，同步点之前发生的内存更新保证对同步点之后的块中的所有线程可见，即相当于 `atomic_thread_fence(memory_order_seq_cst, thread_scope_block)` 以及同步。

此模式包含三个阶段：

1. 同步点之前的代码执行将在同步点之后读取的内存更新。
2. 同步点。
3. 同步点之后的代码，同步点之前发生的内存更新可见。

### Temporal Splitting and Five Stages of Synchronization

使用 std::barrier 的时间分割同步模式如下。

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

__device__ void compute(float* data, int curr_iteration);

__global__ void split_arrive_wait(int iteration_count, float *data) {
    using barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ barrier bar;
    auto block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0) {
        init(&bar, block.size()); // Initialize the barrier with expected arrival count
    }
    block.sync();

    for (int curr_iter = 0; curr_iter < iteration_count; ++curr_iter) {
        /* code before arrive */
        barrier::arrival_token token = bar.arrive(); /* this thread arrives. Arrival does not block a thread */
        compute(data, curr_iter);
        bar.wait(std::move(token)); /* wait for all threads participating in the barrier to complete bar.arrive()*/
        /* code after wait */
    }
}
```

在此模式中，同步点 (`block.sync()`) 被拆分为到达点 (`bar.arrive()`) 和等待点 (`bar.wait(std::move(token))`)。线程首次调用 `bar.arrive()` 时即开始参与 cuda::barrier。当线程调用 `bar.wait(std::move(token))` 时，它将被阻塞，直到参与线程完成 `bar.arrive()` 的预期次数（该次数由传递给 `init()` 的预期到达计数参数指定）。参与线程调用 `bar.arrive()` 之前发生的内存更新，保证在参与线程调用 `bar.wait(std::move(token))` 之后对其可见。请注意，调用 `bar.arrive()` 不会阻塞线程，它可以继续执行其他工作，这些工作不依赖于其他参与线程调用 `bar.arrive()` 之前发生的内存更新。

到达后等待模式包含五个阶段，这些阶段可以迭代重复：

1. 到达之前的代码执行内存更新，这些更新将在等待之后读取。
2. 到达点，带有隐式内存栅栏（即相当于 `atomic_thread_fence(memory_order_seq_cst, thread_scope_block)`）。
3. 到达和等待之间的代码。
4. 等待点。
5. 等待之后的代码，可以看到到达之前执行的更新。

### Bootstrap Initialization, Expected Arrival Count, and Participation

任何线程开始参与 cuda::barrier 之前都必须进行初始化。

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

__global__ void init_barrier() {
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    auto block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0) {
        init(&bar, block.size()); // Single thread initializes the total expected arrival count.
    }
    block.sync();
}
```

在任何线程参与 cuda::barrier 之前，必须使用 `init()` 初始化屏障，并设置预期到达计数（本例中为 `block.size()`）。初始化必须在任何线程调用 `bar.arrive()` 之前进行。这带来了引导方面的挑战，因为线程必须在参与 cuda::barrier 之前同步，但线程创建 cuda::barrier 是为了同步。在本例中，参与的线程属于一个协作组，并使用 `block.sync()` 来引导初始化。在本例中，整个线程块都参与初始化，因此也可以使用 `__syncthreads()`。

`init()` 的第二个参数是预期到达计数，即参与线程在解除 `bar.wait(std::move(token))` 调用阻塞之前，调用 `bar.arrive()` 的次数。在前面的示例中，cuda::barrier 使用线程块中的线程数（即 `cooperative_groups::this_thread_block().size()`）进行初始化，并且线程块内的所有线程都参与屏障。

cuda::barrier 可以灵活地指定线程如何参与（拆分到达/等待）以及哪些线程参与。相比之下，协作组中的 `this_thread_block.sync()` 或 `__syncthreads()` 适用于整个线程块，而 `__syncwarp(mask)` 是 Warp 的指定子集。如果用户的目的是同步整个线程块或整个 Warp，出于性能方面的考虑，我们建议分别使用 `__syncthreads()` 和 `__syncwarp(mask)`。

### A Barrier's Phase: Arrival, Countdown, Completion, and Reset

当参与线程调用 `bar.arrive()` 时，cuda::barrier 会从预期到达计数倒计时至零。当倒计时达到零时，cuda::barrier 在当前阶段的执行完成。如果最后一次调用 `bar.arrive()` 导致倒计时达到零，则倒计时将自动且原子地重置。重置操作会将倒计时赋值为预期到达计数，并将 cuda::barrier 移至下一个阶段。

`token = bar.arrive()` 返回的 `cuda::barrier::arrival_token` 类的 token 对象与屏障的当前阶段相关联。当 cuda::barrier 处于当前阶段时（即，与 token 关联的阶段与 cuda::barrier 的阶段匹配时），调用 `bar.wait(std::move(token))` 会阻塞调用线程。如果在调用 `bar.wait(std::move(token))` 之前阶段已推进（因为倒计时已达到零），则线程不会阻塞；如果在调用 `bar.wait(std::move(token))` 时线程被阻塞，而阶段已推进，则线程将被解除阻塞。

了解何时可能发生或不可能发生重置至关重要，尤其是在复杂的到达/等待同步模式中。

1. 线程对 `token = bar.arrive()` 和 `bar.wait(std::move(token))` 的调用必须按顺序进行，以便 `token = bar.arrive()` 在 cuda::barrier 的当前阶段发生，而 `bar.wait(std::move(token))` 在相同或下一个阶段发生。
2. 线程对 `bar.arrive()` 的调用必须在屏障计数器非零时进行。屏障初始化后，如果线程调用 `bar.arrive()` 导致倒计时归零，则必须先调用 `bar.wait(std::move(token))`，屏障才能被重用用于后续的 `bar.arrive()` 调用。`bar.wait()` 只能使用当前阶段或前一个阶段的 token 对象来调用。对于 token 对象的任何其他值，其行为均未定义。

对于简单的到达/等待同步模式，遵守这些使用规则非常简单。

### Spatial Partitioning (also known as Warp Specialization)

线程块可以进行空间分区，以便 Warp 专门执行独立计算。空间分区用于生产者或消费者模式，其中一个线程子集生成数据，并由另一个（不相交的）线程子集并发消费。

生产者/消费者空间分区模式需要两个单边同步来管理生产者和消费者之间的数据缓冲区。

生产者线程等待消费者线程发出缓冲区已准备好填充的信号；然而，消费者线程不会等待此信号。消费者线程等待生产者线程发出缓冲区已填满的信号；然而，生产者线程不会等待此信号。为了实现完全的生产者/消费者并发，此模式至少具有双缓冲，其中每个缓冲区需要两个 cuda::barrier。

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;

__device__ void producer(barrier ready[], barrier filled[], float* buffer, float* in, int N, int buffer_len)
{
    for (int i = 0; i < (N/buffer_len); ++i) {
        ready[i%2].arrive_and_wait(); /* wait for buffer_(i%2) to be ready to be filled */
        /* produce, i.e., fill in, buffer_(i%2) */
        barrier::arrival_token token = filled[i%2].arrive(); /* buffer_(i%2) is filled */
    }
}

__device__ void consumer(barrier ready[], barrier filled[], float* buffer, float* out, int N, int buffer_len)
{
    barrier::arrival_token token1 = ready[0].arrive(); /* buffer_0 is ready for initial fill */
    barrier::arrival_token token2 = ready[1].arrive(); /* buffer_1 is ready for initial fill */
    for (int i = 0; i < (N/buffer_len); ++i) {
        filled[i%2].arrive_and_wait(); /* wait for buffer_(i%2) to be filled */
        /* consume buffer_(i%2) */
        barrier::arrival_token token = ready[i%2].arrive(); /* buffer_(i%2) is ready to be re-filled */
    }
}

// N is the total number of float elements in arrays in and out
__global__ void producer_consumer_pattern(int N, int buffer_len, float* in, float* out) {

    // Shared memory buffer declared below is of size 2 * buffer_len
    // so that we can alternatively work between two buffers.
    // buffer_0 = buffer and buffer_1 = buffer + buffer_len
    __shared__ extern float buffer[];

    // bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
    // while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively
    __shared__ barrier bar[4];


    auto block = cooperative_groups::this_thread_block();
    if (block.thread_rank() < 4)
        init(bar + block.thread_rank(), block.size());
    block.sync();

    if (block.thread_rank() < warpSize)
        producer(bar, bar+2, buffer, in, N, buffer_len);
    else
        consumer(bar, bar+2, buffer, out, N, buffer_len);
}
```

在此示例中，第一个 Warp 被指定为生产者，其余 Warp 被指定为消费者。所有生产者和消费者线程都会参与（调用 `bar.arrive()` 或 `bar.arrive_and_wait()`）四个 cuda::barrier 中的每一个，因此预期到达计数等于 `block.size()`。

生产者线程等待消费者线程发出共享内存缓冲区可以填满的信号。为了等待 cuda::barrier，生产者线程必须首先到达该 `ready[i%2].arrive()` 以获取令牌，然后使用该令牌执行 `ready[i%2].wait(token)`。为简单起见，`ready[i%2].arrive_and_wait()` 结合了这两个操作。

```cpp
bar.arrive_and_wait();
/* is equivalent to */
bar.wait(bar.arrive());
```

生产者线程计算并填充就绪缓冲区，然后通过到达已填充的屏障 (`filled[i%2].arrive()`) 发出缓冲区已填充的信号。生产者线程此时不会等待，而是等待下一次迭代的缓冲区（双缓冲）准备好填充。

消费者线程首先发出两个缓冲区都已准备好填充的信号。消费者线程此时不会等待，而是等待本次迭代的缓冲区被填充 (`filled[i%2].arrive_and_wait()`)。消费者线程消耗完缓冲区后，会再次发出缓冲区已准备好填充的信号 (`ready[i%2].arrive()`)，然后等待下一次迭代的缓冲区被填充。

### Early Exit (Dropping out of Participation)

当参与同步序列的线程必须提前退出时，该线程必须在退出前明确退出。其余参与线程可以正常进行后续的 cuda::barrier 到达和等待操作。

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

__device__ bool condition_check();

__global__ void early_exit_kernel(int N) {
    using barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ barrier bar;
    auto block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0)
        init(&bar, block.size());
    block.sync();

    for (int i = 0; i < N; ++i) {
        if (condition_check()) {
            bar.arrive_and_drop();
            return;
        }
        /* other threads can proceed normally */
        barrier::arrival_token token = bar.arrive();
        /* code between arrive and wait */
        bar.wait(std::move(token)); /* wait for all threads to arrive */
        /* code after wait */
    }
}
```

此操作到达 cuda::barrier 以履行参与线程在当前阶段到达的义务，然后减少下一阶段的预期到达计数，以便该线程不再预期到达屏障。

### Completion Function

`cuda::barrier<Scope, CompletionFunction>` 的 CompletionFunction 在每个阶段执行一次，在最后一个线程到达之后，任何线程解除等待之前。在该阶段到达屏障的线程执行的内存操作对于执行 CompletionFunction 的线程可见，并且 CompletionFunction 中执行的所有内存操作对于所有在屏障处等待的线程在解除等待后也可见。

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <functional>
namespace cg = cooperative_groups;

__device__ int divergent_compute(int*, int);
__device__ int independent_computation(int*, int);

__global__ void psum(int* data, int n, int* acc) {
  auto block = cg::this_thread_block();

  constexpr int BlockSize = 128;
  __shared__ int smem[BlockSize];
  assert(BlockSize == block.size());
  assert(n % 128 == 0);

  auto completion_fn = [&] {
    int sum = 0;
    for (int i = 0; i < 128; ++i) sum += smem[i];
    *acc += sum;
  };

  // Barrier storage
  // Note: the barrier is not default-constructible because
  //       completion_fn is not default-constructible due
  //       to the capture.
  using completion_fn_t = decltype(completion_fn);
  using barrier_t = cuda::barrier<cuda::thread_scope_block,
                                 completion_fn_t>;
  __shared__ std::aligned_storage<sizeof(barrier_t),
                                  alignof(barrier_t)> bar_storage;

  // Initialize barrier:
  barrier_t* bar = (barrier_t*)&bar_storage;
  if (block.thread_rank() == 0) {
    assert(*acc == 0);
    assert(blockDim.x == blockDim.y == blockDim.y == 1);
    new (bar) barrier_t{block.size(), completion_fn};
    // equivalent to: init(bar, block.size(), completion_fn);
  }
  block.sync();

  // Main loop
  for (int i = 0; i < n; i += block.size()) {
    smem[block.thread_rank()] = data[i] + *acc;
    auto t = bar->arrive();
    // We can do independent computation here
    bar->wait(std::move(t));
    // shared-memory is safe to re-use in the next iteration
    // since all threads are done with it, including the one
    // that did the reduction
  }
}
```

### Memory Barrier Primitives Interface

#### Data Types

```cpp
typedef /* implementation defined */ __mbarrier_t;
typedef /* implementation defined */ __mbarrier_token_t;
```

#### Memory Barrier Primitives API

```cpp
uint32_t __mbarrier_maximum_count();
void __mbarrier_init(__mbarrier_t* bar, uint32_t expected_count);
```

1. `bar` 必须是指向 `__shared__` 内存的指针。
2. `expected_count <= __mbarrier_maximum_count()`。
3. 将当前阶段和下一阶段的预期到达计数 `*bar` 初始化为 `expected_count`。

```cpp
void __mbarrier_inval(__mbarrier_t* bar);
```

1. `bar` 必须是指向共享内存中 mbarrier 对象的指针。
2. 必须先使 `*bar` 失效，才能重新利用相应的共享内存。

```cpp
__mbarrier_token_t __mbarrier_arrive(__mbarrier_t* bar);
```

1. `*bar` 的初始化必须在本次调用之前进行。
2. 待处理计数不得为零。
3. 以原子方式减少屏障当前阶段的待处理计数。
4. 返回与减少前屏障状态关联的到达令牌。

```cpp
__mbarrier_token_t __mbarrier_arrive_and_drop(__mbarrier_t* bar);
```

1. `*bar` 的初始化必须在本次调用之前进行。
2. 待处理计数不得为零。
3. 以原子方式减少屏障当前阶段的待处理计数和下一阶段的预期计数。
4. 返回与减少值之前的屏障状态关联的到达令牌。

```cpp
bool __mbarrier_test_wait(__mbarrier_t* bar, __mbarrier_token_t token);
```

`token` 必须与 `*bar` 的紧接前一个阶段或当前阶段相关联。

如果 `token` 与 `*bar` 的紧接前一个阶段相关联，则返回 `true`，否则返回 `false`。
