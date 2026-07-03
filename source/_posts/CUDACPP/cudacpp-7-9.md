---
title: CUDA C++ 笔记（四）第7章——第9章 Hardware and Performance Guidelines
date: 2024-06-13 20:00:00
tags: [CUDA, Performance, CUDA C++]
categories: [CUDA C++ Programming Guide]
description: 本篇涵盖第 7 到第 9 章，包括 Hardware Implementation（硬件结构、SIMT 架构、层级结构）、Performance Guidelines（最大化并行度、最大化内存吞吐量、最大化指令吞吐量、最小化内存抖动）以及 CUDA-Enabled GPUs。
---

## Chapter 7. Hardware Implementation

### 硬件结构

![H100 GPU 架构](/assets/cudacpp-7-9/image.png)

GPU 的基本架构如图所示。图上是 H100 SXM 型号的 GPU 架构，从图中可以看到有很多个 SM（SM：Streaming Multiprocessors），此外还有所有 SM 共享的 50M 的 L2 cache，80 GB 的 HBM3 device memory，以及 NVLink 和 PCIe 等接口。

下图是 Hopper 架构上的 SM，一个 SM 的结构被分为了 4 个 smsp（SM sub partitions）。

一个 SM 一共有 128 个 FP32 计算单元，64 个 FP64 计算单元，4 个 Tensor Core，以及 16 个 SFU（special function units）单元。

其中每个 smsp 有一个 warp 调度器（warp schedulers），用来管理和调度 warp 的执行。此外还有一共有 32 个 LD/ST 单元，用于数据的读写。

内存方面，一个 SM 共有 64K 的寄存器，256KB 的 L1 cache，最高可分配到 228KB 的 shared memory。以及 Hopper 架构上新出的 TMA（Tensor Memory Accelerator）结构。

- 寄存器文件（Register File）：用于存储线程的局部变量和中间结果，每个线程都有自己的寄存器。
- L1 缓存（L1 Cache）：用于缓存全局内存和常量内存的数据，以减少对全局内存的访问。
- 共享内存（Shared Memory）：用于线程块内线程之间的数据交换和协作，L1 Cache 和共享内存的访问速度非常快，仅次于寄存器。

![Hopper SM 结构](/assets/cudacpp-7-9/image1.png)

### SIMT 架构

GPU 程序在运行时，每个 SM 上会并发执行数百个线程。为了管理如此大量的线程，它采用了一种称为 SIMT（Single-Instruction, Multiple-Thread：单指令多线程）的执行架构。

在 GPU 中，最小的执行单元不是线程，而是线程束 warp，一个 warp 由 32 个线程组成，一个 warp 中的 32 个线程执行相同的指令，所以称为单指令多线程。

warp 在 SM 上由 warp 调度器进行管理和调度。每个 warp 会处于阻塞或就绪等状态，warp 调度器会选择处于就绪状态的 warp 执行指令。当 warp 处于阻塞状态时，调度器会调度其它 warp 到 SM 上运行。当等待的 warp 等到了所需的元素后，会再次处于就绪状态，等待调度器调度运行。

#### 线程束分化

SM 以一个线程束为基本单位进行计算，每个线程束中的线程执行相同的指令。如果在一个线程束中的线程存在条件分支，那么这个线程束会执行所有的语句，导致性能下降，这就是线程束分化。

当线程束中的线程必须要执行不同的条件分支时，满足分支条件的线程会被激活并执行分支内的内容，不满足分支条件的线程会接收同样的指令，但不会被激活，不会实际执行，但也不能跳过去执行其他指令。

换言之，当线程束中的线程遇到分支时，不论线程是否需要执行分支，都会消耗执行该分支的时间，如果分支过多就会导致性能的严重下降，因此在编程的过程中要避免同一个线程束内的线程分化。

但是从 Volta 架构开始，Independent Thread Scheduling 被引入，线程束内的线程不再完全同步。遇到分支时，不再像之前的架构一样，只有线程束内的线程条件一致时，才会跳过分支；Volta 架构的调度优化器会将线程束中的线程，按照分支条件是否满足，重新组合成 SIMT 单元，从而跳过分支。

### 层级结构（Hierarchy）

在 CUDA 编程模型中可以抽象出三个层级，分别是 Grid，Thread Block 和 Thread，如下图所示。

Grid 对应硬件上的整个 GPU Device。一个 kernel 在许多 thread block 组成的 grid 上运行。

thread block 对应硬件上的 SM，一个 thread block 包括多个 thread。thread block 在 SM 上执行，多个并发的 thread blocks 可以在一个 SM 上执行。一个 thread block 可以被分发到任意的 SM 上执行，当分发到 SM 上时，一个 thread block 在它的生命周期内只能在一个 SM 上执行。

thread 对应硬件上的计算单元，如 cuda core，Tensor core 等。每个线程在一个 cuda core 上执行。Tensor core 可能需要多个线程共同执行。

SM 中每个 Warp 的执行上下文（execution context：如程序计数器、寄存器等）在 Warp 的整个生命周期内都保存在芯片上。因此，从一个 warp 切换到另一个 warp 执行几乎是没有成本的。

当内核启动时，会根据线程块（thread block）所需的资源（寄存器和共享内存），决定将线程块调度到哪个 SM 上运行。只要资源足够，一个 SM 上可以同时运行多个线程块。当线程块运行完毕时，线程块会退出 SM，让其他线程块被调度上去。

如果线程块需要的寄存器或共享内存太多，以至于 SM 连一个线程块都无法满足的时候，内核会启动失败。

一个 SM 上被分配多少个线程块和 warp 取决于 SM 中可用的寄存器和共享内存，以及内核需要的寄存器和共享内存大小。

一个 block 中的 warp 总数可以按照下面的公式计算：

```
warps per block = ceil(threads per block, warp size)
```

其中：

- threads per block 是每个 block 中的线程数。
- warp size 是 warp 的大小，默认为 32。
- `ceil(x, y)` 是向上取整。

举个例子，当我们需要计算 N=1000 的向量相加时，如果 block size 设置成 100，那么需要 10 个 thread block。

按照上面的计算公式，一个 block 中的 warp 数是 4，也就是 128 个线程。因为 warp 是最小的执行单位，即使 thread 的数量不是 32 的倍数，设备也会用 inactive 的线程凑到 32 的倍数，所以线程利用率是 1000/1280 = 78.125%。

当 block size 设成 128 时，只需要 8 个 thread block 就可以，此时，前 7 个 block 的所有线程都可以完全利用，只有最后一个 block 的线程有 24 个没用到，所以利用率是 1000/1024 = 97.656%。

与线程层级相对应的还有内存的层级。

如图所示，GPU 中的内存结构由寄存器，L1 cache，L2 cache 和 Device memory 组成，他们的速度从快到慢，容量从小到大。

- Device memory 和 L2 cache 所有的 thread blocks 都可以访问，容量大，速度慢。
- 每个 thread block 有自己的 L1 cache，其中 L1 cache 又可以划分出 shared memory，一个 thread block 中的所有线程都可以访问。
- 对于每一个线程，都有自己的寄存器文件。当寄存器溢出时会分配私有的 local memory。需要注意的是，local memory 在 Device memory 上，速度和 global memory 相同，所以要尽可能避免寄存器溢出。

同样的，当我们进行线程间的同步的时候也是按照 grid，thread block 和 warp 三个等级进行的。从 warp 到 grid，同步的成本会越来越大。

## Chapter 8. Performance Guidelines

### GPU 性能优化概述

GPU 的性能优化主要有四个方面：

1. 提高并行度来提高硬件单元的利用率。
2. 优化内存的使用，提高内存带宽利用率。
3. 优化指令的使用，实现最大的指令吞吐量。
4. 尽量减少内存抖动（memory thrashing）。

程序的性能瓶颈可以简单分为 compute-bound、memory-bound 和 latency-bound 三种，当我们进行优化时首先要确定程序的主要瓶颈在哪里，然后进行针对性优化。

可以通过 Nsight system 和 Nsight compute 等软件来确定性能瓶颈。也可以将程序的运算吞吐量或内存吞吐量与设备的相应峰值理论吞吐量进行比较，确定内核还有多少改进空间。

### 最大化并行度

提高程序的并行度体现在不同的等级上。

#### 应用级别（Application Level）

在应用级别，我们可以通过异步函数来提高主机和 GPU 间的并行度，比如使用异步读写操作避免 CPU 被 GPU 阻塞。同时，应根据不同处理器的优势分配任务：将串行任务交给主机，将并行任务交给设备。

#### 设备级别（Device Level）

在设备级别并行可以通过把任务分配到不同的 GPU 上来提高设备间的并行度，如使用 DP 等。或者启动多个 stream 来提高设备利用率，如将任务的计算和通信分配在不同的 stream 上。

#### SM 级别（Multiprocessor Level）

因为 kernel 主要在 SM 上运行，下面主要介绍 SM 级别的并行。

从前面的介绍可以知道，在一个 SM 上，warp 调度器通过对 warp 进行调度来执行指令。

在 Hopper 架构中，每个 SM 有 4 个 warp 调度器，每个调度器有 16 个 warp slots，所以一个 SM 可以最多管理 64 个 warp。

每个调度器在每个时钟周期可以发射一个 warp。每个 warp 在 warp slots 里有三种状态：stalled，eligible 和 selected。

- stalled 表示当前 warp 正被阻塞，可能是有数据依赖没完成，也可能是正在等待同步的完成。
- eligible 表示 warp 可以执行指令。
- selected 表示 warp 已经被选中执行指令了。

下图是一个 warp 调度器执行的简单说明。一个 warp 调度器用 warp slots 管理 warp，用 issue slot 发射 warp 指令。

假如在第 N 个 cycle，warp slots 里有 3 个 warp，其中前 2 个处于阻塞状态，第三个处于 eligible 状态，此时 warp 调度器选择第 3 个位置的 warp 执行，则第 3 个 warp 在该周期内变为 selected 状态。

下一个周期，所有的 warp 都在执行，没有可发射的 warp，issue slot 就会处于空置状态。N+2 周期，第 1 个 warp 变为 eligible，因此被选中执行，变为 selected 状态。后面所有的 warp 又变为 stalled 状态，而且随着执行的完成，warp 会逐渐退出。

所以，如果一个 SM 上有更多的 active warp 的话，那么每周期内都可以有 warp 执行指令，就可以减少 issue slots 的空置时间以及隐藏其他 warp 因 stalled 导致的 latency。

那么如何才能提高 SM 上活跃 warp 的数量呢，主要有两点：1. 提高指令级别的并行 **Instruction-Level Parallelism**（ILP），2. 提高线程级别的并行，**Thread-Level Parallelism**（TLP）。

指令级别的并行度是指每个线程有更多的独立指令。这样一个 warp 在执行完一个指令后不会进入 stalled 状态，还可以执行其他的独立指令。

以计算两个向量 a 和 b 的点乘为例，首先需要加载 a 和 b，然后计算 a*b，最后把结果保存到 c 中。在这个过程中一共有 3 次读内存，1 次计算和 1 次写内存。

```cpp
__global__ void kernel(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] += a[idx] * b[idx];
}
```

对于读写指令 LDS 和 STS，假设 1 个 cycle 可以发射 1 条读写指令，执行完成需要 1000 cycle。对于 FMA 指令，假设发射一个需要 2 个 cycle，执行完成需要 4 个 cycle。所以运行一共需要 1006 个 cycle。此时一个 thread 只有一个 FMA 指令。

如果一个 thread 执行两个乘加指令，就会变成下面这样。一共只需要 1008 个 cycle 就能计算完成，性能几乎提高 2 倍。

```cpp
#define THREAD_BLOCK_DIM 128
__global__ void kernel(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int off = 2 * THREAD_BLOCK_DIM * blockIdx.x + threadIdx.x;

    #pragma unroll 2
    for (int i = 0; i < 2; i++) {
        const int idx = off + i * THREAD_BLOCK_DIM;
        c[idx] += a[idx] * b[idx];
    }
}
```

提高线程级别的并行就是增加线程数。线程数增加了 warp 就会增加，每个 SM 上可以调度的 warp 就会增加，所以就可以提高整个程序的并发度。

但是，线程数并不能无限制地增加，一个 SM 上可以支持的最大的 warp 数量也是有限的，主要由设备的计算能力和代码的实现决定。

硬件上，对于 Hopper，每个 thread block 最大支持 1024 个线程，每个 SM 最大支持 64 个 warps，2048 个线程。代码实现上，kernel 需要的寄存器和共享内存的数量，也会影响 SM 上 warp 的数量。

我们可以使用 warp 占有率来衡量一个 SM 上 warp 的占比，`occupancy = achievable / device`。

在 Hopper 架构上，每个 SM 最大有 65536 个寄存器，分配的时候按照 256 的 chunk 分配。比如：一个 kernel 的每个线程使用 63 个寄存器，所以一个 warp 需要 63*32 = 2016 个寄存器。申请的时候按照 256 的倍数会申请 2048 个寄存器。所以 SM 上可以存在 65536/2048 = 32 个 warp。由于 Hopper 一个 SM 最高支持 64 个 warp，所以 occupancy = 32/64 = 50%。

对于 shared memory。Hopper 上一个 SM 上最高可以分配 228KB 的 shared memory，其中 1KB 会被系统保留。比如：一个 kernel 在一个有 128 线程的 block 上使用 17408 bytes 的 shared memory，所以一个 SM 可以有 233472 / (17408 + 1024) = 12.66 = 12 个 block。achievable active warps = 12*128/32 = 48，占有率是 48/64 = 75%。

但是 warp 的占有率也不是越高越好。对于一些复杂的计算，warp 占有率低，每个线程可以使用更多的资源，有更高的指令并行度，来获得更好的性能。对于一些简单的算法，单个线程需要的资源少，提高 warp 占有率可以获得更好的性能。

### 最大化内存吞吐量

最大化内存吞吐的主要手段就是减少使用低带宽的内存。这意味着首先要尽可能减少主机端和设备端间的设备传输，其次要尽可能减少全局内存的读写；尽可能的使用片上的内存（寄存器、cache、共享内存）。

GPU 的内存是层级结构，从上到下依次是寄存器文件，L1 cache，L2 cache 和 device memory。

在 H100 上，一个 SM 上有 64K 的 32bits 寄存器，256KB 的 L1 cache，最高可分成 228KB 的 shared memory，50M 的 L2 cache，和 80GB 的 HBM3e，也就是 device memory。

在 GPU cache 中，连续对齐的 32 个 bytes 是一个扇区 sector，连续的 4 个扇区是一个 cache line。

sector 是内存访问的最小粒度，当读内存时，以 sector 为单位进行读取。无论你是否需要扇区中的所有数据，总会读取一个完整的扇区，即使你只访问 4 bytes，也会读取 32 bytes 的数据。写内存时，会根据写入的数据大小进行写。如果你只写 4 个字节，就只会处理 4 个字节。L1 到 L2 最小访问 1 个扇区，L2 到 global 默认最小访问两个扇区。

cache line 是缓存管理的最小单位，128 bytes = 4 sector，合并请求和驱逐都是以 128 bytes 为单位进行的。

当从 global memory 上读取数据时，会首先在 L1 上查找数据是否存在，存在的话直接返回，没有的话继续到 L2 上查找，如果 L2 上还不在就从 global memory 中查找。在这个过程中，都是以整个扇区为单位进行的。

#### 全局内存合并访问

当一个 warp 中的 32 个线程访问连续的 32 个对齐的 4 bytes 数据时，正好访问连续的 128 bytes，也就是一个 cache line。此时，只需要一个 cycle 就能访问全部的 cache line 中的数据。这种访问模式称为合并访问，**COALESCED**。

当线程访问元素间隔变成 8 bytes 时，一个 warp 需要一次访问两个 cache line，这种访问基本也能在一个 cycle 中访问两个 cache line，所以也是合并访问。

但是当 stride 逐渐变大时，一个 warp 需要访问更多的 sector 和 cache line，此时一次访问就需要多个 cycle 才能完成了。最差的一种情况是一个线程访问一个 cache line。

#### Shared Memory

L1 cache 和 shared memory 在同一片物理缓存上，用户可以根据需要分配不同大小的 shared memory。在 Hopper 上，一个 SM 有 256KB 的 L1 cache，最高可以分配 228KB 的 shared memory。

shared memory 的速度很快，所以可以保存一些频繁使用的数据，优化全局内存访问模式。一个 block 中的所有线程都可以访问 shared memory。

虽然 shared memory 和 L1 在同一片物理内存上，但是两者的访问模式却不一样。L1 cache 是以 sector 为基本单位访问的，而 shared memory 则是按照 banks 的形式进行组织的。

如下图所示，shared memory 中一共有 32 个 banks，一个 bank 的长度是 4 bytes，连续的 4 个 bytes 对应到连续的 banks，32 个 banks 一共有 128 bytes，所以 shared memory 中的数据可以看成是 N 行，128 bytes 列的二维数组。

不同 bank 内的数据可以在一个 cycle 内同步访问，从而最大化 shared memory 的带宽。

但是一个 bank 在一个 cycle 内只能有一个线程进行访问，所以当不同的线程访问同一个 bank 中不同地址的数据时会发生 bank conflict，导致串行化访问，降低访问效率。

如下图所示，shared memory 可以有下面三种访问模式。

当一个 warp 中的 32 个线程依次访问 32 个 banks 中的元素时，每个线程对应一个 bank，因此可以在一个 cycle 获取 128 bytes 的数据，并且没有 bank conflict。

当一个 warp 中的 32 个线程按照间隔若干个元素进行访问时，不同的线程就可能会访问同一个 bank。

比如在下图中，每个线程间隔一个元素，前 16 个线程和后 16 个线程会访问同一个 bank 中的不同地址，此时就会触发 bank conflict。每个 bank 有两个线程同时访问，称为 2-way bank conflicts。这种情况下会序列化成前 16 个线程在第一个 cycle 访问，后 16 个线程在第二个 cycle 访问。如果 bank conflict 数过多，就会导致性能下降。

当一个 warp 中的 32 个线程访问同一个 bank 的地址时会触发 broadcast 机制，也不会有 bank conflicts。

假如 shared memory 中有 32*32 个 float 矩阵。矩阵按照 row major 存储，一行刚好占满 32 个 bank，一个 bank 中有 32 个元素。当一个 warp 中的 32 个线程访问一列的 32 个元素时，会导致 32-way bank conflict。

解决 bank conflict 可以通过 padding，在每一行最后 padding 一个元素。

或者使用 swizzling，`threadIdx.y ^ threadIdx.x`。通过行和列进行异或得到新的索引，从而解决 bank conflict。

#### 向量化访问

通过向量化访问也可以提高内存读取的吞吐率。使用 float2 或 float4，一个指令可以读取更多的数据，从而提高带宽利用率。

如下图所示，如果使用 int，一个指令只能读取一个 cache line 的数据，一个 warp 需要执行 4 个指令才能获取完整的 4 个 cache line 的数据。

当使用 int4 时，一个指令就可以获取 4 个 cache line 的数据。

在下面的例子中，向量 A，B 中有 1e8 个 float 数据。如果使用 float 数据类型进行相加，在 H100 上耗时 402 ms。从下面的 ncu profile 结果中可以看到，一共执行了 100000000 / 32 * 2 = 6250000 个指令，读取了 25000000 个扇区，和 8e8 bytes 的数据。

```cpp
__global__ void vector_add1(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}
```

当使用 float4 数据类型时，同样规模的数据在 H100 上耗时 336 ms。ncu profile 显示，一共执行了 100000000 / 128 * 2 = 1562500 个指令，同样读取了 25000000 个扇区，和 8e8 bytes 的数据。

```cpp
template <int stride>
__global__ void vector_add2(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int N)
{
    const int tid = threadIdx.x;
    const int index = blockIdx.x * blockDim.x * stride;
    const int offset = index + tid * stride;

    float4 a, b, c;
    if (offset + 3 < N)
    {
        a = reinterpret_cast<const float4 *>(A + index)[tid];
        b = reinterpret_cast<const float4 *>(B + index)[tid];
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        reinterpret_cast<float4 *>(C + index)[tid] = c;
    }
    else
    {
        for (int i = 0; i < 4 && (offset + i) < N; ++i)
        {
            C[offset + i] = A[offset + i] + B[offset + i];
        }
    }
}
```

#### 异步访问

在 Ampere 和 Hopper 架构中，增加了内存异步访问指令，通过这些指令我们可以把内存读写和计算 overlap 起来，进一步提高算子的性能。

此外，使用异步指令可以直接把数据从 global memory 搬运到 shared memory 中，减少寄存器的使用。可以把更多的寄存器分配给计算线程使用。

```cpp
__global__ void vector_add(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}
```

```cpp
template <int S>
__global__ void vector_add(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int N)
{
    __shared__ float s_A[2][256];
    __shared__ float s_B[2][256];

    int tid = threadIdx.x;
    int index = blockDim.x * blockIdx.x * S;

    int stage = 0;

    // prefetch gmem -> smem
    __pipeline_memcpy_async(&s_A[stage][tid], &A[index + tid], sizeof(float));
    __pipeline_memcpy_async(&s_B[stage][tid], &B[index + tid], sizeof(float));
    __pipeline_commit();

    stage ^= 1;
    index += blockDim.x;

    for (int i = 0; i < S - 1; ++i)
    {
        __pipeline_memcpy_async(&s_A[stage][tid], &A[index + tid], sizeof(float));
        __pipeline_memcpy_async(&s_B[stage][tid], &B[index + tid], sizeof(float));
        __pipeline_commit();

        __pipeline_wait_prior(1);
        __syncthreads();

        float a = s_A[i % 2][tid];
        float b = s_B[i % 2][tid];
        C[index - blockDim.x + tid] = a + b;

        stage ^= 1;
        index += blockDim.x;
    }

    __pipeline_wait_prior(0);
    __syncthreads();

    stage ^= 1;

    s_A[stage][tid] += s_B[stage][tid];
    C[index - blockDim.x + tid] = s_A[stage][tid];
}
```

#### 小结

最大化内存吞吐主要有以下几个方面：

对于 global memory：

- 尽量访问对齐和合并访问，最大化 in-flight 的 bytes 数量来占满内存带宽。
- 一个 thread 访问更多的数据，使用向量化访问，启动更多的线程。

对于 shared memory：

- 向量化访问。
- 减少 bank conflict。
- 对于 shared memory，一个 warp 访问一个 float 时，只要没有 bank conflict，不管每个线程的数据差距多大都可以在一个 cycle 中访问完成。存在 bank conflict 时需要 conflict 数加一个 cycle 访问完成。一个 warp 访问多个 float 时，一个 cycle 也只能访问 128 bytes。

### 最大化指令吞吐量

为了最大化指令吞吐量，应用程序应该：

- 尽量减少使用低吞吐量的算术指令；
- 最大限度地减少由控制流指令引起的线程束分化；
- 减少指令的数量，尽量减少指令数，如少用 `__syncthreads()`。

在本节中，指令吞吐量以每个 SM 在每时钟周期执行的操作数目给出。对于 32 大小的 Warp，一条指令对应 32 次操作，因此若 N 是每个时钟周期的操作数量，则指令吞吐量为 N/32 每时钟周期/指令。

所有吞吐量都是针对单个 SM 的。整个设备的吞吐量必须乘以设备中的所有 SM 的数量。

#### 算数指令

下表给出了在各种计算能力的设备上硬件原生支持的算术指令的吞吐量。

> *Throughput of Native Arithmetic Instructions. (Number of Results per Clock Cycle per Multiprocessor)*

其他指令和函数是在 Native 指令之上实现的。计算能力不同的设备可能有不同的实现，编译后的 Native 指令的数量可能会随着编译器版本的不同而变化。对于复杂的函数，可能有多个代码路径，具体取决于输入。`cuobjdump` 可用于检查 cubin 对象中的特定实现。

**Tips：**

1. 使用 `-ftz=true` 编译的代码（非规范化数字刷新为零，denormalized numbers are flushed to zero）往往比使用 `-ftz=false` 编译的代码具有更高的性能。
2. 使用 `-prec-div=false`（不太精确的除法，less precise division）编译的代码往往比使用 `-prec-div=true` 编译的代码具有更高的性能。
3. 使用 `-prec-sqrt=false`（不太精确的平方根，less precise square root）编译的代码往往比使用 `-prec-sqrt=true` 编译的代码具有更高的性能。
4. `__fdividef(x, y)`（单精度浮点除法，Single-Precision Floating-Point Division）提供了比除法运算符更快的单精度浮点除法。
5. `rsqrtf()`（单精度浮点倒数平方根，Single-Precision Floating-Point Reciprocal Square Root），编译器会将 `1.0 / sqrtf()` 优化为 `rsqrtf()`，这种情况只发生在倒数与平方根都是近似值时（即 `-prec-div=false` 和 `-prec-sqrt=false`）。因此，如果需要时，建议直接调用 `rsqrtf()`。
6. 在某些情况下，可以用位运算代替除法和取模运算：如果 n 是 2 的幂，则 `(i/n)` 等价于 `(i>>log2(n))` 并且 `(i%n)` 等价于 `(i&(n-1))`；如果 n 是字母，则编译器会执行这些转换。
7. `__brev` 和 `__popc` 将映射为一条指令，而 `__brevll` 和 `__popcll` 将映射为几条指令。
8. `__[u]mul24` 是遗留内部函数，在任何情况下都不应该使用。
9. 为了实现 16 位精度浮点加、乘或乘加的良好性能，建议使用 `half2` 数据类型替换 `half` 精度，使用 `__nv_bfloat162` 替换 `__nv_bfloat16` 精度，并使用 vector intrinsics 函数（例如 `__hadd2`、`__hsub2`、`__hmul2`、`__hfma2`）在一条指令中执行两个操作。

**Type Conversion**

有时，编译器必须插入转换指令，而这引入了额外的执行周期。具体情况如下：

- 对 `char` 或 `short` 类型进行操作的函数，其操作数通常需要转换为 `int`，
- （由 C/C++ 标准规定）单精度浮点计算的输入会转换为双精度浮点常量（即转换为那些没有任何类型后缀定义的常量）。

最后一种情况可以通过使用单精度浮点常量来避免，这些常量使用 f 后缀定义，例如 `3.141592653589793f`、`1.0f`、`0.5f`。

#### 控制流指令

任何流控制指令（`if`、`switch`、`do`、`for`、`while`）都会导致同一个 Warp 中的线程分化，并显著影响有效指令的吞吐量。

在控制流依赖线程 ID 的情况下，为了获得最佳性能，应设计控制条件来最小化 Warp 发散的数量。一个简单的例子是当控制条件仅取决于 `(threadIdx / warpSize)` 时，这里 warpSize 是 Warp 的大小。在这种情况下，由于控制条件与 Warp 完全对齐，因此不会出现 Warp 分化。

有时，编译器可能会进行循环展开，或者会通过分支预测来优化短的 if 或 switch 块。在这些情况下，所有 Warp 都不会分化。程序员还可以使用 `#pragma unroll` 指令控制循环展开。

#### 同步指令

`__syncthreads()` 的吞吐量为：

- 对于计算能力为 6.0 的设备，每时钟周期 32 次操作；
- 对于计算能力为 7.x 和 8.x 的设备，每时钟周期 16 次操作；
- 对于计算能力为 5.x、6.1 和 6.2 的设备，每时钟周期 64 次操作。

`__syncthreads()` 可以通过强制 SM 空闲来影响性能。

### 最小化内存抖动（Memory Thrashing）

应用程序频繁地进行内存分配和释放（例如使用 `cudaMalloc` 和 `cudaFree`）可能会导致内存分配调用随着时间的推移变得越来越慢，直到达到某个极限。这是由于释放的内存需要返回给操作系统，供其自身使用。这种情况下，操作系统需要管理更多的内存碎片，从而导致内存分配的效率下降。

1. 合理分配内存：
   - 根据问题规模分配内存：不要试图一次性分配所有可用内存（例如使用 `cudaMalloc`、`cudaMallocHost` 或 `cuMemCreate`），因为这会立即占用大量内存，阻止其他应用程序使用这些内存。这会增加操作系统调度器的压力，甚至可能导致其他使用同一 GPU 的应用程序无法运行。
   - 尽早分配适当大小的内存：在应用程序早期阶段分配适当大小的内存，并在应用程序不再需要这些内存时进行释放。这样可以减少在性能关键区域频繁调用 `cudaMalloc` 和 `cudaFree` 的次数。

2. 减少内存分配和释放的次数：
   - 减少 `cudaMalloc` 和 `cudaFree` 调用：尤其是在性能关键区域，尽量减少内存分配和释放的次数。可以通过预先分配足够的内存并在需要时重用这些内存来实现。

3. 使用其他内存类型：
   - 考虑使用 `cudaMallocHost` 或 `cudaMallocManaged`：如果应用程序无法分配足够的设备内存，可以考虑使用 `cudaMallocHost` 或 `cudaMallocManaged`。虽然这些内存类型的性能可能不如设备内存，但它们可以让应用程序继续运行。
   - 利用 `cudaMallocManaged` 的 oversubscription 功能：对于支持该功能的平台，`cudaMallocManaged` 允许超额分配内存。通过启用正确的 `cudaMemAdvise` 策略，应用程序可以保留大部分甚至全部的 `cudaMalloc` 性能。此外，`cudaMallocManaged` 不会在分配时立即占用内存，直到实际需要或预取时才会占用，从而减少操作系统调度器的整体压力。

## Chapter 9. CUDA-Enabled GPUs

[https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus) lists all CUDA-enabled devices with their compute capability.

The compute capability, number of multiprocessors, clock frequency, total amount of device memory, and other properties can be queried using the runtime (see reference manual).

## 参考链接

1. [CUDA C++ Programming Guide - Performance Guidelines](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance-guidelines)
2. [Introduction to CUDA Programming and Performance Optimization](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62191/)
3. [CUDA Techniques to Maximize Memory Bandwidth and Hide Latency](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/)
4. [CUDA Techniques to Maximize Compute and Instruction Throughput](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72685/)
5. [CUDA Techniques to Maximize Concurrency and System Utilization](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72686/)
