---
title: CUDA C++ 笔记（一）第1章——第5章 Programming Model
date: 2024-06-01 20:00:00
tags: [CUDA, GPU, Programming Model]
categories: [CUDA C++ Programming Guide]
description: 本篇涵盖 CUDA C++ Programming Guide 的第 1 到第 5 章，包括 GPU 的优势、CUDA 平台概述、可扩展编程模型，以及 Programming Model（kernels、线程层次结构、内存层次结构、异构编程、异步 SIMT 编程模型、计算能力）等内容。
---

## Chapter 1. The Benefits of Using GPUs

### 官方文档概述

GPU 提供了比 CPU 更高的指令吞吐量和内存带宽（在相似的价格和功耗范围内）。这种能力差异源于设计目标的不同：

- CPU 设计用于尽可能快地执行一系列操作（称为线程），可以并行执行几十个这样的线程。
- GPU 设计用于并行执行数千个线程（通过摊销较慢的单线程性能来获得更高的吞吐量）。

GPU 专门用于高度并行的计算，因此更多的晶体管被用于数据处理而不是数据缓存和流控制。

> The GPU can hide memory access latencies with computation, instead of relying on large data caches and complex flow control to avoid long memory access latencies, both of which are expensive in terms of transistors.

通常，应用程序既有并行部分也有串行部分，因此系统被设计为 GPU 和 CPU 的混合，以最大化整体性能。具有高度并行性的应用程序可以利用 GPU 的大规模并行特性来获得比 CPU 更高的性能。

### 如何理解 "hide latency" 这段话？

![GPU 与 CPU 的对比](/assets/cudacpp-1-5/image.png)

CPU 的优势：

- complex control flow
- large data cache
- designed to excel at executing a sequence of operations

GPU 的优势：

- specialized for highly parallel computations
- hide latency（Warp Scheduling）【1】

```cpp
#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
    return 0;
}

/*
nvcc sm_count.cu -o sm_count
./sm_count
*/
```

### 开发者的视角

即使是针对数据并行的任务（GEMM 计算，深度学习模型推理），CPU 同样有对应的优化加速技术，因此，在谈论 GPU 在并行计算方面的优势时，我们可能需要更谨慎。

#### CPU 的计算加速技术

**loop optimization**

- loop reordering：优化遍历过程中数据的 locality

![loop reordering](/assets/cudacpp-1-5/image2.png)

- loop tiling：通过调整遍历方式，减少内存地址的访问次数

![loop tiling](/assets/cudacpp-1-5/image3.png)

- loop unrolling
  - 增加了代码量
  - 减少了遍历次数

**SIMD（Single Instruction Multiple Data）加速**

特征：向量寄存器，向量操作

![SIMD](/assets/cudacpp-1-5/image4.png)

基于不同计算机架构支持的不同指令集，我们可以进行加速优化，如下图：

![指令集](/assets/cudacpp-1-5/image5.png)

**多线程加速**

![多线程](/assets/cudacpp-1-5/image6.png)

基于多线程，我们可以实现对不同位置的数据的同时计算。

![多线程加速](/assets/cudacpp-1-5/image7.png)

### 与 GPU 的性能对比

参考链接：[https://github.com/mit-han-lab/parallel-computing-tutorial/tree/main](https://github.com/mit-han-lab/parallel-computing-tutorial/tree/main)

![性能对比](/assets/cudacpp-1-5/image8.png)

A100 测试结果：

```
[root@xxx parallel-computing-tutorial-main]# ./benchmark

naive_mat_mul: 15671 ms
mat_mul_unrolling: 4491 ms
mat_mul_reordering: 1690 ms
mat_mul_tiling: 1148 ms
mat_mul_multithreading: 4241 ms
mat_mul_transpose_simd: 5688 ms
mat_mul_cuda: 9748 ms
mat_mul_fast: 374 ms
```

观点：基于 CPU 和 GPU 都可以进行并行计算的优化，具体的优化效果取决于硬件、任务类型、优化方法。就大模型而言，可以看到虽然目前主流的是 GPU 的推理加速和部署，但是如 llama.cpp 和 llamafile 也提供了 CPU 加速的方案。正因为如此，我们需要保证 CUDA 代码是否有效利用了 GPU 的算力资源。

## Chapter 2. CUDA: A General-Purpose Parallel Computing Platform and Programming Model

2006 年 11 月，NVIDIA 推出了 CUDA，一个通用并行计算平台和编程模型，利用 NVIDIA GPU 中的并行计算引擎以比 CPU 更高效的方式解决许多复杂的计算问题。

CUDA 附带一个软件环境，允许开发者使用 C++ 作为高级编程语言。其他语言、应用程序编程接口或基于指令的方法也得到支持，例如 FORTRAN、DirectCompute、OpenACC 等。

![CUDA 平台](/assets/cudacpp-1-5/image9.png)

## Chapter 3. A Scalable Programming Model

多核 CPU 和众核 GPU 的出现意味着主流处理器芯片现在是并行系统。面临的挑战是开发应用软件，使其能够透明地扩展并行性以利用越来越多的处理器核心，就像 3D 图形应用程序透明地将并行性扩展到具有不同核心数量的众核 GPU 一样。

CUDA 并行编程模型旨在克服这一挑战，同时为熟悉标准编程语言（如 C）的程序员保持较低的学习曲线。

### Three Key Abstractions

其核心是三个关键抽象，作为最小的语言扩展集简单地暴露给程序员：

- a hierarchy of thread groups（线程组层次结构）
- shared memories（共享内存）
- barrier synchronization（屏障同步）

这些抽象提供细粒度数据并行和线程并行，嵌套在粗粒度数据并行和任务并行之中。它们指导程序员将问题分解为可以由线程块独立并行解决的粗子问题，并将每个子问题分解为可以由块内所有线程协作并行解决的更细的部分。

![可扩展编程模型](/assets/cudacpp-1-5/image10.png)

这种分解保留了语言表达能力，允许线程在解决每个子问题时进行协作，同时实现了自动可扩展性。每个线程块可以调度在 GPU 内任何可用的多处理器上，以任何顺序、并发或顺序执行，因此编译后的 CUDA 程序可以在任意数量的多处理器上执行，只有运行时系统需要知道物理多处理器数量。

> A GPU is built around an array of Streaming Multiprocessors (SMs) (see Hardware Implementation for more details). A multithreaded program is partitioned into blocks of threads that execute independently from each other, so that a GPU with more multiprocessors will automatically execute the program in less time than a GPU with fewer multiprocessors.

![SM 阵列](/assets/cudacpp-1-5/image11.png)

## Chapter 5. Programming Model

> 说明：官方最新版本中，第 4 章是 Changelog，第 5 章是 Programming Model。

本章介绍了 CUDA 编程模型背后的主要概念，并概述了它们在 C++ 中的表达方式。

### 5.1 Kernels

CUDA C++ 扩展了 C++，允许程序员定义称为 kernel 的 C++ 函数，当被调用时，由 N 个不同的 CUDA 线程并行执行 N 次，而不是像常规 C++ 函数那样只执行一次。

kernel 使用 `__global__` 声明说明符定义，执行该 kernel 的 CUDA 线程数量通过新的 `<<<...>>>` 执行配置语法指定。每个执行 kernel 的线程都被赋予一个唯一的线程 ID，可通过内置变量在 kernel 内访问。

```cpp
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    // ...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    // ...
}
```

关键概念：

- declaration specifier：`__global__`
- built-in variable：`threadIdx`
- `<<<...>>>`：`kernel_function<<<numBlocks, threadsPerBlock, sharedMemSize, stream>>>(kernelArgs);`

### 5.2 Thread Hierarchy

#### 5.2.1 Thread

为方便起见，`threadIdx` 是一个三维向量，因此可以使用一维、二维或三维的线程索引来标识线程，形成一维、二维或三维的线程块（thread block）。这提供了一种在向量、矩阵或体积等域的元素上调用计算的自然方式。

线程的索引和线程 ID 之间的关系很简单：

- 对于一维块，它们是相同的；
- 对于大小为 (Dx, Dy) 的二维块，索引为 (x, y) 的线程的线程 ID 为 (x + y Dx)；
- 对于大小为 (Dx, Dy, Dz) 的三维块，索引为 (x, y, z) 的线程的线程 ID 为 (x + y Dx + z Dx Dy)。

每个线程块中的线程数量有限制，因为一个块的所有线程都驻留在同一个流多处理器核心上，必须共享该核心有限的内存资源。在当前的 GPU 上，一个线程块最多可以包含 1024 个线程。

然而，一个 kernel 可以由多个相同形状的线程块执行，因此线程总数等于每个块的线程数乘以块数。

案例一（一维向量加法）：

```cpp
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Global thread index
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

案例二（二维矩阵加法）：

```cpp
__global__ void matrixAdd(const float* A, const float* B, float* C, int numRows, int numCols) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < numRows && col < numCols) {
        int idx = row * numCols + col;  // Convert 2D index to 1D
        C[idx] = A[idx] + B[idx];
    }
}
```

以下是官方文档中处理多块的矩阵加法示例。块内使用 `threadIdx`，多块使用 `blockIdx` 和 `blockDim`：

```cpp
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

16x16（256 个线程）的线程块大小是一个常见的选择。网格创建时需要有足够的块来像以前一样每个矩阵元素对应一个线程。

#### 5.2.2 Thread Block

> Thread blocks are required to execute independently: It must be possible to execute them in any order, in parallel or in series.

线程块必须能够独立执行：必须能够以任何顺序、并行或串行地执行块。这种独立性要求允许线程块以任何顺序调度到任何数量的核心上，使程序员能够编写随核心数量扩展的代码。

> Threads within a block can cooperate by sharing data through some shared memory and by synchronizing their execution to coordinate memory accesses.

块内的线程可以通过共享内存共享数据，并通过同步执行来协调内存访问。

> More precisely, one can specify synchronization points in the kernel by calling the `__syncthreads()` intrinsic function; `__syncthreads()` acts as a barrier at which all threads in the block must wait before any is allowed to proceed.

更精确地说，可以通过调用 `__syncthreads()` 内置函数在 kernel 中指定同步点；`__syncthreads()` 充当屏障，块中的所有线程必须在此等待，然后才允许任何一个线程继续执行。除了 `__syncthreads()` 之外，Cooperative Groups API 还提供了丰富的线程同步原语。

为了高效协作，共享内存被期望是靠近每个处理器核心的低延迟内存（类似于 L1 缓存），`__syncthreads()` 被期望是轻量级的。

#### 5.2.3 Thread Block Clusters

随着 NVIDIA Compute Capability 9.0 的引入，CUDA 编程模型引入了一个可选的层次结构级别，称为 Thread Block Clusters（线程块集群），由线程块组成。类似于线程块中的线程保证被共同调度在同一个流多处理器上，集群中的线程块也保证被共同调度在 GPU 的一个 GPU 处理集群（GPC）上。

集群组织为一维、二维或三维的线程块集群网格。集群中的线程块数量可以由用户定义，CUDA 中支持的最大 8 个线程块作为一个可移植的集群大小。

集群可以通过编译时内核属性 `__cluster_dims__(X,Y,Z)` 或 CUDA kernel 启动 API `cudaLaunchKernelEx` 启用。

**编译时集群大小示例：**

```cpp
// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float *output)
{

}

int main()
{
    float *input, *output;
    // Kernel invocation with compile time cluster size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```

**运行时集群大小示例（使用 `cudaLaunchKernelEx`）：**

```cpp
// Kernel definition
// No compile time attribute attached to the kernel
__global__ void cluster_kernel(float *input, float *output)
{

}

int main()
{
    float *input, *output;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // Kernel invocation with runtime cluster size
    {
        cudaLaunchConfig_t config = {0};
        config.gridDim = numBlocks;
        config.blockDim = threadsPerBlock;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, cluster_kernel, input, output);
    }
}
```

在计算能力 9.0 的 GPU 上，集群中的所有线程块保证被共同调度在单个 GPC 上，允许集群中的线程块使用 Cluster Group API `cluster.sync()` 进行硬件支持的同步。集群组还提供了成员函数来查询集群组大小（以线程数或块数为单位）：`num_threads()` 和 `num_blocks()`。

属于集群的线程块可以访问分布式共享内存（Distributed Shared Memory）。集群中的线程块能够对分布式共享内存中的任何地址进行读取、写入和原子操作。

#### 5.2.4 Grid

块被组织成一维、二维或三维的线程块网格（grid）。网格中的线程块数量通常由正在处理的数据大小决定，通常超过系统中的处理器数量。

`<<<...>>>` 语法中指定的每块线程数和每网格块数可以是 `int` 或 `dim3` 类型。网格中的每个块可以通过内置的 `blockIdx` 变量访问的一维、二维或三维唯一索引来标识。线程块的维度可以通过内置的 `blockDim` 变量在 kernel 内访问。

### 5.3 Memory Hierarchy

CUDA 线程在执行期间可以从多个内存空间访问数据：

- 每个线程有私有的本地内存（local memory）。
- 每个线程块有共享内存（shared memory），对块中的所有线程可见，生命周期与块相同。
- 线程块集群中的线程块可以对彼此的共享内存执行读取、写入和原子操作。
- 所有线程都可以访问相同的全局内存（global memory）。

还有两个额外的只读内存空间可被所有线程访问：常量内存（constant memory）和纹理内存（texture memory）。全局、常量和纹理内存空间针对不同的内存使用进行了优化。纹理内存还提供不同的寻址模式以及针对某些特定数据格式的数据过滤。

全局、常量和纹理内存在同一应用程序的 kernel 启动之间是持久的。

CUDA 线程可以从 local memory、thread block 中共享的 shared memory，以及 global memory 中加载数据。具体参考第 8 章。

### 5.4 Heterogeneous Programming

CUDA 编程模型假设 CUDA 线程在物理上独立的设备（device）上执行，该设备作为运行 C++ 程序的主机（host）的协处理器。例如，当 kernel 在 GPU 上执行时，其余 C++ 程序在 CPU 上执行。

CUDA 编程模型还假设主机和设备在 DRAM 中维护各自独立的内存空间，分别称为主机内存和设备内存。因此，程序通过调用 CUDA 运行时来管理 kernel 可见的全局、常量和纹理内存空间。这包括设备内存的分配和释放以及主机和设备内存之间的数据传输。

Unified Memory 提供托管内存（managed memory）来连接主机和设备内存空间。托管内存可从系统中的所有 CPU 和 GPU 作为单个具有统一地址空间的连贯内存映像访问。此功能启用设备内存的超额订阅，并通过消除在主机和设备上显式镜像数据的需要，极大简化了移植应用程序的任务。

Unified Memory provides managed memory to bridge the host and device memory spaces.

unified memory 详情参考第 22 章。

### 5.5 Asynchronous SIMT Programming Model

在 CUDA 编程模型中，线程是进行计算或内存操作的最低抽象级别。从基于 NVIDIA Ampere GPU 架构的设备开始，CUDA 编程模型通过异步编程模型为内存操作提供加速。异步编程模型定义了异步操作相对于 CUDA 线程的行为。

异步编程模型定义了 Asynchronous Barrier 的行为，用于 CUDA 线程之间的同步。该模型还解释并定义了如何使用 `cuda::memcpy_async` 在 GPU 计算时从全局内存异步移动数据。

> The CUDA thread that initiated the asynchronous operation is not required to be among the synchronizing threads.

#### 5.5.1 Asynchronous Operations

异步操作定义为由 CUDA 线程发起并如同由另一个线程异步执行的操作。在格式良好的程序中，一个或多个 CUDA 线程与异步操作同步。发起异步操作的 CUDA 线程不需要在同步线程之中。

这种异步线程（as-if 线程）始终与发起异步操作的 CUDA 线程相关联。异步操作使用同步对象来同步操作的完成。同步对象可以由用户显式管理（如 `cuda::memcpy_async`）或在库内隐式管理（如 `cooperative_groups::memcpy_async`）。

同步对象可以是 `cuda::barrier` 或 `cuda::pipeline`。这些同步对象可以在不同的线程范围（thread scope）内使用。范围定义了可以使用同步对象与异步操作同步的线程集合。下表定义了 CUDA C++ 中可用的线程范围：

| Thread Scope | Description |
|:---:|:---:|
| `cuda::thread_scope::thread_scope_thread` | 只有发起异步操作的 CUDA 线程同步 |
| `cuda::thread_scope::thread_scope_block` | 与发起线程在同一线程块中的所有或任何 CUDA 线程同步 |
| `cuda::thread_scope::thread_scope_device` | 与发起线程在同一 GPU 设备中的所有或任何 CUDA 线程同步 |
| `cuda::thread_scope::thread_scope_system` | 与发起线程在同一系统中的所有或任何 CUDA 或 CPU 线程同步 |

### 5.6 Compute Capability

设备的计算能力（compute capability）由版本号表示，有时也称为"SM 版本"。此版本号标识 GPU 硬件支持的功能，应用程序在运行时使用它来确定当前 GPU 上可用的硬件功能和/或指令。

计算能力由主版本号 X 和次版本号 Y 组成，表示为 X.Y。

主版本号指示设备的核心 GPU 架构。具有相同主版本号的设备共享相同的基础架构。下表列出了对应于每个 NVIDIA GPU 架构的主版本号：

| Major Revision Number | NVIDIA GPU Architecture |
|:---:|:---:|
| 9 | NVIDIA Hopper GPU Architecture |
| 8 | NVIDIA Ampere GPU Architecture |
| 7 | NVIDIA Volta GPU Architecture |
| 6 | NVIDIA Pascal GPU Architecture |
| 5 | NVIDIA Maxwell GPU Architecture |
| 3 | NVIDIA Kepler GPU Architecture |

次版本号对应于核心架构的增量改进，可能包含新功能。例如，计算能力 7.5 的 NVIDIA Turing GPU 架构基于 NVIDIA Volta GPU 架构。

> 注意：特定 GPU 的计算能力版本不应与 CUDA 版本（例如 CUDA 7.5、CUDA 8、CUDA 9）混淆，后者是 CUDA 软件平台的版本。CUDA 平台被应用程序开发者用于创建运行在多代 GPU 架构（包括未来尚未发明的 GPU 架构）上的应用程序。

```cpp
// get_sm_version.cu
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    }

    return 0;
}

// nvcc get_sm_version.cu -o get_sm_version
```

`printf` in CUDA is explicitly supported by the device runtime.

#### `__syncthreads()`

`__syncthreads()` acts as a barrier at which all threads in the block must wait before any is allowed to proceed.

## 参考链接

[How to understand the hide latency](https://forums.developer.nvidia.com/t/how-to-understand-the-hide-latency/258938/7)

[CUDA: the GPU computing revolution begins](https://www.gamesindustry.biz/nvidia-unveils-cuda-the-gpu-computing-revolution-begins)
