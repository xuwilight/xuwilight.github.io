---
title: CUDA C++ 笔记（十二）第16-17章——Graph Memory Nodes 与 Mathematical Functions
date: 2024-07-15 20:00:00
tags: [CUDA, Graph Memory, Mathematical Functions]
categories: [CUDA C++ Programming Guide]
description: CUDA Graph Memory Nodes 允许 Graph 管理内存分配和释放，支持内存复用与跨 GPU 对等访问。第17章涵盖 CUDA 标准数学函数及其 ULP 误差范围、内置函数优化以及 fast math 编译选项。
---

## 第15章 Graph memory node

### 15.1 Intro

对 GPU 进行性能优化时，cudagraph 是绕不开的话题。

不仅是 GPU，大部分的 xpu 都会提供类似优化，相比于每次分别由 CPU 进行 kernel launch 的 eager mode，graph mode 通常都会有较大性能提升。

既然 cudagraph 的名字里包含 graph，那它必然由节点和边组成。

一般来说，cudagraph 是个有向图，边表示节点之间的依赖关系，当一个节点的前序节点都执行完了的时候，后一个节点就可以开始执行。

例如下图中的子图 Y，只有当 B 和 C 节点都执行结束后，D 节点才会开始执行。

![](/assets/cudacpp-16-17/image.png)

我们可以看到下列类型的节点：

GPU kernel node，表示一 kernel

Memcpy node，可以进行 device 与 host 之间的内存拷贝

Memset node，对 device memory 进行初始化

Host (executable) node, 可以执行一个 CPU 上的函数

subgraph node, 一个 cudagraph 子图。

Empty (no-op) node, 一个空节点。

External event wait node，表示等待某个 event 的节点

External event record node，表示记录某个 event 的节点

Memory allocation node，表示一个内存分配的节点

Memory free node，表示一个内存释放的节点

External semaphore signal node、External semaphore wait node、Conditional node 等节点

以上这些节点中，大部分节点都对应于一个 cuda 函数。

Graph 内存节点（Graph Memory Nodes）允许 Graph 创建并拥有内存分配。

Graph 内存节点具有 **GPU 顺序生命周期语义**，这些语义决定了内存何时可以在设备上被访问。

这些 GPU 顺序生命周期语义使得驱动程序能够管理内存的复用，并且与流顺序分配 API（如 cudaMallocAsync 和 cudaFreeAsync）的语义相匹配，这些 API 在创建 Graph 时可以被 capture。

Graph 分配的内存在 Graph 的整个生命周期内（包括重复实例化和启动）具有固定的地址。

这使得内存可以直接被 Graph 中的其他操作引用，而无需更新 Graph。

在一个 Graph 中，如果多个分配的生命周期不重叠，则它们可以使用相同的底层物理内存。

CUDA 可以在多个 Graph 之间复用相同的物理内存，根据 GPU 顺序生命周期语义对虚拟地址映射进行别名化（aliasing）。

例如，当不同的 Graph 被提交到同一个流中执行时，CUDA 可以通过虚拟别名化相同的物理内存来满足具有单 Graph 生命周期的分配需求。

Graph 内存节点的作用：它允许 Graph 自己管理内存的分配和释放，而不是依赖外部的内存管理。

GPU 顺序生命周期语义：内存的分配和释放是根据 GPU 任务的执行顺序来决定的，而不是根据 CPU 调用 API 的顺序。

内存复用：如果多个内存分配的生命周期不重叠，CUDA 可以自动复用同一块物理内存，从而节省内存。

固定虚拟地址：Graph 分配的内存在 Graph 的整个生命周期内地址不变，即使背后的物理内存发生变化。

跨 Graph 内存共享：不同的 Graph 可以共享同一块物理内存，只要它们的执行不会重叠。

### Example：使用 Graph Memory Node 管理内存

假设我们有一个任务流程，需要在 GPU 上完成以下操作：

1. 分配一块内存。
1. 在 GPU 上运行一个内核（Kernel A），将数据写入这块内存。
1. 运行另一个内核（Kernel B），从这块内存中读取数据并进行处理。
1. 释放这块内存。

#include <cuda_runtime.h>

#include <iostream>

#define CHECK_CUDA(call) \

    do { \

        cudaError_t err = call; \

        if (err != cudaSuccess) { \

            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \

            exit(EXIT_FAILURE); \

        } \

    } while (0)

// 内核函数 A：将数据写入内存

__global__ void kernelA(int* data, int value, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {

        data[idx] = value;

    }

}

// 内核函数 B：从内存中读取数据并处理

__global__ void kernelB(int* data, int* result, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {

        result[idx] = data[idx] * 2;

    }

}

void use_cuda_graph() {

    const int N = 1024;

    const int size = N * sizeof(int);

    // 创建 Graph

    cudaGraph_t graph;

    CHECK_CUDA(cudaGraphCreate(&graph, 0));

    // 定义内存分配节点参数

    cudaMemAllocNodeParams allocParams = {};

    allocParams.poolProps.allocType = cudaMemAllocationTypePinned;

    allocParams.poolProps.location.type = cudaMemLocationTypeDevice;

    allocParams.poolProps.location.id = 0; // 在设备 0 上分配内存

    allocParams.bytesize = size;

    // 添加内存分配节点

    cudaGraphNode_t allocNode;

    CHECK_CUDA(cudaGraphAddMemAllocNode(&allocNode, graph, nullptr, 0, &allocParams));  

    int* d_data = reinterpret_cast<int*>(allocParams.dptr); // 获取分配的内存地址

    // 添加 kernel 节点 A（写入数据）

    cudaGraphNode_t kernelNodeA;

    void* kernelArgsA[] = { &d_data, &N };

    CHECK_CUDA(cudaGraphAddKernelNode(&kernelNodeA, graph, &allocNode, 1, &kernelArgsA));

    // 添加 kernel 节点 B（读取并处理数据）

    cudaGraphNode_t kernelNodeB;

    void* kernelArgsB[] = { &d_data, &d_data, &N }; // 这里复用 d_data 作为结果存储

    CHECK_CUDA(cudaGraphAddKernelNode(&kernelNodeB, graph, &kernelNodeA, 1, &kernelArgsB));

    // 添加 mem free 节点

    cudaGraphNode_t freeNode;

    CHECK_CUDA(cudaGraphAddMemFreeNode(&freeNode, graph, &kernelNodeB, 1, d_data));

    // 实例化 Graph

    cudaGraphExec_t graphExec;

    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // 启动 Graph

    CHECK_CUDA(cudaGraphLaunch(graphExec, 0));

    // 等待 Graph 执行完成

    CHECK_CUDA(cudaDeviceSynchronize());

    // 销毁 Graph

    CHECK_CUDA(cudaGraphExecDestroy(graphExec));

    CHECK_CUDA(cudaGraphDestroy(graph));

    std::cout << "Graph execution completed successfully!" << std::endl;

    return ;

}

void manual_launch() {

    const int N = 1024;

    const int size = N * sizeof(int);

    // 分配设备内存

    int* d_data;

    CHECK_CUDA(cudaMalloc(&d_data, size));

    // 创建 CUDA Stream

    cudaStream_t stream;

    CHECK_CUDA(cudaStreamCreate(&stream));

    // 启动内核 A（写入数据）

    kernelA<<<1, N, 0, stream>>>(d_data, 42, N);

    // 启动内核 B（读取并处理数据）

    kernelB<<<1, N, 0, stream>>>(d_data, d_data, N);

    // 释放设备内存

    CHECK_CUDA(cudaFree(d_data));

    // 等待 Stream 中的所有任务完成

    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 销毁 Stream

    CHECK_CUDA(cudaStreamDestroy(stream));

    std::cout << "Program execution completed successfully!" << std::endl;

    return 0;

}

虽然传统方式可以实现相同的功能，但与使用 **CUDA Graph** 相比，存在以下不足：

#### 1. 更高的 CPU 开销 && 性能优化受限

在传统方式中，每次内核启动和内存操作都需要通过 CPU 调用 CUDA API，这会增加 CPU 和 GPU 之间的通信开销。

而 Graph 可以提前定义好所有任务和依赖关系，一次性提交给 GPU，减少了 CPU 的干预。

cudaMalloc(&d_data, size);          // CPU 调用 API

kernelA<<<1, N, 0, stream>>>(...);  // CPU 调用 API

kernelB<<<1, N, 0, stream>>>(...);  // CPU 调用 API

cudaFree(d_data);                   // CPU 调用 API

cudaStreamSynchronize(stream);      // CPU 调用 API

// 提前定义 Graph

cudaGraphAddMemAllocNode(&allocNode, graph, ...);

cudaGraphAddKernelNode(&kernelNodeA, graph, ...);

cudaGraphAddKernelNode(&kernelNodeB, graph, ...);

cudaGraphAddMemFreeNode(&freeNode, graph, ...);

// 一次性提交 Graph

cudaGraphLaunch(graphExec, stream);

#### 手动管理内存生命周期

在传统方式中，开发者需要手动调用 cudaMalloc 和 cudaFree 来管理内存的生命周期。

而 Graph Memory Node 可以自动管理内存的分配和释放，减少了开发者的负担。

#### 3. 缺乏内存复用优化

在传统方式中，内存复用需要开发者手动实现，比如通过池化内存（Memory Pool）来管理。

而 Graph Memory Node 可以根据 GPU 执行顺序自动复用内存，减少了内存占用。

#### 4. 依赖关系管理复杂

在传统方式中，任务之间的依赖关系需要通过 Stream 和 Event 手动管理，代码复杂度较高。

而 Graph 可以显式定义任务之间的依赖关系，代码更清晰且易于维护。

### 15.2 支持与兼容性

Graph 内存节点（Graph Memory Nodes）需要 **CUDA 11.4 或更高版本** 的驱动程序，并且 GPU 需要支持流顺序分配器（stream ordered allocator）。

以下代码片段展示了如何检查当前设备是否支持这些功能：

int driverVersion = 0;

int deviceSupportsMemoryPools = 0;

int deviceSupportsMemoryNodes = 0;

// 获取当前 CUDA 驱动版本

cudaDriverGetVersion(&driverVersion);

// 检查设备是否支持内存池（Memory Pools）

if (driverVersion >= 11020) {  // 避免在 11.0 和 11.1 驱动上返回无效值

    cudaDeviceGetAttribute(&deviceSupportsMemoryPools, cudaDevAttrMemoryPoolsSupported, device);

}

// 检查设备是否支持 Graph 内存节点

deviceSupportsMemoryNodes = (driverVersion >= 11040) && (deviceSupportsMemoryPools != 0);

### 15.3 API

Graph 内存节点（Graph Memory Nodes）是表示内存分配或释放操作的图节点。

简而言之，分配内存的节点称为 **分配节点**（allocation nodes），而释放内存的节点称为 **释放节点**（free nodes）。

由分配节点创建的内存分配称为 **图分配**（graph allocations）。

CUDA 在创建节点时为图分配分配虚拟地址。虽然这些虚拟地址在分配节点的生命周期内是固定的，但分配的内容在释放操作之后不会持久化，并且可能会被其他分配的操作覆盖。

graph allocations 在每次图运行时被视为重新创建。

graph allocations 的生命周期（与节点的生命周期不同）从 GPU 执行到达分配节点时开始，并在以下任一情况发生时结束：

1. GPU 执行到达释放图节点。
1. GPU 执行到达流顺序释放调用 cudaFreeAsync()。
1. 在调用 cudaFree() 时立即释放。

**注意**：图的销毁不会自动释放任何存活的图分配内存，即使它结束了分配节点的生命周期。必须在另一个图中或使用 cudaFreeAsync() / cudaFree() 显式释放这些分配。

与其他图结构一样，图内存节点通过依赖边（dependency edges）在图中保持顺序。程序必须确保访问图内存的操作：

1. 在分配节点之后排序。
1. 在释放内存的操作之前排序。

### 15.3.1 Directly construct

图内存节点可以通过内存节点创建 API 显式创建，包括 cudaGraphAddMemAllocNode 和 cudaGraphAddMemFreeNode。

cudaGraphAddMemAllocNode 分配的内存地址通过传递给 CUDA_MEM_ALLOC_NODE_PARAMS 结构的 dptr 字段返回给用户。

分配图中使用图分配的所有操作必须在分配节点之后排序。同样，任何释放节点必须在图中所有使用该分配的操作之后排序。

cudaGraphAddMemFreeNode 用于创建释放节点。

以下是一个包含分配节点和释放节点的图示例。

kernel 节点 a、b 和 c 在分配节点之后、释放之前执行，因此这些 kernel 可以访问分配的内存。

内核节点 e 不在分配节点之后，因此不能安全地访问内存。

内核节点 d 不在释放节点之前排序，因此它也不能安全地访问内存。

以下代码片段展示了如何创建该图：

// 创建图 - 初始为空

cudaGraphCreate(&graph, 0);

// 基本分配的参数

cudaMemAllocNodeParams params = {};

params.poolProps.allocType = cudaMemAllocationTypePinned;

params.poolProps.location.type = cudaMemLocationTypeDevice;

// 指定设备 0 为驻留设备

params.poolProps.location.id = 0;

params.bytesize = size;

// 内存实际上还没有被分配，直到图被执行时才会分配

// 添加内存分配节点

cudaGraphAddMemAllocNode(&allocNode, graph, NULL, 0, &params);

nodeParams->kernelParams[0] = params.dptr;

// 添加内核节点 a、b 和 c

cudaGraphAddKernelNode(&a, graph, &allocNode, 1, &nodeParams);

cudaGraphAddKernelNode(&b, graph, &a, 1, &nodeParams);

cudaGraphAddKernelNode(&c, graph, &a, 1, &nodeParams);

// 内核节点 b 和 c 使用图分配，因此释放节点必须依赖于它们

cudaGraphNode_t dependencies[2];

dependencies[0] = b;

dependencies[1] = c;

cudaGraphAddMemFreeNode(&freeNode, graph, dependencies, 2, params.dptr);

// 释放节点不依赖于内核节点 d，因此它不能访问已释放的图分配

cudaGraphAddKernelNode(&d, graph, &c, 1, &nodeParams);

// 节点 e 不依赖于分配节点，因此它不能访问分配的内存

cudaGraphAddKernelNode(&e, graph, NULL, 0, &nodeParams);

### 15.3.2 基于 Stream Capture 的 graph construct

图内存节点可以通过捕获相应的流顺序分配和释放调用 cudaMallocAsync 和 cudaFreeAsync 来创建。

在这种情况下，捕获的分配 API 返回的虚拟地址可以被图中的其他操作使用。由于流顺序依赖关系会被捕获到图中，

流顺序分配 API 的排序要求确保了图内存节点与捕获的流操作正确排序（对于正确编写的流代码）。

以下代码片段展示了如何使用流捕获来创建与上图中相同的图（忽略内核节点 d 和 e 以简化）：

cudaMallocAsync(&dptr, size, stream1);

kernel_A<<< ..., stream1 >>>(dptr, ...);

//  stream1 中记录一个事件 event1，表示 stream1 中当前的所有操作（包括 cudaMallocAsync 和 kernel_A）已经完成

// 让 stream2 等待 event1，这样 stream2 中的操作会等待 stream1 中的操作完成后再执行

cudaEventRecord(event1, stream1);

cudaStreamWaitEvent(stream2, event1);

// 在 stream1 中启动内核 kernel_B，继续使用内存 dptr

// kernel_B 的执行会等待 kernel_A 完成

kernel_B<<< ..., stream1 >>>(dptr, ...);

// 在 stream2 中启动内核 kernel_C，使用内存 dptr。

// kernel_C 的执行会等待 event1 完成（即 stream1 中的 cudaMallocAsync 和 kernel_A 完成），因为 stream2 等待了 event1

kernel_C<<< ..., stream2 >>>(dptr, ...);

// 将 stream2 合并回原始流（stream1）

cudaEventRecord(event2, stream2);

cudaStreamWaitEvent(stream1, event2);

// 释放依赖于所有访问内存的工作

cudaFreeAsync(dptr, stream1);

// 在原始流中结束捕获

cudaStreamEndCapture(stream1, &graph);

### 15.3.3 graph 之外 access/free memory

图分配（Graph Allocations）**不一定** 由分配图（Allocating Graph）来释放。

如果图没有释放某个分配，则该分配在图执行后仍然存在，并且可以被后续的 CUDA 操作访问。

1. 常规的 cudaFree 或 cudaFreeAsync 调用。
1. 启动另一个包含相应释放节点的图。
1. 后续启动分配图（如果分配图是以 cudaGraphInstantiateFlagAutoFreeOnLaunch 标志实例化的）。

**注意**：在内存被释放后访问内存是非法操作。释放操作必须通过图依赖关系、CUDA 事件或其他流排序机制在所有访问内存的操作之后排序。

#### 1. 通过 single stream

// 使用图分配内存

void *dptr;

cudaGraphAddMemAllocNode(&allocNode, allocGraph, NULL, 0, &params);

dptr = params.dptr;

cudaGraphInstantiate(&allocGraphExec, allocGraph, NULL, NULL, 0);

cudaGraphLaunch(allocGraphExec, stream);

// 在同一个 stream 中访问分配的内存

kernel<<< ..., stream >>>(dptr, …);

// 在同一个 stream 中释放内存

cudaFreeAsync(dptr, stream);

#### 2. 通过 recording/waiting events 创建依赖

void *dptr;

// allocate graph : allocGraph

cudaGraphAddMemAllocNode(&allocNode, allocGraph, NULL, 0, &params);

dptr = params.dptr;

// free graph : freeGraph

nodeParams->kernelParams[0] = params.dptr;

cudaGraphAddKernelNode(&a, graph, NULL, 0, &nodeParams);

cudaGraphAddMemFreeNode(&freeNode, freeGraph, &a, 1, dptr);

// 实例化 allocateGraph 和 freeGraph

cudaGraphInstantiate(&allocGraphExec, allocGraph, NULL, NULL, 0);

cudaGraphInstantiate(&freeGraphExec, freeGraph, NULL, NULL, 0);

// 启动 allocateGraph

cudaGraphLaunch(allocGraphExec, allocStream);

// 建立 stream2 对分配节点的依赖

cudaEventRecord(allocEvent, allocStream);

cudaStreamWaitEvent(stream2, allocEvent);

// 在 stream2 中访问分配的内存

kernel<<< ..., stream2 >>> (dptr, …);

// 建立 stream3 对内存使用的依赖

cudaStreamRecordEvent(streamUseDoneEvent, stream2);

cudaStreamWaitEvent(stream3, streamUseDoneEvent);

// 现在可以安全地启动 freeGraph

cudaGraphLaunch(freeGraphExec, stream3);

#### 3. graph externel event node

void *dptr;

cudaEvent_t allocEvent;  // 事件指示分配何时准备好使用

cudaEvent_t streamUseDoneEvent;  // 事件指示流操作何时完成内存访问

// 分配图的内容（包含事件记录节点）

cudaGraphAddMemAllocNode(&allocNode, allocGraph, NULL, 0, &params);

dptr = params.dptr;

// 事件记录节点依赖于分配节点

cudaGraphAddEventRecordNode(&recordNode, allocGraph, &allocNode, 1, allocEvent);

cudaGraphInstantiate(&allocGraphExec, allocGraph, NULL, NULL, 0);

// 消费/释放图的内容（包含事件等待节点）

cudaGraphAddEventWaitNode(&streamUseDoneEventNode, waitAndFreeGraph, NULL, 0, streamUseDoneEvent);

cudaGraphAddEventWaitNode(&allocReadyEventNode, waitAndFreeGraph, NULL, 0, allocEvent);

nodeParams->kernelParams[0] = params.dptr;

// allocReadyEventNode 提供与分配节点的

cudaGraphAddKernelNode(&kernelNode, waitAndFreeGraph, &allocReadyEventNode, 1, &nodeParams);

// 释放节点必须在内核节点和 streamUseDoneEventNode 之后排序

cudaGraphNode_t dependencies[2];

dependencies[0] = kernelNode;

dependencies[1] = streamUseDoneEventNode;

cudaGraphAddMemFreeNode(&freeNode, waitAndFreeGraph, &dependencies, 2, dptr);

cudaGraphInstantiate(&waitAndFreeGraphExec, waitAndFreeGraph, NULL, NULL, 0);

// 启动图

cudaGraphLaunch(allocGraphExec, allocStream);

// 建立 stream2 对事件节点的依赖

cudaStreamWaitEvent(stream2, allocEvent);

kernel<<< ..., stream2 >>> (dptr, …);

// 记录 stream2 的内存访问完成事件

cudaStreamRecordEvent(streamUseDoneEvent, stream2);

// 事件等待节点确保释放图在 stream2 完成后执行

cudaGraphLaunch(waitAndFreeGraphExec, stream3);

### 15.4 Memory reuse

CUDA 通过两种方式优化内存复用：

1. graph 内虚拟和物理内存复用：基于虚拟地址分配，类似于流顺序分配器（stream ordered allocator）。
1. graph 之间，物理内存复用：通过虚拟别名化（virtual aliasing）实现，不同的图可以将相同的物理内存映射到它们唯一的虚拟地址。

### 15.4.1 图内地址复用

CUDA 可以在图内通过将相同的虚拟地址范围分配给生命周期不重叠的多个分配来复用内存。

由于虚拟地址可能会被复用，指向生命周期不重叠的不同分配的指针不保证是唯一的。

以下图示展示了如何添加一个新的分配节点（2），它可以复用由依赖节点（1）释放的地址。

### 15.4.2 物理内存管理与共享

CUDA 负责在 GPU 执行到达分配节点之前将 phy mem 映射到 virtual mem。

为了优化内存占用和映射开销，多个图可以使用相同的物理内存来满足不同的分配，前提是这些图不会同时运行。

然而，如果物理页面同时绑定到多个正在执行的图，或者绑定到一个未释放的图分配，则无法 reuse 这些物理页面。

CUDA 可以在图实例化、启动或执行期间的任何时间更新物理内存映射。

CUDA 还可能在未来图启动之间引入同步，以防止 alive graph 分配引用相同的物理内存。

以下图示展示了在同一个 stream 中顺序启动的多个 graph, 在此示例中，每个图都释放了它分配的所有内存。

由于同一流中的图不会同时运行，CUDA 可以并且应该使用相同的物理内存来满足所有分配。

1. 图内内存复用：

通过将相同的虚拟地址分配给生命周期不重叠的分配来实现内存复用。

虚拟地址可能会被复用，因此指向不同分配的指针不保证是唯一的。

1. 图间内存共享：

不同的图可以共享相同的物理内存，前提是它们不会同时运行。

CUDA 会自动管理物理内存的映射和复用，以减少内存占用和映射开销。

### 15.5 Performance considerations

当多个图（Graph）被提交到同一个流（Stream）中时，CUDA 会尝试为它们分配相同的物理内存，因为这些图的执行不会重叠。

为了优化性能，图的物理映射在多次启动之间会被保留，以避免重新映射的开销。

如果在后续某个时刻，其中一个图被提交到一个不同的流中执行（例如，可能导致与其他图的重叠执行），CUDA 必须执行重新内存映射，因为并发的图需要不同的内存以避免数据损坏。

通常，CUDA 中图内存的重新映射可能是由以下操作引起的：

1. 更改 graph launch stream：如果将 graph 提交到不同的 stream 中执行，可能会导致重新映射。
1. 修改 graph memory pool：显式释放未使用的内存（在 物理内存占用 部分讨论）。
1. 在另一个 graph 未释放存储时，relaunch 一个 graph：这会导致在重新启动前进行内存重新映射。

重新映射必须在执行顺序中进行，但在图的任何先前执行完成后进行（否则仍在使用中的内存可能会被取消映射）。由于这种顺序依赖性，以及映射操作是操作系统调用，映射操作可能会相对昂贵。应用程序可以通过将包含分配内存节点的图一致地提交到同一个流中来避免这种开销。

### 15.5.1 首次启动 / cudaGraphUpload

在图实例化期间无法分配或映射物理内存，因为图将执行的 stream 是未知的。

映射操作会在 graph launch 时进行。

调用 cudaGraphUpload 可以将分配的开销与启动分离，通过立即执行该图的所有映射并将其与上传流关联。

如果图随后被提交到同一个流中启动，它将无需额外的重新映射。

如果使用不同的流进行图上传和图启动，其行为类似于切换 stream ，可能会导致重新映射操作。

### 15.6 物理内存占用

1. 异步内存池管理：在 CUDA 中，使用 GPU 进行计算时，内存的分配和释放是异步进行的。这意味着，即使你已经释放了某些内存节点，这些内存并不会立即被操作系统回收，供其他程序使用。这样做的好处是可以提高内存分配的效率，因为内存池可以重复使用这些内存，而不需要每次都向操作系统申请。
1. 显式内存释放：如果你希望这些内存能够立即被操作系统回收，供其他程序使用，你需要显式地调用一个叫做 cudaDeviceGraphMemTrim 的 API。这个 API 会检查当前没有被使用的内存，并将它们释放回操作系统。但是，如果有些内存还在被使用（比如有些任务还在运行或者等待运行），这些内存就不会被释放。
1. 内存重新分配：当你使用 cudaDeviceGraphMemTrim 释放内存后，如果之后再次启动相关的 graph 任务，CUDA 会重新分配和映射内存。这意味着释放内存不会永久影响你的程序性能，只是在需要时会重新 allocate。
1. 内存使用情况查询：CUDA 还提供了 API（cudaDeviceGetGraphMemAttribute）来查询当前内存的使用情况。你可以查询两个属性：reserved，used：（cudaGraphMemAttrReservedMemCurrent），（cudaGraphMemAttrUsedMemCurrent）。

### 15.7 Peer Access（multi gpu

graph launch 可以配置为从多个 GPU 访问，在这种情况下，CUDA 会根据需要将分配映射到对等 GPU。

CUDA 允许需要不同映射的 graph 分配复用相同的虚拟地址。

当这种情况发生时，地址范围会被映射到不同 alloc 所需的所有 GPU。

这意味着分配有时可能允许比创建时请求的更多的对等访问；

### 15.7.1 使用 graph node API 进行 peer access

cudaGraphAddMemAllocNode API 接受节点参数结构中的 accessDescs 数组字段中的映射请求。

poolProps.location 嵌入结构指定分配的驻留设备。假设需要从分配 GPU 访问，因此应用程序不需要在 accessDescs 数组中为驻留设备指定条目。

这段代码展示了如何在 CUDA 中使用（cudaGraphAddMemAllocNode）来分配一块可以从多个 GPU 访问的内存。

它的核心是通过**访问描述符**（cudaMemAccessDesc）来指定哪些 GPU 可以访问这块内存：

cudaMemAllocNodeParams params = {};

params.poolProps.allocType = cudaMemAllocationTypePinned;

params.poolProps.location.type = cudaMemLocationTypeDevice;

// 指定设备 1 为驻留设备

params.poolProps.location.id = 1;

params.bytesize = size;

// 分配驻留在设备 1 上且可从设备 1 访问的内存

cudaGraphAddMemAllocNode(&allocNode, graph, NULL, 0, &params);

**// cudaMemAccessDesc** 是一个结构体，用于描述内存的访问权限和目标设备。

// 这里定义了一个包含 2 个元素的数组 accessDescs，每个元素描述了一个 GPU 的访问权限。

**// flags**：指定访问权限。cudaMemAccessFlagsProtReadWrite 表示允许读写访问。

**// location.type**：指定目标设备的类型。cudaMemLocationTypeDevice 表示目标是一个 GPU 设备。

cudaMemAccessDesc accessDescs[2];

accessDescs[0].flags = cudaMemAccessFlagsProtReadWrite;

accessDescs[0].location.type = cudaMemLocationTypeDevice;

accessDescs[1].flags = cudaMemAccessFlagsProtReadWrite;

accessDescs[1].location.type = cudaMemLocationTypeDevice;

// 指定目标设备 ID

accessDescs[0].location.id = 0;

accessDescs[1].location.id = 2;

// 设置内存分配参数

params.accessDescCount = 2;

params.accessDescs = accessDescs;

// 分配内存

cudaGraphAddMemAllocNode(&allocNode, graph, NULL, 0, &params);

这块内存的物理位置在设备 1 的显存中（即驻留在设备 1 上）。

设备 0 和设备 2 可以通过 Peer-to-Peer (P2P) 访问直接访问这块内存（通过访问描述符指定）。

设备 1 作为驻留设备，自然可以访问这块内存（不需要显式指定）。

### 15.7.2 使用 stream capture 进行 peer access

cudaMemAccessDesc accessDesc;

accessDesc.flags = cudaMemAccessFlagsProtReadWrite;

accessDesc.location.type = cudaMemLocationTypeDevice;

accessDesc.location.id = 1;

// 让 memPool 驻留在设备 0 上并可访问

cudaStreamBeginCapture(stream);

cudaMallocAsync(&dptr1, size, memPool, stream);

cudaStreamEndCapture(stream, &graph1);

// 更改 memPool 的对等可访问性

cudaMemPoolSetAccess(memPool, &accessDesc, 1);

cudaStreamBeginCapture(stream);

cudaMallocAsync(&dptr2, size, memPool, stream);

cudaStreamEndCapture(stream, &graph2);

// 分配 dptr1 的图节点仅具有设备 0 的可访问性，即使 memPool 现在具有设备 1 的可访问性。

// 分配 dptr2 的图节点将具有设备 0 和 1 的可访问性，因为这是 cudaMallocAsync 调用时池的可访问性。

## 第16章 Mathematical Functions

### 16.1 标准函数

本节中的函数可以同时在主机代码和设备代码中使用。

本节指定了每个函数在设备上执行时的误差范围，以及在主机上执行时的误差范围（如果主机未提供该函数）。

误差范围是通过广泛但不完全的测试生成的，因此这些范围并不是绝对保证的。

#### 单精度浮点函数

加法和乘法符合 IEEE 标准，因此最大误差为 0.5 ULP。

将单精度浮点数舍入为整数并返回单精度浮点数的推荐方法是使用 rintf()，而不是 roundf()。原因是 roundf()在设备上映射为 4 条指令，而 rintf()仅映射为 1 条指令。truncf()、ceilf()和 floorf()也各自映射为 1 条指令。

表 13 列出了单精度数学标准库函数的最大 ULP 误差。最大误差表示为 CUDA 库函数返回的结果与根据“舍入到最近偶数”模式计算得到的正确结果之间的 ULP 差的绝对值。

函数最大 ULP 误差 x + y0（IEEE-754 舍入到最近偶数）x * y0（IEEE-754 舍入到最近偶数）x / y0（计算能力≥2 且编译时使用-prec-div=true），否则 2（全范围）1 / x0（计算能力≥2 且编译时使用-prec-div=true），否则 1（全范围）rsqrtf(x)2（全范围）sqrtf(x)0（编译时使用-prec-sqrt=true），否则 1（计算能力≥5.2）或 3（旧架构）cbrtf(x)1（全范围）rcbrtf(x)1（全范围）hypotf(x, y)3（全范围）rhypotf(x, y)2（全范围）norm3df(x, y, z)3（全范围）rnorm3df(x, y, z)2（全范围）norm4df(x, y, z, t)3（全范围）rnorm4df(x, y, z, t)2（全范围）normf(dim, arr)无法提供误差范围，因为使用了快速算法，存在舍入误差。rnormf(dim, arr)无法提供误差范围，因为使用了快速算法，存在舍入误差。expf(x)2（全范围）exp2f(x)2（全范围）exp10f(x)2（全范围）expm1f(x)1（全范围）logf(x)1（全范围）log2f(x)1（全范围）log10f(x)2（全范围）log1pf(x)1（全范围）sinf(x)2（全范围）cosf(x)2（全范围）tanf(x)4（全范围）sincosf(x, sptr, cptr)2（全范围）sinpif(x)1（全范围）cospif(x)1（全范围）sincospif(x, sptr, cptr)1（全范围）asinf(x)2（全范围）acosf(x)2（全范围）atanf(x)2（全范围）atan2f(y, x)3（全范围）sinhf(x)3（全范围）coshf(x)2（全范围）tanhf(x)2（全范围）asinhf(x)3（全范围）acoshf(x)4（全范围）atanhf(x)3（全范围）powf(x, y)4（全范围）erff(x)2（全范围）erfcf(x)4（全范围）erfinvf(x)2（全范围）erfcinvf(x)4（全范围）erfcxf(x)4（全范围）normcdff(x)5（全范围）normcdfinvf(x)5（全范围）lgammaf(x)6（区间-10.001 到-2.264 之外；区间内更大）tgammaf(x)5（全范围）fmaf(x, y, z)0（全范围）frexpf(x, exp)0（全范围）ldexpf(x, exp)0（全范围）scalbnf(x, n)0（全范围）scalblnf(x, l)0（全范围）logbf(x)0（全范围）ilogbf(x)0（全范围）j0f(x)9（x< 8），否则最大绝对误差为 2.2×10⁻⁶j1f(x)9（x< 8），否则最大绝对误差为 2.2×10⁻⁶jnf(n, x)对于 n=128，最大绝对误差为 2.2×10⁻⁶y0f(x)9（x< 8），否则最大绝对误差为 2.2×10⁻⁶y1f(x)9（x< 8），否则最大绝对误差为 2.2×10⁻⁶ynf(n, x)ceil(2 + 2.5n)（x< n），否则最大绝对误差为 2.2×10⁻⁶cyl_bessel_i0f(x)6（全范围）cyl_bessel_i1f(x)6（全范围）fmodf(x, y)0（全范围）remainderf(x, y)0（全范围）remquof(x, y, iptr)0（全范围）modff(x, iptr)0（全范围）fdimf(x, y)0（全范围）truncf(x)0（全范围）roundf(x)0（全范围）rintf(x)0（全范围）nearbyintf(x)0（全范围）ceilf(x)0（全范围）floorf(x)0（全范围）lrintf(x)0（全范围）lroundf(x)0（全范围）llrintf(x)0（全范围）llroundf(x)0（全范围）

#### 双精度浮点函数

将双精度浮点数舍入为整数并返回双精度浮点数的推荐方法是使用 rint()，而不是 round()。原因是 round()在设备上映射为 5 条指令，而 rint()仅映射为 1 条指令。trunc()、ceil()和 floor()也各自映射为 1 条指令。

表 14 列出了双精度数学标准库函数的最大 ULP 误差。最大误差表示为 CUDA 库函数返回的结果与根据“舍入到最近偶数”模式计算得到的正确结果之间的 ULP 差的绝对值。

函数最大 ULP 误差 x + y0（IEEE-754 舍入到最近偶数）x * y0（IEEE-754 舍入到最近偶数）x / y0（IEEE-754 舍入到最近偶数）1 / x0（IEEE-754 舍入到最近偶数）sqrt(x)0（IEEE-754 舍入到最近偶数）rsqrt(x)1（全范围）cbrt(x)1（全范围）rcbrt(x)1（全范围）hypot(x, y)2（全范围）rhypot(x, y)1（全范围）norm3d(x, y, z)2（全范围）rnorm3d(x, y, z)1（全范围）norm4d(x, y, z, t)2（全范围）rnorm4d(x, y, z, t)1（全范围）norm(dim, arr)无法提供误差范围，因为使用了快速算法，存在舍入误差。rnorm(dim, arr)无法提供误差范围，因为使用了快速算法，存在舍入误差。exp(x)1（全范围）exp2(x)1（全范围）exp10(x)1（全范围）expm1(x)1（全范围）log(x)1（全范围）log2(x)1（全范围）log10(x)1（全范围）log1p(x)1（全范围）sin(x)2（全范围）cos(x)2（全范围）tan(x)2（全范围）sincos(x, sptr, cptr)2（全范围）sinpi(x)2（全范围）cospi(x)2（全范围）sincospi(x, sptr, cptr)2（全范围）asin(x)2（全范围）acos(x)2（全范围）atan(x)2（全范围）atan2(y, x)2（全范围）sinh(x)2（全范围）cosh(x)1（全范围）tanh(x)1（全范围）asinh(x)3（全范围）acosh(x)3（全范围）atanh(x)2（全范围）pow(x, y)2（全范围）erf(x)2（全范围）erfc(x)5（全范围）erfinv(x)5（全范围）erfcinv(x)6（全范围）erfcx(x)4（全范围）normcdf(x)5（全范围）normcdfinv(x)8（全范围）lgamma(x)4（区间-23.0001 到-2.2637 之外；区间内更大）tgamma(x)10（全范围）fma(x, y, z)0（IEEE-754 舍入到最近偶数）frexp(x, exp)0（全范围）ldexp(x, exp)0（全范围）scalbn(x, n)0（全范围）scalbln(x, l)0（全范围）logb(x)0（全范围）ilogb(x)0（全范围）j0(x)7（x< 8），否则最大绝对误差为 5×10⁻¹²j1(x)7（x< 8），否则最大绝对误差为 5×10⁻¹²jn(n, x)对于 n=128，最大绝对误差为 5×10⁻¹²y0(x)7（x< 8），否则最大绝对误差为 5×10⁻¹²y1(x)7（x< 8），否则最大绝对误差为 5×10⁻¹²yn(n, x)对于 x> 1.5n，最大绝对误差为 5×10⁻¹²cyl_bessel_i0(x)6（全范围）cyl_bessel_i1(x)6（全范围）fmod(x, y)0（全范围）remainder(x, y)0（全范围）remquo(x, y, iptr)0（全范围）modf(x, iptr)0（全范围）fdim(x, y)0（全范围）trunc(x)0（全范围）round(x)0（全范围）rint(x)0（全范围）nearbyint(x)0（全范围）ceil(x)0（全范围）floor(x)0（全范围）lrint(x)0（全范围）lround(x)0（全范围）llrint(x)0（全范围）llround(x)0（全范围）

### 16.2 内置函数

本节中的函数只能在设备代码中使用。

这些函数包括一些标准函数的低精度但更快的版本。它们的名称前缀为 __（例如 __sinf(x)）。它们更快，因为它们映射到更少的本地指令。编译器有一个选项（-use_fast_math），可以将表 15 中的每个函数编译为其内置函数版本。除了降低受影响函数的精度外，它还可能导致一些特殊情况的处理方式发生变化。更稳健的方法是仅在性能提升显著且可以容忍精度降低和特殊情况处理变化的情况下，选择性地将数学函数调用替换为内置函数调用。

表 15 列出了受-use_fast_math 影响的函数。

操作符/函数设备函数 x / y__fdividef(x, y)sinf(x)__sinf(x)cosf(x)__cosf(x)tanf(x)__tanf(x)sincosf(x, sptr, cptr)__sincosf(x, sptr, cptr)logf(x)__logf(x)log2f(x)__log2f(x)log10f(x)__log10f(x)expf(x)__expf(x)exp10f(x)__exp10f(x)powf(x, y)__powf(x, y)

### 单精度浮点函数

__fadd_[rn, rz, ru, rd]()和 __fmul_[rn, rz, ru, rd]()映射到加法和乘法操作，编译器永远不会将它们合并为 FMAD（Fused Multiply-Add）。相比之下，由*和+运算符生成的加法和乘法通常会合并为 FMAD。

后缀为 _rn 的函数使用“舍入到最近偶数”模式。

后缀为 _rz 的函数使用“向零舍入”模式。

后缀为 _ru 的函数使用“向上舍入（向正无穷）”模式。

后缀为 _rd 的函数使用“向下舍入（向负无穷）”模式。

浮点除法的精度取决于代码是否使用-prec-div=false 或-prec-div=true 编译。当使用-prec-div=false 编译时，常规除法运算符/和 __fdividef(x, y)具有相同的精度，但对于 2¹²⁶ < |y| < 2¹²⁸，__fdividef(x, y)会返回零，而/运算符会返回表 16 中所述精度的正确结果。此外，对于 2¹²⁶ < |y| < 2¹²⁸，如果 x 为无穷大，__fdividef(x, y)会返回 NaN（由于无穷大乘以零），而/运算符会返回无穷大。另一方面，当使用-prec-div=true 或未指定-prec-div 选项时，/运算符是 IEEE 兼容的，因为其默认值为 true。
