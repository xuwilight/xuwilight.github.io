---
title: CUDA C++ 笔记（三）第6章——Programming Interface（二）
date: 2024-06-09 20:00:00
tags: [CUDA, Stream, Graph]
categories: [CUDA C++ Programming Guide]
description: 本篇涵盖第 6 章 Programming Interface 的后半部分（6.2.8-6.2.15），包括异步并发执行、流（Streams）、程序化相关启动、CUDA Graphs、事件（Events）、多设备系统、错误检查等内容。
---

## 异步并发执行（Asynchronous Concurrent Execution）

CUDA 将以下操作公开为可以相互并发操作的独立任务：

- 主机上的计算；
- 设备上的计算；
- 内存传输从主机到设备；
- 内存传输从设备到主机；
- 给定设备内存内的内存传输；
- 设备之间的内存传输。

概括起来，CUDA 并行包含计算和存储传输两类，其中计算并行包括主机计算、设备计算 2 种；存储传输包括：主机到设备、设备到主机、设备内、设备间。

### 主机和设备之间的并发执行

通过异步库函数可以促进并发主机执行，这些函数在设备完成请求的任务之前将控制权返回给主机线程。使用异步调用，许多设备操作可以一起排队，以便在适当的设备资源可用时由 CUDA 驱动程序执行。这减轻了主机线程管理设备的大部分责任，使其可以自由地执行其他任务。以下设备操作相对于主机是异步的：

- 内核启动；
- 单个设备内存中的内存复制；
- 将 64 KB 或更小的内存块从主机复制到设备；
- 由带有 `Async` 后缀的函数执行的内存复制；
- 内存设置函数调用。

通过将 `CUDA_LAUNCH_BLOCKING` 环境变量设置为 1，程序员可以全局禁用系统上运行的所有 CUDA 应用程序的内核启动异步性。此功能仅用于调试目的，不应用作使生产软件可靠运行的方法。

### 并发内核执行（Concurrent Kernel Execution）

设备可以同时执行多个内核。应用程序可以通过检查并发内核设备属性来查询此功能。

### 数据传输和内核执行的重叠

在内核执行的同时，设备可以执行与 GPU 的异步内存复制，数据传输和内核执行的重叠，查询 API 为 `asyncEngineCount`，返回值大于 0 表示支持。

### 并发数据传输

设备可以重叠（Overlap）从设备的数据传入和传出。

## 流（Streams）

应用程序通过流管理并发操作，一条流是按顺序执行的命令序列（可能由不同的主机线程发出）。不同的流可能会相互乱序或同时执行命令；无法保证此行为，因此不应依赖其正确性。`synchronize` 同步调用的成功完成保证了所有启动的命令都已完成。

### 流的创建和销毁（Creation and Destruction of Streams）

流是通过创建流对象并将其指定为一系列内核启动和主机 <-> 设备内存副本的流参数来定义的。以下代码示例创建两个流并在 page-locked 页锁定内存中分配一个 float 数组 `hostPtr`。

```cpp
cudaStream_t stream[2];
for (int i = 0; i < 2; ++i)
    cudaStreamCreate(&stream[i]);
float* hostPtr;
cudaMallocHost(&hostPtr, 2 * size);
```

每个流均由以下代码示例定义为从主机到设备的一次内存复制、一次内核启动以及从设备到主机的一次内存复制的序列：

```cpp
for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size, size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>(outputDevPtr + i * size, inputDevPtr + i * size, size);
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size, size, cudaMemcpyDeviceToHost, stream[i]);
}
```

通过调用 `cudaStreamDestroy()` 来释放流。

```cpp
for (int i = 0; i < 2; ++i)
    cudaStreamDestroy(stream[i]);
```

### 默认流（Default Stream）

未指定任何流参数或等效地将流参数设置为零的内核启动和主机 <-> 设备内存副本将发布到默认流。因此它们是按顺序执行的。

对于使用 `--default-stream` 每线程编译标志编译的代码（或者在包含 CUDA 头文件 `cuda.h` 之前，定义 `CUDA_API_PER_THREAD_DEFAULT_STREAM` 宏）的代码，默认流是常规流，并且每个主机线程有自己的默认流。

### 显式同步（Explicit Synchronization）

有多种方法可以显式地相互同步流：

- `cudaDeviceSynchronize()` 等待所有主机线程的所有流中的所有先前命令完成。
- `cudaStreamSynchronize()` 将流作为参数并等待给定流中的所有先前命令完成。它可用于将主机与特定流同步，从而允许其他流继续在设备上执行。
- `cudaStreamWaitEvent()` 将流和事件作为参数（有关事件的描述，请参阅事件），并使在调用 `cudaStreamWaitEvent()` 后添加到给定流的所有命令延迟执行，直到给定事件完成。
- `cudaStreamQuery()` 为应用程序提供了一种方法来了解流中所有前面的命令是否已完成。

### 隐式同步（Implicit Synchronization）

如果主机线程在来自不同流的两个命令之间发出以下任一操作，则它们不能同时运行：

- page-locked 页锁定主机内存分配：`cudaMallocHost`
- 设备内存分配：`cudaMalloc`
- 设备内存设置：`cudaMemset`
- 两个地址之间的内存复制到同一设备内存
- 任何 CUDA 命令到 NULL 默认流
- L1/共享内存的切换，按 Compute Capability 7.x 描述

应用程序应遵循以下准则来提高并发内核执行的潜力：

- 所有独立操作应在相关操作之前发出，
- 任何类型的同步都应尽可能延迟。

### 重叠行为（Overlapping Behavior）

两个流之间的执行重叠量，取决于向每个流发出命令的顺序，以及设备是否支持数据传输和内核执行的重叠、并发内核执行和并发数据传输。

```cpp
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size, size, cudaMemcpyHostToDevice, stream[i]);
for (int i = 0; i < 2; ++i)
    MyKernel<<<100, 512, 0, stream[i]>>>(outputDevPtr + i * size, inputDevPtr + i * size, size);
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size, size, cudaMemcpyDeviceToHost, stream[i]);
```

### 主机函数（回调）（Host Functions (Callbacks)）

运行时提供了一种通过 `cudaLaunchHostFunc()` 在任意点将 CPU 函数调用插入到流中的方法。当之前向流发出的所有命令完成，会在主机上执行所提供的 host 函数。

![主机函数回调](/assets/cudacpp-6-2/image.png)

### 流优先级（Stream Priorities）

流的相对优先级可以在创建时使用 `cudaStreamCreateWithPriority()` 指定。可以使用 `cudaDeviceGetStreamPriorityRange()` 函数获取允许的优先级范围（按 [最高优先级、最低优先级] 排序）。在运行时，高优先级流中的待处理工作优先于低优先级流中的待处理工作。

以下代码示例获取当前设备允许的优先级范围，并创建具有最高和最低可用优先级的流。

![流优先级](/assets/cudacpp-6-2/image1.png)

## 程序化相关启动和同步

程序化依赖启动机制允许依赖的辅助内核在同一 CUDA 流中依赖的主内核完成执行之前启动。从计算能力 9.0 的设备开始可用，当辅助内核可以完成不依赖于主内核结果的重要工作时，该技术可以提供性能优势。

### 接口说明（API Description）

在程序化相关启动中，主内核和辅助内核在同一 CUDA 流中启动。当主内核准备好启动辅助内核时，应使用所有线程块执行 `cudaTriggerProgrammaticLaunchCompletion`。辅助内核必须使用可扩展启动 API 启动，如图所示。

当使用 `cudaLaunchAttributeProgrammaticStreamSerialization` 属性启动辅助内核时，CUDA 驱动程序可以安全地提前启动辅助内核，而不是在启动辅助内核之前等待主内核的完成和内存刷新。

当所有主线程块已启动并执行 `cudaTriggerProgrammaticLaunchCompletion` 时，CUDA 驱动程序可以启动辅助内核。如果主内核不执行触发器，则它会在主内核中的所有线程块退出后隐式发生。

```cpp
__global__ void primary_kernel() {
    // Initial work that should finish before starting secondary kernel
    // Trigger the secondary kernel
    cudaTriggerProgrammaticLaunchCompletion();
    // Work that can coincide with the secondary kernel
}
__global__ void secondary_kernel()
{
    // Independent work
    // Will block until all primary kernels the secondary kernel is dependent on have
    // completed and flushed results to global memory
    cudaGridDependencySynchronize();
    // Dependent work
}

cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attribute[0].val.programmaticStreamSerializationAllowed = 1;
configSecondary.attrs = attribute;
configSecondary.numAttrs = 1;
primary_kernel<<<grid_dim, block_dim, 0, stream>>>();
cudaLaunchKernelEx(&configSecondary, secondary_kernel);
```

## CUDA Graphs

Graph 是一系列通过依赖关系连接的操作，这些操作是与其执行分开定义。图形被定义一次，重复启动。将图的定义与其执行分离可以实现许多优化：首先，与流相比，CPU 启动成本降低，因为大部分设置都是提前完成的；其次，将整个工作流程呈现给 CUDA 可以实现优化。

使用图的工作提交分为三个不同的阶段：定义、实例化和执行。

算子对应图中一个节点，算子间的依赖关系是边，依赖关系限制了操作的执行顺序。一旦操作所依赖的节点完成，就可以随时安排操作。调度由 CUDA 系统决定。

### 节点类型

图形节点可以是以下之一：

- kernel
- CPU 函数调用
- memory copy
- memset
- 空节点
- 等待事件
- 记录事件
- 向外部信号量发送信号
- 等待外部信号量
- 条件节点
- 子图

### 使用 Graph API 创建图形

图创建有两种方式：显式 API 和流捕获。

可以使用 `cudaStreamIsCapturing()` 查询是否正在捕获流。

可以使用 `cudaStreamBeginCaptureToGraph()` 将工作捕获到现有图表中。

流捕获可以处理用 `cudaEventRecord()` 和 `cudaStreamWaitEvent()` 表示的跨流依赖关系。

### 更新实例化图

Graph 图表是工作流程的快照，包括内核、参数和依赖项，以便尽可能快速有效地重放它。在工作流程发生变化的情况下，图表就会过时并且必须进行修改。对图结构（例如拓扑或节点类型）的重大更改将需要重新实例化源图。

CUDA 提供了一种称为"图形更新"的轻量级机制，它允许就地修改某些节点参数，而无需重建整个图形。这比重新实例化要高效得多。

CUDA 提供了两种更新实例化图参数的机制：全图更新和单个节点更新。CUDA 还提供了一种启用和禁用各个节点而不影响其当前参数的机制。

#### 全图更新

`cudaGraphExecUpdate()` 允许使用拓扑相同的图（"更新"图）中的参数更新实例化图（"原始图"）。更新图的拓扑必须与用于实例化 `cudaGraphExec_t` 的原始图相同。

#### 单节点更新

实例化的图节点参数可以直接更新。这消除了实例化的开销以及创建新 `cudaGraph_t` 的开销。如果需要更新的节点数量相对于图中的节点总数较小，则最好单独更新节点。

以下方法可用于更新 `cudaGraphExec_t` 节点：

- `cudaGraphExecKernelNodeSetParams()`
- `cudaGraphExecMemcpyNodeSetParams()`
- `cudaGraphExecMemsetNodeSetParams()`
- `cudaGraphExecHostNodeSetParams()`
- `cudaGraphExecChildGraphNodeSetParams()`
- `cudaGraphExecEventRecordNodeSetEvent()`
- `cudaGraphExecEventWaitNodeSetEvent()`
- `cudaGraphExecExternalSemaphoresSignalNodeSetParams()`
- `cudaGraphExecExternalSemaphoresWaitNodeSetParams()`

#### 单个节点启用

单节点启用、获取启用状态：

- `cudaGraphNodeSetEnabled()`
- `cudaGraphNodeGetEnabled()`

### 设备图启动

有许多工作流需要在运行时做出依赖于数据的决策，并根据这些决策执行不同的操作。用户可能更愿意在设备上执行此决策过程，而不是将此决策过程卸载到主机（这可能需要从设备进行往返）。为此，CUDA 提供了一种从设备启动图形的机制。

可以从设备启动的图将被称为设备图，而不能从设备启动的图将被称为主机图。设备图可以从主机和设备启动，而主机图只能从主机启动。

#### 设备图创建

为了从设备启动图表，必须为设备启动显式实例化它。通过将 `cudaGraphInstantiateFlagDeviceLaunch` 标志传递给 `cudaGraphInstantiate()` 调用来实现。

#### 设备图上传

设备上启动图前，必须先上传到设备，上传两种方式如下：

首先，可以通过 `cudaGraphUpload()` 或通过 `cudaGraphInstantiateWithParams()` 请求上传作为实例化的一部分来显式上传图表。

或者，可以首先从主机启动图表，主机将在启动过程中隐式执行此上传步骤。

```cpp
// Explicit upload after instantiation
cudaGraphInstantiate(&deviceGraphExec1, deviceGraph1,
    cudaGraphInstantiateFlagDeviceLaunch);
cudaGraphUpload(deviceGraphExec1, stream);

// Explicit upload as part of instantiation
cudaGraphInstantiateParams instantiateParams = {0};
instantiateParams.flags = cudaGraphInstantiateFlagDeviceLaunch |
    cudaGraphInstantiateFlagUpload;
instantiateParams.uploadStream = stream;
cudaGraphInstantiateWithParams(&deviceGraphExec2, deviceGraph2, &instantiateParams);

// Implicit upload via host launch
cudaGraphInstantiate(&deviceGraphExec3, deviceGraph3,
    cudaGraphInstantiateFlagDeviceLaunch);
cudaGraphLaunch(deviceGraphExec3, stream);
```

#### 设备启动模式

设备图启动的几种模式：即发即忘启动、尾部启动、兄弟启动。

### 条件图节点

条件节点允许图的条件执行和循环。这使得动态和迭代工作流程能够在图表中完全呈现，并释放主机 CPU 来并行执行其他工作。

条件节点可以是以下类型之一：

- IF 节点：如果执行节点时条件值非零，则条件 IF 节点将执行其主体图一次。
- WHILE 节点：如果执行节点时条件值非零，条件 WHILE 节点将执行其主体图，并将继续执行其主体图，直到条件值为零。

条件值由 `cudaGraphConditionalHandle` 表示，并由 `cudaGraphConditionalHandleCreate()` 创建。

```cpp
__global__ void setHandle(cudaGraphConditionalHandle handle)
{
    ...
    cudaGraphSetConditional(handle, value);
    ...
}
void graphSetup() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[1];
    int value = 1;
    cudaGraphCreate(&graph, 0);
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph);

    // Use a kernel upstream of the conditional to set the handle value
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void *)setHandle;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;
    cudaGraphAddNode(&node, graph, NULL, 0, &params);
    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type = cudaGraphCondTypeIf;
    cParams.conditional.size = 1;
    cudaGraphAddNode(&node, graph, &node, 1, &cParams);
    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

    // Populate the body of the conditional node
    ...
    cudaGraphAddNode(&node, bodyGraph, NULL, 0, &params);
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}
```

## 事件（Events）

### 事件的创建和销毁

```cpp
// 事件创建
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// 事件销毁
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

### 计时

```cpp
cudaEventRecord(start, 0);
for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDev + i * size, inputHost + i * size, size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>(outputDev + i * size, inputDev + i * size, size);
    cudaMemcpyAsync(outputHost + i * size, outputDev + i * size, size, cudaMemcpyDeviceToHost, stream[i]);
}
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
```

## 同步调用

当调用同步函数时，在设备完成请求的任务之前，控制不会返回到主机线程。在主机线程执行任何其他 CUDA 调用之前，可以通过使用一些特定标志（请参阅参考手册了解详细信息）调用 `cudaSetDeviceFlags()` 来指定主机线程是让出、阻塞还是自旋。

## 多设备系统

### 设备枚举

```cpp
int deviceCount;
cudaGetDeviceCount(&deviceCount);
int device;
for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
}
```

### 设备选择

`cudaSetDevice`

如果将内核发布到与当前设备不关联的流，则内核启动将会失败。

### 点对点内存访问

`cudaDeviceEnablePeerAccess` 启用，`cudaDeviceCanAccessPeer` 查询是否支持。

点对点内存复制：用统一内存地址，或 Memcpy 操作：`cudaMemcpyPeer()`, `cudaMemcpyPeerAsync()`, `cudaMemcpy3DPeer()`, 或 `cudaMemcpy3DPeerAsync()`。

## 错误检查

检查异步错误的唯一方法是在调用 `cudaDeviceSynchronize()` 同步，并检查 `cudaDeviceSynchronize()` 返回的错误代码。

运行时为每个主机线程维护一个错误变量，该变量初始化为 `cudaSuccess` 并在每次发生错误时被错误代码覆盖。`cudaPeekAtLastError()` 返回此变量。`cudaGetLastError()` 返回此变量并将其重置为 `cudaSuccess`。

## 其它

- Texture and Surface Memory：图像处理
- Interprocess Communication：进程间通信
- CUDA User Objects：管理 pgraph 对象生命周期
- Unified Virtual Address Space：统一虚拟地址空间，CUDA 统一管理内存显存
