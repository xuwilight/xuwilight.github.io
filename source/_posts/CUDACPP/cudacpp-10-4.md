---
title: CUDA C++ 笔记（八）第10章——C++ Language Extensions（四）
date: 2024-06-29 20:00:00
tags: [CUDA, Device Functions, PTX, GPU]
categories: [CUDA C++ Programming Guide]
description: 分析器计数器、断言、陷阱/断点函数、printf、动态全局内存分配、执行配置、启动界限、寄存器数量控制、#pragma unroll、SIMD 视频指令、诊断程序
---

# 10.31 分析器计数器功能

每个 SM 都有一组（16个）硬件计数器，可以通过如下指令调用：

```cpp
void __prof_trigger(int counter);
```

指定 `counter` 的 SM 计数器每个 warp 自增 1；8~15 号计数器是保留的，不能被使用。

计数器的数值可以通过 `nvprof` 获得：`nvprof --events prof_trigger_0x`，`x` 取 0~7。

所有的计数器在 kernel 启动时重置。在收集计数器时，所有的 kernel 启动都是同步的。

# 10.32 断言

断言只有计算能力 2.x 以上的设备支持。

```cpp
void assert(int expression);
```

如果表达式等于零，则停止内核执行。如果程序在调试器中运行，这会触发一个断点，调试器可以用来检查设备的当前状态。否则，对于每个表达式等于零的线程，在通过 `cudaDeviceSynchronize()`、`cudaStreamSynchronize()` 或 `cudaEventSynchronize()` 与主机同步后，将向标准错误输出（stderr）打印一条消息。该消息的格式如下：

```text
<filename>:<line number>:<function>:
block: [blockIdx.x, blockIdx.y, blockIdx.z],
thread: [threadIdx.x, threadIdx.y, threadIdx.z]
Assertion `<expression>` failed.
```

对于同一设备的任何后续主机端同步调用都将返回 `cudaErrorAssert`。在调用 `cudaDeviceReset()` 重新初始化设备之前，无法向该设备发送更多命令。

如果表达式不等于零，则内核执行不会受到影响。例如，以下程序来自源文件 `test.cu`：

```cpp
#include <assert.h>
__global__ void testAssert(void)
{
    int is_one = 1;
    int should_be_one = 0;
    // This will have no effect
    assert(is_one);
    // This will halt kernel execution
    assert(should_be_one);
}
int main(int argc, char* argv[])
{
    testAssert<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

输出：

```text
test.cu:19: void testAssert(): block: [0,0,0], thread: [0,0,0] Assertion `should_be_one` failed.
```

断言用于调试目的。它们可能会影响性能，因此建议在生产代码中禁用断言。可以通过在包含 `assert.h` 之前定义预处理宏 `NDEBUG` 来在编译时禁用断言。需要注意的是，表达式不应包含副作用（例如类似于 `(++i > 0)` 的表达式），否则禁用断言会影响代码的功能。

# 10.33 陷阱函数

在设备端调用 `__trap()` 函数后，内核的执行将中止，并在主机程序中触发一个中断。

```cpp
void __trap();
```

示例：

```cpp
#include <cuda_runtime.h>
#include <iostream>

// 内核函数，示例使用 __trap()
__global__ void exampleKernel(int* data, int threshold) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 简单检查：如果数据超过阈值，触发 __trap()
    if (data[idx] > threshold) {
        printf("Error: data[%d] = %d exceeds threshold %d\n", idx, data[idx], threshold);
        __trap(); // 触发陷阱，终止内核
    }

    // 正常处理
    data[idx] += 1;
}

int main() {
    const int arraySize = 10;
    int hostData[arraySize] = {1, 2, 3, 15, 5, 6, 7, 8, 9, 10}; // 数据，其中 15 超过阈值
    int threshold = 10;
    int* deviceData;

    // 分配设备内存
    cudaMalloc(&deviceData, arraySize * sizeof(int));

    // 复制数据到设备
    cudaMemcpy(deviceData, hostData, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // 启动内核
    exampleKernel<<<1, arraySize>>>(deviceData, threshold);

    // 等待设备完成
    cudaError_t err = cudaDeviceSynchronize();

    // 检查是否发生了错误
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    } else {
        // 如果没有错误，复制结果回主机
        cudaMemcpy(hostData, deviceData, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "Processed data: ";
        for (int i = 0; i < arraySize; ++i) {
            std::cout << hostData[i] << " ";
        }
        std::cout << std::endl;
    }

    // 释放设备内存
    cudaFree(deviceData);

    return 0;
}
```

输出：

```text
Error: data[3] = 15 exceeds threshold 10
CUDA error: unspecified launch failure
```

# 10.34 断点函数

通过从任何设备线程调用 `__brkpt()` 函数，可以暂停内核函数的执行。

```cpp
void __brkpt();
```

该功能主要用于调试，通常需要配合调试器（如 `cuda-gdb`）来使用。

# 10.35 格式化输出

格式化输出仅支持计算能力 2.x 及更高的设备。

```cpp
int printf(const char* format[, arg, ...]);
```

从内核中使用 `printf()` 函数可以向主机端输出流打印格式化输出。

内核中的 `printf()` 函数的行为与标准 C 库的 `printf()` 函数类似，用户可以参考主机系统的手册页面以了解 `printf()` 的完整行为描述。本质上，作为 format 参数传入的字符串会被输出到主机的一个流中，当遇到格式说明符时，会从参数列表中进行替换。支持的格式说明符在后文列出。

`printf()` 命令作为任何其他设备端函数一样执行：逐线程执行，并在调用线程的上下文中运行。这意味着，在多线程内核中，每个线程对 `printf()` 的直接调用都会被执行，使用该线程指定的数据作为参数。主机流上会出现多个版本的输出字符串，每个调用 `printf()` 的线程都会输出一次。

如果只希望输出单个字符串，则需要程序员限制输出到单个线程。

与标准 C 的 `printf()` 不同，CUDA 的 `printf()` 返回解析的参数个数。如果没有参数跟随格式字符串，则返回 0。如果格式字符串为 NULL，则返回 -1。如果发生内部错误，则返回 -2。

## 10.35.1 格式说明符

与标准的 `printf()` 类似，格式说明符的形式为：

```text
%[flags][width][.precision][size]type
```

以下字段是支持的（有关所有行为的完整描述，请参阅广泛可用的文档）：

- **Flags（标志）**：`#`、` `（空格）、`0`、`+`、`-`
- **Width（宽度）**：`*`、`0-9`
- **Precision（精度）**：`0-9`
- **Size（大小修饰符）**：`h`、`l`、`ll`
- **Type（类型）**：`%cdiouxXpeEfgGaAs`

需要注意的是，CUDA 的 `printf()` 会接受任何标志、宽度、精度、大小修饰符和类型的组合，无论它们是否形成了一个有效的格式说明符。例如，`"%hd"` 是被接受的，并且 `printf` 会期望在参数列表中的相应位置找到一个双精度变量（double-precision variable）。

## 10.35.2 限制

`printf()` 的最终格式化操作发生在主机系统上。这意味着格式字符串必须被主机系统的编译器和 C 库理解。CUDA 的 `printf()` 函数尽量确保支持的格式说明符是常见主机编译器的通用子集，但其具体行为仍取决于主机操作系统。

正如 **格式说明符** 部分所描述，`printf()` 将接受所有标志和类型的有效组合。这是因为它无法判断在主机系统上进行最终格式化时哪些是有效的。其结果是，如果程序发出包含无效组合的格式字符串，输出可能是未定义的。

**参数数量限制**

`printf()` 最多可以接受 32 个参数（包括格式字符串本身）。超出此限制的参数会被忽略，并且格式说明符会原样输出。

**Windows 平台的特殊情况**

由于在 64 位 Windows 平台上，`long` 类型的大小为 4 字节（而其他 64 位平台上为 8 字节），如果一个内核在非 Windows 的 64 位机器上编译，但在 64 位 Windows 机器上运行，则包含 `"%ld"` 的格式字符串的输出会损坏。为了确保安全，建议编译平台与执行平台保持一致。

**输出缓冲区限制**

`printf()` 的输出缓冲区大小在内核启动前被固定。该缓冲区是循环的，如果在内核执行期间生成的输出超过缓冲区容量，较早的输出会被覆盖。缓冲区仅在以下操作发生时刷新：

- 内核启动（通过 `<<<>>>` 或 `cuLaunchKernel()`，在启动时刷新；如果环境变量 `CUDA_LAUNCH_BLOCKING` 设置为 1，则在启动结束时也刷新）。
- 同步操作（如 `cudaDeviceSynchronize()`、`cuCtxSynchronize()`、`cudaStreamSynchronize()`、`cuEventSynchronize()` 等）。
- 内存拷贝（任何阻塞版本的 `cudaMemcpy*()` 或 `cuMemcpy*()`）。
- 模块加载或卸载（通过 `cuModuleLoad()` 或 `cuModuleUnload()`）。
- 上下文销毁（通过 `cudaDeviceReset()` 或 `cuCtxDestroy()`）。
- 在执行通过 `cudaStreamAddCallback()` 或 `cuStreamAddCallback()` 添加的流回调之前。

需要注意的是，程序退出时缓冲区不会自动刷新。用户必须显式调用 `cudaDeviceReset()` 或 `cuCtxDestroy()`，如以下示例所示。

**线程间执行顺序的潜在影响**

`printf()` 在内部使用共享数据结构，因此调用 `printf()` 可能会改变线程的执行顺序。特别是，调用 `printf()` 的线程可能会经历比不调用 `printf()` 的线程更长的执行路径，并且路径长度取决于 `printf()` 的参数。然而，CUDA 仅在显式的 `__syncthreads()` 屏障上保证线程执行顺序，因此无法确定执行顺序的改变是由于 `printf()` 引起的，还是由于硬件调度的其他行为。

## 10.35.3 关联的主机端 API

以下 API 函数用于获取和设置缓冲区的大小，该缓冲区用于将 `printf()` 的参数和内部元数据传输到主机（默认大小为 1 MB）：

```cpp
cudaDeviceGetLimit(size_t* size, cudaLimitPrintfFifoSize)
cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size_t size)
```

## 10.35.4 示例

```cpp
#include <stdio.h>
__global__ void helloCUDA(float f)
{
    printf("Hello thread %d, f=%f\n", threadIdx.x, f);
}
int main()
{
    helloCUDA<<<1, 5>>>(1.2345f);
    cudaDeviceSynchronize();
    return 0;
}
```

```text
Hello thread 2, f=1.2345
Hello thread 1, f=1.2345
Hello thread 4, f=1.2345
Hello thread 0, f=1.2345
Hello thread 3, f=1.2345
```

```cpp
#include <stdio.h>
__global__ void helloCUDA(float f)
{
    if (threadIdx.x == 0) {
        printf("Hello thread %d, f=%f\n", threadIdx.x, f);
    }
}
int main()
{
    helloCUDA<<<1, 5>>>(1.2345f);
    cudaDeviceSynchronize();
    return 0;
}
```

```text
Hello thread 0, f=1.2345
```

# 10.36 动态全局内存分配和操作

动态全局内存分配和操作仅支持计算能力 **2.x 及更高** 的设备。

```cpp
// 从全局内存中的固定大小堆中动态分配和释放内存
__host__ __device__ void* malloc(size_t size);
__device__ void* __nv_aligned_device_malloc(size_t size, size_t align);
__host__ __device__ void free(void* ptr);

// 堆内存的拷贝
__host__ __device__ void* memcpy(void* dest, const void* src, size_t size);
// 向 ptr 指向的内存设置 size bytes 大小的数值（value），被转换为 unsigned char
__host__ __device__ void* memset(void* ptr, int value, size_t size);
```

CUDA 内核函数中的 `malloc()` 函数从设备堆中分配至少 `size` 字节的内存，并返回指向已分配内存的指针。如果内存不足以满足请求，则返回 `NULL`。返回的指针保证是 16 字节对齐的。

CUDA 内核函数中的 `__nv_aligned_device_malloc()` 函数从设备堆中分配至少 `size` 字节的内存，并返回指向已分配内存的指针。如果无法满足请求的大小或对齐要求，则返回 `NULL`。分配的内存地址将是 `align` 的倍数，其中 `align` 必须是非零的 2 的幂。

CUDA 内核函数中的 `free()` 函数释放由 `malloc()` 或 `__nv_aligned_device_malloc()` 之前分配的内存，该内存由 `ptr` 指向。如果 `ptr` 为 `NULL`，则对 `free()` 的调用将被忽略。对同一指针重复调用 `free()` 的行为未定义。

通过 `malloc()` 或 `__nv_aligned_device_malloc()` 由某个 CUDA 线程分配的内存在 CUDA 上下文的整个生命周期内保持分配状态，或者直到通过调用 `free()` 显式释放。分配的内存可以被任何其他 CUDA 线程使用，即使是在后续的 kernel 启动中。任何 CUDA 线程都可以释放由其他线程分配的内存，但必须注意确保同一个指针不会被多次释放。

## 10.36.1 堆内存分配

设备内存堆的大小是固定的，必须在任何使用 `malloc()`、`__nv_aligned_device_malloc()` 或 `free()` 的程序加载到上下文之前指定。如果程序使用 `malloc()` 或 `__nv_aligned_device_malloc()` 而未明确指定堆大小，则会分配一个默认大小为 8 MB 的堆。

以下 API 函数用于获取和设置堆大小：

```cpp
cudaDeviceGetLimit(size_t* size, cudaLimitMallocHeapSize)
cudaDeviceSetLimit(cudaLimitMallocHeapSize, size_t size)
```

分配的堆大小至少为 `size` 字节。`cuCtxGetLimit()` 和 `cudaDeviceGetLimit()` 返回当前请求的堆大小。

实际的堆内存分配发生在模块加载到上下文中时，可以通过 CUDA 驱动 API（参见 Module）显式加载，也可以通过 CUDA 运行时 API（参见 CUDA Runtime）隐式加载。如果内存分配失败，模块加载将产生 `CUDA_ERROR_SHARED_OBJECT_INIT_FAILED` 错误。

堆大小在模块加载后无法更改，并且不会根据需求动态调整。为设备堆保留的内存是额外的，与通过主机端 CUDA API 调用（例如 `cudaMalloc()`）分配的内存无关。

## 10.36.2 主机内存 API 的互操作性

通过设备端的 `malloc()` 或 `__nv_aligned_device_malloc()` 分配的内存不能使用运行时 API 的释放内存函数（例如设备内存的任何释放函数）释放。

同样，通过运行时 API 分配的内存（即调用设备内存的分配函数，例如 `cudaMalloc`）不能使用 `free()` 释放。

此外，在设备代码中通过 `malloc()` 或 `__nv_aligned_device_malloc()` 分配的内存，不能用于任何运行时或驱动 API 调用（例如 `cudaMemcpy`、`cudaMemset` 等）。

## 10.36.3 示例

### 10.36.3.1 线程级分配

```cpp
#include <stdlib.h>
#include <stdio.h>
__global__ void mallocTest()
{
    size_t size = 123;
    char* ptr = (char*)malloc(size);
    memset(ptr, 0, size);
    printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
    free(ptr);
}
int main()
{
    // Set a heap size of 128 megabytes. Note that this must
    // be done before any kernel is launched.
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
    mallocTest<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

上述代码输出：

```text
Thread 0 got pointer: 00057020
Thread 1 got pointer: 0005708c
Thread 2 got pointer: 000570f8
Thread 3 got pointer: 00057164
Thread 4 got pointer: 000571d0
```

### 10.36.3.2 Block 级分配

```cpp
#include <stdlib.h>
__global__ void mallocTest()
{
    __shared__ int* data;
    // The first thread in the block does the allocation and then
    // shares the pointer with all other threads through shared memory,
    // so that access can easily be coalesced.
    // 64 bytes per thread are allocated.
    if (threadIdx.x == 0) {
        size_t size = blockDim.x * 64;
        data = (int*)malloc(size);
    }
    __syncthreads();
    // Check for failure
    if (data == NULL) return;
    // Threads index into the memory, ensuring coalescence
    int* ptr = data;
    for (int i = 0; i < 64; ++i)
        ptr[i * blockDim.x + threadIdx.x] = threadIdx.x;
    // Ensure all threads complete before freeing
    __syncthreads();
    // Only one thread may free the memory!
    if (threadIdx.x == 0) free(data);
}
int main()
{
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
    mallocTest<<<10, 128>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### 10.36.3.3 分配在内核启动之间的持久性

```cpp
#include <stdio.h>
#include <stdlib.h>
#define NUM_BLOCKS 20
__device__ int* dataptr[NUM_BLOCKS];  // Per-block pointer
__global__ void allocmem() {
  // Only the first thread in the block does the allocation
  // since we want only one allocation per block.
  if (threadIdx.x == 0) dataptr[blockIdx.x] = (int*)malloc(blockDim.x * 4);
  __syncthreads();
  // Check for failure
  if (dataptr[blockIdx.x] == NULL) return;
  // Zero the data with all threads in parallel
  dataptr[blockIdx.x][threadIdx.x] = 0;
}
// Simple example: store thread ID into each element
__global__ void usemem() {
  int* ptr = dataptr[blockIdx.x];
  if (ptr != NULL) ptr[threadIdx.x] += threadIdx.x;
}
// Print the content of the buffer before freeing it
__global__ void freemem() {
  int* ptr = dataptr[blockIdx.x];
  if (ptr != NULL)
    printf("Block %d, Thread %d: final value = %d\n", blockIdx.x, threadIdx.x,
           ptr[threadIdx.x]);
  // Only free from one thread!
  if (threadIdx.x == 0) free(ptr);
}
int main() {
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
  // Allocate memory
  allocmem<<<NUM_BLOCKS, 10>>>();
  // Use memory
  usemem<<<NUM_BLOCKS, 10>>>();
  usemem<<<NUM_BLOCKS, 10>>>();
  usemem<<<NUM_BLOCKS, 10>>>();
  // Free memory
  freemem<<<NUM_BLOCKS, 10>>>();
  cudaDeviceSynchronize();
  return 0;
}
```

# 10.37 执行配置

任何对 `__global__` 函数的调用都必须指定该调用的执行配置。执行配置定义了在设备上执行函数所使用的网格（Grid）和线程块（Block）的维度，以及关联的流（Stream）（有关流的描述，请参阅 CUDA 运行时文档）。

执行配置通过在函数名和括号内的参数列表之间插入如下形式的表达式来指定：

```cpp
<<< Dg, Db, Ns, S >>>
```

其中：

- **Dg**：
  - 类型为 `dim3`（参见 `dim3`），指定网格的维度和大小。
  - 其中 `Dg.x * Dg.y * Dg.z` 等于启动的块的总数。
- **Db**：
  - 类型为 `dim3`（参见 `dim3`），指定每个线程块的维度和大小。
  - 其中 `Db.x * Db.y * Db.z` 等于每个块中的线程总数。
- **Ns**：
  - 类型为 `size_t`，指定每个块为此调用动态分配的共享内存字节数，除了静态分配的共享内存之外。
  - 这些动态分配的内存可由任何声明为外部数组的变量使用（如 `__shared__` 中提到的）。
  - `Ns` 是一个可选参数，默认值为 0。
- **S**：
  - 类型为 `cudaStream_t`，指定关联的流。
  - `S` 是一个可选参数，默认值为 0。

执行配置的参数会在实际的函数参数之前被计算。

如果以下情况发生，函数调用将失败：

- `Dg` 或 `Db` 超过了设备允许的最大尺寸（详见计算能力的限制）。
- `Ns` 超过了设备支持的最大共享内存量（即总共享内存减去静态分配所需的共享内存量）。

对于计算能力 **9.0 及以上**，用户可以指定编译时的线程块簇（Cluster）维度，从而在 CUDA 中利用簇的层次结构。编译时簇的维度可以通过 `__cluster_dims__([x, [y, [z]]])` 指定。以下示例展示了在 X 维度为 2，Y 和 Z 维度为 1 的簇大小的编译时配置：

```cpp
__global__ void __cluster_dims__(2, 1, 1) Func(float* parameter);
```

线程块簇的维度也可以在运行时动态指定，并通过 `cudaLaunchKernelEx` API 启动包含簇的内核函数。该 API 接收以下参数：

1. 类型为 `cudaLaunchConfig_t` 的配置参数。
2. 内核函数指针。
3. 内核参数。

以下示例展示了运行时的内核配置：

```cpp
__global__ void Func(float* parameter);
// Kernel invocation with runtime cluster size
{
    cudaLaunchConfig_t config = {0};
    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension should be a multiple of cluster size.
    config.gridDim = Dg;
    config.blockDim = Db;
    config.dynamicSmemBytes = Ns;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;
    float* parameter;
    cudaLaunchKernelEx(&config, Func, parameter);
}
```

# 10.38 启动界限

正如在 **Multiprocessor Level** 中详细讨论的那样，内核使用的寄存器越少，一个多处理器（Multiprocessor）上可以驻留的线程和线程块就越多，这可能会提升性能。

因此，编译器会使用启发式方法来尽量减少寄存器的使用，同时尽量降低寄存器溢出（见 **Device Memory Accesses**）和指令数量的影响。不过，应用程序可以通过在 `__global__` 函数的定义中使用 `__launch_bounds__()` 限定符，为编译器提供额外的信息，以辅助这些启发式优化。

```cpp
__global__ void
__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor, maxBlocksPerCluster)
MyKernel(...)
{
    ...
}
```

#### maxThreadsPerBlock

- 指定应用程序在调用 `MyKernel()` 时，线程块中线程的最大数量。
- 编译为 PTX 指令 `.maxntid`。
- 如果以超过 `maxThreadsPerBlock` 指定数量的线程调用内核，启动将失败。

#### minBlocksPerMultiprocessor（可选）

- 指定在一个多处理器（SM）上希望驻留的线程块的最小数量。
- 编译为 PTX 指令 `.minnctapersm`。

#### maxBlocksPerCluster（可选）

- 指定应用程序调用 `MyKernel()` 时，每个簇中线程块的最大数量。
- 编译为 PTX 指令 `.maxclusterrank`。
- 如果以超过 `maxBlocksPerCluster` 指定数量的线程块调用内核，启动将失败。

当指定了启动界限时，编译器会首先根据这些界限推导出寄存器使用数量的上限 `L`，以确保以下条件之一得到满足：

1. 每个多处理器可以驻留 `minBlocksPerMultiprocessor` 指定数量的线程块。
2. 如果未指定 `minBlocksPerMultiprocessor`，则至少可以驻留一个线程块，且线程数不超过 `maxThreadsPerBlock`。

如果指定了启动界限（launch bounds），编译器会首先根据这些界限推导出内核应使用的寄存器数量的上限 `L`。这个上限能够确保每个多处理器上可以驻留至少 **`minBlocksPerMultiprocessor`** 个块（如果未指定 **`minBlocksPerMultiprocessor`**，则至少一个块），而每个块包含 **`maxThreadsPerBlock`** 个线程（详见硬件多线程部分关于内核使用寄存器数量与每块分配寄存器数量的关系）。接着，编译器会通过以下方式优化寄存器的使用：

- **如果初始的寄存器使用量高于 L**：编译器会进一步减少寄存器使用量，直到其不超过 `L`。这种优化通常会以更多的本地内存使用和/或更高的指令数量为代价。
- **如果初始的寄存器使用量低于 L**：
  - **如果只指定了 `maxThreadsPerBlock` 而未指定 `minBlocksPerMultiprocessor`**：编译器会使用 **`maxThreadsPerBlock`** 来确定每个多处理器上驻留 `n` 和 `n+1` 个块之间的寄存器使用阈值（例如，在减少一个寄存器的使用可以增加一个驻留块的情况下，如多处理器级别的示例中描述），然后应用类似于未指定启动界限时的启发式方法。
  - **如果同时指定了 `minBlocksPerMultiprocessor` 和 `maxThreadsPerBlock`**：编译器可能会将寄存器使用量增加到 `L` 的最高值，以减少指令数量，从而更好地隐藏单线程指令的延迟。

如果一个内核的启动块中线程数超出其启动界限 **`maxThreadsPerBlock`**，则该内核启动会失败。

如果一个内核的集群中线程块数超出其启动界限 **`maxBlocksPerCluster`**，则该内核启动会失败。

CUDA 内核所需的每线程资源可能会以不希望的方式限制最大块大小。为了在未来硬件和工具包中保持向前兼容性，并确保至少有一个线程块能够运行在 SM（Streaming Multiprocessor）上，开发者应该包含单参数形式的 `__launch_bounds__(maxThreadsPerBlock)`，用以指定内核启动时允许的最大块大小。如果未指定，可能会导致"启动请求的资源过多"错误。

在某些情况下，使用双参数形式的 `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` 可以提高性能。具体的 **`minBlocksPerMultiprocessor`** 值应通过对内核的详细分析来确定。

对于给定内核的最优启动边界（launch bounds），通常会因主要架构版本的不同而有所变化。以下示例代码展示了如何在设备代码中使用 **CUDA_ARCH** 宏（详见应用兼容性部分）来处理此问题。

```cpp
#define THREADS_PER_BLOCK          256
#if __CUDA_ARCH__ >= 200
    #define MY_KERNEL_MAX_THREADS  (2 * THREADS_PER_BLOCK)
    #define MY_KERNEL_MIN_BLOCKS   3
#else
    #define MY_KERNEL_MAX_THREADS  THREADS_PER_BLOCK
    #define MY_KERNEL_MIN_BLOCKS   2
#endif
// Device code
__global__ void
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
MyKernel(...)
{
    ...
}
```

在常见情况下，**MyKernel** 以每块最大线程数（由 **`launch_bounds()`** 的第一个参数指定）调用时，通常会倾向于使用 **`MY_KERNEL_MAX_THREADS`** 作为执行配置中的每块线程数。

```cpp
// Host code
MyKernel<<<blocksPerGrid, MY_KERNEL_MAX_THREADS>>>(...);
```

然而，这种方法无法正常工作，因为在主机代码中 **`CUDA_ARCH`** 是未定义的（详见应用兼容性部分），因此即使 **`CUDA_ARCH`** 大于或等于 200，**MyKernel** 仍将以 256 个线程每块的配置启动。相反，每块的线程数应通过以下方式确定：

- **在编译时**，使用不依赖 **`CUDA_ARCH`** 的宏，例如：
- **在运行时**，根据计算能力（compute capability）确定。

寄存器使用情况可以通过编译器选项 **`-ptxas-options=-v`** 报告。驻留块的数量可以通过 CUDA 分析器报告的占用率（详见设备内存访问部分关于占用率的定义）推导出来。

**`launch_bounds()`** 和 **`maxnreg()`** 修饰符不能应用于同一个内核。寄存器使用量也可以通过 **`maxrregcount`** 编译器选项控制文件中所有的 **`__global__`** 函数。然而，对于具有启动界限（launch bounds）的函数，**`maxrregcount`** 的值将被忽略。

# 10.39 每线程的最大寄存器数量

为了提供低级性能调优的机制，CUDA C++ 提供了 **`maxnreg()`** 函数修饰符，用于向后端优化编译器传递性能调优信息。**`maxnreg()`** 修饰符用于指定在线程块中分配给单个线程的最大寄存器数量。在 **`__global__`** 函数的定义中：

```cpp
__global__ void
__maxnreg__(maxNumberRegistersPerThread)
MyKernel(...)
{
    ...
}
```

- **`maxNumberRegistersPerThread`** 指定内核 **`MyKernel()`** 的线程块中，分配给单个线程的最大寄存器数量；它会编译为 PTX 指令 **`.maxnreg`**。

**`launch_bounds()`** 和 **`maxnreg()`** 修饰符不能应用于同一个内核。寄存器使用量也可以通过 **`maxrregcount`** 编译器选项控制文件中所有的 **`__global__`** 函数。然而，对于带有 **`maxnreg`** 修饰符的函数，**`maxrregcount`** 的值将被忽略。

# 10.40 #pragma unroll

默认情况下，编译器会展开（unroll）具有已知循环次数的小循环。然而，`#pragma unroll` 指令可以用于控制特定循环的展开行为。它必须紧接在目标循环之前，并且仅适用于该循环。该指令后可以选择性地跟随一个整型常量表达式（ICE, Integral Constant Expression）。如果没有提供 ICE 且循环的迭代次数是常量，则循环会被完全展开。如果 ICE 的值为 1，则编译器不会展开该循环。如果 ICE 的值为非正整数或者超过 `int` 数据类型所能表示的最大值，该指令将被忽略。

```cpp
struct S1_t {
  static const int value = 4;
};
template <int X, typename T2>
__device__ void foo(int* p1, int* p2) {
  // no argument specified, loop will be completely unrolled
  #pragma unroll
    for (int i = 0; i < 12; ++i) p1[i] += p2[i] * 2;
  // unroll value = 8
  #pragma unroll(X + 1)
    for (int i = 0; i < 12; ++i) p1[i] += p2[i] * 4;
  // unroll value = 1, loop unrolling disabled
  #pragma unroll 1
    for (int i = 0; i < 12; ++i) p1[i] += p2[i] * 8;
  // unroll value = 4
  #pragma unroll(T2::value)
    for (int i = 0; i < 12; ++i) p1[i] += p2[i] * 16;
}
__global__ void bar(int* p1, int* p2) { foo<7, S1_t>(p1, p2); }
```

# 10.41 SIMD 视频指令

PTX ISA（并行线程执行指令集架构）版本 3.0 包括 SIMD（单指令多数据）视频指令，这些指令可以对成对的 16 位值或四组 8 位值进行操作。这些指令适用于计算能力（compute capability）为 3.0 的设备。

SIMD 视频指令包括：

- **`vadd2`, `vadd4`**
- **`vsub2`, `vsub4`**
- **`vavrg2`, `vavrg4`**
- **`vabsdiff2`, `vabsdiff4`**
- **`vmin2`, `vmin4`**
- **`vmax2`, `vmax4`**
- **`vset2`, `vset4`**

这些 PTX 指令（包括 SIMD 视频指令）可以通过 CUDA 程序中的汇编语句 `asm()` 加以使用。

`asm()` 语句的基本语法为：

```cpp
asm("模板字符串" : "约束"(输出) : "约束"(输入));
```

以下是使用 `vabsdiff4` PTX 指令的示例：

```cpp
asm("vabsdiff4.u32.u32.u32.add" " %0, %1, %2, %3;" : "=r"(result) : "r"(A), "r"(B), "r"(C));
```

该示例使用 `vabsdiff4` 指令以 SIMD 方式计算四字节整数的绝对差值之和。绝对差值的计算是针对无符号整数 A 和 B 的每个字节进行的，并采用 SIMD 方式完成。此外，还指定了可选的累加操作（`.add`），用于对这些差值求和。

有关在代码中使用汇编语句的详细信息，请参阅文档 *Using Inline PTX Assembly in CUDA*。有关 PTX 指令（例如 PTX ISA 版本 3.0）的详细信息，请参阅 PTX ISA 文档（如 *Parallel Thread Execution ISA Version 3.0*）。

# 10.42 诊断程序

以下编译指令（pragmas）可用于控制发出特定诊断消息时的错误严重级别。

```cpp
#pragma nv_diag_suppress
#pragma nv_diag_warning
#pragma nv_diag_error
#pragma nv_diag_default
#pragma nv_diag_once
```

这些指令的使用形式如下：

```cpp
#pragma nv_diag_xxx error_number, error_number ...
```

受影响的诊断消息由警告消息中显示的错误编号指定。任何诊断消息都可以被重定义为错误，但只有警告可以降低其严重性，或在被提升为错误后恢复为警告。`nv_diag_default` 编译指令用于将诊断消息的严重性恢复为未应用任何编译指令前的状态（即消息的正常严重性，该严重性可能已被任何命令行选项修改过）。

以下示例抑制了"声明但从未引用"的警告，针对 `foo` 的声明：

```cpp
#pragma nv_diag_suppress 177
void foo()
{
    int i = 0;
}
#pragma nv_diag_default 177
void bar()
{
    int i = 0;
}
```

以下编译指令（pragmas）可用于保存和恢复当前诊断编译指令的状态：

```cpp
#pragma nv_diagnostic push
#pragma nv_diagnostic pop
```

```cpp
#pragma nv_diagnostic push
#pragma nv_diag_suppress 177

void foo() {
    int i = 0; // 此处抑制了警告 177
}

#pragma nv_diagnostic pop
void bar() {
    int i = 0; // 此处恢复了默认的诊断状态
}
```

请注意，这些编译指令（pragmas）仅对 **nvcc CUDA 前端编译器** 生效，对主机编译器无任何影响。

**移除通知：**

从 **CUDA 12.0** 开始，不带 `nv_` 前缀的诊断编译指令将不再受支持：

- 如果这些指令出现在设备代码中，将触发警告：**"在设备代码中未识别的 #pragma"**。
- 如果这些指令出现在主机代码中，它们会被直接传递给主机编译器处理。

如果这些指令是为 CUDA 代码设计的，请改用带有 `nv_` 前缀的指令，以确保兼容性。
