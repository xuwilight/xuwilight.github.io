---
title: CUDA C++ 笔记（十五）第19章与第9章——Unified Memory 与 CDP1
date: 2024-08-04 20:00:00
tags: [CUDA, Unified Memory, CDP, Dynamic Parallelism]
categories: [CUDA C++ Programming Guide]
description: 第19章介绍统一内存（Unified Memory）编程模型、cudaMallocManaged、预取与数据使用提示；第9章介绍传统 CUDA 动态并行（CDP1）的执行环境、内存模型与编程接口。
---

# 第19章 Unified Memory Programming

## 19.1. 统一内存介绍

CUDA 统一内存为所有处理器提供：

- 单一指针值使系统中的所有处理器（所有 CPU、所有 GPU 等）能够使用其所有本机内存操作（指针解除引用、原子等）访问该内存。
- 系统中的所有处理器都可以并发访问统一内存池。

统一内存通过多种方式改进 GPU 编程：

- 通过将数据迁移到最常访问它的处理器，并可以使用提示来控制迁移启发式方法。
- 通过避免在 CPU 和 GPU 上重复内存，可以减少总系统内存使用量。
- GPU 程序可以同时从 GPU 和 CPU 线程访问统一内存，而无需创建单独的分配（`cudaMalloc()`）并手动来回复制内存（`cudaMemcpy*()`）
- 它使 GPU 程序能够处理超出 GPU 内存容量的数据。

使用 CUDA 统一内存，数据移动仍会发生，使用提示可能会提高性能。这些提示对于正确性或功能性来说不是必需的，也就是说，程序员可以首先专注于在 GPU 和 CPU 之间并行化他们的应用程序，并在开发周期的后期进行数据移动作为性能优化。

获取 CUDA Unified Memory 主要有两种方式：

- **系统分配的内存（allocated）**：使用系统 API 在主机上分配的内存：堆栈变量、全局/文件范围变量、`malloc()`/`mmap()`（19.2）、线程本地变量等。
- **明确分配统一内存的 CUDA API（managed）**：例如，`cudaMallocManaged()`，更多系统上可用，并且可能比系统分配的内存表现更好。

### 19.1.1. 统一内存的系统要求

下表显示了对 CUDA Unified Memory 的不同支持级别、检测这些支持级别所需的设备属性以及每个支持级别特定文档的链接：

| 统一内存支持级别 | 系统 | 设备属性 | 更多文档 |
|---|---|---|---|
| 完整的 CUDA 统一内存：所有内存均得到全面支持。这包括系统分配的内存和 CUDA 管理的内存。 | 设置为 1：`pageableMemoryAccess`。具有硬件加速的系统还将以下属性设置为 1：`hostNativeAtomicSupported`，`pageableMemoryAccessUsesHostPageTable`，`directManagedMemAccessFromHost` | | 具有完整 CUDA 统一内存支持的设备上的统一内存 |
| 仅 CUDA 管理内存具有完全支持。 | 设置为 1：`concurrentManagedAccess`。设置为 0：`pageableMemoryAccess` | | 仅支持 CUDA 管理内存的设备上统一内存 |
| 未完全支持 CUDA 管理内存：统一寻址但不支持并发访问。 | 设置为 1：`managedMemory`。设置为 0：`concurrentManagedAccess` | | Windows 或具有计算能力 5.x 的设备上统一内存 |
| Tegra 内存管理的 CUDA Tegra 上的统一内存 | 不支持统一内存。设置为 0：`managedMemory` | | Tegra 内存管理的 CUDA |

如果应用程序尝试在不支持统一内存的系统上使用统一内存，其行为是未定义的。以下属性使 CUDA 应用程序能够检查系统对统一内存的支持级别，并能够在具有不同支持级别的系统之间移植：

- `pageableMemoryAccess`：在支持 CUDA 统一内存的系统上，此属性设置为 1，其中所有线程都可以访问系统分配内存和 CUDA 管理内存。这些系统包括 NVIDIA Grace Hopper、IBM Power9 + Volta 和启用了 HMM 的现代 Linux 系统（参见下一项）等。
- Linux HMM 需要 Linux 内核版本 6.1.24+、6.2.11+ 或 6.3+、计算能力为 7.5 或更高的设备以及安装有开放内核模块的 CUDA 驱动程序版本 535+。
- `concurrentManagedAccess`：在具有完整 CUDA 管理内存支持的系统上，此属性设置为 1。当此属性设置为 0 时，CUDA 管理内存中仅部分支持统一内存。有关 Tegra 对统一内存的支持，请参阅 CUDA for Tegra 内存管理。

程序可以通过查询上述 [统一内存支持级别概述表中的](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#table-unified-memory-levels) `cudaGetDeviceProperties()` 属性来查询 GPU 对 CUDA 统一内存的支持级别。

#### 19.1.2.6. 运行时检测统一内存支持级别

下面的示例展示了如何在运行时检测统一内存支持级别：

```cpp
int main() {
  int d;
  cudaGetDevice(&d);

  int pma = 0;
  cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, d);
  printf("Full Unified Memory Support: %s\n", pma == 1? "YES" : "NO");
  
  int cma = 0;
  cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess, d);
  printf("CUDA Managed Memory with full support: %s\n", cma == 1? "YES" : "NO");

  return 0;
}
```

![](/assets/cudacpp-24/image.png)

### 19.1.2. 编程模型

使用 CUDA 统一内存，主机和设备之间不再需要单独分配内存，也不再需要在它们之间进行显式内存传输。程序可以通过以下方式分配统一内存：

- **系统分配 API**：在具有完整 CUDA 统一内存支持的系统上，通过主机进程的任何系统分配（`malloc()`、C++ `new` 运算符、POSIX `mmap` 等）。
- **CUDA 管理内存分配 API**：通过 `cudaMallocManaged()` 语法上类似于的 API `cudaMalloc()`。
- **CUDA 管理变量**：用 `__managed__` 声明的变量，其语义类似于 `__device__` 变量。

以下示例说明了四种情况下统一内存如何简化 CUDA 程序：

```cpp
__global__ void write_value(int* ptr, int v) {
  *ptr = v;
}

int main() {
  // Requires System-Allocated Memory support
  int* ptr = (int*)malloc(sizeof(int));
  write_value<<<1, 1>>>(ptr, 1);
  // Synchronize required
  // (before, cudaMemcpy was synchronizing)
  cudaDeviceSynchronize();
  printf("value = %d\n", *ptr); 
  free(ptr); 
  return 0;
} // 堡垒机不支持

int main() {
  // Requires System-Allocated Memory support
  int value;
  write_value<<<1, 1>>>(&value, 1);
  // Synchronize required
  // (before, cudaMemcpy was synchronizing)
  cudaDeviceSynchronize();
  printf("value = %d\n", value);
  return 0;
} // 堡垒机不支持


int main() {
  int* ptr = nullptr;
  // Requires CUDA Managed Memory support
  cudaMallocManaged(&ptr, sizeof(int));
  write_value<<<1, 1>>>(ptr, 1);
  // Synchronize required
  // (before, cudaMemcpy was synchronizing)
  cudaDeviceSynchronize();
  printf("value = %d\n", *ptr); 
  cudaFree(ptr); 
  return 0;
}

// Requires CUDA Managed Memory support
__managed__ int value;

int main() {
  write_value<<<1, 1>>>(&value, 1);
  // Synchronize required
  // (before, cudaMemcpy was synchronizing)
  cudaDeviceSynchronize();
  printf("value = %d\n", value);
  return 0;
}
```

#### 19.1.2.2. CUDA 管理内存的分配 API：`cudaMallocManaged()`

在支持 CUDA 管理内存的系统上，可以使用以下方式分配统一内存：

```cpp
__host__ cudaError_t cudaMallocManaged(void **devPtr, size_t size);
```

此 API 在语法上与 `cudaMalloc()` 相同：它分配 size 托管内存的字节并设置 `devPtr` 为引用分配。CUDA 托管内存也使用 `cudaFree()` 释放。

[在具有完整 CUDA 托管内存支持的系统](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-requirements)上，系统中的所有 CPU 和 GPU 均可同时访问托管内存分配。将主机调用替换为 `cudaMalloc()` 不会 `cudaMallocManaged()` 影响这些系统上的程序语义；设备代码无法调用 `cudaMallocManaged()`。

```cpp
__global__ void printme(char *str) {
  printf(str);
}

int main() {
  // Allocate 100 bytes of memory, accessible to both Host and Device code
  char *s;
  cudaMallocManaged(&s, 100);
  // Note direct Host-code use of "s"
  strncpy(s, "Hello Unified Memory\n", 99);
  // Here we pass "s" to a kernel without explicitly copying
  printme<<< 1, 1 >>>(s);
  cudaDeviceSynchronize();
  // Free as for normal CUDA allocations
  cudaFree(s); 
  return  0;
}
```

#### 19.1.2.3. 使用全局范围管理变量 `__managed__`

CUDA `__managed__` 变量的行为就像是通过 [CUDA 管理内存的分配 API：`cudaMallocManaged()`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-explicit-allocation) 分配的一样。它们使用全局变量简化程序，使得在主机和设备之间交换数据变得特别容易，而无需手动分配或复制。`cudaMallocManaged()`

在[具有完整 CUDA 统一内存支持的系统](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-requirements)上，文件范围或全局范围变量无法由设备代码直接访问。但可以将指向这些变量的指针作为参数传递给内核，有关示例，请参阅[系统分配的内存：深入示例](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-system-allocator)。

```cpp
__global__ void write_value(int* ptr, int v) {
  *ptr = v;
}

// Requires CUDA Managed Memory support
__managed__ int value;

int main() {
  write_value<<<1, 1>>>(&value, 1);
  // Synchronize required
  // (before, cudaMemcpy was synchronizing)
  cudaDeviceSynchronize();
  printf("value = %d\n", value);
  return 0;
}
```

没有明确的 `cudaMemcpy()` 命令，并且返回值在 CPU 和 GPU 上都可见。

CUDA `__managed__` 变量隐含且等同于 `__device__`。但标记为的变量不能写作 `__managed__ __device__`、`__constant__` `__managed__`。

`__managed__` C++ 对象受到某些特定限制，尤其是涉及静态初始化器时。请参阅 [C++ 语言支持](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-cplusplus-language-support) 以获取这些限制的列表。

#### 19.1.2.4. 统一内存和映射内存之间的区别

[统一内存和映射内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mapped-memory)之间的主要区别在于：CUDA 映射内存不支持部分类型的内存访问（例如原子操作），而统一内存可以。CUDA 映射内存保证可移植支持的有限内存操作集比统一内存在更多系统上可用。

#### 19.1.2.5. 指针属性

CUDA 程序可以通过调用并测试指针 `cudaPointerGetAttributes()` 属性是否为来检查指针是否寻址 CUDA 管理内存分配。value `cudaMemoryTypeManaged`

此 API 返回 `cudaMemoryTypeHost` 已注册的系统分配内存 `cudaHostRegister()` 和 `cudaMemoryTypeUnregistered` CUDA 不知道的系统分配内存。

指针属性并不说明内存位于何处，而是说明内存是如何分配或注册的。

下面的示例显示如何在运行时检测指针的类型：

```cpp
char const* kind(cudaPointerAttributes a, bool pma, bool cma) {
    switch(a.type) {
    case cudaMemoryTypeHost: return pma?
      "Unified: CUDA Host or Registered Memory" :
      "Not Unified: CUDA Host or Registered Memory";
    case cudaMemoryTypeDevice: return "Not Unified: CUDA Device Memory";
    case cudaMemoryTypeManaged: return cma?
      "Unified: CUDA Managed Memory" : "Not Unified: CUDA Managed Memory";
    case cudaMemoryTypeUnregistered: return pma?
      "Unified: System-Allocated Memory" :
      "Not Unified: System-Allocated Memory";
    default: return "unknown";
    }
}

void check_pointer(int i, void* ptr) {
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, ptr);
  int pma = 0, cma = 0, device = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, device);
  cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess, device);
  printf("Pointer %d: memory is %s\n", i, kind(attr, pma, cma));
}

__managed__ int managed_var = 5;

int main() {
  int* ptr[5];
  ptr[0] = (int*)malloc(sizeof(int));
  cudaMallocManaged(&ptr[1], sizeof(int));
  cudaMallocHost(&ptr[2], sizeof(int));
  cudaMalloc(&ptr[3], sizeof(int));
  ptr[4] = &managed_var;

  for (int i = 0; i < 5; ++i) check_pointer(i, ptr[i]);
  
  cudaFree(ptr[3]);
  cudaFreeHost(ptr[2]);
  cudaFree(ptr[1]);
  free(ptr[0]);
  return 0;
}
```

默认未开启 full unified 情况时：

```
Pointer 0: memory is Not Unified: System-Allocated Memory
Pointer 1: memory is Unified: CUDA Managed Memory
Pointer 2: memory is Not Unified: CUDA Host or Registered Memory
Pointer 3: memory is Not Unified: CUDA Device Memory
Pointer 4: memory is Unified: CUDA Managed Memory
```

#### 19.1.2.7. GPU 内存超额申请

统一内存使应用程序能够超额申请任何单个处理器的内存，从而实现处理单 GPU 无法处理的内存，而不会增加编程模型的复杂性。在后面的例子里可以看到调优接口中可以指定 GPU 编号。

#### 19.1.2.8. 性能提示

以下部分介绍了可用的统一内存性能提示，这些提示可用于所有统一内存，例如 CUDA 托管内存，或者在[具有完整 CUDA 统一内存支持的系统](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-requirements)上，也可用于所有系统分配内存。这些 API 是提示，也就是说，它们不会影响应用程序的语义，只会影响其性能。也就是说，它们可以在任何应用程序的任何位置添加或删除，而不会影响其结果。

CUDA Unified Memory 可能并不总是具有做出与统一内存相关的最佳性能决策所需的所有信息。这些性能提示使应用程序能够向 CUDA 提供更多信息。

**19.1.2.8.1. 数据预取**

该 `cudaMemPrefetchAsync` API 是一种异步流排序 API，可将数据迁移到更靠近指定处理器的位置（目标设备或 HOST CPU）。在预取数据时可以访问数据。迁移直到流中的所有先前操作都完成后才开始，并在流中的任何后续操作之前完成。

```cpp
void test_prefetch_managed(cudaStream_t s) {
  char *data;
  cudaMallocManaged(&data, N);
  init_data(data, N);                                     // execute on CPU
  cudaMemPrefetchAsync(data, N, myGpuId, s);              // prefetch to GPU
  mykernel<<<(N + TPB - 1) / TPB, TPB, 0, s>>>(data, N);  // execute on GPU
  cudaMemPrefetchAsync(data, N, cudaCpuDeviceId, s);      // prefetch to CPU
  cudaStreamSynchronize(s);
  use_data(data, N);
  cudaFree(data);
}
```

**19.1.2.8.2. 数据使用提示**

当多个处理器同时访问相同的数据时，`cudaMemAdvise` 可用于提示如何访问数据

```cpp
cudaError_t cudaMemAdvise(const void *devPtr,
                         size_t count,
                         enum cudaMemoryAdvise advice,
                         int device);
```

其中 advice 可取以下值：

- `cudaMemAdviseSetReadMostly`：这意味着数据大部分时间都是读取，偶尔才会写入。一般来说，它允许在这个区域上用读取带宽换取写入带宽。
- `cudaMemAdviseSetPreferredLocation`：一般来说，任何内存都可以随时迁移到任何位置，例如，当给定处理器的物理内存不足时。此提示通过将数据的首选位置设置为属于设备的物理内存，告诉系统不希望将此内存区域从其首选位置迁移出去。`cudaCpuDeviceId` 为设备传入的值会将首选位置设置为 CPU 内存。其他提示（如 `cudaMemPrefetchAsync`）可能会覆盖此提示，导致内存从其首选位置迁移出去。
- `cudaMemAdviseSetAccessedBy`：在某些系统中，在从给定处理器访问数据之前建立内存映射可能有利于提高性能。此提示告诉系统数据将被频繁访问 device，从而使系统认为创建这些映射是值得的。此提示并不暗示数据应驻留在何处，但可以与 `cudaMemAdviseSetPreferredLocation` 结合使用以指定这一点。

也可以使用下列值之一来取消设置每个建议：`cudaMemAdviseUnsetReadMostly`、`cudaMemAdviseUnsetPreferredLocation` 和 `cudaMemAdviseUnsetAccessedBy`。

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#define TPB 256  // 每个线程块的线程数，视具体情况设置
#define maxOuterLoopIter 10
#define maxInnerLoopIter 5
#define maxDevices 2
int myGpuId = 0;
// Kernel 示例
__global__ void mykernel(const char *data, char* val, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        *val = data[idx];
    }
}

void init_data(char *data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<char>(i % 256);
    }
}
void test_advise_managed(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * TPB;  // 16 KiB
  // Allocate memory using cudaMallocManaged
  // (malloc may be used on systems with full CUDA Unified memory support)
  cudaMallocManaged(&dataPtr, dataSize);
  // Set the advice on the memory region
  cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, myGpuId);
  int outerLoopIter = 0;
  while (outerLoopIter < maxOuterLoopIter) {
    // The data is written to in the outer loop on the CPU
    init_data(dataPtr, dataSize);
    // The data is made available to all GPUs by prefetching.
    // Prefetching here causes read duplication of data instead
    // of data migration
    for (int device = 0; device < maxDevices; device++) {
      cudaMemPrefetchAsync(dataPtr, dataSize, device, stream);
    }
    // The kernel only reads this data in the inner loop
    int innerLoopIter = 0;
    
    char* t;
    cudaMallocManaged(&t, sizeof(char));
    while (innerLoopIter < maxInnerLoopIter) {
      mykernel<<<32, TPB, 0, stream>>>((const char *)dataPtr, t, dataSize);
      innerLoopIter++;
    }
    outerLoopIter++;
  }
  cudaFree(dataPtr);
}
void test_advise_managed_unset(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * TPB;  // 16 KiB
  // Allocate memory using cudaMallocManaged
  // (malloc may be used on systems with full CUDA Unified memory support)
  cudaMallocManaged(&dataPtr, dataSize);
  // Set the advice on the memory region
  cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseUnsetReadMostly, myGpuId);
  int outerLoopIter = 0;
  while (outerLoopIter < maxOuterLoopIter) {
    // The data is written to in the outer loop on the CPU
    init_data(dataPtr, dataSize);
    // The data is made available to all GPUs by prefetching.
    // Prefetching here causes read duplication of data instead
    // of data migration
    for (int device = 0; device < maxDevices; device++) {
      cudaMemPrefetchAsync(dataPtr, dataSize, device, stream);
    }
    // The kernel only reads this data in the inner loop
    int innerLoopIter = 0;
    char* t;
    cudaMallocManaged(&t, sizeof(char));
    while (innerLoopIter < maxInnerLoopIter) {
      mykernel<<<32, TPB, 0, stream>>>((const char *)dataPtr, t, dataSize);
      innerLoopIter++;
    }
    outerLoopIter++;
  }
  cudaFree(dataPtr);
}

void test_memcpy(cudaStream_t stream) {
  size_t dataSize = 64 * TPB;  // 16 KiB
  char *dataPtrHost = new char[dataSize/4];
  char *dataPtr;
  
  cudaMalloc((void**)&dataPtr, dataSize);
  cudaMemcpy(dataPtr, dataPtrHost, dataSize, cudaMemcpyHostToDevice);
  // Set the advice on the memory region
  
  int outerLoopIter = 0;
  while (outerLoopIter < maxOuterLoopIter) {
  
    int innerLoopIter = 0;
    
    char* t;
    cudaMallocManaged(&t, sizeof(char));
    while (innerLoopIter < maxInnerLoopIter) {
      mykernel<<<32, TPB, 0, stream>>>((const char *)dataPtr, t, dataSize);
      innerLoopIter++;
    }
    outerLoopIter++;
  }
  cudaFree(dataPtr);
}
void measure_time(cudaStream_t stream) {
    cudaEvent_t startEvent, stopEvent;

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, stream);
    test_memcpy(stream);
    cudaEventRecord(stopEvent, stream);
    cudaStreamSynchronize(stream);  

    // base
    cudaEventRecord(startEvent, stream);
    test_memcpy(stream);
    cudaEventRecord(stopEvent, stream);
    cudaStreamSynchronize(stream);  // 等待事件完成

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    std::cout << "test_memcpy took: " << elapsedTime << " ms" << std::endl;

    // set
    cudaEventRecord(startEvent, stream);
    test_advise_managed(stream);
    cudaEventRecord(stopEvent, stream);
    cudaStreamSynchronize(stream);  // 等待事件完成

    float elapsedTime1;
    cudaEventElapsedTime(&elapsedTime1, startEvent, stopEvent);
    std::cout << "test_advise_managed took: " << elapsedTime1 << " ms" << std::endl;

    // unset
    cudaEventRecord(startEvent, stream);
    test_advise_managed_unset(stream);
    cudaEventRecord(stopEvent, stream);
    cudaStreamSynchronize(stream);

    float elapsedTime2;
    cudaEventElapsedTime(&elapsedTime2, startEvent, stopEvent);
    std::cout << "test_advise_managed_unset took: " << elapsedTime2 << " ms" << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

int main() {
    cudaStream_t stream;
    cudaStreamCreate(&stream); 
    measure_time(stream); 
    cudaStreamDestroy(stream); 
    return 0;
}
```

手动管理最优，set 有一定效果。

**19.1.2.8.3. 查询托管内存上的数据使用属性**

```cpp
cudaMemRangeGetAttribute(void *data,
                         size_t dataSize,
                         enum cudaMemRangeAttribute attribute,
                         const void *devPtr,
                         size_t count);
```

查询起始于 `devPtr` 的大小为 count 字节的内存范围的属性。内存范围必须引用通过变量分配 `cudaMallocManaged` 或声明的托管内存 `__managed__`。可以查询以下属性：

- `cudaMemRangeAttributeReadMostly`：如果整个内存范围都设置了该属性，则返回结果为 1 `cudaMemAdviseSetReadMostly`，否则为 0。
- `cudaMemRangeAttributePreferredLocation`：返回的结果将是 GPU 设备 ID，或者 `cudaCpuDeviceId` 如果整个内存范围具有相应的处理器作为首选位置，`cudaInvalidDeviceId` 则将返回结果。应用程序可以使用此查询 API 根据托管指针的首选位置属性来决定通过 CPU 还是 GPU 暂存数据。请注意，查询时内存范围的实际位置可能与首选位置不同。
- `cudaMemRangeAttributeAccessedBy`：将返回针对该内存范围设置了该建议的设备列表。
- `cudaMemRangeAttributeLastPrefetchLocation`：将返回使用明确预取内存范围的最后一个位置 `cudaMemPrefetchAsync`。请注意，这仅返回应用程序请求预取内存范围的最后一个位置。它不会指示对该位置的预取操作是否已完成或甚至是否已开始。

另外，可以使用相应的函数查询多个属性 `cudaMemRangeGetAttributes`。

## 19.2. 具有完整 CUDA 统一内存支持的设备上的统一内存

（目前系统没开启）

#### 19.2.1 基本同上一章，增加了文件支持内存、static 的例子。

#### 19.2.1.1. 文件支持的统一内存

由于[具有完整 CUDA 统一内存支持的系统](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-requirements)允许设备访问主机进程拥有的任何内存，因此它们可以直接访问文件支持内存（比如 mmap 或 IPC 进程间通信）。

```cpp
__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - first %d characters: '", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
  printf("'\n");
}
void test_file_backed() {
  int fd = open(INPUT_FILE_NAME, O_RDONLY);
  ASSERT(fd >= 0, "Invalid file handle");
  struct stat file_stat;
  int status = fstat(fd, &file_stat);
  ASSERT(status >= 0, "Invalid file stats");
  char* mapped = (char*)mmap(0, file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  ASSERT(mapped != MAP_FAILED, "Cannot map file into memory");
  kernel<<<1, 1>>>("file-backed", mapped);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  ASSERT(munmap(mapped, file_stat.st_size) == 0, "Cannot unmap file");
  ASSERT(close(fd) == 0, "Cannot close file");
}
```

### 19.2.2. 性能调优

为了实现统一内存的良好性能，重要的是：

- 考虑分页避免缺页。
- 考虑调整应用程序以适应系统内存传输的粒度。

#### 19.2.2.1. 内存分页和页面大小

**19.2.2.1.1. 选择正确的页面大小**

一般来说，较小的页面大小会导致较少的（虚拟）内存碎片，但会导致更多的 TLB 未命中，而较大的页面大小会导致更多的内存碎片，但会导致较少的 TLB 未命中。此外，与较小的页面大小相比，较大的页面大小的内存迁移成本通常更高，因为我们通常迁移整个内存页面。这可能会导致使用大页面大小的应用程序出现更大的延迟峰值。有关页面错误的更多详细信息，另请参阅下一节。

性能调优的一个重要方面是，与 CPU 相比，TLB 未命中在 GPU 上的代价通常要高得多。这意味着，如果 GPU 线程频繁访问使用足够小的页面大小映射的统一内存的随机位置，那么与使用足够大的页面大小映射的统一内存的相同访问相比，它的速度可能会慢得多。虽然 CPU 线程随机访问使用小页面大小映射的大面积内存也会出现类似的效果，但速度减慢的程度并不明显，这意味着应用程序可能希望通过减少内存碎片来弥补这种速度减慢。

请注意，一般来说，应用程序不应根据给定处理器的物理页面大小调整其性能，因为物理页面大小可能会根据硬件而变化。上述建议仅适用于虚拟页面大小。

**19.2.2.1.2. CPU 和 GPU 页表：硬件一致性与软件一致性**

CPU 和 GPU 具有组合页表的系统称为*硬件一致性*系统。

CPU 和 GPU 具有单独页表的系统称为*软件一致性*系统。

硬件一致性系统（例如 NVIDIA Grace Hopper）为 CPU 和 GPU 提供了逻辑上组合的页表。这很重要，因为为了从 [GPU 访问系统分配的内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-system-allocator)，GPU 将使用 CPU 为请求的内存创建的页表条目。如果该页表条目使用默认的 CPU 页面大小 4KiB 或 64KiB，则访问大型虚拟内存区域将导致严重的 TLB 未命中，从而导致速度严重下降。

另一方面，在 CPU 和 GPU 各自具有逻辑页表的系统中，应考虑不同的性能调优方面：为了[保证一致性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-introduction)，这些系统通常在处理器访问映射到不同处理器的物理内存中的内存地址时使用*页面错误。这样的页面错误意味着：*

- 需要确保当前拥有的处理器（物理页面当前所在的位置）不能再访问该页面，方法是删除页表条目或更新它。
- 需要确保请求访问的处理器可以访问此页面，方法是创建新的页表条目或更新现有条目，以使其变为有效/活动。
- 必须将支持此虚拟页面的物理页面移动/迁移到请求访问的处理器：这可能是一个昂贵的操作，并且工作量与页面大小成正比。

总体而言，在 CPU 和 GPU 线程频繁并发访问同一内存页面的情况下，硬件一致系统与软件一致系统相比具有显著的性能优势：

- 更少的页面错误：这些系统不需要使用页面错误来模拟一致性或迁移内存，
- 更少的争用：这些系统在缓存行粒度而不是页面大小粒度上是一致的，也就是说，当一个缓存行内有来自多个处理器的争用时，只交换比最小页面大小小得多的缓存行，而当不同的处理器访问一个页面内的不同缓存行时，就不会发生争用。

#### 19.2.2.5. 统一内存下的 Memcpy()/Memset() 行为

`cudaMemcpy*()` 和 `cudaMemset*()` 接受任何统一内存指针作为参数。

对于 `cudaMemcpy*()`，指定 `cudaMemcpyKind` 是一个性能提示，如果任何参数是统一内存指针，则会产生更高的性能影响。

```cpp
__host__​ cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
```

因此，有以下性能建议：

- 当知道统一内存的物理位置时，使用准确的 `cudaMemcpyKind` 提示。
- 优于 `cudaMemcpyKindDefault` 不准确的 `cudaMemcpyKind` 暗示。
- 始终使用已填充（已初始化）的缓冲区：避免使用这些 API 来初始化内存。
- `cudaMemcpy*()` 如果两个指针都指向系统分配的内存，则避免使用：启动内核或使用 CPU 内存复制算法 `std::memcpy`。

## 19.3. 未完全支持 CUDA 统一内存的设备上的统一内存

分为两类，计算能力高于 6.0 的同 19.1，计算能力低于 6.0 的就不介绍了。

# 第9章 CDP1

本节总结了新版（CDP2）和旧版（CDP1）CUDA 动态并行接口之间的差异以及兼容性和互操作性。

## 9.5. CDP1 和 CDP2 之间的差异

对于 CDP2 或 cuda > 11.6 或计算能力 9.0 或更高的设备，不再可能进行显式设备端同步（即在设备代码中使用 `cudaDeviceSynchronize()`）。必须改用隐式同步（例如尾部启动）。

尝试使用 CDP2 或计算能力 9.0 或更高的设备查询或设置 `cudaLimitDevRuntimeSyncDepth`（或 `CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH`）会导致 `cudaErrorUnsupportedLimit`。

CDP2 不再具有用于不适合固定大小池的待处理启动的虚拟化池，且一次存在的事件总数有限制（请注意，事件仅在启动完成后才会销毁），等于待启动计数的两倍。[cudaLimitDevRuntimePendingLaunchCount](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#configuration-options) 必须设置为足够大，以避免耗尽启动槽。

使用 CDP2 或计算能力为 9.0 或更高的设备，按网格跟踪流，而不是按线程块跟踪，这允许将工作启动到由另一个线程块创建的流中。而尝试使用 CDP1 执行此操作会导致 `cudaErrorInvalidValue`。

CDP2 引入了尾部启动（[cudaStreamTailLaunch](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#basics)）和即发即弃（[cudaStreamFireAndForget](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#the-tail-launch-stream)）命名流。

CDP2 仅在 64 位编译模式下受支持。

## 9.5.2. 兼容性和互操作性

CDP2 是默认设置。可以使用 `-DCUDA_FORCE_CDP1_IF_SUPPORTED` 编译函数，以选择在计算能力低于 9.0 的设备上不使用 CDP2。

CUDA 12.0 及更新版本的函数编译器（默认）使用 CUDA 12.0 之前的版本或使用 CUDA 12.0 及更高版本编译的函数，并 `-DCUDA_FORCE_CDP1_IF_SUPPORTED` 指定（部分函数在 11.6 以上就已经弃用）汇编如果设备代码引用，则会出现编译错误 `cudaDeviceSynchronize`。如果代码引用 `cudaStreamTailLaunch` 或 `cudaStreamFireAndForget`，则会出现编译错误。如果设备代码引用 `cudaDeviceSynchronize` 并且代码是针对 sm_90 或更新版本编译的，则会出现编译错误。计算能力 < 9.0 使用 CDP2。使用 CDP1。计算能力 9.0 及更高使用 CDP2。使用 CDP2。如果 `cudaDeviceSynchronize` 在设备代码中引用函数，则函数加载会返回 `cudaErrorSymbolNotFound`（如果代码是为计算能力低于 9.0 的设备编译的，但使用 JIT 在计算能力为 9.0 或更高的设备上运行，则可能会发生这种情况）。

使用 CDP1 和 CDP2 的功能可以在同一上下文中同时加载和运行。CDP1 功能可以使用 CDP1 特定的功能（例如 `cudaDeviceSynchronize`），而 CDP2 功能可以使用 CDP2 特定的功能（例如尾部启动和即发即弃启动）。

使用 CDP1 的函数无法启动使用 CDP2 的函数，反之亦然。如果使用 CDP1 的函数在其调用图中包含使用 CDP2 的函数，则 `cudaErrorCdpVersionMismatch` 在函数加载期间会产生结果，反之亦然。

# 9.6. 传统 CUDA 动态并行 (CDP1)

（大部分同 cdp2，除了 9.5.1 提到的部分）

## 9.6.1. 执行环境和内存模型 (CDP1)

### 9.6.1.1. 执行环境 (CDP1)

CUDA 11.6 中已弃用与父块的子内核的显式同步（即在设备代码中使用 `cudaDeviceSynchronize()`），已从 compute_90+ 编译中删除，并计划在未来的 CUDA 版本中完全删除。

其余部分同 CDP2。

### 9.6.1.2. 内存模型 (CDP1)

父网格和子网格共享相同的全局和常量内存存储，但具有不同的本地和共享内存。

#### 9.6.1.2.1. 连贯性和一致性 (CDP1)

**9.6.1.2.1.1. 全局内存 (CDP1)**

父网格和子网格可以一致地访问全局内存，但子网格和父网格之间的一致性保证较弱。在执行子网格时，有两个点表明其内存视图与父线程完全一致：当父网格调用子网格时，以及当子网格完成时（由父线程中的同步 API 调用发出信号）。

在调用子网格之前，父线程中的所有全局内存操作对子网格都是可见的。在父网格同步子网格完成之后，父网格的所有内存操作对父网格都是可见的。

在以下示例中，执行 `child_launch` 的子网格只能保证看到在启动子网格之前对数据所做的修改。由于父网格的线程 0 正在执行启动，因此子网格将与父网格的线程 0 看到的内存一致。由于第一次 `__syncthreads()` 调用，子网格将看到 `data[0]=0`、`data[1]=1`、...、`data[255]=255`（如果没有 `__syncthreads()` 调用，则只能保证子网格看到 `data[0]`）。当子网格返回时，线程 0 可以保证看到其子网格中的线程所做的修改。只有在第二次 `__syncthreads()` 调用之后，这些修改才可供父网格的其他线程使用：

```cpp
__global__ void child_launch(int *data) {
   data[threadIdx.x] = data[threadIdx.x]+1;
}

__global__ void parent_launch(int *data) {
   data[threadIdx.x] = threadIdx.x;

   __syncthreads();

   if (threadIdx.x == 0) {
       child_launch<<< 1, 256 >>>(data);
       cudaDeviceSynchronize();
   }

   __syncthreads();
}

void host_launch(int *data) {
    parent_launch<<< 1, 256 >>>(data);
}
```

**9.6.1.2.1.2. 零拷贝内存 (CDP1)**

零拷贝系统内存具有与全局内存相同的一致性保证，并遵循上文详述的语义。内核不得分配或释放零拷贝内存，但可以使用从主机程序传入的零拷贝指针。

**9.6.1.2.1.3. 常量内存 (CDP1)**

常量是不可变的，即使在父级和子级启动之间，也不能从设备进行修改。也就是说，所有 `__constant__` 变量的值必须在启动前从主机设置。所有子内核都会自动从其各自的父级继承常量内存。

从内核线程中获取常量内存对象的地址与所有 CUDA 程序的语义相同，并且自然支持将该指针从父级传递到子级或从子级传递到父级。

**9.6.1.2.1.4. 共享和本地内存 (CDP1)**

共享和本地内存分别属于线程块或线程私有，在父级和子级之间不可见或不连贯。当其中一个位置中的对象在其所属范围之外被引用时，行为未定义，并且可能导致错误。

如果 NVIDIA 编译器可以检测到指向本地或共享内存的指针作为参数传递给内核启动，它将尝试发出警告。在运行时，程序员可以使用 `__isGlobal()` 内在函数来确定指针是否引用全局内存，因此可以安全地传递给子启动。

请注意，调用 `cudaMemcpy*Async()` 或 `cudaMemset*Async()` 可能会在设备上调用新的子内核，以保留流语义。因此，将共享或本地内存指针传递给这些 API 是非法的，并将返回错误。

**9.6.1.2.1.5. 本地内存 (CDP1)**

本地内存是执行线程的私有存储，在该线程之外不可见。在启动子内核时，将指向本地内存的指针作为启动参数传递是非法的。从子内核取消引用此类本地内存指针的结果将未定义。

例如，如果 `child_launch` 访问 `x_array`，则以下内容是非法的，行为未定义：

```cpp
int x_array[10]; // 在父级的本地内存中创建 x_array
child_launch<<< 1, 1 >>>(x_array);
```

有时很难知道编译器何时将变量放入本地内存。一般来说，传递给子内核的所有存储都应从全局内存堆中明确分配，可以使用 `cudaMalloc()`、`new()` 或在全局范围内声明 `__device__` 存储。例如：

```cpp
// Correct - "value" is global storage
__device__ int value;
__device__ void x() {
    value = 5;
    child<<< 1, 1 >>>(&value);
}
// Invalid - "value" is local storage
__device__ void y() {
    int value = 5;
    child<<< 1, 1 >>>(&value);
}
```

**9.6.1.2.1.6. 纹理内存 (CDP1)**

写入纹理所映射的全局内存区域与纹理访问不一致。纹理内存的一致性在调用子网格时以及子网格完成时强制执行。这意味着在子内核启动之前对内存的写入将反映在子级的纹理内存访问中。同样，子级对内存的写入将反映在父级的纹理内存访问中，但只有在父级在子级完成同步后才会反映。父级和子级的并发访问可能会导致数据不一致。

## 9.6.2. 编程接口 (CDP1)

### 9.6.2.1. CUDA C++ 参考 (CDP1)

#### 9.6.2.1.1. 设备端内核启动 (CDP1)

可以使用标准 CUDA `<<< >>>` 语法从设备启动内核：

```cpp
kernel_name<<< Dg, Db, Ns, S >>>([kernel arguments]);
```

- `Dg` 为 `dim3` 类型，指定网格的尺寸和大小
- `Db` 为 `dim3` 类型，指定每个线程块的尺寸和大小
- `Ns` 为 `size_t` 类型，指定为此调用每个线程块动态分配的共享内存的字节数以及静态分配的内存。`Ns` 是一个可选参数，默认为 0。
- `S` 为 `cudaStream_t` 类型，指定与此调用关联的流。该流必须已分配在进行调用的同一线程块中。`S` 是一个可选参数，默认为 0。

**9.6.2.1.1.1. 启动是异步的 (CDP1)**

与主机端启动相同，所有设备端内核启动相对于启动线程都是异步的。也就是说，`<<<>>>` 启动命令将立即返回，启动线程将继续执行，直到它到达显式启动同步点（例如 `cudaDeviceSynchronize()`）。

网格启动已发布到设备，并将独立于父线程执行。子网格可以在启动后的任何时间开始执行，但不能保证在启动线程到达显式启动同步点之前开始执行。

**9.6.2.1.1.2. 启动环境配置 (CDP1)**

所有全局设备配置设置（例如，从 `cudaDeviceGetCacheConfig()` 返回的共享内存和 L1 缓存大小，以及从 `cudaDeviceGetLimit()` 返回的设备限制）都将从父级继承。同样，堆栈大小等设备限制将保持配置状态。

对于主机启动的内核，从主机设置的每个内核配置将优先于全局设置。当从设备启动内核时，也将使用这些配置。无法从设备重新配置内核的环境。

#### 9.6.2.1.2. 流 (CDP1)

设备运行时提供命名和未命名（NULL）流。命名流可由线程块中的任何线程使用，但流句柄不得传递给其他块或子/父内核。换句话说，流应被视为创建它的块的私有流。流句柄在块之间不保证唯一，因此在未分配流句柄的块中使用流句柄将导致未定义的行为。

与主机端启动类似，启动到单独流中的工作可以同时运行，但不能保证实际的并发性。CUDA 编程模型不支持依赖于子内核之间并发性的程序，并且将具有未定义的行为。

设备不支持主机端 NULL 流的跨流屏障语义。为了保持与主机运行时的语义兼容性，必须使用 `cudaStreamCreateWithFlags()` API 创建所有设备流，并传递 `cudaStreamNonBlocking` 标志。`cudaStreamCreate()` 调用是仅限主机运行时的 API，无法为设备编译。

#### 9.6.2.1.3. 事件 (CDP1)

仅支持 CUDA 事件的流间同步功能。这意味着支持 `cudaStreamWaitEvent()`，但不支持 `cudaEventSynchronize()`、`cudaEventElapsedTime()` 和 `cudaEventQuery()`。由于不支持 `cudaEventElapsedTime()`，因此必须通过 `cudaEventCreateWithFlags()` 创建 `cudaEvents`，并传递 `cudaEventDisableTiming` 标志。

对于所有设备运行时对象，事件对象可以在创建它们的线程块内的所有线程之间共享，但仅限于该块，不能传递给其他内核，也不能在同一个内核的块之间传递。事件句柄不能保证在块之间是唯一的，因此在未创建它的块中使用事件句柄将导致未定义的行为。

#### 9.6.2.1.4. 同步 (CDP1)

`cudaDeviceSynchronize()` 函数将同步线程块中任何线程启动的所有工作，直到调用 `cudaDeviceSynchronize()` 为止。请注意，`cudaDeviceSynchronize()` 可以从不同代码中调用（请参阅块范围同步 (CDP1)）。

如果调用线程打算与从其他线程调用的子网格同步，则由程序执行足够的额外线程间同步，例如通过调用 `__syncthreads()`。

**9.6.2.1.4.1. 块范围同步 (CDP1)**

`cudaDeviceSynchronize()` 函数不表示块内同步。特别是，如果没有通过 `__syncthreads()` 指令进行显式同步，调用线程就无法假设除自身之外的任何线程启动了哪些工作。例如，如果块内的多个线程都在启动工作，并且需要同时同步所有这些工作（可能是由于基于事件的依赖关系），则程序必须保证在调用 `cudaDeviceSynchronize()` 之前所有线程都提交了这项工作。

由于允许实现在块中的任何线程启动时进行同步，因此多个线程同时调用 `cudaDeviceSynchronize()` 很可能会耗尽第一次调用中的所有工作，然后对后续调用没有任何影响。

#### 9.6.2.1.5. 设备管理 (CDP1)

有关 CDP2 版本的文档，请参阅上面的设备管理。

只有内核正在运行的设备才能从该内核控制。这意味着设备运行时不支持诸如 `cudaSetDevice()` 之类的设备 API。从 GPU 看到的活动设备（从 `cudaGetDevice()` 返回）将具有与从主机系统看到的相同的设备号。`cudaDeviceGetAttribute()` 调用可能会请求有关另一个设备的信息，因为此 API 允许将设备 ID 指定为调用的参数。请注意，设备运行时不提供 `cudaGetDeviceProperties()` API —— 必须单独查询属性。

#### 9.6.2.1.6. 内存声明 (CDP1)

**9.6.2.1.6.1. 设备和常量内存 (CDP1)**

使用设备运行时，在文件范围内使用 `__device__` 或 `__constant__` 内存空间说明符声明的内存行为相同。所有内核都可以读取或写入设备变量，无论内核最初是由主机还是设备运行时启动的。同样，所有内核都将具有与在模块范围内声明的 `__constant__`s 相同的视图。

**9.6.2.1.6.2. 纹理和表面 (CDP1)**

CUDA 支持动态创建的纹理和表面对象，其中可以在主机上创建纹理对象，将其传递给内核，由该内核使用，然后从主机销毁。设备运行时不允许在设备代码中创建或销毁纹理或表面对象，但可以使用从主机创建的纹理和表面对象。

```cpp
__global__ void permute(int n, int *data) {
   extern __shared__ int smem[];
   if (n <= 1)
       return;

   smem[threadIdx.x] = data[threadIdx.x];
   __syncthreads();

   permute_data(smem, n);
   __syncthreads();

   // Write back to GMEM since we can't pass SMEM to children.
   data[threadIdx.x] = smem[threadIdx.x];
   __syncthreads();

   if (threadIdx.x == 0) {
       permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data);
       permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data+n/2);
   }
}

void host_launch(int *data) {
    permute<<< 1, 256, 256*sizeof(int) >>>(256, data);
}
```

**9.6.2.1.6.4. 符号地址 (CDP1)**

设备端符号（即标记为 `__device__` 的符号）可通过 `&` 运算符从内核中引用，因为所有全局范围的设备变量都在内核的可见地址空间中。这也适用于 `__constant__` 符号，尽管在这种情况下指针将引用只读数据。

鉴于设备端符号可以直接引用，引用符号的 CUDA 运行时 API（例如，`cudaMemcpyToSymbol()` 或 `cudaGetSymbolAddress()`）是多余的，因此不受设备运行时支持。请注意，这意味着即使在子内核启动之前，也无法从正在运行的内核中更改常量数据，因为对 `__constant__` 空间的引用是只读的。

#### 9.6.2.1.7. API 错误和启动失败 (CDP1)

与 CUDA 运行时一样，任何函数都可能返回错误代码。返回的最后一个错误代码会被记录下来，可通过 `cudaGetLastError()` 调用检索。错误会按线程记录，因此每个线程都可以识别其生成的最新错误。错误代码的类型为 `cudaError_t`。

与主机端启动类似，设备端启动可能因多种原因（参数无效等）而失败。用户必须调用 `cudaGetLastError()` 来确定启动是否生成了错误，但启动后没有错误并不意味着子内核已成功完成。

对于设备端异常（例如访问无效地址），子网格中的错误将返回给主机，而不是由父级调用 `cudaDeviceSynchronize()` 返回。

**9.6.2.1.7.1. 启动设置 API (CDP1)**

内核启动是通过设备运行时库公开的系统级机制，因此可以通过底层 `cudaGetParameterBuffer()` 和 `cudaLaunchDevice()` API 直接从 PTX 获得。CUDA 应用程序可以自行调用这些 API，其要求与 PTX 相同。在这两种情况下，用户都有责任根据规范以正确的格式正确填充所有必要的数据结构。这些数据结构保证向后兼容。

与主机端启动一样，设备端运算符 `<<<>>>` 映射到底层内核启动 API。这样，以 PTX 为目标的用户将能够执行启动，并且编译器前端可以将 `<<<>>>` 转换为这些调用。

新的仅设备启动实现函数差异：

| 运行时 API 启动函数 | 与主机运行时行为差异的描述 |
|---|---|
| `cudaGetParameterBuffer` | 从 `<<<>>>` 自动生成。请注意与主机等效 API 不同。 |
| `cudaLaunchDevice` | 从 `<<<>>>` 自动生成。请注意与主机等效 API 不同。 |

这些启动函数的 API 与 CUDA 运行时 API 不同，定义如下：

```cpp
extern __device__ cudaError_t cudaGetParameterBuffer(void **params);
extern __device__ cudaError_t cudaLaunchDevice(void *kernel,
                                        void *params, dim3 gridDim,
                                        dim3 blockDim,
                                        unsigned int sharedMemSize = 0,
                                        cudaStream_t stream = 0);
```
