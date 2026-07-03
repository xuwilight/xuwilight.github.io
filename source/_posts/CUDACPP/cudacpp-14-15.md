---
title: CUDA C++ 笔记（十一）第14-15章——虚拟内存管理与 Stream Ordered Memory Allocator
date: 2024-07-11 20:00:00
tags: [CUDA, Virtual Memory, Memory Allocator]
categories: [CUDA C++ Programming Guide]
description: CUDA 虚拟内存管理(VMM) API 将地址与内存解耦，提供细粒度的 GPU 内存控制。Stream Ordered Memory Allocator 通过 cudaMallocAsync/cudaFreeAsync 实现流序内存分配，支持内存池、重用策略和跨进程共享。
---

## 第13章 虚拟内存管理

**13.1 引言**

虚拟内存管理（Virtual Memory Management, VMM）API 提供了一种方式，让应用程序能够直接管理 CUDA 提供的统一虚拟地址空间（Unified Virtual Address Space），该地址空间将物理内存映射到 GPU 可访问的虚拟地址。在 CUDA 10.2 中引入的这些 API 还新增了与其他进程及图形 API（如 OpenGL 和 Vulkan）的互操作性，并提供了可供用户调整的新内存属性，以满足特定应用需求。

在历史上，CUDA 编程模型中的内存分配调用（如 cudaMalloc()）会返回指向 GPU 内存的地址。获取的地址可以用于任何 CUDA API 或设备内核中。然而，分配的内存无法根据用户的需求调整大小。若要增加分配的内存大小，用户需要显式地分配一个更大的缓冲区，复制初始分配的数据，释放旧的分配，然后继续跟踪新的分配地址。这种操作通常会导致应用性能下降和峰值内存使用量增高。本质上，用户可以使用类似 malloc 的接口分配 GPU 内存，但却没有对应的 realloc 来进行扩展操作。

虚拟内存管理 API 将地址与内存的概念解耦，让应用程序可以分别处理它们。用户可以根据需要将内存映射到虚拟地址范围，或从中取消映射。

在使用 cudaEnablePeerAccess 为内存分配启用对等设备访问时，所有过去和未来的用户分配都会被映射到目标对等设备。这可能导致用户无意中为所有 cudaMalloc 分配支付映射到对等设备的运行时成本。然而，大多数情况下，应用程序仅需要与另一个设备共享少量分配，而不是将所有分配映射到所有设备。借助虚拟内存管理，应用程序可以选择性地将某些分配设置为对目标设备可访问。

CUDA 虚拟内存管理 API 为用户提供了对 GPU 内存管理的精细控制，具体包括：

- 将不同设备上分配的内存放置在连续的虚拟地址范围内。

- 使用特定平台机制进行内存共享的跨进程通信。

- 在支持的新设备上选择使用新的内存类型。

为了分配内存，虚拟内存管理编程模型提供了以下功能：

- 分配物理内存。

- 保留虚拟地址范围（VA range）。

- 将分配的内存映射到虚拟地址范围。

- 控制映射范围的访问权限。

需要注意的是，本节描述的 API 套件需要支持统一虚拟地址空间（UVA）的系统。

**13.2 支持查询**

在尝试使用虚拟内存管理 API 之前，应用程序必须确保所使用的设备支持 CUDA 虚拟内存管理。以下代码示例展示了如何查询设备是否支持虚拟内存管理：

```cpp
int deviceSupportsVmm;
CUresult result = cuDeviceGetAttribute(
    &deviceSupportsVmm,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
    device
);
if (deviceSupportsVmm != 0) {
    // `device` 支持虚拟内存管理
}
```

**13.3 分配物理内存**

使用虚拟内存管理 API 分配内存的第一步是创建一个物理内存块，作为分配的后备支持。为了分配物理内存，应用程序必须使用 cuMemCreate API。此函数创建的分配不包含任何设备或主机映射。参数 CUmemGenericAllocationHandle 描述了内存分配的属性，例如分配的位置、分配是否会共享给其他进程（或其他图形 API），以及所分配内存的物理属性。

用户必须确保请求的分配大小与适当的粒度对齐。有关分配粒度要求的信息可以使用 cuMemGetAllocationGranularity 查询。以下代码片段展示了如何使用 cuMemCreate 分配物理内存：

```cpp
CUmemGenericAllocationHandle allocatePhysicalMemory(int device, size_t size) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;

    size_t granularity = 0;
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // 确保分配大小满足粒度要求
    size_t padded_size = ROUND_UP(size, granularity);

    // 分配物理内存
    CUmemGenericAllocationHandle allocHandle;
    cuMemCreate(&allocHandle, padded_size, &prop, 0);

    return allocHandle;
}
```

**13.3.3 物理内存的映射和共享**

**13.3.3.1 映射到虚拟地址范围**

cuMemCreate 分配的内存通过返回的 CUmemGenericAllocationHandle 引用。这与 cudaMalloc 风格的内存分配不同，后者返回一个指向 GPU 内存的指针，可供 CUDA 内核直接访问。cuMemCreate 分配的内存只能用于通过 cuMemGetAllocationPropertiesFromHandle 查询属性。为了使这些内存可访问，应用程序必须通过 cuMemAddressReserve 保留一个虚拟地址范围（VA range），并将内存映射到该范围，同时设置合适的访问权限。释放内存时需使用 cuMemRelease。

**13.3.3.2 可共享内存分配**

通过 cuMemCreate，用户现在可以在分配时告知 CUDA，特定分配将用于进程间通信（IPC）或图形互操作。可以通过设置 CUmemAllocationProp::requestedHandleTypes 来指定平台相关的共享句柄类型。例如：

- 在 Windows 平台上，设置为 CU_MEM_HANDLE_TYPE_WIN32 时，还需指定 CUmemAllocationProp::win32HandleMetaData 中的 LPSECURITYATTRIBUTES，定义导出分配的安全范围。

- 在 Linux 平台上，设置为 CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR。

虚拟内存管理 API 替代了传统的进程间通信功能，提供了一种基于操作系统特定句柄的新机制。用户可通过 cuMemExportToShareableHandle 获取分配的句柄，并通过操作系统的本地进程间通信机制传递句柄。接收方需使用 cuMemImportFromShareableHandle 导入该分配。

在尝试导出通过 cuMemCreate 分配的内存前，用户必须查询目标设备是否支持所需的句柄类型。以下代码展示了如何查询支持：

```cpp
int deviceSupportsIpcHandle;
#if defined(__linux__)
cuDeviceGetAttribute(
    &deviceSupportsIpcHandle,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
    device
);
#else
cuDeviceGetAttribute(
    &deviceSupportsIpcHandle,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED,
    device
);
#endif

用户可以根据需要设置 CUmemAllocationProp::requestedHandleTypes：

#if defined(__linux__)
prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_WIN32;
prop.win32HandleMetaData = // Windows-specific LPSECURITYATTRIBUTES 属性。
#endif
```

**13.3.4 内存类型**

**13.3.4.1 可压缩内存**

可压缩内存可以加速对具有非结构化稀疏性和其他可压缩数据模式的数据的访问。其优势包括节省 DRAM 带宽、L2 读取带宽和 L2 容量。用户可以通过设置 CUmemAllocationProp::allocFlags::compressionType 为 CU_MEM_ALLOCATION_COMP_GENERIC 来分配可压缩内存，但需确保设备支持该功能。可通过以下代码查询设备是否支持数据压缩：

```cpp
int compressionSupported = 0;
cuDeviceGetAttribute(
    &compressionSupported,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED,
    device
);
```

在支持数据压缩的设备上，用户需在分配时启用压缩：

```cpp
prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
```

由于硬件资源限制等原因，分配可能未启用压缩属性。用户需通过 cuMemGetAllocationPropertiesFromHandle 查询分配属性以确认是否已启用压缩：

```cpp
CUmemAllocationProp allocationProp = {};
cuMemGetAllocationPropertiesFromHandle(&allocationProp, allocationHandle);
if (allocationProp.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC) {
    // 获得了可压缩内存分配
}
```

**13.4 保留虚拟地址范围**

在虚拟内存管理中，地址与内存的概念是分离的。应用程序必须预留一个虚拟地址范围来容纳通过 cuMemCreate 分配的物理内存。地址范围的大小需至少等于计划放入其中的所有物理内存分配的总大小。

通过调用 cuMemAddressReserve 并传入适当的参数，用户可以保留一个虚拟地址范围。保留的范围不会与任何设备或主机的物理内存相关联，可用于映射系统中任意设备的内存块，从而为应用程序提供一个连续的虚拟地址范围。

在释放虚拟地址范围前，用户需确保整个范围已取消映射。以下代码展示了如何使用 cuMemAddressReserve：

```cpp
CUdeviceptr ptr; // `ptr` 保存保留的虚拟地址范围的起始地址。
CUresult result = cuMemAddressReserve(&ptr, size, 0, 0, 0); // alignment = 0 表示使用默认对齐
```

**13.5 虚拟别名支持**

虚拟内存管理 API 支持为同一分配创建多个虚拟内存映射（“代理”）的功能。这可以通过多次调用 cuMemMap 并指定不同的虚拟地址实现。

需要注意的是，除非在 PTX ISA 中另有说明，对分配的一个代理的写入与对同一分配的其他代理的写入和读取之间可能是非一致和非同步的。以下示例展示了未定义的行为：

```cpp
__global__ void foo(char *A, char *B) {
    *A = 0x1;
    printf("%d\n", *B); // 未定义行为
}
```

如果需要在同一内核中通过不同“代理”访问相同分配，可以使用 fence.proxy.alias 来确保访问一致性。例如：

```cpp
__global__ void foo(char *A, char *B) {
    *A = 0x1;
    asm volatile ("fence.proxy.alias;" ::: "memory");
    printf("%d\n", *B); // 保证一致性
}
```

**13.6 内存映射**

虚拟内存管理 API 中，物理内存和虚拟地址空间的分离由 cuMemCreate 和 cuMemAddressReserve 分别实现。为了使分配的物理内存可用，用户必须通过 cuMemMap 将物理内存映射到预留的虚拟地址范围（VA range）。

用户可以将多个设备的分配映射到连续的虚拟地址范围，只需预留足够的地址空间。为了解除物理内存与地址范围的关联，可以使用 cuMemUnmap 取消映射地址。用户可在同一地址范围内多次映射和取消映射，但需避免在已映射的 VA 范围内创建新映射。以下代码片段展示了 cuMemMap 的使用：

```cpp
CUdeviceptr ptr; // 从 cuMemAddressReserve 获得的地址范围
CUmemGenericAllocationHandle allocHandle; // 从 cuMemCreate 获取的内存句柄
CUresult result = cuMemMap(ptr, size, 0, allocHandle, 0);
```

**13.7 控制访问权限**

虚拟内存管理 API 提供了显式的访问控制机制，允许用户保护虚拟地址范围。通过 cuMemMap 将内存映射到地址范围并不会使地址变得可访问。如果 CUDA 内核尝试访问未授权的地址，将导致程序崩溃。用户需使用 cuMemSetAccess 明确设置地址范围的访问权限。以下代码展示了设置访问权限的示例：

```cpp
void setAccessOnDevice(int device, CUdeviceptr ptr, size_t size) {
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    cuMemSetAccess(ptr, size, &accessDesc, 1); // 设置访问权限
}
```

此机制允许用户精确控制需要共享给其他设备的分配。与 cudaEnablePeerAccess 不同，虚拟内存管理的访问控制机制具有更细粒度的映射控制，从而减少性能开销。

**13.8 Fabric Memory**

CUDA 12.4 引入了新的 VMM 分配句柄类型 CU_MEM_HANDLE_TYPE_FABRIC。在支持的平台上，且 NVIDIA IMEX 守护进程运行时，该类型允许通过 MPI 等机制实现节点间的内存共享。Fabric Memory 使得同属一个 NVLINK Fabric 的多节点 GPU 能够互相映射彼此的内存，即便它们处于不同节点中，从而显著提高了多 GPU 编程的规模。

**13.8.1 支持查询**

使用 Fabric Memory 前，需确保目标设备支持该功能。以下代码展示了查询支持的示例：

```cpp
int deviceSupportsFabricMem;
CUresult result = cuDeviceGetAttribute(
    &deviceSupportsFabricMem,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
    device
);
if (deviceSupportsFabricMem != 0) {
    // 设备支持 Fabric Memory
}
```

Fabric Memory 的使用方式与其他分配句柄类型类似，但不需要操作系统的进程间通信机制来交换共享句柄。

**13.9 多播支持（Multicast Support）**

多播对象管理 API 提供了一种机制，结合虚拟内存管理 API，允许应用程序利用 NVLINK SHARP 在支持 NVSWITCH 的 NVLINK 连接 GPU 上加速广播和归约操作。多播团队中的每个 GPU 都会为多播对象备份一个本地物理内存副本。

**13.9.1 支持查询**

使用多播对象前，需确保目标设备支持该功能。以下代码展示了查询支持的示例：

```cpp
int deviceSupportsMultiCast;
CUresult result = cuDeviceGetAttribute(
    &deviceSupportsMultiCast,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
    device
);
if (deviceSupportsMultiCast != 0) {
    // 设备支持多播对象
}
```

**13.9.2 分配多播对象**

可以通过 cuMulticastCreate 创建多播对象：

```cpp
CUmemGenericAllocationHandle createMCHandle(int numDevices, size_t size) {
    CUmemAllocationProp mcProp = {};
    mcProp.numDevices = numDevices;
    mcProp.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC; // 或单节点 CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

    size_t granularity = 0;
    cuMulticastGetGranularity(&granularity, &mcProp, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    size_t padded_size = ROUND_UP(size, granularity);
    mcProp.size = padded_size;

    CUmemGenericAllocationHandle mcHandle;
    cuMulticastCreate(&mcHandle, &mcProp);
    return mcHandle;
}
```

**13.9.3 将设备加入多播团队**

可以通过 cuMulticastAddDevice 将设备加入多播团队：

```cpp
cuMulticastAddDevice(&mcHandle, device);
```

需在所有设备进程中完成此操作，随后才能为多播对象绑定物理内存。

**13.9.4 为多播对象绑定内存**

绑定物理内存需使用 cuMulticastBindMem：

```cpp
cuMulticastBindMem(mcHandle, mcOffset, memHandle, memOffset, size, 0 /*flags*/);
```

**13.9.5 使用多播映射**

CUDA C++ 中需要使用 multimem PTX 指令配合内联 PTX 汇编以支持多播映射：

```cpp
__global__ void all_reduce_norm_barrier_kernel(
    float* l2_norm, float* partial_l2_norm_mc,
    unsigned int* arrival_counter_uc, unsigned int* arrival_counter_mc,
    const unsigned int expected_count
) {
    assert(1 == blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z);
    float l2_norm_sum = 0.0;
#if __CUDA_ARCH__ >= 900
    asm volatile ("multimem.red.release.sys.global.add.u32 [%0], %1;" :: 
                  "l"(arrival_counter_mc), "n"(1) : "memory");
    asm volatile ("fence.proxy.alias;" ::: "memory");
    cuda::atomic_ref<unsigned int,cuda::thread_scope_system> ac(arrival_counter_uc);
    while (expected_count > ac.load(cuda::memory_order_acquire));
    asm volatile ("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];" : 
                  "=f"(l2_norm_sum) : "l"(partial_l2_norm_mc) : "memory");
#else
    #error "ERROR: multimem instructions require compute capability 9.0 or larger."
#endif
    *l2_norm = std::sqrt(l2_norm_sum);
}
```

**第十三章概述：虚拟内存管理**

**核心思路**

虚拟内存管理（Virtual Memory Management, VMM）API 是 CUDA 提供的一套功能，旨在为 GPU 编程引入更灵活的内存管理机制。传统的 cudaMalloc 等方法直接返回可供使用的 GPU 内存指针，而 VMM 引入了物理内存与虚拟地址的解耦，使得用户可以更高效地管理和共享内存资源，特别是在多设备、多节点和跨进程通信场景中。

VMM 通过一系列 API 实现以下目标：

 1. **灵活的内存管理**：用户可以在需要时动态调整物理内存与虚拟地址之间的映射。

 2. **进程间通信（IPC）和共享内存**：支持通过平台特定句柄共享内存，优化资源利用。

 3. **跨节点 NVLINK 通信**：引入 Fabric Memory，实现 NVLINK Fabric 内的高效通信。

 4. **多播支持（Multicast Support）**：利用 NVSWITCH 和 NVLINK SHARP 加速多设备间的广播和归约操作。

mmap/

![](/assets/cudacpp-14-15/image.png)

 1. **物理内存与虚拟地址的解耦**

- **物理内存分配**：通过 cuMemCreate 分配物理内存，返回一个句柄（CUmemGenericAllocationHandle），而不是直接返回指针。

- **虚拟地址范围预留**：通过 cuMemAddressReserve 为物理内存分配预留虚拟地址范围（VA range）。

- **映射与取消映射**：通过 cuMemMap 和 cuMemUnmap 实现物理内存与虚拟地址的映射和解除，提供灵活性和资源优化。

 2. **访问控制**

- 用户需显式通过 cuMemSetAccess 设置虚拟地址范围的访问权限，确保不同设备对共享内存的正确访问。

- 提供对单个分配的粒度化控制，而非传统的 cudaEnablePeerAccess 的全局映射。

 3. **可压缩内存**

- 允许用户分配压缩内存（Compressible Memory），利用 CUDA 硬件的压缩能力优化带宽和缓存利用。

- 用户需查询设备支持情况，并验证分配是否成功启用了压缩属性。

 4. **Fabric Memory**

- 新增的分配句柄类型 CU_MEM_HANDLE_TYPE_FABRIC，支持跨节点的 NVLINK Fabric 通信，适用于多节点、多 GPU 系统。

- 通过 IMEX 守护进程，简化跨节点内存共享的实现。

 5. **多播支持（Multicast Support）**

- 借助 NVLINK SHARP 和 NVSWITCH，支持广播（broadcast）和归约（reduction）等操作。

- 多播团队（Multicast Team）由多个 NVLINK 连接的 GPU 组成，每个 GPU 拥有多播对象的本地副本。

- 映射和同步依赖 multimem PTX 指令，用户需确保一致性和同步。

 6. **虚拟别名（Virtual Aliasing）**

- 支持为同一物理内存创建多个虚拟地址映射，允许多个代理访问同一内存。

- 对于内核中的并发访问，需使用 fence.proxy.alias 保证内存访问的一致性和同步。

**应用场景**

 1. **多 GPU 编程**：通过虚拟内存管理，用户可以更高效地在多设备间共享资源，优化通信开销。

 2. **跨进程通信**：通过平台特定的共享句柄，简化了多进程协作的实现。

 3. **分布式计算**：Fabric Memory 支持多节点 NVLINK Fabric 的高效通信，是大规模分布式训练的基础。

 4. **性能优化**：通过压缩内存和访问控制机制，用户可以根据具体需求对带宽、延迟和内存分配进行微调。

虚拟内存管理 API 是 CUDA 的一项重要功能升级，为复杂 GPU 应用（如多设备、多节点、高性能通信等）提供了更加灵活和高效的内存管理能力。通过引入物理内存与虚拟地址的解耦、共享内存的新机制，以及 NVLINK Fabric 和多播支持，VMM 显著提升了 CUDA 在大规模异构计算场景中的适用性和性能。

## 第14章 stream 有序内存分配

**14.1 引言**

使用 cudaMalloc 和 cudaFree 管理内存分配会导致 GPU 在所有 CUDA 流上进行同步。流序内存分配器（Stream Ordered Memory Allocator）允许应用程序将内存分配和释放操作与 CUDA 流中启动的其他工作（例如内核启动和异步复制）进行有序化。这种方式通过流序语义来重用内存分配，从而优化内存使用。

分配器的特性包括：

- 支持内存缓存行为的控制。

- 允许在释放阈值设置合适时避免昂贵的操作系统调用。

- 提供安全、便捷的跨进程分配共享。

对于许多应用程序，流序内存分配器减少了对自定义内存管理抽象的需求。即便是需要自定义内存管理的应用，也可以更轻松地实现高性能管理。

先来看一个简单的示例代码，不难看出，第一个 cudaFree 调用必须等待 kernelA 完成，所以在释放内存之前需要先同步设备，效率可想而知。

```cpp
cudaMalloc(&ptrA, sizeA);
kernelA<<<..., stream>>>(ptrA);
cudaFree(ptrA); // Synchronizes the device before freeing memory
cudaMalloc(&ptrB, sizeB);
kernelB<<<..., stream>>>(ptrB);
cudaFree(ptrB);
```

为了提高运行效率，可以预先分配内存，内存大小设为 kernelA 和 kernelB 所需内存大小的较大值，修改代码如下。

```cpp
cudaMalloc(&ptr,   max(sizeA, sizeB));
kernelA<<<...,   stream>>>(ptr);
kernelB<<<...,   stream>>>(ptr);
cudaFree(ptr); 
```

但是这样会增加在实际应用中的复杂性，因为内存管理从业务逻辑中分离了出来。当涉及到其他库时，问题就会比较明细。比如考虑下面这段代码，由库函数内部启动 kernelA 的情况:

```cpp
libraryFuncA(stream);
cudaMalloc(&ptrB, sizeB);
kernelB<<<..., stream>>>(ptrB);
cudaFree(ptrB);
  
void libraryFuncA(cudaStream_t stream) {
    cudaMalloc(&ptrA, sizeA);
    kernelA<<<..., stream>>>(ptrA);
    cudaFree(ptrA);
 } 
```

这在实际应用中就很普遍，但要提高效率相对就困难得多，因为可能无法完全查看或控制库函数中正在执行的操作。为了避免这个问题，库必须在第一次调用该函数时分配内存，并且直到库去初始化时才释放它。这不仅增加了代码的复杂性，而且还会导致库占用内存的时间超过需要的时间，从而可能会阻止应用程序的另一部分使用该内存。

有些应用程序通过实现自己的自定义分配器，进一步提前分配内存。这为应用程序开发增加了大量复杂性。因此为了方便开发者，CUDA 提供了一种低工作量、高性能的替代方案。

CUDA 11.2 引入了[Stream-Ordered Memory Allocator](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-c-programming-guide/index.html%23stream-ordered-memory-allocator) 来解决这些类型的问题，并提供了 cudaMallocAsync 和 cudaFreeAsync 这些新的 API 函数，来将内存分配从同步整个设备的全局作用域操作转移到 Stream-Ordered 操作，从而使开发者能够将内存管理与 GPU 任务协同起来。这样就消除了同步未完成 GPU 任务的需要，并有助于限制分配到访问它的 GPU 任务的生命周期。代码示例如下：

```cpp
cudaMallocAsync(&ptrA, sizeA, stream);
kernelA<<<..., stream>>>(ptrA);
cudaFreeAsync(ptrA, stream); // No synchronization necessary
cudaMallocAsync(&ptrB, sizeB, stream); // Can reuse the memory freed previously
kernelB<<<..., stream>>>(ptrB);
cudaFreeAsync(ptrB, stream); 
现在可以在函数范围内管理内存，如下面在库函数中启动 kernelA 的示例所示。
libraryFuncA(stream);
cudaMallocAsync(&ptrB, sizeB, stream); // Can reuse the memory freed by the library call
kernelB<<<..., stream>>>(ptrB);
cudaFreeAsync(ptrB, stream);
  
void libraryFuncA(cudaStream_t stream) {
    cudaMallocAsync(&ptrA, sizeA, stream);
    kernelA<<<..., stream>>>(ptrA);
    cudaFreeAsync(ptrA, stream); // No synchronization necessary 
```

**CUDA Stream 类比线程并行**

 1. **Stream 相当于线程**：

- 每个 CUDA Stream 是一个独立的任务队列，类似于线程在并行环境中运行。

- 流中的操作按顺序执行，但不同流中的操作可以并行（类似于线程之间并发运行）。

 2. **GPU 的执行模型**：

- GPU 通过多个 **流多处理器（SM）** 并行处理流中的任务。

- 不同流之间的任务调度可以与线程的任务调度机制类比。

 3. **任务队列**：

- 每个 CUDA Stream 是一个任务队列，可以类比为线程的工作队列。

- 一个流中提交的任务（如 kernel 启动、内存拷贝）是按顺序执行的。

**14.2 支持查询**

用户可以通过 cudaDeviceGetAttribute() 查询设备是否支持流序内存分配器。在 CUDA 11.3 及以上版本，支持通过 cudaDevAttrMemoryPoolSupportedHandleTypes 查询 IPC 内存池支持情况。

**cudaMalloc 和 cudaMallocAsync 类比线程资源分配**

**1. cudaMalloc 类比全局锁的分配机制：**

- **阻塞性**：cudaMalloc 在使用时会阻塞整个 GPU 的执行，相当于一个全局锁，所有线程都必须等待资源分配完成后才能继续工作。

- **性能瓶颈**：如果多线程都需要分配内存，而只有一个全局分配器（如 cudaMalloc），就会因为阻塞导致效率低下。

- **高开销**：全局锁的使用和频繁的同步机制导致了较高的时间开销。

**2. cudaMallocAsync 类比线程局部资源分配：**

- **异步分配**：cudaMallocAsync 允许在特定 CUDA Stream 中进行异步内存分配，相当于每个线程有自己的局部资源分配器。

- **局部性优化**：每个 Stream 对应一个流局部内存池，这个内存池的操作不影响其他 Stream，相当于线程局部存储，避免了全局锁。

- **无阻塞**：与全局锁机制不同，流之间的操作可以并行进行，而不受内存分配操作的阻塞。

代码示例如下：

```cpp
int driverVersion = 0;
int deviceSupportsMemoryPools = 0;
int poolSupportedHandleTypes = 0;

cudaDriverGetVersion(&driverVersion);
if (driverVersion >= 11020) {
    cudaDeviceGetAttribute(&deviceSupportsMemoryPools, cudaDevAttrMemoryPoolsSupported, device);
}
if (deviceSupportsMemoryPools != 0) {
    // 设备支持流序内存分配器
}
if (driverVersion >= 11030) {
    cudaDeviceGetAttribute(&poolSupportedHandleTypes, cudaDevAttrMemoryPoolSupportedHandleTypes, device);
}
if (poolSupportedHandleTypes & cudaMemHandleTypePosixFileDescriptor) {
    // 可以创建基于 POSIX 文件描述符的 IPC 内存池
}
```

**14.3 核心 API：cudaMallocAsync 和 cudaFreeAsync**

 1. cudaMallocAsync

用于分配内存，并指定在某一流中进行操作。

 2. cudaFreeAsync

用于释放内存分配，并将释放操作插入到指定流中。

**基本用法**

以下示例展示了 cudaMallocAsync 和 cudaFreeAsync 的基本用法：

```cpp
void *ptr;
size_t size = 512;
cudaMallocAsync(&ptr, size, cudaStreamPerThread);
// 使用分配的内存
kernel<<<..., cudaStreamPerThread>>>(ptr, ...);
// 异步释放内存
cudaFreeAsync(ptr, cudaStreamPerThread);
```

**跨流使用**

如果在分配流之外的其他流中使用内存，用户必须保证访问在分配操作完成之后，否则行为未定义。可通过同步分配流或使用 CUDA 事件实现这一保证。

以下示例使用 CUDA 事件进行流间同步：

```cpp
cudaMallocAsync(&ptr, size, stream1);
cudaEventRecord(event1, stream1);

// stream2 等待分配完成
cudaStreamWaitEvent(stream2, event1);
kernel<<<..., stream2>>>(ptr, ...);
cudaEventRecord(event2, stream2);

// stream3 等待 stream2 完成后释放内存
cudaStreamWaitEvent(stream3, event2);
cudaFreeAsync(ptr, stream3);
```

**与 cudaMalloc 和 cudaFree 的互操作**

- 可以使用 cudaFreeAsync 释放由 cudaMalloc 分配的内存，但需确保在释放操作开始之前，所有访问已完成：

```cpp
cudaMalloc(&ptr, size);
kernel<<<..., stream>>>(ptr, ...);
cudaFreeAsync(ptr, stream);
```

- 可以使用 cudaFree 释放由 cudaMallocAsync 分配的内存。驱动程序假设所有访问已完成，因此不会进行同步，用户需显式同步：

```cpp
cudaMallocAsync(&ptr, size, stream);
kernel<<<..., stream>>>(ptr, ...);
// 确保同步避免过早释放
cudaStreamSynchronize(stream);
cudaFree(ptr);
```

**14.4 内存池与 cudaMemPool_t**

内存池（Memory Pool）封装了虚拟地址和物理内存资源，这些资源根据池的属性和属性进行分配和管理。内存池的核心在于它管理的内存类型和位置。

- 所有 cudaMallocAsync 调用都使用内存池的资源。

- 如果未指定内存池，cudaMallocAsync 使用流所在设备的当前内存池。

- 用户可以使用 cudaDeviceSetMempool 设置设备的当前内存池，并通过 cudaDeviceGetMempool 查询。

- 默认情况下（没有调用 cudaDeviceSetMempool），设备的当前内存池是其默认内存池。

- cudaMallocFromPoolAsync 和 cudaMallocAsync 的 C++ 重载支持用户指定内存池，而无需将其设置为当前池。

```cpp
cudaDeviceGetDefaultMempool(&defaultPool, device);
cudaDeviceSetMempool(device, customPool);
cudaMallocFromPoolAsync(&ptr, size, customPool, stream);
```

**14.5 默认/隐式内存池**

- 每个设备的默认内存池可通过 cudaDeviceGetDefaultMempool 获取。

- 默认池分配的内存是非迁移的，且始终可从该设备访问。

- 可通过 cudaMemPoolSetAccess 修改默认池的可访问性，并通过 cudaMemPoolGetAccess 查询。

- 默认池不支持进程间通信（IPC）。

**14.6 显式内存池**

- 使用 cudaMemPoolCreate 创建显式内存池，支持自定义分配属性，如 IPC 能力、最大池大小、特定 NUMA 节点的分配等。

```cpp
// 创建与设备 0 的隐式内存池类似的内存池
int device = 0;
cudaMemPoolProps poolProps = { };
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = device;
poolProps.location.type = cudaMemLocationTypeDevice;
cudaMemPoolCreate(&memPool, &poolProps);

// 创建支持 IPC 的内存池，驻留在特定 CPU NUMA 节点
int cpu_numa_id = 0;
poolProps.location.id = cpu_numa_id;
poolProps.location.type = cudaMemLocationTypeHostNuma;
poolProps.handleType = cudaMemHandleTypePosixFileDescriptor;
cudaMemPoolCreate(&ipcMemPool, &poolProps);
```

**14.7 物理页缓存行为**

默认情况下，分配器尝试最小化池拥有的物理内存。通过设置释放阈值（cudaMemPoolAttrReleaseThreshold），可以减少操作系统的分配/释放调用。

- **设置释放阈值**：池持有的内存超过阈值时，会尝试在下一次同步操作中将多余内存释放回操作系统。

```cpp
Cuuint64_t setVal = UINT64_MAX;
cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);
```

- **显式缩减池的内存占用**：使用 cudaMemPoolTrimTo 减少池的内存占用，但保留指定大小的内存。

```cpp
cudaStreamSynchronize(stream);
cudaMemPoolTrimTo(memPool, 0);
```

**14.8 资源使用统计**

CUDA 11.3 引入了以下内存池属性，用于查询池的内存使用情况：

- cudaMemPoolAttrReservedMemCurrent：当前池消耗的物理 GPU 内存总量。

- cudaMemPoolAttrReservedMemHigh：自上次重置以来的最大物理内存消耗值。

- cudaMemPoolAttrUsedMemCurrent：池中分配的内存总量。

- cudaMemPoolAttrUsedMemHigh：自上次重置以来的最大分配内存量。

可以使用 cudaMemPoolSetAttribute 重置水位线。

```cpp
// 查询使用统计信息
struct usageStatistics {
    cuuint64_t reserved;
    cuuint64_t reservedHigh;
    cuuint64_t used;
    cuuint64_t usedHigh;
};
void getUsageStatistics(cudaMemoryPool_t memPool, struct usageStatistics *statistics) {
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent, &statistics->reserved);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, &statistics->reservedHigh);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent, &statistics->used);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, &statistics->usedHigh);
}

// 重置水位线
void resetStatistics(cudaMemoryPool_t memPool) {
    cuuint64_t value = 0;
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, &value);
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, &value);
}
```

**14.9 内存重用策略**

在处理内存分配请求时，驱动程序会尝试重用通过 cudaFreeAsync() 释放的内存，而不是直接向操作系统申请更多内存。例如：

- 在同一流中释放的内存可以立即用于该流中的后续分配请求。

- 当流与 CPU 同步时，该流中释放的内存变得可以被其他流分配重用。

通过以下内存池属性可以控制内存重用策略：

- cudaMemPoolReuseFollowEventDependencies

- cudaMemPoolReuseAllowOpportunistic

- cudaMemPoolReuseAllowInternalDependencies

不同的 CUDA 驱动程序版本可能会修改或扩展这些策略。

**14.9.1 cudaMemPoolReuseFollowEventDependencies**

该策略允许分配器在分配内存之前，检查 CUDA 事件建立的依赖关系，并尝试重用在其他流中释放的内存：

```cpp
cudaMallocAsync(&ptr, size, originalStream); 
kernel<<<..., originalStream>>>(ptr, ...); 
cudaFreeAsync(ptr, originalStream); 
cudaEventRecord(event, originalStream);
// 等待事件，允许分配器重用另一流中的内存
cudaStreamWaitEvent(otherStream, event); 
cudaMallocAsync(&ptr2, size, otherStream);
```

**14.9.2 cudaMemPoolReuseAllowOpportunistic**

当启用该策略时，分配器会检查释放的内存是否满足流的语义顺序（例如，释放操作的流已完成对应的执行）。即使该策略禁用，分配器仍会重用与 CPU 同步后可用的内存：

```cpp
cudaMallocAsync(&ptr, size, originalStream); 
kernel<<<..., originalStream>>>(ptr, ...); 
cudaFreeAsync(ptr, originalStream);
// 等待一些时间，分配器可以根据 originalStream 的进度重用内存
wait(10); 
cudaMallocAsync(&ptr2, size, otherStream);
```

**14.9.3 cudaMemPoolReuseAllowInternalDependencies**

如果驱动无法从操作系统分配更多物理内存，它会检查依赖于其他流进度的内存。如果找到这样的内存，驱动会在分配流中插入依赖，并重用内存：

```cpp
cudaMallocAsync(&ptr, size, originalStream); 
kernel<<<..., originalStream>>>(ptr, ...); 
cudaFreeAsync(ptr, originalStream);
// 在启用该策略时，驱动可能插入内部依赖以确保流间工作顺序正确
cudaMallocAsync(&ptr2, size, otherStream);
```

**14.9.4 禁用重用策略**

尽管内存重用策略可以提高效率，但某些情况下用户可能希望禁用这些策略。例如：

- 允许机会性重用（cudaMemPoolReuseAllowOpportunistic）可能导致运行结果因 CPU 和 GPU 的执行交错而变化。

- 内部依赖插入（cudaMemPoolReuseAllowInternalDependencies）可能导致非显式的同步。

**14.10 多 GPU 支持的设备可访问性**

内存池分配的访问权限由 cudaMemPoolSetAccess 修改，而不依赖于 cudaDeviceEnablePeerAccess 或 cuCtxEnablePeerAccess。

默认情况下，分配仅可从驻留设备访问，该访问权限无法撤销。

- 若要启用其他设备访问，需检查目标设备是否具备对驻留设备的对等访问能力（使用 cudaDeviceCanAccessPeer）。若不具备对等访问能力，则访问设置会失败。

- 如果池中尚无分配，cudaMemPoolSetAccess 可能成功，但下一次分配会失败。

以下代码展示了 cudaMemPoolSetAccess 的用法：

```cpp
cudaError_t setAccessOnDevice(cudaMemPool_t memPool, int residentDevice, int accessingDevice) {
    cudaMemAccessDesc accessDesc = {};
    accessDesc.location.type = cudaMemLocationTypeDevice;
    accessDesc.location.id = accessingDevice;
    accessDesc.flags = cudaMemAccessFlagsProtReadWrite;

    int canAccess = 0;
    cudaError_t error = cudaDeviceCanAccessPeer(&canAccess, accessingDevice, residentDevice);
    if (error != cudaSuccess) {
        return error;
    } else if (canAccess == 0) {
        return cudaErrorPeerAccessUnsupported;
    }
    return cudaMemPoolSetAccess(memPool, &accessDesc, 1);
}
```

注意事项：

- cudaMemPoolSetAccess 会影响内存池中所有分配，而不仅仅是未来的分配。

- 不建议频繁更改内存池的可访问性设置。一旦设置为可访问，建议在池的生命周期内保持不变。

**14.11 IPC 内存池**

IPC（进程间通信）内存池支持在进程之间高效、安全地共享 GPU 内存，提供与 CUDA 虚拟内存管理 API 相同的安全性。

跨进程共享内存池分为两个阶段：

 1. **共享内存池访问权限**：导出池的 OS 原生句柄并传递到目标进程，创建导入的内存池。

 2. **共享内存池的具体分配**：协调分配的虚拟地址及其映射的有效性。

**14.11.1 创建和共享 IPC 内存池**

共享内存池需要以下步骤：

- 使用 cudaMemPoolExportToShareableHandle() 获取内存池的 OS 原生句柄。

- 使用操作系统的 IPC 机制将句柄传递到目标进程。

- 在目标进程中使用 cudaMemPoolImportFromShareableHandle() 创建导入的内存池。

以下是代码示例：

**导出内存池：**

```cpp
cudaMemPoolProps poolProps = {};
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = 0;
poolProps.location.type = cudaMemLocationTypeDevice;
poolProps.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

cudaMemPoolCreate(&memPool, &poolProps);

int fdHandle = 0;
cudaMemPoolExportToShareableHandle(&fdHandle, memPool, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
// 使用操作系统的 IPC 机制传递 fdHandle
```

**导入内存池：**

```cpp
int fdHandle;
// 通过 IPC 机制接收 fdHandle
cudaMemPoolImportFromShareableHandle(&importedMemPool, (void*)fdHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
```

**14.11.2 在导入进程中设置访问权限**

- 导入的内存池初始只对其驻留设备可访问。

- 导入进程需要使用 cudaMemPoolSetAccess 设置其他设备的访问权限。

如果导入的内存池属于一个在导入进程中不可见的设备，必须显式启用目标 GPU 的访问权限。

**14.11.3 创建并共享导出的内存池分配**

从导出的内存池中使用 cudaMallocAsync() 创建的分配可以与导入进程共享。

**导出分配：**

```cpp
cudaMemPoolPtrExportData exportData;
cudaEvent_t readyIpcEvent;
cudaIpcEventHandle_t readyIpcEventHandle;

cudaEventCreate(&readyIpcEvent, cudaEventDisableTiming | cudaEventInterprocess);
cudaMallocAsync(&ptr, size, exportMemPool, stream);
cudaEventRecord(readyIpcEvent, stream);
cudaMemPoolExportPointer(&exportData, ptr);
cudaIpcGetEventHandle(&readyIpcEventHandle, readyIpcEvent);

// 将 exportData 和 readyIpcEventHandle 通过共享内存等机制传递到目标进程
```

**导入分配：**

```cpp
cudaMemPoolPtrExportData *importData = &shmem->ptrData;
cudaEvent_t readyIpcEvent;
cudaIpcOpenEventHandle(&readyIpcEvent, &shmem->readyIpcEventHandle);

cudaMemPoolImportPointer(&ptr, importedMemPool, importData);
cudaStreamWaitEvent(stream, readyIpcEvent);
kernel<<<..., stream>>>(ptr, ...);
```

**释放分配：**

释放分配时，必须先在导入进程中释放，再在导出进程中释放。

```cpp
// 导入进程
kernel<<<..., stream>>>(ptr, ...);
cudaFreeAsync(ptr, stream);
cudaIpcEventRecord(finishedIpcEvent, stream);

// 导出进程
cudaStreamWaitEvent(stream, finishedIpcEvent);
cudaFreeAsync(ptrInExportingProcess, stream);
```

**14.11.4 IPC 导出池的限制**

- IPC 内存池目前不支持将物理内存释放回操作系统，因此 cudaMemPoolTrimTo 是无操作的，cudaMemPoolAttrReleaseThreshold 被忽略。

- 此行为由驱动控制，可能在未来的驱动程序更新中更改。

**14.11.5 IPC 导入池的限制**

- 导入的内存池不能用于分配，也不能被设置为当前内存池。

- 无法使用 cudaMallocFromPoolAsync 从导入池中分配内存，因此分配重用策略属性对这些池无意义。

- 资源使用统计属性仅反映导入到进程的分配及其相关的物理内存。

**14.12 同步 API 的行为**

流序内存分配器与 CUDA 驱动中的同步 API 集成，以实现优化：

- 当用户请求同步时，驱动会等待所有异步工作完成。

- 在返回之前，驱动会检查哪些释放操作已由同步保证完成，并使这些内存分配可供重用，无论其所属流或分配策略。

- 同时，驱动会检查 cudaMemPoolAttrReleaseThreshold 属性，释放池中多余的物理内存（若可能）。

**14.13 补充说明**

**14.13.1 cudaMemcpyAsync 的当前上下文/设备敏感性**

对于使用 cudaMallocAsync 分配的内存进行异步复制，调用 cudaMemcpyAsync 时需要确保使用分配时指定流的上下文作为调用线程的当前上下文。但对于 cudaMemcpyPeerAsync，因为该 API 使用设备的主上下文，因此无需设置当前上下文。

**14.13.2 cuPointerGetAttribute 查询**

在调用 cudaFreeAsync 后，若对释放的内存调用 cuPointerGetAttribute 查询，会导致未定义行为。即使该分配仍然可从某个流中访问，行为依然未定义。

**14.13.3 cuGraphAddMemsetNode**

cuGraphAddMemsetNode 不支持使用流序分配器分配的内存。然而，可以对这些内存进行的 memset 操作被流捕获（stream capture）支持。

**14.13.4 指针属性**

- 对流序内存分配的 cuPointerGetAttributes 查询是有效的。

- 由于流序分配的内存不与特定上下文关联，查询 CU_POINTER_ATTRIBUTE_CONTEXT 时会返回 NULL。

- 可通过查询 CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL 属性确定分配的设备位置，这对选择上下文执行 cudaMemcpyPeerAsync 的 p2h2p 复制操作很有用。

- CUDA 11.3 中新增的 CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE 属性，可用于调试或确认分配所属的内存池，在进行 IPC 操作前尤为重要。

**第十四章概述：流序内存分配器**

**核心思路**

流序内存分配器（Stream Ordered Memory Allocator）是 CUDA 提供的一种高级内存管理机制，通过与 CUDA 流的有序语义集成，实现了内存分配与释放的异步化和精细化控制。与传统的 cudaMalloc 和 cudaFree 不同，流序分配器不仅减少了全局同步的需求，还支持灵活的内存重用策略、跨进程共享以及与同步 API 的优化整合。

**关键技术点**

 1. **流序分配与释放**

- 使用 cudaMallocAsync 和 cudaFreeAsync 在指定流中分配和释放内存。

- 分配与释放操作会遵循流的执行顺序，无需显式同步。

- 支持在不同流间共享分配，通过事件或流同步保证内存访问的正确性。

 2. **内存池**

- 内存池（cudaMemPool_t）是流序分配的核心，封装了虚拟地址和物理内存资源。

- 支持默认池（隐式）和显式池，显式池可定制属性（如 IPC 能力、NUMA 节点支持）。

- 通过 cudaMemPoolSetAccess 和 cudaMemPoolGetAccess 控制多设备的访问权限。

 3. **内存重用策略**

- 通过属性控制分配器的内存重用行为，包括：

- **事件依赖重用**：通过事件同步不同流中的内存分配和释放。

- **机会性重用**：基于流的执行进度重用已释放的内存。

- **内部依赖重用**：当分配失败时，自动插入依赖以重用其他流中的内存。

 4. **跨进程通信（IPC）**

- 支持通过 OS 句柄（如 POSIX 文件描述符）导出和导入内存池，实现安全高效的进程间共享。

- 内存池级别的安全性简化了跨进程共享的实现，不需要额外的安全验证。

- 导入池限制：不能直接从导入池中分配内存，只能通过导入已存在的分配来使用。

 5. **同步与释放优化**

- 与同步 API 集成，驱动在同步点释放已完成释放操作的内存，使其可重用。

- 支持通过 cudaMemPoolAttrReleaseThreshold 配置释放阈值，优化物理内存管理。

 6. **扩展与限制**

- 对于 IPC 和流序分配内存的限制：

- cuPointerGetAttribute 在释放内存后查询会导致未定义行为。

- cuGraphAddMemsetNode 不支持流序内存分配，但 memset 可通过流捕获实现。

- cudaMemcpyAsync 对流上下文敏感，需保证使用正确的上下文。

**适用场景**

- **异步流管理**：在多流应用中提高内存分配效率。

- **内存重用与优化**：通过自定义重用策略减少内存浪费。

- **跨进程通信**：在多 GPU 进程间共享资源，简化数据交换。

- **性能调优**：与 CUDA 同步 API 深度集成，支持高效的同步与释放操作。

通过流序内存分配器，CUDA 提供了更灵活的内存管理能力，使其在异构计算、深度学习训练与推理、多进程任务中更加高效和便捷。
