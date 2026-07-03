---
title: CUDA C++ 笔记（二）第6章——Programming Interface（一）
date: 2024-06-05 20:00:00
tags: [CUDA, NVCC, Runtime]
categories: [CUDA C++ Programming Guide]
description: 本篇涵盖第 6 章 Programming Interface 的前半部分（6.1-6.2.7），包括 NVCC 编译流程（离线编译、即时编译、兼容性）、CUDA Runtime 初始化、设备内存分配、L2 缓存访问管理、共享内存、分布式共享内存、页锁定主机内存以及内存同步域等内容。
---

## NVCC 编译

Kernel 可以使用 CUDA 指令集架构（PTX）来编写。然而，通常使用高级编程语言（如 C++）来编写 Kernel 会更为高效。在这两种情况下，Kernel 都必须通过 nvcc 编译成二进制代码才能在设备上执行。nvcc 是一个编译器驱动程序，它简化了编译 C++ 或 PTX 代码的过程：它提供了简单且熟悉的命令行选项，并通过调用实现不同编译阶段的工具集合来执行这些选项。本节概述了 nvcc 的工作流程和命令选项。有关完整描述，请参阅 nvcc 用户手册。可以先简单地把 NVCC 看做是 CUDA 里面类似于 gcc 一样的编译器。

### 编译流程

NVCC 的编译模式可以分为两种：**offline compilation**（离线编译）和 **just-in-time compilation**（即时编译）。

### 离线编译

CUDA 代码的源文件可以包括在 host 端执行以及在 device 端执行的代码，NVCC 会将这两部分分开。其中 device 部分的代码会被编译成 PTX 汇编的形式，以及二进制形式（**cubin object**）。另外 host 部分则会被修改，将 `<<<...>>>` 这样的调用语法替换为 CUDA runtime 的函数调用（function call）语句，从而加载并运行上面已经被编译的 PTX 代码或是二进制码（cubin object）。

被修改过的 host 部分会被输出为 C++ 代码的形式，以供其他工具继续编译；或者是 object code（二进制码）的形式，以供 NVCC 通知 host 端编译器做最后一步的编译工作。（所以可以说这个输出还是一个半成品，后续还需要其它的编译工作）。

之后，应用就可以要么把上面的编译输出跟其他的编译过的 host code 联系在一起（就像 C++ 编译最后的 link 那一步），这也是最常见的情况；要么直接忽视掉上面被修改过的 host code，然后直接使用 CUDA driver API 去加载运行从 device code 编译出来的 PTX 码或者 cubin object。

CUDA 的编程模型中还是认为 host 作为主体是去控制 devices 的。所以 host code 需要负责对 device code 的调度，上面几段如果不好理解的话，可以简单认为 host code 可以被编译成两种形式：C++ 码（后面还需要被 host 端编译，编译好了可以用来运行调度 device code），或者是 PTX 码/cubin object，需要使用 CUDA driver API 去加载运行，也能调度 device code。

### 即时编译

PTX 码经过进一步的编译，使用 device driver 转化成二进制码，之后可以被应用加载执行。这个过程称为即时编译。即时编译模式会增加应用的加载时间，但是可以因此适应不同的编译器和 device driver，从更新版本的编译器或 device driver 中受益。这也是可以让应用在其编译之后才推出的设备上运行的唯一方法，提高了兼容能力。

使用即时编译模式的时候，device driver 将 PTX 码编译为二进制码，并且会自动缓存一份所生成的二进制码的备份，以免在后续的编译中重复已有的工作。但是如果 device driver 升级了，这些备份会被自动弃用，从而保证编译器的提升不会因为这些历史遗留而不能完全发挥。

在即时编译里，还可以使用 **CUDA Environment Variables**（环境变量）来控制编译的一些具体设定。

上述过程如图：

![NVCC 编译流程](/assets/cudacpp-6-1/image.png)

### 二进制兼容性

二进制代码是特定于架构的。使用编译器选项 `-code` 可以生成针对特定架构的 cubin 对象，用于指定目标架构。例如，使用 `-code=sm_80` 进行编译会生成适用于计算能力 8.0 设备的二进制代码。二进制兼容性在设备计算能力版本前到后或者小版本跨度有保证，但设备计算能力版本回退或大版本跨度则不保证。例如：为计算能力 X.y 生成的 cubin 对象只能在计算能力为 X.z（其中 z >= y）的设备上执行。

### PTX 兼容性

某些 PTX 指令仅在计算能力较高的设备上受支持。例如，Warp Shuffle 函数仅在计算能力 5.0 及以上的设备上受支持。`-arch` 编译器选项用于指定在将 C++ 代码编译为 PTX 代码时所假设的计算能力。例如，包含 warp shuffle 的代码必须使用 `-arch=compute_50` 或更高版本进行编译。为特定计算能力生成的 PTX 代码总是可以编译为相同或更高计算能力的二进制代码。请注意，从较早的 PTX 版本编译而来的二进制代码可能无法利用某些硬件功能。例如，针对计算能力 7.0（Volta）的设备，如果其是从为计算能力 6.0（Pascal）生成的 PTX 编译而来的，那么它将无法使用 Tensor Core 指令，因为这些指令在 Pascal 上并不可用。因此，与使用最新版本的 PTX 生成的二进制代码相比，最终生成的二进制代码性能可能会较差。针对目标架构的条件功能编译的 PTX 代码仅在完全相同的物理架构上运行，而在其他任何地方都无法运行。针对架构条件的 PTX 代码不具有向前或向后兼容性。例如，使用 `sm_90a` 或 `compute_90a` 编译的代码仅在计算能力为 9.0 的设备上运行，并且不具有向后或向前兼容性。

### 应用兼容性

为了能够在未来计算能力更高（目前尚无法为其生成二进制代码）的架构上执行代码，应用程序必须加载将针对这些设备进行即时编译的 PTX 代码（请参阅"即时编译"）。CUDA C++ 应用程序中嵌入哪种 PTX 代码和二进制代码，由 `-arch` 和 `-code` 编译器选项或 `-gencode` 编译器选项控制。

例如：

```bash
nvcc x.cu \
    -gencode arch=compute_50,code=sm_50 \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_70,code=\"compute_70,sm_70\"
```

嵌入了与计算能力 5.0 和 6.0 兼容的二进制代码（第一个和第二个 `-gencode` 选项），以及与计算能力 7.0 兼容的 PTX 代码和二进制代码（第三个 `-gencode` 选项）。

`x.cu` 可以包含一个使用 warp reduction 操作的优化代码路径，例如，这种操作仅在计算能力 8.0 及以上的设备上受支持。`__CUDA_ARCH__` 宏可用于根据计算能力区分不同的代码路径。它仅在设备代码中定义。例如，当使用 `-arch=compute_80` 进行编译时，`__CUDA_ARCH__` 等于 800。如果 `x.cu` 是针对架构条件功能（例如使用 `sm_90a` 或 `compute_90a`）进行编译的，那么该代码只能在计算能力为 9.0 的设备上运行。

### C++ 兼容性

编译器的前端会根据 C++ 语法规则来处理 CUDA 源文件。对于主机代码，完全支持完整的 C++。然而对于设备代码，仅完全支持 C++ 的一个子集（第 18 章会更详细介绍）。

## CUDA Runtime

CUDA runtime 库在 cudart 库中实现，该库可以与应用程序进行链接，链接方式可以是静态链接（通过 `cudart.lib` 或 `libcudart.a`），也可以是动态链接（通过 `cudart.dll` 或 `libcudart.so`）。需要动态链接 `cudart.dll` 和/或 `libcudart.so` 的应用程序通常会将它们作为应用程序安装包的一部分。其所有接口的名称都以 `cuda` 为前缀。CUDA 编程模型假设系统由主机和设备组成，它们各自拥有独立的内存。设备内存部分概述了用于管理设备内存的 runtime 函数。

主要元素：

1. "共享内存"：在线程层次结构中引入的共享内存的使用方法，以最大化性能。
2. "页锁定主机内存"：页锁定主机内存，这是在主机与设备内存之间传输数据时实现 Kernel 执行与数据传输重叠所必需的。
3. "异步并发执行"：用于在系统各级别启用异步并发执行的概念和 API。
4. "多设备系统"：编程模型如何扩展到连接至同一主机的多个设备组成的系统。
5. "错误检查"：如何正确检查 runtime 生成的错误。"调用栈"部分提到了用于管理 CUDA C++ 调用栈的 runtime 函数。
6. "纹理和表面内存"：纹理和表面内存空间，它们提供了访问设备内存的另一种方式，并且还暴露了 GPU 纹理处理硬件的一个子集。
7. "图形互操作性"：runtime 提供的各种函数，用于与两大图形 API（OpenGL 和 Direct3D）进行互操作。

### 初始化

自 CUDA 12.0 起，`cudaInitDevice()` 和 `cudaSetDevice()` 调用会初始化 runtime 以及与指定设备相关联的主上下文。若未进行这些调用，runtime 将默认使用设备 0，并根据需要自行初始化以处理其他 runtime API 请求。

runtime 会为系统中的每个设备创建一个 CUDA 上下文（有关 CUDA 上下文的更多详细信息，请参阅"上下文"部分）。此上下文是该设备的主上下文，并在首次需要此设备上活动上下文的 runtime 函数时被初始化。它在应用程序的所有主机线程之间共享。作为此上下文创建的一部分，设备代码会在必要时进行即时编译并加载到设备内存中。这一切都是透明进行的。如果需要（例如，为了实现驱动程序 API 互操作性），则可以从驱动程序 API 访问设备的主上下文。

当某个主机线程调用 `cudaDeviceReset()` 时，这将销毁该主机线程当前操作设备的主上下文（即"设备选择"中定义的当前设备）。接下来，任何将此设备设为当前设备的主机线程所发出的 runtime 函数调用都将为该设备创建一个新的主上下文（**慎用，会销毁当前进程中当前设备上的所有分配并重置所有状态**）。

**注意**：从 CUDA 12.0 开始，`cudaSetDevice()` 现在将在更改主机线程的当前设备后显式地初始化 runtime。在 CUDA 的早期版本中，对新设备的 runtime 初始化会延迟到 `cudaSetDevice()` 之后的第一个 runtime 调用时才进行。因此，现在检查 `cudaSetDevice()` 的返回值以确认是否存在初始化错误变得非常重要。

### 设备内存

CUDA 编程模型假设系统由主机和设备组成，它们各自拥有独立的内存。Kernel 在设备内存中运行，因此 runtime 提供了用于分配、释放和复制设备内存的函数，以及用于在主机内存和设备内存之间传输数据的函数。设备内存可以分配为线性内存或 CUDA 数组。CUDA 数组是为纹理获取优化的不透明内存布局；线性内存分配在单个统一的地址空间中，这意味着分别分配的实体可以通过指针相互引用，例如在二叉树或链表中。

线性内存通常使用 `cudaMalloc()` 进行分配，使用 `cudaFree()` 进行释放，而主机内存和设备内存之间的数据传输则通常通过 `cudaMemcpy()` 来完成。在 Kernel 的向量加法代码示例中，需要将向量从主机内存复制到设备内存：

```cpp
// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Host code
int main()
{
    int N = ...;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input vectors
    ...

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    ...
}
```

线性内存还可以通过 `cudaMallocPitch()` 和 `cudaMalloc3D()` 进行分配：分别用于分配二维或三维数组，因为它们能确保分配的内存适当地填充以满足《设备内存访问》中描述的对齐要求，从而在访问行地址或在二维数组和设备内存的其他区域之间执行复制（使用 `cudaMemcpy2D()` 和 `cudaMemcpy3D()`）时确保最佳性能。以下代码示例分配了一个宽度为 width、高度为 height 的浮点值二维数组，并展示了如何在设备代码中遍历数组元素：

```cpp
// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, &pitch,
                width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// Device code
__global__ void MyKernel(float* devPtr,
                         size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r) {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) {
            float element = row[c];
        }
    }
}
```

以下代码示例分配了一个宽度为 width、高度为 height、深度为 depth 的浮点值三维数组，并展示了如何在设备代码中遍历数组元素：

```cpp
// Host code
int width = 64, height = 64, depth = 64;
cudaExtent extent = make_cudaExtent(width * sizeof(float),
                                    height, depth);
cudaPitchedPtr devPitchedPtr;
cudaMalloc3D(&devPitchedPtr, extent);
MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);

// Device code
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr,
                         int width, int height, int depth)
{
    char* devPtr = devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    for (int z = 0; z < depth; ++z) {
        char* slice = devPtr + z * slicePitch;
        for (int y = 0; y < height; ++y) {
            float* row = (float*)(slice + y * pitch);
            for (int x = 0; x < width; ++x) {
                float element = row[x];
            }
        }
    }
}
```

注意：为了避免分配过多内存从而影响整个系统的性能，请根据问题规模向用户请求分配参数。如果分配失败，可以回退到使用其他较慢的内存类型（如 `cudaMallocHost()`、`cudaHostRegister()` 等），或者返回一个错误，告知用户所需内存量以及为何被拒绝。

### 设备内存 L2 访问管理

当 CUDA 内核反复访问全局内存中的数据区域时，这种数据访问可以被认为是持久的。另一方面，如果数据只被访问一次，那么这种数据访问可以被认为是流式的。从 CUDA 11.0 开始，计算能力 8.0 及以上的设备具有影响 L2 缓存中数据持久性的能力，可以为全局内存访问提供更高的带宽和更低的延迟。

#### 为持久访问保留 L2 缓存

可以将 L2 缓存的一部分预留出来，专门用于对全局内存的持久数据访问。持久访问会优先使用这部分预留的 L2 缓存，而普通或流式访问全局内存时，则只能在这部分缓存未被持久访问使用时加以利用。持久访问的 L2 缓存预留大小可以在一定范围内进行调整：

```cpp
cudaGetDeviceProperties(&prop, device_id);
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* set-aside 3/4 of L2 cache for persisting accesses or the max allowed */
```

当 GPU 配置为多实例 GPU（MIG）模式时，L2 缓存预留功能将被禁用，此时无法通过 `cudaDeviceSetLimit` 更改 L2 缓存的预留大小。相反，预留大小只能在启动 MPS 服务器时通过环境变量 `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT` 来指定。

#### 持久访问的 L2 策略

一个**访问策略窗口**指定了全局内存中的一个连续区域以及 L2 缓存中用于该区域内访问的持久性属性。下面的代码示例展示了如何使用 CUDA 流来设置一个 L2 持久访问窗口。

```cpp
cudaStreamAttrValue stream_attribute;                  // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persistence access.
                                                                               // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

// Set the attributes to a CUDA stream of type cudaStream_t
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
```

当内核随后在 CUDA 流中执行时，在全局内存范围 `[ptr..ptr+num_bytes)` 内的内存访问相较于其他全局内存位置的访问，更有可能在 L2 缓存中持久存在。其他方式：也可以为 CUDA graph 内核节点设置 L2 持久性。

`hitRatio` 参数可用于指定获得 `hitProp` 属性的访问占比。在上述两个示例中，全局内存区域 `[ptr..ptr+num_bytes)` 内有 60% 的内存访问具有持久性属性，而 40% 的内存访问具有流式属性。哪些具体的内存访问被归类为持久性（即 `hitProp`）是随机的，其概率大约为 `hitRatio`；概率分布取决于硬件架构和内存范围。例如，如果 L2 预留缓存大小为 16KB，而 `accessPolicyWindow` 中的 `num_bytes` 为 32KB，则：

1. 当 `hitRatio` 为 0.5 时，硬件会随机选择 32KB 窗口中的 16KB 作为持久性数据，并将其缓存在预留的 L2 缓存区域中。
2. 当 `hitRatio` 为 1.0 时，由于预留区域小于窗口大小，为了保持最近使用的 16KB 数据在 L2 缓存的预留部分中，缓存行将会被逐出（即替换掉）。

因此，`hitRatio` 可用于避免缓存行的频繁替换，并总体上减少数据进出 L2 缓存的量。权衡：希望**达到较高的 L2 cache hit rate，同时避免 cache 数据频繁替换**。

#### L2 访问属性（L2 Access Properties）

为不同的全局内存数据访问定义了三种访问属性：

1. `cudaAccessPropertyStreaming`：带有流属性的内存访问不太可能持久保留在 L2 缓存中，因为这些访问会被优先逐出。
2. `cudaAccessPropertyPersisting`：带有持久属性的内存访问更有可能持久保留在 L2 缓存中，因为这些访问会被优先保留在 L2 缓存的预留部分。
3. `cudaAccessPropertyNormal`：强制将之前应用的持久访问属性重置为正常状态。来自之前 CUDA kernel 的带有持久属性的内存访问可能会在其预期使用之后长时间保留在 L2 缓存中。这种使用后的持久性会减少后续不使用持久属性的 kernel 可用的 L2 缓存量。使用 `cudaAccessPropertyNormal` 属性重置访问属性窗口会移除先前访问的持久（优先保留）状态，就像之前的访问没有访问属性一样。

#### L2 持久化例子

以下示例展示了如何为持久访问预留 L2 缓存，如何通过 CUDA 流在 CUDA kernel 中使用预留的 L2 缓存，以及如何重置 L2 缓存。

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);                                                                  // Create CUDA stream

cudaDeviceProp prop;                                                                        // CUDA device properties variable
cudaGetDeviceProperties( &prop, device_id);                                                 // Query GPU properties
size_t size = min( int(prop.l2CacheSize * 0.75) , prop.persistingL2CacheMaxSize );
cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size);                                  // set-aside 3/4 of L2 cache for persisting accesses or the max allowed

size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes);                        // Select minimum of user defined num_bytes and max window size.

cudaStreamAttrValue stream_attribute;                                                                                       // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data1);               // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = window_size;                                // Number of bytes for persistence access
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                                        // Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;               // Persistence Property
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;                // Type of access property on cache miss

cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Set the attributes to a CUDA Stream

for(int i = 0; i < 10; i++) {
    cuda_kernelA<<<grid_size,block_size,0,stream>>>(data1);                                 // This data1 is used by a kernel multiple times
}                                                                                           // [data1 + num_bytes) benefits from L2 persistence
cuda_kernelB<<<grid_size,block_size,0,stream>>>(data1);                                     // A different kernel in the same stream can also benefit
                                                                                            // from the persistence of data1

stream_attribute.accessPolicyWindow.num_bytes = 0;                                          // Setting the window size to 0 disable it
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Overwrite the access policy attribute to a CUDA Stream
cudaCtxResetPersistingL2Cache();                                                            // Remove any persistent lines in L2

cuda_kernelC<<<grid_size,block_size,0,stream>>>(data2);                                     // data2 can now benefit from full L2 in normal mode
```

#### 管理 L2 预留缓存的利用率

多个在不同 CUDA 流中并发执行的 CUDA kernel 可能会被分配给各自流的不同访问策略窗口。然而 L2 预留缓存部分是所有这些并发 CUDA kernel 共享的。因此，这部分预留缓存的净利用率是所有并发 kernels 各自使用量的总和。当持久访问的数量超过预留 L2 缓存的容量时，将内存访问指定为持久访问所带来的好处就会减少。管理预留 L2 缓存利用率，应用程序需要考虑以下几点：

1. 预留 L2 缓存的大小。
2. 可能并发执行的 CUDA kernel。
3. 所有可能并发执行的 CUDA kernel 的访问策略窗口。
4. 何时以及如何需要重置 L2 缓存，以便正常访问或流式访问能够以相同的优先级利用之前预留的 L2 缓存。

#### 查询 L2 缓存属性

与 L2 缓存相关的属性是 `cudaDeviceProp` 结构体的一部分，可以通过 CUDA 运行时 API `cudaGetDeviceProperties` 进行查询。CUDA 设备属性包括：

1. `l2CacheSize`：GPU 上可用的 L2 缓存大小。
2. `persistingL2CacheMaxSize`：可以为持久内存访问预留的 L2 缓存的最大大小。
3. `accessPolicyMaxWindowSize`：访问策略窗口的最大大小。

### 共享内存（Shared Memory）

共享内存是使用 `__shared__` 内存空间说明符来分配的。共享内存的预期速度要比全局内存快得多。它可以作为暂存内存（或软件管理的缓存）来使用，以最大限度地减少 CUDA 块对全局内存的访问，下面的矩阵乘法示例就说明了这一点。以下代码示例是一个简单的矩阵乘法实现，**没有利用共享内存**。每个线程读取矩阵 A 的一行和矩阵 B 的一列，并计算矩阵 C 的对应元素，如图所示。因此，矩阵 A 会从全局内存中被读取 B.width 次，而矩阵 B 会被读取 A.height 次。

```cpp
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}
```

以下代码示例是**利用共享内存**实现的矩阵乘法。在此实现中，每个线程块负责计算矩阵 C 的一个方形子矩阵 Csub，而块内的每个线程则负责计算 Csub 中的一个元素。Csub 等于两个矩形矩阵的乘积：一个是维度为 `(A.width, block_size)` 的 A 的子矩阵，其行索引与 Csub 相同；另一个是维度为 `(block_size, A.width)` 的 B 的子矩阵，其列索引与 Csub 相同。为了适应设备的资源，这两个矩形矩阵被划分为尽可能多的维度为 `block_size` 的方形矩阵，并且 Csub 被计算为这些方形矩阵乘积的和。每个乘积的计算都是首先由每个线程从全局内存将一个元素加载到共享内存中，从而加载两个对应的方形矩阵，然后由每个线程计算乘积中的一个元素。每个线程将每个乘积的结果累加到寄存器中，一旦完成，就将结果写入全局内存。通过这种方式的计算分块，我们利用了快速的共享内存，并节省了大量的全局内存带宽，因为 A 只从全局内存中读取了 `(B.width / block_size)` 次，而 B 只读取了 `(A.height / block_size)` 次。

这里主要是 kernel 的处理逻辑差别，所以只放 kernel 的代码：

```cpp
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}
// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
// Get a matrix element
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}
```

### 分布式共享内存（Distributed Shared Memory）

在计算能力 9.0 中引入的线程块集群（Thread block clusters）使得线程块集群中的线程能够访问集群中所有参与线程块的共享内存。这种分区共享内存被称为分布式共享内存（Distributed Shared Memory），相应的地址空间则被称为分布式共享内存地址空间。属于同一个线程块集群的线程可以在分布式地址空间中读取、写入或执行原子操作，无论该地址是属于本地线程块还是远程线程块。无论 kernel 是否使用分布式共享内存，共享内存的大小规格（静态或动态）仍然是针对每个线程块的。分布式共享内存的大小是每个集群中的线程块数量乘以每个线程块的共享内存大小。访问分布式共享内存中的数据需要所有线程块都存在。用户可以使用集群组 API 中的 `cluster.sync()` 来确保所有线程块都已开始执行。用户还需要确保所有分布式共享内存操作都在线程块退出之前完成，例如，如果远程线程块试图读取给定线程块的共享内存，用户需要确保在远程线程块能够退出之前，它对共享内存的读取已经完成。CUDA 提供了一种机制来访问分布式共享内存，应用程序可以通过利用其功能来获得好处。

接下来，我们来看一个直方图计算，以及如何使用线程块集群在 GPU 上对其进行优化。计算直方图的标准方法是在每个线程块的共享内存中进行计算，然后执行全局内存原子操作。这种方法的局限性在于共享内存的容量。一旦直方图的桶（bins）不再适合放入共享内存中，用户就需要直接在全局内存中计算直方图，并因此在全局内存中执行原子操作。基于分布式共享内存，CUDA 提供了一个中间步骤，根据直方图桶的大小，直方图可以在共享内存、分布式共享内存或直接在全局内存中计算。下面的 CUDA 内核示例展示了如何根据直方图桶的数量在共享内存或分布式共享内存中计算直方图。

```cpp
#include <cooperative_groups.h>

// Distributed Shared memory histogram kernel
__global__ void clusterHist_kernel(int *bins, const int nbins, const int bins_per_block, const int *__restrict__ input,
                                   size_t array_size)
{
  extern __shared__ int smem[];
  namespace cg = cooperative_groups;
  int tid = cg::this_grid().thread_rank();

  // Cluster initialization, size and calculating local bin offsets.
  cg::cluster_group cluster = cg::this_cluster();
  unsigned int clusterBlockRank = cluster.block_rank();
  int cluster_size = cluster.dim_blocks().x;

  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    smem[i] = 0; // Initialize shared memory histogram to zeros
  }

  // cluster synchronization ensures that shared memory is initialized to zero in
  // all thread blocks in the cluster. It also ensures that all thread blocks
  // have started executing and they exist concurrently.
  cluster.sync();

  for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
  {
    int ldata = input[i];

    // Find the right histogram bin.
    int binid = ldata;
    if (ldata < 0)
      binid = 0;
    else if (ldata >= nbins)
      binid = nbins - 1;

    // Find destination block rank and offset for computing
    // distributed shared memory histogram
    int dst_block_rank = (int)(binid / bins_per_block);
    int dst_offset = binid % bins_per_block;

    // Pointer to target block shared memory
    int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

    // Perform atomic update of the histogram bin
    atomicAdd(dst_smem + dst_offset, 1);
  }

  // cluster synchronization is required to ensure all distributed shared
  // memory operations are completed and no thread block exits while
  // other thread blocks are still accessing distributed shared memory
  cluster.sync();

  // Perform global memory histogram, using the local distributed memory histogram
  int *lbins = bins + cluster.block_rank() * bins_per_block;
  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    atomicAdd(&lbins[i], smem[i]);
  }
}
```

上述内核可以在运行时根据所需的分布式共享内存量来设定集群大小并启动。如果直方图足够小，能够放入单个线程块的共享内存中，那么用户可以将集群大小设置为 1 来启动内核。下面的代码片段展示了如何根据共享内存的需求动态地启动一个集群内核。

```cpp
// Launch via extensible launch
{
  cudaLaunchConfig_t config = {0};
  config.gridDim = array_size / threads_per_block;
  config.blockDim = threads_per_block;

  // cluster_size depends on the histogram size.
  // ( cluster_size == 1 ) implies no distributed shared memory, just thread block local shared memory
  int cluster_size = 2; // size 2 is an example here
  int nbins_per_block = nbins / cluster_size;

  // dynamic shared memory size is per block.
  // Distributed shared memory size =  cluster_size * nbins_per_block * sizeof(int)
  config.dynamicSmemBytes = nbins_per_block * sizeof(int);

  CUDA_CHECK(::cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = cluster_size;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;

  config.numAttrs = 1;
  config.attrs = attribute;

  cudaLaunchKernelEx(&config, clusterHist_kernel, bins, nbins, nbins_per_block, input, array_size);
}
```

### 页锁定主机内存（Page-Locked Host Memory）

runtime 环境提供了函数，允许使用页锁定（也称为固定）主机内存（相对于 `malloc()` 分配的常规可分页主机内存）。`cudaHostAlloc()` 和 `cudaFreeHost()` 用于分配和释放页锁定主机内存；`cudaHostRegister()` 用于将 `malloc()` 分配的一段内存进行页锁定。使用页锁定主机内存具有多重优势：对于某些设备，页锁定主机内存与设备内存之间的复制可以与内核执行同时进行。在某些设备上，页锁定主机内存可以映射到设备的地址空间中，从而无需在设备内存与主机内存之间复制数据。在具有前端总线的系统上，如果主机内存被分配为页锁定，则主机内存与设备内存之间的带宽会更高；如果此外还将其分配为写合并（如写合并内存（Write-Combining Memory）中所述），则带宽会进一步提高。

### 内存同步域（Memory Synchronization Domains）

#### 内存栅栏函数

CUDA 编程模型假设设备具有弱排序内存模型，即 CUDA 线程向共享内存、全局内存、页锁定主机内存或对等设备内存写入数据的顺序，不一定是另一个 CUDA 线程或主机线程观察到数据被写入的顺序。如果两个线程在没有同步的情况下对同一内存位置进行读取或写入，那么这是未定义行为。下面的示例中，线程 1 执行 `writeXY()` 函数，而线程 2 执行 `readXY()` 函数：

```cpp
__device__ int X = 1, Y = 2;

__device__ void writeXY()
{
    X = 10;
    Y = 20;
}

__device__ void readXY()
{
    int B = Y;
    int A = X;
}
```

两个线程同时从相同的内存位置 X 和 Y 进行读写操作。任何数据竞争都是未定义行为，并且没有明确的语义。因此，A 和 B 的最终值可能是任何值。可以使用内存栅栏函数来强制对内存访问进行顺序一致的排序。内存栅栏函数在强制排序的范围上有所不同，但它们与所访问的内存空间（共享内存、全局内存、页锁定主机内存和对等设备的内存）无关。

`void __threadfence_block()` 等价于 `cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block)`，并确保：

1. 调用线程在调用 `__threadfence_block()` 之前对所有内存的所有写入，对于调用线程所在块中的所有线程而言，都被视为发生在调用 `__threadfence_block()` 之后调用线程对所有内存的所有写入之前；
2. 调用线程在调用 `__threadfence_block()` 之前对所有内存的所有读取，都按序排列在调用线程在调用 `__threadfence_block()` 之后对所有内存的所有读取之前。

`void __threadfence();` 等价于 `cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device)`，并确保：调用 `__threadfence()` 之后调用线程对所有内存的任何写入，都不会被设备中的任何线程观察到发生在调用 `__threadfence()` 之前调用线程对所有内存的任何写入之前。在之前的代码示例中，可以在代码中插入栅栏，如下：

```cpp
__device__ int X = 1, Y = 2;

__device__ void writeXY()
{
    X = 10;
    __threadfence();
    Y = 20;
}

__device__ void readXY()
{
    int B = Y;
    __threadfence();
    int A = X;
}
```

对于这段代码，可以观察到以下结果：

1. A=1 且 B=2；
2. A=10 且 B=2；
3. A=10 且 B=20。

第四种结果是不可能的，因为第一次写入必须在第二次写入之前可见。如果线程 1 和线程 2 属于同一个块，那么使用 `__threadfence_block()` 就足够了。如果线程 1 和线程 2 不属于同一个块，但它们是来自同一个设备的 CUDA 线程，则必须使用 `__threadfence()`。
