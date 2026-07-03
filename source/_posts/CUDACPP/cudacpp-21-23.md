---
title: CUDA C++ 笔记（十四）第21-23章 Driver API
date: 2024-07-31 20:00:00
tags: [CUDA, CUDA Environment, API]
categories: [CUDA C++ Programming Guide]
description: 第21-23章，介绍 CUDA 相关的环境变量、API 和其他参考内容。
---

# 第21、22、23章

# Driver API

**location**: `ls /usr/local/cuda/include/cuda.h`

```
ls -al /usr/lib/x86_64-linux-gnu/libcuda.so
```

All its entry points are prefixed with `cu`.

![](/assets/cudacpp-21-23/image.png)

Sample demonstration

![](/assets/cudacpp-21-23/image1.png)

## Context

1. `cuCtxCreate`: create a context with a usage count of 1.
1. `cuCtxDestroy`: destroy a context, clean up all resources associated with the context.
1. Resources:
    - `CUmodule`, `CUfunction`, `CUstream`, `CUevent`...
    - `cuMemAlloc`, `cuMemAllocHost`, `cuMemAllocManaged`, `cuMemAllocPitch`

## Module

1. Dynamically loadable package of device code and data. Like DLL in Windows.

Demo of PTX JIT

![](/assets/cudacpp-21-23/image2.png)

## Interoperability between Runtime and Driver APIs

1. If a context is created and made current via the driver API, subsequent runtime calls will pick up this context instead of creating a new one.
1. If the runtime is initialized (implicitly as mentioned in CUDA Runtime), `cuCtxGetCurrent()` can be used to retrieve the context created during initialization. This context can be used by subsequent driver API calls.
1. Device memory can be allocated and freed using either API. `CUdeviceptr` can be cast to regular pointers and vice-versa.

```cpp
CUdeviceptr devPtr;
float* d_data;
// Allocation using driver API
cuMemAlloc(&devPtr, size);
d_data = (float*)devPtr;
// Allocation using runtime API
cudaMalloc(&d_data, size);
devPtr = (CUdeviceptr)d_data;
```

## Driver Func Ptr

Like `dlsym` on POSIX.

`/usr/local/cuda/include/cudaTypedefs.h` (for `cuda.h`).

Version based naming scheme:

```
PFN_cuMemAlloc_v3020 pfn_cuMemAlloc_v2;  // CUDA 3.2
```

# Environment Variables

`CUDA_VISIBLE_DEVICES`: used to control which GPUs are visible to a CUDA application.

This is particularly useful in systems with multiple GPUs, where you want to restrict a program to use only specific GPUs. By setting `CUDA_VISIBLE_DEVICES`, you can limit the program's access to a subset of the available GPUs.

Sample demo:

![](/assets/cudacpp-21-23/image3.png)

![](/assets/cudacpp-21-23/image4.png)

Setting: `os.environ["CUDA_VISIBLE_DEVICES"] = "0"`

![](/assets/cudacpp-21-23/image5.png)

# Lazy Loading

**Problem**:

1. Program init includes libraries: CUDA Runtime loads all modules.
1. Most of the time, programs only use a small amount of kernels from the libraries they include.

**Lazy Loading**

>= CUDA 11.8

```cpp
#include "cuda.h"
#include "assert.h"
#include "iostream"
int main() {
    CUmoduleLoadingMode mode;
    assert(CUDA_SUCCESS == cuInit(0));
    assert(CUDA_SUCCESS == cuModuleGetLoadingMode(&mode));
    std::cout << "CUDA Module Loading Mode is " << ((mode == CU_MODULE_LAZY_LOADING) ? "lazy" : "eager") << std::endl;
    return 0;
}
```

Delays loading of CUDA modules and kernels from program initialization closer to kernel execution.

Only load kernels they are actually going to use.

**Benefit**:

1. Saving time on initialization.
1. Reduces memory overhead, both on GPU memory and host memory.

**Usage**:

```python
os.environ["CUDA_MODULE_LOADING"] = "LAZY"  # DEFAULT|LAZY|EAGER
#       CUDA Runtime: Each module will be loaded on first usage of a variable or a kernel from that module.
#       CUDA Driver:
#           'cuModuleLoad': will not be loading kernels immediately, instead it will delay loading of a kernel until cuModuleGetFunction() is called.

os.environ["CUDA_MODULE_DATA_LOADING"] = "LAZY"
#       'cuLibraryLoad' to load module data into memory
```

**Mis**:

1. Default is LAZY.
1. Turn off when auto-tuning.
1. Custom allocator cannot allocate the entire VRAM on startup.
