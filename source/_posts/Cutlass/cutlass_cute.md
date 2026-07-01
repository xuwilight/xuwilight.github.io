---
title: Cutlass 学习笔记（一）CUTLASS 与 CuTe
date: 2025-08-15 12:00:00
tags: [CUTLASS, CuTe]
categories: [Cutlass 学习笔记,Cutlass]
description: CUTLASS 是 NVIDIA 推出的开源 CUDA 模板库，专注于高效实现 GPU 上的线性代数运算。本文涵盖 CUTLASS 概览、CuTe 核心概念、API 层级结构、Profiler 使用方法，以及头文件实现的原因分析。
---

## CUTLASS 是什么

**CUTLASS**（**C**UDA **T**emplates for **L**inear **A**lgebra **S**ubroutines）是 NVIDIA 推出的一个开源 CUDA 模板库，专注于高效实现 GPU 上的线性代数运算（如矩阵乘法 GEMM）及相关计算，支持所有层级和规模。

CUTLASS 对高性能矩阵乘进行模板抽象，融合了层次分解和数据移动策略，将这些"移动部件"分解为可重用的模块化软件组件和抽象概念。并行化层次结构中不同层级的基本单元可以通过自定义分块大小、数据类型和其他算法策略进行专门化和调优，由此带来的灵活性简化了它们在自定义内核和应用程序中作为构建模块的使用。

## 为什么需要 CUTLASS

- **灵活性**：cuBLAS 等库是黑盒，难以修改内部实现；而 CUTLASS 提供模板化代码，允许开发者调整数据布局、分块策略、指令级优化等。
- **高性能**：通过利用 GPU 的 Tensor Core、内存层级优化（共享内存、寄存器）和流水线技术，实现接近硬件极限的性能。
- **透明性**：开源且模块化，便于理解底层优化原理，适合教育和研究。
- **特殊需求**：支持混合精度计算、稀疏矩阵等非标准场景，满足新兴算法需求（如 AI 模型中的量化训练）。

## CUTLASS vs. cuBLAS vs. cuDNN

![CUTLASS vs cuBLAS vs cuDNN](/assets/cutlass_cute/image(1).png)

## CUTLASS 架构：cutlass 与 CuTe

CUTLASS 包括 **cutlass** 和 **CuTe** 两个部分。根据矩阵乘的层级结构，cutlass 负责具体的矩阵乘算法的实现，包括 GEMM、split-k、grouped gemm、warp specialized 等。而 CuTe 则通过 Layout 代数，使用 TiledMMA 进行具体数据的计算。

### CuTe 核心概念

CUTLASS 3.0 引入了一个新的核心库 **CuTe**，用于描述和操作线程与数据的张量。CuTe 是一组 C++ CUDA 模板抽象，用于定义和操作线程与数据的分级多维布局（hierarchically multidimensional layouts）。CuTe 提供了 **Layout** 和 **Tensor** 对象，这些对象能够紧凑地封装数据的类型、形状、内存空间和布局，同时为用户处理复杂的索引计算。

这使得程序员可以专注于算法的逻辑描述，而 CuTe 则为他们处理机械的流水账工作。借助这些工具，我们可以快速设计、实现和修改所有密集线性代数操作。

CuTe 的核心抽象是分级多维布局，这些布局可以与数据数组组合以表示张量。布局的表示能力非常强大，足以涵盖我们实现高效密集线性代数所需的几乎所有内容。布局还可以通过函数式组合进行组合和操作，基于此，我们构建了大量常见操作，例如切片（tiling）和分区（partitioning）。

CUTLASS 3.0 及更高版本在其模板的整个 GEMM 层次结构中采用了 CuTe，这极大地简化了设计，并提高了代码的可组合性和可读性。

## cutlass API 层级

1. cutlass api 2.x
2. cutlass api 3.x
3. device level
4. kernel level
5. collective
6. epilogue
7. GEMM
    - default gemm
    - gemm split-k
    - grouped gemm
    - warp specialized

## 为什么 CUTLASS 用头文件实现

CUTLASS 之所以几乎全部由头文件（.h, .hpp, .cuh）实现，而不用 .cpp 或 .cu 文件进行预编译，核心原因在于 **C++ 模板的机制**以及**极致的性能追求**。

### 核心原因 1：C++ 模板的实例化机制

这是最根本的原因。CUTLASS 本质上是一个**模板库**。

- **原理**：在 C++ 中，模板不是真正的代码，而是生成代码的"蓝图"。编译器只有在知道具体要使用什么数据类型（如 float、half、int8）以及什么参数（如 Tile 大小、流水线级数）时，才会真正生成机器码。这个过程叫模板实例化。
- **限制**：为了进行实例化，编译器必须在编译时看到完整的源代码。如果将代码放入 .cpp 编译成 .lib 或 .so，那么模板就已经被"固化"了，用户只能使用库作者预先定义好的那几种类型。
- **CUTLASS 的场景**：CUTLASS 允许用户自定义极其复杂的参数（数据类型、矩阵布局、对齐方式、线程块形状、Warp 形状等）。如果用 .cpp 实现，开发者无法预知用户会组合出什么样的参数，因此必须把源码放在头文件里，让用户的编译器在编译时现场生成代码。

### 核心原因 2：内联与性能优化

GPU 编程（CUDA）对性能极其敏感。

- **内联（Inlining）**：CUTLASS 内部包含大量细粒度的小函数（例如加载数据、乘加运算、指针移动）。如果这些函数在单独的编译单元（.cpp/.cu）中，编译器很难进行跨单元的内联优化。
- **寄存器分配**：CUDA 编译器（NVCC）需要看到完整的 kernel 代码逻辑，才能最优化地分配有限的寄存器资源。如果代码分散在预编译库中，编译器看到的可能是函数调用指令，这会阻止许多针对指令流水线和延迟隐藏的优化。
- **常数折叠**：由于许多参数（如 Tile 大小）是编译期常量，放在头文件中可以让编译器直接把这些数字硬编码进指令里，而不是去内存读取变量，极大提升运行速度。

### 核心原因 3：编译期元编程

CUTLASS 大量使用了模板元编程（TMP），它在**编译阶段**就在计算逻辑，例如：

- 根据数据类型自动选择使用 Tensor Core 指令还是 CUDA Core 指令。
- 计算最优的内存访问步长。
- 计算 Shared Memory 需要的大小。

这些计算必须在编译期完成，因此代码必须存在于头文件中供编译器推导，而不能是运行时链接的二进制代码。

## 性能数据与 Profiler

性能数据文件：`azure_h100_gemm_mnk4096.gemm.csv`（3.27 MB）

### 编译 Profiler

```bash
cmake .. \
  -DCUTLASS_NVCC_ARCHS=90a \
  -DCUTLASS_USE_SYSTEM_GOOGLETEST=ON \
  -DCUTLASS_ENABLE_LIBRARY=ON \
  -DCUTLASS_ENABLE_PROFILER=ON \
  -DCUTLASS_LIBRARY_KERNELS="*sm90_tensorop_gemm_f16_f16_f16_f16_f16" \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=OFF
# -DCMAKE_BUILD_TYPE=Debug

# kernel 名字格式类似：
# cutlass3x_sm90_tensorop_gemm_f16_f16_f16_f16_f16_128x256x64_1x2x1_0_tnn_align8_warpspecialized_pingpong_epi_tma

make cutlass_profiler -j64  # 线程数不宜过多，否则可能爆内存
```

### 运行 Profiler

```bash
./cutlass_profiler --m=4096 --n=4096 --k=4096
```

输出示例：

```
=============================
  Problem ID: 1

         Provider: CUTLASS
    OperationKind: gemm
        Operation: cutlass3x_sm90_tensorop_gemm_f16_f16_f16_f16_f16_128x256x64_1x2x1_0_tnn_align8_warpspecialized_pingpong_epi_tma

           Status: Success
     Verification: ON
      Disposition: Passed

reference_device: Passed
           cuBLAS: Not run
            cuDNN: Not run

        Arguments: --gemm_kind=universal --m=4096 --n=4096 --k=4096 --A=f16:row --B=f16:column --C=f16:column --D=f16:column \
                   --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1 --batch_count=1 --raster_order=heuristic \
                   --runtime_input_datatype_a=invalid --runtime_input_datatype_b=invalid --use_pdl=false --enable_sm90_mixed_dtype_shuffle_test=false \
                   --swizzle_size=1 --op_class=tensorop --accum=f16 --cta_m=128 --cta_n=256 --cta_k=64 --cluster_m=1 --cluster_n=2 \
                   --cluster_k=1 --cluster_m_fallback=0 --cluster_n_fallback=0 --cluster_k_fallback=0 --stages=4 --warps_m=4 \
                   --warps_n=1 --warps_k=1 --inst_m=64 --inst_n=256 --inst_k=16 --min_cc=90 --max_cc=90

            Bytes: 100663296 bytes
            FLOPs: 137472507904 flops
        FLOPs/Byte: 1365

          Runtime: 0.155846 ms
           Memory: 601.554 GiB/s

             Math: 882103 GFLOP/s


=============================
```


## References

1. [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)
2. [知乎专栏](https://www.zhihu.com/column/c_1696937812497235968)
3. CUTLASS3 相关 PDF
