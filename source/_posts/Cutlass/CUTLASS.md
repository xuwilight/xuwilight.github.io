---
title: CUTLASS 概览
date: 2025-08-15 12:00:00
tags: [CUTLASS, GPU, GEMM]
categories: [Cutlass 学习笔记]
description: CUTLASS 对高性能矩阵乘进行模板抽象的集合，用于在 CUDA 中实现高性能矩阵乘法 (GEMM) 及相关计算，支持所有层级和规模。它融合了层次分解和数据移动策略。CUTLASS 将这些“移动部件”分解为可重用的模块化软件组件和抽象概念。
---

CUTLASS 对高性能矩阵乘进行模板抽象的集合，用于在 CUDA 中实现高性能矩阵乘法 (GEMM) 及相关计算，支持所有层级和规模。它融合了层次分解和数据移动策略。CUTLASS 将这些“移动部件”分解为可重用的模块化软件组件和抽象概念。

概念并行化层次结构中不同层级的基本单元可以通过自定义分块大小、数据类型和其他算法策略进行专门化和调优。由此带来的灵活性简化了它们在自定义内核和应用程序中作为构建模块的使用。

cutlass 包括 cutlass 和 cute 两个部分。根据矩阵乘的层级结构，cutlass 负责具体的矩阵乘算法的实现，包括 gemm，split-k，grouped gemm，warp specialized 等。而 cute 则通过 layout 代数，使用 tiledmma 进行具体数据的计算。

1. cutlass api 2x
1. cutlass api 3x
1. device level
1. kernel level
1. collective
1. epilogue
1. GEMM
1. default gemm
1. gemm split-k
1. grouped gemm
1. warp specified

cutlass 性能数据

azure_h100_gemm_mnk4096.gemm.csv(3.27MB)

cutlass profiler

```cpp
cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_USE_SYSTEM_GOOGLETEST=ON -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_ENABLE_LIBRARY=ON -DCUTLASS_ENABLE_PROFILER=ON -DCUTLASS_LIBRARY_KERNELS="*sm90_tensorop_gemm_f16_f16_f16_f16_f16" -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_ENABLE_EXAMPLES=OFF // -DCMAKE_BUILD_TYPE=Debug

// kernel 名字格式类似这样 cutlass3x_sm90_tensorop_gemm_f16_f16_f16_f16_f16_128x256x64_1x2x1_0_tnn_align8_warpspecialized_pingpong_epi_tma

make cutlass_profiler -j64 // 线程多了会爆内存

./cutlass_profiler --m=4096 --n=4096 --k=4096

=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass3x_sm90_tensorop_gemm_f16_f16_f16_f16_f16_128x256x64_1x2x1_0_tnn_align8_warpspecialized_pingpong_epi_tma

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed
          cuBLAS: Not run
           cuDNN: Not run

       Arguments: --gemm_kind=universal --m=4096 --n=4096 --k=4096 --A=f16:row --B=f16:column --C=f16:column --D=f16:column  \
                  --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1 --batch_count=1 --raster_order=heuristic  \
                  --runtime_input_datatype_a=invalid --runtime_input_datatype_b=invalid --use_pdl=false --enable_sm90_mixed_dtype_shuffle_test=false  \
                  --swizzle_size=1 --op_class=tensorop --accum=f16 --cta_m=128 --cta_n=256 --cta_k=64 --cluster_m=1 --cluster_n=2  \
                  --cluster_k=1 --cluster_m_fallback=0 --cluster_n_fallback=0 --cluster_k_fallback=0 --stages=4 --warps_m=4  \
                  --warps_n=1 --warps_k=1 --inst_m=64 --inst_n=256 --inst_k=16 --min_cc=90 --max_cc=90

           Bytes: 100663296  bytes
           FLOPs: 137472507904  flops
           FLOPs/Byte: 1365

         Runtime: 0.155846  ms
          Memory: 601.554 GiB/s

            Math: 882103 GFLOP/s


=============================
```

为什么 cutlass 中都是用头文件实现的。

CUTLASS（CUDA Templates for Linear Algebra Subroutines）之所以几乎全部由头文件（.h, .hpp, .cuh）实现，而不用 .cpp 或 .cu 文件进行预编译，核心原因在于 **C++ 模板（Templates）的机制** 以及 **极致的性能追求**。

以下是详细的深度解析，以及这两种实现方式的区别：

**核心原因 1：C++ 模板的实例化机制 (The Nature of C++ Templates)**

这是最根本的原因。CUTLASS 本质上是一个**模板库**。

原理：在 C++ 中，模板（Templates）不是真正的代码，而是生成代码的“蓝图”。编译器只有在知道你具体要用什么数据类型（如 float, half, int8）以及什么参数（如 Tile 大小、流水线级数）时，才会真正地生成机器码。这个过程叫模板实例化。

限制：为了进行实例化，编译器必须在编译时看到完整的源代码。如果将代码放入 .cpp 编译成 .lib/.so，那么模板就已经被“固化”了，用户只能使用库作者预先定义好的那几种类型。

CUTLASS 的场景：CUTLASS 允许用户自定义极其复杂的参数（数据类型、矩阵布局、对齐方式、线程块形状、Warp 形状等）。如果用 .cpp 实现，开发者无法预知用户会组合出什么样的参数，因此必须把源码放头文件里，让用户的编译器在编译时现场生成代码。

**核心原因 2：内联与性能优化 (Inlining & Optimization)**

GPU 编程（CUDA）对性能极其敏感。

内联 (Inlining)：CUTLASS 内部包含大量细粒度的小函数（例如加载数据、乘加运算、指针移动）。如果这些函数在单独的编译单元（.cpp/.cu）中，编译器很难进行跨单元的内联优化。

寄存器分配：CUDA 编译器（NVCC）需要看到完整的 kernel 代码逻辑，才能最优化地分配有限的寄存器资源。如果代码分散在预编译库中，编译器看到的可能是函数调用指令（Function Call），这会阻止许多针对指令流水线和延迟隐藏的优化。

常数折叠：由于许多参数（如 Tile 大小）是编译期常量（Compile-time constants），放在头文件中可以让编译器直接把这些数字“硬编码”进指令里，而不是去内存读取变量，极大提升速度。

**核心原因 3：编译期元编程 (Template Metaprogramming)**

CUTLASS 大量使用了 TMP（模板元编程）。它在**编译阶段**就在计算逻辑，例如：

根据数据类型自动选择使用 Tensor Core 指令还是 CUDA Core 指令。

计算最优的内存访问步长。

计算 Shared Memory 需要的大小。

这些计算必须在编译期完成，因此代码必须存在于头文件中供编译器推导，而不能是运行时链接的二进制代码。
