---
title: Cutlass 学习笔记（一）CUTLASS 与 CuTe
date: 2025-08-15 12:00:00
tags: [CUTLASS]
categories: [Cutlass 学习笔记,Cutlass]
description: CUTLASS 是 NVIDIA 推出的开源 CUDA 模板库，专注于高效实现 GPU 上的线性代数运算。本文涵盖 CUTLASS 概览、CuTe 核心概念、API 层级结构、Profiler 使用方法，以及头文件实现的原因分析。
---

CUTLASS（CUDA Templates for Linear Algebra Subroutines）是 NVIDIA 开源的 CUDA C++ 模板库，用一组模板抽象在 CUDA 中以所有层级和规模实现高性能矩阵乘（GEMM）及相关计算。它采用与 cuBLAS 类似的分层分解与数据搬运策略，把矩阵乘涉及的各项"移动部件"拆解为可复用、模块化的组件，用户可以针对分块大小、数据类型和算法策略对每个组件单独特化与调优。

## CUTLASS 与 cuBLAS、cuDNN 的对比

相比闭源的 cuBLAS、cuDNN，CUTLASS 的价值主要体现在几点。

- 灵活性。cuBLAS 等库是黑盒，内部实现难以修改；CUTLASS 提供模板化代码，允许开发者调整数据布局、分块策略、指令级优化。
- 高性能。它利用 GPU 的 Tensor Core、内存层级优化（共享内存、寄存器）和软件流水线，达到接近硬件极限的性能。
- 透明性。开源且模块化，便于理解底层优化原理，适合教育和研究。
- 特殊需求。支持混合精度、稀疏矩阵等非标准场景，满足量化训练等新兴算法的需求。

![CUTLASS vs cuBLAS vs cuDNN](/assets/cutlass_cute/image(1).png)

## cutlass 与 CuTe 的分工

CUTLASS 包括 cutlass 和 CuTe 两个部分。按照矩阵乘的层级结构，cutlass 负责具体矩阵乘算法的实现，包括 GEMM、split-k、grouped gemm、warp specialized 等；CuTe 则通过 Layout 代数，用 TiledMMA 完成具体数据的计算。

### CuTe 核心概念

CUTLASS 3.0 引入了新的核心库 CuTe，用于描述和操作线程与数据的张量。CuTe 是一组 C++ CUDA 模板抽象，用于定义和操作线程与数据的分级多维布局（hierarchically multidimensional layouts）。它提供 Layout 和 Tensor 对象，这些对象紧凑地封装了数据的类型、形状、内存空间和布局，同时替用户处理复杂的索引计算。

这样一来，程序员可以专注于算法的逻辑描述，把机械的流水账工作交给 CuTe。借助这些工具，我们可以快速设计、实现和修改各种稠密线性代数操作。

CuTe 的核心抽象是分级多维布局，布局可以与数据数组组合来表示张量。它的表达能力足以涵盖实现高效稠密线性代数所需的几乎所有结构。布局还能通过函数式组合进行组合和操作，在此基础上我们构建了切片（tiling）、分区（partitioning）等大量常见操作。

CUTLASS 3.0 及更高版本在整个 GEMM 层次结构的模板中采用了 CuTe，这大幅简化了设计，并提升了代码的可组合性和可读性。

## CUTLASS 的 API 层级

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

## CUTLASS 采用头文件实现的原因

CUTLASS 几乎全部由头文件（.h、.hpp、.cuh）实现，而没有预先编译成 .cpp 或 .cu，核心在于 C++ 模板的机制和对性能的极致追求。

### 核心原因一 C++ 模板的实例化机制

这是最根本的原因。CUTLASS 本质上是一个模板库。

- 原理。在 C++ 中，模板只是一份生成代码的蓝图，本身并不是可执行的代码。编译器只有在知道具体数据类型（如 float、half、int8）以及参数（如 Tile 大小、流水线级数）时，才会生成机器码，这个过程叫做模板实例化。
- 限制。实例化要求编译器在编译期看到完整源码。如果把代码放进 .cpp 编译成 .lib 或 .so，模板就被固化了，用户只能使用库作者预先定义好的那几种类型。
- CUTLASS 的情况。CUTLASS 允许用户自定义极其复杂的参数（数据类型、矩阵布局、对齐方式、线程块形状、Warp 形状等）。用 .cpp 实现就无法预知用户会组合出什么参数，因此必须把源码放在头文件里，让用户的编译器在编译时现场生成代码。

### 核心原因二 内联与性能优化

GPU 编程对性能极其敏感。

- 内联。CUTLASS 内部有大量细粒度的小函数（加载数据、乘加运算、指针移动）。如果这些函数分散在独立的编译单元（.cpp、.cu）里，编译器很难跨单元做内联优化。
- 寄存器分配。NVCC 需要看到完整的 kernel 逻辑，才能最优化地分配有限的寄存器资源。代码分散在预编译库中时，编译器看到的可能只是函数调用指令，这会阻碍针对指令流水线和延迟隐藏的优化。
- 常数折叠。Tile 大小等许多参数是编译期常量，放在头文件里可以直接硬编码进指令，省去运行时从内存读取变量，运行速度因此明显提升。

### 核心原因三 编译期元编程

CUTLASS 大量使用模板元编程（TMP），在编译阶段就完成计算，例如根据数据类型自动选择 Tensor Core 指令还是 CUDA Core 指令、计算最优的访存步长、计算 shared memory 需要的大小。这些计算必须在编译期完成，因此代码必须放在头文件里供编译器推导，不能是运行时链接的二进制代码。

## 性能数据与 Profiler


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
