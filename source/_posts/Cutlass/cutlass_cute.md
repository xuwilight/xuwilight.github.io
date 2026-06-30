---
title: CUTLASS/CuTe 学习笔记
date: 2025-08-15 12:00:00
tags: [CUTLASS, CuTe, GPU]
categories: [Cutlass 学习笔记]
description: Cutlass 是 NVIDIA 推出的开源 CUDA 模板库，专注于高效实现 GPU 上的线性代数运算如矩阵乘法，并介绍 CuTe 的核心概念与设计思想。
---

![](/assets/cutlass_cute/image.png)

**Cutlass** 是 NVIDIA 推出的一个开源 CUDA 模板库（**C**UDA **T**emplates for **L**inear **A**lgebra **S**ubroutines），专注于高效实现 GPU 上的线性代数运算（如矩阵乘法）。

### 1. Cutlass 是什么？

定位：基于 CUDA 的模板化高性能计算库，用于实现自定义的线性代数运算（尤其是 GEMM，通用矩阵乘法）。

设计目标：提供高度模块化和可定制的 GPU 核函数，允许开发者根据硬件特性（如 Tensor Core）和具体需求调整算法，平衡性能与灵活性。

适用场景：需要精细优化或定制矩阵运算的领域（如深度学习框架、科学计算）。

### 2. 为什么需要 Cutlass？

灵活性：cuBLAS 等库是"黑盒"，难以修改内部实现；而 Cutlass 提供模板化代码，允许开发者调整数据布局、分块策略、指令级优化等。

高性能：通过利用 GPU 的 Tensor Core、内存层级优化（共享内存、寄存器）和流水线技术，实现接近硬件极限的性能。

透明性：开源且模块化，便于理解底层优化原理，适合教育和研究。

特殊需求：支持混合精度计算、稀疏矩阵等非标准场景，满足新兴算法需求（如 AI 模型中的量化训练）。

### 3. Cutlass vs. cuBLAS vs. cuDNN

![](/assets/cutlass_cute/image(1).png)

**Cute**

CUTLASS 3.0 引入了一个新的核心库 **CuTe**，用于描述和操作线程与数据的张量。CuTe 是一组 C++ CUDA 模板抽象，用于定义和操作线程与数据的分级多维布局（hierarchically multidimensional layouts）。CuTe 提供了 **Layout** 和 **Tensor** 对象，这些对象能够紧凑地封装数据的类型、形状、内存空间和布局，同时为用户处理复杂的索引计算。这使得程序员可以专注于算法的逻辑描述，而 CuTe 则为他们处理机械的流水账工作。借助这些工具，我们可以快速设计、实现和修改所有密集线性代数操作。

CuTe 的核心抽象是分级多维布局，这些布局可以与数据数组组合以表示张量。布局的表示能力非常强大，足以涵盖我们实现高效密集线性代数所需的几乎所有内容。布局还可以通过函数式组合进行组合和操作，基于此，我们构建了大量常见操作，例如切片（tiling）和分区（partitioning）。

CUTLASS 3.0 及更高版本在其模板的整个 GEMM 层次结构中采用了 CuTe。这极大地简化了设计，并提高了代码的可组合性和可读性。

cutlass 的设计理念：

通过这一系列笔记记录下学习 cutlass 和 cute 的过程，学习内容主要基于 cutlass 3.8 ([833f699](https://github.com/NVIDIA/cutlass/commit/833f6990e031b48b4cd2fcf55e0849c51ef6bac2))。用到的代码保存在[https://gitlab.temu.team/westin.xu/train_jobs](https://gitlab.temu.team/westin.xu/train_jobs)仓库中。

整个笔记的结构如下：

1. 首先学习 cute 中的 Layout 概念。我们可以将 layout 理解成一个函数，用于将数据从 tensor 中的逻辑位置映射到实际在内存中的物理位置。
1. 然后学习 tensor 的概念，以及一些常用的 tensor 处理操作。
1. 学习完 cute 中基本的数据结构后，接下来会学习 mma 和 copy。cute 针对不同的架构封装了不同的 mma 和 copy 指令，选择合适的操作可以提高计算速度。
1. 有了上面的知识我们可以使用 cute 实现一些基本的操作了，但是性能可能不会太好。这时我们需要了解提高性能的一些方法，如提高数据传输速度（swizzle）和提高计算效率（pipeline）。
1. 接下来，以 cute/examples 下的代码为例，介绍如何使用 cute 实现通用矩阵乘（Gemm)，并针对不同架构的特性提高计算效率。
1. 最后学习如何使用 cutlass 和 cute 实现 flash attention2，以及针对 Hopper 架构优化的 flash attention3。

**References**

1. https://github.com/NVIDIA/cutlass
1. https://www.zhihu.com/column/c_1696937812497235968
1. CUTLASS3_pdf
