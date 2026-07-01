---
title: Cutlass 学习笔记（二）Cutlass API
date: 2025-08-16 12:00:00
tags: [CUTLASS]
categories: [Cutlass 学习笔记,Cutlass]
description: CUTLASS 为层次结构中每个级别的矩阵乘法累加运算提供了一个统一的编程模型。主要分为设备级（device-level）、线程块级（threadblock-level）、warp-level GEMM、thread-level GEMM 和指令级 GEMM。
---

CUTLASS 为层次结构中每个级别的矩阵乘法累加运算提供了一个统一的编程模型。主要分为设备级（device-level）、线程块级（threadblock-level）、warp-level GEMM、thread-level GEMM 和指令级 GEMM。

![](/assets/cutlass_api/image.png)

### Device-wide GEMM API

设备级 GEMM API 旨在简化 GPU 上标准 GEMM 计算的实例化和执行。该运算符旨在用于主机端 .cu 代码，其语义与 cuBLAS 类似。

device-level gemm api 有下面几种。

cutlass::gemm::device::Gemm - basic GEMM operation

cutlass::gemm::device::GemmArray - batched GEMM operation in which input matrices are read from arrays of pointers

cutlass::gemm::device::GemmBatched - batched GEMM operation in which input matrices are separated by a constant stride

cutlass::gemm::device::GemmSplitKParallel - GEMM operation that partitions the GEMM K dimension then launches a separate reduction kernel

例子：混精 GEMM，Volta tensorcore

```python
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // ElementA
    cutlass::layout::ColumnMajor,              // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    cutlass::half_t,                           // ElementOutput
    cutlass::layout::ColumnMajor,              // LayoutOutput
    float,                                     // ElementAccumulator
    cutlass::arch::OpClassTensorOp,            // tag indicating Tensor Cores
    cutlass::arch::Sm70                        // tag indicating target GPU compute architecture
  >;

  Gemm gemm_op;
  cutlass::Status status;
 
  //
  // Launch GEMM on the device
  //
 
  status = gemm_op({
    {m, n, k},
    {ptrA, lda},
    {ptrB, ldb},
    {ptrC, ldc},
    {ptrD, ldd},
    {alpha, beta}
  });

  if (status != cutlass::Status::kSuccess) {
    return -1;
  }
```

### Threadblock-level GEMM API

在此范围内的 GEMM 预计能够高效地将数据块从全局内存加载到内部存储中，然后使用 warp 级 GEMM 算子计算矩阵积。

线程块范围内的矩阵乘法运算由 cutlass::gemm::threadblock::MmaPipelined 实现。该类的灵感来自 std::transform_reduce()，它计算由块迭代器定义的一系列块的累积矩阵积。

### Warp-level Matrix Multiply API

Warp 级 GEMM 算子将共享内存中的块加载到寄存器中，然后使用 Tensor Core 或 CUDA Core 计算矩阵乘法。结果累积在寄存器块中。每个操作数 A、B 和 C 都定义了迭代器。

Warp 级 GEMM API 是 CUDA WMMA API 的泛化，旨在实现以下目标：

Tensor Core 的原生矩阵乘法大小

置换共享内存布局以确保无冲突访问

在主循环之外初始化指针

高效遍历

### Thread-level GEMM API

线程级 GEMM 运算对寄存器中保存的数据执行矩阵乘法累加运算。这些运算仅针对 CUDA 核心。

概念：线程级矩阵乘法运算是满足以下概念的函数对象。

### Efficient Epilogue

CUTLASS GEMM 算子执行 mma 操作，然后进行类似于 cuBLAS 的收尾操作。CUTLASS 实现了高效的行主序收尾操作。因此，为了实现列主序 GEMM，操作数 A 和 B 需要进行转置和交换。

为了使行主序和列主序输出布局都能实现高效的行主序收尾操作，CUTLASS 的设备级 GEMM 算子 cutlass::device::Gemm 和 cutlass::device::GemmUniversal 提供了两种模板定义：

(a) 通用定义

(b) 列主序源/输出的专用定义

高效的行主序收尾操作适用于：

(i) 行主序源/输出的 GEMM 算子使用模板 (a)。它运行行主序 GEMM 和高效的行主序收尾操作。

(ii) 列主序源/输出的 GEMM 算子使用模板 (b)。它转置并交换操作数 A 和 B，以实现高效的收尾。A x B = C => Transpose(B) x Transpose(A) = Transpose(C)。对于列主序源矩阵 (C)，Transpose(C) 是行主序，高效的收尾工作基于行主序。

请注意，cuBLAS 通常需要一个列主序源矩阵 (C) 和输出矩阵 (D)。因此，CUTLASS 库仅实例化并生成具有列主序布局的 GEMM 操作数。然而，CUTLASS 本身可以针对所有输入布局组合运行行主序和列主序输出布局。因此，CUTLASS 支持以下输入和输出布局组合：

{N,T} x {N,T} => {N,T} - NN、TN、TN、TT GEMM，适用于行主序和列主序输出.

例子：

```python
cudaError_t CutlassSgemmNN(int M, int N, int K, float alpha, float const *A, int lda, float const *B, int ldb, float beta, float *C, int ldc)
{
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using RowMajor = cutlass::layout::RowMajor;
    using OpClass = cutlass::arch::OpClassSimt;
    using Arch = cutlass::arch::Sm70;
    using CutlassGemm = cutlass::gemm::device::Gemm<float, ColumnMajor, float, ColumnMajor, float, RowMajor, float, OpClass, Arch>;

    CutlassGemm gemm_op;

    CutlassGemm::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});

    cutlass::Status status = gemm_op(args);

    if (status != cutlass::Status::kSuccess)
    {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}
```

cutlass::gemm::device::Gemm，矩阵乘的入口，包括 GemmKernel，Arguments，以及启动相关的函数。根据不同的模板参数，如 arch 和 stage 等，会实例化不同的 GemmKernel。

在这里 OpClass = cutlass::arch::OpClassSimt，Arch = cutlass::arch::Sm70。所以会实例化下面的参数。

```python
template <
  typename ArchTag,
  typename ElementA, 
  typename ElementB, 
  typename ElementC, 
  typename ElementAccumulator>
struct DefaultGemmConfiguration<
  arch::OpClassSimt, 
  ArchTag,
  ElementA, 
  ElementB, 
  ElementC, 
  ElementAccumulator> {
  
  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  using ThreadblockShape = GemmShape<128, 128, 8>;
  using WarpShape = GemmShape<32, 64, 8>;
  using InstructionShape = GemmShape<1, 1, 1>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    1,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Operator = arch::OpMultiplyAdd;
};
```

可以看到 ThreadblockShape MNK = <128, 128, 8>，说明一个 threadblock 会处理 128*128*8 大小的数据。

WarpShape = GemmShape<32, 64, 8>，可能是一个 warp 处理 32*64*8 的数据，所以一个 threadblock 需要 8 个 warp。

InstructionShape = GemmShape<1, 1, 1>，可能是一个指令处理一个 ffma 的意思？如果一个 warp 处理 32*64 个数据，那么一个线程需要处理 64 个元素。

stage = 2；

mma op 用的是 arch::OpMultiplyAdd。应该是用 OpMultiplyAdd 来计算，但是没看到在哪里算的。

EpilogueOutputOp = epilogue::thread::LinearCombination。尾声使用 linear 计算。

此外，还有 ThreadblockSwizzle_ = typename threadblock::GemmIdentityThreadblockSwizzle<>参数。这个参数是用于控制 threadblock 是如何启动的，可以用来提高 L2 cache 的命中率。详见/data0/xuwenyuan/workloads/cutlass-4.1/include/cutlass/gemm/threadblock/threadblock_swizzle.h

GemmKernel 就是实例化的 DefaultGemm。

DefaultGemm 又包括 cutlass::gemm::threadblock::DefaultMma。这里固定 arch::Sm50。

然后会根据 LayoutC 选择 Epilogue。如果是 RowmMajor 选择 RegularEpilogue，如果不是就选择 Affine2Epilogue。这两个 Epilogue 都是以 LinearCombination 为 OP 进行实例化的。

DefaultMma 又包括

MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore。用于定义一些计算的基本变量，如 smem 迭代器，一共有多少线程，MN 方向有多少线程，每个线程负责多少元素等。Policy 就是一些 shape 大小，线程 layout 等。

1. IteratorA = cutlass::transform::threadblock::PredicatedTileIterator。
1. IteratorB = cutlass::transform::threadblock::PredicatedTileIterator。
1. ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined。

最终 GemmKernel = kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>，通过 GemmKernel 完成 gemm 的计算。
