---
title: Cutlass 学习笔记（二）Cutlass API
date: 2025-08-16 12:00:00
tags: [CUTLASS]
categories: [Cutlass 学习笔记,Cutlass]
description: 梳理 CUTLASS 2.x 的分层 GEMM 编程模型，涵盖 device、threadblock、warp、thread 与指令级 GEMM 的接口、概念与代码示例，并结合 SGEMM 例子追一遍 DefaultGemm 的实例化过程。
---

CUTLASS 为层次结构中每一级的矩阵乘累加运算提供统一的编程模型，分为设备级（device-level）、线程块级（threadblock-level）、warp 级、thread 级和指令级 GEMM。

![](/assets/cutlass_api/image.png)

## CUTLASS GEMM 模型

CUTLASS 把经典的三重循环 GEMM 用分块结构实现，与上面的执行模型层级对应。下面的伪代码描述了一个面向 mma.sync 这类 warp 同步矩阵乘指令的 GEMM kernel 模型。整个操作称为 GEMM，因为假设尾声会执行类似 BLAS 的通用矩阵更新。

```c++
                                                                            // cutlass::gemm::device::Gemm
                                                                            //
for (int cta_n = 0; cta_n < GemmN; cta_n += CtaTileN) {                     // for each CTA       } CTA-level concurrency
  for (int cta_m = 0; cta_m < GemmM; cta_m += CtaTileM) {                   //    for each CTA    }
                                                                            //    
                                                                            // cutlass::gemm::threadblock::Mma
                                                                            //
    for (int cta_k = 0; cta_k < GemmK; cta_k += CtaTileK) {                 //       "GEMM mainloop" - no unrolling - one iteration of this loop is one "stage"
                                                                            //
      for (int warp_n = 0; warp_n < CtaTileN; warp_n += WarpTileN) {        // for each warp      } warp-level concurrency
        for (int warp_m = 0; warp_m < CtaTileM; warp_m += WarpTileM) {      //    for each warp   }
                                                                            //
          for (int warp_k = 0; warp_k < CtaTileK; warp_k += WarpTileK) {    //       fully unroll across CtaTileK - one iteration of this loop is one "k Group"
                                                                            //
            for (int mma_k = 0; mma_k < WarpTileK; mma_k += MmaK) {         // cutlass::gemm::warp::Mma
              for (int mma_n = 0; mma_n < WarpTileN; mma_n += MmaN) {       //
                for (int mma_m = 0; mma_m < WarpTileM; mma_m += MmaM) {     //
                                                                            //
                  mma_instruction(d, a, b, c);                              // cutlass::arch::mma - warp-wide matrix multiply instruction

                }   // for mma_m
              }   // for mma_n
            }   // for mma_k

          }   // for warp_k
        }   // for warp_m
      }   // for warp_n

    }   // for cta_k
  }   // for cta_m
}   // for cta_n
```

最外层的两个循环对应 CTA 级硬件并发，代码里并不会显式写成循环，而是由 CUDA 的 grid 启动语义隐含。

注释 cutlass::gemm::threadblock::Mma 指的线程块级矩阵乘累加概念，即一个线程块在寄存器中计算出矩阵乘积所执行的计算，这里列出的是 GEMM mainloop。注释 cutlass::gemm::warp::Mma 指每个 warp 执行的计算，是一串累加的外积。最内层直接对应硬件支持，在这个例子里终止于面向 Tensor Core 的 warp 同步矩阵乘指令。如果 GEMM 面向单线程指令，还会多出一层对应 thread 级并发的嵌套循环。

## CUTLASS GEMM 组件

上述循环嵌套由以下组件表达，它们分别针对数据类型、布局和数学指令进行特化。

### Device-wide GEMM API

设备级 GEMM API 简化了 GPU 上标准 GEMM 计算的实例化与执行，用于主机端 .cu 代码，语义与 cuBLAS 类似。它由以下算子构成。

- cutlass::gemm::device::Gemm，基本的 GEMM 运算
- cutlass::gemm::device::GemmArray，输入矩阵从指针数组中读取的批处理 GEMM
- cutlass::gemm::device::GemmBatched，输入矩阵之间相隔固定步长的批处理 GEMM
- cutlass::gemm::device::GemmSplitKParallel，对 K 维度分区后另行启动归约 kernel 的 GEMM

示例，面向 Volta Tensor Core 的混精 GEMM。

```c++
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // ElementA
    cutlass::layout::ColumnMajor,              // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    cutlass::half_t,                           // ElementOutput
    cutlass::layout::ColumnMajor,              // LayoutOutput
    float,                                     // ElementAccumulator
    cutlass::arch::OpClassTensorOp,            // tag indicating Tensor Cores
    cutlass::arch::Sm70                        // tag indicating target GPU compute architecture
  >;

  Gemm gemm_op;
  cutlass::Status status;
 
  //
  // Launch GEMM on the device
  //
 
  status = gemm_op({
    {m, n, k},
    {ptrA, lda},
    {ptrB, ldb},
    {ptrC, ldc},
    {ptrD, ldd},
    {alpha, beta}
  });

  if (status != cutlass::Status::kSuccess) {
    return -1;
  }
```

### Threadblock-level GEMM API

线程块级 GEMM 负责把数据块从全局内存高效地加载到内部存储，再用 warp 级 GEMM 算子计算矩阵乘积。线程块级的矩阵乘累加由 cutlass::gemm::threadblock::MmaPipelined 实现，这个类受 std::transform_reduce() 启发，计算由 tile 迭代器定义的一系列 tile 的累积矩阵乘积。

在 GEMM 中，tile 迭代器是 cutlass::transform::threadblock::PredicatedTileIterator，它在全局内存中遍历一系列 tile，并带有合适的谓词以避免越界访问。

概念上，线程块级矩阵乘累加是一个满足下述概念的函数对象。

```c++
struct Mma {
  /// Shape of warp-level matrix operation (concept: GemmShape)
  struct Shape;

  /// Data type of multiplicand A (concept: numeric type)
  struct ElementA;

  /// Layout of multiplicand A (concept: Layout)
  struct LayoutA;

  /// Data type of multiplicand B (concept: numeric type)
  struct ElementB;

  /// Layout of multiplicand B (concept: Layout)
  struct LayoutB;

  /// Data type of accumulator matrix C (concept: numeric type)
  struct ElementC;

  /// Layout of accumulator matrix C (concept: Layout)
  struct LayoutC;

  /// Iterator of A operand in shared memory - satisfies: ReadableRandomAccessTileIteratorConcept
  struct IteratorA;

  /// Fragment object loaded from IteratorA (concept: Array<ElementA, ..>)
  struct FragmentA;

  /// Iterator of B operand in shared memory - satisfies: ReadableRandomAccessTileIteratorConcept
  struct IteratorB;

  /// Fragment object loaded from IteratorB (concept: Array<ElementB, ..>)
  struct FragmentB;

  /// Iterator of C operand in shared memory - 
  ///    satisfies: ReadableRandomAccessTileIteratorConcept | WriteableRandomAccessTileIteratorConcept
  struct IteratorC;

  /// Fragment object loaded from IteratorC (concept: Array<ElementC, ..>)
  struct FragmentC;

  /// Warp-level matrix multiply operator (concept: satisfies gemm::warp::Mma)
  struct Operator;

  //
  // Method
  //

  /// Computes a matrix product accumulated in D
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D, 
    IteratorA iter_A, 
    IteratorB iter_B, 
    FragmentC const &C);
};
```

### Warp-level Matrix Multiply API

warp 级 GEMM 算子把 tile 从共享内存加载到寄存器，再用 Tensor Core 或 CUDA Core 计算矩阵乘法，结果累加在寄存器 tile 中。操作数 A、B、C 各自定义了迭代器。

warp 级 GEMM API 是 CUDA WMMA API 的泛化，目标是支持 Tensor Core 的原生矩阵乘尺寸、置换共享内存布局以保证无冲突访问、在主循环之外初始化指针以及高效遍历。

在 CUTLASS 中定义 warp 级矩阵乘与 WMMA 类似，使用方式也相近。下面的例子计算一个 warp 级 GEMM，把一系列矩阵乘积累加到寄存器数组中。CUTLASS 的 warp 级 GEMM 输入必须是共享内存中由迭代器加载的数据，或寄存器中的片段。

```c++
#include "cutlass/gemm/warp/default_mma_tensor_op.h"

using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
    cutlass::sizeof_bits<Element>::value, 64>;

using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
    cutlass::sizeof_bits<Element>::value, 64>;

using WarpMma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    cutlass::gemm::GemmShape<64, 64, 8>,                            // Overall warp-level GEMM operation
    cutlass::gemm::GemmShape<16, 8, 8>,                             // Target instruction
    cutlass::half_t, LayoutA,                                       // operand A type and layout
    cutlass::half_t, LayoutB,                                       // operand B type and layout
    float,                                                          // accumulator type
    cutlass::layout::RowMajor>::Type;                               // accumulator layout

//
// Define a GEMM operation loading data from shared memory
//
int const kGemmK = 32;

__shared__ ElementA smem_buffer_A[WarpMma::Shape::kM * kGemmK];
__shared__ ElementB smem_buffer_B[WarpMma::Shape::kN * kGemmK];

//
// Construct iterators into SMEM tiles
//

// leading dimensions inferred from matrix problem size
int lda = WarpMma::Shape::kM;
int ldb = WarpMma::Shape::kN;

// iterators into shared memory
WarpMma::IteratorA warp_iterator_A({smem_buffer_A, lda});
WarpMma::IteratorB warp_iterator_B({smem_buffer_B, ldb});

// Fragments in registers storing the operands
FragmentA frag_A;
FragmentB frag_B;
FragmentC accum;

WarpMma mma;

accum.clear();

//
// Accumulated outer product
//

#pragma unroll 1
for (int k = 0; k < kGemmK; k += WarpMma::Shape::kK) {

  
  iter_A.load(frag_A);  // Load fragments from A and B matrices
  iter_B.load(frag_B);

  ++iter_A; ++iter_B;   // Advance along GEMM K to next tile in A
                        //   and B matrices

                        // Compute matrix product
  mma(accum, frag_A, frag_B, accum);
}
```

概念上，warp 级 Mma 是满足下述概念的函数对象。

```c++
struct Mma {
  /// Shape of warp-level matrix operation (concept: GemmShape)
  struct Shape;

  /// Data type of multiplicand A (concept: numeric type)
  struct ElementA;

  /// Layout of multiplicand A (concept: Layout)
  struct LayoutA;

  /// Data type of multiplicand B (concept: numeric type)
  struct ElementB;

  /// Layout of multiplicand B (concept: Layout)
  struct LayoutB;

  /// Data type of accumulator matrix C (concept: numeric type)
  struct ElementC;

  /// Layout of accumulator matrix C (concept: Layout)
  struct LayoutC;

  /// Iterator of A operand in shared memory - satisfies: ReadableRandomAccessTileIteratorConcept
  struct IteratorA;

  /// Fragment object loaded from IteratorA (concept: Array<ElementA, ..>)
  struct FragmentA;

  /// Iterator of B operand in shared memory - satisfies: ReadableRandomAccessTileIteratorConcept
  struct IteratorB;

  /// Fragment object loaded from IteratorB (concept: Array<ElementB, ..>)
  struct FragmentB;

  /// Iterator of C operand in shared memory - 
  ///     satisfies: ReadableRandomAccessTileIteratorConcept | WriteableRandomAccessTileIteratorConcept
  struct IteratorC;

  /// Fragment object loaded from IteratorC (concept: Array<ElementC, ..>)
  struct FragmentC;

  /// Indicates class of matrix operator (arch::OpClassSimt or arch::OpClassTensorOp)
  struct OperatorClass;

  //
  // Methods
  //

  /// Computes a matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D, 
    IteratorA A, 
    IteratorB B, 
    FragmentC const &C);
};
```

面向 Tensor Core 的 warp 级矩阵乘算子和面向 CUDA Core 的 SIMT 版本分别用下面的模板参数定义，其中 Policy 指定可能影响性能或内部实现的细节。

```c++
namespace cutlass {
namespace gemm {
namespace warp {

/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename ElementA_,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA_,
  /// Data type of B elements
  typename ElementB_,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB_,
  /// Element type of C matrix
  typename ElementC_,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_,
  /// Used for partial specialization
  typename Enable = bool
>
class MmaTensorOp {}

} // namespace warp
} // namespace gemm
} // namespace cutlass
```

```c++
/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename ElementA_,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA_,
  /// Data type of B elements
  typename ElementB_,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB_,
  /// Element type of C matrix
  typename ElementC_,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_,
  /// Used for partial specialization
  typename Enable = bool
>
class MmaSimt;
```

### Thread-level GEMM API

thread 级 GEMM 运算对寄存器中的数据进行矩阵乘累加，只面向 CUDA Core。概念上它是满足下述概念的函数对象。

```c++
struct Mma {

  /// Shape of warp-level matrix operation (concept: GemmShape)
  struct Shape;

  /// Data type of multiplicand A (concept: numeric type)
  struct ElementA;

  /// Layout of multiplicand A (concept: Layout)
  struct LayoutA;

  /// Fragment object loaded from IteratorA (concept: Array<ElementA, ..>)
  struct FragmentA;

  /// Data type of multiplicand B (concept: numeric type)
  struct ElementB;

  /// Layout of multiplicand B (concept: Layout)
  struct LayoutB;

  /// Fragment object loaded from IteratorA (concept: Array<ElementB, ..>)
  struct FragmentB;

  /// Data type of accumulator matrix C (concept: numeric type)
  struct ElementC;

  /// Layout of accumulator matrix C (concept: Layout)
  struct LayoutC;

  /// Fragment object loaded from IteratorA (concept: Array<ElementC, ..>)
  struct FragmentC;

  //
  // Methods
  //

  /// Computes a matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D, 
    FragmentA const &A, 
    FragmentB const &B, 
    FragmentC const &C);
};
```

对应的 thread 级 GEMM 模板接受以下参数。

```c++
namespace cutlass {
namespace gemm {
namespace thread {

/// Structure to compute the matrix product
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape,
  /// Data type of A elements
  typename ElementA,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA,
  /// Data type of B elements
  typename ElementB,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB,
  /// Element type of C matrix
  typename ElementC,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC,
  /// Concept: arch::OpMultiplyAdd or arch::Mma<>
  typename Operator = arch::OpMultiplyAdd,
  /// Used for partial specialization
  typename Enable = bool
>
struct Mma;

} // namespace thread
} // namespace gemm
} // namespace cutlass
```

### Efficient Epilogue

CUTLASS GEMM 算子先做 mma，再做类似 cuBLAS 的尾声操作。CUTLASS 实现的是高效的行主序尾声，因此要实现列主序 GEMM，需要把操作数 A、B 转置并交换。

为了让行主序和列主序输出布局都能用上高效的行主序尾声，设备级 GEMM 算子 cutlass::device::Gemm 与 cutlass::device::GemmUniversal 提供两种模板定义。

- (a) 通用定义
- (b) 面向列主序源/输出的特化定义

高效行主序尾声的适用情况如下。

- (i) 行主序源/输出的 GEMM 使用模板 (a)，运行行主序 GEMM 和高效的行主序尾声。
- (ii) 列主序源/输出的 GEMM 使用模板 (b)，它把操作数 A、B 转置并交换以实现高效尾声。A x B = C => Transpose(B) x Transpose(A) = Transpose(C)。对列主序源矩阵 C，Transpose(C) 是行主序，高效尾声就基于行主序完成。

cuBLAS 通常要求列主序的源矩阵 C 和输出矩阵 D，因此 CUTLASS 库只实例化并生成列主序布局的 GEMM 算子。不过 CUTLASS 本身可以对所有输入布局组合运行行主序和列主序输出，因此支持如下输入输出布局组合。

- {N,T} x {N,T} => {N,T}，即 NN、TN、TN、TT GEMM，行主序和列主序输出都适用

### 指令级算子

CUTLASS 为 Tensor Core 操作定义了基于模板的接口，避免直接写内联 PTX。

- mma_sm70.h，Volta Tensor Core 操作
- mma_sm75.h，Turing Tensor Core 操作

## 从一个 SGEMM 例子看 kernel 实例化

下面用一个行主序 SGEMM 的例子，顺着 device 到 kernel 的路径追一遍实例化过程。

```c++
cudaError_t CutlassSgemmNN(int M, int N, int K, float alpha, float const *A, int lda, float const *B, int ldb, float beta, float *C, int ldc)
{
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using RowMajor = cutlass::layout::RowMajor;
    using OpClass = cutlass::arch::OpClassSimt;
    using Arch = cutlass::arch::Sm70;
    using CutlassGemm = cutlass::gemm::device::Gemm<float, ColumnMajor, float, ColumnMajor, float, RowMajor, float, OpClass, Arch>;

    CutlassGemm gemm_op;

    CutlassGemm::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});

    cutlass::Status status = gemm_op(args);

    if (status != cutlass::Status::kSuccess)
    {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}
```

cutlass::gemm::device::Gemm 是矩阵乘的入口，包含 GemmKernel、Arguments 以及启动相关的函数。根据 arch、stage 等模板参数的不同，会实例化不同的 GemmKernel。

这里 OpClass = cutlass::arch::OpClassSimt、Arch = cutlass::arch::Sm70，于是实例化出下面的 DefaultGemmConfiguration。

```c++
template <
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
  ElementAccumulator> {
  
  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  using ThreadblockShape = GemmShape<128, 128, 8>;
  using WarpShape = GemmShape<32, 64, 8>;
  using InstructionShape = GemmShape<1, 1, 1>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    1,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Operator = arch::OpMultiplyAdd;
};
```

其中 ThreadblockShape 的 MNK 为 <128, 128, 8>，即一个 threadblock 处理 128×128×8 的数据。WarpShape 为 <32, 64, 8>，一个 warp 处理 32×64×8 的数据，因此一个 threadblock 需要 8 个 warp。InstructionShape 为 <1, 1, 1>，对应单条 SIMT 乘加指令；一个 warp 处理 32×64 个元素时，每个线程需要处理 64 个元素。kStages 为 2，即双缓冲流水线。mma 算子使用 arch::OpMultiplyAdd，尾声使用 epilogue::thread::LinearCombination 做线性组合。

另外还有一个 ThreadblockSwizzle_ = typename threadblock::GemmIdentityThreadblockSwizzle<> 参数，用来控制 threadblock 的启动方式，以提高 L2 cache 命中率，定义在 cutlass/gemm/threadblock/threadblock_swizzle.h。

GemmKernel 就是实例化出来的 DefaultGemm。DefaultGemm 内部包含 cutlass::gemm::threadblock::DefaultMma，这里固定了 arch::Sm50。随后根据 LayoutC 选择尾声，行主序选择 RegularEpilogue，否则选择 Affine2Epilogue，两者都以 LinearCombination 作为 op 实例化。

DefaultMma 又包含下面几部分。

- MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore，用于定义计算所需的基本量，例如 smem 迭代器、总线程数、MN 方向的线程数、每个线程负责的元素数，其中 Policy 就是 shape 大小和线程布局等。
- IteratorA = cutlass::transform::threadblock::PredicatedTileIterator
- IteratorB = cutlass::transform::threadblock::PredicatedTileIterator
- ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined

最终 GemmKernel = kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>，由 GemmKernel 完成 gemm 的计算。
