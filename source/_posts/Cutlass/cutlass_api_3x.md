---
title: Cutlass 学习笔记（三）Cutlass API 3.x
date: 2025-08-17 12:00:00
tags: [CUTLASS]
categories: [Cutlass 学习笔记,Cutlass]
description: cutlass 3.0 相比较之前的代码主要是多了 collective 和 cute。应该是为了更好的适配 Hopper 等最新的 GPU 框架。 CUTLASS 3.0 具有与以下级别对应的 GEMM API，按从高到低的顺序排列。 1. Device
---

cutlass 3.0 相比较之前的代码主要是多了 collective 和 cute。应该是为了更好的适配 Hopper 等最新的 GPU 框架。

CUTLASS 3.0 具有与以下级别对应的 GEMM API，按从高到低的顺序排列。

1. Device
1. Kernel
1. Collective
1. Tiled MMA and Copy
1. Atom

可以看到与之前的 API 基本一样，从 device 到 kernel，然后多了 collective，这个包括 collective mma 和 collective epilogue。tiledMMA 和 Atom 属于 cute。

本文档将详细介绍前三个级别：device、kernel 和 collective。

## CUTLASS GEMM Model

CUTLASS 实现的算法以分块结构表达了经典的“三重嵌套循环”GEMM 算法，该结构与上述层次结构相对应。

以下伪代码描述了一个针对线程束同步矩阵乘法指令（例如 mma.sync）的 GEMM 内核模型。整个操作被称为“Gemm”，因为假设结尾操作会执行类似于 BLAS 的通用矩阵更新。以下是伪代码，仅用于说明各层中哪些部分对应于 GEMM 的内循环或外循环。

```python
// cutlass::gemm::kernel::GemmUniversal: ClusterTileM and ClusterTileN loops
//   are either rasterized by the hardware or scheduled by the kernel in persistent kernels.
// Parallelism over thread block clusters
for (int cluster_m = 0; cluster_m < GemmM; cluster_m += ClusterTileM) {
  for (int cluster_n = 0; cluster_n < GemmN; cluster_n += ClusterTileN) {

    // cutlass::gemm::collective::CollectiveMma: mainloop that iterates over all k-tiles
    // No loop unrolling is performed at this stage
    for (int k_tile = 0; k_tile < size<2>(gmem_tensor_A); k_tile++) {

      // loops inside cute::gemm(tiled_mma, a, b, c); Dispatch 5: (V,M,K) x (V,N,K) => (V,M,N)
      // TiledMma uses the hardware instruction provided through its Mma_Atom
      // TiledMma's atom layout, value layout, and permutations define the iteration order
      for (int tiled_mma_k = 0; tiled_mma_k < size<2>(A); tiled_mma_k++) {
        for (int tiled_mma_m = 0; tiled_mma_m < size<1>(A); tiled_mma_m++) {
          for (int tiled_mma_n = 0; tiled_mma_n < size<1>(B); tiled_mma_n++) {

            // TiledMma's vector mode dispatches to the underlying instruction.
            mma.call(d, a, b, c);
          } // tiled_mma_n
        } // tiled_mma_m
      } // tiled_mma_k
    } // k_tile mainloop
  } // cluster_m
} // cluster_n
```

前三个嵌套的 for 循环对应于线程块集群上的并行性。代码实际上并没有将它们表示为显式的 for 循环。相反，基于块的并行化方案由 CUDA 网格启动语义隐含。

然而，对于持久性内核 persistent kernels，这三个循环在源代码中表示为单个 while 循环，该循环向工作块调度程序查询要计算的问题块。

在这三个嵌套的 for 循环中，可以找到将矩阵块从全局内存拉取到更“本地”内存（例如共享内存或寄存器）并计算 MMA 的代码。这些块复制和块 mma 迭代通常是完全静态的，并且会完全展开。

## CUTLASS GEMM Components

CUTLASS 使用以下组件来表达上述循环嵌套，这些组件分别用于处理数据类型、布局和数学指令。

![](/assets/cutlass_api_3x/image.png)

在 CUTLASS 3.0 中，我们组装 kernel 的方式是，首先在 kernel 层将 collective mainloop 和 epilogue 组合在一起，然后用主机端 adapter 将它们包装起来，形成指向该内核的 GEMM 句柄。

以下章节将按照用户实例化这些组件以组装内核的顺序进行描述。此顺序为：

1. 组装所需的主循环和尾声，
1. 将它们组合在一起以构建内核类型，然后
1. 用设备层适配器包装内核。

此顺序也体现在 CUTLASS 3.0 Hopper 内核示例中，如下方摘录所示。

```python
// Step 1: Generate the required collective layer mainloop specialization
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TilesShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Step 2: Specify the collective layer epilogue type
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
    ElementC,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>>;

// Step 3: Compose the mainloop and epilogue together at the kernel layer
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>, // ProblemShape [M,N,K,L]
    CollectiveMainloop,
    CollectiveEpilogue
>;

// Step 4: Wrap up the kernel::GemmUniversal kernel class
// with the device adapter to obtain a host-side handle to the kernel
using GemmHandle = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

最后，我们还将简要介绍 CuTe 的平铺 mma 和复制以及原子层 API，然后将用户重定向到 CuTe 专用文档以了解更多详细信息。

## Collective API

Collective 是“平铺 mma 原子和复制原子的最大线程集合”。也就是说，它是网格中能够利用硬件特性进行协作以加速通信和同步的最大线程数量。这些硬件特性包括：

1. 异步数组复制（例如，从全局内存到共享内存）；
1. 用于位于共享内存中的小块的 MMA 指令；
1. 用于集群、线程块和/或 Warp 的同步操作；以及/或
1. 用于确保满足异步操作之间数据依赖关系的硬件加速（例如 Barrier）。

Collective 使用 TiledMma 和 TiledCopy API（见下文）来访问在块上复制和执行 MMA 的操作。

集合体 (Collective) 中的不同并行单元（例如线程、warp 或线程块）可能具有不同的角色。例如，在“warp 专用”算法中，一些 warp 可能负责复制数据，而另一些 warp 可能负责计算。然而，不同的并行单元仍然需要共享数据并协调对共享数据的访问。例如，在 warp 专用算法中，将输入矩阵块复制到共享内存的生产者 warp 需要告知消费者 MMA warp 其 MMA 输入已准备就绪。我们将此与 kernel::layer API 进行对比，后者负责在网格中的独立块上调度集合体。

集合体 API 包含矩阵乘法累加的“主循环”和尾声循环。此 API 是主循环融合和尾声循环融合等优化的组合点。它负责实现上述三重嵌套循环伪代码中的 k_tile 循环。

### Collective Mainloops

`cutlass::gemm::collective::CollectiveMma` 类是集体矩阵乘加 (MMA) 主循环的主要接口。“主循环”指的是遍历图块的“主循环”——即本文档顶部附近伪代码中的“cluster tile k”循环。算法可能需要对多个图块进行的任何循环都将在此处进行。

```python
namespace cutlass::gemm::collective {

template <
  class DispatchPolicy,
  class TileShape,
  class ElementA,
  class StrideA,
  class ElementB,
  class StrideB,
  class TiledMma,
  class GmemTiledCopyA,
  class SmemLayoutAtomA,
  class SmemCopyAtomA,
  class TransformA,
  class GmemTiledCopyB,
  class SmemLayoutAtomB,
  class SmemCopyAtomB,
  class TransformB
>
struct CollectiveMma {
  static_assert(sizeof(ElementA) == 0, "Could not find a mainloop specialization.");
};

} // namespace cutlass::gemm::collective
```

1. DispatchPolicy 是集合体最重要的类型，下文将详细介绍。、
1. StrideA 和 StrideB 是 cute::Stride 类型的实例，表示 A 和 B 张量的全局内存布局。这些步幅必须是 3 阶，分别代表 [外部、内部、批量] 模式。这 3 个阶中的每一个都可以是多模态分层步幅；这在实现张量收缩时适用。
1. TiledMma 是 cute::TiledMma 的实例。
1. GmemTiledCopyA 和 GmemTiledCopyB 是 cute::TiledCopy 类型的实例。这两种平铺操作类型都将在下文详细介绍。
1. SmemLayoutAtomA 和 SmemLayoutAtomB 是 cute::Layout 类型的实例，表示将在整个集合体的共享内存上平铺的最小布局。此布局不包含流水线模式，因此两者预计均为 2 阶布局，形状为 [outer, inner]。
1. SmemCopyAtomA 和 SmemCopyAtomB 是 Copy_Atom，用于将数据从共享内存移动到寄存器内存。

请注意，CUTLASS 3.0 主循环不接受专用的累加器元素类型。我们从类型名 TiledMma::ValTypeC 获取累加器类型。另请注意，顶级 API 的 ElementA 和 ElementB 可能与面向 MMA 的类型名 TiledMma::ValTypeA 和类型名 TiledMma::ValTypeB 的 ElementA 和 ElementB 不同，从而允许 TMA 或用户提供的转换操作执行类型转换。

## Collective Dispatch Policies

CollectiveMma 实现并非通用的，而是必须针对每种算法和 GPU 架构进行专门化。用户可以通过选择与 CollectiveMma 专门化匹配的模板参数来调度该专门化。CUTLASS 3.0 采用基于标签的调度策略类型来专门化主循环实现并为其添加调整旋钮。

以下是用于调度 Hopper TMA warp-specialized 主循环实现的调度策略示例之一：

```python
// n-buffer in smem (Hopper TMA),
// pipelined with Hopper GMMA and TMA,
// warp-specialized dynamic schedule
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecializedCooperative
>
struct MainloopSm90TmaGmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
};
```

Stages_ 模板参数允许用户自由更改流水线阶段的数量，而 ClusterShape_ 类型则允许对将进行 TMA 多播的线程块集群的形状进行参数化。

集体调度策略也是在任何主循环中自由组合各种内核调度的关键所在。每个主循环策略要么规定一个需要运行的调度，要么公开一个模板 API，让用户从以下调度中选择子集：

```python
struct KernelCpAsyncWarpSpecialized { };
struct KernelCpAsyncWarpSpecializedPingpong { };
struct KernelCpAsyncWarpSpecializedCooperative { };
struct KernelTma { };
struct KernelTmaWarpSpecialized { };
struct KernelTmaWarpSpecializedPingpong { };
struct KernelTmaWarpSpecializedCooperative { };
```

1. 单个内核调度可以支持多种主循环实现。例如，KernelMultistage 可以与跨 GPU 架构的多种不同主循环实现组合，例如 MainloopSm70TwoStage、MainloopSm80CpAsyncUnpredicated 等等。
1. 单个主循环可以由多个可能的内核调度组合而成。例如，MainloopSm90TmaGmmaWarpSpecialized 可以与 KernelTmaWarpSpecialized、KernelTmaWarpSpecializedPingpong 或 KernelTmaWarpSpecializedCooperative 等任意内核调度组合而成。

正如 CUTLASS 3.0 设计文档中所讨论的，采用核心词汇表类型的标签调度策略使我们能够为概念上属于同一类的所有操作维护一个类型名称。这种设计具有以下优势：

1. 当主循环可以由多个内核组合时，或者反之亦然，它可以避免代码重复。
1. 它使编写通用代码变得更容易，因为主类型名称 CollectiveMma 在任何实现中都不会改变。
1. 它为用户提供了一个清晰、单一的扩展点，可以插入专门针对他们自己的调度策略的新的、自定义的主循环实现。

## Collective Builder for CollectiveMmas

CollectiveMma 的主要目标是成为一个专家用户界面，允许完全控制 Collective GPU 微内核的所有属性。然而，用户通常只需要一个现成的 GEMM 主循环实现，该实现基于简单的配置参数进行参数化。CUTLASS 3.0 为此类场景提供了 cutlass::gemm::collective::CollectiveBuilder。

```python
namespace cutlass::gemm::collective {
template <
  class ArchTag,
  class OpClass,
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType,
  class Enable = void
>
struct CollectiveBuilder {
  static_assert(sizeof(ElementA) == 0, "Could not build a collective for given parameters.");
};
} // namespace cutlass::gemm::collective
```

CollectiveBuilder 接受与 CUTLASS 2.x 等效的输入模板参数，并尝试根据给定参数构建性能最佳的 CollectiveMma。

1. ArchTag 是 cutlass::arch::Sm* 中的 SM 架构标签之一。
1. OpClass 是 cutlass::arch::OpClass* 中的运算符类标签之一。
1. ElementA 和 ElementB 分别是 A 和 B 张量的逻辑值类型。
1. ElementAccumulator 是指令中使用的累加器类型。
1. GmemLayoutA 和 GmemLayoutB 是 CUTLASS 2.x 布局标签，layout::RowMajor 或 layout::ColumnMajor。
1. AlignmentA 和 AlignmentB 是 A 和 B 张量的全局内存对齐（以元素数量为单位）。
1. TileShape_MNK 是 cute::Shape 的一个 3 阶实例，表示 MxNxK 的集合图块形状。
1. ClusterShape_MNK 是 cute::Shape 的一个 3 阶实例，表示 MxNxK 线程块集群块形状。
1. StageCountType 可以是 collective::StageCountAuto 或 collective::StageCount<N> 的一个实例。
1. KernelScheduleType 可以是 collective::KernelScheduleAuto 或上文调度策略部分讨论过的特定内核调度标记之一。

StageCountAuto 允许集合构建器计算单个阶段在共享内存中的大小，并在假设 1 个线程块/多处理器占用的情况下最大化共享内存使用率。

KernelScheduleAuto 允许集合构建器根据给定的一组参数选择最佳的内核调度，或者允许用户使用特定的内核调度类型覆盖此调度。

请注意，集合构建器仍处于测试阶段，其功能尚未映射到主要专家 CollectiveMma API 所允许的完整设计空间。我们预计其支持的主循环类型将在未来版本中扩展，但在 3.0 版本中，通过构建器 API 仅支持 SM90 TensorOp 内核。随着我们采纳用户反馈，构建器 API 未来也可能会有所变化。

如果构建器能够为给定的一组参数提供集合主循环类型，它将在内部被别名为 CollectiveOp。有关如何使用集合构建器便捷地参数化内核的更多信息，请参阅示例 49_hopper_gemm_with_collective_builder。

## Epilogue

集体尾声实现涉及输出矩阵的元素级运算。用户可以提供自定义尾声，或使用标准尾声之一。这些尾声位于目录 include/cutlass/epilogue/collective/ 中，并包含诸如 cutlass::epilogue::collective::DefaultEpilogue 和 cutlass::epilogue::collective::Epilogue 之类的类。CUTLASS 提供的集体尾声不位于 include/cutlass/gemm 或 cutlass::gemm 命名空间中，因为它们可用于 GEMM 以外的计算。

## Kernel API

内核是“网格中所有集群的集合”。内核层调度器主要负责以下四项任务：

1. 对内核中集群的执行进行排序，并执行任何必要的同步操作；
1. 将 Warp 专用调度器的线程编组到各自的角色中；
1. 执行任何必要的网格混合逻辑；
1. 在对输入张量调用集群之前，使用线程块集群值对输入张量进行平铺；

内核 API 是线程块网格的入口点，这些线程块可能组织在集群中，也可能不在集群中。它是融合连续 GEMM、尾声和/或其他操作的组合点。

CUTLASS 3.0 内核的入口 API 是 cutlass::gemm::kernel::GemmUniversal 类，位于头文件 include/cutlass/gemm/kernel/gemm_universal.hpp 中。 GemmUniversal 是一个无状态通用设备内核，它将 GEMM 实现为两部分：

1. 一个集体主循环，以及
1. 一个集体尾声

```python
namespace cutlass::gemm::kernel {
/*
 * Stateless universal device GEMM kernel type that treats GEMM as
 * a composition of a collective mainloop and a collective epilogue.
 *
 * Supports both the 2.x and 3.x APIs based on whether the first type is
 * a cute::tuple<> or not.
 * 2.x API implementation: cutlass/gemm/kernel/gemm_universal.h
 * 3.x API implementation: cutlass/gemm/kernel/gemm_*.hpp
 *
 * In the following declaration, the name preceding the 'Or' refers to
 * 3.x API type argument order, and the name succeeding the 'Or' refers to
 * 2.x API type argument order. Template arguments without two names
 * belong to the 3.x API only.
**/
template <
  class ProblemShapeOrThreadblockMma_, // (m, n, k) or (m, n, k, l)
  class CollectiveMainloopOrEpilogue_,
  class CollectiveEpilogueOrThreadblockSwizzle_,
  class TileScheduler_ = void,
  class Enable = void
>
class GemmUniversal;
} // namespace cutlass::gemm::kernel
```

无状态意味着调用者（例如，上面描述的设备 API）管理内核的状态。内核只接受输入和输出参数 (Params)。

通用意味着 GemmUniversal 适用于 CUTLASS 3.0 和 2.x 接口，并支持各种内核调度。如果 GemmUniversal 的第一个模板参数是 cute::Shape，则 GemmUniversal 假定其余模板参数实现了 3.0 API。否则，GemmUniversal 假定其余模板参数实现了 2.x API。从 CUTLASS 3.0 开始，问题形状已提升为 GEMM 内核的顶级模板 API。这支持完全静态的 GEMM 实例，用户希望在编译时了解部分或全部问题形状，以提取更高的性能。

集体主循环在局部块上实现 MMA。集体结语 (collective epilogue) 处理 MMA 之后的任何操作，例如应用 C := beta * C + alpha * A * B 中的 beta * C 部分。我们将在下文更详细地解释集体结语。

kernel::GemmUniversal 3.0 API 的特化版本位于 include/cutlass/gemm/kernel/ 目录中的各种 gemm_*.hpp 文件中。2.x API 的特化版本位于头文件 include/cutlass/gemm/kernel/gemm_universal.h 中。

CUTLASS 3.x 实现了 kernel::GemmUniversal 的各种实现。每个内核层调度都针对一种 GEMM 调度算法和 GPU 架构进行特化。kernel::GemmUniversal 3.0 API 的特化版本位于 include/cutlass/gemm/kernel/ 目录中的各种 include/cutlass/gemm/kernel/{arch_tag}*.hpp 文件中。通过调度策略的 Schedule 类型来决定调度到哪个特化版本。

例如，头文件 include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp 包含针对 Hopper 的 kernel::GemmUniversal 特化版本，它使用带有持久调度算法的 Warp 特化主循环；而头文件 include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp 包含针对 Hopper 的 GemmUniversal 特化版本，它使用 Warp 特化但非持久的算法。

为了支持内核调度和主循环调度策略之间的组合，而无需重复集体主循环实现，GEMM 内核层调度可以与任何在策略中指定其对应内核调度作为其 Schedule 类型的主循环组合。这在上面“集体调度策略”部分中有详细讨论。

```python
// An example of the SM90 KernelMultistage kernel's
// specialization logic that allows it to be composed
// with many mainloops such as `MainloopSm80CpAsync`
// and `MainloopSm70TwoStage`.
template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  std::enable_if_t<std::is_base_of_v<KernelMultistage, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
```

## Device API

设备 API 是一个通用的、与内核无关的主机接口，用于启动内核并管理可重用主机端参数的生命周期。

此 API 是用户主机端 .cu 代码调用 CUTLASS 单 GPU GEMM 内核的方式。它与 cuBLAS 的用途相同，行为也类似。

设备 GEMM API 的入口点是 cutlass::gemm::device::GemmUniversalAdapter 类。该类位于头文件 include/cutlass/gemm/device/gemm_universal_adapter.h 中。GemmUniversalAdapter 是一个有状态的可重用句柄，其参数化基于 cutlass::gemm::kernel 类型。

```python
/*! 
  GemmUniversalAdapter is a stateful, reusable GEMM handle built around a kernel
  of type cutlass::gemm::kernel::*

  It manages the lifetime of the underlying `kernel::Params` struct, and exposes APIs
  to create it from the host facing arguments. For power users, new static methods
  are exposed in 3.x APIs that bypass the stateful methods or args->params lowering.

  It supports kernel types that implement both the 2.x and 3.0 APIs,
  however, this is done by specializing the implementation of GemmUniversalAdapter
  on the two kernel API types, and thus, GemmUniversalAdapter's behavior might
  differ between the two specializations.
*/
template <class GemmKernel_, class Enable = void>
class GemmUniversalAdapter;
```

有状态意味着句柄实例包含内核运行所需的状态。这意味着用户必须先初始化句柄，然后使用初始化的句柄实例来运行内核。有状态还意味着句柄可以管理内核参数（内核本身的参数）的生命周期。GemmUniversalAdapter 的一项重要职责是将用户的参数（用户认为是内核的参数）映射到内核实际看到的参数。对于高级用户，该类在 3.0 API 中公开了新的静态方法，这些方法可以绕过有状态方法或直接转到参数，而无需中间参数。

可重用意味着句柄实例可用于使用不同的参数（例如，不同的矩阵）多次调用内核。重用句柄可能比仅为每次内核调用创建一个新的句柄更高效。

基于内核类型进行参数化意味着 GemmUniversalAdapter 类的行为取决于 GEMM 内核类型（请参阅下一节）。具体来说，GemmUniversalAdapter 有一个模板参数 GemmKernel，它是 GEMM 内核类型。GemmKernel 的有效模板参数包括：

1. cutlass::gemm::kernel::GemmUniversal（实现 CUTLASS 3.x API 内核）；
1. cutlass::gemm::kernel::GemmUniversal（实现 CUTLASS 2.x API 内核）；或

任何之前可与 device::GemmUniversalAdapter 组合的有效 CUTLASS 2.x 内核层 GEMM。

GemmUniversalAdapter 为 3.0 和 2.x 内核提供单一主机端接口。CUTLASS 通过在实现内核层 GEMM 的 2.x API 或实现内核层 GEMM 的 3.x API 上特化 GemmUniversalAdapter 的实现来实现这一点。 GemmUniversalAdapter 使用元函数 cutlass::gemm::detail::IsCutlass3GemmKernel 来区分 2.x 和 3.x 内核。

GemmUniversalAdapter 设置并启动内核，并在需要时使用 CUDA 扩展启动 API 来支持线程块集群。注意，GemmUniversalAdapter 不指定网格形状。内核控制网格形状和其他特定于内核的启动参数。这使得所有 3.0 内核都可以使用相同的内核启动代码，从而将内核启动从实际内核中分离出来。

## Tiled MMA and Copy

Tiled MMA 或 Copy 分别是 MMA 原子的平铺。跨线程和跨数据复制原子，并对生成的平铺应用可能的排列。此层与 CUTLASS 2.x 中 MMA 指令的 Warp 级平铺最为相似。然而，它从参与操作的所有线程的角度来看待平铺，并将此概念推广到复制操作。此层的目的是基于大量硬件加速的数学和数据移动操作构建可组合的 GPU 微内核，每个操作在线程和数据中都有其单元布局。Tiled MMA 和 Copy 类型通过单一、一致的 API 呈现所有这些不同的硬件加速 CuTe 原子。

生成的平铺操作充当单个 MMA 或复制操作，用户可以在本文档顶部三层嵌套循环伪代码的“内部”循环中使用 cute::gemm() 或 cute::copy() 调用它。

我们将此 API 称为“tiled”，因为它利用 CuTe 提供的 Atom 构建更大的操作，就像将各个 Tile 拼凑在一起，构建出马赛克的可重用组件一样。例如，CuTe 可能提供一个 MMA Atom，用户可以在单个 Warp 上调用，用于固定的 M、N 和 K 维度。然后，CUTLASS 可以使用类似 make_tiled_mma 的 CuTe 操作将此 Atom 转换为可作用于整个线程块的操作，用于更大的 M、N 和 K 维度。

Atom API

“Atom” 是必须参与执行硬件加速数学运算或复制操作的最小线程和数据集合。

Atom 的“原子性”（不可分割性）并非指像 atomicAdd 这样的并发内存操作（它们“在时间（因果关系）上不可分割”），而是指“空间”上的不可分割性——值的数量和必须同时参与操作的并行工作器组的数量。

Atom 使用 CuTe 布局来表示其输入和输出数组所需的维度和步长。通常，这些在编译时就已确定。

Atom API 封装了对加速 MMA 或复制操作的实际硬件指令的调用。用户可以请求特定于 GPU 架构的实现，也可以选择通用实现并依赖已启用的任何 GPU 架构。
