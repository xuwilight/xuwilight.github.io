---
title: Cutlass 学习笔记（三）Cutlass API 3.x
date: 2025-08-17 12:00:00
tags: [CUTLASS]
categories: [Cutlass 学习笔记,Cutlass]
description: 梳理 CUTLASS 3.0 的分层 GEMM 接口，涵盖 Device、Kernel、Collective 三层的类型、职责与组合方式，并简要介绍 Tiled MMA/Copy 与 Atom 两层。
---

CUTLASS 3.0 相比之前最大的变化是多了 collective 和 CuTe，目的是更好地适配 Hopper 等新架构。它按从高到低的顺序提供以下 GEMM API。

1. Device
2. Kernel
3. Collective
4. Tiled MMA and Copy
5. Atom

这与之前的 API 基本一致，从 device 到 kernel 之外多了 collective，它包含 collective mma 和 collective epilogue，而 tiledMMA 与 Atom 属于 CuTe。本文详细介绍前三个层级，即 device、kernel 和 collective。

## CUTLASS GEMM 模型

CUTLASS 实现的算法用与上述层级对应的分块结构表达经典的三重循环 GEMM。下面的伪代码描述一个面向 mma.sync 这类 warp 同步矩阵乘指令的 GEMM kernel 模型。整个操作称为 Gemm，因为假设尾声会执行类似 BLAS 的通用矩阵更新。这段伪代码只用于说明各层对应 GEMM 的内层还是外层循环。

```c++
// cutlass::gemm::kernel::GemmUniversal: ClusterTileM and ClusterTileN loops
//   are either rasterized by the hardware or scheduled by the kernel in persistent kernels.
// Parallelism over thread block clusters
for (int cluster_m = 0; cluster_m < GemmM; cluster_m += ClusterTileM) {
  for (int cluster_n = 0; cluster_n < GemmN; cluster_n += ClusterTileN) {

    // cutlass::gemm::collective::CollectiveMma: mainloop that iterates over all k-tiles
    // No loop unrolling is performed at this stage
    for (int k_tile = 0; k_tile < size<2>(gmem_tensor_A); k_tile++) {

      // loops inside cute::gemm(tiled_mma, a, b, c); Dispatch 5: (V,M,K) x (V,N,K) => (V,M,N)
      // TiledMma uses the hardware instruction provided through its Mma_Atom
      // TiledMma's atom layout, value layout, and permutations define the iteration order
      for (int tiled_mma_k = 0; tiled_mma_k < size<2>(A); tiled_mma_k++) {
        for (int tiled_mma_m = 0; tiled_mma_m < size<1>(A); tiled_mma_m++) {
          for (int tiled_mma_n = 0; tiled_mma_n < size<1>(B); tiled_mma_n++) {

            // TiledMma's vector mode dispatches to the underlying instruction.
            mma.call(d, a, b, c);
          } // tiled_mma_n
        } // tiled_mma_m
      } // tiled_mma_k
    } // k_tile mainloop
  } // cluster_m
} // cluster_n
```

最外层三个嵌套循环对应线程块集群上的并行，代码里不会真正写成显式的 for 循环，块级并行的划分由 CUDA grid 启动语义隐含。不过对持久化 kernel（persistent kernel）而言，这三层循环会写成源码中的单个 while 循环，向工作块调度器查询要计算的问题块。

在这三个循环内部，代码把矩阵块从全局内存拉进更"本地"的内存（共享内存或寄存器）并计算 MMA。这些 tile 拷贝与 tiled mma 迭代通常是完全静态的，会被完全展开。

## CUTLASS GEMM 组件

CUTLASS 用以下组件表达上述循环嵌套，它们分别针对数据类型、布局和数学指令进行特化。

![](/assets/cutlass_api_3x/image.png)

| API 层 | API 类与函数名 |
| --- | --- |
| Device | cutlass::gemm::device::GemmUniversalAdapter |
| Kernel | cutlass::gemm::kernel::GemmUniversal |
| Collective | cutlass::gemm::collective::CollectiveMma、cutlass::epilogue::collective::DefaultEpilogue、cutlass::epilogue::collective::Epilogue |
| Tiled（MMA 与 Copy） | cute::TiledMma、cute::TiledCopy、cute::gemm()、cute::copy() |
| Atom | cute::Mma_Atom、cute::Copy_Atom |

CUTLASS 3.0 组装 kernel 的方式是，先在 kernel 层把 collective mainloop 和 collective epilogue 组合起来，再用主机端 adapter 包装，形成指向该 kernel 的 GEMM 句柄。

用户实例化这些组件的顺序如下。

1. 组装所需的 collective mainloop 与 epilogue
2. 把它们组合起来构建 kernel 类型
3. 用 device 层 adapter 包装 kernel

这一顺序也体现在 CUTLASS 3.0 的 Hopper kernel 示例里。

```c++
// Step 1: Generate the required collective layer mainloop specialization
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TilesShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Step 2: Specify the collective layer epilogue type
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
    ElementC,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>>;

// Step 3: Compose the mainloop and epilogue together at the kernel layer
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>, // ProblemShape [M,N,K,L]
    CollectiveMainloop,
    CollectiveEpilogue
>;

// Step 4: Wrap up the kernel::GemmUniversal kernel class
// with the device adapter to obtain a host-side handle to the kernel
using GemmHandle = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

最后再简要介绍 CuTe 的 tiled mma/copy 与 atom 层，更多细节请参考 CuTe 专用文档。

## Collective API

Collective 是"平铺 mma 原子和 copy 原子的最大线程集合"，也就是网格中能够靠硬件特性协作以加速通信和同步的最大线程数量。这些硬件特性包括：异步数组拷贝（例如从全局内存到共享内存）、针对共享内存中小块的 MMA 指令、面向集群/线程块/warp 的同步操作，以及保证异步操作之间数据依赖被满足的硬件加速（如 barrier）。

Collective 用 TiledMma 和 TiledCopy API 访问在 tile 上做拷贝和 MMA 的操作。它内部不同的并行单元（线程、warp 或线程块）可以承担不同角色。例如在 warp 专用算法中，一些 warp 负责拷贝数据，另一些负责计算；它们仍需共享数据并协调访问。把输入矩阵块拷贝进共享内存的生产者 warp，需要通知消费者 MMA warp 其输入已就绪。这一点与 kernel:: 层不同，后者负责把 collective 调度到网格中的独立块上。

Collective API 包含矩阵乘累加的主循环和尾声，是主循环融合、尾声融合等优化的组合点，负责实现上面伪代码里的 k_tile 循环。

### Collective Mainloop

cutlass::gemm::collective::CollectiveMma 是 collective 矩阵乘累加主循环的主要接口。"主循环"指遍历 tile 的主循环，即伪代码中的 cluster tile k 循环，算法对多个 tile 的任何循环都发生在这里。它声明在 cutlass/gemm/collective/collective_mma.hpp。

```c++
namespace cutlass::gemm::collective {

template <
  class DispatchPolicy,
  class TileShape,
  class ElementA,
  class StrideA,
  class ElementB,
  class StrideB,
  class TiledMma,
  class GmemTiledCopyA,
  class SmemLayoutAtomA,
  class SmemCopyAtomA,
  class TransformA,
  class GmemTiledCopyB,
  class SmemLayoutAtomB,
  class SmemCopyAtomB,
  class TransformB
>
struct CollectiveMma {
  static_assert(sizeof(ElementA) == 0, "Could not find a mainloop specialization.");
};

} // namespace cutlass::gemm::collective
```

- DispatchPolicy 是 collective 最重要的类型，下文详述。
- StrideA、StrideB 是 cute::Stride 实例，表示 A、B 张量的全局内存布局。步长必须是 3 阶，对应 [outer, inner, batch] 模式，每阶都可以是多模态分层步长，实现张量收缩时会用到。
- TiledMma 是 cute::TiledMma 实例。
- GmemTiledCopyA、GmemTiledCopyB 是 cute::TiledCopy 实例，下文详述。
- SmemLayoutAtomA、SmemLayoutAtomB 是 cute::Layout 实例，表示会在整个 collective 共享内存上平铺的最小布局。它不含流水线模式，因此是形状为 [outer, inner] 的 2 阶布局。
- SmemCopyAtomA、SmemCopyAtomB 是 Copy_Atom，用于把数据从共享内存搬到寄存器。

CUTLASS 3.0 的主循环不接受单独的累加器元素类型，累加器类型取自 typename TiledMma::ValTypeC。顶层 API 的 ElementA、ElementB 也可能与面向 MMA 的 typename TiledMma::ValTypeA、typename TiledMma::ValTypeB 不同，从而允许 TMA 或用户提供的变换操作执行类型转换。

### Collective Dispatch Policy

CollectiveMma 的实现不是通用的，必须针对每种算法和 GPU 架构特化。用户通过选取匹配特化的模板参数来调度，CUTLASS 3.0 采用基于标签的调度策略类型来特化主循环并添加调优旋钮。

下面是用于调度 Hopper TMA warp 专用主循环的调度策略之一。

```c++
// n-buffer in smem (Hopper TMA),
// pipelined with Hopper GMMA and TMA,
// warp-specialized dynamic schedule
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecializedCooperative
>
struct MainloopSm90TmaGmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
};
```

Stages_ 让用户自由改变流水线级数，ClusterShape_ 则用来参数化将进行 TMA 多播的线程块集群形状。

调度策略也是把各种 kernel 调度与任意主循环自由组合的关键。每个主循环策略要么规定一个必须配合的 Schedule，要么开放一个模板 API 让用户从以下调度中挑选子集。

```c++
struct KernelCpAsyncWarpSpecialized { };
struct KernelCpAsyncWarpSpecializedPingpong { };
struct KernelCpAsyncWarpSpecializedCooperative { };
struct KernelTma { };
struct KernelTmaWarpSpecialized { };
struct KernelTmaWarpSpecializedPingpong { };
struct KernelTmaWarpSpecializedCooperative { };
```

- 一个 kernel 调度可以支持多个主循环实现。例如 KernelMultistage 可与 MainloopSm70TwoStage、MainloopSm80CpAsyncUnpredicated 等多种跨架构主循环组合。
- 一个主循环也可以组合多个 kernel 调度。例如 MainloopSm90TmaGmmaWarpSpecialized 可与 KernelTmaWarpSpecialized、KernelTmaWarpSpecializedPingpong 或 KernelTmaWarpSpecializedCooperative 组合。

正如 CUTLASS 3.0 设计文档所述，为核心词汇类型采用标签调度策略，使得概念上属于同一类的操作可以共用一个类型名。这样设计的好处是：当主循环与多个 kernel 组合或反之，避免代码重复；主类型名 CollectiveMma 在任何实现中都不变，编写通用代码更容易；同时为用户提供一个清晰、单一的扩展点，插入针对自定义调度策略的新主循环实现。

### Collective Builder

CollectiveMma 的主要目标是成为专家接口，允许完全控制 collective GPU 微内核的所有属性。不过用户通常只需要一个现成的 GEMM 主循环，用简单配置参数参数化即可，CUTLASS 3.0 为此提供 cutlass::gemm::collective::CollectiveBuilder。

```c++
namespace cutlass::gemm::collective {
template <
  class ArchTag,
  class OpClass,
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType,
  class Enable = void
>
struct CollectiveBuilder {
  static_assert(sizeof(ElementA) == 0, "Could not build a collective for given parameters.");
};
} // namespace cutlass::gemm::collective
```

CollectiveBuilder 接受与 CUTLASS 2.x 等价的输入模板参数，并尝试据此构建性能最佳的 CollectiveMma。

- ArchTag 是 cutlass::arch::Sm* 中的 SM 架构标签。
- OpClass 是 cutlass::arch::OpClass* 中的算子类标签。
- ElementA、ElementB 分别是 A、B 张量的逻辑值类型。
- ElementAccumulator 是指令中使用的累加器类型。
- GmemLayoutA、GmemLayoutB 是 CUTLASS 2.x 布局标签，layout::RowMajor 或 layout::ColumnMajor。
- AlignmentA、AlignmentB 是 A、B 张量的全局内存对齐，以元素个数计。
- TileShape_MNK 是 cute::Shape 的 3 阶实例，表示 MxNxK 的 collective tile 形状。
- ClusterShape_MNK 是 cute::Shape 的 3 阶实例，表示 MxNxK 的线程块集群 tile 形状。
- StageCountType 是 collective::StageCountAuto 或 `collective::StageCount<N>` 实例。
- KernelScheduleType 是 collective::KernelScheduleAuto 或上文讨论过的某个具体 kernel 调度标签。

StageCountAuto 让 builder 计算单个 stage 在共享内存中的大小，并在假设每 SM 一个线程块占用的情况下最大化共享内存使用。KernelScheduleAuto 让 builder 根据参数挑选最佳 kernel 调度，或让用户用具体调度类型覆盖。

注意 collective builder 仍处于 beta，功能还未映射到专家版 CollectiveMma 所允许的完整设计空间。其支持的主循环类型预计在未来扩展，但在 3.0 中通过 builder 只支持 SM90 TensorOp kernel，builder API 未来也可能随用户反馈调整。如果 builder 能为给定参数给出 collective 主循环类型，会在内部别名为 CollectiveOp。如何使用 builder 便捷地参数化 kernel，可参考示例 49_hopper_gemm_with_collective_builder。

### Epilogue

collective 尾声实现涉及输出矩阵的逐元素运算。用户可以自定义尾声，或使用标准尾声之一，它们位于 include/cutlass/epilogue/collective/，包含 cutlass::epilogue::collective::DefaultEpilogue 和 cutlass::epilogue::collective::Epilogue 等类。CUTLASS 提供的 collective 尾声不在 include/cutlass/gemm 下、也不在 cutlass::gemm 命名空间里，因为它们可用于 GEMM 以外的计算。

## Kernel API

kernel 是"网格中所有集群的集合"。kernel 层调度负责四件事：对 kernel 内集群的执行排序并做必要的同步；把 warp 专用调度的线程编组到各自角色；执行必要的 grid swizzle 逻辑；在调用 collective 之前用线程块集群值对输入张量做分块。

Kernel API 是线程块网格的入口，这些线程块可能组成集群也可能不组成，它也是融合连续的 GEMM、尾声和其他操作的组合点。

CUTLASS 3.0 kernel 的入口是 cutlass::gemm::kernel::GemmUniversal，位于 include/cutlass/gemm/kernel/gemm_universal.hpp。GemmUniversal 是无状态的通用设备 kernel，把 GEMM 实现为两部分：一个 collective mainloop 和一个 collective epilogue。

```cpp
namespace cutlass::gemm::kernel {
/*
 * Stateless universal device GEMM kernel type that treats GEMM as
 * a composition of a collective mainloop and a collective epilogue.
 *
 * Supports both the 2.x and 3.x APIs based on whether the first type is
 * a cute::tuple<> or not.
 * 2.x API implementation: cutlass/gemm/kernel/gemm_universal.h
 * 3.x API implementation: cutlass/gemm/kernel/gemm_*.hpp
 *
 * In the following declaration, the name preceding the 'Or' refers to
 * 3.x API type argument order, and the name succeeding the 'Or' refers to
 * 2.x API type argument order. Template arguments without two names
 * belong to the 3.x API only.
**/
template <
  class ProblemShapeOrThreadblockMma_, // (m, n, k) or (m, n, k, l)
  class CollectiveMainloopOrEpilogue_,
  class CollectiveEpilogueOrThreadblockSwizzle_,
  class TileScheduler_ = void,
  class Enable = void
>
class GemmUniversal;
} // namespace cutlass::gemm::kernel
```

无状态意味着调用者（例如上面的 Device API）管理 kernel 的状态，kernel 只接收输入输出参数 Params。

通用意味着 GemmUniversal 同时适用于 CUTLASS 3.0 和 2.x 接口，并支持多种 kernel 调度。如果它的第一个模板参数是 cute::Shape，就假定其余模板参数实现 3.0 API；否则假定实现 2.x API。从 CUTLASS 3.0 起，问题形状被提升为 GEMM kernel 的顶层模板 API，这支持完全静态的 GEMM 实例，让用户在编译期获知部分或全部问题形状以获取更高性能。

collective mainloop 在局部 tile 上实现 MMA，collective epilogue 处理 MMA 之后的操作，例如应用 C := beta * C + alpha * A * B 中的 beta * C 部分。下文会进一步解释 collective。

3.0 API 的 kernel::GemmUniversal 特化位于 include/cutlass/gemm/kernel/ 下的各种 gemm_*.hpp，2.x API 的特化位于 include/cutlass/gemm/kernel/gemm_universal.h。

CUTLASS 3.x 实现了多种 kernel::GemmUniversal。每个 kernel 层调度针对一种 GEMM 调度算法和 GPU 架构特化，调度到哪个特化由调度策略的 Schedule 类型决定。例如 include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp 是面向 Hopper、采用持久化调度的 warp 专用主循环的 kernel::GemmUniversal 特化，而 include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp 是面向 Hopper、warp 专用但非持久化算法的特化。

为了在不复制 collective 主循环实现的前提下支持 kernel 调度与主循环调度策略的组合，GEMM kernel 层调度可以与任何把对应 kernel 调度写进策略 Schedule 类型的主循环组合，详见上文 collective dispatch policy 一节。

```c++
// An example of the SM90 KernelMultistage kernel's
// specialization logic that allows it to be composed
// with many mainloops such as `MainloopSm80CpAsync`
// and `MainloopSm70TwoStage`.
template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  std::enable_if_t<std::is_base_of_v<KernelMultistage, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
```

## Device API

Device API 是通用、与具体 kernel 无关的主机接口，用于启动 kernel 并管理可复用主机端参数的生命周期。它是用户主机端 .cu 代码调用 CUTLASS 单 GPU GEMM kernel 的方式，用途和行为都与 cuBLAS 类似。

Device GEMM API 的入口是 cutlass::gemm::device::GemmUniversalAdapter，位于 include/cutlass/gemm/device/gemm_universal_adapter.h。它是基于 cutlass::gemm::kernel 类型参数化的有状态、可复用句柄。

```c++
/*! 
  GemmUniversalAdapter is a stateful, reusable GEMM handle built around a kernel
  of type cutlass::gemm::kernel::*

  It manages the lifetime of the underlying `kernel::Params` struct, and exposes APIs
  to create it from the host facing arguments. For power users, new static methods
  are exposed in 3.x APIs that bypass the stateful methods or args->params lowering.

  It supports kernel types that implement both the 2.x and 3.0 APIs,
  however, this is done by specializing the implementation of GemmUniversalAdapter
  on the two kernel API types, and thus, GemmUniversalAdapter's behavior might
  differ between the two specializations.
*/
template <class GemmKernel_, class Enable = void>
class GemmUniversalAdapter;
```

有状态意味着句柄实例包含 kernel 运行所需的状态，用户必须先初始化句柄，再用初始化后的句柄运行 kernel。有状态也意味着句柄可以管理 kernel 参数 Params 的生命周期。GemmUniversalAdapter 的一项重要职责，是把用户看到的 Arguments 映射到 kernel 实际看到的 Params。面向高级用户，该类在 3.0 API 中提供了新的静态方法，可以绕过有状态方法，或直接把参数降到 Params 而无需中间 Arguments。

可复用意味着句柄实例可以用不同参数（例如不同矩阵）多次调用 kernel，复用句柄往往比每次调用都新建更高效。

基于 kernel 类型参数化意味着 GemmUniversalAdapter 的行为取决于 GEMM kernel 类型，其模板参数 GemmKernel 就是 GEMM kernel 类型。GemmKernel 的合法取值包括：实现 CUTLASS 3.x API kernel 的 cutlass::gemm::kernel::GemmUniversal；实现 CUTLASS 2.x API kernel 的 cutlass::gemm::kernel::GemmUniversal；或任何先前可与 device::GemmUniversalAdapter 组合的合法 CUTLASS 2.x kernel 层 GEMM。

GemmUniversalAdapter 为 3.0 和 2.x kernel 提供统一的主机端接口，做法是在实现 kernel 层 GEMM 的 2.x API 或 3.x API 上特化其实现。它用元函数 cutlass::gemm::detail::IsCutlass3GemmKernel 区分 2.x 和 3.x kernel。

GemmUniversalAdapter 负责设置并启动 kernel，必要时使用 CUDA 扩展启动 API 支持线程块集群。它不指定 grid 形状，grid 形状和其他 kernel 相关的启动参数由 kernel 控制。这样所有 3.0 kernel 都能共用同一份启动代码，从而把 kernel 启动同实际 kernel 分离。

## Tiled MMA 与 Copy

Tiled MMA 或 Copy 分别是 MMA 原子、Copy 原子在线程和数据上的平铺，必要时对生成的平铺应用置换。这一层最接近 CUTLASS 2.x 中 MMA 指令的 warp 级平铺，但它从参与操作的所有线程的视角看待平铺，并把概念推广到拷贝操作。其目的是用大量硬件加速的数学与数据搬运操作构建可组合的 GPU 微内核，每个操作都有其在线程和数据上的单元布局。Tiled MMA 和 Copy 类型用单一、一致的 API 呈现这些不同的硬件加速 CuTe 原子。

生成的平铺操作相当于单个 MMA 或拷贝操作，用户可以在上面三层循环伪代码的内层循环中，用 cute::gemm() 或 cute::copy() 调用。

之所以叫"tiled"，是因为它用 CuTe 提供的 Atom 构建更大的操作，像是把一块块 tile 拼成可复用的马赛克组件。例如 CuTe 可能提供一个可在单个 warp 上调用、用于固定 M、N、K 的 MMA Atom，CUTLASS 再用 make_tiled_mma 之类的 CuTe 操作把它变成作用于整个线程块、面向更大 M、N、K 的操作。

## Atom API

Atom 是必须参与执行某个硬件加速数学或拷贝操作的最小线程和数据集合。

Atom 的"原子性"（不可分割性）指的不是像 atomicAdd 那样的并发内存操作（那种是"在时间/因果关系上不可分割"），而是"空间"上的不可分割：必须一起参与操作的值的数量，以及并行工作器的组数。

Atom 用 CuTe Layout 表达其输入输出数组所需的维度和步长，通常这些在编译期就已确定。Atom API 封装了对真正加速 MMA 或拷贝操作的硬件指令的调用，用户可以请求特定 GPU 架构的实现，也可以选通用实现并依赖已启用的任何 GPU 架构。
