---
title: CuTe tiledMMA
date: 2026-06-20 20:00:00
tags: [CUTLASS, CuTe, tiledMMA, MMA, GPU]
categories: [Cutlass学习笔记]
description: 深入解析 CuTe tiledMMA 机制，涵盖 TiledMMA 的创建、ThrMMA、partition 操作及 warpgroup 级别 MMA 的实现方式。
---

# CuTe_tiledMMA

# outline

1. MMA Operation
1. MMA Traits
1. volta
1. ampere
1. hopper
1. MMA Atom
1. tiledMMA
1. threadMMA

在现在GPU架构中，PTX的mma指令是GPU调用Tensor Core进行矩阵乘加的基本指令。不同的GPU架构拥有不同的mma指令集，如mma和wgmma等。

从前面介绍mma和wgmma的文章中可以知道，每个指令都有相应的线程和数据布局，直接使用PTX代码非常麻烦。

而CuTe替我们解决了这一问题。通过对不同的mma指令进行封装，可以让开发者不用过分关注mma指令中线程和数据的关系，从而能够专注于功能的实现，而不是具体的指令调用。

简单来说，一个具体的mma指令在cute中称为一个mma_atom。一个mma atom只能计算部分的数据，因此根据用户的问题规模，cute可以将许多mma atom进行拼接组合形成tiled mma。然后通过tiled mma进行运算。

在实际实现过程中，cute又把上述流程细分为下面4个步骤。

1. cute把mma支持的每一种指令都定义了一个Operation结构体。operation结构体中只包含必要的寄存器和内联ptx mma指令。
1. 针对每个operation结构体，cute定义了对应的traits结构体来描述该指令需要的基本信息，比如矩阵A，B，C的数据类型和形状，参与计算的线程数量，以及在计算过程中A，B，C的线程和数据的布局等信息。
1. 结合operation和traits两个结构体定义了MMA Atom结构体。Atom结构体提供了fragments等方法来创建可以进行计算的cute::tensor。
1. 对mma atom进行tiling生成tiledmma。tiledmma等于是mma atom复制粘贴后的结果，可以处理更大的数据范围。

# MMA Operation

op结构体定义在include/cute/arch中，头文件以mma开始。

op结构体名字的定义与其对应的PTX指令有关系。通常是由第一个支持的架构代码，支持的MNK大小数据类型和AB的input的排列组成。比如SM70_8x8x4_F32F16F16F32_NT，其中SM70代表从Volta架构开始支持

8×8×4表示M=8，N=8，K=4，对应的ptx指令是.m8n8k4。F32F16F16F32表示ABCD矩阵的数据类型。因为MMA是D = A * B + C，所以D是fp32，A和B是FP16，C是FP32。对应的ptx指令是.f32.f16.f16.f32。

NT表示ptx指定的线程对A矩阵是按照M-major的形式（not trans, column-major）加载的，对于B是按照N-major的形式（Transposed，row-major）加载的。对应ptx指令中的.col.row

一个op结构体由两部分组成，类型别名和fma函数。

一个op结构体中有4个类型别名，DRegisters, ARegisters, BRegisters和CRegisters，代表ABCD四个输入的数组。比如在SM70_8x8x4_F32F16F16F32_NT中，using DRegisters = float[8]，using ARegisters = uint32_t[2]，using BRegisters = uint32_t[2]，using CRegisters = float[8]。

表示了ABCD四个矩阵中每个线程有多少个元素会进入到ptx中计算。在这个例子中，DC矩阵每个线程会有8个FP32进行计算，所以是float[8]，AB矩阵每个线程需要4个FP16元素，所以是uint32[2]。

fma是静态成员函数。不同的op结构体定义的fma需要不同数量的参数。直接通过内联ptx代码执行mma计算。

```cpp
// MMA 16x8x8 TN
struct SM80_16x8x8_F16F16F16F16_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
      "{%0, %1},"
      "{%2, %3},"
      "{%4},"
      "{%5, %6};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM80_16x8x8_F16F16F16F16_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

// MMA 16x8x16 TN
struct SM80_16x8x16_F16F16F16F16_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM80_16x8x16_F16F16F16F16_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};
```

# MMA_Traits

mma_traits定义在include/cute/atom路径下，头文件是mma_traits开头。

traits包含了ptx指令的一些基本信息，主要包括下面的内容

ValTypeD: 矩阵D的数据类型

ValTypeA: 矩阵A的数据类型

ValTypeB: 矩阵B的数据类型

ValTypeC: 矩阵C的数据类型

Shape_MNK: MMA op需要的矩阵计算的大小

ThrID: 一个MMA op需要的线程数量，可能是一个thread，8个thread，32个thread和一个warpgroup

ALayout: 线程在A矩阵上的布局

BLayout: 线程在B矩阵上的布局

CLayout: 线程在C矩阵上的布局

```cpp
/**
 * concept MMA_Traits
 * {
 *   using ValTypeD =  // Logical A-value type
 *   using ValTypeA =  // Logical B-value type
 *   using ValTypeB =  // Logical C-value type
 *   using ValTypeC =  // Logical D-value type    (NOTE: Not used? Assumed == ValTypeD)
 *
 *   using FrgTypeA =  // A-type consumed by MMA  (if ommitted, same as ValTypeA)
 *   using FrgTypeB =  // B_type consumed by MMA  (if ommitted, same as ValTypeB)
 *   using FrgTypeC =  // C_type consumed by MMA  (if ommitted, same as ValTypeC)
 *
 *   using Shape_MNK =    // Logical MxNxK shape of the MMA
 *
 *   using ThrID     =    // Logical thread id (tid) -> tidx
 *
 *   using ALayout =      // (Logical thread id (tid), Logical value id (vid)) -> Flat MK-coord
 *   using BLayout =      // (Logical thread id (tid), Logical value id (vid)) -> Flat NK-coord
 *   using CLayout =      // (Logical thread id (tid), Logical value id (vid)) -> Flat MN-coord
 * };
 */
```

举例如下：

```cpp
namespace {

// (T32,V1) -> (M8,N8)
using SM80_8x4      = Layout<Shape <Shape < _4,_8>,_1>,
                             Stride<Stride< _8,_1>,_0>>;
// (T32,V2) -> (M8,N8)
using SM80_8x8_Row  = Layout<Shape <Shape < _4,_8>,_2>,
                             Stride<Stride<_16,_1>,_8>>;
// (T32,V4) -> (M8,N16)
using SM80_8x16_Row = Layout<Shape <Shape < _4,_8>,_4>,
                             Stride<Stride<_32,_1>,_8>>;
// (T32,V4) -> (M16,N8)
using SM80_16x8_Row = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                             Stride<Stride<_32,_1>,Stride<_16,_8>>>;

}

template <>
struct MMA_Traits<SM80_16x8x8_F16F16F16F16_TN>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_16,_8,_8>;
  using ThrID   = Layout<_32>;
  using ALayout = SM80_16x8_Row;
  using BLayout = SM80_8x8_Row;
  using CLayout = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_16,_8,_16>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2,  _2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_8,_128>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_2, _2>>,
                         Stride<Stride<_16,_1>,Stride<_8,_64>>>;
  using CLayout = SM80_16x8_Row;
};
```

上面的代码分别是SM80_16x8x8_F16F16F16F16_TN的Traits和SM80_16x8x16_F16F16F16F16_TN的Traits。

可以看到由于两个mma指令计算时ABCD矩阵的数据类型都是fp16，所以ValTypeA，ValTypeB，ValTypeC，ValTypeD 都是half。SM80_16x8x8_F16F16F16F16_TN的Shape_MNK是（16，8，8），SM80_16x8x16_F16F16F16F16_TN的Shape_MNK 是（16，8，16），代表各自需要的计算的矩阵的shape。两个的ThrID都是32，因为两个指令都是warp级别的mma指令，需要一个warp的32个线程参与。如果是其他级别的矩阵乘则需要修改成对应的线程数量。

此外，比较抽象的是ALayout，BLayout和CLayout的定义。这三个是thread-value layout，表示线程和数据关系的layout。因为mma指令中线程和数据的对应关系比较复杂，所以抽象成这三个layout进行描述。

关于layout的详细介绍参考前面的文章。

对于tv-layout，如下图所示，最右边灰色的4*8矩阵是数据value。中间的8*4矩阵是线程的布局，也就是thread layout，里面的数据是矩阵的index，8行代表8个线程，4列代表一个线程对应4个元素，矩阵的元素由其中的index获得。比如0号thread会访问index为0，4，16，20位置的数据。

将数据的layout和thread的layout进行组合（composition），就会把数据从之前的布局转换到一种新的布局，tv-layout。这种布局下第一维是thread的layout，第二维是value的layout，通过对第一维进行索引就可以得到某个线程对应的数据。

这里需要注意数据的layout的thread的layout的区别。数据的layout是由用户决定的，可以是row-major或column-major。但是thread的layout是mma指令决定的，跟数据无关。

![](/assets/cute-tiled-mma/image.png)

根据架构的不同，不同mma指令对应的线程的layout也不一样。

## UniversalFMA

对于UniversalFMA，只需要一个线程执行d = a * b + c，一个线程对应ABC矩阵中的一个元素，所以ABC矩阵的layout就是Layout<Shape<_1,_1>>。

```cpp
template <class D, class A, class B, class C>
struct MMA_Traits<UniversalFMA<D,A,B,C>>
{
  using ValTypeD = D;
  using ValTypeA = A;
  using ValTypeB = B;
  using ValTypeC = C;

  // Logical shape of the MMA
  using Shape_MNK = Shape<_1,_1,_1>;

  // Logical thread id (tid) -> tidx
  using ThrID   = Layout<_1>;

  // (Logical thread id (tid), Logical value id (vid)) -> coord

  // (tid,vid) -> (m,k)
  using ALayout = Layout<Shape<_1,_1>>;
  // (tid,vid) -> (n,k)
  using BLayout = Layout<Shape<_1,_1>>;
  // (tid,vid) -> (m,n)
  using CLayout = Layout<Shape<_1,_1>>;
};
```

## Volta架构

在Volta架构中支持使用mma指令调用Tensor Core计算矩阵乘。mma指令支持的大小是m8n8k4。

从前面mma文章的介绍中可以知道，一个m8n8k4指令需要8个线程计算，8个线程称为一个quadpair(QP)。一个warp中的32个线程可以执行4个m8n8k4，也就是4个QP，一共16×16×4大小的矩阵运算。

下面以SM70_8x8x4_F32F16F16F32_TN和SM70_8x8x4_F32F16F16F32_NT为例进行介绍如何确定Traits。

```cpp
// Logical thread id to thread idx (quadpair)
using SM70_QuadPair = Layout<Shape <_4, _2>,
                             Stride<_1,_16>>;
// (T8,V4) -> (M8,K4)
using SM70_8x4_Row  = Layout<Shape <_8,_4>,
                             Stride<_1,_8>>;
// (T8,V4) -> (M8,K4)
using SM70_8x4_Col  = Layout<Shape <Shape <_4,_2>,_4>,
                             Stride<Stride<_8,_4>,_1>>;
// (T8,V8) -> (M8,N8)
using SM70_8x8_16b  = Layout<Shape <_8,_8>,
                             Stride<_1,_8>>;
// (T8,V8) -> (M8,N8)
using SM70_8x8_32b  = Layout<Shape <Shape <_2, _2,_2>,Shape <_2,_2, _2>>,
                             Stride<Stride<_1,_16,_4>,Stride<_8,_2,_32>>>;

template <>
struct MMA_Traits<SM70_8x8x4_F32F16F16F32_TN>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = SM70_QuadPair;
  using ALayout = SM70_8x4_Row;
  using BLayout = SM70_8x4_Row;
  using CLayout = SM70_8x8_32b;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM70_8x8x4_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = SM70_QuadPair;
  using ALayout = SM70_8x4_Col;
  using BLayout = SM70_8x4_Col;
  using CLayout = SM70_8x8_32b;
};
```

这两个指令的数据类型是F32F16F16F32，说明计算时AB矩阵是FP16类型，CD矩阵是FP32类型，所以ValTypeD = float，ValTypeA = half_t，ValTypeB = half_t，ValTypeC = float。又因为两个指令的大小都是8x8x4，所以Shape_MNK = Shape<_8,_8,_4>。

从前面mma的介绍可知，虽然在Volta架构上一个mma需要8个线程，但是却不是连续的线程，而是[0,1,2,3]和[16,17,18,19]。在cute中默认使用列优先描述线程，所以这8个线程可以看成shape为4行2列，stride为[1,16]的二维线程矩阵。所以thread的layout就是Layout<Shape <_4, _2>, Stride<_1,_16>>，也就是上面定义的SM70_QuadPair。

然后对于ALayout，mma的线程有两种布局，分别是row-major的布局和column-major的布局。

当使用row-major的布局时，线程的layout如下所示：一共8行4列，一行一个线程加载4个元素。因为要计算tv-layout，所以可以先固定value计算thread的layout。这里对于value a0，8个线程之间是连续

，所以shape就是（8，4），默认使用列优先描述线程，stride就是（1，8）。所以row-major下，ALayout就是Layout<Shape <_8,_4>, Stride<_1,_8>>。

![](/assets/cute-tiled-mma/image_1.png)

这时有人会问了，A的布局不是row-major吗，为什么ALayout是col-major？这是因为这里的row-major是线程加载A矩阵的布局，也就是如果设置了.row，线程就会按照这种布局加载数据，跟数据的row-major布局没关系，线程的布局可以应用到任意的数据布局上。

举个例子：如下图所示，最左边上面是row-major的数据，下面是col-major的数据，中间是row-major下线程的布局，一种颜色代表一个线程。composition运算后得到最右边的结果。可以看到不管数据是row-major还是column-major，运算的结果都是一个线程对应原始数据的一行元素，符合预期。

![](/assets/cute-tiled-mma/image_2.png)

至于为什么默认使用列优先描述线程布局，这可能跟cute的底层实现有关系，cute中layout相关的运算都是以列优先进行的。

还有为什么这时不考虑[0,1,2,3]和[16,17,18,19]线程的id了，因为这里是local thread的id，跟全局线程id没关系了。

当使用column-major的布局时，线程的layout如下所示：还是8行4列，但是0-3号线程加载前4行，16-19号线程加载后4行。所以此时ALayout是

![](/assets/cute-tiled-mma/image_3.png)

对于NT类型，也就是.col.row类型，A的线程分布是

B的线程分布是

C的线程分布是，如果是fp32的话

用cute中的layout描述如下图。

因此对于数据类型为

```cpp
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;
```

矩阵形状为

```cpp
  // Logical shape of the MMA
  using Shape_MNK = Shape <_8,_8,_4>;
```

第一个QP需要线程0-3，16-19处理，[0,1,2,3,16,17,18,19]可以看成一个4行2列的数组，列主序。行之间stride=1，列之间stride=16，所以ThrID = Layout<Shape <_4, _2>, Stride<_1,_16>>;

```cpp
  // Mapping from (logical thread id) -> (thread idx)
  using ThrID = Layout<Shape <_4, _2>,
                       Stride<_1,_16>>;
```

同样的，这个layout对另外三个QP也成立。

当数据类型是fp32时，C和D中线程和对应的元素关系如下所示，可以看到一共有8个线程，一个线程有8个元素。

我们需要通过一个Layout确定每个线程和其对应的元素。我们可以将线程可其对应的元素单独弄出来，如下面这样

不难看出第一维是线程的layout，第二维是value的layout，这就是前面tensor里提到的TV layout。

thread [0 1 16 17 4 5 20 21] ->[[[0 1],[16 17]], [[4 5], [20 21]]]，所以shape = [2, 2, 2]，stride=[1, 16, 4]。

value [0 8 2 10 32 40 34 42] ->[[[0 8], [2 10]], [[32, 40], [34, 42]]], shape = [2, 2, 2], stride=[8, 2, 32]。

所以完整的layout = <thread_layout，value_layout> = <<2, 2, 2>, <2, 2, 2>>:<<1, 16, 4>, <8, 2, 32>>。

将这个layout应用到C和D的矩阵上就可以得到每个线程与元素的对应关系，如下图所示

当数据类型是fp16时，C和D的线程和元素的关系如下：

thread_layout = (8):(1)，value_layout = (8):(8)，所以TV_layout = (8, 8):(1, 8)

## 矩阵A和矩阵B

对于矩阵A和B，因为线程有row和col两种对应关系，所以需要分开讨论。需要注意的是，这里的row和col是线程的加载方式，跟数据的row-major和col-major没关系，线程加载方式可以应用到任何数据布局上。详见issue

首先讨论两种布局，NT和TN，分别对应.col.row和.row.col。这里的T表示transposed，N代表not transposed。在cute中默认是列主序，可以简单的理解为不是列主序的都是T，是列主序的都是N。

对于.col.row

因为A是col加载的，所以是N，B是row加载的，所以是T。这种情况下，线程和元素的对应关系如下：

对于A：

thread_layout = [[0 8 16 24], [4 12 20 28]] = (4,2):(8,4)

value_layout = [0,1,2,3] = (4):(1)

所以layout = ((4,2),4):((8,4),1)

同样的，对于B

，B在.row时线程的加载方式如下：

所以layout还是layout = ((4,2),4):((8,4),1)

当AB是TN布局时，也就是.row.col

对于A，

线程与元素的对应关系如下：

容易得出layout = (8,4):(1,8)

这时有人会问了，A的布局不是row-major吗，为什么显示是col-major？这是因为.row是线程加载A矩阵的布局，跟数据的布局没关系，线程的布局可以应用到任意的数据布局上。在cute中A默认是M-major，B默认是N-major，详见issue

研究下这个layout是怎么起作用的。

对于B，

.col对应下面这样

所以Blayout=(8,4):(1,8)

当AB的布局是NN或TT时，根据上面的选择即可

## ampere架构

以SM80_16x8x16_F16F16F16F16_TN为例

显然，ABCD的数据类型都是half，Shape_MNK=(16, 8, 16)，由于这个mma指令需要一个warp中全部的32个线程，所以ThrID的layout就是(32):(1)。

下面分析ABC的layout。

首先时CLayout，

懒得画了，容易得出，Clayout=((4,8),(2,2)) : ((32,1),(16,8))

矩阵A

Alayout = ((4,8),(2,2,2)) : ((32,1),(16,8,128))

矩阵B

BLayout=((4,8),(2,2)):((16,1),(8,64))

## hopper架构

关于指令的TN NT与数据的TN NT不一致的解释

[https://github.com/NVIDIA/cutlass/discussions/1271](https://github.com/NVIDIA/cutlass/discussions/1271)

[https://github.com/NVIDIA/cutlass/issues/1226](https://github.com/NVIDIA/cutlass/issues/1226)

# MMA_Atom

mma_atom的结构比较简单，由MMA_Traits和MMAOperation组成，包括数据类型，线程布局和一些成员函数。

```cpp
template <class MMAOperation, class... Args>
struct MMA_Atom<MMA_Traits<MMAOperation, Args...>>
  : MMA_Traits<MMAOperation, Args...>
{
  using MMA_Op = MMAOperation;
  using Traits = MMA_Traits<MMAOperation, Args...>;

  // Element value types from the MMA_Traits
  using ValTypeD = typename Traits::ValTypeD;
  using ValTypeA = typename Traits::ValTypeA;
  using ValTypeB = typename Traits::ValTypeB;
  using ValTypeC = typename Traits::ValTypeC;

  // Thr-Val layouts from the MMA_Traits
  using Shape_MNK  = typename Traits::Shape_MNK;
  using ThrID      = typename Traits::ThrID;
  using LayoutC_TV = typename Traits::CLayout;
  using LayoutA_TV = typename Traits::ALayout;
  using LayoutB_TV = typename Traits::BLayout;

  // Fragment value types from the MMA_Traits (optional, defaults to Val type)
  using FrgTypeD = typename detail::FrgTypeC_or_Default<Traits>::type;
  using FrgTypeA = typename detail::FrgTypeA_or_Default<Traits>::type;
  using FrgTypeB = typename detail::FrgTypeB_or_Default<Traits>::type;
  using FrgTypeC = typename detail::FrgTypeC_or_Default<Traits>::type;
  
  ...
}
```

成员函数：

with：没看明白干什么的，应该是更新Traits的。

call：调用函数，调用mma_unpack进行mma计算。

make_fragment_C：在寄存器空间创建矩阵C。

make_fragment_A：在寄存器空间创建矩阵A。

make_fragment_B：在寄存器空间创建矩阵B。

# TiledMMA

TiledMMA是cute计算的核心。使用mma atom组成TiledMMA，然后使用tiledmma对矩阵进行分块，进而完成矩阵运算。

TiledMMA的基本定义如下所示，包含三个参数。MMA_Atom：使用的mma指令类型。AtomLayoutMNK：按照什么布局复制mma atom。PermuationsMNK：tiledMMA的大小，一般不设置，会根据atom的shape和atom layout自动计算。

```cpp
// @tparam MMA_Atom The MMA_Atom to use in the TiledMMA
// @tparam AtomLayoutMNK The MNK-tiling of the Atom to be performed.
// @tparam PermuationsMNK Permutations to apply to each MNK-mode before tiling for the Atom.
template <class MMA_Atom,
          class AtomLayoutMNK,
          class PermutationMNK = Tile<Underscore,Underscore,Underscore>>
struct TiledMMA : MMA_Atom
{
  using Atom           = MMA_Atom;
  using AtomShape_MNK  = typename MMA_Atom::Shape_MNK;
  using AtomThrID      = typename MMA_Atom::ThrID;
  using AtomLayoutC_TV = typename MMA_Atom::LayoutC_TV;
  using AtomLayoutA_TV = typename MMA_Atom::LayoutA_TV;
  using AtomLayoutB_TV = typename MMA_Atom::LayoutB_TV;

  static_assert(   rank_v<AtomLayoutMNK>  == 3,   "TiledMMA requires rank-3 AtomLayoutMNK");
  static_assert(   rank_v<PermutationMNK> == 3,   "TiledMMA requires rank-3 PermutationMNK");
  static_assert( is_tuple<PermutationMNK>::value, "TiledMMA requires independent permutations of MNK.");
  static_assert(is_static<PermutationMNK>::value, "TiledMMA requires static permutations of MNK.");

  using ThrLayoutVMNK = decltype(tiled_product(AtomThrID{}, AtomLayoutMNK{}));
  ThrLayoutVMNK thr_layout_vmnk_; // TiledMMA中线程的布局

  CUTE_HOST_DEVICE constexpr
  TiledMMA(MMA_Atom const& mma_atom = {}, AtomLayoutMNK const& thr_layout_mnk = {})
    : MMA_Atom(mma_atom),
      thr_layout_vmnk_(tiled_product(AtomThrID{}, thr_layout_mnk)) {}
  ....
}
```

### ThrLayoutVMNK

tiledmma中的变量，用于记录tiledmma中线程的layout。

如果一个mma需要32个线程参与，atomlayout类型的变量thr_layout_mnk是（2，2，2）的话，mma需要在MNK三个方向各复制2次，一共需要256个线程。

thr_layout_vmnk_是通过tiled_product计算的。

```cpp
thr_layout_vmnk_(tiled_product(AtomThrID{}, thr_layout_mnk)) {}
```

举个例子：

从下面的代码中可以看到，tiled_product的结果一共是4维，第一维是atom的线程数，后面分别是MNK方向需要复制的数量。

```cpp
    auto l1 = Layout<_32>{};
    auto l2 = Layout<Shape<_2, _2, _2>>{};
    auto res = tiled_product(l1, l2);
    print(res); // (_32,_2,_2,_2):(_1,_32,_64,_128)
```

这里用的是tiled_product，只有第一维的维度是完整的，其余的都是单独的维度。

```cpp
    auto l1 = Layout<Shape<_32, _4>>{};
    auto l2 = Layout<Shape<_2, _2, _2>>{};
    auto res = tiled_product(l1, l2);
    print(res); // ((_32,_4),_2,_2,_2):((_1,_32),_128,_256,_512)
```

### thrfrg_C

tiledMMA的核心函数之一，主要作用是使用tiledmma对tensor进行分块，并转换成thread value layout的布局。入参是tensor C的layout，在partition C中被调用。

这里的tensor一般是block后的tensor大小，对于C来说一般是(bM, bN, ...)。

```cpp
  // Tile a tensor or a layout from shape
  //   (M,N,...)
  // to shape
  //   ((ThrV,(ThrM,ThrN)),(FrgV,(RestM,RestN,...)))
  // where
  //   ThrV:  The threads local to an MMA. layout<0>(ThrLayoutVMNK): ThrV -> thread_idx
  //   ThrM:  The threads tiled in M.      layout<1>(ThrLayoutVMNK): ThrM -> thread_idx
  //   ThrN:  The threads tiled in N.      layout<2>(ThrLayoutVMNK): ThrN -> thread_idx
  //   FrgV:  The values local to an MMA.
  //   RestM: The values tiled in M.
  //   RestN: The values tiled in N.
template <class CTensor>
  CUTE_HOST_DEVICE constexpr
  auto
  thrfrg_C(CTensor&& ctensor) const
  {
    CUTE_STATIC_ASSERT_V(rank(ctensor) >= Int<2>{});
    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(permutation_mnk<0>(),
                            permutation_mnk<1>());
    auto t_tensor = logical_divide(ctensor, t_tile);                 // (PermM,PermN)

    // Tile the tensor for the Atom
    auto c_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                            make_layout(size<1>(AtomShape_MNK{})));
    auto c_tensor = zipped_divide(t_tensor, c_tile);                 // ((AtomM,AtomN),(RestM,RestN))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = c_tensor.compose(AtomLayoutC_TV{},_);           // ((ThrV,FrgV),(RestM,RestN))

    // Tile the tensor for the C-threads
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<1>(thr_layout_vmnk_)),
                                        make_layout(size<2>(thr_layout_vmnk_))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrM,ThrN)),(FrgV,(RestM,RestN)))

    return thr_tensor;
  }
```

首先是permutation_mnk，这个函数的作用是根据tiledmma中传进来的参数PermutationMNK{}对MNK维的数据排序。PermutationMNK{}一般是三维layout，分别代表MNK，所以在thrfrg_C中只获取MN两维。auto perm = get<I>(PermutationMNK{})表示获取第I维。

返回的时候是条件返回，如果perm是下划线_，就返回size<I>(AtomShape_MNK{}) * size<I+1>(get_thr_layout_vmnk())的结果，其中AtomShape_MNK是mma atom的shape。get_thr_layout_vmnk返回的是tiledmma中线程的layout。

比如对于SM80_16x8x8_F16F16F16F16_TN来说，AtomShape_MNK{}是Shape<_16,_8,_8>，ThrID   = Layout<_32>。

如果tiledmma的atom layout是<2,2,2>则，get_thr_layout_vmnk返回的是(_32,_2,_2,_2):(_1,_32,_64,_128)，因此size<0>(AtomShape_MNK{}) = 16，size<0+1>(get_thr_layout_vmnk()) = 2，permutation_mnk<0>返回的就是32，permutation_mnk<1>返回的就是16。

这个值代表mma atom的shape在MN方向上tiled后的大小。

```python
  // The permutation applied to the MNK-mode data
  template <int I>
  CUTE_HOST_DEVICE constexpr
  auto
  permutation_mnk() const {
    static_assert(0 <= I && I < 3);
    auto perm = get<I>(PermutationMNK{});
    return conditional_return(is_underscore<decltype(perm)>{}, size<I>(AtomShape_MNK{}) * size<I+1>(get_thr_layout_vmnk()), perm);
  }
```

从上面可以知道，permutation_mnk<I>()获取的是tiled mma在M或N方向的shape大小，因此make_tile后得到的t_tile就是tiledMMA在C tensor上的大小。

然后进行logical_divide。logical_divide的具体计算过程可以参考layout algebra文章。简单来说就是把LayoutA按照LayoutB进行分块并重新组合。

因为这里t_tile是两个Layout的组合，所以logical_divide进行的是按维度除，也就是ctensor的第一维除以t_tile的第一维，ctensor的第二维除以t_tile的第二维。

比如，在上面得到的t_tile = <32:1, 16:1>，假如ctensor的Layout是(64, 64):(1, 64)，那么logical_divide的结果就是((_32,_2),(_16,_4)):((_1,_32),(_64,_1024))。可以看到ctensor的第一维的64被32分成了(32,2)，第二维的64被16分成了(16,4)。

```python
    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(permutation_mnk<0>(),
                            permutation_mnk<1>());
    auto t_tensor = logical_divide(ctensor, t_tile);       // (PermM,PermN)
```

然后是下面的代码。上面的代码的主要功能是用tiledMMA的shape去对ctensor进行分块，下面代码的主要功能是用atom的shape去对tiledMMA分块后的tensor进一步分块。c_tile和t_tile类似，不过大小是atom的大小，这里假设是(16,8)。

```python
    // Tile the tensor for the Atom
    auto c_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                            make_layout(size<1>(AtomShape_MNK{})));
    auto c_tensor = zipped_divide(t_tensor, c_tile); // ((AtomM,AtomN),(RestM,RestN))
```

zipped_divide和logical_divide的计算逻辑相同，只是最后输出的维度组合方式不同。在logical_divide中，比如M维64能被32分成2块，N维的64能被16分成4块，最后的组合方式就是((32,2),(16,4))。而zipped_divide则会按照(32,16),(2,4)的方式组合，这样做的好处是一个tile的数据全部在同一个维度，方便索引。

因此c_tensor = zipped_divide(t_tensor, c_tile) = ((_32,_2),(_16,_4)):((_1,_32),(_64,_1024)) / <16:1,8:1> = ((_16,_8),(_4,_8)):((_1,_64),(_16,_512))。

感觉直接对ctensor做zipped_divide也能得到同样的结果，为啥还要先做logical_divide。可能是因为permutation_mnk？tiledMMA对ctensor的分块方式可能不同。

再然后是下面的代码，这段代码的意思是把得到的c_tensor与mma的CLayout组合，得到tv_layout。

这里的AtomLayoutC_TV{}就是mma的CLayout，因为每个mma都有特定的线程分布，具体可以参考mma文章。

c_tensor.compose(AtomLayoutC_TV{},_)等价于composition(c_tensor, make_tile(AtomLayoutC_TV{}, _))。

composition的具体原理参考Layout algebra。

组合后得到的结果就是所谓的tv_layout。简单来说就是c_tensor是一块数据的布局，AtomLayoutC_TV是线程的布局，组合后会把c_tensor中的数据按照线程的布局重新排列，就得到了tv_layout。对tv_layout的第一维进行索引就能得到某个线程对应的数据。具体参考Tensor文章。

```python
    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = c_tensor.compose(AtomLayoutC_TV{},_); // ((ThrV,FrgV),(RestM,RestN))
```

在上面的例子中，c_tensor是((_16,_8),(_4,_8)):((_1,_64),(_16,_512))，SM80_16x8x8_F16F16F16F16_TN的CLayout是Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>, Stride<Stride<_32,_1>,Stride<_16,_8>>>，表示有32个线程，每个线程对应四个元素。所以组合后的结果是：

(((_4,_8),(_2,_2)),(_4,_8)):(((_128,_1),(_64,_8)),(_16,_512))。

第一维(_4,_8),(_2,_2)表明有32个线程，每个线程对应4个元素，第二维(_4,_8)表示在整个ctensor中需要重复4*8次atom。通过threadIdx对第一个(_4,_8)进行索引即可得到该线程对应的tensor中的数据。

既然得到了tv_layout，最后就是计算当前thread对应的tensor。

```python
    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<1>(thr_layout_vmnk_)),
                                        make_layout(size<2>(thr_layout_vmnk_))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile); // ((ThrV,(ThrM,ThrN)),(FrgV,(RestM,RestN)))
```

thr_tile的结果是(_,(_2:_1,_2:_1))，tv_tensor是(((_4,_8),(_2,_2)),(_4,_8)):(((_128,_1),(_64,_8)),(_16,_512))。zipped_divide等于是把((_4,_8),(_2,_2)) / _，得到((_4,_8),(_2,_2))，等于没除。 (_4,_8) / (_2:_1,_2:_1)，得到((_2,_2),(_4,_2))。按照zipped_divide组合得到的结果是：

(((_4,_8),(_2,_2)),((_2,_2),(_2,_4))):(((_128,_1),(_16,_512)),((_64,_8),(_32,_1024)))。

可以与logical_divide的结果做一个对比。

tv_layout: (((_4,_8),(_2,_2)),(_4,_8)):(((_128,_1),(_64,_8)),(_16,_512))

logical_divide:(((_4,_8),(_2,_2)),((_2,_2),(_2,_4))):(((_128,_1),(_64,_8)),((_16,_32),(_512,_1024)))

zipped_divide:(((_4,_8),(_2,_2)),((_2,_2),(_2,_4))):(((_128,_1),(_16,_512)),((_64,_8),(_32,_1024)))

通过这种方法把thread放到第一维，通过对thread索引就可以得到对应的value。

### thrfrg_A

### thrfrg_B

### get_slice

### get_thread_slice

这两个函数用于返回TiledMMA中某个线程的布局，也就是ThrMMA。

```cpp
  template <class ThrIdx,
            __CUTE_REQUIRES(is_integral<ThrIdx>::value)>
  CUTE_HOST_DEVICE constexpr
  auto
  get_slice(ThrIdx const& thr_idx) const
  {
    auto thr_vmnk = thr_layout_vmnk_.get_flat_coord(thr_idx); // 获取当前线程在TiledMMA线程中的坐标
    return ThrMMA<TiledMMA, decltype(thr_vmnk)>{*this, thr_vmnk};
  }

  template <class ThrIdx,
            __CUTE_REQUIRES(is_integral<ThrIdx>::value)>
  CUTE_HOST_DEVICE constexpr
  auto
  get_thread_slice(ThrIdx const& thr_idx) const
  {
    return get_slice(thr_idx);
  }
```

```cpp
    auto l1 = Layout<Shape<_32>>{};
    auto l2 = Layout<Shape<_2, _2, _2>>{};
    auto res = tiled_product(l1, l2); // ((_32),_2,_2,_2):((_1),_32,_64,_128)
    auto flat_res = res.get_flat_coord(100); // (4,1,1,0)

    auto l1 = Layout<Shape<_32,_2>>{};
    auto l2 = Layout<Shape<_2, _2, _2>>{};
    auto res = tiled_product(l1, l2); // ((_32,_2),_2,_2,_2):((_1,_32),_64,_128,_256)
    auto flat_res = res.get_flat_coord(100); // (36,1,0,0)
```

### permutation_mnk

计算tiledmma在MNK三个方向上的大小。如果没有提供PermutationMNK参数则会根据AtomShape_MNK和get_thr_layout_vmnk计算。

```cpp
  // The permutation applied to the MNK-mode data
  template <int I>
  CUTE_HOST_DEVICE constexpr
  auto
  permutation_mnk() const {
    static_assert(0 <= I && I < 3);
    auto perm = get<I>(PermutationMNK{});
    return conditional_return(is_underscore<decltype(perm)>{}, size<I>(AtomShape_MNK{}) * size<I+1>(get_thr_layout_vmnk()), perm);
  }
```

### tile_size_mnk

返回tiledmma MNK三个维度的大小。

```cpp
  // The size of the MNK-mode
  template <int I>
  CUTE_HOST_DEVICE constexpr
  auto
  tile_size_mnk() const {
    static_assert(0 <= I && I < 3);
    return size(permutation_mnk<I>());
  }
```

### get_layoutC_MN

```cpp
  CUTE_HOST_DEVICE constexpr
  auto
  get_layoutC_MN() const
  {
    // (M,N) -> (M,N)
    auto ref_C = make_layout(make_shape(tile_size_mnk<0>(), tile_size_mnk<1>()));
    // (cthrid,val) -> (M,N)
    auto layoutC_TV = thrfrg_C(ref_C);
    // (M,N) -> (cthrid,frg)
    auto layoutC_MN = right_inverse(layoutC_TV).with_shape(shape(ref_C));

    // cthrid = (v,m,n) -> thr_idx
    auto thrID_C = thr_layout_vmnk_(_,_,_,Int<0>{});

    return cute::make_tuple(layoutC_MN, thrID_C);
  }
```

### get_layoutC_TV

### get_layoutA_MK

### get_layoutA_TV

### get_layoutB_NK

### get_layoutB_TV

# ThrMMA

```cpp
template <class TiledMMA, class ThrVMNK>
struct ThrMMA : TiledMMA
{
  ThrVMNK thr_vmnk_; // 当前线程在TiledMMA线程中flat的坐标
  ...
}
```

## partition_C

对Tensor C进行分块，得到当前线程对应的sub Tensor。

```cpp
  template <class CTensor>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_C(CTensor&& ctensor) const
  {
    auto thr_tensor = make_tensor(static_cast<CTensor&&>(ctensor).data(), this->thrfrg_C(ctensor.layout()));

    auto thr_vmn = make_coord(get<0>(thr_vmnk_), make_coord(get<1>(thr_vmnk_), get<2>(thr_vmnk_)));
    return thr_tensor(thr_vmn, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  }
```

首先是一个make_tensor，第一个参数是tensor C的指针，第二个参数就是线程对应的数据的layout，通过thrfrg_C得到。

```python
auto thr_tensor = make_tensor(static_cast<CTensor&&>(ctensor).data(), this->thrfrg_C(ctensor.layout()));
```

thrfrg_C的具体计算过程参考上面。

然后是

```python
    auto thr_vmn = make_coord(get<0>(thr_vmnk_), make_coord(get<1>(thr_vmnk_), get<2>(thr_vmnk_)));
```

也就是获取当前thread在tilledmma中的坐标。前面提到thr_vmn是一个flatten的坐标，第一维是atom坐标，后面分别是mnk的坐标。因此get<0>(thr_vmnk_)等于获取当前线程在atom线程中的位置，后面两个是获取在mn中的位置。

有了当前线程对应的坐标后就能获取当前线程对应的数据的布局了。

rank<1,1>(thr_tensor)是thr_tensor布局的第一维的第一维的rank。如果thr_tensor是(((_4,_8),(_2,_2)),((_2,_2),(_2,_4))):(((_128,_1),(_16,_512)),((_64,_8),(_32,_1024)))，则第一维是((_2,_2),(_2,_4))，第一维的第一维是(_2,_4)。所以rank = 2。

repeat<2>(_) = (_,_)，

make_coord(_, repeat<rank<1,1>(thr_tensor)>(_))这一堆的意思就是获取thr_tensor第一维的所有数据。

总结：partition_C是ThrMMA中的一个函数，通过thrfrg_C将原始tensor转换成在mma atom下的thread-value Layout，并通过当前thread在tiledmma中的坐标获取原始tensor中对应数据的Layout。这样每个线程都能分到原始tensor对应的数据了。

## partition_A

partition_A和partition_C的相同，只不过处理的是mma在矩阵A上的数据。计算流程一直，维度是MK。

## partition_B

partition_B的作用和partition_C相同，处理的是mma在矩阵B上的数据，计算流程一样，维度是NK。

## partition_fragment_C

partition_fragment_C是先调用partition_C再调用make_fragment_C。make_fragment的作用是在寄存器上创建tensor。

```python
  template <class CTensor>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_fragment_C(CTensor&& ctensor) const
  {
    return TiledMMA::make_fragment_C(partition_C(ctensor));
  }
```

## partition_fragment_A

同partition_fragment_C。

## partition_fragment_B

同partition_fragment_C。

# utils

## make_tiled_mma

make_tiled_mma用来生成tiledmma。接受三个参数，分别是MMA_Atom/MMA_Op，MMAThrLayout 和Permutations。如果第一个参数是MMA_Op 则在函数内部会创建对应的MMA_Atom。

第一个参数就是一个mma atom，第二个参数是atom需要按照什么布局复制，第三个参数是复制时遵循什么规则。

```cpp
//
// These tile the MMA_Atom as a whole
//

template <class MMA_Op,
          class MMAThrLayout = Layout<Shape<_1,_1,_1>>,
          class Permutations = Tile<Underscore,Underscore,Underscore>>
CUTE_HOST_DEVICE constexpr
auto
make_tiled_mma(MMA_Atom<MMA_Op> const& mma_atom,
               MMAThrLayout     const& thr_layout   = {},
               Permutations     const& permutations = {})
{
  auto thr_layout_mnk  = append<3>(thr_layout, Layout<_1,_0>{}); // thr_layout需要是MNK三维的，如果不是就通过Layout<_1,_0>{}补到三维。
  auto permutation_mnk = append<3>(permutations, _); // permutations也需要是三维的。

  // 创建一个TiledMMA类并返回
  return TiledMMA<MMA_Atom<MMA_Op>,
                  decltype(thr_layout_mnk),
                  decltype(permutation_mnk)>{mma_atom, thr_layout_mnk};
}

template <class MMA_Op,
          class MMAThrLayout = Layout<Shape<_1,_1,_1>>,
          class Permutations = Tile<Underscore,Underscore,Underscore>>
CUTE_HOST_DEVICE constexpr
auto
make_tiled_mma(MMA_Op       const&,
               MMAThrLayout const& thr_layout   = {},
               Permutations const& permutations = {})
{
  // Attempt to wrap in an MMA_Atom<> and forward
  return make_tiled_mma(MMA_Atom<MMA_Op>{}, thr_layout, permutations);
}
```

## partition_shape_C

## partition_fragment_C

## partition_shape_A

## partition_shape_B

## size

## tile_size

## tile_shape

## size

## thr_size

# Issues

ccecka

[https://github.com/NVIDIA/cutlass/discussions/1867#discussioncomment-10930513](https://github.com/NVIDIA/cutlass/discussions/1867#discussioncomment-10930513)

[https://github.com/NVIDIA/cutlass/discussions/1846#discussioncomment-10800757](https://github.com/NVIDIA/cutlass/discussions/1846#discussioncomment-10800757)

[https://github.com/NVIDIA/cutlass/discussions/1770#discussioncomment-10535033](https://github.com/NVIDIA/cutlass/discussions/1770#discussioncomment-10535033)

[https://github.com/NVIDIA/cutlass/discussions/1432#discussioncomment-8935019](https://github.com/NVIDIA/cutlass/discussions/1432#discussioncomment-8935019)

[https://github.com/NVIDIA/cutlass/discussions/1381#discussioncomment-8689317](https://github.com/NVIDIA/cutlass/discussions/1381#discussioncomment-8689317)

[https://github.com/NVIDIA/cutlass/discussions/1345#discussioncomment-8485429](https://github.com/NVIDIA/cutlass/discussions/1345#discussioncomment-8485429)

[https://github.com/NVIDIA/cutlass/discussions/1271#discussioncomment-7851221](https://github.com/NVIDIA/cutlass/discussions/1271#discussioncomment-7851221)

[https://github.com/NVIDIA/cutlass/discussions/1142](https://github.com/NVIDIA/cutlass/discussions/1142)

[https://github.com/NVIDIA/cutlass/discussions/933#discussioncomment-5782775](https://github.com/NVIDIA/cutlass/discussions/933#discussioncomment-5782775)

