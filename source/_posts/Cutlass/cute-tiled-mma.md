---
title: CuTe 学习笔记（六）CuTe tiledMMA
date: 2025-03-23 18:00:00
tags: [CUTLASS, CuTe, MMA]
categories: [Cutlass 学习笔记, CuTe]
description: 本文基于 CUTLASS 4.5.0 源码，深入分析 `include/cute/atom` 目录中 MMA（Matrix Multiply-Accumulate）相关组件的设计与实现，涵盖 TiledMMA 的创建、ThrMMA、partition 操作等。
published: true
mathjax: true
---

# CuTe tiledMMA

本文从最底层的 PTX 指令封装开始，逐层向上剖析 MMA Operation → MMA Traits → MMA Atom → TiledMMA → ThrMMA 的完整抽象链条，并以 Ampere（SM80）和 Hopper（SM90）架构为例进行说明。


## 1. 整体架构概览

CuTe 对 MMA 的抽象可以看作一个五层金字塔，自底向上依次为：

* 第一层，MMA Operation，PTX 指令封装：DRegisters, ARegisters, fma()。
* 第二层，MMA_Traits，布局语义：Shape_MNK, ThrID, ALayout, BLayout, CLayout。
* 第三层，MMA_Atom，可调用的原子 MMA：call(), make_fragment_*。
* 第四层，TiledMMA，平铺后的 MMA：thrfrg_A/B/C, get_layout*。
* 第五层，ThrMMA，某个线程的视角：partition_A/B/C。

设计理念：每一层只关注自己的职责，通过模板参数向上传递信息。底层负责硬件指令的封装，中间层负责数据布局的描述，上层负责多线程的平铺和分区。

| 层级 | 源文件 | 核心职责 |
|:------:|:--------:|:---------:|
| MMA Operation | `include/cute/arch/mma_sm80.hpp` 等 | 封装 PTX 内联汇编 |
| MMA Traits | `include/cute/atom/mma_traits*.hpp` | 描述线程-值布局映射 |
| MMA Atom | `include/cute/atom/mma_atom.hpp` | 提供调用接口和片段构造 |
| TiledMMA | `include/cute/atom/mma_atom.hpp` | 将 Atom 在 MNK 方向平铺 |
| ThrMMA | `include/cute/atom/mma_atom.hpp` | 单线程的分区与执行视角 |

## 2. MMA Operation

这里主要是 PTX 指令的 C++ 封装。

### 2.1 基本结构

MMA Operation 是整个抽象的最底层。它直接封装了 GPU 的 PTX 指令，将其表示为一个 C++ 结构体。每个 Operation 结构体主要包含以下几个成员：

- **`DRegisters`**：输出 D 的寄存器数组类型
- **`ARegisters`**：输入 A 的寄存器数组类型
- **`BRegisters`**：输入 B 的寄存器数组类型
- **`CRegisters`**：累加器 C 的寄存器数组类型
- **`fma()`**：静态成员函数，执行 `D = A * B + C`

### 2.2 Ampere 架构示例（SM80）

以 Ampere 架构的 `mma.sync.m16n8k16` 指令为例，对应的 PTX 为：

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%0, %1, %2, %3},     // D: 4个float输出
    {%4, %5, %6, %7},     // A: 4个uint32输入（每个含2个f16）
    {%8, %9},             // B: 2个uint32输入（每个含2个f16）
    {%10, %11, %12, %13}; // C: 4个float累加器
```

CUTLASS 将其封装为（见 `include/cute/arch/mma_sm80.hpp:158-186`）：

```cpp
struct SM80_16x8x16_F32F16F16F32_TN
{
  using DRegisters = float[4];      // 4 个 float 寄存器
  using ARegisters = uint32_t[4];   // 4 个 32-bit 寄存器（每个打包2个f16）
  using BRegisters = uint32_t[2];   // 2 个 32-bit 寄存器
  using CRegisters = float[4];      // 4 个 float 累加器

  CUTE_HOST_DEVICE static void
  fma(float& d0, float& d1, float& d2, float& d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const& c0, float const& c1, float const& c2, float const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM80_16x8x16_F32F16F16F32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};
```

命名规则：`SM{架构}_{M}x{N}x{K}_{D类型}{A类型}{B类型}{C类型}_{布局}`
- `SM80`：Ampere 架构
- `16x8x16`：M=16, N=8, K=16 的 MMA 形状
- `F32F16F16F32`：D=float32, A=float16, B=float16, C=float32
- `TN`：线程布局，A 为行优先（T = row-major/Transpose），B 为列优先（N = col-major）


该指令由一个 warp（32 个线程）协作执行，详见 [mma 指令介绍](/2026/04/14/PTX/mma/#m16n8k16)。

每个线程持有 A 的一部分：4 个 `uint32_t`，每个打包 2 个 `half_t`，共 8 个元素，32 个线程 × 8 元素 = 256 = 16×16（M×K）。

每个线程持有 B 的一部分：2 个 `uint32_t`，共 4 个 `half_t`，32 个线程 × 4 元素 = 128 = 8×16（N×K）。

每个线程持有 D/C 的一部分：4 个 `float`，32 个线程 × 4 元素 = 128 = 16×8（M×N）。

### 2.3 Hopper 架构示例（SM90 WGMMA）

Hopper 架构引入了 Warpgroup MMA（WGMMA），由 4 个 warp（128 个线程）协作执行，且 A/B 可以来自共享内存（Shared Memory），通过描述符（Descriptor）访问。详见 [wgmma 指令](/2026/05/16/PTX/wgmma/)。

```cpp
// 见 include/cute/arch/mma_sm90_gmma.hpp:636
struct MMA_64x128x16_F16F16F16_SS
{
  using DRegisters = void;         // D 寄存器嵌入在 C 寄存器中（原地累加）
  using ARegisters = uint64_t[1];  // 1 个描述符（指向 smem 中的 A）
  using BRegisters = uint64_t[1];  // 1 个描述符（指向 smem 中的 B）
  using CRegisters = uint32_t[32]; // 32 个 32-bit 累加器寄存器

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a, uint64_t const& desc_b,
      uint32_t& d00, uint32_t& d01, /* ... 共32个寄存器 ... */ uint32_t& d31,
      GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One)
  {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    asm volatile(
    "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16 "
      "{%0,  %1,  ...  %31},"
      " %32,"   // desc_a
      " %33,"   // desc_b
      " p,  ... ;\n"  // scale_D 等控制参数
    "}\n"
      : "+r"(d00), "+r"(d01), /* ... */ "+r"(d31)
      : "l"(desc_a), "l"(desc_b), ... );
#endif
  }
};
```

WGMMA 与 MMA 的区别：
1. **线程规模**：128 个线程（4 个 warp = 1 个 warpgroup）协作，而非 32 个线程
2. **数据来源**：A 和 B 可以来自共享内存（`_SS` 后缀）或 A 来自寄存器（`_RS` 后缀）
3. **异步执行**：`wgmma.mma_async` 是异步指令，需要后续 `wgmma.fence` 和 `wgmma.commit_group` 同步
4. **描述符寻址**：A/B 通过 64-bit 描述符访问共享内存，而非直接寄存器传值
5. **更大的 tile**：M 固定为 64，N 可以从 8 到 256

### 2.4 通用 FMA

对于没有专用 MMA 指令的类型，CuTe 提供了 `UniversalFMA`（见 `include/cute/arch/mma.hpp`）：

```cpp
template <class D, class A = D, class B = A, class C = D>
struct UniversalFMA
{
  using DRegisters = D[1];
  using ARegisters = A[1];
  using BRegisters = B[1];
  using CRegisters = C[1];

  CUTE_HOST_DEVICE static constexpr void
  fma(D& d, A const& a, B const& b, C const& c) {
    using cute::fma;
    fma(d, a, b, c);  // 委托给类型的 ADL fma 函数
  }
};
```

这是一个 1×1×1 的标量 FMA，适用于任意类型，作为没有硬件 MMA 支持时的回退方案。

---

## 3. MMA Traits

MMA Traits 在 MMA Operation 的基础上添加布局（Layout）语义。

MMA Operation 只知道执行一次 MMA 需要哪些寄存器，但不知道：这些寄存器中的元素对应矩阵的哪些 (m, n, k) 坐标，32（或128）个线程分别负责哪些元素，输入/输出的逻辑数据类型是什么。

MMA Traits 为了解决这些问题，它为每个 MMA Operation 定义了 [TV-Layout](/2025/03/22/Cutlass/cute-tensor/#线程-值分区（Thread-Value-partitioning）)，也就是线程-值到坐标的映射布局。

MMA Traits 遵循一个 concept（见 `include/cute/atom/mma_traits.hpp:41-61`）：

```cpp
/**
 * concept MMA_Traits
 * {
 *   using ValTypeD =  // D 的逻辑值类型
 *   using ValTypeA =  // A 的逻辑值类型
 *   using ValTypeB =  // B 的逻辑值类型
 *   using ValTypeC =  // C 的逻辑值类型
 *
 *   using FrgTypeA =  // MMA 实际消费的 A 片段类型（可选，默认=ValTypeA）
 *   using FrgTypeB =  // MMA 实际消费的 B 片段类型（可选，默认=ValTypeB）
 *   using FrgTypeC =  // MMA 实际消费的 C 片段类型（可选，默认=ValTypeC）
 *
 *   using Shape_MNK =    // MMA 的逻辑 M×N×K 形状
 *
 *   using ThrID     =    // 逻辑线程 ID 到物理线程索引的映射
 *
 *   using ALayout =      // (tid, vid) -> (m, k) 坐标
 *   using BLayout =      // (tid, vid) -> (n, k) 坐标
 *   using CLayout =      // (tid, vid) -> (m, n) 坐标
 * };
 */
```

| 字段 | 含义 | 示例 |
|:------:|:------:|:------:|
| `ValTypeD/A/B/C` | 矩阵元素的逻辑类型 | `half_t`, `float`, `double` |
| `FrgTypeA/B/C` | MMA 实际消费的片段类型（可覆盖 ValType） | `GMMA::smem_desc<K>`（描述符） |
| `Shape_MNK` | 单次 MMA 的 M×N×K 大小 | `Shape<_16,_8,_16>` |
| `ThrID` | 逻辑线程号 → 物理线程号的映射 | `Layout<_32>`（32 线程恒等映射） |
| `ALayout` | (线程ID, 值ID) → A矩阵的(m,k)坐标 | 见下文 |
| `BLayout` | (线程ID, 值ID) → B矩阵的(n,k)坐标 | |
| `CLayout` | (线程ID, 值ID) → C矩阵的(m,n)坐标 | |

### 3.1 布局的含义：以 SM80 F16 MMA 为例

对于 `SM80_16x8x16_F32F16F16F32_TN`（M=16, N=8, K=16），其 Traits 定义如下（见 `include/cute/atom/mma_traits_sm80.hpp:108-116`）：

```cpp
template <>
struct MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>
     : MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>  // 继承 F16 版本的布局
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;
};
```

其父类 `MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>` 定义了布局（见 `mma_traits_sm80.hpp:77-92`）：

```cpp
template <>
struct MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_16, _8, _16>;
  using ThrID   = Layout<_32>;   // 32线程，tid -> tidx 恒等映射

  // (T32, V8) -> (M16, K16)
  using ALayout = Layout<Shape <Shape < _4, _8>, Shape < _2, _2,  _2>>,
                         Stride<Stride<_32, _1>, Stride<_16, _8, _128>>>;
  // (T32, V4) -> (N8, K16)
  using BLayout = Layout<Shape <Shape < _4, _8>, Shape < _2,  _2>>,
                         Stride<Stride<_16, _1>, Stride< _8, _64>>>;
  // (T32, V4) -> (M16, N8)
  using CLayout = SM80_16x8_Row;
};
```

其中，Shape_MNK 是指令的形状，ThrID 是线程的 Layout。

ALayout，BLayout 和 CLayout 分别是指令在 A，B，C 三个矩阵上的 TV-Layout。

#### ALayout

下面是 `SM80_16x8x16_F16F16F16F16_TN` 指令的 ALayout。

```cpp
  // (T32, V8) -> (M16, K16)
  using ALayout = Layout<Shape <Shape < _4, _8>, Shape < _2, _2,  _2>>,
                         Stride<Stride<_32, _1>, Stride<_16, _8, _128>>>;
```

![mma_m16n8k16_A](/assets/mma/mma_m16n8k16_A.png "mma")

上图是 m16n8k16 指令在矩阵 A 上的 MN 布局，可以看到大小是 16×16，每个线程都负责其中的 8 个元素。下面介绍怎么从 MN-Layout 得到 TV-Layout。

介绍之前要先解释一下 Cutlass 中的逻辑索引。Cutlass 中支持一维的逻辑索引。一个位置的逻辑索引就是该位置按照列主序索引得到的 offset。当使用逻辑索引时，默认是列主序的，也就是默认先从列方向索引。

从前面 TV-Layout 的介绍可以知道，TV-Layout 中，第一维是线程的 Layout，第二维是 value 的 Layout。因为线程数是 32，一个线程负责矩阵 A 中的 8 个元素，所以第一维大小是 32，第二维大小是 8。

然后再看具体的线程间的 Layout。可以看到 T0,T1,T2,T3 的逻辑索引间隔是 32，T0,T4,...,T28 的逻辑索引间隔是 1，所以线程的 layout 就是：Shape < _4, _8>，Stride<_32, _1>。

然后看一个线程对应的元素的 layout。上图中，一个线程有 8 个元素，a0,a1 的逻辑索引是 16。a0,a2 的逻辑索引是 8，a0,a4 的逻辑索引是 128，所以 value 的 Layout 就是：Shape < _2, _2, _2>，Stride<_16, _8, _128>。





#### BLayout

#### CLayout（`SM80_16x8_Row`）：

```cpp
// (T32,V4) -> (M16,N8)
using SM80_16x8_Row = Layout<Shape <Shape < _4, _8>, Shape < _2, _2>>,
                             Stride<Stride<_32, _1>, Stride<_16, _8>>>;
```

这是一个 mma 16x8x16 的 C 矩阵对应的 TV-Layout。

第一个维度 Shape < _4, _8>，Stride<_32, _1> 是线程的 Layout。因为 mma 一共有 32 个线程，所以线程 Layout 的 shape 就是 4×8=32。但是线程间隔又不是连续的，所以又分成了两个维度，分别是 4 和 8。

第二个维度 Shape < _2, _2>，Stride<_16, _8> 是值的 Layout，也就是一个线程处理的 value 的 Layout。在 mma 16x8x16 的指令中，一个线程需要处理 4 个元素，但是 4 个元素也不是连续的，所以就分为了 <2, 2>。

![mma_m16n8k16_C](/assets/mma/mma_m16n8k16_C.png "mma")

上图是 mma 16x8x16 C 矩阵具体的 M×N 布局。从 M×N 布局可以得到上面的 TV-Layout 布局。

Cutlass 中支持一维的逻辑索引。当使用逻辑索引时，默认是列主序的，也就是默认先从列方向索引。

首先看线程的布局，可以看到 T0,T1,T2,T3 的逻辑索引间隔是 32，T0,T4,...,T28 的逻辑索引间隔是 1，所以线程的 layout 就是：Shape < _4, _8>，Stride<_32, _1>。

然后看一个线程对应的元素的 layout。上图中，一个线程有 4 个元素，c0,c1 的逻辑索引是 16。c0,c2 的逻辑索引是 8，所以值的 layout 就是：Shape < _2, _2>，Stride<_16, _8>。

最后 TV-Layout 就是上面 SM80_16x8_Row 的布局。

通过这种方式，再结合 cute 代数运算，当进行 SM80_16x8_Row[tid] 索引时就能得到 tid 线程对应的元素的物理索引了。


### 3.2 SM90 GMMA Traits 的特殊性

Hopper 的 WGMMA Traits（见 `include/cute/atom/mma_traits_sm90_gmma.hpp`）引入了新的模式：

```cpp
template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  // FrgType 覆盖：A/B 不是普通值，而是共享内存描述符！
  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _16>;
  using ThrID   = Layout<_128>;                    // 128 线程（warpgroup）
  using ALayout = GMMA::ABLayout< 64, 16>;         // 描述符布局
  using BLayout = GMMA::ABLayout<128, 16>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;  // 运行时参数
};
```

**关键差异**：
1. **`FrgTypeA/B` = `smem_desc`**：对于 SS（Shared-Shared）模式，A 和 B 的"片段"不是寄存器中的数据，而是指向共享内存的描述符。这意味着 `make_fragment_A` 会返回描述符视图而非数据拷贝。
2. **`ThrID = Layout<_128>`**：128 个线程（一个 warpgroup = 4 个 warp）。
3. **`accumulate_` 成员变量**：Traits 不仅仅是类型定义，还可以携带运行时参数（如是否累加），通过 `with()` 函数修改。

### 3.3 Traits 的 `with()` 函数

某些 Traits 支持 `with()` 方法，用于创建带有不同运行时参数的 Traits 副本：

```cpp
// MMA_Atom 中的 with 转发（mma_atom.hpp:75-81）
template <class... TraitsArgs>
CUTE_HOST_DEVICE
auto
with(TraitsArgs&&... args) const {
  auto traits = Traits::with(static_cast<TraitsArgs&&>(args)...);
  return MMA_Atom<decltype(traits)>{traits};
}
```

例如，SM100 的 Scaled MMA 可以通过 `with()` 修改缩放参数：

```cpp
// 概念示例
auto mma = make_tiled_mma(...);
auto mma_scaled = mma.with(UMMA::ScaleOut::Zero, cute::integral_constant<uint32_t, 2>{});
// 现在 mma_scaled 的 accumulate_ = ScaleOut::Zero, ScaleC = 2
```

### 3.4 `mma_unpack`：从 Traits 到指令调用

`mma_unpack`（见 `mma_traits.hpp:106-151`）是连接 Traits 和 Operation 的桥梁。它的职责是将**按照 Traits 布局组织的张量**重新解释为 **Operation 期望的原始寄存器数组**，然后调用 `fma()`：

```cpp
template <class AnyMMATraits, class... TensorTypes>
CUTE_HOST_DEVICE constexpr
void
mma_unpack(AnyMMATraits const& traits,
           Tensor<TD, DLayout>& D,
           Tensor<TA, ALayout> const& A,
           Tensor<TB, BLayout> const& B,
           Tensor<TC, CLayout> const& C)
{
  static_assert(is_rmem<TD>::value, "D 必须在寄存器中");
  static_assert(is_rmem<TA>::value, "A 必须在寄存器中");
  // ...

  // 从 MMA_Operation 获取寄存器类型和数量
  using MMA_Op   = typename MMA_Op<AnyMMATraits>::type;
  using RegTypeD = typename remove_extent<typename MMA_Op::DRegisters>::type;
  using RegTypeA = typename remove_extent<typename MMA_Op::ARegisters>::type;
  // ...

  // 将张量重新转换为寄存器类型
  Tensor rA = recast<RegTypeA>(A);
  Tensor rB = recast<RegTypeB>(B);
  Tensor rD = recast<RegTypeD>(D);
  Tensor rC = recast<RegTypeC>(C);

  constexpr int RegNumD = extent<typename MMA_Op::DRegisters>::value;
  constexpr int RegNumA = extent<typename MMA_Op::ARegisters>::value;
  // ...

  // 静态断言数量匹配
  CUTE_STATIC_ASSERT_V(size(rA) == Int<RegNumA>{});
  // ...

  // 展开 fma 调用：将 rD[0..RegNumD-1], rA[0..RegNumA-1], ... 作为参数传入
  detail::explode(MMA_Op::fma,
                  rD, make_int_sequence<RegNumD>{},
                  rA, make_int_sequence<RegNumA>{},
                  rB, make_int_sequence<RegNumB>{},
                  rC, make_int_sequence<RegNumC>{});
}
```

**`detail::explode` 的作用**：它类似于 `std::apply`，将一个函数和一系列序列展开为函数调用。例如，如果 `RegNumA=4`，则等价于：

```cpp
MMA_Op::fma(rD[0], rD[1], ...,    // D 寄存器
            rA[0], rA[1], rA[2], rA[3],  // A 寄存器
            rB[0], rB[1],          // B 寄存器
            rC[0], rC[1], ...);    // C 寄存器
```

---

## 4. MMA Atom

### 4.1 MMA_Atom 的定义

`MMA_Atom`（见 `mma_atom.hpp:41-196`）是 Traits 的直接子类，添加了**调用接口**和**片段构造**功能：

```cpp
template <class MMAOperation>
struct MMA_Atom<MMAOperation> : MMA_Atom<MMA_Traits<MMAOperation>>
{};

template <class MMAOperation, class... Args>
struct MMA_Atom<MMA_Traits<MMAOperation, Args...>>
  : MMA_Traits<MMAOperation, Args...>    // 继承所有 Traits 的类型定义
{
  using MMA_Op = MMAOperation;
  using Traits = MMA_Traits<MMAOperation, Args...>;

  // 从 Traits 引入类型别名
  using ValTypeD = typename Traits::ValTypeD;
  using ValTypeA = typename Traits::ValTypeA;
  using ValTypeB = typename Traits::ValTypeB;
  using ValTypeC = typename Traits::ValTypeC;

  using Shape_MNK  = typename Traits::Shape_MNK;
  using ThrID      = typename Traits::ThrID;
  using LayoutC_TV = typename Traits::CLayout;   // TV = Thread-Value
  using LayoutA_TV = typename Traits::ALayout;
  using LayoutB_TV = typename Traits::BLayout;

  // 片段类型（可选，默认 = ValType）
  using FrgTypeD = typename detail::FrgTypeC_or_Default<Traits>::type;
  using FrgTypeA = typename detail::FrgTypeA_or_Default<Traits>::type;
  using FrgTypeB = typename detail::FrgTypeB_or_Default<Traits>::type;
  using FrgTypeC = typename detail::FrgTypeC_or_Default<Traits>::type;
  // ...
};
```

### 4.2 调用接口：`call()`

`MMA_Atom` 提供了两种 `call()` 重载（见 `mma_atom.hpp:88-118`）：

```cpp
// 四参数版本：D = A * B + C（显式提供 D 和 C）
template <class TD, class DLayout, class TA, class ALayout,
          class TB, class BLayout, class TC, class CLayout>
CUTE_HOST_DEVICE constexpr
void
call(Tensor<TD, DLayout>& D,
     Tensor<TA, ALayout> const& A,
     Tensor<TB, BLayout> const& B,
     Tensor<TC, CLayout> const& C) const
{
  static_assert(DLayout::rank == 1, "D 必须是 rank-1 张量");
  static_assert(ALayout::rank == 1, "A 必须是 rank-1 张量");
  // ...
  return mma_unpack(static_cast<Traits const&>(*this), D, A, B, C);
}

// 三参数版本：C = A * B + C（C 既是输入也是输出，复现 C）
template <class TA, class ALayout, class TB, class BLayout, class TC, class CLayout>
CUTE_HOST_DEVICE constexpr
void
call(Tensor<TA, ALayout> const& A,
     Tensor<TB, BLayout> const& B,
     Tensor<TC, CLayout>& C) const
{
  return call(C, A, B, C);  // 转发到四参数版本，D=C
}
```

**为什么要求 rank-1 张量？** 因为 `mma_unpack` 需要将张量 `recast` 为寄存器数组，这要求张量是一维的（展平的寄存器序列）。调用者需要确保 A、B、C 已经按照正确的布局组织成一维张量。

### 4.3 片段构造：`make_fragment_A/B/C`

这三个静态函数（见 `mma_atom.hpp:129-195`）用于从**已分区的张量**构造 MMA 需要的片段：

```cpp
// make_fragment_C：构造累加器片段
template <class CTensor>
CUTE_HOST_DEVICE static constexpr
auto
make_fragment_C(CTensor&& ctensor)
{
  // 检查已分区：rank >= 3 (VMN)，且 V 维大小匹配 LayoutC_TV
  CUTE_STATIC_ASSERT_V(rank(ctensor) >= Int<3>{});  // VMN
  CUTE_STATIC_ASSERT_V(size<0>(ctensor) == size<1>(LayoutC_TV{}));

  // C 比较特殊：累加器类型不必匹配输入/输出类型
  // 直接构造 FrgTypeC 张量
  return make_tensor<FrgTypeC>(shape(ctensor));
}

// make_fragment_A：构造 A 的片段
template <class ATensor>
CUTE_HOST_DEVICE static constexpr
auto
make_fragment_A(ATensor&& atensor)
{
  CUTE_STATIC_ASSERT_V(rank(atensor) >= Int<3>{});  // VMK
  CUTE_STATIC_ASSERT_V(size<0>(atensor) == size<1>(LayoutA_TV{}));

  if constexpr (has_dereference<FrgTypeA>::value) {
    // 如果 FrgTypeA 是视图类型（如 GMMA::smem_desc），直接转发张量
    return make_tensor<FrgTypeA>(static_cast<ATensor&&>(atensor));
  } else {
    // 否则，构造 FrgTypeA 类型的新张量（数据拷贝）
    return make_fragment_like<FrgTypeA>(atensor);
  }
}
```

**`FrgType` vs `ValType` 的关键区别**：
- `ValTypeA` = `half_t`：矩阵 A 的逻辑元素类型
- `FrgTypeA` = `GMMA::smem_desc<K>`：MMA 实际消费的片段类型

对于 SM80 的寄存器 MMA，`FrgTypeA` 默认等于 `ValTypeA`（`half_t`），`make_fragment_A` 返回 `half_t` 张量。

对于 SM90 的 SS WGMMA，`FrgTypeA` = `smem_desc`，`make_fragment_A` 返回描述符张量——它是对共享内存的视图，而非数据拷贝。`has_dereference<smem_desc>` 为 true，走第一个分支。


## 5. TiledMMA

TiledMMA 将 Atom 平铺成更大的计算单元。

### 5.1 TiledMMA 的定义

`TiledMMA`（见 `mma_atom.hpp:208-457`）将一个 `MMA_Atom` 在 M、N、K 三个方向上平铺，形成更大的计算单元：

```cpp
// @tparam MMA_Atom       原子 MMA
// @tparam AtomLayoutMNK  在 MNK 方向上的平铺布局
// @tparam PermutationMNK 在平铺前对 MNK 模式施加的置换
template <class MMA_Atom,
          class AtomLayoutMNK,
          class PermutationMNK = Tile<Underscore, Underscore, Underscore>>
struct TiledMMA : MMA_Atom
{
  using Atom           = MMA_Atom;
  using AtomShape_MNK  = typename MMA_Atom::Shape_MNK;
  using AtomThrID      = typename MMA_Atom::ThrID;
  using AtomLayoutC_TV = typename MMA_Atom::LayoutC_TV;
  using AtomLayoutA_TV = typename MMA_Atom::LayoutA_TV;
  using AtomLayoutB_TV = typename MMA_Atom::LayoutB_TV;

  // 线程布局：(ThrV, ThrM, ThrN, ThrK) -> thread_idx
  using ThrLayoutVMNK = decltype(tiled_product(AtomThrID{}, AtomLayoutMNK{}));
  ThrLayoutVMNK thr_layout_vmnk_;

  CUTE_HOST_DEVICE constexpr
  TiledMMA(MMA_Atom const& mma_atom = {}, AtomLayoutMNK const& thr_layout_mnk = {})
    : MMA_Atom(mma_atom),
      thr_layout_vmnk_(tiled_product(AtomThrID{}, thr_layout_mnk)) {}
  // ...
};
```

**`ThrLayoutVMNK` 的含义**：

`tiled_product(AtomThrID{}, AtomLayoutMNK{})` 将原子内部的线程布局（如 32 线程）与原子间的平铺布局（如 2×2 = 4 个原子）组合，生成一个 4 维布局 `(ThrV, ThrM, ThrN, ThrK) -> thread_idx`：
- **ThrV**：原子内部的线程编号（0\~31 对于 warp MMA，0\~127 对于 warpgroup MMA）
- **ThrM**：M 方向上平铺的原子索引
- **ThrN**：N 方向上平铺的原子索引
- **ThrK**：K 方向上平铺的原子索引

**例如**：`TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>, Layout<Shape<2,4,1>>>` 表示：
- 原子形状 16×8×16
- 在 M 方向平铺 2 个原子，N 方向平铺 4 个原子，K 方向不平铺
- 总形状：M=16×2=32, N=8×4=32, K=16
- 总线程数：32 × 2 × 4 × 1 = 256（即 8 个 warp）

### 5.2 `thrfrg_C`：将 C 张量分区为线程-片段视图

`thrfrg_C`（见 `mma_atom.hpp:249-275`）是 TiledMMA 最核心的函数之一。它将一个 `(M, N)` 的张量转换为 `((ThrV, (ThrM, ThrN)), (FrgV, (RestM, RestN)))` 的线程-片段视图：

```cpp
template <class CTensor>
CUTE_HOST_DEVICE constexpr
auto
thrfrg_C(CTensor&& ctensor) const
{
  CUTE_STATIC_ASSERT_V(rank(ctensor) >= Int<2>{});

  // 第1步：应用 MNK 置换
  // 将 (M,N) 变为 (PermM, PermN)，以便原子平铺与置换后的模式对齐
  auto t_tile = make_tile(permutation_mnk<0>(),   // PermM 的布局
                          permutation_mnk<1>());   // PermN 的布局
  auto t_tensor = logical_divide(ctensor, t_tile);  // (PermM, PermN)

  // 第2步：按原子形状切分
  // 将 (PermM, PermN) 切分为 ((AtomM, AtomN), (RestM, RestN))
  auto c_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),  // 16
                          make_layout(size<1>(AtomShape_MNK{}))); // 8
  auto c_tensor = zipped_divide(t_tensor, c_tile);
  // 结果: ((AtomM, AtomN), (RestM, RestN))

  // 第3步：将原子模式从 (M,N) 转换为 (Thr, Val)
  // 使用 AtomLayoutC_TV 将 AtomM×AtomN 映射为 (ThrV, FrgV)
  auto tv_tensor = c_tensor.compose(AtomLayoutC_TV{}, _);
  // 结果: ((ThrV, FrgV), (RestM, RestN))

  // 第4步：按线程平铺切分
  // 将 RestM×RestN 按 ThrM×ThrN 切分
  auto thr_tile = make_tile(_,
                            make_tile(make_layout(size<1>(thr_layout_vmnk_)),  // ThrM
                                      make_layout(size<2>(thr_layout_vmnk_)))); // ThrN
  auto thr_tensor = zipped_divide(tv_tensor, thr_tile);
  // 结果: ((ThrV, (ThrM, ThrN)), (FrgV, (RestM, RestN)))

  return thr_tensor;
}
```

**四步流程图解**：

```
输入: ctensor 的 layout (M, N)
  │
  ▼ 步骤1: logical_divide + permutation
(PermM, PermN)        -- 应用 MNK 置换
  │
  ▼ 步骤2: zipped_divide by AtomShape
((AtomM, AtomN), (RestM, RestN))   -- 按原子大小切分
  │
  ▼ 步骤3: compose with AtomLayoutC_TV
((ThrV, FrgV), (RestM, RestN))     -- 原子内: (M,N) -> (Thr,Val)
  │
  ▼ 步骤4: zipped_divide by ThrM×ThrN
((ThrV, (ThrM, ThrN)), (FrgV, (RestM, RestN)))  -- 原子间: 按线程平铺切分
```

**最终含义**：
- `ThrV`：原子内的线程编号
- `(ThrM, ThrN)`：原子在 M、N 方向的平铺索引
- `FrgV`：原子内每个线程持有的值编号
- `(RestM, RestN)`：超出平铺范围的剩余部分

`thrfrg_A` 和 `thrfrg_B` 的逻辑完全类似，只是操作的模式从 (M,N) 变为 (M,K) 和 (N,K)，且线程平铺使用 `(ThrM, ThrK)` 和 `(ThrN, ThrK)`。

### 5.3 `permutation_mnk`：MNK 模式置换

```cpp
template <int I>
CUTE_HOST_DEVICE constexpr
auto
permutation_mnk() const {
  static_assert(0 <= I && I < 3);
  auto perm = get<I>(PermutationMNK{});
  // 如果用户指定了 Underscore（未置换），则返回原子大小 × 平铺大小
  return conditional_return(is_underscore<decltype(perm)>{},
                            size<I>(AtomShape_MNK{}) * size<I+1>(get_thr_layout_vmnk()),
                            perm);
}
```

**作用**：返回第 I 个模式（M/N/K）的置换布局。如果用户未指定（使用 `_`），则返回默认的连续布局（大小 = 原子大小 × 平铺大小）。

**为什么需要置换？** 某些 GEMM 布局要求在平铺前对 M/N/K 模式进行重排。例如，对于某些 K-major 的输入，可能需要先对 K 模式进行置换，使得原子 MMA 能正确对齐数据。

### 5.4 `get_layoutC_TV`：获取 (thread_idx, val_idx) → (M,N) 的布局

```cpp
CUTE_HOST_DEVICE constexpr
auto
get_layoutC_TV() const
{
  // 创建参考的 (M,N) 布局
  auto ref_C = make_layout(make_shape(tile_size_mnk<0>(), tile_size_mnk<1>()));

  // 构造 thread_idx -> (ThrV, ThrM, ThrN, ThrK) 的映射
  // 先将 thr_layout_vmnk 的逆布局与 complement 组合
  auto thridx_2_thrid = composition(
      make_layout(make_shape(size(thr_layout_vmnk_), Int<1>{}),
                  make_stride(Int<1>{}, Int<0>{})),
      right_inverse(make_layout(thr_layout_vmnk_, complement(thr_layout_vmnk_))));

  // 对参考布局执行 thrfrg_C，再 compose 线程映射
  // 结果: (thread_idx, val_idx) -> (M, N)
  return thrfrg_C(ref_C).compose(thridx_2_thrid, _);
}
```

**用途**：返回一个布局 `L`，使得 `L(thread_idx, val_idx)` 给出该线程该值对应的 (M, N) 坐标。这对于理解线程-数据映射关系、生成 LaTeX/SVG 可视化非常有用。

`get_layoutA_TV` 和 `get_layoutB_TV` 类似，但额外需要处理 `(ThrV, (ThrM, ThrK))` 到 `(ThrV, (ThrM, ThrN, ThrK))` 的维度扩展（因为 A 的线程布局只涉及 M 和 K，但总线程布局包含 N）。

### 5.5 `get_slice` / `get_thread_slice`：获取单个线程的视角

```cpp
template <class ThrIdx>
CUTE_HOST_DEVICE constexpr
auto
get_slice(ThrIdx const& thr_idx) const
{
  // 将线性线程索引转换为 (ThrV, ThrM, ThrN, ThrK) 坐标
  auto thr_vmnk = thr_layout_vmnk_.get_flat_coord(thr_idx);
  // 返回绑定到该线程的 ThrMMA 对象
  return ThrMMA<TiledMMA, decltype(thr_vmnk)>{*this, thr_vmnk};
}

template <class ThrIdx>
CUTE_HOST_DEVICE constexpr
auto
get_thread_slice(ThrIdx const& thr_idx) const
{
  return get_slice(thr_idx);  // 别名
}
```

---

## 6. ThrMMA

单个线程视角的分区与执行

### 6.1 ThrMMA 的定义

`ThrMMA`（见 `mma_atom.hpp:459-520`）是 TiledMMA 的子类，额外存储了某个线程的 `(ThrV, ThrM, ThrN, ThrK)` 坐标，并提供分区函数：

```cpp
template <class TiledMMA, class ThrVMNK>
struct ThrMMA : TiledMMA
{
  ThrVMNK thr_vmnk_;   // 该线程在 (ThrV, ThrM, ThrN, ThrK) 中的坐标

  // partition_A/B/C ...
  // partition_fragment_A/B/C ...
};
```

### 6.2 `partition_C`：提取本线程负责的 C 片段

```cpp
template <class CTensor>
CUTE_HOST_DEVICE constexpr
auto
partition_C(CTensor&& ctensor) const
{
  // 对张量的布局执行 thrfrg_C
  auto thr_tensor = make_tensor(static_cast<CTensor&&>(ctensor).data(),
                                this->thrfrg_C(ctensor.layout()));
  // thr_tensor: ((ThrV, (ThrM, ThrN)), (FrgV, (RestM, RestN)))

  // 提取本线程的 (ThrV, (ThrM, ThrN)) 坐标
  auto thr_vmn = make_coord(get<0>(thr_vmnk_),               // ThrV
                            make_coord(get<1>(thr_vmnk_),     // ThrM
                                       get<2>(thr_vmnk_)));  // ThrN

  // 用本线程坐标切片，提取 (FrgV, (RestM, RestN))
  return thr_tensor(thr_vmn, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
}
```

**结果**：返回一个 `(FrgV, (RestM, RestN))` 的张量，包含本线程负责的所有 C 元素。其中：
- `FrgV`：原子内本线程持有的值
- `(RestM, RestN)`：本线程在 M、N 方向平铺的剩余部分

### 6.3 `partition_A` 和 `partition_B`

与 `partition_C` 类似，但使用 `thrfrg_A` / `thrfrg_B` 和对应的线程坐标：

```cpp
template <class ATensor>
CUTE_HOST_DEVICE constexpr
auto
partition_A(ATensor&& atensor) const
{
  auto thr_tensor = make_tensor(static_cast<ATensor&&>(atensor).data(),
                                this->thrfrg_A(atensor.layout()));
  // thr_tensor: ((ThrV, (ThrM, ThrK)), (FrgV, (RestM, RestK)))

  auto thr_vmk = make_coord(get<0>(thr_vmnk_),               // ThrV
                            make_coord(get<1>(thr_vmnk_),     // ThrM
                                       get<3>(thr_vmnk_)));  // ThrK
  return thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  // 结果: (FrgV, (RestM, RestK))
}

template <class BTensor>
CUTE_HOST_DEVICE constexpr
auto
partition_B(BTensor&& btensor) const
{
  auto thr_tensor = make_tensor(static_cast<BTensor&&>(btensor).data(),
                                this->thrfrg_B(btensor.layout()));
  auto thr_vnk = make_coord(get<0>(thr_vmnk_),
                            make_coord(get<2>(thr_vmnk_),     // ThrN
                                       get<3>(thr_vmnk_)));  // ThrK
  return thr_tensor(thr_vnk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  // 结果: (FrgV, (RestN, RestK))
}
```

### 6.4 `partition_fragment_C/A/B`：分区 + 构造片段

这些函数组合了 `partition_*` 和 `make_fragment_*`：

```cpp
template <class CTensor>
CUTE_HOST_DEVICE constexpr
auto
partition_fragment_C(CTensor&& ctensor) const
{
  return TiledMMA::make_fragment_C(partition_C(ctensor));
  // 先分区，再用结果构造片段
}

template <class ATensor>
CUTE_HOST_DEVICE constexpr
auto
partition_fragment_A(ATensor&& atensor) const
{
  return TiledMMA::make_fragment_A(partition_A(atensor));
}

template <class BTensor>
CUTE_HOST_DEVICE constexpr
auto
partition_fragment_B(BTensor&& btensor) const
{
  return TiledMMA::make_fragment_B(partition_B(btensor));
}
```

**`partition_C` vs `partition_fragment_C` 的区别**：
- `partition_C`：返回一个**视图**张量，共享输入张量的数据指针（`ctensor.data()`）
- `partition_fragment_C`：返回一个**新片段**张量，具有 `FrgTypeC` 类型和分区后的形状

对于 C（累加器）：`partition_fragment_C` 返回 `make_tensor<FrgTypeC>(shape(partition_C(ctensor)))`，即一个全新的寄存器张量。

对于 A/B：取决于 `FrgTypeA/B` 是视图类型还是值类型，`partition_fragment_A/B` 可能返回数据拷贝或视图。

---

## 7. `make_tiled_mma` 与 `partition_fragment_C`

### 7.1 `make_tiled_mma`

创建 TiledMMA 的函数

`make_tiled_mma`（见 `mma_atom.hpp:526-554`）是创建 `TiledMMA` 的主要入口：

```cpp
// 版本1：接受 MMA_Atom
template <class MMA_Op,
          class MMAThrLayout = Layout<Shape<_1,_1,_1>>,
          class Permutations = Tile<Underscore, Underscore, Underscore>>
CUTE_HOST_DEVICE constexpr
auto
make_tiled_mma(MMA_Atom<MMA_Op> const& mma_atom,
               MMAThrLayout     const& thr_layout   = {},
               Permutations     const& permutations = {})
{
  // 将 thr_layout 补齐为 rank-3（MNK），K 默认为 Layout<_1,_0>（不平铺）
  auto thr_layout_mnk  = append<3>(thr_layout, Layout<_1,_0>{});
  // 将 permutations 补齐为 rank-3
  auto permutation_mnk = append<3>(permutations, _);

  return TiledMMA<MMA_Atom<MMA_Op>,
                  decltype(thr_layout_mnk),
                  decltype(permutation_mnk)>{mma_atom, thr_layout_mnk};
}

// 版本2：接受裸 MMA_Op，自动包装为 MMA_Atom
template <class MMA_Op,
          class MMAThrLayout = Layout<Shape<_1,_1,_1>>,
          class Permutations = Tile<Underscore, Underscore, Underscore>>
CUTE_HOST_DEVICE constexpr
auto
make_tiled_mma(MMA_Op       const&,
               MMAThrLayout const& thr_layout   = {},
               Permutations const& permutations = {})
{
  // 自动包装：MMA_Op -> MMA_Atom<MMA_Op>（内部查找 Traits）
  return make_tiled_mma(MMA_Atom<MMA_Op>{}, thr_layout, permutations);
}
```

**使用方式**：

```cpp
// 基本用法：单原子，不平铺
auto mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{});
// 等价于 TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>, Layout<Shape<1,1,1>>>

// 在 M 和 N 方向平铺
auto mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                          Layout<Shape<_2, _4>>{});  // M×2, N×4
// thr_layout 被补齐为 Layout<Shape<2,4,1>, Stride<4,1,0>>

// 带置换
auto mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                          Layout<Shape<_2, _4>>{},
                          Tile<_1, _1, _>{});  // M和N不置换，K默认
```

### 7.2 `partition_fragment_C`

除了通过 `ThrMMA` 实例调用的 `partition_fragment_C`，还有**静态版本**（不需要线程索引）：

```cpp
// 获取 C 的分区形状（静态，不需线程索引）
template <class... Args, class Shape_MN>
CUTE_HOST_DEVICE constexpr
auto
partition_shape_C(TiledMMA<Args...> const& mma, Shape_MN const& shape_MN)
{
  auto dummy    = make_layout(shape(shape_MN));   // 不分配数据的虚拟布局
  auto dummy_tv = mma.thrfrg_C(dummy);            // 执行 thrfrg_C
  // 模拟 partition_C 的切片操作
  auto dummy_v  = dummy_tv(Int<0>{}, make_coord(_, repeat<rank(dummy)>(_)));
  return shape(dummy_v);
}

// 静态分配累加器片段
template <class... Args, class Shape_MN>
CUTE_HOST_DEVICE constexpr
auto
partition_fragment_C(TiledMMA<Args...> const& mma, Shape_MN const& shapeMN)
{
  return make_tensor<typename TiledMMA<Args...>::FrgTypeC>(
             partition_shape_C(mma, shapeMN));
}
```

**为什么有静态版本？** 累加器 C 的分区只依赖于 TiledMMA 的结构和目标形状，不依赖于具体线程索引（所有线程的 C 片段形状相同）。因此可以在编译期确定形状并分配。

相比之下，`partition_fragment_A/B` 依赖于张量的实际布局和线程索引，不能在静态上下文中使用：

```cpp
// partition_fragment_A 和 partition_fragment_B 通常依赖于
//   A 和 B 的布局和/或请求分区的 thread_idx。
// 因此，它们不应在静态上下文中使用。
// 请使用 TiledMMA::get_slice(thr_idx).partition_fragment_A(tensorA) 代替。
```

### 7.3 辅助尺寸函数

```cpp
// 获取第 I 个模式（M/N/K）的 tile 大小
template <int I, class... Args>
CUTE_HOST_DEVICE constexpr
auto
tile_size(TiledMMA<Args...> const& mma)
{
  return mma.template tile_size_mnk<I>();
}

// 获取完整的 tile 形状 (M, N, K)
template <class... Args>
CUTE_HOST_DEVICE constexpr
auto
tile_shape(TiledMMA<Args...> const& mma)
{
  return make_shape(tile_size<0>(mma), tile_size<1>(mma), tile_size<2>(mma));
}

// 获取线程布局的大小（别名）
template <int... I, class... Args>
CUTE_HOST_DEVICE constexpr
auto
thr_size(TiledMMA<Args...> const& mma)
{
  return size<I...>(mma.get_thr_layout_vmnk());
}
```

---

## 8. 完整使用示例

### 8.1 Ampere 架构：使用 SM80 F16 MMA

```cpp
#include <cute/atom/mma_atom.hpp>
using namespace cute;

// 第1步：创建 TiledMMA
// 使用 16x8x16 的 F32=F16*F16+F32 MMA，在 M 方向平铺2次，N 方向平铺4次
// 总 tile: 32x32x16，总线程: 32 * 2 * 4 = 256（8个warp）
auto mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                          Layout<Shape<_2, _4>>{});

// 第2步：获取本线程的 ThrMMA（假设 thread_idx = 0）
int thread_idx = threadIdx.x;
auto thr_mma = mma.get_slice(thread_idx);

// 第3步：分区 A, B, C（假设已有共享内存张量 sA, sB, sC）
// sA: (M, K) = (32, 16) 的 half_t 张量
// sB: (N, K) = (32, 16) 的 half_t 张量
// sC: (M, N) = (32, 32) 的 float 张量
auto tA = thr_mma.partition_A(sA);    // (FrgV, RestM, RestK)
auto tB = thr_mma.partition_B(sB);    // (FrgV, RestN, RestK)
auto tC = thr_mma.partition_C(sC);    // (FrgV, RestM, RestN)

// 第4步：构造寄存器片段
auto tCrA = thr_mma.make_fragment_A(tA);  // half_t 寄存器片段
auto tCrB = thr_mma.make_fragment_B(tB);  // half_t 寄存器片段
auto tCrC = thr_mma.make_fragment_C(tC);  // float 累加器

// 或者直接使用 partition_fragment_*：
auto tCrA = thr_mma.partition_fragment_A(sA);
auto tCrB = thr_mma.partition_fragment_B(sB);
auto tCrC = thr_mma.partition_fragment_C(sC);

// 第5步：从共享内存加载到寄存器
copy(tA, tCrA);  // smem -> rmem
copy(tB, tCrB);
copy(tC, tCrC);

// 第6步：执行 MMA
// 三参数版本：C = A * B + C（原地累加）
mma.call(tCrA, tCrB, tCrC);
// 或四参数版本：D = A * B + C
// mma.call(tCrD, tCrA, tCrB, tCrC);

// 第7步：将结果存回共享内存
copy(tCrC, tC);  // rmem -> smem
```

### 8.2 Hopper 架构：使用 SM90 WGMMA

```cpp
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
using namespace cute;

// 第1步：通过 ss_op_selector 选择合适的 GMMA
// 自动根据 TileShape 和数据类型选择最优的 WGMMA 指令
using TileShape = Shape<_128, _128, _16>;
auto mma = make_tiled_mma(
    GMMA::ss_op_selector<half_t, half_t, float, TileShape,
                         GMMA::Major::K, GMMA::Major::K>(),
    Layout<Shape<_1, _1, _1>>{});  // WGMMA 通常不再平铺（已经是 warpgroup 级别）

// 第2步：获取本线程的 ThrMMA
auto thr_mma = mma.get_slice(threadIdx.x);

// 第3步：分区
// 对于 SS 模式，A/B 的片段是 smem_desc（描述符），不是数据
auto tCrA = thr_mma.partition_fragment_A(sA);  // smem_desc 张量
auto tCrB = thr_mma.partition_fragment_B(sB);  // smem_desc 张量
auto tCrC = thr_mma.partition_fragment_C(sC);  // float 累加器

// 第4步：构造 GMMA 描述符（从共享内存张量）
auto desc_A = make_gmma_desc<GMMA::Major::K>(sA);  // uint64_t 描述符
auto desc_B = make_gmma_desc<GMMA::Major::K>(sB);

// 第5步：执行 WGMMA（异步）
// 注意：WGMMA 是异步的，需要 fence 和 commit
warpgroup_fence_operand(tCrC);  // 操作前 fence
mma.call(tCrA, tCrB, tCrC);      // 发起异步 wgmma.mma_async
warpgroup_arrive();               // 等待完成
warpgroup_commit_batch();         // 提交批次
warpgroup_wait<0>();              // 等待结果
warpgroup_fence_operand(tCrC);   // 操作后 fence
```

### 8.3 使用 `with()` 修改运行时参数

```cpp
// 创建一个默认的 scaled MMA
auto mma = make_tiled_mma(SM100_MMA_F16BF16_SS_SCALED<...>{});

// 修改 accumulate 行为：设为 Zero（不累加，直接覆盖 D）
auto mma_no_accum = mma.with(UMMA::ScaleOut::Zero);

// 修改 scale 参数
auto mma_scaled = mma.with(UMMA::ScaleOut::One,
                           cute::integral_constant<uint32_t, 2>{});  // ScaleC = 2
```

---

## 9. 总结

### 9.1 分层抽象的价值

CUTLASS CuTe 的 MMA 抽象体现了**关注点分离**的设计原则：

| 层级 | 关注点 | 可替换性 |
|------|--------|---------|
| MMA Operation | 硬件指令的精确编码 | 每种 PTX 指令一个结构体 |
| MMA Traits | 数据布局的数学描述 | 同一 Operation 可有多种 Traits |
| MMA Atom | 调用接口与片段构造 | 统一接口，底层可替换 |
| TiledMMA | 多线程平铺策略 | 通过 AtomLayoutMNK 参数化 |
| ThrMMA | 单线程的局部视角 | 自动从 TiledMMA 派生 |

### 9.2 类型驱动的编译期优化

整个抽象链条几乎完全在编译期完成：
- **类型计算**：`MMA_Op::DRegisters` → `RegTypeD` → `recast` → `fma` 调用
- **布局计算**：`ALayout`, `BLayout`, `CLayout` 是 CuTe Layout 类型，编译期可组合
- **静态断言**：寄存器数量、张量秩等在编译期验证

这意味着，例如，从 SM80 的 warp MMA 切换到 SM90 的 warpgroup MMA，只需更改 `make_tiled_mma` 的参数，上层代码（分区、调用）的结构保持不变。

### 9.3 CuTe Layout 的核心作用

CuTe Layout 是贯穿整个抽象的核心工具。一个 Layout `L: (索引空间) -> (坐标空间)` 完全描述了数据如何组织和访问：

- **`ThrID`**：`Layout<32>` 表示 32 线程的恒等映射
- **`ALayout`**：`(tid, vid) -> (m, k)` 编码了哪个线程的哪个寄存器对应 A 矩阵的哪个元素
- **`ThrLayoutVMNK`**：`(ThrV, ThrM, ThrN, ThrK) -> thread_idx` 编码了线程如何组织为平铺结构

通过 Layout 的组合（`composition`）、切分（`zipped_divide`）、逻辑除法（`logical_divide`）等操作，`thrfrg_*` 函数能够将任意形状的张量映射到正确的线程-片段视图。

### 9.4 从 Ampere 到 Hopper 的演进

| 特性 | Ampere (SM80) | Hopper (SM90) |
|------|--------------|--------------|
| 执行单元 | Warp (32线程) | Warpgroup (128线程) |
| 数据来源 | 寄存器 | 共享内存（描述符）或寄存器 |
| 执行模型 | 同步 `mma.sync` | 异步 `wgmma.mma_async` |
| ThrID | `Layout<_32>` | `Layout<_128>` |
| FrgTypeA/B | 默认=ValType | `smem_desc`（SS模式） |
| Traits 成员 | 纯类型定义 | 可含运行时参数（`accumulate_`） |
| `with()` | 通常不使用 | 用于设置 Scale 等参数 |

这种演进体现了 CuTe 抽象的可扩展性——新的硬件特性可以通过扩展 Traits 的字段和 Operation 的接口来支持，而上层 TiledMMA/ThrMMA 的结构保持稳定。

### 9.5 关键文件索引

| 文件 | 内容 |
|------|------|
| `include/cute/arch/mma.hpp` | `UniversalFMA` 通用回退 |
| `include/cute/arch/mma_sm80.hpp` | Ampere warp MMA PTX 封装 |
| `include/cute/arch/mma_sm90.hpp` | Hopper warp MMA + WGMMA PTX 封装 |
| `include/cute/arch/mma_sm90_gmma.hpp` | WGMMA 指令封装（SS/RS 模式） |
| `include/cute/atom/mma_traits.hpp` | `MMA_Traits` 概念 + `mma_unpack` |
| `include/cute/atom/mma_traits_sm80.hpp` | Ampere MMA 的 Traits 特化 |
| `include/cute/atom/mma_traits_sm90.hpp` | Hopper warp MMA 的 Traits |
| `include/cute/atom/mma_traits_sm90_gmma.hpp` | WGMMA 的 Traits + GMMA 布局 |
| `include/cute/atom/mma_atom.hpp` | `MMA_Atom`, `TiledMMA`, `ThrMMA`, `make_tiled_mma` |
| `include/cute/atom/partitioner.hpp` | 通用 `TV_Tiler` 分区器 |

---

> **参考资料**：
> - CUTLASS 4.5.0 源码：`include/cute/atom/` 目录
> - NVIDIA PTX ISA 文档：`mma.sync` 和 `wgmma.mma_async` 指令说明
> - CuTe Layout 系统：`include/cute/layout.hpp`
>