---
title: CuTe 学习笔记（八）CuTe tilledCOPY
date: 2025-03-25 18:00:00
tags: [CUTLASS, CuTe, Copy]
categories: [Cutlass 学习笔记, CuTe]
description: 本文基于 CUTLASS 4.5.0 源码，深入分析 `include/cute/atom` 目录中 Copy（数据拷贝）相关组件的设计与实现。
published: true
mathjax: true
---



# TiledCopy

本文从最底层的 PTX 指令封装开始，逐层向上剖析 Copy Operation → Copy Traits → Copy Atom → TiledCopy → ThrCopy 的完整抽象链条，并以 Ampere（SM80 `cp.async`）、Turing（SM75 `ldmatrix`）和 Hopper（SM90 TMA）架构为例进行说明。


## 1. 整体架构

CuTe 对 Copy 的抽象与 MMA 非常相似，同样采用五层金字塔结构，自底向上依次为：

* Copy Operation，PTX 指令封装：SRegisters, DRegisters, copy()。
* Copy_Traits，位级布局：ThrID, SrcLayout, DstLayout, RefLayout。
* Copy_Atom，可调用的原子 Copy：call(), with()。
* TiledCopy，平铺后的 Copy：tidfrg_S/D, get_layoutS/D_TV。
* ThrCopy，某个线程的视角：partition_S/D, retile_S/D。


与 MMA 的区别：

| 特性 | MMA | Copy |
|:------:|:-----:|:------:|
| 操作语义 | D = A × B + C | D ← S（数据搬运） |
| 操作数数量 | 4 个（A, B, C, D） | 2 个（Src, Dst） |
| 布局单位 | 坐标 (m, n, k) | **比特（bit）** |
| 布局数量 | 3 个（ALayout, BLayout, CLayout） | 3 个（SrcLayout, DstLayout, **RefLayout**） |
| 片段类型 | FrgTypeA/B/C | 单一 **ValType**（用户指定） |
| 寄存器类型 | DRegisters, ARegisters, BRegisters, CRegisters | **SRegisters, DRegisters** |

Copy 独有的 RefLayout 概念：Copy 引入了"参考布局"（RefLayout），用于在 Src 和 Dst 的线程-值映射之间建立桥梁。这在 Src 和 Dst 有不同线程分布时至关重要（如 `ldmatrix` 指令中，源数据在共享内存按一种分布排列，而目标寄存器按另一种分布排列）。

| 层级 | 源文件 | 核心职责 |
|:------:|:-----:|:------:|
| Copy Operation | `include/cute/arch/copy*.hpp` | 封装 PTX 内联汇编 |
| Copy Traits | `include/cute/atom/copy_traits*.hpp` | 描述线程-值到位的映射 |
| Copy Atom | `include/cute/atom/copy_atom.hpp` | 提供调用接口和 ValType 参数化 |
| TiledCopy | `include/cute/atom/copy_atom.hpp` | 将 Atom 在 MN 方向平铺 |
| ThrCopy | `include/cute/atom/copy_atom.hpp` | 单线程的分区与执行视角 |


## 2. Copy Operation

PTX 指令的 C++ 封装。

### 2.1 基本结构

Copy Operation 是最底层的抽象，直接封装 PTX 指令。每个 Operation 结构体包含：

- **`SRegisters`**：源（Source）的寄存器数组类型
- **`DRegisters`**：目标（Destination）的寄存器数组类型
- **`copy()`**：静态成员函数，执行数据搬运

### 2.2 通用回退：UniversalCopy

对于普通的标量拷贝，CuTe 提供了 `UniversalCopy`（见 `include/cute/arch/copy.hpp:45-61`）：

```cpp
template <class S, class D = S>
struct UniversalCopy
{
  using SRegisters = S[1];   // 1 个源寄存器
  using DRegisters = D[1];   // 1 个目标寄存器

  static_assert(sizeof_bits_v<S> >= 8);
  static_assert(sizeof_bits_v<D> >= 8);

  CUTE_HOST_DEVICE static constexpr void
  copy(S const& src, D& dst)
  {
    dst = src;   // 简单赋值
  }
};

// 自动向量化拷贝（假设最大 128-bit 对齐）
using AutoVectorizingCopy = AutoVectorizingCopyWithAssumedAlignment<128>;
// 默认拷贝（不假设对齐）
using DefaultCopy = AutoVectorizingCopyWithAssumedAlignment<8>;
```

**`AutoVectorizingCopyWithAssumedAlignment<MaxVecBits>`** 是一个特殊的占位类型。它本身不执行具体指令，而是告诉 CuTe 的 copy 算法："可以假设指针和布局对齐到 MaxVecBits，请自动选择最优的向量化拷贝方式"。CuTe 会根据张量的实际布局，自动将这种"通用拷贝"展开为合适宽度的 `UniversalCopy<uint128_t>` 等指令。

### 2.3 Ampere 架构：cp.async（SM80）

Ampere 引入了 `cp.async` 指令，可以直接从全局内存异步拷贝到共享内存，无需经过寄存器中转（见 `include/cute/arch/copy_sm80.hpp:46-70`）：

```cpp
/// cp.async.ca: 全缓存级别的异步拷贝
template <class TS, class TD = TS>
struct SM80_CP_ASYNC_CACHEALWAYS
{
  using SRegisters = TS[1];   // 1 个源值（gmem）
  using DRegisters = TD[1];   // 1 个目标值（smem）

  static_assert(sizeof(TS) == sizeof(TD), "源和目标大小必须相同");
  static_assert(sizeof(TS) == 4 || sizeof(TS) == 8 || sizeof(TS) == 16,
                "cp.async 支持 4/8/16 字节");

  CUTE_HOST_DEVICE static void
  copy(TS const& gmem_src, TD& smem_dst)
  {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    TS const* gmem_ptr = &gmem_src;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
        :: "r"(smem_int_ptr),    // 共享内存地址
           "l"(gmem_ptr),        // 全局内存地址
           "n"(sizeof(TS)));     // 拷贝字节数（编译期常量）
#else
    CUTE_INVALID_CONTROL_PATH("...");
#endif
  }
};

/// cp.async.cg: 全局缓存级别的异步拷贝（仅 16 字节）
template <class TS, class TD = TS>
struct SM80_CP_ASYNC_CACHEGLOBAL
{
  using SRegisters = TS[1];
  using DRegisters = TD[1];

  static_assert(sizeof(TS) == 16, "cp.async.cg 仅支持 16 字节");

  CUTE_HOST_DEVICE static void
  copy(TS const& gmem_src, TD& smem_dst)
  {
    // ... cp.async.cg.shared.global.L2::128B ...
  }
};
```

**`cp.async` 的关键特性**：
- **异步执行**：发起后不阻塞，需要后续 `cp.async.commit_group` 和 `cp.async.wait_group` 同步
- **绕过寄存器**：数据直接从 gmem 到 smem，不占用寄存器
- **缓存策略**：`.ca` 缓存在所有级别，`.cg` 只缓存在全局级别（适合只读一次的数据）
- **ZFILL 变体**：支持条件拷贝，predicate 为 false 时用零填充

**ZFILL 变体**（见 `copy_sm80.hpp:100-157`）：

```cpp
template <class TS, class TD = TS>
struct SM80_CP_ASYNC_CACHEALWAYS_ZFILL
{
  using SRegisters = TS[1];
  using DRegisters = TD[1];

  CUTE_HOST_DEVICE static void
  copy(TS const& gmem_src, TD& smem_dst, bool pred)
  {
    int src_size = pred ? sizeof(TS) : 0;
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n"
        :: "r"(smem_int_ptr), "l"(gmem_ptr), "n"(sizeof(TS)), "r"(src_size));
    //                                                    ^^^^^^^^^^^^^^^
    //                    当 pred=false 时 src_size=0，硬件用零填充
  }
};
```

配套的同步指令封装：

```cpp
// 提交一组 cp.async 操作
CUTE_HOST_DEVICE void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// 等待，直到最多 N 组未完成
template <int N>
CUTE_HOST_DEVICE void cp_async_wait() {
  if constexpr (N == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
  }
}
```

### 2.4 Turing 架构：ldmatrix（SM75）

`ldmatrix` 是 Turing 架构引入的 warp 级矩阵加载指令，专门为 MMA 优化——它能按照 MMA 所需的寄存器分布从共享内存加载数据（见 `include/cute/arch/copy_sm75.hpp:81-99`）：

```cpp
struct SM75_U32x1_LDSM_N
{
  using SRegisters = uint128_t[1];  // 1 个 128-bit 源（smem 地址）
  using DRegisters = uint32_t[1];   // 1 个 32-bit 目标（寄存器）

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src, uint32_t& dst)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
        : "=r"(dst)
        :  "r"(smem_int_ptr));
#endif
  }
};

// x2 变体：加载 2 个 8x8 矩阵
struct SM75_U32x2_LDSM_N
{
  using SRegisters = uint128_t[1];  // 1 个 128-bit 源地址
  using DRegisters = uint32_t[2];   // 2 个 32-bit 目标

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src, uint32_t& dst0, uint32_t& dst1)
  {
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst0), "=r"(dst1)
        :  "r"(smem_int_ptr));
  }
};
```

**`ldmatrix` 的关键特性**：
- **warp 协作**：32 个线程协作，每个线程提供一个 smem 地址，硬件按照 8×8 矩阵的布局将数据分发到各线程的寄存器
- **`x1/x2/x4`**：分别加载 1/2/4 个 8×8 矩阵
- **`.N` vs `.T`**：`.N`（normal）和 `.T`（transpose）变体控制加载时的转置行为
- **与 MMA 对齐**：`ldmatrix` 的输出分布正好匹配 `mma.sync` 指令对 A/B 矩阵的寄存器分布要求

### 2.5 Hopper 架构：TMA（SM90）

Hopper 架构引入了 **TMA（Tensor Memory Accelerator）**，这是对 copy 抽象的最大变革。TMA 通过描述符（TensorMap）描述一个多维张量，硬件自动处理地址计算、边界检查和 swizzle（见 `include/cute/arch/copy_sm90_tma.hpp:47-101`）：

```cpp
struct SM90_TMA_LOAD_1D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,    // TMA 描述符（TensorMap）
       uint64_t* mbar_ptr,      // mbarrier 指针（用于异步同步）
       uint64_t cache_hint,     // L2 缓存提示
       void* smem_ptr,          // 共享内存目标地址
       int32_t const& crd0)     // 坐标（1D: 只有 crd0）
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3}], [%2], %4;"
      :
      : "r"(smem_int_ptr),       // smem 目标
        "l"(gmem_int_desc),      // TMA 描述符
        "r"(smem_int_mbar),      // mbarrier
        "r"(crd0),               // 张量坐标
        "l"(cache_hint)          // 缓存提示
      : "memory");
#endif
  }

  // TMA 还支持 prefetch（预取到 L2）
  struct PREFETCH {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr, int32_t const& crd0) {
      asm volatile(
        "cp.async.bulk.prefetch.tensor.1d.L2.global [%0, {%1}];"
        : : "l"(gmem_int_desc), "r"(crd0) : "memory");
    }
  };
};
```

**TMA 的关键特性**：
1. **描述符驱动**：通过 `TmaDescriptor`（TensorMap）描述全局内存中的张量形状、步长和 swizzle 模式，硬件自动计算地址
2. **多维支持**：1D/2D/3D/4D/5D 变体，坐标通过参数传入
3. **mbarrier 同步**：与共享内存屏障（mbarrier）配合，通过 `complete_tx::bytes` 机制自动通知拷贝完成
4. **边界安全**：硬件自动处理越界访问（OOB），无需软件 predication
5. **单线程发起**：`ThrID = Layout<_1>`，只需一个线程发起 TMA 指令
6. **多播支持**：`SM90_TMA_LOAD_MULTICAST` 可以将数据同时拷贝到多个 SM 的共享内存

**TMA Store**（smem → gmem）类似，但不需要 mbarrier：

```cpp
struct SM90_TMA_STORE {
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, void const* smem_ptr,
       int32_t const& crd0, int32_t const& crd1, ...) {
    asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta [%0, {%2, %3}], [%1];"
      : : "l"(gmem_int_desc), "r"(smem_int_ptr),
          "r"(crd0), "r"(crd1) : "memory");
  }
};
```

**TMA Reduce**（smem → gmem 带归约）：

```cpp
struct SM90_TMA_REDUCE_ADD {
  // 将 smem 中的数据加到 gmem 中（原子归约）
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, void const* smem_ptr,
       int32_t const& crd0, int32_t const& crd1, ...) {
    asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.add ..."
      ...);
  }
};
```

### 2.6 Hopper 架构：Bulk Copy（SM90）

除了 TMA，Hopper 还引入了 `cp.async.bulk`（Bulk Copy），用于大块数据的异步拷贝，不依赖 TMA 描述符（见 `copy_traits_sm90_tma.hpp:548-637`）：

```cpp
// G2S: Global to Shared
struct SM90_BULK_COPY_G2S {
  CUTE_HOST_DEVICE static void
  copy(void const* gmem_ptr, uint64_t* mbar_ptr,
       void* smem_ptr, int32_t num_bytes) {
    asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];"
      : : "r"(smem_int_ptr), "l"(gmem_int_ptr),
          "r"(num_bytes), "r"(smem_int_mbar) : "memory");
  }
};

// S2G: Shared to Global
struct SM90_BULK_COPY_S2G {
  CUTE_HOST_DEVICE static void
  copy(void const* smem_ptr, void* gmem_ptr, int32_t num_bytes) {
    asm volatile(
      "cp.async.bulk.global.shared::cta [%0], [%1], %2;"
      : : "l"(gmem_int_ptr), "r"(smem_int_ptr),
          "r"(num_bytes) : "memory");
  }
};
```

---

## 3. Copy Traits

为 Operation 注入位级布局语义。

### 3.1 为什么需要 Traits？为什么用"比特"？

Copy Operation 只知道"执行一次 copy 需要哪些寄存器"，但**不知道**：
- 这些寄存器中的元素如何映射到源/目标张量的坐标？
- 多个线程如何协作完成拷贝？
- 源和目标的线程-值映射不同时如何转换？

**Copy Traits** 的职责就是回答这些问题——它为每个 Copy Operation 定义了**线程-值到比特（bit）的映射布局**。

**为什么用比特而不是坐标？** 因为 Copy 需要处理各种位宽的数据（4-bit, 8-bit, 16-bit, 32-bit, 64-bit, 128-bit），用比特作为通用单位可以统一描述。Copy Atom 会通过 `recast_layout` 将位级布局转换为实际值类型的布局。

### 3.2 Copy Traits 的概念接口

Copy Traits 遵循一个 concept（见 `include/cute/atom/copy_traits.hpp:40-57`）：

```cpp
/**
 * concept Copy_Traits
 * {
 *   using ThrID     =    // 逻辑线程 ID -> 物理线程索引
 *
 *   using SrcLayout =    // (src-tid, src-vid) -> bit
 *   using DstLayout =    // (dst-tid, dst-vid) -> bit
 *   using RefLayout =    // (ref-tid, ref-vid) -> bit   <-- 关键！
 * };
 *
 * Copy_Traits 的抽象比特序是任意的，仅用于构造映射：
 *   (ref-tid, ref-vid) -> (src-tid, src-vid)
 *   (ref-tid, ref-vid) -> (dst-tid, dst-vid)
 * TiledCopy 中的 Layout_TV 遵循 RefLayout，按需映射到 Src 或 Dst 的 (tid,vid) 表示。
 */
```

**三个布局的含义**：

| 布局 | 含义 | 用途 |
|------|------|------|
| `SrcLayout` | (源线程, 源值) → 比特 | 描述源数据的线程-值分布 |
| `DstLayout` | (目标线程, 目标值) → 比特 | 描述目标数据的线程-值分布 |
| `RefLayout` | (参考线程, 参考值) → 比特 | **桥梁**：TiledCopy 的标准表示 |

**RefLayout 的作用**：当 Src 和 Dst 的线程-值分布不同时（如 `ldmatrix`），需要一个"参考"分布作为中间桥梁。TiledCopy 内部使用 RefLayout 坐标，然后在执行时映射到 SrcLayout 或 DstLayout。通常 `RefLayout = SrcLayout`（以源为参考），但也可以是 DstLayout。

### 3.3 UniversalCopy 的 Traits

最简单的 Traits（见 `copy_traits.hpp:65-78`）：

```cpp
template <class S, class D>
struct Copy_Traits<UniversalCopy<S,D>>
{
  using ThrID = Layout<_1>;   // 单线程

  // (src-thr, src-val) -> bit: 1 个线程, sizeof(S) 个比特
  using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<S>::value>>>;
  // (dst-thr, dst-val) -> bit: 1 个线程, sizeof(D) 个比特
  using DstLayout = Layout<Shape<_1, Int<sizeof_bits<D>::value>>>;

  // 参考布局 = 源布局
  using RefLayout = SrcLayout;
};
```

**解读**：对于 `UniversalCopy<half_t, half_t>`，`SrcLayout = Layout<Shape<_1, _16>>`，即 1 个线程拷贝 16 个比特（一个 half）。

### 3.4 SM80 cp.async 的 Traits

`cp.async` 的 Traits 与 UniversalCopy 类似，因为每条指令只涉及一个线程拷贝一个值（见 `copy_traits_sm80.hpp:41-54`）：

```cpp
template <class S, class D>
struct Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<S,D>>
{
  using ThrID = Layout<_1>;   // 单线程

  // (src-thr, src-val) -> bit: 1 个线程, sizeof(S) 个比特
  using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<S>::value>>>;
  // (dst-thr, dst-val) -> bit: 1 个线程, sizeof(D) 个比特
  using DstLayout = Layout<Shape<_1, Int<sizeof_bits<D>::value>>>;

  using RefLayout = SrcLayout;
};
```

**ZFILL 变体的特殊处理**（见 `copy_traits_sm80.hpp:71-117`）：

ZFILL 变体引入了运行时参数 `pred`（布尔值，控制是否零填充），并重载了 `copy_unpack`：

```cpp
template <class S, class D>
struct Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<S,D>>
{
  using ThrID = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<S>::value>>>;
  using DstLayout = Layout<Shape<_1, Int<sizeof_bits<D>::value>>>;
  using RefLayout = SrcLayout;

  // 运行时参数
  bool pred = true;

  // 通过 with() 构造带特定 pred 值的 Traits
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<S,D>>
  with(bool pred) const {
    return {pred};
  }

  // 重载 copy_unpack，将 pred 传入指令
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>& dst)
  {
    static_assert(is_gmem<TS>::value, "Expected gmem source for cp.async.");
    static_assert(is_smem<TD>::value, "Expected smem destination for cp.async.");

    Tensor rS = recast<S>(src);
    Tensor rD = recast<D>(dst);

    CUTE_STATIC_ASSERT_V(size(rS) == Int<1>{});
    CUTE_STATIC_ASSERT_V(size(rD) == Int<1>{});

    // 关键：将 traits.pred 传入 copy 函数
    SM80_CP_ASYNC_CACHEALWAYS_ZFILL<S,D>::copy(rS[0], rD[0], traits.pred);
  }
};
```

**注意**：ZFILL 变体没有使用通用的 `copy_unpack`（通过 `detail::explode` 展开），而是直接在 Traits 中通过 friend 函数重载了 `copy_unpack`。这是因为 ZFILL 需要传递额外的 `pred` 参数。

### 3.5 SM75 ldmatrix 的 Traits：多线程布局

`ldmatrix` 是 warp 级指令，32 个线程协作，其 Traits 描述了复杂的线程-值分布（见 `copy_traits_sm75.hpp:41-56`）：

```cpp
template <>
struct Copy_Traits<SM75_U32x1_LDSM_N>
{
  using ThrID = Layout<_32>;   // 32 线程（一个 warp）

  // (src-thr, src-val) -> bit: (32, 1) 个 (thr,val) 对
  // 源是共享内存，线程按 8x4 分布，每个线程提供 1 个 128-bit 地址
  using SrcLayout = Layout<Shape <Shape <  _8, _4>, _128>,
                           Stride<Stride<_128, _0>,  _1>>;
  // 解读: Shape((8,4), 128)
  //   - 线程维度: (8,4) = 32 个线程
  //   - 值维度: 128 bits = 16 bytes = 1 个 uint128_t
  //   - Stride: (128, 0), 1
  //   - 线程 0 -> bit 0, 线程 1 -> bit 128, ..., 线程 8 -> bit 0 (但 val 不同)

  // (dst-thr, dst-val) -> bit: (32, 32) 个 (thr,val) 对
  // 目标是寄存器，每个线程获得 1 个 32-bit 寄存器
  using DstLayout = Layout<Shape <_32, _32>,
                           Stride<_32,  _1>>;
  // 解读: 32 个线程，每个线程 32 bits = 1 个 uint32_t

  // 参考布局 = 目标布局
  using RefLayout = DstLayout;
};
```

**关键观察**：
- `SrcLayout` 的线程维度是 `(8, 4)`，表示 32 个线程在源（共享内存）中按 8×4 网格排列
- `DstLayout` 的线程维度是 `32`，表示目标（寄存器）中线程是线性排列
- `RefLayout = DstLayout`：以目标（寄存器）分布为参考

**`x2` 变体**（加载 2 个 8×8 矩阵）：

```cpp
template <>
struct Copy_Traits<SM75_U32x2_LDSM_N>
{
  using ThrID = Layout<_32>;

  // 源: 32 线程，按 (16,2) 排列，每个线程 128 bits
  using SrcLayout = Layout<Shape <Shape < _16, _2>, _128>,
                           Stride<Stride<_128, _0>,  _1>>;
  // 目标: 32 线程，每个线程 2×32 = 64 bits（2 个 uint32_t）
  using DstLayout = Layout<Shape <_32, Shape <_32,   _2>>,
                           Stride<_32, Stride< _1, _1024>>>;
  using RefLayout = DstLayout;
};
```

### 3.6 SM90 TMA 的 Traits：描述符驱动的布局

TMA 的 Traits 是最复杂的，因为 TMA 指令通过描述符和坐标参数化，而非直接的数据指针（见 `copy_traits_sm90_tma.hpp:100-203`）：

```cpp
// 不可执行的 SM90_TMA_LOAD（有 tma_desc，但没有 tma_mbar）
// 必须通过 .with(tma_mbar) 构造可执行版本
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM90_TMA_LOAD, NumBitsPerTMA, AuxParams_>
{
  using ThrID     = Layout<_1>;   // 单线程发起
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;  // (1, NumBits) -> bit
  using DstLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using RefLayout = SrcLayout;

  // TMA 描述符（TensorMap），存储在 Traits 中
  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;   // 辅助参数（步长、swizzle 等）

  // 获取 TMA 描述符
  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const* get_tma_descriptor() const {
    return &tma_desc_;
  }

  // 构造可执行版本：需要 mbarrier
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD_OP, NumBitsPerTMA>
  with(uint64_t& tma_mbar,
       uint16_t const& multicast_mask = 0,
       TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL) const {
    return {&tma_desc_, &tma_mbar, static_cast<uint64_t>(cache_hint)};
  }

  // 生成 TMA 坐标张量
  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  // 禁止直接执行：必须先调用 .with()
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits const&, Tensor<TS,SLayout> const&, Tensor<TD,DLayout>&) = delete;
};
```

**关键设计**：
1. **两阶段构造**：`SM90_TMA_LOAD`（不可执行，有描述符无 mbarrier）→ `SM90_TMA_LOAD_OP`（可执行，有描述符和 mbarrier）
2. **`with()` 方法**：绑定 mbarrier 后返回可执行的 Traits
3. **`get_tma_tensor()`**：根据全局形状生成坐标张量，用于后续分区
4. **`NumBitsPerTMA`**：模板参数，表示单次 TMA 拷贝的比特数（如 1024 = 128 字节）

**可执行版本的 `copy_unpack`**（见 `copy_traits_sm90_tma.hpp:65-92`）：

```cpp
template <class CopyOp, class... Args>
struct TMA_LOAD_Unpack
{
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>& dst)
  {
    static_assert(is_smem<TD>::value, "SM90_TMA_LOAD requires the destination be shared memory.");

    // src 张量存储的是 TMA 坐标
    auto src_coord = src(Int<0>{});
    void* dst_ptr = cute::raw_pointer_cast(dst.data());

    // 将参数展开并调用底层 PTX 指令
    return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                 traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                 make_tuple(dst_ptr), seq<0>{},
                                 src_coord, tuple_seq<decltype(src_coord)>{});
  }
};
```

**TMA 的 SrcLayout/DstLayout 为什么是 `Layout<Shape<_1, NumBitsPerTMA>>`？** 因为 TMA 是"单线程"指令——一个线程发起拷贝，硬件处理所有数据搬运。`NumBitsPerTMA` 表示这次 TMA 搬运的总比特数（如 128 字节 = 1024 比特）。线程-值映射的复杂性被隐藏在 TMA 描述符中。

### 3.7 `copy_unpack`：从 Traits 到指令调用

通用 `copy_unpack`（见 `copy_traits.hpp:108-136`）与 MMA 的 `mma_unpack` 类似：

```cpp
template <class AnyCPYTraits, class... TensorTypes>
CUTE_HOST_DEVICE constexpr
void
copy_unpack(AnyCPYTraits const& traits,
            Tensor<SEngine,SLayout> const& src,
            Tensor<DEngine,DLayout>& dst)
{
  using CopyOp       = typename CPY_Op<AnyCPYTraits>::type;
  using RegistersSrc = typename CopyOp::SRegisters;
  using RegistersDst = typename CopyOp::DRegisters;
  using RegTypeSrc   = typename remove_extent<RegistersSrc>::type;
  using RegTypeDst   = typename remove_extent<RegistersDst>::type;
  constexpr int RegNumSrc = extent<RegistersSrc>::value;
  constexpr int RegNumDst = extent<RegistersDst>::value;

  // 将张量重新转换为寄存器类型
  Tensor rS = recast<RegTypeSrc>(src);
  Tensor rD = recast<RegTypeDst>(dst);

  // 静态断言数量匹配
  CUTE_STATIC_ASSERT_V(size(rS) == Int<RegNumSrc>{});
  CUTE_STATIC_ASSERT_V(size(rD) == Int<RegNumDst>{});

  // 展开调用 CopyOp::copy(rS[0], rS[1], ..., rD[0], rD[1], ...)
  detail::explode(detail::CallCOPY<CopyOp>{},
                  rS, make_int_sequence<RegNumSrc>{},
                  rD, make_int_sequence<RegNumDst>{});
}
```

**`detail::CallCOPY<CopyOp>`** 是一个函数对象，调用 `CopyOp::copy(...)`。`detail::explode` 将寄存器数组展开为函数参数。

**特殊重载**：某些 Traits（如 ZFILL、TMA）通过 friend 函数直接重载 `copy_unpack`，绕过通用版本，以传递额外的运行时参数（如 `pred`、`tma_mbar`）。

## 4. Copy Atom

### 4.1 Copy_Atom 的定义

`Copy_Atom`（见 `copy_atom.hpp:44-176`）与 `MMA_Atom` 类似，但有一个关键区别：**Copy_Atom 需要额外的 `CopyInternalType` 参数**。

```cpp
template <class CopyOperation, class CopyInternalType>
struct Copy_Atom<CopyOperation, CopyInternalType>
  : Copy_Atom<Copy_Traits<CopyOperation>, CopyInternalType>
{};

template <class... Args, class CopyInternalType>
struct Copy_Atom<Copy_Traits<Args...>, CopyInternalType>
  : Copy_Traits<Args...>
{
  using Traits = Copy_Traits<Args...>;

  // 从 Traits 引入位级布局
  using ThrID        = typename Traits::ThrID;
  using BitLayoutSrc = typename Traits::SrcLayout;
  using BitLayoutDst = typename Traits::DstLayout;
  using BitLayoutRef = typename Traits::RefLayout;

  // 用户指定的值类型（如 half_t, float, uint8_t）
  using ValType = CopyInternalType;

  // 将位级布局转换为值级布局（通过 recast_layout）
  using ValLayoutSrc = decltype(recast_layout<uint1_t, ValType>(BitLayoutSrc{}));
  using ValLayoutDst = decltype(recast_layout<uint1_t, ValType>(BitLayoutDst{}));
  using ValLayoutRef = decltype(recast_layout<uint1_t, ValType>(BitLayoutRef{}));

  // 静态断言线程数匹配
  CUTE_STATIC_ASSERT_V(size<0>(ValLayoutSrc{}) == size(ThrID{}));
  CUTE_STATIC_ASSERT_V(size<0>(ValLayoutDst{}) == size(ThrID{}));
  CUTE_STATIC_ASSERT_V(size<0>(ValLayoutRef{}) == size(ThrID{}));

  // 每个原子拷贝的源/目标值数量
  static constexpr int NumValSrc = size<1>(ValLayoutSrc{});
  static constexpr int NumValDst = size<1>(ValLayoutDst{});

  // with() 方法
  template <class... TraitsArgs>
  CUTE_HOST_DEVICE
  auto
  with(TraitsArgs&&... args) const {
    auto traits = Traits::with(static_cast<TraitsArgs&&>(args)...);
    return Copy_Atom<decltype(traits), CopyInternalType>{traits};
  }
  // ...
};
```

**为什么需要 `CopyInternalType`？** Copy Traits 描述的是位级布局，但实际拷贝的数据有具体类型（如 `half_t`、`float`）。`CopyInternalType` 告诉 Atom "我们在拷贝什么类型的数据"，从而将位级布局 `recast` 为值级布局。

例如：
- `Copy_Atom<UniversalCopy<uint128_t>, half_t>`：用 128-bit 通用拷贝，但解释为 `half_t`（一次拷贝 8 个 half）
- `Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>`：用 `cp.async` 拷贝 128-bit，解释为 8 个 half

**`recast_layout<uint1_t, ValType>` 的作用**：将比特布局重新解释为 `ValType` 布局。例如，对于 `ValType = half_t`（16-bit），`Layout<Shape<_1, _128>>`（128 比特）会变为 `Layout<Shape<_1, _8>>`（8 个 half_t）。

### 4.2 调用接口：`call()`

`Copy_Atom` 提供了两种 `call()` 重载（见 `copy_atom.hpp:90-175`）：

```cpp
// 两参数版本：src -> dst（无条件拷贝）
template <class SEngine, class SLayout, class DEngine, class DLayout>
CUTE_HOST_DEVICE
void
call(Tensor<SEngine,SLayout> const& src,
     Tensor<DEngine,DLayout>& dst) const
{
  static_assert(SLayout::rank == 1, "Expected rank-1 src tensor");
  static_assert(DLayout::rank == 1, "Expected rank-1 dst tensor");

  // 检查 src/dst 大小是否匹配指令要求
  if constexpr (is_constant<NumValSrc, decltype(size(src))>::value ||
                is_constant<NumValDst, decltype(size(dst))>::value) {
    // 大小匹配，执行指令
    return copy_unpack(static_cast<Traits const&>(*this), src, dst);
  } else if constexpr (is_tuple<decltype(shape(src))>::value &&
                       is_tuple<decltype(shape(dst))>::value) {
    // 大小不匹配但形状是 tuple，递归剥离外层模式
    // ((A,B,C,...)) -> (A,B,C,...)
    return copy(*this, tensor<0>(src), tensor<0>(dst));
  } else {
    static_assert(dependent_false<SEngine>,
                  "CopyAtom: Src/Dst partitioning does not match the instruction requirement.");
  }
}
```

**递归剥离模式**：当传入的张量大小与指令要求不匹配，但形状是嵌套 tuple 时，`call` 会递归地剥离外层模式。这是一种常见的模式——分区后的张量可能有 `((V, M, N))` 的形状，需要先剥离外层，露出 `(V, M, N)`，然后对整个序列调用 `copy`。

**带谓词的三参数版本**（见 `copy_atom.hpp:128-162`）：

```cpp
// 三参数版本：pred ? src -> dst : (zfill 或 不拷贝)
template <class PEngine, class PLayout,
          class SEngine, class SLayout,
          class DEngine, class DLayout>
CUTE_HOST_DEVICE
void
call(Tensor<PEngine,PLayout> const& prd,   // 谓词张量
     Tensor<SEngine,SLayout> const& src,
     Tensor<DEngine,DLayout>& dst) const
{
  static_assert(PLayout::rank == 1, "Expected rank-1 prd tensor");
  static_assert(SLayout::rank == 1, "Expected rank-1 src tensor");
  static_assert(DLayout::rank == 1, "Expected rank-1 dst tensor");

  if constexpr (is_constant<NumValSrc, decltype(size(src))>::value ||
                is_constant<NumValDst, decltype(size(dst))>::value) {
    Traits const& traits = static_cast<Traits const&>(*this);
    // 检查 Traits 是否支持 with(bool)
    auto has_with_bool = cute::is_valid([](auto t)->void_t<decltype(t.with(true))>{}, traits);
    if constexpr (has_with_bool) {
      // 支持 ZFILL：用 with(pred[0]) 构造带谓词的 Traits
      copy_unpack(traits.with(prd(Int<0>{})), src, dst);
    } else {
      // 不支持 ZFILL：运行时条件判断
      if (prd(Int<0>{})) { copy_unpack(traits, src, dst); }
    }
  } else if constexpr (...) {
    // 递归剥离
    return copy_if(*this, tensor<0>(prd), tensor<0>(src), tensor<0>(dst));
  }
}
```

**谓词处理的两种路径**：
1. **Traits 支持 `with(bool)`**（如 ZFILL 变体）：通过 `with(pred)` 构造一个带谓词的 Traits 副本，然后调用 `copy_unpack`。这样谓词被编译进指令（如 `cp.async` 的 ZFILL 模式）。
2. **Traits 不支持 `with(bool)`**：运行时 `if (pred)` 判断，不满足时跳过拷贝。


## 5. TiledCopy

将 Atom 平铺成更大的拷贝单元。

### 5.1 TiledCopy 的定义

`TiledCopy`（见 `copy_atom.hpp:185-355`）将一个 `Copy_Atom` 与线程-值布局（TV Layout）和切片器（Tiler）组合：

```cpp
template <class Copy_Atom,
          class LayoutCopy_TV,  // (tid,vid) -> coord
          class ShapeTiler_MN>  // coord space
struct TiledCopy : Copy_Atom
{
  // 从 Atom 引入信息
  using AtomThrID     = typename Copy_Atom::ThrID;        // thrid -> thr_idx
  using AtomLayoutSrc = typename Copy_Atom::ValLayoutSrc; // (thr,val) -> offset
  using AtomLayoutDst = typename Copy_Atom::ValLayoutDst;
  using AtomLayoutRef = typename Copy_Atom::ValLayoutRef;

  using AtomNumThr = decltype(size<0>(AtomLayoutRef{}));  // 原子内线程数
  using AtomNumVal = decltype(size<1>(AtomLayoutRef{}));  // 原子内值数

  // 平铺后的信息
  using Tiler_MN       = ShapeTiler_MN;
  using TiledLayout_TV = LayoutCopy_TV;
  using TiledNumThr    = decltype(size<0>(TiledLayout_TV{}));  // 总线程数
  using TiledNumVal    = decltype(size<1>(TiledLayout_TV{}));  // 总值数

  // 断言：平铺后的线程/值数必须是原子线程/值数的整数倍
  CUTE_STATIC_ASSERT_V(TiledNumThr{} % AtomNumThr{} == Int<0>{});
  CUTE_STATIC_ASSERT_V(TiledNumVal{} % AtomNumVal{} == Int<0>{});
  // ...
};
```

**与 TiledMMA 的关键区别**：
- TiledMMA 通过 `AtomLayoutMNK` 参数自动计算 `ThrLayoutVMNK`
- TiledCopy **直接接受** `LayoutCopy_TV` 和 `ShapeTiler_MN`，由用户（或工厂函数）提供

### 5.2 `tidfrg_S` / `tidfrg_D`：线程-片段分区

这两个函数（见 `copy_atom.hpp:218-281`）是 TiledCopy 的核心，将张量转换为线程-片段视图：

```cpp
// 将源张量分区为 (Thr, (FrgV, FrgX), (RestM, RestN, ...))
template <class STensor>
CUTE_HOST_DEVICE constexpr static
auto
tidfrg_S(STensor&& stensor)
{
  CUTE_STATIC_ASSERT_V(rank(stensor) >= rank(Tiler_MN{}));

  // 第1步：用 Tiler 切分张量
  // (M,N,...) -> ((TileM,TileN,...), (RestM,RestN,...))
  auto tiled = zipped_divide(stensor, Tiler_MN{});

  // 第2步：构造 Ref -> Src 的映射
  // right_inverse(AtomLayoutRef) 将 Ref 的 (thr,val) 逆映射为线性索引
  // .compose(AtomLayoutSrc) 再映射到 Src 的 (thr,val)
  auto ref2src = right_inverse(AtomLayoutRef{}).compose(AtomLayoutSrc{});

  // 第3步：调用 tile2thrfrg 完成转换
  return tile2thrfrg(tiled, ref2src);
}
```

**`tile2thrfrg` 的四步流程**（见 `copy_atom.hpp:254-281`）：

```cpp
template <class Tensor, class Ref2TrgLayout>
CUTE_HOST_DEVICE constexpr static
auto
tile2thrfrg(Tensor&& tensor, Ref2TrgLayout const& ref2trg)
{
  // 第1步：将 TiledLayout_TV 按原子大小切分
  // (tid,vid) -> (m,n) 切分为 ((atom_tid, atom_val), (rest_tid, rest_val)) -> (m,n)
  auto atom_layout_TV = zipped_divide(TiledLayout_TV{}, make_shape(AtomNumThr{}, AtomNumVal{}));

  // 第2步：应用 Ref -> Trg 映射（Src 或 Dst）
  // 将参考布局转换为实际的源/目标布局
  auto trg_layout_TV = atom_layout_TV.compose(ref2trg, _);
  // ((trg_tid, trg_val), (rest_tid, rest_val)) -> (m,n)

  // 第3步：重组维度，将线程和值分开
  auto thrval2mn = coalesce(zip(trg_layout_TV), Shape<_1, Shape<_1,_1>>{});
  // ((trg_tid, rest_tid), (trg_val, rest_val)) -> (m,n)

  // 第4步：应用到张量
  auto tv_tensor = tensor.compose(thrval2mn, _);
  // ((thrid, val), (RestM, RestN, ...))

  // 展开并返回
  return tv_tensor(make_coord(_,_), _);
  // (Thr, (FrgV, FrgX), (RestM, RestN, ...))
}
```

**最终结果** `(Thr, (FrgV, FrgX), (RestM, RestN, ...))`：
- **Thr**：逻辑线程 ID
- **FrgV**：原子内每个线程的值（对应 AtomLayoutRef 的 val 维度）
- **FrgX**：原子间每个线程的值（平铺产生的额外值）
- **RestM, RestN**：超出 Tiler 范围的剩余部分

### 5.3 `retile`：重新切片

`retile`（见 `copy_atom.hpp:284-316`）用于将已按某种方式切片的张量重新切片为 Copy_Atom 所需的布局：

```cpp
template <class Tensor>
CUTE_HOST_DEVICE constexpr static
auto
retile(Tensor&& tensor)
{
  constexpr int R = remove_cvref_t<Tensor>::rank;

  auto V = size<0>(tensor);   // 当前片段的值数

  // 第1步：找到 V 个值在 TiledLayout_TV 中的布局
  auto frg_layout_mn = upcast<TiledNumThr{} * V>(
      right_inverse(TiledLayout_TV{}).with_shape(shape(Tiler_MN{})));
  // (m,n) -> v_idx

  // 第2步：按原子值数切分
  auto frg_layout_v = zipped_divide(
      logical_product(make_layout(V), right_inverse(frg_layout_mn)),
      make_layout(AtomNumVal{}));
  // (atom_vals, rest_vals) -> (v, m, n)

  // 第3步：切分张量
  auto t_tensor = zipped_divide(tensor, prepend(product_each(shape(frg_layout_mn)), V));
  // ((TileV, TileM, TileN, ...), (1, RestM, RestN, ...))

  // 第4步：应用布局
  auto v_tensor = t_tensor.compose(frg_layout_v, _);
  // ((atom_vals, rest_vals), (1, RM, RN, ...))

  // 展开并返回
  return v_tensor(_, append<R>(Int<0>{}, _));
}
```

**`retile` 的用途**：当数据已经按照某种方式分布在线程中（如 MMA 的寄存器分布），但需要用不同的 Copy_Atom 重新拷贝时，`retile` 可以将数据重新组织为 Copy_Atom 所需的分布。典型场景是 epilogue（后处理）阶段，将 MMA 输出的寄存器数据重新组织后存回全局内存。

### 5.4 `get_layoutS_TV` / `get_layoutD_TV`：获取布局

```cpp
CUTE_HOST_DEVICE constexpr static
auto
get_layoutS_TV()
{
  // 创建参考布局：(M,N) -> (M,N)
  auto ref_S = make_layout(make_shape(shape(Tiler_MN{}), Int<1>{}));
  // 通过 tile2thrfrg 转换，然后切片到 (thr_idx, val_idx) -> (M,N)
  return tile2thrfrg(ref_S, right_inverse(AtomLayoutRef{}).compose(AtomLayoutSrc{}))(_,_,Int<0>{});
}

CUTE_HOST_DEVICE constexpr static
auto
get_layoutD_TV()
{
  auto ref_D = make_layout(make_shape(shape(Tiler_MN{}), Int<1>{}));
  return tile2thrfrg(ref_D, right_inverse(AtomLayoutRef{}).compose(AtomLayoutDst{}))(_,_,Int<0>{});
}
```

**用途**：返回 `(thread_idx, val_idx) -> (M, N)` 的布局，描述哪个线程的哪个值对应源/目标张量的哪个坐标。与 TiledMMA 的 `get_layoutC_TV` 类似，用于可视化和与其他组件对接。

### 5.5 `get_slice` / `get_thread_slice`：获取单线程视角

```cpp
template <class ThrIdx>
CUTE_HOST_DEVICE static
auto
get_slice(ThrIdx const& thr_idx)
{
  return ThrCopy<TiledCopy, ThrIdx>(thr_idx);
}

template <class ThrIdx>
CUTE_HOST_DEVICE static
auto
get_thread_slice(ThrIdx const& thr_idx)
{
  return get_slice(thr_idx);   // 别名
}
```

## 6. ThrCopy

单个线程视角的分区与执行。

### 6.1 ThrCopy 的定义

`ThrCopy`（见 `copy_atom.hpp:357-402`）比 `ThrMMA` 更简单——它只存储线程索引，不存储复杂的 VMNK 坐标：

```cpp
template <class TiledCopy, class ThrIdx>
struct ThrCopy
{
  ThrIdx thr_idx_;

  CUTE_HOST_DEVICE
  ThrCopy(ThrIdx const& thr_idx) : thr_idx_(thr_idx) {}

  // partition_S / partition_D ...
  // retile_S / retile_D ...
};
```

### 6.2 `partition_S` / `partition_D`：提取本线程的片段

```cpp
template <class STensor>
CUTE_HOST_DEVICE
auto
partition_S(STensor&& stensor) const {
  // 对张量布局执行 tidfrg_S
  auto thr_tensor = make_tensor(static_cast<STensor&&>(stensor).data(),
                                TiledCopy::tidfrg_S(stensor.layout()));
  // thr_tensor: (Thr, (FrgV, FrgX), (RestM, RestN, ...))

  // 用本线程索引切片，提取 (FrgV, FrgX, RestM, RestN, ...)
  return thr_tensor(thr_idx_, _, repeat<rank_v<STensor>>(_));
}

template <class DTensor>
CUTE_HOST_DEVICE
auto
partition_D(DTensor&& dtensor) const {
  auto thr_tensor = make_tensor(static_cast<DTensor&&>(dtensor).data(),
                                TiledCopy::tidfrg_D(dtensor.layout()));
  return thr_tensor(thr_idx_, _, repeat<rank_v<DTensor>>(_));
}
```

**结果**：返回本线程负责的源/目标片段，形状为 `(FrgV, FrgX, RestM, RestN, ...)`。共享输入张量的数据指针（视图，非拷贝）。

### 6.3 `retile_S` / `retile_D`：静态重新切片

这两个是**静态函数**（不需要 ThrCopy 实例），因为 retile 只依赖于 TiledCopy 的布局，不依赖于线程索引：

```cpp
template <class STensor>
CUTE_HOST_DEVICE static
auto
retile_S(STensor&& stensor) {
  return make_tensor(static_cast<STensor&&>(stensor).data(),
                     TiledCopy::retile(stensor.layout()));
}

template <class DTensor>
CUTE_HOST_DEVICE static
auto
retile_D(DTensor&& dtensor) {
  return make_tensor(static_cast<DTensor&&>(dtensor).data(),
                     TiledCopy::retile(dtensor.layout()));
}
```

**为什么 `retile_S` 和 `retile_D` 相同？** 注释说明：`retile_S` 和 `retile_D` 假设使用参考布局（RefLayout）工作，因此它们是相同的。因为 `RefLayout` 通常是 `SrcLayout` 或 `DstLayout` 之一，retile 基于 RefLayout 进行，对 Src 和 Dst 通用。


## 7. `make_tiled_copy` 相关函数

### 7.1 `make_tiled_copy`：从线程和值布局构造

最通用的工厂函数（见 `copy_atom.hpp:490-517`）：

```cpp
/** 从逻辑线程和值布局构造 TiledCopy。
 * 线程和值布局将坐标映射到 thr_idx 和 val_idx。
 *   取这些布局的 raked_product 生成 TV 布局和 Tiler。
 * 当线程和值需要非常特定的坐标映射时很有用。
 */
template <class... Args,
          class ThrLayout,        // (m,n) -> thr_idx
          class ValLayout = Layout<_1>>  // (m,n) -> val_idx
CUTE_HOST_DEVICE
auto constexpr
make_tiled_copy(Copy_Atom<Args...> const& copy_atom,
                ThrLayout          const& thr_layout = {},
                ValLayout          const& val_layout = {})
{
  // 第1步：raked_product 生成 (M,N) -> (thr_idx, val_idx) 的布局
  auto layout_mn = raked_product(thr_layout, val_layout);

  // 第2步：逆布局，生成 (thr_idx, val_idx) -> (M,N) 的 TV 布局
  auto layout_tv = right_inverse(layout_mn).with_shape(
                       make_shape(size(thr_layout), size(val_layout)));

  // 第3步：生成 Tiler（提取相关元素）
  auto tiler = product_each(shape(layout_mn));

  return make_tiled_copy_impl(copy_atom, layout_tv, tiler);
}
```

**参数解读**：
- `thr_layout`：`(m, n) -> thr_idx`，描述哪个坐标由哪个线程负责
- `val_layout`：`(m, n) -> val_idx`，描述哪个坐标对应哪个值
- `raked_product`：将两个布局"梳状"组合，生成 `(m,n) -> (thr_idx, val_idx)`

**使用示例**：

```cpp
// 32 个线程，每个线程拷贝 1 个值
auto copy = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, half_t>{},
                            Layout<Shape<_32>>{},       // 32 线程
                            Layout<Shape<_1>>{});        // 每个线程 1 个值

// 32 个线程，每个线程拷贝 4 个值
auto copy = make_tiled_copy(Copy_Atom<UniversalCopy<uint32_t>, half_t>{},
                            Layout<Shape<_32>>{},       // 32 线程
                            Layout<Shape<_4>>{});        // 每个线程 4 个值
```

### 7.2 `make_cotiled_copy`：从数据布局构造

当不关心线程/值到坐标的具体映射，而更关心向量化宽度和偏移时使用（见 `copy_atom.hpp:525-568`）：

```cpp
/** 从线程和值偏移映射构造 TiledCopy。
 * TV 布局将线程和值映射到 data_layout 的余域。
 * 当线程和值不关心拥有特定坐标，而更关心向量化宽度和偏移时有用。
 */
template <class... Args, class AtomTVLayout, class DataLayout>
CUTE_HOST_DEVICE constexpr
auto
make_cotiled_copy(Copy_Atom<Args...> const& copy_atom,
                  AtomTVLayout const& atom_tv_layout,   // atom (thr,val) -> data addr
                  DataLayout   const& data_layout)      // coord -> data addr
{
  static_assert(is_static<AtomTVLayout>::value);
  static_assert(is_static<DataLayout>::value);

  // data addr -> data coord（逆布局，附加 1:0 处理越界）
  auto inv_data_layout = make_layout(left_inverse(data_layout), Layout<_1,_0>{});

  // (tid,vid) -> data_coord
  auto layout_tv_data = composition(inv_data_layout, atom_tv_layout);

  // 验证有效性：AtomTVLayout 指向的内存确实存在于 DataLayout 中
  CUTE_STATIC_ASSERT_V(
      coalesce(composition(make_layout(data_layout, Layout<_1,_0>{}), layout<1>(layout_tv_data)))
      == coalesce(layout<1>(atom_tv_layout)),
      "The memory pointed to by AtomTVLayout does not exist in the DataLayout.");

  // 生成 Tiler 和 Layout_TV ...
  // （省略具体计算，与 make_tiled_copy 类似但基于数据布局）
}
```

### 7.3 `make_tiled_copy_S` / `make_tiled_copy_D`：匹配已有 TiledCopy

```cpp
// 构造一个 Src 布局匹配 tiled_copy 的 TiledCopy
template <class... Args, class TiledCopy>
CUTE_HOST_DEVICE
auto
make_tiled_copy_S(Copy_Atom<Args...> const& copy_atom,
                  TiledCopy          const& tiled_copy)
{
  return make_tiled_copy_impl(copy_atom,
                              tiled_copy.get_layoutS_TV(),
                              typename TiledCopy::Tiler_MN{});
}

// 构造一个 Dst 布局匹配 tiled_copy 的 TiledCopy
template <class... Args, class TiledCopy>
CUTE_HOST_DEVICE
auto
make_tiled_copy_D(Copy_Atom<Args...> const& copy_atom,
                  TiledCopy          const& tiled_copy)
{
  return make_tiled_copy_impl(copy_atom,
                              tiled_copy.get_layoutD_TV(),
                              typename TiledCopy::Tiler_MN{});
}
```

**用途**：当需要用不同的 Copy_Atom 在同一个数据流上操作时（如先 `cp.async` 加载到 smem，再用 `ldmatrix` 从 smem 加载到寄存器），这两个函数确保新的 TiledCopy 与已有的 TiledCopy 在 Src 或 Dst 布局上对齐。

## 8. `make_tiled_copy_A/B/C`

CuTe 最强大的设计之一是 Copy 与 MMA 的无缝协作。这三个工厂函数（见 `copy_atom.hpp:421-446`）直接从 TiledMMA 构造 TiledCopy：

```cpp
// 构造匹配 MMA 的 A 矩阵布局的 TiledCopy
template <class... CArgs, class... MArgs>
CUTE_HOST_DEVICE
auto constexpr
make_tiled_copy_A(Copy_Atom<CArgs...> const& copy_atom,
                  TiledMMA<MArgs...>  const& mma)
{
  return make_tiled_copy_impl(copy_atom,
                              mma.get_layoutA_TV(),    // MMA 的 A 布局
                              make_shape(tile_size<0>(mma), tile_size<2>(mma)));  // (M,K)
}

// 构造匹配 MMA 的 B 矩阵布局的 TiledCopy
template <class... CArgs, class... MArgs>
CUTE_HOST_DEVICE
auto constexpr
make_tiled_copy_B(Copy_Atom<CArgs...> const& copy_atom,
                  TiledMMA<MArgs...>  const& mma)
{
  return make_tiled_copy_impl(copy_atom,
                              mma.get_layoutB_TV(),    // MMA 的 B 布局
                              make_shape(tile_size<1>(mma), tile_size<2>(mma)));  // (N,K)
}

// 构造匹配 MMA 的 C 矩阵布局的 TiledCopy
template <class... CArgs, class... MArgs>
CUTE_HOST_DEVICE
auto
make_tiled_copy_C(Copy_Atom<CArgs...> const& copy_atom,
                  TiledMMA<MArgs...>  const& mma)
{
  return make_tiled_copy_impl(copy_atom,
                              mma.get_layoutC_TV(),    // MMA 的 C 布局
                              make_shape(tile_size<0>(mma), tile_size<1>(mma)));  // (M,N)
}
```

**设计意图**：
- `make_tiled_copy_A`：构造一个 TiledCopy，使得拷贝后数据的线程-值分布**正好匹配** MMA 对 A 矩阵的分布要求
- `make_tiled_copy_B`：同理，匹配 B 矩阵
- `make_tiled_copy_C`：匹配 C 矩阵（用于 epilogue，将累加器存回）

**这意味着**：用 `make_tiled_copy_A` 构造的 TiledCopy 分区得到的寄存器片段，可以直接传给 MMA 的 `call()` 函数，无需额外重排。这是 CuTe 实现高效 GEMM 的关键——Copy 和 MMA 通过布局系统自动对齐。

### 8.1 `make_tiled_copy_C_atom`：原子级 C 拷贝

一个更精细的变体（见 `copy_atom.hpp:450-482`）：

```cpp
// 返回能 retile LayoutC_TV 的最小 tiled copy
// 用于流水线 epilogue 的子分块存储
template <class... CArgs, class... MArgs>
CUTE_HOST_DEVICE
auto
make_tiled_copy_C_atom(Copy_Atom<CArgs...> const& copy_atom,
                       TiledMMA<MArgs...>  const& mma)
{
  // 截断 V-layout 到 Copy_Atom 大小，保留 V-order
  auto layoutC_TV = mma.get_layoutC_TV();
  auto copy_V     = Int<Copy_Atom<CArgs...>::NumValSrc>{};
  CUTE_STATIC_ASSERT_V(copy_V <= size<1>(layoutC_TV));
  auto layout_TV  = composition(layoutC_TV, make_layout(make_shape(size<0>(layoutC_TV), copy_V)));

  // 重新计算 Tiler 和 TV 布局 ...
  // （省略具体计算）

  return make_tiled_copy_impl(copy_atom, layout_tv, tiler);
}
```

**用途**：当 epilogue 需要将 C 矩阵的子分块存回时，这个函数构造一个最小的 TiledCopy，其 V 布局与 Copy_Atom 大小匹配，但保持 MMA 的 C 布局顺序。

## 9. 完整使用示例

### 9.1 Ampere 架构：cp.async + ldmatrix + mma 的完整流程

```cpp
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
using namespace cute;

// ===== 第1步：定义 TiledMMA =====
auto mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                          Layout<Shape<_2, _4>>{});  // M×2, N×4

// ===== 第2步：定义 Copy Atoms =====
// gmem -> smem: cp.async (16字节 = 8个half)
using CopyAtomG2S = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half_t>;
// smem -> rmem: ldmatrix (匹配 MMA 的 A/B 布局)
using CopyAtomS2R_A = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
using CopyAtomS2R_B = Copy_Atom<SM75_U32x2_LDSM_N, half_t>;

// ===== 第3步：构造 TiledCopy =====
// gmem -> smem: 自定义线程分布
auto copy_g2s = make_tiled_copy(CopyAtomG2S{},
                                Layout<Shape<_128>>{},   // 128 线程
                                Layout<Shape<_1>>{});    // 每线程 1 个 128-bit

// smem -> rmem: 匹配 MMA 的 A/B 布局
auto copy_s2r_A = make_tiled_copy_A(CopyAtomS2R_A{}, mma);
auto copy_s2r_B = make_tiled_copy_B(CopyAtomS2R_B{}, mma);

// ===== 第4步：获取线程切片 =====
int thread_idx = threadIdx.x;
auto thr_copy_g2s = copy_g2s.get_slice(thread_idx);
auto thr_copy_s2r_A = copy_s2r_A.get_slice(thread_idx);
auto thr_copy_s2r_B = copy_s2r_B.get_slice(thread_idx);
auto thr_mma = mma.get_slice(thread_idx);

// ===== 第5步：分区 =====
// 假设 gA: (M, K) gmem 张量, gB: (N, K) gmem 张量
// sA: (M, K) smem 张量, sB: (N, K) smem 张量

// gmem -> smem 分区
auto tAgA = thr_copy_g2s.partition_S(gA);  // (FrgV, RestM, RestK)
auto tAsA = thr_copy_g2s.partition_D(sA);  // (FrgV, RestM, RestK)
auto tBgB = thr_copy_g2s.partition_S(gB);
auto tBsB = thr_copy_g2s.partition_D(sB);

// smem -> rmem 分区（匹配 MMA）
auto tArA = thr_copy_s2r_A.partition_S(sA);  // (FrgV, RestM, RestK)
auto tBrB = thr_copy_s2r_B.partition_S(sB);  // (FrgV, RestN, RestK)

// MMA 分区
auto tCrC = thr_mma.partition_fragment_C(sC); // (FrgV, RestM, RestN)

// ===== 第6步：执行拷贝和计算 =====
// gmem -> smem (异步)
copy(copy_g2s, tAgA, tAsA);
copy(copy_g2s, tBgB, tBsB);
cp_async_fence();              // 提交
cp_async_wait<0>();            // 等待完成
__syncthreads();

// smem -> rmem
copy(copy_s2r_A, tArA, tCrA);  // tCrA: 寄存器片段
copy(copy_s2r_B, tBrB, tCrB);

// rmem: MMA 计算
mma.call(tCrA, tCrB, tCrC);    // C = A * B + C
```

### 9.2 Hopper 架构：TMA + WGMMA 的完整流程

```cpp
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/mma_atom.hpp>
using namespace cute;

// ===== 第1步：创建 TMA 描述符（host 端）=====
// 假设有函数创建 TMA 描述符
auto tma_desc_A = make_tma_descriptor(gA_layout, ...);
auto tma_desc_B = make_tma_descriptor(gB_layout, ...);

// ===== 第2步：创建 TMA Copy_Atom =====
// TMA_LOAD: gmem -> smem
// 需要指定每次 TMA 的比特数（如 1024 = 128 字节）
auto copy_atom_A = Copy_Atom<SM90_TMA_LOAD, half_t>{};
auto copy_atom_B = Copy_Atom<SM90_TMA_LOAD, half_t>{};

// ===== 第3步：构造 TiledCopy =====
// TMA 是单线程发起，通常用 Layout<Shape<_1>> 表示 1 个线程
auto copy_g2s = make_tiled_copy(copy_atom_A,
                                Layout<Shape<_1>>{},   // 1 个线程发起 TMA
                                Layout<Shape<_1>>{});

// ===== 第4步：绑定 TMA 描述符和 mbarrier =====
// 需要先构造 Traits，再 with(mbar)
// 注意：TMA 的 with() 返回可执行的 Traits
auto tma_traits_A = Copy_Traits<SM90_TMA_LOAD, ...>{tma_desc_A, ...};
uint64_t mbar;
auto executable_traits_A = tma_traits_A.with(mbar);
auto copy_atom_exec_A = Copy_Atom<decltype(executable_traits_A), half_t>{executable_traits_A};

// ===== 第5步：获取 TMA 坐标张量 =====
// TMA 不直接用数据指针分区，而是用坐标张量
auto gA_tma = tma_traits_A.get_tma_tensor(make_shape(M, K));  // 坐标张量

// 分区坐标
auto thr_copy = copy_g2s.get_slice(threadIdx.x);
auto tAgA = thr_copy.partition_S(gA_tma);  // TMA 坐标片段
auto tAsA = thr_copy.partition_D(sA);       // smem 目标

// ===== 第6步：执行 TMA 拷贝 =====
// 发起 TMA（需要先设置 mbarrier）
copy(copy_g2s, tAgA, tAsA);
// 等待 mbarrier...
```

### 9.3 ZFILL 条件拷贝示例

```cpp
// 创建带 ZFILL 的 cp.async Copy_Atom
auto copy_atom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, half_t>{};

// 构造 TiledCopy
auto copy = make_tiled_copy(copy_atom,
                            Layout<Shape<_128>>{},
                            Layout<Shape<_1>>{});

auto thr_copy = copy.get_slice(threadIdx.x);
auto tAgA = thr_copy.partition_S(gA);
auto tAsA = thr_copy.partition_D(sA);

// 构造谓词张量（处理边界）
auto pred = make_tensor<bool>(shape(tAgA));
// ... 填充 pred，越界处为 false ...

// 带谓词的拷贝：pred=true 正常拷贝，pred=false 零填充
copy.call(pred, tAgA, tAsA);
// 内部会调用 traits.with(pred[0]) 构造 ZFILL Traits
// 然后调用 cp.async.ca.shared.global [%0], [%1], 16, src_size;
//   当 pred=false 时 src_size=0，硬件零填充
```

### 9.4 Epilogue：retile + C 拷贝

```cpp
// 假设 MMA 计算完成，tCrC 是累加器寄存器片段
// 现在需要将结果存回 gmem

// 第1步：构造 C 的 TiledCopy
auto copy_atom_C = Copy_Atom<UniversalCopy<uint128_t>, float>{};
auto copy_C = make_tiled_copy_C(copy_atom_C, mma);

// 第2步：分区 gmem 目标
auto thr_copy_C = copy_C.get_slice(threadIdx.x);
auto tCgC = thr_copy_C.partition_D(gC);  // gmem 目标片段

// 第3步：retile 寄存器片段以匹配 Copy_Atom
auto tCrC_retiled = ThrCopy<TiledCopy, int>::retile_D(tCrC);
// 或通过 ThrCopy 实例：
// auto tCrC_retiled = thr_copy_C.retile_D(tCrC);

// 第4步：拷贝 rmem -> gmem
copy(copy_C, tCrC_retiled, tCgC);
```

## 10. 总结

### 10.1 Copy 与 MMA 抽象的对比

| 特性 | MMA | Copy |
|:------:|:-----:|:------:|
| **底层 Operation** | `fma()` 函数 | `copy()` 函数 |
| **寄存器类型** | D/A/B/C Registers | S/D Registers |
| **Traits 布局单位** | 坐标 (m,n,k) | 比特 (bit) |
| **Traits 布局数量** | ALayout, BLayout, CLayout | SrcLayout, DstLayout, RefLayout |
| **Atom 参数化** | 无（从 Traits 推导） | **ValType**（用户指定） |
| **Tiled 平铺方式** | `AtomLayoutMNK` 自动计算 | **直接提供** `LayoutCopy_TV` + `Tiler_MN` |
| **与 MMA 对接** | - | `make_tiled_copy_A/B/C` |
| **运行时参数** | `accumulate_` (Scale) | `pred` (ZFILL), `tma_mbar` (TMA) |

### 10.2 RefLayout 的设计意义

Copy 引入 `RefLayout` 是为了处理 **Src 和 Dst 有不同线程-值分布**的情况。典型场景是 `ldmatrix`：

- **Src（共享内存）**：32 个线程按 `(8, 4)` 网格排列，每个线程提供一个 128-bit 地址
- **Dst（寄存器）**：32 个线程线性排列，每个线程获得 32-bit 数据

`RefLayout` 提供了一个统一的"参考坐标系"：TiledCopy 内部用 RefLayout 坐标管理线程-值映射，在执行时通过 `right_inverse(RefLayout).compose(SrcLayout)` 或 `.compose(DstLayout)` 映射到实际的 Src 或 Dst 表示。这种设计使得同一个 TiledCopy 可以正确处理 Src 和 Dst 分布不同的 Copy 指令。

### 10.3 位级布局的统一性

Copy Traits 使用**比特**作为布局单位，这是一个关键的设计决策：

```
Copy_Traits:
  SrcLayout: (thr, val) -> bit    (位级)
  DstLayout: (thr, val) -> bit    (位级)
  
Copy_Atom:
  ValLayoutSrc = recast_layout<uint1_t, ValType>(SrcLayout)   (值级)
  ValLayoutDst = recast_layout<uint1_t, ValType>(DstLayout)   (值级)
```

**好处**：
1. **类型无关**：同一种 Copy 指令（如 `cp.async` 16 字节）可以用于 `half_t`（8 个值）、`float`（4 个值）、`uint8_t`（16 个值），只需改变 `ValType`
2. **精确描述**：位级布局可以精确描述子字节类型（如 4-bit `int4b_t`、1-bit `uint1b_t`）的拷贝
3. **自动转换**：`recast_layout` 自动处理位到值的转换，用户只需指定 `ValType`

### 10.4 从 Ampere 到 Hopper 的演进

| 特性 | Ampere (SM80) | Hopper (SM90) |
|:------:|:--------------:|:--------------:|
| gmem→smem 拷贝 | `cp.async`（每线程 4/8/16 字节） | **TMA**（单线程发起，批量拷贝） |
| smem→rmem 拷贝 | `ldmatrix`（warp 协作） | WGMMA 直接从 smem 读取（无需 `ldmatrix`） |
| 线程数 | `ThrID = Layout<_32>` (warp) | `ThrID = Layout<_1>` (TMA) 或 `_128` (WGMMA) |
| 布局复杂度 | 复杂（warp 级线程分布） | 简单（TMA: 单线程；WGMMA: warpgroup） |
| 运行时参数 | `pred` (ZFILL) | `tma_mbar`, `multicast_mask`, `cache_hint` |
| 异步同步 | `cp.async.commit_group` + `wait_group` | **mbarrier** (内存屏障) |
| 边界处理 | 软件 predication（ZFILL） | **硬件自动**（TMA 描述符内含边界信息） |
| `with()` 用途 | 设置 ZFILL 谓词 | 绑定 mbarrier、cache_hint |

**TMA 带来的变革**：
1. **简化线程分布**：TMA 只需一个线程发起，`ThrID = Layout<_1>`，大大简化了 Traits
2. **描述符驱动**：地址计算、边界检查、swizzle 全部由硬件处理，软件只需提供坐标
3. **坐标张量**：TMA 引入了 `get_tma_tensor()` 和坐标张量的概念，分区操作的对象从数据指针变为坐标
4. **mbarrier 同步**：取代了 `cp.async` 的 `commit_group`/`wait_group`，与线程块集群（cluster）和 DSMEM（分布式共享内存）深度集成

### 10.5 关键文件索引

| 文件 | 内容 |
|:------:|:------:|
| `include/cute/arch/copy.hpp` | `UniversalCopy`, `AutoVectorizingCopy`, `DefaultCopy` |
| `include/cute/arch/copy_sm75.hpp` | `ldmatrix` PTX 封装（Turing） |
| `include/cute/arch/copy_sm80.hpp` | `cp.async` PTX 封装（Ampere） |
| `include/cute/arch/copy_sm90.hpp` | `stmatrix` PTX 封装（Hopper） |
| `include/cute/arch/copy_sm90_tma.hpp` | TMA `cp.async.bulk.tensor` PTX 封装 |
| `include/cute/atom/copy_traits.hpp` | `Copy_Traits` 概念 + `copy_unpack` |
| `include/cute/atom/copy_traits_sm50.hpp` | Shuffle 指令的 Traits |
| `include/cute/atom/copy_traits_sm75.hpp` | `ldmatrix` 的 Traits |
| `include/cute/atom/copy_traits_sm80.hpp` | `cp.async` 的 Traits（含 ZFILL） |
| `include/cute/atom/copy_traits_sm90.hpp` | `stmatrix` 的 Traits |
| `include/cute/atom/copy_traits_sm90_tma.hpp` | TMA / Bulk Copy 的 Traits |
| `include/cute/atom/copy_atom.hpp` | `Copy_Atom`, `TiledCopy`, `ThrCopy`, 工厂函数 |

---

> **参考资料**：
> - CUTLASS 4.5.0 源码：`include/cute/atom/` 和 `include/cute/arch/` 目录
> - NVIDIA PTX ISA 文档：`cp.async`, `ldmatrix`, `stmatrix`, `cp.async.bulk.tensor` 指令说明
> - CuTe Layout 系统：`include/cute/layout.hpp`
>
> 本文所有代码引用均基于 CUTLASS 4.5.0，行号可能与实际文件略有出入，请以源码为准。