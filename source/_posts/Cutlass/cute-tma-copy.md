---
title: CuTe 学习笔记（九）TMA Copy 抽象
date: 2025-03-26 18:00:00
tags: [CUTLASS, CuTe]
categories: [Cutlass 学习笔记, CuTe]
description: 剖析 Hopper TMA 拷贝的四层实现，从描述符与 mbarrier、PTX 指令封装、swizzle 转换，到 Copy_Traits 特化与 make_tma_copy 的描述符构造，并梳理端到端的拷贝流程。
published: true
render:
  math: mathjax
---

Hopper（SM90）引入的 TMA（Tensor Memory Accelerator）是一个异步批量张量拷贝单元。它用一个 128 字节的描述符编码张量的形状、步长和 swizzle，由单条指令完成整个 CTA tile 的搬运，并用 mbarrier 报告完成。与 SM80 的 `cp.async` 逐线程发起、手工计算地址和边界不同，TMA 把多维寻址、越界填充和 swizzle 都交给硬件，还支持 cluster 内多播和归约。CUTLASS 把这套能力拆成四个文件分四层实现，本文自底向上逐层剖析，并梳理端到端的拷贝流程。

## 1. TMA 背景知识

### 1.1 TMA 概述

TMA (Tensor Memory Accelerator) 是 NVIDIA Hopper (SM90) 架构引入的专用硬件单元，用于在全局内存 (GMEM) 和共享内存 (SMEM) 之间执行异步批量张量拷贝。与传统的 `cp.async` 相比，TMA 具有以下核心优势：

- 硬件张量感知：TMA 直接理解多维张量的形状和步长，一个指令可以拷贝整个 CTA tile
- 零寄存器开销：所有坐标计算由硬件完成，不占用线程寄存器
- 内置越界处理：支持 OOB (Out-Of-Bounds) 填充零或常量值
- 原生 Swizzle 支持：直接支持 SMEM 的 swizzle 布局
- 多播 (Multicast)：可将数据同时发送到 cluster 内多个 CTA 的 SMEM
- L2 缓存提示：支持 EVICT_FIRST / EVICT_LAST 等缓存策略
- Reduce 操作：支持 `cp.reduce.async.bulk` 进行归约拷贝

### 1.2 TMA 的核心数据结构

TMA 的核心是一个 128 字节的描述符 (TmaDescriptor)，它本质上就是 CUDA Driver API 中的 `CUtensorMap`。这个描述符编码了：

- GMEM 张量的基地址、形状、步长
- SMEM box 的形状（每次拷贝的 tile 大小）
- Swizzle 模式
- 数据类型
- OOB 填充策略
- L2 promotion 策略

一旦描述符创建完成，GPU 端只需提供描述符指针 + 坐标偏移 + mbarrier，即可触发 TMA 拷贝。

### 1.3 mbarrier 同步

TMA 是异步操作，从全局内存拷贝到共享内存时需要 mbarrier (memory barrier) 来跟踪完成状态：

1. 初始化 mbarrier 并设置期望的事务字节数
2. 发出 TMA 指令（硬件会自动到达 mbarrier）
3. 线程等待 mbarrier 完成
4. 安全使用 SMEM 数据

---

## 2. 四层架构总览

CUTLASS 将 TMA 功能分散在四个文件中，形成清晰的层次结构：

1. 第一层 copy_sm90_desc.hpp。TmaDescriptor = CUtensorMap；mbarrier PTX 封装；Swizzle、OOBFill、L2Promotion、CacheHint 枚举；tensormap.replace.tile PTX（设备端修改描述符）。

2. 第二层 copy_sm90_tma.hpp。SM90_TMA_LOAD、SM90_TMA_STORE、SM90_TMA_REDUCE、SM90_TMA_MULTICAST、SM90_TMA_IM2COL；SM90_BULK_COPY_G2S、SM90_BULK_COPY_S2G；每个结构体封装对应的 PTX 指令。

3. 第三层 copy_traits_sm90_tma_swizzle.hpp。`Swizzle<B,M,S>` 到 TMA swizzle bits/base 的转换。

4. 第四层 copy_traits_sm90_tma.hpp。Copy_Traits 特化（不可执行 → 可执行两阶段）；make_tma_copy 公共 API（构造描述符 + 创建 TiledCopy）；tma_partition 分区器；make_tma_copy_A/B/C_sm90 矩阵乘法专用便捷函数。

设计理念：底层关注硬件细节（PTX 指令、描述符格式），上层关注 CuTe Layout 系统的衔接和用户友好的 API。下面自底向上逐层分析。

## 3. 第一层 `copy_sm90_desc.hpp` 中的描述符与屏障

对应文件 `include/cute/arch/copy_sm90_desc.hpp`（476 行）。这个文件是整个 TMA 体系的基石，定义了三部分核心内容：

### 3.1 mbarrier 同步原语

mbarrier 是 SMEM 中的同步原语，用于 TMA 异步拷贝的完成通知。文件提供了一组内联函数封装对应的 PTX 指令：

```cpp
// 初始化 mbarrier，设置到达线程数
void initialize_barrier(uint64_t& mbar, int arrive_count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(smem_ptr), "r"(arrive_count));
}

// 设置期望的事务字节数（TMA 完成后会自动到达）
void set_barrier_transaction_bytes(uint64_t& mbar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" 
               :: "r"(smem_ptr), "r"(bytes));
}

// 等待 mbarrier 完成（阻塞直到所有事务到达）
void wait_barrier(uint64_t& mbar, uint32_t phase) {
  asm volatile("{ mbarrier.try_wait.parity.shared.b64 _, [%0], %1; }"
               :: "r"(smem_ptr), "r"(phase));
}

// 线程主动到达 mbarrier（非事务性到达）
void arrive_barrier(uint64_t& mbar) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(smem_ptr));
}
```

工作原理：
- `set_barrier_transaction_bytes` 告诉 mbarrier 预计有多少字节的数据要到达
- TMA 硬件在拷贝完成时自动到达 mbarrier 并报告字节数
- `wait_barrier` 使用 phase 计数（0/1 交替）来等待下一轮完成

### 3.2 TmaDescriptor

```cpp
#if defined(__CUDACC_RTC__)
// RTC (运行时编译) 环境没有 CUDA Driver API，用裸字节替代
struct alignas(64) TmaDescriptor { char bytes[128]; };
#else
using TmaDescriptor = CUtensorMap;  // 就是 CUDA Driver API 的 CUtensorMap
#endif
```

`TmaDescriptor` 就是 128 字节的 `CUtensorMap`，64 字节对齐。在设备端可以通过 `tensormap.replace.tile` PTX 指令修改其内容。

### 3.3 TMA 枚举类型

文件定义了一系列映射到 CUDA Driver API 枚举的 C++ 枚举：

#### SmemSwizzleBits 的 Swizzle 粒度

```cpp
enum class SmemSwizzleBits {
  DISABLE = 0,  // 不使用 swizzle
  B32     = 1,  // 32 字节粒度
  B64     = 2,  // 64 字节粒度
  B128    = 3   // 128 字节粒度
};
```

#### SmemSwizzleBase 的 Swizzle 基地址模式

```cpp
enum class SmemSwizzleBase {
  SWIZZLE_BASE_16B         = 0,  // 16B 基地址
  SWIZZLE_BASE_32B         = 1,  // 32B 基地址
  SWIZZLE_BASE_32B_FLIP_8B = 2,  // 32B 基地址 + 8B 翻转
  SWIZZLE_BASE_64B         = 3   // 64B 基地址
};
```

#### OOBFill 的越界填充策略

```cpp
enum class OOBFill {
  ZERO,     // 越界位置填充 0
  CONSTANT  // 越界位置填充常量（需要额外配置）
};
```

#### L2Promotion 的 L2 缓存提升策略

```cpp
enum class L2Promotion {
  DISABLE,  // 不提升
  B64,      // 提升到 64 字节
  B128,     // 提升到 128 字节
  B256      // 提升到 256 字节
};
```

#### CacheHintSm90 的 L2 缓存提示

```cpp
enum class CacheHintSm90 : uint64_t {
  EVICT_NORMAL = 0x0000000000000000,  // 正常驱逐
  EVICT_FIRST  = 0x0000000000008000,  // 优先驱逐
  EVICT_LAST   = 0x0000000000004000   // 最后驱逐
};
```

### 3.4 类型与枚举转换函数

```cpp
// C++ 类型 → CUDA Driver API 数据类型
template <class T>
constexpr CUtensorMapDataType to_CUtensorMapDataType() {
  if constexpr (is_same_v<T, float>)         return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  else if constexpr (is_same_v<T, half_t>)   return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  else if constexpr (is_same_v<T, bfloat16_t>) return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  // ... double, int8, int32, uint8, fp8, etc.
}
```

### 3.5 设备端描述符修改

Hopper 支持在设备端直接修改 TMA 描述符的某些字段，这在需要动态调整 tile 形状或地址时非常有用：

```cpp
// 在 GMEM 中修改描述符的 tile 地址
void tma_descriptor_replace_addr_in_global_mem(
    TmaDescriptor const* desc_src, TmaDescriptor* desc_dst, 
    void const* const gmem_addr) {
  asm volatile("tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1, [%2];"
               :: "l"(desc_dst), "l"(gmem_addr), "l"(desc_src));
}

// 在 SMEM 中修改描述符的维度和步长
void tma_descriptor_replace_dims_strides_in_shared_mem(
    TmaDescriptor* desc_smem, int dim0, ..., int stride1, int stride2) {
  asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], %1, %2;"
               :: "r"(smem_ptr), "r"(dim0), "r"(dim1));
  asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], %1, %2;"
               :: "r"(smem_ptr), "l"(stride1), "l"(stride2));
}
```

典型场景是网格持久化 (grid persistence) 模式：在 host 端创建一个模板描述符，然后在 kernel 内根据当前 CTA 的实际需求修改地址和维度，避免为每个不同的 tile 创建独立的描述符。

---

## 4. 第二层 `copy_sm90_tma.hpp` 中的 PTX 指令封装

对应文件 `include/cute/arch/copy_sm90_tma.hpp`（1497 行）。这个文件将 TMA 相关的 PTX 指令封装成 C++ 结构体，每个结构体对应一种 TMA 操作，是连接软件抽象和硬件指令的桥梁。

### 4.1 TMA LOAD 家族

#### 基础 LOAD（1D ~ 5D）

```cpp
struct SM90_TMA_LOAD_1D {
  CUTE_DEVICE static void
  copy(uint64_t const* const desc_ptr, uint32_t smem_addr, 
       int32_t const& crd0, uint64_t const* mbar_ptr) {
    asm volatile(
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%3}], [%2];"
      :: "r"(smem_addr), "l"(desc_ptr), "r"(mbar_addr), "r"(crd0));
  }
  
  struct PREFETCH {
    CUTE_DEVICE static void
    copy(uint64_t const* const desc_ptr, int32_t const& crd0) {
      asm volatile(
        "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "{%0}, [%1, {%2}];"
        :: "r"(smem_addr), "l"(desc_ptr), "r"(crd0));
    }
  };
};
```

PTX 指令格式分解：
- `cp.async.bulk.tensor` — TMA 异步批量张量拷贝
- `1d` / `2d` / ... / `5d` — 张量维度
- `shared::cluster` — 目标在 cluster 级 SMEM
- `global` — 源在 GMEM
- `mbarrier::complete_tx::bytes` — 使用 mbarrier 跟踪字节数完成
- `[%0]` — SMEM 目标地址
- `[%1, {%3}]` — 描述符 + 坐标偏移
- `[%2]` — mbarrier 地址

#### 多维 LOAD 模式

`SM90_TMA_LOAD_2D` 到 `SM90_TMA_LOAD_5D` 结构类似，只是坐标参数数量不同。`SM90_TMA_LOAD` 是一个调度器，通过重载 `copy()` 根据坐标数量选择正确的特化：

```cpp
struct SM90_TMA_LOAD {
  CUTE_DEVICE static void
  copy(uint64_t const* const desc_ptr, uint32_t smem_addr,
       int32_t const& crd0, uint64_t const* mbar_ptr) {
    SM90_TMA_LOAD_1D::copy(desc_ptr, smem_addr, crd0, mbar_ptr);
  }
  // 2D, 3D, 4D, 5D 重载...
};
```

#### IM2COL LOAD

图像卷积场景下，TMA 支持 im2col 模式，在拷贝时自动执行图像到列的展开：

```cpp
struct SM90_TMA_LOAD_IM2COL_3D {
  // 比普通 LOAD 多了 offset 坐标参数
  CUTE_DEVICE static void
  copy(uint64_t const* const desc_ptr, uint32_t smem_addr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2,
       int32_t const& offset0, int32_t const& offset1,
       uint64_t const* mbar_ptr) {
    asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%3,%4,%5}], [%2], {%6,%7};"
      :: "r"(smem_addr), "l"(desc_ptr), "r"(mbar_addr),
         "r"(crd0), "r"(crd1), "r"(crd2),
         "r"(offset0), "r"(offset1));
  }
};
```

#### MULTICAST LOAD

多播模式允许将同一份数据发送到 cluster 内多个 CTA 的 SMEM，通过 16 位掩码指定目标 CTA：

```cpp
struct SM90_TMA_LOAD_MULTICAST_2D {
  CUTE_DEVICE static void
  copy(uint64_t const* const desc_ptr, uint32_t smem_addr,
       int16_t const& multicast_mask,
       int32_t const& crd0, int32_t const& crd1,
       uint64_t const* mbar_ptr) {
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
      "[%0], [%1, {%3,%4}], [%2], %5;"
      :: "r"(smem_addr), "l"(desc_ptr), "r"(mbar_addr),
         "r"(crd0), "r"(crd1), "h"(multicast_mask));
  }
};
```

`multicast::cluster` 标志和 `%5`（16 位掩码）是多播模式的关键。掩码的第 i 位为 1 表示数据发送到 cluster 内第 i 个 CTA。

### 4.2 TMA STORE 家族

STORE 是 LOAD 的反向操作，从 SMEM 写回 GMEM。与 LOAD 的关键区别：

- 不需要 mbarrier：STORE 使用 `bulk_group` 语义而非 `mbarrier::complete_tx`
- 方向相反：`global.shared::cta`（GMEM 是目标，SMEM 是源）

```cpp
struct SM90_TMA_STORE_2D {
  CUTE_DEVICE static void
  copy(uint64_t const* const desc_ptr, 
       int32_t const& crd0, int32_t const& crd1,
       uint32_t smem_addr) {
    asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
      "[%0, {%1,%2}], [%3];"
      :: "l"(desc_ptr), "r"(crd0), "r"(crd1), "r"(smem_addr));
  }
};
```

STORE 的完成通过 `cp.async.bulk.commit_group` + `cp.async.bulk.wait_group` 来同步，而非 mbarrier。

### 4.3 TMA REDUCE

REDUCE 在拷贝的同时执行归约操作（目前只支持 ADD）：

```cpp
struct SM90_TMA_REDUCE_ADD_2D {
  CUTE_DEVICE static void
  copy(uint64_t const* const desc_ptr,
       int32_t const& crd0, int32_t const& crd1,
       uint32_t smem_addr) {
    asm volatile(
      "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group "
      "[%0, {%1,%2}], [%3];"
      :: "l"(desc_ptr), "r"(crd0), "r"(crd1), "r"(smem_addr));
  }
};
```

### 4.4 BULK_COPY（非张量拷贝）

`SM90_BULK_COPY_G2S` 和 `SM90_BULK_COPY_S2G` 是 TMA 的简化版本，不使用张量描述符，直接按字节拷贝连续内存块：

```cpp
struct SM90_BULK_COPY_G2S {
  CUTE_DEVICE static void
  copy(void const* const gmem_ptr, uint32_t smem_addr,
       int32_t const& bytes, uint64_t const* mbar_ptr) {
    asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];"
      :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr));
  }
};

struct SM90_BULK_COPY_S2G {
  CUTE_DEVICE static void
  copy(void* const gmem_ptr, uint32_t smem_addr,
       int32_t const& bytes) {
    asm volatile(
      "cp.async.bulk.global.shared::cta.bulk_group "
      "[%0], [%1], %2;"
      :: "l"(gmem_ptr), "r"(smem_addr), "r"(bytes));
  }
};
```

BULK_COPY 与 TMA 的区别：
- BULK_COPY 只需提供指针和字节数，不涉及张量形状
- BULK_COPY 适合拷贝连续的扁平数据
- TMA 适合多维张量 tile，自带 swizzle 和 OOB 处理

### 4.5 SM90_BULK_COPY_AUTO

```cpp
struct SM90_BULK_COPY_AUTO {};
```

这是一个空占位符，在 trait 层根据上下文自动推导应该用 G2S 还是 S2G。

---

## 5. 第三层 `copy_traits_sm90_tma_swizzle.hpp` 中的 Swizzle 转换

对应文件 `include/cute/atom/copy_traits_sm90_tma_swizzle.hpp`（115 行）。这个文件很小但很关键，它将 CuTe 的 `Swizzle<B,M,S>` 布局表示转换为 TMA 描述符所需的 swizzle 枚举。

### 5.1 Swizzle<B,M,S> 回顾

CuTe 用 `Swizzle<B,M,S>` 表示 SMEM 的 swizzle 模式，其中：
- `B`：xor 的位宽（0=1B, 1=2B, 2=4B, 3=8B）
- `M`：swizzle 起始位的位置（决定粒度）
- `S`：swizzle 起始位的偏移

### 5.2 get_tma_swizzle_bits

```cpp
template <int B, int M, int S>
CUTE_HOST_DEVICE constexpr
TMA::SmemSwizzleBits
get_tma_swizzle_bits(Swizzle<B,M,S>) {
  if constexpr (M == 4) {
    if constexpr (B == 3) { return TMA::SmemSwizzleBits::B128; }
    if constexpr (B == 2) { return TMA::SmemSwizzleBits::B64; }
    if constexpr (B == 1) { return TMA::SmemSwizzleBits::B32; }
    if constexpr (B == 0) { return TMA::SmemSwizzleBits::DISABLE; }
  } else if constexpr (M == 5 || M == 6) {
    // M==5 或 M==6 时 B 必须为 2
    return TMA::SmemSwizzleBits::B128;
  }
}
```

映射规则：
- `M == 4`（16B 基础粒度）：B 决定 swizzle 粒度（B=3→128B, B=2→64B, B=1→32B, B=0→禁用）
- `M == 5`（32B 基础粒度）：固定 B128
- `M == 6`（64B 基础粒度）：固定 B128

### 5.3 get_tma_swizzle_base

```cpp
template <int B, int M, int S>
CUTE_HOST_DEVICE constexpr
TMA::SmemSwizzleBase
get_tma_swizzle_base(Swizzle<B,M,S>) {
  if constexpr (M == 4) {
    static_assert(S == 3, "Expected S = 3 when M == 4.");
    return TMA::SmemSwizzleBase::SWIZZLE_BASE_16B;
  } else if constexpr (M == 5) {
    static_assert(S == 2, "Expected S = 2 when M == 5.");
    return TMA::SmemSwizzleBase::SWIZZLE_BASE_32B;
  } else if constexpr (M == 6) {
    return TMA::SmemSwizzleBase::SWIZZLE_BASE_64B;
  }
}
```

映射规则：
- `M == 4` → `SWIZZLE_BASE_16B`（要求 S == 3）
- `M == 5` → `SWIZZLE_BASE_32B`（要求 S == 2）
- `M == 6` → `SWIZZLE_BASE_64B`

这两个函数在 `make_tma_copy_desc` 中被调用，将 CuTe Layout 的 swizzle 部分翻译成 TMA 描述符的语言。

---

## 6. 第四层 `copy_traits_sm90_tma.hpp` 中的 Copy_Traits 与描述符构造

对应文件 `include/cute/atom/copy_traits_sm90_tma.hpp`（1625 行）。这是最大也最重要的文件，包含三部分内容：
1. 各种 TMA 操作的 `Copy_Traits` 特化
2. 描述符构造的 `detail` 函数（`construct_tma_gbasis`、`make_tma_copy_desc` 等）
3. 用户友好的公共 API（`make_tma_copy`、`tma_partition` 等）

### 6.1 AuxTmaParams 辅助参数

```cpp
template <class GmemTmaBasisStrides, class TmaGmemBasis, class TmaSwizzle = void>
struct AuxTmaParams {
  GmemTmaBasisStrides g_stride_;  // gmem basis 的步长信息
  TmaGmemBasis tma_gbasis_;       // TMA gmem basis 布局
  TmaSwizzle tma_swizzle_;        // TMA swizzle 模式
};
```

这个结构体存储了 TMA 描述符之外的辅助信息，主要用于 `get_tma_tensor` 构造 TMA 坐标张量。

### 6.2 Copy_Traits 的两阶段设计

TMA 的 Copy_Traits 有一个独特的两阶段设计：

#### 阶段一 不可执行 Traits（持有描述符）

```cpp
template <int NumBitsPerTMA, class AuxParams>
struct Copy_Traits<SM90_TMA_LOAD, NumBitsPerTMA, AuxParams> {
  using ThrID   = Layout<_1>;                           // 单线程
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;   // (1, N) bit 布局
  using DstLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using RefLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  
  TmaDescriptor tma_desc_;       // 128 字节描述符
  AuxParams aux_params_;         // 辅助参数
  
  // 绑定 mbarrier 后变成可执行
  auto with(uint64_t& tma_mbar, uint16_t multicast_mask = 0, 
            uint64_t cache_hint = 0) {
    return Copy_Traits<SM90_TMA_LOAD_OP, NumBitsPerTMA>{
      opargs_{&tma_desc_, &tma_mbar, multicast_mask, cache_hint}};
  }
  
  // 构造 TMA 坐标张量
  template <class GShape>
  auto get_tma_tensor(GShape const& gshape) const { /* ... */ }
  
  // 不可执行状态禁用 copy_unpack
  void copy_unpack(...) = delete;
};
```

关键设计：
- `ThrID = Layout<_1>`：TMA 主需要一个线程启动
- `SrcLayout/DstLayout` 是 bit 级别的布局，`NumBitsPerTMA` 是一次 TMA 拷贝的总比特数
- `with()` 方法将不可执行 trait 转换为可执行 trait

#### 阶段二 可执行 Traits（绑定 mbarrier）

```cpp
template <int NumBitsPerTMA>
struct Copy_Traits<SM90_TMA_LOAD_OP, NumBitsPerTMA> {
  struct opargs_ {
    TmaDescriptor const* desc_;
    uint64_t* mbar_;
    uint16_t multicast_mask_;
    uint64_t cache_hint_;
  };
  
  opargs_ opargs_;
  
  // 友元 copy_unpack：从 CuTe Tensor 中提取参数调用 PTX
  template <class... Args>
  friend CUTE_DEVICE void
  copy_unpack(SM90_TMA_LOAD_OP, opargs_ const& opargs,
              Args const&... args) {
    // 从 src tensor 提取坐标
    // 从 dst tensor 提取 smem 地址
    // 调用 SM90_TMA_LOAD::copy() PTX 封装
  }
};
```

两阶段的原因在于两者的生命周期不同。描述符在 host 端创建一次、可被多次重用；mbarrier 在 kernel 内每次迭代都需要重新初始化。`.with()` 在每次使用时绑定当前迭代的 mbarrier，产生可执行 trait。

### 6.3 各种操作的 Traits

文件为以下操作都提供了类似的 Traits 特化：

| CopyOp | 特化要点 |
|--------|----------|
| `SM90_TMA_LOAD` | 有 mbar，两阶段 |
| `SM90_TMA_LOAD_MULTICAST` | 有 mbar + multicast_mask，两阶段 |
| `SM90_TMA_STORE` | 无 mbar（用 bulk_group），不需要 `.with()` |
| `SM90_TMA_REDUCE_ADD` | 无 mbar（用 bulk_group），不需要 `.with()` |
| `SM90_TMA_LOAD::PREFETCH` | 只需描述符，无 mbar |
| `SM90_BULK_COPY_G2S` | 有 mbar，`.with(bulk_mbar)` |
| `SM90_BULK_COPY_S2G` | 无 mbar（用 bulk_group） |
| `SM90_BULK_COPY_AUTO` | 占位符，根据方向自动选择 |

### 6.4 TMA_LOAD_Unpack 参数解包

```cpp
template <class CopyOp, class... Args>
struct TMA_LOAD_Unpack {
  template <class... Coords>
  friend CUTE_DEVICE void
  copy_unpack(CopyOp, Args const&... opargs,
              SrcTensor const& src, DstTensor const& dst) {
    // 从 src tensor 提取 TMA 坐标
    auto src_coord = ...;  
    // 从 dst tensor 提取 smem 地址
    auto dst_ptr = cast_smem_ptr_to_uint(dst.data());
    // 调用 PTX 封装
    detail::explode_tuple(detail::CallCOPY<CopyOp>{}, 
                          make_tuple(dst_ptr, opargs..., src_coord));
  }
};
```

这是连接 CuTe `copy()` 调用和底层 PTX 的关键：从 CuTe Tensor 中提取坐标和地址，组装成 PTX 指令所需的参数。

### 6.5 coalesce_256 处理 TMA box 大小限制

TMA 描述符的每个维度最大 256 个元素。`coalesce_256` 是一个特殊的布局合并函数，只合并到 size-256 为止：

```cpp
template <class Layout>
auto coalesce_256(Layout const& layout) {
  // 遍历布局的各个模式
  // 如果合并后某个模式不超过 256，则合并
  // 否则保持独立
}
```

这确保了构造的 TMA box 不会超过硬件限制。

### 6.6 construct_tma_gbasis 构造 TMA gmem basis

这是整个 TMA trait 构造中最核心的函数。它的目标是从 GMEM 张量布局和 SMEM 布局推导出 TMA 描述符所需的 gmem basis 布局。

```cpp
template <class TmaInternalType, class GEngine, class GLayout, 
          class SLayout, class VShape, class VStride>
auto construct_tma_gbasis(Tensor<GEngine,GLayout> const& gtensor,
                          SLayout const& slayout,
                          Layout<VShape,VStride> const& cta_v_map) {
  // 1. 求 SMEM 布局的逆（找到 SMEM 中最长的连续向量）
  auto inv_smem = right_inverse(get_nonswizzle_portion(slayout));
  
  // 2. 与 cta_v_map 组合，映射到 gmem 模式
  auto gmem_basis = composition(inv_smem, cta_v_map);
  
  // 3. 截断到 SMEM 大小
  // 4. recast 为 TmaInternalType
  // 5. coalesce_256 合并到 TMA 限制
  // 6. 分组到最大 rank-5
  
  return tma_gbasis;
}
```

核心思想是通过追踪 SMEM 的 "向量"（布局的逆），找到 SMEM 中最长的连续数组，然后映射回 GMEM，从而确定 TMA box 的形状和 gmem 步长。

### 6.7 fill_tma_gmem_shape_stride 填充形状和步长

```cpp
template <class GTensor, class TmaGbasisStride, 
          class GmemProbShape, class GmemProbStride>
void fill_tma_gmem_shape_stride(GTensor const& gtensor,
                                TmaGbasisStride const& tma_gbasis_stride,
                                GmemProbShape& gmem_prob_shape,
                                GmemProbStride& gmem_prob_stride) {
  // 使用 GCD 递推处理多模式贡献
  // 遍历 tma_gbasis_stride 的每个模式
  // 累积到 gmem_prob_shape 和 gmem_prob_stride
}
```

这个函数处理 GMEM 可能有多个模式映射到同一个 TMA 维度的情况，使用 GCD 递推正确计算形状和步长。

### 6.8 make_tma_copy_desc 创建 TMA 描述符

```cpp
template <class TmaInternalType, class GEngine, class GLayout,
          class TmaGbasis, class TmaSwizzle>
auto make_tma_copy_desc(Tensor<GEngine,GLayout> const& gtensor,
                        TmaGbasis const& tma_gbasis,
                        TmaSwizzle const& smem_swizzle,
                        uint32_t const& num_multicast) {
  // 1. 计算 gmem_prob_shape 和 gmem_prob_stride
  auto gmem_prob_shape = ...;
  auto gmem_prob_stride = ...;
  
  // 2. 计算 smem_box_shape 和 smem_box_stride
  auto smem_box_shape = ...;
  auto smem_box_stride = ...;
  
  // 3. 转换 swizzle
  auto swizzle_bits = detail::get_tma_swizzle_bits(smem_swizzle);
  auto swizzle_base = detail::get_tma_swizzle_base(smem_swizzle);
  
  // 4. 断言对齐和大小限制
  CUTE_ASSERT(...);
  
  // 5. 调用 CUDA Driver API 创建描述符
  TmaDescriptor tma_desc;
  cuTensorMapEncodeTiled(
    &tma_desc,
    to_CUtensorMapDataType<TmaInternalType>(),
    rank(gtensor),
    gtensor.data(),
    gmem_prob_shape, gmem_prob_stride,
    smem_box_shape, smem_box_stride,
    to_CUtensorMapSwizzle(swizzle_bits, swizzle_base),
    to_CUtensorMapFloatOOBfill(...),
    to_CUtensorMapL2promotion(...),
    CU_TENSOR_MAP_INTERLEAVE_NONE);
  
  return make_tuple(tma_desc, aux_params);
}
```

这是 host 端函数，调用 CUDA Driver API `cuTensorMapEncodeTiled` 实际创建 128 字节的 TMA 描述符。

### 6.9 make_tma_copy_atom 创建 Copy_Atom

```cpp
template <class TmaInternalType, class CopyOp, 
          class GEngine, class GLayout, class SLayout, 
          class VShape, class VStride>
auto make_tma_copy_atom(CopyOp, Tensor<GEngine,GLayout> const& gtensor,
                        SLayout const& slayout,
                        uint32_t const& num_multicast,
                        Layout<VShape,VStride> const& cta_v_map) {
  auto smem_swizzle = get_swizzle_portion(slayout);
  auto smem_layout  = get_nonswizzle_portion(slayout);
  
  auto tma_gbasis = detail::construct_tma_gbasis<TmaInternalType>(
      gtensor, smem_layout, cta_v_map);
  
  auto [tma_desc, aux_params] = detail::make_tma_copy_desc<TmaInternalType>(
      gtensor, tma_gbasis, smem_swizzle, num_multicast);
  
  constexpr int num_bits_per_tma = size(tma_gbasis) * sizeof_bits_v<TmaInternalType>;
  using Traits = Copy_Traits<CopyOp, C<num_bits_per_tma>, decltype(aux_params)>;
  using Atom   = Copy_Atom<Traits, typename GEngine::value_type>;
  
  Traits tma_traits{tma_desc, aux_params};
  return Atom{tma_traits};
}
```

### 6.10 make_tma_copy_tiled 创建 TiledCopy

```cpp
template <class TmaInternalType, class CopyOp, 
          class GEngine, class GLayout, class SLayout,
          class TShape, class TStride, class VShape, class VStride>
auto make_tma_copy_tiled(CopyOp const& copy_op,
                         Tensor<GEngine,GLayout> const& gtensor,
                         SLayout const& slayout,
                         Layout<TShape,TStride> const& cta_t_map,   // T: CTA thr -> logical TMA tid
                         Layout<VShape,VStride> const& cta_v_map)   // V: CTA val -> gmem mode
{
  Copy_Atom atom = make_tma_copy_atom<TmaInternalType>(
      copy_op, gtensor, slayout, cosize(cta_t_map), cta_v_map);
  
  auto cta_tiler = product_each(shape(cta_v_map));
  
  // 构造 TV 布局（线程-值分区）
  auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));
  auto num_elems_per_tma = ...;
  auto layout_v = composition(inv_smem_layout, num_elems_per_tma);
  auto layout_V = tile_to_shape(make_layout(layout_v), size(cta_v_map));
  auto layout_t = make_layout(cosize(cta_t_map), 
                              safe_div(num_elems_per_tma, cosize(cta_t_map)));
  auto layout_T = composition(inv_smem_layout, composition(layout_t, cta_t_map));
  auto layout_TV = make_layout(layout_T, layout_V);
  
  return TiledCopy<decltype(atom), decltype(layout_TV), decltype(cta_tiler)>{atom};
}
```

### 6.11 make_tma_copy 公共 API

这是用户直接调用的入口函数，有多个重载：

```cpp
// 完整版
template <class TmaInternalType = void, class CopyOp, 
          class GEngine, class GLayout, class SLayout,
          class CTA_Tiler, class Cluster_Size>
auto make_tma_copy(CopyOp const& copy_op,
                   Tensor<GEngine,GLayout> const& gtensor,  // GMEM 张量
                   SLayout const& slayout,                  // SMEM 布局
                   CTA_Tiler const& cta_tiler,              // CTA tile 形状
                   Cluster_Size const& cluster_size)        // 多播大小
{
  if constexpr (is_same_v<CopyOp, SM90_TMA_LOAD_IM2COL> || 
                is_same_v<CopyOp, SM90_TMA_STORE_IM2COL>) {
    return make_im2col_tma_copy(...);
  } else {
    auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);
    auto cta_t_tile = make_layout(cluster_size);
    return detail::make_tma_copy_tiled<TmaType>(copy_op, gtensor, slayout,
                                                cta_t_tile, cta_v_tile);
  }
}

// 默认 cta_tiler = product_each(shape(slayout)), cluster_size = 1
template <class CopyOp, class GEngine, class GLayout, class SLayout>
auto make_tma_copy(CopyOp const& copy_op,
                   Tensor<GEngine,GLayout> const& gtensor,
                   SLayout const& slayout) {
  return make_tma_copy(copy_op, gtensor, slayout, 
                       product_each(shape(slayout)), Int<1>{});
}
```

### 6.12 tma_partition 分区器

```cpp
template <class... Args, class CtaCoord, 
          class TShape, class TStride, class SEngine, class SLayout, 
          class... GTensors>
auto tma_partition(Copy_Atom<Args...> const& copy_atom,
                   CtaCoord const& cta_coord,
                   Layout<TShape,TStride> const& cta_layout,  // T: CTA coord -> multicast id
                   Tensor<SEngine,SLayout> const& stensor,    // SMEM (TMATile, Rest...)
                   GTensors const&... gtensors)               // GMEM (TMATile, Rest...)
{
  // 求 SMEM 逆布局
  Layout inv_smem_layout = right_inverse(get_nonswizzle_portion(layout<0>(stensor)));
  Layout layout_v = tile_to_shape(make_layout(inv_smem_layout), size<0>(stensor));
  
  // 分离单指令部分
  Layout tma_layout_v = make_layout(Int<Copy_Atom<Args...>::NumValSrc>{});
  auto layout_V = make_tile(logical_divide(layout_v, tma_layout_v));
  
  // 计算多播偏移
  auto multicast_offset = cta_layout(cta_coord) * 
      (size(tma_layout_v) / cosize(cta_layout));
  
  // 对每个 tensor 应用分区
  return cute::transform(make_tuple(gtensors..., stensor), [&](auto&& tensor) {
    auto tlayout_V = append<rank<decltype(tensor)>>(layout_V, _);
    Tensor tensor_v = coalesce(tensor.compose(tlayout_V), Shape<Shape<_1,_1>>{});
    auto coord = append<rank<decltype(tensor)>>(make_coord(make_coord(multicast_offset, Int<0>{})), Int<0>{});
    return domain_offset(coord, tensor_v);
  });
}
```

这个函数用于 TMA 多播场景下的 CTA 分区：不同 CTA 负责多播组内不同的部分。

### 6.13 create_tma_multicast_mask 多播掩码

```cpp
template <class CtaLayout, class CtaCoord>
uint16_t create_tma_multicast_mask(CtaLayout const& cta_layout_vmnk,
                                   CtaCoord const& cta_coord_vmnk) {
  auto [cta_layout, elected_cta] = slice_and_offset(cta_coord_vmnk, cta_layout_vmnk);
  
  uint16_t mcast_mask = 0;
  // 根据 cta_layout 计算掩码
  // ...
  mcast_mask <<= elected_cta;
  return mcast_mask;
}
```

根据 cluster 内 CTA 的布局和当前 CTA 的坐标，计算 16 位多播掩码。

### 6.14 make_tma_copy_A/B/C_sm90 矩阵乘法便捷函数

这三个函数是专门为 GEMM 场景定制的便捷接口：

```cpp
// A 矩阵：保留 MK 模式，N 方向多播
template <class TmaInternalType = void, class CopyOp, 
          class GEngine, class GLayout, class SLayout,
          class CTA_Tiler, class Cluster_Size>
auto make_tma_copy_A_sm90(CopyOp const& copy_op,
                          Tensor<GEngine,GLayout> const& gtensor,
                          SLayout const& slayout,
                          CTA_Tiler const& cta_tiler,
                          Cluster_Size const& cluster_size) {
  auto cta_tiler_mk = remove<1>(cta_tiler);   // 去掉 N 维度
  auto cluster_size_n = size<1>(cluster_size); // N 方向多播
  // ...
  return detail::make_tma_copy_tiled<TmaType>(copy_op, gtensor, slayout,
                                              cta_t_tile, cta_v_tile);
}

// B 矩阵：保留 NK 模式，M 方向多播
auto make_tma_copy_B_sm90(...) {
  auto cta_tiler_nk = remove<0>(cta_tiler);   // 去掉 M 维度
  auto cluster_size_m = size<0>(cluster_size); // M 方向多播
  // ...
}

// C 矩阵：保留 MN 模式，不多播
auto make_tma_copy_C_sm90(...) {
  auto cta_tiler_mn = remove<2>(cta_tiler);   // 去掉 K 维度
  // No multicast
  auto cta_t_map = Layout<_1,_0>{};
  // ...
}
```

矩阵乘法中 A/B/C 的 TMA 需求差异：
- A (M×K)：每个 CTA 需要 M×K tile，沿 N 方向的 CTA 可以共享同一份 A，所以 N 方向多播
- B (N×K)：每个 CTA 需要 N×K tile，沿 M 方向的 CTA 可以共享同一份 B，所以 M 方向多播
- C (M×N)：输出矩阵，不多播，每个 CTA 写自己的部分

---

## 7. TMA Copy 端到端流程

将四层文件串联起来，TMA 拷贝的完整流程如下：

### 7.1 Host 端创建 TMA 描述符

```
用户调用 make_tma_copy(SM90_TMA_LOAD{}, gtensor, slayout, cta_tiler, cluster_size)
         │
         ▼
make_tma_copy_tiled(copy_op, gtensor, slayout, cta_t_tile, cta_v_tile)
         │
         ├──► make_tma_copy_atom(copy_op, gtensor, slayout, num_multicast, cta_v_map)
         │         │
         │         ├──► get_swizzle_portion(slayout)   ──► copy_traits_sm90_tma_swizzle.hpp
         │         ├──► get_nonswizzle_portion(slayout)
         │         ├──► construct_tma_gbasis(gtensor, smem_layout, cta_v_map)
         │         │         │
         │         │         ├──► right_inverse(smem_layout)  // 求 SMEM 逆
         │         │         ├──► composition(inv_smem, cta_v_map)
         │         │         ├──► truncate / recast / coalesce_256
         │         │         └──► 分组到 rank-5
         │         │
         │         ├──► make_tma_copy_desc(gtensor, tma_gbasis, smem_swizzle, num_multicast)
         │         │         │
         │         │         ├──► fill_tma_gmem_shape_stride(...)
         │         │         ├──► get_tma_swizzle_bits(smem_swizzle)  ──► swizzle.hpp
         │         │         ├──► get_tma_swizzle_base(smem_swizzle)  ──► swizzle.hpp
         │         │         ├──► to_CUtensorMapDataType<T>()          ──► desc.hpp
         │         │         └──► cuTensorMapEncodeTiled(...)          // CUDA Driver API
         │         │
         │         └──► 构造 Copy_Traits<SM90_TMA_LOAD, NumBitsPerTMA, AuxParams>
         │              └──► 包装成 Copy_Atom<Traits, T>
         │
         ├──► 构造 TiledCopy 的 layout_TV
         │     ├──► layout_V = composition(inv_smem_layout, num_elems_per_tma)
         │     ├──► layout_T = composition(inv_smem_layout, composition(layout_t, cta_t_map))
         │     └──► layout_TV = make_layout(layout_T, layout_V)
         │
         └──► 返回 TiledCopy{atom}
```

### 7.2 Device 端执行 TMA 拷贝

```
1. 准备 mbarrier
   ├── initialize_barrier(mbar, 1)
   └── set_barrier_transaction_bytes(mbar, num_bytes)

2. 获取 TMA 坐标张量
   ├── mA_tma = tma.get_tma_tensor(make_shape(M, N))   // (M,N) TMA coord tensor
   └── gA = local_tile(mA_tma, cta_tile, cta_coord)    // 当前 CTA 的 tile

3. 分区
   ├── cta_tma = tma.get_slice(cta_idx_in_cluster)
   ├── tAgA = cta_tma.partition_S(gA)   // src 分区
   └── tAsA = cta_tma.partition_D(sA)   // dst 分区

4. 执行拷贝
   └── copy(tma.with(mbar, mcast_mask), tAgA, tAsA)
            │
            ▼
       tma.with(mbar, mcast_mask) 产生 Copy_Traits<SM90_TMA_LOAD_OP>
            │
            ▼
       copy_unpack(SM90_TMA_LOAD_OP, opargs, src_tensor, dst_tensor)
            │
            ├──► 从 src_tensor 提取 TMA 坐标 (crd0, crd1, ...)
            ├──► 从 dst_tensor 提取 smem_addr
            └──► SM90_TMA_LOAD::copy(desc_ptr, smem_addr, crd0, ..., mbar_ptr)
                    │
                    ▼
                cp.async.bulk.tensor.Nd.shared::cluster.global.mbarrier::complete_tx::bytes
                    [smem_addr], [desc_ptr, {crd0,...}], [mbar_ptr];
                    │
                    ▼
                硬件执行 TMA 拷贝 ──► 完成后自动 arrive mbarrier

5. 等待完成
   └── wait_barrier(mbar, phase)
            │
            ▼
       mbarrier.try_wait.parity.shared.b64 _, [mbar_ptr], phase;

6. 安全使用 SMEM 数据
   └── 进行 MMA 计算...
```

---

## 8. 完整使用示例

### 8.1 简单 2D TMA Load

```cpp
using T = float;
T* gptr = /* ... */;

// GMEM 张量：1024×256，行主序
Tensor gtensor = make_tensor(gptr, make_shape(1024, 256), GenRowMajor{});

// SMEM 布局：64×32，行主序
auto slayout = make_layout(make_shape(_64{}, _32{}), GenRowMajor{});

// 创建 TMA copy
auto tma = make_tma_copy(SM90_TMA_LOAD{}, gtensor, slayout);

// 在 kernel 中使用
__device__ void kernel(/* ... */) {
  // mbarrier
  __shared__ uint64_t mbar;
  initialize_barrier(mbar, 1);
  set_barrier_transaction_bytes(mbar, sizeof(T) * 64 * 32);
  
  // TMA 坐标张量
  Tensor mA = tma.get_tma_tensor(make_shape(1024, 256));
  Tensor gA = local_tile(mA, make_shape(_64{}, _32{}), make_coord(bx, by));
  
  // SMEM 张量
  __shared__ T smem[64 * 32];
  Tensor sA = make_tensor(make_smem_ptr(smem), slayout);
  
  // 分区
  auto cta_tma = tma.get_slice(0);
  Tensor tAgA = cta_tma.partition_S(gA);
  Tensor tAsA = cta_tma.partition_D(sA);
  
  // 执行拷贝
  copy(tma.with(mbar), tAgA, tAsA);
  
  // 等待
  wait_barrier(mbar, 0);
  
  // 使用数据...
}
```

### 8.2 带 Swizzle 的 GMMA 风格 TMA Load

```cpp
using T = half_t;
T* gptr = /* ... */;

// GMEM 张量：1024×256，MN-Major
Tensor gtensor = make_tensor(gptr, make_shape(1024, 256));

// SMEM 布局：GMMA 风格的 swizzled 128×64
auto slayout = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, make_shape(_128{}, _64{}));

// 创建 TMA copy（自动识别 swizzle）
auto tma = make_tma_copy(SM90_TMA_LOAD{}, gtensor, slayout);

// tma 内部已经正确设置了 TMA 描述符的 swizzle 模式
// 使用方式与上例相同
```

### 8.3 多播 TMA Load（Cluster 场景）

```cpp
// cluster_size = 4，沿 N 方向多播 A 矩阵
auto tma = make_tma_copy_A_sm90(SM90_TMA_LOAD_MULTICAST{}, gtensor_A, slayout_A,
                                cta_tiler, Shape<_1, _4>{});

__device__ void kernel() {
  __shared__ uint64_t mbar;
  initialize_barrier(mbar, 1);
  set_barrier_transaction_bytes(mbar, num_bytes);
  
  // 计算当前 CTA 的多播掩码
  uint16_t mcast_mask = create_tma_multicast_mask(cta_layout, cta_coord);
  
  // 分区（根据 CTA 在 cluster 中的位置）
  auto cta_tma = tma.get_slice(cta_idx_in_cluster);
  Tensor tAgA = cta_tma.partition_S(gA);
  Tensor tAsA = cta_tma.partition_D(sA);
  
  // 带多播掩码执行拷贝
  copy(tma.with(mbar, mcast_mask), tAgA, tAsA);
  
  wait_barrier(mbar, 0);
}
```

### 8.4 TMA Store

```cpp
// 创建 TMA store（不需要 mbarrier）
auto tma_store = make_tma_copy(SM90_TMA_STORE{}, gtensor, slayout);

__device__ void kernel() {
  // SMEM → GMEM
  copy(tma_store, sA, gA);
  
  // Store 用 bulk_group 同步，需要 commit + wait
  cp_async_bulk_commit_group();
  cp_async_bulk_wait_group<0>();
}
```

### 8.5 网格持久化 + 设备端描述符修改

```cpp
// Host: 创建模板描述符
auto tma_template = make_tma_copy(SM90_TMA_LOAD{}, gtensor, slayout);

__device__ void kernel() {
  // 设备端：根据当前 tile 修改描述符地址
  __shared__ TmaDescriptor desc_smem;
  tma_descriptor_replace_addr_in_shared_mem(&desc_smem, current_gmem_addr);
  tma_descriptor_fence_release();
  tma_descriptor_fence_acquire();
  
  // 使用修改后的描述符执行 TMA
  // ...
}
```

---

## 9. 总结

### 9.1 四层架构的设计哲学

CUTLASS 将 TMA 功能分为四层，体现了关注点分离的设计哲学：

| 层次 | 关注点 | 关键类型/函数 |
|------|--------|---------------|
| desc.hpp | 硬件数据结构 | `TmaDescriptor`, `mbarrier`, `SmemSwizzleBits` |
| copy_sm90_tma.hpp | PTX 指令封装 | `SM90_TMA_LOAD`, `SM90_TMA_STORE` |
| swizzle.hpp | 布局转换 | `get_tma_swizzle_bits`, `get_tma_swizzle_base` |
| copy_traits_sm90_tma.hpp | CuTe 抽象衔接 | `Copy_Traits`, `make_tma_copy`, `tma_partition` |

### 9.2 关键设计决策

1. 两阶段 Traits：描述符 (host 创建，可重用) 与 mbarrier (device 每次迭代需要新的) 分离，通过 `.with()` 绑定。

2. 自动 basis 构造：`construct_tma_gbasis` 自动从 SMEM 布局逆推 TMA box 形状，用户无需手动指定。

3. Swizzle 自动转换：CuTe 的 `Swizzle<B,M,S>` 自动映射到 TMA 的 swizzle 枚举，无需用户干预。

4. GEMM 便捷函数：`make_tma_copy_A/B/C_sm90` 封装了矩阵乘法中 A/B/C 矩阵的不同多播需求。

5. 设备端描述符修改：通过 `tensormap.replace.tile` PTX 指令支持网格持久化模式。

### 9.3 与传统 cp.async 的对比

| 特性 | cp.async | TMA |
|------|----------|-----|
| 拷贝粒度 | 每线程 4/8/16 字节 | CTA 集体，单指令拷贝整个 tile |
| 寄存器开销 | 每线程持有地址和坐标 | 零寄存器（描述符 + 坐标在 SMEM） |
| 多维感知 | 否（需手动计算偏移） | 是（硬件理解张量形状） |
| Swizzle | 不支持 | 原生支持 |
| OOB 处理 | 需手动 predicate | 硬件自动填充零/常量 |
| 多播 | 不支持 | 支持 cluster 内多播 |
| 同步 | cp.async.wait_group | mbarrier |
| 适用场景 | 小规模、简单拷贝 | 大规模张量 tile 拷贝 |

TMA 是 Hopper 架构实现高效张量核心计算的关键基础设施。CUTLASS 通过这四层抽象把复杂的硬件特性以类型安全的方式暴露给用户，同时保留了多播、im2col、网格持久化等高级场景所需的灵活性。
