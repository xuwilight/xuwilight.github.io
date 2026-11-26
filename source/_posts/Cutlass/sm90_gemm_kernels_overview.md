---
title: Cutlass 学习笔记（六）SM90 GEMM Kernel Overview
date: 2025-09-01 12:00:00
tags: [CUTLASS, GEMM]
categories: [Cutlass 学习笔记, Cutlass]
description: 梳理 CUTLASS 4.5.0 中 6 个 SM90 半精度 GEMM kernel 与 FP8 的两条 mainloop 路线，涵盖 Warp 特化、多 Consumer 策略、Ptr-Array 支持与 FP8 的 GmmaFP8Accumulation 提升机制。
published: true
render:
  math: mathjax
---

CUTLASS 为 SM90 (Hopper) 提供了 6 个半精度 GEMM kernel，沿 Warp 特化、多 Consumer 策略和 Ptr-Array 支持三个维度逐层演进，接口统一在 GemmUniversal 与 CollectiveBuilder 之下。

它们从最简单的 KernelTma 出发，先引入 Producer/Consumer 分离，再分出 Pingpong 的 tile 交替与 Cooperative 的同 tile 协作，最后叠加运行时 TensorMap 更新以支持 Batched 与 Grouped GEMM。除半精度外，FP8 复用这些 kernel 骨架、只把 mainloop 换成 FP8 特化版本，按 scale 粒度分 per-tensor 与 blockwise 两条路线。

下文先按三条主线梳理 6 个半精度 kernel，再单独讲 FP8 的 mainloop 特化，最后给出调度策略、Scheduler 选择与选型对照。


## 1. SM90 GEMM Kernel 家族概览

CUTLASS 为 SM90 (Hopper) 架构提供了 6 个半精度 GEMM kernel，按复杂度递增排列。

| # | Kernel 文件 | 调度策略 | 线程数 | 核心特点 |
|---|-----------|---------|--------|---------|
| 1 | sm90_gemm_tma.hpp | KernelTma | 128 | 最简单，非 warp 特化 |
| 2 | sm90_gemm_tma_warpspecialized.hpp | KernelTmaWarpSpecialized | 256 | 1 Producer + 1 Consumer |
| 3 | sm90_gemm_tma_warpspecialized_pingpong.hpp | KernelTmaWarpSpecializedPingpong | 384 | 1 Producer + 2 交替 Consumer |
| 4 | sm90_gemm_tma_warpspecialized_cooperative.hpp | KernelTmaWarpSpecializedCooperative | 384 | 1 Producer + 2 协作 Consumer |
| 5 | sm90_gemm_array_tma_warpspecialized_pingpong.hpp | KernelPtrArrayTmaWarpSpecializedPingpong | 384 | #3 + Ptr-Array/Grouped |
| 6 | sm90_gemm_array_tma_warpspecialized_cooperative.hpp | KernelPtrArrayTmaWarpSpecializedCooperative | 384 | #4 + Ptr-Array/Grouped |

所有 kernel 都是 GemmUniversal 模板类的特化版本，通过 SFINAE 机制根据 DispatchPolicy::Schedule 选择。线程数按 CTA 的 launch 配置计：Producer 以整个 warp group（128 线程）为粒度预留，即使基础 WS 版实际只有 Warp 0 发 TMA，也占满 128 线程，所以 WS 是 128 Consumer + 128 Producer = 256。

---

## 2. Kernel 之间的演化关系

```
sm90_gemm_tma (基础)
    │
    │ + Warp 特化 (Producer/Consumer 分离)
    ▼
sm90_gemm_tma_warpspecialized (基础 WS)
    │
    │ + 第二个 Consumer
    ├──→ Pingpong 路线 (交替不同 tile)
    │    │
    │    ├──→ sm90_gemm_tma_warpspecialized_pingpong
    │    │
    │    │ + TensorMap 运行时更新
    │    └──→ sm90_gemm_array_tma_warpspecialized_pingpong
    │
    └──→ Cooperative 路线 (协作同一 tile)
         │
         ├──→ sm90_gemm_tma_warpspecialized_cooperative
         │
         │ + TensorMap 运行时更新
         └──→ sm90_gemm_array_tma_warpspecialized_cooperative
```

### 演化逻辑

1. sm90_gemm_tma → sm90_gemm_tma_warpspecialized：引入 Warp 特化，分离加载和计算
2. sm90_gemm_tma_warpspecialized → Pingpong/Cooperative：增加第二个 Consumer，两种策略
3. Pingpong/Cooperative → Array Pingpong/Array Cooperative：增加 TensorMap 运行时更新

---

## 3. Warp Specification

### 3.1 非特化 (sm90_gemm_tma)

CTA (128 threads = 4 warps)，所有 warp 做 所有事情。

- 所有线程共同参与 TMA 加载和 WGMMA 计算
- Mainloop 通过 collective_mma.operator() 一次性调用
- 加载和计算不能重叠

### 3.2 Warp 特化 (其他 5 个 kernel)

Producer Warp Group 专门负责 TMA 加载，Consumer Warp Group(s) 专门负责 WGMMA 计算。

- Producer 和 Consumer 分离，通过异步 pipeline 通信
- 加载和计算可以重叠（Producer 加载下一个 tile 时，Consumer 计算当前 tile）
- Mainloop 拆分为 load() + mma() 分离调用

### 3.3 Producer Warp 组的角色细分

在 Pingpong/Cooperative 中，Producer warp 组（128 线程 = 4 warp）进一步细分为：

| Warp | 普通 Pingpong/Coop | Array Pingpong/Coop |
|:------:|:-------------------:|:---------------------:|
| Warp 0 | Mainloop (TMA A/B) | Mainloop (TMA A/B) |
| Warp 1 | Warp1 (Scheduler) | MainloopAux (辅助加载) |
| Warp 2 | Epilogue (TMA C) | Epilogue (TMA C) |
| Warp 3 | MainloopAux (辅助) | Scheduler (GroupScheduler) |

Array 版本的 Warp 1 和 Warp 3 角色互换，因为 GroupScheduler 需要专门的 warp。

普通 Pingpong/Coop 里 Warp 1 的调度分支对应 CLC（Cluster Launch Control）驱动的动态持久化调度，但 CLC 是 SM100 及以上架构的指令，SM90 的 PersistentTileSchedulerSm90 里 IsDynamicPersistent = false，所以这一分支在 SM90 上不执行，Warp 1 实际闲置。

---

## 4. Pingpong 与 Cooperative

### 4.1 基本 WS（1 Consumer）

```
Producer:  [Load 0] [Load 1] [Load 2] ...
Consumer:  [MMA 0] [Epi 0] [MMA 1] [Epi 1] [MMA 2] [Epi 2] ...
                     ↑ 串行 ↑
```

MMA 和 Epilogue 在同一个 Consumer 上串行执行，Epilogue 期间 MMA 空闲。

### 4.2 Pingpong（2 Consumer 交替）

```
Consumer0: [MMA 0] [Epi 0] [MMA 2] [Epi 2] ...
Consumer1:         [MMA 1] [Epi 1] [MMA 3] [Epi 3] ...
                     ↕ 重叠 ↕
```

- 2 个 Consumer 处理不同的 tile
- C0 的 Epilogue 与 C1 的 MMA 重叠
- 通过 MathWarpGroupOrderBarrier<2,2> 协调交替
- TiledMma = 128 线程，Pipeline 推进 ×2

### 4.3 Cooperative（2 Consumer 协作）

```
Consumer0 + Consumer1: [MMA 0 (256线程)] [Epi 0] [MMA 1 (256线程)] [Epi 1] ...
```

- 2 个 Consumer 共同处理同一个 tile
- 256 线程的 WGMMA 处理更大的 tile
- 不需要 MathWarpGroupOrderBarrier
- TiledMma = 256 线程，Pipeline 推进 ×1

### 4.4 Pingpong vs Cooperative 核心对比

| 特性 | Pingpong | Cooperative |
|:------:|:----------:|:-------------:|
| 优化目标 | 隐藏 Epilogue 延迟 | 提高 MMA 吞吐量 |
| Consumer 工作方式 | 交替不同 tile | 协作同一 tile |
| TiledMma 线程数 | 128 | 256 |
| Tile 大小 | 中等 (128×128) | 大 (256×128 或更大) |
| MathWarpGroupOrderBarrier | 有 | 无 |
| StreamK | 不支持 | 支持 |
| Pipeline 推进 | ×2 | ×1 |

---

## 5. Ptr-Array 与 Grouped GEMM

### 5.1 普通 GEMM vs Ptr-Array vs Grouped

```
普通 GEMM:
  C = A × B
  一个矩阵乘法

Batched GEMM:
  C[l] = A[l] × B[l],  l = 0, 1, ..., L-1
  多个相同大小的矩阵乘法，stride 固定

Ptr-Array GEMM:
  C[l] = A[l] × B[l],  l = 0, 1, ..., L-1
  多个相同大小的矩阵乘法，每个 batch 指针独立

Grouped GEMM:
  C[g] = A[g] × B[g],  g = 0, 1, ..., G-1
  多个不同大小的矩阵乘法，每个 group 独立
```

### 5.2 TensorMap 运行时更新

Hopper 的 TMA 使用 TensorMap 描述张量布局。对于 Ptr-Array/Grouped GEMM，每个 batch/group 有不同的指针和形状，需要运行时更新 TensorMap：

```
Host 端:
  1. 为每个 SM 创建 TensorMap 副本 → device memory

Device 端 (Kernel 内):
  2. 从 device memory 加载 TensorMap → shared memory
  3. 修改 TensorMap (tensormaps_perform_update)
  4. 释放 fence (tensormaps_cp_fence_release)
  5. 获取 fence (tensormaps_fence_acquire)
  6. 使用更新后的 TensorMap 进行 TMA
```

### 5.3 Array Kernel 的额外特性

| 特性 | 普通 Kernel | Array Kernel |
|:------:|:------------:|:-------------:|
| TensorMap | Host 预创建，不变 | Device 运行时更新 |
| TensorMapStorage | 无 | 有（在 SharedStorage 中） |
| Workspace | 调度器 | 调度器 + mainloop + epilogue |
| TMA Prefetch | 支持 | 不支持（运行时更新） |
| ProblemShape | <M,N,K,L> | 特殊类型（含 get_problem_shape()） |
| GroupScheduler | 不支持 | 支持 |
| TileSchedulerStages | 由策略决定 | 8（固定） |
| IsGroupedGemmKernel | 无 | 有（通过 StrideA 检测） |

---

## 6. FP8 的两条 mainloop 路线

FP8 没有新增 kernel 文件，而是复用 warp specification 的 384 线程骨架，把 mainloop 换成 FP8 特化版本。按 scale 粒度分两条路线。

| | per-tensor | blockwise |
|---|---|---|
| scale 形态 | 标量 | 张量（每 128×128 一块） |
| 例子 | Example 54 | Example 67 |
| 调度策略 | KernelTmaWarpSpecializedCooperative（FP8 输入自动走特化） | KernelTmaWarpSpecializedCooperativeFP8Blockwise |
| mainloop 文件 | sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp | ..._fp8_blockwise_scaling.hpp |
| 提升阶段做什么 | 临时累加器加到主累加器（FADD） | 同上再乘 block scale（FFMA） |
| scale 应用位置 | epilogue，合并进 alpha | mainloop 提升阶段 |

两条路线共享 fp8_accumulation.hpp 的 GmmaFP8Accumulation：WGMMA 只写一块临时累加器，每累加 mma_promotion_interval（默认 4）次 MMA 就把这块小和加到 FP32 主累加器、临时累加器清零，用这个办法把有限尾数带来的累加舍入误差限制在小范围。per-tensor 在这一步只做加法，blockwise 多做一次 scale_a×scale_b 的乘法。

临时累加器（accm_temp = make_fragment_like(accum)）让每线程的累加器寄存器翻倍，这是决定 FP8 只能用 cooperative 的关键。cooperative 沿 M 把 tile 拆给两个 warp group、主累加器被摊薄，M128N128 时单份 64 寄存器、翻倍后 128；pingpong 的单个 warp group 处理整个 tile，翻倍后超出每线程上限。因此 FP8 实际只用 cooperative（或退到 M=64 的 pingpong），选型上 FP8 几乎等价于选 cooperative。

还有一组带 FP8FastAccum 后缀的调度名（如 KernelTmaWarpSpecializedCooperativeFP8FastAccum）。它们不触发上面的 FP8 特化 mainloop，而是由 builder 路由到普通 sm90_mma_tma_gmma_ss.hpp，把 FP8 乘积直接累加进单个 FP32 累加器，省掉临时累加器和定期提升，以更少的加法与寄存器换吞吐。

---

## 7. 技术特性对比矩阵

| 特性 | Tma | WS | Pong | Coop | ArrPong | ArrCoop |
|:------:|:-----:|:-----:|:------:|:------:|:---------:|:---------:|
| 线程数 | 128 | 256 | 384 | 384 | 384 | 384 |
| Warp 特化 | 否 | 是 | 是 | 是 | 是 | 是 |
| Consumer 数 | 1 | 1 | 2 | 2 | 2 | 2 |
| TiledMma 线程 | 128 | 128 | 128 | 256 | 128 | 256 |
| Consumer 策略 | - | 串行 | 交替 | 协作 | 交替 | 协作 |
| Pipeline 推进 | ×1 | ×1 | ×2 | ×1 | ×2 | ×1 |
| 持久化 | 否 | 是 | 是 | 是 | 是 | 是 |
| Ptr-Array | 否 | 否 | 否 | 否 | 是 | 是 |
| Grouped GEMM | 否 | 否 | 否 | 否 | 是 | 是 |
| TensorMap 更新 | 否 | 否 | 否 | 否 | 是 | 是 |
| GroupScheduler | 否 | 否 | 否 | 否 | 是 | 是 |
| TMA Prefetch | 是 | 是 | 是 | 是 | 否 | 否 |
| Reg 重分配 | 否 | 是 | 是 | 是 | 是 | 是 |
| Workspace | 0 | 调度器 | 调度器 | 调度器+epi | +mainloop | +mainloop |
| Tile M 要求 | 无 | 无 | 无 | ≥128 | 无 | ≥128 |

FP8 的两个 mainloop 沿用 cooperative 一行（384 线程、协作、StreamK 支持），只是把累加换成了 GmmaFP8Accumulation 的临时累加器机制。

---

## 8. 调度策略与 SFINAE 分发

### 8.1 调度策略继承关系

```
cutlass::gemm::KernelTma
    └── KernelTmaWarpSpecialized
            ├── KernelTmaWarpSpecializedPingpong
            └── KernelTmaWarpSpecializedCooperative

cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong
cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative

// FP8 相关（都继承自对应的半精度策略，只是作为 builder 选路的标签）
KernelTmaWarpSpecializedCooperativeFP8Blockwise : KernelTmaWarpSpecializedCooperative
KernelTmaWarpSpecializedCooperativeFP8FastAccum : KernelTmaWarpSpecializedCooperative
```

### 8.2 SFINAE 分发机制

GemmUniversal 是一个主模板，通过第 5 个模板参数（enable_if_t）进行 SFINAE 分发：

```cpp
// sm90_gemm_tma.hpp
template <...>
class GemmUniversal<..., 
    enable_if_t<is_base_of_v<KernelTma, Schedule>>> { ... }

// sm90_gemm_tma_warpspecialized.hpp
template <...>
class GemmUniversal<..., 
    enable_if_t<is_base_of_v<KernelTmaWarpSpecialized, Schedule>>> { ... }

// sm90_gemm_tma_warpspecialized_pingpong.hpp
template <...>
class GemmUniversal<..., 
    enable_if_t<is_base_of_v<KernelTmaWarpSpecializedPingpong, Schedule>>> { ... }

// sm90_gemm_tma_warpspecialized_cooperative.hpp
template <...>
class GemmUniversal<..., 
    enable_if_t<is_base_of_v<KernelTmaWarpSpecializedCooperative, Schedule>>> { ... }
```

由于 KernelTmaWarpSpecializedPingpong 继承自 KernelTmaWarpSpecialized，后者继承自 KernelTma，SFINAE 会选择最特化的匹配。FP8 的 ...FP8Blockwise、...FP8FastAccum 都继承自 KernelTmaWarpSpecializedCooperative，所以同一个 cooperative kernel 文件同时承载半精度、per-tensor FP8、blockwise FP8 与 FastAccum FP8，区别全在 builder 选出的 mainloop 上。

### 8.3 用户的选择入口

用户通过 CollectiveBuilder 的 Schedule 参数选择：

```cpp
using CollectiveMainloop = typename CollectiveBuilder<
    Sm90, OpClassTensorOp,
    ElementA, LayoutA, ElementB, LayoutB, ElementC,
    TileShape, StageCount<3>,
    KernelTmaWarpSpecializedPingpong  // <-- 这里选择
>::CollectiveOp;
```

对 FP8，只要把 ElementA/ElementB 设成 float_e4m3_t 之类的 FP8 类型，builder 的 IsFP8Input 分支会自动把 mainloop 切成 FP8 特化版，用户不必显式指定 FP8 的 DispatchPolicy。

---

## 9. Tile Scheduler 选择

### 9.1 Scheduler 类型

| Scheduler | 适用 Kernel | 特点 |
|:-----------:|:------------:|:------:|
| PersistentScheduler | Tma, WS | 基本持久化调度 |
| StreamKScheduler | WS, Cooperative | StreamK 负载均衡 |
| GroupScheduler | Array kernels | 分组调度，8 级管道 |

### 9.2 各 Kernel 的 Scheduler 支持

| Kernel | PersistentScheduler | StreamKScheduler | GroupScheduler |
|:--------:|:-------------------:|:-----------------:|:---------------:|
| Tma | ✓ | ✗ | ✗ |
| WS | ✓ | ✓ | ✗ |
| Pong | ✓ | ✗ | ✗ |
| Coop | ✓ | ✓ | ✗ |
| Fp8-PT | ✓ | ✓ | ✗ |
| Fp8-BW | ✓ | ✓ | ✗ |
| ArrPong | ✓ | ✓ | ✓ |
| ArrCoop | ✓ | ✓ | ✓ |

### 9.3 SchedulerPipelineStageCount

- 普通 kernel: 由 DispatchPolicy::Schedule::SchedulerPipelineStageCount 决定
- Array kernel: 固定 8 级（GroupScheduler 需要更多预取）

---

## 10. 共享基础设施

### 10.1 CollectiveMainloop

所有 kernel 使用 CollectiveMainloop 封装 mainloop 逻辑：

| Mainloop 实现 | 适用 Kernel |
|--------------|------------|
| sm90_mma_tma_gmma_ss.hpp (SS) | Tma, WS (Smem-Smem)，以及 FP8 FastAccum |
| sm90_mma_tma_gmma_rs.hpp (RS) | WS (Register-Smem) |
| sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp | FP8 per-tensor（GmmaFP8Accumulation 提升） |
| sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp | FP8 blockwise（提升时乘 block scale） |
| Array 版本的 mainloop | Array kernels (带 TensorMap 更新) |

### 10.2 CollectiveEpilogue

| Epilogue 实现 | 适用 Kernel |
|--------------|------------|
| sm90_epilogue_tma.hpp | Tma |
| sm90_epilogue_tma_warpspecialized.hpp | WS, Pong, Coop |
| Array 版本的 epilogue | Array kernels (带 TensorMap 更新) |

非特化的 KernelTma 用简单的 LinearCombination；warp-specialized（含 FP8）启用 EVT（Epilogue Visitor Tree）。FP8 的 epilogue 把整条输出公式 ScaledLinCombPerRowBiasEltActAmaxAux 编成一棵 EVT，一次算完 scale、bias、激活、amax 与量化。

### 10.3 Pipeline

所有 warp 特化 kernel 共享 sm90_pipeline.hpp 中的 pipeline 实现：

| Pipeline 类型 | 用途 |
|--------------|------|
| PipelineTmaAsync | Mainloop A/B 加载 |
| PipelineTransactionAsync | Epilogue C 加载 |
| PipelineTmaStore | Epilogue D 存储 |
| OrderedSequenceBarrier | Warp 组间同步 |

### 10.4 关键共享组件

- TMA (Tensor Memory Accelerator)：所有 kernel 使用 TMA 进行异步数据搬运
- WGMMA (Warp Group MMA)：所有 kernel 使用 WGMMA 进行矩阵乘法
- Cluster：支持 CTA 群组的共享内存和同步
- Register Reconfiguration：warpgroup_reg_dealloc/alloc 寄存器重分配
- GDC (Grid Dependency Control)：硬件级 grid 依赖（部分 kernel）

---
<!-- 
## 文章清单

每个算子都有对应的逐行剖析文章：

| 主题 | 剖析文章 |
|------|---------|
| Tma（非特化） | CUTLASS SM90 GEMM TMA 算子剖析 |
| WarpSpecialized（基础 WS） | CUTLASS SM90 GEMM TMA WarpSpecialized 算子剖析 |
| Pingpong | CUTLASS SM90 GEMM TMA WarpSpecialized Pingpong 算子剖析 |
| Cooperative | CUTLASS SM90 GEMM TMA WarpSpecialized Cooperative 算子剖析 |
| Array Pingpong | CUTLASS SM90 GEMM Array TMA WarpSpecialized Pingpong 算子 |
| Array Cooperative | CUTLASS SM90 GEMM Array TMA WarpSpecialized Cooperative 算子 |
| FP8 per-tensor | CUTLASS SM90 FP8 per-tensor 算子实现剖析 |
| FP8 blockwise | CUTLASS SM90 FP8 blockwise scaling 算子实现剖析 |
| Tile Scheduler | CUTLASS SM90 Tile Scheduler 调度机制剖析 |
| Pipeline | CUTLASS SM90 流水线 Pipeline 机制剖析 |

--- -->

## 总结

CUTLASS SM90 GEMM Kernel 家族通过四个正交维度的演化，覆盖了几乎所有 GEMM 场景：

1. Warp 特化维度：从非特化（Tma）到特化（WS），实现加载/计算重叠
2. 多 Consumer 策略维度：从串行（WS）到交替（Pingpong）或协作（Cooperative），优化不同瓶颈
3. 数据布局维度：从固定 TensorMap（普通）到运行时更新（Array），支持不规则 GEMM
4. 数值精度维度：从半精度到 FP8，复用 cooperative 骨架、换用带 GmmaFP8Accumulation 的 mainloop

### 核心设计哲学

- 简单问题用简单 kernel：sm90_gemm_tma 足以处理简单场景
- 针对性优化：Pingpong 优化 Epilogue 延迟，Cooperative 优化 MMA 吞吐
- 组合能力：Array + Pingpong/Cooperative 组合支持最复杂的场景，FP8 在 cooperative 之上再叠一层数值特化
- 统一接口：所有 kernel 通过 GemmUniversal + CollectiveBuilder 统一接口

### Hopper 架构利用

这些 kernel 充分利用了 Hopper 架构的所有关键特性：
- TMA：异步多维张量搬运
- WGMMA：异步 128/256 线程矩阵乘法
- Cluster：CTA 群组协作
- TensorMap Runtime Update：动态 TMA descriptor
- GDC：硬件级 grid 依赖
- Register Reconfiguration：灵活的寄存器分配
- FP8 Tensor Core：e4m3/e5m2 输入、FP32 累加，配 GmmaFP8Accumulation 控精度

理解这 6 个 kernel 加上 FP8 两条路线的关系和差异，是掌握 CUTLASS SM90 GEMM 的关键。每个变体都在前一个的基础上做了针对性优化，共同构成了一个完整的 GEMM 性能优化体系。
