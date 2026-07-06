---
title: multimem 指令
date: 2026-06-11 20:00:00
tags: [CUDA, multimem, NVSwitch, GPU]
categories: [PTX 学习笔记]
description: 文章介绍了 multimem 指令的一些概念和使用方法，包括 multimem.ld_reduce，multimem.st 和 multimem.red 等。最后分析了使用 multimem 实现的 allreduce 和传统 ring allreduce 通信量上的区别。
---


# Multimem 指令

在 NVIDIA 较新的 GPU 架构（如基于 Hopper 架构的 `sm_90` 及以上）中，PTX (Parallel Thread Execution) 指令集架构引入了一组针对数据移动与转换的全新高级指令——**Multimem 指令**。

这组指令专门用于处理“多重内存地址”（Multimem Addresses），在分布式张量计算和高效规约操作中具有重要的底层支撑作用。

## Multimem 地址概念与基础语法

在常规的内存模型中，一个内存地址映射到一个单一的物理存储单元。而 Multimem 地址是一个特殊的内存抽象，它同时指向多个物理内存位置。

`multimem.*` 系列指令专门用于操作这些地址，以此实现高效的数据广播和多地址规约。

Multimem 地址只能由 `multimem.*` 操作访问。使用 `ld`、`st` 或任何其他内存操作访问 multimem 地址会导致未定义行为。

可以使用 NVSHMEM 创建 multimem 地址。

在 PTX 语法层面，Multimem 操作支持整型（Integer）和浮点型（Floating point）数据，并提供了丰富的修饰符（Modifiers）组合。其基础语法结构如下：

*   **加载规约：** `multimem.ld_reduce{.ldsem}{.scope}{.ss}.op.type d, [a];`
*   **多重存储：** `multimem.st{.stsem}{.scope}{.ss}.type [a], b;`
*   **内存规约：** `multimem.red{.redsem}{.scope}{.ss}.op.type [a], b;`

这些语法中包含了内存同步语义 (`.ldsem`/`.stsem`/`.redsem`)、可见性作用域 (`.scope`)、规约操作符 (`.op`) 以及数据类型 (`.type` 和 `.vec`)，以便开发者对硬件行为进行细粒度控制。

```ptx
// 整数类型：

multimem.ld_reduce{.ldsem}{.scope}{.ss}.op.type    d, [a];
multimem.ld_reduce.weak{.ss}.op.type               d, [a];

multimem.st{.stsem}{.scope}{.ss}.type              [a], b;
multimem.st.weak{.ss}.type                         [a], b;

multimem.red{.redsem}{.scope}{.ss}.op.type         [a], b;

.ss =       { .global }
.ldsem =    { .relaxed, .acquire }
.stsem =    { .relaxed, .release }
.redsem =   { .relaxed, .release }
.scope =    { .cta, .cluster, .gpu, .sys }
.op  =      { .min, .max, .add, .and, .or, .xor }
.type =     { .b32, .b64, .u32, .u64, .s32, .s64 }


// 浮点类型：

multimem.ld_reduce{.ldsem}{.scope}{.ss}.op{.acc_prec}{.vec}.type    d, [a];
multimem.ld_reduce.weak{.ss}.op{.acc_prec}{.vec}.type               d, [a];

multimem.st{.stsem}{.scope}{.ss}{.vec}.type                         [a], b;
multimem.st.weak{.ss}{.vec}.type                                    [a], b;

multimem.red{.redsem}{.scope}{.ss}.redop{.vec}.redtype              [a], b;

.ss =       { .global }
.ldsem =    { .relaxed, .acquire }
.stsem =    { .relaxed, .release }
.redsem =   { .relaxed, .release }
.scope =    { .cta, .cluster, .gpu, .sys }
.op  =      { .min, .max, .add }
.redop  =   { .add }
.acc_prec = { .acc::f32, .acc::f16 }
.vec =      { .v2, .v4, .v8 }
.type=      { .f16, .f16x2, .bf16, .bf16x2, .f32, .f64, .e5m2, .e5m2x2, .e5m2x4, .e4m3, .e4m3x2, .e4m3x4 }
.redtype =  { .f16, .f16x2, .bf16, .bf16x2, .f32, .f64 }
```


## 使用方法

文档详细定义了三种核心指令的执行逻辑：

*   **`multimem.ld_reduce` (多重加载并规约)**
    该指令首先从 Multimem 地址 `a` 所指向的所有多个内存位置中加载数据，随后对这些加载上来的多份数据执行由 `.op` 规定的规约操作（如求和、求最值等），最后将唯一的规约结果返回并写入目标寄存器 `d`。
*   **`multimem.st` (多重存储/广播)**
    该指令将输入操作数（寄存器 `b` 中的值）执行存储操作，同时写入到 Multimem 地址 `a` 所指向的所有内存位置。这实质上是一种硬件级别的多播（Multicast）写入。
*   **`multimem.red` (多重内存规约)**
    该指令将输入操作数 `b` 的值，与 Multimem 地址 `a` 指向的所有内存位置中的现有值进行规约计算，并将结果直接更新在这些内存位置中。

`multimem.ld_reduce` 是对“从多处读取上来的数据”进行规约（收集到寄存器）；而 `multimem.red` 是用一个给定值去规约“多处内存中现有的数据”（直接在内存端修改）。

在使用这些指令时，地址操作数 `a` 必须是 Multimem 地址。如果未明确指定状态空间（State Space），硬件将默认使用通用寻址（Generic Addressing）。若指定了状态空间，且地址不属于 `.global` （全局内存）窗口，则会导致未定义行为。

对于浮点类型的多重操作，指令集有着严格的位宽要求：所指定的数据类型大小乘以向量维度（`.vec`），其总位宽必须严格等于 32 位、64 位或 128 位。任何不符合此位宽对齐的组合都是被禁止的。例如，`.f64` 类型无法与任何 `.vec` 向量修饰符组合使用。

同时，文档对操作符 (`.op`) 与基础类型的组合也做了明确规定：例如加法 (`.add`) 和最值 (`.min`, `.max`) 支持广泛的整型和各类浮点型（包括 `.f16`, `.bf16` 及其向量，甚至新型的 8 位浮点格式），而逻辑运算 (`.and`, `.or`, `.xor`) 仅支持 32 位或 64 位无符号位宽类型 (`.b32`, `.b64`)。


| .vec | 支持的基础浮点类型 |
| :--- | :--- |
| 未指定 `.vec` | `.f16x2` , `.bf16x2` , `.f32` , `.f64` , `.e5m2x4` , `.e4m3x4` |
| `.v2` | `.f16` , `.f16x2` , `.bf16` , `.bf16x2` , `.f32` , `.e5m2x2` , `.e5m2x4` , `.e4m3x2` , `.e4m3x4` |
| `.v4` | `.f16` , `.f16x2` , `.bf16` , `.bf16x2` , `.f32` , `.e5m2` , `.e5m2x2` , `.e5m2x4` , `.e4m3` , `.e4m3x2` , `.e4m3x4` |
| `.v8` | `.f16` , `.bf16` , `.e5m2` , `.e4m3` , `.e5m2x2` , `.e4m3x2` |

下表描述了 `.op` 和基础类型的有效组合：

| op | 基础类型 |
| :--- | :--- |
| `.add` | `.u32` , `.u64` , `.s32` , `.f16` , `.f16x2` , `.bf16` , `.bf16x2` , `.f32` , `.f64` , `.e5m2` , `.e5m2x2` , `.e5m2x4` , `.e4m3` , `.e4m3x2` , `.e4m3x4` |
| `.and` , `.or` , `.xor` | `.b32` , `.b64` |
| `.min` , `.max` | `.u32` , `.s32` , `.u64` , `.s64` , `.f16` , `.f16x2` , `.bf16` , `.bf16x2` , `.e5m2` , `.e5m2x2` , `.e5m2x4` , `.e4m3` , `.e4m3x2` , `.e4m3x4` |

对于 `multimem.ld_reduce` ，中间累加的默认精度与指定的类型相同。

可选地，可以指定 `.acc_prec` 限定符来改变中间累加的精度，如下所示：

| .type | .acc::prec | 将精度更改为 |
| :--- | :--- | :--- |
| `.f16` , `.f16x2` , `.bf16` , `.bf16x2` | `.acc::f32` | `.f32` |
| `.e5m2` , `.e4m3` , `.e5m2x2` , `.e4m3x2` , `.e4m3x4` , `.e5m2x4` | `.acc::f16` | `.f16` |

可选限定符 `.ldsem`、`.stsem` 和 `.redsem` 分别指定 `multimem.ld_reduce`、`multimem.st` 和 `multimem.red` 的内存同步效果，如“内存一致性模型 (Memory Consistency Model)”中所述。如果未指定显式的语义限定符，则 `multimem.ld_reduce` 和 `multimem.st` 默认为 `.weak`，而 `multimem.red` 默认为 `.relaxed`。

可选的 `.scope` 限定符指定可以直接观察到此操作的内存同步效果的线程集合，如“内存一致性模型”中所述。如果未为 `multimem.red` 指定 `.scope` 限定符，则默认假定为 `.sys` 作用域。



## 指令应用示例

```ptx
multimem.ld_reduce.and.b32                   val1_b32, [addr1];
multimem.ld_reduce.acquire.gpu.global.add.u32 val2_u32, [addr2];

multimem.st.relaxed.gpu.b32                  [addr3], val3_b32;
multimem.st.release.cta.global.u32           [addr4], val4_u32;

multimem.red.relaxed.gpu.max.f64             [addr5], val5_f64;
multimem.red.release.cta.global.add.v4.f32   [addr6], {val6, val7, val8, val9};
multimem.ld_reduce.add.acc::f32.v2.f16x2     {val_10, val_11}, [addr7];

multimem.ld_reduce.relaxed.cta.min.v2.e4m3x2 {val_12, val_13}, [addr8];
multimem.ld_reduce.relaxed.cta.add.v4.e4m3   {val_14, val_15, val_16, val_17}, [addr9];
multimem.ld_reduce.add.acc::f16.v4.e5m2      {val_18, val_19, val_20, val_21}, [addr10];
```


* 逻辑规约与带同步的整型规约：
    `multimem.ld_reduce.and.b32 val1_b32, [addr1];`
    对 `addr1` 指向的多个 32 位值执行逻辑“与”运算，结果存入 `val1_b32`。
    `multimem.ld_reduce.acquire.gpu.global.add.u32 val2_u32, [addr2];`
    带有 acquire 语义并在 GPU 级别可见的无符号 32 位加法规约。

* 多重存储（广播）：
    `multimem.st.relaxed.gpu.b32 [addr3], val3_b32;`
    将 `val3_b32` 以宽松语义广播到 `addr3` 指向的各个内存位置。

* 浮点数的高级规约与精度控制：
    `multimem.ld_reduce.add.acc::f32.v2.f16x2 {val_10, val_11}, [addr7];`
    该指令读取 FP16 向量数据，执行加法规约，并在底层使用 32 位浮点精度（`.acc::f32`）进行中间累加，最终结果写入寄存器对中。
    `multimem.ld_reduce.add.acc::f16.v4.e5m2 {val_18, ...}, [addr10];`
    演示了针对 8-bit 浮点数（`e5m2`），使用 FP16 作为累加精度进行 4 元素向量加载规约的最新用法。


## 通信量估算

要估算和对比使用 **`multimem` 指令实现的 All-Reduce** 与 **传统的 Ring All-Reduce** 在通信量上的区别，我们需要从分布式通信的理论模型入手。

假设前提：

*   参与规约的 GPU（节点）总数为 **$N$**。
*   需要进行 All-Reduce 的数组总大小为 **$S$**（字节）。
*   通信量的估算我们以**单一节点（单张 GPU）需要发送和接收的数据量**为衡量标准。

以下是详细的估算与对比分析：

### 传统的 Ring All-Reduce 通信量估算

Ring All-Reduce 是一种纯软件算法，它通过将 GPU 组织成一个逻辑环，将整个规约过程分为两个阶段：Reduce-Scatter 和 All-Gather。在操作前，数组 $S$ 会被切分为 $N$ 个数据块，每个块大小为 $S/N$。

*   **阶段一：Reduce-Scatter（规约并散播）**
    *   每个 GPU 将自己的一个数据块（大小 $S/N$）发送给下一个节点，同时接收前一个节点的数据块并与其本地数据相加。
    *   这个过程需要进行 $N-1$ 步。
    *   **单节点发送量：** $(N-1) \times \frac{S}{N} = \frac{N-1}{N}S$
*   **阶段二：All-Gather（全收集）**
    *   此时每个 GPU 手上都有一块（大小 $S/N$）已经是最终规约结果的数据。接下来它们再次在环上传递这些数据，直到所有 GPU 都拥有完整的 $S$。
    *   同样需要 $N-1$ 步。
    *   **单节点发送量：** $(N-1) \times \frac{S}{N} = \frac{N-1}{N}S$

**Ring All-Reduce 总结：**
*   **单节点总发送量：** $2 \times \frac{N-1}{N}S$
*   **单节点总接收量：** $2 \times \frac{N-1}{N}S$
*   **渐近表现：** 当节点数 $N$ 较大时，$\frac{N-1}{N} \approx 1$。因此，单节点无论收发，通信量都约等于 **$2S$**。
*   **延迟（步骤数）：** $2(N-1)$ 步，随节点数线性增长。

---

### 使用 Multimem 指令的 All-Reduce 通信量估算

`multimem` 系列指令（如 `multimem.ld_reduce` 和 `multimem.st`）本质上是暴露给软件的底层硬件能力。它依赖于全连接的底层硬件拓扑（如 NVLink 交换机 NVSwitch），交换机内部具备硬件多播（Multicast）和网内计算（In-Network Reduction，类似于 SHARP 技术）的能力。

使用 `multimem` 实现 All-Reduce 通常采用“数据并行分块 + 硬件直达”的策略：

假设我们将总数据 $S$ 依然分为 $N$ 块，分配给 $N$ 个 GPU 作为“主节点”进行处理：

*   阶段 1：`multimem.ld_reduce`
    *   发送 (TX)：为了配合其他所有 GPU 的拉取请求，一个 GPU 需要把自己的每一块数据都发给交换机，总共发送 $N \times (S/N) = S$。
    *   接收 (RX)：GPU 作为发起者，从交换机收回属于自己负责的那块 $S/N$ 的规约结果。
*   阶段 2：`multimem.st`
    *   发送 (TX)：GPU 将自己手上的最终结果 $S/N$ 发给交换机准备广播。
    *   接收 (RX)：交换机广播其他 $N-1$ 个 GPU 的结果，该 GPU 接收 $(N-1) \times (S/N)$。

Multimem All-Reduce 总结：
*   总发送量 (TX) = $S + S/N \approx \mathbf{1S}$
*   总接收量 (RX) = $S/N + (N-1)S/N = \mathbf{1S}$
*   延迟（步骤数）：**从硬件角度看，指令只需经过 $O(1)$ 或 $O(\log N)$ 层交换机即可完成，不需要接力。


| 对比维度 | 传统 Ring All-Reduce | Multimem 指令加速 All-Reduce | 提升幅度 |
| :--- | :--- | :--- | :--- |
| **单节点通信量 (发/收)** | $\approx \mathbf{2S}$ (精确值为 $2 \frac{N-1}{N} S$) | $\approx \mathbf{1S}$ | 通信量减少 50% (或者说有效带宽利用率翻倍) |
| **算法延迟 (时间步数)** | $\mathbf{O(N)}$ 步，需要 $2(N-1)$ 次接力 | $\mathbf{O(1)}$ 步，依赖 NVSwitch 硬件直达 | 指数级降低，极大优化小数据包的延迟 |
| **计算位置** | 占用 GPU SM (CUDA Core) 算力 | In-Network Computing 或专用存储单元 | 释放了 GPU 算力用于矩阵计算本身 |
| **显存读写次数** | 需不断读写显存进行中间态合并 | 零拷贝/少拷贝，直接操作硬件指针 | 大幅降低 HBM 显存带宽压力 |

从理论估算来看，使用 `multimem` 相关的指令和硬件机制，可以将 All-Reduce 这种集体通信的数据搬运量直接砍半（从接近 $2S$ 降到 $1S$）。

在现代大模型（如 GPT-4 / Llama 3）的训练中，通信带宽往往是最大的瓶颈。`multimem` 指令配合 NVLink 的硬件规约网络，相当于在同样的物理连线速度下，把 All-Reduce 的有效吞吐量直接提升了一倍，同时也因为跳过了漫长的环状接力，极大降低了通信延迟。

## 总结

Multimem 指令集为底层开发者和编译器提供了一种绕过传统显式循环和通信屏障，直接利用硬件机制实现多节点内存读写和规约的强大工具。在编写涉及最新 NVIDIA GPU 架构的高性能计算（HPC）或分布式深度学习通信库时，理解并规范使用这些指令将极大提升内存带宽利用率和计算效率。
