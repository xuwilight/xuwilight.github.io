---
title: Cutlass 学习笔记（六）Hopper Cluster
date: 2025-08-20 12:00:00
tags: [CUTLASS, Hopper, Cluster, GPU]
categories: [Cutlass 学习笔记,Cutlass]
description: 在 Hopper 架构中，使用 cluster 和普通的代码有什么区别，cluster 不就是一个 threadblock 能访问相同 cluster 中其他 SM 上的 shared memory 吗，这有什么优势。而且通过 ncu 发现 distributed shared memory 这块没有数据传输，是不是说明虽然使用了 cluster，但是确并没有发生不同 sm 间的传输。还有 ...
---

在 Hopper 架构中，使用 cluster 和普通的代码有什么区别，cluster 不就是一个 threadblock 能访问相同 cluster 中其他 SM 上的 shared memory 吗，这有什么优势。而且通过 ncu 发现 distributed shared memory 这块没有数据传输，是不是说明虽然使用了 cluster，但是确并没有发生不同 sm 间的传输。还有 cluster shape 具体是怎么划分 SM 的，<1,2,1>和<2,1,1>的 shape 有什么区别？

这是一个非常深刻的问题，触及到了 Hopper 架构设计的精髓。

你通过 ncu 观察到“Distributed Shared Memory (DSMEM) 没有数据传输”，却依然能跑出高性能，这其实揭示了 **Cluster 在 GEMM 场景下的真正用法**。

**简短结论：** 在 CUTLASS GEMM 中，Cluster 的核心优势**不是**让 SM 互相读写对方的 Shared Memory（这是你看到的 DSMEM traffic 为 0 的原因），而是利用 **TMA Multicast（TMA 多播）** 机制，从 Global Memory 读取一次数据，同时写入多个 SM 的 Shared Memory。

以下是详细的技术原理解析：

### 1. Cluster 到底有啥用？（除了互相访问）

在 Hopper 之前的架构（如 Ampere），如果你有两个 Thread Block (TB) 需要相同的数据（比如矩阵 A 的同一块 Tile），它们必须各自发起一次加载请求：

SM 1: Load A[Tile] -> L2 -> SM 1 Shared Memory

SM 2: Load A[Tile] -> L2 -> SM 2 Shared Memory

这导致了双倍的 L2 带宽消耗和重复的指令开销。

在 Hopper 架构上，引入了 **Thread Block Cluster**，对于 GEMM 来说，它带来了两个巨大的改变：

#### A. TMA Multicast (真正的杀手锏)

这是你观察到 DSMEM 流量为 0 但性能依然提升的原因。

机制： 在 Cluster 模式下，SM 1 发出一个 TMA 指令：“把 Global Memory 地址 X 的数据加载进来，并多播（Multicast） 给我和 SM 2”。

效果： 数据只经过 L2 Cache 一次，然后直接分发到 Cluster 内所有 SM 的 Shared Memory 中。

优势： 节省了 50%（如果是 size 2）甚至 75%（如果是 size 4）的 L2 读取带宽。这在 GEMM 這種带宽敏感型计算中是巨大的提升。

#### B. 调度亲和性 (Scheduling Affinity)

Cluster 保证了这组 Thread Block 会被调度到物理上相邻的 SM（通常在一个 GPC 或 TPC 内）。

这不仅降低了多播的延迟，还确保了它们在时间上几乎是同时执行的，这对于依赖同步的算法至关重要。

### 2. 为什么 ncu 显示 DSMEM 传输为 0？

你的观察非常敏锐，但解释稍有偏差。

DSMEM Traffic 指标通常统计的是 LD.shared::cluster 或 ST.shared::cluster 这种显式的指令。也就是 Thread 主动去读写隔壁 SM 的动作。

CUTLASS GEMM 的设计： 标准的矩阵乘法不需要 SM 之间互传数据（除了 Split-K 最后的归约，但那个很少）。SM 只需要相同输入数据（A 或 B）。

发生了什么： 实际上使用的是 TMA Multicast Load。这被算作是 Global 到 Shared 的流量，而不是 Peer-to-Peer 的 DSMEM 流量。

结论： DSMEM 流量为 0 是完全正常的，甚至是非常好的。 这说明你的 Kernel 没有浪费时间在 SM 间搬运数据，而是利用多播完美解决了数据复用问题。

### 3. Cluster Shape <1, 2, 1> vs <2, 1, 1> 的区别

这直接关系到你复用的是哪一个矩阵（A 还是 B），以及如何切分任务。

假设 CUDA Grid 的维度映射通常是：Grid.x -> M 维度，Grid.y -> N 维度。

#### Case 1: Cluster Shape <1, 2, 1> (X=1, Y=2)

含义： 沿着 Y 轴（N 维度）将 2 个 Thread Block 捆绑在一起。

几何意义： 这两个 Block 处理的是相同的 M 坐标，但是不同的 N 坐标。

Block 1 处理：

C[0:128,0:128]Block 2 处理：

C[0:128,128:256]

数据复用：

它们都需要矩阵 A 的行

A[0:128,k]

。它们需要矩阵 B 的不同列。

硬件行为： 启用 Mcast A（多播矩阵 A）。TMA 加载一次 A 的 Tile，同时写入两个 SM。B 矩阵则是各自加载。

适用场景： 当矩阵 A 很大或者读取 A 是瓶颈时。

#### Case 2: Cluster Shape <2, 1, 1> (X=2, Y=1)

含义： 沿着 X 轴（M 维度）将 2 个 Thread Block 捆绑在一起。

几何意义： 这两个 Block 处理的是不同的 M 坐标，但是相同的 N 坐标。

Block 1 处理：

C[0:128,0:128]Block 2 处理：

C[128:256,0:128]

数据复用：

它们需要矩阵 A 的不同行。它们都需要矩阵 B 的列

B[k,0:128]

。

硬件行为： 启用 Mcast B（多播矩阵 B）。TMA 加载一次 B 的 Tile，同时写入两个 SM。

适用场景： 当矩阵 B 很大或者读取 B 是瓶颈时。

### 4. 你的案例分析 (1x2x1)

回到你之前的编译文件名：...128x256x64_1x2x1...

128x256x64: 这是 Thread Block Tile (M=128, N=256, K=64)。

1x2x1: 这是 Cluster Shape (X=1, Y=2)。

**这意味着：**

CUTLASS 选择了在 **N 维度** 上进行 Cluster。

复用 A： 因为 Cluster 里的 SM 处理同样的 M，它们共享矩阵 A 的数据。矩阵 A 被多播了。

吞吐量： 一个 Cluster (包含 2 个 SM) 一次迭代可以计算出的 C 矩阵大小是：

M = 128 (不变)N = 256 * 2 (因为 Y=2, 两个 Block 并在 N 轴) = 512 等一下，这里的 128x256 应该是指单个 CTA 的 Tile。如果 Cluster 是 1x2x1，那么物理上这个 Cluster 一次处理的逻辑大 Tile 是 128 x (256*2) = 128 x 512。

**总结建议：**

如果你想验证这一点，可以在 ncu 中查看 **Memory Workload Analysis** -> **TMA** 部分，寻找 **"Multicast"** 相关的指令计数或吞吐量，那里应该会有巨大的数值，而不是在 DSMEM 部分。

<1, 2, 1>: 也就是你现在的配置，省了 A 矩阵的带宽。

<2, 1, 1>: 省 B 矩阵的带宽。

通常 GEMM kernel 会根据 M 和 N 的大小自动选择。如果 

*M*≪*N*

，说明 A 矩阵很小，B 矩阵很大，这时候应该用 <2, 1, 1> 来多播 B 矩阵以节省带宽。反之亦然。

一般在启动一个 kernel 的时候，物理 SM 是按照 gridx 的顺序处理 CTA 的，比如在 hopper 上一共有 132 个 SM，那么这 132 个 SM 会沿着 M 方向处理不同的 CTA。如果 cluster 的 shape 是<2,1,1>可以理解为两个相邻的物理 SM 组成一个 cluster。但是如果 shape 是<1,2,1>时，如果 M 的大小是 66，那就是第 0 个第 66 个 SM 在一个 cluster 里吗，那如果 M 的大小是 132，物理 SM 刚好沿着 M 方向处理一列，那 Y 方向上就只有一个 SM，这个时候<1,2,1>应该怎么组成 cluster。还有 hopper 架构上一般都是 persistent kernel，也就是只启动 132 个 thread block，一个 thread block 占用一个 SM，循环处理不同 tile 的数据，那这种情况下 cluster 又是怎么分的？

这是一个非常棒的深入问题。你对“物理 SM 线性映射”和“Persistent Kernel”的直觉有一部分是对的，但 Hopper 的 Cluster 调度机制打破了传统的 Grid Dim.x -> Physical SM 的简单线性映射关系。

让我们分三个层次来解构这个过程：**硬件调度（Hardware Scheduling）**、**Cluster 映射机制**，以及 **Persistent Kernel 中的具体实现**。

### 1. 纠正误区：物理 SM 不是简单地沿着 Grid X 填充的

你原本的理解可能是：

调度器先拿 blockIdx.x = 0...M 填满所有 SM，然后再填 blockIdx.y。

**在 Hopper Cluster 模式下，这是不准确的。**

当开启 Cluster 时，GPU 的**GigaThread Engine (调度器)** 不再以单个 Thread Block (CTA) 为单位分发任务，而是以 **Cluster (CGA, Cooperative Group Array)** 为单位分发任务。

假设 Cluster Shape 是 <1, 2, 1> (即 1x2 个 CTA)：

调度器眼中的“最小原子单位”是一个包含 2 个 CTA 的“大块”。

调度器会寻找物理上相邻且属于同一个 GPC (Graphics Processing Cluster) 的 2 个空闲 SM。

它将这 2 个 CTA 同时 放到这 2 个 SM 上。

物理 SM 0 -> 拿到了 Logical Block (0, 0)物理 SM 1 -> 拿到了 Logical Block (0, 1)(注：这里的坐标取决于 Cluster 是沿着 X 还是 Y 轴组合)

**回答你的疑问：**

如果是 <1, 2, 1> (X=1, Y=2)，这意味这一个 Cluster 包含 Logical Block (m, n) 和 (m, n+1)。

调度器会把这两个 Block 捆绑，扔给物理上的 SM0 和 SM1。

绝对不会出现 Block (0,0) 在 SM0，而 Block (0,1) 跑到了 SM66 这种情况。Cluster 保证了物理局部性（Physical Locality），否则 DSMEM 和 Multicast 无法工作（因为它们依赖物理线路的短距离互联）。

### 2. Persistent Kernel 下的 Cluster 又是怎么分的？

这是 CUTLASS 高性能的核心。你是对的，Hopper 上的 GEMM 通常是 Persistent Kernel，即：

**启动的 Block 数量 = 物理 SM 数量（或略少），然后死循环处理数据。**

假设你有 132 个 SM，Cluster Shape 为 <1, 2, 1>。

此时，CUTLASS **实际上只启动了 66 个 Cluster**（一共 132 个 Block）。

#### 静态的物理绑定 (Static Hardware Mapping)

一旦 Kernel 启动：

Cluster 0 (包含 Block 0, 1) -> 锁定在 SM 0, SM 1。

Cluster 1 (包含 Block 2, 3) -> 锁定在 SM 2, SM 3。

...

Cluster 65 -> 锁定在 SM 130, SM 131。

这 132 个 SM 就位后，它们就像 66 对“搭档”，终生不换位置。

#### 动态的数据映射 (Dynamic Tile Scheduler / Rasterization)

现在问题来了：这些“搭档”怎么去处理那巨大的 Grid 数据（比如 M=4096, N=4096）？

这就是 **Swizzle (光栅化)** 逻辑起作用的地方。在 Persistent Loop 的 while 循环内部，代码不是简单地用 blockIdx 作为数据的 Tile Index，而是用一个全局原子的计数器（或者预计算的顺序）来领取任务。

**关键点：** 领取任务的单位也是 **Cluster Tile**。

当 SM 0 和 SM 1 (Cluster 0) 完成了当前工作，准备领下一个任务时：

它们（通常是 Cluster 中的 Leader Block）会去全局计数器“领号”。

领到的号代表一个 Cluster Tile 坐标，比如 (Tile_M=5, Tile_N=10)。

因为 Cluster Shape 是 <1, 2, 1> (X=1, Y=2)：

SM 0 会计算出它应该处理的数据 Tile 是：M=5, N = 10 * 2 + 0 = 20。SM 1 会计算出它应该处理的数据 Tile 是：M=5, N = 10 * 2 + 1 = 21。

**结果：**

无论这一对 SM 领到了矩阵的哪个角落的数据，**SM 0 和 SM 1 永远处理的是同一个 M，相邻的 N**。

这就满足了 <1, 2, 1> 的物理定义：

它们需要相同的矩阵 A 的行（M 相同）。

SM 0 发起对 A 的 TMA 请求，并设置 Multicast Mask 指向 SM 1。

SM 1 坐享其成，直接从 L2 收到数据。

### 3. 回答你的具体场景疑问

问题： 如果 shape 是 <1,2,1>，M 的大小是 132，物理 SM 刚好沿着 M 方向处理一列，那 Y 方向上就只有一个 SM，这个时候 <1,2,1> 应该怎么组成 cluster？

**解答：**

在 Persistent Kernel 模式下，**Grid 的形状（M, N, K）和物理 SM 的排列没有这种僵硬的对应关系。**

物理层： 132 个 SM 被分成了 66 对。每对就是一个 <1, 2, 1> 的硬件执行单元。

逻辑层： 你的矩阵 M=132, N=...。

执行流：

Cluster 0 (SM 0, SM 1) 领到了任务：处理 Row M=0 的部分数据。

SM 0 处理 (M=0, N=0)SM 1 处理 (M=0, N=1)它们共享 A[Row 0]。

Cluster 1 (SM 2, SM 3) 领到了任务：处理 Row M=1 的部分数据（或者 M=0 的后续 N，取决于 Swizzle 策略）。

**不存在“Y 方向只有一个 SM 导致无法组成 Cluster”的情况。**

因为 Cluster 是在 Kernel **启动瞬间** 就由硬件定死的（SM 0 和 SM 1 必须绑定）。如果你的 Grid N 维度太小（比如 N=1），导致无法凑够 2 个 Block 来填满 Cluster，那么 CUTLASS 通常会：

报错或回退到普通 Kernel。

或者启动了 Cluster，但其中一个 SM 空跑（Mask off），这会浪费一半算力。

**总结：**

Cluster 绑定早于数据处理： 物理 SM 是两两（或四四）绑定成 Cluster 的。

数据是喂给 Cluster 的： 软件逻辑（Swizzle）保证了每次喂给这对 SM 的数据块，刚好符合它们“共享 A”或者“共享 B”的特性。

Persistent 模式下： 132 个 SM 就是 66 个“双核处理器”。它们去扫描整个矩阵，每次吃掉一块 <1, 2> 大小的数据。

是不是说在<1,2,1>的情况下，两个相邻的 SM 会组成一个 cluster，cluster0 处理(0,0)和(0,1)的 thread block，cluster1 处理(1,0)和(1,1)的 thread block，以此类推。那么如果 M 有 6 个 tile，是不是 cluster5 处理(5,0)和(5,1)，然后 cluster6 就处理(0,2)和(0,3)了。如果 M 方向有 6 个 tile，那么 N 方向可以有 11 个 cluster，也就是能处理 22 个 tile，但是如果 N 只有 20 个 tile，这样是不是多出来的一列 cluster 就没数据处理了，或者 N 有 21 个 tile，这样最后一列的 cluster 就只有一半的数据能处理吗？还有一种情况就是在<2,1,1>的 cluster shape 下，是不是 cluster0 处理(0,0)和(1,0)，那么如果 M 方向不是 2 的倍数，这种情况下 cluster 会怎么处理？

你的理解非常到位，尤其是对于 **Cluster 内部如何瓜分数据** 的理解是完全正确的。

对于你提出的关于 **边界情况（Padding/Tail Effect）** 和 **处理顺序（Rasterization）** 的疑问，这里有非常明确的硬件和软件层面的解释。

### 1. 核心确认：Cluster 如何“吃”数据

是的，你的模型是正确的：

Cluster Shape <1, 2, 1>: 这里的最小调度单位（我们称为 Cluster Tile）在逻辑 Grid 上就是 1 行 x 2 列 的大小。

如果不考虑复杂的 Swizzle（光栅化）顺序，简单来说：一个 Cluster 领取了一个任务坐标 (m, n)。SM A (Cluster 内第 0 个) 处理 (m, n)。SM B (Cluster 内第 1 个) 处理 (m, n+1)。

### 2. 只有奇数个 Tile 怎么办？（Padding 与 Predication）

这是你最担心的问题：**如果 N 方向有 21 个 Tile，而 Cluster 是 2 个一组，最后一个怎么办？**

在 CUTLASS 和 Hopper 架构中，解决这个问题的通用策略是 **向上取整（Round Up）** 配合 **谓词保护（Predication / Guard）**。

#### 场景：Shape <1, 2, 1>, N_Tiles = 21

逻辑上，Grid 会被补齐到 **N = 22**。

当某个 Cluster（比如最后一个 Cluster）领到了处理 **Column 20 和 21** 的任务时：

SM A 拿到 Tile_N = 20：

检查边界：20 < 21？是。行为：正常加载数据，正常计算，正常写入。

SM B 拿到 Tile_N = 21：

检查边界：21 < 21？否（越界了）。行为：

TMA 加载阶段： 如果使用了 TMA 的边界处理功能，它可能加载进来的是全 0（Padding），或者 TMA 指令被跳过。计算阶段： 虽然 SM B 属于这个 Cluster，但因为它的主循环条件不满足（或者由 pred 变量控制），它实际上会空转或者进行无效计算（计算全 0）。Epilogue 写入阶段： 会有 if (global_n < problem_N) 的判断，防止写出边界导致非法内存访问。

**结论：** 最后一个 Cluster 会正常启动，但其中一个 SM 干活，另一个 SM "摸鱼"（Idle 或执行无效指令）。这确实会有一点点算力浪费，但对于大矩阵（成百上千个 Tile）来说，这一个 SM 的浪费可以忽略不计。

### 3. 处理顺序：是先走完 M 还是先走完 N？

你提到的例子：

“如果 M 有 6 个 tile... Cluster5 处理 (5,0) 和 (5,1)，然后 Cluster6 就处理 (0,2) 和 (0,3) 吗？”

这取决于 **Swizzle（光栅化）策略**。

虽然你描述的这种“先走完 M 列，再换 N 列”（Column-Major）是可能的，但在 GEMM 优化中，通常不这么走。

**为什么？为了 L2 Cache 命中率。**

如果 Cluster Shape 是 <1, 2, 1>： 这意味着我们通过 Multicast 复用了 矩阵 A。

SM 0 和 SM 1 读同一行 A。

接下来的 Cluster 6 应该去哪？

如果去 (0, 2)：它需要加载 A[Row 0]。嘿！这行 A 刚才 Cluster 0 才用过，可能还在 L2 Cache 里！这是好事。如果去 (5, 0) 的下一个位置（比如 M 更大的位置）：需要加载新的 A。

**实际的 CUTLASS Swizzle 策略：**

通常采用 **Hilbert 曲线** 或 **分块光栅化**。它不会简单地把一整列 M 走完才换行。它更像是在画一个个“方块”。

例如，它可能会先处理一个 4x4 的 Cluster 区域，把这个区域里的 A 和 B 都充分复用完，再去下一个区域。

所以，Cluster 6 领到的任务可能既不是 (0, 2) 也不是 (5, 2)，而是取决于 Swizzle 算法计算出的下一个“空间局部性最好”的坐标。但无论去哪，**它领取的依然是一个 (1x2) 的任务块。**

### 4. 关于 <2, 1, 1> 和 M 不是 2 的倍数

逻辑与 N 方向完全一致：

Cluster Shape: <2, 1, 1> (2 行 x 1 列)

任务分配:

SM A 处理 (m, n)SM B 处理 (m+1, n)它们共享 矩阵 B 的数据（Multicast B）。

边界情况： 如果 M 方向有 5 个 Tile。

逻辑 Grid 补齐到 6。处理最后一块的 Cluster 负责 Row 4 和 Row 5。负责 Row 4 的 SM 正常工作。负责 Row 5 的 SM 发现越界，进行边界保护（不写入内存，不加载非法地址）。

### 5. 总结

绑定关系： Cluster 内部的 SM 绑定关系是死的（物理相邻）。

任务单位： 调度器派发的任务单位永远是 Cluster Shape 的大小。

边界处理： 依靠 Grid Padding（逻辑补齐） 和 Kernel 内部的边界检查（Predication）。不够整数倍时，多余出来的那个 SM 会被启动，但不会产生有效副作用（不会写坏内存），只是陪跑。

执行顺序： 并不一定是线性的“先列后行”。为了利用 L2 Cache，通常是走“之”字形或块状路径，但这对你理解 Cluster 的映射机制不影响——你只需要知道每次必定消耗掉 Shape 大小的 Block 即可。
