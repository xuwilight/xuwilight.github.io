---
title: CuTe 学习笔记（十一）GEMM SM80
date: 2025-04-20 18:00:00
tags: [CUTLASS, GEMM]
categories: [Cutlass 学习笔记]
description: sgemm_sm80 用 cp.async 做异步拷贝、用 bP = 3 的多级 shared memory 环形流水重叠搬运与计算；half 版本还引入 mma.sync、ldmatrix 和 swizzle，第一次用上 Tensor Core。
---

代码地址：https://github.com/NVIDIA/cutlass/blob/v4.1.0/examples/cute/tutorial/sgemm_sm80.cu

`sgemm_sm80` 有两个核心改进：

1. 拷贝换成 `cp.async`（LDGSTS 指令），直接 gmem→smem，异步拷贝不占寄存器、也不阻塞线程。
2. shared memory 从单缓冲变成 `bP = 3` 的三级流水，读下一块、存当前块、算上一块三者可以真正重叠。

half 版本还额外用上 `mma.sync`、`ldmatrix` 和 swizzle——这是这个系列第一次用 Tensor Core 做矩阵乘。float 版本则还是普通的 FMA。

和上一版 `sgemm_02` 相比，该版本用的是 `gmem → rmem → smem` 的寄存器中转、逐元素赋值和单缓冲覆盖；`sgemm_sm80` 把寄存器中转换成了硬件异步拷贝加多级 smem 缓冲，流水更深、开销更低。

`sgemm_sm80.cu` 里其实有多个入口：一个通用的 `gemm_tn` / `gemm_nt` 模板（用 `UniversalFMA`），以及一个专门为 half + TN 手写的 `gemm_tn` 重载（用 `SM80_16x8x16` 的 mma）。half + TN 时会优先命中这个更特化的重载。下面先分析 float 的 `gemm_tn`（它和 `sgemm_02` 的骨架最能对上），再单独看 half 版本多了什么。

矩阵规模还是 M = N = 5120、K = 4096，A、B 都是 K-major（TN GEMM），C 列主序。

## 与 sgemm_02 的区别

| | sgemm_02 | sgemm_sm80 |
|---|---|---|
| gmem→smem | 先到 rmem 再到 smem | `cp.async` 直接 gmem→smem，不经过寄存器 |
| smem 缓冲 | 单 stage，每轮覆盖 | `bP` 个 stage 组成环形流水 |
| 同步方式 | `__syncthreads` 硬隔断 | `cp.async.commit/wait` + 少量 `__syncthreads` |
| 计算 | `UniversalFMA` | float 仍 FMA；half 换成 `mma.sync` + `ldmatrix` |
| smem 布局 | padded | padded（float）；swizzle（half） |

核心变化就是把"用寄存器中转"换成了"用硬件异步拷贝 + 多级 smem 缓冲"。

## 定义规模、布局与流水

strides、CTA tile 和前面一样，变化在 `bP` 和 shared memory 布局：

```cpp
  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA_atom                  = make_layout(make_shape (      bM,          bK),
                                              make_stride(Int<1>{}, bM+Int<1>{})); // padded m-major
  [[maybe_unused]] auto sB_atom = make_layout(make_shape (      bN,          bK),
                                              make_stride(Int<1>{}, bN+Int<1>{})); // padded n-major
  auto sA = tile_to_shape(sA_atom, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(sA_atom, make_shape(bN, bK, bP));
  auto sC = make_layout(make_shape(bM, bN));
```

`bK = 8`，一个 block 每次处理 `[128,8] × [8,128] = [128,128]` 的输出。`bP = 3` 表示 shared memory 有 3 个 stage。

`bM + 1` 的 padding 和 `sgemm_02` 一样，是为了打散 shared memory 的 bank：如果 k 方向 stride 直接用 `bM = 128`，128 是 32 的整数倍，沿 k 方向取元素时地址都落在同一个 bank，产生冲突；改成 `129` 后，129 mod 32 = 1，每换一个 k 就在 32 个 bank 上轮转，冲突基本消除。

注意 `sB_atom` 被标了 `[[maybe_unused]]`，实际 `sB` 复用的是 `sA_atom`——因为 A、B 都是 K-major，tile 形状也相同（都是 128×8），布局完全一样，没必要再定义一个。

`tile_to_shape` 把每个 stage 的 `(128,8)` atom 铺成 `(128,8,3)`，多了第三个 pipe 维：

```
sA = ((_128,_1),(_8,_1),(_1,_3)):((_1,_0),(_129,_0),(_0,_1032))
```

pipe 维的 stride 是 1032，约等于单个 stage 的大小（1032 个元素），所以三个 stage 在 shared memory 里首尾相接。

`tile_to_shape` 和 `local_tile` 都是基于分块的，但方向相反：`local_tile` 是把大 tensor 按 tiler 切成小块并选出一块；`tile_to_shape` 则是把一个小 atom 反复复制、铺满到一个目标 shape。这里是让 3 个 stage 相邻摆放。

另外要强调一点：`sC` 只是一个 layout，用来描述输出 `gC` 的分块方式，并不会占用 shared memory。真正申请 smem 的只有 A、B 两块（kernel 里的 `SharedStorage` 只包含 `A`、`B` 两个数组）。

## cp.async 拷贝与 TiledMMA

```cpp
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TA>, TA>{},
                                    Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape< _1,_1>>{});              // Val layout  1x1
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TB>, TB>{},
                                    Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape< _1,_1>>{});              // Val layout  1x1

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 TiledMMA
```

`SM80_CP_ASYNC_CACHEALWAYS` 对应 `cp.async` 指令，`CACHEALWAYS` 是 cache 策略（L1、L2 都缓存），一次拷贝 4/8/16 字节。float 版本这里用的是"逐元素"模式——每个 atom 拷 1 个 float（4 字节），value layout 是 `1×1`；thread layout `32×8` 是 k-major，共 256 个线程。

> 对照一下：float 的 `gemm_nt` 版本用的是 `uint128_t` 的 atom + `4×1` 的 value layout，即每个线程一次拷 16 字节（4 个 float），是向量化拷贝。这里之所以逐元素，是因为 TN 下 A 在 gmem 上是 K-major、而 smem 是 m-major，逐元素搬运最直接。

`mmaC` 用 `UniversalFMA`（`a*b+c`）铺成 16×16 的线程布局，256 个线程各做一个 FMA。具体原理见前面的 TiledMMA / TiledCopy 两篇。

## kernel 内的 gmem→smem 分块与预取

创建 tensor、`local_tile` 分块和前面一样：

```cpp
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)
```

`gA` 的形状是 `(128, 8, k)`，其中 `k = 4096/8 = 512`。

shared memory 通过 `extern __shared__` 动态申请，再用 `SharedStorage` 结构体把 A、B 两块按 layout 的 cosize 放进同一段动态 smem：

```cpp
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (BLK_N,BLK_K,PIPE)
```

用动态 smem 是因为三级流水的 A、B 加起来会超过静态 smem 的上限；`SharedStorage` 里 `ArrayEngine<T, cosize_v<Layout>>` 的字段顺序决定了 A、B 在 smem 里的偏移。

拷贝分区用 `TiledCopy`：

```cpp
  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,PIPE)
  ...  // B 同理
```

`copy_a` 有 256 个线程、每线程 1 个元素。`gA` 是 `(128,8,512)`，单个 K 块 128×8 = 1024 个元素，256 个线程正好分到 4 个。由于 thread layout 是 k-major `(32,8):(8,1)`、M 方向切 4 份，所以线程 0 负责的 4 个元素在 m 方向上分别是 0、32、64、96（k 固定为 0）。

接下来是流水的起点——先把前 `K_PIPE_MAX-1 = 2` 个 stage 的数据异步预取进 smem：

```cpp
  auto K_PIPE_MAX = size<3>(tAsA);
  int k_tile_count = size<3>(tAgA);
  int k_tile_next = 0;

  // Start async loads for all pipes but the last
  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0) { ++k_tile_next; }
  }
```

`copy(copy_a, ...)` 发射 `cp.async` 指令（异步，不等结果），`cp_async_fence()` 即 `cp.async.commit_group`，把到目前为止发射的异步拷贝打包成一个 group。预取两个 stage 后，`k_tile_count` 已经减了 2，`k_tile_next` 指向第 3 块——它会在主循环里被送进第 3 个 stage。

## MMA 分区与累加器

```cpp
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));               // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));               // (MMA,MMA_N,MMA_K)
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
  clear(tCrC);
```

`partition_fragment_A(sA(_,_,0))` 做的事情是"按 MMA 的线程布局切出这个线程该算的那份，并把它做成寄存器 tensor"。传入 `sA(_,_,0)`（只取第 0 个 stage）是因为形状信息就够用，它只是借用 layout 来确定每线程的片段形状，并不真的去读数据。

float 版本的划分关系：`mmaC` 线程布局 16×16、`sA` 是 128×8。16 个 M 方向线程就能覆盖 128 行（每线程 8 行），K 方向每线程吃满 8 列，所以每个线程逻辑上有 8×8 的 A 片段；N 方向的 16 组线程读的是同一份 A。

- `sA`：(128,8)，每线程 8×8。
- `sB`：(128,8)，每线程 8×8，M 方向的 16 组读同一份 B。
- `sC`(128,128)：按 16×16 切成 8×8 块，每线程处理其中一块 8×8 的 C。

![](/assets/gemm_sm80/image.png)

`clear(tCrC)` 把累加器清零。

## smem→rmem 的 Copy Atom retiling

有了 MMA 的分区，还差一步，就是把 `sA` 里的数据高效地搬进 `tCrA` 寄存器。统一做法是通过一个 smem→reg 的 copy atom 重新分块：

```cpp
  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K,PIPE)
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                         // (CPY,MMA_M,MMA_K)
  ...  // B 同理
```

- `make_tiled_copy_A(s2r_atom_a, mma)`：用 copy atom 和 MMA 一起，生成一个"和 MMA 在 A 上分区方式对齐"的 TiledCopy。
- `partition_S(sA)`：按这个 TiledCopy 的线程布局切 `sA`，得到本线程要读的源。
- `retile_D(tCrA)`：把已经分配好的寄存器 tensor `tCrA` 按 TiledCopy 的目标布局重新组织。


smem 用 `partition`、rmem 用 `retile`，是因为两者的数据归属不同。`partition_S` 是"从一块大的共享内存里切出我这个线程的一小片"。而 `tCrA` 是每个线程私有的寄存器，早就按 MMA 分区分配好了，没法也不能再"切"；`retile_D` 只是把这个已分配的寄存器 tensor 的 value 维度重新贴个标签，让它和 copy atom 期望的目标布局一致，寄存器数量和内容都不变。

`retile` 之后 `tXrA` 的 layout 会变，也是这个原因。以 half 为例，`tCrA` 是 `((_2,_2,_2),_4,_4):((_1,_2,_4),_32,_8)`，`tXrA` 是 `((_8,_1),_4,_4):((_1,_0),_32,_8)`。二者元素个数完全相同（都是每线程 8×4×4 个），差别只在第一维：`tCrA` 里那 8 个元素是按 `(_2,_2,_2):(_1,_2,_4)` 交错分布的（MMA atom 的排布），而 `ldmatrix` 输出的是 8 个连续的寄存器，所以 `tXrA` 把它重排成 `(_8,_1):(_1,_0)`——`(_1,_0)` 表示"第 2 维不占寄存器、stride 为 0"，8 个值紧凑连续。线程持有的元素集合没变，变的只是这 8 个元素在 value 维上的组织方式，这样一次 `ldmatrix.x4` 的输出才能直接对上一个 `tXrA` 的各分量。

对 float，`s2r_atom` 传的是 `Copy_Atom<AutoVectorizingCopy, TA>{}`，原理相同，只是没有 `ldmatrix`、只是向量化搬运。

## 主循环与三级 smem 流水

进入循环前还有一层"寄存器流水"的预取：

```cpp
  int smem_pipe_read  = 0;
  int smem_pipe_write = K_PIPE_MAX-1;
  Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
  Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);

  auto K_BLOCK_MAX = size<2>(tCrA);      // float: 8，half: 4

  if (K_BLOCK_MAX > 1) {
    cp_async_wait<K_PIPE_MAX-2>();       // 等第一个 stage 的 gmem→smem 完成
    __syncthreads();
    copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
  }
```

`smem_pipe_read = 0`（从第 0 个 stage 读），`smem_pipe_write = 2`（前两个 stage 已被预取，下一个写第 3 个）。`K_BLOCK_MAX` 是一个 stage 在 K 方向还分几小块（float 的 `bK=8` 且 FMA 的 K atom 为 1 → 8；half 的 `bK=64` 且 mma 的 K atom 为 16 → 4）。

`cp_async_wait<K_PIPE_MAX-2>` 即 `cp_async_wait<1>`：允许最多 1 个 group 还没完成。因为预取时每个 stage 都 `commit` 了一次，此时有两个 group 在飞，`wait<1>` 保证最老的那个（第 0 个 stage）已经落地——而它正是我们马上要读的。

预取完成后打印当前线程在 smem 里拿到的 8×8 片段，`(_1,_8,_8):(_0,_16,_129)` 里的 `129` 正是前面 padding 的 `bM+1`：

```
// smem_ptr[32b](0x7f0ac5000000) o (_1,_8,_8):(_0,_16,_129):
//   6.80e-01  5.51e-01  9.46e-01 -7.61e-02 -4.74e-01 -3.11e-01  4.70e-01 -4.30e-01
// ----------------------------------------
//  -2.11e-01  1.56e-02  7.63e-01  7.28e-01 -3.61e-01 -5.26e-01  3.09e-01 -2.17e-01
// ----------------------------------------
//   5.66e-01  4.11e-01 -7.41e-01  9.93e-01  1.20e-01 -8.77e-01 -5.91e-01 -7.03e-01
// ----------------------------------------
//   5.97e-01 -2.10e-01  7.21e-01 -7.05e-01  8.12e-01 -9.57e-01  2.08e-01 -7.16e-01
// ----------------------------------------
//   8.23e-01  4.64e-01 -2.68e-02 -4.83e-01  1.45e-01  9.62e-02 -3.39e-01  7.05e-01
// ----------------------------------------
//  -6.05e-01 -8.46e-01  9.01e-01  3.79e-01 -8.99e-01 -2.68e-03 -7.91e-01  8.41e-01
// ----------------------------------------
//  -3.30e-01 -3.40e-03  8.76e-01  2.67e-01 -3.55e-03  9.08e-01 -6.34e-01 -6.42e-01
// ----------------------------------------
//   5.36e-01  6.94e-03  3.05e-01 -1.16e-01  6.42e-01 -3.32e-01  7.24e-01  9.39e-01

// smem_ptr[32b](0x7f0ac5000000) o (_1,_8):(_0,_16):
//   6.80e-01  5.51e-01  9.46e-01 -7.61e-02 -4.74e-01 -3.11e-01  4.70e-01 -4.30e-01
```

主循环：

```cpp
  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE_MAX-1))
  {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
      if (k_block == K_BLOCK_MAX - 1)
      {
        tXsA_p = tXsA(_,_,_,smem_pipe_read);
        tXsB_p = tXsB(_,_,_,smem_pipe_read);
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();
      }

      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
      copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
      copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));

      if (k_block == 0)
      {
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
      }

      gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
    }
  }
```

外层 `while` 走 gmem 上的 K tile，内层 `for` 走一个 stage 内 K 方向的 `k_block`。

- `if (k_block == K_BLOCK_MAX-1)`：一个 stage 的 `k_block` 快算完时，切到 `smem_pipe_read`，`cp_async_wait<1>` 等这个 stage 的异步拷贝完成，`__syncthreads()` 保证所有线程都看到。
- 每次迭代搬下一小块到寄存器（用 `k_block_next` 提前一拍），让"smem→reg"和"reg 计算"重叠。
- `if (k_block == 0)`：发射下一个 gmem tile 到 `smem_pipe_write`（此时它正好是刚空出来的那个 stage），然后 `commit`，并推进读写指针。

这里 `cp.async` 从 gmem 直接写 smem，不占用寄存器、不阻塞线程，而写的是刚被读完的那个 stage（`smem_pipe_write` 在推进前等于 `smem_pipe_read` 的旧值）。于是"往 smem 灌下一块"和"对当前寄存器做 gemm"就能并行，这就是三级流水的价值——比 `sgemm_02` 用寄存器中转更省、更深。

`while (k_tile_count > -(K_PIPE_MAX-1))` 的边界：`k_tile_count` 是"还没发射的 gmem tile 数"，初始已被预取减到 510；循环里每轮 `--`，当它减到 `-(K_PIPE_MAX-1) = -2` 时退出，多跑的这两轮正好用来把预取阶段塞进流水、还没被消费的最后两块收尾算掉。

## HALF 版的 mma + ldmatrix + swizzle

half + TN 命中的是一个手写重载，它把 float 版的 FMA 换成真正的 Tensor Core 指令，其余流水骨架完全一致。主要差别有三处。

### 1. 规模、swizzle 和向量化 cp.async

```cpp
  auto bK = Int< 64>{};
  auto bP = Int<3>{};  // Pipeline

  auto swizzle_atom = composition(Swizzle<3,3,3>{},
                                  Layout<Shape <_8,Shape <_8, _8>>,
                                         Stride<_8,Stride<_1,_64>>>{});
  auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(swizzle_atom, make_shape(bN,bK,bP));
```

`bK` 从 8 提到 64——Tensor Core 一次吃 16 个 K，`bK` 大一点摊薄开销。smem 不再用 `+1` padding，改成 swizzle：把一个 8×64（half，即 8×128 字节）的 atom 按 `Swizzle<3,3,3>` 重排地址，让 `ldmatrix` 读取时既不产生 bank conflict、又能做 128-bit 对齐的向量读。`tile_to_shape` 再把它铺成 `(128,64,3)`。swizzle 的原理（几位用来异或地址）在讲 swizzle 的文章里细说过，这里只要知道它替代了 padding 的作用。

拷贝也换成向量化 + 异步：

```cpp
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // 16x8 k-major
                                    Layout<Shape< _1,_8>>{});               // 1x8 k-major
```

atom 是 `uint128_t`（16 字节），value layout `1×8` 表示每个线程一次拷 8 个 half = 16 字节，正好一个 128-bit 事务；256→128 个线程（16×8）就够。

进入 kernel 后先分块，A、B、C 的 gmem 视图和 smem 视图分别是：

```cpp
gA : gmem_ptr[16b](0x7f9c70000000) o (_128,_64,64):(4096,_1,_64)
gB : gmem_ptr[16b](0x7f9c6c000000) o (_128,_64,64):(4096,_1,_64)
gC : gmem_ptr[16b](0x7f9c68000000) o (_128,_128):(_1,5120)
```

A 是 5120×4096、block 是 128×64，所以分块后 `gA` 是 `128×64×64`，最后一维是 K 方向上的 tile 数（4096/64=64）。smem 侧 `sA` 则是 `128×64×3`，多出来的一维是 pipeline stage：

```cpp
sA : smem_ptr[16b](0x7f9d00000400) o ((_128,_1),(_64,_1),(_1,_3)):((_64,_0),(_1,_0),(_0,_8192))
```

再用 copy 的线程布局切 gmem 和 smem，得到每线程的搬运片：

```cpp
tAgA : gmem_ptr[16b](0x7f9c70000000) o ((_8,_1),_8,_1,64):((_1,_0),65536,_0,_64)
tAsA : smem_ptr[16b](0x7f9d00000400) o ((_8,_1),_8,_1,(_1,_3)):((_1,_0),_1024,_0,(_0,_8192))
```

copy 有 16×8=128 个线程、每线程 8 个元素，一个 block 中每线程因此处理 8×8 个元素。`tAgA` 第一维的 8 是连续 8 个元素（stride 1），第二维的 8 相隔 16 行（stride 16×4096=65536），末维 64 是不同 block（stride 64）。`tAsA` 同理，只是行间隔换成了 smem 的 1024、stage 间隔是 8192。

### 2. mma 与 ldmatrix

```cpp
  TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                                 Layout<Shape<_2,_2>>{},    // 2x2x1 MMA Atoms
                                 Tile<_32,_32,_16>{});      // 32x32x16 Tiled MMA for LDSM

  Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;
```

`SM80_16x8x16` 是一个 32 线程的 mma atom，算 16×8×16。`Layout<Shape<_2,_2>>` 把它在 M、N 方向各复制 2 份 → 覆盖 32×16、128 个线程。`Tile<_32,_32,_16>` 把 TiledMMA 总量定成 32×32×16，也就是 N 方向还要再翻一倍。

N 设成 32 是为了对齐 `ldmatrix`。看打印出来的 `ThrLayoutVMNK: (_32,_2,_2,_1):(_1,_32,_64,_0)`——N 方向的 stride 是 `0`，说明线程不沿 N 扩展，多出来的那 16 列由每个线程自己吃两份。这样每个线程手里的 B 片段正好是 8 个 half（16 字节），刚好对齐 `SM75_U32x4_LDSM_N` 一次 `ldmatrix.x4` 输出的 4 个 32-bit 寄存器。float 版不需要这层考虑，因为它不用 ldmatrix。

`SM75_U32x4_LDSM_N` 对应 `ldmatrix.sync.aligned.m8n8.x4`：一个 32 线程的 warp 协同，把 smem 里 4 个 8×8 的小块一次性转置/搬运到各线程寄存器。用它比逐元素搬快得多，并且天然匹配 mma 期望的 `TN` 布局。

编译器打印出的两个 TiledCopy 和 TiledMMA 的静态结构如下，从中可以直接读出线程布局和 value 布局：

```
TiledCopy
  Tiler_MN:       (_16,_64)
  TiledLayout_TV: ((_8,_16),_8):((_128,_1),_16)
Copy_Atom
  ThrID:        _1:_0
  ValLayoutSrc: (_1,_8):(_0,_1)
  ValLayoutDst: (_1,_8):(_0,_1)
  ValLayoutRef: (_1,_8):(_0,_1)
  ValueType:    16b
TiledCopy
  Tiler_MN:       (_16,_64)
  TiledLayout_TV: ((_8,_16),_8):((_128,_1),_16)
Copy_Atom
  ThrID:        _1:_0
  ValLayoutSrc: (_1,_8):(_0,_1)
  ValLayoutDst: (_1,_8):(_0,_1)
  ValLayoutRef: (_1,_8):(_0,_1)
  ValueType:    16b
TiledMMA
  ThrLayoutVMNK:  (_32,_2,_2,_1):(_1,_32,_64,_0)
  PermutationMNK: (_32,_32,_16)
MMA_Atom
  ThrID:      _32:_1
  Shape_MNK:  (_16,_8,_16)
  LayoutA_TV: ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))
  LayoutB_TV: ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
  LayoutC_TV: ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
```

`TiledLayout_TV: ((_8,_16),_8):((_128,_1),_16)` 说明 copy 的 128 个线程按 8×16 排布，每线程沿 K 方向连续吃 8 个 half；`ThrLayoutVMNK: (_32,_2,_2,_1):(_1,_32,_64,_0)` 则是 mma 的线程分布，N 维 stride 为 0 对应前面说的“线程不沿 N 扩展”。

分块依旧走 `partition_S` + `retile_D`（前面已解释过 retile 为什么变 layout），只是 `s2r_atom` 从 `AutoVectorizingCopy` 换成了 `ldmatrix`，所以：

```cpp
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);   // (CPY,MMA_M,MMA_K,PIPE)
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);    // (CPY,MMA_M,MMA_K)
```

`tXsA` 里 `(_8,_1)` 那部分就是一个 atom 内每线程沿 smem 读的 8 个连续 half（stride 1），`_32` 是按 MMA tile 复制的份数。ldmatrix 一次把 `tXsA` 的这块搬进 `tXrA`。

half 的分块结果：

```cpp
tCgC : gmem_ptr[16b](0x7f9c68000000) o ((_2,_2),_4,(_2,_4)):((5120,_8),_32,(81920,163840))
tCsA : smem_ptr[16b](0x7f9d00000400) o ((_2,_2,_2),_4,_4):((_1,_512,_8),_2048,_16)
tCsB : smem_ptr[16b](0x7f9d0000c400) o ((_2,_2),_8,_4):((_1,_8),_1024,_16)
```

- `tCgC` 第一维 `(_2,_2)` 是每线程在一个 mma 里要写的 4 个数据；smem 的 M 方向有 4 个 tiledmma，所以是 4；N 方向一个 tiledmma 里线程 0 负责 2 个 mma，smem 中又有 4 个 tiledmma，所以是 `(_2,_4)`。
- `tCsA` 第一维 `(_2,_2,_2)` 是每线程一个 mma 里的 8 个 A，M 和 K 方向各 4 个，所以是 `((_2,_2,_2),_4,_4)`。
- `tCsB` 第一维 `(_2,_2)` 是每线程一个 mma 里的 4 个 B，N 方向 2×4=8 个，K 方向 4 个，所以是 `((_2,_2),_8,_4)`。

对应到寄存器里申请到的 `tCrA`、`tCrB`、`tCrC`，形状与 smem 侧的 `tCsA/tCsB`、`tCgC` 一一对齐，只是 stride 换成寄存器排布：

```cpp
tCrA : ptr[16b](0x7f9cbdfff9d0) o ((_2,_2,_2),_4,_4):((_1,_2,_4),_32,_8)
tCrB : ptr[16b](0x7f9cbdfffad0) o ((_2,_2),_8,_4):((_1,_2),_16,_4)
tCrC : ptr[16b](0x7f9cbdfffbd0) o ((_2,_2),_4,(_2,_4)):((_1,_2),_4,(_16,_32))
```

`ldmatrix` 分块后，smem 侧和寄存器侧分别是：

```cpp
tXsA : smem_ptr[16b](0x7f9d00000400) o ((_8,_1),_4,_4,(_1,_3)):((_1,_0),_2048,_16,(_0,_8192))
tXrA : ptr[16b](0x7f9cbdfff9d0) o ((_8,_1),_4,_4):((_1,_0),_32,_8)

tXsB : smem_ptr[16b](0x7f9d0000c400) o ((_8,_1),_4,_4,(_1,_3)):((_1,_0),_2048,_16,(_0,_8192))
tXrB : ptr[16b](0x7f9cbdfffad0) o (((_4,_2),_1),_4,_4):(((_1,_16),_0),_32,_4)
```

`tXsA` 是 ldmatrix 对 smem 的划分：`(_8,_1)` 表示一个 atom 中每线程沿 smem 读一行 8 个元素；smem 是 128×64，一个 tiledcopy 是 32×16，MN 方向各要 4 份，再乘 pipe stage=3，于是每线程在 smem 中加载 `((_8,_1),_4,_4,(_1,_3))` 个元素。stride 上，8 个连续元素对应 1，行方向每线程相隔 16，列方向相隔 32×64=2048。

`tXrA` 是 `retile_D` 之后的寄存器视图，同样每线程 8 个元素，但第一维重排成 `(_8,_1):(_1,_0)`——8 个寄存器紧凑连续，正好对上一次 `ldmatrix.x4` 的输出。`tCrA` 原本是 `((_2,_2,_2),_4,_4):((_1,_2,_4),_32,_8)`，线程持有的元素集合没变，只是这 8 个元素在 value 维上的组织方式被重贴了标签。

M、N、K 各维的 tile 划分关系如下图：

![](/assets/gemm_sm80/image(1).png)

`ldmatrix` 的线程/数据对应关系如下（`print_latex` 输出）。A 的源布局是 `((_16,_2,_2,_2),(_8,_1)):((_1@0,_8@1,_16@0,_0),(_1@1,_0))`，B 的源布局是 `((_8,_2,_2,_2,_2),(_8,_1)):((_1@0,_8@1,_16@0,_0,_8@0),(_1@1,_0))`：

![](/assets/gemm_sm80/image(2).png)

![](/assets/gemm_sm80/image(3).png)

### 3. 主循环

和 float 一模一样——同样预取 2 个 stage、同样 `cp_async_wait<K_PIPE_MAX-2>`、同样 `while (k_tile_count > -(K_PIPE_MAX-1))` 和 `if (k_block == K_BLOCK_MAX-1/0)` 的结构。唯一区别是内层：

```cpp
      copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
      copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));
```

`copy(s2r_atom, ...)` 底层走 `ldmatrix`，而 float 那次是向量化赋值。`K_BLOCK_MAX = size<2>(tCrA) = 4`（`bK=64`，mma K atom=16）。

## 小结

`sgemm_sm80` 相比 `sgemm_02` 升级了三处：

1. `cp.async` 异步拷贝：gmem→smem 不再经过寄存器、不阻塞线程，用 `cp.async.commit_group` / `cp_async_wait<N>` 管理。
2. 三级（`bP=3`）smem 流水：读下一块、存当前块、算上一块三者真正重叠；比寄存器中转更省寄存器、流水更深。
3. half 版上 Tensor Core：`SM80_16x8x16` 的 `mma.sync` + `ldmatrix.x4` + swizzle smem，第一次用真正的矩阵乘指令。

还有一个贯穿全篇的抽象：smem→reg 的搬运统一走 `make_tiled_copy_A/B` + `partition_S` + `retile_D`——`partition` 切共享内存、`retile` 重贴私有寄存器的标签，两者配合让任意 copy atom（向量化赋值、`ldmatrix`）都能对上 MMA 的分区，而线程持有的数据不变。

局限也很清楚：A、B 仍要显式地在 gmem 和 smem 之间搬（copy atom 逐元素或小向量），同步还依赖 `__syncthreads()`，而且说到底还是"线程主动搬 + 线程主动算"。下一篇 `sgemm_sm90` 会用 Hopper 的 `wgmma` 和更深的流水，让计算直接吃 shared memory 里的数据。
