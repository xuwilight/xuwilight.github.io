---
title: CuTe 学习笔记（十）SGEMM 02
date: 2025-04-12 18:00:00
tags: [CUTLASS, GEMM]
categories: [Cutlass 学习笔记]
description: sgemm_02 在 sgemm_01 的骨架上，用 TiledCopy 和 TiledMMA 替换手写的线程布局，并通过 gmem→rmem→smem 的寄存器中转实现软件流水。
---

`sgemm_02` 和 `sgemm_01` 的整体结构一样，还是 `local_tile` 分块 + K 方向循环累加。它主要有两点不同：

1. 不再手写 `tA`、`tB`、`tC` 这些线程 layout，而是用 `TiledCopy` 和 `TiledMMA` 来描述“怎么拷贝”和“怎么计算”。
2. 拷贝路径从“gmem 直接到 smem”改成“gmem → rmem → smem”，用寄存器做中转，从而实现软件流水。

矩阵乘的规模依然是 M = N = 5120，K = 4096，数据类型全是 float，A、B 都是 K-major（TN GEMM），C 按列主序存储。

`sgemm_2.cu` 里其实有两个版本的 kernel，`gemm_nt`（用 128-bit 向量拷贝）和 `gemm_tn`（逐元素拷贝 + padded smem）。本文分析的是 `gemm_tn`，因为它的结构最能对上 `sgemm_01`。

直接进入 [`gemm_tn`](https://github.com/NVIDIA/cutlass/blob/v4.1.0/examples/cute/tutorial/sgemm_2.cu#L327)。

## 定义规模与布局

矩阵规模、global stride、CTA tile 都和 `sgemm_01` 一样，bM = bN = 128，bK = 8。区别在 shared memory 的布局：

```cpp
  // Define the smem layouts (static)
  auto sA = make_layout(make_shape (      bM,          bK),
                        make_stride(Int<1>{}, bM+Int<1>{}));        // (m,k) -> smem_idx; padded m-major
  auto sB = make_layout(make_shape (      bN,          bK),
                        make_stride(Int<1>{}, bN+Int<1>{}));        // (n,k) -> smem_idx; padded n-major
  auto sC = make_layout(make_shape(bM, bN));                        // (m,n) -> smem_idx
```

`sA` 是 m-major：m 方向连续（stride = 1），k 方向的 stride 是 `bM + 1`。这里的 `+1` 是为了避免 shared memory 的 bank conflict。

shared memory 被划分为 32 个 bank，每个 bank 4 字节，一个 float 正好占一个 bank。如果把 k 方向的 stride 设成 `bM = 128`，那么固定 m、沿 k 方向取元素时，地址依次是 `m, m+128, m+256, ...`，每隔 128 个 float。128 是 32 的整数倍，所以这些地址全都落在同一个 bank 上。当一组线程同时读取同一列中的不同 k 时，就会全部挤在一个 bank，产生 32 路 bank conflict，串行化访问。

把 stride 改成 `bM + 1 = 129`，相邻 k 的地址差变成 129，而 129 mod 32 = 1，于是每换一个 k，bank 就往右移一格、在 32 个 bank 上轮转，冲突基本被消除。多出来的这一个元素就是“padding”。

```cpp
  // TUTORIAL: Construct TiledCopy to define the Copy_Atom to use and the
  //           partitioning pattern to apply.
  // Each thread will copy 1x1 elements of type TA.
  // Use 32x8 of these threads arranged in k-major.

  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<TA>, TA>{},
                                    Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape< _1,_1>>{});              // Val layout  1x1
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<TB>, TB>{},
                                    Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape< _1,_1>>{});              // Val layout  1x1
```

`make_tiled_copy` 用三个东西描述“如何拷贝”：

- **Copy_Atom**：`UniversalCopy<TA>` 表示最基础的、一个线程拷一个元素的拷贝操作。
- **Thread layout**：`Layout<Shape<_32,_8>,Stride<_8,_1>>`，把原子操作铺成 32×8 个线程、k-major 排列，共 256 个线程。
- **Value layout**：`Layout<Shape<_1,_1>>`，表示每个线程拷 1×1 个元素。

因为每个原子只要一个线程，所以 `copyA` 就是让 256 个线程各执行一次拷贝。

```cpp
  // TUTORIAL: Construct TiledMMA to define the MMA_Atom to use and the
  //           partitioning pattern to apply.
  // Use a 1x1x1 FMA on the types TC += TA * TB. Each atom requires a single thread.
  // Reproduce that atom 16x16x1 times (m-major) across threads so that we use 256 threads.

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 TiledMMA
```

`make_tiled_mma` 同理：`UniversalFMA` 是“a * b + c”这种最基础的运算，只要一个线程；线程布局 `16×16×1` 把它复制成 256 个线程，m-major 排列。

有了 `TiledCopy` 和 `TiledMMA`，后面就不需要像 `sgemm_01` 那样手写线程 layout，直接拿这两个对象去分块即可。kernel 的线程数取 `size(mmaC)`，也就是 256。

## TiledCopy 分区

进入 kernel 后，创建 tensor、`local_tile` 分块、申请 shared memory，这些和 `sgemm_01` 完全一样。到分区这一步换成了 `TiledCopy`：

```cpp
  // TUTORIAL: Example of partitioning via a TiledCopy

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K)
  // Allocate registers same shape/layout as partitioned data
  Tensor tArA = make_fragment_like(tAsA);                              // (CPY,CPY_M,CPY_K)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K)
  // Allocate registers same shape/layout as partitioned data
  Tensor tBrB = make_fragment_like(tBsB);                              // (CPY,CPY_N,CPY_K)
```

`get_slice(threadIdx.x)` 取出当前线程在 `TiledCopy` 中对应的那份，然后用 `partition_S`、`partition_D` 分别对源（gmem 的 `gA`）和目标（smem 的 `sA`）做分块，得到该线程要处理的元素。

这里比 `sgemm_01` 多了一步：用 `make_fragment_like` 在寄存器里创建一份和 `tAsA` 同形状的 `tArA`。数据拷贝的路线因此变成 gmem → rmem（`tArA`）→ smem（`tAsA`），这个中转是实现软件流水的关键。

## TiledMMA 分区与累加器

```cpp
  // TUTORIAL: Example of partitioning via a TiledMMA

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

  // Clear the accumulators
  clear(tCrC);
```

`mma.get_slice(threadIdx.x)` 取出当前线程在 `TiledMMA` 中对应的那份，再用 `partition_A`、`partition_B`、`partition_C` 分别对 shared memory 上的 `sA`、`sB` 和输出 `gC` 分块，得到这个线程在计算阶段需要读取/写入的数据。最后用 `make_fragment_C` 申请累加器 `tCrC` 并清零。

到这里，拷贝和计算两套分区都建好了。注意拷贝用的是 `TiledCopy` 的分区（`tAsA` 等），计算用的是 `TiledMMA` 的分区（`tCsA` 等），两者是独立的——只是它们填/读的是同一块 shared memory。

## 主循环

```cpp
  // Copy gmem to rmem for k_tile=0
  copy(copy_a, tAgA(_,_,_,0), tArA);
  copy(copy_b, tBgB(_,_,_,0), tBrB);
```

进入循环前，先把第 0 块数据从 gmem 预取到寄存器 `tArA`、`tBrB`。这样循环里就能“当前块在算、下一块在预取”，把读取和计算重叠起来。

```cpp
  auto K_TILE_MAX = size<3>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    // Copy rmem to smem with tA|tB thread-partitioned tensors
    __syncthreads();         // Wait for all threads to consume smem
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();         // Wait for all threads to consume smem

    // Copy gmem to rmem for k_tile+1 with tA|tB thread-partitioned tensors
    int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
    copy(copy_a, tAgA(_,_,_,k_tile_next), tArA);
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBrB);

    // Compute gemm on mma-partitioned smem
    gemm(mma, tCsA, tCsB, tCrC);
  }
```

每一轮循环做四件事：

1. **第一个 `__syncthreads()`**：等所有线程都读完上一轮 shared memory 中的数据，避免被本轮覆盖。
2. **rmem → smem**：把寄存器里预取好的数据写进 shared memory。
3. **第二个 `__syncthreads()`**：确保 shared memory 写入完成，对后续计算可见。
4. **预取下一块到 rmem + 计算当前块**：这两步操作的是不同的数据（寄存器 vs shared memory），可以重叠执行。

这里的关键是把拷贝拆成两段——gmem→rmem 和 rmem→smem。如果像 `sgemm_01` 那样直接从 gmem 拷到 smem，那么“读取”和“计算”之间只能用 `__syncthreads()` 强行隔开，无法重叠。有了寄存器这一层中转后：

- 预取下一块 `k_tile+1` 到 `tArA` 的 gmem 读取，可以和当前块 `k_tile` 在 shared memory 上的 `gemm` 计算**并行**；

也就是说，主循环把“阻塞式的读—算”改成了“读下一块的同时算当前块”，这就是最朴素的软件流水（双缓冲）。

## 小结

和 `sgemm_01` 相比，`sgemm_02` 做了两处升级：

1. **用 `TiledCopy` / `TiledMMA` 代替手写线程 layout**。`make_tiled_copy` 和 `make_tiled_mma` 把“原子操作 + 线程布局 + value 布局”封装好，分块时直接 `get_slice` 再 `partition_S`/`partition_D`/`partition_A`/...，比 `sgemm_01` 手动摆 `tA`/`tB`/`tC` 清爽很多。
2. **引入 `gmem → rmem → smem` 的寄存器中转**。借这一层中转，主循环得以把下一块的 gmem 读取和当前块的计算重叠起来。

另外，shared memory 的 k 方向 stride 用了 `bM + 1` 的 padding，用来打散地址、消除沿 k 方向的 bank conflict。

不过这一版仍然有几个“浪费”：每个 `k_tile` 都重新覆盖整块 shared memory（单缓冲，不是多 stage 的 pipeline），拷贝还是逐元素（没有用到 `cp.async` 或向量化），计算也是普通的 FMA（没有 Tensor Core）。下一篇 `sgemm_sm80` 会用上 `cp.async` 和真正的 `TiledMMA`（`mma.sync`），把多 stage pipeline 补进来。
