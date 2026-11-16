---
title: CuTe 学习笔记（九）SGEMM 01
date: 2025-04-06 18:00:00
tags: [CUTLASS, CuTe]
categories: [Cutlass 学习笔记, CuTe]
description: 从最基础的 CuTe 矩阵乘例子 sgemm_1 出发，分析如何用 local_tile 按 thread block 分块、用 local_partition 按线程分块，并在 K 方向上循环累加。
---

前面几篇介绍了 CuTe 的 Layout、Tensor、TiledMMA、TiledCopy 这些基础组件，从这里开始进入实际例子。

`sgemm_1.cu` 是 cutlass 里最基础的一个矩阵乘实现。它没有使用 Tensor Core，也没有使用异步拷贝，只是用 `local_tile` 和 `local_partition` 把矩阵分块后，配合普通的 `gemm` 完成计算。虽然简单，但它把 CuTe 写 GEMM 的基本套路完整地展示了一遍，后面的 SM80、SM90 例子都是在这个骨架上不断叠加新特性。

矩阵乘的规模是 M = N = 5120，K = 4096，数据类型全是 float。transA = T 表示矩阵 A（M×K）在 K 方向连续，transB = N 表示矩阵 B（K×N）在 K 方向连续，也就是 A、B 都是 K-major，这是一个经典的 TN GEMM。

直接进入 [`gemm_tn`](https://github.com/NVIDIA/cutlass/blob/v4.1.0/examples/cute/tutorial/sgemm_1.cu#L304)。

## 定义规模与布局

```cpp
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)
```

这里定义了运算的 shape 和三个矩阵在 global memory 上的 stride。因为 A、B 都是 K-major，所以 K 方向的 stride 是 1，M/N 方向的 stride 是 leading dimension（`ldA`、`ldB`）。C 矩阵按列主序存储，所以 M 方向 stride 为 1，N 方向 stride 为 `ldC`。

接着定义 thread block 处理的规模和 shared memory 的布局。

```cpp
  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM,bK), LayoutRight{});   // (m,k) -> smem_idx; k-major
  auto sB = make_layout(make_shape(bN,bK), LayoutRight{});   // (n,k) -> smem_idx; k-major
  auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major
```

每个 thread block 处理一个 128×128 的输出块，K 方向一次处理 8 个元素（`bK = 8`）。shared memory 中，`sA`、`sB` 用 `LayoutRight` 声明为 K-major，这样和 global memory 上的访问方式一致，拷贝时就是连续读取。

再看 A、B、C 三个矩阵的线程布局。

```cpp
  // Define the thread layouts (static)
  auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}), LayoutRight{});  // (m,k) -> thr_idx; k-major
  auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}), LayoutRight{});  // (n,k) -> thr_idx; k-major
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));                 // (m,n) -> thr_idx; m-major
```

这里都使用 256 个线程：`tA`、`tB` 是 32×8 的线程布局（用于拷贝），`tC` 是 16×16 的线程布局（用于计算）。kernel 的线程数就取 `size(tC)`。

```cpp
  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
```

grid 的大小是 M、N 方向上分块的个数，每个 block 对应输出矩阵中的一块。

## 数据分块

进入 kernel 后，先用 global 指针、shape、stride 创建出代表完整矩阵的 CuTe tensor。

```cpp
  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)
```

新创建的 tensor 相当于原始数据在给定 layout 下的一个视图。

然后用 `local_tile` 根据当前 block 的下标把大矩阵切出这个小块。

```cpp
  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)
```

`local_tile` 的本质是一次 `zipped_divide`：用 `cta_tiler` 去切原始 tensor，得到的 layout 有两部分——一部分是每个 tile 内部的形状，另一部分是“一共有多少个 tile”的形状；然后用 `cta_coord` 去索引“tile 个数”的那一维，就取出了某个 block 对应的 tile。

以 A 为例。`mA` 的大小是 (5120, 4096)，参与分块的 tiler 是 `(bM, bK) = (128, 8)`，所以 M 方向切成 5120/128 = 40 块，K 方向切成 4096/8 = 512 块。`logical_divide` 的结果可以理解为 `((128, 40), (8, 512))`：每一维前面是 tile 内形状，后面是 tile 个数。`zipped_divide` 把 tile 内的维度组合到一起，变成 `((128, 8), (40, 512))`。`Step<_1, X, _1>` 表示只在 M 维上取 `blockIdx.x`，K 维保留 `_`（即取出全部 K 块）。于是 `gA` 的形状就是 `(128, 8, 512)`，第二维仍是所有 K 块。

`local_partition` 用的也是同一套 `zipped_divide`，区别只在于切哪一维：`local_tile` 用坐标去切“tile 个数”那一维（第二维 / Rest），从而选中某一个 tile；`local_partition` 则用线程号去切“tile 内部”那一维（第一维 / BLK），把 tile 内的元素分给各个线程。

## 分配 shared memory

```cpp
  // Shared memory buffers
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)
```

按 layout 的 cosize 申请 shared memory 空间，再用对应的 layout 包装成 tensor。

## 拷贝阶段

```cpp
  // TUTORIAL: Example of simple raked partitioning of ThreadLayouts tA|tB over data A|B tiles

  Tensor tAgA = local_partition(gA, tA, threadIdx.x);                  // (THR_M,THR_K,k)
  Tensor tAsA = local_partition(sA, tA, threadIdx.x);                  // (THR_M,THR_K)

  Tensor tBgB = local_partition(gB, tB, threadIdx.x);                  // (THR_N,THR_K,k)
  Tensor tBsB = local_partition(sB, tB, threadIdx.x);                  // (THR_N,THR_K)
```

这里用拷贝用的线程布局 `tA`、`tB` 对 global 和 shared 上的 tile 分别做 `local_partition`，得到每个线程负责搬运的那部分数据。

以 A 为例，`gA` 的形状是 `(128, 8, 512)`，`tA` 是 `(32, 8)` 的线程布局。`logical_divide` 用 `tA` 去切 `(128, 8)` 这个 tile：M 方向 128/32 = 4 块，K 方向 8/8 = 1 块，得到 `((32, 4), (8, 1), 512)`；`zipped_divide` 重新组合为 `((32, 8), (4, 1), 512)`。用 `threadIdx.x` 索引第一维，每个线程剩下 `((4, 1), 512)`——也就是说，一个线程在单个 K block 里负责 4 个元素。一个 tile 有 128×8 = 1024 个元素，256 个线程正好平均分到 4 个。

## 计算阶段

```cpp
  // Partition sA (BLK_M, BLK_K) by the rows of tC
  Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   // (THR_M,BLK_K)
  // Partition sB (BLK_N, BLK_K) by the cols of tC
  Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   // (THR_N,BLK_K)
  // Partition gC (M,N) by the tile of tC
  Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   // (THR_M,THR_N)

  // Allocate the accumulators -- same shape/layout as the partitioned data
  Tensor tCrC = make_tensor_like(tCgC);                                // (THR_M,THR_N)
```

拷贝阶段用 `tA`、`tB`，是为了把数据高效地搬进 shared memory；计算阶段则用 `tC`，因为它规定了每个线程最终计算输出矩阵中的哪些元素。

- `sA` 是 `(128, 8)`，`tC` 是 `(16, 16)`。`Step<_1, X>` 表示只用 `tC` 的第一维（16:1）去切 `sA` 的 M 方向：128/16 = 8。于是每个线程拿到 `sA` 中的 8 行、全部 8 列，即 8×8 个元素。
- `sB` 是 `(128, 8)`，`Step<X, _1>` 表示只用 `tC` 的第二维去切 `sB` 的 N 方向，同样是 128/16 = 8，每个线程拿到 8×8 个元素。
- `tCgC` 用 `tC` 的两个维度切 `gC`（128×128），每个线程拿到 8×8 个输出元素。

注意 `tC` 有 256 个线程，而 `sA` 的 M 方向只能被 16 个线程整除，所以每 16 个线程为一组、处理相同的数据：0-15 号线程和 16-31 号线程手里的 `tCsA` 是一样的。这是 sgemm_01 这种“朴素分块”的特点——没有用 `TiledMMA`，所以存在重复读取。

用 `make_tensor_like` 在寄存器里创建一个和 `tCgC` 同形状、同 layout 的累加器 `tCrC`。

这一步的分块关系大致如下：

![](/assets/sgemm_01/image.png)

## 主循环

```cpp
  auto K_TILE_MAX = size<2>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    // Copy gmem to smem with tA|tB thread-partitioned tensors
    copy(tAgA(_,_,k_tile), tAsA);      // A   (THR_M,THR_K) -> (THR_M,THR_K)
    copy(tBgB(_,_,k_tile), tBsB);      // B   (THR_N,THR_K) -> (THR_N,THR_K)

    cp_async_fence();        // Label the end of (potential) cp.async instructions
    cp_async_wait<0>();      // Sync on all (potential) cp.async instructions
    __syncthreads();         // Wait for all threads to write to smem

    // Compute gemm on tC thread-partitioned smem
    gemm(tCsA, tCsB, tCrC);            // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)

    __syncthreads();         // Wait for all threads to read from smem
  }
```

`K_TILE_MAX` 是 K 方向上分成的块数（这里是 512）。循环体每轮做三件事：

1. 把 global 上第 `k_tile` 块数据搬到 shared memory（`copy` 作用于 `tA`/`tB` 分区）。
2. 等待 shared memory 对全部线程可见。
3. 调用 `gemm` 计算 shared memory 中的这一小块的乘积，累加进 `tCrC`。

```cpp
    copy(tAgA(_,_,k_tile), tAsA);      // A   (THR_M,THR_K) -> (THR_M,THR_K)
    copy(tBgB(_,_,k_tile), tBsB);      // B   (THR_N,THR_K) -> (THR_N,THR_K)
```

这两句把 global memory 上第 `k_tile` 块拷贝到 shared memory。注意这里的 `copy` 没有指定 copy atom，执行的是逐元素的普通赋值。

```cpp
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();
```

`cp_async_fence()` 和 `cp_async_wait<0>()` 是异步拷贝 `cp.async` 才需要的。这个例子里只是普通赋值，所以这两句其实是多余的，可以去掉。真正保证同步的是 `__syncthreads()`——它确保所有线程写入 shared memory 的结果对大家可见。

`gemm(tCsA, tCsB, tCrC)` 对 shared memory 中这一小块做矩阵乘并累加到寄存器：

```cpp
// 等价于
CUTE_UNROLL
for (int k = 0; k < size<1>(tCsA); ++k) {
  CUTE_UNROLL
  for (int m = 0; m < size<0>(tCrC); ++m) {
    CUTE_UNROLL
    for (int n = 0; n < size<1>(tCrC); ++n) {
      tCrC(m,n) += tCsA(m,k) * tCsB(n,k);
    }
  }
}
```

循环末尾的 `__syncthreads()` 保证在下一轮覆盖 shared memory 之前，本轮的读取都已完成。

## Epilogue

```cpp
  axpby(alpha, tCrC, beta, tCgC);
```

最后用 `axpby` 把累加器结果乘上 `alpha`、加上 `beta * C`，写回 global memory 的 C 矩阵。

## 小结

这个例子用 CuTe 完成了一个基础矩阵乘，核心思路可以归纳为三步：

1. 用 `local_tile` 按 thread block 把 global memory 上的大矩阵切开，每个 block 并行处理输出矩阵的一块。
2. 用 `local_partition` 把 block 内的数据按线程切开，每个线程负责其中的一部分（拷贝用 `tA`/`tB`，计算用 `tC`）。
3. 在 K 方向上循环：搬入一块、算一块、累加，直到 K 走完。

它没有异步拷贝、没有 pipeline、也没有 Tensor Core，但把 CuTe 分块的基本模型讲清楚了。下一篇 `sgemm_02` 会在同样的骨架上，把手工指定线程 layout 的方式替换成 `TiledCopy` 和 `TiledMMA`，并加入寄存器中转的软件流水。
