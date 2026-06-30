---
title: CuTe 学习笔记（九）SGEMM-01
date: 2025-04-06 18:00:00
tags: [CUTLASS, SGEMM, GEMM, GPU]
categories: [Cutlass 学习笔记]
description: cutlass-v4.1 这是一个最基础的 gemm 版本。 矩阵乘的规模 M = N = 5120，K = 4096。 数据类型全是 float，矩阵 A 选择 row-major，矩阵 B 选择 column-major。 直接进入 gemm_tn。
---

cutlass-v4.1

这是一个最基础的 gemm 版本。

矩阵乘的规模 M = N = 5120，K = 4096。

数据类型全是 float，矩阵 A 选择 row-major，矩阵 B 选择 column-major。

直接进入 gemm_tn。

```cpp
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)
```

定义矩阵运算的 shape 和 stride，这里 lda 和 ldb 都是 K，ldc 是 m。

```cpp
  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM,bK), LayoutRight{});   // (m,k) -> smem_idx; k-major
  auto sB = make_layout(make_shape(bN,bK), LayoutRight{});   // (n,k) -> smem_idx; k-major
  auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major
```

定义 thread block 的大小和 shared memory 的布局。bM = bN = 128，bK = 8。

```cpp
  // Define the thread layouts (static)
  auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}), LayoutRight{});  // (m,k) -> thr_idx; k-major
  auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}), LayoutRight{});  // (n,k) -> thr_idx; k-major
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));                 // (m,n) -> thr_idx; m-major
```

定义矩阵 A，B，C 三个矩阵的线程的布局。这里使用了 256 个线程。

```cpp
  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, tA,
       B, dB, sB, tB,
       C, dC, sC, tC,
       alpha, beta);
```

进入 gemm kernel。

```cpp
  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)
```

kernel 里定义矩阵 A，B，C。按照之前定义的 shape 和 ABC 的指针创建 cute 类型的 tensor。

```cpp
  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)
```

然后根据 block 的 idx 对 tensor 进行分块。local_tile 的作用就是把 tensor 的 layout 和 block 的 layout 进行运算，然后按照 cta_coord 索引。

具体来说，如果 tensor 的大小是（256，128），cta_tiler 的大小是（128，8），所以 tensor 在 M 方向上可以被分成 2 块，在 N 方向上可以被分成 16 块。因此分块的结果就是（（128，2），（8，16）），这是 logical_divide 的结果，local_tile 调用的是 zipped_divide，会对计算结果的维度重新组合，变成（（128，8），（2，16）），因此，对第二维索引就可以得到对应的 thread block 中的数据。

这里因为 K 维度的索引是下划线，表示获取 K 维度的全部数据。所以最后索引的结果就是（128，8，16）。

```cpp
  // Shared memory buffers
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)
```

申请 shared memory 空间，根据 layout，创建 shared memory 空间的 tensor。

```cpp
  //
  // Partition the copying of A and B tiles across the threads
  //

  // TUTORIAL: Example of simple raked partitioning of ThreadLayouts tA|tB over data A|B tiles

  Tensor tAgA = local_partition(gA, tA, threadIdx.x);                  // (THR_M,THR_K,k)
  Tensor tAsA = local_partition(sA, tA, threadIdx.x);                  // (THR_M,THR_K)

  Tensor tBgB = local_partition(gB, tB, threadIdx.x);                  // (THR_N,THR_K,k)
  Tensor tBsB = local_partition(sB, tB, threadIdx.x);                  // (THR_N,THR_K)
```

使用 thread 的 layout 对 thread block 中的 tensor 进行分块。

local_partition 和 local_tile 的计算过程完全相同，唯一的区别就是对计算结果索引的位置不同。local_tile 是对第二维进行索引，local_partition 是对第一维进行索引。假如上面得到的 gA 的 layout 是（128，8，16），tA 的 layout 是（32，8），所以 logical_divide 得到的结果就是（（32，4），（8，1），16），按照 zipped_divide 的结果进行排序，分块的结果是（（32，8），（4，1），16）。第一维是线程的维度，使用 threadidx 对第一维进行索引得到的是一个线程对应的所有元素，如果 threadidx = 0，结果就是（（4，1），16）。

所以一个 thread 负责将 4 个数据从 global memory 拷贝到 shared memory。

```cpp
  // Partition sA (BLK_M, BLK_K) by the rows of tC
  Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   // (THR_M,BLK_K)
  // Partition sB (BLK_N, BLK_K) by the cols of tC
  Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   // (THR_N,BLK_K)
  // Partition gC (M,N) by the tile of tC
  Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   // (THR_M,THR_N)

  // Allocate the accumulators -- same shape/layout as the partitioned data
  Tensor tCrC = make_tensor_like(tCgC);                                // (THR_M,THR_N)
```

上面的 tA 和 tB 线程的分块主要用于拷贝，这里 tC 线程的分块用于计算。

sA 是（128,8），tC 是（16,16），因为 Step<_1, X>{}，所以会用 tC 的第一个维度 16:1 对 sA 进行分块。最终得到的结果是（8，8）。也就是一个线程处理 sA 中的 8 行 8 列元素。0-15 线程和 16-31 线程处理的 sA 中的数据相同。

sB 是（128,8），tC 是（16,16），因为 Step< X,_1>{}，所以会用 tC 的第二个维度 16:16 对 sB 进行分块。最终得到的结果也是（8，8）。

同理，使用 tC 的 layout 对 C 矩阵进行分块，并使用 make_tensor_like 在寄存器空间创建累加器。

这一步分块的过程类似下面这样。

![](/assets/sgemm_01/image.png)

```cpp
  // TUTORIAL: Example of a simple mainloop that read tiles of data into shared memory,
  //           and then computes on those tiles.
  //   copy(.) operates on the global and shared memory via the tA|tB partitioning
  //   gemm(.) operates on the shared and register memory via the tC partitioning

  auto K_TILE_MAX = size<2>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    // Copy gmem to smem with tA|tB thread-partitioned tensors
    copy(tAgA(_,_,k_tile), tAsA);      // A   (THR_M,THR_K) -> (THR_M,THR_K)
    copy(tBgB(_,_,k_tile), tBsB);      // B   (THR_N,THR_K) -> (THR_N,THR_K)

    // TUTORIAL: The above call to copy(tAgA(_,_,k_tile), tAsA) is equivalent to
    //   Tensor tAgAk = tAgA(_,_,k_tile);
    //   CUTE_UNROLL
    //   for (int i = 0; i < size(tAsA); ++i) {
    //     tAsA(i) = tAgAk(i);
    //   }

    cp_async_fence();        // Label the end of (potential) cp.async instructions
    cp_async_wait<0>();      // Sync on all (potential) cp.async instructions
    __syncthreads();         // Wait for all threads to write to smem

    // Compute gemm on tC thread-partitioned smem
    gemm(tCsA, tCsB, tCrC);            // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)

    // TUTORIAL: The above call to gemm(tCsA, tCsB, tCrC) is equivalent to
    //   CUTE_UNROLL
    //   for (int k = 0; k < size<1>(tCsA); ++k) {
    //     CUTE_UNROLL
    //     for (int m = 0; m < size<0>(tCrC); ++m) {
    //       CUTE_UNROLL
    //       for (int n = 0; n < size<1>(tCrC); ++n) {
    //         tCrC(m,n) += tCsA(m,k) * tCsB(n,k);
    //       }
    //     }
    //   }

    __syncthreads();         // Wait for all threads to read from smem
  }
```

最后进入核心计算代码。K_TILE_MAX 获取矩阵 AB 在 K 方向上分成多少块。然后在 K 方向上进行循环计算。

```cpp
    copy(tAgA(_,_,k_tile), tAsA);      // A   (THR_M,THR_K) -> (THR_M,THR_K)
    copy(tBgB(_,_,k_tile), tBsB);      // B   (THR_N,THR_K) -> (THR_N,THR_K)
```

这两句的作用是把 global memory 上的第 k_tile 的数据拷贝到 shared memory 的矩阵上。

```cpp
    cp_async_fence();        // Label the end of (potential) cp.async instructions
    cp_async_wait<0>();      // Sync on all (potential) cp.async instructions
    __syncthreads();         // Wait for all threads to write to smem
```

cp_async_fence()和 cp_async_wait<0>()这两句是异步拷贝指令 cp.async 才用到的，这里只是普通的赋值操作，可以去掉。

__syncthreads()用于确保 shared memory 的拷贝结果对所有线程可见。

最后调用 gemm(tCsA, tCsB, tCrC)计算 shared memory 中的矩阵乘，并写入到累加器 tCrC 中。最后通过 axpby(alpha, tCrC, beta, tCgC)，把累加器的结果写回 global memory 的矩阵中。

总结：这个例子使用 cute 完成了一个基本的矩阵运算。其主要思想是：

1. 使用 local_tile 对 global memory 上的大矩阵进行分块，每个 thread block 并行的处理原始矩阵的一部分数据。
1. 使用 local_partition 对 thread block 中的数据进行分块，每个线程处理 thread block 中的一部分数据。
1. 在 K 方向上根据 bK 的大小进行循环累加。
