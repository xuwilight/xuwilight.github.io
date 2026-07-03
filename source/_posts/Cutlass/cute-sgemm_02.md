---
title: CuTe 学习笔记（十）SGEMM 02
date: 2025-04-12 18:00:00
tags: [CUTLASS, GEMM]
categories: [Cutlass 学习笔记]
description: sgemm_02 和 sgemm_01 没什么区别。主要展示了通过 make_tiled_mma 和 make_tiled_copy 的使用方式。 矩阵乘的规模依然是 M = N = 5120，K = 4096。 数据类型全是 float，矩阵 A 选择 row-major，矩阵 B 选择 column-major。
---

sgemm_02 和 sgemm_01 没什么区别。主要展示了通过 make_tiled_mma 和 make_tiled_copy 的使用方式。

矩阵乘的规模依然是 M = N = 5120，K = 4096。

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

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape (      bM,          bK),
                        make_stride(Int<1>{}, bM+Int<1>{}));        // (m,k) -> smem_idx; padded m-major
  auto sB = make_layout(make_shape (      bN,          bK),
                        make_stride(Int<1>{}, bN+Int<1>{}));        // (n,k) -> smem_idx; padded n-major
  auto sC = make_layout(make_shape(bM, bN));                        // (m,n) -> smem_idx
```

矩阵乘相关 tensor 的设置也和 sgemm_01 相同。因为是 TN，所以矩阵 A 和矩阵 B 都是 K-major 的，矩阵 C 默认还是 M-major。block A 和 B 的规模还是 128*8。搞不懂为啥 stride 要+1。

和 01 的区别在下面：

```cpp
  // TUTORIAL: Construct TiledCopy to define the Copy_Atom to use and the
  //           partitioning pattern to apply.
  // Each thread will copy 1x1 elements of type TA.
  // Use 32x8 of these threads arranged in k-major.

  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<TA>, TA>{},
                                    Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape< _1,_1>>{});              // Val layout  1x1
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<TB>, TB>{},
                                    Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape< _1,_1>>{});              // Val layout  1x1

  // TUTORIAL: Construct TiledMMA to define the MMA_Atom to use and the
  //           partitioning pattern to apply.
  // Use a 1x1x1 FMA on the types TC += TA * TB. Each atom requires a single thread.
  // Reproduce that atom 16x16x1 times (m-major) across threads so that we use 256 threads.

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 TiledMMA
```

通过 make_tiled_copy 和 make_tiled_mma 创建了 TiledCopy 和 TiledMMA。后面可以直接通过 TiledCopy 和 TiledMMA 对象对矩阵进行分块，而不需要像 01 那样手动设置线程的 layout。
