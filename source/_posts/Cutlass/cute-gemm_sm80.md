---
title: CuTe 学习笔记（十一）GEMM SM80
date: 2025-04-20 18:00:00
tags: [CUTLASS, GEMM, SM80, GPU]
categories: [Cutlass 学习笔记]
description: 主要功能是使用 LDGSTS 指令，也就是 cp.async 进行异步拷贝。其他的与普通 GEMM 没啥区别，使用的普通矩阵乘法，没有使用 Tensor Core 计算。
---

主要功能是使用 LDGSTS 指令，也就是 cp.async 进行异步拷贝。其他的与普通 GEMM 没啥区别，使用的普通矩阵乘法，没有使用 Tensor Core 计算。

## 初始化

定义矩阵乘的规模和数据类型。这里 m = n = 5120，k = 4096。数据类型都是 float。transA = T 表示 A 矩阵 M×K 在 K 方向是连续的。transB = N，表示 B 矩阵 K×N 在 K 方向是连续的。

```cpp
  int m = 5120;
  int n = 5120;
  int k = 4096;
  char transA = 'T';
  char transB = 'N';
  using TA = float;
  using TB = float;
  using TC = float;
  using TI = float;
```

### gemm_tn

初始化完成后进入 gemm_tn 函数。

首先定义矩阵的规模 prob_shape 和对应的 stride。因为 A 在 K 方向是连续的，所以 K 方向的 stride = 1，M 方向的 stride = ldA。B 矩阵的 stride 也类似，在 K 方向上是 1。C 矩阵是按照列主序保存的，也就是在 M 方向的 stride = 1。

```cpp
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)
  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)
```

定义完 global memory 的矩阵规模后开始定义每个 thread block 处理的规模。

```cpp
  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline
```

从上面的代码可以看到，bM = 128，bN = 128，bK = 8。所以等于每个 thread block 内进行[128,8] * [8,128] = [128,128]大小的矩阵运算。bP = 3，表示 pipeline 的 stage 是 3。

下面定义 shared memory 的规模，因为 stage = 3，所以一个 thread block 需要（128 * 8 + 8 * 128 + 128 * 128）* 3 大小的 shared memory。

```cpp
  auto sA_atom                  = make_layout(make_shape (      bM,          bK),
                                              make_stride(Int<1>{}, bM+Int<1>{})); // (m,k) -> smem_idx; padded m-major
  [[maybe_unused]] auto sB_atom = make_layout(make_shape (      bN,          bK),
                                              make_stride(Int<1>{}, bN+Int<1>{})); // (n,k) -> smem_idx; padded n-major
  auto sA = tile_to_shape(sA_atom, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(sA_atom, make_shape(bN, bK, bP));
  auto sC = make_layout(make_shape(bM, bN));                        // (m,n) -> smem_idx
```

首先定义 A 矩阵的 shared memory sA，shape 是[bM, bK]，stride 是[1, bM + 1]。说明 sA 是列主序，但是为什么 K 的 stride 是 bM + 1？sB 也是相同的定义，但是因为和 sA 一样，所以后面没有用到。

因为 stage = 3，所以需要把 sA 的大小扩大 3 倍，因此使用 tile_to_shape 来得到最终的 sA layout。tile_to_shape 的功能是把一个 tile 的 layout 扩大到 target shape 的大小。上面就是把 sA_atom [128,8] 扩大到 [128,8,3]的大小。

扩大后 sA 的 layout 是：sA = ((_128,_1),(_8,_1),(_1,_3)):((_1,_0),(_129,_0),(_0,_1032))。

sB 和 sA 一样，sC 就是 bM * bN 的大小，列主序。

矩阵 A，B，C 在 global memory 和 shared memory 的布局都定义好后，就可以定义如何把数据从 global memory 搬运到 shared memory，以及如何进行矩阵运算。

```cpp
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TA>, TA>{},
                                    Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape< _1,_1>>{});              // Val layout  1x1
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TB>, TB>{},
                                    Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape< _1,_1>>{});              // Val layout  1x1
```

上面定义了矩阵 A 和矩阵 B 的 copy 方法。

首先是 SM80_CP_ASYNC_CACHEALWAYS。这个类表示使用 cp.async 进行拷贝，底层直接使用的是 ptx 指令，一个线程拷贝一个元素。因为 cp.async 这个指令是 sm80 架构提出来的，所以是 SM80 开头。CACHEALWAYS 是拷贝时 cache 的模式，表示在 L1 和 L2 cache 上都进行 cache，适用于 cp-size = 4, 8, 16 的数据类型。

Copy_Atom 代表最小的 copy 单元，也就是使用 SM80_CP_ASYNC_CACHEALWAYS 进行拷贝。Copy_Atom 在 SM80_CP_ASYNC_CACHEALWAYS 的基础上提供了一些数据类型和函数。

make_tiled_copy 创建了一个 TiledCopy 类，用于将 copy atom 映射为 thread Layout 大小。具体原理可以参考 tiledcopy。

这里第一个 Layout 是 thread 的 Layout，大小是 32*8。因为一个 copy atom 只需要一个 thread，所以 copyA 的 256 个线程会执行 256 个 copy atom。

第二个 Layout 是 value 的 Layout，因为一个 copy atom 只 copy 一个 value，所以 Layout 是 1。

```python
  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 TiledMMA
```

上面通过 make_tiled_mma 创建了一个 tiledmma 类，用于将单个 FMA 映射到 thread Layout 中。这里 thread Layout 是 16*16，UniversalFMA 就是 a * b + c = d，所以 mmaC 得 256 个线程会执行 256 个 FMA。

make_tiled_mma 的具体实现可以参考 tiledmma。

```python
  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA,
       B, dB, sB, copyB,
       C, dC, sC, mmaC,
       alpha, beta);
```

前面的东西定义完成后就可以启动 kernel 了，这里一个 block 的 thread 是 256。

### gemm_device

kernel 里首先是 make tensor。

```python
  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)
```

这里通过矩阵 A，B，C 的 global memory 指针和 shape，stride 创建了新的 tensor mA，mB，mC。新创建的 tensor 等于是原始数据在新的 layout 上的视图。

```python
  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)
```

这里根据 block 的大小对 global memory 的矩阵进行分块。cta_coord 是当前 block 的坐标，对应着要处理原始矩阵的那一部分。

cute 中的分块主要有两种 local_tile 和 local_partition。比如一个矩阵的大小是[8, 24]，如果使用 tiler [4,8]进行分块，则可以分成[2,3]块。如果我们把 tiler 的 layout 放在第一维，分得的块数的 layout 放在第二维，因此可以得到分块后的 layout，((_4,_8),(2,3))。

对于 local_tile，我们使用 block 的 idx 索引第二维，这样每个 block 就对应一个 tiler。

对于 local_partition，我们线程 id 索引第一维，这样每个线程就会得到 block 数据中的一部分。

所以一般使用 local_tile 对 block idx 进行分块，每个 block 得到全部数据的一部分。然后使用 local_partition 对单个 block 中的数据进行分块，这样 block 中的线程就可以分配到每个线程上。

在上面的代码中，local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{}); 表示原始矩阵 A 按照 cta_tiler 进行分块，然后按照 cta_coord 的坐标进行索引。Step<_1, X,_1>{}表示只处理第 1 维和第三维，因为 cta_tiler 的第一维是 blockIdx.x，第三维是_，等于分块后在 M 方向上获取 blockIdx.x，在 K 方向上获取所有的数据。

对 B 和 C 的分块也类似。

```python
  // Shared memory buffers
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K,PIPE)
```

分块完成后创建 smemA 和 smemB 的 buffer，并按照之前创建的 layout make 成 tensor。

```python
  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,PIPE)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K,PIPE)
```

然后处理每个线程需要分配的数据。

前面我们已经创建了 tiled copya。tiled copya 里面有 32*8=256 个线程，每个线程处理一个元素。因此我们通过 get_slice 获取每个线程需要处理的元素的位置。然后通过 partition_S 和 partition_D 获取在 src 和 dst 数据中该线程对应的元素。

比如，当 tid = 0 时，线程需要处理 tiled copy 中的 32*8 的第一个元素。gA 此时的 shape 是[128,8,512]，所以 partition_S 后，线程 0 需要处理 gA 中[4,512]的数据。其中 gA 的 128*8 大小的 block 需要 4 个 tiled copy，线程 0 在每个 tiled copy 中处理 1 个元素，所以线程 0 需要在 128*8 的 gA 中处理 4 个元素，4 个元素的坐标分别是[0,0],[32,0],[64,0],[96,0]。

同理 partition_D 也一样，一个线程需要处理 4 个元素。

下面进入计算阶段。

因为 stage = 3，所以 shared memory 有三个 buffer，可以先读取两个 buffer 的数据，然后利用异步的特性，在读取第三个 buffer 的数据时计算前两个 buffer，这样可以把读取和计算 overlap 起来。

```cpp
  //
  // PREFETCH
  //

  auto K_PIPE_MAX = size<3>(tAsA);

  // Total count of tiles
  int k_tile_count = size<3>(tAgA);
  // Current tile index in gmem to read from
  int k_tile_next = 0;

  // Start async loads for all pipes but the last
  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0) { ++k_tile_next; }
  }
```

上面的代码时 prefetch 阶段，也就是先读取前两个 buffer 的数据。copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe)); 就是使用 copy_a 中的 copy 方法把 tAgA 的第 k_tile_next 的数据读到 tAsA 的第 k_pipe 个 buffer 中。

cp_async_fence() 就是 ptx 的 cp.async.commit_group 指令，用于把前面的异步拷贝提交。

下面是对 tiled mma 进行按线程分块。

```cpp
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.make_fragment_A(tCsA(_,_,_,0));                // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB(_,_,_,0));                // (MMA,MMA_N,MMA_K)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
```

上面对 tiled mma 进行 slice 的作用和对 copy 进行 slice 的作用基本相同。就是指定一个 thread 要计算 shared memory 中的哪些数据。

tiled mma 需要从 sA，sB 中读数据，然后把计算结果保存到 sC 中，因此需要通过 partition_A，partition_B，partition_C 对 sA，sB 和 sC 进行 partition。

由于 tiled mma 的 thread layout 是 16*16*1，sA 的大小是 128*8，所以 16*1 个线程可以读取 sA 中的所有数据，一个线程读取 8*8 个数据，因为 mma 一共 256 个线程，所以其余线程以 16 个为一组会重复读取。

同样的对于 sB，也是 16 个线程可以读取 sB 中所有的数据，一个线程读取 8*8 个数据。由于是矩阵 B，所以以 16 为一组的线程读取的数据是相同的。

对于 sC，sC 的大小是 128*128，按照 16*16 可以分为 8*8 块，每个线程就读取每块中的一个数据，也就是 8*8 个数据。

如下图所示。

对于矩阵 A，0-15 个线程可以读取 sA 中的所有元素，一个线程对应一行元素，对应 sA 中的 8 行，下图只画了两行。其余线程，如 16-31 的读取方式和 0-15 一样。

对于矩阵 B，0-15 个线程对应 B 中相同的一列，0，16，32 等线程对应不同的列。

对于矩阵 C，每个线程的元素和 C 中元素的对应关系如下所示。

![](/assets/gemm_sm80/image.png)

具体是怎么实现的请参考 tiledmma。

```cpp
  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.make_fragment_A(tCsA(_,_,_,0));                // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB(_,_,_,0));                // (MMA,MMA_N,MMA_K)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
```

然后通过 make_fragment_A，make_fragment_B 和 make_fragment_C 申请寄存器。

后面就可以计算了。

但是在新版本的 cutlass 中加了下面的代码：

```cpp
  //
  // Copy Atom retiling
  //

  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K,PIPE)
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                         // (CPY,MMA_M,MMA_K)

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy   s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);                        // (CPY,MMA_N,MMA_K,PIPE)
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);                         // (CPY,MMA_N,MMA_K)
```

老版本中是直接通过赋值操作把元素从 smem 读到 rmem 中的。但是这里可以通过一个 s2r_atom_a 和 s2r_atom_b 的 copy atom 来把数据从 smem 读到 rmem 中。

对于 float，copy atom 使用的是 128bit 的拷贝，对于 half 类型，因为使用 mma 指令计算，所以可以使用 ldmatrix 指令进行拷贝。

先看这里，首先是 make_tiled_copy_A，这个函数和前面 make_tiled_copy 一样，用于生成一个 tiledcopy 类。只不过这里用的是 mma 在 A 矩阵上的线程布局。

然后通过 get_slice 获取当前线程对应的 thrcopy，然后通过 partition_S 获取当前线程对应的 sA 上的元素。

这里多了个 retile_D，意思就是按照 s2r_thr_copy_b 的布局重新对 tensor 进行 tiling。

```cpp
  // 要从第几个 buffer 开始读。
  int smem_pipe_read  = 0;
  // 要从第几个 buffer 开始写，因为前面已经写了两个 buffer 了，所以这里从第三个开始。
  int smem_pipe_write = K_PIPE_MAX-1;

  // 获取每个线程在 stage = 1 时的数据。
  Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
  Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

  // Size of the register pipeline
  auto K_BLOCK_MAX = size<2>(tCrA);

  // PREFETCH register pipeline
  if (K_BLOCK_MAX > 1) {
    // Wait until our first prefetched tile is loaded in
    cp_async_wait<K_PIPE_MAX-2>(); // 等待前面的第一个异步拷贝完成。
    __syncthreads();

    // Prefetch the first rmem from the first k-tile
    // 8 行 8 列中的第一列元素。
    copy(tCsA_p(_,_,Int<0>{}), tCrA(_,_,Int<0>{}));
    copy(tCsB_p(_,_,Int<0>{}), tCrB(_,_,Int<0>{}));
  }
```

```cpp
// smem_ptr[32b](0x7f0ac5000000) o (_1,_8,_8):(_0,_16,_129):
//   6.80e-01  5.51e-01  9.46e-01 -7.61e-02 -4.74e-01 -3.11e-01  4.70e-01 -4.30e-01
// ----------------------------------------
//  -2.11e-01  1.56e-02  7.63e-01  7.28e-01 -3.61e-01 -5.26e-01  3.09e-01 -2.17e-01
// ----------------------------------------
//   5.66e-01  4.11e-01 -7.41e-01  9.93e-01  1.20e-01 -8.77e-01 -5.91e-01 -7.03e-01
// ----------------------------------------
//   5.97e-01 -2.10e-01  7.21e-01 -7.05e-01  8.12e-01 -9.57e-01  2.08e-01 -7.16e-01
// ----------------------------------------
//   8.23e-01  4.64e-01 -2.68e-02 -4.83e-01  1.45e-01  9.62e-02 -3.39e-01  7.05e-01
// ----------------------------------------
//  -6.05e-01 -8.46e-01  9.01e-01  3.79e-01 -8.99e-01 -2.68e-03 -7.91e-01  8.41e-01
// ----------------------------------------
//  -3.30e-01 -3.40e-03  8.76e-01  2.67e-01 -3.55e-03  9.08e-01 -6.34e-01 -6.42e-01
// ----------------------------------------
//   5.36e-01  6.94e-03  3.05e-01 -1.16e-01  6.42e-01 -3.32e-01  7.24e-01  9.39e-01

// smem_ptr[32b](0x7f0ac5000000) o (_1,_8):(_0,_16):
//   6.80e-01  5.51e-01  9.46e-01 -7.61e-02 -4.74e-01 -3.11e-01  4.70e-01 -4.30e-01
```

最后进入 for 循环计算。

```cpp
  CUTE_NO_UNROLL // 在 blockK 方向上循环。
  while (k_tile_count > -(K_PIPE_MAX-1))
  {
    CUTE_UNROLL // 在 tiled k 方向上循环
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
      if (k_block == K_BLOCK_MAX - 1)
      {
        // Slice the smem_pipe_read smem
        tCsA_p = tCsA(_,_,_,smem_pipe_read);
        tCsB_p = tCsB(_,_,_,smem_pipe_read);

        // Commit the smem for smem_pipe_read
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();
      }

      // Load A, B shmem->regs for k_block+1
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;      // 拷贝 tiled k idx 的数据
      copy(tCsA_p(_,_,k_block_next), tCrA(_,_,k_block_next));
      copy(tCsB_p(_,_,k_block_next), tCrB(_,_,k_block_next));
      // Copy gmem to smem before computing gemm on each k-pipe
      if (k_block == 0)
      {
        // 启动下一个 buffer 的异步拷贝
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
        cp_async_fence();

        // Advance the gmem tile
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }

        // Advance the smem pipe
        smem_pipe_write = smem_pipe_read;
        ++smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;
      }
      // Thread-level register gemm for k_block
      gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
    }

  }
```

# HALF

当数据类型全是 half 时，sm80 的计算使用了 mma 和 cp.async。

```cpp
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                    Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // Thr layout 16x8 k-major
                                    Layout<Shape<_1, _8>>{});                 // Val layout  1x8 k-major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                    Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // Thr layout 16x8 k-major
                                    Layout<Shape<_1, _8>>{});                 // Val layout  1x8 n-major

  TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                                 Layout<Shape<_2, _2>>{}, // 2x2x1 MMA Atoms
                                 Tile<_32, _32, _16>{});  // 32x32x16 Tiled MMA for LDSM
```

上面 TiledCopy 使用了 cp.async，有 16*8 个线程，每个线程处理 8 个元素。

tiledmma 使用了 mma，atom 的大小是 16*8*16，所有 A 是 16*16，B 是 16*8，C 是 16*8。其中 Layout<Shape<_2, _2>>{}表示把 atom 在 M 和 N 方向分别复制 2 份，一个 atom 有 32 个线程，复制后共有 128 个线程，大小是 32*16*16。Tile<_32, _32, _16>{}是 tiledmma 的大小可以看到在 N 维度上的大小是线程数的一半，所以 N 维度上每个线程会多处理一份的数据。

此外，smem 也使用了 swizzle，为了简单说明，下面取消 swizzle。

从 smem 到 rmem 的拷贝也使用了 ldmatrix 指令。

如果不使用 ldmatrix 指令的话会根据 mma 的线程布局把每个线程对应的数据复制到线程的寄存器中。使用了 ldmatrix 的话会根据 ldmatrix 的布局把数据从 smem 拷贝到寄存器中。

```cpp
Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;
```

定义完成后进入 kernel

```cpp
((_128,_1),(_64,_1),(_1,_3)):((_64,_0),(_1,_0),(_0,_8192))
TiledCopy
  Tiler_MN:       (_16,_64)
  TiledLayout_TV: ((_8,_16),_8):((_128,_1),_16)
Copy_Atom
  ThrID:        _1:_0
  ValLayoutSrc: (_1,_8):(_0,_1)
  ValLayoutDst: (_1,_8):(_0,_1)
  ValLayoutRef: (_1,_8):(_0,_1)
  ValueType:    16b
TiledCopy
  Tiler_MN:       (_16,_64)
  TiledLayout_TV: ((_8,_16),_8):((_128,_1),_16)
Copy_Atom
  ThrID:        _1:_0
  ValLayoutSrc: (_1,_8):(_0,_1)
  ValLayoutDst: (_1,_8):(_0,_1)
  ValLayoutRef: (_1,_8):(_0,_1)
  ValueType:    16b
TiledMMA
  ThrLayoutVMNK:  (_32,_2,_2,_1):(_1,_32,_64,_0)
  PermutationMNK: (_32,_32,_16)
MMA_Atom
  ThrID:      _32:_1
  Shape_MNK:  (_16,_8,_16)
  LayoutA_TV: ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))
  LayoutB_TV: ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
  LayoutC_TV: ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
--t_tensor--((_32,_4),(_16,_4)):((_64,_2048),(_1,_16))
--a_tensor--((_16,_16),(_8,_4)):((_64,_1),(_1024,_16))
--tv_tensor--(((_4,_8),(_2,_2,_2)),(_8,_4)):(((_2,_64),(_1,_512,_8)),(_1024,_16))
--thr_tensor--(((_4,_8),(_2,_1)),((_2,_2,_2),(_4,_4))):(((_2,_64),(_1024,_0)),((_1,_512,_8),(_2048,_16)))
---
((_128,_1),(_64,_1)):((_64,_0),(_1,_0))
--t_tensor--((_32,_4),(_16,_4)):((_64,_2048),(_1,_16))
--a_tensor--((_16,_16),(_8,_4)):((_64,_1),(_1024,_16))
--tv_tensor--(((_4,_8),(_2,_2,_2)),(_8,_4)):(((_2,_64),(_1,_512,_8)),(_1024,_16))
--thr_tensor--(((_4,_8),(_2,_1)),((_2,_2,_2),(_4,_4))):(((_2,_64),(_1024,_0)),((_1,_512,_8),(_2048,_16)))
(((_4,_8),(_2,_1)),((_2,_2,_2),(_4,_4))):(((_2,_64),(_1024,_0)),((_1,_512,_8),(_2048,_16)))
--t_tensor--((_32,_1),(_16,_1)):((_1,_0),(_32,_0))
--a_tensor--((_16,_16),(_2,_1)):((_1,_32),(_16,_0))
--tv_tensor--(((_4,_8),(_2,_2,_2)),(_2,_1)):(((_64,_1),(_32,_8,_256)),(_16,_0))
--thr_tensor--(((_4,_8),(_2,_1)),((_2,_2,_2),(_1,_1))):(((_64,_1),(_16,_0)),((_32,_8,_256),(_0,_0)))
  mA : gmem_ptr[16b](0x7f9c70000000) o (5120,4096):(4096,_1)
  gA : gmem_ptr[16b](0x7f9c70000000) o (_128,_64,64):(4096,_1,_64)
  sA : smem_ptr[16b](0x7f9d00000400) o ((_128,_1),(_64,_1),(_1,_3)):((_64,_0),(_1,_0),(_0,_8192))
tAgA : gmem_ptr[16b](0x7f9c70000000) o ((_8,_1),_8,_1,64):((_1,_0),65536,_0,_64)
tAsA : smem_ptr[16b](0x7f9d00000400) o ((_8,_1),_8,_1,(_1,_3)):((_1,_0),_1024,_0,(_0,_8192))

  mB : gmem_ptr[16b](0x7f9c6c000000) o (5120,4096):(4096,_1)
  gB : gmem_ptr[16b](0x7f9c6c000000) o (_128,_64,64):(4096,_1,_64)
  sB : smem_ptr[16b](0x7f9d0000c400) o ((_128,_1),(_64,_1),(_1,_3)):((_64,_0),(_1,_0),(_0,_8192))
tBgB : gmem_ptr[16b](0x7f9c6c000000) o ((_8,_1),_8,_1,64):((_1,_0),65536,_0,_64)
tBsB : smem_ptr[16b](0x7f9d0000c400) o ((_8,_1),_8,_1,(_1,_3)):((_1,_0),_1024,_0,(_0,_8192))

  mC : gmem_ptr[16b](0x7f9c68000000) o (5120,5120):(_1,5120)
  gC : gmem_ptr[16b](0x7f9c68000000) o (_128,_128):(_1,5120)
tCgC : gmem_ptr[16b](0x7f9c68000000) o ((_2,_2),_4,(_2,_4)):((5120,_8),_32,(81920,163840))
tCsA : smem_ptr[16b](0x7f9d00000400) o ((_2,_2,_2),_4,_4):((_1,_512,_8),_2048,_16)
tCsB : smem_ptr[16b](0x7f9d0000c400) o ((_2,_2),_8,_4):((_1,_8),_1024,_16)
tCrA : ptr[16b](0x7f9cbdfff9d0) o ((_2,_2,_2),_4,_4):((_1,_2,_4),_32,_8)
tCrB : ptr[16b](0x7f9cbdfffad0) o ((_2,_2),_8,_4):((_1,_2),_16,_4)
tCrC : ptr[16b](0x7f9cbdfffbd0) o ((_2,_2),_4,(_2,_4)):((_1,_2),_4,(_16,_32))

s2r_copy_a : TiledCopy
  Tiler_MN:       (_32,_16)
  TiledLayout_TV: ((_4,_8,_2,_2),((_2,_2,_2),(_1,_1))):((_64,_1,_16,_0),((_32,_8,_256),(_0,_0)))
Copy_Atom
  ThrID:        _32:_1
  ValLayoutSrc: (_32,_8):(_8,_1)
  ValLayoutDst: (_32,(_2,_4)):(_2,(_1,_64))
  ValLayoutRef: (_32,(_2,_4)):(_2,(_1,_64))
  ValueType:    16b

s2r_thr_copy_a : ThrCopy
  ThrIdx: 0
TiledCopy
  Tiler_MN:       (_32,_16)
  TiledLayout_TV: ((_4,_8,_2,_2),((_2,_2,_2),(_1,_1))):((_64,_1,_16,_0),((_32,_8,_256),(_0,_0)))
Copy_Atom
  ThrID:        _32:_1
  ValLayoutSrc: (_32,_8):(_8,_1)
  ValLayoutDst: (_32,(_2,_4)):(_2,(_1,_64))
  ValLayoutRef: (_32,(_2,_4)):(_2,(_1,_64))
  ValueType:    16b

tXsA : smem_ptr[16b](0x7f9d00000400) o ((_8,_1),_4,_4,(_1,_3)):((_1,_0),_2048,_16,(_0,_8192))
tXrA : ptr[16b](0x7f9cbdfff9d0) o ((_8,_1),_4,_4):((_1,_0),_32,_8)

s2r_copy_b : TiledCopy
  Tiler_MN:       (_32,_16)
  TiledLayout_TV: ((_4,_8,_2,_2),((_2,_2),(_2,_1))):((_64,_1,_0,_8),((_32,_256),(_16,_0)))
Copy_Atom
  ThrID:        _32:_1
  ValLayoutSrc: (_32,_8):(_8,_1)
  ValLayoutDst: (_32,(_2,_4)):(_2,(_1,_64))
  ValLayoutRef: (_32,(_2,_4)):(_2,(_1,_64))
  ValueType:    16b

s2r_thr_copy_b : ThrCopy
  ThrIdx: 0
TiledCopy
  Tiler_MN:       (_32,_16)
  TiledLayout_TV: ((_4,_8,_2,_2),((_2,_2),(_2,_1))):((_64,_1,_0,_8),((_32,_256),(_16,_0)))
Copy_Atom
  ThrID:        _32:_1
  ValLayoutSrc: (_32,_8):(_8,_1)
  ValLayoutDst: (_32,(_2,_4)):(_2,(_1,_64))
  ValLayoutRef: (_32,(_2,_4)):(_2,(_1,_64))
  ValueType:    16b

tXsB : smem_ptr[16b](0x7f9d0000c400) o ((_8,_1),_4,_4,(_1,_3)):((_1,_0),_2048,_16,(_0,_8192))
tXrB : ptr[16b](0x7f9cbdfffad0) o (((_4,_2),_1),_4,_4):(((_1,_16),_0),_32,_4)

```

进入 kernel 后，分别对 A，B，C 进行分块，得到

```cpp
gA : gmem_ptr[16b](0x7f9c70000000) o (_128,_64,64):(4096,_1,_64)
gB : gmem_ptr[16b](0x7f9c6c000000) o (_128,_64,64):(4096,_1,_64)
gC : gmem_ptr[16b](0x7f9c68000000) o (_128,_128):(_1,5120)
```

以 gA 为例，因为 A 的大小是 5120*4096，一个 block 的大小是 128*64，所以分块后 gA 的大小是 128*64*64。

同样的 sA 的大小也是 128*64*3

确定 gmem 和 smem 的大小后，使用 tiledcopy 对 gmem 和 smem 进行 tiling，根据 tiledcopy 的布局获取每个线程在矩阵中对应的数据。

从前面可以知道，tiledcopy 使用的是 cp.async，一共有 16*8 = 128 个线程，一个线程对应 8 个元素。所以一个线程在 gmem 的一个 block 中需要处理 8*8 个元素，所以

```cpp
tAgA : gmem_ptr[16b](0x7f9c70000000) o ((_8,_1),_8,_1,64):((_1,_0),65536,_0,_64)
```

第一个维度的 8 是连续的 8 个元素，所以 stride=1，第二个 8 相隔 16 行，所以 stride=16*4096=65536。64 表示不同的 block，stride=64.

同样的，一个线程在 smem 中也要处理 8*8 个元素。

```cpp
tAsA : smem_ptr[16b](0x7f9d00000400) o ((_8,_1),_8,_1,(_1,_3)):((_1,_0),_1024,_0,(_0,_8192))
```

copy 的线程确定完成后需要确定 mma 的线程。

从前面的定义可以知道，mma 使用的是 16*8*16 的 mma 指令，一个 mma 指令有 32 个线程，tiledmma 中复制了 4 份，所以 tiledmma 的线程数是 128，大小是 32*16*16，又因为 tiledmma 的大小是 32*32*16，所以在 N 方向上一个线程需要处理 2 倍的数据。

如下图所示

![](/assets/gemm_sm80/image(1).png)

在 M 方向上大小是 32，有 2 个 mma，0-31 个线程处理前 16 行，32-63 处理后 16 行。在 N 方向上大小是 32，有 4 个 mma，其中 0-31 个线程处理第一个和第三个 8 列，32-64 处理第二个和第 4 个 8 列。

因此 C 需要 2 行 4 列个 mma 计算。如果按行优先顺序的话，第 0 个 mma 需要 0-31 加载 A 中的数据 0-31 加载 B 中的数据。第 1 个 mma 需要 64-95 个线程加载 A 中的数据，64-95 个线程加载 B 中的数据。第 2 个 mma 需要 0-31 加载 A 中和第 0 个 mma 相同的数据，需要 0-31 加载 B 中的第二份数据。

每个线程对应的数据是怎么确定的。比如 N 方向上没有扩展的时候，一共有 128 个线程，每 32 个线程负责不同的数据。比如线程 0 在 partitionA 后需要加载 8 个数据，partitionB 后需要加载 4 个数据，partitionC 后需要处理 4 个数据。这些数据通过 mma 计算。

在 N 方向拓展后，线程 0 在 partitionA 后需要加载 8 个数据，partitionB 后需要加载 4*2=8 个数据，partitionC 后需要处理 4*2=8 个数据。这些数据是怎么通过 mma 计算的呢。

partition 后每个线程对应的数据分别是：

```cpp
tCgC : gmem_ptr[16b](0x7f9c68000000) o ((_2,_2),_4,(_2,_4)):((5120,_8),_32,(81920,163840))
tCsA : smem_ptr[16b](0x7f9d00000400) o ((_2,_2,_2),_4,_4):((_1,_512,_8),_2048,_16)
tCsB : smem_ptr[16b](0x7f9d0000c400) o ((_2,_2),_8,_4):((_1,_8),_1024,_16)
```

tCgC 中，一个线程在一个 mma 中需要处理 4 个数据，所以是(_2,_2)，在 smem 的 M 维度有 4 个 tiledmma，所以是 4，在 N 方向上，一个 tiledmma 中有 2 个线程 0 负责的 mma，smem 中有 4 个 tiledmma，所以是(_2,_4)。

tCsA 中，一个线程在一个 mma 中需要处理 8 个数据，所以是(_2,_2,_2)，M 和 K 方向分别有 4 个，所以是((_2,_2,_2),_4,_4)。

tCsB 中，一个线程在一个 mma 中需要处理 4 个数据，所以是(_2,_2)，N 方向上有 2*4=8 个，K 方向上有 4 个，所以是((_2,_2),_8,_4)。

在寄存器空间申请的 tensor 为：

```cpp
tCrA : ptr[16b](0x7f9cbdfff9d0) o ((_2,_2,_2),_4,_4):((_1,_2,_4),_32,_8)
tCrB : ptr[16b](0x7f9cbdfffad0) o ((_2,_2),_8,_4):((_1,_2),_16,_4)
tCrC : ptr[16b](0x7f9cbdfffbd0) o ((_2,_2),_4,(_2,_4)):((_1,_2),_4,(_16,_32))
```

矩阵 A 和 B 的 smem tensor tCsA tCsB 和 rmem tensor tCrA tCrB 定义完成后，可以使用 ldmatrix 将数据从 smem 加载到 rmem

```cpp
  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA); // (CPY,MMA_M,MMA_K,PIPE)
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);  // (CPY,MMA_M,MMA_K)

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB); // (CPY,MMA_N,MMA_K,PIPE)
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);  // (CPY,MMA_N,MMA_K)
```

使用 ldmatrix 进行加载的时候也需要根据 ldmatrix 的线程 layout 对两个矩阵进行分块。

首先创建 ldmatrix 对应的 tiledcopy s2r_copy_a。这里 atom 使用的是 s2r_atom_a。

```cpp
Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A
```

一个 atom 有 32 个线程参与，加载 16*16 个 half 数据。32 个线程每个线程读取一行，然后按照下面的 layout 分发到每个线程中。

![](/assets/gemm_sm80/image(2).png)

因为要加载 mma 对应的 tensor，所以需要根据 mma 的形状和线程数创建 tiledcopy。创建结果如下。

```cpp
TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);

s2r_thr_copy_a : ThrCopy
  ThrIdx: 0
TiledCopy
  Tiler_MN:       (_32,_16)
  TiledLayout_TV: ((_4,_8,_2,_2),((_2,_2,_2),(_1,_1))):((_64,_1,_16,_0),((_32,_8,_256),(_0,_0)))
Copy_Atom
  ThrID:        _32:_1
  ValLayoutSrc: (_32,_8):(_8,_1)
  ValLayoutDst: (_32,(_2,_4)):(_2,(_1,_64))
  ValLayoutRef: (_32,(_2,_4)):(_2,(_1,_64))
  ValueType:    16b
```

因为 mma 对应的 A 的大小是 32*16，所以创建的 tiledcopy a 的大小也是 32*16。mma 中一共有 128 个线程，所以需要 4 个 ldmatrix atom，因为一个 ldmatrix 的大小是 16*16，所以 4 个就是 32*32，这也是为什么 tiledmma 的 N 方向需要是 32 的原因。

然后通过 partition S 对 smem 的 tensor 按照线程进行分块并得到当前线程对应的数据的 layout。通过 retile_D 使用 ldmatrix 的线程布局重新规划 rmem tensor 上每个线程对应的元素。为啥 smem 使用 partition，rmem 使用 retile 呢。

```cpp
tXsA : smem_ptr[16b](0x7f9d00000400) o ((_8,_1),_4,_4,(_1,_3)):((_1,_0),_2048,_16,(_0,_8192))
tXrA : ptr[16b](0x7f9cbdfff9d0) o ((_8,_1),_4,_4):((_1,_0),_32,_8)
```

tXsA 是 ldmatrix 对 smem 划分的结果，(_8,_1)表示每个 atom 中一个线程在 smem 中加载一行 8 个元素，因为 smem 的大小是 128*64，一个 tiledcopy 有 128 个线程，大小是 32*16，所以在 MN 方向上分别要加载 4 份，再加上 pipe stage = 3，所以一个线程就需要加载 smem 中的((_8,_1),_4,_4,(_1,_3))个元素。对于 stride，因为一个线程加载 8 个连续的元素，所以(_8,_1)对应的 stride=1，因为数据是 k-major，所以在行方向上，每个线程相隔 16，在列方向上，每个线程相隔 32*64=2048。

tXrA 是 ldmatrix 对 rmem 划分的结果，也是一个线程加载 8 个元素，tCrA 的大小是 8*16，layout 是((_2,_2,_2),_4,_4):((_1,_2,_4),_32,_8)，为啥 retile 后变成了((_8,_1),_4,_4):((_1,_0),_32,_8)。算了，只要每个线程的数据是对的就行。

同理对于矩阵 B

```cpp
tXsB : smem_ptr[16b](0x7f9d0000c400) o ((_8,_1),_4,_4,(_1,_3)):((_1,_0),_2048,_16,(_0,_8192))
tXrB : ptr[16b](0x7f9cbdfffad0) o (((_4,_2),_1),_4,_4):(((_1,_16),_0),_32,_4)
```

使用 ldmatrix 的时候每个线程到底对应哪些数据。

print_latex(s2r_copy_a);

% Layout S TV: ArithTuple(_0,_0) o ((_16,_2,_2,_2),(_8,_1)):((_1@0,_8@1,_16@0,_0),(_1@1,_0))

% Layout D TV: ArithTuple(_0,_0) o ((_4,_8,_2,_2),((_2,_2,_2),_1)):((_2@1,_1@0,_16@0,_0),((_1@1,_8@0,_8@1),_0))

![](/assets/gemm_sm80/image(3).png)

print_latex(s2r_copy_b);

% Layout S TV: ArithTuple(_0,_0) o ((_8,_2,_2,_2,_2),(_8,_1)):((_1@0,_8@1,_16@0,_0,_8@0),(_1@1,_0))

% Layout D TV: ArithTuple(_0,_0) o ((_4,_8,_2,_2),((_2,_2,_2),_1)):((_2@1,_1@0,_0,_8@0),((_1@1,_8@1,_16@0),_0))

下面进行计算。

在主循环开始前，已经有 2 次异步拷贝开始运行了，把数据从 gmem 拷贝到 smem 中的前 2 个 stage。

```cpp
  // Start async loads for all pipes but the last
  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe)
  {
    copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
    copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0)
    {
      ++k_tile_next;
    }
  }
```

然后使用 ldmatrix 把数据从 smem 中拷贝到 rmem 中。K_BLOCK_MAX 在这里等于 4，是 K 维度上的 block 数。cp_async_wait<K_PIPE_MAX - 2>();等价于 cp_async_wait<1>();意思是允许最近的一个异步拷贝没有完成，也就是第一个异步拷贝必须完成。

确保第一个异步拷贝已经把数据从 gmem 拷贝到 smem 中后，就可以使用 ldmatrix 把第一个 pipe 的数据从 smem 拷贝到 rmem 中了。这里等于是使用 4 个 ldmatrix 拷贝第一个 k block 中的元素。

```cpp
  // Current pipe index in smem to read from
  int smem_pipe_read = 0;
  // Current pipe index in smem to write to
  int smem_pipe_write = K_PIPE_MAX - 1;

  // Pipe slice
  Tensor tXsA_p = tXsA(_, _, _, smem_pipe_read);
  Tensor tXsB_p = tXsB(_, _, _, smem_pipe_read);

  // Size of the register pipeline
  auto K_BLOCK_MAX = size<2>(tCrA);
  CUTE_STATIC_ASSERT_V(K_BLOCK_MAX == size<2>(tXrA));

  // PREFETCH register pipeline
  if (K_BLOCK_MAX > 1)
  {
    // Wait until our first prefetched tile is loaded in
    cp_async_wait<K_PIPE_MAX - 2>();
    __syncthreads();

    // Prefetch the first rmem from the first k-tile
    copy(s2r_atom_a, tXsA_p(_, _, Int<0>{}), tXrA(_, _, Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_, _, Int<0>{}), tXrB(_, _, Int<0>{}));
  }
```

然后进入主循环。先对一个 block 中的 K 进行循环。当 k_block == 0 时，使用 ldmatrix 加载下一个 k tile 中的元素到 rmem 中，然后执行新的异步拷贝，拷贝数据到 smem 的第三个 pipe stage 中。然后使用 gemm 计算第一个 k tile 中的数据。

后面就是在 k tile 上的循环加载数据到 rmem，然后计算的过程了。

当一个 k block 中的所有 k tile 计算完成后，使用 ldmatrix 加载 smem 的下一个 pipe stage 的数据，并确保改 stage 的数据已经从 gmem 中加载完成。

```cpp
  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE_MAX - 1))
  {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
      if (k_block == K_BLOCK_MAX - 1)
      {
        // Slice the smem_pipe_read smem
        tXsA_p = tXsA(_, _, _, smem_pipe_read);
        tXsB_p = tXsB(_, _, _, smem_pipe_read);

        // Commit the smem for smem_pipe_read
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();
      }

      // Load A, B shmem->regs for k_block+1
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX; // static
      copy(s2r_atom_a, tXsA_p(_, _, k_block_next), tXrA(_, _, k_block_next));
      copy(s2r_atom_b, tXsB_p(_, _, k_block_next), tXrB(_, _, k_block_next));
      // Copy gmem to smem before computing gemm on each k-pipe
      if (k_block == 0)
      {
        copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_pipe_write));
        copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_pipe_write));
        cp_async_fence();

        // Advance the gmem tile
        --k_tile_count;
        if (k_tile_count > 0)
        {
          ++k_tile_next;
        }

        // Advance the smem pipe
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;
      }
      // Thread-level register gemm for k_block
      gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
    }
  }
```

在 k tile 方向上进行循环累加。在 k block 方向上循环展开。

如果上一个 pipe stage 计算完成，等待上一个异步拷贝完成，然后使用 ldmatrix 拷贝数据到寄存器中，使用 mma 计算矩阵乘。同时启动下一个异步拷贝。
