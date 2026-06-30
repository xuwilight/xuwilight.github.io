---
title: CuTe 学习笔记（十三）GEMM SM90 TMA
date: 2025-04-25 18:00:00
tags: [CUTLASS, GEMM, TMA, SM90, GPU]
categories: [Cutlass 学习笔记]
description: sm90_tma 是在 sm90 上的拓展，通过使用 tma 进行异步数据拷贝，并结合 wgmma 来进行异步计算。 初始化没什么好说的 矩阵乘的规模 M = N = 5120，K = 4096。 数据类型全是 half，矩阵 A 选择 row-major，矩阵 B 选择 column-major。
---

sm90_tma 是在 sm90 上的拓展，通过使用 tma 进行异步数据拷贝，并结合 wgmma 来进行异步计算。

初始化没什么好说的

矩阵乘的规模 M = N = 5120，K = 4096。

数据类型全是 half，矩阵 A 选择 row-major，矩阵 B 选择 column-major。

直接进入 gemm_tn。

## gemm_tn

首先定义矩阵乘的大小和每个 block 的大小。这里 smem 的大小是 128*128*64。pipe stage = 3。

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
  auto bK = Int< 64>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline
```

然后定义 smem 的 layout，tiledmma 和 tma copy。

```cpp
  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  // Define the MMA
  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{});

  // Define the TMAs
  // Create Global memory tensors for TMA inspection
  Tensor mA = make_tensor(A, make_shape(M,K), dA);
  Tensor mB = make_tensor(B, make_shape(N,K), dB);

  // Create TMA Atoms with the desired copy operation on the source and destination
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));
```

这里 smem 的 layout 使用了 128B swizzle，防止出现 bank conflict。wgmma 使用了 SM90_64x64x16_F16F16F16_SS。说明矩阵 A 和矩阵 B 的大小分别是 64*16，64*16，且都是从 smem 中加载。

tiled_mma 的结果如下，ThrLayoutVMNK 是 tiledmma 的 tv-layout，因为一个 wgmma 有 128 个线程，而且 tiledmma 也没有进行拓展，所以这里 V 等于 128，MNK 都等于 1。PermutationMNK 是 tiledmma 实际的大小，因为没有设置，所以 tiledmma 实际的大小就等于使用的 wgmma 的大小。

再看 MMA atom。ThrID 表示一共有 128 个线程，Shape_MNK 表示这个 wgmma 指令计算的大小。LayoutA_TV 代表指令在矩阵 A 上的 tv-layout，因为是从 smem 中加载的，所以 128 个线程每个线程都能看到全部的数据。

```cpp
tiled_mma:TiledMMA
  ThrLayoutVMNK:  (_128,_1,_1,_1):(_1,_0,_0,_0)
  PermutationMNK: (_,_,_)
MMA_Atom
  ThrID:      _128:_1
  Shape_MNK:  (_64,_64,_16)
  LayoutA_TV: (_128,(_64,_16)):(_0,(_1,_64))
  LayoutB_TV: (_128,(_64,_16)):(_0,(_1,_64))
  LayoutC_TV: ((_4,_8,_4),(_2,_2,_8)):((_128,_1,_16),(_64,_8,_512))
```

重点是 tma 的设置。

tma 相关的代码都在 cutlass-4.1/include/cute/atom/copy_traits_sm90_tma.hpp 中。

介绍 tma 前先介绍下 tma tensor。

tma 的 ptx 指令如下：

```cpp
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%3, %4}], [%2];" ::"r"(smem_int_ptr),
                     "l"(&src_tensor_map), "r"(smem_int_mbar), "r"(crd0), "r"(crd1) : "memory");
```

可以看到，tma 指令需要一个 tma 描述符 src_tensor_map，这个描述符里保存有 gmem tensor 的地址，类型等信息。然后还需要一个坐标来确定需要加载的 tensor 在整个 tensor 中的位置。

所以 tma 指令不会直接使用 gmem 的地址，而是需要 gmem 的元素的坐标。

因此 cute 中通过坐标来创建 tma tensor。这个 tensor 可以使用所有的 layout 运算，从而方便的得出每个元素的具体坐标。

具体创建方法如下：

make_inttuple_iter 是用来创建一个元组迭代器。shape 是（4，5）。stride 是(E<0>{}, E<1>{}))。

通过普通 tensor 的 stride (a,b)可以把逻辑坐标(i,j)转换为一维线性 index = i * a + j * b。同样的，对于 tma tensor，也可以通过 stride 把逻辑坐标(i,j)转换为 TMA 的坐标。

```cpp
Tensor a = make_tensor(make_inttuple_iter(0,0),
                       make_shape (     4,      5),
                       make_stride(E<0>{}, E<1>{}));
print_tensor(a);

Tensor b = make_tensor(make_inttuple_iter(0,0),
                       make_shape (     4,      5),
                       make_stride(E<1>{}, E<0>{}));
print_tensor(b);
```

C++ objectDescriptionString representationE<>{}11E<0>{}(1,0,...)1@0E<1>{}(0,1,0,...)1@1E<0,0>{}((1,0,...),0,...)1@0@0E<0,1>{}((0,1,0,...),0,...)1@1@0E<1,0>{}(0,(1,0,...),0,...)1@0@1E<1,1>{}(0,(0,1,0,...),0,...)1@1@1

这里 E<0>表示在第 0 个位置是 1，E<1>表示在第 1 个位置是 1，也可以分别用 1@0 和 1@1 表示。

计算时，5*E<1>{} = 5*(1@1) = 5@1 = (0,5,0,....)

3*E<0>{} + 4*E<1>{} = 3*(1@0) + 4*(1@1) = 3@0 + 4@1 = (3,4)。也就是第 0 个位置是 3，第 1 个位置是 4。

假如 stride = (1@0, 1@1)，坐标是(i,j)，计算公式为：i@0 + j@1 = (i,j)，就是 TMA 元素对应的坐标。

上述代码的结果是：

```cpp
ArithTuple(0,0) o (4,5):(_1@0,_1@1):
  (0,0)  (0,1)  (0,2)  (0,3)  (0,4)
  (1,0)  (1,1)  (1,2)  (1,3)  (1,4)
  (2,0)  (2,1)  (2,2)  (2,3)  (2,4)
  (3,0)  (3,1)  (3,2)  (3,3)  (3,4)

ArithTuple(0,0) o (4,5):(_1@1,_1@0):
  (0,0)  (1,0)  (2,0)  (3,0)  (4,0)
  (0,1)  (1,1)  (2,1)  (3,1)  (4,1)
  (0,2)  (1,2)  (2,2)  (3,2)  (4,2)
  (0,3)  (1,3)  (2,3)  (3,3)  (4,3)
```

因为在创建 tma 的描述符时，坐标是从 stride=1 的维度开始的。所以当矩阵是 column-major 时，逻辑坐标（i,j）的顺序和（cord0,cord1）的顺序相同，也就是 i 行 j 列位置的坐标正好也是 tma 需要的坐标（i,j），因此，column-major 时，stride = (_1@0,_1@1)。

当矩阵时 row-major 时，cord0 对应 j，cord1 对应 i。所以 i 行 j 列对应的 tma 坐标实际是（j，i），因此 stride = (_1@1,_1@0)。

下面的代码通过 make_tma_atom 使用 SM90_TMA_LOAD op 创建了一个 tma 类型的 copy atom。

```cpp
  // Create TMA Atoms with the desired copy operation on the source and destination
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));
```

打印创建的结果可以看到，ThrID = 1 是因为 tma 只需要一个线程启动。ValLayoutSrc = 8192 表示一个 tma 就可以拷贝 smem 128*64=8192 的数据量。

```cpp
tmaA:Copy_Atom
  ThrID:        _1:_0
  ValLayoutSrc: (_1,_8192):(_0,_1)
  ValLayoutDst: (_1,_8192):(_0,_1)
  ValLayoutRef: (_1,_8192):(_0,_1)
  ValueType:    16b

tmaB:Copy_Atom
  ThrID:        _1:_0
  ValLayoutSrc: (_1,_8192):(_0,_1)
  ValLayoutDst: (_1,_8192):(_0,_1)
  ValLayoutRef: (_1,_8192):(_0,_1)
  ValueType:    16b
```

之后进入 kernel。这里 dimCluster(2, 1, 1)应该是使用了 smem cluster。

下面是 smem 相关的存储结构体。

```cpp
template <class ElementA,
          class ElementB,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB>  // (N,K,P)
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;

  uint64_t tma_barrier[size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};
```

使用 tma 需要数据必须 16bytes 对齐，tma_barrier 和 mma_barrier 用于异步操作之间的同步，都是 64 位变量。

下面是定义 cute 类型 tensor 的代码，跟之前没什么区别。不过 tma 下是使用 tma_a.get_tma_tensor 获取 gmem 上的 tensor 的。

```cpp
  // Represent the full tensors
  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));                   // (M,K) TMA Tensor
  Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));                   // (N,K) TMA Tensor
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);      // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory tensors
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
```

通过 get_tma_tensor 获取 mA 和 mB。返回的是 TMA 坐标 tensor。shape 是(128,64)，stride = (_1@1,_1@0)。

```cpp
  // Generate the TMA coord tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    if (thread0()) {print(g_shape);print("\n");print(aux_params_.g_stride_);print("\n");}
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }
```

其中 mA 和 mB 是类似下面这种的 tensor。列主序是下面这样，行主序是反的。

```cpp
ArithTuple(_0,_0) o (_4,_8):(_1@0,_1@1):
  (0,0)  (0,1)  (0,2)  (0,3)  (0,4)  (0,5)  (0,6)  (0,7)
  (1,0)  (1,1)  (1,2)  (1,3)  (1,4)  (1,5)  (1,6)  (1,7)
  (2,0)  (2,1)  (2,2)  (2,3)  (2,4)  (2,5)  (2,6)  (2,7)
  (3,0)  (3,1)  (3,2)  (3,3)  (3,4)  (3,5)  (3,6)  (3,7)
```

K-major

```cpp
  (0,0)  (1,0)  (2,0)  (3,0)  (4,0)  (5,0)  (6,0)  (7,0)  (8,0)  (9,0)  (10,0)  (11,0)  (12,0)  (13,0)  (14,0)  (15,0)  (16,0)  (17,0)  (18,0)  (19,0)  (20,0)  (21,0)  (22,0)  (23,0)  (24,0)  (25,0)  (26,0)  (27,0)  (28,0)  (29,0)  (30,0)  (31,0)  (32,0)  (33,0)  (34,0)  (35,0)  (36,0)  (37,0)  (38,0)  (39,0)  (40,0)  (41,0)  (42,0)  (43,0)  (44,0)  (45,0)  (46,0)  (47,0)  (48,0)  (49,0)  (50,0)  (51,0)  (52,0)  (53,0)  (54,0)  (55,0)  (56,0)  (57,0)  (58,0)  (59,0)  (60,0)  (61,0)  (62,0)  (63,0)
  (0,1)  (1,1)  (2,1)  (3,1)  (4,1)  (5,1)  (6,1)  (7,1)  (8,1)  (9,1)  (10,1)  (11,1)  (12,1)  (13,1)  (14,1)  (15,1)  (16,1)  (17,1)  (18,1)  (19,1)  (20,1)  (21,1)  (22,1)  (23,1)  (24,1)  (25,1)  (26,1)  (27,1)  (28,1)  (29,1)  (30,1)  (31,1)  (32,1)  (33,1)  (34,1)  (35,1)  (36,1)  (37,1)  (38,1)  (39,1)  (40,1)  (41,1)  (42,1)  (43,1)  (44,1)  (45,1)  (46,1)  (47,1)  (48,1)  (49,1)  (50,1)  (51,1)  (52,1)  (53,1)  (54,1)  (55,1)  (56,1)  (57,1)  (58,1)  (59,1)  (60,1)  (61,1)  (62,1)  (63,1)
  (0,2)  (1,2)  (2,2)  (3,2)  (4,2)  (5,2)  (6,2)  (7,2)  (8,2)  (9,2)  (10,2)  (11,2)  (12,2)  (13,2)  (14,2)  (15,2)  (16,2)  (17,2)  (18,2)  (19,2)  (20,2)  (21,2)  (22,2)  (23,2)  (24,2)  (25,2)  (26,2)  (27,2)  (28,2)  (29,2)  (30,2)  (31,2)  (32,2)  (33,2)  (34,2)  (35,2)  (36,2)  (37,2)  (38,2)  (39,2)  (40,2)  (41,2)  (42,2)  (43,2)  (44,2)  (45,2)  (46,2)  (47,2)  (48,2)  (49,2)  (50,2)  (51,2)  (52,2)  (53,2)  (54,2)  (55,2)  (56,2)  (57,2)  (58,2)  (59,2)  (60,2)  (61,2)  (62,2)  (63,2)
  (0,3)  (1,3)  (2,3)  (3,3)  (4,3)  (5,3)  (6,3)  (7,3)  (8,3)  (9,3)  (10,3)  (11,3)  (12,3)  (13,3)  (14,3)  (15,3)  (16,3)  (17,3)  (18,3)  (19,3)  (20,3)  (21,3)  (22,3)  (23,3)  (24,3)  (25,3)  (26,3)  (27,3)  (28,3)  (29,3)  (30,3)  (31,3)  (32,3)  (33,3)  (34,3)  (35,3)  (36,3)  (37,3)  (38,3)  (39,3)  (40,3)  (41,3)  (42,3)  (43,3)  (44,3)  (45,3)  (46,3)  (47,3)  (48,3)  (49,3)  (50,3)  (51,3)  (52,3)  (53,3)  (54,3)  (55,3)  (56,3)  (57,3)  (58,3)  (59,3)  (60,3)  (61,3)  (62,3)  (63,3)
  (0,4)  (1,4)  (2,4)  (3,4)  (4,4)  (5,4)  (6,4)  (7,4)  (8,4)  (9,4)  (10,4)  (11,4)  (12,4)  (13,4)  (14,4)  (15,4)  (16,4)  (17,4)  (18,4)  (19,4)  (20,4)  (21,4)  (22,4)  (23,4)  (24,4)  (25,4)  (26,4)  (27,4)  (28,4)  (29,4)  (30,4)  (31,4)  (32,4)  (33,4)  (34,4)  (35,4)  (36,4)  (37,4)  (38,4)  (39,4)  (40,4)  (41,4)  (42,4)  (43,4)  (44,4)  (45,4)  (46,4)  (47,4)  (48,4)  (49,4)  (50,4)  (51,4)  (52,4)  (53,4)  (54,4)  (55,4)  (56,4)  (57,4)  (58,4)  (59,4)  (60,4)  (61,4)  (62,4)  (63,4)
  
```

然后是 tma_partition。入参分别是 copy_atom，也就是 tma 的 atom。cta_coord 和 cta_layout，multicast 时有用，如果不使用 multicast 默认是 0 和 1。然后是 smem tensor 和 gmem tensor。这里 smem 是正常的 tensor，gmem 还是坐标 tensor。

```cpp
// The "VectorCopy Partitioner" for TMA
template <class... Args,
          class CtaCoord,
          class TShape, class TStride,
          class SEngine, class SLayout,
          class GEngine, class GLayout>
CUTE_DEVICE
auto
tma_partition(Copy_Atom<Args...>      const& copy_atom,
              CtaCoord                const& cta_coord,
              Layout<TShape,TStride>  const& cta_layout,  // T: CTA coord -> logical multicast id
              Tensor<SEngine,SLayout> const& stensor,     // SMEM Tensor (TMATile, Rest...)
              Tensor<GEngine,GLayout> const& gtensor)     // GMEM Tensor (TMATile, Rest...)
{
  CUTE_STATIC_ASSERT_V(size<0>(stensor) == size<0>(gtensor));

  // Invert the smem to get the largest contiguous vector in the smem layout
  Layout inv_smem_layout = right_inverse(get_nonswizzle_portion(layout<0>(stensor)));
  // Scale that up to cover all of the smem_coords
  Layout layout_v = tile_to_shape(make_layout(inv_smem_layout), size<0>(stensor));

  // Factor out the single-instrucion portion
  Layout tma_layout_v = make_layout(Int<Copy_Atom<Args...>::NumValSrc>{});
  auto layout_V = make_tile(logical_divide(layout_v, tma_layout_v));

  // Append with _ until we cover all Rest... modes
  auto glayout_V = append<GLayout::rank>(layout_V, _);
  auto slayout_V = append<SLayout::rank>(layout_V, _);
  // Transform tile mode and coalesce
  Tensor gtensor_v = coalesce(gtensor.compose(glayout_V), Shape<Shape<_1,_1>>{});    // ((TMA,TMA_Iter), Rest...)
  Tensor stensor_v = coalesce(stensor.compose(slayout_V), Shape<Shape<_1,_1>>{});    // ((TMA,TMA_Iter), Rest...)
  // Offset inside the TMA-mode for the multicast
  auto multicast_offset = cta_layout(cta_coord) * (size(tma_layout_v) / cosize(cta_layout));
  auto multicast_coord  = make_coord(make_coord(multicast_offset, Int<0>{}));
  auto gcoord = append<GLayout::rank>(multicast_coord, Int<0>{});
  auto scoord = append<SLayout::rank>(multicast_coord, Int<0>{});

  Tensor gresult = domain_offset(gcoord, gtensor_v);
  Tensor sresult = domain_offset(scoord, stensor_v);

  return cute::make_tuple(gresult, sresult);
}
```

经过一系列复杂的 shape 运算，得到线程在 gmem 和 smem 上的分区结果。

tAgA:ArithTuple(_0,0) o (((_64,_128),_1),1):(((_1@0,_1@1),_0),_64@0)

tAsA:Sw<3,4,3>_smem_ptr[16b](https://0x7f7700000400) o ((_8192,_1),(_1,_3)):((_1,_0),(_0,_8192))

反正就是 tma 对应 gmem 上(_64,_128)大小的数据。smem 上_8192 的数据。

然后设置 tma_transaction_bytes，说明 tma 异步传输时需要传输多少的数据量。这里的数据量时 smemA 和 smemB 的总和。

```cpp
 constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                      + sizeof(make_tensor_like(tensor<0>(tBsB)));
```

下面开始设置 mbarrier。设置 mbarrier 时只需要一个线程就可以了。这里设置了两个 barrier，分别是 producer_mbar 和 consumer_mbar 。

```cpp
  auto K_PIPE_MAX = size<1>(tAsA);

  // Total count of tiles
  int k_tile_count = size<1>(tAgA);
  // Current tile index in gmem to read from
  int k_tile = 0;

  // Initialize Barriers
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  uint64_t* producer_mbar = smem.tma_barrier;
  uint64_t* consumer_mbar = smem.mma_barrier;

  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;  // TMA
  using ConsumerBarType = cutlass::arch::ClusterBarrier;             // MMA
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
      ProducerBarType::init(&producer_mbar[pipe],   1);
      ConsumerBarType::init(&consumer_mbar[pipe], 128);
    }
  }
  // Ensure barrier init is complete on all CTAs
  cluster_sync();
```

在设置的时候一个 pipe stage 设置一个。

ClusterBarrier 和 ClusterTransactionBarrier 是 cutlass 实现的 mbarrier 相关的类，里面封装了一些 mbarrier 需要用到的函数。详见 mbarrier。

ProducerBarType 因为需要统计传输多少数据量，所以使用的是 ClusterTransactionBarrier。

在初始化时，因为 tma 只需要一个线程参与，而且后续会使用 mbarrier.arrive.expect_tx 指令，所以 arrive_count 设为了 1。ConsumerBarType 因为需要所以线程参与计算，所以 arrive_count 是所有的线程数。

初始化完成后使用 cluster_sync();同步一下其他线程。

mbarrier 初始化完成后就可以进行异步拷贝了。

```cpp
  // Start async loads for all pipes
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
  {
    if ((warp_idx == 0) && lane_predicate)
    {
      // Set expected Tx Bytes after each reset / init
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
    }
    --k_tile_count;
    ++k_tile;
  }
```

tma 使用一个线程就可以启动。arrive_and_expect_tx 函数底层使用的是 ptx 的 mbarrier.arrive.expect_tx 指令。该指令用于同时执行 mbarrier 的 arrive-on 和 expect_tx 操作。当 mbarrier 设置 transaction_bytes 时，必须 arrive_count 和 transaction_bytes 同时为 0 才说明当前阶段完成。执行 arrive_and_expect_tx 后，mbarrier 的 arrive_count 会减去 1，变成 0，但是 tma_transaction_bytes 不是 0，所以说明当前阶段没有结束。

然后进行 copy 操作，tma_a.with(producer_mbar[pipe])会返回一个带有 mbarrier 的 copy traits，然后被 copy 调用 tma op 进行拷贝。

拷贝结束后会对 mbarrier 执行 complete-tx 操作，把 transaction_bytes 减去传输的数据量。当 transaction_bytes=0 时表示传输已经完成。

这里是三个 pipe stage 都进行了 tma 拷贝。

然后是 tiledmma 相关的 partition，没什么好说的，详见 sgemm_sm90 中的解释。

```cpp
  //
  // Define A/B partitioning and C accumulators
  //
  // TUTORIAL:
  //   The tCrA and tCrB are actually Tensors of MMA Descriptors constructed as views of SMEM.
  //   The MMA Descriptor generation is automatic via inspection and validation of the SMEM Layouts.
  //   Because the MMA reads directly from SMEM and the fragments are descriptors rather than registers,
  //     there is no need for copy(tCsA, tCrA) in the mainloop.
  //

  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate accumulators and clear them
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
  clear(tCrC);

  // Allocate "fragments"
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)
```

tCrA 和 tCrB 类似于下面这种，是 wgmma 描述符组成的 tensor。

tCrA : GMMA::DescriptorIterator o (_1,_2,_4,(_1,_3)):(_0,_512,_2,(_0,_1024))

tCrB : GMMA::DescriptorIterator o (_1,_2,_4,(_1,_3)):(_0,_512,_2,(_0,_1024))

tCrC : ptr[16b](https://0x7fea69fffbd0) o ((_2,_2,(_8,_2)),_2,_1):((_1,_2,(_4,_32)),_64,_0)

最后进入计算主循环。

进入之前会先定义两个 PipelineState 对象记录每一个 stage。

```cpp
  // A PipelineState is a circular pipe index [.index()] and a pipe phase [.phase()]
  //   that flips each cycle through K_PIPE_MAX.
  auto write_state = cutlass::PipelineState<K_PIPE_MAX>();             // TMA writes
  auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();             // MMA  reads
```

PipelineState 由 index_ ，phase_ 和 count_ 组成。

```cpp
// Circular Buffer Index + Associated Phase
// Assumes only one operation possible - i.e., ++
template<uint32_t Stages_>
struct PipelineState {

  static constexpr uint32_t Stages = Stages_;

  int index_ = 0;
  uint32_t phase_ = 0;
  uint32_t count_ = 0;

  CUTLASS_DEVICE
  PipelineState(): index_{}, phase_{}, count_{} {}

  CUTLASS_DEVICE
  PipelineState(int index, uint32_t phase, uint32_t count)
    : index_(index)
    , phase_(phase)
    , count_(count) {}

  CUTLASS_DEVICE
  int index() const {
    return index_;
  }

  CUTLASS_DEVICE
  uint32_t phase() const {
    return phase_;
  }

  CUTLASS_DEVICE
  uint32_t count() const {
    return count_;
  }

  CUTLASS_DEVICE
  void operator++() {
    if constexpr (Stages > 0) {
      ++index_;
      ++count_;
      if (index_ == Stages) {
        index_ = 0;
        phase_ ^= 1;
      }
    }
  }

  CUTLASS_DEVICE
  PipelineState& operator+=(uint32_t num_iterations) {
    return advance(num_iterations);
  }
```

在主循环中代码如下：

```cpp
  CUTE_NO_UNROLL
  while (k_tile_count > -K_PIPE_MAX)
  {
    // Wait for Producer to complete
    int read_pipe = read_state.index(); // 获取 read_state 当前是第一个 pipe stage。
    ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase()); // 通过 try wait 确保 mbarrier 对应的异步拷贝完成。

    // MMAs to cover 1 K_TILE
    warpgroup_arrive(); // 执行 wgmma fence
    // 使用 wgmma 计算，一次会计算很多个 wgmma
    gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);     // (V,M) x (V,N) => (V,M,N)
    warpgroup_commit_batch(); // 把前面的 wgmma 提交到一个 wgmma-group 中。

    // Wait for all MMAs in a K_TILE to complete
    warpgroup_wait<0>(); // 等待前面所有的 wgmma 计算完成。

    // Notify that consumption is done
    ConsumerBarType::arrive(&consumer_mbar[read_pipe]); // 所有的线程对 consumer 的 mbarrier 执行 arrive-on，表示线程已经到这里了，前面的计算完成了。
    ++read_state; // 读取 stage 前进一步

    if ((warp_idx == 0) && lane_predicate)
    {
      int pipe = write_state.index(); // 获取写的 stage。
      // Wait for Consumer to complete consumption
      ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase()); // 确保对应 mbarrier 跟踪的前面计算完成。
      // Set expected Tx Bytes after each reset / init
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes); // 向 pipe 中写入
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe)); // 启动异步拷贝。
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
      ++write_state; // 写的 stage 前进一步。
    }
    --k_tile_count;
    ++k_tile;
  }
```

整体流程比较简单，主要就是在计算前通过 wait 确保读取完成。在读取时确保计算完成。在计算时确保计算完成。
