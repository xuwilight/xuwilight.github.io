---
title: CuTe 学习笔记（十四）GEMM SM90 TMA
date: 2025-04-25 18:00:00
tags: [CUTLASS, CuTe]
categories: [Cutlass 学习笔记, CuTe]
description: gemm_sm90_tma 用 TMA 由单线程一条指令搬完整个 tile、把 gmem→smem 搬运卸载给硬件，并用 mbarrier 做 producer/consumer 双向同步，写出真正重叠的异步流水。
---

[gemm_sm90_tma](https://github.com/NVIDIA/cutlass/blob/v4.1.0/examples/cute/tutorial/hopper/wgmma_tma_sm90.cu) 用 TMA（Tensor Memory Accelerator）搬数据：一条指令由一个线程触发，硬件一次搬完整个 128×64 tile，不占线程指令周期、不经过寄存器。再用 mbarrier 做细粒度的 producer/consumer 双向同步，让 TMA 与 wgmma 真正并行。

和上一版 gemm_sm90 相比，它虽然也用了 wgmma，但 gmem→smem 还是靠线程逐条发 cp.async，且主循环里的 cp_async_wait<0> 把"搬下一块"和"算当前块"串了起来；这一版把搬运卸载给 TMA、用屏障替代指令顺序做同步，gmem 读取与计算才真正重叠。

矩阵规模 M = N = 5120、K = 4096，全是 half，A、B 都是 K-major（TN GEMM），C 列主序。编译同样需要 sm_90a。

| | gemm_sm90 | gemm_sm90_tma |
|---|---|---|
| gmem→smem | 线程 cp.async（每线程 16B） | TMA 指令（一条搬整个 128×64 tile） |
| 搬运发起者 | 全部 128 线程 | 单线程（warp 0 的 elect one） |
| gmem 寻址 | 指针 + layout | TMA 描述符 + 坐标 tensor |
| 同步 | `cp_async_wait<N>` | mbarrier（producer/consumer 两个方向） |
| 流水控制 | 手动指针环形 | PipelineState（index + phase） |
| 计算 | wgmma | wgmma（不变） |

核心变化就是把"线程主动搬"换成"硬件异步搬 + 屏障通知"，gmem 拷贝不再和计算抢线程资源。

## Host 端设置

矩阵规模 M = N = 5120、K = 4096，全是 half。CTA tile 是 128×128×64，bP = 3 级流水：

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

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int< 64>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline
```

dA/dB 的 stride 都是 (ld, 1)，K 方向连续，即 A、B 都是 K-major 的 TN 布局；dC 是 (1, ld) 的列主序。smem layout、MMA 沿用 gemm_sm90（Layout_K_SW128_Atom 做 128B swizzle 防 bank conflict，SM90_64x64x16_SS 表示指令一次算 64×64、K=16、两个操作数都来自 smem），再加上 gmem 的 mA/mB 和两个 TMA atom：

```cpp
  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  // Define the MMA
  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{});

  // Define the TMAs
  // Create Global memory tensors for TMA inspection
  Tensor mA = make_tensor(A, make_shape(M,K), dA);
  Tensor mB = make_tensor(B, make_shape(N,K), dB);

  // Create TMA Atoms with the desired copy operation on the source and destination
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));
```

TiledMMA 没有在 M/N 方向再做扩展，所以 ThrLayoutVMNK 里 V 就是 wgmma 的 128 个线程、MNK 都是 1；LayoutA_TV/LayoutB_TV 在线程维上的 stride 是 0，表示每个线程看到的都是整个 64×16 操作数——_SS 操作数放在 smem 里，通过描述符共享，不按线程拆：

```
tiled_mma:TiledMMA
  ThrLayoutVMNK:  (_128,_1,_1,_1):(_1,_0,_0,_0)
  PermutationMNK: (_,_,_)
MMA_Atom
  ThrID:      _128:_1
  Shape_MNK:  (_64,_64,_16)
  LayoutA_TV: (_128,(_64,_16)):(_0,(_1,_64))
  LayoutB_TV: (_128,(_64,_16)):(_0,(_1,_64))
  LayoutC_TV: ((_4,_8,_4),(_2,_2,_8)):((_128,_1,_16),(_64,_8,_512))
```

make_tma_atom(拷贝op, gmem tensor, smem tile, tile shape) 描述了"从 gmem 的某个 tile 搬到 smem 的某个 tile"。得到的 atom：

```
tmaA:Copy_Atom
  ThrID:        _1:_0
  ValLayoutSrc: (_1,_8192):(_0,_1)
  ValLayoutDst: (_1,_8192):(_0,_1)
  ValLayoutRef: (_1,_8192):(_0,_1)
  ValueType:    16b

tmaB:Copy_Atom
  ThrID:        _1:_0
  ValLayoutSrc: (_1,_8192):(_0,_1)
  ValLayoutDst: (_1,_8192):(_0,_1)
  ValLayoutRef: (_1,_8192):(_0,_1)
  ValueType:    16b
```

ThrID = _1 表示 TMA 只由一个线程触发；(_1,_8192) 表示一条 TMA 指令就搬 128×64 = 8192 个 half。


TMA 的 PTX 指令如下（2D 版本）：

```cpp
asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
             " [%0], [%1, {%3, %4}], [%2];" ::"r"(smem_int_ptr),
             "l"(&src_tensor_map), "r"(smem_int_mbar), "r"(crd0), "r"(crd1) : "memory");
```

它会用到一个 TMA 描述符 src_tensor_map（编码了 base 地址、维度、元素类型）+ 一组坐标 {crd0, crd1}。所以 CuTe 里要通过"坐标"来寻址，这就引出了 make_inttuple_iter 构造的坐标 tensor。

普通 tensor 用 stride 把逻辑坐标 (i,j) 映射成一维地址 `i*a + j*b`；坐标 tensor 则用 E<> 把逻辑坐标映射成元组坐标：

```cpp
Tensor a = make_tensor(make_inttuple_iter(0,0),
                       make_shape (     4,      5),
                       make_stride(E<0>{}, E<1>{}));
print_tensor(a);

Tensor b = make_tensor(make_inttuple_iter(0,0),
                       make_shape (     4,      5),
                       make_stride(E<1>{}, E<0>{}));
print_tensor(b);
```

E<0> 表示"在第 0 个位置加 1"（等价记号 1@0），E<1> 表示在第 1 个位置加 1（1@1）。加法就是逐分量相加，例如 `3*E<0>{} + 4*E<1>{} = 3@0 + 4@1 = (3,4)`，即第 0 位为 3、第 1 位为 4。常用记号对应关系：

| 对象 | coord | 记号 |
|---|---|---|
| E<0>{} | (1,0,...) | 1@0 |
| E<1>{} | (0,1,0,...) | 1@1 |
| E<0,0>{} | ((1,0,...),0,...) | 1@0@0 |
| E<1,0>{} | (0,(1,0,...),0,...) | 1@0@1 |

所以 stride (1@0, 1@1)、坐标 (i,j) → i@0 + j@1 = (i,j)，正好是 TMA 需要的二维坐标。上面两个 tensor 打印出来分别是：

```
ArithTuple(0,0) o (4,5):(_1@0,_1@1):
  (0,0)  (0,1)  (0,2)  (0,3)  (0,4)
  (1,0)  (1,1)  (1,2)  (1,3)  (1,4)
  (2,0)  (2,1)  (2,2)  (2,3)  (2,4)
  (3,0)  (3,1)  (3,2)  (3,3)  (3,4)

ArithTuple(0,0) o (4,5):(_1@1,_1@0):
  (0,0)  (1,0)  (2,0)  (3,0)  (4,0)
  (0,1)  (1,1)  (2,1)  (3,1)  (4,1)
  (0,2)  (1,2)  (2,2)  (3,2)  (4,2)
  (0,3)  (1,3)  (2,3)  (3,3)  (4,3)
```

(_1@0,_1@1) 把逻辑 (i,j) 一一映射到坐标 (i,j)；(_1@1,_1@0) 则把逻辑 (i,j) 映射到 (j,i)，两个坐标分量互换。

关键点：TMA 描述符的坐标是从 stride=1 的维度开始编号的。

- column-major（MN-major）：逻辑 (i,j) 的连续维是 i（行），TMA 的 crd0 也对应 i，所以 stride 就是 (_1@0, _1@1)。
- row-major（K-major）：逻辑 (i,j) 的连续维是 j（K），TMA 的 crd0 对应 j，于是 (i,j) 对应的 TMA 坐标是 (j,i)，stride 变成 (_1@1, _1@0)。

这个差异由 make_tma_atom 根据 gmem tensor 的实际 layout 和 smem tile 自动推出来，不用手写。

## TMA tensor

进入 kernel，gmem tensor 改用 get_tma_tensor 生成——返回的是坐标 tensor：

```cpp
  // Represent the full tensors
  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));                   // (M,K) TMA Tensor
  Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));                   // (N,K) TMA Tensor
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);      // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory tensors
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
```

```cpp
  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }
```

g_stride_ 就是前面说的 (_1@1, _1@0)（K-major）。之后的 local_tile 分块和前面一样，只不过作用在坐标 tensor 上。打印出的坐标 tensor，MN-major 是逻辑顺序、K-major 则是转置的：

```
// MN-major
ArithTuple(_0,_0) o (_4,_8):(_1@0,_1@1):
  (0,0)  (0,1)  (0,2)  (0,3)  (0,4)  (0,5)  (0,6)  (0,7)
  (1,0)  (1,1)  (1,2)  (1,3)  (1,4)  (1,5)  (1,6)  (1,7)
  (2,0)  (2,1)  (2,2)  (2,3)  (2,4)  (2,5)  (2,6)  (2,7)
  (3,0)  (3,1)  (3,2)  (3,3)  (3,4)  (3,5)  (3,6)  (3,7)
```

K-major 下每个逻辑行沿 crd0 铺满整个 K，坐标是 (k,m) 的形式（首行 (0,0) (1,0) ... (63,0)，次行 (0,1) (1,1) ... (63,1)），这里从略。

smem 端多了两组 barrier：

```cpp
template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;

  uint64_t tma_barrier[size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};
```

TMA 要求 16 字节对齐，所以 A、B 用 alignas(128)。每个 pipe stage 配一对 64-bit 的 barrier：tma_barrier（producer，通知"数据已到位"）和 mma_barrier（consumer，通知"数据已被算完、可以覆盖"）。

拷贝分区用 TMA 专用的 tma_partition，它的完整实现是：

```cpp
// The "VectorCopy Partitioner" for TMA
template <class... Args,
          class CtaCoord,
          class TShape, class TStride,
          class SEngine, class SLayout,
          class GEngine, class GLayout>
CUTE_DEVICE
auto
tma_partition(Copy_Atom<Args...>      const& copy_atom,
              CtaCoord                const& cta_coord,
              Layout<TShape,TStride>  const& cta_layout,  // T: CTA coord -> logical multicast id
              Tensor<SEngine,SLayout> const& stensor,     // SMEM Tensor (TMATile, Rest...)
              Tensor<GEngine,GLayout> const& gtensor)     // GMEM Tensor (TMATile, Rest...)
{
  CUTE_STATIC_ASSERT_V(size<0>(stensor) == size<0>(gtensor));

  // Invert the smem to get the largest contiguous vector in the smem layout
  Layout inv_smem_layout = right_inverse(get_nonswizzle_portion(layout<0>(stensor)));
  // Scale that up to cover all of the smem_coords
  Layout layout_v = tile_to_shape(make_layout(inv_smem_layout), size<0>(stensor));

  // Factor out the single-instrucion portion
  Layout tma_layout_v = make_layout(Int<Copy_Atom<Args...>::NumValSrc>{});
  auto layout_V = make_tile(logical_divide(layout_v, tma_layout_v));

  // Append with _ until we cover all Rest... modes
  auto glayout_V = append<GLayout::rank>(layout_V, _);
  auto slayout_V = append<SLayout::rank>(layout_V, _);
  // Transform tile mode and coalesce
  Tensor gtensor_v = coalesce(gtensor.compose(glayout_V), Shape<Shape<_1,_1>>{});    // ((TMA,TMA_Iter), Rest...)
  Tensor stensor_v = coalesce(stensor.compose(slayout_V), Shape<Shape<_1,_1>>{});    // ((TMA,TMA_Iter), Rest...)
  // Offset inside the TMA-mode for the multicast
  auto multicast_offset = cta_layout(cta_coord) * (size(tma_layout_v) / cosize(cta_layout));
  auto multicast_coord  = make_coord(make_coord(multicast_offset, Int<0>{}));
  auto gcoord = append<GLayout::rank>(multicast_coord, Int<0>{});
  auto scoord = append<SLayout::rank>(multicast_coord, Int<0>{});

  Tensor gresult = domain_offset(gcoord, gtensor_v);
  Tensor sresult = domain_offset(scoord, stensor_v);

  return cute::make_tuple(gresult, sresult);
}
```

调用时传入 TMA atom、multicast 的 cta coord / layout、smem tensor 和 gmem 坐标 tensor：

```cpp
  auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));  // (TMA,k) 和 (TMA,PIPE)
  auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sB), group_modes<0,2>(gB));
```

- Int<0>{} 和 Layout<_1>{} 表示不做 multicast（cta coord 为 0、cta layout 为 1）。如果要多播（一个 CTA 搬的数据给 cluster 内多个 CTA 用），就传真实坐标和 layout。
- group_modes<0,2> 把 (X,Y,Z) 压成 ((X,Y),Z)：把 tile 内的所有维度合进 mode-0，因为 TMA 一次负责整个 mode-0 的数据，剩下的 Z 就是 stage / k-tile。

经过内部这一串 shape 运算，得到每个 pipe stage 在 gmem 坐标 tensor 和 smem tensor 上的分区：

```
tAgA: ArithTuple(_0,0) o (((_64,_128),_1),1):(((_1@0,_1@1),_0),_64@0)
tAsA: Sw<3,4,3>_smem_ptr[16b](0x7f7700000400) o ((_8192,_1),(_1,_3)):((_1,_0),(_0,_8192))
```

tAgA 的坐标 tensor 是 (_64,_128)——TMA 一次覆盖 gmem 上 64×128 的坐标（K-major 下 crd0 = K = 64、crd1 = M = 128）。tAsA 是一次覆盖 8192 个 half、叠 3 个 stage。

传输字节数用于告诉 barrier "这次要搬多少"：

```cpp
  constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                      + sizeof(make_tensor_like(tensor<0>(tBsB)));
```

A、B 各 8192 个 half = 16 KB，加起来 32 KB。

## mbarrier 定义

```cpp
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  uint64_t* producer_mbar = smem.tma_barrier;
  uint64_t* consumer_mbar = smem.mma_barrier;

  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;  // TMA
  using ConsumerBarType = cutlass::arch::ClusterBarrier;             // MMA
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
      ProducerBarType::init(&producer_mbar[pipe],   1);
      ConsumerBarType::init(&consumer_mbar[pipe], 128);
    }
  }
  cluster_sync();
```

- producer barrier（TMA）：ClusterTransactionBarrier，带 transaction bytes 统计。init(..., 1) 的 arrive_count = 1，因为只有 1 个线程（warp 0 的 elect one）负责发起 TMA。TMA 完成时硬件自动对 barrier 做 complete-tx，把剩余字节数扣掉。
- consumer barrier（MMA）：ClusterBarrier，arrive_count = 128，因为 128 个线程都要参与计算，每个线程算完 arrive 一次，全员到齐才算"这块 smem 消费完毕、可以覆盖"。
- cluster_sync()：barrier 初始化后要跨 CTA 同步（这里 cluster 是 2×1×1，见后面 launch 参数）。

一个 mbarrier 的"一个阶段完成"要同时满足：arrive_count 归零且 expectation bytes 归零。

## TMA 预取

```cpp
  auto K_PIPE_MAX = size<1>(tAsA);
  int k_tile_count = size<1>(tAgA);
  int k_tile = 0;

  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
  {
    if ((warp_idx == 0) && lane_predicate)
    {
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
    }
    --k_tile_count;
    ++k_tile;
  }
```

注意这里和 SM80/SM90 不同：三个 stage 全部预取（pipe < K_PIPE_MAX，不是 K_PIPE_MAX-1），因为现在搬运完全异步、发起一下就走，不需要留一块给主循环腾手。

arrive_and_expect_tx 对应 mbarrier.arrive.expect_tx：一次完成"arrive"（count 减 1 → 0）和"expect_tx"（登记本次要收 tma_transaction_bytes 字节）。之后 `copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe))` 发射 TMA，tma_a.with(...) 返回一个绑定了该 barrier 的 copy traits，TMA 每传完一部分就对 barrier 做 complete-tx 扣减字节。等字节数扣到 0，barrier 的这一阶段就 flip，消费者就能通过 wait 看到。

## MMA 分区与累加器

```cpp
  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
  clear(tCrC);

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // descriptor tensor
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
```

和 gemm_sm90 完全一样——tCrA / tCrB 是描述符 tensor（不是寄存器数据），所以主循环里不需要 copy(tCsA, tCrA) 这一步。生成描述符的过程（make_fragment_A → MakeTensor<smem_desc> → make_gmma_desc → LBO/SBO）我们已在上一篇文章里拆过。打印出来 tCrA/tCrB 是 GMMA::DescriptorIterator，tCrC 是真正的寄存器指针：

```
tCrA : GMMA::DescriptorIterator o (_1,_2,_4,(_1,_3)):(_0,_512,_2,(_0,_1024))
tCrB : GMMA::DescriptorIterator o (_1,_2,_4,(_1,_3)):(_0,_512,_2,(_0,_1024))
tCrC : ptr[16b](0x7fea69fffbd0) o ((_2,_2,(_8,_2)),_2,_1):((_1,_2,(_4,_32)),_64,_0)
```

## 主循环

```cpp
  auto write_state = cutlass::PipelineState<K_PIPE_MAX>();             // TMA writes
  auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();             // MMA  reads

  CUTE_NO_UNROLL
  while (k_tile_count > -K_PIPE_MAX)
  {
    // Wait for Producer to complete
    int read_pipe = read_state.index();
    ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

    // MMAs to cover 1 K_TILE
    warpgroup_arrive();
    gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);     // (V,M) x (V,N) => (V,M,N)
    warpgroup_commit_batch();

    // Wait for all MMAs in a K_TILE to complete
    warpgroup_wait<0>();

    // Notify that consumption is done
    ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
    ++read_state;

    // Only issue new TMA copies if there are more tiles to fetch
    if ((warp_idx == 0) && lane_predicate && (k_tile_count > 0))
    {
      int pipe = write_state.index();
      ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
      ++write_state;
    }
    --k_tile_count;
    ++k_tile;
  }

  axpby(alpha, tCrC, beta, tCgC);
```

PipelineState 就是一个环形指针 + phase 位：每 ++ 一次 index 前进一格，绕回 0 时 phase_ ^= 1。phase 用来区分"这格 barrier 现在是第几次使用"——因为 barrier 是复用的，phase 翻转能避免把上一轮的完成误判成本轮的完成。结构体只有 index_、phase_、count_ 三个字段：

```cpp
// Circular Buffer Index + Associated Phase
// Assumes only one operation possible - i.e., ++
template<uint32_t Stages_>
struct PipelineState {

  static constexpr uint32_t Stages = Stages_;

  int index_ = 0;
  uint32_t phase_ = 0;
  uint32_t count_ = 0;

  CUTLASS_DEVICE
  PipelineState(): index_{}, phase_{}, count_{} {}

  CUTLASS_DEVICE
  PipelineState(int index, uint32_t phase, uint32_t count)
    : index_(index)
    , phase_(phase)
    , count_(count) {}

  CUTLASS_DEVICE
  int index() const {
    return index_;
  }

  CUTLASS_DEVICE
  uint32_t phase() const {
    return phase_;
  }

  CUTLASS_DEVICE
  uint32_t count() const {
    return count_;
  }

  CUTLASS_DEVICE
  void operator++() {
    if constexpr (Stages > 0) {
      ++index_;
      ++count_;
      if (index_ == Stages) {
        index_ = 0;
        phase_ ^= 1;
      }
    }
  }

  CUTLASS_DEVICE
  PipelineState& operator+=(uint32_t num_iterations) {
    return advance(num_iterations);
  }
};
```

在一次循环中：

消费者（MMA）一侧：
1. ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase())：等这个 stage 的 TMA 数据到齐。
2. warpgroup_arrive() + gemm(...) + warpgroup_commit_batch() + warpgroup_wait<0>()：发出本 stage 的 wgmma（bM/bN=128 对 64、bK=64 对 16，共 2×2×4 = 16 条）并等它们算完。
3. ConsumerBarType::arrive(&consumer_mbar[read_pipe])：128 个线程各自 arrive，告诉 producer 可以写入了。

生产者（TMA）一侧（只在 warp 0 的 elect one 上执行）：
1. ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase())：等目标 stage 被消费完（上一轮它的 barrier 全到齐）。
2. arrive_and_expect_tx + 两次 copy(tma_...)：发起这个 stage 的下一块 TMA。

和 gemm_sm90 那种"指令顺序即同步"（cp_async_wait<0> 硬等全部）不同，这里每个 stage 的同步是按 pipe 独立的：算第 0 块时，第 2 块的 TMA 可以还在飞；TMA 推进到哪一格、MMA 推进到哪一格，各有各的 PipelineState，只是通过 barrier 在"该格"上握手。这就把搬运和计算真正解耦开了。

while (k_tile_count > -K_PIPE_MAX)：初始 k_tile_count 已因预取减到 K_TILE_MAX - K_PIPE_MAX，循环里每轮 --，跑到 -K_PIPE_MAX 时退出——多出来的这几轮用来把预取塞进流水的最后几块收尾算掉。

## 关于 cluster

gemm_tn 里 dimCluster(2, 1, 1)，kernel 通过 launch_kernel_on_cluster 以 2×1×1 的 cluster 启动。这一版示例里 cluster 主要影响 grid 对齐（dimGrid 用 round_up(..., dimCluster)），为 TMA multicast 做准备；本示例的 tma_partition 传的是 Int<0>{}, Layout<_1>{}，即没有启用 multicast。真正的 multicast 场景里，同 cluster 的多个 CTA 会共享一次 TMA 搬运，让 tma_partition 接收真实的 cta coord 和 layout。

## 小结

gemm_sm90_tma 在 gemm_sm90 的基础上把"数据搬运"也彻底异步化，是本系列里第一个真正重叠的流水：

1. TMA 搬运：一条指令由一个线程触发，硬件搬整个 128×64 tile，不占线程指令周期、不经过寄存器。用 TMA 坐标 tensor（make_inttuple_iter + get_tma_tensor）寻址，坐标从 stride=1 的维度起编号，K-major 下坐标顺序相对逻辑坐标是转置的。
2. mbarrier 双向同步：producer（ClusterTransactionBarrier，按传输字节计数）通知"数据到位"，consumer（ClusterBarrier，128 线程 attend）通知"数据算完可覆盖"。
3. PipelineState：环形 index + phase，让每个 stage 的 barrier 复用而不混淆。
4. 按 stage 解耦：TMA 和 wgmma 各自推进自己的 PipelineState，只在具体某一格上用 barrier 握手，从而 gmem 读取与计算真正并行。

和 SM80 的 cp.async + __syncthreads、SM90 的 cp.async + cp_async_wait<0> 相比，这一版把"同步"从"指令顺序"变成了"硬件屏障"，是 CUTLASS 现代 mainloop（以及后面的 warp specialization、cluster multicast）的基础模型。
