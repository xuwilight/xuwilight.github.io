---
title: CuTe 学习笔记（十三）GEMM SM90
date: 2025-04-22 18:00:00
tags: [CUTLASS, CuTe]
categories: [Cutlass 学习笔记, CuTe]
description: gemm_sm90 用 Hopper 的 wgmma 指令做计算，A、B 直接从 shared memory 经 GMMA 描述符读取，配合 swizzle smem 布局和 warpgroup 同步机制。
---

[gemm_sm90](https://github.com/NVIDIA/cutlass/blob/v4.1.0/examples/cute/tutorial/hopper/wgmma_sm90.cu) 用 Hopper 的 wgmma 指令做矩阵乘：矩阵 A、B 直接从 shared memory 读取——不需要 ldmatrix，也不占寄存器，一次指令算 64×64×16。

和 SM80 版本相比，上一版用 mma.sync，每算一小块都要先把 A、B 从 smem 经 ldmatrix 搬进寄存器；这一版改用 64-bit 的 GMMA 描述符告诉硬件数据在 smem 的什么位置、什么 swizzle，wgmma 自己去读，既省寄存器，也让一个 warpgroup（128 线程）用一条指令算 64×64。

矩阵规模 M = N = 5120、K = 4096，数据类型全是 half，A、B 都是 K-major（TN GEMM），C 列主序。编译需要 sm_90a：

```bash
nvcc wgmma_sm90.cu -o wgmma -I ../../../../include/ -I ../../../../tools/util/include/ -arch=sm_90a -std=c++17
```

直接进入 gemm_tn。

## Host 端参数

先定义全局矩阵的规模与 stride。A、B 都是 K-major，K 方向 stride = 1，M/N 方向是 leading dim；C 列主序，M 方向 stride = 1：

```cpp
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K); // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{}); // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{}); // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)
```

CTA tile 和前面一样，bK 提到 64，bP = 3：

```cpp
  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int< 64>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN,bK,bP));
```

smem 不再手写 Swizzle<3,3,3>，改用 GMMA::Layout_K_SW128_Atom。这套 atom 定义在 include/cute/atom/mma_traits_sm90_gmma.hpp，分 MN 方向和 K 方向，各 4 种 swizzle 力度：

```cpp
// M|N-major GMMA layouts in units of bits
using Layout_MN_INTER_Atom_Bits = ComposedLayout<Swizzle<0,4,3>, smem_ptr_flag, Layout<Shape< _128,_8>,Stride<_1, _128>>>;
using Layout_MN_SW32_Atom_Bits  = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape< _256,_8>,Stride<_1, _256>>>;
using Layout_MN_SW64_Atom_Bits  = ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag, Layout<Shape< _512,_8>,Stride<_1, _512>>>;
using Layout_MN_SW128_Atom_Bits = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_1024,_8>,Stride<_1,_1024>>>;

// K-major GMMA layouts in units of bits
using Layout_K_INTER_Atom_Bits = ComposedLayout<Swizzle<0,4,3>, smem_ptr_flag, Layout<Shape<_8, _128>,Stride< _128,_1>>>;
using Layout_K_SW32_Atom_Bits  = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape<_8, _256>,Stride< _256,_1>>>;
using Layout_K_SW64_Atom_Bits  = ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag, Layout<Shape<_8, _512>,Stride< _512,_1>>>;
using Layout_K_SW128_Atom_Bits = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_8,_1024>,Stride<_1024,_1>>>;
```

Swizzle<B,M,S> 三个参数：M 是 base（多少 bit 一个 swizzle 基元），S 是 swizzle 的列数，B 是参与 swizzle 的行数。以 K-major 为例逐个看：

- Layout_K_INTER_Atom_Bits：Swizzle<0,4,3> 第一个参数是 0，表示在 1 行上 swizzle，等于不 swizzle；基元 2⁴ = 16 bit、8 列一组；底层是 8 行 × 128 bits、K 方向连续。
- Layout_K_SW32_Atom_Bits：Swizzle<1,4,3>，32B 一个 swizzle pattern，底层 8 行 × 256 bits。若元素 2 字节，就是 8 行 × 16 列；数据连续也可看成 2 行 × 64 列。3 表示 8 列一组参与 swizzle，1 表示 2 行一组。
- Layout_K_SW128_Atom_Bits：Swizzle<3,4,3>，基元 2⁴ = 16 字节（正好一个 128-bit），底层 8 行 × 1024 bits（= 8 行 × 128 字节），K 方向连续；3 表示 8 行一组、每组 8 列。

MN 方向的 atom 是转置过来的形式，shape 的 MN 维在左、stride 沿 MN 连续，用于 A/B 取 MN-major 布局时。

这些 _Bits 版本按 bit 寻址，实际使用时会按元素类型重排：

```cpp
template <class Type>
using Layout_K_SW128_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW128_Atom_Bits{}));
```

对 half（16-bit），upcast<16> 把 bit 寻址换成元素寻址：

```
Layout_K_SW128_Atom_Bits = Sw<3,4,3> o smem_ptr[1b]  o (_8,_1024):(_1024,_1)
Layout_K_SW128_Atom      = Sw<3,4,3> o smem_ptr[16b] o (_8,_64):(_64,_1)
```

print 出来可以看到元素地址已经被 swizzle 打散（下图，8×64 的地址矩阵，能看到列方向的交错条纹）。

![](/assets/gemm_sm90/image.png)

tile_to_shape 用这个 8×64 的 atom 铺满 (bM,bK,bP) = (128,64,3)：bM 方向铺 16 份、bK 方向刚好 1 份、再叠 3 个 stage。铺完后 sA 的 layout 是：

```
Sw<3,4,3> o smem_ptr[16b] o ((_8,16),(_64,1)):((_64,_512),(_1,8192))
```

8×64 的 atom 在 bM 方向复制 16 份（(_8,16)，stride 512），bK 方向刚好一份（(_64,1)）。bK 若小于 64 也不会报错，ceil_div 结果是 1。

> 关于 Swizzle 中间那个数：_Bits 版本里是 4（字节基元，2⁴=16 字节），upcast 成 half 元素后应变成 3（2³=8 个 half = 16 字节）。但直接打印 sA（swizzle 还挂在字节寻址的 smem_ptr_flag 上）时看到的仍是 4；等后面调用 as_position_independent_swizzle_tensor 把 swizzle 真正 recast 到 half 元素寻址，就会变成 Sw<3,3,3>。

## cp.async 与 TiledMMA

```cpp
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{}, // Thr layout 16x8 k-major
                                    Layout<Shape< _1,_8>>{});              // Val layout  1x8
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{}, // Thr layout 16x8 k-major
                                    Layout<Shape< _1,_8>>{});              // Val layout  1x8

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{});
```

拷贝还是 SM80 的 cp.async + uint128_t（16 字节 / 线程 × 8 个 half），16×8 = 128 个线程，和 warpgroup 大小一致。thread layout (_16,_8):(_8,_1) 是 k-major，value layout (_1,_8) 表示每线程拷 8 个 half。TiledCopy 的中间推导（thr layout → val layout → tv layout）打印出来是：

```cpp
thr_layout: (_16,_8):(_8,_1)
val_layout: (_1,_8):(_0,_1)
layout_mn : ((_1,_16),(_8,_8)):((_0,_8),(_128,_1))
layout_tv : ((_8,_16),_8):((_128,_1),_16)
tiler     : (_16,_64)
```

最终 TiledLayout_TV: ((_8,_16),_8):((_128,_1),_16) 就是「对 threadIdx 索引得到线程负责的那片数据」的映射。

make_tiled_mma 只传了一个 atom、没有 thread layout，所以 TiledMMA 就等于这个 atom 本身：

```cpp
TiledMMA
  ThrLayoutVMNK:  (_128,_1,_1,_1):(_1,_0,_0,_0)
  PermutationMNK: (_,_,_)
MMA_Atom
  ThrID:      _128:_1
  Shape_MNK:  (_64,_64,_16)
  LayoutA_TV: (_128,(_64,_16)):(_0,(_1,_64))
  LayoutB_TV: (_128,(_64,_16)):(_0,(_1,_64))
  LayoutC_TV: ((_4,_8,_4),(_2,_2,_8)):((_128,_1,_16),(_64,_8,_512))
```

SM90_64x64x16_F16F16F16_SS 的 SS 表示 A、B 都从 smem 读，64x64x16 是单条 wgmma 的规模。ThrID = _128 表示这是一条 warpgroup（4 个 warp = 128 线程）协同的指令。LayoutA_TV 的第一维是 128 线程、stride 为 0——说明 128 个线程看到的 A 数据是同一份（不像 SM80 每个线程各读各的），这正是 wgmma 从 smem 直接读的特点。

## GMMA 描述符

sgemm_sm80 里 A、B 要在寄存器里存一堆 half；这里换成一个 GmmaDescriptor。它编码了 smem 起始地址、swizzle 类型、以及两个偏移（LBO / SBO），wgmma 据此自己去 smem 取数。

描述符是在 partition_A → make_fragment_A 这条链里"顺便"造出来的。先看完整的 Traits：

```cpp
template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_64,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout< 64, 16>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};
```

FrgTypeA 是 smem_desc 而不是值类型，这一点决定了后面的分支。make_fragment_A 里有：

```cpp
if constexpr (has_dereference<FrgTypeA>::value) {
  // If the intended FrgTypeA is a view (of the current tensor), forward the whole
  return make_tensor<FrgTypeA>(static_cast<ATensor&&>(atensor));
} else {
  // Else, the intended FrgTypeA is a value type, construct a new tensor with a fragment layout
  return make_fragment_like<FrgTypeA>(atensor);
}
```

smem_desc 可解引用，所以走 `make_tensor<FrgTypeA>(...)` 这条分支（不分配寄存器），最终落到 `MakeTensor<GMMA::smem_desc<MajorMode>>`：

```cpp
// Customization point for creating a GMMA::smem_desc Tensor
template <SM90::GMMA::Major MajorMode>
struct MakeTensor<SM90::GMMA::smem_desc<MajorMode>>
{
  template <class TEngine, class TLayout>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Tensor<TEngine,TLayout> const& smem_tensor)
  {
    static_assert(is_smem<TEngine>::value, "Expected SMEM Tensor to construct a GMMA Desc Tensor");
    return make_tensor(SM90::GMMA::DescriptorIterator{SM90::GMMA::make_gmma_desc<MajorMode>(tensor<0>(smem_tensor))},
                       replace<0>(recast<uint128_t const>(smem_tensor).layout(), Layout<_1,_0>{}));
  }
};
```

也就是说，它把 smem tensor 的第一个值块 tensor<0>(smem_tensor) 交给 make_gmma_desc 造一个描述符，再用一个 DescriptorIterator 包起来，最后用 Layout<_1,_0> 把 value 维替换掉——对 tensor 索引就等于在描述符的起始地址上做偏移。

传入 make_gmma_desc 的 tensor 是 Sw<3,4,3>_smem_ptr[16b] o (_64,_16):(_64,_1)，stride 是 (64,1)：行之间相隔 64（sA 有 64 列），K 方向 stride 1。

make_gmma_desc 依次填四样东西。先 recast 成 uint128_t：

```
tensor      = Sw<3,4,3>_smem_ptr[16b]  o (_64,_16):(_64,_1)
u128_tensor = Sw<3,4,3>_smem_ptr[128b] o (_64,_2):(_8,_1)
```

然后填 layout type、起始地址和 base offset：

```cpp
// Layout type
constexpr LayoutType LAYOUT_TYPE = layout_type(u128_tensor);
desc.bitfield.layout_type_ = uint8_t(LAYOUT_TYPE);

// Start address (4LSB not included)
uint32_t start_address = cast_smem_ptr_to_uint(raw_pointer_cast(u128_tensor.data()));
desc.bitfield.start_address_ = static_cast<uint16_t>(start_address >> 4);

constexpr uint8_t base_offset = 0;
desc.bitfield.base_offset_ = base_offset;
```

起始地址右移 4 位是因为描述符只编码到 16 字节粒度的低 4 位之外的地址。所有线程传进去的 LAYOUT_TYPE、start_address 都一样，指向 sA 的起始地址。

接着两项断言约束 canonical 布局。MN 方向必须能被 8 整除，K 方向（u128 单位）必须是 2（稠密）或 4（稀疏）：

```cpp
CUTE_STATIC_ASSERT_V(size<0>(u128_tensor) % Int<8>{} == Int<0>{},          // N|M size
                     "Not a canonical GMMA_K Layout: Expected MN-size multiple of 8.");
CUTE_STATIC_ASSERT_V(size<1>(u128_tensor) == Int<2>{} || size<1>(u128_tensor) == Int<4>{},      // K size
                     "Not a canonical GMMA_K Layout: Expected K-size 2 for dense or 4 for sparse (in units of uint128_t).");
```

K 方向固定是 2 是因为一条 wgmma 的 K 是 16 个 half = 32 字节 = 2 个 uint128_t；K-major 下这个 16 长度正好对应"2 个 128-bit 列"，所以 recast 后 K 维长度固定是 2；稀疏矩阵（每 2 个用 1 个）则是 4。

最后把 tensor layout 除以 canonical pattern 得到规范 layout，取其中的 stride 作为 LBO / SBO：

```cpp
// Construct the canonical GMMA N Layout with shape ((8,n),(2,1))
Layout canonical_layout = logical_divide(layout(u128_tensor), Tile<Layout<_8,_1>,Layout<_2,_1>>{}); // ((_8,_8),(_2,_1)):((_8,_64),(_1,_0))
```

stride_00 = 8 对 expected W = 8、stride_10 = 1（K-major 稠密必须为 1），这两个有 static_assert 校验；stride_01 = 64：

```cpp
constexpr uint32_t stride_00 = stride<0,0>(canonical_layout);
constexpr uint32_t expected_stride_00 = W;
static_assert(stride_00 == expected_stride_00, "Not a canonical GMMA_K Layout: Expected stride failure.");
constexpr uint32_t stride_10 = stride<1,0>(canonical_layout);
constexpr uint32_t expected_stride_10 = (LAYOUT_TYPE == LayoutType::INTERLEAVE) ? stride<1,0>(canonical_layout) : 1;
static_assert(stride_10 == expected_stride_10, "Not a canonical GMMA_K Layout: Expected stride failure.");

// stride dimension byte offset and leading dimension byte offset (4LSB not included == uint128_t units)
constexpr uint32_t stride_01 = stride<0,1>(canonical_layout);

desc.bitfield.stride_byte_offset_  = stride_01;   // 64  → SBO
desc.bitfield.leading_byte_offset_ = stride_10;   // 1   → LBO
```

指针偏移之后得到 tCrA：

```cpp
tCrA : GMMA::DescriptorIterator o (_1,_2,_4,(_1,_3)):(_0,_512,_2,(_0,_1024))
```

(_1,_2,_4) 是 (V=1, MMA_M=2, MMA_K=4)：128×64 的 sA 被 64×16 切成 2×4 = 8 块，每块一个描述符。索引它时 DescriptorIterator 只加偏移，不改其余字段：

```cpp
struct DescriptorIterator
{
  using reference    = GmmaDescriptor;
  using element_type = GmmaDescriptor;
  using value_type   = GmmaDescriptor;

  GmmaDescriptor desc_;

  // Dereference returns the GmmaDescriptor
  CUTE_HOST_DEVICE constexpr
  reference operator*() const { return desc_; }

  // Advance and return a new GmmaDescriptor
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  reference operator[](Index const& i) const { return *(*this + i); }

  // Return an advanced iterator
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  DescriptorIterator operator+(Index const& offset) const
  {
    // Use 32bit calculation rather than 64 bit calculation as we only update the part of desc
    GmmaDescriptor ret;
    ret.reg32_[0] = desc_.reg32_[0] + uint32_t(offset);
    ret.reg32_[1] = desc_.reg32_[1];
    return { ret };
  }
};
```

operator+ 只更新 reg32_[0]（低 32 位，含 start_address），所以 tCrA(_,1,0,0) 是在起始地址上加 512、`tCrA(_,0,1,0)` 加 2。打印 `print_tensor(tCrA(_,_,_,0))` 可以看到 2×4 个块的描述符各不相同，只有 start_addr 在变：

```
GMMA::DescriptorIterator o (_1,_2,_4):(_0,_512,_2):
  GmmaDescriptor: 0x4000004000010040
  start_addr :  0x0040
  leading_off:  0x0001 (1)
  stride_off :  0x0040 (64)
  base_offset:  0x0
  layout_type:  0x1 (B128)
  ...
  GmmaDescriptor: 0x4000004000010042   # K 方向相邻块，start_addr + 2
  start_addr :  0x0042
  ...
  GmmaDescriptor: 0x4000004000010240   # M 方向相邻块，start_addr + 512
  start_addr :  0x0240
  leading_off:  0x0001 (1)
  stride_off :  0x0040 (64)
  base_offset:  0x0
  layout_type:  0x1 (B128)
```

layout_type = 0x1 (B128) 对应 LayoutType::B128（Swizzle<3,4,3>），leading_off = 1、stride_off = 64 正是前面算出的 LBO / SBO。整个过程 A、B 一个数据寄存器都没占。

## as_position_independent_swizzle_tensor

拷贝端要把数据写进 swizzle 过的 smem，但 partition_D 需要知道 swizzle 后的真实地址。这一步由 as_position_independent_swizzle_tensor 完成：

```cpp
  Tensor sA_ = as_position_independent_swizzle_tensor(sA);
  Tensor tAsA = thr_copy_a.partition_D(sA_);
```

它把 swizzle 从"字节寻址 + 绝对地址"改写成"元素寻址、位置无关"：

```cpp
sA  = Sw<3,4,3>_smem_ptr[16b](0x...) o ((_8,_16),(_64,_1),(_1,_3)):((_64,_512),(_1,_0),(_0,_8192))
sA_ = smem_ptr[16b](0x...) o Sw<3,3,3> o _0 o ((_8,_16),(_64,_1),(_1,_3)):((_64,_512),(_1,_0),(_0,_8192))
```

指针和 layout 形状都没变，只是 swizzle 从 Sw<3,4,3>（字节基元 16B）recast 成 Sw<3,3,3>（half 元素基元 8 个 = 16B）——这就是前面提到的"4 变 3"。position independent 指的就是不再依赖 smem 起始地址的绝对对齐，便于在 kernel 里对任意偏移的 smem 做分区。

分区后 copy 的两端分别是：

```cpp
tAgA = gmem_ptr[16b](0x...) o ((_8,_1),_8,_1,64):((_1,_0),65536,_0,_64)
tAsA = smem_ptr[16b](0x...) o ((_8,_1),_8,_1,(_1,_3)):((_1,_0),_1024,_0,(_0,_8192))
```

tAgA 第一维 8 是连续的 8 个 half（stride 1），第二维相隔 16 行（stride 16×4096=65536），末维 64 是不同 K tile（stride 64）；tAsA 结构相同，只是行间隔换成 smem 的 1024、stage 间隔 8192。

## 分区与预取

先为 gmem 创建视图，再 local_tile 分块，用动态 `extern __shared__` 申请 smem：

```cpp
  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

  // Shared memory tensors
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{}); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), BSmemLayout{}); // (BLK_N,BLK_K,PIPE)
```

然后分别用 TiledCopy 和 TiledMMA 做分区：

```cpp
  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA_);                           // (CPY,CPY_M,CPY_K,PIPE)

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // descriptor 版本
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
  clear(tCrC);
```

tCsA 是 smem 视图 ((_64,_16),_2,_4,(_1,_3)):((_64,_1),_4096,_16,(_0,_8192))：每个线程（128 个都一样）看到完整的 64×16，再按 MMA_M=2、MMA_K=4 重复。

这来自 partition_A 内部——它先算 thrfrg_A(atensor.layout())，得到每个线程的 A 片段视图：

```cpp
template <class ATensor>
CUTE_HOST_DEVICE constexpr
auto
partition_A(ATensor&& atensor) const
{
  auto thr_tensor = make_tensor(static_cast<ATensor&&>(atensor).data(), this->thrfrg_A(atensor.layout()));
  ...
  auto thr_vmk = make_coord(get<0>(thr_vmnk_), make_coord(get<1>(thr_vmnk_), get<3>(thr_vmnk_)));
  return thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
}
```

sA 的 layout 是 ((_8,16),(_64,1),(_1,3)):((_64,_512),(_1,_0),(_0,_8192))。LayoutA_TV = (_128,(_64,_16)):(_0,(_1,_64))，第一维 128 线程 stride 为 0，表示 128 个线程看到的是同一份完整 64×16（因为从 smem 读）。thrfrg_A 的结果是：

```
((_128,(_1,_1)),((_64,_16),(_2,_4,(_1,_3)))):((_0,(_0,_0)),((_64,_1),(_4096,_16,(_0,_8192))))
```

第一维是线程、第二维是 value；一个线程看到完整的 64×16，而 64×16 的 atom 把 sA 的 128×64 切成 2×4 份。用当前线程索引第一维后，就是上面的 tCsA，stride (64,1) 是因为行间隔 64 列、K 方向 stride 1。

也就是说 128×64 的 sA 被 64×16 的 atom 切成 2×4 = 8 块，每次 wgmma 只有 128 个线程共同处理其中一块。

tCrA 则把这 8 个块变成 8 个 smem 描述符。

预取前 K_PIPE_MAX-1 = 2 个 stage：

```cpp
  auto K_TILE_MAX = size<3>(tAgA);
  auto K_PIPE_MAX = size<3>(tAsA);

  CUTE_UNROLL
  for (int k = 0; k < K_PIPE_MAX-1; ++k) {
    copy(copy_a, tAgA(_,_,_,k), tAsA(_,_,_,k));
    copy(copy_b, tBgB(_,_,_,k), tBsB(_,_,_,k));
    cp_async_fence();
  }

  clear(tCrC);
  __syncthreads();
```

和 SM80 一样，cp_async_fence() 即 cp.async.commit_group，把已发射的异步拷贝打包成一个 group。

## 主循环

```cpp
  int k_pipe_read  = 0;
  int k_pipe_write = K_PIPE_MAX-1;

  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    int k_tile_next = k_tile + (K_PIPE_MAX-1);
    k_tile_next = (k_tile_next >= K_TILE_MAX) ? K_TILE_MAX-1 : k_tile_next;

    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe_write));
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe_write));
    cp_async_fence();

    ++k_pipe_write;
    k_pipe_write = (k_pipe_write == K_PIPE_MAX) ? 0 : k_pipe_write;

    cp_async_wait<0>();

    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    cute::gemm(mma, tCrA(_,_,_,k_pipe_read), tCrB(_,_,_,k_pipe_read), tCrC);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);

    ++k_pipe_read;
    k_pipe_read = (k_pipe_read == K_PIPE_MAX) ? 0 : k_pipe_read;
  }

  axpby(alpha, tCrC, beta, tCgC);
```

`cute::gemm(mma, tCrA(_,_,_,k_pipe_read), ...)` 会自动在 MMA_M=2、MMA_N=2、MMA_K=4 上展开，对每个 (m,n,k) 块发出一次 wgmma——一个 stage 就是 2×2×4 = 16 条 wgmma。

这个展开经过两层 gemm 重载。外层先在 K 方向循环：

```cpp
CUTE_HOST_DEVICE
void
gemm(MMA_Atom<MMA>       const& mma,
     Tensor<TD, DLayout>      & D,  // (V,M,N) Logical data
     Tensor<TA, ALayout> const& A,  // (V,M,K) Logical data
     Tensor<TB, BLayout> const& B,  // (V,N,K) Logical data
     Tensor<TC, CLayout> const& C)  // (V,M,N) Logical data
{
  CUTE_STATIC_ASSERT_V(size<1>(A) == size<1>(C));  // AM == CM
  CUTE_STATIC_ASSERT_V(size<1>(B) == size<2>(C));  // BN == CN
  CUTE_STATIC_ASSERT_V(size<2>(A) == size<2>(B));  // AK == BK
  CUTE_STATIC_ASSERT_V(size<0>(C) == size<0>(D) && size<1>(C) == size<1>(D) && size<2>(C) == size<2>(D));
  auto K = size<2>(A);

  CUTE_UNROLL
  for (int k = 0; k < K; ++k) {
    gemm(mma, D, A(_,_,k), B(_,_,k), C);
  }
}
```

再进到在 M、N 方向上循环的那层。这里对 N 做了蛇形（serpentine）遍历来复用寄存器：当 (V) 维正好是 8 字节（这里是 size<0>(A)*sizeof == 8）时走 64-bit 特化，按行优先蛇形走，m 为奇数时 n 反向：

```cpp
// REGISTER .reuse OPTIMIZATIONS
// 64-bit traversal specialization -- serpentine path
if constexpr (decltype(size<0>(A))::value * sizeof(typename TA::value_type) == 8 &&
              decltype(size<0>(B))::value * sizeof(typename TB::value_type) == 8)
{
  // Row-major serpentine iteration
  CUTE_UNROLL
  for (int m = 0; m < M; ++m) {
    CUTE_UNROLL
    for (int n = 0; n < N; ++n) {
      int ns = (m & 1) ? N-1-n : n;  // Serpentine coordinate
      gemm(mma, D(_,m,ns), A(_,m), B(_,ns), C(_,m,ns));
    }
  }
}
```

最内层才真正调 mma.call(D, A, B, C)，发出单条 wgmma。所以在 M=2、N=2、K=4 的块上，一个 stage 一共 2×2×4 = 16 次 atom 调用，每次一条 wgmma。

其中：

- warpgroup_arrive() = wgmma.fence.sync.aligned。在 wgmma 之前发一条"围栏"，保证此前对寄存器/描述符的写入对异步的 wgmma 可见。wgmma 是异步指令，需要这个 fence 来划分"普通代码" 和 "异步批次"。
- warpgroup_commit_batch() = wgmma.commit_group.sync.aligned。把到目前为止发出的 wgmma 打包成一个 group。
- warpgroup_wait<0>() = wgmma.wait_group.sync.aligned 0。等待"最多 0 个 group 未完成"，即等本 group 全部算完。这里 wait<0> 是因为下一轮要复用 k_pipe_read 这个 stage 的 smem（被 k_pipe_write 环形覆盖），必须确保本轮的 wgmma 已经把这块 smem 读完了。
- warpgroup_fence_operand(tCrC) 是个编译器屏障，不是硬件同步。它对 tCrC 里每个寄存器发一条空的 asm volatile("" : "+f"(reg) :: "memory")，告诉编译器"这些寄存器可能被外面的东西（异步 wgmma）改动，别乱重排、别假设它们的值"。wgmma 在硬件上异步地写累加器，编译器并不知道，这个 fence 防止编译器把 tCrC 的读写优化到 wgmma 的前面或后面。


代码注释写着 "optimize by pipelining to overlap GMEM reads"，但看实际指令顺序会发现：本轮先发射 k_tile_next 的 cp.async 到 k_pipe_write，紧接着 cp_async_wait<0>()——它会等待所有未完成的 cp.async，包括刚发射的这一批。于是"灌下一块"和"算当前块"之间被硬生生串起来，并没有重叠。

这也是这一版性能上不去的原因。在 H200 上跑（M=N=5120、K=4096、half）大约 290 TFlop/s，MFU 只有 29.3%；若把 swizzle 关掉换成不 swizzle 的布局，只剩 108 TFlop/s、MFU 10.95%——swizzle 对消除 bank conflict 的作用很明显，但流水本身还没跑满。

正确做法应当是让 cp_async_wait 只等"当前要算的那个 stage"，允许更晚的 stage 仍在飞，例如 cp_async_wait<K_PIPE_MAX-2>()（等 1 个未完成）或按 stage 精细计数；再配合 `wgmma.wait_group<N>` 只等够用的批次，就能让 gmem 读取、smem→计算真正并行。具体到三段流水，理想顺序是：

1. 开头两次 load，预取前两个 buffer。
2. 进循环后先 load 第三个 buffer。
3. cp.async wait<1> 确保前两个 buffer 加载完成，只留第三个在飞。
4. 对前两个 buffer 发 wgmma。
5. wgmma wait<1> 确保第一个 batch 完成，释放对应 smem。
6. 再 load buffer 0，如此环形推进。

这也正是后续 gemm_sm90_tma 用 TMA + mbarrier 重构流水的动机——把"等哪一块"交给硬件屏障计数，而不是粗暴的 wait<0>。


## 小结

gemm_sm90 相比 SM80 的升级集中在：

1. wgmma 指令：一条指令由 128 线程的 warpgroup 协同，算 64×64×16，A、B 直接从 smem 读（SS）。
2. GMMA 描述符：A、B 从"寄存器里的一堆 half"变成"一个 64-bit 描述符"，编码了 smem 地址、swizzle 类型和 LBO/SBO；DescriptorIterator 让对 tensor 的索引变成描述符地址的偏移。
3. `GMMA::Layout_*_SW*_Atom`：标准化的 smem swizzle atom，直接满足 wgmma 对 canonical 布局的要求。
4. warpgroup 同步原语：`warpgroup_arrive/commit_batch/wait<N>` 管理异步 wgmma 的批次，warpgroup_fence_operand 是防止编译器重排的寄存器屏障。

