---
title: CuTe 学习笔记（十二）GEMM SM90
date: 2025-04-22 18:00:00
tags: [CUTLASS, GEMM, SM90, GPU]
categories: [Cutlass 学习笔记]
description: SM90 架构下使用 wgmma 指令实现 GEMM 矩阵乘法，分析 gemm_tn 的完整实现流程。
---

# wgmma_sm90

```cpp
nvcc wgmma_sm90.cu -o wgmma -I ../../../../include/ -I ../../../../tools/util/include/ -arch=sm_90a -std=c++17
```

矩阵乘的规模 M = N = 5120，K = 4096。

数据类型全是 half，矩阵 A 选择 row-major，矩阵 B 选择 column-major。

直接进入 gemm_tn。

## gemm_tn

```cpp
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K); // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{}); // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{}); // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};                      // Pipeline
```

前面还是一样的定义，其中 CTA 的大小分别是 bM=128，bN=128，bK=64。pipeline = 3。但是定义 shared memory 的 layout 时有了变化。

```cpp
  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN, bK, bP));
```

sA 的 shared memory 的大小还是（bM，bN，bK）。但是使用了 GMMA::Layout_K_SW128_Atom<TA>{}这个 layout 进行 swizzle，来减少 bank conflict。

Layout_K_SW128_Atom 中的 K 表示在 K 方向上进行 swizzle，SW128 表示使用 128B 的 swizzle pattern。这个 layout 定义在 cutlass-4.1/include/cute/atom/mma_traits_sm90_gmma.hpp 文件中。

在这个文件中定义了几个基础的 swizzle pattern。分别是 MN 方向上的和 K 方向上的。每个方向有 4 中 pattern，分别对应 wgmma 支持的 4 中 pattern。

```cpp
// M|N-major GMMA layouts in units of bits
using Layout_MN_INTER_Atom_Bits = ComposedLayout<Swizzle<0,4,3>, smem_ptr_flag, Layout<Shape< _128,_8>,Stride<_1, _128>>>;
using Layout_MN_SW32_Atom_Bits  = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape< _256,_8>,Stride<_1, _256>>>;
using Layout_MN_SW64_Atom_Bits  = ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag, Layout<Shape< _512,_8>,Stride<_1, _512>>>;
using Layout_MN_SW128_Atom_Bits = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_1024,_8>,Stride<_1,_1024>>>;

// K-major GMMA layouts in units of bits
using Layout_K_INTER_Atom_Bits  = ComposedLayout<Swizzle<0,4,3>, smem_ptr_flag, Layout<Shape<_8, _128>,Stride< _128,_1>>>;
using Layout_K_SW32_Atom_Bits   = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape<_8, _256>,Stride< _256,_1>>>;
using Layout_K_SW64_Atom_Bits   = ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag, Layout<Shape<_8, _512>,Stride< _512,_1>>>;
using Layout_K_SW128_Atom_Bits  = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_8,_1024>,Stride<_1024,_1>>>;
```

以 K-major 为例。

Layout_K_INTER_Atom_Bits 表示不进行 swizzle，所以 Swizzle<0,4,3>中的第一个元素是 0，表示在 1 行上进行 swizzle，等于就是没有 swizzle。这里 4 表示 2^4 = 16 个元素作为 swizzle 的基本元素。128bit 正好对应 16 个字节。3 表示 swizzle 有 8 列。Layout<Shape<_8, _128>,Stride< _128,_1>>的 shape 是 8 行 128bits，这是没有 swizzle 的基本 shape。stride 表示在 K 方向上连续。

同理 Layout_K_SW32_Atom_Bits 表示使用 32B 作为基本的 swizzle pattern，基本的 layout 是 8 行 256bits，如果一个元素是 2 字节，layout 就是 8 行 16 列，由于数据在内存中是连续的，也可以视为 2 行 64 列。其中 Swizzle<1,4,3>的 3 表示 8 列元素进行 swizzle，所以进行 swizzle 的 1 列有 8 个元素。1 表示 2 行一组进行 swizzle。

Layout_K_SW128_Atom_Bits 表示使用 128B 作为基本的 swizzle pattern，layout 是 8 行 1024bits。如果一个元素是 4 字节，layout 就是 8 行 32 列。其中 Swizzle<3,4,3>的第一个 3 表示 8 行为一组进行 swizzle，第二个 3 表示 swizzle 有 8 列，其中一列 4 个元素。

然后 bits atom 会根据实际的数据类型重新进行 recast。

```cpp
// M|N-major layouts in units of Type
template <class Type>
using Layout_MN_INTER_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_INTER_Atom_Bits{}));
template <class Type>
using Layout_MN_SW32_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_SW32_Atom_Bits{}));
template <class Type>
using Layout_MN_SW64_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_SW64_Atom_Bits{}));
template <class Type>
using Layout_MN_SW128_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_SW128_Atom_Bits{}));

// K-major layouts in units of Type
template <class Type>
using Layout_K_INTER_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_INTER_Atom_Bits{}));
template <class Type>
using Layout_K_SW32_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW32_Atom_Bits{}));
template <class Type>
using Layout_K_SW64_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW64_Atom_Bits{}));
template <class Type>
using Layout_K_SW128_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW128_Atom_Bits{}));
```

比如以 Layout_K_SW128_Atom_Bits 和 half 为例，

Layout_K_SW128_Atom_Bits = Sw<3,4,3> o smem_ptr[1b](https://unset) o (_8,_1024):(_1024,_1)，recast 后

Layout_K_SW128_Atom = Sw<3,4,3> o smem_ptr[16b](https://unset) o (_8,_64):(_64,_1)

把 layout 打印如下，可以看到 layout 中的 index 已经进行了 swizzle。

![](/assets/gemm_sm90/image.png)

回到前面

```cpp
  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN, bK, bP));
```

这两行的意思就是用 Layout_K_SW128_Atom 这个 layout pattern 去对 sA 和 sB 的 shape 进行 tilling。从前面介绍可知，Layout_K_SW128_Atom 的 shape 是 8*64，因此对于 sA，bM=128，可以分成 16 份，bK=64，刚好满足分块要求。其实 bK 如果小于 64 也不会报错，ceil_div 的结果是 1。

tile_to_shape 后的结果是 Sw<3,4,3> o smem_ptr[16b](https://unset) o ((_8,16),(_64,1)):((_64,_512),(_1,8192))，这里因为是 half 类型，Sw 中间的 4 应该变成 3，不知道为什么没变。

shared memory 的 layout 确定了后就可以创建 tiledmma 和 tiledcopy 了

```cpp
  // Define the thread layouts (static)
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
                                    Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // Thr layout 16x8 k-major
                                    Layout<Shape<_1, _8>>{});                 // Val layout  1x8
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
                                    Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // Thr layout 16x8 k-major
                                    Layout<Shape<_1, _8>>{});                 // Val layout  1x8

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
```

其中 Copy_Atom 使用的是 SM80_CP_ASYNC_CACHEALWAYS，也就是 sm80 的 cp.async，uint128_t 说明一次拷贝 16 个字节。Layout<Shape<_16, _8>, Stride<_8, _1>>{}是线程的 layout，Layout<Shape<_1, _8>>{}是一个线程拷贝多少 value。这两个 layout 说明 TiledCopy 有 16 行 8 列的 128 个线程，其中 1 个线程负责 8 个数据。

tiledcopy 如下，中间的计算过程很复杂，反正就是最后得到了一个 tv-layout，通过对 threadIdx 进行索引可以得到线程对应的数据。但是搞不懂为啥右逆得到的是 tv-layout。

```cpp
thr_layout: (_16,_8):(_8,_1)
val_layout: (_1,_8):(_0,_1)
layout_mn : ((_1,_16),(_8,_8)):((_0,_8),(_128,_1))
layout_tv : ((_8,_16),_8):((_128,_1),_16)
tiler     : (_16,_64)

TiledCopy
  Tiler_MN:       (_16,_64)
  TiledLayout_TV: ((_8,_16),_8):((_128,_1),_16)
Copy_Atom
  ThrID:        _1:_0
  ValLayoutSrc: (_1,_8):(_0,_1)
  ValLayoutDst: (_1,_8):(_0,_1)
  ValLayoutRef: (_1,_8):(_0,_1)
  ValueType:    16b
```

tiledmma 使用的 atom 是 SM90_64x64x16_F16F16F16_SS，使用 wgmma 指令计算矩阵乘，其中 SS 表示矩阵 A 和矩阵 B 都从 shared memory 中加载数据。64*64*16 表示矩阵 A 的 shape 是 64*16，矩阵 B 的 shape 是 16*64。

```cpp
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x16_F16F16F16_SS = SM90::GMMA::MMA_64x64x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_64,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout< 64, 16>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};
```

上面是对应的 Traits，ValType 没什么好说的，都是 half。wgmma 从 smem 中加载数据的时候需要设置一个 gemm desc 描述符，因此这里矩阵 A 和矩阵 B 都需要描述符。

GEMM 的描述符是在 make_tensor 的时候调用的。看起来是在对 sA 和 sB 进行 partition 的时候调用的。

((_8,_16),(_64,_1),(_1,_3)):((_64,_512),(_1,_0),(_0,_8192))

((_128,(_1,_1)),((_64,_16),(_2,_4,(_1,_3)))):((_0,(_0,_0)),((_64,_1),(_4096,_16,(_0,_8192))))

调用后，make_gmma_desc 传进来的 tensor 是 Sw<3,4,3>_smem_ptr[16b](https://0x7fc400000400) o (_64,_16):(_64,_1)。stride 为什么是 64，1。见下。

```cpp
Tensor u128_tensor = recast<uint128_t const>(tensor);
```

然后 recast 到 128bit 的 tensor，就是一个元素是 128bit。Sw<3,4,3>_smem_ptr[128b](https://0x7fe100000400) o (_64,_2):(_8,_1)

然后创建 desc 对象，GmmaDescriptor desc;

确定 layout 的 type，也就是哪种 swizzle 类型，地址和 offset。

```cpp
  // Result
  GmmaDescriptor desc;

  // Layout type
  constexpr LayoutType LAYOUT_TYPE = layout_type(u128_tensor);
  desc.bitfield.layout_type_ = uint8_t(LAYOUT_TYPE);

  // Start address (4LSB not included)
  uint32_t start_address = cast_smem_ptr_to_uint(raw_pointer_cast(u128_tensor.data()));
  desc.bitfield.start_address_ = static_cast<uint16_t>(start_address >> 4);

  constexpr uint8_t base_offset = 0;
  desc.bitfield.base_offset_ = base_offset;
```

最后确定 LBO 和 SBO

```cpp
  else if constexpr (MajorMode == Major::K)
  {
    /* In units of uint128_t, each GmmaDescriptor Major-K describes a canonical layout of the form
     *
     * LayoutType::INTERLEAVE    : Swizzle<0,4,3> o smem_ptr o ((8,n),2):((1,SBO),LBO)
     * LayoutType::B32           : Swizzle<1,4,3> o smem_ptr o ((8,n),2):((2,SBO),1)
     * LayoutType::B64           : Swizzle<2,4,3> o smem_ptr o ((8,n),2):((4,SBO),1)
     * LayoutType::B128          : Swizzle<3,4,3> o smem_ptr o ((8,n),2):((8,SBO),1)
     */
```

先进行判断，MN 维度必须能被 8 整除，第一维度必须等于 2。为啥必须等于 2 呢，难道是因为 wgmma 的 K 只支持 16 个 fp16，所以 recast 到 128bit 后就只能等于 2？

```cpp
    CUTE_STATIC_ASSERT_V(size<0>(u128_tensor) % Int<8>{} == Int<0>{},          // N|M size
                         "Not a canonical GMMA_K Layout: Expected MN-size multiple of 8.");
    CUTE_STATIC_ASSERT_V(size<1>(u128_tensor) == Int<2>{} || size<1>(u128_tensor) == Int<4>{},      // K   size
                         "Not a canonical GMMA_K Layout: Expected K-size 2 for dense or 4 for sparse (in units of uint128_t).");
```

然后把 tensor 的 layout 和基础 pattern 的大小相除，得到((_8,_8),(_2,_1)):((_8,_64),(_1,_0))。

```cpp
    // Construct the canonical GMMA N Layout with shape ((8,n),(2,1))
    Layout canonical_layout = logical_divide(layout(u128_tensor), Tile<Layout<_8,_1>,Layout<_2,_1>>{}); // ((_8,_8),(_2,_1)):((_8,_64),(_1,_0))
```

最后计算 LBO 和 SBO。stride_00  = 8，expected_stride_00 = W = 8。stride_10 = 1，expected_stride_10 = 1。stride_01 = 64。所以 SBO = 64，LBO = 1。

```cpp
    constexpr uint32_t stride_00 = stride<0,0>(canonical_layout);
    constexpr uint32_t expected_stride_00 = W;
    static_assert(stride_00 == expected_stride_00, "Not a canonical GMMA_K Layout: Expected stride failure.");
    constexpr uint32_t stride_10 = stride<1,0>(canonical_layout);
    constexpr uint32_t expected_stride_10 = (LAYOUT_TYPE == LayoutType::INTERLEAVE) ? stride<1,0>(canonical_layout) : 1;
    static_assert(stride_10 == expected_stride_10, "Not a canonical GMMA_K Layout: Expected stride failure.");

    // stride dimension byte offset and leading dimension byte offset (4LSB not included == uint128_t units)
    constexpr uint32_t stride_01 = stride<0,1>(canonical_layout);

    desc.bitfield.stride_byte_offset_  = stride_01;
    desc.bitfield.leading_byte_offset_ = stride_10;
```

```cpp
TiledMMA
  ThrLayoutVMNK:  (_128,_1,_1,_1):(_1,_0,_0,_0)
  PermutationMNK: (_,_,_)
MMA_Atom
  ThrID:      _128:_1
  Shape_MNK:  (_64,_64,_16)
  LayoutA_TV: (_128,(_64,_16)):(_0,(_1,_64))
  LayoutB_TV: (_128,(_64,_16)):(_0,(_1,_64))
  LayoutC_TV: ((_4,_8,_4),(_2,_2,_8)):((_128,_1,_16),(_64,_8,_512))
```

因为 make_tiled_mma 只有一个 atom 参数，所以 tiledmma 的大小和 atom 的大小一样。

都准备好后启动 kernel，进入 gemm_device。

## gemm_device

首先是分别为 gmem 和 smem 创建 cute tensor。

```cpp
  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

  // Shared memory tensors
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{}); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), BSmemLayout{}); // (BLK_N,BLK_K,PIPE)
```

然后使用 tiledcopy 对矩阵 A 和矩阵 B 进行 partition。

```cpp
  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
  Tensor sA_ = as_position_independent_swizzle_tensor(sA);
  Tensor tAsA = thr_copy_a.partition_D(sA_); // (CPY,CPY_M,CPY_K,PIPE)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
  Tensor sB_ = as_position_independent_swizzle_tensor(sB);
  Tensor tBsB = thr_copy_b.partition_D(sB_); // (CPY,CPY_N,CPY_K,PIPE)
```

as_position_independent_swizzle_tensor 这个函数的作用主要是改变 swizzle 中 M 的数值。

sA = Sw<3,4,3>_smem_ptr[16b](https://0x7f0000000400) o ((_8,_16),(_64,_1),(_1,_3)):((_64,_512),(_1,_0),(_0,_8192))

sA_ = smem_ptr[16b](https://0x7f0000000400) o Sw<3,3,3> o _0 o ((_8,_16),(_64,_1),(_1,_3)):((_64,_512),(_1,_0),(_0,_8192))

可以看到，指针和 layout 都相同，只是 sw 的 M 从 4 变成了 3，符合 16bit 的值。

此时

tAgA = gmem_ptr[16b](https://0x7f181a000000) o ((_8,_1),_8,_1,64):((_1,_0),65536,_0,_64)

tAsA = smem_ptr[16b](https://0x7f1900000400) o ((_8,_1),_8,_1,(_1,_3)):((_1,_0),_1024,_0,(_0,_8192))

然后就是使用 tiledMMA 对 sA，sB 和 gC 进行分区。而且创建 gemm desc 看起来是在 make_fragment_A 和 make_fragment_B 中进行的。

```cpp
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.make_fragment_A(tCsA); // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB); // (MMA,MMA_N,MMA_K,PIPE)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_M,MMA_N)
```

先看一下 partition_A 是怎么做的。

```cpp
  template <class ATensor>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_A(ATensor&& atensor) const
  {
    auto thr_tensor = make_tensor(static_cast<ATensor&&>(atensor).data(), this->thrfrg_A(atensor.layout()));

    if (thread0()) {
      print("---\n");
      print(atensor.layout());print("\n");
      print(this->thrfrg_A(atensor.layout()));print("\n");
    }

    auto thr_vmk = make_coord(get<0>(thr_vmnk_), make_coord(get<1>(thr_vmnk_), get<3>(thr_vmnk_)));
    return thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  }
```

从上面知道 sA 的 layout 是 ((_8,_16),(_64,_1),(_1,_3)):((_64,_512),(_1,_0),(_0,_8192))。

LayoutA_TV = (_128,(_64,_16)):(_0,(_1,_64))，表示 128 个线程每个线程都能看到完整的 64*16 个数据，这是因为在 smem 中读取的。

thrfrg_A(atensor.layout())的结果是((_128,(_1,_1)),((_64,_16),(_2,_4,(_1,_3)))):((_0,(_0,_0)),((_64,_1),(_4096,_16,(_0,_8192))))。其中第一维是线程的 layout 为 128，第二维是 value 的 layout，一个线程可以看到完整的 64*16，且 64*16 把 sA 的 128*64 分成了 2*4 份。

使用当前线程对第一维进行索引后得到 tCsA = Sw<3,4,3>_smem_ptr[16b](https://0x7f9a00000400) o ((_64,_16),_2,_4,(_1,_3)):((_64,_1),_4096,_16,(_0,_8192))。这里 stride（64，1）是因为行之间的 stride 是 64，因为 sA 有 64 列，1 是因为 k-major。

然后使用 make_fragment_A 申请寄存器空间。

```cpp
  template <class ATensor>
  CUTE_HOST_DEVICE static constexpr
  auto
  make_fragment_A(ATensor&& atensor)
  {
    // Check that this tensor is likely already partitioned
    CUTE_STATIC_ASSERT_V(rank(atensor) >= Int<3>{});  // VMK
    CUTE_STATIC_ASSERT_V(size<0>(atensor) == size<1>(LayoutA_TV{}));

    if constexpr (has_dereference<FrgTypeA>::value) {
      // If the intended FrgTypeA is a view (of the current tensor), forward the whole
      static_assert(is_same<ValTypeA, typename remove_cvref_t<ATensor>::value_type>::value
                        || (sizeof_bits_v<typename remove_cvref_t<ATensor>::value_type> == 8 &&
                            (sizeof_bits_v<ValTypeA> == 8 || sizeof_bits_v<ValTypeA> == 6 || sizeof_bits_v<ValTypeA> == 4))
                        || (sizeof_bits_v<typename remove_cvref_t<ATensor>::value_type> == 4 &&
                            (sizeof_bits_v<ValTypeA> == 4 || sizeof_bits_v<ValTypeA> == 3 || sizeof_bits_v<ValTypeA> == 2))
                      , "Expecting ValTypeA type");
      return make_tensor<FrgTypeA>(static_cast<ATensor&&>(atensor));
    } else {
      // Else, the intended FrgTypeA is a value type, construct a new tensor with a fragment layout
      return make_fragment_like<FrgTypeA>(atensor);
    }

    CUTE_GCC_UNREACHABLE;
  }
```

这里跟不是直接在寄存器上申请空间，而是走的 make_tensor<FrgTypeA>(static_cast<ATensor&&>(atensor))，会创建一个 gemm desc。

反正看不懂为什么就跑到 cutlass-4.1/include/cute/atom/mma_traits_sm90_gmma.hpp 下的 MakeTensor 了。

```cpp
// Customization point for creating a GMMA::smem_desc Tensor
template <SM90::GMMA::Major MajorMode>
struct MakeTensor<SM90::GMMA::smem_desc<MajorMode>>
{
  template <class TEngine, class TLayout>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Tensor<TEngine,TLayout> const& smem_tensor)
  {
    static_assert(is_smem<TEngine>::value, "Expected SMEM Tensor to construct a GMMA Desc Tensor");
    return make_tensor(SM90::GMMA::DescriptorIterator{SM90::GMMA::make_gmma_desc<MajorMode>(tensor<0>(smem_tensor))},
                       replace<0>(recast<uint128_t const>(smem_tensor).layout(), Layout<_1,_0>{}));
  }
};
```

然后会调用 make_gmma_desc 函数，tensor<0>(smem_tensor)的结果就是 Sw<3,4,3>_smem_ptr[16b](https://0x7fc400000400) o (_64,_16):(_64,_1)。

然后 make_gmma_desc 看上面的解析。是怎么创建 LBO 和 SBO 的。

看起来所有线程传进去的参数都一样，LAYOUT_TYPE 一样，start_address 是 sA 的起始地址。base_offset = 0。

tensor = Sw<3,4,3>_smem_ptr[16b](https://0x7f4000000400) o (_64,_16):(_64,_1)

u128_tensor = Sw<3,4,3>_smem_ptr[128b](https://0x7f4000000400) o (_64,_2):(_8,_1)

然后计算 canonical_layout，得到 ((_8,_8),(_2,_1)):((_8,_64),(_1,_0))

```cpp
Layout canonical_layout = logical_divide(layout(u128_tensor), Tile<Layout<_8,_1>,Layout<_2,_1>>{});
```

最后得到 stride_00 = 8，stride_10 = 1，stride_01 = 64。

desc.bitfield.stride_byte_offset_  = stride_01 = 64。

desc.bitfield.leading_byte_offset_ = stride_10 = 1。

创建完成后，使用 Layout<_1,_0>{}把第一维替换了。

tCrA : GMMA::DescriptorIterator o (_1,_2,_4,(_1,_3)):(_0,_512,_2,(_0,_1024))

tCrB : GMMA::DescriptorIterator o (_1,_2,_4,(_1,_3)):(_0,_512,_2,(_0,_1024))

tCrC : ptr[16b](https://0x7f3f59fffb60) o ((_2,_2,_8),_2,_2):((_1,_2,_4),_32,_64)

使用 wgmma 指令计算的时候会根据 gemm desc 寄存器来获取数据的范围。

此时对于 sA 来说，128*64 的大小被 64*16 分成 2*4 块，所有的 128 个线程都处理同一块的数据计算。但是是怎么区分当前计算的是哪一块的。

下面直接调用的是 gemm，此时 AB 是 2*4 块所有的数据，C 是 2*2 块所有的数据。

```cpp
cute::gemm(mma, tCrA(_, _, _, k_pipe_read), tCrB(_, _, _, k_pipe_read), tCrC);
```

这个函数会进入下面的代码：通过修改循环的次序实现寄存器的重用。

```cpp
CUTE_HOST_DEVICE
void
gemm(MMA_Atom<MMA>       const& mma,
     Tensor<TD, DLayout>      & D,  // (V,M,N) Logical data
     Tensor<TA, ALayout> const& A,  // (V,M)   Logical data
     Tensor<TB, BLayout> const& B,  // (V,N)   Logical data
     Tensor<TC, CLayout> const& C)  // (V,M,N) Logical data
{
  CUTE_STATIC_ASSERT_V(size<1>(A) == size<1>(C));  // AM == CM
  CUTE_STATIC_ASSERT_V(size<1>(B) == size<2>(C));  // BN == CN
  CUTE_STATIC_ASSERT_V(size<0>(C) == size<0>(D) && size<1>(C) == size<1>(D) && size<2>(C) == size<2>(D));
  auto M = size<1>(A);
  auto N = size<1>(B);
  // REGISTER .reuse OPTIMIZATIONS
  // 64-bit traversal specialization -- serpentine path
  if constexpr (decltype(size<0>(A))::value * sizeof(typename TA::value_type) == 8 &&
                decltype(size<0>(B))::value * sizeof(typename TB::value_type) == 8)
  {
#if 1 // NOTE: Row- vs Col- major could depend on the C-matrix order... (which we can test)
    // Row-major serpentine iteration
    CUTE_UNROLL
    for (int m = 0; m < M; ++m) {
      CUTE_UNROLL
      for (int n = 0; n < N; ++n) {
        int ns = (m & 1) ? N-1-n : n;  // Serpentine coordinate
        gemm(mma, D(_,m,ns), A(_,m), B(_,ns), C(_,m,ns));
      }
    }
#else
    // Col-major serpentine iteration
    CUTE_UNROLL
    for (int n = 0; n < N; ++n) {
      CUTE_UNROLL
      for (int m = 0; m < M; ++m) {
        int ms = (n & 1) ? M-1-m : m;  // Serpentine coordinate
        gemm(mma, D(_,ms,n), A(_,ms), B(_,n), C(_,ms,n));
      }
    }
#endif
```

然后调用到下面的 GEMM

```cpp
CUTE_HOST_DEVICE
void
gemm(MMA_Atom<MMA>       const& mma,
     Tensor<TD, DLayout>      & D,  // (V) Logical data
     Tensor<TA, ALayout> const& A,  // (V) Logical data
     Tensor<TB, BLayout> const& B,  // (V) Logical data
     Tensor<TC, CLayout> const& C)  // (V) Logical data
{
  // No static assertions on (V), MMA checks compatibility
  if (thread0()) {
    print("*******\n");
    print(D);print("\n");
    print(A);print("\n");
    print(B);print("\n");
    print(C);print("\n");
  }
  mma.call(D, A, B, C);
}
```

其中：

tCrA : GMMA::DescriptorIterator o (_1,_2,_4,(_1,_3)):(_0,_512,_2,(_0,_1024))

tCrB : GMMA::DescriptorIterator o (_1,_2,_4,(_1,_3)):(_0,_512,_2,(_0,_1024))

tCrC : ptr[16b](https://0x7f3f59fffb60) o ((_2,_2,_8),_2,_2):((_1,_2,_4),_32,_64)

```cpp
-------
ptr[16b](0x7f574dfffb60) o ((_2,_2,_8),_2,_2):((_1,_2,_4),_32,_64)
GMMA::DescriptorIterator o (_1,_2):(_0,_512)
GMMA::DescriptorIterator o (_1,_2):(_0,_512)
ptr[16b](0x7f574dfffb60) o ((_2,_2,_8),_2,_2):((_1,_2,_4),_32,_64)
*******
ptr[16b](0x7f574dfffb60) o ((_2,_2,_8)):((_1,_2,_4))
GMMA::DescriptorIterator o (_1):(_0)
GMMA::DescriptorIterator o (_1):(_0)
ptr[16b](0x7f574dfffb60) o ((_2,_2,_8)):((_1,_2,_4))
*******
ptr[16b](0x7f574dfffbe0) o ((_2,_2,_8)):((_1,_2,_4))
GMMA::DescriptorIterator o (_1):(_0)
GMMA::DescriptorIterator o (_1):(_0)
ptr[16b](0x7f574dfffbe0) o ((_2,_2,_8)):((_1,_2,_4))
*******
ptr[16b](0x7f574dfffc20) o ((_2,_2,_8)):((_1,_2,_4))
GMMA::DescriptorIterator o (_1):(_0)
GMMA::DescriptorIterator o (_1):(_0)
ptr[16b](0x7f574dfffc20) o ((_2,_2,_8)):((_1,_2,_4))
*******
ptr[16b](0x7f574dfffba0) o ((_2,_2,_8)):((_1,_2,_4))
GMMA::DescriptorIterator o (_1):(_0)
GMMA::DescriptorIterator o (_1):(_0)
ptr[16b](0x7f574dfffba0) o ((_2,_2,_8)):((_1,_2,_4))
```

感觉不太对啊，tCrA 是 GMMA::DescriptorIterator o (_1,_2,_4,(_1,_3)):(_0,_512,_2,(_0,_1024))，tCrA(_, _, _, 0) = GMMA::DescriptorIterator o (_1,_2,_4):(_0,_512,_2)。

print_tensor(tCrA(_, _, _, 0))如下：可以看到 2*4 块的每个块的 desc 都是不一样的。

```cpp
GMMA::DescriptorIterator o (_1,_2,_4):(_0,_512,_2):
  GmmaDescriptor: 0x4000004000010040
  start_addr :  0x0040
  leading_off:  0x0001 (1)
  stride_off :  0x0040 (64)
  base_offset:  0x0
  layout_type:  0x1 (B128)
  GmmaDescriptor: 0x4000004000010240
  start_addr :  0x0240
  leading_off:  0x0001 (1)
  stride_off :  0x0040 (64)
  base_offset:  0x0
  layout_type:  0x1 (B128)

----------
  GmmaDescriptor: 0x4000004000010042
  start_addr :  0x0042
  leading_off:  0x0001 (1)
  stride_off :  0x0040 (64)
  base_offset:  0x0
  layout_type:  0x1 (B128)
  GmmaDescriptor: 0x4000004000010242
  start_addr :  0x0242
  leading_off:  0x0001 (1)
  stride_off :  0x0040 (64)
  base_offset:  0x0
  layout_type:  0x1 (B128)

----------
  GmmaDescriptor: 0x4000004000010044
  start_addr :  0x0044
  leading_off:  0x0001 (1)
  stride_off :  0x0040 (64)
  base_offset:  0x0
  layout_type:  0x1 (B128)
  GmmaDescriptor: 0x4000004000010244
  start_addr :  0x0244
  leading_off:  0x0001 (1)
  stride_off :  0x0040 (64)
  base_offset:  0x0
  layout_type:  0x1 (B128)

----------
  GmmaDescriptor: 0x4000004000010046
  start_addr :  0x0046
  leading_off:  0x0001 (1)
  stride_off :  0x0040 (64)
  base_offset:  0x0
  layout_type:  0x1 (B128)
  GmmaDescriptor: 0x4000004000010246
  start_addr :  0x0246
  leading_off:  0x0001 (1)
  stride_off :  0x0040 (64)
  base_offset:  0x0
  layout_type:  0x1 (B128)
```

重新看下是怎么分块的

首先 sA 是 Sw<3,4,3>_smem_ptr[16b](https://0x7fa500000400) o ((_8,_16),(_64,_1),(_1,_3)):((_64,_512),(_1,_0),(_0,_8192))

然后使用 Tensor tCsA = thr_mma.partition_A(sA);进行分区，得到 tCsA。由于 wgmma 的特殊性，所有的线程共享分区结果。

tCsA = Sw<3,4,3>_smem_ptr[16b](https://0x7f2700000400) o ((_64,_16),_2,_4,(_1,_3)):((_64,_1),_4096,_16,(_0,_8192))

然后使用 Tensor tCrA = thr_mma.make_fragment_A(tCsA);创建 gemm desc。得到 tCrA。

tCrA = GMMA::DescriptorIterator o (_1,_2,_4,(_1,_3)):(_0,_512,_2,(_0,_1024))。

打印结果如上，起始地址是 0x0040，在列方向上每块相隔两个 128bit 元素，行方向上相隔 512 个。所以行方向的地址刚好相差 200。

但是 make_fragment_A 是怎么区分地址的呢。原来是这样的：

在 make_tensor 的时候创建的是一个 DescriptorIterator 类，这个类的起始地址就是 smem 的初始地址。

当对 tensor 进行索引的时候就等于对 DescriptorIterator 进行索引。会在 DescriptorIterator 内对地址进行偏移。

```cpp
struct DescriptorIterator
{
  using reference    = GmmaDescriptor;
  using element_type = GmmaDescriptor;
  using value_type   = GmmaDescriptor;

  GmmaDescriptor desc_;

  // Dereference returns the GmmaDescriptor
  CUTE_HOST_DEVICE constexpr
  reference operator*() const { return desc_; }

  // Advance and return a new GmmaDescriptor
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  reference operator[](Index const& i) const { return *(*this + i); }

  // Return an advanced iterator
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  DescriptorIterator operator+(Index const& offset) const
  {
    // Use 32bit calculation rather than 64 bit calculation as we only update the part of desc
    GmmaDescriptor ret;
    ret.reg32_[0] = desc_.reg32_[0] + uint32_t(offset);
    ret.reg32_[1] = desc_.reg32_[1];
    return { ret };
  }
};
```

```cpp
  if (thread0()) {print(tCrA(_,0,0,0));print("\n");}
  if (thread0()) {print(tCrA(_,1,0,0));print("\n");}
  if (thread0()) {print(tCrA(_,0,1,0));print("\n");}
  if (thread0()) {print(tCrA(_,1,1,0));print("\n");}
++0
GMMA::DescriptorIterator o (_1):(_0)
++512
GMMA::DescriptorIterator o (_1):(_0)
++2
GMMA::DescriptorIterator o (_1):(_0)
++514
GMMA::DescriptorIterator o (_1):(_0)
```

OK，现在知道 wgmma 的描述符是怎么工作的了，接下来需要考虑其他情况。详见：wgmma desc

## mainloop

tensor 都创建完成后就可以进行循环计算了。

因为 pipeline 是 3，所以先提前加载 2 个 buffer 的数据。

```cpp
 // Prefetch all but the last
  CUTE_UNROLL
  for (int k = 0; k < K_PIPE_MAX - 1; ++k)
  {
    copy(copy_a, tAgA(_, _, _, k), tAsA(_, _, _, k));
    copy(copy_b, tBgB(_, _, _, k), tBsB(_, _, _, k));
    cp_async_fence();
  }

  // Clear the accumulators
  clear(tCrC);

  __syncthreads();
```

cp_async_fence 就是 asm volatile("cp.async.commit_group;\n" ::);

异步预加载完成后就可以进入循环了。

```cpp
  //
  // PIPELINED MAIN LOOP
  //

  // Current pipe to read from
  int k_pipe_read = 0;
  // Current pipe to write to
  int k_pipe_write = K_PIPE_MAX - 1;

  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    int k_tile_next = k_tile + (K_PIPE_MAX - 1);
    k_tile_next = (k_tile_next >= K_TILE_MAX) ? K_TILE_MAX - 1 : k_tile_next;

    //
    // Copy gmem to smem for k_tile_write
    //

    copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe_write));
    copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe_write));
    cp_async_fence();

    // Advance k_pipe_write
    ++k_pipe_write;
    k_pipe_write = (k_pipe_write == K_PIPE_MAX) ? 0 : k_pipe_write;

    //
    // Compute on k_tile
    //

    // Wait on all cp.async -- optimize by pipelining to overlap GMEM reads
    cp_async_wait<0>();

    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    // (V,M,K) x (V,N,K) => (V,M,N)
    cute::gemm(mma, tCrA(_, _, _, k_pipe_read), tCrB(_, _, _, k_pipe_read), tCrC);
    warpgroup_commit_batch();
    /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);

    // Advance k_pipe_read
    ++k_pipe_read;
    k_pipe_read = (k_pipe_read == K_PIPE_MAX) ? 0 : k_pipe_read;
  }
```

进入循环先异步加载数据到第三个 buffer 上。

然后更新 k_pipe_write 到 0。

然后是 cp_async_wait<0>();，为啥是 0 啊，0 不是要等前面提交的所有异步都完成吗，这样不就串行了吗。

然后 warpgroup_fence_operand，这个是干啥的。

asm volatile("" : "+f"(reg) :: "memory")：这是关键部分

volatile：禁止编译器优化此汇编语句

""：空汇编模板，不生成实际指令

"+f"(reg)：约束条件，表示 reg 变量使用浮点寄存器，且同时作为输入和输出

"memory"：告知编译器内存可能被修改，防止读写操作被重排序

创建一个编译器屏障，防止对 reg 变量的操作被编译器优化重排

确保在此操作之前的所有内存操作对后续代码可见

常用于多线程/并行编程中确保内存一致性

然后是 warpgroup_arrive，这个是 wgmma.fence.sync.aligned，等于是让所有寄存器可用。

然后直接调用 gemm 计算。cute::gemm(mma, tCrA(_, _, _, k_pipe_read), tCrB(_, _, _, k_pipe_read), tCrC);

这个函数会先进到下面的 gemm 里，在 k 方向进行循环。

```cpp
CUTE_HOST_DEVICE
void
gemm(MMA_Atom<MMA>       const& mma,
     Tensor<TD, DLayout>      & D,  // (V,M,N) Logical data
     Tensor<TA, ALayout> const& A,  // (V,M,K) Logical data
     Tensor<TB, BLayout> const& B,  // (V,N,K) Logical data
     Tensor<TC, CLayout> const& C)  // (V,M,N) Logical data
{
  CUTE_STATIC_ASSERT_V(size<1>(A) == size<1>(C));  // AM == CM
  CUTE_STATIC_ASSERT_V(size<1>(B) == size<2>(C));  // BN == CN
  CUTE_STATIC_ASSERT_V(size<2>(A) == size<2>(B));  // AK == BK
  CUTE_STATIC_ASSERT_V(size<0>(C) == size<0>(D) && size<1>(C) == size<1>(D) && size<2>(C) == size<2>(D));
  auto K = size<2>(A);

  CUTE_UNROLL
  for (int k = 0; k < K; ++k) {
    gemm(mma, D, A(_,_,k), B(_,_,k), C);
  }
}
```

然后进到下面的 gemm 里，在 M 和 N 方向进行循环。循环的时候对 N 做了处理，可以重用寄存器。

```cpp
  // REGISTER .reuse OPTIMIZATIONS
  // 64-bit traversal specialization -- serpentine path
  if constexpr (decltype(size<0>(A))::value * sizeof(typename TA::value_type) == 8 &&
                decltype(size<0>(B))::value * sizeof(typename TB::value_type) == 8)
  {
#if 1 // NOTE: Row- vs Col- major could depend on the C-matrix order... (which we can test)
    // Row-major serpentine iteration
    CUTE_UNROLL
    for (int m = 0; m < M; ++m) {
      CUTE_UNROLL
      for (int n = 0; n < N; ++n) {
        int ns = (m & 1) ? N-1-n : n;  // Serpentine coordinate
        gemm(mma, D(_,m,ns), A(_,m), B(_,ns), C(_,m,ns));
      }
    }
```

测试结果：在 H200 上

CUTE_GEMM:     [ 290.1]TFlop/s  (0.9187)ms。

MFU = 29.3%。

[290162.6]GFlop/s  (0.7401)ms

[409406.1]GFlop/s  (0.5245)ms

如果不用 swizzle：

CUTE_GEMM:     [ 108.3]TFlop/s  (1.9830)ms

MFU =  10.95%。

感觉这段代码写的有问题，看起来是加载和计算 overlap，但是都是串行的，因为每次加载和计算后都需要 wait<0>。

![](/assets/gemm_sm90/image(1).png)

应该可以改成

1. 前面两次 load
1. 进入循环，加载第三个 buffer。
1. cp.async wait<1> 确保前两个 buffer 加载完成。
1. wgmma 1, wgmma 2。
1. wgmma wait<1>，确保 wgmma 1 完成。
1. 然后 load buffer 0。
