---
title: CuTe 学习笔记（六）CuTe tilledCOPY
date: 2025-03-23 18:00:00
tags: [CUTLASS, CuTe, Copy]
categories: [Cutlass 学习笔记, CuTe]
description: 深入解析 CuTe tilledCOPY 机制，涵盖 Copy 指令的创建、partition 操作及多种 Copy 实现方式。
published: true
mathjax: true
---

# CuTe_tilledCOPY

# copy op

最基础的 copy 单元，底层调用的是 PTX 指令。根据指令的不同，可以将数据从 global memory 拷贝到 shared memory 或者从 smem 到 rmem。

UniversalCopy，单个线程的赋值操作。

```cpp
template <class S, class D = S>
struct UniversalCopy
{
  using SRegisters = S[1];
  using DRegisters = D[1];

  // Sanity
  static_assert(sizeof_bits_v<S> >= 8);
  static_assert(sizeof_bits_v<D> >= 8);

  CUTE_HOST_DEVICE static constexpr void
  copy(S const& src,
       D      & dst)
  {
    dst = src;
  }
};
```

cp.async 拷贝，异步拷贝，一个线程可以拷贝一个或多个字节。直接将数据从 gmem 拷贝到 smem。

```cpp
template <class TS, class TD = TS>
struct SM80_CP_ASYNC_CACHEALWAYS
{
  using SRegisters = TS[1];
  using DRegisters = TD[1];

  static_assert(sizeof(TS) == sizeof(TD), "cp.async requires sizeof(src_value_type) == sizeof(dst_value_type)");
  static_assert(sizeof(TS) == 4 || sizeof(TS) == 8 || sizeof(TS) == 16, "cp.async sizeof(TS) is not supported");

  CUTE_HOST_DEVICE static void
  copy(TS const& gmem_src,
       TD      & smem_dst)
  {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    TS const* gmem_ptr    = &gmem_src;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
        :: "r"(smem_int_ptr),
           "l"(gmem_ptr),
           "n"(sizeof(TS)));
#else
    CUTE_INVALID_CONTROL_PATH("Support for cp.async instructions has not been enabled");
#endif
  }
};
```

ldmatrix 拷贝，warp 级别的拷贝，从 smem 到 rmem。

```cpp
struct SM75_U32x4_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
  }
};
```

TMA 拷贝，异步拷贝，单个线程可以拷贝大量数据。从 gmem 到 smem。或从 smem 到 gmem。

```cpp
struct SM90_TMA_LOAD_2D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
    asm volatile (
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "l"(cache_hint)
      : "memory");
#else
    asm volatile (
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "l"(cache_hint)
      : "memory");
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  struct PREFETCH
  {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& crd0, int32_t const& crd1)
    {
  #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      asm volatile (
        "cp.async.bulk.prefetch.tensor.2d.L2.global"
        " [%0, {%1, %2}];"
        :
        : "l"(gmem_int_desc),
          "r"(crd0), "r"(crd1)
        : "memory");
  #else
      CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
  #endif
    }
  };
};
```

# copy tratis

在 PTX 拷贝指令上添加一些拷贝时需要的基本信息，比如线程数量，数据类型和分布等。

UniversalCopy，只需要一个线程参与，所以 ThrID 是 1。SrcLayout 和 DstLayout，src 和 dst 的 tv-layout。一个线程对应一个 S 类型的数据。

```cpp
template <class S, class D>
struct Copy_Traits<UniversalCopy<S,D>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};
```

cp.async，异步拷贝 traits。只需要一个线程参与，也是一个线程对应 S 类型的字节数。

```cpp
template <class S, class D>
struct Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<S,D>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};
```

ldmatrix，warp 级别的拷贝，一个指令需要一个 warp 的 32 个线程参与。对于 x4 类型，读取数据时，一个线程对应一行 128bit 的元素，所以 tv-layout 是(32,128)，stride 是(128,1)。

保存数据时，32 个线程交错保存。一个线程保存 32*4 个 bit，所以 layout 是(32,(32,4))。

```cpp
template <>
struct Copy_Traits<SM75_U32x4_LDSM_N>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape < _32,_128>,
                           Stride<_128,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,Shape <_32,   _4>>,
                           Stride<_32,Stride< _1,_1024>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};
```

# copy atom

copy 的最小单位，由 Copy_Traits 组成。

```cpp
template <class... Args, class CopyInternalType>
struct Copy_Atom<Copy_Traits<Args...>, CopyInternalType>
  : Copy_Traits<Args...>
{
  using Traits = Copy_Traits<Args...>;

  // Bit and Thr layouts from the Copy_Traits
  using ThrID        = typename Traits::ThrID;
  using BitLayoutSrc = typename Traits::SrcLayout;
  using BitLayoutDst = typename Traits::DstLayout;
  using BitLayoutRef = typename Traits::RefLayout;

  using ValType = CopyInternalType;

  using ValLayoutSrc = decltype(recast_layout<uint1_t, ValType>(BitLayoutSrc{}));
  using ValLayoutDst = decltype(recast_layout<uint1_t, ValType>(BitLayoutDst{}));
  using ValLayoutRef = decltype(recast_layout<uint1_t, ValType>(BitLayoutRef{}));

  CUTE_STATIC_ASSERT_V(size<0>(ValLayoutSrc{}) == size(ThrID{}), "CopyOperation is not valid for Src of ValType.");
  CUTE_STATIC_ASSERT_V(size<0>(ValLayoutDst{}) == size(ThrID{}), "CopyOperation is not valid for Dst of ValType.");
  CUTE_STATIC_ASSERT_V(size<0>(ValLayoutRef{}) == size(ThrID{}), "CopyOperation is not valid for Ref of ValType.");

  static constexpr int NumValSrc = size<1>(ValLayoutSrc{});
  static constexpr int NumValDst = size<1>(ValLayoutDst{});

  // Additional Trait parameters/transformations
  template <class... TraitsArgs>
  CUTE_HOST_DEVICE
  auto
  with(TraitsArgs&&... args) const {
    auto traits = Traits::with(static_cast<TraitsArgs&&>(args)...);
    return Copy_Atom<decltype(traits), CopyInternalType>{traits};
  }
  ...
};
```

# tiled copy

通过对 Copy_Atom 进行复制组成的 copy 块。需要三个参数，第一个是 Copy_Atom，确定使用哪一种指令进行拷贝。第二个参数是 LayoutCopy_TV，是一种 tv-layout，描述线程和对应元素的坐标之间的关系。通过对线程访问可以得到该线程对应元素在要拷贝 tensor 种的坐标，一般是通过右逆运算得到。ShapeTiler_MN 是 TiledCopy 的大小，也可以理解为坐标空间大小。

```cpp
template <class Copy_Atom,
          class LayoutCopy_TV,  // (tid,vid) -> coord   [Need not be 2D...]
          class ShapeTiler_MN>  // coord space
struct TiledCopy : Copy_Atom
{
  // Layout information from the CopyAtom
  using AtomThrID     = typename Copy_Atom::ThrID;        // thrid -> thr_idx
  using AtomLayoutSrc = typename Copy_Atom::ValLayoutSrc; // (thr,val) -> offset
  using AtomLayoutDst = typename Copy_Atom::ValLayoutDst; // (thr,val) -> offset
  using AtomLayoutRef = typename Copy_Atom::ValLayoutRef; // (thr,val) -> offset

  using AtomNumThr = decltype(size<0>(AtomLayoutRef{}));
  using AtomNumVal = decltype(size<1>(AtomLayoutRef{}));

  // Layout information for the TiledCopy
  using Tiler_MN       = ShapeTiler_MN;
  using TiledLayout_TV = LayoutCopy_TV;
  using TiledNumThr    = decltype(size<0>(TiledLayout_TV{}));
  using TiledNumVal    = decltype(size<1>(TiledLayout_TV{}));
  ...
};
```

## tidfrg_S

对 TiledCopy 在 tensor 上进行 layout 变换，变成 thread layout 在第一维，value 在第二维的 layout。由 partition_S 调用。

```cpp
  // Tile a tensor or a layout from shape
  //   (M,N,...)
  // to shape
  //   ((ThrV,ThrX),FrgV,(RestM,RestN,...))
  // where
  //   ThrV:  The threads local to a COPY_ATOM Src.
  //   ThrX:  The threads tiled across COPY_ATOMs Src.
  //   FrgV:  The values local to a COPY_ATOM Src.
  //   RestM: The values tiled in M.
  //   RestN: The values tiled in N.
  template <class STensor>
  CUTE_HOST_DEVICE constexpr static
  auto
  tidfrg_S(STensor&& stensor)
  {
    CUTE_STATIC_ASSERT_V(rank(stensor) >= rank(Tiler_MN{}), "Rank of tensor to be partitioned too small.");

    // Tile the stensor and compute the (src-thr, src-val) -> (ref-thr, ref-val) layout
    return tile2thrfrg(zipped_divide(stensor,Tiler_MN{}), right_inverse(AtomLayoutRef{}).compose(AtomLayoutSrc{}));
  }
```

zipped_divide(stensor,Tiler_MN{})就是使用 TiledCopy 的大小对 stensor 进行分块，zipped_divide 后一个块中所有的元素在第一维，分成的块的 layout 在第二维。

right_inverse(AtomLayoutRef{}).compose(AtomLayoutSrc{})。AtomLayoutRef{}是线程在 src 上的 tv 布局，通过 right_inverse 运算变成线程和坐标的 tv layout，并 compose 到原 shape 的形状。

举个例子：

以 ldmatrix x4 为例，Traits 中的 layout 如下：加载时按照 srclayout 加载，一个线程加载 128bit 的数据。如果数据类型是 fp16 的话就是一个线程加载 8 个 fp16，所以在 copy atom 中 recast 为 ValLayoutSrc = (_32,_8):(_8,_1)，也就是 tiledcopy 中的 AtomLayoutRef。

```cpp
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape < _32,_128>,
                           Stride<_128,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,Shape <_32,   _4>>,
                           Stride<_32,Stride< _1,_1024>>>;
```

对 AtomLayoutRef 进行右逆运算。得到 right_inverse = (_8,_32):(_32,_1)。然后再与 AtomLayoutSrc 进行组合得到 compose = (_32,_8):(_1,_32)。

下面是这三个 layout 的布局，从图中可以看到，AtomLayoutRef = (_32,_8):(_8,_1) 是线程和数据的布局，一行对应一个线程，一个线程加载 8 个连续的 float16。right_inverse = (_8,_32):(_32,_1) 是 AtomLayoutRef 中对应坐标的布局。compose = (_32,_8):(_1,_32) 是把右逆按照 AtomLayoutRef 的形状重排一下。可以看到此时一个线程对应的元素就是线程在 AtomLayoutRef 布局中的元素的坐标。

```cpp
(_32,_8):(_8,_1)
        0     1     2     3     4     5     6     7
    +-----+-----+-----+-----+-----+-----+-----+-----+
 0  |   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 1  |   8 |   9 |  10 |  11 |  12 |  13 |  14 |  15 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 2  |  16 |  17 |  18 |  19 |  20 |  21 |  22 |  23 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 3  |  24 |  25 |  26 |  27 |  28 |  29 |  30 |  31 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 4  |  32 |  33 |  34 |  35 |  36 |  37 |  38 |  39 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 5  |  40 |  41 |  42 |  43 |  44 |  45 |  46 |  47 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 6  |  48 |  49 |  50 |  51 |  52 |  53 |  54 |  55 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 7  |  56 |  57 |  58 |  59 |  60 |  61 |  62 |  63 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 8  |  64 |  65 |  66 |  67 |  68 |  69 |  70 |  71 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 9  |  72 |  73 |  74 |  75 |  76 |  77 |  78 |  79 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  80 |  81 |  82 |  83 |  84 |  85 |  86 |  87 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  88 |  89 |  90 |  91 |  92 |  93 |  94 |  95 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  96 |  97 |  98 |  99 | 100 | 101 | 102 | 103 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 112 | 113 | 114 | 115 | 116 | 117 | 118 | 119 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 120 | 121 | 122 | 123 | 124 | 125 | 126 | 127 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 128 | 129 | 130 | 131 | 132 | 133 | 134 | 135 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 136 | 137 | 138 | 139 | 140 | 141 | 142 | 143 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 144 | 145 | 146 | 147 | 148 | 149 | 150 | 151 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 152 | 153 | 154 | 155 | 156 | 157 | 158 | 159 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 160 | 161 | 162 | 163 | 164 | 165 | 166 | 167 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 168 | 169 | 170 | 171 | 172 | 173 | 174 | 175 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 176 | 177 | 178 | 179 | 180 | 181 | 182 | 183 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 184 | 185 | 186 | 187 | 188 | 189 | 190 | 191 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 192 | 193 | 194 | 195 | 196 | 197 | 198 | 199 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 200 | 201 | 202 | 203 | 204 | 205 | 206 | 207 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 208 | 209 | 210 | 211 | 212 | 213 | 214 | 215 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 216 | 217 | 218 | 219 | 220 | 221 | 222 | 223 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 224 | 225 | 226 | 227 | 228 | 229 | 230 | 231 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 232 | 233 | 234 | 235 | 236 | 237 | 238 | 239 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 240 | 241 | 242 | 243 | 244 | 245 | 246 | 247 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 248 | 249 | 250 | 251 | 252 | 253 | 254 | 255 |
    +-----+-----+-----+-----+-----+-----+-----+-----+

(_8,_32):(_32,_1)
        0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30    31
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 0  |   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |   8 |   9 |  10 |  11 |  12 |  13 |  14 |  15 |  16 |  17 |  18 |  19 |  20 |  21 |  22 |  23 |  24 |  25 |  26 |  27 |  28 |  29 |  30 |  31 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 1  |  32 |  33 |  34 |  35 |  36 |  37 |  38 |  39 |  40 |  41 |  42 |  43 |  44 |  45 |  46 |  47 |  48 |  49 |  50 |  51 |  52 |  53 |  54 |  55 |  56 |  57 |  58 |  59 |  60 |  61 |  62 |  63 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 2  |  64 |  65 |  66 |  67 |  68 |  69 |  70 |  71 |  72 |  73 |  74 |  75 |  76 |  77 |  78 |  79 |  80 |  81 |  82 |  83 |  84 |  85 |  86 |  87 |  88 |  89 |  90 |  91 |  92 |  93 |  94 |  95 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 3  |  96 |  97 |  98 |  99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 | 116 | 117 | 118 | 119 | 120 | 121 | 122 | 123 | 124 | 125 | 126 | 127 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 4  | 128 | 129 | 130 | 131 | 132 | 133 | 134 | 135 | 136 | 137 | 138 | 139 | 140 | 141 | 142 | 143 | 144 | 145 | 146 | 147 | 148 | 149 | 150 | 151 | 152 | 153 | 154 | 155 | 156 | 157 | 158 | 159 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 5  | 160 | 161 | 162 | 163 | 164 | 165 | 166 | 167 | 168 | 169 | 170 | 171 | 172 | 173 | 174 | 175 | 176 | 177 | 178 | 179 | 180 | 181 | 182 | 183 | 184 | 185 | 186 | 187 | 188 | 189 | 190 | 191 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 6  | 192 | 193 | 194 | 195 | 196 | 197 | 198 | 199 | 200 | 201 | 202 | 203 | 204 | 205 | 206 | 207 | 208 | 209 | 210 | 211 | 212 | 213 | 214 | 215 | 216 | 217 | 218 | 219 | 220 | 221 | 222 | 223 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 7  | 224 | 225 | 226 | 227 | 228 | 229 | 230 | 231 | 232 | 233 | 234 | 235 | 236 | 237 | 238 | 239 | 240 | 241 | 242 | 243 | 244 | 245 | 246 | 247 | 248 | 249 | 250 | 251 | 252 | 253 | 254 | 255 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

(_32,_8):(_1,_32)
        0     1     2     3     4     5     6     7
    +-----+-----+-----+-----+-----+-----+-----+-----+
 0  |   0 |  32 |  64 |  96 | 128 | 160 | 192 | 224 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 1  |   1 |  33 |  65 |  97 | 129 | 161 | 193 | 225 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 2  |   2 |  34 |  66 |  98 | 130 | 162 | 194 | 226 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 3  |   3 |  35 |  67 |  99 | 131 | 163 | 195 | 227 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 4  |   4 |  36 |  68 | 100 | 132 | 164 | 196 | 228 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 5  |   5 |  37 |  69 | 101 | 133 | 165 | 197 | 229 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 6  |   6 |  38 |  70 | 102 | 134 | 166 | 198 | 230 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 7  |   7 |  39 |  71 | 103 | 135 | 167 | 199 | 231 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 8  |   8 |  40 |  72 | 104 | 136 | 168 | 200 | 232 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 9  |   9 |  41 |  73 | 105 | 137 | 169 | 201 | 233 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  10 |  42 |  74 | 106 | 138 | 170 | 202 | 234 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  11 |  43 |  75 | 107 | 139 | 171 | 203 | 235 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  12 |  44 |  76 | 108 | 140 | 172 | 204 | 236 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  13 |  45 |  77 | 109 | 141 | 173 | 205 | 237 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  14 |  46 |  78 | 110 | 142 | 174 | 206 | 238 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  15 |  47 |  79 | 111 | 143 | 175 | 207 | 239 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  16 |  48 |  80 | 112 | 144 | 176 | 208 | 240 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  17 |  49 |  81 | 113 | 145 | 177 | 209 | 241 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  18 |  50 |  82 | 114 | 146 | 178 | 210 | 242 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  19 |  51 |  83 | 115 | 147 | 179 | 211 | 243 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  20 |  52 |  84 | 116 | 148 | 180 | 212 | 244 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  21 |  53 |  85 | 117 | 149 | 181 | 213 | 245 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  22 |  54 |  86 | 118 | 150 | 182 | 214 | 246 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  23 |  55 |  87 | 119 | 151 | 183 | 215 | 247 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  24 |  56 |  88 | 120 | 152 | 184 | 216 | 248 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  25 |  57 |  89 | 121 | 153 | 185 | 217 | 249 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  26 |  58 |  90 | 122 | 154 | 186 | 218 | 250 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  27 |  59 |  91 | 123 | 155 | 187 | 219 | 251 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  28 |  60 |  92 | 124 | 156 | 188 | 220 | 252 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  29 |  61 |  93 | 125 | 157 | 189 | 221 | 253 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  30 |  62 |  94 | 126 | 158 | 190 | 222 | 254 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  31 |  63 |  95 | 127 | 159 | 191 | 223 | 255 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
```

最后将分块的结果和线程和坐标的 tv-layout 送到 tile2thrfrg 中计算。这里为啥要用线程和坐标的 tv-layout

## tidfrg_D

tidfrg_D 和 tidfrg_S 基本相同，区别就是这个函数是对 dst tensor 进行分块的。

```cpp
  // Tile a tensor or a layout from shape
  //   (M,N,...)
  // to shape
  //   ((ThrV,ThrX),FrgV,(RestM,RestN,...))
  // where
  //   ThrV:  The threads local to a COPY_ATOM Dst.
  //   ThrX:  The threads tiled across COPY_ATOMs Dst.
  //   FrgV:  The values local to a COPY_ATOM Dst.
  //   RestM: The values tiled in M.
  //   RestN: The values tiled in N.
  template <class DTensor>
  CUTE_HOST_DEVICE constexpr static
  auto
  tidfrg_D(DTensor&& dtensor)
  {
    CUTE_STATIC_ASSERT_V(rank(dtensor) >= rank(Tiler_MN{}), "Rank of tensor to be partitioned too small.");

    // Tile the dtensor and compute the (dst-thr, dst-val) -> (ref-thr, ref-val) layout
    return tile2thrfrg(zipped_divide(dtensor,Tiler_MN{}), right_inverse(AtomLayoutRef{}).compose(AtomLayoutDst{}));
  }
```

还是以 ldmatrix 为例，DstLayout = Layout<Shape <_32,Shape <_32,   _4>>, Stride<_32,Stride< _1,_1024>>>，float16 类型的话会 recast 成(_32,(_2,_4)):(_2,(_1,_64))，layout 如下所示。

![](/assets/cute-tilled-copy/image.png)

src 的右逆是 (_8,_32):(_32,_1)，将 src 的右逆按照 dst 的 layout 组合得到((_4,_8),(_2,_4)):((_64,_1),(_32,_8))，其中的元素就是对应位置上的元素在 src layout 中的坐标？

```cpp
(_32,(_2,_4)):(_2,(_1,_64))
        0     1     2     3     4     5     6     7
    +-----+-----+-----+-----+-----+-----+-----+-----+
 0  |   0 |   1 |  64 |  65 | 128 | 129 | 192 | 193 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 1  |   2 |   3 |  66 |  67 | 130 | 131 | 194 | 195 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 2  |   4 |   5 |  68 |  69 | 132 | 133 | 196 | 197 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 3  |   6 |   7 |  70 |  71 | 134 | 135 | 198 | 199 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 4  |   8 |   9 |  72 |  73 | 136 | 137 | 200 | 201 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 5  |  10 |  11 |  74 |  75 | 138 | 139 | 202 | 203 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 6  |  12 |  13 |  76 |  77 | 140 | 141 | 204 | 205 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 7  |  14 |  15 |  78 |  79 | 142 | 143 | 206 | 207 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 8  |  16 |  17 |  80 |  81 | 144 | 145 | 208 | 209 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 9  |  18 |  19 |  82 |  83 | 146 | 147 | 210 | 211 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  20 |  21 |  84 |  85 | 148 | 149 | 212 | 213 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  22 |  23 |  86 |  87 | 150 | 151 | 214 | 215 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  24 |  25 |  88 |  89 | 152 | 153 | 216 | 217 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  26 |  27 |  90 |  91 | 154 | 155 | 218 | 219 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  28 |  29 |  92 |  93 | 156 | 157 | 220 | 221 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  30 |  31 |  94 |  95 | 158 | 159 | 222 | 223 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  32 |  33 |  96 |  97 | 160 | 161 | 224 | 225 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  34 |  35 |  98 |  99 | 162 | 163 | 226 | 227 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  36 |  37 | 100 | 101 | 164 | 165 | 228 | 229 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  38 |  39 | 102 | 103 | 166 | 167 | 230 | 231 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  40 |  41 | 104 | 105 | 168 | 169 | 232 | 233 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  42 |  43 | 106 | 107 | 170 | 171 | 234 | 235 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  44 |  45 | 108 | 109 | 172 | 173 | 236 | 237 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  46 |  47 | 110 | 111 | 174 | 175 | 238 | 239 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  48 |  49 | 112 | 113 | 176 | 177 | 240 | 241 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  50 |  51 | 114 | 115 | 178 | 179 | 242 | 243 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  52 |  53 | 116 | 117 | 180 | 181 | 244 | 245 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  54 |  55 | 118 | 119 | 182 | 183 | 246 | 247 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  56 |  57 | 120 | 121 | 184 | 185 | 248 | 249 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  58 |  59 | 122 | 123 | 186 | 187 | 250 | 251 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  60 |  61 | 124 | 125 | 188 | 189 | 252 | 253 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  62 |  63 | 126 | 127 | 190 | 191 | 254 | 255 |
    +-----+-----+-----+-----+-----+-----+-----+-----+

(_8,_32):(_32,_1)
        0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30    31
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 0  |   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |   8 |   9 |  10 |  11 |  12 |  13 |  14 |  15 |  16 |  17 |  18 |  19 |  20 |  21 |  22 |  23 |  24 |  25 |  26 |  27 |  28 |  29 |  30 |  31 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 1  |  32 |  33 |  34 |  35 |  36 |  37 |  38 |  39 |  40 |  41 |  42 |  43 |  44 |  45 |  46 |  47 |  48 |  49 |  50 |  51 |  52 |  53 |  54 |  55 |  56 |  57 |  58 |  59 |  60 |  61 |  62 |  63 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 2  |  64 |  65 |  66 |  67 |  68 |  69 |  70 |  71 |  72 |  73 |  74 |  75 |  76 |  77 |  78 |  79 |  80 |  81 |  82 |  83 |  84 |  85 |  86 |  87 |  88 |  89 |  90 |  91 |  92 |  93 |  94 |  95 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 3  |  96 |  97 |  98 |  99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 | 116 | 117 | 118 | 119 | 120 | 121 | 122 | 123 | 124 | 125 | 126 | 127 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 4  | 128 | 129 | 130 | 131 | 132 | 133 | 134 | 135 | 136 | 137 | 138 | 139 | 140 | 141 | 142 | 143 | 144 | 145 | 146 | 147 | 148 | 149 | 150 | 151 | 152 | 153 | 154 | 155 | 156 | 157 | 158 | 159 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 5  | 160 | 161 | 162 | 163 | 164 | 165 | 166 | 167 | 168 | 169 | 170 | 171 | 172 | 173 | 174 | 175 | 176 | 177 | 178 | 179 | 180 | 181 | 182 | 183 | 184 | 185 | 186 | 187 | 188 | 189 | 190 | 191 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 6  | 192 | 193 | 194 | 195 | 196 | 197 | 198 | 199 | 200 | 201 | 202 | 203 | 204 | 205 | 206 | 207 | 208 | 209 | 210 | 211 | 212 | 213 | 214 | 215 | 216 | 217 | 218 | 219 | 220 | 221 | 222 | 223 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 7  | 224 | 225 | 226 | 227 | 228 | 229 | 230 | 231 | 232 | 233 | 234 | 235 | 236 | 237 | 238 | 239 | 240 | 241 | 242 | 243 | 244 | 245 | 246 | 247 | 248 | 249 | 250 | 251 | 252 | 253 | 254 | 255 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

((_4,_8),(_2,_4)):((_64,_1),(_32,_8))
        0     1     2     3     4     5     6     7
    +-----+-----+-----+-----+-----+-----+-----+-----+
 0  |   0 |  32 |   8 |  40 |  16 |  48 |  24 |  56 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 1  |  64 |  96 |  72 | 104 |  80 | 112 |  88 | 120 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 2  | 128 | 160 | 136 | 168 | 144 | 176 | 152 | 184 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 3  | 192 | 224 | 200 | 232 | 208 | 240 | 216 | 248 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 4  |   1 |  33 |   9 |  41 |  17 |  49 |  25 |  57 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 5  |  65 |  97 |  73 | 105 |  81 | 113 |  89 | 121 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 6  | 129 | 161 | 137 | 169 | 145 | 177 | 153 | 185 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 7  | 193 | 225 | 201 | 233 | 209 | 241 | 217 | 249 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 8  |   2 |  34 |  10 |  42 |  18 |  50 |  26 |  58 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
 9  |  66 |  98 |  74 | 106 |  82 | 114 |  90 | 122 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 130 | 162 | 138 | 170 | 146 | 178 | 154 | 186 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 194 | 226 | 202 | 234 | 210 | 242 | 218 | 250 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |   3 |  35 |  11 |  43 |  19 |  51 |  27 |  59 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  67 |  99 |  75 | 107 |  83 | 115 |  91 | 123 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 131 | 163 | 139 | 171 | 147 | 179 | 155 | 187 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 195 | 227 | 203 | 235 | 211 | 243 | 219 | 251 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |   4 |  36 |  12 |  44 |  20 |  52 |  28 |  60 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  68 | 100 |  76 | 108 |  84 | 116 |  92 | 124 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 132 | 164 | 140 | 172 | 148 | 180 | 156 | 188 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 196 | 228 | 204 | 236 | 212 | 244 | 220 | 252 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |   5 |  37 |  13 |  45 |  21 |  53 |  29 |  61 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  69 | 101 |  77 | 109 |  85 | 117 |  93 | 125 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 133 | 165 | 141 | 173 | 149 | 181 | 157 | 189 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 197 | 229 | 205 | 237 | 213 | 245 | 221 | 253 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |   6 |  38 |  14 |  46 |  22 |  54 |  30 |  62 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  70 | 102 |  78 | 110 |  86 | 118 |  94 | 126 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 134 | 166 | 142 | 174 | 150 | 182 | 158 | 190 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 198 | 230 | 206 | 238 | 214 | 246 | 222 | 254 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |   7 |  39 |  15 |  47 |  23 |  55 |  31 |  63 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  |  71 | 103 |  79 | 111 |  87 | 119 |  95 | 127 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 135 | 167 | 143 | 175 | 151 | 183 | 159 | 191 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
  | 199 | 231 | 207 | 239 | 215 | 247 | 223 | 255 |
    +-----+-----+-----+-----+-----+-----+-----+-----+
```

tile2thrfrg

使用 copy atom 的 tv-layout 对 tensor 进行 tiling。这里的 tensor 的 layout 已经在前面的函数中被 tiledcopy 给 tiling 过了。

```cpp
  // Tile a tensor or a layout from shape
  //   ((TileM,TileN,...), (RestM,RestN,...))
  // to shape
  //   ((ThrV,ThrX),FrgV,(RestM,RestN,...))
  template <class Tensor, class Ref2TrgLayout>
  CUTE_HOST_DEVICE constexpr static
  auto
  tile2thrfrg(Tensor&& tensor, Ref2TrgLayout const& ref2trg)
  {
    // Take the thrs/vals that the atom is interested in
    // NOTE: Assumes the AtomNumThr are contiguous and identity within TiledThrID
    auto atom_layout_TV = zipped_divide(TiledLayout_TV{}, make_shape(AtomNumThr{}, AtomNumVal{}));
    // ((atom_tid,atom_val),(rest_tid,rest_val)) -> (m,n)

    // Transform to the trg layout
    auto trg_layout_TV = atom_layout_TV.compose(ref2trg, _);
    // ((trg_tid,trg_val),(rest_tid,rest_val)) -> (m,n)

    // Transform the thrs mode from thrid to thr_idx
    // NOTE: Assumes the AtomNumThr are contiguous and identity within TiledThrID
    auto thrval2mn = coalesce(zip(trg_layout_TV), Shape<_1,Shape<_1,_1>>{});
    // ((trg_tid,rest_tid),(trg_val,rest_val)) -> (m,n)

    /// ==================

    // Transform the tile mode
    auto tv_tensor = tensor.compose(thrval2mn, _);
    // ((thrid,val),(RestM,RestN,...))

    // Unfold and return
    return tv_tensor(make_coord(_,_), _);
  }
```

retile

```cpp
  // retile_S and retile_D assume they are working with the reference layout -- they are the same
  template <class Tensor>
  CUTE_HOST_DEVICE constexpr static
  auto
  retile(Tensor&& tensor)
  {
    constexpr int R = remove_cvref_t<Tensor>::rank;
    // Assert that AtomLayoutSrc|Dst is identity so we can skip the Ref transformation

    // Assume the first size<0>(tensor) elements are the first val_ids in TiledLayout_TV.
    // Then, we only need the shape+layout of those size<0>(tensor) elements in TiledLayout_TV
    //   and that shape is what we gather from the other modes of tensor

    auto V = size<0>(tensor);

    auto frg_layout_mn = upcast<TiledNumThr{} * V>(right_inverse(TiledLayout_TV{}).with_shape(shape(Tiler_MN{})));
    // (m,n) -> v_idx -- The shape and order of the V inside of TiledLayout_TV

    auto frg_layout_v = zipped_divide(logical_product(make_layout(V), right_inverse(frg_layout_mn)), make_layout(AtomNumVal{}));
    // (atom_vals,rest_vals) -> (v,m,n)

    /// =======

    // Tile the tensor for TileFrg
    auto t_tensor = zipped_divide(tensor, prepend(product_each(shape(frg_layout_mn)), V));
    // ((TileV,TileM,TileN,...),(1,RestM,RestN,...))

    // Transform the tile mode
    auto v_tensor = t_tensor.compose(frg_layout_v, _);
    // ((atom_vals,rest_vals),(1,RM,RN,...))

    // Unfold and return
    return v_tensor(_, append<R>(Int<0>{},_));
  }
```

get_layoutS_TV

```cpp
  CUTE_HOST_DEVICE constexpr static
  auto
  get_layoutS_TV()
  {
    // (M,N) -> (M,N)
    auto ref_S = make_layout(make_shape(shape(Tiler_MN{}), Int<1>{}));
    // (thr_idx,val_idx) -> (M,N)
    return tile2thrfrg(ref_S, right_inverse(AtomLayoutRef{}).compose(AtomLayoutSrc{}))(_,_,Int<0>{});
  }
```

get_layoutD_TV

```cpp
  CUTE_HOST_DEVICE constexpr static
  auto
  get_layoutD_TV()
  {
    // (M,N) -> (M,N)
    auto ref_D = make_layout(make_shape(shape(Tiler_MN{}), Int<1>{}));
    // (thr_idx,val_idx) -> (M,N)
    return tile2thrfrg(ref_D, right_inverse(AtomLayoutRef{}).compose(AtomLayoutDst{}))(_,_,Int<0>{});
  }
```

get_slice

```cpp
  template <class ThrIdx,
            __CUTE_REQUIRES(is_integral<ThrIdx>::value)>
  CUTE_HOST_DEVICE static
  auto
  get_slice(ThrIdx const& thr_idx)
  {
    return ThrCopy<TiledCopy, ThrIdx>(thr_idx);
  }
```

get_thread_slice

```cpp
  template <class ThrIdx,
            __CUTE_REQUIRES(is_integral<ThrIdx>::value)>
  CUTE_HOST_DEVICE  static
  auto
  get_thread_slice(ThrIdx const& thr_idx)
  {
    return get_slice(thr_idx);
  }
```

# thread copy

```cpp
template <class TiledCopy, class ThrIdx>
struct ThrCopy
{
  ThrIdx thr_idx_;

  CUTE_HOST_DEVICE
  ThrCopy(ThrIdx const& thr_idx) : thr_idx_(thr_idx) {}
  ...
```

partition_S

```cpp
  template <class STensor>
  CUTE_HOST_DEVICE
  auto
  partition_S(STensor&& stensor) const {
    //static_assert(sizeof(typename remove_cvref_t<STensor>::value_type) == sizeof(typename TiledCopy::ValType),
    //              "Expected ValType for tiling SrcTensor.");
    auto thr_tensor = make_tensor(static_cast<STensor&&>(stensor).data(), TiledCopy::tidfrg_S(stensor.layout()));
    return thr_tensor(thr_idx_, _, repeat<rank_v<STensor>>(_));
  }

  template <class DTensor>
  CUTE_HOST_DEVICE
  auto
  partition_D(DTensor&& dtensor) const {
    //static_assert(sizeof(typename remove_cvref_t<DTensor>::value_type) == sizeof(typename TiledCopy::ValType),
    //              "Expected ValType for tiling DstTensor.");
    auto thr_tensor = make_tensor(static_cast<DTensor&&>(dtensor).data(), TiledCopy::tidfrg_D(dtensor.layout()));
    return thr_tensor(thr_idx_, _, repeat<rank_v<DTensor>>(_));
  }
```

retile_S

```cpp
  template <class STensor>
  CUTE_HOST_DEVICE static
  auto
  retile_S(STensor&& stensor) {
    // static_assert(sizeof(typename remove_cvref_t<STensor>::value_type) == sizeof(typename TiledCopy::ValType),
    //               "Expected ValType for tiling SrcTensor.");
    return make_tensor(static_cast<STensor&&>(stensor).data(), TiledCopy::retile(stensor.layout()));
  }

  template <class DTensor>
  CUTE_HOST_DEVICE static
  auto
  retile_D(DTensor&& dtensor) {
    // static_assert(sizeof(typename remove_cvref_t<DTensor>::value_type) == sizeof(typename TiledCopy::ValType),
    //               "Expected ValType for tiling DstTensor.");
    return make_tensor(static_cast<DTensor&&>(dtensor).data(), TiledCopy::retile(dtensor.layout()));
  }
```

# make_tiled_copy_impl

创建一个 tiled copy。参数分别是 copy atom，LayoutCopy_TV 和 Tiler。这里的 LayoutCopy_TV 是一个 tv-layout，第一维是 thread 的 layout，第二维是 thread 对应的数据的坐标。通过 threadIdx 可以得到一个线程对应的数据的坐标，通过坐标得到真实数据。

```cpp
template <class... Args,
          class LayoutCopy_TV,
          class Tiler>
CUTE_HOST_DEVICE
auto
make_tiled_copy_impl(Copy_Atom<Args...> const& atom,
                     LayoutCopy_TV      const&,
                     Tiler              const&)
{
  return TiledCopy<Copy_Atom<Args...>, LayoutCopy_TV, Tiler>{atom};
}
```

make_tiled_copy_A

make_tiled_copy_B

make_tiled_copy_C

make_tiled_copy_C_atom

# make_tiled_copy

根据 thread layout 和 value layout 和 copy atom 生成一个 tiledcopy。

第一个参数是 copy atom 表示使用哪种 copy 方式。第二个参数是 thread layout，表示一个 tiledcopy 里有多少线程，以及线程的布局。第二个参数是 value layout，表示一个线程加载的数据的布局。

```cpp
/** Produce a TiledCopy from logical thread and values layouts.
 * The thread and value layouts map coordinates to thr_idx and val_idx.
 *    The product of these layouts is taken to produce the TV layout and the Tiler.
 * Useful when threads and values need very specific mappings onto coordinates
 *    in the target tensors.
 */
template <class... Args,
          class ThrLayout,
          class ValLayout = Layout<_1>>
CUTE_HOST_DEVICE
auto
make_tiled_copy(Copy_Atom<Args...> const& copy_atom,
                ThrLayout          const& thr_layout = {},     // (m,n) -> thr_idx
                ValLayout          const& val_layout = {})     // (m,n) -> val_idx
{
  // Take the raked_products to compute the Layout_MN
  // (M,N) -> (thr_idx, val_idx)
  auto layout_mn = raked_product(thr_layout, val_layout);
  // (thr_idx, val_idx) -> (M,N)
  auto layout_tv = right_inverse(layout_mn).with_shape(make_shape(size(thr_layout), size(val_layout)));
  // Tiler for extracting relevant elements
  // (M,N) -> tensor coord
  auto tiler = product_each(shape(layout_mn));

#if 0
  print("thr_layout: "); print(thr_layout); print("\n");
  print("val_layout: "); print(val_layout); print("\n");
  print("layout_mn : "); print(layout_mn);  print("\n");
  print("layout_tv : "); print(layout_tv);  print("\n");
  print("tiler     : "); print(tiler);      print("\n");
#endif

  return make_tiled_copy_impl(copy_atom, layout_tv, tiler);
}
```

首先将 thr_layout 和 val_layout 进行 raked_product，得到完整的数据 layout，layout_mn，表示在 MN 方向上数据的布局是什么样的。

然后对 layout_mn 进行一个右逆运算。右逆运算得到一个布局，满足 layout(inv_layout(i)) = i 。右逆得到的是一个 flatten 的 layout，然后修改成(size(thr_layout), size(val_layout))的形状。

然后计算 tiledcopy 的大小，也就是 tiler。

最后调用 make_tiled_copy_impl。

什么是右逆。

以下面的代码为例：用 l1 表示线程的 layout，l2 表示 value 的 layout，进行 raked_product 后可以得到 layout rake，如下所示，此时一个线程占据(3,4)大小 block 的数据。

对 rake 进行求逆后得到 rinv。此时满足 i = rake(rinv(i))，可以理解为 rinv 是 rake 的 offset 的一维坐标的 layout。通过坐标 i 对 rinv 进行索引可以得到一个值，这个值就是 rake 中 offset = i 的位置的坐标。

换句话说，如果想得到 rake 中 offset = i 的位置的坐标，只需要计算 rinv(i)就行。

```cpp
auto l1 = Layout<Shape<_2,_5>, Stride<_5,_1>>{};            // (_2,_5):(_5,_1)
auto l2 = Layout<Shape<_3,_4>, Stride<_1,_3>>{};            // (_3,_4):(_1,_3)
auto rake = raked_product(l1, l2);                          // ((_3,_2),(_4,_5)):((_10,_5),(_30,_1))
auto rinv = right_inverse(rake);                            // (_5,_2,_3,_4):(_24,_3,_1,_6)
auto res = rinv.with_shape(make_shape(size(l1), size(l2))); // ((_5,_2),(_3,_4)):((_24,_3),(_1,_6))

// print_layout(rake);
((_3,_2),(_4,_5)):((_10,_5),(_30,_1))
        0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 0  |   0 |  30 |  60 |  90 |   1 |  31 |  61 |  91 |   2 |  32 |  62 |  92 |   3 |  33 |  63 |  93 |   4 |  34 |  64 |  94 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 1  |  10 |  40 |  70 | 100 |  11 |  41 |  71 | 101 |  12 |  42 |  72 | 102 |  13 |  43 |  73 | 103 |  14 |  44 |  74 | 104 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 2  |  20 |  50 |  80 | 110 |  21 |  51 |  81 | 111 |  22 |  52 |  82 | 112 |  23 |  53 |  83 | 113 |  24 |  54 |  84 | 114 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 3  |   5 |  35 |  65 |  95 |   6 |  36 |  66 |  96 |   7 |  37 |  67 |  97 |   8 |  38 |  68 |  98 |   9 |  39 |  69 |  99 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 4  |  15 |  45 |  75 | 105 |  16 |  46 |  76 | 106 |  17 |  47 |  77 | 107 |  18 |  48 |  78 | 108 |  19 |  49 |  79 | 109 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 5  |  25 |  55 |  85 | 115 |  26 |  56 |  86 | 116 |  27 |  57 |  87 | 117 |  28 |  58 |  88 | 118 |  29 |  59 |  89 | 119 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

// print_layout(res);
((_5,_2),(_3,_4)):((_24,_3),(_1,_6))
        0     1     2     3     4     5     6     7     8     9    10    11
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 0  |   0 |   1 |   2 |   6 |   7 |   8 |  12 |  13 |  14 |  18 |  19 |  20 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 1  |  24 |  25 |  26 |  30 |  31 |  32 |  36 |  37 |  38 |  42 |  43 |  44 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 2  |  48 |  49 |  50 |  54 |  55 |  56 |  60 |  61 |  62 |  66 |  67 |  68 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 3  |  72 |  73 |  74 |  78 |  79 |  80 |  84 |  85 |  86 |  90 |  91 |  92 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 4  |  96 |  97 |  98 | 102 | 103 | 104 | 108 | 109 | 110 | 114 | 115 | 116 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 5  |   3 |   4 |   5 |   9 |  10 |  11 |  15 |  16 |  17 |  21 |  22 |  23 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 6  |  27 |  28 |  29 |  33 |  34 |  35 |  39 |  40 |  41 |  45 |  46 |  47 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 7  |  51 |  52 |  53 |  57 |  58 |  59 |  63 |  64 |  65 |  69 |  70 |  71 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 8  |  75 |  76 |  77 |  81 |  82 |  83 |  87 |  88 |  89 |  93 |  94 |  95 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 9  |  99 | 100 | 101 | 105 | 106 | 107 | 111 | 112 | 113 | 117 | 118 | 119 |
    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
```

make_cotiled_copy

make_tiled_copy_S

make_tiled_copy_D

tile_size

size

# TMA

## TMA Traits Swizzle

```cpp
template <int B, int M, int S>
CUTE_HOST_DEVICE constexpr
TMA::SmemSwizzleBits
get_tma_swizzle_bits(Swizzle<B,M,S>)
{
  if constexpr (M == 4) {
    static_assert(0 <= B && B <= 3, "Expected B = 0,1,2, or 3 when M == 4. Unsupported layout swizzle.");
    if constexpr (B == 3) { return TMA::SmemSwizzleBits::B128; }
    if constexpr (B == 2) { return TMA::SmemSwizzleBits::B64; }
    if constexpr (B == 1) { return TMA::SmemSwizzleBits::B32; }
    if constexpr (B == 0) { return TMA::SmemSwizzleBits::DISABLE; }
  } else

  if constexpr (M == 5 || M == 6) {
    static_assert(B == 2, "Expected B = 2 when M == 5 or 6. Unsupported layout swizzle.");
    // S-condition as well?
    return TMA::SmemSwizzleBits::B128;
  } else

  {
    static_assert(M < 0, "Unsupported layout swizzle.");
  }
}

template <class Layout>
TMA::SmemSwizzleBits
get_tma_swizzle_bits(Layout const& layout)
{
  return get_tma_swizzle_bits(get_swizzle_portion(layout));
}

template <int B, int M, int S>
CUTE_HOST_DEVICE constexpr
TMA::SmemSwizzleBase
get_tma_swizzle_base(Swizzle<B,M,S>)
{
  if constexpr (M == 4) {
    static_assert(0 <= B && B <= 3, "Expected B = 0,1,2, or 3 when M == 4. Unsupported layout swizzle.");
    static_assert(S == 3, "Expected S = 3 when M == 4. Unsupported layout swizzle.");
    return TMA::SmemSwizzleBase::SWIZZLE_BASE_16B;
  } 
  
  else if constexpr (M == 5) {
    static_assert(B == 2, "Expected B = 2 when M == 5. Unsupported layout swizzle.");
    static_assert(S == 2, "Expected S = 2 when M == 5. Unsupported layout swizzle.");
    return TMA::SmemSwizzleBase::SWIZZLE_BASE_32B;
  } else if constexpr (M == 6) {
    static_assert(B == 2, "Expected B = 2 when M == 5. Unsupported layout swizzle.");
    return TMA::SmemSwizzleBase::SWIZZLE_BASE_64B;
  } 
  #if 1
  else {
    static_assert(4 <= M && M <= 6, "Expected 128b=16B=(2^4)B to 512b=64B=(2^6)B base swizzle.");
  }
  #else 
  
  else {
    static_assert(M == 4, "Expected 128b=16B=(2^4)B base swizzle.");
  }
  #endif 
}

template <class Layout>
TMA::SmemSwizzleBase
get_tma_swizzle_base(Layout const& layout)
{
  return get_tma_swizzle_base(get_swizzle_portion(layout));
}
```

## TMA Traits

```cpp
// The non-executable SM90_TMA_LOAD with tma_desc and no tma_mbar
// Use .with(tma_mbar) to construct an executable version
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM90_TMA_LOAD, NumBitsPerTMA, AuxParams_>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_LOAD arguments
  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;

  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const*
  get_tma_descriptor() const {
    return &tma_desc_;
  }

  // Construct an executable SM90_TMA_LOAD with tma_mbar
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD_OP, NumBitsPerTMA>
  with(
    uint64_t& tma_mbar,
    [[maybe_unused]] uint16_t const& multicast_mask = 0,
    TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    return {&tma_desc_, &tma_mbar, static_cast<uint64_t>(cache_hint)};
  }

  // Construct an executable SM90_TMA_LOAD with tma_mbar (temp. overloaded for grouped gemm/ptr array gemm)
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD_OP, NumBitsPerTMA>
  with(
    TmaDescriptor const* new_tma_desc,
    uint64_t& tma_mbar,
    [[maybe_unused]] uint16_t const& multicast_mask = 0,
    TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    return {new_tma_desc, &tma_mbar, static_cast<uint64_t>(cache_hint)};
  }

  // Generate the TMA coord tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  // Don't try to execute a copy with SM90_TMA_LOAD before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};
```

### make_tma_copy_desc

创建 tma 描述符。

### make_tma_copy_atom

创建一个 tma 类型的 copy atom。首先通过 make_tma_copy_desc 获取一个 tma 描述符，然后构建 traits 和 atom。

```cpp
template <class TmaInternalType,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class VShape, class VStride>
CUTE_HOST_RTC
auto
make_tma_copy_atom(CopyOp,
                   Tensor<GEngine,GLayout> const& gtensor,       // Full GMEM Tensor
                   SLayout                 const& slayout,       // CTA Tile of SMEM, potentially swizzled
                   uint32_t                const& num_multicast, // The number of CTAs involved in multicasting
                   Layout<VShape,VStride>  const& cta_v_map)     // V: CTA val idx -> gmem mode
{
  //
  // TMA truncated layout
  //

  auto smem_swizzle = get_swizzle_portion(slayout);
  auto smem_layout  = get_nonswizzle_portion(slayout);

  auto tma_gbasis = detail::construct_tma_gbasis<TmaInternalType>(gtensor, smem_layout, cta_v_map);

  //
  // Construct the TMA Desc and the strides of the TMA Tensor
  //

  auto [tma_desc, aux_params] = detail::make_tma_copy_desc<TmaInternalType>(gtensor,
                                                                            tma_gbasis,
                                                                            smem_swizzle,
                                                                            num_multicast);

  //
  // Construct the Copy_Traits
  //

  constexpr int num_bits_per_tma = size(tma_gbasis) * sizeof_bits_v<TmaInternalType>;
  using Traits = Copy_Traits<CopyOp, cute::C<num_bits_per_tma>, decltype(aux_params)>;
  using Atom   = Copy_Atom<Traits, typename GEngine::value_type>;

  Traits tma_traits{tma_desc, aux_params};

#if 0
  print("num_bits_per_tma :  "); print(num_bits_per_tma); print("\n");
  print("g_stride_bases   :  "); print(tma_traits.aux_params_.g_stride_); print("\n");
#endif

  // Return the Copy_Atom
  return Atom{tma_traits};
}
```

### make_tma_copy_tiled

用于创建一个 tma 类型的 tiledcopy。函数内部也是先调用的 make_tma_copy_atom，然后和 layout_TV 组合形成 tiledcopy。

### make_tma_copy

用于创建 tma copy，入参主要有三个，第一个是 copy op，就是使用哪一种 tma 进行拷贝，默认是 cp.async.bulk.tensor。第二个是 gmem 的 tensor，第三个是 smem 的 layout，第四个是可选的 cluster size。

```cpp
// Explicit defaulting
template <class CopyOp,
          class GEngine, class GLayout,
          class SLayout>
CUTE_HOST_RTC
auto
make_tma_copy(CopyOp                  const& copy_op,
              Tensor<GEngine,GLayout> const& gtensor,
              SLayout                 const& slayout)
{
  return make_tma_copy(copy_op, gtensor, slayout, product_each(shape(slayout)), Int<1>{});
}

// Explicit defaulting
template <class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class Cluster_Size>
CUTE_HOST_RTC
auto
make_tma_copy(CopyOp                  const& copy_op,
              Tensor<GEngine,GLayout> const& gtensor,
              SLayout                 const& slayout,
              Cluster_Size            const& cluster_size)
{
  return make_tma_copy(copy_op, gtensor, slayout, product_each(shape(slayout)), cluster_size);
}
```

之后会调用 make_tma_copy_tiled。

### make_tma_atom

看着跟 make_tma_copy 类似，调用的是 make_tma_copy_atom 函数。

```cpp
template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler,
          class Cluster_Size = Int<1>>
CUTE_HOST_RTC
auto
make_tma_atom(CopyOp                  const& copy_op,
              Tensor<GEngine,GLayout> const& gtensor,
              SLayout                 const& slayout,
              CTA_Tiler               const& cta_tiler,
              Cluster_Size            const& cluster_size = {})
{
  auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);
  // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
  using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
  return detail::make_tma_copy_atom<TmaType>(copy_op,
                                             gtensor, slayout,
                                             size(cluster_size), cta_v_tile);
}
```

### tma_partition

类似于普通 tiledcopy 的 partition 函数，不过 tma 把 partitionS 和 partitionD 合成一个了，返回 gtensor 和 stensor。

### make_tma_copy_A_sm90

### make_tma_copy_B_sm90

### make_tma_copy_C_sm90

这三个与 make_tma_copy 函数相同，区别是 cta_tiler 是根据矩阵乘的 MNK 确定的。

