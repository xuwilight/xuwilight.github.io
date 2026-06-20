---
title: CuTe Swizzle
date: 2026-06-20 18:00:00
tags: [CUTLASS, CuTe, Swizzle, GPU, Bank Conflict]
categories: [Cutlass 学习笔记]
description: 深入解析 CuTe Swizzle 机制，涵盖 Bank Conflict 原理、Swizzle 布局、Swizzle 与 Layout 的组合方式等核心概念。
---

# CuTe_Swizzle

# Shared Memory

GPU 的物理存储主要分为 4 个部分，寄存器，L1 cache，L2 cache 和 device memory。其中寄存器速度最快，L1 cache 次之。 

L1 cache 可以根据用户配置划分部分空间作为 shared memory。比如在 A100 上，一个 SM 共有 192KB 的 L1 cache，最高可以分配 164KB 的 shared memory。 

虽然 shared memory 和 L1 在同一片物理内存上，但是两者的访问模式却不一样。L1 cache 是以 sector 为基本单位访问的，而 shared memory 则是按照 banks 的形式进行组织的。 

如下图所示，shared memory 中一共有 32 个 banks，一个 bank 的长度是 4bytes，连续的 4 个 bytes 对应到连续的 banks，32 个 banks 一共有 128bytes，所以 shared memory 中的数据可以看成是 N 行，128bytes 列的二维数组。

![](/assets/cute-swizzle/image.png)

不同 bank 内的数据可以同步访问，从而最大化 shared memory 的带宽。但是一个 bank 在一个 cycle 内只能有一个线程进行访问，所以当不同的线程访问同一个 bank 中不同地址的数据时会发生 bank conflict，导致串行化访问，降低访问效率。 

如下图所示，shared memory 可以有下面三种访问模式。 

当一个 warp 中的 32 个线程依次访问 32 个 banks 中的元素时，每个线程对应一个 bank，因此可以在一个 cycle 获取 128bytes 的数据，并且没有 bank conflict。 

当一个 warp 中的 32 个线程按照间隔若干个元素进行访问时，不同的线程就可能会访问同一个 bank。比如在下图中，每个线程间隔一个元素，前 16 个线程和后 16 个线程会访问同一个 bank 中的不同地址，此时就会触发 bank conflict。每个 bank 有两个线程同时访问，称为 2-way bank conflicts。这种情况下会序列化成前 16 个线程在第一个 cycle 访问，后 16 个线程在第二个 cycle 访问。如果 bank conflict 数过多，就会导致性能下降。 

当线程访问同一个地址时，会触发线程间的 broadcast，这种情况下没有 bank conflict。

![](/assets/cute-swizzle/image_1.png)

需要注意的是，如果一个 warp 的一个请求访问的数据量超出了 128bytes，会被分成多个内存事务（Transaction）来完成，因为一个内存事务一次最多可以访问 128bytes 的数据。所以 bank conflict 是发生在一次内存事务中的。 

如果一个内存事务中发生了 bank conflicts，访问就会需要多个 cycles 来完成，其中 cycles 又可以用 wavefronts 来表示。wavefronts 表示流水线能处理的最大工作负载，一个 cycle 能并行的处理一个 wavefront 中的所有 item，不同的 wavefronts 中的 item 会在不同的 cycles 中序列化处理。 

因此在一个内存事务中，每个 bank 被访问次数的最大值就是一次访问需要的 cycle 数，也就是 wavefronts 的数量，cycle 1 就是 bank conflict 的次数。 

在上面的例子中，一个线程访问一个 4bytes 数据，所以一个 warp 的 32 个线程正好对应一个内存事务（128 bytes)。由于没有发生 bank conflicts，因此只需要一个 wavefront 就可以处理完成。 当一个线程一次访问大于 4bytes 时，一个 warp 就会超过 128bytes，因此就需要多个内存事务完成一个访问请求。比如，当一个线程读取 8 bytes 的数据，一个 warp 需要读取 256 bytes，所以需要两个内存事务。前半个 warp 一个，后半个 warp 一个。如果一个线程读取 16 bytes，则需要 4 个内存事务。 

看几个例子，假如有一个 32×32 大小的 float 数组保存在 shared memory 中，32 列数据正好对应 32 个 banks，每个 bank 中都有 32 行的数据。 

例 1：当一个 warp 中的线程按行加载数据，一个线程处理一个 float。此时一个 warp 的 32 个线程刚好能加载一行数据，32 行数据需要加载 32 次。因此会有 32 个指令（instructions)，每个指令产生一次内存请求(requests）。由于每个 request 中，warp 中的线程访问的是不同的 bank，因此不会有 bank conflict。而且一个 warp 访问的数据量刚好可以在一个内存事务中完成，所以每个 request 只需要一个 wavefront，32 个 request 对应 32 个 wavefronts。

代码：

```cpp
template <class TS, class TD>
__global__ void bank_test_1(TS *S, TD *D, int M, int N)
{
    int tid = threadIdx.x;

    __shared__ TS smem_src[1024];
    for (int i = 0; i < 32; ++i)
    {
        smem_src[tid + i * 32] = S[tid + i * 32];
    }
    __syncthreads();

    for (int i = 0; i < 32; ++i)
    {
        D[tid + i * 32] = smem_src[tid + i * 32];
    }
}
```

ncu 结果：

![](/assets/cute-swizzle/image_2.png)

例 2：当一个 warp 中的线程按行加载数据，但是一个线程处理 4 个 float。此时只需要 8 个线程就能加载一行，所以一个 warp 中的 32 个线程可以加载 4 行。所以 32 行数据需要 8 个 instructions 和 requests。 

由于在一个 request 中访问的数据是 512bytes，超过了一个内存事务能处理的最大 128bytes，所以一个 request 需要 4 个内存事务才能完成，8 个 request 一共需要 32 个内存事务。由于一个内存事务内没有 bank conflict，所以整个访问过程也没有 bank conflict，32 个内存事务对应 32 个 wavefronts。 

代码：

```cpp
template <class TS, class TD>
__global__ void bank_test_2(TS *S, TD *D, int M, int N)
{
    int tid = threadIdx.x;

    __shared__ TS smem_src[1024];
    for (int i = 0; i < 8; ++i)
    {
        reinterpret_cast<float4 *>(smem_src + i * 32 * 4)[tid] = reinterpret_cast<float4 *>(S + i * 32 * 4)[tid];
    }
    __syncthreads();

    for (int i = 0; i < 8; ++i)
    {
        reinterpret_cast<float4 *>(D + i * 32 * 4)[tid] = reinterpret_cast<float4 *>(smem_src + i * 32 * 4)[tid];
    }
}
```

ncu 结果：

![](/assets/cute-swizzle/image_3.png)

例 3：当一个 warp 中的线程按列访问数据，一个线程处理一个 float。此时 32 列需要 32 个 instructions 和 request 才能够完成。这种情况下一列会产生 32-way bank conflicts，需要 32 个 wavefronts，32 列一共需要 32×32=1024 个 wavefronts 加载数据，产生 31×32=992 个 bank conflicts。 

代码：不加 pragma unroll 1 的话编译器会优化成使用 LDS.128，即一次加载四个 float。

```cpp
template <class TS, class TD>
__global__ void bank_test_3(TS *S, TD *D, int M, int N)
{
    int tid = threadIdx.x;

    __shared__ TS smem_src[1024];

#pragma unroll 1
    for (int i = 0; i < 32; ++i)
    {
        smem_src[tid * 32 + i] = S[tid + i * 32];
    }
    __syncthreads();

#pragma unroll 1
    for (int i = 0; i < 32; ++i)
    {
        D[tid + i * 32] = smem_src[tid * 32 + i];
    }
}
```

ncu 结果：

*[Remote image: (placeholder)]*

例 4：当一个 warp 中的线程按列加载数据，一个线程一次处理 4 个 float。此时会使用 LDS。128 或 STS。128 进行加载和保存数据，一个 warp 一个指令会加载 4 列数据，32 列只需要 8 个指令就能完成，因此 instructions 和 requests 都是 8。 

由于一个请求要加载 512bytes 的数据（32×16 bytes），所以一个请求会被分成 4 个内存事务，0-7，8-15，16-23，24-31 号线程分别对应 4 个内存事务。 在每个内存事务中，8 个线程访问相同的 bank，形成 8-way bank conflicts，因此需要 8 个 wavefronts，产生 7 次 bank conflicts。所以一共会产生 8×4×8=256 个 wavefronts，7×4×8=224 次 bank conflicts。 

代码：

```cpp
template <class TS, class TD>
__global__ void bank_test_4(TS *S, TD *D, int M, int N)
{
    int tid = threadIdx.x;

    __shared__ TS smem_src[1024];

    for (int i = 0; i < 8; ++i)
    {
        reinterpret_cast<float4 *>(smem_src + tid * 32)[i] = reinterpret_cast<float4 *>(S + i * 32 * 4)[tid];
    }
    __syncthreads();

    for (int i = 0; i < 8; ++i)
    {
        reinterpret_cast<float4 *>(D + i * 32 * 4)[tid] = reinterpret_cast<float4 *>(smem_src + tid * 32)[i];
    }
}
```

ncu 结果：

*[Remote image: (placeholder)]*

# Swizzle

如下图所示，如果有一个 N×NX 的二维矩阵，我们可以通过 y × NX + x 获取 y 行 x 列数据的物理地址。如果 NX=32，矩阵在 shared memory 中和 bank 的对应关系如下，一行 32 个元素分别对应从 0 到 31 的 bank。 

从上面的分析可以知道，如果按列读取矩阵元素的话，所有的线程都会访问到同一个 bank，产生 32-way bank conflicts，导致性能下降。

*[Remote image: (placeholder)]*

为了避免 bank conflicts，我们可以对矩阵进行 padding，将 32 列矩阵 padding 成 33 列。如下图所示，多出来的 33 列元素会占用 0 号 bank，因此每一列元素对应的 bank 就不一样了，按列访问元素时就不会产生 bank conflict。 但是这种方法需要占用额外的 shared memory 资源。而且向量化访问时需要额外处理 padding 元素。

*[Remote image: (placeholder)]*

另外一种避免 bank conflicts 的方法就是 swizzle。 正常的逻辑坐标到 index 之间的关系是 index = y * NX + x，而 swizzle 是 index = y * NX + （y ^ x），通过逻辑坐标的行号与列号进行异或得到一个新的物理坐标的位置，从而可以将一列数据放置到不同的 bank 上，避免了访问同一个 bank。

关于swizzle有效性的具体证明可以参考：[https://zhuanlan.zhihu.com/p/4746910252](https://zhuanlan.zhihu.com/p/4746910252)

*[Remote image: (placeholder)]*

在上一章的例三中产生了 992 次 bank conflicts。我们仅将 shared memory 中的 tid * 32 + i 变成了 tid * 32 + i ^ tid。可以看到 instructions 和 requests 还是 32，但是 bank conflicts 变成了 0。

```cpp
template <class TS, class TD>
__global__ void bank_test_3_swizzle(TS *S, TD *D, int M, int N)
{
    int tid = threadIdx.x;

    __shared__ TS smem_src[1024];

#pragma unroll 1
    for (int i = 0; i < 32; ++i)
    {
        smem_src[tid * 32 + i ^ tid] = S[tid + i * 32];
    }
    __syncthreads();

#pragma unroll 1
    for (int i = 0; i < 32; ++i)
    {
        D[tid + i * 32] = smem_src[tid * 32 + i ^ tid];
    }
}
```

ncu profile

*[Remote image: (placeholder)]*

同样的，我们将上一章例四中的 index 从 i 改为 i ^（tid % 8），也可以解决 bank conflicts。

```cpp
template <class TS, class TD>
__global__ void bank_test_4_swizzle(TS *S, TD *D, int M, int N)
{
    int tid = threadIdx.x;

    __shared__ TS smem_src[1024];

    for (int i = 0; i < 8; ++i)
    {
        reinterpret_cast<float4 *>(smem_src + tid * 32)[i ^ (tid % 8)] = reinterpret_cast<float4 *>(S + i * 32 * 4)[tid];
    }
    __syncthreads();

    for (int i = 0; i < 8; ++i)
    {
        reinterpret_cast<float4 *>(D + i * 32 * 4)[tid] = reinterpret_cast<float4 *>(smem_src + tid * 32)[i ^ (tid % 8)];
    }
}
```

profile 结果显示如下，8 个 instructions 和 requests，0 次 bank conflict。

*[Remote image: (placeholder)]*

# cute中swizzle的实现

cute 中 swizzle 的基本实现如下。可以看到定义一个 swizzle 需要三个模板参数，B=Bits，M=Base，S=Shift。通过这三个参数对原始的 index（offset）进行位运算，得到新的 index，也就是 swizzle 后的 index。

```cpp
// A generic Swizzle functor
/* 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
 *                               ^--^ MBase is the number of least-sig bits to keep constant
 *                  ^-^       ^-^     BBits is the number of bits in the mask
 *                    ^---------^     SShift is the distance to shift the YYY mask
 *                                       (pos shifts YYY to the right, neg shifts YYY to the left)
 *
 * e.g. Given
 * 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
 * the result is
 * 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY
 */
template <int BBits, int MBase, int SShift = BBits>
struct Swizzle
{
  static constexpr int num_bits = BBits;
  static constexpr int num_base = MBase;
  static constexpr int num_shft = SShift;

  static_assert(num_base >= 0,             "MBase must be positive.");
  static_assert(num_bits >= 0,             "BBits must be positive.");
  static_assert(abs(num_shft) >= num_bits, "abs(SShift) must be more than BBits.");

  // using 'int' type here to avoid unintentially casting to unsigned... unsure.
  using bit_msk = cute::constant<int, (1 << num_bits) - 1>;
  using yyy_msk = cute::constant<int, bit_msk{} << (num_base + max(0,num_shft))>;
  using zzz_msk = cute::constant<int, bit_msk{} << (num_base - min(0,num_shft))>;
  using msk_sft = cute::constant<int, num_shft>;

  template <class Offset>
  CUTE_HOST_DEVICE constexpr static
  auto
  apply(Offset const& offset)
  {
    return offset ^ shiftr(offset & yyy_msk{}, msk_sft{});   // ZZZ ^= YYY
  }
};
```

通过开头注释部分可以大致了解到这三个参数的作用，base 代表一个 index 的低 base 位，bits 代表 mask ZZZ 的位数，shift 代表高位 YYY 和低位 ZZZ 的偏移量。给定一个 index，将其高位 YYY 与低位 ZZZ 进行异或，得到新的 index。 

比如原始的 index1 的二进制如下 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx，其中 base=3，bits=2，shift = 11，异或后得到的新的 index2 的二进制为 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx，其中 AA = ZZ xor YY。

下面来分析这个三个参数的作用以及具体的计算过程。

假设一个矩阵有 rows 行，cols 列，其中 rows = 2^bits，cols = 2^shift。对于一个 y 行，x 列的位置，index1 = y * cols + x，所以 y = index1 / cols = index1 >> shift。因此 y ^ x = （index1 >> shift) ^ x，即 index1 的若干个高位表示的 y 与低位表示的 x 进行异或，其中 x 就是 index1 的低 shift 位。 接下来需要确定 y 的取值范围，也就是需要用多少位表示 y。由于 y 的取值范围是[0， rows - 1]，即[0， 2^bits - 1]，所以用 bits 位就可以表示 y 的取值范围。 

因此计算过程为：

1. 首先通过 bits 计算出需要进行异或运算的掩码 bit_mask，bits 对应了 y 的取值范围。bit_mask = (1 << bits) - 1
1. 然后将 bit_mask 右移 shift 位得到当前 index1 对应的 yyy_mask。yyy_msk = bit_msk << max(0，shift)
1. 将 index1 与 yyy_mask 进行按位与得到 yyy。yyy = offset & yyy_msk
1. 然后将 yyy 右移 shift 位得到 y，并与 index1 进行异或得到 y * cols + y ^ x 的结果 index2。offset ^ (yyy >> shift)

上述过程是单个元素的位置进行异或，如果需要将多个元素看成一个整体，还需要 base 这个参数。比如一次加载 4 个 float，需要将 4 个 float 看成一个整体，此时 base = 2，因为 2^base = 4。这种情况下只需要将 index 保留 base 位的低位即可，即在第二步计算 yyy_mask 时右移 shift + base 位。yyy_msk = bit_msk << (base + max(0，shift)），其余步骤不变。 

结合下面的图可以更好的理解三个参数的作用。2^B 表示行数，2^M 表示基本单元的元素数，2^S 表示列数。下图中 B=1，M=1，S=2。

*[Remote image: (placeholder)]*

如果有一个 16 行 32 列的 float 矩阵，假设该矩阵的 Layout 是（16，32)：(32，1），定义一个 swizzle = Swizzle<3， 2， 3>{}。第一个 3 表示异或的范围是 8 行，2 表示 4 个元素为一组，因此一组是 16bytes。第二个 3 表示一行包括 8 列。 

原始的 Layout 打印如下，此时可以看到相同颜色的数据都在相同的 bank 下，访问相同颜色的数据会产生 bank conflicts。

*[Remote image: (placeholder)]*

swizzle 后的 Layout 如下。经过 swizzle 后，相同颜色的数据以 8 行为循环被 swizzle 到不同的bank下，此时再访问相同颜色的数据就不会导致 bank conflicts 了。

*[Remote image: (placeholder)]*

# References：

1. https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41723/
1. https://www.nvidia.com/en-us/on-demand/session/gtc24-s62191/
1. https://www.nvidia.com/en-us/on-demand/session/gtc24-s62192/
1. https://zhuanlan.zhihu.com/p/4746910252
1. https://zhuanlan.zhihu.com/p/671419093

# backup

共享内存有32个bank，每个bank有4bytes，32个bank一共128bytes，属于一个内存读写事务（128bytes）。

如果每个线程在一个内存读写事务中访问不同bank中的地址，就不会有bank conflict。如果不同的线程访问同一个bank中的同一个地址，会进行broadcast，也不会有bank conflict

但是如果不同的线程访问同一个bank中的不同地址，就会出现bank conflict。此时2个线程间会进行串行访问。产生两个wavefronts。

现在测试的现状，使用ldmatrix指令，当num=x1时，不会产生bank conflict。

当num=x4时，会产生4个bank conflict，8个wavefronts。

疑问：

1. ldmatrix中是0-7一个线程读取8个元素然后分发到32个线程中，还是32个线程分别读取2个元素？

一个线程读取128bits的数据，然后再分发到32个线程中。证据是当ldmatrix的num=x1时，一个线程加载了128bits的数据。如果是后者的话，一个线程只会加载32bits的数据。

*[Remote image: (placeholder)]*

1. 如何解释num=x4时的现象。

当num=x4时，每个线程加载8个fp16，一共16bytes数据。每8个线程属于一个内存事务。所以ldmatrix在加载的时候会先加载前8个线程。当矩阵的大小时16*16时，前8个线程中0和4，1和5，2和6，3和7会产生bank conflict。其中0-3属于一个wavefronts，4-7属于一个wavefronts。所以0-3和4-7会产生一个bank conflict。所以一共会产生8个wavefronts和4个bank conflict。

1. 怎么样算一个bank conflict，如果0-15和16-31线程产生了bank conflict，那么这算16个bank conflict还是算一个bank conflict。

bank conflict是根据wavefronts来算的，如果0-15属于一个wavefronts，16-31属于另一个wavefronts，那么属于一个bank conflict。

如果是float16，两个线程访问同一行同一个bank中的两个元素会发生什么？没有bank conflict。

两个元素访问同一个bank不同行会发生什么？fp16下一个bank可以看成2列，访问同一列还是会发生bank conflict。访问不同列，只要在一个bank里也会有bank conflict

bank conflict test 1

S = 32 * 32 float

```cpp
template <class TS, class TD>
__global__ void bank_test_1(TS *S, TD *D, int M, int N)
{
    int tid = threadIdx.x;

    __shared__ TS smem_src[1024];
    for (int i = 0; i < 32; ++i)
    {
        smem_src[tid + i * 32] = tid;
    }

    __syncthreads();

    for (int i = 0; i < 32; ++i)
    {
        D[tid + i * 32] = smem_src[tid + i * 32];
    }
}
```

load: wavefronts 32, bank conflict 0

store: wavefronts 32, bank conflict 0

test2

```cpp
template <class TS, class TD>
__global__ void bank_test_2(TS *S, TD *D, int M, int N)
{
    int tid = threadIdx.x;

    __shared__ TS smem_src[1024];
    for (int i = 0; i < 32; ++i)
    {
        smem_src[tid + i * 32] = tid;
    }

    __syncthreads();

    for (int i = 0; i < 32; ++i)
    {
        D[tid + i * 32] = smem_src[tid * 32 + i];
    }
}
```

load: wavefronts 256, bank conflict 224

store: wavefronts 32, bank conflict 0

这种情况会用LDS.128来加载shared memory中的数据，32个数据会加载8次，一次加载4个float，16bytes。

由于一个内存事务加载128bytes，所以0-7号线程会在一个事务中加载，会产生8 way bank conflict，bank conflict的次数是7。

所以wavefronts = 32 * 8 = 256。bank conflict = 7 * 4 * 8 = 224次。

The memory access pattern for shared loads might not be optimal and causes on average a 32.0 - way bank conflict across all 8 shared load requests. This results in 224 bank conflicts, which represent 87.50% of the overall 256 wavefronts for shared loads.

test2x

```cpp
template <class TS, class TD>
__global__ void bank_test_2_1(TS *S, TD *D, int M, int N)
{
    int tid = threadIdx.x;

    __shared__ TS smem_src[1024];
    for (int i = 0; i < 32; ++i)
    {
        smem_src[tid + i * 32] = tid;
    }

    __syncthreads();

    for (int i = 0; i < 32; ++i)
    {
        D[tid * 32 + i] = smem_src[tid * 32 + i];
    }
}

template <class TS, class TD>
__global__ void bank_test_2_2(TS *S, TD *D, int M, int N)
{
    int tid = threadIdx.x;

    __shared__ TS smem_src[1024];
    for (int i = 0; i < 32; ++i)
    {
        smem_src[tid + i * 32] = tid;
    }

    __syncthreads();

    for (int i = 0; i < 4; ++i)
    {
        D[tid * 32 + i] = smem_src[tid * 32 + i];
    }
}

template <class TS, class TD>
__global__ void bank_test_2_3(TS *S, TD *D, int M, int N)
{
    int tid = threadIdx.x;

    __shared__ TS smem_src[1024];
    for (int i = 0; i < 32; ++i)
    {
        smem_src[tid + i * 32] = tid;
    }

    __syncthreads();

    if (tid < 8) {
        for (int i = 0; i < 32; ++i)
        {
            D[tid * 32 + i] = smem_src[tid * 32 + i];
        }
    }
}
```

bank_test_2_1修改了保存到全局内存中的顺序，没有影响shared memory的访问，还是224个bank conflict

bank_test_2_2 保存了32行，4列，一行一个线程。所以会产生4*7=28个bank conflict。这里的4是32个线程分成了4个内存事务，不是4列的意思。一个内存事务里有7个bank conflict。

The memory access pattern for shared loads might not be optimal and causes on average a 32.0 - way bank conflict across all 1 shared load requests.This results in 28 bank conflicts.

bank_test_2_3保存了8行，32列，一行一个线程。对于8个线程，4列一个内存事务，所以32行有8个内存事务，一个内存事务中有7个bank conflict。所以一共8*7=56个bank conflict。

The memory access pattern for shared loads might not be optimal and causes on average a 8.0 - way bank conflict across all 8 shared load requests.

test3

```cpp
template <class TS, class TD>
__global__ void bank_test_3(TS *S, TD *D, int M, int N)
{
    int tid = threadIdx.x;

    __shared__ TS smem_src[1024];
    for (int i = 0; i < 32; ++i)
    {
        smem_src[tid + i * 32] = tid;
    }

    __syncthreads();

    D[tid / 8 * 32 + tid % 8] = smem_src[tid / 8 * 32 + tid % 8];
}
```

一行8个线程，一共4行，行与行间bank conflict。8个线程，一个线程加载一个fp32，所以一共属于一个内存事务。

The memory access pattern for shared loads might not be optimal and causes on average a 4.0 - way bank conflict across all 1 shared load requests.This results in 3 bank conflicts

结果有4个wavefronts，3个bank conflict。0-32个线程属于一个内存事务，但是0-7,8-15,16-23,24-31分别产生了bank conflict，所以0-7是一个wavefronts，8-15是一个wavefronts，等等。4个wave fronts产生了3次bank conflict。

test4

32个线程每个加载一个float，每行一个线程错开。

```cpp
template <class TS, class TD>
__global__ void bank_test_4(TS *S, TD *D, int M, int N)
{
    int tid = threadIdx.x;

    __shared__ TS smem_src[1024];
    for (int i = 0; i < 32; ++i)
    {
        smem_src[tid + i * 32] = tid;
    }

    __syncthreads();

    D[tid * 32 + tid] = smem_src[tid * 32 + tid];
}
```

wavefront = 1，bank conflict = 0

如果一个线程访问8个bytes，会有几个wavefronts，几个conflict，2 0
