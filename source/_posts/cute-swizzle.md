---
title: Shared memory 和 Swizzle
date: 2026-06-20 18:00:00
tags: [CUTLASS, CuTe, Swizzle, GPU, Bank Conflict, Shared memory]
categories: [Cutlass 学习笔记, GPU]
mathjax: true
description: 深入解析 CuTe Swizzle 机制，涵盖 Bank Conflict 原理、Swizzle 布局、Swizzle 与 Layout 的组合方式等核心概念。
---

## Shared Memory

GPU 的物理存储主要分为 4 个部分，寄存器，L1 cache，L2 cache 和 device memory。其中 L1 cache 可以根据用户配置划分部分空间作为 shared memory。比如在 A100 上，一个 SM 共有 192KB 的 L1 cache，最高可以分配 164KB 的 shared memory。 

虽然 shared memory 和 L1 在同一片物理内存上，但是两者的访问模式却不一样。L1 cache 以 sector 为基本访问单元，shared memory 则是以 banks 的形式组织。 

如下图所示，shared memory 中一共有 32 个 banks，一个 bank 的长度是 4 bytes，32 个 banks 一共有 128 bytes，所以 shared memory 中的数据可以看成是 N 行，128 bytes 列的二维数组。

![](/assets/cute-swizzle/image.png)

不同 bank 内的数据可以同步访问，从而最大化利用 shared memory 的带宽。但是一个 bank 在一个 cycle 内只能有一个线程进行访问，当不同的线程访问同一个 bank 中地址不同的数据时会发生 bank conflict，导致串行化访问，降低访问效率。 

如下图所示，shared memory 可以有下面三种访问模式。 

当一个 warp 中的 32 个线程依次访问 32 个 banks 中的元素时，每个线程对应一个 bank，此时可以在一个 cycle 获取 128 bytes 的数据，并且没有 bank conflict。 

当一个 warp 中的 32 个线程按照间隔若干个元素进行访问时，不同的线程就可能会访问同一个 bank。比如在下图中，每个线程间隔一个元素，前 16 个线程和后 16 个线程会访问同一个 bank 中的不同地址，此时就会触发 bank conflict。

每个 bank 有两个线程同时访问，称为 2-way bank conflicts。这种情况下会序列化成前 16 个线程在第一个 cycle 访问，后 16 个线程在第二个 cycle 访问。如果 bank conflict 数过多，就会导致性能下降。 

当线程访问同一个地址时，会触发线程间的 broadcast，这种情况下没有 bank conflict。

![](/assets/cute-swizzle/image_1.png)

需要注意的是，bank conflict 是在内存事务的粒度上发生的，而不是 warp 的粒度。

如果一个 warp 的一个请求访问的数据量超出了 128 bytes，会被分成多个内存事务（Transaction）来完成，因为一个内存事务一次最多可以处理 128 bytes 的数据。

如果一个内存事务中发生了 bank conflicts，访问就会需要多个 cycles 来完成，其中 cycles 又可以用 wavefronts 来表示。

wavefronts 表示流水线能处理的最大工作负载，一个 cycle 能并行的处理一个 wavefront 中的所有 item，不同的 wavefronts 中的 item 会在不同的 cycles 中序列化处理。 

因此在一个内存事务中，每个 bank 被访问次数的最大值就是一次访问需要的 cycle 数，也就是 wavefronts 的数量。cycle 数减去 1 就是 bank conflict 的次数。 

在上面的例子中，一个线程访问一个 4 bytes 数据，所以一个 warp 的 32 个线程正好对应一个内存事务 (128 bytes)。因为没有发生 bank conflicts，因此只需要一个 cycle 就可以处理完成。 

当一个线程一次访问大于 4 bytes 时，一个 warp 就会超过 128 bytes，因此就需要多个内存事务完成一个访问请求。比如，当一个线程读取 8 bytes 的数据，一个 warp 需要读取 256 bytes，所以需要两个内存事务。如果一个线程读取 16 bytes，则需要 4 个内存事务。 

下面看几个例子，假如有一个 32×32 大小的 float 数组保存在 shared memory 中，32 列数据正好对应 32 个 banks，每个 bank 中都有 32 行的数据。 

**例 1**

当一个 warp 中的线程按行加载数据，一个线程处理一个 float。此时一个 warp 的 32 个线程刚好能加载一行数据，32 行数据需要加载 32 次。因此会有 32 个指令 (instructions)，每个指令产生一次内存请求 (requests）。

由于每个 request 中，warp 中的线程访问的是不同的 bank，因此不会有 bank conflict。而且一个 warp 访问的数据量刚好可以在一个内存事务中完成，所以每个 request 只需要一个 wavefront，32 个 request 对应 32 个 wavefronts。


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

ncu profile 结果如下：

![](/assets/cute-swizzle/image_2.png)

**例 2**

当一个 warp 中的线程按行加载数据，但是一个线程处理 4 个 float。此时只需要 8 个线程就能加载一行，一个 warp 中的 32 个线程可以加载 4 行。所以 32 行数据需要 8 个 instructions 和 requests。 

由于在一个 request 中访问的数据是 512 bytes，超过了一个内存事务能处理的最大 128 bytes，所以一个 request 需要 4 个内存事务才能完成，8 个 request 一共需要 32 个内存事务。

由于一个内存事务内没有 bank conflict，所以整个访问过程也没有 bank conflict，32 个内存事务对应 32 个 wavefronts。 


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

ncu profile 结果：

![](/assets/cute-swizzle/image_3.png)

**例 3**

当一个 warp 中的线程按列访问数据，一个线程处理一个 float。此时 32 列需要 32 个 instructions 和 request 才能够完成。

这种情况下一列会产生 32-way bank conflicts，需要 32 个 wavefronts，32 列一共需要 $32×32=1024$ 个 wavefronts 加载数据，产生 $31×32=992$ 个 bank conflicts。 

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

ncu profile 结果：

![](/assets/cute-swizzle/image_4.png)


**例 4**

当一个 warp 中的线程按列加载数据，一个线程一次处理 4 个 float 时，会使用 LDS。128 或 STS。128 进行加载和保存数据。一个 warp 一个指令会加载 4 列数据，32 列只需要 8 个指令就能完成，因此 instructions 和 requests 都是 8。 

由于一个请求要加载 512 bytes 的数据（$32×16 bytes$），所以一个请求会被分成 4 个内存事务，0-7，8-15，16-23，24-31 号线程分别对应 4 个内存事务。 

在每个内存事务中，8 个线程访问相同的 bank，形成 8-way bank conflicts，因此需要 8 个 wavefronts，产生 7 次 bank conflicts。所以一共会产生 $8×4×8=256$ 个 wavefronts，$7×4×8=224$ 次 bank conflicts。 

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

ncu profile 结果：

![](/assets/cute-swizzle/image_5.png)

## 通过 swizzle 避免 bank conflict

如下图所示，如果有一个 $N×NX$ 的二维矩阵，我们可以通过 $y × NX + x$ 获取 $y$ 行 $x$ 列数据的物理地址。如果 $NX=32$，矩阵在 shared memory 中和 bank 的对应关系如下，一行 32 个元素分别对应从 0 到 31 的 bank。 

从上面的分析可以知道，如果按列读取矩阵元素的话，所有的线程都会访问到同一个 bank，产生 32-way bank conflicts，导致性能下降。

![](/assets/cute-swizzle/image_6.png)

为了避免 bank conflicts，我们可以对矩阵进行 padding，将 32 列矩阵 padding 成 33 列。

如下图所示，多出来的 33 列元素会占用 0 号 bank，因此每一列元素对应的 bank 就不一样了，按列访问元素时就不会产生 bank conflict。 但是这种方法需要占用额外的 shared memory 资源。而且向量化访问时需要额外处理 padding 元素。

![](/assets/cute-swizzle/image_7.png)

另外一种避免 bank conflicts 的方法就是 **swizzle**。 

正常的逻辑坐标到 index 之间的关系是 $index = y * NX + x$，而 swizzle 是 $index = y * NX + (y \oplus x)$。通过逻辑坐标的行号与列号进行异或得到一个新的物理坐标的位置，从而可以将一列数据放置到不同的 bank 上，避免了访问同一个 bank。

![](/assets/cute-swizzle/image_8.png)

在上一章的例三中产生了 992 次 bank conflicts。如果我们仅将 shared memory 中的 $tid * 32 + i$ 变成了 $tid * 32 + i \oplus tid$，可以看到 instructions 和 requests 还是 32，但是 bank conflicts 变成了 0。

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

![](/assets/cute-swizzle/image_9.png)

同样的，我们将上一章例四中的 index 从 i 改为 i ^ (tid % 8)，也可以解决 bank conflicts。

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

![](/assets/cute-swizzle/image_10.png)


## cute 中 swizzle 的实现

cute 中 swizzle 的基本实现如下。

可以看到定义一个 swizzle 需要三个模板参数，B=Bits，M=Base，S=Shift。通过这三个参数对原始的 index（offset）进行位运算，得到新的 index，也就是 swizzle 后的 index。

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

比如原始的 index1 的二进制如下 

$$0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx$$

其中 base=3，bits=2，shift = 11。

异或后得到的新的 index2 的二进制为 

$$0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx$$

其中 AA = ZZ xor YY。

下面来分析这个三个参数的作用以及具体的计算过程。

假设一个矩阵有 `rows` 行，`cols` 列，其中： 

$$\text{rows} = 2^{\text{bits}}, \quad \text{cols} = 2^{\text{shift}}$$

对于一个位于第 `y` 行、第 `x` 列的位置，其线性索引为：

$$\text{index1} = y \times \text{cols} + x$$

因此：

$$y = \frac{\text{index1}}{\text{cols}} = \text{index1} \gg \text{shift}$$

于是有：

$$y \oplus x = (\text{index1} \gg \text{shift}) \oplus x$$

也就是说，`index1` 的高若干位表示 `y`，低 `shift` 位表示 `x`，二者进行异或运算，其中 `x` 就是 `index1` 的低 `shift` 位。

接下来需要确定 `y` 的取值范围，即需要用多少位表示 `y`。

由于：

$$y \in [0, \ \text{rows} - 1] = [0, \ 2^{\text{bits}} - 1]$$

所以用 `bits` 位即可完整表示 `y` 的取值范围。


因此，计算流程如下：

1. 首先根据 `bits` 计算用于异或运算的掩码 `bit_mask`，其中 `bits` 对应 `y` 的取值范围：bit\_mask = (1 << bits) - 1.

2. 将 `bit_mask` 左移 `shift` 位，得到当前 `index1` 对应的 `yyy_mask`：yyy\_mask = bit\_mask << max(0, shift).

3. 将 `index1` 与 `yyy_mask` 按位与，提取出 `yyy`：yyy = offset & yyy\_mask.

4. 将 `yyy` 右移 `shift` 位得到 `y`，再与 `index1` 进行异或，得到目标索引 index2 = y * cols + (y ^ x)：index2 = index1 ^ (yyy >> shift).

上述过程针对的是单个元素的地址重映射。若需将多个元素视为一个整体进行批量操作，则需要引入 `base` 参数。

例如，若一次加载 4 个 `float`，即将 4 个元素视为一个整体，则 `base = 2`（因为 \(2^2 = 4\)）。此时，只需在地址计算中保留 `index` 的低 `base` 位不变，即在第 2 步构造 `yyy_mask` 时额外左移 `base` 位：yyy\_mask = bit\_mask << (base + max(0, shift))，其余步骤保持不变。

如果有一个 16 行 32 列的 float 矩阵，假设该矩阵的 Layout 是(16,32):(32,1)，定义一个 `swizzle = Swizzle<3,2,3>{}`。第一个 3 表示异或的范围是 8 行，2 表示 4 个元素为一组，因此一组是 16 bytes。第二个 3 表示一行包括 8 列。 

原始的 Layout 打印如下，此时可以看到相同颜色的数据都在相同的 bank 下，访问相同颜色的数据会产生 bank conflicts。

![](/assets/cute-swizzle/image_11.png)

swizzle 后的 Layout 如下。经过 swizzle 后，相同颜色的数据以 8 行为循环被 swizzle 到不同的 bank 下，此时再访问相同颜色的数据就不会导致 bank conflicts 了。

![](/assets/cute-swizzle/image_12.png)

## References

1. https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41723/
1. https://www.nvidia.com/en-us/on-demand/session/gtc24-s62191/
1. https://www.nvidia.com/en-us/on-demand/session/gtc24-s62192/
1. https://zhuanlan.zhihu.com/p/4746910252
1. https://zhuanlan.zhihu.com/p/671419093



{% blockquote %}
文章首发于知乎，此处做了部分修改。
{% link https://zhuanlan.zhihu.com/p/1906737849576953561 desc:true %}
{% endblockquote %}
