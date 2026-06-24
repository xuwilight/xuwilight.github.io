---
title: cp.async 系列指令（二）
date: 2026-06-20 23:00:00
tags: [CUDA, PTX, cp.async, GPU, TMA]
categories: [PTX 学习笔记]
description: 本文介绍了 cp.async 指令在 Hopper 架构上的拓展指令 cp.async.bulk，也就是 Tensor Memory Accelerate（TMA）支持的指令之一。文章先介绍了 TMA 的基本信息，然后详细介绍了 cp.async.bulk 的使用方法。
---

## Hopper TMA

TMA（Tensor Memory Accelerator）是 NVIDIA Hopper 架构中引入的一个独立硬件加速单元。传统 GPU 的内存流水线在物理层面仅面向一维线性地址空间，高维张量的寻址、边界处理与数据布局转换需要由线程通过大量整数指令完成。TMA 通过一套专用的硬件状态机、地址生成器与格式转换电路，使显存子系统与 L1/共享内存首次能够直接识别并处理高维张量描述符，实现了从“线程驱动搬运”到“硬件自主搬运”的转变。

TMA 的核心功能主要体现在以下四个方面：

### 硬件级多维地址生成

TMA 接受软件预先配置的 Tensor Map（张量描述符），该描述符承载了 1D 至 5D 张量的基地址、各维度尺寸与步长等信息。

当单个线程发射一条 TMA 指令时，内部的专用多维地址生成器（Hardware AGU）接管整个地址计算过程，自动生成一个完整张量分块内所有元素的全局内存物理地址。这消除了一系列原本需要 Warp 中所有线程共同执行的整数乘加运算，显著降低了 SM 的 ALU 占用和指令发射带宽压力，使计算资源能够更集中地服务于核心算术逻辑。

### 内联数据重排

为配合 WGMMA（Warp-Group MMA）指令实现无 bank conflicts 的最高吞吐，共享内存中的数据常需经过 Swizzle（地址交错重排）处理。

TMA 在将数据从 L2 缓存搬移至共享内存的路径上集成了内联重排电路，能够在数据传输的飞行过程中（on-the-fly）完成所需的地址位异或与格式调整，既避免了额外软件重排指令的开销，又保证了数据送达时即可按最优布局直接供计算单元使用。


### 硬件边界侦测与自动零填充

在矩阵分块计算中，处理边缘非完整分块通常需要软件显式判定线程是否越界，并通过条件分支执行填充操作，这不仅引入分支发散，还增加了控制流开销。

TMA 硬件的状态机会在搬运每个数据元素时，将当前坐标与 Tensor Map 中记录的全局边界进行实时比较。一旦发现越界访问，硬件将在向共享内存写入的路径上直接注入零值或特殊值，并拦截对全局内存的非法请求。这使得边缘分块的搬运指令流与内部完整分块保持完全一致，无任何额外分支指令。

### 分布式多播与硬件事务屏障绑定

TMA 并非局限于单个 SM 内的本地搬运器，它通过与 Hopper 线程块集群（Thread Block Cluster）互联协作，支持将同一数据块多播（Multicast）至集群内其他 SM 的分布式共享内存（DSMEM）。

同时，TMA 的传输完成信号直接与 mbarrier（内存屏障）的相位状态机硬连线。当指定字节量的数据到达后，TMA 硬件主动向 mbarrier 硬件寄存器发出事务到达确认，实现基于传输字节数的异步同步。

### 为什么需要独立 TMA 硬件

在 Hopper 之前的架构（如 A100）中，cp.async 指令已实现异步数据搬运，即数据从全局内存直达共享内存而不占用寄存器，且 Warp 可继续执行后续指令。然而，这种异步性仅体现在数据传输阶段，数据的“寻址”和“搬运控制”仍需 Warp 中的每个线程亲自完成。这就会导致消耗 ALU 和指令带宽，且边界处理引发分支发散，也无法以单指令粒度完成整块搬运与集群同步。TMA 将这些任务全部卸载至专用硬件，将计算资源释放给矩阵运算，从而在物理层面解耦数据供给与计算。


## cp.async 系列指令之 bulk-copy (TMA)

使用 TMA 的 bulk 异步复制有两种形式，一种是异步复制一维数据的 `cp.async.bulk`，还有一种是可以复制多维 Tensor 的 `cp.async.bulk.tensor`。本文重点介绍 `cp.async.bulk` 指令。

### cp.async.bulk

启动从一个内存空间到另一个内存空间的异步复制操作。这种是 1D 非 tensor 的异步拷贝。

cp.async.bulk 支持从 global memory 到 shared memory，global memory 到 distributed shared memory，shared memory 到 distributedshared memory 和 shared memory 到 global memory 的拷贝。

PTX 在 8.6 版本对指令做了调整，主要修改了从 global memory 到 shared memory 和 distributed shared memory 的使用方式。调整前 dst 统一使用 .shared::cluster，使用 .multicast 区分 cta 和 cluster。调整后 dst 可以直接使用 .shared::cta 或者 .shared::cluster 区分 cta 和 cluster。

下面是 PTX 8.8 版本的指令格式。

```cpp
// global -> shared::cta
cp.async.bulk.dst.src.completion_mechanism{.level::cache_hint}
                      [dstMem], [srcMem], size, [mbar] {, cache-policy}

.dst =                  { .shared::cta }
.src =                  { .global }
.completion_mechanism = { .mbarrier::complete_tx::bytes }
.level::cache_hint =    { .L2::cache_hint }


// global -> shared::cluster
cp.async.bulk.dst.src.completion_mechanism{.multicast}{.level::cache_hint}
                      [dstMem], [srcMem], size, [mbar] {, ctaMask} {, cache-policy}

.dst =                  { .shared::cluster }
.src =                  { .global }
.completion_mechanism = { .mbarrier::complete_tx::bytes }
.level::cache_hint =    { .L2::cache_hint }
.multicast =            { .multicast::cluster  }


// shared::cta -> shared::cluster
cp.async.bulk.dst.src.completion_mechanism [dstMem], [srcMem], size, [mbar]

.dst =                  { .shared::cluster }
.src =                  { .shared::cta }
.completion_mechanism = { .mbarrier::complete_tx::bytes }


// shared::cta -> global
cp.async.bulk.dst.src.completion_mechanism{.level::cache_hint}{.cp_mask}
                      [dstMem], [srcMem], size {, cache-policy} {, byteMask}

.dst =                  { .global }
.src =                  { .shared::cta }
.completion_mechanism = { .bulk_group }
.level::cache_hint =    { .L2::cache_hint }
```

cp.async.bulk 是一条非阻塞指令，它启动一个异步批量（bulk）复制操作，从源地址 srcMem 指定的位置复制到目标地址 dstMem 指定的位置。

批量复制的方向是从 .src 修饰符指定的空间复制到 .dst 修饰符指定的空间。

32 位的 size 指定要复制的内存大小（以字节数表示）。size 必须是 16 的倍数。内存范围 [dstMem, dstMem + size - 1] 不能溢出目标内存空间，内存范围 [srcMem, srcMem + size - 1] 也不能溢出源内存空间。地址 dstMem 和 srcMem 必须对齐到 16 字节。

当复制的目标是 .shared::cta 时，目标地址位于集群内正在执行指令的 CTA 的共享内存中。当复制的源是 .shared::cta 且目标是 .shared::cluster 时，目标地址需要位于 CTA cluster 内另一个 CTA 的共享内存中。

修饰符 .completion_mechanism 指定指令变体支持的完成机制。下表总结了不同变体支持的完成机制：

![completion](/assets/cp-async/cpasyncbulk_completion.png "completion")

修饰符 .mbarrier::complete_tx::bytes 指定 cp.async.bulk 变体使用基于 mbarrier 的完成机制。complete-tx 操作的 completeCount 参数等于复制的数据量（以字节为单位），将在操作数 mbarrier 指定的 mbarrier 对象上执行。

修饰符 .bulk_group 指定 cp.async.bulk 变体使用 bulk async-group 的完成机制。

简而言之就是 dst 是 shared memory 时用 mbarrier 完成机制，dst 是 global memory 时用 bulk async-group 完成机制。

可选修饰符 .multicast::cluster 允许将数据从全局内存复制到 cluster 中多个 CTA 的共享内存。操作数 ctaMask 指定 cluster 中的目标 CTA，使得 16 位 ctaMask 操作数中的每个位对应于目标 CTA 的 %ctaid。源数据将 multicast 到与每个目标 CTA 共享内存中 dstMem 相同的 CTA 相对偏移量。 mbarrier 信号也会被 multicast 到目标 CTA 共享内存中与 mbar 相同的 CTA 相对偏移量。

当指定可选参数 cache-policy 时，必须使用限定符 .level::cache_hint。64 位操作数 cache-policy 指定了内存访问期间可能使用的缓存驱逐策略。

cache-policy 是针对缓存子系统的提示，它仅被视为性能提示，不会改变程序的内存一致性行为。仅当 .src 或 .dst 状态空间中至少有一个是 .global 状态空间时，才支持限定符 .level::cache_hint。


```cpp
// .global -> .shared::cta (strictly non-remote):
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [mbar];

cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint
                                             [dstMem], [srcMem], size, [mbar], cache-policy;

// .global -> .shared::cluster:
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [mbar];

cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
                                             [dstMem], [srcMem], size, [mbar], ctaMask;

cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint
                                             [dstMem], [srcMem], size, [mbar], cache-policy;


// .shared::cta -> .shared::cluster (strictly remote):
cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [mbar];

// .shared::cta -> .global:
cp.async.bulk.global.shared::cta.bulk_group [dstMem], [srcMem], size;

cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint} [dstMem], [srcMem], size, cache-policy;

// .shared::cta -> .global with .cp_mask:
cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.cp_mask [dstMem], [srcMem], size, cache-policy, byteMask;
```

cp.reduce.async.bulk 和 cp.async.bulk.prefetch 不再详细介绍， 详见 [cp.reduce.async.bulk](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=mma%2520sync%2520aligned%2520m8n8k4#data-movement-and-conversion-instructions-cp-reduce-async-bulk) 和 [cp.async.bulk.prefetch](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=mma%2520sync%2520aligned%2520m8n8k4#data-movement-and-conversion-instructions-cp-async-bulk-prefetch)。

#### 示例代码

```cpp
__global__ void cp_async_bulk(float *src, float *dst, int N)
{
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x;

    extern __shared__ float smem[];
    __shared__ uint64_t bar[1]; // 定义一个 64 位的 barrier
    bar[0] = 0;

    int transaction_bytes = blockDim.x * sizeof(float);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(bar);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem);

    // 只需要一个线程就可以启动 TMA 异步传输
    if (tid == 0)
    {
        // 初始化 barrier，参与 barrier 的线程数是整个 block 的 thread
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_mbar),
                     "r"(blockDim.x));
        asm volatile ("fence.proxy.async.shared::cta;"); // 初始化后通过 fence 对异步拷贝可见

        // 设置异步传输的数据量
        asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_mbar),
                     "r"(transaction_bytes));

        // 启动异步传输
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :
                     : "r"(smem_int_ptr), "l"(src + index), "r"(transaction_bytes), "r"(smem_int_mbar)
                     : "memory");
    }
    __syncthreads();

    // arrive，这里使用 arrive 的返回值 token 记录异步传输状态
    uint64_t token = 0;
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n" ::"l"(token), "r"(smem_int_mbar));

    // wait，通过 token 判断传输是否完成
    asm volatile(
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra DONE;\n"
        "bra                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n" ::"r"(smem_int_mbar),
        "l"(token));

    // compute

    asm volatile("fence.proxy.async.shared::cta;"); // TMA 和 LSU 属于不同的 proxy，对同一个内存进行读写需要进行 fence
    __syncthreads();

    // store, shared memory to global memory
    if (tid == 0)
    {
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                     :
                     : "l"(dst + index), "r"(smem_int_ptr), "r"(transaction_bytes)
                     : "memory");
        asm volatile("cp.async.bulk.commit_group;");
        asm volatile(
            "cp.async.bulk.wait_group.read %0;"
            :
            : "n"(0)
            : "memory");
    }
}

// nvcc async_copy.cu -o cpasync -arch=sm_90a -std=c++17
int main()
{
    srand(1234);

    int N = 1e8;

    thrust::host_vector<float> h_S(N);
    thrust::host_vector<float> h_D(N);
    thrust::host_vector<float> copy_result(N);

    for (int i = 0; i < N; ++i)
    {
        h_S[i] = static_cast<float>(rand() % 9 + 1);
    }

    thrust::device_vector<float> d_S = h_S;
    thrust::device_vector<float> d_D = h_D;

    constexpr int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cp_async_bulk<<<blocks, threads, threads * sizeof(float)>>>(d_S.data().get(), d_D.data().get(), N);

    copy_result = d_D;
    test_copy(h_S.data(), copy_result.data(), N);

    return 0;
}
```

从 global memory 到 shared memory 还有种写法如下，和第一种的主要区别是使用了 mbarrier.arrive.expect_tx.shared。这个指令可以先设置 TMA 传输的数据，然后执行 arrive-on。

此外，mbarrier.try_wait 也使用了另一种奇偶校验的写法。通过一个 phase 变量判断与当前 mbarrier 中的状态是否一致。


```cpp
__global__ void cp_async_bulk(float *src, float *dst, int N)
{
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x;

    extern __shared__ float smem[];
    __shared__ alignas(8) uint64_t bar[1]; // mbarrier 需要 8 字节对齐
    bar[0] = 0;

    int transaction_bytes = blockDim.x * sizeof(float);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(bar);

    if (tid == 0)
    {
        // 初始化 barrier, 这里参与线程数为 1
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_mbar),
                     "r"(1));

        // 这里会先设置 expect_tx，再执行 arrive-on 操作
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_int_mbar),
                     "r"(transaction_bytes));

        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem);
        auto gmem_ptr = src + index;
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :
                     : "r"(smem_int_ptr), "l"(gmem_ptr), "r"(transaction_bytes), "r"(smem_int_mbar)
                     : "memory");
    }
    __syncthreads();

    // wait，使用奇偶校验判断 phase_bit 是否与当前 phase 的奇偶性相同
    int phase_bit = 0;
    asm volatile(
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra DONE;\n"
        "bra                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n" ::"r"(smem_int_mbar),
        "r"(phase_bit));

    // phase_bit ^= 1; // 如果是多阶段，phase_bit 使用后需要切换状态
    dst[index + tid] = smem[tid];
}
```

#### Profile

![cpasyncbulk_profile](/assets/cp-async/profile_cpasyncbulk.png "profile")

从 profile 中可以看到， 加载数据使用了 TMA，而且从 global memory 到 shared memory 没有经过 L1 cache。此时使用的 SASS 指令是 UBLKCP。


## 总结

本文介绍了 cp.async 指令在 Hopper 架构上的拓展 cp.async.bulk 指令。这个指令可以调用 Hopper 架构上引入的 TMA 硬件进行异步数据搬运。

文章首先介绍了 TMA 的一些特性以及优点，包括硬件级多维地址生成，内联数据重排，越界检测等特性，然后介绍了非 Tensor 类型的数据搬运方法，最后通过两个例子介绍了应该如何使用相关 PTX 指令进行数据搬运，并使用 ncu profile 验证结果。

下一篇将详细介绍 cp.async.bulk.tensor 指令，即针对多维 Tensor 类型的数据搬运指令，也是 TMA 主要处理的场景。