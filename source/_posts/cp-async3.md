---
title: cp.async 系列指令（二）—— TMA
date: 2026-06-20 22:00:00
tags: [CUDA, PTX, cp.async, GPU, TMA]
categories: [PTX 学习笔记]
description: 本文介绍了 cp.async 指令在 Hopper 架构上的拓展指令 cp.async.bulk，也就是 Tensor Memory Accelerate（TMA）支持的指令之一。文章先介绍了 TMA 的基本信息，然后详细介绍了 cp.async.bulk 的使用方法。
published: false
---



## Tensor copy (TMA Tensor)

Tensor 是一种多维数据结构，维度可以从 1D 到 5D，具有 shape，stride，元素类型等多种属性。PTX 支持对 Tensor 进行复杂的处理，包括 padding，im2col，swizzle 等，详见：[Tensors](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=mma%2520sync%2520aligned%2520m8n8k4#tensors)。

在 Hopper 架构中，TMA 支持对 Tensor 进行异步拷贝，底层使用的是 cp.async.bulk.tensor 指令。

由于 Tensor 是多维结构，所以对 Tensor 进行拷贝时限制比较多。主要的限制有下面一些：
1. smem 地址必须 128B 对齐。
2. gmem 地址必须 16B 对齐。
3. strides 必须是 16B 的倍数。
4. copy box 的最内部维度 (stride = 1 的维度) 必须是 16B 的倍数。
5. copy box 的起始地址必须 16B 对齐。

![cpasyncbulktensor_align](/assets/cp-async/cpasyncbulktensor_align.png "align")

与 tensor copy 有关的信息可以用 tensor map 描述。Tensor-Map 是一个 128 字节的对象，记录了 tensor 的属性以及 copy 时的相关信息。目前只能使用 CUDA API 创建 Tensor-Map。

### cp.async.bulk.tensor

启动一个异步复制操作，从一个空间复制 tensor 到另一个空间。方向可以是 global -> shared::cta，global -> shared::cluster，shared::cta -> global，不支持 shared::cluster和shared::cta 之间的传输。

PTX 在 8.6 版本对指令做了调整，主要修改了从 global memory 到 shared memory 和 distributed shared memory 的使用方式。
调整前 dst 统一使用 .shared::cluster，使用 .multicast 区分 cta 和 cluster。调整后 dst 可以直接使用 .shared::cta 或者 .shared::cluster 区分 cta 和 cluster。

下面是 PTX 8.8 版本的指令格式。

```cpp
// global -> shared::cta
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.cta_group}{.level::cache_hint}
                                   [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo} {, cache-policy}

.dst =                  { .shared::cta }
.src =                  { .global }
.dim =                  { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .mbarrier::complete_tx::bytes }
.cta_group =            { .cta_group::1, .cta_group::2 }
.load_mode =            { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 }
.level::cache_hint =    { .L2::cache_hint }


// global -> shared::cluster
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.multicast}{.cta_group}{.level::cache_hint}
                                   [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo}
                                   {, ctaMask} {, cache-policy}

.dst =                  { .shared::cluster }
.src =                  { .global }
.dim =                  { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .mbarrier::complete_tx::bytes }
.cta_group =            { .cta_group::1, .cta_group::2 }
.load_mode =            { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 }
.level::cache_hint =    { .L2::cache_hint }
.multicast =            { .multicast::cluster  }


// shared::cta -> global
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.level::cache_hint}
                                   [tensorMap, tensorCoords], [srcMem] {, cache-policy}

.dst =                  { .global }
.src =                  { .shared::cta }
.dim =                  { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .bulk_group }
.load_mode =            { .tile, .tile::scatter4, .im2col_no_offs }
.level::cache_hint =    { .L2::cache_hint }
```

cp.async.bulk.tensor 是一条非阻塞指令，它启动一个异步复制操作，将 tensor 数据从 .src 空间中的位置复制到 .dst 空间中的位置。dstMem 指定 tensor 数据在 .dst 空间中要复制到的位置，而 srcMem 指定 tensor 数据在 .src 状态空间中要从的位置复制。

操作数 tensorMap 是 Tensor-Map 对象的地址, tensorMap 指定 tensor 复制操作的属性。

Tensor 的维度由 .dim 修饰符指定。
向量操作数 tensorCoords 指定全局内存中 tensor 的起始坐标。tensorCoords 中各个张量坐标的类型为 .s32。向量参数 tensorCoords 的格式取决于指定的 .load_mode，如下所示：

![tensor_loadmode](/assets/cp-async/tensor_loadmode.png "tensor_loadmode")

限定符 .load_mode 可以是 .tile 或 .im2col，用于指定如何将源位置的数据复制到目标位置。如果未指定 .load_mode，则默认为 .tile。.tile 和 .im2col 的具体原理可以参考 Tensor 的介绍。

修饰符 .completion_mechanism 指定指令变体支持的完成机制。下表总结了不同变体支持的完成机制：

![completion_mechanism](/assets/cp-async/tensor_completion.png "completion_mechanism")

修饰符 .mbarrier::complete_tx::bytes 指定 cp.async.bulk.tensor 变体使用基于 mbarrier 的完成机制。异步复制操作完成后，将对操作数 mbarrier 指定的 mbarrier 对象执行 complete-tx 操作，其 completeCount 参数等于复制的数据量（以字节为单位）。

修饰符 .cta_group 只能与基于 mbarrier 的完成机制一起使用。修饰符 .cta_group 用于标识 CTA-Pair 中 CTA 是奇数编号还是偶数编号。指定 .cta_group::1 时，指定的 mbarrier 对象 mbar 必须位于与共享内存目标 dstMem 相同的 CTA 的共享内存中。指定 .cta_group::2 时，mbarrier 对象 mbar 可以位于与共享内存目标 dstMem 相同的 CTA 的共享内存中，也可以位于其对等 CTA 中。如果未指定 .cta_group，则默认为 .cta_group::1。
修饰符 .bulk_group 指定 cp.async.bulk.tensor 变体使用基于批量异步组的完成机制。


```cpp
.reg .b16 ctaMask;
.reg .u16 i2cOffW, i2cOffH, i2cOffD;
.reg .b64 l2CachePolicy;

cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes.tile  [sMem0], [tensorMap0, {tc0}], [mbar0];

@p cp.async.bulk.tensor.5d.shared::cta.global.im2col.mbarrier::complete_tx::bytes
                     [sMem2], [tensorMap2, {tc0, tc1, tc2, tc3, tc4}], [mbar2], {i2cOffW, i2cOffH, i2cOffD};

cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.tile  [sMem0], [tensorMap0, {tc0}], [mbar0];

@p cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
                     [sMem1], [tensorMap1, {tc0, tc1}], [mbar2], ctaMask;

@p cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes
                     [sMem2], [tensorMap2, {tc0, tc1, tc2, tc3, tc4}], [mbar2], {i2cOffW, i2cOffH, i2cOffD};

@p cp.async.bulk.tensor.3d.im2col.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint
                     [sMem3], [tensorMap3, {tc0, tc1, tc2}], [mbar3], {i2cOffW}, policy;

@p cp.async.bulk.tensor.1d.global.shared::cta.bulk_group  [tensorMap3, {tc0}], [sMem3];

cp.async.bulk.tensor.2d.tile::gather4.shared::cluster.global.mbarrier::complete_tx::bytes
                     [sMem5], [tensorMap6, {x0, y0, y1, y2, y3}], [mbar5];

cp.async.bulk.tensor.3d.im2col::w.shared::cluster.global.mbarrier::complete_tx::bytes
                     [sMem4], [tensorMap5, {t0, t1, t2}], [mbar4], {im2colwHalo, im2colOff};

cp.async.bulk.tensor.1d.shared::cluster.global.tile.cta_group::2
                     [sMem6], [tensorMap7, {tc0}], [peerMbar];
```

cp.reduce.async.bulk.tensor 详见 [cp.reduce.async.bulk.tensor](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=mma%2520sync%2520aligned%2520m8n8k4#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor)，cp.async.bulk.prefetch.tensor 详见 [cp.async.bulk.prefetch.tensor](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=mma%2520sync%2520aligned%2520m8n8k4#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor)，这里不再详细介绍。

### tensor map

tensor map 用于记录拷贝时 tensor 的相关信息，必须在 host 上创建然后传给 CUDA kernel。具体创建方法可以参考 [tensormap](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma)


创建 tensor map 需要许多参数。其中包括指向全局内存中数组的指针、数组的大小（以元素数量为单位）、行与行之间的步长（以字节为单位）、共享内存的大小（以元素数量为单位）。具体如下，下面代码展示了通过 CUDA API 创建 tensor map的过程。创建的 tensor map 用于描述一个大小为 GMEM_HEIGHT x GMEM_WIDTH 的二维行主数组。

```cpp
  CUtensorMap tensor_map{}; // 定义 tensor map对象
  constexpr uint32_t rank = 2; // 定义 tensor 的维度，最高支持到5维
  uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};  // 确定 tensor 的 shape

  // 确定每一维的 stride，最内层的stride是1，可以忽略。stride的单位是bytes，所以要乘sizeof(float)，大小必须是16bytes的倍数
  uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
  uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT}; // 定义shared memory的shape
  uint32_t elem_stride[rank] = {1, 1}; // smem的stride，默认是1

  // Get a function pointer to the cuTensorMapEncodeTiled driver API.
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
    &tensor_map,                // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32, // 数据类型
    rank,                       // cuuint32_t tensorRank,
    tensor_ptr,                 // void *globalAddress,
    size,                       // const cuuint64_t *globalDim,
    stride,                     // const cuuint64_t *globalStrides,
    box_size,                   // const cuuint32_t *boxDim,
    elem_stride,                // const cuuint32_t *elementStrides,
    // Interleave patterns，在 channel 上进行分块，只能用于 .im2col模式
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    // Swizzling模式，可以避免bank conflict
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    // L2 Promotion
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    // Any element that is outside of bounds will be set to zero by the TMA transfer.
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
```

当从全局内存读取到共享内存的 copy box 部分超出范围时，与超出范围区域对应的共享内存将以零填充。copy box 的左上角索引也可能为负数。当从共享内存写入全局内存时，copy box 的部分内容可能超出范围，但左上角不能有任何负索引。

Tensor 的大小 shape 是沿一个维度的元素数量。所有大小都必须大于 1。步长 stride 是同一维度元素之间的字节数。例如，一个 4 x 4 的整数矩阵的大小为 4 和 4。由于每个元素有 4 个字节，因此步长分别为 4 和 16 个字节。由于对齐要求，一个 4 x 3 的行主整数矩阵的步长也必须为 4 和 16 个字节。每行都会填充 4 个额外字节，以确保下一行的起始位置与 16 个字节对齐。

#### make_gemm_tma_desc

一般可以使用下面的函数创建 Tensor map。

```cpp
template <uint32_t RANK>
CUtensorMap make_gemm_tma_desc(void *gmem_tensor_ptr, std::vector<int> &gmem_shape, std::vector<int> &smem_shape)
{
    CUtensorMap tensor_map{};

    uint64_t gmem_prob_shape[5] = {1, 1, 1, 1, 1};
    uint64_t gmem_prob_stride[5] = {0, 0, 0, 0, 0};
    uint32_t smem_box_shape[5] = {1, 1, 1, 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    gmem_prob_shape[0] = gmem_shape[0];
    gmem_prob_stride[0] = sizeof(float);
    smem_box_shape[0] = smem_shape[0];

    for (int i = 1; i < RANK; ++i)
    {
        gmem_prob_shape[i] = gmem_shape[i];
        gmem_prob_stride[i] = gmem_prob_stride[i - 1] * gmem_shape[i - 1];
        smem_box_shape[i] = smem_shape[i];
    }

    auto tma_format = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    auto tma_interleave = CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE;
    auto smem_swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
    auto tma_l2Promotion = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE;
    auto tma_oobFill = CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    // Create the tensor descriptor.
    CUresult result = cuTensorMapEncodeTiled(
        &tensor_map,          // CUtensorMap *tensorMap,
        tma_format,
        RANK,                 // cuuint32_t tensorRank,
        gmem_tensor_ptr,      // void *globalAddress,
        gmem_prob_shape,      // const cuuint64_t *globalDim,
        gmem_prob_stride + 1, // const cuuint64_t *globalStrides,
        smem_box_shape,       // const cuuint32_t *boxDim,
        smem_box_stride,      // const cuuint32_t *elementStrides,
        tma_interleave,       // Interleave patterns can be used to accelerate loading of values that are less than 4 bytes long.
        smem_swizzle,         // Swizzling can be used to avoid shared memory bank conflicts.
        tma_l2Promotion,      // L2 Promotion can be used to widen the effect of a cache-policy to a wider set of L2 cache lines.
        tma_oobFill           // Any element that is outside of bounds will be set to zero by the TMA transfer.
    );

    if (result != CUDA_SUCCESS)
    {
        std::cerr << "TMA Desc Addr:   " << &tensor_map
                  << "\nformat         " << tma_format
                  << "\ndim            " << RANK
                  << "\ngmem_address   " << gmem_tensor_ptr
                  << "\nglobalDim      " << gmem_prob_shape
                  << "\nglobalStrides  " << gmem_prob_stride
                  << "\nboxDim         " << smem_box_shape
                  << "\nelementStrides " << smem_box_stride
                  << "\ninterleave     " << tma_interleave
                  << "\nswizzle        " << smem_swizzle
                  << "\nl2Promotion    " << tma_l2Promotion
                  << "\noobFill        " << tma_oobFill << std::endl;
        std::cerr << "Error: Failed to initialize the TMA descriptor " << result << std::endl;
        assert(false);
    }

    return tensor_map;
}

```

#### cp_async_bulk_tensor_1d


```cpp
__global__ void cp_async_bulk_tensor_1d(const __grid_constant__ CUtensorMap src_tensor_map, const __grid_constant__ CUtensorMap dst_tensor_map)
{
    int tid = threadIdx.x;
    int crd0 = blockIdx.x * 256;
    __shared__ alignas(128) float smem[256]; // 256 float
    __shared__ alignas(8) uint64_t bar[1];

    int transaction_bytes = blockDim.x * sizeof(float);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(bar);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem);

    if (tid == 0)
    {
        /// Initialize shared memory barrier
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_mbar),
                     "r"(blockDim.x));
        asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_mbar),
                     "r"(transaction_bytes));
        asm volatile("fence.proxy.async.shared::cta;");
        asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%3}], [%2];" ::"r"(smem_int_ptr),
                     "l"(&src_tensor_map), "r"(smem_int_mbar), "r"(crd0) : "memory");
    }
    __syncthreads();

    // arrive
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_int_mbar));

    // wait
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

    // compute

    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();

    // store shared memory to global memory
    if (tid == 0)
    {
        asm volatile("cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%0, {%2}], [%1];" ::"l"(&dst_tensor_map), "r"(smem_int_ptr), "r"(crd0) : "memory");
        asm volatile("cp.async.bulk.commit_group;");
        asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(0) : "memory");
    }
}
```

#### cp_async_bulk_tensor_2d

```cpp
__global__ void cp_async_bulk_tensor_2d(const __grid_constant__ CUtensorMap src_tensor_map, const __grid_constant__ CUtensorMap dst_tensor_map)
{
    int tid = threadIdx.x;
    int crd0 = blockIdx.x % 32 * 32; // cols
    int crd1 = blockIdx.x / 32 * 32; // rows

    __shared__ alignas(128) float smem[1024];
    __shared__ alignas(8) uint64_t bar[1];

    int transaction_bytes = 1024 * sizeof(float);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(bar);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem);

    if (tid == 0)
    {
        /// Initialize shared memory barrier
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_mbar),
                     "r"(blockDim.x));
        asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_mbar),
                     "r"(transaction_bytes));
        asm volatile("fence.proxy.async.shared::cta;");
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%3, %4}], [%2];" ::"r"(smem_int_ptr),
                     "l"(&src_tensor_map), "r"(smem_int_mbar), "r"(crd0), "r"(crd1) : "memory");
    }
    __syncthreads();

    // arrive
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_int_mbar));

    // wait
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

    // compute

    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();

    // store shared memory to global memory
    if (tid == 0)
    {
        asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];" ::"l"(&dst_tensor_map), "r"(smem_int_ptr), "r"(crd0), "r"(crd1) : "memory");
        asm volatile("cp.async.bulk.commit_group;");
        asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(0) : "memory");
    }
}
```

```cpp
// nvcc async_copy.cu -o cpasync -arch=sm_90a -lcuda
int main()
{
    srand(1234);

    int N = 1024 * 1024;

    thrust::host_vector<float> h_S(N);
    thrust::host_vector<float> h_D(N);
    thrust::host_vector<float> copy_result(N);

    for (int i = 0; i < N; ++i)
    {
        h_S[i] = static_cast<float>(i % 1024);
    }

    thrust::device_vector<float> d_S = h_S;
    thrust::device_vector<float> d_D = h_D;

    std::vector<int> gmem_shape = {1024, 1024};
    std::vector<int> smem_shape = {32, 32};

    auto src_gmem_desc = make_gemm_tma_desc<2>(d_S.data().get(), gmem_shape, smem_shape);
    auto dst_gmem_desc = make_gemm_tma_desc<2>(d_D.data().get(), gmem_shape, smem_shape);

    constexpr int threads = 1024;
    int blocks = (N + threads - 1) / threads;

    // cp_async_bulk_tensor_1d<<<blocks, threads>>>(src_gmem_desc, dst_gmem_desc);
    cp_async_bulk_tensor_2d<<<blocks, threads>>>(src_gmem_desc, dst_gmem_desc);

    copy_result = d_D;
    test_copy(h_S.data(), copy_result.data(), N);

    return 0;
}
```

## TMA Swizzle

默认情况下，TMA 会按照数据在全局内存中的布局顺序将数据加载到共享内存中。但是，这种布局对于某些共享内存访问模式可能导致 bank conflict。为了提高性能并减少 bank conflict，TMA 支持使用 swizzle 来更改共享内存布局。关于共享内存 bank conflict 和 swizzle 的介绍详见 swizzle。

在数据传输过程中，TMA 会根据 swizzle 模式对数据进行 shuffle。如下图所示，图中每个色块代表连续的 16 bytes，也就是 4 个 bank，所以一行有128 bytes，32 个 bank。TMA 支持 4 种 swizzle 类型，分别是 None，32B，64B 和 128B。图中从左到右依次是 None，128B，64B 和 32B 对应的swizzle 模式。

None 代表不使用 swizzle，从图中可以看到相同颜色的数据位于相同的 bank，如果访问相同颜色的数据就会产生 bank conflict。

128B 模式可以支持一行 128B，一共 8 行 1024B 的数据参与 swizzle，如左2所示。从图中可以看到，在该模式下相同颜色的数据被 swizzle 到不同的 bank 上，因此访问相同颜色的数据不会发生 bank conflict。

64B 模式可以支持一行 64B，一共 8 行 512B 的数据参与 swizzle，实际 swizzle 的效果等同于 4 行 128B 的效果，但是 64B 模式却不能直接对 4 行 128B 进行swizzle，因为进行 swizzle 的数据宽度不能超过 swizzle 的宽度。从图中可以看到，64B 模式下前 4 行和后 4 行属于重复的 pattern。

同样的 32B 模式可以支持一行 32B，一共 8 行 256B 的数据参与 swizzle，实际 swizzle 的效果等同于 2 行 128B 的效果。

需要注意的是，如果使用 swizzle，则 需要进行 swizzle 的数据的宽度 (假设该维度 stride = 1) 必须小于或等于 swizzle 的跨度。也就是当使用 128B swizzle 时，数据的宽度必须小于等于 128B。当使用 64B swizzle 时，必须小于等于 64B，当使用 32B 时，宽度必须小于等于 32B。

![swizzle-pattern](/assets/cp-async/swizzle-pattern.png "swizzle")

应用 TMA swizzle 模式时，还有一些额外的限制。
1. 全局内存对齐：全局内存必须对齐到 128 字节。
2. 共享内存对齐：共享内存应根据 swizzle 模式重复后的字节数进行对齐。比如 128B 的 swizzle 一共有 1024B 的元素参与，所以需要按照 1024 进行对齐。

共享内存块的内部维度必须满足规定的大小要求。如果不满足这些要求会报错。此外，如果 swizzle 宽度超过内部维度，需要分配的共享内存能够容纳完整的 swizzle 宽度。也就是说如果使用 128B 对 N 行 64B 的数据进行 swizzle，实际分配的共享内存大小需要是 N×128B 大小。

swizzle 中单个元素固定为 16 字节。这意味着数据以 16 字节的块来组织和访问，在规划内存布局和访问模式时必须考虑到这一点。

![swizzle-align](/assets/cp-async/swizzle-align.png "swizzle")

### Swizzle 示例

假如 global memory 的大小是 1024 × 1024，每一行的元素分别是 0 - 1023，然后使用 2D TMA 拷贝数据到 shared memory。
我们使用不同的 swizzle pattern 进行拷贝，并打印 (0, 0) 位置 block 的拷贝结果。

#### 32B swizzle

当我们使用 32B swizzle 时，copy box 的宽度不能超过 32B，所以我们设置 box 为 16 行 8 列，因为元素是 float，所以 8 列刚好等于 32B。

当不用 swizzle 时，shared memory 中的结果如下：
上面是 16 × 8 视图，因为数据在行方向上是连续的，所以下面是 4 × 32 视图，每个元素占用一个 bank。

```cpp
// 16 × 8 view
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7

// 4 × 32 view
0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
```

当使用 32B swizzle 时拷贝结果如下：

```cpp
// 16 × 8 view
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
4 5 6 7 0 1 2 3
4 5 6 7 0 1 2 3
4 5 6 7 0 1 2 3
4 5 6 7 0 1 2 3
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7
4 5 6 7 0 1 2 3
4 5 6 7 0 1 2 3
4 5 6 7 0 1 2 3
4 5 6 7 0 1 2 3

// 4 × 32 view
0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3
0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3
```

如果用不同的颜色表示连续的四个 bank 中的数据，则 32B swizzle 前后分别是

![32B_None_view2](/assets/cp-async/32B_None_view2.png "swizzle")

![32B_swizzle_view2](/assets/cp-async/32B_swizzle_view2.png "swizzle")

#### 64B swizzle

64B swizzle，copy box 的宽度不能超过 64B，所以我们设置 box 大小是 16 × 16。

没有进行 swizzle 时，数据分布如下。

```cpp
// 16 × 16 view
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

// 8 × 32 view
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
```

64B swizzle 的结果是下面这样。上面是 16 × 16 的视图，下面是 8 × 32 的视图。

```cpp
// 16 × 16 view
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11
4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11
8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7
8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7
12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3
12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11
4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11
8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7
8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7
12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3
12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3

// 8 × 32 view
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11
8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7
12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11
8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7
12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3
```

用颜色表示如下：

![64B_none_view2](/assets/cp-async/64B_none_view2.png "swizzle")

![64B_swizzle_view2](/assets/cp-async/64B_swizzle_view2.png "swizzle")

#### 128B swizzle

128B swizzle 时我们设置 copy box 的大小是 16 × 32。下面是没有进行 swizzle 的结果。

```cpp
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
```

下面是 128B swizzle 的结果。

```cpp
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 20 21 22 23 16 17 18 19 28 29 30 31 24 25 26 27
8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 16 17 18 19 20 21 22 23
12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 28 29 30 31 24 25 26 27 20 21 22 23 16 17 18 19
16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
20 21 22 23 16 17 18 19 28 29 30 31 24 25 26 27 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11
24 25 26 27 28 29 30 31 16 17 18 19 20 21 22 23 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7
28 29 30 31 24 25 26 27 20 21 22 23 16 17 18 19 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 20 21 22 23 16 17 18 19 28 29 30 31 24 25 26 27
8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31 16 17 18 19 20 21 22 23
12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 28 29 30 31 24 25 26 27 20 21 22 23 16 17 18 19
16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
20 21 22 23 16 17 18 19 28 29 30 31 24 25 26 27 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11
24 25 26 27 28 29 30 31 16 17 18 19 20 21 22 23 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7
28 29 30 31 24 25 26 27 20 21 22 23 16 17 18 19 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3
```

用颜色表示如下：

![128B_none](/assets/cp-async/128B_none.png "swizzle")

![128B_swizzle](/assets/cp-async/128B_swizzle.png "swizzle")

note：有一点需要注意的是，当 copy box 的宽度小于 swizzle 的宽度时，共享内存也需要按照 swizzle 的宽度进行分配，不然会报错。

比如在 copy box 是32行，16列时，一共有 512 个数据，但是进行 128B swizzle 时 要分配 1024B 大小的空间。

copy box 大小是 32 × 16 时，不 swizzle 结果

```cpp
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
```

copy box 大小是 32 × 16 时，128B swizzle 结果。32 行 16 列在进行 128B swizzle 时因为列数小于 128B，需要多余的空间进行 swizzle。如果大小还是 512 会报内存越界的错误，设置成 1024 可以正常运行。

运行结果如下：

```cpp
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3
```

按照 32 × 32 格式打印，结果如下，可以看到结果比较奇怪。CUTLASS 中这种 copy box 宽度小于 swizzle 宽度的情况会直接报错，最好还是选择合适的shape 和swizzle 吧。

```cpp
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3
```

## Bulk and Tensor copy completion instructions

### cp.async.bulk.commit_group

将所有先前启动但未提交的 cp.async.bulk 指令提交到 cp.async.bulk-group。

```cpp
cp.async.bulk.commit_group;
```

cp.async.bulk.commit_group 指令创建一个新的 bulk async-group，并将所有由执行线程创建，但是还没被提交到任何一个 bulk async-group 的异步指令提交到新创建的 bulk async-group 中。

如果没有未提交的 cp{.reduce}.async.bulk.{.prefetch}{.tensor} 指令，则 cp.async.bulk.commit_group 会导致 bulk async-group 为空。

正在执行的线程可以使用 cp.async.bulk.wait_group 等待 bulk async-group 中所有 cp{.reduce}.async.bulk.{.prefetch}{.tensor} 操作完成。

同一 bulk async-group 中任意两个 cp{.reduce}.async.bulk.{.prefetch}{.tensor} 操作之间不提供内存顺序保证。


### cp.async.bulk.wait_group

等待 bulk async-group 完成。

```cpp
cp.async.bulk.wait_group{.read} N;
```

cp.async.bulk.wait_group 指令将使执行线程等待，直到最近的 bulk async-group 中只有 N 个或更少的 group 处于待处理状态，并且执行线程提交的所有先前的 bulk async-group 均已完成。例如，当 N 为 0 时，执行线程将等待所有先前的 bulk async-group 完成。操作数 N 是一个整数常量。

默认情况下，cp.async.bulk.wait_group 指令将导致执行线程等待，直到指定 bulk async-group 中的所有 bulk 异步操作完成。这里的异步操作包括以下内容：
1. 从 TensorMap 读取。
2. 从 src 位置读取。
3. 写入各自的目标位置。
4. 写入操作对执行线程可见。

当使用 .read 修饰符时，只需要等待下面读取相关的异步操作完成即可。
1. 从 TensorMap 读取
2. 从 src 位置读取。


```cpp
cp.async.bulk.wait_group.read   0;
cp.async.bulk.wait_group        2;
```

