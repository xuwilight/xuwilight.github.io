---
title: CUDA C++ 笔记（十八）——张量内存加速器 TMA
date: 2024-08-16 20:00:00
tags: [CUDA, TMA, Tensor Memory Accelerator]
categories: [CUDA C++ Programming Guide]
description: 介绍张量内存加速器 (TMA) 的使用，包括一维和多维数组的批量异步复制、张量映射的创建与传输、TMA Swizzle 模式以及设备端张量映射编码。
---

# 张量内存加速器 Tensor Memory Accelerator

## Asynchronous Data Copies using the Tensor Memory Accelerator (TMA)

许多应用程序需要在全局内存之间移动大量数据。通常，数据在全局内存中以多维数组的形式布局，并采用非顺序数据访问模式。为了减少全局内存的使用，此类数组的子块会在计算之前复制到共享内存中。加载和存储过程涉及容易出错且重复的地址计算。为了减轻这些计算负担，Compute Capability 9.0 引入了张量内存加速器 (TMA)。TMA 的主要目标是为多维数组提供一种从全局内存到共享内存的高效数据传输机制。

命名。张量内存加速器 (TMA) 是一个广义术语，用于指代本节中描述的特性。为了向前兼容并减少与 PTX ISA 的差异，本节中的文本将 TMA 操作称为批量异步复制或批量张量异步复制，具体取决于所使用的具体复制类型。"批量"一词用于将这些操作与上文描述的异步内存操作进行对比。

维度。TMA 支持复制一维和多维数组（最多 5 维）。一维连续数组的批量异步复制的编程模型与多维数组的批量张量异步复制的编程模型不同。要执行多维数组的批量张量异步复制，硬件需要一个张量映射。该对象描述了多维数组在全局内存和共享内存中的布局。张量映射通常使用 `cuTensorMapEncode` API 在主机上创建，然后作为带有 `__grid_constant__` 注释的常量内核参数从主机传输到设备。张量映射作为带有 `__grid_constant__` 注释的常量内核参数从主机传输到设备，并可用于在设备上在共享内存和全局内存之间复制数据块。相比之下，执行连续一维数组的批量异步复制不需要张量映射：它可以通过指针和大小参数在设备上执行。

源和目标。批量异步复制操作的源地址和目标地址可以位于共享内存或全局内存中。这些操作可以将数据从全局内存读取到共享内存，将数据从共享内存写入全局内存，也可以将数据从共享内存复制到同一集群中另一个块的分布式共享内存。此外，在集群中，批量异步操作可以指定为多播。在这种情况下，数据可以从全局内存传输到集群内多个块的共享内存。多播功能针对目标架构 `sm_90a` 进行了优化，在其他目标架构上可能会显著降低性能。因此，建议将其与计算架构 `sm_90a` 一起使用。

异步。使用 TMA 的数据传输是异步的。这使得启动线程可以在硬件异步复制数据的同时继续计算。数据传输是否异步进行实际上取决于硬件实现，并且未来可能会发生变化。批量异步操作可以使用多种完成机制来表示操作已完成。当操作从全局内存读取共享内存时，块中的任何线程都可以通过等待共享内存屏障来等待数据在共享内存中可读。当批量异步操作将数据从共享内存写入全局或分布式共享内存时，只有启动线程可以等待操作完成。这是使用基于批量异步组的完成机制来实现的。描述完成机制的表格可以在下方以及 PTX ISA 中找到。

![](/assets/cudacpp-tma/image.png)

### 使用 TMA 传输一维数组

本节演示如何编写一个简单的内核，使用 TMA 对一维数组进行读取-修改-写入操作。本部分展示了如何使用批量异步复制加载和存储数据，以及如何将执行线程与这些复制同步。

内核代码如下所示。某些功能需要内联 PTX 汇编，目前可通过 libcu++ 获得。可以使用以下代码检查这些包装器的可用性：

```cpp
#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
static_assert(false, "Device code is being compiled with older architectures that are incompatible with TMA.");
#endif // __CUDA_MINIMUM_ARCH__
```

内核会经历以下阶段：

1. 初始化共享内存屏障。
2. 启动从全局内存到共享内存的批量异步复制。
3. 到达共享内存屏障并等待。
4. 增加共享内存缓冲区的值。
5. 等待共享内存写入操作对后续批量异步复制可见，即在执行下一步之前，对异步代理中的共享内存写入操作进行排序。
6. 启动将共享内存缓冲区批量异步复制到全局内存的操作。
7. 在内核结束时等待批量异步复制完成共享内存的读取。

```cpp
#include <cuda/barrier>
#include <cuda/ptx>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

static constexpr size_t buf_len = 1024;
__global__ void add_one_kernel(int* data, size_t offset)
{
  // Shared memory buffer. The destination shared memory buffer of
  // a bulk operations should be 16 byte aligned.
  __shared__ alignas(16) int smem_data[buf_len];

  // 1. a) Initialize shared memory barrier with the number of threads participating in the barrier.
  //    b) Make initialized barrier visible in async proxy.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) { 
    init(&bar, blockDim.x);                      // a)
    ptx::fence_proxy_async(ptx::space_shared);   // b)
  }
  __syncthreads();

  // 2. Initiate TMA transfer to copy global to shared memory.
  if (threadIdx.x == 0) {
    // 3a. cuda::memcpy_async arrives on the barrier and communicates
    //     how many bytes are expected to come in (the transaction count)
    cuda::memcpy_async(
        smem_data, 
        data + offset, 
        cuda::aligned_size_t<16>(sizeof(smem_data)),
        bar
    );
  }
  // 3b. All threads arrive on the barrier
  barrier::arrival_token token = bar.arrive();
  
  // 3c. Wait for the data to have arrived.
  bar.wait(std::move(token));

  // 4. Compute saxpy and write back to shared memory
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
    smem_data[i] += 1;
  }

  // 5. Wait for shared memory writes to be visible to TMA engine.
  ptx::fence_proxy_async(ptx::space_shared);   // b)
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // 6. Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    ptx::cp_async_bulk(
        ptx::space_global,
        ptx::space_shared,
        data + offset, smem_data, sizeof(smem_data));
    // 7. Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    ptx::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }
}
```

屏障初始化。屏障使用参与块的线程数进行初始化。因此，只有当所有线程都到达屏障时，屏障才会翻转。共享内存屏障的更详细描述请参见使用 `cuda::barrier` 进行异步数据复制。为了使已初始化的屏障对后续的批量异步复制可见，使用了 `fence.proxy.async.shared::cta` 指令。该指令确保后续的批量异步复制操作在已初始化的屏障上进行。

TMA 读取。批量异步复制指令指示硬件将大量数据复制到共享内存中，并在完成读取后更新共享内存屏障的事务计数。通常，发出尽可能少且尽可能大的批量复制可获得最佳性能。由于复制可以由硬件异步执行，因此无需将复制拆分成更小的块。

启动批量异步复制操作的线程使用 `mbarrier.expect_tx` 到达屏障。此操作由 `cuda::memcpy_async` 自动执行。这会告知屏障线程已到达，以及预计到达的字节数（tx / 事务数）。只需一个线程更新预期事务数。如果多个线程更新事务数，则预期事务数将是所有更新的总和。屏障只有在所有线程和所有字节都到达后才会翻转。屏障翻转后，线程以及后续的批量异步复制都可以安全地从共享内存中读取字节。有关屏障事务核算的更多信息，请参阅 PTX ISA。

屏障等待。使用 `mbarrier.try_wait` 等待屏障翻转。它可以返回 `true`，表示等待结束，也可以返回 `false`，这可能表示等待超时。`while` 循环等待完成，并在超时时重试。

SMEM 写入和同步。缓冲区值的增量用于读写共享内存。为了使写入操作对后续批量异步复制可见，使用了 `fence.proxy.async.shared::cta` 指令。该指令将对共享内存的写入操作排序在后续批量异步复制操作（通过异步代理读取）的读取操作之前。因此，每个线程首先通过 `fence.proxy.async.shared::cta` 对异步代理中共享内存中对象的写入进行排序，并且所有线程的这些操作都排序在线程 0 使用 `__syncthreads()` 执行的异步操作之前。

TMA 写入和同步。从共享内存到全局内存的写入操作同样由单个线程发起。写入操作的完成情况不受共享内存屏障的跟踪。而是使用线程本地机制。多个写入操作可以分批放入所谓的批量异步组中。之后，线程可以等待该组中的所有操作完成从共享内存的读取（如上代码所示）或完成写入全局内存，从而使写入操作对发起线程可见。更多信息，请参阅 `cp.async.bulk.wait_group` 的 PTX ISA 文档。请注意，批量异步和非批量异步复制指令具有不同的异步组：`cp.async.wait_group` 和 `cp.async.bulk.wait_group` 指令均存在。

批量异步指令对其源地址和目标地址有特定的对齐要求。更多信息请参见下表。

<!-- TODO: 对齐要求表格（原图缺失，待补充） -->
![对齐要求表格占位符](/assets/cudacpp-tma/placeholder-align.png)

### Using TMA to transfer multi-dimensional arrays

一维和多维情况的主要区别在于，必须在主机上创建张量映射并将其传递给 CUDA 内核。本节介绍如何使用 CUDA 驱动程序 API 创建张量映射、如何将其传递给设备以及如何在设备上使用它。

驱动程序 API。使用 `cuTensorMapEncodeTiled` 驱动程序 API 创建张量映射。您可以通过直接链接到驱动程序 (`-lcuda`) 或使用 `cudaGetDriverEntryPoint` API 来访问此 API。下面我们将展示如何获取指向 `cuTensorMapEncodeTiled` API 的指针。有关更多信息，请参阅驱动程序入口点访问。

```cpp
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
  // Get pointer to cuTensorMapEncodeTiled
  cudaDriverEntryPointQueryResult driver_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;
  CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status));
  assert(driver_status == cudaDriverEntryPointSuccess);

  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}
```

创建。创建张量映射需要许多参数。其中包括指向全局内存中数组的基指针、数组的大小（以元素数量为单位）、行与行之间的步长（以字节为单位）、共享内存缓冲区的大小（以元素数量为单位）。以下代码创建了一个张量映射，用于描述大小为 `GMEM_HEIGHT x GMEM_WIDTH` 的二维行主数组。请注意参数的顺序：移动速度最快的维度优先。

```cpp
  CUtensorMap tensor_map{};
  // rank is the number of dimensions of the array.
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
  // The box_size is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  uint32_t elem_stride[rank] = {1, 1};

  // Get a function pointer to the cuTensorMapEncodeTiled driver API.
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
    &tensor_map,                // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
    rank,                       // cuuint32_t tensorRank,
    tensor_ptr,                 // void *globalAddress,
    size,                       // const cuuint64_t *globalDim,
    stride,                     // const cuuint64_t *globalStrides,
    box_size,                   // const cuuint32_t *boxDim,
    elem_stride,                // const cuuint32_t *elementStrides,
    // Interleave patterns can be used to accelerate loading of values that
    // are less than 4 bytes long.
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    // Swizzling can be used to avoid shared memory bank conflicts.
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    // L2 Promotion can be used to widen the effect of a cache-policy to a wider
    // set of L2 cache lines.
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    // Any element that is outside of bounds will be set to zero by the TMA transfer.
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
```

主机到设备的传输。有三种方法可以使设备代码访问张量映射。推荐的方法是将张量映射作为 `const __grid_constant__` 参数传递给内核。其他方法包括使用 `cudaMemcpyToSymbol` 将张量映射复制到设备 `__constant__` 内存中，或通过全局内存访问。将张量映射作为参数传递时，某些版本的 GCC C++ 编译器会发出警告"在 GCC 4.6 中，用于传递 64 字节对齐参数的 ABI 已更改"。此警告可以忽略。

```cpp
#include <cuda.h>

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map)
{
   // Use tensor_map here.
}
int main() {
  CUtensorMap map;
  // [ ..Initialize map.. ]
  kernel<<<1, 1>>>(map);
}
```

可以使用全局常量变量来替代 `__grid_constant__` 内核参数。下面提供了一个示例。

```cpp
#include <cuda.h>

__constant__ CUtensorMap global_tensor_map;
__global__ void kernel()
{
  // Use global_tensor_map here.
}
int main() {
  CUtensorMap local_tensor_map;
  // [ ..Initialize map.. ]
  cudaMemcpyToSymbol(global_tensor_map, &local_tensor_map, sizeof(CUtensorMap));
  kernel<<<1, 1>>>();
}
```

最后，可以将张量映射复制到全局内存。使用指向全局设备内存中张量映射的指针，需要在每个线程块中设置栅栏，在该块中的任何线程使用更新后的张量映射之前进行设置。除非再次修改张量映射，否则该线程块对张量映射的后续使用无需设置栅栏。请注意，此机制可能比上述两种机制更慢。

```cpp
#include <cuda.h>
#include <cuda/ptx>
namespace ptx = cuda::ptx;

__device__ CUtensorMap global_tensor_map;
__global__ void kernel(CUtensorMap *tensor_map)
{
  // Fence acquire tensor map:
  ptx::n32_t<128> size_bytes;
  // Since the tensor map was modified from the host using cudaMemcpy,
  // the scope should be .sys.
  ptx::fence_proxy_tensormap_generic(
     ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes
  );
 // Safe to use tensor_map after fence inside this thread..
}
int main() {
  CUtensorMap local_tensor_map;
  // [ ..Initialize map.. ]
  cudaMemcpy(&global_tensor_map, &local_tensor_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  kernel<<<1, 1>>>(global_tensor_map);
}
```

使用。下面的内核从一个更大的二维数组中加载一个大小为 `SMEM_HEIGHT x SMEM_WIDTH` 的二维图块。图块的左上角由索引 `x` 和 `y` 表示。该图块被加载到共享内存中，进行修改，然后写回全局内存。

```cpp
#include <cuda.h>         // CUtensormap
#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map, int x, int y) {
  // The destination shared memory buffer of a bulk tensor operation should be
  // 128 byte aligned.
  __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

  // Initialize shared memory barrier with the number of threads participating in the barrier.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  if (threadIdx.x == 0) {
    // Initialize barrier. All `blockDim.x` threads in block participate.
    init(&bar, blockDim.x);
    // Make initialized barrier visible in async proxy.
    cde::fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    // Initiate bulk tensor copy.
    cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &tensor_map, x, y, bar);
    // Arrive on the barrier and tell how many bytes are expected to come in.
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  } else {
    // Other threads just arrive.
    token = bar.arrive();
  }
  // Wait for the data to have arrived.
  bar.wait(std::move(token));

  // Symbolically modify a value in shared memory.
  smem_buffer[0][threadIdx.x] += threadIdx.x;

  // Wait for shared memory writes to be visible to TMA engine.
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, x, y, &smem_buffer);
    // Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    cde::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    cde::cp_async_bulk_wait_group_read<0>();
  }

  // Destroy barrier. This invalidates the memory region of the barrier. If
  // further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (threadIdx.x == 0) {
    (&bar)->~barrier();
  }
}
```

负索引和越界。当从全局内存读取到共享内存的图块部分超出范围时，与超出范围区域对应的共享内存将以零填充。图块的左上角索引也可能为负数。当从共享内存写入全局内存时，图块的部分内容可能超出范围，但左上角不能有任何负索引。

大小和步长。张量的大小是沿一个维度的元素数量。所有大小都必须大于 1。步长是同一维度元素之间的字节数。例如，一个 4 x 4 的整数矩阵的大小为 4 和 4。由于每个元素有 4 个字节，因此步长分别为 4 和 16 个字节。由于对齐要求，一个 4 x 3 的行主整数矩阵的步长也必须为 4 和 16 个字节。每行都会填充 4 个额外字节，以确保下一行的起始位置与 16 个字节对齐。有关对齐的更多信息，请参阅计算能力 9.0 中多维批量张量异步复制操作的表对齐要求。

<!-- TODO: 多维批量张量异步复制操作的表对齐要求（原图缺失，待补充） -->
![对齐要求表格占位符](/assets/cudacpp-tma/placeholder-align2.png)

#### Multi-dimensional TMA PTX wrappers

下面，PTX 指令按其在上述示例代码中的使用情况排序。

`cp.async.bulk.tensor` 指令在全局内存和共享内存之间启动批量张量异步复制。下面的包装器从全局内存读取数据到共享内存，并从共享内存写入全局内存。

```cpp
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_1d_global_to_shared(
    void *dest, const CUtensorMap *tensor_map , int c0, cuda::barrier<cuda::thread_scope_block> &bar
);

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
    void *dest, const CUtensorMap *tensor_map , int c0, int c1, cuda::barrier<cuda::thread_scope_block> &bar
);

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_3d_global_to_shared(
    void *dest, const CUtensorMap *tensor_map, int c0, int c1, int c2, cuda::barrier<cuda::thread_scope_block> &bar
);

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_4d_global_to_shared(
    void *dest, const CUtensorMap *tensor_map , int c0, int c1, int c2, int c3, cuda::barrier<cuda::thread_scope_block> &bar
);

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_5d_global_to_shared(
    void *dest, const CUtensorMap *tensor_map , int c0, int c1, int c2, int c3, int c4, cuda::barrier<cuda::thread_scope_block> &bar
);
```

```cpp
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_1d_shared_to_global(
    const CUtensorMap *tensor_map, int c0, const void *src
);

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_2d_shared_to_global(
    const CUtensorMap *tensor_map, int c0, int c1, const void *src
);

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_3d_shared_to_global(
    const CUtensorMap *tensor_map, int c0, int c1, int c2, const void *src
);

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_4d_shared_to_global(
    const CUtensorMap *tensor_map, int c0, int c1, int c2, int c3, const void *src
);

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_5d_shared_to_global(
    const CUtensorMap *tensor_map, int c0, int c1, int c2, int c3, int c4, const void *src
);
```

### TMA Swizzle

默认情况下，TMA 引擎会按照数据在全局内存中的布局顺序将数据加载到共享内存中。但是，这种布局对于某些共享内存访问模式可能并非最佳，因为它可能导致共享内存库冲突。为了提高性能并减少库冲突，我们可以通过应用"调配模式"来更改共享内存布局。

共享内存包含 32 个库，这些库的组织方式是将连续的 32 位字映射到连续的库。每个库的带宽为每时钟周期 32 位。在加载和存储共享内存时，如果在事务中多次使用同一个库，则会发生库冲突，从而导致带宽降低。请参阅共享内存，库冲突。

为了确保数据在共享内存中的布局方式能够避免用户代码发生共享内存库冲突，可以指示 TMA 引擎在将数据存储到共享内存之前进行"调配"，并在将数据从共享内存复制回全局内存时进行"取消调配"。张量映射对"混合模式"进行编码，指示使用了哪种混合模式。

#### Example 'Matrix Transpose'

一个例子是矩阵的转置，其中数据从行优先映射到列优先访问。数据在全局内存中以行优先的方式存储，但我们也希望在共享内存中以列优先的方式访问，这会导致存储体冲突。但是，通过使用 128 字节的"swizzle"模式和新的共享内存索引，可以消除冲突。

在本例中，我们将一个 8x8 的 int4 类型矩阵（以行优先的方式存储在全局内存中）加载到共享内存中。然后，每组八个线程从共享内存缓冲区加载一行，并将其存储到单独的转置共享内存缓冲区中的一列。这会导致存储时发生八路存储体冲突。最后，将转置缓冲区写回全局内存。

为了避免存储体冲突，可以使用 `CU_TENSOR_MAP_SWIZZLE_128B` 布局。此布局与 128 字节的行长度匹配，并更改共享内存布局，使列优先和行优先访问在每次事务中不需要相同的存储体。

下方的两个表格（图 27 和图 28）分别展示了 int4 类型的 8x8 矩阵及其转置矩阵的正常和混合共享内存布局。颜色指示矩阵元素映射到八组（每组四个）中的哪一组，边距行和边距列列出了全局内存的行和列索引。表中的条目显示了 16 字节矩阵元素的共享内存索引。

<!-- TODO: 图 27 - 无 swizzle 的共享内存数据布局（原图缺失，待补充） -->
![无 swizzle 的共享内存数据布局占位符](/assets/cudacpp-tma/placeholder-swizzle-none.png)

图 27 在没有 swizzle 的共享内存数据布局中，共享内存索引与全局内存索引相同。每条加载指令都会读取一行并将其存储在转置缓冲区的某一列中。由于转置列的所有矩阵元素都位于同一个存储体中，因此存储操作必须进行序列化，从而导致八个存储事务，即每个存储列存在八路存储体冲突。

<!-- TODO: 图 28 - CU_TENSOR_MAP_SWIZZLE_128B swizzle 的共享内存数据布局（原图缺失，待补充） -->
![128B swizzle 共享内存数据布局占位符](/assets/cudacpp-tma/placeholder-swizzle-128b.png)

采用 `CU_TENSOR_MAP_SWIZZLE_128B` swizzle 的共享内存数据布局。一行存储在一列中，每个矩阵元素分别来自行和列的不同存储体 (bank)，因此不存在任何存储体冲突。

```cpp
__global__ void kernel_tma(const __grid_constant__ CUtensorMap tensor_map) {
   // The destination shared memory buffer of a bulk tensor operation
   // with the 128-byte swizzle mode, it should be 1024 bytes aligned.
   __shared__ alignas(1024) int4 smem_buffer[8][8];
   __shared__ alignas(1024) int4 smem_buffer_tr[8][8];

   // Initialize shared memory barrier
   #pragma nv_diag_suppress static_var_with_dynamic_init
   __shared__ barrier bar;

   if (threadIdx.x == 0) {
     init(&bar, blockDim.x);
     cde::fence_proxy_async_shared_cta();
   }

   __syncthreads();

   barrier::arrival_token token;
   if (threadIdx.x == 0) {
     // Initiate bulk tensor copy from global to shared memory,
     // in the same way as without swizzle.
     cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &tensor_map, 0, 0, bar);
     token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
   } else {
     token = bar.arrive();
   }

   bar.wait(std::move(token));

   /* Matrix transpose
    *  When using the normal shared memory layout, there are eight
    *  8-way shared memory bank conflict when storing to the transpose.
    *  When enabling the 128-byte swizzle pattern and using the according access pattern,
    *  they are eliminated both for load and store. */
   for(int sidx_j = threadIdx.x; sidx_j < 8; sidx_j += blockDim.x){
      for(int sidx_i = 0; sidx_i < 8; ++sidx_i){
         const int swiz_j_idx = (sidx_i % 8) ^ sidx_j;
         const int swiz_i_idx_tr = (sidx_j % 8) ^ sidx_i;
         smem_buffer_tr[sidx_j][swiz_i_idx_tr] = smem_buffer[sidx_i][swiz_j_idx];
      }
   }

   // Wait for shared memory writes to be visible to TMA engine.
   cde::fence_proxy_async_shared_cta();
   __syncthreads();

   /* Initiate TMA transfer to copy the transposed shared memory buffer back to global memory,
    * it will 'unswizzle' the data. */
   if (threadIdx.x == 0) {
     cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, 0, 0, &smem_buffer_tr);
     cde::cp_async_bulk_commit_group();
     cde::cp_async_bulk_wait_group_read<0>();
   }

   // Destroy barrier
   if (threadIdx.x == 0) {
     (&bar)->~barrier();
   }
}

// --------------------------------- main ----------------------------------------

int main(){

...
   void* tensor_ptr = d_data;

   CUtensorMap tensor_map{};
   // rank is the number of dimensions of the array.
   constexpr uint32_t rank = 2;
   // global memory size
   uint64_t size[rank] = {4*8, 8};
   // global memory stride, must be a multiple of 16.
   uint64_t stride[rank - 1] = {8 * sizeof(int4)};
   // The inner shared memory box dimension in bytes, equal to the swizzle span.
   uint32_t box_size[rank] = {4*8, 8};

   uint32_t elem_stride[rank] = {1, 1};

   // Create the tensor descriptor.
   CUresult res = cuTensorMapEncodeTiled(
       &tensor_map,                // CUtensorMap *tensorMap,
       CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
       rank,                       // cuuint32_t tensorRank,
       tensor_ptr,                 // void *globalAddress,
       size,                       // const cuuint64_t *globalDim,
       stride,                     // const cuuint64_t *globalStrides,
       box_size,                   // const cuuint32_t *boxDim,
       elem_stride,                // const cuuint32_t *elementStrides,
       CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
       // Using a swizzle pattern of 128 bytes.
       CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
       CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
       CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
   );

   kernel_tma<<<1, 8>>>(tensor_map);
 ...
}
```

备注：此示例旨在展示 swizzle 的使用，"原样"模式既不高效，也无法扩展到给定的维度之外。

说明：在数据传输过程中，TMA 引擎会根据 swizzle 模式对数据进行 shuffle，如下表所示。这些 swizzle 模式定义了 16 字节数据块沿 swizzle 宽度映射到四个存储体子组。其类型为 `CUtensorMapSwizzle`，有四个选项：无、32 字节、64 字节和 128 字节。请注意，共享内存盒的内部维度必须小于或等于 swizzle 模式的跨度。

#### The Swizzle Modes

如前所述，swizzle 模式有四种。下表展示了不同的 swizzle 模式，包括新共享内存索引的关系。这些表定义了 128 个字节中 16 字节块到 4 个 bank 的 8 个子组的映射。

<!-- TODO: Swizzle 模式映射表格（原图缺失，待补充） -->
![Swizzle 模式映射表格占位符](/assets/cudacpp-tma/placeholder-swizzle-modes.png)

注意事项。应用 TMA swizzle 模式时，务必遵守以下特定的内存要求：

全局内存对齐：全局内存必须对齐到 128 字节。

共享内存对齐：为简单起见，共享内存应根据 swizzle 模式重复后的字节数进行对齐。当共享内存缓冲区未按 swizzle 模式重复后的字节数对齐时，swizzle 模式与共享内存之间存在偏移。请参阅下方注释。

内部维度：共享内存块的内部维度必须满足表 13 中规定的大小要求。如果不满足这些要求，则该指令被视为无效。此外，如果 swizzle 宽度超过内部维度，请确保分配的共享内存能够容纳完整的 swizzle 宽度。

粒度：swizzle 映射的粒度固定为 16 字节。这意味着数据以 16 字节为单位进行组织和访问，在规划内存布局和访问模式时必须考虑到这一点。

Swizzle 模式指针偏移计算。本文将介绍当共享内存缓冲区未按 Swizzle 模式重复的字节数对齐时，如何确定 Swizzle 模式与共享内存之间的偏移量。使用 TMA 时，共享内存需要按 128 字节对齐。要计算共享内存缓冲区相对于 Swizzle 模式偏移了多少倍，请应用相应的偏移量公式。

<!-- TODO: Swizzle 模式指针偏移计算表格（原图缺失，待补充） -->
![Swizzle 指针偏移计算占位符](/assets/cudacpp-tma/placeholder-swizzle-offset.png)

在图 29 中，此偏移量表示初始行偏移量，因此，在 swizzle 索引计算中，它会被添加到行索引 `y` 中。以下代码片段展示了如何在 `CU_TENSOR_MAP_SWIZZLE_128B` 模式下访问 swizzled 共享内存。

```cpp
data_t* smem_ptr = &smem[0][0];
int offset = (reinterpret_cast<uintptr_t>(smem_ptr)/128)%8;
smem[y][((y+offset)%8)^x] = ...
```

摘要。下表"计算能力 9 的不同混合模式的要求和属性"总结了计算能力 9 的不同混合模式的要求和属性。

<!-- TODO: 计算能力 9 不同 swizzle 模式的要求和属性表格（原图缺失，待补充） -->
![Swizzle 模式要求和属性表格占位符](/assets/cudacpp-tma/placeholder-swizzle-summary.png)

## 10.30. Encoding a Tensor Map on Device

前面几节介绍了如何使用 CUDA 驱动程序 API 在主机上创建张量映射。

本节介绍如何在设备上编码平铺类型的张量映射。这在典型的张量映射传输方式（使用 `const __grid_constant__` 内核参数）不理想的情况下非常有用，例如，在单次内核启动中处理一批大小各异的张量时。

推荐的模式如下：

1. 使用主机上的驱动程序 API 创建张量映射"模板"，即 `template_tensor_map`。
2. 在设备内核中，复制 `template_tensor_map`，修改副本，将其存储在全局内存中，并进行适当的隔离。
3. 在内核中使用张量映射并进行适当的隔离。

高级代码结构如下：

```cpp
// Initialize device context:
CUDA_CHECK(cudaDeviceSynchronize());

// Create a tensor map template using the cuTensorMapEncodeTiled driver function
CUtensorMap template_tensor_map = make_tensormap_template();

// Allocate tensor map and tensor in global memory
CUtensorMap* global_tensor_map;
CUDA_CHECK(cudaMalloc(&global_tensor_map, sizeof(CUtensorMap)));
char* global_buf;
CUDA_CHECK(cudaMalloc(&global_buf, 8 * 256));

// Fill global buffer with data.
fill_global_buf<<<1, 1>>>(global_buf);

// Define the parameters of the tensor map that will be created on device.
tensormap_params p{};
p.global_address    = global_buf;
p.rank              = 2;
p.box_dim[0]        = 128; // The box in shared memory has half the width of the full buffer
p.box_dim[1]        = 4;   // The box in shared memory has half the height of the full buffer
p.global_dim[0]     = 256; //
p.global_dim[1]     = 8;   //
p.global_stride[0]  = 256; //
p.element_stride[0] = 1;   //
p.element_stride[1] = 1;   //

// Encode global_tensor_map on device:
encode_tensor_map<<<1, 32>>>(template_tensor_map, p, global_tensor_map);

// Use it from another kernel:
consume_tensor_map<<<1, 1>>>(global_tensor_map);

// Check for errors:
CUDA_CHECK(cudaDeviceSynchronize());
```

以下部分描述了高级步骤。在整个示例中，以下 `tensormap_params` 结构体包含待更新字段的新值。此处包含此结构体以供阅读示例时参考。

```cpp
struct tensormap_params {
  void* global_address;
  int rank;
  uint32_t box_dim[5];
  uint64_t global_dim[5];
  size_t global_stride[4];
  uint32_t element_stride[5];
};
```

### Device-side Encoding and Modification of a Tensor Map

在全局内存中编码张量映射的推荐流程如下。

1. 将现有的张量映射 `template_tensor_map` 传递给内核。与在 `cp.async.bulk.tensor` 指令中使用张量映射的内核不同，这可以通过任何方式完成：指向全局内存的指针、内核参数、`__constant__` 变量等等。
2. 使用 `template_tensor_map` 值在共享内存中复制初始化张量映射。
3. 使用 `cuda::ptx::tensormap_replace` 函数修改共享内存中的张量映射。这些函数包装了 `tensormap.replace` PTX 指令，该指令可用于修改平铺类型张量映射的任何字段，包括基址、大小、步长等等。
4. 使用 `cuda::ptx::tensormap_copy_fenceproxy` 函数，将修改后的张量映射从共享内存复制到全局内存，并执行任何必要的隔离操作。

以下代码包含一个遵循这些步骤的内核。为了完整性，它会修改张量映射的所有字段。通常，内核只会修改几个字段。

在此内核中，`template_tensor_map` 作为内核参数传递。这是将 `template_tensor_map` 从主机移动到设备的首选方式。如果内核打算更新设备内存中现有的张量映射，它可以接受指向现有张量映射的指针进行修改。

注意：张量映射的格式可能会随时间而变化。因此，`cuda::ptx::tensormap_replace` 函数和相应的 `tensormap.replace.tile` PTX 指令被标记为 `sm_90a` 专用。要使用它们，请使用 `nvcc -arch sm_90a` 进行编译。

提示：在 `sm_90a` 上，共享内存中初始化为零的缓冲区也可以用作初始张量映射值。这使得张量映射可以完全在设备上编码，而无需使用驱动程序 API 来编码 `template_tensor_map` 值。

注意：仅支持在设备上修改平铺类型的张量映射；其他类型的张量映射无法在设备上修改。有关张量映射类型的更多信息，请参阅驱动程序 API 参考。

```cpp
#include <cuda/ptx>

namespace ptx = cuda::ptx;

// launch with 1 warp.
__launch_bounds__(32)
__global__ void encode_tensor_map(const __grid_constant__ CUtensorMap template_tensor_map, tensormap_params p, CUtensorMap* out) {
   __shared__ alignas(128) CUtensorMap smem_tmap;
   if (threadIdx.x == 0) {
      // Copy template to shared memory:
      smem_tmap = template_tensor_map;

      const auto space_shared = ptx::space_shared;
      ptx::tensormap_replace_global_address(space_shared, &smem_tmap, p.global_address);
      // For field .rank, the operand new_val must be ones less than the desired
      // tensor rank as this field uses zero-based numbering.
      ptx::tensormap_replace_rank(space_shared, &smem_tmap, p.rank - 1);

      // Set box dimensions:
      if (0 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.box_dim[0]); }
      if (1 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.box_dim[1]); }
      if (2 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.box_dim[2]); }
      if (3 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.box_dim[3]); }
      if (4 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<4>{}, p.box_dim[4]); }
      // Set global dimensions:
      if (0 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<0>{}, (uint32_t) p.global_dim[0]); }
      if (1 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<1>{}, (uint32_t) p.global_dim[1]); }
      if (2 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<2>{}, (uint32_t) p.global_dim[2]); }
      if (3 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<3>{}, (uint32_t) p.global_dim[3]); }
      if (4 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<4>{}, (uint32_t) p.global_dim[4]); }
      // Set global stride:
      if (1 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.global_stride[0]); }
      if (2 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.global_stride[1]); }
      if (3 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.global_stride[2]); }
      if (4 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.global_stride[3]); }
      // Set element stride:
      if (0 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.element_stride[0]); }
      if (1 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.element_stride[1]); }
      if (2 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.element_stride[2]); }
      if (3 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.element_stride[3]); }
      if (4 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<4>{}, p.element_stride[4]); }

      // These constants are documented in this table:
      // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensormap-new-val-validity
      auto u8_elem_type = ptx::n32_t<0>{};
      ptx::tensormap_replace_elemtype(space_shared, &smem_tmap, u8_elem_type);
      auto no_interleave = ptx::n32_t<0>{};
      ptx::tensormap_replace_interleave_layout(space_shared, &smem_tmap, no_interleave);
      auto no_swizzle = ptx::n32_t<0>{};
      ptx::tensormap_replace_swizzle_mode(space_shared, &smem_tmap, no_swizzle);
      auto zero_fill = ptx::n32_t<0>{};
      ptx::tensormap_replace_fill_mode(space_shared, &smem_tmap, zero_fill);
   }
   // Synchronize the modifications with other threads in warp
   __syncwarp();
   // Copy the tensor map to global memory collectively with threads in the warp.
   // In addition: make the updated tensor map visible to other threads on device that
   // for use with cp.async.bulk.
   ptx::n32_t<128> bytes_128;
   ptx::tensormap_cp_fenceproxy(ptx::sem_release, ptx::scope_gpu, out, &smem_tmap, bytes_128);
}
```

### Usage of a Modified Tensor Map

与使用作为 `const __grid_constant__` 内核参数传递的张量映射相比，在全局内存中使用张量映射需要在修改张量映射的线程和使用它的线程之间，在张量映射代理中明确建立释放-获取模式。

该模式的释放部分已在上一节中展示。它使用 `cuda::ptx::tensormap.cp_fenceproxy` 函数完成。

获取部分使用 `cuda::ptx::fence_proxy_tensormap_generic` 函数完成，该函数包装了 `fence.proxy.tensormap::generic.acquire` 指令。如果参与释放-获取模式的两个线程位于同一设备上，则 `.gpu` 作用域即可。如果线程位于不同的设备上，则必须使用 `.sys` 作用域。一旦一个线程获取了张量映射，在进行充分同步（例如使用 `__syncthreads()`）后，块中的其他线程就可以使用它。使用张量映射的线程和执行栅栏的线程必须位于同一个块中。也就是说，如果线程位于同一集群、同一网格或不同内核的两个不同线程块中，则诸如 `cooperative_groups::cluster` 或 `grid_group::sync()` 之类的同步 API 或流顺序同步不足以建立张量映射更新的顺序，也就是说，这些其他线程块中的线程仍然需要在正确的范围内获取张量映射代理，然后才能使用更新后的张量映射。如果没有中间修改，则无需在每个 `cp.async.bulk.tensor` 指令之前重复栅栏。

以下示例展示了栅栏以及张量映射的后续使用。

```cpp
// Consumer of tensor map in global memory:
__global__ void consume_tensor_map(CUtensorMap* tensor_map) {
  // Fence acquire tensor map:
  ptx::n32_t<128> size_bytes;
  ptx::fence_proxy_tensormap_generic(ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes);
  // Safe to use tensor_map after fence..

  __shared__ uint64_t bar;
  __shared__ alignas(128) char smem_buf[4][128];

  if (threadIdx.x == 0) {
    // Initialize barrier
    ptx::mbarrier_init(&bar, 1);
    // Make barrier init visible in async proxy, i.e., to TMA engine
    ptx::fence_proxy_async(ptx::space_shared);
    // Issue TMA request
    ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global, smem_buf, tensor_map, {0, 0}, &bar);

    // Arrive on barrier. Expect 4 * 128 bytes.
    ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, sizeof(smem_buf));
  }
  const int parity = 0;
  // Wait for load to have completed
  while (!ptx::mbarrier_try_wait_parity(&bar, parity)) {}

  // print items:
  printf("Got:\n\n");
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 128; ++i) {
      printf("%3d ", smem_buf[j][i]);
      if (i % 32 == 31) { printf("\n"); };
    }
    printf("\n");
  }
}
```

### Creating a Template Tensor Map Value Using the Driver API

以下代码创建了一个最小的平铺类型张量映射，随后可以在设备上进行修改。

```cpp
CUtensorMap make_tensormap_template() {
  CUtensorMap template_tensor_map{};
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  uint32_t dims_32         = 16;
  uint64_t dims_strides_64 = 16;
  uint32_t elem_strides    = 1;

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
    &template_tensor_map, // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
    1,                // cuuint32_t tensorRank,
    nullptr,          // void *globalAddress,
    &dims_strides_64, // const cuuint64_t *globalDim,
    &dims_strides_64, // const cuuint64_t *globalStrides,
    &dims_32,         // const cuuint32_t *boxDim,
    &elem_strides,    // const cuuint32_t *elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  CU_CHECK(res);
  return template_tensor_map;
}
```
