---
title: CUDA C++ 笔记（五）第10章——C++ Language Extensions（一）
date: 2024-06-17 20:00:00
tags: [CUDA, C++ Extensions, CUDA C++]
categories: [CUDA C++ Programming Guide]
description: 本篇涵盖第 10 章 C++ Language Extensions 的前半部分（10.1-10.18），包括函数执行空间说明符、变量存储空间说明符、内建向量类型与变量、Memory Fence、同步函数、只读数据缓存加载、原子函数、地址空间判定与转换、alloca 函数以及编译器优化提示函数。
---

## 10.1 CUDA 中的函数执行空间说明

在 CUDA 编程中，函数会根据其执行场所与可调用方来进行标记，这有助于开发者清晰地管理代码在 CPU（主机）与 GPU（设备）之间的分工。下面是对几类常见函数说明符的思路梳理。

### 函数分类与用途

1. **`__global__` 函数（内核函数）**
   - 执行场所：在设备（GPU）上并行执行。
   - 调用方：通常由主机端发起调用（使用 `<<<...>>>` 的执行配置），在高计算能力的 GPU 上也可由设备端调用（即一个内核调用另一个内核，称为动态并行）。
   - 特性：无返回值（void）、不能是类成员函数、调用后立即返回给主机（异步调用）。

2. **`__device__` 函数（设备端函数）**
   - 执行场所：在设备（GPU）上执行。
   - 调用方：只能在设备端被调用（如在 `__global__` 或 `__device__` 函数内部）。
   - 用途：辅助内核函数进行更细粒度的逻辑封装。

3. **`__host__` 函数（主机端函数）**
   - 执行场所：在主机（CPU）上执行。
   - 调用方：只能由主机端代码调用。
   - 说明：如果函数未标记为设备相关，就相当于是 `__host__` 函数。这是 CPU 上的普通函数。

4. **`__host__ __device__` 双标记函数**
   - 执行场所：在主机和设备上均可编译相应的版本。
   - 用途：一份代码，同时适应 CPU 和 GPU 的逻辑需要。
   - 注意：可使用 `__CUDA_ARCH__` 宏区分当前编译路径是主机还是设备，从而在同一函数中为不同平台编写不同的实现分支。

![函数执行空间说明](/assets/cudacpp-10-1/image.png)

![函数调用关系](/assets/cudacpp-10-1/image1.png)

### 注意事项与潜在问题

不同类型函数间的交叉调用需谨慎：在设备端代码中调用主机函数、或在主机代码中调用设备函数，可能造成未定义行为。

编译器在处理内联（inline）行为时，可通过 `__noinline__`、`__forceinline__`、`__inline_hint__` 等修饰符进行干预，不过需要理解它们之间的互斥关系与使用限制。

通过合理使用这些说明符，我们可以更好地组织 CUDA 程序结构、优化数据和计算的交互，并灵活掌控在 CPU 与 GPU 之间的调用关系。

### 从软件抽象到硬件执行单元：Grid、Block、Thread 与 SM、Warp 的关系及 A100/H100 硬件规模

CUDA 编程中，Grid、Block、Thread 是开发者用于组织并行任务的**软件抽象**，而底层硬件则通过 SM（Streaming Multiprocessor）和 Warp 来实际执行这些线程。理解二者关系有助于更高效地利用 GPU 资源。

#### 软件抽象：Grid、Block、Thread

1. **Thread（线程）**：
   - 每个线程独立执行同一内核代码，对不同数据片段进行处理。
   - 每个线程拥有自己的寄存器、程序计数器和本地内存。

2. **Block（线程块）**：
   - 若干线程组成 Block，共享同一块的共享内存。
   - Block 内线程可使用 `__syncthreads()` 进行同步。
   - Block 大小受限于 SM 资源（寄存器、共享内存），需要选择合适的 blockDim 以充分利用硬件。

3. **Grid（网格）**：
   - 网格由多个 Block 组成。
   - 每个 Block 独立执行相同的内核函数，只是处理不同数据区段。
   - Grid 是一个逻辑组织，不同 Block 间默认无直接同步。

#### 硬件抽象：SM、Warp

1. **SM（Streaming Multiprocessor）**：
   - GPU 中的基本执行单元，每个 SM 含有寄存器、共享内存、调度器等资源。
   - 内核启动时，Grid 中的 Block 会被分配到多个 SM 上执行。
   - 随着 GPU 架构升级（如从 A100 到 H100），SM 数量和性能增强，更高并发度和吞吐量得以实现。

2. **Warp（束）**：
   - 硬件调度的基本单位，每个 Warp 包含 32 个线程。
   - Warp 内线程以 SIMD 方式执行。当线程无分支发散时，Warp 利用率最高。

#### 对应关系与优化启示

- **Thread 与 Warp**：线程是编程抽象，Warp 是硬件实际执行单位。Warp 大小固定为 32 个线程，故建议将 Block 维度设置为 32 的整数倍，以提高 Warp 利用率。
- **Block 与 SM**：每个 Block 一般驻留在一个 SM 上，一个 SM 可同时容纳多个 Block。Block 大小与数量影响 SM 的占用率（Occupancy），合理选择有助于提升性能。
- **Grid 与 GPU**：Grid 中的 Block 会分配至 GPU 上的所有 SM，确保所有硬件资源得到充分利用。

#### A100 与 H100 的硬件规模对编程的启示

**A100 (Ampere)**：
- SM 数量约为 108 个。
- 每个 SM 支持 64 个 Warp 并行执行（64×32 = 2048 线程）。

**H100 (Hopper)**：
- SM 数量约为 144 个，更多 SM 意味着更高的并行度。
- 每 SM 同样支持 64 个 Warp 并行（仍为 2048 线程/SM），整体并发能力更强。

为充分利用这些高端 GPU 的计算潜能，开发者应：

1. 选择合适的 Block 大小：尽量使用 32 的倍数，如 128、256、512，为 Warp 对齐提供便利。
2. 增加并发度：为更高 SM 数量准备足够多的 Block 和线程总数，使每个 SM 都有工作可做，从而提高整体吞吐率。
3. 提升 Occupancy：通过合理安排 Block 尺寸、减少寄存器和共享内存消耗，使更多 Warp 处于就绪状态，在出现延迟时，SM 能快速切换 Warp 以提高执行效率。

![Grid Block Thread 与 SM Warp](/assets/cudacpp-10-1/image2.png)

## 10.2 CUDA 中的变量存储空间说明

在 CUDA 编程中，变量的存储空间说明符用于指示该变量在设备端的存放位置和访问范围。合理使用这些说明符有助于优化数据访问性能。

### 变量存储类型及特性

1. **`__device__` 变量**
   - 默认分配在全局内存（global memory）中。
   - 在 CUDA 上下文（context）生命周期内存活，并且对每个设备（device）有独立的对象副本。
   - 可以从主机（host）通过 `cudaGetSymbolAddress()`、`cudaGetSymbolSize()`、`cudaMemcpyToSymbol()`、`cudaMemcpyFromSymbol()` 等函数访问，也可以在设备端（所有线程）访问。

2. **`__constant__` 变量（可与 `__device__` 同时使用）**
   - 存放在常量内存（constant memory）中。
   - 在 CUDA 上下文生命周期内存活，每个设备有独立的副本。
   - 可被同一网格（grid）中所有线程访问，也可通过主机访问（与 `__device__` 变量类似）。
   - 在并发访问期间修改常量将导致未定义行为（不可在有网格访问该常量的同时从主机端写入）。

3. **`__shared__` 变量（可与 `__device__` 同时使用）**
   - 存放在共享内存（shared memory）中。
   - 在线程块（block）的生命周期内存活，每个 block 有独立的副本。
   - 仅能被所在线程块内的所有线程访问，且地址在不同 block 间不固定。
   - 对于 `extern __shared__` 声明的动态共享内存，数组大小在核函数启动（kernel launch）时决定，需要通过偏移手动管理数据布局与对齐。

4. **`__grid_constant__` 变量（计算架构 >= 7.0）**
   - 用于修饰 `__global__` 函数的 const 参数，使之在网格生命周期内保持不变。
   - 每个网格有独立副本，对所有网格内线程可见，但对其他网格与主机不可见。
   - 因为为 const 修饰，不应修改该对象或其子对象，否则为未定义行为。
   - grid 与 device 区别在于 grid 是软件概念，每次 launch kernel 都会生成，device 是 GPU 硬件。

5. **`__managed__` 变量**
   - 由 CUDA 统一内存（Unified Memory）管理，可在主机和设备代码中通用访问（读写）。
   - 在整个应用程序生命周期内存活。
   - 在需要同时在 CPU 与 GPU 中访问同一数据时非常方便。

6. **`__restrict__` 限定词**
   - 与 C99 中的 `restrict` 类似，用于指示指针不会彼此引用同一块内存（无别名），便于编译器优化。
   - 在 CUDA 代码中使用 `__restrict__` 可提升某些优化的可能性（如消除冗余内存读写），但可能也会因为占用更多寄存器导致性能与占用率折中。

```cpp
void foo(int* __restrict__ a, int* __restrict__ b) {
    a[0] = 1;
    b[0] = a[0] + 2;
}
```

![变量存储空间说明](/assets/cudacpp-10-1/image3.png)

## 10.3 内建向量类型

### 基本向量类型

CUDA 提供 `char`, `short`, `int`, `long`, `longlong`, `float`, `double` 对应的向量类型。例如：

- `int2` 表示一个含有两个 `int` 类型分量的结构，字段名为 `.x` 与 `.y`。
- `float4` 表示一个含有四个 `float` 分量的结构，字段名为 `.x`, `.y`, `.z`, `.w`。

这些类型都有相应的构造函数，如：

```cpp
int2 make_int2(int x, int y);
float4 make_float4(float x, float y, float z, float w);

__global__ void vectorKernel(float4* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // 构造一个 float4
    float4 val = make_float4(idx * 1.0f, idx * 2.0f, idx * 3.0f, idx * 4.0f);
    data[idx] = val;
}
```

**对齐要求：**

CUDA 内建向量类型对齐要求详见参考表格，例如 `int4` 需要 16 字节对齐，`float4` 需要 16 字节对齐。这在进行内存拷贝或使用共享内存时尤其需要注意，以确保数据访问的对齐和性能。

### dim3 类型

`dim3` 是基于 `uint3` 的整数向量类型，常用于指定网格和线程块的维度。声明 `dim3` 变量时，未指定的维度默认为 1。

```cpp
dim3 gridDim(2, 3);  // z 默认 1
dim3 blockDim(16, 16, 1);
```

## 10.4 内建变量

以下内建变量可在**设备端代码（`__device__`、`__global__` 函数中）**使用，用于获取线程组织信息：

- `gridDim`：类型为 `dim3`，表示网格的维度（grid 的 x、y、z 大小）。
- `blockIdx`：类型为 `uint3`，表示当前线程块在网格中的索引（block 的 x、y、z 坐标）。
- `blockDim`：类型为 `dim3`，表示线程块的维度（block 的 x、y、z 大小）。
- `threadIdx`：类型为 `uint3`，表示当前线程在所在线程块中的索引（thread 的 x、y、z 坐标）。
- `warpSize`：类型为 `int`，表示当前设备上 warp 的大小（通常为 32）。

```cpp
__global__ void kernelExample() {
    // 获取线程全局索引 (flat index)
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

    // 也可以访问多维信息
    // 对于二维 grid/block：
    // 全局 x 坐标：blockIdx.x * blockDim.x + threadIdx.x
    // 全局 y 坐标：blockIdx.y * blockDim.y + threadIdx.y

    // 打印当前线程的一些信息 (此处仅示意，实际中可能需要条件打印)
    if (globalIdx == 0) {
        printf("Grid dimensions: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
        printf("Block dimensions: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
        printf("Thread index: (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
        printf("Warp size: %d\n", warpSize);
    }
}
```

## 10.5 CUDA 中的 Memory Fence

在 CUDA 编程中，设备端有着弱排序（weakly-ordered）的内存模型，这意味着不同线程间对内存读写的可见性和顺序并非严格按照程序的书写顺序来保证。尤其是当多个线程同时对同一内存位置执行读/写操作时，如果没有使用同步或内存序约束函数，就可能产生数据竞争（data race），进而导致未定义行为。

**内存栅栏（Memory Fence）函数的作用**：

- `__threadfence_block()`：保证调用该函数之前的所有线程对内存的写操作与读操作都在该 Block 内的其他线程看来已经完成（即在该函数调用之前的内存操作不会被重新排序到该函数之后）。
- `__threadfence()`：作用域从 Block 扩大到整个 Device，即同一设备上的其他线程可以看到在 `__threadfence()` 之前完成的内存写入不会排在此函数调用之后。
- `__threadfence_system()`：作用域进一步扩大到整个 System（包括主机和可能的对等设备）。它确保在此函数之前的所有内存写入在系统范围内被观察时都发生在该函数调用之前。

**总结**：

这些函数让线程在多线程环境下对共享内存、全局内存、对等设备内存或者 page-locked 主机内存的读写操作有一个可预期的顺序，以避免出现不一致的读写结果。

**代码案例解释**

```cpp
__device__ int X = 1, Y = 2;
__device__ void writeXY() {
    X = 10;
    __threadfence(); // 在此之后的内存写入不会被重排序到此之前
    Y = 20;
}

__device__ void readXY() {
    int B = Y;
    __threadfence(); // 确保在此之后的读操作不会被重排序到此之前
    int A = X;
}
```

在上面的 `writeXY` 中，先写 `X = 10`，然后使用 `__threadfence()` 强制之前的操作在设备内具有顺序保证。这样当另外一个线程执行 `readXY` 时，根据内存序的规定，它不可能读到 `A = 10` 而 `B` 还是旧值 2 的情况（即结果 `(A=10, B=2)` 和 `(A=10, B=20)`、`(A=1, B=2)` 是可能的，但 `(A=1, B=20)` 不会出现）。`__threadfence()` 的存在保证了对 **X** 的写入在逻辑顺序上出现在对 **Y** 的写入之前。

**内存栅栏与同步的区别**

需要强调的是，这些内存栅栏函数并不能像 `__syncthreads()` 那样让线程之间真正停下来等待对方。`__threadfence()` 系列函数仅保证内存操作顺序，不保证线程间的执行同步。要实现线程之间的同步与等待，仍需要使用 `__syncthreads()`（用于同一 Block 内）、或使用其他更高级的同步策略（如使用原子操作构建信号机制）。

## 10.6 CUDA 中的同步函数

在 CUDA 中，多线程并行执行的线程可能需要在某些时刻进行同步与内存操作顺序的保证。同步函数有助于在同一个线程块（Block）或同一个 Warp 内协调线程间的数据访问与通信。

![同步函数](/assets/cudacpp-10-1/image7.png)

### 1. `__syncthreads()` 函数

`__syncthreads()` 用于在同一线程块内对所有线程进行同步。当 `__syncthreads()` 执行时：

- 所有执行到该点的线程将等待，直到线程块内的所有其他线程也执行到该点。
- 保证在此函数调用之前对全局和共享内存所做的所有写入对线程块内其他线程可见。

这在处理共享内存（Shared Memory）数据时非常有用。例如，当某些线程写入共享内存，其他线程需要读取更新后的数据时，可以使用 `__syncthreads()` 来确保写入完成且对读者可见。

**示例代码：**

```cpp
__global__ void exampleKernel(float* data) {
    __shared__ float sharedData[256];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 将全局内存的数据拷贝到共享内存
    sharedData[threadIdx.x] = data[idx];

    // 同步，确保所有线程都完成拷贝
    __syncthreads();

    // 现在共享内存中已包含所有线程的数据，后续可安全进行并行处理
    float val = sharedData[threadIdx.x];
    // ... 对 val 进行计算操作 ...

    // 再次同步可根据实际需要决定
    __syncthreads();

    // 将结果写回全局内存
    data[idx] = val;
}
```

需要注意，如果 `__syncthreads()` 出现在条件分支中，那么该条件必须在线程块的所有线程中都一致成立，否则可能导致挂起或未定义行为。

### 2. 带返回值的同步函数

针对具有计算能力（Compute Capability）2.x 及更高版本的设备，CUDA 提供了增强版同步函数，这些函数在同步的同时对线程的条件值（Predicate）进行聚合计算，并返回结果：

- `int __syncthreads_count(int predicate);`
  - 返回线程块内满足 predicate 条件（非零）的线程数目。
- `int __syncthreads_and(int predicate);`
  - 如果线程块内所有线程的 predicate 都为非零，则返回非零，否则返回 0。
- `int __syncthreads_or(int predicate);`
  - 如果线程块内有任意线程的 predicate 为非零，则返回非零，否则返回 0。

**示例代码：**

```cpp
__global__ void syncExample(int* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int val = data[idx];

    // 同步并统计有多少线程的 val 大于 10
    int count = __syncthreads_count(val > 10);
    if (threadIdx.x == 0) {
        printf("Block %d: Threads with val>10: %d\n", blockIdx.x, count);
    }
}
```

### 3. `__syncwarp()` 函数

`__syncwarp(unsigned mask = 0xffffffff)` 是一个针对 Warp 内同步的函数。

- Warp 是 GPU 调度的基本单元，一般为 32 个线程。
- `__syncwarp()` 会让同一个 Warp 内指定的线程（由 mask 指定）在该点进行同步。
- 与 `__syncthreads()` 类似，它也确保在 `__syncwarp()` 前的内存操作在该 Warp 内对其他参与同步的线程可见。

如果 mask 中的线程集与实际参与的线程有不一致（且在较低计算能力的架构下要求一致的收敛性），则行为未定义。

**示例代码：**

```cpp
__global__ void warpSyncExample(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x % 32; // Warp 内的线程号
    unsigned mask = 0xffffffff; // 全部 32 个线程

    float val = data[idx];

    // 对 Warp 内所有线程同步（这里 mask=0xffffffff 意味着全部 Warp 内线程同步）
    __syncwarp(mask);

    // 现在 Warp 内的线程都已经到达此点，可以安全地假设前面的内存操作已完成
    // ... 后续对 val 的处理 ...
    data[idx] = val;
}
```

**总结**

- `__syncthreads()`：在整个 Block 范围内同步。
- `__syncthreads_count()` / `__syncthreads_and()` / `__syncthreads_or()`：在 Block 范围内同步的同时，对线程的谓词结果进行逻辑聚合运算。
- `__syncwarp()`：在 Warp 范围内同步，可用于更精细粒度的同步与内存可见性控制。

## 10.10 只读数据缓存加载函数

```cpp
T __ldg(const T* address);
```

此函数从 `address` 指向的内存中加载数据到只读缓存（read-only data cache）中。

`T` 可以是常见的标量类型（`int`, `float`, `double` 等）、对应的向量类型（`int2`, `float4` 等），以及当包含 `<cuda_fp16.h>` 时支持 `__half`、`__half2`，包含 `<cuda_bf16.h>` 时支持 `__nv_bfloat16`、`__nv_bfloat162` 等。

此操作特别适合在访问模式为只读且高访问频率的场景中使用，从而减少访问全局内存的延迟。

**示例代码：**

```cpp
__global__ void readKernel(const float* __restrict__ input, float* output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        // 使用 __ldg 从只读数据缓存加载
        float val = __ldg(&input[idx]);
        output[idx] = val * 2.0f;
    }
}
```

## 10.11 带缓存提示的加载函数

- `T __ldcg(const T* address);`
- `T __ldca(const T* address);`
- `T __ldcs(const T* address);`
- `T __ldlu(const T* address);`
- `T __ldcv(const T* address);`

这些函数在加载数据时使用不同的缓存操作符（Cache Operators），以针对不同访问模式优化内存访问。支持的类型与 `__ldg()` 类似，均包含常用标量、向量类型以及半精度和 BF16 类型（需包含相应头文件）。

## 10.12 带缓存提示的存储函数

- `void __stwb(T* address, T value);`
- `void __stcg(T* address, T value);`
- `void __stcs(T* address, T value);`
- `void __stwt(T* address, T value);`

这些函数在存储数据时应用特定的缓存操作符，以期在不同内存访问模式下提升性能。类型支持与加载函数一致。

## 10.13 时间函数

- `clock_t clock();`
- `long long int clock64();`

当在设备端代码中执行时，这两个函数返回每个 SM（Streaming Multiprocessor）内部计数器的当前值。该计数器每个时钟周期递增一次。通过在核函数开始和结束时采样该计数器的值并计算差值，可以大致估计该线程所经历的时钟周期数。不过需要注意，这个周期数并不是该线程实际执行指令的纯占用周期数，因为 GPU 的线程执行是通过时分复用进行的。

## 10.14 CUDA 中的原子函数（Atomic Functions）

在并行计算中，当多个线程需要对同一共享变量进行读写更新时，若没有适当的同步机制，极易产生数据竞争（Data Race）并导致未定义行为。为了解决这个问题，CUDA 提供了一组**原子函数（atomic functions）**，可以在全局或共享内存中以原子方式对单个 32 位、64 位或 128 位字（word）执行「读-改-写」操作。

原子函数确保在多个线程并行对同一地址进行操作时，这些操作以不可分割的事务（Transaction）顺序执行，从而避免数据竞争。

### 原子函数的作用域与内存顺序

CUDA 提供多种作用域（scope）和内存序（memory ordering）语义的原子函数：

- 无后缀（如 `atomicAdd`）：在 **device 范围** 原子化（`cuda::thread_scope_device`）。
- `_block` 后缀（如 `atomicAdd_block`）：在 **block 范围** 原子化（`cuda::thread_scope_block`）。
- `_system` 后缀（如 `atomicAdd_system`）：在 **system 范围** 原子化（`cuda::thread_scope_system`）。该功能要求较高的计算能力（如 6.x 及以上）和特定设备支持。

需要注意的是，这里原子化的范围指对该操作在何种层面保证其原子性与可见性。例如，`atomicAdd_system` 可以保证该操作对整个系统（包括其他 GPU 和主机）是原子的，而普通的 `atomicAdd` 仅保证在当前设备的线程间原子性。

### 原子函数的类型支持与实现细节

原子函数可以对以下类型操作：

- 整数类型（`int`, `unsigned int`, `unsigned long long int` 等）
- 浮点类型（`float`, `double`，以及半精度 `__half` 和 `__nv_bfloat16` 等类型在更高的计算能力下支持）
- 一些函数可对矢量类型如 `float2`, `float4` 等进行原子操作（仅在更高架构级别下支持，且这些矢量操作的原子性是针对各元素分别保证的）

**版本支持示例：**

- `atomicAdd(double* addr, double val)` 需要计算能力 6.x 及以上设备支持。
- `atomicAdd(__half *addr, __half val)` 需要计算能力 7.x 及以上设备支持。
- `atomicAdd(float2 *addr, float2 val)` 需要计算能力 9.x 及以上且仅适用于全局内存地址。

如果目标设备不支持某些高精度或特定类型的原子操作，可以通过 `atomicCAS`（比较并交换，Compare-And-Swap）原语自行实现。例如，在计算能力小于 6.0 的设备上实现对 `double` 的 `atomicAdd` 就需要用 `atomicCAS` 来构建。

### 常用原子函数分类

#### 1. 算术类原子操作

- `atomicAdd()`：对指定地址的值进行加法并返回旧值。支持 `int`, `unsigned int`, `unsigned long long int`, `float`, `double`, `__half`, `__nv_bfloat16` 以及对应向量类型（在特定计算能力下）。
- `atomicSub()`：对指定地址的值进行减法并返回旧值。
- `atomicExch()`：将指定值写入地址，并返回该地址原有的值。支持 32/64 位，以及在计算能力 9.x 及以上设备上支持 128 位交换（前提是类型满足对齐和可复制要求）。
- `atomicMin()` 和 `atomicMax()`：对指定地址的值与给定值求最小/最大值，并原子化地写回，返回旧值。64 位版本需要计算能力 5.0 及以上。
- `atomicInc()` 和 `atomicDec()`：对无符号整数进行特殊的增/减操作，带有 wrap-around 行为。
- `atomicCAS()`：比较并交换（Compare and Swap）。如果内存中旧值等于给定的 compare 值，就将其更新为新值，否则保持不变，并返回旧值。支持 16 位、32 位、64 位和在 9.x 及以上设备支持 128 位的原子 CAS。

#### 2. 位操作类原子操作

- `atomicAnd()`、`atomicOr()`、`atomicXor()`：对地址处的值与给定值进行按位与/或/异或运算，并以原子方式更新该地址的值。64 位版本需要计算能力 5.0 及以上。

### 使用场景示例

**统计计数器：**

```cpp
__device__ int counter = 0;

__global__ void incrementCounter() {
    // 将 counter 原子加 1，返回旧值
    int oldVal = atomicAdd(&counter, 1);
    // 使用 oldVal 做一些逻辑判断或统计...
}
```

**分层原子操作：**

```cpp
__global__ void sum(int *input, int *result)
{
    __shared__ int partial_sum;
    // thread 0 is responsible for initializing partial_sum
    if (threadIdx.x == 0)
        partial_sum = 0;
    __syncthreads();
    // each thread updates the partial sum
    atomicAdd(&partial_sum, input[threadIdx.x]);
    __syncthreads();
    // thread 0 updates the total sum
    if (threadIdx.x == 0)
        atomicAdd(result, partial_sum);
}
```

### atomic 和 sync 对比

从性能与适用性角度比较，`atomic` 原子操作与 `__syncthreads()` 都有各自的优劣和适用场景。

若考虑单一操作成本：

- 一个简单的 `__syncthreads()` 调用通常比频繁的原子操作整体代价更低，因为 `__syncthreads()` 不会对内存造成额外的争用，只是一次块内的同步点。
- 原子操作在竞争环境下可能导致显著的序列化和性能降低，可能比多次 `__syncthreads()` 带来的等待更昂贵。

若从应用场景出发：

- 当需要对单个全局变量聚合数据、计数或者实现复杂同步原语时，原子操作不可或缺；即使有额外开销，仍是必要的方案。
- 当只是需要在线程块内部协调共享内存访问顺序、确保数据一致性，`__syncthreads()` 通常更简洁、低开销且更易于使用。

换言之，`__syncthreads()` 和 `atomic` 各有其角色：

- `atomic`：更像是局部的互斥锁机制，适合处理单点数据争用；在高争用下代价高。
- `__syncthreads()`：更像是块内的全线程集合点，用于确保数据准备就绪与一致性，不引发全局内存更新的序列化，却需要等待所有线程到齐。

## 10.15 地址空间判定函数

这些函数在设备端（`__device__`）调用时返回 1 或 0，用于判断指针指向的内存是否属于特定的地址空间。

```cpp
unsigned int __isGlobal(const void *ptr);
//  返回 1 表示 ptr 为全局内存（global memory）地址的通用指针（generic pointer），否则返回 0。

unsigned int __isShared(const void *ptr);
//  返回 1 表示 ptr 为共享内存（shared memory）地址的通用指针，否则返回 0。

unsigned int __isConstant(const void *ptr);
//  返回 1 表示 ptr 为常量内存（constant memory）地址的通用指针，否则返回 0。

unsigned int __isGridConstant(const void *ptr);
//  返回 1 表示 ptr 指向的对象为使用 __grid_constant__ 修饰的内核参数的地址空间（仅适用于计算能力 >= 7.x 的设备），否则返回 0。

unsigned int __isLocal(const void *ptr);
//  返回 1 表示 ptr 为本地内存（local memory）地址的通用指针，否则返回 0。
```

这些函数的行为在参数为 null pointer 时为未指定（unspecified），使用时应确保参数非空。

## 10.16 地址空间转换函数

这些函数可在设备端使用，用于在通用指针与特定地址空间指针之间转换。转换函数以 `__cvta_` 开头，并以指示目标地址空间或源地址空间的后缀结尾。

从通用指针（generic pointer）转换为特定地址空间指针：

```cpp
size_t __cvta_generic_to_global(const void *ptr);
size_t __cvta_generic_to_shared(const void *ptr);
size_t __cvta_generic_to_constant(const void *ptr);
size_t __cvta_generic_to_local(const void *ptr);
```

从特定地址空间的原始值转换为通用指针：

```cpp
void * __cvta_global_to_generic(size_t rawbits);
void * __cvta_shared_to_generic(size_t rawbits);
void * __cvta_constant_to_generic(size_t rawbits);
void * __cvta_local_to_generic(size_t rawbits);
```

使用这些函数时，应了解 PTX 中对应的指令，如 `cvta.to.global`、`cvta.to.shared` 等，用以控制指针地址解析的过程。

## 10.17 alloca 函数

`alloca()` 函数可用于在调用者的栈帧中分配临时内存。与主机端的 `alloca()` 类似，CUDA 在设备端对 `alloca()` 的支持要求计算能力（compute capability）>= 5.2 的设备。

### 函数原型

```cpp
__host__ __device__ void * alloca(size_t size);
```

- 当在设备端调用时，该函数会在当前线程的栈上分配 size 字节的内存，该内存对齐到 16 字节边界。
- 调用函数返回时，这块内存会自动释放。
- 在 Windows 平台上使用 `alloca()` 需要包含 `<malloc.h>` 头文件。
- 使用 `alloca()` 时应注意栈大小限制，避免栈溢出。

**示例**

```cpp
__device__ void foo(unsigned int num) {
    // 在栈上分配 num 个 int4 的空间
    int4 *ptr = (int4 *)alloca(num * sizeof(int4));
    // 使用 ptr
    // ...
}
```

## 10.18 CUDA 编译器优化提示函数

在 CUDA 编程中，有些情况下我们可以向编译器提供额外的优化信息。这些优化提示并不保证程序的逻辑正确性，也不能替代必要的检查与错误处理。但在适当的场景使用这些函数，可以让编译器对代码进行更激进的优化。

### 10.18.1 `__builtin_assume_aligned()`

**用法：**

```cpp
void * __builtin_assume_aligned (const void *exp, size_t align);
void * __builtin_assume_aligned (const void *exp, size_t align, <integral type> offset);
```

通过调用 `__builtin_assume_aligned()` 告诉编译器 `exp` 指针所指向的地址具有至少 `align` 字节的对齐。对于三参数版本，编译器可以假设 `(char*)exp - offset` 按照 `align` 字节对齐。

**示例：**

```cpp
void *res = __builtin_assume_aligned(ptr, 32);
// 编译器可假设 res 至少 32 字节对齐

void *res2 = __builtin_assume_aligned(ptr, 32, 8);
// 编译器可假设 (char*)res2 - 8 至少 32 字节对齐
```

### 10.18.2 `__builtin_assume()`

```cpp
void __builtin_assume(bool exp);
```

`__builtin_assume()` 告诉编译器在运行时 `exp` 条件始终为真。如果条件在实际运行时不满足，行为未定义。通常用于指明某些分支条件一定为真，从而简化分支预测和优化。

**示例：**

```cpp
__device__ int get(int *ptr, int idx) {
    __builtin_assume(idx <= 2);
    return ptr[idx];
}
```

在此示例中，编译器会优化代码为始终满足 `idx <= 2`。

### 10.18.3 `__assume()`

```cpp
void __assume(bool exp);
```

`__assume()` 功能与 `__builtin_assume()` 类似，也是告诉编译器 `exp` 为真。但 `__assume()` 仅在使用 cl.exe 作为主机编译器时支持。

**示例：**

```cpp
__device__ int get(int *ptr, int idx) {
    __assume(idx <= 2);
    return ptr[idx];
}
```

### 10.18.4 `__builtin_expect()`

```cpp
long __builtin_expect (long exp, long c);
```

通过 `__builtin_expect()` 告诉编译器在运行时更有可能出现 `exp == c` 的情况。这主要用于分支预测优化。编译器可以根据此信息对代码布局进行优化，以减少分支不预测成功时的开销。

**示例：**

```cpp
if (__builtin_expect(var, 0))
    doit(); // var 很可能为 0，所以此 if 条件很可能为假
```

### 10.18.5 `__builtin_unreachable()`

```cpp
void __builtin_unreachable(void);
```

使用 `__builtin_unreachable()` 告诉编译器这条语句不可能被执行到。如果实际在运行时执行到这里，将导致未定义行为。编译器可以根据此信息移除一些额外的检查或分支。

**示例：**

```cpp
switch (in) {
    case 1: return 4;
    case 2: return 10;
    default: __builtin_unreachable();
}
```

在此示例中，编译器会认为 default 分支永远不会被执行，从而优化代码布局。

### 10.18.6 限制

- `__assume()` 仅在使用 cl.exe 作为主机编译器时支持。
- 其他函数在所有平台上支持，但如果主机编译器不支持这些函数，那么这些函数必须在 `__device__` 或 `__global__` 函数内部使用，或者在 `__CUDA_ARCH__` 宏被定义时才可用。

通过合理使用上述编译器优化提示函数，开发者可帮助编译器更好地理解代码的行为和数据特性，从而可能获得更好的性能优化效果。当然，这些提示应谨慎使用，一旦约束不符合实际运行情况，会导致未定义行为。
