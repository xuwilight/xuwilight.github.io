---
title: CUDA C++ 笔记（六）第10章——C++ Language Extensions（二）
date: 2024-06-21 20:00:00
tags: [CUDA, Warp, WMMA]
categories: [CUDA C++ Programming Guide]
description: 本篇涵盖第 10 章 C++ Language Extensions 的第二部分（10.19-10.26），包括束内表决函数、束内匹配函数、束内规约函数、束内洗牌函数（Warp Shuffle）、Nanosleep 函数、束内矩阵函数（Warp Matrix Functions / Tensor Cores）、DPX 指令以及异步屏障（Asynchronous Barrier）。
---

## 硬件抽象 - SM、Warp

NVIDIA GPUs 和 CUDA 编程模型采用一种称为 SIMT（单指令，多线程）的执行模型，其中一个重要的概念称为线程束（warp），需要先了解下，才能深入理解介绍的 warp-level 函数。

Warp 是 SM（Streaming Multiprocessor）的基本执行单元，一个 warp 包含 32 个并行 thread，这 32 个 thread 遵循 SIMT 模式，也就是说所有 thread 会执行同一条指令，但每个 thread 会访问各自的数据。CUDA 程序通过显式地利用 warp-level 编程尽可能频繁地一起执行相同的指令序列，从而最大限度地提高性能。

## 10.19 束内表决函数（Warp Vote Functions）

```cpp
int __all_sync(unsigned mask, int predicate);       // 在 warp 中的线程之间交换数据
int __any_sync(unsigned mask, int predicate);       // 在 warp 中的线程之间交换数据
unsigned __ballot_sync(unsigned mask, int predicate); // 在 warp 中的线程之间交换数据
unsigned __activemask();  // 指示 warp 中的哪些线程在当前执行线程中处于活动状态
```

弃用：`__any`、`__all` 和 `__ballot` 在 CUDA 9.0 中已针对所有设备弃用。

删除：当面向具有 7.x 或更高计算能力的设备时，`__any`、`__all` 和 `__ballot` 不再使用，而应使用它们的同步变体（加了 `_sync` 后缀的）。

![Warp Vote Functions](/assets/cudacpp-10-2/image.png)

### 计算逻辑

warp 表决功能**允许给定 warp 的线程执行缩减和广播操作**。这些函数将来自 warp 中每个线程的 int 类型 predicate 作为输入，并将这些值与零进行比较。比较的结果通过以下方式之一在 warp 的活动线程中组合（减少），向每个参与线程广播单个返回值：

- `__all_sync(unsigned mask, predicate)`：评估 mask 中所有未退出线程的 predicate，当且仅当 predicate 对所有线程的评估结果都为非零时，才返回非零值。
- `__any_sync(unsigned mask, predicate)`：评估 mask 中所有未退出线程的 predicate，当且仅当 predicate 对其中任何一个的评估为非零时才返回非零。
- `__ballot_sync(unsigned mask, predicate)`：当且仅当 predicate 对 warp 的第 N 个线程计算为非零并且第 N 个线程处于活动状态时，为 mask 中所有未退出的线程计算 predicate 并返回一个其第 N 位被设置的整型。
- `__activemask()`：返回调用 warp 中所有当前活动线程的 32 位整数掩码。如果调用 `__activemask()` 时，warp 中的第 N 条通道处于活动状态，则设置第 N 位。非活动线程由返回掩码中的 0 位表示。退出程序的线程总是被标记为非活动的。请注意，在 `__activemask()` 调用中收敛的线程不能保证在后续指令中收敛，除非这些指令正在同步 warp 内置函数。

**Note:**

对于 `__all_sync`、`__any_sync` 和 `__ballot_sync`，必须传递一个掩码（mask）来指定参与调用的线程。必须为每个参与线程设置一个表示线程通道 ID 的位，以确保它们在硬件执行内部函数之前正确收敛。掩码中命名的所有活动线程必须使用相同的掩码执行相同的内部函数，否则结果未定义。

### 总结

束内表决函数（Warp Vote Function）是**设备端的内置函数，主要用于进行一些 warp 内数据交换操作**，总的来说具有两个功能：**1-bit 数据交换、1-bit 数据规约**。

束内表决函数在写程序的时候不是必须的，很多源码中很少看到使用，但是有了这个函数，可以让开发人员多一种选择。一方面是这个系列的内置函数出得晚，早期束内数据交换主要使用共享内存。另一方面是可以进行束内数据交换的函数实在太多，比如它的兄弟系列，如束内匹配函数、束内规约函数、束内洗牌函数等等。

## 10.20 束内匹配函数（Warp Match Functions）

`__match_any_sync` 和 `__match_all_sync` **在 warp 中的线程之间执行变量的广播和比较操作**。

由计算能力 7.x 或更高版本的设备支持。

```cpp
unsigned int __match_any_sync(unsigned mask, T value);
unsigned int __match_all_sync(unsigned mask, T value, int *pred);
```

`T` 可以是 `int`、`unsigned int`、`long`、`unsigned long`、`long long`、`unsigned long long`、`float` 或 `double`。

### 计算逻辑

`__match_sync()` 的 intrinsics 允许在对 mask 中命名的线程进行同步之后，在不同的线程之间广播和比较一个值。

- `__match_any_sync`：返回 mask 中具有相同 value 的线程掩码
- `__match_all_sync`：如果掩码中的所有线程的 value 值都相同，则返回 mask；否则返回 0。如果 mask 中的所有线程具有相同的 value 值，则 pred 设置为 true；否则 predicate 设置为假。

新的 `*_sync` 匹配内在函数采用一个掩码，指示参与调用的线程。必须为每个参与线程设置一个表示线程通道 ID 的位，以确保它们在硬件执行内部函数之前正确收敛。掩码中命名的所有非退出线程必须使用相同的掩码执行相同的内在函数，否则结果未定义。

## 10.21 束内规约函数（Warp Reduce Functions）

束内规约函数（Warp Reduce Functions）是 NVIDIA 新引入的一种设备端的内置函数，顾名思义，**主要用于进行 warp 内数据的规约操作**。

`__reduce_sync(unsigned mask, T value)` 内在函数在同步 mask 中命名的线程后对 value 中提供的数据执行归约操作。`T` 对于 `{add, min, max}` 可以是无符号的或有符号的，并且仅对于 `{and, or, xor}` 操作是无符号的。

由计算能力 8.x 或更高版本的设备支持。

```cpp
// add/min/max
unsigned __reduce_add_sync(unsigned mask, unsigned value); // 返回对 mask 中指定的线程中的变量 value 的值进行加法规约操作的计算结果
unsigned __reduce_min_sync(unsigned mask, unsigned value); // 最小值
unsigned __reduce_max_sync(unsigned mask, unsigned value); // 最大值
int __reduce_add_sync(unsigned mask, int value);
int __reduce_min_sync(unsigned mask, int value);
int __reduce_max_sync(unsigned mask, int value);

// and/or/xor
unsigned __reduce_and_sync(unsigned mask, unsigned value); // 逻辑与
unsigned __reduce_or_sync(unsigned mask, unsigned value);  // 逻辑或
unsigned __reduce_xor_sync(unsigned mask, unsigned value); // 异或
```

### 计算逻辑

- `__reduce_add_sync`、`__reduce_min_sync`、`__reduce_max_sync`：返回对 mask 中命名的每个线程在 value 中提供的值应用算术加法、最小或最大规约操作的结果。
- `__reduce_and_sync`、`__reduce_or_sync`、`__reduce_xor_sync`：返回对 mask 中命名的每个线程在 value 中提供的值应用逻辑 AND、OR 或 XOR 规约操作的结果。

### 例子

```cpp
#include <stdio.h>
__global__ void testWarpReduce() {
    int a = threadIdx.x;
    int ret_add = __reduce_add_sync(0xffffffff, a);
    int ret_min = __reduce_min_sync(0xffffffff, a);
    int ret_max = __reduce_max_sync(0xffffffff, a);
    unsigned int b = a & 1;
    unsigned int ret_and = __reduce_and_sync(0xffffffff, b);
    unsigned int ret_or = __reduce_or_sync(0xffffffff, b);
    unsigned int ret_xor = __reduce_xor_sync(0xffffffff, b);
    printf("threadId: %d  reduce_add: %d  reduce_min: %d  reduce_max: %d  reduce_and: %x  reduce_or: %x  reduce_xor: %x\n",
        threadIdx.x, ret_add, ret_min, ret_max, ret_and, ret_or, ret_xor);
}

int main() {
    testWarpReduce<<<1, 32>>>();
    return 0;
}
```

## 10.22 束内洗牌函数（Warp Shuffle Functions）

束内洗牌函数（Warp Shuffle Functions）是设备端的内置函数，与束内表决函数一样，主要用于进行一些 warp 内数据交换操作。但有两个主要区别：**没有数据的规约处理功能、交换的数据通常大于 1-bit。**

```cpp
T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize);
```

`__shfl_sync`、`__shfl_up_sync`、`__shfl_down_sync` 和 `__shfl_xor_sync` 在 warp 内的线程之间交换变量。

由计算能力 3.x 或更高版本的设备支持。

弃用通知：`__shfl`、`__shfl_up`、`__shfl_down` 和 `__shfl_xor` 在 CUDA 9.0 中已针对所有设备弃用。

`T` 可以是 `int`、`unsigned int`、`long`、`unsigned long`、`long long`、`unsigned long long`、`float` 或 `double`。包含 **`cuda_fp16.h` 头文件**后，**T 也可以是 `__half` 或 `__half2`**。同样，包含 `cuda_bf16.h` 头文件后，T 也可以是 `__nv_bfloat16` 或 `__nv_bfloat162`。

### 计算逻辑

`__shfl_sync()` 内在函数允许在 warp 内的线程之间交换变量，而无需使用共享内存。交换同时发生在 warp 中的所有活动线程（并以 mask 命名），根据类型移动每个线程 4 或 8 个字节的数据。

warp 中的线程称为通道（lanes），并且可能具有介于 0 和 `warpSize-1`（包括）之间的索引。支持四种源通道（source-lane）寻址模式：

- `__shfl_sync()`：从索引通道直接复制
- `__shfl_up_sync()`：从相对于调用者 ID 较低的通道复制
- `__shfl_down_sync()`：从相对于调用者具有更高 ID 的通道复制
- `__shfl_xor_sync()`：基于自身通道 ID 的按位异或从通道复制

线程只能从积极参与 `__shfl_sync()` 命令的另一个线程读取数据。如果目标线程处于非活动状态，则检索到的值未定义。

所有 `__shfl_sync()` 内在函数都采用一个可选的宽度参数，该参数会改变内在函数的行为。`width` 的值必须是 2 的幂；如果 width 不是 2 的幂，或者是大于 warpSize 的数字，则结果未定义。

- `__shfl_sync()` 返回由 `srcLane` 给定 ID 的线程持有的 `var` 的值。如果 width 小于 warpSize，则 warp 的每个子部分都表现为一个单独的实体，其起始逻辑通道 ID 为 0。如果 srcLane 超出范围 [0:width-1]，则返回的值对应于通过 `srcLane modulo width` 所持有的 var 的值（即在同一部分内）。
- `__shfl_up_sync()` 通过从调用者的通道 ID 中减去 delta 来计算源通道 ID。返回由生成的通道 ID 保存的 var 的值：实际上，var 通过 delta 通道向上移动。如果宽度小于 warpSize，则 warp 的每个子部分都表现为一个单独的实体，起始逻辑通道 ID 为 0。源通道索引不会环绕宽度值，因此实际上较低的 delta 通道将保持不变。
- `__shfl_down_sync()` 通过将 delta 加调用者的通道 ID 来计算源通道 ID。返回由生成的通道 ID 保存的 var 的值：这具有将 var 向下移动 delta 通道的效果。如果 width 小于 warpSize，则 warp 的每个子部分都表现为一个单独的实体，起始逻辑通道 ID 为 0。至于 `__shfl_up_sync()`，源通道的 ID 号不会环绕宽度值，因此 upper delta lanes 将保持不变。
- `__shfl_xor_sync()` 通过对调用者的通道 ID 与 laneMask 执行按位异或来计算源通道 ID：返回结果通道 ID 所持有的 var 的值。如果宽度小于 warpSize，那么每组宽度连续的线程都能够访问早期线程组中的元素，但是如果它们尝试访问后面线程组中的元素，则将返回它们自己的 var 值。这种模式实现了一种蝶式寻址模式，例如用于树规约和广播。

### 例子

#### 在 warp 内广播单个变量的值

```cpp
#include <stdio.h>
__global__ void bcast(int arg) {
    // 获取当前线程的 lane ID（线程在 warp 中的位置）
    int laneId = threadIdx.x & 0x1f;
    int value;
    // 只有 lane ID 为 0 的线程（即 warp 中的第一个线程）才会将参数 arg 赋值给 value
    if (laneId == 0) value = arg;

    // 使用 __shfl_sync 函数，warp 中的所有线程同步，并从 lane 0 获取 value 的值
    value = __shfl_sync(0xffffffff, value, 0);
    printf("value: %d\n", value);
    // 检查每个线程获取的 value 是否等于传入的参数 arg
    if (value != arg)
        printf("线程 %d 失败。\n", threadIdx.x);
}
int main() {
    bcast<<< 1, 32 >>>(1234);
    cudaDeviceSynchronize();
    return 0;
}
```

#### 在含有 8 个线程的子线程组内进行包含扫描

```cpp
#include <stdio.h>
__global__ void scan4() {
    int laneId = threadIdx.x & 0x1f;
    int value = 31 - laneId;
    // 循环累加扫描操作，对于 8 个线程，扫描需要 log2(n) == 3 步
    for (int i = 1; i <= 4; i *= 2) {
        int n = __shfl_up_sync(0xffffffff, value, i, 8);
        if ((laneId & 7) >= i)
            value += n;
    }
    printf("线程 %d 的最终值 = %d\n", threadIdx.x, value);
}
int main() {
    scan4<<< 1, 32 >>>();
    cudaDeviceSynchronize();
    return 0;
}
```

#### 束内规约

```cpp
#include <stdio.h>
__global__ void warpReduce() {
    int laneId = threadIdx.x & 0x1f;
    int value = 31 - laneId;
    // 使用异或模式（XOR mode）进行蝴蝶归约（butterfly reduction）
    for (int i = 16; i >= 1; i /= 2)
        value += __shfl_xor_sync(0xffffffff, value, i, 32);
    // 此时，"value" 包含了 warp 中所有线程的累加和
    printf("线程 %d 的最终值 = %d\n", threadIdx.x, value);
}
int main() {
    warpReduce<<< 1, 32 >>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### 小总结

相比使用共享内存进行线程间数据交换，束内洗牌函数具有如下特点：

- 不需要为参与数据交换的 warp 分配共享内存，这样可以减少共享内存的使用。
- warp shuffle 可以直接交换，不需要进行显式同步。
- warp shuffle 的效率要高于基于共享内存的数据交换，因此除非 warp shuffle 满足不了计算诉求，否则都应该使用 warp shuffle 而不是共享内存。

## 10.23 Nanosleep 函数

```cpp
T __nanosleep(unsigned ns);
```

### 计算逻辑

`__nanosleep(ns)` 将线程挂起大约接近延迟 ns 的睡眠持续时间，以纳秒为单位指定。它受计算能力 7.0 或更高版本的支持。

### 例子

以下代码实现了一个具有指数回退的互斥锁。

```cpp
__device__ void mutex_lock(unsigned int *mutex) {
    unsigned int ns = 8;
    while (atomicCAS(mutex, 0, 1) == 1) {
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
    }
}
__device__ void mutex_unlock(unsigned int *mutex) {
    atomicExch(mutex, 0);
}
```

## 10.24 束内矩阵函数（Warp Matrix Functions）

C++ warp 矩阵运算利用 Tensor Cores 来加速 `D = A*B + C` 形式的矩阵问题。计算能力 7.0 或更高版本的设备的混合精度浮点数据支持这些操作。这需要一个 warp 中所有线程的合作。此外，仅当条件在整个 warp 中的计算结果相同时，才允许在条件代码中执行这些操作，否则代码执行可能会挂起。

以下所有函数和类型都在命名空间 `nvcuda::wmma` 中定义。Sub-byte 操作被视为预览版，即它们的数据结构和 API 可能会发生变化，并且可能与未来版本不兼容。这个额外的功能在 `nvcuda::wmma::experimental` 命名空间中定义。

```cpp
template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;

void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm);
void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout);
void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);
void fill_fragment(fragment<...> &a, const T& v);
void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);
```

### 计算逻辑

**fragment**：包含矩阵的一部分的重载类，分布在 warp 中的所有线程中。矩阵元素到 fragment 内部存储的映射是未指定的，并且在未来的架构中可能会发生变化。

只允许模板参数的某些组合。第一个模板参数指定片段将如何参与矩阵运算。可接受的使用值是：

- `matrix_a`：当 fragment 用作第一个被乘数时，A
- `matrix_b`：当 fragment 用作第二个被乘数时，B
- `accumulator`：当 fragment 用作源或目标累加器（分别为 C 或 D）时的累加器。

m、n 和 k 大小描述了参与乘法累加操作的 warp-wide 矩阵块的形状。每个 tile 的尺寸取决于它的作用。对于 matrix_a，图块的尺寸为 m x k；对于 matrix_b，维度是 k x n，累加器块是 m x n。

**load_matrix_sync**：等到所有 warp 通道（lanes）都到达 `load_matrix_sync`，然后从内存中加载矩阵片段 a。`mptr` 必须是一个 256 位对齐的指针，指向内存中矩阵的第一个元素。`ldm` 描述连续行（对于行主序）或列（对于列主序）之间的元素跨度，对于 `__half` 元素类型必须是 8 的倍数，对于浮点元素类型必须是 4 的倍数。

**store_matrix_sync**：等到所有 warp 通道都到达 `store_matrix_sync`，然后将矩阵片段 a 存储到内存中。

**fill_fragment**：用常量 v 填充矩阵片段。由于未指定矩阵元素到每个片段的映射，因此该函数通常由 warp 中的所有线程调用，并具有共同的 v 值。

**mma_sync**：等到所有 warp lanes 都到达 `mma_sync`，然后执行 warp 同步的矩阵乘法累加操作 `D = A*B + C`。还支持原位（in-place）操作，`C = A*B + C`。

### 替代类型浮点数

Tensor Core 支持在具有 8.0 及更高计算能力的设备上进行替代类型的浮点运算。

- **`__nv_bfloat16`**：此数据格式是另一种 fp16 格式，其范围与 f32 相同，但精度降低（7 位）。您可以直接将此数据格式与 `cuda_bf16.h` 中提供的 `__nv_bfloat16` 类型一起使用。具有 `__nv_bfloat16` 数据类型的矩阵片段需要与浮点类型的累加器组合。支持的形状和操作与 `__half` 相同。
- **`tf32`**：这种数据格式是 Tensor Cores 支持的特殊浮点格式，范围与 f32 相同，但精度降低（>=10 位）。这种格式的内部布局是实现定义的。为了在 WMMA 操作中使用这种浮点格式，输入矩阵必须手动转换为 tf32 精度。唯一支持的矩阵大小是 16x16x8 (m-n-k)。

### 双精度浮点数

Tensor Core 支持计算能力 8.0 及更高版本的设备上的双精度浮点运算。要使用这个新功能，必须使用具有 `double` 类型的片段。`mma_sync` 操作将使用 `.rn`（四舍五入到最接近的偶数）舍入修饰符执行。

### Sub-byte Operations

Sub-byte WMMA 操作提供了一种访问 Tensor Core 的低精度功能的方法。它们被视为预览功能，即它们的数据结构和 API 可能会发生变化，并且可能与未来版本不兼容。此功能可通过 `nvcuda::wmma::experimental` 命名空间获得。

对于 4 位精度，可用的 API 保持不变，但您必须指定 `experimental::precision::u4` 或 `experimental::precision::s4` 作为片段数据类型。

### 例子

以下代码在单个 warp 中实现 16x16x16 矩阵乘法：

```cpp
#include <mma.h>
using namespace nvcuda;

__global__ void wmma_ker(half *a, half *b, float *c) {
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}
```

## 10.25 DPX

**Dynamic programming X**

DPX 指令集是 Hopper 架构的新指令，Hopper 引入了一组名为 DPX 的新指令集，DPX 可加速动态规划编程算法，解决路径优化、基因组学等算法优化问题，与 CPU 和上一代 GPU 相比，其速度提升分别可达 40 倍和 7 倍。

DPX 的一个示例是使用 Floyd-Warshall 全对最短路径算法进行具有数百或数千个约束或权重的路由优化。另一个用例是使用 Needleman-Wunsch 或 Smith-Waterman 算法进行基因组序列比对的读取比对。

## 10.26 异步屏障（Asynchronous Barrier）

NVIDIA C++ 标准库引入了 [std::barrier](https://nvidia.github.io/libcudacxx/extended_api/synchronization_primitives/barrier.html) 的 GPU 实现。除了 std::barrier 的实现，**该库还提供允许用户指定屏障对象范围的扩展**。计算能力 8.0 或更高版本的设备为屏障操作和这些屏障与 memcpy_async 功能的集成提供硬件加速。在计算能力低于 8.0 但从 7.0 开始的设备上，这些屏障在没有硬件加速的情况下可用。

`nvcuda::experimental::awbarrier` 被弃用，取而代之的是 `cuda::barrier`。

### 简单同步模式

在没有到达/等待障碍的情况下，使用 `__syncthreads()`（同步块中的所有线程）或 `group.sync()` 使用[协作组](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)时实现同步。

```cpp
#include <cooperative_groups.h>

__global__ void simple_sync(int iteration_count) {
    auto block = cooperative_groups::this_thread_block();
    for (int i = 0; i < iteration_count; ++i) {
        /* code before arrive */
        block.sync(); /* wait for all threads to arrive here */
        /* code after wait */
    }
}
```

### 时间分割同步模式

使用 std::barrier 的时间分割同步模式如下。

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

__device__ void compute(float* data, int curr_iteration);

__global__ void split_arrive_wait(int iteration_count, float *data) {
    using barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__  barrier bar;
    auto block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0) {
        init(&bar, block.size()); // Initialize the barrier with expected arrival count
    }
    block.sync();

    for (int curr_iter = 0; curr_iter < iteration_count; ++curr_iter) {
        /* code before arrive */
        barrier::arrival_token token = bar.arrive(); /* this thread arrives. Arrival does not block a thread */
        compute(data, curr_iter);
        bar.wait(std::move(token)); /* wait for all threads participating in the barrier to complete bar.arrive()*/
        /* code after wait */
    }
}
```

在此模式中，同步点（`block.sync()`）分为到达点（`bar.arrive()`）和等待点（`bar.wait(std::move(token))`）。一个线程通过第一次调用 `bar.arrive()` 开始参与 `cuda::barrier`。当一个线程调用 `bar.wait(std::move(token))` 时，它将被阻塞，直到参与线程完成 `bar.arrive()` 的预期次数，该次数由传递给 `init()` 的预期到达计数参数指定。请注意，对 `bar.arrive()` 的调用不会阻塞线程，它可以继续其他不依赖于在其他参与线程调用 `bar.arrive()` 之前发生的内存更新的工作。

### Bootstrap Initialization, Expected Arrival Count, and Participation

在任何线程可以参与 `cuda::barrier` 之前，必须使用带有预期到达计数的 `init()` 初始化屏障。必须在任何线程调用 `bar.arrive()` 之前进行初始化。这带来了一个引导挑战，因为线程必须在参与 `cuda::barrier` 之前进行同步，但是线程正在创建 `cuda::barrier` 以进行同步。

`cuda::barrier` 可以灵活地指定线程如何参与（拆分到达/等待）以及哪些线程参与。相比之下，来自协作组的 `this_thread_block.sync()` 或 `__syncthreads()` 适用于整个线程块，而 `__syncwarp(mask)` 是 warp 的指定子集。如果用户的意图是同步一个完整的线程块或一个完整的 warp，出于性能原因，建议分别使用 `__syncthreads()` 和 `__syncwarp(mask)`。

### A Barrier's Phase: Arrival, Countdown, Completion, and Reset

当参与线程调用 `bar.arrive()` 时，`cuda::barrier` 从预期到达计数倒数到零。当倒计时达到零时，当前阶段的 `cuda::barrier` 就完成了。当最后一次调用 `bar.arrive()` 导致倒计时归零时，倒计时会自动重置。重置将倒计时分配给预期到达计数，并将 `cuda::barrier` 移动到下一阶段。

### Spatial Partitioning (also known as Warp Specialization)

线程块可以在空间上进行分区，以便 warp 专门用于执行独立计算。空间分区用于生产者或消费者模式，其中一个线程子集产生的数据由另一个（不相交的）线程子集同时使用。

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;

__device__ void producer(barrier ready[], barrier filled[], float* buffer, float* in, int N, int buffer_len)
{
    for (int i = 0; i < (N/buffer_len); ++i) {
        ready[i%2].arrive_and_wait(); /* wait for buffer_(i%2) to be ready to be filled */
        /* produce, i.e., fill in, buffer_(i%2)  */
        barrier::arrival_token token = filled[i%2].arrive(); /* buffer_(i%2) is filled */
    }
}

__device__ void consumer(barrier ready[], barrier filled[], float* buffer, float* out, int N, int buffer_len)
{
    barrier::arrival_token token1 = ready[0].arrive(); /* buffer_0 is ready for initial fill */
    barrier::arrival_token token2 = ready[1].arrive(); /* buffer_1 is ready for initial fill */
    for (int i = 0; i < (N/buffer_len); ++i) {
        filled[i%2].arrive_and_wait(); /* wait for buffer_(i%2) to be filled */
        /* consume buffer_(i%2) */
        barrier::arrival_token token = ready[i%2].arrive(); /* buffer_(i%2) is ready to be re-filled */
    }
}

__global__ void producer_consumer_pattern(int N, int buffer_len, float* in, float* out) {
    __shared__ extern float buffer[];
    __shared__ barrier bar[4];

    auto block = cooperative_groups::this_thread_block();
    if (block.thread_rank() < 4)
        init(bar + block.thread_rank(), block.size());
    block.sync();

    if (block.thread_rank() < warpSize)
        producer(bar, bar+2, buffer, in, N, buffer_len);
    else
        consumer(bar, bar+2, buffer, out, N, buffer_len);
}
```

`bar.arrive_and_wait()` 等价于 `bar.wait(bar.arrive())`。

### Early Exit (Dropping out of Participation)

当参与同步序列的线程必须提前退出该序列时，该线程必须在退出之前显式退出参与。

```cpp
bar.arrive_and_drop();
```

此操作到达 `cuda::barrier` 以履行参与线程到达当前阶段的义务，然后减少下一阶段的预期到达计数，以便不再期望该线程到达屏障。

### Completion Function

Completion Function 的功能是**让 barrier 在 arrive 计数减到 0 时进行函数回调。**

### Memory Barrier Primitives Interface

内存屏障原语是 `cuda::barrier` 功能的 C 类型（C-like）接口。这些原语可通过包含 `<cuda_awbarrier_primitives.h>` 头文件获得。

```cpp
typedef /* implementation defined */ __mbarrier_t;
typedef /* implementation defined */ __mbarrier_token_t;

uint32_t __mbarrier_maximum_count();
void __mbarrier_init(__mbarrier_t* bar, uint32_t expected_count);
void __mbarrier_inval(__mbarrier_t* bar);
__mbarrier_token_t __mbarrier_arrive(__mbarrier_t* bar);
__mbarrier_token_t __mbarrier_arrive_and_drop(__mbarrier_t* bar);
bool __mbarrier_test_wait(__mbarrier_t* bar, __mbarrier_token_t token);
```
