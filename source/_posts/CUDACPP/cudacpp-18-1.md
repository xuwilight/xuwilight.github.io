---
title: CUDA C++ 笔记（十一）第17章——C++ Language Support（一）
date: 2024-07-19 20:00:00
tags: [CUDA, CUDA C++]
categories: [CUDA C++ Programming Guide]
description: 第17章 C++ Language Support 第一部分，介绍 CUDA 对 C++11/14/17/20 语言特性的支持，以及 NVCC 编译器的各种限制（内存空间说明符、函数、类、模板等）。
---

# 第17章 C++ Language Support Part1

17.1-17.5

如使用 [NVCC 编译中所述](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compilation-with-nvcc)，使用 nvcc 编译的 CUDA 源文件可以包含主机代码和设备代码的混合。CUDA 前端编译器旨在模拟主机编译器对 C++ 输入代码的行为。输入源代码根据 C++ ISO/IEC 14882:2003、C++ ISO/IEC 14882:2011、C++ ISO/IEC 14882:2014 或 C++ ISO/IEC 14882:2017 规范进行处理，CUDA 前端编译器旨在模拟任何主机编译器与 ISO 规范的差异。此外，支持的语言使用本文档中描述的特定于 CUDA 的结构进行了扩展，并受到下面描述的限制。

[C++11 语言特性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cpp11-language-features)、[C++14](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cpp14-language-features) 语言特性和 [C++17](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cpp17-language-features) 语言特性分别为 C++11、C++14、C++17 和 C++20 特性提供支持矩阵。[限制](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#restrictions)列出了语言限制。[多态函数包装器](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#polymorphic-function-wrappers)和[扩展 Lambda](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda) 描述了其他特性。[代码示例](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#code-samples)提供代码示例。

![](/assets/cudacpp-18-1/image.png)

CUDA 编译的工作原理如下：输入程序首先被设备编译器（nvcc）编译，即在设备编译器（nvcc）预处理过程，并将 CUDA 相关代码（主要是核函数）编译为放置在 fatbinary 中的 CUDA 二进制（Cubin）和/或 PTX 中间代码，并将 CUDA 特定的 C++ 扩展转换为标准 C++ 构造合成嵌入 fatbinary。输入程序再被主机端编译器编译即主机端的预处理，当两个预处理过程完成后，C++ 主机编译器将 fatbinary 的嵌入合成到主机对象（库文件或可执行文件）。

可以看到，编译过程其实分两部分，一部分是主机端和普通 C++ 一样的编译，另一部分是针对 CUDA 中扩展的 C++ 程序的编译，设备端的编译最终的结果文件为 fatbinary 文件，GPU（的驱动）通过 fatbinary 文件来执行 GPU 功能。

为什么是 fatbinary，fat 是肥的意思，为了让应用程序适应不同的 GPU，fatbinary 里可能会有多种 GPU 的实现，程序在运行的时候会根据自己的特点选择合适的最高效的 GPU 实现进行运行：

CUDA 运行时系统（GPU 驱动程序）会监视 fatbinary 文件中的内容，每次程序运行时，CUDA 运行时系统（GPU 驱动程序）都会找到 fatbinary 中最合适部分并映射到当前 GPU。（fatbinary 会有适合不同的 GPU 的实现）

## 17.1. C++11 Language Features

下表列出了已被 C++11 标准接受的新语言功能。"Proposal" 列提供了描述该功能的 ISO C++ 委员会提案的链接，而 "Available in nvcc (device code)" 列表示包含此功能实现的第一个 nvcc 版本（如果已实现）用于设备代码。

Table 12. C++11 Language Features

| Language Feature | C++11 Proposal | Available in nvcc (device code) |
|---|---|---|
| Rvalue references | N2118 | 7.0 |
| Rvalue references for *this | N2439 | 7.0 |
| Initialization of class objects by rvalues | N1610 | 7.0 |
| Non-static data member initializers | N2756 | 7.0 |
| Variadic templates | N2242 | 7.0 |
| Extending variadic template template parameters | N2555 | 7.0 |
| Initializer lists | N2672 | 7.0 |
| Static assertions | N1720 | 7.0 |
| auto-typed variables | N1984 | 7.0 |
| Multi-declarator auto | N1737 | 7.0 |
| Removal of auto as a storage-class specifier | N2546 | 7.0 |
| New function declarator syntax | N2541 | 7.0 |
| Lambda expressions | N2927 | 7.0 |
| Declared type of an expression | N2343 | 7.0 |
| Incomplete return types | N3276 | 7.0 |
| Right angle brackets | N1757 | 7.0 |
| Default template arguments for function templates | DR226 | 7.0 |
| Solving the SFINAE problem for expressions | DR339 | 7.0 |
| Alias templates | N2258 | 7.0 |
| Extern templates | N1987 | 7.0 |
| Null pointer constant | N2431 | 7.0 |
| Strongly-typed enums | N2347 | 7.0 |
| Forward declarations for enums | N2764 | DR1206 | 7.0 |
| Standardized attribute syntax | N2761 | 7.0 |
| Generalized constant expressions | N2235 | 7.0 |
| Alignment support | N2341 | 7.0 |
| Conditionally-support behavior | N1627 | 7.0 |
| Changing undefined behavior into diagnosable errors | N1727 | 7.0 |
| Delegating constructors | N1986 | 7.0 |
| Inheriting constructors | N2540 | 7.0 |
| Explicit conversion operators | N2437 | 7.0 |
| New character types | N2249 | 7.0 |
| Unicode string literals | N2442 | 7.0 |
| Raw string literals | N2442 | 7.0 |
| Universal character names in literals | N2170 | 7.0 |
| User-defined literals | N2765 | 7.0 |
| Standard Layout Types | N2342 | 7.0 |
| Defaulted functions | N2346 | 7.0 |
| Deleted functions | N2346 | 7.0 |
| Extended friend declarations | N1791 | 7.0 |
| Extending sizeof | N2253 | DR850 | 7.0 |
| Inline namespaces | N2535 | 7.0 |
| Unrestricted unions | N2544 | 7.0 |
| Local and unnamed types as template arguments | N2657 | 7.0 |
| Range-based for | N2930 | 7.0 |
| Explicit virtual overrides | N2928 | N3206 | N3272 | 7.0 |
| Minimal support for garbage collection and reachability-based leak detection | N2670 | N/A (see Restrictions) |
| Allowing move constructors to throw [noexcept] | N3050 | 7.0 |
| Defining move special member functions | N3053 | 7.0 |
| **Concurrency** | | |
| Sequence points | N2239 | |
| Atomic operations | N2427 | |
| Strong Compare and Exchange | N2748 | |
| Bidirectional Fences | N2752 | |
| Memory model | N2429 | |
| Data-dependency ordering: atomics and memory model | N2664 | |
| Propagating exceptions | N2179 | |
| Allowing atomics use in signal handlers | N2547 | |
| Thread-local storage | N2659 | |
| Dynamic initialization and destruction with concurrency | N2660 | |
| **C99 Features in C++11** | | |
| `__func__` predefined identifier | N2340 | 7.0 |
| C99 preprocessor | N1653 | 7.0 |
| long long | N1811 | 7.0 |
| Extended integral types | N1988 | |

## 17.2. C++14 Language Features

下表列出了已被 C++14 标准接受的新语言功能。

Table 13. C++14 Language Features

| Language Feature | C++14 Proposal | Available in nvcc (device code) |
|---|---|---|
| Tweak to certain C++ contextual conversions | N3323 | 9.0 |
| Binary literals | N3472 | 9.0 |
| Functions with deduced return type | N3638 | 9.0 |
| Generalized lambda capture (init-capture) | N3648 | 9.0 |
| Generic (polymorphic) lambda expressions | N3649 | 9.0 |
| Variable templates | N3651 | 9.0 |
| Relaxing requirements on constexpr functions | N3652 | 9.0 |
| Member initializers and aggregates | N3653 | 9.0 |
| Clarifying memory allocation | N3664 | |
| Sized deallocation | N3778 | |
| `[[deprecated]]` attribute | N3760 | 9.0 |
| Single-quotation-mark as a digit separator | N3781 | 9.0 |

## 17.3. C++17 Language Features

nvcc 版本 11.0 及更高版本支持所有 C++17 语言功能，但受[此处](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cpp17)描述的限制的约束。

## 17.4. C++20 Language Features

nvcc 版本 12.0 及更高版本支持所有 C++20 语言功能，但受[此处](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cpp20)描述的限制的约束。

## 17.5. Restrictions

### 17.5.1. Host Compiler Extensions

设备代码不支持主机编译器特定的语言扩展。

`_Complex` 类型仅在主机代码中受支持。

当与支持它的主机编译器一起编译时，设备代码中支持 `__int128` 类型。

`__float128` 类型仅在 64 位 x86 Linux 平台上的主机代码中受支持。`__float128` 类型的常量表达式可以由编译器以较低精度的浮点表示形式处理。

### 17.5.2. Preprocessor Symbols

#### 17.5.2.1. `__CUDA_ARCH__`

1. 以下实体的类型签名不应取决于是否定义了 `__CUDA_ARCH__`，或者取决于 `__CUDA_ARCH__` 的特定值：
   - `__global__` 函数和函数模板
   - `__device__` 和 `__constant__` 变量
   - 纹理和表面

例子：

```cpp
#if !defined(__CUDA_ARCH__)
typedef int mytype;
#else
typedef double mytype;
#endif

__device__ mytype xxx;         // error: xxx's type depends on __CUDA_ARCH__
__global__ void foo(mytype in, // error: foo's type depends on __CUDA_ARCH__
                    mytype *ptr)
{
  *ptr = in;
}
```

2. 如果 `__global__` 函数模板被实例化并从主机启动，则无论是否定义了 `__CUDA_ARCH__` 以及无论 `__CUDA_ARCH__` 的值如何，都必须使用相同的模板参数实例化该函数模板。

例子：

```cpp
__device__ int result;
template <typename T>
__global__ void kern(T in)
{
  result = in;
}

__host__ __device__ void foo(void)
{
#if !defined(__CUDA_ARCH__)
  kern<<<1,1>>>(1);      // error: "kern<int>" instantiation only
                         // when __CUDA_ARCH__ is undefined!
#endif
}

int main(void)
{
  foo();
  cudaDeviceSynchronize();
  return 0;
}
```

3. 在单独编译模式下，是否存在具有外部链接的函数或变量的定义不应取决于是否定义了 `__CUDA_ARCH__` 或 `__CUDA_ARCH__` 的特定值。

例子：

```cpp
#if !defined(__CUDA_ARCH__)
void foo(void) { }                  // error: The definition of foo()
                                    // is only present when __CUDA_ARCH__
                                    // is undefined
#endif
```

4. 在单独的编译中，`__CUDA_ARCH__` 不得在头文件中使用，这样不同的对象可能包含不同的行为。或者，必须保证所有对象都将针对相同的 compute_arch 进行编译。如果在头文件中定义了弱函数或模板函数，并且其行为取决于 `__CUDA_ARCH__`，那么如果为不同的计算架构编译对象，则对象中该函数的实例可能会发生冲突。

例如，如果 a.h 包含：

```cpp
template<typename T>
__device__ T* getptr(void)
{
#if __CUDA_ARCH__ == 700
  return NULL; /* no address */
#else
  __shared__ T arr[256];
  return arr;
#endif
}
```

然后，如果 a.cu 和 b.cu 都包含 a.h 并为同一类型实例化 `getptr`，并且 b.cu 需要一个非 NULL 地址，则编译：

```bash
nvcc –arch=compute_20 –dc a.cu
nvcc –arch=compute_30 –dc b.cu
nvcc –arch=sm_30 a.o b.o
```

在链接时只使用一个版本的 `getptr`，因此行为将取决于选择哪个版本。为避免这种情况，必须为相同的计算架构编译 a.cu 和 b.cu，或者 `__CUDA_ARCH__` 不应在共享头函数中使用。

编译器不保证将为上述不受支持的 `__CUDA_ARCH__` 使用生成诊断。

### 17.5.3. Qualifiers

#### 17.5.3.1. Device Memory Space Specifiers

`__device__`、`__shared__`、`__managed__` 和 `__constant__` 内存空间说明符不允许用于：

- 类、结构和联合数据成员，
- 形式参数，
- 在主机上执行的函数中的非外部变量声明。

`__device__`、`__constant__` 和 `__managed__` 内存空间说明符不允许用于在设备上执行的函数中既不是外部也不是静态的变量声明。

`__device__`、`__constant__`、`__managed__` 或 `__shared__` 变量定义不能具有包含非空构造函数或非空析构函数的类。一个类的构造函数在翻译单元中的某个点被认为是空的，如果它是一个普通的构造函数或者它满足以下所有条件：

- 构造函数已定义。
- 构造函数没有参数，初始化列表是空的，函数体是一个空的复合语句。
- 它的类没有虚函数，没有虚基类，也没有非静态数据成员初始化器。
- 其类的所有基类的默认构造函数都可以认为是空的。
- 对于其类的所有属于类类型（或其数组）的非静态数据成员，默认构造函数可以被认为是空的。

一个类的析构函数在翻译单元中的某个点被认为是空的，如果它是一个普通的析构函数或者它满足以下所有条件：

- 已定义析构函数。
- 析构函数体是一个空的复合语句。
- 它的类没有虚函数，也没有虚基类。
- 其类的所有基类的析构函数都可以认为是空的。
- 对于其类的所有属于类类型（或其数组）的非静态数据成员，析构函数可以被认为是空的。

在整个程序编译模式下编译时（有关此模式的说明，请参见 nvcc 用户手册），`__device__`、`__shared__`、`__managed__` 和 `__constant__` 变量不能使用 extern 关键字定义为外部变量。唯一的例外是动态分配的 `__shared__` 变量，如 [`__shared__`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared) 中所述。

在单独编译模式下编译时（有关此模式的说明，请参阅 nvcc 用户手册），可以使用 extern 关键字将 `__device__`、`__shared__`、`__managed__` 和 `__constant__` 变量定义为外部变量。当 nvlink 找不到外部变量的定义时（除非它是动态分配的 `__shared__` 变量），它会产生错误。

#### 17.5.3.2. `__managed__` Memory Space Specifier

用 `__managed__` 内存空间说明符标记的变量（"managed——托管"变量）具有以下限制：

- 托管变量的地址不是常量表达式。
- 托管变量不应具有 const 限定类型。
- 托管变量不应具有引用类型。
- 当 CUDA 运行时可能不处于有效状态时，不应使用托管变量的地址或值，包括以下情况：
  - 在具有静态或线程本地存储持续时间的对象的静态/动态初始化或销毁中。
  - 在调用 `exit()` 之后执行的代码中（例如，一个标有 gcc 的 `__attribute__((destructor))` 的函数）。
  - 在 CUDA 运行时可能未初始化时执行的代码中（例如，标有 gcc 的 `__attribute__((constructor))` 的函数）。
- 托管变量不能用作 `decltype()` 表达式的未加括号的 id 表达式参数。
- 托管变量具有与为动态分配的托管内存指定的相同的连贯性和一致性行为。
- 当包含托管变量的 CUDA 程序在具有多个 GPU 的执行平台上运行时，变量仅分配一次，而不是每个 GPU。
- 在主机上执行的函数中不允许使用没有外部链接的托管变量声明。
- 在设备上执行的函数中不允许使用没有外部或静态链接的托管变量声明。

以下是托管变量的合法和非法使用示例：

```cpp
__device__ __managed__ int xxx = 10;         // OK

int *ptr = &xxx;                             // error: use of managed variable
                                             // (xxx) in static initialization
struct S1_t {
  int field;
  S1_t(void) : field(xxx) { };
};
struct S2_t {
  ~S2_t(void) { xxx = 10; }
};

S1_t temp1;                                 // error: use of managed variable
                                            // (xxx) in dynamic initialization

S2_t temp2;                                 // error: use of managed variable
                                            // (xxx) in the destructor of
                                            // object with static storage
                                            // duration

__device__ __managed__ const int yyy = 10;  // error: const qualified type

__device__ __managed__ int &zzz = xxx;      // error: reference type

template <int *addr> struct S3_t { };
S3_t<&xxx> temp;                            // error: address of managed
                                            // variable(xxx) not a
                                            // constant expression

__global__ void kern(int *ptr)
{
  assert(ptr == &xxx);                      // OK
  xxx = 20;                                 // OK
}
int main(void)
{
  int *ptr = &xxx;                          // OK
  kern<<<1,1>>>(ptr);
  cudaDeviceSynchronize();
  xxx++;                                    // OK
  decltype(xxx) qqq;                        // error: managed variable(xxx) used
                                            // as unparenthized argument to
                                            // decltype

  decltype((xxx)) zzz = yyy;                // OK
}
```

#### 17.5.3.3. Volatile Qualifier

编译器可以自由优化对全局或共享内存的读取和写入（例如，通过将全局读取缓存到寄存器或 L1 缓存中），只要它尊重内存围栏函数（[Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)）的内存排序语义和内存可见性语义同步函数（[Synchronization Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions)）。

可以使用 `volatile` 关键字禁用这些优化：如果将位于全局或共享内存中的变量声明为 `volatile`，编译器假定它的值可以随时被另一个线程更改或使用，因此对该变量的任何引用都会编译为实际的内存读取或写入指令。

对 volatile 限定对象的读取和写入不是原子的，并且被编译为一个或多个 [.volatile 指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#volatile-operation)，这些指令不保证：

a. 内存操作的顺序，或

b. 硬件执行的内存操作数与 PTX 指令数匹配。

也就是说，CUDA C++ volatile 不适用于：

**线程间同步**：通过 [cuda::atomic_ref](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html)、[cuda::atomic](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html) 或 [Atomic Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions) 使用原子操作。原子内存操作提供线程间同步保证，并提供比易失性操作更好的性能。CUDA C++ volatile 操作不提供任何线程间同步保证，因此不适合线程间同步。以下示例演示如何使用原子操作在两个线程之间传递消息。

```cpp
__global__ void kernel(int* flag, int* data) {
  cuda::atomic_ref<int, cuda::thread_scope_device> f{*flag};
  if (threadIdx.x == 0) {
    // Consumer: blocks until flag is set by producer, then reads data
    while(f.load(cuda::memory_order_acquire) == 0);
    if (*data != 42) __trap(); // Errors if wrong data read
  } else if (threadIdx.x == 1) {
    // Producer: writes data then sets flag
    *data = 42;
    f.store(1, cuda::memory_order_release);
  }
}
```

```cpp
__global__ void kernel(cuda::atomic<int, cuda::thread_scope_device>* flag, int* data) {
  if (threadIdx.x == 0) {
    // Consumer: blocks until flag is set by producer, then reads data
    while(flag->load(cuda::memory_order_acquire) == 0);
    if (*data != 42) __trap(); // Errors if wrong data read
  } else if (threadIdx.x == 1) {
    // Producer: writes data then sets flag
    *data = 42;
    flag->store(1, cuda::memory_order_release);
  }
}
```

```cpp
__global__ void kernel(int* flag, int* data) {
  if (threadIdx.x == 0) {
    // Consumer: blocks until flag is set by producer, then reads data
    while(atomicAdd(flag, 0) == 0); // Load with Relaxed Read-Modify-Write
    __threadfence();                // SequentiallyConsistent fence
    if (*data != 42) __trap();      // Errors if wrong data read
  } else if (threadIdx.x == 1) {
    // Producer: writes data then sets flag
    *data = 42;
    __threadfence();     // SequentiallyConsistent fence
    atomicExch(flag, 1); // Store with Relaxed Read-Modify-Write
  }
}
```

**内存映射 IO**（MMIO）：请改用通过内联 PTX 的 [PTX MMIO 操作](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mmio-operation)。PTX MMIO 操作严格保留执行的内存访问次数。CUDA C++ 易失性操作不会保留执行的内存访问次数，并且可能会以非确定性方式执行比请求的更多或更少的访问，这使得它们对于 MMIO 不正确。以下示例说明如何使用 PTX MMIO 操作从寄存器中读取和写入数据。

```cpp
__global__ void kernel(int* mmio_reg0, int* mmio_reg1) {
  // Write to MMIO register:
  int value = 13;
  asm volatile("st.relaxed.mmio.sys.u32 [%0], %1;" :: "l"(mmio_reg0), "r"(value) : "memory");

  // Read MMIO register:
  asm volatile("ld.relaxed.mmio.sys.u32 %0, [%1];" : "=r"(value) : "l"(mmio_reg1) : "memory");
  
  if (value != 42) __trap(); // Errors if wrong data read
}
```

CPU 中不同代 CPU 应用程序的兼容性很好，已发布的指令集体系结构是确保当这些分布式应用程序成为主流时能够继续在新版 CPU 上运行的常用机制。

这种情况对于 GPU 而言是不同的，因为 NVIDIA 不能保证二进制兼容性，同时不会牺牲 GPU 的改进。相反，正如在图形编程领域中已经习惯的那样，nvcc 依靠两阶段编译模型来确保应用程序与未来 GPU 世代的兼容性。

即虚拟架构和真实架构：虚拟架构确定编译成的代号的功能，真实架构确定编译成的真实代号的功能和性能。

### 17.5.4. Pointers

解引用指向在主机上执行的代码中的全局或共享内存的指针，或在设备上执行的代码中指向主机内存的指针会导致未定义的行为，最常见的是 segmentation fault 和应用程序终止。

获取 `__device__`、`__shared__` 或 `__constant__` 变量的地址获得的地址只能在设备代码中使用。设备内存中描述的通过 `cudaGetSymbolAddress()` 获得的 `__device__` 或 `__constant__` 变量的地址只能在主机代码中使用。

### 17.5.5. Operators

#### 17.5.5.1. Assignment Operator

- `__constant__` 变量只能通过运行时函数（设备内存）从主机代码分配；它们不能从设备代码中分配。
- `__shared__` 变量不能将初始化作为其声明的一部分。（这是因为 `__shared__` 变量的内存分配和初始化由 CUDA 运行时管理，不能在声明时直接赋值。）
- 不允许为内置变量中定义的任何内置变量赋值。（`gridDim` 等）

#### 17.5.5.2. Address Operator

不允许使用内置变量中定义的任何内置变量的地址。

### 17.5.6. Run Time Type Information (RTTI)

主机代码支持以下与 RTTI 相关的功能，但设备代码不支持。

- `typeid` operator
- `std::type_info`
- `dynamic_cast` operator

### 17.5.7. Exception Handling

异常处理仅在主机代码中受支持，但在设备代码中不支持。

`__global__` 函数不支持异常规范。

### 17.5.8. Standard Library

除非另有说明，标准库仅在主机代码中受支持，而不在设备代码中受支持。

### 17.5.9. Namespace Reservations

除非另有说明，否则将任何声明或定义添加到 `cuda::`、`nv::`、`cooperative_groups::` 或嵌套在其中的任何命名空间都是未定义的行为。

```cpp
namespace cuda{
   // Bad: class declaration added to namespace cuda
   struct foo{};

   // Bad: function definition added to namespace cuda
   cudaStream_t make_stream(){
     cudaStream_t s;
     cudaStreamCreate(&s);
     return s;
   }
} // namespace cuda

namespace cuda{
   namespace utils{
      // Bad: function definition added to namespace nested within cuda
      cudaStream_t make_stream(){
         cudaStream_t s;
         cudaStreamCreate(&s);
         return s;
      }
   } // namespace utils
} // namespace cuda

namespace utils{
   namespace cuda{
     // Okay: namespace cuda may be used nested within a non-reserved namespace
     cudaStream_t make_stream(){
         cudaStream_t s;
         cudaStreamCreate(&s);
         return s;
      }
   } // namespace cuda
} // namespace utils

// Bad: Equivalent to adding symbols to namespace cuda at global scope
using namespace utils;
```

### 17.5.10. Functions

#### 17.5.10.1. External Linkage

仅当函数在与设备代码相同的编译单元中定义时，才允许在某些设备代码中调用使用 `extern` 限定符声明的函数，即单个文件或通过可重定位设备代码和 nvlink 链接在一起的多个文件。

#### 17.5.9.2. Implicitly-declared and explicitly-defaulted functions

设 F 表示一个函数，该函数要么是隐式声明的，要么是显式默认的。F 的执行空间说明符（`__host__`、`__device__`）是调用它的所有函数的执行空间说明符的并集（请注意，对于此分析，`__global__` 调用者将被视为 `__device__` 调用者）。例如：

```cpp
class Base {
  int x;
public:
  __host__ __device__ Base(void) : x(10) {}
};

class Derived : public Base {
  int y;
};

class Other: public Base {
  int z;
};

__device__ void foo(void)
{
  Derived D1;
  Other D2;
}

__host__ void bar(void)
{
  Other D3;
}
```

这里，隐式声明的构造函数 `Derived::Derived` 将被视为 `__device__` 函数，因为它仅从 `__device__` 函数 `foo` 调用。隐式声明的构造函数 `Other::Other` 将被视为 `__host__ __device__` 函数，因为它是从 `__device__` 函数 `foo` 和 `__host__` 函数 `bar` 调用的。

此外，如果 F 是虚拟析构函数，则如果 D 未隐式定义或显式默认在其第一次声明以外的声明上，则被 F 覆盖的每个虚拟析构函数 D 的执行空间将添加到 F 的执行空间集中。

例如：

```cpp
struct Base1 { virtual __host__ __device__ ~Base1() { } };
struct Derived1 : Base1 { }; // implicitly-declared virtual destructor
                             // ~Derived1 has __host__ __device__
                             // execution space specifiers

struct Base2 { virtual __device__ ~Base2(); };
__device__ Base2::~Base2() = default;
struct Derived2 : Base2 { }; // implicitly-declared virtual destructor
                             // ~Derived2 has __device__ execution
                             // space specifiers
```

#### 17.5.10.3. Function Parameters

`__global__` 函数参数通过常量内存传递给设备，从 Volta 开始限制为 32,764 字节，在较旧的架构上限制为 4 KB。

`__global__` 函数不能有可变数量的参数。

`__global__` 函数参数不能通过引用传递。

在单独编译模式下，如果 `__device__` 或 `__global__` 函数在特定翻译单元中被 ODR（**One Definition Rule** 单一定义规则）使用，则该函数的参数和返回类型在该翻译单元中必须是完整的。

```cpp
//first.cu:
struct S;
__device__ void foo(S); // error: type 'S' is incomplete
// 编译器无法确定 S 的大小和布局，因此在 first.cu 中无法使用 S 作为函数参数。
__device__ auto *ptr = foo;

int main() { }

//second.cu:
struct S { int x; };
__device__ void foo(S) { }

//compiler invocation
$ nvcc -std=c++14 -rdc=true first.cu second.cu -o first
nvlink error   : Prototype doesn't match for '_Z3foo1S' in '/tmp/tmpxft_00005c8c_00000000-18_second.o', 
                    first defined in '/tmp/tmpxft_00005c8c_00000000-18_second.o'
nvlink fatal   : merge_elf failed
// nvcc 在链接时发现 foo(S) 的符号在 first.cu 和 second.cu 中的定义不一致，导致 nvlink 报错
// 解决方法：
// 1、将结构体定义放在头文件中
// 2、使用 extern 关键字
// 3、将代码合并到一个文件中
```

**17.5.10.3.1. global Function Argument Processing**

当从设备代码启动 `__global__` 函数时，每个参数都必须是可简单复制和可简单销毁的。

当从主机代码启动 `__global__` 函数时，每个参数类型都可以是不可复制或不可销毁的，但对此类类型的处理不遵循标准 C++ 模型，如下所述。用户代码必须确保此工作流程不会影响程序的正确性。工作流在两个方面与标准 C++ 不同：

1. Memcpy instead of copy constructor invocation;

从主机代码降低 `__global__` 函数启动时，编译器会生成存根函数，这些函数按值复制参数一次或多次，然后最终使用 `memcpy` 将参数复制到设备上的 `__global__` 函数的参数内存中。即使参数是不可复制的，也会发生这种情况，因此可能会破坏复制构造函数具有副作用的程序。

```cpp
#include <cassert>
struct S {
 int x;
 int *ptr;
 __host__ __device__ S() { }
 __host__ __device__ S(const S &) { ptr = &x; }
};

__global__ void foo(S in) {
 // this assert may fail, because the compiler
 // generated code will memcpy the contents of "in"
 // from host to kernel parameter memory, so the
 // "in.ptr" is not initialized to "&in.x" because
 // the copy constructor is skipped.
 assert(in.ptr == &in.x);
}

int main() {
 S tmp;
 foo<<<1,1>>>(tmp);
 cudaDeviceSynchronize();
}
```

```cpp
#include <cassert>

__managed__ int counter;
struct S1 {
S1() { }
S1(const S1 &) { ++counter; }
};

__global__ void foo(S1) {

/* this assertion may fail, because
   the compiler generates stub
   functions on the host for a kernel
   launch, and they may copy the
   argument by value more than once.
*/
assert(counter == 1);
}

int main() {
S1 V;
foo<<<1,1>>>(V);
cudaDeviceSynchronize();
}
```

2. Destructor may be invoked before the global function has finished;

内核启动与主机执行是异步的。因此，如果 `__global__` 函数参数具有非平凡的析构函数，则析构函数甚至可以在 `__global__` 函数完成执行之前在宿主代码中执行。这可能会破坏析构函数具有副作用的程序。示例：

```cpp
struct S {
 int *ptr;
 S() : ptr(nullptr) { }
 S(const S &) { cudaMallocManaged(&ptr, sizeof(int)); }
 ~S() { cudaFree(ptr); }
};

__global__ void foo(S in) {

  //error: This store may write to memory that has already been
  //       freed (see below).
  *(in.ptr) = 4;

}

int main() {
 S V;

 /* The object 'V' is first copied by value to a compiler-generated
  * stub function that does the kernel launch, and the stub function
  * bitwise copies the contents of the argument to kernel parameter
  * memory.
  * However, GPU kernel execution is asynchronous with host
  * execution.
  * As a result, S::~S() will execute when the stub function returns, 
  * releasing allocated memory, even though the kernel may not have finished execution.
  */
 foo<<<1,1>>>(V);
 cudaDeviceSynchronize();
}
```

#### 17.5.10.3.2. Toolkit and Driver Compatibility

开发人员必须使用 12.1 Toolkit 和 r530 驱动程序或更高版本来编译、启动和调试接受大于 4KB 参数的内核。如果在较旧的驱动程序上启动此类内核，CUDA 将发出错误 `CUDA_ERROR_NOT_SUPPORTED`。

**17.5.10.3.3. Link Compatibility across Toolkit Revisions**

在链接设备对象时，如果至少一个设备对象包含参数大于 4KB 的内核，则开发人员必须使用 12.1 工具包或更高版本重新编译各自设备源中的所有对象，然后才能将它们链接在一起。否则将导致链接器错误。

#### 17.5.10.4. Static Variables within Function

在函数 F 的直接或嵌套块范围内，静态变量 V 的声明中允许使用可变内存空间说明符，其中：

- F 是一个 `__global__` 或 `__device__`-only 函数。
- F 是一个 `__host__ __device__` 函数，`__CUDA_ARCH__` 被使用。

如果 V 的声明中没有显式的内存空间说明符，则在设备编译期间假定隐式 `__device__` 说明符。

V 具有与在命名空间范围内声明的具有相同内存空间说明符的变量相同的初始化限制，例如 `__device__` 变量不能有"非空"构造函数（请参阅设备内存空间说明符）。

函数范围静态变量的合法和非法使用示例如下所示。

```cpp
struct S1_t {
  int x;
};

struct S2_t {
  int x;
  __device__ S2_t(void) { x = 10; }
};

struct S3_t {
  int x;
  __device__ S3_t(int p) : x(p) { }
};

__device__ void f1() {
  static int i1;              // OK, implicit __device__ memory space specifier
  static int i2 = 11;         // OK, implicit __device__ memory space specifier
  static __managed__ int m1;  // OK
  static __device__ int d1;   // OK
  static __constant__ int c1; // OK

  static S1_t i3;             // OK, implicit __device__ memory space specifier
  static S1_t i4 = {22};      // OK, implicit __device__ memory space specifier

  static __shared__ int i5;   // OK

  int x = 33;
  static int i6 = x;          // error: dynamic initialization is not allowed
  static S1_t i7 = {x};       // error: dynamic initialization is not allowed

  static S2_t i8;             // error: dynamic initialization is not allowed
  static S3_t i9(44);         // error: dynamic initialization is not allowed
}

__host__ __device__ void f2() {
  static int i1;              // OK, implicit __device__ memory space specifier
                              // during device compilation.
#ifdef __CUDA_ARCH__
  static __device__ int d1;   // OK, declaration is only visible during device
                              // compilation  (__CUDA_ARCH__ is defined)
#else
  static int d0;              // OK, declaration is only visible during host
                              // compilation (__CUDA_ARCH__ is not defined)
#endif

  static __device__ int d2;   // error: __device__ variable inside
                              // a host function during host compilation
                              // i.e. when __CUDA_ARCH__ is not defined

  static __shared__ int i2;  // error: __shared__ variable inside
                             // a host function during host compilation
                             // i.e. when __CUDA_ARCH__ is not defined
}
```

#### 17.5.10.5. Function Pointers

在主机代码中获取的 `__global__` 函数的地址不能在设备代码中使用（例如，启动内核）。同样，在设备代码中获取的 `__global__` 函数的地址不能在主机代码中使用。

不允许在主机代码中获取 `__device__` 函数的地址。

#### 17.5.10.6. Function Recursion

`__global__` 函数不支持递归。

#### 17.5.10.7. Friend Functions

`__global__` 函数或函数模板不能在友元声明中定义。

```cpp
struct S1_t {
  friend __global__
  void foo1(void);  // OK: not a definition
  template<typename T>
  friend __global__
  void foo2(void); // OK: not a definition

  friend __global__
  void foo3(void) { } // error: definition in friend declaration

  template<typename T>
  friend __global__
  void foo4(void) { } // error: definition in friend declaration
};
```

#### 17.5.10.8. Operator Function

运算符函数不能是 `__global__` 函数。

#### 17.5.10.9. Allocation and Deallocation Functions

用户定义的运算符 `new`、`operator new[]`、`operator delete` 或 `operator delete[]` 不能用于替换编译器提供的相应 `__host__` 或 `__device__` 内置函数。

### 17.5.11. Classes

#### 17.5.11.1. 数据成员

不支持静态数据成员，除了那些也是 const 限定的变量除外（请参阅 [Const 限定变量](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#const-variables)）。

#### 17.5.11.2. 函数成员

静态成员函数不能是 `__global__` 函数。

#### 17.5.11.3. 虚函数

当派生类中的函数覆盖基类中的虚函数时，被覆盖函数和覆盖函数上的执行空间说明符（即 `__host__`、`__device__`）必须匹配。

不允许将具有虚函数的类的对象作为参数传递给 `__global__` 函数。

如果在主机代码中创建对象，则在设备代码中调用该对象的虚函数具有未定义的行为。

如果在设备代码中创建对象，则在主机代码中调用该对象的虚函数具有未定义的行为。

使有关使用 Microsoft 主机编译器时的其他约束，请参阅 [Windows 特定](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#windows-specific)。

```cpp
struct S1 { virtual __host__ __device__ void foo() { } };

__managed__ S1 *ptr1, *ptr2;

__managed__ __align__(16) char buf1[128];
__global__ void kern() {
 ptr1->foo();     // error: virtual function call on a object
                  //        created in host code.
 ptr2 = new(buf1) S1();
}

int main(void) {
  void *buf;
  cudaMallocManaged(&buf, sizeof(S1), cudaMemAttachGlobal);
 ptr1 = new (buf) S1();
 kern<<<1,1>>>();
  cudaDeviceSynchronize();
 ptr2->foo();  // error: virtual function call on an object
               //        created in device code.
}
```

#### 17.5.11.4. Virtual Base Classes

不允许将派生自虚拟基类的类的对象作为参数传递给 `__global__` 函数。

有关使用 Microsoft 主机编译器时的其他约束，请参阅 [Windows 特定](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#windows-specific)。

#### 17.5.11.5. Anonymous Unions

命名空间范围匿名联合的成员变量不能在 `__global__` 或 `__device__` 函数中引用。

#### 17.5.11.6. 特定于 Windows 的

CUDA 编译器遵循 IA64 ABI 进行类布局，而 Microsoft 主机编译器则不遵循。令 T 表示指向成员类型的指针，或满足以下任一条件的类类型：

- T has virtual functions.
- T has a virtual base class.
- T has multiple inheritance with more than one direct or indirect empty base class.
- All direct and indirect base classes B of T are empty and the type of the first field F of T uses B in its definition, such that B is laid out at offset 0 in the definition of F.

让 C 表示 T 或以 T 作为字段类型或基类类型的类类型。CUDA 编译器计算类布局和大小的方式可能不同于 C 类型的 Microsoft 主机编译器。只要类型 C 专门用于主机或设备代码，程序就应该可以正常工作。

在主机和设备代码之间传递 C 类型的对象具有未定义的行为，例如，作为 `__global__` 函数的参数或通过 `cudaMemcpy*()` 调用。

如果在主机代码中创建对象，则访问 C 类型的对象或设备代码中的任何子对象，或调用设备代码中的成员函数具有未定义的行为。

如果对象是在设备代码中创建的，则访问 C 类型的对象或主机代码中的任何子对象，或调用主机代码中的成员函数具有未定义的行为。

### 17.5.12. Templates

如果满足以下任一条件，则不能在 `__global__` 函数模板实例化或 `__device__`/`__constant__` 变量实例化的类型、非类型或模板模板参数中使用类型或模板：

- 类型或模板在 `__host__` 或 `__host__ __device__` 中定义。
- 类型或模板是具有私有或受保护访问的类成员，其父类未在 `__device__` 或 `__global__` 函数中定义。
- 该类型未命名。
- 该类型由上述任何类型复合而成。

```cpp
template <typename T>
__global__ void myKernel(void) { }

class myClass {
private:
    struct inner_t { };
public:
    static void launch(void)
    {
       // error: inner_t is used in template argument
       // but it is private
       myKernel<inner_t><<<1,1>>>();
    }
};

// C++14 only
template <typename T> __device__ T d1;

template <typename T1, typename T2> __device__ T1 d2;

void fn() {
  struct S1_t { };
  // error (C++14 only): S1_t is local to the function fn
  d1<S1_t> = {};

  auto lam1 = [] { };
  // error (C++14 only): a closure type cannot be used for
  // instantiating a variable template
  d2<int, decltype(lam1)> = 10;
}
```

### 17.5.13. Trigraphs and Digraphs

任何平台都不支持三元组。Windows 不支持双字符组。

### 17.5.14. Const-qualified variables

让"V"表示名称空间范围变量或具有 const 限定类型且没有执行空间注释的类静态成员变量（例如，`__device__`、`__constant__`、`__shared__`）。V 被认为是主机代码变量。

V 的值可以直接在设备代码中使用，如果

- V 在使用点之前已经用常量表达式初始化，
- V 的类型不是 volatile 限定的，并且
- 它具有以下类型之一：
  - 内置浮点类型，除非将 Microsoft 编译器用作主机编译器，
  - 内置整型。

设备源代码不能包含对 V 的引用或获取 V 的地址。

```cpp
const int xxx = 10;
struct S1_t {  static const int yyy = 20; };

extern const int zzz;
const float www = 5.0;
__device__ void foo(void) {
  int local1[xxx];          // OK
  int local2[S1_t::yyy];    // OK

  int val1 = xxx;           // OK

  int val2 = S1_t::yyy;     // OK

  int val3 = zzz;           // error: zzz not initialized with constant
                            // expression at the point of use.

  const int &val3 = xxx;    // error: reference to host variable
  const int *val4 = &xxx;   // error: address of host variable
  const float val5 = www;   // OK except when the Microsoft compiler is used as
                            // the host compiler.
}
const int zzz = 20;
```

### 17.5.15. Long Double

设备代码不支持使用 `long double` 类型。

### 17.5.16. Deprecation Annotation

nvcc 支持在使用 gcc、clang、xlC、icc 或 pgcc 主机编译器时使用 `deprecated` 属性，以及在使用 `cl.exe` 主机编译器时使用 `deprecated` declspec。当启用 C++14 时，它还支持 `[[deprecated]]` 标准属性。当定义 `__CUDA_ARCH__` 时（即在设备编译阶段），CUDA 前端编译器将为从 `__device__`、`__global__` 或 `__host__ __device__` 函数的主体内对已弃用实体的引用生成弃用诊断。对不推荐使用的实体的其他引用将由主机编译器处理，例如，来自 `__host__` 函数中的引用。

CUDA 前端编译器不支持各种主机编译器支持的 `#pragma gcc 诊断` 或 `#pragma 警告` 机制。因此，CUDA 前端编译器生成的弃用诊断不受这些 pragma 的影响，但主机编译器生成的诊断会受到影响。要抑制设备代码的警告，用户可以使用 NVIDIA 特定的 pragma `#pragma nv_diag_suppress`。nvcc 标志 `-Wno-deprecated-declarations` 可用于禁止所有弃用警告，标志 `-Werror=deprecated-declarations` 可用于将弃用警告转换为错误。

### 17.5.17. Noreturn Annotation

nvcc 支持在使用 gcc、clang、xlC、icc 或 pgcc 主机编译器时使用 `noreturn` 属性，并在使用 `cl.exe` 主机编译器时使用 `noreturn` declspec。当启用 C++11 时，它还支持 `[[noreturn]]` 标准属性。

attribute/declspec 可用于主机和设备代码。

### 17.5.18. `[[likely]]` / `[[unlikely]]` Standard Attributes

所有支持 C++ 标准属性语法的配置都接受这些属性。这些属性可用于向设备编译器优化器提示与不包含该语句的任何替代路径相比，该语句是否更有可能被执行。

```cpp
__device__ int foo(int x) {

 if (i < 10) [[likely]] { // the 'if' block will likely be entered
  return 4;
 }
 if (i < 20) [[unlikely]] { // the 'if' block will not likely be entered
  return 1;
 }
 return 0;
}
```

如果在 `__CUDA_ARCH__` 未定义时在主机代码中使用这些属性，则它们将出现在主机编译器解析的代码中，如果不支持这些属性，则可能会生成警告。例如，clang11 主机编译器将生成"unknown attribute"警告。

### 17.5.19. const and pure GNU Attributes

当使用也支持这些属性的语言和主机编译器时，主机和设备功能都支持这些属性，例如使用 g++ 主机编译器。

对于使用 `pure` 属性注释的设备函数，设备代码优化器假定该函数不会更改调用者函数（例如内存）可见的任何可变状态。

- 函数不能修改全局内存、共享内存或任何外部状态。
- 函数可以调用其他 `pure` 或 `const` 的函数。

对于使用 `const` 属性注释的设备函数，设备代码优化器假定该函数不会访问或更改调用者函数可见的任何可变状态（例如内存）。

- 函数不能访问全局内存、共享内存或任何外部状态。
- 函数不能调用非 `const` 的函数。

```cpp
__attribute__((const)) __device__ int get(int in);

__device__ int doit(int in) {
int sum = 0;

//because 'get' is marked with 'const' attribute
//device code optimizer can recognize that the
//second call to get() can be commoned out.
sum = get(in);
sum += get(in);

return sum;
}
```

### 17.5.20. `__nv_pure__` Attribute

主机和设备功能都支持 `__nv_pure__` 属性。对于主机函数，当使用支持 `pure` GNU 属性的语言时，`__nv_pure__` 属性将转换为 `pure` GNU 属性。同样，当使用 MSVC 作为主机编译器时，该属性将转换为 MSVC `noalias` 属性。

当设备函数使用 `__nv_pure__` 属性进行注释时，设备代码优化器会假定该函数不会更改调用方函数可见的任何可变状态（例如内存）。

### 17.5.21. Intel Host Compiler Specific

CUDA 前端编译器解析器无法识别英特尔编译器（例如 icc）支持的某些内在函数。因此，当使用 Intel 编译器作为主机编译器时，nvcc 将在预处理期间启用宏 `__INTEL_COMPILER_USE_INTRINSIC_PROTOTYPES`。此宏允许在相关头文件中显式声明英特尔编译器内部函数，从而允许 nvcc 支持在主机代码中使用[此类函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_21)。

### 17.5.22. C++11 Features

nvcc 也支持主机编译器默认启用的 C++11 功能，但须遵守本文档中描述的限制。此外，使用 `-std=c++11` 标志调用 nvcc 会打开所有 C++11 功能，还会使用相应的 C++11 [选项](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_21)调用主机预处理器、编译器和链接器。

#### 17.5.22.1. Lambda Expressions

与 lambda 表达式关联的闭包类的所有[成员函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_22)的执行空间说明符由编译器派生如下。如 C++11 标准中所述，编译器在包含 lambda 表达式的最小块范围、类范围或命名空间范围内创建闭包类型。计算封闭闭包类型的最内层函数作用域，并将相应函数的执行空间说明符分配给闭包类成员函数。如果没有封闭函数范围，则执行空间说明符为 `__host__`。

lambda 表达式和计算的执行空间说明符的示例如下所示（在注释中）。

```cpp
auto globalVar = [] { return 0; }; // __host__

void f1(void) {
  auto l1 = [] { return 1; };      // __host__
}

__device__ void f2(void) {
  auto l2 = [] { return 2; };      // __device__
}

__host__ __device__ void f3(void) {
  auto l3 = [] { return 3; };      // __host__ __device__
}

__device__ void f4(int (*fp)() = [] { return 4; } /* __host__ */) {
}

__global__ void f5(void) {
  auto l5 = [] { return 5; };      // __device__
}

__device__ void f6(void) {
  struct S1_t {
    static void helper(int (*fp)() = [] {return 6; } /* __device__ */) {
    }
  };
}
```

lambda 表达式的闭包类型不能用于 `__global__` 函数模板实例化的类型或非类型参数，除非 lambda 在 `__device__` 或 `__global__` 函数中定义。

```cpp
template <typename T>
__global__ void foo(T in) { };

template <typename T>
struct S1_t { };

void bar(void) {
  auto temp1 = [] { };

  foo<<<1,1>>>(temp1);                    // error: lambda closure type used in
                                          // template type argument
  foo<<<1,1>>>( S1_t<decltype(temp1)>()); // error: lambda closure type used in
                                          // template type argument
}
```

```cpp
template <typename T>
__global__ void foo(T in) { };

template <typename T>
struct S1_t { };

void bar(void) {
  auto temp1 = [] __device__ (void) { };  // __device__ lambda 需要启用 --extended-lambda 编译选项

  foo<<<1,1>>>(temp1);                    // error: lambda closure type used in
                                          // template type argument
  foo<<<1,1>>>( S1_t<decltype(temp1)>()); // error: lambda closure type used in
                                          // template type argument
}
```

#### 17.5.22.2. `std::initializer_list`

默认情况下，CUDA 编译器将隐式认为 `std::initializer_list` 的成员函数具有 `__host__ __device__` 执行空间说明符，因此可以直接从设备代码调用它们。nvcc 标志 `--no-host-device-initializer-list` 将禁用此行为；然后，`std::initializer_list` 的成员函数将被视为 `__host__` 函数，并且不能直接从设备代码调用。

```cpp
#include <initializer_list>

__device__ int foo(std::initializer_list<int> in);

__device__ void bar(void)
  {
    foo({4,5,6});   // (a) initializer list containing only
                    // constant expressions.

    int i = 4;
    foo({i,5,6});   // (b) initializer list with at least one
                    // non-constant element.
                    // This form may have better performance than (a).
  }
```

#### 17.5.22.3. Rvalue references

默认情况下，CUDA 编译器将隐式认为 `std::move` 和 `std::forward` 函数模板具有 `__host__ __device__` 执行空间说明符，因此可以直接从设备代码调用它们。nvcc 标志 `--no-host-device-move-forward` 将禁用此行为；`std::move` 和 `std::forward` 将被视为 `__host__` 函数，不能直接从设备代码调用。

#### 17.5.22.4. Constexpr functions and function templates

默认情况下，不能从执行空间不兼容的函数中调用 `constexpr` [函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_23)。实验性 nvcc 标志 `--expt-relaxed-constexpr` 消除了[此限制](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_24)。当指定此标志时，主机代码可以调用 `__device__ constexpr` 函数和设备代码可以调用 `__host__ constexpr` 函数。当指定了 `--expt-relaxed-constexpr` 时，nvcc 将定义宏 `__CUDACC_RELAXED_CONSTEXPR__`。请注意，即使相应的模板用关键字 `constexpr` 标记（C++11 标准节 [dcl.constexpr.p6]），函数模板实例化也可能不是 `constexpr` 函数。

```cpp
constexpr int square(int x) {
    return x * x;
}

__device__ void foo() {
    constexpr int size = square(5);  // 编译时计算，size 是常量 25
    int array[size];                 // 使用编译时常量定义数组大小
}

template <typename T>
constexpr T add(T a, T b) {
    return a + b;
}

__device__ void bar() {
    constexpr int result = add(3, 4);  // 编译时计算，result 是常量 7
}
```

#### 17.5.22.5. Constexpr variables

让"V"表示命名空间范围变量或已标记为 `constexpr` 且没有执行空间注释的类静态成员变量（例如，`__device__`、`__constant__`、`__shared__`）。V 被认为是主机代码变量。

如果 V 是除 `long double` 以外的[标量类型](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_25)并且该类型不是 volatile 限定的，则 V 的值可以直接在设备代码中使用。此外，如果 V 是非标量类型，则 V 的标量元素可以在 `constexpr __device__` 或 `__host__ __device__` 函数中使用，如果对函数的调用是[常量表达式](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_26)。设备源代码不能包含对 V 的引用或取 V 的地址。

```cpp
constexpr int xxx = 10;
constexpr int yyy = xxx + 4;
struct S1_t { static constexpr int qqq = 100; };

constexpr int host_arr[] = { 1, 2, 3};
constexpr __device__ int get(int idx) { return host_arr[idx]; }

__device__ int foo(int idx) {
  int v1 = xxx + yyy + S1_t::qqq;  // OK
  const int &v2 = xxx;             // error: reference to host constexpr
                                   // variable
  const int *v3 = &xxx;            // error: address of host constexpr
                                   // variable
  const int &v4 = S1_t::qqq;       // error: reference to host constexpr
                                   // variable
  const int *v5 = &S1_t::qqq;      // error: address of host constexpr
                                   // variable

  v1 += get(2);                    // OK: 'get(2)' is a constant
                                   // expression.
  v1 += get(idx);                  // error: 'get(idx)' is not a constant
                                   // expression
  v1 += host_arr[2];               // error: 'host_arr' does not have
                                   // scalar type.
  return v1;
}
```

#### 17.5.22.6. Inline namespaces

```cpp
namespace Library {
    inline namespace v1 {
        void foo() { /* 旧版本实现 */ }
    }

    namespace v2 {
        void foo() { /* 新版本实现 */ }
    }
}

int main() {
    Library::foo(); // 默认使用 v1 版本的 foo
    Library::v2::foo(); // 显式使用 v2 版本的 foo
    return 0;
}
```

对于输入的 CUDA 翻译单元，CUDA 编译器可以调用主机编译器来编译翻译单元内的主机代码。在传递给主机编译器的代码中，如果输入的 CUDA 翻译单元包含以下任何实体的定义，CUDA 编译器将注入额外的编译器生成的代码：

- `__global__` 函数或函数模板实例化
- `__device__`、`__constant__`
- 具有 Surface 或 Texture 类型的变量

编译器生成的代码包含对已定义实体的引用。如果实体是在内联命名空间中定义的，而另一个具有相同名称和类型签名的实体在封闭命名空间中定义，则主机编译器可能会认为此引用不明确，主机编译将失败。可以通过对内联命名空间中定义的此类实体使用唯一名称来避免此限制。

```cpp
__device__ int Gvar;
inline namespace N1 {
  __device__ int Gvar;
}

// <-- CUDA compiler inserts a reference to "Gvar" at this point in the
// translation unit. This reference will be considered ambiguous by the
// host compiler and compilation will fail.
```

```cpp
inline namespace N1 {
  namespace N2 {
    __device__ int Gvar;
  }
}

namespace N2 {
  __device__ int Gvar;
}

// <-- CUDA compiler inserts reference to "::N2::Gvar" at this point in
// the translation unit. This reference will be considered ambiguous by
// the host compiler and compilation will fail.
```

**17.5.22.6.1. Inline unnamed namespaces**

以下实体不能在内联 unnamed 命名空间的命名空间范围内声明：

- `__managed__`、`__device__`、`__shared__` 和 `__constant__` 变量
- `__global__` 函数和函数模板
- 具有表面或纹理类型的变量

```cpp
inline namespace {
  namespace N2 {
    template <typename T>
    __global__ void foo(void);            // error

    __global__ void bar(void) { }         // error

    template <>
    __global__ void foo<int>(void) { }    // error

    __device__ int x1b;                   // error
    __constant__ int x2b;                 // error
    __shared__ int x3b;                   // error

    texture<int> q2;                      // error
    surface<int> s2;                      // error
  }
};
```

#### 17.5.22.7. `thread_local`

设备代码中不允许使用 `thread_local` 存储说明符。

```cpp
#include <iostream>
#include <thread>

thread_local int tls_var = 0; // 每个线程都有独立的 tls_var

void thread_func(int id) {
    tls_var = id; // 修改当前线程的 tls_var
    std::cout << "Thread " << id << ", tls_var = " << tls_var << std::endl;
}

int main() {
    std::thread t1(thread_func, 1);
    std::thread t2(thread_func, 2);

    t1.join();
    t2.join();

    std::cout << "Main thread, tls_var = " << tls_var << std::endl;
    return 0;
}
```

#### 17.5.22.8. global functions and function templates

如果在 `__global__` 函数模板实例化的模板参数中使用与 lambda 表达式关联的闭包类型，则 lambda 表达式必须在 `__device__` 或 `__global__` 函数的直接或嵌套块范围内定义，或者必须是扩展 lambda。

```cpp
template <typename T>
__global__ void kernel(T in) { }

__device__ void foo_device(void)
{
  // All kernel instantiations in this function
  // are valid, since the lambdas are defined inside
  // a __device__ function.

  kernel<<<1,1>>>( [] __device__ { } );
  kernel<<<1,1>>>( [] __host__ __device__ { } );
  kernel<<<1,1>>>( []  { } );
}

auto lam1 = [] { };

auto lam2 = [] __host__ __device__ { };

void foo_host(void)
{
   // OK: instantiated with closure type of an extended __device__ lambda
   kernel<<<1,1>>>( [] __device__ { } );

   // OK: instantiated with closure type of an extended __host__ __device__
   // lambda
   kernel<<<1,1>>>( [] __host__ __device__ { } );

   // error: unsupported: instantiated with closure type of a lambda
   // that is not an extended lambda
   kernel<<<1,1>>>( []  { } );

   // error: unsupported: instantiated with closure type of a lambda
   // that is not an extended lambda
   kernel<<<1,1>>>( lam1);

   // error: unsupported: instantiated with closure type of a lambda
   // that is not an extended lambda
   kernel<<<1,1>>>( lam2);
}
```

`__global__` 函数或函数模板不能声明为 `constexpr`。

`__global__` 函数或函数模板不能有 `std::initializer_list` 或 `va_list` 类型的参数。

`__global__` 函数不能有右值引用类型的参数。

可变参数 `__global__` 函数模板具有以下限制：

- 只允许一个包参数。
- pack 参数必须在模板参数列表中最后列出。

```cpp
// ok
template <template <typename...> class Wrapper, typename... Pack>
__global__ void foo1(Wrapper<Pack...>);

// error: pack parameter is not last in parameter list
template <typename... Pack, template <typename...> class Wrapper>
__global__ void foo2(Wrapper<Pack...>);

// error: multiple parameter packs
template <typename... Pack1, int...Pack2, template<typename...> class Wrapper1,
          template<int...> class Wrapper2>
__global__ void foo3(Wrapper1<Pack1...>, Wrapper2<Pack2...>);
```

#### 17.5.22.9. managed and shared variables

`__managed__` 和 `__shared__` 变量不能用关键字 `constexpr` 标记。

#### 17.5.22.10. Defaulted functions

CUDA 编译器会忽略在第一个声明中显式默认的函数上的执行空间说明符。相反，CUDA 编译器将推断执行空间说明符，如[隐式声明和显式默认函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compiler-generated-functions)中所述。

如果函数是显式默认的，则不会忽略执行空间说明符，但不会在其第一次声明时忽略。

```cpp
struct S1 {
  // warning: __host__ annotation is ignored on a function that
  //          is explicitly-defaulted on its first declaration
  __host__ S1() = default;
};

__device__ void foo1() {
  //note: __device__ execution space is derived for S1::S1
  //       based on implicit call from within __device__ function
  //       foo1
  S1 s1;
}

struct S2 {
  __host__ S2();
};

//note: S2::S2 is not defaulted on its first declaration, and
//      its execution space is fixed to __host__  based on its
//      first declaration.
S2::S2() = default;

__device__ void foo2() {
   // error: call from __device__ function 'foo2' to
   //        __host__ function 'S2::S2'
   S2 s2;
}
```

### 17.5.23. C++14 Features

nvcc 也支持主机编译器默认启用的 C++14 功能。传递 nvcc `-std=c++14` 标志打开所有 C++14 功能，并使用相应的 C++14 [选项](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_27)调用主机预处理器、编译器和链接器。本节描述了对受支持的 C++14 的限制特点。

#### 17.5.23.1. Functions with deduced return type

`__global__` 函数不能有推导的返回类型。

如果 `__device__` 函数推导出返回类型，CUDA 前端编译器将在调用主机编译器之前将函数声明更改为具有 void 返回类型。这可能会导致在主机代码中自省 `__device__` 函数的推导返回类型时出现问题。因此，CUDA 编译器将发出编译时错误，用于在设备函数体之外引用此类推导的返回类型，除非在 `__CUDA_ARCH__` 未定义时引用不存在。

```cpp
__device__ auto fn1(int x) {
  return x;
}

__device__ decltype(auto) fn2(int x) {
  return x;
}

__device__ void device_fn1() {
  // OK
  int (*p1)(int) = fn1;
}

// error: referenced outside device function bodies
decltype(fn1(10)) g1;

void host_fn1() {
  // error: referenced outside device function bodies
  int (*p1)(int) = fn1;

  struct S_local_t {
    // error: referenced outside device function bodies
    decltype(fn2(10)) m1;

    S_local_t() : m1(10) { }
  };
}

// error: referenced outside device function bodies
template <typename T = decltype(fn2)>
void host_fn2() { }

template<typename T> struct S1_t { };

// error: referenced outside device function bodies
struct S1_derived_t : S1_t<decltype(fn1)> { };
```

#### 17.5.23.2. Variable templates

使用 Microsoft 主机编译器时，`__device__`/`__constant__` 变量模板不能具有 const 限定类型。

```cpp
// error: a __device__ variable template cannot
// have a const qualified type on Windows
template <typename T>
__device__ const T d1(2);

int *const x = nullptr;
// error: a __device__ variable template cannot
// have a const qualified type on Windows
template <typename T>
__device__ T *const d2(x);

// OK
template <typename T>
__device__ const T *d3;

__device__ void fn() {
  int t1 = d1<int>;

  int *const t2 = d2<int>;

  const int *t3 = d3<int>;
}
```

### 17.5.24. C++17 Features

nvcc 也支持主机编译器默认启用的 C++17 功能。传递 nvcc `-std=c++17` 标志会打开所有 C++17 功能，并使用相应的 C++17 [选项](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_28)调用主机预处理器、编译器和链接器。本节描述对支持的 C++17 的限制特点。

#### 17.5.24.1. Inline Variable

如果代码在整个程序编译模式下使用 nvcc 编译，则使用 `__device__` 或 `__constant__` 或 `__managed__` 内存空间说明符声明的命名空间范围内联变量必须具有内部链接。

```cpp
inline __device__ int xxx; //error when compiled with nvcc in
                           //whole program compilation mode.
                           //ok when compiled with nvcc in
                           //separate compilation mode.

inline __shared__ int yyy0; // ok.

static inline __device__ int yyy; // ok: internal linkage
namespace {
inline __device__ int zzz; // ok: internal linkage
}
```

使用 g++ 主机编译器时，使用 `__managed__` 内存空间说明符声明的内联变量可能对调试器不可见。

#### 17.5.24.2. Structured Binding

不能使用可变内存空间说明符声明结构化绑定。

```cpp
struct S { int x; int y; };
__device__ auto [a1, b1] = S{4,5}; // error
```

### 17.5.25. C++20 Features

nvcc 也支持主机编译器默认启用的 C++20 功能。传递 nvcc `-std=c++20` 标志将打开所有 C++20 功能，并且还会使用相应的 C++20 dialect 选项 [28](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fn29)调用主机预处理器、编译器和链接器。本节介绍支持的 C++20 功能的限制。

#### 17.5.25.1. Module support

CUDA C++ 不支持主机或设备代码中的模块。使用 `module`、`export` 和 `import` 关键字被诊断为错误。

#### 17.5.25.2. Coroutine support

设备代码不支持协程。在设备功能范围内使用 `co_await`、`co_yield` 和 `co_return` 关键字时，在设备编译过程中会被诊断为错误。

#### 17.5.25.3. Three-way comparison operator

主机和设备代码都支持三向比较运算符，但某些用途隐式依赖于主机实现提供的标准模板库中的功能。使用这些运算符可能需要指定标志 `--expt-relaxed-constexpr` 来静默警告，并且该功能要求主机实现满足设备代码的要求。

```cpp
#include<compare>
struct S {
  int x, y, z;
  auto operator<=>(const S& rhs) const = default;
  __host__ __device__ bool operator<=>(int rhs) const { return false; }
};
__host__ __device__ bool f(S a, S b) {
  if (a <=> 1) // ok, calls a user-defined host-device overload
    return true;
  return a < b; // call to an implicitly-declared function and requires
                // a device-compatible std::strong_ordering implementation
}
```

#### 17.5.25.4. Consteval functions

通常，不允许交叉执行空间调用，这会导致编译器诊断（警告或错误）。当使用 `consteval` 说明符声明被调用的函数时，此限制不适用。因此，`__device__` 或 `__global__` 函数可以调用 `__host__ consteval` 函数，而 `__host__` 函数可以调用 `__device__ consteval` 函数。

```cpp
namespace N1 {
//consteval host function
consteval int hcallee() { return 10; }

__device__ int dfunc() { return hcallee(); /* OK */ }
__global__ void gfunc() { (void)hcallee(); /* OK */ }
__host__ __device__ int hdfunc() { return hcallee();  /* OK */ }
int hfunc() { return hcallee(); /* OK */ }
} // namespace N1


namespace N2 {
//consteval device function
consteval __device__ int dcallee() { return 10; }

__device__ int dfunc() { return dcallee(); /* OK */ }
__global__ void gfunc() { (void)dcallee(); /* OK */ }
__host__ __device__ int hdfunc() { return dcallee();  /* OK */ }
int hfunc() { return dcallee(); /* OK */ }
}
```
