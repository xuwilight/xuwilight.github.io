---
title: CuTe 学习笔记（四）CuTe Tensor
date: 2025-03-22 18:00:00
tags: [CUTLASS, CuTe, Tensor]
categories: [Cutlass 学习笔记, CuTe]
mathjax: true
description: 文章介绍了 CuTe Tensor 的核心概念，涵盖 Tensor 的创建、Layout 映射、make_tensor 等基础操作。
---

# CuTe_Tensor

Tensor 是 CuTe 的核心容器，融合了之前描述的 Layout 概念。Tensor 表示一个多维数组，它抽象了数组元素的组织方式以及数组元素的存储细节。简单来说，数据 + Layout = Tensor。

Tensor 由两个模板参数表示：**Engine** 和 **Layout**。Engine 用于管理和访问 tensor 中的数据，类似于 std::vector 和 std::array 中的迭代器；Layout 用于描述 tensor 中元素的布局。

Tensor 可以在任何类型的内存中创建，全局内存、共享内存、寄存器内存以及 tensor 内存等。

## 基础操作

CuTe 的 Tensor 提供了类似于容器的操作运算。

* .data() 返回 Tensor 的第一个元素对应的地址。

* .size() 返回 Tensor 的大小。

* .operator[](Coord) 通过坐标访问 Tensor 中的元素。

* .operator()(Coord) 也是通过坐标访问 Tensor 中的元素。

* .operator()(Coords...) 支持多维坐标访问，等价于 operator()(make_coord(c0, c1...))。

CuTe 也提供了 Layout 支持的操作。以下操作等价于处理 tensor.layout()，详见 [layout](/2025/03/20/Cutlass/cute-layout/) 章节。

* rank<I...>(Tensor) 返回 Tensor 的 rank。

* depth<I...>(Tensor) 返回 Tensor 的 depth。

* shape<I...>(Tensor) 返回 Tensor 的 shape。

* size<I...>(Tensor) 返回 Tensor 的 size。

* layout<I...>(Tensor) 返回 Tensor 的 layout。

* tensor<I...>(Tensor) 返回 Tensor 的子 Tensor。

## Tensor Engines

Tensor 中的 Engine 可以理解为是用于访问数组的迭代器或指针，类似于 std::array 的迭代器。

```cpp
using iterator     =  // The iterator type
using value_type   =  // The iterator value-type
using reference    =  // The iterator reference-type
iterator begin()      // The iterator
```

通常情况下，用户不需要自己构造 Engine。当创建一个 Tensor 时，CuTe 会自动构造适当的 Engine。

## 创建 Tensor

CuTe 可以创建两种类型的 tensor，一种是栈上 Tensor（owning tensor），一种是堆上 Tensor（nonowning tensor）。

owning tensor 负责数据的生命周期（包括内存分配和释放）。拷贝时进行深拷贝（deep copy），完全拷贝底层数据，生成独立的新数据副本。

nonowning tensor 仅作为现有数据的一种"视图"（view）。复制时仅传递对同一内存的引用，不复制底层数据。析构时不会释放底层内存，需由外部管理内存生命周期。

在 cute 中通过 make_tensor() 来创建 tensor。通过传入不同的参数即可创建不同类型的 tensor。

```cpp
// Make an owning Tensor that will allocate a static array
// e.g. make_tensor<float>(Int<12>{})
template <class T, class... Args>
CUTE_HOST_DEVICE constexpr
auto
make_tensor(Args const&... args)
{
  static_assert((not has_dereference<Args>::value && ...), "Expected layout args... in make_tensor<T>(args...)");
  return MakeTensor<T>{}(args...);
}

// Make a non-owning Tensor that will use a pointer (view)
// e.g. make_tensor(vec.data(), 12)
template <class Iterator, class... Args>
CUTE_HOST_DEVICE constexpr
auto
make_tensor(Iterator const& iter, Args const&... args)
{
  static_assert(has_dereference<Iterator>::value, "Expected iterator iter in make_tensor(iter, args...)");
  static_assert((not has_dereference<Args>::value && ...), "Expected layout args... in make_tensor(iter, args...)");
  return MakeTensor<Iterator>{}(iter, args...);
}
```

## Nonowning Tensor

这种 Tensor 就基本上是用户自己申请显存，然后结合特定的 Layout 创建的 Tensor。

在 make_tensor 函数中传入一个自己申请的数组的指针和一个 Layout 或用于构造 Layout 的参数，即可创建一个 nonowning tensor。

用户可以通过 make_gmem_ptr(g) 或 make_smem_ptr(s) 来表明指针指向的数组是在全局内存还是共享内存。

以下是一些示例。

```cpp
float* A = ...;

// 没有指明指针类型
Tensor tensor_8   = make_tensor(A, make_layout(Int<8>{}));  // Construct with Layout
Tensor tensor_8s  = make_tensor(A, Int<8>{});               // Construct with Shape
Tensor tensor_8d2 = make_tensor(A, 8, 2);                   // Construct with Shape and Stride

// Global memory (static or dynamic layouts)
Tensor gmem_8s     = make_tensor(make_gmem_ptr(A), Int<8>{});
Tensor gmem_8d     = make_tensor(make_gmem_ptr(A), 8);
Tensor gmem_8sx16d = make_tensor(make_gmem_ptr(A), make_shape(Int<8>{},16));
Tensor gmem_8dx16s = make_tensor(make_gmem_ptr(A), make_shape (      8  ,Int<16>{}),
                                                       make_stride(Int<16>{},Int< 1>{}));

// Shared memory (static or dynamic layouts)
Layout smem_layout = make_layout(make_shape(Int<4>{},Int<8>{}));
__shared__ float smem[decltype(cosize(smem_layout))::value];   // (static-only allocation)
Tensor smem_4x8_col = make_tensor(make_smem_ptr(smem), smem_layout);
Tensor smem_4x8_row = make_tensor(make_smem_ptr(smem), shape(smem_layout), LayoutRight{});
```

如上所示，通过 make_gmem_ptr 和 make_smem_ptr 可以指明指针指向的内存空间。通过 layout 可以重新解释内存布局。

使用 print 打印上述 tensor 会显示指针类型，指针的宽度，指针地址以及关联的 layout。

```cpp
tensor_8     : ptr[32b](0x7f42efc00000) o _8:_1
tensor_8s    : ptr[32b](0x7f42efc00000) o _8:_1
tensor_8d2   : ptr[32b](0x7f42efc00000) o 8:2
gmem_8s      : gmem_ptr[32b](0x7f42efc00000) o _8:_1
gmem_8d      : gmem_ptr[32b](0x7f42efc00000) o 8:_1
gmem_8sx16d  : gmem_ptr[32b](0x7f42efc00000) o (_8,16):(_1,_8)
gmem_8dx16s  : gmem_ptr[32b](0x7f42efc00000) o (8,_16):(_16,_1)
smem_4x8_col : smem_ptr[32b](0x7f4316000000) o (_4,_8):(_1,_4)
smem_4x8_row : smem_ptr[32b](0x7f4316000000) o (_4,_8):(_8,_1)
```

## Owning tensor

这种是 CuTe 自己创建数据 buffer 和 layout 来生成的 Tensor，通常在寄存器层面创建，比如矩阵累加的 fragments。

通过调用 make_tensor<T> 可以创建一个 Owning tensor，其中 T 是数组元素的类型，入参是一个 Layout 或用于构造 Layout 的参数。这种 Tensor 必须使用具有静态形状和静态步幅的 Layout 来构造。

以下是一些创建 Owning tensor 的示例。

```cpp
// Register memory (static layouts only)
Tensor rmem_4x8_col = make_tensor<float>(Shape<_4,_8>{});
Tensor rmem_4x8_row = make_tensor<float>(Shape<_4,_8>{},
                                         LayoutRight{});
Tensor rmem_4x8_pad = make_tensor<float>(Shape <_4, _8>{},
                                         Stride<_32,_2>{});
Tensor rmem_4x8_like = make_tensor_like(rmem_4x8_pad);
```

上面的例子中，make_tensor 和 make_tensor_like 函数会在寄存器中创建一个拥有所有权的 Tensor。其中 make_tensor_like 创建的 tensor 的数据类型和形状与输入的 tensor 相同，并尝试使用相同的步幅。

使用 print 打印上述 tensor 会产生类似的输出：

```cpp
rmem_4x8_col  : ptr[32b](0x7fff48929460) o (_4,_8):(_1,_4)
rmem_4x8_row  : ptr[32b](0x7fff489294e0) o (_4,_8):(_8,_1)
rmem_4x8_pad  : ptr[32b](0x7fff489295e0) o (_4,_8):(_32,_2)
rmem_4x8_like : ptr[32b](0x7fff48929560) o (_4,_8):(_8,_1)
```

可以看到，每个 tensor 的地址都不一样。

除了 make_tensor 和 make_tensor_like 外，还有下面几种创建 tensor 的方法：

### make_fragment_like

创建一个与输入 tensor 形状相同，但是是列主序的 tensor。

```cpp
Tensor t1 = make_tensor<float>(Shape<_4,_8>{}, LayoutRight{});
Tensor t2 = make_fragment_like(t1);
print_tensor(t1);
print_tensor(t2);
```

结果：

```cpp
make_tensor:
ptr[32b](0x7ffd6f7229a0) o (_4,_8):(_8,_1):
  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00
make_fragment_like:
ptr[32b](0x7ffd6f722a20) o (_4,(_8)):(_1,(_4)):
  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00
```

### make_counting_tensor

根据输入的 layout 创建一个 tensor，元素是 layout 的 offset。

```cpp
Tensor t3 = make_counting_tensor(make_layout(Shape<_4,_8>{}, Stride<_2, _8>{}));
print_tensor(t3);
```

结果：

```cpp
make_counting_tensor:
ArithTuple_0 o (_4,_8):(_2,_8):
    0    8   16   24   32   40   48   56
    2   10   18   26   34   42   50   58
    4   12   20   28   36   44   52   60
    6   14   22   30   38   46   54   62
```

### make_identity_tensor

创建一个元素是自身坐标的 Tensor。这种 Tensor 通常用于边界位置判断和 TMA 数据加载。

```cpp
Tensor t4 = make_identity_tensor(t1.shape());
print_tensor(t4);
```

结果：

```cpp
make_identity_tensor:
ArithTuple(_0,_0) o (_4,_8):(_1@0,_1@1):
  (0,0)  (0,1)  (0,2)  (0,3)  (0,4)  (0,5)  (0,6)  (0,7)
  (1,0)  (1,1)  (1,2)  (1,3)  (1,4)  (1,5)  (1,6)  (1,7)
  (2,0)  (2,1)  (2,2)  (2,3)  (2,4)  (2,5)  (2,6)  (2,7)
  (3,0)  (3,1)  (3,2)  (3,3)  (3,4)  (3,5)  (3,6)  (3,7)
```

## 访问 Tensor

用户可以通过 operator() 和 operator[] 访问 tensor 的元素，这些操作符接受元组类型的逻辑坐标。此外 operator() 还支持使用 "_" 进行切片（slice）访问。

当用户访问 tensor 时，会首先根据 tensor 的 layout 计算对应坐标的偏移量，然后根据指针位置访问对应偏移量位置处的元素。

```cpp
template <class Coord>
decltype(auto) operator[](Coord const& coord) {
  return data()[layout()(coord)];
}
```

可以使用下面几种方式访问 Tensor 数据，包括可变参数运算符 operator() 或类似容器的运算符 operator[] 来读写 tensor。

```cpp
Tensor A = make_tensor<float>(Shape <Shape < _4,_5>,Int<13>>{},
                              Stride<Stride<_12,_1>,    _64>{});
float* b_ptr = ...;
Tensor B = make_tensor(b_ptr, make_shape(13, 20));

// Fill A via natural coordinates op[]
for (int m0 = 0; m0 < size<0,0>(A); ++m0)
  for (int m1 = 0; m1 < size<0,1>(A); ++m1)
    for (int n = 0; n < size<1>(A); ++n)
      A[make_coord(make_coord(m0,m1),n)] = n + 2 * m0;

// Transpose A into B using variadic op()
for (int m = 0; m < size<0>(A); ++m)
  for (int n = 0; n < size<1>(A); ++n)
    B(n,m) = A(m,n);

// Copy B to A as if they are arrays
for (int i = 0; i < A.size(); ++i)
  A[i] = B[i];
```

## Tensor Algebra

Layout Algebra 中的操作同样可以应用到 tensor 中，其本质就是对 Tensor 的 Layout 进行处理，具体参见 [Layout Algebra（一）](/2025/03/21/Cutlass/cute-layout-algebra/) 和 [Layout Algebra（二）](/2025/03/21/Cutlass/cute-layout-algebra2/)。

```cpp
composition(Tensor, Tiler)
logical_divide(Tensor, Tiler)
zipped_divide(Tensor, Tiler)
tiled_divide(Tensor, Tiler)
flat_divide(Tensor, Tiler)
```

需要注意的是，Tensor 没有实现乘积操作 product operations，因为这个操作会使 layout 的范围超出 tensor 的范围了，导致越界。

## Slicing a Tensor

通过坐标访问 tensor 会返回该 tensor 的一个元素，而对 tensor 进行切片访问则会返回切片中所有的元素对应的子 tensor。

切片操作通过在 operator() 中传入 "\_"（下划线字符，cute::Underscore 类型的一个实例）实现。"\_" 的效果与 python 中的 ":" 相同。

举例：

```cpp
// ((_3,2),(2,_5,_2)):((4,1),(_2,13,100))
Tensor A = make_tensor(ptr, make_shape (make_shape (Int<3>{},2), make_shape (       2,Int<5>{},Int<2>{})),
                            make_stride(make_stride(       4,1), make_stride(Int<2>{},      13,     100)));

// ((2,_5,_2)):((_2,13,100))
Tensor B = A(2,_);

// ((_3,_2)):((4,1))
Tensor C = A(_,5);

// (_3,2):(4,1)
Tensor D = A(make_coord(_,_),5);

// (_3,_5):(4,13)
Tensor E = A(make_coord(_,1),make_coord(0,_,1));

// (2,2,_2):(1,_2,100)
Tensor F = A(make_coord(2,_),make_coord(_,3,_));
```

<div align="center">
        <img src="/assets/cute-tensor/image.png" width="100%" height="auto" alt="layout">
        <small></small>
</div>
<br>

在上图中，tensor A 以多种方式被切片，这些切片生成的子 tensor 在原始 tensor 中用阴影部分显示。

需要注意的是，tensor C 和 D 包含相同的元素，但由于使用了 "\_" 与 "make_coord(\_, \_)"，它们的 rank 和 shape 不同。在这两种情况下，结果的 rank 等于切片坐标中 "\_" 的数量。

* A(2, \_)：第 2 行所有元素。

* A(\_, 5)：第 5 列所有元素。

* A(make_coord(\_, 1), make_coord(0, \_, 1))：(\_, 1) 中的 1 对应 3-5 行，"\_" 代表获取 3-5 行的所有元素。(0, \_, 1) 中的 1 对应 10-19 列，0 对应 10，12，14，16，18 列，"\_" 代表全选 10，12，14，16，18 列，与 3-5 行相交即为阴影部分元素。

* A(make_coord(2, \_), make_coord(\_, 3, \_))：(2, \_) 中的 "\_" 代表 0-2 和 3-5 行的全部元素，2 代表从 0-2 中选第 2 行，3-5 中选第 5 行。同理，(\_, 3,\_) 的第二个 "\_" 代表 0-9 列和 10-19 列的全部元素，3 代表 0-9 列的第 6，7 列，10-19 列的第 16，17 列。第一个 "\_" 代表全选第 6，7 列和第 16，17 列所有元素，与第 2 行和第 5 行相交即为阴影部分结果。

## Partitioning a Tensor

通过 composition，tiling 和 slicing 等操作可以实现对 tensor 的分块。在 cute 中常用的有三种分块方式。内部分区 `(inner-partitioning)`，外部分区 `(outer-partitioning)` 和 thread-value 分区 `(TV-layout-partitioning)`。

### 内部分区（inner-partitioning）

在下面的例子中 tensor A 是一个 8 行 24 列的列主序数组，通过一个大小为 4 行 8 列的 tiler 进行 zipped_divide 运算，具体运算过程见[Layout Algebra（二）](/2025/03/21/Cutlass/cute-layout-algebra2/)。此时 tensor A 被解释成行为 (4, 8)，列为 (2, 3) 的 tiled_a。

```cpp
Tensor A = make_tensor(ptr, make_shape(8,24));  // (8,24)
auto tiler = Shape<_4,_8>{};                    // (_4,_8)

Tensor tiled_a = zipped_divide(A, tiler);       // ((_4,_8),(2,3))
```

假设我们想要将其中 4x8 的数据块分配给每个线程块。那么，我们可以使用线程块的坐标来索引第二个维度。

```cpp
Tensor cta_a = tiled_a(make_coord(_,_), make_coord(blockIdx.x, blockIdx.y));  // (_4,_8)
```

此时，每个线程块都对应了 tensor A 中 4*8 大小的数据。这种分区方法称为内部分区 inner-partitioning。

CuTe 中通过调用 inner_partition(Tensor, Tiler, Coord) 函数来实现这种分区方式。同样的还有 local_tile(Tensor, Tiler, Coord)，它是 inner_partition 的一个别名。local_tile 通常应用在线程块级别，用于把一个矩阵分到不同的线程块上。

### 外部分区（outer-partitioning）

继续看上面的例子，假设我们有 32 个线程，并且希望为每个线程处理这些 4x8 数据块中的一个元素。那么，我们可以使用线程来索引第一个维度。

```cpp
Tensor thr_a = tiled_a(threadIdx.x, make_coord(_,_)); // (2,3)
```

此时 32 个线程对应 tiled_a 的 32 行，每个线程处理一行的 6 个元素。

这种分区方式称为外部分区 outer-partitioning。同样的，cute 中通过调用 outer_partition(Tensor, Tiler, Coord) 或 local_partition(Tensor, Layout, Idx) 来实现这种分区。这种分区方法经常应用在线程级别的计算上。

### 线程-值分区（Thread-Value partitioning）

在 cute 中，还有一种常见的分区称为 thread-value 分区。在这种模式中，我们构造一个 layout，这个 layout 的第一维代表线程 thread 之间的 layout，第二维代表一个线程需要处理的元素 value 之间的 layout。我们称这种 layout 为 tv-layout。

在下面的代码中，我们构造了一个 tv-layout，它的第一维是 (2,4):(8,1)，代表了线程的 layout。第二维是 (2,2):(4,16)，代表每个线程处理的元素的 layout。

对于一个 4*8 大小的 tensor A，通过 composition 操作，将 A 中的元素按照 tv-layout 的布局重新排列得到 tensor tv。使用 threadIdx 对 tensor tv 进行索引，就可以得到每个线程需要处理的元素。

```cpp
// Construct a TV-layout that maps 8 thread indices and 4 value indices
//   to 1D coordinates within a 4x8 tensor
// (T8,V4) -> (M4,N8)
auto tv_layout = Layout<Shape <Shape <_2,_4>,Shape <_2, _2>>,
                        Stride<Stride<_8,_1>,Stride<_4,_16>>>{}; // (8,4)

// Construct a 4x8 tensor with any layout
Tensor A = make_tensor<float>(Shape<_4,_8>{}, LayoutRight{});    // (4,8)
// Compose A with the tv_layout to transform its shape and order
Tensor tv = composition(A, tv_layout);                           // (8,4)
// Slice so each thread has 4 values in the shape and order that the tv_layout prescribes
Tensor  v = tv(threadIdx.x, _);                                  // (4)
```

![](/assets/cute-tensor/image_1.png)

上图解释了代码的计算过程。左上角是原始 tensor A 以及其中的元素排列，中上是 tv-layout 的布局，每一行代表一个线程，每一列代表一个线程处理的元素。通过组合操作得到右上角的 tensor tv。此时通过 threadIdx 对 tv 的第一维进行索引就可以得到当前 thread 处理的元素。

底部的图片描绘了每个 thread 处理的元素在原始 tensor 中排布，T0V0 代表 thread0 处理的第 0 个元素。

## 总结

Tensor 由 Engine 和 Layout 组成。Engine 是一个迭代器，用于访问 Tensor 中的元素。Layout 描述了 Tensor 中元素的逻辑布局，并将坐标映射到偏移量。

对 Tensor 的 Tile 操作等价于对 Tensor 的 Layout 进行 Tile 操作。Tensor 可以通过 slice 操作获取子 Tensor。根据应用场景不同可以使用不同的方式对 Tensor 分区。
