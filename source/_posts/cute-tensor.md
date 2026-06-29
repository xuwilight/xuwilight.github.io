---
title: CuTe 学习笔记（四）CuTe Tensor
date: 2025-03-22 18:00:00
tags: [CUTLASS, CuTe, Tensor, GPU]
categories: [Cutlass 学习笔记]
mathjax: true
description: 文章介绍了 CuTe Tensor 的核心概念，涵盖 Tensor 的创建、Layout 映射、make_tensor 等基础操作。
---

# CuTe_Tensor

Tensor是CuTe的核心容器，融合了之前描述的Layout概念。从根本上说，Tensor表示一个多维数组，它抽象了数组元素的组织方式以及数组元素的存储细节。

Tensor由两个模板参数表示：**Engine**和**Layout**。Engine用于管理和访问tensor中的数据，类似与std::vector和std::array中的迭代器；Layout用于描述tensor中元素的布局。Tensor可以在任何类型的内存中创建，全局内存、共享内存、寄存器内存以及Blackwell的tensor内存。

## 基础操作

CuTe的Tensor提供了类似与容器的操作运算。

.data(). 返回Tensor的第一个元素对应的迭代器（地址）。

.size(). 返回Tensor的大小。

.operator[](Coord). 通过坐标访问Tensor中的元素。

.operator()(Coord). 通过坐标访问Tensor中的元素。

.operator()(Coords...). 支持多维坐标访问，等价于operator()(make_coord(c0,c1...))。

CuTe也提供了Layout支持的操作。以下操作等价与处理tensor.layout()，详见layout章节。

rank<I...>(Tensor). 返回Tensor的rank。

depth<I...>(Tensor). 返回Tensor的depth。

shape<I...>(Tensor). 返回Tensor的shape。

size<I...>(Tensor). 返回Tensor的size。

layout<I...>(Tensor). 返回Tensor的layout。

tensor<I...>(Tensor). 返回Tensor的子Tensor。

## Tensor Engines

Tensor中的Engine可以理解为是用于访问数组数组的迭代器或指针，类似于std::array的迭代器。

```cpp
using iterator     =  // The iterator type
using value_type   =  // The iterator value-type
using reference    =  // The iterator reference-type
iterator begin()      // The iterator
```

通常情况下，用户不需要自己构造Engine。当创建一个Tensor时，会自动构造适当的Engine。

## 创建Tensor

cute可以创建两种类型的tensor，一种是拥有数据所有权的owning tensor，一种是不拥有数据所有权的nonowning tensor。

拥有数据所有全的tensor负责数据的生命周期（包括内存分配和释放）。拷贝时进行深拷贝（deep copy），完全拷贝底层数据，生成独立的新数据副本。

不拥有数据所有权的tensor仅作为现有数据的一种"视图"（view）。复制时仅传递对同一内存的引用，不复制底层数据。析构时不会释放底层内存，需由外部管理内存生命周期。

在cute中通过make_tensor()来创建tensor。通过传入不同的参数即可创建不同类型的tensor。

```cpp
// Make an owning Tensor that will allocate a static array
// e.g. make_tensor<float>(Int<12>{})
template <class T, class... Args>
CUTE_HOST_DEVICE constexpr
auto
make_tensor(Args const&... args)
{
  static_assert((not has_dereference<Args>::value && ...), "Expected layout args... in make_tensor<T>(args...)");
  return MakeTensor<T>{}(args...);
}

// Make a non-owning Tensor that will use a pointer (view)
// e.g. make_tensor(vec.data(), 12)
template <class Iterator, class... Args>
CUTE_HOST_DEVICE constexpr
auto
make_tensor(Iterator const& iter, Args const&... args)
{
  static_assert(has_dereference<Iterator>::value, "Expected iterator iter in make_tensor(iter, args...)");
  static_assert((not has_dereference<Args>::value && ...), "Expected layout args... in make_tensor(iter, args...)");
  return MakeTensor<Iterator>{}(iter, args...);
}
```

### Nonowning Tensor

在make_tensor函数中传入一个数组的迭代器（指针）和一个Layout或用于构造Layout的参数，即可创建一个nonowning tensor。

用户可以通过make_gmem_ptr(g)或make_smem_ptr(s)来表明指针指向的数组是在全局内存还是共享内存。

以下是一些示例。

```cpp
float* A = ...;

// 没有指明指针类型
Tensor tensor_8   = make_tensor(A, make_layout(Int<8>{}));  // Construct with Layout
Tensor tensor_8s  = make_tensor(A, Int<8>{});               // Construct with Shape
Tensor tensor_8d2 = make_tensor(A, 8, 2);                   // Construct with Shape and Stride

// Global memory (static or dynamic layouts)
Tensor gmem_8s     = make_tensor(make_gmem_ptr(A), Int<8>{});
Tensor gmem_8d     = make_tensor(make_gmem_ptr(A), 8);
Tensor gmem_8sx16d = make_tensor(make_gmem_ptr(A), make_shape(Int<8>{},16));
Tensor gmem_8dx16s = make_tensor(make_gmem_ptr(A), make_shape (      8  ,Int<16>{}),
                                                   make_stride(Int<16>{},Int< 1>{}));

// Shared memory (static or dynamic layouts)
Layout smem_layout = make_layout(make_shape(Int<4>{},Int<8>{}));
__shared__ float smem[decltype(cosize(smem_layout))::value];   // (static-only allocation)
Tensor smem_4x8_col = make_tensor(make_smem_ptr(smem), smem_layout);
Tensor smem_4x8_row = make_tensor(make_smem_ptr(smem), shape(smem_layout), LayoutRight{});
```

如上所示，通过make_gmem_ptr和make_smem_ptr可以指明指针指向的内存空间。通过layout可以重新解释内存布局。

使用print打印上述tensor会显示指针类型，指针的宽度，指针地址以及关联的layout。

```cpp
tensor_8     : ptr[32b](0x7f42efc00000) o _8:_1
tensor_8s    : ptr[32b](0x7f42efc00000) o _8:_1
tensor_8d2   : ptr[32b](0x7f42efc00000) o 8:2
gmem_8s      : gmem_ptr[32b](0x7f42efc00000) o _8:_1
gmem_8d      : gmem_ptr[32b](0x7f42efc00000) o 8:_1
gmem_8sx16d  : gmem_ptr[32b](0x7f42efc00000) o (_8,16):(_1,_8)
gmem_8dx16s  : gmem_ptr[32b](0x7f42efc00000) o (8,_16):(_16,_1)
smem_4x8_col : smem_ptr[32b](0x7f4316000000) o (_4,_8):(_1,_4)
smem_4x8_row : smem_ptr[32b](0x7f4316000000) o (_4,_8):(_8,_1)
```

### Owning tensor

通过调用make_tensor<T>可以创建一个拥有所有权的Tensor，其中T是数组元素的类型，入参是一个Layout或用于构造Layout的参数。这种Tensor必须使用具有静态形状和静态步幅的Layout来构造。

以下是一些创建拥有所有权的Tensor的示例。

```cpp
// Register memory (static layouts only)
Tensor rmem_4x8_col = make_tensor<float>(Shape<_4,_8>{});
Tensor rmem_4x8_row = make_tensor<float>(Shape<_4,_8>{},
                                         LayoutRight{});
Tensor rmem_4x8_pad = make_tensor<float>(Shape <_4, _8>{},
                                         Stride<_32,_2>{});
Tensor rmem_4x8_like = make_tensor_like(rmem_4x8_pad);
```

上面的例子中，make_tensor和make_tensor_like函数会在寄存器中创建一个拥有所有权的Tensor。其中make_tensor_like创建的tensor的数据类型和形状与输入的tensor相同，并尝试使用相同的步幅。

使用print打印上述tensor会产生类似的输出：

```cpp
rmem_4x8_col  : ptr[32b](0x7fff48929460) o (_4,_8):(_1,_4)
rmem_4x8_row  : ptr[32b](0x7fff489294e0) o (_4,_8):(_8,_1)
rmem_4x8_pad  : ptr[32b](0x7fff489295e0) o (_4,_8):(_32,_2)
rmem_4x8_like : ptr[32b](0x7fff48929560) o (_4,_8):(_8,_1)
```

我们从指针地址可以看到，每个tensor的地址都不一样。

除了make_tensor和make_tensor_like外，还有几种创建tensor的方法：

### make_fragment_like

创建一个与输入tensor形状相同，但是是列主序的tensor。

```cpp
Tensor t1 = make_tensor<float>(Shape<_4,_8>{}, LayoutRight{});
Tensor t2 = make_fragment_like(t1);
print_tensor(t1);
print_tensor(t2);
```

结果：

```cpp
make_tensor:
ptr[32b](0x7ffd6f7229a0) o (_4,_8):(_8,_1):
  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00
make_fragment_like:
ptr[32b](0x7ffd6f722a20) o (_4,(_8)):(_1,(_4)):
  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00
```

### make_counting_tensor

根据输入的layout创建一个tensor，其值是layout的offset。

```cpp
Tensor t3 = make_counting_tensor(make_layout(Shape<_4,_8>{}, Stride<_2, _8>{}));
print_tensor(t3);
```

结果：

```cpp
make_counting_tensor:
ArithTuple_0 o (_4,_8):(_2,_8):
    0    8   16   24   32   40   48   56
    2   10   18   26   34   42   50   58
    4   12   20   28   36   44   52   60
    6   14   22   30   38   46   54   62
```

### make_identity_tensor

创建一个tensor，将形状内的坐标映射到它们自身。

```cpp
Tensor t4 = make_identity_tensor(t1.shape());
print_tensor(t4);
```

结果：

```cpp
make_identity_tensor:
ArithTuple(_0,_0) o (_4,_8):(_1@0,_1@1):
  (0,0)  (0,1)  (0,2)  (0,3)  (0,4)  (0,5)  (0,6)  (0,7)
  (1,0)  (1,1)  (1,2)  (1,3)  (1,4)  (1,5)  (1,6)  (1,7)
  (2,0)  (2,1)  (2,2)  (2,3)  (2,4)  (2,5)  (2,6)  (2,7)
  (3,0)  (3,1)  (3,2)  (3,3)  (3,4)  (3,5)  (3,6)  (3,7)
```

## 访问Tensor

用户可以通过operator()和operator[]访问tensor的元素，这些操作符接受元组类型的逻辑坐标。此外operator()还支持使用"_"进行切片（slice）访问。

当用户访问tensor时，会首先根据tensor的layout计算对应坐标的偏移量，然后通过迭代器访问对应偏移量位置处的元素。

```cpp
template <class Coord>
decltype(auto) operator[](Coord const& coord) {
  return data()[layout()(coord)];
}
```

我们可以使用自然坐标、可变参数运算符operator()或类似容器的运算符operator[]来读写tensor。

```cpp
Tensor A = make_tensor<float>(Shape <Shape < _4,_5>,Int<13>>{},
                              Stride<Stride<_12,_1>,    _64>{});
float* b_ptr = ...;
Tensor B = make_tensor(b_ptr, make_shape(13, 20));

// Fill A via natural coordinates op[]
for (int m0 = 0; m0 < size<0,0>(A); ++m0)
  for (int m1 = 0; m1 < size<0,1>(A); ++m1)
    for (int n = 0; n < size<1>(A); ++n)
      A[make_coord(make_coord(m0,m1),n)] = n + 2 * m0;

// Transpose A into B using variadic op()
for (int m = 0; m < size<0>(A); ++m)
  for (int n = 0; n < size<1>(A); ++n)
    B(n,m) = A(m,n);

// Copy B to A as if they are arrays
for (int i = 0; i < A.size(); ++i)
  A[i] = B[i];
```

## Tiling a Tensor

Layout Algebra中的操作同样可以应用到tensor中。

```cpp
   composition(Tensor, Tiler)
logical_divide(Tensor, Tiler)
 zipped_divide(Tensor, Tiler)
  tiled_divide(Tensor, Tiler)
   flat_divide(Tensor, Tiler)
```

需要注意的是，乘积操作product operations并未为tensor 实现，因为这些操作通常会生成具有更大陪域的布局，这意味着tensor可能需要访问其先前边界之外的元素。布局（Layouts）可以用于乘积操作，但tensor不行。

## Slicing a Tensor

通过坐标访问tensor会返回该tensor的一个元素，而对tensor进行切片访问则会返回切片中所有的元素对应的子tensor。

切片操作通过在operator()中传入"_"（下划线字符，cute::Underscore类型的一个实例）实现。"_"的效果与python中的":"相同。

举例：

```cpp
// ((_3,2),(2,_5,_2)):((4,1),(_2,13,100))
Tensor A = make_tensor(ptr, make_shape (make_shape (Int<3>{},2), make_shape (       2,Int<5>{},Int<2>{})),
                            make_stride(make_stride(       4,1), make_stride(Int<2>{},      13,     100)));

// ((2,_5,_2)):((_2,13,100))
Tensor B = A(2,_);

// ((_3,_2)):((4,1))
Tensor C = A(_,5);

// (_3,2):(4,1)
Tensor D = A(make_coord(_,_),5);

// (_3,_5):(4,13)
Tensor E = A(make_coord(_,1),make_coord(0,_,1));

// (2,2,_2):(1,_2,100)
Tensor F = A(make_coord(2,_),make_coord(_,3,_));
```

![](/assets/cute-tensor/image.png)

在上图中， tensor A以多种方式被切片，这些切片生成的子tensor在原始tensor中用阴影部分显示。

需要注意的是，tensor C 和 D 包含相同的元素，但由于使用了 "_"与"make_coord(_,_)"，它们的rank和shape不同。在这两种情况下，结果的rank等于切片坐标中"_"的数量。

A(2,_)：第2行所有元素。

A(_,5)：第5列所有元素。

A(make_coord(_,1),make_coord(0,_,1))：(_,1)中的1对应3-5行，"_"代表获取3-5行的所有元素。(0,_,1)中的1对应10-19列，0对应10，12，14，16，18列，"_"代表全选10，12，14，16，18列，与3-5行相交即为阴影部分元素。

A(make_coord(2,_),make_coord(_,3,_))：(2,_)中的"_"代表0-2和3-5行的全部元素，2代表从0-2中选第2行，3-5中选第5行。同理，(_,3,_)的第二个"_"代表0-9列和10-19列的全部元素，3代表0-9列的第6，7列，10-19列的第16，17列。第一个"_"代表全选16，7列和16，17列所有元素，与第2行和第5行相交即为阴影部分结果。

## Partitioning a Tensor

通过composition，tiling和slicing等操作可以实现对tensor的分块。在cute中常用的有三种分块方式。内部分区（inner-partitioning），外部分区（ outer-partitioning）和thread-value分区（ TV-layout-partitioning）。

### 内部分区（inner-partitioning）

在下面的例子中tensor A是一个8行24列的列主序数组，通过一个大小为4行8列的tiler进行zipped_divide运算，具体运算过程见第二章。此时tensor A被解释成行为(4,8)，列为(2,3)的tiled_a。

```cpp
Tensor A = make_tensor(ptr, make_shape(8,24));  // (8,24)
auto tiler = Shape<_4,_8>{};                    // (_4,_8)

Tensor tiled_a = zipped_divide(A, tiler);       // ((_4,_8),(2,3))
```

假设我们想要将其中 4x8 的数据块分配给每个线程块。那么，我们可以使用线程块的坐标来索引第二个维度。

```cpp
Tensor cta_a = tiled_a(make_coord(_,_), make_coord(blockIdx.x, blockIdx.y));  // (_4,_8)
```

此时，每个线程块都对应了tensor A中4*8大小的数据。这种分区方法称为内部分区inner-partitioning。cute中通过调用inner_partition(Tensor, Tiler, Coord)函数来实现这种分区方式。同样的还有local_tile(Tensor, Tiler, Coord)，它是inner_partition的一个别名。local_tile通常应用在线程块级别，用于把一个矩阵分到不同的线程块上。

### 外部分区（outer-partitioning）

继续看上面的例子，假设我们有 32 个线程，并且希望为每个线程处理这些 4x8 数据块中的一个元素。那么，我们可以使用线程来索引第一个维度。

```cpp
Tensor thr_a = tiled_a(threadIdx.x, make_coord(_,_)); // (2,3)
```

此时32个线程对应tiled_a的32行，每个线程处理一行的6个元素。

这种分区方式称为外部分区outer-partitioning。同样的，cute中通过调用outer_partition(Tensor, Tiler, Coord)或local_partition(Tensor, Layout, Idx)来实现这种分区。这种分区方法经常应用在线程级别的计算上。

### 线程-值分区（Thread-Value partitioning）

在cute中，还有一种常见的分区称为thread-value分区。在这种模式中，我们构造一个layout，这个layout的第一维代表线程thread之间的layout，第二维代表一个线程需要处理的元素value之间的layout。我们称这种layout为tv-layout。

在下面的代码中，我们构造了一个tv-layout，它的第一维是(2,4):(8,1)，代表了线程的layout。第二维是(2,2):(4,16)，代表每个线程处理的元素的layout。

对于一个4*8大小的tensor A，通过composition操作，将A中的元素按照tv-layout的布局重新排列得到tensor tv。使用threadIdx对tensor tv进行索引，就可以得到每个线程需要处理的元素。

```cpp
// Construct a TV-layout that maps 8 thread indices and 4 value indices
//   to 1D coordinates within a 4x8 tensor
// (T8,V4) -> (M4,N8)
auto tv_layout = Layout<Shape <Shape <_2,_4>,Shape <_2, _2>>,
                        Stride<Stride<_8,_1>,Stride<_4,_16>>>{}; // (8,4)

// Construct a 4x8 tensor with any layout
Tensor A = make_tensor<float>(Shape<_4,_8>{}, LayoutRight{});    // (4,8)
// Compose A with the tv_layout to transform its shape and order
Tensor tv = composition(A, tv_layout);                           // (8,4)
// Slice so each thread has 4 values in the shape and order that the tv_layout prescribes
Tensor  v = tv(threadIdx.x, _);                                  // (4)
```

![](/assets/cute-tensor/image_1.png)

上图解释了代码的计算过程。左上角是原始tensor A以及其中的元素排列，中上是tv-layout的布局，每一行代表一个线程，每一列代表一个线程处理的元素。通过组合操作得到右上角的tensor tv。此时通过threadIdx对tv的第一维进行索引就可以得到当前thread处理的元素。

底部的图片描绘了每个thread处理的元素在原始tensor中排布，T0V0代表thread0处理的第0个元素。

## 总结

Tensor由Engine和Layout组成。

Engine是一个迭代器，用于访问Tensor中的元素。

Layout描述了Tensor中元素的逻辑布局，并将坐标映射到偏移量。

对Tensor的Tile操作等价与对Tensor的Layout进行Tile操作。

Tensor可以通过slice操作获取子Tensor。

根据应用场景不同可以使用不同的方式对Tensor分区。
