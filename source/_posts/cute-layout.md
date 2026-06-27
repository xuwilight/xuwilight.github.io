---
title: CuTe 学习笔记（一） Layout
date: 2025-03-20 16:00:00
tags: [CUTLASS, CuTe, Layout, GPU]
categories: [Cutlass 学习笔记]
description: 文章介绍了 CuTe Layout 的核心概念，涵盖 Layout 的组成、Stride、Tile、Partition、Slice 等操作。
mathjax: true
---


我们对于数据的描述通常是多维的，但是在计算机的内存中数据都是一维的，所以需要通过一种方式将数据的高维坐标和一维索引联系起来。

使用 shape 和 stride 可以很好的描述这种关系，即 shape 表示数据的形状，stride 表示数据的间隔。

shape = [2, 4]，stride = [4, 1] 表示数据有两行四列，其中第一维元素间隔为 4，第二维元素间隔为 1，表示数据在行方向上连续，这也是常说的行主序。

shape = [2, 4]，stride = [1, 2] 同样表示数据有两行四列，不过一行之间元素间隔为 2，一列之间元素间隔为 1，也就是元素按照列主序排列。

但是这种方法的一个局限性是，一个维度的 stride 必须是相同的。如果一行的元素是 [0, 1, 4, 5] 就没法只用一个 stride 表示了。

所以为了能够表示更复杂的数据逻辑，cute 在 shape 和 stride 的基础上引入了层级多维布局（hierarchically multidimensional layouts）。

## CuTe Layouts

CuTe Layout 可以理解为是一种描述逻辑空间和物理空间关系的函数，通过 layout 将数据从逻辑空间坐标映射到物理空间索引。

Layout 由 shape 和 stride 组成，只不过这里的 shape 和 stride 可以使用元组（Tuple）类型来表达更复杂的结构。

介绍 Layout 前首先介绍下 cute 中的 **Integers** 和 **Tuple**。

### Integers

CuTe 中的整数分为两种：dynamic (run-time) 和 static (compile-time) integers。

dynamic integers 就是运行时整数，与常见的 int，size_t，uint16_t 类型相同，是在运行时确定的。

static integers 是静态整数，也可以称为编译时整数。这种数是在编译时就确定的，可以在编译时进行计算。比如矩阵分块的大小，这种值一般在编译时就可以确定，因此可以使用静态整数。

在 cute 中，静态整数的构造方式为 Int<Value>{}，其中 Int<Value> 是静态整数类型 cute::C<Value>，加上 {} 是类型的实例化。

对于常用的数值，如：Int<1>, Int<2>, Int<32>, Int<128>，可以直接使用 _1, _2, _32, _128 代替。具体可以参考 cutlass/include/cute/numeric/integral_constant.hpp。

### Tuple

元组（tuple）是一个有限的、有序的、包含零个或多个元素的列表。cute::tuple 类的行为类似于 std::tuple，但它可以在设备和主机上运行。它对模板参数施加了一些限制，并简化了实现以提高性能和简洁性。

### Layout

CuTe 中的 shape 和 stride 都是由 tuple 构成的。而 layout 是由 shape 和 stride 组成的，即 `cute::Layout<Shape, Stride>`

## Layout 的构造

Layout 可以通过 cute::make_layout 函数进行构造，构造时整数可以根据需要选择动态和静态。

```cpp
cute::Layout layout1 = make_layout(make_shape(4, 8));  // column-major, shape = [4, 8], stride = [1, 4]

cute::tuple shape = make_shape(_4{}, _8{});
cute::tuple stride = make_stride(_8{}, _1{});
cute::Layout layout2 = make_layout(shape, stride);  // row-major, shape = [4, 8], stride = [8, 1]
```

layout1 的 shape 是 [4, 8]，没有指定 stride，但是由于 cutlass 中默认是 column-major，所以 stride 默认是 [1, 4]。

layout2 的 shape 和 layout1 相同，但是 stride 是 [8, 1]。

print，print_layout 或 print_latex 函数可以打印 layout 的形状。通过打印可以看到，layout1 是 (4,8):(1,4)，layout2 是 (4,8):(8,1)。其中冒号前面是 shape，冒号后面是 stride。

通过 print_latex 打印可以看到两个 layout 的形状，其中下图的上边是 layout1 的形状，下边是 layout2 的形状。根据 stride 的不同，两个 layout 中 offset 的排列也不同。

<div align="center">
        <img src="/assets/cute-layout/image_3.png" width="60%" height="auto" alt="layout1">
        <small>layout 1</small>
</div>
<br>

<div align="center">
        <img src="/assets/cute-layout/image_4.png" width="60%" height="auto" alt="layout2">
        <small>layout 2</small>
</div>
<br>


shape 和 stride 通过嵌套可以表示更复杂的逻辑结构。

```cpp
cute::Layout layout3 = make_layout(make_shape(4, make_shape(2, 4)), make_stride(2, make_stride(1, 8))); // shape = [4, (2, 4)], stride = [2, (1, 8)]

cute::tuple shape = make_shape(make_shape(2, 2), make_shape(2, 4));
cute::tuple stride = make_stride(make_stride(1, 4), make_stride(2, 8));
cute::Layout layout4 = make_layout(shape, stride); // shape = [(2, 2), (2, 4)], stride = [(1, 4), (2, 8)]
```

上面代码构建的 layout3 的 shape = [4, (2, 4)]，stride = [2, (1, 8)]，layout4 的 shape = [(2, 2), (2, 4)], stride = [(1, 4), (2, 8)]。

可以看到，layout3 和 layout4 的 shape 和 stride 从整数元组变成了嵌套的元组。

layout3 的 shape = [4, (2, 4)] 代表第一维有 4 行，第二维有 (2, 4) 列，这里的 (2, 4) 可以理解为第二维一共有 2 * 4 = 8 列，8 列又从逻辑上被分为了 4 组，一组有 2 列。

stride = [2, (1, 8)]，维度和 shape 一一对应，stride=2 代表第一维中相邻元素在物理空间的间隔是 2；(1, 8) 对应 shape 中的 (2, 4)，表示 shape 为 2 的维度的 stride=1，shape=4 的维度的 stride=8。

layout3 print_latex 打印结果如下：

<div align="center">
        <img src="/assets/cute-layout/image_5.png" width="60%" height="auto" alt="layout3">
        <small>layout3</small>
</div>
<br>

从图中可以看到，虽然还是 4 行 8 列，但是 layout 中每个位置的 index 发生了变化。

下面分析 layout3 是如何得到这种排列的。

从前面可知，layout3 是一个 4 行 8 列的形状，由于 8 被分成了 (2, 4)，我们使用不同的颜色进行区分，那么一行一共有 4 种颜色，每种颜色包含两个元素，如下图所示。

<div align="center">
        <img src="/assets/cute-layout/image_6.png" width="60%" height="auto" alt="layout3 colors">
        <small>layout3 颜色分组</small>
</div>
<br>

有了形状后再根据 stride 计算每个位置的 offset。第 0 个位置的 offset 就是 0，第一个维度的 stride=2，表示在第一个维度中相邻元素之间相差 2，因此第 0 列的 4 个元素的 offset 为 [0, 2, 4, 6]。

<div align="center">
        <img src="/assets/cute-layout/image_7.png" width="60%" height="auto" alt="layout3 dim1">
        <small>layout3 第一维 offset</small>
</div>
<br>

第二个维度的 stride 是 (1,8)。(1,8) 中的 1 表示对应维度的元素间隔是 1，即相同色块之间的元素间隔是 1，因此可以得到第一列元素的 offset 为 [1, 3, 5, 7]。

<div align="center">
        <img src="/assets/cute-layout/image_8.png" width="60%" height="auto" alt="layout3 dim2 inner">
        <small>layout3 第二维 offset (内层)</small>
</div>
<br>

stride(1,8) 中的第二个维度是 8，表示相邻色块之间的间隔是 8，因此可以得到第 2，4，6 列元素的 offset。不难发现，他们也同时满足第一维 stride=2 的要求。

<div align="center">
        <img src="/assets/cute-layout/image_9.png" width="60%" height="auto" alt="layout3 dim2 outer">
        <small>layout3 第二维 offset (外层)</small>
</div>
<br>

以此类推，可以推导出其他位置的元素。

<div align="center">
        <img src="/assets/cute-layout/image_10.png" width="60%" height="auto" alt="layout3 full">
        <small>layout3 完整布局</small>
</div>
<br>

同样的，我们可以得到 layout4 的元素布局。layout4 的 shape 是 [(2, 2), (2, 4)], stride 是 [(1, 4), (2, 8)]。print_latex 的打印结果如下：

<div align="center">
        <img src="/assets/cute-layout/image_11.png" width="60%" height="auto" alt="layout4">
        <small>layout4 打印结果</small>
</div>
<br>

下面分析 layout4 是如何得到这种排列的。

layout4 的 shape 的第一个维度是 (2, 2)，表明第一维一共有 2×2=4 行，每一行被分成 2 块，每一块有 2 个元素。第二个维度是 (2, 4)，表示第二个维度有 2×4=8 列，被分成 4 块，每一块有 2 个元素。

因此，layout4 的形状可以看成下面这样，由 2×4 个色块组成，每一个色块有 2×2 个元素。

<div align="center">
        <img src="/assets/cute-layout/image_12.png" width="60%" height="auto" alt="layout4 blocks">
        <small>layout4 色块组成</small>
</div>
<br>

然后根据 stride 排列 layout4 中的元素。首先，stride 的第一个维度是 (1, 4)，表示 shape 的第一个维度的内层维度相邻元素间隔是 1，外层维度相邻元素间隔是 4。可以得到

<div align="center">
        <img src="/assets/cute-layout/image_13.png" width="60%" height="auto" alt="layout4 dim1">
        <small>layout4 第一维 stride</small>
</div>
<br>

stride 的第二个维度是 (2, 8)，表示 shape 的第二个维度的内层维度相邻元素间隔是 2，第二个维度的外层维度相邻元素间隔是 8。可以得到

<div align="center">
        <img src="/assets/cute-layout/image_14.png" width="60%" height="auto" alt="layout4 dim2">
        <small>layout4 第二维 stride</small>
</div>
<br>

以此类推，得到 layout4 最终的形状

<div align="center">
        <img src="/assets/cute-layout/image_15.png" width="60%" height="auto" alt="layout4 full">
        <small>layout4 完整布局</small>
</div>
<br>

总的来说，可以把这种嵌套的 shape 理解成((维度1内层大小，维度1外层1大小，维度1外层2大小， ...)， (维度2内层大小，维度2外层1大小，维度2外层2大小，...))。

通过这种嵌套的 shape 和 stride，cute 的 layout 可以表示更加复杂的数据分布。

## 属性函数

`rank(Layout)`: 获取 layout 的维度。

`depth(Layout)`: 获取 layout 的嵌套层数。

| Layout | rank | depth |
| :---: | :---: | :---: |
| 2 | 1 | 0 |
| (4) | 1 | 1 |
| (2,3) | 2 | 1 |
| (2,(2,4)) | 2 | 2 |
| ((2,4),(2,(2,3))) | 2 | 3 |
| ((3,2),(2,4),(2,3)) | 3 | 2 |

`get<I>(Layout)`: 获取 layout 中第 I 个维度的 sub-layout。

```cpp
auto layout = make_layout(make_shape(make_shape(3,2), make_shape(2,4), make_shape(2,3)));
auto sub_layout = get<2>(layout);
print("layout = ");print(layout);print("\n");
print("sub_layout = ");print(sub_layout);print("\n");

// layout = ((3,2),(2,4),(2,3)):((_1,3),(6,12),(48,96))
// sub_layout = (2,3):(48,96)
```

`shape(Layout)`: 获取 layout 的 shape。

`stride(Layout)`: 获取 layout 的 stride。

`size(Layout)`: layout 在逻辑空间包含的元素数量，即 shape 对应的大小。

`cosize(Layout)`: layout 在物理空间包含的真实的元素数量。

layout A = (2,2):(2,4) 的 shape = (2,2)，代表一共有 4 个元素，因此 size(A) = 4。但是实际覆盖的元素是 8，因此 cosize(A) = 8。

## 访问 layout 中的元素

通过逻辑空间的坐标可以获取对应位置数据在物理空间的 index 或者说是 offset。

$$
\text{offset} = \text{Layout}(\text{coordinate})
$$


cute 中有两种访问元素的方法，一种是使用具体坐标访问特定的元素位置，另一种是使用下划线 "_" 进行 slice 访问，类似于 python 中的 ":"。

### 单个元素的访问

cute 有三种坐标形式，1-D 坐标，2-D 坐标和 natural 坐标。

以(3,(2,3))为例，使用不同类型的坐标访问 layout 都会返回对应位置的 offset。

<div align="center">
        <img src="/assets/cute-layout/image_16.png" width="60%" height="auto" alt="coordinate access">
        <small>坐标访问 layout</small>
</div>
<br>

```cpp
auto layout = make_layout(make_shape(3, make_shape(2,3)));  // (3, (2,3))
int offset1 = layout(5);  // 5
int offset2 = layout(make_coord(2,1)); // 5
int offset3 = layout(make_coord(2,make_coord(1,0))); // 5
```

| 1-D | 2-D | Natural | offset | 1-D | 2-D | Natural | offset |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | (0,0) | (0,(0,0)) | 0 | 9 | (0,3) | (0,(1,1)) | 9 |
| 1 | (1,0) | (1,(0,0)) | 1 | 10 | (1,3) | (1,(1,1)) | 10 |
| 2 | (2,0) | (2,(0,0)) | 2 | 11 | (2,3) | (2,(1,1)) | 11 |
| 3 | (0,1) | (0,(1,0)) | 3 | 12 | (0,4) | (0,(0,2)) | 12 |
| 4 | (1,1) | (1,(1,0)) | 4 | 13 | (1,4) | (1,(0,2)) | 13 |
| 5 | (2,1) | (2,(1,0)) | 5 | 14 | (2,4) | (2,(0,2)) | 14 |
| 6 | (0,2) | (0,(0,1)) | 6 | 15 | (0,5) | (0,(1,2)) | 15 |
| 7 | (1,2) | (1,(0,1)) | 7 | 16 | (1,5) | (1,(1,2)) | 16 |
| 8 | (2,2) | (2,(0,1)) | 8 | 17 | (2,5) | (2,(1,2)) | 17 |

#### 使用“_”访问元素。

使用 "_" 可以访问对应位置的多个元素。

<div align="center">
        <img src="/assets/cute-layout/image_17.png" width="90%" height="auto" alt="underscore access">
        <small>使用 _ 访问元素</small>
</div>
<br>

以上图为例，

A(37)：按 column-major 的顺序数 37 个数得到 49。

A(5,4)：第 5 行第 4 列，得到 49。

A((1,2),(0,2))：(1,2) 的外层维度坐标是 2，确定 [4,5] 行，内层维度坐标是 1，确定第 5 行；(0,2) 的外层维度坐标是 2，确定 [4,5] 列，内层维度坐标是 0，确定第 4 列。

A((1,(0,1)),(0,(0,1)))：根据第一维 (1,(0,1)) 最外层维度 (0,1) 中的 1，确定 [4,5,6,7] 行；(0,1) 中的 0，确定 [4,5] 行；最内层的 1，确定第 5 行；同理可确定第 4 列。

A(_,2) 表示第 2 列的所有元素；

A((\_,1),(\_,2))：(\_,1) 的外层维度 index 是 1，确定 [2,3] 行，内层 index 是 "\_"，表示 [2,3] 行的所有元素。同理可确定 [4,5] 列的所有元素，行与列交叉部分就是上图蓝框部分数据。

## Layout操作

### 获取 sub-layouts

`layout<I...>`

```cpp
Layout a   = Layout<Shape<_4,Shape<_3,_6>>>{}; // (4,(3,6)):(1,(4,12))
Layout a0  = layout<0>(a);                     // 4:1
Layout a1  = layout<1>(a);                     // (3,6):(4,12)
Layout a10 = layout<1,0>(a);                   // 3:4
Layout a11 = layout<1,1>(a);                   // 6:12
```

`select<I...>`

```cpp
Layout a   = Layout<Shape<_2,_3,_5,_7>>{};     // (2,3,5,7):(1,2,6,30)
Layout a13 = select<1,3>(a);                   // (3,7):(2,30)
Layout a01 = select<0,1,3>(a);                 // (2,3,7):(1,2,30)
Layout a2  = select<2>(a);                     // (5):(6)
```

`take<ModeBegin, ModeEnd>`

```cpp
Layout a   = Layout<Shape<_2,_3,_5,_7>>{};     // (2,3,5,7):(1,2,6,30)
Layout a13 = take<1,3>(a);                     // (3,5):(2,6)
Layout a14 = take<1,4>(a);                     // (3,5,7):(2,6,30)
// take<1,1> not allowed. Empty layouts not allowed.
```

### Concatenation

Layout 可以通过 `make_layout` 实现 concatenation

```cpp
Layout a = Layout<_3,_1>{};                     // 3:1
Layout b = Layout<_4,_3>{};                     // 4:3
Layout row = make_layout(a, b);                 // (3,4):(1,3)
Layout col = make_layout(b, a);                 // (4,3):(3,1)
Layout q   = make_layout(row, col);             // ((3,4),(4,3)):((1,3),(3,1))
Layout aa  = make_layout(a);                    // (3):(1)
Layout aaa = make_layout(aa);                   // ((3)):((1))
Layout d   = make_layout(a, make_layout(a), a); // (3,(3),3):(1,(1),1)
```

也可以通过这三个函数实现 `append`, `prepend`, or `replace`.

```cpp
Layout a = Layout<_3,_1>{};                     // 3:1
Layout b = Layout<_4,_3>{};                     // 4:3
Layout ab = append(a, b);                       // (3,4):(1,3)
Layout ba = prepend(a, b);                      // (4,3):(3,1)
Layout c  = append(ab, ab);                     // (3,4,(3,4)):(1,3,(1,3))
Layout d  = replace<2>(c, b);                   // (3,4,4):(1,3,3)
```

### Grouping and flattening

通过 `group<ModeBegin, ModeEnd>` 或 `flatten` 可以对 layout 进行组合和展开。

```cpp
Layout a = Layout<Shape<_2,_3,_5,_7>>{};  // (_2,_3,_5,_7):(_1,_2,_6,_30)
Layout b = group<0,2>(a);                 // ((_2,_3),_5,_7):((_1,_2),_6,_30)
Layout c = group<1,3>(b);                 // ((_2,_3),(_5,_7)):((_1,_2),(_6,_30))
Layout f = flatten(b);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
Layout e = flatten(c);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
```

## 总结

Layouts are functions from integers to integers.

## reference

1. https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md
2. https://zhuanlan.zhihu.com/p/661182311
3. https://www.cs.utexas.edu/~flame/BLISRetreat2023/slides/Thakkar_BLISRetreat2023.pdf
4. https://dl.acm.org/doi/pdf/10.1145/3582016.3582018
