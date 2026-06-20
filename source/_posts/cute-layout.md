---
title: CuTe Layout
date: 2026-06-20 16:00:00
tags: [CUTLASS, CuTe, Layout, GPU]
categories: [Cutlass学习笔记]
description: 深入解析 CuTe Layout 的核心概念，涵盖 Layout 的组成、Stride、Profile、Tile、Partition 等操作。
---

# CuTe_Layout

我们对于数据的描述通常是高维的，但是在计算机的内存中，数据都是按一维线性地址排布的。因此我们需要通过一种方法将数据的逻辑空间和物理空间联系起来。

目前常用的方法是使用shape和stride描述数据在逻辑空间和物理空间的联系，即shape用于描述形状，stride用于描述中元素之间的间隔。

对于按顺序在物理存储中排列的8个元素，我们可以使用shape = [2, 4]，stride = [4, 1]表示。stride=4代表在第一维（列方向）上相邻元素间隔为4，stride=1代表行方向上相邻元素间隔为1。

![](/assets/cute-layout/image.png)

也可以使用shape = [2, 4]，stride = [1, 2]表示。

![](/assets/cute-layout/image_1.png)

但是这种方法的一个局限性是，一个维度只能有一个stride，也就是说同一个维度的数据间隔必须是相同的。

如果一个shape中的元素按照下面的方式排列，就没有办法只使用同一个stride进行表示了。

![](/assets/cute-layout/image_2.png)

所以为了能够表示更复杂的数据逻辑，cute在shape和stride的基础上引入了层级多维布局（hierarchically multidimensional layouts）。

# CuTe Layouts

CuTe Layout可以理解为是一种描述逻辑空间和物理空间关系的函数，通过layout将数据从逻辑空间映射到物理空间。与前面介绍的基本相同，layout由shape和stride组成，只不过这里的shape和stride可以使用元组（Tuple）类型来表达更复杂的结构。

介绍Layout前首先介绍下cute中的**Integers**和**Tuple**。

## Integers

cute中的整数分为两种：dynamic (run-time) 和 static (compile-time) integers。

dynamic integers也就是运行时整数，与常见的int or size_t or uint16_t类型相同，是在运行时确定的。

static integers是静态整数，也可以称为编译时整数。这种数是在编译时就确定的，可以在编译时进行计算。比如矩阵分块的大小，这种值一般在编译时就可以确定，因此可以使用静态整数。

在cute中，静态整数的构造方式为Int<Value>{}，其中Int<Value>是静态整数类型cute::C<Value>，加上{}是类型的实例化。

对于常用的数值，如：Int<1>, Int<2>, Int<32>, Int<128>，可以直接使用_1, _2, _32, _128代替。具体可以参考cutlass/include/cute/numeric/integral_constant.hpp。

## Tuple

元组（tuple）是一个有限的、有序的、包含零个或多个元素的列表。cute::tuple 类的行为类似于 std::tuple，但它可以在设备和主机上运行。它对模板参数施加了一些限制，并简化了实现以提高性能和简洁性。

## Layout

CuTe中的shape和stride都是由tuple构成的。而layout是由shape和stride组成的。

cute::Layout<Shape, Stride>

### Layout的构造

Layout可以通过cute::make_layout函数进行构造，构造时整数可以根据需要选择动态和静态。

```cpp
cute::Layout layout1 = make_layout(make_shape(4, 8));  // column-major, shape = [4, 8], stride = [1, 4]

cute::tuple shape = make_shape(_4{}, _8{});
cute::tuple stride = make_stride(_8{}, _1{});
cute::Layout layout2 = make_layout(shape, stride);  // row-major, shape = [4, 8], stride = [8, 1]
```

layout1的shape是[4, 8]，没有指定stride，但是由于cutlass中默认是column-major，所以stride默认是[1, 4]。layout2的shape和layout1相同，但是stride是[8, 1]。

print，print_layout或print_latex函数可以打印layout的形状。通过打印可以看到，layout1是(4,8):(1,4)，layout2是(4,8):(8,1)。其中冒号前面是shape，冒号后面是stride。

通过print_latex打印可以看到两个layout的形状，其中下图的左边是layout1的形状，右边是layout2的形状。根据stride的不同，两个layout中offset的排列也不同。

![](/assets/cute-layout/image_3.png)

![](/assets/cute-layout/image_4.png)

layout1                                                             layout2

shape和stride通过嵌套可以表示更复杂的逻辑结构。

```cpp
cute::Layout layout3 = make_layout(make_shape(4, make_shape(2, 4)), make_stride(2, make_stride(1, 8))); // shape = [4, (2, 4)], stride = [2, (1, 8)]

cute::tuple shape = make_shape(make_shape(2, 2), make_shape(2, 4));
cute::tuple stride = make_stride(make_stride(1, 4), make_stride(2, 8));
cute::Layout layout4 = make_layout(shape, stride); // shape = [(2, 2), (2, 4)], stride = [(1, 4), (2, 8)]
```

通过上面代码构建的layout3的shape = [4, (2, 4)]，stride = [2, (1, 8)]，layout4的shape = [(2, 2), (2, 4)], stride = [(1, 4), (2, 8)]。可以看到，layout3和layout4的shape和stride从整数元组变成了嵌套的元组。

layout3的shape = [4, (2, 4)]代表第一维有4行，第二维有(2, 4)列，这里的(2, 4)可以理解为第二维一共有2 * 4 = 8列，8列又从逻辑上被分为了4组，一组有2列。

stride = [2, (1, 8)]，维度和shape一一对应，stride=2代表第一维中相邻元素在物理空间的间隔是2；(1, 8)对应shape中的(2, 4)，表示shape为2的维度的stride=1，shape=4的维度的stride=8。

layout3 print_latex打印结果如下：

![](/assets/cute-layout/image_5.png)

从图中可以看到，虽然还是4行8列，但是layout中的offset发生了变化。

下面分析layout3是如何得到这种排列的。

从前面可知，layout3是一个4行8列的形状，由于8被分成了(2, 4)，我们使用不同的颜色进行区分，那么一行一共有4种颜色，每种颜色包含两个元素，如下图所示。

![](/assets/cute-layout/image_6.png)

有了形状后再根据stride计算每个位置的offset。第0个位置的offset就是0，第一个维度的stride=2，表示在第一个维度中相邻元素之间相差2，因此第0列的4个元素的offset为[0, 2, 4, 6]。

![](/assets/cute-layout/image_7.png)

第二个维度的stride是(1,8)。(1,8)中的1表示对应维度的元素间隔是1，即相同色块之间的元素间隔是1，因此可以得到第一列元素的offset为[1, 3, 5, 7]。

![](/assets/cute-layout/image_8.png)

stride(1,8)中的第二个维度是8，表示相邻色块之间的间隔是8，因此可以得到第2，4，6列元素的offset。不难发现，他们也同时满足第一维stride=2的要求。

![](/assets/cute-layout/image_9.png)

以此类推，可以推导出其他位置的元素。

![](/assets/cute-layout/image_10.png)

同样的，我们可以得到layout4的元素布局。layout4的shape是[(2, 2), (2, 4)], stride是 [(1, 4), (2, 8)]。print_latex的打印结果如下：

![](/assets/cute-layout/image_11.png)

下面分析layout4是如何得到这种排列的。

layout4 shape的第一个维度是(2, 2)，表明第一维一共有2*2=4行，每一行被分成2块，每一块有2个元素。第二个维度是(2, 4)，表示第二个维度有2*4=8列，被分成4块，每一块有2个元素。

因此，layout4的形状可以看成下面这样，由2*4个色块组成，每一个色块有2*2个元素。

![](/assets/cute-layout/image_12.png)

然后根据stride排列layout4中的元素。首先，stride的第一个维度是(1, 4)，表示shape的第一个维度的内层维度相邻元素间隔是1，外层维度相邻元素间隔是4。可以得到

![](/assets/cute-layout/image_13.png)

stride的第二个维度是(2, 8)，表示shape的第二个维度的内层维度相邻元素间隔是2，第二个维度的外层维度相邻元素间隔是8。可以得到

![](/assets/cute-layout/image_14.png)

以此类推，得到layout4最终的形状

![](/assets/cute-layout/image_15.png)

总的来说，可以把这种嵌套的shape理解成((维度1内层大小，维度1外层1大小，维度1外层2大小 。。。)， (维度2内层大小，维度2外层1大小，维度2外层2大小 。。。))。

通过这种嵌套的shape和stride，cute的layout可以表示更加复杂的数据分布。

### 属性函数

rank(Layout): 获取layout的维度。

depth(Layout): 获取layout的嵌套层数。

Layoutrankdepth210(4)11(2,3)21(2,(2,4))22((2,4),(2,(2,3)))23((3,2),(2,4),(2,3))32

get<I>(Layout): 获取layout中第 I个维度的sub-layout。

```cpp
auto layout = make_layout(make_shape(make_shape(3,2), make_shape(2,4), make_shape(2,3)));
auto sub_layout = get<2>(layout);
print("layout = ");print(layout);print("\n");
print("sub_layout = ");print(sub_layout);print("\n");

// layout = ((3,2),(2,4),(2,3)):((_1,3),(6,12),(48,96))
// sub_layout = (2,3):(48,96)
```

shape(Layout): 获取layout的shape。

stride(Layout): 获取layout的stride。

size(Layout): layout在逻辑空间包含的元素数量，即shape对应的大小。

cosize(Layout): layout在物理空间包含的真实的元素数量。

layout A = (2,2):(2,4)的shape = (2,2)，代表一共有4个元素，因此size(A) = 4。但是实际覆盖的元素是8，因此cosize(A) = 8。

### 访问layout中的元素

通过逻辑空间的坐标可以获取对应位置数据在物理空间的index或者说是offset。

offset=Layout(coordinate)offset = Layout(coordinate)offset=Layout(coordinate)

cute中有两种访问元素的方法，一种是使用具体坐标访问特定的元素位置，另一种是使用下划线"_"进行slice访问，类似与python中的":"。

#### 单个元素的访问

cute有三种坐标形式，1-D坐标，2-D坐标和natural坐标。

以(3,(2,3))为例，使用不同类型的坐标访问layout都会返回对应位置的offset。

![](/assets/cute-layout/image_16.png)

```cpp
auto layout = make_layout(make_shape(3, make_shape(2,3)));  // (3, (2,3))
int offset1 = layout(5);  // 5
int offset2 = layout(make_coord(2,1)); // 5
int offset3 = layout(make_coord(2,make_coord(1,0))); // 5
```

1-D2-DNaturaloffset1-D2-DNaturaloffset0(0,0)(0,(0,0))09(0,3)(0,(1,1))91(1,0)(1,(0,0))110(1,3)(1,(1,1))102(2,0)(2,(0,0))211(2,3)(2,(1,1))113(0,1)(0,(1,0))312(0,4)(0,(0,2))124(1,1)(1,(1,0))413(1,4)(1,(0,2))135(2,1)(2,(1,0))514(2,4)(2,(0,2))146(0,2)(0,(0,1))615(0,5)(0,(1,2))157(1,2)(1,(0,1))716(1,5)(1,(1,2))168(2,2)(2,(0,1))817(2,5)(2,(1,2))17

#### 使用“_”访问元素。

使用"_"可以访问对应位置的多个元素。

![](/assets/cute-layout/image_17.png)

以上图为例，

A(37)：按column-major的顺序数37个数得到49。

A(5,4)：第5行第4列，得到49。

A((1,2),(0,2))：(1,2)的外层维度坐标是2，确定[4,5]行，内层维度坐标是1，确定第5行；(0,2)的外层维度坐标是2，确定[4,5]列，内层维度坐标是0，确定第4列。

A((1,(0,1)),(0,(0,1)))：根据第一维(1,(0,1))最外层维度(0,1)中的1，确定[4,5,6,7]行；(0,1)中的0，确定[4,5]行；最内层的1，确定第5行；同理可确定第4列。

A(_,2)表示第2列的所有元素；

A((_,1),(_,2))：(_,1)的外层维度index是1，确定[2,3]行，内层index是"_"，表示[2,3]行的所有元素。同理可确定[4,5]列的所有元素，行与列交叉部分就是上图蓝框部分数据。

### Layout操作

#### 获取sub-layouts

 layout<I...>

```cpp
Layout a   = Layout<Shape<_4,Shape<_3,_6>>>{}; // (4,(3,6)):(1,(4,12))
Layout a0  = layout<0>(a);                     // 4:1
Layout a1  = layout<1>(a);                     // (3,6):(4,12)
Layout a10 = layout<1,0>(a);                   // 3:4
Layout a11 = layout<1,1>(a);                   // 6:12
```

select<I...>

```cpp
Layout a   = Layout<Shape<_2,_3,_5,_7>>{};     // (2,3,5,7):(1,2,6,30)
Layout a13 = select<1,3>(a);                   // (3,7):(2,30)
Layout a01 = select<0,1,3>(a);                 // (2,3,7):(1,2,30)
Layout a2  = select<2>(a);                     // (5):(6)
```

take<ModeBegin, ModeEnd>

```cpp
Layout a   = Layout<Shape<_2,_3,_5,_7>>{};     // (2,3,5,7):(1,2,6,30)
Layout a13 = take<1,3>(a);                     // (3,5):(2,6)
Layout a14 = take<1,4>(a);                     // (3,5,7):(2,6,30)
// take<1,1> not allowed. Empty layouts not allowed.
```

#### Concatenation

 Layout 可以通过 make_layout 实现concatenation

```cpp
Layout a = Layout<_3,_1>{};                     // 3:1
Layout b = Layout<_4,_3>{};                     // 4:3
Layout row = make_layout(a, b);                 // (3,4):(1,3)
Layout col = make_layout(b, a);                 // (4,3):(3,1)
Layout q   = make_layout(row, col);             // ((3,4),(4,3)):((1,3),(3,1))
Layout aa  = make_layout(a);                    // (3):(1)
Layout aaa = make_layout(aa);                   // ((3)):((1))
Layout d   = make_layout(a, make_layout(a), a); // (3,(3),3):(1,(1),1)
```

也可以通过这三个函数实现 append, prepend, or replace.

```cpp
Layout a = Layout<_3,_1>{};                     // 3:1
Layout b = Layout<_4,_3>{};                     // 4:3
Layout ab = append(a, b);                       // (3,4):(1,3)
Layout ba = prepend(a, b);                      // (4,3):(3,1)
Layout c  = append(ab, ab);                     // (3,4,(3,4)):(1,3,(1,3))
Layout d  = replace<2>(c, b);                   // (3,4,4):(1,3,3)
```

#### Grouping and flattening

通过 group<ModeBegin, ModeEnd> 或 flatten可以对layout进行组合和展开。

```cpp
Layout a = Layout<Shape<_2,_3,_5,_7>>{};  // (_2,_3,_5,_7):(_1,_2,_6,_30)
Layout b = group<0,2>(a);                 // ((_2,_3),_5,_7):((_1,_2),_6,_30)
Layout c = group<1,3>(b);                 // ((_2,_3),(_5,_7)):((_1,_2),(_6,_30))
Layout f = flatten(b);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
Layout e = flatten(c);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
```

## 总结

Layouts are functions from integers to integers.

## reference：

1. https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md
1. https://zhuanlan.zhihu.com/p/661182311
1. https://www.cs.utexas.edu/~flame/BLISRetreat2023/slides/Thakkar_BLISRetreat2023.pdf
1. https://dl.acm.org/doi/pdf/10.1145/3582016.3582018
