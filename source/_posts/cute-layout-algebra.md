---
title: CuTe Layout Algebra
date: 2026-06-20 17:00:00
tags: [CUTLASS, CuTe, Layout, Algebra, GPU]
categories: [Cutlass学习笔记]
description: 深入解析 CuTe Layout Algebra 的核心概念，涵盖 Layout 的代数运算、Composition、Complement、Division 等操作。
---

# CuTe Layout Algebra

CuTe提供了layout代数来支持layout以不同的方式进行组合。这些代数主要包括composition，product，divide等。

通过product操作可以基于一个简单的layout构建一个复杂的layout。通过divide操作可以对一个layout根据另一个layout进行切分重组。而product和divide都要依赖composition和complement函数。

下面将按照从简单到复杂的顺序依次介绍layout中常用的函数。

## Flatten (展平操作)

对tuple进行展平，将多层tuple展平为一层tuple。

举例：

```text
(5, 4) --> (5, 4)
(5, (2, 3)) --> (5, 2, 3)
((2, 2), (2, 3)) --> (2, 2, 2, 3)
```

## Coalesce (合并操作)

Coalesce是一个基础的layout操作，主要作用是合并layout的维度。如果layout某些维度的shape和stride满足某些要求，coalesce会把这些维度合并成一维。不满足条件的维度会保留。

例如一个layout是(5,4):(1,5)，则经过coalesce后layout会变成20:1。

### 函数用法

Layout coalesce(Layout const& layout)

### 计算过程

首先将layout的shape和stride使用flatten进行展平。假设展平后layout的

以前两维为例，layout的coalesce操作主要可以分为四种情况：

(1,s1):(d0,d1)(1, s_1):(d_0,d_1)(1,s1):(d0,d1)或(s0,1):(d0,d1)(s_0, 1):(d_0,d_1)(s0,1):(d0,d1)。这两种情况下layout虽然是二维的，但是由于有一维shape=1，所以可以看成是1维的，因此可以直接合并成s1:d1s_1:d_1s1:d1或s0:d0s_0:d_0s0:d0。

(s0,s1):(d0,d1)(s_0, s_1):(d_0, d_1)(s0,s1):(d0,d1)，其中d1=s0∗d0d_1 = s_0 * d_0d1=s0∗d0。这种情况下如果按照列主序展开，所有元素的间隔都相同，因此可以把这两维合并成(s0∗s1):d0(s_0*s_1):d_0(s0∗s1):d0。

(s0,s1):(d0,d1)(s_0, s_1):(d_0, d_1)(s0,s1):(d0,d1)。如果以上三种条件都不满足，则需要保留原维度的shape和stride，即合并后还是(s0,s1):(d0,d1)(s_0, s_1):(d_0, d_1)(s0,s1):(d0,d1)。

上面只描述了前两维的处理方法，前两维处理完成后需要继续循环遍历后面的维度，直到所有维度处理完毕。

**Python伪代码**

下面是python实现的coalesce核心代码，详见：[https://github.com/NVIDIA/cutlass/blob/v3.8.0/python/pycute/layout.py#L137](https://github.com/NVIDIA/cutlass/blob/v3.8.0/python/pycute/layout.py#L137).

```python
  result_shape  = [1]
  result_stride = [0]
  for (shape,stride) in zip(flatten(layout.shape),flatten(layout.stride)):
    # skip their shape-1s
    if shape == 1:
      continue
    # replace our shape-1 with anything
    elif result_shape[-1] == 1:
      result_shape[-1]  = shape
      result_stride[-1] = stride
    # merge modes if the shape*stride match
    elif result_shape[-1] * result_stride[-1] == stride:
      result_shape[-1] = result_shape[-1] * shape
    # append a new mode
    else:
      result_shape.append(shape)
      result_stride.append(stride)
```

**举例：**

```python
(5, 4):(1, 5) --> 20:1
(5, 4):(2, 10) --> 20:2
(5, 4):(4, 1) --> (5, 4):(4, 1)
(5, (2, 3)):(1, (5, 10)) --> 30:1
(5, (2, 3)):(2, (10, 5)) --> (10, 3):(2, 5)
((2, 2), (2, 3)):((1, 2), (4, 8)) --> 24:1
((2, 2), (2, 3)):((2, 4), (8, 16)) --> 24:2
((2, 2), (2, 3)):((2, 4), (4, 8)) --> (4, 6):(2, 4)
((2, 2), (2, 3)):((2, 2), (8, 16)) --> (2, 2, 6):(2, 2, 8)
((2, 2), (2, 3)):((2, 1), (8, 4)) --> (2, 2, 2, 3):(2, 1, 8, 4)
```

### By-mode Coalesce (按维度合并)

coalesce默认是对layout的全部维度进行处理，如果我们只想处理layout的特定维度，可以传入一个表示维度位置的元组。

Layout coalesce(Layout const& layout, IntTuple const& trg_profile)

例如

layout a = (2,(1,6)):(1,(6,2))

coalesce(a)的结果是12:1。

coalesce(a, (1,1))会分别处理2:1和(1,6):(6,2)，结果是(2,6):(1,2)。

coalesce(a, (1,(1,1)))会分别处理2:1,1:6和6:2，结果是(2,(1,6)):(1,(0,2))。

这里(1,1)只用于表示形状，可以是任意数值。

```python
coalesce with (1,1)
(5, 4):(1, 5) --> (5, 4):(1, 5)
(5, (2, 3)):(1, (5, 10)) --> (5, 6):(1, 5)
(5, (2, 3)):(2, (10, 5)) --> (5, (2, 3)):(2, (10, 5))
((2, 2), (2, 3)):((1, 2), (4, 8)) --> (4, 6):(1, 4)
((2, 2), (2, 3)):((2, 2), (8, 16)) --> ((2, 2), 6):((2, 2), 8)
((2, 2), (2, 3)):((2, 1), (8, 4)) --> ((2, 2), (2, 3)):((2, 1), (8, 4))
```

## Composition (组合操作)

对于两个函数f(x)和g(x)，f(g(x))称为他们的函数组合（composition），函数组合的详细定义见附录1。

layout可以看成是逻辑空间的坐标到物理空间索引的映射函数，因此layout之间也可以定义函数的组合。

offset=Layout(coordinate)offset = Layout(coordinate)offset=Layout(coordinate)

layout的组合是cute的核心概念，用于所有的高阶layout操作。

Layout composition(LayoutA const& layout_a, LayoutB const& layout_b)

举个例子：

```text
Functional composition, R := A o B
R(c) := (A o B)(c) := A(B(c))

Example
A = (6,2):(8,2)
B = (4,3):(3,1)

R( 0) = A(B( 0)) = A(B(0,0)) = A( 0) = A(0,0) =  0
R( 1) = A(B( 1)) = A(B(1,0)) = A( 3) = A(3,0) = 24
R( 2) = A(B( 2)) = A(B(2,0)) = A( 6) = A(0,1) =  2
R( 3) = A(B( 3)) = A(B(3,0)) = A( 9) = A(3,1) = 26
R( 4) = A(B( 4)) = A(B(0,1)) = A( 1) = A(1,0) =  8
R( 5) = A(B( 5)) = A(B(1,1)) = A( 4) = A(4,0) = 32
R( 6) = A(B( 6)) = A(B(2,1)) = A( 7) = A(1,1) = 10
R( 7) = A(B( 7)) = A(B(3,1)) = A(10) = A(4,1) = 34
R( 8) = A(B( 8)) = A(B(0,2)) = A( 2) = A(2,0) = 16
R( 9) = A(B( 9)) = A(B(1,2)) = A( 5) = A(5,0) = 40
R(10) = A(B(10)) = A(B(2,2)) = A( 8) = A(2,1) = 18
R(11) = A(B(11)) = A(B(3,2)) = A(11) = A(5,1) = 42

R = ((2,2),3):((24,2),8)
```

在上面的例子中，B的每一个坐标都会映射到一个索引，而B的映射结果作为A的坐标又得到了新的索引。而从B到A的过程可以用一个新的layout R来表示，R就是A和B组合的结果。不难看到，B的输入和R的输入是等价的。

用print_latex分别打印出A，B和R的layout，如下图所示。

![](/assets/cute-layout-algebra/image.png)

![](/assets/cute-layout-algebra/image_1.png)

![](/assets/cute-layout-algebra/image_2.png)

layout A       o                              layout B                =                              layout R

可以看到，A和B的组合就相当于把A中的元素按照B的布局重新排列。

### Composition计算过程

首先有两个定义

B=(B0,B1,...)B = (B_0, B_1, ...)B=(B0,B1,...)。一个layout可以表达为若干个sublayout的拼接结果。

A∘B=A∘(B0,B1,...)=(A∘B0,A∘B1,...)A ∘ B = A ∘ (B_0, B_1, ...) = (A ∘ B_0, A ∘ B_1, ...)A∘B=A∘(B0,B1,...)=(A∘B0,A∘B1,...)。当布局B是单射（injective）时，组合操作满足左分配律。

不失一般性，我们假设B是一个具有整数形状和步幅的layout，即。假设A是一个经过展平且合并过的layout（flattened，coalesce layout）。

当A是整数layout时，A和B的组合结果很简单：

R=A∘B=a:b∘s:d=s:(b×d)R=A∘B=a:b∘s:d=s:(b×d)R=A∘B=a:b∘s:d=s:(b×d)

但是当A是个多维layout时，计算过程稍微有点复杂，主要分为两步。

1. 从A中每隔d个元素取一个元素。
1. 保留第一步中取到元素的前s个。

在cute中，通过形状除法和形状取余实现这两步过程。

#### 形状除法（Shape Division）

为了计算每隔d个元素取一个元素，cute定义了形状除法的操作。

代码示例：

```cpp
void shape_div(int* shapeA, int N, int& strideB) {
   for (int i = 0; i < N; ++i) {
      assert(shapeA[i] %   strideB == 0 or
               strideB % shapeA[i] == 0);
      int new_shape  = ceil_div(shapeA[i], strideB);
      int new_stride = ceil_div(strideB, shapeA[i]);
      shapeA[i] = new_shape;
      strideB   = new_stride;
   }
}
```

这个函数通过B的步幅strideB来从shapeA中获取组合后的形状。其中shapeA是layoutA的形状，N是shapeA的维度，strideB是layoutB的步幅。

以shapeA = (6,4)，layoutB = 4:3为例。

当i = 0时，

new_shape = ceil_div(shapeA[0], strideB) = ceil_div(6, 3) = 2； // 这种情况下shapeA[0]能被strideB整除。如果shapeA[0]小于strideB时说明shapeA的第一维不够strideB分的，需要到下一维处理。这种情况后面再说。

new_stride = ceil_div(strideB，shapeA[0]) = ceil_div(3, 6) = 1； // 由于shapeA的第一维能被strideB整除，所以第二维对应的stride是1。需要注意这里的stride与layoutA的stride相乘后就是最终组合结果的stride。

shapeA[0] = 2;  // 更新A的shape

strideB = 1;

当i = 1时，

new_shape = ceil_div(shapeA[1], strideB) = ceil_div(4, 1) = 4；

new_stride = ceil_div(strideB，shapeA[1]) = ceil_div(1, 4) = 1；

shapeA[1] = 4;

strideB = 1;

计算结束，此时shapeA从(6,4)变成了(2,4)

下图表示shapeA的变化过程，深色部分就是计算后的shapeA对应的新形状。

![](/assets/cute-layout-algebra/image_3.png)

上面的例子是shapeA的维度能被strideB整除的情况，当shapeA = (6,4)，layoutB = 4:12时计算结果会有所不同。

当i = 0时，

new_shape = ceil_div(shapeA[0], strideB) = ceil_div(6, 12) = 1；// 这种情况说明A的当前维度不够分的。

new_stride = ceil_div(strideB，shapeA[0]) = ceil_div(12, 6) = 2；// A的第一个列有6个元素，需要再来一列才够12，所以第二个维度的stride就变成了2。

shapeA[0] = 1; // 更新shapeA[0]

strideB = 2；   // 更新下一个维度对应的stride

当i = 1时，

new_shape = ceil_div(shapeA[1], strideB) = ceil_div(4, 2) = 2；// 这种情况说明A的shape在当前维度可以被strideB分了。

new_stride = ceil_div(strideB，shapeA[1]) = ceil_div(2, 4) = 1；// 既然能够被分了，那后面的维度对应的strideB就等于1了。

shapeA[0] = 3; // 更新shapeA[1]

strideB = 1；   // 更新下一个维度对应的stride

计算结束，此时shapeA从(6,4)变成了(1,2)。

下图表示shapeA的变化过程，深色部分是计算后shapeA的形状。

更多的例子：

```python
(6,2) /  2 => (3,2) # shapeA = (6,2) strideB = 2, A的第一维能被2整除，所以结果是(3,2)
(6,2) /  3 => (2,2)
(6,2) /  6 => (1,2)
(6,2) / 12 => (1,1) # A的第一维不能被12整除，第一维new_shape = ceil_div(6,12) = 1, 第二维对应的strideB更新为ceil_div(12,6) = 2，2可以被A的第二维整除
(3,6,2,8) / 6 => (1,3,2,8)
(3,6,2,8) / 9 => (1,2,2,8) 
(42,16,3) / 2 => (21,16,3)
(42,16,3) / 6 => ( 7,16,3)
```

形状除法可以想象成将一个shapeA分成strideB份，第一维不够分就加上第二维，第二维还不够就在往后加。因此shapeA能被strideB可分的条件是：strideB能被shapeA中的某一维前缀积整除。

cute中通过代码中的assert来控制shape的可除性。

#### 形状取模（Shape Mod）

通过形状除法更新shapeA后，需要根据shapeB保留前s个元素。这一步是通过形状模运算（Shape Mod）得到的。

代码示例

```cpp
void shape_mod(int* shapeA, int N, int& shapeB) {
   for (int i = 0; i < N; ++i) {
      assert(shapeA[i] %    shapeB == 0 or
                shapeB % shapeA[i] == 0);
      int new_shapeA =      min(shapeA[i], shapeB);
      int new_shapeB = ceil_div(shapeB, shapeA[i]);
      shapeA[i] = new_shapeA;
      shapeB    = new_shapeB;
   }
}
```

输入：形状数组shapeA，这里的形状是shape_div处理后得到的形状；维度N和目标保留元素数shapeB

操作：逐维度更新shapeA，保留每个维度不超过shapeB的部分，并更新剩余需要保留的元素数。

可除性条件：与shape_div类似，通过assert控制。

继续以shapeA = (6,4)，layoutB = 4:3为例。

shapeA在进行形状除法后变为(2,4)。shapeB = 4

当i = 0时，

new_shapeA = min(shapeA[0], shapeB) = min(2, 4) = 2；

new_shapeB = ceil_div(shapeB, shapeA[0]) = ceil_div(4, 2) = 2；// 因为shapeA的第一维保留了2个元素，所以后面还需要保留2个元素。

shapeA[0] = 2;

shapeB = 2;

当i = 1时，

new_shapeA = min(shapeA[1], shapeB) = min(4, 2) = 2；

new_shapeB = ceil_div(shapeB, shapeA[1]) = ceil_div(2, 4) = 1；// 这里表示shapeB已经被选择完了，所以后面维度会全部更新为1，然后通过coalesce去掉这些1。

shapeA[0] = 2;

shapeB = 1;

最终shapeA从(2,4)变成了(2,2)

更多的例子：

```cpp
(6,2) %  2 => (2,1)
(6,2) %  3 => (3,1)
(6,2) %  6 => (6,1)
(6,2) % 12 => (6,2)
(3,6,2,8) %  6 => (3,2,1,1)
(3,6,2,8) %  9 => (3,3,1,1)
(1,2,2,8) %  2 => (1,2,1,1)
(1,2,2,8) % 16 => (1,2,2,4)
```

#### 计算stride

得到最终shape后，根据A和B的stride可以计算得到组合后shape对应的stride。从前面形状除法的过程中也不难发现，在更新strideB的同时与A对应的stride相乘就能得到最终结果的stride。

还是以shapeA = (6,4)，layoutB = 4:3为例。假设shapeA对应的stride是(1,6)。

通过形状除法和形状取模可以得到最终的shape是(2,2)。

因为原始shapeA的第一维的strideA是1，所以新的shape的第一维的stride = strideA * strideB = 1 * 3 = 3。

计算第二维时strideB变成了1，所以新shape的第二维stride = strideA * strideB = 6 * 1 = 6。

所以最终shape (2,2)对应的stride为(3,6)。

需要注意的是，每一维组合结束后会对结果做一次合并，因此最终结果是：

 o  = 

#### python伪代码

下面是用python实现的组合的代码。完整代码见[https://github.com/NVIDIA/cutlass/blob/v3.8.0/python/pycute/layout.py#L190](https://github.com/NVIDIA/cutlass/blob/v3.8.0/python/pycute/layout.py#L190)

```python
# Layout composition
def composition(layoutA, layoutB):
    if is_tuple(layoutB.shape):
        # 判断layoutB的shape是否是tuple，是的话拆分处理。
        return make_layout(composition(layoutA, layoutB_i) for layoutB_i in layoutB)

    if layoutB.stride == 0:
        return Layout(layoutB.shape, 0)
    else:
        result_shape = []
        result_stride = []
        rest_shape = layoutB.shape
        rest_stride = layoutB.stride
        for s, d in zip(flatten(layoutA.shape)[:-1], flatten(layoutA.stride)[:-1]):
            s1 = shape_div(s, rest_stride)  # 按照形状除法计算当前维度shape大小，这里的shape_div基本等于ceil_div
            result_shape.append(min(s1, rest_shape)) # 按照形状取余计算需要保留的shape大小
            result_stride.append(rest_stride * d) # 计算当前维度的stride
            rest_shape = shape_div(rest_shape, abs(s1)) # 计算剩余shape大小
            rest_stride = shape_div(rest_stride, s) # 计算剩余stride的大小

        result_shape.append(rest_shape)
        result_stride.append(rest_stride * flatten(layoutA.stride)[-1])

        # 因为是对layoutB的单个维度处理的，因此最终结果会做一个coalesce简化维度
        return coalesce(Layout(tuple(result_shape), tuple(result_stride)))
```

#### Example 1

20:2∘(5,4):(4,1)20:2  ∘  (5,4):(4,1)20:2∘(5,4):(4,1)

20:2∘(5,4):(4,1)=20:2∘(5:4,4:1)=(20:2∘5:4,20:2∘4:1)=(5:8,4:2)=(5,4):(8,2) \begin{equation}
\begin{split}   20:2  ∘  (5,4):(4,1) &=20:2 ∘ (5:4,4:1)\\
      &= (20:2  ∘  5:4, 20:2  ∘  4:1) \\
     &= (5:8, 4:2) \\
&=(5,4):(8,2)
\end{split}
\end{equation}20:2∘(5,4):(4,1)=20:2∘(5:4,4:1)=(20:2∘5:4,20:2∘4:1)=(5:8,4:2)=(5,4):(8,2)

#### Example 2

(10,2):(16,4)∘(5,4):(1,5)(10,2):(16,4)  ∘  (5,4):(1,5)(10,2):(16,4)∘(5,4):(1,5)

(10,2):(16,4)∘(5,4):(1,5)=((10,2):(16,4)∘(5:1,4:5))=((10,2):(16,4)∘5:1,(10,2):(16,4)∘4:5)=(5:16,(2,2):(80,4))=(5,(2,2)):(16,(80,4)) \begin{equation}
\begin{split}   (10,2):(16,4)  ∘  (5,4):(1,5) &=((10,2):(16,4) ∘ (5:1, 4:5))\\
      &= ((10,2):(16,4) ∘ 5:1, (10,2):(16,4) ∘ 4:5) \\
     &= (5:16, (2,2):(80,4)) \\
&=(5,(2,2)):(16,(80,4))
\end{split}
\end{equation}(10,2):(16,4)∘(5,4):(1,5)=((10,2):(16,4)∘(5:1,4:5))=((10,2):(16,4)∘5:1,(10,2):(16,4)∘4:5)=(5:16,(2,2):(80,4))=(5,(2,2)):(16,(80,4))

这个组合可以描述为将layout (10,2):(16,4) 重新组合为一个5*4的列主序矩阵。

首先根据上面的规则，(5,4):(1,5)可以拆分为(5:1,4:5)。

因此

首先计算(10,2):(16,4) ∘ 5:1，根据上面的计算方法得到(5,1):(16,4)，coalesce后变成5:16。

然后计算(10,2):(16,4) ∘ 4:5，(10,2) / 5 = (2,2)，(2,2) % 4 = (2,2)，stride = (16*5, 4*1) = (80,4)，coalesce后得到(2,2):(80,4)。

最后将5:16和(2,2):(80,4)拼接一起得到(5:16, (2,2):(80,4)) = (5,(2,2))):(16,(80,4))。

在cute中，上面的shape和stride如果用静态整数表示，组合结果为(5,(2,2)):(16,(80,4))，如果用动态整数表示，组合结果为((5,1),(2,2)):((16,4),(80,4))。

这两种结果在数学上是等价的。当使用动态整数时，由于一些限制没有对结果做coalesce。

下图展示了layoutA和layoutB以及他们组合结果layoutC的布局。

o   = 

layoutA         o                                layoutB                  =                          layoutC

### By-mode Composition （按维度组合）

与按维度进行coalesce一样，composition也可以按维度操作。

当在composition函数中传入的第二个参数是Layout时，会按照普通的组合进行。当传进去的是一个Layout元组时，就可以按维度进行组合。cute中将这种layout元组称为Tiler。

举例：

```cpp
// (12,(4,8)):(59,(13,1))
auto a = make_layout(make_shape (12,make_shape ( 4,8)),
                     make_stride(59,make_stride(13,1)));
// <3:4, 8:2>
auto tiler = make_tile(Layout<_3,_4>{},  // Apply 3:4 to mode-0
                       Layout<_8,_2>{}); // Apply 8:2 to mode-1

// (_3,(2,4)):(236,(26,1))
auto result = composition(a, tiler);
// Identical to
auto same_r = make_layout(composition(layout<0>(a), get<0>(tiler)),
                          composition(layout<1>(a), get<1>(tiler)));
```

在上面的例子中，layout a = (12,(4,8)):(59,(13,1))，tiler是由3:4和8:2组成的元组。因此composition(a, tiler)就相当于12:59 ∘ 3:4和(4,8):(13,1) ∘ 8:2。结果为(3,(2,4)):(236,(26,1))，即下图中灰色部分。

这一过程相当于对layout a按照间隔为4取3行，间隔为2取8列。

(12,(4,8)):(59,(13,1))∘⟨(3:4),(8:2)⟩=(12:59∘3:4,(4,8):(13,1)∘8:2)=(3:236,(2,4):(26,1))=(3,(2,4)):(236,(26,1)) \begin{equation}
\begin{split}   (12,(4,8)):(59,(13,1)) ∘  ⟨(3:4),(8:2)⟩ &=(12:59 ∘ 3:4,(4,8):(13,1) ∘ 8:2)\\
      &= (3:236,(2,4):(26,1)) \\
     &= (3,(2,4)):(236,(26,1))
\end{split}
\end{equation}(12,(4,8)):(59,(13,1))∘⟨(3:4),(8:2)⟩=(12:59∘3:4,(4,8):(13,1)∘8:2)=(3:236,(2,4):(26,1))=(3,(2,4)):(236,(26,1))

如果第二个参数是一个shape元组，也会按照tiler处理。因为一个shape可以看作一个stride=1的layout。

```cpp
// (12,(4,8)):(59,(13,1))
auto a = make_layout(make_shape (12,make_shape ( 4,8)),
                     make_stride(59,make_stride(13,1)));
// (8, 3)
auto tiler = make_shape(Int<3>{}, Int<8>{});
// Equivalent to <3:1, 8:1>
// auto tiler = make_tile(Layout<_3,_1>{},  // Apply 3:1 to mode-0
//                        Layout<_8,_1>{}); // Apply 8:1 to mode-1

// (_3,(4,2)):(59,(13,1))
auto result = composition(a, tiler);
```

继续以(12,(4,8)):(59,(13,1))为例，shape (3,8)可以看作tiler (3:1,8:1)，因此组合结果如下图灰色部分所示。这个结果就相当于在layout a中取一个大小为3行4列的数据块(Tile)。

Tile这个概念在后面会反复提及。一个Tile可以理解为具有一定布局的数据块。

这个概念经常用在矩阵分块中。比如一个32*32的矩阵按照8*8的大小可以分为16个tile，每个tile的大小就是8*8。根据tile的layout不同，每个tile分到的数据就不一样，tile与tile之间的layout（也就是后面提到的补集）也不一样。

## Complement

CuTe中补集（Complement）操作用于**找到未被原布局选中的元素在内存中的排列方式**。其核心目标是为原布局A相对于形状M生成一个互补布局R。

Layout complement(LayoutA const& layout_a, Shape const& cotarget)

补集操作通过函数 complement(LayoutA, Shape M) 实现，满足以下性质：

1. 大小约束：

补集布局 R 的大小（size）和陪域大小（cosize）受目标形状 M大小的限制。

1. 有序性：

补集布局R是有序的。R的步幅（stride）必须是递增且正数的，保证其唯一性。

1. 互斥性：

补集布局 R 的陪域（codomains）与原布局 A 的陪域互不相交。布局R试图“补全”布局A的陪域。

上述定义翻译自原文档，可能不太严谨。

只看定义理解起来有些抽象。根据对示例和代码的理解，个人觉得：对于一个layout A，如果存在一个layout R，当把A按照R排列后得到的结果能完全覆盖M空间的所有元素，就称layout R是layout A在M下的补集。

换句话说就是：一个大矩阵M，分成很多块（Tile），每个块的布局是layout A，块与块之间的布局就是layout A在大矩阵上的补集R。

### 计算过程

总的来说，计算一个layout的补集可以分为2步：第一步：填充；第二步：复制。

1. 填充；

如果layout A的stride不是1的话，就表明layoutA的codomain空间中有些元素没有覆盖到。比如layout 4:2的stride=2，第1，3，5位置处的元素没有被该layout覆盖。这些没被覆盖的元素就是“窟窿”，需要进行填充。

此时存在layout R1，将layout A按照R1进行排列，得到的新的layout就可以完全覆盖到layout A的codomain中的元素。

对于layout 4:2，layout R1为2:1。因为4:2对应[0,2,4,6]，把[0,2,4,6]看成一个元素按照shape=2，stride=1排列就可以得到[[0,2,4,6],[1,3,5,7]]，此时就可以覆盖0-7中的所有元素。

1. 复制

当通过layout R1处理后，layout A能覆盖的所有元素为cosize(layout A)。如果小于M的大小，就需要根据layout R2对layoutA进行复制，直到复制后的layout能完全覆盖M的大小。

继续以layout 4:2为例，经过layout R1的处理，其能覆盖的范围为8，当M等于24时，需要将其复制3次才能覆盖M，所以layout R2的shape=ceil_div(24,8)=3，stride=8，即3:8。

所以layout A在M下的补集就是R = (R1，R2)。4:2在M=24下的补集就是(2,3):(1,8)。

下图表示complement(4:2, 24)的结果。灰色是layoutA，蓝色是对layout的填充，红色和绿色是复制的结果。

### Python代码

详见[https://github.com/NVIDIA/cutlass/blob/v3.8.0/python/pycute/layout.py#L223](https://github.com/NVIDIA/cutlass/blob/v3.8.0/python/pycute/layout.py#L223)

```python
# Layout complement
def complement(layout, max_idx=1):
  if is_int(layout):
    return complement(Layout(layout))

  result_shape  = []
  result_stride = []
  current_idx = 1

  sorted_DS = sorted(zip(flatten(layout.stride), flatten(layout.shape)))
  for (stride, shape) in sorted_DS:
    if stride == 0 or shape == 1:
      continue

    in_bound = current_idx <= shape * stride
    # To support symbolic value which can't be evaluated now
    assert (type(in_bound) is not bool) or in_bound

    result_shape.append(stride // current_idx)
    result_stride.append(current_idx)
    current_idx = shape * stride

  result_shape.append((max_idx + current_idx - 1) // current_idx)  # ceil_div
  result_stride.append(current_idx)

  return coalesce(Layout(tuple(result_shape), tuple(result_stride)))
```

### Examples

1. complement(4:1, 24) = 6:4

布局 4:1的stride已经是1了，所以不需要对其进行填充。但是此时该布局只覆盖了4个元素，距离24还需要复制ceil_div(24,4) = 6次，所以补集R等于6:4。stride=4是因为4:1占了4个元素。

complement(6:4, 24) = 4:1

布局6:4的shape=6，stride=4。stride不是1，所以需要先对其进行填充。由于stride=4，所以补集R在该维度的shape=4，stride=1。填充完毕后，6*4=24，已经覆盖了M的范围，所以就不需要复制了。因此6:4相对于24的补集是4:1。

complement((4,6):(1,4), 24) = 1:0

布局(4,6):(1,4)的stride=1且cosize=24，已经覆盖了24的范围，所以不存在补集。

complement((2,4):(1,6), 24) = 3:2

首先处理第一维。第一维的stride=1，shape=2。因为是连续的，所以不需要填充。再处理第二维，stride=6，shape=4。因为第一维的coszie=2，小于6，说明在第二维上不连续，需要填充6/2=3个元素，因此补集在第二维的shape=3，stride=2。

此时layout的stride=1且cosize=24，所以不需要复制了。因为在第一维上没有补集，所以补集就是3:2。

complement((2,2):(1,6), 24) = (3,2):(2,12)

首先处理第一维。因为第一维是连续的，所以不需要填充。第二维上不连续，需要补充3个元素，每个元素的stride=2，所以补集在第二维上的shape=3，stride=2。

此时虽然layout已经连续，但是cosize=12，所以还需要复制一次。因此补集在第三维上的shape=24/12=2，stride=12。最终，补集layout=(3,2):(2,12)。

下图表示complement((2,2):(1,6), 24)的结果。根据图片也可以理解为一个6行4列的矩阵，被layout = (2,2):(1,6)的tile分，可以分为6个tile，这6个tile之间的layout是(3,2):(2,12)。

## Division (Tiling) 除法操作

前面提到通过tile可以对矩阵进行分块，但是每个tile分到的数据可能并不连续，因此cute中定义了logical_divide，将一个tile包含的数据重新组合到一起。

CuTe中定义了Layout的除法操作logical_divide(A, B)，通过logical_divide把一个布局划分为两部分，一部分是由布局 B 指向的元素（通过组合操作 A ∘ B 实现）；另一部分是未被 B 指向的元素（通过补集 B* 表示）。

### 计算方法

用公式表示为：

A⊘B=A∘(B,B∗)A⊘B=A∘(B,B∗)A⊘B=A∘(B,B∗)

其中 B* 是 B 在 A 大小范围内的补集。

所以logical_divide主要有三个操作组成：

1. 组合（Composition）：将布局 A 与目标布局结合。
1. 补集（Complement）：生成未被 B 覆盖的部分 B*。
1. 拼接（Concatenation）：将 B 和 B* 拼接成一个新布局。

代码实现：

```cpp
template <class LShape, class LStride,
          class TShape, class TStride>
auto logical_divide(Layout<LShape,LStride> const& layout,
                    Layout<TShape,TStride> const& tiler)
{
  return composition(layout, make_layout(tiler, complement(tiler, size(layout))));
}
```

### 计算过程

Division操作主要用于对矩阵进行分块处理。对于一个布局为layout A的矩阵，我们根据layout B对矩阵进行分块。logical_divide(A, B)的结果就是把矩阵按照layout B分到的元素放到一起。

举例：

layout A = (4,6):(1,4)

layout B = (2,2):(2,4)

logical_divide(A, B) = ((2,2),(2,3)):((2,4),(1,8))

layout A的形状如下：

layoutB的形状如下：

首先通过composition操作获取layout A中被layout B选中的元素的布局。如下图所示，可以看到被B选中的元素在A中并不连续。

计算layout A与layout B补集的组合。

这一步先计算layout B相对layout A的补集。如下图所示，layout A被layout B分成6块，用不同的颜色表示。块与块之间的layout是(2,3):(1,8)，也就是layout B的补集。有了补集后再与A进行组合。

将前面计算的结果进行拼接。把layout B对应元素的布局放在第一维，块与块的布局放在第二维得到最终的除法结果。从下图可以看到，通过logical_divide，layout A中的元素按照layout B分块的结果重新排列了。

### By-mode Logical Divide (按维度除法)

logical_divide也支持按维度计算。

以一个二维布局A = (9,(4,8)):(59,(13,1))为例，如果在第一维（列方向）上按照3:3计算，第二维（行方向）上按照(2,4):(1,8)计算，整个计算过程可以写成：

A⊘B=(9,(4,8)):(59,(13,1))⊘⟨3:3,(2,4):(1,8)⟩=9:59⊘3:3,(4,8):(13,1)⊘(2,4):(1,8)=(3,3):(177,59),((2,4),(2,2)):((13,2),(26,1))=((3,3),((2,4),(2,2))):((177,59),((13,2),(26,1))) \begin{equation}
\begin{split}   A⊘B &=(9,(4,8)):(59,(13,1))⊘⟨3:3, (2,4):(1,8)⟩\\
      &= 9:59⊘3:3,  (4,8):(13,1)⊘(2,4):(1,8) \\
     &= (3, 3):(177, 59),((2, 4), (2, 2)):((13, 2), (26, 1)) \\
&=((3, 3),((2, 4), (2, 2))):((177, 59),((13, 2), (26, 1)))

\end{split}
\end{equation}A⊘B=(9,(4,8)):(59,(13,1))⊘⟨3:3,(2,4):(1,8)⟩=9:59⊘3:3,(4,8):(13,1)⊘(2,4):(1,8)=(3,3):(177,59),((2,4),(2,2)):((13,2),(26,1))=((3,3),((2,4),(2,2))):((177,59),((13,2),(26,1)))

具体计算过程如下图所示：

首先按照layout B的布局对layout A分块。列方向上9个元素分成三块，每块3个元素。行方向上32个元素分成4块，每块8个元素。相同块中的元素用相同的颜色表示。

通过logical_divide操作把相同的块对应的元素放到一起。由于是按维度计算的，所以在列方向上相同块中的三个元素按顺序排列到一起。行方向上相同块中的8个元素按顺序排列到一起，最终结果如下图所示。

### Zipped, Tiled, Flat Divides

除了logical_divide，CuTe还提供了zipped_divide，tiled_divide和flat_divide。这四种除法的计算过程相同，区别在于最后对分块的元素排列不同。

```cpp
Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)
zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))
tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...)
flat_divide    : (TileM, TileN, RestM, RestN, L, ...)
```

其中，TileM可以理解为layout B对应的tile的行数，RestM是layout A的行数被layout B分成了几份。同理TileN是layout B的列数，RestN是行方向上layout B的数量。

以上图为例：

logical_divide是按照((TileM,RestM), (TileN,RestN))的顺序排列最终的结果，即((3,3),((2,4),(2,2))):((177,59),((13,2),(26,1)))。

zipped_divide则是按照((TileM,TileN), (RestM,RestN))排列，结果是((3,(2,4)),(3,(2,2))):((177,(13,2)),(59,(26,1)))。

同样的，

tiled_divide = ((3,(2,4)),3,(2,2)):((177,(13,2)),59,(26,1))；

flat_divide = (3,(2,4),3,(2,2)):(177,(13,2),59,(26,1))

zipped_divide会变成下图的样子。可以看到，相同块的元素全部在同一维度中。

tiled_divide和flat_divide的结果与zipped_divide类似，只是维度变成了多维的。

## Product (Tiling) 乘法操作

CuTe定义了layout与layout之间的乘法，logical_product(A, B)。

logical_product的作用是把layout A按照layout B的布局进行复制。logical_product的计算公式为：

A⊗B=(A,A∗∘B)A⊗B=(A,A^∗∘B)A⊗B=(A,A∗∘B)

```cpp
template <class LShape, class LStride,
          class TShape, class TStride>
auto logical_product(Layout<LShape,LStride> const& layout,
                     Layout<TShape,TStride> const& tiler)
{
  return make_layout(layout, composition(complement(layout, size(layout)*cosize(tiler)), tiler));
}
```

### 计算过程

从公式理解，product的含义就是把layout A按照layout B进行复制。layout A是计算结果的第一维，layout A的补集表示A需要按照什么布局进行复制。

以layout A = (2,2):(4,1)和layout B = (4,2):(2,1)相乘为例

其中size(A) = 4, coszie(B) = 8。

首先计算A在size(A)*cosize(B) = 32空间下的补集。通过上面补集的计算过程不难算出，A的补集是(2,4):(2,8)。

此时layout A在32空间下的完整表示如下图。

然后把补集按照layout B的布局排列。

与layout A进行组合得到最后结果。

### By-mode Logical Product (按维度乘)

logical_product也可以按照维度分别计算。

如下图所示，layout A (2,5):(5,1)与layout元组B <3:5, 4:6>相乘的结果如下所示。从图中可以看到，layout A按行扩大3倍，按列扩大4倍。

A⊗B=((2,5):(5,1))⊗⟨3:5,4:6⟩=(2:5⊗3:5),(5:1⊗4:6)=(2,3):(5,10),(5,4):(1,30)=((2,3),(5,4)):((5,10),(1,30)) \begin{equation}
\begin{split}   A \otimes B &=((2,5):(5,1))\otimes ⟨3:5, 4:6⟩\\
      &= (2:5\otimes 3:5),  (5:1\otimes 4:6) \\
     &= (2,3):(5,10),(5,4):(1,30) \\
&=((2,3),(5,4)):((5,10),(1,30))

\end{split}
\end{equation}A⊗B=((2,5):(5,1))⊗⟨3:5,4:6⟩=(2:5⊗3:5),(5:1⊗4:6)=(2,3):(5,10),(5,4):(1,30)=((2,3),(5,4)):((5,10),(1,30))

但是这种方法并不推荐使用，原因是不能直观的通过A和B的layout判断出计算结果是什么样子。

因此cute提供了blocked_product和raked_product这两个函数。

blocked_product的计算结果与logical_product类似，但是输入的第二个参数不同。

blocked_product的第二个参数是对layout A复制的数量。在下图中，第二个参数是(3,4):(1,3)，代表把layout A在第一维复制3次，在第二维复制4次。使用这种方法可以更直观的理解两个layout相乘之后的结果。

raked_product的计算结果与blocked_product略有不同，他不是像blocked_product一样按照(rowA, rowB),(colA, colB)这种顺序组合的，而是按照(rowB, rowA),(colB, colA)进行组合的，计算结果如下图所示。

### Zipped and Tiled Products

与zipped_divide和tiled_divide类似，cute也提供了zipped_product，tiled_product和flat_product。与logical_product的区别在于最终结果的组合维度不同。

```cpp
Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_product : ((M,TileM), (N,TileN), L, ...)
zipped_product  : ((M,N), (TileM,TileN,L,...))
tiled_product   : ((M,N), TileM, TileN, L, ...)
flat_product    : (M, N, TileM, TileN, L, ...)
```

以layout A (2,5):(5,1)与layout元组B <3:5, 4:6>相乘为例：

logical_product是按照((M,TileM), (N,TileN))的顺序排列最终的结果，即((2,3),(5,4)):((5,10),(1,30))。

zipped_product则是按照((M,N), (TileM,TileN))排列，结果是((2,5),(3,4)):((5,1),(10,30))。

同样的，

tiled_product= ((2,5),3,4):((5,1),10,30)。

flat_product= (2,5,3,4):(5,1,10,30)。

下图是zipped_product的结果。可以看到，相同块的元素全部在同一维度中。

# 附录1：函数的组合

**定义来自deepseek**

