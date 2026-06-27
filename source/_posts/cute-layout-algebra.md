---
title: CuTe 学习笔记（二）Layout Algebra（一）
date: 2025-03-20 17:00:00
tags: [CUTLASS, CuTe, Layout, GPU]
categories: [Cutlass 学习笔记]
mathjax: true
description: 文章介绍 CuTe Layout Algebra 的核心概念和具体的代数运算流程，包括 Flatten、Coalesce、Composition、Complement 操作。
---

# CuTe Layout Algebra

CuTe 提供了 layout 代数来支持 layout 以不同的方式进行组合。这些代数主要包括 composition，product，divide 等。

通过 product 操作可以基于一个简单的 layout 构建一个复杂的 layout。通过 divide 操作可以对一个 layout 根据另一个 layout 进行切分重组。而 product 和 divide 都要依赖 composition 和 complement 函数。

本文重点介绍一些基础的 Layout 代数运算函数，包括 `Flatten`, `Coalesce`, `Composition`, `Complement`。

## Flatten (展平操作)

对 tuple 进行展平，将多层 tuple 展平为一层 tuple。

举例：

```text
(5, 4) --> (5, 4)
(5, (2, 3)) --> (5, 2, 3)
((2, 2), (2, 3)) --> (2, 2, 2, 3)
```

## Coalesce (合并操作)

Coalesce 是一个基础的 layout 操作，主要作用是合并 layout 的维度。如果 layout 某些维度的 shape 和 stride 满足某些要求，coalesce 会把这些维度合并成一维。不满足条件的维度会保留。

例如一个 layout 是 (5,4):(1,5)，则经过 coalesce 后 layout 会变成 20:1。

### 函数用法

\begin{equation*}
\text{Layout} = \text{coalesce}(\text{Layout const\& layout})
\end{equation*}

### 计算过程

首先将 layout 的 shape 和 stride 使用 flatten 进行展平。

假设展平后 layout 的

\begin{equation*}
Shape = [s_0, s_1, s_2, \dots, s_n],\ Stride = [d_0, d_1, d_2, \dots, d_n]
\end{equation*}

以前两维为例，layout 的 coalesce 操作主要可以分为四种情况：

- $(1, s_1) : (d_0, d_1)$ 或 $(s_0, 1) : (d_0, d_1)$。这两种情况下 layout 虽然是二维的，但是由于有一维 shape=1，所以可以看成是 1 维的，因此可以直接合并成 $s_1 : d_1$ 或 $s_0 : d_0$。

- $(s_0, s_1) : (d_0, d_1)$，其中 $d_1 = s_0 \times d_0$。这种情况下如果按照 column-major 展开，所有元素的间隔都相同，因此可以把这两维合并成 $(s_0 \times s_1) : d_0$。

- $(s_0, s_1) : (d_0, d_1)$。如果以上三种条件都不满足，需要保留原维度的 shape 和 stride，即合并后还是 $(s_0, s_1) : (d_0, d_1)$。

上面只描述了前两维的处理方法，前两维处理完成后需要继续循环遍历后面的维度，直到所有维度处理完毕。

**Python 伪代码**

下面是 python 实现的 coalesce 核心代码，
详见：[https://github.com/NVIDIA/cutlass/blob/v3.8.0/python/pycute/layout.py#L137](https://github.com/NVIDIA/cutlass/blob/v3.8.0/python/pycute/layout.py#L137).

```python
  result_shape  = [1]
  result_stride = [0]
  for (shape,stride) in zip(flatten(layout.shape),flatten(layout.stride)):
    # skip their shape-1s
    if shape == 1:
      continue
    # replace our shape-1 with anything
    elif result_shape[-1] == 1:
      result_shape[-1]  = shape
      result_stride[-1] = stride
    # merge modes if the shape*stride match
    elif result_shape[-1] * result_stride[-1] == stride:
      result_shape[-1] = result_shape[-1] * shape
    # append a new mode
    else:
      result_shape.append(shape)
      result_stride.append(stride)
```

**举例：**

```python
(5, 4):(1, 5)  -->  20:1
(5, 4):(2, 10)  -->  20:2
(5, 4):(4, 1)  -->  (5, 4):(4, 1)
(5, (2, 3)):(1, (5, 10))  -->  30:1
(5, (2, 3)):(2, (10, 5))  -->  (10, 3):(2, 5)
((2, 2), (2, 3)):((1, 2), (4, 8))  -->  24:1
((2, 2), (2, 3)):((2, 4), (8, 16))  -->  24:2
((2, 2), (2, 3)):((2, 4), (4, 8))  -->  (4, 6):(2, 4)
((2, 2), (2, 3)):((2, 2), (8, 16))  -->  (2, 2, 6):(2, 2, 8)
((2, 2), (2, 3)):((2, 1), (8, 4))  -->  (2, 2, 2, 3):(2, 1, 8, 4)
```

### By-mode Coalesce (按维度合并)

coalesce 默认是对 layout 的全部维度进行处理，如果我们只想处理 layout 的特定维度，可以传入一个表示维度位置的元组。

```
Layout coalesce(Layout const& layout, IntTuple const& trg_profile)
```

例如

layout a = (2,(1,6)):(1,(6,2))

coalesce(a) 的结果是 12:1。

coalesce(a, (1,1)) 会分别处理 2:1 和 (1,6):(6,2)，结果是 (2,6):(1,2)。

coalesce(a, (1,(1,1))) 会分别处理 2:1, 1:6 和 6:2，结果是 (2,(1,6)):(1,(0,2))。

这里 (1,1) 只用于表示形状，可以是任意数值。

```python
coalesce with (1,1)
(5, 4):(1, 5)  -->  (5, 4):(1, 5)
(5, (2, 3)):(1, (5, 10))  -->  (5, 6):(1, 5)
(5, (2, 3)):(2, (10, 5))  -->  (5, (2, 3)):(2, (10, 5))
((2, 2), (2, 3)):((1, 2), (4, 8))  -->  (4, 6):(1, 4)
((2, 2), (2, 3)):((2, 2), (8, 16))  -->  ((2, 2), 6):((2, 2), 8)
((2, 2), (2, 3)):((2, 1), (8, 4))  -->  ((2, 2), (2, 3)):((2, 1), (8, 4))
```

## Composition (组合操作)

对于两个函数 f(x) 和 g(x)，f(g(x)) 称为他们的函数组合（composition）。

Layout 可以看成是逻辑空间的坐标到物理空间索引的映射函数，因此 layout 之间也可以定义函数的组合。

```
offset = Layout(coordinate)
```

layout 的组合是 cute 的核心概念，用于所有的高阶 layout 操作。

```
Layout composition(LayoutA const& layout_a, LayoutB const& layout_b)
```

举个例子：

```text
Functional composition, R := A o B
R(c) := (A o B)(c) := A(B(c))

Example
A = (6,2):(8,2)
B = (4,3):(3,1)

R( 0) = A(B( 0)) = A(B(0,0)) = A( 0) = A(0,0) =  0
R( 1) = A(B( 1)) = A(B(1,0)) = A( 3) = A(3,0) = 24
R( 2) = A(B( 2)) = A(B(2,0)) = A( 6) = A(0,1) =  2
R( 3) = A(B( 3)) = A(B(3,0)) = A( 9) = A(3,1) = 26
R( 4) = A(B( 4)) = A(B(0,1)) = A( 1) = A(1,0) =  8
R( 5) = A(B( 5)) = A(B(1,1)) = A( 4) = A(4,0) = 32
R( 6) = A(B( 6)) = A(B(2,1)) = A( 7) = A(1,1) = 10
R( 7) = A(B( 7)) = A(B(3,1)) = A(10) = A(4,1) = 34
R( 8) = A(B( 8)) = A(B(0,2)) = A( 2) = A(2,0) = 16
R( 9) = A(B( 9)) = A(B(1,2)) = A( 5) = A(5,0) = 40
R(10) = A(B(10)) = A(B(2,2)) = A( 8) = A(2,1) = 18
R(11) = A(B(11)) = A(B(3,2)) = A(11) = A(5,1) = 42

R = ((2,2),3):((24,2),8)
```

在上面的例子中，B 的每一个坐标都会映射到一个索引，而 B 的映射结果作为 A 的坐标又得到了新的索引。而从 B 到 A 的过程可以用一个新的 layout R 来表示，R 就是 A 和 B 组合的结果。不难看到，B 的输入和 R 的输入是等价的。

用 print_latex 分别打印出 A，B 和 R 的 layout，如下图所示。

<div align="center">
        <img src="/assets/cute-layout-algebra/compositionABC.png" width="100%" height="auto" alt="layout1">
        <small>layout A o layout B = layout R</small>
</div>
<br>

可以看到，A 和 B 的组合就相当于把 A 中的元素按照 B 的布局重新排列。

### Composition 计算过程

首先有两个定义：

$B = (B_0, B_1, ...)$。

一个 layout 可以表达为若干个 sublayout 的拼接结果。

$A \circ B = A \circ (B_0, B_1, ...) = (A \circ B_0, A \circ B_1, ...)$。

当布局 B 是单射（injective）时，组合操作满足左分配律。

我们先假设 B 是一个具有整数形状和步幅的 layout，即 $B = s:d$。假设 A 是一个经过展平且合并过的 layout（flattened，coalesce layout）。

当 A 是整数 layout 时，A 和 B 的组合结果如下：

\begin{equation*}
R = A \circ B = a:b \circ s:d = s:(b \times d)
\end{equation*}

但是当 A 是个多维 layout 时，计算过程稍微复杂，主要分为两步。

1. 从 A 中每隔 d 个元素取一个元素。
2. 保留第一步中取到元素的前 s 个。

在 cute 中，通过形状除法和形状取余实现这两步过程。

#### 形状除法（Shape Division）

为了计算每隔 d 个元素取一个元素，cute 定义了形状除法的操作。

代码示例：

```cpp
void shape_div(int* shapeA, int N, int& strideB) {
   for (int i = 0; i < N; ++i) {
      assert(shapeA[i] %   strideB == 0 or
               strideB % shapeA[i] == 0);
      int new_shape  = ceil_div(shapeA[i], strideB);
      int new_stride = ceil_div(strideB, shapeA[i]);
      shapeA[i] = new_shape;
      strideB   = new_stride;
   }
}
```

这个函数通过 B 的步幅 strideB 来从 shapeA 中获取组合后的形状。其中 shapeA 是 layoutA 的形状，N 是 shapeA 的维度，strideB 是 layoutB 的步幅。

以 shapeA = (6,4)，layoutB = 4:3 为例。

当 i = 0 时，

```cpp
// 这种情况下 shapeA[0] 能被 strideB 整除。如果 shapeA[0] 小于 strideB 时说明 shapeA 的第一维不够 strideB 分的，需要到下一维处理。这种情况后面再说。
new_shape = ceil_div(shapeA[0], strideB) = ceil_div(6, 3) = 2；

// 由于 shapeA 的第一维能被 strideB 整除，所以第二维对应的 stride 是 1。需要注意这里的 stride 与 layoutA 的 stride 相乘后就是最终组合结果的 stride。
new_stride = ceil_div(strideB, shapeA[0]) = ceil_div(3, 6) = 1；

shapeA[0] = 2;  // 更新 A 的 shape

strideB = 1;
```

当 i = 1 时，

```cpp
new_shape = ceil_div(shapeA[1], strideB) = ceil_div(4, 1) = 4；

new_stride = ceil_div(strideB, shapeA[1]) = ceil_div(1, 4) = 1；

shapeA[1] = 4;

strideB = 1;
```

计算结束，此时 shapeA 从 (6,4) 变成了 (2,4)

下图表示 shapeA 的变化过程，深色部分就是计算后的 shapeA 对应的新形状。

<div align="center">
        <img src="/assets/cute-layout-algebra/image1.png" width="100%" height="auto" alt="layout1">
        <small>(6,4) / 3 = (2,4)</small>
</div>
<br>

上面的例子是 shapeA 的维度能被 strideB 整除的情况。

当 shapeA = (6,4)，layoutB = 4:12 时计算结果会有所不同。

当 i = 0 时，

```cpp
new_shape = ceil_div(shapeA[0], strideB) = ceil_div(6, 12) = 1；// 这种情况说明 A 的当前维度不够分的。

new_stride = ceil_div(strideB, shapeA[0]) = ceil_div(12, 6) = 2；// A 的第一个列有 6 个元素，需要再来一列才够 12，所以第二个维度的 stride 就变成了 2。

shapeA[0] = 1; // 更新 shapeA[0]

strideB = 2;   // 更新下一个维度对应的 stride
```

当 i = 1 时，

```cpp
new_shape = ceil_div(shapeA[1], strideB) = ceil_div(4, 2) = 2；// 这种情况说明 A 的 shape 在当前维度可以被 strideB 分了。

new_stride = ceil_div(strideB, shapeA[1]) = ceil_div(2, 4) = 1；// 既然能够被分了，那后面的维度对应的 strideB 就等于 1 了。

shapeA[1] = 2; // 更新 shapeA[1]

strideB = 1;   // 更新下一个维度对应的 stride
```

计算结束，此时 shapeA 从 (6,4) 变成了 (1,2)。

下图表示 shapeA 的变化过程，深色部分是计算后 shapeA 的形状。

<div align="center">
        <img src="/assets/cute-layout-algebra/image2.png" width="100%" height="auto" alt="layout1">
        <small>(6,4) / 12 = (1,2)</small>
</div>
<br>

更多的例子：

```python
(6,2)  /  2  =>  (3,2)  # shapeA = (6,2), strideB = 2, A的第一维能被2整除，所以结果是(3,2)
(6,2)  /  3  =>  (2,2)
(6,2)  /  6  =>  (1,2)
(6,2)  / 12  =>  (1,1)  # A的第一维不能被12整除，第一维new_shape = ceil_div(6,12) = 1,
                          # 第二维对应的strideB更新为ceil_div(12,6) = 2，2可以被A的第二维整除
(3,6,2,8)  /  6  =>  (1,3,2,8)
(3,6,2,8)  /  9  =>  (1,2,2,8)
(42,16,3)  /  2  =>  (21,16,3)
(42,16,3)  /  6  =>  ( 7,16,3)
```

形状除法可以想象成将一个 shapeA 分成 strideB 份，第一维不够分就加上第二维，第二维还不够就再往后加。因此 shapeA 能被 strideB 可分的条件是：strideB 能被 shapeA 中的某一维前缀积整除。

cute 中通过代码中的 assert 来控制 shape 的可除性。

#### 形状取模（Shape Mod）

通过形状除法更新 shapeA 后，需要根据 shapeB 保留前 s 个元素。这一步是通过形状模运算（Shape Mod）得到的。

代码示例

```cpp
void shape_mod(int* shapeA, int N, int& shapeB) {
   for (int i = 0; i < N; ++i) {
      assert(shapeA[i] %    shapeB == 0 or
                shapeB % shapeA[i] == 0);
      int new_shapeA =      min(shapeA[i], shapeB);
      int new_shapeB = ceil_div(shapeB, shapeA[i]);
      shapeA[i] = new_shapeA;
      shapeB    = new_shapeB;
   }
}
```

输入：形状数组 shapeA，这里的形状是 shape_div 处理后得到的形状；维度 N 和目标保留元素数 shapeB

操作：逐维度更新 shapeA，保留每个维度不超过 shapeB 的部分，并更新剩余需要保留的元素数。

可除性条件：与 shape_div 类似，通过 assert 控制。

继续以 shapeA = (6,4)，layoutB = 4:3 为例。

shapeA 在进行形状除法后变为 (2,4)。shapeB = 4

当 i = 0 时，

```cpp
new_shapeA = min(shapeA[0], shapeB) = min(2, 4) = 2；

new_shapeB = ceil_div(shapeB, shapeA[0]) = ceil_div(4, 2) = 2；// 因为 shapeA 的第一维保留了 2 个元素，所以后面还需要保留 2 个元素。

shapeA[0] = 2;

shapeB = 2;
```

当 i = 1 时，

```cpp
new_shapeA = min(shapeA[1], shapeB) = min(4, 2) = 2；

new_shapeB = ceil_div(shapeB, shapeA[1]) = ceil_div(2, 4) = 1；// 这里表示 shapeB 已经被选择完了，所以后面维度会全部更新为 1，然后通过 coalesce 去掉这些 1。

shapeA[1] = 2;

shapeB = 1;
```

最终 shapeA 从 (2,4) 变成了 (2,2)

更多的例子：

```cpp
(6,2)  %  2  =>  (2,1)
(6,2)  %  3  =>  (3,1)
(6,2)  %  6  =>  (6,1)
(6,2)  % 12  =>  (6,2)
(3,6,2,8)  %  6  =>  (3,2,1,1)
(3,6,2,8)  %  9  =>  (3,3,1,1)
(1,2,2,8)  %  2  =>  (1,2,1,1)
(1,2,2,8)  % 16  =>  (1,2,2,4)
```

#### 计算 stride

得到最终 shape 后，根据 A 和 B 的 stride 可以计算得到组合后 shape 对应的 stride。从前面形状除法的过程中也不难发现，在更新 strideB 的同时与 A 对应的 stride 相乘就能得到最终结果的 stride。

还是以 shapeA = (6,4)，layoutB = 4:3 为例。假设 shapeA 对应的 stride 是 (1,6)。

通过形状除法和形状取模可以得到最终的 shape 是 (2,2)。

因为原始 shapeA 的第一维的 strideA 是 1，所以新的 shape 的第一维的 stride = strideA × strideB = 1 × 3 = 3。

计算第二维时 strideB 变成了 1，所以新 shape 的第二维 stride = strideA × strideB = 6 × 1 = 6。

所以最终 shape (2,2) 对应的 stride 为 (3,6)。

需要注意的是，每一维组合结束后会对结果做一次合并，因此最终结果是：

\begin{equation*}
(6,4):(1,6) \circ 4:3 = (2,2):(3,6)
\end{equation*}

#### Python 伪代码

下面是用 python 实现的组合的代码。完整代码见 [https://github.com/NVIDIA/cutlass/blob/v3.8.0/python/pycute/layout.py#L190](https://github.com/NVIDIA/cutlass/blob/v3.8.0/python/pycute/layout.py#L190)

```python
# Layout composition
def composition(layoutA, layoutB):
    if is_tuple(layoutB.shape):
        # 判断layoutB的shape是否是tuple，是的话拆分处理。
        return make_layout(composition(layoutA, layoutB_i) for layoutB_i in layoutB)

    if layoutB.stride == 0:
        return Layout(layoutB.shape, 0)
    else:
        result_shape = []
        result_stride = []
        rest_shape = layoutB.shape
        rest_stride = layoutB.stride
        for s, d in zip(flatten(layoutA.shape)[:-1], flatten(layoutA.stride)[:-1]):
            s1 = shape_div(s, rest_stride)   # 按照形状除法计算当前维度shape大小，这里的shape_div基本等于ceil_div
            result_shape.append(min(s1, rest_shape))  # 按照形状取余计算需要保留的shape大小
            result_stride.append(rest_stride * d)     # 计算当前维度的stride
            rest_shape = shape_div(rest_shape, abs(s1))   # 计算剩余shape大小
            rest_stride = shape_div(rest_stride, s)       # 计算剩余stride的大小

        result_shape.append(rest_shape)
        result_stride.append(rest_stride * flatten(layoutA.stride)[-1])

        # 因为是对layoutB的单个维度处理的，因此最终结果会做一个coalesce简化维度
        return coalesce(Layout(tuple(result_shape), tuple(result_stride)))
```

#### Example 1

\begin{align*}
20:2 \circ (5,4):(4,1) &= 20:2 \circ (5:4, 4:1) \\\\
&= (20:2 \circ 5:4,\ 20:2 \circ 4:1) \\\\
&= (5:8,\ 4:2) \\\\
&= (5,4):(8,2)
\end{align*}

#### Example 2

\begin{align*}
(10,2):(16,4) \circ (5,4):(1,5) &= ((10,2):(16,4) \circ (5:1, 4:5)) \\\\
&= ((10,2):(16,4) \circ 5:1,\ (10,2):(16,4) \circ 4:5) \\\\
&= (5:16,\ (2,2):(80,4)) \\\\
&= (5,(2,2)):(16,(80,4))
\end{align*}


这个组合过程可以描述为将 layout \((10, 2) : (16, 4)\) 重新组合为一个 \(5*4\) 的列主序矩阵。

首先根据上面的规则， \((5, 4) : (1, 5)\) 可以拆分为 \((5 : 1, 4 : 5)\)。

因此
$$
(10, 2) : (16, 4) \circ (5, 4) : (1, 5) = ((10, 2) : (16, 4) \circ 5 : 1, (10, 2) : (16, 4) \circ 4 : 5)
$$

首先计算 $\((10, 2) : (16, 4) \circ 5 : 1\)$ ，根据上面的计算方法得到 $\((5, 1) : (16, 4)\)$，coalesce 后变成 $\(5 : 16\)$ 。

然后计算 $\((10, 2) : (16, 4) \circ 4 : 5\)$ ， $\((10, 2)/5 = (2, 2)\)$ ， $\((2, 2)\%4 = (2, 2)\)$ ， $\(stride = (16 * 5, 4 * 1) = (80, 4)\)$ ，coalesce后得到 $\((2, 2) : (80, 4)\)$ 。

最后将 $\(5 : 16\)$ 和 $\((2, 2) : (80, 4)\)$ 拼接一起得到
$$
(5 : 16, (2, 2) : (80, 4)) = (5, (2, 2)) : (16, (80, 4)) 。
$$

在 cute 中，上面的 shape 和 stride 如果用静态整数表示，组合结果为 (5,(2,2)):(16,(80,4))，如果用动态整数表示，组合结果为 ((5,1),(2,2)):((16,4),(80,4))。

这两种结果在数学上是等价的。当使用动态整数时，由于一些限制没有对结果做 coalesce。

下图展示了 layoutA 和 layoutB 以及他们组合结果 layoutC 的布局。

<div align="center">
        <img src="/assets/cute-layout-algebra/image3.png" width="100%" height="auto" alt="layout1">
        <small>(10,2):(16,4) ∘ (5,4):(1,5) = (5,(2,2)):(16,(80,4))</small>
</div>
<br>

### By-mode Composition（按维度组合）

与按维度进行 coalesce 一样，composition 也可以按维度操作。

当在 composition 函数中传入的第二个参数是 Layout 时，会按照普通的组合进行。当传进去的是一个 Layout 元组时，就可以按维度进行组合。cute 中将这种 layout 元组称为 Tiler。

举例：

```cpp
// (12,(4,8)):(59,(13,1))
auto a = make_layout(make_shape (12,make_shape ( 4,8)),
                     make_stride(59,make_stride(13,1)));
// <3:4, 8:2>
auto tiler = make_tile(Layout<_3,_4>{},  // Apply 3:4 to mode-0
                       Layout<_8,_2>{}); // Apply 8:2 to mode-1

// (_3,(2,4)):(236,(26,1))
auto result = composition(a, tiler);
// Identical to
auto same_r = make_layout(composition(layout<0>(a), get<0>(tiler)),
                          composition(layout<1>(a), get<1>(tiler)));
```

在上面的例子中，layout a = (12,(4,8)):(59,(13,1))，tiler 是由 3:4 和 8:2 组成的元组。因此 composition(a, tiler) 就相当于 12:59 ∘ 3:4 和 (4,8):(13,1) ∘ 8:2。结果为 (3,(2,4)):(236,(26,1))，即下图中灰色部分。

这一过程相当于对 layout a 按照间隔为 4 取 3 行，间隔为 2 取 8 列。

\begin{align*}
(12,(4,8)):(59,(13,1)) \circ \langle(3:4),(8:2)\rangle &= (12:59 \circ 3:4,\ (4,8):(13,1) \circ 8:2) \\\\
&= (3:236,\ (2,4):(26,1)) \\\\
&= (3,(2,4)):(236,(26,1))
\end{align*}

<div align="center">
        <img src="/assets/cute-layout-algebra/image4.png" width="100%" height="auto" alt="layout1">
        <small>(10,2):(16,4) ∘ (5,4):(1,5) = (5,(2,2)):(16,(80,4))</small>
</div>
<br>

如果第二个参数是一个 shape 元组，也会按照 tiler 处理。因为一个 shape 可以看作一个 stride=1 的 layout。

```cpp
// (12,(4,8)):(59,(13,1))
auto a = make_layout(make_shape (12,make_shape ( 4,8)),
                     make_stride(59,make_stride(13,1)));
// (3, 8)
auto tiler = make_shape(Int<3>{}, Int<8>{});
// Equivalent to <3:1, 8:1>
// auto tiler = make_tile(Layout<_3,_1>{},  // Apply 3:1 to mode-0
//                        Layout<_8,_1>{}); // Apply 8:1 to mode-1

// (_3,(4,2)):(59,(13,1))
auto result = composition(a, tiler);
```

继续以 (12,(4,8)):(59,(13,1)) 为例，shape (3,8) 可以看作 tiler (3:1, 8:1)，因此组合结果如下图灰色部分所示。这个结果就相当于在 layout a 中取一个大小为 3 行 4 列的数据块（Tile）。

> **TODO:** 图片待补充（shape tiler 示意图）

Tile 这个概念在后面会反复提及。一个 Tile 可以理解为具有一定布局的数据块。

这个概念经常用在矩阵分块中。比如一个 32×32 的矩阵按照 8×8 的大小可以分为 16 个 tile，每个 tile 的大小就是 8×8。根据 tile 的 layout 不同，每个 tile 分到的数据就不一样，tile 与 tile 之间的 layout（也就是后面提到的补集）也不一样。

## Complement

CuTe 中补集（Complement）操作用于**找到未被原布局选中的元素在内存中的排列方式**。其核心目标是为原布局 A 相对于形状 M 生成一个互补布局 R。

```
Layout complement(LayoutA const& layout_a, Shape const& cotarget)
```

补集操作通过函数 complement(LayoutA, Shape M) 实现，满足以下性质：

1. **大小约束：**

补集布局 R 的大小（size）和陪域大小（cosize）受目标形状 M 大小的限制。

2. **有序性：**

补集布局 R 是有序的。R 的步幅（stride）必须是递增且正数的，保证其唯一性。

3. **互斥性：**

补集布局 R 的陪域（codomain）与原布局 A 的陪域互不相交。布局 R 试图"补全"布局 A 的陪域。

上述定义翻译自原文档，可能不太严谨。

只看定义理解起来有些抽象。根据对示例和代码的理解，个人觉得：对于一个 layout A，如果存在一个 layout R，当把 A 按照 R 排列后得到的结果能完全覆盖 M 空间的所有元素，就称 layout R 是 layout A 在 M 下的补集。

换句话说就是：一个大矩阵 M，分成很多块（Tile），每个块的布局是 layout A，块与块之间的布局就是 layout A 在大矩阵上的补集 R。

### 计算过程

总的来说，计算一个 layout 的补集可以分为 2 步：第一步：填充；第二步：复制。

**1. 填充**

如果 layout A 的 stride 不是 1 的话，就表明 layout A 的 codomain 空间中有些元素没有覆盖到。比如 layout 4:2 的 stride=2，第 1，3，5 位置处的元素没有被该 layout 覆盖。这些没被覆盖的元素就是"窟窿"，需要进行填充。

此时存在 layout R1，将 layout A 按照 R1 进行排列，得到的新的 layout 就可以完全覆盖到 layout A 的 codomain 中的元素。

对于 layout 4:2，layout R1 为 2:1。因为 4:2 对应 [0,2,4,6]，把 [0,2,4,6] 看成一个元素按照 shape=2，stride=1 排列就可以得到 [[0,2,4,6],[1,3,5,7]]，此时就可以覆盖 0-7 中的所有元素。

**2. 复制**

当通过 layout R1 处理后，layout A 能覆盖的所有元素为 cosize(layout A)。如果小于 M 的大小，就需要根据 layout R2 对 layout A 进行复制，直到复制后的 layout 能完全覆盖 M 的大小。

继续以 layout 4:2 为例，经过 layout R1 的处理，其能覆盖的范围为 8，当 M 等于 24 时，需要将其复制 3 次才能覆盖 M，所以 layout R2 的 shape = ceil_div(24,8) = 3，stride = 8，即 3:8。

所以 layout A 在 M 下的补集就是 R = (R1, R2)。4:2 在 M=24 下的补集就是 (2,3):(1,8)。

下图表示 complement(4:2, 24) 的结果。灰色是 layout A，蓝色是对 layout 的填充，红色和绿色是复制的结果。

> **TODO:** 图片待补充（complement(4:2, 24) 示意图）

### Python 代码

详见 [https://github.com/NVIDIA/cutlass/blob/v3.8.0/python/pycute/layout.py#L223](https://github.com/NVIDIA/cutlass/blob/v3.8.0/python/pycute/layout.py#L223)

```python
# Layout complement
def complement(layout, max_idx=1):
  if is_int(layout):
    return complement(Layout(layout))

  result_shape  = []
  result_stride = []
  current_idx = 1

  sorted_DS = sorted(zip(flatten(layout.stride), flatten(layout.shape)))
  for (stride, shape) in sorted_DS:
    if stride == 0 or shape == 1:
      continue

    in_bound = current_idx <= shape * stride
    # To support symbolic value which can't be evaluated now
    assert (type(in_bound) is not bool) or in_bound

    result_shape.append(stride // current_idx)
    result_stride.append(current_idx)
    current_idx = shape * stride

  result_shape.append((max_idx + current_idx - 1) // current_idx)  # ceil_div
  result_stride.append(current_idx)

  return coalesce(Layout(tuple(result_shape), tuple(result_stride)))
```

### Examples

1. **complement(4:1, 24) = 6:4**

布局 4:1 的 stride 已经是 1 了，所以不需要对其进行填充。但是此时该布局只覆盖了 4 个元素，距离 24 还需要复制 ceil_div(24,4) = 6 次，所以补集 R 等于 6:4。stride=4 是因为 4:1 占了 4 个元素。

2. **complement(6:4, 24) = 4:1**

布局 6:4 的 shape=6，stride=4。stride 不是 1，所以需要先对其进行填充。由于 stride=4，所以补集 R 在该维度的 shape=4，stride=1。填充完毕后，6×4=24，已经覆盖了 M 的范围，所以就不需要复制了。因此 6:4 相对于 24 的补集是 4:1。

3. **complement((4,6):(1,4), 24) = 1:0**

布局 (4,6):(1,4) 的 stride=1 且 cosize=24，已经覆盖了 24 的范围，所以不存在补集。

4. **complement((2,4):(1,6), 24) = 3:2**

首先处理第一维。第一维的 stride=1，shape=2。因为是连续的，所以不需要填充。再处理第二维，stride=6，shape=4。因为第一维的 cosize=2，小于 6，说明在第二维上不连续，需要填充 6/2=3 个元素，因此补集在第二维的 shape=3，stride=2。

此时 layout 的 stride=1 且 cosize=24，所以不需要复制了。因为在第一维上没有补集，所以补集就是 3:2。

5. **complement((2,2):(1,6), 24) = (3,2):(2,12)**

首先处理第一维。因为第一维是连续的，所以不需要填充。第二维上不连续，需要补充 3 个元素，每个元素的 stride=2，所以补集在第二维上的 shape=3，stride=2。

此时虽然 layout 已经连续，但是 cosize=12，所以还需要复制一次。因此补集在第三维上的 shape=24/12=2，stride=12。最终，补集 layout = (3,2):(2,12)。

下图表示 complement((2,2):(1,6), 24) 的结果。根据图片也可以理解为一个 6 行 4 列的矩阵，被 layout = (2,2):(1,6) 的 tile 分，可以分为 6 个 tile，这 6 个 tile 之间的 layout 是 (3,2):(2,12)。

> **TODO:** 图片待补充（complement((2,2):(1,6), 24) 示意图）
