---
title: CuTe 学习笔记（二）Layout Algebra（二）
date: 2025-03-20 18:00:00
tags: [CUTLASS, CuTe, Layout, GPU]
categories: [Cutlass 学习笔记]
mathjax: true
description: 文章继续介绍 CuTe Layout Algebra 的具体的代数运算流程，包括 Division, Product, Inverse 操作。
---

# CuTe Layout Algebra

CuTe 提供了 layout 代数来支持 layout 以不同的方式进行组合。这些代数主要包括 composition，product，divide 等。

下面在上一篇文章的基础上继续介绍 layout 中常用的函数，包括 `Division`, `Product`, `Inverse`。

## Division (Tiling) 除法操作

CuTe 中定义了 Layout 的除法操作 `logical_divide(A, B)`，通过 `logical_divide` 把一个布局划分为两部分，一部分是由布局 B 指向的元素（通过组合操作 $A \circ B$ 实现）；另一部分是未被 B 指向的元素（通过补集 $B^*$ 表示）。

### 计算方法

用公式表示为：

\begin{equation*}
A \oslash B = A \circ (B, B\^\*) 
\end{equation*}

其中 $B^*$ 是 $B$ 在 $A$ 大小范围内的补集。

所以 `logical_divide` 主要有三个操作组成：

1. 组合（Composition）：将布局 $A$ 与目标布局结合。
2. 补集（Complement）：生成未被 $B$ 覆盖的部分 $B^*$。
3. 拼接（Concatenation）：将 $B$ 和 $B^*$ 拼接成一个新布局。

代码实现：

```cpp
template <class LShape, class LStride,
          class TShape, class TStride>
auto logical_divide(Layout<LShape,LStride> const& layout,
                    Layout<TShape,TStride> const& tiler)
{
  return composition(layout, make_layout(tiler, complement(tiler, size(layout))));
}
```

### 计算过程

Division 操作主要用于对矩阵进行分块处理。对于一个布局为 layout A 的矩阵，我们根据 layout B 对矩阵进行分块。`logical_divide(A, B)` 的结果就是把矩阵按照 layout B 分到的元素放到一起。

举例：

layout A = $(4,6):(1,4)$

layout B = $(2,2):(2,4)$

logical_divide(A, B) = $((2,2),(2,3)):((2,4),(1,8))$

layout A 的形状如下：

<div align="center">
        <img src="/assets/cute-layout-algebra/divide1.png" width="100%" height="auto" alt="layout">
        <small>layout A 的形状</small>
</div>
<br>

layout B 的形状如下：

<div align="center">
        <img src="/assets/cute-layout-algebra/divide2.png" width="100%" height="auto" alt="layout">
        <small>layout B 的形状</small>
</div>
<br>

首先通过 composition 操作获取 layout A 中被 layout B 选中的元素的布局。如下图所示，可以看到被 B 选中的元素在 A 中并不连续。

<div align="center">
        <img src="/assets/cute-layout-algebra/divide3.png" width="100%" height="auto" alt="layout">
        <small></small>
</div>
<br>

然后计算 layout A 与 layout B 补集的组合。

这一步先计算 layout B 相对 layout A 的补集。如下图所示，layout A 被 layout B 分成 6 块，用不同的颜色表示。块与块之间的 layout 是 $(2,3):(1,8)$，也就是 layout B 的补集。有了补集后再与 A 进行组合。

<div align="center">
        <img src="/assets/cute-layout-algebra/divide4.png" width="100%" height="auto" alt="layout">
        <small></small>
</div>
<br>

将前面计算的结果进行拼接。把 layout B 对应元素的布局放在第一维，块与块的布局放在第二维得到最终的除法结果。从下图可以看到，通过 `logical_divide`，layout A 中的元素按照 layout B 分块的结果重新排列了。

<div align="center">
        <img src="/assets/cute-layout-algebra/divide5.png" width="100%" height="auto" alt="layout">
        <small></small>
</div>
<br>

### By-mode Logical Divide (按维度除法)

`logical_divide` 支持按维度计算。

以一个二维布局 $A = (9,(4,8)):(59,(13,1))$ 为例，如果在第一维（列方向）上按照 $3:3$ 计算，第二维（行方向）上按照 $(2,4):(1,8)$ 计算，整个计算过程可以写成：

\begin{equation*}
\begin{aligned}
A \oslash B &= (9,(4,8)):(59,(13,1)) \oslash \langle 3:3, (2,4):(1,8) \rangle \\\\
            &= 9:59 \oslash 3:3, \ (4,8):(13,1) \oslash (2,4):(1,8) \\\\
            &= (3,3):(177,59), ((2,4),(2,2)):((13,2),(26,1)) \\\\
            &= ((3,3),((2,4),(2,2))):((177,59),((13,2),(26,1)))
\end{aligned}
\end{equation*}

具体计算过程如下图所示：

首先按照 layout B 的布局对 layout A 分块。列方向上 9 个元素分成三块，每块 3 个元素。行方向上 32 个元素分成 4 块，每块 8 个元素。相同块中的元素用相同的颜色表示。

通过 `logical_divide` 操作把相同的块对应的元素放到一起。由于是按维度计算的，所以在列方向上相同块中的三个元素按顺序排列到一起。行方向上相同块中的 8 个元素按顺序排列到一起，最终结果如下图所示。

<div align="center">
        <img src="/assets/cute-layout-algebra/divide6.png" width="100%" height="auto" alt="layout">
        <small></small>
</div>
<br>

### Zipped, Tiled, Flat Divides

除了 `logical_divide`，CuTe 还提供了 `zipped_divide`，`tiled_divide` 和 `flat_divide`。这四种除法的计算过程相同，区别在于最后对分块的元素排列不同。

```cpp
Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)
zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))
tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...)
flat_divide    : (TileM, TileN, RestM, RestN, L, ...)
```

其中，TileM 可以理解为 layout B 对应的 tile 的行数，RestM 是 layout A 的行数被 layout B 分成了几份。同理 TileN 是 layout B 的列数，RestN 是行方向上 layout B 的数量。

以上图为例：

`logical_divide` 是按照 ((TileM,RestM), (TileN,RestN)) 的顺序排列最终的结果，即：
$$((3,3),((2,4),(2,2))):((177,59),((13,2),(26,1)))$$

`zipped_divide` 则是按照 ((TileM,TileN), (RestM,RestN)) 排列，结果是: 
$$((3,(2,4)),(3,(2,2))):((177,(13,2)),(59,(26,1)))$$

同样的，

`tiled_divide` = $((3,(2,4)),3,(2,2)):((177,(13,2)),59,(26,1))$

`flat_divide` = $(3,(2,4),3,(2,2)):(177,(13,2),59,(26,1))$

`zipped_divide` 会变成下图的样子。可以看到，相同块的元素全部在同一维度中。

<div align="center">
        <img src="/assets/cute-layout-algebra/divide7.png" width="100%" height="auto" alt="layout">
        <small></small>
</div>
<br>

`tiled_divide` 和 `flat_divide` 的结果与 `zipped_divide` 类似，只是维度变成了多维的。

## Product (Tiling) 乘法操作

CuTe 定义了 layout 与 layout 之间的乘法，`logical_product(A, B)`。

`logical_product` 的作用是把 layout A 按照 layout B 的布局进行复制。`logical_product` 的计算公式为：

\begin{equation*}
A \otimes B = (A, A^* \circ B)
\end{equation*}

```cpp
template <class LShape, class LStride,
          class TShape, class TStride>
auto logical_product(Layout<LShape,LStride> const& layout,
                     Layout<TShape,TStride> const& tiler)
{
  return make_layout(layout, composition(complement(layout, size(layout)*cosize(tiler)), tiler));
}
```

### 计算过程

从公式理解，product 的含义就是把 layout A 按照 layout B 进行复制。layout A 是计算结果的第一维，layout A 的补集表示 A 需要按照什么布局进行复制。

以 layout A = (2,2):(4,1) 和 layout B = (4,2):(2,1) 相乘为例。

<div align="center">
        <img src="/assets/cute-layout-algebra/product1.png" width="100%" height="auto" alt="layout">
        <small>layout A 和 layout B</small>
</div>
<br>

其中 size(A) = 4, cosize(B) = 8。

首先计算 A 在 size(A)*cosize(B) = 32 空间下的补集。通过上面补集的计算过程不难算出，A 的补集是 (2,4):(2,8)。

<div align="center">
        <img src="/assets/cute-layout-algebra/product2.png" width="40%" height="auto" alt="layout">
        <small>A的补集</small>
</div>
<br>

此时 layout A 在 32 空间下的完整表示如下图。

<div align="center">
        <img src="/assets/cute-layout-algebra/product3.png" width="100%" height="auto" alt="layout">
        <small></small>
</div>
<br>


然后把补集按照 layout B 的布局排列。

<div align="center">
        <img src="/assets/cute-layout-algebra/product4.png" width="100%" height="auto" alt="layout">
        <small></small>
</div>
<br>

与 layout A 进行组合得到最后结果。

<div align="center">
        <img src="/assets/cute-layout-algebra/product5.png" width="100%" height="auto" alt="layout">
        <small></small>
</div>
<br>


### By-mode Logical Product (按维度乘)

`logical_product` 也可以按照维度分别计算。

如下图所示，layout A $(2,5):(5,1)$ 与 layout 元组 B $<3:5, 4:6>$ 相乘的结果如下所示。从图中可以看到，layout A 按行扩大 3 倍，按列扩大 4 倍。

\begin{equation*}
\begin{aligned}
A \otimes B &= ((2,5):(5,1)) \otimes \langle 3:5, 4:6 \rangle \\\\
            &= (2:5 \otimes 3:5), \ (5:1 \otimes 4:6) \\\\
            &= (2,3):(5,10), (5,4):(1,30) \\\\
            &= ((2,3),(5,4)):((5,10),(1,30))
\end{aligned}
\end{equation*}

<div align="center">
        <img src="/assets/cute-layout-algebra/product6.png" width="100%" height="auto" alt="layout">
        <small></small>
</div>
<br>

但是官方并不推荐使用这种方法，原因是不能直观的通过 A 和 B 的 layout 判断出计算结果是什么样子。

因此 cute 提供了 `blocked_product` 和 `raked_product` 这两个函数。

`blocked_product` 的计算结果与 `logical_product` 类似，但是输入的第二个参数不同。

`blocked_product` 的第二个参数是对 layout A 复制的数量。在下图中，第二个参数是 (3,4):(1,3)，代表把 layout A 在第一维复制 3 次，在第二维复制 4 次。使用这种方法可以更直观的理解两个 layout 相乘之后的结果。

<div align="center">
        <img src="/assets/cute-layout-algebra/product7.png" width="100%" height="auto" alt="layout">
        <small></small>
</div>
<br>

`raked_product` 的计算结果与 `blocked_product` 略有不同，他不是像 `blocked_product` 一样按照 (rowA, rowB),(colA, colB) 这种顺序组合的，而是按照 (rowB, rowA),(colB, colA) 进行组合的，计算结果如下图所示。

<div align="center">
        <img src="/assets/cute-layout-algebra/product8.png" width="100%" height="auto" alt="layout">
        <small></small>
</div>
<br>

### Zipped and Tiled Products

与 `zipped_divide` 和 `tiled_divide` 类似，cute 也提供了 `zipped_product`，`tiled_product` 和 `flat_product`，与 `logical_product` 的区别在于最终结果的组合维度不同。

```cpp
Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_product : ((M,TileM), (N,TileN), L, ...)
zipped_product  : ((M,N), (TileM,TileN,L,...))
tiled_product   : ((M,N), TileM, TileN, L, ...)
flat_product    : (M, N, TileM, TileN, L, ...)
```

以 layout A $(2,5):(5,1)$ 与 layout 元组 B $<3:5, 4:6>$ 相乘为例：

`logical_product` 是按照 ((M,TileM), (N,TileN)) 的顺序排列最终的结果，即：
$$((2,3),(5,4)):((5,10),(1,30))$$

`zipped_product` 则是按照 ((M,N), (TileM,TileN)) 排列，结果是:
$$((2,5),(3,4)):((5,1),(10,30))$$

同样的，

`tiled_product` = $((2,5),3,4):((5,1),10,30)$。

`flat_product` = $(2,5,3,4):(5,1,10,30)$。

下图是 `zipped_product` 的结果。可以看到，相同块的元素全部在同一维度中。

<div align="center">
        <img src="/assets/cute-layout-algebra/product9.png" width="100%" height="auto" alt="layout">
        <small></small>
</div>
<br>

## Inverse (逆)

在 CuTe 中，一个布局（Layout, 记为 $L$）本质上是一个数学函数：它将张量的逻辑坐标映射到物理内存的偏移量。即 `逻辑坐标 -> 物理偏移`。

对于普通的数学函数，如果它是双射（Bijection），我们就能求出它的完全逆函数（Inverse, 记为 $L^{-1}$），此时 $L^{-1}(L(k)) = k$。在布局代数的语境下，如果一个布局是完全连续且无重叠的（例如标准的行优先二维数组，步长严格等于其尺寸），我们可以直接求逆，从物理偏移反向找回逻辑坐标。这种完全可逆的布局也被称为“紧凑（Compact）”布局。

但是，在实际的硬件和 AI 算子优化中，我们遇到的布局往往并不完美。它们可能包含空洞（如较长的步长 stride），或者重叠（如步长为 0，实现广播）。此时，标准的“完全逆”不存在，于是 CuTe 引入了**右逆**和**左逆**这两个广义的伪逆概念。

### Right-Inverse（右逆）

**右逆（记为 $L^\ddagger$）** 解决的是这样一个问题：“当我有一段物理内存是连续时，这些连续的内存对应着张量的哪些逻辑坐标？”因为我们要考虑的是“连续内存”而不一定是“所有内存”，所以右逆常用于识别张量中可向量化（SIMD）处理的数据块。

它的数学定义如下：
$$
 \forall k \in \mathbb{Z}_{|L^\ddagger|}, \quad L^\ddagger(L(L^\ddagger(k))) = L^\ddagger(k) 
$$

这表示右逆 $L^\ddagger$ 作为从连续物理地址到逻辑坐标的映射，与原始布局 $L$ 重新组合，必须在右逆的定义域内保持恒等。

*注意：右逆的定义域大小 $L^\ddagger$ 往往小于等于原始布局的大小 $L$，因为只有物理上连续的那部分才能构成右逆的有效输入。*

#### 右逆的具体示例

下表展示了不同的布局 $L$ 及其对应的右逆 $L^\ddagger$。通过观察表格，我们可以直观感受到右逆的作用：**它剥离了原始布局中的“空洞”，只保留物理上连续的内存块，并将其在逻辑上重组成一个新的紧凑布局。**

| 原始布局 $L$ (形状:步长) | 右逆 $L^\ddagger$ | 注释 (Comments) |
| :--- | :--- | :--- |
| `(4,8):(1,4)` | `32:1` | 物理内存完全连续，可以一次性索引 32 个元素 |
| `(4,8):(8,1)` | `(8,4):(4,1)` | 物理上依然连续（跨度 8 只是表示到下一行），重组为 8x4 的逻辑形状 |
| `(3,7,5):(5,15,1)` | `(5,21):(21,1)` | 非 2 次幂的步长/尺寸也能正确求右逆 |
| `(4,8):(1,5)` | `5:1` | 尺寸变小，因为 `(4,8)` 实际上跨越了更大的内存，但真正连续的大小只有 `5` |
| `(4,(4,2)):(4,(1,16))` | `(4,4,2):(4,1,16)` | 结果域与 `Z` 共轭 |
| `((2,2),(2,4)):((1,8),(2,16))` | `(2,4,2,2):(1,4,2,16)` | |
| `((2,2),(2,4)):((0,1),(0,4))` | `(2,2):(4,8)` | 步长为 0（广播模式）不参与连续空间的贡献 |
| `((2,2),(2,4)):((0,2),(0,4))` | `1:0` | 全为步长 0，右逆退化为一个元素的平凡逆（Trivial right-inverse） |
| `(4,8):(e0,e1)` | `(4,8):(1,4)` | 结果域与 `Z^{(*,*)}` 共轭，可以处理整数模（General integer-semimodule） |
| `(4,(4,2)):(e1,(e0,6e1))` | `(4,6,2):(4,1,16)` | |
| `(4,(4,3)):(f1,(f5,f16))` | `(4,4,3):(f1,f5,f16)` | 支持整数模步长定义 |

#### 应用：向量化示例（Vectorization Example）

右逆的实战价值体现在自动向量化上。

假设我们要将张量 A 拷贝到张量 B。如果不做优化，我们可能会一个元素一个元素地拷贝。但 GPU/CPU 支持 SIMD 指令（如 AVX/Neon），一次可以加载 4 个或 8 个 Float 数据——前提是这 4 个数据在物理内存里是连续且对齐的。

CuTe 通过右逆来计算最大安全向量化的元素个数 K，需要满足：

$$
 \forall k \in \mathbb{Z}_K, \quad A^\ddagger(k) = B^\ddagger(k) 
$$

这表示在物理偏移 0 到 K-1 的范围内，A 和 B 在逻辑上的右逆映射是完全一致的（即两者完全对齐）。通过代数运算，CuTe 能剥离出物理连续的逻辑子布局：

$$
 A^\ddagger \circ [B \circ A^\ddagger]|_K = A^\ddagger \circ I_K = [A^\ddagger]|_K 
$$

$$
 B^\ddagger \circ [A \circ B^\ddagger]|_K = B^\ddagger \circ I_K = [B^\ddagger]|_K 
$$

<div align="center">
        <img src="/assets/cute-layout-algebra/inverse1.png" width="100%" height="auto" alt="layout">
        <small></small>
</div>
<br>

如上图所示：
*   在 图 a 中，右逆计算得出只有前 2 个逻辑元素是连续且对齐的，因此只能安全地向量化拷贝 2 个元素。
*   在 图 b 中，通过右逆的代数计算，发现可以安全地向量化拷贝 4 个元素。这 4 个元素的逻辑坐标是 `{0, 2, 8, 10}`，虽然逻辑上不连续，但在物理偏移上正好是连续的。

右逆将复杂的多维步长问题，降维成了纯物理内存连续性的代数问题，使得自动向量化不再需要开发者编写繁琐的边界判断代码。


### Left-Inverse（左逆）

如果说右逆是为数据搬运服务的，那么左逆（记为 $L^\dagger$）就是为硬件指令与逻辑布局的匹配服务的。左逆解决的问题是：“给定一个物理内存偏移，它对应的逻辑坐标是什么？” （即使该布局不是单射的，即多个逻辑坐标对应同一个物理地址，左逆也能找到其中一个合理的逻辑坐标）。

它的数学定义如下所示：

$$\forall k \in \mathbb{Z}_{|L|}, \quad L(L^\dagger(L(k))) = L(k) $$


左逆存在的前提是原始布局 $L$ 必须是单射（Injective），即逻辑坐标到物理偏移没有重叠。如果在单射的前提下，左逆并不一定唯一，特别是当布局存在 stride=0（导致映射不是单射）时，左逆的求解结果会出现不确定性。

### 左逆的具体示例

下表展示了不同布局 $L$ 与其对应的左逆 $L^\dagger$：

| 原始布局 $L$ (形状:步长) | 左逆 $L^\dagger$ | 注释 (Comments) |
| :--- | :--- | :--- |
| `(4,8):(1,4)` | `32:1` | 对于连续图像，结果与右逆相同 |
| `(4,8):(8,1)` | `(8,4):(4,1)` | |
| `(3,7,5):(5,15,1)` | `(5,21):(21,1)` | 非 2 次幂的尺寸/步长，左逆仍是可处理的 |
| `(4,8):(1,5)` | `(5,8):(1,4)` | 结果尺寸较大（非连续图像的伪逆表现） |
| `(4,(4,2)):(4,(1,16))` | `(4,4,2):(4,1,16)` | 结果域与 `Z` 共轭 |
| `((2,2),(2,4)):((1,8),(2,16))` | `(2,4,2,2):(1,4,2,16)` | |
| `((2,2),(2,4)):((0,2),(0,4))` | `(2,2,4):(0,2,8)` | **结果不唯一**（Result is not unique） |
| `((2,2),(2,4)):((0,1),(0,2))` | `(2,2):(4,8)` | 任何 mode-0 步长均可作为左逆 |
| `(4,8):(e0,e1)` | `(4,8):(1,4)` | 结果域与 `Z^{(*,*)}` 共轭 |
| `(4,(4,2)):(e1,(e0,6e1))` | `(4,6,2):(4,1,16)` | 结果尺寸较大，且域兼容 |
| `(4,(4,3)):(f1,(f5,f16))` | `(4,4,3):(f1,f5,f16)` | 可定义于整数模步长 |

### 应用：容许性示例（Application: Admissibility Example）

下面以 TMEM（Tensor Memory，张量内存）为例介绍左逆的应用。

NVIDIA Blackwell 架构引入了 TMEM，TMEM 拥有一系列非常固定的硬件指令（如 `tcgen05`），这些指令只能以特定的物理访问模式（如特定的行、列跨度组合）从 TMEM 中读写数据。

我们定义的数据布局 `A` 是任意的，而硬件指令布局 `T` 是固定的。我们需要判断：“这个特定的 TMEM 指令，是否能安全地访问我当前定义的张量布局 `A` ？”

我们计算复合布局 $A^\dagger \circ T$，将硬件指令要求的物理偏移 `T(i)` 作为输入，利用张量的左逆 `A†`，计算出这个物理偏移对应在张量逻辑坐标中的位置。如果计算成功，说明硬件指令访问的物理偏移完全落在这个张量的有效范围内，则该指令是可容许（Admissible）的。

通过结合左逆和 `zipped_divide` 等算子，CuTe 可以在编译期自动推导出 TMEM 硬件指令（如 x1, x2, x16 等）如何映射到逻辑矩阵的子布局上。这使得我们可以将寄存器级的硬件指令直接与高维张量布局的代数描述相统一，极大简化了底层硬件的算子开发流程。

### 小结
*   **右逆（$L^\ddagger$）**：面向数据的连续搬运。将复杂跨步的布局降维，提取出物理上连续的逻辑块，实现自动向量化。结果不唯一时，通常取物理大小最大的映射。
*   **左逆（$L^\dagger$）**：面向指令与数据的交互。将固定的硬件物理访问模式（如 TMEM 偏移）反向映射到任意的张量逻辑布局中，实现硬件指令的自动化判等与映射。


## 总结

本文依次介绍了 layout 的除法，乘法和逆。通过这些代数运算可以结合特定的硬件指令实现高效的数据处理。