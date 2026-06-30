---
title: CuTe 学习笔记（八）Permutations 参数的作用
date: 2025-03-27 18:00:00
tags: [CUTLASS, CuTe, Permutations]
categories: [Cutlass 学习笔记, CuTe]
description: 深入解析 make_tiled_mma 中 Permutations 参数的作用，通过实例演示不同 Permutations 设置对 tiledMMA 线程布局和数据分区的影响。
---

cutlass 的 make_tiled_mma 函数定义如下，可以看到有三个参数，

第一个是 MMA_OP，代表使用哪种类型的 mma 指令进行计算，比如有 ffma，mma.sync 以及 wgmma 等。

第二个参数是 MMAThrLayout，代表将 MMA_OP 的线程，按照什么 layout 进行复制。如果使用 sm80 的 mma，一个 mma 指令有 32 个线程，MMAThrLayout = <2,2,1>的话就是在 M 和 N 方向上各复制一份 mma，这样一共就有 128 个线程了。

第三个参数是 Permutations，这个刚开始一直以为只是确定 tiled_mma 的大小的参数，就没怎么在意。但是如果只用来确定大小为什么叫 Permutations（重排），而不是直接叫 Tiler 呢。

```cpp
template <class MMA_Op,
          class MMAThrLayout = Layout<Shape<_1,_1,_1>>,
          class Permutations = Tile<Underscore,Underscore,Underscore>>
CUTE_HOST_DEVICE constexpr
auto
make_tiled_mma(MMA_Op       const&,
               MMAThrLayout const& thr_layout   = {},
               Permutations const& permutations = {})
{
  // Attempt to wrap in an MMA_Atom<> and forward
  return make_tiled_mma(MMA_Atom<MMA_Op>{}, thr_layout, permutations);
}
```

最近研究 cuteDSL 时有了更深的认识，这个参数除了设置 tiled_mma 的大小外，确实还具有重排 tiled_mma 中布局的作用。

以下面的例子为例进行说明。

```cpp
    auto thr_layout = Layout<Shape<_8, _8>, Stride<_8, _1>>{};
    TiledMMA mma = make_tiled_mma(UniversalFMA<float,float,float>{}, thr_layout, Tile<_32, _32, _8>{});
    print_latex(mma);
```

这里定义一个 8*8 的 thread layout，表示一共有 64 个线程。mma op 使用 UniversalFMA，这个 op 只有一个线程，会按照 thr_layout 的布局进行复制。Tile<_32, _32, _8>表示 tiled_mma 最终的大小 MNK 是 32*32*8。这样每个线程还需要在 M 和 N 方向各自处理 4 个元素才能达到这个大小。

打印的结果如下所示，可以看到线程以 8*8 为单元，每个单元里有 64 个线程。每个 M 和 N 方向分别处理 4 个元素。

![](/assets/Permutations/image.png)

上面这种模式下，M 和 N 方向都是线程连续的，每个线程的数据是有间隔的。如果我想让每个线程处理的数据是连续的，线程之间有间隔应该怎么做呢？

这就要用到 Permutations 参数了。通过对 Permutations 参数设置具体的 layout，可以对 tiled_mma 默认的 layout 进行重排。

以 N 维度为例，假设默认的坐标为

```cpp
 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
```

如果想要变成数据连续，线程间隔的模式，就需要做如下调整。

对于 T0 来说，之前的坐标分别是 0 8 16 24，变成数据连续的话新坐标就是 0 1 2 3，因此新老坐标的映射关系如下。

```cpp
old coord: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
new coord: 0               1                     2                       3
```

也就是老坐标的第 0 个位置在新坐标里是第 0 个位置，老坐标的第 8 个位置在新坐标里是第 1 个位置，老坐标的第 16 个位置在新坐标里是第 2 个位置，老坐标的第 24 个位置在新坐标里是第 3 个位置。其他的线程以此类推，因此新老坐标的完整对应关系如下。

```cpp
old coord: 0 1 2  3  4  5  6  7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
new coord: 0 4 8 12 16 20 24 28 1 5  9 13 17 21 25 29  2  6 10 14 18 22 26 30  3  7 11 15 19 23 27 31
```

如果想得到新的坐标，需要进行的 layout 变换是(8,4):(4,1)，因此 Permutations 参数如下：

```cpp
    auto thr_layout = Layout<Shape<_8, _8>, Stride<_8, _1>>{};
    TiledMMA mma = make_tiled_mma(UniversalFMA<float, float, float>{}, thr_layout,
                                  Tile<_32, Layout<Shape<_8, _4>, Stride<_4, _1>>, _8>{});
    print_latex(mma);
```

打印结果如下，可以看到 N 维度上的布局变成数据连续，线程间隔了。

![](/assets/Permutations/image(1).png)

同理，对 M 方向也应用相同的 layout，就可以得到下面的结果。

```cpp
    auto thr_layout = Layout<Shape<_8, _8>, Stride<_8, _1>>{};

    // TiledMMA mma = make_tiled_mma(UniversalFMA<float,float,float>{}, thr_layout, Tile<_32, _32, _8>{});
    TiledMMA mma = make_tiled_mma(UniversalFMA<float, float, float>{}, thr_layout,
                                  Tile<Layout<Shape<_8, _4>, Stride<_4, _1>>,
                                       Layout<Shape<_8, _4>, Stride<_4, _1>>, _8>{});

    print_latex(mma);
```

![](/assets/Permutations/image(2).png)

如果我有 128 个线程，按照 thread_tile = (8, 4), warp_tile = (2, 2), val_per_thread = (2, 4) 的 layout 创建 tiled_mma 应该怎么做。

此时一个 warp 的 32 个线程按照 8*4 排列，4 个 warp 按照 2*2 排列，这样 M 维度一共有 16 个线程，N 维度一共有 8 个线程，但是顺序不是上面那种某一维度连续的了，而是((8,2),(4,2):((4,32),(1,64))这种。

一个线程在 M 维度处理 2 个元素，在 N 维度处理 4 个元素，所以一共会处理 32*32 个元素，因为 print_latex 不能打印太多，所以就设为 32*32。

首先按照下面设置 thr_layout。

```cpp
    auto thr_layout = Layout<Shape<Shape<_8, _2>, Shape<_4, _2>>, Stride<Stride<_4, _32>, Stride<_1, _64>>>{};
    TiledMMA mma = make_tiled_mma(UniversalFMA<float,float,float>{}, thr_layout, Tile<_32, _32, _8>{});
    print_latex(mma);
```

此时打印结果如下。可以看到线程以 16*8 为单元，按照预期的 layout 排列。后面只需要修改 Permutations 把数据弄连续就好了。

![](/assets/Permutations/image(3).png)

代码如下，基本和前面一样。N 维度上一个线程的 4 个数据连续，M 维度上一个线程的 2 个数据连续。

```cpp
    auto thr_layout = Layout<Shape<Shape<_8, _2>, Shape<_4, _2>>, Stride<Stride<_4, _32>, Stride<_1, _64>>>{};

    // TiledMMA mma = make_tiled_mma(UniversalFMA<float,float,float>{}, thr_layout, Tile<_32, _32, _8>{});
    TiledMMA mma = make_tiled_mma(UniversalFMA<float, float, float>{}, thr_layout,
                                  Tile<Layout<Shape<_16, _2>, Stride<_2, _1>>,
                                       Layout<Shape<_8, _4>, Stride<_4, _1>>, _8>{});

    print_latex(mma);
```

![](/assets/Permutations/image(4).png)
