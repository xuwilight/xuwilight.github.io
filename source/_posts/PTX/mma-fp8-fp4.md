---
title: mma FP8/FP4 指令
date: 2026-04-24 15:00:00
tags: [CUDA, PTX, mma, Tensor Core, FP8, FP4, MXFP8, MXFP4, NVFP4, Block Scaling]
categories: [PTX 学习笔记]
description: 深入解析 CUDA PTX mma FP8/FP4 系列指令，涵盖 FP8 (E4M3/E5M2)、MXFP8、FP4 (E2M1)、MXFP4、NVFP4 的数据格式、Block Scaling 机制及 mma 语法详解。
published: true
---

# mma fp8 fp4

mma fp8 mxfp8 fp4 mxfp4 nvfp4

[https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)

[https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

[https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

## FP8

FP8 有 E4M3 和 E5M2 两种形式。

![](/assets/mma-fp8-fp4/image.png)

## MXFP8

MX 是 Microscaling (MX) 的意思。MX 的数据类型都有一个 E8M0 的 scale 数据，每 32 个 blocksize 共享这个 scale 数据。

![](/assets/mma-fp8-fp4/image_1.png)

![](/assets/mma-fp8-fp4/image_2.png)

![](/assets/mma-fp8-fp4/image_3.png)

## FP4

FP4 数据只有 4 位，就是 E2M1。只能表示下面这么多数字。0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6 (same for the negative range).

![](/assets/mma-fp8-fp4/image_4.png)

## MXFP4

MXFP4 也是 E2M1，但是有一个 E8M0 的 scale，被 32 个数据共享。

![](/assets/mma-fp8-fp4/image_5.png)

## NVFP4

NVFP4 也是 E2M1，但是它对应的缩放因子是 E4M3 的 FP8，而且只被 16 个数据共享。

![](/assets/mma-fp8-fp4/image_6.png)

![](/assets/mma-fp8-fp4/image_7.png)

## NVFP4 量化过程

![](/assets/mma-fp8-fp4/image_8.png)

![](/assets/mma-fp8-fp4/image_9.png)

![](/assets/mma-fp8-fp4/image_10.png)

先根据 x 的最大值计算一个从 fp32 或 fp16 到 fp8 的 Senc，然后根据每个 block 的最大值计算从 fp4 到 fp8 的 Sdecb，然后再和 Senc 相乘得到 Sdecbe4m3，然后再与 Sdec 相乘取倒数得到 Sencb，后面就可以用 Sencb 直接把数据从 fp32 量化到 fp4 了。

需要注意的是，Sencb 并不是把数据从 FP8 转到 FP4，它是一个"合一"的缩放因子，直接一步到位地把 FP32 数据映射到 FP4 的量化网格上。

1. 为什么不分两步走（FP32 到 FP8 到 FP4）？

虽然论文里提到了"两级缩放（two-level）"，但那是针对**缩放因子（Scales）的存储方式，而不是针对数据（Data）**的转换路径。

如果分两步，这样会引入两次舍入误差（第一次到 FP8，第二次到 FP4）。

实际做法（公式 4）：直接计算一个综合的 $s_{enc,b}$，让 $x_{fp32}$ 直接一步跳到 FP4。

优点： 只有一次舍入误差，精度更高；同时计算更简单，直接一个乘法搞定。

![](/assets/mma-fp8-fp4/image_11.png)

## MMA 支持

![](/assets/mma-fp8-fp4/image_12.png)

从图上可以看到，

mma 支持 FP8（e4m3,e5m2）的 shape 有 m16n8k32 和 m16n8k16。

mma 支持 FP6 和 FP4 的 shape 有 m16n8k32。指令需要加 kind::f8f6f4。

mma 支持 MXFP8 的 shape 有 m16n8k32，其中 scale 是 ue8m0 格式。指令需要加 kind::mxf8f6f4。

mma 支持 MXFP4 的 shape 有 m16n8k32 和 m16n8k64，其中 scale 是 ue8m0 格式。可以用 kind::mxf8f6f4 或 kind::mxf4nvf4 或 kind::mxf4。

mma 支持 NVFP4 的 shape 有 m16n8k64，其中 scale 是 ue4m3 格式。指令需要用 kind::mxf4nvf4。

![](/assets/mma-fp8-fp4/image_13.png)

普通 FP8 和 FP4 的累加器可以是 fp16 或 fp32，MXFP 只能是 FP32。

### block scaling

如果是 MX 格式的数据需要设置 block scale。从前面的 shape 可以看到，MX 格式的 K 可以是 32 或 64。

![](/assets/mma-fp8-fp4/image_14.png)

![](/assets/mma-fp8-fp4/image_15.png)

从这个图上可以看到，

如果是 mxf8f6f4 的情况下，数据类型支持 mxfp8,mxfp6,mxfp4 的计算，这种情况下只有 m16n8k32 这种 shape 支持，所以 K 是 32，因为 32 个元素一个 scale，所以就只有 1 个 scale。

如果是 mxf4 的情况下，数据类型支持 mxfp4，这种情况下有 m16n8k32 和 m16n8k64 两种 shape，但是只能使用 2X，这是因为指令里只能用 m16n8k64 这个 shape。

如果是 mxf4nvf4 的情况下，数据类型支持 mxfp4 和 nvfp4。如果是 mxfp4，可以是 2X 和 4X。如果是 nvfp4，就只能是 4X。这是因为 nvfp4 只能用 m16n8k64，而且 16 个元素一个 scale，所以需要 4 个 scale。

具体 scale 怎么读取如下：

![](/assets/mma-fp8-fp4/image_16.png)

![](/assets/mma-fp8-fp4/image_17.png)

![](/assets/mma-fp8-fp4/image_18.png)

![](/assets/mma-fp8-fp4/image_19.png)

### mma.m16n8k32

A 矩阵：每个线程包含 4 个 32 位寄存器，每个寄存器有 4 个各种类型的数据。

看起来只有 4 个 fp8 能填满 32 位寄存器，其他的都填不满。

![](/assets/mma-fp8-fp4/image_20.png)

B 矩阵也类似：一个线程包含 2 个 32 位寄存器，每个寄存器有 4 个各种类型的元素。

![](/assets/mma-fp8-fp4/image_21.png)

![](/assets/mma-fp8-fp4/image_22.png)

![](/assets/mma-fp8-fp4/image_23.png)

矩阵 C：对于 fp32 格式，一个线程有 4 个 fp32 寄存器。

![](/assets/mma-fp8-fp4/image_24.png)

### mma.m16n8k64

A 矩阵：一个线程有 4 个 32 位寄存器，每个寄存器包含 8 个 fp4 数据。这样一个线程一共有 32 个 fp4 数据。

![](/assets/mma-fp8-fp4/image_25.png)

![](/assets/mma-fp8-fp4/image_26.png)

B 矩阵：一个线程有两个 32 位寄存器，每个包含 8 个 fp4 元素，一共有 16 个 fp4 元素。

![](/assets/mma-fp8-fp4/image_27.png)

![](/assets/mma-fp8-fp4/image_28.png)

![](/assets/mma-fp8-fp4/image_29.png)

累加器 C：一个线程有 4 个 fp32 寄存器，共包含 4 个元素。

![](/assets/mma-fp8-fp4/image_30.png)

![](/assets/mma-fp8-fp4/image_31.png)

## mma 语法

Alternate floating point type

只需要看后两个。

mma.sync.aligned.shape.row.col.dtype.f8type.f8type.ctype 是两个 fp8 数据相乘。支持的 shape 是 .m16n8k16, .m16n8k32，支持的 ctype 和 dtype 是 fp16 和 fp32。

最后一个应该是支持任意的 fp8、fp6 和 fp4 相乘，此时 shape 只能是 m16n8k32。

```cpp
mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32        d, a, b, c;
mma.sync.aligned.m16n8k8.row.col.f32.atype.btype.f32      d, a, b, c;
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32       d, a, b, c;
mma.sync.aligned.shape.row.col.dtype.f8type.f8type.ctype  d, a, b, c;
mma.sync.aligned.m16n8k32.row.col.kind.dtype.f8f6f4type.f8f6f4type.ctype d, a, b, c;

.atype      = {.bf16, .tf32};
.btype      = {.bf16, .tf32};
.f8type     = {.e4m3, .e5m2};
.f8f6f4type = {.e4m3, .e5m2, .e3m2, .e2m3, .e2m1};
.ctype      = {.f16, .f32};
.dtype      = {.f16, .f32};
.shape      = {.m16n8k16, .m16n8k32};
.kind       = {.kind::f8f6f4};
```

Alternate floating point type with block scaling

第一个是 m16n8k64 对应的指令。这种情况下 kind 只能是 mxf4，scale_vec_size 只能是 2X，这是因为 mxf4 的 scale 是 32 个元素共享的，而 K 等于 64，所以需要两个 scale。stype 固定是 e8m0。

第二个也是 m16n8k64 对应的指令。这种情况下 kind 是 mxf4nvf4，scale_vec_size 可以是 2X 或 4X，stype 可以是.ue8m0, .ue4m3。这种应该是支持 mxf4 相乘，nvf4 相乘，不知道支不支持两两相乘。而且感觉相乘的时候 scale_vec_size 和 stype 也不能乱设置。

第三个是 m16n8k32 的指令。这种情况下应该是支持 mxf8,mxf6,mxf4 之间的运算，不知道支不支持互相乘。

```cpp
mma.sync.aligned.m16n8k64.row.col.kind.block_scale{.scale_vec_size}.f32.e2m1.e2m1.f32.stype d, a, b, c, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};

.kind           = {.kind::mxf4};
.scale_vec_size = {.scale_vec::2X};
.stype          = {.ue8m0};

mma.sync.aligned.m16n8k64.row.col.kind.block_scale.scale_vec_size.f32.e2m1.e2m1.f32.stype d, a, b, c, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};

.kind           = {.kind::mxf4nvf4};
.scale_vec_size = {.scale_vec::2X, .scale_vec::4X};
.stype          = {.ue8m0, .ue4m3};

mma.sync.aligned.m16n8k32.row.col.kind.block_scale{.scale_vec_size}.f32.f8f6f4type.f8f6f4type.f32.stype d, a, b, c, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};

.kind           = {.kind::mxf8f6f4};
.scale_vec_size = {.scale_vec::1X};
.f8f6f4type     = {.e4m3, .e5m2, .e3m2, .e2m3, .e2m1};
.stype          = {.ue8m0};
```

描述

执行 MxNxK 矩阵乘累加运算，即 D = A*B+C，其中矩阵 A 的大小为 MxK，矩阵 B 的大小为 KxN，矩阵 C 和 D 的大小均为 MxN。

限定符 `.block_scale` 指定在执行矩阵乘累加运算之前，分别使用 scale_A 和 scale_B 矩阵对矩阵 A 和 B 进行缩放，具体说明见"mma.sync 的块缩放"一节。scale_A 和 scale_B 矩阵中每个元素对应的数据类型由 `.stype` 指定。限定符 `.scale_vec_size` 指定 scale_A 矩阵的列数以及 scale_B 矩阵的行数。

`.kind`、`.stype` 和 `.scale_vec_size` 的有效组合见表 36。对于带有 `.kind::mxf4` 的 mma 指令，若未指定 `.scale_vec_size`，则默认值为 2X。相比之下，当 `.kind` 指定为 `.kind::mxf8f6f4` 时，限定符 `.scale_vec_size` 的默认值为 1X。然而，对于 `.kind::mxf4nvf4`，必须提供有效的 `.scale_vec_size`。

执行 `mma.sync.m8n8k4` 指令的一个线程束会计算 4 次矩阵乘累加运算。其余 `mma.sync` 运算每个线程束仅计算一次矩阵乘累加运算。

对于单比特 mma.sync，乘法被替换为一系列逻辑运算：具体而言，`mma.xor.popc` 和 `mma.and.popc` 分别计算 A 的 k 比特行与 B 的 k 比特列的异或（XOR）和与（AND），然后计算结果中置位比特的数量（popc）。该结果被加到 C 的对应元素上，并写入 D。

操作数 a 和 b 表示两个被乘数矩阵 A 和 B，而 c 和 d 表示累加器矩阵和目标矩阵，它们分布在线程束中的各线程上。当指定了 `.block_scale` 限定符时，操作数 scale-a-data 和 scale-b-data 分别表示对应于 scale_A 和 scale_B 矩阵的缩放矩阵元数据。元组 {byte-id-a, thread-id-a} 和 {byte-id-b, thread-id-b} 分别表示从对应元数据参数 scale-a-data 和 scale-b-data 中选择 scale_A 和 scale_B 矩阵的选择器。操作数 scale-a-data 和 scale-b-data 的类型为 `.b32`。操作数 byte-id-a、thread-id-a、byte-id-b、thread-id-b 为无符号 16 位整数值。关于选择器参数的更多详情，请参见"mma.sync 的块缩放"一节。

每个线程中的寄存器持有矩阵的一个片段，具体描述见"使用 mma 指令进行矩阵乘累加操作"一节。

限定符 `.dtype`、`.atype`、`.btype` 和 `.ctype` 分别表示矩阵 D、A、B 和 C 中元素的数据类型。限定符 `.stype` 表示 scale_A 和 scale_B 矩阵中元素的数据类型。特定形状具有类型限制：

- `.m8n8k4`：当 `.ctype` 为 `.f32` 时，`.dtype` 也必须为 `.f32`。

- `.m16n8k8`：

 - `.dtype` 必须与 `.ctype` 相同。

 - `.atype` 必须与 `.btype` 相同。

- `.m16n8k16` 和 `.m16n8k32`：

 - `.dtype` 必须与 `.ctype` 相同。

限定符 `.alayout` 和 `.blayout` 分别表示矩阵 A 和 B 的行主序或列主序布局。

当 `.kind` 为 `.kind::mxf8f6f4` 或 `.kind::f8f6f4` 时，各个 4 位和 6 位浮点类型的元素必须打包在一个 8 位容器中。类型为 `.e2m1` 的矩阵元素位于 8 位容器的中间 4 位，容器的高 2 位和低 2 位为填充位。当矩阵元素类型为 `.e3m2` 或 `.e2m3` 时，矩阵元素位于 8 位容器的低 6 位，高 2 位为填充位。相比之下，请注意，当使用 `.kind::mxf4` 或 `.kind::mxf4nvf4` 的 mma 指令时，即使矩阵元素类型为 `.e2m1`，也无需显式填充。

![](/assets/mma-fp8-fp4/image_32.png)

**Examples of mma with block scale**

第一个 mxf4 相乘，scale_vec 默认是 2X。

第二个是 nvf4 相乘。

第三个没看懂，scale 用的是 e8m0，所以应该是 mxf4，但是 scale_vec 是 4X，表示有 4 个 scale，那不是超过 64 了吗。

```cpp
 .reg .b32 %Ra<4>, %Rb<4>;
 .reg .f32 %Rc<4>, %Rd<4>;
 .reg .b32 scaleAData, scaleBData;
 mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0
   {%Rd0, %Rd1, %Rd2, %Rd3},
   {%Ra0, %Ra1, %Ra2, %Ra3},
   {%Rb0, %Rb1},
   {%Rc0, %Rc1, %Rc2, %Rc3},
   scaleAData, {2, 1}, scaleBData, {2, 3};

 .reg .b32 %Ra<4>, %Rb<4>;
 .reg .f32 %Rc<4>, %Rd<4>;
 .reg .b32 scaleAData, scaleBData;
 .reg .u16 bidA, bidB, tidA, tidB;
 mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
   {%Rd0, %Rd1, %Rd2, %Rd3},
   {%Ra0, %Ra1, %Ra2, %Ra3},
   {%Rb0, %Rb1},
   {%Rc0, %Rc1, %Rc2, %Rc3},
   scaleAData, {bidA, tidA}, scaleBData, {bidB, tidB};

.reg .b32 %Ra<4>, %Rb<4>;
.reg .f32 %Rc<4>, %Rd<4>;
.reg .b32 scaleAData, scaleBData;
.reg .u16 bidA, bidB, tidA, tidB;
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0
   {%Rd0, %Rd1, %Rd2, %Rd3},
   {%Ra0, %Ra1, %Ra2, %Ra3},
   {%Rb0, %Rb1},
   {%Rc0, %Rc1, %Rc2, %Rc3},
   scaleAData, {bidA, tidA}, scaleBData, {bidB, tidB};

.reg .b32 %Ra<4>, %Rb<4>;
.reg .f32 %Rc<4>, %Rd<4>;
.reg .b32 scaleAData, scaleBData;
mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e3m2.e2m1.f32.ue8m0
  {%Rd0, %Rd1, %Rd2, %Rd3},
  {%Ra0, %Ra1, %Ra2, %Ra3},
  {%Rb0, %Rb1},
  {%Rc0, %Rc1, %Rc2, %Rc3},
  scaleAData, {0, 1}, scaleBData, {0, 1};

.reg .b32 %Ra<4>, %Rb<4>;
.reg .f32 %Rc<4>, %Rd<4>;
.reg .b32 scaleAData, scaleBData;
mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e5m2.f32.ue8m0
  {%Rd0, %Rd1, %Rd2, %Rd3},
  {%Ra0, %Ra1, %Ra2,  %Ra3},
  {%Rb0, %Rb1},
  {%Rc0, %Rc1, %Rc2, %Rc3},
  scaleAData, {0, 1}, scaleBData, {0, 0};
```

## 总结：

### 普通 fp8 计算

* mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32
* mma.sync.aligned.m16n8k16.row.col.f32.e5m2.e4m3.f32

### 普通 fp6fp4 计算

* mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e2m3.f32

### mxfp8 计算

* mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e5m2.f32.ue8m0

### mxfp4 计算

* mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e3m2.e2m1.f32.ue8m0
* mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0
* mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0

### nvfp4 计算

* mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3


