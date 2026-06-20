---
title: mma FP8/FP4 指令
date: 2026-06-20 15:00:00
tags: [CUDA, PTX, mma, GPU, Tensor Core, FP8, FP4, MXFP8, MXFP4, NVFP4, Block Scaling]
categories: [PTX 学习笔记]
description: 深入解析 CUDA PTX mma FP8/FP4 系列指令，涵盖 FP8 (E4M3/E5M2)、MXFP8、FP4 (E2M1)、MXFP4、NVFP4 的数据格式、Block Scaling 机制及 mma 语法详解。
published: false
---

# mma fp8 fp4

mma fp8 mxfp8 fp4 mxfp4 nvfp4

[https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)

[https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

[https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

# FP8

FP8 有 E4M3 和E5M2两种形式。

![](/assets/mma-fp8-fp4/image.png)

# MXFP8

MX是Microscaling (MX)的意思。MX的数据类型都有一个E8M0的scale数据，每32个blocksize共享这个scale数据。

![](/assets/mma-fp8-fp4/image_1.png)

![](/assets/mma-fp8-fp4/image_2.png)

![](/assets/mma-fp8-fp4/image_3.png)

# FP4

FP4数据只有4位，就是E2M1。只能表示下面这么多数字。 0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6 (same for the negative range). 

![](/assets/mma-fp8-fp4/image_4.png)

# MXFP4

MXFP4也是E2M1，但是有一个E8M0的scale，被32个数据共享。

![](/assets/mma-fp8-fp4/image_5.png)

# NVFP4

NVFP4也是E2M1，但是它对应的缩放因子是E4M3的FP8，而且只被16个数据共享。

![](/assets/mma-fp8-fp4/image_6.png)

![](/assets/mma-fp8-fp4/image_7.png)

# NVFP4量化过程

![](/assets/mma-fp8-fp4/image_8.png)

![](/assets/mma-fp8-fp4/image_9.png)

![](/assets/mma-fp8-fp4/image_10.png)

先根据x的最大值计算一个从fp32或fp16到fp8的Senc，然后根据每个block的最大值计算从fp4到fp8的Sdecb，然后再和Senc相乘得到Sdecbe4m3，然后再与Sdec相乘取到数得到Sencb，后面就可以用Sencb直接把数据从fp32量化到fp4了。

需要注意的是，Sencb 并不是把数据从 FP8 转到 FP4，它是一个"合一"的缩放因子，直接一步到位地把 FP32 数据映射到 FP4 的量化网格上。

1. 为什么不分两步走（FP32到FP8到FP4）？

虽然论文里提到了"两级缩放（two-level）"，但那是针对**缩放因子（Scales）的存储方式，而不是针对数据（Data）**的转换路径。

如果分两步 这样会引入两次舍入误差（第一次到 FP8，第二次到 FP4）。

实际做法（公式4）： 直接计算一个综合的 $s_{enc,b}$，让 $x_{fp32}$ 直接一步跳到 FP4。

优点： 只有一次舍入误差，精度更高；同时计算更简单，直接一个乘法搞定。

![](/assets/mma-fp8-fp4/image_11.png)

# MMA支持

![](/assets/mma-fp8-fp4/image_12.png)

从图上可以看到，

mma支持FP8（e4m3,e5m2）的shape有m16n8k32和m16n8k16。

mma支持FP6和FP4的shape有m16n8k32。指令需要加kind::f8f6f4

mma支持MXFP8的shape有m16n8k32，其中scale是ue8m0格式。指令需要加kind::mxf8f6f4

mma支持MXFP4的shape有m16n8k32和m16n8k64，其中scale是ue8m0格式。可以用kind::mxf8f6f4或kind::mxf4nvf4或kind::mxf4

mma支持NVFP4的shape有m16n8k64，其中scale是ue4m3格式。指令需要用kind::mxf4nvf4

![](/assets/mma-fp8-fp4/image_13.png)

普通FP8和FP4的累加器可以是fp16或fp32，MXFP只能是FP32。

## block scaling

如果是MX格式的数据需要设置block scale。从前面的shape可以看到，MX格式的K可以是32或64。

![](/assets/mma-fp8-fp4/image_14.png)

![](/assets/mma-fp8-fp4/image_15.png)

从这个图上可以看到，

如果是mxf8f6f4的情况下，数据类型支持mxfp8,mxfp6,mxfp4的计算，这种情况下只有m16m8k32这种shape支持，所以K是32，因为32个元素一个scale，所以就只有1个scale。

如果是mxf4的情况下，数据类型支持mxfp4，这种情况下有m16n8k32和m16n8k64两种shape，但是只能使用2X，这是因为指令里只能用m16n8k64这个shape。

如果是mxf4nvf4的情况下，数据类型支持mxfp4和nvfp4,。如果是mxfp4，可以是2X和4X。如果是nvfp4,就只能是4X。这是因为nvfp4只能用m16n8k64，而且16个元素一个scale，所以需要4个scale。

具体scale怎么读取如下：

![](/assets/mma-fp8-fp4/image_16.png)

![](/assets/mma-fp8-fp4/image_17.png)

![](/assets/mma-fp8-fp4/image_18.png)

![](/assets/mma-fp8-fp4/image_19.png)

## mma.m16n8k32

A矩阵：每个线程包含4个32为寄存器，每个寄存器有4个各种类型的数据。

看起来只有4个fp8能填满32为寄存器，其他的都填不满。

![](/assets/mma-fp8-fp4/image_20.png)

B矩阵也类似：一个线程包含2个32位寄存器，每个寄存器有4个各种类型的元素。

![](/assets/mma-fp8-fp4/image_21.png)

![](/assets/mma-fp8-fp4/image_22.png)

![](/assets/mma-fp8-fp4/image_23.png)

矩阵C：对于fp32格式，一个线程有4个fp32寄存器。

![](/assets/mma-fp8-fp4/image_24.png)

## mma.m16n8k64

A矩阵：一个线程有4个32位寄存器，每个寄存器包含8个fp4数据。这样一个线程一共有32个fp4数据。

![](/assets/mma-fp8-fp4/image_25.png)

![](/assets/mma-fp8-fp4/image_26.png)

B矩阵：一个线程有两个32位寄存器，每个包含8个fp4元素，一共有16个fp4元素。

![](/assets/mma-fp8-fp4/image_27.png)

![](/assets/mma-fp8-fp4/image_28.png)

![](/assets/mma-fp8-fp4/image_29.png)

累加器C：一个线程有4个fp32寄存器，共包含4个元素。

![](/assets/mma-fp8-fp4/image_30.png)

![](/assets/mma-fp8-fp4/image_31.png)

## mma语法

Alternate floating point type

只需要看后两个

mma.sync.aligned.shape.row.col.dtype.f8type.f8type.ctype 是两个fp8数据相乘。支持的shape是.m16n8k16, .m16n8k32，支持的ctype和dtype是fp16和fp32。

最后一个应该是支持任意的fp8fp6和fp4相乘，此时shape只能是m16n8k32。

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

第一个是m16n8k64对应的指令。这种情况下kind只能是mxf4，scale_vec_size只能是2X，这是因为mxf4的scale是32个元素共享的，因为K等于64，所以需要两个scale。stype固定是e8m0。

第二个也是m16n8k64对应的指令。这种情况下kind是mxf4nvf4，scale_vec_size可以是2X或4X，stype可以是.ue8m0, .ue4m3。这种应该是支持mxf4相乘，nvf4相乘，不知道支不支持两两相乘。而且感觉相乘的时候scale_vec_size 和stype 也不能乱设置。

第三个是m16n8k32的指令。这种情况下应该是支持mxf8,mxf6,mxf4之间的运算，不知道支不支持互相乘。

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

第一个mxf4相乘，scale_vec默认是X2。

第二个是nvf4相乘。

第三个没看懂，scale用的是e8m0，所以应该是mxf4，但是scale_vec是4X，表示有4个scale，那不是超过64了吗。

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

总结：

普通fp8计算

mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32

mma.sync.aligned.m16n8k16.row.col.f32.e5m2.e4m3.f32

普通fp6fp4计算

mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e2m3.f32

mxfp8计算

mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e5m2.f32.ue8m0

mxfp4计算

mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e3m2.e2m1.f32.ue8m0 ？

mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0 ？

mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0

nvfp4计算

mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3

sm120 mma fp8

sm120 mma fp4

sm120 mma mxfp8

sm120 mma mxfp4

sm120 mma nvfp4
