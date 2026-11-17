---
title: Cutlass 学习笔记（四）Efficient GEMM
date: 2025-08-18 12:00:00
tags: [CUTLASS]
categories: [Cutlass 学习笔记,Cutlass]
description: 深入解析 CUTLASS GEMM 的分层结构，从线程块级、warp 级到指令级的分块策略与并行编程模型，并介绍尾声、软件流水线、线程块光栅化、分块归约和 Hopper warp 专用等优化手段。
---

## 分层结构

基本的矩阵乘法三重循环嵌套可以分块、分 tile 执行，以匹配硬件并发、内存局部性和并行编程模型。CUTLASS 把 GEMM 映射到 NVIDIA GPU 时采用如下循环嵌套结构。

```c++
for (int cta_n = 0; cta_n < GemmN; cta_n += CtaTileN) {                     // for each threadblock_y           } threadblock-level concurrency
  for (int cta_m = 0; cta_m < GemmM; cta_m += CtaTileM) {                   //    for each threadblock_x        }

    for (int cta_k = 0; cta_k < GemmK; cta_k += CtaTileK) {                 //       "GEMM mainloop" - no unrolling
                                                                            //                       - one iteration of this loop is one "stage"
                                                                            //
      for (int warp_n = 0; warp_n < CtaTileN; warp_n += WarpTileN) {        // for each warp_y                  } warp-level parallelism
        for (int warp_m = 0; warp_m < CtaTileM; warp_m += WarpTileM) {      //    for each warp_x               }
                                                                            //
          for (int warp_k = 0; warp_k < CtaTileK; warp_k += WarpTileK) {         //       fully unroll across CtaTileK
                                                                            //         - one iteration of this loop is one "k Group"
                                                                            //
            for (int mma_k = 0; mma_k < WarpTileK; mma_k += MmaK) {         // for each mma instruction         } instruction-level parallelism
              for (int mma_n = 0; mma_n < WarpTileN; mma_n += MmaN) {       //    for each mma instruction      }
                for (int mma_m = 0; mma_m < WarpTileM; mma_m += MmaM) {     //        for each mma instruction  }
                                                                            //
                  mma_instruction(d, a, b, c);                              //            TensorCore matrix computation

                }   // for mma_m
              }   // for mma_n
            }   // for mma_k

          }   // for warp_k
        }   // for warp_m
      }   // for warp_n

    }   // for cta_k
  }   // for cta_m
}   // for cta_n
```

这种分 tile 的循环嵌套针对线程块、warp 以及 CUDA 与 Tensor Core 之间的并发，并利用共享内存和寄存器中的内存局部性。

下图展示了该结构中的数据流，这是 CUTLASS 所实现的层级式 GEMM 计算。每一级都描绘了一个嵌套的分块层级，对应 CUDA 执行模型中的一个并发层级和内存层次结构中的一个层级，从左到右越来越细。

![](/assets/efficient_gemm/image.png)

### 线程块级 GEMM

每个线程块迭代地加载输入矩阵的分块并计算累积乘积，以此完成它在输出 GEMM 中的份额。在线程块这一级，数据从全局内存加载。分块策略通常是效率的关键，但程序员要在多个相互冲突的目标间权衡。更大的线程块意味着更少的全局内存读取，从而保证 DRAM 带宽不成为瓶颈；然而大的线程块 tile 可能与问题维度并不匹配。如果 GEMM 的 M 或 N 维度较小，线程块就可能部分落在问题边界之外，使其中部分线程无法做有意义的工作。如果 M、N 都小、K 很大，这种方案启动的线程块可能相对较少，无法充分利用 GPU 的所有多处理器。针对这种情况，如并行归约一节所述，优化的做法是把 K 维度划分到多个线程块或多个 warp 上，并行计算矩阵乘积，再对乘积做归约得到结果。

在 CUTLASS 中，线程块 tile 的尺寸由 ThreadblockShape::{kM, kN, kK} 指定，可以针对目标处理器和 GEMM 问题维度进行调优。

### warp 级 GEMM

warp 级 GEMM 对应 CUDA 执行模型中的 warp 级并行。线程块内的多个 warp 从共享内存取数据到寄存器并执行计算。warp 级 GEMM 既可以用 Tensor Core 发出 mma.sync 或 wmma 指令实现，也可以用面向 CUDA Core 的 thread 级矩阵计算指令实现。为了获得最佳性能，共享内存访问应当无 bank 冲突；为了最大化 warp 内的数据复用，应选择较大的 warp 级 GEMM tile。

### thread 级 GEMM

在分块的最底层，每个线程负责处理一定数量的元素。线程之间无法访问彼此的寄存器，因此需要选择一种组织方式，让寄存器中的值能被多条数学指令复用。这带来线程内部的二维分块结构，每个线程向 CUDA Core 发出一系列独立的数学指令，计算一个累积外积。

SGEMM、IGEMM、HGEMM 和 DGEMM 由 thread 级矩阵乘过程发出的 SIMT 数学指令计算得出。

## 尾声

上面的代码只关注矩阵乘法 C = AB，其结果保存在线程块内每个线程的寄存器中。把输出 tile 的逻辑元素映射到各线程的方式是为了最大化矩阵乘性能，并不能带来高效、合并的全局内存读写。

尾声（epilogue）是一个独立阶段，线程先通过共享内存交换数据，再用高效的条带访问模式协作访问全局内存。在这个阶段，还可以方便地以矩阵乘积结果为输入，计算线性缩放和其他逐元素操作。

CUTLASS 定义了一些典型的尾声操作，例如线性缩放和钳制，但也可以使用其他设备端函数调用算子执行自定义操作。

## 优化

上述层级结构能高效映射到 NVIDIA GPU 的 CUDA 执行模型和 CUDA/Tensor Core 上。下面几节介绍在设计空间各个角落取得峰值性能的策略，尽可能最大化并行并利用数据局部性。

### 软件流水线

这种分块结构要求每个 CUDA 线程在寄存器中分配大量存储，累加器元素通常至少占一个线程寄存器预算的一半。因此与其它类别的 GPU 负载相比，占用率（并发的线程、warp 和线程块数量）相对较低，这限制了 GPU 通过在 SM 内切换到其他并发线程来隐藏内存延迟和其他停顿的能力。

为缓解内存延迟的影响，CUTLASS 用软件流水线把内存访问与线程内的其他计算重叠起来，具体通过在以下作用域做双缓冲实现。

- 线程块级共享内存 tile。在共享内存中分配两个 tile，一个用于加载当前矩阵运算的数据，另一个用于缓冲从全局内存加载、供下一次主循环迭代使用的数据。
- warp 级矩阵片段。在寄存器中分配两个片段，一个在当前矩阵计算期间交给 CUDA 与 Tensor Core，另一个用于接收共享内存取回的数据，供下一次 warp 级矩阵运算使用。

下图展示了 CUTLASS GEMM 中高效、流水化的主循环体。

![](/assets/efficient_gemm/image(1).png)

### 线程块光栅化

为了最大化利用 L2 缓存中的数据，CUTLASS 定义了若干函数来影响线程块到 GEMM 问题逻辑分区的映射。它们把连续启动的线程块映射到分区问题的二维紧密区域，提高这些线程块在大致相同时间访问同一批全局内存 tile 的概率。相关定义位于 cutlass/gemm/threadblock/threadblock_swizzle.h。

### 并行归约

#### Split K 跨线程块归约

矩阵乘积计算在 O(MN) 个独立内积之间暴露了并行性。对足够大的问题，CUTLASS 的 GEMM kernel 可以接近理论最大计算吞吐；但对于小问题，线程块太少，无法有效占满整个 GPU。

作为一种补救，把内积计算中的归约并行化，可以让更多线程块并发执行，同时仍享受大线程块级 GEMM tile 带来的吞吐优势。CUTLASS 通过划分 GEMM 的 K 维度、为每个分区额外启动一组线程块来实现跨线程块的并行归约，这种策略在内称为"并行归约 splitK"。它需要执行两个 kernel：partitionedK GEMM 和批处理归约。

PartitionedK GEMM 类似一种批处理步长 GEMM 的变体。它不要求用户给出每个批次的问题规模，而是给出总体问题规模和沿 K 维度的分区数。例如 m=128、n=128、k=4096、partition=16，会得到 16 个批处理步长 GEMM，每个批次为 m=128、n=128、k=256。PartitionedK 也允许 k 不能被分区数整除。例如 m=128、n=128、k=4096、partition=20，会得到 20 个批处理步长 GEMM，前 19 个批次为 m=128、n=128、k=4096/20=204，最后一个批次为 m=128、n=128、k=220。批处理归约 kernel 以 partitionedK GEMM 的输出 C 为输入，沿 K 维度做归约，用户需要管理工作区内存来存放这个中间结果。

#### Sliced K 跨 warp 归约

与 split-k 类似，sliced-k 旨在提升 M、N 维度较小而 K 维度较大时的 kernel 效率。在线程块一级，CtaTileN 和 CtaTileM 通过把工作划分给不同 warp 来体现并行性。更大的 warpTile 能带来更好的指令级并行（ILP）和复用，但也会限制每个线程块内运行的 warp 数量，从而降低效率。

为提升这类场景的效率，除了 ctaTileK 之外再对 warpTile 做划分，能让更多 warp 在 CTA 中并发运行，更充分地利用硬件。sliced-k kernel 不仅在 CtaTileN、CtaTileM 维度，也在 CtaTileK 维度上划分线程块的计算。因此 sliced-k 会带来一点开销，需要在最后对参与的各 warp 做归约，因为每个 warp 只用 CtaTileK 的一个切片计算，归约前只持有部分和。

### Hopper warp 专用

下面关于 warp 专用的内容针对 Hopper kernel 设计。Blackwell SM100 kernel 的 warp 专用结构与 Hopper 截然不同，但把生产者与消费者分离的概念仍然适用。

从 Hopper 开始，CUTLASS 3.0 把 warp 专用（Warp Specialization）融入 kernel 设计。一个线程块被划分为两组 warp：生产者 warp 组和消费者 warp 组。生产者 warp 组使用新的张量内存加速器（TMA）把数据从全局内存加载到共享内存缓冲区。

生产者 warp 组（DMA）等待消费者 warp 组通过新增的异步 pipeline 类把共享内存缓冲区标记为空；数据写入共享内存后，TMA 会更新与该 stage 关联的 barrier，通知相应线程缓冲区已填满。消费者 warp 组（MMA）则等待生产者组发出缓冲区已填满的信号，然后发起 Tensor Core MMA 操作，最后释放缓冲区，以便下一批 TMA 加载进行。

#### Warp 专用持久化 Cooperative kernel 设计

从 Hopper 开始引入的另一种 warp 专用 kernel 设计是 Warp-Specialized Persistent Cooperative kernel。与 warp 专用 kernel 一样，cooperative 设计保留 warp 组以及 warp 组之间 barrier 同步的概念，其显著特点是：

1. 启动持久化线程块占满 GPU 架构中的所有 SM（数量见 KernelHardwareInfo 结构体）。这些持久化线程块负责给输出分 tile，在其生命周期内可能计算多个输出 tile，主要好处是分摊线程块启动和 kernel 序言（prologue）的开销。
2. 两个消费者 warp 组协作处理同一个输出 tile，方法是把 tile 沿 M 维度对半分。这样每个消费者 warp 组的寄存器压力降低，因此可以用更大的 tile 尺寸，从而提升性能。

由于每个线程块现在要计算多个输出 tile，grid 启动的形状以及 tile 到线程块的调度由新的 tile 调度器管理。它会考虑集群形状和可用 SM 数量，计算出输出 tile 到已启动线程块的有效调度方案。

#### Warp 专用持久化 Ping-Pong kernel 设计

第三种 kernel 设计是 Warp-Specialized Persistent Ping-Pong kernel。与持久化 cooperative 类似，ping-pong 设计保留 warp 组、warp 组之间的 barrier 同步以及 grid 启动形状，其独特之处在于：

1. 两个消费者 warp 组通过 tile 调度器分配到不同的输出 tile。这让一个消费者 warp 组的尾声可以与另一个消费者 warp 组的数学运算重叠，从而最大化 Tensor Core 利用率。
2. 生产者 warp 组使用有序序列屏障（Ordered Sequence Barrier）同步，按顺序依次填充两个消费者 warp 组的缓冲区。
