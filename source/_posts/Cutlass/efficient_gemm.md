---
title: Cutlass 学习笔记（五）Efficient GEMM
date: 2025-08-19 12:00:00
tags: [CUTLASS, GEMM]
categories: [Cutlass 学习笔记,Cutlass]
description: 深入解析 CUTLASS GEMM 的分层结构，从线程块级、warp 级到指令级的分块策略与并行编程模型。
---

## Hierarchical Structure

基本的矩阵乘法三重循环嵌套计算可以分块和分块执行，以匹配硬件并发性、内存局部性和并行编程模型。在 CUTLASS 中，GEMM 被映射到 NVIDIA GPU，其结构如下循环嵌套所示。

```python
for (int cta_n = 0; cta_n < GemmN; cta_n += CtaTileN) {                     // for each threadblock_y           } threadblock-level concurrency
  for (int cta_m = 0; cta_m < GemmM; cta_m += CtaTileM) {                   //    for each threadblock_x        }

    for (int cta_k = 0; cta_k < GemmK; cta_k += CtaTileK) {                 //       "GEMM mainloop" - no unrolling
                                                                            //                       - one iteration of this loop is one "stage"
                                                                            //
      for (int warp_n = 0; warp_n < CtaTileN; warp_n += WarpTileN) {        // for each warp_y                  } warp-level parallelism
        for (int warp_m = 0; warp_m < CtaTileM; warp_m += WarpTileM) {      //    for each warp_x               }
                                                                            //
          for (int warp_k = 0; warp_k < CtaTileK; warp_k += WarpTileK) {         //       fully unroll across CtaTileK
                                                                            //         - one iteration of this loop is one "k Group"
                                                                            //
            for (int mma_k = 0; mma_k < WarpTileK; mma_k += MmaK) {         // for each mma instruction         } instruction-level parallelism
              for (int mma_n = 0; mma_n < WarpTileN; mma_n += MmaN) {       //    for each mma instruction      }
                for (int mma_m = 0; mma_m < WarpTileM; mma_m += MmaM) {     //        for each mma instruction  }
                                                                            //
                  mma_instruction(d, a, b, c);                              //            TensorCore matrix computation

                }   // for mma_m
              }   // for mma_n
            }   // for mma_k

          }   // for warp_k
        }   // for warp_m
      }   // for warp_n

    }   // for cta_k
  }   // for cta_m
}   // for cta_n
```

这种平铺循环嵌套旨在提高 threadblock，warps 和 CUDA Tensor Core 的并发性，并且可以在寄存器和共享内存中很好的利用内存的局部性。

下图展示了该结构中的数据流。这是 CUTLASS 所实现的层级式 GEMM 计算。每个阶段都描绘了一个嵌套的平铺层级，对应于 CUDA 执行模型中的一个并发层级以及内存层次结构中的一个层级，从左到右层级越来越精细。

![](/assets/efficient_gemm/image.png)

### Threadblock-level GEMM

每个线程块通过迭代加载输入矩阵的分块并计算累积矩阵乘积来计算其在输出 GEMM 中的份额。在线程块级别，数据从全局内存加载。通常，分块策略是提高效率的关键。然而，程序员必须平衡多个相互冲突的目标。更大的线程块意味着更少的全局内存读取操作，从而确保 DRAM 带宽不会成为瓶颈。但是，较大的线程块可能与问题的维度不匹配。如果 GEMM 的 M 或 N 维度较小，则线程块中的某些线程可能无法执行有意义的工作，因为线程块可能部分超出问题的边界。如果 M 和 N 都很小而 K 很大，则此方案可能启动的线程块相对较少，并且无法充分利用 GPU 中的所有多处理器。针对这种情况，如“并行归约”部分所述，优化性能的策略是将 GEMM 的 K 维度划分到多个线程块或多个线程束中。这些线程块或线程束并行计算矩阵乘积；然后对乘积进行归约以计算结果。

在 CUTLASS 中，线程块单元的尺寸指定为 ThreadblockShape::{kM, kN, kK}，并且可以进行调整，以使 GEMM 计算更适合目标处理器和 GEMM 问题的维度。

### Warp-level GEMM

warp 级 GEMM 映射到 CUDA 执行模型中的 warp 级并行性。线程块内的多个 warp 从共享内存中获取数据到寄存器并执行计算。warp 级 GEMM 可以通过 TensorCores 发出 mma.sync 或 wmma 指令来实现，也可以通过向 CUDA 核心发出线程级矩阵计算指令来实现。为了获得最佳性能，对共享内存的访问应避免 bank 冲突。为了最大限度地提高 warp 内的数据重用率，应选择较大的 warp 级 GEMM tile。

### Thread-level GEMM

在分块的最底层，每个线程负责处理一定数量的元素。线程之间无法访问彼此的寄存器，因此我们选择了一种允许多个数学指令重用寄存器中存储的值的组织方式。这导致线程内部形成一个二维的平铺结构，其中每个线程向 CUDA 核心发出一系列独立的数学指令，并计算累积的外积。

SGEMM、IGEMM、HGEMM 和 DGEMM 由线程级矩阵乘法过程发出的 SIMT 数学指令计算得出。

## Epilogue

上述代码仅关注矩阵乘法运算 C = AB，其结果保存在线程块内每个线程的寄存器中。输出块中逻辑元素到每个线程的映射旨在最大化矩阵乘法运算的性能，但并未实现高效的全局内存加载和存储操作。

尾声阶段是一个独立的阶段，在此阶段，线程通过共享内存交换数据，然后使用高效的访问模式协作访问全局内存。此外，在此阶段，可以使用矩阵乘法结果作为输入，方便地计算线性缩放和其他逐元素运算。

CUTLASS 定义了一些典型的尾声阶段操作，例如线性缩放和 clamp，但也可以使用其他设备端函数调用运算符来执行自定义操作。

## Optimizations

上述层级结构能够高效地映射到 NVIDIA GPU 中的 CUDA 执行模型和 CUDA/TensorCores。以下章节将介绍如何在设计空间的各个角落获得最佳性能，最大限度地提高并行性，并尽可能利用数据局部性。

### Pipelining

这种 block 结构需要在每个 CUDA 线程的寄存器中分配大量的存储空间。累加器元素通常至少占用线程总寄存器预算的一半。因此，与其他类型的 GPU 工作负载相比，其占用率（即并发线程、线程束和线程块的数量）相对较低。这限制了 GPU 通过在 SM 内切换到其他并发线程来隐藏内存延迟和其他停顿的能力。

为了缓解内存延迟的影响，CUTLASS 使用软件流水线技术将内存访问与线程内的其他计算重叠。CUTLASS 通过在以下作用域中使用双缓冲来实现这一点：

线程块作用域的共享内存块：在共享内存中分配两个内存块。一个用于加载当前矩阵运算的数据，另一个用于缓冲从全局内存加载的数据，以供下一次主循环迭代使用。

线程束作用域的矩阵片段：在寄存器中分配两个矩阵片段。一个片段在当前矩阵计算期间传递给 CUDA 和 TensorCores，而另一个片段用于接收共享内存的获取结果，以供下一个 warp 级矩阵运算使用。

下图展示了 CUTLASS GEMM 中使用的高效流水线式主循环体。

![](/assets/efficient_gemm/image(1).png)

### Threadblock Rasterization

为了最大限度地重用 L2 缓存中的数据，CUTLASS 定义了多个函数来影响线程块到 GEMM 问题逻辑分区的映射。这些函数将连续启动的线程块映射到分区 GEMM 问题的二维压缩区域，从而提高它们在大致相同的时间访问全局内存相同块的概率。

这些函数定义在 cutlass/gemm/threadblock_swizzle.h 文件中。

### Parallelized Reductions

**Split K - reduction across threadblocks**

矩阵乘法计算揭示了 O(MN) 个独立内积计算之间的并行性。对于足够大的问题规模，CUTLASS 中的 GEMM 内核可以接近理论最大计算吞吐量。然而，对于小问题，线程块数量太少，无法有效地利用整个 GPU。

作为一种解决方案，并行化内积计算期间执行的归约操作，使得更多线程块可以并发执行，同时仍然能够利用大型线程块级 GEMM 块的吞吐量优势。

CUTLASS 通过对 GEMM 的 K 维度进行分区，并为每个分区启动一组额外的线程块，来实现跨线程块的并行归约。因此，我们在 CUTLASS 中将此策略称为“并行归约 splitK”。“并行归约 splitK”策略需要执行两个内核：分区 K GEMM 和批处理归约。

分区 K GEMM 类似于批处理步长 GEMM 的一种变体。 partitionedK GEMM 不要求用户指定每个批次的具体问题规模，而是要求用户提供总体问题规模以及沿 K 维度对操作数 A 和 B 进行分区的数量。例如，参数 m=128、n=128、k=4096 和 partition=16 将生成 16 个分批步长 GEMM，每个批次的参数为 m=128、n=128 和 k=256。PartitionedK 也允许 k 不能被分区数整除的情况。

例如，参数 m=128、n=128、k=4096 和 partition=20 将生成 20 个分批步长 GEMM。前 19 个批次的参数为 m=128、n=128 和 k=4096/20=204，最后一个批次的参数为 m=128、n=128 和 k=220。批量归约内核以分区 K 维 GEMM 的输出(C)作为输入，并沿 K 维执行归约。用户必须管理工作区内存以存储此中间结果。

**Sliced K - reduction across warps**

与 split-k 方案类似，sliced-k 旨在提高 M 和 N 维度较小但 K 维度较大的内核的效率。在线程块级别，参数 CtaTileN 和 CtaTileM 通过将工作划分到不同的线程束 (warp) 来体现并行性。更大的 warpTile 可以带来更好的指令级并行性 (ILP) 和重用性，但也会限制每个线程块中运行的线程束数量，从而降低效率。

为了提高此类方案的效率，除了 ctaTileK 维度之外，对 warpTile 进行划分有助于更有效地利用硬件，允许在 CTA 中并发运行更多线程束。sliced-k 内核不仅在 CtaTileN 和 CtaTileM 维度上，而且在 CtaTileK 维度上也对线程块的计算进行划分。因此，sliced-k 会带来少量开销，即需要在最后对参与的线程束进行归约。这是因为每个 warp 只使用 CtaTileK 的一个“切片”进行计算，因此每个 warp 在归约之前只有一个部分和。

### Hopper Warp Specialization

注意：以下关于 warp-specialization 的部分包含针对 Hopper 内核设计的具体细节。Blackwell SM100 内核的 warp-specialization 结构与 Hopper 内核截然不同，但生产者和消费者代理分离的概念仍然适用。

从 Hopper 内核开始，CUTLASS 3.0 将 warp-specialization 的概念融入到内核设计中。一个线程块被划分为两组线程束：生产者线程束组和消费者线程束组。生产者线程束组使用新的张量内存加速器 (TMA) 将数据从全局内存加载到共享内存缓冲区中。

生产者线程束组 (DMA) 等待消费者线程束组使用新增的**Async Pipeline class**发出共享内存缓冲区已满的信号。一旦数据写入共享内存，TMA 还会更新与该阶段关联的屏障，以通知受影响的线程缓冲区已满。另一方面，消费者线程束组（MMA）等待生产者线程束组发出缓冲区已满的信号，然后启动 Tensor Core MMA 操作。最后，消费者线程束组释放缓冲区，以便进行下一组 TMA 加载。

**Warp-Specialized Persistent Cooperative kernel design**

从 Hopper 内核开始引入的另一种 warp-specialization 内核设计是  *Warp-Specialized Persistent Cooperative* 内核。与  warp-specialization 内核类似，cooperative 设计中也保留了 Warp 组和 Warp 组间屏障同步的概念。Warp-Specialized Persistent Cooperative kernel 的显著特点如下：

1. 持久的线程块被启动来占用 GPU 架构中所有的 SM。这些持久的线程块用于 tile output，因此在其生命周期内可能计算多个 output。其主要优势在于分摊了线程块启动和内核 prologue 的开销，这些开销是所有内核的典型特征。
1. 存在两个消费者 warp groups 协作处理同一块 output tile，方法是将 output tile 沿着 M 维度分割成两半。这允许使用更大的 output tile，因为每个消费者 warp group 的寄存器压力降低了，从而可以提高性能。

由于每个线程块现在都要计算多个输出图块，因此网格启动的形状以及图块到线程块的调度都由新的图块调度器来管理。图块调度器会考虑集群的形状以及可用 SM 的数量，从而计算出输出图块到已启动线程块的有效调度方案。

**Warp-Specialized Persistent Ping-Pong kernel design**

第三种内核设计是 *Warp-Specialized Persistent Ping-Pong* 内核。与 Warp-Specialized Persistent Cooperative 内核类似，Persistent Ping-Pong 内核也保留了 Warp 组的概念、Warp 组之间的屏障同步以及网格启动的形式。Persistent Ping-Pong 的独特之处在于：

1. 两个消费者 Warp 组使用 Tile 调度器分配不同的输出 tile。这使得一个消费者 Warp 组的 epilogue 可以与另一个消费者 Warp 组的数学运算重叠，从而最大限度地利用张量核心。
1. 生产者 Warp 组使用有序序列屏障进行同步，按顺序依次填充两个消费者 Warp 组的缓冲区。
