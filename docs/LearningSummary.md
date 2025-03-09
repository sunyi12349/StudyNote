# 第一周学习汇报（2025/3/2-2025/3/8）

## 1 分布式并行框架

### 1.1 大模型业务全流程

分为八个部分：总体、数据处理、模型算法、模型训练（分布式训练）、模型微调、模型验证、推理与智能体

### 1.2 分布式训练的简要概括

1. 分布式加速库：如DeepSpeed,Megatron-LM,Colossal-AI,BMTrain
2. 分布式并行的意义：提高训练速率（训练速率 = 单设备计算速率 * 设备数 * 并行效率）

### 1.3 大模型遇到AI系统

1. AI系统全栈架构
2. AI系统+LLM全栈架构（应用层多了一些内容，多了一个使能层即各种分布式训练加速库，底层多了一个由网络加速器和AI芯片构成的超级计算节点）

## 2 DeepSpeed基础学习

### 2.1 创新特性

1. Training(最重要)：ZeRO-DP系列，3D-Parallelism，DeepSpeed-MoE，ZeRO-Infinity等
2. Inference(推理)
3. Compression(压缩)
4. Science

### 2.2 软件架构

1. APIs：调用接口
2. RunTime：核心，运行时组件
3. OPs：底层内核组件

### 2.3 基础知识点：

####  2.3.1 混合精度训练

##### 第一种分类：

1. 浮点数精度： 双精度（FP64）、单精度（FP32，TF32）、半精度（FP16，BF16）、8位精度（FP8）、4位精度（FP4，NF4）
2. 量化精度：INT8，INT4
   

##### 第二种分类：


1. FP精度：FP64，FP32，FP16，FP8两种，FP4
2. 特殊精度：TF32（NVIDIA，1+8+10bits），BF16（Google Brain，1+8+7bits），NF4（一种用于量化的特殊格式）
3. 量化精度：INT8，INT4（可以通过多种量化算法将FP转化为INT）

##### 常见形式

FP16+FP32 或 BF16+FP32两种形式，原始主权重为FP32，在更新主权重时要将梯度转为FP32以及BWD中计算LOSS时也要用FP32，其他时候（FWD，BWD）都为FP16

####  2.3.2 显存占用分析

1. Model States(important): Parameters(W,B), Gradients, Optimizer States(Master Weight, Adam momentum, Adam Variance)
2. Residual States: Activation, Temporary Buffers(临时存储), Unable Fragmented Memory(内存碎片化)


### 2.4 ZeRO并行技术的分类

1. ZeRO-DP : ZeRO 1/2/3
2. ZeRO-R(ruduce)
3. ZeRO-Offload
4. ZeRO-Infinity

### 2.5 ZeRO-DP

1. ZeRO stage 1：对Optimizer States切分
2. ZeRO stage 2：对Optimizer States，Gradients切分
3. ZeRO stage 3：对Optimizer States，Gradients，Parameters切分

## 3 DeepSpeed的ZeRO 1/2/3 原理

### 3.1 ZeRO Stage 1

1. 将每个batch数据样本分为N份，每份为一个mini-batch，设⨚为模型参数量
2. FWD，BWD，每一块GPU个得到一份梯度Gi
3. 对Gi执行All-Reduce，每一块GPU得到完整的梯度G（单GPU通讯量2⨚），实际进行了改进，执行Scatter-Reduce（先切分再归纳，每个GPU只得到一部分完整梯度），可以减少显存占用，降低通信开销（单GPU通讯量⨚）
4. 在每一块GPU上更新Wi（由Oi和G决定）
5. 对Wi执行All-Gather，每个GPU得到完整更新后的权重W（单GPU通讯量⨚）
6. 总结：显存从(4+K)⨚降低到(4+K/N) * ⨚;整体通讯量2⨚

### 3.2 ZeRO Stage 2

1. 同 stage 1
2. 同 stage 1
3. 对Gi执行Reduce-Scatter(先归纳后切分)，这一步就是切分梯度（单GPU通讯量⨚）<font color= #871F78>（这里不太明白）</font>
4. 同 stage 1
5. 同 stage 1（单GPU通讯量⨚）
6. 总结：显存从(4+K)⨚降低到(2+(K+2)/N) * ⨚;整体通讯量2⨚

### 3.3 ZeRO Stage 3

1. 同 stage 1，且切分了W为N份
2. FWD，对Wi执行All-Gather（单GPU通讯量⨚）
3. BWD，对Wi执行All-Gather（单GPU通讯量⨚）
4. 对Gi执行Reduce-Scatter(先归纳后切分)（单GPU通讯量⨚）
5. 每个GPU只保存自己的Wi，无需执行All-Gather
6. 总结：显存从(4+K)⨚降低到(K+4)/N * ⨚;整体通讯量3⨚

### 3.4 ZeRO-Offload原理

1. 把占用显存多且计算量较低（大多为标量或向量运算）的部分卸载到CPU上运算，例：Optimizer States 以及更新权重
2. 把计算量较大（大多为矩阵运算）的部分留在GPU上运算，例：FWD，BWD大部分都在GPU上算

### 3.5 调用DeepSpeed

1. DeepSpeed提供了命令行工具
2. HuggingFace中的Transformers集成了DeepSpeed（from Transformers import Trainer）





