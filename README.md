# Group-Meeting
Paper shared in the weekly group meeting.
Recording from August, 2023 onwards.

## 8.11

1. [*Separable Self-attention for Mobile Vision Transformers*](https://arxiv.org/pdf/2206.02680.pdf) **[github](https://github.com/apple/ml-cvnets)**
   A new attention in ViT to reduce ViT complexity / latency and improve efficiency. Mainly on mobile devices.
2. [Learning A Sparse Transformer Network for Effective Image Deraining](https://arxiv.org/pdf/2303.11950.pdf) **[github](https://github.com/cschenxiang/DRSformer)**
   A new transformer (top-k) and FNN (multi-level).
3. [WIRE: Wavelet Implicit Neural Representations](https://arxiv.org/pdf/2301.05187.pdf) **[github](https://github.com/vishwa91/wire)**
   A new nonlinear activation function used in INR.


## 8.18

1. [*Vision Transformer with Attention Map Hallucination and FFN Compaction*](https://arxiv.org/pdf/2306.10875.pdf) **[github]()**
   A new transormer (attention map hallucination) and FNN (compact) to improve efficiency and performance.
2. [Fast Vision Transformers with HiLo Attention](https://arxiv.org/pdf/2205.13213.pdf) **[github](https://github.com/ziplab/LITv2)**
   A new transformer (attention in high and low frequency domain) with higher speed, efficiency and lower memory consumption, FLOPs(float-point operations), but maybe lower performance.
3. [EcoFormer: Energy-Saving Attention with Linear Complexity](https://arxiv.org/pdf/2209.09004.pdf) **[github](https://github.com/ziplab/EcoFormer)**
   A new transformer (a new binarization paradigm) which improve the efficiency at the cost of tiny performance degradation.
4. [Shunted Self-Attention via Multi-Scale Token Aggregation](https://arxiv.org/pdf/2111.15193.pdf) **[github](https://github.com/OliverRensu/Shunted-Transformer)**
   A new transformer (multi-scaled spatial K/V) and Detail-specific Feedforward Layers to improve performance.


## 8.25

1. [*Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks*](https://arxiv.org/pdf/2303.03667.pdf) **[github](https://github.com/JierunChen/FasterNet)**
   A new convolution(partial conv, only to a few channel and the rest remain the same) that can reduce FLOPs and improve FLOPS, the performance is also not bad.
2. [*Green Hierarchical Vision Transformer for Masked Image Modeling*](https://arxiv.org/pdf/2205.13515.pdf) **[github](https://github.com/LayneH/GreenMIM)**
   Group window attention & Apply MAE to hierarchical architecture.
   

## 9.1
1. [*MAGVIT: Masked Generative Video Transformer*](https://arxiv.org/pdf/2212.05199.pdf) **[github](https://github.com/google-research/magvit)**
   A tansformer which can be used in any video generative tasks and a new masked idea used in 3D-VQ(vector quatization).
2. [*Detail-Preserving Transformer for Light Field Image Super-Resolution*](https://arxiv.org/pdf/2201.00346.pdf) **[github](https://github.com/BITszwang/DPT)**
   A simple transformer.

## 9.8
1. [*Pin the Memory: Learning to Generalize Semantic Segmentation*](https://arxiv.org/pdf/2204.03609.pdf) **[github](https://github.com/Genie-Kim/PintheMemory)**
   A memory mechanism for semantic Segmentation. 为每一个待区分物体搞一个memory.
2. [*Multi-Scale Memory-Based Video Deblurring*](https://arxiv.org/pdf/2204.02977.pdf) **[github](https://github.com/jibo27/MemDeblur)**
   视频memory机制，和我们的很相似，在于如何去memory中查询有用的信息。
   
## 9.14
1. [*Towards Interpretable Deep Metric Learning with Structural Matching*](https://arxiv.org/pdf/2108.05889.pdf) **[github](https://github.com/wl-zhao/DIML)**
   计算两张特征图的相似度，给出排序。
2. [*Neural Matching Fields: Implicit Representation of Matching Fields for Visual Correspondence*](https://arxiv.org/pdf/2210.02689.pdf) **[github](https://github.com/KU-CVLAB/NeMF/)**
   用IRN来语义匹配，匹配两张图对应的地方

## 9.14
1. [*Recurrent Video Restoration Transformer with Guided Deformable Attention*](https://arxiv.org/pdf/2206.02146.pdf) **[github](https://github.com/JingyunLiang/RVRT)**
   算一个比较好的recurrent的baseline
2. [*GMFlow: Learning Optical Flow via Global Matching*](https://arxiv.org/pdf/2111.13680.pdf) **[github](https://github.com/haofeixu/gmflow)**
   用transformer计算光流

## 9.27
1. [*Memory-aided Contrastive Consensus Learning for Co-salient Object Detection*](https://arxiv.org/pdf/2302.14485.pdf) **[github](https://github.com/ZhengPeng7/MCCL)**
   memory可以用，给memory整了个loss
2. [*Adaptive Human Matting for Dynamic Videos*](https://arxiv.org/pdf/2304.06018.pdf) **[github](https://github.com/microsoft/AdaM )**


## 10.19
1. [*Implicit Identity Representation Conditioned Memory Compensation Network for Talking Head Video Generation*](https://arxiv.org/pdf/2307.09906.pdf) **[github](https://github.com/harlanhong/ICCV2023-MCNET)**
   模仿其中整一个伪的特征图，然后使用memory融合一下
2. [*MeMOTR: Long-Term Memory-Augmented Transformer for Multi-Object Tracking*](https://arxiv.org/pdf/2307.15700.pdf) **[github](https://github.com/MCG-NJU/MeMOTR)**
   too simple to use

## 10.26
1. [*Visual Prompt Tuning*](https://arxiv.org/pdf/2203.12119) **[github](https://github.com/kmnp/vpt)**
   微调预训练时在输入的地方加上一些prompt，仅仅微调prompt和最后的head，但是对自监督的预训练模型作用不大
2. [*Sensitivity-Aware Visual Parameter-Efficient Fine-Tuning*](https://arxiv.org/pdf/2303.08566.pdf) **[github](https://github.com/ziplab/SPT)**
   找一些最敏感的参数进行微调

## 11.02
1. [*MixPro: Data Augmentation with MaskMix and Progressive Attention Labeling for Vision Transformer*](https://arxiv.org/pdf/2304.12043) **[github](https://github.com/fistyee/MixPro)**
   结合transmix和cutmix两者的数据增强的方法

## 11.09
1. [*How to cheat with metrics in single-image HDR reconstruction*](https://arxiv.org/pdf/2108.08713.pdf) **[github]()**
   深入阐述了一下HDR的恢复的影响因素和机制，很多人其实都学习了错误的东西。

## 11.16
1. [*Focal Modulation Networks*](https://arxiv.org/pdf/2203.11926.pdf) **[github](https://github.com/microsoft/FocalNet)**
   能代替transformer的一种结构，通过构建局部和周围的注意力关系。
2. [*Video-FocalNets: Spatio-Temporal Focal Modulation for Video Action Recognition*](https://arxiv.org/pdf/2307.06947.pdf) **[github](https://talalwasim.github.io/Video-FocalNets/)**
   拓展到了视频域，增加了t维度，其他啥也没干。

## 11.23
1. [*Temporal FiLM: Capturing Long-Range Sequence Dependencies with Feature-Wise Modulation*](https://proceedings.neurips.cc/paper/2019/file/2afc4dfb14e55c6face649a1d0c1025b-Paper.pdf) **[github](https://github.com/kuleshov/audio-super-res)**
   用rnn捕获时序信息
2. [*An Implicit Alignment for Video Super-Resolution*](https://arxiv.org/pdf/2305.00163.pdf)**[github](https://github.com/kai422/IART)**
   代替光流的对齐

## 11.30
1. [*SCOTCH and SODA: A Transformer Video Shadow Detection Framework*](https://arxiv.org/pdf/2211.06885.pdf) **[github](https://lihaoliu-cambridge.github.io/scotch_and_soda/)**

   
