# Group-Meeting
Paper shared in the weekly group meeting.
Recording from August, 2023 onwards.

## 2023.8.11

1. [*Separable Self-attention for Mobile Vision Transformers*](https://arxiv.org/pdf/2206.02680.pdf) **[github](https://github.com/apple/ml-cvnets)**
   A new attention in ViT to reduce ViT complexity / latency and improve efficiency. Mainly on mobile devices.
2. [Learning A Sparse Transformer Network for Effective Image Deraining](https://arxiv.org/pdf/2303.11950.pdf) **[github](https://github.com/cschenxiang/DRSformer)**
   A new transformer (top-k) and FNN (multi-level).
3. [WIRE: Wavelet Implicit Neural Representations](https://arxiv.org/pdf/2301.05187.pdf) **[github](https://github.com/vishwa91/wire)**
   A new nonlinear activation function used in INR.


## 2023.8.18

1. [*Vision Transformer with Attention Map Hallucination and FFN Compaction*](https://arxiv.org/pdf/2306.10875.pdf) **[github]()**
   A new transormer (attention map hallucination) and FNN (compact) to improve efficiency and performance.
2. [Fast Vision Transformers with HiLo Attention](https://arxiv.org/pdf/2205.13213.pdf) **[github](https://github.com/ziplab/LITv2)**
   A new transformer (attention in high and low frequency domain) with higher speed, efficiency and lower memory consumption, FLOPs(float-point operations), but maybe lower performance.
3. [EcoFormer: Energy-Saving Attention with Linear Complexity](https://arxiv.org/pdf/2209.09004.pdf) **[github](https://github.com/ziplab/EcoFormer)**
   A new transformer (a new binarization paradigm) which improve the efficiency at the cost of tiny performance degradation.
4. [Shunted Self-Attention via Multi-Scale Token Aggregation](https://arxiv.org/pdf/2111.15193.pdf) **[github](https://github.com/OliverRensu/Shunted-Transformer)**
   A new transformer (multi-scaled spatial K/V) and Detail-specific Feedforward Layers to improve performance.


## 2023.8.25

1. [*Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks*](https://arxiv.org/pdf/2303.03667.pdf) **[github](https://github.com/JierunChen/FasterNet)**
   A new convolution(partial conv, only to a few channel and the rest remain the same) that can reduce FLOPs and improve FLOPS, the performance is also not bad.
2. [*Green Hierarchical Vision Transformer for Masked Image Modeling*](https://arxiv.org/pdf/2205.13515.pdf) **[github](https://github.com/LayneH/GreenMIM)**
   Group window attention & Apply MAE to hierarchical architecture.
   

## 2023.9.1
1. [*MAGVIT: Masked Generative Video Transformer*](https://arxiv.org/pdf/2212.05199.pdf) **[github](https://github.com/google-research/magvit)**
   A tansformer which can be used in any video generative tasks and a new masked idea used in 3D-VQ(vector quatization).
2. [*Detail-Preserving Transformer for Light Field Image Super-Resolution*](https://arxiv.org/pdf/2201.00346.pdf) **[github](https://github.com/BITszwang/DPT)**
   A simple transformer.

## 2023.9.8
1. [*Pin the Memory: Learning to Generalize Semantic Segmentation*](https://arxiv.org/pdf/2204.03609.pdf) **[github](https://github.com/Genie-Kim/PintheMemory)**
   A memory mechanism for semantic Segmentation. 为每一个待区分物体搞一个memory.
2. [*Multi-Scale Memory-Based Video Deblurring*](https://arxiv.org/pdf/2204.02977.pdf) **[github](https://github.com/jibo27/MemDeblur)**
   视频memory机制，和我们的很相似，在于如何去memory中查询有用的信息。
   
## 2023.9.14
1. [*Towards Interpretable Deep Metric Learning with Structural Matching*](https://arxiv.org/pdf/2108.05889.pdf) **[github](https://github.com/wl-zhao/DIML)**
   计算两张特征图的相似度，给出排序。
2. [*Neural Matching Fields: Implicit Representation of Matching Fields for Visual Correspondence*](https://arxiv.org/pdf/2210.02689.pdf) **[github](https://github.com/KU-CVLAB/NeMF/)**
   用IRN来语义匹配，匹配两张图对应的地方

## 2023.9.14
1. [*Recurrent Video Restoration Transformer with Guided Deformable Attention*](https://arxiv.org/pdf/2206.02146.pdf) **[github](https://github.com/JingyunLiang/RVRT)**
   算一个比较好的recurrent的baseline
2. [*GMFlow: Learning Optical Flow via Global Matching*](https://arxiv.org/pdf/2111.13680.pdf) **[github](https://github.com/haofeixu/gmflow)**
   用transformer计算光流

## 2023.9.27
1. [*Memory-aided Contrastive Consensus Learning for Co-salient Object Detection*](https://arxiv.org/pdf/2302.14485.pdf) **[github](https://github.com/ZhengPeng7/MCCL)**
   memory可以用，给memory整了个loss
2. [*Adaptive Human Matting for Dynamic Videos*](https://arxiv.org/pdf/2304.06018.pdf) **[github](https://github.com/microsoft/AdaM )**


## 2023.10.19
1. [*Implicit Identity Representation Conditioned Memory Compensation Network for Talking Head Video Generation*](https://arxiv.org/pdf/2307.09906.pdf) **[github](https://github.com/harlanhong/ICCV2023-MCNET)**
   模仿其中整一个伪的特征图，然后使用memory融合一下
2. [*MeMOTR: Long-Term Memory-Augmented Transformer for Multi-Object Tracking*](https://arxiv.org/pdf/2307.15700.pdf) **[github](https://github.com/MCG-NJU/MeMOTR)**
   too simple to use

## 2023.10.26
1. [*Visual Prompt Tuning*](https://arxiv.org/pdf/2203.12119) **[github](https://github.com/kmnp/vpt)**
   微调预训练时在输入的地方加上一些prompt，仅仅微调prompt和最后的head，但是对自监督的预训练模型作用不大
2. [*Sensitivity-Aware Visual Parameter-Efficient Fine-Tuning*](https://arxiv.org/pdf/2303.08566.pdf) **[github](https://github.com/ziplab/SPT)**
   找一些最敏感的参数进行微调

## 2023.11.02
1. [*MixPro: Data Augmentation with MaskMix and Progressive Attention Labeling for Vision Transformer*](https://arxiv.org/pdf/2304.12043) **[github](https://github.com/fistyee/MixPro)**
   结合transmix和cutmix两者的数据增强的方法

## 2023.11.09
1. [*How to cheat with metrics in single-image HDR reconstruction*](https://arxiv.org/pdf/2108.08713.pdf) **[github]()**
   深入阐述了一下HDR的恢复的影响因素和机制，很多人其实都学习了错误的东西。

## 2023.11.16
1. [*Focal Modulation Networks*](https://arxiv.org/pdf/2203.11926.pdf) **[github](https://github.com/microsoft/FocalNet)**
   能代替transformer的一种结构，通过构建局部和周围的注意力关系。
2. [*Video-FocalNets: Spatio-Temporal Focal Modulation for Video Action Recognition*](https://arxiv.org/pdf/2307.06947.pdf) **[github](https://talalwasim.github.io/Video-FocalNets/)**
   拓展到了视频域，增加了t维度，其他啥也没干。

## 2023.11.23
1. [*Temporal FiLM: Capturing Long-Range Sequence Dependencies with Feature-Wise Modulation*](https://proceedings.neurips.cc/paper/2019/file/2afc4dfb14e55c6face649a1d0c1025b-Paper.pdf) **[github](https://github.com/kuleshov/audio-super-res)**
   用rnn捕获时序信息
2. [*An Implicit Alignment for Video Super-Resolution*](https://arxiv.org/pdf/2305.00163.pdf)**[github](https://github.com/kai422/IART)**
   代替光流的对齐

## 2023.11.30
1. [*SCOTCH and SODA: A Transformer Video Shadow Detection Framework*](https://arxiv.org/pdf/2211.06885.pdf) **[github](https://lihaoliu-cambridge.github.io/scotch_and_soda/)**
   其中的一个物体随时间大小发生变化，然后对这个变化的物体（可以是好几个物体）进行随着时空的注意力追踪还不错
2. [*Neural Compression-Based Feature Learning for Video Restoration*](https://arxiv.org/pdf/2203.09208.pdf) **[github](https://github.com/zhihaohu/pytorchvideocompression)**
   利用压缩（其中使用了adaptive quantification）进行去噪

## 2023.12.07
1. [*Memory-guided Image De-raining Using Time-Lapse Data*](https://arxiv.org/pdf/2201.01883.pdf) **[github]()**
   用了一个memory机制，来记录rain streak的信息，并用一个loss让不变的背景弱化

## 2023.12.14
1. [*Self-conditioned Image Generation via Generating Representations*](https://arxiv.org/pdf/2312.03701.pdf) **[github](https://github.com/LTH14/rcg)**
   kaiming he新作，采用自监督的方式生成有监督图像生成的所需的“监督”，成为自监督的图像生成方式。
2. [*Cross-Modal Learning with 3D Deformable Attention for Action Recognition*](https://arxiv.org/pdf/2212.05638.pdf) **[github]()**
   将姿势和图像数据结合起来作为动作检测，其中token的使用方式（cls）什么的用作两个模态之间的信息交流值得借鉴。

## 2024.1.4
1. [*MED-VT: Multiscale Encoder-Decoder Video Transformer with Application to Object Segmentation*](https://arxiv.org/pdf/2304.05930.pdf) **[github](https://rkyuca.github.io/medvt/)**
   使用多尺度encoder-decoder的transformer结构，多尺度间的注意力、positional embedding可以参考。
2. [*U-Net v2: Rethinking the Skip Connections of U-Net for Medical Image Segmentation*](https://arxiv.org/pdf/2311.17791.pdf) **[github](https://github.com/yaoppeng/U-Net_v2)**
   非常简单，在skip connection上多加了一个模块而已。

## 2024.02.01
1. [*Sparse Sampling Transformer with Uncertainty-Driven Ranking for Unified Removal of Raindrops and Rain Streaks*](https://arxiv.org/pdf/2308.14153.pdf) **[github](https://ephemeral182.github.io/UDR_S2Former_deraining/)**
   利用uncertainty map（把图中一些雨的位置都找出来的感觉），考虑了去雨中不同degradation（rain streak、rain drop）之间的关系。
2. [*Learning Vision from Models Rivals Learning Vision from Data*](https://arxiv.org/pdf/2312.17742.pdf) **[github](https://github.com/google-research/syn-rep-learn)**
   利用生成数据来进行训练

## 2024.03.07
1. [*LAN-HDR: Luminance-based Alignment Network for High Dynamic Range Video Reconstruction*](https://arxiv.org/pdf/2308.11116.pdf) **[github](https://github.com/haesoochung/LAN-HDR)**
   用Y通道，所谓亮度分量，和一个时间一致损失减少视频flickering，网络用了一个非transformer的qkv对齐。

## 2024.03.14
1. [*LAN-HDR: Luminance-based Alignment Network for High Dynamic Range Video Reconstruction*](https://arxiv.org/pdf/2308.07314) **[github](https://github.com/LIAGM/DAEFR)**
   仅使用High Quality自重建得到的codebook在LQ上用会有domain gap，因此在LQ上也做了一下然后融合的方法。
2. [*RAMiT: Reciprocal Attention Mixing Transformer for Lightweight Image Restoration*](https://arxiv.org/pdf/2305.11474) **[github](https://github.com/rami0205/RAMiT/tree/main)**
   在空间和channel纬度同时attention（值得借鉴），接着在unet中将dowansample的特征上采样后合并。

## 2024.03.21
1. [*Vmamba: Visual state space model*](https://arxiv.org/html/2401.10166v1) **[github](https://github.com/MzeroMiko/VMamba)**
   将state space model用于视觉，线性复杂度，长距离建模，更少内存，有可能代替ViT。
2. [*VideoMamba: State Space Model for Efficient Video Understanding*](https://arxiv.org/pdf/2403.06977) **[github](https://github.com/opengvlab/videomamba)**
   将state space model用于视频，并运用一个mask预训练策略，双向的扫描策略。

## 2024.03.28
1. [*Disentangling Spatial and Temporal Learning for Efficient Image-to-Video Transfer Learning*](https://arxiv.org/pdf/2309.07911) **[github](https://github.com/alibaba-mmai-research/DiST?tab=readme-ov-file)**
   视频理解任务。大规模预训练语言-图像模型（如CLIP）在理解空间内容方面表现出色，但将这种模型直接转移到视频识别领域仍然存在着时间建模能力不足的问题。他采用并行的方法，规避了反向传播经过大模型的方法；在时序信息上额外提取信息并和空间特征融合。
2. [*PASTA: Towards Flexible and Efficient HDR Imaging Via Progressively Aggregated Spatio-Temporal Alignment*](https://arxiv.org/pdf/2403.10376) **[github](https://github.com/opengvlab/videomamba)**
   利用小波变换实现下采样。

## 2024.04.04
1. [*EfficientVMamba: Atrous Selective Scan for Light Weight Visual Mamba*](https://arxiv.org/pdf/2403.09977v1) **[github](https://github.com/TerryPei/EfficientVMamba)**
   轻量级的视觉mamba，一路用cnn，一路用mamba，两路合并采用senet，mamba中扫描采用间隔扫描。
## 2024.04.11
1. [*GTA-HDR: A Large-Scale Synthetic Dataset for HDR Image Reconstruction*](https://arxiv.org/pdf/2403.17837) **[github](https://github.com/HrishavBakulBarua/GTA-HDR)**
   从游戏中获得的数据集。

## 2024.04.18
1. [*FreMIM: Fourier Transform Meets Masked Image Modeling for Medical Image Segmentation*](https://openaccess.thecvf.com/content/WACV2024/papers/Wang_FreMIM_Fourier_Transform_Meets_Masked_Image_Modeling_for_Medical_Image_WACV_2024_paper.pdf) **[github](https://github.com/Rubics-Xuan/FreMIM)**
   同时预测低频和高频的预训练mask。  

## 2024.04.25
1. [*Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction*](https://arxiv.org/pdf/2404.02905) **[github](https://github.com/FoundationVision/VAR)**
   将next token prediction变为next scale prediction进行图像生成，有类似于gpt一样的scaling law，超越diffusion。
2. [*Video Adverse-Weather-Component Suppression Network via Weather Messenger and Adversarial Backpropagation*](https://arxiv.org/pdf/2309.13700) **[github](https://github.com/scott-yjyang/ViWS-Net)**
   用一个可学习的token（weather messenger）在帧之间提取时序信息，用一个对抗网络使网络分辨出天气类型。

## 2024.05.02
1. [*Rewrite the Stars*](https://arxiv.org/pdf/2403.19967) **[github](https://github.com/ma-xu/Rewrite-the-Stars)**
   逐点相乘可以用于在不增加计算量的同时拓展维度（升维）。
2. [*Towards Layer-wise Image Vectorization*](https://arxiv.org/abs/2206.04655) **[github](https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization)**
   将像素图片矢量化，其中udf损失只关注有效像素并忽略不相关的像素。

## 2024.05.09
1. [*White-Box Transformers via Sparse Rate Reduction*](https://proceedings.neurips.cc/paper_files/paper/2023/file/1e118ba9ee76c20df728b42a35fb4704-Paper-Conference.pdf) **[github](https://github.com/Ma-Lab-Berkeley/CRATE)**
   很难懂的一篇，最数学的一集。
2. [*Image as Set of Points*](https://arxiv.org/pdf/2303.01494) **[github](https://ma-xu.github.io/Context-Cluster/)**
   提出用聚类来代替卷积、ViT，效果一般，但很创新。

## 2024.05.16
1. [*Visual Anagrams: Generating Multi-View Optical Illusions with Diffusion Models*](https://arxiv.org/pdf/2311.17919) **[github](https://dangeng.github.io/visual_anagrams)**
   利用扩散模型不同去噪方式加在一起生成有意思的视错觉等，原理非常intuitive。

## 2024.05.23
1. [*Rethinking RGB Color Representation for Image Restoration Models*](https://arxiv.org/pdf/2402.03399) 

   

