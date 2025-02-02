# 基于形变恢复扩散模型（DRDM）的实例形变图像合成与修改 #

在医学成像领域，图像合成与操作是现代人工智能的重要应用，尤其在数据增强、少样本学习和图像注册等任务中具有重要意义。然而，现有扩散模型在医学图像中的应用常面临几个核心问题：
1. **生成图像与真实图像缺乏可解释与可靠的关联性**。
2. **因此无法保留像素级别的语义信息，进而无法用于针对诸如图像分割等的像素级别任务做数据增强与合成**。
2. **此外，现有方法容易生成不真实的伪影与粗糙的细节，进而影响下游任务，尤其是像素级别任务的效果**。

本文提出了一种全新的扩散生成模型，**形变恢复扩散模型（Deformation-Recovery Diffusion Model, DRDM）**，旨在通过生成形变场而非直接生成图像，解决上述问题。

![image](https://github.com/jianqingzheng/def_diff_rec/blob/main/docs/static/images/graphic_abstract.png)

### 核心创新点 ###

1. **实例特异性形变生成**：
   **DRDM**首次提出在无需参考图像的情况下，为单张图像生成多样化且合理的形变。

2. **形变扩散模型**：
   基于形变的扩散与恢复过程，取代传统像素值或隐空间特征扩散模型。

3. **多尺度随机形变速度场采样与整合**：
   通过多尺度形变速度流场采样与路径积分，生成物理合理的形变场以训练**DRDM**。

4. **无标注训练**：
   **DRDM**无需人工标注或外部模型支撑，能够从零开始完成训练。

![image](https://github.com/jianqingzheng/def_diff_rec/blob/main/docs/static/images/demo_3d_3x3.gif)

> - **论文链接**：[https://arxiv.org/abs/2407.07295](https://arxiv.org/abs/2407.07295)
> - **项目地址**：[https://jianqingzheng.github.io/def_diff_rec/](https://jianqingzheng.github.io/def_diff_rec/)

## 研究背景 ##

扩散模型因其高质量数据生成能力和可扩展性，已在医学图像合成中得到广泛应用。然而，传统扩散模型以像素值或特征为基础，缺乏对形态学变化的建模能力。在医学图像分析中，特别是解剖结构的图像合成与操作，准确捕捉形变特征至关重要。

**DRDM**通过构建形变场生成与恢复机制，利用拓扑保持的形变场生成方法，随机采样多尺度形变速度流场（Deformation Velocity Fields, DVFs）并将其路径积分，使模型能够从随机形变恢复到真实分布，从而生成多样且解剖学合理的形变。

![image](https://github.com/jianqingzheng/def_diff_rec/blob/main/docs/static/images/different_diff.jpg)


## 方法设计 ##

### 框架设计 ###

**DRDM**框架由 形变扩散（Deformation Diffusion）与 形变恢复（Deformation Recovery）两部分组成：
- **形变扩散**：通过固定的马尔科夫过程，随机生成形变场。
- **形变恢复**：递归估计并恢复形变场，使得随机形变图像回归真实形变分布。

![image](https://github.com/jianqingzheng/def_diff_rec/blob/main/docs/static/images/drdm_framework.jpg)

### 形变扩散 ###
其中形变扩散基于三点假设：
1. **随机性**：要求每个位置的形变向量应符合正态分布；
2. **局部性**：要求连续介质的形变场应保持连续性，定义为在局部区域内的任意两点向量差值满足受限于与该两点距离相关的正态分布；
3. **可逆性**：要求生成的形变场应物理可逆，通过雅可比矩阵的行列式的负值比例来表示。

因此在该文章中设计了一种多尺度的随机速度流场采样，并通过积分得到最终的随机形变流场：
![image](https://github.com/jianqingzheng/def_diff_rec/blob/main/docs/static/images/def_diff_proc.jpg)


### 形变扩散 ###
基于生成的形变流场，**DRDM**在训练过程中学习形变扩散中的向量场分布，并生成对应的逆向量场，从而恢复模型认为不合理的形变分量：
![image](https://github.com/jianqingzheng/def_diff_rec/blob/main/docs/static/images/drdm_framework.jpg)

从而生成针对该图像合理且具有多样性的形变场：
![image](https://github.com/jianqingzheng/def_diff_rec/blob/main/docs/static/images/data_diff_lvl.jpg)

## 实验与结果 ##

### 数据集 ###
本文在两个医学图像模态中验证了**DRDM**的性能：
1. **心脏MRI**：使用公开的心脏数据集（包括Sunnybrook、M&Ms等）。
2. **肺部CT**：使用来自Learn2Reg挑战赛的肺部CT数据。

### 图像合成与形变场评估 ###
实验显示，**DRDM**生成的形变场具有以下特点：
- **高多样性**：支持大规模形变（图像尺寸10%以上）。
- **高质量**：Jacobian矩阵负值比例低于1%，保证形变的物理合理性。

![image](https://github.com/jianqingzheng/def_diff_rec/blob/main/docs/static/images/examples_2d.jpg)
![image](https://github.com/jianqingzheng/def_diff_rec/blob/main/docs/static/images/examples_3d.jpg)


### 下游任务改进 ###
1. **医学图像分割**：
   - **DRDM**生成的合成数据用于少样本学习，显著提升分割准确性。
   - 相比传统数据增强方法（如BigAug），**DRDM**在Dice系数等指标上表现更优。

2. **图像配准**：
   - **DRDM**生成的形变场作为配准模型的合成训练数据，提升了肺部CT的配准性能。
   - 实验结果表明，**DRDM**生成的数据接近真实数据在配准任务中的效果。

## 结论 ##

- 提出形变恢复扩散模型**DRDM**，解决医学图像生成中形态学建模的核心难题。
- 提供无标注形变生成的新方法，简化了医学图像数据增强与合成的流程。
- 在医学图像分割与配准任务中，证明了**DRDM**的高效性与实用性。

## 展望 ##

**DRDM**不仅可以用于数据增强与合成，还可以扩展至更多场景：
- **条件形变生成**：结合条件输入，生成特定类型的形变场。
- **多模态图像合成**：结合模态转换模块，实现跨模态图像形变生成。
- **动态图像生成**：与传统扩散模型结合，解决动态影像生成中的纹理不一致问题。


> 扫描下方二维码查看DRDM项目主页：
![image](https://github.com/jianqingzheng/def_diff_rec/blob/main/docs/drdm_project_page.png)
