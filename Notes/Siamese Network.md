# Siamese Network

[TOC]

## Siamese Network (2-branches network)

Siamese Network，孪生神经网络， 于2005年由Yann Lecun提出。

Siamese Network 就是 “连体的神经网络”，神经网络的 “连体” 是通过共享权值来实现的。孪生网络的目标是寻找两个可比较对象的相似程度（例如，签名验证、人脸识别等）。这个网络有两个相同的子网络，两个子网络有着相同的参数和权重。

它的特点是它接收两张图片作为输入，而不是一张图片作为输入。

![](http://img.cdn.leonwang.top/sianet.png)

---

![](http://img.cdn.leonwang.top/decisionnet.png)

目的：比较两幅图片是否相似（相似度）

输入：两幅图片

输出：一个相似度数值

**Siamese & Pseudo-Siamese**

Siamese Network: 两个神经网络共享权值，在代码实现的时候，甚至可以是同一个网络，不用实现另外一个，因为权值都一样。

![](http://img.cdn.leonwang.top/Xnip2019-07-21_10-39-10.jpg)



Pseudo-Siamese Network（伪孪生神经网络）：两边可以是不同的神经网络（如一个是lstm，一个是cnn），也可以是相同类型的神经网络。

![](http://img.cdn.leonwang.top/Xnip2019-07-21_10-42-34.jpg)

**孪生神经网络和伪孪生神经网络分别适用的场景**

孪生神经网络用于处理两个输入**"比较类似"**的情况。伪孪生神经网络适用于处理两个输入**"有一定差别"**的情况。比如，我们要计算两个句子或者词汇的语义相似度，使用siamese network比较适合；如果验证标题与正文的描述是否一致（标题和正文长度差别很大），或者文字是否描述了一幅图片（一个是图片，一个是文字），就应该使用pseudo-siamese network。也就是说，要根据具体的应用，判断应该使用哪一种结构，哪一种Loss。

**Siamese network的用途**

应用领域很多，nlp&cv领域都有很多应用。

- 词汇的语义相似度分析，QA中question和answer的匹配，签名/人脸验证。
- 手写体识别（已有github代码）。
- kaggle上Quora的question pair的比赛，即判断两个提问是不是同一问题，冠军队伍用的就是n多特征+Siamese network。
- 在图像上，基于Siamese网络的视觉跟踪算法也已经成为热点《[Fully-convolutional siamese networks for object tracking](https://link.springer.com/chapter/10.1007/978-3-319-48881-3_56)》。



## 改进的 Siamese Network (2-channel network)

2015年CVPR的一篇关于图像相似度计算的文章：《Learning to Compare Image Patches via Convolutional Neural Networks》，本篇文章对经典的算法Siamese Networks 做了改进。



Siamese 网络(2-branches networks)的大体思路：

1. 让patch1、patch2分别经过网络，进行提取特征向量(Siamese 对于两张图片patch1、patch2的特征提取过程是相互独立的)

2. 然后在最后一层对两个两个特征向量做一个相似度损失函数，进行网络训练。



paper所提出的算法(2-channel networks) 的大体思路：

1. 把patch1、patch2合在一起，把这两张图片，看成是一张双通道的图像。也就是把两个(1，64，64)单通道的数据，放在一起，成为了(2，64，64)的双通道矩阵，

2. 然后把这个矩阵数据作为网络的输入，这就是所谓的：2-channel。



这样，跳过了分支的显式的特征提取过程，而是直接学习相似度评价函数。最后一层直接是全连接层，输出神经元个数直接为1，直接表示两张图片的相似度。当然CNN，如果输入的是双通道图片，也就是相当于网络的输入的是2个feature map，经过第一层的卷积后网，两张图片的像素就进行了相关的加权组合并映射，这也就是说，用2-channel的方法，经过了第一次的卷积后，两张输入图片就不分你我了。而Siamese网络是到了最后全连接的时候，两张图片的相关神经元才联系在一起。



## 损失函数：Contrastive Loss（对比损失）

传统的siamese network使用Contrastive Loss。损失函数还有更多的选择，siamese network的初衷是计算两个输入的相似度,。左右两个神经网络分别将输入转换成一个"向量"，在新的空间中，通过判断cosine距离就能得到相似度了。Cosine是一个选择，exp function也是一种选择，欧式距离什么的都可以，训练的目标是让两个相似的输入距离尽可能的小，两个不同类别的输入距离尽可能的大。

在Keras的孪生神经网络（siamese network）中，其采用的损失函数是contrastive loss，这种损失函数可以有效的处理孪生神经网络中的paired data的关系。contrastive loss的表达式如下： 
$$
L=\frac{1}{2 N} \sum_{n=1}^{N} y d^{2}+(1-y) m a x(\text {margin}-d, 0)^{2}
$$
其中$d=\left\|a_{n}-b_{n}\right\|_{2}$，代表两个样本特征的欧氏距离，y为两个样本是否匹配的标签，y=1代表两个样本相似或者匹配，y=0则代表不匹配，margin为设定的阈值。

这种损失函数最初来源于Yann LeCun的Dimensionality Reduction by Learning an Invariant Mapping，主要是用在降维中，即本来相似的样本，在经过降维（特征提取）后，在特征空间中，两个样本仍旧相似；而原本不相似的样本，在经过降维后，在特征空间中，两个样本仍旧不相似。

观察上述的contrastive loss的表达式可以发现，这种损失函数可以很好的表达成对样本的匹配程度，也能够很好用于训练提取特征的模型。当y=1（即样本相似）时，损失函数只剩下$\sum y d^{2}$，即原本相似的样本，如果在特征空间的欧式距离较大，则说明当前的模型不好，因此加大损失。而当y=0时（即样本不相似）时，损失函数为$\sum(1-y) \max (\operatorname{margin}-d, 0)^{2}$，即当样本不相似时，其特征空间的欧式距离反而小的话，损失值会变大，这也正好符合我们的要求。



---------------------
## *References*

[1] S. Chopra, R. Hadsell, and Y. LeCun. Learning a similarity metric discriminatively, with application to face verification. In Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on, volume 1, pages 539–546. IEEE, 2005. 

[2]Hadsell, R., Chopra, S., LeCun, Y.. Dimensionality Reduction by Learning an Invariant Mapping[P]. Computer Vision and Pattern Recognition, 2006 IEEE Computer Society Conference on,2006.

[3] [siamese(孪生) 网络 ——CSDN](https://blog.csdn.net/qq_15192373/article/details/78404761 )

\[4][Contrastive Loss（损失函数）——CSDN](https://blog.csdn.net/autocyz/article/details/53149760)