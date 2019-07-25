# Global Average Pooling

GAP 最早是由 Min Lin 等人在论文[《Network In Network》](https://arxiv.org/abs/1312.4400)中提出的，推荐阅读一下这篇文章，里面的创新点值得一看。

在讲解 GAP 之前先做些铺垫，以便更好的理解。我们知道，常见 CNN 的网络结构如下图所示：

![](http://img.cdn.leonwang.top/20190723152433.png)

在全连接层以前的`卷积层`负责对图像进行特征提取，`池化层`负责降采样：保留显著的特征、降低特征维度同时增大 kernel 的感受野 1。实际运用中发现网络层数越深越能获得较丰富的`空间信息`和`语义信息`，而这些也需要较大的感受野才能满足。在获取特征后，传统的做法是接上`全连接层`后再进行激活分类。问题也出在这个全连接层，虽然可以做到对 feature map2 进行降维，但是参数实在太大了，很容易造成过拟合, 既然都是为了降维，那么这个工作是不是`池化层`也可以做呢？

**答案是肯定的。**

论文作者使用 GAP 来替代最后的全连接层, 直接实现了降维，并且也极大地降低了网络参数 (全连接层的参数在整个 CNN 中占有很大的比重)，更重要的一点是保留了由前面各个卷积层和池化层提取到的空间信息，实际应用中效果提升也比较明显。另外 GAP 的另一个重要作用就是去除了对输入大小的限制，这一方面在卷积可视化 Grad-CAM 中比较重要。GAP 的网络结构图如下：

![](http://img.cdn.leonwang.top/20190723152449.png)

GAP 真正的意义在于它实现了在整个网络结构上的正则化以实现防止过拟合的功能，原理在于传统的全连接网络 (如上图的左图) 对 feature map 进行处理时附带了庞大的参数以达到 “暗箱操作” 获取足够多的非线性特征，然后接上分类器, 由于参数众多，难免存在过拟合的现象。GAP 直接从 feature map 的通道信息下手，比如我们现在的分类有 20 种，那么最后一层的卷积输出的 feature map 就只有 20 个通道，然后对这个 feature map 进行全局池化操作，获得长度为 20 的向量，这就相当于直接赋予了每个通道类别的意义。

```
注：
1. 增大感受野就好比你站在地球上看不出它是圆的，站在外太空就可以；
2. feature map 就是全连接层之前的长x高x深的块，通常称一片长x高的特征为一个feature map，深也就是最初的图像通道数；
```

另外在 keras 中已经定义好了，调用方式如下：

```python
from keras.layers import GlobalAveragePooling2D,Dense
from keras.applications import VGG16
from keras.models import Model
def build_model():
    base_model = VGG16(weights="imagenet",include_top=False)
    #在分类器之前使用
    gap = GlobalAveragePooling2D()(base_model)
    predictions = Dense(20,activation="softmax")(gap)
    model = Model(inputs=base_model.input,outputs=predictions)
    return model
```

参考文献：

- [Network In Network](https://arxiv.org/abs/1312.4400)

