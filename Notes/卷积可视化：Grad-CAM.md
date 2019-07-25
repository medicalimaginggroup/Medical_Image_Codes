# 卷积可视化：Grad-CAM

论文[《Grad-CAM:Visual Explanations from Deep Networks via Gradient-based Localization》](http://cn.arxiv.org/pdf/1610.02391v3)，文中介绍了一种`卷积神经网络的解释方法`，通过构建类似热力图 (heatmap) 的形式，直观展示出卷积神经网络学习到的特征，简单说就是：到底我的模型关注点在哪？凭啥认为这张图中有猫？当然，其本质还是从像素角度去解释，并不能像人类那样直观的解释某一个动物为什么是猫。



##### 1. CAM

在很长一段时间内，CNN 虽然效果显著但却饱受争议，根源在于其可解释性较差，同时也因此衍生出一个新的领域：深度学习的可解释性研究。比较经典的研究方法是采用反卷积（Deconvolution）和导向反向传播（Guided-backpropagation），相关的论文也比较多就不在一一列举了，这里给出一个相对比较好的：[《Striving for Simplicity: The All Convolutional Net》](http://cn.arxiv.org/pdf/1412.6806.pdf)。

在介绍 CAM 之前最好对 [GAP](http://spytensor.com/index.php/archives/19/) 有一定的了解，因为 CAM 的作者就是借鉴了这个方法来处理。下图是 CAM 的一些直观描述：

![](http://img.cdn.leonwang.top/20190723144640.png)

特征图经过 GAP 处理后每一个特征图包含了不同类别的信息，其具体效果如上图的 Class Activation Mapping 中的图片所示（只看图片，忽略公式），其中的权重 w 对应分类时的权重。这样做的缺陷是因为要替换全连接层为 GAP 层，因此模型要重新训练，这样的处理方式对于一些复杂的模型是行不通的，Grad-CAM 很好的解决了这个问题，具体继续往下看。

现在的问题是，即使模型训练好了，我们怎么绘制出热点图？这个比较简单，我们只需要提取出所有的权重，往回找到对应的特征图，然后进行加权求和即可。另外如果有兴趣进一步了解 CAM 的话，可以参考一下 Jacob Gildenblat 的复现： [keras-cam](https://github.com/jacobgil/keras-cam), 提醒一下，这个代码本人没有进行测试，所以不能保证顺利运行。

**总结起来，CAM 的意义就是以热力图的形式告诉我们，模型通过哪些像素点得知图片属于某个类别。**



##### 2. Grad-CAM

其实 CAM 得到的效果已经很不错了，但是由于其需要修改网络结构并对模型进行重新训练，这样就导致其应用起来很不方便。Grad-CAM 和 CAM 基本思路一样，区别就在于如何获取每个特征图的权重，采用了梯度的全局平均来计算权重，论文中也给出了证明两种方式得到的权重是否等价的详细过程，如果有需要可以阅读论文进行推导。这里为了与 CAM 的权重进行区分，定义 Grad-CAM 中第 k 个特征图对应类别 c 的权重为 αkc，可以通过下面的公式计算得到：


$$
\alpha_{k}^{c}=\frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^{c}}{\partial A_{i j}^{k}}
$$


参数解析：

- Z: 特征图的像素个数;
- yc: 第 c 类得分的梯度 (the gradient of the score for class c)；
- Aijk: 第 k 个特征图中，(i,j) 位置处的像素值；

然后再求得所有的特征图对应的类别的权重后进行加权求和，这样便可以得到最后的热力图，求和公式如下：


$$
L_{G r a d-C A M}^{c}=\operatorname{Re} L U\left(\sum_{k} \alpha_{k}^{c} A^{k}\right)
$$


下图是论文中给出的 Grad-CAM 整体结构图：

![](http://img.cdn.leonwang.top/20190723144659.png)

**提醒：**
论文中对最终的加权结果进行了一次 ReLU 激活处理，目的是只考虑对类别 c 有正影响的像素点。



##### 3. 效果展示

![](http://img.cdn.leonwang.top/20190723144712.png)



##### 4. keras 实现 Grad-CAM

源码为 [Github:keras-grad-cam](https://github.com/jacobgil/keras-grad-cam)，但是可能因为框架版本的原因，存在较多的 bug 我修改后可以正常运行，贴出我纠正好的代码，另外这里只给出了 VGG16 的实现，其他模型请自行阅读模型复现源码，进行修改即可，比较容易。
`python 3.6` `python-opencv 3.4.2.17` `keras 2.2.0` `tensorflow 1.9.0`

```python
from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Model
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = VGG16(weights='imagenet')
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
def _compute_gradients(tensor, var_list):
  grads = tf.gradients(tensor, var_list)
  return [grad if grad is not None else tf.zeros_like(var)
          for var, grad in zip(var_list, grads)]

def grad_cam(input_model, image, category_index, layer_name):
    nb_classes = 1000
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(input_model.output)
    model = Model(inputs=input_model.input, outputs=x)
    model.summary()
    loss = K.sum(model.output)
    conv_output =  [l for l in model.layers if l.name is layer_name][0].output
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

preprocessed_input = load_image("../../images/dog-cat.jpg")

model = VGG16(weights='imagenet')

predictions = model.predict(preprocessed_input)
top_1 = decode_predictions(predictions)[0][0]
print('Predicted class:')
print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

predicted_class = np.argmax(predictions)
cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "block5_conv3")

# cv2.imwrite("../../images/gradcam.jpg", cam)

register_gradient()
guided_model = modify_backprop(model, 'GuidedBackProp')
saliency_fn = compile_saliency_function(guided_model)
saliency = saliency_fn([preprocessed_input, 0])
gradcam = saliency[0] * heatmap[..., np.newaxis]
# cv2.imwrite("../../images/guided_gradcam.jpg", deprocess_image(gradcam))

origin_img = cv2.imread("../../images/dog-cat.jpg")
origin_img = cv2.resize(origin_img,(414,414))
cam = cv2.resize(cam,(414,414))
guided_gradcam = cv2.resize(deprocess_image(gradcam),(414,414))

plt.subplot(2,2,1), plt.imshow(origin_img), plt.title('origin'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3), plt.imshow(cam), plt.title('gradcam'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4), plt.imshow(guided_gradcam), plt.title('guided_gradcam'), plt.xticks([]), plt.yticks([])
plt.show()   
```



##### 5. pytorch 实现 Grad-CAM展开目录

另外再附上一份 pytorch 版本的实现，原地址为：[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)，由于版本的原因，需要做一些调整，我这边使用的是 `pytorch 0.4.0`，pytorch 不像 keras 那样接口一致，所以不同的网络模型实现方式有所不同，这里只给出了 VGG 的实现方式，若想要进行修改，详细阅读模型复现源码进行修改，或者移步 [pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations) ，这里给出了比较多的可视化方法。

```python
import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output

def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("../../images/cam01.jpg", np.uint8(255 * cam))

class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input) 

    def __call__(self, input, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        #one_hot.backward(retain_variables=True)
        one_hot.backward()
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward()

        output = input.grad.cpu().data.numpy()
        output = output[0,:,:,:]

        return output

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    image_path = "../../images/dog-cat.jpg"

    # Can work with any model, but it assumes that the model has a 
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    grad_cam = GradCam(model = models.vgg19(pretrained=True), \
                    target_layer_names = ["35"], use_cuda=True)

    img = cv2.imread(image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None

    mask = grad_cam(input, target_index)

    show_cam_on_image(img, mask)

    gb_model = GuidedBackpropReLUModel(model = models.vgg19(pretrained=True), use_cuda=True)
    gb = gb_model(input, index=target_index)
    utils.save_image(torch.from_numpy(gb), '../../images/gb.jpg')

    cam_mask = np.zeros(gb.shape)
    for i in range(0, gb.shape[0]):
        cam_mask[i, :, :] = mask

    cam_gb = np.multiply(cam_mask, gb)
    utils.save_image(torch.from_numpy(cam_gb), '../../images/cam_gb.jpg')
```



##### 6. 参考文献

- [《Grad-CAM:Visual Explanations from Deep Networks via Gradient-based Localization》](http://cn.arxiv.org/pdf/1610.02391v3)
- [《Striving for Simplicity: The All Convolutional Net》](http://cn.arxiv.org/pdf/1412.6806.pdf)
- [keras-cam](https://github.com/jacobgil/keras-cam)
- [Github:keras-grad-cam](https://github.com/jacobgil/keras-grad-cam)
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- [pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)