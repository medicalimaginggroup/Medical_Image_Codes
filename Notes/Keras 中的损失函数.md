# Keras 中的损失函数

[Keras 中的损失函数](https://zhuanlan.zhihu.com/p/34667893)

损失函数是模型优化的目标，所以又叫目标函数、优化评分函数，在keras中，模型编译的参数loss指定了损失函数的类别，有两种指定方法：

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

或者

```python
from keras import losses
model.compile(loss=losses.mean_squared_error, optimizer='sgd')
```

你可以传递一个现有的损失函数名，或者一个TensorFlow/Theano符号函数。 该符号函数为每个数据点返回一个标量，有以下两个参数:

- y_true: 真实标签. TensorFlow/Theano张量
- y_pred: 预测值. TensorFlow/Theano张量，其shape与y_true相同

实际的优化目标是所有数据点的输出数组的平均值。

## **mean_squared_error：均方误差**

```python
mean_squared_error(y_true, y_pred)
```

源码：

```python
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
```

说明：

MSE: ![[公式]](https://www.zhihu.com/equation?tex=L%3D+%5Cfrac+%7B1%7D%7Bn%7D+%5Csum%5En_%7Bi%3D1%7D+%28y_%7Btrue%7D%5E%7B%28i%29%7D+-+y_%7Bpred%7D%5E%7B%28i%29%7D%29%5E2)



## **mean_absolute_error**

```python
mean_absolute_error(y_true, y_pred)
```

源码：

```python
def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)
```

说明：

MAE： ![[公式]](https://www.zhihu.com/equation?tex=L%3D+%5Cfrac+%7B1%7D%7Bn%7D+%5Csum%5En_%7Bi%3D1%7D+%7C%28y_%7Btrue%7D%5E%7B%28i%29%7D+-+y_%7Bpred%7D%5E%7B%28i%29%7D%29%7C)

## **mean_absolute_percentage_error**

```python
mean_absolute_percentage_error(y_true, y_pred)
```

源码：

```python
def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return 100. * K.mean(diff, axis=-1)
```

说明：

MAPE： ![[公式]](https://www.zhihu.com/equation?tex=L%3D+%5Cfrac+%7B1%7D%7Bn%7D+%5Csum%5En_%7Bi%3D1%7D+%7C%5Cfrac+%7By_%7Btrue%7D%5E%7B%28i%29%7D+-+y_%7Bpred%7D%5E%7B%28i%29%7D%7D%7By_%7Btrue%7D%5E%7B%28i%29%7D%7D%7C+%5Ccdot+100)



## **mean_squared_logarithmic_error**

```python
mean_squared_logarithmic_error(y_true, y_pred)
```

源码：

```python
def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)
```

说明：

MAPE： ![[公式]](https://www.zhihu.com/equation?tex=L%3D+%5Cfrac+%7B1%7D%7Bn%7D+%5Csum%5En_%7Bi%3D1%7D+%28log%28y_%7Btrue%7D%5E%7B%28i%29%7D+%2B1%29+-+log%28+y_%7Bpred%7D%5E%7B%28i%29%7D%2B1%29%29%5E2)



## **squared_hinge**

```python
squared_hinge(y_true, y_pred)
```

源码：

```python
def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)
```

说明：

![[公式]](https://www.zhihu.com/equation?tex=L%3D+%5Cfrac+%7B1%7D%7Bn%7D+%5Csum%5En_%7Bi%3D1%7D+%28max%280%2C1-y_%7Bpred%7D%5E%7B%28i%29%7D+%5Ccdot+y_%7Btrue%7D%5E%7B%28i%29%7D%29%29%5E2)

## **hinge**

```python
hinge(y_true, y_pred)
```

源码：

```python
def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)
```

说明：

![[公式]](https://www.zhihu.com/equation?tex=L%3D+%5Cfrac+%7B1%7D%7Bn%7D+%5Csum%5En_%7Bi%3D1%7D+max%280%2C1-y_%7Bpred%7D%5E%7B%28i%29%7D+%5Ccdot+y_%7Btrue%7D%5E%7B%28i%29%7D%29)

## **categorical_hinge**

```python
categorical_hinge(y_true, y_pred)
```

源码：

```python
def categorical_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.maximum(0., neg - pos + 1.)
```



## **logcosh**

```python
logcosh(y_true, y_pred)
```

源码：

```python
def logcosh(y_true, y_pred):
    """Logarithm of the hyperbolic cosine of the prediction error.
    `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
    to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
    like the mean squared error, but will not be so strongly affected by the
    occasional wildly incorrect prediction.
    # Arguments
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.
    # Returns
        Tensor with one scalar loss entry per sample.
    """
    def _logcosh(x):
        return x + K.softplus(-2. * x) - K.log(2.)
    return K.mean(_logcosh(y_pred - y_true), axis=-1)
```



## **categorical_crossentropy**

```python
categorical_crossentropy(y_true, y_pred)
```

源码：

```python
def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)
```

亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列

注意: 当使用”categorical_crossentropy”作为目标函数时,标签应该为多类模式,即one-hot编码的向量,而不是单个数值. (即，如果你有10个类，每个样本的目标值应该是一个10维的向量，这个向量除了表示类别的那个索引为1，其他均为0)。可以使用工具中的to_categorical函数完成该转换.示例如下:

```python
from keras.utils.np_utils import to_categorical
categorical_labels = to_categorical(int_labels, num_classes=None)
```

## **sparse_categorical_crossentropy**

如上，但接受稀疏标签。注意，使用该函数时仍然需要你的标签与输出值的维度相同，你可能需要在标签数据上增加一个维度：np.expand_dims(y,-1)

```python
sparse_categorical_crossentropy(y_true, y_pred)
```

源码：

```python
def sparse_categorical_crossentropy(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred)
def sparse_categorical_crossentropy(target, output, from_logits=False):
    """Categorical crossentropy with integer targets.

    # Arguments
        target: An integer tensor.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.

    # Returns
        Output tensor.
    """
    # Note: tf.nn.sparse_softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output)

    output_shape = output.get_shape()
    targets = cast(flatten(target), 'int64')
    logits = tf.reshape(output, [-1, int(output_shape[-1])])
    res = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets,
        logits=logits)
    if len(output_shape) >= 3:
        # if our output includes timestep dimension
        # or spatial dimensions we need to reshape
        return tf.reshape(res, tf.shape(output)[:-1])
    else:
        return res
```



## **binary_crossentropy**

（亦称作对数损失，logloss）

```python
binary_crossentropy(y_true, y_pred)
```

源码：

```python
def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
def binary_crossentropy(target, output, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.

    # Returns
        A tensor.
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))

    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                   logits=output)
```



## **kullback_leibler_divergence**

从预测值概率分布Q到真值概率分布P的信息增益,用以度量两个分布的差异

```python
kullback_leibler_divergence(y_true, y_pred)
```

源码：

```python
def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)
```



## **poisson**

(predictions - targets * log(predictions))的均值

```python
poisson(y_true, y_pred)
```

源码：

```python
def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)
```

说明：

![[公式]](https://www.zhihu.com/equation?tex=L%3D+%5Cfrac+%7B1%7D%7Bn%7D+%5Csum%5En_%7Bi%3D1%7D+%28y_%7Bpred%7D%5E%7B%28i%29%7D+-+y_%7Btrue%7D%5E%7B%28i%29%7D%5Ccdot+log%28y_%7Bpred%7D%5E%7B%28i%29%7D%29%29)

## **cosine_proximity**

即预测值与真实标签的余弦距离平均值的相反数

```python
cosine_proximity(y_true, y_pred)
```

源码：

```python
def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.sum(y_true * y_pred, axis=-1)
```

说明：

![[公式]](https://www.zhihu.com/equation?tex=L%3D+-+%5Cfrac%7B+%5Csum%5En_%7Bi%3D1%7D+y_%7Btrue%7D%5E%7B%28i%29%7D+%5Ccdot+y_%7Bpred%7D%5E%7B%28i%29%7D%7D+%7B%5Csqrt%7B+%5Csum%5En_%7Bi%3D1%7D+%28y_%7Btrue%7D%5E%7B%28i%29%7D%29%5E2%7D+%5Ccdot+%5Csqrt+%7B%5Csum%5En_%7Bi%3D1%7D+%28y_%7Bpred%7D%5E%7B%28i%29%7D%29%5E2%7D%7D)



简写：

```python
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence
cosine = cosine_proximity
```