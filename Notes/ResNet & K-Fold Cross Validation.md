[TOC]

# 概念

## ResNet

目的：解决深度网络的退化问题

随着网络的加深，出现了**训练集**准确率下降的现象，我们可以确定**这不是由于Overfit过拟合造成的**(过拟合的情况训练集应该准确率很高)；所以作者针对这个问题提出了一种全新的网络，叫深度残差网络，它允许网络尽可能的加深，其中引入了全新的结构如图（残差学习单元）； 

![](http://img.cdn.leonwang.top/resnet.png)

**残差**

残差指的是预测值和观测值之间的差异

> 不要和误差混淆
>
> 误差：观测值和真实值之间的差异

ResNet提出了两种mapping：一种是identity mapping，指的就是图1中”弯弯的曲线”，另一种residual mapping，指的就是除了”弯弯的曲线“那部分，所以最后的输出是 y=F(x)+x

identity mapping顾名思义，就是指本身，也就是公式中的x，而residual mapping指的是“差”，也就是y−x，所以残差指的就是F(x)部分。 



理论上，对于“随着网络加深，准确率下降”的问题，Resnet提供了两种选择方式，也就是identity mapping和residual mapping，如果网络已经到达最优，继续加深网络，residual mapping将被push为0，只剩下identity mapping，这样理论上网络一直处于最优状态了，网络的性能也就不会随着深度增加而降低了。



## K-Fold Cross Validation

>  K次交叉验证，将训练集分割成K个子样本，一个单独的子样本被保留作为验证模型的数据，其他K-1个样本用来训练。交叉验证重复K次，每个子样本验证一次，平均K次的结果或者使用其它结合方式，最终得到一个单一估测。这个方法的优势在于，同时重复运用随机产生的子样本进行训练和验证，每次的结果验证一次，10次交叉验证是最常用的。

![](http://img.cdn.leonwang.top/kfold.png)



# scikit-Learn和Keras实践

Keras在深度学习很受欢迎，但是只能做深度学习：Keras是最小化的深度学习库，目标在于快速搭建深度学习模型。

基于SciPy的scikit-learn，数值运算效率很高，适用于普遍的机器学习任务，提供很多机器学习工具，包括但不限于：

- 使用K折验证模型
- 快速搜索并测试超参

Keras为scikit-learn封装了`KerasClassifier`和`KerasRegressor`。

[Keras 中的 Scikit-Learn API 的封装器文档](https://keras.io/zh/scikit-learn-api/)

[Scikit-Learn 交叉验证文档](https://scikit-learn.org/stable/modules/cross_validation.html#k-fold)

## Scikit-Learn 中的交叉验证

基础验证法：

```python
from sklearn.datasets import load_iris # iris数据集
from sklearn.model_selection import train_test_split # 分割数据模块
from sklearn.neighbors import KNeighborsClassifier # K最近邻(kNN，k-NearestNeighbor)分类算法

#加载iris数据集
iris = load_iris()
X = iris.data
y = iris.target

#分割数据并
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

#建立模型
knn = KNeighborsClassifier()

#训练模型
knn.fit(X_train, y_train)

#将准确率打印出
print(knn.score(X_test, y_test))
# 0.973684210526
```

交叉验证法：

```python
from sklearn.cross_validation import cross_val_score # K折交叉验证模块

#使用K折交叉验证模块
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')

#将5次的预测准确率打印出
print(scores)
# [ 0.96666667  1.          0.93333333  0.96666667  1.        ]

#将5次的预测准确平均率打印出
print(scores.mean())
# 0.973333333333
```



## Scikit-Learn 对 Keras 模型的交叉验证

### Tutorial 1：使用Scikit-learn调用Keras模型

[使用Scikit-Learn调用Keras的模型](https://cnbeining.github.io/deep-learning-with-python-cn/3-multi-layer-perceptrons/ch9-use-keras-models-with-scikit-learn-for-general-machine-learning.html)

- 使用scikit-learn封装Keras的模型
- 使用scikit-learn对Keras的模型进行交叉验证
- 使用scikit-learn，利用网格搜索调整Keras模型的超参

(关于交叉验证的代码有点过时，cross_validation已经被model_selection取代)

使用交叉验证检验深度学习模型

```python
# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
import numpy
import pandas
# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu')) model.add(Dense(8, init='uniform', activation='relu')) model.add(Dense(1, init='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, nb_epoch=150, batch_size=10)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(y=Y, n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

### Tutorial 2：基于 skl3earn 和 keras 的数据切分与交叉验证

[[基于sklearn和keras的数据切分与交叉验证](https://www.cnblogs.com/bymo/p/9026198.html)](https://www.cnblogs.com/bymo/p/9026198.html#_label2)

#### 自动切分

在Keras中，可以从数据集中切分出一部分作为验证集，并且在每次迭代(epoch)时在验证集中评估模型的性能．

具体地，调用**model.fit()**训练模型时，可通过**validation_split**参数来指定从数据集中切分出验证集的比例．

```python
# MLP with automatic validation set
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)
```

validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。

**注意，validation_split的划分在shuffle之前，因此如果你的数据本身是有序的，需要先手工打乱再指定validation_split，否则可能会出现验证集样本不均匀。**



#### 手动切分

Keras允许在训练模型的时候手动指定验证集．

例如，用**sklearn**库中的**train_test_split()**函数将数据集进行切分，然后在**keras**的**model.fit()**的时候通过**validation_data**参数指定前面切分出来的验证集．

```python
# MLP with manual validation set
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=10)
```



#### K折交叉验证

将数据集分成k份，每一轮用其中(k-1)份做训练而剩余1份做验证，以这种方式执行k轮，得到k个模型．将k次的性能取平均，作为该算法的整体性能．k一般取值为5或者10．

- 优点：能比较鲁棒性地评估模型在未知数据上的性能．
- 缺点：计算复杂度较大．因此，在数据集较大，模型复杂度较高，或者计算资源不是很充沛的情况下，可能不适用，尤其是在训练深度学习模型的时候．

**sklearn.model_selection**提供了**KFold**以及**RepeatedKFold, LeaveOneOut, LeavePOut, ShuffleSplit, StratifiedKFold, GroupKFold, TimeSeriesSplit**等变体．

下面的例子中用的StratifiedKFold采用的是分层抽样，它保证各类别的样本在切割后每一份小数据集中的比例都与原数据集中的比例相同．

```python
# MLP for Pima Indians Dataset with 10-fold cross validation
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
  	# create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
    # evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
```

> 这里只用了sklearn做了数据集的切分，没有用sklearn中的cross_val_score()做训练和交叉验证，而是用的keras中的model.fit()进行训练



## *References*

[ResNet 解析 ——CSDN](https://blog.csdn.net/lanran2/article/details/79057994)

