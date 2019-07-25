#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
from glob import glob
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


# In[8]:


TRUE_IMG_PATH='./True/'
FALSE_IMG_PATH='./False/'
pixel_size=64


true_imgs=glob(TRUE_IMG_PATH+'*')
false_imgs=glob(FALSE_IMG_PATH+'*')
all_imgs=true_imgs+false_imgs


len_imgs=len(true_imgs)+len(false_imgs)
print(len_imgs)

x_input=np.array((len_imgs,pixel_size,pixel_size,1))
x_label=[]

true_label=[]
false_label=[]

def read_img_label(img_list,label):
    label_list=[]
    imgs_array=np.zeros((len(img_list),pixel_size,pixel_size,1))
    for i,img in enumerate(img_list):
        img_array=cv2.resize(cv2.imread(img,0),(pixel_size,pixel_size))
        label_list.append(label)
        imgs_array[i,:,:,0]=img_array
    label_list=np.array(label_list)
    return imgs_array,label_list
        
true_inputs,true_labels = read_img_label(true_imgs,[1])
false_inputs,false_labels = read_img_label(false_imgs,[0])



print(len(true_inputs))
print (len(false_inputs))

        


# In[ ]:





# In[9]:


print (true_inputs.shape)

plt.imshow(true_inputs[10,:,:,0],cmap=plt.cm.gray)
plt.show()

def shuffle_data(a,b,r1):
    assert len(a)==len(b)
    r=range(len(a))
    random.shuffle(r,lambda: r1)
    p=np.array(r)
    
    return a[p],b[p]


data=np.vstack((true_inputs,false_inputs))
labels=np.vstack((true_labels,false_labels))


data,labels=shuffle_data(data,labels,0.1)


split_point=int(round(0.8*len(data)))
(x_train,x_test)=(data[:split_point],data[split_point:])
(y_train,y_test)= (labels[:split_point],labels[split_point:])
print(y_test)
print(len(x_train))
print(len(x_test))
print(len(y_test))
# print(len(y_train))


# In[10]:


from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K

num_classes = 2
epochs = 20

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            for j in range(i + 1, n):
                z1, z2 = digit_indices[d][i], digit_indices[d][j]
                pairs += [[x[z1], x[z2]]]
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                z1, z2 = digit_indices[d][i], digit_indices[dn][j]
                pairs += [[x[z1], x[z2]]]
                labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


# In[11]:


# the data, split between train and test sets
from sklearn.model_selection import StratifiedKFold

(x_train_mi, y_train_mi), (x_test_mi, y_test_mi) = mnist.load_data()

seed = 7
np.random.seed(seed)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

cvscores = []


# print(y_train_mi.shape)
# print('this is y_train_mi', y_train_mi)

for train, test in kfold.split(data, labels):
    x_train = data[train]
    x_test = data[test]
    y_train = labels[train]
    y_test = labels[test]
#     print(x_train.shape)
    x_train = x_train.astype('float32')
    #x_train = x_train.reshape((127, 64, 64))
    x_test = x_test.astype('float32')
    #x_test = x_test.reshape((32, 64, 64))
    #print('ytest_shape', y_test.shape)
#     print(len(y_test))
    y_test = y_test.reshape((len(y_test),))
    y_train = y_train.reshape((len(y_train),))
#     print('this is y_test', y_test)
#     print('this is y_train', y_train)


    x_train /= 255

    x_test /= 255
    input_shape = x_test.shape[1:]
#     print('this is input_shape', input_shape)

    # create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
#     print(digit_indices)
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)
    #print('tr_pairs', tr_pairs)

    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
#     print(digit_indices)
    te_pairs, te_y = create_pairs(x_test, digit_indices)

    # network definition
    base_network = create_base_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    rms = RMSprop()


    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=16,
              epochs=epochs,
               verbose=1,
              validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    scores = te_acc = compute_accuracy(te_y, y_pred)
    cvscores.append(scores * 100)

#     print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))









# In[ ]:





# In[ ]:




