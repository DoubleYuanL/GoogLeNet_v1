import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

def load_dataset():
    train_dataset = h5py.File('../../datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    # print("1:train_set_x_orig,shape"+str(train_set_x_orig.shape))
    # print("1:train_set_y_orig,shape"+str(train_set_y_orig.shape))

    test_dataset = h5py.File('../../datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    # print("2:test_set_x_orig,shape"+str(test_set_x_orig.shape))
    # print("2:test_set_y_orig,shape"+str(test_set_y_orig.shape))

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# def load_dataset():
#     train_dataset = h5py.File('datasets/flowers/227/flowers_random.h5', "r")
#     train_set_x_orig = np.array(train_dataset["flowers"][:]) # your train set features
#     train_set_y_orig = np.array(train_dataset["label"][:]) # your train set labels

#     test_dataset = h5py.File('datasets/flowers/227/test_flowers_random.h5', "r")
#     test_set_x_orig = np.array(test_dataset["flowers"][:]) # your test set features
#     test_set_y_orig = np.array(test_dataset["label"][:]) # your test set labels

#     classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
#     train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
#     test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
#     return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

#创建占位符
def create_placeholder(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name = "X")
    Y = tf.placeholder(tf.float32, [None,n_y], name = "Y")
    keep_prob = tf.placeholder(tf.float32)
    return X,Y,keep_prob

def inception_module_v1(X):
    Z11 = tf.contrib.layers.conv2d(inputs=X, num_outputs=64, kernel_size=[1,1], stride=[1,1], padding='SAME', activation_fn=tf.nn.relu) 

    Z21 = tf.contrib.layers.conv2d(inputs=X, num_outputs=64, kernel_size=[1,1], stride=[1,1], padding='SAME', activation_fn=tf.nn.relu) 
    Z22 = tf.contrib.layers.conv2d(inputs=Z21, num_outputs=128, kernel_size=[3,3], stride=[1,1], padding='SAME', activation_fn=tf.nn.relu) 

    Z31 = tf.contrib.layers.conv2d(inputs=X, num_outputs=32, kernel_size=[1,1], stride=[1,1], padding='SAME', activation_fn=tf.nn.relu) 
    Z32 = tf.contrib.layers.conv2d(inputs=Z31, num_outputs=32, kernel_size=[5,5], stride=[1,1], padding='SAME', activation_fn=tf.nn.relu) 

    Z41 = tf.contrib.layers.max_pool2d(inputs=X, kernel_size=[3,3], stride=[1,1], padding='SAME')   
    Z42 = tf.contrib.layers.conv2d(inputs=Z41, num_outputs=32, kernel_size=[1,1], stride=[1,1], padding='SAME', activation_fn=tf.nn.relu) 

    ZX = tf.concat([Z11,Z22,Z32,Z42],3)
    return ZX

#前向传播
def forward_propagation(X,keep_prob):
    print("X.shape"+str(X.shape))
    Z1 = tf.contrib.layers.conv2d(inputs=X, num_outputs=64, kernel_size=[7,7], stride=[2,2], padding='SAME', activation_fn=tf.nn.relu) #kernel_size=[7,7]
    Z1 = tf.contrib.layers.max_pool2d(inputs=Z1, kernel_size=[3,3], stride=[2,2], padding='SAME')   

    Z2 = tf.contrib.layers.conv2d(inputs=Z1, num_outputs=64, kernel_size=[1,1], stride=[1,1], padding='SAME', activation_fn=tf.nn.relu) 
    Z2 = tf.contrib.layers.conv2d(inputs=Z2, num_outputs=192, kernel_size=[3,3], stride=[1,1], padding='SAME', activation_fn=tf.nn.relu) 
    Z2 = tf.contrib.layers.max_pool2d(inputs=Z2, kernel_size=[3,3], stride=[2,2], padding='SAME') 

    I1 = inception_module_v1(Z2)#inception (3a)
    I2 = inception_module_v1(I1)#inception (3b)
    # I2 = tf.contrib.layers.max_pool2d(inputs=I2, kernel_size=[3,3], stride=[2,2], padding='SAME')
    I3 = inception_module_v1(I2)#inception (4a)
    A1 = tf.contrib.layers.avg_pool2d(inputs=I3, kernel_size=[5,5], stride=[3,3], padding='VALID')
    Z9 = tf.contrib.layers.conv2d(inputs=A1, num_outputs=16, kernel_size=[1,1], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu)
    Z9 = tf.contrib.layers.flatten(Z9)
    Z9 = tf.contrib.layers.fully_connected(Z9,64,activation_fn=tf.nn.relu)
    Z9 = tf.contrib.layers.fully_connected(Z9,6,activation_fn=None)

    I4 = inception_module_v1(I3)#inception (4b)
    I5 = inception_module_v1(I4)#inception (4c)
    I6 = inception_module_v1(I5)#inception (4d)
    A2 = tf.contrib.layers.avg_pool2d(inputs=I6, kernel_size=[5,5], stride=[3,3], padding='VALID')
    Z15 = tf.contrib.layers.conv2d(inputs=A2, num_outputs=16, kernel_size=[1,1], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu)
    Z15 = tf.contrib.layers.flatten(Z15)
    Z15 = tf.contrib.layers.fully_connected(Z15,64,activation_fn=tf.nn.relu)
    Z15 = tf.contrib.layers.fully_connected(Z15,6,activation_fn=None)

    I7 = inception_module_v1(I6)#inception (4e)
    # I7 = tf.contrib.layers.max_pool2d(inputs=I7, kernel_size=[3,3], stride=[2,2], padding='SAME')
    I8 = inception_module_v1(I7)#inception (5a)
    I9 = inception_module_v1(I8)#inception (5b)
    A3 = tf.contrib.layers.avg_pool2d(inputs=I9, kernel_size=[7,7], stride=[1,1], padding='VALID')
    Z22 = tf.contrib.layers.flatten(A3)
    Z22 = tf.contrib.layers.fully_connected(Z22,6,activation_fn=None)#tf.nn.softmax)
    print("Z22.shape"+str(Z22.shape))
    Z22 = 0.3*Z9 + 0.3*Z15 + 0.4*Z22
    return Z22

#计算loss
def compute_loss(Z6,Y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z6, labels = Y))
    return loss 