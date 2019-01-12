#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2019/1/6 21:56"
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import tensorflow as tf
import numpy as np

cifar_dir = './cifar-10-batches-py'
# dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
# CIFAR-10 是一个包含60000张图片的数据集。其中每张照片为32*32的彩色照片，每个像素点包括RGB三个数值，数值范围 0 ~ 255。
with open(os.path.join(cifar_dir, 'data_batch_2'), 'rb') as f:
    data = pickle.load(f, encoding='bytes')


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']

# tensorfloa 的dataset 也可以实现shuffle_data
class CifarData:
    """
    mini batch 训练  shuffle使得数据之前没有联系 更具有泛化能力
    """

    def __init__(self, filenames, need_shuffle):
        all_datas = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)
            for item, labels in zip(data, labels):
                if labels in [0, 1]:
                    all_datas.append(item)
                    all_labels.append(labels)
        self._data = np.vstack(all_datas)
        self._data = self._data / 127.5 -1   # 归一化
        """
        # 纵向合并
        [
        [1 2 3]
        [4 5 6]
        [7 8 9]
        ]"""
        self.num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._labels = np.hstack(all_labels)
        self.indicator = 0  # 指明当前数据集遍历到哪个位置
        """
        # 横向合并
        [1 0 1]
        """
        if self._need_shuffle:
            self._shuffle_data()
        #  [0,1,2,3,4,5] -> [5,2,3,4,1,0]

    def _shuffle_data(self):
        p = np.random.permutation(self.num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        """
        返回batch_size个样本
        :param batch_size:
        :return:
        """
        end_indicator = self.indicator + batch_size
        if end_indicator > self.num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self.indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("没有更多数据遍历完了.")
        if end_indicator > self.num_examples:
            raise Exception("batch_size比数据集还有大")
        batch_data = self._data[self.indicator: end_indicator]
        batch_labels = self._labels[self.indicator: end_indicator]
        self.indicator = end_indicator
        return batch_data, batch_labels


train_filnames = [os.path.join(cifar_dir, 'data_batch_%d' % i) for i in range(1, 6)]  # 所有的数据
test_data_filenames = [os.path.join(cifar_dir, 'test_batch')]

train_data = CifarData(train_filnames, need_shuffle=True)
test_data = CifarData(test_data_filenames, need_shuffle=False)
x = tf.placeholder(tf.float32, [None, 3072])  # None 输入的样本是不确定的
y = tf.placeholder(tf.int64, [None])
# w = tf.get_variable('w', [x.get_shape()[-1], 1], initializer=tf.random_normal_initializer(0, 1))  # 均值是0.0, 方差是1.0

n_hidden_1 = 2560  # 第一层神经元个数
n_hidden_2 = 5890  # 第二层神经元个数
# num_input = 784  # 28*28
num_classes = 1

weights = {
    "weight1": tf.get_variable('w1', [x.get_shape()[-1], n_hidden_1], initializer=tf.random_normal_initializer(0, 1)),
    "weight2": tf.get_variable('w2', [n_hidden_1, n_hidden_2], initializer=tf.random_normal_initializer(0, 1)),
    'out': tf.get_variable('out1', [n_hidden_2, num_classes], initializer=tf.random_normal_initializer(0, 1))

}
bias = {
        "b1": tf.get_variable('b1', [n_hidden_1], initializer=tf.random_normal_initializer(0, 1)),
        "b2": tf.get_variable('b2', [n_hidden_2], initializer=tf.random_normal_initializer(0, 1)),
        'out': tf.get_variable('out2', [num_classes], initializer=tf.random_normal_initializer(0, 1))

}

b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.0))

a1 = tf.matmul(x, weights['weight1']) + bias['b1']  # [None, n_hidden_1]
a2 = tf.matmul(a1, weights['weight2']) + bias['b2']
y_ = tf.matmul(a2, weights['out']) + bias['out']  # [None ,1]
p_y_1 = tf.nn.sigmoid(y_)  # 洛吉斯蒂回归概率值
y_reshaped = tf.reshape(y, [-1, 1])
y_reshaped_float = tf.cast(y_reshaped, tf.float32)
loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))
# loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
predict = p_y_1 > 0.5
# 1,0,1,0...
corrected = tf.equal(tf.cast(predict, tf.int64), y_reshaped)
accuracy = tf.reduce_mean(tf.cast(corrected, tf.float32))  # 这边一定是float32 不能是int

# learning_rate = tf.train.exponential_decay(0.1, global_step=100000, 100, 0.96, staircase=True)
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

init = tf.global_variables_initializer()
batch_size = 20
train_steps = 100000
test_steps = 100
# mini batch训练
with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        loss_val, accuracy_val, _ = sess.run([loss, accuracy, train_op],
                                             feed_dict={
                                                 x: batch_data,
                                                 y: batch_labels
                                             })
        if (i+1) % 500 == 0:
            print("train step :%d , loss:%4.5f, acc:%4.5f" %
                  (i+1, loss_val, accuracy_val))

        if (i+1) % 5000 == 0:
            test_data = CifarData(test_data_filenames, False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
                test_accuracy_val = sess.run([
                    accuracy
                ], feed_dict=
                {x: test_batch_data,
                 y: test_batch_labels}
                )
                all_test_acc_val.append(test_accuracy_val)
            test_acc = np.mean(all_test_acc_val)
            print('[Test ] Step: %d, acc: %4.5f' % (i + 1, test_acc))
