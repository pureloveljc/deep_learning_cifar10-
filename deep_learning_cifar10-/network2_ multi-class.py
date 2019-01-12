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
            # for item, labels in zip(data, labels):
            #     if labels in [0, 1]:
            all_datas.append(data)
            all_labels.append(labels)
        self._data = np.vstack(all_datas)
        self._data = self._data / 127.5 - 1   # 归一化
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
# w = tf.get_variable('w', [x.get_shape()[-1], 10], initializer=tf.random_normal_initializer(0, 1))  # 均值是0.0, 方差是1.0
# b = tf.get_variable('b', [10], initializer=tf.constant_initializer(0.0))
# # [None, 10]
# y_ = tf.matmul(x, w) + b  # [None, 10]
# x_reshape = tf.reshape(-1, 3, 32, 32)
# x_transponse = tf.transpose(x_reshape, perm=[0, 2, 3, 1])
hidden1 = tf.layers.dense(x, 256, activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, 10, activation=tf.nn.relu)
y_ = tf.layers.dense(hidden2, 10, activation=tf.nn.relu)
# e*x/sum(e*x)
# [[0.1, 0.2 ...], [0.2, 0.2, ...]]  # 每个样本都有一个概率分布

# p_y = tf.nn.softmax(y_)
# y_one_hot = tf.one_hot(y, 10, dtype=tf.float32)
# loss = tf.reduce_mean(tf.square(y_one_hot - p_y))


loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
scale = 0.001 # l1正则化参数

# with tf.name_scope("loss"):
#     xentropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
#     #loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
#     base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")
#     reg_loss = tf.reduce_sum(tf.abs(w))
#     loss = tf.add(base_loss, scale * reg_loss, name="loss")

"""
此函数实现
y_- >softmax
y -> one+hot
loss = y*logy_
"""


predict = tf.argmax(y_, 1)  # 输出对应位置的值
# 1,0,1,0...
corrected = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(corrected, tf.float32))
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
