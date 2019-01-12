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

# [Test ] Step: 53000, acc: 0.81150
# [Test ] Step: 100000, acc: 0.80800
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
        # self._data = self._data / 127.5 - 1  # 归一化
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

batch_size = 20
x = tf.placeholder(tf.float32, [batch_size, 3072])  # None 输入的样本是不确定的
y = tf.placeholder(tf.int64, [batch_size])
is_training = tf.placeholder(tf.bool, [])
x_reshape = tf.reshape(x, [-1, 3, 32, 32])
x_transpose = tf.transpose(x_reshape, perm=[0, 2, 3, 1])

x_transpose = tf.split(x_transpose, num_or_size_splits=batch_size, axis=0)  # 切割
result_x_image_arr = []
for x_single_image in x_transpose:
    x_single_image = tf.reshape(x_single_image, [32, 32, 3])
    data_aug_1 = tf.image.random_flip_left_right(x_single_image)  # 随机左右翻转
    data_aug_2 = tf.image.random_brightness(data_aug_1, max_delta=63)  # 这些参数看文档
    data_aug_3 = tf.image.random_contrast(data_aug_2, lower=0.2, upper=0.8)
    x_single_image = tf.reshape(data_aug_3, [1, 32, 32, 3])
    result_x_image_arr.append(x_single_image)
result_x_images = tf.concat(result_x_image_arr, axis=0)
normal_result_x_images = result_x_images / 127.5 - 1


# VGG 卷积层后面继续添加卷积层
# [Test ] Step: 10000, acc: 0.73850
# batch normalization : conv -> bn  # 批归一化

# 85.6%
def convnet_conv(inputs, filters, kernel_size, activation, padding, name, is_training):
    # conv->bn->activation
    with tf.name_scope(name):
        conv2d = tf.layers.conv2d(inputs,
                                  filters=filters,  # 输出通道数量=卷积核数量
                                  kernel_size=kernel_size,
                                  padding=padding,  # 参数有两个一个是same, 一个是valid ，保证得到的新图与原图的shape相同；VALID则表示不需要填充，valid会减小
                                  # strides=(2, 2),  # 步长  卧槽  这个步长要非常注意，  设置为2 准确率下降很多，图像信息丢失越多
                                  activation=None,
                                  name=name)
        bn = tf.layers.batch_normalization(conv2d, training=is_training)
    return activation(bn)


def pool_conv(inputs, name, pool_size, strides):
    with tf.name_scope(name):
        pool = tf.layers.max_pooling2d(inputs,
                                       pool_size, strides,
                                       padding='valid', data_format='channels_last',
                                       name=name)
    return pool


conv1_1 = convnet_conv(normal_result_x_images, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                       name='conv1_1',
                       is_training=is_training)
conv1_2 = convnet_conv(conv1_1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                       name='conv1_2',
                       is_training=is_training)
conv1_3 = convnet_conv(conv1_2, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                       name='conv1_3',
                       is_training=is_training)
pool1 = pool_conv(conv1_3, name='pool1', pool_size=(2, 2), strides=(2, 2))

conv2_1 = convnet_conv(pool1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                       name='conv2_1',
                       is_training=is_training)
conv2_2 = convnet_conv(conv2_1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                       name='conv2_2',
                       is_training=is_training)
conv2_3 = convnet_conv(conv2_2, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                       name='conv2_3',
                       is_training=is_training)
pool2 = pool_conv(conv2_3, name='pool2', pool_size=(2, 2), strides=(2, 2))

conv3_1 = convnet_conv(pool2, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                       name='conv3_1',
                       is_training=is_training)
conv3_2 = convnet_conv(conv3_1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                       name='conv3_2',
                       is_training=is_training)
conv3_3 = convnet_conv(conv3_2, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                       name='conv3_3',
                       is_training=is_training)
pool3 = pool_conv(conv3_3, name='pool3', pool_size=(2, 2), strides=(2, 2))

flatten = tf.layers.flatten(pool3)
# 全连接
y_ = tf.layers.dense(flatten, 10)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
scale = 0.001  # l1正则化参数

predict = tf.argmax(y_, 1)  # 输出对应位置的值
# 1,0,1,0...
corrected = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(corrected, tf.float64))
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# with tf.control_dependencies(update_ops):
#     with tf.name_scope('train_op'):
#         train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    # train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
    # train_op = tf.train.MomentumOptimizer(learning_rate=1e-4, momentum=0.9).minimize(loss)
Log_dir = '.'
run_label = 'run_vgg_tensorboard'
run_dir = os.path.join(Log_dir, run_label)
if not os.path.exists(run_dir):
    os.mkdir(run_dir)

model_dir = os.path.join(run_dir, 'model')
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

saver = tf.train.Saver()
model_name = 'ckpt-100000'
model_path = os.path.join(model_dir, model_name)

init = tf.global_variables_initializer()

train_steps = 1000000  # [Test ] Step: 10000, acc: 0.67350
test_steps = 100
output_model_every_steps = 1000
# mini batch训练
with tf.Session() as sess:
    sess.run(init)
    # if os.path.join(model_path + '.index'):
    #     saver.restore(sess, model_path)
    #     print('正在载入模型 %s' % model_path)
    # else:
    #     print('model %s 不存在' % model_path)
    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        loss_val, accuracy_val, _ = sess.run([loss, accuracy, train_op],
                                             feed_dict={
                                                 x: batch_data,
                                                 y: batch_labels,
                                                 is_training: True
                                             })
        if (i + 1) % 100 == 0:
            print("train step :%d , loss:%4.5f, acc:%4.5f" %
                  (i + 1, loss_val, accuracy_val))

        if (i + 1) % 1000 == 0:
            test_data = CifarData(test_data_filenames, False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
                test_accuracy_val = sess.run([
                    accuracy
                ], feed_dict=
                {x: test_batch_data,
                 y: test_batch_labels,
                 is_training: True
                 }
                )
                all_test_acc_val.append(test_accuracy_val)
            test_acc = np.mean(all_test_acc_val)
            print('[Test ] Step: %d, acc: %4.5f' % (i + 1, test_acc))

        if (i + 1) % output_model_every_steps == 0:
            saver.save(sess, os.path.join(model_dir, 'ckpt-%05d' % (i + 1)))
            print('model saver to ckpt-%07d ' % (i + 1))
