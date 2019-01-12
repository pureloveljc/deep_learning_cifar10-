#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2019/1/6 21:56"
import pickle
import os
import tensorflow as tf
import numpy as np

cifar_dir = './cifar-10-batches-py'

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
        self._data = self._data / 127.5 - 1  # 归一化
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
x_reshape = tf.reshape(x, [-1, 3, 32, 32])
x_transpose = tf.transpose(x_reshape, perm=[0, 2, 3, 1])
# restnet残差网络
# [Test ] Step: 10000, acc: 0.77400
# [Test ] Step: 10000, acc: 0.74700
# [Test ] Step: 10000, acc: 0.74450


def residual_bloack(x, output_channel):
    """
    每经过一个降采样过程 都会使输出通过翻倍
    如果F(x)和x的channel个数不同怎么办，因为F(x)和x是按照channel维度相加的，channel不同怎么相加呢？
    针对channel个数是否相同，要分成两种情况考虑
    :param x:
    :param output_channel:
    :return:
    """
    input_channel = x.get_shape().as_list()[-1]  # 32
    if input_channel*2 == output_channel:
        increase_dim = True
        strides = (2, 2)
    elif input_channel == output_channel:
        increase_dim = False
        strides = (1, 1)
    else:
        raise Exception("input channel 不能匹配 output channel")
    conv1 = tf.layers.conv2d(x,
                             output_channel,
                             (3, 3),
                             strides=strides,  # 降采样的步长
                             activation=tf.nn.relu,
                             padding='same',
                             name='conv1')
    conv2 = tf.layers.conv2d(conv1,
                             output_channel,
                             (3, 3),
                             strides=(1, 1),
                             padding='same',
                             activation=tf.nn.relu,
                             name='conv2')
    if increase_dim:
        # [none, img_weight, img_height, channel] ->[none, img_weight, img_height, channel*2]
        pooled_x = tf.layers.max_pooling2d(x,
                                           (2, 2),
                                           (2, 2),
                                           padding='valid')  # 图片变成一半
        pooled_x = tf.pad(pooled_x,
                          [[0, 0],
                          [0, 0],
                          [0, 0],
                          [input_channel//2, input_channel//2]])
    else:
        pooled_x = x
    output_x = conv2 + pooled_x
    return output_x


def res_net(
            x,
            num_residual_blocks,  # 残差块
            num_filter_base,  # 32
            class_num  # 10
            ):
    """

    :param x:  输入
    :param num_residual_blocks: 残差连接块
    :param num_subsamping:
    :param num_filter_base:
    :param class_num:
    :return:
    """
    num_subsamping = len(num_residual_blocks)  # 5个残差快
    layers = []
    with tf.variable_scope('conv0'):
        # 32 32 32
        conv0 = tf.layers.conv2d(x,
                                 num_filter_base,
                                 (3, 3),
                                 strides=(1, 1),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='conv1'
                                 )
        layers.append(conv0)
    for i in range(num_subsamping):
        for j in range(num_residual_blocks[i]):
            with tf.variable_scope("conv%d_%d" % (i, j)):
                conv = residual_bloack(layers[-1], num_filter_base*(2**i))
                layers.append(conv)
    with tf.variable_scope('fc'):
        # layer[-1].shape : [None, width, height, channel]
        # kernal_size: image_width, image_height
        global_pool = tf.reduce_mean(layers[-1], [1, 2])
        logits = tf.layers.dense(global_pool, class_num)
        layers.append(logits)
    return layers[-1]

# 全连接

#  [Test ] Step: 10000, acc: 0.75000

y_ = res_net(x_transpose, [4, 6, 10],  32, 10)
# y_ = res_net(x_transpose, [3, 8, 36, 3], 32, 10)
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
scale = 0.001  # l1正则化参数

# y_ = res_net(x_image, [2,3,2], 32, 10)
predict = tf.argmax(y_, 1)  # 输出对应位置的值
# 1,0,1,0...
corrected = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(corrected, tf.float64))
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)


# 给变量建立summary
def variabel_summary(var, name):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))  # 平方差
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.histogram('histogram', var)  # 直方图


with tf.name_scope('summary'):
    variabel_summary(y_, 'resnet')


loss_summary = tf.summary.scalar('loss', loss)
accuracy_summary = tf.summary.scalar('accuracy', accuracy)  # scalar 一个值
souce_image = (x_transpose+1)*127.5  # 原图像
inputs_summary = tf.summary.image('inputs_image', souce_image)
merged_summary = tf.summary.merge_all()   # 所有把掉过的summarymerge 全都merge起来
merged_summary_test = tf.summary.merge([loss_summary, accuracy_summary])
Log_dir = '.'
run_label = 'run_resnet_tensorboard'
run_dir = os.path.join(Log_dir, run_label)
if not os.path.exists(run_dir):
    os.mkdir(run_dir)
train_log_dir = os.path.join(run_dir, 'train')
test_log_dir = os.path.join(run_dir, 'test')
if not os.path.exists(train_log_dir):
    os.mkdir(train_log_dir)
if not os.path.exists(test_log_dir):
    os.mkdir(test_log_dir)


model_dir = os.path.join(run_dir, 'model')
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
saver = tf.train.Saver()
model_name = 'ckpt-10000'
model_path = os.path.join(model_dir, model_name)

init = tf.global_variables_initializer()
batch_size = 20
train_steps = 10000  # [Test ] Step: 10000, acc: 0.67350
test_steps = 100


output_summary_every_steps = 100
output_model_every_steps = 100  # 每隔100步保存一次模型

# mini batch训练
with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)  # sess.graph计算图
    test_writer = tf.summary.FileWriter(test_log_dir)

    fixed_test_batch_data, fixed_test_batch_labels = test_data.next_batch(batch_size)

    if os.path.join(model_path + '.index'):
        saver.restore(sess, model_path)
        print('正在载入模型 %s' % model_path)
    else:
        print('model %s 不存在' % model_path)

    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        eval_ops = [loss, accuracy, train_op]
        summary_ops = ((i+1) % output_summary_every_steps == 0)  # 每100步计算一次summary
        if summary_ops:
            eval_ops.append(merged_summary)
        eval_ops_results = sess.run(eval_ops,
                                             feed_dict={
                                                 x: batch_data,
                                                 y: batch_labels
                                             })
        loss_val, accuracy_val = eval_ops_results[0:2]
        if summary_ops:
            train_summary_str = eval_ops_results[-1]
            train_writer.add_summary(train_summary_str, i+1)
            test_summary_str = sess.run([merged_summary_test],
                                        feed_dict={
                                            x: fixed_test_batch_data,
                                            y: fixed_test_batch_labels
                                        })[0]
            test_writer.add_summary(test_summary_str, i + 1)

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
                 y: test_batch_labels}
                )
                all_test_acc_val.append(test_accuracy_val)
            test_acc = np.mean(all_test_acc_val)
            print('[Test ] Step: %d, acc: %4.5f' % (i + 1, test_acc))

        if (i+1) % output_model_every_steps == 0:
            saver.save(sess, os.path.join(model_dir, 'ckpt-%05d' % (i+1)))
            print('model saver to ckpt-%005d '% (i+1))

# tensorboard
# 1制定面板图显示的变量
# 2训练过程中讲这些变量计算出来，输出到文件中
# 3文件解析 。/tensorboard --logdir=dir
#  tensorboard --logdir=train:E:\pycharmprojects\all_projects\deep_learning_cifar\run_resnet_tensorboard\train,test:E:\pycharmprojects\all_projects\deep_learning_cifar\run_resnet_tensorboard\test

#  迁移学习
# 1 保存模型（第三方）
# 2 载入模型
# 3 冻结前面几层(底层)，训练后面几层  冻结层次只需要 layers 里面的参数trainable 设置成False
