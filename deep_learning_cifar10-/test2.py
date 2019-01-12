#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2019/1/7 22:05"
import pickle
import os
import numpy as np

cifar_dir = './cifar-10-batches-py'
# dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
# CIFAR-10 是一个包含60000张图片的数据集。其中每张照片为32*32的彩色照片，每个像素点包括RGB三个数值，数值范围 0 ~ 255。
with open(os.path.join(cifar_dir, 'data_batch_2'), 'rb') as f:
    data = pickle.load(f, encoding='bytes')

def load_data(filename):
    with open(filename, 'rb') as f :
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']


class CifarData:
    """
    mini batch 训练  shuffle使得数据之前没有联系 更具有泛化能力
    """
    def __init__(self, filenames, need_shuffle):
        all_datas = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)
            all_datas.append(data)
            all_labels.append(labels)
            # for item, labels in zip(data, labels):
            #     if labels in [0, 1]:
            #         all_datas.append(item)
            #         all_labels.append(labels)
        self._data = np.vstack(all_datas)
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
        print(self._data.shape)
        print(self._labels)


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

