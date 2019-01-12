#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2019/1/6 20:53"
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
cifar_dir = './cifar-10-batches-py'
# dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
# CIFAR-10 是一个包含60000张图片的数据集。其中每张照片为32*32的彩色照片，每个像素点包括RGB三个数值，数值范围 0 ~ 255。
# 10个类，每类6000张图。这里面有50000张用于训练，构成了5个训练批，每一批10000张图；另外10000用于测试
with open(os.path.join(cifar_dir, 'data_batch_2'), 'rb') as f:
    data = pickle.load(f, encoding='bytes')
    # print(type(data[b'data']))  # <class 'numpy.ndarray'>
    # print(type(data[b'labels']))  # <class 'list'>
    # print(type(data[b'batch_label']))  # <class 'bytes'>
    # print(type(data[b'filenames']))  # <class 'list'>
    # print(data[b'data'].shape)  # (10000, 3072)  3072=32*32*3
    # print(data[b'data'][0:2])
    print(data[b'labels'])  # [1, 6]  len(10000) 10000个类别
    # print(data[b'batch_label'])   # 'training batch 2 of 5' # 文件的含义  5个batch中的第2个
    # print(data[b'filenames'][0:2])  # [b'auto_s_000241.png', b'bufo_viridis_s_001109.png']

# 生成一张正确的图片
img_arr = data[b'data'][1]
img_arr = img_arr.reshape(3, 32, 32)
img_arr = img_arr.transpose((1, 2, 0))
img_labels = data[b'labels'][1]
print(img_labels)
imshow(img_arr)
plt.show()