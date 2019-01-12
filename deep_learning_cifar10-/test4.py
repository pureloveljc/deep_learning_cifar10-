#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2019/1/11 11:33"
import tensorflow as tf
import numpy as np

# max_pooling_shape = max_pooling.get_shape().as_list()[1:]  # 8 8 32
# input_shape = x.get_shape().as_list()[1:]  # 16 16 16
# width_padding = (input_shape[0] - max_pooling_shape[0]) // 2
# height_padding = (input_shape[1] - max_pooling_shape[1]) // 2
# padded_pooling = tf.pad(max_pooling,
#                         [[0, 0],
#                          [width_padding, width_padding],
#                          [height_padding, height_padding],
#                          [0, 0]])
a = np.random.randn(1,8, 8, 32)
b = np.random.randn(1,16, 16, 16)
width_padding = (b.shape[1] - a.shape[1]) // 2
height_padding = (b.shape[2] - a.shape[2]) // 2
print(width_padding)
print(height_padding)
padded_pooling = tf.pad(a,
                        [[0, 0],
                         [width_padding, width_padding],
                         [height_padding, height_padding],
                         [0, 0]])

# padded_pooling = tf.pad(max_pooling,
#                         [[0, 0],
#                          [width_padding, width_padding],
#                          [height_padding, height_padding],
#                          [0, 0]])
print(padded_pooling.shape)