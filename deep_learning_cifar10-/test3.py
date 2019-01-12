#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2019/1/10 20:20"

import tensorflow as tf

import numpy as np

c_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

print(c_tensor.get_shape())
print(c_tensor.get_shape().as_list()[-1])

# with tf.Session() as sess:
#     print(sess.run(tf.shape(a_array)))
#     print(sess.run(tf.shape(b_list)))

