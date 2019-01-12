#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2019/1/9 11:13"
import tensorflow as tf
import numpy as np

x = np.random.randn(4, 12)
print(x)
x_reshape = tf.reshape(x, [-1, 4, 3])
x_transpose = tf.transpose(x_reshape, perm=[0, 2 ,1])

with tf.Session() as sess:
    a = sess.run(x_reshape)
    b = sess.run(x_transpose)
    print(a)
    print('~~~~~~~~~~~~~~')
    print(b)