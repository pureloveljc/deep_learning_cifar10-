#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2019/1/8 14:14"
import  tensorflow as tf
import numpy as np

y = np.array([[4], [3]])
print(y.T)
y_one_hot = tf.one_hot(y.T, 10, dtype=tf.float32)
print(y_one_hot)
with tf.Session() as sess:
    a = sess.run(y_one_hot)
    print(a)