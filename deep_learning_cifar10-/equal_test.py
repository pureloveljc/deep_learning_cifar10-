#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2019/1/9 18:28"
import tensorflow as tf
import numpy as np

a = np.array([[1, 0, 1, 1, 0, 1]])
b = np.array([[1, 0, 0, 0, 1, 1]])
c = tf.equal(a, b)
d = tf.reduce_mean(tf.cast(c, tf.float32))
# accuracy = tf.reduce_mean(tf.cast(corrected, tf.float32))
with tf.Session() as sess:
    c, d = sess.run([c, d])
    print(c)
    print(d)