#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2019/1/8 14:28"
import numpy as np
import tensorflow as tf

# [None, 10]
a = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
              [0.98, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.96]]
             )
b = tf.argmax(a, 1)
with tf.Session() as sess:
    b = sess.run(b)
    print(b)
    print(b.reshape(-1,1))