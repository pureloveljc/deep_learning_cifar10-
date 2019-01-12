#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2019/1/11 17:28"
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

name = '.\images\\1.jpg'  # (1200, 1920, 3)
img_string = tf.read_file(name)
img_decode = tf.image.decode_image(img_string)
img_decode = tf.reshape(img_decode, [1, 1200, 1920, 3])
resize_img = tf.image.resize_bicubic(img_decode, [600, 960])
with tf.Session() as sess:
    img_dvl = sess.run(resize_img)
    img_dvl = img_dvl.reshape((600, 960, 3))
    img_dvl = np.asarray(img_dvl, np.uint8)
    print(img_dvl.shape)
imshow(img_dvl)
plt.show()

# 图像缩放

# tf.image.resize_area
# tf.image.resize_bicubic
# tf.image.resize_nearest_neighbor