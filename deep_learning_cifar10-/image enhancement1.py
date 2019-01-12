#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2019/1/11 17:28"
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
#
# name = '.\images\\1.jpg'  # (1200, 1920, 3)
# img_string = tf.read_file(name)
# img_decode = tf.image.decode_image(img_string)
# img_decode = tf.reshape(img_decode, [1, 1200, 1920, 3])
# padding_img = tf.image.pad_to_bounding_box(img_decode,
#                                            500, 1000, 5000, 8000)
# with tf.Session() as sess:
#     img_dvl = sess.run(padding_img)
#     img_dvl = img_dvl.reshape((5000, 8000, 3))
#     img_dvl = np.asarray(img_dvl, np.uint8)
#     print(img_dvl.shape)
# imshow(img_dvl)
# plt.show()

# 图像缩放

# tf.image.resize_area
# tf.image.resize_bicubic
# tf.image.resize_nearest_neighbor
# pad_to_bounding_box  加padding
# tf.image.flip_up_down 翻转

# 上下翻转
# name1 = '.\images\\1.jpg'  # (1200, 1920, 3)
# img_string = tf.read_file(name1)
# img_decode1 = tf.image.decode_image(img_string)
# # img_decode1 = tf.reshape(img_decode1, [1, 1200, 1920, 3])
# padding_img = tf.image.flip_up_down(img_decode1)
# with tf.Session() as sess:
#     img_dv2 = sess.run(padding_img)
#     # img_dvl = img_dvl.reshape((5000, 8000, 3))
#     img_dv2 = np.asarray(img_dv2, np.uint8)
#     print(img_dv2.shape)
# imshow(img_dv2)
# plt.show()

# 改变光照  人脸识别中比较常用
name = '.\images\\1.jpg'  # (1200, 1920, 3)
img_string = tf.read_file(name)
img_decode = tf.image.decode_image(img_string)
img_decode = tf.reshape(img_decode, [1, 1200, 1920, 3])
padding_img = tf.image.adjust_brightness(img_decode, 0.4)
with tf.Session() as sess:
    img_dvl = sess.run(padding_img)
    img_dvl = img_dvl.reshape((1200, 1920, 3))
    img_dvl = np.asarray(img_dvl, np.uint8)
    print(img_dvl.shape)
imshow(img_dvl)
plt.show()
