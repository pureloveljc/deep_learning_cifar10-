#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2019/1/11 15:56"
import tensorflow as tf

# 定义一个简单的计算图，实现向量加法的操作。
input1 = tf.constant([1.0, 2.0, 3.0], name = 'input1')
input2 = tf.Variable(tf.random_uniform([3]), name = 'input2')
output = tf.add_n([input1, input2], name = 'add')

# 生成一个写日志的writer，并将当前的tensorflow计算图写入日志。
# tensorflow提供了多种写日志文件的API
writer = tf.summary.FileWriter('C:/logfile', tf.get_default_graph())
writer.close()
