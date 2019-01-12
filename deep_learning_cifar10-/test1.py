#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2019/1/6 21:42"
import numpy as np
import tensorflow as tf

import math

all_data = []
all_labels = []
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([1, 0, 1])
for item, label in zip(a, b):
    all_data.append(item)
    all_labels.append(label)
print(all_data)
print('~~~~~~')
print(all_labels)
datas = np.vstack(all_data)
print(datas)
print('~~~~~~')
labels = np.hstack(all_labels)
print(labels)
print('~~~~~~')
p = np.random.permutation(3)
print(p)
c = a[p]
print(c)