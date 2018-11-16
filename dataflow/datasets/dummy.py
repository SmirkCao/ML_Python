#! /usr/bin/env python
# -*- coding=utf-8 -*-
# Project:  ML_Python
# Filename: dummy.py
# Date: 11/2/18
# Author: 😏 <smirk dot cao at gmail dot com>
import numpy as np
import struct
import urllib.request
import os
import gzip


def load_dummy():
    # dummy data
    np.random.seed(1)
    x0 = np.random.normal(-2, 1, (100, 2))
    x1 = np.random.normal(2, 1, (100, 2))
    y0 = np.zeros((100, 1), dtype=np.int32)
    y1 = np.ones((100, 1), dtype=np.int32)
    x = np.concatenate((x0, x1), axis=0)
    y = np.concatenate((y0, y1), axis=0)
    return x, y


def load_line():
    np.random.seed(2016)
    x = np.arange(100)
    y = x*0.3 + 0.6
    y += np.random.normal(-2, 5, 100)
    return x, y


def load_curve1():
    x = np.linspace(-5, 6, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x.shape)
    y = np.sin(x) + 3 * np.tanh(x) - 0.5 + noise
    return x, y


def load_xor():
    """
    XOR 问题
    这个问题纯拟合训练集
    :return:
    """
    x = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    return x, y


def load_xor3d():
    """
    3D XOR data
    X = [x1,x2,x3]
    Y = x1 xor x2 xor x3

    ref to https://www.zhihu.com/question/301385613/answer/526433461
    :return: x, y
    """
    data = np.array([[0, 0, 0, 0],
                     [0, 0, 1, 1],
                     [0, 1, 0, 1],
                     [0, 1, 1, 0],
                     [1, 0, 0, 1],
                     [1, 0, 1, 0],
                     [1, 1, 0, 0],
                     [1, 1, 1, 1]])
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def load_mnist():
    def callback(a, b, c):
        """
        回调函数
        @a:已经下载的数据块
        @b:数据块的大小
        @c:远程文件的大小
        """
        per = 100.0 * a * b / c
        if per > 100:
            per = 100
        print('Total size %d bytes. | Downloaded %.2f%%' % (c, per))

    if not os.path.exists("train-images-idx3-ubyte.gz"):
        print("start to download train-images-idx3-ubyte.gz")
        urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                                   'train-images-idx3-ubyte.gz', callback)
    if not os.path.exists("train-labels-idx1-ubyte.gz"):
        print("start to download train-labels-idx1-ubyte.gz")
        urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                                   'train-labels-idx1-ubyte.gz', callback)
    # load train data
    f = open("train-images-idx3-ubyte.gz", "rb")
    raw_data = gzip.GzipFile(mode="rb", fileobj=f).read()
    f.close()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, raw_data, offset)
    offset += struct.calcsize(fmt_header)
    datasize = num_images*num_rows*num_cols
    fmt_header = ">" + str(datasize)+"B"
    x_train = np.array(struct.unpack_from(fmt_header, raw_data, offset))
    x_train = x_train.reshape(num_images, -1)
    print(x_train.shape)

    # load train label
    f = open("train-labels-idx1-ubyte.gz", "rb")
    raw_data = gzip.GzipFile(mode="rb", fileobj=f).read()
    f.close()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_labels = struct.unpack_from(fmt_header, raw_data, offset)
    offset += struct.calcsize(fmt_header)
    datasize = num_labels
    fmt_header = ">" + str(datasize)+"B"
    y_train = np.array(struct.unpack_from(fmt_header, raw_data, offset))
    y_train = y_train.reshape(num_images, -1)
    print(y_train.shape)
    # print(y_train[:10])
    return x_train, y_train

# TODO: iris
# TODO: 3rd curve
# TODO: sklearn里面有makemoon, makecircle
