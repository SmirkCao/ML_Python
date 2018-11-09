#! /usr/bin/env python
# -*- coding=utf-8 -*-
# Project:  ML_Python
# Filename: dummy.py
# Date: 11/2/18
# Author: 😏 <smirk dot cao at gmail dot com>
import numpy as np


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
# TODO: mnist
# TODO: iris
# TODO: 3rd curve
# TODO: sklearn里面有makemoon, makecircle
