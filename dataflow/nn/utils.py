#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  ML_Python
# Filename: utils
# Date: 11/8/18
# Author: 😏 <smirk dot cao at gmail dot com>
from itertools import repeat
import numpy as np


def _ntuple(n):
    def parse(x):
        if isinstance(x, (tuple, list)):
            return tuple(x)
        elif isinstance(x, int):
            return tuple(repeat(x, n))
        else:
            raise TypeError

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)


def get_padded_and_tmp_out(img, kernel_size, strides, fan_out, padding):
    """
    according to: http://machinelearninguru.com/computer_vision/basics/convolution/convolution_layer.html

    :param img: input image
    :param kernel_size:
    :param strides:
    :param fan_out:
    :param padding:
    :return:
    """
    # out nhwc
    n, h, w = img.shape[:3]  # channel last
    # print("img shape", img.shape)
    fh, fw, sh, sw = kernel_size + strides
    # print("fh %d, fw %d, sh %d, sw %d" % (fh, fw, sh, sw))

    if padding == "same":
        out_h, out_w = int(np.ceil(h / sh)), int(np.ceil(w / sw))
        ph = int(np.max([0, (out_h - 1) * sh + fh - h]))
        pw = int(np.max([0, (out_w - 1) * sw + fw - w]))
        pt, pl = int(np.floor(ph / 2)), int(np.floor(pw / 2))
        pb, pr = ph - pt, pw - pl
    elif padding == "valid":
        out_h, out_w = int(np.ceil((h - fh + 1) / sh)), int(np.ceil((w - fw + 1 )/ sw))
        pt, pb, pl, pr = 0, 0, 0, 0
    elif padding == "full":
        # 补了足够多的0, 保证每个点都被卷积到
        out_h, out_w = h, w
        pt, pb, pl, pr = fh - 1, fh - 1, fw - 1, fw - 1
    else:
        raise TypeError
    # np.pad 常用预处理, 数组填充
    padded_img = np.pad(img, ((0, 0), (pt, pb), (pl, pr), (0, 0)), 'constant', constant_values=0.).astype(np.float32)
    # print("padding: %s, out_h %d, out_w %d" % (padding, out_h, out_w))
    # zeros, zeros_like
    tmp_conved = np.zeros((n, out_h, out_w, fan_out), dtype=np.float32)
    return padded_img, tmp_conved, (pt, pb, pl, pr)
