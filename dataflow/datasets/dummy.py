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
