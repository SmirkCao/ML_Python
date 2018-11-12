#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  ML_Python
# Filename: initializer
# Date: 11/12/18
# Author: 😏 <smirk dot cao at gmail dot com>
import numpy as np


class BaseInitializer:
    """
    ref to: https://github.com/MorvanZhou/simple-neural-networks/blob/master/neuralnets/initializers.py
    """
    def initialize(self, x):
        raise NotImplementedError


class TruncatedNormal(BaseInitializer):
    def __init__(self, mean=0., std=1.):
        self._mean = mean
        self._std = std

    def initialize(self, x):
        x[:] = np.random.normal(loc=self._mean, scale=self._std, size=x.shape)
        truncated = 2*self._std + self._mean
        x[:] = np.clip(x, -truncated, truncated)
