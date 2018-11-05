#! /usr/bin/env python
# -*- coding=utf-8 -*-
# Project:  ML_Python
# Filename: d_mlp
# Date: 11/4/18
# Author: 😏 <smirk dot cao at gmail dot com>
import logging
import numpy as np
from dataflow.nn.module import Module
from dataflow.nn.layers import Dense
from dataflow.nn.activations import sigmoid, tanh


def accuracy(predictions, labels):
    assert predictions.shape == labels.shape
    p, l = predictions.astype(np.int32), labels.astype(np.int32)
    return np.where(p == l, 1., 0.).mean()


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.l1 = Dense(fan_in=2, fan_out=10, activation=tanh)
        self.l2 = Dense(fan_in=10, fan_out=10, activation=tanh)
        self.out = Dense(fan_in=10, fan_out=1, activation=sigmoid)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        o = self.out(x)
        return o


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
