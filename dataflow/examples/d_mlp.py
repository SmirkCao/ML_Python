#! /usr/bin/env python
# -*- coding=utf-8 -*-
# Project:  ML_Python
# Filename: d_mlp
# Date: 11/4/18
# Author: 😏 <smirk dot cao at gmail dot com>
import logging
import numpy as np
import sys
import os
p = os.path.join(os.getcwd(), "../../")
sys.path.append(p)
from dataflow.nn.module import Module
from dataflow.datasets.dummy import load_dummy
from dataflow.nn.layers import Dense
from dataflow.optim import optimizer as opt
from dataflow.nn.losses import SigmoidCrossEntropy
from dataflow.nn.activations import sigmoid, tanh
# import matplotlib.pyplot as plt


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

    x, y = load_dummy()
    clf = MLP()
    opt = opt.SGD(params=clf.params, lr=0.1)
    loss_fn = SigmoidCrossEntropy()

    for n_iter in range(100):
        o = clf.forward(x)
        loss = loss_fn(o, y)
        clf.backward(loss)
        opt.step()
        acc = accuracy(o.data < 0.5, y)
        logger.info("Step: %i | loss: %.5f | acc: %.2f" % (n_iter, loss.data, acc))
    logger.info("==")
    # print(clf.forward(x[:10]).data.ravel(), "\n", y[:10].ravel())
    # plt.scatter(x[:, 0], x[:, 1], c=(o.data > 0.5).ravel(), s=100, lw=0, cmap='RdYlGn')
    # plt.show()


else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
