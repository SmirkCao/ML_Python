#! /usr/bin/env python
# ! -*- coding=utf-8 -*-
# Project:  ML_Python
# Filename: test_gmlp
# Date: 11/2/18
# Author: 😏 <smirk dot cao at gmail dot com>
import sys
import os

p = os.path.join(os.getcwd(), "../../")
sys.path.append(p)

import logging
import unittest
from dataflow.examples.g_clf import *
from dataflow.examples.d_mlp import *
from dataflow.optim.optimizer import SGD
from dataflow.nn.losses import SigmoidCrossEntropy
# import matplotlib.pyplot as plt


class TestCLF(unittest.TestCase):

    def test_g_clf(self):
        x, y = load_dummy()
        clf = Net()
        # print(dir(opt))
        opt = SGD(clf.params, lr=0.1)
        max_iter = 100

        for n_iter in range(max_iter):
            o = clf.forward(x)
            # logger.info(o)
            delta, loss = sigmoid_cross_entropy(o, y)
            # logger.info(delta)
            clf.backward(delta)
            opt.step()
            # 这个和反向传播没有关系, 正向一次就有了这个结果.
            acc = accuracy(o > 0.5, y)
            logger.info("n_iter: %i | loss: %.5f | acc: %.2f" % (n_iter, loss, acc))

    def test_d_mlp(self):
        x, y = load_dummy()
        clf = MLP()
        opt = SGD(params=clf.params, lr=0.1)
        loss_fn = SigmoidCrossEntropy()

        for n_iter in range(100):
            o = clf.forward(x)
            loss = loss_fn(o, y)
            clf.backward(loss)
            opt.step()
            acc = accuracy(o.data < 0.5, y)
            logger.info("n_iter: %i | loss: %.5f | acc: %.2f" % (n_iter, loss.data, acc))
        logger.info("==")
        # print(clf.forward(x[:10]).data.ravel(), "\n", y[:10].ravel())
        # plt.scatter(x[:, 0], x[:, 1], c=(o.data > 0.5).ravel(), s=100, lw=0, cmap='RdYlGn')
        # plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    unittest.main()

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
