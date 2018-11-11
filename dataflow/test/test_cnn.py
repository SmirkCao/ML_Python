#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  ML_Python
# Filename: test_cnn
# Date: 11/6/18
# Author: 😏 <smirk dot cao at gmail dot com>
import logging
import unittest
import sys
import os
import torch
p = os.path.join(os.getcwd(), "../../")
sys.path.append(p)

from dataflow.examples.cnn import *
from dataflow.optim.optimizer import *
from dataflow.nn.losses import *
from dataflow.datasets.dummy import *


class TestCNN(unittest.TestCase):

    def test_cnn(self):
        logger.info("start to load data")
        x, y = load_mnist()
        logger.info("data loaded")
        train_x, train_y = x[:100], y[:100]
        img_h = 28
        img_w = 28
        channel = 1
        # n, h, w, c
        train_x = train_x.reshape((-1, img_h, img_w, channel))
        max_iter = 100
        clf = CNN()
        opt = Adam(params=clf.params)
        loss_fn = SoftmaxCrossEntropy()

        for n_iter in range(max_iter):
            # train
            pred = clf.forward(x=train_x)
            loss = loss_fn(pred, train_y)
            clf.backward(loss)
            opt.step()
            logger.info("n_iter :%d, loss: %f "% (n_iter, loss.data))
        # test
        pred = clf.forward(train_x)
        logger.info("result: %s" % str(zip(pred, train_y)))
            # acc =


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    unittest.main()

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

