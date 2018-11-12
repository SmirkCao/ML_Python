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
        # 1. Input Data
        train_x, train_y = x[:1000], y[:1000]
        test_x, test_y = x[1000:1200], y[1000:1200]

        img_h = 28
        img_w = 28
        channel = 1
        # n, h, w, c
        train_x = train_x.reshape((-1, img_h, img_w, channel))
        test_x = test_x.reshape((-1, img_h, img_w, channel))
        # 2. Epoch
        max_iter = 100
        # 3. Net
        clf = CNN()
        opt = Adam(params=clf.params)
        # 4. Loss
        loss_fn = SparseSoftMaxCrossEntropyWithLogits()
        # 5. Iteration
        for n_iter in range(max_iter):
            # 5.1 Forward
            pred = clf.forward(x=train_x)
            # 5.2 Loss Calculation
            loss = loss_fn(pred, train_y)
            # 5.3 Backward
            clf.backward(loss)
            # 5.4 Optimization
            opt.step()
            logger.info("n_iter :%d, loss: %f "% (n_iter, loss.data))
            # 5.5 Performance Watching
            # acc =
            # logger.info("performance")
        # test
        # need argmax
        pred = clf.forward(test_x)
        logger.info("result: %s %s" % (str(np.argmax(pred.data, axis=-1)), str(test_y.ravel())))
        logger.info(np.sum(np.argmax(pred.data, axis=-1) == test_y.ravel()))
            # acc =


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    unittest.main()

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

