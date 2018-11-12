#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  ML_Python
# Filename: cnn
# Date: 11/6/18
# Author: 😏 <smirk dot cao at gmail dot com>
import logging
from dataflow.nn.module import *
from dataflow.nn.layers import *


class CNN(Module):

    def __init__(self):
        super().__init__()
        # Conv2D: fan_in, fan_out, kernel_size, strides
        self.l1 = Conv2D(1, 6, (5, 5), (1, 1), "same")
        self.l2 = MaxPoll2D((2, 2), (2, 2))
        self.l3 = Conv2D(6, 16, (5, 5), (1, 1), "same")
        self.l4 = MaxPoll2D((2, 2), (2, 2))
        self.l5 = Flatten()
        self.out = Dense(7*7*16, 10,)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        o = self.out(x)
        return o


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

