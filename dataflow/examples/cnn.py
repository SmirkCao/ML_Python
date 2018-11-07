#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  ML_Python
# Filename: cnn
# Date: 11/6/18
# Author: 😏 <smirk dot cao at gmail dot com>
import logging
from dataflow.nn.module import *


class CNN(Module):

    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, loss):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

