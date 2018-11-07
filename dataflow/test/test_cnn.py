#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  ML_Python
# Filename: test_cnn
# Date: 11/6/18
# Author: 😏 <smirk dot cao at gmail dot com>
import logging
import unittest
from dataflow.examples.cnn import *


class TestCNN(unittest.TestCase):

    def test_cnn(self):
        x = None
        clf = CNN()
        clf.forward(x=x)
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    unittest.main()

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

