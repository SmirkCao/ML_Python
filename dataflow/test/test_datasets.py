#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  ML_Python
# Filename: test_datasets
# Date: 11/10/18
# Author: 😏 <smirk dot cao at gmail dot com>
import logging
import sys
import os
p = os.path.join(os.getcwd(), "../../")
sys.path.append(p)
from dataflow.datasets.dummy import *
import unittest


class TestDatasets(unittest.TestCase):
    def test_mnist(self):
        load_mnist()
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    unittest.main()

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
