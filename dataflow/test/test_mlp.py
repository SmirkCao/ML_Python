# -*-coding:utf-8-*-
# Project: ML_Python  
# Filename: test_mlp.py
# Author: 😏 <smirk dot cao at gmail dot com>
import sys
import os

p = os.path.join(os.getcwd(), "../../")
sys.path.append(p)

from dataflow.examples.mlp import *
import numpy as np
import unittest
import logging


class TestMLP(unittest.TestCase):

    def test_xor(self):
        X = np.array([[0, 0, 1],
                      [0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]])

        y = np.array([[0],
                      [1],
                      [1],
                      [0]])
        clf = MLP(n_layers=7, hidden_layer_sizes=[10, 5, 4])
        clf.fit(X, y)
        rst = clf.predict(X)
        logger.info(rst)
        np.testing.assert_allclose(rst, y, rtol=0, atol=1e-2)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    unittest.main()

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
