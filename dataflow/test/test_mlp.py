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


class Test_MLP(unittest.TestCase):

    def test_mlp(self):
        X = np.array([[0, 0, 1],
                      [0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]])

        y = np.array([[0],
                      [1],
                      [1],
                      [0]])
        mlpc = MLP(n_layers=7, hidden_layer_sizes=[10, 5, 4])
        mlpc.fit(X, y)
        print(mlpc.predict(X))
        return mlpc


if __name__ == '__main__':
    unittest.main()
