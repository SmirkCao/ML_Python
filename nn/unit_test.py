# -*-coding:utf-8-*-
# Project: ML_Python  
# Filename: unit_test
# Author: 😏 <smirk dot cao at gmail dot com>
from mlp import *
import numpy as np


def test_mlp():
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
    rst = test_mlp()
