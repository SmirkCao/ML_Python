# -*-coding:utf-8-*-
# Project: ML_Python  
# Filename: mlp
# Author: 😏 <smirk dot cao at gmail dot com>
from functools import reduce
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    """
    derivative sigmoid
    :param x: 
    :return: 
    """
    return x * (1 - x)


class MLP(object):
    def __init__(self,
                 learning_rate=0.0001,
                 max_iter=60000,
                 random_state=2018,
                 n_layers=3,
                 hidden_layer_sizes=[100]
                 ):
        self.learning_rate_ = learning_rate
        self.coefs_ = None
        self.max_iter_ = max_iter
        self.f_ = sigmoid
        self.g_ = sigmoid_grad
        self.random_state_ = random_state
        self.n_layers_ = n_layers
        self.layers_ = None
        self.layer_units = None
        self.hidden_layer_sizes_ = hidden_layer_sizes

    def fit(self, x_, y_):
        np.random.seed(self.random_state_)
        # generate layer_units based on n_layers and hidden_layer_sizes
        fan_in = x_.shape[1]
        fan_out = y_.shape[1]
        n_units = [fan_in] + [100]*(self.n_layers_ - 2 - len(self.hidden_layer_sizes_)) + self.hidden_layer_sizes_ + [fan_out]
        layer_units = []
        for idx in range(len(n_units)-1):
            layer_units.append(tuple(n_units[idx: idx+2]))
        print(layer_units)
        coefs_ = [2 * np.random.random(layer_unit) - 1 for layer_unit in layer_units]
        # print(self.n_layers_, coefs_)  # 3, 2
        for n_iter_ in range(self.max_iter_):
            # Feed forward through layers 0, 1, and 2
            self.layers_ = []
            deltas_ = []
            layer_ = x_
            self.layers_.append(layer_)
            for coef_ in coefs_:
                layer_ = self.f_(np.dot(layer_, coef_))
                self.layers_.append(layer_)
                deltas_.append(1)

            assert(len(self.layers_) == self.n_layers_)
            assert(len(coefs_) == len(deltas_))

            # error
            error_ = y_ - self.layers_[-1]
            deltas_[-1] = error_ * self.g_(self.layers_[-1])
            for idx in range(self.n_layers_ - 1 - 1, 0, -1):
                error_ = deltas_[idx].dot(coefs_[idx].T)
                deltas_[idx-1] = error_ * self.g_(self.layers_[idx])

            # update weights
            for idx in range(self.n_layers_ - 1):
                coefs_[idx] += self.layers_[idx].T.dot(deltas_[idx])
        self.coefs_ = coefs_
        # print(coefs_[0])
        # print(coefs_[1])

    def _inti_coef(self, fan_in_, fan_out_):
        coef_ = 2 * np.random.random((fan_in_, fan_out_)) - 1
        return coef_

    def predict(self, x_):
        # 1 for loop
        # rst = []
        # for layer_ in x_:
        #     for coef_ in self.coefs_:
        #         layer_ = sigmoid(layer_.T.dot(coef_))
        #     rst.append(layer_[0])

        # 2 matrix proud
        # layer_ = x_
        # for coef_ in self.coefs_:
        #     layer_ = sigmoid(layer_.dot(coef_))
        # return layer_

        # 3 reduce
        x_ = [x_]
        x_.extend(self.coefs_)
        return reduce(lambda x, y: self.f_(np.dot(x, y)), x_)

    def summary(self):
        pass


if __name__ == '__main__':
    pass
