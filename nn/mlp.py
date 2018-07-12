# -*-coding:utf-8-*-
# Project: ML_Python  
# Filename: nn
# Author: 😏 <smirk dot cao at gmail dot com>
from functools import reduce
import numpy as np


def sigmoid(x_):
    return 1 / (1 + np.exp(-x_))


def sigmoid_grad(y_):
    """
    
    :param y_: sigmoid(x)
    :return: 
    """
    return y_ * (1 - y_)


class MLP(object):
    def __init__(self,
                 learning_rate=0.0001,
                 max_iter=60000,
                 random_state=2018,
                 n_layers=3
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

    def fit(self, x_, y_):
        np.random.seed(self.random_state_)
        # todo update this method, related to n_layers
        layer_units = [(3, 5), (5, 5), (5, 1)]
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

            assert (len(self.layers_) == self.n_layers_)
            assert (len(coefs_) == len(deltas_))

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
