#! /usr/bin/env python
# ! -*- coding=utf-8 -*-
# Project:  ML_Python
# Filename: g_clf
# Date: 10/31/18
# Author: 😏 <smirk dot cao at gmail dot com>
import logging
import numpy as np
import matplotlib.pyplot as plt

"""
在mlp.py的基础上, 开始解耦
划分下面几个部分:
1. Net
2. Dense Layer
3. Activation
4. Loss
5. Optimizer
6. Risk
"""

"""
Dummy Data
"""


def load_dummy():
    # dummy data
    np.random.seed(1)
    x0 = np.random.normal(-2, 1, (100, 2))
    x1 = np.random.normal(2, 1, (100, 2))
    y0 = np.zeros((100, 1), dtype=np.int32)
    y1 = np.ones((100, 1), dtype=np.int32)
    x = np.concatenate((x0, x1), axis=0)
    y = np.concatenate((y0, y1), axis=0)
    return x, y


"""
Dense Layer
"""


class Dense(object):
    # output = activation(dot(input, kernel) + bias)
    # y = (wx+b)
    def __init__(self, fan_in, fan_out,
                 activation=None):
        self.w = None
        self.order = None
        self.name = None
        self.x = None

        w_shape = (fan_in, fan_out)
        self.w = 2 * np.random.random(w_shape) - 1
        self._wx_b = None

        if activation is None:
            self._a = linear
        elif isinstance(activation, Activation):
            self._a = activation
        else:
            raise TypeError

    def forward(self, x):
        self.x = x
        self._wx_b = self.x.dot(self.w)
        return self._a(self._wx_b)

    def backward(self, dz):
        # dw, db
        dz = dz * self._a.derivative(self._wx_b)
        grads = self.x.T.dot(dz)
        # dx
        dx = dz.dot(self.w.T)
        return dx, grads

    def __call__(self, *args, **kwargs):
        return self.forward(*args)


"""
Activations
"""


class Activation(object):
    def forward(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)


class Linear(Activation):
    def forward(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


class Sigmoid(Activation):
    def forward(self, x):
        return 1 / (np.exp(-x) + 1)

    def derivative(self, x):
        y = self.forward(x)
        return y * (1 - y)


class Tanh(Activation):
    def forward(self, x):
        return np.tanh(x)

    def derivative(self, x):
        y = np.tanh(x)
        return 1. - np.square(y)


sigmoid = Sigmoid()
linear = Linear()
tanh = Tanh()

"""
Loss Function
"""


def sigmoid_cross_entropy(p, q):
    # p pred
    # q ground truth
    loss = - np.mean(q * np.log(p + 1e-6) + (1. - q) * np.log(1 - p + 1e-6))
    # logger.info(loss.shape)
    return p - q, loss


"""
SGD Optimizer
"""


class SGD(object):
    def __init__(self, params, lr):
        self._params = params
        self._lr = lr
        self.vars = []
        self.grads = []

        for layer_p in self._params.values():
            for p_name in layer_p["vars"].keys():
                self.vars.append(layer_p["vars"][p_name])
                self.grads.append(layer_p["grads"][p_name])

    def step(self):
        # vars means variants
        for var, grad in zip(self.vars, self.grads):
            var -= self._lr * grad


def accuracy(predictions, labels):
    assert predictions.shape == labels.shape
    p, l = predictions.astype(np.int32), labels.astype(np.int32)
    return np.where(p == l, 1., 0.).mean()


class Net(object):
    # Network
    def __init__(self):
        self._ordered_layers = []
        self.params = {}
        # layers definition
        self.l1 = Dense(fan_in=2, fan_out=10, activation=tanh)
        self.l2 = Dense(fan_in=10, fan_out=10, activation=tanh)
        self.out = Dense(fan_in=10, fan_out=1, activation=sigmoid)

        self.l1.order = 0
        self.l2.order = 1
        self.out.order = 2

    def forward(self, x):
        # layers relation
        x = self.l1(x)
        x = self.l2(x)
        o = self.out(x)
        return o

    def backward(self, delta):
        # find net order
        layers = []
        for name, v in self.__dict__.items():
            if not isinstance(v, Dense):
                continue
            layer = v
            layer.name = name
            layers.append((layer.order, layer))

        self._ordered_layers = [l[1] for l in sorted(layers, key=lambda x: x[0])]

        # back propagate through this order
        dz = delta
        for layer in self._ordered_layers[::-1]:
            dz, grads = layer.backward(dz)
            if grads.any():
                self.params[layer.name]["grads"]["w"][:] = grads

    def __call__(self, *args):
        return self.forward(*args)

    def __setattr__(self, key, value):
        if isinstance(value, Dense):
            layer = value
            self.params[key] = {
                "vars": {"w": layer.w},
                "grads": {"w": np.empty_like(layer.w)}
            }

        object.__setattr__(self, key, value)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    x, y = load_dummy()
    clf = Net()
    # print(dir(opt))
    opt = SGD(clf.params, lr=0.1)
    max_iter = 100

    for n_iter in range(max_iter):
        o = clf.forward(x)
        # logger.info(o)
        delta, loss = sigmoid_cross_entropy(o, y)
        # logger.info(delta)
        clf.backward(delta)
        opt.step()
        # 这个和反向传播没有关系, 正向一次就有了这个结果.
        acc = accuracy(o > 0.5, y)
        logger.info("n_iter: %i | loss: %.5f | acc: %.2f" % (n_iter, loss, acc))

    plt.scatter(x[:, 0], x[:, 1], c=(o > 0.5).ravel(), s=100, lw=0, cmap='RdYlGn')
    plt.show()
