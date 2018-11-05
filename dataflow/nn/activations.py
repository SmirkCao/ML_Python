"""

"""
import numpy as np


class Activation(object):
    def __init__(self):
        pass

    def forward(self, *inputs):
        raise NotImplementedError

    def derivative(self, dz):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)


class Linear(Activation):
    def __init__(self):
        pass

    def forward(self, x):
        return x

    def derivative(self, dz):
        return np.ones_like(dz)


class Sigmoid(Activation):

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, dz):
        y = self.forward(dz)
        return y * (1 - y)


class Tanh(Activation):

    def forward(self, x):
        return np.tanh(x)

    def derivative(self, dz):
        y = self.forward(dz)
        return 1 - np.square(y)


linear = Linear()
sigmoid = Sigmoid()
tanh = Tanh()
