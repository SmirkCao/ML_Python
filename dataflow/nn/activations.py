"""

"""
import numpy as np


class Activation(object):
    def __init__(self):
        pass

    def forward(self, *inputs):
        raise NotImplementedError

    def derivate(self, dz):
        raise NotImplementedError

    def __call__(self, x):
        self.forward(x)


class Linear(Activation):
    def __init__(self):
        pass

    def forward(self, x):
        return x

    def derivate(self, dz):
        return np.ones_like(dz)

    def __call__(self, x):
        return self.forward(x)


class Sigmoid(Activation):
    def __init__(self):
        pass


class Tanh(Activation):
    pass


linear = Linear()
sigmoid = Sigmoid()
tanh = Tanh()
