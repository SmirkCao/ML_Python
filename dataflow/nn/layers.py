"""
BaseLayer->ParamLayer->[Dense, ]
->[Pooling,]
"""
import numpy as np
import sys
import os
p = os.path.join(os.getcwd(), "../../")
sys.path.append(p)
import dataflow.nn.activations as act


class BaseLayer(object):

    def __init__(self):
        pass

    def forward(self, x):
        raise NotImplementedError

    def backward(self, loss):
        raise NotImplementedError


class ParamLayer(BaseLayer):

    def __init__(self):
        self.w = None
        self.wx_b = None
        self.order = None
        self._a = None

    def forward(self, x):
        raise NotImplementedError

    def backward(self, loss):
        raise NotImplementedError


class Dense(ParamLayer):
    """
    y = xA^T + b
    """
    def __init__(self, fan_in, fan_out,
                 activation=None):
        w_shape = (fan_in, fan_out)
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.w = np.empty(w_shape)
        self.order = 0
        self._a = activation if activation else act.linear

    def forward(self, x):
        self.wx_b = x.dot(self.w)
        return self._a(self.wx_b)

    def backward(self, loss):
        pass

    def __call__(self, x):
        self.forward(x)

    def __repr__(self):
        return "Dense: fan_in {} fan_out {}".format(self.fan_in, self.fan_out)
