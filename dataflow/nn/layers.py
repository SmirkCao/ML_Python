"""
BaseLayer->ParamLayer->[Dense, ]
->[Pooling,]
"""
import numpy as np
import dataflow
from dataflow.nn.variable import Variable


class BaseLayer(object):

    def __init__(self):
        self.data_vars = {}
        self.order = None
        self._x = None
        self._activated = None

    def forward(self, x):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def _process_input(self, x):
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32)
            x = Variable(x)
            x.info["new_layer_order"] = 0
        self.data_vars["in"] = x
        self.order = x.info["new_layer_order"]
        _x = x.data
        return _x

    def _wrap_out(self, out):
        out = Variable(out)
        out.info["new_layer_order"] = self.order + 1
        self.data_vars["out"] = out
        return out


class ParamLayer(BaseLayer):

    def __init__(self, w_shape):
        super().__init__()
        self.w = None
        self._wx_b = None
        self._a = None
        self.param_vars = {}

        self.w = np.empty(w_shape, dtype=np.float32)
        self.param_vars["w"] = self.w

    def forward(self, x):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Dense(ParamLayer):
    """
    y = xA^T + b
    """
    def __init__(self, fan_in, fan_out,
                 activation=None):
        w_shape = (fan_in, fan_out)
        super().__init__(w_shape=w_shape)
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.order = 0
        self._a = activation if activation else dataflow.act.linear
        # 参数初始化很重要
        self.w = 2 * np.random.random(w_shape) - 1

    def forward(self, x):
        self._x = self._process_input(x)
        self._wx_b = self._x.dot(self.w)
        self._activated = self._a(self._wx_b)
        wrapped_out = self._wrap_out(self._activated)
        return wrapped_out

    def backward(self):
        # dw, db
        dz = self.data_vars["out"].error
        dz *= self._a.derivative(self._wx_b)
        grads = {"w": self._x.T.dot(dz)}

        # dx
        self.data_vars["in"].set_error(dz.dot(self.w.T))     # pass error to the layer before
        return grads

    __call__ = forward

    def __repr__(self):
        return "Dense: fan_in {} fan_out {}".format(self.fan_in, self.fan_out)


class Conv2D(ParamLayer):
    def forward(self, x):
        pass

    def backward(self):
        pass

    __call__ = forward

    def __repr__(self):
        raise NotImplementedError

