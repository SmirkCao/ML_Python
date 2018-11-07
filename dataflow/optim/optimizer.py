import numpy as np


class Optimizer(object):
    def __init__(self, params, lr):
        self._params = params
        self._lr = lr
        self._vars = []
        self._grads = []
        self._name = None

        for layer_p in self._params.values():
            for p_name in layer_p["vars"].keys():
                self._vars.append(layer_p["vars"][p_name])
                self._grads.append(layer_p["grads"][p_name])

    def step(self):
        raise NotImplementedError

    @property
    def info(self):
        return dict(lr=self._lr, name=self._name)


class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, lr)
        self._name = "Gradient Decent"

    def step(self):
        # var is a np.ndarray, -= operation is by ref
        for var, grad in zip(self._vars, self._grads):
            var -= self._lr * grad


class Momentum(Optimizer):
    r"""

    http://www.cs.toronto.edu/~hinton/absps/momentum.pdf

    .. note::
       Ref to torch
       The implementation of SGD with Momentum/Nesterov subtly differs from
       Sutskever et. al. and implementations in some other frameworks.

       Considering the specific case of Momentum, the update can be written as

       .. math::
           v = \rho * v + g \\
           p = p - lr * v

       where p, g, v and :math:`\rho` denote the parameters, gradient,
       velocity, and momentum respectively.

       This is in contrast to Sutskever et. al. and
       other frameworks which employ an update of the form

       .. math::
            v = \rho * v + lr * g \\
            p = p - v

       The Nesterov version is analogously modified.

    """

    def __init__(self, params, lr=0.001, momentum=0.9):
        super().__init__(params, lr)
        self._name = "Momentum GD"
        self.momentum = momentum

    def step(self):
        for var, grad, mv in zip(self._vars, self._grads, self._mv):
            # mv[:] by ref, -= by ref
            # like DVR
            mv[:] = self.momentum*mv + self._lr*grad
            var -= mv

