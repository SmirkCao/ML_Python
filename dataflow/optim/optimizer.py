import numpy as np


class Optimizer(object):
    def __init__(self, params, lr):
        self._params = params
        self._lr = lr
        self._vars = []
        self._grads = []
        self._name = None
        self._eps = 1e-7

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
        """
        batch越大, 越接近GD; 随机来自batch
        :param params:
        :param lr:
        """
        super().__init__(params, lr)
        self._name = "Stochastic Gradient Decent"

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


class AdaGrad(Optimizer):
    r"""
    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    Ref to Deep Learning: CH08 Algo_8.4

    """
    def __init__(self, params, lr=0.001, delta=1e-7):
        super().__init__(params=params, lr=lr)
        self._name = "AdaGrad"
        self._delta = delta
        self._rs = None

    def step(self):
        for var, grad, r in zip(self._vars, self._grads, self._rs):
            # like DVR
            r += grad*grad
            var -= self._lr/(self._delta + np.sqrt(r))*grad


class RMSProp(Optimizer):
    r"""
    RMSProp [Hilton]_

    Ref to Deep Learning: CH08 Algo_8.5

    .. [Hilton] http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

    """
    def __init__(self, params, lr=0.001, delta=1e-6, rho=0.99):
        super().__init__(params=params, lr=lr)
        self._name = "RMSProp"
        self._delta = delta
        self._rho = rho
        self._rs = None

    def step(self):
        for var, grad, r in zip(self._vars, self._grads, self._rs):
            r[:] = self._rho * r + (1 - self._rho) * np.square(grad)
            var -= self._lr/(np.sqrt(r + self._delta))*grad


class Adam(Optimizer):
    r"""

    """
    def __init__(self, params, lr=0.001, rho1=0.9, rho2=0.999, delta=1e-8):
        super().__init__(params=params, lr=lr)
        self._name = "Adam"
        self._delta = delta
        self._rho1 = rho1
        self._rho2 = rho2
        self._rs = [np.zeros_like(v) for v in self._vars]
        self._ss = [np.zeros_like(v) for v in self._vars]
        self._rho1_ = 1
        self._rho2_ = 1

    def step(self):
        for var, grad, r, s in zip(self._vars, self._grads, self._rs, self._ss):
            s[:] = self._rho1 * s + (1 - self._rho1)*grad
            r[:] = self._rho2 * r + (1 - self._rho2) * np.square(grad)
            self._rho1_ = self._rho1_ * self._rho1
            self._rho2_ = self._rho2_ * self._rho2
            s_ = s/(1 - self._rho1_)
            r_ = r/(1 - self._rho2_)
            var -= self._lr*s_/(np.sqrt(r_) + self._delta)
