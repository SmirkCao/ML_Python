#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  ML_Python
# Filename: losses
# Date: 11/4/18
# Author: 😏 <smirk dot cao at gmail dot com>
import logging
import numpy as np


class Loss(object):
    """
    Loss base
    """
    def __init__(self, loss=None, delta=None):
        self.data = loss
        self.delta = delta

    def __repr__(self):
        return str(self.data)


class LossFunction(object):
    """
    loss function base class
    """
    def __init__(self):
        self._pred = None
        self._target = None

    def apply(self, pred, target):
        raise NotImplementedError

    @property
    def delta(self):
        raise NotImplementedError

    def __call__(self, pred, target):
        return self.apply(pred, target)


class MSE(LossFunction):
    r"""Creates a criterion that measures the mean squared error between
    `n` elements in the predicted `x` and target `y`.

    The loss can described as

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size.

    """
    def apply(self, pred, target):
        r"""

        :param pred:
        :param target:
        :return:
        """
        self._pred = pred
        self._target = target
        # 2 for gradient
        loss = np.mean(np.square(self._pred - self._target))/2
        return loss, self.delta

    def delta(self):
        return self._pred - self._target


class CrossEntropy(LossFunction):
    def __init__(self):
        super().__init__()
        self._eps = 1e-6

    def apply(self, pred, target):
        raise NotImplementedError

    @property
    def delta(self):
        raise NotImplementedError


class SigmoidCrossEntropy(CrossEntropy):

    def apply(self, pred, target):
        p = self._pred = pred.data
        t = self._target = target
        loss = Loss()
        loss.data = -np.mean(t * np.log(p + self._eps) + (1 - t) * np.log(1 - p + self._eps))
        loss.delta = self.delta
        return loss

    @property
    def delta(self):
        return self._pred - self._target


class SoftmaxCrossEntropy(CrossEntropy):

    def apply(self, pred, target):
        p = self._pred = pred.data
        t = self._target = target
        loss = Loss()
        loss.data = -np.mean(np.sum(t*np.log(p), axis=-1))
        loss.delta = self.delta
        return loss

    @property
    def delta(self):
        return self._pred - self._target


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
