#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  ML_Python
# Filename: losses
# Date: 11/4/18
# Author: 😏 <smirk dot cao at gmail dot com>
import logging
import numpy as np


class Loss(object):
    def __init__(self, loss=None, delta=None):
        self.data = loss
        self.delta = delta

    def __repr__(self):
        return str(self.data)


class LossFunction(object):
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

    def apply(self, pred, target):
        self._pred = pred
        self._target = target
        loss = np.mean(np.square(self._pred - self._target))
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
