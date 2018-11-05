#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  ML_Python
# Filename: module
# Date: 11/4/18
# Author: 😏 <smirk dot cao at gmail dot com>
import logging
from dataflow.nn.layers import *
from dataflow.nn.losses import Loss


class Module(object):
    def __init__(self):
        self.params = {}
        self._ordered_layers = []

    def forward(self, x):
        raise NotImplementedError

    def backward(self, loss):
        assert isinstance(loss, Loss)

        # find net order
        layers = []
        for name, v in self.__dict__.items():
            if not isinstance(v, BaseLayer):
                continue
            layer = v
            layer.name = name
            layers.append((layer.order, layer))
        self._ordered_layers = [l[1] for l in sorted(layers, key=lambda x: x[0])]

        # back propagate through this order
        last_layer = self._ordered_layers[-1]

        last_layer.data_vars["out"].set_error(loss.delta)
        for layer in self._ordered_layers[::-1]:
            grads = layer.backward()
            if isinstance(layer, ParamLayer):
                for k in layer.param_vars.keys():
                    self.params[layer.name]["grads"][k][:] = grads[k]

    def __setattr__(self, key, value):
        if isinstance(value, ParamLayer):
            layer = value
            self.params[key] = {
                "vars": layer.param_vars,
                "grads": {k: np.empty_like(layer.param_vars[k]) for k in layer.param_vars.keys()}
            }
        object.__setattr__(self, key, value)

    __call__ = forward


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
