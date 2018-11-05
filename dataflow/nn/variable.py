import numpy as np


class Variable(object):
    def __init__(self, v):
        self.info = {}
        self.data = v
        self._error = np.empty_like(v)

    def set_error(self, error):
        assert self._error.shape == error.shape
        self._error[:] = error

    @property
    def error(self):
        return self._error

    @property
    def shape(self):
        return self.data.shape
