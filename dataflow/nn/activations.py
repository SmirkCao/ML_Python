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


class Softmax(Activation):
    r"""
    多分类
    softmax常用于预测与Multinoulli分布相关联的概率
    定义为

    .. math::
        y_i = softmax(x)_i = \frac{\exp{x_i}}{\sum_{j=1}^n\exp(x_j)}

        输出满足 y_i \in (0,1), \sum_iy_i = 1

    ref to :
    https://deepnotes.io/softmax-crossentropy
    https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    """
    def forward(self, x, axis=-1):
        # overflow handling, numerical stability
        exps = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exps/np.sum(exps+1e-6, axis=axis, keepdims=True)

    def derivative(self, dz):
        return np.ones_like(dz)



linear = Linear()
sigmoid = Sigmoid()
tanh = Tanh()
softmax = Softmax()
