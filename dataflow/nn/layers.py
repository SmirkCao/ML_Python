"""
BaseLayer->ParamLayer->[Dense, ]
->[Pooling,]
"""
import numpy as np
import dataflow
from dataflow.nn.variable import Variable
from .utils import _single, _pair, _triple, get_padded_and_tmp_out


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
    .. math::
       y = xA^T + b

    todo: add bias
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

    def __init__(self,
                 fan_in,
                 fan_out,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="valid",
                 activation=None,
                 # dilation=1,
                 # groups=1,
                 bias=True):
        self.kernel_size = _pair(kernel_size)
        self.strides = _pair(strides)
        super().__init__(w_shape=(fan_in,) + self.kernel_size + (fan_out,),)
        self._a = activation if activation else dataflow.act.tanh

        self.fan_in = fan_in
        self.fan_out = fan_out
        self.padding = padding.lower()
        self._padded = None
        if padding not in ("valid", "same", "full"):
            assert ValueError
        self._p_tblr = None

    def forward(self, x):

        self._x = self._process_input(x)
        # [batch, channel, high, width]
        self._padded, tmp_conved, self._p_tblr = get_padded_and_tmp_out(self._x, self.kernel_size, self.strides, self.fan_out, "same")
        self._wx_b = self.convolution(self._padded, self.w, tmp_conved)
        self._activated = self._a(self._wx_b)
        out = self._wrap_out(self._activated)
        return out

    def backward(self):
        """

        https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e
        关于卷积和互相关这个里面也有说明
        :return:
        """
        dz = self.data_vars["out"].error
        dz *= self._a.derivative(self._wx_b)

        # dw, db
        dw = np.empty_like(self.w)  # [c,h,w,out]
        dw = self.convolution(self._padded.transpose((3, 1, 2, 0)), dz, dw)

        # todo:关于bias的处理
        grads = {"w": dw}

        # dx
        padded_dx = np.zeros_like(self._padded)  # [n, h, w, c]
        s0, s1, k0, k1 = self.strides + self.kernel_size

        t_flt = self.w.transpose((3, 1, 2, 0))  # [c, fh, hw, out] => [out, fh, fw, c]

        for i in range(dz.shape[1]):
            for j in range(dz.shape[2]):
                padded_dx[:, i * s0:i * s0 + k0, j * s1:j * s1 + k1, :] += dz[:, i, j, :].reshape(
                    (-1, self.fan_out)).dot(
                    t_flt.reshape((self.fan_out, -1))
                ).reshape((-1, k0, k1, padded_dx.shape[-1]))
        t, b, l, r = [self._p_tblr[0], padded_dx.shape[1] - self._p_tblr[1],
                      self._p_tblr[2], padded_dx.shape[2] - self._p_tblr[3]]
        self.data_vars["in"].set_error(padded_dx[:, t:b, l:r, :])  # pass error to the layer before

        return grads

    def convolution(self, x, flt, conved):
        """

        在神经网络的范畴讨论卷积的时候, 通常是一个4D Tensor.
        Channel first:  输入ncwh  输出cwhn
        Channel last:   输入nwhc  输出whcn


        ..  note::

            refer to
            https://www.tensorflow.org/guide/performance/overview#use_nchw_imag

            Data formats

            Data formats refers to the structure of the Tensor passed to a given op.
            The discussion below is specifically about 4D Tensors representing images.
            In TensorFlow the parts of the 4D tensor are often referred to by the following letters:

            - N refers to the number of images in a batch.
            - H refers to the number of pixels in the vertical (height) dimension.
            - W refers to the number of pixels in the horizontal (width) dimension.
            - C refers to the channels. For example, 1 for black and white or grayscale and 3 for RGB.
            Within TensorFlow there are two naming conventions representing the two most common data formats:

            NCHW or channels_first
            NHWC or channels_last
            NHWC is the TensorFlow default and NCHW is the optimal format to use when training on NVIDIA GPUs using cuDNN.

        :param x:
        :param flt:
        :param conved:
        :return:
        """
        # number of images in a batch
        batch_size = x.shape[0]
        # default chanel order: channel last, for cpu
        t_flt = flt.transpose((1, 2, 0, 3))  # [c,h,w,out] => [h,w,c,out]
        # strides.shape[0], strides.shape[1], kernel.shape[0], kernel.shape[1]
        s0, s1, k0, k1 = self.strides + tuple(flt.shape[1:3])

        # conved: feature map
        for i in range(0, conved.shape[1]):  # in each row of the convoluted feature map
            for j in range(0, conved.shape[2]):  # in each column of the convoluted feature map
                # [n,h,w,c] => [n, h*w*c]
                # 每个样本一行, 做了拉伸
                x_seg_matrix = x[:, i*s0:i*s0+k0, j*s1:j*s1+k1, :].reshape((batch_size, -1))
                # [h,w,c, out] => [h*w*c, out]
                # flter做拉伸
                flt_matrix = t_flt.reshape((-1, flt.shape[-1]))
                # sum(product)
                # 因为两个都是拉伸过的, 所以做dot, [n, h*w*c] dot [h*w*c, out] -> [n, out]
                filtered = x_seg_matrix.dot(flt_matrix)  # sum of filtered window [n, out]
                # by ref
                conved[:, i, j, :] = filtered
        return conved

    __call__ = forward

    def __repr__(self):
        raise NotImplementedError


class MaxPoll2D(BaseLayer):
    def __init__(self,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="valid",
                 ):
        self.kernel_size = _pair(kernel_size)
        self.strides = _pair(strides)
        super().__init__()

        self.padding = padding.lower()
        self._padded = None
        if padding not in ("valid", "same", "full"):
            assert ValueError
        self._p_tblr = None

    def forward(self, x):
        self._x = self._process_input(x)
        self._padded, out, self._p_tblr = get_padded_and_tmp_out(self._x,
                                                                 self.kernel_size,
                                                                 self.strides,
                                                                 self._x.shepa[-1],
                                                                 self.padding)
        s0, s1, k0, k1 = self.strides + self.kernel_size
        # out is shaped [n, h, w, c]
        for i in range(out.shape[1]):
            for j in range(out.shape[2]):
                window = self._padded[:, i*s0:i*s0+k0, j*s1:j*s1+k1, :]  # [n, h, w, c]
                out[:, i, j, :] = window.max(axis=(1, 2))
        out = None
        wrapped_out = self._wrap_out(out.transpose((0, 3, 1, 2)))
        return wrapped_out

    def backward(self):
        dz = self.data_vars["out"].error
        grad = None
        s0, s1, k0, k1 = self.strides + self.kernel_size
        padded_dx = np.zeros_like(self._padded)  # [n, h, w, c]
        for i in range(dz.shape[1]):
            for j in range(dz.shape[2]):
                window = self._padded[:, i*s0:i*s0+k0, j*s1:j*s1+k1, :]  # [n, fh, fw, c]?
                window_mask = window == np.max(window, axis=(1, 2), keepdims=True)
                window_dz = dz[:, i:i+1, j:j+1, :] * window_mask.astype(np.float32)
                padded_dx[:, i*s0:i*s0+k0, j*s1:j*s1+k1, :] += window_dz
        t, b, l, r = [self._p_tblr[0], padded_dx.shape[1]-self._p_tblr[1],
                      self._p_tblr[2], padded_dx.shape[2]-self._p_tblr[3]]
        self.data_vars["in"].set_error(padded_dx[:, t:b, l:r, :])  #
        return grad


class Flatten(BaseLayer):

    def forward(self, x):
        self._x = self._process_input(x)
        # 每个样本一行
        out = self._x.reshape((self._x.shap[0], -1))
        wrapped_out = self._wrap_out(out)
        return wrapped_out

    def backward(self):
        dz = self.data_vars["out"].error
        grad = None
        self.data_vars["in"].set_error(dz.reshape(self._x.shape))
        return grad
