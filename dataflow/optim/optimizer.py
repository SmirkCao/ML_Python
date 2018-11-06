class Optimizer(object):
    def __init__(self, params, lr):
        self._params = params
        self._lr = lr
        self._vars = []
        self._grads = []

        for layer_p in self._params.values():
            for p_name in layer_p["vars"].keys():
                self._vars.append(layer_p["vars"][p_name])
                self._grads.append(layer_p["grads"][p_name])

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):

    def step(self):
        # var is a np.ndarray, -= operation is by ref
        for var, grad in zip(self._vars, self._grads):
            var -= self._lr * grad


class Momentum(Optimizer):
    pass

