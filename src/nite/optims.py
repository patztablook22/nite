import torch

class Optim:
    def setup(self, params):
        pass

    def apply(self):
        raise NotImplementedError

class Sgd(Optim):
    def __init__(self, lr=1e-3):
        self._lr = lr
        self._internal = None

    def setup(self, params):
        self._internal = torch.nn.SGD(params=params,
                                      lr=self._lr)

    def apply(self):
        self._internal.step()

class Adam(Optim):
    def __init__(self, lr=1e-3):
        self._lr = lr
        self._internal = None

    def setup(self, params):
        self._internal = torch.optim.Adam(params=params,
                                       lr=self._lr)

    def apply(self):
        self._internal.step()

class Clip(Optim):
    def __init__(self, value):
        self._value = value
        self._params = None

    def setup(self, params):
        self._params = list(params)

    def apply(self):
        for param in self._params:
            param.grad = torch.clip(param.grad, -self._value, +self._value)
