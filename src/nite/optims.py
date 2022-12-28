import torch

class Optim:
    def setup(self, params):
        pass

    def step(self):
        raise NotImplementedError

class Sgd(Optim):
    def __init__(self, lr=1e-3):
        self._lr = lr
        self._internal = None

    def setup(self, params):
        self._internal = torch.nn.SGD(params=params,
                                      lr=self._lr)

    def step(self):
        self._internal.step()

class Adam(Optim):
    def __init__(self, lr=1e-3):
        self._lr = lr
        self._internal = None

    def setup(self, params):
        self._internal = torch.optim.Adam(params=params,
                                       lr=self._lr)

    def step(self):
        self._internal.step()
