from nite.Module import Module
import nite.optims
import torch

from torch.nn import Dropout, Flatten, Sigmoid, Tanh, Softmax, Identity
from torch.nn import ReLU as Relu
from torch.nn import GELU as Gelu

from torch.nn import MultiheadAttention
from torch.nn import LayerNorm
from torch.nn import Embedding


class Dense(Module):
    def __init__(self, prev, size, flags=''):
        self._linear = torch.nn.Linear(prev, size)
        activation = 'none'

        tokens = flags.lower().split(',')
        for token in tokens:
            if token in ['relu', 'tanh', 'sigmoid', 'softmax', 'none', 'gelu']:
                activation = token

        if activation == 'none':
            self._activation = Identity()
        elif activation == 'relu':
            self._activation = Relu()
        elif activation == 'gelu':
            self._activation = Gelu()
        elif activation == 'softmax':
            self._activation = Softmax()
        elif activation == 'sigmoid':
            self._activation = Sigmoid()
        elif activation == 'tanh':
            self._activation = Tanh()

    def forward(self, feed):
        return self._activation(self._linear(feed))


class Conv2d(Module):
    def __init__(self, prev_channels, channels, flags='', 
                 kernel=3, stride=1, padding='valid'):

        activation = 'none'
        batchnorm = False

        tokens = flags.lower().split(',')
        for token in tokens:
            if token in ['relu', 'gelu', 'tanh', 'sigmoid', 'softmax', 'none']:
                activation = token
            if token in ['bn', 'batchnorm', 'batchnormalization']:
                batchnorm = True

        if activation == 'none':
            self._activation = Identity()
        elif activation == 'relu':
            self._activation = Relu()
        elif activation == 'gelu':
            self._activation = Gelu()
        elif activation == 'softmax':
            self._activation = Softmax()
        elif activation == 'sigmoid':
            self._activation = Sigmoid()
        elif activation == 'tanh':
            self._activation = Tanh()

        if batchnorm:
            self._batchnorm = torch.nn.BatchNorm2d(channels)
        else:
            self._batchnorm = torch.nn.Identity()

        if padding == 'valid':
            padding_explicit = 0
        else:
            padding_explicit = (kernel - 1) // 2

        self._conv = torch.nn.Conv2d(prev_channels, channels, 
                                     kernel, stride, padding_explicit,
                                     bias=not batchnorm)

    def forward(self, feed):
        return self._activation(self._batchnorm(self._conv(feed)))
