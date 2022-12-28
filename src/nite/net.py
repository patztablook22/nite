import torch
import sys
import nite.serialization
import nite.optims

class Net(torch.nn.Module):
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        super(cls, obj).__init__()
        obj.loss = None
        obj.optim = nite.optims.Adam()
        obj.metrics = []
        return obj

    def fit(self, train_dataloader, epochs=1, val_dataloader=None):
        self.optim.setup(self.parameters())
        if not isinstance(self.metrics, list):
            self.metrics = [self.metrics]

        for epoch in range(epochs):
            if epochs == 1:
                prefix = ""
            else:
                prefix = f"::  epoch {str(epoch+1).rjust(len(str(epochs)))}/{epochs}"

            batches = len(train_dataloader)
            with ProgressBar() as bar:
                for batch, (data, target) in enumerate(train_dataloader):
                    pred = self(data)
                    loss = self.loss(pred, target)
                    loss.backward()
                    metrics_results = [metric(pred, target) for metric in self.metrics]
                    metrics_formatted = [f"{m:.2f}".rjust(6) for m in metrics_results]
                    bar.update(f" {prefix}  ::  batch {str(batch+1).rjust(len(str(batches)))}/{batches}  ::  loss {loss:.3f}  ::  metrics {' '.join(metrics_formatted)} ")
                    self.optim.step()

    def save(self, path):
        nite.serialization.save(self, path)


class Dense(torch.nn.Module):
    def __init__(self, prev, size, flags=''):
        super().__init__()
        self._linear = torch.nn.Linear(prev, size)
        activation = 'none'

        tokens = flags.lower().split(',')
        for token in tokens:
            if token in ['relu', 'sigmoid', 'softmax', 'none']:
                activation = token

        if activation == 'none':
            self._activation = torch.nn.Identity()
        elif activation == 'relu':
            self._activation = torch.nn.ReLU()
        elif activation == 'softmax':
            self._activation = torch.nn.Softmax()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()

    def forward(self, feed):
        return self._activation(self._linear(feed))


class Conv2d(torch.nn.Module):
    def __init__(self, prev_channels, channels, flags='', 
                 kernel=3, stride=1, padding='valid'):
        super().__init__()

        activation = 'none'
        batchnorm = False

        tokens = flags.lower().split(',')
        for token in tokens:
            if token in ['relu', 'sigmoid', 'softmax', 'none']:
                activation = token
            if token in ['bn', 'batchnorm', 'batchnormalization']:
                batchnorm = True

        if activation == 'none':
            self._activation = torch.nn.Identity()
        elif activation == 'relu':
            self._activation = torch.nn.ReLU()
        elif activation == 'softmax':
            self._activation = torch.nn.Softmax()
        elif activation == 'sigmoid':
            self._activation == torch.nn.Sigmoid()

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

from torch.nn import Dropout



from torch.nn import Flatten

class Sequential(Net):
    def __init__(self, *layers):
        for i in range(len(layers)):
            self._modules[str(i)] = layers[i]


    def forward(self, feed):
        for submodule in self._modules.values():
            feed = submodule(feed)
        return feed

class ProgressBar:
    def __enter__(self):
        self._len = 0
        return self

    def update(self, s: str):
        sys.stdout.write('\b' * self._len + s)
        sys.stdout.flush()
        self._len = len(s)

    def __exit__(self, type, value, traceback):
        print()
