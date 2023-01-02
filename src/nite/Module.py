import torch
from typing import Callable, Any, Union, Optional, Iterable
import nite.optims
import sys
import numpy as np


class Training:
    def __init__(self, module):
        self._module = module

        self._loss: Callable = None
        self._flow: list[Any] = [nite.optims.Adam()]
        self._metrics: list[Callable] = []

    @property
    def flow(self) -> list[Any]:
        return self._flow

    @flow.setter
    def flow(self, f):
        if isinstance(f, list):
            self._flow 
        else:
            self._flow = [f]

    @property
    def metrics(self) -> list[Callable]:
        return self._metrics

    @metrics.setter
    def metrics(self, ms):
        if isinstance(ms, list):
            self._metrics = ms
        else:
            self._metrics = [ms]

    @property
    def loss(self) -> Callable:
        return self._loss

    @loss.setter
    def loss(self, l):
        self._loss = l


    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def minimize(self, loss):
        loss.backward()
        flow = self.flow
        if not isinstance(flow, list):
            flow = [flow]

        for f in flow:
            f.step()

    def step(self, data, targets) -> tuple[float, np.ndarray]:
        preds = self._module(data)

        loss: float = self.loss(preds, targets)
        metrics = np.array([m(preds, targets) for m in self.metrics])

        self.minimize(loss)

        return loss, metrics

    def setup(self):
        for f in self.flow:
            f.setup(self._module.parameters())

    def epoch(self, data_train: Iterable, data_val: Optional[Iterable]):
        self.setup()
        batches = len(data_train)
        loss_ema: Optional[float] = None
        metrics_ema: Optional[np.ndarray] = None
        ema_momentum = 0.95

        with ProgressBar() as bar:
            for batch, (data, targets) in enumerate(data_train):
                loss, metrics = self.step(data, targets)

                if loss_ema is None:
                    loss_ema = loss
                    metrics_ema = metrics

                else:
                    loss_ema = ema_momentum * loss_ema + (1 - ema_momentum) * loss
                    metrics_ema = ema_momentum * metrics_ema + (1 - ema_momentum) * metrics

                prefix = ""
                metrics_formatted = [f"{m:.2f}".rjust(6) for m in metrics_ema]
                log = f" {prefix}::  batch {str(batch+1).rjust(len(str(batches)))}/{batches}  ::  loss {loss_ema:.3f}  ::  metrics {' '.join(metrics_formatted)} "

                bar.update(log)

    def start(self, data_train, data_val=None, epochs=1):
        for epoch in range(epochs):
            self.epoch(data_train, data_val)

class ProgressBar:
    def __enter__(self):
        self._len = 0
        return self

    def update(self, s: str):
        sys.stdout.write('\r' + ' ' * self._len + '\r' + s)
        sys.stdout.flush()
        self._len = len(s)

    def __exit__(self, type, value, traceback):
        print()

class Module(torch.nn.Module):
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        super(cls, obj).__init__()
        return obj

    def ftrain(self) -> Training:
        return Training(self)

