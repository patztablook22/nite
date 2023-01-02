from nite.Module import Module
import nite.optims
import torch
import sys
import numpy as np
from typing import Callable, Any, Union
import time
import datetime
import threading


class Net(Module):
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        super(cls, obj).__init__()
        obj.loss: Callable = None
        obj._optim: list = [nite.optims.Adam()]
        obj._metrics: list[Callable] = []
        return obj

    @property
    def optim(self) -> list:
        return self._optim

    @optim.setter
    def optim(self, optim):
        if isinstance(optim, (list, tuple)):
            self._optim = list(optim)
        else:
            self._optim = [optim]

    @property
    def metrics(self) -> list[Callable]:
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        if isinstance(metrics, (list, tuple)):
            self._metrics = list(metrics)
        else:
            self._metrics = [metrics]

    def step(self, data, targets) -> tuple[float, dict]:
        preds = self(data)
        loss = self.loss(preds, targets)
        metrics = {m: m(preds, targets) for m in self.metrics}

        loss.backward()
        for op in self.optim:
            op.apply()
        
        return loss, metrics

    def epoch(self, train_data, val_data, logger=None):
        batches = len(train_data)

        if logger is None:
            logger = Logger()

        with logger:
            logger.begin_epoch(batches)
            for data, targets in train_data:
                logger.begin_batch()
                loss, metrics = self.step(data, targets)
                logger.end_batch(loss, metrics)

            logger.end_epoch()


    def fit(self, train_data, val_data=None, epochs=1):
        with Logger() as logger:
            logger.begin_fit(epochs)

            for op in self.optim:
                op.setup(self.parameters())

            for _ in range(epochs):
                self.epoch(train_data, val_data, logger=logger)

            logger.end_fit()

    def save(self, path):
        nite.serialization.save(self, path)


class Sequential(Net):
    def __init__(self, *layers):
        self.layers = layers
        for i in range(len(layers)):
            if hasattr(layers[i], 'named_modules'):
                self._modules[str(i)] = layers[i]

    def forward(self, feed):
        for layer in self.layers:
            feed = layer(feed)
        return feed

class ProgressBar:
    animation = '⣾⣽⣻⢿⡿⣟⣯⣷'

    def __init__(self, prefix=""):
        self._prefix=prefix
        self._len = 0
        self._spinner_i = 0
        self._spinner_s = ProgressBar.animation[0]
        self._thread = None
        self._text = ''
        self._exit = False

    def __enter__(self):
        sys.stdout.write(self._prefix)
        self._len = len(self._prefix)
        self._thread = threading.Thread(target=self._spinner)
        self._thread.start()
        return self

    def _render(self):
        sys.stdout.write('\033[F' + ' ' * self._len + '\r ' + \
                         self._spinner_s + self._text)
        sys.stdout.flush()

    def _spinner(self):
        while True:
            if self._exit:
                break

            time.sleep(0.05)
            self._spinner_i = (self._spinner_i + 1) % len(ProgressBar.animation)
            self._spinner_s = ProgressBar.animation[self._spinner_i]
            self._render()

    def log(self, s: str):
        self._text = self._prefix + s
        self._len = len(self._text) + 2

    def __exit__(self, type, value, traceback):
        self._exit = True
        self._thread.join()
        self._spinner_s = '*'
        self._render()
        print()

class Logger:
    animation = '⣾⣽⣻⢿⡿⣟⣯⣷'

    def __init__(self):
        self._epoch = 0
        self._batch = 0
        self._epochs = None
        self._batches = None

        self._epoch_begin_time = None
        self._eta_seconds = None
        self._loss_ema = None
        self._metrics_ema = None

        self._buffer_animation = " "
        self._buffer_epoch = ""
        self._buffer_batch = ""
        self._buffer_eta = ""
        self._buffer_loss = ""
        self._buffer_metrics = ""

        self._erase = []

        self._paused = False
        self._closed = False
        self._animation_i = 0

        self._thread = None

    def __enter__(self):
        self._closed = False
        self._paused = False

        def renderer():
            while not self._closed:
                time.sleep(0.08)
                if self._paused: 
                    continue

                self._render()
                self._animation_i = (self._animation_i + 1) % len(Logger.animation)
                self._buffer_animation = Logger.animation[self._animation_i]

        self._thread = threading.Thread(target=renderer)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._closed = True
        self._thread.join()


    def _render(self):
        up = '\033[F'

        buffer_animation = self._buffer_animation
        buffer_epoch = self._buffer_epoch
        buffer_batch = self._buffer_batch
        buffer_eta = self._buffer_eta
        buffer_loss = self._buffer_loss
        buffer_metrics = self._buffer_metrics

        lines = [
            f" {buffer_animation}   {buffer_epoch}{buffer_batch}{buffer_eta}",
            f"",
            f"         {buffer_loss}",
            f"         {buffer_metrics} ",
            f"",
            f"",
        ]


        s = up.join(["\r" + " " * erase for erase in reversed(self._erase)]) + "\r" + "\n\r".join(lines)

        self._erase = [len(line) for line in lines]

        sys.stdout.write(s)
        sys.stdout.flush()


    def begin_fit(self, epochs):
        self._epochs = epochs

    def end_fit(self):
        pass

    def begin_epoch(self, batches):
        self._batch = 0
        self._epoch += 1
        self._batches = batches

        if self._epochs:
            se = str(self._epoch)
            ses = str(self._epochs)
            self._buffer_epoch = f"epoch {se.rjust(len(ses))}/{ses}  ::  "
        else:
            self._buffer_epoch = ""

        self._epoch_begin_time = datetime.datetime.now()

    def end_epoch(self):
        self._paused = True
        self._buffer_animation = "*"
        self._buffer_eta = ""
        self._buffer_batch = ""
        self._buffer_epoch = self._buffer_epoch[:-4]
        self._render()
        self._erase = []
        self._paused = False

    def begin_batch(self):
        self._batch += 1

        if self._batches:
            sb = str(self._batch)
            sbs = str(self._batches)
            self._buffer_batch = f"batch {sb.rjust(len(sbs))}/{sbs}  ::  "
        else:
            self._buffer_batch = ""

    def end_batch(self, loss: float, metrics: dict):
        if self._epoch_begin_time is not None and self._batches is not None:
            dur_seconds = (datetime.datetime.now() - self._epoch_begin_time).seconds
            linear = int(dur_seconds / self._batch * (self._batches - self._batch))
            self._eta_seconds = linear

            total = self._eta_seconds
            hours = total // 3600
            minutes = (total % 3600) // 60
            seconds = total % 60

            if hours:
                self._buffer_eta = f"ETA {hours} h {minutes} min {seconds} s"
            elif minutes:
                self._buffer_eta = f"ETA {minutes} min {seconds} s"
            else:
                self._buffer_eta = f"ETA {seconds} s"

        momentum = 0.95
        if self._loss_ema is None:
            self._loss_ema = loss
            self._metrics_ema = metrics
        else:
            self._loss_ema = momentum * self._loss_ema + (1 - momentum) * loss
            for key in metrics:
                self._metrics_ema[key] = momentum * self._metrics_ema[key] + (1 - momentum) * metrics[key]

        self._buffer_loss = f"loss {self._loss_ema:.3f}  "

        mb = [f"{k.__class__.__name__} {v:.3f}" for k, v in self._metrics_ema.items()]
        self._buffer_metrics = "  ::  ".join(mb)
