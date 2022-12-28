![Nite](media/img/banner.jpg)

# About

Nite is a Python neural networks framework built on top of Torch. It focuses on reducing boilerplate.
For example, having to write a training loop is something we unfortunately have to do with vanilla Torch.
This is both boring and prone to bugs. Nite wraps commonly used Torch components in a way convenient to work with.

# Example

```python3
import torch
import nite


dataset = ...

class Net(nite.Net):
  def __init__(self):
    self._convs = nite.Sequential(
      nite.Conv2d(1, 2, 'relu,batchnorm', 2, 'same'),
      nite.Conv2d(2, 3, 'relu,batchnorm', 2, 'same'),
    )
    self._head = nite.Sequential(
      nite.Flatten(),
      nite.Dense(7*7, 30, 'relu'),
      nite.Dense(30, 10)
    )
    
  def forward(self, feed):
    return self._head(self._convs(feed))
    
net = Net()
net.fit(dataset)
```

# Installation

Run the following:

```sh
pip3 install git+https://github.com/patztablook22/nite
```

You should now be able to import the library:

```python3
import nite

...
```
