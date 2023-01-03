![Nite](media/img/banner.jpg)

[this whole thing is work in progress]
## About

Nite is a Python neural networks framework built on top of [Torch](https://pytorch.org/) with the following goals:

1. No extra dependencies. \
   Nite is as lightweight as possible.
   
2. No runtime overhead. \
   Nite makes code cleaner, not slower.
   
3. Flexibility matters. \
   Nite shines in a plenty of scenarios.
   

## Example

```python3
import torch
import nite


data_train, data_test = ...

net = nite.Sequential(
    nite.Conv2d(1, 2, 'relu,batchnorm', 2, 'same'),
    nite.Conv2d(2, 3, 'relu,batchnorm', 2, 'same'),
    nite.flatten,
    nite.Dense(7*7, 30, 'relu'),
    nite.Dense(30, 10)
)
    
net.loss = nite.losses.CrossEntropy()
net.metrics = nite.metrics.CategoricalAccuracy()
    
net.fit(data_train, data_test, epochs=10)
```

## Installation

Run the following:

```sh
pip3 install git+https://github.com/patztablook22/nite
```

You should now be able to import the library:

```python3
import nite

...
```
