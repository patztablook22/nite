import torch

class CategoricalAccuracy:
    def __call__(self, predicted, expected):
        return torch.mean((torch.argmax(predicted, axis=-1) == expected).to(torch.float32))
