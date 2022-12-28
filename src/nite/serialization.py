import torch
import pickle

def save(obj, path):
    return torch.save(obj, path)

def load(path):
    return torch.load(path)
