from torch.nn.functional import relu, sigmoid, tanh, softmax

def flatten(batch):
    return batch.reshape([batch.shape[0], -1])
