import torch

def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)