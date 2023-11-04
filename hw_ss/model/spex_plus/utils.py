import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Performs layer normalization on channels of a sequence
    Expects input with shape: B x C x L
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.layer_norm(
            x.transpose(1, 2)
        ).transpose(1, 2)
    

class GlobalLayerNorm(nn.Module):
    """
    Performs normalization across sequence and channels dimensions
    Expects input with shape: B x C x L
    Code based on nn.LayerNorm
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, 1))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)


    def forward(self, x):
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.var(x, (1, 2), unbiased=False, keepdim=True)
        if self.elementwise_affine:
            return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
        return (x - mean) / torch.sqrt(var + self.eps)