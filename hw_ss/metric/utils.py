import torch
import torch.nn as nn


def si_sdr(estimated, target, eps):
    alpha = (target * estimated).sum(dim=-1) / torch.linalg.norm(target, dim=-1) ** 2
    return 20 * torch.log10(torch.linalg.norm(alpha[:, None] * target, dim=-1) / (torch.linalg.norm(alpha[:, None] * target - estimated, dim=-1) + eps) + eps)