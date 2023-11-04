import torch
import torch.nn as nn
import torch.nn.functional as F

from hw_ss.metric import si_sdr
from hw_ss.utils import length_to_mask


class SpexPlusLoss(nn.Module):
    def __init__(self, ce_scale=0.5, middle_scale=0.1, long_scale=0.1, eps=1e-6):
        super().__init__()
        self.ce_scale = ce_scale
        self.middle_scale = middle_scale
        self.long_scale = long_scale
        self.eps = eps
    

    def forward(self, mix_short, mix_middle, mix_long, speaker_logits, speaker_id, target, **kwargs):
        target = target.squeeze(1)

        mix_short = mix_short.squeeze(1)
        mix_middle = mix_middle.squeeze(1)
        mix_long = mix_long.squeeze(1)

        sisdr_short = si_sdr(mix_short, target, self.eps)
        sisdr_middle = si_sdr(mix_middle, target, self.eps)
        sisdr_long = si_sdr(mix_long, target, self.eps)

        sisdr_loss = (-(1 - self.middle_scale - self.long_scale) * sisdr_short.sum() -
                     self.middle_scale * sisdr_middle.sum() - self.long_scale * sisdr_long.sum()) / mix_short.shape[0]
        ce_loss = F.cross_entropy(speaker_logits, speaker_id)

        return sisdr_loss + self.ce_scale * ce_loss
