from typing import List

import torch
from torch import Tensor

from hw_ss.base.base_metric import BaseMetric
from hw_ss.metric.utils import si_sdr
from hw_ss.utils import length_to_mask


class SISDRMetric(BaseMetric):
    def __init__(self, eps=1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps


    def __call__(self, mix_short: Tensor, target: Tensor, **kwargs):
        target = target.squeeze(1)
        mix_short = mix_short.squeeze(1)
        sisdr_short = si_sdr(mix_short, target, self.eps)
        return sisdr_short.mean().item()
