from typing import List

import torch
from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from hw_ss.base.base_metric import BaseMetric


class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = ScaleInvariantSignalDistortionRatio()

    def __call__(self, mix_short: Tensor, target: Tensor, **kwargs):
        metric = self.si_sdr.to(mix_short.device)
        return metric(mix_short, target).item()
