from typing import List

import torch
from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from hw_ss.base.base_metric import BaseMetric


class SISDRMetric(BaseMetric):
    def __init__(self, pred="short", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = ScaleInvariantSignalDistortionRatio()
        self.pred = pred


    def __call__(self, mix_short: Tensor, mix_middle: Tensor, mix_long: Tensor, target: Tensor, **kwargs):
        metric = self.si_sdr.to(mix_short.device)
        if self.pred == "short":
            return metric(mix_short, target).item()
        if self.pred == "middle":
            return metric(mix_middle, target).item()
        if self.pred == "long":
            return metric(mix_long, target).item()
        if self.pred == "all":
            return metric((mix_short + mix_middle + mix_long) / 3, target).item()
