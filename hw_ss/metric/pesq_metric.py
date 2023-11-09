import torch
from torch import Tensor
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

from hw_ss.base.base_metric import BaseMetric
from hw_ss.utils import normalize_audio


class PESQMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(fs=kwargs["sample_rate"], mode=kwargs["mode"])


    def __call__(self, mix_short: Tensor, target: Tensor, **kwargs):
        metric = self.pesq.to(mix_short.device)
        return metric(normalize_audio(mix_short), target).mean().item()
