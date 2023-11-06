import torch
from torch import Tensor
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

from hw_ss.base.base_metric import BaseMetric
from hw_ss.model import normalize_audio


class PESQMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(fs=kwargs["sample_rate"], mode=kwargs["mode"])


    def __call__(self, mix_short: Tensor, target: Tensor, **kwargs):
        return self.pesq(normalize_audio(mix_short), target).item()
