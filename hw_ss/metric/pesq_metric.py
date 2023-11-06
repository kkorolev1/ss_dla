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
        mix_short = mix_short.squeeze(1).detach().cpu().numpy()
        normalized_batch = []
        for x in mix_short:
            normalized_batch += [normalize_audio(x)]
        normalized_batch = Tensor(normalized_batch)
        return self.pesq(normalized_batch.unsqueeze(1), target).item()
