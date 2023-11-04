from typing import List

import torch
from torch import Tensor

from hw_ss.base.base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __call__(self, speaker_logits: Tensor, speaker_id: Tensor, **kwargs):
        speaker_classes = speaker_logits.argmax(dim=-1)
        return (speaker_classes == speaker_id).float().mean().item()
