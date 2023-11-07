import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hw_ss.model.spex_plus.encoder import Encoder
from hw_ss.model.spex_plus.utils import LayerNorm
from hw_ss.model.spex_plus.tcn import TcnStack
from hw_ss.model.spex_plus.resnet import ResnetBlock
from hw_ss.model.spex_plus.decoder import Decoder


class SpexPlus(nn.Module):
    def __init__(self, 
                 short_kernel_size=20, middle_kernel_size=80, long_kernel_size=160,
                 encoder_out_channels=256, stem_channels=[256, 512],
                 speaker_channels=256, num_speakers=100,
                 tcn_kernel_size=3, tcn_num_blocks=8):
        super().__init__()

        self.short_kernel_size = short_kernel_size
        self.encoder = Encoder(
            short_kernel_size=short_kernel_size,
            middle_kernel_size=middle_kernel_size,
            long_kernel_size=long_kernel_size,
            encoder_out_channels=encoder_out_channels
        )
        
        self.mix_post_encoder = nn.Sequential(
            LayerNorm(3 * encoder_out_channels),
            nn.Conv1d(3 * encoder_out_channels, stem_channels[0], kernel_size=1)
        )

        self.speaker_backbone = nn.Sequential(
            LayerNorm(3 * encoder_out_channels),
            nn.Conv1d(3 * encoder_out_channels, stem_channels[0], kernel_size=1),
            ResnetBlock(stem_channels[0], stem_channels[0]),
            ResnetBlock(stem_channels[0], stem_channels[1]),
            ResnetBlock(stem_channels[1], stem_channels[1]),
            nn.Conv1d(stem_channels[1], speaker_channels, kernel_size=1)
        )

        self.tcn_stacks = nn.ModuleList([
            TcnStack(
                num_blocks=tcn_num_blocks, 
                stem_channels=stem_channels,
                speaker_channels=speaker_channels,
                kernel_size=tcn_kernel_size
            ) for _ in range(4)
        ])

        self.masks_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(stem_channels[0], encoder_out_channels, kernel_size=1),
                nn.ReLU()
            ) for _ in range(3)
        ])

        self.decoder = Decoder(
            short_kernel_size=short_kernel_size,
            middle_kernel_size=middle_kernel_size,
            long_kernel_size=long_kernel_size,
            encoder_out_channels=encoder_out_channels
        )

        self.speaker_head = nn.Linear(speaker_channels, num_speakers)


    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
    

    def _reference_length_after_encoder(self, reference_length):
        new_length = (reference_length - self.short_kernel_size) // (self.short_kernel_size // 2) + 1
        return ((new_length // 3) // 3) // 3


    def forward(self, mix, reference, reference_length, **kwargs):
        mix_length = mix.shape[-1]
        mix_tuple = self.encoder(mix)
        mix = self.mix_post_encoder(
            torch.cat(mix_tuple, dim=1)
        )
        reference_tuple = self.encoder(reference)
        new_reference_length = self._reference_length_after_encoder(reference_length).float()
        reference = self.speaker_backbone(
            torch.cat(reference_tuple, dim=1)
        ).sum(dim=-1, keepdim=True) / new_reference_length[:, None, None]
        for tcn_stack in self.tcn_stacks:
            mix = tcn_stack(mix, reference)
        masked_mixes = []
        for mask_layer, mix_after_encoder in zip(self.masks_layers, mix_tuple):
            masked_mixes.append(mix_after_encoder * mask_layer(mix))
        mix_short, mix_middle, mix_long = self.decoder(*masked_mixes)
        mix_short = F.pad(mix_short, (0, mix_length - mix_short.shape[-1]))
        speaker_logits = self.speaker_head(reference.squeeze(-1))
        return {
            "mix_short": mix_short,
            "mix_middle": mix_middle[:, :, :mix_length],
            "mix_long": mix_long[:, :, :mix_length],
            "speaker_logits": speaker_logits
        }

# import torch
# model = SpexPlus()
# mix = torch.ones((3, 1, 3313))
# reference = torch.ones((3, 1, 600))
# out = model(mix, reference, torch.ones(3))
# print([(k, v.shape) for k,v in out.items()])