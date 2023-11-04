import torch
import torch.nn as nn
import torch.nn.functional as F

from hw_ss.model.spex_plus.utils import GlobalLayerNorm


class TcnBlock(nn.Module):
    def __init__(self, 
                 in_channels=256, speaker_channels=256,
                 block_channels=512, kernel_size=3, dilation=1):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels + speaker_channels, block_channels, kernel_size=1),
            nn.PReLU(),
            GlobalLayerNorm(block_channels),
            nn.Conv1d(
                block_channels, block_channels, kernel_size=kernel_size,
                groups=block_channels, padding=dilation * (kernel_size - 1) // 2,
                dilation=dilation
            ),
            nn.PReLU(),
            GlobalLayerNorm(block_channels),
            nn.Conv1d(block_channels, in_channels, kernel_size=1)
        )


    def forward(self, mix, reference=None):
        if reference is not None:
            reference = reference.repeat(1, 1, mix.shape[-1])
            x = torch.cat([mix, reference], dim=1)
        else:
            x = mix
        return self.sequential(x) + mix
    

class TcnStack(nn.Module):
    def __init__(self, num_blocks=8, stem_channels=[256, 512], speaker_channels=256, kernel_size=3):
        super().__init__()
        self.head = TcnBlock(
            in_channels=stem_channels[0],
            speaker_channels=speaker_channels,
            block_channels=stem_channels[1],
            kernel_size=kernel_size,
            dilation=1
        )
        self.tail_blocks = nn.ModuleList([TcnBlock(
            in_channels=stem_channels[0],
            speaker_channels=0,
            block_channels=stem_channels[1],
            kernel_size=kernel_size,
            dilation=2 ** i
        ) for i in range(1, num_blocks)])


    def forward(self, mix, reference):
        mix = self.head(mix, reference)
        for block in self.tail_blocks:
            mix = block(mix)
        return mix
