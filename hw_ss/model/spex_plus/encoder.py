import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, 
                 short_kernel_size=20, middle_kernel_size=80, long_kernel_size=160,
                 encoder_out_channels=256):
        super().__init__()

        self.encoder_short = nn.Sequential(
            nn.Conv1d(
                1, encoder_out_channels, kernel_size=short_kernel_size,
                stride=short_kernel_size // 2
            ),
            nn.ReLU()
        )
        
        self.encoder_middle = nn.Sequential(
            nn.ConstantPad1d((0, middle_kernel_size - short_kernel_size), 0),
            nn.Conv1d(
                1, encoder_out_channels, kernel_size=middle_kernel_size,
                stride=short_kernel_size // 2
            ),
            nn.ReLU()
        )

        self.encoder_long = nn.Sequential(
            nn.ConstantPad1d((0, long_kernel_size - short_kernel_size), 0),
            nn.Conv1d(
                1, encoder_out_channels, kernel_size=long_kernel_size,
                stride=short_kernel_size // 2
            ),
            nn.ReLU()
        )

    def forward(self, x):
        x_short = self.encoder_short(x)
        x_middle = self.encoder_middle(x)
        x_long = self.encoder_long(x)
        return x_short, x_middle, x_long
