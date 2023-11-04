import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, 
                 short_kernel_size=20, middle_kernel_size=80, long_kernel_size=160,
                 encoder_out_channels=256):
        super().__init__()

        self.decoder_short = nn.Sequential(
            nn.ConvTranspose1d(
                encoder_out_channels, 1, kernel_size=short_kernel_size, stride=short_kernel_size // 2
            )
        )
        
        self.decoder_middle = nn.Sequential(
            nn.ConvTranspose1d(
                encoder_out_channels, 1, kernel_size=middle_kernel_size, stride=short_kernel_size // 2
            )
        )

        self.decoder_long = nn.Sequential(
            nn.ConvTranspose1d(
                encoder_out_channels, 1, kernel_size=long_kernel_size, stride=short_kernel_size // 2
            )
        )

    def forward(self, short, middle, long):
        short = self.decoder_short(short)
        middle = self.decoder_middle(middle)
        long = self.decoder_long(long)
        return short, middle, long
