import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        if in_channels != out_channels:
            self.channels_mismatch = True
            self.channels_matcher = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.channels_mismatch = False
        self.tail = nn.Sequential(
            nn.PReLU(),
            nn.MaxPool1d(3)
        )

    def forward(self, x):
        out = self.head(x) 
        if self.channels_mismatch:
            out += self.channels_matcher(x)
        else:
            out += x
        return self.tail(out)