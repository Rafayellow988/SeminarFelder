import torch
import torch.nn as nn
import numpy as np

class FourierEmbedding(nn.Module):
    def __init__(self, num_frequencies=2):
        super().__init__()
        self.num_frequencies = num_frequencies
        # Frequencies: [1, 2, 4, ..., 2^(num_frequencies - 1)]
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)

    def forward(self, x):
        x = x * self.freq_bands.to(x.device) * np.pi
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)