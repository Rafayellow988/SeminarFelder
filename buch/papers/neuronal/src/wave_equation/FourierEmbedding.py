import torch
import torch.nn as nn
import numpy as np

class FourierEmbedding(nn.Module):
    def __init__(self, num_frequencies=3):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies) # 1, 2, 4

    def forward(self, x):
        # Für jedes x: [sin(1*pi*x), cos(1*pi*x], [sin(2*pi*x), cos(2*pi*x], [sin(4*pi*x), cos(4*pi*x]
        # Codiert Perioden in die Daten, bevor diese ins NN gehen

        x = x * self.freq_bands.to(x.device) * np.pi
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)