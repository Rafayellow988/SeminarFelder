import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import math
import numpy as np
from Residuals import total_loss, wave_equation_residual, boundary_condition_loss, initial_condition_loss
from FourierEmbedding import FourierEmbedding

embedder = FourierEmbedding()


class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0 = 1.,
        c = 6.,
        is_first = False,
        use_bias = True,
        activation = None,
        dropout = 0.
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class WaveNet(nn.Module):
    train_error = []
    test_error = []

    def __init__(self):
        super(WaveNet, self).__init__()
        self.fc1 = nn.Linear(12, 64) # 64 * 3 params = 192
        self.fc2 = nn.Linear(64, 128) # 256 * 64 params = 16'384
        self.fc3 = nn.Linear(128, 256) # 512 * 256 params = 131'072
        self.fc4 = nn.Linear(256, 128) # 128 * 512 params = 65'536
        self.fc5 = nn.Linear(128, 64) # 64 * 128 params = 8'192
        self.fc6 = nn.Linear(64, 1) # 1 * 64 params = 64

        self.t1 = nn.Linear(1, 16)
        self.t2 = nn.Linear(16, 1)

    def forward(self, xyt):
        x = embedder(xyt[:, 0:1])
        y = embedder(xyt[:, 1:2])
        t = xyt[:, 2:3]
        xy = torch.cat([x, y], dim=-1)

        xy = torch.sin(torch.pi * self.fc1(xy))
        xy = -torch.sin(torch.pi * self.fc2(xy)) # (torch.pi / 1.062)
        xy = torch.sin(torch.pi * self.fc3(xy))
        xy = -torch.sin(torch.pi * self.fc4(xy))
        xy = torch.sin(torch.pi * self.fc5(xy))
        xy = self.fc6(xy)

        t = torch.sin(torch.pi * self.t1(t))
        t = self.t2(t)

        return t * xy

    # Train model
    def fit(self, x_train, y_train, t_train, device, n_epochs):
        optimizerAdam = optim.Adam(self.parameters(), lr=0.001)
        optimizerLBFGS = optim.LBFGS(self.parameters(), lr=0.1, max_iter=20, history_size=10)

        for epoch in range(n_epochs):
            if epoch < int(n_epochs/3):
                optimizerAdam.zero_grad()

                train_loss = total_loss(self, x_train, y_train, t_train, device)
                train_loss.backward()
                optimizerAdam.step()

            else:
                def closure():
                    optimizerLBFGS.zero_grad()

                    train_loss = total_loss(self, x_train, y_train, t_train, device)
                    train_loss.backward()
                    return train_loss

                train_loss = optimizerLBFGS.step(closure)

            self.train_error.append(train_loss.item())

            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {0 if epoch == 0 else epoch + 1}, Train-Loss: {train_loss.item():.6f}")
            if epoch + 1 == n_epochs:
                wave_residual = wave_equation_residual(self, x_train, y_train, t_train, device)
                boundary_residual = boundary_condition_loss(self, t_train, device)
                initial_residual = initial_condition_loss(self, x_train, y_train, device)
                print("\nUnweighed final loss: " + str(torch.mean(wave_residual + (1/16) * boundary_residual + (1/5) * initial_residual))+"\n")

        return self.train_error

