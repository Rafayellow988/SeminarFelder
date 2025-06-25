import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import math
import numpy as np
from Residuals import total_loss, wave_equation_residual, boundary_condition_loss, initial_condition_loss
from FourierEmbedding import FourierEmbedding

embedder = FourierEmbedding()

class WaveNet(nn.Module):
    train_error = []
    test_error = []

    def __init__(self):
        super(WaveNet, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.act = nn.Tanh()

        self.t1 = nn.Linear(1, 64)
        self.t2 = nn.Linear(64, 32)
        self.t3 = nn.Linear(32, 1)

    def forward(self, xyt, device):
        x = embedder(xyt[:, 0:1])
        y = embedder(xyt[:, 1:2])
        t = xyt[:, 2:3].to(device)
        xy = torch.cat([x, y], dim=-1).to(device)

        xy = torch.cos(torch.pi * self.fc1(xy))
        xy = torch.cos((torch.pi / 1.062) * self.fc2(xy))
        xy = torch.cos(torch.pi * self.fc3(xy))
        xy = self.fc4(xy)

        t = self.act(self.t1(t))
        t = self.act(self.t2(t))
        t = self.t3(t)

        return t * xy

    # Train model
    def fit(self, x_train, y_train, t_train, x_test, y_test, t_test, device, n_epochs):
        optimizerAdam = optim.Adam(self.parameters(), lr=0.001)
        optimizerLBFGS = optim.LBFGS(self.parameters(), lr=0.1, max_iter=20, history_size=10)

        for epoch in range(n_epochs):
            if epoch < int(n_epochs / 3):
                optimizerAdam.zero_grad()

                train_loss = total_loss(self, x_train, y_train, t_train, device)
                test_loss = total_loss(self, x_test, y_test, t_test, device)
                train_loss.backward()
                optimizerAdam.step()

            else:
                def closure():
                    optimizerLBFGS.zero_grad()

                    train_loss = total_loss(self, x_train, y_train, t_train, device)
                    train_loss.backward()
                    return train_loss

                train_loss = optimizerLBFGS.step(closure)
                test_loss = total_loss(self, x_test, y_test, t_test, device)

            self.train_error.append(train_loss.item())
            self.test_error.append(test_loss.item())

            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {0 if epoch == 0 else epoch + 1}, Train-Loss: {train_loss.item():.6f}, Test-Loss: {test_loss.item():.6f}")
            if epoch + 1 == n_epochs:
                wave_residual = wave_equation_residual(self, x_train, y_train, t_train, device)
                boundary_residual = boundary_condition_loss(self, x_train, y_train, t_train, device, lambd=1)
                initial_residual = initial_condition_loss(self, x_train, y_train, device, lambd=1)
                print("\nUnweighed final train loss: " + str(torch.mean(wave_residual + boundary_residual + initial_residual).item())+"\n")

        return self.train_error, self.test_error

