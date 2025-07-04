import torch.nn as nn
import torch.optim as optim
import torch
from Residuals import burgers_equation_residual, initial_boundary_loss, initial_condition_loss, boundary_condition_loss

class BurgersNet(nn.Module):
    def __init__(self, hidden_layers=8, neurons=20):
        super(BurgersNet, self).__init__()
        layers = [nn.Linear(2, neurons), nn.Tanh()]
        for _ in range(hidden_layers):
            layers += [nn.Linear(neurons, neurons), nn.Tanh()]
        layers += [nn.Linear(neurons, 1)]
        self.model = nn.Sequential(*layers)

        self.loss_history = []

    def forward(self, t, x):
        input = torch.cat([t, x], dim=1)
        return self.model(input)

    def fit(self, x_d, t_d, x_c, t_c, u_d, epochs):
        opt = optim.Adam(self.parameters(), lr=5e-4)

        for epoch in range(epochs):
            opt.zero_grad()
            loss = burgers_equation_residual(self, t_c, x_c) + initial_boundary_loss(self, u_d, t_d, x_d)
            loss.backward()
            opt.step()
            self.loss_history.append(loss.item())
            if (epoch + 1) % 1000 == 0 or epoch == 0:
                print(f"Epoch {0 if epoch == 0 else epoch + 1}, Train-Loss: {loss.item():.6f}")

        return self.loss_history