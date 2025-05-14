import torch.nn as nn
import torch.optim as optim
from Residuals import total_loss

class WaveNet(nn.Module):
    train_error = []
    test_error = []

    def __init__(self):
        super(WaveNet, self).__init__()
        self.net = nn.Sequential(
            # Input layer with 3 input units for x, y and t
            nn.Linear(3, 10),
            nn.Sigmoid(),
            # First hidden layer
            nn.Linear(10, 50),
            nn.Sigmoid(),
            # Second hidden layer
            nn.Linear(50, 20),
            nn.Sigmoid(),
            # Third hidden layer
            nn.Linear(20, 10),
            nn.Sigmoid(),
            # Output layer with a single output unit for the z-coordinate
            nn.Linear(10, 1)  # Output u(x, y, t)
        )

    def forward(self, x):
        return self.net(x)

    # Train model
    def fit(self, x_train, y_train, t_train, x_test, y_test, t_test, device, n_epochs):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(n_epochs):
            optimizer.zero_grad()

            train_loss = total_loss(self, x_train, y_train, t_train, device)
            test_loss = total_loss(self, x_test, y_test, t_test, device)

            train_loss.backward()
            optimizer.step()

            self.train_error.append(train_loss.item())
            self.test_error.append(test_loss.item())

            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {0 if epoch == 0 else epoch + 1}, Train-Loss: {train_loss.item():.6f}, Test-Loss: {test_loss.item():.6f}")

        return self.train_error, self.test_error
