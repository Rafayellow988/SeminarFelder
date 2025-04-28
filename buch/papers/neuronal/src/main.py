import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

matplotlib.use('TkAgg')


# neural network architecture
class WaveNet(nn.Module):
    train_error = []
    def __init__(self):
        super(WaveNet, self).__init__()
        self.net = nn.Sequential(
            # Input layer with 3 input units for x, y and t
            nn.Linear(3, 32),
            nn.Tanh(),
            # First hidden layer
            nn.Linear(32, 32),
            nn.Tanh(),
            # Output layer with a single output unit for the z-coordinate
            nn.Linear(32, 1)  # Output u(x, y, t) --> Model gibt einzelnen Wert (z-Koordinate) zurück
        )

    def forward(self, x):
        return self.net(x)

    # Train model
    def fit(self, xyt, optimizer, n_epochs):
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            residual = wave_equation_residual(self, xyt)
            loss = torch.mean(residual ** 2)  # Minimize the squared wave equation residual
            loss.backward()
            optimizer.step()

            self.train_error.append(loss.item())

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        return self.train_error

    # Plots training error over training epochs
    def train_error_plot(self):
        epochs = range(1, len(self.train_error) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_error, label='Training Error', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Training Error over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



# Define the wave equation residual
# Equation 23.4 in the paper
def wave_equation_residual(model, xyt, c=2.0):
    xyt.requires_grad_(True)
    u = model(xyt)

    # first-order derivatives
    grad_u = torch.autograd.grad(u, xyt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = grad_u[:, 2:3]  # du/dt

    # second-order derivatives
    grad_u_xyt = torch.autograd.grad(u_t, xyt, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
    u_tt = grad_u_xyt[:, 2:3]  # d^2u/dt^2

    laplacian_u = \
    torch.autograd.grad(grad_u[:, 0:2], xyt, grad_outputs=torch.ones_like(grad_u[:, 0:2]), create_graph=True)[0]
    laplacian_u = laplacian_u[:, 0:1] + laplacian_u[:, 1:2]  # d^2u/dx^2 + d^2u/dy^2

    residual = u_tt - c ** 2 * laplacian_u  # Equivalent to equation 23.4
    return residual


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WaveNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate training points
# Uniformly generated in the x-y-t coordinate system
n_samples = 5000
x_train = torch.FloatTensor(n_samples, 1).uniform_(-2, 2)
y_train = torch.FloatTensor(n_samples, 1).uniform_(-2, 2)
t_train = torch.FloatTensor(n_samples, 1).uniform_(0, 5)
xyt_train = torch.cat([x_train, y_train, t_train], dim=1).to(device)

# Training
train_error = model.fit(xyt_train, optimizer, 200)


# Testing loop
test_error = []
grid_size = 1000
t_test = np.linspace(0, 5, 51)
x_vals_test = np.linspace(-10, 10, grid_size)
y_vals_test = np.linspace(-10, 10, grid_size)
X_test, Y_test = np.meshgrid(x_vals_test, y_vals_test)

u_pred_array = []

for t in t_test:
    xyt_test = torch.FloatTensor(
        np.column_stack([X_test.ravel(), Y_test.ravel(), np.full_like(X_test.ravel(), t)])
    ).to(device)
    u_pred_array.append(model(xyt_test).cpu().detach().numpy().reshape(grid_size, grid_size))
    test_error.append(torch.mean(wave_equation_residual(model, xyt_test) ** 2).item())

print(f"Average test error: {np.mean(test_error)}")

# Evaluates the model at random x and y values in the ranges (-1, 1)
# Creates a contour plot at a single timestep t (t is fixed at 0.5)
def plot_solution(model, t_fixed=0.5):
    grid_size = 100
    x_vals = np.linspace(-1, 1, grid_size)
    y_vals = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)

    xy_t = torch.FloatTensor(np.column_stack([X.ravel(), Y.ravel(), np.full_like(X.ravel(), t_fixed)])).to(device)
    U_pred = model(xy_t).cpu().detach().numpy().reshape(grid_size, grid_size)

    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, U_pred, levels=100, cmap='coolwarm')
    plt.colorbar(label='Wave Height')
    plt.title(f"Wave solution at t = {t_fixed}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# Same as above but updates at every second for t values in [0; 1]
def animate_solution(t_values, X, Y, u_pred_array):
    fig, ax = plt.subplots()
    cbar = None

    def update(frame):
        nonlocal cbar
        t = t_values[frame]
        U_pred = u_pred_array[frame]

        cont = ax.contourf(X, Y, U_pred, levels=100, cmap='coolwarm')
        ax.set_title(f"Wave solution at t = {t:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if cbar is None:
            cbar = fig.colorbar(cont, ax=ax, label='Wave Height')

    ani = animation.FuncAnimation(fig, update, frames=len(t_values), interval=100, repeat=True)
    plt.show()

#plot_solution(model, t_fixed=0.5)
#animate_solution(t_test, X_test, Y_test, u_pred_array)
model.train_error_plot()
