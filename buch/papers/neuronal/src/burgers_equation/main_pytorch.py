import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import qmc

matplotlib.use('TkAgg')
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Data Generation

N_ic, N_bc1, N_bc2 = 50, 25, 25

lhs = qmc.LatinHypercube(d=1)

t_d = lhs.random(n=N_bc1 + N_bc2)
t_d = np.append(np.zeros((N_ic, 1)), t_d, axis=0)

x_d = lhs.random(n=N_ic)
x_d = 2 * (x_d - 0.5)
x_d = np.append(x_d, -1 * np.ones((N_bc1, 1)), axis=0)
x_d = np.append(x_d, +1 * np.ones((N_bc2, 1)), axis=0)

y_d = np.zeros_like(x_d)
y_d[:N_ic] = -np.sin(np.pi * x_d[:N_ic])

N_c = 10000
lhs_2d = qmc.LatinHypercube(d=2)
points = lhs_2d.random(n=N_c)
points[:, 1] = 2 * (points[:, 1] - 0.5)

t_c = points[:, [0]]
x_c = points[:, [1]]

x_d = torch.tensor(x_d, device=device)
t_d = torch.tensor(t_d, device=device)
y_d = torch.tensor(y_d, device=device)
x_c = torch.tensor(x_c, device=device)
t_c = torch.tensor(t_c, device=device)

### Model Definition

class PINN(nn.Module):
    def __init__(self, hidden_layers=8, neurons=20):
        super(PINN, self).__init__()
        layers = [nn.Linear(2, neurons), nn.Tanh()]
        for _ in range(hidden_layers):
            layers += [nn.Linear(neurons, neurons), nn.Tanh()]
        layers += [nn.Linear(neurons, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, t, x):
        input = torch.cat([t, x], dim=1)
        return self.model(input)

model = PINN().to(device)

def u(t, x):
    return model(t, x)

def gradients(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True, retain_graph=True)[0]

def F(t, x):
    t.requires_grad_(True)
    x.requires_grad_(True)
    u_val = u(t, x)
    u_t = gradients(u_val, t)
    u_x = gradients(u_val, x)
    u_xx = gradients(u_x, x)
    return torch.mean((u_t + u_val * u_x - (0.01 / np.pi) * u_xx) ** 2)

def mse(y, y_pred):
    return torch.mean((y - y_pred) ** 2)

### Training

epochs = 15000
opt = optim.Adam(model.parameters(), lr=5e-4)
loss_history = []

if os.path.exists("burgers_net.pth"):
    model.load_state_dict(torch.load("burgers_net.pth"))
    print("Loaded trained model from file")
else:
    print("Training model...")
    for epoch in range(epochs):
        opt.zero_grad()
        y_pred = u(t_d, x_d)
        loss = F(t_c, x_c) + mse(y_d, y_pred)
        loss.backward()
        opt.step()
        loss_history.append(loss.item())
        if (epoch + 1) % 1000 == 0 or epoch == 0:
            print(f"Epoch {0 if epoch == 0 else epoch + 1}, Train-Loss: {loss.item():.6f}")


    torch.save(model.state_dict(), "burgers_net.pth")

plt.figure(figsize=(6, 3))
plt.semilogy(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.tight_layout()
plt.show()

def initial_condition_loss(x_init):
    t0 = np.zeros_like(x_init)
    X0, T0 = np.meshgrid(x_init, t0)
    X_flat = torch.tensor(X0.reshape(-1, 1), device=device)
    T_flat = torch.tensor(T0.reshape(-1, 1), device=device)

    u0_pred = u(T_flat, X_flat)
    u0_true = torch.sin(-np.pi * X_flat)
    return torch.mean((u0_pred + u0_true) ** 2)

def boundary_condition_loss(t_bc):
    x_left = np.zeros_like(t_bc) - 1
    x_right = np.zeros_like(t_bc) + 1

    X_left, t_bcl = np.meshgrid(x_left, t_bc)
    X_right, t_bcr = np.meshgrid(x_right, t_bc)

    X_left_flat = torch.tensor(X_left.reshape(-1, 1), device=device)
    T_left_flat = torch.tensor(t_bcl.reshape(-1, 1), device=device)
    X_right_flat = torch.tensor(X_right.reshape(-1, 1), device=device)
    T_right_flat = torch.tensor(t_bcr.reshape(-1, 1), device=device)

    u_pred_left = u(T_left_flat, X_left_flat)
    u_pred_right = u(T_right_flat, X_right_flat)

    return torch.mean((0.5 * u_pred_left + 0.5 * u_pred_right) ** 2)

### Solution Plot

n, m = 100, 200
X = np.linspace(-1, +1, m)
T = np.linspace(0, 1, n)
X0, T0 = np.meshgrid(X, T)

X_flat = torch.tensor(X0.reshape(-1, 1), device=device)
T_flat = torch.tensor(T0.reshape(-1, 1), device=device)

with torch.no_grad():
    U = u(T_flat, X_flat).cpu().numpy().reshape(n, m)

pde_loss = F(T_flat, X_flat).item()
ic_loss = initial_condition_loss(X).item()
bc_loss = boundary_condition_loss(T).item()
total_loss = (m*n - m - 2*n)/(m*n) * pde_loss + m/(m*n) * ic_loss + 2*n/(m*n) * bc_loss
print(total_loss)

plt.figure(figsize=(8, 4), dpi=150)
plt.pcolormesh(T0, X0, U, cmap='coolwarm', shading='auto')
plt.colorbar(label="u(x, t)")
plt.xlim(0, 1)
plt.ylim(-1, 1)
plt.xlabel("t")
plt.ylabel("x")
plt.title(f"1D Burgers-Gleichung")
plt.tight_layout()
plt.show()
