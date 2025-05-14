import torch
import boundaries
import numpy as np

# Analytical solution to the wave equation
def analytical_solution(x, y, t, c=1.0):
    omega = np.pi * c * np.sqrt(2)
    return np.cos(omega * t) * np.sin(np.pi * x) * np.sin(np.pi * y)

# Define the wave equation residual
# Equation 23.4 in the paper
def wave_equation_residual(model, x, y, t, device, c=1.0):
    xyt = torch.cat([x, y, t], dim=1).to(device)
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
    return torch.mean(residual ** 2)


def initial_condition_loss(model, x_init, y_init, device):
    t0 = torch.zeros_like(x_init)
    xyt0 = torch.cat([x_init, y_init, t0], dim=1).to(device)
    xyt0.requires_grad_(True)

    # Predicted displacement
    u0_pred = model(xyt0)

    # True displacement
    u0_true = torch.sin(np.pi * x_init) * torch.sin(np.pi * y_init)

    # First derivative w.r.t. time (initial velocity)
    grad_u = torch.autograd.grad(u0_pred, xyt0, grad_outputs=torch.ones_like(u0_pred), create_graph=True)[0]
    u0_t = grad_u[:, 2:3]

    # Initial velocity is 0 in this example
    loss_disp = torch.mean((u0_pred - u0_true.to(device)) ** 2)
    loss_vel = torch.mean((u0_t - 0.0) ** 2)
    return torch.mean(loss_disp + loss_vel)


def boundary_condition_loss(model, t_bc, device):
    n_samples = len(t_bc)

    # Fix x at boundaries
    x_left = torch.full((n_samples, 1), boundaries.X_MIN)
    x_right = torch.full((n_samples, 1), boundaries.X_MAX)
    y_rand = torch.FloatTensor(n_samples, 1).uniform_(boundaries.Y_MIN, boundaries.Y_MAX)
    xyt_left = torch.cat([x_left, y_rand, t_bc], dim=1).to(device)
    xyt_right = torch.cat([x_right, y_rand, t_bc], dim=1).to(device)

    # Fix y at boundaries
    x_rand = torch.FloatTensor(n_samples, 1).uniform_(boundaries.X_MIN, boundaries.X_MAX)
    y_bottom = torch.full((n_samples, 1), boundaries.Y_MIN)
    y_top = torch.full((n_samples, 1), boundaries.Y_MAX)
    xyt_bottom = torch.cat([x_rand, y_bottom, t_bc], dim=1).to(device)
    xyt_top = torch.cat([x_rand, y_top, t_bc], dim=1).to(device)

    u_left = model(xyt_left)
    u_right = model(xyt_right)
    u_bottom = model(xyt_bottom)
    u_top = model(xyt_top)

    return torch.mean(u_left**2 + u_right**2 + u_bottom**2 + u_top**2)

def total_loss(model, x, y, t, device):
    residual = torch.mean(wave_equation_residual(model, x, y, t, device))
    ic_loss = initial_condition_loss(model, x, y, device)
    bc_loss = boundary_condition_loss(model, t, device)

    # You can weight them if needed
    return 0.8 * residual + 0.1 * ic_loss + 0.1 * bc_loss
