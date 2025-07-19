import torch
import numpy as np


def gradients(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True, retain_graph=True)[0]

def burgers_equation_residual(model, t, x):
    t.requires_grad_(True)
    x.requires_grad_(True)
    u_val = model(t, x)
    u_t = gradients(u_val, t)
    u_x = gradients(u_val, x)
    u_xx = gradients(u_x, x)
    return torch.mean((u_t + u_val * u_x - (0.01 / np.pi) * u_xx) ** 2)

def initial_boundary_loss(model, u_d, t_d, x_d):
    u_pred = model(t_d, x_d)
    return torch.mean((u_d - u_pred) ** 2)



def initial_condition_loss(x_init, model, device):
    t0 = np.zeros_like(x_init)
    X0, T0 = np.meshgrid(x_init, t0)
    X_flat = torch.tensor(X0.reshape(-1, 1), device=device)
    T_flat = torch.tensor(T0.reshape(-1, 1), device=device)

    u0_pred = model(T_flat, X_flat)
    u0_true = -torch.sin(np.pi * X_flat)
    return torch.mean((u0_pred + u0_true) ** 2)

def boundary_condition_loss(t_bc, model, device):
    x_left = np.zeros_like(t_bc) - 1
    x_right = np.zeros_like(t_bc) + 1

    X_left, t_bcl = np.meshgrid(x_left, t_bc)
    X_right, t_bcr = np.meshgrid(x_right, t_bc)

    X_left_flat = torch.tensor(X_left.reshape(-1, 1), device=device)
    T_left_flat = torch.tensor(t_bcl.reshape(-1, 1), device=device)
    X_right_flat = torch.tensor(X_right.reshape(-1, 1), device=device)
    T_right_flat = torch.tensor(t_bcr.reshape(-1, 1), device=device)

    u_pred_left = model(T_left_flat, X_left_flat)
    u_pred_right = model(T_right_flat, X_right_flat)

    return torch.mean((u_pred_left + u_pred_right) ** 2)