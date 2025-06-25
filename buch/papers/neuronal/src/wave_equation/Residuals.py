import torch
import boundaries
import numpy as np

# Analytical solution to the wave equation
def analytical_solution(xyt, c=1.0):
    x = xyt[:, 0:1]
    y = xyt[:, 1:2]
    t = xyt[:, 2:3]

    omega = torch.pi * c * np.sqrt(2)
    return torch.cos(omega * t) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

# Define the wave equation residual
# Equation 23.4 in the paper
def wave_equation_residual(model, x, y, t, device, c=1.0):

    xyt = torch.cat([x, y, t], dim=-1).to(device)
    xyt.requires_grad_(True)

    u = model(xyt, device)
    u.requires_grad_(True)

    grad_u = torch.autograd.grad(u, xyt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = grad_u[:, 2:3]  # du/dt

    grad_u_xyt = torch.autograd.grad(u_t, xyt, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
    u_tt = grad_u_xyt[:, 2:3]  # d^2u/dt^2

    laplacian_u = \
    torch.autograd.grad(grad_u[:, 0:2], xyt, grad_outputs=torch.ones_like(grad_u[:, 0:2]), create_graph=True)[0]
    laplacian_u = laplacian_u[:, 0:1] + laplacian_u[:, 1:2]  # d^2u/dx^2 + d^2u/dy^2

    residual = u_tt - c ** 2 * laplacian_u  # equation 23.4
    return residual ** 2


def initial_condition_loss(model, x_init, y_init, device, lambd=4):
    t0 = torch.zeros_like(x_init)

    xyt0 = torch.cat([x_init, y_init, t0], dim=-1).to(device)
    xyt0.requires_grad_(True)

    # NN output for u(x, y, 0)
    u0_pred = model(xyt0, device)
    u0_pred.requires_grad_(True)

    # True value for u(x, y, 0)
    u0_true = torch.sin(np.pi * x_init) * torch.sin(np.pi * y_init)

    grad_u = torch.autograd.grad(u0_pred, xyt0, grad_outputs=torch.ones_like(u0_pred), create_graph=True)[0]
    u0_t = grad_u[:, 2:3] # du/dt

    loss_disp = (u0_pred - u0_true.to(device)) ** 2
    loss_vel = u0_t ** 2 # d/dt u(x, y, 0) = 0
    return lambd * loss_disp + loss_vel


def boundary_condition_loss(model, x_rand, y_rand, t_bc, device, lambd=4):
    n_samples = len(t_bc)

    # x on boundaries, y & t anything
    x_left = torch.full((n_samples, 1), boundaries.X_MIN)
    x_right = torch.full((n_samples, 1), boundaries.X_MAX)

    xyt_left = torch.cat([x_left, y_rand, t_bc], dim=-1).to(device)
    xyt_left.requires_grad_(True)
    xyt_right = torch.cat([x_right, y_rand, t_bc], dim=-1).to(device)
    xyt_right.requires_grad_(True)

    # y on boundaries, x & t anything
    y_bottom = torch.full((n_samples, 1), boundaries.Y_MIN)
    y_top = torch.full((n_samples, 1), boundaries.Y_MAX)

    xyt_bottom = torch.cat([x_rand, y_bottom, t_bc], dim=-1).to(device)
    xyt_bottom.requires_grad_(True)
    xyt_top = torch.cat([x_rand, y_top, t_bc], dim=-1).to(device)
    xyt_top.requires_grad_(True)

    u_left = model(xyt_left, device)
    u_left.requires_grad_(True)

    u_right = model(xyt_right, device)
    u_right.requires_grad_(True)

    u_bottom = model(xyt_bottom, device)
    u_bottom.requires_grad_(True)

    u_top = model(xyt_top, device)
    u_top.requires_grad_(True)

    return lambd * u_left**2 + lambd * u_right**2 + lambd * u_bottom**2 + lambd * u_top**2


def hessian_f_norm(model, x, y, t, device):
    # Frobenius norm of the Hessian matrix of the NN --> measures "smoothness" of the output of the NN
    # If added to the loss function, this constrains the NN to produce a smoother output
    xyt = torch.cat([x, y, t], dim=-1).to(device)
    xyt.requires_grad_(True)

    u = model(xyt, device)
    u.requires_grad_(True)

    grad_u = torch.autograd.grad(u, xyt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    hessian_norms = []

    for i in range(3):
        grad_i = grad_u[:, i]
        hess_i = torch.autograd.grad(grad_i, xyt, grad_outputs=torch.ones_like(grad_i), create_graph=True, retain_graph=True)[0]
        hessian_norms.append(hess_i **2)

    hessian_squared = sum(hessian_norms)
    frob_norm_squared = torch.sum(hessian_squared, dim=1)

    return 0.1 * frob_norm_squared

def h1_norm(model, x, y, t, device):
    # H1 Norm of the NN --> measures "size" of the NN
    # Similar idea as Frobenius norm, produces a smoother output
    xyt = torch.cat([x, y, t], dim=-1).to(device)
    xyt.requires_grad_(True)

    u = model(xyt, device)
    u.requires_grad_(True)
    grads = torch.autograd.grad(u, xyt, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    u_sq = u ** 2
    grad_sq = (grads ** 2).sum(dim=1)
    h1 = u_sq + grad_sq

    return 0.1 * h1


def total_loss(model, x, y, t, device):
    # Loss function that is minimized to find the model parameters

    residual = wave_equation_residual(model, x, y, t, device)
    ic_loss = initial_condition_loss(model, x, y, device)
    bc_loss = boundary_condition_loss(model, x, y, t, device)
    sr_loss = h1_norm(model, x, y, t, device) + hessian_f_norm(model, x, y, t, device)

    return torch.mean(residual + ic_loss + bc_loss + sr_loss)
