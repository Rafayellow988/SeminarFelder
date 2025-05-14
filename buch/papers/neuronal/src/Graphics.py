import numpy as np
import torch
import boundaries
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from Residuals import total_loss, analytical_solution

matplotlib.use('TkAgg')

# Plots train & test error over training epochs
def error_plot(train_error, test_error):
    epochs = range(1, len(train_error) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_error, label='Training Error')
    plt.plot(epochs, test_error, label='Test Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Errors over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Creates an animated contour plot that updates through time
def animate_neural_network(model, device):
    error = []
    grid_size = 1000
    t_values = np.linspace(boundaries.T_MIN, boundaries.T_MAX, 51)
    x_vals = np.linspace(boundaries.X_MIN, boundaries.X_MAX, grid_size)
    y_vals = np.linspace(boundaries.Y_MIN, boundaries.Y_MAX, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)

    u_pred_array = []

    for t in t_values:
        xyt = torch.FloatTensor(
            np.column_stack([X.ravel(), Y.ravel(), np.full_like(X.ravel(), t)])
        ).to(device)
        u_pred_array.append(model(xyt).cpu().detach().numpy().reshape(grid_size, grid_size))

        x = torch.FloatTensor(np.column_stack([X.ravel()])).to(device)
        y = torch.FloatTensor(np.column_stack([Y.ravel()])).to(device)
        t = torch.FloatTensor(np.column_stack([np.full_like(X.ravel(), t)])).to(device)
        error.append(total_loss(model, x, y, t, device).item())

    print(f"Average loss on animated data: {np.mean(error)}")

    fig, ax = plt.subplots(figsize=(20, 20))

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
    ani.save("wave_animation.gif", writer='pillow', fps=5)
    plt.show()

def animate_comparison(model, device):
    grid_size = 500  # Use 1000 if memory allows

    t_values = np.linspace(boundaries.T_MIN, boundaries.T_MAX, 51)
    x_vals = np.linspace(boundaries.X_MIN, boundaries.X_MAX, grid_size)
    y_vals = np.linspace(boundaries.Y_MIN, boundaries.Y_MAX, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)

    u_pred_array = []
    u_true_array = []
    error_accumulator = []

    for t in t_values:
        # Predict with neural net
        xyt = torch.FloatTensor(
            np.column_stack([X.ravel(), Y.ravel(), np.full_like(X.ravel(), t)])
        ).to(device)
        u_pred = model(xyt).cpu().detach().numpy().reshape(grid_size, grid_size)
        u_pred_array.append(u_pred)

        # Analytical solution
        u_true = analytical_solution(X, Y, t)
        u_true_array.append(u_true)

        # MSE at this time step
        error = np.mean((u_pred - u_true) ** 2)
        error_accumulator.append(error)

    avg_mse = np.mean(error_accumulator)
    print(f"Average MSE over all frames: {avg_mse}")

    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    cbar1 = cbar2 = None

    def update(frame):
        nonlocal cbar1, cbar2
        for ax in axes:
            ax.clear()

        t = t_values[frame]
        U_pred = u_pred_array[frame]
        U_true = u_true_array[frame]

        cont1 = axes[0].contourf(X, Y, U_pred, levels=100, cmap='coolwarm')
        axes[0].set_title(f"Neural Network Prediction at t = {t:.1f}")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")

        cont2 = axes[1].contourf(X, Y, U_true, levels=100, cmap='coolwarm')
        axes[1].set_title(f"Analytical Solution at t = {t:.1f}")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")

        # Recreate colorbars each frame for consistency
        if cbar1 is None:
            cbar1 = fig.colorbar(cont1, ax=axes[0], label='Wave Height')
        if cbar2 is None:
            cbar2 = fig.colorbar(cont2, ax=axes[1], label='Wave Height')

    ani = animation.FuncAnimation(fig, update, frames=len(t_values), interval=100, repeat=True)
    ani.save("comparison_wave_animation.gif", writer='pillow', fps=5)
    plt.show()
