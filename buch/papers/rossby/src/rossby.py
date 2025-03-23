import numpy as np


def rossby():
    pass


def random_field(theta, phi):
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    t = np.linspace(0, 2 * np.pi, 25)
    fields = []
    for t_i in t:
        fields.append(np.sin(5 * theta_grid - t_i) ** 2 + 0.2 * np.cos(7 * phi_grid + t_i))
    return fields
