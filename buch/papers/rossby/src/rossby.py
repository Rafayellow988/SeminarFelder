import numpy as np
import plotly.graph_objects as go

def rossby():
    pass


def random_field(theta, phi):
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    t = np.linspace(0, 2 * np.pi, 25)
    fields = []
    for t_i in t:
        fields.append(np.sin(5 * theta_grid - t_i) ** 2 + 0.2 * np.cos(7 * phi_grid + t_i))
    return fields

def vorticity_field(theta, phi, t=None):
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    Omega = 1

    values = []
    for t_i in t:
        phi_grid = phi_grid + Omega * t_i  # simple rotation in time
        x = np.sin(theta_grid)*np.cos(phi_grid)
        y = np.sin(theta_grid)*np.sin(phi_grid)
        z = np.cos(theta_grid)
        u = -Omega * y     # vx
        v = Omega * x     # vy
        w =  np.zeros_like(z)
        print(x[23,20])
        values.append((x, y, z, u, v, w))

    return values

    # # Add the cone plot to the figure
    # fig.add_trace(go.Cone(
    #     x=x.flatten(),
    #     y=y.flatten(),
    #     z=z.flatten(),
    #     u=u.flatten(),
    #     v=v.flatten(),
    #     w=w.flatten(),
    #     colorscale='Viridis',
    #     sizemode='absolute',
    #     sizeref=0.5
    # ))

    # # Update layout for better visualization
    # fig.update_layout(
    #     scene=dict(
    #         xaxis_title='X',
    #         yaxis_title='Y',
    #         zaxis_title='Z',
    #         aspectratio=dict(x=1, y=1, z=1)
    #     ),
    #     title="Vorticity Field Cone Plot"
    # )

    # # Show the plot
    # fig.show()