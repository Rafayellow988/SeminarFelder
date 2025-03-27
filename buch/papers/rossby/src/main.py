import numpy as np
# from plot import animate_sphere, export_frames, render_video
from plot_matplotlib import animate_sphere
import plotly.graph_objects as go
from dash_app import dash_app
from rossby import random_field

def plot_basic_sphere():

    n_latitudes = 100
    n_longitudes = 100
    theta = np.linspace(0, np.pi, n_latitudes)  # Latitude angles
    phi = np.linspace(0, 2 * np.pi, n_longitudes)  # Longitude angles

    field = random_field(theta, phi)
    fig = animate_sphere(theta, phi, field)

    # traces = plot_sphere(
    #     theta, phi, field=None, plot_coastlines=True, plot_lonlat_lines=True
    # )
    # fig = make_fig(traces)
    # # Show plot
    # app = dash_app(fig)
    # app.run(debug=True)

if __name__ == "__main__":
    plot_basic_sphere()
