import numpy as np
from plot import Plot_Sphere
import plotly.graph_objects as go
from dash_app import dash_app
from rossby import random_field, vorticity_field


def plot_basic_sphere():

    n_latitudes = 100
    n_longitudes = 100
    theta = np.linspace(0, np.pi, n_latitudes)  # Latitude angles
    phi = np.linspace(0, 2 * np.pi, n_longitudes)  # Longitude angles
    plot_sphere = Plot_Sphere(
        theta=theta, phi=phi, fields=None, plot_coastlines=True, plot_lonlat_lines=True
    )

    t = np.linspace(0, 0.1, 25)
    values = vorticity_field(theta=theta, phi=phi, t=t)

    cone_plots = plot_sphere.cone_plot3D(values)
    sphere_plot = plot_sphere.plot_sphere()
    figs = []
    for cone_plot in cone_plots:
        fig = plot_sphere.make_fig([cone_plot] + sphere_plot)
        figs.append(fig)
    app = dash_app(figs)
    app.run(debug=True)

    # fields = random_field(theta, phi)

    # plot_sphere.rotating_sphere()
    # figs = plot_sphere.animate_sphere_multiple_figs()

    # fig = plot_sphere.animate_sphere()
    # figs = [fig]
    # traces = plot_sphere(
    #     theta, phi, field=None, plot_coastlines=True, plot_lonlat_lines=True
    # )
    # fig = make_fig(traces)
    # # Show plot
    # app = dash_app(figs)
    # app.run(debug=True)


if __name__ == "__main__":
    plot_basic_sphere()
