import numpy as np
import plotly.graph_objects as go
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs


# Function to get coastline data
def get_coastline():
    reader = shpreader.Reader(
        shpreader.natural_earth(
            resolution="110m", category="physical", name="coastline"
        )
    )
    coastlines = []
    for record in reader.records():
        coords = (
            np.array(record.geometry.coords.xy)
            if record.geometry.geom_type == "LineString"
            else None
        )
        if coords is not None:
            coastlines.append(coords)
    return coastlines


def plot_lon_lat_lines(n_latitudes, n_longitudes, theta, phi):

    # Longitude lines (meridians)
    longitude_lines = []
    for p in np.linspace(0, 2 * np.pi, n_latitudes):
        x_line = np.sin(theta) * np.cos(p)
        y_line = np.sin(theta) * np.sin(p)
        z_line = np.cos(theta)
        longitude_lines.append(
            go.Scatter3d(
                x=x_line,
                y=y_line,
                z=z_line,
                mode="lines",
                line=dict(color="black", width=2),
            )
        )

    # Latitude lines (parallels)
    latitude_lines = []
    for t in np.linspace(-np.pi / 2, np.pi / 2, n_longitudes):
        x_line = np.cos(t) * np.cos(phi)
        y_line = np.cos(t) * np.sin(phi)
        z_line = np.full_like(phi, np.sin(t))
        latitude_lines.append(
            go.Scatter3d(
                x=x_line,
                y=y_line,
                z=z_line,
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )
        )

    return longitude_lines, latitude_lines


def plot_coastline():
    coastline_data = get_coastline()
    coastline = []
    for coords in coastline_data:
        lon, lat = coords
        lon = np.radians(lon)
        lat = np.radians(lat)
        x_coast = np.cos(lat) * np.cos(lon)
        y_coast = np.cos(lat) * np.sin(lon)
        z_coast = np.sin(lat)
        coastline.append(
            go.Scatter3d(
                x=x_coast,
                y=y_coast,
                z=z_coast,
                mode="lines",
                line=dict(color="black", width=2),
            )
        )
    return coastline


def animate_sphere(theta, phi, fields):
    frames = [
        go.Frame(
            data=plot_sphere(
                theta, phi, field=field, plot_coastlines=True, plot_lonlat_lines=True
            ),
            name=str(i),
        )
        for i, field in enumerate(fields)
    ]
    fig = go.Figure(
    data=[go.Surface(z=fields[0])],
        layout=go.Layout(
            title="Oscillating Field",
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                        )
                    ],
                )
            ],
        ),
        frames=frames,
    )

    fig.show()


def get_spehre():
    pass


def plot_sphere(theta, phi, field=None, plot_coastlines=True, plot_lonlat_lines=True):
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    # Convert spherical coordinates to Cartesian
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)

    coastline = []
    if plot_coastlines:
        coastline = plot_coastline()

    longitude_lines = []
    latitude_lines = []
    if plot_lonlat_lines:
        longitude_lines, latitude_lines = plot_lon_lat_lines(
            len(theta), len(phi), theta, phi
        )

    sphere_surface = go.Surface(
        x=x,
        y=y,
        z=z,
        colorscale="Blues",
        opacity=1,
        showscale=False,
        surfacecolor=field,
    )

    traces = [sphere_surface] + longitude_lines + latitude_lines + coastline

    return traces


def make_fig(traces):
    fig = go.Figure(data=traces)

    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        width=1000,
        height=1000,
    )

    return fig


# Create sphere surface
