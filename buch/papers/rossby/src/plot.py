import numpy as np
import plotly.graph_objects as go
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import time
import os
import plotly.io as pio


class Plot_Sphere:
    def __init__(
        self, theta, phi, fields=None, plot_coastlines=True, plot_lonlat_lines=True
    ):
        self.theta = theta
        self.phi = phi
        self.fields = fields
        self.plot_coastlines = plot_coastlines
        self.plot_lonlat_lines = plot_lonlat_lines
        self.n_longitudes = len(phi)
        self.n_latitudes = len(theta)

    # Function to get coastline data
    def get_coastline(self):
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

    def plot_lon_lat_lines(self):

        # Longitude lines (meridians)
        longitude_lines = []
        for p in np.linspace(0, 2 * np.pi, self.n_latitudes):
            x_line = np.sin(self.theta) * np.cos(p)
            y_line = np.sin(self.theta) * np.sin(p)
            z_line = np.cos(self.theta)
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
        for t in np.linspace(-np.pi / 2, np.pi / 2, self.n_longitudes):
            x_line = np.cos(t) * np.cos(self.phi)
            y_line = np.cos(t) * np.sin(self.phi)
            z_line = np.full_like(self.phi, np.sin(t))
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

    def plot_coastline(self):
        coastline_data = self.get_coastline()
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

    def animate_sphere(self):
        t = time.time()
        static_traces = self.plot_sphere_static()
        frames = [
            go.Frame(
                data=(static_traces + [self.plot_sphere_surface(field)]),
                name=str(i),
            )
            for i, field in enumerate(self.fields)
        ]
        print("Time to create frames:", time.time() - t)
        # fig = go.Figure(frames[0])
        fig = go.Figure(
            data=static_traces + [self.plot_sphere_surface(self.fields[0])],
            layout=go.Layout(
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
                                        "frame": {"duration": 10, "redraw": True},
                                    },
                                ],
                            )
                        ],
                    )
                ],
            ),
            frames=frames,
        )
        fig.update_layout(height=850, showlegend=False)
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="rgba(0,0,0,0)",  # transparent background
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=10, b=0, t=10),
        )
        return fig

    def plot_sphere(self):
        traces = self.plot_sphere_static() + [self.plot_sphere_surface(None)]
        return traces

    def animate_sphere_multiple_figs(self):
        figs = []
        for field in self.fields:
            figs.append(
                self.make_fig(
                    [self.plot_sphere_surface(field)] + self.plot_sphere_static()
                )
            )
        return figs

    def plot_sphere_surface(self, field):
        theta_grid, phi_grid = np.meshgrid(self.theta, self.phi)
        x = np.sin(theta_grid) * np.cos(phi_grid)
        y = np.sin(theta_grid) * np.sin(phi_grid)
        z = np.cos(theta_grid)

        return go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=field,
            colorscale=[[0, "white"], [1, "white"]],
            opacity=1,
            showscale=False,
        )

    def cone_plot3D(self, values):
        traces = []
        for value in values:
            x = value[0]
            y = value[1]
            z = value[2]
            u = value[3]
            v = value[4]
            w = value[5]
            trace = go.Cone(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                u=u.flatten(),
                v=v.flatten(),
                w=w.flatten(),
                colorscale="Viridis",
                sizemode="absolute",
                sizeref=2,
            )
            traces.append(trace)

        return traces

    def plot_sphere_static(self):

        coastline = []
        if self.plot_coastlines:
            coastline = self.plot_coastline()

        longitude_lines = []
        latitude_lines = []
        if self.plot_lonlat_lines:
            longitude_lines, latitude_lines = self.plot_lon_lat_lines()

        traces = longitude_lines + latitude_lines + coastline

        return traces

    def make_fig(self, traces):
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
            uirevision="keep",  # preserves 3D view
        )

        return fig

    def export_frames(self, fig, folder="frames"):
        os.makedirs(folder, exist_ok=True)
        for i, frame in enumerate(fig.frames):
            fig.update(data=frame.data)
            pio.write_image(fig, f"{folder}/frame_{i:03d}.png", width=800, height=800)
        print(f"Exported {len(fig.frames)} frames to '{folder}/'")

    def render_video(self, folder="frames", output="out.mp4", speed=2):
        framerate = 20
        fast_filter = 1 / speed
        os.system(
            f"""
        ffmpeg -y -framerate {framerate} -i {folder}/frame_%03d.png -c:v libx264 -preset veryslow -crf 18 -pix_fmt yuv420p temp_out.mp4
        ffmpeg -y -i temp_out.mp4 -filter:v "setpts={fast_filter}*PTS" {output}
        rm temp_out.mp4
        """
        )
        print(f"Saved sped-up video as '{output}'")
