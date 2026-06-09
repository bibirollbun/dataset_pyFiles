import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, Image, display


from IPython.display import HTML, display
import base64

def show_gif_inline(path, width=400):
    """Display a GIF inline using base64 embedding."""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        html = f'<img src="data:image/gif;base64,{data}" width="{width}">'
        display(HTML(html))
    except FileNotFoundError:
        print(f"File not found: {path}")



DATASET_CSV = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
df = pd.read_csv(DATASET_CSV)

CHOSEN_SEQ = "SEQ_031678"
filtered_data = df[df['sequence_id'] == CHOSEN_SEQ]
print(f"chosen sequence: {CHOSEN_SEQ}")


class SensorGrid:
    def __init__(self, sensor_id, dataframe):
        self.sensor_id = sensor_id
        self.frames = [
            np.array([row[f"tof_{sensor_id}_v{i}"] for i in range(64)]).reshape(8, 8)
            for _, row in dataframe.iterrows()
        ]
        self.vmin = np.min([frame.min() for frame in self.frames])
        self.vmax = np.max([frame.max() for frame in self.frames])

    def get_matrix_at(self, frame_idx):
        return self.frames[frame_idx]

    def get_min_max(self):
        return self.vmin, self.vmax

    def rotate(self, direction):
        if direction == 'left':
            k = 1
        elif direction == 'right':
            k = 3
        elif direction == '180':
            k = 2
        else:
            raise ValueError("Direction must be 'left', 'right', or '180'")
        self.frames = [np.rot90(mat, k=k) for mat in self.frames]

    def flip_horizontal(self):
        
        self.frames = [np.fliplr(mat) for mat in self.frames]

    def flip_vertical(self):
        
        self.frames = [np.flipud(mat) for mat in self.frames]

class SimpleAnimation:
    def __init__(self, sensor_grid, save_path="sensor_simple.gif", max_frames=None):
        self.grid = sensor_grid
        self.save_path = save_path
        self.frame_count = len(sensor_grid.frames) if max_frames is None else min(max_frames, len(sensor_grid.frames))

    def animate(self, fps=10):
        fig, ax = plt.subplots(figsize=(4, 4))
        vmin, vmax = self.grid.get_min_max()
        im = ax.imshow(np.zeros((8, 8)), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')

        def update(frame_idx):
            mat = self.grid.get_matrix_at(frame_idx)
            im.set_array(mat)
            ax.set_title(f"TOF{self.grid.sensor_id} â€“ frame {frame_idx}")
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=self.frame_count, blit=False)
        ani.save(self.save_path, writer='pillow', fps=fps)
        plt.close()



grid = SensorGrid(1, filtered_data)

simple_anim = SimpleAnimation(grid, save_path="/kaggle/working/sensor_simple.gif")
simple_anim.animate(fps=10)

show_gif_inline("/kaggle/working/sensor_simple.gif")


class SensorLayout:
    def __init__(self, layout_spec):
        self.layout_spec = layout_spec
        self.rows = len(layout_spec)
        self.cols = len(layout_spec[0])

    def get_position(self, sensor_id):
        for r, row in enumerate(self.layout_spec):
            for c, val in enumerate(row):
                if val == sensor_id:
                    return r, c
        return None

    def get_sensor_ids(self):
        return [val for row in self.layout_spec for val in row if val is not None]


class SensorPlot:
    def __init__(self, sensor_grid):
        self.sensor_id = sensor_grid.sensor_id
        self.grid = sensor_grid
        self.ax = None
        self.im = None

    def init_plot(self, ax):
        self.ax = ax
        vmin, vmax = self.grid.get_min_max()
        self.im = ax.imshow(np.zeros((8, 8)), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')

    def update(self, frame_idx):
        matrix = self.grid.get_matrix_at(frame_idx)
        self.im.set_array(matrix)


class SensorAnimator:
    def __init__(self, sensor_grids, layout, filename, max_frames=None):
        self.layout = layout
        self.sensor_grids = {grid.sensor_id: grid for grid in sensor_grids}
        self.sensor_plots = {}
        self.save_path = f"/kaggle/working/{filename}"
        self.fig = None

        # Determine frame count from any one grid
        example_grid = next(iter(self.sensor_grids.values()))
        self.frame_count = len(example_grid.frames) if max_frames is None else min(max_frames, len(example_grid.frames))

    def _create_figure_and_axes(self):
        self.fig = plt.figure(figsize=((self.layout.cols * 3)/2, (self.layout.rows * 3)/2))
        gs = self.fig.add_gridspec(self.layout.rows, self.layout.cols)

        for sid in self.layout.get_sensor_ids():
            r, c = self.layout.get_position(sid)
            ax = self.fig.add_subplot(gs[r, c])
            plot = SensorPlot(self.sensor_grids[sid])
            plot.init_plot(ax)
            self.sensor_plots[sid] = plot

        self.fig.suptitle("TOF[1-5] â€“ sequence_counter = 0")

    def _update(self, frame_idx):
        for sid, plot in self.sensor_plots.items():
            plot.update(frame_idx)
        self.fig.suptitle(f"TOF[1-5] â€“ frame {frame_idx}")
        return [plot.im for plot in self.sensor_plots.values()]

    def animate(self, fps=10):
        self._create_figure_and_axes()
        ani = animation.FuncAnimation(
            self.fig,
            self._update,
            frames=self.frame_count,
            blit=False,
            repeat=False
        )
        ani.save(self.save_path, writer='pillow', fps=fps)
        plt.close()



layout_spec = [
    [None, 2,    None],
    [None, 1,    None],
    [5,    None, 3],
    [None, 4,    None]
]
layout = SensorLayout(layout_spec)

grids = [
    SensorGrid(1, filtered_data),
    SensorGrid(2, filtered_data),
    SensorGrid(3, filtered_data),
    SensorGrid(4, filtered_data),
    SensorGrid(5, filtered_data)
]

working_filename = "as_provided.gif"

animator = SensorAnimator(sensor_grids=grids, layout=layout, filename=working_filename)
animator.animate(fps=10)

show_gif_inline(f"/kaggle/working/{working_filename}")


grids = [
    SensorGrid(1, filtered_data),
    SensorGrid(2, filtered_data),
    SensorGrid(3, filtered_data),
    SensorGrid(4, filtered_data),
    SensorGrid(5, filtered_data)
]

for grid in grids:
    grid.flip_horizontal()

grids[2].rotate('left')
grids[4].rotate('left')

working_filename = "flipped_horizontal_rotated.gif"

animator = SensorAnimator(sensor_grids=grids, layout=layout, filename=working_filename)
animator.animate(fps=10)

show_gif_inline(f"/kaggle/working/{working_filename}")


layout_spec = [
    [None, 2,    None],
    [None, 1,    None],
    [3,    None, 5],
    [None, 4,    None]
]
layout = SensorLayout(layout_spec)

grids = [
    SensorGrid(1, filtered_data),
    SensorGrid(2, filtered_data),
    SensorGrid(3, filtered_data),
    SensorGrid(4, filtered_data),
    SensorGrid(5, filtered_data)
]

for grid in grids:
    grid.flip_horizontal()

grids[2].rotate('left')
grids[4].rotate('left')

for grid in grids:
    grid.flip_vertical()

working_filename = "flipped_horizontal_rotated_vertical.gif"

animator = SensorAnimator(sensor_grids=grids, layout=layout, filename=working_filename)
animator.animate(fps=10)

show_gif_inline(f"/kaggle/working/{working_filename}")

