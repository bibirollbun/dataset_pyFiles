%load_ext autoreload
%autoreload 2


! pip install -r /kaggle/input/clone-herios-data-visualizer/helios-render-kaggle/requirements.txt


%%writefile helios_render.py
"""
Helios Data Visualizer

Installation:
1. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv/Scripts/activate

2. Install dependencies:
   pip install -r requirements.txt

3. Install FFmpeg (required for video generation):
   - Windows: Download from https://ffmpeg.org/download.html and add to PATH
   - macOS: brew install ffmpeg
   - Linux: sudo apt install ffmpeg

Usage:
   python helios_render.py --csv cmi-detect-behavior-with-sensor-data/train.csv --subject SUBJ_032761 --gesture "Wave hello"
   python helios_render.py --csv cmi-detect-behavior-with-sensor-data/train.csv --subject SUBJ_032761 --gesture "Neck - pinch skin" --sequence_index 3

Updates:

v2:
- filter by handedness
- bug fix on loading rotation matrix (scalar-first order)
- align arm mesh to sensor axes
- change projection to "xz" plane
"""

import argparse
import os
import pathlib
import re
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

WINDOW_SIZE = [1280, 720]
VIDEO_SIZE = (16, 9)
DPI = 120
TOF_GRID_SIZE = (8, 8)
NUM_TOF_SENSORS = 5
NUM_THERMOPILE_SENSORS = 5

COLORS = {
    "background": [0.05, 0.05, 0.07],
    "background_top": [0.12, 0.12, 0.16],
    "mesh": "#e0e0e8",
    "text": "#ffffff",
    "text_secondary": "#e0e0e8",
    "text_tertiary": "#c0c0d0",
    "title_bg": "#1a1a2e",
    "figure_bg": "#0a0a12",
    "spine": "#4a4a6a",
}


def align_mesh_to_sensor_axes(mesh):
    rot_fix = R.from_euler("z", -90, degrees=True)
    transform = create_transform_matrix(rot_fix.as_matrix())
    mesh.transform(transform, inplace=True)
    return mesh


def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def create_transform_matrix(rotation_matrix):
    """Create 4x4 transformation matrix from 3x3 rotation matrix."""
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    return transform_matrix


def extract_tof_data(row, sensor_num):
    """Extract ToF sensor data and reshape to 8x8 grid."""
    tof_data = []
    for i in range(64):
        column_name = f"tof_{sensor_num}_v{i}"
        if column_name in row.keys():
            value = row[column_name]
            tof_data.append(np.nan if value == -1 else float(value))
        else:
            tof_data.append(np.nan)
    return np.array(tof_data).reshape(TOF_GRID_SIZE)


def load_data(csv_path, demographic_path, filters=None, use_cache=True):
    """Load and filter data with optional caching."""
    csv_path = pathlib.Path(csv_path)
    cache_dir = pathlib.Path(CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{csv_path.stem}.parquet"

    if use_cache and cache_file.exists():
        print(f"Loading data from cache: {cache_file}")
        start_time = time.time()
        df = pl.read_parquet(cache_file)
        print(f"Loaded from cache in {time.time() - start_time:.2f} seconds")
    else:
        print(f"Loading data from {csv_path}...")
        start_time = time.time()
        df = pl.read_csv(csv_path)
        demographics_df = pl.read_csv(demographic_path)
        df = df.join(demographics_df, on="subject", how="left")
        print(f"Loaded CSV in {time.time() - start_time:.2f} seconds")

        if use_cache:
            print(f"Creating cache file: {cache_file}")
            df.write_parquet(cache_file)

    # Apply filters
    if filters:
        for column, value in filters.items():
            if value:
                df = df.filter(pl.col(column) == value)
                if df.height == 0:
                    raise ValueError(f"No data found for {column} = {value}")
                print(f"Filtered for {column}: {value}")

    unique_sequence_ids = df["sequence_id"].unique(maintain_order=True).to_list()
    if len(unique_sequence_ids) == 0:
        raise ValueError("No valid sequences found in the dataset")

    print(f"Found {len(unique_sequence_ids)} sequences")
    return df, unique_sequence_ids


def load_arm_mesh():
    """Load arm mesh from file."""
    mesh_path = Path(MESH_DIR) / "arm.obj"
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Required arm mesh file not found: {mesh_path}")

    print(f"Loading arm mesh from: {mesh_path}")
    return pv.read(mesh_path)


def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def prepare_sequence_data(df: pl.DataFrame, sequence_id):
    """Prepare sequence data for rendering."""
    sequence_df = df.filter(pl.col("sequence_id") == sequence_id).clone()
    sequence_df = sequence_df.sort("sequence_counter")

    sequence_data = []
    for row in sequence_df.iter_rows(named=True):
        quat = [
            float(row[col]) for col in ["rot_w", "rot_x", "rot_y", "rot_z"]
        ]  # scalar_first order
        acc = [float(row[col]) for col in ["acc_x", "acc_y", "acc_z"]]

        sequence_data.append(
            {
                "row": row,
                "sequence_counter": int(row["sequence_counter"]),
                "row_id": row["row_id"],
                "quat": quat,
                "acc": acc,
                "gesture": row["gesture"],
                "behavior": row["behavior"],
                "orientation": row["orientation"],
                "phase": row["phase"],
                "sequence_type": row["sequence_type"],
            }
        )

    return sequence_data, sequence_df["gesture"][0]


def setup_plotter():
    """Setup PyVista plotter with consistent lighting and camera."""
    plotter = pv.Plotter(notebook=False, off_screen=True, window_size=WINDOW_SIZE)
    plotter.set_background(COLORS["background"], top=COLORS["background_top"])
    plotter.add_axes(interactive=True, line_width=2)  # type: ignore
    plotter.camera_position = "yz"
    plotter.camera.up = [0, 0, 1]  # type: ignore

    # Remove default lights and add custom lighting
    plotter.remove_all_lights()
    light_configs = [
        {"position": (0, 10, 10), "color": [1, 1, 1], "intensity": 0.7},
        {"position": (10, -5, 0), "color": [0.9, 0.9, 1], "intensity": 0.5},
        {"position": (-10, -10, -10), "color": [0.7, 0.7, 0.8], "intensity": 0.3},
    ]

    for config in light_configs:
        light = pv.Light(
            position=config["position"],
            focal_point=(0, 0, 0),
            color=config["color"],
            intensity=config["intensity"],
        )
        plotter.add_light(light)

    return plotter


def create_sensor_subplot(fig, frame_data, sensor_num, subplot_pos):
    """Create a single ToF sensor visualization."""
    ax = plt.subplot(subplot_pos)
    ax.set_facecolor(COLORS["figure_bg"])

    tof_data = extract_tof_data(frame_data["row"], sensor_num)
    ax.imshow(tof_data, cmap=plt.cm.plasma, vmin=0, vmax=254, interpolation="bilinear")  # type: ignore
    ax.set_title(
        f"ToF {sensor_num}",
        fontsize=10,
        color=COLORS["text_secondary"],
        fontweight="medium",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["spine"])
        spine.set_linewidth(0.8)


def create_thermopile_subplot(fig, frame_data, subplot_pos):
    """Create thermopile sensor visualization."""
    ax = plt.subplot(subplot_pos)
    ax.set_facecolor(COLORS["figure_bg"])

    # Extract thermopile data
    therm_data = []
    for j in range(1, NUM_THERMOPILE_SENSORS + 1):
        col_name = f"thm_{j}"
        if col_name in frame_data["row"].keys():
            therm_data.append(float(frame_data["row"][col_name]))
        else:
            therm_data.append(np.nan)

    y_pos = np.arange(NUM_THERMOPILE_SENSORS)

    # Normalize temperatures for color mapping (20-40Â°C range)
    temp_min, temp_max = 20, 40
    normalized_temps = [
        (temp - temp_min) / (temp_max - temp_min) if not np.isnan(temp) else 0.5
        for temp in therm_data
    ]

    # colors based on temperature values
    colors = plt.cm.magma([max(0, min(1, norm_temp)) for norm_temp in normalized_temps])  # type: ignore

    ax.barh(y_pos, therm_data, color=colors, height=0.7, edgecolor="none")

    ax.set_xlim(20, 40)  # Set min and max temperature range in Celsius

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{j + 1}" for j in range(NUM_THERMOPILE_SENSORS)],
        fontsize=10,
        color=COLORS["text_secondary"],
    )
    ax.set_title(
        "Thermopile", fontsize=10, color=COLORS["text_secondary"], fontweight="medium"
    )
    ax.set_xlabel("Â°C", fontsize=10, color=COLORS["text_tertiary"])
    ax.tick_params(
        axis="both", which="major", labelsize=9, colors=COLORS["text_tertiary"]
    )
    ax.grid(axis="x", linestyle="--", alpha=0.15, color=COLORS["text_tertiary"])

    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["spine"])
        spine.set_linewidth(0.8)


def create_frame_info_text(frame_data):
    """Generate frame information text."""
    return (
        f"Subject: {frame_data['row']['subject']} | "
        f"Gesture: {frame_data['gesture']} | "
        f"Orientation: {frame_data['orientation']} | "
        f"Phase: {frame_data['phase']} | "
        f"Counter: {frame_data['sequence_counter']}"
    )


def render_frame_with_sensors(
    plotter,
    mesh,
    original_points,
    frame_data,
    frame_index,
    temp_dir,
    acc_scale=0.02,
    linear_acc_scale=0.05,
    length_eps=1e-6,
):
    """Render a single frame with sensor data."""
    plotter.clear_actors()

    # rotate arm mesh
    mesh.points = original_points.copy()
    rot = R.from_quat(frame_data["quat"], scalar_first=True)  # rot=[w, x, y, z]
    R_mat = rot.as_matrix()
    transform = create_transform_matrix(R_mat)
    mesh.transform(transform, inplace=True)

    # add arm mesh to plotter
    plotter.add_mesh(
        mesh,
        color=COLORS["mesh"],
        specular=0.3,
        specular_power=5,
        ambient=0.5,
        diffuse=0.6,
        smooth_shading=False,
    )

    # plot acc vector in the world coordinate
    raw_acc = np.array(frame_data["acc"])  # [acc_x, acc_y, acc_z]
    world_acc = rot.apply(raw_acc)
    gravity = np.array([0.0, 0.0, 9.81])
    linear_acc = world_acc - gravity

    # world acc vector
    arrow_length = np.linalg.norm(world_acc) * acc_scale
    if arrow_length > length_eps:
        arrow_dir = world_acc / arrow_length
        arrow = pv.Arrow(start=(0, 0, 0), direction=arrow_dir, scale=arrow_length)  # type: ignore
        plotter.add_mesh(arrow, color="red", name=f"acc_arrow_{frame_index}")

    # linear acc vector
    arrow_length = np.linalg.norm(linear_acc) * linear_acc_scale
    if arrow_length > length_eps:
        arrow_dir = linear_acc / arrow_length
        arrow = pv.Arrow(start=(0, 0, 0), direction=arrow_dir, scale=arrow_length)  # type: ignore
        plotter.add_mesh(arrow, color="green", name=f"linear_acc_{frame_index}")

    # take a screenshot of the 3D view
    pv_img = plotter.screenshot(return_img=True)

    # create composite figure
    fig = plt.figure(figsize=VIDEO_SIZE, dpi=DPI, facecolor=COLORS["figure_bg"])
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1], figure=fig)  # type: ignore

    ax_3d = plt.subplot(gs[0])
    ax_3d.imshow(pv_img)
    ax_3d.set_xticks([])
    ax_3d.set_yticks([])
    ax_3d.axis("off")
    ax_3d.set_title(
        create_frame_info_text(frame_data),
        fontsize=11,
        color=COLORS["text"],
        backgroundcolor=COLORS["title_bg"],
        pad=10,
        fontweight="medium",
    )

    # sensor subplots
    gs_sensors = plt.GridSpec(1, 6, wspace=0.3, figure=fig)  # type: ignore
    gs_sensors.update(top=0.45, bottom=0.05, left=0.05, right=0.95)

    for j in range(NUM_TOF_SENSORS):
        create_sensor_subplot(fig, frame_data, j + 1, gs_sensors[0, j])

    create_thermopile_subplot(fig, frame_data, gs_sensors[0, 5])

    # save frame
    frame_file = os.path.join(temp_dir, f"frame_{frame_index:05d}.png")
    plt.savefig(
        frame_file,
        facecolor=COLORS["figure_bg"],
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=DPI,
    )
    plt.close(fig)
    return frame_file


def create_video_from_frames(frame_files, output_file, framerate, temp_dir):
    """Combine frames into video using FFmpeg."""
    print(f"Combining frames into video: {output_file}")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(framerate),
        "-i",
        os.path.join(temp_dir, "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1:1",
        output_file,
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video successfully created: {output_file}")

        # Cleanup
        for file in frame_files:
            os.remove(file)
        os.rmdir(temp_dir)

    except subprocess.SubprocessError as e:
        print(f"Error creating video with FFmpeg: {e}")
        print(f"Individual frames are saved in {temp_dir}")


def render_animation(sequence_data, output_file, framerate=10):
    """Main rendering function with sensor visualizations."""
    if not check_ffmpeg():
        raise RuntimeError(
            "FFmpeg is required but not found in system PATH. Please install FFmpeg."
        )

    mesh = load_arm_mesh()
    mesh = align_mesh_to_sensor_axes(mesh)  # Align mesh to sensor axes
    original_points = mesh.points.copy()
    temp_dir = tempfile.mkdtemp()
    print(f"Creating composite frames in: {temp_dir}")

    plotter = setup_plotter()
    frame_files = []

    print(f"Rendering {len(sequence_data)} frames...")
    for i, frame_data in enumerate(tqdm(sequence_data, desc="Creating frames")):
        frame_file = render_frame_with_sensors(
            plotter, mesh, original_points, frame_data, i, temp_dir
        )
        frame_files.append(frame_file)

    plotter.close()
    create_video_from_frames(frame_files, output_file, framerate, temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Render 3D motion data animation from sensor data"
    )
    parser.add_argument("--subject", help="Subject ID to filter (e.g., SUBJ_059520)")
    parser.add_argument("--gesture", help="Gesture description to filter")
    parser.add_argument("--behavior", help="Behavior description to filter")
    parser.add_argument("--phase", help="Phase to filter")
    parser.add_argument("--orientation", help="Orientation to filter")
    parser.add_argument(
        "--handedness",
        type=int,
        default=1,
        help="Handedness to filter (0=left, 1=right)",
    )
    parser.add_argument(
        "--framerate", type=int, default=10, help="Frame rate for output video"
    )
    parser.add_argument(
        "--sequence_index",
        type=int,
        default=0,
        help="Index of the sequence to render (if multiple sequences match)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable parquet caching"
    )

    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not check_ffmpeg():
        raise RuntimeError(
            "FFmpeg is required but not found in system PATH. Please install it:\n"
            "  - Windows: Download from https://ffmpeg.org/download.html and add to PATH\n"
            "  - macOS: Run 'brew install ffmpeg' (requires Homebrew)\n"
            "  - Linux: Run 'sudo apt install ffmpeg' or equivalent for your distro"
        )

    # Prepare filters
    filters = {
        "subject": args.subject,
        "gesture": args.gesture,
        "behavior": args.behavior,
        "phase": args.phase,
        "orientation": args.orientation,
        "handedness": args.handedness,
    }
    print("ğŸ‘€ Applying filters:")
    for key, value in filters.items():
        if value is not None:
            print(f"  {key}: {value}")

    df, unique_sequence_ids = load_data(
        CSV_PATH, DEMOGRAPHIC_PATH, filters, use_cache=not args.no_cache
    )

    # Select sequence
    if args.sequence_index >= len(unique_sequence_ids):
        print(
            f"Warning: Selected index {args.sequence_index} out of range. Using first sequence."
        )
        selected_sequence_id = unique_sequence_ids[0]
    else:
        selected_sequence_id = unique_sequence_ids[args.sequence_index]

    sequence_data, gesture_name = prepare_sequence_data(df, selected_sequence_id)

    # output filename
    subject = (
        args.subject
        or df.filter(pl.col("sequence_id") == selected_sequence_id)["subject"][0]
    )
    safe_gesture_name = gesture_name.replace(" - ", "_").replace(" ", "_")
    safe_gesture_name = sanitize_filename(safe_gesture_name)
    output_file = os.path.join(
        OUTPUT_DIR, f"{subject}_{safe_gesture_name}_{selected_sequence_id}.mp4"
    )

    render_animation(sequence_data, output_file, args.framerate)

    print("Animation complete!")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    import os
    import pyvista as pv
    from pathlib import Path

    
    pv.start_xvfb()
    
    CSV_PATH = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
    DEMOGRAPHIC_PATH = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
    MESH_DIR = Path("/kaggle/input/clone-herios-data-visualizer/helios-render-kaggle")
    OUTPUT_DIR = Path("movie")
    CACHE_DIR = Path("/tmp/cache")
    main()


NON_TARGET_GESTURES = [
    "Drink from bottle/cup",
    "Feel around in tray and pull out an object",
    "Glasses on/off",
    "Pinch knee/leg skin",
    "Pull air toward your face",
    "Scratch knee/leg skin",
    "Text on phone",
    "Wave hello",
    "Write name in air",
    "Write name on leg",
]
TARGET_GESTURES = [
    "Above ear - pull hair",
    "Cheek - pinch skin",
    "Eyebrow - pull hair",
    "Eyelash - pull hair",
    "Forehead - pull hairline",
    "Forehead - scratch",
    "Neck - pinch skin",
    "Neck - scratch",
]
GESTURES = NON_TARGET_GESTURES + TARGET_GESTURES
ORIENTATIONS = [
    "Lie on Back",
    "Lie on Side - Non Dominant",
    "Seated Lean Non Dom - FACE DOWN",
    "Seated Straight",
]


import subprocess

for orientation in ORIENTATIONS:
    for gesture in GESTURES:
        try:
            command = [
                "python",
                "helios_render.py",
                "--gesture", gesture,
                "--orientation", orientation,
            ]
            subprocess.run(command, check=True)
        except:
            print("âš ï¸� Failed to generate movie. Skipping...")


! rm *.py


!ls -l

