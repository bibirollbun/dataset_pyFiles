import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.notebook import tqdm

pd.set_option('display.max_columns', 400)


DATA_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
assert os.path.exists(DATA_DIR)

train_df = pd.read_csv(DATA_DIR / "train.csv")
train_demo_df = pd.read_csv(DATA_DIR / "train_demographics.csv")

print("ğŸ‘‰ train.csv shape:", train_df.shape)
print("ğŸ‘‰ train_demographics.csv shape:", train_demo_df.shape)



train_df.sample(3)


train_df.info()
train_df.isnull().sum().sort_values(ascending=False).head(20)


train_df.describe(include='all')


train_df.describe(include='all').T


missing_cols = train_df.columns[train_df.isnull().any()]
missing_summary = train_df[missing_cols].isnull().mean().sort_values(ascending=False)

plt.figure(figsize=(12,5))
missing_summary.plot(kind='bar', color='coral')
plt.title("Columns with Missing Data (%)")
plt.ylabel("Fraction Missing")
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()



gesture_counts = train_df.groupby(['gesture'])['sequence_id'].nunique().sort_values(ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(y=gesture_counts.index, x=gesture_counts.values, palette="viridis")
plt.title("Gesture Distribution (by sequence count)")
plt.xlabel("Number of sequences")
plt.ylabel("Gesture label")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()



train_demo_df.describe(include='all').T


train_merged = train_df.merge(train_demo_df, on="subject", how="left")
print("Merged dataset shape:", train_merged.shape)



train_merged.info()
train_merged.describe(include='all').T


imu_features = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']

imu_stats = train_merged.groupby("sequence_id")[imu_features].agg(
    ['mean', 'std', 'min', 'max', 'skew']
)

imu_stats.columns = ['_'.join(col).strip() for col in imu_stats.columns.values]
imu_stats.reset_index(inplace=True)

imu_stats.sample(3)



tof_cols = [col for col in train_df.columns if col.startswith("tof_")]
thm_cols = [col for col in train_df.columns if col.startswith("thm_")]

print("ğŸ”¹ ToF Columns:", len(tof_cols), "ğŸ”¹ Thermopile Columns:", len(thm_cols))

tof_nulls = train_df[tof_cols].isnull().mean().mean()
thm_nulls = train_df[thm_cols].isnull().mean().mean()

print(f"ğŸ’€ Avg Missing (ToF): {tof_nulls:.2%}")
print(f"ğŸ’€ Avg Missing (Thermopile): {thm_nulls:.2%}")



seq_lengths = train_df.groupby("sequence_id")["sequence_counter"].max() + 1

plt.hist(seq_lengths, bins=30, color="mediumseagreen", edgecolor="black")
plt.title("â�³ Sequence Lengths")
plt.xlabel("Number of timestamps")
plt.ylabel("Count")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()



import random
from scipy.spatial.transform import Rotation as R
import matplotlib.patches as mpatches
import cv2  # for image resizing

# Pick a random sequence
selected_sequence = random.choice(train_df['sequence_id'].unique())
print(f"ğŸ�¯ Selected sequence: {selected_sequence}")

seq_df = train_df[train_df['sequence_id'] == selected_sequence].copy()

# Reset index for plotting
seq_df.reset_index(drop=True, inplace=True)



# Thermopile columns
thm_cols = [col for col in seq_df.columns if col.startswith("thm_")]

# Plot them over time
plt.figure(figsize=(14, 5))
for col in thm_cols:
    plt.plot(seq_df["sequence_counter"], seq_df[col], label=col)

plt.title(f"Thermopile Sensor Readings â€“ Sequence {selected_sequence}")
plt.xlabel("Timestep")
plt.ylabel("Temperature")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# Plot acceleration
axes[0].plot(seq_df['acc_x'], label='acc_x')
axes[0].plot(seq_df['acc_y'], label='acc_y')
axes[0].plot(seq_df['acc_z'], label='acc_z')
axes[0].set_ylabel("Acceleration")
axes[0].legend(loc='upper right')
axes[0].set_title("Acceleration over Time")

# Plot rotation
axes[1].plot(seq_df['rot_x'], label='rot_x')
axes[1].plot(seq_df['rot_y'], label='rot_y')
axes[1].plot(seq_df['rot_z'], label='rot_z')
axes[1].plot(seq_df['rot_w'], label='rot_w')
axes[1].set_ylabel("Rotation")
axes[1].legend(loc='upper right')
axes[1].set_title("Rotation over Time")

# Add shaded backgrounds for behavior
for ax in axes:
    for i in range(len(seq_df)):
        color = "orange" if seq_df.loc[i, "behavior"] == 1 else "lightgray"
        ax.axvspan(i, i+1, color=color, alpha=0.05)

plt.xlabel("Timestep")
plt.tight_layout()
plt.show()



# Ensure quaternion order is (x, y, z, w)
quaternions = seq_df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].to_numpy()
euler_angles = R.from_quat(quaternions).as_euler('xyz', degrees=True)

seq_df['pitch'], seq_df['roll'], seq_df['yaw'] = euler_angles.T

# Plot
plt.figure(figsize=(16,6))
plt.plot(seq_df['pitch'], label='Pitch')
plt.plot(seq_df['roll'], label='Roll')
plt.plot(seq_df['yaw'], label='Yaw')
plt.title("Euler Angles (Degrees)")
plt.xlabel("Timestep")
plt.ylabel("Angle (Â°)")
plt.legend(loc='upper right')

# Highlight behavior
for i in range(len(seq_df)):
    color = "orange" if seq_df.loc[i, "behavior"] == 1 else "lightgray"
    plt.axvspan(i, i+1, color=color, alpha=0.05)

plt.grid(True, linestyle='--', alpha=0.3)
plt.show()



# Group ToF columns by sensor index (v1, v2, ..., v5)
tof_groups = {
    f"tof_v{i}": [col for col in seq_df.columns if f"tof_{i}_" in col] 
    for i in range(1,6)
}

# Pick a middle timestep
mid_index = len(seq_df) // 2

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, (sensor_name, cols) in enumerate(tof_groups.items()):
    # Get 64 ToF values for this sensor
    tof_values = seq_df[cols].iloc[mid_index].values.reshape(8, 8)
    
    # Resize to 128x128 for better visuals
    tof_resized = cv2.resize(tof_values, (128, 128), interpolation=cv2.INTER_NEAREST)
    
    ax = axes[i]
    im = ax.imshow(tof_resized, cmap='hot')
    ax.set_title(f"Sensor {sensor_name.split('_')[-1]}")
    ax.axis('off')

fig.suptitle(f"ToF Heatmaps â€“ Sequence {selected_sequence}, Timestep {mid_index}", fontsize=16)
plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Distance")
plt.tight_layout()
plt.show()



## In Google Colab:
# !sudo apt-get update
# !sudo apt-get install -y ffmpeg


## In Kaggle Notebooks:
# !apt-get update
# !apt-get install -y ffmpeg

## In Windows 
# import matplotlib as mpl
# mpl.rcParams['animation.ffmpeg_path'] = r'path\to\ffmpeg\bin\ffmpeg.exe'




!apt-get update
!apt-get install -y ffmpeg


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from scipy.spatial.transform import Rotation as R
from IPython.display import HTML
from matplotlib.animation import FFMpegWriter

# Pick a random sequence
selected_sequence = random.choice(train_df['sequence_id'].unique())
print(f"ğŸ�¯ Selected sequence: {selected_sequence}")

sequence_data = train_df[train_df['sequence_id'] == selected_sequence].reset_index(drop=True)
n_frames = len(sequence_data)

# --- Detect phase and behavior changes ---
combined_changes = (
    sequence_data["phase"].ne(sequence_data["phase"].shift()) |
    sequence_data["behavior"].ne(sequence_data["behavior"].shift())
)
change_frames = np.where(combined_changes)[0]

change_labels = [
    f"{sequence_data.loc[i, 'phase']}\n{sequence_data.loc[i, 'behavior']}"
    for i in change_frames
]

# --- Prepare figure layout ---
fig = plt.figure(figsize=(20, 13))
gs = gridspec.GridSpec(5, 5, height_ratios=[0.2, 2, 1, 1, 1])  # Row 0: Title text

# --- Dynamic Text for Target Info ---
title_ax = fig.add_subplot(gs[0, :])
title_ax.axis("off")
title_text = title_ax.text(
    0.01, 0.5, "", fontsize=18, ha="left", va="center", transform=title_ax.transAxes
)

# --- ToF plots (row 1) ---
tof_groups = {
    f"tof_v{i}": [col for col in sequence_data.columns if f"tof_{i}_" in col]
    for i in range(1, 6)
}
tof_axes = [fig.add_subplot(gs[1, i]) for i in range(5)]
tof_imshows = []
for ax in tof_axes:
    im = ax.imshow(np.zeros((8, 8)), cmap="hot", vmin=-1, vmax=255)
    ax.axis("off")
    tof_imshows.append(im)

# --- Acceleration plot (row 2) ---
acc_ax = fig.add_subplot(gs[2, :])
(acc_x_line,) = acc_ax.plot([], [], label="acc_x")
(acc_y_line,) = acc_ax.plot([], [], label="acc_y")
(acc_z_line,) = acc_ax.plot([], [], label="acc_z")
acc_ax.legend(loc="upper right")
acc_ax.set_ylabel("Acceleration", fontsize=14)
acc_ax.set_xlim(0, n_frames)
acc_ax.set_ylim(
    sequence_data[["acc_x", "acc_y", "acc_z"]].min().min() - 1,
    sequence_data[["acc_x", "acc_y", "acc_z"]].max().max() + 1,
)

# --- Euler angles (row 3) ---
quats = sequence_data[["rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy()
eulers = R.from_quat(quats).as_euler("xyz", degrees=True)
sequence_data["pitch"], sequence_data["roll"], sequence_data["yaw"] = eulers.T

euler_ax = fig.add_subplot(gs[3, :])
(pitch_line,) = euler_ax.plot([], [], label="pitch")
(roll_line,) = euler_ax.plot([], [], label="roll")
(yaw_line,) = euler_ax.plot([], [], label="yaw")
euler_ax.legend(loc="upper right")
euler_ax.set_ylabel("Euler Angles (Â°)", fontsize=14)
euler_ax.set_xlim(0, n_frames)
euler_ax.set_ylim(
    sequence_data[["pitch", "roll", "yaw"]].min().min() - 5,
    sequence_data[["pitch", "roll", "yaw"]].max().max() + 5,
)

# --- Thermal sensors (row 4) ---
thm_cols = [col for col in sequence_data.columns if col.startswith("thm_")]
thm_ax = fig.add_subplot(gs[4, :])
thm_lines = []
for col in thm_cols:
    (line,) = thm_ax.plot([], [], label=col, linewidth=1)
    thm_lines.append(line)
thm_ax.set_xlim(0, n_frames)
thm_ax.set_ylim(
    sequence_data[thm_cols].min().min() - 1, sequence_data[thm_cols].max().max() + 1
)
thm_ax.set_ylabel("Temperature", fontsize=14)
thm_ax.legend(ncol=len(thm_cols) // 2, loc="upper right", fontsize=8)


def draw_change_lines(ax, let_text=False):
    y_min, y_max = ax.get_ylim()
    label_spacing = (y_max - y_min) * 0.05  # spacing between labels
    used_y_positions = {}  # track how many labels placed at each x

    for i, frame in enumerate(change_frames):
        ax.axvline(frame, color="purple", linestyle="--", alpha=0.5)

        if let_text:
            # Count how many labels already placed at this x to stack
            count = used_y_positions.get(frame, 0)
            y_pos = y_min + label_spacing * count
            used_y_positions[frame] = count + 1

            ax.text(
                frame + 0.5,
                y_pos,
                change_labels[i],
                color="purple",
                rotation=90,
                fontsize=16,
                ha="left",
                va="bottom",
            )



draw_change_lines(acc_ax)
draw_change_lines(euler_ax)
draw_change_lines(thm_ax,let_text=True)


# --- Update function ---
def update(frame):
    # ğŸ“Œ Update title info
    s = sequence_data.loc[
        frame, ["subject", "orientation", "behavior", "phase", "gesture"]
    ]
    title_text.set_text(
        # f"Sequence: {selected_sequence} | Subject: {s.subject} | Orientation: {s.orientation} | "
        # f"Behavior: {s.behavior} | Phase: {s.phase} | Gesture: {s.gesture}"
        
        f"{selected_sequence} | {s.subject} | Frame {frame:04}\n"+
        f"Orientation: {s.orientation} | Gesture: {s.gesture}\n"+
        f"Phase: {s.phase} | Behavior: {s.behavior}"
    )

    # ğŸ�›ï¸� ToF
    for i, (sensor_name, cols) in enumerate(tof_groups.items()):
        vals = sequence_data.loc[frame, cols].astype(float).values.reshape(8, 8)
        tof_imshows[i].set_array(vals)

    # ğŸ“ˆ Acc
    acc_x_line.set_data(range(frame), sequence_data["acc_x"][:frame])
    acc_y_line.set_data(range(frame), sequence_data["acc_y"][:frame])
    acc_z_line.set_data(range(frame), sequence_data["acc_z"][:frame])

    # ğŸ§­ Euler
    pitch_line.set_data(range(frame), sequence_data["pitch"][:frame])
    roll_line.set_data(range(frame), sequence_data["roll"][:frame])
    yaw_line.set_data(range(frame), sequence_data["yaw"][:frame])

    # ğŸŒ¡ï¸� Thermal
    for i, col in enumerate(thm_cols):
        thm_lines[i].set_data(range(frame), sequence_data[col][:frame])

    return (
        [title_text]
        + tof_imshows
        + [acc_x_line, acc_y_line, acc_z_line, pitch_line, roll_line, yaw_line]
        + thm_lines
    )


# ğŸ�¬ Create Animation
ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=100, blit=True)
plt.close()

# ğŸ’¾ Save to MP4
writer = FFMpegWriter(fps=10, metadata=dict(artist="You"), bitrate=1800)
ani.save(f"sequence_animation_{selected_sequence}.mp4", writer=writer, dpi=150)

# ğŸ’» Display inline (for Jupyter)
HTML(ani.to_jshtml())


display(seq_df[['sequence_counter', 'orientation', 'behavior', 'phase', 'gesture']].head(10))


