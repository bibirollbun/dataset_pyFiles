# Display a short video from tracking parquet data
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display

# Paths
DATA_DIR = "/kaggle/input/MABe-mouse-behavior-detection/"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_TRACKING = os.path.join(DATA_DIR, "train_tracking")

# Load metadata
train_df = pd.read_csv(TRAIN_CSV)

# Pick a sample video
sample_row = train_df.iloc[0]
lab_id = sample_row['lab_id']
video_id = sample_row['video_id']
parquet_path = os.path.join(TRAIN_TRACKING, lab_id, f"{video_id}.parquet")

print(f"Loading tracking data: {parquet_path}")
df = pd.read_parquet(parquet_path)

# Show first 100 frames
frames = sorted(df['video_frame'].unique())[:100]
bodyparts = df['bodypart'].unique()
mice = df['mouse_id'].unique()

# Prepare figure
fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.tab10.colors

def animate(i):
    ax.clear()
    frame = frames[i]
    ax.set_title(f"Frame {frame}")
    ax.set_xlim(df['x'].min(), df['x'].max())
    ax.set_ylim(df['y'].min(), df['y'].max())
    for j, mouse in enumerate(mice):
        d = df[(df['video_frame']==frame) & (df['mouse_id']==mouse)]
        ax.scatter(d['x'], d['y'], label=f"Mouse {mouse}", color=colors[j%len(colors)])
    ax.legend()

ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50)
plt.close(fig)
display(HTML(ani.to_jshtml()))

