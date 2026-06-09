import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


names=os.listdir('/kaggle/input/MABe-mouse-behavior-detection/train_annotation')
print(names)


tracking = pd.read_parquet('/kaggle/input/MABe-mouse-behavior-detection/train_tracking/AdaptableSnail/1212811043.parquet')

print("Tracking data shape:", tracking.shape)

print("\nTracking columns:")
print(tracking.columns.tolist())

print("\nTracking head:")
print(tracking.head())


bodycenter=tracking[tracking['bodypart']=='body_center']
display(bodycenter)


# Summary statistics
print("Summary statistics:")
print(tracking[['x', 'y']].describe())

# Plot the distribution of coordinates
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(tracking['x'], bins=50)
plt.title('Distribution of X coordinates')

plt.subplot(1, 2, 2)
sns.histplot(tracking['y'], bins=50)
plt.title('Distribution of Y coordinates')
plt.tight_layout()
plt.show()

# Plot positions of different mice
plt.figure(figsize=(10, 8))
for mouse_id in tracking['mouse_id'].unique():
    mouse_data = tracking[tracking['mouse_id'] == mouse_id]
    plt.scatter(mouse_data['x'], mouse_data['y'], alpha=0.1, label=f'Mouse {mouse_id}')

plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Mouse Positions')
plt.legend()
plt.show()


# Analyze movement over time for a specific mouse
mouse_1_data = tracking[tracking['mouse_id'] == 1].sort_values('video_frame')

plt.figure(figsize=(15, 5))
plt.plot(mouse_1_data['video_frame'], mouse_1_data['x'], label='X position')
plt.plot(mouse_1_data['video_frame'], mouse_1_data['y'], label='Y position')
plt.xlabel('Video Frame')
plt.ylabel('Position')
plt.title('Mouse 1 Position Over Time')
plt.legend()
plt.show()

# Calculate velocity (pixels per frame)
mouse_1_data['x_velocity'] = mouse_1_data['x'].diff()
mouse_1_data['y_velocity'] = mouse_1_data['y'].diff()
mouse_1_data['speed'] = np.sqrt(mouse_1_data['x_velocity']**2 + mouse_1_data['y_velocity']**2)

plt.figure(figsize=(12, 4))
plt.plot(mouse_1_data['video_frame'], mouse_1_data['speed'])
plt.xlabel('Video Frame')
plt.ylabel('Speed (pixels/frame)')
plt.title('Mouse 1 Speed Over Time')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import imageio

# Load data
tracking = pd.read_parquet('/kaggle/input/MABe-mouse-behavior-detection/train_tracking/AdaptableSnail/1212811043.parquet')

# Extract only Mouse 1 data
mouse1_data = tracking[tracking['mouse_id'] == 1].sort_values('video_frame').copy()

# Handle infinite values
mouse1_data['x'] = mouse1_data['x'].replace([np.inf, -np.inf], np.nan)
mouse1_data['y'] = mouse1_data['y'].replace([np.inf, -np.inf], np.nan)

# Get coordinate ranges
x_min, x_max = mouse1_data['x'].min(), mouse1_data['x'].max()
y_min, y_max = mouse1_data['y'].min(), mouse1_data['y'].max()

# Create animation
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(x_min - 10, x_max + 10)
ax.set_ylim(y_min - 10, y_max + 10)
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_title('Mouse 1 Movement Animation')

# Trajectory line and current position point
line, = ax.plot([], [], 'b-', alpha=0.5, linewidth=1)
point, = ax.plot([], [], 'ro', markersize=8)

# Frame number display text
frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    point.set_data([], [])
    frame_text.set_text('')
    return line, point, frame_text

def update(frame):
    # Get data up to current frame
    current_data = mouse1_data[mouse1_data['video_frame'] <= frame]
    
    if len(current_data) > 0:
        # Update trajectory
        line.set_data(current_data['x'], current_data['y'])
        
        # Update current position (latest position)
        current_pos = current_data.iloc[-1]
        point.set_data([current_pos['x']], [current_pos['y']])
        
        # Update frame number
        frame_text.set_text(f'Frame: {frame}\nPosition: ({current_pos["x"]:.1f}, {current_pos["y"]:.1f})')
    
    return line, point, frame_text

# Create animation (limit frames for lighter processing)
frames = mouse1_data['video_frame'].unique()
if len(frames) > 500:  # Skip frames if there are too many
    frames = frames[::len(frames)//500]

ani = FuncAnimation(fig, update, frames=frames,
                    init_func=init, blit=True, interval=50)

# Save as GIF
print("Creating animation...")
ani.save('mouse1_movement.gif', writer='pillow', fps=20, dpi=80)
plt.close()

print("Animation GIF saved: mouse1_movement.gif")

output_path="mouse1_movement.gif"

from IPython.display import Image
Image(open(output_path, 'rb').read())

