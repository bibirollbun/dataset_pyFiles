
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import Image


filename='AdaptableSnail/1212811043.parquet'

anno=pd.read_parquet(f'/kaggle/input/MABe-mouse-behavior-detection/train_annotation/{filename}')
track=pd.read_parquet(f'/kaggle/input/MABe-mouse-behavior-detection/train_tracking/{filename}')


display(anno)


display(anno.iloc[367:368])


track2 = track[
    (track['video_frame'] >= 89171) &
    (track['video_frame'] <= 89194) &
    (track['mouse_id'].isin([1, 3]))
]


display(track2)





import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import Image

# Prepare data - using only mouse1
mouse1 = track2[track2['mouse_id'] == 1].copy()

# Handle infinite values and NaN values
mouse1['x'] = mouse1['x'].replace([np.inf, -np.inf], np.nan)
mouse1['y'] = mouse1['y'].replace([np.inf, -np.inf], np.nan)
mouse1.dropna(subset=['x', 'y'], inplace=True)

# Coordinate ranges
x_min = mouse1['x'].min()
x_max = mouse1['x'].max()
y_min = mouse1['y'].min()
y_max = mouse1['y'].max()

# Get a list of unique body parts
bodyparts = mouse1['bodypart'].unique()
print("Bodyparts found:", bodyparts)

# Assign a color to each body part
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
if len(bodyparts) > len(colors):
    # If there are too many body parts and not enough colors, generate additional colors
    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())[:len(bodyparts)-len(colors)]

# Create animation
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(x_min - 10, x_max + 10)
ax.set_ylim(y_min - 10, y_max + 10)
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_title('Mouse 1 Body Parts Movement Animation')

# Lists to store the trajectory and current position of each body part
lines = []
points = []

# Create a trajectory line and a point for each body part
for i, bodypart in enumerate(bodyparts):
    color = colors[i % len(colors)]
    line, = ax.plot([], [], color=color, alpha=0.5, linewidth=1, label=f'{bodypart}')
    point, = ax.plot([], [], 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=0.5)
    lines.append(line)
    points.append(point)

frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
ax.legend(loc='upper right')

def init():
    for line, point in zip(lines, points):
        line.set_data([], [])
        point.set_data([], [])
    frame_text.set_text('')
    return lines + points + [frame_text]

def update(frame):
    # Update the data for each body part
    for i, bodypart in enumerate(bodyparts):
        # Filter the data for the current body part
        bodypart_data = mouse1[mouse1['bodypart'] == bodypart]
        data = bodypart_data[bodypart_data['video_frame'] <= frame]
        
        if len(data) > 0:
            lines[i].set_data(data['x'], data['y'])
            current = data.iloc[-1]
            points[i].set_data([current['x']], [current['y']])
        else:
            lines[i].set_data([], [])
            points[i].set_data([], [])
    
    frame_text.set_text(f'Frame: {frame}')
    return lines + points + [frame_text]

# Frames (using only mouse1)
frames = mouse1['video_frame'].unique()
frames.sort()

print(f"Total frames: {len(frames)}")

if len(frames) > 500:
    step = max(1, len(frames) // 500)
    frames = frames[::step]
    print(f"Sampled frames: {len(frames)}")

# Animation
ani = FuncAnimation(fig, update, frames=frames,
                    init_func=init, blit=True, interval=100)

# Save as GIF
output_path = "mouse1_bodyparts_movement.gif"
ani.save(output_path, writer='pillow', fps=10, dpi=100)
plt.close()
print("Animation GIF saved:", output_path)

# Display in notebook
Image(open(output_path, 'rb').read())





from matplotlib.lines import Line2D
import numpy as np

# Create Line2D objects for the full mouse body lines
mouse_lines = {}

# Initialization (body, tail, head)
tail_line, = ax.plot([], [], 'brown', linewidth=3, alpha=0.7)
body_line, = ax.plot([], [], 'gray', linewidth=4, alpha=0.7)
head_line, = ax.plot([], [], 'gray', linewidth=3, alpha=0.7)

mouse_lines['tail'] = tail_line
mouse_lines['body'] = body_line
mouse_lines['head'] = head_line

# Modify the initialization function
def init():
    for i, bodypart in enumerate(bodyparts):
        lines[i].set_data([], [])
        points[i].set_data([], [])
    
    # Initialize mouse lines as well
    for line in mouse_lines.values():
        line.set_data([], [])
    
    frame_text.set_text('')
    return lines + points + [frame_text] + list(mouse_lines.values())

# Modify the update function
def update2(frame):
    # Original body part animation update
    for i, bodypart in enumerate(bodyparts):
        bodypart_data = mouse1[mouse1['bodypart'] == bodypart]
        data = bodypart_data[bodypart_data['video_frame'] <= frame]
        
        if len(data) > 0:
            lines[i].set_data(data['x'], data['y'])
            current = data.iloc[-1]
            points[i].set_data([current['x']], [current['y']])
        else:
            lines[i].set_data([], [])
            points[i].set_data([], [])
    
    # Update full mouse body lines
    current_points = {}
    for bp in bodyparts:
        d = mouse1[(mouse1['bodypart']==bp) & (mouse1['video_frame']==frame)]
        if len(d) > 0:
            current_points[bp] = np.array([d['x'].values[0], d['y'].values[0]])
    
    # Draw the tail (connecting 3 points smoothly)
    if all(k in current_points for k in ['tail_base','tail_midpoint','tail_tip']):
        tail_x = [current_points['tail_base'][0], current_points['tail_midpoint'][0], current_points['tail_tip'][0]]
        tail_y = [current_points['tail_base'][1], current_points['tail_midpoint'][1], current_points['tail_tip'][1]]
        mouse_lines['tail'].set_data(tail_x, tail_y)
    else:
        mouse_lines['tail'].set_data([], [])
    
    # Draw the body (a more natural shape)
    if all(k in current_points for k in ['lateral_left','lateral_right','body_center']):
        # Draw a shape similar to an ellipse
        body_x = [
            current_points['lateral_left'][0],
            current_points['body_center'][0],
            current_points['lateral_right'][0],
            current_points['body_center'][0],
            current_points['lateral_left'][0]  # close the shape
        ]
        body_y = [
            current_points['lateral_left'][1],
            current_points['body_center'][1] + 5,  # slightly up
            current_points['lateral_right'][1],
            current_points['body_center'][1] - 5,  # slightly down
            current_points['lateral_left'][1]  # close the shape
        ]
        mouse_lines['body'].set_data(body_x, body_y)
    else:
        mouse_lines['body'].set_data([], [])
    
    # Draw the head (a shape similar to a triangle)
    if all(k in current_points for k in ['neck','nose','ear_left','ear_right']):
        head_x = [
            current_points['neck'][0],
            current_points['ear_left'][0],
            current_points['nose'][0],
            current_points['ear_right'][0],
            current_points['neck'][0]  # close the shape
        ]
        head_y = [
            current_points['neck'][1],
            current_points['ear_left'][1],
            current_points['nose'][1],
            current_points['ear_right'][1],
            current_points['neck'][1]  # close the shape
        ]
        mouse_lines['head'].set_data(head_x, head_y)
    else:
        mouse_lines['head'].set_data([], [])
    
    frame_text.set_text(f'Frame: {frame}')
    
    # Modify the return value
    return_lines = lines + points + [frame_text] + [mouse_lines['tail'], mouse_lines['body'], mouse_lines['head']]
    return return_lines

# Create the animation (with the correct variable name)
ani2 = FuncAnimation(fig, update2, frames=frames,
                    init_func=init, blit=True, interval=100)

# Save using ani2 (important!)
output_path2 = "mouse1_bodyparts_movement2.gif"
ani2.save(output_path2, writer='pillow', fps=10, dpi=100)
plt.close()
print("Animation GIF saved:", output_path2)

# Display in notebook
Image(open(output_path2, 'rb').read())





from matplotlib.patches import Polygon, Ellipse
import numpy as np
from scipy.interpolate import interp1d

# Create patch objects for the silhouettes
mouse_patches = {}

# Patch for the overall body silhouette
silhouette_patch = Polygon([[0,0]], closed=True, color='darkgray', alpha=0.6, edgecolor='black', linewidth=2)
ax.add_patch(silhouette_patch)
mouse_patches['silhouette'] = silhouette_patch

# Patches for individual parts (optional)
body_patch = Ellipse((0,0), width=10, height=5, color='gray', alpha=0.5)
head_patch = Ellipse((0,0), width=8, height=6, color='gray', alpha=0.5)
ax.add_patch(body_patch)
ax.add_patch(head_patch)
mouse_patches['body'] = body_patch
mouse_patches['head'] = head_patch

# Update the initialization function
def init():
    for i, bodypart in enumerate(bodyparts):
        lines[i].set_data([], [])
        points[i].set_data([], [])
    
    # Initialize mouse lines
    for line in mouse_lines.values():
        line.set_data([], [])
    
    # Initialize patches
    for patch in mouse_patches.values():
        if isinstance(patch, Polygon):
            patch.set_xy(np.array([[0,0]]))
        elif isinstance(patch, Ellipse):
            patch.set_center((0,0))
            patch.set_width(0)
            patch.set_height(0)
    
    frame_text.set_text('')
    return_lines = lines + points + [frame_text] + list(mouse_lines.values())
    return_patches = list(mouse_patches.values())
    return return_lines + return_patches

# Improve the update function to add a silhouette
def update3(frame):
    # Update original body part animation
    for i, bodypart in enumerate(bodyparts):
        bodypart_data = mouse1[mouse1['bodypart'] == bodypart]
        data = bodypart_data[bodypart_data['video_frame'] <= frame]
        
        if len(data) > 0:
            lines[i].set_data(data['x'], data['y'])
            current = data.iloc[-1]
            points[i].set_data([current['x']], [current['y']])
        else:
            lines[i].set_data([], [])
            points[i].set_data([], [])
    
    # Update full mouse body lines
    current_points = {}
    for bp in bodyparts:
        d = mouse1[(mouse1['bodypart']==bp) & (mouse1['video_frame']==frame)]
        if len(d) > 0:
            current_points[bp] = np.array([d['x'].values[0], d['y'].values[0]])
    
    # Create the full body silhouette
    if all(k in current_points for k in ['tail_base', 'lateral_left', 'lateral_right', 'neck', 'ear_left', 'ear_right', 'nose']):
        # Define the contour points of the mouse (clockwise or counter-clockwise)
        silhouette_points = [
            current_points['tail_base'],      # Base of the tail
            current_points['lateral_left'],   # Left side
            current_points['ear_left'],       # Left ear
            current_points['nose'],           # Tip of the nose
            current_points['ear_right'],      # Right ear
            current_points['lateral_right'],  # Right side
            current_points['tail_base']       # Close the shape
        ]
        
        # Extract the coordinates of the points
        silhouette_xy = np.array(silhouette_points)
        mouse_patches['silhouette'].set_xy(silhouette_xy)
    else:
        mouse_patches['silhouette'].set_xy(np.array([[0,0]]))
    
    # Draw the tail
    if all(k in current_points for k in ['tail_base','tail_midpoint','tail_tip']):
        tail_x = [current_points['tail_base'][0], current_points['tail_midpoint'][0], current_points['tail_tip'][0]]
        tail_y = [current_points['tail_base'][1], current_points['tail_midpoint'][1], current_points['tail_tip'][1]]
        mouse_lines['tail'].set_data(tail_x, tail_y)
    else:
        mouse_lines['tail'].set_data([], [])
    
    # Body silhouette (represented by an ellipse)
    if all(k in current_points for k in ['lateral_left','lateral_right','body_center']):
        # Calculate the center and size of the body
        body_center_x = current_points['body_center'][0]
        body_center_y = current_points['body_center'][1]
        body_width = abs(current_points['lateral_left'][0] - current_points['lateral_right'][0]) * 1.2
        body_height = body_width * 0.6
        
        mouse_patches['body'].set_center((body_center_x, body_center_y))
        mouse_patches['body'].set_width(body_width)
        mouse_patches['body'].set_height(body_height)
    else:
        mouse_patches['body'].set_width(0)
        mouse_patches['body'].set_height(0)
    
    # Head silhouette
    if all(k in current_points for k in ['neck','nose','ear_left','ear_right']):
        head_center_x = (current_points['neck'][0] + current_points['nose'][0]) / 2
        head_center_y = (current_points['neck'][1] + current_points['nose'][1]) / 2
        head_width = abs(current_points['ear_left'][0] - current_points['ear_right'][0]) * 1.1
        head_height = abs(current_points['neck'][1] - current_points['nose'][1]) * 1.3
        
        mouse_patches['head'].set_center((head_center_x, head_center_y))
        mouse_patches['head'].set_width(head_width)
        mouse_patches['head'].set_height(head_height)
        
        # Also update the head line
        head_x = [
            current_points['neck'][0],
            current_points['ear_left'][0],
            current_points['nose'][0],
            current_points['ear_right'][0],
            current_points['neck'][0]
        ]
        head_y = [
            current_points['neck'][1],
            current_points['ear_left'][1],
            current_points['nose'][1],
            current_points['ear_right'][1],
            current_points['neck'][1]
        ]
        mouse_lines['head'].set_data(head_x, head_y)
    else:
        mouse_patches['head'].set_width(0)
        mouse_patches['head'].set_height(0)
        mouse_lines['head'].set_data([], [])
    
    frame_text.set_text(f'Frame: {frame}')
    
    # Add patches to the return value
    return_lines = lines + points + [frame_text] + list(mouse_lines.values())
    return_patches = list(mouse_patches.values())
    return return_lines + return_patches

# Create the animation
ani3 = FuncAnimation(fig, update3, frames=frames,
                    init_func=init, blit=True, interval=100)

# Save the GIF
output_path3 = "mouse1_silhouette_animation3.gif"
ani3.save(output_path3, writer='pillow', fps=10, dpi=100)
plt.close()
print("Silhouette Animation GIF saved:", output_path3)

# Display
Image(open(output_path3, 'rb').read())




