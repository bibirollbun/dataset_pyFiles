# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.auto import tqdm
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


# Set some display options for pandas for better readability
pd.set_option('display.max_columns', 100)
sns.set_style('whitegrid')

# Define the path to your data
DATA_PATH = '/kaggle/input/MABe-mouse-behavior-detection/'

# Load the metadata files
print("Loading train.csv (metadata)...")
df_train_meta = pd.read_csv(DATA_PATH + 'train.csv')

print("Loading test.csv (metadata)...")
df_test_meta = pd.read_csv(DATA_PATH + 'test.csv')

print("\n--- Train Metadata ---")
print(f"Shape: {df_train_meta.shape}")
display(df_train_meta.head())

print("\n--- Test Metadata ---")
print(f"Shape: {df_test_meta.shape}")
display(df_test_meta.head())


# Select the first video from the metadata as our sample
sample_video_meta = df_train_meta.iloc[0]
sample_lab_id = sample_video_meta['lab_id']
sample_video_id = sample_video_meta['video_id']

print(f"Loading sample video...\n  Lab ID: {sample_lab_id}\n  Video ID: {sample_video_id}")


# Construct the file paths using the lab and video IDs
tracking_path = os.path.join(DATA_PATH, 'train_tracking', sample_lab_id, f'{sample_video_id}.parquet')
annotation_path = os.path.join(DATA_PATH, 'train_annotation', sample_lab_id, f'{sample_video_id}.parquet')

# Load the actual data from the parquet files
df_tracking_sample = pd.read_parquet(tracking_path)
df_annot_sample = pd.read_parquet(annotation_path)

print("\n--- Sample Tracking Data ---")
print(f"Shape: {df_tracking_sample.shape}")
print("Info:")
df_tracking_sample.info()
print("\nFirst 5 rows:")
display(df_tracking_sample.head())

print("\n\n--- Sample Annotation Data ---")
print(f"Shape: {df_annot_sample.shape}")
print("Info:")
df_annot_sample.info()
print("\nFirst 5 rows:")
display(df_annot_sample.head())


# 1. See what bodyparts are available
unique_bodyparts = df_tracking_sample['bodypart'].unique()
print(f"Unique bodyparts tracked: {unique_bodyparts}\n")


# 2. Pivot the table to get a "wide" format
# We want one row per video_frame, and columns for each mouse's bodypart's x and y coordinates.

print("Pivoting data from long to wide format...")


# Create a pivot for the 'x' coordinates
pivot_x = df_tracking_sample.pivot(
    index='video_frame', 
    columns=['mouse_id', 'bodypart'], 
    values='x'
)
# Rename columns for clarity, e.g., (1, 'nose') -> 'mouse1_nose_x'
pivot_x.columns = [f"mouse{m}_{bp}_x" for m, bp in pivot_x.columns]


# Create a pivot for the 'y' coordinates
pivot_y = df_tracking_sample.pivot(
    index='video_frame', 
    columns=['mouse_id', 'bodypart'], 
    values='y'
)
# Rename columns for clarity
pivot_y.columns = [f"mouse{m}_{bp}_y" for m, bp in pivot_y.columns]


# 3. Merge the x and y pivots into a single DataFrame
df_wide_sample = pd.concat([pivot_x, pivot_y], axis=1)

# Sort columns alphabetically for consistent order
df_wide_sample = df_wide_sample.sort_index(axis=1)


print("Pivoting complete.\n")
print("--- Reshaped Wide DataFrame ---")
print(f"Shape: {df_wide_sample.shape}")
display(df_wide_sample.head())


# 1. Define the core bodyparts we want to visualize
# We will ignore the 'headpiece' parts for this general visualization
ANATOMICAL_BODYPARTS = [
    'nose', 'ear_left', 'ear_right', 'neck', 'body_center', 
    'lateral_left', 'lateral_right', 'tail_base'
]


# 2. Define connections to draw a simple skeleton
# Each tuple represents a line from one bodypart to another
SKELETON_CONNECTIONS = [
    ('nose', 'ear_left'), ('nose', 'ear_right'), ('ear_left', 'ear_right'),
    ('nose', 'neck'), ('neck', 'body_center'),
    ('body_center', 'lateral_left'), ('body_center', 'lateral_right'),
    ('body_center', 'tail_base')
]

# Define a color for each mouse for consistent plotting
MOUSE_COLORS = {1: 'blue', 2: 'orange', 3: 'green', 4: 'red'}


# 3. Create the plotting function
def plot_frame(frame_data):
    """Plots the skeletons of all mice for a single frame of data."""
    
    plt.figure(figsize=(8, 8))
    
    # Iterate through each mouse
    for mouse_id in range(1, 5): # Assumes up to 4 mice
        
        # Check if data for this mouse exists in the frame
        if f'mouse{mouse_id}_nose_x' not in frame_data or pd.isna(frame_data[f'mouse{mouse_id}_nose_x']):
            continue # Skip if this mouse isn't tracked in this frame
            
        # Plot the keypoints (bodyparts)
        for part in ANATOMICAL_BODYPARTS:
            col_x = f'mouse{mouse_id}_{part}_x'
            col_y = f'mouse{mouse_id}_{part}_y'
            if col_x in frame_data and col_y in frame_data:
                plt.scatter(frame_data[col_x], frame_data[col_y], color=MOUSE_COLORS[mouse_id], label=f'Mouse {mouse_id}' if part == 'nose' else "")

        # Plot the skeleton connections
        for part1, part2 in SKELETON_CONNECTIONS:
            col1_x, col1_y = f'mouse{mouse_id}_{part1}_x', f'mouse{mouse_id}_{part1}_y'
            col2_x, col2_y = f'mouse{mouse_id}_{part2}_x', f'mouse{mouse_id}_{part2}_y'

            # Check if both points for the line exist
            if all(c in frame_data for c in [col1_x, col1_y, col2_x, col2_y]) and \
               pd.notna(frame_data[col1_x]) and pd.notna(frame_data[col2_x]):
                
                plt.plot([frame_data[col1_x], frame_data[col2_x]], 
                         [frame_data[col1_y], frame_data[col2_y]], 
                         color=MOUSE_COLORS[mouse_id], alpha=0.7)

    plt.title(f"Mouse Positions at Frame {frame_data.name}")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    
    # Invert the y-axis because image coordinates (0,0) are usually at the top-left
    plt.gca().invert_yaxis()
    plt.legend()
    plt.axis('equal') # Ensure aspect ratio is maintained
    plt.show()


# 4. Use the function to plot a specific frame
FRAME_TO_PLOT = 1000
plot_frame(df_wide_sample.loc[FRAME_TO_PLOT])


# 1. Pick a behavior to animate from our annotation sample
behavior_to_animate = df_annot_sample.iloc[0]
start_frame = behavior_to_animate['start_frame']
stop_frame = behavior_to_animate['stop_frame']
action = behavior_to_animate['action']
agent = behavior_to_animate['agent_id']

print(f"Preparing to animate behavior: '{action}' by Mouse {agent}")
print(f"Frame range: {start_frame} to {stop_frame}")


# Add a small buffer before and after to see the context
ANIM_START = max(0, start_frame - 20)
ANIM_STOP = stop_frame + 20

# Slice our wide dataframe to get only the frames we need for the animation
anim_df = df_wide_sample.loc[ANIM_START:ANIM_STOP]


# --- Animation Setup ---

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Determine axis limits from the entire animation sequence to prevent jittering
x_min, x_max = anim_df.filter(like='_x').min().min(), anim_df.filter(like='_x').max().max()
y_min, y_max = anim_df.filter(like='_y').min().min(), anim_df.filter(like='_y').max().max()
padding = 50 # Add some padding to the plot
ax.set_xlim(x_min - padding, x_max + padding)
ax.set_ylim(y_min - padding, y_max + padding)


# The function that will draw each frame of the animation
def update(frame_num):
    ax.clear() # Clear the previous frame
    
    # Get the data for the current frame
    frame_data = anim_df.iloc[frame_num]
    current_real_frame = anim_df.index[frame_num]
    
    # Plot each mouse for the current frame
    for mouse_id in range(1, 5):
        if f'mouse{mouse_id}_nose_x' not in frame_data or pd.isna(frame_data[f'mouse{mouse_id}_nose_x']):
            continue

        # Plot keypoints
        for part in ANATOMICAL_BODYPARTS:
            col_x, col_y = f'mouse{mouse_id}_{part}_x', f'mouse{mouse_id}_{part}_y'
            if col_x in frame_data and col_y in frame_data:
                ax.scatter(frame_data[col_x], frame_data[col_y], color=MOUSE_COLORS[mouse_id])

        # Plot skeleton
        for part1, part2 in SKELETON_CONNECTIONS:
            col1_x, col1_y = f'mouse{mouse_id}_{part1}_x', f'mouse{mouse_id}_{part1}_y'
            col2_x, col2_y = f'mouse{mouse_id}_{part2}_x', f'mouse{mouse_id}_{part2}_y'
            if all(c in frame_data for c in [col1_x, col1_y, col2_x, col2_y]) and \
               pd.notna(frame_data[col1_x]) and pd.notna(frame_data[col2_x]):
                ax.plot([frame_data[col1_x], frame_data[col2_x]], [frame_data[col1_y], frame_data[col2_y]], color=MOUSE_COLORS[mouse_id], alpha=0.7)

    # Set titles and labels for the frame
    ax.set_title(f"Behavior: '{action}' by Mouse {agent} | Frame: {current_real_frame}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.invert_yaxis() # Invert y-axis for image coordinates
    return ax,


# Create the animation
# frames=len(anim_df) specifies how many times to call the update function
# interval=50 is the delay between frames in milliseconds
ani = FuncAnimation(fig, update, frames=len(anim_df), interval=50, blit=False)

# Display the animation in the notebook
# This may take a little while to render
HTML(ani.to_jshtml())


# This cell is designed to be self-contained. It will build the full annotation
# dataframe if it doesn't already exist in memory.
if 'df_annotations_full' not in locals():
    print("Building the full annotations dataframe by combining all individual annotation files...")

    all_annotations_list = []
    for index, row in tqdm(df_train_meta.iterrows(), total=df_train_meta.shape[0]):
        lab_id = row['lab_id']
        video_id = row['video_id']
        annotation_path = os.path.join(DATA_PATH, 'train_annotation', lab_id, f'{video_id}.parquet')
        
        if os.path.exists(annotation_path):
            temp_df = pd.read_parquet(annotation_path)
            temp_df['video_id'] = video_id
            all_annotations_list.append(temp_df)

    df_annotations_full = pd.concat(all_annotations_list, ignore_index=True)
    print(f"\nSuccessfully created full annotation dataframe with shape: {df_annotations_full.shape}")
else:
    print("Full annotation dataframe already exists in memory. Proceeding with analysis.")



# --- Behavior Frequency Analysis ---
print("\n--- Behavior Frequency Analysis ---")

behavior_counts = df_annotations_full['action'].value_counts()
plt.figure(figsize=(12, 8))
sns.barplot(x=behavior_counts.index, y=behavior_counts.values, palette='viridis')
plt.title('Frequency of Each Behavior Across the Entire Training Set', fontsize=16)
plt.xlabel('Behavior', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- Behavior Duration Analysis ---
print("\n--- Behavior Duration Analysis ---")
df_annotations_full['duration_frames'] = df_annotations_full['stop_frame'] - df_annotations_full['start_frame']

# Let's see how many zero-duration events we have
zero_duration_count = (df_annotations_full['duration_frames'] == 0).sum()
print(f"Found {zero_duration_count} events with a duration of 0 frames.")

print("\nBasic statistics for behavior durations (in frames):")
display(df_annotations_full['duration_frames'].describe())

# Add 1 to duration before plotting on a log scale to handle zeros
plt.figure(figsize=(12, 6))
sns.histplot(df_annotations_full['duration_frames'] + 1, bins=100, log_scale=True)
plt.title('Distribution of Behavior Durations (Log Scale, Duration+1)', fontsize=16)
plt.xlabel('Duration (Frames) + 1 - Log Scale', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

plt.figure(figsize=(12, 10))



order = df_annotations_full.groupby('action')['duration_frames'].median().sort_values(ascending=False).index
# Use duration + 1 for the x-axis in the boxplot as well
sns.boxplot(x=df_annotations_full['duration_frames'] + 1, y='action', data=df_annotations_full, order=order, palette='coolwarm')
plt.title('Duration of Each Behavior Type', fontsize=16)
plt.xlabel('Duration (Frames) + 1 - Log Scale', fontsize=12)
plt.ylabel('Behavior', fontsize=12)
plt.xscale('log')
plt.tight_layout()
plt.show()


# First, ensure 'df_annotations_full' exists
if 'df_annotations_full' not in locals():
    print("Full annotation dataframe is not in memory. Please re-run the previous cell (Step 6).")
else:
    print("--- Lab Variability Analysis ---")
    
    # We need to merge our annotations with the metadata to get the lab_id for each event
    # We select only the 'video_id' and 'lab_id' from the metadata to keep the merge light
    df_lab_info = df_train_meta[['video_id', 'lab_id']]
    
    # Perform a left merge to add 'lab_id' to each annotation
    df_annotations_with_lab = pd.merge(df_annotations_full, df_lab_info, on='video_id', how='left')
    
    print(f"Successfully merged lab info. New shape: {df_annotations_with_lab.shape}")
    display(df_annotations_with_lab.head())
    
    # Now, let's plot the behavior counts per lab
    plt.figure(figsize=(15, 8))
    
    # We use crosstab to count occurrences of each action within each lab, then normalize
    # to see the percentage/proportion, which is better for comparison
    crosstab_norm = pd.crosstab(df_annotations_with_lab['lab_id'], 
                               df_annotations_with_lab['action'], 
                               normalize='index') # 'normalize=index' calculates percentages per lab
    
    sns.heatmap(crosstab_norm, cmap='viridis', annot=False) # 'annot=True' can be messy if too many classes
    plt.title('Proportion of Behaviors by Lab', fontsize=16)
    plt.xlabel('Behavior', fontsize=12)
    plt.ylabel('Lab ID', fontsize=12)
    plt.show()




