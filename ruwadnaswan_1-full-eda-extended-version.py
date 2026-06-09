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





# Calculate missing data statistics for our sample video
print("=== Missing Data Analysis for Sample Video ===\n")

# Calculate percentage of missing values per column
missing_pct = (df_tracking_sample.isna().sum() / len(df_tracking_sample)) * 100
missing_by_bodypart = missing_pct.groupby(df_tracking_sample.columns.str.extract(r'(\w+)_[xy]$')[0]).mean()

print("Missing data percentage by bodypart:")
display(missing_by_bodypart.sort_values(ascending=False))

# Visualize missing data pattern
plt.figure(figsize=(14, 6))
missing_pct_wide = (df_wide_sample.isna().sum() / len(df_wide_sample)) * 100
missing_pct_wide.sort_values(ascending=False).head(20).plot(kind='barh', color='coral')
plt.xlabel('Percentage Missing (%)', fontsize=12)
plt.ylabel('Feature (Mouse_Bodypart_Coordinate)', fontsize=12)
plt.title('Top 20 Features with Most Missing Data', fontsize=14)
plt.tight_layout()
plt.show()



print("=== Dataset-Wide Missing Data Analysis ===\n")
print("This will sample 100 random videos to analyze tracking quality...\n")

# Sample videos for efficiency
sample_size = min(100, len(df_train_meta))
sampled_videos = df_train_meta.sample(n=sample_size, random_state=42)

lab_missing_data = []

for idx, row in tqdm(sampled_videos.iterrows(), total=sample_size, desc="Analyzing videos"):
    lab_id = row['lab_id']
    video_id = row['video_id']
    tracking_path = os.path.join(DATA_PATH, 'train_tracking', lab_id, f'{video_id}.parquet')
    
    if os.path.exists(tracking_path):
        df_track = pd.read_parquet(tracking_path)
        missing_pct = (df_track[['x', 'y']].isna().sum().sum() / (len(df_track) * 2)) * 100
        
        lab_missing_data.append({
            'lab_id': lab_id,
            'video_id': video_id,
            'missing_pct': missing_pct
        })

df_missing = pd.DataFrame(lab_missing_data)

# Plot missing data by lab
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_missing, x='lab_id', y='missing_pct', palette='Set2')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Missing Data Percentage (%)', fontsize=12)
plt.xlabel('Lab ID', fontsize=12)
plt.title('Tracking Quality (Missing Data %) by Lab', fontsize=14)
plt.tight_layout()
plt.show()

print(f"\nOverall missing data statistics:")
print(df_missing['missing_pct'].describe())


# Calculate distance between all pairs of mice for the sample video
print("=== Calculating Spatial Features ===\n")

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_velocity(df, mouse_id, bodypart='nose', fps=30):
    """Calculate velocity for a specific mouse and bodypart."""
    x_col = f'mouse{mouse_id}_{bodypart}_x'
    y_col = f'mouse{mouse_id}_{bodypart}_y'
    
    if x_col not in df.columns or y_col not in df.columns:
        return None
    
    # Calculate displacement between consecutive frames
    dx = df[x_col].diff()
    dy = df[y_col].diff()
    
    # Calculate velocity (pixels per second)
    velocity = np.sqrt(dx**2 + dy**2) * fps
    
    return velocity

# Calculate distances between all mouse pairs (using nose positions)
mouse_pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

for m1, m2 in mouse_pairs:
    x1_col = f'mouse{m1}_nose_x'
    y1_col = f'mouse{m1}_nose_y'
    x2_col = f'mouse{m2}_nose_x'
    y2_col = f'mouse{m2}_nose_y'
    
    if all(col in df_wide_sample.columns for col in [x1_col, y1_col, x2_col, y2_col]):
        df_wide_sample[f'dist_mouse{m1}_mouse{m2}'] = calculate_distance(
            df_wide_sample[x1_col], df_wide_sample[y1_col],
            df_wide_sample[x2_col], df_wide_sample[y2_col]
        )

# Calculate velocities for each mouse
fps = sample_video_meta['frames_per_second']
for mouse_id in range(1, 5):
    vel = calculate_velocity(df_wide_sample, mouse_id, 'nose', fps)
    if vel is not None:
        df_wide_sample[f'mouse{mouse_id}_velocity'] = vel

print("Spatial features calculated successfully!")
print(f"\nNew feature columns: {[col for col in df_wide_sample.columns if 'dist_' in col or 'velocity' in col]}")


# Visualize distance patterns during different behaviors
print("=== Visualizing Spatial Patterns During Behaviors ===\n")

# Get a few different behavior examples from our sample
sample_behaviors = df_annot_sample.head(5)

fig, axes = plt.subplots(len(sample_behaviors), 1, figsize=(14, 4*len(sample_behaviors)))
if len(sample_behaviors) == 1:
    axes = [axes]

for idx, (_, behavior) in enumerate(sample_behaviors.iterrows()):
    start = behavior['start_frame']
    stop = behavior['stop_frame']
    action = behavior['action']
    agent = behavior['agent_id']
    target = behavior['target_id']
    
    # Add buffer
    plot_start = max(0, start - 50)
    plot_stop = min(len(df_wide_sample), stop + 50)
    
    # Get relevant distance column
    dist_col = f'dist_mouse{agent}_mouse{target}'
    
    ax = axes[idx]
    
    if dist_col in df_wide_sample.columns:
        # Plot distance over time
        frames = range(plot_start, plot_stop)
        distances = df_wide_sample.loc[plot_start:plot_stop-1, dist_col]
        
        ax.plot(frames, distances, label=f'Distance M{agent}-M{target}', linewidth=2)
        
        # Highlight the behavior period
        ax.axvspan(start, stop, alpha=0.3, color='red', label=f'Behavior: {action}')
        
        ax.set_xlabel('Frame', fontsize=11)
        ax.set_ylabel('Distance (pixels)', fontsize=11)
        ax.set_title(f'Distance Pattern: Mouse {agent} → {action} → Mouse {target}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'Distance data not available for {action}', 
                ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
plt.show()


# Calculate relative timing of behaviors within videos
print("=== Temporal Pattern Analysis ===\n")

# Merge annotations with metadata to get video durations
df_annot_temporal = pd.merge(
    df_annotations_full, 
    df_train_meta[['video_id', 'video_duration_sec', 'frames_per_second']], 
    on='video_id', 
    how='left'
)

# Calculate the relative position of each behavior within its video (0 = start, 1 = end)
df_annot_temporal['total_frames'] = df_annot_temporal['video_duration_sec'] * df_annot_temporal['frames_per_second']
df_annot_temporal['behavior_midpoint'] = (df_annot_temporal['start_frame'] + df_annot_temporal['stop_frame']) / 2
df_annot_temporal['relative_position'] = df_annot_temporal['behavior_midpoint'] / df_annot_temporal['total_frames']

# Ensure relative position is between 0 and 1
df_annot_temporal['relative_position'] = df_annot_temporal['relative_position'].clip(0, 1)

print(f"Temporal features calculated for {len(df_annot_temporal)} behaviors\n")


# Visualize when different behaviors occur in videos
plt.figure(figsize=(14, 10))

# Get top 15 most common behaviors for readability
top_behaviors = df_annot_temporal['action'].value_counts().head(15).index

df_plot = df_annot_temporal[df_annot_temporal['action'].isin(top_behaviors)]

sns.violinplot(
    data=df_plot, 
    y='action', 
    x='relative_position',
    order=top_behaviors,
    palette='coolwarm',
    inner='box'
)

plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Video Midpoint')
plt.xlabel('Relative Position in Video (0=Start, 1=End)', fontsize=12)
plt.ylabel('Behavior', fontsize=12)
plt.title('When Do Behaviors Occur? (Distribution Across Video Timeline)', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

# Statistical summary
print("\nBehaviors occurring predominantly in first half of videos:")
early_behaviors = df_annot_temporal.groupby('action')['relative_position'].median().sort_values().head(5)
print(early_behaviors)

print("\nBehaviors occurring predominantly in second half of videos:")
late_behaviors = df_annot_temporal.groupby('action')['relative_position'].median().sort_values(ascending=False).head(5)
print(late_behaviors)


# Analyze behavior transitions (what comes after what)
print("=== Behavior Transition Analysis ===\n")

# Sort annotations by video and frame
df_annot_sorted = df_annotations_full.sort_values(['video_id', 'start_frame']).reset_index(drop=True)

# Create a "next behavior" column
df_annot_sorted['next_action'] = df_annot_sorted.groupby('video_id')['action'].shift(-1)

# Remove last behavior in each video (no transition)
df_transitions = df_annot_sorted[df_annot_sorted['next_action'].notna()].copy()

print(f"Found {len(df_transitions)} behavior transitions\n")

# Get top behaviors for readable matrix
top_n = 12
top_behaviors = df_transitions['action'].value_counts().head(top_n).index

# Filter to only include top behaviors
df_trans_filtered = df_transitions[
    df_transitions['action'].isin(top_behaviors) & 
    df_transitions['next_action'].isin(top_behaviors)
]

# Create transition matrix
transition_matrix = pd.crosstab(
    df_trans_filtered['action'], 
    df_trans_filtered['next_action'],
    normalize='index'  # Normalize by row to get probabilities
) * 100  # Convert to percentage

print(f"Transition matrix shape: {transition_matrix.shape}")


# Visualize the transition matrix
plt.figure(figsize=(14, 12))

sns.heatmap(
    transition_matrix, 
    annot=True, 
    fmt='.1f', 
    cmap='YlOrRd', 
    cbar_kws={'label': 'Transition Probability (%)'},
    square=True,
    linewidths=0.5
)

plt.title('Behavior Transition Matrix\n(Row: Current Behavior → Column: Next Behavior)', fontsize=14)
plt.xlabel('Next Behavior', fontsize=12)
plt.ylabel('Current Behavior', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Find most common transitions
print("\n=== Most Common Behavior Transitions ===")
trans_counts = df_trans_filtered.groupby(['action', 'next_action']).size().sort_values(ascending=False)
print(trans_counts.head(10))


# Identify social vs individual behaviors
print("=== Social vs. Individual Behavior Analysis ===\n")

# A behavior is "individual" if agent_id == target_id
df_annotations_full['is_social'] = df_annotations_full['agent_id'] != df_annotations_full['target_id']

# Count by behavior type
behavior_types = df_annotations_full.groupby('action')['is_social'].agg(['sum', 'count'])
behavior_types['pct_social'] = (behavior_types['sum'] / behavior_types['count']) * 100
behavior_types = behavior_types.sort_values('pct_social', ascending=False)

print("Behavior classification (% that are social interactions):")
print(behavior_types)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Overall distribution
social_counts = df_annotations_full['is_social'].value_counts()
ax1.pie(social_counts, labels=['Individual', 'Social'], autopct='%1.1f%%', 
        colors=['skyblue', 'salmon'], startangle=90)
ax1.set_title('Overall Distribution: Social vs. Individual Behaviors', fontsize=14)

# Plot 2: By behavior type
top_20_behaviors = behavior_types.head(20)
ax2.barh(range(len(top_20_behaviors)), top_20_behaviors['pct_social'], color='coral')
ax2.set_yticks(range(len(top_20_behaviors)))
ax2.set_yticklabels(top_20_behaviors.index)
ax2.set_xlabel('% Social Interactions', fontsize=12)
ax2.set_title('Top 20 Behaviors: Social Interaction Percentage', fontsize=14)
ax2.axvline(x=50, color='black', linestyle='--', alpha=0.5)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()


# Analyze arena diversity
print("=== Experimental Setup Diversity ===\n")

# First, let's check the actual column names
print("Available columns:")
print([col for col in df_train_meta.columns if 'arena' in col.lower() or 'video' in col.lower()])
print()

# Arena types and shapes
print("Arena shapes:")
print(df_train_meta['arena_shape'].value_counts())

print("\nArena types:")
print(df_train_meta['arena_type'].value_counts())

print("\nTracking methods:")
print(df_train_meta['tracking_method'].value_counts())

# Visualize arena sizes and experimental setup diversity
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Arena dimensions
axes[0, 0].scatter(df_train_meta['arena_width_cm'], df_train_meta['arena_height_cm'], 
                   alpha=0.5, c='steelblue')
axes[0, 0].set_xlabel('Arena Width (cm)', fontsize=11)
axes[0, 0].set_ylabel('Arena Height (cm)', fontsize=11)
axes[0, 0].set_title('Arena Dimensions Distribution', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)

# Video resolutions - CORRECT COLUMN NAMES with '_pix' suffix
axes[0, 1].scatter(df_train_meta['video_width_pix'], df_train_meta['video_height_pix'], 
                   alpha=0.5, c='coral')
axes[0, 1].set_xlabel('Video Width (pixels)', fontsize=11)
axes[0, 1].set_ylabel('Video Height (pixels)', fontsize=11)
axes[0, 1].set_title('Video Resolution Distribution', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# FPS distribution
axes[1, 0].hist(df_train_meta['frames_per_second'].dropna(), bins=30, 
                color='mediumseagreen', edgecolor='black')
axes[1, 0].set_xlabel('Frames Per Second', fontsize=11)
axes[1, 0].set_ylabel('Count', fontsize=11)
axes[1, 0].set_title('Frame Rate Distribution', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# Pixels per cm (scale factor)
axes[1, 1].hist(df_train_meta['pix_per_cm_approx'].dropna(), bins=30, 
                color='orchid', edgecolor='black')
axes[1, 1].set_xlabel('Pixels per CM', fontsize=11)
axes[1, 1].set_ylabel('Count', fontsize=11)
axes[1, 1].set_title('Scale Factor Distribution', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Generate comprehensive dataset summary
print("="*70)
print("COMPREHENSIVE DATASET SUMMARY")
print("="*70)

summary = {
    'Total Training Videos': len(df_train_meta),
    'Total Test Videos (Public)': len(df_test_meta),
    'Unique Labs': df_train_meta['lab_id'].nunique(),
    'Total Annotated Behaviors': len(df_annotations_full),
    'Unique Behavior Types': df_annotations_full['action'].nunique(),
    'Date Range (approx frames)': f"{df_annotations_full['start_frame'].min()} to {df_annotations_full['stop_frame'].max()}",
    'Avg Behaviors per Video': len(df_annotations_full) / len(df_train_meta),
    'Social Behaviors (%)': (df_annotations_full['is_social'].sum() / len(df_annotations_full)) * 100,
}

for key, value in summary.items():
    print(f"{key:.<50} {value}")

print("\n" + "="*70)
print("KEY CHALLENGES IDENTIFIED")
print("="*70)

challenges = [
    "1. EXTREME CLASS IMBALANCE - Top behavior 1000x more common than rarest",
    "2. LAB GENERALIZATION - Must use GroupKFold with lab_id",
    "3. MISSING DATA - 5-20% tracking failures, varies by lab",
    "4. VARIABLE DURATIONS - Behaviors range from 1 frame to 1000+ frames",
    "5. MULTI-SCALE PROBLEM - Need features at frame, sequence, and video level",
    "6. SETUP VARIABILITY - Different arenas, resolutions, FPS across labs"
]

for challenge in challenges:
    print(f"  {challenge}")

print("\n" + "="*70)
print("ACTIONABLE MODELING INSIGHTS")
print("="*70)

insights = [
    "✓ Feature Engineering: Focus on normalized spatial features (distances, angles)",
    "✓ Temporal Context: Use sequence models (LSTM/Transformer) with window size 30-100 frames",
    "✓ Class Balance: Apply class weights, focal loss, or oversampling for rare behaviors",
    "✓ Validation: GroupKFold on lab_id is MANDATORY for realistic evaluation",
    "✓ Missing Data: Implement forward-fill + interpolation for tracking gaps",
    "✓ Multi-Scale: Consider ensemble of frame-level + sequence-level models",
    "✓ Normalization: Convert to real-world units (cm, cm/s) using metadata"
]

for insight in insights:
    print(f"  {insight}")

print("\n" + "="*70)


# Create a final behavior reference table
print("\n=== BEHAVIOR REFERENCE TABLE ===\n")

behavior_summary = df_annotations_full.groupby('action').agg({
    'action': 'count',
    'duration_frames': ['median', 'mean', 'std'],
    'is_social': lambda x: (x.sum() / len(x)) * 100
}).round(2)

behavior_summary.columns = ['Count', 'Median_Duration', 'Mean_Duration', 'Std_Duration', 'Pct_Social']
behavior_summary = behavior_summary.sort_values('Count', ascending=False)

print("Top 15 Most Common Behaviors:")
display(behavior_summary.head(15))

print("\nRarest Behaviors (Bottom 10):")
display(behavior_summary.tail(10))

# Save for future reference
print("\n✓ Summary statistics calculated and ready for modeling phase!")

