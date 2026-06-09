# MABe Mouse Behavior Detection - Data Reading and EDA
# This notebook provides comprehensive exploratory data analysis of the MABe mouse behavior dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# Define data paths
DATA_DIR = "/kaggle/input/MABe-mouse-behavior-detection/"
TRAIN_CSV = DATA_DIR + "train.csv"
TEST_CSV = DATA_DIR + "test.csv"
SAMPLE_SUB = DATA_DIR + "sample_submission.csv"
TRAIN_TRACKING = DATA_DIR + "train_tracking/"
TRAIN_ANNOTATION = DATA_DIR + "train_annotation/"
TEST_TRACKING = DATA_DIR + "test_tracking/"


train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

print(f"ğŸ“ˆ Training videos: {len(train_df)}")
print(f"ğŸ“ˆ Test videos: {len(test_df)}")
print(f"ğŸ“ˆ Sample submission rows: {len(sample_sub)}")
print(f"ğŸ“ˆ Total unique labs: {train_df['lab_id'].nunique()}")
print(f"ğŸ“ˆ Total training video duration: {train_df['video_duration_sec'].sum()/3600:.1f} hours")
print(f"ğŸ“ˆ Total test video duration: {test_df['video_duration_sec'].sum()/3600:.1f} hours")


# Display basic info about training data
print("ğŸ”� Training Data Structure:")
print(train_df.info())
print("\n" + "="*50)
print("ğŸ“‹ Column Names:")
for i, col in enumerate(train_df.columns):
    print(f"{i+1:2d}. {col}")
    
print("\n" + "="*50)
print("ğŸ“Š First Few Rows:")
train_df.head(3)


# Laboratory distribution
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Lab distribution
lab_counts = train_df['lab_id'].value_counts()
ax1 = axes[0, 0]
lab_counts.plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('ğŸ�¢ Distribution of Videos by Laboratory', fontsize=14, fontweight='bold')
ax1.set_xlabel('Laboratory ID')
ax1.set_ylabel('Number of Videos')
ax1.tick_params(axis='x', rotation=45)

# Video duration distribution
ax2 = axes[0, 1]
train_df['video_duration_sec'].hist(bins=50, ax=ax2, color='lightcoral', alpha=0.7)
ax2.set_title('â�±ï¸� Video Duration Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Duration (seconds)')
ax2.set_ylabel('Frequency')

# Arena shape distribution
ax3 = axes[1, 0]
arena_counts = train_df['arena_shape'].value_counts()
arena_counts.plot(kind='pie', ax=ax3, autopct='%1.1f%%', colors=['lightgreen', 'orange'])
ax3.set_title('ğŸ�Ÿï¸� Arena Shape Distribution', fontsize=14, fontweight='bold')

# Frames per second distribution
ax4 = axes[1, 1]
fps_counts = train_df['frames_per_second'].value_counts()
fps_counts.plot(kind='bar', ax=ax4, color='gold')
ax4.set_title('ğŸ“¹ Frames Per Second Distribution', fontsize=14, fontweight='bold')
ax4.set_xlabel('FPS')
ax4.set_ylabel('Number of Videos')

plt.tight_layout()
plt.show()

print(f"ğŸ“Š Laboratory Statistics:")
print(f"   â€¢ Total laboratories: {train_df['lab_id'].nunique()}")
print(f"   â€¢ Videos per lab (avg): {len(train_df) / train_df['lab_id'].nunique():.1f}")
print(f"   â€¢ Most active lab: {lab_counts.index[0]} ({lab_counts.iloc[0]} videos)")
print(f"   â€¢ Least active lab: {lab_counts.index[-1]} ({lab_counts.iloc[-1]} videos)")

print(f"\nğŸ“¹ Video Statistics:")
print(f"   â€¢ Average duration: {train_df['video_duration_sec'].mean():.1f} seconds")
print(f"   â€¢ Duration range: {train_df['video_duration_sec'].min():.1f} - {train_df['video_duration_sec'].max():.1f} seconds")
print(f"   â€¢ Most common FPS: {train_df['frames_per_second'].mode().iloc[0]}")
print(f"   â€¢ Arena shapes: {', '.join(train_df['arena_shape'].unique())}")


# Analyze mouse characteristics
def get_mouse_data(df, mouse_num):
    """Extract mouse data for analysis"""
    return df[[f'mouse{mouse_num}_strain', f'mouse{mouse_num}_color', 
               f'mouse{mouse_num}_sex', f'mouse{mouse_num}_age', 
               f'mouse{mouse_num}_condition']].dropna()

# Combine all mouse data
all_mice_data = []
for i in range(1, 5):  # mice 1-4
    mouse_data = get_mouse_data(train_df, i)
    mouse_data.columns = ['strain', 'color', 'sex', 'age', 'condition']
    mouse_data['mouse_number'] = i
    all_mice_data.append(mouse_data)

mice_df = pd.concat(all_mice_data, ignore_index=True)

# Create comprehensive mouse analysis plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Mouse strain distribution
ax1 = axes[0, 0]
strain_counts = mice_df['strain'].value_counts()
strain_counts.plot(kind='bar', ax=ax1, color='lightblue')
ax1.set_title('ğŸ§¬ Mouse Strain Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Strain')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)

# Mouse color distribution
ax2 = axes[0, 1]
color_counts = mice_df['color'].value_counts()
colors_map = {'white': 'white', 'black': 'black', 'brown': 'brown', 'gray': 'gray'}
plot_colors = [colors_map.get(color, 'blue') for color in color_counts.index]
color_counts.plot(kind='bar', ax=ax2, color=plot_colors, edgecolor='black')
ax2.set_title('ğŸ�¨ Mouse Color Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Color')
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', rotation=45)

# Mouse sex distribution
ax3 = axes[0, 2]
sex_counts = mice_df['sex'].value_counts()
sex_counts.plot(kind='pie', ax=ax3, autopct='%1.1f%%', colors=['lightpink', 'lightblue'])
ax3.set_title('âš¥ Mouse Sex Distribution', fontsize=14, fontweight='bold')

# Age distribution
ax4 = axes[1, 0]
age_counts = mice_df['age'].value_counts()
age_counts.plot(kind='bar', ax=ax4, color='orange')
ax4.set_title('ğŸ“… Mouse Age Distribution', fontsize=14, fontweight='bold')
ax4.set_xlabel('Age')
ax4.set_ylabel('Count')
ax4.tick_params(axis='x', rotation=45)

# Condition distribution
ax5 = axes[1, 1]
condition_counts = mice_df['condition'].value_counts()
condition_counts.plot(kind='bar', ax=ax5, color='lightgreen')
ax5.set_title('ğŸ”¬ Mouse Condition Distribution', fontsize=14, fontweight='bold')
ax5.set_xlabel('Condition')
ax5.set_ylabel('Count')
ax5.tick_params(axis='x', rotation=45)

# Mouse number distribution
ax6 = axes[1, 2]
mouse_num_counts = mice_df['mouse_number'].value_counts().sort_index()
mouse_num_counts.plot(kind='bar', ax=ax6, color='purple')
ax6.set_title('ğŸ�­ Mouse Position Distribution', fontsize=14, fontweight='bold')
ax6.set_xlabel('Mouse Number')
ax6.set_ylabel('Count')

plt.tight_layout()
plt.show()

print("ğŸ�­ Mouse Characteristics Summary:")
print(f"   â€¢ Total mouse records: {len(mice_df):,}")
print(f"   â€¢ Unique strains: {mice_df['strain'].nunique()}")
print(f"   â€¢ Most common strain: {strain_counts.index[0]} ({strain_counts.iloc[0]:,} mice)")
print(f"   â€¢ Color distribution: {dict(color_counts)}")
print(f"   â€¢ Sex distribution: {dict(sex_counts)}")
print(f"   â€¢ Age groups: {mice_df['age'].nunique()}")
print(f"   â€¢ Experimental conditions: {mice_df['condition'].nunique()}")


# Parse behavior labels
import ast

def parse_behaviors(behavior_str):
    """Parse behavior string into list of behaviors"""
    try:
        return ast.literal_eval(behavior_str)
    except:
        return []

# Get all behaviors from training data
all_behaviors = []
behavior_types = {}

for idx, row in train_df.iterrows():
    behaviors = parse_behaviors(row['behaviors_labeled'])
    all_behaviors.extend(behaviors)
    
    # Extract behavior types
    for behavior in behaviors:
        if ',' in behavior:
            parts = behavior.split(',')
            if len(parts) == 3:
                agent, target, action = parts
                if action not in behavior_types:
                    behavior_types[action] = 0
                behavior_types[action] += 1

# Count behavior frequencies
behavior_counts = pd.Series(all_behaviors).value_counts()

print("ğŸ�­ Behavior Analysis:")
print(f"   â€¢ Total behavior instances: {len(all_behaviors):,}")
print(f"   â€¢ Unique behaviors: {len(behavior_counts)}")
print(f"   â€¢ Unique behavior types: {len(behavior_types)}")

# Plot behavior types
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Behavior types distribution
ax1 = axes[0, 0]
behavior_type_series = pd.Series(behavior_types)
behavior_type_series.sort_values(ascending=False).plot(kind='bar', ax=ax1, color='coral')
ax1.set_title('ğŸ�­ Behavior Types Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Behavior Type')
ax1.set_ylabel('Frequency')
ax1.tick_params(axis='x', rotation=45)

# Top 20 most common specific behaviors
ax2 = axes[0, 1]
top_behaviors = behavior_counts.head(20)
top_behaviors.plot(kind='barh', ax=ax2, color='lightgreen')
ax2.set_title('ğŸ”� Top 20 Most Common Behaviors', fontsize=14, fontweight='bold')
ax2.set_xlabel('Frequency')

# Behavior complexity (number of behaviors per video)
ax3 = axes[1, 0]
behaviors_per_video = []
for idx, row in train_df.iterrows():
    behaviors = parse_behaviors(row['behaviors_labeled'])
    behaviors_per_video.append(len(behaviors))

pd.Series(behaviors_per_video).hist(bins=30, ax=ax3, color='skyblue', alpha=0.7)
ax3.set_title('ğŸ“Š Behaviors Per Video Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Number of Behaviors')
ax3.set_ylabel('Frequency')

# Arena type vs behavior complexity
ax4 = axes[1, 1]
train_df['num_behaviors'] = behaviors_per_video
arena_behavior = train_df.groupby('arena_type')['num_behaviors'].mean()
arena_behavior.plot(kind='bar', ax=ax4, color='gold')
ax4.set_title('ğŸ�Ÿï¸� Avg Behaviors by Arena Type', fontsize=14, fontweight='bold')
ax4.set_xlabel('Arena Type')
ax4.set_ylabel('Average # Behaviors')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print(f"\nğŸ�­ Top 10 Behavior Types:")
for i, (behavior, count) in enumerate(behavior_type_series.sort_values(ascending=False).head(10).items()):
    print(f"   {i+1:2d}. {behavior}: {count:,}")

print(f"\nğŸ“Š Behavior Complexity:")
print(f"   â€¢ Average behaviors per video: {np.mean(behaviors_per_video):.1f}")
print(f"   â€¢ Max behaviors in a video: {max(behaviors_per_video)}")
print(f"   â€¢ Min behaviors in a video: {min(behaviors_per_video)}")
print(f"   â€¢ Videos with no behaviors: {behaviors_per_video.count(0)}")


# Analyze body parts tracking
def parse_body_parts(body_parts_str):
    """Parse body parts string into list"""
    try:
        return ast.literal_eval(body_parts_str)
    except:
        return []

# Get all body parts
all_body_parts = []
for idx, row in train_df.iterrows():
    body_parts = parse_body_parts(row['body_parts_tracked'])
    all_body_parts.extend(body_parts)

body_parts_counts = pd.Series(all_body_parts).value_counts()

# Analyze tracking methods
tracking_methods = train_df['tracking_method'].value_counts()

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Body parts frequency
ax1 = axes[0, 0]
body_parts_counts.plot(kind='barh', ax=ax1, color='lightblue')
ax1.set_title('ğŸ�¯ Body Parts Tracking Frequency', fontsize=14, fontweight='bold')
ax1.set_xlabel('Frequency')

# Tracking methods
ax2 = axes[0, 1]
tracking_methods.plot(kind='pie', ax=ax2, autopct='%1.1f%%', colors=['lightcoral', 'gold'])
ax2.set_title('ğŸ“± Tracking Methods Distribution', fontsize=14, fontweight='bold')

# Number of body parts per video
ax3 = axes[1, 0]
body_parts_per_video = []
for idx, row in train_df.iterrows():
    body_parts = parse_body_parts(row['body_parts_tracked'])
    body_parts_per_video.append(len(body_parts))

pd.Series(body_parts_per_video).hist(bins=20, ax=ax3, color='lightgreen', alpha=0.7)
ax3.set_title('ğŸ“Š Body Parts Per Video Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Number of Body Parts')
ax3.set_ylabel('Frequency')

# Video resolution distribution
ax4 = axes[1, 1]
train_df['resolution'] = train_df['video_width_pix'].astype(str) + 'x' + train_df['video_height_pix'].astype(str)
resolution_counts = train_df['resolution'].value_counts().head(10)
resolution_counts.plot(kind='bar', ax=ax4, color='purple')
ax4.set_title('ğŸ“º Top Video Resolutions', fontsize=14, fontweight='bold')
ax4.set_xlabel('Resolution')
ax4.set_ylabel('Count')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("ğŸ�¯ Body Parts Analysis:")
print(f"   â€¢ Total body part instances: {len(all_body_parts):,}")
print(f"   â€¢ Unique body parts: {len(body_parts_counts)}")
print(f"   â€¢ Average body parts per video: {np.mean(body_parts_per_video):.1f}")
print(f"   â€¢ Most tracked body part: {body_parts_counts.index[0]} ({body_parts_counts.iloc[0]:,} times)")

print(f"\nğŸ“± Tracking Methods:")
for method, count in tracking_methods.items():
    print(f"   â€¢ {method}: {count:,} videos ({count/len(train_df)*100:.1f}%)")

print(f"\nğŸ�¯ Top 10 Body Parts:")
for i, (part, count) in enumerate(body_parts_counts.head(10).items()):
    print(f"   {i+1:2d}. {part}: {count:,}")

print(f"\nğŸ“º Video Specifications:")
print(f"   â€¢ Width range: {train_df['video_width_pix'].min()} - {train_df['video_width_pix'].max()} pixels")
print(f"   â€¢ Height range: {train_df['video_height_pix'].min()} - {train_df['video_height_pix'].max()} pixels")
print(f"   â€¢ Pixel density range: {train_df['pix_per_cm_approx'].min()} - {train_df['pix_per_cm_approx'].max()} pix/cm")


# Load and examine a sample tracking file
import os

# Get a sample tracking file
sample_lab = 'AdaptableSnail'
tracking_path = f"{TRAIN_TRACKING}{sample_lab}/"
tracking_files = os.listdir(tracking_path)
sample_file = [f for f in tracking_files if f.endswith('.parquet')][0]
sample_tracking_path = f"{tracking_path}{sample_file}"

print(f"ğŸ“� Loading sample tracking data: {sample_file}")
tracking_data = pd.read_parquet(sample_tracking_path)

print(f"ğŸ“Š Tracking Data Shape: {tracking_data.shape}")
print(f"ğŸ“‹ Columns: {list(tracking_data.columns)}")
print(f"ğŸ�¯ Body parts detected: {len([col for col in tracking_data.columns if '_x' in col])}")

# Display basic info
print("\n" + "="*50)
print("TRACKING DATA STRUCTURE")
print("="*50)
print(tracking_data.info())

# Show first few rows
print("\nğŸ“Š First 5 rows:")
tracking_data.head()


# Analyze tracking data structure
print("ğŸ”� Tracking Data Analysis:")

# Basic statistics
num_frames = tracking_data['video_frame'].nunique()
num_mice = tracking_data['mouse_id'].nunique()
num_bodyparts = tracking_data['bodypart'].nunique()

print(f"   â€¢ Total frames: {num_frames:,}")
print(f"   â€¢ Number of mice: {num_mice}")
print(f"   â€¢ Body parts tracked: {num_bodyparts}")
print(f"   â€¢ Total data points: {len(tracking_data):,}")
print(f"   â€¢ Data points per frame: {len(tracking_data) / num_frames:.1f}")

# Body parts in this video
bodyparts = tracking_data['bodypart'].unique()
print(f"\nğŸ�¯ Body parts in this video:")
for i, part in enumerate(sorted(bodyparts)):
    print(f"   {i+1:2d}. {part}")

# Mouse tracking
mouse_counts = tracking_data['mouse_id'].value_counts().sort_index()
print(f"\nğŸ�­ Mouse tracking distribution:")
for mouse_id, count in mouse_counts.items():
    print(f"   Mouse {mouse_id}: {count:,} data points")

# Coordinate ranges
print(f"\nğŸ“� Coordinate ranges:")
print(f"   â€¢ X coordinates: {tracking_data['x'].min():.1f} - {tracking_data['x'].max():.1f}")
print(f"   â€¢ Y coordinates: {tracking_data['y'].min():.1f} - {tracking_data['y'].max():.1f}")

# Create tracking visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Frame distribution
ax1 = axes[0, 0]
frame_counts = tracking_data.groupby('video_frame').size()
frame_counts.plot(ax=ax1, color='blue', alpha=0.7)
ax1.set_title('ğŸ“¹ Data Points per Frame', fontsize=14, fontweight='bold')
ax1.set_xlabel('Frame Number')
ax1.set_ylabel('Number of Data Points')

# Body part frequency
ax2 = axes[0, 1]
bodypart_counts = tracking_data['bodypart'].value_counts()
bodypart_counts.plot(kind='bar', ax=ax2, color='orange')
ax2.set_title('ğŸ�¯ Body Part Frequency', fontsize=14, fontweight='bold')
ax2.set_xlabel('Body Part')
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', rotation=45)

# Mouse movement (sample trajectory for mouse 1)
ax3 = axes[1, 0]
mouse1_data = tracking_data[tracking_data['mouse_id'] == 1]
nose_data = mouse1_data[mouse1_data['bodypart'] == 'nose'].head(100)  # First 100 frames
ax3.plot(nose_data['x'], nose_data['y'], 'b-', alpha=0.7, linewidth=2)
ax3.scatter(nose_data['x'].iloc[0], nose_data['y'].iloc[0], color='green', s=100, label='Start')
ax3.scatter(nose_data['x'].iloc[-1], nose_data['y'].iloc[-1], color='red', s=100, label='End')
ax3.set_title('ğŸ�­ Mouse 1 Nose Trajectory (First 100 frames)', fontsize=14, fontweight='bold')
ax3.set_xlabel('X Coordinate')
ax3.set_ylabel('Y Coordinate')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Coordinate distribution
ax4 = axes[1, 1]
ax4.scatter(tracking_data['x'], tracking_data['y'], alpha=0.1, s=1)
ax4.set_title('ğŸ“� All Coordinate Points', fontsize=14, fontweight='bold')
ax4.set_xlabel('X Coordinate')
ax4.set_ylabel('Y Coordinate')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Load and examine annotation data
annotation_path = f"{TRAIN_ANNOTATION}{sample_lab}/"
annotation_files = os.listdir(annotation_path)
sample_annotation_file = [f for f in annotation_files if f.endswith('.parquet')][0]
sample_annotation_path = f"{annotation_path}{sample_annotation_file}"

print(f"ğŸ“� Loading sample annotation data: {sample_annotation_file}")
annotation_data = pd.read_parquet(sample_annotation_path)

print(f"ğŸ“Š Annotation Data Shape: {annotation_data.shape}")
print(f"ğŸ“‹ Columns: {list(annotation_data.columns)}")

# Display basic info
print("\n" + "="*50)
print("ANNOTATION DATA STRUCTURE")
print("="*50)
print(annotation_data.info())

# Show first few rows
print("\nğŸ“Š First 10 rows:")
display(annotation_data.head(10))

# Analyze annotation patterns
print("\nğŸ�­ Behavior Annotation Analysis:")

# Get unique actions in this file (correct column name is 'action')
actions = annotation_data['action'].unique()
print(f"   â€¢ Actions in this file: {len(actions)}")

action_counts = annotation_data['action'].value_counts()
print(f"\nğŸ”� Action frequency in this video:")
for action, count in action_counts.head(10).items():
    print(f"   â€¢ {action}: {count}")

# Analyze frame ranges
print(f"\nğŸ“¹ Frame Analysis:")
print(f"   â€¢ Start frames range: {annotation_data['start_frame'].min()} - {annotation_data['start_frame'].max()}")
print(f"   â€¢ Stop frames range: {annotation_data['stop_frame'].min()} - {annotation_data['stop_frame'].max()}")
print(f"   â€¢ Average action duration: {(annotation_data['stop_frame'] - annotation_data['start_frame']).mean():.1f} frames")

# Agents and targets
print(f"\nğŸ�¯ Agent/Target Analysis:")
agents = annotation_data['agent_id'].unique()
targets = annotation_data['target_id'].unique()
print(f"   â€¢ Unique agents: {sorted(agents)}")
print(f"   â€¢ Unique targets: {sorted(targets)}")

# Visualize annotation data
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Action frequency
ax1 = axes[0, 0]
action_counts.plot(kind='bar', ax=ax1, color='lightcoral')
ax1.set_title('ğŸ�­ Action Frequency in Sample Video', fontsize=14, fontweight='bold')
ax1.set_xlabel('Action')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)

# Action duration distribution
ax2 = axes[0, 1]
durations = annotation_data['stop_frame'] - annotation_data['start_frame']
durations.hist(bins=30, ax=ax2, color='lightblue', alpha=0.7)
ax2.set_title('â�±ï¸� Action Duration Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Duration (frames)')
ax2.set_ylabel('Frequency')

# Timeline of actions
ax3 = axes[1, 0]
for i, (idx, row) in enumerate(annotation_data.head(20).iterrows()):
    ax3.barh(i, row['stop_frame'] - row['start_frame'], 
             left=row['start_frame'], alpha=0.7,
             label=row['action'] if i < 5 else "")
ax3.set_title('ğŸ“… Action Timeline (First 20 actions)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Frame Number')
ax3.set_ylabel('Action Instance')

# Agent-Target interaction matrix
ax4 = axes[1, 1]
interaction_matrix = annotation_data.groupby(['agent_id', 'target_id']).size().unstack(fill_value=0)
sns.heatmap(interaction_matrix, annot=True, fmt='d', ax=ax4, cmap='YlOrRd')
ax4.set_title('ğŸ¤� Agent-Target Interaction Matrix', fontsize=14, fontweight='bold')
ax4.set_xlabel('Target ID')
ax4.set_ylabel('Agent ID')

plt.tight_layout()
plt.show()

