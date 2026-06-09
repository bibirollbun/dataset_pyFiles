import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
from tqdm.auto import tqdm
import os


# Define the path to your data
DATA_PATH = '/kaggle/input/MABe-mouse-behavior-detection/'

# Load the metadata files
print("Loading train.csv (metadata)...")
df_train_meta = pd.read_csv(DATA_PATH + 'train.csv')

print("\n--- Train Metadata ---")
print(f"Shape: {df_train_meta.shape}")
display(df_train_meta.head())




# This cell is designed to be self-contained. It will build the full annotation
# dataframe if it doesn't already exist in memory.  (Copied directly from [1.Full EDA]('https://www.kaggle.com/code/wafaaalayoubi/1-full-eda'))
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


df_annotations_full.head()



# Check if, for each video and agent, the start/stop frames of actions overlap with different actions
def check_action_frame_overlap(df):
    overlap_found = False
    grouped = df.groupby(['video_id', 'agent_id'])
    for (video_id, agent_id), group in grouped:
        # Sort by start_frame and include action information
        sorted_group = group.sort_values('start_frame')
        intervals = sorted_group[['start_frame', 'stop_frame', 'action']].values
        
        for i in range(1, len(intervals)):
            prev_start, prev_stop, prev_action = intervals[i-1]
            curr_start, curr_stop, curr_action = intervals[i]
            
            # Check for overlap AND different actions
            if curr_start < prev_stop and prev_action != curr_action:
                print(f"Overlap found in video {video_id}, agent {agent_id}: "
                      f"'{prev_action}' (frames {prev_start}-{prev_stop}) overlaps with "
                      f"'{curr_action}' (frames {curr_start}-{curr_stop})")
                overlap_found = True
    
    if not overlap_found:
        print("✅ No overlapping frames found between different actions for any agent in any video.")
    return not overlap_found

check_action_frame_overlap(df_annotations_full)


# Filter for video 44566106 and agent 1, and check for the specified frame intervals
video_id = 44566106
agent_id = 1

mask = (
    (df_annotations_full['video_id'] == video_id) &
    (df_annotations_full['agent_id'] == agent_id) &
    (
        ((df_annotations_full['start_frame'] == 15907) & (df_annotations_full['stop_frame'] == 15957)) |
        ((df_annotations_full['start_frame'] == 15930) & (df_annotations_full['stop_frame'] == 15953))
    )
)

df_annotations_full[mask]


# Count the number of apparent conflicts (overlapping intervals with different actions)
conflict_count = 0
total_behaviors = 0
conflict_frames = 0

grouped = df_annotations_full.groupby(['video_id', 'agent_id'])
for (video_id, agent_id), group in grouped:
    sorted_group = group.sort_values('start_frame')
    intervals = sorted_group[['start_frame', 'stop_frame', 'action']].values
    for i in range(1, len(intervals)):
        prev_start, prev_stop, prev_action = intervals[i-1]
        curr_start, curr_stop, curr_action = intervals[i]
        if curr_start < prev_stop and prev_action != curr_action:
            conflict_count += 1
            # Calculate overlapping frames
            overlap_start = curr_start
            overlap_stop = min(prev_stop, curr_stop)
            conflict_frames += overlap_stop - overlap_start + 1
        total_behaviors += 1

total_frames = (df_annotations_full['stop_frame'] - df_annotations_full['start_frame'] + 1).sum()

print(f"Total apparent conflicts: {conflict_count}, total behaviors: {total_behaviors}, conflict fraction: {conflict_count / total_behaviors:.6f}  ")
print(f"Total frames in conflicts: {conflict_frames}, total annotated frames: {total_frames} conflict fraction: {conflict_frames / total_frames:.6f}")


import numpy as np
import pandas as pd

# Create a matrix of action conflicts (rows: previous action, columns: current action)
actions = df_annotations_full['action'].unique()
conflict_matrix = pd.DataFrame(0, index=actions, columns=actions)

grouped = df_annotations_full.groupby(['video_id', 'agent_id'])
for (video_id, agent_id), group in grouped:
    sorted_group = group.sort_values('start_frame')
    intervals = sorted_group[['start_frame', 'stop_frame', 'action']].values
    for i in range(1, len(intervals)):
        prev_start, prev_stop, prev_action = intervals[i-1]
        curr_start, curr_stop, curr_action = intervals[i]
        if curr_start < prev_stop and prev_action != curr_action:
            conflict_matrix.loc[prev_action, curr_action] += 1



# Drop any action (row or column) with less than 5 total conflicts from filtered_conflict_matrix
conflict_sums =conflict_matrix.sum(axis=1) + conflict_matrix.sum(axis=0)
actions_to_keep = conflict_sums[conflict_sums >= 5].index
filtered_conflict_matrix =  conflict_matrix.loc[actions_to_keep, actions_to_keep]
filtered_conflict_matrix




sns.heatmap(filtered_conflict_matrix, cmap='Blues', annot=True)

