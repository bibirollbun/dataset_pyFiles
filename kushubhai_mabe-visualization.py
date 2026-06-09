import pandas as pd
import numpy as np
import polars as pl
import json
import os
import gc
import itertools
from collections import defaultdict
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from IPython.display import HTML
pd.options.display.max_columns = 100
tqdm.pandas()
BASE_PATH = "/kaggle/input/MABe-mouse-behavior-detection/"
TRAIN_TRACKING_DIR = os.path.join(BASE_PATH, "train_tracking")
TEST_TRACKING_DIR = os.path.join(BASE_PATH, "test_tracking")
ANNOTATION_DIR = os.path.join(BASE_PATH, "train_annotation")
train_df = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
sample_submission_df = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))
print("ENVIRONMENT SETUP COMPLETE!")


# train_df.head()


# train_df.shape


train_df = train_df[~train_df['lab_id'].str.startswith('MABe22_')].reset_index(drop=True)


# fig, axes = plt.subplots(2, 1, figsize = (16, 14))
# lab_counts = train_df["lab_id"].value_counts()
# sns.barplot(x = lab_counts.index, y = lab_counts.values, ax = axes[0], palette = "plasma")
# axes[0].set_title("Lab_id wise distribution of Videos", fontsize = 18)
# axes[0].set_xlabel("Lab IDs")
# axes[0].set_ylabel("No. of Videos")
# axes[0].tick_params(axis = "x", rotation = 45)

# body_part_counts = train_df["body_parts_tracked"].apply(lambda x: f"{len(json.loads(x))} parts").value_counts()
# sns.barplot(x = body_part_counts.index, y = body_part_counts.values, ax = axes[1], palette = "magma")
# axes[0].set_title("Body Parts wise distribution of Videos", fontsize = 18)
# axes[0].set_xlabel("No. of Body Parts")
# axes[0].set_ylabel("No. of Videos")
# axes[0].tick_params(axis = "x", rotation = 45)


# print("################################################################")
# for i in (train_df['body_parts_tracked'].unique()):
#     print(i)
#     print("################################################################")


# behaviour_durations = defaultdict(list)
# DIFFERENT_BEHAVIOURS = {"chase", "mount", "attack", "approach", "avoid", "rear"}

# print("Opening the PARQUET files")

# for i, row in tqdm(train_df.head(50).iterrows(), total=50):
#     annotation_path = os.path.join(ANNOTATION_DIR, row['lab_id'], f"{row['video_id']}.parquet")
#     print(annotation_path)
#     try:
#         annotation_df = pd.read_parquet(annotation_path)
#         print(annotation_df)
#         print(annotation_df.shape)
#         print(annotation_df["action"].value_counts())
#         # if not annotation_df.empty:
#         #     durations = annotation_df['stop_frame'] - annotation_df['start_frame']
#         #     for action, duration in zip(annotation_df['action'], durations):
#         #         behavior_durations[action].append(duration)
                
#     except FileNotFoundError:
#         # This is a graceful way to handle videos that have tracking data but no annotations.
#         # Our pipeline must not fail if an annotation file is missing.
#         # print(f"HANDLED ERROR: Annotation file not found for video {row['video_id']}. Skipping.")
#         pass
#     if i == 0:
#         break


# Define your desired frame range here --
START_FRAME = 0
END_FRAME = 100
# Note : Change the train_tracking_path for the parquet file you want to check out

coords_1, coords_2, coords_3, coords_4 = [], [], [], []

# This loop finds and processes the first available parquet file
# It breaks after one successful iteration.
for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
    print(f"Loading the PARQUET File...")
    train_tracking_path = os.path.join(TRAIN_TRACKING_DIR, row['lab_id'], f"{row['video_id']}.parquet")
    
    try:
        train_tracking_df = pd.read_parquet(train_tracking_path)
        print(f"Successfully loaded: {train_tracking_path}")
        print(f"Original shape: {train_tracking_df.shape}")

        filtered_df = train_tracking_df[
            (train_tracking_df["video_frame"] >= START_FRAME) & 
            (train_tracking_df["video_frame"] < END_FRAME)
        ]
        print(f"Shape after filtering for frames {START_FRAME}-{END_FRAME}: {filtered_df.shape}")

        for j in range(START_FRAME, END_FRAME):
            df_1 = filtered_df[(filtered_df["video_frame"] == j) & (filtered_df["mouse_id"] == 1)]
            df_2 = filtered_df[(filtered_df["video_frame"] == j) & (filtered_df["mouse_id"] == 2)]
            df_3 = filtered_df[(filtered_df["video_frame"] == j) & (filtered_df["mouse_id"] == 3)]
            df_4 = filtered_df[(filtered_df["video_frame"] == j) & (filtered_df["mouse_id"] == 4)]

            coords_1.append(df_1.loc[:, ["x", "y"]])
            coords_2.append(df_2.loc[:, ["x", "y"]])
            coords_3.append(df_3.loc[:, ["x", "y"]])
            coords_4.append(df_4.loc[:, ["x", "y"]])

        break
            
    except FileNotFoundError:
        print(f"File not found for video {row['video_id']}. Searching for next file.")
        pass


print(len(coords_1))


# print(df_1)
# print(df_2)
# print(df_3)
# print(df_4)


fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_title("DataFrame Animation")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_xlim(0, 1200)
ax.set_ylim(0, 1200)

scatter_1 = ax.scatter(coords_1[0]['x'], coords_1[0]['y'], s=50, c='red', label='Mouse 1')
scatter_2 = ax.scatter(coords_2[0]['x'], coords_2[0]['y'], s=50, c='blue', label='Mouse 2')
scatter_3 = ax.scatter(coords_3[0]['x'], coords_3[0]['y'], s=50, c='green', label='Mouse 3')
scatter_4 = ax.scatter(coords_4[0]['x'], coords_4[0]['y'], s=50, c='purple', label='Mouse 4')

ax.legend(loc='upper right')

def update(frame):
    """Updates the positions of all four scatter plots for each frame."""
    
    # Update Mouse 1
    new_pos_1 = coords_1[frame][['x', 'y']].values
    scatter_1.set_offsets(new_pos_1)
    
    # Update Mouse 2
    new_pos_2 = coords_2[frame][['x', 'y']].values
    scatter_2.set_offsets(new_pos_2)
    
    # Update Mouse 3
    new_pos_3 = coords_3[frame][['x', 'y']].values
    scatter_3.set_offsets(new_pos_3)
    
    # Update Mouse 4
    new_pos_4 = coords_4[frame][['x', 'y']].values
    scatter_4.set_offsets(new_pos_4)
    
    return scatter_1, scatter_2, scatter_3, scatter_4

ani = animation.FuncAnimation(
    fig=fig, 
    func=update, 
    frames=len(coords_1),
    interval=200,
    blit=True
)

HTML(ani.to_html5_video())




