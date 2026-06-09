import pandas as pd
import numpy as np
import polars as pl
import json
import os
import gc
import itertools
from functools import reduce
from collections import defaultdict
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from IPython.display import HTML
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
tqdm.pandas()
BASE_PATH = "/kaggle/input/MABe-mouse-behavior-detection/"
TRAIN_TRACKING_DIR = os.path.join(BASE_PATH, "train_tracking")
TEST_TRACKING_DIR = os.path.join(BASE_PATH, "test_tracking")
ANNOTATION_DIR = os.path.join(BASE_PATH, "train_annotation")
train_df = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
sample_submission_df = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))

OUTPUT_PATH = "/kaggle/working/"
print("ENVIRONMENT SETUP COMPLETE!")


train_df = train_df[~train_df['lab_id'].str.startswith('MABe22_')].reset_index(drop=True)


train_df["lab_id"].value_counts()


def process_mouse_dataframe(df, bodyparts):
    """Pivots, cleans, and vectorizes a dataframe for a single mouse."""
    
    pivoted_df = df.pivot_table(
        index='video_frame',
        columns='bodypart',
        values=['x', 'y']
    )

    pivoted_df.columns = [f'{col[1]}_{col[0]}' for col in pivoted_df.columns]
    pivoted_df = pivoted_df.reset_index()

    return pivoted_df

for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    train_tracking_path = os.path.join(TRAIN_TRACKING_DIR, row['lab_id'], f"{row['video_id']}.parquet")
    FINAL_OUTPUT_PATH = os.path.join(OUTPUT_PATH, row["lab_id"])
    os.makedirs(FINAL_OUTPUT_PATH, exist_ok=True)
    try:
        train_tracking_df = pd.read_parquet(train_tracking_path)
        
        # Find common bodyparts across all mice present in this file
        mouse_bodyparts_combined = [
            group['bodypart'].unique() for _, group in train_tracking_df.groupby('mouse_id')
        ]
        if not mouse_bodyparts_combined:
            continue # Skip file if no data
        common_bodyparts = reduce(np.intersect1d, mouse_bodyparts_combined)
        
        # Dynamically process for each mouse in the file
        for mouse_id, mouse_df in train_tracking_df.groupby('mouse_id'):
            MOUSE_OUTPUT_PATH = os.path.join(FINAL_OUTPUT_PATH, f"Mouse_{mouse_id}")
            os.makedirs(MOUSE_OUTPUT_PATH, exist_ok=True)
            if mouse_id in [1,2,3,4]:
                # Call our single processing function
                processed_df = process_mouse_dataframe(mouse_df, common_bodyparts)
                parquet_file_format = f"{row['video_id']}_new.parquet"
                parquet_file_path = os.path.join(MOUSE_OUTPUT_PATH, parquet_file_format)
                processed_df.to_parquet(parquet_file_path)

    except FileNotFoundError:
        print(f"File not found for video {row['video_id']}. Skipping.")
        pass




