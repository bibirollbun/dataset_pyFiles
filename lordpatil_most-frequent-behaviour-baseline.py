import pandas as pd
import os
import glob
from collections import Counter

BASE_PATH = '/kaggle/input/MABe-mouse-behavior-detection'

# Load metadata
train_meta = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
test_meta = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))

# Load all training annotations to find the most common action
annotation_files = glob.glob(os.path.join(BASE_PATH, 'train_annotation', '*', '*.parquet'))
all_actions = []

for file in annotation_files:
    df = pd.read_parquet(file)
    all_actions.extend(df['action'].tolist())

# Find the most frequent non-trivial action (ignore if 'no_action' were present, but it's not in annotations)
action_counter = Counter(all_actions)
most_common_action = action_counter.most_common(1)[0][0]
print(f"Most common action in training: {most_common_action}")

# Now generate submission for test set
submission_rows = []

for _, test_row in test_meta.iterrows():
    video_id = test_row['video_id']
    
    # Parse behaviors_labeled to get valid (agent, target, action_type) templates
    # Example: '["mouse1,mouse2,approach", "mouse2,mouse1,sniff"]'
    behaviors_str = test_row['behaviors_labeled']
    if pd.isna(behaviors_str) or behaviors_str == '[]':
        # Fallback: assume all pairwise interactions among existing mice
        mouse_cols = [col for col in test_meta.columns if col.startswith('mouse') and col.endswith('_id')]
        mouse_ids = [int(test_row[col]) for col in mouse_cols if not pd.isna(test_row[col])]
        # Generate all ordered pairs (including self)
        pairs = [(a, b) for a in mouse_ids for b in mouse_ids]
    else:
        # Parse the string safely
        import ast
        try:
            behavior_list = ast.literal_eval(behaviors_str)
        except:
            behavior_list = []
        pairs = []
        for item in behavior_list:
            if isinstance(item, str):
                parts = item.replace("'", "").split(',')
                if len(parts) == 3:
                    agent_str, target_str, _ = parts
                    # Map 'mouse1' -> actual mouse1_id, etc.
                    agent_key = f"{agent_str}_id"
                    target_key = f"{target_str}_id"
                    if agent_key in test_row and target_key in test_row:
                        aid = test_row[agent_key]
                        tid = test_row[target_key]
                        if not pd.isna(aid) and not pd.isna(tid):
                            pairs.append((int(aid), int(tid)))
        if not pairs:
            # Fallback to all mice
            mouse_cols = [col for col in test_meta.columns if col.startswith('mouse') and col.endswith('_id')]
            mouse_ids = [int(test_row[col]) for col in mouse_cols if not pd.isna(test_row[col])]
            pairs = [(a, b) for a in mouse_ids for b in mouse_ids]

    # Total frames = fps * duration
    total_frames = int(round(test_row['frames_per_second'] * test_row['video_duration_sec']))

    # For each valid pair, output one long segment with the most common action
    for agent_id, target_id in pairs:
        # Convert numeric IDs back to 'mouse1', 'mouse2', etc.
        agent_str = None
        target_str = None
        for i in range(1, 5):
            if f'mouse{i}_id' in test_row and test_row[f'mouse{i}_id'] == agent_id:
                agent_str = f'mouse{i}'
            if f'mouse{i}_id' in test_row and test_row[f'mouse{i}_id'] == target_id:
                target_str = f'mouse{i}'
        if agent_str is None or target_str is None:
            continue  # Skip if mapping fails

        submission_rows.append({
            'video_id': video_id,
            'agent_id': agent_str,
            'target_id': target_str,
            'action': most_common_action,
            'start_frame': 0,
            'stop_frame': total_frames - 1  # inclusive
        })

# Create submission
if submission_rows:
    sub_df = pd.DataFrame(submission_rows)
    sub_df['row_id'] = range(len(sub_df))
    sub_df = sub_df[['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']]
else:
    # Fallback to sample submission format
    sample_sub = pd.read_csv(os.path.join(BASE_PATH, 'sample_submission.csv'))
    sub_df = sample_sub.head(0)  # empty but correct schema

sub_df.to_csv('submission.csv', index=False)
print("âœ… Submission generated with", len(sub_df), "rows.")
print(sub_df.head())




