import pandas as pd
import numpy as np
import os
import joblib
import json

# --- PART 1: Load Model and Training Columns ---
# IMPORTANT: Update 'mabe-final-assets' if you named your dataset something else
model_path = '/kaggle/input/mabe-final-assets/final_model.pkl' 
columns_path = '/kaggle/input/mabe-final-assets/training_columns.json'

model = joblib.load(model_path)
with open(columns_path, 'r') as f:
    training_columns = json.load(f)

# --- PART 2: Feature Engineering Function ---
def create_features_for_video(video_id, lab_id, data_path):
    try:
        tracking_file_path = f"{data_path}/{lab_id}/{video_id}.parquet"
        tracking_df = pd.read_parquet(tracking_file_path)
        mice_in_video = tracking_df['mouse_id'].unique()
        if 1 not in mice_in_video or 2 not in mice_in_video: return None
        wide_df = tracking_df.pivot(index=['video_frame', 'mouse_id'], columns='bodypart', values=['x', 'y'])
        wide_df.columns = ['_'.join(col) for col in wide_df.columns.values]
        wide_df = wide_df.reset_index().interpolate(method='linear', limit_direction='both')
        wide_df['body_length'] = np.sqrt((wide_df['x_nose'] - wide_df['x_tail_base'])**2 + (wide_df['y_nose'] - wide_df['y_tail_base'])**2)
        bodyparts = ['nose', 'tail_base']
        for part in bodyparts:
            for coord in ['x', 'y']:
                col_name = f'{coord}_{part}'
                wide_df[f'vel_{col_name}'] = wide_df.groupby('mouse_id')[col_name].diff().fillna(0)
                wide_df[f'accel_{col_name}'] = wide_df.groupby('mouse_id')[f'vel_{col_name}'].diff().fillna(0)
        mouse1_df = wide_df[wide_df['mouse_id'] == 1].set_index('video_frame')
        mouse2_df = wide_df[wide_df['mouse_id'] == 2].set_index('video_frame')
        features_df = pd.merge(mouse1_df, mouse2_df, on='video_frame', suffixes=('_m1', '_m2'))
        features_df['nose_to_nose_dist'] = np.sqrt((features_df['x_nose_m1'] - features_df['x_nose_m2'])**2 + (features_df['y_nose_m1'] - features_df['y_nose_m2'])**2)
        features_df['nose_to_tail_dist'] = np.sqrt((features_df['x_nose_m1'] - features_df['x_tail_base_m2'])**2 + (features_df['y_nose_m1'] - features_df['y_tail_base_m2'])**2)
        features_df = features_df.drop(columns=['mouse_id_m1', 'mouse_id_m2'])
        
        # Force the columns to match the training set
        return features_df.reindex(columns=training_columns, fill_value=0)
    except Exception as e:
        return None

# --- PART 3: Process the Test Set and Create Submission File ---
competition_data_path = '/kaggle/input/MABe-mouse-behavior-detection'
test_tracking_path = f'{competition_data_path}/test_tracking'
test_df = pd.read_csv(f'{competition_data_path}/test.csv')
video_to_lab_map = test_df.set_index('video_id')['lab_id'].to_dict()
all_predictions = []

print("Starting prediction on the hidden test set...")
for video_id, lab_id in video_to_lab_map.items():
    features = create_features_for_video(video_id, lab_id, test_tracking_path)
    if features is not None and not features.empty:
        frame_predictions = model.predict(features)
        current_action = 'background'
        start_frame = -1
        for i, action in enumerate(frame_predictions):
            frame_num = features.index[i]
            if action != current_action:
                if current_action != 'background':
                    all_predictions.append([video_id, 1, 2, current_action, start_frame, frame_num - 1])
                current_action = action
                start_frame = frame_num
        if current_action != 'background':
            all_predictions.append([video_id, 1, 2, current_action, start_frame, features.index[-1]])

print("Prediction complete. Creating submission file...")
submission_df = pd.DataFrame(all_predictions, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
submission_df['agent_id'] = 'mouse' + submission_df['agent_id'].astype(str)
submission_df['target_id'] = 'mouse' + submission_df['target_id'].astype(str)
submission_df.insert(0, 'row_id', range(len(submission_df)))
submission_df.to_csv('submission.csv', index=False)
print("submission.csv created successfully!")

