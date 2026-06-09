# --- Core Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.auto import tqdm

# --- Modeling Libraries ---
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


# --- Set display options ---
pd.set_option('display.max_columns', 200)
sns.set_style('whitegrid')


# --- Define Constants and Paths ---
DATA_PATH = '/kaggle/input/MABe-mouse-behavior-detection/' 


# --- Reusable Function Definition ---

def load_and_process_video(video_id, lab_id, data_path):
    """
    Loads the tracking data for a single video and pivots it into a wide format.
    """
    # Using os.path.join is the most robust way to build paths
    tracking_path = os.path.join(data_path, 'train_tracking', lab_id, f'{video_id}.parquet')
    
    if not os.path.exists(tracking_path):
        print(f"Warning: File not found at {tracking_path}")
        return None
        
    df_long = pd.read_parquet(tracking_path)
    
    pivot_x = df_long.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='x')
    pivot_y = df_long.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='y')
    
    pivot_x.columns = [f"mouse{m}_{bp}_x" for m, bp in pivot_x.columns]
    pivot_y.columns = [f"mouse{m}_{bp}_y" for m, bp in pivot_y.columns]
    
    df_wide = pd.concat([pivot_x, pivot_y], axis=1)
    df_wide = df_wide.sort_index(axis=1)
    
    return df_wide


# --- Test the function with our sample video from Notebook 1 ---
print("Testing the data processing function...")
df_train_meta = pd.read_csv(os.path.join(DATA_PATH, 'train.csv')) # Using os.path.join here for safety
sample_video_meta = df_train_meta.iloc[0]

df_wide_sample = load_and_process_video(sample_video_meta['video_id'], sample_video_meta['lab_id'], DATA_PATH)

print("\n--- Function Test Output ---")
if df_wide_sample is not None:
    print(f"Successfully loaded and processed video {sample_video_meta['video_id']}")
    print(f"Shape of the resulting wide DataFrame: {df_wide_sample.shape}")
    display(df_wide_sample.head())
else:
    print("Failed to load the sample video.")


# --- 1. Select a Subset of Videos ---
# Let's use 50 videos for our baseline model. It's enough to be representative but fast to process.
N_VIDEOS_TO_USE = 50
df_subset_meta = df_train_meta.head(N_VIDEOS_TO_USE)

print(f"Using a subset of {len(df_subset_meta)} videos for this baseline model.")


# --- 2. Load and Combine Data for the Subset ---
all_wide_dfs = []
for index, row in tqdm(df_subset_meta.iterrows(), total=len(df_subset_meta)):
    df_wide = load_and_process_video(row['video_id'], row['lab_id'], DATA_PATH)
    if df_wide is not None:
        # Add a video_id column so we can link back to annotations
        df_wide['video_id'] = row['video_id']
        all_wide_dfs.append(df_wide)

# Combine all individual video dataframes into one big one
df_train_full = pd.concat(all_wide_dfs)

print(f"\nLoaded and combined data for all videos. Full training shape: {df_train_full.shape}")



# --- 3. Load Annotations and Create Frame-wise Labels ---
# Load all annotations for our subset of videos
all_annotations_list = []
for video_id in tqdm(df_subset_meta['video_id'].unique(), desc="Loading annotations"):
    row = df_subset_meta[df_subset_meta['video_id'] == video_id].iloc[0]
    annot_path = os.path.join(DATA_PATH, 'train_annotation', row['lab_id'], f"{row['video_id']}.parquet")
    if os.path.exists(annot_path):
        df_annot = pd.read_parquet(annot_path)
        df_annot['video_id'] = video_id
        all_annotations_list.append(df_annot)

df_annotations_subset = pd.concat(all_annotations_list)

# Initialize the target column with a "no_behavior" label
df_train_full['behavior'] = 'no_behavior'

# --- Apply labels to each frame ---
# This is a complex loop, but it's the core of the labeling process
print("\nApplying annotations to each frame...")
for index, row in tqdm(df_annotations_subset.iterrows(), total=len(df_annotations_subset)):
    video_id = row['video_id']
    start_frame = row['start_frame']
    stop_frame = row['stop_frame']
    action = row['action']
    
    # This is a simplification for the baseline: we create a single 'behavior' target.
    # We are not yet handling multiple simultaneous behaviors.
    df_train_full.loc[
        (df_train_full['video_id'] == video_id) & 
        (df_train_full.index >= start_frame) & 
        (df_train_full.index <= stop_frame),
        'behavior'
    ] = action

print("Labeling complete.")
print("\nValue counts of our new 'behavior' target column:")
print(df_train_full['behavior'].value_counts())


# --- 1. Define Features (X) and Target (y) ---
# Our features are all columns EXCEPT 'video_id' and our target 'behavior'
features = [col for col in df_train_full.columns if col not in ['video_id', 'behavior']]
X = df_train_full[features]
y = df_train_full['behavior']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")


# --- 2. Handle Missing Data ---
# LightGBM can handle NaNs, but filling them explicitly can sometimes be more stable.
# We'll fill with -1, a value that doesn't appear in the coordinate data.
X = X.fillna(-1)
print("\nFilled NaN values with -1.")


# --- 3. Encode String Labels into Numbers ---
# The model needs numerical targets, so 'attack' -> 0, 'chase' -> 1, etc.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Let's see the mapping
print("\nLabel Encoding Mapping:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name} -> {i}")


# --- 4. Split Data into Training and Validation Sets ---
# We use a simple 80/20 split. stratify=y_encoded ensures that the proportion
# of each behavior is the same in both the train and validation sets.
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)
print(f"\nTraining data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")


# --- 5. Train the LightGBM Model ---
print("\nTraining LightGBM model...")

lgbm = lgb.LGBMClassifier(
    objective='multiclass',
    n_estimators=500,  # Number of trees to build
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,         # Use all available CPU cores
    colsample_bytree=0.8, # Subsample columns to prevent overfitting
    subsample=0.8       # Subsample rows to prevent overfitting
)

# We use the validation set to monitor for early stopping
# This prevents the model from training for too long and overfitting.
lgbm.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss',
    callbacks=[lgb.early_stopping(10, verbose=True)]
)

print("\nModel training complete.")


# --- 1. Evaluate Performance on the Validation Set ---
print("--- Model Performance on Validation Set ---")

# Make predictions
y_pred = lgbm.predict(X_val)

# Convert the numerical predictions back to string labels for the report
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_val_labels = label_encoder.inverse_transform(y_val)

# Generate and print the classification report
report = classification_report(y_val_labels, y_pred_labels)
print(report)


# --- 2. Develop Post-Processing Logic ---
def predictions_to_submission(df_preds, video_id):
    """
    Converts frame-by-frame predictions into a submission-ready format.
    
    Args:
        df_preds (pd.DataFrame): DataFrame with 'frame' and 'behavior' columns.
        video_id (int or str): The ID of the video being processed.
        
    Returns:
        pd.DataFrame: A submission-formatted DataFrame for this video.
    """
    submission_rows = []
    
    # Ignore 'no_behavior' predictions
    df_preds = df_preds[df_preds['behavior'] != 'no_behavior'].copy()
    
    # Find contiguous blocks of the same behavior
    # This clever trick identifies where a block of the same behavior changes
    df_preds['block'] = (df_preds['behavior'] != df_preds['behavior'].shift()).cumsum()
    
    for _, group in df_preds.groupby('block'):
        # For our simple baseline, we'll assign mouse1 as agent and mouse2 as target
        # This is a major simplification we'll improve later.
        submission_rows.append({
            'video_id': video_id,
            'agent_id': 'mouse1',
            'target_id': 'mouse2',
            'action': group['behavior'].iloc[0],
            'start_frame': group['frame'].min(),
            'stop_frame': group['frame'].max(),
        })
        
    return pd.DataFrame(submission_rows)


# --- Test the post-processing function ---
print("\n--- Testing Post-Processing ---")

# Create a dummy prediction dataframe to test the logic
dummy_data = {
    'frame': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'behavior': ['no_behavior', 'attack', 'attack', 'attack', 'no_behavior', 'rear', 'rear', 'no_behavior', 'attack']
}
dummy_df = pd.DataFrame(dummy_data).set_index('frame')
dummy_submission = predictions_to_submission(dummy_df.reset_index(), 'dummy_video')

print("Dummy predictions converted to submission format:")
display(dummy_submission)


# --- 1. Load Test Metadata ---
print("Loading test metadata...")
df_test_meta = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
print(f"Found {len(df_test_meta)} videos in the test set.")

all_submissions = []


# --- 2. Iterate and Predict on Test Set ---
for index, row in tqdm(df_test_meta.iterrows(), total=len(df_test_meta)):
    video_id = row['video_id']
    lab_id = row['lab_id']
    
    print(f"\nProcessing video: {video_id}")
    
    # Load the video's tracking data
    # NOTE: The test tracking files are in the 'test_tracking' folder
    test_tracking_path = os.path.join(DATA_PATH, 'test_tracking', lab_id, f'{video_id}.parquet')
    
    # A bit of code duplication here, but it's safer to be explicit for the test set
    if not os.path.exists(test_tracking_path):
        print(f"  Warning: Test file not found at {test_tracking_path}. Skipping.")
        continue
        
    df_long_test = pd.read_parquet(test_tracking_path)
    
    # Pivot to wide format
    pivot_x = df_long_test.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='x')
    pivot_y = df_long_test.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='y')
    pivot_x.columns = [f"mouse{m}_{bp}_x" for m, bp in pivot_x.columns]
    pivot_y.columns = [f"mouse{m}_{bp}_y" for m, bp in pivot_y.columns]
    df_wide_test = pd.concat([pivot_x, pivot_y], axis=1).sort_index(axis=1)
    
    # Ensure test set has the same columns as the training set
    X_test = df_wide_test.reindex(columns=features, fill_value=-1)
    
    # Preprocess (fill NaNs)
    X_test = X_test.fillna(-1)
    
    # Predict
    print(f"  Predicting {len(X_test)} frames...")
    preds_encoded = lgbm.predict(X_test)
    preds_labels = label_encoder.inverse_transform(preds_encoded)
    
    # Post-process
    df_preds = pd.DataFrame({
        'frame': X_test.index,
        'behavior': preds_labels
    })
    
    video_submission = predictions_to_submission(df_preds, video_id)
    all_submissions.append(video_submission)
    print(f"  Found {len(video_submission)} behavior events in this video.")


# --- 3. Combine and Save ---
if all_submissions:
    df_submission = pd.concat(all_submissions, ignore_index=True)
    
    # The submission requires a 'row_id' column
    df_submission.index.name = 'row_id'
    
    # Make sure columns are in the exact order required
    final_columns = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
    df_submission = df_submission[final_columns]
    
    df_submission.to_csv('submission.csv', index=True)
    
    print("\n--- Submission File Generated ---")
    print(f"Total events predicted: {len(df_submission)}")
    display(df_submission.head())
else:
    # If no events were predicted for any video, create an empty submission file
    print("\nWarning: No behavior events were predicted. Creating an empty submission file.")
    pd.DataFrame(columns=['row_id'] + final_columns).to_csv('submission.csv', index=False)




