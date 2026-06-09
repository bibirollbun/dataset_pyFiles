# --- Core Libraries ---
import pandas as pd
import numpy as np
import os
import seaborn as sns
from tqdm.auto import tqdm
import warnings

# --- Modeling Libraries ---
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


# --- Set display options ---
pd.set_option('display.max_columns', 200)
sns.set_style('whitegrid')

# --- Define Constants and Paths ---
DATA_PATH = '/kaggle/input/MABe-mouse-behavior-detection' 

warnings.filterwarnings("ignore")


# --- Reusable Data Loading Function (from Notebook 2) ---
def load_and_process_video(video_id, lab_id, data_path):
    tracking_path = os.path.join(data_path, 'train_tracking', lab_id, f'{video_id}.parquet')
    if not os.path.exists(tracking_path):
        return None
    df_long = pd.read_parquet(tracking_path)
    pivot_x = df_long.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='x')
    pivot_y = df_long.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='y')
    pivot_x.columns = [f"mouse{m}_{bp}_x" for m, bp in pivot_x.columns]
    pivot_y.columns = [f"mouse{m}_{bp}_y" for m, bp in pivot_y.columns]
    df_wide = pd.concat([pivot_x, pivot_y], axis=1).sort_index(axis=1)
    return df_wide



# --- New Feature Engineering Function ---

# 2. Define a core set of bodyparts to build features from
CORE_BODYPARTS = ['nose', 'ear_left', 'ear_right', 'neck', 'body_center', 'tail_base']

def create_advanced_features(df_wide):
    """
    Creates a rich set of kinematic, interaction, and postural features.
    """
    # Start with a copy of the original data
    features_df = df_wide.copy()
    
    mouse_ids = [1, 2, 3, 4] # Assuming up to 4 mice
    
    # --- 1. Kinematic Features (Speeds) ---
    for mid in mouse_ids:
        for part in CORE_BODYPARTS:
            col_x, col_y = f'mouse{mid}_{part}_x', f'mouse{mid}_{part}_y'
            if col_x in features_df.columns:
                delta_x = features_df[col_x].diff()
                delta_y = features_df[col_y].diff()
                features_df[f'mouse{mid}_{part}_speed'] = np.sqrt(delta_x**2 + delta_y**2)

    # --- 2. Postural Features (Body Elongation) ---
    for mid in mouse_ids:
        nose_x, nose_y = f'mouse{mid}_nose_x', f'mouse{mid}_nose_y'
        tail_x, tail_y = f'mouse{mid}_tail_base_x', f'mouse{mid}_tail_base_y'
        if all(c in features_df.columns for c in [nose_x, nose_y, tail_x, tail_y]):
            features_df[f'mouse{mid}_elongation'] = np.sqrt(
                (features_df[nose_x] - features_df[tail_x])**2 + 
                (features_df[nose_y] - features_df[tail_y])**2
            )

    # --- 3. Interaction Features (Distances) ---
    mouse_pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    for m1, m2 in mouse_pairs:
        for part1 in CORE_BODYPARTS:
            for part2 in CORE_BODYPARTS:
                p1_x, p1_y = f'mouse{m1}_{part1}_x', f'mouse{m1}_{part1}_y'
                p2_x, p2_y = f'mouse{m2}_{part2}_x', f'mouse{m2}_{part2}_y'
                if all(c in features_df.columns for c in [p1_x, p1_y, p2_x, p2_y]):
                    features_df[f'dist_m{m1}{part1}_m{m2}{part2}'] = np.sqrt(
                        (features_df[p1_x] - features_df[p2_x])**2 + 
                        (features_df[p1_y] - features_df[p2_y])**2
                    )
                    
    # Drop the original coordinate columns to force the model to use our new features
    features_df = features_df.drop(columns=df_wide.columns)
    
    return features_df


# --- Test the function with our sample video ---
print("Testing the feature engineering function...")
df_train_meta = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
sample_video_meta = df_train_meta.iloc[0]

df_wide_sample = load_and_process_video(sample_video_meta['video_id'], sample_video_meta['lab_id'], DATA_PATH)
df_features_sample = create_advanced_features(df_wide_sample)

print("\n--- Function Test Output ---")
if df_features_sample is not None:
    print(f"Successfully created features for video {sample_video_meta['video_id']}")
    print(f"Original coordinate columns: {len(df_wide_sample.columns)}")
    print(f"New engineered feature columns: {len(df_features_sample.columns)}")
    display(df_features_sample.head())
else:
    print("Failed to create features for the sample video.")


# --- 1. Select the Same Subset of Videos ---
N_VIDEOS_TO_USE = 50
df_subset_meta = df_train_meta.head(N_VIDEOS_TO_USE)
print(f"Using the same subset of {len(df_subset_meta)} videos for a fair comparison.")


# --- 2. Load, Process, Create Features, and Combine ---
all_featured_dfs = []
for index, row in tqdm(df_subset_meta.iterrows(), total=len(df_subset_meta)):
    # Load and pivot the data
    df_wide = load_and_process_video(row['video_id'], row['lab_id'], DATA_PATH)
    
    if df_wide is not None:
        # ** NEW STEP: Create advanced features **
        df_features = create_advanced_features(df_wide)
        
        # Add a video_id column for linking annotations
        df_features['video_id'] = row['video_id']
        all_featured_dfs.append(df_features)

# Combine all individual video dataframes into one big one
df_train_full_featured = pd.concat(all_featured_dfs)

print(f"\nLoaded and created features for all videos. Full training shape: {df_train_full_featured.shape}")


# --- 3. Load Annotations and Create Frame-wise Labels ---
all_annotations_list = []
for video_id in tqdm(df_subset_meta['video_id'].unique(), desc="Loading annotations"):
    row = df_subset_meta[df_subset_meta['video_id'] == video_id].iloc[0]
    annot_path = os.path.join(DATA_PATH, 'train_annotation', row['lab_id'], f"{row['video_id']}.parquet")
    if os.path.exists(annot_path):
        df_annot = pd.read_parquet(annot_path)
        df_annot['video_id'] = video_id
        all_annotations_list.append(df_annot)
df_annotations_subset = pd.concat(all_annotations_list)

# Initialize and apply labels
df_train_full_featured['behavior'] = 'no_behavior'
print("\nApplying annotations to each frame...")
for index, row in tqdm(df_annotations_subset.iterrows(), total=len(df_annotations_subset)):
    video_id, start, stop, action = row['video_id'], row['start_frame'], row['stop_frame'], row['action']
    
    df_train_full_featured.loc[
        (df_train_full_featured['video_id'] == video_id) & 
        (df_train_full_featured.index >= start) & 
        (df_train_full_featured.index <= stop),
        'behavior'
    ] = action

print("Labeling complete.")
print("\nValue counts of the target column (should be identical to Notebook 2):")
print(df_train_full_featured['behavior'].value_counts())


# --- 1. Define Features (X) and Target (y) ---
# Our features are all columns EXCEPT 'video_id' and our target 'behavior'
features = [col for col in df_train_full_featured.columns if col not in ['video_id', 'behavior']]
X = df_train_full_featured[features]
y = df_train_full_featured['behavior']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")


# --- 2. Handle Missing Data ---
# The first frame of each video will have NaN for speed features. Fill them with -1.
X = X.fillna(-1)
print("\nFilled NaN values with -1.")


# --- 3. Encode String Labels into Numbers ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("\nLabels have been encoded.")


# --- 4. Split Data into Training and Validation Sets ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)
print(f"\nTraining data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")


# --- 5. Train the LightGBM Model ---
print("\nTraining LightGBM model on NEW features...")

# Use the same model parameters as Notebook 2 for a fair comparison
lgbm_featured = lgb.LGBMClassifier(
    objective='multiclass',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    colsample_bytree=0.8,
    subsample=0.8
)

lgbm_featured.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss',
    callbacks=[lgb.early_stopping(10, verbose=True)]
)

print("\nModel training complete.")


# --- 1. Evaluate Performance on the Validation Set ---
print("--- Model Performance on Validation Set (with Engineered Features) ---")

# Make predictions with the new model
y_pred_featured = lgbm_featured.predict(X_val)

# Convert the numerical predictions back to string labels for the report
y_pred_labels = label_encoder.inverse_transform(y_pred_featured)
y_val_labels = label_encoder.inverse_transform(y_val)

# Generate and print the classification report
report_featured = classification_report(y_val_labels, y_pred_labels)
print(report_featured)

print("\n--- For Comparison: Baseline Report from Notebook 2 ---")
# (I've pasted the key metrics from the previous notebook's output here for easy comparison)
baseline_report_text = """
              precision    recall  f1-score   support

      approach       0.58      0.23      0.32      3568
        attack       0.66      0.18      0.28      7646
         avoid       0.80      0.16      0.27      4256
         chase       0.73      0.22      0.33      3130
   chaseattack       0.69      0.64      0.66      1067
     disengage       0.62      0.28      0.39      2460
dominancemount       0.25      0.52      0.33        91
         mount       0.81      0.62      0.70      2123
   no_behavior       0.97      0.99      0.98   1090760
          rear       0.72      0.46      0.56     15781
     selfgroom       0.79      0.46      0.58      2948
      shepherd       0.69      0.14      0.24      5930
         sniff       0.65      0.54      0.59      7360
     sniffbody       0.35      0.52      0.42       290
     sniffface       0.13      0.18      0.15        79
  sniffgenital       0.71      0.75      0.73       716
        submit       0.76      0.76      0.76      1204

    macro avg       0.64      0.45      0.49   1149409
 weighted avg       0.96      0.96      0.95   1149409
"""
print(baseline_report_text)


# --- 1. Create a Hybrid Feature Function ---
def create_hybrid_features(df_wide):
    """
    Creates a combined set of raw coordinates and engineered features.
    """
    # Start with the original coordinate features from df_wide
    # Create the engineered features, but this time, don't drop the originals
    engineered_features = create_advanced_features(df_wide.copy())
    
    # Combine them
    hybrid_features = pd.concat([df_wide, engineered_features], axis=1)
    return hybrid_features

print("--- Training a Hybrid Model (Best of Both Worlds) ---")




# --- 2. Build the Hybrid Dataset (using just the first 10 videos for speed) ---
print("\nBuilding a small hybrid dataset for a quick test...")
N_VIDEOS_HYBRID = 10
df_hybrid_meta = df_train_meta.head(N_VIDEOS_HYBRID)

all_hybrid_dfs = []
for index, row in tqdm(df_hybrid_meta.iterrows(), total=len(df_hybrid_meta)):
    df_wide = load_and_process_video(row['video_id'], row['lab_id'], DATA_PATH)
    if df_wide is not None:
        df_hybrid = create_hybrid_features(df_wide)
        df_hybrid['video_id'] = row['video_id']
        all_hybrid_dfs.append(df_hybrid)
df_train_hybrid = pd.concat(all_hybrid_dfs)

# Apply labels
df_train_hybrid['behavior'] = 'no_behavior'
# We only need the annotations for these 10 videos
annot_subset_hybrid = df_annotations_subset[df_annotations_subset['video_id'].isin(df_hybrid_meta['video_id'])]
for index, row in tqdm(annot_subset_hybrid.iterrows(), total=len(annot_subset_hybrid)):
    video_id, start, stop, action = row['video_id'], row['start_frame'], row['stop_frame'], row['action']
    df_train_hybrid.loc[
        (df_train_hybrid['video_id'] == video_id) & (df_train_hybrid.index >= start) & (df_train_hybrid.index <= stop),
        'behavior'
    ] = action


# --- 3. Train and Evaluate the Hybrid Model ---
features_hybrid = [col for col in df_train_hybrid.columns if col not in ['video_id', 'behavior']]
X_hybrid = df_train_hybrid[features_hybrid].fillna(-1)
y_hybrid = df_train_hybrid['behavior']
y_encoded_hybrid = label_encoder.transform(y_hybrid) # Use the same encoder

X_train_h, X_val_h, y_train_h, y_val_h = train_test_split(
    X_hybrid, y_encoded_hybrid, test_size=0.2, random_state=42, stratify=y_encoded_hybrid
)

print("\nTraining Hybrid LightGBM model...")
lgbm_hybrid = lgb.LGBMClassifier(objective='multiclass', random_state=42, n_jobs=-1) # Using simpler params for speed
lgbm_hybrid.fit(X_train_h, y_train_h)

print("\n--- Hybrid Model Performance ---")
y_pred_hybrid = lgbm_hybrid.predict(X_val_h)
y_pred_labels_h = label_encoder.inverse_transform(y_pred_hybrid)
y_val_labels_h = label_encoder.inverse_transform(y_val_h)
print(classification_report(y_val_labels_h, y_pred_labels_h))




