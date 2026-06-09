# Cell 1: Install all dependencies once
%pip install ultralytics==8.0.111 torch torchvision pandas numpy pydicom albumentations scikit-learn tqdm



# Point to the RSNA CSVs in your Kaggle dataset
Data_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'

import os
import pandas as pd

# 1) Read train_label_coordinates.csv and train_series_descriptions.csv
train_label_coordinates = pd.read_csv(os.path.join(Data_path, 'train_label_coordinates.csv'))
train_series_descriptions = pd.read_csv(os.path.join(Data_path, 'train_series_descriptions.csv'))

# 2) Merge on ['study_id', 'series_id'] to pull in series_description
merged_csv = pd.merge(
    train_label_coordinates,
    train_series_descriptions[['study_id', 'series_id', 'series_description']],
    on=['study_id', 'series_id'],
    how='left'
)

# 3) Read train.csv (holds the severity scores)
train_df = pd.read_csv(os.path.join(Data_path, 'train.csv'))

# 4) Helper to look up the correct column (e.g. 'neural_foraminal_narrowing_l1_l2') and fetch its value
def get_score(row):
    study_id = row['study_id']
    condition = row['condition']
    level = row['level']  # like "L1/L2" or "R3/R4"

    # Split "L1/L2" → "L1", "L2" so we can build a column name same as in train.csv
    level_1, level_2 = level.split('/')
    condition_level = f"{condition}_{level_1}_{level_2}".replace(' ', '_').lower()
    # e.g., "neural_foraminal_narrowing_l1_l2"

    if condition_level in train_df.columns and study_id in train_df['study_id'].values:
        return train_df.loc[train_df['study_id'] == study_id, condition_level].values[0]
    else:
        return None

# 5) Apply it to every row
merged_csv['score'] = merged_csv.apply(get_score, axis=1)

# 6) Save the merged result into /kaggle/working so downstream steps can consume it
out_path = '/kaggle/working/dataset_description.csv'
merged_csv.to_csv(out_path, index=False)
print(f"✅ Wrote merged CSV with scores to: {out_path}")



import os
import pandas as pd

# 1) Path to the merged file you created earlier
input_csv = '/kaggle/working/dataset_description.csv'

# 2) Load the full dataset_description.csv
df = pd.read_csv(input_csv)

# 3) Define which 'condition' values belong to each output group
condition_groups = {
    'Spinal Canal Stenosis': ['Spinal Canal Stenosis'],
    'Neural Foraminal Narrowing': ['Right Neural Foraminal Narrowing', 'Left Neural Foraminal Narrowing'],
    'Subarticular Stenosis': ['Right Subarticular Stenosis', 'Left Subarticular Stenosis']
}

# 4) For each group, filter and save a separate CSV
for group_name, conditions in condition_groups.items():
    filtered_df = df[df['condition'].isin(conditions)].copy()
    # Make a filesystem‐friendly name, e.g. "Spinal_Canal_Stenosis.csv"
    out_name = group_name.replace(' ', '_') + '.csv'
    out_path = os.path.join('/kaggle/working', out_name)
    filtered_df.to_csv(out_path, index=False)
    print(f"→ Wrote {len(filtered_df)} rows to {out_name}")

print("✅ Done splitting into three CSVs.")



import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def cross_validation_5fold(csv_path, output_name):
    """
    Reads a condition‐specific CSV, creates 'class_id' from condition+level,
    performs a 2‐fold stratified split, and writes out a new CSV with a 'fold' column.
    """
    df = pd.read_csv(csv_path)

    # Create a combined “condition_level” string
    df['condition_level'] = df['condition'] + '_' + df['level']

    # Convert to numeric class IDs for stratification
    df['class_id'] = df['condition_level'].astype('category').cat.codes

    # Use 2 splits instead of 5
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    df['fold'] = -1
    for fold_number, (_, val_idx) in enumerate(skf.split(df, df['class_id'])):
        df.loc[val_idx, 'fold'] = fold_number

    output_path = os.path.join('/kaggle/working', f"{output_name}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved 5‐fold CSV to: {output_path}")


# Paths to the condition‐specific CSVs (from the “split by condition” step)
spinal_csv = '/kaggle/working/Spinal_Canal_Stenosis.csv'
neural_csv = '/kaggle/working/Neural_Foraminal_Narrowing.csv'
subart_csv = '/kaggle/working/Subarticular_Stenosis.csv'

# Run 2‐fold stratified splitting for each condition
cross_validation_5fold(spinal_csv, 'Spinal_Canal_Stenosis_5folds')
cross_validation_5fold(neural_csv, 'Neural_Foraminal_Narrowing_5folds')
cross_validation_5fold(subart_csv, 'Subarticular_Stenosis_5folds')



import os

print(os.listdir('/kaggle/working'))
# You should see:
# [
#   'dataset_description.csv',
#   'Spinal_Canal_Stenosis.csv',
#   'Neural_Foraminal_Narrowing.csv',
#   'Subarticular_Stenosis.csv',
#   'Spinal_Canal_Stenosis_folds.csv',
#   'Neural_Foraminal_Narrowing_folds.csv',
#   'Subarticular_Stenosis_folds.csv',
#   … (any other files)
# ]





