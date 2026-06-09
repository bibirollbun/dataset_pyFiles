import os
import pandas as pd, numpy as np
from glob import glob
import matplotlib.pyplot as plt
VER = 1


# Define where all the parquet files live
BASE_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/'

# Gather every parquet file recursively under BASE_PATH
file_paths = glob(os.path.join(BASE_PATH, '**', '*.parquet'), recursive=True)

# Build a DataFrame listing each file
df = pd.DataFrame({'path': file_paths})

# Derive the test type by taking the parent folder name after the underscore
df['test_type'] = df['path'].apply(
    lambda p: os.path.basename(os.path.dirname(p)).split('_')[-1]
)

# Derive the sample ID by stripping directory and extension from the filename
df['id'] = df['path'].apply(
    lambda p: os.path.splitext(os.path.basename(p))[0]
)

# Read one example parquet to verify contents
df_eeg = pd.read_parquet(
    os.path.join(BASE_PATH, 'train_eegs', '1000913311.parquet')
)
df_eeg.head()


# Count how many EEG channels are present:
# Each column in df_eeg corresponds to one channel
n_channels = len(df_eeg.columns)

# Display the channel count
n_channels


# Specify the full path to the CSV of metadata and labels
csv_path = '/kaggle/input/hms-harmful-brain-activity-classification/train.csv'

# Read in the training table
df = pd.read_csv(csv_path)

# The final six columns are our classification targets
TARGETS = df.columns[-6:]

# Print a concise summary of rows/columns and list out the target names
num_rows, num_cols = df.shape
print(f"Train.csv contains {num_rows} records across {num_cols} columns.")
print(f"Target columns: {TARGETS.tolist()}")

# Quick look at the first few rows to confirm everything loaded correctly
df.head()


# Step 1: Extract the start of each EEG segment (per eeg_id)
# For each eeg_id, get the first spectrogram_id and the minimum offset time
segment_bounds = df.groupby('eeg_id')[['spectrogram_id', 'spectrogram_label_offset_seconds']].agg({
    'spectrogram_id': 'first',
    'spectrogram_label_offset_seconds': 'min'
})
segment_bounds.columns = ['spec_id', 'min']

# Step 2: Append the end time of each EEG segment
# Find the max offset time per eeg_id to mark the segment's end
end_times = df.groupby('eeg_id')[['spectrogram_label_offset_seconds']].agg('max')
segment_bounds['max'] = end_times

# Step 3: Attach patient_id per eeg_id
# Map each EEG to its corresponding patient
patient_info = df.groupby('eeg_id')[['patient_id']].agg('first')
segment_bounds['patient_id'] = patient_info

# Step 4: Aggregate target label counts
# Sum all target values per eeg_id (e.g. votes for each class)
target_counts = df.groupby('eeg_id')[TARGETS].agg('sum')
for label in TARGETS:
    segment_bounds[label] = target_counts[label].values

# Step 5: Normalize targets to get class probabilities
# Convert vote counts to probability distributions
label_matrix = segment_bounds[TARGETS].values
label_matrix = label_matrix / label_matrix.sum(axis=1, keepdims=True)
segment_bounds[TARGETS] = label_matrix

# Step 6: Add expert consensus label
# Pull in the expert-assigned label per eeg_id
expert_labels = df.groupby('eeg_id')[['expert_consensus']].agg('first')
segment_bounds['target'] = expert_labels

# Step 7: Finalize the training DataFrame
# Reset index so eeg_id becomes a column instead of index
train = segment_bounds.reset_index()

print('Train non-overlapp eeg_id shape:', train.shape)
train.head()


READ_SPEC_FILES =  False # If READ_SPEC_FILES is False, the code reads the combined file instead of individual files.
FEATURE_ENGINEER = True


%%time
# === Load all spectrogram parquet files from directory ===
SPECTROGRAM_DIR = '/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/'
file_list = os.listdir(SPECTROGRAM_DIR)
print(f'There are {len(file_list)} spectrogram parquets')

if READ_SPEC_FILES:
    # Initialize dictionary to store loaded spectrogram data
    spectrograms = {}
    for idx, file_name in enumerate(file_list):
        if idx % 100 == 0:
            print(idx, ', ', end='')

        # Read parquet, skip first column (e.g., time offset)
        spec_df = pd.read_parquet(f'{SPECTROGRAM_DIR}{file_name}')
        spec_id = int(file_name.split('.')[0])
        spectrograms[spec_id] = spec_df.iloc[:, 1:].values
else:
    # Load pre-processed spectrograms from .npy file
    spectrograms = np.load('/kaggle/input/d/wenzhedeng/brain-spectrograms/specs.npy', allow_pickle=True).item()


%time

# === Feature Extraction from Spectrograms ===
import warnings
warnings.filterwarnings('ignore')

# Compute derived features from spectrogram data
# Each spectrogram has 400 frequency channels, and we extract summary stats:
#   - Mean and Min over a 10-minute window
#   - Mean and Min over a 20-second window
# This results in 1600 features per EEG sample (400 x 4)

SPEC_COLS = pd.read_parquet(f'{SPECTROGRAM_DIR }1000086677.parquet').columns[1:]

FEATURES = [f'{col}_mean_15m' for col in SPEC_COLS]
FEATURES += [f'{col}_min_15m' for col in SPEC_COLS]
FEATURES += [f'{col}_mean_50s' for col in SPEC_COLS]
FEATURES += [f'{col}_min_50s' for col in SPEC_COLS]

print(f'We are creating {len(FEATURES)} features for {len(train)} rows... ', end='')

# Initialize and populate the feature matrix
if FEATURE_ENGINEER:
    feature_matrix = np.zeros((len(train), len(FEATURES)))

    for idx in range(len(train)):
        if idx % 100 == 0:
            print(idx, ', ', end='')

        row_data = train.iloc[idx]
        center_index = int((row_data['min'] + row_data['max']) // 4)

        # --- 15-minute window statistics (approx. 450 time steps) ---
        segment = spectrograms[row_data.spec_id][center_index:center_index + 450, :]
        feature_vals = np.nanmean(segment, axis=0)
        feature_matrix[idx, :400] = feature_vals
        feature_vals = np.nanmin(segment, axis=0)
        feature_matrix[idx, 400:800] = feature_vals

        # --- 50-second window statistics (approx. 25 time steps) ---
        short_segment = spectrograms[row_data.spec_id][center_index + 145:center_index + 170, :]
        feature_vals = np.nanmean(short_segment, axis=0)
        feature_matrix[idx, 800:1200] = feature_vals
        feature_vals = np.nanmin(short_segment, axis=0)
        feature_matrix[idx, 1200:1600] = feature_vals


    # Add features to training dataframe
    train[FEATURES] = feature_matrix
else:
    # Load pre-engineered features if skipping extraction
    train = pd.read_parquet('/kaggle/input/brain-spectrograms/train.pqt')

print()
print('New train shape:', train.shape)


from scipy import signal
from sklearn.decomposition import PCA


def extract_frequency_band_features(segment):
    # Specify frequency ranges corresponding to common EEG bands
    eeg_bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 45)
    }

    band_features = []

    for band in eeg_bands:
        low, high = eeg_bands[band]
        
        # Design a 3rd-order bandpass filter for the current frequency band
        bandpass_sos = signal.butter(3, [low, high], btype='bandpass', fs=200, output='sos')
        
        # Apply the filter to isolate the frequency content within the band
        filtered_signal = signal.sosfilt(bandpass_sos, segment)
        # Compute statistical summaries from the filtered signal
        band_features.extend([
            np.nanmean(filtered_signal),   # Average amplitude
            np.nanstd(filtered_signal),    # Signal variability
            np.nanmax(filtered_signal),    # Peak value
            np.nanmin(filtered_signal)     # Minimum value
        ])
    
    return band_features



# Here we comment all code in this cell is because this cell takes 30 mins and all fetures are not important(not in top 30)

# import time
# from sklearn.impute import SimpleImputer

# # Initialize a PCA model
# pca = PCA(n_components=0.95)
# print("PCA model initialized.")

# # Initialize an array for original features
# num_rows = len(train)
# num_features = 20 * n_channels  # 20 features per channel
# data_original = np.zeros((num_rows, num_features))

# print("Starting feature extraction and PCA processing...")
# start_time = time.time()

# for k in range(num_rows):
#     if k % 1000 == 0:
#         print(f"Processing row {k} of {num_rows}...")

#     row = train.iloc[k]
#     r = int((row['min'] + row['max']) // 4)
#     eeg_segment = spectrograms[row.spec_id][r:r+300, :]

#     # Apply the feature extraction function to each EEG channel
#     all_channel_features = []
#     for i in range(n_channels):
#         channel_features = extract_frequency_band_features(eeg_segment[:, i])
#         all_channel_features.extend(channel_features)
    
#     data_original[k, :] = all_channel_features

# print("Data matrix constructed")

# # Impute NaN values in the data matrix
# imputer = SimpleImputer(strategy='mean')
# data_imputed = imputer.fit_transform(data_original)

# print(f"NaN values handled. Imputed data matrix shape: {data_imputed.shape}")

# # Apply PCA on the imputed data
# pca.fit(data_imputed)
# print("PCA fitting completed.")

# # Transform data using PCA
# data_pca = pca.transform(data_imputed)

# # Add PCA features to DataFrame
# pca_feature_columns = [f'pca_feature_{i}' for i in range(data_pca.shape[1])]
# train[pca_feature_columns] = data_pca

# # Measure total processing time
# total_time = time.time() - start_time
# print(f"Total processing time: {total_time:.2f} seconds.")


train.head()


import gc
import xgboost as xgb
from sklearn.model_selection import KFold, GroupKFold

print('XGBoost version', xgb.__version__)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import gc

# (Assume train, FEATURES, TARGETS, TARS, VER are already defined)

all_oof   = []
all_true  = []
all_evals = []    # <-- we’ll collect eval results here

TARS = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}
gkf  = GroupKFold(n_splits=5)

for i, (train_index, valid_index) in enumerate(
        gkf.split(train, train.target, train.patient_id)
    ):
    print('#'*25)
    print(f'### Fold {i+1}')
    print(f'### train size {len(train_index)}, valid size {len(valid_index)}')
    print('#'*25)

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(TARS),
        learning_rate=0.1,
        tree_method='gpu_hist',  # or 'hist' if you don’t want GPU
        eval_metric='mlogloss'
    )

    # Prepare training and validation data
    X_train = train.loc[train_index, FEATURES]
    y_train = train.loc[train_index, 'target'].map(TARS)
    X_valid = train.loc[valid_index, FEATURES]
    y_valid = train.loc[valid_index, 'target'].map(TARS)

    # Fit with early stopping; capture eval results
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
        early_stopping_rounds=10
    )
    # Grab the eval history for the validation set (mlogloss per boosting round)
    evals_result = model.evals_result()
    # In XGBoost’s scikit‐learn API, this will be a dict like:
    # {'validation_0': {'mlogloss': [...], 'merror': [...] (if tracked)} }
    val_logloss = evals_result['validation_0']['mlogloss']
    all_evals.append(val_logloss)

    # Save OOF probabilities and truths
    oof = model.predict_proba(X_valid)
    all_oof.append(oof)
    all_true.append(train.loc[valid_index, TARGETS].values)

    # Optionally save the model to disk
    model.save_model(f'XGB_v{VER}_f{i}.model')

    # Clean up
    del X_train, y_train, X_valid, y_valid, oof
    gc.collect()

# Concatenate OOF arrays if you need them later
all_oof  = np.concatenate(all_oof, axis=0)
all_true = np.concatenate(all_true, axis=0)

# -----------------------
# Now: Plot the learning curves
# -----------------------
plt.figure(figsize=(8, 6))

for fold_idx, logloss_curve in enumerate(all_evals):
    plt.plot(
        logloss_curve,
        label=f'Fold {fold_idx+1}',
        linewidth=1.5
    )

plt.xlabel('Boosting Round', fontsize=12)
plt.ylabel('Validation Log‐Loss', fontsize=12)
plt.title('XGBoost Validation Log‐Loss vs. Boosting Round (5‐Fold)', fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


TOP_K = 15

# Retrieve importance scores assigned by the trained model
importances = model.feature_importances_

# Extract feature names from the training DataFrame
all_features = train.columns

# Determine the order of features based on importance (ascending)
ranking_indices = np.argsort(importances)

# Visualize the top K most important features
plt.figure(figsize=(10, 8))
top_indices = ranking_indices[-TOP_K:]
plt.barh(np.arange(TOP_K), importances[top_indices], align='center')
plt.yticks(np.arange(TOP_K), all_features[top_indices])
plt.title(f'Top {TOP_K} Feature Importances')
plt.tight_layout()
plt.show()


test = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/test.csv')
print('Test shape',test.shape)
test.head()


s = 853520
PATH2 = '/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/'
spec = pd.read_parquet(f'{PATH2}{s}.parquet')
spec


# FEATURE ENGINEER TEST
PATH2 = '/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/'
data = np.zeros((len(test),len(FEATURES)))
    
for k in range(len(test)):
    row = test.iloc[k]
    s = int( row.spectrogram_id )
    spec = pd.read_parquet(f'{PATH2}{s}.parquet')
    
    # 10 MINUTE WINDOW FEATURES
    x = np.nanmean( spec.iloc[:,1:].values, axis=0)
    data[k,:400] = x
    x = np.nanmin( spec.iloc[:,1:].values, axis=0)
    data[k,400:800] = x

    # 20 SECOND WINDOW FEATURES
    x = np.nanmean( spec.iloc[145:155,1:].values, axis=0)
    data[k,800:1200] = x
    x = np.nanmin( spec.iloc[145:155,1:].values, axis=0)
    data[k,1200:1600] = x
    
test[FEATURES] = data
print('New test shape',test.shape)
print(test)



# INFER XGBOOST ON TEST
preds = []

for i in range(5):
    print(i, ', ', end='')
    
    # Load the XGBoost model
    model = xgb.XGBClassifier()
    model.load_model(f'XGB_v{VER}_f{i}.model')
    
    # Make predictions
    pred = model.predict_proba(test[FEATURES])
    preds.append(pred)

# Average the predictions from each fold
pred = np.mean(preds, axis=0)
print()
print('Test preds shape', pred.shape)
print(pred)



sub = pd.DataFrame({'eeg_id':test.eeg_id.values})
sub[TARGETS] = pred
sub.to_csv('submission.csv',index=False)
print('Submission shape',sub.shape)

sub.head()
print(sub)


# SANITY CHECK TO CONFIRM PREDICTIONS SUM TO ONE
sub.iloc[:,-6:].sum(axis=1)




