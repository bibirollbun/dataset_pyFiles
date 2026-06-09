import os
import pandas as pd, numpy as np
from glob import glob
import matplotlib.pyplot as plt
VER = 1


BASE_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/'
file_paths = glob(os.path.join(BASE_PATH, '**', '*.parquet'), recursive=True)
df = pd.DataFrame({'path': file_paths})
df['test_type'] = df['path'].apply(
    lambda p: os.path.basename(os.path.dirname(p)).split('_')[-1]
)
df['id'] = df['path'].apply(
    lambda p: os.path.splitext(os.path.basename(p))[0]
)
df_eeg = pd.read_parquet(
    os.path.join(BASE_PATH, 'train_eegs', '1000913311.parquet')
)
df_eeg.head()


n_channels = len(df_eeg.columns)
n_channels


csv_path = '/kaggle/input/hms-harmful-brain-activity-classification/train.csv'
df = pd.read_csv(csv_path)
TARGETS = df.columns[-6:]
num_rows, num_cols = df.shape
print(f"Train.csv contains {num_rows} records across {num_cols} columns.")
print(f"Target columns: {TARGETS.tolist()}")
df.head()


segment_bounds = df.groupby('eeg_id')[['spectrogram_id', 'spectrogram_label_offset_seconds']].agg({
    'spectrogram_id': 'first',
    'spectrogram_label_offset_seconds': 'min'
})
segment_bounds.columns = ['spec_id', 'min']
end_times = df.groupby('eeg_id')[['spectrogram_label_offset_seconds']].agg('max')
segment_bounds['max'] = end_times

patient_info = df.groupby('eeg_id')[['patient_id']].agg('first')
segment_bounds['patient_id'] = patient_info

target_counts = df.groupby('eeg_id')[TARGETS].agg('sum')
for label in TARGETS:
    segment_bounds[label] = target_counts[label].values

label_matrix = segment_bounds[TARGETS].values
label_matrix = label_matrix / label_matrix.sum(axis=1, keepdims=True)
segment_bounds[TARGETS] = label_matrix

expert_labels = df.groupby('eeg_id')[['expert_consensus']].agg('first')
segment_bounds['target'] = expert_labels

train = segment_bounds.reset_index()
print('Train non-overlapp eeg_id shape:', train.shape)
train.head()


READ_SPEC_FILES = False 
FEATURE_ENGINEER = True


%%time
SPECTROGRAM_DIR = '/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/'
file_list = os.listdir(SPECTROGRAM_DIR)
print(f'There are {len(file_list)} spectrogram parquets')

if READ_SPEC_FILES:
    spectrograms = {}
    for idx, file_name in enumerate(file_list):
        if idx % 100 == 0:
            print(idx, ', ', end='')
        spec_df = pd.read_parquet(f'{SPECTROGRAM_DIR}{file_name}')
        spec_id = int(file_name.split('.')[0])
        spectrograms[spec_id] = spec_df.iloc[:, 1:].values
else:
    spectrograms = np.load('/kaggle/input/brain-spectrograms/specs.npy', allow_pickle=True).item()


%time
import warnings
warnings.filterwarnings('ignore')
SPEC_COLS = pd.read_parquet(f'{SPECTROGRAM_DIR }1000086677.parquet').columns[1:]

FEATURES = [f'{col}_mean_15m' for col in SPEC_COLS]
FEATURES += [f'{col}_min_15m' for col in SPEC_COLS]
FEATURES += [f'{col}_mean_50s' for col in SPEC_COLS]
FEATURES += [f'{col}_min_50s' for col in SPEC_COLS]

print(f'We are creating {len(FEATURES)} features for {len(train)} rows... ', end='')

if FEATURE_ENGINEER:
    feature_matrix = np.zeros((len(train), len(FEATURES)))
    
    for idx in range(len(train)):
        if idx % 100 == 0:
            print(idx, ', ', end='')
        row_data = train.iloc[idx]
        center_index = int((row_data['min'] + row_data['max']) // 4)

        segment = spectrograms[row_data.spec_id][center_index:center_index + 450, :]
        feature_vals = np.nanmean(segment, axis=0)
        feature_matrix[idx, :400] = feature_vals
        feature_vals = np.nanmin(segment, axis=0)
        feature_matrix[idx, 400:800] = feature_vals
        
        short_segment = spectrograms[row_data.spec_id][center_index + 145:center_index + 170, :]
        feature_vals = np.nanmean(short_segment, axis=0)
        feature_matrix[idx, 800:1200] = feature_vals
        feature_vals = np.nanmin(short_segment, axis=0)
        feature_matrix[idx, 1200:1600] = feature_vals
        
    train[FEATURES] = feature_matrix
else:
    train = pd.read_parquet('/kaggle/input/brain-spectrograms/train.pqt')

print()
print('New train shape:', train.shape)


from scipy import signal
from sklearn.decomposition import PCA


def extract_frequency_band_features(segment):
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
        bandpass_sos = signal.butter(3, [low, high], btype='bandpass', fs=200, output='sos')
        filtered_signal = signal.sosfilt(bandpass_sos, segment)
        band_features.extend([
            np.nanmean(filtered_signal),  
            np.nanstd(filtered_signal),    
            np.nanmax(filtered_signal),    
            np.nanmin(filtered_signal)     
        ])
    return band_features


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

all_oof   = []
all_true  = []
all_evals = []    

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
        tree_method='hist', 
        eval_metric='mlogloss'
    )
    X_train = train.loc[train_index, FEATURES]
    y_train = train.loc[train_index, 'target'].map(TARS)
    X_valid = train.loc[valid_index, FEATURES]
    y_valid = train.loc[valid_index, 'target'].map(TARS)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
        early_stopping_rounds=10
    )
    
    evals_result = model.evals_result()
    val_logloss = evals_result['validation_0']['mlogloss']
    all_evals.append(val_logloss)
   
    oof = model.predict_proba(X_valid)
    all_oof.append(oof)
    all_true.append(train.loc[valid_index, TARGETS].values)

    model.save_model(f'XGB_v{VER}_f{i}.model')

    del X_train, y_train, X_valid, y_valid, oof
    gc.collect()

all_oof  = np.concatenate(all_oof, axis=0)
all_true = np.concatenate(all_true, axis=0)

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


TOP_K = 30
importances = model.feature_importances_
all_features = train.columns
ranking_indices = np.argsort(importances)

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


PATH2 = '/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/'
data = np.zeros((len(test),len(FEATURES))) 
for k in range(len(test)):
    row = test.iloc[k]
    s = int( row.spectrogram_id )
    spec = pd.read_parquet(f'{PATH2}{s}.parquet')
    x = np.nanmean( spec.iloc[:,1:].values, axis=0)
    data[k,:400] = x
    x = np.nanmin( spec.iloc[:,1:].values, axis=0)
    data[k,400:800] = x
    x = np.nanmean( spec.iloc[145:155,1:].values, axis=0)
    data[k,800:1200] = x
    x = np.nanmin( spec.iloc[145:155,1:].values, axis=0)
    data[k,1200:1600] = x

test[FEATURES] = data
print('New test shape',test.shape)
print(test)


preds = []
for i in range(5):
    print(i, ', ', end='')
    model = xgb.XGBClassifier()
    model.load_model(f'XGB_v{VER}_f{i}.model')
    pred = model.predict_proba(test[FEATURES])
    preds.append(pred)

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


sub.iloc[:,-6:].sum(axis=1)

