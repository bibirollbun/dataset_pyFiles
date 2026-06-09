# Basic check: list all files in input directory
import os

print("Listing files and folders in /kaggle/input/ ...")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load basic CSV files to check
import pandas as pd

labels_df = pd.read_csv('/kaggle/input/inria-bci-challenge/TrainLabels.csv')


# Adjust paths to your actual files inside the competition data folder
train_labels = pd.read_csv('/kaggle/input/inria-bci-challenge/TrainLabels.csv')
channels_loc = pd.read_csv('/kaggle/input/inria-bci-challenge/ChannelsLocation.csv')
sample_submission = pd.read_csv('/kaggle/input/inria-bci-challenge/SampleSubmission.csv')

print("\nTrain labels example:")
print(train_labels.head())

print("\nChannels Location example:")
print(channels_loc.head())

print("\nSample Submission example:")
print(sample_submission.head())



import zipfile

zip_train_path = '/kaggle/input/inria-bci-challenge/train.zip'
zip_test_path = '/kaggle/input/inria-bci-challenge/test.zip'
extract_path = '/kaggle/working/bci_eeg_data'  # Working directory

# Unzip train data
with zipfile.ZipFile(zip_train_path, 'r') as zip_ref:
    zip_ref.extractall(f'{extract_path}/train')

# Unzip test data
with zipfile.ZipFile(zip_test_path, 'r') as zip_ref:
    zip_ref.extractall(f'{extract_path}/test')

print("Train and test data extracted.")



import os

for root, dirs, files in os.walk(extract_path):
    print(f"Folder: {root}")
    for f in files[:5]:  # show first 5 files per folder for brevity
        print(f"  {f}")



import pandas as pd
import numpy as np

train_folder = f'{extract_path}/train'
example_file = os.listdir(train_folder)[0]
example_path = f'{train_folder}/{example_file}'

def load_eeg_csv(path):
    df = pd.read_csv(path)
    eeg_data = df.drop(columns=['Time', 'FeedBackEvent']).values.T  # channels x samples
    return eeg_data

eeg_raw = load_eeg_csv(example_path)
print(f"Loaded EEG shape (channels x samples): {eeg_raw.shape}")



import matplotlib.pyplot as plt

channel = 0  # Plot first channel
plt.figure(figsize=(15, 4))
plt.plot(eeg_raw[channel, :1500])
plt.title('Raw EEG Signal (Channel 0) - First 1500 samples')
plt.xlabel('Sample number')
plt.ylabel('Amplitude')
plt.show()



from scipy import signal

def preprocess_eeg(eeg_data, fs=600):
    # Band-pass filter between 0.1 Hz and 30 Hz
    sos = signal.butter(4, [0.1, 30], btype='band', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, eeg_data, axis=1)
    
    # Common Average Reference (CAR)
    car = filtered - np.mean(filtered, axis=0, keepdims=True)
    return car

eeg_processed = preprocess_eeg(eeg_raw)
print('Preprocessing done.')



plt.figure(figsize=(15, 4))
plt.plot(eeg_processed[channel, :1500])
plt.title('Preprocessed EEG Signal (Channel 0) - First 1500 samples')
plt.xlabel('Sample number')
plt.ylabel('Amplitude')
plt.show()



def extract_epochs(file_path, fs=600, epoch_duration=1.3):
  
    df = pd.read_csv(file_path)
    eeg_data = df.drop(columns=['Time', 'FeedBackEvent']).values.T  # (channels, samples)
    
    feedback_idxs = df.index[df['FeedBackEvent'] == 1].tolist()
    epoch_length = int(fs * epoch_duration)
    
    epochs = []
    for idx in feedback_idxs:
        if idx + epoch_length <= eeg_data.shape[1]:
            segment = eeg_data[:, idx:idx + epoch_length]
            segment_preprocessed = preprocess_eeg(segment, fs)
            epochs.append(segment_preprocessed)
    
    return epochs

# Example run on one EEG file
example_epochs = extract_epochs(example_path)
print(f"Extracted {len(example_epochs)} feedback-aligned epochs from example file.")
print(f"Each epoch shape: {example_epochs[0].shape if example_epochs else 'N/A'}")



def reject_artifacts(epochs, thresh=200):
    clean_epochs = []
    for e in epochs:
        if np.all(np.abs(e) < thresh):  # keep epoch if no channel exceeds threshold
            clean_epochs.append(e)
    return clean_epochs

all_clean_epochs = []
all_clean_epoch_ids = []

# Iterate over all training files
for filename in sorted(os.listdir(train_folder)):
    prefix = filename.replace('Data_','').replace('.csv','')
    path = os.path.join(train_folder, filename)
    epochs = extract_epochs(path)
    
    # Reject bad epochs by threshold
    clean_epochs = reject_artifacts(epochs)

    all_clean_epochs.extend(clean_epochs)
    
    # Keep only IDs for accepted epochs
    for i in range(len(clean_epochs)):
        epoch_id = f"{prefix}_FB{i + 1:03d}"
        all_clean_epoch_ids.append(epoch_id)

print(f"Extracted total clean epochs: {len(all_clean_epochs)}")

# Map labels to clean epochs
labels_map = labels_df.set_index('IdFeedBack')['Prediction'].to_dict()
epoch_labels = [labels_map.get(eid, 0) for eid in all_clean_epoch_ids]

print(f"Total labels assigned after artifact rejection: {len(epoch_labels)}")
print(f"Label distribution: {pd.Series(epoch_labels).value_counts().to_dict()}")



import numpy as np

def extract_temporal_features(epochs, fs=600):
    # Temporal windows means in multiple sliding windows and lags
    window_sizes_ms = np.arange(50, 700, 50)  # 50 to 650 ms
    lags_ms = np.arange(0, 1300, 50)          # 0 to 1250 ms
    window_sizes = (window_sizes_ms * fs // 1000).astype(int)
    lags = (lags_ms * fs // 1000).astype(int)

    features = []
    for epo in epochs:
        feat_epo = []
        for ch in range(epo.shape[0]):
            for w in window_sizes:
                for lag in lags:
                    if lag + w <= epo.shape[1]:
                        segment = epo[ch, lag:lag + w]
                        feat_epo.append(segment.mean())
                    else:
                        feat_epo.append(0)
        features.append(feat_epo)
    return np.nan_to_num(np.array(features))


def extract_template_features(epochs, labels):
    positive_epochs = [epochs[i] for i in range(len(epochs)) if labels[i] == 1]
    if len(positive_epochs) == 0:
        # No positive epochs, return zero array of proper shape
        return np.zeros((len(epochs), epochs[0].shape[0]))
    
    template = np.mean(positive_epochs, axis=0)
    
    features = []
    for epo in epochs:
        corr_channels = []
        for ch in range(epo.shape[0]):
            std_epo_ch = np.std(epo[ch])
            std_tmpl_ch = np.std(template[ch])
            if std_epo_ch > 0 and std_tmpl_ch > 0:
                corr = np.corrcoef(epo[ch], template[ch])[0, 1]
            else:
                corr = 0
            corr_channels.append(corr)
        features.append(corr_channels)
    return np.nan_to_num(np.array(features))


def extract_statistical_features(epochs, fs=600):
    # Statistics between 200-600 ms of each channel (mean, std, max, min, median)
    start_idx = int(0.2 * fs)
    end_idx = int(0.6 * fs)

    features = []
    for epo in epochs:
        feat_epo = []
        for ch in range(epo.shape[0]):
            segment = epo[ch, start_idx:end_idx] if end_idx <= epo.shape[1] else epo[ch]
            feat_epo.extend([
                segment.mean(),
                segment.std(),
                segment.max(),
                segment.min(),
                np.median(segment)
            ])
        features.append(feat_epo)
    return np.array(features)



from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Extract features
X_temp = extract_temporal_features(all_clean_epochs)
X_tmpl = extract_template_features(all_clean_epochs, epoch_labels)
X_stat = extract_statistical_features(all_clean_epochs)

print("Feature shapes:", X_temp.shape, X_tmpl.shape, X_stat.shape)

# Normalize
scaler_temp = StandardScaler()
scaler_tmpl = StandardScaler()
scaler_stat = StandardScaler()

X_temp_norm = scaler_temp.fit_transform(X_temp)
X_tmpl_norm = scaler_tmpl.fit_transform(X_tmpl)
X_stat_norm = scaler_stat.fit_transform(X_stat)

# Prepare feature sets for two SVMs
X1 = np.hstack([X_temp_norm, X_stat_norm])
X2 = np.hstack([X_tmpl_norm, X_stat_norm])

y_array = np.array(epoch_labels)

# Train and validate with 5-fold Stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
svm1 = SVC(kernel='linear', probability=True)
svm2 = SVC(kernel='linear', probability=True)

auc1 = cross_val_score(svm1, X1, y_array, cv=cv, scoring='roc_auc')
auc2 = cross_val_score(svm2, X2, y_array, cv=cv, scoring='roc_auc')

print(f"Linear SVM (Temporal+Stats) AUC: {auc1.mean():.3f} ± {auc1.std():.3f}")
print(f"Linear SVM (Template+Stats) AUC: {auc2.mean():.3f} ± {auc2.std():.3f}")

# Fit models on full data for test prediction later
svm1.fit(X1, y_array)
svm2.fit(X2, y_array)



test_folder = f'{extract_path}/test'

all_test_epochs = []
all_test_epoch_ids = []

# Extract epochs from each test EEG file
for filename in sorted(os.listdir(test_folder)):
    prefix = filename.replace('Data_','').replace('.csv','')
    path = os.path.join(test_folder, filename)
    epochs = extract_epochs(path)
    
    all_test_epochs.extend(epochs)
    for i in range(len(epochs)):
        epoch_id = f"{prefix}_FB{i + 1:03d}"
        all_test_epoch_ids.append(epoch_id)

print(f"Extracted {len(all_test_epochs)} epochs from test data")



X_test_temp = extract_temporal_features(all_test_epochs)
X_test_tmpl = extract_template_features(all_test_epochs, [0]*len(all_test_epochs))  # unlabeled, default zeros
X_test_stat = extract_statistical_features(all_test_epochs)

# Normalize using training scalers
X_test_temp_norm = scaler_temp.transform(X_test_temp)
X_test_tmpl_norm = scaler_tmpl.transform(X_test_tmpl)
X_test_stat_norm = scaler_stat.transform(X_test_stat)

# Combine for prediction
X_test_1 = np.hstack([X_test_temp_norm, X_test_stat_norm])
X_test_2 = np.hstack([X_test_tmpl_norm, X_test_stat_norm])

probs1 = svm1.predict_proba(X_test_1)[:, 1]  # probability of class 1
probs2 = svm2.predict_proba(X_test_2)[:, 1]

# Average ensemble probabilities
final_probs = (probs1 + probs2) / 2


