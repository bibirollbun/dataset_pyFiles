import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import os
import torch


os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

gpus = tf.config.list_physical_devices('GPU')
if len(gpus)<=1: 
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f'Using {len(gpus)} GPU')
else: 
    strategy = tf.distribute.MirroredStrategy()
    print(f'Using {len(gpus)} GPUs')
    
MIX = True
if MIX:
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    print('Mixed precision enabled')
else:
    print('Using full precision')


sample_submission_df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv')
sample_submission_df.info()


sample_submission_df.head(5)


train_data = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
train_data.info()


train_data.head(20)


sns.countplot(x='expert_consensus', data=train_data)
plt.show()


class_names = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
class_name_to_index = {'Seizure' : 0 , 'LPD' : 1 , 
                       'LRDA' : 3 , 'GPD' : 2 , 
                       'GRDA' : 4 , 'Other' : 5}

plt.figure(figsize=(15, 10)) 

for i, class_name in enumerate(class_names):
    plt.subplot(2, 3, i+1) 
    sns.countplot(x=f'{class_name}_vote', data=train_data)
    plt.title(f'Distribution of {class_name} votes')
    plt.tight_layout()

plt.show()


train_data.hist(bins=10, figsize=(15, 20), layout=(7, 2))
plt.suptitle('Feature Distributions')
plt.show()


vote_columns = [f'{name}_vote' for name in class_names]
corr_matrix = train_data[vote_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='viridis')
plt.title('Correlation Matrix for Vote Columns')
plt.show()


plt.figure(figsize=(15, 10))

for i, col in enumerate([f'{name}_vote' for name in class_names]):
    plt.subplot(2, 3, i+1)
    sns.boxplot(y='expert_consensus', x=col, data=train_data)
    plt.title(f'Box Plot of {col} vs Expert Consensus')
    plt.tight_layout() 

plt.show()


eeg_dir = '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs'
spectrogram_dir = '/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms'
metadata_path = '/kaggle/input/hms-harmful-brain-activity-classification/train.csv'


def load_data(ids, file_dir):
    file_path = f"{file_dir}/{int(ids)}.parquet"
    data_df = pd.read_parquet(file_path)
    return data_df

def load_eeg_data(ids):
    return load_data(ids, eeg_dir)

def load_spectrogram_data(ids):
    return load_data(ids, spectrogram_dir).drop(columns=['time'])


df_eeg_example = load_eeg_data(1628180742)
df_eeg_example.info()


df_spectro_example = load_spectrogram_data(999431)
df_spectro_example.info()


df_spectro_example.columns


load_eeg_data(train_data['eeg_id'][190])


load_spectrogram_data(train_data['spectrogram_id'][28])


df_train = train_data.drop(columns=['eeg_sub_id','eeg_label_offset_seconds',
                         'spectrogram_sub_id','spectrogram_label_offset_seconds',
                         'label_id','patient_id'])

df_train = df_train.drop_duplicates().reset_index()
df_train.drop(columns=['index'], inplace=True)


df_train['total'] = df_train[vote_columns].sum(axis=1)
df_train[vote_columns] = df_train[vote_columns].div(df_train['total'], axis=0)
df_train.drop(columns=['total'], inplace=True)

df_train['expert_consensus'] = df_train['expert_consensus'].map(class_name_to_index)


df_train


df_train[vote_columns]


def preprocess(dataframe, eeg_dir, spectrogram_dir, vote_columns):
    eeg_features_list = []
    spectrogram_features_list = []
    labels_list = []

    for idx in range(len(dataframe)):
        eeg_id = dataframe.iloc[idx]['eeg_id']
        spectrogram_id = dataframe.iloc[idx]['spectrogram_id']

        eeg_data = load_data(eeg_id, eeg_dir)
        eeg_features = extract_features(eeg_data)
        eeg_features_list.append(eeg_features)

        spectrogram_data = load_data(spectrogram_id, spectrogram_dir).drop(columns=['time'])
        spectrogram_features = extract_features(spectrogram_data)
        spectrogram_features_list.append(spectrogram_features)

        label = dataframe.iloc[idx][vote_columns].values
        labels_list.append(label)

    eeg_features_tensor = torch.tensor(eeg_features_list, dtype=torch.float32)
    spectrogram_features_tensor = torch.tensor(spectrogram_features_list, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_list, dtype=torch.float32)

    return eeg_features_tensor, spectrogram_features_tensor, labels_tensor


def extract_features(df):
    current_size = len(df)

    # Basic statistical features
    min_values = df.min()
    max_values = df.max()
    mean_values = df.mean()
    std_values = df.std()

    # Time-domain features
    rms_values = np.sqrt(np.mean(np.square(df), axis=0))
    var_values = df.var()
    skew_values = df.skew()
    kurtosis_values = df.kurtosis()

    # Concatenate all features
    features = np.concatenate([
        min_values, max_values, mean_values, std_values, 
        rms_values, var_values, skew_values, kurtosis_values
    ])


    return features


eeg_features_tensor, spectrogram_features_tensor, labels_tensor = preprocess(df_train, eeg_dir, spectrogram_dir, vote_columns)

