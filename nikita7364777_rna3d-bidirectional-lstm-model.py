# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# Загрузка данных
train_seq = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
validation_seq = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv')
validation_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_labels.csv')
test_seq = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')


# EDA (Exploratory Data Analysis)
#1. Sequence Length Analysis
train_seq['sequence_length'] = train_seq['sequence'].apply(len)
validation_seq['sequence_length'] = validation_seq['sequence'].apply(len)
test_seq['sequence_length'] = test_seq['sequence'].apply(len)

plt.figure(figsize=(12, 6))
sns.histplot(train_seq['sequence_length'], bins=50, kde=True, label='Train')
plt.title('Distribution of Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
sns.histplot(validation_seq['sequence_length'], bins=50, kde=True, label='Validation')
plt.title('Distribution of Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
sns.histplot(test_seq['sequence_length'], bins=50, kde=True, label='Test')
plt.title('Distribution of Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.legend()
plt.show()


#2. Analysis of the nucleotide distribution
nucleotides = ['A', 'C', 'G', 'U']
train_nuc_counts = train_seq['sequence'].apply(lambda x: pd.Series([x.count(nuc) for nuc in nucleotides]))
validation_nuc_counts = validation_seq['sequence'].apply(lambda x: pd.Series([x.count(nuc) for nuc in nucleotides]))
test_nuc_counts = test_seq['sequence'].apply(lambda x: pd.Series([x.count(nuc) for nuc in nucleotides]))

train_nuc_counts.columns = nucleotides
validation_nuc_counts.columns = nucleotides
test_nuc_counts.columns = nucleotides


plt.figure(figsize=(12, 6))
train_nuc_counts.sum().plot(kind='bar', label='Train')
plt.title('Nucleotide Distribution')
plt.xlabel('Nucleotide')
plt.ylabel('Count')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
validation_nuc_counts.sum().plot(kind='bar', label='Validation', alpha=0.7)
plt.title('Nucleotide Distribution')
plt.xlabel('Nucleotide')
plt.ylabel('Count')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
test_nuc_counts.sum().plot(kind='bar', label='Test', alpha=0.7)
plt.title('Nucleotide Distribution')
plt.xlabel('Nucleotide')
plt.ylabel('Count')
plt.legend()
plt.show()


# 3. Timestamp Analysis (temporal_cutoff)
train_seq['temporal_cutoff'] = pd.to_datetime(train_seq['temporal_cutoff'])
validation_seq['temporal_cutoff'] = pd.to_datetime(validation_seq['temporal_cutoff'])
test_seq['temporal_cutoff'] = pd.to_datetime(test_seq['temporal_cutoff'])

plt.figure(figsize=(12, 6))
sns.histplot(train_seq['temporal_cutoff'], bins=50, kde=True, label='Train')
plt.title('Distribution of Temporal Cutoff Dates')
plt.xlabel('Temporal Cutoff')
plt.ylabel('Frequency')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
sns.histplot(validation_seq['temporal_cutoff'], bins=50, kde=True, label='Validation')
plt.title('Distribution of Temporal Cutoff Dates')
plt.xlabel('Temporal Cutoff')
plt.ylabel('Frequency')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
sns.histplot(test_seq['temporal_cutoff'], bins=50, kde=True, label='Test')
plt.title('Distribution of Temporal Cutoff Dates')
plt.xlabel('Temporal Cutoff')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# Adding missing characters to categories
nucleotides = ['A', 'C', 'G', 'U', 'N', '-', 'X']
max_len = max(
    train_seq['sequence'].apply(len).max(),
    validation_seq['sequence'].apply(len).max(),
    test_seq['sequence'].apply(len).max()
)
# OneHotEncoder with consideration of all characters
encoder = OneHotEncoder(sparse=False, categories=[nucleotides])
encoder.fit(np.array(nucleotides).reshape(-1, 1))


# Sequence preprocessing function
def preprocess_sequences(seq_list, max_len):
    encoded = []
    for seq in seq_list:
        padded = seq.ljust(max_len, 'N')[:max_len]  # Padding to max_len
        onehot = encoder.transform(np.array(list(padded)).reshape(-1, 1))
        encoded.append(onehot)
    return np.array(encoded)


train_seq['sequence']


# Data conversion
X_train = preprocess_sequences(train_seq['sequence'], max_len)
X_val = preprocess_sequences(validation_seq['sequence'], max_len)
X_test = preprocess_sequences(test_seq['sequence'], max_len)


# Fixed align_labels function
def align_labels(sequences_df, labels_df, max_len):
    aligned = []
    for target_id in sequences_df['target_id']:
        # Filtering labels for the current target_id
        target_labels = labels_df[labels_df['ID'].str.startswith(f"{target_id}_")]
        # Sort by balance number
        target_labels = target_labels.sort_values('resid')
        # Replace NaN in coordinates with 0 for Masking
        target_labels[['x_1', 'y_1', 'z_1']] = target_labels[['x_1', 'y_1', 'z_1']].fillna(-1)
        # Extracting coordinates
        coords = target_labels[['x_1', 'y_1', 'z_1']].values
        # Truncate to max_len
        if len(coords) > max_len:
            coords = coords[:max_len]
        padded_coords = np.zeros((max_len, 3))
        padded_coords[:len(coords)] = coords
        aligned.append(padded_coords)
    return np.array(aligned)


y_train = align_labels(train_seq, train_labels, max_len)
y_val = align_labels(validation_seq, validation_labels, max_len)


# Checking the dimensions
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")


# Checking for NaN and infinity in data
print("NaN in X_train:", np.isnan(X_train).any())
print("Infinity in X_train:", np.isinf(X_train).any())
print("NaN in y_train:", np.isnan(y_train).any())
print("Infinity in y_train:", np.isinf(y_train).any())


y_val.min(), y_val.max()


y_val.shape


count_below_threshold = np.sum(y_val < -1e+4)
print(f"The number of values is less than -10^10: {count_below_threshold}")


median_value = np.median(y_val[y_val >= -1e+4])
y_val[y_val < -1e+4] = -1


y_val.min(), y_val.max()


'''
# Normalization of data
from sklearn.preprocessing import StandardScaler

# Scaling the input data (X)
scaler_X = StandardScaler()
X_train_flat = X_train.reshape(-1, X_train.shape[-1])
X_train_scaled_flat = scaler_X.fit_transform(X_train_flat)
X_train_scaled = X_train_scaled_flat.reshape(X_train.shape)

X_val_flat = X_val.reshape(-1, X_val.shape[-1])
X_val_scaled_flat = scaler_X.transform(X_val_flat)
X_val_scaled = X_val_scaled_flat.reshape(X_val.shape)

# Scaling the output (y)
scaler_y = StandardScaler()
y_train_flat = y_train.reshape(-1, y_train.shape[-1])
y_train_scaled_flat = scaler_y.fit_transform(y_train_flat)
y_train_scaled = y_train_scaled_flat.reshape(y_train.shape)

y_val_flat = y_val.reshape(-1, y_val.shape[-1])
y_val_scaled_flat = scaler_y.transform(y_val_flat)
y_val_scaled = y_val_scaled_flat.reshape(y_val.shape)
'''


np.sum(X_train == -1), np.sum(X_val == -1), np.sum(y_train == -1), np.sum(y_val == -1), 


# Building a model
inputs = Input(shape=(max_len, len(nucleotides)))

x = Masking(mask_value=-1)(inputs)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Dropout(0.1)(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
outputs = Dense(3)(x)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(0.001), loss='mse')
model.summary()

# Model training
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=1,
                    verbose=1)

# Visualization of learning
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()


# Prediction generation
preds = model.predict(X_test)


# Formation of the submission file
submission_rows = []
for i, (_, row) in enumerate(test_seq.iterrows()):
    seq_len = len(row.sequence)
    for res_idx in range(seq_len):
        coords = preds[i][res_idx].tolist()
        # Duplicate for 5 models
        all_coords = coords * 5  
        submission_rows.append([
            f"{row.target_id}_{res_idx+1}",
            row.sequence[res_idx],
            res_idx+1,
            *all_coords
        ])


# Creating a DataFrame
columns = ['ID', 'resname', 'resid'] + [f'{c}_{i+1}' for i in range(5) for c in ['x', 'y', 'z']]
submission = pd.DataFrame(submission_rows, columns=columns)
submission.to_csv('submission.csv', index=False)
print("Submission file created!")


# A list for storing data of each model
models_data = []

for model_num in range(1, 6):
    # Selecting columns for the current model
    cols = ['ID', 'resname', 'resid'] + [f'{c}_{model_num}' for c in ['x', 'y', 'z']]
    model_data = submission[cols].copy()
    model_data = submission[cols].copy()
    # Rename columns for convenience
    model_data.columns = ['ID', 'resname', 'resid', 'x', 'y', 'z']
    # Adding a model number
    model_data['model'] = f'Model {model_num}'
    models_data.append(model_data)
# Combining data from all models
all_models_data = pd.concat(models_data, ignore_index=True)


import plotly.express as px
fig = px.scatter_3d(all_models_data,
                    x='x',
                    y='y',
                    z='z',
                    color='z',
                    color_continuous_scale='rainbow',
                    title='3D Distribution of Predicted Points',
                    labels={'x': 'X Coordinate', 'y': 'Y Coordinate', 'z': 'Z Coordinate'},
                    opacity=0.7,
                    hover_name='ID',
                    hover_data=['resname', 'resid'])

fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y',zaxis_title='Z',),
                  margin=dict(l=0, r=0, b=0, t=30),
                  legend=dict(title='Model', x=0.8, y=0.9))

fig.show()




