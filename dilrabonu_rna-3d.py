import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, BatchNormalization, Dropout
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
np.random.seed(42)
tf.random.set_seed(42)


train_seq = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
train_lab = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
test_seq = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
val_lab = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")
val_seq = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv")
sub_sample = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/sample_submission.csv")




dfs = [train_seq, train_lab, test_seq, val_lab, val_seq, sub_sample]

for i, df in enumerate(dfs):
    print(f"DataFrame {i+1} Info:")
    print(df.info())
    print("\n" + "="*50 + "\n")



train_lab.isnull().sum()


train_lab.head()


for i, df in enumerate(dfs):
    print(f"DataFrame {i+1} Info:")
    print(df.head())
    print("\n" + "="*50 + "\n")



for i, df in enumerate(dfs):
    print(f"DataFrame {i+1} Missing Values:")
    print(df.isnull().sum())  # Check for missing values
    print("\n" + "="*50 + "\n")



train_lab[['x_1', 'y_1', 'z_1']] = train_lab[['x_1', 'y_1', 'z_1']].fillna(train_lab[['x_1', 'y_1', 'z_1']].median())


for i, df in enumerate(dfs):
    print(f"DataFrame {i+1} Missing Values:")
    print(df.isnull().sum())  # Check for missing values
    print("\n" + "="*50 + "\n")


train_lab[['x_1', 'y_1', 'z_1']].hist(bins=30, figsize=(10, 5))
plt.show()


for i, df in enumerate(dfs):
    print(f"DataFrame {i+1} Duplicates:")
    print(df.duplicated().sum())  
    print("\n" + "="*50 + "\n")


from collections import Counter

def kmer_freq(sequence, k=3):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)

train_seq['kmer_features'] = train_seq['sequence'].apply(lambda x: kmer_freq(x, k=3))



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X = train_lab.drop(columns=['ID', 'resname', 'resid'])
y = train_lab[['x_1', 'y_1', 'z_1']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f'MSE: {mean_squared_error(y_test, predictions)}')



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout

cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3)  # Predicting x, y, z coordinates
])

cnn_model.compile(optimizer='adam', loss='mse')
cnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))





