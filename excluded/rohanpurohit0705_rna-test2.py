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


from sklearn.preprocessing import LabelEncoder

train_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
test_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
sample_submission = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/sample_submission.csv")


unique_chars = set("".join(train_sequences["sequence"].dropna().values))
print("Unique RNA bases:", unique_chars)

base_encoder = LabelEncoder()
base_encoder.fit(['A', 'U', 'G', 'C', 'N'])

def clean_and_encode_sequence(seq):
    clean_seq = seq.replace("-", "").replace("X", "N") 
    return np.array(base_encoder.transform(list(clean_seq)))  

train_sequences["encoded_seq"] = train_sequences["sequence"].apply(clean_and_encode_sequence)

train_sequences.head()


print("Sample target_id values (train_sequences):", train_sequences["target_id"].unique()[:5])
print("Sample ID values (train_labels):", train_labels["ID"].unique()[:5])


train_labels["target_base"] = train_labels["ID"].apply(lambda x: "_".join(x.split("_")[:2]))

merged_data = train_sequences.merge(
    train_labels, left_on="target_id", right_on="target_base"
)

merged_data.drop(columns=["target_base"], inplace=True)

merged_data.head()


label_counts = merged_data.groupby("target_id").size()

print(label_counts.head())


Y_grouped = merged_data.groupby("target_id")[["x_1", "y_1", "z_1"]].apply(lambda x: x.values)

Y = np.array(Y_grouped.tolist(), dtype=object)

print(f"Before padding: {Y.shape}")  


X_grouped = merged_data.groupby("target_id")["encoded_seq"].apply(lambda x: list(x)).tolist()

X_grouped = [np.concatenate(seq).tolist() for seq in X_grouped]
Y_grouped = merged_data.groupby("target_id")[["x_1", "y_1", "z_1"]].apply(lambda x: x.values).tolist()

Y_grouped = [np.array(seq).tolist() for seq in Y_grouped]


max_seq_length = train_sequences.groupby("target_id")["sequence"].apply(len).max()
print(f"Max sequence length: {max_seq_length}")


from tensorflow.keras.preprocessing.sequence import pad_sequences

X_padded = pad_sequences(X_grouped, maxlen=max_seq_length, padding="post", dtype="float32")

Y_padded = pad_sequences(Y_grouped, maxlen=max_seq_length, padding="post", dtype="float32")

Y_padded = np.array(Y_padded)

print(f"X shape: {X_padded.shape}, Y shape: {Y_padded.shape}")


from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_padded, Y_padded, test_size=0.2, random_state=42)

print(f"Train Shape: X={X_train.shape}, Y={Y_train.shape}")
print(f"Validation Shape: X={X_val.shape}, Y={Y_val.shape}")


print("NaNs in X_padded:", np.isnan(X_padded).sum())
print("NaNs in Y_padded:", np.isnan(Y_padded).sum())

print("Infs in X_padded:", np.isinf(X_padded).sum())
print("Infs in Y_padded:", np.isinf(Y_padded).sum())


nan_rows = np.isnan(Y_padded).sum(axis=1) > 0

print(f"Number of sequences with NaNs: {nan_rows.sum()} out of {len(Y_padded)}")


for i in range(Y_padded.shape[1]):  
    for j in range(3):  
        nan_mask = np.isnan(Y_padded[:, i, j])
        Y_padded[nan_mask, i, j] = np.nanmean(Y_padded[:, :, j])  


print(f"Shape of X_padded: {X_padded.shape}")


print(train_sequences.head())


max_seq_length = train_sequences["sequence"].apply(len).max()
print(f"Max sequence length: {max_seq_length}")

X_padded = pad_sequences(
    train_sequences["encoded_seq"].tolist(),  
    maxlen=max_seq_length,  
    padding="post",  
    dtype="float32"
)

print(f"Updated X_padded shape: {X_padded.shape}")


X_padded = np.expand_dims(X_padded, axis=-1)  
print(f"Final X_padded shape: {X_padded.shape}") 


print(train_labels.columns)


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense

# Extract features (RNA sequences)
X_raw = train_sequences["sequence"].apply(lambda x: [ord(c) for c in x])  # Convert to numerical values
max_seq_length = max(X_raw.apply(len))  # Find longest sequence

# Pad RNA sequences to ensure uniform shape
X_padded = pad_sequences(X_raw, maxlen=max_seq_length, padding="post", dtype="float32")

# Extract labels
Y_padded = np.array(train_labels[["x_1", "y_1", "z_1"]].fillna(0))  # Ensure correct columns & no NaNs

# Reshape X to fit CNN input
X_padded = np.expand_dims(X_padded, axis=-1)  # Shape (samples, max_seq_length, 1)



# Define CNN Model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(X_padded.shape[1], 1)),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(3)  # Predicting (x_1, y_1, z_1) only
])

# Compile Model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Train Model
model.fit(X_padded, Y_padded, epochs=10, batch_size=32, validation_split=0.1)


# Preprocess test data
X_test_raw = test_sequences["sequence"].apply(lambda x: [ord(c) for c in x])  # Convert sequences to numbers
X_test_padded = pad_sequences(X_test_raw, maxlen=max_seq_length, padding="post", dtype="float32")
X_test_padded = np.expand_dims(X_test_padded, axis=-1)  # Shape (samples, max_seq_length, 1)

# Make Predictions
predictions = model.predict(X_test_padded)


# Ensure submission matches required format
submission_df = pd.DataFrame({
    "ID": test_sequences["target_id"],  
    "resname": "G",  # Assuming all residues are 'G'
    "resid": range(1, len(test_sequences) + 1),
    "x_1": predictions[:, 0], "y_1": predictions[:, 1], "z_1": predictions[:, 2],
    "x_2": predictions[:, 0], "y_2": predictions[:, 1], "z_2": predictions[:, 2],  # Duplicate for missing data
    "x_3": predictions[:, 0], "y_3": predictions[:, 1], "z_3": predictions[:, 2],
    "x_4": predictions[:, 0], "y_4": predictions[:, 1], "z_4": predictions[:, 2],
    "x_5": predictions[:, 0], "y_5": predictions[:, 1], "z_5": predictions[:, 2],
})

# Save to CSV
submission_df.to_csv("submission.csv", index=False)
print("Submission saved successfully!")
submission_df.head()

