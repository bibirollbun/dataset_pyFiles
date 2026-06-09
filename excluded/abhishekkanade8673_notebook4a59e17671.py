import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, BatchNormalization, Dropout, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error
import warnings
warnings.simplefilter(action='ignore')


train_sequence = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
val_sequence = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv')
val_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_labels.csv')
test_sequence = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
print(train_sequence.info())
print(train_labels.info())
print(val_sequence.info())
print(val_labels.info())
print(test_sequence.info())
# Fill missing values
train_labels.fillna(0, inplace=True)
val_labels.fillna(0, inplace=True)


duplicate_sequences = train_sequence[train_sequence.duplicated('sequence', keep=False)]
print(f"Number of duplicate sequences in train_sequences: {len(duplicate_sequences)}")

# Check for duplicate target_ids in train_labels
duplicate_targets = train_labels[train_labels.duplicated('ID', keep=False)]
print(f"Number of duplicate targets in train_labels: {len(duplicate_targets)}")


# Extract coordinates for a sample target
sample_target = train_labels
x = sample_target['x_1'].values
y = sample_target['y_1'].values
z = sample_target['z_1'].values

# Plot 3D structure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='blue', marker='o')
ax.set_title('3D RNA Structure')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


# Check for missing values
print("Missing values in train_sequences:")
print(train_sequence.isnull().sum())

print("\nMissing values in train_labels:")
print(train_labels.isnull().sum())


# Encoding Sequence
seq_cod = {'A': 1,
           'C': 2,
           'G': 3, 
           'U': 4
          }
def map_seq(seq):
    return [seq_cod.get(char, 0) for char in seq]

train_sequence['encoded_seq'] = train_sequence['sequence'].apply(map_seq)
test_sequence['encoded_seq'] = test_sequence['sequence'].apply(map_seq)
val_sequence['encoded_seq'] = val_sequence['sequence'].apply(map_seq)


def labels_coord_generator(df):
    result = {}
    df["label"] = df.ID.str.rsplit('_', n=1, expand=True).iloc[:,0]
    for _, row in df.iterrows():
        label = row['label']
        resid = row['resid']
        if label not in result:
            result[label] = []
        

        if all(col in row for col in ['x_1', 'y_1', 'z_1']):
            coords = np.array([
                [row['x_1'], row['y_1'], row['z_1']],
                [row['x_1'], row['y_1'], row['z_1']], 
                [row['x_1'], row['y_1'], row['z_1']],  
                [row['x_1'], row['y_1'], row['z_1']],  
                [row['x_1'], row['y_1'], row['z_1']]   
            ], dtype=np.float32)
        else:
            # If no coordinates are available, put zeros
            coords = np.zeros((5, 3), dtype=np.float32)
        
        result[label].append((resid, coords))
    
    for key in result:
        coords = np.stack([c for r, c in result[key]])
        result[key] = coords
    
    return result

train_coords = labels_coord_generator(train_labels)
val_coords = labels_coord_generator(val_labels)


def dataset_generator(seq, stacked_coords):
    X, y, tids = [], [], []
    for idx, row in seq.iterrows():
        tid = row['target_id']
        if tid in stacked_coords:
            X.append(row['encoded_seq'])
            y.append(stacked_coords[tid])
            tids.append(tid)
    return X, y, tids

train_X, train_y, train_tids = dataset_generator(train_sequence, train_coords)
val_X, val_y, val_tids = dataset_generator(val_sequence, val_coords)

max_len = max(len(seq) for seq in train_X)
train_X_pad = pad_sequences(train_X, maxlen=max_len, padding='post', value=0)
val_X_pad = pad_sequences(val_X, maxlen=max_len, padding='post', value=0)
test_X = test_sequence['encoded_seq'].tolist()
test_X_pad = pad_sequences(test_X, maxlen=max_len, padding='post', value=0)

def coord_padding(coords, max_len):
    L = coords.shape[0]
    if L < max_len:
        pad_width = ((0, max_len-L), (0, 0), (0, 0))
        return np.pad(coords, pad_width, mode='constant', constant_values=0)
    else:
        return coords

train_y_pad = np.array([coord_padding(y, max_len) for y in train_y])
val_y_pad = np.array([coord_padding(y, max_len) for y in val_y])


max_len


import tensorflow.keras.backend as K
def mish(x):
    return x * K.tanh(K.softplus(x))


# CNN Model

input_seq = Input(shape=(max_len,), name='input_seq')
x = Embedding(input_dim=5, output_dim=16, mask_zero=False, name='embedding')(input_seq)
x = Conv1D(filters=64, kernel_size=3, padding='same', activation=mish, name='conv1')(x)
x = BatchNormalization(name='norm1')(x)
x = Dropout(0.2, name='drop1')(x)
x = Conv1D(filters=128, kernel_size=3, padding='same', activation=mish, name='conv2')(x)
x = BatchNormalization(name='norm2')(x)
x = Dropout(0.2, name='drop2')(x)
x = Conv1D(filters=128, kernel_size=3, padding='same', activation=mish, name='conv3')(x)
x = BatchNormalization(name='norm3')(x)
x = Dropout(0.2, name='drop3')(x)
x = Conv1D(filters=256, kernel_size=3, padding='same', activation=mish, name='conv4')(x)
x = BatchNormalization(name='norm4')(x)
x = Dropout(0.2, name='drop4')(x)    
# Output 15 values per residue (5 sets of x, y, z coordinates)
x = Conv1D(filters=15, kernel_size=1, padding='same', activation='linear', name='predicted_coords')(x)
model = Model(inputs=input_seq, outputs=x)
model.compile(optimizer='adam', loss='mae')
   


# Train CNN
model_history = model.fit(
    train_X_pad, train_y_pad.reshape(train_y_pad.shape[0], train_y_pad.shape[1], -1),
    validation_data=(val_X_pad, val_y_pad.reshape(val_y_pad.shape[0], val_y_pad.shape[1], -1)),
    epochs=50, batch_size=32, verbose=1
)


model.summary()


def eval_model(model, X, y):
    preds = model.predict(X)
    preds = preds.reshape(preds.shape[0], preds.shape[1], 5, 3)  # Reshape to [batch_size, seq_len, 5, 3]
    rmse = np.sqrt(mean_squared_error(y.reshape(-1), preds.reshape(-1)))
    return rmse

cnn_rmse = eval_model(model, val_X_pad, val_y_pad)

print(f"CNN Validation RMSE: {cnn_rmse}")


preds = model.predict(test_X_pad)
preds=preds.reshape(preds.shape[0],preds.shape[1],5,3)


#Creating Submission file
submission_rows = []
for idx, row in test_sequence.iterrows():
    target_id = row['target_id']
    coords = preds[idx]  
    seq_length = len(row['encoded_seq'])
    coords = coords[:seq_length, :, :]  
    for i in range(seq_length):
        x_coords = coords[i, :, 0]  # x_1, x_2, x_3, x_4, x_5
        y_coords = coords[i, :, 1]  # y_1, y_2, y_3, y_4, y_5
        z_coords = coords[i, :, 2]  # z_1, z_2, z_3, z_4, z_5
        submission_rows.append({
            'ID': f"{target_id}_{i+1}",
            'resname': row['sequence'][i],
            'resid': i+1,
            'x_1': x_coords[0], 'x_2': x_coords[1], 'x_3': x_coords[2], 'x_4': x_coords[3], 'x_5': x_coords[4],
            'y_1': y_coords[0], 'y_2': y_coords[1], 'y_3': y_coords[2], 'y_4': y_coords[3], 'y_5': y_coords[4],
            'z_1': z_coords[0], 'z_2': z_coords[1], 'z_3': z_coords[2], 'z_4': z_coords[3], 'z_5': z_coords[4]})

submission = pd.DataFrame(submission_rows)
submission.to_csv("submission.csv", index=False)
print("Submission file is created")

