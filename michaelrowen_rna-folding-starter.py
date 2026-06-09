import numpy as np
import pandas as pd


train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
train_sequence = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
val_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")
val_sequence = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv")
test_sequence = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")

print("Train Seq: " + str(train_sequence.shape))
print("Train Label: " + str(train_labels.shape))
print("Validation Seq: " + str(val_sequence.shape))
print("Validation Label: " + str(val_labels.shape))
print("Test: "+str(test_sequence.shape))


train_labels.head()


train_labels.fillna(0, inplace=True)
val_labels.fillna(0, inplace=True)
train_labels.info()


import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'iframe'

plot_labels = train_labels[['ID', 'x_1', 'y_1', 'z_1']].copy()
plot_labels['label'] = plot_labels.ID.str.rsplit('_', n=1, expand=True).iloc[:,0]

target = '1SCL_A'

fig = px.line_3d(
    plot_labels.loc[plot_labels.label==target],
    x='x_1', y='y_1', z='z_1',
    title=f'{target}'
)

fig.show()


train_sequence.head()


val_labels.head()


seq_dict = {'A': 1, 'C': 2, 'G': 3, 'U': 4}
def seq_map(seq):
    return [seq_dict.get(char, 0) for char in seq]

train_sequence['encoded_seq'] = train_sequence['sequence'].apply(seq_map)
test_sequence['encoded_seq'] = test_sequence['sequence'].apply(seq_map)
val_sequence['encoded_seq'] = val_sequence['sequence'].apply(seq_map)


def generate_label_coord(df):
    result = {}
    df["label"] = df.ID.str.rsplit('_', n=1, expand=True).iloc[:,0]
    for _, row in df.iterrows():
        label = row['label']
        resid = row['resid']
        if label not in result:
            result[label] = []
        else:
            coord = np.array([row['x_1'], row['y_1'], row['z_1']], dtype=np.float32)
            result[label].append((resid, coord))
    for key in result:
        coords = np.stack([c for r, c in result[key]])
        result[key] = coords
    return result

train_stacked_coords = generate_label_coord(train_labels)
val_stacked_coords = generate_label_coord(val_labels)
train_stacked_coords[list(train_stacked_coords.keys())[0]]


def generate_dataset(seq, stacked_coords):
    X, y, tids = [], [], []
    for idx, row in seq.iterrows():
        tid = row['target_id']
        if tid in stacked_coords:
            X.append(row['encoded_seq'])
            y.append(stacked_coords[tid])
            tids.append(tid)
    return X, y, tids

train_X, train_y, train_tids = generate_dataset(train_sequence, train_stacked_coords)
val_X, val_y, val_tids = generate_dataset(val_sequence, val_stacked_coords)



import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

max_len = max(len(seq) for seq in train_X)

train_X_pad = pad_sequences(train_X, maxlen=max_len, padding='post', value=0)
val_X_pad = pad_sequences(val_X, maxlen=max_len, padding='post', value=0)
test_X = test_sequence['encoded_seq'].tolist()
test_X_pad = pad_sequences(test_X, maxlen=max_len, padding='post', value=0)
train_X_pad.shape


def pad_coords(coords, max_len):
    L = coords.shape[0]
    if L < max_len:
        pad_width = ((0, max_len-L), (0, 0)) # pad only vertically
        return np.pad(coords, pad_width, mode='constant', constant_values = 0)
    else:
        return coords

train_y_pad = np.array([pad_coords(y, max_len) for y in train_y])
val_y_pad = np.array([pad_coords(y, max_len) for y in val_y])
train_y_pad.shape


class CNN:
    def __init__(self, X_train_pad, y_train_pad, X_val_pad, y_val_pad):
        self.x_train = X_train_pad
        self.y_train = y_train_pad
        self.x_val = X_val_pad
        self.y_val = y_val_pad
        self.config = {
            'seq_mapping_size':max(seq_dict.values()) + 1,
            'embedding_dim':16,
            'num_filters':64,
            'kernel_size':3,
            'drop_rate':0.2,
            'train_epochs':50,
            'batch_size':16
        }
        self.model= None
    def first_conv(self, max_length):
        input_seq = Input(shape=(max_len,), name='input_seq')
        x_cnn = Embedding(input_dim=self.config["seq_mapping_size"],
                         output_dim=self.config["embedding_dim"],
                         mask_zero=True,
                         name='embedding')(input_seq)
        x_cnn = Conv1D(filters=self.config["num_filters"],
                      kernel_size=self.config["kernel_size"],
                      padding='same',
                      activation='relu',
                      name='conv1')(x_cnn)
        x_cnn = BatchNormalization(name='norm1')(x_cnn)
        x_cnn = Dropout(self.config['drop_rate'], name='drop1')(x_cnn)
        
        return input_seq, x_cnn
        
    def second_conv_full_c(self, x_cnn):
        x_cnn = Conv1D(filters=self.config["num_filters"],
                      kernel_size=self.config["kernel_size"],
                      padding='same',
                      activation='relu',
                      name='conv2')(x_cnn)
        x_cnn = BatchNormalization(name='norm2')(x_cnn)
        x_cnn = Dropout(self.config['drop_rate'], name='drop2')(x_cnn)
        full_c = Conv1D(filters=3, kernel_size=1,
                       padding='same',
                       activation='linear',
                       name='predicted_coords')(x_cnn)
        return full_c
    def build_model(self, max_length):
        input_seq, x_cnn = self.first_conv(max_length)
        full_c = self.second_conv_full_c(x_cnn)
        model = Model(inputs=input_seq,
                     outputs=full_c)
        model.compile(optimizer='adam', loss='mae')
        self.model=model
        return model.summary()
        
    def train(self):
        history_cnn = self.model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_val, self.y_val),
            epochs=self.config["train_epochs"],
            batch_size=self.config["batch_size"],
            verbose=1
        )
        return self.model


cnn = CNN(train_X_pad, train_y_pad, val_X_pad, val_y_pad)
cnn.build_model(max_len)
model = cnn.train()


pred = model.predict(test_X_pad)
pred.shape


submission_rows = []
for idx, row in test_sequence.iterrows():
    target_id = row['target_id']
    coords = pred[idx]
    seq_length = len(row['encoded_seq'])
    coords = coords[:seq_length, :]
    for i in range(seq_length):
        x, y, z = coords[i, :]
        submission_rows.append(
        {
            'ID':f"{target_id}_{i+1}",
            'resname':row['sequence'][i],
            'resid':i+1,
             **{f"x_{j+1}": x for j in range(5)},
             **{f"y_{j+1}": y for j in range(5)},
             **{f"z_{j+1}": z for j in range(5)}
            
        }
        )
submission_df = pd.DataFrame(submission_rows)
submission_df.shape


submission_df.to_csv("submission.csv", index=False)

