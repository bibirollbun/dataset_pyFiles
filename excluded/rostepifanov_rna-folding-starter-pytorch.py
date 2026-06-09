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


from collections import defaultdict

def generate_label_coord(frame):
    output = defaultdict(list)

    frame['label'] = frame.ID.str.rsplit('_', n=1, expand=True).iloc[:, 0]

    for _, row in frame.iterrows():
        label = row['label']
        resid = row['resid']

        coord = np.array((row['x_1'], row['y_1'], row['z_1']), dtype=np.float32)
        output[label].append((resid, coord))

    for key, value in output.items():
        coords = np.stack([c for r, c in value])
        masks = np.isnan(coords) | np.isclose(coords, -1.0000e+18)

        coords[masks] = 0.

        output[key] = {
            'coords': coords,
            'masks': ~masks,
        }

    return output

train_stacked_coords = generate_label_coord(train_labels)
val_stacked_coords = generate_label_coord(val_labels)

train_stacked_coords[list(train_stacked_coords.keys())[0]]


def generate_dataset(seq, stacked_coords):
    X, y, my, tids = [], [], [], []

    for idx, row in seq.iterrows():
        tid = row['target_id']
        if tid in stacked_coords:
            X.append(row['encoded_seq'])
            y.append(stacked_coords[tid]['coords'])
            my.append(stacked_coords[tid]['masks'])
            tids.append(tid)

    return X, y, my, tids

train_X, train_y, train_my, train_tids = generate_dataset(train_sequence, train_stacked_coords)
val_X, val_y, val_my, val_tids = generate_dataset(val_sequence, val_stacked_coords)


import torch
from torch.nn.utils.rnn import pad_sequence

def pad_sequences_torch(sequences, max_len, padding='post', value=0):
    padded = pad_sequence([
        torch.cat([seq, torch.full((max_len - len(seq),), value, dtype=seq.dtype)])
        if len(seq) < max_len else seq[:max_len]
        for seq in sequences
    ], batch_first=True, padding_value=value)
    return padded

max_len = max(len(seq) for seq in train_X)

train_X_pad = pad_sequences_torch([torch.tensor(x) for x in train_X], max_len)
val_X_pad = pad_sequences_torch([torch.tensor(x) for x in val_X], max_len)
test_X = test_sequence['encoded_seq'].tolist()
test_X_pad = pad_sequences_torch([torch.tensor(x) for x in test_sequence['encoded_seq'].tolist()], max_len)

train_X_pad.shape


import torch.nn.functional as F

def pad_coords_torch(coords, max_len):
    L = coords.size(0)
    if L < max_len:
        pad = (0, 0, 0, max_len - L)  # (left, right, top, bottom)
        return F.pad(coords, pad, "constant", 0)
    else:
        return coords[:max_len]

train_y_pad = torch.stack([pad_coords_torch(torch.tensor(y), max_len) for y in train_y])
val_y_pad = torch.stack([pad_coords_torch(torch.tensor(y), max_len) for y in val_y])

train_my_pad = torch.stack([pad_coords_torch(torch.tensor(my), max_len) for my in train_my])
val_my_pad = torch.stack([pad_coords_torch(torch.tensor(my), max_len) for my in val_my])

train_y_pad.shape


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

class CNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=config['seq_mapping_size'],
            embedding_dim=config['embedding_dim'],
            padding_idx=0
        )
        
        # First Conv Block
        self.conv1 = nn.Conv1d(
            in_channels=config['embedding_dim'],
            out_channels=config['num_filters'],
            kernel_size=config['kernel_size'],
            padding='same'
        )
        self.bn1 = nn.BatchNorm1d(config['num_filters'])
        self.drop1 = nn.Dropout(config['drop_rate'])
        
        # Second Conv Block
        self.conv2 = nn.Conv1d(
            in_channels=config['num_filters'],
            out_channels=config['num_filters'],
            kernel_size=config['kernel_size'],
            padding='same'
        )
        self.bn2 = nn.BatchNorm1d(config['num_filters'])
        self.drop2 = nn.Dropout(config['drop_rate'])
        
        # Final Prediction Layer
        self.final_conv = nn.Conv1d(
            in_channels=config['num_filters'],
            out_channels=3,
            kernel_size=1,
            padding='same'
        )
        
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, x):
        # Embedding
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch, embedding_dim, seq_len)
        
        # First Conv Block
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.drop1(x)
        
        # Second Conv Block
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.drop2(x)
        
        # Final Prediction
        x = self.final_conv(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, 3)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y, my = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        loss = (my * loss).mean()

        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, my = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        loss = (my * loss).mean()

        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# Конфигурация модели
config = {
    'seq_mapping_size': max(seq_dict.values()) + 1,
    'embedding_dim': 16,
    'num_filters': 64,
    'kernel_size': 3,
    'drop_rate': 0.2,
    'train_epochs': 50,
    'batch_size': 16
}

# Подготовка данных
train_dataset = TensorDataset(train_X_pad, train_y_pad, train_my_pad.long())
val_dataset = TensorDataset(val_X_pad, val_y_pad, val_my_pad.long())

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

# Инициализация и обучение модели
model = CNN(config)
early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='min')

trainer = pl.Trainer(
    max_epochs=config['train_epochs'],
    callbacks=[early_stop],
    accelerator='auto',
    enable_progress_bar=True,
    log_every_n_steps=1,
)
trainer.fit(model, train_loader, val_loader)

# Предсказание на тестовых данных
test_tensor = torch.LongTensor(test_X_pad)
model.eval()
with torch.no_grad():
    pred = model(test_tensor).cpu().numpy()

print(pred.shape)


rows = []

for idx, row in test_sequence.iterrows():
    target_id = row['target_id']
    coords = pred[idx]
    seq_length = len(row['encoded_seq'])
    coords = coords[:seq_length, :]

    for i in range(seq_length):
        x, y, z = coords[i, :]
        rows.append(
            {
                'ID': f"{target_id}_{i+1}",
                'resname': row['sequence'][i],
                'resid': i+1,
                 **{f"x_{j+1}": x for j in range(5)},
                 **{f"y_{j+1}": y for j in range(5)},
                 **{f"z_{j+1}": z for j in range(5)}
            }
        )

submission = pd.DataFrame(rows)
submission.to_csv("submission.csv", index=False)

submission.shape

