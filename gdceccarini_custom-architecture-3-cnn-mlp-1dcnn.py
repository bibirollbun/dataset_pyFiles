from tqdm import tqdm

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


base_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data'
data = pd.read_csv(f'{base_path}/train.csv')

total_subjects = 81

# Split in train, val, test tramite gli id degli individui: 10 (12%) per il test, 5 (6%) per la val e il 66 (81%) per il train 

test_subjects = sorted(data.subject.value_counts().index)[-10:]
val_subjects = sorted(data.subject.value_counts().index)[-15:-10]
train_subjects = sorted(data.subject.value_counts().index)[:-15]
print(len(test_subjects), len(val_subjects), len(train_subjects))

train = data.loc[data['subject'].isin(train_subjects)]
val = data.loc[data['subject'].isin(val_subjects)]
test = data.loc[data['subject'].isin(test_subjects)]

del data

print(train.shape, test.shape)


def get_dataset(data):

    not_features_col = ["row_id","sequence_id", "sequence_type","sequence_counter","subject","orientation","behavior","phase","gesture"]
    features_col = [c for c in train.columns if c not in not_features_col]
    target = 'gesture'

    n_gestures = 18
    
    gestures_list = ['Forehead - pull hairline', 'Cheek - pinch skin',
       'Write name on leg', 'Feel around in tray and pull out an object',
       'Neck - scratch', 'Eyelash - pull hair', 'Eyebrow - pull hair',
       'Forehead - scratch', 'Above ear - pull hair', 'Wave hello',
       'Write name in air', 'Neck - pinch skin', 'Text on phone',
       'Pull air toward your face', 'Pinch knee/leg skin',
       'Scratch knee/leg skin', 'Drink from bottle/cup', 'Glasses on/off']

    target_map = {g: i for i, g in enumerate(gestures_list)}

    
    X_ = []
    y_ = []
    for sequence in tqdm(data.sequence_id.unique()):
        sequence_data = data.query(f"sequence_id == '{sequence}'")
        sequence_X = sequence_data[features_col]
        X_.append(sequence_X)
        
        # y = sequence_data[target].apply(lambda x: target_map.get(x))
        # y_train_one_hot = keras.utils.to_categorical(y, n_gestures)[-1]  # non serve one_hot

        y = sequence_data[target]
        y_.append(target_map.get(y.iloc[-1]))
    return X_, y_


def get_tof(df_list, print_first_img=False):
    tof_data_list = []
    for df in tqdm(df_list):
        tof_cols = [c for c in df.columns if 'tof' in c]
        tof_data = np.reshape((df[tof_cols].values/255).astype(float), (df.shape[0], len(tof_cols)//8, 8))
        tof_data_list.append(tof_data)
        if print_first_img:
            plt.imshow(tof_data[0])
    return tof_data_list


def get_sensors(df_list):
    """
    Return all the sensors as table
    """
    sens_list = []
    for df in tqdm(df_list):
        sens = df[[c for c in df.columns if 'acc' in c or 'rot' in c or 'thm' in c]]
        sens_list.append(sens)
    return sens_list


X_train, y_train = get_dataset(train)
X_val, y_val = get_dataset(val)
X_test, y_test = get_dataset(test)

del train
del val
del test


scaler = StandardScaler()
scaler.fit(pd.concat(X_train))

X_train = [pd.DataFrame(scaler.transform(e), columns=e.columns, index=e.index).fillna(0) for e in X_train]
X_val   = [pd.DataFrame(scaler.transform(e), columns=e.columns, index=e.index).fillna(0) for e in X_val]
X_test  = [pd.DataFrame(scaler.transform(e), columns=e.columns, index=e.index).fillna(0) for e in X_test]


class GestureSequenceDataset(Dataset):
    def __init__(self, tof_sequences, sensor_sequences, targets):
        self.tof_sequences = tof_sequences  # list of (T, H, W)
        self.sensor_sequences = sensor_sequences  # list of (T, F)
        self.targets = targets  # list of int

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        tof = torch.tensor(self.tof_sequences[idx], dtype=torch.float32)  # (T, H, W)
        sens = torch.tensor(self.sensor_sequences[idx].values, dtype=torch.float32)  # (T, F)
        label = torch.tensor(self.targets[idx], dtype=torch.long)
        return tof, sens, label


class CNNFeatureExtractor(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            
            nn.Linear(32 * 10 * 2, out_dim),  # assuming input 40x8 → pool to 10x5
            nn.ReLU()
        )

    def forward(self, x):  # x: (B*T, 1, H, W)
        return self.cnn(x)


class SensorMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # print(f"in_dim: {in_dim}, out_dim: {out_dim}")
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU()
        )

    def forward(self, x):  # (B*T, F)
        return self.mlp(x)


class CNN1D(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, 
                               out_channels=64, 
                               kernel_size=5, 
                               padding=2)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(in_channels=64, 
                               out_channels=output_size, 
                               kernel_size=5, 
                               padding=2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(output_size)

        self.global_pool = nn.AdaptiveAvgPool1d(1)  # output: (batch, channels, 1)



class GestureModel(nn.Module):
    def __init__(self, img_feat_dim, sens_feat_dim, cnn1d_out=128, num_classes=18):
        super().__init__()
        self.cnn_extractor = CNNFeatureExtractor(img_feat_dim)
        self.sens_mlp = SensorMLP(in_dim=sens_feat_dim, out_dim=img_feat_dim)
        self.fusion = nn.Linear(img_feat_dim + img_feat_dim, 10)  # compress to 10D
        self.cnn1d = CNN1D(input_size=10, output_size=cnn1d_out)
        self.classifier = nn.Linear(cnn1d_out, num_classes)

    def forward(self, x_tof, x_sens):
        B, T, H, W = x_tof.shape
        x_tof = x_tof.view(B*T, 1, H, W)
        x_sens = x_sens.view(B*T, -1)

        tof_feat = self.cnn_extractor(x_tof)
        sens_feat = self.sens_mlp(x_sens)

        fused = torch.cat([tof_feat, sens_feat], dim=1)
        compressed = self.fusion(fused)
        compressed = compressed.view(B, T, -1)

        conv = compressed.permute(0, 2, 1)  # → (batch, features, seq_len) for Conv1d
        conv = self.cnn1d.relu1(self.cnn1d.bn1(self.cnn1d.conv1(conv)))
        conv = self.cnn1d.relu2(self.cnn1d.bn2(self.cnn1d.conv2(conv)))
        conv = self.cnn1d.global_pool(conv).squeeze(-1)  # → (batch, channels)
        return self.classifier(conv)


# # 1dCNN collate
# def collate_fn(batch):
#     sequences, labels = zip(*batch)
#     lengths = torch.tensor([len(seq) for seq in sequences])
#     padded_sequences = pad_sequence(sequences, batch_first=True)
#     labels = torch.tensor(labels, dtype=torch.long)
#     return padded_sequences, lengths, labels

def collate_fn(batch):
    tof_seqs, sens_seqs, labels = zip(*batch)

    tof_padded = pad_sequence([torch.tensor(x) for x in tof_seqs], batch_first=True)  # (B, T, H, W)
    sens_padded = pad_sequence([torch.tensor(x) for x in sens_seqs], batch_first=True)  # (B, T, F)
    labels = torch.tensor(labels)

    return tof_padded, sens_padded, labels


BATCH_SIZE = 32
NUM_FEATURES = X_train[0].shape[1]
NUM_CLASSES = 18


train_dataset = GestureSequenceDataset(get_tof(X_train), 
                                       get_sensors(X_train), 
                                       y_train)
val_dataset   = GestureSequenceDataset(get_tof(X_val), 
                                       get_sensors(X_val), 
                                       y_val)
test_dataset   = GestureSequenceDataset(get_tof(X_test), 
                                        get_sensors(X_test), 
                                        y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


def get_eval_metric(model, data_loader):

    gestures_list = ['Forehead - pull hairline', 'Cheek - pinch skin',
       'Write name on leg', 'Feel around in tray and pull out an object',
       'Neck - scratch', 'Eyelash - pull hair', 'Eyebrow - pull hair',
       'Forehead - scratch', 'Above ear - pull hair', 'Wave hello',
       'Write name in air', 'Neck - pinch skin', 'Text on phone',
       'Pull air toward your face', 'Pinch knee/leg skin',
       'Scratch knee/leg skin', 'Drink from bottle/cup', 'Glasses on/off']

    reversed_target_map = {i: g for i, g in enumerate(gestures_list)}
    
    binary_map = {
        'Above ear - pull hair': 1, 
        'Cheek - pinch skin': 1,
        'Eyelash - pull hair': 1, 
        'Eyebrow - pull hair':1,
        'Forehead - pull hairline': 1, 
        'Forehead - scratch': 1, 
        'Neck - scratch': 1, 
        'Neck - pinch skin': 1, 
        
        'Drink from bottle/cup': 0, 
        'Feel around in tray and pull out an object': 0,
        'Glasses on/off': 0,
        'Pull air toward your face': 0, 
        'Pinch knee/leg skin': 0,
        'Scratch knee/leg skin': 0, 
        'Text on phone': 0,
        'Wave hello': 0,
        'Write name in air': 0, 
        'Write name on leg': 0, 
    }

    # Evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_tof, x_sens, y in data_loader:
            x_tof, x_sens = x_tof.to(device), x_sens.to(device)
            out = model(x_tof, x_sens)
            preds = out.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(y.numpy())
    
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    # binary score
    all_text_preds = [reversed_target_map.get(e) for e in all_preds]
    all_binary_preds = [binary_map.get(e) for e in all_text_preds]
    
    all_text_target = [reversed_target_map.get(e) for e in all_targets]
    all_binary_target = [binary_map.get(e) for e in all_text_target]
    
    f1_binary = f1_score(all_binary_target, all_binary_preds)

    print(f"Test Acc: {acc:.4f} - F1 ('macro'): {f1:.4f} - F1 binary: {f1_binary}")


def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_tof, x_sens, y in train_loader:
            x_tof, x_sens, y = x_tof.to(device), x_sens.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x_tof, x_sens)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x_tof, x_sens, y in val_loader:
                x_tof, x_sens = x_tof.to(device), x_sens.to(device)
                out = model(x_tof, x_sens)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.numpy())

        acc = np.mean(np.array(all_preds) == np.array(all_targets))
        f1 = f1_score(all_targets, all_preds, average='macro')

        print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f} - Val Acc: {acc:.4f} - F1: {f1:.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GestureModel(img_feat_dim=40*8, 
                     sens_feat_dim=12, 
                     cnn1d_out=128, 
                     num_classes=18)

train_model(model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            device=device, 
            epochs=500,
            lr=0.001)


get_eval_metric(model, test_loader)

