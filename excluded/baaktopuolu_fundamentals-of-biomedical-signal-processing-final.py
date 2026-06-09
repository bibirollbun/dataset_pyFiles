import os
import zipfile
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm



# 1. Opt parametreleri (EÄŸitimde kullanÄ±lan deÄŸerler)
# --------------------------
class Opt:
    in_channels = 32
    in_len = 1000
    batch_size = 64
    n_cpu = 4
    lr = 0.001
    n_epochs = 50   # kaÃ§ epoch ile eÄŸitilecek
opt = Opt()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ======================
# 1. VERI HAZIRLIK
# ======================

base_path = '/kaggle/input/grasp-and-lift-eeg-detection/'
train_zip_path = base_path + 'train.zip'
test_zip_path = base_path + 'test.zip'

with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall('./train')
with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall('./test')

train_data_path = './train/train/'
test_data_path = './test/test/'

train_data_files = glob.glob(train_data_path + '*_data.csv')[:10]  # Dosya sayÄ±sÄ±nÄ± sÄ±nÄ±rladÄ±k
train_events_files = glob.glob(train_data_path + '*_events.csv')[:10]


def get_subject_series(filename):
    base = filename.split('/')[-1]
    parts = base.split('_')
    subj = int(parts[0].replace('subj',''))
    series = int(parts[1].replace('series',''))
    return subj, series

train_data_dict, train_events_dict = {}, {}
for f in train_data_files:
    subj, series = get_subject_series(f)
    train_data_dict[(subj,series)] = pd.read_csv(f, index_col=0)
for f in train_events_files:
    subj, series = get_subject_series(f)
    train_events_dict[(subj,series)] = pd.read_csv(f, index_col=0)


# Basit pencereleme ile Ã¶rnek oluÅŸturma

def extract_features_and_labels(data_df, label_df, window_size=1000, step_size=100):
    X, y = [], []
    for start in range(0, len(data_df)-window_size, step_size):
        end = start + window_size
        window = data_df.iloc[start:end].values.T  # (features, time)
        labels = label_df.iloc[end-1].values.astype(np.float32)
        X.append(window)
        y.append(labels)
    return X, y


X_list, y_list = [], []
for key in train_data_dict:
    if key in train_events_dict:
        x, y = extract_features_and_labels(train_data_dict[key], train_events_dict[key])
        X_list.extend(x)
        y_list.extend(y)

X = np.stack(X_list).astype(np.float32)  # (N, channels, time)
y = np.stack(y_list).astype(np.float32)  # (N, 6)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


# --------------------------
# 2. Dataset TanÄ±mÄ±
# --------------------------
class EEGDataset(Dataset):
    def __init__(self, data, labels=None, window_size=opt.in_len, train=True):
        # data: (channels, total_time)
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.train = train

        self.total_len = data.shape[1]
        self.samples = self.total_len - window_size + 1

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        x = self.data[:, idx:idx+self.window_size]  # (channels, window_size)
        if self.train:
            y = self.labels[idx+self.window_size-1]  # son zaman noktasÄ±ndaki label
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        else:
            return torch.tensor(x, dtype=torch.float32)


# --------------------------
# 3. Model TanÄ±mÄ± (CNN + BiLSTM)
# --------------------------
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(opt.in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(128*2, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, channels, time)
        x = self.cnn(x)  # (batch, 128, time)
        x = x.permute(0, 2, 1)  # (batch, time, features)
        out, _ = self.lstm(x)   # (batch, time, hidden*2)
        out = out[:, -1, :]     # son zaman adÄ±mÄ±
        out = self.classifier(out)
        return out


# --------------------------
# 4. EÄŸitim ve DoÄŸrulama Verilerini HazÄ±rla
# --------------------------
# EÄŸitim verisini yÃ¼kle, normalize et ve bÃ¶l
# (Ã¶rnek olarak burada rastgele veri kullanÄ±yorum, kendi verinle deÄŸiÅŸtir)
np.random.seed(42)
total_time = 15000
train_data = np.random.randn(opt.in_channels, total_time).astype(np.float32)
train_labels = np.random.randint(0, 2, size=(total_time, 6)).astype(np.float32)


# Normalize et (kanal bazlÄ±)
mean = train_data.mean(axis=1, keepdims=True)
std = train_data.std(axis=1, keepdims=True)
train_data = (train_data - mean) / (std + 1e-7)


# Veri setini train / val olarak ayÄ±r
val_ratio = 0.2
val_start = int(total_time * (1 - val_ratio))

train_dataset = EEGDataset(train_data[:, :val_start], train_labels[:val_start], train=True)
val_dataset = EEGDataset(train_data[:, val_start:], train_labels[val_start:], train=True)

train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)



# --------------------------
# 5. Model, KayÄ±p ve Optimizasyon
# --------------------------
model = EEGNet().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)



import matplotlib.pyplot as plt
from tqdm import tqdm

patience = 15
best_val_loss = float('inf')
counter = 0

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(opt.n_epochs):
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.n_epochs} - Train"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Accuracy hesaplama (sÄ±nÄ±flandÄ±rma varsayÄ±mÄ±yla)
        _, predicted = torch.max(out, 1)
        _, labels = torch.max(y, 1)  # EÄŸer y one-hot ise, deÄŸilse direkt y olabilir
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_loss /= len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item()

            _, predicted = torch.max(out, 1)
            _, labels = torch.max(y, 1)  # Yine y'nin yapÄ±sÄ±na gÃ¶re dÃ¼zenle
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}/{opt.n_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Train Acc: {train_accuracy:.4f} - Val Acc: {val_accuracy:.4f}")

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pt")
        print("Model kaydedildi.")
    else:
        counter += 1
        print(f"EarlyStopping counter: {counter} / {patience}")
        if counter >= patience:
            print("Early stopping devreye girdi. EÄŸitim durduruluyor.")
            break


# EÄŸitim bittikten sonra grafik Ã§izimi
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs, train_losses, 'b-', label='Train Loss')
plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()


# --------------------------
# 7. Model Kaydet
# --------------------------
torch.save(model.state_dict(), "best_model.pt")



# --------------------------
# 4. Test Verisini HazÄ±rla ve Normalize Et
# --------------------------
TEST_DIR = '/kaggle/working/test/test'
FNAME = os.path.join(TEST_DIR, 'subj{}_series{}_data.csv')

trial_lengths = {}
test_data = []

for subj in range(1, 13):
    for series in [9, 10]:
        df = pd.read_csv(FNAME.format(subj, series))
        df = df.select_dtypes(include=[np.number])  # Sadece sayÄ±sal sÃ¼tunlarÄ± al
        trial_lengths[f'{subj}_{series}'] = len(df)
        test_data.append(df.values.T.astype(np.float32))  # (channels=32, T)

test_data = np.concatenate(test_data, axis=1)  # (32, total_T)

# Burada normalizasyon iÃ§in ideal olan eÄŸitim verisinden alÄ±nan mean/std kullanmaktÄ±r
# EÄŸer eÄŸitimde yoksa test verisinden hesapla (Ã¶rnek)
mean = test_data.mean(axis=1, keepdims=True)
std = test_data.std(axis=1, keepdims=True)
test_data = (test_data - mean) / (std + 1e-7)


# Dataset ve DataLoader oluÅŸtur
test_dataset = EEGDataset(test_data, labels=None, train=False)



class EEGBatchDataset(Dataset):
    def __init__(self, data, labels=None, train=True):
        self.data = data
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # (channels, window_size)
        if self.train and self.labels is not None:
            y = self.labels[idx]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        else:
            return torch.tensor(x, dtype=torch.float32)


# Biz pencereleme yaptÄ±ÄŸÄ±mÄ±z iÃ§in test iÃ§in de pencereleme yapalÄ±m:

def sliding_window(data, window_size=opt.in_len, step_size=100):
    windows = []
    for start in range(0, data.shape[1]-window_size+1, step_size):
        windows.append(data[:, start:start+window_size])
    return np.stack(windows)

test_windows = sliding_window(test_data, window_size=opt.in_len, step_size=100)  # (num_windows, channels, window_size)

# ğŸ”� HatalÄ± EEGDataset yerine doÄŸru sÄ±nÄ±fÄ± kullan
test_dataset = EEGBatchDataset(test_windows, train=False)
test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)


# Modeli yÃ¼kle
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()


# Tahmin yap
y_pred = []
with torch.no_grad():
    for x in tqdm(test_loader, desc="Test Prediction"):
        x = x.to(device)
        out = model(x)
        y_pred.append(out.cpu().numpy())

y_pred = np.concatenate(y_pred, axis=0)  # (num_samples, 6)


# Submission iÃ§in index hazÄ±rla
submission_index = []
for subj in range(1, 13):
    for series in [9, 10]:
        length = trial_lengths[f'{subj}_{series}']
        for t in range(length):
            submission_index.append(f'subj{subj}_series{series}_{t}')



# Pencere boyutundan dolayÄ± baÅŸta eksik tahmin olabilir, padding yap
pad_len = len(submission_index) - len(y_pred)
if pad_len > 0:
    y_pred = np.vstack([np.zeros((pad_len, 6)), y_pred])



# Etiket isimleri
labels = ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff', 'Replace', 'BothReleased']


# Submission DataFrame oluÅŸtur ve kaydet
submission = pd.DataFrame(y_pred, index=submission_index, columns=labels)
submission.to_csv("submission.csv", index_label="id", float_format='%.3f')
print("submission.csv dosyasÄ± kaydedildi.")

