import pandas as pd
import numpy as np
import librosa
import glob 
import librosa.display
import torch
import torch.nn as nn
import os

import random
from matplotlib import pyplot as plt
import seaborn as sns
from ast import literal_eval
import timm
import pandas.api.types

import kaggle_metric_utilities

import sklearn.metrics
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import gc


from warnings import filterwarnings
filterwarnings("ignore")


class Config:
    train_dir = "/kaggle/input/birdclef-2025/train_audio"
    seed = 42
    train_csv = "/kaggle/input/birdclef-2025/train.csv"
    sample_csv = "/kaggle/input/birdclef-2025/sample_submission.csv"
    test_soundscapes = "/kaggle/input/birdclef-2025/test_soundscapes.csv"

    sr = int(32e3)
    num_classes = 206
    n_fft = 1024
    hop_length = 500

    n_mels = 128
    fmin = 50
    fmax = 16000
    power = 2
    


def set_seed(seed : int = Config.seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if(torch.cuda.is_available()):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Set seed: {seed}")

set_seed()


data_df = pd.read_csv(Config.train_csv)

for col in ('secondary_labels', 'type'):
    data_df[col] = data_df[col].apply(lambda x : "###".join(literal_eval(x)))
data_df['filename'] = data_df['filename'].apply(lambda x: Config.train_dir + '/' + x)

data_df.sample(10)


def rating_to_weight(rating):
    if rating in {5.0, 4.5}:
        return 2.5
    elif rating in {4.0, 0}:
        return 2.0
    elif rating in {3.0, 3.5}:
        return 1.5
    else:
        return 1.0

data_df["weight"] = data_df["rating"].apply(rating_to_weight)


# Null check
# data_df.isnull().sum()


#Distribution of the raitings
# plt.figure(figsize = (15,3))
# sns.histplot(data_df, x='rating')
# plt.xticks(np.arange(0, 5.5, 0.5))
# plt.show()


# Distribution of primary label
# plt.figure(figsize = (15,3))
# sns.countplot(data_df, x='primary_label')
# plt.xticks(rotation=90)
# plt.show()


# Distribution of primary label with different ratings
# for r in range(0,6):
#     plt.figure(figsize = (20,3))
#     sns.countplot(data_df[data_df['rating'] == float(r)], x = 'primary_label')
#     plt.title(f"Rating {r}")
#     plt.xticks(rotation=90)
#     plt.show()



# Statistic of audio durations
# durations = []
# for idx, row in data_df.sample(100).iterrows() :
#     data, _ = librosa.load(row['filename'], sr = Config.sr)
#     durations.append(librosa.get_duration(y = data, sr = Config.sr))

# d_df = pd.DataFrame(columns = ["durations"], data = durations)
# plt.figure(figsize = (10, 5))
# plt.title("Distribution of audio lengths")
# sns.histplot(d_df, x = "durations", bins = 100)
# plt.show();

# d_df.describe()


# def show_signal(file_path):
#     class_, collector = file_path.split("/")[-2:]
#     y, sr = librosa.load(file_path, sr=Config.sr)
#     fig, axes = plt.subplots(2, 2, figsize=(20, 10))
#     fig.suptitle(f"Class: {class_} | Collector: {collector}", fontsize=16)

#     # Plotting raw signal
#     librosa.display.waveshow(y, sr=sr, ax=axes[0, 0])
#     axes[0, 0].set_title("Raw signal")
    


#     # Plotting fourier transformed signal
#     ft = np.abs(librosa.stft(
#         y,
#         n_fft = Config.n_fft,
#         hop_length = Config.hop_length
#     ))
#     im1 = librosa.display.specshow(
#         ft,
#         sr = sr,
#         x_axis = 'time',
#         y_axis = 'linear',
#         ax = axes[0, 1]
#     )
#     fig.colorbar(im1, ax = axes[0, 1])
#     axes[0, 1].set_title("Spectrogram")

#     # Plotting log scale fourier transformed signal
#     ft_db = librosa.amplitude_to_db(ft, ref = np.max)
#     im2 = librosa.display.specshow(
#     ft_db,
#     sr=sr,
#     x_axis='time',
#     y_axis='log',
#     ax=axes[1, 0]
#     )

#     fig.colorbar(im2, ax=axes[1, 0])
#     axes[1, 0].set_title("Log Scaled spectrogram")


#     # Pplotting mel spectrograms
#     mel_sp = librosa.feature.melspectrogram(
#         y = y,
#         sr = Config.sr,
#         fmin = Config.fmin,
#         fmax = Config.fmax,
#         power = Config.power,
#         n_mels = Config.n_mels,
#     )
#     mel_sp = librosa.power_to_db(mel_sp, ref=np.max)

#     im3 = librosa.display.specshow(
#         mel_sp,
#         y_axis='mel',
#         sr=Config.sr,
#         fmin=Config.fmin,
#         x_axis='time',
#         fmax=Config.fmax,
#         ax=axes[1, 1]
#     )
#     fig.colorbar(im3, ax=axes[1, 1])
#     axes[1, 1].set_title("Mel Spectrogram")


#     plt.show()


# show_signal(data_df['filename'].values[0])

# data_df['filename'].values[0]
# data_df['filename'].values[0].split("/")[-2:]



# for idx, row in data_df.sample(10).iterrows(): show_signal(row['filename'])





label_mapper = {
    label: idx 
    for idx, label in enumerate(sorted(data_df['primary_label'].unique()))
}

rev_mapper = {
    idx: label 
    for label, idx in label_mapper.items()
}

class BirdClefDataset(torch.utils.data.Dataset):
    def __init__(self, df, mode="train"):
        import pickle

        self.df = df
        self.mode = mode

        # Load voice_data từ file
        with open("/kaggle/input/bc25-separation-voice-from-data-by-silero-vad/train_voice_data.pkl", "rb") as f:
            self.voice_data = pickle.load(f)

    def __len__(self):
        return len(self.df)

    def crop_voice(self, y, audio_path, sr):
        if audio_path in self.voice_data:
            mask = np.ones(len(y), dtype=bool)
            for seg in self.voice_data[audio_path]:
                start = int(seg['start'] * sr)
                end = int(seg['end'] * sr)
                mask[start:end] = False
            y = y[mask]
        return y


    def process(self, audio_path):
        data, _ = librosa.load(audio_path, sr=Config.sr)
        data = self.crop_voice(data, audio_path, Config.sr)
        data = data * 1024
        chunk_duration = 10
        min_len = chunk_duration * Config.sr

        # If the audio signal is less than min_len
        if len(data) < min_len:
            cnt = int(np.ceil(min_len / len(data)))
            data = np.tile(data, cnt)
        
        # Making the data length divisible by min_len
        leftover = len(data) % min_len
        if leftover > 0:
            front_crop = leftover // 2
            back_crop = leftover - front_crop
            data = data[front_crop : len(data) - back_crop]
        
        # Truncating the signal to min_len
        data = data[:min_len]
        data = data.reshape(-1, min_len)
        # Creating mel spectrogram
        mel_sp = librosa.feature.melspectrogram(
            y = data,
            sr = Config.sr,
            fmin = Config.fmin,
            fmax = Config.fmax,
            power = Config.power,
            n_mels = Config.n_mels,
            n_fft = Config.n_fft,
            hop_length = Config.hop_length
        )

        mel_sp = librosa.power_to_db(mel_sp, ref = 1)

        # Normalizing the features
        eps = 1e-12
        mel_sp = (mel_sp - mel_sp.min()) / (mel_sp.max() - mel_sp.min() + eps)

        mel_sp = mel_sp[:, :, :640]
        return mel_sp

    def __getitem__(self, idx):
        row = self.df.loc[idx, :]
        filename = row['filename']

        # TODO: Chuyển đổi sang spectrogram
        x = self.process(filename)

        if self.mode == "train":
            y = label_mapper[row['primary_label']]
            w = row['weight']
            return x, y, w

        return x



# mel_sp = BirdClefDataset(data_df).process(data_df['filename'].values[0])
# print(mel_sp.shape)

# Transform from C, H, W -> H, W, C
# plt.imshow(mel_sp.reshape(128, 640, -1))
# plt.show();



 # Định nghĩa Model
class Model(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.base_model = timm.create_model(
            model_name=model_name,
            num_classes=Config.num_classes,
            pretrained=False,
            in_chans=1
        )

    def forward(self, x):
        return self.base_model(x)



# Checking Dataset and Model Class
# tmp_ds = BirdClefDataset(data_df.sample(10).reset_index())
# model = Model("tf_efficientnet_b0")

# for i in range(10):
#     x, y = tmp_ds[i]

#     model.eval()

#     preds = model(torch.tensor([x]))
#     preds = torch.argmax(torch.softmax(preds, dim=1), dim=1).item()

#     plt.imshow(x.reshape(128, 640, -1))
#     plt.title(f"Label: {rev_mapper[y]} | {x.shape} | {rev_mapper[preds]}")
#     plt.show()

# del model
# del tmp_ds



class ParticipantVisibleError(Exception):
    pass

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Version of macro-averaged ROC-AUC score that ignores all classes that have no true positive labels.
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if not pandas.api.types.is_numeric_dtype(submission.values):
        bad_dtypes = {x: submission[x].dtype for x in submission.columns if not pandas.api.types.is_numeric_dtype(submission[x])}
        raise ParticipantVisibleError(f'Invalid submission data types found: {bad_dtypes}')

    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    assert len(scored_columns) > 0

    return kaggle_metric_utilities.safe_call_score(
        sklearn.metrics.roc_auc_score,
        solution[scored_columns].values,
        submission[scored_columns].values,
        average='macro'
    )

def cal_score(labels, preds):
    labels = np.concatenate(labels)
    preds = np.concatenate(preds)

     # Lọc những dòng có NaN trong label hoặc pred
    valid_mask = ~np.isnan(labels).any(axis=1) & ~np.isnan(preds).any(axis=1)
    labels = labels[valid_mask]
    preds = preds[valid_mask]
    
    labels_df = pd.DataFrame(labels > 0.5, columns = list(label_mapper.keys()))
    pred_df = pd.DataFrame(preds, columns = list(label_mapper.keys()))

    labels_df['id'] = np.arange(len(labels_df))
    pred_df['id'] = np.arange(len(pred_df))

    return score(labels_df, pred_df, row_id_column_name = 'id')



# Training Configs
epochs = 5
num_folds = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-4
target_col = 'primary_label'
df = data_df

skf = StratifiedKFold(
    n_splits = num_folds,
    shuffle = True,
    random_state = Config.seed
)

df['kfold'] = -1

for fold, (train_idx, val_idx) in enumerate(skf.split(X = df, y = df[target_col])):
    df.loc[val_idx, 'kfold'] = fold



model_choices = {
    0: "tf_efficientnet_b0",
    1: "tf_efficientnet_b1",
    2: "regnety_008",
    3: "mobilenetv2_100",
    4: "efficientvit_b0"
}

for fold in range(num_folds):
    print(f"\n[INFO] Fold {fold} | Model: {model_choices[fold]}")

    train_df = df[df['kfold'] != fold].reset_index(drop=True)
    val_df = df[df['kfold'] == fold].reset_index(drop=True)

    train_ds = BirdClefDataset(train_df)
    val_ds = BirdClefDataset(val_df)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )

    # Khởi tạo model theo tên được chọn cho fold hiện tại
    model_name = model_choices[fold]
    model = Model(model_name=model_name).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc = 0

    for epoch in range(epochs):
        model.train()
        pred_train, label_train = [], []
        running_loss = 0.0

        for (x, y, w) in tqdm(train_loader, desc="Training"):
            x, y, w = x.to(device), y.to(device), w.to(device)
            y_one_hot = nn.functional.one_hot(y, num_classes=Config.num_classes).float()
        
            optimizer.zero_grad()
            outputs = model(x)
        
           
            losses = criterion(outputs, y) 
            weighted_loss = (losses * w).mean()  
        
            weighted_loss.backward()
            optimizer.step()
        
            running_loss += weighted_loss.item()
            probs = torch.softmax(outputs, dim=1)
            pred_train.append(probs.detach().cpu().numpy())
            label_train.append(y_one_hot.detach().cpu().numpy())


        # Validation
        model.eval()
        pred_val, label_val = [], []
        running_val_loss = 0.0

        with torch.no_grad():
            for (x, y, *_) in tqdm(val_loader, desc="Validation"):
                x, y = x.to(device), y.to(device)
                y_one_hot = nn.functional.one_hot(y, num_classes=Config.num_classes).float()
                outputs = model(x)
                loss = criterion(outputs, y)
                running_val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                pred_val.append(probs.detach().cpu().numpy())
                label_val.append(y_one_hot.detach().cpu().numpy())

        # AUC and Loss
        auc_train = cal_score(label_train, pred_train)
        auc_val = cal_score(label_val, pred_val)

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = running_val_loss / len(val_loader)

        print(f"[Fold]: {fold} | [EPOCH]: {epoch} | Loss: {avg_train_loss:.4f} | Val_Loss: {avg_val_loss:.4f}")
        print(f"[Fold]: {fold} | [EPOCH]: {epoch} | Train AUC: {auc_train:.4f} | Val AUC: {auc_val:.4f}")

        if best_auc <= auc_val:
            best_auc = auc_val
            filename = f"fold_{fold}_{model_name}_epoch_{epoch}_val_auc_{float(auc_val):.4f}_val_loss_{float(avg_val_loss):.4f}.pth"
            torch.save(model.state_dict(), filename)
            print(f"[INFO] Model saved to: {filename}")

    del train_df, val_df, train_ds, val_ds, train_loader, val_loader, model, criterion, optimizer
    gc.collect()
    torch.cuda.empty_cache()



len(label_train), label_train[0].shape, len(pred_train), pred_train[0].shape



np.concatenate(pred_train).shape


