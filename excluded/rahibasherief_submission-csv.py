import pandas as pd
import numpy as np
import librosa

import torch
import torch.nn as nn
import os
import random
from matplotlib import pyplot as plt
import seaborn as sns
from ast import literal_eval
import timm
import kaggle_metric_utilities

import pandas.api.types

import sklearn.metrics
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import gc

from warnings import filterwarnings
filterwarnings("ignore")


class Config:
    train_dir = "/kaggle/input/birdclef-2025/train_audio"
    seed =42
    train_csv = "/kaggle/input/birdclef-2025/train.csv"
    sample_csv = "/kaggle/input/birdclef-2025/sample_submission.csv"
    test_soundscapes = "/kaggle/input/birdclef-2025/test_soundscapes"

    sr = int(32e3)
    num_classes = 206
    n_fft = 1024
    hop_length = 500

    n_mels = 128
    fmin = 50
    fmax = 16000
    power = 2


def set_seed(seed: int = Config.seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Set Seed: {seed}")

set_seed()


from IPython.display import HTML

HTML("<style>div.output_area pre {font-size: 10px;}</style>")


data_df = pd.read_csv(Config.train_csv) 

for col in ('secondary_labels', 'type'):
    data_df[col] = data_df[col].apply(lambda x: "###".join(literal_eval(x))) #stringifying the secondary label

data_df['filename'] = data_df['filename'].apply(lambda x: Config.train_dir + "/" + x) #adapting the file name to the train audio
data_df.sample(10)


#Null check
data_df.isnull().sum()


#Distributuion of the ratings

plt.figure(figsize = (15,3))
sns.histplot(data_df, x ='rating')
plt.xticks(np.arange(0,5.5,0.5))
plt.show();


#Distribution of the primary label
plt.figure(figsize = (15,3))
sns.histplot(data_df, x ='primary_label')
plt.xticks(rotation =90)
plt.show();


#Distribution of the primary label with different ratings
for r in range(0,6):
    plt.figure(figsize = (20,3))
    sns.countplot(data_df[data_df['rating']== float(r)], x ='primary_label')
    plt.title(f"Rating {r}")
    plt.xticks(rotation =90)
    plt.show();


#statistics of audio duration

durations = []
for idx, row in data_df.sample(100).iterrows():
    data, _ = librosa.load(row['filename'],sr= Config.sr)
    durations.append(librosa.get_duration(y=data, sr = Config.sr))

d_df = pd.DataFrame(columns = ["durations"],data = durations)

plt.figure(figsize = (10,5))
plt.title("Distribution of audio lengths")
sns.histplot(d_df, x = "durations")
plt.show();

d_df.describe()


# Plot species frequency
species_counts = data_df['primary_label'].value_counts()
plt.figure(figsize=(12, 6))
species_counts[:50].plot(kind='bar')  # Top 50 species
plt.title("Class Imbalance (Top 50 Species)")
plt.xlabel("Species ID")
plt.ylabel("Count")
plt.show()


def show_signal(file_path):
    class_, collector = file_path.split("/")[-2:]

    y, sr = librosa.load(file_path, sr = Config.sr)\

    fig, axes = plt.subplots(2,2, figsize = (20, 10))
    
    fig.suptitle(f"class: {class_} | collector: {collector}", fontsize =16)

    #plottinig raw signal
    librosa.display.waveshow(y, sr=sr, ax=axes[0,0])
    axes[0,0].set_title("Raw signal")
    

    #plotting fourier transformed signal
    ft = np.abs(librosa.stft(
        y,
        n_fft = Config.n_fft,
        hop_length = Config.hop_length
    ))
    im1 = librosa.display.specshow(
        ft,
        sr = sr,
        x_axis = 'time',
        y_axis = 'linear',
        ax = axes[0, 1]
    )
    fig.colorbar(im1, ax = axes[0, 1])
    axes[0, 1].set_title("Spectrogram")

    #plotting log scaled fourier transformes signal
    ft_db = librosa.amplitude_to_db(ft, ref = np.max)
    im2 = librosa.display.specshow(
        ft_db,
        sr = sr,
        x_axis = 'time',
        y_axis = 'log',
        ax = axes[1, 0]
    )
    fig.colorbar(im2, ax = axes[1, 0])
    axes[1, 0].set_title("Log scaled spectrogram")

    #plotting mel spectrogram
    mel_sp = librosa.feature.melspectrogram(
        y = y,
        sr = Config.sr,
        fmin = Config.fmin,
        fmax = Config.fmax,
        power = Config.power,
        n_mels = Config.n_mels,
        
    )
    mel_sp = librosa.power_to_db(mel_sp, ref = np.max)
    im3 = librosa.display.specshow(
        mel_sp,
        y_axis = 'mel',
        sr = Config.sr,
        fmin = Config.fmin,
        x_axis = 'time',
        fmax = Config.fmax,
        ax = axes[1, 1]
    )

    fig.colorbar(im3, ax = axes[1, 0])
    axes[1, 1].set_title("Mel spectrogram")

    plt.show()

show_signal(data_df['filename'].values[0])


for idx, row in data_df.sample(10).iterrows():show_signal(row['filename'])


label_mapper = {
    label: idx
    for idx, label in enumerate(sorted(data_df['primary_label'].unique()))
}

rev_mapper = {
    idx: label
    for label, idx in label_mapper.items()
}

class BirdClefDataset(torch.utils.data.Dataset):
    def __init__(self, df, mode ="train"):
        self.df = df
        self.mode = mode
        
    def __len__(self): return len(self.df)

    def process(self, audio_path):
        data, _ = librosa.load(audio_path, sr = Config.sr)

        data = data * 1024 #scaling

        chunk_duration = 10
        min_len = chunk_duration * Config.sr

        #if the audio signal is less than min_len
        if len(data) < min_len:
            cnt = int(np.ceil(min_len / len(data)))
            data = np.tile(data, cnt)
            
        #making the data length divisible by min_len
        leftover = len(data) % min_len
        if leftover > 0 :
            front_crop = leftover // 2
            back_crop = leftover - front_crop
            data = data[front_crop : len(data) - back_crop]


        #Truncating the signal to min_len
        data = data[:min_len] 
        data = data.reshape(-1, min_len)
        
        #creating mel spectrogram
        mel_sp = librosa.feature.melspectrogram(
            y =data,
            sr = Config.sr,
            fmin = Config.fmin,
            fmax = Config.fmax,
            power = Config.power,
            n_mels = Config.n_mels,
            n_fft = Config.n_fft,
            hop_length = Config.hop_length
            
        )

        mel_sp = librosa.power_to_db(mel_sp, ref = 1)
        
        #Normalizing the feature values
        eps = 1e-12
        mel_sp = (mel_sp - mel_sp.min())/ (mel_sp.max() - mel_sp.min() + eps)
        
        mel_sp = mel_sp[:, :,:640]
        return mel_sp

    
    
    def __getitem__(self, idx):
        row = self.df.loc[idx,:]
        filename = row['filename']

    #TODO: Spectrogram conversion
        x = self.process(filename)

        if self.mode =="train":
            y =label_mapper[row['primary_label']]
            return x, y

        return x




mel_sp = BirdClefDataset(data_df).process(data_df['filename'].values[0])
print(mel_sp.shape)

#Trasnform from C, H, W -> H, W, C
plt.imshow(mel_sp.reshape(128,640, -1))
plt.show();


class Model(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.base_model = timm.create_model(
            model_name = model_name,
            num_classes = Config.num_classes,
            pretrained = False,
            in_chans = 1
        )
    def forward(self, x):
        return self.base_model(x)


# Checking Dataset and Model class

tmp_ds = BirdClefDataset(data_df.sample(10).reset_index())
model = Model("tf_efficientnet_b0")

for i in range(10):
    x, y = tmp_ds[i]

    model.eval()

    preds = model(torch.tensor([x]))
    preds = torch.argmax(torch.softmax(preds, dim=1), dim = 1).item()
    plt.imshow(x.reshape(128, 640, -1))
    plt.title(f"Label:{rev_mapper[y]} | {x.shape} | model_pred: {rev_mapper[preds]}") 
    plt.show()

del model
del tmp_ds


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Version of macro-averaged ROC-AUC score that ignores all classes that have no true positive labels.
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if not pandas.api.types.is_numeric_dtype(submission.values):
        bad_dtypes = {x: submission[x].dtype  for x in submission.columns if not pandas.api.types.is_numeric_dtype(submission[x])}
        raise ParticipantVisibleError(f'Invalid submission data types found: {bad_dtypes}')

    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    assert len(scored_columns) > 0

    return kaggle_metric_utilities.safe_call_score(sklearn.metrics.roc_auc_score, solution[scored_columns].values, submission[scored_columns].values, average='macro')

def cal_score(labels, preds):
    labels = np.concatenate(labels)
    preds = np.concatenate(preds)

    labels_df = pd.DataFrame(labels > 0.5, columns = list(label_mapper.keys()))
    pred_df = pd.DataFrame(preds, columns = list(label_mapper.keys()))

    labels_df['id'] = np.arange(len(labels_df))
    pred_df['id'] = np.arange(len(pred_df))

    return score(labels_df, pred_df, row_id_column_name = 'id')



#Training configs

epochs = 5
num_folds = 3
device = "cuda" if torch.cuda.is_available() else "CPU"
lr = 1e-4
target_col = 'primary_label'
df = data_df

skf = StratifiedKFold(
    n_splits = num_folds,
    shuffle = True,
    random_state = Config.seed
    
)

df['kfold'] = -1

for fold, (train_idx, val_idx) in enumerate(skf.split(X= df, y= df[target_col])):
    df.loc[val_idx, 'kfold'] = fold


df


for fold in range(num_folds):
    train_df = df[df['kfold'] != fold].reset_index(drop = True)
    val_df = df[df['kfold'] == fold].reset_index(drop = True)

    train_ds = BirdClefDataset(train_df)
    val_ds = BirdClefDataset(val_df)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size = 8,
        shuffle = True,
        num_workers = 2,
        drop_last = True
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size = 16,
        shuffle = False,
        num_workers = 2,
        drop_last = False
    )

    # Initializing the model
    model = Model(model_name = "tf_efficientnet_b0").to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)


    best_auc = 0

    #per fold training
    for epoch in range (epochs):
        model.train()

        pred_train =[]
        label_train =[]

        running_loss = 0.0

        for(x, y) in tqdm (train_loader, desc = "Training"):
            x, y = x.to(device), y.to(device)

            #Converting to One hot encoding
            y_one_hot = nn.functional.one_hot(
                y,
                num_classes = Config.num_classes
            ).float()
            
            optimizer.zero_grad()
            outputs = model(x)

            loss = criterion(outputs, y)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            probs = torch.softmax(outputs, dim = 1)
            pred_train.append(probs.detach().cpu().numpy())
            label_train.append(y_one_hot.detach().cpu().numpy())
            
        # Validation
        model.eval()
    
        pred_val =[]
        label_val =[]
    
        running_val_loss = 0.0
    
        with torch.no_grad():
            for (x, y) in tqdm (val_loader, desc = "Validation"):
                x, y = x.to(device), y.to(device)
        
                 #Converting to One hot encoding
                y_one_hot = nn.functional.one_hot(
                    y,
                    num_classes = Config.num_classes
                ).float()
                    
                outputs = model(x)
        
                loss = criterion(outputs, y)
            
                running_val_loss += loss.item()
        
                probs = torch.softmax(outputs, dim = 1)
                pred_val.append(probs.detach().cpu().numpy())
                label_val.append(y_one_hot.detach().cpu().numpy())

        # Computing AUC and loss
        auc_train = cal_score (label_train, pred_train)
        auc_val = cal_score(label_val, pred_val)
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss =  running_loss / len(val_loader)
        
        print(f"[Fold]: {fold} | [EPOCH]: {epoch} | Loss: {avg_train_loss:.4f} | Val_loss: {avg_val_loss:.4f}")
        print(f"[Fold]: {fold} | [EPOCH]: {epoch} | Train AUC: {auc_train:.4f} | Val AUC: {auc_val:.4f}")
        
        
        if best_auc <= auc_val:
            best_auc = auc_val
            torch.save(
                model.state_dict(),
                f"fold_{fold}_epoch_{epoch}_effnetB0_val_auc_{auc_val}_val_loss_{avg_val_loss}.pth"
            )

    del train_df, val_df, train_ds, val_ds, train_loader, val_loader, criterion, optimizer
    gc.collect()
    torch.cuda.empty_cache() 


df




