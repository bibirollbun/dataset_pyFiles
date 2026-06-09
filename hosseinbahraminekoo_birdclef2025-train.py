import pandas as pd
import numpy as np
import librosa
import glob
import torch
import torchaudio.transforms as T
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
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import gc
from warnings import filterwarnings
filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"



class Config:
    train_dir = "/kaggle/input/birdclef-2025/train_audio"
    seed = 42
    train_csv = "/kaggle/input/birdclef-2025/train.csv"
    sample_csv = "/kaggle/input/birdclef-2025/sample_submission.csv"
    test_soundscapes = "/kaggle/input/birdclef-2025/test_soundscapes"
    sr = int(32e3)

    num_classes = 206
    n_fft = 2048
    hop_length = 500

    n_mels = 256
    fmin = 50
    fmax = 16000
    power = 2
    
    


def set_seed(seed: int = Config.seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # reproducible weight initialization
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    print(f"[info] set seed: {seed}")

set_seed()


data_df = pd.read_csv(Config.train_csv)
for col in ('secondary_labels', 'type'):
    data_df[col] = data_df[col].apply(lambda x: '###'.join(literal_eval(x)))

data_df['filename'] = data_df['filename'].apply(lambda x: Config.train_dir + '/' + x)
data_df.sample(10)


data_df.isnull().sum()


plt.figure(figsize = (15, 3))
sns.histplot(data_df, x = 'rating')
plt.xticks(np.arange(0, 5.5, 0.5))
plt.show()


plt.figure(figsize = (15, 3))
sns.countplot(data_df, x = 'primary_label')
plt.xticks(rotation = 90)
plt.show()


for r in range(0, 6):
    plt.figure(figsize = (20, 3))
    sns.countplot(data_df[data_df['rating'] == float(r)], x = 'primary_label')
    plt.title(f"Rating {r}")
    plt.xticks(rotation = 90)
    plt.show()


durations = []
for idx, row in data_df.sample(100).iterrows():
    data, _ = librosa.load(row['filename'], sr = Config.sr)
    durations.append(librosa.get_duration(y = data, sr = Config.sr))




d_df = pd.DataFrame(columns = ['durations'], data = durations)

plt.figure(figsize = (10, 5))
plt.title("Distribution of audio legnths")
sns.histplot(d_df, x = 'durations', bins = 100)
plt.show()

d_df.describe()


def show_signal(file_path):
    class_, collector = file_path.split("/")[-2:]
    
    y, sr = librosa.load(file_path, sr = Config.sr)
    
    fig, axes = plt.subplots(2, 2, figsize = (20, 10))
    fig.suptitle(f"Class: {class_} | Collector: {collector}", fontsize = 16)

    # Plotting raw signal
    librosa.display.waveshow(y, sr = sr, ax = axes[0, 0])
    axes[0, 0].set_title('Raw Signal')

    # Plotting Fourier Transformed Signal
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

    # Plotting log scaled fourier transformed signal
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

    # Plotting mel spectrogram
    mel_sp = librosa.feature.melspectrogram(
        y = y,
        sr = Config.sr,
        fmin = Config.fmin,
        fmax = Config.fmax,
        power = Config.power,
        n_mels = Config.n_mels  
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
    fig.colorbar(im3, ax = axes[1, 1])
    axes[1, 1].set_title("Mel spectrogram")

    plt.show()    


for idx, row in data_df.sample(10).iterrows():
    show_signal(row['filename'])


label_mapper = {
    label: idx
    for idx, label in enumerate(sorted(data_df['primary_label'].unique()))
}
# Map labels to indices for class_weight
data_df['label_idx'] = data_df['primary_label'].map(label_mapper)

rev_mapper = {
    idx: label
    for label, idx in label_mapper.items()
}

Config.num_classes = len(label_mapper)  # Update the config

class BirdClefDataset(torch.utils.data.Dataset):
    def __init__(self, df, mode = 'train', aug = True):
        self.df = df
        self.mode = mode
        # Initialize augmentations
        self.aug = aug
        self.spec_aug = torch.nn.Sequential(
            T.TimeMasking(time_mask_param=16),  
            T.FrequencyMasking(freq_mask_param=4)  
        ) if aug else None

        # Gaussian noise parameters
        self.noise_prob = 0.5  # 50% chance to apply noise
        self.noise_scale = 0.01  # Adjust based on your audio volume

    def __len__(self): return len(self.df)

    def _add_gaussian_noise(self, audio):
        """Add Gaussian noise to raw audio waveform"""
        if random.random() < self.noise_prob:
            noise = np.random.normal(0, self.noise_scale, len(audio))
            return audio + noise
        return audio

    def process(self, audio_path):
        data, _ = librosa.load(audio_path, sr = Config.sr)

        data = data * 1024

        # Apply Gaussian noise to RAW AUDIO (before spectrogram)
        if self.aug:
            data = self._add_gaussian_noise(data)
        
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

        # Creating Mel  spectrogram
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
        mel_sp = (mel_sp - mel_sp.min())/(mel_sp.max() - mel_sp.min() + eps)

        mel_sp = mel_sp[:, :, :640]
        return mel_sp
            
    
    def __getitem__(self, idx):
        row = self.df.loc[idx, :]
        filename = row['filename']

        # TODO: Spectrogram conversion
        x = self.process(filename) # Returns numpy array

        # Convert to tensor FIRST before augmentation
        x = torch.from_numpy(x).float()

        if self.mode == 'train':
            if self.aug:
                x = self.spec_aug(x)  # Apply augmentations    
            try:
                # Get label index
                label_idx = label_mapper[row['primary_label']]
                # Convert to multi-hot vector
                label_vec = torch.zeros(Config.num_classes, dtype=torch.float)
                label_vec[label_idx] = 1.0
                # y = label_mapper[row['primary_label']]
            except KeyError as e:
                print(f"Error: Label '{row['primary_label']}' not in label_mapper!")
                raise
            return x, label_vec

        return x


class Model(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.base_model = timm.create_model(
            model_name = model_name,
            pretrained = True,        # use pretrained weights
            in_chans = 1,             # for single-channel input
            num_classes = Config.num_classes,
            drop_rate = 0.3           # Dropout before classifier
        )


    def forward(self, x):
        return self.base_model(x)


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


# Create save directory
save_dir = "model_checkpoints"
os.makedirs(save_dir, exist_ok=True)


# Training configs
# epochs = 20
epochs = 15
num_folds = 3
save_interval = 3
lr = 1e-3

target_col = 'primary_label'
#df = data_df.sample(1000).reset_index()
df = data_df

# Setup GroupKFold
gkf = GroupKFold(
    n_splits = num_folds,
)

# 1. Assign fold numbers once
df['kfold'] = -1
for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups = df['filename'])):
    df.loc[val_idx, 'kfold'] = fold


df


def mixup_data(x, y, alpha=0.4):
    '''Apply Mixup to inputs and labels'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_model(resume_checkpoint=None):
    # Initialize model and optimizer ONCE (outside folds)
    model = Model(model_name='tf_efficientnet_b0').to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr = lr, 
        weight_decay=0
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)
    
    start_fold, start_epoch = 0, 0
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_fold = checkpoint['fold']
        scheduler.load_state_dict(checkpoint["scheduler"])
        # start_epoch = checkpoint['epoch'] + 1
        start_epoch = checkpoint['epoch'] 

    for fold in range(start_fold, num_folds):
        train_df = df[df['kfold'] != fold].reset_index(drop=True)
        val_df = df[df['kfold'] == fold].reset_index(drop=True)

        # Compute class weights for this fold
        y_train = train_df['label_idx'].values
        present_classes = np.unique(y_train)
        computed_weights = compute_class_weight(
            class_weight='balanced',
            classes = present_classes,
            y = y_train
        )
        # Ensure consistent shape (num_classes) with model output
        class_weights_tensor = torch.zeros(Config.num_classes, dtype=torch.float32).to(device)
        for cls, w in zip(present_classes, computed_weights):
            class_weights_tensor[cls] = w
            
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight = class_weights_tensor)
        
        train_ds = BirdClefDataset(train_df, mode='train', aug = True)
        val_ds = BirdClefDataset(val_df, mode='train', aug = False)
        
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=16, shuffle=True, num_workers=2, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=16, shuffle=False, num_workers=2, drop_last=False
        )

        best_auc = 0
        patience = 3  # Early stopping patience
        patience_counter = 0

        for epoch in range(start_epoch, epochs):
            model.train()
            pred_train, label_train = [], []
            running_loss = 0.0

            for x, y in tqdm(train_loader, desc="Training"):
                x, y = x.to(device), y.to(device)
            
                # Apply Mixup
                x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
            
                optimizer.zero_grad()
                outputs = model(x)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping

                optimizer.step()
                   
                running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                pred_train.append(probs.detach().cpu().numpy())
                #label_train.append(y_one_hot.detach().cpu().numpy())  # <-- Store one-hot for AUC
                label_train.append(y.detach().cpu().numpy())

            # Validation
            model.eval()
            pred_val, label_val = [], []
            running_val_loss = 0.0

            with torch.no_grad():
                for x, y in tqdm(val_loader, desc="Validation"):
                    x, y = x.to(device), y.to(device)
                    
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    running_val_loss += loss.item()
                    
                    probs = torch.softmax(outputs, dim = 1)
                    pred_val.append(probs.detach().cpu().numpy())
                    #label_val.append(y_one_hot.detach().cpu().numpy())
                    label_val.append(y.detach().cpu().numpy())

            # Compute metrics 
            auc_train = cal_score(label_train, pred_train)
            auc_val = cal_score(label_val, pred_val)
            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = running_val_loss / len(val_loader)

            print(f"Fold {fold} | Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"Fold {fold} | Epoch {epoch} | Train AUC: {auc_train:.4f} | Val AUC: {auc_val:.4f}")

            # Update LR scheduler based on validation AUC
            scheduler.step(auc_val)
            for i, param_group in enumerate(optimizer.param_groups):
                current_lr = param_group['lr']
                print(f"Epoch {epoch} | Fold {fold} | Learning Rate (group {i}): {current_lr}")

            # Save best model 
            if auc_val > best_auc:
                best_auc = auc_val
                torch.save(
                    model.state_dict(), 
                    f"fold_{fold}_epoch_{epoch}_best_effnetB0_val_auc_{auc_val:.4f}_val_loss_{avg_val_loss}.pth"
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch} for fold {fold}")
                    break

            # Save checkpoint every 3 epochs
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(
                    save_dir, 
                    f"fold{fold}_epoch{epoch+1}.pth.tar"
                )
                
                torch.save({
                    'epoch': epoch + 1,
                    'fold': fold,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, checkpoint_path)
                
                print(f"Saved checkpoint to {checkpoint_path}")
        
        start_epoch = 0  # Reset for next fold
        torch.cuda.empty_cache()

train_model()  # Start fresh
#checkpoint_path = os.path.join(
#    save_dir, 
#    "fold2_epoch3.pth.tar"
#)
#train_model("/kaggle/input/birdclef-fold1-epoch-9-mixup/fold1_epoch9.pth.tar")  # Resume from a checkpoint

