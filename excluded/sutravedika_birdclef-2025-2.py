import numpy as np
import pandas as pd

import librosa
import glob

import torch
import torch.nn as nn
import albumentations

import os
import random
from joblib import Parallel, delayed
import json
from matplotlib import pyplot as plt
import seaborn as sns
import timm

import pandas.api.types
import sklearn.metrics

from tqdm import tqdm
import gc

from sklearn.model_selection import StratifiedKFold

from ast import literal_eval

import torch.nn.functional as F

from warnings import filterwarnings
filterwarnings('ignore')


class Config:
    train_dir = "/kaggle/input/birdclef-2025/train_audio"
    seed = 42
    train_csv = "/kaggle/input/birdclef-2025/train.csv"
    
    train_soundscapes = "/kaggle/input/birdclef-2025/train_soundscapes"
    test_soundscapes = "/kaggle/input/birdclef-2025/test_soundscapes"
    test_audio = "/kaggle/input/birdclef-2025/test_audio"
    sample_submission_csv = "/kaggle/input/birdclef-2025/sample_submission.csv"
    sr = int(32e3)
    n_fft = 1024
    hop_length = 500
    n_mels = 128
    fmin = 50
    fmax = 16000
    power = 2
    num_classes = 206
    image_shape = (128, 648, 1)
    submission_mode = len(glob.glob("/kaggle/input/birdclef-2025/test_soundscapes/*.ogg")) > 0
    


def set_seed(seed:int=Config.seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determistic = True
    torch.backends.cudnn.benchmark = False
    print(f'[info] Set Seeds:{seed}')

set_seed()


if Config.submission_mode: sound_dir = glob.glob(Config.test_soundscapes + "/*.ogg")
else: sound_dir = glob.glob(Config.train_soundscapes + "/*.ogg")[:5]

sound_dir





def process(audio_path):
    filename = audio_path.split("/")[-1].split(".")[0]
    data, _ = librosa.load(audio_path, sr = Config.sr)
    data = data *1024
    local_mapper={}

    #Dividing the  data into 5sec chunks
    chunk_duration =  5
    min_len = chunk_duration * Config.sr

    for i in range(0, len(data), min_len):
        #Making row ids
        t= i // Config.sr
        row_id = f"{filename}_{t + chunk_duration}"
        chunk_5s = data[i:i+min_len]
        chunk_10s= np.tile(chunk_5s, 2)
        chunk_10s = chunk_10s.reshape(-1, len(chunk_10s))

    # Converting to mel spectrogram
        mel_sp = librosa.feature.melspectrogram(
            y= chunk_10s,
            sr= Config.sr,
            fmin = Config.fmin,
            power= Config.power,
            n_mels = Config.n_mels,
            n_fft = Config.n_fft,
            hop_length = Config.hop_length
        )
        mel_sp = librosa.power_to_db(mel_sp, ref=1)
    
        
        eps = 1e-12
        mel_sp = (mel_sp - mel_sp.min()/(mel_sp.max() - mel_sp.min() + eps))
        mel_sp =mel_sp[:,:,:640]
        local_mapper[row_id] = mel_sp
    return local_mapper

#Loading the audio files
all_mappers = Parallel(
    n_jobs = -1,
    backend = 'loky')(delayed(process)(filepath)for filepath in sound_dir)


#Creating complete mapping
global_mapper = {}
for mapper in all_mappers: global_mapper.update(mapper)

print(f"[INFO]Loaded all sudio files, total_items:{len(global_mapper)}")


for key, val in global_mapper.items(): 
    print(key)
    print(val)
    break


%%time

models = [
    "/kaggle/input/effnet-b0-512-3200-spectro-1-mixup-2-avg-bg-bird/classifier_timm_tf_efficientnet_b0_512_32000_spectro_1_mixup_2_avg_bg_bird/lightning_logs/version_fold0/_ckpt_epoch_18.ckpt",
    "/kaggle/input/effnet-b0-512-3200-spectro-1-mixup-2-avg-bg-bird/classifier_timm_tf_efficientnet_b0_512_32000_spectro_1_mixup_2_avg_bg_bird/lightning_logs/version_fold0/_ckpt_epoch_32.ckpt",
    "/kaggle/input/effnet-b0-512-3200-spectro-1-mixup-2-avg-bg-bird/classifier_timm_tf_efficientnet_b0_512_32000_spectro_1_mixup_2_avg_bg_bird/lightning_logs/version_fold0/_ckpt_epoch_25.ckpt"
]

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model class
class Model(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.base_model = timm.create_model(
            model_name=model_name,
            num_classes=Config.num_classes,
            pretrained=False,
            in_chans=1,
        )
    
    def forward(self, x):
        return self.base_model(x)
models_pool = []

for model_path in models:
    model = Model(model_name="tf_efficientnet_b0")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.", "").replace("base_model.", "")
        new_state_dict[new_key] = v


    model.load_state_dict(new_state_dict, strict=False)

    model.eval().to(device)
    models_pool.append(model)

print("[INFO] Loaded all the models successfully!")


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, mapper):
        self.mapper = mapper
        self.ids = list(self.mapper.keys())
    def __len__(self): return len(self.mapper)
    def __getitem__(self, idx): return self.ids[idx], self.mapper[self.ids[idx]]


test_loader = torch.utils.data.DataLoader(
    test_ds := TestDataset(global_mapper),
    batch_size = 16,
    num_workers = 2,
    shuffle = False,
    drop_last = False
)

# to capture the model prediction per row id

pred_mapper = {}


for (row_ids, mels) in test_loader:
    mels_t = torch.tensor(mels).to(device)

    model_preds = []

    with torch.no_grad():
        for model in models_pool:
            output = model(mels_t)

            probs = torch.sigmoid(output).detach().cpu().numpy().squeeze()
            model_preds.append(probs) #prediction of every model on current batch

    # Avg the model predictions
    mel_preds = np.mean (model_preds, axis = 0)

    for idx, row_id in enumerate(row_ids):
        pred_mapper[row_id] = mel_preds[idx]

    del mels_t

len(global_mapper), len(pred_mapper.keys())


# #Creating submission df
# columns = pd.read_csv(Config.sample_submission_csv).columns
# sub_df = pd.DataFrame(columns = columns[1:], data = list(pred_mapper.values()))
# sub_df['row_id'] = list(pred_mapper.keys())
# sub_df = sub_df[[sub_df.columns[-1]]+ [*sub_df.columns[:-1].tolist()]]



sample_submission_path = '/kaggle/input/birdclef-2025/sample_submission.csv'

print(f"Looking for: {sample_submission_path}")
print(f"Files in /kaggle/input/birdclef-2025: {os.listdir('/kaggle/input/birdclef-2025')}")

columns = pd.read_csv(sample_submission_path).columns
species_columns = columns[1:]

sub_df = pd.DataFrame(data=list(pred_mapper.values()), columns=species_columns)
sub_df['row_id'] = list(pred_mapper.keys())

sub_df = sub_df[['row_id'] + species_columns.tolist()]

sub_df.to_csv('/kaggle/working/submission.csv', index=False)
print("[INFO] Submission file saved successfully at /kaggle/working/submission.csv")


sample_sub = pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')
print("Sample submission shape:", sample_sub.shape)

print("Your submission shape:", sub_df.shape)


sample_row_ids = pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')['row_id'].tolist()
your_row_ids = sub_df['row_id'].tolist()

missing = set(sample_row_ids) - set(your_row_ids)
extra = set(your_row_ids) - set(sample_row_ids)

print("Missing row_ids:", missing)
print("Extra row_ids:", extra)


predictions = sub_df.iloc[:, 1:].values
print("Min prediction value:", predictions.min())
print("Max prediction value:", predictions.max())
print("Any NaN values?", np.isnan(predictions).any())


print(sub_df.dtypes)


sub_df.to_csv('/kaggle/working/submission.csv', index=False)


print(sub_df.head())


print(pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv').head())


sub_df.to_csv('/kaggle/working/submission.csv', index=False)
print("[INFO] Submission file saved successfully at /kaggle/working/submission.csv")




