%matplotlib inline

import os
import random
import time
import math

import cv2
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Audio

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset

from tqdm.auto import tqdm
import timm


class cfg:
    
    output_dir = '/kaggle/working/'
    train_datadir = '/kaggle/input/birdclef-2025/train_audio'
    train_csv = '/kaggle/input/birdclef-2025/train.csv'
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'

    SEED = 42
    debug_on = True
    is_files_loaded = True
    
    SR = 32000
    TARGET_SHAPE = (256, 256)
    TARGET_DURATION = 5.0
    N_FFT = 1024
    HOP_LENGTH = 500
    N_MELS = 128
    FMIN = 40
    FMAX = 15000
    POWER = 2
    is_normalized = True
    
                        
    model_name = 'efficientnet_b0'
    is_pre_trained = True
    input_channels = 1
    
    optimizer = 'AdamW'
    lr = 5e-4 
    weight_decay = 1e-5
    epochs = 10  
    batch_size = 32  
    criterion = 'BCEWithLogitsLoss'


    device = 'cpu'
    gpu_on = True
    if gpu_on:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    if debug_on:
        epochs = 2

    def load_spectrograms(self):
        if self.is_files_loaded:
            loaded_specs = '/kaggle/input/pre-loaded-spectrograms/chunks_picked_by_amplitude.npy'
            return np.load(loaded_specs, allow_pickle=True).item()
        
cfg = cfg()


# Retirada do Notebook do Kadircan İdrisoğlu
def set_seed(seed=42):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.SEED)


taxonomy = pd.read_csv(cfg.taxonomy_csv)
df = pd.read_csv(cfg.train_csv)
loaded_spectrograms = cfg.load_spectrograms()
df['class'] = df['primary_label'].map(taxonomy.set_index('primary_label')['class_name'])


bird_taxonomy = taxonomy[taxonomy['class_name'] == 'Aves']
bird_df = df.loc[df['class'] == 'Aves']
bird_df


#Funções para acessar um arquivo aleatório e para visualização

#Atenção, código mais feio do mundo abaixo
def random_file(idx=None, class_name=None):
    if idx is not None and class_name is None:
        return df['filename'][idx]
    if class_name is not None and idx is None:
        aux = df.loc[df['class'] == class_name]
        auxIndex = df.loc[df['class'] == class_name].index.tolist()
        return aux.iloc[auxIndex[random.randint(0, len(aux)-1)]]['filename']
    elif class_name is not None and idx is not None:
        aux = df.loc[df['class'] == class_name]
        auxIndex = df.loc[df['class'] == class_name].index.tolist()
        return aux.iloc[auxIndex[idx]]['filename']
    return df['filename'][random.randint(0, len(df)-1)]

def visualize_melspec(spec, cfg):
    fig, ax = plt.subplots()

    img = librosa.display.specshow(spec, x_axis='time',
                             y_axis='mel', sr=cfg.SR,
                             fmax=cfg.FMAX, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')


def audio_to_melspec(data: np.ndarray, cfg: object) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(
            y=data,
            sr=cfg.SR,
            n_fft=cfg.N_FFT,
            hop_length=cfg.HOP_LENGTH,
            n_mels=cfg.N_MELS,
            fmin=cfg.FMIN,
            fmax=cfg.FMAX,
            power=cfg.POWER
        )
    
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    if cfg.is_normalized:
        mel_spec_norm = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
        return mel_spec_norm
        
    return mel_spec

def crop_soundscape(wave: np.ndarray, cfg: object) -> list[tuple[np.float32, np.int64]]:
    center_samples = []
    sample_range = int(cfg.TARGET_DURATION * cfg.SR / 2)
    
    wave = np.pad(wave, (sample_range, sample_range), mode='constant', constant_values=0)
    limite = np.percentile(wave, 99.95)  # top 0,05%
    
    for amp in wave:
        if np.float32(amp) >= limite:
            if len(center_samples) == 0:
                sample_idx = np.where(wave == amp)[0][0]
                center_samples.append((amp, sample_idx))
            else:
                sample_idx = np.where(wave == amp)[0][0]
                inf_limit = center_samples[-1][-1] - sample_range
                sup_limit = center_samples[-1][-1] + sample_range
                if sample_idx >= center_samples[-1][-1] + sample_range:
                    center_samples.append((amp, sample_idx))
                    
    return center_samples

def cropped_audios(wave: np.ndarray, cfg: object):
    audio_arr = []
    center_samples = crop_soundscape(wave, cfg)
    
    for _, idx in center_samples:
        start = max(0, idx - sample_range)
        end = idx + sample_range
        audio_arr.append(wave[start:end])
    return audio_arr

def process_audio_file(file: str, cfg: object) -> np.ndarray:
    
    audio, _ = librosa.load(os.path.join(cfg.train_datadir, file), sr = cfg.SR)
    target_samples = int(cfg.TARGET_DURATION * cfg.SR)

    mel_specs = []
    
    if len(audio) < target_samples:                            # áudio pequeno
        n_copy = math.ceil(target_samples / len(audio))
        if n_copy > 1:
            audio = np.concatenate([audio] * n_copy)           # duplica o áudio

    
    audio_arr = cropped_audios(audio, cfg)
    for audio_slice in audio_arr:
        if len(audio_slice) < target_samples:
            audio_slice = np.pad(audio_slice, 
                                 (0, target_samples - len(audio_slice)), 
                                 mode='constant')
        mel_specs.append(audio_to_melspec(audio_slice, cfg))             # cria o mel spectrograma
    
        if mel_specs[-1].shape != cfg.TARGET_SHAPE:                      # resize pro input da cnn
            mel_specs[-1] = cv2.resize(
                mel_specs[-1], cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
    
    return mel_specs

def generate_spectrograms(df: pd.DataFrame, cfg: object, name: str = 'spectrograms', save = False) -> dict:
    
    """
    df: dataframe de treino
    cfg: classe de configuração
    name: nome do arquivo ao salvar
    save: se True, salva o dicionário em '/kaggle/working/name.npy'.
        Lembre-se de baixar o arquivo antes de finalizar a sessão do Kaggle.
    """
    print("Generating mel spectrograms from audio files...")
    start_time = time.time()

    all_bird_data = {}
    errors = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        if cfg.debug_on and i >= 100:
            break
        
        try:
            samplename = row['filename'].split('/')[0]+'-'+row['filename'].split('/')[1].split('.')[0]
            filepath = row['filename']
            
            mel_specs = process_audio_file(filepath, cfg)
            
            if mel_specs is not None:
                all_bird_data[samplename] = mel_specs
            
        except Exception as e:
            print(f"Error processing {row.filepath}: {e}")
            errors.append((row.filepath, str(e)))

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Successfully processed {len(all_bird_data)} files out of {len(df)}")
    print(f"Failed to process {len(errors)} files")

    if save:
        np.save(f'{os.path.join(cfg.output_dir,name)}.npy', all_bird_data)
    
    return all_bird_data


spectrogramas = generate_spectrograms(bird_df, cfg)


audio_files


center_samples = crop_soundscape(wave, cfg)
for _, idx in center_samples:
    start = max(0, idx - sample_range)
    end = idx + sample_range
    display(Audio(wave[start:end], rate=cfg.SR))


gen_new_specs = False
name = 'chunks_picked_by_amplitude'
if gen_new_specs:
    generate_spectrograms(df, cfg, name, save=True)


species_ids = bird_taxonomy['primary_label'].tolist()
len({label: idx for idx, label in enumerate(species_ids)})


# Dataset
# Sem nenhum data augmentation
# Recebe train.csv e o dicionário dos spectrogramas e retorna o tensor dos spectrogramas e um tensor do one hot encoding das labels

class AudioDataset(Dataset):
    def __init__(self, df, taxonomy, spectrograms = None, cfg = cfg):
        self.df = df
        self.cfg = cfg
        self.spectrograms = spectrograms

        self.species_ids = taxonomy['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}
        
        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file = os.path.join(self.cfg.train_datadir, row['filename'])
        label = row['primary_label']
        samplename = row['samplename']

        if self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
            spec_tensor = torch.tensor(spec).float().unsqueeze(0)
            label = self.encode_label(label)
            return spec_tensor, label

        
        
        audio = process_audio_file(file, self.cfg)
        spec = audio_to_melspec(audio, self.cfg)
        spec_tensor = torch.tensor(spec).float().unsqueeze(0)
        label = self.encode_label(label)
        
        return spec_tensor, label

    # Retirada do Notebook do Kadircan İdrisoğlu
    def encode_label(self, label):
        """Encode label to one-hot vector"""
        target = np.zeros(self.num_classes)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target
    


class CNN(nn.Module):
    def __init__(self, taxonomy, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = len(taxonomy)
        
        # Backbone: Camadas iniciais de um modelo. Faz a extração das características
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.is_pre_trained,                 # Se false, reseta os pesos pré treinados.
            in_chans=cfg.input_channels,                   # Número de canais de entrada.
            drop_rate=0.2,
            drop_path_rate=0.2,
        )

        self.backbone = self.backbone.to(cfg.device)
        
        # Remove a camada final do backbone e passa a diante para conectar no meu nn.Linear
        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')

        # Head: Parte da rede neural responsável por fazer a predição.
        self.pooling = nn.AdaptiveAvgPool2d(1)         # ajusta a feature map pra 1D p/ classifier
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, self.num_classes).to(cfg.device)    # Classificação final

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, dict):
            features = features['features']
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size((0), -1))
        logits = self.classifier(features)
        return logits
        


#Training loop
def training_loop(df, taxonomy, model, spectrograms, cfg):
    model = model.to(cfg.device)
    train_loss_values = []
    valid_loss_values = []
    
    if cfg.criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
        
    if cfg.optimizer == 'AdamW':
        optimizer = optim.Adam(
                model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay
            )

    labels = df['primary_label'].tolist()
    
    train_dataframe, val_dataframe = train_test_split(df, test_size=0.2, random_state=42, stratify=labels)

    train_dataset = AudioDataset(train_dataframe,taxonomy = taxonomy, spectrograms = spectrograms, cfg=cfg)
    val_dataset = AudioDataset(val_dataframe,taxonomy = taxonomy, spectrograms = spectrograms, cfg=cfg)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        pin_memory=True,
        drop_last=True,
    )
    
    valid_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        pin_memory=True,
        drop_last=True,           #gera outputs e labels de tamanhos diferentes no último batch se drop_last = False
    )

    for epoch in tqdm(range(cfg.epochs)):   #tqdm serve para plotar uma barra de progessão para o treinamento
        model.train()
        running_loss = 0.0
        all_train_outputs, all_train_labels, all_val_outputs, all_val_labels = [], [], [], []
        
        for inputs, labels in train_loader:
            
            inputs = inputs.to(cfg.device)
            labels = labels.to(cfg.device)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            all_train_labels.append(labels.detach().cpu().numpy())
            all_train_outputs.append(output.detach().cpu().numpy())
            
        all_train_labels = np.concatenate(all_train_labels)
        all_train_outputs = np.concatenate(all_train_outputs)
        
        avg_train_loss = running_loss / len(train_loader)
        train_loss_values.append(avg_train_loss)
        train_auc = calculate_auc(all_train_labels, all_train_outputs)
                
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(valid_loader):
                inputs = inputs.to(cfg.device)
                labels = labels.to(cfg.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                all_val_labels.append(labels.detach().cpu().numpy())
                all_val_outputs.append(output.detach().cpu().numpy())

        
        all_val_labels = np.concatenate(all_val_labels)
        all_val_outputs = np.concatenate(all_val_outputs)
        avg_valid_loss = running_loss / len(valid_loader)
        valid_loss_values.append(avg_valid_loss)
        val_auc = calculate_auc(all_val_labels, all_val_outputs)
        
        print(f"Epoch {epoch+1}/{cfg.epochs} => Train Loss: {avg_train_loss:.4f}, Train AUC: {train_auc:.4f}, Validation Loss: {avg_valid_loss:.4f}, Validation AUC:{val_auc}")

    return train_loss_values, train_auc, valid_loss_values, val_auc


# Retirada do Notebook do Kadircan İdrisoğlu
def calculate_auc(targets, outputs):
    '''
    Recebe dois np.2darray com todos os targets e outputs de cada batch de uma época
    '''
    num_classes = targets.shape[1]  # Nº de colunas do target
    aucs = []
    
    probs = 1 / (1 + np.exp(-outputs)) # Função sigmoide
    
    for i in range(num_classes):
        
        if np.sum(targets[:, i]) > 0: 
            class_auc = roc_auc_score(targets[:, i], probs[:, i])
            aucs.append(class_auc)
    
    return np.mean(aucs) if aucs else 0.0


loaded_spectrograms = cfg.load_spectrograms()
train_loss, train_auc, val_loss, val_auc = training_loop(
    bird_df,
    bird_taxonomy,
    CNN(bird_taxonomy, cfg),
    spectrograms = loaded_spectrograms,
    cfg = cfg
)


plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


torch.save(CNN(bird_taxonomy, cfg).state_dict(), os.path.join(cfg.output_dir, '2stModel.pth'))


'''
todo:
    o algorítmo não vai filtrar muito bem se o espetrograma conter muitas bandas
    com altos níveis de decibéis, precisa ser repensado ou remendado com algum condicional
    (vide /kaggle/input/birdclef-2025/train_audio/bobfly1/XC304176.ogg)
        - ideia: tomar um comprimento do target_dB dinâmico para abrangir diversos casos
    
'''

def get_freq(spec, cfg):

    mel_freq = librosa.mel_frequencies(n_mels=cfg.N_MELS, fmin = cfg.FMIN, fmax = cfg.FMAX)
    band_higher_dB = [max(dB) for dB in spec]
    target_dB = sorted(band_higher_dB, reverse=True)[:20]
    
    freq_range=[]
    freq_range = [
        tuple(np.where(spec == db)) for db in target_dB
    ] 
    
    bands = [x[0].item() for x in freq_range]
    mel_freq = mel_freq[min(bands):max(bands)]
    samples = [x[1].item() for x in freq_range]
    return mel_freq, (bands, samples)
    


import scipy.signal as signal

def bandpass_filter(y,low,high,Fs,order=4):
        '''
        Fitro passa-banda Butterworth
        Recebe: frequências de corte superior e inferior e a frequência da amostragem.
        Retorna: O áudio filtrado.
        '''
        nyquist = 0.5*Fs
        b,a = signal.butter(order,[low/nyquist,high/nyquist],btype='bandpass',analog=False)
        y_filter = signal.filtfilt(b, a, y)
        return y_filter


#antiga função
def process_audio_file(file: str, cfg: object) -> np.ndarray:
    
    audio, _ = librosa.load(os.path.join(cfg.train_datadir, file), sr = cfg.SR)
    target_samples = int(cfg.TARGET_DURATION * cfg.SR)
    
    if len(audio) < target_samples:                            # áudio pequeno
        n_copy = math.ceil(target_samples / len(audio))
        if n_copy > 1:
            audio = np.concatenate([audio] * n_copy)           # duplica o áudio

    
    peak_db = np.where(audio == max(audio))[0][0]              # sample de maior amplitude
    start = peak_db - int(target_samples/5)
    end = peak_db + int(target_samples * 4/5) 
    
    if peak_db - int(target_samples) < 0:
        start = peak_db
        end = peak_db + target_samples
        
    audio_slice = audio[start:end]
    if len(audio_slice) < target_samples:
        audio_slice = np.pad(audio_slice, 
                             (0, target_samples - len(audio_slice)), 
                             mode='constant')
        
    mel_spec = audio_to_melspec(audio_slice, cfg)             # cria o mel spectrograma
    
    if mel_spec.shape != cfg.TARGET_SHAPE:                      # resize pro input da cnn
        mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
    
    return mel_spec


taxonomy[taxonomy['primary_label'] == 'greani1']


path = os.path.join(cfg.train_datadir, random_file(idx=1,class_name='Aves'))
wave, _ = librosa.load(path, sr = cfg.SR)

plt.figure(figsize=(10,5))
plt.plot(wave)
plt.title(f'{path.split("/")[-2]}')
plt.show()
display(Audio(wave, rate=cfg.SR))


def crop_soundscape(wave, cfg):

    center_samples = []
    sample_range = int(cfg.TARGET_DURATION * cfg.SR / 2)
    
    wave = np.pad(wave, (sample_range, sample_range), mode='constant', constant_values=0)
    limite = np.percentile(wave, 99.95)  # top 5%
    for amp in wave:
        if np.float32(amp) >= limite:
            if len(center_samples) == 0:
                sample_idx = np.where(wave == amp)[0][0]
                center_samples.append((amp, sample_idx))
            else:
                sample_idx = np.where(wave == amp)[0][0]
                inf_limit = center_samples[-1][-1] - sample_range
                sup_limit = center_samples[-1][-1] + sample_range
                if sample_idx >= center_samples[-1][-1] + sample_range:
                    center_samples.append((amp, sample_idx))
    return center_samples


def cropped_audios(wave, cfg):
    audio_arr = []
    center_samples = crop_soundscape(wave, cfg)
    for _, idx in center_samples:
        start = max(0, idx - sample_range)
        end = idx + sample_range
        audio_arr.append(wave[start:end])
    return audio_arr


center_samples = crop_soundscape(wave, cfg)
for _, idx in center_samples:
    start = max(0, idx - sample_range)
    end = idx + sample_range
    display(Audio(wave[start:end], rate=cfg.SR))





type(center_samples[0][0])


left_index =  sample_range - sample
right_index = sample+sample_range
wave[right_index:left_index]


center_samples = []
sample_range = cfg.TARGET_DURATION * cfg.SR / 2

aux = 0
while aux<len(top_amps):
    if aux==0:
        center_samples = np.where(wave == top_amps[aux])[0][0]
    else:
        aux_amp = np.where(wave == top_amps[aux][0][0])
        for amp in top_amps:
            if aux_amp > center_samples[]
    aux+=1

np.where(wave == top_amps[i])[0][0]


np.where(wave == top_amps[1])[0][0] > np.where(wave == top_amps[0])[0][0] - cfg.TARGET_DURATION * cfg.SR / 2


np.where(wave == top_amps[1])[0][0] < np.where(wave == top_amps[0])[0][0] + cfg.TARGET_DURATION * cfg.SR / 2

