import os
import pandas as pd, numpy as np
from glob import glob
import matplotlib.pyplot as plt
VER = 2


# check the reading of one parquet for understanding

BASE_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/'

df = pd.DataFrame({'path': glob(BASE_PATH + '**/*.parquet')})
df['test_type'] = df['path'].str.split('/').str.get(-2).str.split('_').str.get(-1)
df['id'] = df['path'].str.split('/').str.get(-1).str.split('.').str.get(0)

df_eeg = pd.read_parquet(BASE_PATH + 'train_eegs/1000913311.parquet')
df_eeg.head()


# Determine the number of channels
# Assuming each row is a time point and each column is a channel
n_channels = df_eeg.shape[1]
n_channels


df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
TARGETS = df.columns[-6:]
print('Train shape:', df.shape )
print('Targets', list(TARGETS))
df.head()


# Creating a Unique EEG Segment per eeg_id:
# The code groups (groupby) the EEG data (df) by eeg_id. Each eeg_id represents a different EEG recording.
# It then picks the first spectrogram_id and the earliest (min) spectrogram_label_offset_seconds for each eeg_id. This helps in identifying the starting point of each EEG segment.
# The resulting DataFrame train has columns spec_id (first spectrogram_id) and min (earliest spectrogram_label_offset_seconds).
train = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_id':'first','spectrogram_label_offset_seconds':'min'})
train.columns = ['spec_id','min']


# Finding the Latest Point in Each EEG Segment:
# The code again groups the data by eeg_id and finds the latest (max) spectrogram_label_offset_seconds for each segment.
# This max value is added to the train DataFrame, representing the end point of each EEG segment.
tmp = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_label_offset_seconds':'max'})
train['max'] = tmp


tmp = df.groupby('eeg_id')[['patient_id']].agg('first') # The code adds the patient_id for each eeg_id to the train DataFrame. This links each EEG segment to a specific patient.
train['patient_id'] = tmp


tmp = df.groupby('eeg_id')[TARGETS].agg('sum') # The code sums up the target variable counts (like votes for seizure, LPD, etc.) for each eeg_id.
for t in TARGETS:
    train[t] = tmp[t].values
    
y_data = train[TARGETS].values # It then normalizes these counts so that they sum up to 1. This step converts the counts into probabilities, which is a common practice in classification tasks.
y_data = y_data / y_data.sum(axis=1,keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first') # For each eeg_id, the code includes the expert_consensus on the EEG segment's classification.
train['target'] = tmp

train = train.reset_index() # This makes eeg_id a regular column, making the DataFrame easier to work with.
print('Train non-overlapp eeg_id shape:', train.shape )
train.head()


READ_SPEC_FILES = False # If READ_SPEC_FILES is False, the code reads the combined file instead of individual files.
FEATURE_ENGINEER = True


%%time
# READ ALL SPECTROGRAMS
PATH = '/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/'
files = os.listdir(PATH)
print(f'There are {len(files)} spectrogram parquets')

if READ_SPEC_FILES:    
    spectrograms = {}
    for i,f in enumerate(files):
        if i%100==0: print(i,', ',end='')
        tmp = pd.read_parquet(f'{PATH}{f}')
        name = int(f.split('.')[0])
        spectrograms[name] = tmp.iloc[:,1:].values
else:
    spectrograms = np.load('/kaggle/input/brain-spectrograms/specs.npy',allow_pickle=True).item()


%time
# ENGINEER FEATURES
import warnings
warnings.filterwarnings('ignore')

# The code generates features from the spectrogram data for use in a model 
# The features are derived by calculating the mean and minimum values over time for each of the 400 spectrogram frequencies.
# Two types of windows are used for these calculations:
# A 10-minute window (_mean_10m, _min_10m).
# A 20-second window (_mean_20s, _min_20s).
# This process results in 1600 features (400 features × 4 calculations) for each EEG ID.

SPEC_COLS = pd.read_parquet(f'{PATH}1000086677.parquet').columns[1:]
FEATURES = [f'{c}_mean_10m' for c in SPEC_COLS]
FEATURES += [f'{c}_min_10m' for c in SPEC_COLS]
FEATURES += [f'{c}_mean_20s' for c in SPEC_COLS]
FEATURES += [f'{c}_min_20s' for c in SPEC_COLS]
print(f'We are creating {len(FEATURES)} features for {len(train)} rows... ',end='')


# A data matrix data is initialized to store the new features for each eeg_id in the train DataFrame.
# For each row in train, the code calculates the mean and minimum values within the specified 10-minute and 20-second windows.
# These calculated values are then stored in the data matrix.
# Finally, the matrix is added to the train DataFrame as new columns.

if FEATURE_ENGINEER:
    data = np.zeros((len(train),len(FEATURES)))
    for k in range(len(train)):
        if k%100==0: print(k,', ',end='')
        row = train.iloc[k]
        r = int( (row['min'] + row['max'])//4 ) 
        
        # 10 MINUTE WINDOW FEATURES (MEANS and MINS)
        x = np.nanmean(spectrograms[row.spec_id][r:r+300,:],axis=0)
        data[k,:400] = x
        x = np.nanmin(spectrograms[row.spec_id][r:r+300,:],axis=0)
        data[k,400:800] = x
        
        # 20 SECOND WINDOW FEATURES (MEANS and MINS)
        x = np.nanmean(spectrograms[row.spec_id][r+145:r+155,:],axis=0)
        data[k,800:1200] = x
        x = np.nanmin(spectrograms[row.spec_id][r+145:r+155,:],axis=0)
        data[k,1200:1600] = x

    train[FEATURES] = data
else:
    train = pd.read_parquet('/kaggle/input/brain-spectrograms/train.pqt')
print()
print('New train shape:',train.shape)


# CÉLULA 8.1 — Wavelets (PyWavelets) e utilitários
import pywt
from scipy.stats import entropy

FS = 200  # frequência de amostragem típica do HMS EEG
EEG_BASE = '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/'

# Mapeamento esperado para db4 nível 5 @ FS=200 Hz:
# D1: 50–100 (≈ruído alta freq), D2: 25–50 (Gama alta), D3: 12.5–25 (Beta),
# D4: 6.25–12.5 (Alpha), D5: 3.125–6.25 (Theta), A5: 0–3.125 (Delta)

def dwt_band_energies(x, wavelet='db4', level=5):
    """Retorna energias por banda (Delta..Gamma) + entropia para um canal 1D."""
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, mode='symmetric')
    # coeffs = [A5, D5, D4, D3, D2, D1]
    A5, D5, D4, D3, D2, D1 = coeffs

    band_coeffs = {
        'Delta': A5,        # ~0–3.1 Hz
        'Theta': D5,        # ~3.1–6.25 Hz
        'Alpha': D4,        # ~6.25–12.5 Hz
        'Beta':  D3,        # ~12.5–25 Hz
        'Gamma': D2,        # ~25–50 Hz (observação: D1 ~50–100 Hz fica de fora por robustez)
    }
    # energia por banda
    energies = {b: float(np.sum(np.square(c))) for b, c in band_coeffs.items()}
    total_energy = sum(energies.values()) + 1e-12
    # energias relativas (robustas a escala)
    rel = {f'{b}_relE': energies[b] / total_energy for b in energies}

    # estatísticas dos coeficientes por banda
    stats = {}
    for b, c in band_coeffs.items():
        c = np.asarray(c)
        stats.update({
            f'{b}_mean': float(np.mean(c)),
            f'{b}_std':  float(np.std(c)),
            f'{b}_mad':  float(np.mean(np.abs(c - np.mean(c)))),
        })

    # entropia espectral (a partir das energias relativas)
    p = np.array(list(rel.values()), dtype=float)
    stats['wavelet_entropy'] = float(entropy(p, base=np.e))

    # pacote final por canal
    feats = {}
    feats.update(rel)
    feats.update(stats)
    return feats

def extract_wavelet_features_from_eeg(eeg_df, seconds=60, fs=FS):
    """
    Recebe o DataFrame do EEG (colunas=canais), corta uma janela de 'seconds' centrada,
    e extrai features DWT por canal. Retorna dict {feature_name: value}.
    """
    x = eeg_df.values  # shape: [N_amostras, N_canais]
    n = len(x)
    win = int(seconds * fs)
    if n < win:
        # padding simples se sinal for curto
        pad = win - n
        x = np.pad(x, ((pad//2, pad - pad//2), (0, 0)), mode='edge')
        n = len(x)
    start = (n - win) // 2
    seg = x[start:start+win, :]  # [win, n_channels]

    feats = {}
    n_channels = seg.shape[1]
    for ch in range(n_channels):
        ch_sig = seg[:, ch]
        ch_feats = dwt_band_energies(ch_sig)
        for k, v in ch_feats.items():
            feats[f'ch{ch}_{k}'] = v

    # versões agregadas (média por banda entre canais) para reduzir dimensionalidade
    bands = ['Delta','Theta','Alpha','Beta','Gamma']
    agg = {}
    for key in ['relE','mean','std','mad']:
        for b in bands:
            cols = [f'ch{ch}_{b}_{key}' for ch in range(n_channels)]
            vals = [feats[c] for c in cols if c in feats]
            if vals:
                agg[f'agg_{b}_{key}_mean'] = float(np.mean(vals))
                agg[f'agg_{b}_{key}_std']  = float(np.std(vals))

    # entropia média/STD entre canais
    ent_cols = [f'ch{ch}_wavelet_entropy' for ch in range(n_channels)]
    ent_vals = [feats[c] for c in ent_cols if c in feats]
    if ent_vals:
        agg['agg_wavelet_entropy_mean'] = float(np.mean(ent_vals))
        agg['agg_wavelet_entropy_std']  = float(np.std(ent_vals))

    # retorna tanto por-canal (mais rico) quanto agregados (mais compacto)
    feats.update(agg)
    return feats



from scipy import signal
from sklearn.decomposition import PCA


def extract_frequency_band_features(segment):
    # Define EEG frequency bands
    eeg_bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30), 'Gamma': (30, 45)}
    
    band_features = []
    for band in eeg_bands:
        low, high = eeg_bands[band]
        # Filter signal for the specific band
        band_pass_filter = signal.butter(3, [low, high], btype='bandpass', fs=200, output='sos')
        filtered = signal.sosfilt(band_pass_filter, segment)
        # Extract features like mean, standard deviation, etc.
        band_features.extend([np.nanmean(filtered), np.nanstd(filtered), np.nanmax(filtered), np.nanmin(filtered)])
    
    return band_features


# import time
# from sklearn.impute import SimpleImputer

# # Initialize a PCA model
# pca = PCA(n_components=0.95)
# print("PCA model initialized.")

# # Initialize an array for original features
# num_rows = len(train)
# num_features = 20 * n_channels  # 20 features per channel
# data_original = np.zeros((num_rows, num_features))

# print("Starting feature extraction and PCA processing...")
# start_time = time.time()

# for k in range(num_rows):
#     if k % 1000 == 0:
#         print(f"Processing row {k} of {num_rows}...")

#     row = train.iloc[k]
#     r = int((row['min'] + row['max']) // 4)
#     eeg_segment = spectrograms[row.spec_id][r:r+300, :]

#     # Apply the feature extraction function to each EEG channel
#     all_channel_features = []
#     for i in range(n_channels):
#         channel_features = extract_frequency_band_features(eeg_segment[:, i])
#         all_channel_features.extend(channel_features)
    
#     data_original[k, :] = all_channel_features

# print("Data matrix constructed")

# # Impute NaN values in the data matrix
# imputer = SimpleImputer(strategy='mean')
# data_imputed = imputer.fit_transform(data_original)

# print(f"NaN values handled. Imputed data matrix shape: {data_imputed.shape}")

# # Apply PCA on the imputed data
# pca.fit(data_imputed)
# print("PCA fitting completed.")

# # Transform data using PCA
# data_pca = pca.transform(data_imputed)

# # Add PCA features to DataFrame
# pca_feature_columns = [f'pca_feature_{i}' for i in range(data_pca.shape[1])]
# train[pca_feature_columns] = data_pca

# # Measure total processing time
# total_time = time.time() - start_time
# print(f"Total processing time: {total_time:.2f} seconds.")


# CÉLULA 11.1 — Extração de features Wavelet (DWT) para o TRAIN
# Estratégia: para cada eeg_id em 'train', abre o .parquet bruto e extrai features DWT (60s janela).
# Armazenamos como colunas em 'train' e montamos FEATURES_WAVELET.

from tqdm import tqdm

FEATURES_WAVELET = None
wavelet_rows = []

print('Extraindo wavelet features (DWT) para TRAIN...')
for k in tqdm(range(len(train))):
    eeg_id = int(train.iloc[k]['eeg_id'])
    eeg_path = os.path.join(EEG_BASE, f'{eeg_id}.parquet')
    if not os.path.exists(eeg_path):
        # fallback: sem arquivo (raro), cria zeros
        wavelet_rows.append({})
        continue
    eeg_df = pd.read_parquet(eeg_path)
    # remove colunas não-numéricas, se houver
    eeg_df = eeg_df.select_dtypes(include=[np.number])
    feats = extract_wavelet_features_from_eeg(eeg_df, seconds=60, fs=FS)
    wavelet_rows.append(feats)

# alinhar para DataFrame
wavelet_df = pd.DataFrame(wavelet_rows).fillna(0.0)
if FEATURES_WAVELET is None:
    FEATURES_WAVELET = list(wavelet_df.columns)

print(f'Wavelet features geradas: {len(FEATURES_WAVELET)}')
# anexar ao train
train = pd.concat([train.reset_index(drop=True), wavelet_df.reset_index(drop=True)], axis=1)
print('Train+Wavelet shape:', train.shape)



train.head()


# from sklearn.preprocessing import StandardScaler

# # Columns to be excluded from scaling
# excluded_columns = ['eeg_id', 'spec_id', 'min', 'max', 'patient_id', 'seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote','target']

# # Save the columns to be excluded
# excluded_data = train[excluded_columns]

# # DataFrame with only the columns to be scaled
# features = train.drop(columns=excluded_columns)

# # Initialize the StandardScaler
# scaler = StandardScaler()

# # Fit the scaler to the features and transform them
# features_scaled = scaler.fit_transform(features)

# # Create a DataFrame from the scaled features
# features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

# # Concatenate the scaled features with the excluded columns
# train_scaled_df = pd.concat([excluded_data.reset_index(drop=True),features_scaled_df,], axis=1)
# train_scaled_df 



import xgboost as xgb
import gc
from sklearn.model_selection import KFold, GroupKFold

print('XGBoost version', xgb.__version__)


all_oof = []
all_true = []
TARS = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}

gkf = GroupKFold(n_splits=5)
for i, (train_index, valid_index) in enumerate(gkf.split(train , train .target, train .patient_id)):   
    
    print('#'*25)
    print(f'### Fold {i+1}')
    print(f'### train size {len(train_index)}, valid size {len(valid_index)}')
    print('#'*25)
    
    model = xgb.XGBClassifier(
        objective='multi:softprob', 
        num_class=len(TARS),
        learning_rate = 0.1, 
                      
#         tree_method='gpu_hist',  #skip GPU acceleration
    )
    
    # Prepare training and validation data
    X_train = train.loc[train_index, FEATURES]
    y_train = train.loc[train_index, 'target'].map(TARS)
    X_valid = train.loc[valid_index, FEATURES]
    y_valid = train.loc[valid_index, 'target'].map(TARS)
    
    model.fit(X_train, y_train, 
              eval_set=[(X_valid, y_valid)], 
              verbose=True, 
              early_stopping_rounds=10)
    model.save_model(f'XGB_v{VER}_f{i}.model')
    
    oof = model.predict_proba(X_valid)
    all_oof.append(oof)
    all_true.append(train.loc[valid_index, TARGETS].values)
    
    del X_train, y_train, X_valid, y_valid, oof
    gc.collect()
    
all_oof = np.concatenate(all_oof)
all_true = np.concatenate(all_true)


# CÉLULA 14.3 — Teste local (holdout por patient_id, sem mudanças no treino)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb

def soft_log_loss(y_true, y_pred, eps=1e-12):
    """
    Log loss (cross-entropy) para alvos probabilísticos (soft targets).
    y_true: (N, C) com linhas somando ≈1
    y_pred: (N, C) probabilidades preditas
    """
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    y_true = y_true / (y_true.sum(axis=1, keepdims=True) + eps)
    return float(-(y_true * np.log(y_pred)).sum(axis=1).mean())

# 1) Seleciona 20% dos pacientes para teste local (holdout estrito por patient_id)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
idx_train_local, idx_test_local = next(gss.split(train, groups=train['patient_id']))

train_local = train.iloc[idx_train_local].reset_index(drop=True)
test_local  = train.iloc[idx_test_local].reset_index(drop=True)

print(f'Holdout local — pacientes únicos: '
      f'train={train_local["patient_id"].nunique()} | test={test_local["patient_id"].nunique()}')
print(f'Linhas: train={len(train_local)} | test={len(test_local)}')

# 2) Define colunas de features (usa wavelets se existirem)
feature_cols = FEATURES_ALL if 'FEATURES_ALL' in globals() else FEATURES
print(f'Usando {len(feature_cols)} features.')

X_test_local = test_local[feature_cols]
y_true_soft  = test_local[TARGETS].values  # alvos probabilísticos
y_true_hard  = np.argmax(y_true_soft, axis=1)

# 3) Carrega os 5 modelos salvos (um por fold) e faz ensemble (média)
preds = []
for i in range(5):
    model = xgb.XGBClassifier()
    model.load_model(f'XGB_v{VER}_f{i}.model')
    preds.append(model.predict_proba(X_test_local))
y_pred = np.mean(preds, axis=0)

# 4) Métricas no teste local
ll_soft = soft_log_loss(y_true_soft, y_pred)
acc     = accuracy_score(y_true_hard, np.argmax(y_pred, axis=1))

print(f'\n=== Resultados no TESTE LOCAL (holdout 20% por paciente) ===')
print(f'LogLoss (soft targets): {ll_soft:.6f}')
print(f'Accuracy (top-1):       {acc:.4f}')



# CÉLULA 14.3B — Holdout local sem vazamento (modelo treinado só em train_local)

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
import xgboost as xgb

def soft_log_loss(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    y_true = y_true / (y_true.sum(axis=1, keepdims=True) + eps)
    return float(-(y_true * np.log(y_pred)).sum(axis=1).mean())

# Usa os mesmos objetos criados na 14.3:
# train_local, test_local, feature_cols, TARGETS

# 1) split interno de validação DENTRO de train_local (por paciente) para early stopping
gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=7)
idx_tr, idx_va = next(gss_inner.split(train_local, groups=train_local['patient_id']))

tr = train_local.iloc[idx_tr].reset_index(drop=True)
va = train_local.iloc[idx_va].reset_index(drop=True)

X_tr, y_tr = tr[feature_cols], tr['target'].map({'Seizure':0,'LPD':1,'GPD':2,'LRDA':3,'GRDA':4,'Other':5})
X_va, y_va = va[feature_cols], va['target'].map({'Seizure':0,'LPD':1,'GPD':2,'LRDA':3,'GRDA':4,'Other':5})

# 2) treina um ÚNICO modelo só com train_local
model_holdout = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=6,
    learning_rate=0.1,
    # tree_method='gpu_hist',  # habilite se tiver GPU
    eval_metric='mlogloss',
)
model_holdout.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    verbose=True,
    early_stopping_rounds=20
)

# 3) avalia no test_local (sem vazamento)
X_te = test_local[feature_cols]
y_true_soft = test_local[TARGETS].values
y_true_hard = np.argmax(y_true_soft, axis=1)

y_pred = model_holdout.predict_proba(X_te)
ll_soft = soft_log_loss(y_true_soft, y_pred)
acc     = accuracy_score(y_true_hard, np.argmax(y_pred, axis=1))

print('\n=== TESTE LOCAL (sem vazamento) ===')
print(f'LogLoss (soft targets): {ll_soft:.6f}')
print(f'Accuracy (top-1):       {acc:.4f}')



# CÉLULA 14.1 (corrigida) — Métricas OOF com alvos probabilísticos

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

def soft_log_loss(y_true, y_pred, eps=1e-12):
    """
    Log loss (cross-entropy) para alvos probabilísticos (soft targets).
    y_true: (N, C) com linhas somando ≈1
    y_pred: (N, C) com probabilidades (não precisa estar perfeitamente normalizado)
    """
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    y_true = y_true / (y_true.sum(axis=1, keepdims=True) + eps)
    return float(-(y_true * np.log(y_pred)).sum(axis=1).mean())

# 1) LogLoss principal com soft targets
oof_logloss = soft_log_loss(all_true, all_oof)
print(f'OOF LogLoss (multiclasse, soft targets): {oof_logloss:.6f}')

# 2) Diagnóstico: Accuracy top-1 (usando rótulo duro por argmax)
true_hard = np.argmax(all_true, axis=1)
pred_hard = np.argmax(all_oof, axis=1)
oof_acc = accuracy_score(true_hard, pred_hard)
print(f'OOF Accuracy (top-1, diagnóstico): {oof_acc:.4f}')

# 3) (Opcional) LogLoss "padrão" do sklearn com rótulo duro
#    Útil para comparar, mas não substitui o soft log loss oficial do desafio.
oof_logloss_hard = log_loss(true_hard, all_oof)
print(f'OOF LogLoss (sklearn, usando rótulo duro): {oof_logloss_hard:.6f}')



# CÉLULA 14.2 — Accuracy no treino (validação cruzada)
from sklearn.metrics import accuracy_score
import numpy as np

# all_oof = predições das folds de validação
# all_true = alvos probabilísticos originais (votos normalizados)

# converte para rótulo “duro” (classe mais votada)
true_labels = np.argmax(all_true, axis=1)
pred_labels = np.argmax(all_oof, axis=1)

acc_train = accuracy_score(true_labels, pred_labels)
print(f'Accuracy média (OOF - treino cruzado): {acc_train:.4f}')



# import optuna
# from sklearn.metrics import log_loss


# def objective(trial):
#     # Hyperparameters to be tuned by Optuna
#     param = {
#         'objective': 'multi:softprob',
#         'num_class': len(TARS),
#         'tree_method': 'gpu_hist',  # use 'gpu_hist' for GPU
#         'lambda': trial.suggest_loguniform('lambda', 1e-4, 10.0),
#         'alpha': trial.suggest_loguniform('alpha', 1e-4, 10.0),
#         'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
#         'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 0.9, 1.0]),
#         'learning_rate': trial.suggest_categorical('learning_rate', [0.008, 0.01, 0.02, 0.05, 0.1]),
#         'n_estimators': 1000,
#         'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13]),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
#     }

#     gkf = GroupKFold(n_splits=5)
#     cv_scores = []

#     for train_index, valid_index in gkf.split(train, train.target, train.patient_id):
#         X_train, X_valid = train.loc[train_index, FEATURES], train.loc[valid_index, FEATURES]
#         y_train, y_valid = train.loc[train_index, 'target'].map(TARS), train.loc[valid_index, 'target'].map(TARS)

#         model = xgb.XGBClassifier(**param)
#         model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False, early_stopping_rounds=10)
#         preds = model.predict_proba(X_valid)
#         cv_scores.append(log_loss(y_valid, preds))

#     return np.mean(cv_scores)

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=10)  # Increase n_trials for more extensive search

# print('Number of finished trials:', len(study.trials))
# print('Best trial:', study.best_trial.params)


TOP = 30

# Assuming 'model' is your trained model
feature_importance = model.feature_importances_

# Get the feature names from 'train'
feature_names = train.columns

# Sort the feature importances and get the indices of the sorted array
sorted_idx = np.argsort(feature_importance)

# Plot only the top 'TOP' features
fig = plt.figure(figsize=(10, 8))
plt.barh(np.arange(len(sorted_idx))[-TOP:], feature_importance[sorted_idx][-TOP:], align='center')
plt.yticks(np.arange(len(sorted_idx))[-TOP:], feature_names[sorted_idx][-TOP:])
plt.title(f'Feature Importance - Top {TOP}')
plt.show()


test = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/test.csv')
print('Test shape',test.shape)
test.head()


# PATH2 = '/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/'
# spec = pd.read_parquet(f'{PATH2}{s}.parquet')
# spec


# %%time
# # READ ALL TEST SPECTROGRAMS
# PATH2 = '/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/'
# files = os.listdir(PATH2)
# print(f'There are {len(files)} spectrogram parquets')

# spectrograms = {}
# for i,f in enumerate(files):
#     if i%100==0: print(i,', ',end='')
#     tmp = pd.read_parquet(f'{PATH2}{f}')
#     name = int(f.split('.')[0])
#     spectrograms_test[name] = tmp.iloc[:,1:].values



# %time
# # ENGINEER FEATURES
# import warnings
# warnings.filterwarnings('ignore')

# # The code generates features from the spectrogram data for use in a model 
# # The features are derived by calculating the mean and minimum values over time for each of the 400 spectrogram frequencies.
# # Two types of windows are used for these calculations:
# # A 10-minute window (_mean_10m, _min_10m).
# # A 20-second window (_mean_20s, _min_20s).
# # This process results in 1600 features (400 features × 4 calculations) for each EEG ID.

# SPEC_COLS = pd.read_parquet(f'{PATH}1000086677.parquet').columns[1:]
# FEATURES = [f'{c}_mean_10m' for c in SPEC_COLS]
# FEATURES += [f'{c}_min_10m' for c in SPEC_COLS]
# FEATURES += [f'{c}_mean_20s' for c in SPEC_COLS]
# FEATURES += [f'{c}_min_20s' for c in SPEC_COLS]
# print(f'We are creating {len(FEATURES)} features for {len(test)} rows... ',end='')


# # A data matrix data is initialized to store the new features for each eeg_id in the train DataFrame.
# # For each row in train, the code calculates the mean and minimum values within the specified 10-minute and 20-second windows.
# # These calculated values are then stored in the data matrix.
# # Finally, the matrix is added to the train DataFrame as new columns.

# data = np.zeros((len(test),len(FEATURES)))
# for k in range(len(test)):
#     if k%100==0: print(k,', ',end='')
#     row = test.iloc[k]
            
#     # 10 MINUTE WINDOW FEATURES
#     x = np.nanmean( spec.iloc[:,1:].values, axis=0)
#     data[k,:400] = x
#     x = np.nanmin( spec.iloc[:,1:].values, axis=0)
#     data[k,400:800] = x

#     # 20 SECOND WINDOW FEATURES
#     x = np.nanmean( spec.iloc[145:155,1:].values, axis=0)
#     data[k,800:1200] = x
#     x = np.nanmin( spec.iloc[145:155,1:].values, axis=0)
#     data[k,1200:1600] = x

#     test[FEATURES] = data

    
# print()
# print('New test shape:',test.shape)


# from sklearn.impute import SimpleImputer

# # Initialize a PCA model
# pca = PCA(n_components=0.95)
# print("PCA model initialized.")

# # Initialize an array for original features
# num_rows = len(test)
# num_features = 20 * n_channels  # 20 features per channel
# data_original = np.zeros((num_rows, num_features))

# print("Starting feature extraction and PCA processing...")
# start_time = time.time()

# for k in range(num_rows):
#     if k % 1000 == 0:
#         print(f"Processing row {k} of {num_rows}...")

#     row = train.iloc[k]
#     eeg_segment = spectrograms_test[853520][r:r+300, :]

#     # Apply the feature extraction function to each EEG channel
#     all_channel_features = []
#     for i in range(n_channels):
#         channel_features = extract_frequency_band_features(eeg_segment[:, i])
#         all_channel_features.extend(channel_features)
    
#     data_original[k, :] = all_channel_features

# print("Data matrix constructed")

# # Impute NaN values in the data matrix
# imputer = SimpleImputer(strategy='mean')
# data_imputed = imputer.fit_transform(data_original)

# print(f"NaN values handled. Imputed data matrix shape: {data_imputed.shape}")

# # Apply PCA on the imputed data
# pca.fit(data_imputed)
# print("PCA fitting completed.")

# # Transform data using PCA
# data_pca = pca.transform(data_imputed)

# # Add PCA features to DataFrame
# pca_feature_columns = [f'pca_feature_{i}' for i in range(data_pca.shape[1])]
# test[pca_feature_columns] = data_pca

# # Measure total processing time
# total_time = time.time() - start_time
# print(f"Total processing time: {total_time:.2f} seconds.")

# test.head()


# # Columns to be excluded from scaling
# excluded_columns = ['eeg_id', 'spectrogram_id', 'patient_id']

# # Save the columns to be excluded
# excluded_data = test[excluded_columns]

# # DataFrame with only the columns to be scaled
# features = test.drop(columns=excluded_columns)

# # Initialize the StandardScaler
# scaler = StandardScaler()

# # Fit the scaler to the features and transform them
# features_scaled = scaler.fit_transform(features)

# # Create a DataFrame from the scaled features
# features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

# # Concatenate the scaled features with the excluded columns
# test_scaled_df = pd.concat([excluded_data.reset_index(drop=True),features_scaled_df,], axis=1)
# test_scaled_df 



# FEATURE ENGINEER TEST
PATH2 = '/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/'
data = np.zeros((len(test),len(FEATURES)))
    
for k in range(len(test)):
    row = test.iloc[k]
    s = int( row.spectrogram_id )
    spec = pd.read_parquet(f'{PATH2}{s}.parquet')
    
    # 10 MINUTE WINDOW FEATURES
    x = np.nanmean( spec.iloc[:,1:].values, axis=0)
    data[k,:400] = x
    x = np.nanmin( spec.iloc[:,1:].values, axis=0)
    data[k,400:800] = x

    # 20 SECOND WINDOW FEATURES
    x = np.nanmean( spec.iloc[145:155,1:].values, axis=0)
    data[k,800:1200] = x
    x = np.nanmin( spec.iloc[145:155,1:].values, axis=0)
    data[k,1200:1600] = x

test[FEATURES] = data
print('New test shape',test.shape)


# INFER XGBOOST ON TEST
preds = []

for i in range(5):
    print(i, ', ', end='')
    
    # Load the XGBoost model
    model = xgb.XGBClassifier()
    model.load_model(f'XGB_v{VER}_f{i}.model')
    
    # Make predictions
    pred = model.predict_proba(test[FEATURES])
    preds.append(pred)

# Average the predictions from each fold
pred = np.mean(preds, axis=0)
print()
print('Test preds shape', pred.shape)


# CÉLULA 23.1A — Diagnóstico de confiança das predições no TEST (sem rótulos)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

probs_max = np.max(pred, axis=1)
entropy = -np.sum(np.clip(pred, 1e-12, 1).astype(float) * np.log(np.clip(pred, 1e-12, 1).astype(float)), axis=1)

print(f'Confiança média (max prob): {probs_max.mean():.4f} ± {probs_max.std():.4f}')
print(f'Entropia média: {entropy.mean():.4f} ± {entropy.std():.4f}')

# distribuição da classe predita
classes = ['Seizure','LPD','GPD','LRDA','GRDA','Other']
pred_labels = np.argmax(pred, axis=1)
class_counts = pd.Series(pred_labels).value_counts().reindex(range(len(classes)), fill_value=0)
disp = pd.DataFrame({'class': classes, 'count': class_counts.values, 'pct': class_counts.values / len(pred)})
print('\nDistribuição de classes previstas no TEST:')
print(disp)

# histograma de confiança
plt.figure(figsize=(6,4))
plt.hist(probs_max, bins=20, edgecolor='k')
plt.title('Histograma da confiança (max prob) — TEST')
plt.xlabel('max prob'); plt.ylabel('freq')
plt.show()



sub = pd.DataFrame({'eeg_id':test.eeg_id.values})
sub[TARGETS] = pred
sub.to_csv('submission.csv',index=False)
print('Submission shape',sub.shape)
sub.head()


# SANITY CHECK TO CONFIRM PREDICTIONS SUM TO ONE
sub.iloc[:,-6:].sum(axis=1)

