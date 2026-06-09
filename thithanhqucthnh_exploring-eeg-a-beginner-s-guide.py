import os
import pandas as pd, numpy as np
from glob import glob
import matplotlib.pyplot as plt
VER = 1


# Ä�á»‹nh nghÄ©a Ä‘Æ°á»�ng dáº«n chá»©a dá»¯ liá»‡u cá»§a cuá»™c thi
BASE_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/'

# Táº¡o má»™t DataFrame chá»©a danh sÃ¡ch táº¥t cáº£ cÃ¡c tá»‡p .parquet trong thÆ° má»¥c BASE_PATH (bao gá»“m cáº£ thÆ° má»¥c con)
df = pd.DataFrame({'path': glob(BASE_PATH + '**/*.parquet')})

# TrÃ­ch xuáº¥t loáº¡i test tá»« Ä‘Æ°á»�ng dáº«n cá»§a tá»«ng tá»‡p
# Cá»¥ thá»ƒ:
# - Chia Ä‘Æ°á»�ng dáº«n theo dáº¥u '/' vÃ  láº¥y pháº§n tá»­ thá»© hai tá»« cuá»‘i lÃªn (tÃªn thÆ° má»¥c chá»©a tá»‡p).
# - Chia tÃªn thÆ° má»¥c Ä‘Ã³ theo dáº¥u '_' vÃ  láº¥y pháº§n cuá»‘i cÃ¹ng.
# VÃ­ dá»¥: "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/1000913311.parquet"
# - TÃªn thÆ° má»¥c chá»©a file: "train_eegs"
# - Sau khi tÃ¡ch "_", láº¥y pháº§n cuá»‘i: "eegs"
df['test_type'] = df['path'].str.split('/').str.get(-2).str.split('_').str.get(-1)


# TrÃ­ch xuáº¥t ID cá»§a tá»‡p tá»« tÃªn file
# - Láº¥y pháº§n cuá»‘i cá»§a Ä‘Æ°á»�ng dáº«n (tÃªn file, vÃ­ dá»¥: "1000913311.parquet").
# - Chia theo dáº¥u '.' vÃ  láº¥y pháº§n Ä‘áº§u tiÃªn (bá»� Ä‘i pháº§n má»Ÿ rá»™ng .parquet).
# VÃ­ dá»¥: "1000913311.parquet" -> "1000913311"
df['id'] = df['path'].str.split('/').str.get(-1).str.split('.').str.get(0)


# Ä�á»�c dá»¯ liá»‡u EEG tá»« má»™t tá»‡p cá»¥ thá»ƒ (ID 1000913311) trong thÆ° má»¥c train_eegs
df_eeg = pd.read_parquet(BASE_PATH + 'train_eegs/1000913311.parquet')
df_eeg.head()


# XÃ¡c Ä‘á»‹nh sá»‘ lÆ°á»£ng kÃªnh (channels)
# Giáº£ Ä‘á»‹nh ráº±ng má»—i hÃ ng (row) trong df_eeg lÃ  má»™t Ä‘iá»ƒm theo thá»�i gian (time point),
# vÃ  má»—i cá»™t (column) lÃ  má»™t kÃªnh tÃ­n hiá»‡u EEG (channel).
n_channels = df_eeg.shape[1]
n_channels


df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')

# XÃ¡c Ä‘á»‹nh cÃ¡c cá»™t má»¥c tiÃªu (TARGETS) báº±ng cÃ¡ch láº¥y 6 cá»™t cuá»‘i cÃ¹ng cá»§a DataFrame
TARGETS = df.columns[-6:]

# KÃ­ch thÆ°á»›c cá»§a táº­p dá»¯ liá»‡u
print('Train shape:', df.shape )

# Ddanh sÃ¡ch cÃ¡c cá»™t má»¥c tiÃªu
print('Targets', list(TARGETS))

df.head()


# Táº¡o má»™t phÃ¢n Ä‘oáº¡n EEG duy nháº¥t cho má»—i eeg_id:
# - NhÃ³m dá»¯ liá»‡u (groupby) theo eeg_id, má»—i eeg_id Ä‘áº¡i diá»‡n cho má»™t báº£n ghi EEG riÃªng biá»‡t.
# - Chá»�n spectrogram_id Ä‘áº§u tiÃªn vÃ  thá»�i Ä‘iá»ƒm báº¯t Ä‘áº§u sá»›m nháº¥t (min) spectrogram_label_offset_seconds cho má»—i eeg_id.
# - Káº¿t quáº£ giÃºp xÃ¡c Ä‘á»‹nh Ä‘iá»ƒm báº¯t Ä‘áº§u cá»§a má»—i phÃ¢n Ä‘oáº¡n EEG.
train = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_id':'first','spectrogram_label_offset_seconds':'min'})

# Ä�á»•i tÃªn cÃ¡c cá»™t Ä‘á»ƒ dá»… Ä‘á»�c hÆ¡n:
# - 'spec_id': Chá»©a giÃ¡ trá»‹ spectrogram_id Ä‘áº§u tiÃªn cá»§a má»—i EEG.
# - 'min': Thá»�i Ä‘iá»ƒm báº¯t Ä‘áº§u sá»›m nháº¥t cá»§a nhÃ£n EEG.
train.columns = ['spec_id','min']


# TÃ¬m thá»�i Ä‘iá»ƒm káº¿t thÃºc cá»§a má»—i phÃ¢n Ä‘oáº¡n EEG:
# - NhÃ³m dá»¯ liá»‡u theo eeg_id, tÃ¬m giÃ¡ trá»‹ lá»›n nháº¥t (max) cá»§a spectrogram_label_offset_seconds.
# - GiÃ¡ trá»‹ nÃ y biá»ƒu thá»‹ Ä‘iá»ƒm cuá»‘i cá»§a phÃ¢n Ä‘oáº¡n EEG.
tmp = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_label_offset_seconds':'max'})


# ThÃªm giÃ¡ trá»‹ thá»�i Ä‘iá»ƒm káº¿t thÃºc vÃ o DataFrame train.
train['max'] = tmp

# ThÃªm thÃ´ng tin bá»‡nh nhÃ¢n vÃ o train DataFrame:
# - NhÃ³m dá»¯ liá»‡u theo eeg_id, láº¥y giÃ¡ trá»‹ Ä‘áº§u tiÃªn cá»§a patient_id.
# - Ä�iá»�u nÃ y giÃºp liÃªn káº¿t má»—i EEG vá»›i má»™t bá»‡nh nhÃ¢n cá»¥ thá»ƒ.
tmp = df.groupby('eeg_id')[['patient_id']].agg('first')

# GÃ¡n patient_id vÃ o train
train['patient_id'] = tmp

# TÃ­nh tá»•ng sá»‘ láº§n xuáº¥t hiá»‡n cá»§a má»—i má»¥c tiÃªu (target labels) theo eeg_id:
# - NhÃ³m dá»¯ liá»‡u theo eeg_id, tÃ­nh tá»•ng giÃ¡ trá»‹ cá»§a cÃ¡c nhÃ£n má»¥c tiÃªu (TARGETS).
# - Ä�iá»�u nÃ y giá»‘ng nhÆ° "Ä‘áº¿m phiáº¿u báº§u" cho tá»«ng loáº¡i nhÃ£n (seizure, LPD, GPD, ...).
tmp = df.groupby('eeg_id')[TARGETS].agg('sum') 


# ThÃªm giÃ¡ trá»‹ tá»•ng cá»§a tá»«ng nhÃ£n má»¥c tiÃªu vÃ o train DataFrame.
for t in TARGETS:
    train[t] = tmp[t].values

# Chuáº©n hÃ³a dá»¯ liá»‡u nhÃ£n má»¥c tiÃªu:
# - Chuyá»ƒn Ä‘á»•i sá»‘ lÆ°á»£ng phiáº¿u báº§u thÃ nh xÃ¡c suáº¥t báº±ng cÃ¡ch chuáº©n hÃ³a tá»•ng cá»§a chÃºng vá»� 1.
# - Ä�Ã¢y lÃ  bÆ°á»›c quan trá»�ng Ä‘á»ƒ sá»­ dá»¥ng dá»¯ liá»‡u trong bÃ i toÃ¡n phÃ¢n loáº¡i.
y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1,keepdims=True)


# GÃ¡n giÃ¡ trá»‹ chuáº©n hÃ³a vÃ o train DataFrame.
train[TARGETS] = y_data

# ThÃªm thÃ´ng tin Ä‘Ã¡nh giÃ¡ tá»« chuyÃªn gia:
# - Vá»›i má»—i eeg_id, láº¥y giÃ¡ trá»‹ Ä‘áº§u tiÃªn cá»§a expert_consensus (nháº­n Ä‘á»‹nh cá»§a chuyÃªn gia vá»� phÃ¢n Ä‘oáº¡n EEG nÃ y).
tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')

# ThÃªm vÃ o train
train['target'] = tmp

# Reset index Ä‘á»ƒ eeg_id trá»Ÿ thÃ nh má»™t cá»™t thay vÃ¬ chá»‰ sá»‘ index.
train = train.reset_index()

# KÃ­ch thÆ°á»›c cá»§a táº­p train sau khi xá»­ lÃ½.
print('Train non-overlapp eeg_id shape:', train.shape )

train.head()


READ_SPEC_FILES = False  # Náº¿u READ_SPEC_FILES = False, code sáº½ Ä‘á»�c tá»‡p káº¿t há»£p thay vÃ¬ cÃ¡c tá»‡p riÃªng láº».
FEATURE_ENGINEER = True  # Báº­t cháº¿ Ä‘á»™ trÃ­ch xuáº¥t Ä‘áº·c trÆ°ng (feature engineering).


%%time
# Ä�á»‹nh nghÄ©a Ä‘Æ°á»�ng dáº«n chá»©a dá»¯ liá»‡u spectrogram
PATH = '/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/'

# Láº¥y danh sÃ¡ch táº¥t cáº£ cÃ¡c tá»‡p trong thÆ° má»¥c nÃ y
files = os.listdir(PATH)

# Sá»‘ lÆ°á»£ng tá»‡p .parquet cÃ³ trong thÆ° má»¥c
print(f'There are {len(files)} spectrogram parquets')


if READ_SPEC_FILES:    
    spectrograms = {}
    for i, f in enumerate(files):
        if i % 100 == 0: print(i, ', ', end='')  # In tiáº¿n trÃ¬nh Ä‘á»�c dá»¯ liá»‡u sau má»—i 100 tá»‡p
        tmp = pd.read_parquet(f'{PATH}{f}')  # Ä�á»�c tá»‡p .parquet
        name = int(f.split('.')[0])  # Láº¥y ID tá»« tÃªn file (loáº¡i bá»� pháº§n má»Ÿ rá»™ng .parquet)
        spectrograms[name] = tmp.iloc[:, 1:].values  # LÆ°u dá»¯ liá»‡u spectrogram (bá»� cá»™t Ä‘áº§u tiÃªn)
else:
    spectrograms = np.load('/kaggle/input/brain-spectrograms/specs.npy', allow_pickle=True).item() 


%time
# ENGINEER FEATURES
import warnings
warnings.filterwarnings('ignore')

# Ä�oáº¡n code nÃ y táº¡o ra cÃ¡c Ä‘áº·c trÆ°ng tá»« dá»¯ liá»‡u spectrogram Ä‘á»ƒ sá»­ dá»¥ng trong mÃ´ hÃ¬nh.
# CÃ¡c Ä‘áº·c trÆ°ng Ä‘Æ°á»£c tÃ­nh báº±ng cÃ¡ch láº¥y giÃ¡ trá»‹ trung bÃ¬nh (mean) vÃ  giÃ¡ trá»‹ nhá»� nháº¥t (min) theo thá»�i gian 
# trÃªn má»—i trong sá»‘ 400 táº§n sá»‘ cá»§a spectrogram.
# Hai loáº¡i cá»­a sá»• thá»�i gian Ä‘Æ°á»£c sá»­ dá»¥ng Ä‘á»ƒ tÃ­nh toÃ¡n:
# - Cá»­a sá»• 10 phÃºt (_mean_10m, _min_10m).
# - Cá»­a sá»• 20 giÃ¢y (_mean_20s, _min_20s).
# QuÃ¡ trÃ¬nh nÃ y táº¡o ra tá»•ng cá»™ng 1600 Ä‘áº·c trÆ°ng (400 táº§n sá»‘ Ã— 4 phÃ©p tÃ­nh) cho má»—i EEG ID.

# TrÃ­ch xuáº¥t danh sÃ¡ch cÃ¡c cá»™t spectrogram (bá»� cá»™t Ä‘áº§u tiÃªn)
SPEC_COLS = pd.read_parquet(f'{PATH}1000086677.parquet').columns[1:]

FEATURES = [f'{c}_mean_10m' for c in SPEC_COLS]  # GiÃ¡ trá»‹ trung bÃ¬nh trÃªn 10 phÃºt
FEATURES += [f'{c}_min_10m' for c in SPEC_COLS]   # GiÃ¡ trá»‹ nhá»� nháº¥t trÃªn 10 phÃºt
FEATURES += [f'{c}_mean_20s' for c in SPEC_COLS]  # GiÃ¡ trá»‹ trung bÃ¬nh trÃªn 20 giÃ¢y
FEATURES += [f'{c}_min_20s' for c in SPEC_COLS]   # GiÃ¡ trá»‹ nhá»� nháº¥t trÃªn 20 giÃ¢y

print(f'ChÃºng ta Ä‘ang táº¡o {len(FEATURES)} Ä‘áº·c trÆ°ng cho {len(train)} máº«u dá»¯ liá»‡u... ', end='')

# Má»™t ma tráº­n dá»¯ liá»‡u `data` Ä‘Æ°á»£c khá»Ÿi táº¡o Ä‘á»ƒ lÆ°u trá»¯ cÃ¡c Ä‘áº·c trÆ°ng má»›i cho má»—i `eeg_id` trong DataFrame `train`.
# Ä�á»‘i vá»›i má»—i dÃ²ng trong `train`, Ä‘oáº¡n code sáº½ tÃ­nh toÃ¡n giÃ¡ trá»‹ trung bÃ¬nh (mean) vÃ  giÃ¡ trá»‹ nhá»� nháº¥t (min) 
# trong cÃ¡c cá»­a sá»• thá»�i gian Ä‘Æ°á»£c chá»‰ Ä‘á»‹nh (10 phÃºt vÃ  20 giÃ¢y).
# CÃ¡c giÃ¡ trá»‹ Ä‘Ã£ tÃ­nh toÃ¡n nÃ y sau Ä‘Ã³ Ä‘Æ°á»£c lÆ°u vÃ o ma tráº­n dá»¯ liá»‡u.
# Cuá»‘i cÃ¹ng, ma tráº­n nÃ y Ä‘Æ°á»£c thÃªm vÃ o DataFrame `train` dÆ°á»›i dáº¡ng cÃ¡c cá»™t má»›i.

if FEATURE_ENGINEER:
    # Khá»Ÿi táº¡o ma tráº­n dá»¯ liá»‡u rá»—ng Ä‘á»ƒ lÆ°u Ä‘áº·c trÆ°ng
    data = np.zeros((len(train), len(FEATURES)))  
    
    # Duyá»‡t tá»«ng máº«u dá»¯ liá»‡u trong train
    for k in range(len(train)):
        if k % 100 == 0: print(k, ', ', end='')  # Cá»© má»—i 100 máº«u in sá»‘ lÆ°á»£ng Ä‘Ã£ xá»­ lÃ½
        
        row = train.iloc[k]  # Láº¥y má»™t dÃ²ng tá»« DataFrame train
        r = int((row['min'] + row['max']) // 4)  # XÃ¡c Ä‘á»‹nh vá»‹ trÃ­ trung tÃ¢m Ä‘á»ƒ trÃ­ch xuáº¥t Ä‘áº·c trÆ°ng
        
        # ğŸš€ TÃ­nh toÃ¡n Ä‘áº·c trÆ°ng trÃªn cá»­a sá»• 10 phÃºt
        x = np.nanmean(spectrograms[row.spec_id][r:r+300, :], axis=0)  # Trung bÃ¬nh 10 phÃºt
        data[k, :400] = x  # LÆ°u vÃ o 400 cá»™t Ä‘áº§u
        
        x = np.nanmin(spectrograms[row.spec_id][r:r+300, :], axis=0)  # GiÃ¡ trá»‹ nhá»� nháº¥t 10 phÃºt
        data[k, 400:800] = x  # LÆ°u vÃ o cá»™t 400 - 799

        # ğŸš€ TÃ­nh toÃ¡n Ä‘áº·c trÆ°ng trÃªn cá»­a sá»• 20 giÃ¢y
        x = np.nanmean(spectrograms[row.spec_id][r+145:r+155, :], axis=0)  # Trung bÃ¬nh 20 giÃ¢y
        data[k, 800:1200] = x  # LÆ°u vÃ o cá»™t 800 - 1199

        x = np.nanmin(spectrograms[row.spec_id][r+145:r+155, :], axis=0)  # GiÃ¡ trá»‹ nhá»� nháº¥t 20 giÃ¢y
        data[k, 1200:1600] = x  # LÆ°u vÃ o cá»™t 1200 - 1599

    # GÃ¡n cÃ¡c Ä‘áº·c trÆ°ng vá»«a tÃ­nh vÃ o DataFrame train
    train[FEATURES] = data
else:
    train = pd.read_parquet('/kaggle/input/brain-spectrograms/train.pqt')

print('New train shape:',train.shape)


from scipy import signal
from sklearn.decomposition import PCA


import numpy as np
from scipy import signal

def extract_frequency_band_features(segment):
    """
    HÃ m nÃ y trÃ­ch xuáº¥t cÃ¡c Ä‘áº·c trÆ°ng cá»§a dáº£i táº§n sá»‘ EEG tá»« má»™t Ä‘oáº¡n tÃ­n hiá»‡u EEG.

    Ä�áº·c trÆ°ng Ä‘Æ°á»£c trÃ­ch xuáº¥t tá»« cÃ¡c dáº£i táº§n sá»‘ phá»• biáº¿n cá»§a EEG:
    - Delta: 0.5 - 4 Hz
    - Theta: 4 - 8 Hz
    - Alpha: 8 - 12 Hz
    - Beta: 12 - 30 Hz
    - Gamma: 30 - 45 Hz
    
    Ä�á»‘i vá»›i má»—i dáº£i táº§n, cÃ¡c Ä‘áº·c trÆ°ng sau Ä‘Æ°á»£c tÃ­nh toÃ¡n:
    - Trung bÃ¬nh (mean)
    - Ä�á»™ lá»‡ch chuáº©n (standard deviation)
    - GiÃ¡ trá»‹ lá»›n nháº¥t (max)
    - GiÃ¡ trá»‹ nhá»� nháº¥t (min)

    Ä�áº§u vÃ o:
    - segment: Má»™t Ä‘oáº¡n tÃ­n hiá»‡u EEG (numpy array).

    Ä�áº§u ra:
    - band_features: Danh sÃ¡ch chá»©a Ä‘áº·c trÆ°ng cá»§a táº¥t cáº£ cÃ¡c dáº£i táº§n EEG.
    """

    # Ä�á»‹nh nghÄ©a cÃ¡c dáº£i táº§n sá»‘ EEG
    eeg_bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 45)
    }
    
    band_features = []  # Danh sÃ¡ch lÆ°u Ä‘áº·c trÆ°ng cá»§a tá»«ng dáº£i táº§n

    for band in eeg_bands:
        low, high = eeg_bands[band]  # Láº¥y giÃ¡ trá»‹ táº§n sá»‘ tháº¥p nháº¥t vÃ  cao nháº¥t cá»§a dáº£i táº§n hiá»‡n táº¡i

        # Ã�p dá»¥ng bá»™ lá»�c thÃ´ng dáº£i (band-pass filter) Ä‘á»ƒ chá»‰ giá»¯ láº¡i tÃ­n hiá»‡u trong khoáº£ng táº§n sá»‘ cáº§n thiáº¿t
        band_pass_filter = signal.butter(
            3, [low, high], btype='bandpass', fs=200, output='sos'
        )  # Bá»™ lá»�c báº­c 3, táº§n sá»‘ láº¥y máº«u 200Hz

        # Lá»�c tÃ­n hiá»‡u EEG theo dáº£i táº§n sá»‘ hiá»‡n táº¡i
        filtered = signal.sosfilt(band_pass_filter, segment)

        # TrÃ­ch xuáº¥t cÃ¡c Ä‘áº·c trÆ°ng thá»‘ng kÃª tá»« tÃ­n hiá»‡u Ä‘Ã£ lá»�c
        band_features.extend([
            np.nanmean(filtered),  # GiÃ¡ trá»‹ trung bÃ¬nh cá»§a tÃ­n hiá»‡u trong dáº£i táº§n
            np.nanstd(filtered),   # Ä�á»™ lá»‡ch chuáº©n cá»§a tÃ­n hiá»‡u trong dáº£i táº§n
            np.nanmax(filtered),   # GiÃ¡ trá»‹ lá»›n nháº¥t cá»§a tÃ­n hiá»‡u trong dáº£i táº§n
            np.nanmin(filtered)    # GiÃ¡ trá»‹ nhá»� nháº¥t cá»§a tÃ­n hiá»‡u trong dáº£i táº§n
        ])

    return band_features  # Tráº£ vá»� danh sÃ¡ch chá»©a Ä‘áº·c trÆ°ng cá»§a táº¥t cáº£ dáº£i táº§n


import time
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import numpy as np

# Khá»Ÿi táº¡o mÃ´ hÃ¬nh PCA, giá»¯ láº¡i 95% phÆ°Æ¡ng sai cá»§a dá»¯ liá»‡u
pca = PCA(n_components=0.95)
print("PCA model initialized.")

# Khá»Ÿi táº¡o ma tráº­n dá»¯ liá»‡u gá»‘c Ä‘á»ƒ lÆ°u trá»¯ Ä‘áº·c trÆ°ng
num_rows = len(train)  # Sá»‘ lÆ°á»£ng máº«u trong táº­p dá»¯ liá»‡u
num_features = 20 * n_channels  # 20 Ä‘áº·c trÆ°ng cho má»—i kÃªnh EEG

# Táº¡o ma tráº­n dá»¯ liá»‡u vá»›i giÃ¡ trá»‹ ban Ä‘áº§u lÃ  0
data_original = np.zeros((num_rows, num_features))

print("Starting feature extraction and PCA processing...")

# Báº¯t Ä‘áº§u Ä‘o thá»�i gian thá»±c thi
start_time = time.time()

# VÃ²ng láº·p Ä‘á»ƒ trÃ­ch xuáº¥t Ä‘áº·c trÆ°ng tá»« tá»«ng EEG sample
for k in range(num_rows):
    if k % 1000 == 0:
        print(f"Processing row {k} of {num_rows}...")  # In tiáº¿n trÃ¬nh má»—i 1000 máº«u

    row = train.iloc[k]  # Láº¥y má»™t dÃ²ng dá»¯ liá»‡u EEG
    r = int((row['min'] + row['max']) // 4)  # XÃ¡c Ä‘á»‹nh vá»‹ trÃ­ trung tÃ¢m Ä‘á»ƒ láº¥y tÃ­n hiá»‡u EEG

    # Láº¥y má»™t Ä‘oáº¡n tÃ­n hiá»‡u EEG tá»« spectrograms
    eeg_segment = spectrograms[row.spec_id][r:r+300, :]

    # Danh sÃ¡ch lÆ°u trá»¯ Ä‘áº·c trÆ°ng cá»§a táº¥t cáº£ kÃªnh EEG
    all_channel_features = []
    
    # Duyá»‡t qua tá»«ng kÃªnh EEG Ä‘á»ƒ trÃ­ch xuáº¥t Ä‘áº·c trÆ°ng
    for i in range(n_channels):
        channel_features = extract_frequency_band_features(eeg_segment[:, i])  # Gá»�i hÃ m trÃ­ch xuáº¥t Ä‘áº·c trÆ°ng
        all_channel_features.extend(channel_features)  # Gá»™p táº¥t cáº£ Ä‘áº·c trÆ°ng láº¡i
    
    # LÆ°u Ä‘áº·c trÆ°ng vÃ o ma tráº­n dá»¯ liá»‡u
    data_original[k, :] = all_channel_features

print("Data matrix constructed")

# Xá»­ lÃ½ giÃ¡ trá»‹ NaN trong ma tráº­n dá»¯ liá»‡u
imputer = SimpleImputer(strategy='mean')  # Thay tháº¿ giÃ¡ trá»‹ NaN báº±ng giÃ¡ trá»‹ trung bÃ¬nh cá»§a má»—i cá»™t
data_imputed = imputer.fit_transform(data_original)  # Ã�p dá»¥ng Imputer Ä‘á»ƒ xá»­ lÃ½ dá»¯ liá»‡u

print(f"NaN values handled. Imputed data matrix shape: {data_imputed.shape}")

# Huáº¥n luyá»‡n mÃ´ hÃ¬nh PCA trÃªn dá»¯ liá»‡u Ä‘Ã£ Ä‘Æ°á»£c xá»­ lÃ½ NaN
pca.fit(data_imputed)
print("PCA fitting completed.")

# Biáº¿n Ä‘á»•i dá»¯ liá»‡u sang khÃ´ng gian má»›i vá»›i sá»‘ chiá»�u tháº¥p hÆ¡n
data_pca = pca.transform(data_imputed)

# Ä�áº·t tÃªn cho cÃ¡c Ä‘áº·c trÆ°ng PCA
pca_feature_columns = [f'pca_feature_{i}' for i in range(data_pca.shape[1])]

# ThÃªm cÃ¡c Ä‘áº·c trÆ°ng PCA vÃ o DataFrame train
train[pca_feature_columns] = data_pca

# Ä�o thá»�i gian thá»±c thi toÃ n bá»™ quy trÃ¬nh
total_time = time.time() - start_time
print(f"Total processing time: {total_time:.2f} seconds.")


train.head()


from sklearn.preprocessing import StandardScaler

# Danh sÃ¡ch cÃ¡c cá»™t KHÃ”NG thá»±c hiá»‡n chuáº©n hÃ³a (cÃ¡c cá»™t ID vÃ  nhÃ£n má»¥c tiÃªu)
excluded_columns = [
    'eeg_id', 'spec_id', 'min', 'max', 'patient_id',  # ThÃ´ng tin EEG
    'seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote',  # Phiáº¿u cháº©n Ä‘oÃ¡n
    'target'  # NhÃ£n má»¥c tiÃªu
]

# LÆ°u láº¡i cÃ¡c cá»™t bá»‹ loáº¡i trá»« Ä‘á»ƒ giá»¯ nguyÃªn dá»¯ liá»‡u nÃ y
excluded_data = train[excluded_columns]

# Táº¡o DataFrame chá»‰ chá»©a cÃ¡c cá»™t cáº§n Ä‘Æ°á»£c chuáº©n hÃ³a (bá»� cÃ¡c cá»™t bá»‹ loáº¡i trá»«)
features = train.drop(columns=excluded_columns)

# Khá»Ÿi táº¡o StandardScaler Ä‘á»ƒ chuáº©n hÃ³a dá»¯ liá»‡u
scaler = StandardScaler()

# Ã�p dá»¥ng StandardScaler lÃªn dá»¯ liá»‡u Ä‘áº·c trÆ°ng
features_scaled = scaler.fit_transform(features)

# Chuyá»ƒn Ä‘á»•i dá»¯ liá»‡u Ä‘Ã£ chuáº©n hÃ³a thÃ nh DataFrame vá»›i cÃ¹ng tÃªn cá»™t
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

# Káº¿t há»£p láº¡i dá»¯ liá»‡u Ä‘Ã£ chuáº©n hÃ³a vá»›i cÃ¡c cá»™t bá»‹ loáº¡i trá»« (giá»¯ nguyÃªn index)
train_scaled_df = pd.concat([excluded_data.reset_index(drop=True), features_scaled_df], axis=1)

# Hiá»ƒn thá»‹ DataFrame sau khi chuáº©n hÃ³a
train_scaled_df


train_scaled_df.info()


# import xgboost as xgb
# import gc
# from sklearn.model_selection import KFold, GroupKFold

# print('XGBoost version', xgb.__version__)


# all_oof = []
# all_true = []
# TARS = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}

# gkf = GroupKFold(n_splits=5)
# for i, (train_index, valid_index) in enumerate(gkf.split(train , train .target, train .patient_id)):   
    
#     print('#'*25)
#     print(f'### Fold {i+1}')
#     print(f'### train size {len(train_index)}, valid size {len(valid_index)}')
#     print('#'*25)
    
#     model = xgb.XGBClassifier(
#         objective='multi:softprob', 
#         num_class=len(TARS),
#         learning_rate = 0.1, 
                      
# #         tree_method='gpu_hist',  #skip GPU acceleration
#     )
    
#     # Prepare training and validation data
#     X_train = train.loc[train_index, FEATURES]
#     y_train = train.loc[train_index, 'target'].map(TARS)
#     X_valid = train.loc[valid_index, FEATURES]
#     y_valid = train.loc[valid_index, 'target'].map(TARS)
    
#     model.fit(X_train, y_train, 
#               eval_set=[(X_valid, y_valid)], 
#               verbose=True, 
#               early_stopping_rounds=10)
#     model.save_model(f'XGB_v{VER}_f{i}.model')
    
#     oof = model.predict_proba(X_valid)
#     all_oof.append(oof)
#     all_true.append(train.loc[valid_index, TARGETS].values)
    
#     del X_train, y_train, X_valid, y_valid, oof
#     gc.collect()
    
# all_oof = np.concatenate(all_oof)
# all_true = np.concatenate(all_true)


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


# TOP = 30

# # Assuming 'model' is your trained model
# feature_importance = model.feature_importances_

# # Get the feature names from 'train'
# feature_names = train.columns

# # Sort the feature importances and get the indices of the sorted array
# sorted_idx = np.argsort(feature_importance)

# # Plot only the top 'TOP' features
# fig = plt.figure(figsize=(10, 8))
# plt.barh(np.arange(len(sorted_idx))[-TOP:], feature_importance[sorted_idx][-TOP:], align='center')
# plt.yticks(np.arange(len(sorted_idx))[-TOP:], feature_names[sorted_idx][-TOP:])
# plt.title(f'Feature Importance - Top {TOP}')
# plt.show()


# test = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/test.csv')
# print('Test shape',test.shape)
# test.head()


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
# # This process results in 1600 features (400 features Ã— 4 calculations) for each EEG ID.

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



# # FEATURE ENGINEER TEST
# PATH2 = '/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/'
# data = np.zeros((len(test),len(FEATURES)))
    
# for k in range(len(test)):
#     row = test.iloc[k]
#     s = int( row.spectrogram_id )
#     spec = pd.read_parquet(f'{PATH2}{s}.parquet')
    
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

# test[FEATURES] = data
# print('New test shape',test.shape)


# # INFER XGBOOST ON TEST
# preds = []

# for i in range(5):
#     print(i, ', ', end='')
    
#     # Load the XGBoost model
#     model = xgb.XGBClassifier()
#     model.load_model(f'XGB_v{VER}_f{i}.model')
    
#     # Make predictions
#     pred = model.predict_proba(test[FEATURES])
#     preds.append(pred)

# # Average the predictions from each fold
# pred = np.mean(preds, axis=0)
# print()
# print('Test preds shape', pred.shape)


# sub = pd.DataFrame({'eeg_id':test.eeg_id.values})
# sub[TARGETS] = pred
# sub.to_csv('submission.csv',index=False)
# print('Submission shape',sub.shape)
# sub.head()


# # SANITY CHECK TO CONFIRM PREDICTIONS SUM TO ONE
# sub.iloc[:,-6:].sum(axis=1)

