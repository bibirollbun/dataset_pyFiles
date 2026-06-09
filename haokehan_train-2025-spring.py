import os
import logging
import random
import gc
import time
import cv2
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import timm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)



class CFG:

    train_audio_path = '/kaggle/input/birdclef-2025/train_audio/' # ä½ çš„è®­ç»ƒéŸ³é¢‘æ–‡ä»¶å­˜æ”¾è·¯å¾„
    train_metadata_csv = '/kaggle/input/birdclef-2025/train.csv' # åŒ…å�«éŸ³é¢‘æ–‡ä»¶å��å’Œå¯¹åº”é¸Ÿç±»æ ‡ç­¾çš„CSVæ–‡ä»¶è·¯å¾„

    # 2. ç‰©ç§�åˆ†ç±»æ–‡ä»¶ (ç”¨äº�è�·å�–ç±»åˆ«æ€»æ•°å’Œæ ‡ç­¾æ˜ å°„)
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'

    # 3. ä½ è®­ç»ƒå¥½çš„æ¨¡å�‹çš„ä¿�å­˜è·¯å¾„ (æ£€æŸ¥ç‚¹)
    output_model_dir = '/kaggle/working/models/' # æˆ–è€…å…¶ä»–ä»»ä½•ä½ æœ‰å†™å…¥æ�ƒé™�çš„ç›®å½•
                                                 # æ¯�ä¸€æŠ˜æˆ–æ¯�ä¸€ä¸ªepochçš„æ¨¡å�‹éƒ½ä¼šä¿�å­˜åœ¨è¿™é‡Œ
  # --- éŸ³é¢‘å’Œæ¢…å°”é¢‘è°±å›¾å�‚æ•° (é€šå¸¸å’Œæ�¨ç�†æ—¶ä¿�æŒ�ä¸€è‡´) ---
    FS = 32000  
    WINDOW_SIZE = 5  
    
    N_FFT = 1024
    HOP_LENGTH = 64
    N_MELS = 136
    FMIN = 20
    FMAX = 16000
    TARGET_SHAPE = (256, 256)
    # --- æ¨¡å�‹æ�¶æ�„ ---
    model_name = 'efficientnet_b0'
    in_channels = 1 # è¾“å…¥é€šé�“æ•° (1ä»£è¡¨å�•é€šé�“æ¢…å°”é¢‘è°±å›¾)
    # æ˜¯å�¦ä½¿ç”¨ ImageNet é¢„è®­ç»ƒæ�ƒé‡�ä½œä¸ºä½ éª¨å¹²ç½‘ç»œçš„èµ·ç‚¹
    # è¿™å¯¹äº�è¿�ç§»å­¦ä¹ é€šå¸¸æ˜¯æœ‰ç›Šçš„
    pretrained_on_imagenet = True # è®¾ç½®ä¸º True æ�¥åŠ è½½ ImageNet æ�ƒé‡�åˆ°éª¨å¹²ç½‘ç»œ
    
    # --- è®­ç»ƒè¶…å�‚æ•° ---
    # 1. é€šç”¨è®¾ç½®
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # å¦‚æ�œæœ‰GPUåˆ™ä½¿ç”¨GPU
    seed = 42               # ç”¨äº�ä¿�è¯�å®�éªŒå�¯å¤�ç�°çš„éš�æœºç§�å­�
    num_epochs = 10         # æ€»è®­ç»ƒè½®æ¬¡
    train_batch_size = 32   # è®­ç»ƒæ‰¹æ¬¡å¤§å°�
    valid_batch_size = 64   # éªŒè¯�æ‰¹æ¬¡å¤§å°� (é€šå¸¸å�¯ä»¥è®¾å¤§ä¸€äº›ï¼Œå› ä¸ºéªŒè¯�æ—¶æ²¡æœ‰å��å�‘ä¼ æ’­)

    # 2. ä¼˜åŒ–å™¨
    optimizer_name = 'AdamW' # ä¾‹å¦‚: 'Adam', 'AdamW', 'SGD'
    learning_rate = 1e-3     # å­¦ä¹ ç�‡
    weight_decay = 1e-5      # æ�ƒé‡�è¡°å‡� (ç”¨äº�åƒ� AdamW è¿™æ ·çš„ä¼˜åŒ–å™¨)

    # 3. å­¦ä¹ ç�‡è°ƒåº¦å™¨ (å�¯é€‰ï¼Œä½†é€šå¸¸æœ‰å¸®åŠ©)
    scheduler_name = 'CosineAnnealingLR' # ä¾‹å¦‚: 'StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR'
    lr_scheduler_params = {  # å­¦ä¹ ç�‡è°ƒåº¦å™¨çš„å…·ä½“å�‚æ•°
        'T_max': num_epochs, # å¯¹äº� CosineAnnealingLR
        'eta_min': 1e-6      # å¯¹äº� CosineAnnealingLR
    }
    # æˆ–è€…å¯¹äº� StepLR: {'step_size': 10, 'gamma': 0.1}
    if num_epochs != 50: # å¦‚æ�œ num_epochs è¢«ä¿®æ”¹ï¼Œéœ€è¦�ç¡®ä¿� T_max ä¹Ÿæ›´æ–°
        lr_scheduler_params['T_max'] = num_epochs
    # 4. æ�Ÿå¤±å‡½æ•°
    loss_fn_name = 'CrossEntropyLoss' # å¦‚æ�œä½ çš„æ ‡ç­¾æ˜¯æ¯�ä¸ªæ ·æœ¬ä¸€ä¸ªé¸Ÿç±»æ•´æ•°IDï¼Œå°±ç”¨è¿™ä¸ª
                                      # å¦‚æ�œä¸€ä¸ªå£°éŸ³é‡Œå�¯èƒ½æœ‰å¤šç§�é¸Ÿ (å¤šæ ‡ç­¾)ï¼Œå�¯èƒ½ç”¨ 'BCEWithLogitsLoss'
    # --- æ•°æ�®å¤„ç�†ä¸�éªŒè¯� ---
    num_workers = 2         # DataLoader ä½¿ç”¨çš„å·¥ä½œè¿›ç¨‹æ•°
    # KæŠ˜äº¤å�‰éªŒè¯� (åœ¨ç«�èµ›ä¸­å¾ˆå¸¸è§�)
    n_folds = 5             # æ€»å…±åˆ†å‡ æŠ˜
    current_fold_to_train = 0 # å½“å‰�è®­ç»ƒçš„æ˜¯ç¬¬å‡ æŠ˜ (ä»�0åˆ° n_folds-1)

    # --- æ•°æ�®å¢�å¼º (å�¯é€‰, ç”¨äº�è®­ç»ƒæ•°æ�®) ---
    # ä½ å�¯ä»¥åœ¨è¿™é‡Œå®šä¹‰éŸ³é¢‘æ•°æ�®å¢�å¼ºçš„å�‚æ•°ï¼Œä¾‹å¦‚:
    # use_noise_injection = True  # æ˜¯å�¦ä½¿ç”¨å™ªå£°æ³¨å…¥
    # noise_level = 0.005         # å™ªå£°æ°´å¹³
    # use_random_shift = True     # æ˜¯å�¦ä½¿ç”¨éš�æœºæ—¶é—´å¹³ç§»
    # ç­‰ç­‰...

    # --- æ—¥å¿—ä¸�æ¨¡å�‹ä¿�å­˜ ---
    print_freq_epochs = 1   # æ¯�éš”å¤šå°‘è½®æ‰“å�°ä¸€æ¬¡è®­ç»ƒæ—¥å¿—
    save_best_model_only = True # å�ªä¿�å­˜éªŒè¯�é›†ä¸Šè¡¨ç�°æœ€å¥½çš„æ¨¡å�‹

    # --- ç”¨äº�è°ƒè¯• (å�ªå¤„ç�†ä¸€å°�éƒ¨åˆ†æ•°æ�®) ---
    debug_mode = False
    debug_subset_size = 100 # å¦‚æ�œ debug_mode ä¸º True, ä½¿ç”¨çš„æ ·æœ¬æ•°é‡�

cfg = CFG()

cfg.lr_scheduler_params = {
    'T_max': cfg.num_epochs,
    'eta_min': 1e-6
}

print(f"--- CFG å®�ä¾‹åŒ–å�� ---")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"cfg.device è®¾ç½®ä¸º: {cfg.device}")
# ä¸‹é�¢è¿™ä¸¤è¡Œå…ˆæ³¨é‡Šæ�‰ï¼Œå› ä¸º model è¿˜æ²¡å®šä¹‰
# print(f"æ¨¡å�‹å·²ç§»åŠ¨åˆ°è®¾å¤‡: {cfg.device}")
# print(f"è®­ç»ƒå°†åœ¨è®¾å¤‡: {cfg.device} ä¸Šè¿›è¡Œ")
print(f"æ¢…å°”é¢‘è°±å›¾å�‚æ•°: N_FFT={cfg.N_FFT}, HOP_LENGTH={cfg.HOP_LENGTH}, N_MELS={cfg.N_MELS}")
print(f"ç›®æ ‡å›¾åƒ�å½¢çŠ¶: {cfg.TARGET_SHAPE}")


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU


set_seed(cfg.seed)


# åˆ›å»ºæ¨¡å�‹ä¿�å­˜ç›®å½• (å¦‚æ�œä¸�å­˜åœ¨)
os.makedirs(cfg.output_model_dir, exist_ok=True)
print(f"æ¨¡å�‹å°†ä¿�å­˜åœ¨: {cfg.output_model_dir}")



# --- æ­¥éª¤ 2: æ•°æ�®å‡†å¤‡ ---

# 2.1. åŠ è½½å…ƒæ•°æ�®å’Œç‰©ç§�åˆ†ç±»ä¿¡æ�¯ï¼Œç¡®å®š num_classes
print("\n--- å¼€å§‹æ•°æ�®å‡†å¤‡ ---")
try:
    train_df = pd.read_csv(cfg.train_metadata_csv)
    print(f"æˆ�åŠŸåŠ è½½è®­ç»ƒå…ƒæ•°æ�®: {cfg.train_metadata_csv}, å½¢çŠ¶: {train_df.shape}")
except FileNotFoundError:
    print(f"é”™è¯¯: è®­ç»ƒå…ƒæ•°æ�®æ–‡ä»¶æœªæ‰¾åˆ°äº� {cfg.train_metadata_csv}")
    # åœ¨è¿™é‡Œä½ å�¯ä»¥å�œæ­¢æ‰§è¡Œï¼Œæˆ–è€…åˆ›å»ºä¸€ä¸ªç©ºçš„DataFrameæ�¥é�¿å…�å��ç»­ä»£ç �å‡ºé”™ï¼ˆä½†ä¸�æ�¨è��ï¼‰
    # exit() # æˆ–è€… raise FileNotFoundError
    train_df = pd.DataFrame() # ä»…ä¸ºé�¿å…�ç›´æ�¥æŠ¥é”™

try:
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    print(f"æˆ�åŠŸåŠ è½½ç‰©ç§�åˆ†ç±»æ–‡ä»¶: {cfg.taxonomy_csv}, å½¢çŠ¶: {taxonomy_df.shape}")
except FileNotFoundError:
    print(f"é”™è¯¯: ç‰©ç§�åˆ†ç±»æ–‡ä»¶æœªæ‰¾åˆ°äº� {cfg.taxonomy_csv}")
    taxonomy_df = pd.DataFrame() # ä»…ä¸ºé�¿å…�ç›´æ�¥æŠ¥é”™

if not train_df.empty and not taxonomy_df.empty and 'primary_label' in taxonomy_df.columns:
    unique_labels = sorted(taxonomy_df['primary_label'].unique())
    num_classes = len(unique_labels)
    print(f"æ€»å…±çš„é¸Ÿç±»ç±»åˆ«æ•°é‡� (num_classes): {num_classes}")

    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for i, label in enumerate(unique_labels)}

    if 'primary_label' in train_df.columns:
        train_df['label_id'] = train_df['primary_label'].map(label_to_int)
        train_df.dropna(subset=['label_id'], inplace=True) # ç§»é™¤æ²¡æœ‰å¯¹åº” label_id çš„è¡Œ
        train_df['label_id'] = train_df['label_id'].astype(int)
        print(f"å¤„ç�†å�� train_df ä¸­æœ‰æ•ˆæ•°æ�®æ�¡æ•°: {len(train_df)}")
    else:
        print(f"è­¦å‘Š: train_df ä¸­ç¼ºå°‘ 'primary_label' åˆ—ã€‚")
else:
    print("é”™è¯¯æˆ–è­¦å‘Š: æ— æ³•ç¡®å®š num_classesï¼Œå› ä¸ºå…ƒæ•°æ�®æˆ–ç‰©ç§�åˆ†ç±»æ–‡ä»¶åŠ è½½å¤±è´¥æˆ–ç¼ºå°‘å…³é”®åˆ—ã€‚")
    num_classes = -1 # è¡¨ç¤ºé”™è¯¯çŠ¶æ€�



# 2.2. K æŠ˜äº¤å�‰éªŒè¯�åˆ‡åˆ†
if not train_df.empty and 'label_id' in train_df.columns:
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    train_df['fold'] = -1 # åˆ�å§‹åŒ–foldåˆ—
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label_id'])):
        train_df.loc[val_idx, 'fold'] = fold_num
    train_df['fold'] = train_df['fold'].astype(int)
    print(f"æ•°æ�®å·²åˆ’åˆ†ä¸º {cfg.n_folds} æŠ˜ã€‚")

    current_train_df = train_df[train_df.fold != cfg.current_fold_to_train].reset_index(drop=True)
    current_valid_df = train_df[train_df.fold == cfg.current_fold_to_train].reset_index(drop=True)
    print(f"å½“å‰�è®­ç»ƒæŠ˜: {cfg.current_fold_to_train}")
    print(f"è®­ç»ƒé›†å¤§å°�: {len(current_train_df)}, éªŒè¯�é›†å¤§å°�: {len(current_valid_df)}")
else:
    print("è­¦å‘Š: ç”±äº� train_df ä¸ºç©ºæˆ–ç¼ºå°‘ 'label_id'ï¼Œæ— æ³•è¿›è¡Œ K æŠ˜åˆ‡åˆ†ã€‚")
    # åˆ›å»ºç©ºçš„DataFrameä»¥é�¿å…�å��ç»­ä»£ç �å‡ºé”™ï¼Œä½†è®­ç»ƒæ— æ³•è¿›è¡Œ
    current_train_df = pd.DataFrame()
    current_valid_df = pd.DataFrame()


# 2.3. åˆ›å»ºè‡ªå®šä¹‰ PyTorch Dataset ç±» (è¿™é‡Œçš„ä»£ç �å’Œæˆ‘ä¸Šæ¬¡æ��ä¾›çš„ä¸€æ ·)
class BirdSoundDataset(Dataset):
    def __init__(self, df, cfg, audio_path_base, num_classes_fallback, augmentations=None, is_training=True): # æ·»åŠ  num_classes_fallback
        self.df = df
        self.cfg = cfg
        self.audio_path_base = audio_path_base
        self.augmentations = augmentations
        self.is_training = is_training
        if not df.empty and 'filename' in df.columns and 'label_id' in df.columns:
            self.filenames = df['filename'].values
            self.labels = df['label_id'].values
        else: # å¤„ç�†ç©ºçš„æˆ–ä¸�å®Œæ•´çš„DataFrame
            self.filenames = []
            self.labels = []
        self.num_classes_fallback = num_classes_fallback # ç”¨äº�æ ‡ç­¾æ˜¯-1çš„æƒ…å†µ

    def __len__(self):
        return len(self.filenames) # ä½¿ç”¨ self.filenames çš„é•¿åº¦

    def __getitem__(self, idx):
        if self.filenames.size == 0: # å¦‚æ�œæ²¡æœ‰æ–‡ä»¶å��ï¼Œè¿”å›�å� ä½�ç¬¦
            dummy_spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            # å¯¹äº�æ ‡ç­¾ï¼Œå¦‚æ�œç±»åˆ«æ•°å·²çŸ¥ï¼Œæˆ‘ä»¬å�¯ä»¥è¿”å›�ä¸€ä¸ªæœ‰æ•ˆçš„ç±»åˆ«ï¼ˆæ¯”å¦‚0ï¼‰ï¼Œæˆ–è€…ä¸€ä¸ªç‰¹æ®Šå€¼
            # è¿™é‡Œè¿”å›�ä¸€ä¸ªåœ¨ç±»åˆ«èŒƒå›´å†…çš„å€¼ï¼Œæˆ–è€…å¦‚æ�œnum_classes_fallbackæ˜¯-1ï¼Œå°±è¿”å›�0
            dummy_label_val = 0 if self.num_classes_fallback == -1 else (self.num_classes_fallback -1 if self.num_classes_fallback > 0 else 0)
            return torch.tensor(dummy_spec).unsqueeze(0), torch.tensor(dummy_label_val, dtype=torch.long)

        filename = self.filenames[idx]
        audio_file_path = os.path.join(self.audio_path_base, filename)

        try:
            y, sr = librosa.load(audio_file_path, sr=self.cfg.FS, mono=True)
        except Exception as e:
            print(f"é”™è¯¯: æ— æ³•åŠ è½½éŸ³é¢‘æ–‡ä»¶ {audio_file_path}: {e}")
            dummy_spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            # è¿”å›�ä¸€ä¸ªåœ¨ç±»åˆ«èŒƒå›´å†…çš„æ ‡ç­¾ï¼Œæˆ–è€…å¦‚æ�œ num_classes_fallback æ˜¯-1ï¼Œå°±è¿”å›�0
            # é�¿å…�æ ‡ç­¾æ˜¯-1å¯¼è‡´ CrossEntropyLoss å‡ºé”™
            error_label_val = 0 if self.num_classes_fallback == -1 else (self.num_classes_fallback -1 if self.num_classes_fallback > 0 else 0)
            return torch.tensor(dummy_spec).unsqueeze(0), torch.tensor(error_label_val, dtype=torch.long)

        target_samples = int(self.cfg.WINDOW_SIZE * self.cfg.FS)
        current_samples = len(y)

        if current_samples > target_samples:
            if self.is_training:
                start = random.randint(0, current_samples - target_samples)
            else:
                start = (current_samples - target_samples) // 2
            y_segment = y[start : start + target_samples]
        elif current_samples < target_samples:
            y_segment = np.pad(y, (0, target_samples - current_samples), 'constant')
        else:
            y_segment = y

        # (å�¯é€‰éŸ³é¢‘å¢�å¼º)
        # if self.augmentations:
        # y_segment = self.augmentations(samples=y_segment, sample_rate=self.cfg.FS)

        melspec = librosa.feature.melspectrogram(
            y=y_segment, sr=self.cfg.FS, n_fft=self.cfg.N_FFT,
            hop_length=self.cfg.HOP_LENGTH, n_mels=self.cfg.N_MELS,
            fmin=self.cfg.FMIN, fmax=self.cfg.FMAX
        )
        melspec_db = librosa.power_to_db(melspec, ref=np.max)
        norm_melspec = (melspec_db - melspec_db.min()) / (melspec_db.max() - melspec_db.min() + 1e-8)

        if norm_melspec.shape != self.cfg.TARGET_SHAPE:
            resized_melspec = cv2.resize(norm_melspec, self.cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
        else:
            resized_melspec = norm_melspec

        image = np.expand_dims(resized_melspec, axis=0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        image_tensor = torch.tensor(image, dtype=torch.float32)

        return image_tensor, label


# å®�ä¾‹åŒ– Dataset
# ç¡®ä¿� current_train_df å’Œ current_valid_df ä¸�æ˜¯ç©ºçš„ï¼Œå¹¶ä¸” num_classes æ˜¯æœ‰æ•ˆå€¼
if not current_train_df.empty and not current_valid_df.empty and num_classes > 0:
    train_dataset = BirdSoundDataset(current_train_df, cfg, cfg.train_audio_path, num_classes, is_training=True)
    valid_dataset = BirdSoundDataset(current_valid_df, cfg, cfg.train_audio_path, num_classes, is_training=False)
    print("Dataset å®�ä¾‹åŒ–æˆ�åŠŸã€‚")
else:
    print("è­¦å‘Š: ç”±äº�æ•°æ�®å¸§ä¸ºç©ºæˆ–num_classesæ— æ•ˆï¼ŒDataset å�¯èƒ½æœªæ­£ç¡®å®�ä¾‹åŒ–æˆ–ä¸ºç©ºã€‚")
    # åˆ›å»ºç©ºçš„Dataseté�¿å…�å��ç»­ä»£ç �ç›´æ�¥æŠ¥é”™ï¼Œä½†è®­ç»ƒæ— æ³•è¿›è¡Œ
    train_dataset = BirdSoundDataset(pd.DataFrame(columns=['filename', 'label_id']), cfg, cfg.train_audio_path, num_classes if num_classes > 0 else 1, is_training=True)
    valid_dataset = BirdSoundDataset(pd.DataFrame(columns=['filename', 'label_id']), cfg, cfg.train_audio_path, num_classes if num_classes > 0 else 1, is_training=False)



# 2.4. åˆ›å»º PyTorch DataLoader
if len(train_dataset) > 0 and len(valid_dataset) > 0 : # ç¡®ä¿�datasetä¸�æ˜¯ç©ºçš„
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.valid_batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False
    )
    print("DataLoader å®�ä¾‹åŒ–æˆ�åŠŸã€‚")
else:
    print("è­¦å‘Š: ç”±äº� Dataset ä¸ºç©ºï¼Œæ— æ³•åˆ›å»º DataLoaderã€‚è®­ç»ƒæ— æ³•è¿›è¡Œã€‚")

print("\n--- æ•°æ�®å‡†å¤‡é˜¶æ®µç»“æ�Ÿ ---")


# --- æ­¥éª¤ 3: æ¨¡å�‹å®šä¹‰ ---
print("\n--- å¼€å§‹æ¨¡å�‹å®šä¹‰ ---")

# ç¡®ä¿� num_classes æ˜¯åœ¨æ•°æ�®å‡†å¤‡æ­¥éª¤ä¸­æ­£ç¡®è®¡ç®—å‡ºæ�¥çš„
# å¦‚æ�œä½ æ˜¯åœ¨ä¸�å�Œçš„å�•å…ƒæ ¼ä¸­è¿�è¡Œï¼Œéœ€è¦�ç¡®ä¿� num_classes åœ¨å½“å‰�ä½œç”¨åŸŸæ˜¯å�¯è®¿é—®çš„
# ä¾‹å¦‚ï¼Œå®ƒå�¯ä»¥æ˜¯å…¨å±€å�˜é‡�ï¼Œæˆ–è€…ä½ ä»�ä¸Šä¸€ä¸ªå�•å…ƒæ ¼çš„è¾“å‡ºä¸­è�·å�–å®ƒã€‚
# è¿™é‡Œæˆ‘ä»¬å�‡è®¾ num_classes å�˜é‡�å·²ç»�å­˜åœ¨å¹¶ä¸”åŒ…å�«äº†æ­£ç¡®çš„é¸Ÿç±»ç±»åˆ«æ•°é‡�ã€‚
if 'num_classes' not in globals() or num_classes <= 0:
    print("é”™è¯¯: 'num_classes' æœªå®šä¹‰æˆ–æ— æ•ˆã€‚è¯·ç¡®ä¿�åœ¨æ•°æ�®å‡†å¤‡æ­¥éª¤ä¸­å·²æ­£ç¡®è®¡ç®—ã€‚")
    # ä½ å�¯èƒ½éœ€è¦�ä»�ä¸Šä¸€ä¸ªå�•å…ƒæ ¼é‡�æ–°è�·å�–æˆ–è®¾ç½®å®ƒï¼Œä¾‹å¦‚ï¼š
    # if not taxonomy_df.empty and 'primary_label' in taxonomy_df.columns:
    #     unique_labels = sorted(taxonomy_df['primary_label'].unique())
    #     num_classes = len(unique_labels)
    # else:
    #     num_classes = 264 # æˆ–è€…ä¸€ä¸ªé»˜è®¤çš„å›�é€€å€¼ï¼Œä½†è¿™ä¸�æ�¨è��ç”¨äº�å®�é™…è®­ç»ƒ
    #     print(f"è­¦å‘Š: ä½¿ç”¨å›�é€€çš„ num_classes = {num_classes}")
    # ä¸ºäº†ç»§ç»­ï¼Œæˆ‘ä»¬å�‡è®¾ä¸€ä¸ªå€¼ï¼Œä½†å®�é™…ä¸­ä½ éœ€è¦�ç¡®ä¿�å®ƒæ˜¯æ­£ç¡®çš„
    if 'num_classes' not in globals() or num_classes <= 0: # å†�æ¬¡æ£€æŸ¥
        if not isinstance(num_classes, int) or num_classes <=0 : # ç¡®ä¿� num_classes æ˜¯æ­£æ•´æ•°
             print("é”™è¯¯ï¼šnum_classes ä¸�æ˜¯ä¸€ä¸ªæœ‰æ•ˆçš„æ­£æ•´æ•°ã€‚æ— æ³•ç»§ç»­å®šä¹‰æ¨¡å�‹ã€‚")
             # exit() # å¦‚æ�œåœ¨è„šæœ¬ä¸­ï¼Œå�¯ä»¥é€€å‡º



class BirdSoundModelTrain(nn.Module):
    def __init__(self, cfg_config, num_classes_model, pretrained_imagenet=True):
        super().__init__()
        self.cfg = cfg_config # å�¯ä»¥ç›´æ�¥ç”¨ cfgï¼Œé�¿å…�å‘½å��å†²çª�
        self.num_classes = num_classes_model

        # 1. åˆ›å»ºéª¨å¹²ç½‘ç»œ (Backbone)
        #    ä½¿ç”¨ cfg.model_name, cfg.in_channels
        #    pretrained_imagenet å�‚æ•°å†³å®šæ˜¯å�¦åŠ è½½ ImageNet æ�ƒé‡�
        self.backbone = timm.create_model(
            self.cfg.model_name,
            pretrained=pretrained_imagenet, # ä½¿ç”¨ä¼ å…¥çš„å�‚æ•°
            in_chans=self.cfg.in_channels,
            drop_rate=0.0,       # é€šå¸¸åœ¨è®­ç»ƒåˆ�æœŸæˆ–å¾®è°ƒæ—¶ï¼Œå�¯ä»¥ä»�0å¼€å§‹ï¼Œæˆ–æ ¹æ�®éœ€è¦�è°ƒæ•´
            drop_path_rate=0.0   # å�Œä¸Š
            # num_classes=0 # å�¦ä¸€ç§�ç§»é™¤å�Ÿå§‹åˆ†ç±»å¤´çš„æ–¹å¼�ï¼Œtimmä¼šè¿”å›�ä¸�å¸¦åˆ†ç±»å¤´çš„ç‰¹å¾�
        )

        # 2. è�·å�–éª¨å¹²ç½‘ç»œçš„è¾“å‡ºç‰¹å¾�ç»´åº¦å¹¶ç§»é™¤/æ›¿æ�¢å�Ÿå§‹åˆ†ç±»å™¨
        #    è¿™é‡Œçš„é€»è¾‘å’Œä½ æ�¨ç�†è„šæœ¬ä¸­çš„ç±»ä¼¼
        if 'efficientnet' in self.cfg.model_name:
            backbone_out_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity() # æ›¿æ�¢ä¸ºç©ºæ“�ä½œå±‚
        elif 'resnet' in self.cfg.model_name or 'resnext' in self.cfg.model_name:
            backbone_out_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        # ä¸ºå…¶ä»–å�¯èƒ½çš„timmæ¨¡å�‹æ·»åŠ é€šç”¨å¤„ç�† (å¦‚æ�œéœ€è¦�)
        # ä¾‹å¦‚ï¼Œå¯¹äº� Vision Transformer (vit) ç³»åˆ—:
        # elif 'vit' in self.cfg.model_name:
        #     backbone_out_features = self.backbone.head.in_features
        #     self.backbone.head = nn.Identity()
        else:
            # å°�è¯•ä¸€ä¸ªæ›´é€šç”¨çš„æ–¹æ³•è�·å�–åˆ†ç±»å™¨ç‰¹å¾�æ•°å¹¶ç§»é™¤åˆ†ç±»å™¨
            try:
                backbone_out_features = self.backbone.get_classifier().in_features
                self.backbone.reset_classifier(0, '') # num_classes=0, global_pool=''
            except AttributeError:
                # å¦‚æ�œä¸Šé�¢çš„æ–¹æ³•å¤±è´¥ï¼Œå�¯èƒ½éœ€è¦�é’ˆå¯¹ç‰¹å®šæ¨¡å�‹ç³»åˆ—æ·»åŠ å¤„ç�†
                # æˆ–è€…æ£€æŸ¥timmæ¨¡å�‹æ˜¯å¦‚ä½•å‘½å��çš„åˆ†ç±»å±‚
                print(f"è­¦å‘Š: æ— æ³•è‡ªåŠ¨ç¡®å®šæ¨¡å�‹ '{self.cfg.model_name}' çš„åˆ†ç±»å™¨è¾“å‡ºç‰¹å¾�æ•°æˆ–ç§»é™¤åˆ†ç±»å™¨ã€‚")
                print(f"è¯·æ£€æŸ¥timmåº“ä¸­è¯¥æ¨¡å�‹çš„ç»“æ�„ï¼Œæˆ–æ‰‹åŠ¨æŒ‡å®šbackbone_out_featuresã€‚")
                # ä½ å�¯èƒ½éœ€è¦�ç¡¬ç¼–ç �backbone_out_featuresï¼Œä¾‹å¦‚å¯¹äº�EfficientNet-B0æ˜¯1280
                if self.cfg.model_name == 'efficientnet_b0':
                    backbone_out_features = 1280 # EfficientNet-B0 çš„ç‰¹å¾�æ•°
                    if hasattr(self.backbone, 'classifier'):
                         self.backbone.classifier = nn.Identity()
                    else:
                         print(f"é”™è¯¯ï¼šEfficientNet-B0 æ¨¡å�‹æ²¡æœ‰ 'classifier' å±�æ€§ã€‚")
                else:
                    raise ValueError(f"æ— æ³•å¤„ç�†æ¨¡å�‹ {self.cfg.model_name} çš„åˆ†ç±»å¤´ï¼Œè¯·æ‰‹åŠ¨é€‚é…�ã€‚")


        # 3. æ·»åŠ å…¨å±€å¹³å�‡æ± åŒ–å±‚ (æˆ–å…¶å®ƒæ± åŒ–æ–¹å¼�)
        self.pooling = nn.AdaptiveAvgPool2d(output_size=1)
        # AdaptiveAvgPool2d(1) ä¼šå°† HxW çš„ç‰¹å¾�å›¾è½¬æ�¢ä¸º 1x1

        # 4. æ·»åŠ æ–°çš„åˆ†ç±»å™¨ (å…¨è¿�æ�¥å±‚)
        self.classifier = nn.Linear(backbone_out_features, self.num_classes)

    def forward(self, x):
        print(f"Input x shape: {x.shape}")
        features = self.backbone(x)
        print(f"After backbone, features shape: {features.shape}")

        # æ£€æŸ¥ features æ˜¯å�¦å·²ç»�æ˜¯ 2D (N, C)
        if features.ndim == 2: # ä¾‹å¦‚ (batch_size, num_features)
            # å¦‚æ�œå·²ç»�æ˜¯2Dï¼Œè¯´æ˜�timmæ¨¡å�‹å†…éƒ¨å�¯èƒ½å·²ç»�å�šäº†æ± åŒ–å’Œå±•å¹³
            # æ­¤æ—¶ï¼Œæˆ‘ä»¬ä¸�éœ€è¦�å†�è¿›è¡Œ pooling æ“�ä½œ
            flattened_features = features
            print(f"Features were 2D, using as flattened_features. Shape: {flattened_features.shape}")
        elif features.ndim == 4: # (batch_size, num_features, H_feat, W_feat)
            pooled_features = self.pooling(features)
            print(f"After pooling, pooled_features shape: {pooled_features.shape}")

            flattened_features = torch.flatten(pooled_features, start_dim=1)
            print(f"After flatten, flattened_features shape: {flattened_features.shape}")
        else:
            raise ValueError(f"Unexpected features dimension: {features.ndim}. Shape: {features.shape}")

        # åœ¨è¿™é‡Œæ£€æŸ¥ flattened_features çš„å½¢çŠ¶æ˜¯å�¦æ˜¯ (batch_size, 1280)
        if flattened_features.shape[1] != self.classifier.in_features: # self.classifier.in_features åº”è¯¥æ˜¯ 1280
            print(f"CRITICAL WARNING: flattened_features.shape[1] ({flattened_features.shape[1]}) "
                  f"does not match self.classifier.in_features ({self.classifier.in_features})!")
            # è¿™é‡Œå�¯ä»¥å¼•å�‘é”™è¯¯æˆ–å°�è¯•è°ƒæ•´ï¼Œä½†æœ€å¥½æ˜¯æ‰¾å‡ºæ ¹æœ¬å�Ÿå› 

        logits = self.classifier(flattened_features)
        print(f"Output logits shape: {logits.shape}")
        return logits
        
# å®�ä¾‹åŒ–æ¨¡å�‹
# ç¡®ä¿� num_classes æ˜¯æœ‰æ•ˆçš„
if 'num_classes' in globals() and isinstance(num_classes, int) and num_classes > 0:
    model = BirdSoundModelTrain(cfg_config=cfg,
                                num_classes_model=num_classes,
                                pretrained_imagenet=cfg.pretrained_on_imagenet)
    model.to(cfg.device) # å°†æ¨¡å�‹ç§»åŠ¨åˆ°æŒ‡å®šè®¾å¤‡
    print(f"æ¨¡å�‹ {cfg.model_name} å·²æˆ�åŠŸå®šä¹‰å¹¶ç§»åŠ¨åˆ°è®¾å¤‡: {cfg.device}")
    # (å�¯é€‰) æ‰“å�°æ¨¡å�‹ç»“æ�„ï¼Œæ£€æŸ¥æ˜¯å�¦ç¬¦å�ˆé¢„æœŸ
    # print(model)
else:
    print("é”™è¯¯: ç”±äº� num_classes æ— æ•ˆï¼Œæ¨¡å�‹æœªå®�ä¾‹åŒ–ã€‚è¯·è¿”å›�æ•°æ�®å‡†å¤‡æ­¥éª¤æ£€æŸ¥ã€‚")


# (å�¯é€‰) æµ‹è¯•æ¨¡å�‹æ˜¯å�¦èƒ½å¤„ç�†ä¸€ä¸ªä¼ªé€ çš„è¾“å…¥æ‰¹æ¬¡
if 'model' in globals() and 'train_loader' in globals() and len(train_loader) > 0 :
    try:
        print("\næµ‹è¯•æ¨¡å�‹å‰�å�‘ä¼ æ’­...")
        dummy_images, dummy_labels = next(iter(train_loader)) # ä»� DataLoader å�–ä¸€ä¸ªæ‰¹æ¬¡
        dummy_images = dummy_images.to(cfg.device)
        dummy_labels = dummy_labels.to(cfg.device)

        print(f"ä¼ªé€ è¾“å…¥å›¾åƒ�å½¢çŠ¶: {dummy_images.shape}") # åº”è¯¥æ˜¯ (batch_size, in_channels, H, W)
        print(f"ä¼ªé€ è¾“å…¥æ ‡ç­¾å½¢çŠ¶: {dummy_labels.shape}") # åº”è¯¥æ˜¯ (batch_size)

        with torch.no_grad(): # åœ¨æµ‹è¯•æ—¶ä¸�è®¡ç®—æ¢¯åº¦
            model.eval() # è®¾ç½®ä¸ºè¯„ä¼°æ¨¡å¼� (ä¸»è¦�å½±å“� Dropout, BatchNorm ç­‰)
            output_logits = model(dummy_images)
            model.train() # åˆ‡æ�¢å›�è®­ç»ƒæ¨¡å¼�

        print(f"æ¨¡å�‹è¾“å‡º (logits) çš„å½¢çŠ¶: {output_logits.shape}") # åº”è¯¥æ˜¯ (batch_size, num_classes)
        assert output_logits.shape == (cfg.train_batch_size, num_classes), "æ¨¡å�‹è¾“å‡ºå½¢çŠ¶ä¸�é¢„æœŸä¸�ç¬¦ï¼�"
        print("æ¨¡å�‹å‰�å�‘ä¼ æ’­æµ‹è¯•æˆ�åŠŸï¼�")
    except Exception as e:
        print(f"æ¨¡å�‹å‰�å�‘ä¼ æ’­æµ‹è¯•å¤±è´¥: {e}")
        print("è¯·æ£€æŸ¥ï¼š")
        print("1. DataLoader æ˜¯å�¦èƒ½æ­£ç¡®è¾“å‡ºæ•°æ�®ã€‚")
        print("2. æ¨¡å�‹å®šä¹‰çš„è¾“å…¥è¾“å‡ºç»´åº¦æ˜¯å�¦æ­£ç¡®ã€‚")
        print(f"3. num_classes ({num_classes if 'num_classes' in globals() else 'æœªå®šä¹‰'}) æ˜¯å�¦ä¸� DataLoader ä¸­çš„æ ‡ç­¾å¯¹åº”ã€‚")

print("\n--- æ¨¡å�‹å®šä¹‰é˜¶æ®µç»“æ�Ÿ ---")


# --- æ­¥éª¤ 4: è®­ç»ƒç»„ä»¶è®¾ç½® ---
print("\n--- å¼€å§‹è®¾ç½®è®­ç»ƒç»„ä»¶ ---")

# 1. å®šä¹‰æ�Ÿå¤±å‡½æ•°
# æ ¹æ�® CFG ä¸­çš„ loss_fn_name
if cfg.loss_fn_name.lower() == 'crossentropyloss':
    # CrossEntropyLoss é€‚ç”¨äº�å¤šåˆ†ç±»é—®é¢˜ï¼Œå½“æ¨¡å�‹è¾“å‡ºå�Ÿå§‹ logits ä¸”æ ‡ç­¾æ˜¯ç±»åˆ«ç´¢å¼•æ—¶ã€‚
    # å®ƒå†…éƒ¨å·²ç»�åŒ…å�«äº† Softmax æ“�ä½œã€‚
    criterion = nn.CrossEntropyLoss()
    print(f"æ�Ÿå¤±å‡½æ•°å·²è®¾ç½®: nn.CrossEntropyLoss")
elif cfg.loss_fn_name.lower() == 'bcewithlogitsloss':
    # BCEWithLogitsLoss é€‚ç”¨äº�å¤šæ ‡ç­¾åˆ†ç±»é—®é¢˜ï¼Œæˆ–è€…äºŒåˆ†ç±»é—®é¢˜ã€‚
    # æ¨¡å�‹è¾“å‡ºå�Ÿå§‹ logitsï¼Œæ ‡ç­¾æ˜¯ one-hot ç¼–ç �æˆ–è€…æ¯�ä¸ªç±»åˆ«å¯¹åº”ä¸€ä¸ª0æˆ–1çš„å€¼ã€‚
    criterion = nn.BCEWithLogitsLoss()
    print(f"æ�Ÿå¤±å‡½æ•°å·²è®¾ç½®: nn.BCEWithLogitsLoss")
else:
    raise ValueError(f"ä¸�æ”¯æŒ�çš„æ�Ÿå¤±å‡½æ•°: {cfg.loss_fn_name}. è¯·åœ¨CFGä¸­é€‰æ‹© 'CrossEntropyLoss' æˆ– 'BCEWithLogitsLoss'ï¼Œæˆ–åœ¨æ­¤å¤„æ·»åŠ æ›´å¤šé€‰é¡¹ã€‚")



# 2. å®šä¹‰ä¼˜åŒ–å™¨
# æ ¹æ�® CFG ä¸­çš„ optimizer_name å’Œç›¸å…³å�‚æ•°
# model.parameters() å‘Šè¯‰ä¼˜åŒ–å™¨éœ€è¦�æ›´æ–°å“ªäº›å�‚æ•°
if cfg.optimizer_name.lower() == 'adamw':
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    print(f"ä¼˜åŒ–å™¨å·²è®¾ç½®: AdamW, å­¦ä¹ ç�‡: {cfg.learning_rate}, æ�ƒé‡�è¡°å‡�: {cfg.weight_decay}")
elif cfg.optimizer_name.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay) # Adam ä¹Ÿå�¯ä»¥è®¾ç½® weight_decay
    print(f"ä¼˜åŒ–å™¨å·²è®¾ç½®: Adam, å­¦ä¹ ç�‡: {cfg.learning_rate}, æ�ƒé‡�è¡°å‡�: {cfg.weight_decay}")
elif cfg.optimizer_name.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.learning_rate,
                          momentum=0.9, # SGD é€šå¸¸éœ€è¦� momentum
                          weight_decay=cfg.weight_decay)
    print(f"ä¼˜åŒ–å™¨å·²è®¾ç½®: SGD, å­¦ä¹ ç�‡: {cfg.learning_rate}, Momentum: 0.9, æ�ƒé‡�è¡°å‡�: {cfg.weight_decay}")
else:
    raise ValueError(f"ä¸�æ”¯æŒ�çš„ä¼˜åŒ–å™¨: {cfg.optimizer_name}. è¯·åœ¨CFGä¸­é€‰æ‹© 'AdamW', 'Adam', 'SGD'ï¼Œæˆ–åœ¨æ­¤å¤„æ·»åŠ æ›´å¤šé€‰é¡¹ã€‚")



# 3. å®šä¹‰å­¦ä¹ ç�‡è°ƒåº¦å™¨ (å�¯é€‰)
# æ ¹æ�® CFG ä¸­çš„ scheduler_name å’Œ lr_scheduler_params
scheduler = None # åˆ�å§‹åŒ–ä¸º None
if cfg.scheduler_name: # å�ªæœ‰åœ¨ CFG ä¸­æŒ‡å®šäº† scheduler_name æ‰�åˆ›å»º
    if cfg.scheduler_name.lower() == 'cosineannealinglr':
        # ç¡®ä¿� cfg.lr_scheduler_params['T_max'] å’Œ cfg.lr_scheduler_params['eta_min'] å·²æ­£ç¡®è®¾ç½®
        # åœ¨æˆ‘ä»¬ä¹‹å‰�çš„CFGå®�ä¾‹åŒ–å��ï¼Œæˆ‘ä»¬å·²ç»�å¤„ç�†äº† T_max = cfg.num_epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=int(cfg.lr_scheduler_params['T_max']), # T_maxåº”ä¸ºæ•´æ•°
                                                         eta_min=cfg.lr_scheduler_params['eta_min'])
        print(f"å­¦ä¹ ç�‡è°ƒåº¦å™¨å·²è®¾ç½®: CosineAnnealingLR, T_max: {cfg.lr_scheduler_params['T_max']}, eta_min: {cfg.lr_scheduler_params['eta_min']}")
    elif cfg.scheduler_name.lower() == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=int(cfg.lr_scheduler_params['step_size']), # step_sizeåº”ä¸ºæ•´æ•°
                                              gamma=cfg.lr_scheduler_params['gamma'])
        print(f"å­¦ä¹ ç�‡è°ƒåº¦å™¨å·²è®¾ç½®: StepLR, step_size: {cfg.lr_scheduler_params['step_size']}, gamma: {cfg.lr_scheduler_params['gamma']}")
    elif cfg.scheduler_name.lower() == 'reducelronplateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min', # é€šå¸¸ç›‘æ�§éªŒè¯�é›†æ�Ÿå¤±ï¼Œæ‰€ä»¥æ˜¯ 'min'
                                                         factor=0.1,
                                                         patience=5, # 5ä¸ªepochéªŒè¯�é›†æ�Ÿå¤±æ²¡æœ‰æ”¹å–„åˆ™é™�ä½�å­¦ä¹ ç�‡
                                                         verbose=True)
        print(f"å­¦ä¹ ç�‡è°ƒåº¦å™¨å·²è®¾ç½®: ReduceLROnPlateau")
    # ä½ å�¯ä»¥åœ¨è¿™é‡Œæ·»åŠ æ›´å¤šè°ƒåº¦å™¨çš„é€‰é¡¹
    else:
        print(f"è­¦å‘Š: ä¸�æ”¯æŒ�çš„å­¦ä¹ ç�‡è°ƒåº¦å™¨ '{cfg.scheduler_name}'ã€‚å°†ä¸�ä½¿ç”¨è°ƒåº¦å™¨ã€‚")
else:
    print("æœªåœ¨CFGä¸­æŒ‡å®šå­¦ä¹ ç�‡è°ƒåº¦å™¨ (scheduler_name ä¸ºç©ºæˆ–None)ã€‚å°†ä¸�ä½¿ç”¨è°ƒåº¦å™¨ã€‚")

print("\n--- è®­ç»ƒç»„ä»¶è®¾ç½®é˜¶æ®µç»“æ�Ÿ ---")


# --- éªŒè¯�è®­ç»ƒç»„ä»¶æ˜¯å�¦å·²æ­£ç¡®è®¾ç½® ---
print("\n--- å¼€å§‹éªŒè¯�è®­ç»ƒç»„ä»¶ ---")

# æ ‡å¿—å�˜é‡�ï¼Œè¡¨ç¤ºæ‰€æœ‰æ£€æŸ¥æ˜¯å�¦é€šè¿‡
all_checks_passed = True

# 1. æ£€æŸ¥ cfg å¯¹è±¡
try:
    if 'cfg' in globals() and cfg is not None:
        print(f"âœ… cfg: å·²å®šä¹‰ã€‚æ¨¡å�‹å��ç§°: {cfg.model_name}, è®¾å¤‡: {cfg.device}, Epochs: {cfg.num_epochs}")
        # å�¯ä»¥æ·»åŠ æ›´å¤š cfg å±�æ€§çš„æ£€æŸ¥
        if not hasattr(cfg, 'output_model_dir') or not cfg.output_model_dir:
            print("    âš ï¸� è­¦å‘Š: cfg.output_model_dir æœªè®¾ç½®æˆ–ä¸ºç©ºï¼Œæ¨¡å�‹å�¯èƒ½æ— æ³•ä¿�å­˜ã€‚")
            # all_checks_passed = False # å�¯ä»¥é€‰æ‹©æ˜¯å�¦å› æ­¤å¤±è´¥
    else:
        print("â�Œ cfg: æœªå®šä¹‰æˆ–ä¸º Noneã€‚")
        all_checks_passed = False
except Exception as e:
    print(f"â�Œ cfg: æ£€æŸ¥æ—¶å�‘ç”Ÿé”™è¯¯ - {e}")
    all_checks_passed = False

# 2. æ£€æŸ¥ model å¯¹è±¡
try:
    if 'model' in globals() and isinstance(model, torch.nn.Module):
        # æ£€æŸ¥æ¨¡å�‹æ˜¯å�¦åœ¨æ­£ç¡®çš„è®¾å¤‡ä¸Š (è¿™æ˜¯ä¸€ä¸ªå�¯å�‘å¼�æ£€æŸ¥ï¼Œä¸�å®Œå…¨å�¯é� )
        # æ›´å�¯é� çš„æ–¹å¼�æ˜¯æ£€æŸ¥æ¨¡å�‹å�‚æ•°çš„è®¾å¤‡
        model_device = next(model.parameters()).device
        print(f"âœ… model: å·²å®šä¹‰ï¼Œç±»å�‹: {type(model).__name__}, æ‰€åœ¨è®¾å¤‡: {model_device}")
        if str(model_device) != str(cfg.device): # è½¬æ�¢ä¸ºå­—ç¬¦ä¸²æ¯”è¾ƒä»¥é˜²è®¾å¤‡å¯¹è±¡ç±»å�‹ä¸�å�Œ
             print(f"    âš ï¸� è­¦å‘Š: æ¨¡å�‹è®¾å¤‡ ({model_device}) ä¸� CFGè®¾å¤‡ ({cfg.device}) ä¸�ç¬¦ã€‚ç¡®ä¿�å·²æ‰§è¡Œ model.to(cfg.device)ã€‚")
             # all_checks_passed = False # å�¯ä»¥é€‰æ‹©æ˜¯å�¦å› æ­¤å¤±è´¥
    else:
        print("â�Œ model: æœªå®šä¹‰æˆ–ä¸�æ˜¯ torch.nn.Module çš„å®�ä¾‹ã€‚")
        all_checks_passed = False
except Exception as e:
    print(f"â�Œ model: æ£€æŸ¥æ—¶å�‘ç”Ÿé”™è¯¯ - {e}")
    all_checks_passed = False

# 3. æ£€æŸ¥ criterion å¯¹è±¡
try:
    if 'criterion' in globals() and isinstance(criterion, torch.nn.modules.loss._Loss):
        print(f"âœ… criterion: å·²å®šä¹‰ï¼Œç±»å�‹: {type(criterion).__name__}")
    else:
        print("â�Œ criterion: æœªå®šä¹‰æˆ–ä¸�æ˜¯æ�Ÿå¤±å‡½æ•°çš„æœ‰æ•ˆå®�ä¾‹ã€‚")
        all_checks_passed = False
except Exception as e:
    print(f"â�Œ criterion: æ£€æŸ¥æ—¶å�‘ç”Ÿé”™è¯¯ - {e}")
    all_checks_passed = False

# 4. æ£€æŸ¥ optimizer å¯¹è±¡
try:
    if 'optimizer' in globals() and isinstance(optimizer, torch.optim.Optimizer):
        print(f"âœ… optimizer: å·²å®šä¹‰ï¼Œç±»å�‹: {type(optimizer).__name__}")
        # æ£€æŸ¥ä¼˜åŒ–å™¨æ˜¯å�¦å…³è�”äº†æ¨¡å�‹å�‚æ•°
        if not optimizer.param_groups or not optimizer.param_groups[0]['params']:
            print("    âš ï¸� è­¦å‘Š: ä¼˜åŒ–å™¨ä¼¼ä¹�æ²¡æœ‰å…³è�”ä»»ä½•æ¨¡å�‹å�‚æ•°ã€‚")
            all_checks_passed = False
    else:
        print("â�Œ optimizer: æœªå®šä¹‰æˆ–ä¸�æ˜¯ä¼˜åŒ–å™¨çš„æœ‰æ•ˆå®�ä¾‹ã€‚")
        all_checks_passed = False
except Exception as e:
    print(f"â�Œ optimizer: æ£€æŸ¥æ—¶å�‘ç”Ÿé”™è¯¯ - {e}")
    all_checks_passed = False
    
# åœ¨éªŒè¯� scheduler çš„ä»£ç �å�—ä¸­

try:
    if 'scheduler' in globals(): # æ£€æŸ¥å�˜é‡�æ˜¯å�¦å­˜åœ¨
        if scheduler is None and (not hasattr(cfg, 'scheduler_name') or not cfg.scheduler_name):
            print(f"âœ… scheduler: ä¸º None (ç¬¦å�ˆé¢„æœŸï¼Œå› ä¸º CFG ä¸­æœªæŒ‡å®š scheduler_name)ã€‚")
        elif scheduler is not None:
            # ç›´æ�¥æ‰“å�° scheduler çš„ç±»å�‹ï¼Œå¹¶æ£€æŸ¥å®ƒæ˜¯å�¦å…·æœ‰ step æ–¹æ³•
            print(f"âœ… scheduler: å·²å®šä¹‰ï¼Œå®�é™…ç±»å�‹: {type(scheduler).__name__}")
            if not hasattr(scheduler, 'step'):
                print(f"    âš ï¸� è­¦å‘Š: scheduler å¯¹è±¡ ({type(scheduler).__name__}) ç¼ºå°‘ 'step' æ–¹æ³•ï¼Œå�¯èƒ½ä¸�æ˜¯ä¸€ä¸ªæœ‰æ•ˆçš„å­¦ä¹ ç�‡è°ƒåº¦å™¨ã€‚")
                all_checks_passed = False
            # ä½ ä»�ç„¶å�¯ä»¥ä¿�ç•™ isinstance æ£€æŸ¥ï¼Œä½†å¦‚æ�œå®ƒæŒ�ç»­æŠ¥é”™ï¼Œä¸Šé�¢çš„ hasattr æ£€æŸ¥æ›´å®�ç”¨
            # if not isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
            #     print(f"    â„¹ï¸� ä¿¡æ�¯: isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler) è¿”å›� Falseï¼Œä½†å�ªè¦�æœ‰ 'step' æ–¹æ³•é€šå¸¸å°±å�¯ç”¨ã€‚")

        # ... (else if scheduler is None ä½† cfg.scheduler_name å·²æŒ‡å®š çš„é€»è¾‘ä¿�æŒ�ä¸�å�˜) ...
    # ... (except å’Œå…¶ä»–é€»è¾‘ä¿�æŒ�ä¸�å�˜) ...
except Exception as e:
    print(f"â�Œ scheduler: æ£€æŸ¥æ—¶å�‘ç”Ÿé”™è¯¯ - {e}")
    all_checks_passed = False


# 6. æ£€æŸ¥ train_loader å’Œ valid_loader å¯¹è±¡
def check_dataloader(loader_name, loader_instance, batch_size_cfg):
    global all_checks_passed # å…�è®¸ä¿®æ”¹å¤–éƒ¨çš„ all_checks_passed
    try:
        if loader_name in globals() and isinstance(loader_instance, torch.utils.data.DataLoader):
            print(f"âœ… {loader_name}: å·²å®šä¹‰ï¼Œç±»å�‹: DataLoader, Batch Size: {loader_instance.batch_size}")
            if loader_instance.batch_size != batch_size_cfg:
                print(f"    âš ï¸� è­¦å‘Š: {loader_name} çš„ batch_size ({loader_instance.batch_size}) ä¸� CFG ({batch_size_cfg}) ä¸�ç¬¦ã€‚")
            if len(loader_instance) == 0:
                print(f"    âš ï¸� è­¦å‘Š: {loader_name} ä¸ºç©º (é•¿åº¦ä¸º0)ï¼Œæ— æ³•è¿›è¡Œè®­ç»ƒ/éªŒè¯�ã€‚è¯·æ£€æŸ¥ Dataset æ˜¯å�¦æ­£ç¡®åŠ è½½æ•°æ�®ã€‚")
                all_checks_passed = False # é€šå¸¸ DataLoader ä¸ºç©ºæ˜¯ä¸¥é‡�é—®é¢˜
        else:
            print(f"â�Œ {loader_name}: æœªå®šä¹‰æˆ–ä¸�æ˜¯ DataLoader çš„æœ‰æ•ˆå®�ä¾‹ã€‚")
            all_checks_passed = False
    except Exception as e:
        print(f"â�Œ {loader_name}: æ£€æŸ¥æ—¶å�‘ç”Ÿé”™è¯¯ - {e}")
        all_checks_passed = False

if 'train_loader' in globals(): check_dataloader('train_loader', train_loader, cfg.train_batch_size)
else: print("â�Œ train_loader: æœªå®šä¹‰ã€‚"); all_checks_passed = False

if 'valid_loader' in globals(): check_dataloader('valid_loader', valid_loader, cfg.valid_batch_size)
else: print("â�Œ valid_loader: æœªå®šä¹‰ã€‚"); all_checks_passed = False


# 7. æ£€æŸ¥ num_classes
try:
    if 'num_classes' in globals() and isinstance(num_classes, int) and num_classes > 0:
        print(f"âœ… num_classes: å·²å®šä¹‰ï¼Œå€¼ä¸º: {num_classes}")
    else:
        print(f"â�Œ num_classes: æœªå®šä¹‰ï¼Œä¸�æ˜¯æ•´æ•°ï¼Œæˆ–å€¼æ— æ•ˆ (å½“å‰�å€¼: {num_classes if 'num_classes' in globals() else 'æœªå®šä¹‰'})ã€‚")
        all_checks_passed = False
except Exception as e:
    print(f"â�Œ num_classes: æ£€æŸ¥æ—¶å�‘ç”Ÿé”™è¯¯ - {e}")
    all_checks_passed = False

# 8. æ£€æŸ¥ int_to_label (å�¯é€‰)
try:
    if 'int_to_label' in globals() and isinstance(int_to_label, dict):
        print(f"âœ… int_to_label: å·²å®šä¹‰ (å�¯é€‰)ï¼Œç±»å�‹: dict, åŒ…å�« {len(int_to_label)} ä¸ªæ�¡ç›®ã€‚")
        if 'num_classes' in globals() and isinstance(num_classes, int) and num_classes > 0 and len(int_to_label) != num_classes:
             print(f"    âš ï¸� è­¦å‘Š: int_to_label çš„æ�¡ç›®æ•° ({len(int_to_label)}) ä¸� num_classes ({num_classes}) ä¸�ç¬¦ã€‚")
    elif 'int_to_label' not in globals():
        print(f"â„¹ï¸� int_to_label: æœªå®šä¹‰ (æ­¤é¡¹ä¸ºå�¯é€‰ï¼Œç”¨äº�æ—¥å¿—å�¯è¯»æ€§)ã€‚")
    else:
        print(f"âš ï¸� int_to_label: å·²å®šä¹‰ä½†ä¸�æ˜¯å­—å…¸ç±»å�‹ (ç±»å�‹: {type(int_to_label).__name__})ã€‚")

except Exception as e:
    print(f"â�Œ int_to_label: æ£€æŸ¥æ—¶å�‘ç”Ÿé”™è¯¯ - {e}")
    # all_checks_passed = False # å› ä¸ºæ˜¯å�¯é€‰çš„ï¼Œæ‰€ä»¥ä¸�å› æ­¤å¤±è´¥

print("\n--- éªŒè¯�ç»“æ�Ÿ ---")
if all_checks_passed:
    print("ğŸ‘� æ‰€æœ‰å…³é”®ç»„ä»¶çœ‹èµ·æ�¥éƒ½å·²æ­£ç¡®è®¾ç½®ï¼�å�¯ä»¥å‡†å¤‡å¼€å§‹è®­ç»ƒå¾ªç�¯äº†ã€‚")
else:
    print("ğŸ”¥ æ³¨æ„�ï¼šä¸€ä¸ªæˆ–å¤šä¸ªå…³é”®ç»„ä»¶æœªæ­£ç¡®è®¾ç½®æˆ–å­˜åœ¨è­¦å‘Šã€‚è¯·æ£€æŸ¥ä¸Šé�¢çš„é”™è¯¯/è­¦å‘Šä¿¡æ�¯ï¼Œå¹¶åœ¨å¼€å§‹è®­ç»ƒå‰�ä¿®å¤�å®ƒä»¬ã€‚")


import time # ç”¨äº�è®°å½•è®­ç»ƒæ—¶é—´
import torch # ç¡®ä¿�torchå·²å¯¼å…¥
from tqdm.auto import tqdm # è¿›åº¦æ�¡
import os # ç”¨äº�æ–‡ä»¶è·¯å¾„æ“�ä½œ
import copy # ç”¨äº�æ·±æ‹·è´�æ¨¡å�‹æ�ƒé‡� (ä¿�å­˜æœ€ä½³æ¨¡å�‹æ—¶)


# --- æ­¥éª¤ 5: è®­ç»ƒå¾ªç�¯ ---
print("\n--- å¼€å§‹è®­ç»ƒå¾ªç�¯ ---")

# ç”¨äº�è®°å½•æ¯�ä¸ªepochçš„å¹³å�‡æ�Ÿå¤±å’Œå‡†ç¡®ç�‡ (æˆ–å…¶ä»–æŒ‡æ ‡)
history = {
    'train_loss': [],
    'train_acc': [],
    'valid_loss': [],
    'valid_acc': [],
    'lr': []
}


best_valid_acc = 0.0 # ç”¨äº�è¿½è¸ªæœ€ä½³éªŒè¯�å‡†ç¡®ç�‡
best_epoch = -1
best_model_weights = None # ç”¨äº�ä¿�å­˜æœ€ä½³æ¨¡å�‹çš„æ�ƒé‡�

# ç¡®ä¿�æ¨¡å�‹åœ¨æ­£ç¡®çš„è®¾å¤‡ä¸Š
model.to(cfg.device)



for epoch in range(cfg.num_epochs):
    epoch_start_time = time.time()
    print(f"\nEpoch {epoch+1}/{cfg.num_epochs}")
    print("-" * 30)

    # --- è®­ç»ƒé˜¶æ®µ ---
    model.train()  # è®¾ç½®æ¨¡å�‹ä¸ºè®­ç»ƒæ¨¡å¼� (å�¯ç”¨ Dropout, BatchNorm æ›´æ–°ç­‰)
    running_train_loss = 0.0
    correct_train_preds = 0
    total_train_samples = 0

    # ä½¿ç”¨ tqdm åˆ›å»ºè®­ç»ƒè¿›åº¦æ�¡
    train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)

    for batch_idx, (images, labels) in enumerate(train_pbar):
        images = images.to(cfg.device)
        labels = labels.to(cfg.device)

        # 1. æ¸…é›¶æ¢¯åº¦
        optimizer.zero_grad()

        # 2. å‰�å�‘ä¼ æ’­
        outputs = model(images)  # æ¨¡å�‹è¾“å‡º logits

        # 3. è®¡ç®—æ�Ÿå¤±
        loss = criterion(outputs, labels)

        # 4. å��å�‘ä¼ æ’­
        loss.backward()

        # 5. æ›´æ–°æ�ƒé‡�
        optimizer.step()

        # ç»Ÿè®¡æ�Ÿå¤±å’Œå‡†ç¡®ç�‡
        running_train_loss += loss.item() * images.size(0) # loss.item()æ˜¯å½“å‰�batchçš„å¹³å�‡loss
        _, predicted_classes = torch.max(outputs, 1) # è�·å�–é¢„æµ‹ç±»åˆ« (æ¦‚ç�‡æœ€é«˜çš„é‚£ä¸ª)
        correct_train_preds += (predicted_classes == labels).sum().item()
        total_train_samples += labels.size(0)

        # æ›´æ–°è¿›åº¦æ�¡æ��è¿° (å�¯é€‰)
        if batch_idx % 20 == 0: # æ¯�20ä¸ªbatchæ›´æ–°ä¸€æ¬¡
            train_pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{(predicted_classes == labels).sum().item() / labels.size(0):.4f}"
            })

    epoch_train_loss = running_train_loss / total_train_samples
    epoch_train_acc = correct_train_preds / total_train_samples
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)

    # --- éªŒè¯�é˜¶æ®µ ---
    model.eval()   # è®¾ç½®æ¨¡å�‹ä¸ºè¯„ä¼°æ¨¡å¼� (ç¦�ç”¨ Dropout, BatchNorm ä½¿ç”¨è¿�è¡Œæ—¶çš„ç»Ÿè®¡æ•°æ�®)
    running_valid_loss = 0.0
    correct_valid_preds = 0
    total_valid_samples = 0

    valid_pbar = tqdm(valid_loader, desc=f"Validating Epoch {epoch+1}", leave=False)

    with torch.no_grad(): # åœ¨éªŒè¯�é˜¶æ®µä¸�è®¡ç®—æ¢¯åº¦ï¼ŒèŠ‚çœ�å†…å­˜å’Œè®¡ç®—
        for images, labels in valid_pbar:
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_valid_loss += loss.item() * images.size(0)
            _, predicted_classes = torch.max(outputs, 1)
            correct_valid_preds += (predicted_classes == labels).sum().item()
            total_valid_samples += labels.size(0)

            if batch_idx % 20 == 0:
                 valid_pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{(predicted_classes == labels).sum().item() / labels.size(0):.4f}"
                })


    epoch_valid_loss = running_valid_loss / total_valid_samples
    epoch_valid_acc = correct_valid_preds / total_valid_samples
    history['valid_loss'].append(epoch_valid_loss)
    history['valid_acc'].append(epoch_valid_acc)

    # è®°å½•å½“å‰�å­¦ä¹ ç�‡
    current_lr = optimizer.param_groups[0]['lr']
    history['lr'].append(current_lr)

    # --- å­¦ä¹ ç�‡è°ƒåº¦å™¨æ­¥éª¤ ---
    if scheduler:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_valid_loss) # ReduceLROnPlateau éœ€è¦�ç›‘æ�§ä¸€ä¸ªæŒ‡æ ‡
        else:
            scheduler.step() # å…¶ä»–å¤§å¤šæ•°è°ƒåº¦å™¨åœ¨ epoch ç»“æ�Ÿæ—¶ step

    # --- æ‰“å�°å½“å‰� Epoch çš„ç»“æ�œ ---
    epoch_duration = time.time() - epoch_start_time
    if (epoch + 1) % cfg.print_freq_epochs == 0: # æ ¹æ�® CFG ä¸­çš„é¢‘ç�‡æ‰“å�°
        print(f"Epoch {epoch+1}/{cfg.num_epochs} - "
              f"Duration: {epoch_duration:.2f}s - "
              f"LR: {current_lr:.1e}")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"  Valid Loss: {epoch_valid_loss:.4f}, Valid Acc: {epoch_valid_acc:.4f}")

    # --- æ¨¡å�‹ä¿�å­˜ ---
    # æ£€æŸ¥æ˜¯å�¦æ˜¯å½“å‰�æœ€ä½³æ¨¡å�‹ (åŸºäº�éªŒè¯�å‡†ç¡®ç�‡)
    if epoch_valid_acc > best_valid_acc:
        best_valid_acc = epoch_valid_acc
        best_epoch = epoch + 1
        # ä¿�å­˜æœ€ä½³æ¨¡å�‹çš„æ�ƒé‡� (ä½¿ç”¨æ·±æ‹·è´�ä»¥é˜²å��ç»­modelè¢«ä¿®æ”¹)
        best_model_weights = copy.deepcopy(model.state_dict())
        print(f"ğŸ�‰ New best model found at Epoch {best_epoch} with Valid Acc: {best_valid_acc:.4f}")

        # å¦‚æ�œå�ªä¿�å­˜æœ€ä½³æ¨¡å�‹ï¼Œåˆ™åœ¨è¿™é‡Œä¿�å­˜
        if cfg.save_best_model_only:
            save_path = os.path.join(cfg.output_model_dir,
                                     f"{cfg.model_name}_fold{cfg.current_fold_to_train}_best_acc.pth")
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_weights,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_valid_acc': best_valid_acc,
                'config': vars(cfg) # ä¿�å­˜é…�ç½®ä¿¡æ�¯ (å°† CFG å¯¹è±¡è½¬ä¸ºå­—å…¸)
            }, save_path)
            print(f"   Best model (so far) saved to: {save_path}")

    # (å�¯é€‰) å¦‚æ�œä¸�æ˜¯å�ªä¿�å­˜æœ€ä½³æ¨¡å�‹ï¼Œä¹Ÿå�¯ä»¥æ¯�éš”ä¸€å®šepochä¿�å­˜ä¸€æ¬¡ï¼Œæˆ–è€…æ¯�ä¸ªepochéƒ½ä¿�å­˜
    # if not cfg.save_best_model_only:
    #     if (epoch + 1) % 5 == 0: # ä¾‹å¦‚æ¯�5ä¸ªepochä¿�å­˜ä¸€æ¬¡
    #         save_path = os.path.join(cfg.output_model_dir,
    #                                  f"{cfg.model_name}_fold{cfg.current_fold_to_train}_epoch{epoch+1}.pth")
    #         torch.save({
    #             'epoch': epoch + 1,
    #             'model_state_dict': model.state_dict(),
    #             # ... (å…¶ä»–ä½ æƒ³ä¿�å­˜çš„ä¿¡æ�¯)
    #         }, save_path)
    #         print(f"   Model at epoch {epoch+1} saved to: {save_path}")

# --- è®­ç»ƒå¾ªç�¯ç»“æ�Ÿ ---
print("\n--- è®­ç»ƒå¾ªç�¯å·²ç»“æ�Ÿ ---")
print(f"æœ€ä½³éªŒè¯�å‡†ç¡®ç�‡: {best_valid_acc:.4f} åœ¨ç¬¬ {best_epoch} è½®è¾¾åˆ°ã€‚")
if cfg.save_best_model_only and best_model_weights is not None:
    print(f"æœ€ä½³æ¨¡å�‹å·²ä¿�å­˜åœ¨ {cfg.output_model_dir} ç›®å½•ä¸‹ã€‚")
elif best_model_weights is None:
    print("è­¦å‘Š: æœªæ‰¾åˆ°æˆ–ä¿�å­˜ä»»ä½•æœ€ä½³æ¨¡å�‹ï¼Œå�¯èƒ½æ˜¯å› ä¸ºéªŒè¯�å‡†ç¡®ç�‡æ²¡æœ‰æ��å�‡æˆ–save_best_model_onlyä¸ºFalseä¸”æœªå®�ç�°å…¶ä»–ä¿�å­˜ç­–ç•¥ã€‚")



plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['valid_loss'], label='Valid Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['valid_acc'], label='Valid Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history['lr'], label='Learning Rate')
plt.title('Learning Rate vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.legend()

plt.tight_layout()
plt.show()


import os, torch
model_dir = "/kaggle/working/models"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "efficientnet_b0_fold0_best_acc.pth")
torch.save(model.state_dict(), model_path)      # å�ªä¿�å­˜æ�ƒé‡�
# æˆ–è€…
# torch.save(model, model_path)                # ä¿�å­˜æ•´æ¨¡å�‹ï¼Œä½“ç§¯æ›´å¤§
print("æ¨¡å�‹å·²å†™å…¥:", model_path)




class CFG:

    train_audio_path = '/kaggle/input/birdclef-2025/train_audio/' # ä½ çš„è®­ç»ƒéŸ³é¢‘æ–‡ä»¶å­˜æ”¾è·¯å¾„
    train_metadata_csv = '/kaggle/input/birdclef-2025/train.csv' # åŒ…å�«éŸ³é¢‘æ–‡ä»¶å��å’Œå¯¹åº”é¸Ÿç±»æ ‡ç­¾çš„CSVæ–‡ä»¶è·¯å¾„

    # 2. ç‰©ç§�åˆ†ç±»æ–‡ä»¶ (ç”¨äº�è�·å�–ç±»åˆ«æ€»æ•°å’Œæ ‡ç­¾æ˜ å°„)
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'

    # 3. ä½ è®­ç»ƒå¥½çš„æ¨¡å�‹çš„ä¿�å­˜è·¯å¾„ (æ£€æŸ¥ç‚¹)
    output_model_dir = '/kaggle/working/models/' # æˆ–è€…å…¶ä»–ä»»ä½•ä½ æœ‰å†™å…¥æ�ƒé™�çš„ç›®å½•
                                                 # æ¯�ä¸€æŠ˜æˆ–æ¯�ä¸€ä¸ªepochçš„æ¨¡å�‹éƒ½ä¼šä¿�å­˜åœ¨è¿™é‡Œ
  # --- éŸ³é¢‘å’Œæ¢…å°”é¢‘è°±å›¾å�‚æ•° (é€šå¸¸å’Œæ�¨ç�†æ—¶ä¿�æŒ�ä¸€è‡´) ---
    FS = 32000  
    WINDOW_SIZE = 5  
    
    N_FFT = 1024
    HOP_LENGTH = 64
    N_MELS = 136
    FMIN = 20
    FMAX = 16000
    TARGET_SHAPE = (256, 256)
    # --- æ¨¡å�‹æ�¶æ�„ ---
    model_name = 'efficientnet_b0'
    in_channels = 1 # è¾“å…¥é€šé�“æ•° (1ä»£è¡¨å�•é€šé�“æ¢…å°”é¢‘è°±å›¾)
    # æ˜¯å�¦ä½¿ç”¨ ImageNet é¢„è®­ç»ƒæ�ƒé‡�ä½œä¸ºä½ éª¨å¹²ç½‘ç»œçš„èµ·ç‚¹
    # è¿™å¯¹äº�è¿�ç§»å­¦ä¹ é€šå¸¸æ˜¯æœ‰ç›Šçš„
    pretrained_on_imagenet = True # è®¾ç½®ä¸º True æ�¥åŠ è½½ ImageNet æ�ƒé‡�åˆ°éª¨å¹²ç½‘ç»œ
    
    # --- è®­ç»ƒè¶…å�‚æ•° ---
    # 1. é€šç”¨è®¾ç½®
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # å¦‚æ�œæœ‰GPUåˆ™ä½¿ç”¨GPU
    seed = 42               # ç”¨äº�ä¿�è¯�å®�éªŒå�¯å¤�ç�°çš„éš�æœºç§�å­�
    num_epochs = 1         # æ€»è®­ç»ƒè½®æ¬¡
    train_batch_size = 32   # è®­ç»ƒæ‰¹æ¬¡å¤§å°�
    valid_batch_size = 64   # éªŒè¯�æ‰¹æ¬¡å¤§å°� (é€šå¸¸å�¯ä»¥è®¾å¤§ä¸€äº›ï¼Œå› ä¸ºéªŒè¯�æ—¶æ²¡æœ‰å��å�‘ä¼ æ’­)

    # 2. ä¼˜åŒ–å™¨
    optimizer_name = 'AdamW' # ä¾‹å¦‚: 'Adam', 'AdamW', 'SGD'
    learning_rate = 1e-3     # å­¦ä¹ ç�‡
    weight_decay = 1e-5      # æ�ƒé‡�è¡°å‡� (ç”¨äº�åƒ� AdamW è¿™æ ·çš„ä¼˜åŒ–å™¨)

    # 3. å­¦ä¹ ç�‡è°ƒåº¦å™¨ (å�¯é€‰ï¼Œä½†é€šå¸¸æœ‰å¸®åŠ©)
    scheduler_name = 'CosineAnnealingLR' # ä¾‹å¦‚: 'StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR'
    lr_scheduler_params = {  # å­¦ä¹ ç�‡è°ƒåº¦å™¨çš„å…·ä½“å�‚æ•°
        'T_max': num_epochs, # å¯¹äº� CosineAnnealingLR
        'eta_min': 1e-6      # å¯¹äº� CosineAnnealingLR
    }
    # æˆ–è€…å¯¹äº� StepLR: {'step_size': 10, 'gamma': 0.1}
    if num_epochs != 50: # å¦‚æ�œ num_epochs è¢«ä¿®æ”¹ï¼Œéœ€è¦�ç¡®ä¿� T_max ä¹Ÿæ›´æ–°
        lr_scheduler_params['T_max'] = num_epochs
    # 4. æ�Ÿå¤±å‡½æ•°
    loss_fn_name = 'CrossEntropyLoss' # å¦‚æ�œä½ çš„æ ‡ç­¾æ˜¯æ¯�ä¸ªæ ·æœ¬ä¸€ä¸ªé¸Ÿç±»æ•´æ•°IDï¼Œå°±ç”¨è¿™ä¸ª
                                      # å¦‚æ�œä¸€ä¸ªå£°éŸ³é‡Œå�¯èƒ½æœ‰å¤šç§�é¸Ÿ (å¤šæ ‡ç­¾)ï¼Œå�¯èƒ½ç”¨ 'BCEWithLogitsLoss'
    # --- æ•°æ�®å¤„ç�†ä¸�éªŒè¯� ---
    num_workers = 2         # DataLoader ä½¿ç”¨çš„å·¥ä½œè¿›ç¨‹æ•°
    # KæŠ˜äº¤å�‰éªŒè¯� (åœ¨ç«�èµ›ä¸­å¾ˆå¸¸è§�)
    n_folds = 5             # æ€»å…±åˆ†å‡ æŠ˜
    current_fold_to_train = 0 # å½“å‰�è®­ç»ƒçš„æ˜¯ç¬¬å‡ æŠ˜ (ä»�0åˆ° n_folds-1)

    # --- æ•°æ�®å¢�å¼º (å�¯é€‰, ç”¨äº�è®­ç»ƒæ•°æ�®) ---
    # ä½ å�¯ä»¥åœ¨è¿™é‡Œå®šä¹‰éŸ³é¢‘æ•°æ�®å¢�å¼ºçš„å�‚æ•°ï¼Œä¾‹å¦‚:
    # use_noise_injection = True  # æ˜¯å�¦ä½¿ç”¨å™ªå£°æ³¨å…¥
    # noise_level = 0.005         # å™ªå£°æ°´å¹³
    # use_random_shift = True     # æ˜¯å�¦ä½¿ç”¨éš�æœºæ—¶é—´å¹³ç§»
    # ç­‰ç­‰...

    # --- æ—¥å¿—ä¸�æ¨¡å�‹ä¿�å­˜ ---
    print_freq_epochs = 1   # æ¯�éš”å¤šå°‘è½®æ‰“å�°ä¸€æ¬¡è®­ç»ƒæ—¥å¿—
    save_best_model_only = True # å�ªä¿�å­˜éªŒè¯�é›†ä¸Šè¡¨ç�°æœ€å¥½çš„æ¨¡å�‹

    # --- ç”¨äº�è°ƒè¯• (å�ªå¤„ç�†ä¸€å°�éƒ¨åˆ†æ•°æ�®) ---
    debug_mode = False
    debug_subset_size = 100 # å¦‚æ�œ debug_mode ä¸º True, ä½¿ç”¨çš„æ ·æœ¬æ•°é‡�

cfg = CFG()

cfg.lr_scheduler_params = {
    'T_max': cfg.num_epochs,
    'eta_min': 1e-6
}

print(f"--- CFG å®�ä¾‹åŒ–å�� ---")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"cfg.device è®¾ç½®ä¸º: {cfg.device}")
# ä¸‹é�¢è¿™ä¸¤è¡Œå…ˆæ³¨é‡Šæ�‰ï¼Œå› ä¸º model è¿˜æ²¡å®šä¹‰
# print(f"æ¨¡å�‹å·²ç§»åŠ¨åˆ°è®¾å¤‡: {cfg.device}")
# print(f"è®­ç»ƒå°†åœ¨è®¾å¤‡: {cfg.device} ä¸Šè¿›è¡Œ")
print(f"æ¢…å°”é¢‘è°±å›¾å�‚æ•°: N_FFT={cfg.N_FFT}, HOP_LENGTH={cfg.HOP_LENGTH}, N_MELS={cfg.N_MELS}")
print(f"ç›®æ ‡å›¾åƒ�å½¢çŠ¶: {cfg.TARGET_SHAPE}")

