import os
import gc
import time
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # K-Fold yerine bunu kullanacağız
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as AT
import torchaudio.functional as AF
import timm
import torch.nn.functional as F
from tqdm.auto import tqdm

# --- KONFİGÜRASYON (EfficientNetV2_M Tekli Split için Ayarlandı) ---
TRAIN_AUDIO_DIR = '/kaggle/input/birdclef-2025/train_audio'
MODEL_OUTPUT_DIR = '/kaggle/working/trained_effnetv2_m_single_split_models/'
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

CONFIG_EFFNET_M_SINGLE = {
    "seed": 42,
    "sample_rate": 32000,
    "duration_secs": 5,
    "n_mels": 128,
    "f_min": 50,
    "f_max": 16000,
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 512,
    # --- EfficientNetV2_M MODEL İÇİN AYARLAR ---
    "base_model_name": 'tf_efficientnetv2_m', # <<< MODEL ADI GÜNCELLENDİ
    "effnet_in_chans": 3,
    # --- --- ---
    "num_epochs": 30, # Senin isteğin üzerine
    "batch_size": 16,  # <<< _m için batch boyutunu DÜŞÜK BAŞLA (VRAM'e göre ayarla: 8, 16)
    "learning_rate": 5e-4, # _m için biraz daha düşük LR (örn: 1e-3, 5e-4, 1e-4)
    "optimizer_eps": 1e-7,
    "weight_decay": 1e-5,
    "patience": 7,
    "valid_split_ratio": 0.15, # <<< %15 Validasyon ayrımı
    "threshold": 0.5,
    # Veri Artırma (Aynı kalabilir veya ayarlanabilir)
    "augment_prob": 0.75,
    "gain_min_db": -8.0,  
    "gain_max_db": 8.0,
    "max_time_shift_ratio": 0.15,
    "spec_augment_prob": 0.6,
    "freq_mask_param": int(128 * 0.15),
    "time_mask_param": int(313 * 0.15),
    "num_freq_masks": 2,
    "num_time_masks": 2,
    "timm_model_drop_rate": 0.3, # tf_efficientnetv2_m için varsayılan dropout (timm'den kontrol et)
}

CONFIG_EFFNET_M_SINGLE["in_chans_spectrogram"] = 1
CONFIG_EFFNET_M_SINGLE["num_frames_in_segment"] = (CONFIG_EFFNET_M_SINGLE["duration_secs"] * CONFIG_EFFNET_M_SINGLE["sample_rate"]) // CONFIG_EFFNET_M_SINGLE["hop_length"] + 1
CONFIG_EFFNET_M_SINGLE["max_time_shift_samples"] = int(CONFIG_EFFNET_M_SINGLE["duration_secs"] * CONFIG_EFFNET_M_SINGLE["sample_rate"] * CONFIG_EFFNET_M_SINGLE["max_time_shift_ratio"])

def set_seed(seed=42):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

set_seed(CONFIG_EFFNET_M_SINGLE["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}. UYARI: Eğer CPU ise eğitim ÇOK ÇOK ÇOK UZUN sürecektir!")

def normalize_std(spec, eps=1e-6):
    mean = torch.mean(spec, dim=(-1, -2), keepdim=True)
    std = torch.std(spec, dim=(-1, -2), keepdim=True)
    return torch.where(std < eps, spec - mean, (spec - mean) / (std + eps))

if not os.path.exists(TRAIN_AUDIO_DIR):
    print(f"HATA: Eğitim verisi dizini bulunamadı: {TRAIN_AUDIO_DIR}"); exit()
all_labels_list = sorted([d for d in os.listdir(TRAIN_AUDIO_DIR) if os.path.isdir(os.path.join(TRAIN_AUDIO_DIR, d))])
CONFIG_EFFNET_M_SINGLE["num_classes"] = len(all_labels_list)
label_to_int_map = {label: i for i, label in enumerate(all_labels_list)}
print(f"Bulunan sınıf sayısı: {CONFIG_EFFNET_M_SINGLE['num_classes']}")

filepaths_all = []
labels_for_stratify_all = []
for bird_label in all_labels_list:
    bird_dir = os.path.join(TRAIN_AUDIO_DIR, bird_label)
    for filename in os.listdir(bird_dir):
        if filename.lower().endswith(".ogg"):
            filepaths_all.append(os.path.join(bird_dir, filename))
            labels_for_stratify_all.append(label_to_int_map[bird_label])

multilabel_binarizer_global = MultiLabelBinarizer(classes=list(range(CONFIG_EFFNET_M_SINGLE["num_classes"])))
_dummy_labels_for_mlb_fit = [[l] for l in list(range(CONFIG_EFFNET_M_SINGLE["num_classes"]))]
multilabel_binarizer_global.fit(_dummy_labels_for_mlb_fit)

# <<< DEĞİŞİKLİK: K-Fold yerine train_test_split kullanılıyor >>>
train_filepaths, valid_filepaths, _, _ = train_test_split(
    filepaths_all, labels_for_stratify_all,
    test_size=CONFIG_EFFNET_M_SINGLE["valid_split_ratio"],
    random_state=CONFIG_EFFNET_M_SINGLE["seed"],
    stratify=labels_for_stratify_all
)
print(f"Eğitim seti boyutu: {len(train_filepaths)}, Validasyon seti boyutu: {len(valid_filepaths)}")

# --- BirdSoundDataset (Değişiklik yok) ---
class BirdSoundDataset(Dataset):
    def __init__(self, filepaths, config, label_to_int_map, multilabel_binarizer_instance, is_train=True):
        self.filepaths = filepaths; self.config = config
        self.mel_transform = AT.MelSpectrogram(
            sample_rate=config["sample_rate"], n_fft=config["n_fft"], win_length=config["win_length"],
            hop_length=config["hop_length"], center=True, f_min=config["f_min"], f_max=config["f_max"],
            pad_mode="reflect", power=2.0, norm='slaney', n_mels=config["n_mels"], mel_scale="htk")
        self.sr = config["sample_rate"]; self.duration_samples = config["sample_rate"] * config["duration_secs"]
        self.num_classes = config["num_classes"]; self.label_to_int_map = label_to_int_map
        self.mlb = multilabel_binarizer_instance; self.is_train = is_train
        if self.is_train:
            self.frequency_masking = AT.FrequencyMasking(freq_mask_param=config["freq_mask_param"])
            self.time_masking = AT.TimeMasking(time_mask_param=config["time_mask_param"])
    def __len__(self): return len(self.filepaths)
    def apply_wav_augmentations(self, waveform):
        if random.random() < self.config["augment_prob"]:
            if random.random() < 0.5:
                gain_db = random.uniform(self.config["gain_min_db"], self.config["gain_max_db"])
                waveform = AF.gain(waveform, gain_db)
            if random.random() < 0.5:
                shift_samples = random.randint(-self.config["max_time_shift_samples"], self.config["max_time_shift_samples"])
                if shift_samples != 0: waveform = torch.roll(waveform, shifts=shift_samples, dims=1)
        return waveform
    def apply_spec_augmentations(self, melspec):
        if random.random() < self.config["spec_augment_prob"]:
            for _ in range(self.config["num_freq_masks"]): melspec = self.frequency_masking(melspec)
            for _ in range(self.config["num_time_masks"]): melspec = self.time_masking(melspec)
        return melspec
    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        try:
            waveform, sr_loaded = torchaudio.load(filepath, backend="soundfile")
        except Exception as e:
            print(f"HATA ({filepath} yüklenirken): {e}")
            dummy_mel = torch.zeros((self.config["in_chans_spectrogram"], self.config["n_mels"], self.config["num_frames_in_segment"]))
            dummy_labels = torch.zeros(self.num_classes, dtype=torch.float); return dummy_mel, dummy_labels
        if sr_loaded != self.sr: waveform = AT.Resample(orig_freq=sr_loaded, new_freq=self.sr)(waveform)
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        if self.is_train: waveform = self.apply_wav_augmentations(waveform)
        current_samples = waveform.shape[1]
        if current_samples > self.duration_samples:
            start = random.randint(0, current_samples - self.duration_samples) if self.is_train else (current_samples - self.duration_samples) // 2
            waveform = waveform[:, start : start + self.duration_samples]
        elif current_samples < self.duration_samples:
            waveform = F.pad(waveform, (0, self.duration_samples - current_samples))
        melspec = self.mel_transform(waveform)
        if self.is_train: melspec = self.apply_spec_augmentations(melspec)
        melspec = torch.log(melspec + 1e-6); melspec = normalize_std(melspec)    
        primary_label_str = os.path.basename(os.path.dirname(filepath))
        primary_label_int = self.label_to_int_map[primary_label_str]
        labels_encoded = self.mlb.transform([[primary_label_int]])[0]
        labels_encoded = torch.tensor(labels_encoded, dtype=torch.float)
        return melspec, labels_encoded

# DataLoader'lar (K-Fold'suz, doğrudan train_filepaths ve valid_filepaths ile)
train_dataset = BirdSoundDataset(train_filepaths, CONFIG_EFFNET_M_SINGLE, label_to_int_map, multilabel_binarizer_global, is_train=True)
valid_dataset = BirdSoundDataset(valid_filepaths, CONFIG_EFFNET_M_SINGLE, label_to_int_map, multilabel_binarizer_global, is_train=False)
num_workers = max(1, os.cpu_count() // 2 if os.cpu_count() is not None else 1)
train_loader = DataLoader(train_dataset, batch_size=CONFIG_EFFNET_M_SINGLE["batch_size"], shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=CONFIG_EFFNET_M_SINGLE["batch_size"], shuffle=False, num_workers=num_workers, pin_memory=True)

# --- MODEL TANIMI: TimmEffNetModel (Değişiklik yok) ---
class TimmEffNetModel(nn.Module):
    def __init__(self, base_model_name: str, pretrained=True, num_classes=206,
                 in_chans_spectrogram=1, effnet_in_chans=3, model_drop_rate=0.2):
        super().__init__()
        self.in_chans_spectrogram = in_chans_spectrogram; self.effnet_in_chans = effnet_in_chans
        self.encoder = timm.create_model(
            base_model_name, pretrained=pretrained, num_classes=num_classes,
            in_chans=self.effnet_in_chans, drop_rate=model_drop_rate)
        print(f"EfficientNetV2 Modeli {base_model_name} yüklendi. Giriş kanalı (encoder): {self.encoder.conv_stem.in_channels if hasattr(self.encoder, 'conv_stem') else 'N/A'}, Çıkış sınıfı: {num_classes}")
    def forward(self, x):
        if self.effnet_in_chans == 3 and x.size(1) == 1: x = x.repeat(1, 3, 1, 1)
        elif self.effnet_in_chans == 1 and x.size(1) == 3: x = torch.mean(x, dim=1, keepdim=True)
        elif hasattr(self.encoder, 'conv_stem') and self.encoder.conv_stem.in_channels != x.size(1):
             raise ValueError(f"Model {self.encoder.conv_stem.in_channels} giriş kanalı bekliyor, ancak {x.size(1)} alındı!")
        logits = self.encoder(x); return {"logit": logits}

# --- MODEL, OPTİMİZATÖR, ZAMANLAYICI, KAYIP FONKSİYONU (Tek Seferlik Tanımlama) ---
print(f"EfficientNetV2_M modeli ({CONFIG_EFFNET_M_SINGLE['base_model_name']}) oluşturuluyor...")
model = TimmEffNetModel(
    base_model_name=CONFIG_EFFNET_M_SINGLE["base_model_name"], pretrained=True,
    num_classes=CONFIG_EFFNET_M_SINGLE["num_classes"],
    in_chans_spectrogram=CONFIG_EFFNET_M_SINGLE["in_chans_spectrogram"],
    effnet_in_chans=CONFIG_EFFNET_M_SINGLE["effnet_in_chans"],
    model_drop_rate=CONFIG_EFFNET_M_SINGLE["timm_model_drop_rate"])
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=CONFIG_EFFNET_M_SINGLE["learning_rate"], weight_decay=CONFIG_EFFNET_M_SINGLE["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG_EFFNET_M_SINGLE["num_epochs"], eta_min=1e-7)
criterion = nn.BCEWithLogitsLoss()

# --- EĞİTİM VE VALİDASYON DÖNGÜSÜ (Tekli, K-Fold'suz) ---
best_val_loss = float('inf')
epochs_no_improve = 0
saved_model_path = None # En iyi modelin yolunu tutacak

for epoch in range(1, CONFIG_EFFNET_M_SINGLE["num_epochs"] + 1):
    start_time_epoch = time.time(); model.train(); train_loss_epoch = 0
    all_train_preds_epoch, all_train_labels_epoch = [], []
    progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG_EFFNET_M_SINGLE['num_epochs']} [Train]", unit="B", leave=True) # leave=True
    for batch_idx, (inputs, labels) in enumerate(progress_bar_train):
        inputs, labels = inputs.to(device), labels.to(device); optimizer.zero_grad()
        outputs = model(inputs); logits = outputs['logit']
        loss = criterion(logits, labels); loss.backward(); optimizer.step()
        train_loss_epoch += loss.item()
        preds = torch.sigmoid(logits) > CONFIG_EFFNET_M_SINGLE["threshold"]
        all_train_preds_epoch.extend(preds.cpu().numpy()); all_train_labels_epoch.extend(labels.cpu().numpy())
        if batch_idx > 0 and batch_idx % 75 == 0: progress_bar_train.set_postfix(loss=train_loss_epoch / (batch_idx+1))
   
    train_loss_epoch /= len(train_loader) if len(train_loader) > 0 else 1
    train_accuracy_epoch = accuracy_score(np.array(all_train_labels_epoch), np.array(all_train_preds_epoch))
    train_f1_epoch = f1_score(np.array(all_train_labels_epoch), np.array(all_train_preds_epoch), average='macro', zero_division=0)

    model.eval(); val_loss_epoch = 0
    all_val_preds_epoch, all_val_labels_epoch = [], []
    progress_bar_val = tqdm(valid_loader, desc=f"Epoch {epoch}/{CONFIG_EFFNET_M_SINGLE['num_epochs']} [Valid]", unit="B", leave=True) # leave=True
    with torch.no_grad():
        for inputs, labels in progress_bar_val:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs); logits = outputs['logit']; loss = criterion(logits, labels)
            val_loss_epoch += loss.item()
            preds = torch.sigmoid(logits) > CONFIG_EFFNET_M_SINGLE["threshold"]
            all_val_preds_epoch.extend(preds.cpu().numpy()); all_val_labels_epoch.extend(labels.cpu().numpy())
   
    val_loss_epoch /= len(valid_loader) if len(valid_loader) > 0 else 1
    val_accuracy_epoch = accuracy_score(np.array(all_val_labels_epoch), np.array(all_val_preds_epoch)) if len(all_val_labels_epoch) > 0 else 0.0
    val_f1_epoch = f1_score(np.array(all_val_labels_epoch), np.array(all_val_preds_epoch), average='macro', zero_division=0) if len(all_val_labels_epoch) > 0 else 0.0
   
    scheduler.step()
    epoch_duration = time.time() - start_time_epoch
    print(f"Epoch {epoch}/{CONFIG_EFFNET_M_SINGLE['num_epochs']} ({epoch_duration:.0f}s) "
          f"TrL: {train_loss_epoch:.4f} TrA: {train_accuracy_epoch:.4f} TrF1: {train_f1_epoch:.4f} | "
          f"VaL: {val_loss_epoch:.4f} VaA: {val_accuracy_epoch:.4f} VaF1: {val_f1_epoch:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.1e}")

    if val_loss_epoch < best_val_loss:
        best_val_loss = val_loss_epoch; epochs_no_improve = 0
        # Dosya adından _foldX kısmını kaldırıyoruz
        current_model_save_path = os.path.join(MODEL_OUTPUT_DIR, f"{CONFIG_EFFNET_M_SINGLE['base_model_name']}_ep{epoch}_vl{val_loss_epoch:.4f}.pth")
        torch.save(model.state_dict(), current_model_save_path);
        saved_model_path = current_model_save_path # En son kaydedilen genel en iyi modelin yolu
        print(f"Model kaydedildi: {saved_model_path} (En iyi val_loss: {val_loss_epoch:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= CONFIG_EFFNET_M_SINGLE["patience"]:
            print(f"Erken durdurma: Validasyon kaybı {CONFIG_EFFNET_M_SINGLE['patience']} epoch boyunca iyileşmedi.")
            break
    gc.collect(); torch.cuda.empty_cache()

print(f"\nEğitim tamamlandı. En iyi validasyon kaybı: {best_val_loss:.4f}")
if saved_model_path and os.path.exists(saved_model_path):
    final_model_path = os.path.join(MODEL_OUTPUT_DIR, f"{CONFIG_EFFNET_M_SINGLE['base_model_name']}_best_overall_model.pth")
    import shutil; shutil.copyfile(saved_model_path, final_model_path)
    print(f"En iyi model ayrıca şuraya kopyalandı/kaydedildi: {final_model_path}")
else: print("Hiçbir model (erken durdurma kriterlerini karşılayacak kadar iyileşen) kaydedilmedi.")

