import os
import time
import warnings
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from tqdm.auto import tqdm
import cv2
import timm

warnings.filterwarnings('ignore')

# Configuration
class CFG:
    # Yollar - Kaggle ortamı için
    test_soundscapes = "/kaggle/input/birdclef-2025/test_soundscapes"
    sample_submission = "/kaggle/input/birdclef-2025/sample_submission.csv"
    taxonomy_csv = "/kaggle/input/birdclef-2025/taxonomy.csv"
    
    # Eğitilmiş modellerin yolu
    kaggle_model_path = "/kaggle/input/seed-2023-numwork2/seed-2023-fold-0.pth"
    local_model_2 = "/kaggle/input/final-model-fold-01/final_model_fold1.pth"
    local_model_1 = "/kaggle/input/0-788-imp-seed-77-fold-0/0.788-imp-seed-77-fold-0.pth"
    
    # Ses işleme parametreleri
    sr = 32000
    duration = 5
    
    # Spectogram parametreleri
    n_mels = 224
    n_fft = 2048
    hop_length = 256
    fmin = 20
    fmax = 16000
    img_size = 224
    
    # Tahmin parametreleri - Hızlandırılmış
    batch_size = 64        # Daha büyük batch boyutu
    use_tta = True          # TTA kullanımı korundu (performans için)
    tta_count = 3           # TTA sayısı
    
    # Donanım
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hata ayıklama
    debug = False
    debug_count = 5

# Önceden hesaplayabileceğimiz değişkenler
SEGMENT_SAMPLES = CFG.duration * CFG.sr

# Yardımcı işlevler - Hafif optimizasyonlar
def mono_to_color(X, eps=1e-6, mean=None, std=None):
    """Grayscale spektrogramı normalize et ve renkli görüntüye dönüştür"""
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)
    
    # Clipleme ve [0, 255] aralığına normalize etme
    _min, _max = X.min(), X.max()
    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)
    
    # RGB kanallarını oluştur
    return np.stack([V, V, V], axis=-1)

def audio_to_image(audio, sr=32000, n_mels=224, n_fft=2048, hop_length=256, 
                   fmin=20, fmax=16000, img_size=224):
    """Ses verisini spektrogram görüntüsüne dönüştür"""
    # Mel spektrogramı oluştur
    melspec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=n_mels,
        n_fft=n_fft, 
        hop_length=hop_length, 
        fmin=fmin,
        fmax=fmax
    )
    
    # dB cinsinden dönüştür
    melspec = librosa.power_to_db(melspec, ref=np.max)
    
    # Renk dönüşümü ve boyutlandırma
    image = mono_to_color(melspec)
    image = cv2.resize(image, (img_size, img_size))
    
    # Kanalları modelin beklediği sıraya getir (HWC -> CHW)
    image = np.moveaxis(image, -1, 0)
    image = image.astype(np.float32) / 255.0
    
    return image

# Test-time augmentation
def apply_tta(image, tta_idx):
    # Keep only the 3 most effective TTAs
    if tta_idx == 0:
        return image
    elif tta_idx == 1:
        return torch.flip(image, dims=[-1])
    elif tta_idx == 2:
        mask = torch.ones_like(image)
        t = image.shape[-1] // 8
        mask[:, :, t:2*t, :] = 0
        return image * mask

# Model sınıfı - Eğitim kodlarıyla aynı olmalı
class BirdCLEFModel(nn.Module):
    def __init__(self, model_name, num_classes, in_channels=3):
        super(BirdCLEFModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        # timm kütüphanesiyle model oluştur
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=in_channels
        )
        
        # Özellik boyutunu al
        if hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            in_features = self.backbone.num_features if hasattr(self.backbone, 'num_features') else 1280
        
        # Özel sınıflandırıcı
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Omurga özellikleri
        features = self.backbone(x)
        
        # Özellik tensörünün boyutunu kontrol et ve gerekirse düzleştir
        if len(features.shape) == 4:
            features = F.adaptive_avg_pool2d(features, output_size=1).squeeze(-1).squeeze(-1)
        
        # Sınıflandırıcı
        logits = self.classifier(features)
        
        return logits

# Modelleri yükle - Optimizasyonlar
def load_models(cfg, species_ids):
    """Kaggle ve Yerel modelleri yükle"""
    models = []
    
    # Kaggle'da eğitilmiş fold0 modelini yükle
    if os.path.exists(cfg.kaggle_model_path):
        print(f"Kaggle'da eğitilmiş fold0 modeli yükleniyor: {cfg.kaggle_model_path}")
        
        # Model oluştur
        model_kaggle = BirdCLEFModel('efficientnet_b0', len(species_ids))
        
        # Ağırlıkları yükle
        checkpoint = torch.load(cfg.kaggle_model_path, map_location=cfg.device)
        model_kaggle.load_state_dict(checkpoint['model_state_dict'])
        model_kaggle = model_kaggle.to(cfg.device)
        model_kaggle.eval()  # Değerlendirme moduna getir
        
        print(f"Kaggle fold0 modeli başarıyla yüklendi, AUC: {checkpoint.get('best_auc', 'N/A')}")
        models.append(model_kaggle)
    else:
        print(f"UYARI: Kaggle fold0 modeli bulunamadı: {cfg.kaggle_model_path}")

    if os.path.exists(cfg.local_model_2):
        print(f"Yerel ortamda eğitilmiş model2 yükleniyor: {cfg.local_model_2}")
        
        # Model oluştur
        model_local_2 = BirdCLEFModel('efficientnet_b0', len(species_ids))
        
        # Ağırlıkları yükle
        checkpoint = torch.load(cfg.local_model_2, map_location=cfg.device)
        model_local_2.load_state_dict(checkpoint['model_state_dict'])
        model_local_2 = model_local_2.to(cfg.device)
        model_local_2.eval()  # Değerlendirme moduna getir
        
        print(f"Yerel model2 başarıyla yüklendi, AUC: {checkpoint.get('best_auc', 'N/A')}")
        models.append(model_local_2)
    else:
        print(f"UYARI: Yerel model2 bulunamadı: {cfg.local_model_2}")




    if os.path.exists(cfg.local_model_1):
        print(f"Yerel ortamda eğitilmiş model1 yükleniyor: {cfg.local_model_1}")
        
        # Model oluştur
        model_local_1 = BirdCLEFModel('efficientnet_b0', len(species_ids))
        
        # Ağırlıkları yükle
        checkpoint = torch.load(cfg.local_model_1, map_location=cfg.device)
        model_local_1.load_state_dict(checkpoint['model_state_dict'])
        model_local_1 = model_local_1.to(cfg.device)
        model_local_1.eval()  # Değerlendirme moduna getir
        
        print(f"Yerel model1 başarıyla yüklendi, AUC: {checkpoint.get('best_auc', 'N/A')}")
        models.append(model_local_1)
    else:
        print(f"UYARI: Yerel model1 bulunamadı: {cfg.local_model_1}")


    # En az bir model var mı kontrol et
    if len(models) == 0:
        raise ValueError("Hiçbir model yüklenemedi! Lütfen model yollarını kontrol edin.")
    
    print(f"Toplam {len(models)} model başarıyla yüklendi.")
    return models

# Ses üzerinde tahmin yapma - Batch işleme ile optimize edildi
def predict_on_audio(audio_path, models, cfg, species_ids):
    """Ses dosyası üzerinde tahmin yap"""
    predictions = []
    row_ids = []
    soundscape_id = Path(audio_path).stem
    
    try:
        print(f"İşleniyor: {soundscape_id}")
        
        # Ses dosyasını yükle
        try:
            audio_data, _ = librosa.load(audio_path, sr=cfg.sr)
        except Exception as e:
            print(f"Ses dosyası okuma hatası. Güvenli modda yeniden deneniyor: {audio_path}")
            import soundfile as sf
            audio_data, orig_sr = sf.read(audio_path)
            if len(audio_data.shape) > 1:  # Stereo -> Mono
                audio_data = librosa.to_mono(audio_data.T)
            if orig_sr != cfg.sr:
                audio_data = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=cfg.sr)
        
        # Segmentlere böl (5 saniyelik pencereler)
        total_segments = int(len(audio_data) / SEGMENT_SAMPLES)
        
        # Batch işleme için listeler
        batch_images = []
        batch_rows = []
        batch_indices = []
        
        for segment_idx in range(total_segments):
            start_sample = segment_idx * SEGMENT_SAMPLES
            end_sample = start_sample + SEGMENT_SAMPLES
            segment_audio = audio_data[start_sample:end_sample]
            
            # Son segmentin uzunluğunu kontrol et
            if len(segment_audio) < SEGMENT_SAMPLES:
                segment_audio = np.pad(segment_audio, (0, SEGMENT_SAMPLES - len(segment_audio)), 'constant')
            
            # Row ID oluştur (her 5 saniyelik segment için)
            end_time_sec = (segment_idx + 1) * cfg.duration
            row_id = f"{soundscape_id}_{end_time_sec}"
            
            # Ses segmentini görüntüye dönüştür
            img = audio_to_image(
                segment_audio,
                sr=cfg.sr,
                n_mels=cfg.n_mels,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                fmin=cfg.fmin,
                fmax=cfg.fmax,
                img_size=cfg.img_size
            )
            
            # Batch'e ekle
            batch_images.append(img)
            batch_rows.append(row_id)
            batch_indices.append(segment_idx)
            
            # Batch doldu mu veya son segment mi kontrol et
            if len(batch_images) >= cfg.batch_size or segment_idx == total_segments - 1:
                # Batch'i işle
                if batch_images:
                    # Tüm modellerin tahminlerini sakla
                    all_models_preds = []
                    
                    # Tensöre dönüştür
                    batch_tensor = torch.tensor(np.array(batch_images), dtype=torch.float32)
                    batch_tensor = batch_tensor.to(cfg.device)
                    
                    for model in models:
                        # TTA kullanılacak mı?
                        if cfg.use_tta:
                            all_tta_preds = []
                            
                            for tta_idx in range(cfg.tta_count):
                                # TTA uygula
                                tta_tensor = apply_tta(batch_tensor, tta_idx)
                                
                                # Tahmin
                                with torch.no_grad():
                                    outputs = model(tta_tensor)
                                    outputs = torch.sigmoid(outputs).cpu().numpy()
                                    all_tta_preds.append(outputs)
                            
                            # TTA tahminlerinin ortalaması
                            model_preds = np.mean(all_tta_preds, axis=0)
                        else:
                            # TTA kullanmadan tahmin
                            with torch.no_grad():
                                outputs = model(batch_tensor)
                                model_preds = torch.sigmoid(outputs).cpu().numpy()
                        
                        all_models_preds.append(model_preds)
                    
                    # Tüm modellerin tahminlerini ortala (ensemble)
                    # Modellerin ağırlıkları
                    if len(models) >= 3:
                        batch_ensemble_preds = all_models_preds[0] * 0.2 + all_models_preds[1] * 0.5 + all_models_preds[2] * 0.3
                    elif len(models) == 2:
                        batch_ensemble_preds = all_models_preds[0] * 0.25 + all_models_preds[1] * 0.75
                    else:
                        batch_ensemble_preds = all_models_preds[0]
                    
                    # Sonuçları ekle
                    row_ids.extend(batch_rows)
                    predictions.extend(batch_ensemble_preds)
                    
                    # Batch'i temizle
                    batch_images = []
                    batch_rows = []
                    batch_indices = []
            
    except Exception as e:
        print(f"Hata: {audio_path} dosyasını işlerken hata oluştu: {e}")
    
    return row_ids, predictions

# Tahminleri düzleştir (zaman içinde yumuşatma)
def smooth_predictions(row_ids, predictions, win_size=5):
    """Tahminleri zamansal olarak düzleştir"""
    if len(predictions) <= 1:
        return predictions
    
    # Aynı ses dosyasına ait segmentleri grupla
    row_id_parts = [r.rsplit('_', 1)[0] for r in row_ids]
    unique_groups = np.unique(row_id_parts)
    
    smoothed_predictions = np.copy(predictions)
    
    for group in unique_groups:
        # Bu gruba ait tahminleri bul
        group_indices = [i for i, r in enumerate(row_id_parts) if r == group]
        group_preds = predictions[group_indices]
        
        # Her segment için kaydırma penceresi uygula
        for i in range(len(group_indices)):
            # Pencere sınırlarını belirle
            win_start = max(0, i - win_size // 2)
            win_end = min(len(group_indices), i + win_size // 2 + 1)
            
            # Pencere içindeki tahminlerin ağırlıklı ortalaması
            if win_end - win_start > 1:
                # Merkezdeki tahmine daha fazla ağırlık ver
                weights = np.ones(win_end - win_start)
                center_idx = i - win_start
                if 0 <= center_idx < len(weights):
                    weights[center_idx] = 2.0  # Merkeze daha fazla ağırlık
                weights = weights / weights.sum()
                
                weighted_sum = np.zeros_like(group_preds[0])
                for w_idx, pred_idx in enumerate(range(win_start, win_end)):
                    weighted_sum += weights[w_idx] * group_preds[pred_idx]
                
                smoothed_predictions[group_indices[i]] = weighted_sum
    
    return smoothed_predictions

# Tahmin işlemi - Paralel işleme için optimize edildi
def run_inference(cfg, models, species_ids):
    """Tüm test dosyaları üzerinde tahmin yap"""
    test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))
    
    if cfg.debug:
        print(f"Debug modu: Sadece {cfg.debug_count} dosya işlenecek")
        test_files = test_files[:cfg.debug_count]
    
    print(f"{len(test_files)} test ses dosyası bulundu")
    
    # Test dosyası yoksa
    if len(test_files) == 0:
        print(f"Uyarı: {cfg.test_soundscapes} dizininde .ogg dosyası bulunamadı!")
        print("Bu muhtemelen yerel test ortamında beklenen bir durum.")
        print("Kaggle submission sırasında dosyalar otomatik olarak eklenecektir.")
        
        # Örnek submission'dan blank template oluştur
        sample_sub = pd.read_csv(cfg.sample_submission)
        return sample_sub['row_id'].tolist(), [np.zeros(len(species_ids)) for _ in range(len(sample_sub))]
    
    all_row_ids = []
    all_predictions = []
    
    for audio_path in tqdm(test_files):
        row_ids, predictions = predict_on_audio(str(audio_path), models, cfg, species_ids)
        
        if row_ids and len(row_ids) > 0:
            all_row_ids.extend(row_ids)
            all_predictions.extend(predictions)
    
    # NumPy dizisine dönüştür
    if all_predictions:
        all_predictions = np.array(all_predictions)
        
        # Tahminleri düzleştir
        all_predictions = smooth_predictions(all_row_ids, all_predictions, win_size=5)
    
    return all_row_ids, all_predictions

# Submission oluştur
def create_submission(row_ids, predictions, species_ids, cfg):
    """Kaggle submission dosyası oluştur"""
    print("Submission dosyası oluşturuluyor...")
    
    # Submission sözlüğü oluştur
    submission_dict = {'row_id': row_ids}
    
    # Her tür için tahminleri ekle
    for i, species in enumerate(species_ids):
        submission_dict[species] = [pred[i] for pred in predictions]
    
    # DataFrame oluştur
    submission_df = pd.DataFrame(submission_dict)
    
    # Örnek submission ile karşılaştır ve eksik satırları kontrol et
    sample_sub = pd.read_csv(cfg.sample_submission)
    
    missing_rows = set(sample_sub['row_id']) - set(submission_df['row_id'])
    if missing_rows:
        print(f"Uyarı: {len(missing_rows)} satır eksik. Tahmin: {len(row_ids)}, Gerekli: {len(sample_sub)}")
        
        # Eksik satırlar için sıfır tahmin ekle
        for row_id in missing_rows:
            missing_dict = {'row_id': row_id}
            for species in species_ids:
                missing_dict[species] = 0.0
            
            # DataFrame'e ekle
            missing_df = pd.DataFrame([missing_dict])
            submission_df = pd.concat([submission_df, missing_df], ignore_index=True)
    
    # row_id'ye göre sırala
    submission_df = submission_df.sort_values('row_id').reset_index(drop=True)
    
    return submission_df

def main():
    start_time = time.time()
    print("BirdCLEF 2025 ensemble submission süreci başlatılıyor...")
    print(f"Kaggle model: {CFG.kaggle_model_path}")
    print(f"Yerel model 2: {CFG.local_model_2}")
    print(f"Yerel model 1: {CFG.local_model_1}")
    
    # Tür bilgilerini yükle
    taxonomy_df = pd.read_csv(CFG.taxonomy_csv)
    species_ids = taxonomy_df['primary_label'].tolist()
    num_classes = len(species_ids)
    print(f"{num_classes} türü için tahmin yapılacak.")
    
    # Modelleri yükle
    models = load_models(CFG, species_ids)
    
    # Tahmin yap
    print("Test ses dosyaları üzerinde tahmin yapılıyor...")
    row_ids, predictions = run_inference(CFG, models, species_ids)
    
    # Submission oluştur
    submission_df = create_submission(row_ids, predictions, species_ids, CFG)
    
    # Submission dosyasını kaydet
    submission_path = "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission dosyası kaydedildi: {submission_path}")
    
    # İstatistikler
    print("\nSubmission İstatistikleri:")
    print(f"Toplam satır sayısı: {len(submission_df)}")
    print(f"Tahmin edilen örnek sayısı: {len(row_ids)}")
    
    # Tahmin dağılımı kontrolü
    pred_means = submission_df.iloc[:, 1:].mean().mean()
    pred_std = submission_df.iloc[:, 1:].std().mean()
    print(f"Ortalama tahmin değeri: {pred_means:.4f}")
    print(f"Tahmin standart sapması: {pred_std:.4f}")
    
    # Çalışma süresi
    end_time = time.time()
    minutes = (end_time - start_time) / 60
    print(f"Tamamlandı! Çalışma süresi: {minutes:.2f} dakika")

if __name__ == "__main__":
    main()

