import os

base_path = '/kaggle/input/modeling'  # thư mục mặc định chứa các file/dataset bạn đã thêm vào
for root, dirs, files in os.walk(base_path):
    if 'ModelTuanTran1 ' in dirs:
        print("Đường dẫn tới modeling:", os.path.join(root, 'ModelTuanTran1'))


import os

paths = '/kaggle/input/modeling/ModelTuanTran1 /ResNet50_best.weights.h5'

if os.path.exists(paths):
      print(f"{paths} ✅ tồn tại.")
else:
      print(f"{paths} ❌ không tồn tại.")


import os
import time
import numpy as np
import pandas as pd
import librosa
import cv2
from tqdm import tqdm
import tensorflow as tf
import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.models import Model
from pathlib import Path

# Cấu hình CFG
class CFG:
    # Mel-spectrogram & Audio Params
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 256
    FMIN = 20
    FMAX = 16000
    TARGET_SHAPE = (256, 256)
    FS = 32000
    WINDOW_SIZE = 5
    POWER = 2.0
    NORM = 'slaney'
    PAD_MODE = 'reflect'
    IN_CHANNELS = 3

    # Model
    model_path_efficient = '/kaggle/input/modeling/ModelTuanTran1 /EfficientNetV2S_best.weights.h5'
    model_path_resnet = '/kaggle/input/modeling/ModelTuanTran1 /ResNet50_best.weights.h5'
    model_name_efficient = 'EfficientNetV2S'
    model_name_resnet = 'ResNet50'
    
    batch_size = 16
    use_tta = False
    threshold = 0.5

    # Datasets
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'

    # Utility
    debug = False
    debug_count = 3

cfg = CFG()
print(f"Loading taxonomy data...")
taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
species_ids = taxonomy_df['primary_label'].tolist()
num_classes = len(species_ids)
print(f"Number of classes: {num_classes}")

# Hàm tạo model giống cách huấn luyện
def create_model(preset, num_classes, cfg):
    inp = Input(shape=(cfg.TARGET_SHAPE[0], cfg.TARGET_SHAPE[1], cfg.IN_CHANNELS))
    if preset == 'EfficientNetV2S':
        backbone = keras.applications.EfficientNetV2S(
            include_top=False,
            weights=None,  # Sẽ load trọng số sau
            input_shape=(cfg.TARGET_SHAPE[0], cfg.TARGET_SHAPE[1], cfg.IN_CHANNELS)
        )
    elif preset == 'ResNet50':
        backbone = keras.applications.ResNet50(
            include_top=False,
            weights=None,  # Sẽ load trọng số sau
            input_shape=(cfg.TARGET_SHAPE[0], cfg.TARGET_SHAPE[1], cfg.IN_CHANNELS)
        )
    else:
        raise ValueError(f"Unsupported preset: {preset}")

    x = backbone(inp)
    x = GlobalAveragePooling2D()(x)
    out = Dense(num_classes, activation='sigmoid', name='classifier')(x)

    model = Model(inputs=inp, outputs=out)
    return model

# Load cả hai mô hình và trọng số
def load_models(cfg, num_classes):
    models = {}
    
    # Load EfficientNetV2S
    print(f"Loading EfficientNetV2S from: {cfg.model_path_efficient}")
    if not os.path.exists(cfg.model_path_efficient):
        raise FileNotFoundError(f"EfficientNetV2S weights not found at {cfg.model_path_efficient}")
    model_efficient = create_model(cfg.model_name_efficient, num_classes, cfg)
    model_efficient.load_weights(cfg.model_path_efficient)
    models['efficient'] = model_efficient
    
    # Load ResNet50
    print(f"Loading ResNet50 from: {cfg.model_path_resnet}")
    if not os.path.exists(cfg.model_path_resnet):
        raise FileNotFoundError(f"ResNet50 weights not found at {cfg.model_path_resnet}")
    model_resnet = create_model(cfg.model_name_resnet, num_classes, cfg)
    model_resnet.load_weights(cfg.model_path_resnet)
    models['resnet'] = model_resnet
    
    return models

# Xử lý mel-spectrogram
def audio2melspec(audio_data, cfg, tta_variant=None):
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    # TTA: Chỉ dùng biến thể 'noise'
    if tta_variant == 'noise':
        noise = np.random.normal(0, 0.01, audio_data.shape)
        audio_data = audio_data + noise
        
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=cfg.POWER,
        pad_mode=cfg.PAD_MODE,
        norm=cfg.NORM,
        htk=True,
        center=True,
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return mel_spec_norm

def process_audio_segment(audio_data, cfg, tta_variant=None):
    # Padding nếu đoạn âm thanh ngắn hơn 5 giây
    window_samples = int(cfg.FS * cfg.WINDOW_SIZE)
    if len(audio_data) < window_samples:
        audio_data = np.pad(
            audio_data,
            (0, window_samples - len(audio_data)),
            mode='constant'
        )
    
    mel_spec = audio2melspec(audio_data, cfg, tta_variant)
    
    if mel_spec.shape != cfg.TARGET_SHAPE:
        mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
    
    mel_spec = np.stack([mel_spec] * 3, axis=-1)
    return mel_spec.astype(np.float32)

def predict_on_spectrogram(audio_path, models, cfg, species_ids):
    predictions = []
    row_ids = []
    soundscape_id = os.path.splitext(os.path.basename(audio_path))[0]
    
    try:
        print(f"Processing {soundscape_id}")
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
        
        total_segments = int(len(audio_data) / (cfg.FS * cfg.WINDOW_SIZE))
        
        for segment_idx in range(total_segments):
            start_sample = segment_idx * cfg.FS * cfg.WINDOW_SIZE
            end_sample = start_sample + cfg.FS * cfg.WINDOW_SIZE
            segment_audio = audio_data[start_sample:end_sample]
            
            end_time_sec = (segment_idx + 1) * cfg.WINDOW_SIZE
            row_id = f"{soundscape_id}_{end_time_sec}"
            row_ids.append(row_id)

            tta_variants = None if not cfg.use_tta else 'noise'
            
            # Dự đoán trên toàn bộ segment 
            mel_spec = process_audio_segment(segment_audio, cfg, tta_variants)
            mel_spec = np.expand_dims(mel_spec, axis=0)
            
            # EfficientNetV2S
            preds_efficient = models['efficient'].predict(mel_spec, verbose=0).squeeze()
            
            # # ResNet50
            # preds_resnet = models['resnet'].predict(mel_spec, verbose=0).squeeze()
            
            # # # Kết hợp dự đoán của EfficientNetV2S và ResNet50 với trọng số 0.6/0.4
            # segment_preds = 0.65 * preds_efficient + 0.35 * preds_resnet
            segment_preds = preds_efficient
            
            predictions.append(segment_preds)
            
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
    
    return row_ids, predictions

def run_inference(cfg, models, species_ids):
    test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))
    
    if cfg.debug:
        print(f"Debug mode enabled, using only {cfg.debug_count} files")
        test_files = test_files[:cfg.debug_count]
    
    print(f"Found {len(test_files)} test soundscapes")

    all_row_ids = []
    all_predictions = []

    for audio_path in tqdm(test_files):
        row_ids, predictions = predict_on_spectrogram(str(audio_path), models, cfg, species_ids)
        all_row_ids.extend(row_ids)
        all_predictions.extend(predictions)
    
    return all_row_ids, all_predictions

def create_submission(row_ids, predictions, species_ids, cfg):
    print("Creating submission dataframe...")
    submission_dict = {'row_id': row_ids}
    
    for i, species in enumerate(species_ids):
        submission_dict[species] = [pred[i] for pred in predictions]

    submission_df = pd.DataFrame(submission_dict)
    submission_df.set_index('row_id', inplace=True)
    sample_sub = pd.read_csv(cfg.submission_csv, index_col='row_id')

    missing_cols = set(sample_sub.columns) - set(submission_df.columns)
    if missing_cols:
        print(f"Warning: Missing {len(missing_cols)} species columns in submission")
        for col in missing_cols:
            submission_df[col] = 0.0

    submission_df = submission_df[sample_sub.columns]
    submission_df = submission_df.reset_index()
    
    return submission_df

def main():
    start_time = time.time()
    print("Starting BirdCLEF-2025 inference...")
    print(f"TTA enabled: {cfg.use_tta}")

    try:
        models = load_models(cfg, num_classes)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    print("Model loaded successfully")

    row_ids, predictions = run_inference(cfg, models, species_ids)
    submission_df = create_submission(row_ids, predictions, species_ids, cfg)
    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    
    # Post-processing
    sub = pd.read_csv('submission.csv')
    cols = sub.columns[1:]
    groups = sub['row_id'].str.rsplit('_', n=1).str[0]
    groups = groups.values
    for group in np.unique(groups):
        sub_group = sub[group == groups]
        predictions = sub_group[cols].values
        new_predictions = predictions.copy()
        for i in range(1, predictions.shape[0]-1):
            new_predictions[i] = (predictions[i-1] * 0.2) + (predictions[i] * 0.6) + (predictions[i+1] * 0.2)
        new_predictions[0] = (predictions[0] * 0.9) + (predictions[1] * 0.1)
        new_predictions[-1] = (predictions[-1] * 0.9) + (predictions[-2] * 0.1)
        sub_group[cols] = new_predictions
        sub[group == groups] = sub_group
    sub.to_csv("submission.csv", index=False)
    
    end_time = time.time()
    print(f"Inference completed in {(end_time - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()


import pandas as pd

pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv').head()


import pandas as pd

pd.read_csv('submission.csv').head()

