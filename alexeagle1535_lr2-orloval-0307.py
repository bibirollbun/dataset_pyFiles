import os
import shutil

shutil.copytree('../input/resnest50-fast-package/resnest-0.0.6b20200701/resnest', 'resnet', dirs_exist_ok=True)
os.system('pip install "./resnet" --no-deps')

import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from resnest.torch import resnest50
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ================== CONFIG ==================
DATA_ROOT = '../input/birdclef-2021'
TRAIN_SHORT_AUDIO = os.path.join(DATA_ROOT, 'train_short_audio')
TRAIN_SOUNDCAPES = os.path.join(DATA_ROOT, 'train_soundscapes')
TEST_AUDIO_PATH = os.path.join(DATA_ROOT, 'test_soundscapes')
TRAIN_META = os.path.join(DATA_ROOT, 'train_metadata.csv')
TEST_META = os.path.join(DATA_ROOT, 'test.csv')
SUBMISSION_CSV = 'submission.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SR = 32000
SEG_DUR = 5  # seconds
BATCH_SIZE = 64
THRESH = 0.25


# ================== LOAD METADATA ==================
train_meta = pd.read_csv(TRAIN_META)
species_list = sorted(train_meta['primary_label'].unique())
label_encoder = LabelEncoder().fit(species_list)
NUM_SPECIES = len(species_list)

# ================== AUDIO -> MEL-SPECT ==================
def transform_audio_to_spec(signal, sr=SR):
    mel_spec = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_mels=128, fmin=0, fmax=sr//2, n_fft=sr//10, hop_length=sr//40
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    norm = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    min_val, max_val = norm.min(), norm.max()
    if max_val - min_val > 1e-6:
        scaled = 255 * (norm - min_val) / (max_val - min_val)
    else:
        scaled = np.zeros_like(norm)
    return scaled.astype(np.uint8)

def make_rgb_tensor(spec_img):
    return np.stack([spec_img]*3, axis=0).astype(np.float32)/255.0



# ================== DATASET ==================
class AudioSegmentDataset(Dataset):
    def __init__(self, annotations, audio_folder, sr=SR, seg_len_sec=SEG_DUR):
        self.annotations = annotations.reset_index(drop=True)
        self.folder = audio_folder
        self.sr = sr
        self.seg_len = seg_len_sec
        self._audio_cache = {}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        row_id = row['row_id']
        full_prefix = '_'.join(row_id.split('_')[:2])
        end_time = int(row_id.split('_')[-1])

        if full_prefix not in self._audio_cache:
            audio_file = next(f for f in os.listdir(self.folder) if f.startswith(full_prefix))
            audio_full, orig_sr = librosa.load(os.path.join(self.folder, audio_file), sr=None, res_type='kaiser_fast')
            if orig_sr != self.sr:
                audio_full = librosa.resample(audio_full, orig_sr=orig_sr, target_sr=self.sr)
            self._audio_cache[full_prefix] = audio_full
        else:
            audio_full = self._audio_cache[full_prefix]

        start_sample = max(0, (end_time - self.seg_len) * self.sr)
        end_sample = min(len(audio_full), end_time * self.sr)
        segment = audio_full[start_sample:end_sample]
        if len(segment) < self.seg_len * self.sr:
            segment = np.pad(segment, (0, self.seg_len * self.sr - len(segment)))

        spec = transform_audio_to_spec(segment, self.sr)
        tensor = make_rgb_tensor(spec)
        return tensor


def build_inference_model(weight_path, num_classes):
    net = resnest50(pretrained=False)
    net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    checkpoint = torch.load(weight_path, map_location='cpu')
    clean_state = {k.replace('model.', ''): v for k,v in checkpoint.items()}
    net.load_state_dict(clean_state)
    net.to(DEVICE)
    net.eval()
    return net


def run_inference_on_batch(batch_data, model, thr=THRESH):
    with torch.no_grad():
        inputs = torch.from_numpy(batch_data).to(DEVICE)
        logits = model(inputs)
        probs = torch.sigmoid(logits).cpu().numpy()

    results = []
    for prob_vec in probs:
        active_labels = np.where(prob_vec > thr)[0]
        if len(active_labels) == 0:
            results.append('nocall')
        else:
            names = label_encoder.inverse_transform(active_labels)
            results.append(' '.join(sorted(names)))
    return results


# ================== TEST / SUBMISSION ==================
use_train_as_test = False
try:
    test_meta = pd.read_csv(TEST_META)
    if len(test_meta) < 10:
        use_train_as_test = True
except:
    use_train_as_test = True

if use_train_as_test:
    test_meta = pd.read_csv(os.path.join(DATA_ROOT, 'train_soundscape_labels.csv'))
    audio_source = TRAIN_SOUNDCAPES
else:
    audio_source = TEST_AUDIO_PATH

audio_dataset = AudioSegmentDataset(test_meta, audio_source)
loader = DataLoader(audio_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

MODEL_WEIGHTS_FILE = '../input/kkiller-birdclef-models-public/birdclef_resnest50_fold0_epoch_10_f1_val_06471_20210417161101.pth'
model = build_inference_model(MODEL_WEIGHTS_FILE, NUM_SPECIES)

final_predictions = []
for batch in tqdm(loader, desc='Running inference'):
    stacked_batch = np.stack([b for b in batch])
    batch_preds = run_inference_on_batch(stacked_batch, model, thr=THRESH)
    final_predictions.extend(batch_preds)

submission_df = pd.DataFrame({
    'row_id': test_meta['row_id'],
    'birds': final_predictions
})
submission_df.to_csv(SUBMISSION_CSV, index=False)
print(submission_df.head(10))


