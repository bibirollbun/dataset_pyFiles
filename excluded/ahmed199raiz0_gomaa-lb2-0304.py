import shutil
import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

RESNEST_SOURCE_DIR = '../input/resnest50-fast-package/resnest-0.0.6b20200701'
RESNEST_DEST_DIR = 'resnest_pkg'

print("Attempting to install ResNeSt using directory copy...")

try:
    if os.path.exists(RESNEST_DEST_DIR):
        print(f"Cleaning up existing directory: {RESNEST_DEST_DIR}")
        shutil.rmtree(RESNEST_DEST_DIR)
        
    shutil.copytree(os.path.join(RESNEST_SOURCE_DIR, 'resnest'), RESNEST_DEST_DIR)
    
    os.system(f'pip install "./{RESNEST_DEST_DIR}" --no-deps')
    print("ResNeSt installed successfully.")

except Exception as e:
    print(f"Failed to install ResNeSt. Final check on the dataset structure is needed if this fails again. Error: {e}")

from resnest.torch import resnest50

SR = 32000
DURATION = 5
THRESHOLD = 0.28 
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "../input/birdclef-2021"
TEST_AUDIO = os.path.join(DATA_ROOT, "test_soundscapes")
WEIGHTS = "../input/kkiller-birdclef-models-public/birdclef_resnest50_fold0_epoch_10_f1_val_06471_20210417161101.pth"

print(f"Device: {DEVICE}")
print(f"Threshold: {THRESHOLD}")


train_meta = pd.read_csv(os.path.join(DATA_ROOT, "train_metadata.csv"))
species = sorted(train_meta["primary_label"].unique())
encoder = LabelEncoder().fit(species)
NUM_CLASSES = len(species)

print(f"Species count: {NUM_CLASSES}")


def audio_to_melspec(audio, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128, fmin=300, fmax=sr//2,
        n_fft=sr//10, hop_length=sr//40
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    mean, std = log_mel.mean(), log_mel.std()
    normalized = (log_mel - mean) / (std + 1e-8)
    
    vmin, vmax = normalized.min(), normalized.max()
    if vmax - vmin > 1e-6:
        scaled = 255 * (normalized - vmin) / (vmax - vmin)
    else:
        scaled = np.zeros_like(normalized)
    
    return scaled.astype(np.uint8)

def to_rgb(spec):
    return np.stack([spec] * 3, axis=0).astype(np.float32) / 255.0


class SoundscapeDataset(Dataset):
    def __init__(self, df, audio_dir, sr=SR, duration=DURATION):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.sr = sr
        self.duration = duration
        self.cache = {}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        row_id = row["row_id"]
        
        file_id = "_".join(row_id.split("_")[:2])
        end_time = int(row_id.split("_")[-1])
        
        try:
            if file_id not in self.cache:
                filename = next(f for f in os.listdir(self.audio_dir) if f.startswith(file_id))
                audio, orig_sr = librosa.load(
                    os.path.join(self.audio_dir, filename), 
                    sr=None, res_type='kaiser_fast'
                )
                if orig_sr != self.sr:
                    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sr)
                self.cache[file_id] = audio
            
            audio = self.cache[file_id]
            
            start = max(0, (end_time - self.duration) * self.sr)
            end = min(len(audio), end_time * self.sr)
            segment = audio[start:end]
            
            if len(segment) < self.duration * self.sr:
                segment = np.pad(segment, (0, self.duration * self.sr - len(segment)))
            
            spec = audio_to_melspec(segment, self.sr)
            return to_rgb(spec)
        
        except:
            return np.zeros((3, 128, 313), dtype=np.float32)


def load_model(weights_path, num_classes):
    model = resnest50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    state = torch.load(weights_path, map_location="cpu")
    state = {k.replace("model.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    
    model.to(DEVICE)
    model.eval()
    return model

model = load_model(WEIGHTS, NUM_CLASSES)
print("Model loaded")


@torch.no_grad()
def predict_batch(batch, model, threshold=THRESHOLD):
    inputs = torch.from_numpy(batch).to(DEVICE)
    logits = model(inputs)
    probs = torch.sigmoid(logits).cpu().numpy()
    
    predictions = []
    for prob in probs:
        indices = np.where(prob > threshold)[0]
        if len(indices) == 0:
            predictions.append("nocall")
        else:
            labels = encoder.inverse_transform(indices)
            predictions.append(" ".join(sorted(labels)))
    
    return predictions


test_df = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))

if len(test_df) < 10:
    print("Using train_soundscapes for testing")
    test_df = pd.read_csv(os.path.join(DATA_ROOT, "train_soundscape_labels.csv"))
    audio_dir = os.path.join(DATA_ROOT, "train_soundscapes")
else:
    audio_dir = TEST_AUDIO

print(f"Segments to process: {len(test_df)}")


dataset = SoundscapeDataset(test_df, audio_dir)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

all_predictions = []

print("Starting inference...")
for batch in tqdm(loader, desc="Processing"):
    batch_array = np.stack([b.numpy() for b in batch])
    preds = predict_batch(batch_array, model, THRESHOLD)
    all_predictions.extend(preds)

print("Inference complete")


submission = pd.DataFrame({
    "row_id": test_df["row_id"],
    "birds": all_predictions
})

submission.to_csv("submission.csv", index=False)

print("Saved to submission.csv")


if os.path.exists('resnest_pkg'):
    shutil.rmtree('resnest_pkg')
    print("Cleaned up resnest_pkg directory")




