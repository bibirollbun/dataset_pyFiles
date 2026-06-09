import os
import numpy as np
import pandas as pd
import librosa
import torch
import timm
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm

# Parameters
SR = 32000
DURATION = 5
N_MELS = 128
HOP_LENGTH = 512
IMG_SIZE = 224

# Load the model and label encoder
model_path = "/kaggle/input/svm-from-efficient-b0/scikitlearn/default/1/svm_model.pkl"
encoder_path = "/kaggle/input/svm-from-efficient-b0/scikitlearn/default/1/label_encoder.pkl"
clf = joblib.load(model_path)
le = joblib.load(encoder_path)

# Load EfficientNet feature extractor
class CNNFeatureExtractor(torch.nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        model = timm.create_model("efficientnet_b0", pretrained=False)
        if model_path:
            state_dict = torch.load(model_path, weights_only=True)
            model.load_state_dict(state_dict)
        model.classifier = torch.nn.Identity()
        self.model = model

    def forward(self, x):
        return self.model(x)

extractor = CNNFeatureExtractor(model_path="/kaggle/input/efficientnet-b0/other/default/1/efficientnet_b0.pth").eval()
device = torch.device("cpu")
extractor.to(device)

# Helper functions
def load_audio(filepath, sr=SR):
    try:
        y, _ = librosa.load(filepath, sr=sr, duration=DURATION)
        if len(y) < sr * DURATION:
            y = np.pad(y, (0, sr * DURATION - len(y)))
        return y
    except:
        return None

def compute_pcen(y, sr=SR):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    pcen = librosa.pcen(mel * (2**31))  # Amplify to avoid zeros
    return pcen

def pcen_to_rgb(pcen):
    pcen = (pcen - pcen.min()) / (pcen.max() - pcen.min())
    pcen = (pcen * 255).astype(np.uint8)
    pcen_rgb = np.stack([pcen] * 3, axis=-1)
    return cv2.resize(pcen_rgb, (IMG_SIZE, IMG_SIZE))

def extract_feature(pcen_rgb):
    img_tensor = torch.tensor(pcen_rgb).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        feature = extractor(img_tensor)
    return feature.cpu().numpy().flatten()

# Load class labels
train_audio_path = "/kaggle/input/birdclef-2025/train_audio/"
class_labels = sorted(os.listdir(train_audio_path))

# Load submission template
sample_df = pd.read_csv("/kaggle/input/birdclef-2025/sample_submission.csv")
tests_path = "/kaggle/input/birdclef-2025/test_soundscapes"
soundscape_files = [os.path.join(tests_path, afile) for afile in sorted(os.listdir(tests_path)) if afile.endswith('.ogg')]

# fallback if no files found
if len(soundscape_files) == 0:
    tests_path = '/kaggle/input/birdclef-2025/train_soundscapes'
    soundscape_files = [os.path.join(tests_path, f) for f in sorted(os.listdir(tests_path)) if f.endswith('.ogg')][:3]

submission_rows = []

# Process each soundscape file
for soundscape_file in tqdm(soundscape_files, desc="Processing soundscapes"):
    path = soundscape_file  # full path already

    try:
        y, _ = librosa.load(path, sr=SR)
        duration = int(len(y) / SR)

        for t in range(0, duration - DURATION + 1, DURATION):
            segment = y[t * SR : (t + DURATION) * SR]
            if len(segment) < SR * DURATION:
                segment = np.pad(segment, (0, SR * DURATION - len(segment)))

            pcen = compute_pcen(segment)
            pcen_rgb = pcen_to_rgb(pcen)
            feature = extract_feature(pcen_rgb)

            # Predict probabilities using SVM
            probs = clf.predict_proba([feature])[0]

            # Format prediction row
            soundscape_id = os.path.basename(soundscape_file).split('.')[0]
            row = {"row_id": f"{soundscape_id}_{(t + DURATION)}"}
            pred_row = dict(zip(class_labels, probs))
            output = [pred_row.get(lbl, 0.0) for lbl in class_labels]
            row.update(dict(zip(map(str, class_labels), output)))
            submission_rows.append(row)

    except Exception as e:
        print(f"Error processing {soundscape_file}: {e}")

# Save to CSV
submission_df = pd.DataFrame(submission_rows)
submission_df = submission_df[["row_id"] + class_labels]
submission_df.to_csv("submission.csv", index=False)
print("✅ submission.csv saved.")

