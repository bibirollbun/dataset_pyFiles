import os
import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
from torchvision import models


# ------------------------
# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------
# Define the model (same architecture as used for training)
model = models.efficientnet_b0(pretrained=False)  # Do not load pre-trained weights
NUM_CLASSES = 206  # Update to your actual number of classes
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

# Load model weights
model.load_state_dict(torch.load("/kaggle/input/llllll/birdclef_test_model.pth"))
model = model.to(device)
model.eval()


# ------------------------
# Mel spectrogram processor (match training config!)
def process_audio_chunk(chunk, sr=32000):
    waveform = torch.tensor(chunk).unsqueeze(0)  # shape: (1, n_samples)
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    mel = mel_transform(waveform)
    mel = torchaudio.functional.amplitude_to_DB(mel, multiplier=10.0, db_multiplier=0.0, amin=1e-10, top_db=80.0)
    return mel  # shape: (1, n_mels, time)

# ------------------------
# Load model
class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))
num_classes = len(class_labels)



# ------------------------
# Predict on test soundscapes
test_soundscape_path = '/kaggle/input/birdclef-2025/test_soundscapes/'
test_soundscapes = [os.path.join(test_soundscape_path, afile) for afile in sorted(os.listdir(test_soundscape_path)) if afile.endswith('.ogg')]

predictions = pd.DataFrame(columns=['row_id'] + class_labels)

for soundscape in test_soundscapes:
    sig, rate = librosa.load(path=soundscape, sr=32000)  # Ensure sr matches training

    # Split into 5-second chunks
    for i in range(0, len(sig), rate * 5):
        chunk = sig[i:i + rate * 5]
        if len(chunk) < rate * 5:
            continue  # skip short tail

        # Process chunk
        mel = process_audio_chunk(chunk)  # shape: (1, n_mels, time)
        mel = mel.unsqueeze(0).to(device)  # shape: (B, 1, n_mels, time)

        with torch.no_grad():
            output = model(mel)
            scores = torch.sigmoid(output).cpu().numpy().flatten()

        soundscape_id = os.path.basename(soundscape).replace('.ogg', '')
        end_time = (i + rate * 5) // rate
        row_id = f"{soundscape_id}_{end_time}"

        new_row = pd.DataFrame([[row_id] + list(scores)], columns=['row_id'] + class_labels)
        predictions = pd.concat([predictions, new_row], ignore_index=True)

# ------------------------
# Save predictions
predictions.to_csv('submission.csv', index=False)
print("Submission saved as 'submission.csv'")
print(predictions.head())



