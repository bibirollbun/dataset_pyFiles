import timm
import torch.nn as nn
import torch
import librosa
import numpy as np
import os
from pathlib import Path
import pandas as pd
import glob


import os
print(os.listdir("/kaggle/input/birdclef-2025/test_soundscapes")[:5])


class BirdCLEFNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_s", pretrained=False, in_chans=1)
        self.backbone.global_pool = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)
    
    def forward(self, x):
        x = self.backbone.forward_features(x)  # [B, C, H, W]
        x = self.pooling(x).squeeze(-1).squeeze(-1)  # [B, C]
        x = self.classifier(x)  # [B, num_classes]
        return x


ckpt_path = "/kaggle/input/gba-filtered-checkpoint-random-5-sec-sample/checkpoints/baseline_v1_gba_clean_softsec_epoch5_auc0.96300.pth"
num_classes = 206
model = BirdCLEFNet(num_classes)
model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
model.eval()


SR = 32000
N_MELS = 128
TARGET_T = 309

def audio_to_logmelspec_segment(audio, sr=SR, n_mels=N_MELS, fmin=20, fmax=16000):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    logmel = librosa.power_to_db(mel).astype(np.float32)
    return logmel
    

def pad_or_crop(logmel, target_len=TARGET_T):
    _, t = logmel.shape
    if t < target_len:
        pad_width = target_len - t
        logmel = np.pad(logmel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        logmel = logmel[:, :target_len]
    return logmel


def prepare_input_for_model(audio_segment, sr=SR):
    if len(audio_segment) < sr * 5:
        pad_width = sr * 5 - len(audio_segment)
        audio_segment = np.pad(audio_segment, (0, pad_width), mode='constant')
    
    logmel = audio_to_logmelspec_segment(audio_segment, sr=sr)
    logmel = pad_or_crop(logmel)
    tensor = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor  # [1, 1, 128, 313]



def predict_for_soundscape(audio_path: str, model, device="cpu"):
    model.eval()
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    duration = librosa.get_duration(y=y, sr=SR)

    predictions = []
    row_ids = []

    filename = Path(audio_path).stem  # already 'soundscape_123456'

    for t_start in range(0, int(duration), 5):
        t_end = t_start + 5
        segment = y[t_start * SR : t_end * SR]

        x = prepare_input_for_model(segment).to(device)

        with torch.no_grad():
            probs = torch.sigmoid(model(x)).cpu().numpy().flatten()
        
        row_id = f"{filename}_{t_end}"
        
        predictions.append(probs)
        row_ids.append(row_id)

    return row_ids, predictions


from pathlib import Path
import pandas as pd

def get_audio_files(
    test_dir="/kaggle/input/birdclef-2025/test_soundscapes",
    fallback_dir="/kaggle/input/birdclef-2025/train_soundscapes",
    fallback_limit=8
):
    test_files = glob.glob(f"{test_dir}/*.ogg")
    
    if len(test_files) > 0:
        print("ğŸš€ Using real test data from test_soundscapes/")
        return test_files
    else:
        print("ğŸ§ª Test data not available â€” falling back to train_soundscapes/ for debugging.")
        fallback_files = sorted(glob.glob(f"{fallback_dir}/*.ogg"))[:fallback_limit]
        return fallback_files


def generate_submission(
    model, 
    audio_files, 
    taxonomy_csv: str, 
    device="cpu", 
    output_path="submission.csv"
):
    # Ğ—Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ° Ñ�Ğ¿Ğ¸Ñ�ĞºĞ° Ğ²Ğ¸Ğ´Ğ¾Ğ²
    taxonomy = pd.read_csv(taxonomy_csv)
    species_ids = taxonomy["primary_label"].tolist()

    all_row_ids = []
    all_probs = []

    for audio_path in audio_files:
        row_ids, predictions = predict_for_soundscape(str(audio_path), model, device=device)
        all_row_ids.extend(row_ids)
        all_probs.extend(predictions)

    # Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¸Ğµ submission DataFrame
    submission_df = pd.DataFrame(all_probs, columns=species_ids)
    submission_df.insert(0, "row_id", all_row_ids)

    submission_df.to_csv(output_path, index=False)
    print(f"âœ… Submission saved to {output_path} with shape {submission_df.shape}")
    return submission_df


TAXONOMY_CSV = "/kaggle/input/birdclef-2025/taxonomy.csv"
DEVICE = "cpu"

audio_files = get_audio_files()
submission_df = generate_submission(model, audio_files=audio_files, taxonomy_csv=TAXONOMY_CSV, device=DEVICE)


import pandas as pd
submission_df = pd.read_csv("submission.csv")
submission_df.head()


import pandas as pd
sample_df = pd.read_csv("/kaggle/input/birdclef-2025/sample_submission.csv")


sample_df


assert submission_df.shape[1] == sample_df.shape[1], f"â�Œ Submission must have {sample_df.shape[1]} columns (got {submission_df.shape[1]})"
assert submission_df.columns.tolist() == sample_df.columns.tolist(), "â�Œ Submission columns must match sample_submission.csv exactly"
assert submission_df["row_id"].is_unique, "â�Œ row_id must be unique"
assert submission_df.isnull().sum().sum() == 0, "â�Œ No NaNs allowed"




