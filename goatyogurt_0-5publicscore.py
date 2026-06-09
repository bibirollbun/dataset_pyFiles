import torch.nn as nn
import torchvision.models as models

import pytorch_lightning as pl
from torchvision.models import regnet_y_400mf, RegNet_Y_400MF_Weights


# the efficientnet model
class BirdClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes, pretrained=True):
        super(BirdClassifier, self).__init__()
        
        # Get the EfficientNet backbone
        if backbone_name == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b3(weights=weights)
            backbone_features = backbone.features
            num_ftrs = 1536  # For EfficientNet B3
        
        # Extract the feature extractor (everything except the classifier)
        self.backbone = backbone_features
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Custom classifier with an additional hidden layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 512),  # Add a hidden layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Softmax activation for final layer (not included in training since CrossEntropyLoss has it)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Extract features using the backbone
        x = self.backbone(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        # Classification head
        x = self.classifier(x)
        
        # Note: We don't apply softmax during training since CrossEntropyLoss includes it
        # We only apply it during inference or when raw probabilities are needed
        return x
    
    def predict_proba(self, x):
        # For getting probabilities during inference
        logits = self.forward(x)
        return self.softmax(logits)


# the regnet model
class RegNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=4, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # RegNet setup
        self.model = regnet_y_400mf(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        # For manual tracking
        self.val_losses = []
        self.val_accuracies = []
        self._val_loss_batches = []
        self._val_acc_batches = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Save to temporary lists for epoch-end aggregation
        self._val_loss_batches.append(loss)
        self._val_acc_batches.append(acc)

        # Log per batch for trainer bar
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        # Average batch metrics for this epoch
        avg_loss = torch.stack(self._val_loss_batches).mean().item()
        avg_acc = torch.stack(self._val_acc_batches).mean().item()

        # Store for plotting
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_acc)

        # Log epoch-level values
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_acc", avg_acc, prog_bar=True)

        # Clear lists for next epoch
        self._val_loss_batches.clear()
        self._val_acc_batches.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.8)
        return [optimizer], [scheduler]


import torch

device = torch.device('cpu')
print(device)

efficient_net_model = BirdClassifier('efficientnet_b3', 206, pretrained=False)
efficient_net_model.load_state_dict(torch.load('/kaggle/input/efficientnet/pytorch/default/1/best_custom_efficientnet_b3_model.pth', map_location=device))
efficient_net_model.eval()

reg_net_model = RegNetClassifier()
reg_net_model.load_state_dict(torch.load('/kaggle/input/regnet/pytorch/default/1/regnet_final_weights_20.pth', map_location=device))
reg_net_model.eval()


import os
import librosa
import numpy as np
import pandas as pd
import torch.nn.functional as F

SAMPLE_RATE = 32000
DURATION = 5
N_MELS = 128 # EfficientNet works well with larger image sizes, 128 or 224 are common
FMIN = 20
FMAX = 16000
HOP_LENGTH = 512
N_FFT = 2048

# Set seed
np.random.seed(42)

# Class labels from train audio
class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))

# List of test soundscapes (only visible during submission)
test_soundscape_path = '/kaggle/input/birdclef-2025/test_soundscapes'
test_soundscapes = [os.path.join(test_soundscape_path, afile) for afile in sorted(os.listdir(test_soundscape_path)) if afile.endswith('.ogg')]
# Open each soundscape and make predictions for 5-second segments
# Use pandas df with 'row_id' plus class labels as columns
predictions = pd.DataFrame(columns=['row_id'] + class_labels)


for soundscape in test_soundscapes:

    # Load audio
    sig, rate = librosa.load(path=soundscape, sr=None)

    # Split into 5-second chunks
    chunks = []
    for i in range(0, len(sig), rate*5):
        chunk = sig[i:i+rate*5]
        chunks.append(chunk)
        
        
    # Make predictions for each chunk
    for i, chunk in enumerate(chunks):
        
        # Get row id  (soundscape id + end time of 5s chunk)      
        row_id = os.path.basename(soundscape).split('.')[0] + f'_{i * 5 + 5}'
        
        # Make prediction (let's use random scores for now)
        target_length = len(chunk)

        mel_spec = librosa.feature.melspectrogram(
            y=chunk, sr=32000, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # --- Normalization (Optional but recommended for pre-trained models) ---
        # Normalize to roughly [0, 1] or [-1, 1] range if needed,
        # or use ImageNet stats if model expects that.
        # Simple min-max scaling to [0, 1]:
        min_val = np.min(mel_spec_db)
        max_val = np.max(mel_spec_db)
        if max_val > min_val:
             mel_spec_db = (mel_spec_db - min_val) / (max_val - min_val)
            # Convert to PyTorch Tensor and add channel dimension
        spectrogram_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)
            
            # Repeat channel 3 times to mimic RGB
        spectrogram_tensor_3channel = spectrogram_tensor.repeat(3, 1, 1)  # Output shape: (3, H, W)
        spectrogram_tensor_3channel = spectrogram_tensor_3channel.unsqueeze(0)
        scores = F.softmax(efficient_net_model(spectrogram_tensor_3channel), dim=1)
        
        # Append to predictions as new row
        # Flatten the scores tensor and convert to a list of floats
        scores_list = scores.squeeze().detach().cpu().numpy().tolist()
        # print(scores_list.index(max(scores_list)))
        # Create the DataFrame row
        new_row = pd.DataFrame([[row_id] + scores_list], columns=['row_id'] + class_labels)
        predictions = pd.concat([predictions, new_row], axis=0, ignore_index=True)
# Save prediction as csv
predictions.to_csv('submission1.csv', index=False)
predictions.head()


import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import cv2

taxonomy_df = pd.read_csv('/kaggle/input/birdclef-2025/taxonomy.csv')


def save_chunked_rgb_images_trimmed(df, speech_info, audio_root, output_dir, chunk_duration=5, img_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    speech_map = {entry['filename']: entry['speech_timestamps'] for entry in speech_info}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['filename']
        animal_class = row['animal_class']
        input_path = os.path.join(audio_root, filename)

        try:
            y, sr = librosa.load(input_path, sr=None)
        except Exception as e:
            print(f"[ERROR] loading {filename}: {e}")
            continue

        duration_sec = len(y) / sr

        # Trim human speech if found
        if filename in speech_map:
            speech = speech_map[filename]
            if speech:
                speech_starts = [mmss_to_seconds(seg['start']) for seg in speech]
                speech_ends = [mmss_to_seconds(seg['end']) for seg in speech]
                earliest = min(speech_starts)
                latest = max(speech_ends)

                if latest < duration_sec / 2:
                    trim_sec = int(np.ceil(latest / chunk_duration) * chunk_duration)
                    y = y[int(trim_sec * sr):]
                elif earliest > duration_sec / 2:
                    trim_sec = int(np.floor(earliest / chunk_duration) * chunk_duration)
                    y = y[:int(trim_sec * sr)]
                else:
                    # Mid-speech → skip
                    pass

        # 5-second chunking
        samples_per_chunk = int(sr * chunk_duration)
        n_chunks = int(len(y) / samples_per_chunk)
        if n_chunks == 0:
            pass

        base_name = os.path.splitext(os.path.basename(filename))[0]
        class_dir = os.path.join(output_dir, animal_class)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(n_chunks):
            start = i * samples_per_chunk
            end = start + samples_per_chunk
            chunk = y[start:end]

            rgb_img = extract_rgb_features(chunk, sr, img_size=img_size)
            if rgb_img is None:
                continue

            out_name = f"{base_name}_clip_{i}.png"
            out_path = os.path.join(class_dir, out_name)

            try:
                cv2.imwrite(out_path, rgb_img)
            except Exception as e:
                print(f"[ERROR] saving {out_path}: {e}")

def extract_rgb_features(y, sr, img_size=(224, 224)):
    try:
        # Compute features
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        delta = librosa.feature.delta(log_mel)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        # Resize features
        log_mel_resized = resize_feat(log_mel, img_size)
        delta_resized = resize_feat(delta, img_size)
        chroma_resized = resize_feat(chroma, img_size)

        # Normalize
        log_mel_norm = normalize(log_mel_resized)
        delta_norm = normalize(delta_resized)
        chroma_norm = normalize(chroma_resized)

        # Stack to form RGB image
        rgb_image = np.stack([log_mel_norm, delta_norm, chroma_norm], axis=-1)
        rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)

        return rgb_image_uint8

    except Exception as e:
        print(f"[ERROR in feature extraction]: {e}")
        return None

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)
    
def resize_feat(feat, target_shape):
    from cv2 import resize, INTER_LINEAR
    return resize(feat, target_shape, interpolation=INTER_LINEAR)


#Đọc các file trong test_soundscapes và tiến hành dự đoán
chunk_dir = "/kaggle/working/test_soundscapes_chunked"
os.makedirs(chunk_dir, exist_ok=True)

test_dir = "/kaggle/input/birdclef-2025/test_soundscapes"
test_files = [f for f in sorted(os.listdir(test_dir)) if f.endswith(".ogg")]
#test_dir = get_soundscape_dir()
#all_files = [f for f in os.listdir(test_dir) if f.endswith(".ogg")]
#test_files = all_files[:100]

# Nếu có file .ogg thì xử lý như bình thường
if test_files:
    test_df = pd.DataFrame({'filename': test_files, 'animal_class': 'Unknown'})

    save_chunked_rgb_images_trimmed(
        df=test_df,
        speech_info=[],
        audio_root=test_dir,
        output_dir=chunk_dir,
        chunk_duration=5,
        img_size=(224, 224)
    )
else:
    print("⚠️ No test soundscape files found. Skipping chunking. Will create empty submission.")

# Tiếp tục nếu thư mục chứa ảnh đã được tạo
if os.path.exists(chunk_dir) and any(os.scandir(chunk_dir)):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    test_ds = ImageFolder(chunk_dir, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    predictions = []
    file_names = []

    with torch.no_grad():
        for batch, _ in test_loader:
            batch = batch.to(reg_net_model.device)
            logits = reg_net_model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            predictions.extend(probs)

            batch_indices = test_loader.dataset.samples[
                len(file_names):len(file_names)+len(batch)
            ]
            file_names.extend([os.path.basename(path[0]) for path in batch_indices])

    # Tạo row_ids
    row_ids = []
    for name in file_names:
        base = name.replace('.png', '')
        parts = base.split('_clip_')
        if len(parts) == 2:
            row_id = f"{parts[0]}_{(int(parts[1]) + 1) * 5}"
            row_ids.append(row_id)
        else:
            row_ids.append(name)

else:
    # Nếu không có ảnh nào thì tạo row_id & predict rỗng từ sample_submission
    #sample_sub = pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')
    #row_ids = sample_sub['row_id'].tolist()
    #predictions = [[0.0] * len(taxonomy_df['primary_label'])] * len(row_ids)
    row_ids = []
    predictions = []


# Hàm tạo submission
def create_submission(row_ids, predictions, submission_template_path, species_ids):
    submission_dict = {'row_id': row_ids}
    for i, species in enumerate(species_ids):
        submission_dict[species] = [pred[i] if i < len(pred) else 0.0 for pred in predictions]

    submission_df = pd.DataFrame(submission_dict)
    submission_df.set_index('row_id', inplace=True)

    sample_sub = pd.read_csv(submission_template_path, index_col='row_id')
    for col in sample_sub.columns:
        if col not in submission_df.columns:
            submission_df[col] = 0.0
    submission_df = submission_df[sample_sub.columns]
    return submission_df.reset_index()

species_ids = taxonomy_df['primary_label'].tolist()
submission_df = create_submission(
    row_ids=row_ids,
    predictions=predictions,
    submission_template_path='/kaggle/input/birdclef-2025/sample_submission.csv',
    species_ids=species_ids
)

submission_df.to_csv("submission2.csv", index=False)

# ✅ In kết quả
print("✅ Submission preview:")
display(submission_df.head())


# !rm -rf /kaggle/working/*


FILES_SUBM = [
    '/kaggle/working/submission1.csv',
    '/kaggle/working/submission2.csv'
]
ENSEMBLE_SOLUTIONS = ['SOLUTION_1', 'SOLUTION_2']  # arbitrary names
WEIGHTS = [0.9, 0.1]  # or any custom weights (must sum to 1)
LB = ['N/A', 'N/A']   # leaderboard scores (optional)
OPTION = 'weighted'


def ens2(lbs        = LB,
         solution   = ENSEMBLE_SOLUTIONS,
         wts        = WEIGHTS,
         files_subm = FILES_SUBM,
         option     = OPTION):

    soluts = [solut.replace("SOLUTION_", "") for solut in solution]
    print(f'Ensemble: {soluts},   LB: {lbs},   weights: {wts}')
    
    list_TARGETs = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))

    list_targets_0 = [f'{TARGET} 0' for TARGET in list_TARGETs]
    list_targets_1 = [f'{TARGET} 1' for TARGET in list_TARGETs]

    df0 = pd.read_csv(files_subm[0])
    df1 = pd.read_csv(files_subm[1])

    # Rename columns to avoid collisions
    df0 = df0.rename(columns={TARGET: f'{TARGET} 0' for TARGET in list_TARGETs})
    print('df0:', df0)
    df1 = df1.rename(columns={TARGET: f'{TARGET} 1' for TARGET in list_TARGETs})
    print('df1:', df1)
    
    dfs = pd.merge(df0, df1, on='row_id')
    print('dfs: ', dfs)

    if option == 'weighted':
        # Efficiently compute ensembled targets using dictionary comprehension
        new_targets_df = pd.DataFrame({
            tgt: dfs[tgt0] * wts[0] + dfs[tgt1] * wts[1]
            for tgt, tgt0, tgt1 in zip(list_TARGETs, list_targets_0, list_targets_1)
        })

    if option == 'max':
        new_targets_df = pd.DataFrame({
            tgt: dfs[[tgt0, tgt1]].max(axis=1)
            for tgt, tgt0, tgt1 in zip(list_TARGETs, list_targets_0, list_targets_1)
        })

    print('new_targets_df: ', new_targets_df)

    # Concatenate with 'row_id' column only
    dfs = pd.concat([dfs[['row_id']], new_targets_df], axis=1)

    return dfs


# Now call the function
ensemble_submission = ens2(
    lbs=LB,
    solution=ENSEMBLE_SOLUTIONS,
    wts=WEIGHTS,
    files_subm=FILES_SUBM,
    option=OPTION
)

ensemble_submission.to_csv('/kaggle/working/submission.csv', index=False)


a = pd.read_csv('/kaggle/working/submission.csv')
a

