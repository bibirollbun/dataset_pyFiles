import os
import numpy as np
import pandas as pd

import librosa
import librosa.display
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms as T
from torch.optim import Adam

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import timm
from PIL import Image
from tqdm import tqdm


def plot_waveform(y, sr):
    """Plot the audio waveform."""
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")

def plot_spectrogram(y, sr):
    """Plot the spectrogram."""
    plt.subplot(3, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")

def plot_mel_spectrogram(y, sr):
    """Plot the Mel spectrogram."""
    plt.subplot(3, 1, 3)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")

def plot_all_spectrograms(file_path):
    """Load the audio and plot the waveform, spectrogram, and Mel spectrogram."""
    y, sr = librosa.load(file_path, sr=None)
    
    plt.figure(figsize=(12, 8))

    plot_waveform(y, sr)
    plot_spectrogram(y, sr)
    plot_mel_spectrogram(y, sr)
    
    plt.tight_layout()
    plt.show()
    
    return y, sr


def ogg_to_mono_spectrogram(file_path, sr=32000, n_mels=128, n_fft=1024, hop_length=512):
    """
    Convert an .ogg audio file into a 1-channel 256x256 tensor for a modified EfficientNet.

    Returns:
        torch.Tensor: Float tensor of shape (1, 256, 256).
    """
    y, _ = librosa.load(file_path, sr=sr)
    
    # Compute Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize and convert to image
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    mel_img = Image.fromarray(np.uint8(mel_norm * 255))

    # Resize and convert to 1-channel tensor
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()  # Shape: (1, H, W)
    ])

    return transform(mel_img)


dataset_path = "/kaggle/input/birdclef-2025"

test_soundscapes = "/kaggle/input/birdclef-2025/test_soundscapes"
train_audio= "/kaggle/input/birdclef-2025/train_audio"
train_soundscapes = "/kaggle/input/birdclef-2025/train_soundscapes"

recording_location_txt = "/kaggle/input/birdclef-2025/recording_location.txt"
sample_submission_csv = "/kaggle/input/birdclef-2025/sample_submission.csv"
taxonomy_csv = "/kaggle/input/birdclef-2025/taxonomy.csv"
train_csv = "/kaggle/input/birdclef-2025/train.csv"

DEVICE = "cuda"
SUBMISSION = True
ENSEMBLE = False


sample_submission = pd.read_csv(sample_submission_csv)
taxonomy = pd.read_csv(taxonomy_csv)
df = pd.read_csv(train_csv)


sample_submission.head()


plot_all_spectrograms("/kaggle/input/birdclef-2025/train_audio/65373/iNat1141476.ogg")


taxonomy.head()


taxonomy['class_name'].unique()


taxonomies = pd.read_csv(taxonomy_csv)
idtax = taxonomies.set_index('primary_label').index.to_series().reset_index(drop=True).to_dict()
# Create a mapping from ID to primary_label
taxid = {id_: label for label, id_ in idtax.items()}


taxid['1139490']


idtax[0]


print(len(idtax))


df.head()


if not SUBMISSION:
    df['fold'] = -1
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['primary_label'])):
        df.loc[val_idx, 'fold'] = fold


class BirdCLEF(Dataset):
    def __init__(self, df, data_dir, label2id, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame with columns like ['filepath', 'primary_label', 'fold']
            data_dir (str): Path to the directory with .ogg files
            label2id (dict): Dictionary mapping bird labels to class indices
            transform (callable, optional): Optional transform to apply to spectrogram
        """
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.label2id = label2id
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.data_dir, row["filename"])
        label = taxid[row["primary_label"]]

        spec = ogg_to_mono_spectrogram(file_path)  # shape: (1, 256, 256)

        if self.transform:
            spec = self.transform(spec)

        return spec, label


class BirdCLEF_EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
        
        
        in_channels = 1 # change first conv layer to accept 1-channel input
        old_conv = self.backbone.conv_stem
        self.backbone.conv_stem = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes) # just add linear classifier head

    def forward(self, x):
        return self.backbone(x)


from sklearn.metrics import precision_score, recall_score
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_one_fold(model, train_loader, val_loader, fold,
                   device='cuda', num_epochs=10, patience=3):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float('inf')
    patience_counter = 0
    num_classes = 206

    train_hist, val_hist = [], []

    for epoch in range(1, num_epochs+1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        model.train()
        train_losses = []
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            B = len(labels)
            targets = torch.zeros(B, num_classes, device=device)
            for i, labs in enumerate(labels):
                targets[i, labs] = 1.0

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                B = len(labels)
                targets = torch.zeros(B, num_classes, device=device)
                for i, labs in enumerate(labels):
                    targets[i, labs] = 1.0

                logits = model(inputs)
                loss = criterion(logits, targets)
                val_losses.append(loss.item())

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_preds.append(preds)
                all_targets.append(targets.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        avg_train = np.mean(train_losses)
        avg_val   = np.mean(val_losses)

        train_hist.append(avg_train)
        val_hist.append(avg_val)

        acc  = (all_preds == all_targets).sum() / all_targets.size
        prec = precision_score(all_targets, all_preds, average='micro')
        rec  = recall_score   (all_targets, all_preds, average='micro')

        print(
            f"[Fold {fold}] "
            f"Train Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val:.4f} | "
            f"Val Acc: {acc:.4f} | "
            f"Precision: {prec:.4f} | "
            f"Recall: {rec:.4f}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_fold{fold}.pt")
            print("Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return train_hist, val_hist


if not SUBMISSION:
    num_classes = len(taxid)
    model = BirdCLEF_EfficientNet(num_classes=num_classes)
    model.to(DEVICE)


if not SUBMISSION:
    train_losses, val_losses = train_one_fold(
        model, train_loader, val_loader, fold=0,
        device='cuda', num_epochs=10, patience=3
    )


if not SUBMISSION:
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Fold 0 Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()


if not SUBMISSION:
    to_plot = []
    for fold in range(5):
        print(f"Training fold {fold}")
        
        train_df = df[df.fold != fold]
        val_df = df[df.fold == fold]
    
        train_dataset = BirdCLEF(train_df, data_dir='/kaggle/input/birdclef-2025/train_audio', label2id=taxid)
        val_dataset = BirdCLEF(val_df, data_dir='/kaggle/input/birdclef-2025/train_audio', label2id=taxid)
    
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
        all_train, all_val = train_one_fold(model, train_loader, val_loader, fold)
        to_plot.append([all_train, all_val])



def waveform_to_mel_tensor(chunk, sr=32000, n_mels=128, n_fft=1024, hop_length=512):
    mel = librosa.feature.melspectrogram(
        y=chunk, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to 0-1
    mel_db -= mel_db.min()
    mel_db /= mel_db.max() + 1e-6

    # Convert to image and transform to tensor
    mel_img = Image.fromarray(np.uint8(mel_db * 255))
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()  # shape: [1, H, W], values in [0, 1]
    ])

    return transform(mel_img)  # shape: [1, 256, 256]


if ENSEMBLE:
    model_path0 = "/kaggle/input/best_sound_classification/pytorch/default/1/best_model_fold4.pt"
    model_path1 = "/kaggle/input/2best_sound_classification/pytorch/default/1/best_model_fold3.pt"
    model_path2 = "/kaggle/input/3best_sound_classification/pytorch/default/1/best_model_fold2.pt"
    model_path3 = "/kaggle/input/4best_sound_classification/pytorch/default/1/best_model_fold1.pt"
    model_path4 = "/kaggle/input/5best_sound_classification/pytorch/default/1/best_model_fold0.pt"


if ENSEMBLE:
    import os
    import numpy as np
    import pandas as pd
    import librosa
    import torch
    from PIL import Image
    import torchvision.transforms as T
    
    model_paths = [model_path0, model_path1, model_path2, model_path3, model_path4]
    models = []
    for p in model_paths:
        m = BirdCLEF_EfficientNet(num_classes=len(taxid))
        m.to("cpu")
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.eval()
        models.append(m)
    
    np.random.seed(42)
    class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))
    test_soundscape_path = '/kaggle/input/birdclef-2025/test_soundscapes/'
    test_soundscapes = [
        os.path.join(test_soundscape_path, afile)
        for afile in sorted(os.listdir(test_soundscape_path))
        if afile.endswith('.ogg')
    ]
    
    predictions = pd.DataFrame(columns=['row_id'] + class_labels)
    
    for soundscape in test_soundscapes:
        sig, rate = librosa.load(path=soundscape, sr=None)
        chunks = []
        for i in range(0, len(sig), rate * 5):
            chunk = sig[i:i + rate * 5]
            if len(chunk) < rate * 5:
                continue
            chunks.append(chunk)
        for i, chunk in enumerate(chunks):
            row_id = os.path.basename(soundscape).split('.')[0] + f'_{i * 5 + 5}'
            mel_spec_tensor = waveform_to_mel_tensor(chunk, sr=rate).unsqueeze(0).to("cpu")
            with torch.no_grad():
                logits_sum = sum(m(mel_spec_tensor)[0] for m in models)
                probs = torch.softmax(logits_sum / len(models), dim=0).cpu().numpy()
            new_row = pd.DataFrame([[row_id] + probs.tolist()], columns=['row_id'] + class_labels)
            predictions = pd.concat([predictions, new_row], axis=0, ignore_index=True)
    
    predictions.to_csv('submission.csv', index=False, float_format='%.16f')
    print(predictions.head())



if not ENSEMBLE:
    import os
    import numpy as np
    import pandas as pd
    import librosa
    import torch
    from PIL import Image
    import torchvision.transforms as T

    model_path = "/kaggle/input/new_best_sound_classification_model/pytorch/default/1/new_best_model.pt"
    model = BirdCLEF_EfficientNet(num_classes=len(taxid))
    model.to("cpu")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    np.random.seed(42)
    
    class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))
    
    test_soundscape_path = '/kaggle/input/birdclef-2025/test_soundscapes/'
    test_soundscapes = [
        os.path.join(test_soundscape_path, afile)
        for afile in sorted(os.listdir(test_soundscape_path))
        if afile.endswith('.ogg')
    ]
    
    predictions = pd.DataFrame(columns=['row_id'] + class_labels)
    
    for soundscape in test_soundscapes:
        sig, rate = librosa.load(path=soundscape, sr=None)
    
        chunks = []
        for i in range(0, len(sig), rate * 5):
            chunk = sig[i:i + rate * 5]
            if len(chunk) < rate * 5:
                continue  # skip too-short chunk
            chunks.append(chunk)
    
        for i, chunk in enumerate(chunks):
            row_id = os.path.basename(soundscape).split('.')[0] + f'_{i * 5 + 5}'
    
            mel_spec_tensor = waveform_to_mel_tensor(chunk, sr=rate).unsqueeze(0).to("cpu")  # [1, 1, 256, 256]
    
            with torch.no_grad():
                logits = model(mel_spec_tensor)[0]  # shape: [num_classes]
                scores = torch.softmax(logits, dim=0).cpu().numpy()
    
            new_row = pd.DataFrame([[row_id] + list(scores)], columns=['row_id'] + class_labels)
            predictions = pd.concat([predictions, new_row], axis=0, ignore_index=True)
    
    predictions.to_csv('submission.csv', index=False, float_format='%.16f')
    print(predictions.head())

