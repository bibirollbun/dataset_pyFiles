import os
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from collections import OrderedDict
import cv2

class CFG:
    # path
    train_audio = '/kaggle/input/birdclef-2025/train_audio'
    train_soundscapes = '/kaggle/input/birdclef-2025/train_soundscapes'
    test_soundscapes = '/kaggle/input/birdclef-2025/train_soundscapes/'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    model = '/kaggle/input/psuedo_model_01/pytorch/default/1/best_model.pth'

    # audio
    FS = 32000
    WINDOW_SIZE = 5  # 오디오 분할 길이(초)
    
    # MEL_SPEC
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    TARGET_SHAPE = (224, 224)

    # inference
    threshold = 0.5

    # model
    model_name = 'efficientnet_b0' # resnet도 가능
    in_channels = 1 # submission test를 위해 임의로 1로 했다.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # another
    seed = 42

cfg = CFG()
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg
        
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False,
            in_chans=cfg.in_channels,
            drop_rate=0.2,    
            drop_path_rate=0.1
        )

        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        
        if isinstance(features, dict):
            features = features['features']
            
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        return logits


class_labels = sorted(os.listdir(cfg.train_audio))
num_classes = len(class_labels)
model = BirdCLEFModel(cfg, num_classes)
model = model.to(cfg.device)

print("Loading best model...")
ckpt = torch.load(cfg.model, map_location=cfg.device, weights_only=True)
state = ckpt.get('model_state_dict', ckpt)
model.load_state_dict(state)
model.eval()
print("finished!")


test_soundscapes = [os.path.join(cfg.test_soundscapes, afile) 
                    for afile in sorted(os.listdir(cfg.test_soundscapes))]
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

        # Audio -> Mel-spec
        mel_spec = librosa.feature.melspectrogram(
            y=chunk,
            sr=rate,
            n_fft=cfg.N_FFT,
            hop_length=cfg.HOP_LENGTH,
            n_mels=cfg.N_MELS,
            fmin=cfg.FMIN,
            fmax=cfg.FMAX
        )

        # Mel-spec -> db scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        mel_spec_resized = cv2.resize(mel_spec_db_norm, cfg.TARGET_SHAPE)

        # db scaled Mel-spec -> input tensor
        input_tensor = torch.tensor(mel_spec_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(cfg.device)

        # Make prediction
        with torch.no_grad():
            logits = model(input_tensor)
            scores = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Append to predictions as new row
        new_row = pd.DataFrame([[row_id] + list(scores)], columns=['row_id'] + class_labels)
        predictions = pd.concat([predictions, new_row], axis=0, ignore_index=True)
        
# Save prediction as csv
predictions.to_csv('submission.csv', index=False)
predictions.head()




