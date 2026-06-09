import torch.nn as nn # Import the neural network module from PyTorch
import timm # Import the timm library


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.backbone = timm.create_model(cfg.model_name, pretrained=True, in_chans=cfg.in_channels, num_classes=0)
        self.lstm = nn.LSTM(self.backbone.num_features, 512, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(512 * 2, 1)
        self.classifier = nn.Linear(512 * 2, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = features.unsqueeze(1)
        lstm_out, _ = self.lstm(features)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        return self.classifier(context)


import torch
# Load the entire model (architecture + weights)
model = torch.load('/kaggle/input/birdclef/pytorch/default/1/full_model.pth', weights_only=False)

# Set the model to evaluation mode
model.eval()



import os
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Set seed
np.random.seed(42)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Class labels from train audio
class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))

# List of test soundscapes
test_soundscape_path = '/kaggle/input/birdclef-2025/test_soundscapes/'
test_soundscapes = [os.path.join(test_soundscape_path, afile) 
                    for afile in sorted(os.listdir(test_soundscape_path)) if afile.endswith('.ogg')]

# Prepare predictions DataFrame
predictions = pd.DataFrame(columns=['row_id'] + class_labels)

# Process each test soundscape
for soundscape in test_soundscapes:
    # Load audio
    sig, rate = librosa.load(path=soundscape, sr=None)

    # Split into 5-second chunks
    chunks = []
    for i in range(0, len(sig), rate * 5):
        chunk = sig[i:i + rate * 5]
        if len(chunk) < rate * 5:  # pad if less than 5 sec
            pad_len = rate * 5 - len(chunk)
            chunk = np.pad(chunk, (0, pad_len))
        chunks.append(chunk)

    # Run model inference on each chunk
    for i, chunk in enumerate(chunks):
        row_id = os.path.basename(soundscape).split('.')[0] + f'_{i * 5 + 5}'

        # Convert chunk to tensor
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # shape: [1, 1, samples]

        with torch.no_grad():
            output = model(chunk_tensor)  # output shape: [1, num_classes]
            probs = F.sigmoid(output).cpu().numpy().flatten()  # sigmoid if multi-label

        # Add prediction to dataframe
        new_row = pd.DataFrame([[row_id] + list(probs)], columns=['row_id'] + class_labels)
        predictions = pd.concat([predictions, new_row], axis=0, ignore_index=True)

# Save submission
predictions.to_csv('submission.csv', index=False)
predictions.head()  

