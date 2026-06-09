import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import librosa
import os
import torchaudio
import torch
from IPython.display import Audio, display
import timm
from torch.utils.data import Dataset,DataLoader,random_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


!nvidia-smi


from transformers import AutoFeatureExtractor, AutoModelForAudioClassification,AutoConfig


totol_target_class = 206
config = AutoConfig.from_pretrained("Simon-Kotchou/ssast-small-patch-audioset-16-16")
feature_extractor = AutoFeatureExtractor.from_pretrained("Simon-Kotchou/ssast-small-patch-audioset-16-16")
model = AutoModelForAudioClassification.from_pretrained("Simon-Kotchou/ssast-small-patch-audioset-16-16")


print(model.classifier)


model.classifier = nn.Sequential(
    nn.LayerNorm(384,eps=1e-12),
    nn.Linear(384,206)
    
)
config.num_labels = 206


print(model.classifier)


model = torch.nn.DataParallel(model)  # use all available GPUs
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()


def preprocess_function(waveform,max_duration=5):
    
    inputs = feature_extractor(
        waveform,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return inputs



from torch.utils.data import Dataset
import os
import librosa

class CustomDataset(Dataset):
    def __init__(self, audio_folder, preprocess_function, class_names=None, class_to_idx=None):
        super(CustomDataset, self).__init__()
        self.audio_folder = audio_folder
        self.clipped_audio = []
        self.audio_labels = []
        self.preprocess_function = preprocess_function

        if class_names is None:
            self.class_names = sorted(os.listdir(self.audio_folder))
        else:
            self.class_names = class_names

        if class_to_idx is None:
            self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        else:
            self.class_to_idx = class_to_idx

        for class_name in self.class_names:
            class_path = os.path.join(self.audio_folder, class_name)
            for audio_file in os.listdir(class_path):
                self.clipped_audio.append(os.path.join(class_path, audio_file))
                self.audio_labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        audio_clip = self.clipped_audio[idx]
        label = self.audio_labels[idx]
        waveform, sample_rate = librosa.load(audio_clip, sr=None)
        model_input = self.preprocess_function(waveform) # returns dictionary
        model_input = {k: torch.tensor(v).squeeze(0) if isinstance(v, (list, np.ndarray)) else v.squeeze(0) for k, v in model_input.items()}
        
        return model_input,label



audio_folder = "/kaggle/input/preprocessed-data/kaggle/working/output"
dataset = CustomDataset(audio_folder,preprocess_function)
class_name =   dataset.class_names
class_to_idx = dataset.class_to_idx

idx_to_class = {v:k for k,v in class_to_idx.items()}


total_size = len(dataset)
val_size = int(0.1*total_size)
train_size = total_size- val_size

train_dataset,valid_dataset = random_split(dataset,[train_size,val_size])

train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
valid_dataloader = DataLoader(valid_dataset,batch_size=32,shuffle=False)


for i,(audio,label) in enumerate(train_dataloader):
    print(f"audio shape : {audio['input_values'].shape}")
    break
    


Epochs = 25
train_loss_list = []
valid_loss_list = []
best_loss = 1_000_000
best_model = None

for epoch in range(Epochs):
    model.train()
    training_loss = 0.0
    validation_loss = 0.0

    # Training loop
    for i, (audio, label) in enumerate(train_dataloader):
        audio = {k: v.to(device) for k, v in audio.items()}  # move each tensor to device
        label = label.to(device)
    
        optimizer.zero_grad()
        output = model(**audio)
        loss = loss_fn(output.logits, label)  
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    # Validation loop
    model.eval()
    with torch.no_grad():
        for j, (vaudio, vlabel) in enumerate(valid_dataloader):
            vaudio = {k: v.to(device) for k, v in vaudio.items()}
            vlabel = vlabel.to(device)
    
            voutput = model(**vaudio)
            vloss = loss_fn(voutput.logits, vlabel)
            validation_loss += vloss.item()

    
    print(f"Epoch [{epoch+1}/{Epochs}] - Train Loss: {training_loss:.4f}, Val Loss: {validation_loss:.4f}")
    train_loss_list.append(training_loss/(i+1))
    valid_loss_list.append(validation_loss/(j+1))

    if validation_loss < best_loss:
        best_loss = validation_loss
        best_model = model
        torch.save(model.state_dict(), "best_model.pth")

    
    


# Assuming best_model is wrapped in DataParallel
# And you want to save Hugging Face-compatible model, config, and feature extractor

save_dir = "ssast-206-final"

# 1. Save the actual model (not the DataParallel wrapper)
best_model.module.save_pretrained(save_dir)

# 2. Save the feature extractor (if you're using AutoFeatureExtractor or similar)
feature_extractor.save_pretrained(save_dir)

# 3. Save the model config (optional if not already included)
config.save_pretrained(save_dir)



import shutil
shutil.make_archive(save_dir, 'zip', save_dir)


import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plt.plot(train_loss_list, label="Train Loss")
plt.plot(valid_loss_list, label="Validation Loss")
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()




