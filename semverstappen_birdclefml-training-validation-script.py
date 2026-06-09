import os 
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import accuracy_score, classification_report


#Parameters
AUDIO_DIR = '/kaggle/input/birdclef-2025/train_audio/' #locate the audio files
SR        = 32000 #Sampling rate 32K hz
CHUNK_LEN = 5  #5 seconds
N_MELS    = 128 #the amount of mel frequency bands used to convert audios to spectrogram
BATCH_SIZE = 32 #Choose batch size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Define when to use gpu or cpu

###Helper functions
#Split audio in 5 seconds function
def split_audio(audio, sr=SR, chunk_length=CHUNK_LEN):
    samples_per_chunk = chunk_length * sr
    chunks = []
    for i in range(0, len(audio), samples_per_chunk):
        chunk = audio[i:i + samples_per_chunk]
        if len(chunk) < samples_per_chunk:
            chunk = np.pad(chunk,
                           (0, samples_per_chunk - len(chunk)),
                           mode='constant')
        chunks.append(chunk)
    return chunks

#Converts audio chunks into mel spectrograms
def to_mel_spectrogram(audio_chunk, sr=SR, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(y=audio_chunk,
                                         sr=sr,
                                         n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

#Normalizes all spectrograms created
def normalize_mel(mel_db):
    mel_db -= mel_db.min()
    max_val = mel_db.max()
    if max_val > 0:
        mel_db /= max_val
    else:
        mel_db[:] = 0.0
    return mel_db


#Stores filepaths and labels inside BirdclefDataset object
class BirdclefDataset(Dataset):
    def __init__(self, filepaths, labels):
        self.filepaths = filepaths
        self.labels = labels
        
#Returns the length of the amount of samples
    def __len__(self):
        return len(self.filepaths)

##Audio processing
    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

#Loads audio and splits with helper function
        audio, _ = librosa.load(filepath, sr=SR)
        chunks = split_audio(audio, sr=SR)

        if len(chunks) == 0:
            #If the audio file is shorter than CHUNK_LEN, pad it with 0s
            chunk = np.pad(audio, (0, SR*CHUNK_LEN - len(audio)), mode='constant')
        else:
            chunk = random.choice(chunks)
        #Convert chunks to actual spectrograms
        mel = to_mel_spectrogram(chunk)
        mel = normalize_mel(mel)
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        return mel, label


#CNN model layers
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, num_classes)

    #Define how input moves through layers
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


#Create 2 empty lists
ogg_paths = []
labels = []

#This part walks through all audio files in the directory and extracts the full paths and associated labels
for root, _, files in os.walk(AUDIO_DIR):
    for fname in files:
        if fname.lower().endswith('.ogg'):
            ogg_paths.append(os.path.join(root, fname))
            label = os.path.basename(root)  #foldername = bird species
            labels.append(label)

#Map the bird labels to integers
unique_labels = sorted(list(set(labels)))
label2idx = {label: idx for idx, label in enumerate(unique_labels)}
idx2label = {idx: label for label, idx in label2idx.items()}
labels_idx = [label2idx[label] for label in labels]

#Print label order (later used for the actual submission script)
print("✅ unique_labels (label → index mapping):")
for i, label in enumerate(unique_labels):
    print(f"{i}: {label}")

full_dataset = BirdclefDataset(ogg_paths, labels_idx)

#80% training and 20% validation set splitting
train_paths, val_paths, train_labels, val_labels = train_test_split(
    ogg_paths,
    labels_idx,
    test_size=0.2,
    stratify=labels_idx,
    random_state=42
)

#Create the datasets from created split
train_dataset = BirdclefDataset(train_paths, train_labels)
val_dataset   = BirdclefDataset(val_paths, val_labels)

###This is the code for non-stratified splitting
#train_size = int(0.8 * len(full_dataset))
#val_size   = len(full_dataset) - train_size
#train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

#These data loaders below will be used later during training to feed batches into the model and to evaluate the performance during training
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


#Training the model
model = CNNModel(num_classes=len(unique_labels)).to(device) #Initiliazes CNN model to output num_classes
optimizer = optim.Adam(model.parameters(), lr=1e-3) #Adam optimizer is used for updating the model weights
criterion = nn.CrossEntropyLoss() #Loss function used for classification (cross entropy)

for epoch in range(10):  #Epoch number = 10, with 10 epochs this takes like 4 hours
    model.train() #Sets model in training mode
    train_loss = 0.0

    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)

    model.eval() #Sets the model in evaluation mode
    val_loss = 0.0

    with torch.no_grad(): #this makes sure the model validation is done without updating the weights
        for X_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)

    print(f"Epoch {epoch+1}: Train loss {train_loss/len(train_dataset):.4f} | Val loss {val_loss/len(val_dataset):.4f}") #Reports average training and validation loss for each epoch

#Save the model
torch.save(model.state_dict(), '/kaggle/working/birdclef_cnn.pth')
print("✅ Model saved to /kaggle/working/birdclef_cnn.pth") #To make sure the model is saved


#Define the CNN model architecture if not already defined
try:
    CNNModel
except NameError:
    class CNNModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.fc = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.cnn(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

#Define paths
SAVE_PATH = '/kaggle/working/birdclef_cnn.pth' #Newly trained model
MODEL_PATH = '/kaggle/input/birdclef_stratifiedcnn/pytorch/stratified/2/birdclef_cnnstratified.pth' #Pretrained option
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Initialize and load model, this code will favor using the newly trained model but if not detected it will use the pretrained option
model = CNNModel(num_classes=len(unique_labels)).to(device)

if os.path.exists(SAVE_PATH):
    print(f"✅ Using newly trained model at {SAVE_PATH}")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
elif os.path.exists(MODEL_PATH):
    print(f"⚠️ Newly trained model not found. Using pretrained model at {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    raise FileNotFoundError("❌ No model file found at either SAVE_PATH or MODEL_PATH.")

from sklearn.metrics import accuracy_score, classification_report

model.eval()
val_preds = []
val_true = []

for path, label_idx in tqdm(zip(val_paths, val_labels), desc="Predicting 20% validation set", total=len(val_paths)):
    #Load audio
    audio, _ = librosa.load(path, sr=SR)
    chunks = split_audio(audio)

    chunk_probs = []

    for i, chunk in enumerate(chunks):
        if len(chunk) < CHUNK_LEN * SR:
            chunk = np.pad(chunk, (0, CHUNK_LEN * SR - len(chunk)))

        mel = to_mel_spectrogram(chunk)
        mel = normalize_mel(mel)  #Normalization
        mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(mel_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
            chunk_probs.append(probs)

    #Aggregate across chunks (e.g. mean of softmax probabilities)
    avg_probs = np.mean(chunk_probs, axis=0)
    pred_idx = int(np.argmax(avg_probs))

    val_preds.append(pred_idx)
    val_true.append(label_idx)

#Evaluation metrics
acc = accuracy_score(val_true, val_preds)
print(f"\n✅ Validation accuracy: {acc:.4f}\n")

labels_present = sorted(set(val_true) | set(val_preds))
print("Classification Report:")
print(classification_report(
    val_true,
    val_preds,
    labels=labels_present,
    target_names=[unique_labels[i] for i in labels_present],
    zero_division=0
))

##Additional output
from collections import Counter

train_dist = Counter(train_labels)
val_dist = Counter(val_labels)

print(f"Train classes: {len(train_dist)}")
print(f"Val classes:   {len(val_dist)}")

#Visualization
plt.figure(figsize=(10, 4))
plt.hist(train_dist.values(), bins=50, alpha=0.6, label='Train')
plt.hist(val_dist.values(), bins=50, alpha=0.6, label='Validation')
plt.legend()
plt.title("")
plt.xlabel("Number of samples per class")
plt.ylabel("Number of classes")
plt.show()


#Visualization
plt.figure(figsize=(10, 4))
plt.hist(train_dist.values(), bins=50, alpha=0.6, label='Train')
plt.hist(val_dist.values(), bins=50, alpha=0.6, label='Validation')
plt.legend()
plt.title("")
plt.xlabel("Number of samples per class")
plt.ylabel("Number of classes")
plt.show()

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

##Visualizing one spectrogram for poster
#Construct the path to a known file
label_id = "1194042"
file_name = "CSA18783.ogg"
sample_path = os.path.join(AUDIO_DIR, label_id, file_name)

#Load and process
y, sr = librosa.load(sample_path, sr=SR)
chunks = split_audio(y)

#Use the first chunk for visualization
chunk = chunks[0]
mel = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=N_MELS)
mel_db = librosa.power_to_db(mel, ref=np.max)

#Plot without normalization (to preserve dB scale)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, sr=SR, x_axis='time', y_axis='mel', cmap='magma')
plt.colorbar(format="%+2.0f dB")
#plt.title(f"Mel spectrogram - label {label_id} ({file_name})")
plt.title("")
plt.tight_layout()
plt.show()

######
import pandas as pd

# You already calculated this earlier
val_acc = acc * 100  # Convert to %
leaderboard_acc = 78.5  # Example value – replace with your actual leaderboard result

# Create a simple accuracy comparison table
accuracy_df = pd.DataFrame({
    'Dataset': ['Validation (20%)', 'Leaderboard'],
    'Accuracy': [acc, 0.597]  # Replace acc with your actual validation score if needed
})

# Display nicely
display(accuracy_df.style.set_properties(**{
    'text-align': 'center',
    'font-size': '14px'
}))

