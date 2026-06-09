# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import gc
import sys
import cv2
import math
import numpy as np
import pandas as pd
from glob import glob
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import librosa
from scipy import signal as sci_signal
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torchvision.models import efficientnet

from torch.utils.data import DataLoader

import albumentations as albu



class config:
    SEED = 2024  # random seed
    DEVICE = 'cuda'  # device to be used
    MIXED_PRECISION = False  # whether to use mixed-16 precision
    OUTPUT_DIR = '/kaggle/working/'  # output folder
    
    # == data config ==
    DATA_ROOT = '/kaggle/input/birdclef-2024'  # root folder
    PREPROCESSED_DATA_ROOT = '/kaggle/input/birdclef24-spectrograms-via-cupy'
    LOAD_DATA = True  # whether to load data from pre-processed dataset
    FS = 32000  # sample rate
    N_FFT = 1095  # n FFT of Spec.
    WIN_SIZE = 412  # WIN_SIZE of Spec.
    WIN_LAP = 100  # overlap of Spec.
    LR_MAX = 3e-4  # Maximum learning rate
    N_STEPS = 20 * len(train_loader)  # Total steps = epochs * batches per epoch
    N_CLASSES = len(label_list)  # Number of classes in your dataset
    MIN_FREQ = 40  # min frequency
    MAX_FREQ = 15000  # max frequency
    USE_XYMASKING = True 


# labels
label_list = sorted(os.listdir(os.path.join(config.DATA_ROOT, 'train_audio')))
label_id_list = list(range(len(label_list)))
label2id = dict(zip(label_list, label_id_list))
id2label = dict(zip(label_id_list, label_list))


metadata_df = pd.read_csv(f'{config.DATA_ROOT}/train_metadata.csv')
metadata_df.head()


def oog2spec_via_cupy(audio_data):
    
    import cupy as cp
    from cupyx.scipy import signal as cupy_signal
    
    audio_data = cp.array(audio_data)
    
    # handles NaNs
    mean_signal = cp.nanmean(audio_data)
    audio_data = cp.nan_to_num(audio_data, nan=mean_signal) if cp.isnan(audio_data).mean() < 1 else cp.zeros_like(audio_data)
    
    # to spec.
    frequencies, times, spec_data = cupy_signal.spectrogram(
        audio_data, 
        fs=config.FS, 
        nfft=config.N_FFT, 
        nperseg=config.WIN_SIZE, 
        noverlap=config.WIN_LAP, 
        window='hann'
    )
    
    # Filter frequency range
    valid_freq = (frequencies >= config.MIN_FREQ) & (frequencies <= config.MAX_FREQ)
    spec_data = spec_data[valid_freq, :]
    
    # Log
    spec_data = cp.log10(spec_data + 1e-20)
    
    # min/max normalize
    spec_data = spec_data - spec_data.min()
    spec_data = spec_data / spec_data.max()
    
    return spec_data.get()


if config.LOAD_DATA:
    print('load from file')
    all_bird_data = np.load(f'{config.PREPROCESSED_DATA_ROOT}/spec_center_5sec_256_256.npy', allow_pickle=True).item()
else:
    all_bird_data = dict()
    for i, row_metadata in tqdm(train_df.iterrows()):

        # load ogg
        audio_data, _ = librosa.load(row_metadata.filepath, sr=config.FS)

        # crop
        n_copy = math.ceil(5 * config.FS / len(audio_data))
        if n_copy > 1: audio_data = np.concatenate([audio_data]*n_copy)

        start_idx = int(len(audio_data) / 2 - 2.5 * config.FS)
        end_idx = int(start_idx + 5.0 * config.FS)
        input_audio = audio_data[start_idx:end_idx]

        # ogg to spec.
        input_spec = oog2spec_via_cupy(input_audio)
        
        input_spec = cv2.resize(input_spec, (256, 256), interpolation=cv2.INTER_AREA)

        all_bird_data[row_metadata.samplename] = input_spec.astype(np.float32)

    # save to file
    np.save(os.path.join(config.OUTPUT_DIR, f'spec_center_5sec_256_256.npy'), all_bird_data)


def get_transforms(_type):
    
    if _type == 'train':
        return albu.Compose([
            albu.HorizontalFlip(0.5),
            albu.XYMasking(
                p=0.3,
                num_masks_x=(1, 3),
                num_masks_y=(1, 3),
                mask_x_length=(1, 10),
                mask_y_length=(1, 20),
            ) if config.USE_XYMASKING else albu.NoOp()
        ])
    elif _type == 'valid':
        return albu.Compose([])
def show_batch(ds, row=3, col=3):
    fig = plt.figure(figsize=(10, 10))
    img_index = np.random.randint(0, len(ds)-1, row*col)
    
    for i in range(len(img_index)):
        img, label = dummy_dataset[img_index[i]]
        
        if isinstance(img, torch.Tensor):
            img = img.detach().numpy()
        
        ax = fig.add_subplot(row, col, i + 1, xticks=[], yticks=[])
        ax.imshow(img, cmap='jet')
        ax.set_title(f'ID: {img_index[i]}; Target: {label}')
    
    plt.tight_layout()
    plt.show()


class BirdDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        metadata,
        augmentation=None,
        mode='train'
    ):
        super().__init__()
        self.metadata = metadata
        self.augmentation = augmentation
        self.mode = mode
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        
        row_metadata = self.metadata.iloc[index]
        
        # Load spec. data (image)
        input_spec = all_bird_data[row_metadata.samplename]
        
        # Ensure input_spec is a numpy array or tensor
        input_spec = np.array(input_spec)

        # Augmentation
        if self.augmentation is not None:
            input_spec = self.augmentation(image=input_spec)['image']
        resize = albu.Resize(224, 224)
        input_spec = resize(image=input_spec)['image']
        # Check if input_spec is 2D (grayscale) or 3D (RGB)
        if input_spec.ndim == 2:  # Grayscale image (Height x Width)
            input_spec = np.expand_dims(input_spec, axis=-1)  # Convert to (Height, Width, 1)

        # Now, ensure it's a 3-channel image (if needed)
        if input_spec.shape[-1] == 1:  # If single channel (grayscale)
            input_spec = np.repeat(input_spec, 3, axis=-1)  # Convert to 3 channels (RGB)
        
        # Convert to tensor
        input_spec = torch.tensor(input_spec, dtype=torch.float32)

        # Convert to the format (C, H, W)
        input_spec = input_spec.permute(2, 0, 1)  # Change to (3, H, W)

        # Target label
        target = row_metadata.target
        
        return input_spec, torch.tensor(target, dtype=torch.long)



train_metadata_split, valid_metadata_split = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df['primary_label']
)
train_dataset = BirdDataset(metadata=train_metadata_split, augmentation=get_transforms('train'), mode='train')
valid_dataset = BirdDataset(metadata=valid_metadata_split, augmentation=get_transforms('valid'), mode='valid')

# Create DataLoaders for both train and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)


for inputs, labels in train_loader:
    print(inputs.size())  # Should now print: torch.Size([32, 1, 256, 256])
    break



ACC = torchmetrics.Accuracy(task='multiclass', num_classes=CONFIG.N_CLASSES).cuda()
ROC_AUC = torchmetrics.AUROC(task='multiclass', num_classes=CONFIG.N_CLASSES).cuda()


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        
        # Load pre-trained EfficientNet
        self.efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # Modify the final fully connected layer for your number of classes
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        return self.efficientnet(x)
model = EfficientNetModel(num_classes=CONFIG.N_CLASSES)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)  # Wrap model in DataParallel for multi-GPU usage

model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define the OneCycleLR scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer,
    max_lr=CONFIG.LR_MAX,
    total_steps=CONFIG.N_STEPS,
    pct_start=0.10,  # 10% of the total steps for increasing LR
    anneal_strategy='cos',  # Cosine annealing
    div_factor=1e3,  # Initial LR is LR_MAX/div_factor
    final_div_factor=1e4,  # Final LR is LR_MAX/final_div_factor
)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()  # Set model to training mode
    train_loss = 0
    ACC.reset()  # Reset metrics before each epoch
    ROC_AUC.reset()

    pbar = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}')
    
    for step, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear gradients
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        # Update the learning rate
        scheduler.step()
        
        # Track loss
        train_loss += loss.item()

        # Convert logits to probabilities
        probs = torch.softmax(outputs, dim=1)  

        # Update Accuracy and ROC AUC
        ACC.update(probs, labels)
        ROC_AUC.update(probs, labels)

    # Compute final ACC and ROC AUC for epoch
    acc_value = ACC.compute().item()
    roc_auc_value = ROC_AUC.compute().item()

    # Update progress bar and print metrics
    pbar.set_postfix(loss=train_loss / len(train_loader), acc=acc_value, roc_auc=roc_auc_value)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}, ACC: {acc_value:.4f}, ROC AUC: {roc_auc_value:.4f}")
    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")



# Assuming `model` is your trained model
torch.save(model.state_dict(), '/kaggle/working/my_model.pth')


