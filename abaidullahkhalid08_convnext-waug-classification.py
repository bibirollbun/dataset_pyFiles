!pip install -q iterative-stratification


import os
import cv2
import glob
import torch
import wandb
import random
import shutil
import pydicom
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
import torch.nn as nn
from typing import List
from torch import Tensor
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as v2
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


seed = 210
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print('Finish seeding with seed {}'.format(seed))

seed_everything(seed)
print('Training on device {}'.format(device))


train = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv")
train_coor = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv")
train_series = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv")
train_dummy = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv")
train_meta = pd.read_csv("/kaggle/input/meta-csv/meta.csv")


train_dummy = train_dummy.fillna("Normal/Mild")


train_coor = train_coor.merge(train_series[['study_id', 'series_id', 'series_description']], on=['study_id', 'series_id'])



class SpineDataset(Dataset):
    def __init__(self, coor, meta, condition, mode):
        if condition == 'scs':
            self.coor = coor.loc[coor.condition == "Spinal Canal Stenosis"]
        elif condition == 'nfn':
            self.coor = coor.loc[coor.condition.isin([
                'Left Neural Foraminal Narrowing',
                'Right Neural Foraminal Narrowing'
            ])]
        elif condition == 'ss':
            self.coor = coor.loc[coor.condition.isin([
                'Left Subarticular Stenosis',
                'Right Subarticular Stenosis'
            ])]


        g_coor = self.coor.groupby(['study_id']).count()
        if condition == 'scs':
            self.id = g_coor[g_coor.series_id == 5].reset_index().study_id.unique()
        else:
            self.id = g_coor[g_coor.series_id == 10].reset_index().study_id.unique()
        
        if condition == 'ss':
            self.resize = v2.Resize((256, 256))
        else:
            self.resize = v2.Resize((384, 384))

        self.condition = condition
        self.meta = meta
        self.mode = mode


    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        study_id = self.id[idx]
        
        if self.condition == 'scs':
            volume, label = self.volume_scs(study_id)
            if volume is None or volume.shape[0] == 0:
                print(study_id)
        elif self.condition == 'nfn':
            volume, label = self.volume_nfn(study_id)
        elif self.condition == 'ss':
            volume, label = self.volume_ss(study_id)

        return  volume, label

    def volume_scs(self, study_id):
        depth = 32
        meta = self.meta.loc[(self.meta.study_id == study_id) & (self.meta.series_description == 'Sagittal T2/STIR')]
        meta = meta.sort_values('ipp_x', ascending=True).reset_index(drop=True)
    
        img = [self.load_dicom(f"/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{study_id}/{row.series_id}/{row.instance_number}.dcm") for _, row in meta.iterrows()]
        
        coor = self.coor.loc[(self.coor.study_id == study_id) & (self.coor.series_description == 'Sagittal T2/STIR')]
        coor_dict = {}
        
        for _, row in coor.iterrows():
            series_id, instance_number = row.series_id, row.instance_number
            target_row = meta.loc[(meta.series_id == series_id) & (meta.instance_number == instance_number)]
            if not target_row.empty:
                idx = target_row.index[0]
                z = idx/depth if len(img) < depth else idx/len(img)
                x, y = row.x / img[idx].shape[1], row.y / img[idx].shape[0]
                coor_dict[row.level] = torch.tensor([x, y, z], dtype=torch.float32)
    
        # Ensure all levels have the same shape (Default: [0, 0, 0])
        coor_dict = {
            'L1/L2': coor_dict.get('L1/L2', torch.zeros(3)),  
            'L2/L3': coor_dict.get('L2/L3', torch.zeros(3)),
            'L3/L4': coor_dict.get('L3/L4', torch.zeros(3)),
            'L4/L5': coor_dict.get('L4/L5', torch.zeros(3)),
            'L5/S1': coor_dict.get('L5/S1', torch.zeros(3))
        }
    
        # Resize volume tensor
        volume = torch.cat([self.resize(torch.tensor(i)[None, ...]).to(torch.float32) for i in img]).contiguous()
        
        if volume.shape[0] < depth:
            volume = torch.cat([volume, torch.zeros(depth - volume.shape[0], volume.shape[1], volume.shape[2])])
        else:
            volume = torch.nn.functional.interpolate(volume[None, None, ...], (depth, volume.shape[1], volume.shape[2])).squeeze()
    
        return volume, coor_dict


    def volume_nfn(self, study_id):
        pass

    def volume_ss(self, study_id):
        pass

    def normalize(self, x): # in real world data dicom has extreme pixel value that is why we use normalization
        upper, lower = torch.quantile(x, torch.tensor([0.99, 0.01]))
        x = torch.clip(x, lower, upper)  # Remove Extreme outliers

        # x = (x - lower) / (upper - lower)

        # x = x - torch.min(x)
        # x = x / (torch.max(x)+1e-6)

        x = (x - lower) / (upper - lower + 1e-6)  # Min-max normalization
        return x


    def load_dicom(self, path):
        return pydicom.dcmread(path).pixel_array



class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            nn.Conv3d(in_features, out_features, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GroupNorm(num_groups=1, num_channels=out_features)
        )


class BottleNeckBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        expansion: int = 4,
        drop_p: float = .0,
        layer_scaler_init_value: float = 1e-6,
    ):
        super().__init__()
        expanded_features = out_features * expansion
        self.block = nn.Sequential(
            # narrow -> wide (with depth-wise and bigger kernel)
            nn.Conv3d(
                in_features, in_features, kernel_size=(2, 7, 7), padding='same', bias=False, groups=in_features
            ),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=in_features, num_channels=in_features),
            # wide -> wide
            nn.Conv3d(in_features, expanded_features, kernel_size=1),
            nn.GELU(),
            # wide -> narrow
            nn.Conv3d(expanded_features, out_features, kernel_size=1),
        )
        #self.layer_scaler = LayerScaler(layer_scaler_init_value, out_features)
        #self.drop_path = StochasticDepth(drop_p, mode="batch")


    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        #x = self.layer_scaler(x)
        #x = self.drop_path(x)
        x += res
        return x

class ConvNexStage(nn.Sequential):
    def __init__(
        self, in_features: int, out_features: int, depth: int, **kwargs
    ):
        super().__init__(
            # add the downsampler
            nn.Sequential(
                nn.GroupNorm(num_groups=in_features, num_channels=in_features),
                nn.Conv3d(in_features, out_features, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            ),
            *[
                BottleNeckBlock(out_features, out_features, **kwargs)
                for _ in range(depth)
            ],
        )

class ConvNextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_features: int,
        depths: List[int],
        widths: List[int],
        drop_p: float = .0,
    ):
        super().__init__()
        self.stem = ConvNextStem(in_channels, stem_features)

        in_out_widths = list(zip(widths, widths[1:]))
        # create drop paths probabilities (one for each stage)
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))]

        self.stages = nn.ModuleList(
            [
                ConvNexStage(stem_features, widths[0], depths[0], drop_p=drop_probs[0]),
                *[
                    ConvNexStage(in_features, out_features, depth, drop_p=drop_p)
                    for (in_features, out_features), depth, drop_p in zip(
                        in_out_widths, depths[1:], drop_probs[1:]
                    )
                ],
            ]
        )


    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


class ConvNextSCSDepthDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels=1, stem_features=64, depths=[3,3,9,3], widths=[128, 256, 512, 1024])
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)),
                                    nn.Flatten(1),
                                    nn.LayerNorm(1024),
                                     )
        self.l1 = nn.Linear(1024, 3)
        self.l2 = nn.Linear(1024, 3)
        self.l3 = nn.Linear(1024, 3)
        self.l4 = nn.Linear(1024, 3)
        self.l5 = nn.Linear(1024, 3)
    def forward(self, x, label=None):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.flatten(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        l3 = self.l3(x)
        l4 = self.l4(x)
        l5 = self.l5(x)
        return {'L1/L2': l1, 'L2/L3': l2, 'L3/L4': l3, 'L4/L5': l4, 'L5/S1': l5}


model = ConvNextSCSDepthDetect().to(device)


class SCSLoss(nn.Module):
    def __init__(self):
        super(SCSLoss, self).__init__()

    def forward(self, outputs, targets):
        loss = 0
        for level in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']:
            # Debugging: Print shapes
            # print(f"ðŸ”¹ {level} Output Shape: {outputs[level].shape}, Target Shape: {targets[level].shape}")

            # # Ensure both tensors have the same shape before L1 loss
            # if outputs[level].shape != targets[level].shape:
            #     targets[level] = targets[level].expand_as(outputs[level])  # Expand target if needed

            _loss = nn.functional.l1_loss(outputs[level], targets[level])
            loss += _loss
        return loss / 5



from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

label = train.columns[1:]
train_dummy['fold'] = -1  # Initialize before assigning
kfold = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_idx, valid_idx) in enumerate(kfold.split(train_dummy, train_dummy[label])):
    train_dummy.loc[valid_idx, 'fold'] = i
train_series = train_series.merge(train_dummy[['study_id', 'fold']], on='study_id')
train_coor = train_coor.merge(train_dummy[['study_id', 'fold']], on='study_id')
train = train.merge(train_dummy[['study_id', 'fold']], on='study_id')
train_meta = train_meta.merge(train_dummy[['study_id', 'fold']], on='study_id')



# def calculate_accuracy(outputs, targets):
#     """Computes accuracy given model outputs and true labels."""
    
#     total_correct = 0
#     total_samples = 0
    
#     for level in outputs.keys():
#         _, preds = torch.max(outputs[level], dim=1)  # Get predicted class indices
        
#         # Ensure targets have correct shape
#         target_labels = targets[level]  # Check if this is one-hot
#         if target_labels.ndim > 1 and target_labels.shape[1] > 1:
#             target_labels = target_labels.argmax(dim=1)  # Convert one-hot to class index
#         correct = (preds == target_labels).sum().item()
#         total_correct += correct
#         total_samples += target_labels.size(0)

#     return total_correct, total_samples



def calculate_mse_score(outputs, targets):
    """Computes Mean Squared Error (MSE) for model predictions."""
    total_mse = 0
    total_samples = 0

    for level in outputs.keys():
        mse = nn.functional.mse_loss(outputs[level], targets[level], reduction='sum')  # Sum over batch
        total_mse += mse.item()
        total_samples += targets[level].numel()  # Total elements

    return total_mse / total_samples  # Average MSE per element



epochs = 10

for i in range(1):
    train_dataset = SpineDataset(train_coor.loc[train_coor.fold!=i], train_meta.loc[train_meta.fold!=i], 'scs', 'train')
    valid_dataset = SpineDataset(train_coor.loc[train_coor.fold==i], train_meta.loc[train_meta.fold==i], 'scs', 'valid')

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=False)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=False)

    model.to(device)  # Move model to GPU
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=2 * len(train_loader), num_training_steps=epochs * len(train_loader), num_cycles=0.5
    )
    criterion = SCSLoss()

    # train_losses, val_losses, train_accs, val_accs = [], [], [], []

    train_losses, val_losses, train_mses, val_mses = [], [], [], []

    for epoch in range(epochs):
        print(f"\nðŸš€ Epoch {epoch+1}/{epochs}")

        # Training phase
        model.train()
        # running_loss, correct, total = 0.0, 0, 0
        running_loss, total_mse = 0.0, 0.0
        progress_bar = tqdm(train_loader, desc="Training Progress", leave=False)

        for volume, batch in progress_bar:
            volume = volume.to(device)  # Move input tensors to GPU
            # Debugging: Check available keys in batch
            
            # Ensure all expected keys are present
            expected_levels = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
            batch = {key: value.to(device) for key, value in batch.items() if key in expected_levels}

            optimizer.zero_grad()
            outputs = model(volume)  
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()

            scheduler.step()  # âœ… Moved outside batch loop

            # running_loss += loss.item()
            # batch_correct, batch_total = calculate_accuracy(outputs, batch)
            # correct += batch_correct
            # total += batch_total

            running_loss += loss.item()
            batch_mse = calculate_mse_score(outputs, batch)
            total_mse += batch_mse

            # âœ… Show loss in tqdm bar
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            # âœ… Free memory
            del volume, batch, outputs, loss
            torch.cuda.empty_cache()


        # epoch_train_loss = running_loss / len(train_loader)
        # epoch_train_acc = correct / total
        # train_losses.append(epoch_train_loss)
        # train_accs.append(epoch_train_acc)
        # print(f"ðŸ”¥ Training Loss: {epoch_train_loss:.4f} | Accuracy: {epoch_train_acc:.4%}")

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_mse = total_mse / len(train_loader)
        train_losses.append(epoch_train_loss)
        train_mses.append(epoch_train_mse)
        print(f"ðŸ”¥ Training Loss: {epoch_train_loss:.4f} | MSE Score: {epoch_train_mse:.4f}")

        # Validation phase
        model.eval()
        val_running_loss, val_total_mse = 0.0, 0.0
        with torch.no_grad():
            for volume, batch in tqdm(valid_loader, desc="Validation Progress"):
                volume = volume.to(device)
                batch = {key: value.to(device) for key, value in batch.items()}

                outputs = model(volume)
                loss = criterion(outputs, batch)
                val_running_loss += loss.item()

                # batch_correct, batch_total = calculate_accuracy(outputs, batch)
                # val_correct += batch_correct
                # val_total += batch_total

                batch_mse = calculate_mse_score(outputs, batch)
                val_total_mse += batch_mse

                del volume, batch, outputs, loss
                torch.cuda.empty_cache()

        # epoch_val_loss = val_running_loss / len(valid_loader)
        # epoch_val_acc = val_correct / val_total
        # val_losses.append(epoch_val_loss)
        # val_accs.append(epoch_val_acc)
        # print(f"âœ… Validation Loss: {epoch_val_loss:.4f} | Accuracy: {epoch_val_acc:.4%}")

        epoch_val_loss = val_running_loss / len(valid_loader)
        epoch_val_mse = val_total_mse / len(valid_loader)
        val_losses.append(epoch_val_loss)
        val_mses.append(epoch_val_mse)
        print(f"âœ… Validation Loss: {epoch_val_loss:.4f} | MSE Score: {epoch_val_mse:.4f}")



import gc

# âœ… After each epoch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
gc.collect()  # Optional, forces Python garbage collection

torch.cuda.empty_cache()







