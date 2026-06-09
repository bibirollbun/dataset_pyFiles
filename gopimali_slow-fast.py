!pip install pytorchvideo


import os
import h5py
import math
import cv2
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorchvideo.models.hub import slowfast_r50
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


class SlowFastRGBD(nn.Module):
    def __init__(self, pretrained=True, alpha=4):
        super().__init__()
        self.base = slowfast_r50(pretrained=pretrained)
        self._update_stem(self.base.blocks[0].multipathway_blocks[0], 4)
        self._update_stem(self.base.blocks[0].multipathway_blocks[1], 4)
        in_features = self.base.blocks[-1].proj.in_features
        self.base.blocks[-1].proj = nn.Linear(in_features, 1)
        self.alpha = alpha

    def _update_stem(self, stem, in_channels):
        old_conv = stem.conv
        new_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias= False
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            if in_channels > 3:
                nn.init.kaiming_normal_(new_conv.weight[:, 3:])
        stem.conv = new_conv

    def pack_pathway_input(self, x):
        # x: [batch, 4, T, H, W]
        fast_pathway = x
        slow_pathway = x[:, :, ::self.alpha, :, :]
        return [slow_pathway, fast_pathway]

    def forward(self, x):
        x = self.pack_pathway_input(x)
        return self.base(x)  # Output: (batch, 1)


class H5DataLoader(Dataset):
    def __init__(self,path,df,mode='Train',transform=False):
        self.path = path
        self.df = df
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        file_name = int(self.df.loc[idx,'id'])
        feature_path = os.path.join(self.path, f'{file_name}.h5')
        with h5py.File(feature_path, 'r') as h5f:
            frames = h5f['frames'][:]
            depth_channel = h5f['depth_channel'][:]
            video_id = h5f.attrs['video_id']
            if self.mode == 'Train':
                target = h5f.attrs['target']
                return frames,depth_channel,video_id,target
            else:
                return frames,depth_channel,video_id


TRAIN_FILES = r'/kaggle/input/nexar-train-slow-fast/train_video_features/'
TEST_FILES = r'/kaggle/input/nexar-test-slow-fast/test_video_features/'

TRAIN_DF = pd.read_csv('/kaggle/input/nexar-collision-prediction/train.csv')
TEST_DF = pd.read_csv('/kaggle/input/nexar-collision-prediction/test.csv')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


output_dir = "output_weights"
os.makedirs(output_dir, exist_ok=True)


model = SlowFastRGBD(pretrained=True)

checkpoint_path = "/kaggle/input/slow_fast_model_2_epoch/pytorch/default/1/model_epoch_2.pth"
model.load_state_dict(torch.load(checkpoint_path))

model = nn.DataParallel(model)
model.to(DEVICE)
print('Done')
# loss_func = torch.nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)


# from sklearn.model_selection import train_test_split


# X_train, X_valid, y_train, y_valid = train_test_split(TRAIN_DF.drop('target',axis=1) ,TRAIN_DF['target'],test_size = 0.20,stratify=TRAIN_DF['target'] )


# X_train.reset_index(drop=True , inplace=True)
# X_valid.reset_index(drop=True , inplace=True)


# print(X_train.shape)
# X_train.head()


# print(X_valid.shape)
# X_valid.head()


TRAIN_DATASET = H5DataLoader(path=TRAIN_FILES, mode='Train', df=TRAIN_DF)
# VALID_DATASET = H5DataLoader(path=TRAIN_FILES, mode='Train', df=X_valid)
TEST_DATASET = H5DataLoader(path=TEST_FILES, mode='Test', df=TEST_DF)

TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size=16,num_workers = 3,pin_memory=True, shuffle=True)
# VALID_LOADER = DataLoader(VALID_DATASET, batch_size=16,num_workers = 2)
TEST_LOADER = DataLoader(TEST_DATASET, batch_size=16,num_workers = 3)


# model.train()
# EPOCHS = 10

# epoch_bar = tqdm(range(EPOCHS), desc="Training")

# for epc in epoch_bar:
#     total_loss = 0
#     train_batch_bar = tqdm(TRAIN_LOADER, leave=False, desc=f"Epoch {epc+1}")
#     for i , (frames,depth_channel,video_id,target) in enumerate(train_batch_bar):
#         frames = frames.permute(0, 4, 1, 2, 3)
#         depth_channel = depth_channel.unsqueeze(1)
        
#         inputs = torch.cat([frames, depth_channel], dim=1).float().to(DEVICE, non_blocking=True)
#         target = target.float().to(DEVICE,non_blocking=True)
        
#         optimizer.zero_grad()
#         output = model(inputs)
#         loss = loss_func(output.view(-1), target)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         train_batch_bar.set_postfix(loss=loss.item(), avg_loss=total_loss / (i + 1))
    
#     avg_loss = total_loss / len(TRAIN_LOADER)
#     print(f'AVG LOSS :- {avg_loss}')
#     epoch_bar.set_postfix(avg_loss=avg_loss)

#     if (epc + 1) % 2 == 0:
#         checkpoint_path = os.path.join(output_dir, f"model_epoch_{epc+1}.pth")
#         torch.save(model.module.state_dict(), checkpoint_path)
#         print(f"Checkpoint saved: {checkpoint_path}")


# checkpoint_path = os.path.join(output_dir, f"model_epoch_{epc+1}.pth")
# torch.save(model.module.state_dict(), checkpoint_path)
# print(f"Checkpoint saved: {checkpoint_path}")


sub = pd.read_csv('/kaggle/input/nexar-collision-prediction/sample_submission.csv')


model.eval()
test_preds = []
video_ids = []
# test_targets = []
with torch.no_grad():
    valid_batch_bar = tqdm(TEST_LOADER, leave=False)
    for i, (frames,depth_channel,video_id) in enumerate(valid_batch_bar):
        if i % 5 == 0:
            print(f'{i} Done !!!')
        frames = frames.permute(0, 4, 1, 2, 3).to(DEVICE)
        depth_channel = depth_channel.unsqueeze(1).to(DEVICE)
        inputs = torch.cat([frames, depth_channel], dim=1).float()

        output = model(inputs)
        probs = torch.sigmoid(output).squeeze().cpu().numpy()
        test_preds.extend(probs)
        video_ids.extend(video_id.cpu().numpy())


sub.target = test_preds
sub.to_csv('submission.csv',index=False)

