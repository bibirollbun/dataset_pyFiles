import os
import sys
sys.path.append('/kaggle/input/timm-0-6-9/pytorch-image-models-master')
import glob
import numpy as np
import pandas as pd
import random
import math
import gc
import cv2
from tqdm import tqdm
import time
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef


CFG = {
    'seed': 42,
    'model': 'resnet18',
    'img_size': 128,
    'epochs': 2,
    'train_bs': 32, 
    'valid_bs': 64,
    'lr': 1e-3, 
    'weight_decay': 1e-6,
    'num_workers': 1
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def expand_contact_id(df):
#     """
#     Splits out contact_id into seperate columns.
#     """
#     df["game_play"] = df["contact_id"].str[:12]
#     df["step"] = df["contact_id"].str.split("_").str[-3].astype("int")
#     df["nfl_player_id_1"] = df["contact_id"].str.split("_").str[-2]
#     df["nfl_player_id_2"] = df["contact_id"].str.split("_").str[-1]
#     return df

# train_labels = expand_contact_id(pd.read_csv("/kaggle/input/nfl-player-contact-detection/train_labels.csv"))

# train_tracking = pd.read_csv("/kaggle/input/nfl-player-contact-detection/train_player_tracking.csv")

# train_helmets = pd.read_csv("/kaggle/input/nfl-player-contact-detection/train_baseline_helmets.csv")

# train_video_metadata = pd.read_csv("/kaggle/input/nfl-player-contact-detection/train_video_metadata.csv")


# !mkdir -p ../work/frames

# for video in tqdm(train_helmets.video.unique()):
#     if 'Endzone2' not in video:
#         !ffmpeg -i /kaggle/input/nfl-player-contact-detection/train/{video} -q:v 2 -f image2 /kaggle/work/frames/{video}_%04d.jpg -hide_banner -loglevel error


# def create_features(df, tr_tracking, merge_col="step", use_cols=["x_position", "y_position"]):
#     output_cols = []
#     df_combo = (
#         df.astype({"nfl_player_id_1": "str"})
#         .merge(
#             tr_tracking.astype({"nfl_player_id": "str"})[
#                 ["game_play", merge_col, "nfl_player_id",] + use_cols
#             ],
#             left_on=["game_play", merge_col, "nfl_player_id_1"],
#             right_on=["game_play", merge_col, "nfl_player_id"],
#             how="left",
#         )
#         .rename(columns={c: c+"_1" for c in use_cols})
#         .drop("nfl_player_id", axis=1)
#         .merge(
#             tr_tracking.astype({"nfl_player_id": "str"})[
#                 ["game_play", merge_col, "nfl_player_id"] + use_cols
#             ],
#             left_on=["game_play", merge_col, "nfl_player_id_2"],
#             right_on=["game_play", merge_col, "nfl_player_id"],
#             how="left",
#         )
#         .drop("nfl_player_id", axis=1)
#         .rename(columns={c: c+"_2" for c in use_cols})
#         .sort_values(["game_play", merge_col, "nfl_player_id_1", "nfl_player_id_2"])
#         .reset_index(drop=True)
#     )
#     output_cols += [c+"_1" for c in use_cols]
#     output_cols += [c+"_2" for c in use_cols]
    
#     if ("x_position" in use_cols) & ("y_position" in use_cols):
#         index = df_combo['x_position_2'].notnull()
        
#         distance_arr = np.full(len(index), np.nan)
#         tmp_distance_arr = np.sqrt(
#             np.square(df_combo.loc[index, "x_position_1"] - df_combo.loc[index, "x_position_2"])
#             + np.square(df_combo.loc[index, "y_position_1"]- df_combo.loc[index, "y_position_2"])
#         )
        
#         distance_arr[index] = tmp_distance_arr
#         df_combo['distance'] = distance_arr
#         output_cols += ["distance"]
        
#     df_combo['G_flug'] = (df_combo['nfl_player_id_2']=="G")
#     output_cols += ["G_flug"]
#     return df_combo, output_cols


# use_cols = [
#     'x_position', 'y_position', 'speed', 'distance',
#     'direction', 'orientation', 'acceleration', 'sa'
# ]

# train, feature_cols = create_features(train_labels, train_tracking, use_cols=use_cols)
# train


# train_filtered = train.query('not distance>2').reset_index(drop=True)
# train_filtered['frame'] = (train_filtered['step']/10*59.94+5*59.94).astype('int')+1
# train_filtered


# del train, train_labels, train_tracking
# gc.collect()


train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
    A.Normalize(mean=[0.], std=[1.]),
    ToTensorV2()
])

valid_aug = A.Compose([
    A.Normalize(mean=[0.], std=[1.]),
    ToTensorV2()
])


# video2helmets = {}
# train_helmets_new = train_helmets.set_index('video')
# for video in tqdm(train_helmets.video.unique()):
#     video2helmets[video] = train_helmets_new.loc[video].reset_index(drop=True)
    
# del train_helmets, train_helmets_new
# gc.collect()


# video2frames = {}

# for game_play in tqdm(train_video_metadata.game_play.unique()):
#     for view in ['Endzone', 'Sideline']:
#         video = game_play + f'_{view}.mp4'
#         video2frames[video] = max(list(map(lambda x:int(x.split('_')[-1].split('.')[0]), \
#                                            glob.glob(f'/kaggle/work/frames/{video}*'))))


# class MyDataset(Dataset):
#     def __init__(self, df, aug=train_aug, mode='train'):
#         df = df[:len(df)//10]
#         self.df = df
#         self.frame = df.frame.values
#         self.feature = df[feature_cols].fillna(-1).values
#         self.players = df[['nfl_player_id_1','nfl_player_id_2']].values
#         self.game_play = df.game_play.values
#         self.aug = aug
#         self.mode = mode
        
#     def __len__(self):
#         return len(self.df)
    
#     # @lru_cache(1024)
#     # def read_img(self, path):
#     #     return cv2.imread(path, 0)
   
#     def __getitem__(self, idx):   
#         window = 24
#         frame = self.frame[idx]
        
#         if self.mode == 'train':
#             frame = frame + random.randint(-6, 6)

#         players = []
#         for p in self.players[idx]:
#             if p == 'G':
#                 players.append(p)
#             else:
#                 players.append(int(p))
        
#         imgs = []
#         for view in ['Endzone', 'Sideline']:
#             video = self.game_play[idx] + f'_{view}.mp4'

#             tmp = video2helmets[video]
# #             tmp = tmp.query('@frame-@window<=frame<=@frame+@window')
#             tmp[tmp['frame'].between(frame-window, frame+window)]
#             tmp = tmp[tmp.nfl_player_id.isin(players)]#.sort_values(['nfl_player_id', 'frame'])
#             tmp_frames = tmp.frame.values
#             tmp = tmp.groupby('frame')[['left','width','top','height']].mean()
# #0.002s

#             bboxes = []
#             for f in range(frame-window, frame+window+1, 1):
#                 if f in tmp_frames:
#                     x, w, y, h = tmp.loc[f][['left','width','top','height']]
#                     bboxes.append([x, w, y, h])
#                 else:
#                     bboxes.append([np.nan, np.nan, np.nan, np.nan])
#             bboxes = pd.DataFrame(bboxes).interpolate(limit_direction='both').values
#             bboxes = bboxes[::4]

#             if bboxes.sum() > 0:
#                 flag = 1
#             else:
#                 flag = 0
# #0.03s
                    
#             for i, f in enumerate(range(frame-window, frame+window+1, 4)):
#                 img_new = np.zeros((128, 128), dtype=np.float32)

#                 if flag == 1 and f <= video2frames[video]:
#                     img = cv2.imread(f'/kaggle/work/frames/{video}_{f:04d}.jpg', 0)

#                     x, w, y, h = bboxes[i]

#                     img = img[int(y+h/2)-64:int(y+h/2)+64,int(x+w/2)-64:int(x+w/2)+64].copy()
#                     img_new[:img.shape[0], :img.shape[1]] = img
                    
#                 imgs.append(img_new)
# #0.06s
                
#         feature = np.float32(self.feature[idx])

#         img = np.array(imgs).transpose(1, 2, 0)    
#         img = self.aug(image=img)["image"]
#         label = np.float32(self.df.contact.values[idx])

#         return img, feature, label


# img, feature, label = MyDataset(train_filtered, train_aug, 'train')[0]
# plt.imshow(img.permute(1,2,0)[:,:,7])
# plt.show()
# img.shape, feature, label


# class PositionAttention(nn.Module):

#     def __init__(self, in_channels):
#         super().__init__()
#         self.query_conv = nn.Conv2d(
#             in_channels, in_channels // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))

#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):

#         N, C, H, W = x.shape
#         query = self.query_conv(x).view(
#             N, -1, H*W).permute(0, 2, 1)  # (N, H*W, C')
#         key = self.key_conv(x).view(N, -1, H*W)  # (N, C', H*W)

#         # caluculate correlation
#         energy = torch.bmm(query, key)    # (N, H*W, H*W)
#         # spatial normalize
#         attention = self.softmax(energy)

#         value = self.value_conv(x).view(N, -1, H*W)    # (N, C, H*W)

#         out = torch.bmm(value, attention.permute(0, 2, 1))
#         out = out.view(N, C, H, W)
#         out = self.gamma*out + x
#         return out


# class ChannelAttention(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):

#         N, C, H, W = x.shape
#         query = x.view(N, C, -1)    # (N, C, H*W)
#         key = x.view(N, C, -1).permute(0, 2, 1)    # (N, H*W, C)

#         # calculate correlation
#         energy = torch.bmm(query, key)    # (N, C, C)
#         energy = torch.max(
#             energy, -1, keepdim=True)[0].expand_as(energy) - energy
#         attention = self.softmax(energy)

#         value = x.view(N, C, -1)

#         out = torch.bmm(attention, value)
#         out = out.view(N, C, H, W)
#         out = self.gamma*out + x
#         return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = timm.create_model(CFG['model'], pretrained=False, num_classes=500, in_chans=13)

        self.feature_reduction = nn.Linear(1000, 512)

        #self.position_attention = PositionAttention(512)  # Update input size here
        #self.channel_attention = ChannelAttention()  # Update input size here

        self.mlp = nn.Sequential(
            nn.Linear(18, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.Linear(64, 64),
            # nn.LayerNorm(64),
            # nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.fc = nn.Linear(64 + 512, 1)

    def forward(self, img, feature):
        B, C, H, W = img.shape
        img = img.reshape(B*2, C//2, H, W)
        img = self.backbone(img).reshape(B, -1, 1, 1)  # Output is [B, 1000, 1, 1]
        
        img = self.feature_reduction(img.view(B, -1))  # Reduce from 1000 -> 512
        img = img.view(B, 512, 1, 1)  # Reshape back to (B, C, H, W)
    
        # Apply position and channel attention in parallel
        #pos_att = self.position_attention(img)
        #chan_att = self.channel_attention(img)
    
        # Fuse outputs (e.g., element-wise sum)
        #img = pos_att + chan_att  # You can also use concatenation or other fusion methods
        img = img.view(B, -1)  # Flatten
        feature = self.mlp(feature) 
        y = self.fc(torch.cat([img, feature], dim=1))
        
        return y


# train_set = MyDataset(train_filtered, train_aug, 'train')
# train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], shuffle=True, num_workers=CFG['num_workers'], pin_memory=True)


# model = Model().to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
# criterion = nn.BCEWithLogitsLoss()

# scaler = torch.cuda.amp.GradScaler()  # Add this before training

# for epoch in range(CFG['epochs']):
#     model.train()
#     running_loss = 0.0
#     for img, feature, label in tqdm(train_loader):
#         img, feature, label = img.to(device), feature.to(device), label.to(device).float()
        
#         optimizer.zero_grad()
#         with torch.cuda.amp.autocast():  # Enables mixed precision
#             output = model(img, feature).squeeze(-1)
#             loss = criterion(output, label)

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
        
#         running_loss += loss.item()
    
#     print(f"Epoch [{epoch+1}/{CFG['epochs']}], Loss: {running_loss/len(train_loader):.4f}")



# torch.save(model.state_dict(), "/kaggle/working/weights_resnet18.pth")


def expand_contact_id(df):
    """
    Splits out contact_id into seperate columns.
    """
    df["game_play"] = df["contact_id"].str[:12]
    df["step"] = df["contact_id"].str.split("_").str[-3].astype("int")
    df["nfl_player_id_1"] = df["contact_id"].str.split("_").str[-2]
    df["nfl_player_id_2"] = df["contact_id"].str.split("_").str[-1]
    return df

labels = expand_contact_id(pd.read_csv("/kaggle/input/nfl-player-contact-detection/sample_submission.csv"))

test_tracking = pd.read_csv("/kaggle/input/nfl-player-contact-detection/test_player_tracking.csv")

test_helmets = pd.read_csv("/kaggle/input/nfl-player-contact-detection/test_baseline_helmets.csv")

test_video_metadata = pd.read_csv("/kaggle/input/nfl-player-contact-detection/test_video_metadata.csv")


!mkdir -p ../work/frames

for video in tqdm(test_helmets.video.unique()):
    if 'Endzone2' not in video:
        !ffmpeg -i /kaggle/input/nfl-player-contact-detection/test/{video} -q:v 2 -f image2 /kaggle/work/frames/{video}_%04d.jpg -hide_banner -loglevel error


def create_features(df, tr_tracking, merge_col="step", use_cols=["x_position", "y_position"]):
    output_cols = []
    df_combo = (
        df.astype({"nfl_player_id_1": "str"})
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id",] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_1"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={c: c+"_1" for c in use_cols})
        .drop("nfl_player_id", axis=1)
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id"] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_2"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .drop("nfl_player_id", axis=1)
        .rename(columns={c: c+"_2" for c in use_cols})
        .sort_values(["game_play", merge_col, "nfl_player_id_1", "nfl_player_id_2"])
        .reset_index(drop=True)
    )
    output_cols += [c+"_1" for c in use_cols]
    output_cols += [c+"_2" for c in use_cols]
    
    if ("x_position" in use_cols) & ("y_position" in use_cols):
        index = df_combo['x_position_2'].notnull()
        
        distance_arr = np.full(len(index), np.nan)
        tmp_distance_arr = np.sqrt(
            np.square(df_combo.loc[index, "x_position_1"] - df_combo.loc[index, "x_position_2"])
            + np.square(df_combo.loc[index, "y_position_1"]- df_combo.loc[index, "y_position_2"])
        )
        
        distance_arr[index] = tmp_distance_arr
        df_combo['distance'] = distance_arr
        output_cols += ["distance"]
        
    df_combo['G_flug'] = (df_combo['nfl_player_id_2']=="G")
    output_cols += ["G_flug"]
    return df_combo, output_cols


use_cols = [
    'x_position', 'y_position', 'speed', 'distance',
    'direction', 'orientation', 'acceleration', 'sa'
]

test, feature_cols = create_features(labels, test_tracking, use_cols=use_cols)
test


test_filtered = test.query('not distance>2').reset_index(drop=True)
test_filtered['frame'] = (test_filtered['step']/10*59.94+5*59.94).astype('int')+1
test_filtered


del test, labels, test_tracking
gc.collect()


video2helmets = {}
test_helmets_new = test_helmets.set_index('video')
for video in tqdm(test_helmets.video.unique()):
    video2helmets[video] = test_helmets_new.loc[video].reset_index(drop=True)
    
del test_helmets, test_helmets_new
gc.collect()


video2frames = {}

for game_play in tqdm(test_video_metadata.game_play.unique()):
    for view in ['Endzone', 'Sideline']:
        video = game_play + f'_{view}.mp4'
        video2frames[video] = max(list(map(lambda x:int(x.split('_')[-1].split('.')[0]), \
                                           glob.glob(f'/kaggle/work/frames/{video}*'))))



class MyTestDataset(Dataset):
    def __init__(self, df, aug=valid_aug, mode='train'):
        self.df = df
        self.frame = df.frame.values
        self.feature = df[feature_cols].fillna(-1).values
        self.players = df[['nfl_player_id_1','nfl_player_id_2']].values
        self.game_play = df.game_play.values
        self.aug = aug
        self.mode = mode
        
    def __len__(self):
        return len(self.df)
    
    # @lru_cache(1024)
    # def read_img(self, path):
    #     return cv2.imread(path, 0)
   
    def __getitem__(self, idx):   
        window = 24
        frame = self.frame[idx]
        
        if self.mode == 'train':
            frame = frame + random.randint(-6, 6)

        players = []
        for p in self.players[idx]:
            if p == 'G':
                players.append(p)
            else:
                players.append(int(p))
        
        imgs = []
        for view in ['Endzone', 'Sideline']:
            video = self.game_play[idx] + f'_{view}.mp4'

            tmp = video2helmets[video]
#             tmp = tmp.query('@frame-@window<=frame<=@frame+@window')
            tmp[tmp['frame'].between(frame-window, frame+window)]
            tmp = tmp[tmp.nfl_player_id.isin(players)]#.sort_values(['nfl_player_id', 'frame'])
            tmp_frames = tmp.frame.values
            tmp = tmp.groupby('frame')[['left','width','top','height']].mean()
#0.002s

            bboxes = []
            for f in range(frame-window, frame+window+1, 1):
                if f in tmp_frames:
                    x, w, y, h = tmp.loc[f][['left','width','top','height']]
                    bboxes.append([x, w, y, h])
                else:
                    bboxes.append([np.nan, np.nan, np.nan, np.nan])
            bboxes = pd.DataFrame(bboxes).interpolate(limit_direction='both').values
            bboxes = bboxes[::4]

            if bboxes.sum() > 0:
                flag = 1
            else:
                flag = 0
#0.03s
                    
            for i, f in enumerate(range(frame-window, frame+window+1, 4)):
                img_new = np.zeros((256, 256), dtype=np.float32)

                if flag == 1 and f <= video2frames[video]:
                    img = cv2.imread(f'/kaggle/work/frames/{video}_{f:04d}.jpg', 0)

                    x, w, y, h = bboxes[i]

                    img = img[int(y+h/2)-128:int(y+h/2)+128,int(x+w/2)-128:int(x+w/2)+128].copy()
                    img_new[:img.shape[0], :img.shape[1]] = img
                    
                imgs.append(img_new)
#0.06s
                
        feature = np.float32(self.feature[idx])

        img = np.array(imgs).transpose(1, 2, 0)    
        img = self.aug(image=img)["image"]
        label = np.float32(self.df.contact.values[idx])

        return img, feature, label


img, feature, label = MyTestDataset(test_filtered, valid_aug, 'test')[0]
plt.imshow(img.permute(1,2,0)[:,:,7])
plt.show()
img.shape, feature, label


test_set = MyTestDataset(test_filtered, valid_aug, 'test')
test_loader = DataLoader(test_set, batch_size=CFG['valid_bs'], shuffle=False, num_workers=CFG['num_workers'], pin_memory=True)

model = Model().to(device)
model.load_state_dict(torch.load('/kaggle/input/resnet18only/weights_resnet18.pth'))

model.eval()
    
y_pred = []
with torch.no_grad():
    tk = tqdm(test_loader, total=len(test_loader))
    for step, batch in enumerate(tk):
        img, feature, label = [x.to(device) for x in batch]
        output = model(img, feature).squeeze(-1)

        y_pred.extend(output.sigmoid().cpu().numpy())

y_pred = np.array(y_pred)


th = 0.29

test_filtered['contact'] = (y_pred >= th).astype('int')

sub = pd.read_csv('/kaggle/input/nfl-player-contact-detection/sample_submission.csv')

sub = sub.drop("contact", axis=1).merge(test_filtered[['contact_id', 'contact']], how='left', on='contact_id')
sub['contact'] = sub['contact'].fillna(0).astype('int')

sub[["contact_id", "contact"]].to_csv("submission.csv", index=False)

sub.head()




