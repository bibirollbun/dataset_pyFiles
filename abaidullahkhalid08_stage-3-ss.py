!pip install -q iterative-stratification


import os
import timm
import torch
import random
import pydicom
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
import seaborn as sns
import torch.nn as nn
from typing import List
from torch import Tensor
import albumentations as A
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


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


class SeverityPrediction(Dataset):
    def __init__(self, df, series, coor, meta, condition, usage='train'):
        self.series = series
        self.coor = coor
        self.meta = meta
        self.df = df
        self.condition = condition
        self.usage = usage
        self.sag_window = 5
        self.ax_window = 5
        
        # Define labels based on condition
        if condition == 'scs':
            self.label = [
                'spinal_canal_stenosis_l1_l2', 'spinal_canal_stenosis_l2_l3', 'spinal_canal_stenosis_l3_l4', 'spinal_canal_stenosis_l4_l5', 'spinal_canal_stenosis_l5_s1'
            ]
        elif condition == 'ss':
            self.label = [
                'left_subarticular_stenosis_l1_l2', 'left_subarticular_stenosis_l2_l3', 'left_subarticular_stenosis_l3_l4', 'left_subarticular_stenosis_l4_l5', 'left_subarticular_stenosis_l5_s1',
                'right_subarticular_stenosis_l1_l2', 'right_subarticular_stenosis_l2_l3', 'right_subarticular_stenosis_l3_l4', 'right_subarticular_stenosis_l4_l5', 'right_subarticular_stenosis_l5_s1'
            ]
        elif condition == 'nfn':
            self.label = [
                'left_neural_foraminal_narrowing_l1_l2', f'left_neural_foraminal_narrowing_l2_l3', f'left_neural_foraminal_narrowing_l3_l4', f'left_neural_foraminal_narrowing_l4_l5', f'left_neural_foraminal_narrowing_l5_s1',
                'right_neural_foraminal_narrowing_l1_l2', f'right_neural_foraminal_narrowing_l2_l3', f'right_neural_foraminal_narrowing_l3_l4', f'right_neural_foraminal_narrowing_l4_l5', f'right_neural_foraminal_narrowing_l5_s1'
            ]
        
        # Clean and prepare labels
        # Get only study IDs with complete label data (no NaN values)
        self.id = df.loc[~(df[self.label].isna().any(axis=1)), 'study_id'].unique()
        
        # Remove specific problematic study ID
        self.id = list(set(self.id) - set([3637444890]))
        
        # Map string labels to numeric values
        for l in self.label:
            df[l] = df[l].map({'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2})
            # Ensure any remaining NaNs are filled with a valid class
            df[l] = df[l].fillna(0).astype(int)

        # Set up image transformations
        self.wide_resize = v2.Resize((128, 224))
        self.rec_resize = v2.Resize((256, 256))
        self.resize = v2.Resize((128, 128))
        self.resize_3d = v2.Resize((256, 256))
        self.pre_resize = v2.Resize((512, 512))
        
        # Data augmentation for training
        self.wide_transforms = A.Compose([
            A.RandomBrightnessContrast(p=0.25),
            # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=(-0.1, 0.1), rotate_limit=20, border_mode=0, p=0.5),
            A.Resize(128, 224),
        ])
        self.rec_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.25),
            # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=(-0.1, 0.1), rotate_limit=20, border_mode=0, p=0.5),
            A.Resize(256, 256),
        ])
        
    def __getitem__(self, index):
        study_id = self.id[index]
        res = {}
        
        # Load images based on condition
        ax_img, ax_depth = self.volume_ss(study_id)
        res['ax'] = ax_img.to(torch.float32)
        res['ax_depth'] = ax_depth

        # Extract labels for this study
        study_labels = self.df.loc[(self.df.study_id==study_id), self.label]
        
        # Verify we have valid labels
        if len(study_labels) > 0:
            # Convert to tensor
            label_values = study_labels.values.squeeze()
            # Check for NaN or problematic values and replace with 0
            if isinstance(label_values, np.ndarray):
                # Replace any NaN or invalid values with 0
                label_values = np.nan_to_num(label_values, nan=0.0)
                # Ensure all values are 0, 1, or 2
                label_values = np.clip(label_values, 0, 2).astype(np.int64)
            
            label = torch.tensor(label_values, dtype=torch.long)
            res['label'] = label
        else:
            # Create a fallback label of all zeros
            num_labels = len(self.label)
            res['label'] = torch.zeros(num_labels, dtype=torch.long)
            
        return res
    
    def crop(self, image, x, y, z, x_left, x_right, y_bottom, y_top, wide):
        size = [image[i].shape for i in z]
        data = torch.stack([
            self.pre_resize(torch.tensor(image[i])[None, ...]).squeeze()[
                max(int((y/shape[0]) * 512 - y_top), 0): int((y/shape[0]) * 512 + y_bottom),
                max(int((x/shape[1]) * 512 - x_left), 0): int((x/shape[1]) * 512 + x_right)
            ]
            for i, shape in zip(z, size)
        ])

        if wide:
            data = self.wide_resize(data)
        else:
            data = self.rec_resize(data)

        if self.usage == 'train':

            if wide:
                transformed = self.wide_transforms(
                    image=data.numpy().transpose((1,2,0)).astype(np.float32)
                )['image']
                data = torch.from_numpy(transformed.transpose((2,0,1))).float()
            else:
                transformed = self.rec_transforms(
                    image=data.numpy().transpose((1,2,0)).astype(np.float32)
                )['image']
                data = torch.from_numpy(transformed.transpose((2,0,1))).float()

        return data

    def volume_ss(self, study_id):
        ax_meta = self.meta.loc[(self.meta.study_id==study_id) & (self.meta.series_description=='Axial T2')]
        # sagt1_meta = self.meta.loc[(self.meta.study_id==study_id) & (self.meta.series_description=='Sagittal T1')]
        
        ax_meta = ax_meta.sort_values('ipp_z', ascending=False).reset_index(drop=True)
        # sagt1_meta = sagt1_meta.sort_values('ipp_x', ascending=True).reset_index(drop=True)
        
        ax_img = [self.normalize(self.load_dicom(f'/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{row.study_id}/{row.series_id}/{row.instance_number}.dcm')) for _, row in ax_meta.iterrows()]
        # sagt1_img = [self.normalize(self.load_dicom(f'/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{row.study_id}/{row.series_id}/{row.instance_number}.dcm')) for _, row in sagt1_meta.iterrows()]
        
        ax_right_sub_coor = self.coor.loc[(self.coor.study_id==study_id) & (self.coor.condition=='Right Subarticular Stenosis')]
        ax_left_sub_coor = self.coor.loc[(self.coor.study_id==study_id) & (self.coor.condition=='Left Subarticular Stenosis')]
        # sagt1_right_sub_coor = self.coor.loc[(self.coor.study_id==study_id) & (self.coor.condition=='Right Neural Foraminal Narrowing')]
        # sagt1_left_sub_coor = self.coor.loc[(self.coor.study_id==study_id) & (self.coor.condition=='Left Neural Foraminal Narrowing')]
        
        # AXIAL T2
        ax_right_dict = {}
        ax_right_label_dict = {}
        for _, row in ax_right_sub_coor.iterrows():
            label = 2
            u = np.random.uniform(0, 1)
            if u < 0.5:
                z_shift = 0
            elif u > 0.5 and u < 0.85:
                z_shift = np.random.choice([-1, 1])
            else:
                z_shift = np.random.choice([-2, 2])
            if self.usage == 'train':
                y_shift = random.randint(-10, 10)
                x_shift = random.randint(-10, 10)
            else:
                y_shift = 0
                x_shift = 0
            ax_right_label_dict[row.level] = label + z_shift
            ax_meta_sub = ax_meta.loc[(ax_meta.series_id==row.series_id)]
            ax_meta_sub_original_idx = ax_meta_sub.index.tolist()
            ax_meta_sub = ax_meta_sub.reset_index(drop=True)
            mid = ax_meta_sub.loc[(ax_meta_sub.instance_number==row.instance_number)].index[0]
            z = [min(max(mid+w+z_shift, 0), len(ax_meta_sub)-1) for w in range(-(self.ax_window-1)//2, ((self.ax_window-1)//2)+1)]
            ax_right_dict[row.level] = self.crop([ax_img[i] for i in range(len(ax_img)) if i in ax_meta_sub_original_idx], row.x+x_shift, row.y+y_shift, z, 160-16, 32+16, 64+32, 64+32, wide=False)
        ax_left_dict = {}
        ax_left_label_dict = {}
        for _, row in ax_left_sub_coor.iterrows():
            label = 2
            u = np.random.uniform(0, 1)
            if u < 0.5:
                z_shift = 0
            elif u > 0.5 and u < 0.85:
                z_shift = np.random.choice([-1, 1])
            else:
                z_shift = np.random.choice([-2, 2])
            if self.usage == 'train':
                y_shift = random.randint(-10, 10)
                x_shift = random.randint(-10, 10)
            else:
                y_shift = 0
                x_shift = 0
            ax_left_label_dict[row.level] = label + z_shift
            ax_meta_sub = ax_meta.loc[(ax_meta.series_id==row.series_id)]
            ax_meta_sub_original_idx = ax_meta_sub.index.tolist()
            ax_meta_sub = ax_meta_sub.reset_index(drop=True)
            mid = ax_meta_sub.loc[(ax_meta_sub.instance_number==row.instance_number)].index[0]
            z = [min(max(mid+w+z_shift, 0), len(ax_meta_sub)-1) for w in range(-(self.ax_window-1)//2, ((self.ax_window-1)//2)+1)]
            ax_left_dict[row.level] = self.crop([ax_img[i] for i in range(len(ax_img)) if i in ax_meta_sub_original_idx], row.x+x_shift, row.y+y_shift, z, 32+16, 160-16, 64+32, 64+32, wide=False)

        ax_right_img = [ax_right_dict.get(l, torch.zeros((self.ax_window, 256, 256))) for l in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']]
        ax_left_img = [ax_left_dict.get(l, torch.zeros((self.ax_window, 256, 256))) for l in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']]
        # sagt1_right_img = [sagt1_right_dict.get(l, torch.zeros((self.sag_window, 128, 224))) for l in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']]
        # sagt1_left_img = [sagt1_left_dict.get(l, torch.zeros((self.sag_window, 128, 224))) for l in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']]
        ax_right_label = {'right_' + l: torch.tensor(ax_right_label_dict.get(l, 2)).to(torch.long) for l in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']}
        ax_left_label = {'left_' + l: torch.tensor(ax_left_label_dict.get(l, 2)).to(torch.long) for l in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']}
        # sagt1_right_label = {'right_' + l: torch.tensor(sagt1_right_label_dict.get(l, 2)).to(torch.long) for l in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']}
        # sagt1_left_label = {'left_' + l: torch.tensor(sagt1_left_label_dict.get(l, 2)).to(torch.long) for l in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']}
        ax_label = dict(**ax_left_label, **ax_right_label)
        # sagt1_label = dict(**sagt1_left_label, **sagt1_right_label)
        ax_img = ax_left_img + ax_right_img
        # sagt1_img = sagt1_left_img + sagt1_right_img
        return torch.stack(ax_img).contiguous(), ax_label,
        
    def normalize(self, x):
        lower, upper = np.percentile(x, (1, 99))
        x = np.clip(x, lower, upper)
        x = x - np.min(x)
        x = x / np.max(x)
        return x

    def __len__(self):
        return len(self.id)

    def load_dicom(self, path):
        dicom = pydicom.dcmread(path)      
        return dicom.pixel_array

    def check_position(self, first, last, pos):
        first_dcm = pydicom.read_file(first)
        last_dcm = pydicom.read_file(last)
        first_dcm = first_dcm.ImagePositionPatient[pos]
        last_dcm = last_dcm.ImagePositionPatient[pos]
        if pos == 0:
            return first_dcm > last_dcm
        elif pos == 2:
            return first_dcm < last_dcm
        else:
            raise ValueError



class LSTMMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(LSTMMIL, self).__init__()
        self.lstm = nn.LSTM(input_dim, input_dim//2, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
        self.aux_attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        self.attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
    def forward(self, bags):
        """
        Args:
            bags: (batch_size, num_instances, input_dim)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, num_instances, input_dim = bags.size()
        bags_lstm, _ = self.lstm(bags)
        attn_scores = self.attention(bags_lstm).squeeze(-1)
        aux_attn_scores = self.aux_attention(bags_lstm).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        weighted_instances = torch.bmm(attn_weights.unsqueeze(1), bags_lstm).squeeze(1)  # (batch_size, input_dim)
        return weighted_instances, aux_attn_scores

class SSMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.ax_encoder = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', in_chans=1, pretrained=True, num_classes=0)
        self.ax_flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(1))
        self.ax_num_features = self.ax_encoder.num_features
        self.ax_head = LSTMMIL(self.ax_num_features, 512, 3)
        self.out = nn.Linear(self.ax_num_features, 3)
        self.dropout = nn.Dropout(0.1)
    def forward(self, ax):
        if isinstance(ax, tuple):
            ax = ax[0]
        ax_shape = ax.shape
        ax = ax.reshape(ax_shape[0]*ax_shape[1]*ax_shape[2], 1, ax_shape[-2], ax_shape[-1])
        ax = self.ax_encoder.forward_features(ax)
        ax = self.ax_flatten(ax)
        ax = ax.reshape(ax_shape[0]*ax_shape[1], ax_shape[2], -1)
        ax_weighted_sum, ax_attn = self.ax_head(ax)
        ax_attn = ax_attn.reshape(ax_shape[0], ax_shape[1], -1)
        out = ax_weighted_sum
        out = self.dropout(out)
        out = self.out(out)
        ax_attn = {'left_L1/L2': ax_attn[:, 0, :],'left_L2/L3': ax_attn[:, 1, :],'left_L3/L4': ax_attn[:, 2, :], 'left_L4/L5': ax_attn[:, 3, :], 'left_L5/S1': ax_attn[:, 4, :],
                'right_L1/L2': ax_attn[:, 5, :], 'right_L2/L3': ax_attn[:, 6, :], 'right_L3/L4': ax_attn[:, 7, :], 'right_L4/L5': ax_attn[:, 8, :], 'right_L5/S1': ax_attn[:, 9, :]}
        return out, ax_attn


# import torch
# import torch.nn as nn
# import timm

# class GatedAttentionMIL(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_classes):
#         super(GatedAttentionMIL, self).__init__()
#         self.attention_V = nn.Linear(input_dim, hidden_dim)
#         self.attention_U = nn.Linear(input_dim, hidden_dim)
#         self.attention_weights = nn.Linear(hidden_dim, 1)

#     def forward(self, bags):
#         """
#         Args:
#             bags: (batch_size, num_instances, input_dim)

#         Returns:
#             weighted_instances: (batch_size, input_dim)
#             attn_scores: (batch_size, num_instances)
#         """
#         # Apply gating: tanh(Vx) * sigmoid(Ux)
#         A_V = torch.tanh(self.attention_V(bags))  # (batch_size, num_instances, hidden_dim)
#         A_U = torch.sigmoid(self.attention_U(bags))  # (batch_size, num_instances, hidden_dim)
#         A = A_V * A_U  # (batch_size, num_instances, hidden_dim)

#         # Compute attention scores
#         attn_scores = self.attention_weights(A).squeeze(-1)  # (batch_size, num_instances)

#         # Softmax over instances
#         attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch_size, num_instances)

#         # Weighted sum of instance features
#         weighted_instances = torch.bmm(attn_weights.unsqueeze(1), bags).squeeze(1)  # (batch_size, input_dim)

#         return weighted_instances, attn_scores

# class SSMIL(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Encoder: EfficientNetV2 backbone
#         self.ax_encoder = timm.create_model(
#             'tf_efficientnetv2_s.in21k_ft_in1k', in_chans=1, pretrained=True, num_classes=0
#         )
#         self.ax_flatten = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(1)
#         )
#         self.ax_num_features = self.ax_encoder.num_features

#         # Updated: using GatedAttentionMIL instead of LSTM-MIL
#         self.ax_head = GatedAttentionMIL(self.ax_num_features, 512, 3)
#         self.out = nn.Linear(self.ax_num_features, 3)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, ax):
#         """
#         Args:
#             ax: Tensor of shape (batch_size, num_slices, num_levels, H, W)

#         Returns:
#             out: (batch_size, num_classes)
#             ax_attn: dict of attention scores for each vertebral level
#         """
#         if isinstance(ax, tuple):
#             ax = ax[0]
#         ax_shape = ax.shape  # (batch_size, num_slices, num_levels, H, W)

#         # Reshape: (B * S * L, 1, H, W)
#         ax = ax.reshape(ax_shape[0] * ax_shape[1] * ax_shape[2], 1, ax_shape[-2], ax_shape[-1])

#         # CNN encoder
#         ax = self.ax_encoder.forward_features(ax)  # (B * S * L, C, H', W')
#         ax = self.ax_flatten(ax)  # (B * S * L, C)

#         # Reshape back: (B * S, L, C)
#         ax = ax.reshape(ax_shape[0] * ax_shape[1], ax_shape[2], -1)

#         # Apply MIL head
#         ax_weighted_sum, ax_attn = self.ax_head(ax)  # (B*S, C), (B*S, L)

#         # Reshape attention: (B, S, L)
#         ax_attn = ax_attn.reshape(ax_shape[0], ax_shape[1], -1)

#         # Classifier
#         out = self.dropout(ax_weighted_sum)
#         out = self.out(out)

#         # Build attention dictionary
#         ax_attn = {
#             'left_L1/L2': ax_attn[:, 0, :],
#             'left_L2/L3': ax_attn[:, 1, :],
#             'left_L3/L4': ax_attn[:, 2, :],
#             'left_L4/L5': ax_attn[:, 3, :],
#             'left_L5/S1': ax_attn[:, 4, :],
#             'right_L1/L2': ax_attn[:, 5, :],
#             'right_L2/L3': ax_attn[:, 6, :],
#             'right_L3/L4': ax_attn[:, 7, :],
#             'right_L4/L5': ax_attn[:, 8, :],
#             'right_L5/S1': ax_attn[:, 9, :],
#         }
#         return out, ax_attn



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



class NFNSSDepthDetectLoss(nn.Module):
    def __init__(self):
        super(NFNSSDepthDetectLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        loss = 0
        for level in ['left_L1/L2', 'left_L2/L3', 'left_L3/L4', 'left_L4/L5', 'left_L5/S1',
              'right_L1/L2', 'right_L2/L3', 'right_L3/L4', 'right_L4/L5', 'right_L5/S1']:
            target = targets[level].to(outputs[level].device)  # Move to same device
            _loss = nn.functional.cross_entropy(outputs[level], target.reshape(-1))
            loss += _loss
            
        return loss / 10  


class SCSNFNSSLoss(nn.Module):
    def __init__(self, is_train=False):
        super(SCSNFNSSLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0]).to(device))
        self.aux_loss_ax = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0]).to(device))
        self.aux_loss_sagt2 = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0]).to(device))
        self.aux_loss_sagt1 = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0]).to(device))
        self.is_train = is_train
    def forward(self, outputs, targets, ax=None, sagt2=None, sagt1=None):
        targets = targets.reshape(-1)
        #print(outputs, targets, outputs.shape, targets.shape)
        ax_loss = 0
        sagt2_loss = 0
        sagt1_loss = 0
        num_loss = 1
        loss = self.loss(outputs, targets)
        if ax is not None:
            ax_loss = self.aux_loss_ax(ax, targets)
            if self.is_train:
                loss += ax_loss*0.5
                num_loss += 0.5
        if sagt2 is not None:
            sagt2_loss = self.aux_loss_sagt2(sagt2, targets)
            if self.is_train:
                loss += sagt2_loss*0.5
                num_loss += 0.5
        if sagt1 is not None:
            sagt1_loss = self.aux_loss_sagt1(sagt1, targets)
            if self.is_train:
                loss += sagt1_loss*0.5
                num_loss += 0.5
        #print(loss)
        loss = loss/num_loss
        return loss, ax_loss, sagt2_loss, sagt1_loss


def calculate_accuracy_score(outputs, targets):
    """Computes accuracy for model predictions across multiple spinal levels."""
    correct_total = 0
    samples_total = 0
    
    pred_classes = {}
    true_classes = {}
    
    for level in outputs.keys():
        if level in targets:
            # Get predicted class (argmax along class dimension)
            preds = torch.argmax(outputs[level], dim=1)

            # Ensure targets are reshaped and on the correct device
            true_labels = targets[level].reshape(-1).long()

            # Store predictions and ground truth (ensure they're on CPU)
            pred_classes[level] = preds.detach().cpu().numpy()
            true_classes[level] = true_labels.detach().cpu().numpy()
            
            # Calculate accuracy for this level
            correct = (preds == true_labels).sum().item()
            correct_total += correct
            samples_total += true_labels.size(0)
    
    # Avoid division by zero
    if samples_total == 0:
        return 0.0, pred_classes, true_classes
    
    accuracy = correct_total / samples_total
    return accuracy, pred_classes, true_classes



def calculate_classification_tolerance(outputs, batch, tolerances=[0, 1, 2]):
    """Calculate tolerance metrics for classification"""
    tolerance_counts = {f"Â±{tol}": 0 for tol in tolerances}
    tolerance_counts[">Â±2"] = 0
    total_predictions = 0
    
    predicted_classes = outputs.argmax(dim=1)
    
    if 'label' in batch:
        targets = batch['label']
        abs_diff = torch.abs(predicted_classes - targets)
        total_predictions = targets.size(0)
        
        # Count predictions within each tolerance
        for tol in tolerances:
            tolerance_counts[f"Â±{tol}"] = (abs_diff <= tol).sum().item()
        
        # Count predictions beyond the maximum tolerance
        tolerance_counts[">Â±2"] = (abs_diff > max(tolerances)).sum().item()
    else:
        # Process by level
        levels = [key for key in batch.keys() if key.startswith('spinal_canal_stenosis')]
        for i, level_key in enumerate(levels):
            level_preds = outputs[i::len(levels)].argmax(dim=1)
            level_targets = batch[level_key]
            abs_diff = torch.abs(level_preds - level_targets)
            total_predictions += level_targets.size(0)
            
            # Count predictions within each tolerance
            for tol in tolerances:
                tolerance_counts[f"Â±{tol}"] += (abs_diff <= tol).sum().item()
            
            # Count predictions beyond the maximum tolerance
            tolerance_counts[">Â±2"] += (abs_diff > max(tolerances)).sum().item()
    
    return tolerance_counts, total_predictions


def train_spine_model(model, train_coor, train_meta, df, series, n_folds=5, epochs=7, batch_size=1,
                      learning_rate=0.001, weight_decay=0.0001, patience=5,
                      mixed_precision=True, experiment_name="spine_classification", tolerances=[0, 1, 2],
                      condition='ss'):
    """
    Improved training function for spine classification model using advanced loss functions

    Args:
        model: Your defined model (SCSMIL or equivalent)
        train_coor: DataFrame containing coordinate annotations
        train_meta: DataFrame containing metadata
        df: DataFrame with labels
        series: Series data
        n_folds: Number of folds for cross-validation
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        patience: Patience for early stopping
        mixed_precision: Whether to use mixed precision training
        experiment_name: Name for experiment logs
        tolerances: List of tolerance values for evaluation
        condition: Condition type ('scs', 'ss', or 'nfn')
    """

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory for models
    os.makedirs("models", exist_ok=True)

    # Initialize tensorboard writer
    writer = SummaryWriter(f'runs/{experiment_name}')

    # Print training configuration
    print(f"\n===== TRAINING CONFIGURATION =====")
    print(f"Number of folds: {n_folds}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Mixed precision: {mixed_precision}")
    print(f"Condition: {condition}")
    print(f"==============================\n")

    # Initialize scaler for mixed precision
    scaler = torch.amp.GradScaler() if mixed_precision else None

    # Initialize loss functions from your friend's code
    loss_module = SCSNFNSSLoss(is_train=True)
    val_loss_module = SCSNFNSSLoss(is_train=False)
    ss_depth_loss_module = NFNSSDepthDetectLoss()  # You might want to use NFNSSDepthDetectLoss() for NFN condition

    # Cross-validation loop
    all_val_accuracies = []
    all_val_f1_scores = []

    for fold in range(n_folds):
        print(f"\n{'='*20} FOLD {fold+1}/{n_folds} {'='*20}")

        # Initialize fold-specific writer
        fold_writer = SummaryWriter(f'runs/{experiment_name}/fold_{fold}')

        # Initialize datasets and dataloaders
        train_dataset = SeverityPrediction(
            df=df,
            series=series,
            coor=train_coor.loc[train_coor.fold != fold],
            meta=train_meta.loc[train_meta.fold != fold],
            condition=condition,
            usage='train'
        )

        valid_dataset = SeverityPrediction(
            df=df,
            series=series,
            coor=train_coor.loc[train_coor.fold == fold],
            meta=train_meta.loc[train_meta.fold == fold],
            condition=condition,
            usage='valid'
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=0, 
            pin_memory=True
        )

        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )

        # Initialize model, optimizer and loss
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Learning rate scheduler with warmup
        total_steps = epochs * len(train_loader)
        warmup_steps = int(0.1 * total_steps)  # 10% warmup

        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5
        )

        # Initialize tracking variables
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_acc = 0.0
        counter = 0  # For early stopping

        # Training loop
        for epoch in range(epochs):
            print(f"\nğŸš€ Epoch {epoch+1}/{epochs} - Fold {fold+1}/{n_folds}")

            # === Training phase ===
            model.train()
            running_loss = 0.0
            running_ax_depth_loss = 0.0
            total_acc = 0.0

            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                               desc="Training Progress", leave=False)

            all_train_preds = []
            all_train_targets = []
            
            # Initialize tolerance counts and totals for both train and validation
            epoch_train_tolerances = {f"Â±{tol}": 0 for tol in tolerances}
            epoch_train_tolerances[">Â±2"] = 0
            epoch_val_tolerances = {f"Â±{tol}": 0 for tol in tolerances}
            epoch_val_tolerances[">Â±2"] = 0
            train_total_predictions = 0
            val_total_predictions = 0

            for batch_idx, batch in progress_bar:
                # Process batch data based on condition
                ax = batch['ax'].to(device)
                label = batch['label'].to(device)
                ax_depth = batch['ax_depth']
                
                # Zero gradients before forward pass
                optimizer.zero_grad()

                # Mixed precision training
                if mixed_precision and torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        # Forward pass with model
                        preds, ax_depth_pred = model(ax)
                        loss, _, _, _ = loss_module(preds, label)
                        
                        # Additional depth losses
                        ax_depth_loss = ss_depth_loss_module(ax_depth_pred, ax_depth)
                        
                        # Combined loss
                        total_loss = loss + ax_depth_loss

                    # Scale loss and backward pass
                    scaler.scale(total_loss).backward()

                    # Unscale before gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                else:
                    # Standard precision training
                    preds, ax_depth_pred = model(ax)
                    loss, _, _, _ = loss_module(preds, label)
                    
                    # Additional depth losses
                    ax_depth_loss = ss_depth_loss_module(ax_depth_pred, ax_depth)
                    
                    # Combined loss
                    total_loss = loss + ax_depth_loss
                    
                    total_loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Optimizer step
                    optimizer.step()
                    scheduler.step()

                # Track metrics
                running_loss += loss.item()
                running_ax_depth_loss += ax_depth_loss.item() 

                # Store predictions for later evaluation
                all_train_preds.append(preds.detach().cpu())
                all_train_targets.append(label.detach().cpu())
                
                # Calculate batch accuracy
                # Handle different possible prediction shapes
                B = preds.size(0)  # batch size
                
                # Ground truth should be shape [B], if not already
                label = label.view(-1)
                
                # Convert predictions to class indices
                pred_classes = preds.argmax(dim=-1)
                
                # Compute per-element accuracy
                correct = (pred_classes == label).float().sum()
                batch_acc = correct / label.numel()
                total_acc += batch_acc.item()
                
                # Calculate tolerance metrics
                abs_diff = torch.abs(pred_classes - label)
                batch_total = label.size(0)
                train_total_predictions += batch_total
                
                # Count predictions within each tolerance
                for tol in tolerances:
                    count = (abs_diff <= tol).sum().item()
                    epoch_train_tolerances[f"Â±{tol}"] += count
                
                # Count predictions beyond the maximum tolerance
                count = (abs_diff > max(tolerances)).sum().item()
                epoch_train_tolerances[">Â±2"] += count

                # Update progress bar
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    ax_depth_loss=f"{ax_depth_loss.item():.4f}",
                    total_loss=f"{total_loss.item():.4f}",
                    acc=f"{batch_acc.item():.4f}",
                    lr=f"{current_lr:.6f}"
                )

                # Free memory
                variables_to_delete = [
                    'batch', 'loss', 'total_loss',
                    'ax_depth_loss', 'ax_depth_pred',
                    'preds', 'label'
                ]
                
                for var in variables_to_delete:
                    if var in locals():
                        del locals()[var]
                
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()  # Optional: helps with inter-process caching

            # Calculate epoch metrics
            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_ax_depth_loss = running_ax_depth_loss / len(train_loader) 
            epoch_train_acc = total_acc / len(train_loader)
            train_losses.append(epoch_train_loss)
            train_accs.append(epoch_train_acc)

            # Convert tolerance counts to percentages
            for k in epoch_train_tolerances:
                if train_total_predictions > 0:
                    epoch_train_tolerances[k] = epoch_train_tolerances[k] * 100 / train_total_predictions
                else:
                    epoch_train_tolerances[k] = 0.0

            # Log metrics
            fold_writer.add_scalar('Loss/train', epoch_train_loss, epoch)
            fold_writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
            fold_writer.add_scalar('Loss/train_ax_depth', epoch_train_ax_depth_loss, epoch)
            writer.add_scalar(f'Loss/train/fold_{fold}', epoch_train_loss, epoch)
            writer.add_scalar(f'Accuracy/train/fold_{fold}', epoch_train_acc, epoch)

            print(f"ğŸ”¥ Training Loss: {epoch_train_loss:.4f} | Accuracy: {epoch_train_acc:.4f}")
            print(f"   Depth Losses - AX: {epoch_train_ax_depth_loss:.4f}")

            # Log tolerance metrics
            for tolerance_key, tolerance_value in epoch_train_tolerances.items():
                fold_writer.add_scalar(f'Tolerance/train/{tolerance_key}', tolerance_value, epoch)
                writer.add_scalar(f'Tolerance/train/{tolerance_key}/fold_{fold}', tolerance_value, epoch)

            # === Validation phase ===
            model.eval()
            val_running_loss = 0.0
            val_running_ax_depth_loss = 0.0
            val_total_acc = 0.0
            
            all_val_preds = []
            all_val_targets = []

            with torch.no_grad():
                for batch in tqdm(valid_loader, desc="Validation Progress", leave=False):
                    ax = batch['ax'].to(device)
                    label = batch['label'].to(device)
                    ax_depth = batch['ax_depth']
                    
                    # Forward pass
                    preds, ax_depth_pred = model(ax)
                    loss, _, _, _ = val_loss_module(preds, label)
                    
                    # Additional depth losses
                    ax_depth_loss = ss_depth_loss_module(ax_depth_pred, ax_depth)
                    
                    val_running_loss += loss.item()
                    val_running_ax_depth_loss += ax_depth_loss.item()
                    
                    # Handle different possible prediction shapes
                    B = preds.size(0)  # batch size
                    
                    # Ground truth should be shape [B], if not already
                    label = label.view(-1)
                    
                    # Convert predictions to class indices
                    pred_classes = preds.argmax(dim=-1)
                    
                    # Compute per-element accuracy
                    correct = (pred_classes == label).float().sum()
                    batch_acc = correct / label.numel()
                    val_total_acc += batch_acc.item()

                    # Store predictions for evaluation
                    all_val_preds.append(preds.cpu())
                    all_val_targets.append(label.cpu())
                    
                    # Calculate tolerance metrics for validation
                    abs_diff = torch.abs(pred_classes - label)
                    batch_total = label.size(0)
                    val_total_predictions += batch_total
                    
                    # Count predictions within each tolerance
                    for tol in tolerances:
                        count = (abs_diff <= tol).sum().item()
                        epoch_val_tolerances[f"Â±{tol}"] += count
                    
                    # Count predictions beyond the maximum tolerance
                    count = (abs_diff > max(tolerances)).sum().item()
                    epoch_val_tolerances[">Â±2"] += count

                    # Free memory
                    variables_to_delete = [
                        'batch', 'loss',
                        'ax_depth_loss',
                        'preds', 'label'
                    ]
                    
                    for var in variables_to_delete:
                        if var in locals():
                            del locals()[var]
                    
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            
            # Calculate validation metrics
            epoch_val_loss = val_running_loss / len(valid_loader)
            epoch_val_ax_depth_loss = val_running_ax_depth_loss / len(valid_loader) 
            epoch_val_acc = val_total_acc / len(valid_loader)
            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc)

            # Convert tolerance counts to percentages
            for k in epoch_val_tolerances:
                if val_total_predictions > 0:
                    epoch_val_tolerances[k] = 100 * epoch_val_tolerances[k] / val_total_predictions
                else:
                    epoch_val_tolerances[k] = 0.0
            
            # Combine predictions for metrics
            all_val_preds_tensor = torch.cat(all_val_preds)
            all_val_targets_tensor = torch.cat(all_val_targets)
            
            # Calculate F1 score - FIX THE SHAPE ISSUE HERE
            pred_classes = all_val_preds_tensor.argmax(dim=-1).cpu().numpy()
            target_classes = all_val_targets_tensor.cpu().numpy()
            
            # Calculate the overall F1 score (no per-column calculation)
            # This fixes the IndexError by not assuming target_classes has a second dimension
            avg_f1_score = f1_score(
                target_classes, 
                pred_classes, 
                average='weighted', 
                zero_division=0
            )

            # Log validation metrics
            fold_writer.add_scalar('Loss/val', epoch_val_loss, epoch)
            fold_writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)
            fold_writer.add_scalar('F1/val', avg_f1_score, epoch)
            fold_writer.add_scalar('Loss/val_ax_depth', epoch_val_ax_depth_loss, epoch)
            writer.add_scalar(f'Loss/val/fold_{fold}', epoch_val_loss, epoch)
            writer.add_scalar(f'Accuracy/val/fold_{fold}', epoch_val_acc, epoch)
            writer.add_scalar(f'F1/val/fold_{fold}', avg_f1_score, epoch)

            # Log validation tolerance metrics
            for tolerance_key, tolerance_value in epoch_val_tolerances.items():
                fold_writer.add_scalar(f'Tolerance/val/{tolerance_key}', tolerance_value, epoch)
                writer.add_scalar(f'Tolerance/val/{tolerance_key}/fold_{fold}', tolerance_value, epoch)

            print(f"âœ… Validation Loss: {epoch_val_loss:.4f} | Accuracy: {epoch_val_acc:.4f} | F1 Score: {avg_f1_score:.4f}")
            print(f"   Depth Losses - AX: {epoch_val_ax_depth_loss:.4f}")
            print(f"ğŸ“� Validation Tolerances: " + " | ".join([f"{k}: {v:.1f}%" for k, v in epoch_val_tolerances.items()]))

            # Model checkpointing - save best model based on validation accuracy
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_f1 = avg_f1_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': epoch_val_loss,
                    'val_acc': epoch_val_acc,
                    'val_f1': avg_f1_score,
                    'val_tolerances': epoch_val_tolerances,
                }, f'models/{experiment_name}_fold_{fold}_best.pt')

                print(f"ğŸ“Œ New best model saved with Accuracy: {best_val_acc:.4f} and F1: {best_f1:.4f}")
                counter = 0  # Reset early stopping counter
            else:
                counter += 1
                print(f"âš ï¸� No improvement for {counter}/{patience} epochs")

            # Early stopping
            if counter >= patience:
                print(f"â›” Early stopping triggered after {epoch+1} epochs")
                break

        # End of fold - record best validation accuracy and F1
        all_val_accuracies.append(best_val_acc)
        all_val_f1_scores.append(best_f1)
        print(f"Fold {fold+1} completed. Best validation Accuracy: {best_val_acc:.4f}, F1: {best_f1:.4f}")

        # Print classification report
        print("\n--- Classification Report ---")
        # Fix the classification report call to match the corrected data shapes
        report = classification_report(
                target_classes,
                pred_classes,
                zero_division=0
            )
        print(report)

        # Save final model for this fold
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_losses[-1],
            'val_acc': val_accs[-1],
            'val_f1': avg_f1_score,
        }, f'models/{experiment_name}_fold_{fold}_final.pt')

        # Close fold writer
        fold_writer.close()

    # End of cross-validation
    avg_val_acc = sum(all_val_accuracies) / len(all_val_accuracies) if all_val_accuracies else 0.0
    avg_val_f1 = sum(all_val_f1_scores) / len(all_val_f1_scores) if all_val_f1_scores else 0.0
    print(f"\n===== TRAINING COMPLETE =====")
    print(f"Cross-validation results:")
    for fold, (acc, f1) in enumerate(zip(all_val_accuracies, all_val_f1_scores)):
        print(f"Fold {fold+1}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")
    print(f"Average validation Accuracy: {avg_val_acc:.4f}")
    print(f"Average validation F1 Score: {avg_val_f1:.4f}")

    # Save experiment summary
    with open(f'models/{experiment_name}_summary.txt', 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Folds: {n_folds}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Weight decay: {weight_decay}\n")
        f.write(f"Mixed precision: {mixed_precision}\n")
        f.write(f"Condition: {condition}\n")
        f.write(f"Tolerance thresholds: {tolerances}\n")
        f.write("\nResults:\n")
        for fold, (acc, f1) in enumerate(zip(all_val_accuracies, all_val_f1_scores)):
            f.write(f"Fold {fold+1}: Accuracy = {acc:.4f}, F1 = {f1:.4f}\n")
        f.write(f"Average validation Accuracy: {avg_val_acc:.4f}\n")
        f.write(f"Average validation F1 Score: {avg_val_f1:.4f}\n")
        
        # Add tolerance metrics summary across all folds
        f.write("\nTolerance Metrics Summary:\n")
        for tol in tolerances:
            f.write(f"Â±{tol}: Represents predictions within {tol} grades of ground truth\n")
        f.write(f">Â±{max(tolerances)}: Represents predictions more than {max(tolerances)} grades away from ground truth\n")

    # Close writer
    writer.close()
    
    return all_val_accuracies, all_val_f1_scores, avg_val_acc, avg_val_f1

# Example usage:
# model = SSMIL()
# accuracies, f1_scores, avg_acc, avg_f1 = train_spine_model(model, train_coor, train_meta, train_dummy, train_series, n_folds=5, condition='ss')

# Example usage:
model = SSMIL()
accuracies, f1_scores, avg_acc, avg_f1 = train_spine_model(model, train_coor, train_meta, train_dummy, train_series, n_folds=5, condition='ss')



def train_spine_model(model, train_coor, train_meta, df, series, n_folds=5, epochs=7, batch_size=1,
                      learning_rate=0.001, weight_decay=0.0001, patience=5,
                      mixed_precision=True, experiment_name="spine_classification", tolerances=[0, 1, 2],
                      condition='ss'):
    """
    Improved training function for spine classification model using advanced loss functions

    Args:
        model: Your defined model (SCSMIL or equivalent)
        train_coor: DataFrame containing coordinate annotations
        train_meta: DataFrame containing metadata
        df: DataFrame with labels
        series: Series data
        n_folds: Number of folds for cross-validation
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        patience: Patience for early stopping
        mixed_precision: Whether to use mixed precision training
        experiment_name: Name for experiment logs
        tolerances: List of tolerance values for evaluation
        condition: Condition type ('scs', 'ss', or 'nfn')
    """

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory for models
    os.makedirs("models", exist_ok=True)

    plots_dir = f"plots/{experiment_name}"
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(f"{plots_dir}/confusion_matrices", exist_ok=True)
    os.makedirs(f"{plots_dir}/learning_curves", exist_ok=True)
    os.makedirs(f"{plots_dir}/tolerance_metrics", exist_ok=True)
    os.makedirs(f"{plots_dir}/predictions", exist_ok=True)

    # Initialize tensorboard writer
    writer = SummaryWriter(f'runs/{experiment_name}')

    # Print training configuration
    print(f"\n===== TRAINING CONFIGURATION =====")
    print(f"Number of folds: {n_folds}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Mixed precision: {mixed_precision}")
    print(f"Condition: {condition}")
    print(f"==============================\n")

    # Initialize scaler for mixed precision
    scaler = torch.amp.GradScaler() if mixed_precision else None

    # Initialize loss functions from your friend's code
    loss_module = SCSNFNSSLoss(is_train=True)
    val_loss_module = SCSNFNSSLoss(is_train=False)
    nfn_depth_loss_module = NFNSSDepthDetectLoss()  # You might want to use NFNSSDepthDetectLoss() for NFN condition

    # Cross-validation loop
    all_val_accuracies = []
    all_val_f1_scores = []
    
    # Initialize lists to store data for visualization
    all_fold_train_losses = []
    all_fold_val_losses = []
    all_fold_train_accs = []
    all_fold_val_accs = []
    all_fold_train_tolerances = []
    all_fold_val_tolerances = []
    all_fold_confusion_matrices = []

    for fold in range(n_folds):
        print(f"\n{'='*20} FOLD {fold+1}/{n_folds} {'='*20}")

        # Initialize fold-specific writer
        fold_writer = SummaryWriter(f'runs/{experiment_name}/fold_{fold}')

        # Initialize datasets and dataloaders
        train_dataset = SeverityPrediction(
            df=df,
            series=series,
            coor=train_coor.loc[train_coor.fold != fold],
            meta=train_meta.loc[train_meta.fold != fold],
            condition=condition,
            usage='train'
        )

        valid_dataset = SeverityPrediction(
            df=df,
            series=series,
            coor=train_coor.loc[train_coor.fold == fold],
            meta=train_meta.loc[train_meta.fold == fold],
            condition=condition,
            usage='valid'
        )
        

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=0, 
            pin_memory=True
        )

        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )

        # Initialize model, optimizer and loss
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Learning rate scheduler with warmup
        total_steps = epochs * len(train_loader)
        warmup_steps = int(0.1 * total_steps)  # 10% warmup

        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5
        )

        # Initialize tracking variables
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_acc = 0.0
        counter = 0  # For early stopping

        # Training loop
        for epoch in range(epochs):
            print(f"\nğŸš€ Epoch {epoch+1}/{epochs} - Fold {fold+1}/{n_folds}")

            # === Training phase ===
            model.train()
            running_loss = 0.0
            running_ax_depth_loss = 0.0
            running_sagt1_depth_loss = 0.0
            total_acc = 0.0

            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                               desc="Training Progress", leave=False)

            all_train_preds = []
            all_train_targets = []
            
            # Initialize tolerance counts and totals for both train and validation
            epoch_train_tolerances = {f"Â±{tol}": 0 for tol in tolerances}
            epoch_train_tolerances[">Â±2"] = 0
            epoch_val_tolerances = {f"Â±{tol}": 0 for tol in tolerances}
            epoch_val_tolerances[">Â±2"] = 0
            train_total_predictions = 0
            val_total_predictions = 0

            for batch_idx, batch in progress_bar:
                # Process batch data based on condition
                
                ax = batch['ax'].to(device)
                label = batch['label'].to(device)
                ax_depth = batch['ax_depth']
                
                # Zero gradients before forward pass
                optimizer.zero_grad()

                # Mixed precision training
                if mixed_precision and torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        # Forward pass with model
                        
                        preds, ax_depth_pred = model(ax)
                        loss, _, _, _ = loss_module(preds, label)
                        
                        # Additional depth losses
                        ax_depth_loss = nfn_depth_loss_module(ax_depth_pred, ax_depth)
                        
                        # Combined loss
                        total_loss = loss + ax_depth_loss

                    # Scale loss and backward pass
                    scaler.scale(total_loss).backward()

                    # Unscale before gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                else:
                    # Standard precision training
                    
                    preds, ax_depth_pred = model(ax)
                    loss, _, _, _ = loss_module(preds, label)
                    
                    # Additional depth losses
                    ax_depth_loss = nfn_depth_loss_module(ax_depth_pred, ax_depth)
                    
                    # Combined loss
                    total_loss = loss + ax_depth_loss
                    
                    total_loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Optimizer step
                    optimizer.step()
                    scheduler.step()

                # Track metrics
                running_loss += loss.item()
                running_ax_depth_loss += ax_depth_loss.item() 

                
                # Store predictions for later evaluation
                all_train_preds.append(preds.detach().cpu())
                all_train_targets.append(label.detach().cpu())

                
                
                # Calculate batch accuracy
                
                # Handle different possible prediction shapes
                B = preds.size(0)  # batch size

                # # preds: [B, 25, 3] â†’ reshape to [B, 5, 5, 3] (5 disc levels Ã— 5 views each)
                # preds = preds.view(B, 5, 5, 3)
                
                # # Aggregate over 5 instances per disc level (mean pooling) â†’ [B, 5, 3]
                # preds = preds.mean(dim=2)
                
                # Ground truth should be shape [B, 5], if not already
                label = label.view(B)
                
                # Convert predictions to class indices â†’ [B, 5]
                pred_classes = preds.argmax(dim=-1)
                
                # Compute per-element accuracy
                correct = (pred_classes == label).float().sum()
                batch_acc = correct / label.numel()
                total_acc += batch_acc.item()
                
                # Calculate tolerance metrics
                abs_diff = torch.abs(pred_classes - label)
                batch_total = label.size(0)
                train_total_predictions += batch_total
                
                # Count predictions within each tolerance
                for tol in tolerances:
                    count = (abs_diff <= tol).sum().item()
                    epoch_train_tolerances[f"Â±{tol}"] += count
                
                # Count predictions beyond the maximum tolerance
                count = (abs_diff > max(tolerances)).sum().item()
                epoch_train_tolerances[">Â±2"] += count

                # Update progress bar
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    ax_depth_loss=f"{ax_depth_loss.item():.4f}",
                    total_loss=f"{total_loss.item():.4f}",
                    acc=f"{batch_acc.item():.4f}",
                    lr=f"{current_lr:.6f}"
                )

                # Free memory
                variables_to_delete = [
                    'batch', 'loss', 'total_loss',
                    'ax_depth_loss', 'ax_depth_pred',
                    'preds', 'label'
                ]
                
                for var in variables_to_delete:
                    if var in locals():
                        del locals()[var]
                
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()  # Optional: helps with inter-process caching

            # Calculate epoch metrics
            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_ax_depth_loss = running_ax_depth_loss / len(train_loader) 
            epoch_train_acc = total_acc / len(train_loader)
            train_losses.append(epoch_train_loss)
            train_accs.append(epoch_train_acc)

            # Convert tolerance counts to percentages
            for k in epoch_train_tolerances:
                if train_total_predictions > 0:
                    epoch_train_tolerances[k] = epoch_train_tolerances[k] * 100 / train_total_predictions
                else:
                    epoch_train_tolerances[k] = 0.0

            # Log metrics
            fold_writer.add_scalar('Loss/train', epoch_train_loss, epoch)
            fold_writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
            
            fold_writer.add_scalar('Loss/train_ax_depth', epoch_train_ax_depth_loss, epoch)
            
            writer.add_scalar(f'Loss/train/fold_{fold}', epoch_train_loss, epoch)
            writer.add_scalar(f'Accuracy/train/fold_{fold}', epoch_train_acc, epoch)

            print(f"ğŸ”¥ Training Loss: {epoch_train_loss:.4f} | Accuracy: {epoch_train_acc:.4f}")
            
            print(f"   Depth Losses - AX: {epoch_train_ax_depth_loss:.4f}")

            # Log tolerance metrics
            for tolerance_key, tolerance_value in epoch_train_tolerances.items():
                fold_writer.add_scalar(f'Tolerance/train/{tolerance_key}', tolerance_value, epoch)
                writer.add_scalar(f'Tolerance/train/{tolerance_key}/fold_{fold}', tolerance_value, epoch)

            # === Validation phase ===
            model.eval()
            val_running_loss = 0.0
            val_running_ax_depth_loss = 0.0
            val_running_sagt1_depth_loss = 0.0
            val_total_acc = 0.0
            
            all_val_preds = []
            all_val_targets = []

            with torch.no_grad():
                for batch in tqdm(valid_loader, desc="Validation Progress", leave=False):
                    
                    ax = batch['ax'].to(device)
                    label = batch['label'].to(device)
                    ax_depth = batch['ax_depth']
                    
                    # Forward pass
                    preds, ax_depth_pred = model(ax)
                    loss, _, _, _ = val_loss_module(preds, label)
                    
                    # Additional depth losses
                    ax_depth_loss = nfn_depth_loss_module(ax_depth_pred, ax_depth)
                    
                    val_running_loss += loss.item()
                    
                    val_running_ax_depth_loss += ax_depth_loss.item()
                    
                    
                    # Handle different possible prediction shapes
                    B = preds.size(0)  # batch size
    
                    # # preds: [B, 25, 3] â†’ reshape to [B, 5, 5, 3] (5 disc levels Ã— 5 views each)
                    # preds = preds.view(B, 5, 5, 3)
                    
                    # # Aggregate over 5 instances per disc level (mean pooling) â†’ [B, 5, 3]
                    # preds = preds.mean(dim=2)
                    
                    # Ground truth should be shape [B, 5], if not already
                    label = label.view(B)
                    
                    # Convert predictions to class indices â†’ [B, 5]
                    pred_classes = preds.argmax(dim=-1)
                    
                    # Compute per-element accuracy
                    correct = (pred_classes == label).float().sum()
                    batch_acc = correct / label.numel()
                    val_total_acc += batch_acc.item()

                    
                    # Store predictions for evaluation
                    all_val_preds.append(preds.cpu())
                    all_val_targets.append(label.cpu())
                    
                    # Calculate tolerance metrics for validation
                    
                    abs_diff = torch.abs(pred_classes - label)
                    batch_total = label.size(0)
                    val_total_predictions += batch_total
                    
                    # Count predictions within each tolerance
                    for tol in tolerances:
                        count = (abs_diff <= tol).sum().item()
                        epoch_val_tolerances[f"Â±{tol}"] += count
                    
                    # Count predictions beyond the maximum tolerance
                    count = (abs_diff > max(tolerances)).sum().item()
                    epoch_val_tolerances[">Â±2"] += count

                    # Free memory
                    variables_to_delete = [
                        'batch', 'loss',
                        'ax_depth_loss',
                        'preds', 'label'
                    ]
                    
                    for var in variables_to_delete:
                        if var in locals():
                            del locals()[var]
                    
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            # Calculate validation metrics
            epoch_val_loss = val_running_loss / len(valid_loader)
            epoch_val_ax_depth_loss = val_running_ax_depth_loss / len(valid_loader) 
            epoch_val_acc = val_total_acc / len(valid_loader)
            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc)

            # Convert tolerance counts to percentages
            for k in epoch_val_tolerances:
                if val_total_predictions > 0:
                    epoch_val_tolerances[k] = 100 * epoch_val_tolerances[k] / val_total_predictions
                else:
                    epoch_val_tolerances[k] = 0.0
            
            # Combine predictions for metrics
            all_val_preds_tensor = torch.cat(all_val_preds)
            all_val_targets_tensor = torch.cat(all_val_targets)
            
            # Calculate F1 score
            pred_classes = all_val_preds_tensor.argmax(dim=-1).cpu().numpy()
            target_classes = all_val_targets_tensor.cpu().numpy()
            
            # Flatten for confusion matrix later
            target_classes_flat = target_classes.flatten()
            pred_classes_flat = pred_classes.flatten()
            
            # Compute weighted F1 for each disc level (column)
            f1_scores_per_level = [
                f1_score(target_classes[:, i], pred_classes[:, i], average='weighted', zero_division=0)
                for i in range(target_classes.shape[1])
            ]
            
            avg_f1_score = sum(f1_scores_per_level) / len(f1_scores_per_level)

            # Log validation metrics
            fold_writer.add_scalar('Loss/val', epoch_val_loss, epoch)
            fold_writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)
            fold_writer.add_scalar('F1/val', avg_f1_score, epoch)
            
            fold_writer.add_scalar('Loss/val_ax_depth', epoch_val_ax_depth_loss, epoch)
            
            writer.add_scalar(f'Loss/val/fold_{fold}', epoch_val_loss, epoch)
            writer.add_scalar(f'Accuracy/val/fold_{fold}', epoch_val_acc, epoch)
            writer.add_scalar(f'F1/val/fold_{fold}', avg_f1_score, epoch)

            # Log validation tolerance metrics
            for tolerance_key, tolerance_value in epoch_val_tolerances.items():
                fold_writer.add_scalar(f'Tolerance/val/{tolerance_key}', tolerance_value, epoch)
                writer.add_scalar(f'Tolerance/val/{tolerance_key}/fold_{fold}', tolerance_value, epoch)

            print(f"âœ… Validation Loss: {epoch_val_loss:.4f} | Accuracy: {epoch_val_acc:.4f} | F1 Score: {avg_f1_score:.4f}")
            
            print(f"   Depth Losses - AX: {epoch_val_ax_depth_loss:.4f}")
            print(f"ğŸ“� Validation Tolerances: " + " | ".join([f"{k}: {v:.1f}%" for k, v in epoch_val_tolerances.items()]))

            # Model checkpointing - save best model based on validation accuracy
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_f1 = avg_f1_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': epoch_val_loss,
                    'val_acc': epoch_val_acc,
                    'val_f1': avg_f1_score,
                    'val_tolerances': epoch_val_tolerances,
                }, f'models/{experiment_name}_fold_{fold}_best.pt')

                print(f"ğŸ“Œ New best model saved with Accuracy: {best_val_acc:.4f} and F1: {best_f1:.4f}")
                counter = 0  # Reset early stopping counter
            else:
                counter += 1
                print(f"âš ï¸� No improvement for {counter}/{patience} epochs")

            # Early stopping
            if counter >= patience:
                print(f"â›” Early stopping triggered after {epoch+1} epochs")
                break

        # End of fold - record best validation accuracy and F1
        all_val_accuracies.append(best_val_acc)
        all_val_f1_scores.append(best_f1)
        print(f"Fold {fold+1} completed. Best validation Accuracy: {best_val_acc:.4f}, F1: {best_f1:.4f}")

        # Print classification report
        print("\n--- Classification Report ---")
        report = classification_report(
                target_classes_flat,
                pred_classes_flat,
                zero_division=0
            )
        print(report)

        cm = confusion_matrix(target_classes_flat, pred_classes_flat)
        fold_cm = {'matrix': cm, 'fold': fold}
        all_fold_confusion_matrices.append(fold_cm)
                
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f'Confusion Matrix - Fold {fold+1}')
        plt.savefig(f"{plots_dir}/confusion_matrices/fold_{fold+1}_confusion_matrix.png", bbox_inches='tight')
        plt.close()

        # Save final model for this fold
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_losses[-1],
            'val_acc': val_accs[-1],
            'val_f1': avg_f1_score,
        }, f'models/{experiment_name}_fold_{fold}_final.pt')

        #Later Visualization
        all_fold_train_losses.append(train_losses)
        all_fold_val_losses.append(val_losses)
        all_fold_train_accs.append(train_accs)
        all_fold_val_accs.append(val_accs)
        all_fold_train_tolerances.append(epoch_train_tolerances)
        all_fold_val_tolerances.append(epoch_val_tolerances)
            
        # Create fold-specific learning curves
        plt.figure(figsize=(12, 5))
            
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Fold {fold+1} Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
            
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.title(f'Fold {fold+1} Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/learning_curves/fold_{fold+1}_learning_curves.png")
        plt.close()
            
        # Create tolerance visualization for this fold
        plt.figure(figsize=(10, 6))
        tol_keys = list(epoch_val_tolerances.keys())
        tol_values = [epoch_val_tolerances[k] for k in tol_keys]
            
        bars = plt.bar(tol_keys, tol_values, color='skyblue')
        plt.title(f'Validation Tolerance Metrics - Fold {fold+1}')
        plt.xlabel('Tolerance Level')
        plt.ylabel('Percentage of Predictions (%)')
        plt.ylim(0, 100)
            
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
                
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(f"{plots_dir}/tolerance_metrics/fold_{fold+1}_tolerance_metrics.png")
        plt.close()

        # Close fold writer
        fold_writer.close()

    # End of cross-validation
    avg_val_acc = sum(all_val_accuracies) / len(all_val_accuracies) if all_val_accuracies else 0.0
    avg_val_f1 = sum(all_val_f1_scores) / len(all_val_f1_scores) if all_val_f1_scores else 0.0
    print(f"\n===== TRAINING COMPLETE =====")
    print(f"Cross-validation results:")
    for fold, (acc, f1) in enumerate(zip(all_val_accuracies, all_val_f1_scores)):
        print(f"Fold {fold+1}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")
    print(f"Average validation Accuracy: {avg_val_acc:.4f}")
    print(f"Average validation F1 Score: {avg_val_f1:.4f}")

    # Save experiment summary
    with open(f'models/{experiment_name}_summary.txt', 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Folds: {n_folds}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Weight decay: {weight_decay}\n")
        f.write(f"Mixed precision: {mixed_precision}\n")
        f.write(f"Condition: {condition}\n")
        f.write(f"Tolerance thresholds: {tolerances}\n")
        f.write("\nResults:\n")
        for fold, (acc, f1) in enumerate(zip(all_val_accuracies, all_val_f1_scores)):
            f.write(f"Fold {fold+1}: Accuracy = {acc:.4f}, F1 = {f1:.4f}\n")
        f.write(f"Average validation Accuracy: {avg_val_acc:.4f}\n")
        f.write(f"Average validation F1 Score: {avg_val_f1:.4f}\n")
        
        # Add tolerance metrics summary across all folds
        f.write("\nTolerance Metrics Summary:\n")
        for tol in tolerances:
            f.write(f"Â±{tol}: Represents predictions within {tol} grades of ground truth\n")
        f.write(f">Â±{max(tolerances)}: Represents predictions more than {max(tolerances)} grades away from ground truth\n")

    # Create visualizations for the entire experiment
    
    # 1. Average learning curves across all folds
    plt.figure(figsize=(12, 5))
        
    # Find the minimum length of epochs (in case early stopping kicked in)
    min_epochs_train = min([len(losses) for losses in all_fold_train_losses])
    min_epochs_val = min([len(losses) for losses in all_fold_val_losses])
        
    # Prepare data for averaging
    train_losses_truncated = [losses[:min_epochs_train] for losses in all_fold_train_losses]
    val_losses_truncated = [losses[:min_epochs_val] for losses in all_fold_val_losses]
    train_accs_truncated = [accs[:min_epochs_train] for accs in all_fold_train_accs]
    val_accs_truncated = [accs[:min_epochs_val] for accs in all_fold_val_accs]
        
    # Calculate mean and std for each epoch
    mean_train_loss = np.mean(train_losses_truncated, axis=0)
    std_train_loss = np.std(train_losses_truncated, axis=0)
    mean_val_loss = np.mean(val_losses_truncated, axis=0)
    std_val_loss = np.std(val_losses_truncated, axis=0)
        
    mean_train_acc = np.mean(train_accs_truncated, axis=0)
    std_train_acc = np.std(train_accs_truncated, axis=0)
    mean_val_acc = np.mean(val_accs_truncated, axis=0)
    std_val_acc = np.std(val_accs_truncated, axis=0)
        
    # Plot loss curves with shaded std dev
    plt.subplot(1, 2, 1)
    epochs_range = np.arange(1, min_epochs_train + 1)
    plt.plot(epochs_range, mean_train_loss, label='Train Loss', color='blue')
    plt.fill_between(epochs_range, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, 
                    alpha=0.2, color='blue')
        
    plt.plot(epochs_range, mean_val_loss, label='Validation Loss', color='red')
    plt.fill_between(epochs_range, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, 
                    alpha=0.2, color='red')
        
    plt.title('Average Loss Across All Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/average_learning_curves.png")
    plt.close()
        
    # 2. Combined confusion matrix across all folds
    if all_fold_confusion_matrices:
        # Sum up all confusion matrices
        combined_cm = sum([item['matrix'] for item in all_fold_confusion_matrices])
        
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=combined_cm)
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title('Combined Confusion Matrix Across All Folds')
        plt.savefig(f"{plots_dir}/combined_confusion_matrix.png", bbox_inches='tight')
        plt.close()
        
    # 3. Bar chart comparing fold performances
    plt.figure(figsize=(10, 6))
    fold_indices = np.arange(1, n_folds + 1)
    
    width = 0.35
    plt.bar(fold_indices - width/2, all_val_accuracies, width, label='Accuracy', color='royalblue')
    plt.bar(fold_indices + width/2, all_val_f1_scores, width, label='F1 Score', color='darkslateblue')
    
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Performance Metrics by Fold')
    plt.xticks(fold_indices)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (acc, f1) in enumerate(zip(all_val_accuracies, all_val_f1_scores)):
        plt.text(i + 1 - width/2, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom')
        plt.text(i + 1 + width/2, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/fold_performance_comparison.png")
        plt.close()
        
        # 4. Average tolerance metrics across folds
        avg_val_tolerances = {}
        for tol_key in all_fold_val_tolerances[0].keys():
            avg_val_tolerances[tol_key] = np.mean([fold_tol[tol_key] for fold_tol in all_fold_val_tolerances])
            
        plt.figure(figsize=(10, 6))
        tol_keys = list(avg_val_tolerances.keys())
        tol_values = [avg_val_tolerances[k] for k in tol_keys]
        
        bars = plt.bar(tol_keys, tol_values, color='skyblue')
    plt.title('Average Validation Tolerance Metrics Across All Folds')
    plt.xlabel('Tolerance Level')
    plt.ylabel('Percentage of Predictions (%)')
    plt.ylim(0, 100)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
            
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(f"{plots_dir}/tolerance_metrics/average_tolerance_metrics.png")
    plt.close()
    
    # 5. Create radar chart for model performance overview
    # Prepare data
    metrics = ['Accuracy', 'F1 Score', 'Â±0 Tolerance', 'Â±1 Tolerance', 'Â±2 Tolerance']
    values = [
        avg_val_acc,
        avg_val_f1,
        avg_val_tolerances['Â±0'] / 100,  # Convert percentage to 0-1 scale
        avg_val_tolerances['Â±1'] / 100,
        avg_val_tolerances['Â±2'] / 100
    ]
        
    # Create radar chart
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Set the angles for the metrics
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    values.append(values[0])  # Close the loop
    angles.append(angles[0])  # Close the loop
    metrics.append(metrics[0])  # For labeling
    
    # Plot the values
    ax.plot(angles, values, 'o-', linewidth=2, color='dodgerblue')
    ax.fill(angles, values, color='dodgerblue', alpha=0.25)
    
    # Set the labels
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics[:-1])
    
    # Set y limits
    ax.set_ylim(0, 1)
    
    # Add gridlines
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], angle=0)
    ax.grid(True)
    
    plt.title('Model Performance Overview', size=15, y=1.1)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/model_performance_radar.png")
    plt.close()
    
    print(f"\nâœ¨ All visualizations saved to the '{plots_dir}' directory")

    # Cose writer
    writer.close()
    
    
    return all_val_accuracies, all_val_f1_scores, avg_val_acc, avg_val_f1
# Example usage:
# model = SSMIL()
# accuracies, f1_scores, avg_acc, avg_val_acc, avg_val_f1 = train_spine_model(model, train_coor, train_meta, train_dummy, train_series, n_folds=5, condition='ss')




