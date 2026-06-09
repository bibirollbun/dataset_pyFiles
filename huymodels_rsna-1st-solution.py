import gc
import wandb
from pytorch_lightning.loggers import WandbLogger
import os
import yaml
import sys
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
import torchvision.transforms as T
import albumentations as A
import pandas.api.types
import sklearn.metrics
import timm
import scipy
import albumentations as A
from torchvision.transforms import v2
from torchvision import models
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from torch.utils.data import default_collate
import pydicom as dcm
import transformers


os.listdir('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939')


SEED = 1710 # My birth day
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # Fix the network according to random seed
    print('Finish seeding with seed {}'.format(seed))

seed_everything(SEED)
print('Training on device {}'.format(device))


%%writefile config.yaml 
data_path : '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'
out_put_dir : '/kaggle/working/models'

seed : 1101
debug : False
train_bs : 4
valid_bs : 4
test_bs : 8
worker : 1

progress_bar_refresh_rate : 1

pseudo_train : 0

save_topk : 1
fold : 5 # Cross Validation 

task:
    kind: 'detect' # coordinate x, y
    #kind : 'classify' # severity -> label
    #kind : 'depth'
    condition: 'nfn'
    #condition: 'scs'
    #condition: 'scs'
    #condition: 'all'
    #direction: 'satg2'
    direction: 'ax'
    #direction: 'sagt1'
    position:
        - 'L1/L2'
        - 'L2/L3'
        - 'L3/L4'
        - 'L4/L5'
        - 'L5/S1'

in_chans : 3

image_size : 384

model:


import os 
os.listdir('/kaggle/input/')


with open("config.yaml", "r") as file_obj:
    config = yaml.safe_load(file_obj)


# Sử dụng train data trong mở debug mode
if config['debug']:
    IMAGE_PATH = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/'
    series = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv')
else:
    IMAGE_PATH = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/'
    series = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_series_descriptions.csv')


print(IMAGE_PATH)
series.head()


# Đường dẫn đến folder chứa ảnh
folder1 = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089'
folder2 = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518'
folder3 = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845'

file1 = [f for f in os.listdir(folder1) if f.endswith('.dcm')]
file2 = [f for f in os.listdir(folder2) if f.endswith('.dcm')]
file3 = [f for f in os.listdir(folder3) if f.endswith('.dcm')]

# Đọc một file (ví dụ: ảnh đầu tiên)
dicom_path1 = os.path.join(folder1, file1[0])
dicom_path2 = os.path.join(folder2, file1[1])
dicom_path3 = os.path.join(folder3, file1[2])
dcm1 = dcm.dcmread(dicom_path1)
dcm2 = dcm.dcmread(dicom_path2)
dcm3 = dcm.dcmread(dicom_path3)

# Chuyển pixel data sang numpy array
img1 = dcm1.pixel_array
img2 = dcm2.pixel_array
img3 = dcm3.pixel_array

# Hiển thị ảnh
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img1, cmap='gray')
plt.title("Sagittal T2/STIR")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img2, cmap='gray')
plt.title("Axial T2")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img3, cmap='gray')
plt.title("Sagittal T1")
plt.axis('off')

plt.tight_layout()
plt.show()


def create_dcm_df(study_id, series_id, series_description):
    try:
        # Find all .dcm to extract Instance_number từ tên file 1, 2, 3,...  /* để tìm all file dcm
        path_list = glob(IMAGE_PATH + f'{study_id}/{series_id}/*.dcm')
        in_list = sorted([int(s.split('/')[-1].split('.')[0]) for s in path_list])

        # read file dicom by pydicom
        dcm_list = []
        for i in in_list:
            dcm_list.append(dcm.dcmread(IMAGE_PATH + f'{study_id}/{series_id}/{i}.dcm'))
            
        '''
        Extract metadata chính:
        ipp : vị trí x,y,z của ảnh trong cơ thể
        ioo : hướng trục của ảnh trong không gian
        '''
        ipp = np.asarray([d.ImagePositionPatient for d in dcm_list]).astype('float')
        iop = [d.ImageOrientationPatient for d in dcm_list]
        iop = [[float(d[0]), float(d[1]), float(d[2]), 
                float(d[3]), float(d[4]), float(d[5])] for d in iop]
        ipp_x = ipp[:, 0]
        ipp_y = ipp[:, 1]
        ipp_z = ipp[:, 2]

        # Extract độ phân giải (Pixel Spacing) và kích thước
        shape = np.array([d.pixel_array.shape for d in dcm_list])
        sbs = np.asarray([d.SpacingBetweenSlices for d in dcm_list]).astype('float')
        ps = np.asarray([d.PixelSpacing for d in dcm_list]).astype('float') # x, y
        ps_x = ps[:, 0]
        ps_y = ps[:, 1]

        # create dataframe chứa metadata
        meta_dict = {
            'instance_number' : in_list,
            'ipp_x' : ipp_x,
            'ipp_y' : ipp_y,
            'ipp_z' : ipp_z,
            'sbs' : sbs,
            'ps_x' : ps_x, 
            'ps_y' : ps_y
        }
        meta_df = pd.DataFrame(meta_dict)

        #add other fearture
        meta_df['series_id'] = series_id
        meta_df['study_id'] = study_id
        meta_df['series_description'] = series_description
        meta_df['height'] = shape[:, 0]
        meta_df['width'] = shape[:, 1]
        meta_df['iop'] = pd.Series(iop)
        
        # xoá dữ liệu giải phóng bộ nhớ
        del dcm_list, ipp, iop, sbs, ps
        gc.collect()
        return meta_df[['study_id', 'series_id', 'series_description', 'instance_number', 'height', 'width', 'ipp_x', 'ipp_y', 'ipp_z', 'iop', 'sbs', 'ps_x', 'ps_y']]
    except:
        print(study_id, series_id, series_description)
        return None
        


create_dcm_df(44036939, 2828203845, 'Sagittal T1').head()


%%time
if config['debug']:
    meta_df_list = []
    
    # Parallbel song song các 
    meta_df_list = Parallel(n_jobs=-1)([delayed(create_dcm_df)(row.study_id, row.series_id, row.series_description) for _, row in series.iterrows()])

    # merge to big dataframe
    meta_df = pd.concat(meta_df_list)
    del meta_df_list
    gc.collect()

    #turn to file parquet
    meta_df.to_parquet('meta.parquet')
else:
    meta_df_list = []
    meta_df_list = Parallel(n_job = -1)([delayed(create_dcm_df)(row.study_id, row.series_id, row.series_description) for _, row in series.iterrows()])

    meta_df = pd.concat(meta_df_list)
    del meta_df_list
    gc.collect()
    meta_df.to_parquet('meta.parquet')


# Kiểm tra file parquet: file lớn thích hợp để lưu với nhiều dữ liệu hơn csv
import os 
os.listdir('/kaggle/working/')


class DepthDetectDataset(Dataset):
    def __init__(self, meta, condition, usage='sub'):
        if condition == 'scs': 
            meta = meta.loc[meta.series_description=='Sagittal T2/STIR']
        else: 
            meta = meta.loc[meta.series_description=='Sagittal T1']
        self.id = list(meta.study_id.unique())
        if 3637444890 in self.id: 
            self.id.remove(3637444890)
        self.meta = meta
        self.condition = condition
        self.usage = usage
        
        self.resize = v2.Resize((384, 384))
        
    def __getitem__(self, index):
        study_id = self.id[index]
        #print(study_id)
        #try:
        if self.condition == 'scs':
            volume = self.for_scs(study_id)
        elif self.condition == 'nfn':
            try: 
                volume = self.for_nfn(study_id)
            except: 
                print(study_id)
        return volume, torch.tensor([study_id])

    def for_scs(self, study_id):
        depth = 32
        #lọc dữ liệu với 1 study_id
        meta = self.meta.loc[(self.meta.study_id==study_id) & (self.meta.series_description=='Sagittal T2/STIR')]
        
        #sắp xếp trục theo không gian x
        meta = meta.sort_values('ipp_x', ascending=True).reset_index(drop=True)

        #đọc ảnh dicom thành mảng 2d numpy
        img = [self.load_dicom(IMAGE_PATH + f'{row.study_id}/{row.series_id}/{row.instance_number}.dcm') for _, row in meta.iterrows()]
        
        #resize, chuyển ảnh thành tensor rồi stack lại
        volume = self.normalize(torch.cat([self.resize(torch.tensor(i.astype(np.float32))[None, ...]).to(torch.float32) for i in img]).contiguous())
        
        #chuẩn hoá độ sâu depth
        if volume.shape[0] < depth:
            volume = torch.cat([volume, torch.zeros(depth-volume.shape[0], volume.shape[1], volume.shape[2])])
        elif volume.shape[0] > depth:
            volume = torch.nn.functional.interpolate(volume[None, None, ...], (depth, volume.shape[1], volume.shape[2])).squeeze()
        return volume.to(torch.float32)

    def for_nfn(self, study_id):
        depth = 32
        meta = self.meta.loc[(self.meta.study_id==study_id) & (self.meta.series_description=='Sagittal T1')]
        meta = meta.sort_values('ipp_x', ascending=True).reset_index(drop=True)
        img = [self.load_dicom(IMAGE_PATH + f'{row.study_id}/{row.series_id}/{row.instance_number}.dcm') for _, row in meta.iterrows()]
        volume = self.normalize(torch.cat([self.resize(torch.tensor(i.astype(np.float32))[None, ...]).to(torch.float32) for i in img]).contiguous())
        if volume.shape[0] < depth:
            volume = torch.cat([volume, torch.zeros(depth-volume.shape[0], volume.shape[1], volume.shape[2])])
        elif volume.shape[0] > depth:
            volume = torch.nn.functional.interpolate(volume[None, None, ...], (depth, volume.shape[1], volume.shape[2])).squeeze()
        return volume.to(torch.float32)
    
    def normalize(self, x):
        upper = torch.quantile(x, torch.tensor([0.99]))
        lower = torch.quantile(x, torch.tensor([0.01]))
        x = torch.clip(x, lower, upper)
        x = x - torch.min(x)
        x = x / (torch.max(x)+1e-6)
        return x

    def __len__(self):
        return len(self.id)

    def load_dicom(self, path):
        dicom = dcm.dcmread(path)
        data = dicom.pixel_array
        return data


demo_path = '/kaggle/working/meta.parquet'
meta_df = pd.read_parquet(demo_path)
dataset_demo = DepthDetectDataset(meta=meta_df, condition = 'nfn')
print(len(dataset_demo))
dataset_demo.__getitem__(0)


from torchvision.ops import StochasticDepth 
from typing import List, Dict
from torch import Tensor


# Stem block: để sử lý đầu vào ảnh 3D và chuẩn hoá
# 3D Conv - dùng kernel_size [D, H, W] - (1, 2, 2) - giữ nguyên depth, chỉ tích chập trên H, W. Tương tự với stride
# Output h/2, w/2 với stride = 2
class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features : int):
        super().__init__(
            nn.Conv3d(in_features, out_features, kernel_size = (1, 2, 2), stride = (1, 2, 2)),
            nn.GroupNorm(num_groups = 1, num_channels = out_features)
        )
        
# Tránh vanish: nhân hệ số x vơi gamma (learnable scaling) học được
class LayerScaler(nn.Module):
    def __init__(self, init_value : float, dimensions : int):
        super.__init__()
        self.gamma = nn.Parameter(init_value * torch.ones((dimensions)), requires_grad = True)

    def forward(self, x):
        return self.gamma[None, ..., None, None]* x

# Residual bottleneck block trong các kiến trúc hiện đại: 
# Học được nhiều đặc trưng hơn thông qua: expansion, ...
class BottleNeckBlock(nn.Module):
    def __init__(
        self, 
        in_features : int,
        out_features: int,
        expansion : int = 4,
        drop_p : float = .0,
        layer_scaler_init_value : float = 1e-6,
    ):
        super().__init__()
        expanded_features = out_features * expansion 
        
        # 3 lần Conv để lấy được nhiều feature đặc trưng hơn
        self.block = nn.Sequential(
            nn.Conv3d(
                in_features, in_features, kernel_size = (2, 7, 7), padding = 'same', bias = False, groups = in_features
            ),

            # chuẩn hoá theo channel
            nn.GroupNorm(num_groups = in_features, num_channels = in_features),
            nn.Conv3d(in_features, expanded_features, kernel_size = 1),
            nn.GELU(),
            nn.Conv3d(expanded_features, out_features, kernel_size = 1),
        )

    def forward(self, x : Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x += res
        return x


# tăng channel không đồng nghĩa với học được nhiều feature hữu ích
class ConvNexStage(nn.Sequential):
    def __init__(
        self, in_features: int, out_features: int, depth: int, **kwargs
    ):
        super().__init__(
            # Downsampler giảm độ phân giải không gian (D, H, W), tăng số lượng channel.
            nn.Sequential(
                nn.GroupNorm(num_groups=in_features, num_channels=in_features),
                nn.Conv3d(in_features, out_features, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            ),
            # BottleNeckBlock: áp dụng nhiều khối tích chập (Conv3D) nâng cao biểu diễn đặc trưng.
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
        drop_p: float = .0, # regularization
    ):
        super().__init__()
        # giảm khích thước ảnh đầu vào (spatial downsampling), tăng số lượng features - channel
        self.stem = ConvNextStem(in_channels, stem_features)

        in_out_widths = list(zip(widths, widths[1:]))
        # create drop paths probabilities (one for each stage)
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))]

        # sử dụng các NexStage để trích xuất được feature trừu tượng
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
        # qua stem để tiền sử lý
        x = self.stem(x)

        #rồi đi qua từng stage để extract
        for stage in self.stages:
            x = stage(x)
        return x


class ClassificationHead(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.AdaptiveAvgPool3d((1, 1, 1)), # (B, C, 1, 1, 1)
            nn.Flatten(1), # (B, C)
            nn.LayerNorm(512),
            nn.Linear(512, 3)
        )
class Flatten(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(1),
            nn.LayerNorm(512)
        )


# #test phase
# x = torch.randn(4, 512, 32, 384, 384)
# test = ClassificationHead()
# print(test(x).shape) 
# run to long must to gpu
x = torch.randn(4, 512, 8, 8, 8)
test = Flatten()
print(test(x).shape) 


# dự đoán độ sâu của đốt sống bên trái và bên phải
class ConvNextSSDepthDetect(nn.Module):
    def __init__(self):
        super().__init__()
        # Extract feature từ 3D ConvNextEncoder 
        self.encoder = ConvNextEncoder(in_channels=1, stem_features=32, depths=[3,3,9,3], widths=[64, 128, 256, 512])
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)),
                                    nn.Flatten(1),
                                    nn.LayerNorm(512))
        # Output là 10 branches: mỗi branch là 1 đoạn đốt sống
        self.ll1 = nn.Linear(512, 96)
        self.ll2 = nn.Linear(512, 96)
        self.ll3 = nn.Linear(512, 96)
        self.ll4 = nn.Linear(512, 96)
        self.ll5 = nn.Linear(512, 96)
        self.rl1 = nn.Linear(512, 96)
        self.rl2 = nn.Linear(512, 96)
        self.rl3 = nn.Linear(512, 96)
        self.rl4 = nn.Linear(512, 96)
        self.rl5 = nn.Linear(512, 96)
        # trả về 10 keys: mỗi key là tensor [B, 96]
    def forward(self, x, label=None):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.flatten(x)
        ll1 = self.ll1(x)
        ll2 = self.ll2(x)
        ll3 = self.ll3(x)
        ll4 = self.ll4(x)
        ll5 = self.ll5(x)
        rl1 = self.rl1(x)
        rl2 = self.rl2(x)
        rl3 = self.rl3(x)
        rl4 = self.rl4(x)
        rl5 = self.rl5(x)
        # [B, 96]
        return {'left_L1/L2': ll1,'left_L2/L3': ll2,'left_L3/L4': ll3, 'left_L4/L5': ll4, 'left_L5/S1': ll5,
                'right_L1/L2': rl1, 'right_L2/L3': rl2, 'right_L3/L4': rl3, 'right_L4/L5': rl4, 'right_L5/S1': rl5}


# Bổ sung thông tin về vị trí trong chuỗi vào embedding vector
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # tạo vector position từ 0 - max_len-1
        position = torch.arange(max_len).unsqueeze(1)
        
        # tấn suất sin/cos 
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Ánh xạ vào sin cos
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # lưu pe và buffer 
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Thêm encoding vào x. Sau đó chuyển lại về [B, T, D]
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x.permute(1, 0, 2))



# pipeline 3 usepretrain model 
class AttentionSSDepthDetect(nn.Module):
    def __init__(self):
        super().__init__()
        # load pretrained model ConvNeXt-base
        self.encoder = timm.create_model('convnext_base.fb_in22k_ft_in1k_384', in_chans=3, pretrained=True, num_classes=0)
        
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(1),
                                    nn.LayerNorm(512))
        
        self.in_features = self.encoder.num_features

        # tạo position encoding dạng sin cos
        self.rpe = PositionalEncoding(self.in_features, dropout=0., max_len=64)
        
        self.transformer0 = nn.TransformerEncoderLayer(d_model=self.in_features, nhead=8, activation='gelu',dropout=0.1, batch_first=True)
        self.transformer1 = nn.TransformerEncoderLayer(d_model=self.in_features, nhead=8, activation='gelu',dropout=0.1, batch_first=True)

        # L1/L2 -> L5/S1
        self.ll1 = nn.Linear(512, 64)
        self.ll2 = nn.Linear(512, 64)
        self.ll3 = nn.Linear(512, 64)
        self.ll4 = nn.Linear(512, 64)
        self.ll5 = nn.Linear(512, 64)
        self.rl1 = nn.Linear(512, 64)
        self.rl2 = nn.Linear(512, 64)
        self.rl3 = nn.Linear(512, 64)
        self.rl4 = nn.Linear(512, 64)
        self.rl5 = nn.Linear(512, 64)
    def forward(self, x, label=None):
        # Đảm bảo là x có 3 kênh đầu vào, nếu chỉ có 1 thì thêm 1 dims vào
        # hơi dư thừa nếu về mặt Conv2D
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.flatten(x)
        ll1 = self.ll1(x) # [B, 64]
        ll2 = self.ll2(x)
        ll3 = self.ll3(x)
        ll4 = self.ll4(x)
        ll5 = self.ll5(x)
        rl1 = self.rl1(x)
        rl2 = self.rl2(x)
        rl3 = self.rl3(x)
        rl4 = self.rl4(x)
        rl5 = self.rl5(x)

        # Trả về dictionary chứa 10 output embeddings
        return {'left_L1/L2': ll1,'left_L2/L3': ll2,'left_L3/L4': ll3, 'left_L4/L5': ll4, 'left_L5/S1': ll5,
                'right_L1/L2': rl1, 'right_L2/L3': rl2, 'right_L3/L4': rl3, 'right_L4/L5': rl4, 'right_L5/S1': rl5}


# ConvNeXtNFNDepthDetect sử dụng backbone của ConvNeXt 3D để phát hiện bệnh nfn cột sống ở thắt lưng
class ConvNextNFNDepthDetect(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone (encoder)
        # Chức năng chính là trích xuất các đặc trưng từ 1 ảnh - volumn 3D MRI kích thước [B, 1, D, H, W]
        self.encoder = ConvNextEncoder(in_channels=1, stem_features=32, depths=[3,3,9,3], widths=[64, 128, 256, 512])
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)),
                                    nn.Flatten(1),
                                    nn.LayerNorm(512))
        # left l1 -> l5
        self.ll1 = nn.Linear(512, 32)
        self.ll2 = nn.Linear(512, 32)
        self.ll3 = nn.Linear(512, 32)
        self.ll4 = nn.Linear(512, 32)
        self.ll5 = nn.Linear(512, 32)

        # right l1 -> l5
        self.rl1 = nn.Linear(512, 32)
        self.rl2 = nn.Linear(512, 32)
        self.rl3 = nn.Linear(512, 32)
        self.rl4 = nn.Linear(512, 32)
        self.rl5 = nn.Linear(512, 32)
    def forward(self, x, label=None):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.flatten(x)
        ll1 = self.ll1(x)
        ll2 = self.ll2(x)
        ll3 = self.ll3(x)
        ll4 = self.ll4(x)
        ll5 = self.ll5(x)
        rl1 = self.rl1(x)
        rl2 = self.rl2(x)
        rl3 = self.rl3(x)
        rl4 = self.rl4(x)
        rl5 = self.rl5(x)
        # 10 Linear header dự đoán đặc trung cho từng kh đốt sống
        return {'left_L1/L2': ll1,'left_L2/L3': ll2,'left_L3/L4': ll3, 'left_L4/L5': ll4, 'left_L5/S1': ll5,
                'right_L1/L2': rl1, 'right_L2/L3': rl2, 'right_L3/L4': rl3, 'right_L4/L5': rl4, 'right_L5/S1': rl5}

class ConvNextNFNDepthDetect(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels=1, stem_features=widths[0]//2, depths=[3,3,9,3], widths=widths)
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)),
                                    nn.Flatten(1),
                                    nn.LayerNorm(widths[-1]))
        self.ll1 = nn.Linear(widths[-1], 32)
        self.ll2 = nn.Linear(widths[-1], 32)
        self.ll3 = nn.Linear(widths[-1], 32)
        self.ll4 = nn.Linear(widths[-1], 32)
        self.ll5 = nn.Linear(widths[-1], 32)
        self.rl1 = nn.Linear(widths[-1], 32)
        self.rl2 = nn.Linear(widths[-1], 32)
        self.rl3 = nn.Linear(widths[-1], 32)
        self.rl4 = nn.Linear(widths[-1], 32)
        self.rl5 = nn.Linear(widths[-1], 32)
    def forward(self, x, label=None):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.flatten(x)
        ll1 = self.ll1(x)
        ll2 = self.ll2(x)
        ll3 = self.ll3(x)
        ll4 = self.ll4(x)
        ll5 = self.ll5(x)
        rl1 = self.rl1(x)
        rl2 = self.rl2(x)
        rl3 = self.rl3(x)
        rl4 = self.rl4(x)
        rl5 = self.rl5(x)
        return {'left_L1/L2': ll1,'left_L2/L3': ll2,'left_L3/L4': ll3, 'left_L4/L5': ll4, 'left_L5/S1': ll5,
                'right_L1/L2': rl1, 'right_L2/L3': rl2, 'right_L3/L4': rl3, 'right_L4/L5': rl4, 'right_L5/S1': rl5}


class ConvNextSCSDepthDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels=1, stem_features=32, depths=[3,3,9,3], widths=[64, 128, 256, 512])
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)),
                                        nn.Flatten(1),
                                        nn.LayerNorm(512))
        self.l1 = nn.Linear(512, 32)
        self.l2 = nn.Linear(512, 32)
        self.l3 = nn.Linear(512, 32)
        self.l4 = nn.Linear(512, 32)
        self.l5 = nn.Linear(512, 32)
    def forward(self, x, label = None):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        
        return {'L1/L2': l1, 'L2/L3': l2, 'L3/L4': l3, 'L4/L5': l4, 'L5/S1': l5}
        
class ConvNextSCSDepthDetect(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels=1, stem_features=widths[0]//2, depths=[3,3,9,3], widths=widths)
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)),
                                        nn.Flatten(1),
                                        nn.LayerNorm(512))
        self.l1 = nn.Linear(widths[-1], 32)
        self.l2 = nn.Linear(widths[-1], 32)
        self.l3 = nn.Linear(widths[-1], 32)
        self.l4 = nn.Linear(widths[-1], 32)
        self.l5 = nn.Linear(widths[-1], 32)
    def forward(self, x, label = None):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        
        return {'L1/L2': l1, 'L2/L3': l2, 'L3/L4': l3, 'L4/L5': l4, 'L5/S1': l5}


class RegConvNextNFNDepthDetect(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels=1, stem_features=widths[0]//2, depths=[3,3,9,3], widths=widths)
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)),
                                    nn.Flatten(1),
                                    nn.LayerNorm(1024))
        self.ll1 = nn.Linear(1024, 3)
        self.ll2 = nn.Linear(1024, 3)
        self.ll3 = nn.Linear(1024, 3)
        self.ll4 = nn.Linear(1024, 3)
        self.ll5 = nn.Linear(1024, 3)
        self.rl1 = nn.Linear(1024, 3)
        self.rl2 = nn.Linear(1024, 3)
        self.rl3 = nn.Linear(1024, 3)
        self.rl4 = nn.Linear(1024, 3)
        self.rl5 = nn.Linear(1024, 3)
    def forward(self, x, label=None):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.flatten(x)
        ll1 = self.ll1(x).sigmoid()
        ll2 = self.ll2(x).sigmoid()
        ll3 = self.ll3(x).sigmoid()
        ll4 = self.ll4(x).sigmoid()
        ll5 = self.ll5(x).sigmoid()
        rl1 = self.rl1(x).sigmoid()
        rl2 = self.rl2(x).sigmoid()
        rl3 = self.rl3(x).sigmoid()
        rl4 = self.rl4(x).sigmoid()
        rl5 = self.rl5(x).sigmoid()
        return {'left_L1/L2': ll1,'left_L2/L3': ll2,'left_L3/L4': ll3, 'left_L4/L5': ll4, 'left_L5/S1': ll5,
                'right_L1/L2': rl1, 'right_L2/L3': rl2, 'right_L3/L4': rl3, 'right_L4/L5': rl4, 'right_L5/S1': rl5}


class RegConvNextSCSDepthDetect(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels=1, stem_features=widths[0]//2, depths=[3,3,9,3], widths=widths)
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
        l1 = self.l1(x).sigmoid()
        l2 = self.l2(x).sigmoid()
        l3 = self.l3(x).sigmoid()
        l4 = self.l4(x).sigmoid()
        l5 = self.l5(x).sigmoid()
        return {'L1/L2': l1, 'L2/L3': l2, 'L3/L4': l3, 'L4/L5': l4, 'L5/S1': l5}


# Base Models


class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            nn.Conv3d(in_features, out_features, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GroupNorm(num_groups=1, num_channels=out_features)
        )

class LayerScaler(nn.Module):
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones((dimensions)),
                                    requires_grad=True)

    def forward(self, x):
        return self.gamma[None,...,None,None] * x

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

class ClassificationHead(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(1),
            nn.LayerNorm(512),
            nn.Linear(512, 3)
        )
class Flatten(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(1),
            nn.LayerNorm(512)
        )


class ConvNextSSDepthDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels=1, stem_features=32, depths=[3,3,9,3], widths=[64, 128, 256, 512])
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)),
                                    nn.Flatten(1),
                                    nn.LayerNorm(512))
        self.ll1 = nn.Linear(512, 96)
        self.ll2 = nn.Linear(512, 96)
        self.ll3 = nn.Linear(512, 96)
        self.ll4 = nn.Linear(512, 96)
        self.ll5 = nn.Linear(512, 96)
        self.rl1 = nn.Linear(512, 96)
        self.rl2 = nn.Linear(512, 96)
        self.rl3 = nn.Linear(512, 96)
        self.rl4 = nn.Linear(512, 96)
        self.rl5 = nn.Linear(512, 96)
    def forward(self, x, label=None):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.flatten(x)
        ll1 = self.ll1(x)
        ll2 = self.ll2(x)
        ll3 = self.ll3(x)
        ll4 = self.ll4(x)
        ll5 = self.ll5(x)
        rl1 = self.rl1(x)
        rl2 = self.rl2(x)
        rl3 = self.rl3(x)
        rl4 = self.rl4(x)
        rl5 = self.rl5(x)
        return {'left_L1/L2': ll1,'left_L2/L3': ll2,'left_L3/L4': ll3, 'left_L4/L5': ll4, 'left_L5/S1': ll5,
                'right_L1/L2': rl1, 'right_L2/L3': rl2, 'right_L3/L4': rl3, 'right_L4/L5': rl4, 'right_L5/S1': rl5}
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x.permute(1, 0, 2))

class AttentionSSDepthDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model('convnext_base.fb_in22k_ft_in1k_384', in_chans=3, pretrained=True, num_classes=0)
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(1),
                                    nn.LayerNorm(512))
        self.in_features = self.encoder.num_features
        self.rpe = PositionalEncoding(self.in_features, dropout=0., max_len=64)
        self.transformer0 = nn.TransformerEncoderLayer(d_model=self.in_features, nhead=8, activation='gelu',dropout=0.1, batch_first=True)
        self.transformer1 = nn.TransformerEncoderLayer(d_model=self.in_features, nhead=8, activation='gelu',dropout=0.1, batch_first=True)

        self.ll1 = nn.Linear(512, 64)
        self.ll2 = nn.Linear(512, 64)
        self.ll3 = nn.Linear(512, 64)
        self.ll4 = nn.Linear(512, 64)
        self.ll5 = nn.Linear(512, 64)
        self.rl1 = nn.Linear(512, 64)
        self.rl2 = nn.Linear(512, 64)
        self.rl3 = nn.Linear(512, 64)
        self.rl4 = nn.Linear(512, 64)
        self.rl5 = nn.Linear(512, 64)
    def forward(self, x, label=None):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.flatten(x)
        ll1 = self.ll1(x)
        ll2 = self.ll2(x)
        ll3 = self.ll3(x)
        ll4 = self.ll4(x)
        ll5 = self.ll5(x)
        rl1 = self.rl1(x)
        rl2 = self.rl2(x)
        rl3 = self.rl3(x)
        rl4 = self.rl4(x)
        rl5 = self.rl5(x)
        return {'left_L1/L2': ll1,'left_L2/L3': ll2,'left_L3/L4': ll3, 'left_L4/L5': ll4, 'left_L5/S1': ll5,
                'right_L1/L2': rl1, 'right_L2/L3': rl2, 'right_L3/L4': rl3, 'right_L4/L5': rl4, 'right_L5/S1': rl5}

class ConvNextNFNDepthDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels=1, stem_features=32, depths=[3,3,9,3], widths=[64, 128, 256, 512])
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)),
                                    nn.Flatten(1),
                                    nn.LayerNorm(512))
        self.ll1 = nn.Linear(512, 32)
        self.ll2 = nn.Linear(512, 32)
        self.ll3 = nn.Linear(512, 32)
        self.ll4 = nn.Linear(512, 32)
        self.ll5 = nn.Linear(512, 32)
        self.rl1 = nn.Linear(512, 32)
        self.rl2 = nn.Linear(512, 32)
        self.rl3 = nn.Linear(512, 32)
        self.rl4 = nn.Linear(512, 32)
        self.rl5 = nn.Linear(512, 32)
    def forward(self, x, label=None):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.flatten(x)
        ll1 = self.ll1(x)
        ll2 = self.ll2(x)
        ll3 = self.ll3(x)
        ll4 = self.ll4(x)
        ll5 = self.ll5(x)
        rl1 = self.rl1(x)
        rl2 = self.rl2(x)
        rl3 = self.rl3(x)
        rl4 = self.rl4(x)
        rl5 = self.rl5(x)
        return {'left_L1/L2': ll1,'left_L2/L3': ll2,'left_L3/L4': ll3, 'left_L4/L5': ll4, 'left_L5/S1': ll5,
                'right_L1/L2': rl1, 'right_L2/L3': rl2, 'right_L3/L4': rl3, 'right_L4/L5': rl4, 'right_L5/S1': rl5}

class ConvNextSCSDepthDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels=1, stem_features=32, depths=[3,3,9,3], widths=[64, 128, 256, 512])
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)),
                                    nn.Flatten(1),
                                    nn.LayerNorm(512))
        self.l1 = nn.Linear(512, 32)
        self.l2 = nn.Linear(512, 32)
        self.l3 = nn.Linear(512, 32)
        self.l4 = nn.Linear(512, 32)
        self.l5 = nn.Linear(512, 32)
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
    
class ConvNextSCSDepthDetect(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels=1, stem_features=widths[0]//2, depths=[3,3,9,3], widths=widths)
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)),
                                    nn.Flatten(1),
                                    nn.LayerNorm(widths[-1]),
                                     )
        self.l1 = nn.Linear(widths[-1], 32)
        self.l2 = nn.Linear(widths[-1], 32)
        self.l3 = nn.Linear(widths[-1], 32)
        self.l4 = nn.Linear(widths[-1], 32)
        self.l5 = nn.Linear(widths[-1], 32)
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

class ConvNextNFNDepthDetect(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels=1, stem_features=widths[0]//2, depths=[3,3,9,3], widths=widths)
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)),
                                    nn.Flatten(1),
                                    nn.LayerNorm(widths[-1]))
        self.ll1 = nn.Linear(widths[-1], 32)
        self.ll2 = nn.Linear(widths[-1], 32)
        self.ll3 = nn.Linear(widths[-1], 32)
        self.ll4 = nn.Linear(widths[-1], 32)
        self.ll5 = nn.Linear(widths[-1], 32)
        self.rl1 = nn.Linear(widths[-1], 32)
        self.rl2 = nn.Linear(widths[-1], 32)
        self.rl3 = nn.Linear(widths[-1], 32)
        self.rl4 = nn.Linear(widths[-1], 32)
        self.rl5 = nn.Linear(widths[-1], 32)
    def forward(self, x, label=None):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.flatten(x)
        ll1 = self.ll1(x)
        ll2 = self.ll2(x)
        ll3 = self.ll3(x)
        ll4 = self.ll4(x)
        ll5 = self.ll5(x)
        rl1 = self.rl1(x)
        rl2 = self.rl2(x)
        rl3 = self.rl3(x)
        rl4 = self.rl4(x)
        rl5 = self.rl5(x)
        return {'left_L1/L2': ll1,'left_L2/L3': ll2,'left_L3/L4': ll3, 'left_L4/L5': ll4, 'left_L5/S1': ll5,
                'right_L1/L2': rl1, 'right_L2/L3': rl2, 'right_L3/L4': rl3, 'right_L4/L5': rl4, 'right_L5/S1': rl5}
    
class RegConvNextNFNDepthDetect(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels=1, stem_features=widths[0]//2, depths=[3,3,9,3], widths=widths)
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)),
                                    nn.Flatten(1),
                                    nn.LayerNorm(1024))
        self.ll1 = nn.Linear(1024, 3)
        self.ll2 = nn.Linear(1024, 3)
        self.ll3 = nn.Linear(1024, 3)
        self.ll4 = nn.Linear(1024, 3)
        self.ll5 = nn.Linear(1024, 3)
        self.rl1 = nn.Linear(1024, 3)
        self.rl2 = nn.Linear(1024, 3)
        self.rl3 = nn.Linear(1024, 3)
        self.rl4 = nn.Linear(1024, 3)
        self.rl5 = nn.Linear(1024, 3)
    def forward(self, x, label=None):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.flatten(x)
        ll1 = self.ll1(x).sigmoid()
        ll2 = self.ll2(x).sigmoid()
        ll3 = self.ll3(x).sigmoid()
        ll4 = self.ll4(x).sigmoid()
        ll5 = self.ll5(x).sigmoid()
        rl1 = self.rl1(x).sigmoid()
        rl2 = self.rl2(x).sigmoid()
        rl3 = self.rl3(x).sigmoid()
        rl4 = self.rl4(x).sigmoid()
        rl5 = self.rl5(x).sigmoid()
        return {'left_L1/L2': ll1,'left_L2/L3': ll2,'left_L3/L4': ll3, 'left_L4/L5': ll4, 'left_L5/S1': ll5,
                'right_L1/L2': rl1, 'right_L2/L3': rl2, 'right_L3/L4': rl3, 'right_L4/L5': rl4, 'right_L5/S1': rl5}

class RegConvNextSCSDepthDetect(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels=1, stem_features=widths[0]//2, depths=[3,3,9,3], widths=widths)
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
        l1 = self.l1(x).sigmoid()
        l2 = self.l2(x).sigmoid()
        l3 = self.l3(x).sigmoid()
        l4 = self.l4(x).sigmoid()
        l5 = self.l5(x).sigmoid()
        return {'L1/L2': l1, 'L2/L3': l2, 'L3/L4': l3, 'L4/L5': l4, 'L5/S1': l5}


# Là module wrapper cho các model khác, sử dụng pytorch_lightning 
class DepthDetectModule(pl.LightningModule):
    def __init__(self, condition, widths=None, model_type='regression'):
        super().__init__()
        self.config = config
        if condition == 'scs':
            if model_type == 'regression': 
                self.model = RegConvNextSCSDepthDetect(widths)
            else: 
                self.model = ConvNextSCSDepthDetect(widths)
        elif condition == 'nfn':
            if model_type == 'regression': 
                self.model = RegConvNextNFNDepthDetect(widths)
            else: 
                self.model = ConvNextNFNDepthDetect(widths)
    def forward(self, batch):
        preds = self.model(batch)
        return preds


%%time
prefix = ''
import warnings
warnings.filterwarnings("ignore")
depth_predict = {'scs': {
                     'L1/L2':[], 
                     'L2/L3': [], 
                     'L3/L4': [], 
                     'L4/L5': [], 
                     'L5/S1': []
                     }, 
                 'nfn': {
                     'left_L1/L2': [], 
                     'left_L2/L3': [], 
                     'left_L3/L4': [], 
                     'left_L4/L5': [], 
                     'left_L5/S1': [], 
                     'right_L1/L2': [], 
                     'right_L2/L3': [], 
                     'right_L3/L4': [], 
                     'right_L4/L5': [], 
                     'right_L5/S1': [], 
                     }
                    }
model_path_dict = {
    'scs': [
        '/kaggle/input/rsna-spine-final-models/scs_depth_1024_ssr_0.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_depth_1024_ssr_1.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_depth_1024_ssr_2.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_depth_1024_ssr_3.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_depth_1024_ssr_4.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_depth_0.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_depth_1.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_depth_2.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_depth_3.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_depth_4.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_depth_1024_ssr_l1_0.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_depth_1024_ssr_l1_1.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_depth_1024_ssr_l1_2.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_depth_1024_ssr_l1_3.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_depth_1024_ssr_l1_4.ckpt', 
    ], 
    'nfn': [
        '/kaggle/input/rsna-spine-final-models/nfn_depth_1024_ssr_0.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_depth_1024_ssr_1.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_depth_1024_ssr_2.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_depth_1024_ssr_3.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_depth_1024_ssr_4.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_depth_0.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_depth_1.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_depth_2.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_depth_3.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_depth_4.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_depth_1024_ssr_l1_0.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_depth_1024_ssr_l1_1.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_depth_1024_ssr_l1_2.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_depth_1024_ssr_l1_3.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_depth_1024_ssr_l1_4.ckpt', 
    ]
}
##############DEPTH DETECT#########################
for condition in ['nfn', 'scs']:
    print(condition)
    model_path_list = model_path_dict[condition]
    for model_path in model_path_list:
        _meta_df = meta_df.copy()
        #_series = series.copy()
        dataset_test = DepthDetectDataset(_meta_df, condition, 'sub')
        data_loader_test = DataLoader(
            dataset_test,
            batch_size=config["test_bs"], 
            shuffle=False,
            num_workers=4,
            pin_memory=False
        )
        model_name = model_path.split('/')[-1]
        if '1024' in model_name: 
            widths = [128, 256, 512, 1024]
        else: 
            widths = [64, 128, 256, 512]
        if 'l1' in model_name: 
            model_type = 'regression'
        else: 
            model_type = 'classification'
        model = DepthDetectModule.load_from_checkpoint(model_path, condition=condition, widths=widths, model_type=model_type)
        model.eval()
        model.zero_grad()
        model.to(device)

        pred_temp = {}
        for k in depth_predict[condition].keys(): 
            pred_temp[k] = []
        study_id_list = []
        with torch.no_grad():
            for data in tqdm(data_loader_test, total=len(data_loader_test)):
                images, study_id = data
                images = images.to(device)
                preds = model.forward(images)
                #print(preds)
                if model_type == 'regression': 
                    for k, v in preds.items(): 
                        pred_temp[k].append((v[:, -1]*32).to('cpu').detach().numpy())
                else: 
                    for k, v in preds.items(): 
                        pred_temp[k].append(torch.argmax(v, dim=1).to('cpu').detach().numpy())
                study_id_list.append(study_id.to('cpu').reshape(-1).detach().numpy())
                del images, study_id, preds
                gc.collect()
        for k, v in pred_temp.items(): 
            depth_predict[condition][k].append(np.concatenate(v))
        study_id = np.concatenate(study_id_list)
        del pred_temp, study_id_list
        gc.collect()
        
    for k, v in depth_predict[condition].items(): 
        depth_predict[condition][k] = np.median(np.array(depth_predict[condition][k]), axis=0)
    depth_predict[condition]['study_id'] = study_id
    del study_id
    gc.collect()


def create_label_ins(study_id, depth, level, condition, desc): 
    coor_dict = {'study_id': [], 'series_id': [], 'instance_number': []}
    _meta = meta_df.loc[meta_df.series_description==desc]
    for s, d in zip(study_id, depth): 
        sub_meta = _meta.loc[_meta.study_id==s]
        sub_meta = sub_meta.sort_values('ipp_x', ascending=True).reset_index(drop=True)
        if len(sub_meta) > 32: 
            d = (d/32)*len(sub_meta)
        try: 
            row = sub_meta.iloc[round(d)]
        except: 
            if condition == 'Spinal Canal Stenosis': 
                row = sub_meta.iloc[int(len(sub_meta)//2)]
            elif condition == 'Left Neural Foraminal Narrowing': 
                row = sub_meta.iloc[int(2*(len(sub_meta)//3))]
            elif condition == 'Right Neural Foraminal Narrowing': 
                row = sub_meta.iloc[int(len(sub_meta)//3)]
            print(s)
        coor_dict['study_id'].append(s)
        coor_dict['series_id'].append(row.series_id)
        coor_dict['instance_number'].append(row.instance_number)
    coor_dict['condition'] = condition
    coor_dict['level'] = level.split('_')[-1]
    return pd.DataFrame(coor_dict)

scs_study_id = depth_predict['scs']['study_id']
scs_coor_list = []
for k, v in depth_predict['scs'].items(): 
    if k != 'study_id': 
        scs_coor_list.append(create_label_ins(scs_study_id, v, k, 'Spinal Canal Stenosis', 'Sagittal T2/STIR'))

nfn_study_id = depth_predict['nfn']['study_id']
nfn_coor_list = []
for k, v in depth_predict['nfn'].items(): 
    if k != 'study_id': 
        if k.split('_')[0] == 'left': 
            condition = 'Left Neural Foraminal Narrowing'
        else: 
            condition = 'Right Neural Foraminal Narrowing'
        nfn_coor_list.append(create_label_ins(nfn_study_id, v, k, condition, 'Sagittal T1'))
scs_coor = pd.concat(scs_coor_list)
nfn_coor = pd.concat(nfn_coor_list)
pred_coor = pd.concat([scs_coor, nfn_coor]).sort_values(['study_id', 'series_id', 'level'])


del scs_coor, nfn_coor, scs_coor_list, nfn_coor_list, depth_predict
gc.collect()
pred_coor.head()
pred_coor.to_csv('stage1_coor.csv', index=False)
os.listdir('/kaggle/working/')


stage1 = pd.read_csv('stage1_coor.csv')
stage1


class CoorDetectDataset(Dataset):
    def __init__(self, coor, meta, condition, usage='train'):
        if condition == 'scs':
            coor = coor.loc[coor.condition=='Spinal Canal Stenosis']
        elif condition == 'ss':
            coor = coor.loc[(coor.condition=='Left Subarticular Stenosis') | (coor.condition=='Right Subarticular Stenosis')]
        elif condition == 'nfn':
            coor = coor.loc[(coor.condition=='Right Neural Foraminal Narrowing') | (coor.condition=='Left Neural Foraminal Narrowing')]
        #g_coor = coor.groupby('study_id').count()
        #if condition == 'scs':
        #    self.id = g_coor.loc[g_coor.series_id==5].reset_index().study_id.unique()
        #else:
        #    self.id = g_coor.loc[g_coor.series_id==10].reset_index().study_id.unique()
        self.id = coor.study_id.unique()
        self.coor = coor
        self.meta = meta
        self.condition = condition
        self.usage = usage
        if 3637444890 in self.id: 
            self.id.remove(3637444890)
        #self.id = [2773343225]
        #self.id = [1782095928]

        self.resize = v2.Resize((384, 384))
        
    def __getitem__(self, index):
        study_id = self.id[index]
        #print(study_id)
        #try:
        if self.condition == 'scs':
            volume = self.for_scs(study_id)
        elif self.condition == 'nfn':
            volume = self.for_nfn(study_id)
        if self.condition == 'ss':
            volume = self.for_ss(study_id)
        return volume, torch.tensor(study_id)

    def for_scs(self, study_id):
        meta = self.meta.loc[(self.meta.study_id==study_id) & (self.meta.series_description=='Sagittal T2/STIR')]
        meta = meta.sort_values('ipp_x', ascending=True).reset_index(drop=True)
        #img = [self.normalize(self.load_dicom(f'/content/train_images/{row.study_id}/{row.series_id}/{row.instance_number}.dcm')) for _, row in meta.iterrows()]
        coor = self.coor.loc[(self.coor.study_id==study_id) & (self.coor.condition=='Spinal Canal Stenosis')]
        meta_list = []
        for _, row in coor.iterrows():
            series_id, instance_number = row.series_id, row.instance_number
            meta_list.append(meta.loc[(meta.series_id==series_id) & (meta.instance_number==instance_number)])
        sub_meta = pd.concat(meta_list)
        idx = meta.loc[meta.ipp_x == sub_meta.ipp_x.median()].index[0]
        #print(old_idx)
        img_row = meta.iloc[idx]
        before_img_row = meta.iloc[idx-1]
        after_img_row = meta.iloc[idx+1]
        img = self.normalize(self.load_dicom(IMAGE_PATH + f'{img_row.study_id}/{img_row.series_id}/{img_row.instance_number}.dcm'))
        bimg = self.normalize(self.load_dicom(IMAGE_PATH + f'{before_img_row.study_id}/{before_img_row.series_id}/{before_img_row.instance_number}.dcm'))
        aimg = self.normalize(self.load_dicom(IMAGE_PATH + f'{after_img_row.study_id}/{after_img_row.series_id}/{after_img_row.instance_number}.dcm'))
        img = self.resize(torch.tensor(img[None, ...]))
        bimg = self.resize(torch.tensor(bimg[None, ...]))
        aimg = self.resize(torch.tensor(aimg[None, ...]))
        img = torch.cat([bimg, img, aimg]).to(torch.float32)
        return img
    def for_ss(self, study_id):
        meta = self.meta.loc[(self.meta.study_id==study_id) & (self.meta.series_description=='Axial T2')]
        meta = meta.sort_values('ipp_z', ascending=False).reset_index(drop=True)
        img = [self.normalize(self.load_dicom(f'/content/train_images/{row.study_id}/{row.series_id}/{row.instance_number}.dcm')) for _, row in meta.iterrows()]
        coor = self.coor.loc[(self.coor.study_id==study_id)]
        coor_dict = {}
        for _, row in coor.iterrows():
            series_id, instance_number = row.series_id, row.instance_number
            target_row = meta.loc[(meta.series_id==series_id) & (meta.instance_number==instance_number)]
            idx = target_row.index[0]
            #print(row.level, idx, idx/len(img))
            #plt.title(row.level)
            #plt.imshow(img[idx])
            #mask = torch.zeros(img[idx].shape)
            #mask[int(row.y)-10:int((row.y))+10, int(row.x)-10:int((row.x))+10] = 1
            #plt.imshow(mask, alpha=0.5)
            #plt.show()
            height, width = img[idx].shape
            z = idx/depth if len(img) < depth else idx/len(img)
            x = row.x/width
            y = row.y/height
            if row.condition == 'Right Subarticular Stenosis':
                coor_dict['right_' + row.level] = torch.tensor([x, y, z]).to(torch.float32)
            else:
                coor_dict['left_' + row.level] = torch.tensor([x, y, z]).to(torch.float32)
        volume = torch.cat([self.resize(torch.tensor(i)[None, ...]).to(torch.float32) for i in img]).contiguous()
        if volume.shape[0] < depth:
            volume = torch.cat([volume, torch.zeros(depth-volume.shape[0], volume.shape[1], volume.shape[2])])
        elif volume.shape[0] > depth:
            volume = torch.nn.functional.interpolate(volume[None, None, ...], (depth, volume.shape[1], volume.shape[2])).squeeze()
        return volume, coor_dict

    def for_nfn(self, study_id):
        meta = self.meta.loc[(self.meta.study_id==study_id) & (self.meta.series_description=='Sagittal T1')]
        meta = meta.sort_values('ipp_x', ascending=True).reset_index(drop=True)
        #img = [self.normalize(self.load_dicom(f'/content/train_images/{row.study_id}/{row.series_id}/{row.instance_number}.dcm')) for _, row in meta.iterrows()]
        coor = self.coor.loc[(self.coor.study_id==study_id)]
        right_meta_list = []
        left_meta_list = []
        for _, row in coor.iterrows():
            series_id, instance_number = row.series_id, row.instance_number
            if row.condition == 'Right Neural Foraminal Narrowing':
                right_meta_list.append(meta.loc[(meta.series_id==series_id) & (meta.instance_number==instance_number)])
            else: 
                left_meta_list.append(meta.loc[(meta.series_id==series_id) & (meta.instance_number==instance_number)])

        right_sub_meta = pd.concat(right_meta_list)
        left_sub_meta = pd.concat(left_meta_list)
        ridx = meta.loc[meta.ipp_x == right_sub_meta.ipp_x.median()].index[0]
        lidx = meta.loc[meta.ipp_x == left_sub_meta.ipp_x.median()].index[0]
        right_img_row = meta.iloc[min(max(ridx, 0), len(meta)-1)]
        #display(right_img_row)
        right_before_img_row = meta.iloc[min(max(ridx-1, 0), len(meta)-1)]
        rightafter_img_row = meta.iloc[min(max(ridx+1, 0), len(meta)-1)]
        left_img_row = meta.iloc[min(max(lidx, 0), len(meta)-1)]
        left_before_img_row = meta.iloc[min(max(lidx-1, 0), len(meta)-1)]
        leftafter_img_row = meta.iloc[min(max(lidx+1, 0), len(meta)-1)]
        rimg = self.normalize(self.load_dicom(IMAGE_PATH + f'{right_img_row.study_id}/{right_img_row.series_id}/{right_img_row.instance_number}.dcm'))
        rbimg = self.normalize(self.load_dicom(IMAGE_PATH + f'{right_before_img_row.study_id}/{right_before_img_row.series_id}/{right_before_img_row.instance_number}.dcm'))
        raimg = self.normalize(self.load_dicom(IMAGE_PATH + f'{rightafter_img_row.study_id}/{rightafter_img_row.series_id}/{rightafter_img_row.instance_number}.dcm'))
        limg = self.normalize(self.load_dicom(IMAGE_PATH + f'{left_img_row.study_id}/{left_img_row.series_id}/{left_img_row.instance_number}.dcm'))
        lbimg = self.normalize(self.load_dicom(IMAGE_PATH + f'{left_before_img_row.study_id}/{left_before_img_row.series_id}/{left_before_img_row.instance_number}.dcm'))
        laimg = self.normalize(self.load_dicom(IMAGE_PATH + f'{leftafter_img_row.study_id}/{leftafter_img_row.series_id}/{leftafter_img_row.instance_number}.dcm'))
              
        rimg = torch.cat([self.resize(torch.tensor(i)[None, ...]).to(torch.float32) for i in [rbimg, rimg, raimg]])
        limg = torch.cat([self.resize(torch.tensor(i)[None, ...]).to(torch.float32) for i in [lbimg, limg, laimg]])
        img = torch.stack([limg, rimg]).to(torch.float32).contiguous()
        return img

    def normalize(self, x):
        lower, upper = np.percentile(x, (1, 99))
        x = np.clip(x, lower, upper)
        x = x - np.min(x)
        x = x / np.max(x)
        return x

    def __len__(self):
        return len(self.id)

    def load_dicom(self, path):
        dicom = dcm.dcmread(path)
        data = dicom.pixel_array
        return data


class ConvNextSCSDetect(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        #self.size = 384
        if encoder == 'convnext': 
            self.encoder = timm.create_model('convnext_base.fb_in22k_ft_in1k_384', in_chans=3, pretrained=False, num_classes=0)
        elif encoder == 'efficientnetv2-l': 
            self.encoder = timm.create_model('tf_efficientnetv2_l.in21k_ft_in1k', in_chans=3, pretrained=False, num_classes=0, drop_rate=0.)
        self.in_features = self.encoder.num_features
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(1),
                                    #nn.LayerNorm(self.in_features)
                                    )
        self.l1 = nn.Linear(self.in_features, 2)
        self.l2 = nn.Linear(self.in_features, 2)
        self.l3 = nn.Linear(self.in_features, 2)
        self.l4 = nn.Linear(self.in_features, 2)
        self.l5 = nn.Linear(self.in_features, 2)
    def forward(self, x, label=None):
        #for loc, img in x.items():
            #print(img.shape)
        #    img = self.encoder.forward_features(img)
        #    img = self.flatten(img)
        #    x[loc] = img
        x = self.encoder.forward_features(x)
        x = self.flatten(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        l3 = self.l3(x)
        l4 = self.l4(x)
        l5 = self.l5(x)
        return {'L1/L2': l1.sigmoid(), 'L2/L3': l2.sigmoid(), 'L3/L4': l3.sigmoid(), 'L4/L5': l4.sigmoid(), 'L5/S1': l5.sigmoid()}

class ConvNextNFNDetect(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        if encoder == 'convnext': 
            self.encoder = timm.create_model('convnext_base.fb_in22k_ft_in1k_384', in_chans=3, pretrained=False, num_classes=0)
        elif encoder == 'efficientnetv2-l': 
            self.encoder = timm.create_model('tf_efficientnetv2_l.in21k_ft_in1k', in_chans=3, pretrained=False, num_classes=0, drop_rate=0.)
        self.in_features = self.encoder.num_features
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(1),
                                    #nn.LayerNorm(self.in_features)
                                    )
        self.ll1 = nn.Linear(self.in_features, 2)
        self.ll2 = nn.Linear(self.in_features, 2)
        self.ll3 = nn.Linear(self.in_features, 2)
        self.ll4 = nn.Linear(self.in_features, 2)
        self.ll5 = nn.Linear(self.in_features, 2)
        self.rl1 = nn.Linear(self.in_features, 2)
        self.rl2 = nn.Linear(self.in_features, 2)
        self.rl3 = nn.Linear(self.in_features, 2)
        self.rl4 = nn.Linear(self.in_features, 2)
        self.rl5 = nn.Linear(self.in_features, 2)
    def forward(self, x, label=None):
        shape = x.shape
        x = x.reshape(shape[0]*shape[1], 3, shape[-2], shape[-1])
        x = self.encoder.forward_features(x)
        x = self.flatten(x)
        x = x.reshape(shape[0], shape[1], -1)
        x_left = x[:, 0, :]
        x_right = x[:, 1, :]
        ll1 = self.ll1(x_left)
        ll2 = self.ll2(x_left)
        ll3 = self.ll3(x_left)
        ll4 = self.ll4(x_left)
        ll5 = self.ll5(x_left)
        rl1 = self.rl1(x_right)
        rl2 = self.rl2(x_right)
        rl3 = self.rl3(x_right)
        rl4 = self.rl4(x_right)
        rl5 = self.rl5(x_right)
        return {'left_L1/L2': ll1.sigmoid(),'left_L2/L3': ll2.sigmoid(),'left_L3/L4': ll3.sigmoid(), 'left_L4/L5': ll4.sigmoid(), 'left_L5/S1': ll5.sigmoid(),
                'right_L1/L2': rl1.sigmoid(), 'right_L2/L3': rl2.sigmoid(), 'right_L3/L4': rl3.sigmoid(), 'right_L4/L5': rl4.sigmoid(), 'right_L5/S1': rl5.sigmoid()}


class DetectModule(pl.LightningModule):
    def __init__(self, condition, encoder):
        super().__init__()
        self.config = condition
        if condition == 'scs':
            self.model = ConvNextSCSDetect(encoder)
        elif condition == 'nfn':
            self.model = ConvNextNFNDetect(encoder)
        elif  condition == 'ss': 
            pass
        #self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        #self.ema.to(device)

        #self.model = torch.optim.swa_utils.AveragedModel(self.model,
        #                                                 multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))

    def forward(self, batch):
        preds = self.model(batch)
        return preds


%%time
prefix = ''
import warnings
warnings.filterwarnings("ignore")
coor_predict = {'scs': {
                     'L1/L2':[], 
                     'L2/L3': [], 
                     'L3/L4': [], 
                     'L4/L5': [], 
                     'L5/S1': []
                     }, 
                 'nfn': {
                     'left_L1/L2': [], 
                     'left_L2/L3': [], 
                     'left_L3/L4': [], 
                     'left_L4/L5': [], 
                     'left_L5/S1': [], 
                     'right_L1/L2': [], 
                     'right_L2/L3': [], 
                     'right_L3/L4': [], 
                     'right_L4/L5': [], 
                     'right_L5/S1': [], 
                     }
                    }
model_path_dict = {
    'scs': [
        '/kaggle/input/rsna-spine-final-models/scs_detect_pre_0.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_detect_pre_1.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_detect_pre_2.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_detect_pre_3.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_detect_pre_4.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_detect_pre_effv2l_0.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_detect_pre_effv2l_1.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_detect_pre_effv2l_2.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_detect_pre_effv2l_3.ckpt', 
        '/kaggle/input/rsna-spine-final-models/scs_detect_pre_effv2l_4.ckpt', 
    ], 
    'nfn': [
        '/kaggle/input/rsna-spine-final-models/nfn_detect_pre_0.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_detect_pre_1.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_detect_pre_2.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_detect_pre_3.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_detect_pre_4.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_detect_pre_effv2l_0.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_detect_pre_effv2l_1.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_detect_pre_effv2l_2.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_detect_pre_effv2l_3.ckpt', 
        '/kaggle/input/rsna-spine-final-models/nfn_detect_pre_effv2l_4.ckpt', 
    ]
}
##############COOR DETECT#########################
for condition in ['nfn', 'scs']:
    print(condition)
    model_path_list = model_path_dict[condition]
    for path in model_path_list:
        if 'effv2l' in path.split('/')[-1]: 
            encoder = 'efficientnetv2-l'
        else: 
            encoder = 'convnext'
        _meta_df = meta_df.copy()
        _coor = pred_coor.copy()
        dataset_test = CoorDetectDataset(_coor, _meta_df, condition, 'sub')
        data_loader_test = DataLoader(
            dataset_test,
            batch_size=config["test_bs"],
            shuffle=False,
            num_workers=4,
            pin_memory=False
        )
        print(path, encoder)
        model = DetectModule.load_from_checkpoint(path, condition=condition, encoder=encoder)
        model.eval()
        model.zero_grad()
        model.to(device)

        pred_temp = {}
        for k in coor_predict[condition].keys(): 
            pred_temp[k] = []
        study_id_list = []
        with torch.no_grad():
            for data in tqdm(data_loader_test, total=len(data_loader_test)):
                images, study_id = data
                images = images.to(device)
                preds = model.forward(images)
                #print(preds)
                for k, v in preds.items(): 
                    pred_temp[k].append(v.to('cpu').detach().numpy())
                study_id_list.append(study_id.to('cpu').reshape(-1).detach().numpy())
        for k, v in pred_temp.items(): 
            coor_predict[condition][k].append(np.concatenate(v))
        del pred_temp
        gc.collect()
        study_id = np.concatenate(study_id_list)
    for k, v in coor_predict[condition].items(): 
        coor_predict[condition][k] = np.mean(np.array(coor_predict[condition][k]), axis=0)
    coor_predict[condition]['study_id'] = study_id
    del study_id
    gc.collect()


pred_coor.head(1)


%%time
def create_label_coor(study_id, coor_df, coor, level, condition, desc): 
    _meta = meta_df.loc[meta_df.series_description==desc].copy()
    _coor = coor_df.loc[coor_df.condition == condition]
    _coor_df = {'study_id': [], 'series_id': [], 'x': [], 'y': []}
    for s, c in zip(study_id, coor): 
        sub_meta = _meta.loc[_meta.study_id == s]
        sub_coor = _coor.loc[(_coor.study_id==s) & (_coor.level==level.split('_')[-1])].squeeze(axis=0)
        #display(sub_coor)
        meta_row = sub_meta.loc[(sub_meta.instance_number==sub_coor.instance_number) & (sub_meta.series_id==sub_coor.series_id)].squeeze(axis=0)
        x = round(meta_row.width * c[0])
        y = round(meta_row.height * c[1])
        _coor_df['study_id'].append(s)
        _coor_df['series_id'].append(sub_coor.series_id)
        _coor_df['x'].append(x)
        _coor_df['y'].append(y)
    _coor_df['level'] = level.split('_')[-1]
    _coor_df['condition'] = condition
    del _meta, _coor, sub_meta, sub_coor, meta_row
    return pd.DataFrame(_coor_df)

scs_study_id = coor_predict['scs']['study_id']
scs_coor_list = []
for k, v in coor_predict['scs'].items(): 
    if k != 'study_id': 
        scs_coor_list.append(create_label_coor(scs_study_id, pred_coor, v, k, 'Spinal Canal Stenosis', 'Sagittal T2/STIR'))

nfn_study_id = coor_predict['nfn']['study_id']
nfn_coor_list = []
for k, v in coor_predict['nfn'].items(): 
    if k != 'study_id': 
        if k.split('_')[0] == 'left': 
            condition = 'Left Neural Foraminal Narrowing'
        else: 
            condition = 'Right Neural Foraminal Narrowing'
        nfn_coor_list.append(create_label_coor(nfn_study_id, pred_coor, v, k, condition, 'Sagittal T1'))
scs_coor = pd.concat(scs_coor_list)
nfn_coor = pd.concat(nfn_coor_list)
_pred_coor = pd.concat([scs_coor, nfn_coor]).sort_values(['study_id', 'series_id', 'level'])


pred_coor_stage2 = pd.merge(pred_coor, _pred_coor, on=['study_id', 'series_id', 'level', 'condition'], how='inner')


display(pred_coor_stage2.head())
pred_coor_stage2.to_csv('stage2_coor.csv', index=False)


os.listdir('/kaggle/working/')


stage2 = pd.read_csv('stage2_coor.csv')
stage2


def project_to_3d(row):
    sx, sy, sz = row.ipp_x, row.ipp_y, row.ipp_z
    x, y = row.x, row.y
    o0, o1, o2, o3, o4, o5 = row.iop
    delx, dely = row.ps_x, row.ps_y
    xx = o0 * delx * x + o3 * dely * y + sx
    yy = o1 * delx * x + o4 * dely * y + sy
    zz = o2 * delx * x + o5 * dely * y + sz
    return xx,yy,zz

def sag_to_ax(sub_coor, sub_meta): 
    point = sub_coor[['ipp_x', 'ipp_y', 'ipp_z']].values #2d
    level_list = sub_coor.level.tolist()
    # here we project 2d to 3d
    center=[] 
    for _, row in sub_coor.iterrows():
        xx,yy,zz = project_to_3d(row)
        center.append([xx,yy,zz])
    center = np.array(center) #3d

    # == 2. we get closest axial slices to the CSC points =================
    #df = valid_data[0].axial_t2[0].df

    orientation = np.array(sub_meta.iop.values.tolist())
    position= np.array(sub_meta[['ipp_x', 'ipp_y', 'ipp_z']].values.tolist())
    ox = orientation[:, :3]
    oy = orientation[:, 3:]
    oz = np.cross(ox,oy)
    t = center.reshape(-1,1,3) - position.reshape(1,-1,3)
    dis = (oz.reshape(1,-1,3) * t).sum(-1)  # np.dot(point-s,oz)
    dis = np.fabs(dis)
    closest = dis.argmin(-1)
    closest_df = sub_meta.iloc[closest]
    closest_df['level'] = level_list#['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
    closest_df['x'] = 0
    closest_df['y'] = 0
    #closest_df = pd.concat([closest_df, closest_df])
    #closest_df['condition'] = ['Left Subarticular Stenosis']*5 + ['Right Subarticular Stenosis']*5
    return closest_df[['study_id', 'series_id', 'instance_number', 'level']]


# sagittal t2 => axial t2
scs_coor = pred_coor_stage2.loc[pred_coor_stage2.condition=='Spinal Canal Stenosis'].copy()
scs_coor = scs_coor.merge(meta_df, on=['study_id', 'series_id', 'instance_number'], how='left')
study_id = scs_coor.study_id.unique()
ax_meta  = meta_df.loc[(meta_df.series_description=='Axial T2')]
closest_ax_list = []
for s in tqdm(study_id, total=len(study_id)): 
    sub_coor = scs_coor.loc[scs_coor.study_id==s]
    sub_meta = ax_meta.loc[ax_meta.study_id==s]
    closest_ax_list.append(sag_to_ax(sub_coor, sub_meta)) 


closest_ax = pd.concat(closest_ax_list)
closest_ax.head()


class SSDetectDataset(Dataset):
    def __init__(self, ax, usage='train'):
        self.ax = ax
        self.id = ax.study_id.unique()
        self.usage = usage
        self.id = list(set(self.id) - set([3637444890]))
        #self.id = [2773343225]
        #self.id = [1782095928]

        self.resize = v2.Resize((384, 384))
        
    def __getitem__(self, index):
        study_id = self.id[index]
        volume = self.for_ss(study_id)
        return volume, torch.tensor(study_id)

    def for_ss(self, study_id):
        ax = self.ax.loc[self.ax.study_id==study_id]
        img_dict = {}
        for _, row in ax.iterrows():
            series_id, instance_number = row.series_id, row.instance_number
            img = self.load_dicom(IMAGE_PATH + f'{study_id}/{series_id}/{instance_number}.dcm').astype(np.float32)
            img = self.resize(torch.tensor(img)[None, ...])
            img = self.normalize(img)
            img_dict[row.level] = img
        img_list = []
        for k in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']: 
            img_list.append(img_dict[k])
        volume = torch.stack(img_list).contiguous()
        return volume

    def normalize(self, x):
        upper = torch.quantile(x, torch.tensor([0.99]))
        lower = torch.quantile(x, torch.tensor([0.01]))
        x = torch.clip(x, lower, upper)
        x = x - torch.min(x)
        x = x / (torch.max(x)+1e-6)
        return x

    def __len__(self):
        return len(self.id)

    def load_dicom(self, path):
        dicom = dcm.dcmread(path)
        data = dicom.pixel_array
        return data


class SSDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model('convnext_base.fb_in22k_ft_in1k_384', in_chans=1, pretrained=False, num_classes=0)
        self.in_features = self.encoder.num_features
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(1),
                                    #nn.LayerNorm(self.in_features)
                                    )
        self.left = nn.Linear(self.in_features, 2)
        self.right = nn.Linear(self.in_features, 2)
    def forward(self, x, label=None):
        shape = x.shape
        x = x.reshape(shape[0]*shape[1], 1, shape[-2], shape[-1])
        x = self.encoder.forward_features(x)
        x = self.flatten(x)
        x = x.reshape(shape[0], shape[1], -1)
        x_left = x
        x_right = x
        left = self.left(x_left)
        right = self.right(x_right)
        return {'left_L1/L2': left[:, 0, :].sigmoid(),'left_L2/L3': left[:, 1, :].sigmoid(),'left_L3/L4': left[:, 2, :].sigmoid(), 'left_L4/L5': left[:, 3, :].sigmoid(), 'left_L5/S1': left[:, 4, :].sigmoid(),
                'right_L1/L2': right[:, 0, :].sigmoid(), 'right_L2/L3': right[:, 1, :].sigmoid(), 'right_L3/L4': right[:, 2, :].sigmoid(), 'right_L4/L5': right[:, 3, :].sigmoid(), 'right_L5/S1': right[:, 4, :].sigmoid()}


class SSDetectModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SSDetect()
    def forward(self, batch):
        preds = self.model(batch)
        return preds


%%time
prefix = ''
import warnings
warnings.filterwarnings("ignore")
coor_predict = {
             'left_L1/L2': [], 
             'left_L2/L3': [], 
             'left_L3/L4': [], 
             'left_L4/L5': [], 
             'left_L5/S1': [], 
             'right_L1/L2': [], 
             'right_L2/L3': [], 
             'right_L3/L4': [], 
             'right_L4/L5': [], 
             'right_L5/S1': [], 
                }

##############COOR DETECT#########################
for i in [0, 1, 2, 3, 4]:
    _meta_df = meta_df.copy()
    _series = series.copy()
    _coor = pred_coor.copy()
    dataset_test = SSDetectDataset(closest_ax, 'sub')
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=config["test_bs"],
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )

    model = SSDetectModule.load_from_checkpoint(f'/kaggle/input/rsna-spine-final-models/ss_detect_{i}.ckpt')
    model.eval()
    model.zero_grad()
    model.to(device)

    pred_temp = {}
    for k in coor_predict.keys(): 
        pred_temp[k] = []
    study_id_list = []
    with torch.no_grad():
        for data in tqdm(data_loader_test, total=len(data_loader_test)):
            images, study_id = data
            images = images.to(device)
            preds = model.forward(images)
            #print(preds)
            for k, v in preds.items(): 
                pred_temp[k].append(v.to('cpu').detach().numpy())
            study_id_list.append(study_id.to('cpu').reshape(-1).detach().numpy())
    for k, v in pred_temp.items(): 
        coor_predict[k].append(np.concatenate(v))
    del pred_temp
    gc.collect()
    study_id = np.concatenate(study_id_list)
for k, v in coor_predict.items(): 
    coor_predict[k] = np.mean(np.array(coor_predict[k]), axis=0)
coor_predict['study_id'] = study_id
del study_id
gc.collect()


study_id = coor_predict['study_id']
coor_dict = {'study_id': [], 'x': [], 'y': [], 'condition': [], 'level': []}
for k, v in coor_predict.items(): 
    if k == 'study_id': 
        continue
    _lr, location = k.split('_')
    if _lr == 'left': 
        lr = 'Left Subarticular Stenosis'
    else: 
        lr = 'Right Subarticular Stenosis'
    coor_dict['study_id'].extend(list(coor_predict['study_id']))
    coor_dict['x'].extend(list(v[:, 0]))
    coor_dict['y'].extend(list(v[:, 1]))
    coor_dict['condition'].extend([lr]*len(v))
    coor_dict['level'].extend([location]*len(v))
ax_coor_pred = pd.DataFrame(coor_dict)
ax_coor_pred = pd.merge(closest_ax, ax_coor_pred, on=['study_id', 'level'], how='left')
print(ax_coor_pred.shape)
ax_coor_pred.head()


ax_coor_pred = ax_coor_pred.merge(meta_df[['study_id', 'series_id', 'instance_number', 'height', 'width']], on=['study_id', 'series_id', 'instance_number'], how='left')
ax_coor_pred['x'] = ax_coor_pred['x']*ax_coor_pred['width']
ax_coor_pred['y'] = ax_coor_pred['y']*ax_coor_pred['height']
ax_coor_pred['x'] = ax_coor_pred['x'].apply(lambda x: round(x))
ax_coor_pred['y'] = ax_coor_pred['y'].apply(lambda x: round(x))
ax_coor_pred = ax_coor_pred.drop(['height', 'width'], axis=1)


stage3 = pd.read_csv('stage3_coor.csv')
stage3.tail()


pred_coor_stage3 = pd.concat([pred_coor_stage2, ax_coor_pred])
display(pred_coor_stage3)
pred_coor_stage3.to_csv('stage3_coor.csv', index=False)


del pred_coor_stage2, ax_coor_pred, pred_coor, closest_ax
gc.collect()


os.listdir('/kaggle/working/')


class ClassDataset(Dataset):
    def __init__(self, coor, meta, condition, channel, usage='sub'):
        self.coor = coor
        self.meta = meta
        self.condition = condition
        self.usage = usage
        self.sag_window = channel
        self.ax_window = channel
        self.wide_resize = v2.Resize((128, 224))
        #self.wide_resize = v2.Resize((224, 224))
        self.rec_resize = v2.Resize((256, 256))
        self.resize = v2.Resize((128, 128))
        self.resize_3d = v2.Resize((256, 256))
        self.pre_resize = v2.Resize((512, 512))
        self.id = list(meta.study_id.unique())
        if 3637444890 in self.id: 
            self.id.remove(3637444890)
    def __getitem__(self, index):
        study_id = self.id[index]
        #print(study_id)
        res = {}
        #try:
        if self.condition == 'scs':
            sagt2_img, ax_img = self.for_scs(study_id)
            res['sagt2'] = sagt2_img.to(torch.float32)
            res['ax'] = ax_img.to(torch.float32)
            #res['sagt1'] = sagt1_img.to(torch.float32)
        elif self.condition == 'nfn':
            ax_img, sagt1_img = self.for_nfn(study_id)
            res['ax'] = ax_img.to(torch.float32)
            res['sagt1'] = sagt1_img.to(torch.float32)
        if self.condition == 'ss':
            ax_img = self.for_ss(study_id)
            #ax_img, sagt1_img, sagt2_img = self.for_ss(study_id)
            res['ax'] = ax_img.to(torch.float32)
        return res, torch.tensor(study_id)

    def crop(self, image, x, y, z, x_left, x_right, y_bottom, y_top, wide):
        size = [image[i].shape for i in z]
        #print([self.pre_resize(torch.tensor(image[i])[None, ...]).squeeze() for i, shape in zip(z, size)][0].shape)
        data = torch.stack([torch.tensor(self.pre_resize(torch.tensor(image[i])[None, ...]).squeeze()[max(int((y/shape[0])*512-y_top), 0):int((y/shape[0])*512+y_bottom), max(int((x/shape[1])*512-x_left), 0): int((x/shape[1])*512+x_right)]) for i, shape in zip(z, size)])

        if wide:
            data = self.wide_resize(data)
        else:
            data = self.rec_resize(data)

        return data
    def for_scs(self, study_id):
        sagt2_meta = self.meta.loc[(self.meta.study_id==study_id) & (self.meta.series_description=='Sagittal T2/STIR')]
        #display(sagt2_meta)
        ax_meta = self.meta.loc[(self.meta.study_id==study_id) & (self.meta.series_description=='Axial T2')]
        sagt1_meta = self.meta.loc[(self.meta.study_id==study_id) & (self.meta.series_description=='Sagittal T1')]
        #display(ax_meta)
        sagt2_meta = sagt2_meta.sort_values('ipp_x', ascending=True).reset_index(drop=True)
        ax_meta = ax_meta.sort_values('ipp_z', ascending=False).reset_index(drop=True)
        sagt1_meta = sagt1_meta.sort_values('ipp_x', ascending=True).reset_index(drop=True)
        sagt2_img = [self.normalize(self.load_dicom(IMAGE_PATH + f'{row.study_id}/{row.series_id}/{row.instance_number}.dcm')) for _, row in sagt2_meta.iterrows()]
        ax_img = [self.normalize(self.load_dicom(IMAGE_PATH + f'{row.study_id}/{row.series_id}/{row.instance_number}.dcm')) for _, row in ax_meta.iterrows()]
        sagt1_img = [self.normalize(self.load_dicom(IMAGE_PATH + f'{row.study_id}/{row.series_id}/{row.instance_number}.dcm')) for _, row in sagt1_meta.iterrows()]
        sagt1_img = [img if (img.shape[0]> 1 and img.shape[1] > 1) else np.zeros((512, 512)) for img in sagt1_img]
        sagt2_sub_coor = self.coor.loc[(self.coor.study_id==study_id) & (self.coor.condition=='Spinal Canal Stenosis')]
        ax_right_sub_coor = self.coor.loc[(self.coor.study_id==study_id) & (self.coor.condition=='Right Subarticular Stenosis')]
        ax_left_sub_coor = self.coor.loc[(self.coor.study_id==study_id) & (self.coor.condition=='Left Subarticular Stenosis')]
        sagt2_dict = {}
        for _, row in sagt2_sub_coor.iterrows():
            try: 
                y_shift = 0
                x_shift = 0
                z_shift = 0
                #display(sagt2_meta)
                #display(row)
                mid = sagt2_meta.loc[(sagt2_meta.series_id==row.series_id)&(sagt2_meta.instance_number==row.instance_number)].index[0]
                if row.level == 'L5/S1':
                    ushift = 20
                else:
                    ushift = 0
                z = [min(max(mid+w+z_shift, 0), len(sagt2_meta)-1) for w in range(-(self.sag_window-1)//2, ((self.sag_window-1)//2)+1)]
                sagt2_dict[row.level] = self.crop(sagt2_img, row.x+x_shift, row.y+y_shift, z, 96, 32, 40+ushift, 40-ushift, wide=True)
            except: 
                pass
                
        # AXIAL T2
        #in_list = ax_meta.instance_number.tolist()
        ax_dict = {}
        if np.random.choice([0, 1]) == 0:
            ax_sub_coor = ax_right_sub_coor
            lrshift = +20
        else:
            ax_sub_coor = ax_left_sub_coor
            lrshift = -20
        for _, row in ax_sub_coor.iterrows():
            try: 
                y_shift = 0
                x_shift = 0
                z_shift = 0
                ax_meta_sub = ax_meta.loc[(ax_meta.series_id==row.series_id)]
                ax_meta_sub_original_idx = ax_meta_sub.index.tolist()
                ax_meta_sub = ax_meta_sub.reset_index(drop=True)
                mid = ax_meta_sub.loc[(ax_meta_sub.instance_number==row.instance_number)].index[0]
                z = [min(max(mid+w+z_shift, 0), len(ax_meta_sub)-1) for w in range(-(self.ax_window-1)//2, ((self.ax_window-1)//2)+1)]
                ax_dict[row.level] = self.crop([ax_img[i] for i in range(len(ax_img)) if i in ax_meta_sub_original_idx], row.x+x_shift+lrshift, row.y+y_shift, z, 96, 96, 96, 96, wide=False)
            except: 
                pass
        sagt2_img = [sagt2_dict.get(l, torch.zeros((self.sag_window, 128, 224))) for l in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']]
        ax_img = [ax_dict.get(l, torch.zeros((self.ax_window, 256, 256))) for l in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']]
        return torch.stack(sagt2_img).contiguous(), torch.stack(ax_img).contiguous()#, torch.stack(sagt1_img).contiguous()
    def for_ss(self, study_id):
        #display(sagt2_meta)
        ax_meta = self.meta.loc[(self.meta.study_id==study_id) & (self.meta.series_description=='Axial T2')]
        sagt1_meta = self.meta.loc[(self.meta.study_id==study_id) & (self.meta.series_description=='Sagittal T1')]
        sagt2_meta = self.meta.loc[(self.meta.study_id==study_id) & (self.meta.series_description=='Sagittal T2/STIR')]
        #display(ax_meta)
        ax_meta = ax_meta.sort_values('ipp_z', ascending=False).reset_index(drop=True)
        ax_img = [self.normalize(self.load_dicom(IMAGE_PATH + f'{row.study_id}/{row.series_id}/{row.instance_number}.dcm')) for _, row in ax_meta.iterrows()]
        ax_right_sub_coor = self.coor.loc[(self.coor.study_id==study_id) & (self.coor.condition=='Right Subarticular Stenosis')]
        ax_left_sub_coor = self.coor.loc[(self.coor.study_id==study_id) & (self.coor.condition=='Left Subarticular Stenosis')]

        # AXIAL T2
        #in_list = ax_meta.instance_number.tolist()
        ax_right_dict = {}
        for _, row in ax_right_sub_coor.iterrows():
            try: 
                y_shift = 0
                x_shift = 0
                z_shift = 0
                ax_meta_sub = ax_meta.loc[(ax_meta.series_id==row.series_id)]
                ax_meta_sub_original_idx = ax_meta_sub.index.tolist()
                ax_meta_sub = ax_meta_sub.reset_index(drop=True)
                mid = ax_meta_sub.loc[(ax_meta_sub.instance_number==row.instance_number)].index[0]
                z = [min(max(mid+w+z_shift, 0), len(ax_meta_sub)-1) for w in range(-(self.ax_window-1)//2, ((self.ax_window-1)//2)+1)]
                ax_right_dict[row.level] = self.crop([ax_img[i] for i in range(len(ax_img)) if i in ax_meta_sub_original_idx], row.x+x_shift, row.y+y_shift, z, 160-16, 32+16, 64+32, 64+32, wide=False)
            except: 
                pass
        ax_left_dict = {}
        for _, row in ax_left_sub_coor.iterrows():
            try: 
                y_shift = 0
                x_shift = 0
                z_shift = 0
                ax_meta_sub = ax_meta.loc[(ax_meta.series_id==row.series_id)]
                ax_meta_sub_original_idx = ax_meta_sub.index.tolist()
                ax_meta_sub = ax_meta_sub.reset_index(drop=True)
                mid = ax_meta_sub.loc[(ax_meta_sub.instance_number==row.instance_number)].index[0]
                z = [min(max(mid+w+z_shift, 0), len(ax_meta_sub)-1) for w in range(-(self.ax_window-1)//2, ((self.ax_window-1)//2)+1)]
                ax_left_dict[row.level] = self.crop([ax_img[i] for i in range(len(ax_img)) if i in ax_meta_sub_original_idx], row.x+x_shift, row.y+y_shift, z, 32+16, 160-16, 64+32, 64+32, wide=False)
            except: 
                pass
        ax_right_img = [ax_right_dict.get(l, torch.zeros((self.ax_window, 256, 256))) for l in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']]
        ax_left_img = [ax_left_dict.get(l, torch.zeros((self.ax_window, 256, 256))) for l in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']]
        ax_img = ax_left_img + ax_right_img
        return torch.stack(ax_img).contiguous()#, torch.stack(sagt1_img).contiguous(), torch.stack(sagt2_img).contiguous()

    def for_nfn(self, study_id):
        ax_meta = self.meta.loc[(self.meta.study_id==study_id) & (self.meta.series_description=='Axial T2')]
        sagt1_meta = self.meta.loc[(self.meta.study_id==study_id) & (self.meta.series_description=='Sagittal T1')]

        ax_meta = ax_meta.sort_values('ipp_z', ascending=False).reset_index(drop=True)
        sagt1_meta = sagt1_meta.sort_values('ipp_x', ascending=True).reset_index(drop=True)
        ax_img = [self.normalize(self.load_dicom(IMAGE_PATH + f'{row.study_id}/{row.series_id}/{row.instance_number}.dcm')) for _, row in ax_meta.iterrows()]
        sagt1_img = [self.normalize(self.load_dicom(IMAGE_PATH + f'{row.study_id}/{row.series_id}/{row.instance_number}.dcm')) for _, row in sagt1_meta.iterrows()]
        ax_right_sub_coor = self.coor.loc[(self.coor.study_id==study_id) & (self.coor.condition=='Right Subarticular Stenosis')]
        ax_left_sub_coor = self.coor.loc[(self.coor.study_id==study_id) & (self.coor.condition=='Left Subarticular Stenosis')]
        sagt1_right_sub_coor = self.coor.loc[(self.coor.study_id==study_id) & (self.coor.condition=='Right Neural Foraminal Narrowing')]
        sagt1_left_sub_coor = self.coor.loc[(self.coor.study_id==study_id) & (self.coor.condition=='Left Neural Foraminal Narrowing')]

        # SAGITTAL T2
        # not implemented

        # AXIAL T2
        #in_list = ax_meta.instance_number.tolist()
        ax_right_dict = {}
        for _, row in ax_right_sub_coor.iterrows():
            try: 
                y_shift = 0
                x_shift = 0
                z_shift = 0
                ax_meta_sub = ax_meta.loc[(ax_meta.series_id==row.series_id)]
                ax_meta_sub_original_idx = ax_meta_sub.index.tolist()
                ax_meta_sub = ax_meta_sub.reset_index(drop=True)
                mid = ax_meta_sub.loc[(ax_meta_sub.instance_number==row.instance_number)].index[0]
                z = [min(max(mid+w+z_shift, 0), len(ax_meta_sub)-1) for w in range(-(self.ax_window-1)//2, ((self.ax_window-1)//2)+1)]
                ax_right_dict[row.level] = self.crop([ax_img[i] for i in range(len(ax_img)) if i in ax_meta_sub_original_idx], row.x+x_shift, row.y+y_shift, z, 160-16, 32+16, 64+32, 64+32, wide=False)
            except: 
                pass
        ax_left_dict = {}
        for _, row in ax_left_sub_coor.iterrows():
            try: 
                y_shift = 0
                x_shift = 0
                z_shift = 0
                ax_meta_sub = ax_meta.loc[(ax_meta.series_id==row.series_id)]
                ax_meta_sub_original_idx = ax_meta_sub.index.tolist()
                ax_meta_sub = ax_meta_sub.reset_index(drop=True)
                mid = ax_meta_sub.loc[(ax_meta_sub.instance_number==row.instance_number)].index[0]
                z = [min(max(mid+w+z_shift, 0), len(ax_meta_sub)-1) for w in range(-(self.ax_window-1)//2, ((self.ax_window-1)//2)+1)]
                ax_left_dict[row.level] = self.crop([ax_img[i] for i in range(len(ax_img)) if i in ax_meta_sub_original_idx], row.x+x_shift, row.y+y_shift, z, 32+16, 160-16, 64+32, 64+32, wide=False)
            except: 
                pass
        # SAGITTAL T1
        sagt1_right_dict = {}
        for _, row in sagt1_right_sub_coor.iterrows():
            try: 
                y_shift = 0
                x_shift = 0
                z_shift = 0
                if row.level == 'L5/S1':
                    ushift = 10
                else:
                    ushift = 0
                sagt1_meta_sub = sagt1_meta.loc[(sagt1_meta.series_id==row.series_id)]
                sagt1_meta_sub_original_idx = sagt1_meta_sub.index.tolist()
                sagt1_meta_sub = sagt1_meta_sub.reset_index(drop=True)
                #display(sagt2_meta)
                #display(row)
                mid = sagt1_meta_sub.loc[(sagt1_meta_sub.instance_number==row.instance_number)].index[0]
                z = [min(max(mid+w+z_shift, 0), len(sagt1_meta_sub)-1) for w in range(-(self.sag_window-1)//2, ((self.sag_window-1)//2)+1)]
                sagt1_right_dict[row.level] = self.crop([sagt1_img[i] for i in range(len(sagt1_img)) if i in sagt1_meta_sub_original_idx], row.x+x_shift, row.y+y_shift, z, 96, 64, 32+ushift, 32-ushift, wide=True)
            except: 
                pass
        sagt1_left_dict = {}
        for _, row in sagt1_left_sub_coor.iterrows():
            try: 
                y_shift = 0
                x_shift = 0
                z_shift = 0
                if row.level == 'L5/S1':
                    ushift = 10
                else:
                    ushift = 0
                sagt1_meta_sub = sagt1_meta.loc[(sagt1_meta.series_id==row.series_id)]
                sagt1_meta_sub_original_idx = sagt1_meta_sub.index.tolist()
                sagt1_meta_sub = sagt1_meta_sub.reset_index(drop=True)
                mid = sagt1_meta_sub.loc[(sagt1_meta_sub.instance_number==row.instance_number)].index[0]
                z = [min(max(mid+w+z_shift, 0), len(sagt1_meta_sub)-1) for w in range(-(self.sag_window-1)//2, ((self.sag_window-1)//2)+1)]
                sagt1_left_dict[row.level] = self.crop([sagt1_img[i] for i in range(len(sagt1_img)) if i in sagt1_meta_sub_original_idx], row.x+x_shift, row.y+y_shift, z, 96, 64, 32+ushift, 32-ushift, wide=True)
            except: 
                pass
        ax_right_img = [ax_right_dict.get(l, torch.zeros((self.ax_window, 256, 256))) for l in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']]
        ax_left_img = [ax_left_dict.get(l, torch.zeros((self.ax_window, 256, 256))) for l in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']]
        sagt1_right_img = [sagt1_right_dict.get(l, torch.zeros((self.sag_window, 128, 224))) for l in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']]
        sagt1_left_img = [sagt1_left_dict.get(l, torch.zeros((self.sag_window, 128, 224))) for l in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']]
        ax_img = ax_left_img + ax_right_img
        sagt1_img = sagt1_left_img + sagt1_right_img
        return torch.stack(ax_img).contiguous(), torch.stack(sagt1_img).contiguous()


    def normalize(self, x):
        lower, upper = np.percentile(x, (1, 99))
        x = np.clip(x, lower, upper)
        x = x - np.min(x)
        x = x / np.max(x)
        return x

    def __len__(self):
        return len(self.id)

    def load_dicom(self, path):
        dicom = dcm.dcmread(path)
        data = dicom.pixel_array
        return data


class Flatten(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            #nn.LayerNorm(512)
        )
        
class ConConvnextSCS(nn.Module):
    def __init__(self, direction='sagt2'):
        super().__init__()
        self.direction = direction
        self.ax = timm.create_model('convnext_base.fb_in22k_ft_in1k_384', in_chans=3, pretrained=False, num_classes=0)
        self.sagt2 = timm.create_model('convnext_base.fb_in22k_ft_in1k_384', in_chans=3, pretrained=False, num_classes=0)
        self.num_features = self.ax.num_features
        self.flatten_ax = Flatten()
        self.flatten_sagt2 = Flatten()
        self.lin_ax = nn.Linear(self.num_features, 512)
        self.lin_sagt2 = nn.Linear(self.num_features, 512)
        self.aux_ax = nn.Linear(512, 3)
        self.aux_sagt2 = nn.Linear(512, 3)
        self.lin = nn.Linear(512*2, 512)
        self.out = nn.Linear(512, 3)
        self.dropout = nn.Dropout(0.0)
    def forward(self, ax, sagt2, sagt1=None, label=None):
        shape = ax.shape
        ax = ax.reshape(shape[0]*shape[1], 3, shape[-2], shape[-1])
        shape=sagt2.shape
        sagt2 = sagt2.reshape(shape[0]*shape[1], 3, shape[-2], shape[-1])
        ax = nn.functional.leaky_relu(self.lin_ax(self.flatten_ax(self.ax.forward_features(ax))))
        sagt2 = nn.functional.leaky_relu(self.lin_sagt2(self.flatten_sagt2(self.sagt2.forward_features(sagt2))))
        x = torch.cat([ax, sagt2], dim=1)
        x = self.lin(x)
        x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x

class ConConvnextNFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ax = timm.create_model('convnext_base.fb_in22k_ft_in1k_384', in_chans=3, pretrained=False, num_classes=0)
        self.sagt1 = timm.create_model('convnext_base.fb_in22k_ft_in1k_384', in_chans=3, pretrained=False, num_classes=0)
        self.num_features = self.ax.num_features
        self.flatten_ax = Flatten()
        self.flatten_sagt1 = Flatten()
        self.lin_ax = nn.Linear(self.num_features, 512)
        self.lin_sagt1 = nn.Linear(self.num_features, 512)
        self.aux_ax = nn.Linear(512, 3)
        self.aux_sagt1 = nn.Linear(512, 3)
        self.lin = nn.Linear(512*2, 512)
        self.out = nn.Linear(512, 3)
        self.dropout = nn.Dropout(0.0)
    def forward(self, ax, sagt1, sagt2=None, label=None):
        shape = ax.shape
        ax = ax.reshape(shape[0]*shape[1], 3, shape[-2], shape[-1])
        shape=sagt1.shape
        sagt1 = sagt1.reshape(shape[0]*shape[1], 3, shape[-2], shape[-1])
        ax = nn.functional.leaky_relu(self.lin_ax(self.flatten_ax(self.ax.forward_features(ax))))
        sagt1 = nn.functional.leaky_relu(self.lin_sagt1(self.flatten_sagt1(self.sagt1.forward_features(sagt1))))
        x = torch.cat([ax, sagt1], dim=1)
        x = self.lin(x)
        x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x
    
class ConvnextSS(nn.Module):
    def __init__(self, direction='ax'):
        super().__init__()
        self.direction = direction
        self.encoder = timm.create_model('convnext_large.fb_in22k_ft_in1k_384', in_chans=3, pretrained=False, num_classes=0)
        self.in_features = self.encoder.num_features
        self.flatten = Flatten()
        self.lin = nn.Linear(self.in_features, 512)
        self.out = nn.Linear(512, 3)
        self.dropout = nn.Dropout(0.0)
    def forward(self, ax, sagt1=None, sagt2=None, label=None):
        if self.direction == 'ax':
            shape = ax.shape
            x = ax.reshape(shape[0]*shape[1], 3, shape[-2], shape[-1])
        elif self.direction == 'sagt1':
            shape = sagt1.shape
            x = sagt1.reshape(shape[0]*shape[1], 3, shape[-2], shape[-1])
        elif self.direction == 'sagt2':
            shape = sagt2.shape
            x = sagt2.reshape(shape[0]*shape[1], 3, shape[-2], shape[-1])
        x = self.flatten(self.encoder.forward_features(x))
        x = self.lin(x)
        x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x#, ax, sagt1


class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, condition):
        super(AttentionMIL, self).__init__()
        self.condition = condition
        if condition == 'nfn': 
            self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
            )
        else: 
            self.lin = nn.Linear(input_dim, hidden_dim)
            self.attn_score = nn.Linear(hidden_dim, 1)
            self.act = nn.Tanh()
    def forward(self, bags):
        """
        Args:
            bags: (batch_size, num_instances, input_dim)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, num_instances, input_dim = bags.size()

        # Attention mechanism
        if self.condition=='nfn': 
            attn_scores = self.attention(bags).squeeze(-1)  # (batch_size, num_instances)
        else: 
            x = self.lin(bags)
            attn_scores = self.attn_score(self.act(x)).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch_size, num_instances)
        # Weighted sum of instances
        weighted_instances = torch.bmm(attn_weights.unsqueeze(1), bags).squeeze(1)  # (batch_size, input_dim)

        # Classification
        #logits = self.classifier(weighted_instances)
        return weighted_instances, attn_scores
class SelfAttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, is_layer_norm=False):
        super(SelfAttentionMIL, self).__init__()
        self.is_layer_norm = is_layer_norm

        # Self-Attention層
        self.self_attn = nn.MultiheadAttention(input_dim, num_heads=8, batch_first=True)


        # バッグレベルの分類器
        #self.bag_classifier = nn.Sequential(
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(p=0.0)
        self.lin = nn.Linear(input_dim, hidden_dim)
        self.act = nn.Tanh()
        self.calc_attn_score = nn.Linear(hidden_dim, 1)  # バッグレベルのスコア
        #)

    def forward(self, bags):
        # Self-Attention
        attn_output, _ = self.self_attn(bags, bags, bags)
        x = attn_output + bags
        if self.is_layer_norm: 
            x = self.layer_norm(x)
        # バッグレベルのAttentionスコアを計算
        #bag_attn_scores = self.bag_classifier(attn_output).squeeze(-1)
        x = self.lin(x)
        bag_attn_scores = self.calc_attn_score(self.act(x)).squeeze(-1)
        bag_attn_weights = torch.softmax(bag_attn_scores, dim=-1)

        # Attention重み付き平均でバッグレベルの特徴量を計算
        bag_features = torch.bmm(bag_attn_weights.unsqueeze(1), attn_output).squeeze(1)

        return bag_features, bag_attn_scores
    
class LSTMMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(LSTMMIL, self).__init__()
        #self.attention = nn.Sequential(
        #    nn.Linear(input_dim, hidden_dim),
        #    nn.Tanh(),
        #    nn.Linear(hidden_dim, 1)
        #)
        #self.classifier = nn.Linear(input_dim, num_classes)
        self.lstm = nn.LSTM(input_dim, input_dim//2, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
        #self.lin = nn.Linear(input_dim, hidden_dim)
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

        # Attention mechanism
        #attn_scores = self.attention(bags).squeeze(-1)  # (batch_size, num_instances)
        bags_lstm, _ = self.lstm(bags)
        attn_scores = self.attention(bags_lstm).squeeze(-1)
        aux_attn_scores = self.aux_attention(bags_lstm).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch_size, num_instances)
        #aux_attn_weights = torch.softmax(aux_attn_scores, dim=-1)
        # Weighted sum of instances
        weighted_instances = torch.bmm(attn_weights.unsqueeze(1), bags_lstm).squeeze(1)  # (batch_size, input_dim)
        # Classification
        #logits = self.classifier(weighted_instances)
        return weighted_instances, aux_attn_scores
    
class SCSMIL(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        if 'convnext' in model_name: 
            self.sagt2_encoder = timm.create_model('convnext_small.fb_in22k_ft_in1k_384', in_chans=1, pretrained=False, num_classes=0)
            self.ax_encoder = timm.create_model('convnext_small.fb_in22k_ft_in1k_384', in_chans=1, pretrained=False, num_classes=0)
        elif 'effv2s' in model_name: 
            self.sagt2_encoder = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', in_chans=1, pretrained=False, num_classes=0)
            self.ax_encoder = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', in_chans=1, pretrained=False, num_classes=0)
        self.sagt2_flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(1))
        #self.sagt1_flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
        #                            nn.Flatten(1))
        self.ax_flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(1))
        self.sagt2_num_features = self.sagt2_encoder.num_features
        #self.sagt1_num_features = self.sagt1_encoder.num_features
        self.ax_num_features = self.ax_encoder.num_features
        self.sagt2_head = LSTMMIL(self.sagt2_num_features, 512, 3)
        #self.sagt1_head = AttentionMIL(self.sagt1_num_features, 512, 3)
        self.ax_head = LSTMMIL(self.ax_num_features, 512, 3)
        
        self.out = nn.Linear(self.sagt2_num_features + self.ax_num_features, 3)
        self.aux_out = nn.Linear(self.sagt2_num_features, 3)
        self.dropout = nn.Dropout(0.0)
    def forward(self, ax, sagt2, sagt1=None):
        ax_shape = ax.shape
        ax = ax.reshape(ax_shape[0]*ax_shape[1]*ax_shape[2], 1, ax_shape[-2], ax_shape[-1])
        ax = self.ax_encoder.forward_features(ax)
        ax = self.ax_flatten(ax)
        ax = ax.reshape(ax_shape[0]*ax_shape[1], ax_shape[2], -1)
        ax_weighted_sum, ax_attn = self.ax_head(ax)
        ax_attn = ax_attn.reshape(ax_shape[0], ax_shape[1], -1)

        sagt2_shape = sagt2.shape
        sagt2 = sagt2.reshape(sagt2_shape[0]*sagt2_shape[1]*sagt2_shape[2], 1, sagt2_shape[-2], sagt2_shape[-1])
        sagt2 = self.sagt2_encoder.forward_features(sagt2)
        sagt2 = self.sagt2_flatten(sagt2)
        sagt2 = sagt2.reshape(sagt2_shape[0]*sagt2_shape[1], sagt2_shape[2], -1)
        sagt2_weighted_sum, sagt2_attn = self.sagt2_head(sagt2)
        sagt2_attn = sagt2_attn.reshape(sagt2_shape[0], sagt2_shape[1], -1)

        out = torch.cat([ax_weighted_sum, sagt2_weighted_sum], dim=1)
        out = self.out(out)
        sagt2_out = self.aux_out(sagt2_weighted_sum)
        ax_out = self.aux_out(ax_weighted_sum)
        #print(sagt2_attn.shape, ax_attn.shape)
        ax_attn = {'L1/L2': ax_attn[:, 0, :], 'L2/L3': ax_attn[:, 1, :], 'L3/L4': ax_attn[:, 2, :], 'L4/L5': ax_attn[:, 3, :], 'L5/S1': ax_attn[:, 4, :]}
        sagt2_attn = {'L1/L2': sagt2_attn[:, 0, :], 'L2/L3': sagt2_attn[:, 1, :], 'L3/L4': sagt2_attn[:, 2, :], 'L4/L5': sagt2_attn[:, 3, :], 'L5/S1': sagt2_attn[:, 4, :]}
        #sagt1_attn = {'L1/L2': sagt1_attn[:, 0, :], 'L2/L3': sagt1_attn[:, 1, :], 'L3/L4': sagt1_attn[:, 2, :], 'L4/L5': sagt1_attn[:, 3, :], 'L5/S1': sagt1_attn[:, 4, :]}
        return out


class NFNMIL(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        if 'convnext' in model_name: 
            self.sagt1_encoder = timm.create_model('convnext_small.fb_in22k_ft_in1k_384', in_chans=1, pretrained=False, num_classes=0)
            self.ax_encoder = timm.create_model('convnext_small.fb_in22k_ft_in1k_384', in_chans=1, pretrained=False, num_classes=0)
        elif 'effv2s' in model_name: 
            self.sagt1_encoder = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', in_chans=1, pretrained=False, num_classes=0)
            self.ax_encoder = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', in_chans=1, pretrained=False, num_classes=0)
        self.sagt1_flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(1))
        self.ax_flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(1))
        self.sagt1_num_features = self.sagt1_encoder.num_features
        self.ax_num_features = self.ax_encoder.num_features
        self.sagt1_head = LSTMMIL(self.sagt1_num_features, 512, 3)
        self.ax_head = LSTMMIL(self.ax_num_features, 512, 3)
        self.out = nn.Linear(self.sagt1_num_features+self.ax_num_features, 3)
        self.aux_out = nn.Linear(self.sagt1_num_features, 3)
        self.dropout = nn.Dropout(0.0)
    def forward(self, ax, sagt1):
        ax_shape = ax.shape
        #print(ax.shape, sagt2.shape)
        ax = ax.reshape(ax_shape[0]*ax_shape[1]*ax_shape[2], 1, ax_shape[-2], ax_shape[-1])
        ax = self.ax_encoder.forward_features(ax)
        ax = self.ax_flatten(ax)
        ax = ax.reshape(ax_shape[0]*ax_shape[1], ax_shape[2], -1)
        ax_weighted_sum, ax_attn = self.ax_head(ax)
        ax_attn = ax_attn.reshape(ax_shape[0], ax_shape[1], -1)
        #ax_attn = ax_attn.transpose(1, 2)
        sagt1_shape = sagt1.shape
        sagt1 = sagt1.reshape(sagt1_shape[0]*sagt1_shape[1]*sagt1_shape[2], 1, sagt1_shape[-2], sagt1_shape[-1])
        sagt1 = self.sagt1_encoder.forward_features(sagt1)
        sagt1 = self.sagt1_flatten(sagt1)
        sagt1 = sagt1.reshape(sagt1_shape[0]*sagt1_shape[1], sagt1_shape[2], -1)
        sagt1_weighted_sum, sagt1_attn = self.sagt1_head(sagt1)
        sagt1_attn = sagt1_attn.reshape(sagt1_shape[0], sagt1_shape[1], -1)
        x = torch.cat([ax_weighted_sum, sagt1_weighted_sum], dim=1)
        out = self.out(x)
        return out

class SSMIL(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        if 'convnext' in model_name: 
            self.ax_encoder = timm.create_model('convnext_small.fb_in22k_ft_in1k_384', in_chans=1, pretrained=False, num_classes=0)
        elif 'effv2s' in model_name: 
            self.ax_encoder = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', in_chans=1, pretrained=False, num_classes=0)
        
        self.ax_flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(1))
        self.ax_num_features = self.ax_encoder.num_features
        self.ax_head = LSTMMIL(self.ax_num_features, 512, 3)
        self.out = nn.Linear(self.ax_num_features, 3)
        self.dropout = nn.Dropout(0.0)
    def forward(self, ax):
        ax_shape = ax.shape
        #print(ax.shape, sagt2.shape)
        ax = ax.reshape(ax_shape[0]*ax_shape[1]*ax_shape[2], 1, ax_shape[-2], ax_shape[-1])
        ax = self.ax_encoder.forward_features(ax)
        ax = self.ax_flatten(ax)
        ax = ax.reshape(ax_shape[0]*ax_shape[1], ax_shape[2], -1)
        ax_weighted_sum, ax_attn = self.ax_head(ax)
        ax_attn = ax_attn.reshape(ax_shape[0], ax_shape[1], -1)

        #out = torch.cat([ax_weighted_sum, sagt2_weighted_sum], dim=1)
        out = ax_weighted_sum
        out = self.out(out)
        #print(sagt2_attn.shape, ax_attn.shape)
        ax_attn = {'left_L1/L2': ax_attn[:, 0, :],'left_L2/L3': ax_attn[:, 1, :],'left_L3/L4': ax_attn[:, 2, :], 'left_L4/L5': ax_attn[:, 3, :], 'left_L5/S1': ax_attn[:, 4, :],
                'right_L1/L2': ax_attn[:, 5, :], 'right_L2/L3': ax_attn[:, 6, :], 'right_L3/L4': ax_attn[:, 7, :], 'right_L4/L5': ax_attn[:, 8, :], 'right_L5/S1': ax_attn[:, 9, :]}
        return out


class ClassModule(pl.LightningModule):
    def __init__(self, condition, model_name):
        super().__init__()
        self.condition = condition
        if condition == 'scs':
            if 'mil' in model_name: 
                self.model = SCSMIL(model_name)
            else: 
                self.model = ConConvnextSCS('sagt2')
        elif condition == 'nfn':
            if 'mil' in model_name: 
                self.model = NFNMIL(model_name)
            else: 
                self.model = ConConvnextNFN()
        elif condition == 'ss':
            if 'mil' in model_name: 
                self.model = SSMIL(model_name)
            else: 
                self.model = ConvnextSS('ax')

    def forward(self, batch):
        preds = self.model(**batch)
        return preds


%%time
prefix = ''
import warnings
warnings.filterwarnings("ignore")
severity_predict = {'scs': {
                     'L1/L2':[], 
                     'L2/L3': [], 
                     'L3/L4': [], 
                     'L4/L5': [], 
                     'L5/S1': []
                     }, 
                 'nfn': {
                     'left_L1/L2': [], 
                     'left_L2/L3': [], 
                     'left_L3/L4': [], 
                     'left_L4/L5': [], 
                     'left_L5/S1': [], 
                     'right_L1/L2': [], 
                     'right_L2/L3': [], 
                     'right_L3/L4': [], 
                     'right_L4/L5': [], 
                     'right_L5/S1': [], 
                     }, 
                'ss': {
                     'left_L1/L2': [], 
                     'left_L2/L3': [], 
                     'left_L3/L4': [], 
                     'left_L4/L5': [], 
                     'left_L5/S1': [], 
                     'right_L1/L2': [], 
                     'right_L2/L3': [], 
                     'right_L3/L4': [], 
                     'right_L4/L5': [], 
                     'right_L5/S1': [], 
                     }
                    }
model_path_dict = {
    'scs': [
           '/kaggle/input/rsna-spine-final-models/_scs_classify_5ch_axsagt2-lstm-mil_auxloss_auxdepth_convnext-s_for_exp0.ckpt', 
           '/kaggle/input/rsna-spine-final-models/_scs_classify_5ch_axsagt2-lstm-mil_auxloss_auxdepth_convnext-s_for_exp1.ckpt', 
           '/kaggle/input/rsna-spine-final-models/_scs_classify_5ch_axsagt2-lstm-mil_auxloss_auxdepth_convnext-s_for_exp2.ckpt', 
           '/kaggle/input/rsna-spine-final-models/_scs_classify_5ch_axsagt2-lstm-mil_auxloss_auxdepth_convnext-s_for_exp3.ckpt', 
           '/kaggle/input/rsna-spine-final-models/_scs_classify_5ch_axsagt2-lstm-mil_auxloss_auxdepth_convnext-s_for_exp4.ckpt', 
           '/kaggle/input/rsna-spine-final-models/_scs_classify_5ch_axsagt2-lstm-mil_auxloss_auxdepth_effv2s_for_exp0.ckpt', 
           '/kaggle/input/rsna-spine-final-models/_scs_classify_5ch_axsagt2-lstm-mil_auxloss_auxdepth_effv2s_for_exp1.ckpt', 
           '/kaggle/input/rsna-spine-final-models/_scs_classify_5ch_axsagt2-lstm-mil_auxloss_auxdepth_effv2s_for_exp2.ckpt', 
           '/kaggle/input/rsna-spine-final-models/_scs_classify_5ch_axsagt2-lstm-mil_auxloss_auxdepth_effv2s_for_exp3.ckpt', 
           '/kaggle/input/rsna-spine-final-models/_scs_classify_5ch_axsagt2-lstm-mil_auxloss_auxdepth_effv2s_for_exp4.ckpt', 
           ], 
    'nfn': [
          '/kaggle/input/rsna-spine-final-models/nfn_classify_5ch_axsagt1-lstm-mil_auxloss_auxdepth_2shift_convnext-s_0.ckpt', 
          '/kaggle/input/rsna-spine-final-models/nfn_classify_5ch_axsagt1-lstm-mil_auxloss_auxdepth_2shift_convnext-s_1.ckpt', 
          '/kaggle/input/rsna-spine-final-models/nfn_classify_5ch_axsagt1-lstm-mil_auxloss_auxdepth_2shift_convnext-s_2.ckpt', 
          '/kaggle/input/rsna-spine-final-models/nfn_classify_5ch_axsagt1-lstm-mil_auxloss_auxdepth_2shift_convnext-s_3.ckpt',
          '/kaggle/input/rsna-spine-final-models/nfn_classify_5ch_axsagt1-lstm-mil_auxloss_auxdepth_2shift_convnext-s_4.ckpt',
          '/kaggle/input/rsna-spine-final-models/nfn_classify_5ch_axsagt1-lstm-mil_auxloss_auxdepth_2shift_effv2s_0.ckpt', 
          '/kaggle/input/rsna-spine-final-models/nfn_classify_5ch_axsagt1-lstm-mil_auxloss_auxdepth_2shift_effv2s_1.ckpt', 
          '/kaggle/input/rsna-spine-final-models/nfn_classify_5ch_axsagt1-lstm-mil_auxloss_auxdepth_2shift_effv2s_2.ckpt', 
          '/kaggle/input/rsna-spine-final-models/nfn_classify_5ch_axsagt1-lstm-mil_auxloss_auxdepth_2shift_effv2s_3.ckpt',
          '/kaggle/input/rsna-spine-final-models/nfn_classify_5ch_axsagt1-lstm-mil_auxloss_auxdepth_2shift_effv2s_4.ckpt',
           ], 
    'ss': [
          '/kaggle/input/rsna-spine-final-models/ss_classify_5ch_ax-lstm-mil_auxloss_auxdepth_effv2s_0.ckpt', 
          '/kaggle/input/rsna-spine-final-models/ss_classify_5ch_ax-lstm-mil_auxloss_auxdepth_effv2s_1.ckpt', 
          '/kaggle/input/rsna-spine-final-models/ss_classify_5ch_ax-lstm-mil_auxloss_auxdepth_effv2s_2.ckpt', 
          '/kaggle/input/rsna-spine-final-models/ss_classify_5ch_ax-lstm-mil_auxloss_auxdepth_effv2s_3.ckpt',
          '/kaggle/input/rsna-spine-final-models/ss_classify_5ch_ax-lstm-mil_auxloss_auxdepth_effv2s_4.ckpt',
          '/kaggle/input/rsna-spine-final-models/ss_classify_5ch_ax-lstm-mil_auxloss_auxdepth_convnext-s_0.ckpt', 
          '/kaggle/input/rsna-spine-final-models/ss_classify_5ch_ax-lstm-mil_auxloss_auxdepth_convnext-s_1.ckpt', 
          '/kaggle/input/rsna-spine-final-models/ss_classify_5ch_ax-lstm-mil_auxloss_auxdepth_convnext-s_2.ckpt', 
          '/kaggle/input/rsna-spine-final-models/ss_classify_5ch_ax-lstm-mil_auxloss_auxdepth_convnext-s_3.ckpt',
          '/kaggle/input/rsna-spine-final-models/ss_classify_5ch_ax-lstm-mil_auxloss_auxdepth_convnext-s_4.ckpt',
          ]
}
##############SEVERITY PREDICT#########################
for condition in ['nfn', 'scs', 'ss']:
    print(condition)
    for path in model_path_dict[condition]:
        _meta_df = meta_df.copy()
        _coor_df = pred_coor_stage3.copy()
        model_name = path.split('/')[-1]
        if '5ch' in model_name: 
            dataset_channel = 5
        else: 
            dataset_channel = 3
        dataset_test = ClassDataset(_coor_df, _meta_df,  condition, dataset_channel, 'sub')
        data_loader_test = DataLoader(
            dataset_test,
            batch_size=4,
            shuffle=False,
            num_workers=4,
            pin_memory=False
        )

        model = ClassModule.load_from_checkpoint(path, condition=condition, model_name=model_name, strict=False)
        model.eval()
        model.zero_grad()
        model.to(device)
        pred_temp = {}
        for k in severity_predict[condition].keys(): 
            pred_temp[k] = []
        study_id_list = []
        with torch.no_grad():
            for data in tqdm(data_loader_test, total=len(data_loader_test)):
                images, study_id = data
                for k, v in images.items(): 
                    images[k] = v.to(device)
                    bs = v.shape[0]
                preds = model.forward(images)
                preds = nn.functional.softmax(preds, dim=1)
                preds = preds.reshape((bs, -1, 3))
                preds = preds.to('cpu').detach().numpy()
                #print(preds)
                if condition == 'scs':  
                    pred_temp['L1/L2'].append(preds[:, 0, :])
                    pred_temp['L2/L3'].append(preds[:, 1, :])
                    pred_temp['L3/L4'].append(preds[:, 2, :])
                    pred_temp['L4/L5'].append(preds[:, 3, :])
                    pred_temp['L5/S1'].append(preds[:, 4, :])
                else: 
                    pred_temp['left_L1/L2'].append(preds[:, 0, :])
                    pred_temp['left_L2/L3'].append(preds[:, 1, :])
                    pred_temp['left_L3/L4'].append(preds[:, 2, :])
                    pred_temp['left_L4/L5'].append(preds[:, 3, :])
                    pred_temp['left_L5/S1'].append(preds[:, 4, :])
                    pred_temp['right_L1/L2'].append(preds[:, 5, :])
                    pred_temp['right_L2/L3'].append(preds[:, 6, :])
                    pred_temp['right_L3/L4'].append(preds[:, 7, :])
                    pred_temp['right_L4/L5'].append(preds[:, 8, :])
                    pred_temp['right_L5/S1'].append(preds[:, 9, :])
                study_id_list.append(study_id.to('cpu').reshape(-1).detach().numpy())
                del images, preds
                gc.collect()
        for k, v in pred_temp.items(): 
            severity_predict[condition][k].append(np.concatenate(v))
        all_study_id = np.concatenate(study_id_list)
        del pred_temp, study_id_list
        gc.collect()
        
    for k, v in severity_predict[condition].items(): 
        severity_predict[condition][k] = np.mean(np.array(severity_predict[condition][k]), axis=0)
    severity_predict[condition]['study_id'] = all_study_id
    gc.collect()


condition_mapper = {'scs': 'spinal_canal_stenosis', 'nfn': 'neural_foraminal_narrowing', 'ss': 'subarticular_stenosis'}
predict_list = []
for k, v in severity_predict.items():
    condition = condition_mapper[k]
    study_id = severity_predict[k]['study_id']
    for kk, vv in v.items(): 
        if kk == 'study_id': 
            continue
        level = kk.split('_')[-1].lower().replace('/', '_')
        loc = kk.split('_')[0]
        if loc == 'left':
            loc = 'left_'
        elif loc == 'right':
            loc = 'right_'
        else: 
            loc = ''
        row_id = [f'{str(si)}_' + loc + condition + '_' + level for si in study_id]
        df = pd.DataFrame({'row_id': row_id, 'normal_mild': vv[:, 0], 'moderate': vv[:, 1], 'severe': vv[:, 2]})
        predict_list.append(df)


predict_df = pd.concat(predict_list)


predict_df


sub = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/sample_submission.csv')
sub = sub.drop(['normal_mild', 'moderate', 'severe'], axis=1)
sub = sub.merge(predict_df, on='row_id', how='left')
sub = sub.fillna(1/3)
sub[['normal_mild', 'moderate', 'severe']] = sub[['normal_mild', 'moderate', 'severe']]


sub.to_csv('submission.csv', index=False)


os.listdir('/kaggle/working/')


data_submiss = pd.read_csv('submission.csv')
data_submiss

