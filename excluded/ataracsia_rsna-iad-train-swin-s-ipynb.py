# Data handling
import os
import random
import warnings
warnings.filterwarnings('ignore')
from typing import List, Dict, Optional, Tuple
from IPython.display import display
import datetime
import time
import numpy as np
import polars as pl
import pandas as pd
from sklearn.model_selection import  StratifiedKFold

# Medical imaging
import cv2

# Machine Lerning 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import timm

# Transformations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Visualization
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# Experiment Management
import wandb

# Competition API
# import kaggle_evaluation.rsna_inference_server


# datetime for unique checkpoint filenames
date_time = datetime.datetime.now()
date_time = date_time.strftime('%Y-%m-%d_%H-%M-%S')


# Run Configuration
RUN_NAME = "swin-s-original-agg-meta-5folds"
SAVE_DIR = "/kaggle/working/results"
TEST_RUN = True
SEED = 42
DEVICE = "cuda"

# Model Configuration
PRETRAINED = False

# Input Data Configuration
IMAGE_SIZE = 384
NUM_SLICES = 3
USE_AGGREGATED_SLICES = True
BATCH_SIZE = 5
NUM_FOLDS = 5
LABEL_NAMES = [
    # 13 classes
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    # 'Aneurysm Present',
]
NUM_LABELS = len(LABEL_NAMES)

# Training Configuration
NUM_EPOCHS = 20
PATIENCE = 5
POS_WEIGHT = torch.tensor([
    54.743589743589745,
    43.36734693877551,
    12.13595166163142,
    14.696750902527075,
    18.85388127853881,
    13.789115646258503,
    10.977961432506888,
    93.52173913043478,
    76.64285714285714,
    49.55813953488372,
    42.04950495049505,
    38.527272727272724,
    37.47787610619469,
    # 1.332618025751073
])



RUN_NAME = RUN_NAME + f'-{IMAGE_SIZE}-{NUM_SLICES}'
SAVE_DIR = SAVE_DIR + '/' + RUN_NAME + f'-{date_time}'

# Weights & Biases Configuration
if TEST_RUN:
    USE_WANDB = False
    WANDB_INIT = {}
    ARTIFACT = {}
else:
    USE_WANDB = True
    WANDB_INIT = {
        'project': 'RSNA-IAD',
        'group': 'Image Classification',
        'job_type': 'training model',
        'save_code': True,
    }
    ARTIFACT = {
        'name': RUN_NAME,
        'type': 'model, optimizer, scheduler',
    }


class Configuration:
    
    # Run
    run_name = RUN_NAME
    save_dir = SAVE_DIR
    test_run = TEST_RUN
    seed = SEED
    device = DEVICE
    
    # Model
    pretrained = PRETRAINED
    
    # Input Data
    image_size = IMAGE_SIZE
    num_slices = NUM_SLICES
    use_aggregated_slices = USE_AGGREGATED_SLICES
    batch_size = BATCH_SIZE
    num_folds = NUM_FOLDS
    label_names = LABEL_NAMES
    num_labels = NUM_LABELS
    
    # Training
    num_epochs = NUM_EPOCHS
    patience = PATIENCE
    pos_weight = POS_WEIGHT
    
    # Weights & Biases
    use_wandb = USE_WANDB
    wandb_init = WANDB_INIT
    artifact = ARTIFACT

CFG = Configuration



# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA version: {torch.version.cuda}")
    torch.cuda.empty_cache()
    CFG.device = 'cuda'
    CFG.pos_weight = CFG.pos_weight.to(CFG.device, dtype=torch.float32)
else:
    raise RuntimeError("CUDA is not available! This code requires GPU.")


def set_random_seed(seed=CFG.seed, deterministic=True):
    """
    Set random seed.
    
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    if deterministic:
        torch.backends.cudnn.benchmark = True


set_random_seed(seed=CFG.seed, deterministic=True)


if CFG.use_wandb:
    os.environ['WANDB_NOTEBOOK_NAME'] = CFG.run_name
    wandb.login()
    run = wandb.init(**CFG.wandb_init)
    artifact = wandb.Artifact(**CFG.artifact)
else:
    run = None
    artifact = None


class SwinWithMetaModel(nn.Module):
    
    def __init__(self, model_name='swin_s', pretrained=CFG.pretrained,
                 num_classes=CFG.num_labels):
        super().__init__()
        self.model_name = model_name
        
        if model_name == 'swin_s':
            self.backbone = timm.create_model(
                'swin_small_patch4_window7_224',
                pretrained=pretrained,
                img_size=CFG.image_size,
                num_classes=0,
                drop_rate=0.3,
                drop_path_rate=0.2,
                global_poopling='')
            
            # input layer modification: 3 channels -> CFG.num_slices channels
            self.backbone.patch_embed.proj = nn.Conv2d(
                in_channels=CFG.num_slices,
                out_channels=96,
                kernel_size=4,
                stride=4,
            )
        else:
            raise ValueError(f"Model {model_name} is not supported.")
        
        with torch.no_grad():
            dummy_input = torch.zeros(
                1,
                CFG.num_slices,
                CFG.image_size,
                CFG.image_size
                )
            features = self.backbone(dummy_input)
            
            if len(features.shape) == 4:
                # Conv features (batch, channels, height, width)
                num_features = features.shape[1]
                self.needs_pool = True
            elif len(features.shape) == 3:
                # Transformer features (batch, sequence, features)
                num_features = features.shape[-1]
                self.needs_pool = False
                self.needs_seq_pool = True
            else:
                # Already flat features (batch, features)
                num_features = features.shape[1]
                self.needs_pool = False
                self.needs_seq_pool = False
        # print(f"Model name: {model_name}")
        # print(f"Features: {num_features}")
        # print(f"Output Shape: {features.shape}")
        
        # Add global pooling for models that output spatial features
        if self.needs_pool:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.meta_features = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        # According to "LB #1"
        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, images, meta):
        # Extract image features
        image_features = self.backbone(images)
        
        # Apply appropriate pooling based on model type
        if hasattr(self, 'needs_pool') and self.needs_pool:
            # Conv features - apply global pooling
            image_features = self.global_pool(image_features)
            image_features = image_features.flatten(1)
        elif hasattr(self, 'needs_seq_pool') and self.needs_seq_pool:
            # Transformer features - average across sequence dimension
            image_features = image_features.mean(dim=1)
        elif len(image_features.shape) == 4:
            # Fallback for any 4D output
            image_features = F.adaptive_avg_pool2d(image_features, 1).flatten(1)
        elif len(image_features.shape) == 3:
            # Fallback for any 3D output
            image_features = image_features.mean(dim=1)
        
        # Process Meta Features
        meta_fieatures = self.meta_features(meta)
        
        # Combine Features
        x = torch.cat([image_features, meta_fieatures], dim=1)
        
        # Classicication
        x = self.classifier(x)
        x = torch.nn.Sigmoid()(x)
        return x

model = SwinWithMetaModel(model_name='swin_s', pretrained=CFG.pretrained)


model.to(CFG.device)

is_in_cuda_list = []

for name, parameter in model.named_parameters():
    # determination of cuda and its storage
    is_in_cuda_list.append(parameter.is_cuda)
    
if all(is_in_cuda_list):
    print('All parameters is in cuda')
        
else:
    print('One of the parameters is not in the cuda.')



class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class WeightedMultiLabelLoss(nn.Module):
    """Weighted multi-label loss"""
    def __init__(self, aneurysm_weight=3.0):
        super(WeightedMultiLabelLoss, self).__init__()
        self.weights = torch.ones(CFG.num_labels, device=device)
        # self.weights[-1] = aneurysm_weight
        
    def forward(self, outputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        weighted_loss = bce_loss * self.weights
        return weighted_loss.mean()


class ImprovedLoss(nn.Module):
    """Advanced combined loss function"""
    def __init__(self, aneurysm_weight=3.0, focal_weight=0.3):
        super(ImprovedLoss, self).__init__()
        self.aneurysm_weight = aneurysm_weight
        self.focal_weight = focal_weight
        
        self.weights = torch.ones(CFG.num_labels, device=device)
        # self.weights[-1] = aneurysm_weight
        
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        
    def forward(self, outputs, targets):
        # Weighted BCE
        bce_loss = F.binary_cross_entropy_with_logits(outputs,
                                                      targets,
                                                      reduction='none'
                                                     )
        weighted_bce = (bce_loss * self.weights).mean()
        
        # Focal Loss
        focal_loss = self.focal_loss(outputs, targets)
        
        # Combination
        loss = (1 - self.focal_weight) * weighted_bce \
            + self.focal_weight * focal_loss
            
        return loss
            


# # Optimizer
# optimizer = torch.optim.AdamW(model.parameters())

# # Loss Function
# criterion = nn.BCEWithLogitsLoss(pos_weight=CFG.pos_weight)

# # Schedulers
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max=CFG.num_epochs,
#     eta_min=1e-6
# )



def build_models():
    
    # Model
    model = SwinWithMetaModel(model_name='swin_s', pretrained=CFG.pretrained)
    model.to(CFG.device)
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters())
    # Loss Function
    # criterion = nn.BCEWithLogitsLoss(pos_weight=CFG.pos_weight)
    criterion = ImprovedLoss(aneurysm_weight=3.0, focal_weight=0.3)
    # Schedulers
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG.num_epochs,
        eta_min=1e-6
    )
    
    return model, optimizer, criterion, scheduler


# SeriesInstanceUID list
# series_list = os.listdir(f'../series_npy/{CFG.image_size}-aggregated')
series_list = [
    '1.2.826.0.1.3680043.8.498.10023411164590664678534044036963716636',
    '1.2.826.0.1.3680043.8.498.10022796280698534221758473208024838831',
    '1.2.826.0.1.3680043.8.498.10022688097731894079510930966432818105',
    '1.2.826.0.1.3680043.8.498.10021411248005513321236647460239137906',
    '1.2.826.0.1.3680043.8.498.10014757658335054766479957992112625961',
    '1.2.826.0.1.3680043.8.498.10012790035410518400400834395242853657',
    '1.2.826.0.1.3680043.8.498.10009383108068795488741533244914370182',
    '1.2.826.0.1.3680043.8.498.10005158603912009425635473100344077317',
    '1.2.826.0.1.3680043.8.498.10004684224894397679901841656954650085',
    '1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647',
    ]

# .npy path DataFrame
image_path_df = pd.read_csv(
    f'/kaggle/input/rsna-iad-csv/image_{CFG.image_size}-aggregated_path_df.csv'
)

# Meta DataFrame
meta_df = pd.read_csv('/kaggle/input/rsna-iad-csv/meta.csv')

# Label DataFrame
label_df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')
label_df = label_df[['SeriesInstanceUID'] + CFG.label_names]
label_df = label_df.loc[label_df['SeriesInstanceUID'].isin(series_list)]


series_list


label_df


# for training
train_transform = A.Compose(
    [   
        # Rotation
        A.Rotate(limit=(-3, 3), p=0.5, border_mode=cv2.BORDER_WRAP,  # cv2.BORDER_WRAP,
                 seed=CFG.seed
        ),
        
        # Normalization
        A.Normalize(normalization='min_max'),
        
        # ToTensor
        ToTensorV2(),
    ]
)

# for inference
inference_transform = A.Compose(
    [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
)
    
# for TTA
tta_transform = A.Compose(
    [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
        # Horizontal flip
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # Vertical flip
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # 90 degree rotation
        A.RandomRotate90(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # ↓ Original
        # Sharpen
        A.Sharpen(alpha=(0, 1.0), p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        ToTensorV2(),
    ]
)



class BaseDataset(torch.utils.data.Dataset):
    '''
    Datasetの__getitem__()は、num_slicesの枚数分だけ画像を出力する。
    
    Arguments:
    - series_list: 画像のSeriesInstanceUIDのリスト
    - image_path_df: 画像のパスを含むDataFrame
    - meta_df: 患者のメタデータが入ったDataFrame
    - label_df: ラベルが入ったDataFrame
    - num_slices: 1つのシリーズから抽出するスライス数
    - transforms: 画像変換のためのAlbumentationsのComposeオブジェクト
    '''
    def __init__(self,
                 series_list: list,
                 image_path_df=image_path_df,
                 meta_df=meta_df,
                 label_df=label_df,
                 transforms=None
                ):
        self.series_list = series_list
        self.image_path_df = image_path_df
        self.meta_df = meta_df
        self.label_df = label_df
        self.transforms = transforms
        self.num_slices = CFG.num_slices
        self.use_aggregated_slices = CFG.use_aggregated_slices

    def __len__(self):
        return len(self.series_list)

    def __getitem__(self, index):
        # Index to SeriesInstanceUID
        series_id = self.series_list[index]
        # Extract image paths from DataFrame
        image_path_df = self.image_path_df.loc[
            self.image_path_df['series_id'] == series_id
        ].reset_index(drop=True)
        
        # Get image paths
        mean_path = image_path_df.loc[0, 'mean_path']
        std_path = image_path_df.loc[0, 'std_path']
        kurt_path = image_path_df.loc[0, 'kurtosis_path']
        
        # Convert My Local Path to Kaggle Path
        mean_path = mean_path.replace('../series_npy/384-aggregated', '/kaggle/input/rsna-iad-aggregated-npy-dataset')
        std_path = std_path.replace('../series_npy/384-aggregated', '/kaggle/input/rsna-iad-aggregated-npy-dataset')
        kurt_path = kurt_path.replace('../series_npy/384-aggregated', '/kaggle/input/rsna-iad-aggregated-npy-dataset')

        # Stack Images
        images = []
        images.append(np.load(mean_path).astype(np.uint8))
        images.append(np.load(std_path).astype(np.uint8))
        images.append(np.load(kurt_path).astype(np.uint8))
        images = np.stack(images, axis=-1)
        
        # Transform
        if self.transforms:
            # ToTensorV2はnumpy.ndarrayをtorch.Tensorに変換する
            augmented = self.transforms(image=images)
            images = augmented['image']
        else:
            images = torch.tensor(images, dtype=torch.float32)
            images = torch.permute(images, (2, 0, 1))
            # Min-Max Normalization
            if torch.max(images) > 1.0:
                max_value = torch.max(images)
                min_value = torch.min(images)
                images = (images - min_value) / (max_value - min_value)
                
        # Meta data
        meta = self.meta_df.loc[
            self.meta_df['SeriesInstanceUID'] == series_id, ['age', 'sex']
        ]
        age = min(meta['age'].values[0], 100)
        age = age / 100
        sex = meta['sex'].values[0]
        meta = torch.tensor([age, sex], dtype=torch.float32)

        # Labels
        labels = self.label_df.loc[
            self.label_df['SeriesInstanceUID']==series_id, \
                CFG.label_names].values
        labels = torch.tensor(labels, dtype=torch.float32)
        labels = torch.squeeze(labels, dim=0)
        
        return images, meta, labels, series_id



# Represent multi-label with a single number
label_df['label_id'] = 0
exponentiations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
for i in range(len(CFG.label_names)):
    label_name = CFG.label_names[i]
    label_df['label_id'] += label_df[label_name] * exponentiations[i]
    
# Stratified K-Fold
skf = StratifiedKFold(
    n_splits=CFG.num_folds,
    shuffle=True,
    random_state=CFG.seed
)
label_df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(\
    skf.split(X=label_df, y=label_df['label_id'])):
    label_df.loc[val_idx, 'fold'] = fold


def build_dataloaders(fold: int):

    train_series = label_df.loc[\
        label_df['fold']!=fold, "SeriesInstanceUID"].values
    val_series = label_df.loc[\
        label_df['fold']==fold, "SeriesInstanceUID"].values
    train_labels = label_df.loc[label_df['fold']!=fold, CFG.label_names].values
    val_labels = label_df.loc[label_df['fold']==fold, CFG.label_names].values

    if CFG.test_run:
        
        train_series = train_series[:5]
        val_series = val_series[:5]
        train_labels = train_labels[:5]
        val_labels = val_labels[:5]

    # 2 dimensions -> 1 dimension
    train_series, val_series = train_series.flatten(), val_series.flatten()
    print(f"Train size: {len(train_series)}, Val size: {len(val_series)}")

    # Datasets
    train_dataset = BaseDataset(
        series_list=train_series,
        transforms=train_transform
    )
    val_dataset = BaseDataset(
        series_list=val_series,
        transforms=train_transform # or tta_transform
    )
    
    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_dataloader, val_dataloader


train_dataloaders = []
val_dataloaders = []

for fold in range(CFG.num_folds):
    print(f"Fold {fold}")
    train_dataloader, val_dataloader = build_dataloaders(fold)
    train_dataloaders.append(train_dataloader)
    val_dataloaders.append(val_dataloader)


# count execution time for one epoch
def count_time(start:float) -> float:
    
    elapsed_time = time.time() - start
    elapsed_time /= 60
    
    return elapsed_time



# to save model, optimizer, scheduler
def save_checkpoint(model, optimizer, scheduler,
                    fold=100, save_dir=CFG.save_dir):
    
    model.to('cpu')
    
    model_state_dict =  model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    scheduler_state_dict = scheduler.state_dict()
    
    model_path = save_dir + f'/model_fold{fold}.pth'
    optimizer_path = save_dir + f'/optimizer_fold{fold}.pth'
    scheduler_path = save_dir + f'/scheduler_fold{fold}.pth'
        
    torch.save(model_state_dict, model_path)
    torch.save(optimizer_state_dict, optimizer_path)
    torch.save(scheduler_state_dict, scheduler_path)
    
    model.to(device)
    
    print(f"Model saved.")

# # to load model, optimizer, scheduler
# def load_checkpoint(model, optimizer, scheduler, save_dir=''):
    
#     model.to('cpu')
    
#     model.load_state_dict(save_dir + '/model.pth')
#     optimizer.load_state_dict(save_dir + '/optimizer.pth')
#     scheduler.load_state_dict(save_dir + '/scheduler.pth')
    
#     model.to(device)
    
#     return model, optimizer, scheduler


def add_files_to_artifact(fold=100, save_dir=CFG.save_dir):
    
    artifact.add_file(save_dir + f'/model_fold{fold}.pth')
    artifact.add_file(save_dir + f'/optimizer_fold{fold}.pth')
    artifact.add_file(save_dir + f'/scheduler_fold{fold}.pth')
    
    print("Files added to the artifact.")


def train_one_epoch(model, optimizer, scheduler, criterion,
                    train_dataloader, val_dataloader,
                    epoch=100) -> Tuple[float, float, List, List]:
    
    print(f'---------- Epoch {epoch} ----------')
    
    # Training
    train_losses = []
    
    for images, meta, labels, series_ids in tqdm(train_dataloader):    
        images = images.to(CFG.device)
        meta = meta.to(CFG.device)
        labels = labels.to(CFG.device)
        optimizer.zero_grad()
        with autocast(device_type=CFG.device):
            outputs = model(images, meta)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
    
    mean_train_loss = np.mean(train_losses)
    print(f'Inner Mean Train Loss: {mean_train_loss:.4f}')
    
    # Validation
    val_losses = []
    inner_series_ids_list = []
    inner_predicted_list = []
    model.eval()
    
    with torch.no_grad():
        for images, meta, labels, series_ids in tqdm(val_dataloader):
            images = images.to(CFG.device)
            meta = meta.to(CFG.device)
            labels = labels.to(CFG.device)
            with autocast(device_type=CFG.device):
                outputs = model(images, meta)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                inner_series_ids_list.extend(series_ids)
                inner_predicted_list.extend(outputs.cpu().numpy().tolist())
        
    mean_val_loss = np.mean(val_losses)
    print(f'Inner Mean Validation Loss: {mean_val_loss:.8f}')
    
    scheduler.step()
        
    return mean_train_loss, mean_val_loss,\
        inner_series_ids_list, inner_predicted_list


def main():
    
    if not CFG.test_run:
        os.makedirs(CFG.save_dir, exist_ok=True)
    
    fold_val_losses = []
    outer_series_ids_list = []
    outer_predicted_list = []
    
    for fold in range(CFG.num_folds):
        print(f'======================== Fold {fold} ========================')
        
        # Model, Optimizer, Criterion, Scheduler
        model, optimizer, criterion, scheduler = build_models()
        
        # Dataloaders
        train_dataloader = train_dataloaders[fold]
        val_dataloader = val_dataloaders[fold]
        
        best_predicted_list = []
        
        best_val_loss = np.inf
    
        for epoch in range(CFG.num_epochs):
            
            # Train & Validation
            start_time = time.time()
            train_loss, val_loss, inner_series_ids, inner_predicted_list \
                = train_one_epoch(model, optimizer, scheduler, criterion,
                                  train_dataloader, val_dataloader,
                                  epoch=epoch)
            elapsed_time = count_time(start_time)
            print(f'Elapsed time: {elapsed_time}')
            
            # Log Losses to W&B
            if CFG.use_wandb:
                losses = {
                    f'train_loss_fold{fold}': train_loss,
                    f'val_loss_fold{fold}': val_loss
                }
                wandb.log(losses)
            
            if CFG.test_run:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_predicted_list = inner_predicted_list
                print('Test run: Skip saving checkpoint.')
            else:
                # Save best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, scheduler, fold=fold)
                    print(f'Best checkpoint saved at {CFG.save_dir}')
                    best_predicted_list = inner_predicted_list
        
        # Add model, optimizer, scheduler files to W&B
        if CFG.use_wandb:
            add_files_to_artifact(fold=fold)
        
        # Collect results for all folds
        fold_val_losses.append(best_val_loss)
        
        # Collect results for all folds
        outer_series_ids_list.extend(inner_series_ids)
        outer_predicted_list.extend(best_predicted_list)
        
    print(f'====================== All folds completed ======================')
    mean_fold_val_losses = np.mean(fold_val_losses)
    print(f'Each Fold Validation Losses: {fold_val_losses}')
    print(f'Mean Validation Losses: {mean_fold_val_losses:.8f}')
        
    if CFG.use_wandb:
        # Log Mean Validation Loss to W&B
        wandb.log({'mean_fold_val_losses': mean_fold_val_losses})
        
        # Log all files to W&B
        run.log_artifact(artifact)
        print('All artifacts were logged to W&B')
            
    return outer_series_ids_list, outer_predicted_list


outer_series_ids_list, outer_predicted_list = main()


result_df = pd.DataFrame()

result_df['SeriesInstanceUID'] = outer_series_ids_list
result_df[CFG.label_names] = outer_predicted_list

if not CFG.test_run:
    result_df.to_csv(f'{CFG.save_dir}/predicted_labels.csv', index=False)


result_df.describe()


if CFG.use_wandb:
    run.finish()


