!pip install iterative-stratification


"""
Use EVA-X series as your backbone. You could get 
EVA-X representations simply with timm. Try them 
with your own X-ray tasks. 
Enjoy!

Reference:
    https://github.com/baaivision/EVA
    https://github.com/huggingface/pytorch-image-models
Thanks for their work!
    
by Jingfeng Yao 
from HUST-VL
"""

import torch
import torch.nn as nn
from timm.layers import resample_abs_pos_embed, resample_patch_embed
from timm.models.eva import Eva


def checkpoint_filter_fn(
        state_dict,
        model,
        interpolation='bicubic',
        antialias=True,
):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    state_dict = state_dict.get('model_ema', state_dict)
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('module', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    # prefix for loading OpenCLIP compatible weights
    if 'visual.trunk.pos_embed' in state_dict:
        prefix = 'visual.trunk.'
    elif 'visual.pos_embed' in state_dict:
        prefix = 'visual.'
    else:
        prefix = ''
    mim_weights = prefix + 'mask_token' in state_dict
    no_qkv = prefix + 'blocks.0.attn.q_proj.weight' in state_dict

    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if k.startswith(prefix):
                k = k[len_prefix:]
            else:
                continue

        if 'rope' in k:
            # fixed embedding no need to load buffer from checkpoint
            continue

        if 'patch_embed.proj.weight' in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )

        k = k.replace('mlp.ffn_ln', 'mlp.norm')
        k = k.replace('attn.inner_attn_ln', 'attn.norm')
        k = k.replace('mlp.w12', 'mlp.fc1')
        k = k.replace('mlp.w1', 'mlp.fc1_g')
        k = k.replace('mlp.w2', 'mlp.fc1_x')
        k = k.replace('mlp.w3', 'mlp.fc2')
        if no_qkv:
            k = k.replace('q_bias', 'q_proj.bias')
            k = k.replace('v_bias', 'v_proj.bias')

        if mim_weights and k in ('mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight', 'norm.bias'):
            if k == 'norm.weight' or k == 'norm.bias':
                # try moving norm -> fc norm on fine-tune, probably a better starting point than new init
                k = k.replace('norm', 'fc_norm')
            else:
                # skip pretrain mask token & head weights
                continue

        out_dict[k] = v

    return out_dict

class EVA_X(Eva):
    def __init__(self, **kwargs):
        super(EVA_X, self).__init__(**kwargs)

        self.head4 = nn.Linear(self.head.in_features, 4)
    def forward_features(self, x):
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        for blk in self.blocks:
            x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x) # LayerNorm
        x = self.head_drop(x) # Dropout
        return x if pre_logits else self.head(x) # Linear

    def forward_head4(self, x):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x) # LayerNorm
        x = self.head_drop(x) # Dropout
        return self.head4(x) # Linear

    def contrastive_loss(feats, labels):
        return
    
    def forward(self, x, out4 = False):
        feats = self.forward_features(x)
        x14 = self.forward_head(feats)
        if out4:
            x4 = self.forward_head4(feats)
            return x14, x4
        else:
            return x14

def eva_x_tiny_patch16(pretrained=False):
    model = EVA_X(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14, 14),  # 224/16
    )
    eva_ckpt = checkpoint_filter_fn(torch.load(pretrained, map_location='cpu'), 
                        model)
    msg = model.load_state_dict(eva_ckpt, strict=False)
    print(msg)
    return model

def eva_x_small_patch16(pretrained=False):
    model = EVA_X(
        img_size=224, 
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14, 14),   # 224/16
        # num_classes = 14,
        # drop_path_rate=0.2
    )
    eva_ckpt = checkpoint_filter_fn(torch.load(pretrained, map_location='cpu', weights_only=False), 
                        model)
    msg = model.load_state_dict(eva_ckpt, strict=False)
    print(msg)
    return model

def eva_x_base_patch16(pretrained=False):
    model = EVA_X(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14, 14),  # 224/16
        num_classes = 14,
    )
    eva_ckpt = checkpoint_filter_fn(torch.load(pretrained, map_location='cpu',weights_only=False), 
                        model)
    msg = model.load_state_dict(eva_ckpt, strict=False)
    print(msg)
    return model


import copy
import os
import warnings
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold
from timm.data import Mixup
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import optuna
from functools import partial
# from evax.eva_x import eva_x_base_patch16, eva_x_small_patch16
from albumentations.core.transforms_interface import ImageOnlyTransform

def seed_everything_torch(seed=42):
    import os
    import random
    import torch
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything_torch(42)
warnings.filterwarnings("ignore")

# --- 1. Configuration ---
class Config:
    DEBUG = False
    BASE_PATH = "/kaggle/input/grand-xray-slam-division-a/"
    TRAIN_IMG_PATH = os.path.join(BASE_PATH, "train1") #train_npy/train_npy
    TEST_IMG_PATH = os.path.join(BASE_PATH, "test1") #test_npy/test_npy
    TRAIN_CSV = os.path.join(BASE_PATH, "train1.csv")
    SAMPLE_SUB_CSV = os.path.join(BASE_PATH, "sample_submission_1.csv")
    
    IMAGE_COLUMN_NAME = 'Image_name'
    VIEW_POSITION_COLUMN = 'ViewPosition'
    MODEL_NAME = 'convnext_tiny.fb_in22k'
    IMG_SIZE = 224 
    BATCH_SIZE = 64
    EPOCHS_INITIAL = 9
    EPOCHS_PSEUDO = 3
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DROP_RATE = 0.10
    DROP_PATH_RATE = 0.10
    PSEUDO_CONFIDENCE_HIGH = 0.97
    PSEUDO_CONFIDENCE_LOW = 0.03
    DEBUG_PSEUDO_SAMPLE_SIZE = 500
    DEBUG_FINAL_INFERENCE_SIZE = 50
    
    TARGET_LABELS = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
        'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 
        'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]
    NUM_CLASSES = len(TARGET_LABELS)
    VIEW_POSITION_LABELS = ['AP', 'PA', 'Lateral', 'LL']
    NUM_VIEW_CLASSES = len(VIEW_POSITION_LABELS)
    LOSS_HEAD_WEIGHTS = (0.7, 0.3)
    NUM_FOLDS = 5

    
print(f"Using device: {Config.DEVICE}")
print(f"Using model: {Config.MODEL_NAME} with image size {Config.IMG_SIZE}")

# --- 2. Loss Functions ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        return focal_loss.mean()

class AsymmetricLossOptimized(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None
    def forward(self, x, y):
        self.targets = y
        self.anti_targets = 1 - y
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w
        return -self.loss.sum()
    
class HybridLoss(nn.Module):
    def __init__(self, bce_weights=None, alpha=0.3, beta=0.7, gamma_neg=1.2, gamma_pos=2.5, clip=0.05, eps=1e-8):
        """
        Hybrid loss combining weighted BCE and Asymmetric Loss Optimized (ASL).
        
        Args:
            bce_weights (torch.Tensor, optional): Weights for each class in BCE loss. Shape: (num_classes,).
            alpha (float): Weight for BCE loss in the hybrid combination.
            beta (float): Weight for ASL loss in the hybrid combination.
            gamma_neg (float): Gamma parameter for negative samples in ASL.
            gamma_pos (float): Gamma parameter for positive samples in ASL.
            clip (float): Clipping margin for ASL to stabilize negative samples.
            eps (float): Small value to prevent log(0) in ASL.
        """
        super(HybridLoss, self).__init__()
        self.bce_weights = bce_weights if bce_weights is not None else torch.ones(Config.NUM_CLASSES).to(Config.DEVICE)
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss(weight=self.bce_weights, reduction='mean')
        self.asl = AsymmetricLossOptimized(
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
            eps=eps,
            disable_torch_grad_focal_loss=False
        )

    def forward(self, inputs, targets):
        """
        Compute the hybrid loss.
        
        Args:
            inputs (torch.Tensor): Model predictions (logits), shape: (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels, shape: (batch_size, num_classes).
        
        Returns:
            torch.Tensor: Combined loss value.
        """
        bce_loss = self.bce(inputs, targets)
        asl_loss = self.asl(inputs, targets)
        return self.alpha * bce_loss + self.beta * asl_loss
        
def exclusivity_regularizer(p, y, idx_nf, cond_on_ynf=True):
    mask = torch.ones(p.size(1), dtype=torch.bool, device=p.device)
    mask[idx_nf] = False
    p_any = 1.0 - torch.prod(1.0 - p[:, mask], dim=1)
    reg = p[:, idx_nf] * p_any
    if cond_on_ynf:
        reg = reg * y[:, idx_nf]
    return reg.mean()

def multilabel_mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def multilabel_cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    y_mix = lam * y + (1 - lam) * y[index, :]
    return x, y_mix

def mixup_cutmix(x, y, mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5):
    if np.random.rand() < prob:
        if np.random.rand() < switch_prob:
            return multilabel_mixup(x, y, alpha=mixup_alpha)
        else:
            return multilabel_cutmix(x, y, alpha=cutmix_alpha)
    return x, y

def get_scheduler(optimizer, warmup_epochs, total_epochs, base_lr, min_lr):
    def warmup_fn(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(total_epochs - warmup_epochs),
        eta_min=min_lr
    )
    return warmup_scheduler, cosine_scheduler

# --- 3. EMA Helper ---
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)
    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(v * self.decay + msd[k].detach() * (1. - self.decay))

# --- 4. Load Data ---
print("Loading data...")
train_df = pd.read_csv(Config.TRAIN_CSV)
sample_submission_df = pd.read_csv(Config.SAMPLE_SUB_CSV)

def construct_image_path(base_path, image_name):
    npy_path = os.path.join(base_path, image_name.replace('.jpg', '.npy'))
    jpg_path = os.path.join(base_path, image_name)
    if os.path.exists(npy_path):
        return npy_path
    elif os.path.exists(jpg_path):
        return jpg_path
    else:
        raise FileNotFoundError(f"No file found for {image_name} at {npy_path} or {jpg_path}")

train_df['ImagePath'] = train_df[Config.IMAGE_COLUMN_NAME].apply(
    lambda x: construct_image_path(Config.TRAIN_IMG_PATH, x)
)
sample_submission_df['ImagePath'] = sample_submission_df[Config.IMAGE_COLUMN_NAME].apply(
    lambda x: construct_image_path(Config.TEST_IMG_PATH, x)
)

if Config.DEBUG:
    train_df = train_df.sample(frac=0.01, random_state=42).reset_index(drop=True)

mskf = MultilabelStratifiedKFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=42)
fold_indices = list(mskf.split(train_df, train_df[Config.TARGET_LABELS].values))

# --- 5. Dataset ---
class ChestXRayDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test
        self.view_map = {'AP': 0, 'PA': 1, 'Lateral': 2, 'LL': 3}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['ImagePath']
        # Determine file type based on extension
        if img_path.endswith('.npy'):
            image = np.load(img_path).astype(np.uint8)
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)
        elif img_path.endswith('.jpg') or img_path.endswith('.jpeg'):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            if image is None:
                raise FileNotFoundError(f"Could not load image at {img_path}")
        else:
            raise ValueError(f"Unsupported file format for {img_path}")

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        if self.is_test:
            return image
        else:
            labels = self.df.iloc[idx][Config.TARGET_LABELS].values.astype(np.float32)
            view_position = self.df.iloc[idx][Config.VIEW_POSITION_COLUMN]
            view_idx = self.view_map[view_position]
            view_one_hot = np.zeros(Config.NUM_VIEW_CLASSES, dtype=np.float32)
            view_one_hot[view_idx] = 1.0
            return image, torch.tensor(labels, dtype=torch.float32), torch.tensor(view_one_hot, dtype=torch.float32)
            
train_transform = A.Compose([
    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
    A.HorizontalFlip(p=0.6),
    A.Rotate(limit=15, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.4),
    A.GridDistortion(
        num_steps=5,
        distort_limit=0.2,
        p=0.5
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
# --- 6. Model ---

def get_model():
    model = eva_x_base_patch16(pretrained='/kaggle/input/eva-x-base-pos/evax_base_pos_fold0.pth')
    # model = eva_x_small_patch16(pretrained='eva_x_small_patch16_merged520k_mim.pt')
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, Config.NUM_CLASSES)  
    # model = timm.create_model(Config.MODEL_NAME, pretrained=True, num_classes=Config.NUM_CLASSES, drop_rate=Config.DROP_RATE, drop_path_rate=Config.DROP_PATH_RATE)
    return model.to(Config.DEVICE)

# --- 7. Training Loop (with EMA + AMP) ---
def run_training(model, train_loader, val_loader, criterion, ce_criterion, optimizer, scheduler, epochs, save_path, fold):
    best_auc = 0
    scaler = GradScaler()
    ema = ModelEMA(model)
    log_file = f"training_log_pos_fold{fold}.txt"
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for images, labels, view_one_hot in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            view_one_hot = view_one_hot.to(Config.DEVICE)
            # images, labels = mixup_cutmix(
            #     images, labels,
            #     mixup_alpha=0.2,
            #     cutmix_alpha=0.1,
            #     prob=0.5,        
            #     switch_prob=0.5 
            # )
            optimizer.zero_grad()
            with autocast():
                outputs, out4 = model(images, out4=True)
                base_loss = criterion(outputs, labels)
                loss4 = ce_criterion(out4, view_one_hot)
                loss = base_loss + 0.1 * loss4
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)
            ema.update(model)

        model.eval()
        val_loss = 0
        all_labels_14, all_preds_14, all_labels_4, all_preds_4 = [], [], [], []
        with torch.no_grad():
            for images, labels, view_one_hot in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{epochs}"):
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                view_one_hot = view_one_hot.to(Config.DEVICE)
                outputs, out4 = ema.ema(images, out4=True)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                all_labels_14.append(labels.cpu().numpy())
                all_preds_14.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels_4.append(view_one_hot.cpu().numpy())
                all_preds_4.append(F.softmax(out4, dim=1).cpu().numpy())

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        all_labels_14 = np.concatenate(all_labels_14)
        all_preds_14 = np.concatenate(all_preds_14)
        val_auc_14 = roc_auc_score(all_labels_14, all_preds_14, average='macro')
        per_class_auc_14 = []
        for i, label_name in enumerate(Config.TARGET_LABELS):
            try:
                auc = roc_auc_score(all_labels_14[:, i], all_preds_14[:, i])
                per_class_auc_14.append(auc)
            except ValueError:
                per_class_auc_14.append(np.nan)
        all_labels_4 = np.concatenate(all_labels_4)
        all_preds_4 = np.concatenate(all_preds_4)
        val_auc_4 = roc_auc_score(all_labels_4, all_preds_4, multi_class='ovr', average='macro')
        per_class_auc_4 = []
        for i, label_name in enumerate(Config.VIEW_POSITION_LABELS):
            try:
                auc = roc_auc_score(all_labels_4[:, i], all_preds_4[:, i])
                per_class_auc_4.append(auc)
            except ValueError:
                per_class_auc_4.append(np.nan)

        with open(log_file, 'a') as f:
            f.write(f"\nEpoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro AUC (14 classes): {val_auc_14:.4f} | Val Macro AUC (View Positions): {val_auc_4:.4f}\n")
            f.write("Per-class AUC scores (14 classes):\n")
            for label_name, auc in zip(Config.TARGET_LABELS, per_class_auc_14):
                f.write(f"  {label_name}: {auc:.4f}\n" if not np.isnan(auc) else f"  {label_name}: NaN (insufficient data)\n")
            f.write("Per-class AUC scores (View Positions):\n")
            for label_name, auc in zip(Config.VIEW_POSITION_LABELS, per_class_auc_4):
                f.write(f"  {label_name}: {auc:.4f}\n" if not np.isnan(auc) else f"  {label_name}: NaN (insufficient data)\n")

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro AUC (14 classes): {val_auc_14:.4f} | Val Macro AUC (View Positions): {val_auc_4:.4f} | Metrics saved to {log_file}")

        scheduler.step(val_auc_14)
        if val_auc_14 > best_auc:
            best_auc = val_auc_14
            torch.save(ema.ema.state_dict(), save_path)
            with open(log_file, 'a') as f:
                f.write(f"New best model saved with Macro AUC (14 classes): {best_auc:.4f} at {save_path}\n")
            print(f"New best model saved with Macro AUC (14 classes): {best_auc:.4f} at {save_path}")
    return best_auc


# # --- 8. Stage 1: Training with Stratified K-Fold ---
# print("\n" + "="*20 + " STAGE 1: Initial Stratified K-Fold Training " + "="*20)
# initial_model_paths = []
# for fold in range(Config.NUM_FOLDS):
#     print(f"\nTraining Fold {fold+1}/{Config.NUM_FOLDS}")
#     train_idx, val_idx = fold_indices[fold]
#     train_split_df = train_df.iloc[train_idx].reset_index(drop=True)
#     val_split_df = train_df.iloc[val_idx].reset_index(drop=True)
#     train_loader = DataLoader(
#         ChestXRayDataset(train_split_df, transform=train_transform),
#         batch_size=Config.BATCH_SIZE,
#         shuffle=True,
#         num_workers=Config.NUM_WORKERS
#     )
#     val_loader = DataLoader(
#         ChestXRayDataset(val_split_df, transform=val_transform),
#         batch_size=Config.BATCH_SIZE,
#         shuffle=False,
#         num_workers=Config.NUM_WORKERS
#     )
#     model = get_model()
#     class_frequencies = train_df[Config.TARGET_LABELS].mean().values
#     bce_weights = torch.tensor(1.0 / (class_frequencies + 1e-6)).to(Config.DEVICE)
#     criterion = AsymmetricLossOptimized(gamma_neg=1.2, gamma_pos=2.5, clip=0.05)
#     # criterion = FocalZLPR(tau=0.4, reduction='sum')
#     # criterion = HybridLoss(
#     #     bce_weights=bce_weights,
#     #     alpha=0.1,
#     #     beta=1,
#     #     gamma_neg=1.2,
#     #     gamma_pos=3.1,
#     #     clip=0.05,
#     #     eps=1e-8
#     # )
#     ce_criterion = nn.CrossEntropyLoss()
#     # optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
#     head_params = list(model.head.parameters()) + list(model.head4.parameters())

#     optimizer = torch.optim.AdamW([
#         {'params': [p for p in model.parameters() if id(p) not in list(map(id, head_params))],
#         'lr': Config.LEARNING_RATE, 'weight_decay': 1e-4},
#         {'params': head_params,
#         'lr': Config.LEARNING_RATE * 10, 'weight_decay': 1e-4}
#     ])
#     scheduler = optim.lr_scheduler.OneCycleLR(
#         optimizer,
#         max_lr=Config.LEARNING_RATE * 10,
#         epochs=Config.EPOCHS_INITIAL,
#         steps_per_epoch=len(train_loader),
#         pct_start=0.4,
#         anneal_strategy='cos',
#         div_factor=25.0,
#         final_div_factor=1e4
#     )
#     save_path = f"initial_best_model_pos_fold{fold}.pth"
#     run_training(model, train_loader, val_loader, criterion, ce_criterion, optimizer, scheduler, Config.EPOCHS_INITIAL, save_path, fold)
#     initial_model_paths.append(save_path)

# --- FINAL INFERENCE with Ensemble ---
print("\n" + "="*20 + " FINAL INFERENCE with Ensemble " + "="*20)
final_models = []
initial_model_paths= ['/kaggle/input/eva-x-base-pos/evax_base_pos_fold0.pth',
                     '/kaggle/input/eva-x-base-pos/evax_base_pos_fold1.pth',
                     '/kaggle/input/eva-x-base-pos/evax_base_pos_fold2.pth',
                     '/kaggle/input/eva-x-base-pos/evax_base_pos_fold3.pth',
                     '/kaggle/input/eva-x-base-pos/evax_base_pos_fold4.pth']
for path in initial_model_paths:
    model = get_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    final_models.append(model)

final_test_df = sample_submission_df.sample(n=Config.DEBUG_FINAL_INFERENCE_SIZE, random_state=42) if Config.DEBUG else sample_submission_df
if Config.DEBUG: print(f"Debug mode ON: Running final inference on {len(final_test_df)} samples.")

final_test_loader = DataLoader(ChestXRayDataset(final_test_df, val_transform, is_test=True), batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
final_preds = []
with torch.no_grad():
    for images in tqdm(final_test_loader, desc="Final Predicting"):
        avg_outputs = torch.zeros((images.size(0), Config.NUM_CLASSES), device=Config.DEVICE)
        for model in final_models:
            outputs = model(images.to(Config.DEVICE))
            avg_outputs += outputs / Config.NUM_FOLDS
        final_preds.append(torch.sigmoid(avg_outputs).cpu().numpy())

predictions = np.concatenate(final_preds)
submission_df = pd.DataFrame(predictions, columns=Config.TARGET_LABELS)
submission_df[Config.IMAGE_COLUMN_NAME] = final_test_df[Config.IMAGE_COLUMN_NAME].values
submission_df = submission_df[[Config.IMAGE_COLUMN_NAME] + Config.TARGET_LABELS]
submission_df.to_csv("submission.csv", index=False)

print("Submission file created successfully!")
print(submission_df.head())
print("\nAll fold models saved for ensemble:")
for path in initial_model_paths:
    print(path)




