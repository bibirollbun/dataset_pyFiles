RUN_TRAIN = True # bfloat16 or float32 recommended
RUN_VALID = True
RUN_TEST  = True

import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError("Requires >= 2 GPUs with CUDA enabled.")

try: 
    import monai
except: 
    !pip install --no-deps monai -q


%%writefile _cfg.py

from types import SimpleNamespace
import torch

cfg= SimpleNamespace()
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.local_rank = 0
cfg.seed = 42
cfg.subsample = None

cfg.backbone = "convnext_small.fb_in22k_ft_in1k"
cfg.ema = True
cfg.ema_decay = 0.99

cfg.epochs = 25
cfg.batch_size = 64
cfg.batch_size_val = 16

cfg.early_stopping = {"patience": 10, "streak": 0}
cfg.logging_steps = 100

cfg.dataset2classidx = {
    "FlatVel_A": 0,
    "FlatVel_B": 1,
    "Style_A": 2,
    "Style_B": 3,
    "CurveVel_A": 4,
    "CurveVel_B": 5,
    "CurveFault_A": 6,
    "CurveFault_B": 7,
    "FlatFault_A": 8,
    "FlatFault_B": 9,
}


%%writefile _dataset.py

import os
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        cfg,
        mode = "train", 
    ):
        self.cfg = cfg
        self.mode = mode
        
        self.data, self.labels, self.records = self.load_metadata()

    def load_metadata(self, ):

        # Select rows
        df= pd.read_csv("/kaggle/input/openfwi-preprocessed-72x72/folds.csv")

        
        if self.cfg.subsample is not None:
            df= df.groupby(["dataset", "fold"]).head(self.cfg.subsample)

        if self.mode == "train":
            df= df[df["fold"] != 0]
        else:
            df= df[df["fold"] == 0]

        
        data = []
        labels = []
        records = []
        mmap_mode = "r"

        for idx, row in tqdm(df.iterrows(), total=len(df), disable=self.cfg.local_rank != 0):
            row= row.to_dict()

            # Hacky way to get exact file name
            p1 = os.path.join("/kaggle/input/open-wfi-1/openfwi_float16_1/", row["data_fpath"])
            p2 = os.path.join("/kaggle/input/open-wfi-1/openfwi_float16_1/", row["data_fpath"].split("/")[0], "*", row["data_fpath"].split("/")[-1])
            p3 = os.path.join("/kaggle/input/open-wfi-2/openfwi_float16_2/", row["data_fpath"])
            p4 = os.path.join("/kaggle/input/open-wfi-2/openfwi_float16_2/", row["data_fpath"].split("/")[0], "*", row["data_fpath"].split("/")[-1])
            farr= glob.glob(p1) + glob.glob(p2) + glob.glob(p3) + glob.glob(p4)
        
            # Map to lbl fpath
            farr= farr[0]
            flbl= farr.replace('seis', 'vel').replace('data', 'model')
            
            # Load
            arr= np.load(farr, mmap_mode=mmap_mode)
            lbl= np.load(flbl, mmap_mode=mmap_mode)

            # Append
            data.append(arr)
            labels.append(lbl)
            records.append(row["dataset"])

        return data, labels, records

    def __getitem__(self, idx):
        row_idx= idx // 500
        col_idx= idx % 500

        d= self.records[row_idx]
        x= self.data[row_idx][col_idx, ...]
        y= self.labels[row_idx][col_idx, ...]

        # Augs 
        if self.mode == "train":
            
            # Temporal flip
            if np.random.random() < 0.5:
                x= x[::-1, :, ::-1]
                y= y[..., ::-1]

        x= x.copy()
        y= y.copy()
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        type_label = self.cfg.dataset2classidx[d]  # 需要在cfg里定义
        type_label = torch.tensor(type_label).long()
        
        return x, y,type_label 

    def __len__(self, ):
        return len(self.records) * 500


import pandas as pd

# 读取CSV文件
df = pd.read_csv("/kaggle/input/openfwi-preprocessed-72x72/folds.csv")

# 计算总样本数
total_count = len(df)

# 过滤指定类别
filtered_df = df[df["dataset"].isin(["Style_B", "CurveFault_B", "FlatFault_B"])]

# 计算指定类别的样本数
filtered_count = len(filtered_df)

# 计算占比（百分比）
percentage = (filtered_count / total_count) * 100

print(f"总样本数: {total_count}")
print(f"指定类别样本数: {filtered_count}")
print(f"占比: {percentage:.2f}%")



%%writefile _model.py
from copy import deepcopy
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.convnext import ConvNeXtBlock

from monai.networks.blocks import UpSample, SubpixelUpsample

####################
## EMA + Ensemble ##
####################

class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.99, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models).eval()

    def forward(self, x):
        output = None
        for m in self.models:
            logits= m(x)
            if output is None:
                output = logits
            else:
                output += logits
        output /= len(self.models)
        return output
        
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models).eval()
            
    def forward(self, x):
        output = None
        for m in self.models:
            out = m(x)  # 先不解包
            if isinstance(out, tuple):  # 如果是元组，取第一个元素（logits）
                logits = out[0]
            else:  # 如果不是元组，直接使用
                logits = out
            
            if output is None:
                output = logits
            else:
                output += logits
        output /= len(self.models)##
        return output, None  # 返回 (logits, None)
#############
## Decoder ##
#############

class ConvBnAct2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding: int = 0,
        stride: int = 1,
        norm_layer: nn.Module = nn.Identity,
        act_layer: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.conv= nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size,
            stride=stride, 
            padding=padding, 
            bias=False,
        )
        self.norm = norm_layer(out_channels) if norm_layer != nn.Identity else nn.Identity()
        self.act= act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SCSEModule2d(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.Tanh(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1), 
            nn.Sigmoid(),
            )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class Attention2d(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule2d(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

class DecoderBlock2d(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer: nn.Module = nn.Identity,
        attention_type: str = None,
        intermediate_conv: bool = False,
        upsample_mode: str = "deconv",
        scale_factor: int = 2,
    ):
        super().__init__()

        # Upsample block
        if upsample_mode == "pixelshuffle":
            self.upsample= SubpixelUpsample(
                spatial_dims= 2,
                in_channels= in_channels,
                scale_factor= scale_factor,
            )
        else:
            self.upsample = UpSample(
                spatial_dims= 2,
                in_channels= in_channels,
                out_channels= in_channels,
                scale_factor= scale_factor,
                mode= upsample_mode,
            )

        if intermediate_conv:
            k= 3
            c= skip_channels if skip_channels != 0 else in_channels
            self.intermediate_conv = nn.Sequential(
                ConvBnAct2d(c, c, k, k//2),
                ConvBnAct2d(c, c, k, k//2),
                )
        else:
            self.intermediate_conv= None

        self.attention1 = Attention2d(
            name= attention_type, 
            in_channels= in_channels + skip_channels,
            )

        self.conv1 = ConvBnAct2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size= 3,
            padding= 1,
            norm_layer= norm_layer,
        )

        self.conv2 = ConvBnAct2d(
            out_channels,
            out_channels,
            kernel_size= 3,
            padding= 1,
            norm_layer= norm_layer,
        )
        self.attention2 = Attention2d(
            name= attention_type, 
            in_channels= out_channels,
            )

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if self.intermediate_conv is not None:
            if skip is not None:
                skip = self.intermediate_conv(skip)
            else:
                x = self.intermediate_conv(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class UnetDecoder2d(nn.Module):
    """
    Unet decoder.
    Source: https://arxiv.org/abs/1505.04597
    """
    def __init__(
        self,
        encoder_channels: tuple[int],
        skip_channels: tuple[int] = None,
        decoder_channels: tuple = (256, 128, 64, 32),
        scale_factors: tuple = (2,2,2,2),
        norm_layer: nn.Module = nn.Identity,
        # attention_type: str = "scse",
        attention_type: str = None,
        intermediate_conv: bool = False,
        upsample_mode: str = "deconv",
    ):
        super().__init__()
        
        if len(encoder_channels) == 4:
            decoder_channels= decoder_channels[1:]
        self.decoder_channels= decoder_channels
        
        if skip_channels is None:
            skip_channels= list(encoder_channels[1:]) + [0]

        # Build decoder blocks
        in_channels= [encoder_channels[0]] + list(decoder_channels[:-1])
        self.blocks = nn.ModuleList()

        for i, (ic, sc, dc) in enumerate(zip(in_channels, skip_channels, decoder_channels)):
            self.blocks.append(
                DecoderBlock2d(
                    ic, sc, dc, 
                    norm_layer= norm_layer,
                    attention_type= attention_type,
                    intermediate_conv= intermediate_conv,
                    upsample_mode= upsample_mode,
                    scale_factor= scale_factors[i],
                    )
            )

    def forward(self, feats: list[torch.Tensor]):
        res= [feats[0]]
        feats= feats[1:]

        for i, b in enumerate(self.blocks):
            skip= feats[i] if i < len(feats) else None
            res.append(
                b(res[-1], skip=skip),
                )
            
        return res

class SegmentationHead2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor: tuple[int] = (2,2),
        kernel_size: int = 3,
        mode: str = "nontrainable",
    ):
        super().__init__()
        self.conv= nn.Conv2d(
            in_channels, out_channels, kernel_size= kernel_size,
            padding= kernel_size//2
        )
        self.upsample = UpSample(
            spatial_dims= 2,
            in_channels= out_channels,
            out_channels= out_channels,
            scale_factor= scale_factor,
            mode= mode,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x
        

#############
## Encoder ##
#############

def _convnext_block_forward(self, x):
    shortcut = x
    x = self.conv_dw(x)

    if self.use_conv_mlp:
        x = self.norm(x)
        x = self.mlp(x)
    else:
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()

    if self.gamma is not None:
        x = x * self.gamma.reshape(1, -1, 1, 1)

    x = self.drop_path(x) + self.shortcut(shortcut)
    return x


class Net(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool = True,
        n_classes: int = 10,
    ):
        super().__init__()
        
        self.backbone= timm.create_model(
            backbone,
            in_chans= 5,
            pretrained= pretrained,
            features_only= True,
            drop_path_rate=0.0,
        )
        ecs= [_["num_chs"] for _ in self.backbone.feature_info][::-1]
        # self.backbone.set_grad_checkpointing()

        self.decoder= UnetDecoder2d(
            encoder_channels= ecs,
        )

        self.seg_head= SegmentationHead2d(
            in_channels= self.decoder.decoder_channels[-1],
            out_channels= 1,
            scale_factor= 1,
        )

        # 分类头：联合loss
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.decoder.decoder_channels[-1], n_classes)
        )
        
        self._update_stem(backbone)
        self.replace_activations(self.backbone, log=True)
        self.replace_norms(self.backbone, log=True)
        self.replace_forwards(self.backbone, log=True)

    def _update_stem(self, backbone):
        if backbone.startswith("convnext"):
            self.backbone.stem_0.stride = (4, 1)
            self.backbone.stem_0.padding = (0, 2)
            with torch.no_grad():
                w = self.backbone.stem_0.weight
                new_conv= nn.Conv2d(w.shape[0], w.shape[0], kernel_size=(4, 4), stride=(4, 1), padding=(0, 1))
                new_conv.weight.copy_(w.repeat(1, (128//w.shape[1])+1, 1, 1)[:, :new_conv.weight.shape[1], :, :])
                new_conv.bias.copy_(self.backbone.stem_0.bias)
            self.backbone.stem_0= nn.Sequential(
                nn.ReflectionPad2d((1,1,80,80)),
                self.backbone.stem_0,
                new_conv,
            )
        else:
            raise ValueError("Custom striding not implemented.")
        pass

    def replace_activations(self, module, log=False):
        if log:
            print(f"Replacing all activations with GELU...")
        for name, child in module.named_children():
            if isinstance(child, (
                nn.ReLU, nn.LeakyReLU, nn.Mish, nn.Sigmoid, 
                nn.Tanh, nn.Softmax, nn.Hardtanh, nn.ELU, 
                nn.SELU, nn.PReLU, nn.CELU, nn.GELU, nn.SiLU,
            )):
                setattr(module, name, nn.GELU())
            else:
                self.replace_activations(child)

    def replace_norms(self, mod, log=False):
        if log:
            print(f"Replacing all norms with InstanceNorm...")
        for name, c in mod.named_children():
            n_feats= None
            if isinstance(c, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                n_feats= c.num_features
            elif isinstance(c, (nn.GroupNorm,)):
                n_feats= c.num_channels
            elif isinstance(c, (nn.LayerNorm,)):
                n_feats= c.normalized_shape[0]
            if n_feats is not None:
                new = nn.InstanceNorm2d(
                    n_feats,
                    affine=True,
                )
                setattr(mod, name, new)
            else:
                self.replace_norms(c)

    def replace_forwards(self, mod, log=False):
        if log:
            print(f"Replacing forward functions...")
        for name, c in mod.named_children():
            if isinstance(c, ConvNeXtBlock):
                c.forward = MethodType(_convnext_block_forward, c)
            else:
                self.replace_forwards(c)

    def proc_flip(self, x_in):
        x_in= torch.flip(x_in, dims=[-3, -1])
        x= self.backbone(x_in)
        x= x[::-1]
        x= self.decoder(x)
        x_seg= self.seg_head(x[-1])
        x_seg= x_seg[..., 1:-1, 1:-1]
        x_seg= torch.flip(x_seg, dims=[-1])
        x_seg= x_seg * 1500 + 3000
        type_logits = self.classifier(x[-1])
        return x_seg, type_logits

    def forward(self, batch):
        x_in = batch
        x = self.backbone(x_in)
        x = x[::-1]
        x = self.decoder(x)
        dec_out = x[-1]
        x_seg = self.seg_head(dec_out)
        x_seg = x_seg[..., 1:-1, 1:-1]
        x_seg = x_seg * 1500 + 3000
        type_logits = self.classifier(dec_out)
        if self.training:
            return x_seg, type_logits
        else:
            p1, p1_type = self.proc_flip(x_in)
            x_seg = torch.mean(torch.stack([x_seg, p1]), dim=0)
            type_logits = torch.mean(torch.stack([type_logits, p1_type]), dim=0)
            return x_seg, type_logits



%%writefile _utils.py

import datetime

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


%%writefile _train.py
import os
import time 
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from _cfg import cfg
from _dataset import CustomDataset
from _model import ModelEMA, Net
from _utils import format_time

def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return

def cleanup():
    dist.barrier()
    dist.destroy_process_group()
    return

import math

from torch.optim.lr_scheduler import _LRScheduler

class ConstantCosineLR(_LRScheduler):
    def __init__(
        self, 
        optimizer,
        total_steps, 
        pct_cosine, 
        min_lr=1e-5,              # 新增参数
        last_epoch=-1,
    ):
        self.total_steps = total_steps
        self.milestone = int(total_steps * (1 - pct_cosine))
        self.cosine_steps = max(total_steps - self.milestone, 1)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.milestone:
            factor = 1.0
        else:
            s = step - self.milestone
            factor = 0.5 * (1 + math.cos(math.pi * s / self.cosine_steps))
        # 支持 min_lr
        return [
            self.min_lr + (lr - self.min_lr) * factor
            for lr in self.base_lrs
        ]

def format_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))
def main(cfg):
    # 去除WandB

    # ========== Datasets / Dataloaders ==========
    if cfg.local_rank == 0:
        print("="*25)
        print("Loading data..")
    train_ds = CustomDataset(cfg=cfg, mode="train")
    sampler = DistributedSampler(
        train_ds, 
        num_replicas=cfg.world_size, 
        rank=cfg.local_rank,
    )
    train_dl = DataLoader(
        train_ds, 
        sampler=sampler,
        batch_size=cfg.batch_size, 
        num_workers=4,
    )
    
    valid_ds = CustomDataset(cfg=cfg, mode="valid")
    sampler = DistributedSampler(
        valid_ds, 
        num_replicas=cfg.world_size, 
        rank=cfg.local_rank,
    )
    valid_dl = DataLoader(
        valid_ds, 
        sampler=sampler,
        batch_size=cfg.batch_size_val, 
        num_workers=4,
    )

    # ========== Model / Optim ==========
    model = Net(backbone=cfg.backbone)
    model = model.to(cfg.local_rank)

    # Resume training (在DDP前加载)
    if cfg.resume_path is not None and os.path.exists(cfg.resume_path):
        if cfg.local_rank == 0:
            print(f"Resuming training from {cfg.resume_path}")
        map_location = {"cuda:%d" % 0: "cuda:%d" % cfg.local_rank}
        state_dict = torch.load(cfg.resume_path, map_location=map_location)
    
        def strip_prefix(state_dict):
            new_sd = {}
            for k, v in state_dict.items():
                k = k.replace('module.', '')
                k = k.replace('_orig_mod.', '')
                new_sd[k] = v
            return new_sd
        state_dict = strip_prefix(state_dict)
    
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        print("In model but not in checkpoint:", sorted(model_keys - ckpt_keys))
        print("In checkpoint but not in model:", sorted(ckpt_keys - model_keys))
    
        def filter_state_dict(model, state_dict):
            model_dict = model.state_dict()
            filtered_dict = {}
            for k, v in state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    print(f"Skip loading {k}: checkpoint shape {v.shape}, model shape {model_dict[k].shape if k in model_dict else 'N/A'}")
            return filtered_dict
    
        # 主模型加载过滤后的权重
        filtered_state_dict = filter_state_dict(model, state_dict)
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
    
    if cfg.ema:
        ema_model = ModelEMA(model, decay=cfg.ema_decay, device=cfg.local_rank)
        if cfg.resume_path is not None and os.path.exists(cfg.resume_path):
            # EMA模型也加载过滤后的权重
            ema_state_dict = filter_state_dict(ema_model.module, state_dict)
            ema_model.module.load_state_dict(ema_state_dict, strict=False)
    else:
        ema_model = None

    model = DistributedDataParallel(
        model, 
        device_ids=[cfg.local_rank], 
        find_unused_parameters=True
    )

    criterion = nn.L1Loss()
    
    for param in model.module.backbone.parameters():
        param.requires_grad = False  # 冻结backbone
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scaler = GradScaler()
    criterion_cls = nn.CrossEntropyLoss()

    # 新增调度器初始化 -------------------------------------------------
    total_train_steps = cfg.epochs * len(train_dl)  # 总训练步数 = epoch数 × 每epoch步数
    scheduler = ConstantCosineLR(
        optimizer,
        total_steps=total_train_steps,
        pct_cosine=0.3  # 最后30%训练步使用cosine衰减
    )

    # ========== Training ==========
    if cfg.local_rank == 0:
        print("="*25)
        print("Give me warp {}, Mr. Sulu.".format(cfg.world_size))
        print("="*25)
    
    best_loss = 1_000_000
    val_loss = 1_000_000

    for epoch in range(0, cfg.epochs+1):
        if epoch != 0:
            tstart = time.time()
            train_dl.sampler.set_epoch(epoch)
    
            # Train loop
            model.train()
            total_loss = []
            for i, (x, y, type_label) in enumerate(train_dl):

                x = x.to(cfg.local_rank)
                y = y.to(cfg.local_rank)
                type_label = type_label.to(cfg.local_rank)
            
                with autocast(cfg.device.type):
                    seg_pred, type_logits = model(x)

                if torch.isnan(seg_pred).any() or torch.isinf(seg_pred).any():
                    print('seg_pred has nan or inf!')
                if torch.isnan(type_logits).any() or torch.isinf(type_logits).any():
                    print('type_logits has nan or inf!')
                if torch.isnan(y).any() or torch.isinf(y).any():
                    print('y has nan or inf!')
                if torch.isnan(type_label).any() or torch.isinf(type_label).any():
                    print('type_label has nan or inf!')
            
                loss_seg = criterion(seg_pred, y)
                loss_cls = criterion_cls(type_logits, type_label)
                loss = loss_seg + cfg.cls_weight * loss_cls  
            
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
                total_loss.append(loss.item())
                
                if ema_model is not None:
                    ema_model.update(model)
                    
                if cfg.local_rank == 0 and (len(total_loss) >= cfg.logging_steps or i == 0):
                    train_loss = np.mean(total_loss)
                    total_loss = []
                    print("Epoch {}:     Train MAE: {:.2f}     Val MAE: {:.2f}     Time: {}     Step: {}/{}".format(
                        epoch, 
                        train_loss,
                        val_loss,
                        format_time(time.time() - tstart),
                        i+1, 
                        len(train_dl)+1, 
                    ))

        # ========== Valid ==========
        model.eval()
        val_logits = []
        val_targets = []
        with torch.no_grad():
            for x, y, type_label in tqdm(valid_dl, disable=cfg.local_rank != 0):
                x = x.to(cfg.local_rank)
                y = y.to(cfg.local_rank)
    
                with autocast(cfg.device.type):
                    if ema_model is not None:
                        out, type_logits = ema_model.module(x)
                    else:
                        out, type_logits = model(x)

                val_logits.append(out.cpu())
                val_targets.append(y.cpu())

            val_logits = torch.cat(val_logits, dim=0)
            val_targets = torch.cat(val_targets, dim=0)
                
            loss = criterion(val_logits, val_targets).item()

        # Gather loss
        v = torch.tensor([loss], device=cfg.local_rank)
        torch.distributed.all_reduce(v, op=dist.ReduceOp.SUM)
        val_loss = (v[0] / cfg.world_size).item()

        # ========== Weights / Early stopping ==========
        stop_train = torch.tensor([0], device=cfg.local_rank)
        if cfg.local_rank == 0:
            es = cfg.early_stopping
            if val_loss < best_loss:
                print("New best: {:.2f} -> {:.2f}".format(best_loss, val_loss))
                print("Saved weights..")
                best_loss = val_loss
                if ema_model is not None:
                    torch.save(ema_model.module.state_dict(), f'best_model_{cfg.seed}.pt')
                else:
                    torch.save(model.state_dict(), f'best_model_{cfg.seed}.pt')
        
                es["streak"] = 0
            else:
                es["streak"] += 1
                if es["streak"] > es["patience"]:
                    print("Ending training (early_stopping).")
                    # 保存最后一次的权重
                    if ema_model is not None:
                        torch.save(ema_model.module.state_dict(), f'last_model_{cfg.seed}.pt')
                    else:
                        torch.save(model.state_dict(), f'last_model_{cfg.seed}.pt')
                    stop_train = torch.tensor([1], device=cfg.local_rank)

        # Exits training on all ranks
        dist.broadcast(stop_train, src=0)
        if stop_train.item() == 1:
            return

    return


if __name__ == "__main__":

    # GPU Specs
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    _, total = torch.cuda.mem_get_info(device=rank)

    # Init
    setup(rank, world_size)
    time.sleep(rank)
    print(f"Rank: {rank}, World size: {world_size}, GPU memory: {total / 1024**3:.2f}GB", flush=True)
    time.sleep(world_size - rank)

    # Seed
    set_seed(cfg.seed+rank)

    # Run
    cfg.resume_path = "/kaggle/input/simple-further-finetuned-bartley-open-models/bartley_unet2d_convnext_seed1_epochbest_FT.pth"
    # cfg.resume_path = None # 如果不需要恢复训练就设置为 None

    cfg.local_rank= rank
    cfg.world_size= world_size
    cfg.name="calss"
    cfg.cls_weight=0.1
    cfg.batch_size = 512

    main(cfg)
    cleanup()


# if RUN_TRAIN:
#     print("Starting training..")
#     !OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 _train.py


# import glob

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from _cfg import cfg
# from _model import Net, EnsembleModel

# if RUN_VALID or RUN_TEST:

#     # Load pretrained models
#     models = []
#     # for f in sorted(glob.glob("/kaggle/input/simple-further-finetuned-bartley-open-models/*.pth")):
#     #     print("Loading: ", f)
#     #     m = Net(
#     #         backbone="convnext_small.fb_in22k_ft_in1k",
#     #         pretrained=False,
#     #     )
#     #     state_dict= torch.load(f, map_location=cfg.device, weights_only=True)

#     #     models.append(m)
#     m = Net(
#                 backbone="convnext_small.fb_in22k_ft_in1k",
#                 pretrained=False,
#             )
#     state_dict= torch.load('/kaggle/working/best_model_42.pt', map_location=cfg.device, weights_only=True)
#     models.append(m)
#     # Combine
#     model = EnsembleModel(models)
#     model = model.to(cfg.device)
#     model = model.eval()
#     print("n_models: {:_}".format(len(models)))


# from tqdm import tqdm
# import numpy as np

# import torch
# import torch.nn as nn
# from torch.amp import autocast

# from _dataset import CustomDataset


# if RUN_VALID:

#     # Dataset / Dataloader
#     valid_ds = CustomDataset(cfg=cfg, mode="valid")
#     sampler = torch.utils.data.SequentialSampler(valid_ds)
#     valid_dl = torch.utils.data.DataLoader(
#         valid_ds, 
#         sampler= sampler,
#         batch_size= cfg.batch_size_val, 
#         num_workers= 4,
#     )

#     # Valid loop
#     criterion = nn.L1Loss()
#     val_logits = []
#     val_targets = []
    
#     # with torch.no_grad():
#     #     for step, (x, y) in enumerate(tqdm(valid_dl)):
#     #         x = x.to(cfg.device)
#     #         y = y.to(cfg.device)
    
#     #         with autocast(cfg.device.type):
#     #             out = model(x)
    
#     #         val_logits.append(out.cpu())
#     #         val_targets.append(y.cpu())

#     #         #if step==10: break
    
#     #     val_logits= torch.cat(val_logits, dim=0)
#     #     val_targets= torch.cat(val_targets, dim=0)
#     with torch.no_grad():
#         for step, (x, y, _) in enumerate(tqdm(valid_dl)):
#             x = x.to(cfg.device)
#             y = y.to(cfg.device)
#             with autocast(cfg.device.type):
#                 out,_ = model(x)  # 先不解包
                
#             val_logits.append(out.cpu())
#             val_targets.append(y.cpu())
#         val_logits = torch.cat(val_logits, dim=0)  # 现在可以正确拼接
#         val_targets = torch.cat(val_targets, dim=0)

#         # == 统计被clamp掉的像素 ==
#         lower_bound = 1500
#         upper_bound = 4500
#         val_logits_np = val_logits.numpy()
#         num_total = val_logits_np.size
#         num_low = (val_logits_np < lower_bound).sum()
#         num_high = (val_logits_np > upper_bound).sum()
#         print(f"Clamp统计: 低于{lower_bound}: {num_low}, 高于{upper_bound}: {num_high}, 总数: {num_total}, 占比: {(num_low+num_high)/num_total:.5%}")

#         total_loss= criterion(val_logits, val_targets).item()

    
#     # Dataset Scores
#     ds_idxs= np.array([valid_ds.records])
#     ds_idxs= np.repeat(ds_idxs, repeats=500)
    
#     print("="*25)
#     with torch.no_grad():    
#         for idx in sorted(np.unique(ds_idxs)):
    
#             # Mask
#             mask = ds_idxs == idx
#             logits_ds = val_logits[mask]
#             targets_ds = val_targets[mask]
    
#             # Score predictions
#             loss = criterion(val_logits[mask], val_targets[mask]).item()
#             print("{:15} {:.2f}".format(idx, loss))
#     print("="*25)
#     print("Val MAE: {:.2f}".format(total_loss))
#     print("="*25)


# import torch

# class TestDataset(torch.utils.data.Dataset):
#     def __init__(self, test_files):
#         self.test_files = test_files

#     def __len__(self):
#         return len(self.test_files)

#     def __getitem__(self, i):
#         test_file = self.test_files[i]
#         test_stem = test_file.split("/")[-1].split(".")[0]
#         return np.load(test_file), test_stem



# import csv
# import time
# import glob
# from tqdm import tqdm
# import numpy as np
# import pandas as pd

# from _utils import format_time


# if RUN_TEST:

#     ss= pd.read_csv("/kaggle/input/waveform-inversion/sample_submission.csv")    
#     row_count = 0
#     t0 = time.time()
    
#     test_files = sorted(glob.glob("/kaggle/input/open-wfi-test/test/*.npy"))
#     x_cols = [f"x_{i}" for i in range(1, 70, 2)]
#     fieldnames = ["oid_ypos"] + x_cols
    
#     test_ds = TestDataset(test_files)
#     test_dl = torch.utils.data.DataLoader(
#         test_ds, 
#         sampler=torch.utils.data.SequentialSampler(test_ds),
#         batch_size=cfg.batch_size_val, 
#         num_workers=4,
#     )
    
#     with open("submission.csv", "wt", newline="") as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()

#         with torch.inference_mode():
#             with torch.autocast(cfg.device.type):
#                 for inputs, oids_test in tqdm(test_dl, total=len(test_dl)):
#                     inputs = inputs.to(cfg.device)
            
#                     outputs = model(inputs)
                            
#                     y_preds = outputs[:, 0].cpu().numpy()
                    
#                     for y_pred, oid_test in zip(y_preds, oids_test):
#                         for y_pos in range(70):
#                             row = dict(zip(x_cols, [y_pred[y_pos, x_pos] for x_pos in range(1, 70, 2)]))
#                             row["oid_ypos"] = f"{oid_test}_y_{y_pos}"
            
#                             writer.writerow(row)
#                             row_count += 1

#                             # Clear buffer
#                             if row_count % 100_000 == 0:
#                                 csvfile.flush()
    
#     t1 = format_time(time.time() - t0)
#     print(f"Inference Time: {t1}")


# import matplotlib.pyplot as plt 

# if RUN_TEST:
#     # Plot a few samples
#     fig, axes = plt.subplots(3, 5, figsize=(10, 6))
#     axes= axes.flatten()

#     n = min(len(outputs), len(axes))
    
#     for i in range(n):
#         img= outputs[0, 0, ...].cpu().numpy()
#         img = outputs[i, 0].cpu().numpy()
#         idx= oids_test[i]
    
#         # Plot
#         axes[i].imshow(img, cmap='gray')
#         axes[i].set_title(idx)
#         axes[i].axis('off')

#     for i in range(n, len(axes)):
#         axes[i].axis('off')
    
#     plt.tight_layout()
#     plt.show()

