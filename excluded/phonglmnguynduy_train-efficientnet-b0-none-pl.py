import os
from pathlib import Path
from types import SimpleNamespace

cfg = SimpleNamespace(**{})
cfg.num_folds = 5
cfg.fold = -1
cfg.gpu = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

cfg.fname = 'tf_efficientnetv2_s'
cfg.seed = 2025

cfg.input_path = Path('../input')
cfg.comp_data_path = cfg.input_path / 'birdclef-2025'
cfg.save_path = Path('../working')
cfg.soundscape_path = cfg.comp_data_path / 'train_soundscapes'
cfg.audio_path = Path("/kaggle/input/birdclef-2025/train_audio")

cfg.logger_file = True

# image size
cfg.image_height = cfg.image_width = 224

# audio
cfg.duration = 5
cfg.sr = 32000
cfg.fmin = 40
cfg.fmax = 16000
cfg.n_fft = 2048
cfg.n_mels = cfg.image_height
cfg.win_length = 512
# cfg.hop_length = int((cfg.duration * cfg.sr - cfg.win_length + cfg.n_fft) / (cfg.image_width)) + 1 
cfg.hop_length = int((cfg.duration * cfg.sr) / (cfg.image_width)) + 1 

# training HP
cfg.num_epochs = 15
cfg.train_batch_size = 128
cfg.valid_batch_size = 128
cfg.workers = 2
cfg.grad_value = 2.0
cfg.grad_norm = 0.0
cfg.grad_norm_type = 2
cfg.device = "cuda"
cfg.accumulate = 1

# optimizer
cfg.lr = 7e-5
cfg.decay = 0.01
cfg.opt_beta1 = 0.9
cfg.opt_beta2 = 0.999
cfg.opt_eps = 1e-8
cfg.optimizer = 'AdamW'
cfg.no_decay = True

# scheduler
cfg.pct_start = 0.1
cfg.max_lr = 3e-3
cfg.final_div_factor = 100

# augmentations
cfg.resample_train = 10
cfg.other_samples = 2
cfg.max_shift = 1
cfg.loudness_range = 10.0

# logging
cfg.local_rank = 0
cfg.verbose=True

# model
cfg.backbone = 'tf_efficientnetv2_b0.in1k'
cfg.gem_pooling = False
cfg.bce = True
cfg.drop_rate = 0.1

# tasks hp
cfg.train_model = True
cfg.num_folds = 5
cfg.pl = None
cfg.pretrained_path = None 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
import gc
import pickle as pkl

import librosa

from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as T

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.amp import autocast, GradScaler
from torch.optim import lr_scheduler, Adam, AdamW

import timm

from glob import glob
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.special import logit, expit


# from pprint import pprint
# model_names = timm.list_models(pretrained=True)
# pprint(model_names)


if cfg.train_model:
    checkpoint_path = cfg.save_path / cfg.fname
    if not cfg.save_path.exists():
        cfg.save_path.mkdir()
    if not checkpoint_path.exists():
        checkpoint_path.mkdir()
    checkpoint_path = checkpoint_path / "exp_0"
    exp = 0
    while(checkpoint_path.exists()):
        exp += 1
        checkpoint_path = cfg.save_path / f"exp_{exp}"
    checkpoint_path.mkdir()
    cfg.checkpoint_path = checkpoint_path
    print(f"Saving checkpoint path to {checkpoint_path}")


def get_logger(cfg):
    logger = getLogger(cfg.fname)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    if cfg.logger_file and cfg.train_model:
        filename= cfg.checkpoint_path / "run.log"
        handler2 = FileHandler(filename=filename)
        handler1.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler2)
    return logger

def seed_torch(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
    if torch.backends.cudnn.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


train = pd.read_csv(cfg.comp_data_path / 'train.csv')
train


train["species"] = [filename.split("/")[0] for filename in train["filename"]]
train["record"] = [filename.split("/")[1] for filename in train["filename"]]
train["secondary_labels"] = [eval(sls) for sls in train["secondary_labels"]]


df = train.groupby("record").agg({
    'species': ['first', 'last'],
    'secondary_labels': ['first', 'last'],
})

df.columns = ["first_species", "last_species", "first_secondary", "last_secondary"]
df.reset_index(inplace=True)
df


train = train.merge(df[['record', 'first_species', 'last_species',]],
                   how='left',
                   on='record')


def load_audio(file_name, isfirst, istrain, cfg):
    filepath = file_name.split("/")[0]
    fname = file_name.split("/")[1].split(".")[0]
    filepath = cfg.input_path / "birdclef-data" / "birdclef_data" / "birdclef_data" / filepath

    if istrain:
        max_duration = int((cfg.duration + cfg.max_shift) * cfg.sr)
    else:
        max_duration = cfg.duration * cfg.sr

    if isfirst:
        filepath = filepath / f"first10_{fname}.npy"
        audio = np.load(filepath)
        audio = audio[:max_duration]
    else:
        filepath = filepath / f"last10_{fname}.npy"
        audio = np.load(filepath)
        audio = audio[-max_duration:]

    return audio


new_train = []

kf = KFold(n_splits=cfg.num_folds, shuffle=True, random_state=0)
for species, df in train.groupby('species'):
    df['fold'] = -1
    if len(df) < 5:
        num_requires = 5 - len(df)
        samples = df.sample(n=num_requires, replace=True)
        df = pd.concat([df, samples])
    df = df.reset_index(drop=True)
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(df, df.primary_label)):
        df.loc[valid_idx, "fold"] = fold
    new_train.append(df)
new_train = pd.concat(new_train).reset_index(drop=True)    
new_train.fold.value_counts()


new_train.groupby(['species', 'fold']).size().unstack()
train = new_train


def metric(preds, targets, cfg):
    score = {}
    for j, label in enumerate(cfg.labels):
        y_true = targets[:, j]
        y_pred = preds[:, j]
        if len(np.unique(y_true)) < 2:
            score[label] = np.nan
            continue
        score[label] = roc_auc_score(y_true, y_pred)
    score_avg = np.nanmean([v for v in score.values()])
    return score_avg, score

def metric_db(train, oofs, cfg):
    score = {}
    for j,label in enumerate(cfg.labels):
        score[label] = roc_auc_score(train.primary_label == label, oofs[label])
    score_avg = np.mean([v for k,v in score.items()])
    return score_avg, score
    
def my_softmax(preds):
    preds = preds - preds.max(1, keepdims=True)
    preds = np.exp(preds.clip(-20, 0))
    preds = preds / preds.sum(1, keepdims=True)
    return preds

def bce_with_mask(preds, targets, mask):
    loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
    loss = loss * mask
    loss = loss.mean()
    return loss

class FocalLossBCE(torch.nn.Module):
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = "mean",
            bce_weight: float = 0.7,
            focal_weight: float = 1.3,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets, mask):
        focall_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = bce_with_mask(logits, targets, mask)
        return (self.bce_weight * bce_loss + self.focal_weight * focall_loss) / 2


cfg.device = torch.device("cuda")
if cfg.bce:
    cfg.loss = bce_with_mask
else:
    cfg.loss = bce_with_mask
    
cfg.labels = np.array(sorted(train.primary_label.unique()))
cfg.num_labels = len(cfg.labels)
cfg.targets = {v: i for i, v in enumerate(cfg.labels)}

cfg.logger = get_logger(cfg)
seed_torch(cfg.seed)


class BirdDataset(Dataset):
    def __init__(self, train, istrain, cfg):
        self.cfg = cfg
        self.istrain = istrain
        self.filename = train.filename.values
        self.secondary_labels = train.secondary_labels.values
        self.first_species = train.first_species.values
        self.last_species = train.last_species.values
        
    def __len__(self):
        return len(self.filename)
    
    def get_audio(self, idx):
        filename = self.filename[idx]
        duration = self.cfg.duration * self.cfg.sr
        if self.istrain:
            first = np.random.rand() < 0.5
            audio = load_audio(filename, first, self.istrain, self.cfg)
            
            if len(audio) < duration:
                pad_length = np.random.randint(0, duration - len(audio))
                audio = np.pad(audio,
                               ((pad_length, duration - len(audio) - pad_length),),
                               mode='constant')
            else:
                start = np.random.randint(0, len(audio) - duration + 1)
                audio = audio [start: start + duration]
        else:
            audio = load_audio(filename, True, self.istrain, self.cfg)
            audio = audio[:duration]
            if len(audio) < duration:
                pad_length = (duration - len(audio)) // 2
                audio = np.pad(audio,
                               ((pad_length, duration - len(audio) - pad_length),),
                               mode='constant')

        return audio
    
    def __getitem__(self, idx):
        audio = self.get_audio(idx)
        
        targets = np.zeros(self.cfg.num_labels, dtype=np.float32)
        targets[self.cfg.targets[self.first_species[idx]]] = 1.0
        targets[self.cfg.targets[self.last_species[idx]]] = 1.0
        
        secondary_mask = np.ones(self.cfg.num_labels, dtype=np.float32)
        secondary_labels = self.secondary_labels[idx]
        
        if len(secondary_labels) > 0:
            for label in secondary_labels:
                if label in self.cfg.targets:
                    secondary_mask[self.cfg.targets[label]] = 0.0
                    
        if self.istrain and self.cfg.other_samples:
            num_samples = np.random.randint(0, self.cfg.other_samples + 1)
            for _ in range(num_samples):
                other_idx = np.random.randint(0, len(self.filename))
                other_audio = self.get_audio(other_idx)
                weight = 0.2 + 0.8 * np.random.rand()
                audio += weight * other_audio
                targets[self.cfg.targets[self.first_species[other_idx]]] = 1.0
                targets[self.cfg.targets[self.last_species[other_idx]]] = 1.0
                secondary_labels = self.secondary_labels[other_idx]
                
            if len(secondary_labels) > 0:
                for label in secondary_labels:
                    if label in self.cfg.targets:
                        secondary_mask[self.cfg.targets[label]] = 0.0
                    
        secondary_mask = np.maximum(secondary_mask, targets)
        out = {
            'audio': audio,
            'targets': targets,
            'secondary_mask': secondary_mask
        }
        return out
    
def batch_to_device(batch, device):
        return {k:batch[k].to(device, non_blocking=True) for k in batch.keys() if k not in []}


dataset = BirdDataset(train, True, cfg)
elt = dataset[0]
for k,v in elt.items():
    print(k, v.shape)


def get_pl_iter(pl_preds, fold, istrain, cfg):
    pl_dataset = BirdPLDataset(pl_preds, fold, cfg)
    pl_data_loader = DataLoader(
        pl_dataset,
        batch_size=cfg.pl_batch_size,
        num_workers=0,
        shuffle=istrain,
        pin_memory=False,
        drop_last = False,
    )
    
    return iter(pl_data_loader), pl_data_loader

def get_pl_batch(pl_dataloader):
    try:
        batch = next(pl_dataloader[0])
    except StopIteration:
        pl_data_loader = pl_dataloader[1]
        pl_data_loader = iter(pl_data_loader), pl_data_loader
        batch = next(pl_dataloader[0])
    new_batch = {k : v.view(v.shape[0] * v.shape[1], v.shape[2]) 
                 for k,v in batch.items()}
    return new_batch


def get_data_loader(dataset, istrain, cfg):
    if istrain:
        batch_size = cfg.train_batch_size
    else:
        batch_size = cfg.valid_batch_size
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.workers,
        shuffle=istrain,
        pin_memory=False,
        #collate_fn=collate_pad,
        drop_last = istrain,
    )
    return data_loader 


class BirdModel(nn.Module):
    def __init__(self, cfg, pretrained: bool = True):
        super(BirdModel, self).__init__()
        self.cfg = cfg
        self.mel = T.MelSpectrogram(
            sample_rate=cfg.sr, n_fft=cfg.n_fft, win_length=cfg.win_length, 
            hop_length= cfg.hop_length, f_min=cfg.fmin, f_max=cfg.fmax, 
            n_mels=cfg.n_mels, mel_scale='htk', power=2.0)
        self.A2DB = T.AmplitudeToDB(stype="power")
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=pretrained,
            drop_rate = cfg.drop_rate,
            num_classes=cfg.num_labels
        )
         
    def forward(self, input_dict):
        x = input_dict['audio']
        with autocast(enabled=False, device_type="cuda"), torch.no_grad():
            x = x / torch.std(x, 1, keepdim=True)
            x = x.float()
            x = self.mel(x)
            x = self.A2DB(x)
        with torch.no_grad():
            x = x.unsqueeze(1)
            pos = torch.linspace(0., 1., x.size(2)).to(x.device)
            pos = pos.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            pos = pos.expand(x.size(0), 1, x.size(2), x.size(3))
            x = x.expand(-1, 2, -1, -1)
            x = torch.cat([x, pos], 1)

        x = self.backbone(x)
        return x


data_loader = get_data_loader(dataset, False, cfg)
model = BirdModel(cfg)
for batch in data_loader:
    break
out = model(batch)
out.shape


def train_epoch(loader, model, optimizer, scheduler, scaler, device, cfg):
    model.train()
    model.zero_grad()
    if cfg.verbose:
        bar = tqdm(range(len(loader)))
    else:
        bar = range(len(loader))
    load_iter = iter(loader)
    loss_l = []
    grad_norm_l = []
    
    accumulate = cfg.accumulate
    
    for i, batch in zip(bar, load_iter):
        input_dict = batch_to_device(batch, device)
        with autocast(enabled=True, device_type="cuda"):
            targets = input_dict['targets']
            if cfg.loudness_range:
                loudness = - np.log(cfg.loudness_range)
                bs = targets.shape[0]
                weight = torch.rand(bs, 1).to(targets.device)
                weight = torch.exp(weight * loudness)
                audio = input_dict['audio']
                audio = audio * weight
                input_dict['audio'] = audio
            preds = model(input_dict)
            secondary_mask = input_dict['secondary_mask']
            loss = cfg.loss(preds, targets, secondary_mask).mean()
        loss_l.append(loss.detach().cpu().item())
        scaler.scale(loss / cfg.accumulate).backward() 
        accumulate -= 1
        if accumulate == 0:
            if cfg.grad_value:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_value_(model.parameters(), cfg.grad_value)
            if cfg.grad_norm:
                scaler.unscale_(optimizer)
                total_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm).item()
                if np.isnan(total_norm):
                    total_norm = cfg.grad_norm
                else:
                    total_norm = np.clip(total_norm, 0, cfg.grad_norm)
                grad_norm_l.append(total_norm)
            scaler.step(optimizer)     
            scaler.update()
            optimizer.zero_grad()
            accumulate = cfg.accumulate
            scheduler.step()  
        del preds, targets, loss, input_dict
        if cfg.verbose:
            if cfg.grad_norm:
                bar.set_description('loss: %.4f grad norm %.1f' % (np.mean(loss_l), np.mean(grad_norm_l),))
            else:
                bar.set_description('loss: %.4f ' % np.mean(loss_l))
    optimizer.zero_grad()
    del loss_l, bar
    gc.collect()


def valid_epoch(loader, model, device, cfg):
    model.eval()
    model.zero_grad()
    if cfg.verbose:
        bar = tqdm(range(len(loader)))
    else:
        bar = range(len(loader))
    load_iter = iter(loader)
    preds_l = []
    targets_l = []
    with torch.no_grad():
        for i, batch in zip(bar, load_iter):      
            input_dict = batch_to_device(batch, device)
            with autocast(enabled=False, device_type="cuda"):
                preds = model(input_dict)                
            preds_l.append(preds.detach().cpu())
            del preds, input_dict
            targets = batch['targets']
            targets_l.append(targets)
        preds = torch.cat(preds_l)
        targets = torch.cat(targets_l)
        return preds.numpy(), targets.numpy()


def get_optimizer(model, cfg):
    no_decay = ["bias", "norm"]
    if cfg.no_decay:
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() 
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': cfg.decay},
            {'params': [p for n, p in model.named_parameters() 
                        if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ] 
    else:
        optimizer_parameters = model.parameters()
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=cfg.lr)
    return optimizer

def get_scheduler(optimizer, train_data_loader, cfg):
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.max_lr,
        epochs=cfg.num_epochs,
        steps_per_epoch=len(train_data_loader),
        pct_start=cfg.pct_start,
        anneal_strategy="cos",
        final_div_factor=cfg.final_div_factor,
    )
    return scheduler


def save_checkpoint(model, fold, cfg):
    checkpoint = {
        'model' : model.state_dict(),
        'fold' : fold,
        'seed' : seed,
        }
    checkpoint_path = cfg.checkpoint_path
    save_path = checkpoint_path / ('%s_%d.pth' % (cfg.fname, fold))
    if cfg.local_rank == 0:
        cfg.logger.info('saving %s ...' % save_path)
    torch.save(checkpoint, save_path)
    if cfg.local_rank == 0:
        cfg.logger.info('done')

def load_checkpoint(fold, cfg):
    if cfg.pretrained_path:
        checkpoint_path = cfg.pretrained_path
    else:
        checkpoint_path = cfg.checkpoint_path
    save_path = checkpoint_path / ('%s_%d.pth' % (cfg.fname, fold))
    cfg.logger.info('loading %s ...' % save_path)
    checkpoint = torch.load(save_path, map_location='cpu')
    model = BirdModel(cfg, pretrained=False).to(cfg.device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    cfg.logger.info('done')
    return model


def resample(train_fold, cfg):
    new_train = []
    for species, df in train.groupby('species'):
        new_train.append(df)
        if len(df) < cfg.resample_train:
            df = df.sample(n=(cfg.resample_train - len(df)), replace=True, random_state=cfg.seed)
            new_train.append(df)
    new_train = pd.concat(new_train).reset_index(drop=True)  
    return new_train  


scores = []
oofs = pd.DataFrame(columns=cfg.labels, index=train.index, data = 0.0)
oofs['filename'] = train.filename


for fold in range(cfg.num_folds):
    if cfg.fold >= 0 and cfg.fold != fold:
        continue
    seed = cfg.seed + fold
    seed_torch(seed)
    if cfg.pl is not None:
        pl_train = get_pl_train(cfg)
    else:
        pl_train = None
        train_fold = train[train.fold != fold]
    valid_dataset = BirdDataset(train[train.fold == fold], False, cfg)
    valid_dataloader = get_data_loader(valid_dataset, istrain=False, cfg=cfg)
    device = cfg.device
    
    if cfg.pretrained_path:
        model = load_checkpoint(fold, seed, cfg)
    else:
        model = BirdModel(cfg, pretrained=True).to(device)
    optimizer = get_optimizer(model, cfg)
    scheduler = None
    scaler = GradScaler()
    result = None
    for epoch in range(cfg.num_epochs):
        if cfg.pl is not None:
            train_fold = sample_pl(train, fold, pl_train)
        if cfg.resample_train:
            train_fold = resample(train_fold, cfg)
        train_dataset = BirdDataset(train_fold, True, cfg)
        train_dataloader = get_data_loader(train_dataset, istrain=True, cfg=cfg)

        if scheduler is None:
            scheduler = get_scheduler(optimizer, train_dataloader, cfg)
        train_epoch(train_dataloader, model, optimizer, scheduler, scaler, device, cfg)
        if valid_dataset is not None:
            preds, targets = valid_epoch(valid_dataloader, model, device, cfg)
            if cfg.bce:
                preds = expit(preds)
            else:
                preds = my_softmax(preds)
            result, _ = metric(preds, targets, cfg)
            msg = f"seed {cfg.seed} fold {fold} epoch {epoch} metric {result:.4f}"
            cfg.logger.info(msg)
        else:
            msg = f"seed {cfg.seed} fold {fold} epoch {epoch}"
            cfg.logger.info(msg)
    if cfg.local_rank == 0:
        save_checkpoint(model, fold, cfg)
    del model, optimizer, scheduler, scaler, train_dataloader, 
    if valid_dataset is not None:
        del valid_dataloader
    gc.collect()
    torch.cuda.empty_cache()
    scores.append(result)
    
    for j,c in enumerate(cfg.labels):
        oofs.loc[train.fold == fold, c] = preds[:, j]
oofs.to_csv(cfg.checkpoint_path / 'oofs.csv', index=False)

np.mean(scores), metric_db(train, oofs, cfg)


res = metric_db(train, oofs, cfg)
res


dfe = train.groupby('primary_label').size()
dfe.name = 'size'
dfe = dfe.reset_index().sort_values('primary_label').reset_index(drop=True)
dfe


df = pd.DataFrame({'label' : cfg.labels, 
                   'roc_auc' : [res[1][label] for label in cfg.labels],
                   'size' : dfe['size'].values,
                  })
df


plt.scatter(np.log(df['size']), df['roc_auc'], marker='+')


# class SoundScapeDataset():
#     def __init__(self, soundscape, istrain, cfg):
#         self.cfg = cfg
#         self.istrain = istrain
#         self.soundscape = soundscape
#         self.duration = int(np.round(soundscape.shape[0] / cfg.sr))
#         self.starts = np.arange(0, self.duration, cfg.duration)
#         self.ends = self.starts + cfg.duration
#         self.starts = cfg.sr * self.starts
#         self.ends = cfg.sr * self.ends
        
#     def __len__(self):
#         return len(self.starts)
    
#     def __getitem__(self, idx):
#         audio = self.soundscape[self.starts[idx] : self.ends[idx]]
#         duration = self.cfg.sr * self.cfg.duration
#         if len(audio) < duration:
#             pad_length = (duration - len(audio)) // 2
#             audio = np.pad(audio, 
#                            ((pad_length, duration - len(audio) - pad_length),), 
#                            mode='constant')
#         out = {
#             'audio' : torch.from_numpy(audio),
#         }
#         return out

# def predict(loader, models, cfg):
#     for model in models:
#         model.eval()
#         model.zero_grad()
#     if cfg.verbose:
#         bar = tqdm(range(len(loader)))
#     else:
#         bar = range(len(loader))
#     load_iter = iter(loader)
#     preds_l = [[] for model in models]
#     targets_l = []
#     with torch.no_grad():
#         for i, batch in zip(bar, load_iter):      
#             input_dict = batch_to_device(batch, cfg.device)
#             with autocast(enabled=False, device_type="cuda"):
#                 for preds, model in zip(preds_l, models):
#                     preds.append(model(input_dict).detach().cpu()) 
#             del input_dict
#         preds_l = [torch.cat(preds).squeeze().numpy() for preds in preds_l]
#         return preds_l

# def load_soundscape(pathname, cfg):
#     #audio = librosa.load(pathname, sr=32000)[0].astype(np.float32)
#     filename = pathname.split('/')[-1].split('.')[0]
#     audio = np.load(cfg.input_path / "birdclef-data" / 'unlabeled_soundscapes' / 'unlabeled_soundscapes' / (filename + '.npy'))

#     return audio

# def predict_soundscape(filepath, models, cfg):
#     waveform = load_soundscape(filepath, cfg)
#     #return len(waveform) / cfg.sr
#     #return waveform
#     #print('duration',  len(waveform) / cfg.sr)
#     dataset = SoundScapeDataset(waveform, False, cfg)
#     dataloader = get_data_loader(dataset, False, cfg)
#     preds_l = predict(dataloader, models, cfg)
#     if cfg.bce:
#         preds_l = [expit(preds) for preds in preds_l] # model uses logits
#     else:
#         preds_l = [my_softmax(preds) for preds in preds_l] # model uses logits
#     return preds_l


# soundscapes = sorted(glob(str(cfg.soundscape_path / '*.ogg')))
# soundscapes[:5]


# cfg.verbose = False
# cfg.workers = 0
# cfg.valid_batch_size = 128

# models = [load_checkpoint(fold, cfg).to(cfg.device)
#           for fold in range(cfg.num_folds)
#          ]
# preds = {}
# for soundscape in tqdm(soundscapes):
#     preds[soundscape]  = predict_soundscape(soundscape, models, cfg)
# with open(cfg.checkpoint_path / f"pl_all.pkl", "wb") as file:
#     pkl.dump(preds, file)

