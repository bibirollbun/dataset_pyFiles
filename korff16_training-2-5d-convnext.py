import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import os, sys
import timm
import torch.nn.functional as F
from glob import glob
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
from pathlib import Path
from collections import OrderedDict
from transformers.models.distilbert.modeling_distilbert import Transformer as T
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
torch.multiprocessing.set_sharing_strategy('file_descriptor')


rd = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'
OUTPUT_DIR = '/kaggle/working/rsna-results-2.5d'

if not Path(OUTPUT_DIR).exists():
    os.mkdir(OUTPUT_DIR)
    
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = [256, 256]
N_FOLDS = 5
EPOCHS = 100
USE_AMP = True
N_LABELS = 25
N_CLASSES = 3 * N_LABELS
AUG_PROB = 0.75
SELECTED_FOLDS = [0, 1, 2, 3, 4]
SEED = 69
GRAD_ACC = 1
TGT_BATCH_SIZE = 8
IN_CHANS = 18
BATCH_SIZE = TGT_BATCH_SIZE // GRAD_ACC // 2
MAX_GRAD_NORM = None
EARLY_STOPPING_EPOCH = 20
LR = 2e-4 * TGT_BATCH_SIZE / 32
WD = 1e-2
AUG = True
MODEL_NAME = 'convnext_pico.d1_in1k'
# MODEL_NAME = 'edgenext_base.in21k_ft_in1k'

# MODEL_NAME = 'convnextv2_pico.fcmae'
NOT_DEBUG = True
N_WORKERS = 4


os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_random_seed(seed: int = 2222, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = deterministic

set_random_seed(SEED)


df = pd.read_csv(f'{rd}/train.csv')
df.head()


df = df.fillna(-100)


label2id = {'Normal/Mild': 0, 'Moderate':1, 'Severe':2}
df = df.replace(label2id)
df.head()


CONDITIONS = [
    'Spinal Canal Stenosis', 
    'Left Neural Foraminal Narrowing', 
    'Right Neural Foraminal Narrowing',
    'Left Subarticular Stenosis',
    'Right Subarticular Stenosis'
]

LEVELS = [
    'L1/L2',
    'L2/L3',
    'L3/L4',
    'L4/L5',
    'L5/S1',
]
model_names = list(df.columns)[1:]
model_names


from pathlib import Path
class RSNA24Dataset(Dataset):
    def __init__(self, df, phase='train', transform=None):
        self.df = df
        self.transform = transform
        self.phase = phase
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = np.zeros((IMG_SIZE[0], IMG_SIZE[1], IN_CHANS, 3), dtype=np.float32)
        t = self.df.iloc[idx]
        st_id = int(t['study_id'])
        label = t[1:].values.astype(np.int64)
        
        # Sagittal T1

        sat1 = glob(f'/kaggle/input/cvt_png/{st_id}/Sagittal T1/*.png')
        sat1 = sorted(sat1)
    
        step = len(sat1) / (IN_CHANS - 1)
        st = 0
        end = len(sat1)+0.0001
        if len(sat1) != 0:
            for i, j in enumerate(np.arange(st, end, step)):
                try:
                    p = sat1[max(0, int((j-0.5001).round()))]
                    img = Image.open(p).convert('L')
                    img = np.array(img)
                    x[..., i, 0] = img.astype(np.float32)
                except:
#                     print(f'failed to load on {st_id}, Sagittal T1')
                    pass
            
        #Sagittal T2/STIR
        sat2 = glob(f'/kaggle/input/cvt_png/{st_id}/Sagittal T2_STIR/*.png')
        sat2 = sorted(sat2)
    
        step = len(sat2) / (IN_CHANS - 1)
        st = 0
        end = len(sat2)+0.0001

        if len(sat2) != 0:
            for i, j in enumerate(np.arange(st, end, step)):
                try:
                    p = sat2[max(0, int((j-0.5001).round()))]
                    img = Image.open(p).convert('L')
                    img = np.array(img)
                    x[..., i, 1] = img.astype(np.float32)
                except:
#                     print(f'failed to load on {st_id}, Sagittal T2/STIR')
                    pass
            
        # Axial T2
        axt2 = glob(f'/kaggle/input/cvt_png/{st_id}/Axial T2/*.png')
        axt2 = sorted(axt2)
    
        step = len(axt2) / (IN_CHANS - 1)
        st = 0
        end = len(axt2)+0.0001

        if len(axt2) != 0:
            for i, j in enumerate(np.arange(st, end, step)):
                try:
                    p = axt2[max(0, int((j-0.5001).round()))]
                    img = Image.open(p).convert('L')
                    img = np.array(img)
                    x[..., i, 2] = img.astype(np.float32)
                except:
#                     print(f'failed to load on {st_id}, Axial T2')
                    pass  
            
#         assert np.sum(x)>0
        if self.transform is not None:
            for i in range(x.shape[-1]):
                x[..., i] = self.transform(image=x[..., i])['image']

        x = x.transpose(2, 3, 0, 1)
        
                
        return x, label


transforms_train = A.Compose([
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=50),
    ], p=AUG_PROB),

    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ], p=AUG_PROB),

    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=AUG_PROB),
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
#     A.CoarseDropout(max_holes=16, max_height=16, max_width=16, min_holes=1, min_height=2, min_width=2, p=AUG_PROB),    
    A.Normalize(mean=0.5, std=0.5)
])

transforms_val = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=0.5, std=0.5)
])

if not AUG:
    transforms_train = transforms_val


tmp_ds = RSNA24Dataset(df, phase='train', transform=transforms_train)
tmp_dl = DataLoader(
            tmp_ds,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
            num_workers=0
            )

for i, (x, t) in enumerate(tmp_dl):
    if i==2:break
    print('x stat:', x.shape, x.min(), x.max(),x.mean(), x.std())
    print(t, t.shape)
    y = x.numpy()[0,1,0,:,:]
    plt.imshow(y)
    plt.show()
    print('y stat:', y.shape, y.min(), y.max(),y.mean(), y.std())
    print()
plt.close()
del tmp_ds, tmp_dl


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
#         nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim 
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)
        




class TimmModelCombo(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(TimmModelCombo, self).__init__()

        self.encoder_sagittal = timm.create_model(
            backbone,
            in_chans=2,
            num_classes=1,
            features_only=False,
            drop_rate=0.4,
            pretrained=pretrained
        )
        
        self.encoder_axial = timm.create_model(
            backbone,
            in_chans=1,
            num_classes=1,
            features_only=False,
            drop_rate=0.4,
            pretrained=pretrained
        )

        if 'efficient' in backbone:
            hdim = self.encoder_sagittal.conv_head.out_channels
            self.encoder_sagittal.classifier = nn.Identity()
            self.encoder_axial.classifier = nn.Identity()
            
        elif 'convnext' in backbone:
            hdim = self.encoder_sagittal.head.fc.in_features
            self.encoder_sagittal.head.fc = nn.Identity()
            self.encoder_axial.head.fc = nn.Identity()
            
        if 'densenet121' in backbone:
            hdim = 1024
            self.encoder_sagittal.classifier = nn.Identity()
            self.encoder_axial.classifier = nn.Identity()
            
        if 'densenet161' in backbone:
            hdim = 2208
            self.encoder_sagittal.classifier = nn.Identity()
            self.encoder_axial.classifier = nn.Identity()
            
        if 'densenet201' in backbone:
            hdim = 1920
            self.encoder_sagittal.classifier = nn.Identity()
            self.encoder_axial.classifier = nn.Identity()


#         self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=0., bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 75),
        )
        self.attention_layer_sagittal = Attention(512, IN_CHANS)
        self.attention_layer_axial = Attention(512, IN_CHANS)
        
        self.fc_axial = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            nn.SiLU()
        )
        self.fc_sagittal = nn.Sequential(
            nn.Linear(512, 256), 
            nn.Dropout(0.3),
            nn.SiLU()
        )

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        bs = x.shape[0]
        img_size = x.shape[3]
        
        x_sagittal = x[:, :, :2, :, :]
        x_axial = x[:, :, 2:3, :, :]
        
        x_sagittal = x_sagittal.view(bs * IN_CHANS, x_sagittal.shape[2], img_size, img_size)
        feat_sagittal = self.encoder_sagittal(x_sagittal)
        feat_sagittal = feat_sagittal.view(bs, IN_CHANS, -1)
        
        x_axial = x_axial.view(bs * IN_CHANS, x_axial.shape[2], img_size, img_size)
        feat_axial = self.encoder_axial(x_axial)
        feat_axial = feat_axial.view(bs, IN_CHANS, -1)
#         feat_lstm, _ = self.lstm(feat)
#         feat_lstm = feat_lstm.contiguous().view(bs * 12, -1)
#         feat_lstm = self.head(feat_lstm)
#         feat_lstm = feat_lstm.view(bs, 12, 75).contiguous()
        atten_sagittal = self.attention_layer_sagittal(feat_sagittal)
        atten_axial = self.attention_layer_axial(feat_axial)
#         atten = torch.cat((atten_sagittal, atten_axial), dim=1)
        atten_sagittal = self.fc_sagittal(atten_sagittal)
        atten_axial = self.fc_axial(atten_axial)
        atten = (atten_sagittal + atten_axial) / 2
        out = self.head(atten)
        return out



class TimmModel(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(TimmModel, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            in_chans=2,
            num_classes=1,
            features_only=False,
            drop_rate=0.4,
            pretrained=pretrained
        )

        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()
            
        if 'densenet121' in backbone:
            hdim = 1024
            self.encoder.classifier = nn.Identity()
            
        if 'densenet161' in backbone:
            hdim = 2208
            self.encoder.classifier = nn.Identity()
        if 'densenet201' in backbone:
            hdim = 1920
            self.encoder.classifier = nn.Identity()
            
        if 'edgenext' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()


        self.lstm = nn.LSTM(hdim, 256, num_layers=1, dropout=0., bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hdim, 256),
            nn.Dropout(0.4),
            nn.SiLU(),
            nn.Linear(256, 75),
        )
        self.attention_layer = Attention(hdim, IN_CHANS)

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        x = x[:, :, 0:2, :, :]
        bs = x.shape[0]
        img_size = x.shape[3]
        x = x.view(bs * IN_CHANS, 2, img_size, img_size)

        feat = self.encoder(x)
        feat = feat.view(bs, IN_CHANS, -1)
        
        
#         feat_lstm, _ = self.lstm(feat)
        
#         feat_lstm = feat_lstm.contiguous().view(bs * 12, -1)
#         feat_lstm = self.head(feat_lstm)
#         feat_lstm = feat_lstm.view(bs, 12, 75).contiguous()
        atten = self.attention_layer(feat)
        
        out = self.head(atten)
        return out






m = TimmModel(MODEL_NAME)
m = m.to(DEVICE)
i = torch.randn(8, IN_CHANS, 3, 224, 224).to(DEVICE)
with torch.no_grad():
    out = m(i)
for o in out:
    print(o.shape, o.min(), o.max())


del m, i, out
torch.cuda.empty_cache()


# m = RSNA24Model('efficientnet_b0', in_c=1, n_classes=512, pretrained=False)
# m = m.to(DEVICE)
# i = torch.randn(2, IN_CHANS // 3, 256, 256).to(DEVICE)
# out = m(i)
# for o in out:
#     print(o.shape, o.min(), o.max())


# del m, i, out


%time
#autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.bfloat16) # if your gpu is newer Ampere, you can use this, lesser appearance of nan than half
autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half) # you can use with T4 gpu. or newer
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=2048)

val_losses = []
train_losses = []
df_tr, df_test = train_test_split(df, test_size=2/7, random_state=SEED)
skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
device = DEVICE

for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
    loss_scale = 1
    if NOT_DEBUG == False:
        if fold == 1: break;
    if fold not in SELECTED_FOLDS: 
        print(f"Jump fold {fold}")
        continue;
    else:
        print('#'*30)
        print(f'Start fold {fold}')
        print('#'*30)
        print(len(trn_idx), len(val_idx))
        df_train = df.iloc[trn_idx]
        df_valid = df.iloc[val_idx]

        train_ds = RSNA24Dataset(df_train, phase='train', transform=transforms_train)
        train_dl = DataLoader(
                        train_ds,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        pin_memory=False,
                        drop_last=True,
                        num_workers=N_WORKERS
                        )

        valid_ds = RSNA24Dataset(df_valid, phase='valid', transform=transforms_val)
        valid_dl = DataLoader(
                        valid_ds,
                        batch_size=BATCH_SIZE*2,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=False,
                        num_workers=N_WORKERS
                        )

    #         model = RSNA24Model(MODEL_NAME, IN_CHANS, N_CLASSES, pretrained=True)
        model = TimmModel(MODEL_NAME, pretrained=True)
            
        fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}.pt'
    #         if os.path.exists(fname):
    #             model = TimmModel(MODEL_NAME, pretrained=False)
    #             model.load_state_dict(torch.load(fname))
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=LR*2, weight_decay=WD)
    #         optimizer = torch.optim.SGD(model.parameters(), lr=LR*2, weight_decay=WD, nesterov=True, momentum=0.9)

        warmup_steps = EPOCHS/10 * len(train_dl) // GRAD_ACC
        num_total_steps = EPOCHS * len(train_dl) // GRAD_ACC
        num_cycles = 0.475
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=warmup_steps,
                                                        num_training_steps=num_total_steps,
                                                        num_cycles=num_cycles)
    #         scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                                     num_warmup_steps=warmup_steps,
    #                                                     num_training_steps=num_total_steps)

        weights = torch.tensor([1.0, 2.0, 4.0])
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        criterion_cpu = nn.CrossEntropyLoss(weight=weights)
        best_loss = 1.2
        es_step = 0

        for epoch in range(1, EPOCHS+1):
            print(f'start epoch {epoch}')
            model.train()
            total_loss = 0
            with tqdm(train_dl, leave=True) as pbar:
                optimizer.zero_grad()
                for idx, (x, t) in enumerate(pbar):  
                    op = ['nothing', 'nothing', 'nothing', 'nothing', 'nothing']
                    x = x.to(device)
                    t = t.to(device)
    #                     t = torch.tensor(np.array(one_h(list(t.detach().cpu().numpy())))).to(device)
                    rc = random.sample(op, 1)
                    if rc[0] == 'mixup':
                        x = x.detach().cpu().numpy()
                        t = t.detach().cpu().numpy()
                        reference_data = [{'image':x[i], 'proba': t[i]} 
                                            for i in range(len(x))]
                        tr = A.Compose([A.MixUp(reference_data=reference_data,
                                                  read_fn=read_fn, p=0.5)])
                        for i in range(len(x)):
                            transformed = tr(image=x[i], global_label=t[i])
                            x[i] = transformed['image']
                            t[i] = transformed['global_label']

                        x = torch.tensor(x).to(device)
                        t = torch.tensor(t).to(device)

                    with autocast:
                        loss = 0
                        y = model(x)
                        for col in range(N_LABELS):
                            pred = y[:,col*3:col*3+3]
                            gt = t[:,col]
                            loss = loss + loss_scale * criterion(pred, gt) / N_LABELS

                        if not math.isfinite(loss):
                            loss = torch.tensor(1.2 * loss_scale * GRAD_ACC, requires_grad=True)
                        total_loss += loss.item()
                        if GRAD_ACC > 1:
                            loss = loss / GRAD_ACC

                    pbar.set_postfix(
                            OrderedDict(
                                loss=f'{loss.item()*GRAD_ACC:.6f}',
                                lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                            )
                    )
    #                     scaler.scale(loss).backward()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM or 1e9)

                    if (idx + 1) % GRAD_ACC == 0:
    #                         scaler.step(optimizer)
    #                         scaler.update()
                        optimizer.step()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()                    

            train_loss = total_loss/len(train_dl)
            print(f'train_loss:{train_loss/loss_scale:.6f}')
            train_losses.append(train_loss)
            total_loss = 0

            model.eval()
            y_preds, labels = [], []
            with tqdm(valid_dl, leave=True) as pbar:
                with torch.no_grad():
                    for idx, (x, t) in enumerate(pbar):

                        x = x.to(device)
                        t = t.to(device)

                        with autocast:
                            loss = 0
                            loss_ema = 0
                            y = model(x)
                            for col in range(N_LABELS):
                                pred = y[:,col*3:col*3+3]
                                gt = t[:,col]

                                loss = loss + criterion(pred, gt) / N_LABELS
                                y_pred = pred.float()
                                y_preds.append(y_pred.cpu())
                                labels.append(gt.cpu())

                            if not math.isfinite(loss):
                                loss = torch.tensor(1.2 * loss_scale * GRAD_ACC, requires_grad=True)

                            total_loss += loss.item()   

            val_loss = total_loss/len(valid_dl)
            y_preds = torch.cat(y_preds, dim=0)
            print(y_preds.shape)
            labels = torch.cat(labels)

            val_weighted_loss = criterion_cpu(y_preds, labels)
            writer.add_scalar('val_wll', val_weighted_loss, epoch)
            writer.flush()
            print(f'val_loss:{val_loss:.6f}')
            val_losses.append(val_loss)
            if val_weighted_loss < best_loss:

                if device!='cuda:0':
                        model.to('cuda:0')                

                print(f'epoch:{epoch}, best weighted_logloss updated from {best_loss:.6f} to {val_weighted_loss:.6f}')
                best_loss = val_weighted_loss
                fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}.pt'
                torch.save(model.state_dict(), fname)
                print(f'{fname} is saved')
                es_step = 0

                if device!='cuda:0':
                    model.to(device)

            else:
                es_step += 1
                if es_step >= EARLY_STOPPING_EPOCH:
                    print('early stopping')
                    break  
                                


cv = 0
y_preds = []
labels = []
weights = torch.tensor([1.0, 2.0, 4.0])
criterion2 = nn.CrossEntropyLoss(weight=weights)
autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half) # you can use with T4 gpu. or newer


## TODO: Modify EXIST_FOLDS by how many fold you've trained
EXIST_FOLDS = [0, 1, 2, 3, 4]
skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
    #     if NOT_DEBUG == False:
    #         if fold == 1: break;
    if fold not in EXIST_FOLDS: 
        print(f"Jump fold {fold}")
        continue;
    else:
        print('#'*30)
        print(f'Start fold {fold}')
        print('#'*30)
        df_valid = df.iloc[val_idx]
        valid_ds = RSNA24Dataset(df_valid, phase='valid', transform=transforms_val)
        valid_dl = DataLoader(
                        valid_ds,
                        batch_size=16,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=False,
                        num_workers=N_WORKERS
                        )
            

        model = TimmModelCombo(MODEL_NAME)
                
            # print("No internet read")
        fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}.pt'
        model.load_state_dict(torch.load(fname))
        model.to(device)   

        model.eval()
        with tqdm(valid_dl, leave=True) as pbar:
            with torch.no_grad():
                for idx, (x, t) in enumerate(pbar):

                    x = x.to(device)
                    t = t.to(device)

                    with autocast:
                        y = model(x)
                        for col in range(N_LABELS):
                            pred = y[:,col*3:col*3+3]
                            gt = t[:,col] 
                            y_pred = pred.float()
                            y_preds.append(y_pred.cpu())
                            labels.append(gt.cpu())

y_preds = torch.cat(y_preds)
labels = torch.cat(labels)


cv = criterion2(y_preds, labels)
print('cv score:', cv.item())










