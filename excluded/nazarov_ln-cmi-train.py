%%writefile main.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW

from transformers import get_cosine_schedule_with_warmup  # Cosine schedule with warmup
import argparse
import mlcrate as mlc
from cmi_utils import score, score_from_int
from types import SimpleNamespace
import random
import gc
from sklearn.metrics import log_loss
from scipy.special import softmax
from scipy.spatial.transform import Rotation as R
from sklearn.utils.class_weight import compute_class_weight
from copy import deepcopy
from joblib import dump, load
import json
from collections import defaultdict
from cmi_utils import *



KAGGLE = 'kaggle' in os.getcwd()

COMPETITION = 'cmi-detect-behavior-with-sensor-data'
if KAGGLE:
    data_path = f'/kaggle/input/{COMPETITION}/'
    my_data_path = '/kaggle/input/my-cmi-data/'
else:
    data_path = 'data/'
    my_data_path = 'my_data/'


cfg = SimpleNamespace()
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.debug=False # True # 
cfg.train= True # not KAGGLE # True #  # predict
cfg.apex=True
cfg.print_freq=1
available_cpus = os.cpu_count()
cfg.num_workers=1 if KAGGLE else max(1, min(available_cpus - 1, 4))
cfg.prefix = ''
cfg.target_gestures = ['Above ear - pull hair','Cheek - pinch skin','Eyebrow - pull hair','Eyelash - pull hair','Forehead - pull hairline','Forehead - scratch','Neck - pinch skin','Neck - scratch']
cfg.non_target_gestures = ['Write name on leg','Wave hello','Glasses on/off','Text on phone','Write name in air','Feel around in tray and pull out an object','Scratch knee/leg skin','Pull air toward your face','Drink from bottle/cup','Pinch knee/leg skin']
cfg.gestures = sorted(cfg.target_gestures) + sorted(cfg.non_target_gestures)
cfg.gesture2id = {v:k for k,v in enumerate(cfg.gestures)}
cfg.gradient_checkpointing=False
cfg.scheduler='cosine' # ['linear', 'cosine']
cfg.batch_scheduler=True
cfg.num_cycles=0.5
cfg.min_lr=1e-6
cfg.eps=1e-6
cfg.betas=(0.9, 0.999)
cfg.max_grad_norm=100
cfg.phase_loss = nn.BCEWithLogitsLoss() # nn.BCELoss() 
cfg.n_fold=5
cfg.trn_folds= list(range(cfg.n_fold))
cfg.folds=my_data_path + 'team_folds.csv'
cfg.head = ''
cfg.no_decay = True
cfg.use_groups = ['acc','rot','thm','tof']
cfg.acc_cols = ['acc_x', 'acc_y', 'acc_z']
cfg.rot_cols = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
cfg.imu_fe = False # True
slurm_job_name = os.getenv("SLURM_JOB_NAME", "Unknown")
print(f"SLURM Job Name: {slurm_job_name}")
cfg.job_id = slurm_job_name[-1]
cfg.output_dir = '' if KAGGLE else f'models/{cfg.job_id}/'
cfg.g_loss_weight = 0
cfg.max_len = 88
cfg.batch_size = 32
cfg.model = 'b0' # 0 initial, 1 public_72, 2 - public_75, , 3 - public_80
# from public_72 import TwoBranchModel
# from public_75 import ModelVariant_GRU
# from public_80 import TwoBranchModel80
# from public_82 import CMIModel
cfg.save_history = not KAGGLE
cfg.image_size = 224
cfg.tof_win = 16
cfg.sep = 0

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

from torchvision import transforms
from torchvision.models import (efficientnet_b0,efficientnet_b1,efficientnet_b3,
                                efficientnet_b5,efficientnet_b6,resnet18,
                                efficientnet_v2_s,efficientnet_v2_m,maxvit_t,
                                convnext_tiny,convnext_small,convnext_base,
                                resnet50,mobilenet_v2,squeezenet1_0, squeezenet1_1)

def save_namespace_json(obj, filename, omit=['device','phase_loss','target_gestures',
                                        'non_target_gestures','gestures','gesture2id']):
    omit = omit or []  # Handle None case
    # Get object attributes and filter out omitted ones
    obj_dict = {k: v for k, v in vars(obj).items() if k not in omit}
    with open(filename, 'w') as f:
        json.dump(obj_dict, f, indent=2)

def load_namespace_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        return SimpleNamespace(**data)


def should_decay(param_name):
    """Determine if a parameter should have weight decay applied."""
    no_decay_keywords = ['bias','bn','batch_norm','layer_norm',
                         'group_norm','weight_norm']
    return not any(nd in param_name.lower() for nd in no_decay_keywords)


def separate_decay_params(module, weight_decay_value):
    """Separate parameters into decay and no-decay groups."""
    decay_params = []
    no_decay_params = []

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter should have weight decay
        if should_decay(name):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({
            'params': decay_params,
            'weight_decay': weight_decay_value
        })
    if no_decay_params:
        param_groups.append({
            'params': no_decay_params,
            'weight_decay': 0.0
        })    
    return param_groups


class CustomMovingAverage(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a convolution layer with custom weights
        self.conv = nn.Conv2d(3, 3, kernel_size=(5, 1), padding=(2, 0), bias=False)
        kernel = torch.zeros(3, 3, 5, 1)
        # Channel 0: Identity (keep unchanged)
        kernel[0, 0, 2, 0] = 1.0
        # Channel 1: Moving average window=3
        kernel[1, 1, 1:4, 0] = 1.0/3.0
        # Channel 2: Moving average window=5
        kernel[2, 2, 0:5, 0] = 1.0/5.0
        # Set the weights (requires detaching from gradients if needed)
        self.conv.weight.data = kernel

    def forward(self, x):
        return self.conv(x)


class CustomEfficientNetB0(torch.nn.Module):
    def __init__(self, num_classes=18, ver = 'b0', n_clf=0):
        super().__init__()
        weights_path = ("/kaggle/input/cmi-models-new/pretrained/" if KAGGLE 
                        else 'models/pretrained/')
        self.n_clf = n_clf
        if ver == 'b0':
            self.base_model = efficientnet_b0(weights=None)
            self.base_model.load_state_dict(torch.load(
                weights_path + 'efficientnet_b0_rwightman-7f5810bc.pth'))
        elif ver == 'b1':
            self.base_model = efficientnet_b1(weights=None)
            self.base_model.load_state_dict(torch.load(
                weights_path + 'efficientnet_b1_rwightman-bac287d4.pth'))
        elif ver == 'b3':
            self.base_model = efficientnet_b3(weights=None)
            self.base_model.load_state_dict(torch.load(
                weights_path + 'efficientnet_b3_rwightman-cf984f9c.pth'))
        elif ver == 'b5':
            self.base_model = efficientnet_b5(weights=None)
            self.base_model.load_state_dict(torch.load(
                weights_path + 'efficientnet_b5_lukemelas-b6417697.pth'))
        elif ver == 'b6':
            self.base_model = efficientnet_b6(weights=None)
            self.base_model.load_state_dict(torch.load(
                weights_path + 'efficientnet_b6_lukemelas-24a108a5.pth'))
        elif ver == 'v2_s':
            self.base_model = efficientnet_v2_s(weights=None)
            self.base_model.load_state_dict(torch.load(
                weights_path + 'efficientnet_v2_s-dd5fe13b.pth'))
        elif ver == 'v2_m':
            self.base_model = efficientnet_v2_m(weights=None)
            self.base_model.load_state_dict(torch.load(
                weights_path + 'efficientnet_v2_m-dc08266a.pth'))
        elif ver == 'cn_tiny':
            self.base_model = convnext_tiny(weights=None)
            self.base_model.load_state_dict(torch.load(
                weights_path + 'convnext_tiny-983f1562.pth'))
        elif ver == 'cn_small':
            self.base_model = convnext_small(weights=None)
            self.base_model.load_state_dict(torch.load(
                weights_path + 'convnext_small-0c510722.pth'))
        elif ver == 'cn_base':
            self.base_model = convnext_base(weights=None)
            self.base_model.load_state_dict(torch.load(
                weights_path + 'convnext_base-6075fbad.pth'))
        elif ver == 'maxvit_t':
            self.base_model = maxvit_t(weights=None)
            self.base_model.load_state_dict(torch.load(
                weights_path + 'maxvit_t-bc5ab103.pth'))
        elif ver == 'mobilenet_v2':
            self.base_model = mobilenet_v2(weights=None)
            self.base_model.load_state_dict(torch.load(
                weights_path + 'mobilenet_v2-b0353104.pth'))
        elif ver == 'squeezenet1_0':
            self.base_model = squeezenet1_0(weights=None)
            self.base_model.load_state_dict(torch.load(
                weights_path + 'squeezenet1_0-b66bff10.pth'))

        if ver == 'squeezenet1_0':
            self.base_model.classifier[1] = torch.nn.Conv2d(
                        in_channels=512,out_channels=num_classes,kernel_size=1)
        elif 'cn_' in ver:
            self.base_model.classifier[2] = torch.nn.Linear(self.base_model.classifier[2].in_features, num_classes)
        elif ver == 'maxvit_t':
            self.base_model.classifier[5] = torch.nn.Linear(self.base_model.classifier[5].in_features, num_classes)
        elif self.n_clf > 0:
            in_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = torch.nn.Identity()
            self.multi_classifier = MultiClassifier(in_features, num_classes,
                                                    self.n_clf, dropout_rate=0.2)
        else:
            self.base_model.classifier[1] = torch.nn.Linear(self.base_model.classifier[1].in_features, num_classes)

    def get_param_groups(self,low_lr,high_lr,weight_decay=1e-4):
        feature_layers = list(self.base_model.features.children())
        num_feature_groups = len(feature_layers)
        lrs = np.linspace(low_lr, high_lr, num_feature_groups + 1)
        param_groups = []
        for lr, layer in zip(lrs,feature_layers):
            layer_groups = separate_decay_params(layer, weight_decay)
            for group in layer_groups:
                group['lr'] = float(lr)
                param_groups.append(group)

        # Classifier layer
        classifier_groups = separate_decay_params(
            self.base_model.classifier[1],weight_decay
        )
        for group in classifier_groups:
            group['lr'] = high_lr
            param_groups.append(group)

        return param_groups


    def forward(self, x):
        if isinstance(x, dict):
            x = x['imu']
        out = self.base_model(x)
        if self.n_clf > 0:
            out = self.multi_classifier(out)
            if not self.training:
                out = out.mean(dim=0)
        return out

class CustomResNet(torch.nn.Module):
    def __init__(self, num_classes=18, ver = '18'):
        super().__init__()
        weights_path = ("/kaggle/input/cmi-models-new/pretrained/" if KAGGLE 
                        else 'models/pretrained/')
        if ver == '18':
            self.base_model = resnet18(weights=None)
            self.base_model.load_state_dict(torch.load(
                weights_path + 'resnet18-f37072fd.pth'))
        elif ver == '50':
            self.base_model = resnet50(weights=None)
            self.base_model.load_state_dict(torch.load(
                weights_path + 'resnet50-0676ba61.pth'))
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        if isinstance(x, dict):
            x = x['imu']
        return self.base_model(x)


# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, data, cfg, mix_alpha=0, demo=None, # data is dataframe
                 # quat_rel_stat = []  # list[mean (1,-1),std ]
                ): 
        self.max_len = cfg.max_len
        self.imu_fe = cfg.imu_fe
        self.cfg = cfg
        self.scale_quant = cfg.scale_quant if hasattr(cfg,'scale_quant') else False
        self.tof_drop = cfg.tof_drop if hasattr(cfg,'tof_drop') else 0

        if self.scale_quant > 0:
            rel_quat_stats = np.load(my_data_path + f'rel_quat_stats_{self.scale_quant}.npy')
            self.rq_median = rel_quat_stats[1:2]
            self.rq_IQR = rel_quat_stats[2:] - rel_quat_stats[:1]
            self.clip_quat = cfg.clip_quat
        self.clip_quat = cfg.clip_quat if hasattr(cfg,'clip_quat') else 0
        self.imu_image_weight = cfg.imu_image_weight if hasattr(cfg,'imu_image_weight') else 0
        self.euler = cfg.euler if hasattr(cfg,'euler') else False
        self.gaf = cfg.gaf if hasattr(cfg,'gaf') else ''
        self.train_mode = 'gesture' in data.columns
        if self.train_mode:
            if args.data_aug > 0:
                data = pd.concat([data,DataAugmentation(data, args.data_aug,
                    y_grid=cfg.y_aug_grid,z_grid=cfg.z_aug_grid)], ignore_index=True)
                # data.to_feather(f'data_aug_{args.data_aug}.feather')
        self.start,self.end = self.get_bounds(data.sequence_id.values)
        self.col_groups = {prefix: sorted(data.filter(like=prefix))
                           for prefix in cfg.use_groups}
        self.X = {}
        self.imu_image = cfg.model.lower() in ['b0','b1','b3','b5','b6','v2_s','v2_m','maxvit_t',
            'cn_tiny','cn_small','cn_base','mobilenet_v2','resnet18','resnet50','squeezenet1_0']
        if self.cfg.flip_left | self.cfg.flip_rot:
            self.handedness = (demo[['subject','handedness']].set_index('subject')
                .reindex(data['subject'].values).handedness.to_numpy('int8'))
        d=-110
        self.quat_rot_z = np.array([np.cos(np.deg2rad(d)/2),0,0,
                                    np.sin(np.deg2rad(d)/2)])
        for key,cols in self.col_groups.items():
            if len(cols) > 0:
                self.X[key] = self.preprocess(data[cols], key, data)
        self.preprocess_b0 = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array or tensor to PIL image
            transforms.Resize((cfg.image_size, cfg.image_size)),  # Resize to match EfficientNet input size
            transforms.Grayscale(num_output_channels=3),  # Replicate grayscale to 3 channels
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for RGB
        ])
        # concat individual images
        if self.imu_image_weight > 0:
            sep_size = cfg.sep * 3
            sizes = {'acc':int(0.5 * (cfg.image_size - sep_size) 
                                      *  self.imu_image_weight),
                     'thm':1                    
                    }
            sizes['tof'] = cfg.image_size - sep_size - 2*sizes['acc'] - 1
            sizes['rot'] = sizes['acc']
            self.ind_prep = {
                k: transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Resize((cfg.image_size,v)),
                    transforms.Grayscale(num_output_channels=3),transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                for k,v in sizes.items()
                }
        if self.train_mode:
            gesture2id = {g:i for i,g in enumerate(cfg.gestures)}
            self.target = data.groupby('sequence_id',sort=False).gesture.last()\
                            .map(gesture2id).to_numpy('int8')
            self.oh_target = np.eye(len(cfg.gestures))[self.target].astype(np.float32) 
            self.gesture_phase = (data.phase == 'Gesture').to_numpy('int8')
            self.mix_alpha = mix_alpha
            self.rng = np.random.default_rng(1)
            if args.rand_aug:
                self.preprocess_b0 = transforms.Compose([
                    transforms.ToPILImage(),  # Convert numpy array or tensor to PIL image
                    transforms.Resize((self.cfg.image_size, self.cfg.image_size)),  # Resize to match EfficientNet input size

                    # Apply random augmentations
                    # transforms.RandomHorizontalFlip(p=args.hflip),  # Randomly flip the image horizontally
                    # transforms.RandomVerticalFlip(p=0.5),    # Randomly flip the image vertically
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust color properties    

                    transforms.Grayscale(num_output_channels=3),  # Replicate grayscale to 3 channels
                    transforms.ToTensor(),  # Convert PIL image to tensor
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for RGB
                ])


    def __len__(self):
        return len(self.start)

    def __getitem__(self, idx):
        start,end = self.start[idx],self.end[idx]
        start = max(start,end - self.max_len)
        x = {key: torch.tensor(self.X[key][start:end], dtype=torch.float32)
             for key in self.X}
        delta = self.max_len - (end - start)
        if delta > 0:
            x = {k: F.pad(v,(0,0,delta,0)) for k,v in x.items()}

        if self.train_mode:
            x['label'] = torch.tensor(self.target[idx], dtype=torch.long)
            x['oh_label'] = torch.tensor(self.oh_target[idx], dtype=torch.float32)
            if self.cfg.g_loss_weight > 0:
                x['gesture_phase'] = torch.tensor(self.gesture_phase[start:end], dtype=torch.float32)
                if delta > 0:
                    x['gesture_phase'] = F.pad(x['gesture_phase'],(delta,0))
        else:
            self.tof_drop = 0

        if self.imu_image:
            if self.imu_image_weight > 0:
                x['imu'] = self.ind_transform(x) 
            sep = torch.zeros_like(x['rot'][...,:self.cfg.sep]) if self.cfg.sep > 0 else []
            features = [x['acc'],sep,0.5 * (x['rot'] + 1)]
            if 'tof' in self.X:
                features.append(sep)
                features.append(self.tof_smoothing(x['tof']))
                features.append(sep)
                features.append(torch.median(x['thm'],dim=-1,keepdims=True)[0])

                del x['tof']  
                del x['thm']  
            if len(self.gaf) > 0:
                # features in [0,1] gaf output in [-1,1]
                gaf = []
                for i in np.arange(0,len(features),2):
                    gaf.append(np.mean(0.5 * (1 + compute_multivariate_gaf(
                        (2 * features[i] - 1)[np.newaxis,...], 
                        method=self.gaf))[0],axis=-1))
                gaf = np.column_stack(gaf).astype(np.float32)
                features.extend([sep,torch.tensor(gaf)]) 
            features = torch.cat([f for f in features if len(f) > 0],dim=-1)
            imu = self.preprocess_b0(features)
            del x['acc'],x['rot']
            x['imu'] = imu
        return x

# src_order = ["acc_x","acc_y","acc_z","acc_mag","acc_mag_jerk",
#                 "rot_w","rot_x","rot_y","rot_z","rot_angle","rot_angle_vel",
# ]
# dst_order = ['acc_x',"rot_x","acc_y","rot_y","acc_z","rot_z",
                # "rot_w","rot_angle","rot_angle_vel","acc_mag","acc_mag_jerk",]

    def preprocess(self, df, key, data):
        if key == 'tof':
            x = ((df + 1.).clip(0,250).fillna(0) / 256.).to_numpy(np.float16)
            if self.cfg.flip_tof & (self.handedness==0).any():
                x[self.handedness==0] = flip_tof(x[self.handedness==0])
        elif key == 'thm':
            low,high = 22,35
            x = (df.fillna(0).clip(low,high) - low).to_numpy(np.float16) / (high - low)
        elif key == 'acc':
            low,high = np.array([-24.41,-12.70,-20.52]),np.array([20.70,17.25,17.45])
            x = df.fillna(0).to_numpy(np.float32)
            if self.cfg.flip_left:
                x[self.handedness==0,0] = - x[self.handedness==0,0]
            if self.imu_fe:
                acc_mag = np.sqrt((x**2).sum(axis=1))
                acc_mag_jerk = np.append(0,np.diff(acc_mag))
                acc_mag_jerk[self.start] = 0
                x = np.hstack([x,acc_mag.reshape(-1,1),acc_mag_jerk.reshape(-1,1)])
                low,high = np.append(low,[4.2,-13]),np.append(high,[21.3,13])
            x = (x.clip(low,high) - low) / (high - low)
        elif key == 'rot': # df:w,x,y,z
            if self.cfg.fill_rot_mean:
                x = df.fillna({'rot_x':-0.119916,'rot_y':-0.059953,'rot_z':-0.188298,
                               'rot_w':0.360375}).clip(-1,1).to_numpy(np.float32)
            else:
                x = df.fillna(0).clip(-1,1).to_numpy(np.float32) # ['rot_w','rot_x','rot_y','rot_z']
            if self.cfg.flip_rot:
                proper_left = (~df.isnull().any(axis=1)) & (self.handedness == 0)
                if proper_left.any():
                    quat_flip = x[proper_left].copy() 
                    quat_flip = quaternion_multiply(self.quat_rot_z, quat_flip)[:,[1,2,3,0]] 
                    euler_x,euler_y,euler_z = quaternion_to_euler(quat_flip)  
                    quat_flip = euler_to_quaternion(euler_x, -euler_y, -euler_z)
                    quat_flip = quat_flip*((quat_flip[:,0:1]>0)*2-1) 
                    x[proper_left] = quat_flip.clip(-1,1)
            if self.imu_fe:
                rot_angle = 2 * np.arccos(x[:,0])
                rot_angle_vel = np.append(0,np.diff(rot_angle)).clip(-1,1)
                rot_angle_vel[self.start] = 0
                rot_angle = (2 * rot_angle / np.pi - 1).reshape(-1,1)
                x = np.hstack([x,rot_angle,rot_angle_vel.reshape(-1,1)])
                n_rot_base = x.shape[1]
                if self.euler:
                    proper = ~np.isclose(x[:,:4],0,atol=1e-7).any(axis=1)
                    euler_angles = np.zeros_like(x[:,:3])
                    euler_angles[proper] = np.column_stack(quaternion_to_euler(x[proper,:4]))
                    x = np.hstack([x,euler_angles / np.pi])
                    n_rot_base += euler_angles.shape[1]
                # relative quat to next step
                if hasattr(self.cfg,'rel_quat'):
                    # note that relative_rotation_quaternion uses only x[:,:4]
                    for i in range(self.cfg.rel_quat):
                        quat_shift = np.zeros_like(x)
                        quat_shift[i+1:] = x[:-i-1]
                        for j in range(i+1):
                            quat_shift[self.start + j] = 0
                        quat_rel = relative_rotation_quaternion(x,quat_shift)
                        quat_rel = quat_rel*((quat_rel[:,:1]>0)*2-1)
                        x = np.hstack([x,quat_rel])
                    if self.scale_quant > 0:
                        rq_len = x.shape[1] - n_rot_base
                        x[:,n_rot_base:] = (x[:,n_rot_base:] - self.rq_median[:,:rq_len])/self.rq_IQR[:,:rq_len]
                    if self.clip_quat > 0:
                        x[:,n_rot_base:] = x[:,n_rot_base:].clip(-self.clip_quat,self.clip_quat)
        return x

    def get_bounds(self, x):
        end = np.append(np.where(x[1:] != x[:-1])[0] +1, len(x))
        start = np.append(0,end[:-1])
        return start,end

    def tof_smoothing(self, tof):
        t = tof.size(0)  # seq length
        window_size = self.cfg.tof_win  # Window size for averaging
        n_windows = tof.size(-1) // window_size
        x = tof.view(t, n_windows, window_size)
        if self.tof_drop > 0:
            mask = torch.bernoulli(torch.full_like(x,1 - self.tof_drop))
            return (x * mask).sum(dim=-1) / (1e-8 + mask.sum(dim=-1))
        else: 
            return x.mean(dim=-1)

    def ind_transform(self, x):
        # sep = torch.zeros_like(x['rot'][...,:self.cfg.sep])
        sep = torch.zeros(3,self.cfg.image_size,self.cfg.sep)
        features = [self.ind_prep['acc'](x['acc']),sep,
                    self.ind_prep['rot'](0.5 * (x['rot'] + 1)),sep,
                    self.ind_prep['thm'](torch.median(x['thm'],dim=-1,keepdims=True)[0]),sep,
                    self.ind_prep['thm'](self.tof_smoothing(x['tof']))
                   ]
        return torch.cat(features,dim=-1)


class MixtureDataset(Dataset):
    def __init__(self, base_dataset, alpha, exclude_keys = [], delta = 0):
        self.base_dataset = base_dataset
        self.alpha = alpha
        self.delta = delta
        self.rng = np.random.default_rng()
        self.length = len(base_dataset)
        self.exclude_keys = exclude_keys
        if isinstance(exclude_keys,str):
            self.exclude_keys = [exclude_keys]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_i = self.base_dataset[idx]
        j = np.random.randint(0, self.length)
        x_j = self.base_dataset[j]
        # lam = self.rng.beta(self.alpha, self.alpha)
        lam = 1 - self.alpha * self.rng.random()
        mixture = {k: lam * x_i[k] + (1 - lam) * x_j[k] 
                   for k in x_i if k not in self.exclude_keys}
        if self.delta > 0:
            mixture = {k: v * (1 + self.delta * (2 * torch.rand_like(v) - 1))
                       for k,v in mixture.items() }

        for k in self.exclude_keys:
            mixture[k] = x_i[k]
        return mixture


class AWP:
    def __init__(self, model, optimizer, scaler, criterion, adv_lr=1, adv_eps=0.001,
                 start_epoch=0, adv_param="weight"):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler
        self.criterion = criterion

    def attack_backward(self, x, targets, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save() 
        self._attack_step() 
        with torch.cuda.amp.autocast():
            outputs = self.model(x)
            adv_loss = self.criterion(outputs.squeeze(), targets)
        self.optimizer.zero_grad()
        self.scaler.scale(adv_loss).backward()

        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(torch.max(
                        param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class CustomNetwork(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_transformer_layers=2, output_size=18):
        super().__init__()
        # CNN for processing input4: [32*8, 5, 8, 8] -> [32*8, 4, 2, 2]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        input_len = 3 + 4
        if 'thm' in cfg.use_groups: input_len += 5
        if 'tof' in cfg.use_groups: 
            input_len += 4*2*2
        # Linear layer to project concatenated inputs to d_model
        self.input_projection = nn.Linear(input_len, d_model)

        # Transformer encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_transformer_layers
        )

        # Output layer
        self.output_layer = nn.Linear(d_model, output_size)
        if cfg.g_loss_weight > 0:
            self.g_weight = nn.Linear(d_model, 1)


    def forward(self, inputs):
        # Input shapes: [32, 8, 3], [32, 8, 4], [32, 8, 5], [32, 8, 320]

        if 'tof' in inputs:
            batch_size,seq_len = inputs['tof'].size()[:2]
            # Reshape input4: [32, 8, 320] -> [32, 8, 5, 8, 8] -> [32*8, 5, 8, 8]
            input4 = inputs['tof'].reshape(batch_size*seq_len, 5, 8, 8)

            # Apply CNN to input4: [32*8, 5, 8, 8] -> [32*8, 4, 2, 2]
            input4 = self.cnn(input4)

            # reshape and flatten CNN output: [32*8, 4, 2, 2] -> [32, 8, 16]
            inputs['tof'] = input4.reshape(batch_size, seq_len, -1)

        # Concatenate inputs: [32, 8, 3 + 4 + 5 + 16] = [32, 8, 16]
        concat_inputs = torch.cat(list(inputs.values()), dim=-1)

        # Project to d_model: [32, 8, 16] -> [32, 8, d_model]
        x = self.input_projection(concat_inputs)

        # Apply Transformer: [32, 8, d_model] -> [32, 8, d_model]
        x = self.transformer(x)
        if cfg.g_loss_weight > 0:
            log_weight = self.g_weight(x) # [32, 8, 1]
            log_weight = F.softplus(log_weight).cumsum(dim=1)
            log_weight = log_weight - log_weight[:,-1:,:]
            x = x * torch.exp(log_weight)
        # Pool over sequence dimension (mean) and project to output: [32, 8, d_model] -> [32, d_model] -> [32, 18]
        x = x.mean(dim=1)
        output = self.output_layer(x)

        # Apply softmax for classification probabilities
        # output = F.softmax(output, dim=-1)

        if self.training & (cfg.g_loss_weight > 0):
            return output,log_weight.squeeze()
        else:
            return output


def get_optimizer(model, args):
    no_decay = ["bias", "norm"]
    if cfg.no_decay:
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() 
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': args.wd},
            {'params': [p for n, p in model.named_parameters() 
                        if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]
    elif args.dif_lr & (args.model.lower() in ['b0','b1','b3','b5','b6','v2_s','v2_m','maxvit_t',
        'cn_tiny','cn_small','cn_base','mobilenet_v2','resnet18','resnet50','squeezenet1_0']):
        optimizer_parameters = model.get_param_groups(
            args.low_lr, args.lr, args.wd)
    else:
        optimizer_parameters = model.parameters()
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.lr)
    return optimizer


# Training function with mixed precision
def train_fn(model, dataloader, optimizer, scaler, criterion, scheduler, epoch, awp=None):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs = {k:batch[k].to(cfg.device) for k in batch 
                  if k not in ['label','oh_label','gesture_phase']}
        targets = batch['oh_label'].to(cfg.device)
        if cfg.g_loss_weight > 0:
             gesture_phase = batch['gesture_phase'].to(cfg.device)

        # Zero gradients
        optimizer.zero_grad()

        # Mixed precision context
        # with autocast():  # Enable FP16 computation
        with torch.amp.autocast(cfg.device.type):    
            # Forward pass
            outputs = model(inputs)  # Remove extra dimension

            # Compute loss
            if cfg.g_loss_weight > 0:
                inverse_probs = torch.logit(outputs[1])
                loss = (criterion(outputs[0].squeeze(), targets) 
                        + cfg.g_loss_weight * cfg.phase_loss(inverse_probs, gesture_phase))
            else:
                loss = criterion(outputs.squeeze(), targets)

        # Backward pass with scaling
        scaler.scale(loss).backward()

        # AWP perturbation
        if awp is not None:
            if (np.random.random() < args.awp_prob) & (epoch >= args.awp_start):
                awp.attack_backward(inputs, targets, epoch)

        # Update optimizer with scaled gradients
        scaler.step(optimizer)

        # Update the scale factor
        scaler.update()

        # Step the scheduler
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Validation function
def valid_fn(model, dataloader, criterion,return_imu_preds=False):
    model.eval()
    total_loss = 0
    preds = []
    imu_only_preds = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k:batch[k].to(cfg.device) for k in batch 
                      if k not in ['label','oh_label','gesture_phase']}
            # targets = batch['oh_label'].to(cfg.device)
            targets = batch['label'].to(cfg.device)

            # Forward pass
            outputs = model(inputs).squeeze()
            preds.append(outputs.cpu().numpy())
            if return_imu_preds:
                for k in ['tof','thm']:
                    if k in inputs:
                        inputs[k].zero_()
                imu_only_preds.append(model(inputs).squeeze().cpu().numpy())
            # Compute loss
            # loss = F.cross_entropy(outputs, targets)
            # if cfg.public == 1:
            #     # loss = F.cross_entropy(outputs, torch.argmax(targets, dim=-1))
            #     loss = F.cross_entropy(outputs, targets)
            # else:
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    if len(imu_only_preds) > 0:
        imu_only_preds = np.vstack(imu_only_preds)
    return avg_loss,np.vstack(preds),imu_only_preds

# Testing function
def test_fn(model, batch, device, return_imu_preds=False): # batch is one sample here
    model.eval()
    with torch.no_grad():
        inputs = {k:batch[k].unsqueeze(dim=0).to(device) for k in batch 
                  if k not in ['label','gesture_phase']}
        if return_imu_preds:
            for k in ['tof','thm']:
                if k in inputs:
                    inputs[k].zero_()
        outputs = model(inputs).squeeze()
    return outputs.cpu().numpy()


def get_model(cfg):
    if cfg.model == '0':
        model = CustomNetwork(d_model=cfg.d_model, nhead=cfg.nhead, num_transformer_layers=2, 
                              output_size=len(cfg.gestures))
    elif cfg.model == '1':
        model = TwoBranchModel(imu_dim=7,tof_dim=325 if 'tof' in cfg.use_groups else 0,
                               n_classes=len(cfg.gestures))
    elif cfg.model == '2':
        model = ModelVariant_GRU(num_classes=len(cfg.gestures))
    elif cfg.model == '3':
        model = TwoBranchModel80(cfg, tof_dim = 325 if 'tof' in cfg.use_groups else 0)
    elif cfg.model == '4':
        model = CMIModel(imu_dim = 7, thm_dim=5, tof_dim=320, n_classes=len(cfg.gestures))
    elif cfg.model.lower() in ['b0','b1','b3','b5','b6','v2_s','v2_m','cn_tiny','cn_small','cn_base',
                               'mobilenet_v2','squeezenet1_0','maxvit_t']: 
        model = CustomEfficientNetB0(num_classes=len(cfg.gestures),ver=cfg.model.lower(),
                                     n_clf = cfg.n_clf if hasattr(cfg,'n_clf') else 0)
    elif 'resnet' in cfg.model.lower(): 
        model = CustomResNet(num_classes=len(cfg.gestures), ver = cfg.model.lower()[-2:])
    else:
        raise ValueError(f'No model {cfg.model.lower()}')

    return model


def train_loop(ds, fold, seed):
    seed_everything(seed)
    mlc_t.add('train_loop')
    if args.folds_file in ['ln_folds','minerppdy_folds']:
        val_fold = pd.read_csv(cfg.folds)[f'5fold_seed{args.sgkf_seed}'].to_numpy('int8')
    else:
        val_fold = pd.read_csv(cfg.folds)[f'{cfg.n_fold}fold_seed1'].to_numpy('int8')
    train_idx = np.where(val_fold != fold)[0]
    val_idx = np.where(val_fold == fold)[0]
    if args.data_aug > 0:
        train_idx = np.hstack([train_idx + i*len(val_fold) for i in range(args.data_aug+1)])
    train_dataset = MixtureDataset(Subset(ds,train_idx), args.mix_alpha, 
                                   exclude_keys = 'label', delta = args.delta)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True,drop_last=True)
    fw_preds = []
    imu_fw_preds = []
    if len(val_idx) > 0:
        valid_dataset = Subset(ds,val_idx)
        valid_loader = DataLoader(valid_dataset, batch_size=4*args.batch_size,shuffle=False,
                                  num_workers=cfg.num_workers, pin_memory=True,drop_last=False)
    seed_everything(seed)    
    model = get_model(cfg).to(cfg.device)

    num_train_steps = int(len(train_dataset) / args.batch_size * args.epochs)
    num_warmup_steps = int(num_train_steps * args.warmup_ratio)
    print(f'Fold {fold} {num_train_steps} train steps {num_warmup_steps} warmup steps')

    optimizer = get_optimizer(model, args)
    scaler =(torch.amp.GradScaler(cfg.device,enabled=cfg.apex) if KAGGLE
             else torch.cuda.amp.GradScaler(enabled=cfg.apex))
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=num_warmup_steps,
                            num_training_steps=num_train_steps, num_cycles=cfg.num_cycles)
    class_weights = compute_class_weight('balanced', 
                classes=np.arange(len(cfg.gestures)), y=ds.target[train_idx])
    if args.loss == '0':
        pass
    elif args.loss == '1':
        criterion = PublicLoss(cfg=cfg, class_weights=class_weights)
    elif args.loss == '2':
        criterion = FlexLoss(cfg, class_weights, args.bce_weight, args.aux_weight,
                             args.ce_for_targets)
    elif args.loss == '3':
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights,dtype=torch.float32).to(cfg.device))

    score,avg_val_loss = 0,0
    predictions = []
    history = []
    awp = AWP(model,optimizer,scaler,criterion,args.awp_lr,args.awp_eps,
              args.awp_start) if args.awp_lr * args.awp_eps > 0 else None

    for epoch in range(args.epochs):
        avg_loss = train_fn(model,train_loader,optimizer,scaler,criterion,scheduler,epoch,awp)
        if len(val_idx) > 0:
            avg_val_loss,predictions,_ = valid_fn(model,valid_loader,criterion)
            # scoring
            class_pred = predictions.argmax(-1)
            val_y = ds.target[val_idx]
            score = score_from_int(val_y, class_pred)
            accuracy = (val_y == class_pred).mean()
            n = len(cfg.target_gestures)
            # is_target = softmax(predictions,axis=-1)[:,:n].sum(-1) > 0.5
            # class_pred[is_target] = predictions[:,:n].argmax(-1)[is_target]
            # score1 = score_from_int(val_y, class_pred)
            if (epoch % cfg.print_freq == 0) | (epoch == args.epochs - 1):
                print(f'Epoch {epoch+1:2} train_loss: {avg_loss:.3f}  val_loss: {avg_val_loss:.3f}',
                      f'score {score:<.3f} acc {accuracy:<.3f} time: {mlc_t.fsince("train_loop")} ', flush=True)
            history.append((avg_loss,avg_val_loss,score,accuracy))

    if args.save_fw_preds:
        valid_loader = DataLoader(ds, batch_size=4*args.batch_size,shuffle=False,
                                  num_workers=cfg.num_workers, pin_memory=True,drop_last=False)
        fw_preds,imu_fw_preds = valid_fn(model,valid_loader,criterion,return_imu_preds
                                         = args.save_imu_fw_preds is not None)[1:3]

    if (args.full_data is not None) | (args.save_models is not None):
        fname = cfg.output_dir + f"s{seed}.bin"
        if args.save_models:
            fname = fname.replace('.bin',f'f{fold}.bin')
        torch.save(model.state_dict(),fname)
        print(fname,'saved')


    torch.cuda.empty_cache()
    gc.collect()

    return predictions,score,avg_val_loss,val_idx,history,fw_preds,imu_fw_preds


def four_digit(a):
    return f"{a:.4f}".split(".")[-1]


class WeightedCE(nn.Module):
    def __init__(self, cfg, class_weights):
        super().__init__()
        self.class_weights = torch.FloatTensor(class_weights).to(cfg.device)

    def forward(self, logits, batch_y):
        if args.new_wce:
            log_probs = F.log_softmax(logits, dim=-1)
            weighted_log_probs = log_probs * self.class_weights
            loss = -torch.sum(weighted_log_probs * batch_y, dim=-1)
        else:
            loss = -torch.sum(F.log_softmax(logits, dim=-1) * batch_y, dim=-1).mean()
            sample_weights = torch.sum(batch_y * self.class_weights, dim=-1)
            loss = loss * sample_weights
        return loss.mean()


class FlexLoss(nn.Module):
    def __init__(self, cfg, class_weights, bce_weight=0, aux_weight=0, 
                 ce_for_targets = False):
        super().__init__()
        sum_loss_weights = 1 + bce_weight + aux_weight
        self.ce_weight = 1 / sum_loss_weights
        self.bce_weight = bce_weight / sum_loss_weights
        self.aux_weight = aux_weight /sum_loss_weights
        self.num_targets = len(cfg.target_gestures)
        self.num_classes = len(cfg.gestures)
        target_weights = [1/np.sum(1/class_weights[self.num_targets:]),
                          1/np.sum(1/class_weights[:self.num_targets])]
        self.ce_for_targets = ce_for_targets
        if ce_for_targets:
            ce_weights = np.append(class_weights[:self.num_targets],target_weights[1])
            self.ce_loss = WeightedCE(cfg, ce_weights)
        else:
            self.ce_loss = WeightedCE(cfg, class_weights)
        self.bce_loss = WeightedCE(cfg, target_weights)
        self.aux_loss = WeightedCE(cfg, class_weights[self.num_targets:])


    def forward(self, logits, batch_y):
        if len(batch_y.shape) == 1:
            batch_y = F.one_hot(batch_y, num_classes=self.num_classes).float()
        if self.ce_for_targets:
            ce_logits = torch.cat([logits[...,:self.num_targets],
                                   logits[...,:self.num_targets].logsumexp(dim=-1,keepdims=True)],dim=-1)
            ce_y = torch.cat([batch_y[:,:self.num_targets],
                              batch_y[:,self.num_targets:].sum(dim=-1,keepdims=True)],dim=-1)
            total_loss = self.ce_weight * self.ce_loss(ce_logits, ce_y)
        else:
            total_loss = self.ce_weight * self.ce_loss(logits, batch_y)
        if self.bce_weight > 0:
            target_logits = torch.cat([logits[...,self.num_targets:].logsumexp(dim=-1,keepdims=True),
                                       logits[...,:self.num_targets].logsumexp(dim=-1,keepdims=True)],dim=-1)
            target_y = torch.cat([batch_y[:,self.num_targets:].sum(dim=-1,keepdims=True),
                                  batch_y[:,:self.num_targets].sum(dim=-1,keepdims=True)],dim=-1)
            total_loss = total_loss + self.bce_weight * self.bce_loss(target_logits, target_y)
        if self.aux_weight > 0:
            aux_logits = logits[...,self.num_targets:]
            aux_y = batch_y[:,self.num_targets:] / (batch_y[:,self.num_targets:].sum(
                dim=-1,keepdims=True) + 1e-7)
            total_loss = total_loss + self.aux_weight * self.aux_loss(aux_logits, aux_y)
        return total_loss


def label_smoothing_loss(pred, target, smoothing=0.1):
    """Label smoothing loss"""
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(pred, dim=-1)
    nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -log_probs.mean(dim=-1)
    loss = confidence * nll_loss + smoothing * smooth_loss
    return loss.mean()


class PublicLoss(nn.Module):
    def __init__(self, cfg, class_weights):
        super().__init__()
        self.class_weights = torch.FloatTensor(class_weights).to(cfg.device)

    def forward(self, logits, batch_y):
        # Handle mixup targets
        if len(batch_y.shape) == 2 and batch_y.shape[1] > 1:  # One-hot or mixed
            loss = -torch.sum(F.log_softmax(logits, dim=1) * batch_y, dim=1).mean()
            # For accuracy, use argmax of target
            # targets = batch_y.argmax(dim=1)
        else:
            targets = batch_y.long()
            loss = label_smoothing_loss(logits, targets, smoothing=0.1)

        # Apply class weights
        if len(batch_y.shape) == 2:
            sample_weights = torch.sum(batch_y * self.class_weights.unsqueeze(0), dim=1)
            loss = (loss * sample_weights).mean()
        return loss


class MultiClassifier(nn.Module):
    def __init__(self, in_features, n_classes, n_clf, dropout_rate=0.2):
        super().__init__()

        self.n_clf = n_clf
        self.dropout_rate = dropout_rate

        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.SELU(),
                nn.Dropout(dropout_rate), 
                nn.Linear(in_features // 2, n_classes)
            ) for _ in range(n_clf)
        ])

    def forward(self, x):
        return torch.stack([classifier(x) for classifier in self.classifiers])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='0', required=False)
    parser.add_argument("--lr", type=float, default=0.001, required=False)
    parser.add_argument("--low_lr", type=float, default=1e-5, required=False)
    parser.add_argument("--head_lr", type=float, default=1, required=False)
    parser.add_argument("--wd", type=float, default=0.001, required=False)
    parser.add_argument("--max_len", type=int, default=32, required=False)
    parser.add_argument("--seed", type=str, default='123', required=False) # list of seeds
    parser.add_argument("--epochs", type=int, default=1, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--input", type=str, default="./data/", required=False)
    parser.add_argument("--output", type=str, default='', required=False)
    parser.add_argument("--my_data_path", type=str, default="./my_data/", required=False)
    parser.add_argument("--n_cycles", type=float, default=0.5, required=False)
    parser.add_argument("--warmup_ratio", type=float, default=0, required=False)
    parser.add_argument('--full_data', action=argparse.BooleanOptionalAction)
    parser.add_argument("--grad_norm", type=float, default=1000, required=False)
    parser.add_argument("--gamma", type=float, default=0,required=False) # for focal loss
    # path to the model checkpoints
    parser.add_argument("--weights", type=str, default='', required=False)  
    parser.add_argument('--predict', action=argparse.BooleanOptionalAction)
    parser.add_argument('--g_loss_weight', type=float, default=0,required=False)    
    parser.add_argument('--speed_test', action=argparse.BooleanOptionalAction)
    # checkpoints indices to be selected for prediction; the order in not specified
    parser.add_argument("--checkpoint_idx", type=str, default='01234', required=False)
    parser.add_argument("--gpu", type=str, default='t4', required=False) 
    # 'ln_folds' my with sgkf seeds 'minerppdy_folds' minerppdy with sgkf seeds
    parser.add_argument("--folds_file", type=str, default='team_folds', required=False) 
    parser.add_argument("--d_model", type=int, default=64, required=False)
    parser.add_argument("--nhead", type=int, default=4, required=False)
    parser.add_argument("--loss", type=str, default='0', required=False)
    parser.add_argument('--bce_weight', type=float, default=0,required=False)    
    parser.add_argument('--aux_weight', type=float, default=0,required=False)    
    parser.add_argument('--ce_for_targets', action=argparse.BooleanOptionalAction)    
    parser.add_argument('--mix_alpha', type=float, default=0,required=False)    
    parser.add_argument('--delta', type=float, default=0,required=False)    
    parser.add_argument('--save_fw_preds', action=argparse.BooleanOptionalAction)
    parser.add_argument('--save_imu_fw_preds', action=argparse.BooleanOptionalAction)
    parser.add_argument('--save_models', action=argparse.BooleanOptionalAction)
    parser.add_argument('--imu_only', action=argparse.BooleanOptionalAction)
    parser.add_argument('--imu_fe', action=argparse.BooleanOptionalAction)
    parser.add_argument('--awp_lr', type=float, default=0,required=False)    
    parser.add_argument('--awp_eps', type=float, default=0,required=False)    
    parser.add_argument('--awp_start', type=int, default=0,required=False)    
    parser.add_argument('--awp_prob', type=float, default=0,required=False)   
    parser.add_argument('--tof_win', type=int, default=16,required=False)    
    parser.add_argument('--sep', type=int, default=0,required=False)    
    parser.add_argument('--hflip', type=float, default=0,required=False)   
    parser.add_argument('--dif_lr', action=argparse.BooleanOptionalAction)
    parser.add_argument('--add_ave', action=argparse.BooleanOptionalAction)
    parser.add_argument('--flip_left', action=argparse.BooleanOptionalAction)
    parser.add_argument('--flip_rot', action=argparse.BooleanOptionalAction)
    parser.add_argument('--flip_tof', action=argparse.BooleanOptionalAction)
    parser.add_argument('--sgkf_seed', type=int, default=0,required=False)
    parser.add_argument('--fill_rot_mean', action=argparse.BooleanOptionalAction)
    parser.add_argument('--rand_aug', action=argparse.BooleanOptionalAction)
    parser.add_argument('--data_aug', type=int, default=0,required=False)
    parser.add_argument('--y_aug_max', type=float, default=7,required=False)   
    parser.add_argument('--z_aug_max', type=float, default=45,required=False)   
    parser.add_argument('--new_wce', action=argparse.BooleanOptionalAction)
    # n_clf classifiers trained independently
    parser.add_argument('--n_clf', type=int, default=0,required=False)
    parser.add_argument('--imu_image_weight', type=float, default=0,required=False)   
    parser.add_argument('--rel_quat', type=int, default=0,required=False)
    # percentile for robast scaling of rel_quat (e.g. 15 or 25)
    parser.add_argument('--scale_quant', type=int, default=0,required=False)
    parser.add_argument('--clip_quat', type=float, default=0,required=False)
    parser.add_argument('--euler', action=argparse.BooleanOptionalAction)
    # methods: cosine,angular_sum, gramian
    parser.add_argument("--gaf", type=str, default='', required=False) 
    parser.add_argument('--fix_outliers', action=argparse.BooleanOptionalAction)
    parser.add_argument('--tof_drop', type=float, default=0,required=False)
    return parser.parse_args()


# Main script
if __name__ == "__main__":
    print('start')
    mlc_t = mlc.time.Timer()
    # Parse command-line arguments
    args = parse_args()
    if args.predict:
        cfg.train = False
    if args.full_data:
        cfg.trn_folds = [-1]
    cfg.folds = cfg.folds.replace('team_folds',args.folds_file)
    for key in ['max_len','d_model','nhead','imu_fe','tof_win','sep','n_clf',
                'imu_image_weight','rel_quat','scale_quant','clip_quat','gaf',
                'tof_drop']:
        setattr(cfg, key, getattr(args, key))
    cfg.model = args.model.lower()
    image_size_dic = defaultdict(lambda: 224, {'b1':240,'b3':300,'b5':456,'b6':528,
                                               'v2_s':384,'v2_m':480})
    cfg.image_size = image_size_dic[cfg.model]
    cfg.no_decay = args.dif_lr is None
    for key in ['flip_left','flip_rot','flip_tof','fill_rot_mean','euler']:
        setattr(cfg, key, getattr(args, key) is not None)
    if args.data_aug > 0:
        print(args.data_aug,args.y_aug_max,args.z_aug_max)
        cfg.z_aug_grid = [d for d in np.linspace(
            -args.z_aug_max,args.z_aug_max,args.data_aug + 1) if abs(d) > 2.1]
        cfg.y_aug_grid = [d for d in np.linspace(
            -args.y_aug_max,args.y_aug_max,args.data_aug + 2) if abs(d) > 2.1]
        print('y_aug_grid =',cfg.y_aug_grid)
        print('z_aug_grid =',cfg.z_aug_grid)

    if args.imu_only:
        cfg.use_groups = ['acc','rot']

    # save after all changes in cfg   
    save_namespace_json(cfg, cfg.output_dir + 'cfg.json')
    # Load datasets
    mlc_t.add('load')
    # train = pd.read_csv(data_path + 'train.csv')
    train = pd.read_feather(my_data_path + 'train.feather')
    print('loaded',mlc_t.fsince('load'))
    if args.fix_outliers:
        train = Fix_outlier(train, args.imu_only)
        print('outliers fixed',mlc_t.fsince('load'))
    demo = pd.read_csv(
        data_path + 'train_demographics.csv') if args.flip_left else None
    ds = CustomDataset(train, cfg, args.mix_alpha, demo)
    print('ds created',mlc_t.fsince('load'))

    if cfg.train:
        scores,losses = [],[]
        oof = np.zeros((train.sequence_id.nunique(),len(cfg.gestures)),dtype=np.float32)
        imu_oof = oof.copy()
        fw_preds = np.zeros((len(cfg.trn_folds),) + oof.shape, dtype=np.float32)
        imu_fw_preds = fw_preds.copy()
        if args.folds_file in ['ln_folds','minerppdy_folds']:
            print(f'folds_file {args.folds_file}, sgkf {args.sgkf_seed}')
        for seed in args.seed: # each seed in args.seed is one digit number
            print('Training with seed',seed)
            # mlc_t.add('load')
            # random.seed(int(seed))
            # random.shuffle(cfg.y_aug_grid)
            # random.shuffle(cfg.z_aug_grid)
            # ds = CustomDataset(train, cfg, args.mix_alpha, demo)
            # print('ds created',mlc_t.fsince('load'))
            seed_oof = np.zeros(len(oof),dtype='int8')
            for fold in cfg.trn_folds:
                # print(10*int(seed) + fold,type(10*int(seed) + fold))
                pred,score,val_loss,val_idx,history,fw_pred,imu_fw_pred = train_loop(ds,fold,seed)
                if args.save_fw_preds:
                    fw_preds[fold] += fw_pred
                    if len(imu_fw_pred) > 0:
                        imu_fw_preds[fold] += imu_fw_pred
                if cfg.save_history:
                    if (fold == 0) & (seed == args.seed[0]) & (not KAGGLE):
                        pd.DataFrame(history,columns=['train_loss','val_loss','score','accuracy']).to_csv(
                            f'models/{cfg.job_id}/history.csv',index=False)
                if len(val_idx) > 0:
                    scores.append(score)
                    losses.append(val_loss) 
                    oof[val_idx] += pred / len(args.seed)
                    seed_oof[val_idx] = pred.argmax(-1)
            if len(val_idx) > 0:
                print(f'Average score: {np.mean(scores):.4f} loss: {np.mean(losses):.4f} scores std: {np.std(scores):.4f}')
                if args.folds_file in ['ln_folds','minerppdy_folds']:
                    out = '\t'.join(f'={_:.4f}' for _ in scores[-5:])
                    print(f'Scores and overall: {out}',
                          f'{score_from_int(ds.target, seed_oof):.4f}')

        if len(val_idx) > 0:
            tg = ds.target[:len(oof)]
            oof_score = score_from_int(tg, oof.argmax(-1))
            logloss = log_loss(tg, softmax(oof,-1),labels=np.arange(oof.shape[-1]))
            print(f'Total for mean oof score/logloss {oof_score:.4f}/{logloss:.2f}')
            if args.save_fw_preds:
                np.save(f'models/{cfg.job_id}/fw_preds.npy',fw_preds / len(args.seed))
                print('Foldwise predictions with shape',fw_preds.shape,'saved')
            if args.save_imu_fw_preds:
                np.save(f'models/{cfg.job_id}/imu_fw_preds.npy',imu_fw_preds / len(args.seed))
                print('imu only fw_preds with shape',imu_fw_preds.shape,'saved')

            if not KAGGLE:
                np.save(f'models/{cfg.job_id}/oof_{four_digit(oof_score)}.npy',oof)
                print(f'Debug output. Saved oof after {seed=} {fold=}')
    # predict
    else:
        pass
    print(f'Total time ',mlc_t.fsince(0))



['tf79-imu-fl-aug-8000','tf79-imu-fl-7928',
 'nb0-imu-rq2-8036','nb3-imu-rq5-8220',
 'nb5-imu-aug-8095','nb5-imu-rq4-8223',
 'tf79-fl-8325',
 'nb0-8424','nb3-rq2-sc15-aug-8671',
 'nb5-rq3-sc15-8694','nv2_m-aug-8704'
 ]


['nb0-imu-rq2-8036',
# python main.py --epochs 2 --lr 0.0007 --low_lr 1e-5 --warmup_ratio 0.1 --batch_size 32 --seed 123 \
# --mix_alpha 0.4 --wd 0.1 --max_len 88 --model b0 --loss 2 --bce_weight 4 --aux_weight 0.3 \
# --ce_for_targets --imu_only --imu_fe --flip_left --sep 2 --data_aug 6 \
# --y_aug_max 9 --z_aug_max 30 --rel_quat 2 --full_data
'nb3-imu-rq5-8220',
#  python main.py --epochs 2 --lr 0.0011 --low_lr 1e-5 --warmup_ratio 0.1 --batch_size 32 --seed 123 \
# --mix_alpha 0.4 --wd 0.1 --max_len 88 --model b3 --loss 2 --bce_weight 4 --aux_weight 0.3 \
# --ce_for_targets --imu_fe --rel_quat 5 --imu_only --sep 2 \
# --flip_left --flip_rot --fill_rot_mean \
# --data_aug 6 --y_aug_max 9 --z_aug_max 30 --full_data
'nb5-imu-aug-8095',
#  python main.py --epochs 2 --lr 0.0009 --low_lr 1e-5 --warmup_ratio 0.1 --batch_size 16 --seed 123 \
# --mix_alpha 0.4 --wd 0.1 --max_len 88 --model b5 --loss 2 --bce_weight 4 --aux_weight 0.3 \
# --ce_for_targets --tof_win 4 --imu_fe --imu_only --sep 2 --flip_left --flip_rot \
# --data_aug 6 --y_aug_max 9 --z_aug_max 30 --full_data
'nb5-imu-rq4-8223',
#  python main.py --epochs 2 --lr 0.0009 --low_lr 1e-5 --warmup_ratio 0.1 --batch_size 16 --seed 123 \
# --mix_alpha 0.4 --wd 0.1 --max_len 88 --model b5 --loss 2 --bce_weight 4 --aux_weight 0.3 \
# --ce_for_targets --imu_fe --rel_quat 4 --imu_only --sep 2 --flip_left --flip_rot \
# --data_aug 6 --y_aug_max 9 --z_aug_max 30 --full_data
'nb0-8424',
#  python main.py --epochs 11 --lr 0.0014 --low_lr 1e-5 --warmup_ratio 0.1 --batch_size 32 --seed 123 \
# --mix_alpha 0.4 --wd 0.1 --max_len 88 --model b0 --loss 2 --bce_weight 4 --aux_weight 0.3 \
# --ce_for_targets --tof_win 4 --flip_left --flip_rot --flip_tof --full_data
'nb3-rq2-sc15-aug-8671',
#  python main.py --epochs 3 --lr 0.0005 --low_lr 1e-5 --warmup_ratio 0.1 --batch_size 16 --seed 123 \
# --mix_alpha 0.4 --wd 0.1 --max_len 88 --model b3 --loss 2 --bce_weight 4 --aux_weight 0.3 \
# --ce_for_targets --tof_win 4 --imu_fe --sep 2 --flip_left --flip_rot --fill_rot_mean --rel_quat 3 \
# --scale_quant 15 --clip_quat 1 --tof_drop 0.5 --data_aug 6 \
# --y_aug_max 9 --z_aug_max 30 --full_data
'nb5-rq3-sc15-8694',
#  python main.py --epochs 9 --lr 0.001 --low_lr 1e-5 --warmup_ratio 0.1 --batch_size 16 --seed 123 \
# --mix_alpha 0.4 --wd 0.1 --max_len 88 --model b5 --loss 2 --bce_weight 4 --aux_weight 0.3 \
# --ce_for_targets --tof_win 4 --imu_fe --sep 2 --flip_left --flip_rot --fill_rot_mean --rel_quat 3 \
# --scale_quant 15 --clip_quat 1 --full_data
'nv2_m-aug-8704'
# python main.py --epochs 3 --lr 0.0005 --low_lr 1e-5 --warmup_ratio 0.1 --batch_size 16 --seed 123 \
# --mix_alpha 0.4 --wd 0.01 --max_len 88 --model v2_m --loss 2 --bce_weight 4 --aux_weight 0.3 \
# --ce_for_targets --tof_win 4 --imu_fe --sep 2 --flip_left --flip_rot --rel_quat 3 \
# --scale_quant 25 --clip_quat 1 --tof_drop 0.5 --data_aug 6 \
# --y_aug_max 9 --z_aug_max 30 --full_data
]


!python main.py --epochs 1 --lr 0.0009 --low_lr 1e-5 --warmup_ratio 0.1 --batch_size 16 --seed 123 \
--mix_alpha 0.4 --wd 0.1 --max_len 88 --model b5 --loss 2 --bce_weight 4 --aux_weight 0.3 \
--ce_for_targets --imu_fe --rel_quat 4 --imu_only --sep 2 --flip_left --flip_rot \
--data_aug 2 --y_aug_max 9 --z_aug_max 30 --full_data

