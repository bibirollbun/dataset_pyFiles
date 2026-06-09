import random
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import torchaudio.transforms as T
import timm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from typing import Optional
from types import SimpleNamespace
from tqdm import tqdm


cfg = SimpleNamespace(**{})

cfg.train_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv'

cfg.imu_cols = [
    'acc_x', 'acc_y', 'acc_z',
    'rot_w', 'rot_x', 'rot_y', 'rot_z',
]
cfg.static_cols = [
    'sequence_id', 'sequence_type', 'gesture', 'orientation', 'subject',
    'adult_child', 'age', 'sex', 'handedness',
    'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm'
]

cfg.main_target = 'gesture'
cfg.main_num_classes = 18
cfg.group = 'subject'
cfg.seq_len = 100
cfg.n_splits = 5
cfg.curr_fold = 1
cfg.seed = 42

cfg.model_dir = 'weights'
cfg.oof_dir = 'oofs'

cfg.encoder_name = 'timm/resnet50.a1_in1k' # just for test :)
cfg.img_sz = 224
cfg.sepc_model_dropout = 0.3
cfg.im_pretrained = True

cfg.bs = 256
cfg.n_epochs = 50
cfg.patience = 5
cfg.lr = 1e-4
cfg.weight_decay = 1e-2
cfg.num_warmup_steps_ratio = 0.03
cfg.max_norm = 2.0


class SpecNormalize(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # batch, channel
        # x: (batch, channel, freq, time)
        min_ = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_ = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        return (x - min_) / (max_ - min_ + self.eps)


class SpecFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        height: int,
        hop_length: int,
        win_length: Optional[int] = None,
        out_size: Optional[int] = None,
    ):
        super().__init__()
        self.height = height
        self.out_chans = in_channels
        n_fft = height * 2 - 1
        self.feature_extractor = nn.Sequential(
            T.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length),
            T.AmplitudeToDB(top_db=80),
            SpecNormalize(),
        )
        self.out_size = out_size

        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img = self.feature_extractor(x)
        if self.out_size is not None:
            img = self.pool(img)

        return img


class SpecCNN2d(nn.Module):
    def __init__(self, in_channels, n_classes, dropout=0.2):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.gap(x)
        x = x.flatten(1)

        return self.classifier(x)


class IMG_CMIModel(nn.Module):
    def __init__(
        self, 
        imu_vars=len(cfg.imu_cols),
        n_classes=cfg.main_num_classes, 
        dropout=cfg.sepc_model_dropout, 
        img_sz=cfg.img_sz, 
        spec_height=12, 
        spec_hop_length=4, 
        spec_win_length=None,
        pretrained=cfg.im_pretrained,
        encoder=cfg.encoder_name,
    ):
        super().__init__()
        
        self.spec_feature_extractor = SpecFeatureExtractor(
            in_channels=imu_vars,
            height=spec_height,
            hop_length=spec_hop_length,
            win_length=spec_win_length,
            out_size=None # for my cnn2d
            # out_size=img_sz # for timm model
        )
        
        # self.tmodel = timm.create_model(encoder, in_chans=imu_vars, drop_rate=dropout, num_classes=n_classes, pretrained=pretrained)
        self.tmodel = SpecCNN2d(in_channels=imu_vars, n_classes=n_classes, dropout=dropout)
        
    def forward(self, _x):
        x = _x.transpose(1, 2)
        x = self.spec_feature_extractor(x)
        out = self.tmodel(x)
        
        return out


class TS_CMIDataset(Dataset):
    def __init__(
        self, 
        dataframe, 
        seq_len=cfg.seq_len, 
        main_target=cfg.main_target
    ):
        self.df = dataframe.copy().reset_index(drop=True)
        self.seq_len = seq_len
        self.main_target = main_target
        
        self.imu_cols = cfg.imu_cols
        self.has_target = self.main_target in self.df.columns
        
    def _prepare_sensor_data_raw(self, row, sensor_cols):
        processed_series_list = []
        original_lengths = []
        
        for col_name in sensor_cols:
            series = row[col_name]
            series_array = np.asarray(series, dtype=np.float64)
            original_lengths.append(len(series_array))
            processed_series_list.append(series_array)
        
        data_stacked = np.stack(processed_series_list, axis=1)
        
        for i in range(data_stacked.shape[1]):
            column_data = data_stacked[:, i]
            if np.all(np.isnan(column_data)):
                data_stacked[:, i] = 0.0
            elif np.any(np.isnan(column_data)):
                s = pd.Series(column_data)
                s_filled = s.interpolate(method='linear', limit_direction='both').ffill().bfill().fillna(0.0)
                data_stacked[:, i] = s_filled.values
        
        return data_stacked

    def __len__(self):
        return len(self.df)
    
    def _pad_or_truncate_final(self, data, target_len):
        current_len = data.shape[0]
        
        if current_len > target_len:
            truncated_data = data[-target_len:]
            return truncated_data
        elif current_len < target_len:
            padding_rows = np.zeros((target_len - current_len, data.shape[1]), dtype=data.dtype)
            padded_data = np.concatenate([padding_rows, data], axis=0)
            return padded_data
        
        return data

    def _prepare_sensor_data(self, row, sensor_cols):
        data_stacked = self._prepare_sensor_data_raw(row, sensor_cols)
        data_stacked = self._pad_or_truncate_final(data_stacked, self.seq_len)
        
        return data_stacked
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        imu_data = self._prepare_sensor_data(row, self.imu_cols)

        features = {
            'imu': torch.tensor(imu_data, dtype=torch.float32), # (seq_len, 7)
        }
        
        if self.has_target:
            features['main_target'] = torch.tensor(row[self.main_target], dtype=torch.long)
        
        return features


def fast_seq_agg(df):
    sc = cfg.static_cols
    seq_cols = [c for c in df.columns if c not in sc + ['sequence_counter', 'row_id']]
    static_cols = [c for c in sc if c in df.columns]

    df = df.sort_values(['sequence_id', 'sequence_counter']).reset_index(drop=True)

    seq_id_codes, _ = pd.factorize(df['sequence_id'])
    _, seq_start_idxs = np.unique(seq_id_codes, return_index=True)

    res = {'sequence_id': df['sequence_id'].values[seq_start_idxs]}

    for c in static_cols:
        res[c] = df[c].values[seq_start_idxs]

    for c in seq_cols:
        res[c] = np.split(df[c].values, seq_start_idxs[1:])

    res_df = pd.DataFrame(res)

    return res_df


def le(df):
    mapper_main = {
        "Above ear - pull hair": 0,
        "Cheek - pinch skin": 1,
        "Eyebrow - pull hair": 2,
        "Eyelash - pull hair": 3, 
        "Forehead - pull hairline": 4,
        "Forehead - scratch": 5,
        "Neck - pinch skin": 6, 
        "Neck - scratch": 7,
        "Drink from bottle/cup": 8,
        "Feel around in tray and pull out an object": 9,
        "Glasses on/off": 10,
        "Pinch knee/leg skin": 11, 
        "Pull air toward your face": 12,
        "Scratch knee/leg skin": 13,
        "Text on phone": 14,
        "Wave hello": 15,
        "Write name in air": 16,
        "Write name on leg": 17,
    }

    df[cfg.main_target] = df[cfg.main_target].map(mapper_main)

    return df


def comp_metric(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    bscore = f1_score(
        np.where(y_true <= 7, 1, 0),
        np.where(y_pred <= 7, 1, 0),
        zero_division=0.0,
    )

    mscore = f1_score(
        np.where(y_true <= 7, y_true, 99),
        np.where(y_pred <= 7, y_pred, 99),
        average="macro", 
        zero_division=0.0,
    )

    return (bscore + mscore) / 2, bscore, mscore


def seed_everything(seed: int = cfg.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_epoch(train_loader, model, optimizer, main_criterion, device, scheduler, current_step=0, fold=None):
    model.train()
        
    total_loss = 0
    total_samples = 0
    all_targets = []
    all_preds = []
    
    loop = tqdm(train_loader, desc='train', leave=False)

    for batch in loop:
        optimizer.zero_grad()
        
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        
        outputs = model(batch['imu'])
        targets = batch['main_target']
        loss = main_criterion(outputs, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)
        
        preds = torch.argmax(outputs, dim=1)
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        
        loop.set_postfix(loss=loss.item())
        current_step += 1
    
    avg_loss = total_loss / total_samples
    avg_m, bm, mm = comp_metric(all_targets, all_preds)

    return avg_loss, avg_m, bm, mm, current_step


def valid_epoch(val_loader, model, main_criterion, device):
    model.eval()

    total_loss = 0
    total_samples = 0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        loop = tqdm(val_loader, desc='val', leave=False)
        for batch in loop:
            for key in batch.keys():
                batch[key] = batch[key].to(device)
    
            outputs = model(batch['imu'])
            targets = batch['main_target']
            loss = main_criterion(outputs, targets)
            
            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            loop.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / total_samples
    avg_m, bm, mm = comp_metric(all_targets, all_preds)
    
    return avg_loss, avg_m, bm, mm, all_targets, all_preds


def run_training_with_stratified_group_kfold():
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.oof_dir, exist_ok=True)

    sgkf = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    targets = train_seq[cfg.main_target].values
    groups = train_seq[cfg.group].values
    
    oof_preds = np.zeros((len(train_seq), cfg.main_num_classes))
    oof_targets = train_seq[cfg.main_target].values
    
    best_models = []
    best_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_seq, targets, groups)):
        print(f'fold {fold+1}/{cfg.n_splits}')
        # if fold != cfg.curr_fold:
        #     continue
        
        train_subset = train_seq.iloc[train_idx].reset_index(drop=True)
        val_subset = train_seq.iloc[val_idx].reset_index(drop=True)
        
        train_dataset = TS_CMIDataset(
            dataframe=train_subset,
            seq_len=cfg.seq_len,
            main_target=cfg.main_target,
        )
        val_dataset = TS_CMIDataset(
            dataframe=val_subset,
            seq_len=cfg.seq_len,
            main_target=cfg.main_target,
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.bs, 
            shuffle=True, 
            pin_memory=True, 
            persistent_workers=True, 
            prefetch_factor=4, 
            num_workers=4,
            generator=g,
            worker_init_fn=lambda worker_id: np.random.seed(cfg.seed + worker_id)
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg.bs, 
            shuffle=False, 
            pin_memory=True, 
            persistent_workers=True, 
            prefetch_factor=4, 
            num_workers=4,
            generator=g,
            worker_init_fn=lambda worker_id: np.random.seed(cfg.seed + worker_id)           
        )

        model = IMG_CMIModel().to(device)

        fucking_kaggle_p100 = True
        if not fucking_kaggle_p100:
            model = torch.compile(model, mode='max-autotune')

        optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        
        main_criterion = nn.CrossEntropyLoss()

        num_training_steps = cfg.n_epochs * len(train_loader)
        num_warmup_steps = int(cfg.num_warmup_steps_ratio * num_training_steps)
        current_step = 0

        scheduler_params = {
            'optimizer': optimizer,
            'num_warmup_steps': num_warmup_steps,
            'num_training_steps': num_training_steps
        }
        scheduler = get_cosine_schedule_with_warmup(**scheduler_params)
        
        best_val_score = -np.inf
        patience_counter = 0
        fold_checkpoints = []
        
        for epoch in range(cfg.n_epochs):
            print(f'{epoch=}')
            
            train_loss, avg_m_train, bm_train, mm_train, current_step = train_epoch(
                train_loader, model, optimizer, main_criterion, device, scheduler, fold
            )
            val_loss, avg_m_val, bm_val, mm_val, _, _ = valid_epoch(
                val_loader, model, main_criterion, device
            )
            
            print(f'{train_loss=}, {avg_m_train=}, {bm_train=}, {mm_train=},')
            print(f'{val_loss=}, {avg_m_val=}, {bm_val=}, {mm_val=}')

            model_path = os.path.join(cfg.model_dir, f'model_fold{fold}_val_f1_{avg_m_val:.4f}_epoch{epoch:03d}.pt')
            torch.save(model.state_dict(), model_path)

            fold_checkpoints.append({
                'score': avg_m_val,
                'epoch': epoch,
                'model_path': model_path,
            })

            fold_checkpoints.sort(key=lambda x: x['score'], reverse=True)
            
            if len(fold_checkpoints) > 5:
                to_remove = fold_checkpoints[5:]
                fold_checkpoints = fold_checkpoints[:5]
                
                for checkpoint in to_remove:
                    if os.path.exists(checkpoint['model_path']):
                        os.remove(checkpoint['model_path'])

            if avg_m_val > best_val_score:
                best_val_score = avg_m_val
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= cfg.patience:
                print('early stopping')
                break

        best_checkpoint = fold_checkpoints[0]
        if best_checkpoint:
            model.load_state_dict(torch.load(best_checkpoint['model_path']))

        best_models.append(model)
        best_f1_scores.append(best_val_score)
        
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in val_loader:
                for key in batch.keys():
                    batch[key] = batch[key].to(device)

                outputs = model(batch['imu'])
                all_preds.append(outputs.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        oof_preds[val_idx] = all_preds

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_pred_labels = np.argmax(oof_preds, axis=1)
    oof_m, oof_bm, oof_mm = comp_metric(oof_targets, oof_pred_labels)
    print(f'{oof_m=}, {oof_bm=}, {oof_mm=}, ')
    
    oof_preds_path = os.path.join(cfg.oof_dir, f'oof_preds.npy')
    oof_targets_path = os.path.join(cfg.oof_dir, f'oof_targets.npy')
    oof_pred_labels_path = os.path.join(cfg.oof_dir, f'oof_pred_labels.npy')

    np.save(oof_preds_path, oof_preds)
    np.save(oof_targets_path, oof_targets)
    np.save(oof_pred_labels_path, oof_pred_labels)

    oof_info = {
        'oof_avg_f1': oof_m,
        'oof_binary_f1': oof_bm,
        'oof_macro_f1': oof_mm,
        'best_avg_f1_scores_per_fold': best_f1_scores,
        'mean_cv_avg_f1': np.mean(best_f1_scores),
        'std_cv_avg_f1': np.std(best_f1_scores)
    }
    
    oof_info_path = os.path.join(cfg.oof_dir, f'oof_info.json')
    with open(oof_info_path, 'w') as f:
        json.dump(oof_info, f, indent=2)

    return best_models, oof_preds


seed_everything()
g = torch.Generator(device='cpu').manual_seed(cfg.seed)


train = pd.read_csv(cfg.train_path)


train = le(train)
train_seq = fast_seq_agg(train)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


best_models, oof_preds = run_training_with_stratified_group_kfold()

