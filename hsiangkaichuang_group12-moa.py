import sys

# sys.path.append('../input/iterative-stratification/iterative-stratification-master')
%mkdir model
%mkdir interim
%mkdir exp
%mkdir preprocessed

import pandas as pd
import numpy as np
import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import os
import time
from umap import UMAP

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import KFold, RepeatedKFold

from tqdm import tqdm
import random
import os
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

DEFAULT_SEED = 512
seed_everything(seed_value=DEFAULT_SEED)


class MOADataset(torch.utils.data.Dataset):
    def __init__(self, feature_file_path, target_file_path, feature_mode):
        self.features = pd.read_csv(feature_file_path)
        if target_file_path:
            self.targets = pd.read_csv(target_file_path).drop(columns=["sig_id"]).to_numpy()
        else:
            self.targets = [0] * len(self.features)

        self.cp_type = pd.Categorical(self.features["cp_type"], categories=["ctl_vehicle", "trt_cp"]).codes
        self.cp_dose = pd.Categorical(self.features["cp_dose"], categories=['D1', 'D2']).codes
        self.cp_time = pd.Categorical(self.features["cp_time"], categories=['24', '48', '72']).codes
        self.cnt_features = self.features.drop(columns=["sig_id", "cp_type", "cp_time", "cp_dose"])

        if feature_mode == "CNT":
            self.feat = self.cnt_features.to_numpy()
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        return {'feature': self.feat[index], 'target': self.targets[index]}
    
    def __len__(self):
        return len(self.feat)
    
class MOABatchCollate(object):
    def __call__(self, batch):
        features = torch.from_numpy(np.array([item['feature'] for item in batch])).float()
        targets = torch.from_numpy(np.array([item['target'] for item in batch])).float()
        return {'features': features, 'targets': targets}


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class MLP(nn.Module):
    def __init__(self, num_features = 256, num_targets = 256, dropout_r=0.1):
        super(MLP, self).__init__()
        self.num_features = num_features
        self.num_target = num_targets
        self.mlp = nn.Sequential(nn.BatchNorm1d(self.num_features),
                                 nn.Dropout(p=dropout_r),
                                 nn.utils.weight_norm(nn.Linear(self.num_features, self.num_features * 4)),
                                 Mish(),
                                 nn.BatchNorm1d(self.num_features*4),
                                 nn.Dropout(p=dropout_r),
                                 nn.utils.weight_norm(nn.Linear(self.num_features * 4, self.num_target*4)),
                                 Mish(),
                                 nn.BatchNorm1d(self.num_target*4),
                                 nn.Dropout(p=dropout_r),
                                 nn.utils.weight_norm(nn.Linear(self.num_target * 4, self.num_target)))
        
    def forward(self, features):
        logits = self.mlp(features)

        return logits


DATA_SET_DIR = "../input/lish-moa"
MODEL_DIR = "./exp"
train_features = pd.read_csv(f'{DATA_SET_DIR}/train_features.csv')
train_targets_scored = pd.read_csv(f'{DATA_SET_DIR}/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv(f'{DATA_SET_DIR}/train_targets_nonscored.csv')

test_features = pd.read_csv(f'{DATA_SET_DIR}/test_features.csv')
sample_submission = pd.read_csv(f'{DATA_SET_DIR}/sample_submission.csv')

IS_TRAIN = True
DATA_DIR = './preprocessed'
# label smoothing
PMIN = 0.0
PMAX = 1.0

# submission smoothing
SMIN = 0.0
SMAX = 1.0



GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]

for col in (GENES + CELLS):
    vec_len = len(train_features[col].values)
    vec_len_test = len(test_features[col].values)
    raw_vec = pd.concat([train_features, test_features])[col].values.reshape(vec_len+vec_len_test, 1)
    if IS_TRAIN:
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
        transformer.fit(raw_vec)
        pd.to_pickle(transformer, f'{DATA_DIR}/{col}_quantile_transformer.pkl')
    else:
        transformer = pd.read_pickle(f'{DATA_DIR}/{col}_quantile_transformer.pkl')        

    train_features[col] = transformer.transform(train_features[col].values.reshape(vec_len, 1)).reshape(1, vec_len)[0]
    test_features[col] = transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]


# GENES
n_comp = 50
n_dim = 15

data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])

if IS_TRAIN:
    pca = PCA(n_components=n_comp, random_state=DEFAULT_SEED).fit(train_features[GENES])
    umap = UMAP(n_components=n_dim, random_state=DEFAULT_SEED).fit(train_features[GENES])
    pd.to_pickle(pca, f"{DATA_DIR}/pca_g.pkl")
    pd.to_pickle(umap, f"{DATA_DIR}/umap_g.pkl")
else:
    pca = pd.read_pickle(f"{DATA_DIR}/pca_g.pkl")
    umap = pd.read_pickle(f"{DATA_DIR}/umap_g.pkl")
    
data2 = pca.transform(data[GENES])
data3 = umap.transform(data[GENES])

train2 = data2[:train_features.shape[0]]
test2 = data2[-test_features.shape[0]:]
train3 = data3[:train_features.shape[0]]
test3 = data3[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])
train3 = pd.DataFrame(train3, columns=[f'umap_G-{i}' for i in range(n_dim)])
test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])
test3 = pd.DataFrame(test3, columns=[f'umap_G-{i}' for i in range(n_dim)])

train_features = pd.concat((train_features, train2, train3), axis=1)
test_features = pd.concat((test_features, test2, test3), axis=1)

#CELLS
n_comp = 15
n_dim = 5

data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])


if IS_TRAIN:
    pca = PCA(n_components=n_comp, random_state=DEFAULT_SEED).fit(train_features[CELLS])
    umap = UMAP(n_components=n_dim, random_state=DEFAULT_SEED).fit(train_features[CELLS])
    pd.to_pickle(pca, f"{DATA_DIR}/pca_c.pkl")
    pd.to_pickle(umap, f"{DATA_DIR}/umap_c.pkl")
else:
    pca = pd.read_pickle(f"{DATA_DIR}/pca_c.pkl")
    umap = pd.read_pickle(f"{DATA_DIR}/umap_c.pkl")   

data2 = pca.transform(data[CELLS])
data3 = umap.transform(data[CELLS])

train2 = data2[:train_features.shape[0]]
test2 = data2[-test_features.shape[0]:]
train3 = data3[:train_features.shape[0]]
test3 = data3[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])
train3 = pd.DataFrame(train3, columns=[f'umap_C-{i}' for i in range(n_dim)])
test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])
test3 = pd.DataFrame(test3, columns=[f'umap_C-{i}' for i in range(n_dim)])

train_features = pd.concat((train_features, train2, train3), axis=1)
test_features = pd.concat((test_features, test2, test3), axis=1)


print(train_features)
print("------")
print(test_features)


from sklearn.feature_selection import VarianceThreshold

if IS_TRAIN:
    var_thresh = VarianceThreshold(threshold=0.5).fit(train_features.iloc[:, 4:])
    pd.to_pickle(var_thresh, f"{DATA_DIR}/variance_thresh0_5.pkl")
else:
    var_thresh = pd.read_pickle(f"{DATA_DIR}/variance_thresh0_5.pkl")
                                
data = pd.concat([train_features, test_features])
data_transformed = var_thresh.transform(data.iloc[:, 4:])

train_features_transformed = data_transformed[ : train_features.shape[0]]
test_features_transformed = data_transformed[-test_features.shape[0] : ]


train_features = pd.DataFrame(train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                              columns=['sig_id','cp_type','cp_time','cp_dose'])

train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)


test_features = pd.DataFrame(test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                             columns=['sig_id','cp_type','cp_time','cp_dose'])

test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)

print(train_features.shape)
print(test_features.shape)

train_features.to_csv("./preprocessed/train_preprocessed.csv")
test_features.to_csv("./preprocessed/test_preprocessed.csv")


# training hyper params
EPOCHS = 15
BATCH_SIZE = 2048
NFOLDS = 10 # 10
NREPEATS = 1
NSEEDS = 5 # 5
NUM_FEATURE = train_features.shape[1]
NUM_TARGET = 206
LR = 5e-4

SEED = 3600
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

PCT_START = 0.2
DIV_FACS = 1e3
MAX_LR = 1e-2


train_dataset = MOADataset("./preprocessed/train_preprocessed.csv", "../input/lish-moa/train_targets_scored.csv", feature_mode="CNT")
test_dataset = MOADataset("./preprocessed/test_preprocessed.csv", target_file_path=None, feature_mode='CNT')


def train_step(model:torch.nn.Module, optimizor:torch.optim.Optimizer, scheduler,
            train_loader: DataLoader, criteria: torch.nn.Module):
    
    model.train()
    total_loss = 0
    for batch in train_loader:
        features = batch['features'].to(DEVICE)
        targets = batch['targets'].to(DEVICE)
        
        
        optimizor.zero_grad()
        predict = model(features)
        loss = criteria(predict, targets)
        loss.backward()
        optimizor.step()
        scheduler.step()

        total_loss += loss.item()
    
    total_loss /= len(train_loader)

    return total_loss

def valid_step(model:torch.nn.Module, data_loader: DataLoader, criteria: torch.nn.Module):
    model.eval()
    total_loss = 0
    preds = []
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
        
            predict = model(features)
            loss = criteria(predict, targets)

            total_loss += loss.item()
            preds.append(predict.sigmoid().detach().cpu().numpy())
        
        total_loss /= len(data_loader)
        preds = np.concatenate(preds)

    return total_loss, preds

def inference_step(model:torch.nn.Module, data_loader: DataLoader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(DEVICE)
        
            predict = model(features)
            preds.append(predict.sigmoid().detach().cpu().numpy())
        preds = np.concatenate(preds)

    return preds


def run_single_fold(model:torch.nn.Module, optimizor:torch.optim.Optimizer, scheduler,
                    train_loader: DataLoader, valid_loader: DataLoader, criteria: torch.nn.Module, seed, fold):
    seed_everything(seed)
    train_loss_history = []
    valid_loss_history = []
    best_loss = np.inf
    best_valid_preds = None
    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_step(model, optimizor, scheduler, train_loader, criteria)
        valid_loss, valid_preds = valid_step(model, valid_loader, criteria)

        if valid_loss < best_loss:            
            best_loss = valid_loss
            best_loss_epoch = epoch
            model.to('cpu')
            torch.save(model.state_dict(), f"./exp/mlp_SEED{seed}_FOLD{fold}_best.pth")
            model.to(DEVICE)
        
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        print(valid_loss)

    return train_loss_history, valid_loss_history

def run_k_fold(train_dataset, test_dataset, seed):
    mskf = RepeatedKFold(n_splits=NFOLDS, n_repeats=NREPEATS, random_state=seed)
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    predictions = np.zeros((len(test_dataset), NUM_TARGET))
    
    for fold, (t_idx, v_idx) in enumerate(mskf.split(train_dataset.cnt_features)):
        print(f"Training fold {fold+1}/{NFOLDS * NREPEATS}")
        
        train_fold_dataset = Subset(train_dataset, t_idx)
        valid_fold_dataset = Subset(train_dataset, v_idx)
        train_fold_loader = DataLoader(train_fold_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=MOABatchCollate(), num_workers=4)  # 減少 num_workers 避免記憶體問題
        valid_fold_loader = DataLoader(valid_fold_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=MOABatchCollate(), num_workers=4)
        
        model = MLP(NUM_FEATURE, NUM_TARGET, dropout_r=0.2).to(DEVICE)
        optimizer = torch.optim.Adam(lr=LR, params=list(model.parameters()))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, 
            pct_start=PCT_START, 
            div_factor=DIV_FACS, 
            max_lr=MAX_LR, 
            epochs=EPOCHS, 
            steps_per_epoch=len(train_fold_loader)
        )
        
        train_loss_history, valid_loss_history = run_single_fold(model, optimizer, scheduler, train_fold_loader, valid_fold_loader, loss_fn, seed, fold)

        model = MLP(NUM_FEATURE, NUM_TARGET, dropout_r=0.2).to(DEVICE)
        model.load_state_dict(torch.load(f"./exp/mlp_SEED{seed}_FOLD{fold}_best.pth"))
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=MOABatchCollate(), num_workers=4)
        pred = inference_step(model, test_loader)
        predictions += pred / (NFOLDS * NREPEATS)

    return predictions


# Check actual feature count
print("Check training data shape:")
print(f"train_dataset.feat.shape: {train_dataset.feat.shape}")
print(f"Actual feature count: {train_dataset.feat.shape[1]}")

print("\nCheck test data shape:")
print(f"test_dataset.feat.shape: {test_dataset.feat.shape}")

# Check preprocessed data
train_preprocessed = pd.read_csv("./preprocessed/train_preprocessed.csv")
test_preprocessed = pd.read_csv("./preprocessed/test_preprocessed.csv")

print(f"\nPreprocessed training data shape: {train_preprocessed.shape}")
print(f"Preprocessed test data shape: {test_preprocessed.shape}")

# Feature count after removing ID columns
actual_feature_count = train_preprocessed.shape[1] - 4  # Remove sig_id, cp_type, cp_time, cp_dose
print(f"Actual available feature count: {actual_feature_count}")

# Correct NUM_FEATURE parameter
NUM_FEATURE = actual_feature_count
print(f"Corrected NUM_FEATURE: {NUM_FEATURE}")


predictions = run_k_fold(train_dataset, test_dataset, SEED)
print("DONE")


result_df = pd.read_csv("../input/lish-moa/sample_submission.csv")
result_df[result_df.columns[1:]] = predictions
result_df.to_csv("submission.csv", index=False)
print("Submission created.")


df_sub = pd.read_csv("submission.csv")
df_sub

