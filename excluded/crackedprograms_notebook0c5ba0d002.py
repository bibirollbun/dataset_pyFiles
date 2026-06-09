# Ariel NeurIPS 2025 — 1000 Dâhi Beyni Tek Dosya Pipeline
# Tek dosya: .py olarak çalıştırılabilir

import os
import gc
import math
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import product

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

try:
    import lightgbm as lgb
except:
    lgb = None
try:
    import xgboost as xgb
except:
    xgb = None
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except:
    torch = None

SEED = 2025
np.random.seed(SEED)
DATA_DIR = Path('/kaggle/input/ariel-data-challenge-2025')
OUT_DIR = Path('./working')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Seed & reproducibility ----------------
def seed_everything(seed=SEED):
    np.random.seed(seed)
    import random
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

seed_everything(SEED)

# ---------------- Gaussian Log-Likelihood ----------------
def gaussian_log_likelihood(y_true, mu, sigma, eps=1e-8, axis=-1):
    sigma = np.maximum(sigma, eps)
    term = -0.5*(np.log(2*np.pi)+2*np.log(sigma)+((y_true-mu)/sigma)**2)
    return term.sum(axis=axis)

def compute_competition_L(y_true, mu_pred, sigma_pred, channel_weights=None):
    if channel_weights is None:
        channel_weights=np.ones(y_true.shape[1])
    cw = np.array(channel_weights)/(np.sum(channel_weights)+1e-12)
    per_pixel = -0.5*(np.log(2*np.pi)+2*np.log(np.maximum(sigma_pred,1e-8))+((y_true-mu_pred)/np.maximum(sigma_pred,1e-8))**2)
    weighted = per_pixel*cw[np.newaxis,:]
    return weighted.sum()

# ---------------- Submission writer ----------------
def write_submission(planet_ids, mu_preds, sigma_preds, out_path:Path):
    n = len(planet_ids)
    arr = np.hstack([planet_ids.reshape(-1,1), mu_preds, sigma_preds])
    cols = ['planet_id'] + [f'mu_{i}' for i in range(mu_preds.shape[1])] + [f'sigma_{i}' for i in range(sigma_preds.shape[1])]
    df = pd.DataFrame(arr, columns=cols)
    df['planet_id'] = df['planet_id'].astype(object)
    df.to_csv(out_path/'submission.csv', index=False)
    print('submission.csv written to', out_path/'submission.csv')

# ---------------- Spectral preprocessing ----------------
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import median_filter

def basic_savgol_transform(arr, window=11, poly=3):
    if arr.ndim==1:
        return savgol_filter(arr, window_length=min(window, arr.shape[0]//2*2+1), polyorder=poly)
    else:
        out=np.zeros_like(arr)
        for i in range(arr.shape[0]):
            w=min(window, arr.shape[1]//2*2+1)
            out[i]=savgol_filter(arr[i], window_length=w, polyorder=poly)
        return out

def common_mode_subtraction(obs_matrix, method='median'):
    if method=='median': common=np.median(obs_matrix, axis=0)
    elif method=='mean': common=np.mean(obs_matrix, axis=0)
    else: common=PCA(n_components=1).fit_transform(obs_matrix.T).flatten()
    cleaned=obs_matrix-common[np.newaxis,:]
    return cleaned, common

def poly_detrend_vector(vec, degree=3):
    x=np.arange(len(vec))
    coeffs=np.polyfit(x, vec, deg=degree)
    baseline=np.polyval(coeffs, x)
    return vec-baseline, baseline

def denoise_spectrum(vec, method='combined', savgol_window=11, savgol_poly=3, medfilt_kernel=3):
    if method=='savgol':
        return savgol_filter(vec, window_length=min(savgol_window,len(vec)//2*2+1), polyorder=savgol_poly)
    elif method=='med':
        k=min(medfilt_kernel,len(vec) if len(vec)%2==1 else len(vec)-1)
        return medfilt(vec, kernel_size=k)
    else:
        filtered=medfilt(vec,kernel_size=medfilt_kernel)
        return savgol_filter(filtered, window_length=min(savgol_window,len(vec)//2*2+1), polyorder=savgol_poly)

def spectral_derivatives(vec):
    v=np.asarray(vec)
    first=np.gradient(v)
    second=np.gradient(first)
    return first, second

def log_ratio_features(vec, eps=1e-9):
    v=np.maximum(vec, eps)
    cont=median_filter(v, size=max(3,len(v)//100))
    lr=np.log(v/(cont+eps))
    return lr, cont

def aggregate_visits(visits_matrix, method='weighted_mean', weights=None):
    if method=='median': return np.median(visits_matrix, axis=0)
    elif method=='mean': return np.mean(visits_matrix, axis=0)
    elif method=='weighted_mean':
        if weights is None: var=np.var(visits_matrix, axis=0); weights=1.0/(var+1e-12)
        else: weights=np.array(weights)
        return np.average(visits_matrix, axis=0, weights=weights)

# ---------------- PyTorch MLP heteroscedastic ----------------
if torch is not None:
    class SpectraDataset(Dataset):
        def __init__(self, X, y=None):
            self.X=X.astype(np.float32)
            self.y=None if y is None else y.astype(np.float32)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            if self.y is None:
                return self.X[idx]
            return self.X[idx], self.y[idx]

    class HeteroNet(nn.Module):
        def __init__(self, in_dim, hidden=[512,256], out_dim=283):
            super().__init__()
            layers=[]
            d=in_dim
            for h in hidden:
                layers.append(nn.Linear(d,h))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(h))
                d=h
            self.body=nn.Sequential(*layers)
            self.mu_head=nn.Linear(d,out_dim)
            self.log_sigma_head=nn.Linear(d,out_dim)
        def forward(self,x):
            h=self.body(x)
            mu=self.mu_head(h)
            sigma=torch.exp(self.log_sigma_head(h))
            return mu, sigma

    def hetero_loss(mu, sigma, y, eps=1e-8):
        sigma=torch.clamp(sigma, min=eps)
        loss=0.5*(torch.log(2*math.pi)+2*torch.log(sigma)+((y-mu)/sigma)**2)
        return loss.mean()

# ---------------- Ensemble configs ----------------
def generate_model_configs():
    configs=[]
    model_types=['lgb','xgb','mlp']; seeds=[2025,42,7,123]
    lgb_params=[{'num_leaves':31,'learning_rate':0.05,'n_estimators':500},{'num_leaves':63,'learning_rate':0.03,'n_estimators':800}]
    xgb_params=[{'max_depth':6,'eta':0.05,'nrounds':500},{'max_depth':8,'eta':0.03,'nrounds':700}]
    mlp_params=[{'hidden':[512,256],'lr':1e-3,'epochs':30},{'hidden':[256,128],'lr':5e-4,'epochs':40}]
    for mtype in model_types:
        for seed in seeds:
            if mtype=='lgb':
                for p in lgb_params: configs.append({'type':'lgb','seed':seed,'params':p})
            elif mtype=='xgb':
                for p in xgb_params: configs.append({'type':'xgb','seed':seed,'params':p})
            else:
                for p in mlp_params: configs.append({'type':'mlp','seed':seed,'params':p})
    return configs

MODEL_CONFIGS=generate_model_configs()
print('Generated', len(MODEL_CONFIGS), 'model configs')

# ---------------- OOF LightGBM ----------------
def run_oof_lightgbm(X,Y,n_splits=5,params=None):
    n_targets=Y.shape[1]
    oof_mu=np.zeros_like(Y); oof_sigma=np.zeros_like(Y)
    kf=KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    for fold,(tr_idx,val_idx) in enumerate(kf.split(X)):
        X_tr,X_val=X[tr_idx],X[val_idx]
        Y_tr,Y_val=Y[tr_idx],Y[val_idx]
        
        for t in range(n_targets):
            y_tr=Y_tr[:,t]; y_val=Y_val[:,t]
            
            if lgb is not None:
                dtrain=lgb.Dataset(X_tr, label=y_tr)
                dval=lgb.Dataset(X_val, label=y_val, reference=dtrain)
                param=params or {'objective':'regression','metric':'rmse','verbosity':-1}
                
                bst=lgb.train(param, dtrain, num_boost_round=500,
                              valid_sets=[dval],
                              callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
                
                oof_mu[val_idx,t]=bst.predict(X_val)
                resid=y_tr-bst.predict(X_tr)
                oof_sigma[val_idx,t]=np.std(resid)
            else:
                oof_mu[val_idx,t]=y_tr.mean()
                oof_sigma[val_idx,t]=y_tr.std()
                
        print(f'Fold {fold} done')
    return oof_mu,oof_sigma

# ---------------- Ensemble combine ----------------
def combine_ensemble_predictions(mu_list, sigma_list):
    mus=np.stack(mu_list, axis=0)
    sigs=np.stack(sigma_list, axis=0)
    mu_mean=mus.mean(axis=0)
    var_models=mus.var(axis=0)
    aleatoric=(sigs**2).mean(axis=0)
    sigma_ensemble=np.sqrt(aleatoric+var_models)
    return mu_mean, sigma_ensemble

# ---------------- Main pipeline ----------------
def main():
    # ---------------- Load train ----------------
    train_path = next(DATA_DIR.glob('**/train.csv'))
    train_df=pd.read_csv(train_path)
    planet_ids=train_df['planet_id'].values
    y_train=train_df.drop(columns=['planet_id']).values

    # ---------------- Feature prep placeholder ----------------
    # TODO: add spectral loading + preprocessing + feature extraction
    X_train=np.random.randn(len(y_train), 100)  # dummy features, replace with real spectral processing

    # ---------------- Ensemble OOF ----------------
    mu_list=[]; sigma_list=[]
    for cfg in MODEL_CONFIGS[:3]:  # test küçük subset
        if cfg['type']=='lgb':
            mu,sigma=run_oof_lightgbm(X_train, y_train, params=cfg['params'])
        else:
            mu=np.zeros_like(y_train); sigma=np.ones_like(y_train)  # placeholder
        mu_list.append(mu); sigma_list.append(sigma)

    mu_ens, sigma_ens=combine_ensemble_predictions(mu_list, sigma_list)

    # ---------------- Submission ----------------
    sample_submission_path = next(DATA_DIR.glob('**/sample_submission.csv'))
    submission_df=pd.read_csv(sample_submission_path)
    test_ids=submission_df['planet_id'].values
    write_submission(test_ids, mu_ens[:len(test_ids)], sigma_ens[:len(test_ids)], OUT_DIR)

if __name__=='__main__':
    main()


