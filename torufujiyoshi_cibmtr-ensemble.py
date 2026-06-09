# !pip install lifelines -q --no-index --find-links=/kaggle/input/cibmtr2024-import/lifelines
# !pip install scikit-learn==1.4.0 -q --no-index --find-links=/kaggle/input/cibmtr2024-import/scikit_learn
# !pip install rtdl_num_embeddings -q --no-index --find-links=/kaggle/input/cibmtr2024-import/rtdl_num_embeddings
# !pip install delu -q --no-index --find-links=/kaggle/input/cibmtr2024-import/delu


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl
!pip install /kaggle/input/download-lightning-and-pytorch-tabular/pytorch_lightning-2.4.0-py3-none-any.whl
!pip install /kaggle/input/download-lightning-and-pytorch-tabular/scikit_learn-1.6.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install /kaggle/input/download-lightning-and-pytorch-tabular/torchmetrics-1.5.2-py3-none-any.whl
!pip install /kaggle/input/download-lightning-and-pytorch-tabular/pytorch_tabnet-4.1.0-py3-none-any.whl
!pip install /kaggle/input/download-lightning-and-pytorch-tabular/einops-0.7.0-py3-none-any.whl
!pip install /kaggle/input/antlr43/antlr4_python3_runtime-4.9.3-py3-none-any.whl
!pip install --no-index --find-links=/kaggle/input/omegaconf3/pytorch-tabular-deps omegaconf
!pip install /kaggle/input/download-lightning-and-pytorch-tabular/pytorch_tabular-1.1.1-py2.py3-none-any.whl


# !pip install /kaggle/input/torch-geom8/torch_geom/torch_spline_conv-1.2.2+pt24cu121-cp310-cp310-linux_x86_64.whl
# !pip install /kaggle/input/torch-geom8/torch_geom/torch_sparse-0.6.18+pt24cu121-cp310-cp310-linux_x86_64.whl
# !pip install /kaggle/input/torch-geom8/torch_geom/pyg_lib-0.4.0+pt24cu121-cp310-cp310-linux_x86_64.whl
# !pip install /kaggle/input/torch-geom8/torch_geom/torch_cluster-1.6.3+pt24cu121-cp310-cp310-linux_x86_64.whl
# !pip install /kaggle/input/torch-geom8/torch_geom/torch_scatter-2.1.2+pt24cu121-cp310-cp310-linux_x86_64.whl
# !pip install /kaggle/input/torch-geom7/torch_geom_packages/torch_geometric-2.6.1-py3-none-any.whl


# from tabm_reference import Model, make_parameter_groups

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
# import rtdl_num_embeddings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, root_mean_squared_error, roc_auc_score, root_mean_squared_log_error, mean_squared_log_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from IPython.display import clear_output

from metric import score
import warnings
warnings.filterwarnings('ignore')

import joblib
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset

# import delu
import math

from collections import OrderedDict
from tqdm import tqdm

import functools
from typing import List

import pytorch_lightning as pl
import numpy as np
import torch
from lifelines.utils import concordance_index
from pytorch_lightning.cli import ReduceLROnPlateau
from pytorch_tabular.models.common.layers import ODST
from torch import nn
from pytorch_lightning.utilities import grad_norm

import json

import pytorch_lightning as pl

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.callbacks import StochasticWeightAveraging
from sklearn.model_selection import StratifiedKFold
# from torch_geometric.nn import GCNConv



# edge_index=torch.load("/kaggle/input/edge-index1/edge_index.pth")


# train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
# test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')


# from lifelines import KaplanMeierFitter
# from scipy.stats import gamma
# def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
#     kmf = KaplanMeierFitter()
#     kmf.fit(df[time_col], df[event_col])
#     y = kmf.survival_function_at_times(df[time_col]).values
#     return y

# train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')

# # train["label"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')
# # train.loc[train['efs']==0, 'label'] -= 0.2


# # k, beta = 4, -0.5 #4,-0.5
# # train['label'] = 1 - gamma.cdf(train.efs_time / np.exp(-beta), k)

# train['label']=np.log(1 + train.efs_time)

# train["efs_time2"] = train.efs_time.copy()
# train.loc[train.efs==0,"efs_time2"] *= -1


# combined = pd.concat([train, test], axis=0)

# RMV = ["ID","efs","efs_time", "label", "y", "efs_time2"]
# FEATURES = [c for c in train.columns if not c in RMV]
# print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


# CATS = []
# for c in FEATURES:
#     num_unique = combined[c].nunique()
#     if num_unique < 100:
#         CATS.append(c)
#         train[c] = train[c].fillna(999)
#         test[c] = test[c].fillna(999)
# print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")

# NUMS = [c for c in FEATURES if not c in CATS]


# combined = pd.concat([train,test],axis=0,ignore_index=True)
# #print("Combined data shape:", combined.shape )

# # LABEL ENCODE CATEGORICAL FEATURES
# print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
# for c in FEATURES:

#     # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
#     if c in CATS:
#         print(f"{c}, ",end="")
#         combined[c],_ = combined[c].factorize()
#         combined[c] -= combined[c].min()
#         combined[c] = combined[c].astype("int32")
#         combined[c] = combined[c].astype("category")
        
#     # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
#     else:
#         if combined[c].dtype=="float64":
#             combined[c] = combined[c].astype("float32")
#         if combined[c].dtype=="int64":
#             combined[c] = combined[c].astype("int32")
    


# cat_unique = combined[CATS].nunique().to_list()

# # for c in NUMS:
# #     combined[c] = combined[c].fillna(combined[c].mean())
# imputer = SimpleImputer(strategy='mean', add_indicator=True)

# # NUMSãƒªã‚¹ãƒˆã�®ã‚³ãƒ”ãƒ¼ã�§ãƒ«ãƒ¼ãƒ—å‡¦ç�†ã‚’è¡Œã�†
# for c in NUMS[:]:  # ã�¾ã�Ÿã�¯ NUMS.copy()
#     imputed_data = imputer.fit_transform(combined[[c]])
#     combined[c] = imputed_data[:, 0]  # å…ƒã�®æ•°å€¤ã‚«ãƒ©ãƒ ã‚’æ›´æ–°
#     if imputed_data.shape[1] > 1:  # indicator column ã�Œè¿½åŠ ã�•ã‚Œã�Ÿå ´å�ˆã�®ã�¿
#         combined[f'{c}_æ¬ æ��ãƒ•ãƒ©ã‚°'] = imputed_data[:, 1]
#         NUMS.append(f'{c}_æ¬ æ��ãƒ•ãƒ©ã‚°')

# train = combined.iloc[:len(train)].copy()
# test = combined.iloc[len(train):].reset_index(drop=True).copy()


# cats_index = [train[FEATURES].columns.get_loc(cat) for cat in CATS]


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# train[NUMS] = scaler.fit_transform(train[NUMS])
# test[NUMS] = scaler.transform(test[NUMS])


# folds = 5
# train['kfold'] = -1  

# target = 'label'
# kf = KFold(n_splits=5, random_state=42, shuffle=True)
# groups = train['efs'].astype(str)
# for fold, (train_idx, val_idx) in enumerate(kf.split(X=train)):
#     train.loc[val_idx, 'kfold'] = fold

# oof_metric = train[['kfold','ID','efs','efs_time','label','race_group']].copy()
# oof_metric['prediction'] = 0.0

# oof_tabm = np.zeros(train.shape[0])
# test_tabm = np.zeros((5, test.shape[0]))


# X_num = train[NUMS].values
# X_cat = train[CATS].values

# X_num_test = test[NUMS].values
# X_cat_test = test[CATS].values

# y = train[target].values


# test_dl = DataLoader(TensorDataset(torch.tensor(X_num_test, dtype=torch.float32), torch.tensor(X_cat_test, dtype=torch.int64)), batch_size=1024, shuffle=False)


# n_cont_features = len(NUMS)
# n_cat_features = len(CATS)
# n_classes = None
# cat_cardinalities = cat_unique

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# # TabM
# # arch_type = 'tabm'
# # bins = None

# # TabM-mini with the piecewise-linear embeddings.
# arch_type = 'tabm-mini'

# class RMSELoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss()
        
#     def forward(self, y_pred, y_true):
#         return torch.sqrt(self.mse(y_pred, y_true))

# loss_fn = RMSELoss()




# val_rmse_scores = []

# val_cindex_scores = []    
# for i, (train_index, val_index) in enumerate(kf.split(train[FEATURES])):
#     best = {
#         "val": -math.inf,
#         "epoch": -1,
#     }
#     ds_true = oof_metric.loc[oof_metric.kfold==i, ["ID","efs","efs_time","race_group"]].copy().reset_index(drop=True)
#     ds_pred = oof_metric.loc[oof_metric.kfold==i, ["ID"]].copy().reset_index(drop=True)

#     X_num_train = X_num[train_index]
#     X_cat_train = X_cat[train_index]
#     y_train = y[train_index]

#     X_num_val = X_num[val_index]
#     X_cat_val = X_cat[val_index]
#     y_val_all = y[val_index]

#     train_dl = DataLoader(TensorDataset(torch.tensor(X_num_train, dtype=torch.float32), torch.tensor(X_cat_train, dtype=torch.int64), 
#                                         torch.tensor(y_train, dtype=torch.float32)), batch_size=32, shuffle=True)
#     valid_dl = DataLoader(TensorDataset(torch.tensor(X_num_val, dtype=torch.float32), torch.tensor(X_cat_val, dtype=torch.int64), 
#                                         torch.tensor(y_val_all, dtype=torch.float32)), batch_size=32, shuffle=False)
    
#     bins = rtdl_num_embeddings.compute_bins(torch.tensor(X_num_train, dtype=torch.float32))

#     model = Model(
#         n_num_features=n_cont_features,
#         cat_cardinalities=cat_cardinalities,
#         n_classes=n_classes,
#         backbone={
#             'type': 'MLP',
#             'n_blocks': 3 ,
#             'd_block': 512,
#             'dropout': 0.1,
#         },
#         bins=bins,
#         num_embeddings=(
#             None
#             if bins is None
#             else {
#                 'type': 'PiecewiseLinearEmbeddings',
#                 'd_embedding': 64,
#                 'activation': True,
#                 'version': 'B',
#             }
#         ),
#         arch_type=arch_type,
#         k=32,
#     ).to(device)

#     optimizer = torch.optim.AdamW(
#         # Instead of model.parameters(),
#         make_parameter_groups(model),
#         lr=1e-4,
#         weight_decay=1e-3 ,
#     )
    
#     patience = 15
#     early_stopping = delu.tools.EarlyStopping(patience, mode="max")

#     for epoch in range(100):
#         model.train()   
#         with tqdm(train_dl, total=len(train_dl), leave=True) as phar :
#             for train_tensor in phar:
#                 optimizer.zero_grad()
#                 X_num_train, X_cat_train, y_train = [t.to(device) for t in train_tensor]

#                 output = model(X_num_train, X_cat_train).squeeze(-1)
#                 loss = loss_fn(output.flatten(0, 1), y_train.repeat_interleave(32))
#                 loss.backward()
#                 optimizer.step()

#                 phar.set_postfix(
#                     OrderedDict(
#                         epoch=f'{epoch+1}/{100}',
#                         loss=f'{loss.item():.6f}'
#                     )
#                 )
#                 phar.update(1)

        
#         model.eval()
#         valid_pred_list = []
#         for valid_tensor in valid_dl:
#             X_num_val, X_cat_val, y_val = [t.to(device) for t in valid_tensor]
#             with torch.no_grad():
#                 output = model(X_num_val, X_cat_val).squeeze(-1)
#             valid_pred_list.append((output.mean(1).cpu().numpy(), y_val.cpu().numpy()))
        
#         valid_pred = np.concatenate([p[0] for p in valid_pred_list])
#         valid_true = np.concatenate([p[1] for p in valid_pred_list])
#         val_loss = loss_fn(torch.tensor(valid_pred), torch.tensor(valid_true)).item()

#         ds_pred["prediction"] = valid_pred
#         val_cindex = score(ds_true.copy(), ds_pred.copy(), "ID")
        
#         if val_cindex > best["val"]:
#             print("ğŸŒ¸ New best epoch! ğŸŒ¸ with cindex: ", val_cindex)
#             best = {
#                 "val": val_cindex,
#                 "epoch": epoch,
#                 'pred' : valid_pred,
#             }

#         early_stopping.update(val_cindex)
#         if early_stopping.should_stop():
#             print("Early stopping")
#             break
    
#     oof_tabm[val_index] = best['pred']
#     val_rmse = root_mean_squared_error(y_val_all, best['pred'])
#     val_rmse_scores.append(val_rmse)

#     ds_pred["prediction"] = best['pred']
#     val_cindex = score(ds_true.copy(), ds_pred.copy(), "ID")

#     val_cindex_scores.append(val_cindex)

#     # predict test
#     model.eval()
#     test_pred_list = []
#     with torch.no_grad():
#         for test_tensor in test_dl:
#             X_num_test, X_cat_test = [t.to(device) for t in test_tensor]
#             output = model(X_num_test, X_cat_test).squeeze(-1)
#             test_pred_list.append(output.mean(1).cpu().numpy())
    
#     test_pred = np.concatenate([p for p in test_pred_list])
#     test_tabm[i] = test_pred
    
#     print(" *************************************************************************************** ")
#     print(f"Fold {i+1} RMSE: {val_rmse:.6f}", f"Fold {i+1} C-Index: {val_cindex:.6f}")
#     print("\n")
#     print(" *************************************************************************************** ")



# print("Mean Validation RMSE: {:.6f}".format(np.mean(val_rmse_scores)))
# print("Mean Validation C-Index: {:.6f}".format( np.mean(val_cindex_scores)))
# print("OOF RMSE: {:.6f}".format(root_mean_squared_error(train[target], oof_tabm)))

# results_df = pd.DataFrame({
#         'Fold': np.arange(1, 5+1),
#         'Validation RMSE': val_rmse_scores,
#         'Validation C-Index': val_cindex_scores
#     })


# print("\n=== KFold RMSE Results ===")
# print(results_df)

# y_true = train[["ID","efs","efs_time","race_group"]].copy()
# y_pred = train[["ID"]].copy()
# y_pred["prediction"] = -oof_tabm
# m = score(y_true.copy(), y_pred.copy(), "ID")
# print(f"\nOverall CV for TabM KaplanMeier =",m)


# test_mean = np.mean( test_tabm , axis=0) 





# !pip install --no-cache-dir --force-reinstall "/kaggle/input/scikitlearn1-6-1-for3-10-all/whl_file2/scikit_learn-1.6.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" --no-deps


# !python -c "import sklearn; print(sklearn.__version__)"


import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from torch.utils.data import TensorDataset
from scipy.stats import gamma
from scipy.stats import boxcox


def transform_target(train):
    # kmf = KaplanMeierFitter()
    # kmf.fit(durations=train['efs_time'], event_observed=train['efs'])
    # train['y'] = kmf.survival_function_at_times(train['efs_time']).values
    # train.loc[train.efs==0,'efs_time']=2*train.loc[train.efs==0,'efs_time']
    naf = NelsonAalenFitter()
    naf.fit(durations=train['efs_time'], event_observed=train['efs'])
    train['y'] = naf.cumulative_hazard_at_times(train['efs_time']).values
    train['y_bin'] = pd.qcut(train['y'], q=10, labels=False)
    kf = StratifiedKFold(n_splits=5, shuffle=True, )
    for i, (train_index, test_index) in enumerate(
    kf.split(
        train, train.race_group.astype(str) + train_original['y_bin'].astype(str)
    )
):
        naf = NelsonAalenFitter()
        naf.fit(durations=train.loc[train_index,'efs_time'], event_observed=train.loc[train_index,'efs'])
        train.loc[test_index,'y'] = naf.cumulative_hazard_at_times(train.loc[test_index,'efs_time']).values
    # train['y'] = train['y'] * 1
    # train['y']=train.efs_time
    # train.loc[train.efs_time==0,'y']=2*train.loc[train.efs_time==0,'y']
    # train['y'] = np.log(1+train.efs_time)
    # train['y'], lam = boxcox(train['y'])
    # mu = train.efs_time.mean()
    # sigma = train.efs_time.std()
    
    # train["y"] = 1 / (1 + np.exp(-(train['efs_time'] - mu) / sigma))
    # train['y'] = np.arcsinh(train['efs_time'])
    # train['y'] = np.sqrt(train['efs_time'])
    # train['y'], lam = boxcox(train['efs_time'])
    # k, beta = 4, -0.5 #4,0.5
    # train['y'] = 1 - gamma.cdf(train.efs_time / np.exp(-beta), k)
    # train['y']=1000*train['y']
    # train["y"] = train.efs_time.values
    # mx = train.loc[train.efs==1,"efs_time"].max()
    # mn = train.loc[train.efs==0,"efs_time"].min()
    # train.loc[train.efs==0,"y"] = train.loc[train.efs==0,"y"] + mx - mn
    # train.y = train.y.rank()
    # train.loc[train.efs==0,"y"] += 2*len(train)
    # train.y = train.y / train.y.max()
    # train.y = np.log( train.y )
    # train.y -= train.y.mean()
    # train.y *= -1.0
    return train


def get_X_cat(df, cat_cols, transformers=None):
    if transformers is None:
        transformers = [LabelEncoder().fit(df[col]) for col in cat_cols]
    return transformers, np.array(
        [transformer.transform(df[col]) for col, transformer in zip(cat_cols, transformers)]
    ).T


def preprocess_data(train, val):
    X_cat_train, X_cat_val, numerical, transformers = get_categoricals(train, val)
    scaler = StandardScaler()
    imp = SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=True)
    X_num_train = imp.fit_transform(train[numerical])
    X_num_train = scaler.fit_transform(X_num_train)
    X_num_val = imp.transform(val[numerical])
    X_num_val = scaler.transform(X_num_val)
    dl_train = init_dl(X_cat_train, X_num_train, train, training=True)
    dl_val = init_dl(X_cat_val, X_num_val, val)
    return X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers


def get_categoricals(train, val):
    categorical_cols, numerical = get_feature_types(train)
    remove = []
    for col in categorical_cols:
        if train[col].nunique() == 1:
            remove.append(col)
        ind = ~val[col].isin(train[col])
        if ind.any():
            val.loc[ind, col] = np.nan
    categorical_cols = [col for col in categorical_cols if col not in remove]
    transformers, X_cat_train = get_X_cat(train, categorical_cols)
    _, X_cat_val = get_X_cat(val, categorical_cols, transformers)
    return X_cat_train, X_cat_val, numerical, transformers


def init_dl(X_cat, X_num, df, training=False):
    ds_train = TensorDataset(
        torch.tensor(X_cat, dtype=torch.long),
        torch.tensor(X_num, dtype=torch.float32),
        torch.tensor(df.y.values, dtype=torch.float32),
        torch.tensor(df.efs.values, dtype=torch.long)
    )
    bs = 2048#2048
    # if not training:
    #     bs = 2048 * 8
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=bs, pin_memory=True, shuffle=training)
    return dl_train


def get_feature_types(train):
    RMV = ["ID", "efs", "efs_time", "y","y_bin"]
    FEATURES = [c for c in train.columns if not c in RMV]
    categorical_cols = [col for i, col in enumerate(FEATURES) if ((train[col].dtype == "object") | (2 < train[col].nunique() < 25))]
    print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
    numerical = [i for i in FEATURES if i not in categorical_cols]
    return categorical_cols, numerical


def add_features(df):
    sex_match = df.sex_match.astype(str)
    sex_match = sex_match.str.split("-").str[0] == sex_match.str.split("-").str[1]
    df['sex_match_bool'] = sex_match
    df.loc[df.sex_match.isna(), 'sex_match_bool'] = np.nan
    df['big_age'] = df.age_at_hct > 16
    df.loc[df.year_hct == 2019, 'year_hct'] = 2020
    df['is_cyto_score_same'] = (df['cyto_score'] == df['cyto_score_detail']).astype(int)
    df['strange_age'] = df.age_at_hct == 0.044
    df['age_bin'] = pd.cut(df.age_at_hct, [0, 0.0441, 16, 30, 50, 100])
    df['age_ts'] = df.age_at_hct / df.donor_age
    return df


def load_data():
    test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
    test = add_features(test)
    print("Test shape:", test.shape)
    train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
    train = add_features(train)
    print("Train shape:", train.shape)
    return test, train



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.models.layers import DropPath  # Stochastic Depth

# class FTTransformer(nn.Module):
#     def __init__(self, projection_dim, continuous_dim, num_heads=8, num_layers=3, dropout=0.1):
#         super().__init__()

#         self.tokenizer = nn.Linear(projection_dim + continuous_dim, projection_dim)  # ãƒˆãƒ¼ã‚¯ãƒ³åŒ–

#         self.transformer_blocks = nn.Sequential(
#             *[TransformerBlock(projection_dim, num_heads, dropout) for _ in range(num_layers)]
#         )

#         self.head = nn.Sequential(
#             nn.LayerNorm(projection_dim),
#             nn.Linear(projection_dim,112)
#         )

#     def forward(self, x):
#         x = self.tokenizer(x)
#         x = self.transformer_blocks(x)
#         return self.head(x)

# class TransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads, dropout):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
#         self.drop_path = DropPath(dropout)  # Stochastic Depth
#         self.norm2 = nn.LayerNorm(dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, 4 * dim),
#             nn.GELU(),
#             nn.Linear(4 * dim, dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
#         x = x + self.drop_path(self.ffn(self.norm2(x)))
#         return x



# class CatEmbeddings(nn.Module):
#     def __init__(
#         self,
#         projection_dim: int,
#         categorical_cardinality: List[int],
#         embedding_dim: int
#     ):
#         super(CatEmbeddings, self).__init__()
#         self.embeddings = nn.ModuleList([
#             nn.Embedding(cardinality, embedding_dim)
#             for cardinality in categorical_cardinality
#         ])
#         self.projection = nn.Sequential(
#             nn.Linear(embedding_dim * len(categorical_cardinality), projection_dim),
#             nn.GELU(),
#             nn.Linear(projection_dim, projection_dim)
#         )

#     def forward(self, x_cat):
#         x_cat = [embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)]
#         x_cat = torch.cat(x_cat, dim=1)
#         return self.projection(x_cat)
# class GNNLayer(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(GNNLayer, self).__init__()
#         self.conv = GCNConv(input_dim, hidden_dim)
        
#     def forward(self, x, edge_index):
#         return self.conv(x, edge_index)


# class NN(nn.Module):
#     def __init__(
#             self,
#             continuous_dim: int,
#             categorical_cardinality: List[int],
#             embedding_dim: int,
#             projection_dim: int,
#             hidden_dim: int,
#             dropout: float = 0.0 #0.0
#     ):
#         super(NN, self).__init__()
#         self.embeddings = CatEmbeddings(projection_dim, categorical_cardinality, embedding_dim)
#         self.mlp = nn.Sequential(
#             ODST(projection_dim + continuous_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Dropout(dropout)
#         )
#         # self.mlp = nn.Sequential(
#         #     nn.Linear(projection_dim + continuous_dim, hidden_dim * 2),  # GLUã�®ã�Ÿã‚�ã�«æ¬¡å…ƒã‚’hidden_dim * 2ã�«å¤‰æ�›
#         #     nn.GLU(),  # GLUã‚’é�©ç”¨
#         #     nn.BatchNorm1d(hidden_dim),  # ãƒ�ãƒƒãƒ�æ­£è¦�åŒ–
#         #     nn.Dropout(dropout)# ãƒ‰ãƒ­ãƒƒãƒ—ã‚¢ã‚¦ãƒˆ
#         # )
#         self.out = nn.Linear(hidden_dim, 1)
#         self.dropout = nn.Dropout(dropout)
#         self.ft_transformer = FTTransformer(
#             projection_dim,continuous_dim
#         )

#         # initialize weights
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 nn.init.zeros_(m.bias)

#     def forward(self, x_cat, x_cont):
#         x = self.embeddings(x_cat)
#         x = torch.cat([x, x_cont], dim=1)
#         x = self.dropout(x)
#         # x = self.mlp(x)  # GNNLayer ã‚’æ˜�ç¤ºçš„ã�«é�©ç”¨
#         # x=self.deepfm(x)
#         x = x.unsqueeze(1)  # (batch, feature_dim) â†’ (batch, seq_len=1, feature_dim)
#         x = self.ft_transformer(x)  # FT-Transformer ã‚’é�©ç”¨
#         x = x.squeeze(1)  
#         return self.out(x), x


# @functools.lru_cache
# def combinations(N):
#     ind = torch.arange(N)
#     comb = torch.combinations(ind, r=2)
#     return comb.cuda()


# class LitNN(pl.LightningModule):
#     def __init__(
#             self,
#             continuous_dim: int,
#             categorical_cardinality: List[int],
#             embedding_dim: int,
#             projection_dim: int,
#             hidden_dim: int,
#             lr: float = 1e-3,
#             dropout: float = 0.2,
#             weight_decay: float = 1e-3,
#             aux_weight: float = 0.1,#0.1
#             margin: float = 0.5,
#             race_index: int = 0
#     ):
#         super(LitNN, self).__init__()
#         self.save_hyperparameters()
#         self.model = NN(
#             continuous_dim=self.hparams.continuous_dim,
#             categorical_cardinality=self.hparams.categorical_cardinality,
#             embedding_dim=self.hparams.embedding_dim,
#             projection_dim=self.hparams.projection_dim,
#             hidden_dim=self.hparams.hidden_dim,
#             dropout=self.hparams.dropout
#         )
#         self.targets = []
#         self.aux_cls = nn.Sequential(
#             nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim // 3),
#             nn.GELU(),
#             nn.Linear(self.hparams.hidden_dim // 3, 1)
#         )

#     def on_before_optimizer_step(self, optimizer):
#         # Compute the 2-norm for each layer
#         # If using mixed precision, the gradients are already unscaled here
#         norms = grad_norm(self.model, norm_type=2)
#         self.log_dict(norms)

#     def forward(self, x_cat, x_cont):
#         x, emb = self.model(x_cat, x_cont)
#         return x.squeeze(1), emb

#     def training_step(self, batch, batch_idx):
#         x_cat, x_cont, y, efs = batch
#         y_hat, emb = self(x_cat, x_cont)
#         aux_pred = self.aux_cls(emb).squeeze(1)
#         loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
#         aux_loss = nn.functional.mse_loss(aux_pred, y, reduction='none')
#         # aux_loss=self.calc_loss(y,aux_pred,efs)
#         aux_mask = efs == 1
#         aux_loss = (aux_loss * aux_mask).sum() / aux_mask.sum()
#         self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
#         self.log("race_loss", race_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
#         self.log("aux_loss", aux_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
#         return loss + aux_loss * self.hparams.aux_weight

#     def get_full_loss(self, efs, x_cat, y, y_hat):
#         loss = self.calc_loss(y, y_hat, efs)
#         race_loss = self.get_race_losses(efs, x_cat, y, y_hat)
#         loss += 0.1*race_loss
#         return loss, race_loss

#     def get_race_losses(self, efs, x_cat, y, y_hat):
#         races = torch.unique(x_cat[:, self.hparams.race_index])
#         race_losses = []
#         for race in races:
#             ind = x_cat[:, self.hparams.race_index] == race
#             race_losses.append(self.calc_loss(y[ind], y_hat[ind], efs[ind]))
#         race_loss = sum(race_losses) / len(race_losses)
#         races_loss_std = sum((r - race_loss)**2 for r in race_losses) / len(race_losses)
#         return torch.sqrt(races_loss_std)

#     def calc_loss(self, y, y_hat, efs):
#         N = y.shape[0]
#         comb = combinations(N)
#         comb = comb[(efs[comb[:, 0]] == 1) | (efs[comb[:, 1]] == 1)]
#         pred_left = y_hat[comb[:, 0]]
#         pred_right = y_hat[comb[:, 1]]
#         y_left = y[comb[:, 0]]
#         y_right = y[comb[:, 1]]
#         y = 2 * (y_left > y_right).int() - 1
#         loss = nn.functional.relu(-y *(pred_left - pred_right) + self.hparams.margin)
#         mask = self.get_mask(comb, efs, y_left, y_right)
#         loss = (loss.double() * (mask.double())).sum() / mask.sum()
#         return loss

#     def get_mask(self, comb, efs, y_left, y_right):
#         # mask1 = (efs[comb[:, 0]] == 1) | (efs[comb[:, 1]] == 1)
#         left_outlived = y_left >= y_right
#         left_1_right_0 = (efs[comb[:, 0]] == 1) & (efs[comb[:, 1]] == 0)
#         mask2 = (left_outlived & left_1_right_0)
#         right_outlived = y_right >= y_left
#         right_1_left_0 = (efs[comb[:, 1]] == 1) & (efs[comb[:, 0]] == 0)
#         mask2 |= (right_outlived & right_1_left_0)
#         mask2 = ~mask2
#         mask = mask2
#         return mask

#     def validation_step(self, batch, batch_idx):
#         x_cat, x_cont, y, efs = batch
#         y_hat, emb = self(x_cat, x_cont)
#         loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
#         self.targets.append([y, y_hat.detach(), efs, x_cat[:, self.hparams.race_index]])
#         self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
#         return loss

#     def on_validation_epoch_end(self):
#         cindex, metric = self._calc_cindex()
#         self.log("cindex", metric, on_epoch=True, prog_bar=True, logger=True)
#         self.log("cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
#         self.targets.clear()

#     def _calc_cindex(self):
#         y = torch.cat([t[0] for t in self.targets]).cpu().numpy()
#         y_hat = torch.cat([t[1] for t in self.targets]).cpu().numpy()
#         efs = torch.cat([t[2] for t in self.targets]).cpu().numpy()
#         races = torch.cat([t[3] for t in self.targets]).cpu().numpy()
#         metric = self._metric(efs, races, y, y_hat)
#         cindex = concordance_index(y, y_hat, efs)
#         return cindex, metric

#     def _metric(self, efs, races, y, y_hat):
#         metric_list = []
#         for race in np.unique(races):
#             y_ = y[races == race]
#             y_hat_ = y_hat[races == race]
#             efs_ = efs[races == race]
#             metric_list.append(concordance_index(y_, y_hat_, efs_))
#         metric = float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))
#         return metric

#     def test_step(self, batch, batch_idx):
#         x_cat, x_cont, y, efs = batch
#         y_hat, emb = self(x_cat, x_cont)
#         loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
#         self.targets.append([y, y_hat.detach(), efs, x_cat[:, self.hparams.race_index]])
#         self.log("test_loss", loss)
#         return loss

#     def on_test_epoch_end(self) -> None:
#         cindex, metric = self._calc_cindex()
#         self.log("test_cindex", metric, on_epoch=True, prog_bar=True, logger=True)
#         self.log("test_cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
#         self.targets.clear()


#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
#         scheduler_config = {
#             "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
#                 optimizer,
#                 T_max=45,
#                 eta_min=6e-3
#             ),
#             "interval": "epoch",
#             "frequency": 1,
#             "strict": False,
#         }

#         return {"optimizer": optimizer, "lr_scheduler": scheduler_config}







pl.seed_everything(42)

def train_final(X_num_train, dl_train, dl_val, transformers, hparams=None, categorical_cols=None):
    if hparams is None:
        hparams = {
            "embedding_dim": 32,#16 32
            "projection_dim": 224,#112 224
            "hidden_dim": 112,#56 112
            "lr": 0.015,#0.06464861983337984
            "dropout": 0.05463240181423116,#0.05463240181423116
            "aux_weight": 1.7,# 0.26545778308743806 0.4 1.2
            "margin": 0.2588153271003354,#0.2588153271003354
            "weight_decay": 0.0002773544957610778 #0.0002773544957610778
        }
    model = LitNN(
        continuous_dim=X_num_train.shape[1],
        categorical_cardinality=[len(t.classes_) for t in transformers],
        race_index=categorical_cols.index("race_group"),
        **hparams
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1)
    trainer = pl.Trainer(
        accelerator='cuda',
        max_epochs=60,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='epoch'),
            TQDMProgressBar(),
            StochasticWeightAveraging(swa_lrs=1e-5, swa_epoch_start=45, annealing_epochs=15)
        ],
    )
    trainer.fit(model, dl_train)
    trainer.test(model, dl_val)
    return model.eval()


# from sklearn.model_selection import StratifiedKFold


# test, train_original = load_data()
# test['efs_time'] = 1
# test['efs'] = 1
# test_pred = np.zeros(test.shape[0])
# oof_nn=np.zeros(len(train_original))
# categorical_cols, numerical = get_feature_types(train_original)
# kf = StratifiedKFold(n_splits=5, shuffle=True, )
# for i, (train_index, test_index) in enumerate(
#     kf.split(
#         train_original, train_original.race_group.astype(str) + (train_original.age_at_hct == 0.044).astype(str)
#     )
# ):
#     tt = train_original.copy()
#     train = tt.iloc[train_index]
#     val = tt.iloc[test_index]
#     X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers = preprocess_data(train, val)
#     model = train_final(X_num_train, dl_train, dl_val, transformers, categorical_cols=categorical_cols)
#     # Create submission
#     train = tt.iloc[train_index]
#     X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers = preprocess_data(train, test)
#     pred, _ = model.cuda().eval()(
#         torch.tensor(X_cat_val, dtype=torch.long).cuda(),
#         torch.tensor(X_num_val, dtype=torch.float32).cuda()
#     )
#     test_pred += pred.detach().cpu().numpy()
#     X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers = preprocess_data(train, val)
#     pred2, _ = model.cuda().eval()(
#         torch.tensor(X_cat_val, dtype=torch.long).cuda(),
#         torch.tensor(X_num_val, dtype=torch.float32).cuda()
#     )
#     oof_nn[test_index]=pred2.detach().cpu().numpy()
# subm_data = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
# oof_nn=-oof_nn
# test_preds_nn = -test_pred

# display(subm_data.head())
    





class CatEmbeddings(nn.Module):
    def __init__(
        self,
        projection_dim: int,
        categorical_cardinality: List[int],
        embedding_dim: int
    ):
        super(CatEmbeddings, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality in categorical_cardinality
        ])
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * len(categorical_cardinality), projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x_cat):
        x_cat = [embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)]
        x_cat = torch.cat(x_cat, dim=1)
        return self.projection(x_cat)
class GNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNLayer, self).__init__()
        self.conv = GCNConv(input_dim, hidden_dim)
        
    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class NN(nn.Module):
    def __init__(
            self,
            continuous_dim: int,
            categorical_cardinality: List[int],
            embedding_dim: int,
            projection_dim: int,
            hidden_dim: int,
            dropout: float = 0.0 #0.0
    ):
        super(NN, self).__init__()
        self.embeddings = CatEmbeddings(projection_dim, categorical_cardinality, embedding_dim)
        self.mlp = nn.Sequential(
            ODST(projection_dim + continuous_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        # self.mlp = nn.Sequential(
        #     nn.Linear(projection_dim + continuous_dim, hidden_dim * 2),  # GLUã�®ã�Ÿã‚�ã�«æ¬¡å…ƒã‚’hidden_dim * 2ã�«å¤‰æ�›
        #     nn.GLU(),  # GLUã‚’é�©ç”¨
        #     nn.BatchNorm1d(hidden_dim),  # ãƒ�ãƒƒãƒ�æ­£è¦�åŒ–
        #     nn.Dropout(dropout)# ãƒ‰ãƒ­ãƒƒãƒ—ã‚¢ã‚¦ãƒˆ
        # )
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_cat, x_cont):
        x = self.embeddings(x_cat)
        x = torch.cat([x, x_cont], dim=1)
        x = self.dropout(x)
        x = self.mlp(x)  # GNNLayer ã‚’æ˜�ç¤ºçš„ã�«é�©ç”¨
        # x=self.deepfm(x)
        return self.out(x), x


@functools.lru_cache
def combinations(N):
    ind = torch.arange(N)
    comb = torch.combinations(ind, r=2)
    return comb.cuda()


class LitNN(pl.LightningModule):
    def __init__(
            self,
            continuous_dim: int,
            categorical_cardinality: List[int],
            embedding_dim: int,
            projection_dim: int,
            hidden_dim: int,
            lr: float = 1e-3,
            dropout: float = 0.2,
            weight_decay: float = 1e-3,
            aux_weight: float = 0.25,#0.1
            margin: float = 0.5,
            race_index: int = 0
    ):
        super(LitNN, self).__init__()
        self.save_hyperparameters()
        self.model = NN(
            continuous_dim=self.hparams.continuous_dim,
            categorical_cardinality=self.hparams.categorical_cardinality,
            embedding_dim=self.hparams.embedding_dim,
            projection_dim=self.hparams.projection_dim,
            hidden_dim=self.hparams.hidden_dim,
            dropout=self.hparams.dropout
        )
        self.targets = []
        self.aux_cls = nn.Sequential(
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim // 3),
            nn.GELU(),
            nn.Linear(self.hparams.hidden_dim // 3, 1)
        )

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def forward(self, x_cat, x_cont):
        x, emb = self.model(x_cat, x_cont)
        return x.squeeze(1), emb

    def training_step(self, batch, batch_idx):
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        aux_pred = self.aux_cls(emb).squeeze(1)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        aux_loss = nn.functional.mse_loss(aux_pred, y, reduction='none')
        # aux_loss=self.calc_loss(y,aux_pred,efs)
        aux_mask = efs == 1
        aux_loss = (aux_loss * aux_mask).sum() / aux_mask.sum()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("race_loss", race_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        self.log("aux_loss", aux_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        return loss + aux_loss * self.hparams.aux_weight

    def get_full_loss(self, efs, x_cat, y, y_hat):
        loss = self.calc_loss(y, y_hat, efs)
        race_loss = self.get_race_losses(efs, x_cat, y, y_hat)
        loss += 0.1 * race_loss
        return loss, race_loss

    def get_race_losses(self, efs, x_cat, y, y_hat):
        races = torch.unique(x_cat[:, self.hparams.race_index])
        race_losses = []
        for race in races:
            ind = x_cat[:, self.hparams.race_index] == race
            race_losses.append(self.calc_loss(y[ind], y_hat[ind], efs[ind]))
        race_loss = sum(race_losses) / len(race_losses)
        races_loss_std = sum((r - race_loss)**2 for r in race_losses) / len(race_losses)
        return torch.sqrt(races_loss_std)

    def calc_loss(self, y, y_hat, efs):
        N = y.shape[0]
        comb = combinations(N)
        comb = comb[(efs[comb[:, 0]] == 1) | (efs[comb[:, 1]] == 1)]
        pred_left = y_hat[comb[:, 0]]
        pred_right = y_hat[comb[:, 1]]
        y_left = y[comb[:, 0]]
        y_right = y[comb[:, 1]]
        y = 2 * (y_left > y_right).int() - 1
        z = 2 * (pred_left > pred_right).int() - 1
        loss = nn.functional.relu(-y *(pred_left - pred_right) + self.hparams.margin)
        mask = self.get_mask(comb, efs, y_left, y_right)
        loss = (loss.double() * (mask.double())).sum() / mask.sum()
        return loss

    def get_mask(self, comb, efs, y_left, y_right):
        # mask1 = (efs[comb[:, 0]] == 1) | (efs[comb[:, 1]] == 1)
        left_outlived = y_left >= y_right
        left_1_right_0 = (efs[comb[:, 0]] == 1) & (efs[comb[:, 1]] == 0)
        mask2 = (left_outlived & left_1_right_0)
        right_outlived = y_right >= y_left
        right_1_left_0 = (efs[comb[:, 1]] == 1) & (efs[comb[:, 0]] == 0)
        mask2 |= (right_outlived & right_1_left_0)
        mask2 = ~mask2
        mask = mask2
        return mask

    def validation_step(self, batch, batch_idx):
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        self.targets.append([y, y_hat.detach(), efs, x_cat[:, self.hparams.race_index]])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        cindex, metric = self._calc_cindex()
        self.log("cindex", metric, on_epoch=True, prog_bar=True, logger=True)
        self.log("cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
        self.targets.clear()

    def _calc_cindex(self):
        y = torch.cat([t[0] for t in self.targets]).cpu().numpy()
        y_hat = torch.cat([t[1] for t in self.targets]).cpu().numpy()
        efs = torch.cat([t[2] for t in self.targets]).cpu().numpy()
        races = torch.cat([t[3] for t in self.targets]).cpu().numpy()
        metric = self._metric(efs, races, y, y_hat)
        cindex = concordance_index(y, y_hat, efs)
        return cindex, metric

    def _metric(self, efs, races, y, y_hat):
        metric_list = []
        for race in np.unique(races):
            y_ = y[races == race]
            y_hat_ = y_hat[races == race]
            efs_ = efs[races == race]
            metric_list.append(concordance_index(y_, y_hat_, efs_))
        metric = float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))
        return metric

    def test_step(self, batch, batch_idx):
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        self.targets.append([y, y_hat.detach(), efs, x_cat[:, self.hparams.race_index]])
        self.log("test_loss", loss)
        return loss

    def on_test_epoch_end(self) -> None:
        cindex, metric = self._calc_cindex()
        self.log("test_cindex", metric, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
        self.targets.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=45,
                eta_min=6e-3
            ),
            "interval": "epoch",
            "frequency": 1,
            "strict": False,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}







# from sklearn.model_selection import StratifiedKFold


# test, train_original = load_data()
# train_original=transform_target(train_original)
# test['efs_time'] = 1
# test['efs'] = 1
# test['y']=1
# test_pred2 = np.zeros(test.shape[0])
# oof_nn2=np.zeros(len(train_original))
# categorical_cols, numerical = get_feature_types(train_original)
# kf = StratifiedKFold(n_splits=5, shuffle=True, )
# train_original['y_bin'] = pd.qcut(train_original['y'], q=10, labels=False)
# for i, (train_index, test_index) in enumerate(
#     kf.split(
#         train_original, train_original.race_group.astype(str) + train_original['y_bin'].astype(str)
#     )
# ):
#     tt = train_original.copy()
#     train = tt.iloc[train_index]
#     val = tt.iloc[test_index]
#     X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers = preprocess_data(train, val)
#     model = train_final(X_num_train, dl_train, dl_val, transformers, categorical_cols=categorical_cols)
#     # Create submission
#     train = tt.iloc[train_index]
#     X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers = preprocess_data(train, test)
#     pred, _ = model.cuda().eval()(
#         torch.tensor(X_cat_val, dtype=torch.long).cuda(),
#         torch.tensor(X_num_val, dtype=torch.float32).cuda()
#     )
#     test_pred2 += pred.detach().cpu().numpy()
#     X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers = preprocess_data(train, val)
#     pred2, _ = model.cuda().eval()(
#         torch.tensor(X_cat_val, dtype=torch.long).cuda(),
#         torch.tensor(X_num_val, dtype=torch.float32).cuda()
#     )
#     oof_nn2[test_index]=pred2.detach().cpu().numpy()
# subm_data = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
# oof_nn2=-oof_nn2
# test_preds_nn2 = -test_pred2

# display(subm_data.head())
    


import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from torch.utils.data import TensorDataset
from scipy.stats import gamma
from scipy.stats import boxcox


def transform_target(train):
    # kmf = KaplanMeierFitter()
    # kmf.fit(durations=train['efs_time'], event_observed=train['efs'])
    # train['y'] = kmf.survival_function_at_times(train['efs_time']).values
    # naf = NelsonAalenFitter()
    # naf.fit(durations=train['efs_time'], event_observed=train['efs'])
    # train['y'] = naf.cumulative_hazard_at_times(train['efs_time']).values
    # train['y']=train['efs_time']
    # train.loc[train.efs==0,'y']=2*train.loc[train.efs==0,'y']
    # train['y'] = train['y'] * 1
    # train['y']=train.efs_time
    # train.loc[train.efs_time==0,'y']=2*train.loc[train.efs_time==0,'y']
    # train['y'] = np.log(1+train.efs_time)
    # train['y'], lam = boxcox(train['y'])
    # mu = train.efs_time.mean()
    # sigma = train.efs_time.std()
    
    # train["y"] = 1 / (1 + np.exp(-(train['efs_time'] - mu) / sigma))
    # train['y'] = np.arcsinh(train['efs_time'])
    # train['y'] = np.sqrt(train['efs_time'])
    # train['y'], lam = boxcox(train['efs_time'])
    # k, beta = 4, -0.5 #4,0.5
    # train['y'] = 1 - gamma.cdf(train.efs_time / np.exp(-beta), k)
    # train['y']=1000*train['y']
    # train["y"] = train.efs_time.values
    # mx = train.loc[train.efs==1,"efs_time"].max()
    # mn = train.loc[train.efs==0,"efs_time"].min()
    # train.loc[train.efs==0,"y"] = train.loc[train.efs==0,"y"] + mx - mn
    # train.y = train.y.rank()
    # train.loc[train.efs==0,"y"] += 2*len(train)
    # train.y = train.y / train.y.max()
    # train.y = np.log( train.y )
    # train.y -= train.y.mean()
    # train.y *= -1.0
    train['y']=np.log(train['efs_time'])
    return train


def get_X_cat(df, cat_cols, transformers=None):
    if transformers is None:
        transformers = [LabelEncoder().fit(df[col]) for col in cat_cols]
    return transformers, np.array(
        [transformer.transform(df[col]) for col, transformer in zip(cat_cols, transformers)]
    ).T


def preprocess_data(train, val):
    X_cat_train, X_cat_val, numerical, transformers = get_categoricals(train, val)
    scaler = StandardScaler()
    imp = SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=True)
    X_num_train = imp.fit_transform(train[numerical])
    X_num_train = scaler.fit_transform(X_num_train)
    X_num_val = imp.transform(val[numerical])
    X_num_val = scaler.transform(X_num_val)
    dl_train = init_dl(X_cat_train, X_num_train, train, training=True)
    dl_val = init_dl(X_cat_val, X_num_val, val)
    return X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers


def get_categoricals(train, val):
    categorical_cols, numerical = get_feature_types(train)
    remove = []
    for col in categorical_cols:
        if train[col].nunique() == 1:
            remove.append(col)
        ind = ~val[col].isin(train[col])
        if ind.any():
            val.loc[ind, col] = np.nan
    categorical_cols = [col for col in categorical_cols if col not in remove]
    transformers, X_cat_train = get_X_cat(train, categorical_cols)
    _, X_cat_val = get_X_cat(val, categorical_cols, transformers)
    return X_cat_train, X_cat_val, numerical, transformers


def init_dl(X_cat, X_num, df, training=False):
    ds_train = TensorDataset(
        torch.tensor(X_cat, dtype=torch.long),
        torch.tensor(X_num, dtype=torch.float32),
        torch.tensor(df.efs_time.values, dtype=torch.float32).log(),
        torch.tensor(df.efs.values, dtype=torch.long)
    )
    bs = 2048#2048
    # if not training:
    #     bs = 2048 * 8
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=bs, pin_memory=True, shuffle=training)
    return dl_train


def get_feature_types(train):
    RMV = ["ID", "efs", "efs_time", "y","y_bin"]
    FEATURES = [c for c in train.columns if not c in RMV]
    categorical_cols = [col for i, col in enumerate(FEATURES) if ((train[col].dtype == "object") | (2 < train[col].nunique() < 25))]
    print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
    numerical = [i for i in FEATURES if i not in categorical_cols]
    return categorical_cols, numerical


def add_features(df):
    sex_match = df.sex_match.astype(str)
    sex_match = sex_match.str.split("-").str[0] == sex_match.str.split("-").str[1]
    df['sex_match_bool'] = sex_match
    df.loc[df.sex_match.isna(), 'sex_match_bool'] = np.nan
    df['big_age'] = df.age_at_hct > 16
    df.loc[df.year_hct == 2019, 'year_hct'] = 2020
    df['is_cyto_score_same'] = (df['cyto_score'] == df['cyto_score_detail']).astype(int)
    df['strange_age'] = df.age_at_hct == 0.044
    df['age_bin'] = pd.cut(df.age_at_hct, [0, 0.0441, 16, 30, 50, 100])
    df['age_ts'] = df.age_at_hct / df.donor_age
    return df


def load_data():
    test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
    test = add_features(test)
    print("Test shape:", test.shape)
    train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
    train = add_features(train)
    print("Train shape:", train.shape)
    return test, train






pl.seed_everything(42)

def train_final(X_num_train, dl_train, dl_val, transformers, hparams=None, categorical_cols=None):
    if hparams is None:
        hparams = {
            "embedding_dim": 32,#16 32
            "projection_dim": 224,#112 224
            "hidden_dim": 112,#56 112
            "lr": 0.06464861983337984,#0.06464861983337984
            "dropout": 0.05463240181423116,#0.05463240181423116
            "aux_weight": 0.4,# 0.26545778308743806 0.4
            "margin": 0.2588153271003354,#0.2588153271003354
            "weight_decay": 0.0002773544957610778 #0.0002773544957610778
        }
    model = LitNN(
        continuous_dim=X_num_train.shape[1],
        categorical_cardinality=[len(t.classes_) for t in transformers],
        race_index=categorical_cols.index("race_group"),
        **hparams
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1)
    trainer = pl.Trainer(
        accelerator='cuda',
        max_epochs=60,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='epoch'),
            TQDMProgressBar(),
            StochasticWeightAveraging(swa_lrs=1e-5, swa_epoch_start=45, annealing_epochs=15)
        ],
    )
    trainer.fit(model, dl_train)
    trainer.test(model, dl_val)
    return model.eval()


# from sklearn.model_selection import StratifiedKFold


# test, train_original = load_data()
# train_original=transform_target(train_original)
# test['efs_time'] = 1
# test['efs'] = 1
# test['y']=1
# test_pred3 = np.zeros(test.shape[0])
# oof_nn3=np.zeros(len(train_original))
# categorical_cols, numerical = get_feature_types(train_original)
# kf = StratifiedKFold(n_splits=5, shuffle=True, )
# for i, (train_index, test_index) in enumerate(
#     kf.split(
#         train_original, train_original.race_group.astype(str) + (train_original.age_at_hct == 0.044).astype(str)
#     )
# ):
#     tt = train_original.copy()
#     train = tt.iloc[train_index]
#     val = tt.iloc[test_index]
#     X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers = preprocess_data(train, val)
#     model = train_final(X_num_train, dl_train, dl_val, transformers, categorical_cols=categorical_cols)
#     # Create submission
#     train = tt.iloc[train_index]
#     X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers = preprocess_data(train, test)
#     pred, _ = model.cuda().eval()(
#         torch.tensor(X_cat_val, dtype=torch.long).cuda(),
#         torch.tensor(X_num_val, dtype=torch.float32).cuda()
#     )
#     test_pred3 += pred.detach().cpu().numpy()
#     X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers = preprocess_data(train, val)
#     pred2, _ = model.cuda().eval()(
#         torch.tensor(X_cat_val, dtype=torch.long).cuda(),
#         torch.tensor(X_num_val, dtype=torch.float32).cuda()
#     )
#     oof_nn3[test_index]=pred2.detach().cpu().numpy()
# subm_data = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
# oof_nn3=-oof_nn3
# test_preds_nn3 = -test_pred3

# display(subm_data.head())
    





class CatEmbeddings(nn.Module):
    def __init__(
        self,
        projection_dim: int,
        categorical_cardinality: List[int],
        embedding_dim: int
    ):
        super(CatEmbeddings, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality in categorical_cardinality
        ])
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * len(categorical_cardinality), projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x_cat):
        x_cat = [embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)]
        x_cat = torch.cat(x_cat, dim=1)
        return self.projection(x_cat)
class GNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNLayer, self).__init__()
        self.conv = GCNConv(input_dim, hidden_dim)
        
    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class NN(nn.Module):
    def __init__(
            self,
            continuous_dim: int,
            categorical_cardinality: List[int],
            embedding_dim: int,
            projection_dim: int,
            hidden_dim: int,
            dropout: float = 0.0 #0.0
    ):
        super(NN, self).__init__()
        self.embeddings = CatEmbeddings(projection_dim, categorical_cardinality, embedding_dim)
        self.mlp = nn.Sequential(
            ODST(projection_dim + continuous_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        # self.mlp = nn.Sequential(
        #     nn.Linear(projection_dim + continuous_dim, hidden_dim * 2),  # GLUã�®ã�Ÿã‚�ã�«æ¬¡å…ƒã‚’hidden_dim * 2ã�«å¤‰æ�›
        #     nn.GLU(),  # GLUã‚’é�©ç”¨
        #     nn.BatchNorm1d(hidden_dim),  # ãƒ�ãƒƒãƒ�æ­£è¦�åŒ–
        #     nn.Dropout(dropout)# ãƒ‰ãƒ­ãƒƒãƒ—ã‚¢ã‚¦ãƒˆ
        # )
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_cat, x_cont):
        x = self.embeddings(x_cat)
        x = torch.cat([x, x_cont], dim=1)
        x = self.dropout(x)
        x = self.mlp(x)  # GNNLayer ã‚’æ˜�ç¤ºçš„ã�«é�©ç”¨
        # x=self.deepfm(x)
        return self.out(x), x


@functools.lru_cache
def combinations(N):
    ind = torch.arange(N)
    comb = torch.combinations(ind, r=2)
    return comb.cuda()


class LitNN(pl.LightningModule):
    def __init__(
            self,
            continuous_dim: int,
            categorical_cardinality: List[int],
            embedding_dim: int,
            projection_dim: int,
            hidden_dim: int,
            lr: float = 1e-3,
            dropout: float = 0.2,
            weight_decay: float = 1e-3,
            aux_weight: float = 0.25,#0.1
            margin: float = 0.5,
            race_index: int = 0
    ):
        super(LitNN, self).__init__()
        self.save_hyperparameters()
        self.model = NN(
            continuous_dim=self.hparams.continuous_dim,
            categorical_cardinality=self.hparams.categorical_cardinality,
            embedding_dim=self.hparams.embedding_dim,
            projection_dim=self.hparams.projection_dim,
            hidden_dim=self.hparams.hidden_dim,
            dropout=self.hparams.dropout
        )
        self.targets = []
        self.aux_cls = nn.Sequential(
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim // 3),
            nn.GELU(),
            nn.Linear(self.hparams.hidden_dim // 3, 1)
        )

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def forward(self, x_cat, x_cont):
        x, emb = self.model(x_cat, x_cont)
        return x.squeeze(1), emb

    def training_step(self, batch, batch_idx):
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        aux_pred = self.aux_cls(emb).squeeze(1)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        aux_loss = nn.functional.mse_loss(aux_pred, y, reduction='none')
        # aux_loss=self.calc_loss(y,aux_pred,efs)
        aux_mask = efs == 1
        aux_loss = (aux_loss * aux_mask).sum() / aux_mask.sum()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("race_loss", race_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        self.log("aux_loss", aux_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        return loss + aux_loss * self.hparams.aux_weight

    def get_full_loss(self, efs, x_cat, y, y_hat):
        loss = self.calc_loss(y, y_hat, efs)
        race_loss = self.get_race_losses(efs, x_cat, y, y_hat)
        loss += 0.44264024279725767 * race_loss
        return loss, race_loss

    def get_race_losses(self, efs, x_cat, y, y_hat):
        races = torch.unique(x_cat[:, self.hparams.race_index])
        race_losses = []
        for race in races:
            ind = x_cat[:, self.hparams.race_index] == race
            race_losses.append(self.calc_loss(y[ind], y_hat[ind], efs[ind]))
        race_loss = sum(race_losses) / len(race_losses)
        races_loss_std = sum((r - race_loss)**2 for r in race_losses) / len(race_losses)
        return torch.sqrt(races_loss_std)

    def calc_loss(self, y, y_hat, efs):
        N = y.shape[0]
        comb = combinations(N)
        comb = comb[(efs[comb[:, 0]] == 1) | (efs[comb[:, 1]] == 1)]
        pred_left = y_hat[comb[:, 0]]
        pred_right = y_hat[comb[:, 1]]
        y_left = y[comb[:, 0]]
        y_right = y[comb[:, 1]]
        y = 2 * (y_left > y_right).int() - 1
        z = 2 * (pred_left > pred_right).int() - 1
        loss = nn.functional.relu(-y *(pred_left - pred_right) + self.hparams.margin)
        mask = self.get_mask(comb, efs, y_left, y_right)
        loss = (loss.double() * (mask.double())).sum() / mask.sum()
        return loss

    def get_mask(self, comb, efs, y_left, y_right):
        # mask1 = (efs[comb[:, 0]] == 1) | (efs[comb[:, 1]] == 1)
        left_outlived = y_left >= y_right
        left_1_right_0 = (efs[comb[:, 0]] == 1) & (efs[comb[:, 1]] == 0)
        mask2 = (left_outlived & left_1_right_0)
        right_outlived = y_right >= y_left
        right_1_left_0 = (efs[comb[:, 1]] == 1) & (efs[comb[:, 0]] == 0)
        mask2 |= (right_outlived & right_1_left_0)
        mask2 = ~mask2
        mask = mask2
        return mask

    def validation_step(self, batch, batch_idx):
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        self.targets.append([y, y_hat.detach(), efs, x_cat[:, self.hparams.race_index]])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        cindex, metric = self._calc_cindex()
        self.log("cindex", metric, on_epoch=True, prog_bar=True, logger=True)
        self.log("cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
        self.targets.clear()

    def _calc_cindex(self):
        y = torch.cat([t[0] for t in self.targets]).cpu().numpy()
        y_hat = torch.cat([t[1] for t in self.targets]).cpu().numpy()
        efs = torch.cat([t[2] for t in self.targets]).cpu().numpy()
        races = torch.cat([t[3] for t in self.targets]).cpu().numpy()
        metric = self._metric(efs, races, y, y_hat)
        cindex = concordance_index(y, y_hat, efs)
        return cindex, metric

    def _metric(self, efs, races, y, y_hat):
        metric_list = []
        for race in np.unique(races):
            y_ = y[races == race]
            y_hat_ = y_hat[races == race]
            efs_ = efs[races == race]
            metric_list.append(concordance_index(y_, y_hat_, efs_))
        metric = float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))
        return metric

    def test_step(self, batch, batch_idx):
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        self.targets.append([y, y_hat.detach(), efs, x_cat[:, self.hparams.race_index]])
        self.log("test_loss", loss)
        return loss

    def on_test_epoch_end(self) -> None:
        cindex, metric = self._calc_cindex()
        self.log("test_cindex", metric, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
        self.targets.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=35,
                eta_min=0.00445470411593173
            ),
            "interval": "epoch",
            "frequency": 1,
            "strict": False,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}




#{'embedding_dim': 33, 'projection_dim': 101, 'hidden_dim': 63, 'lr': 0.04087902783639046, 'dropout': 0.37451068439964463, 'aux_weight': 1.3015428400288374, 'margin': 0.8742364144882628, 'weight_decay': 0.008154710605190912, 'fairness_weight': 0.44264024279725767, 'scheduler_T_max': 35, 'scheduler_eta_min': 0.00445470411593173}


pl.seed_everything(42)

def train_final(X_num_train, dl_train, dl_val, transformers, hparams=None, categorical_cols=None):
    if hparams is None:
        hparams = {
            "embedding_dim": 33,#16 32
            "projection_dim": 101,#112 224
            "hidden_dim": 63,#56 112
            "lr": 0.04087902783639046,#0.06464861983337984
            "dropout": 0.37451068439964463,#0.05463240181423116
            "aux_weight": 1.3015428400288374,# 0.26545778308743806 0.4
            "margin": 0.8742364144882628,#0.2588153271003354
            "weight_decay": 0.008154710605190912 #0.0002773544957610778
        }
    model = LitNN(
        continuous_dim=X_num_train.shape[1],
        categorical_cardinality=[len(t.classes_) for t in transformers],
        race_index=categorical_cols.index("race_group"),
        **hparams
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1)
    trainer = pl.Trainer(
        accelerator='cuda',
        max_epochs=60,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='epoch'),
            TQDMProgressBar(),
            StochasticWeightAveraging(swa_lrs=1e-5, swa_epoch_start=45, annealing_epochs=15)
        ],
    )
    trainer.fit(model, dl_train)
    trainer.test(model, dl_val)
    return model.eval()


# from sklearn.model_selection import StratifiedKFold


# test, train_original = load_data()
# train_original=transform_target(train_original)
# test['efs_time'] = 1
# test['efs'] = 1
# test['y']=1
# test_pred4 = np.zeros(test.shape[0])
# oof_nn4=np.zeros(len(train_original))
# categorical_cols, numerical = get_feature_types(train_original)
# kf = StratifiedKFold(n_splits=5, shuffle=True, )
# for i, (train_index, test_index) in enumerate(
#     kf.split(
#         train_original, train_original.race_group.astype(str) + (train_original.age_at_hct == 0.044).astype(str)
#     )
# ):
#     tt = train_original.copy()
#     train = tt.iloc[train_index]
#     val = tt.iloc[test_index]
#     X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers = preprocess_data(train, val)
#     model = train_final(X_num_train, dl_train, dl_val, transformers, categorical_cols=categorical_cols)
#     # Create submission
#     train = tt.iloc[train_index]
#     X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers = preprocess_data(train, test)
#     pred, _ = model.cuda().eval()(
#         torch.tensor(X_cat_val, dtype=torch.long).cuda(),
#         torch.tensor(X_num_val, dtype=torch.float32).cuda()
#     )
#     test_pred4 += pred.detach().cpu().numpy()
#     X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers = preprocess_data(train, val)
#     pred2, _ = model.cuda().eval()(
#         torch.tensor(X_cat_val, dtype=torch.long).cuda(),
#         torch.tensor(X_num_val, dtype=torch.float32).cuda()
#     )
#     oof_nn4[test_index]=pred2.detach().cpu().numpy()
# subm_data = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
# oof_nn4=-oof_nn4
# test_preds_nn4 = -test_pred4

# display(subm_data.head())
    



"""
To evaluate the equitable prediction of transplant survival outcomes,
we use the concordance index (C-index) between a series of event
times and a predicted score across each race group.

It represents the global assessment of the model discrimination power:
this is the modelâ€™s ability to correctly provide a reliable ranking
of the survival times based on the individual risk scores.

The concordance index is a value between 0 and 1 where:

0.5 is the expected result from random predictions,
1.0 is perfect concordance (with no censoring, otherwise <1.0),
0.0 is perfect anti-concordance (with no censoring, otherwise >0.0)

"""

import pandas as pd
import pandas.api.types
import numpy as np
from lifelines.utils import concordance_index

class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_pred = {'prediction': {0: 1.0, 1: 0.0, 2: 1.0}}
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred.insert(0, row_id_column_name, range(len(y_pred)))
    >>> y_true = { 'efs': {0: 1.0, 1: 0.0, 2: 0.0}, 'efs_time': {0: 25.1234,1: 250.1234,2: 2500.1234}, 'race_group': {0: 'race_group_1', 1: 'race_group_1', 2: 'race_group_1'}}
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true.insert(0, row_id_column_name, range(len(y_true)))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.75
    """

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])
        metric_list.append(c_index_race)
    # return float(np.mean(metric_list)-np.sqrt(np.var(metric_list))),metric_list,merged_df_race_dict.keys()
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))




"""
To evaluate the equitable prediction of transplant survival outcomes,
we use the concordance index (C-index) between a series of event
times and a predicted score across each race group.

It represents the global assessment of the model discrimination power:
this is the modelâ€™s ability to correctly provide a reliable ranking
of the survival times based on the individual risk scores.

The concordance index is a value between 0 and 1 where:

0.5 is the expected result from random predictions,
1.0 is perfect concordance (with no censoring, otherwise <1.0),
0.0 is perfect anti-concordance (with no censoring, otherwise >0.0)

"""

import pandas as pd
import pandas.api.types
import numpy as np
from lifelines.utils import concordance_index

class ParticipantVisibleError(Exception):
    pass


def score2(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_pred = {'prediction': {0: 1.0, 1: 0.0, 2: 1.0}}
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred.insert(0, row_id_column_name, range(len(y_pred)))
    >>> y_true = { 'efs': {0: 1.0, 1: 0.0, 2: 0.0}, 'efs_time': {0: 25.1234,1: 250.1234,2: 2500.1234}, 'race_group': {0: 'race_group_1', 1: 'race_group_1', 2: 'race_group_1'}}
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true.insert(0, row_id_column_name, range(len(y_true)))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.75
    """

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])
        metric_list.append(c_index_race)
    return metric_list
    # return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))






# test, train_original = load_data()


# from scipy.stats import rankdata

# y_true = train_original[["ID","efs","efs_time","race_group"]].copy()
# y_pred = train_original[["ID"]].copy()
# y_pred["prediction"] = oof_nn2
# m = score(y_true.copy(), y_pred.copy(), "ID")
# print(f"\nOverall CV for XGBoost KaplanMeier =",m)


# from scipy.stats import rankdata

# y_true = train_original[["ID","efs","efs_time","race_group"]].copy()
# y_pred = train_original[["ID"]].copy()
# y_pred["prediction"] = oof_nn3
# m = score(y_true.copy(), y_pred.copy(), "ID")
# print(f"\nOverall CV for XGBoost KaplanMeier =",m)


# from scipy.stats import rankdata

# y_true = train_original[["ID","efs","efs_time","race_group"]].copy()
# y_pred = train_original[["ID"]].copy()
# y_pred["prediction"] = oof_nn4
# m = score(y_true.copy(), y_pred.copy(), "ID")
# print(f"\nOverall CV for XGBoost KaplanMeier =",m)


# def add_features(df):
#     # sex_match = df.sex_match.astype(str)
#     # sex_match = sex_match.str.split("-").str[0] == sex_match.str.split("-").str[1]
#     # df['sex_match_bool'] = sex_match
#     # df.loc[df.sex_match.isna(), 'sex_match_bool'] = np.nan
#     # df['big_age'] = df.age_at_hct > 16
#     df.loc[df.year_hct == 2019, 'year_hct'] = 2020
#     df['is_cyto_score_same'] = (df['cyto_score'] == df['cyto_score_detail']).astype(int)
#     df['strange_age'] = df.age_at_hct == 0.044
#     # df['age_bin'] = pd.cut(df.age_at_hct, [0, 0.0441, 16, 30, 50, 100])
#     df['age_ts'] = df.age_at_hct / df.donor_age
#     return df


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

# test = add_features(test)

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)
train.head()

# train=add_features(train)


from scipy.stats import gamma
k, beta = 4, -0.5 #4,0.5
train['y'] = 1 - gamma.cdf(train.efs_time / np.exp(-beta), k)


RMV = ["ID","efs","efs_time","y","stratify_label","donor_age_bin",'age_at_hct_bin','donor_age_bin_missing']
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


combined = pd.concat([train,test],axis=0,ignore_index=True)
#print("Combined data shape:", combined.shape )

# LABEL ENCODE CATEGORICAL FEATURES
print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
for c in FEATURES:

    # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
    if c in CATS:
        print(f"{c}, ",end="")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
        
    # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


# from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
print("Using XGBoost version",xgb.__version__)


from sklearn.model_selection import StratifiedKFold


train['y_bin'] = pd.qcut(train['y'], q=10, labels=False)

# 2. race_group ã�¨ donor_age_bin ã‚’çµ„ã�¿å�ˆã‚�ã�›ã�Ÿæ–°ã�—ã�„ã‚«ãƒ†ã‚´ãƒªã‚’ä½œæˆ�
train['stratify_label'] = train['race_group'].astype(str) + "_" + train['y_bin'].astype(str)



# k, beta = 4, -0.5 #4,0.5
# train.loc[train.efs==0,'y']=1 - gamma.cdf(train.loc[train.efs==0,'efs_time']*2 / np.exp(-beta), k)


# for col in train[FEATURES].columns:
#     # æ¬ æ��å€¤ã�Œã�‚ã‚‹å ´å�ˆ1ã€�ã�ªã�„å ´å�ˆ0ã�®ã‚«ãƒ©ãƒ ã‚’è¿½åŠ 
#     new_col = f'{col}_missing'
#     train[new_col] = train[col].isnull().astype(int)
#     test[new_col] = test[col].isnull().astype(int)
    
#     # æ–°ã�—ã�„ã‚«ãƒ©ãƒ ã‚’FEATURESãƒªã‚¹ãƒˆã�«è¿½åŠ 
#     FEATURES.append(new_col)


from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_xgb_c = np.zeros(len(train))
pred_efs_c = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train, train["efs"])):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index, FEATURES].copy()
    y_train = train.loc[train_index, "efs"]
    x_valid = train.loc[test_index, FEATURES].copy()
    y_valid = train.loc[test_index, "efs"]
    x_test = test[FEATURES].copy()

    model_xgb = XGBClassifier(
        device="cuda",
        max_depth=3,  
        colsample_bytree=0.7129400756425178, 
        subsample=0.8185881823156917, 
        n_estimators=20_000, 
        learning_rate=0.04425768131771064,  
        eval_metric="auc", 
        early_stopping_rounds=50, 
        objective='binary:logistic',
        scale_pos_weight=1.5379160847615545,  
        min_child_weight=4,
        enable_categorical=True,
        gamma=3.1330719334577584
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=100
    )

    # INFER OOF (Probabilities -> Binary)
    oof_xgb_c[test_index] = (model_xgb.predict_proba(x_valid)[:, 1] > 0.5).astype(int)
    # INFER TEST (Probabilities -> Average Probs)
    pred_efs_c += model_xgb.predict_proba(x_test)[:, 1]

# COMPUTE AVERAGE TEST PREDS
pred_efs_c = (pred_efs_c / FOLDS > 0.5).astype(int)

# EVALUATE PERFORMANCE
accuracy = accuracy_score(train["efs"], oof_xgb_c)
f1 = f1_score(train["efs"], oof_xgb_c)
roc_auc = roc_auc_score(train["efs"], oof_xgb_c)
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")


%%time
FOLDS = 10
# kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)



# 5åˆ†å‰²ã�®Stratified K-Fold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(skf.split(train,train['stratify_label'])):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)

    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    model_xgb = XGBRegressor(
        device="cuda",
        objective='reg:tweedie',
        max_depth=6,#3 6
        colsample_bytree=0.5,#0.5
        subsample=0.9,#0.8 0.9
        n_estimators=2000,#2000
        learning_rate=0.02,#0.02
        enable_categorical=True,
        min_child_weight=45,#80,45
        max_cat_to_onehot=8,#7 8
        reg_lambda=1,#1
        reg_alpha=0.1,#0.1
        gamma=0.9,
        eta=0.0,
        # early_stopping_rounds=200,
        monotone_constraints={
                    # 'comorbidity_score': 1,
                    # 'hla_match_c_high': -1,
                    # 'hla_high_res_10': -1,
                    'hla_high_res_6': -1,
                    'hla_high_res_8': -1,
                    # 'hla_low_res_10': -1,
                    'hla_low_res_6': -1,
                    # 'hla_low_res_8': -1,
                    'hla_match_a_high': -1,
                    # 'hla_match_a_low': -1,
                    # 'hla_match_b_high': -1,
                    'hla_match_drb1_low': -1,
                    'hla_match_c_low': -1,
                    # 'hla_match_c_high': -1,
                    # 'donor_age': -1,
                    # 'hla_match_drb1_high': -1,
                    'hla_match_dqb1_low': -1,
                    'hla_nmdp_6': -1,
                    # 'karnofsky_score': -1,

                }
        #early_stopping_rounds=25,
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=500
    )

    # INFER OOF
    oof_xgb[test_index] = model_xgb.predict(x_valid)
    # INFER TEST
    pred_xgb += model_xgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS


from scipy.stats import rankdata

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)


from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
print("Using CatBoost version",cb.__version__)


# %%time
# FOLDS = 10
# # kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) 
    
# oof_cat = np.zeros(len(train))
# pred_cat = np.zeros(len(test))

# for i, (train_index, test_index) in enumerate(skf.split(train,train['stratify_label'])):

#     print("#"*25)
#     print(f"### Fold {i+1}")
#     print("#"*25)
    
#     x_train = train.loc[train_index,FEATURES].copy()
#     y_train = train.loc[train_index,"y"]
#     x_valid = train.loc[test_index,FEATURES].copy()
#     y_valid = train.loc[test_index,"y"]
#     x_test = test[FEATURES].copy()

#     model_cat = CatBoostRegressor(
#         task_type="GPU", 
#         feature_border_type='GreedyLogSum',  # ç·šå½¢å›�å¸°ã�«é�©ã�—ã�Ÿç‰¹å¾´å‡¦ç�†
#         learning_rate=0.022249526148312184,
#         verbose=250,
#         iterations=4491,
#         l2_leaf_reg=0.7357305158548999, 
#         border_count= 246, 
#         bagging_temperature= 0.6896908609682827,
#         depth=5,
#     )
#     model_cat.fit(x_train,y_train,
#               eval_set=(x_valid, y_valid),
#               cat_features=CATS,
#               verbose=250)

#     # INFER OOF
#     oof_cat[test_index] = model_cat.predict(x_valid)
#     # INFER TEST
#     pred_cat += model_cat.predict(x_test)

# # COMPUTE AVERAGE TEST PREDS
# pred_cat /= FOLDS


from lightgbm import LGBMRegressor
import lightgbm as lgb
print("Using LightGBM version",lgb.__version__)


from sklearn.model_selection import GroupKFold
FOLDS = 10
# fold2=custom_group_stratified_kfold(train,train['y'],train['stratify_label'],train['dri_score'],n_splits=10)
# kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# gkf=GroupKFold(n_splits=5)
oof_lgb = np.zeros(len(train))
pred_lgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(skf.split(train,train['stratify_label'])):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)

    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    model_lgb = LGBMRegressor(
        # device="gpu",
        max_depth=4,#3 4
        colsample_bytree=0.2,#0.4 0.2
        # subsample=0.8,
        n_estimators=2500,#2500
        learning_rate=0.02,#0.02
        objective="tweedie",#tweedie
        verbose=-1,
        max_cat_to_onehot=7,#9 7
        cat_smooth=100,#100
        # monotone_constraints=monotone_constraints_list,
        lambda_l1=0.75,#0.75
        lambda_l2=1,#1
        #early_stopping_rounds=25,
    )
    model_lgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
    )

    # INFER OOF
    oof_lgb[test_index] = model_lgb.predict(x_valid)
    # INFER TEST
    pred_lgb += model_lgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_lgb /= FOLDS


from scipy.stats import rankdata

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_lgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)


%%time
from catboost import CatBoostRegressor
import numpy as np

FOLDS = 10
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_cat3 = np.zeros(len(train))
pred_cat3 = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(skf.split(train, train['stratify_label'])):
    print("#" * 25)
    print(f"### Fold {i+1}")
    print("#" * 25)

    x_train = train.loc[train_index, FEATURES].copy()
    y_train = train.loc[train_index, "y"]
    x_valid = train.loc[test_index, FEATURES].copy()
    y_valid = train.loc[test_index, "y"]
    x_test = test[FEATURES].copy()

    model_cat = CatBoostRegressor(
        task_type="GPU",
        boosting_type='Plain',  # ç·šå½¢ãƒ¢ãƒ¼ãƒ‰
        feature_border_type='GreedyLogSum',  # ç·šå½¢å›�å¸°ã�«é�©ã�—ã�Ÿç‰¹å¾´å‡¦ç�†
        learning_rate=0.022249526148312184,
        verbose=250,
        iterations=4491,
        l2_leaf_reg=0.7357305158548999, 
        border_count= 246, 
        bagging_temperature= 0.6896908609682827,
        depth=5,
    )

    model_cat.fit(
        x_train, y_train,
        eval_set=(x_valid, y_valid),
        cat_features=CATS,
        verbose=250
    )

    # OOF äºˆæ¸¬
    oof_cat3[test_index] = model_cat.predict(x_valid)
    # ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿äºˆæ¸¬
    pred_cat3 += model_cat.predict(x_test)

# å¹³å�‡åŒ–
pred_cat3 /= FOLDS



from scipy.stats import rankdata

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_cat3
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)


# SURVIVAL COX NEEDS THIS TARGET (TO DIGEST EFS AND EFS_TIME)
train["efs_time2"] = train.efs_time.copy()
train.loc[train.efs==0,"efs_time2"] *= -1


# FOLDS = 10
# # kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  
# oof_cat_cox = np.zeros(len(train))
# pred_cat_cox = np.zeros(len(test))

# for i, (train_index, test_index) in enumerate(skf.split(train,train['stratify_label'])):

#     print("#"*25)
#     print(f"### Fold {i+1}")
#     print("#"*25)
    
#     x_train = train.loc[train_index,FEATURES].copy()
#     y_train = train.loc[train_index,"efs_time2"]    
#     x_valid = train.loc[test_index,FEATURES].copy()
#     y_valid = train.loc[test_index,"efs_time2"]
#     x_test = test[FEATURES].copy()

#     model_cat_cox = CatBoostRegressor(
#         loss_function="Cox",
#         # task_type="GPU",
#         boosting_type='Plain',
#         learning_rate=0.1,  
#         grow_policy='Lossguide',
#         use_best_model=False,
#         verbose=250,
#         depth=5,
#     )
#     model_cat_cox.fit(x_train,y_train,
#               eval_set=(x_valid, y_valid),
#               cat_features=CATS,
#               verbose=100)
    
#     # INFER OOF
#     oof_cat_cox[test_index] = model_cat_cox.predict(x_valid)
#     # INFER TEST
#     pred_cat_cox += model_cat_cox.predict(x_test)

# # COMPUTE AVERAGE TEST PREDS
# pred_cat_cox /= FOLDS


# from scipy.stats import rankdata

# y_true = train[["ID","efs","efs_time","race_group"]].copy()
# y_pred = train[["ID"]].copy()
# y_pred["prediction"] = oof_cat_cox
# m = score(y_true.copy(), y_pred.copy(), "ID")
# print(f"\nOverall CV for XGBoost KaplanMeier =",m)


# from lifelines import  NelsonAalenFitter
# def create_nelson(data):
#     data=data.copy()
#     naf = NelsonAalenFitter(nelson_aalen_smoothing=0)
#     naf.fit(durations=data['efs_time'], event_observed=data['efs'])
#     return naf.cumulative_hazard_at_times(data['efs_time']).values*-1


# train["y_nel"] = create_nelson(train)

# #important
# train.loc[train.efs == 0, "y_nel"] = (-(-train.loc[train.efs == 0, "y_nel"])**0.5)


# from sklearn.model_selection import KFold,StratifiedKFold
# from xgboost import XGBRegressor, XGBClassifier
# import xgboost as xgb
# print("Using XGBoost version",xgb.__version__)


# %%time
# FOLDS = 10
# def create_stratified_folds(data, target, n_splits=10):
#     data['fold'] = -1
#     # num_bins = int(np.floor(1 + np.log2(len(data))))  # Sturges' rule for binning
#     if (target!="race_group"):
#         data['bins'] = pd.qcut(data[target], q=50, duplicates='drop',labels=False)
#     data["bins"]=data["race_group"]
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     for fold, (_, val_idx) in enumerate(skf.split(data, data['bins'])):
#         data.loc[val_idx, 'fold'] = fold
    
#     data = data.drop(columns=['bins'])
#     return data

# train=create_stratified_folds(train,"race_group",FOLDS)


# FOLDS=10
# oof_xgb2 = np.zeros(len(train))
# pred_xgb2 = np.zeros(len(test))

# for i in range(FOLDS):

#     print("#"*25)
#     print(f"### Fold {i+1}")
#     print("#"*25)
    
#     x_train = train.loc[train.fold!=i,FEATURES].copy()
#     y_train = train.loc[train.fold!=i,"y_nel"]
#     x_valid = train.loc[train.fold==i,FEATURES].copy()
#     y_valid = train.loc[train.fold==i,"y_nel"]
#     x_test = test[FEATURES].copy()
#     model_xgb = XGBRegressor(
#         # device="cuda",
#         max_depth=4,  
#         colsample_bytree=0.55,  
#         subsample=0.8,  
#         n_estimators=5000,  
#         learning_rate=0.02,  
#         enable_categorical=True,
#         min_child_weight=80,
#         early_stopping_rounds=200,
#         n_jobs=4
#     )
#     model_xgb.fit(
#         x_train, y_train,
#         eval_set=[(x_valid, y_valid)],  
#         verbose=500 
#     )
#     oof_xgb2[train.index[train.fold==i]] = (model_xgb.predict(x_valid))
#     # INFER TEST
#     pred_xgb2 += (model_xgb.predict(x_test))

# from metric import score

# y_true = train[["ID","efs","efs_time","race_group"]].copy()
# y_pred = train[["ID"]].copy()
# y_pred["prediction"] = oof_xgb2

# m = score(y_true.copy(), y_pred.copy(), "ID")
# print(f"\nOverall CV for XGBoost NelsonAalenFitter =",m)


#post-processing
mask = oof_xgb_c == 1

# ã��ã�®ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹ã�«å¯¾ã�—ã�¦ oof_xgb ã‚’ +0.1 ã�™ã‚‹
# oof_nn2[mask] += 0.1
# oof_nn3[mask] += 0.1
# oof_xgb[mask]+=0.1


# oof_lgb[mask]+=0.1


# oof_cat3[mask]+=0.1


# mask=pred_efs_c==1
# pred_xgb[mask]+=0.1
# pred_lgb[mask]+=0.1
# test_preds_nn2[mask]+=0.1
# test_preds_nn3[mask]+=0.1
# pred_cat3[mask]+=0.1


# from scipy.stats import rankdata

# y_true = train[["ID","efs","efs_time","race_group"]].copy()
# y_pred = train[["ID"]].copy()
# y_pred["prediction"] = oof_nn2
# m = score(y_true.copy(), y_pred.copy(), "ID")
# print(f"\nOverall CV for XGBoost KaplanMeier =",m)


# from scipy.stats import rankdata

# y_true = train[["ID","efs","efs_time","race_group"]].copy()
# y_pred = train[["ID"]].copy()
# y_pred["prediction"] = oof_nn3
# m = score(y_true.copy(), y_pred.copy(), "ID")
# print(f"\nOverall CV for XGBoost KaplanMeier =",m)


from scipy.stats import rankdata

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)


from scipy.stats import rankdata

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_lgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)


from scipy.stats import rankdata

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_cat3
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)


import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
import warnings
warnings.filterwarnings("ignore")
# OOF äºˆæ¸¬ã‚’ numpy é…�åˆ—ã�«å¤‰æ�›
oof_preds = np.stack([rankdata(oof_xgb), rankdata(oof_lgb),rankdata(oof_cat3)], axis=1)

# ç›®çš„é–¢æ•° (ã‚¹ã‚³ã‚¢ã�®è² ã�®å€¤ã‚’æœ€å°�åŒ–)
def objective(weights):
    weights = np.abs(weights)  # é‡�ã�¿ã�Œè² ã�«ã�ªã‚‰ã�ªã�„ã‚ˆã�†ã�«ã�™ã‚‹
    weights /= np.sum(weights)  # é‡�ã�¿ã�®å�ˆè¨ˆã‚’ 1 ã�«æ­£è¦�åŒ–
    weighted_preds = np.sum(oof_preds * weights, axis=1)
    y_pred = train[["ID"]].copy()
    y_pred["prediction"] = weighted_preds
    return -score(y_true.copy(), y_pred.copy(), "ID")  # ã‚¹ã‚³ã‚¢ã‚’æœ€å¤§åŒ–ã�™ã‚‹ã�Ÿã‚�ã�«è² ã�®å€¤ã‚’è¿”ã�™

# åˆ�æœŸé‡�ã�¿ (å�‡ç­‰ã�«ã�™ã‚‹)
init_weights = np.ones(oof_preds.shape[1]) / oof_preds.shape[1]

# Nelder-Mead ã�§æœ€é�©åŒ–
result = minimize(objective, init_weights, method="Nelder-Mead")

# æœ€é�©ã�ªé‡�ã�¿ã‚’å�–å¾— & æ­£è¦�åŒ–
best_weights = np.abs(result.x)
best_weights /= np.sum(best_weights)

print("Optimal Weights:", best_weights)

# æœ€é�©ã�ªé‡�ã�¿ã‚’é�©ç”¨ã�—ã�¦äºˆæ¸¬
y_pred = train[["ID"]].copy()
y_pred["prediction"] = np.sum(oof_preds * best_weights, axis=1)

# æœ€çµ‚ã‚¹ã‚³ã‚¢
final_score = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for Ensemble = {final_score}")



# import numpy as np
# from scipy.optimize import minimize
# from scipy.stats import rankdata
# import warnings
# warnings.filterwarnings("ignore")
# # OOF äºˆæ¸¬ã‚’ numpy é…�åˆ—ã�«å¤‰æ�›
# oof_preds = np.stack([rankdata(oof_xgb), rankdata(oof_lgb), rankdata(oof_nn2),rankdata(oof_nn3),rankdata(oof_cat3)], axis=1)

# # ç›®çš„é–¢æ•° (ã‚¹ã‚³ã‚¢ã�®è² ã�®å€¤ã‚’æœ€å°�åŒ–)
# def objective(weights):
#     weights = np.abs(weights)  # é‡�ã�¿ã�Œè² ã�«ã�ªã‚‰ã�ªã�„ã‚ˆã�†ã�«ã�™ã‚‹
#     weights /= np.sum(weights)  # é‡�ã�¿ã�®å�ˆè¨ˆã‚’ 1 ã�«æ­£è¦�åŒ–
#     weighted_preds = np.sum(oof_preds * weights, axis=1)
#     y_pred = train[["ID"]].copy()
#     y_pred["prediction"] = weighted_preds
#     return -score2(y_true.copy(), y_pred.copy(), "ID")[0]  # ã‚¹ã‚³ã‚¢ã‚’æœ€å¤§åŒ–ã�™ã‚‹ã�Ÿã‚�ã�«è² ã�®å€¤ã‚’è¿”ã�™

# # åˆ�æœŸé‡�ã�¿ (å�‡ç­‰ã�«ã�™ã‚‹)
# init_weights = np.ones(oof_preds.shape[1]) / oof_preds.shape[1]

# # Nelder-Mead ã�§æœ€é�©åŒ–
# result = minimize(objective, init_weights, method="Nelder-Mead")

# # æœ€é�©ã�ªé‡�ã�¿ã‚’å�–å¾— & æ­£è¦�åŒ–
# best_weights0 = np.abs(result.x)
# best_weights0 /= np.sum(best_weights0)

# print("Optimal Weights:", best_weights0)

# # æœ€é�©ã�ªé‡�ã�¿ã‚’é�©ç”¨ã�—ã�¦äºˆæ¸¬
# y_pred = train[["ID"]].copy()
# y_pred["prediction"] = np.sum(oof_preds * best_weights, axis=1)

# # æœ€çµ‚ã‚¹ã‚³ã‚¢
# final_score = score2(y_true.copy(), y_pred.copy(), "ID")[0]
# print(f"\nOverall CV for Ensemble = {final_score}")



# import numpy as np
# from scipy.optimize import minimize
# from scipy.stats import rankdata
# import warnings
# warnings.filterwarnings("ignore")
# # OOF äºˆæ¸¬ã‚’ numpy é…�åˆ—ã�«å¤‰æ�›
# oof_preds = np.stack([rankdata(oof_xgb), rankdata(oof_lgb), rankdata(oof_nn2),rankdata(oof_nn3),rankdata(oof_cat3)], axis=1)

# # ç›®çš„é–¢æ•° (ã‚¹ã‚³ã‚¢ã�®è² ã�®å€¤ã‚’æœ€å°�åŒ–)
# def objective(weights):
#     weights = np.abs(weights)  # é‡�ã�¿ã�Œè² ã�«ã�ªã‚‰ã�ªã�„ã‚ˆã�†ã�«ã�™ã‚‹
#     weights /= np.sum(weights)  # é‡�ã�¿ã�®å�ˆè¨ˆã‚’ 1 ã�«æ­£è¦�åŒ–
#     weighted_preds = np.sum(oof_preds * weights, axis=1)
#     y_pred = train[["ID"]].copy()
#     y_pred["prediction"] = weighted_preds
#     return -score2(y_true.copy(), y_pred.copy(), "ID")[1]  # ã‚¹ã‚³ã‚¢ã‚’æœ€å¤§åŒ–ã�™ã‚‹ã�Ÿã‚�ã�«è² ã�®å€¤ã‚’è¿”ã�™

# # åˆ�æœŸé‡�ã�¿ (å�‡ç­‰ã�«ã�™ã‚‹)
# init_weights = np.ones(oof_preds.shape[1]) / oof_preds.shape[1]

# # Nelder-Mead ã�§æœ€é�©åŒ–
# result = minimize(objective, init_weights, method="Nelder-Mead")

# # æœ€é�©ã�ªé‡�ã�¿ã‚’å�–å¾— & æ­£è¦�åŒ–
# best_weights1 = np.abs(result.x)
# best_weights1 /= np.sum(best_weights1)

# print("Optimal Weights:", best_weights1)

# # æœ€é�©ã�ªé‡�ã�¿ã‚’é�©ç”¨ã�—ã�¦äºˆæ¸¬
# y_pred = train[["ID"]].copy()
# y_pred["prediction"] = np.sum(oof_preds * best_weights, axis=1)

# # æœ€çµ‚ã‚¹ã‚³ã‚¢
# final_score = score2(y_true.copy(), y_pred.copy(), "ID")[1]
# print(f"\nOverall CV for Ensemble = {final_score}")



# import numpy as np
# from scipy.optimize import minimize
# from scipy.stats import rankdata
# import warnings
# warnings.filterwarnings("ignore")
# # OOF äºˆæ¸¬ã‚’ numpy é…�åˆ—ã�«å¤‰æ�›
# oof_preds = np.stack([rankdata(oof_xgb), rankdata(oof_lgb), rankdata(oof_nn2),rankdata(oof_nn3),rankdata(oof_cat3)], axis=1)

# # ç›®çš„é–¢æ•° (ã‚¹ã‚³ã‚¢ã�®è² ã�®å€¤ã‚’æœ€å°�åŒ–)
# def objective(weights):
#     weights = np.abs(weights)  # é‡�ã�¿ã�Œè² ã�«ã�ªã‚‰ã�ªã�„ã‚ˆã�†ã�«ã�™ã‚‹
#     weights /= np.sum(weights)  # é‡�ã�¿ã�®å�ˆè¨ˆã‚’ 1 ã�«æ­£è¦�åŒ–
#     weighted_preds = np.sum(oof_preds * weights, axis=1)
#     y_pred = train[["ID"]].copy()
#     y_pred["prediction"] = weighted_preds
#     return -score2(y_true.copy(), y_pred.copy(), "ID")[2]  # ã‚¹ã‚³ã‚¢ã‚’æœ€å¤§åŒ–ã�™ã‚‹ã�Ÿã‚�ã�«è² ã�®å€¤ã‚’è¿”ã�™

# # åˆ�æœŸé‡�ã�¿ (å�‡ç­‰ã�«ã�™ã‚‹)
# init_weights = np.ones(oof_preds.shape[1]) / oof_preds.shape[1]

# # Nelder-Mead ã�§æœ€é�©åŒ–
# result = minimize(objective, init_weights, method="Nelder-Mead")

# # æœ€é�©ã�ªé‡�ã�¿ã‚’å�–å¾— & æ­£è¦�åŒ–
# best_weights2 = np.abs(result.x)
# best_weights2 /= np.sum(best_weights2)

# print("Optimal Weights:", best_weights2)

# # æœ€é�©ã�ªé‡�ã�¿ã‚’é�©ç”¨ã�—ã�¦äºˆæ¸¬
# y_pred = train[["ID"]].copy()
# y_pred["prediction"] = np.sum(oof_preds * best_weights, axis=1)

# # æœ€çµ‚ã‚¹ã‚³ã‚¢
# final_score = score2(y_true.copy(), y_pred.copy(), "ID")[2]
# print(f"\nOverall CV for Ensemble = {final_score}")



# import numpy as np
# from scipy.optimize import minimize
# from scipy.stats import rankdata
# import warnings
# warnings.filterwarnings("ignore")
# # OOF äºˆæ¸¬ã‚’ numpy é…�åˆ—ã�«å¤‰æ�›
# oof_preds = np.stack([rankdata(oof_xgb), rankdata(oof_lgb), rankdata(oof_nn2),rankdata(oof_nn3),rankdata(oof_cat3)], axis=1)

# # ç›®çš„é–¢æ•° (ã‚¹ã‚³ã‚¢ã�®è² ã�®å€¤ã‚’æœ€å°�åŒ–)
# def objective(weights):
#     weights = np.abs(weights)  # é‡�ã�¿ã�Œè² ã�«ã�ªã‚‰ã�ªã�„ã‚ˆã�†ã�«ã�™ã‚‹
#     weights /= np.sum(weights)  # é‡�ã�¿ã�®å�ˆè¨ˆã‚’ 1 ã�«æ­£è¦�åŒ–
#     weighted_preds = np.sum(oof_preds * weights, axis=1)
#     y_pred = train[["ID"]].copy()
#     y_pred["prediction"] = weighted_preds
#     return -score2(y_true.copy(), y_pred.copy(), "ID")[3]  # ã‚¹ã‚³ã‚¢ã‚’æœ€å¤§åŒ–ã�™ã‚‹ã�Ÿã‚�ã�«è² ã�®å€¤ã‚’è¿”ã�™

# # åˆ�æœŸé‡�ã�¿ (å�‡ç­‰ã�«ã�™ã‚‹)
# init_weights = np.ones(oof_preds.shape[1]) / oof_preds.shape[1]

# # Nelder-Mead ã�§æœ€é�©åŒ–
# result = minimize(objective, init_weights, method="Nelder-Mead")

# # æœ€é�©ã�ªé‡�ã�¿ã‚’å�–å¾— & æ­£è¦�åŒ–
# best_weights3 = np.abs(result.x)
# best_weights3 /= np.sum(best_weights3)

# print("Optimal Weights:", best_weights3)

# # æœ€é�©ã�ªé‡�ã�¿ã‚’é�©ç”¨ã�—ã�¦äºˆæ¸¬
# y_pred = train[["ID"]].copy()
# y_pred["prediction"] = np.sum(oof_preds * best_weights, axis=1)

# # æœ€çµ‚ã‚¹ã‚³ã‚¢
# final_score = score2(y_true.copy(), y_pred.copy(), "ID")[3]
# print(f"\nOverall CV for Ensemble = {final_score}")



# import numpy as np
# from scipy.optimize import minimize
# from scipy.stats import rankdata
# import warnings
# warnings.filterwarnings("ignore")
# # OOF äºˆæ¸¬ã‚’ numpy é…�åˆ—ã�«å¤‰æ�›
# oof_preds = np.stack([rankdata(oof_xgb), rankdata(oof_lgb), rankdata(oof_nn2),rankdata(oof_nn3),rankdata(oof_cat3)], axis=1)
# # ç›®çš„é–¢æ•° (ã‚¹ã‚³ã‚¢ã�®è² ã�®å€¤ã‚’æœ€å°�åŒ–)
# def objective(weights):
#     weights = np.abs(weights)  # é‡�ã�¿ã�Œè² ã�«ã�ªã‚‰ã�ªã�„ã‚ˆã�†ã�«ã�™ã‚‹
#     weights /= np.sum(weights)  # é‡�ã�¿ã�®å�ˆè¨ˆã‚’ 1 ã�«æ­£è¦�åŒ–
#     weighted_preds = np.sum(oof_preds * weights, axis=1)
#     y_pred = train[["ID"]].copy()
#     y_pred["prediction"] = weighted_preds
#     return -score2(y_true.copy(), y_pred.copy(), "ID")[4]  # ã‚¹ã‚³ã‚¢ã‚’æœ€å¤§åŒ–ã�™ã‚‹ã�Ÿã‚�ã�«è² ã�®å€¤ã‚’è¿”ã�™

# # åˆ�æœŸé‡�ã�¿ (å�‡ç­‰ã�«ã�™ã‚‹)
# init_weights = np.ones(oof_preds.shape[1]) / oof_preds.shape[1]

# # Nelder-Mead ã�§æœ€é�©åŒ–
# result = minimize(objective, init_weights, method="Nelder-Mead")

# # æœ€é�©ã�ªé‡�ã�¿ã‚’å�–å¾— & æ­£è¦�åŒ–
# best_weights4 = np.abs(result.x)
# best_weights4 /= np.sum(best_weights4)

# print("Optimal Weights:", best_weights4)

# # æœ€é�©ã�ªé‡�ã�¿ã‚’é�©ç”¨ã�—ã�¦äºˆæ¸¬
# y_pred = train[["ID"]].copy()
# y_pred["prediction"] = np.sum(oof_preds * best_weights, axis=1)

# # æœ€çµ‚ã‚¹ã‚³ã‚¢
# final_score = score2(y_true.copy(), y_pred.copy(), "ID")[4]
# print(f"\nOverall CV for Ensemble = {final_score}")



# import numpy as np
# from scipy.optimize import minimize
# from scipy.stats import rankdata
# import warnings
# warnings.filterwarnings("ignore")
# # OOF äºˆæ¸¬ã‚’ numpy é…�åˆ—ã�«å¤‰æ�›
# oof_preds = np.stack([rankdata(oof_xgb), rankdata(oof_lgb), rankdata(oof_nn2),rankdata(oof_nn3),rankdata(oof_cat3)], axis=1)

# # ç›®çš„é–¢æ•° (ã‚¹ã‚³ã‚¢ã�®è² ã�®å€¤ã‚’æœ€å°�åŒ–)
# def objective(weights):
#     weights = np.abs(weights)  # é‡�ã�¿ã�Œè² ã�«ã�ªã‚‰ã�ªã�„ã‚ˆã�†ã�«ã�™ã‚‹
#     weights /= np.sum(weights)  # é‡�ã�¿ã�®å�ˆè¨ˆã‚’ 1 ã�«æ­£è¦�åŒ–
#     weighted_preds = np.sum(oof_preds * weights, axis=1)
#     y_pred = train[["ID"]].copy()
#     y_pred["prediction"] = weighted_preds
#     return -score2(y_true.copy(), y_pred.copy(), "ID")[5]  # ã‚¹ã‚³ã‚¢ã‚’æœ€å¤§åŒ–ã�™ã‚‹ã�Ÿã‚�ã�«è² ã�®å€¤ã‚’è¿”ã�™

# # åˆ�æœŸé‡�ã�¿ (å�‡ç­‰ã�«ã�™ã‚‹)
# init_weights = np.ones(oof_preds.shape[1]) / oof_preds.shape[1]

# # Nelder-Mead ã�§æœ€é�©åŒ–
# result = minimize(objective, init_weights, method="Nelder-Mead")

# # æœ€é�©ã�ªé‡�ã�¿ã‚’å�–å¾— & æ­£è¦�åŒ–
# best_weights5 = np.abs(result.x)
# best_weights5 /= np.sum(best_weights5)

# print("Optimal Weights:", best_weights5)

# # æœ€é�©ã�ªé‡�ã�¿ã‚’é�©ç”¨ã�—ã�¦äºˆæ¸¬
# y_pred = train[["ID"]].copy()
# y_pred["prediction"] = np.sum(oof_preds * best_weights, axis=1)

# # æœ€çµ‚ã‚¹ã‚³ã‚¢
# final_score = score2(y_true.copy(), y_pred.copy(), "ID")[5]
# print(f"\nOverall CV for Ensemble = {final_score}")



# oof_preds = np.stack([rankdata(oof_xgb), rankdata(oof_lgb), rankdata(oof_nn2),rankdata(oof_nn3),rankdata(oof_cat3)], axis=1)


# race_group_0_indices = train[train['race_group'] == 0].index.tolist()
# race_group_1_indices = train[train['race_group'] == 1].index.tolist()
# race_group_2_indices = train[train['race_group'] == 2].index.tolist()
# race_group_3_indices = train[train['race_group'] == 3].index.tolist()
# race_group_4_indices = train[train['race_group'] == 4].index.tolist()
# race_group_5_indices = train[train['race_group'] == 5].index.tolist()


# oof_preds2=np.zeros(len(train))
# oof_preds2[race_group_0_indices] = np.sum(oof_preds[race_group_0_indices] * best_weights0, axis=1)
# oof_preds2[race_group_1_indices] = np.sum(oof_preds[race_group_1_indices] * best_weights1, axis=1)
# oof_preds2[race_group_2_indices] = np.sum(oof_preds[race_group_2_indices] * best_weights2, axis=1)
# oof_preds2[race_group_3_indices] = np.sum(oof_preds[race_group_3_indices] * best_weights3, axis=1)
# oof_preds2[race_group_4_indices] = np.sum(oof_preds[race_group_4_indices] * best_weights4, axis=1)
# oof_preds2[race_group_5_indices] = np.sum(oof_preds[race_group_5_indices] * best_weights5, axis=1)


# from scipy.stats import rankdata

# y_true = train[["ID","efs","efs_time","race_group"]].copy()
# y_pred = train[["ID"]].copy()
# y_pred["prediction"] = oof_preds2
# m = score(y_true.copy(), y_pred.copy(), "ID")
# print(f"\nOverall CV for XGBoost KaplanMeier =",m)


# preds = np.stack([rankdata(pred_xgb), rankdata(pred_lgb), rankdata(test_preds_nn2),rankdata(test_preds_nn3),rankdata(pred_cat3)],axis=1)


# race_group_0_indices = test[test['race_group'] == 0].index.tolist()
# race_group_1_indices = test[test['race_group'] == 1].index.tolist()
# race_group_2_indices = test[test['race_group'] == 2].index.tolist()
# race_group_3_indices = test[test['race_group'] == 3].index.tolist()
# race_group_4_indices = test[test['race_group'] == 4].index.tolist()
# race_group_5_indices = test[test['race_group'] == 5].index.tolist()


# test_preds2=np.zeros(len(test))
# if race_group_0_indices!=[]:
#     test_preds2[race_group_0_indices] = np.sum(preds[race_group_0_indices] * best_weights0, axis=1)
# if race_group_1_indices!=[]:
#     test_preds2[race_group_1_indices] = np.sum(preds[race_group_1_indices] * best_weights1, axis=1)
# if race_group_2_indices!=[]:
#     test_preds2[race_group_2_indices] = np.sum(preds[race_group_2_indices] * best_weights2, axis=1)
# if race_group_3_indices!=[]:
#     test_preds2[race_group_3_indices] = np.sum(preds[race_group_3_indices] * best_weights3, axis=1)
# if race_group_4_indices!=[]:
#     test_preds2[race_group_4_indices] = np.sum(preds[race_group_4_indices] * best_weights4, axis=1)
# if race_group_5_indices!=[]:
#     test_preds2[race_group_5_indices] = np.sum(preds[race_group_5_indices] * best_weights5, axis=1)


from scipy.stats import rankdata
sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
preds = np.stack([rankdata(pred_xgb), rankdata(pred_lgb),rankdata(pred_cat3)],axis=1)
sub.prediction = np.sum(preds * best_weights, axis=1)
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()


# from scipy.stats import rankdata

# y_true = train[["ID","efs","efs_time","race_group"]].copy()
# y_pred = train[["ID"]].copy()
# y_pred["prediction"] = rankdata(oof_nn2)+rankdata(oof_xgb)+rankdata(oof_cat)+rankdata(oof_lgb)+rankdata(oof_cat3)
# m = score(y_true.copy(), y_pred.copy(), "ID")
# print(f"\nOverall CV for XGBoost KaplanMeier =",m)


# from scipy.stats import rankdata
# sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
# preds = rankdata(test_preds_nn2)+rankdata(pred_xgb)+rankdata(pred_cat)+rankdata(pred_lgb)+rankdata(pred_cat3)
# sub.prediction = preds
# sub.to_csv("submission.csv",index=False)
# print("Sub shape:",sub.shape)
# sub.head()

