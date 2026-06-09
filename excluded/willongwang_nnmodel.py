# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

%pip install --upgrade pip
%pip install xgboost polars pytorch_lightning
import time
import sys 
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
import xgboost as xgb, time

import pickle
import polars as pl
import gc

import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from pandas import read_parquet


train = pd.read_csv("/kaggle/input/playground-series-s4e12/train.csv",)
test = pd.read_csv("/kaggle/input/playground-series-s4e12/test.csv")

col_map = {}
for col in train.columns:
  col_map[col] = '_'.join(col.split(' '))
train = train.rename(columns=col_map)
test = test.rename(columns=col_map)

## 处理日期变量
def pre_process(data):
  data["Policy_Start_Date"] = pd.to_datetime( data["Policy_Start_Date"] )
  data["year"] = data["Policy_Start_Date"].dt.year.astype("float32")
  data["month"] = data["Policy_Start_Date"].dt.month.astype("float32")
  data["day"] = data["Policy_Start_Date"].dt.day.astype("float32")
  data["dow"] = data["Policy_Start_Date"].dt.dayofweek.astype("float32")
  data["seconds"] = (data["Policy_Start_Date"].astype("int64") // 10**9).astype("float32")

  return data

train = pre_process(train)
test = pre_process(test)
train = train.drop('Policy_Start_Date',axis=1)
test = test.drop('Policy_Start_Date',axis=1)

train["target"] = np.log1p( train["Premium_Amount"])  ##np.log(train["Premium_Amount"]+1)
train.drop('Premium_Amount',axis=1,inplace=True)

import pandas as pd

# 定义移除的列
columns_to_remove = ["id", "Policy Start Date", "Premium Amount", "target"]

# 从训练集和测试集中筛选出需要的特征列
features = [col for col in train.columns if col not in columns_to_remove]

# 合并训练集和测试集
combined = pd.concat([train, test], axis=0, ignore_index=True)

# 初始化分类特征和高基数特征列表
categorical_features = []
high_cardinality_features = []

print(f"THE {len(features)} BASIC features ARE:")

# 遍历特征列，处理数据类型和缺失值
for feature in features:
    feature_type = "numerical"
    unique_values_count = combined[feature].nunique()

    # 处理分类特征
    if combined[feature].dtype == "object":
        categorical_features.append(feature)
        combined[feature] = combined[feature].fillna("NAN")
        combined[feature], _ = combined[feature].factorize()
        combined[feature] -= combined[feature].min()
        feature_type = "categorical"

    # 优化数据类型
    if combined[feature].dtype == "int64":
        combined[feature] = combined[feature].astype("int32")
    elif combined[feature].dtype == "float64":
        combined[feature] = combined[feature].astype("float32")

    print(f"{feature} ({feature_type}) with {unique_values_count} unique values")

    # 检测高基数特征
    if unique_values_count >= 9:
        high_cardinality_features.append(feature)

# 分离训练集和测试集
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()

print("\nTHE FOLLOWING HAVE 9 OR MORE UNIQUE VALUES:", high_cardinality_features)

encoding_features = [
    ['Annual_Income', 'Health_Score'], 
    ['Credit_Score', 'Health_Score'], 
    ['Customer_Feedback', 'Gender', 'Marital_Status', 'Occupation', 'Smoking_Status', 'year'],
    ['Exercise_Frequency', 'Health_Score'],
    ['Health_Score', 'Marital_Status'],
    ['Education_Level', 'Gender', 'Health_Score'],
    ['Health_Score', 'Occupation'],
    ['Age', 'Health_Score'],
    ['Health_Score', 'dow'],
    ['Age', 'Exercise_Frequency', 'Location'],
    ['Health_Score', 'Smoking_Status', 'month'],
    ['Health_Score', 'Location', 'Policy_Type'],
    ['Health_Score', 'Insurance_Duration'],
    ['Health_Score', 'Number_of_Dependents'],
    ['Customer_Feedback', 'Exercise_Frequency', 'Previous_Claims', 'Property_Type', 'dow'],
    ['Customer_Feedback', 'Health_Score'],
    ['Health_Score', 'Property_Type'],
    ['Health_Score', 'day', 'seconds'],
    ['Health_Score', 'year'],
    ['Age', 'Gender', 'Insurance_Duration', 'year']
]

def target_encode(train, valid, test, col, target="target", kfold=5, smooth=20, agg="mean"):
    """
    对指定列进行目标编码（Target Encoding），并使用 k 折交叉验证来避免过拟合。

    参数：
        train (pd.DataFrame): 训练集。
        valid (pd.DataFrame): 验证集。
        test (pd.DataFrame): 测试集。
        col (list): 需要进行目标编码的列名列表。
        target (str): 目标变量列名，默认为 "target"。
        kfold (int): 交叉验证的折数，默认为 5。
        smooth (int): 平滑参数，用于防止过拟合，默认为 20。
        agg (str): 聚合方法，可选值为 "mean"、"median"、"min"、"max"、"nunique"，默认为 "mean"。

    返回：
        tuple: 包含目标编码列的 (train, valid, test)。
    """
    # 初始化 k 折列和目标编码列名
    train['kfold'] = train.index % kfold
    col_name = '_'.join(col)
    target_encoded_col = f'TE_{agg.upper()}_{col_name}'
    train[target_encoded_col] = 0.0

    # 计算全局聚合值
    if agg == "mean":
        global_agg = train[target].mean()
    elif agg == "median":
        global_agg = train[target].median()
    elif agg == "min":
        global_agg = train[target].min()
    elif agg == "max":
        global_agg = train[target].max()
    elif agg == "nunique":
        global_agg = 0
    else:
        raise ValueError("无效的聚合方法。请选择 'mean'、'median'、'min'、'max' 或 'nunique'。")

    # 对训练集进行 k 折目标编码
    for fold in range(kfold):
        # 分割数据为训练折和验证折
        df_train = train[train['kfold'] != fold]
        df_valid = train[train['kfold'] == fold]

        # 计算每个组合的聚合值
    
        agg_values = df_train.groupby(col)[target].agg([agg, 'count']).reset_index()

        agg_values.columns = col + [agg, 'count']

        if agg == "nunique":
            # 如果是 nunique，计算唯一值的比例
            agg_values['TE_tmp'] = agg_values[agg] / agg_values['count']
        else:
            # 使用平滑公式计算目标编码值
            agg_values['TE_tmp'] = ((agg_values[agg] * agg_values['count']) + (global_agg * smooth)) / (agg_values['count'] + smooth)

        # 将目标编码值合并回验证折
        df_valid = df_valid.merge(agg_values[col + ['TE_tmp']], on=col, how='left')
        df_valid[target_encoded_col] = df_valid['TE_tmp'].fillna(global_agg)

        # 更新训练集的目标编码列
        train.loc[train['kfold'] == fold, target_encoded_col] = df_valid[target_encoded_col].values

    # 删除 k 折列
    train = train.drop('kfold', axis=1)
    train[target_encoded_col] = train[target_encoded_col].astype("float32")

    # 计算全局目标编码值，用于验证集和测试集
    agg_values = train.groupby(col)[target].agg([agg, 'count']).reset_index()
    agg_values.columns = col + [agg, 'count']

    if agg == "nunique":
        agg_values['TE_tmp'] = agg_values[agg] / agg_values['count']
    else:
        agg_values['TE_tmp'] = ((agg_values[agg] * agg_values['count']) + (global_agg * smooth)) / (agg_values['count'] + smooth)

    # 将目标编码值合并到验证集和测试集，未删TE_tmp列
    valid = valid.merge(agg_values[col + ['TE_tmp']], on=col, how='left',suffixes=['','_'])
    valid[target_encoded_col] = valid['TE_tmp'].fillna(global_agg).astype("float32")

    test = test.merge(agg_values[col + ['TE_tmp']], on=col, how='left',suffixes=['','_'])
    test[target_encoded_col] = test['TE_tmp'].fillna(global_agg).astype("float32")

    return train, valid, test



class config:
    def __init__(self):
        self.usegpu = True
        self.gpuid = 1
        self.seed = 2025
        self.model = 'nn'
        self.loader_workers = 0
        self.bs = 1000 #batch_size
        self.lr = 1e-3            ###1e-3 --->2e-4
        self.weight_decay = 5e-4  ##5e-4 --->3e-4
        self.dropouts = [0.1, 0.1,0.1]
        self.n_hidden = [512,256,256] #hidden_dims
        self.max_epochs = 200
        self.N_fold = 5
        self.patience =  10
        self.accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'

my_args = config()

class NNDataset(Dataset):
    def __init__(self, df, accelerator,feature_names,label):
        self.features = torch.FloatTensor(df[feature_names].values).to(accelerator)
        self.labels = torch.FloatTensor(df[label].values).to(accelerator)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y


class DataModule(LightningDataModule):
    def __init__(self, train_df, batch_size, valid_df=None, accelerator='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.df = train_df
        self.batch_size = batch_size
        self.accelerator = accelerator
        self.train_dataset = None
        self.valid_df = None
        if valid_df is not None:
            self.valid_df = valid_df
        self.val_dataset = None

    #下面是overwrite LightningDataModule的
    def setup(self):
        self.train_dataset = NNDataset(self.df, self.accelerator,feature_names=self.df.drop("target",axis=1).columns,label='target')
        if self.valid_df is not None:
            self.val_dataset = NNDataset(self.valid_df, self.accelerator,feature_names=self.df.drop("target",axis=1).columns,label='target')

    def train_dataloader(self, n_workers=0):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=n_workers)

    def val_dataloader(self, n_workers=0):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=n_workers)

class NN(LightningModule):
    def __init__(self, input_dim, hidden_dims, dropouts, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        layers = []
        in_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.BatchNorm1d(in_dim))
            if i > 0:
                layers.append(nn.SiLU())
            if i < len(dropouts):
                layers.append(nn.Dropout(dropouts[i]))
            layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        # layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
        self.lr = lr
        self.weight_decay = weight_decay
        self.validation_step_outputs = []

    def forward(self, x):
        return  self.model(x).squeeze(-1)
    
    def root_mean_squared_error(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    #下面是overwrite LightningModule的
    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        # print(y_hat)
        loss = F.mse_loss(y_hat, y, reduction='none')
        loss = loss.mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='none')
        loss = loss.mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.validation_step_outputs.append((y_hat, y,))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }
    
    #下面的在LightningDataModule中没找到，怎么打印？
    def on_validation_epoch_end(self):
        """Calculate validation WRMSE at the end of the epoch."""
        y = torch.cat([x[1] for x in self.validation_step_outputs]).cpu().numpy()
        if self.trainer.sanity_checking:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
        else:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
            # print('validation_step_outputs',self.validation_step_outputs)
            # print('y',y)
            # print('prob',prob)
            val_score = self.root_mean_squared_error(y, prob)
            self.log("val_score", val_score, prog_bar=True, on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        epoch = self.trainer.current_epoch
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.trainer.logged_metrics.items()}
        formatted_metrics = {k: f"{v:.5f}" for k, v in metrics.items()}
        print(f"Epoch {epoch}: {formatted_metrics}")


FOLDS = 5
from sklearn.model_selection import KFold
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(len(train))
pred = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,features+["target"] ].copy()
    y_train = train.loc[train_index,"target"]
    x_valid = train.loc[test_index,features+["target"]].copy()
    y_valid = train.loc[test_index,"target"]
    x_test = test[features].copy()

    start = time.time()
    print(f"FEATURE ENGINEER {len(features)} COLUMNS and {len(encoding_features)} GROUPS: ",end="")
    for j,f in enumerate(features+encoding_features):

        if j<len(features): c = [f]
        else: c = f 
        print(f"({j+1}){c}",", ",end="")

        # LOW CARDINALITY features - TARGET ENCODE MEAN AND MEDIAN
        x_train, x_valid, x_test = target_encode(x_train, x_valid, x_test, c, smooth=20, agg="mean")
        x_train, x_valid, x_test = target_encode(x_train, x_valid, x_test, c, smooth=0, agg="median")

        # HIGH CARDINALITY features - TE MIN, MAX, NUNIQUE and CE
        if (j>=len(features)) | (c[0] in high_cardinality_features):
            x_train, x_valid, x_test = target_encode(x_train, x_valid, x_test, c, smooth=0, agg="min")
            x_train, x_valid, x_test = target_encode(x_train, x_valid, x_test, c, smooth=0, agg="max")
            x_train, x_valid, x_test = target_encode(x_train, x_valid, x_test, c, smooth=0, agg="nunique")
    
            # COUNT ENCODING (USING COMBINED TRAIN TEST)
            tmp = combined.groupby(c)['target'].count()
            nm = f"CE_{'_'.join(c)}"; tmp.name = nm
            x_train = x_train.merge(tmp, on=c, how="left")
            x_valid = x_valid.merge(tmp, on=c, how="left")
            x_test = x_test.merge(tmp, on=c, how="left")

            x_train[nm] = x_train[nm].fillna(x_train[nm].mean())
            x_valid[nm] = x_valid[nm].fillna(x_valid[nm].mean())
            x_test[nm] = x_test[nm].fillna(x_test[nm].mean())

            x_train[nm] = x_train[nm].astype("int32").fillna(0)
            x_valid[nm] = x_valid[nm].astype("int32").fillna(0)
            x_test[nm] = x_test[nm].astype("int32").fillna(0)

    FEATURES = x_train.columns.tolist()

    x_train[FEATURES] =  x_train[FEATURES].fillna(0).astype('float32')
    x_valid[FEATURES] =  x_valid[FEATURES].fillna(0).astype('float32')
    x_test =  x_test.fillna(0).astype('float32')

    data_module = DataModule(train_df = x_train, valid_df=x_valid,batch_size=my_args.bs, accelerator=my_args.accelerator)
    data_module.setup()
    input_dim = data_module.train_dataset.features.shape[1]
    model = NN(
        input_dim=input_dim,
        hidden_dims=my_args.n_hidden,
        dropouts=my_args.dropouts,
        lr=my_args.lr,
        weight_decay=my_args.weight_decay
    )
    early_stopping = EarlyStopping('val_loss', patience=my_args.patience, mode='min', verbose=False)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, verbose=False, filename=f"./models/nn.model") 
    timer = Timer()

    trainer = Trainer(
        max_epochs=my_args.max_epochs,
        accelerator=my_args.accelerator,
        devices= my_args.gpuid if my_args.usegpu else None,
        logger=None,
        callbacks=[early_stopping, checkpoint_callback, timer],
        enable_progress_bar=True
    )
    trainer.fit(model, data_module.train_dataloader(my_args.loader_workers), data_module.val_dataloader(my_args.loader_workers))
    print(f' Training completed in {timer.time_elapsed("train"):.2f}s')
    end = time.time()
    elapsed = end-start
    print(f"Feature engineering took {elapsed:.1f} seconds")


def root_mean_squared_error( y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

preds = np.zeros((len(test),))
for i in range(0,5):
   model_path = f"lightning_logs/version_{i}/checkpoints/models/nn.model.ckpt"
   model = NN.load_from_checkpoint(model_path)
   model.eval()
   model = model.to(my_args.accelerator)
   test_input = torch.FloatTensor(x_test[x_train.drop('target',axis=1).columns.tolist()].fillna(0).to_numpy()).to(my_args.accelerator)
   with torch.no_grad():
    nn_preds = model(test_input)
    preds += nn_preds.cpu().numpy()
preds /= 5

submission=pd.read_csv(r"/kaggle/input/playground-series-s4e12/sample_submission.csv")
submission['Premium Amount']=np.exp(preds)-1
submission.to_csv(r"submission.csv",index=False)
submission.head()

