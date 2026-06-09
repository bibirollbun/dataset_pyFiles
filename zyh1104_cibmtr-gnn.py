!pip install -q /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install -q /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/pytorch_lightning-2.4.0-py3-none-any.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/scikit_learn-1.6.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/torchmetrics-1.5.2-py3-none-any.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/pytorch_tabnet-4.1.0-py3-none-any.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/einops-0.7.0-py3-none-any.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/pytorch_tabular-1.1.1-py2.py3-none-any.whl


!pip install /kaggle/input/pip-install-pyg/torch_spline_conv-1.2.2+pt25cu124-cp310-cp310-linux_x86_64.whl
!pip install /kaggle/input/pip-install-pyg/torch_sparse-0.6.18+pt25cu124-cp310-cp310-linux_x86_64.whl
!pip install /kaggle/input/pip-install-pyg/pyg_lib-0.4.0+pt25cu124-cp310-cp310-linux_x86_64.whl
!pip install /kaggle/input/pip-install-pyg/torch_cluster-1.6.3+pt25cu124-cp310-cp310-linux_x86_64.whl
!pip install /kaggle/input/pip-install-pyg/torch_geometric-2.6.1-py3-none-any.whl


from pathlib import Path
from metric import score
import pandas as pd
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')

ROOT_DATA_PATH = Path(r"/kaggle/input/equity-post-HCT-survival-predictions")

train = pd.read_csv(ROOT_DATA_PATH.joinpath("train.csv"))
test = pd.read_csv(ROOT_DATA_PATH.joinpath("test.csv"))


import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset
from warnings import filterwarnings
from torchvision.models.resnet import BasicBlock, ResNet
filterwarnings('ignore')


def get_X_cat(df, cat_cols, transformers=None):
    """
    Apply a specific categorical data transformer or a LabelEncoder if None.
    """
    if transformers is None:
        transformers = [LabelEncoder().fit(df[col]) for col in cat_cols]
    return transformers, np.array(
        [transformer.transform(df[col]) for col, transformer in zip(cat_cols, transformers)]
    ).T


def preprocess_data(train, val):
    """
    Standardize numerical variables and transform (Label-encode) categoricals.
    Fill NA values with mean for numerical.
    Create torch dataloaders to prepare data for training and evaluation.
    """
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
    """
    Remove constant categorical columns and transform them using LabelEncoder.
    Return the label-transformers for each categorical column, categorical dataframes and numerical columns.
    """
    categorical_cols, numerical = get_feature_types(train)
    remove = []
    for col in categorical_cols:
        if train[col].nunique() == 1:
            remove.append(col)
        # 处理验证集中出现训练集未见的类别，用训练集众数填充
        ind = ~val[col].isin(train[col])
        if ind.any():
            mode_val = train[col].mode()[0]
            val.loc[ind, col] = mode_val  # 修改：用训练集的众数填充未知类别
    categorical_cols = [col for col in categorical_cols if col not in remove]
    
    # 添加缺失值处理
    for col in categorical_cols:
        train[col] = train[col].fillna(train[col].mode()[0])
        val[col] = val[col].fillna(train[col].mode()[0])
    
    transformers, X_cat_train = get_X_cat(train, categorical_cols)
    _, X_cat_val = get_X_cat(val, categorical_cols, transformers)
    return X_cat_train, X_cat_val, numerical, transformers
    
    
   
    
def init_dl(X_cat, X_num, df, training=False):
    """
    Initialize data loaders with NaN检查
    """
    # 添加数据检查
    assert not np.isnan(X_cat).any(), "NaN in categorical features"
    assert not np.isnan(X_num).any(), "NaN in numerical features"
    
    # 对目标变量进行log变换时添加最小值保护
    y = np.log(df.efs_time.values.clip(1e-6, None))  # 防止取log(0)
    
    ds_train = TensorDataset(
        torch.tensor(X_cat, dtype=torch.long),
        torch.tensor(X_num, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(df.efs.values, dtype=torch.long)
    )
    
    bs = 512  # 减小batch size
    dl_train = torch.utils.data.DataLoader(
        ds_train, 
        batch_size=bs, 
        pin_memory=True, 
        shuffle=training,
        drop_last=training  # 训练时丢弃不完整batch
    )
    return dl_train


def get_feature_types(train):
    """
    Utility function to return categorical and numerical column names.
    """
    categorical_cols = [col for i, col in enumerate(train.columns) if ((train[col].dtype == "object") | (2 < train[col].nunique() < 25))]
    RMV = ["ID", "efs", "efs_time", "y"]
    FEATURES = [c for c in train.columns if not c in RMV]
    numerical = [i for i in FEATURES if i not in categorical_cols]
    return categorical_cols, numerical


def add_features(df):
    # 增加时间相关特征
    if 'year_hct' in df.columns:
        df['year_hct_squared'] = df['year_hct']**2
    if 'age_at_hct' in df.columns and 'year_hct' in df.columns:
        df['age_hct_ratio'] = df['age_at_hct'] / (df['year_hct'] - 2000 + 1e-5)
    
    # 增加交互特征
    if 'disease_group' in df.columns and 'cyto_score' in df.columns:
        df['disease_cyto_interaction'] = df['disease_group'] * df['cyto_score']
    
    # 增加统计特征
    if 'donor_type' in df.columns and 'hla_match' in df.columns:
        df['hct_complexity'] = df[['donor_type', 'hla_match']].apply(
            lambda x: 1 if x['donor_type'] == 'URD' and x['hla_match'] < 8 else 0, axis=1
        )

    df['is_cyto_score_same'] = (df['cyto_score'] == df['cyto_score_detail']).astype(int)
    df['year_hct'] -= 2000
    
    return df



def load_data():
    """
    Load data and add features.
    """
    test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
    test = add_features(test)
    print("Test shape:", test.shape)
    train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
    train = add_features(train)
    print("Train shape:", train.shape)
    return test, train


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


from typing import List
import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import GCNConv, knn_graph
import pytorch_lightning as pl


class CatEmbeddings(nn.Module):
    """
    嵌入模块：对所有分类特征进行嵌入并通过投影层降维
    """
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
        # x_cat: [batch_size, num_cat_features]
        x_cat = [embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)]
        x_cat = torch.cat(x_cat, dim=1)
        return self.projection(x_cat)

# 损失函数
class TukeyBiweightLoss(nn.Module):
    def __init__(self, c=4.685):
        super(TukeyBiweightLoss, self).__init__()
        self.c = c

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        scaled_error = error / self.c
        condition = torch.abs(scaled_error) < 1
        loss = torch.where(
            condition,
            (self.c ** 2 / 6) * (1 - (1 - scaled_error ** 2) ** 3),
            torch.tensor(self.c ** 2 / 6, device=y_pred.device)
        )
        return loss.mean()


#GNN
class GNN(nn.Module):
    def __init__(
        self,
        continuous_dim: int,
        categorical_cardinality: List[int],
        embedding_dim: int,
        projection_dim: int,
        hidden_dim: int,
        dropout: float = 0.3,
        knn_k: int = 8  # 修正参数名称
    ):
        super(GNN, self).__init__()
        self.embeddings = CatEmbeddings(projection_dim, categorical_cardinality, embedding_dim)
        self.input_dim = projection_dim + continuous_dim
        
        # 补充缺失的组件
        self.dropout_layer = nn.Dropout(dropout)  # 正确定义dropout
        self.relu = nn.ReLU()  # 定义激活函数
        self.knn_k = knn_k  # 保存为实例变量

        # 层归一化
        self.norm1 = nn.LayerNorm(self.input_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # GCN层
        self.gcn1 = GCNConv(self.input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)

        # 输出层
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x_cat, x_cont):
        # 分类特征嵌入并投影
        x_emb = self.embeddings(x_cat)
        # 拼接数值特征并应用归一化
        x = torch.cat([x_emb, x_cont], dim=1)
        x = self.norm1(x)  # 应用层归一化
        x = self.dropout_layer(x)
        
        # 构造k-NN图
        edge_index = knn_graph(x, k=self.knn_k, batch=None, loop=False)
        
        # GCN处理流程
        x = self.gcn1(x, edge_index)
        x = self.norm2(self.relu(x))  # 归一化+激活
        x = self.dropout_layer(x)
        
        x = self.gcn2(x, edge_index)
        x = self.norm3(self.relu(x))  # 归一化+激活
        x = self.dropout_layer(x)
        
        x = self.gcn3(x, edge_index)
        x = self.relu(x)
        out = self.out(x)
        return out, x


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
            aux_weight: float = 0.1,
            margin: float = 0.5,
            race_index: int = 0
    ):
        super(LitNN, self).__init__()
        self.save_hyperparameters()
        self.tukey_loss = TukeyBiweightLoss(c=4.685)
        # 使用 GNN 模型
        self.model = GNN(
            continuous_dim=self.hparams.continuous_dim,
            categorical_cardinality=self.hparams.categorical_cardinality,
            embedding_dim=self.hparams.embedding_dim,
            projection_dim=self.hparams.projection_dim,
            hidden_dim=self.hparams.hidden_dim,
            dropout=self.hparams.dropout
        )
        self.targets = []

        # 辅助任务的小型前馈网络
        self.aux_cls = nn.Sequential(
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim // 3),
            nn.GELU(),
            nn.Linear(self.hparams.hidden_dim // 3, 1)
        )

    def on_before_optimizer_step(self, optimizer):
        norms = pl.utilities.grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def forward(self, x_cat, x_cont):
        x, emb = self.model(x_cat, x_cont)
        return x.squeeze(1), emb


    def training_step(self, batch, batch_idx):
        x_cat, x_cont, y, efs = batch
    
    # 添加输入数据检查
        if torch.isnan(x_cont).any():
            raise ValueError("NaN detected in input features")
     
        y_hat, emb = self(x_cat, x_cont)
    
    # 主损失计算（使用新的calc_loss）
        main_loss = self.calc_loss(y, y_hat, efs)
    
    # 辅助损失修正
        aux_mask = efs == 1
        if aux_mask.sum() > 0:  # 只在有事件样本时计算
        # 关键修改：对emb进行mask筛选
            aux_pred = self.aux_cls(emb[aux_mask]).squeeze(1)
            aux_loss = nn.functional.mse_loss(aux_pred, y[aux_mask])
        else:
            aux_loss = torch.tensor(0.0, device=efs.device)
    
        final_loss = main_loss + 0.1 * aux_loss
        self.log('train_loss', final_loss, prog_bar=True)
        self.log('aux_loss', aux_loss, prog_bar=True)
    
        return final_loss
    

    def get_full_loss(self, efs, x_cat, y, y_hat):
        """
        Output loss and race_group loss.
        """
        loss = self.calc_loss(y, y_hat, efs)
        race_loss = self.get_race_losses(efs, x_cat, y, y_hat)
        loss += 0.1 * race_loss
        return loss, race_loss

    def get_race_losses(self, efs, x_cat, y, y_hat):
        """
        Calculate loss for each race_group based on deviation/variance.
        """
        races = torch.unique(x_cat[:, self.hparams.race_index])
        race_losses = []
        for race in races:
            ind = x_cat[:, self.hparams.race_index] == race
            race_losses.append(self.calc_loss(y[ind], y_hat[ind], efs[ind]))
        race_loss = sum(race_losses) / len(race_losses)
        races_loss_std = sum((r - race_loss)**2 for r in race_losses) / len(race_losses)
        return torch.sqrt(races_loss_std)

    
    def calc_loss(self, y, y_hat, efs):
        # 仅对事件发生（efs=1）的样本计算损失
        mask = efs == 1
        if mask.sum() == 0:
            return torch.tensor(0.0, device=efs.device)
        return self.tukey_loss(y_hat[mask], y[mask])
    
    
    def validation_step(self, batch, batch_idx):
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        loss = self.calc_loss(y, y_hat, efs)  # 直接使用新的损失计算
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        """
        At the end of the validation epoch, it computes and logs the concordance index
        """
        cindex, metric = self._calc_cindex()
        self.log("cindex", metric, on_epoch=True, prog_bar=True, logger=True)
        self.log("cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
        self.targets.clear()

    def _calc_cindex(self):
        """
        Calculate c-index accounting for each race_group or global.
        """
        y = torch.cat([t[0] for t in self.targets]).cpu().numpy()
        y_hat = torch.cat([t[1] for t in self.targets]).cpu().numpy()
        efs = torch.cat([t[2] for t in self.targets]).cpu().numpy()
        races = torch.cat([t[3] for t in self.targets]).cpu().numpy()
        metric = self._metric(efs, races, y, y_hat)
        cindex = concordance_index(y, y_hat, efs)
        return cindex, metric

    
    def _metric(self, efs, races, y, y_hat):
        """
        Calculate c-index accounting for each race_group
        """
        metric_list = []
        for race in np.unique(races):
            y_ = y[races == race]
            y_hat_ = y_hat[races == race]
            efs_ = efs[races == race]
            metric_list.append(concordance_index(y_, y_hat_, efs_))
        metric = float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))
        return metric

    def test_step(self, batch, batch_idx):
        """
        Same as training step but to log test data
        """
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        self.targets.append([y, y_hat.detach(), efs, x_cat[:, self.hparams.race_index]])
        self.log("test_loss", loss)
        return loss

    def on_test_epoch_end(self) -> None:
        """
        At the end of the test epoch, calculates and logs the concordance index for the test set
        """
        cindex, metric = self._calc_cindex()
        self.log("test_cindex", metric, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
        self.targets.clear()


    def configure_optimizers(self):
        """优化器配置改进"""
        optimizer = torch.optim.AdamW(  # 改用AdamW增加稳定性
            self.parameters(), 
            lr=0.01,  # 降低初始学习率
            weight_decay=0.001  # 增加权重衰减
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

        #return {"optimizer": optimizer, "lr_scheduler": scheduler_config}


import json
import pytorch_lightning as pl
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.callbacks import StochasticWeightAveraging
from sklearn.model_selection import StratifiedKFold

pl.seed_everything(42)

def main(hparams):
    """
    Main function to train the model.
    The steps are as following :
    * load data and fill efs and efs time for test data with 1
    * initialize pred array with 0
    * get categorical and numerical columns
    * split the train data on the stratified criterion : race_group * newborns yes/no
    * preprocess the fold data (create dataloaders)
    * train the model and create final submission output
    """
    test, train_original = load_data()
    test['efs_time'] = 1
    test['efs'] = 1
    oof_nn_pairwise = np.zeros(len(train_original))
    test_pred = np.zeros(test.shape[0])
    categorical_cols, numerical = get_feature_types(train_original)
    kf = StratifiedKFold(n_splits=10, shuffle=True, )
    for i, (train_index, test_index) in enumerate(
        kf.split(
            train_original, train_original.race_group.astype(str) + (train_original.age_at_hct == 0.044).astype(str)
        )
    ):
        tt = train_original.copy()
        train = tt.iloc[train_index]
        val = tt.iloc[test_index]
        X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers = preprocess_data(train, val)
        model = train_final(X_num_train, dl_train, dl_val, transformers, categorical_cols=categorical_cols)
        oof_pred, _ = model.cuda().eval()(
            torch.tensor(X_cat_val, dtype=torch.long).cuda(),
            torch.tensor(X_num_val, dtype=torch.float32).cuda()
        )
        oof_nn_pairwise[test_index] = oof_pred.detach().cpu().numpy()
        # Create submission
        train = tt.iloc[train_index]
        X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers = preprocess_data(train, test)
        pred, _ = model.cuda().eval()(
            torch.tensor(X_cat_val, dtype=torch.long).cuda(),
            torch.tensor(X_num_val, dtype=torch.float32).cuda()
        )
        test_pred += pred.detach().cpu().numpy()
        
    
    return -test_pred, -oof_nn_pairwise


def train_final(X_num_train, dl_train, dl_val, transformers, hparams=None, categorical_cols=None):
    """
    Defines model hyperparameters and fit the model.
    """
    if hparams is None:
        hparams = {
            "embedding_dim": 64,#32
            "projection_dim": 112,
            "hidden_dim": 56,
            "lr": 0.06464861983337984,
            "dropout": 0.05463240181423116,
            "aux_weight": 0.26545778308743806,
            "margin": 0.2588153271003354,
            "weight_decay": 0.0002773544957610778
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
        log_every_n_steps=6,
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


hparams = None
pairwise_ranking_pred, pairwise_ranking_oof = main(hparams)

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = pairwise_ranking_oof
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nPairwise ranking NN CV =", m)


subm_data = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
subm_data['prediction'] = pairwise_ranking_pred
subm_data.to_csv('submission.csv', index=False)
print(subm_data.head())

