%%time
!pip install -q torchviz


import os
import gc
import math
import ctypes
import random
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import matplotlib.image as mpimg
from colorama import Fore, Style
from typing import Optional, Tuple
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

import torch
import torch.nn as nn
from torchviz import make_dot
import torch.nn.functional as F

warnings.filterwarnings('ignore')
print("PyTorch version:", torch.__version__)


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
seed_everything(seed=42)

def clean_memory():
    gc.collect()
    ctypes.CDLL('libc.so.6').malloc_trim(0)
clean_memory()


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
print("Train shape:", train.shape)
print("Test shape:", test.shape)


%%time
X = train.drop(['id', 'Calories'], axis=1)
y = train.Calories; train_id = train.id
X_test = test.drop(['id'], axis=1)
del train, test; clean_memory()
print("X      shape:", X.shape)
print("X_test shape:", X_test.shape, '\n')

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
print("init len(cat_cols):", len(cat_cols))
print("init len(num_cols):", len(num_cols), '\n')

def feature_engineering(df):
#    df['Body_Temp_binary_'] = pd.Series(np.where(df['Body_Temp'] <= 39.5, 0, 1)).astype('category')    
#    df['_Duration_*_Heart_Rate'] = (df['Duration'] * df['Heart_Rate']).astype('float32')
#    df['_Duration_*_Body_Temp'] = (df['Duration'] * df['Body_Temp']).astype('float32')
    
    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    new_num_cols = [col for col in df.columns if col.startswith('_')]
    return df, new_cat_cols, new_num_cols

X, new_cat_cols, new_num_cols = feature_engineering(X)
X_test, new_cat_cols, new_num_cols = feature_engineering(X_test)
num_cols += new_num_cols; cat_cols += new_cat_cols
print("len(new_cat_cols):", len(new_cat_cols))
print("len(new_num_cols):", len(new_num_cols), '\n')
print("prep len(cat_cols):", len(cat_cols))
print("prep len(num_cols):", len(num_cols), '\n')
clean_memory()


class LiquidNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.tau = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x, h):
        dx = F.silu(self.W(x) + self.U(h))
        dh = (dx - h) / self.tau
        return h + dh

class LiquidNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2,
                 num_groups=4, dropout_prob=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.liquid_layers = nn.ModuleList(
            [LiquidNeuron(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.GroupNorm(num_groups, hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.readout_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        for i in range(self.num_layers):
            residual = x  
            h = self.liquid_layers[i](x, h)
            x = h
            
            if residual.shape == h.shape:
                h = h + residual  
            h = self.layer_norms[i](h)
            h = self.dropout(h)
    
        return F.softplus(self.readout_linear(h)).squeeze(1)


class CFG:
    folds = 3

class LNNRegressor:
    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_layers: int = 3,
        num_groups: int = 4,
        dropout_prob: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_ratio: float = 0.2,
        clip_grad_norm: bool = True,
        max_norm: float = 0.1,
        batch_size: int = 64,
        max_epochs: int = 25,
        patience: int = 25,
        noise_std: float = 1e-5,
        random_state: int = 0,
        device: str = 'cuda:0',
        compile_model: bool = False,
        verbose: bool = True
    ):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_groups = num_groups
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.clip_grad_norm = clip_grad_norm
        self.max_norm = max_norm
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.noise_std = noise_std
        self.random_state = random_state
        self.device = torch.device(device)
        self.compile_model = compile_model
        self.verbose = verbose

    def fit(
        self,
        X: pd.DataFrame,
        y: np.array,
        eval_set: Tuple[pd.DataFrame, np.array]
    ):
        # PREPROCESS DATA.
        X_cat_train, X_cont_train, X_cat_cont_train, cat_cardinalities, y_train = self._preprocess_data(X, y, training=True)
        X_cat_val, X_cont_val, X_cat_cont_val, _, y_val = self._preprocess_data(eval_set[0], eval_set[1], training=False)

        # CREATE MODEL & TRAINING ALGO.
        self.model = LiquidNetwork(input_dim=X_cat_cont_train.shape[1], hidden_dim=self.hidden_dim,
                                   output_dim=self.output_dim, num_layers=self.num_layers,
                                   num_groups=self.num_groups, dropout_prob=self.dropout_prob).to(self.device)
        
        total_steps = self.max_epochs * math.ceil(len(X) / self.batch_size)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps=total_steps,
                                                        pct_start=self.warmup_ratio, cycle_momentum=False)
        if self.compile_model:
            self.model = torch.compile(self.model)

        loss_fn = torch.nn.MSELoss().to(self.device)

        # TRAIN & TEST MODEL.
        best = {
            'epoch': -1,
            'eval_loss': math.inf,
            'model_state_dict': None,
        }
        remaining_patience = self.patience
        epoch_size = math.ceil(len(X) / self.batch_size)   
        for epoch in range(self.max_epochs):
            # TRAIN.
            self.model.train()
#            optimizer.zero_grad()  # Accumulate gradients during epoch
            train_losses = []
            progress_bar = torch.randperm(len(y_train), device=self.device).split(self.batch_size)
            progress_bar = tqdm(progress_bar, desc=f"Epoch {epoch+1}", total=epoch_size) if self.verbose else progress_bar
            for batch_idx in progress_bar:
                optimizer.zero_grad()  # Normal mode
                with torch.amp.autocast(device_type='cuda', dtype = torch.bfloat16):
                    y_pred = self.model(
                        X_cat_cont_train[batch_idx],
                    ).squeeze(-1).float()
                    loss = loss_fn(y_pred, y_train[batch_idx])
                    
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())

            # EVALUATE.
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    for batch_idx in torch.arange(0, len(y_val), self.batch_size, device=self.device):
                        y_pred = self.model(
                            X_cat_cont_val[batch_idx:batch_idx+self.batch_size],
                        ).squeeze(-1).float()

                        loss = loss_fn(y_pred, y_val[batch_idx:batch_idx+self.batch_size])
                        val_losses.append(loss.item())

            # PRINT INFO.
            mean_train_loss = np.mean(train_losses)
            mean_val_loss = np.mean(val_losses)      
            if self.verbose:
                print(f'Epoch {epoch+1} | Train Loss: {mean_train_loss} | Val Loss: {mean_val_loss}')

            # COMPARE TO BEST.
            if mean_val_loss < best['eval_loss']:
                best['epoch'] = epoch
                best['eval_loss'] = mean_val_loss
                best['model_state_dict'] = self.model.state_dict()
                remaining_patience = self.patience
                if self.verbose:
                    print("ðŸŒ¸ New best epoch! ðŸŒ¸")
            else:
                remaining_patience -= 1

            # EARLY STOPPING.
            if remaining_patience == 0:
                print("Early stopping triggered.")
                break

        # RESTORE BEST MODEL.
        self.model.load_state_dict(best['model_state_dict'])

    def predict(
        self,
        X: pd.DataFrame,
        batch_size: Optional[int] = 8096
    ) -> np.ndarray:
        # PREPROCESS DATA.
        X_cat, X_cont, X_cat_cont, _, _ = self._preprocess_data(X, y=None, training=False)

        # PREDICT.
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for batch_idx in torch.arange(0, len(X), batch_size, device=self.device):
                y_pred.append(
                    self.model(
                        X_cat_cont[batch_idx:batch_idx+batch_size],
                    ).squeeze(-1).float().cpu().numpy()
                )
        y_pred = np.concatenate(y_pred)

        # DENORMALIZE PREDS.
        y_pred = np.expm1(y_pred)
        y_pred = np.clip(y_pred, 0, None)

        return y_pred

    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series, training: bool):
        # PICK NON-CONSTANT COLUMNS.
        if training:
            self._non_constant_columns = X.columns[X.nunique() > 1]
        X = X[self._non_constant_columns]

        # SEPARATE CATEGORICAL & CONTINUOUS FEATURES.
        categorical_features = [col for col in X.columns if X[col].dtype.name in ['object', 'category']]
        X_cat = X[categorical_features].to_numpy()
        X_cont = X.drop(columns=categorical_features).to_numpy()

        # ENCODE CATEGORICAL FEATURES.
        cat_cardinalities = [X[col].nunique()+1 for col in categorical_features] if training else None
        if training:
            self._categorical_encoders = [
                OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(X_cat[:, i:i+1])
                for i in range(X_cat.shape[1])
            ]  
        X_cat = np.concatenate([
            self._categorical_encoders[i].transform(X_cat[:, i:i+1])
            for i in range(X_cat.shape[1])
        ], axis=1)
        X_cat[X_cat < 0] = 0

        # SCALE CONTINUOUS FEATURES.
        if training:
            noise = (
                np.random.default_rng(self.random_state)
                .normal(0.0, self.noise_std, X_cont.shape)
                .astype(X_cont.dtype)
            )
            self._cont_feature_preprocessor = MinMaxScaler(
            ).fit(X_cont + noise)    
        X_cont = self._cont_feature_preprocessor.transform(X_cont)

        # NORMALIZE TARGETS.
        if y is not None:
            y = np.log1p(y)

        # CONVERT TO TENSORS.
        X_cat = torch.tensor(X_cat, dtype=torch.long, device=self.device)
        X_cont = torch.tensor(X_cont, dtype=torch.float32, device=self.device)
        X_cat_cont = torch.cat([X_cat, X_cont], dim=1)
        if y is not None:
            y = torch.tensor(y, dtype=torch.float32, device=self.device)

        return X_cat, X_cont, X_cat_cont, cat_cardinalities, y

    def plot_model_arch(self, X: pd.DataFrame):
        # BUILD & SAVE ARCH IMAGE.
        X_cat, X_cont, X_cat_cont, _, _ = self._preprocess_data(X, y=None, training=False)
        sample_input = torch.randn(1, X_cat_cont.shape[1], device=self.device)
        
        self.model.eval()
        output = self.model(sample_input).squeeze(-1).float()
        make_dot(output, params=dict(self.model.named_parameters())).render("model_architecture", format="png", cleanup=True)
        
        # READ ARCH IMAGE.
        img = mpimg.imread('/kaggle/working/model_architecture.png')
        plt.figure(figsize=(12, 20))
        plt.imshow(img)
        plt.axis('off') 
        plt.show()


%%time
def train_model(X, y):
    print("Data shape:", X.shape, "\n")
    kf = KFold(n_splits=CFG.folds, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    models = []
    
    for fi, (train_idx, valid_idx) in enumerate(kf.split(X)):
        print("#"*25)
        print(f"### Fold {fi+1}/{CFG.folds} ...")
        print("#"*25)

        model = LNNRegressor()
        model.fit(X.iloc[train_idx], y.iloc[train_idx].to_numpy(),
                  eval_set=(X.iloc[valid_idx], y.iloc[valid_idx].to_numpy()))
        models.append(model)

        oof_pred = model.predict(X.iloc[valid_idx])
        m = np.round(np.sqrt(np.mean((np.log1p(oof_pred) - np.log1p(y.iloc[valid_idx]))**2)),4)
        print(f"{Fore.GREEN}{Style.BRIGHT}\nFold {fi+1} | score: {m:.4f}{Style.RESET_ALL}\n")
        oof[valid_idx] = oof_pred
        
    m_all = np.round(np.sqrt(np.mean((np.log1p(oof) - np.log1p(y))**2)),4)
    print(f"{Fore.BLUE}{Style.BRIGHT}Overall CV score: {m_all:.4f}{Style.RESET_ALL}\n")
    model.plot_model_arch(X)
    return models, oof

models, oof = train_model(X, y)
oof_df = pd.DataFrame({'id': train_id, 'oof_pred': oof})
oof_df.to_csv('oof_preds.csv', index=False)
clean_memory()


# Inference
class AvgModel:
    def __init__(self, models: list[BaseEstimator]):
        self.models = models
    def predict(self, X: ArrayLike):
        preds = []
        for model in self.models:
            pred = model.predict(X)
            preds.append(pred)
        return np.mean(preds, axis=0)

avg_model = AvgModel(models)
test_pred = avg_model.predict(X_test)


sub = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
sub.Calories = test_pred
sub.to_csv("submission.csv", index=False)
sub.head()




