%%time
!pip install -q torch==2.4.0 torchviz
!pip install -q -U tabm==0.0.1.dev0 --no-index --find-links=/kaggle/input/tabm-tabular-dl-library


import os
import gc
import sys
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
import tabm
import rtdl_num_embeddings
from torchviz import make_dot
sys.path.append("/kaggle/input/tabm-tabular-dl-library")
from tabm_reference import Model, make_parameter_groups

warnings.filterwarnings('ignore')
print("PyTorch version:", torch.__version__)
print("TabM version:", tabm.__version__)


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


class CFG:
    folds = 3

class TabMRegressor:
    def __init__(
        self,
        arch_type: str = 'tabm-mini',
        backbone: dict = {'type': 'MLP', 'n_blocks': 3, 'd_block': 512,
                          'activation': 'PReLU', 'dropout': 0.1},
        bin_count: int = 52,
        d_embedding: int = 64,
        activation: bool = False,
        version: str = 'B',
        k: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 5e-3,
        warmup_ratio: float = 0.2,
        clip_grad_norm: bool = True,
        max_norm: float = 0.1,
        batch_size: int = 32,
        max_epochs: int = 6,
        patience: int = 3,
        noise_std: float = 1e-5,
        random_state: int = 0,
        device: str = 'cuda:0',
        compile_model: bool = False,
        verbose: bool = True
    ):
        self.arch_type = arch_type
        self.backbone = backbone
        self.bin_count = bin_count
        self.d_embedding = d_embedding
        self.activation = activation
        self.version = version
        self.k = k
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
        X_cat_train, X_cont_train, cat_cardinalities, y_train = self._preprocess_data(X, y, training=True)
        X_cat_val, X_cont_val, _, y_val = self._preprocess_data(eval_set[0], eval_set[1], training=False)

        # CREATE MODEL & TRAINING ALGO.
        bins = rtdl_num_embeddings.compute_bins(X_cont_train, n_bins=self.bin_count)
        self.model = Model(
            n_num_features=X_cont_train.shape[1],
            cat_cardinalities=cat_cardinalities,
            n_classes=None,
            backbone=self.backbone,
            bins=bins,
            num_embeddings=(
                None
                if bins is None
                else {
                    'type': 'PiecewiseLinearEmbeddings',
                    'd_embedding': self.d_embedding,
                    'activation': self.activation,
                    'version': self.version,
                }
            ),
            arch_type=self.arch_type,
            k=self.k,
        ).to(self.device)
        
        total_steps = self.max_epochs * math.ceil(len(X) / self.batch_size)
        optimizer = torch.optim.AdamW(make_parameter_groups(self.model), lr=self.learning_rate, weight_decay=self.weight_decay)
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
            optimizer.zero_grad()  # Accumulate gradients during epoch
            train_losses = []
            progress_bar = torch.randperm(len(y_train), device=self.device).split(self.batch_size)
            progress_bar = tqdm(progress_bar, desc=f"Epoch {epoch+1}", total=epoch_size) if self.verbose else progress_bar
            for batch_idx in progress_bar:
#                optimizer.zero_grad()  # Normal mode
                with torch.amp.autocast(device_type='cuda', dtype = torch.bfloat16):
                    y_pred = self.model(
                        X_cont_train[batch_idx],
                        X_cat_train[batch_idx],
                    ).squeeze(-1).float()
                    loss = loss_fn(y_pred.flatten(0, 1), y_train[batch_idx].repeat_interleave(self.k))

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
                            X_cont_val[batch_idx:batch_idx+self.batch_size],
                            X_cat_val[batch_idx:batch_idx+self.batch_size],
                        ).squeeze(-1).float()

                        loss = loss_fn(y_pred.flatten(0, 1), y_val[batch_idx:batch_idx+self.batch_size].repeat_interleave(self.k))
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
                break

        # RESTORE BEST MODEL.
        self.model.load_state_dict(best['model_state_dict'])

    def predict(
        self,
        X: pd.DataFrame,
        batch_size: Optional[int] = 8096
    ) -> np.ndarray:
        # PREPROCESS DATA.
        X_cat, X_cont, _, _ = self._preprocess_data(X, y=None, training=False)

        # PREDICT.
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for batch_idx in torch.arange(0, len(X), batch_size, device=self.device):
                y_pred.append(
                    self.model(
                        X_cont[batch_idx:batch_idx+batch_size],
                        X_cat[batch_idx:batch_idx+batch_size],
                    ).squeeze(-1).float().cpu().numpy()
                )
        y_pred = np.concatenate(y_pred)

        # DENORMALIZE PREDS.
        y_pred = np.expm1(y_pred)
        y_pred = np.clip(y_pred, 0, None)

        # COMPUTE ENSEMBLE MEAN.
        y_pred = np.mean(y_pred, axis=1)

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
        if y is not None:
            y = torch.tensor(y, dtype=torch.float32, device=self.device)

        return X_cat, X_cont, cat_cardinalities, y

    def plot_model_arch(self, X: pd.DataFrame):
        # BUILD & SAVE ARCH IMAGE.
        X_cat, X_cont, cat_cardinalities, _ = self._preprocess_data(X, y=None, training=True)
        dummy_cont = torch.randn(1, X_cont.shape[1], device=self.device)
        dummy_cat = torch.tensor([[np.random.randint(0, card) for card in cat_cardinalities]], device=self.device)
        
        self.model.eval()
        output = self.model(dummy_cont, dummy_cat).squeeze(-1).float()
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

        model = TabMRegressor()
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




