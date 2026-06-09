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


import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar, StochasticWeightAveraging
from lifelines.utils import concordance_index
from pytorch_tabular.models.common.layers import ODST
from torch import nn
from pytorch_lightning.utilities import grad_norm
import functools
from tqdm import tqdm  # Import tqdm for progress tracking

# ---------------------------
# Model definitions
# ---------------------------
class CatEmbeddings(nn.Module):
    """
    Embedding module for categorical features.
    """
    def __init__(self, projection_dim: int, categorical_cardinality: list, embedding_dim: int):
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
        # Get an embedding for each categorical feature and then concatenate
        x_cat = [embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)]
        x_cat = torch.cat(x_cat, dim=1)
        return self.projection(x_cat)

class NN(nn.Module):
    """
    Neural network that combines categorical embeddings and continuous data.
    """
    def __init__(self, continuous_dim: int, categorical_cardinality: list,
                 embedding_dim: int, projection_dim: int, hidden_dim: int,
                 dropout: float = 0):
        super(NN, self).__init__()
        self.embeddings = CatEmbeddings(projection_dim, categorical_cardinality, embedding_dim)
        self.mlp = nn.Sequential(
            ODST(projection_dim + continuous_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Weight initialization (xavier for Linear layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x_cat, x_cont):
        x = self.embeddings(x_cat)
        x = torch.cat([x, x_cont], dim=1)
        x = self.dropout(x)
        x = self.mlp(x)
        return self.out(x), x

@functools.lru_cache(maxsize=None)
def combinations(N):
    """
    Cache and return all possible 2-combinations of indices from 0 to N-1.
    """
    ind = torch.arange(N)
    comb = torch.combinations(ind, r=2)
    return comb.cuda()

class LitNN(pl.LightningModule):
    """
    LightningModule that wraps our neural network and defines losses,
    training, validation, and test steps.
    """
    def __init__(self, continuous_dim: int, categorical_cardinality: list,
                 embedding_dim: int, projection_dim: int, hidden_dim: int,
                 lr: float = 1e-3, dropout: float = 0.2, weight_decay: float = 1e-3,
                 aux_weight: float = 0.1, margin: float = 0.5, race_index: int = 0):
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
        aux_mask = efs == 1
        aux_loss = (aux_loss * aux_mask).sum() / aux_mask.sum()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("race_loss", race_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("aux_loss", aux_loss, on_epoch=True, prog_bar=True, logger=True)
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
        y_sign = 2 * (y_left > y_right).int() - 1
        loss = nn.functional.relu(-y_sign * (pred_left - pred_right) + self.hparams.margin)
        mask = self.get_mask(comb, efs, y_left, y_right)
        loss = (loss.double() * mask.double()).sum() / mask.sum()
        return loss
    
    def get_mask(self, comb, efs, y_left, y_right):
        left_outlived = y_left >= y_right
        left_1_right_0 = (efs[comb[:, 0]] == 1) & (efs[comb[:, 1]] == 0)
        mask2 = (left_outlived & left_1_right_0)
        right_outlived = y_right >= y_left
        right_1_left_0 = (efs[comb[:, 1]] == 1) & (efs[comb[:, 0]] == 0)
        mask2 |= (right_outlived & right_1_left_0)
        mask2 = ~mask2
        return mask2
    
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
    
    def on_test_epoch_end(self):
        cindex, metric = self._calc_cindex()
        self.log("test_cindex", metric, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
        self.targets.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay)
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=45, eta_min=6e-3
            ),
            "interval": "epoch",
            "frequency": 1,
            "strict": False,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

# ---------------------------
# Pipeline class definition
# ---------------------------
class HCTSurvicePredictionPipelineMLModel:
    def __init__(self, hparams=None):
        # Set default hyperparameters if none are provided
        self.hparams = hparams or {
            "embedding_dim": 16,
            "projection_dim": 112,
            "hidden_dim": 56,
            "lr": 0.06864861983337984,
            "dropout": 0.05463240181423116,
            "aux_weight": 0.26545778308743806,
            "margin": 0.2588153271003354,
            "weight_decay": 0.0002773544957610778
        }
        pl.seed_everything(42)
        self.train_df = None
        self.test_df = None
        self.test_pred = None  # To store final test predictions
        self.categorical_cols = None
        self.numerical_cols = None

    # ---------------------------
    # Data utility methods
    # ---------------------------
    def _get_X_cat(self, df, cat_cols, transformers=None):
        """
        Transform categorical columns with LabelEncoder(s). If no transformers are provided, they are fitted.
        """
        if transformers is None:
            transformers = [LabelEncoder().fit(df[col]) for col in cat_cols]
        X_cat = np.array([
            transformer.transform(df[col])
            for col, transformer in zip(cat_cols, transformers)
        ]).T
        return transformers, X_cat

    def _get_feature_types(self, df):
        """
        Return list of categorical columns and numerical columns.
        """
        categorical_cols = [col for col in df.columns if (df[col].dtype == "object") or (2 < df[col].nunique() < 25)]
        RMV = ["ID", "efs", "efs_time", "y"]
        FEATURES = [col for col in df.columns if col not in RMV]
        print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
        numerical = [col for col in FEATURES if col not in categorical_cols]
        return categorical_cols, numerical

    def _add_features(self, df):
        """
        Create new features.
        """
        df['is_cyto_score_same'] = (df['cyto_score'] == df['cyto_score_detail']).astype(int)
        df['year_hct'] = df['year_hct'] - 2000  # Adjust the year
        return df

    def _load_data(self):
        """
        Load train and test data and apply feature engineering.
        """
        self.test_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
        self.test_df = self._add_features(self.test_df)
        print("Test shape:", self.test_df.shape)
        
        self.train_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
        self.train_df = self._add_features(self.train_df)
        print("Train shape:", self.train_df.shape)

    def _get_categoricals(self, train, val):
        """
        Remove constant categorical columns, adjust unseen categories in validation,
        and transform categoricals using LabelEncoder.
        Returns:
            X_cat_train, X_cat_val, numerical columns list, transformers, categorical column names.
        """
        categorical_cols, numerical = self._get_feature_types(train)
        remove = []
        for col in categorical_cols:
            if train[col].nunique() == 1:
                remove.append(col)
            # For validation, set unseen categories to NaN
            ind = ~val[col].isin(train[col])
            if ind.any():
                val.loc[ind, col] = np.nan
        categorical_cols = [col for col in categorical_cols if col not in remove]
        transformers, X_cat_train = self._get_X_cat(train, categorical_cols)
        _, X_cat_val = self._get_X_cat(val, categorical_cols, transformers)
        return X_cat_train, X_cat_val, numerical, transformers, categorical_cols

    def _init_dl(self, X_cat, X_num, df, training=False):
        """
        Create a DataLoader from categorical data, numerical data, and target columns.
        Notice that `efs_time` is log-transformed.
        """
        ds = TensorDataset(
            torch.tensor(X_cat, dtype=torch.long),
            torch.tensor(X_num, dtype=torch.float32),
            torch.tensor(df.efs_time.values, dtype=torch.float32).log(),
            torch.tensor(df.efs.values, dtype=torch.long)
        )
        bs = 2048
        return torch.utils.data.DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=training)

    def _preprocess_data(self, train, val):
        """
        Preprocess data: transform categoricals and standardize numerical features.
        Returns:
            X_cat_val, X_num_train, X_num_val, train DataLoader, validation DataLoader,
            transformers, numerical column names, categorical column names.
        """
        X_cat_train, X_cat_val, numerical, transformers, categorical_cols = self._get_categoricals(train, val)
        scaler = StandardScaler()
        imp = SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=True)
        
        X_num_train = imp.fit_transform(train[numerical])
        X_num_train = scaler.fit_transform(X_num_train)
        X_num_val = imp.transform(val[numerical])
        X_num_val = scaler.transform(X_num_val)
        
        dl_train = self._init_dl(X_cat_train, X_num_train, train, training=True)
        dl_val = self._init_dl(X_cat_val, X_num_val, val)
        return X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers, numerical, categorical_cols

    # ---------------------------
    # Training and prediction
    # ---------------------------
    def train_model(self):
        """
        Perform cross-validation training.
        For each fold, train a model and predict on the test set.
        Predictions are averaged over all folds.
        """
        # Prepare the test set (dummy target values)
        test = self.test_df.copy()
        test['efs_time'] = 1
        test['efs'] = 1
        self.test_pred = np.zeros(test.shape[0])
        
        # Determine feature types from the full training set
        self.categorical_cols, self.numerical_cols = self._get_feature_types(self.train_df)
        
        # Create a stratification column (using race_group and a newborn indicator)
        stratify_col = self.train_df.race_group.astype(str) + (self.train_df.age_at_hct == 0.044).astype(str)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Wrap the CV folds loop with tqdm for progress tracking
        for fold, (train_index, val_index) in enumerate(tqdm(kf.split(self.train_df, stratify_col), total=5, desc="CV Folds")):
            print(f"\nStarting Fold {fold + 1}")
            train_fold = self.train_df.iloc[train_index].copy()
            val_fold = self.train_df.iloc[val_index].copy()
            
            # Preprocess data for the fold
            X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers, numerical, categorical_cols = self._preprocess_data(train_fold, val_fold)
            
            # Train the model on this fold
            model = self.train_final(X_num_train, dl_train, dl_val, transformers, categorical_cols)
            
            # Preprocess test data using the current training fold for consistency
            X_cat_test, _, X_num_test, _, _, _, _, _ = self._preprocess_data(train_fold, test)
            model = model.cuda().eval()
            with torch.no_grad():
                X_cat_test_tensor = torch.tensor(X_cat_test, dtype=torch.long).cuda()
                X_num_test_tensor = torch.tensor(X_num_test, dtype=torch.float32).cuda()
                pred, _ = model(X_cat_test_tensor, X_num_test_tensor)
                pred = pred.detach().cpu().numpy()
            self.test_pred += pred
        
        # Average predictions across folds
        self.test_pred /= 5

    def train_final(self, X_num_train, dl_train, dl_val, transformers, categorical_cols):
        """
        Train the final model for one fold.
        """
        model = LitNN(
            continuous_dim=X_num_train.shape[1],
            categorical_cardinality=[len(t.classes_) for t in transformers],
            race_index=categorical_cols.index("race_group") if "race_group" in categorical_cols else 0,
            **self.hparams
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

    def save_submission(self, filename='submission.csv'):
        """
        Save the submission file using the averaged predictions.
        """
        subm_data = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
        subm_data['prediction'] = -self.test_pred  # Note: predictions are negated as in the original code
        subm_data.to_csv(filename, index=False)
        print("Submission head:")
        print(subm_data.head())

    # ---------------------------
    # Pipeline runner
    # ---------------------------
    def run_pipeline(self):
        """
        Run the entire pipeline step by step: load data, train the model, and save the submission.
        A tqdm progress bar is used to indicate high-level pipeline progress.
        """
        steps = [
            ("Loading Data", self._load_data),
            ("Training Model (CV)", self.train_model),
            ("Saving Submission", self.save_submission)
        ]
        for step_name, func in tqdm(steps, desc="Pipeline Steps", total=len(steps)):
            tqdm.write(f"Step '{step_name}' started...")
            func()
            tqdm.write(f"Step '{step_name}' completed.")




# ---------------------------
# Run the pipeline
# ---------------------------
if __name__ == '__main__':
    pipeline = HCTSurvicePredictionPipelineMLModel()
    pipeline.run_pipeline()
    print("Pipeline execution completed.")



import pandas as pd
from scipy.stats import rankdata

# Load submission files
sub1 = pd.read_csv('/kaggle/input/cibmtr-ensemble/submission1.csv')
sub2 = pd.read_csv('/kaggle/input/cibmtr-ensemble/submission2.csv')
sub3 = pd.read_csv('/kaggle/input/cibmtr-ensemble/submission3.csv')
sub4 = pd.read_csv('/kaggle/input/cibmtr-ensemble/submission.csv')

# Calculate ranks for each submission's predictions<sup data-citation="1" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://www.programcreek.com/python/example/57193/scipy.rankdata">1</a></sup><sup data-citation="3" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html">3</a></sup>
rank1 = rankdata(sub1['prediction'], method='average')
rank2 = rankdata(sub2['prediction'], method='average') 
rank3 = rankdata(sub3['prediction'], method='average')
rank4 = rankdata(sub4['prediction'], method='average')

# Create DataFrame of ranks and average them<sup data-citation="1" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://www.programcreek.com/python/example/57193/scipy.rankdata">1</a></sup>
rank_df = pd.DataFrame({
    'rank1': rank1,
    'rank2': rank2,
    'rank3': rank3,
    'rank4': rank4
})
ensemble_rank = rank_df.mean(axis=1)

# Create final submission file with averaged ranks<sup data-citation="3" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html">3</a></sup>
final_sub = sub1[['ID']].copy()
final_sub['prediction'] = ensemble_rank
final_sub.to_csv('submission.csv', index=False)

