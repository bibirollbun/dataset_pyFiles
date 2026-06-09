!pip install kaggle catboost optuna

import os
import numpy as np
import random
import torch
import pandas as pd
import optuna
from optuna.samplers import TPESampler

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(42)


!pip install colorama


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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset
from warnings import filterwarnings

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
        ind = ~val[col].isin(train[col])
        if ind.any():
            val.loc[ind, col] = np.nan
    categorical_cols = [col for col in categorical_cols if col not in remove]
    transformers, X_cat_train = get_X_cat(train, categorical_cols)
    _, X_cat_val = get_X_cat(val, categorical_cols, transformers)
    return X_cat_train, X_cat_val, numerical, transformers


def init_dl(X_cat, X_num, df, training=False):
    """
    Initialize data loaders with 4 dimensions : categorical dataframe, numerical dataframe and target values (efs and efs_time).
    Notice that efs_time is log-transformed.
    Fix batch size to 2048 and return dataloader for training or validation depending on training value.
    """
    ds_train = TensorDataset(
        torch.tensor(X_cat, dtype=torch.long),
        torch.tensor(X_num, dtype=torch.float32),
        torch.tensor(df.efs_time.values, dtype=torch.float32).log(),
        torch.tensor(df.efs.values, dtype=torch.long)
    )
    bs = 2048
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=bs, pin_memory=True, shuffle=training)
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
    """
    Create some new features to help the model focus on specific patterns.
    """
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


class CatEmbeddings(nn.Module):
    """
    Embedding module for the categorical dataframe.
    """
    def __init__(
        self,
        projection_dim: int,
        categorical_cardinality: List[int],
        embedding_dim: int
    ):
        """
        projection_dim: The dimension of the final output after projecting the concatenated embeddings into a lower-dimensional space.
        categorical_cardinality: A list where each element represents the number of unique categories (cardinality) in each categorical feature.
        embedding_dim: The size of the embedding space for each categorical feature.
        self.embeddings: list of embedding layers for each categorical feature.
        self.projection: sequential neural network that goes from the embedding to the output projection dimension with GELU activation.
        """
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
        """
        Apply the projection on concatened embeddings that contains all categorical features.
        """
        x_cat = [embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)]
        x_cat = torch.cat(x_cat, dim=1)
        return self.projection(x_cat)


class NN(nn.Module):
    """
    Train a model on both categorical embeddings and numerical data.
    """
    def __init__(
            self,
            continuous_dim: int,
            categorical_cardinality: List[int],
            embedding_dim: int,
            projection_dim: int,
            hidden_dim: int,
            dropout: float = 0
    ):
        """
        continuous_dim: The number of continuous features.
        categorical_cardinality: A list of integers representing the number of unique categories in each categorical feature.
        embedding_dim: The dimensionality of the embedding space for each categorical feature.
        projection_dim: The size of the projected output space for the categorical embeddings.
        hidden_dim: The number of neurons in the hidden layer of the MLP.
        dropout: The dropout rate applied in the network.
        self.embeddings: previous embeddings for categorical data.
        self.mlp: defines an MLP model with an ODST layer followed by batch normalization and dropout.
        self.out: linear output layer that maps the output of the MLP to a single value
        self.dropout: defines dropout
        Weights initialization with xavier normal algorithm and biases with zeros.
        """
        super(NN, self).__init__()
        self.embeddings = CatEmbeddings(projection_dim, categorical_cardinality, embedding_dim)
        self.mlp = nn.Sequential(
            ODST(projection_dim + continuous_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_cat, x_cont):
        """
        Create embedding layers for categorical data, concatenate with continous variables.
        Add dropout and goes through MLP and return raw output and 1-dimensional output as well.
        """
        x = self.embeddings(x_cat)
        x = torch.cat([x, x_cont], dim=1)
        x = self.dropout(x)
        x = self.mlp(x)
        return self.out(x), x


@functools.lru_cache
def combinations(N):
    """
    calculates all possible 2-combinations (pairs) of a tensor of indices from 0 to N-1,
    and caches the result using functools.lru_cache for optimization
    """
    ind = torch.arange(N)
    comb = torch.combinations(ind, r=2)
    return comb.cuda()


class LitNN(pl.LightningModule):
    """
    Main Model creation and losses definition to fully train the model.
    """
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
        """
        continuous_dim: The number of continuous input features.
        categorical_cardinality: A list of integers, where each element corresponds to the number of unique categories for each categorical feature.
        embedding_dim: The dimension of the embeddings for the categorical features.
        projection_dim: The dimension of the projected space after embedding concatenation.
        hidden_dim: The size of the hidden layers in the feedforward network (MLP).
        lr: The learning rate for the optimizer.
        dropout: Dropout probability to avoid overfitting.
        weight_decay: The L2 regularization term for the optimizer.
        aux_weight: Weight used for auxiliary tasks.
        margin: Margin used in some loss functions.
        race_index: An index that refer to race_group in the input data.
        """
        super(LitNN, self).__init__()
        self.save_hyperparameters()

        # Creates an instance of the NN model defined above
        self.model = NN(
            continuous_dim=self.hparams.continuous_dim,
            categorical_cardinality=self.hparams.categorical_cardinality,
            embedding_dim=self.hparams.embedding_dim,
            projection_dim=self.hparams.projection_dim,
            hidden_dim=self.hparams.hidden_dim,
            dropout=self.hparams.dropout
        )
        self.targets = []

        # Defines a small feedforward neural network that performs an auxiliary task with 1-dimensional output
        self.aux_cls = nn.Sequential(
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim // 3),
            nn.GELU(),
            nn.Linear(self.hparams.hidden_dim // 3, 1)
        )

    def on_before_optimizer_step(self, optimizer):
        """
        Compute the 2-norm for each layer
        If using mixed precision, the gradients are already unscaled here
        """
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def forward(self, x_cat, x_cont):
        """
        Forward pass that outputs the 1-dimensional prediction and the embeddings (raw output)
        """
        x, emb = self.model(x_cat, x_cont)
        return x.squeeze(1), emb

    def training_step(self, batch, batch_idx):
        """
        defines how the model processes each batch of data during training.
        A batch is a combination of : categorical data, continuous data, efs_time (y) and efs event.
        y_hat is the efs_time prediction on all data and aux_pred is auxiliary prediction on embeddings.
        Calculates loss and race_group loss on full data.
        Auxiliary loss is calculated with an event mask, ignoring efs=0 predictions and taking the average.
        Returns loss and aux_loss multiplied by weight defined above.
        """
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        aux_pred = self.aux_cls(emb).squeeze(1)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        aux_loss = nn.functional.mse_loss(aux_pred, y, reduction='none')
        aux_mask = efs == 1
        aux_loss = (aux_loss * aux_mask).sum() / aux_mask.sum()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("race_loss", race_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        self.log("aux_loss", aux_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        return loss + aux_loss * self.hparams.aux_weight

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
        """
        Most important part of the model : loss function used for training.
        We face survival data with event indicators along with time-to-event.

        This function computes the main loss by the following the steps :
        * create all data pairs with "combinations" function (= all "two subjects" combinations)
        * make sure that we have at least 1 event in each pair
        * convert y to +1 or -1 depending on the correct ranking
        * loss is computed using a margin-based hinge loss
        * mask is applied to ensure only valid pairs are being used (censored data can't be ranked with event in some cases)
        * average loss on all pairs is returned
        """
        N = y.shape[0]
        comb = combinations(N)
        comb = comb[(efs[comb[:, 0]] == 1) | (efs[comb[:, 1]] == 1)]
        pred_left = y_hat[comb[:, 0]]
        pred_right = y_hat[comb[:, 1]]
        y_left = y[comb[:, 0]]
        y_right = y[comb[:, 1]]
        y = 2 * (y_left > y_right).int() - 1
        loss = nn.functional.relu(-y * (pred_left - pred_right) + self.hparams.margin)
        mask = self.get_mask(comb, efs, y_left, y_right)
        loss = (loss.double() * (mask.double())).sum() / mask.sum()
        return loss

    def get_mask(self, comb, efs, y_left, y_right):
        """
        Defines all invalid comparisons :
        * Case 1: "Left outlived Right" but Right is censored
        * Case 2: "Right outlived Left" but Left is censored
        Masks for case 1 and case 2 are combined using |= operator and inverted using ~ to create a "valid pair mask"
        """
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
        """
        This method defines how the model processes each batch during validation
        """
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        self.targets.append([y, y_hat.detach(), efs, x_cat[:, self.hparams.race_index]])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
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
        """
        configures the optimizer and learning rate scheduler:
        * Optimizer: Adam optimizer with weight decay (L2 regularization).
        * Scheduler: Cosine Annealing scheduler, which adjusts the learning rate according to a cosine curve.
        """
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


import json
import pytorch_lightning as pl
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.callbacks import StochasticWeightAveraging
from sklearn.model_selection import StratifiedKFold
from metric import score, custom_score

pl.seed_everything(42)

# Directory to save models
MODEL_DIR = "/kaggle/input/cibmtr-final-models/CIBMTR final models/prlnn_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_predict():
    test, train_original = load_data()
    test['efs_time'] = 1
    test['efs'] = 1
    categorical_cols, numerical = get_feature_types(train_original)

    #FOLDS = 10
    #SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    FOLDS = 10
    SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 22, 34, 47, 53, 66, 71, 85, 91, 104, 118, 125, 137, 149, 156, 167, 178, 189, 195, 203]

    oof_nn = np.zeros(len(train_original))
    test_pred_nn = np.zeros(len(test))

    for seed in SEEDS:
        for fold in range(FOLDS):
            print(f"Loading model for Seed {seed}, Fold {fold}")
            model_path = os.path.join(MODEL_DIR, f"model_seed{seed}_fold{fold}.pth")

            hparams = {
            "embedding_dim": 32,
            "projection_dim": 112,
            "hidden_dim": 112,
            "lr": 0.06464861983337984,
            "dropout": 0.05463240181423116,
            "aux_weight": 0.5,
            "margin": 0.2588153271003354,
            "weight_decay": 0.0002773544957610778
        }

            train_original_copy = train_original.copy()
            train_original_copy['y_bins'] = pd.cut(train_original['efs_time'], bins=100, labels=False)
            train_original_copy['stratify_col'] = train_original['race_group'].astype(str) + '_' + train_original_copy['y_bins'].astype(str)
            skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_original, train_original_copy['stratify_col'])):
                if fold_idx != fold:
                    continue
                train = train_original.iloc[train_idx]
                val = train_original.iloc[val_idx]

                X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers = preprocess_data(train, val)

                model = LitNN(
                    continuous_dim=X_num_train.shape[1],
                    categorical_cardinality=[len(t.classes_) for t in transformers],
                    race_index=categorical_cols.index("race_group"),
                    **hparams
                )

                model.load_state_dict(torch.load(model_path))
                model.eval()

                val_preds, _ = model.cuda().eval()(
                    torch.tensor(X_cat_val, dtype=torch.long).cuda(),
                    torch.tensor(X_num_val, dtype=torch.float32).cuda()
                )
                oof_nn[val_idx] += val_preds.detach().cpu().numpy() / len(SEEDS)

                X_cat_test, X_num_train, X_num_test, dl_train, dl_test, transformers = preprocess_data(train, test)
                test_preds, _ = model.cuda().eval()(
                    torch.tensor(X_cat_test, dtype=torch.long).cuda(),
                    torch.tensor(X_num_test, dtype=torch.float32).cuda()
                )
                test_pred_nn += test_preds.detach().cpu().numpy() / (FOLDS * len(SEEDS))
    return -oof_nn, -test_pred_nn

test, train = load_data()


# Load models and re-predict
oof_nn_reloaded, test_pred_nn_reloaded = load_and_predict()

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred_reloaded = train[["ID"]].copy()
y_pred_reloaded["prediction"] = oof_nn_reloaded
m_reloaded = score(y_true.copy(), y_pred_reloaded.copy(), "ID")
print(f"\nOverall CV for reloaded models =", m_reloaded)

#assert np.allclose(oof_nn, oof_nn_reloaded), "Mismatch in OOF predictions!"
#assert np.allclose(test_pred_nn, test_pred_nn_reloaded), "Mismatch in test predictions!"
#print("Model reloading successful! Predictions match.")


test_pred_nn_reloaded


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)
train.head()


plt.hist(train.loc[train.efs==1,"efs_time"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to event, or time observed without event.")
plt.legend()
plt.show()


from lifelines import KaplanMeierFitter
def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y
train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')

plt.hist(train.loc[train.efs==1,"y"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"y"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("KaplanMeier Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


train['missing_values_count'] = train.isna().sum(axis=1)
test['missing_values_count'] = test.isna().sum(axis=1)


RMV = ["ID","efs","efs_time","y"]
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


from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
print("Using CatBoost version",cb.__version__)


import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostRegressor
import pandas as pd
import pickle

# Parameters
FOLDS = 10
#SEEDS = [11,12,13,14,15,16,17,18,19,20]  # Multiple random seeds for seed averaging
SEEDS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 33, 41, 52, 64, 73, 82, 95, 101, 112, 126, 135, 148, 159, 171, 183, 194, 205, 217, 229] # Multiple random seeds for seed averaging
NUM_BINS = 100  # Number of bins for discretizing the target variable

# Directory to save models
MODEL_DIR = "/kaggle/input/cibmtr-final-models/CIBMTR final models/cat_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def predict_with_saved_models(test, train, model_dir=MODEL_DIR):
    """Load saved models and perform inference"""
    oof_cat = np.zeros(len(train))
    pred_cat = np.zeros(len(test))

    train['y_bins'] = pd.cut(train['y'], bins=NUM_BINS, labels=False)
    train['stratify_col'] = train['race_group'].astype(str) + '_' + train['y_bins'].astype(str)

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)

        for i, (train_index, test_index) in enumerate(skf.split(train, train['stratify_col'])):
            model_path = os.path.join(model_dir, f"catboost_fold{i}_seed{seed}.pkl")
            with open(model_path, 'rb') as f:
                model_cat = pickle.load(f)
            print(f"Loaded model from {model_path}")

            x_valid = train.loc[test_index, FEATURES].copy()
            x_test = test[FEATURES].copy()

            # Infer OOF
            oof_cat[test_index] += model_cat.predict(x_valid) / len(SEEDS)

            # Infer test
            pred_cat += model_cat.predict(x_test) / (FOLDS * len(SEEDS))

    train.drop(columns=['y_bins', 'stratify_col'], inplace=True)
    return oof_cat, pred_cat


oof_cat, pred_cat = predict_with_saved_models(test, train, model_dir=MODEL_DIR)


from metric import score, custom_score

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_cat
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost KaplanMeier =",m)

race_dict = custom_score(y_true.copy(), y_pred.copy(), "ID", print_info=False)[1]
print(race_dict)


pred_cat


from pathlib import Path
from metric import score
import pandas as pd
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')

ROOT_DATA_PATH = Path(r"/kaggle/input/equity-post-HCT-survival-predictions")

pd.set_option('display.max_columns', 100)

train = pd.read_csv(ROOT_DATA_PATH.joinpath("train.csv"))
test = pd.read_csv(ROOT_DATA_PATH.joinpath("test.csv"))

CATEGORICAL_VARIABLES = [
    # Graft and HCT reasons
    'dri_score', 'graft_type', 'prod_type', 'prim_disease_hct',

    # Patient health status (risk factors)
    'psych_disturb', 'diabetes', 'arrhythmia', 'vent_hist', 'renal_issue', 'pulm_moderate',
    'pulm_severe', 'obesity', 'hepatic_mild', 'hepatic_severe', 'peptic_ulcer', 'rheum_issue',
    'cardiac', 'prior_tumor', 'mrd_hct', 'tbi_status', 'cyto_score', 'cyto_score_detail',

    # Patient demographics
    'ethnicity', 'race_group',

    # Biological matching with donor
    'sex_match', 'donor_related', 'cmv_status', 'tce_imm_match', 'tce_match', 'tce_div_match',

    # Medication/operation related data
    'melphalan_dose', 'rituximab', 'gvhd_proph', 'in_vivo_tcd', 'conditioning_intensity'
]

HLA_COLUMNS = [
    'hla_match_a_low', 'hla_match_a_high',
    'hla_match_b_low', 'hla_match_b_high',
    'hla_match_c_low', 'hla_match_c_high',
    'hla_match_dqb1_low', 'hla_match_dqb1_high',
    'hla_match_drb1_low', 'hla_match_drb1_high',

    # Matching at HLA-A(low), -B(low), -DRB1(high)
    'hla_nmdp_6',
    # Matching at HLA-A,-B,-DRB1 (low or high)
    'hla_low_res_6', 'hla_high_res_6',
    # Matching at HLA-A, -B, -C, -DRB1 (low or high)
    'hla_low_res_8', 'hla_high_res_8',
    # Matching at HLA-A, -B, -C, -DRB1, -DQB1 (low or high)
    'hla_low_res_10', 'hla_high_res_10'
]

OTHER_NUMERICAL_VARIABLES = ['year_hct', 'donor_age', 'age_at_hct', 'comorbidity_score', 'karnofsky_score']
NUMERICAL_VARIABLES = HLA_COLUMNS + OTHER_NUMERICAL_VARIABLES

TARGET_VARIABLES = ['efs_time', 'efs']
ID_COLUMN = ["ID"]


def preprocess_data(df):
    df[CATEGORICAL_VARIABLES] = df[CATEGORICAL_VARIABLES].fillna("Unknown")
    df[OTHER_NUMERICAL_VARIABLES] = df[OTHER_NUMERICAL_VARIABLES].fillna(df[OTHER_NUMERICAL_VARIABLES].median())

    return df

train = preprocess_data(train)
test = preprocess_data(test)


def features_engineering(df):
    # Change year_hct to relative year from 2000
    df['year_hct'] = df['year_hct'] - 2000

    return df


train = features_engineering(train)
test = features_engineering(test)

train[CATEGORICAL_VARIABLES] = train[CATEGORICAL_VARIABLES].astype('category')
test[CATEGORICAL_VARIABLES] = test[CATEGORICAL_VARIABLES].astype('category')

FEATURES = train.drop(columns=['ID', 'efs', 'efs_time']).columns.tolist()


import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Parameters
FOLDS = 10
#SEEDS = [10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009, 10010]
SEEDS = [10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009, 10010,
10011, 10012, 10013, 10014, 10015, 10016, 10017, 10018, 10019, 10020,
10021, 10022, 10023, 10024, 10025, 10026, 10027, 10028, 10029, 10030]
NUM_BINS = 100
MODEL_DIR = "/kaggle/input/cibmtr-final-models/CIBMTR final models/xgb_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Function to reload and use saved models
def predict_with_saved_models(test, train, model_dir=MODEL_DIR):
    oof_xgb = np.zeros(len(train))
    pred_xgb = np.zeros(len(test))

    train['efs_bins'] = pd.cut(train['efs'], bins=NUM_BINS, labels=False)
    train['stratify_col'] = train['race_group'].astype(str) + '_' + train['efs_bins'].astype(str)

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)

        for i, (train_index, test_index) in enumerate(skf.split(train, train['stratify_col'])):
            model_path = os.path.join(model_dir, f"xgb_fold{i}_seed{seed}.pkl")
            with open(model_path, 'rb') as f:
                model_xgb = pickle.load(f)

            x_valid = train.loc[test_index, FEATURES]
            x_test = test[FEATURES]

            oof_xgb[test_index] += model_xgb.predict_proba(x_valid)[:, 1] / len(SEEDS)
            pred_xgb += model_xgb.predict_proba(x_test)[:, 1] / (FOLDS * len(SEEDS))

    train.drop(columns=['efs_bins', 'stratify_col'], inplace=True)
    return (oof_xgb > 0.5).astype(int), (pred_xgb > 0.5).astype(int)

# Example usage in another runtime:
# oof_xgb_reloaded, pred_xgb_reloaded = predict_with_saved_models(test, train)


oof_xgb_reloaded, pred_xgb_reloaded = predict_with_saved_models(test, train)

# Evaluate performance
accuracy = accuracy_score(train["efs"], oof_xgb_reloaded)
f1 = f1_score(train["efs"], oof_xgb_reloaded)
roc_auc = roc_auc_score(train["efs"], oof_xgb_reloaded)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")


pred_xgb_reloaded


from pathlib import Path
from metric import score
import pandas as pd
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')

ROOT_DATA_PATH = Path(r"/kaggle/input/equity-post-HCT-survival-predictions")

pd.set_option('display.max_columns', 100)

train = pd.read_csv(ROOT_DATA_PATH.joinpath("train.csv"))
test = pd.read_csv(ROOT_DATA_PATH.joinpath("test.csv"))

CATEGORICAL_VARIABLES = [
    # Graft and HCT reasons
    'dri_score', 'graft_type', 'prod_type', 'prim_disease_hct',

    # Patient health status (risk factors)
    'psych_disturb', 'diabetes', 'arrhythmia', 'vent_hist', 'renal_issue', 'pulm_moderate',
    'pulm_severe', 'obesity', 'hepatic_mild', 'hepatic_severe', 'peptic_ulcer', 'rheum_issue',
    'cardiac', 'prior_tumor', 'mrd_hct', 'tbi_status', 'cyto_score', 'cyto_score_detail',

    # Patient demographics
    'ethnicity', 'race_group',

    # Biological matching with donor
    'sex_match', 'donor_related', 'cmv_status', 'tce_imm_match', 'tce_match', 'tce_div_match',

    # Medication/operation related data
    'melphalan_dose', 'rituximab', 'gvhd_proph', 'in_vivo_tcd', 'conditioning_intensity'
]

HLA_COLUMNS = [
    'hla_match_a_low', 'hla_match_a_high',
    'hla_match_b_low', 'hla_match_b_high',
    'hla_match_c_low', 'hla_match_c_high',
    'hla_match_dqb1_low', 'hla_match_dqb1_high',
    'hla_match_drb1_low', 'hla_match_drb1_high',

    # Matching at HLA-A(low), -B(low), -DRB1(high)
    'hla_nmdp_6',
    # Matching at HLA-A,-B,-DRB1 (low or high)
    'hla_low_res_6', 'hla_high_res_6',
    # Matching at HLA-A, -B, -C, -DRB1 (low or high)
    'hla_low_res_8', 'hla_high_res_8',
    # Matching at HLA-A, -B, -C, -DRB1, -DQB1 (low or high)
    'hla_low_res_10', 'hla_high_res_10'
]

OTHER_NUMERICAL_VARIABLES = ['year_hct', 'donor_age', 'age_at_hct', 'comorbidity_score', 'karnofsky_score']
NUMERICAL_VARIABLES = HLA_COLUMNS + OTHER_NUMERICAL_VARIABLES

TARGET_VARIABLES = ['efs_time', 'efs']
ID_COLUMN = ["ID"]


def preprocess_data(df):
    df[CATEGORICAL_VARIABLES] = df[CATEGORICAL_VARIABLES].fillna("Unknown")
    df[OTHER_NUMERICAL_VARIABLES] = df[OTHER_NUMERICAL_VARIABLES].fillna(df[OTHER_NUMERICAL_VARIABLES].median())

    return df

train = preprocess_data(train)
test = preprocess_data(test)


def features_engineering(df):
    # Change year_hct to relative year from 2000
    df['year_hct'] = df['year_hct'] - 2000

    return df


train = features_engineering(train)
test = features_engineering(test)

train[CATEGORICAL_VARIABLES] = train[CATEGORICAL_VARIABLES].astype('category')
test[CATEGORICAL_VARIABLES] = test[CATEGORICAL_VARIABLES].astype('category')

FEATURES = train.drop(columns=['ID', 'efs', 'efs_time']).columns.tolist()


import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Parameters
FOLDS = 10
#SEEDS = [20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010]
SEEDS = [20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010,
20011, 20012, 20013, 20014, 20015, 20016, 20017, 20018, 20019, 20020,
20021, 20022, 20023, 20024, 20025, 20026, 20027, 20028, 20029, 20030]
NUM_BINS = 100
MODEL_DIR = "/kaggle/input/cibmtr-final-models/CIBMTR final models/lgb_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Function to reload and use saved models
def predict_with_saved_models(test, train, model_dir=MODEL_DIR):
    oof_lgb = np.zeros(len(train))
    pred_lgb = np.zeros(len(test))

    train['efs_bins'] = pd.cut(train['efs'], bins=NUM_BINS, labels=False)
    train['stratify_col'] = train['race_group'].astype(str) + '_' + train['efs_bins'].astype(str)

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)

        for i, (train_index, test_index) in enumerate(skf.split(train, train['stratify_col'])):
            model_path = os.path.join(model_dir, f"lgbm_fold{i}_seed{seed}.pkl")
            with open(model_path, 'rb') as f:
                model_lgb = pickle.load(f)

            x_valid = train.loc[test_index, FEATURES]
            x_test = test[FEATURES]

            oof_lgb[test_index] += model_lgb.predict_proba(x_valid)[:, 1] / len(SEEDS)
            pred_lgb += model_lgb.predict_proba(x_test)[:, 1] / (FOLDS * len(SEEDS))

    train.drop(columns=['efs_bins', 'stratify_col'], inplace=True)
    return (oof_lgb > 0.5).astype(int), (pred_lgb > 0.5).astype(int)

# Example usage in another runtime:
# oof_lgb_reloaded, pred_lgb_reloaded = predict_with_saved_models(test, train)



oof_lgb_reloaded, pred_lgb_reloaded = predict_with_saved_models(test, train)

# Evaluate performance
accuracy = accuracy_score(train["efs"], oof_lgb_reloaded)
f1 = f1_score(train["efs"], oof_lgb_reloaded)
roc_auc = roc_auc_score(train["efs"], oof_lgb_reloaded)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")


pred_lgb_reloaded


from pathlib import Path
from metric import score
import pandas as pd
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')

ROOT_DATA_PATH = Path(r"/kaggle/input/equity-post-HCT-survival-predictions")

pd.set_option('display.max_columns', 100)

train = pd.read_csv(ROOT_DATA_PATH.joinpath("train.csv"))
test = pd.read_csv(ROOT_DATA_PATH.joinpath("test.csv"))

CATEGORICAL_VARIABLES = [
    # Graft and HCT reasons
    'dri_score', 'graft_type', 'prod_type', 'prim_disease_hct',

    # Patient health status (risk factors)
    'psych_disturb', 'diabetes', 'arrhythmia', 'vent_hist', 'renal_issue', 'pulm_moderate',
    'pulm_severe', 'obesity', 'hepatic_mild', 'hepatic_severe', 'peptic_ulcer', 'rheum_issue',
    'cardiac', 'prior_tumor', 'mrd_hct', 'tbi_status', 'cyto_score', 'cyto_score_detail',

    # Patient demographics
    'ethnicity', 'race_group',

    # Biological matching with donor
    'sex_match', 'donor_related', 'cmv_status', 'tce_imm_match', 'tce_match', 'tce_div_match',

    # Medication/operation related data
    'melphalan_dose', 'rituximab', 'gvhd_proph', 'in_vivo_tcd', 'conditioning_intensity'
]

HLA_COLUMNS = [
    'hla_match_a_low', 'hla_match_a_high',
    'hla_match_b_low', 'hla_match_b_high',
    'hla_match_c_low', 'hla_match_c_high',
    'hla_match_dqb1_low', 'hla_match_dqb1_high',
    'hla_match_drb1_low', 'hla_match_drb1_high',

    # Matching at HLA-A(low), -B(low), -DRB1(high)
    'hla_nmdp_6',
    # Matching at HLA-A,-B,-DRB1 (low or high)
    'hla_low_res_6', 'hla_high_res_6',
    # Matching at HLA-A, -B, -C, -DRB1 (low or high)
    'hla_low_res_8', 'hla_high_res_8',
    # Matching at HLA-A, -B, -C, -DRB1, -DQB1 (low or high)
    'hla_low_res_10', 'hla_high_res_10'
]

OTHER_NUMERICAL_VARIABLES = ['year_hct', 'donor_age', 'age_at_hct', 'comorbidity_score', 'karnofsky_score']
NUMERICAL_VARIABLES = HLA_COLUMNS + OTHER_NUMERICAL_VARIABLES

TARGET_VARIABLES = ['efs_time', 'efs']
ID_COLUMN = ["ID"]


def preprocess_data(df):
    df[CATEGORICAL_VARIABLES] = df[CATEGORICAL_VARIABLES].fillna("Unknown")
    df[OTHER_NUMERICAL_VARIABLES] = df[OTHER_NUMERICAL_VARIABLES].fillna(df[OTHER_NUMERICAL_VARIABLES].median())

    return df

train = preprocess_data(train)
test = preprocess_data(test)


def features_engineering(df):
    # Change year_hct to relative year from 2000
    df['year_hct'] = df['year_hct'] - 2000

    return df


train = features_engineering(train)
test = features_engineering(test)

train[CATEGORICAL_VARIABLES] = train[CATEGORICAL_VARIABLES].astype('category')
test[CATEGORICAL_VARIABLES] = test[CATEGORICAL_VARIABLES].astype('category')

FEATURES = train.drop(columns=['ID', 'efs', 'efs_time']).columns.tolist()


import os
import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Parameters
FOLDS = 10
#SEEDS = [30001, 30002, 30003, 30004, 30005, 30006, 30007, 30008, 30009, 30010]
SEEDS = [30001, 30002, 30003, 30004, 30005, 30006, 30007, 30008, 30009, 30010,
30011, 30012, 30013, 30014, 30015, 30016, 30017, 30018, 30019, 30020,
30021, 30022, 30023, 30024, 30025, 30026, 30027, 30028, 30029, 30030]
NUM_BINS = 100
MODEL_DIR = "/kaggle/input/cibmtr-final-models/CIBMTR final models/cat_classifier_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Function to reload and use saved models
def predict_with_saved_models(test, train, model_dir=MODEL_DIR):
    oof_cat = np.zeros(len(train))
    pred_cat = np.zeros(len(test))

    train['efs_bins'] = pd.cut(train['efs'], bins=NUM_BINS, labels=False)
    train['stratify_col'] = train['race_group'].astype(str) + '_' + train['efs_bins'].astype(str)

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)

        for i, (train_index, test_index) in enumerate(skf.split(train, train['stratify_col'])):
            model_path = os.path.join(model_dir, f"catboost_fold{i}_seed{seed}.pkl")
            with open(model_path, 'rb') as f:
                model_cat = pickle.load(f)

            x_valid = train.loc[test_index, FEATURES]
            x_test = test[FEATURES]

            oof_cat[test_index] += model_cat.predict_proba(x_valid)[:, 1] / len(SEEDS)
            pred_cat += model_cat.predict_proba(x_test)[:, 1] / (FOLDS * len(SEEDS))

    train.drop(columns=['efs_bins', 'stratify_col'], inplace=True)
    return (oof_cat > 0.5).astype(int), (pred_cat > 0.5).astype(int)

# Example usage in another runtime:
# oof_cat_reloaded, pred_cat_reloaded = predict_with_saved_models(test, train)


oof_cat_reloaded, pred_cat_reloaded = predict_with_saved_models(test, train)

# Evaluate performance
accuracy = accuracy_score(train["efs"], oof_cat_reloaded)
f1 = f1_score(train["efs"], oof_cat_reloaded)
roc_auc = roc_auc_score(train["efs"], oof_cat_reloaded)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")


pred_cat_reloaded


# Stack the predictions from all three classifiers
oof_ensemble = np.round((oof_xgb_reloaded + oof_lgb_reloaded + oof_cat_reloaded) >= 2).astype(int)

# Compute the F1 score for the ensemble
f1 = f1_score(train["efs"], oof_ensemble)
print(f"Ensemble F1 Score: {f1}")


pred_ensemble = np.round((pred_xgb_reloaded + pred_lgb_reloaded + pred_cat_reloaded) >= 2).astype(int)


pred_ensemble


import numpy as np

# Store results
best_m = None
best_coeff = None
results = []

for coeff in np.arange(0.0, 1.01, 0.01):
    y_pred = train[["ID"]].copy()
    adjusted_oof_nn = oof_cat.copy()
    adjusted_oof_nn[oof_ensemble == 1] += coeff  # Tune coefficient

    y_pred["prediction"] = adjusted_oof_nn
    m = score(y_true.copy(), y_pred.copy(), "ID")
    results.append((coeff, m))

    if best_m is None or m > best_m:
        best_m = m
        best_coeff = coeff

    print(f"Coefficient {coeff:.2f} -> CV Score: {m}")

print(f"\nBest coefficient: {best_coeff:.2f} with CV Score: {best_m}")



import numpy as np

# Store results
best_m = None
best_coeff = None
results = []

for coeff in np.arange(0.0, 1.01, 0.01):
    y_pred = train[["ID"]].copy()
    adjusted_oof_nn = oof_nn_reloaded.copy()
    adjusted_oof_nn[oof_ensemble == 1] += coeff  # Tune coefficient

    y_pred["prediction"] = adjusted_oof_nn
    m = score(y_true.copy(), y_pred.copy(), "ID")
    results.append((coeff, m))

    if best_m is None or m > best_m:
        best_m = m
        best_coeff = coeff

    print(f"Coefficient {coeff:.2f} -> CV Score: {m}")

print(f"\nBest coefficient: {best_coeff:.2f} with CV Score: {best_m}")



oof_cat[oof_ensemble == 1] += 0.13
oof_nn_reloaded[oof_ensemble == 1] += 0.16


pred_cat


test_pred_nn_reloaded


pred_cat[pred_ensemble == 1] += 0.13
test_pred_nn_reloaded[pred_ensemble == 1] += 0.16


pred_cat


test_pred_nn_reloaded


from metric import score, custom_score
from scipy.stats import rankdata

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_cat
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost KaplanMeier =",m)

race_dict = custom_score(y_true.copy(), y_pred.copy(), "ID", print_info=False)[1]
print(race_dict)


from metric import score, custom_score
from scipy.stats import rankdata

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_nn_reloaded
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost KaplanMeier =",m)

race_dict = custom_score(y_true.copy(), y_pred.copy(), "ID", print_info=False)[1]
print(race_dict)


from metric import score, custom_score
from scipy.stats import rankdata

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = rankdata(oof_nn_reloaded) + rankdata(oof_cat)
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost KaplanMeier =",m)

race_dict = custom_score(y_true.copy(), y_pred.copy(), "ID", print_info=False)[1]
print(race_dict)


from metric import score, custom_score
from scipy.stats import rankdata

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_nn_reloaded + oof_cat
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost KaplanMeier =",m)

race_dict = custom_score(y_true.copy(), y_pred.copy(), "ID", print_info=False)[1]
print(race_dict)


import numpy as np
from metric import score, custom_score
from scipy.stats import rankdata

# Initialize best score and weights
best_score = float("-inf")  # Assuming lower score is better
best_weights = None

# Store results
results = []

# Define true values
y_true = train[["ID", "efs", "efs_time", "race_group"]].copy()

for w1 in np.arange(0, 1.01, 0.01):  # Iterate over possible weight values
    w2 = 1 - w1  # Ensure sum equals 1

    # Compute weighted ensemble predictions
    y_pred = train[["ID"]].copy()
    y_pred["prediction"] = rankdata(oof_cat) * w1 + rankdata(oof_nn_reloaded) * w2

    # Compute score
    m = score(y_true.copy(), y_pred.copy(), "ID")

    # Compute per-race scores
    race_dict = custom_score(y_true.copy(), y_pred.copy(), "ID", print_info=False)[1]

    # Store results
    results.append((w1, w2, m, race_dict))

    # Update best score if applicable
    if m > best_score:
        best_score = m
        best_weights = (w1, w2)
        print(f"Best weights: CatBoost = {best_weights[0]:.2f}, NN = {best_weights[1]:.2f} with score = {best_score}")

# Print best weights
print(f"Best weights: CatBoost = {best_weights[0]:.2f}, NN = {best_weights[1]:.2f} with score = {best_score}")



import numpy as np
from metric import score, custom_score
from scipy.stats import rankdata

# Initialize best score and weights
best_score = float("-inf")  # Assuming lower score is better
best_weights = None

# Store results
results = []

# Define true values
y_true = train[["ID", "efs", "efs_time", "race_group"]].copy()

for w1 in np.arange(0, 1.01, 0.01):  # Iterate over possible weight values
    w2 = 1 - w1  # Ensure sum equals 1

    # Compute weighted ensemble predictions
    y_pred = train[["ID"]].copy()
    y_pred["prediction"] = oof_cat * w1 + oof_nn_reloaded * w2

    # Compute score
    m = score(y_true.copy(), y_pred.copy(), "ID")

    # Compute per-race scores
    race_dict = custom_score(y_true.copy(), y_pred.copy(), "ID", print_info=False)[1]

    # Store results
    results.append((w1, w2, m, race_dict))

    # Update best score if applicable
    if m > best_score:
        best_score = m
        best_weights = (w1, w2)
        print(f"Best weights: CatBoost = {best_weights[0]:.2f}, NN = {best_weights[1]:.2f} with score = {best_score}")

# Print best weights
print(f"Best weights: CatBoost = {best_weights[0]:.2f}, NN = {best_weights[1]:.2f} with score = {best_score}")



from metric import score, custom_score
from scipy.stats import rankdata

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_cat*0.61 + oof_nn_reloaded*0.39
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost KaplanMeier =",m)

race_dict = custom_score(y_true.copy(), y_pred.copy(), "ID", print_info=False)[1]
print(race_dict)


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = pred_cat*0.61 + test_pred_nn_reloaded*0.39
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()

