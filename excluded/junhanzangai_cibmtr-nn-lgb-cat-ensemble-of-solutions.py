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
    print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
    numerical = [i for i in FEATURES if i not in categorical_cols]
    return categorical_cols, numerical


def add_features(df):
    """
    Create some new features to help the model focus on specific patterns.
    """
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
    test_pred = np.zeros(test.shape[0])
    categorical_cols, numerical = get_feature_types(train_original)
    kf = StratifiedKFold(n_splits=5, shuffle=True, )
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
        # Create submission
        train = tt.iloc[train_index]
        X_cat_val, X_num_train, X_num_val, dl_train, dl_val, transformers = preprocess_data(train, test)
        pred, _ = model.cuda().eval()(
            torch.tensor(X_cat_val, dtype=torch.long).cuda(),
            torch.tensor(X_num_val, dtype=torch.float32).cuda()
        )
        test_pred += pred.detach().cpu().numpy()
        
    subm_data = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
    subm_data['prediction'] = -test_pred
    subm_data.to_csv('submission1.csv', index=False)
    
    display(subm_data.head())
    return 



def train_final(X_num_train, dl_train, dl_val, transformers, hparams=None, categorical_cols=None):
    """
    Defines model hyperparameters and fit the model.
    """
    if hparams is None:
        hparams = {
            "embedding_dim": 16,
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
res = main(hparams)
print("done")


source_file_path = '/kaggle/input/yunbase/Yunbase/baseline.py'
target_file_path = '/kaggle/working/baseline.py'
with open(source_file_path, 'r', encoding='utf-8') as file:
    content = file.read()
with open(target_file_path, 'w', encoding='utf-8') as file:
    file.write(content)


!pip install -q --requirement /kaggle/input/yunbase/Yunbase/requirements.txt  \
--no-index --find-links file:/kaggle/input/yunbase/


from baseline import Yunbase
import pandas as pd#read csv,parquet
import numpy as np#for scientific computation of matrices
from  lightgbm import LGBMRegressor,LGBMClassifier,log_evaluation,early_stopping
from catboost import CatBoostRegressor,CatBoostClassifier
from lifelines import KaplanMeierFitter
import warnings#avoid some negligible errors
#The filterwarnings () method is used to set warning filters, which can control the output method and level of warning information.
warnings.filterwarnings('ignore')
import random#provide some function to generate random_seed.
#set random seed,to make sure model can be recurrented.
def seed_everything(seed):
    np.random.seed(seed)#numpy's random seed
    random.seed(seed)#python built-in random seed
seed_everything(seed=2025)


# train=pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
# test=pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
# train_solution=train[['ID','efs','efs_time','race_group']].copy()

# def logit(p):
#     return np.log(p) - np.log(1 - p)
# max_efs_time,min_efs_time=80,-100
# train['efs_time']=train['efs_time']/(max_efs_time-min_efs_time)
# train['efs_time']=train['efs_time'].apply(lambda x:logit(x))
# train['efs_time']+=10
# print(train['efs_time'].max(),train['efs_time'].min())

# race2weight={'American Indian or Alaska Native':0.68,
# 'Asian':0.7,'Black or African-American':0.67,
# 'More than one race':0.68,
# 'Native Hawaiian or other Pacific Islander':0.66,
# 'White':0.64}
# train['weight']=0.5*train['efs']+0.5
# train['raceweight']=train['race_group'].apply(lambda x:race2weight.get(x,1))
# train['weight']=train['weight']/train['raceweight']
# train.drop(['raceweight'],axis=1,inplace=True)

# train.head()


# import seaborn as sns
# import matplotlib.pyplot as plt
# def transform_survival_probability(df, time_col='efs_time', event_col='efs'):

#     kmf = KaplanMeierFitter()
    
#     kmf.fit(df[time_col], event_observed=df[event_col])
    
#     survival_probabilities = kmf.survival_function_at_times(df[time_col]).values.flatten()

#     return survival_probabilities

# race_group=sorted(train['race_group'].unique())
# for race in race_group:
#     train.loc[train['race_group']==race,"target"] = transform_survival_probability(train[train['race_group']==race], time_col='efs_time', event_col='efs')
#     gap=0.7*(train.loc[(train['race_group']==race)&(train['efs']==0)]['target'].max()-train.loc[(train['race_group']==race)&(train['efs']==1)]['target'].min())/2
#     train.loc[(train['race_group']==race)&(train['efs']==0),'target']-=gap

# sns.histplot(data=train, x='target', hue='efs', element='step', stat='density', common_norm=False)
# plt.legend(title='efs')
# plt.title('Distribution of Target by EFS')
# plt.xlabel('Target')
# plt.ylabel('Density')
# plt.show()

# train.drop(['efs','efs_time'],axis=1,inplace=True)


# #nunique=2
# nunique2=[col for col in train.columns if train[col].nunique()==2 and col!='efs']
# #nunique<50
# nunique50=[col for col in train.columns if train[col].nunique()<50 and col not in ['efs','weight']]+['age_group','dri_score_NA']

# def FE(df):
#     print("< deal with outlier >")
#     df['nan_value_each_row'] = df.isnull().sum(axis=1)
#     #year_hct=2020 only 4 rows.
#     df['year_hct']=df['year_hct'].replace(2020,2019)
#     df['age_group']=df['age_at_hct']//10
#     #karnofsky_score 40 only 10 rows.
#     df['karnofsky_score']=df['karnofsky_score'].replace(40,50)
#     #hla_high_res_8=2 only 2 rows.
#     df['hla_high_res_8']=df['hla_high_res_8'].replace(2,3)
#     #hla_high_res_6=0 only 1 row.
#     df['hla_high_res_6']=df['hla_high_res_6'].replace(0,2)
#     #hla_high_res_10=3 only 1 row.
#     df['hla_high_res_10']=df['hla_high_res_10'].replace(3,4)
#     #hla_low_res_8=2 only 1 row.
#     df['hla_low_res_8']=df['hla_low_res_8'].replace(2,3)
#     df['dri_score']=df['dri_score'].replace('Missing disease status','N/A - disease not classifiable')
#     df['dri_score_NA']=df['dri_score'].apply(lambda x:int('N/A' in str(x)))
#     for col in ['diabetes','pulm_moderate','cardiac']:
#         df.loc[df[col].isna(),col]='Not done'

#     print("< cross feature >")
#     df['donor_age-age_at_hct']=df['donor_age']-df['age_at_hct']
#     df['comorbidity_score+karnofsky_score']=df['comorbidity_score']+df['karnofsky_score']
#     df['comorbidity_score-karnofsky_score']=df['comorbidity_score']-df['karnofsky_score']
#     df['comorbidity_score*karnofsky_score']=df['comorbidity_score']*df['karnofsky_score']
#     df['comorbidity_score/karnofsky_score']=df['comorbidity_score']/df['karnofsky_score']
    
#     print("< fillna >")
#     df[nunique50]=df[nunique50].astype(str).fillna('NaN')
    
#     print("< combine category feature >")
#     for i in range(len(nunique2)):
#         for j in range(i+1,len(nunique2)):
#             df[nunique2[i]+nunique2[j]]=df[nunique2[i]].astype(str)+df[nunique2[j]].astype(str)
    
#     print("< drop useless columns >")
#     df.drop(['ID'],axis=1,inplace=True,errors='ignore')
#     return df

# combine_category_cols=[]
# for i in range(len(nunique2)):
#     for j in range(i+1,len(nunique2)):
#         combine_category_cols.append(nunique2[i]+nunique2[j])  

# total_category_feature=nunique50+combine_category_cols

# target_stat=[]
# for j in range(len(total_category_feature)):
#    for col in ['donor_age','age_at_hct','target']:
#     target_stat.append( (total_category_feature[j],col,['count','mean','max','std','skew']) )

# num_folds=10

# lgb_params={"boosting_type": "gbdt","metric": 'mae',
#             'random_state': 2025,  "max_depth": 9,"learning_rate": 0.1,
#             "n_estimators": 768,"colsample_bytree": 0.6,"colsample_bynode": 0.6,
#             "verbose": -1,"reg_alpha": 0.2,
#             "reg_lambda": 5,"extra_trees":True,'num_leaves':64,"max_bin":255,
#             'importance_type': 'gain',#better than 'split'
#             'device':'gpu','gpu_use_dp':True
#            }

# cat_params={'random_state':2025,'eval_metric' : 'MAE',
#             'bagging_temperature': 0.50,'iterations': 650,
#             'learning_rate': 0.1,'max_depth': 8,
#             'l2_leaf_reg': 1.25,'min_data_in_leaf': 24,
#             'random_strength' : 0.25, 'verbose': 0,
#             'task_type':'GPU',
#             }

# yunbase=Yunbase(num_folds=num_folds,
#                   models=[(LGBMRegressor(**lgb_params),'lgb'),
#                           (CatBoostRegressor(**cat_params),'cat')
#                          ],
#                   FE=FE,
#                   seed=2025,
#                   objective='regression',
#                   metric='mae',
#                   target_col='target',
#                   device='gpu',
#                   one_hot_max=-1,
#                   early_stop=1000,
#                   cross_cols=['donor_age','age_at_hct'],
#                   target_stat=target_stat,
#                   use_data_augmentation=True,
#                   use_scaler=True,
#                   log=250,
#                   plot_feature_importance=True,
#                   #print metric score when model training
#                   use_eval_metric=False,
# )
# yunbase.fit(train,category_cols=nunique2)



# import pandas.api.types
# from lifelines.utils import concordance_index

# class ParticipantVisibleError(Exception):
#     pass

# def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    
#     del solution[row_id_column_name]
#     del submission[row_id_column_name]
    
#     event_label = 'efs'
#     interval_label = 'efs_time'
#     prediction_label = 'prediction'
#     for col in submission.columns:
#         if not pandas.api.types.is_numeric_dtype(submission[col]):
#             raise ParticipantVisibleError(f'Submission column {col} must be a number')
#     # Merging solution and submission dfs on ID
#     merged_df = pd.concat([solution, submission], axis=1)
#     merged_df.reset_index(inplace=True)
#     merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
#     metric_list = []
#     for race in merged_df_race_dict.keys():
#         # Retrieving values from y_test based on index
#         indices = sorted(merged_df_race_dict[race])
#         merged_df_race = merged_df.iloc[indices]
#         # Calculate the concordance index
#         c_index_race = concordance_index(
#                         merged_df_race[interval_label],
#                         -merged_df_race[prediction_label],
#                         merged_df_race[event_label])
#         metric_list.append(c_index_race)
#     return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))

# weights = [0.602, 0.398]

# lgb_prediction=np.load(f"Yunbase_info/lgb_seed{yunbase.seed}_repeat0_fold{yunbase.num_folds}_{yunbase.target_col}.npy")
# lgb_prediction=pd.DataFrame({'ID':train_solution['ID'],'prediction':lgb_prediction})
# print(f"lgb_score:{score(train_solution.copy(),lgb_prediction.copy(),row_id_column_name='ID')}")
# cat_prediction=np.load(f"Yunbase_info/cat_seed{yunbase.seed}_repeat0_fold{yunbase.num_folds}_{yunbase.target_col}.npy")
# cat_prediction=pd.DataFrame({'ID':train_solution['ID'],'prediction':cat_prediction})
# print(f"cat_score:{score(train_solution.copy(),cat_prediction.copy(),row_id_column_name='ID')}")

# y_preds=[lgb_prediction.copy(),cat_prediction.copy()]
# final_prediction=lgb_prediction.copy()
# final_prediction['prediction']=0
# for i in range(len(y_preds)):
#     final_prediction['prediction']+=weights[i]*y_preds[i]['prediction']
# metric=score(train_solution.copy(),final_prediction.copy(),row_id_column_name='ID')
# print(f"final_CV:{metric}")


# test_preds=yunbase.predict(test,weights=weights)
# yunbase.target_col='prediction'
# yunbase.submit("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv",test_preds,
#                save_name='submission2'
#               )


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)
train.head()


plt.hist(train.loc[train.efs==1,"efs_time"],bins=100,label="efs=1, Did Not Survive")
plt.hist(train.loc[train.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Survived")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to death, or time observed alive.")
plt.legend()
plt.show()


from lifelines import KaplanMeierFitter

def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    """
    Transform using survival probability estimates
    """
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    
    # Get survival probabilities at each time point
    y = kmf.survival_function_at_times(df[time_col]).values
    
    # Adjust for censoring
    # censored_mask = df[event_col] == 0
    #y[censored_mask] = y[censored_mask] * 1.2  # Increase survival prob for censored
    
    return y

train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')


plt.hist(train.loc[train.efs==1,"y"],bins=100,label="efs=1, Did Not Survive")
plt.hist(train.loc[train.efs==0,"y"],bins=100,label="efs=0, Maybe Survived")
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


RMV = ["ID","efs","efs_time","y"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


hct_ci_mapping = {
    "arrhythmia": {"No": 0, "Not done": 0, "Yes": 1},  
    "cardiac": {"No": 0, "Not done": 0, "Yes": 1}, 
    "diabetes": {"No": 0, "Not done": 0, "Yes": 1},  
    "hepatic_mild": {"No": 0, "Not done": 0, "Yes": 1},
    "hepatic_severe": {"No": 0, "Not done": 0, "Yes": 3},
    "psych_disturb": {"No": 0, "Not done": 0, "Yes": 1}, 
    "obesity": {"No": 0, "Not done": 0, "Yes": 1}, 
    "rheum_issue": {"No": 0, "Not done": 0, "Yes": 2},
    "peptic_ulcer": {"No": 0, "Not done": 0, "Yes": 2},  
    "renal_issue": {"No": 0, "Not done": 0, "Yes": 2}, 
    "prior_tumor": {"No": 0, "Not done": 0, "Yes": 3}, 
    "pulm_moderate": {"No": 0, "Not done": 0, "Yes": 2}, 
    "pulm_severe": {"No": 0, "Not done": 0, "Yes": 3},  
}
def calculate_hct_ci_score(row, mapping):
        """
        This function calculates the hct_ci score
    
        Args:
            row (pd.Series): Patient Clinical Data
            mapping (dict): HCT-CI score mapping
    
        Returns:
            int: HCT-CI score
        """
    
        score = 0
    
        if "hepatic_severe" in row and row["hepatic_severe"] == "Yes":
            score += mapping["hepatic_severe"]["Yes"]
        elif "hepatic_mild" in row and row["hepatic_mild"] == "Yes":
            score += mapping["hepatic_mild"]["Yes"]
        if "pulm_moderate" in row and row["pulm_moderate"] == "Yes":
            score += mapping["pulm_moderate"]["Yes"]
        elif "pulm_severe" in row and row["pulm_severe"] == "Yes":
            score += mapping["pulm_severe"]["Yes"]
    
        # Other Conditions
        for condition, mapping_values in mapping.items():
            if condition not in ["hepatic_mild", "hepatic_severe","pulm_moderate", "pulm_severe"] and condition in row:
                score += mapping_values.get(row[condition], 0)
    
        return score


# cat2num function is used for mapping some of the Categorical Values into Numerical Values

def cat2num(df):
    df['conditioning_intensity'] = df['conditioning_intensity'].map({
    'NMA': 1, 
    'RIC': 2,
    'MAC': 3,
    'TBD': None,
    'No drugs reported': None,
    'N/A, F(pre-TED) not submitted': None})
    
    df['tbi_status'] = df['tbi_status'].map({
    'No TBI': 0, 
    'TBI +- Other, <=cGy': 1,
    'TBI +- Other, -cGy, fractionated': 2,
    'TBI + Cy +- Other': 3,
    'TBI +- Other, -cGy, single': 4,
    'TBI +- Other, >cGy': 5,
    'TBI +- Other, unknown dose': None})
    
    df['dri_score'] = df['dri_score'].map({
    'Low': 1, 
    'Intermediate': 2,
    'Intermediate - TED AML case <missing cytogenetics': 3,
    'High': 4,
    'High - TED AML case <missing cytogenetics': 5,
    'Very High': 6,
    'N/A - pediatric': -3,
    'N/A - non-malignant indication': -1,
    'TBD cytogenetics': -2,
    'N/A - disease not classifiable': -4,
    'Missing disease status': 0})
    
    df['cyto_score'] = df['cyto_score'].map({
    'Poor': 4,
    'Normal': 3,
    'Intermediate': 2,
    'Favorable': 1,
    'TBD': -1,
    'Other': -2,
    'Not tested': None})
    
    df['cyto_score_detail'] = df['cyto_score_detail'].map({
    'Poor': 3, 
    'Intermediate': 2,
    'Favorable': 1,
    'TBD': -1,
    'Not tested': None})
    
    return df


def fill_hla_combined_low(row):
    if np.isnan(row['hla_combined_low']): 
        components = [
            row['hla_match_drb1_low'], row['hla_match_dqb1_low'], 
            row['hla_match_a_low'], row['hla_match_b_low'], row['hla_match_c_low']
        ]
        if all([not np.isnan(x) for x in components]):
            return sum(components)
        else:
            if not np.isnan(row['hla_low_res_8']) and not np.isnan(row['hla_match_dqb1_low']):
                return row['hla_low_res_8'] + row['hla_match_dqb1_low']
            elif not np.isnan(row['hla_low_res_6']): 
                components_6 = [
                    row['hla_match_dqb1_low'], row['hla_match_c_low']
                ]
                if all([not np.isnan(x) for x in components_6]):
                    return row['hla_low_res_6'] + sum(components_6)
                else: 
                    return sum([x for x in components if not np.isnan(x)])
    return row['hla_combined_low']


def add_features(df):
    df["hct_ci_score"] = df.apply(lambda row: calculate_hct_ci_score(row, hct_ci_mapping), axis=1)
    df['donor_recipient_age_diff'] = abs(df['donor_age'] - df['age_at_hct'])
    df = cat2num(df)
    df['hla_combined_low'] = df['hla_low_res_10']
    df['hla_combined_low'] = df.apply(fill_hla_combined_low, axis=1)
    df['hla_match_ratio'] = (df['hla_high_res_8'] + df['hla_low_res_8']) / 16
    df['years_since_2000'] = df['year_hct'] - 2000
    df['null_count'] = df.isnull().sum(axis=1)
    df['ci_score_danger'] = df['hct_ci_score'].apply(lambda x: 2 if x >= 3 else 1 if x >= 1 else 0)
    return df

train = add_features(train)
test = add_features(test)


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

# for c in cat2num:
#     combined[c] = combined[c].astype("int32")

train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


FEATURES += ["hct_ci_score", 'donor_recipient_age_diff', "hla_combined_low", "hla_match_ratio", 
             "years_since_2000", "null_count","ci_score_danger"]


train.head()


from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost
print("Using XGBoost version",xgboost.__version__)


from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train))
pred_efs = np.zeros(len(test))

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
    oof_xgb[test_index] = (model_xgb.predict_proba(x_valid)[:, 1] > 0.5).astype(int)
    # INFER TEST (Probabilities -> Average Probs)
    pred_efs += model_xgb.predict_proba(x_test)[:, 1]

# COMPUTE AVERAGE TEST PREDS
pred_efs = (pred_efs / FOLDS > 0.5).astype(int)

# EVALUATE PERFORMANCE
accuracy = accuracy_score(train["efs"], oof_xgb)
f1 = f1_score(train["efs"], oof_xgb)
roc_auc = roc_auc_score(train["efs"], oof_xgb)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")


bin_pred = oof_xgb
bin_pred


pred_classifier = pred_efs
pred_classifier


%%time
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

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
        max_depth=4,  
        colsample_bytree=0.5, 
        subsample=0.8, 
        n_estimators=10_000,  
        learning_rate=0.05, 
        eval_metric="mae",
        early_stopping_rounds=25,
        objective='reg:logistic',
        enable_categorical=True,
        min_child_weight=5
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=100 
    )

    # INFER OOF
    oof_xgb[test_index] = model_xgb.predict(x_valid)
    # INFER TEST
    pred_xgb += model_xgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS


oof_xgb


pred_xgb


bin_pred_np = np.array(bin_pred)
oof_xgb_np = np.array(oof_xgb)

combined_array = np.column_stack((bin_pred_np, oof_xgb_np))

print(combined_array)  # (2, 28800)


data_0 = combined_array[combined_array[:, 0] == 0]
data_1 = combined_array[combined_array[:, 0] == 1]


plt.figure(figsize=(10, 6))
plt.hist(data_0[:, 1], bins=100, color='blue', alpha=0.6, label='bin_pred = 0')
plt.hist(data_1[:, 1], bins=100, color='red', alpha=0.6, label='bin_pred = 1')

plt.xlabel('oof_xgb values (Second Column)')
plt.ylabel('Density')
plt.title('Histogram of oof_xgb values by bin_pred')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


combined_array[combined_array[:, 0] == 1, 1] += 0.1


data_0 = combined_array[combined_array[:, 0] == 0]
data_1 = combined_array[combined_array[:, 0] == 1]

plt.figure(figsize=(10, 6))

plt.hist(data_0[:, 1], bins=100, color='blue', alpha=0.6, label='bin_pred = 0')

plt.hist(data_1[:, 1], bins=100, color='red', alpha=0.6, label='bin_pred = 1')

plt.xlabel('oof_xgb values (Second Column)')
plt.ylabel('Density')
plt.title('Histogram of oof_xgb values by bin_pred')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = combined_array[:, 1]

print(y_pred)


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = combined_array[:, 1]

print(y_pred)


feature_importance = model_xgb.feature_importances_
importance_df = pd.DataFrame({
    "Feature": FEATURES,  # Replace FEATURES with your list of feature names
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


from catboost import CatBoostRegressor, CatBoostClassifier
import catboost
print("Using CatBoost version",catboost.__version__)


%%time
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_cat = np.zeros(len(train))
pred_cat = np.zeros(len(test))
cat_params = {
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.2,
        "l2_leaf_reg": 5,
        "random_seed": 42,
        "task_type": "CPU",
        "verbose": 100
    }

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)

    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    model_cat = CatBoostRegressor(**cat_params, cat_features=CATS)
    model_cat.fit(
        x_train,
        y_train,
        eval_set=(x_valid, y_valid),
    )

    # INFER OOF
    oof_cat[test_index] = model_cat.predict(x_valid)
    # INFER TEST
    pred_cat += model_cat.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat /= FOLDS


bin_pred_np = np.array(bin_pred)
oof_cat_np = np.array(oof_cat)

combined_array_cat = np.column_stack((bin_pred_np, oof_cat_np))

print(combined_array)  # (2, 28800)


data_0 = combined_array_cat[combined_array_cat[:, 0] == 0]
data_1 = combined_array_cat[combined_array_cat[:, 0] == 1]

plt.figure(figsize=(10, 6))

plt.hist(data_0[:, 1], bins=100, color='blue', alpha=0.6, label='bin_pred = 0')

plt.hist(data_1[:, 1], bins=100, color='red', alpha=0.6, label='bin_pred = 1')

# 레이블과 제목 추가
plt.xlabel('oof_cat values (Second Column)')
plt.ylabel('Density')
plt.title('Histogram of oof_cat values by bin_pred')
plt.legend()
plt.grid(alpha=0.3)
plt.show()



combined_array_cat[combined_array_cat[:, 0] == 1, 1] += 0.1


data_0 = combined_array_cat[combined_array_cat[:, 0] == 0]
data_1 = combined_array_cat[combined_array_cat[:, 0] == 1]

plt.figure(figsize=(10, 6))

plt.hist(data_0[:, 1], bins=100, color='blue', alpha=0.6, label='bin_pred = 0')

plt.hist(data_1[:, 1], bins=100, color='red', alpha=0.6, label='bin_pred = 1')

plt.xlabel('oof_cat values (Second Column)')
plt.ylabel('Density')
plt.title('Histogram of oof_cat values by bin_pred')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = combined_array_cat[:, 1]

print(y_pred)


from metric import score

m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost =",m)


feature_importance = model_cat.get_feature_importance()
importance_df = pd.DataFrame({
    "Feature": FEATURES, 
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("CatBoost Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = combined_array[:, 1] + combined_array_cat[:, 1]
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for Ensemble =",m)


prediction = pred_xgb + pred_cat

pred_classifier_np = np.array(pred_classifier)
prediction_np = np.array(prediction)

combined_pred = np.column_stack((pred_classifier_np, prediction_np))
print(combined_pred)  # (2, 3)

combined_pred[combined_pred[:, 0] == 1, 1] += 0.1
print(combined_pred)  # (2, 3)


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = combined_pred[:, 1]

print(sub)


sub.to_csv("submission3.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata 

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )
train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)
train.head()


RMV = ["ID","efs","efs_time","y","efs_time2"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


from lifelines import  NelsonAalenFitter
def create_nelson(data):
    data=data.copy()
    naf = NelsonAalenFitter(nelson_aalen_smoothing=0)
    naf.fit(durations=data['efs_time'], event_observed=data['efs'])
    return naf.cumulative_hazard_at_times(data['efs_time']).values*-1


combined = pd.concat([train,test],axis=0,ignore_index=True)
print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
for c in FEATURES:

    if c in CATS:
        print(f"{c}, ",end="")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
        
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


from sklearn.model_selection import KFold,StratifiedKFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
print("Using XGBoost version",xgb.__version__)


train["y_nel"] = create_nelson(train)

#important
train.loc[train.efs == 0, "y_nel"] = (-(-train.loc[train.efs == 0, "y_nel"])**0.5)

plt.hist(train.loc[train.efs==1,"y_nel"],bins=200,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"y_nel"],bins=200,label="efs=0, Maybe Event")
plt.xlabel("Transformed Target y_nelson")
plt.ylabel("Density")
plt.title("NelsonAalenFitter Transformed Target y_nelson using both efs and efs_time.")
plt.legend()
plt.show()


%%time
FOLDS = 10
def create_stratified_folds(data, target, n_splits=10):
    data['fold'] = -1
    # num_bins = int(np.floor(1 + np.log2(len(data))))  # Sturges' rule for binning
    if (target!="race_group"):
        data['bins'] = pd.qcut(data[target], q=50, duplicates='drop',labels=False)
    data["bins"]=data["race_group"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (_, val_idx) in enumerate(skf.split(data, data['bins'])):
        data.loc[val_idx, 'fold'] = fold
    
    data = data.drop(columns=['bins'])
    return data

train=create_stratified_folds(train,"race_group",FOLDS)


oof_xgb1 = np.zeros(len(train))
pred_xgb1 = np.zeros(len(test))

for i in range(FOLDS):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train.fold!=i,FEATURES].copy()
    y_train = train.loc[train.fold!=i,"y_nel"]
    x_valid = train.loc[train.fold==i,FEATURES].copy()
    y_valid = train.loc[train.fold==i,"y_nel"]
    x_test = test[FEATURES].copy()

    model_xgb = XGBRegressor(
        # device="cuda",
        max_depth=4,  
        colsample_bytree=0.55,  
        subsample=0.8,  
        n_estimators=5000,  
        learning_rate=0.02,  
        enable_categorical=True,
        min_child_weight=80,
        early_stopping_rounds=200,
        n_jobs=4
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=500 
    )

    # INFER OOF
    oof_xgb1[train.index[train.fold==i]] = (model_xgb.predict(x_valid))
    # INFER TEST
    pred_xgb1 += (model_xgb.predict(x_test))

from metric import score

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb1

m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost NelsonAalenFitter =",m)


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = rankdata(pred_xgb1)

sub.to_csv("submission4.csv",index=False)


import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor



def add_features(df):
    sex_match = df.sex_match.astype(str)
    sex_match = sex_match.str.split("-").str[0] == sex_match.str.split("-").str[1]
    df['sex_match_bool'] = sex_match.astype("object")
    return df




from lifelines import KaplanMeierFitter


def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    df_ = df.loc[(df[time_col] < 24) | (df[event_col] == 0)]
    kmf.fit(df_[time_col], df_[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    plt.hist(df.efs_time[df[event_col] == 0], bins=150, alpha=0.5, label="Event")
    plt.show()
    #y = y / (y.max() - y.min())
    # y = np.log(1 + np.log(y / (1 - y)))
    return y



from metric import score as score_f



def run(hparams=None):
    FOLDS, oof_xgb, pred_xgb, train = make_predictions(hparams)

    # COMPUTE AVERAGE TEST PREDS
    pred_xgb /= FOLDS

    y_true = train[["ID", "efs", "efs_time", "race_group"]].copy()
    y_pred = train[["ID"]].copy()
    y_pred["prediction"] = oof_xgb
    m = score_f(train.copy(), y_pred.copy(), "ID")
    plt.hist(oof_xgb[train.efs==1], bins=np.linspace(-3, 3, 200), alpha=0.5, label="Event")
    plt.hist(oof_xgb[train.efs==0], bins=np.linspace(-3, 3, 200), alpha=0.5, label="No event")
    plt.legend()
    plt.show()
    print(f"\nOverall CV for XGBoost KaplanMeier =", m)
    return pred_xgb


def make_predictions(hparams):
    if hparams is None:
        hparams = dict(
            max_depth=6,
            colsample_bytree=0.5,
            subsample=0.8,
            n_estimators=3000,
            learning_rate=0.01,
            min_child_weight=40,
            gamma=1,
            eta=0.0,
            reg_lambda=0.1,
            reg_alpha=0.1,
            eps=2e-2,
            eps_mul=1.01,
            pos_shift=0.2
        )
    FEATURES, test, train, y = prepare_data(hparams.pop('eps'), hparams.pop('eps_mul'))
    FOLDS = 5
    kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    oof_xgb = np.zeros(len(train))
    pred_xgb = np.zeros(len(test))
    pos_shift = hparams.pop('pos_shift')
    for i, (train_index, test_index) in enumerate(kf.split(train, train.race_group)):
        print("#" * 25)
        print(f"### Fold {i + 1}")
        print("#" * 25)
        x_test, x_train, x_valid, y_train, y_valid = _prepare_training_data(
            FEATURES, test.copy(), test_index, train.copy(), train_index, y.copy(), pos_shift=pos_shift
        )
        hparams.update(
            dict(
                objective='reg:pseudohubererror',
                max_cat_to_onehot=10,
                device="cuda",
                enable_categorical=True,
                random_state=42,
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
            )
        )
        print(hparams['n_estimators'])
        model_xgb = XGBRegressor(
            **hparams
        )
        tt = train.loc[train_index]

        array = np.array([.95 if x else 1 for x in ((tt.efs_time < 24) & (tt.efs == 0)).values])
        array /= np.array([1.3 if x else 1 for x in tt.efs_time > 36])
        model_xgb.fit(
            x_train, y_train,
            eval_set=[(x_train, y_train), (x_valid, y_valid)],
            verbose=100,
            sample_weight=array
        )

        # INFER OOF
        oof_xgb[test_index] = model_xgb.predict(x_valid)
        # INFER TEST
        pred_xgb += model_xgb.predict(x_test)
    return FOLDS, oof_xgb, pred_xgb, train


def _prepare_training_data(FEATURES, test, test_index, train, train_index, y, pos_shift=0.1):
    y[train.efs == 0] = y[train.efs == 1].min() - pos_shift
    std = np.std(y[train_index])
    x_train = train.loc[train_index, FEATURES].copy()
    y_train = y[train_index] / std
    x_valid = train.loc[test_index, FEATURES].copy()
    y_valid = y[test_index] / std
    x_test = test[FEATURES].copy()
    # ind = (train.loc[train_index].efs_time < 24) | (train.loc[train_index].efs == 0)
    # x_train = x_train[ind]
    # y_train = y_train[ind]

    le = LabelEncoder()
    val = train.loc[test_index]
    return x_test, x_train, x_valid, y_train, y_valid



def prepare_data(eps=2e-2, eps_mul=1.1):
    test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
    test = add_features(test)
    train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
    train = add_features(train)
    train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')
    RMV = ["ID", "efs", "efs_time", "y"]
    FEATURES = [c for c in train.columns if not c in RMV]

    CATS = []
    for c in FEATURES:
        if train[c].dtype == "object":
            CATS.append(c)
            train[c] = train[c].fillna("NAN")
            test[c] = test[c].fillna("NAN")
    print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")
    print(f"In these features, there are {len(CATS)} NUMERICAL FEATURES: {sorted([f for f in FEATURES if f not in CATS])}")
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    # print("Combined data shape:", combined.shape )
    # LABEL ENCODE CATEGORICAL FEATURES
    for c in FEATURES:

        # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
        if c in CATS:
            combined[c], _ = combined[c].factorize()
            combined[c] -= combined[c].min()
            combined[c] = combined[c].astype("int32")
            combined[c] = combined[c].astype("category")

        # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
        else:
            if combined[c].dtype == "float64":
                combined[c] = combined[c].astype("float32")
            if combined[c].dtype == "int64":
                combined[c] = combined[c].astype("int32")
    train = combined.iloc[:len(train)].copy()
    test = combined.iloc[len(train):].reset_index(drop=True).copy()
    y = train.y.copy()
    # y = np.log(y)
    y = (y - y.min() + eps) / (y.max() - y.min() + eps_mul * eps)
    y = np.log(y / (1 - y))
    plt.hist(y[train.efs == 1], bins=np.linspace(-4, 8, 150), label="Event")
    plt.hist(y[train.efs == 0], bins=np.linspace(-4, 8, 150), label="No event")
    plt.legend()
    plt.show()
    return FEATURES, test, train, y


pred_xgb = run(None)


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = pred_xgb
sub.to_csv("submission5.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# CSV 파일 불러오기
sub1 = pd.read_csv('submission1.csv')  # 0.691
# sub2 = pd.read_csv('submission2.csv')  # 0.689
sub3 = pd.read_csv('submission3.csv')  # 0.688
#sub4 = pd.read_csv('submission4.csv')  # 0.685
sub5 = pd.read_csv('submission5.csv')  # 0.688

# MinMaxScaler를 사용하여 각 예측값을 0~1 범위로 정규화
scaler = MinMaxScaler()

# 정규화된 예측값 계산
sub1_norm = scaler.fit_transform(sub1[['prediction']])
sub3_norm = scaler.fit_transform(sub3[['prediction']])
#sub4_norm = scaler.fit_transform(sub4[['prediction']])
sub5_norm = scaler.fit_transform(sub5[['prediction']])

# 성능 기반 가중치 설정 (성능이 좋을수록 높은 가중치)
# 성능 점수를 가중치로 직접 사용하거나 정규화할 수 있음
weights = [3, 1, 1]
total_weight = sum(weights)
normalized_weights = [w/total_weight for w in weights]

# 가중 평균 계산
weighted_ensemble = (
    normalized_weights[0] * sub1_norm + 
    normalized_weights[1] * sub3_norm + 
    #normalized_weights[2] * sub4_norm + 
    normalized_weights[2] * sub5_norm
)

# 결과 데이터프레임 생성
sub_ensemble = sub1.copy()
sub_ensemble['prediction'] = weighted_ensemble

# 결과 파일 저장
sub_ensemble.to_csv("submission.csv", index=False)

# 가중치 출력 (참고용)
print(f"모델별 가중치: sub1={normalized_weights[0]:.4f}, sub3={normalized_weights[1]:.4f}, sub5={normalized_weights[2]:.4f}")


print(sub_ensemble.head())




