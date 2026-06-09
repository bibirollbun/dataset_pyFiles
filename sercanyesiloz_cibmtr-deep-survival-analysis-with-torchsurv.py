!pip install /kaggle/input/cibmtr-torchsurv/torchsurv-0.1.4-py3-none-any.whl
!pip install lifelines -q --no-index --find-links=/kaggle/input/cibmtr2024-import/lifelines
!pip install scikit-learn==1.4.0 -q --no-index --find-links=/kaggle/input/cibmtr2024-import/scikit_learn
!pip install rtdl_num_embeddings -q --no-index --find-links=/kaggle/input/cibmtr2024-import/rtdl_num_embeddings
!pip install delu -q --no-index --find-links=/kaggle/input/cibmtr2024-import/delu


import os
import gc
import torch
import copy
import warnings
import lifelines
import pandas as pd
import numpy as np
from tqdm import tqdm
from metric import score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, PredefinedSplit
warnings.filterwarnings("ignore")

from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.loss.weibull import neg_log_likelihood, log_hazard, survival_function
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.auc import Auc
from torchsurv.stats.kaplan_meier import KaplanMeierEstimator

print(f"torch version: {torch.__version__}")
print(f"lifelines version: {lifelines.__version__}")


class config:
    root = "/kaggle/input/equity-post-HCT-survival-predictions"
    train_path = os.path.join(root, "train.csv")
    test_path = os.path.join(root, "test.csv")
    sub_path = os.path.join(root, "sample_submission.csv")
    seed = 42
    n_folds = 5
    epochs = 20
    batch_size = 2048
    learning_rate = 1e-3


def add_features(df):
    """
    Create some new features to help the model focus on specific patterns.
    """
    df['is_cyto_score_same'] = (df['cyto_score'] == df['cyto_score_detail']).astype(int)
    df['year_hct'] -= 2000
    
    return df

train_df = pd.read_csv(config.train_path)
eval_df = train_df[["ID", "efs", "efs_time", "race_group"]].copy()
test_df = pd.read_csv(config.test_path)

train_df = add_features(train_df)
test_df = add_features(test_df)

target_cols = ["efs", "efs_time"]
drop_cols = ["ID"]

cat_cols = [col for col in train_df.select_dtypes(include=["object"]).columns if col not in target_cols + drop_cols]
num_cols = [col for col in train_df.columns if col not in cat_cols + target_cols + drop_cols]

print(f"cat_cols: {len(cat_cols)}")
print(f"num_cols: {len(num_cols)}")

# Categorical Features
for col in cat_cols:
    train_df[col].fillna("Unknown", inplace=True)
    test_df[col].fillna("Unknown", inplace=True)

    labels = train_df[col].unique()
    for i in labels:
        train_df[f"{col}_{i}"] = train_df[col].apply(lambda x: 1 if x == i else 0)
        test_df[f"{col}_{i}"] = test_df[col].apply(lambda x: 1 if x == i else 0)

    if col != "race_group":
        train_df.drop(columns=[col], axis=1, inplace=True)
    test_df.drop(columns=[col], axis=1, inplace=True)

# Numerical Features
for col in num_cols:
    imputer = SimpleImputer(strategy='mean')
    train_df[col] = imputer.fit_transform(train_df[col].values.reshape(-1, 1))
    test_df[col] = imputer.transform(test_df[col].values.reshape(-1, 1))

train_df = train_df.drop(columns=drop_cols, axis=1)
test_df = test_df.drop(columns=drop_cols, axis=1)

print(f"train: {train_df.shape}")
print(f"test: {test_df.shape}")

train_df.head()


skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)

split_fn = skf.split(X=train_df, y=train_df.race_group.astype(str) + (train_df.age_at_hct == 0.044).astype(str))

for idx, (train_idx, val_idx) in enumerate(split_fn):
    train_df.loc[val_idx, "fold"] = idx+1

cv = PredefinedSplit(train_df["fold"].values)

print(train_df["fold"].value_counts().sort_index())

train_df = train_df.drop(columns=["race_group", "fold"], axis=1)
test_df["efs"] = np.nan
test_df["efs_time"] = np.nan

scaler = StandardScaler()
inputs = [col for col in train_df.columns if col not in target_cols]
train_df[inputs] = scaler.fit_transform(train_df[inputs])
test_df[inputs] = scaler.transform(test_df[inputs])


class Custom_dataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        event = torch.tensor(sample["efs"]).bool()
        time = torch.tensor(sample["efs_time"]).float()
        x = torch.tensor(sample.drop(["efs", "efs_time"]).values).float()
        return x, (event, time)

def load_cox_model(num_features: int):
    cox_model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(num_features),
        torch.nn.Linear(num_features, 32),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.2, inplace=False),
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.2, inplace=False),
        torch.nn.Linear(64, 1),
    )
    return cox_model

def plot_losses(train_losses: list, val_losses: list, title: str = "Cox") -> None:
    train_losses = torch.stack(train_losses) / train_losses[0]
    val_losses = torch.stack(val_losses) / val_losses[0]
    plt.plot(train_losses, label="training")
    plt.plot(val_losses, label="validation")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Normalized loss")
    plt.title(title)
    plt.yscale("log")
    plt.show()

def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader_train: torch.utils.data.DataLoader,
    dataloader_val: torch.utils.data.DataLoader,
    epochs: int,
    plot: bool = True
):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        epoch_loss = torch.tensor(0.0)
        for i, batch in enumerate(dataloader_train):
            x, (event, time) = batch
            optimizer.zero_grad()
            log_hz = cox_model(x)
            loss = neg_partial_log_likelihood(log_hz, event, time, reduction="mean")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach()
        epoch_loss /= i + 1

        with torch.no_grad():
            x, (event, time) = next(iter(dataloader_val))
            val_loss = neg_partial_log_likelihood(cox_model(x), event, time, reduction="mean")
            val_losses.append(val_loss)
            
        print(f"Epoch: {epoch}, Train Loss: {epoch_loss:.5f} Val Loss: {val_loss:.5f}")
        train_losses.append(epoch_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)

    if plot:
        plot_losses(train_losses, val_losses, "Cox")

    return model

def infer(model: torch.nn.Module, dataloader_test: torch.utils.data.DataLoader):
    model.eval()
    with torch.no_grad():
        x, (event, time) = next(iter(dataloader_test))
        log_hz = model(x)
    return log_hz, event, time

def validate_model(data: pd.DataFrame, preds):
    y_true = data[['ID', 'efs', 'efs_time', 'race_group']].copy()
    y_pred = data[['ID']].copy()
    y_pred['prediction'] = preds
    c_index_score = score(y_true.copy(), y_pred.copy(), 'ID')
    return c_index_score


dataloader_test = DataLoader(
    Custom_dataset(test_df), batch_size=len(test_df), shuffle=False
)

models = list()
fold_scores = list()
for idx, (train_idx, val_idx) in enumerate(cv.split(train_df)):
    print(f"| Fold {idx+1} |".center(90, "="))
    
    dataloader_train = DataLoader(
        Custom_dataset(train_df.loc[train_idx]), batch_size=config.batch_size, shuffle=True
    )
    dataloader_val = DataLoader(
        Custom_dataset(train_df.loc[val_idx]), batch_size=len(train_df.loc[val_idx]), shuffle=False
    )

    num_features = next(iter(dataloader_train))[0].size(1)
    cox_model = load_cox_model(num_features)
    optimizer = torch.optim.Adam(cox_model.parameters(), lr=config.learning_rate)
    
    cox_model = train_model(cox_model, optimizer, dataloader_train, dataloader_val, config.epochs)
    log_hz, event, time = infer(cox_model, dataloader_val)
    models.append(cox_model)

    cox_cindex = ConcordanceIndex()
    fold_score = validate_model(eval_df.loc[val_idx], log_hz.view(-1).numpy())
    fold_scores.append(fold_score)

    print(f"C-Index: {cox_cindex(log_hz, event, time):.5f}")
    print(f"Stratified C-Index: {fold_score:.5f}")
    print(f"Confidence Interval: {cox_cindex.confidence_interval()}\n")

print(f"\nCV: {np.mean(fold_scores):.5f}")


sub = pd.read_csv(config.sub_path)
test_preds = np.zeros(shape=(sub.shape[0]))

for model in tqdm(models):
    test_preds += infer(model, dataloader_test)[0].view(-1).numpy() / len(models)

sub["prediction"] = test_preds
sub.to_csv("submission.csv", index=False)
sub.head()

