# --------------------
# Imports & settings
# --------------------
import warnings
warnings.filterwarnings("ignore")

import os
import gc
import math
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

plt.style.use("seaborn")
sns.set_style("whitegrid")
pd.set_option("display.max_columns", 200)

# --------------------
# Config
# --------------------
class Config:
    target = "BeatsPerMinute"

    train_path = "/kaggle/input/playground-series-s5e9/train.csv"
    test_path = "/kaggle/input/playground-series-s5e9/test.csv"
    sample_path = "/kaggle/input/playground-series-s5e9/sample_submission.csv"

    # Load right away (index col NOT used here to keep 'id' column)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    submission = pd.read_csv(sample_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = 42
    n_splits = 5            # reduced from 10 to 5 to save time; set to 10 if you want
    early_stop = 20
    training = True

# --------------------
# Basic EDA (quick)
# --------------------
def run_eda(df_train, df_test):
    print("TRAIN SHAPE:", df_train.shape)
    print("TEST SHAPE :", df_test.shape)
    display(df_train.head())
    display(df_train.info())
    print("\nMissing values (train):")
    print(df_train.isnull().sum())
    print("\nMissing values (test):")
    print(df_test.isnull().sum())
    print("\nTarget statistics:")
    display(df_train[Config.target].describe())

    # Target distribution
    plt.figure(figsize=(8,4))
    sns.histplot(df_train[Config.target], bins=80, kde=True)
    plt.title("Target distribution: BeatsPerMinute")
    plt.show()

    # Target boxplot
    plt.figure(figsize=(8,3))
    sns.boxplot(x=df_train[Config.target])
    plt.title("Target boxplot")
    plt.show()

    # Basic numeric histograms (all numeric features)
    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != Config.target and c != "id"]
    # show up to first 10 numeric columns
    to_plot = num_cols[:10]
    n = len(to_plot)
    cols = 2
    rows = math.ceil(n / cols)
    plt.figure(figsize=(12, 4*rows))
    for i, c in enumerate(to_plot, 1):
        plt.subplot(rows, cols, i)
        sns.histplot(df_train[c], bins=60, kde=True)
        plt.title(c)
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12,10))
    corr = df_train.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation heatmap (numeric)")
    plt.show()

    # Scatter plots vs target for first 6 numeric columns
    plt.figure(figsize=(12, 10))
    for i, c in enumerate(to_plot[:6], 1):
        plt.subplot(3,2,i)
        sns.scatterplot(x=df_train[c], y=df_train[Config.target], alpha=0.3, s=8)
        plt.title(f"{c} vs {Config.target}")
    plt.tight_layout()
    plt.show()

# --------------------
# Transform class (robust)
# --------------------
class Transform:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train = train.copy()
        self.test = test.copy()

        # Ensure test has target column as NaN for concatenation consistency
        if Config.target not in self.test.columns:
            self.test[Config.target] = np.nan

        # identify features
        self.num_features = self.train.drop([Config.target, "id"], axis=1).select_dtypes(include=[np.number]).columns.tolist()
        self.cat_features = self.train.drop([Config.target, "id"], axis=1).select_dtypes(include=['object', 'bool']).columns.tolist()

        # flags - you can change these if desired
        self.feature_eng = True
        self.log_trf = False
        self.missing = False
        self.outliers = False

        if self.feature_eng:
            # create new features on a copy to avoid explosion issues - do cautiously
            self.train = self.new_features(self.train)
            self.test = self.new_features(self.test)
            # recompute numeric features after feature engineering
            self.num_features = self.train.drop([Config.target, "id"], axis=1).select_dtypes(include=[np.number]).columns.tolist()

        if self.missing:
            self._impute_missing()

        self.encode()

    def _impute_missing(self):
        # simple numeric median imputation
        for c in self.num_features:
            med = self.train[c].median()
            self.train[c].fillna(med, inplace=True)
            self.test[c].fillna(med, inplace=True)

    def new_features(self, data):
        # Create pairwise interactions but limit number to avoid explosion:
        cols = data.drop([Config.target, "id"], axis=1).select_dtypes(include=[np.number]).columns.tolist()
        # only create interactions for first 6 numeric columns to reduce blowup
        cols_subset = cols[:6]
        for c1, c2 in combinations(cols_subset, 2):
            data[f"{c1}_x_{c2}"] = data[c1] * data[c2]
            data[f"{c1}_div_{c2}"] = data[c1] / (data[c2] + 1e-6)

        # quartile & decile for first 6 numeric columns
        for c in cols[:6]:
            data[f"{c}_quartile"] = pd.cut(data[c], bins=4, labels=False, include_lowest=True)
            data[f"{c}_decile"] = pd.cut(data[c], bins=10, labels=False, include_lowest=True)

        return data

    def encode(self):
        # concatenate train & test for consistent scaling/encoding
        data = pd.concat([self.train, self.test], axis=0, ignore_index=True)

        # categorical encoding (only if cat features exist)
        if len(self.cat_features) > 0:
            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            data[self.cat_features] = oe.fit_transform(data[self.cat_features].astype(object))
        else:
            # no categorical features
            pass

        # scale numerics
        scaler = StandardScaler()
        # recompute numeric columns in concatenated data (drop id and target)
        num_cols = [c for c in data.columns if c not in [Config.target, "id"] and data[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
        data[num_cols] = scaler.fit_transform(data[num_cols])

        # split back
        train_enc = data[~data[Config.target].isna()].reset_index(drop=True)
        test_enc = data[data[Config.target].isna()].drop(columns=[Config.target]).reset_index(drop=True)

        # restore id columns from original frames (safer)
        train_enc["id"] = self.train["id"].values
        test_enc["id"] = self.test["id"].values

        # assign to self
        self.train_enc = train_enc
        self.test_enc = test_enc

    def get(self):
        # Return processed structures (X_enc without id, test_enc without id)
        X_enc = self.train_enc.drop(columns=[Config.target, "id"])
        y = self.train_enc[Config.target].astype(np.float32)
        test_enc = self.test_enc.drop(columns=["id"])
        return X_enc, y, test_enc

# --------------------
# RMSE helper
# --------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# --------------------
# LightGBM quick baseline (feature importance)
# --------------------
def lgb_baseline(X, y, X_test):
    print("Training LightGBM baseline (quick)...")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=Config.state)
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=Config.state)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
    val_pred = model.predict(X_val)
    print("LGB baseline RMSE:", rmse(y_val, val_pred))
    # feature importance
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(30)
    plt.figure(figsize=(8,10))
    fi.plot(kind="barh")
    plt.title("LightGBM top 30 importances")
    plt.gca().invert_yaxis()
    plt.show()
    # return preds
    test_pred = model.predict(X_test)
    return test_pred

# --------------------
# Deep model (Residual MLP)
# --------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        res = self.shortcut(x)
        return self.act(out + res)

class MLPRegressor(nn.Module):
    def __init__(self, in_features, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.append(ResidualBlock(prev, h, dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

# --------------------
# Training utilities
# --------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_fold(X_tr, y_tr, X_va, y_va, params, epochs=500, batch_size=256, patience=30):
    # to np arrays
    X_tr = X_tr.astype(np.float32)
    X_va = X_va.astype(np.float32)
    y_tr = y_tr.astype(np.float32)
    y_va = y_va.astype(np.float32)

    ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    ds_va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = Config.device
    model = MLPRegressor(X_tr.shape[1], params["hidden_dims"], params["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=max(patience//4, 8), factor=0.5)

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs+1):
        # train
        model.train()
        train_loss = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(ds_tr)

        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * xb.size(0)
        val_loss /= len(ds_va)

        scheduler.step(val_loss)

        if epoch % max(1, patience//2) == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} Train: {train_loss:.6e} Val: {val_loss:.6e}")

        # early stopping logic
        if val_loss < best_loss - 1e-9:
            best_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}, best val loss {best_loss:.6e}")
                break

    # load best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    # final predictions on val
    with torch.no_grad():
        va_out = model(torch.from_numpy(X_va).to(device)).cpu().numpy()

    return va_out, np.sqrt(mean_squared_error(y_va, va_out)), model

# --------------------
# CV fit / predict
# --------------------
def fit_predict_cv(X, y, X_test, seed=Config.state):
    set_seed(seed)

    X_np = X.values.astype(np.float32)
    y_np = y.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)

    kf = KFold(n_splits=Config.n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_np), dtype=np.float32)
    test_pred = np.zeros(len(X_test_np), dtype=np.float32)

    params = {
        "hidden_dims": [256, 192, 160, 128],  # reduced size to be feasible
        "dropout": 0.3,
        "lr": 1e-3,
        "weight_decay": 1e-3
    }

    fold = 0
    for tr_idx, va_idx in kf.split(X_np, y_np):
        fold += 1
        print(f"\n--- Fold {fold} / {Config.n_splits} ---")
        X_tr, X_va = X_np[tr_idx], X_np[va_idx]
        y_tr, y_va = y_np[tr_idx], y_np[va_idx]

        va_out, val_rmse, model = train_one_fold(X_tr, y_tr, X_va, y_va, params,
                                                 epochs=800, batch_size=512, patience=Config.early_stop)
        print(f"Fold {fold} RMSE: {val_rmse:.6f}")
        oof[va_idx] = va_out

        # test preds accumulate
        with torch.no_grad():
            test_out = model(torch.from_numpy(X_test_np).to(Config.device)).cpu().numpy()
        test_pred += test_out / Config.n_splits

        # cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\nOOF RMSE:", rmse(y_np, oof))
    return oof, test_pred

# --------------------
# Main execution
# --------------------
def main():
    print("Device:", Config.device)
    # load raw
    df_train = Config.train.copy()
    df_test = Config.test.copy()

    # EDA
    run_eda(df_train, df_test)

    # Transform / encode / scale
    transformer = Transform(df_train, df_test)
    X_enc, y, test_enc = transformer.get()

    print("Processed X shape:", X_enc.shape)
    print("Processed test shape:", test_enc.shape)

    # quick LightGBM baseline and show importance + quick test preds (optional)
    lgb_preds = lgb_baseline(X_enc, y, test_enc)

    # if you want to use LGB baseline as submission immediately, uncomment:
    # sub = Config.submission.copy()
    # sub[Config.target] = lgb_preds
    # sub.to_csv("/kaggle/working/submission_lgb.csv", index=False)

    # If deep training requested (heavy), run CV training
    if Config.training:
        oof, test_preds = fit_predict_cv(X_enc, y, test_enc)
    else:
        test_preds = lgb_preds

    # prepare submission
    submission = Config.submission.copy()
    submission[Config.target] = test_preds
    submission.to_csv("/kaggle/working/submission.csv", index=False)
    print("Saved /kaggle/working/submission.csv")
    display(submission.head())

if __name__ == "__main__":
    main()


