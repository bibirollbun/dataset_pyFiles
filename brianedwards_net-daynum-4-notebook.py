!nvcc --version
!pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120
!pip3 install -U numpy pandas scikit-learn xgboost catboost lightgbm
!pip3 install -U --extra-index-url=https://pypi.nvidia.com "cudf-cu12==25.2.*" "cuml-cu12==25.2.*"
print("Hello world")


import warnings

warnings.simplefilter("ignore")

import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
import catboost as ctb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import cudf
from cuml.preprocessing import StandardScaler as CuStandardScaler

pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', 6)
pd.set_option("display.width", None)


def tensor(data):
    return torch.tensor(data, dtype=torch.float32, device="cuda")

def weight(*size):
    return nn.Parameter(0.1 * torch.randn(*size, dtype=torch.float32, device="cuda"))

def zeros(*size):
    return torch.zeros(*size, dtype=torch.float32, device="cuda")

def bias(*size):
    return nn.Parameter(zeros(*size))

def forward(m, X_i):
    y_pred = X_i
    for j, (w, b) in enumerate(m):
        y_pred = y_pred @ w + b
        if j < (len(m)-1):
            y_pred = F.leaky_relu(y_pred, negative_slope=0.1)
    return y_pred

mse_ = torch.nn.MSELoss()

def mse(y_pred_epoch, y_i):
    return  mse_(y_pred_epoch, y_i.view(-1, 1))

def aslist(param):
    return param.cpu().detach().numpy().tolist()

def aspy(m):
    return [(aslist(w), aslist(b)) for w, b in m]

def scale_back_to_margin(sy, y_pred):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    return sy.inverse_transform(y_pred.reshape(-1, 1)).flatten()

def brier_score(margin_true, margin_pred):
    win_true = (margin_true > 0).astype("int32")
    win_prob_pred = 1 / (1 + np.exp(-margin_pred * 0.175))
    return np.mean((win_prob_pred - win_true) ** 2)

def train_nn(train):
    sx = StandardScaler()
    X = tensor(sx.fit_transform(train.select_dtypes("float32")))
    print(f"X {X.shape}")

    sy = StandardScaler()
    y = tensor(sy.fit_transform(train[["Margin"]]))
    print(f"y {y.shape}")
    
    d = [X.shape[1], 64, 32, 16, y.shape[1]]
    n_epochs = 10_000
    patience = 60
    kfold = KFold(shuffle=True, random_state=42)
    y_pred_oof = torch.zeros(y.shape[0], dtype=torch.float32, device="cuda")
    models = []

    for fold_n, (i_fold, i_oof) in enumerate(kfold.split(X), 1):
        print(f"  fold {fold_n}")
        start = datetime.now()

        m = [(  weight(d[i], d[i+1]),
                bias(d[i+1]),
            )
            for i in range(len(d)-1)]

        optim = torch.optim.Adam(
            [h[0] for h in m] + [h[1] for h in m],
            weight_decay=1e-4)

        for epoch_n in range(1, n_epochs + 1):
            y_pred_epoch_fold = forward(m, X[i_fold])
            mse_epoch_fold = mse(y_pred_epoch_fold, y[i_fold])
            optim.zero_grad()
            mse_epoch_fold.backward()
            optim.step()

            with torch.no_grad():
                y_pred_epoch_oof = forward(m, X[i_oof])
                mse_epoch_oof = mse(y_pred_epoch_oof, y[i_oof])

            if epoch_n == 1 or m_best[0] > mse_epoch_oof:
                m_best = (mse_epoch_oof, 0, aspy(m))
            else:
                m_best = (m_best[0], m_best[1]+1, m_best[2])

            if ((epoch_n % (n_epochs // 100) == 0)
                    or (epoch_n > (n_epochs - 3))
                    or (m_best[1] > patience)):
                print(
                    f"    epoch {epoch_n:>6}: "
                    f"fold={mse_epoch_fold.item():.4f} "
                    f"oof={mse_epoch_oof.item():.4f}"
                )

            if m_best[1] > patience:
                print(f"    out of patience: oof={m_best[0]:.4f}")
                break

        with torch.no_grad():
            m = [(tensor(w), tensor(b)) for w, b in m_best[2]]
            y_pred_oof[i_oof] = forward(m, X[i_oof]).flatten()

        models.append(aspy(m))
        t = (datetime.now() - start).total_seconds()
        print(f"  done fold {fold_n} {t} seconds")

    margin_pred_oof = scale_back_to_margin(sy, y_pred_oof)
    score = brier_score(train["Margin"], margin_pred_oof)
    print(f"nn oof brier score: {score:.4f}")
    return sx, sy, models

def test_nn(test, sx, sy, models):
    X = tensor(sx.transform(test.select_dtypes("float32")))
    y = sy.fit(test[["Margin"]])
    y_pred = zeros(X.shape[0])

    for m_py in models:
        m = [(tensor(w), tensor(b)) for w, b in m_py]
        with torch.no_grad():
            y_pred += forward(m, X).flatten()

    margin_pred = scale_back_to_margin(sy, y_pred/len(models))
    score = brier_score(test["Margin"], margin_pred)
    print(f"nn test brier score: {score:.4f}")


train = pd.read_csv("../input/net-daynum-4-dataset/train_daynum.csv")

train = pd.concat([
    train.select_dtypes("int64").astype("int32"),
    train.select_dtypes("float64").astype("float32"),
], axis=1)

train.loc[train["TeamID"] < train["OppID"], "ID"] = \
    train["Season"].astype("str") + "_" + train["TeamID"].astype("str") + "_" + train["OppID"].astype("str")

train.loc[train["OppID"] < train["TeamID"], "ID"] = \
    train["Season"].astype("str") + "_" + train["OppID"].astype("str") + "_" + train["TeamID"].astype("str")

stage1 = pd.read_csv("../input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv")
test = train[train['ID'].isin(stage1['ID'])]
train = train[~train['ID'].isin(stage1['ID'])]
print(f"train {train.shape}")
print(f"test {test.shape}")


sx, sy, models = train_nn(train)


test_nn(test, sx, sy, models)


def train_xgb(train):
    df = cudf.DataFrame.from_pandas(train)

    sx = CuStandardScaler()
    X = sx.fit_transform(df.select_dtypes("float32"))
    print(f"X {X.shape}")

    sy = CuStandardScaler()
    y = sy.fit_transform(df[["Margin"]])
    print(f"y {y.shape}")
    
    kfold = KFold(shuffle=True, random_state=42)
    y_pred_oof = cudf.Series(np.zeros(len(y), dtype=np.float32))
    models = []

    for fold_n, (i_fold, i_oof) in enumerate(kfold.split(range(len(y))), 1):
        print(f"  fold {fold_n}")
        start = datetime.now()
        
        m = xgb.XGBRegressor(
            tree_method="hist",
            device="cuda",
            
            # learning_rate=0.02,
            # learning_rate=0.05,
            # learning_rate=0.10,
            learning_rate=0.20,
            # learning_rate=0.30,  # default

            # max_depth=7,
            max_depth=6,  # default
            # max_depth=5,

            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_estimators=10000,
            random_state=42,

            # callbacks=[xgb.callback.EarlyStopping(rounds=60)],
            early_stopping_rounds=60,
        )
        
        m.fit(
            X.iloc[i_fold].to_pandas(), y[i_fold].to_pandas(),
            eval_set=[(X.iloc[i_oof].to_pandas(), y[i_oof].to_pandas())],
            verbose=100,
        )
        
        y_pred_oof.iloc[i_oof] = m.predict(X.iloc[i_oof].to_pandas())
        models.append(m)
        
        print(f"    best iteration: {m.best_iteration}, oof rmse: {m.best_score:.4f}")
        t = (datetime.now() - start).total_seconds()
        print(f"  done fold {fold_n} {t} seconds")

    margin_pred_oof = scale_back_to_margin(sy, y_pred_oof.values.get())
    score = brier_score(train["Margin"].values, margin_pred_oof)
    print(f"xgb oof brier score: {score:.4f}")
    return sx, sy, models

def test_xgb(test, sx, sy, models):
    df = cudf.DataFrame.from_pandas(test)
    X = sx.transform(df.select_dtypes("float32"))
    y_pred = np.zeros(len(X), dtype=np.float32)
    
    for m in models:
        y_pred += m.predict(X.to_pandas())
    
    margin_pred = scale_back_to_margin(sy, y_pred/len(models))
    score = brier_score(test["Margin"].values, margin_pred)
    print(f"xgb test brier score: {score:.4f}")
    return margin_pred

def train_lgb(train):
    df = cudf.DataFrame.from_pandas(train)
    
    sx = CuStandardScaler()
    X = sx.fit_transform(df.select_dtypes("float32"))
    print(f"X {X.shape}")

    sy = CuStandardScaler()
    y = sy.fit_transform(df[["Margin"]])
    print(f"y {y.shape}")
    
    kfold = KFold(shuffle=True, random_state=42)
    y_pred_oof = cudf.Series(np.zeros(len(y), dtype=np.float32))
    models = []

    for fold_n, (i_fold, i_oof) in enumerate(kfold.split(range(len(y))), 1):
        print(f"  fold {fold_n}")
        start = datetime.now()
        
        m = lgb.LGBMRegressor(
            device="gpu",
            
            # max_depth=-1,  # default
            # max_depth=8,
            max_depth=3,
            
            feature_fraction=1.0,  # default
            # feature_fraction=0.8,
            # feature_fraction=0.4,

            num_iterations=10000,
            # num_iterations=2500,
            # num_iterations=100,  # default
            
            learning_rate=0.10,  # default
            # learning_rate=0.02,
            # learning_rate=0.01,
            
            # num_leaves=127,
            num_leaves=31,  # default
            # num_leaves=8,
            
            # min_data_in_leaf=20,  # default
            
            # bagging_freq=0,  # default
            # bagging_fraction=1.0,  # default
            # bagging_fraction=0.8,
            
            lambda_l1=0.1,
            # lambda_l1=0.0,  # default

            lambda_l2=1.0,
            # lambda_l2=0.0,  # default

            early_stopping_round=60,
            # early_stopping_round=0,  # default
            
            random_state=42,
            verbosity=-1,  # -1=Fatal, 0=Error/Warning, 1=Info
            verbose=-1,
        )
        
        m.fit(
            X.iloc[i_fold].to_pandas(), y[i_fold].to_pandas(),
            eval_set=[(X.iloc[i_oof].to_pandas(), y[i_oof].to_pandas())],
            eval_metric="rmse",
            # callbacks=[lgb.early_stopping(stopping_rounds=60)],
        )
        
        y_pred_oof.iloc[i_oof] = m.predict(X.iloc[i_oof].to_pandas())
        models.append(m)
        print(f"    best iteration: {m.best_iteration_}, oof rmse: {m.best_score_['valid_0']['rmse']:.4f}")
        t = (datetime.now() - start).total_seconds()
        print(f"  done fold {fold_n} {t} seconds")

    margin_pred_oof = scale_back_to_margin(sy, y_pred_oof.values.get())
    score = brier_score(train["Margin"].values, margin_pred_oof)
    print(f"lgb oof brier score: {score:.4f}")
    return sx, sy, models

def test_lgb(test, sx, sy, models):
    df = cudf.DataFrame.from_pandas(test)
    X = sx.transform(df.select_dtypes("float32"))
    y_pred = np.zeros(len(X), dtype=np.float32)
    
    for m in models:
        y_pred += m.predict(X.to_pandas())
    
    margin_pred = scale_back_to_margin(sy, y_pred/len(models))
    score = brier_score(test["Margin"].values, margin_pred)
    print(f"lgb test brier score: {score:.4f}")
    return margin_pred

def train_ctb(train):
    df = cudf.DataFrame.from_pandas(train)
    
    sx = CuStandardScaler()
    X = sx.fit_transform(df.select_dtypes("float32"))
    print(f"X {X.shape}")

    sy = CuStandardScaler()
    y = sy.fit_transform(df[["Margin"]])
    print(f"y {y.shape}")
    
    kfold = KFold(shuffle=True, random_state=42)
    y_pred_oof = cudf.Series(np.zeros(len(y), dtype=np.float32))
    models = []

    for fold_n, (i_fold, i_oof) in enumerate(kfold.split(range(len(y))), 1):
        print(f"  fold {fold_n}")
        start = datetime.now()

        m = ctb.CatBoostRegressor(
            task_type="GPU",
            devices="0",
            learning_rate=0.02,
            depth=8,
            min_data_in_leaf=20,
            l2_leaf_reg=3.0,

            # bootstrap_type="Bernoulli",
            # bootstrap_type="Bayesian",
            
            # subsample=0.8,
            # colsample_bylevel=0.8,

            iterations=10000,
            
            # early_stopping_rounds=60,
            od_type="Iter",
            od_wait=60,
            
            random_seed=42,
        )
        
        m.fit(
            X.iloc[i_fold].to_pandas(), y[i_fold].to_pandas(),
            eval_set=[(X.iloc[i_oof].to_pandas(), y[i_oof].to_pandas())],
            verbose=100
        )
        
        y_pred_oof.iloc[i_oof] = m.predict(X.iloc[i_oof].to_pandas())
        models.append(m)
        print(f"    best iteration: {m.best_iteration_}, oof rmse: {m.best_score_}")
        t = (datetime.now() - start).total_seconds()
        print(f"  done fold {fold_n} {t} seconds")


    margin_pred_oof = scale_back_to_margin(sy, y_pred_oof.values.get())
    score = brier_score(train["Margin"].values, margin_pred_oof)
    print(f"ctb oof brier score: {score:.4f}")
    return sx, sy, models

def test_ctb(test, sx, sy, models):
    df = cudf.DataFrame.from_pandas(test)
    X = sx.transform(df.select_dtypes("float32"))
    y_pred = np.zeros(len(X), dtype=np.float32)
    
    for m in models:
        y_pred += m.predict(X.to_pandas())
    
    margin_pred = scale_back_to_margin(sy, y_pred/len(models))
    score = brier_score(test["Margin"].values, margin_pred)
    print(f"ctb test brier score: {score:.4f}")
    return margin_pred


sx, sy, models = train_xgb(train)
test_xgb(test, sx, sy, models)


sx, sy, models = train_lgb(train)
test_lgb(test, sx, sy, models)


sx, sy, models = train_ctb(train)
test_ctb(test, sx, sy, models)




