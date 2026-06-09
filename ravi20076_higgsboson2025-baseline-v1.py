


!pip install uv -q
!uv pip install -q scikit-learn==1.6.1 xgboost==2.1.4 --system

import pandas as pd, numpy as np
from tqdm.notebook import tqdm
from gc import collect

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold as SKF
from xgboost import XGBClassifier as XGBC
from lightgbm import LGBMClassifier as LGBMC, log_evaluation, early_stopping
from catboost import CatBoostClassifier as CBC
from sklearn.linear_model import LogisticRegression as LRC



test_req = False
target   = "label"
n_splits = 10 if test_req == False else 3


train  = pd.read_csv(f"/kaggle/input/higgs-boson-detection-2025/train.csv")
test   = pd.read_csv(f"/kaggle/input/higgs-boson-detection-2025/test.csv")
sub_fl = pd.read_csv(f"/kaggle/input/higgs-boson-detection-2025/sample_submission.csv")

Xtrain = train.drop(target, axis=1)
ytrain = train[target].astype(np.uint8)
Xtest  = test.copy()

Xtrain.index = range(len(Xtrain))
ytrain.index = range(len(Xtrain))
Xtest.index  = range(len(Xtest))

print(f"---> Shapes = {Xtrain.shape} {ytrain.shape} {Xtest.shape}")




def ScoreMetric(ytrue, ypreds) :
    return roc_auc_score(ytrue, ypreds)

cv        = SKF(n_splits = n_splits, random_state = 42, shuffle = True)
OOF_Preds = {}
Mdl_Preds = {}
FtreImp   = {}


print(f"\n ================== OFFLINE MODEL TRAINING - XGB ================== \n")


oof_preds    = np.zeros(len(Xtrain))
mdl_preds    = np.zeros(len(Xtest))
ftreimp      = 0

for fold_nb, (train_idx, dev_idx) in enumerate(cv.split(Xtrain, ytrain) ):
    print(f"\n ============== FOLD {fold_nb} ==============")

    Xtr, ytr   = Xtrain.iloc[train_idx], ytrain.loc[train_idx]
    Xdev, ydev = Xtrain.iloc[dev_idx],   ytrain.loc[dev_idx]
    print(f"\n---> Shapes {Xtr.shape} {Xdev.shape} {ytr.shape} {ydev.shape}")

    model = \
    XGBC(
        n_estimators          = 5_000,
        random_state          = 42,
        objective             = "binary:logistic",
        eval_metric           = "auc",
        learning_rate         = 0.02,
        max_depth             = 6, 
        reg_alpha             = 0.001,
        reg_lambda            = 0.001,
        colsample_bytree      = 0.35,
        colsample_bylevel     = 0.40,
        device                = "cpu",
        enable_categorical    = True,    
        early_stopping_rounds = 100,
    )

    model.fit(Xtr, ytr, eval_set = [(Xdev, ydev)], verbose = 500)
    dev_preds  = model.predict_proba(Xdev)[:,1]
    test_preds = model.predict_proba(Xtest)[:,1]
    score      = ScoreMetric(ydev, dev_preds)
    print(f"\n---> score = {score:,.8f}")
    
    oof_preds[dev_idx] = dev_preds
    mdl_preds = mdl_preds + (test_preds / n_splits)
    ftreimp  += model.feature_importances_
    
score = ScoreMetric(ytrain, oof_preds)
print(f"\n---> Overall score = {score :,.8f}")

OOF_Preds["XGB"] = oof_preds
Mdl_Preds["XGB"] = mdl_preds
FtreImp["XGB"]   = pd.Series(ftreimp / n_splits, index =  Xtest.columns)



print(f"\n ================== OFFLINE MODEL TRAINING - LGBM ================== \n")

oof_preds    = np.zeros(len(Xtrain))
mdl_preds    = np.zeros(len(Xtest))
ftreimp      = 0

for fold_nb, (train_idx, dev_idx) in enumerate(cv.split(Xtrain, ytrain) ):
    print(f"\n ============== FOLD {fold_nb} ==============")

    Xtr, ytr   = Xtrain.iloc[train_idx], ytrain.loc[train_idx]
    Xdev, ydev = Xtrain.iloc[dev_idx],   ytrain.loc[dev_idx]
    print(f"\n---> Shapes {Xtr.shape} {Xdev.shape} {ytr.shape} {ydev.shape}")

    model = \
    LGBMC(
        n_estimators       = 5_000,
        random_state       = 42,
        objective          = "binary",
        metric             = "auc",
        learning_rate      = 0.02,
        max_depth          = 6, 
        reg_alpha          = 0.001,
        reg_lambda         = 0.001,
        colsample_bytree   = 0.45,
        device             = "cpu", 
        verbosity          = -1,
    )

    model.fit(
        Xtr, ytr, 
        eval_set = [(Xdev, ydev)], 
        eval_names = [("Dev")],
        callbacks = [log_evaluation(500), early_stopping(100, verbose = False)],
    )
    
    dev_preds  = model.predict_proba(Xdev)[:,1]
    test_preds = model.predict_proba(Xtest)[:,1]
    score      = ScoreMetric(ydev, dev_preds)
    print(f"\n---> score = {score:,.8f}")
    
    oof_preds[dev_idx] = dev_preds
    mdl_preds = mdl_preds + (test_preds / n_splits)
    ftreimp  += model.feature_importances_
     
score = ScoreMetric(ytrain, oof_preds)
print(f"\n---> Overall score = {score :,.8f}")

OOF_Preds["LGBM"] = oof_preds
Mdl_Preds["LGBM"] = mdl_preds
FtreImp["LGBM"]   = pd.Series(ftreimp / n_splits, index = Xtest.columns)



print(f"\n ================== OFFLINE MODEL TRAINING - CATBOOST ================== \n")

oof_preds    = np.zeros(len(Xtrain))
mdl_preds    = np.zeros(len(Xtest))
ftreimp      = 0

for fold_nb, (train_idx, dev_idx) in enumerate(cv.split(Xtrain, ytrain) ):
    print(f"\n ============== FOLD {fold_nb} ==============")

    Xtr, ytr   = Xtrain.iloc[train_idx], ytrain.loc[train_idx]
    Xdev, ydev = Xtrain.iloc[dev_idx],   ytrain.loc[dev_idx]
    print(f"\n---> Shapes {Xtr.shape} {Xdev.shape} {ytr.shape} {ydev.shape}")

    model = \
    CBC(
        iterations = 5_000,
        loss_function = "Logloss",
        eval_metric   = "AUC",
        verbose       = 0,
        learning_rate = 0.02,
        max_depth     = 6,
        colsample_bylevel = 0.50,
        l2_leaf_reg   = 0.25,
        early_stopping_rounds = 50,
    )

    model.fit(
        Xtr, ytr, 
        eval_set = [(Xdev, ydev)], 
        verbose  = 500,
    )
    
    dev_preds  = model.predict_proba(Xdev)[:,1]
    test_preds = model.predict_proba(Xtest)[:,1]
    score      = ScoreMetric(ydev, dev_preds)
    print(f"\n---> score = {score:,.8f}")
    
    oof_preds[dev_idx] = dev_preds
    mdl_preds = mdl_preds + (test_preds / n_splits)
    ftreimp  += model.feature_importances_
     
score = ScoreMetric(ytrain, oof_preds)
print(f"\n---> Overall score = {score :,.8f}")

OOF_Preds["CB"] = oof_preds
Mdl_Preds["CB"] = mdl_preds
FtreImp["CB"]   = pd.Series(ftreimp / n_splits, index = Xtest.columns)


df = pd.DataFrame(FtreImp)

with sns.axes_style("white") :
    fig, axes = \
    plt.subplots(
        len(df.columns), 1, 
        figsize = (20, 20), 
        gridspec_kw = {"hspace": 0.30},
        sharex = True,
    )
    
    for i, col in enumerate(df.columns):
        ax = axes[i]
        df[[col]].plot.bar(ax = ax, color = "tab:blue")
        ax.set_title(f"{col}", fontweight = "bold", color = "black")

    plt.suptitle(f"Feature importances", 
                 y = 0.95, 
                 color = "maroon", fontsize = 16, fontweight = "bold"
                )
    
    plt.show()
          


print(f"\n ================== OFFLINE MODEL TRAINING - BLENDER ================== \n")

Xtrain       = pd.DataFrame(OOF_Preds)
Xtest        = pd.DataFrame(Mdl_Preds)
oof_preds    = np.zeros(len(Xtrain))
mdl_preds    = np.zeros(len(Xtest))

for fold_nb, (train_idx, dev_idx) in enumerate(cv.split(Xtrain, ytrain) ):
    print(f"\n ============== FOLD {fold_nb} ==============")

    Xtr, ytr   = Xtrain.iloc[train_idx], ytrain.loc[train_idx]
    Xdev, ydev = Xtrain.iloc[dev_idx],   ytrain.loc[dev_idx]
    print(f"\n---> Shapes {Xtr.shape} {Xdev.shape} {ytr.shape} {ydev.shape}")

    model = LRC(random_state = 42)

    model.fit(Xtr, ytr)
    dev_preds  = model.predict_proba(Xdev)[:,1]
    test_preds = model.predict_proba(Xtest)[:,1]
    score      = ScoreMetric(ydev, dev_preds)
    print(f"\n---> score = {score:,.8f}")
    
    oof_preds[dev_idx] = dev_preds
    mdl_preds = mdl_preds + (test_preds / n_splits)
     
score = ScoreMetric(ytrain, oof_preds)
print(f"\n---> Overall score = {score :,.8f}")




public_nb           = pd.read_csv(f"/kaggle/input/hbd2025-tabm/sub.csv")["Predicted"].values
sub_fl["Predicted"] = mdl_preds * 0.3 + public_nb * 0.7
sub_fl['Id']        = sub_fl['Id'].astype(np.int64)
sub_fl['Id']        = sub_fl['Id'].apply(lambda x: f"{float(x):.18e}")

sub_fl.to_csv("sample_submission.csv", index = None)

!ls
!head sample_submission.csv

