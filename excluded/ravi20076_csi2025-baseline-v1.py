


%%writefile mytraining.py

import pandas as pd, numpy as np

from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import *

from xgboost import XGBRegressor as XGBR
from lightgbm import LGBMRegressor as LGBMR
from catboost import CatBoostRegressor as CBR
from sklearn.ensemble import *

from warnings import filterwarnings
filterwarnings("ignore")
from gc import collect 
from tqdm.auto import tqdm

target = "cosmic_stability_index"


%%writefile -a mytraining.py

train  = pd.read_csv(f"/kaggle/input/tda-aiml-cosmic-stability-problem-0f3ebc/train.csv", index_col = "id")
test   = pd.read_csv(f"/kaggle/input/tda-aiml-cosmic-stability-problem-0f3ebc/test.csv", index_col = "id")
sub_fl = pd.read_csv(f"/kaggle/input/tda-aiml-cosmic-stability-problem-0f3ebc/submission.csv", index_col = "id")

print(f"---> Shapes = {train.shape} {test.shape} {sub_fl.shape}\n")


%%writefile -a mytraining.py

OOF_Preds = []
Mdl_Preds = []

cv = KFold(n_splits = 5, shuffle = True, random_state = 42)


%%writefile -a mytraining.py

oof_preds = np.zeros(len(train))
mdl_preds = np.zeros(len(test))

for fold_nb, (train_idx, dev_idx) in enumerate(cv.split(train, train[target])) :
    Xtr  = train.loc[train_idx].drop(target, axis = 1)
    Xdev = train.loc[dev_idx].drop(target, axis = 1)
    ytr  = train.loc[train_idx, target]
    ydev = train.loc[train_idx, target] 

    model = RandomForestRegressor(
            n_estimators = 800,
            random_state = 42,
            n_jobs       = -1,
            min_samples_leaf = 4,
        )

    model.fit(Xtr, ytr)
    oof_preds[dev_idx] = model.predict(Xdev)
    mdl_preds += model.predict(test) / cv.n_splits

score = r2_score(train[target], oof_preds)
print(f"---> Overall score = {score:,.8f}")

OOF_Preds.append(oof_preds)
Mdl_Preds.append(mdl_preds)


%%writefile -a mytraining.py

oof_preds = np.zeros(len(train))
mdl_preds = np.zeros(len(test))

for fold_nb, (train_idx, dev_idx) in enumerate( cv.split(train, train[target]) ) :
    Xtr  = train.loc[train_idx].drop(target, axis = 1)
    Xdev = train.loc[dev_idx].drop(target, axis = 1)
    ytr  = train.loc[train_idx, target]
    ydev = train.loc[train_idx, target] 

    model = RandomForestRegressor(
            n_estimators = 100,
            random_state = 42,
            n_jobs       = -1,
            min_samples_leaf = 2,
        )

    model.fit(Xtr, ytr)
    oof_preds[dev_idx] = model.predict(Xdev)
    mdl_preds += model.predict(test) / cv.n_splits

score = r2_score(train[target], oof_preds)
print(f"---> Overall score = {score:,.8f}")

OOF_Preds.append(oof_preds)
Mdl_Preds.append(mdl_preds)


%%writefile -a mytraining.py

oof_preds = np.zeros(len(train))
mdl_preds = np.zeros(len(test))

for fold_nb, (train_idx, dev_idx) in enumerate( cv.split(train, train[target]) ):
    Xtr  = train.loc[train_idx].drop(target, axis = 1)
    Xdev = train.loc[dev_idx].drop(target, axis = 1)
    ytr  = train.loc[train_idx, target]
    ydev = train.loc[train_idx, target] 

    model = RandomForestRegressor(
            n_estimators = 500,
            random_state = 42,
            n_jobs       = -1,
            min_samples_leaf = 8,
        )

    model.fit(Xtr, ytr)
    oof_preds[dev_idx] = model.predict(Xdev)
    mdl_preds += model.predict(test) / cv.n_splits

score = r2_score(train[target], oof_preds)
print(f"---> Overall score = {score:,.8f}")

OOF_Preds.append(oof_preds)
Mdl_Preds.append(mdl_preds)


%%writefile -a mytraining.py

oof_preds = np.zeros(len(train))
mdl_preds = np.zeros(len(test))

for fold_nb, (train_idx, dev_idx) in enumerate( cv.split(train, train[target]) ):
    Xtr  = train.loc[train_idx].drop(target, axis = 1)
    Xdev = train.loc[dev_idx].drop(target, axis = 1)
    ytr  = train.loc[train_idx, target]
    ydev = train.loc[train_idx, target] 

    model = RandomForestRegressor(
            n_estimators = 450,
            random_state = 42,
            n_jobs       = -1,
            min_samples_leaf = 12,
        )

    model.fit(Xtr, ytr)
    oof_preds[dev_idx] = model.predict(Xdev)
    mdl_preds += model.predict(test) / cv.n_splits

score = r2_score(train[target], oof_preds)
print(f"---> Overall score = {score:,.8f}")

OOF_Preds.append(oof_preds)
Mdl_Preds.append(mdl_preds)


%%writefile -a mytraining.py

oof_preds = np.zeros(len(train))
mdl_preds = np.zeros(len(test))

for fold_nb, (train_idx, dev_idx) in enumerate( cv.split(train, train[target]) ):
    Xtr  = train.loc[train_idx].drop(target, axis = 1)
    Xdev = train.loc[dev_idx].drop(target, axis = 1)
    ytr  = train.loc[train_idx, target]
    ydev = train.loc[train_idx, target] 

    model = ExtraTreesRegressor(
        n_estimators = 800,
        random_state = 42,
        n_jobs       = -1,
        min_samples_leaf = 4,
    )

    model.fit(Xtr, ytr)
    oof_preds[dev_idx] = model.predict(Xdev)
    mdl_preds += model.predict(test) / cv.n_splits

score = r2_score(train[target], oof_preds)
print(f"---> Overall score = {score:,.8f}")

OOF_Preds.append(oof_preds)
Mdl_Preds.append(mdl_preds)


%%writefile -a mytraining.py

oof_preds = np.zeros(len(train))
mdl_preds = np.zeros(len(test))

for fold_nb, (train_idx, dev_idx) in enumerate( cv.split(train, train[target]) ):
    Xtr  = train.loc[train_idx].drop(target, axis = 1)
    Xdev = train.loc[dev_idx].drop(target, axis = 1)
    ytr  = train.loc[train_idx, target]
    ydev = train.loc[train_idx, target] 

    model = ExtraTreesRegressor(
        n_estimators = 200,
        random_state = 42,
        n_jobs       = -1,
        min_samples_leaf = 2,
    )

    model.fit(Xtr, ytr)
    oof_preds[dev_idx] = model.predict(Xdev)
    mdl_preds += model.predict(test) / cv.n_splits

score = r2_score(train[target], oof_preds)
print(f"---> Overall score = {score:,.8f}")

OOF_Preds.append(oof_preds)
Mdl_Preds.append(mdl_preds)


%%writefile -a mytraining.py

OOF_Preds = np.mean( np.stack(OOF_Preds, axis=1), axis=1).flatten()
Mdl_Preds = np.mean( np.stack(Mdl_Preds, axis=1), axis=1).flatten()

score = r2_score(train[target], OOF_Preds)
print(f"---> Overall ensemble score = {score:,.8f} | without clipping")
score = r2_score(
    train[target], 
    np.clip(OOF_Preds, a_min = train[target].min(), a_max = train[target].max())
)
print(f"---> Overall ensemble score = {score:,.8f} | with clipping" )

sub_fl[target] = np.clip(
    Mdl_Preds, a_min = train[target].min(), a_max = train[target].max()
)
sub = pd.read_csv(
    f"/kaggle/input/cosmic-stability-index-regression-challenge/submission.csv"
)[target].values
sub_fl[target] = sub_fl[target].values * 0.50 + sub * 0.50
sub_fl.to_csv(f"submission.csv", index = True)


!pip install -q scikit-learn==1.7.2
!python mytraining.py
print()
!head submission.csv

