


!pip install -q polars==1.29.0      --no-index --find-links=/kaggle/input/cmi2025-public-imports-v2/packages
!pip install -q lightgbm==4.6.0     --no-index --find-links=/kaggle/input/cmi2025-public-imports-v2/packages
!pip install -q scikit-learn==1.6.1 --no-index --find-links=/kaggle/input/cmi2025-public-imports-v2/packages

import os
import numpy as np
import polars as pl
import pandas as pd
import joblib
from tqdm.notebook import tqdm
from gc import collect
from pprint import pprint

import torch

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, PredefinedSplit
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.metrics import *
from lightgbm import LGBMClassifier as LGBMC, log_evaluation, early_stopping
from catboost import CatBoostClassifier as CBC



target  = "gesture"
grouper = "subject"

mapper = \
{
    "Above ear - pull hair" : 0,
    "Cheek - pinch skin" : 1,
    "Eyebrow - pull hair" : 2,
    "Eyelash - pull hair" : 3, 
    "Forehead - pull hairline" : 4,
    "Forehead - scratch" : 5,
    "Neck - pinch skin" : 6, 
    "Neck - scratch" : 7,
    
    "Drink from bottle/cup" : 8,
    "Feel around in tray and pull out an object" : 9,
    "Glasses on/off" : 10,
    "Pinch knee/leg skin" : 11, 
    "Pull air toward your face" : 12,
    "Scratch knee/leg skin" : 13,
    "Text on phone" : 14,
    "Wave hello" : 15,
    "Write name in air" : 16,
    "Write name on leg" : 17,
}


%%time 

def make_ftre(sequence : pl.DataFrame, demographics : pl.DataFrame) :
    """
    Makes aggregate columns for the model
    Source - https://www.kaggle.com/code/farisalahmdi/lgbm-inference
    """
    
    agg_exprs = []
    stat_cols = ['acc_x','acc_y','acc_z','rot_w','rot_x','rot_y','rot_z']
    
    for c in stat_cols:
        agg_exprs += [
            pl.col(c).mean().alias(f"{c}_mean"),
            pl.col(c).std().alias(f"{c}_std"),
            pl.col(c).var().alias(f"{c}_var"),
            pl.col(c).quantile(0.25).alias(f"{c}_q25"),
            pl.col(c).median().alias(f"{c}_q50"),
            pl.col(c).quantile(0.75).alias(f"{c}_q75"),
            pl.col(c).max().alias(f"{c}_max"),
            pl.col(c).min().alias(f"{c}_min"),
            pl.col(c).first().alias(f"{c}_first"),
            pl.col(c).last().alias(f"{c}_last"),
            pl.col(c).quantile(0.25, "nearest").alias(f"{c}_t25"),
            pl.col(c).quantile(0.75, "nearest").alias(f"{c}_t75"),
            (pl.col(c).last() - pl.col(c).first()).alias(f"{c}_delta"),
            pl.corr("sequence_counter", c).alias(f"{c}_corr_time"),
            pl.col(c).diff().mean().alias(f"{c}_diff_mean"),
            pl.col(c).diff().std().alias(f"{c}_diff_std"),
            pl.col(c).skew().alias(f"{c}_skew"),
            pl.col(c).kurtosis().alias(f"{c}_kurt"),
            pl.col(c).diff().abs().gt(0).sum().alias(f"{c}_n_changes")
        ]
        
        agg_exprs += [
                pl.when(pl.col("sequence_counter") < 0.1 * pl.max("sequence_counter"))
                  .then(pl.col(c)).otherwise(None).mean().alias(f"{c}_seg1_mean"),
                pl.when(pl.col("sequence_counter") > 0.9 * pl.max("sequence_counter"))
                  .then(pl.col(c)).otherwise(None).mean().alias(f"{c}_seg3_mean"),
        ]
    
    return \
    (
        sequence.
        group_by(pl.col(["sequence_id", "subject"]), maintain_order=True).
        agg(agg_exprs).
        select(pl.all().shrink_dtype()).
        join(
            demographics.select(pl.all().shrink_dtype()),
            how = "left",
            on = ["subject"],
        ).
        drop(["subject"], strict = False).
        to_pandas()
    )
    



%%time 
 
train    = pl.read_csv(f"/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
traind   = pl.read_csv(f"/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
test     = pl.read_csv(f"/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
testd    = pl.read_csv(f"/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")

sel_cols = [ c for c in test.collect_schema().names() if "thm" not in c and "tof" not in c ]

Xtrain         = make_ftre(train.select(pl.col(sel_cols)), traind)
Xtest          = make_ftre(test.select(pl.col(sel_cols)),  testd)

Ytrain         = train.select(pl.col(["sequence_id", "subject", target])).unique().to_pandas()
Ytrain[target] = Ytrain[target].map(mapper).astype(np.int8)
Ytrain         = Xtrain[["sequence_id"]].merge(Ytrain, how = "inner", on = ["sequence_id"])
ytrain         = Ytrain[target]

ygrp = np.zeros(len(Xtrain))
cv   = StratifiedGroupKFold(n_splits = 5, shuffle = True, random_state = 42)

for fold_nb, (train_idx, dev_idx) in tqdm( 
    enumerate(cv.split(Xtrain, Ytrain[target], Ytrain[grouper] )) 
) :
    ygrp[dev_idx] = fold_nb

cv = PredefinedSplit(ygrp)
print(
    f"\n\n---> Shape = {Xtrain.shape} {ytrain.shape} {ygrp.shape} {Xtest.shape}"
)


def ScoreMetric(ytrue, ypreds)-> tuple:
    "Defines the score metric for the competition using the true and predicted labels"

    bscore = f1_score(
        np.where(ytrue  <= 7, 1, 0),
        np.where(ypreds <= 7, 1, 0),
        zero_division = 0.0,
    )

    mscore = f1_score(
        np.where(ytrue   <= 7, ytrue, 99),
        np.where(ypreds  <= 7, ypreds, 99),
        average = "macro", 
        zero_division = 0.0,
    )

    return (0.5 * (bscore + mscore), bscore, mscore)


%%time 

method = "LGBM1C"
mymodel = \
LGBMC(**{'device'             : "gpu" if torch.cuda.is_available() else "cpu",
         'objective'          : "multiclass",
         "n_estimators"       : 10000, 
         "max_depth"          : 8,
         "learning_rate"      : 0.025,
         "colsample_bytree"   : 0.55,               
         "n_jobs"             : -1,
         "num_leaves"         : 75,
         "random_state"       : 42,
         "reg_alpha"          : 0.001,
         "reg_lambda"         : 0.001,
         "subsample"          : 0.40,
         "verbosity"          : -1,
      }
)

oof_preds     = []
fitted_models = []
OOF_Preds     = {}
drop_cols     = ["sequence_id", target, grouper]

for fold_nb, (train_idx, dev_idx) in tqdm( 
    enumerate(cv.split(Xtrain, ytrain) , start = 1)
):
    
    Xtr  = Xtrain.iloc[train_idx].drop(drop_cols, axis = 1, errors = "ignore")
    Xdev = Xtrain.iloc[dev_idx].drop(drop_cols, axis = 1, errors = "ignore")
    ytr  = ytrain.iloc[train_idx]
    ydev = ytrain.iloc[dev_idx]

    model = clone(mymodel)
    model.fit(
        Xtr, ytr, 
        eval_set = [(Xdev, ydev)], 
        **{"callbacks" : [log_evaluation(0), early_stopping(100, verbose = False)]}
    )
    dev_preds = pd.DataFrame(model.predict_proba(Xdev), index = dev_idx)
    oof_preds.append(dev_preds)
    fitted_models.append(model)
    
    score, bscore, mscore = ScoreMetric(
        ydev.values.flatten(), 
        dev_preds.idxmax(axis=1).values.flatten()
    )
    print(
        f"---> OOF = {score:,.8f} | {bscore:,.8f} | {mscore:,.8f} -- Fold {fold_nb}"
    )

oof_preds = pd.concat(oof_preds, axis=0).sort_index(ascending = True)
OOF_Preds[method] = oof_preds

score, bscore, mscore = \
ScoreMetric(
    ytrain.values.flatten(), 
    oof_preds.idxmax(axis=1).values.flatten()
)
print(
    f"\n\n---> Overall OOF = {score:,.8f} | {bscore:,.8f} | {mscore:,.8f}"
)

collect()



%%time 

import kaggle_evaluation.cmi_inference_server

def predict(
    sequence     : pl.DataFrame, 
    demographics : pl.DataFrame,
) -> str:
    """
    Prediction function for Kaggle evaluation
    """

    drop_cols  = ["sequence_id", target, grouper]
    Xtest      = make_ftre(sequence, demographics).drop(drop_cols, axis=1, errors = "ignore")

    test_preds = []
    for model in fitted_models :
        test_preds.append( 
            pd.DataFrame(model.predict_proba(Xtest) , index = Xtest.index) 
        )

    test_preds = pd.concat(test_preds, axis=0).groupby(level = 0).mean()
    test_preds = test_preds.sort_index(ascending = True)
    pred = test_preds.idxmax(axis=1).map({v: k for k, v in mapper.items()}).values.flatten()[0]
    
    print(f"Final prediction = {pred}")
    return pred


inference_server = \
kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

