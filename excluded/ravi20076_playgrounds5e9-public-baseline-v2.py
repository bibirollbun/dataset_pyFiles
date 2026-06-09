


%load_ext cudf.pandas


%%time 

!pip install -q scikit-learn==1.7.1

import torch
from gc import collect 
import pandas as pd, numpy as np
import joblib
from tqdm.notebook import tqdm
from gc import collect

from cuml.preprocessing import TargetEncoder
from sklearn.metrics import *
from sklearn.model_selection import *
from lightgbm import LGBMRegressor as LGBMR, log_evaluation, early_stopping

import matplotlib.pyplot as plt
import seaborn as sns


target      = "BeatsPerMinute"
version_lbl = "V1_3"
drop_cols   = ["Source", "id", "Label"]

nb_orig     = 1
test_req    = False
nest        = 10 if test_req == True else 10_000

plot_req    = True
n_splits    = 5
n_repeats   = 1
FOLDS       = n_splits * n_repeats


%%time

Xtrain = pd.read_parquet(
    f"/kaggle/input/playgrounds5e9-public-features-v1/XYtrain.parquet"
)

extra  = Xtrain.loc[Xtrain.Source == "Original"]
Xtrain = Xtrain.loc[Xtrain.Source == "Competition"]
ytrain = Xtrain[target]

Xtest = pd.read_parquet(
    f"/kaggle/input/playgrounds5e9-public-features-v1/Xtest.parquet"
)

ygrp = pd.read_parquet(
    f"/kaggle/input/playgrounds5e9-public-features-v1/ygrp.parquet"
)

print(
    f"\n---> Shapes = {Xtrain.shape} {ytrain.shape} {Xtest.shape} {extra.shape}\n"
)

FEATURES = joblib.load(
    f"/kaggle/input/playgrounds5e9-public-features-v1/sel_cols.joblib"
)[0:-1]
te_cols = list( Xtest[FEATURES].filter(regex = "^C", axis=1).columns )

print()
with np.printoptions(linewidth = 150, threshold = 500) :
    print(f"\nTarget encoded columns\n")
    print(np.array(te_cols))
    
collect();


%%time 

def ScoreMetric(ytrue, ypreds) :
    "Calculates the competition metric for the OOF evaluation"
    return root_mean_squared_error(ytrue, ypreds)

params = {
   "objective"         : "regression_l2",
   "metric"            : "rmse",
   "learning_rate"     : 0.005,
   "max_depth"         : 6,
   "subsample"         : 0.90,
   "colsample_bytree"  : 0.35,
   "random_state"      : 42,
   "device"            : "gpu",
   "max_leaves"        : 32,
   "reg_alpha"         : 0.001,
   "max_bin"           : 255,
   "verbosity"         : -1,
}


%%time 

cv = PredefinedSplit(
    ygrp.iloc[0 : len(Xtrain)]["fold_nb"].values.flatten()
)

oof_preds  = []
test_preds = 0
ftreimp    = 0

for fold_nb, (train_idx, val_idx) in tqdm(
    enumerate(cv.split(Xtrain, ytrain), start = 1)
) : 
 
    Xy_train = Xtrain.iloc[train_idx][FEATURES + [ target ] ].copy()
    X_valid  = Xtrain.iloc[val_idx][FEATURES].copy()
    y_valid  = Xtrain.iloc[val_idx][ target ]
    X_test   = Xtest[FEATURES].copy()

    if nb_orig <= 0:
        pass
    else:
        Xy_train = pd.concat(
            [Xy_train, extra[FEATURES + [ target ]]], 
            axis = 0, 
            ignore_index = True
        )

    print(f"---> Shapes = {Xy_train.shape} {X_valid.shape} {X_test.shape}")
    
    for i, c in tqdm( enumerate(te_cols), "Target encoding"):
            
        TE          = TargetEncoder(
            n_folds        = 3, 
            smooth         = 0, 
            split_method   = 'random', 
            stat           = 'mean',
        )
  
        Xy_train[c] = TE.fit_transform(Xy_train[c], Xy_train[target]).astype(np.float32)
        X_valid[c]  = TE.transform(X_valid[c]).astype(np.float32)
        X_test[c]   = TE.transform(X_test[c]).astype(np.float32)
        
    model = LGBMR(**params)
    model.fit(
        Xy_train[FEATURES],
        Xy_train[target],
        eval_set    = [(X_valid[FEATURES], y_valid )],
        eval_metric = "rmse",
        callbacks   = [log_evaluation(0), early_stopping(250, verbose = False)],
    )

    dev_preds = pd.DataFrame(
        model.predict(X_valid[FEATURES]),
        index   = val_idx,
        columns = ["Preds"],
        dtype   = np.float32,
    )

    ftreimp += model.feature_importances_
    oof_preds.append(dev_preds)    
    test_preds += model.predict(X_test[FEATURES] ) / FOLDS
    
    print()
    collect();
    torch.cuda.empty_cache()
 
if plot_req :

    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(20, 10))
        (
            pd.DataFrame(
                ftreimp, index = FEATURES, columns = ["Imp"]
            ).
            sort_values("Imp", ascending = False).
            head(25).
            plot.
            barh(ax = ax)
        )
        plt.title("Feature Importances")
        plt.show()  
        
    print()


%%time 

oof_preds = pd.concat(oof_preds, axis = 0, ignore_index = False)
oof_preds = oof_preds.groupby(level = 0).mean().values.flatten()

score     = ScoreMetric(ytrain.values.flatten(), oof_preds)   
print(f"---> OOF Score = {score:,.6f}\n\n")

np.save(f"OOF_Preds_ML{version_lbl}.npy", oof_preds )
np.save(f"Mdl_Preds_ML{version_lbl}.npy", test_preds)

sub_fl = pd.read_csv(
    f"/kaggle/input/playground-series-s5e9/sample_submission.csv",
    index_col = "id",
)

sub1 = pd.read_csv(
    f"/kaggle/input/predicting-the-beats-per-minute-of-songs-s5e9/CatBoostRegressor_prediction.csv"
)[target].values.flatten()
sub2 = pd.read_csv(
    f"/kaggle/input/predicting-the-beats-per-minute-of-songs-s5e9/LGBMRegressor_prediction.csv"
)[target].values.flatten()
sub3 = pd.read_csv(
    f"/kaggle/input/predicting-the-beats-per-minute-of-songs-s5e9/XGBRegressor_prediction.csv"
)[target].values.flatten()

sub_fl[target] = ( test_preds + sub1 + sub2 + sub3 ) / 4.0
sub_fl.to_csv(f"submission.csv")


print()
!head submission.csv

print()
!ls
print()

