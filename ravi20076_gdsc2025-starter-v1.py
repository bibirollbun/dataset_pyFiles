


!pip install -q xgboost==3.0.2 lightgbm==4.6.0 scikit-learn==1.7.1 


%%time 

import pandas as pd, numpy as np, polars as pl
from gc import collect
from tqdm.notebook import tqdm

from xgboost import XGBRegressor as XGBR
from lightgbm import LGBMRegressor as LGBMR, log_evaluation, early_stopping
from catboost import CatBoostRegressor as CBR, Pool

from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.base import clone
from sklearn.preprocessing import *
from sklearn.pipeline import make_pipeline, Pipeline

from warnings import filterwarnings 
filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt


target = "CORRUCYSTIC_DENSITY"


%%time 

train  = pd.read_csv(f"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv", index_col = "LOCAL_IDENTIFIER")
test   = pd.read_csv(f"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv", index_col = "LOCAL_IDENTIFIER")
sub_fl = pd.read_csv(f"/kaggle/input/recruitment-task-for-gdsc-ml/SPECIMEN.csv", index_col = "LOCAL_IDENTIFIER")

print(f"\n\n---> Shapes = {train.shape}, {test.shape}")
strt_ftre = test.columns.tolist()

display(
    pd.concat(
        [
            train.describe().transpose(),
            train.nunique().to_frame().rename(columns = {0 : "n_unique"}),
            train.isna().sum().to_frame().rename(columns = {0 : "null_count"})
        ], axis=1
    ).
    style.
    set_caption(
        f"Basic description and analysis - train data"
    )
)

print("\n\n\n\n")
display(
    pd.concat(
        [
            test.describe().transpose(),
            test.nunique().to_frame().rename(columns = {0 : "n_unique"}),
            test.isna().sum().to_frame().rename(columns = {0 : "null_count"})
        ], axis=1
    ).
    style.
    set_caption(
        f"Basic description and analysis - test data"
    )
)

print("\n\n\n")

fig, ax = plt.subplots(1,1, figsize = (6, 4))
train[target].plot.kde(ax = ax)
ax.set_title(f"Target plot", fontweight = "bold", color = "maroon")
ax.set(xlabel = "", ylabel = "")
plt.tight_layout()
plt.show()


%%time 

train = train.dropna(subset = [target])

Xtrain = train.drop(columns = target)
ytrain = train[target]
Xtest  = test.copy()

proxy_cols = [f"C{i}" for i in range(len(strt_ftre))]
Xtrain.columns = proxy_cols
Xtest.columns  = proxy_cols

cat_cols = list(Xtrain.select_dtypes(exclude = np.number).columns)

Xtrain[cat_cols] = Xtrain[cat_cols].astype("string").fillna("missing").astype("category")
Xtest[cat_cols]  = Xtest[cat_cols].astype("string").fillna("missing").astype("category")


%%time 

cv = KFold(5, shuffle = True, random_state = 42)

Mdl_Master = {
    "XGB1R" : XGBR(
                    n_estimators  = 600,
                    learning_rate = 0.03,
                    max_depth     = 5,
                    random_state  = 42,
                    colsample_bytree = 0.60,
                    reg_alpha        = 0.01,
                    reg_lambda       = 0.001,
                    enable_categorical = True,
                    verbosity          = 0,
                  ),

    "LGBM1R" : LGBMR(
                        n_estimators     = 600,
                        learning_rate    = 0.025,
                        max_depth        = 5,
                        random_state     = 42,
                        subsample        = 0.60,
                        reg_alpha        = 0.01,
                        reg_lambda       = 0.001,                     
                        verbosity        = -1,
                    ),

    "CB1R" : CBR(
                    iterations       = 500,
                    learning_rate    = 0.025,
                    max_depth        = 5,
                    l2_leaf_reg      = 0.65,
                    loss_function    = "RMSE",
                    colsample_bylevel = 0.55,
                    verbose           = 0,
                    random_state      = 42,
                ),
    
}


%%time 

OOF_Preds, Mdl_Preds = [], []

for fold_nb, (train_idx, dev_idx) in tqdm(enumerate( cv.split(Xtrain, ytrain) ) ):

    print(f"---> Starting Fold {fold_nb + 1}")

    Xtr, ytr   = Xtrain.iloc[train_idx], ytrain.iloc[train_idx]
    Xdev, ydev = Xtrain.iloc[dev_idx],   ytrain.iloc[dev_idx]
    Xt         = Xtest.copy()

    oof_preds, test_preds = [], []
    
    for method, mymodel in tqdm( Mdl_Master.items() ):

        model = make_pipeline(*[TargetEncoder(random_state = 42), mymodel])
        model.fit(Xtr, ytr)
        dev_preds = pd.DataFrame( model.predict(Xdev), index = Xdev.index, columns = ["Preds"])
        mdl_preds = pd.DataFrame( model.predict(Xt), index = Xtest.index, columns = ["Preds"])

        oof_preds.append(dev_preds)
        test_preds.append(mdl_preds)

    oof_preds = pd.concat(oof_preds, axis= 0).groupby(level = 0).mean()
    test_preds = pd.concat(test_preds, axis= 0).groupby(level = 0).mean()
    OOF_Preds.append(oof_preds)
    Mdl_Preds.append(test_preds)
    

OOF_Preds = pd.concat(OOF_Preds, axis= 0).sort_index(ascending = True)
Mdl_Preds = (
    pd.concat(Mdl_Preds, axis= 0).
    sort_index(ascending = True).
    groupby(level = 0).
    mean()
)

score = root_mean_squared_error(ytrain, OOF_Preds.values.flatten())
print(f"\n---> Combined Score = {score:,.8f}\n\n")        


%%time 

sub_fl[target] = Mdl_Preds["Preds"].values
sub_fl.to_csv(f"submission.csv", index = True) 
!ls
print()
!head submission.csv

