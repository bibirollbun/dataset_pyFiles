


!pip install -q uv
!uv pip install -q scikit-learn==1.6.1 lightgbm==4.6.0 xgboost==2.1.4 --system


import pandas as pd, numpy as np
import os, sys, re, joblib
from warnings import filterwarnings 
from tqdm.notebook import tqdm

from sklearn.metrics import *
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import TransformerMixin, BaseEstimator, RegressorMixin, clone

from sklearn.preprocessing import (
StandardScaler, RobustScaler, MinMaxScaler, FunctionTransformer, OrdinalEncoder, TargetEncoder
)

import torch
from lightgbm import LGBMRegressor as LGBMR, log_evaluation, early_stopping
from xgboost import XGBRegressor as XGBR
from catboost import CatBoostRegressor as CBR, Pool

import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style, Back

filterwarnings("ignore")


target      = "e_users"
version_lbl = "MLV1_2"


def ScoreMetric(ytrue, ypred) :
    return root_mean_squared_error( ytrue, ypred )

def PrintColor(text, color = Fore.BLUE, style = Style.BRIGHT) :
    print(color  + style + text + Style.RESET_ALL)


%%time 

train  = pd.read_csv(f"/kaggle/input/prediction-of-e-commerce-users/train_df.csv", parse_dates = ["datetime"])
test   = pd.read_csv(f"/kaggle/input/prediction-of-e-commerce-users/test_df.csv", parse_dates = ["datetime"])
sub_fl = pd.read_csv(f"/kaggle/input/prediction-of-e-commerce-users/submission.csv", parse_dates = ["datetime"])

display(
    train.head(10)
)

strt_ftre = list(test.columns)
df = pd.concat([train.drop(target, axis=1), test], axis=0)


%%time 

with sns.axes_style("white") :
    df_1 = \
    df.groupby( df["datetime"].dt.date )[["promotion_1","promotion_2", "promotion_3"]].mean()
    
    fig, ax = plt.subplots( 1, 1, figsize = (30, 9))
    df_1.plot(ax = ax, color = ["#005ce6", "#4dff4d", "#992600"])
    ax.axvline( x = pd.to_datetime("2024-09-01"), c = "black", linewidth = 2.5)

    start_quarter = pd.Timestamp(df["datetime"].min()).to_period('Q').to_timestamp(how='end')
    end_quarter   = pd.Timestamp(df["datetime"].max()).to_period('Q').to_timestamp(how='end')
    ax.set_xticks(
        pd.date_range(start=start_quarter, end=end_quarter, freq='Q'), 
        [mydate.strftime('%Y%m') for mydate in 
          pd.date_range(start=start_quarter, end=end_quarter, freq='Q')
        ],
        rotation = 45 
    )
    ax.set(xlabel = "", ylabel = "")
    ax.set_title(f"Time series plot by date-average", fontweight = "bold", color = "maroon")
    plt.show()

    print("\n\n\n")
    df_1 = df.groupby( df["datetime"].dt.hour)[["promotion_1","promotion_2", "promotion_3"]].mean()
    
    fig, axes = \
    plt.subplots(
        3, 1, 
        figsize = (25 , 20), 
        gridspec_kw = {"hspace" : 0.35},
        sharex = True,
    )
    for i, col in enumerate( ["promotion_1","promotion_2", "promotion_3"] ):
        ax = axes[i]
        df_1[col].plot.bar(color = "tab:blue", ax = ax)
        ax.set_title(f"{col} - users by hour",  fontweight = "bold", color = "maroon")
        
    plt.show()
      



df.groupby( df["datetime"].dt.dayofweek )[["promotion_1","promotion_2", "promotion_3"]].mean()


%%time 

def make_ftre(X : pd.DataFrame):
    "This function makes secondary features for the provided dataset"

    df = X.copy()
    
    df["p2versusp1"] = df["promotion_2"] / df["promotion_1"]
    df["p3versusp1"] = df["promotion_3"] / df["promotion_1"]
    df["p3versusp2"] = df["promotion_3"] / df["promotion_2"]
    
    df["p2mulp1"] = df["promotion_2"] * df["promotion_1"]
    df["p3mulp1"] = df["promotion_3"] * df["promotion_1"]
    df["p3mulp2"] = df["promotion_3"] * df["promotion_2"]
    
    cols = list( df.columns[1:] )
    
    for nper in [1,3,6,9,12,18,24,36,48,72,168] :
        for col in cols :
            df[f"l{nper}{col}"]   = df[col].shift(nper)
            df[f"d{nper}{col}"]   = df[col].diff(nper)
            df[f"sma{nper}{col}"] = df[col].rolling(nper).mean()

    df["hour_nb"]  = df["datetime"].dt.hour
    df["sin_hour"] = np.sin(df["hour_nb"])
    df["cos_hour"] = np.cos(df["hour_nb"])
    del df["hour_nb"]
    
    df["day_nb"]   = df["datetime"].dt.day
    df["sin_day"] = np.sin(df["day_nb"])
    df["cos_day"] = np.cos(df["day_nb"])
    del df["day_nb"]
    
    df["dow_nb"]  = df["datetime"].dt.dayofweek
    df["sin_dow"] = np.sin(df["dow_nb"])
    df["cos_dow"] = np.cos(df["dow_nb"])
    del df["dow_nb"]
    
    df["week_nb"]  = df["datetime"].dt.isocalendar().week
    df["sin_week"] = np.sin(df["week_nb"])
    df["cos_week"] = np.cos(df["week_nb"]) 
    del df["week_nb"]
  
    df["month_nb"]  = df["datetime"].dt.month
    df["sin_month"] = np.sin(df["month_nb"])
    df["cos_month"] = np.cos(df["month_nb"])
    del df["month_nb"]

    return df


%%time 

drop_cols = ["datetime", "fold_nb", target]

df    = make_ftre(df)
Xtr   = \
(
    df.loc[df.datetime <= pd.to_datetime("2024-04-30")].
    drop(drop_cols, axis = 1, errors = "ignore")
)

Xdev  = \
(
    df.loc[df.datetime.between(pd.to_datetime("2024-05-01"), 
                               pd.to_datetime("2024-08-31"),
                               inclusive = "both") 
    ].
    drop(drop_cols, axis = 1, errors = "ignore")
)

ytr   = \
train.loc[train.datetime <= pd.to_datetime("2024-04-30"), target]

ydev  = \
train.loc[train.datetime.between(pd.to_datetime("2024-05-01"), pd.to_datetime("2024-08-31"), inclusive = "both"), target]

Xtest = \
(
    df.loc[df.datetime >= pd.to_datetime("2024-09-01")].
    drop(drop_cols, axis = 1, errors = "ignore")
)

X     = pd.concat([Xtr, Xdev], axis = 0, ignore_index = True)
y     = pd.concat([ytr, ydev], axis = 0, ignore_index = True)

PrintColor(
    f"---> Shapes = {Xtr.shape} {ytr.shape} {Xdev.shape} {ydev.shape} {Xtest.shape} {sub_fl.shape}"
)

try:
    del df
except:
    pass


%%time 

Mdl_Master = \
{
    "CB1R" : CBR(iterations             = 10_000,
                 loss_function          = "RMSE",
                 eval_metric            = "RMSE",
                 max_depth              = 3,
                 l2_leaf_reg            = 0.30,
                 random_state           = 42,
                 learning_rate          = 0.025,
                 leaf_estimation_method = "Newton", 
                 task_type              = "GPU" if torch.cuda.is_available() else "CPU",
                 colsample_bylevel      = 0.35 if torch.cuda.is_available() == False else None,   
                 early_stopping_rounds  = 50,
                 verbose                = 0,
                ),
    
    "LGBM1R" : LGBMR(n_estimators           = 10_000,
                     objective              = "regression_l2",
                     metric                 = "rmse",
                     max_depth              = 3,
                     colsample_bytree       = 0.30,
                     random_state           = 42,
                     learning_rate          = 0.025, 
                     device                 = "cpu", 
                     verbosity              = -1,
                    ),

    "XGB1R" :  XGBR( n_estimators           = 10_000,
                     objective              = "reg:squarederror",
                     metric                 = "rmse",
                     max_depth              = 3,
                     colsample_bytree       = 0.30,
                     colsample_bynode       = 0.40,
                     random_state           = 42,
                     learning_rate          = 0.025, 
                     device                 = "cpu", 
                     verbosity              = 0,
                     early_stopping_rounds  = 50,
                    ),    
}

Mdl_Preds = {}



%%time 

for method, mymodel in Mdl_Master.items() :
    PrintColor(
        f"\n\n ========== {method} MODEL OFFLINE TRAINING ========== \n", 
        color = Fore.RED
    )

    model = clone(mymodel)

    if "CB" in method:
        model.fit(
            Xtr, ytr, 
            eval_set = [(Xdev.drop(drop_cols, axis=1, errors = "ignore"), ydev)], 
            verbose  = 50
        )
        best_iter = model.get_best_iteration()

    elif "LGB" in method :
        model.fit(
            Xtr, ytr, 
            eval_set = [(Xdev, ydev)], 
            eval_names = [("Dev")] ,
            callbacks = [log_evaluation(50), early_stopping(50, verbose = False)],
        )
        best_iter = model.best_iteration_
        
    elif "XGB" in method :
        model.fit(
            Xtr, ytr, 
            eval_set = [(Xdev, ydev)], 
            verbose  = 50
        )
        best_iter = model.best_iteration
        
    print(f"---> Best iteration = {best_iter :,.0f}\n\n")

    ftreimp = pd.Series( model.feature_importances_, index = Xtest.columns, name = "FtreImp" )
    with sns.axes_style("white"):
        fig, ax = plt.subplots(1,1, figsize = (20, 6))
        ftreimp.sort_values(ascending = False).head(50).plot.bar(ax = ax, color = "tab:blue")
        ax.set_title(f"{method} feature importances", fontweight = "bold", color = "maroon")
        plt.show()
    del ftreimp

    PrintColor(f"\n---> Refitting fully for test set predictions")
    model = clone(mymodel)
    if "CB" in method :
        model.iterations = int( 1.05 * best_iter )
        print(f"---> Iterations = {model.iterations} ")
    else:
        model.n_estimators = int( 1.05 * best_iter )
        print(f"---> Iterations = {model.n_estimators} ")

    try:
        model.early_stopping_rounds = None
    except: 
        pass
    
    model.fit(X,y,)
    Mdl_Preds[method] = model.predict(Xtest)
    del best_iter



%%time 

sub_fl[target] = \
np.round(
    np.average(
        pd.DataFrame( Mdl_Preds ).values, 
        axis    = 1, 
        weights = [0.15, 0.60, 0.35]
    ), 0
)

sub_fl.to_csv(f"submission.csv", index = None)


fig, ax = plt.subplots(1,1, figsize = (30, 6))
preds = \
pd.concat( 
    [ train[["datetime", target]] , sub_fl[["datetime", target]] ], axis = 0
)

sns.lineplot(
    data = preds, 
    x = "datetime", 
    y = target, 
    color = "tab:blue", 
    linewidth = 1.5, 
    ax = ax
)

start_quarter = pd.Timestamp(train["datetime"].min()).to_period('Q').to_timestamp(how='end')
end_quarter   = pd.Timestamp(sub_fl["datetime"].max()).to_period('Q').to_timestamp(how='end')
ax.set_xticks(
    pd.date_range(start=start_quarter, end=end_quarter, freq='Q'), 
    [mydate.strftime('%Y%m') for mydate in 
      pd.date_range(start=start_quarter, end=end_quarter, freq='Q')
    ],
    rotation = 45 
)

ax.set(xlabel = "", ylabel = "")
ax.axvline(x = pd.to_datetime(f"2024-09-01"), color = "red", linewidth = 3.0)
ax.set_title(f"Target plot - train + predictions", fontweight = "bold", color = "maroon", fontsize = 16)
plt.show()

print("\n\n\n")
!head submission.csv


