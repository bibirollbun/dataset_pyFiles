


!pip install -q scikit-learn==1.6.1 xgboost==2.1.4 


import pandas as pd, numpy as np
from gc import collect
from tqdm.notebook import tqdm
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier as XGBC
from lightgbm import LGBMClassifier as LGBMC
from catboost import CatBoostClassifier as CBC
from sklearn.ensemble import RandomForestClassifier as RFC, HistGradientBoostingClassifier as HGBC
from sklearn.linear_model import LogisticRegression as LRC

import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
from colorama import Fore, Style, Back

filterwarnings("ignore")



def PrintColor(text: str, color = Fore.BLUE, style = Style.BRIGHT) :
    print(color + style + text + Style.RESET_ALL)


target      = "rainfall"

# Defining the rows in the OOF data and the public LB representation
valid_rows  = 730
public_rows = 146

# Time lagged feature periods
npers   = [1,5,10,]
max_lag = max(npers)


train    = pd.read_csv(f"/kaggle/input/playground-series-s5e3/train.csv", index_col = "id")
test     = pd.read_csv(f"/kaggle/input/playground-series-s5e3/test.csv", index_col = "id")
original = pd.read_csv(f"/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
sub_fl   = pd.read_csv(f"/kaggle/input/playground-series-s5e3/sample_submission.csv", index_col = "id")

original.columns = original.columns.str.replace("\s+", "", regex = True)
original         = original[train.columns]
original[target] = np.where(original[target] == "Yes", 1, 0)

test["winddirection"] = test["winddirection"].ffill()

PrintColor(f"---> Shapes = {train.shape} {original.shape} {test.shape}")

print()
display(train.head(10))


def make_ftre(df : pd.DataFrame, npers) :
    "This function makes the secondary features for the provided dataframe"

    full_df = df.copy()
    PrintColor(f"---> Shape before FE = {full_df.shape}")

    strt_ftre = list(full_df.columns)[1:]

    full_df["range_temp"] = full_df["maxtemp"]     - full_df["mintemp"]
    full_df["dtemp1"]     = full_df["maxtemp"]     - full_df["temparature"]
    full_df["dtemp2"]     = full_df["temparature"] - full_df["mintemp"]

    full_df["actual_day_nb"] = (np.array(full_df.index) + 1)
    full_df["day_of_year"]   = np.where(full_df["actual_day_nb"] % 365 == 0, 365, full_df["actual_day_nb"] % 365 )
    
    full_df['month_nb'] = \
    pd.to_datetime(
        full_df['day_of_year'] - 1, 
        unit='D', 
        origin = pd.Timestamp('2023-01-01')
    ).dt.month

    full_df['day_sin']   = np.sin(2 * np.pi * full_df['day_of_year']/365)
    full_df['day_cos']   = np.cos(2 * np.pi * full_df['day_of_year']/365)
    full_df['month_sin'] = np.sin(2 * np.pi * full_df['month_nb']/ 12)
    full_df['month_sin'] = np.cos(2 * np.pi * full_df['month_nb'] / 12)

    for nper in npers :
        dfs = \
        [full_df[strt_ftre].diff(nper).add_prefix(f"d{nper}_"),
         full_df[strt_ftre].shift(nper).add_prefix(f"l{nper}_"),
        ]
        
        full_df = pd.concat([full_df] + dfs, axis = 1)
        del dfs

    del full_df["actual_day_nb"]
    PrintColor(f"---> Shape after FE = {full_df.shape}")

    return full_df

    


Xtrain = train.drop(target, axis=1, errors = "ignore")
Xtest  = test.copy()
ytrain = train[target].astype(np.uint8)

full_df = pd.concat([Xtrain, Xtest], axis = 0, ignore_index = True)
print(f"---> Shape = {full_df.shape}")

print()
display(
    full_df.
    sample(10).
    style.
    format(precision = 2).
    set_caption(f"Full dataset sample")
)

print("\n\n")
full_df = make_ftre(full_df, npers)
del Xtest

Xtr   = full_df.iloc[0 : len(Xtrain) - valid_rows]
ytr   = ytrain.iloc[Xtr.index]

Xdev  = full_df.iloc[len(Xtrain) - valid_rows : len(Xtrain)]
ydev  = ytrain.iloc[len(Xtrain) - valid_rows : len(Xtrain)]
Xt    = full_df.iloc[len(Xtrain) :]

PrintColor(f"---> Shapes - Train = {Xtr.shape} {ytr.shape} | Dev = {Xdev.shape} {ydev.shape} | Test = {Xt.shape}")


def ScoreMetric(ytrue, ypreds) :
    return roc_auc_score(ytrue, ypreds)

def PlotFtreImp(ftreimp : pd.Series, step = "Step1") :
    with sns.axes_style("white") :
        fig, ax = plt.subplots(1,1, figsize = (20, 5))
        (ftreimp.
         sort_values(ascending = False).
         head(50).
         plot.
         bar(color = "tab:blue", ax = ax)
        )
        ax.set_title(
            f"{step} Feature importance - top 50", 
            color = "maroon", 
            fontweight = "bold", 
            fontsize = 11
        )

        plt.show()
        print()


# Dictionary to store the test set predictions
Mdl_Preds = {}


%%time 

method = "CB1C" 
model  = CBC(iterations        = 200,
             learning_rate     = 0.015,
             l2_leaf_reg       = 0.20,
             colsample_bylevel = 0.30,
             max_depth         = 5,
             random_state      = 42,
             verbose           = 0,
             loss_function     = "Logloss",
             eval_metric       = "AUC",
            )

ftreimp = 0

PrintColor(f"\n============= {method} MODEL TRAINING =============\n", color = Fore.RED)
PrintColor(f"---> STEP1 TRAINING")

new_Xtr = Xtr.iloc[max_lag:]
new_ytr = ytr.iloc[max_lag:]
print(f"---> Shape = {new_Xtr.shape} {new_ytr.shape} | Training data")

model.fit(new_Xtr, new_ytr)

try:
    ftreimp = pd.Series(model.feature_importances_, index = new_Xtr.columns)
    PlotFtreImp(ftreimp, step = "Step1")
except:
    ftreimp = pd.Series(np.abs(model["M"].coef_.flatten()), index = new_Xtr.columns)
    PlotFtreImp(ftreimp, step = "Step1")

Xd  = Xdev.iloc[max_lag : public_rows]
yd  = ydev.iloc[max_lag : public_rows]
score1 = ScoreMetric(yd, model.predict_proba(Xd)[:,1])

Xd  = Xdev.iloc[max_lag + public_rows : ]
yd  = ydev.iloc[max_lag + public_rows : ]
score2 = ScoreMetric(yd, model.predict_proba(Xd)[:,1])

Xd  = Xdev.iloc[max_lag : ]
yd  = ydev.iloc[max_lag : ]
score3 = ScoreMetric(yd, model.predict_proba(Xd)[:,1])
PrintColor(
    f"---> Public = {score1:,.6f} | Private = {score2:,.6f} | Full OOF = {score3:,.6f} \n",
    color = Fore.CYAN
)

PrintColor(f"---> STEP2 TRAINING")

new_Xtr = pd.concat([ Xtr.iloc[max_lag:], Xdev.iloc[0 : public_rows]], axis=0, ignore_index = True)
new_ytr = pd.concat([ ytr.iloc[max_lag:], ydev.iloc[0 : public_rows]], axis=0, ignore_index = True)
print(f"---> Shape = {new_Xtr.shape} {new_ytr.shape} | Training data")

model.fit(new_Xtr, new_ytr)

try:
    ftreimp = pd.Series(model.feature_importances_, index = new_Xtr.columns)
    PlotFtreImp(ftreimp, step = "Step2")
except:
    ftreimp = pd.Series(np.abs(model["M"].coef_.flatten()), index = new_Xtr.columns)
    PlotFtreImp(ftreimp, step = "Step2")

Xd  = Xdev.iloc[max_lag + public_rows : ]
yd  = ydev.iloc[max_lag + public_rows : ]
score = ScoreMetric(yd, model.predict_proba(Xd)[:,1])
PrintColor(f"---> Private = {score:,.6f}\n", color = Fore.CYAN)

PrintColor(f"---> STEP3 TRAINING - FULL REFIT")
new_Xtr = pd.concat([ Xtr, Xdev ], axis=0, ignore_index = True).iloc[max_lag : ]
new_ytr = pd.concat([ ytr, ydev ], axis=0, ignore_index = True).iloc[max_lag : ]
print(f"---> Shape = {new_Xtr.shape} {new_ytr.shape} | Training data")

model.fit(new_Xtr, new_ytr)
print(f"---> Fitted online model\n")
Mdl_Preds[method] = model.predict_proba(Xt)[:,1]


%%time 

method = "LR1C" 
model  = Pipeline([("SS", StandardScaler()), ("M", LRC(random_state = 42, max_iter = 10_000, ))])

PrintColor(f"\n============= {method} MODEL TRAINING =============\n", color = Fore.RED)
PrintColor(f"---> STEP1 TRAINING")

new_Xtr = Xtr.iloc[max_lag:]
new_ytr = ytr.iloc[max_lag:]
print(f"---> Shape = {new_Xtr.shape} {new_ytr.shape} | Training data")

model.fit(new_Xtr, new_ytr)
try:
    ftreimp = pd.Series(model.feature_importances_, index = new_Xtr.columns)
    PlotFtreImp(ftreimp, step = "Step1")
except:
    ftreimp = pd.Series(np.abs(model["M"].coef_.flatten()), index = new_Xtr.columns)
    PlotFtreImp(ftreimp, step = "Step1")

Xd  = Xdev.iloc[max_lag : public_rows]
yd  = ydev.iloc[max_lag : public_rows]
score1 = ScoreMetric(yd, model.predict_proba(Xd)[:,1])

Xd  = Xdev.iloc[max_lag + public_rows : ]
yd  = ydev.iloc[max_lag + public_rows : ]
score2 = ScoreMetric(yd, model.predict_proba(Xd)[:,1])

Xd  = Xdev.iloc[max_lag : ]
yd  = ydev.iloc[max_lag : ]
score3 = ScoreMetric(yd, model.predict_proba(Xd)[:,1])
PrintColor(
    f"---> Public = {score1:,.6f} | Private = {score2:,.6f} | Full OOF = {score3:,.6f} \n",
    color = Fore.CYAN
)

PrintColor(f"---> STEP2 TRAINING")

new_Xtr = pd.concat([ Xtr.iloc[max_lag:], Xdev.iloc[0 : public_rows]], axis=0, ignore_index = True)
new_ytr = pd.concat([ ytr.iloc[max_lag:], ydev.iloc[0 : public_rows]], axis=0, ignore_index = True)
print(f"---> Shape = {new_Xtr.shape} {new_ytr.shape} | Training data")

model.fit(new_Xtr, new_ytr)
try:
    ftreimp = pd.Series(model.feature_importances_, index = new_Xtr.columns)
    PlotFtreImp(ftreimp, step = "Step2")
except:
    ftreimp = pd.Series(np.abs(model["M"].coef_.flatten()), index = new_Xtr.columns)
    PlotFtreImp(ftreimp, step = "Step2")

Xd  = Xdev.iloc[max_lag + public_rows : ]
yd  = ydev.iloc[max_lag + public_rows : ]
score = ScoreMetric(yd, model.predict_proba(Xd)[:,1])
PrintColor(f"---> Private = {score:,.6f}\n", color = Fore.CYAN)

PrintColor(f"---> STEP3 TRAINING - FULL REFIT")
new_Xtr = pd.concat([ Xtr, Xdev ], axis=0, ignore_index = True).iloc[max_lag : ]
new_ytr = pd.concat([ ytr, ydev ], axis=0, ignore_index = True).iloc[max_lag : ]
print(f"---> Shape = {new_Xtr.shape} {new_ytr.shape} | Training data")

model.fit(new_Xtr, new_ytr)
print(f"---> Fitted online model\n")
Mdl_Preds[method] = model.predict_proba(Xt)[:,1]


sub_fl[target] = Mdl_Preds["CB1C"].flatten()
sub_fl.to_csv("submission.csv")
!ls
print()
!head submission.csv

print()
fig, ax = plt.subplots(1,1, figsize = (7, 4))
sub_fl[target].plot.hist(bins = 100, color = "tab:blue")
ax.set_title(f"Prediction histogram", color = "maroon", fontsize = 10, fontweight = "bold")
plt.show()

