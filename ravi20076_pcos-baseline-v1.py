


%%writefile -a req.txt

polars==1.20.0
scikit-learn==1.6.0
numpy==1.26.4
scipy==1.14.1
pandas==2.2.3
xgboost==2.1.3


%%time 

!pip install -q uv
!uv pip install -q -r req.txt --system

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from gc import collect
from warnings import filterwarnings
from IPython.display import display_html, clear_output
from colorama import Fore, Style, Back

# ML Model training:-
from sklearn.metrics import *
from xgboost import XGBClassifier as XGBC
from catboost import CatBoostClassifier as CBC
from lightgbm import LGBMClassifier as LGBMC

import matplotlib.pyplot as plt
import seaborn as sns

# Color printing
def PrintColor(text: str, color = Fore.BLUE, style = Style.BRIGHT):
    "Prints color outputs using colorama using a text F-string"
    print(style + color + text + Style.RESET_ALL)

filterwarnings('ignore')


%%time 

target   = 'PCOS'
orig_req = True


%%time 

train    = pd.read_csv(f"/kaggle/input/exploring-predictive-health-factors/train.csv", index_col = ["ID"])
test     = pd.read_csv(f"/kaggle/input/exploring-predictive-health-factors/test.csv", index_col = ["ID"])
sub_fl   = pd.read_csv(f"/kaggle/input/exploring-predictive-health-factors/sample_submission.csv", index_col = ["ID"])
original = pd.read_csv(f"/kaggle/input/diet-exercise-and-pcos-insights/Cleaned-Data.csv")

PrintColor(
    f"---> Shapes = {train.shape} {test.shape} {original.shape}",
    color = Fore.CYAN
)

if orig_req:
    train = \
    (pd.concat([train.assign(Source = "Competition"), 
                original[train.columns].assign(Source = "Original")
               ], 
               axis=0, 
               ignore_index = True
              )
    )
else:
    PrintColor("---> Original data is not needed", color = Fore.RED)

train = train.loc[train[target].isin(["No", "Yes"])]
train[target] = train[target].map({"No" : 0, "Yes" : 1}).astype(np.int8)

PrintColor(
    f"---> Shapes = {train.shape} {test.shape} {original.shape}"
)


%%time 

Xtrain = train.drop([target,"Source"], axis=1, errors = "ignore")
Xtest  = test.copy()
ytrain = train[target]

cat_cols = list(test.select_dtypes(["string", "category", "object"]).columns)

PrintColor(f"---> Category columns")
print(np.array(cat_cols))

Xtrain[cat_cols] = Xtrain[cat_cols].astype("string").fillna("missing")
Xtest[cat_cols]  = Xtest[cat_cols].astype("string").fillna("missing")

PrintColor(f"\n\n---> Shapes = {Xtrain.shape} {ytrain.shape} {Xtest.shape}")


%%time 

model = \
CBC(
    loss_function = "Logloss",
    eval_metric = "AUC",
    iterations = 70,
    random_state = 42,
    learning_rate = 0.025,
    verbose = 0,
    max_depth = 3,
    cat_features = cat_cols,
)

model.fit(Xtrain, ytrain,)
test_preds = model.predict_proba(Xtest)[:,1]

sub_fl[target] = test_preds
sub_fl.to_csv(f"submission.csv", index = True)

print()
sns.histplot(sub_fl[target], bins = 40)
print()

!ls
print()
!head submission.csv
print()


