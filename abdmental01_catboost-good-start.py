%%time

!pip install -qq lifelines

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore
from IPython.display import clear_output

from sklearn.model_selection import *
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
from lightgbm import LGBMRegressor
import lightgbm as lgb
from tqdm import tqdm
import numpy as np

def print_heading(title):
    print("#" * 50)
    print(f" {title} ")
    print("#" * 50)


%%time

SEED = 42
n_splits = 5

!git clone https://github.com/muhammadabdullah0303/AbdML

import sys
sys.path.append('/kaggle/working/repository')

from AbdML.main import AbdBase

train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
train = pd.concat([train, train_extra])
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

train.head()


%%time

print_heading("Train Shape")
print(train.shape)
print_heading("Test Shape")
print(test.shape)


%%time

print_heading("Train Null Values")
print(train.isnull().sum())

print_heading("Test Null Values")
print(test.isnull().sum())


%%time

cat_cols = train.select_dtypes(include='object').columns
print_heading('CAT_COLS')
print(f"{cat_cols}\n")

num_cols = train.select_dtypes(include='float').columns
print_heading('NUM_COLS')
print(f"{num_cols}")


%%time

def update(df):
    
    df[cat_cols] = df[cat_cols].fillna('Missing').astype('category')

    return df

train = update(train)
test = update(test)


%%time

ohe_cols = {'cat_c': cat_cols}

base = AbdBase(train_data=train, test_data=test, target_column='Price',gpu=False,
                 problem_type="regression", metric="rmse", seed=SEED,
                 n_splits=n_splits,early_stop=True,num_classes=0,cat_features=None,ohe_fe=ohe_cols,
                 fold_type='KF')


%%time

print_heading("Training Data")
base.X_train.head()


%%time

params = {'n_estimators': 1186, 'max_depth': 4, 'learning_rate': 0.05366077236403933, 'min_child_samples': 126,
          'eval_metric': 'RMSE'}

CAT_R = base.Train_ML(params, 'CAT', e_stop=50)


%%time

sample["Price"] = CAT_R[1]
sample.to_csv("submission.csv", index=False)
print_heading("Sub shape:")
print(sample.shape)
sample.head()

