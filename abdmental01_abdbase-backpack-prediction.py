%%time

!pip install -qq lifelines

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore
from IPython.display import clear_output
import seaborn as sns

from sklearn.model_selection import *
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
from lightgbm import LGBMRegressor
import lightgbm as lgb
from tqdm import tqdm

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

train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col='id')
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
train = pd.concat([train, train_extra], axis=0, ignore_index=True)

test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col='id')
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

    for col in cat_cols:
        df[col] = df[col].fillna('Missing').astype('category')

    return df

train = update(train)
test = update(test)


%%time

ohe_cols = {'cat_c': cat_cols}

base = AbdBase(train_data=train, test_data=test, target_column='Price',gpu=False,handle_date=False,
                 problem_type="regression", metric="rmse", seed=SEED,
                 n_splits=n_splits,early_stop=True,num_classes=0,cat_features=None,ohe_fe=ohe_cols,
                 fold_type='KF')


%%time

print_heading("Training Data")
base.X_train.head()


%%time

params =  {'n_estimators': 693, 'max_depth': 8, 'colsample_bytree': 0.5075413041956253,
           'subsample': 0.83306258044554, 'learning_rate': 0.011639115298956882, 'min_child_samples': 23} # CV : 38.9979

rLGBM = base.Train_ML(params,'LGBM')


%%time

feature_importance = rLGBM[2].feature_importances_
feature_names = base.X_train.columns 

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(18, 18))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('LightGBM Feature Importance (Gain)')
plt.show()


%%time

sample["Price"] = rLGBM[1]
sample.to_csv("submission.csv", index=False)
print_heading("Sub shape:")
print(sample.shape)
sample.head()

