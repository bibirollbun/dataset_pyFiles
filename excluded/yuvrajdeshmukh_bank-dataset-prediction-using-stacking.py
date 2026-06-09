# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import xgboost as xgb
import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
train_data.head()


test_data = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
test_data.head()


df = train_data
df.info()


df.describe()


list(train_data.columns)


print(f"Training Data Shape: {train_data.shape}")
print(f"Test Data Shape: {test_data.shape}")


print("Unique Values in Categorical Features (Training Data):")
categorical_cols = train_data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"{col}: {train_data[col].nunique()} unique values")


object_cols = train_data.select_dtypes(include="object").columns

for col_name in object_cols:
    print(f"{col_name} \n Training Data \n {train_data[col_name].value_counts()}")


object_cols = test_data.select_dtypes(include="object").columns

for col_name in object_cols:
    print(f"{col_name} \n Training Data \n {test_data[col_name].value_counts()}")


def no_score(df):
    conditions = [(df['default'] == "no") & (df['housing'] == "no") & (df['loan'] == "no"),
                  (df['default'] == "no") & (df['housing']== "no"),
                  (df['default'] == "no") & (df['loan']== "no"),
                  (df['housing'] == "no") & (df['loan'] == "no"),
                  (df['default'] == "no") | (df['housing'] == "no") | (df['loan'] == "no")] 
    
    choices = [21, 7, 7, 7, 3] 
    
    df['no_score'] = np.select(conditions, choices, default=0)
    return df


def unknown_score(df):
    conditions = [(df['education'] == "unknown") & (df['contact'] == "unknown") & (df['poutcome'] == "unknown"),
                  (df['education'] == "unknown") & (df['contact'] == "unknown"),
                  (df['contact'] == "unknown") & (df['poutcome'] == "unknown"),
                  (df['education'] == "unknown") | (df['contact'] == "unknown") | (df['poutcome'] == "unknown")]
    
    choices = [21, 7, 7, 3] 
    
    df['unknown_score'] = np.select(conditions, choices, default=0)
    return df



no_score(train_data)
print(train_data['no_score'].value_counts())
no_score(test_data)
print(test_data['no_score'].value_counts())

unknown_score(train_data)
print(train_data['unknown_score'].value_counts())
unknown_score(test_data)
print(test_data['unknown_score'].value_counts())


month_map = {'jan': 1,
             'feb': 2,
             'mar': 3,
             'apr': 4,
             'may': 5,
             'jun': 6,
             'jul': 7,
             'aug': 8,
             'sep': 9,
             'oct': 10,
             'nov': 11,
             'dec': 12
        }


train_data['month_num'] = train_data['month'].map(month_map)
test_data['month_num'] = test_data['month'].map(month_map)

train_data['month_num']
test_data['month_num']


def new_feats(df):
    df = df.copy()
    df['balance_posi'] = (df['balance'] > 0).astype(int)
    df['has_previous'] = (df['previous'] > 0).astype(int)
    df['long_duration'] = (df['duration'] >= 365).astype(int)
    df['campaign_multi'] = (df['campaign'] >= 2).astype(int)
    df['is_first_contact'] = (df['campaign'] == 1).astype(int)
    df['high_campaign'] = (df['campaign'] >= 3).astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 97], 
                             labels=['young', 'adult', 'middle', 'senior', 'elderly']).astype("object")
    df['log_duration'] = np.log1p(df['duration'])
    df['sqrt_duration'] = np.sqrt(df['duration'])
    df['log_campaign']=np.log1p(df['campaign'])
    df['sqrt_age'] = df['age'] ** 2
    df['cubed_age'] = df['age'] ** 3
    df['log_age'] = np.log1p(df['age'])
    return df


train_data = new_feats(train_data)
train_data = train_data.drop(columns=['month', 'month_num', 'duration'])
test_data = new_feats(test_data)
test_data = test_data.drop(columns=['month', 'month_num', "duration"])

object_cols = train_data.select_dtypes(include="object").columns


train_data.describe()


test_data.describe()


y = train_data['y']
train_id = train_data['id']
test_id = test_data['id']


def base_model(df):
    df = df.copy()
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].astype(str)
        df[col] = df[col].fillna("None")
        df[col] = LabelEncoder().fit_transform(df[col])
    for col in df.select_dtypes(include=['int64']):
        df[col] = df[col].fillna(df[col].median())
    return df


X_train_drop = base_model(train_data.drop(columns=["id", "y"]))
X_test_drop = base_model(test_data.drop("id", axis=1))


"""xgb_model = XGBClassifier(
    n_estimators=25000,
    learning_rate=0.4,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,
    tree_method='hist',
    reg_lambda=3,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

cat_model = CatBoostClassifier(
    iterations=25000,
    learning_rate=0.4,
    depth=8,
    random_seed=42,
    verbose=0
)

lgbm_model = LGBMClassifier(
                n_estimators=25000,
                learning_rate=0.04,
                max_depth=8,
                num_leaves=100,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1,
                reg_lambda=3,
                max_bin=4523,
                random_state=42,
                verbosity=-1
            )"""


"""stack_model = StackingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('cat', cat_model),
        ('lgbm', lgbm_model)
    ],
    final_estimator=XGBClassifier(),
    cv=5,
    n_jobs=-1,
    passthrough=False
)"""


n_splits=10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train_drop))
test_preds = np.zeros(len(X_test_drop))


for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_drop, y)):
    X_train, X_val = X_train_drop.iloc[train_idx], X_train_drop.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    stack_model = LGBMClassifier(
                n_estimators=20000,
                learning_rate=0.008,
                max_depth=-1,
                num_leaves=128,
                min_data_in_leaf=50,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1,
                reg_lambda=3,
                max_bin=4523,
                min_child_samples=20,
                min_split_gain=0.005,
                lambda_l1=0.5,
                lambda_l2=0.5,
                feature_fraction=0.85,
                bagging_fraction=0.85,
                random_state=42,
                verbosity=-1
            )
    
    stack_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='binary_logloss',
                callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=True)]
            )
    
    oof_preds[val_idx] = stack_model.predict(X_val)
    test_preds += stack_model.predict_proba(X_test_drop)[:, 1] / n_splits


score = accuracy_score(y, oof_preds)
print(f"LGBM CV Accuracy: {score:.4f}")


submission = pd.DataFrame({
    "id": test_data["id"],
    "y": test_preds
})
submission.to_csv("submission.csv", index=False)
print("ðŸš€ submission.csv has been created.")
output = pd.read_csv('/kaggle/working/submission.csv')
output.head(20)




