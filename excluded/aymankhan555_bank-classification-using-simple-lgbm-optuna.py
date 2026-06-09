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


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier , VotingClassifier
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV , train_test_split ,RandomizedSearchCV

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error,roc_auc_score ,f1_score
from sklearn.preprocessing import OneHotEncoder ,LabelEncoder ,OrdinalEncoder

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv") 
sample = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


print(df.columns)
print(df.isnull().sum())
df.head()


sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()



num_cols = df.select_dtypes(exclude=['object','category']).drop(['id','y'],axis=1).columns.tolist()
cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
print(f'num cols : {num_cols}')
print(f'cat cols : {cat_cols}')


import math
rows = math.ceil(len(num_cols) / 3)

plt.figure(figsize=(15, rows*4 ))
for i in range(len(num_cols)):
    plt.subplot(rows, 3, i+1) 
    sns.histplot(df[num_cols[i]], kde=True)
    plt.title(num_cols[i])

plt.tight_layout()
plt.show()



def NEW_FE(df):
    
    df['balance_log'] = np.log1p(df['balance'].clip(lower=0))
    df['job_edu'] = df['job'].astype(str) + "_" + df['education'].astype(str)
   

    df['duration_sin'] = np.sin(2*np.pi * df['duration'] / 400)
    df['duration_cos'] = np.cos(2*np.pi * df['duration'] / 400)
    df['loan'] = df['loan'].map({'yes':1,'no':0})
    df['housing'] = df['housing'].map({'yes':1,'no':0})
    # df['has_loan'] = df['housing']+ df['loan']



    return df

df = NEW_FE(df)
test_data = NEW_FE(test_data)


num_cols = df.select_dtypes(exclude=['object','category']).drop(['id','y'],axis=1).columns.tolist()
cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
print(f'num cols : {num_cols}')
print(f'cat cols : {cat_cols}')


X = df[num_cols+ cat_cols]
y = df.y
# le=LabelEncoder()
oe  = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.2,random_state=42)

X_train[cat_cols] = oe.fit_transform(X_train[cat_cols])
X_valid[cat_cols] = oe.transform(X_valid[cat_cols])


# import optuna
# from lightgbm import LGBMClassifier
# from lightgbm import early_stopping

# from sklearn.metrics import roc_auc_score

# def objective(trial):
#     param= {
#         'device' :'gpu',
#         'boosting_type':'gbdt',
#         'learning_rate' : trial.suggest_float('learning_rate',0.01,0.1),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 150),
#         'max_depth': trial.suggest_int('max_depth', 3, 15),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
#         'n_estimators': trial.suggest_int('n_estimators',100,1000),  # int here!
#         'random_state': 42,
#         'verbosity': -1
#     }
#     model = LGBMClassifier(**param)
#     model.fit(
#     X_train, y_train,
#     eval_set=[(X_valid, y_valid)],
#     eval_metric='auc',
#     callbacks=[early_stopping(stopping_rounds=50)]
# )
#     y_pred_proba = model.predict_proba(X_valid)[:,1]  # missing in your code
#     return roc_auc_score(y_valid, y_pred_proba)

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=40)

# print("Best LightGBM params:", study.best_params)


#Paramets selected using optuna 
lgb_model = LGBMClassifier(
    learning_rate=0.07459282861701985,
    num_leaves=139,
    max_depth=14,
    min_child_samples=93,
    subsample=0.7337142954327327,
    colsample_bytree=0.781462628945671,
    reg_alpha=2.603318780335231,
    reg_lambda=4.254119374067986,
    n_estimators=778,
    random_state=42,
    verbosity=-1,
    #device='gpu'
)

lgb_model.fit(X_train,y_train)
pred = lgb_model.predict(X_valid)
proba = lgb_model.predict_proba(X_valid)[:,1]


lgb_f1 = f1_score(y_valid,pred)
lgb_auc = roc_auc_score(y_valid,proba)


print("ğŸ”� ROC AUC:", lgb_auc)
print("ğŸ�¯ F1 Score:", lgb_f1)


full_data_X = df[num_cols + cat_cols]
full_data_y = df.y
full_data_X[cat_cols] = oe.fit_transform(full_data_X[cat_cols])
lgb_model.fit(full_data_X,full_data_y)



test_data = test_data[num_cols + cat_cols]

test_data[cat_cols] = test_data[cat_cols].astype(str)
test_data[cat_cols] = oe.transform(test_data[cat_cols])

test_pred = lgb_model.predict(test_data)
test_proba = lgb_model.predict_proba(test_data)[:,1]
sample['y']=test_proba

sample.to_csv('submission.csv',index=False)

