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
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


class cfg:
    train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
    test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
    submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


train = cfg.train
test = cfg.test
submission = cfg.submission


cat_cols = train.select_dtypes(include='object').columns.tolist()
num_cols = train.select_dtypes(include='int64').columns.tolist()
id_col = 'id'
tar_col = 'y'


print(train.shape)
print(test.shape)
print(submission.shape)


cat_cols = train.select_dtypes(include='object').columns.tolist()
num_cols = train.select_dtypes(include='int64').columns.tolist()
id_col = 'id'
tar_col = 'y'


train['default'] = train['default'].map({'yes': 1, 'no': 0})
train['housing'] = train['housing'].map({'yes': 1, 'no': 0})
train['loan'] = train['loan'].map({'yes': 1, 'no': 0})


test['default'] = test['default'].map({'yes': 1, 'no': 0})
test['housing'] = test['housing'].map({'yes': 1, 'no': 0})
test['loan'] = test['loan'].map({'yes': 1, 'no': 0})


def fe(df):
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 45, 65, np.inf], labels=['Young', 'Adult', 'Senior', 'Elder'])
    df['balance_level'] = pd.qcut(df['balance'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    df['campaign_level'] = pd.cut(df['campaign'], bins=[0, 2, 5, np.inf], labels=['Low', 'Medium', 'High'])
    #df['pdays_cat'] = df['pdays'].apply(lambda x: 'Never' if x == 999 else ('Recent' if x <= 10 else 'Old'))
    #df['pdays_cat'] = df['pdays_cat'].astype('category')
    df['total_contacts'] = df['campaign'] + df['previous']
    df['has_any_loan'] = ((df['loan'] == 1) | (df['housing'] == 1)).astype(int)
    return df


num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'] 
cat_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome']


def preprocess(df):
    
    for col in num_cols:
        lower_bound = df[col].quantile(0.005)
        upper_bound = df[col].quantile(0.995)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['has_previous'] = (df['previous'] > 0).astype(int)
    df['duration_long'] = (df['duration'] > 300).astype(int)
    df['campaign_multiple'] = (df['campaign'] > 2).astype(int)
    df['sqrt_age'] = np.sqrt(df['age'])
    
    df['duration_log']=np.log(df['duration'])
    df['campaign_log']=np.log(df['campaign'])
    df['pdays_log']=np.log(df['pdays']+2)
    df['previous_log']=np.log(df['previous']+1)
    
    for feature in cat_cols:
        df[feature] = df[feature].astype("category")
    
    return df


train = preprocess(train)
test = preprocess(test)


#train = pd.get_dummies(train, columns=cat_cols, drop_first=True)
#test = pd.get_dummies(test, columns=cat_cols, drop_first=True)


#cols_drop = "duration"


#train = train.drop(cols_drop, axis=1)
#test = test.drop(cols_drop, axis=1)


from sklearn.preprocessing import LabelEncoder

categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col]).astype("int64")


from sklearn.preprocessing import LabelEncoder

categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

for col in categorical_cols:
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col]).astype("int64")


import lightgbm as lgb


X = train.drop(['id','y'], axis=1)
y = train['y']
X_test = test.drop(['id'], axis=1)


from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer
from lightgbm import early_stopping, log_evaluation
import optuna


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'n_estimators': 1000
    }

    auc_scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**param)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[
                early_stopping(50),
                log_evaluation(0)
            ]
        )

        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        auc_scores.append(auc)

    return np.mean(auc_scores)


#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=50)

#print("Best AUC-ROC: {:.4f}".format(study.best_value))
#print("Best hyperparameters:\n", study.best_params)


params = {'lambda_l1': 1.7309507742718422, 
 'lambda_l2': 1.0120095245719341e-05, 
 'num_leaves': 255, 
 'feature_fraction': 0.6087835516948767, 
 'bagging_fraction': 0.6638489787990103, 
 'bagging_freq': 5, 
 'min_child_samples': 18, 
 'learning_rate': 0.0194178090103108}


train_data = lgb.Dataset(X, label=y)

#params = study.best_params

# Train model (no validation set with labels, so no early stopping)
model_lgb = lgb.train(
    params,
    train_data,
    num_boost_round=100)


y_proba_lgb_train = model_lgb.predict(X)


from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# AUC score
auc = roc_auc_score(y, y_proba_lgb_train)
print(f"AUC-ROC: {auc:.4f}")


y_proba_lgb_test = model_lgb.predict(X_test)


import xgboost as xgb
from xgboost import XGBClassifier


model_xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=True,
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


def convert_to_numeric(X):
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category')

    # Convert any category columns to numeric
    for col in X.select_dtypes(include='category').columns:
        X[col] = X[col].cat.codes
    return X


X = convert_to_numeric(X)
X_test = convert_to_numeric(X_test)


dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
dtest = xgb.DMatrix(X_test, enable_categorical=True)


params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.1,
    'max_depth': 4,
    'tree_method': 'hist',  # required for categorical support
    'enable_categorical': True,
    'scale_pos_weight': (y.size - y.sum()) / y.sum()
}


# Train the model
model_xgb = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=200,
    evals=[(dtrain, 'train')],
    early_stopping_rounds=10,
    verbose_eval=10
)


y_proba_xgb_train = model_xgb.predict(dtrain)


from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# AUC score
auc = roc_auc_score(y, y_proba_xgb_train)
print(f"AUC-ROC: {auc:.4f}")


y_proba_xgb_test = model_xgb.predict(dtest)


from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, roc_auc_score


model_cbt = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.01,
    depth=6,
    eval_metric='AUC',
    verbose=500,
    random_state=42,
    early_stopping_rounds=20
)


#.fit(X, y, use_best_model=True)


#y_proba_cbt_train = model_cbt.predict(X)


"""from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# AUC score
auc = roc_auc_score(y, y_proba_cbt_train)
print(f"AUC-ROC: {auc:.4f}")"""


#y_proba_cbt_test = model_cbt.predict(X_test)


y_proba_ensemble = 0.8 * y_proba_lgb_test + 0.2 * y_proba_xgb_test 


submission['y'] = y_proba_ensemble


submission.head()


submission.to_csv("submission.csv", index=False)




