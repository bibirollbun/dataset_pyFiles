# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import optuna

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,RandomizedSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier,StackingClassifier,VotingClassifier
from lightgbm import LGBMClassifier

import joblib

import warnings 
warnings.filterwarnings('ignore')


data_tr = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
data_tr.head(3)


data_tr.shape


data_te = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
data_te.head(3)


data_te.shape


data = pd.concat([data_tr,data_te],axis=0)
data


sns.heatmap(data.isnull())


data.drop(['poutcome','contact'],axis =1 ,inplace = True)


data.isnull().sum()


data.info()


data.describe().T


categorical_cols = data.select_dtypes(include=['object']).columns
categorical_cols


for i in categorical_cols :
    print(i)
    print(data[i].unique())
    print('\n')


sns.boxplot(x = 'balance',data=data_tr)


encoder = OrdinalEncoder()

data[categorical_cols] = encoder.fit_transform(data[categorical_cols])
#data.drop(categorical_cols,inplace=True,axis=1)
data


# encoder = OneHotEncoder(drop = 'first')

# data[encoder.get_feature_names_out()] = pd.DataFrame(encoder.fit_transform(data[categorical_cols]).toarray(),columns = encoder.get_feature_names_out())
# data.drop(categorical_cols,axis=1,inplace=True)


data.head()


data_te=data[data['y'].isnull()]
data_tr=data[~data['y'].isnull()]


data_tr.head(3)


data_tr.shape


data_te.drop('y',axis = 1,inplace = True)
data_te.head(3)


data_te.shape


X = data_tr.drop('y',axis = 1)
y = data_tr['y']

X.shape , y.shape


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


def objective_xgb(trial):
    params_xgb = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 5),
        "n_estimators": 1000,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_jobs": -1,
        "random_state": 42
    }

    model_xgb = XGBClassifier(**params_xgb)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model_xgb, X, y, scoring='roc_auc', cv=skf, n_jobs=-1)

    return scores.mean()


study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=5)

print(study_xgb.best_params)


# print("Best AUC:", study_xgb.best_value)
# print("Best Parameters:")
# print(study_xgb.best_params)


# optuna.visualization.plot_optimization_history(study_xgb)


# LGBMClassifier

# def objective_lgbm(trial):
#     params_lgbm = {
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
#         "max_depth": trial.suggest_int("max_depth", 5, 15),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),  
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
#         "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 5),
#         "n_estimators": 900,
#         "objective": "binary",
#         "boosting_type": "gbdt",
#         "n_jobs": -1,
#         "random_state": 42
#     }

#     model_lgb = LGBMClassifier(**params_lgbm)

#     skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#     scores = cross_val_score(model_lgb, X, y, scoring='roc_auc', cv=skf, n_jobs=-1)

#     return scores.mean()


# optuna.logging.set_verbosity(optuna.logging.INFO)

# study_lgbm = optuna.create_study(direction="maximize")
# study_lgbm.optimize(objective_lgbm, n_trials=1,show_progress_bar=True)

# print(study_lgbm.best_params)


# print("Best AUC:", study_xgb.best_value)
# print("Best Parameters:")
# print(study_xgb.best_params)


optuna.visualization.plot_optimization_history(study_xgb)


optuna.visualization.plot_parallel_coordinate(study_xgb)


optuna.visualization.plot_slice(study_xgb,params=['learning_rate','max_depth','min_child_weight','subsample'])


optuna.visualization.plot_param_importances(study_xgb)


# #CatBoostClassifier

# def objective_cat(trial):
#     params = {
#         "iterations": 300,  
#         "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2),
#         "depth": trial.suggest_int("depth", 4, 10),
#         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 5),
#         "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
#         "random_strength": trial.suggest_float("random_strength", 1, 10),
#         "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 5),
#         "loss_function": "Logloss",
#         "eval_metric": "AUC",
#         "task_type": "CPU",
#         "random_seed": 42,
#         "verbose": 0
#     }

#     model = CatBoostClassifier(**params)

#     skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#     auc_scores = []

#     for train_idx, val_idx in skf.split(X, y):
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#         model.fit(X_train, y_train,
#                   eval_set=(X_val, y_val),
#                   early_stopping_rounds=15,
#                   verbose=0)

#         preds = model.predict_proba(X_val)[:, 1]
#         auc = roc_auc_score(y_val, preds)
#         auc_scores.append(auc)

#     return np.mean(auc_scores)


# study_cat = optuna.create_study(direction="maximize")
# study_cat.optimize(objective_cat, n_trials=30,show_progress_bar=True)
# optuna.logging.set_verbosity(optuna.logging.INFO)

# print(study_cat.best_params)


# print("Best AUC:", study_cat.best_value)
# print("Best Parameters:")
# print(study_cat.best_params)


# optuna.visualization.plot_optimization_history(study_cat)


# --- Best parameters for XGBoost ---

# Best AUC: 0.9658946627074059
# Best Parameters:
# {'learning_rate': 0.02397985537156955, 'max_depth': 15, 'min_child_weight': 8, 'gamma': 0.6280818324265819, 
# 'subsample': 0.9293528161387496, 'colsample_bytree': 0.5021162296258401, 'reg_lambda': 9.97647162591141, 
# 'reg_alpha': 0.5337726666739991, 'scale_pos_weight': 1}

best_params_xgb = {
    'tree_method': 'hist',
    'grow_policy': 'lossguide',
    'learning_rate': 0.02397985537156955,
    'max_depth': 15,
    'min_child_weight': 8,
    'gamma': 0.6280818324265819,
    'subsample': 0.9293528161387496,
    'colsample_bytree': 0.5021162296258401,
    'reg_lambda': 9.97647162591141,
    'reg_alpha': 0.5337726666739991,
    'scale_pos_weight': 7.3,
    'n_estimators': 1000,
    'n_jobs': -1,
    'use_label_encoder': False,
    'random_state': 42
}

# --- Best parameters for LightGBM ---

# Best AUC: 0.9660675435996612
# Best Parameters:
# {'learning_rate': 0.11235175929137245, 'max_depth': 14, 'min_child_weight': 7, 'subsample': 0.5941717562234573, 
# 'colsample_bytree': 0.6379404167499119, 'reg_lambda': 8.227997337974575, 'reg_alpha': 0.6782738955642913, 
# 'scale_pos_weight': 1}

best_params_lgb = {
    'boosting_type': 'gbdt', 
    'objective': 'binary',
    'learning_rate': 0.11235175929137245,
    'max_depth': 14,
    'min_child_weight': 7.3,
    'subsample': 0.5941717562234573,
    'colsample_bytree': 0.6379404167499119,
    'reg_lambda': 8.227997337974575,
    'reg_alpha': 0.6782738955642913,
    'scale_pos_weight': 7.3,
    'n_estimators': 1000,
    'n_jobs': -1,
    'random_state': 42
}

# --- Best parameters for CatBoost ---

# Best AUC: 0.9636387741898083
# Best Parameters:
# {'learning_rate': 0.18994920012980027, 'depth': 8, 'l2_leaf_reg': 4.598210032803631, 
# 'bagging_temperature': 0.913760790136421, 'random_strength': 6.152008494995199, 'scale_pos_weight': 3.4394486639201385}

best_params_cat = {
    'bootstrap_type': 'Bayesian',        
    'od_type': 'Iter',
    'learning_rate': 0.18994920012980027,
    'depth': 8,
    'l2_leaf_reg': 4.598210032803631,
    'bagging_temperature': 0.913760790136421,
    'random_strength': 6.152008494995199,
    'scale_pos_weight': 7.3,
    'iterations': 1000,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': 0,
    'task_type': 'CPU',
    'random_seed': 42
}


# model_lgb = LGBMClassifier(**best_params_lgb)
# model_xgb = XGBClassifier(**best_params_xgb)
# model_cat = CatBoostClassifier(**best_params_cat)

# stacked_model = StackingClassifier(
#     estimators=[
#         ('lgb', model_lgb),
#         ('xgb', model_xgb),
#         ('cat', model_cat)
#     ],
#     final_estimator=LogisticRegression(max_iter=100000,solver='saga'),
#     n_jobs=-1,
#     passthrough=False
# )

# stacked_model.fit(X, y)


model_lgb = LGBMClassifier(**best_params_lgb)
model_xgb = XGBClassifier(**best_params_xgb)
model_cat = CatBoostClassifier(**best_params_cat)

vote_model = VotingClassifier(
    estimators=[
        ('LGBM', model_lgb),
        ('XGB', model_xgb),
        ('CatBoost', model_cat)
    ],
    weights=[1.5,1.3,1.1],
    voting='soft',        
    n_jobs=-1             
)

vote_model.fit(X, y)


accuracy_score(y,vote_model.predict(X))


prediction = vote_model.predict(X)


report = classification_report(y, prediction)
print(report)


cm = confusion_matrix(y, prediction)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


test_predict = vote_model.predict(data_te)
test_predict


Submission = pd.DataFrame({'id':data_te['id'],'y':test_predict})
Submission


Submission.to_csv('submission.csv',index = False)


Submission['y'].value_counts()




