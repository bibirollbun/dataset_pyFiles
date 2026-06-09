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


df_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


df_train.shape


df_train.head()


df_train.columns


df_train['Crop Type'].value_counts().to_dict()


df_train['Fertilizer Name'].unique()


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score


df_train = df_train.sample(frac=1,random_state=42).reset_index(drop=True)


y = df_train['Fertilizer Name']
X = df_train.drop(columns=['Fertilizer Name','id'],axis=1)
X_test = df_test.drop(columns=['id'],axis=1)


X = pd.get_dummies(X, columns=['Soil Type', 'Crop Type'])
X_test = pd.get_dummies(X_test, columns=['Soil Type', 'Crop Type'])


X


Le = LabelEncoder()
y = Le.fit_transform(y)


type(y)





X_small,_,y_small,_ = train_test_split(X,y,train_size=0.15,random_state=42,stratify=y)


X_small.shape


from sklearn.model_selection import StratifiedKFold,cross_val_score
import optuna
from xgboost import XGBClassifier


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "eval_metric":"mlogloss",
        'use_label_encoder':False,
        "tree_method":"hist",
        "device":"cuda",
        "random_state": 42,
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    top3_scores=[]
    # labels = list(range(7))
    for train_idx,val_idx in skf.split(X_small,y_small):
        X_train_fold,X_val_fold = X.iloc[train_idx],X.iloc[val_idx]
        y_train_fold,y_val_fold = y[train_idx],y[val_idx]
        # model = XGBClassifier(**params,objective='multi:softprob',num_class=7,verbosity=0, n_jobs=-1)
        model = XGBClassifier(**params,n_jobs=-1)
        model.fit(X_train_fold,y_train_fold)
        y_proba = model.predict_proba(X_val_fold)
        # score = top_k_accuracy_score(y_val_fold,y_proba,k=3,labels=labels)
        score = top_k_accuracy_score(y_val_fold,y_proba,k=3)
        top3_scores.append(score)
    print(np.mean(top3_scores))
    return np.mean(top3_scores)
    


study = optuna.create_study(direction = 'maximize',sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50)
print("Best trial:")
print(study.best_trial)


best_params_optuna = study.best_trial.params
best_params_optuna


best_model = XGBClassifier(**best_params_optuna,use_label_encoder=False,eval_metric = 'mlogloss',tree_method='hist',device='cuda',random_state=42,n_jobs=-1)


best_model.fit(X,y)


y_proba = best_model.predict_proba(X_test)
top3_preds = np.argsort(y_proba,axis=1)[:,-3:][:,::-1]


top3_preds


top3_labels = Le.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)
top3_labels


top3_str = [' '.join(row) for row in top3_labels]
len(top3_str)


df_sub['Fertilizer Name'] = top3_str
df_sub.head()


df_sub.to_csv('submission_1.csv',index=False)

