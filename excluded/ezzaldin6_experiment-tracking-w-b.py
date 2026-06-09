import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import wandb
from kaggle_secrets import UserSecretsClient

# Use wandb-core, temporary for wandb's new backend
wandb.require("core")
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("WNADB_API_KEY")
wandb.login(key=api_key)


train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train_df.set_index("id", inplace=True)
train_df.info()


from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

SPACE = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'num_leaves': hp.quniform('num_leaves', 20, 300, 1),
    'max_depth': hp.quniform('max_depth', 3, 12, 1),
    'min_child_samples': hp.quniform('min_child_samples', 10, 200, 1),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
}


from sklearn.model_selection import train_test_split

X, y = train_df.drop("rainfall", axis=1), train_df["rainfall"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)


import lightgbm as lgb
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                             roc_auc_score)
from wandb.sklearn import plot_precision_recall, plot_feature_importances
from wandb.sklearn import plot_class_proportions, plot_learning_curve, plot_roc

def objective(params):
    # Initialize W&B run
    with wandb.init(project="Binary Prediction with a Rainfall Dataset", reinit=True):
        # Convert parameters to appropriate types
        params = {
            'learning_rate': params['learning_rate'],
            'num_leaves': int(params['num_leaves']),
            'max_depth': int(params['max_depth']),
            'min_child_samples': int(params['min_child_samples']),
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'reg_alpha': params['reg_alpha'],
            'reg_lambda': params['reg_lambda'],
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'random_state': 42,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  #early_stopping_rounds=10,
                  #verbose=False
                 )
        
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_proba[:, 1])
        
        wandb.log({
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'best_iteration': model.best_iteration_
        })
        wandb.log({"pr": plot_precision_recall(y_val, y_proba)})
        wandb.log({"roc": plot_roc(y_val, y_proba)})
        cm = wandb.sklearn.plot_confusion_matrix(y_val, y_pred, [0, 1]) 
        wandb.log({"conf_mat": cm})
        wandb.log({"hyperparameters": params})
        return {'loss': 1 - f1, 'status': STATUS_OK}


trials = Trials()
best = fmin(fn=objective,
            space=SPACE,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)


# Train final model with best hyperparameters
best_params = {
    'learning_rate': best['learning_rate'],
    'num_leaves': int(best['num_leaves']),
    'max_depth': int(best['max_depth']),
    'min_child_samples': int(best['min_child_samples']),
    'subsample': best['subsample'],
    'colsample_bytree': best['colsample_bytree'],
    'reg_alpha': best['reg_alpha'],
    'reg_lambda': best['reg_lambda'],
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'random_state': 42,
}

final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X=X, y=y)

X_test = test_df.set_index("id")
y_preds = final_model.predict(X_test)




