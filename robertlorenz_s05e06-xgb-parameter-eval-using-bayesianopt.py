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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

try:
    from bayes_opt import BayesianOptimization
except ImportError:
    print(f"[INFO] Bayesian-optimization is required, installing...")
    %pip install -U bayesian-optimization
    from bayes_opt import BayesianOptimization


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", index_col="id")


cat_cols = [col for col in train.select_dtypes(include=['object']).columns if col != "Fertilizer Name"]

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

label_encoder = LabelEncoder()
train['Fertilizer Name'] = label_encoder.fit_transform(train['Fertilizer Name'])


# Splitting the training data into training and validation sets
train_, val = train_test_split(train, test_size=0.2, random_state=42)

X_train = train_.drop(columns=["Fertilizer Name"]).reset_index(drop=True)
y_train = train_["Fertilizer Name"].reset_index(drop=True)

X_val = val.drop(columns=["Fertilizer Name"]).reset_index(drop=True)
y_val = val["Fertilizer Name"].reset_index(drop=True)

dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)



def xgb_evaluate(eta, gamma, max_depth, min_child_weight, max_delta_step, subsample, colsample_bytree, reg_lambda, alpha, n_estimators, early_stopping_rounds): 
    
    # model = xgb.XGBClassifier(
    #     eta=eta,
    #     gamma=gamma,
    #     max_depth=int(max_depth),
    #     min_child_weight=int(min_child_weight),
    #     max_delta_step=int(max_delta_step),
    #     subsample=subsample,
    #     colsample_bytree=colsample_bytree, 
    #     reg_lambda=reg_lambda, 
    #     alpha=alpha,
    #     n_estimators=int(n_estimators),
    #     early_stopping_rounds=int(early_stopping_rounds),
    #     objective='multi:softprob',
    #     eval_metric='mlogloss',
    #     device= "cuda",
    #     n_jobs=-1,
    #     random_state=42,        
    #     enable_categorical=True
    # )
    # model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=None)
    # y_pred_proba = model.predict_proba(X_val)

    params = {
        'eta': eta,
        'gamma': gamma,
        'max_depth': int(max_depth),
        'min_child_weight': int(min_child_weight),
        'max_delta_step': int(max_delta_step),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_lambda': reg_lambda,
        'alpha': alpha,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'device': 'cuda',
        'n_jobs': -1,
        'random_state': 42,
        'num_class': 7
    }

    # Use xgb.train() instead of XGBClassifier
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(n_estimators),
        evals=[(dval, 'validation')],
        early_stopping_rounds=int(early_stopping_rounds),
        verbose_eval=False
    )

    y_pred_proba = model.predict(dval)
    # MAP@3 evaluation
    map_score = 0.0
    for j in range(len(y_val)):
        # Get top 3 predictions for this sample
        top_3_preds = np.argsort(y_pred_proba[j])[::-1][:3]
        
        correct = 0
        precision = 0.0
        
        for k, pred in enumerate(top_3_preds):
            if pred == y_val[j]:
                correct += 1
                precision += correct / (k + 1)
        
        # Average precision for this sample
        if correct > 0:
            map_score += precision / min(1, correct)
    map3_score = map_score / len(y_val)
    return map3_score

optimizer = BayesianOptimization(
    f=xgb_evaluate,
    pbounds={ # more values = more time
        'eta': (0.005, 0.03),
        'gamma': (0.15, 0.25),
        'max_depth': (15, 25),
        'min_child_weight': (3, 7),
        'max_delta_step': (3, 7),
        'subsample': (0.5, 1),
        'colsample_bytree': (0.5, 0.8),
        'reg_lambda': (1, 2),
        'alpha': (1, 5),
        'n_estimators': (3000, 10000), #(1000, 5000),
        'early_stopping_rounds': (250, 500),
    },
    random_state=42
)
optimizer.maximize(init_points=5, n_iter=25)
print("Best parameters:", optimizer.max['params'])


xgb_params = optimizer.max['params']
xgb_params['max_depth'] = int(xgb_params['max_depth'])
xgb_params['min_child_weight'] = int(xgb_params['min_child_weight'])  
xgb_params['n_estimators'] = int(xgb_params['n_estimators'])
xgb_params['max_delta_step'] = int(xgb_params['max_delta_step'])
xgb_params['early_stopping_rounds'] = int(xgb_params['early_stopping_rounds'])

model_submission = xgb.XGBClassifier(**xgb_params)

model_submission.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=500)

X_test = test
test_ids = test.index
test_preds = model_submission.predict_proba(X_test)
top3_idx = np.argsort(-test_preds, axis=1)[:, :3]
labels = label_encoder.inverse_transform(top3_idx.ravel())
pred_names = labels.reshape(top3_idx.shape)

submission_format = [' '.join(row) for row in pred_names]

submission = pd.DataFrame({'id': test_ids, 'Fertilizer Name': submission_format})
submission.to_csv('submission.csv', index=False)
submission.head()

