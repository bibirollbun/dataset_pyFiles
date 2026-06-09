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


from lightgbm import LGBMClassifier
import lightgbm as lgb


from sklearn.compose import make_column_selector
from sklearn.metrics import accuracy_score


train_path = "/kaggle/input/playground-series-s5e12/train.csv"
test_path = "/kaggle/input/playground-series-s5e12/test.csv"
X = pd.read_csv(train_path).drop("diagnosed_diabetes", axis = 1)
y = pd.read_csv(train_path)["diagnosed_diabetes"]
X_test = pd.read_csv(test_path)


cat_feats = make_column_selector(dtype_include = object)(X)
X[cat_feats] = X[cat_feats].astype("category")
X_test[cat_feats] = X_test[cat_feats].astype("category")


from sklearn.model_selection import StratifiedKFold
import numpy as np

## kf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
kf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 24)

todas_las_predicciones = []
scores_accuracy = []


lgbm_params = {'learning_rate': 0.059216255749261655,
               'num_leaves': 26,
               'max_depth': 4,
               'lambda_l1': 1.3404844864067962,
               'lambda_l2': 3.1381681073903975e-07,
               'min_child_samples': 95,
               'subsample': 0.9745291249731525,
               'colsample_bytree': 0.5645863195919457,
               'objective': 'binary',
               'metric': 'auc',
               'verbosity': -1,
               'n_jobs': -1,
               'random_state': 42,
               'n_estimators': 5000}


oof_predictions = np.zeros(len(X))

scores_folds = []
predicciones_test_por_modelo = []


for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(**lgbm_params)
    
    model.fit(X_train_fold, y_train_fold, eval_set = [(X_val_fold, y_val_fold)],
              callbacks = [lgb.early_stopping(100), lgb.log_evaluation(0)])

    prob_val = model.predict_proba(X_val_fold)[:, 1]
    oof_predictions[val_idx] = prob_val
    
    pred_val_clase = (prob_val > 0.5).astype(int) # 0 o 1
    score_fold = accuracy_score(y_val_fold, pred_val_clase)
    scores_folds.append(score_fold)
    
    pred_test = model.predict_proba(X_test)[:, 1]
    predicciones_test_por_modelo.append(pred_test)
    
    print(f"--> Fold {fold + 1} Accuracy: {score_fold:.4f}")
    
    print(f"--> Fold {fold + 1} sucessfully trained.")


prediccion_final_test = np.mean(predicciones_test_por_modelo, axis = 0)


sub_path = "/kaggle/input/playground-series-s5e12/sample_submission.csv"
df_sub = pd.read_csv(sub_path)


df_sub["diagnosed_diabetes"] = prediccion_final_test


df_sub["diagnosed_diabetes"].value_counts()


df_sub.to_csv("submission1.csv", index = False)




