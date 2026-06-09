import warnings
import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

warnings.filterwarnings('ignore')
os.environ['LGBM_LOG_LEVEL'] = '-1'


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.compose import make_column_selector
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


import lightgbm as lgb
from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation


## Data loading and initializing

train_path = "/kaggle/input/playground-series-s5e12/train.csv"
test_path = "/kaggle/input/playground-series-s5e12/test.csv"
X = pd.read_csv(train_path).drop("diagnosed_diabetes", axis = 1)
y = pd.read_csv(train_path)["diagnosed_diabetes"]
X_test = pd.read_csv(test_path)


## AGE & FAMILY HISTORY INTERACTION
X["af_risk"] = X["family_history_diabetes"] + X["family_history_diabetes"] * X["age"] * 0.30
X["af_risk"] = X["af_risk"].astype(float)

X_test["af_risk"] = X_test["family_history_diabetes"] + X_test["family_history_diabetes"] * X_test["age"] * 0.30
X_test["af_risk"] = X_test["af_risk"].astype(float)


## AGE & TRIGLYCERIDES
X["at_risk"] = X["triglycerides"] + X["triglycerides"] * X["age"] * 0.15
X["at_risk"] = X["at_risk"].astype(float)

X_test["at_risk"] = X_test["triglycerides"] + X_test["triglycerides"] * X_test["age"] * 0.15
X_test["at_risk"] = X_test["at_risk"].astype(float)


## Colesterol Rate
X["chol_rate"] = X["cholesterol_total"] / X["hdl_cholesterol"]
X["chol_rate"] = X["chol_rate"].astype(float)

X_test["chol_rate"] = X_test["cholesterol_total"] / X_test["hdl_cholesterol"]
X_test["chol_rate"] = X_test["chol_rate"].astype(float)


# Categorical Feature Handling to leverage LightGBM's native support for categorical variables

cat_feats = make_column_selector(dtype_include = object)(X)
X[cat_feats] = X[cat_feats].astype("category")
X_test[cat_feats] = X_test[cat_feats].astype("category")
## cat_features_indices = [X.columns.get_loc(col) for col in cat_feats]


X.drop("gender", axis = 1, inplace = True)
X_test.drop("gender", axis = 1, inplace = True)


## lightgbm Hyperparameter configuration

lgb_params = {'learning_rate': 0.059216255749261655,
               'num_leaves': 26,
               'max_depth': 4,
               'lambda_l1': 1.3404844864067962,
               'lambda_l2': 3.1381681073903975e-07,
               'min_child_samples': 95,
               'subsample': 0.9745291249731525,
               'colsample_bytree': 0.5645863195919457,
               "device": "gpu",
               'objective': 'binary',
               'metric': 'auc',
               'verbosity': -1,
               'n_jobs': -1,
               'random_state': 42,
               'n_estimators': 6000} ## 5000


kf = StratifiedKFold(n_splits = 12, shuffle = True, random_state = 136)

todas_las_predicciones = []
scores_accuracy = []

oof_predictions = np.zeros(len(X))

scores_folds = []
predicciones_test_por_modelo = []


for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    
    # Preparamos los datos del fold
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]

    # 2. Creamos los Pools pasando los ÍNDICES numéricos
    ##train_pool = Pool(X_train_fold, y_train_fold, cat_features = cat_features_indices)
    ## val_pool = Pool(X_val_fold, y_val_fold, cat_features = cat_features_indices)

    # 3. Definimos el modelo y le pasamos los mismos índices
    lgb_model = lgb.LGBMClassifier(**lgb_params)

    # 4. Entrenamos
    lgb_model.fit(X_train_fold, y_train_fold, eval_set = [(X_val_fold, y_val_fold)])

    prob_val = lgb_model.predict_proba(X_val_fold)[:, 1]
    oof_predictions[val_idx] = prob_val
    
    pred_val_clase = (prob_val > 0.5).astype(int) # Convertimos probabilidad a 0 o 1
    score_fold = accuracy_score(y_val_fold, pred_val_clase)
    scores_folds.append(score_fold)
    
    pred_test = lgb_model.predict_proba(X_test[X.columns])[:, 1]
    predicciones_test_por_modelo.append(pred_test)
    
    print(f"--> Fold {fold + 1} listo. Accuracy: {score_fold:.4f}")
    print(f"--> Fold {fold + 1} entrenado con éxito.")


prediccion_final_test = np.mean(predicciones_test_por_modelo, axis = 0)


submission = pd.DataFrame({'id': X_test['id'], 'diagnosed_diabetes': prediccion_final_test})


submission.to_csv("submission.csv", index = False)

