import numpy as np
import pandas as pd
import os
import warnings
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
warnings.filterwarnings('ignore')
os.environ['LGBM_LOG_LEVEL'] = '-1'


from sklearn.compose import make_column_selector
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


import xgboost as xgb


train_path = "/kaggle/input/playground-series-s5e12/train.csv"
test_path = "/kaggle/input/playground-series-s5e12/test.csv"
X = pd.read_csv(train_path).drop("diagnosed_diabetes", axis = 1)
y = pd.read_csv(train_path)["diagnosed_diabetes"]
X_test = pd.read_csv(test_path)
X.shape, y.shape, X_test.shape


X.drop("gender", axis = 1, inplace = True)
X_test.drop("gender", axis = 1, inplace = True)


## AGE & TRIGLYCERIDES
X["at_risk2"] = X["triglycerides"] + X["triglycerides"] * X["age"] * 0.30
X["at_risk2"] = X["at_risk2"].astype(float)

X_test["at_risk2"] = X_test["triglycerides"] + X_test["triglycerides"] * X_test["age"] * 0.30
X_test["at_risk2"] = X_test["at_risk2"].astype(float)


cat_feats = make_column_selector(dtype_include = object)(X)
X[cat_feats] = X[cat_feats].astype("category")
X_test[cat_feats] = X_test[cat_feats].astype("category")


kf = StratifiedKFold(n_splits = 2, shuffle = True, random_state = 42)

todas_las_predicciones = []
scores_accuracy = []

oof_predictions = np.zeros(len(X))

scores_folds = []
predicciones_test_por_modelo = []


xgb_params ={'n_estimators': 4500,
             'early_stopping_rounds': 100,
             'booster': 'gbtree',
             'tree_method': 'hist', 
             'eval_metric': 'auc',
             'learning_rate': 0.010586281318793418, 
             'max_depth': 5, 
             'subsample': 0.9419910623833896, 
             'colsample_bytree': 0.5244058847875112, 
             'min_child_weight': 7, 
             'reg_alpha': 0.00015151084454479046, 
             'reg_lambda': 2.161158791085214e-08, 
             'gamma': 2.240078485583776e-07,
             "verbosity": 0,
             "enable_categorical": True,
             "random_state": 42,
             "device": "cuda"}


for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    xgb_model = xgb.XGBClassifier(**xgb_params)
    
    xgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)])

    prob_val = xgb_model.predict_proba(X_val_fold)[:, 1]
    oof_predictions[val_idx] = prob_val
    
    pred_val_clase = (prob_val > 0.5).astype(int) # Convertimos probabilidad a 0 o 1
    score_fold = accuracy_score(y_val_fold, pred_val_clase)
    scores_folds.append(score_fold)
    
    pred_test = xgb_model.predict_proba(X_test[X.columns])[:, 1]
    predicciones_test_por_modelo.append(pred_test)
    
    print(f"--> Fold {fold + 1} listo. Accuracy: {score_fold:.4f}")
    
    print(f"--> Fold {fold + 1} entrenado con éxito.")


prediccion_final_test = np.mean(predicciones_test_por_modelo, axis = 0)


sub_path = "/kaggle/input/playground-series-s5e12/sample_submission.csv"
df_sub = pd.read_csv(sub_path)
df_sub["diagnosed_diabetes"] = prediccion_final_test


df_sub.to_csv("submission.csv", index = False)




