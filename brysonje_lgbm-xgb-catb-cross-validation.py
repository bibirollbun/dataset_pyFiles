import os
import warnings
import numpy as np
import pandas as pd


import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool


from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.compose import make_column_selector
from sklearn.model_selection import StratifiedKFold


## Environment Setup

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
warnings.filterwarnings('ignore', category = UserWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
os.environ['LGBM_LOG_LEVEL'] = '-1'


## Data loading and initializing

train_path = "/kaggle/input/playground-series-s5e12/train.csv"
test_path = "/kaggle/input/playground-series-s5e12/test.csv"
X = pd.read_csv(train_path).drop("diagnosed_diabetes", axis = 1)
y = pd.read_csv(train_path)["diagnosed_diabetes"]
X_test = pd.read_csv(test_path)


X.drop("gender", axis = 1, inplace = True)
X_test.drop("gender", axis = 1, inplace = True)


## AGE & TRIGLYCERIDES
X["at_risk"] = X["triglycerides"] + X["triglycerides"] * X["age"] * 0.30
X["at_risk"] = X["at_risk"].astype(float)

X_test["at_risk"] = X_test["triglycerides"] + X_test["triglycerides"] * X_test["age"] * 0.30
X_test["at_risk"] = X_test["at_risk"].astype(float)

## AGE & FAMILY HISTORY INTERACTION
X["af_risk"] = X["family_history_diabetes"] + X["family_history_diabetes"] * X["age"] * 0.15
X["af_risk"] = X["af_risk"].astype(float)

X_test["af_risk"] = X_test["family_history_diabetes"] + X_test["family_history_diabetes"] * X_test["age"] * 0.15
X_test["af_risk"] = X_test["af_risk"].astype(float)

## Colesterol Rate
X["chol_rate"] = X["cholesterol_total"] / X["hdl_cholesterol"]
X["chol_rate"] = X["chol_rate"].astype(float)

X_test["chol_rate"] = X_test["cholesterol_total"] / X_test["hdl_cholesterol"]
X_test["chol_rate"] = X_test["chol_rate"].astype(float)


## Categorical Feature Handling to leverage LightGBM's native support for categorical variables

cat_feats = make_column_selector(dtype_include = object)(X)
X[cat_feats] = X[cat_feats].astype("category")
X_test[cat_feats] = X_test[cat_feats].astype("category")
## cat_features_indices = [X.columns.get_loc(col) for col in cat_feats]


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
               'n_estimators': 5000} ## 5000


## CatBoost Hyperparameter configuration

cbc_params = {"iterations": 1500, ##1156
              'learning_rate': 0.04359224667234545,
              'depth': 6,
              'l2_leaf_reg': 7.678939079305491,
              'random_strength': 0.6252073339200809,
              'min_data_in_leaf': 21,
              "verbose": 0,
              "task_type": "GPU",
              "devices": "0"}
              ## 'subsample': 0.9251861002711826}


## XgBoost Hyperparameter configuration

xgb_params ={'n_estimators': 4500, ## 4500
             'early_stopping_rounds': 100,
             'booster': 'gbtree',
             'tree_method': 'hist', 
             'eval_metric': 'auc',
             'verbosity': 0,
             'learning_rate': 0.010586281318793418, 
             'max_depth': 5, 
             'subsample': 0.9419910623833896, 
             'colsample_bytree': 0.5244058847875112, 
             'min_child_weight': 7, 
             'reg_alpha': 0.00015151084454479046, 
             'reg_lambda': 2.161158791085214e-08, 
             'gamma': 2.240078485583776e-07,
             "enable_categorical": True,
             "random_state": 42,
             "device": "cuda"}


def train_and_predict(base_model, X, y, X_test, n_splits = 10):
    
    X = X.reset_index(drop = True)
    y = y.reset_index(drop = True)
    
    oof_preds = np.zeros(len(X))
    test_preds_list = []
    fold_scores = []
    
    kf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 136)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model = clone(base_model)
        
        model.fit(X_train_fold, y_train_fold, eval_set = [(X_val_fold, y_val_fold)])
        
        val_probs = model.predict_proba(X_val_fold)[:, 1]
        oof_preds[val_idx] = val_probs  
        
        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds_list.append(test_probs)
        
        fold_auc = roc_auc_score(y_val_fold, val_probs)
        fold_scores.append(fold_auc)
        
        print(f"Fold {fold + 1} | AUC: {fold_auc:.5f}")
    
    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    
    # Cálculo extra: Accuracy con umbral 0.5 solo para comparar
    temp_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))
    
    print(f"\nOverall Mean AUC: {mean_auc:.5f} (+/- {std_auc:.5f})")
    print(f"Overall OOF Accuracy (0.5 threshold): {temp_acc:.5f}")
    
    final_test_preds = np.mean(test_preds_list, axis=0)
    
    return oof_preds, final_test_preds, mean_auc


# Execute for LightGBM
lgb_model = lgb.LGBMClassifier(**lgb_params)
oof_lgb, test_lgb, auc_lgb = train_and_predict(lgb_model, X, y, X_test)


# Execute for catboost
cbc_model = CatBoostClassifier(**cbc_params, cat_features = cat_feats)
oof_cbc, test_cbc, auc_cbc = train_and_predict(cbc_model, X, y, X_test)


# Execute for xgboost
xgb_model = xgb.XGBClassifier(**xgb_params)
oof_xgb, test_xgb, auc_xgb = train_and_predict(xgb_model, X, y, X_test)


preds_dict = {'LGBM': test_lgb, 'CatBoost': test_cbc, 'XGBoost': test_xgb}

df_corr = pd.DataFrame(preds_dict).corr()

print("Matriz de Correlación entre modelos:")
print(df_corr)

mean_corr = df_corr.values[np.triu_indices(df_corr.shape[0], k=1)].mean()
print(f"\nCorrelación promedio: {mean_corr:.4f}")


best_auc = 0
best_weights = (0, 0, 0)

# Probamos combinaciones de pesos en pasos de 0.05
for w_lgb in np.linspace(0.5, 0.9, 9): # LightGBM suele ser el más fuerte
    for w_xgb in np.linspace(0, 0.4, 9):
        w_cbc = 1.0 - w_lgb - w_xgb
        
        if w_cbc < 0: continue # Aseguramos que el tercer peso no sea negativo
        
        # Calculamos la predicción combinada en OOF
        current_oof = (w_lgb * oof_lgb) + (w_xgb * oof_xgb) + (w_cbc * oof_cbc)
        current_auc = roc_auc_score(y, current_oof)
        
        if current_auc > best_auc:
            best_auc = current_auc
            best_weights = (w_lgb, w_xgb, w_cbc)

print(f"Mejor AUC en OOF: {best_auc:.5f}")
print(f"Mejores Pesos -> LGBM: {best_weights[0]:.2f}, XGB: {best_weights[1]:.2f}, CBC: {best_weights[2]:.2f}")


## Submission file generation
submission = pd.DataFrame({'id': X_test['id'], 'diagnosed_diabetes': test_lgb})

submission.to_csv('submission.csv', index = False)




