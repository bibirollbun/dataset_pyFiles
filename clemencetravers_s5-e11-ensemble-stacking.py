import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score 

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier


df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


df.head()


df.info()


OHE= OneHotEncoder(handle_unknown='ignore', sparse_output=False)


s =(df.dtypes =='object')
object_cols =list (s[s].index)


OH_cols = pd.DataFrame(OHE.fit_transform(df[object_cols]))

# One-hot encoding removed index; put it back
OH_cols.index = df.index

# Remove categorical columns (will replace with one-hot encoding)
num_df = df.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_df = pd.concat([num_df, OH_cols], axis=1)


X=OH_df.drop(['id', 'loan_paid_back'],axis=1)
y=OH_df.pop('loan_paid_back')


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


model_XGB = XGBClassifier(n_estimators=700, learning_rate=0.05,early_stopping_rounds=10, eval_metric='auc', subsample=0.8)

model_XGB.fit(train_X, train_y,
           eval_set=[(val_X, val_y)]
)

pred_y_XGB = model_XGB.predict_proba(val_X)[:,1]

XGB_score = roc_auc_score(val_y, pred_y_XGB)

print(f"ROC AUC Score: {XGB_score}")



model_LGB= LGBMClassifier(num_leaves=50, min_child_samples= None, max_depth=20, learning_rate=0.08, n_estimators=400, force_row_wise= True, metric ='auc') 

model_LGB.fit(train_X, train_y,
           eval_set=[(val_X, val_y)]
)

pred_y_LGB = model_LGB.predict_proba(val_X)[:,1]


LGB_score = roc_auc_score(val_y, pred_y_LGB)

print(f"ROC AUC Score: {LGB_score}")


model_cat= CatBoostClassifier(iterations= 1000, depth=8, learning_rate=0.08, eval_metric='AUC', od_wait=20)


model_cat.fit(train_X, train_y,
           eval_set=[(val_X, val_y)]
)

pred_y_CAT = model_cat.predict_proba(val_X)[:, 1]


CAT_score = roc_auc_score(val_y, pred_y_CAT)

print(f"ROC AUC Score: {CAT_score}")


def auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def normalize(weights):
    w = np.abs(np.array(weights, dtype=float))
    s = w.sum()
    return (w / s) if s > 0 else np.array([1/3, 1/3, 1/3])

def eval_weights(w):
    w = normalize(w)
    pred = w[0]*pred_y_XGB + w[1]*pred_y_LGB + w[2]*pred_y_CAT
    return auc(val_y, pred)

best_w = np.array([1/3, 1/3, 1/3])
best_score = eval_weights(best_w)

try:
    from scipy.optimize import minimize
    # on MINIMISE -AUC pour MAXIMISER l'AUC
    res = minimize(
        lambda w: -eval_weights(w),
        x0=np.array([0.2, 0.4, 0.4]),
        method='Nelder-Mead',
        options={'maxiter': 1500}
    )
    w_opt = normalize(res.x)
    score_opt = eval_weights(w_opt)
    # maintenant on veut le score LE PLUS GRAND
    if score_opt > best_score:
        best_w, best_score = w_opt, score_opt
except Exception as _:
    # Random simplex fallback
    rng = np.random.default_rng(SEED)
    for _ in range(500):
        w = rng.random(3); w = w / w.sum()
        s = eval_weights(w)
        # pareil ici : on garde le meilleur (max) score
        if s > best_score:
            best_w, best_score = w, s

ensemble_weights = {
    'xgb': float(best_w[0]),
    'lgb': float(best_w[1]),
    'cat':  float(best_w[2])
}

ensemble_oof = (
    ensemble_weights['xgb'] * pred_y_XGB +
    ensemble_weights['lgb'] * pred_y_LGB +
    ensemble_weights['cat']  * pred_y_CAT
)

ensemble_cv = auc(val_y, ensemble_oof)

print("="*50)
print("FINAL MODEL COMPARISON")
print("="*50)
print(f"XGBoost CV AUC:     {XGB_score:.5f}")
print(f"LightGBM CV AUC:    {LGB_score:.5f}")
print(f"CatBoost CV AUC:    {CAT_score:.5f}")
print("-"*50)
print("Optimal Weights:")
print(f"  XGB: {ensemble_weights['xgb']:.4f}")
print(f"  LGB: {ensemble_weights['lgb']:.4f}")
print(f"  CAT: {ensemble_weights['cat']:.4f}")
print("="*50)


OH_cols_test = pd.DataFrame(OHE.fit_transform(df_test[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_test.index = df_test.index

# Remove categorical columns (will replace with one-hot encoding)
num_df_test = df_test.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_df_test = pd.concat([num_df_test, OH_cols_test], axis=1)


id=OH_df_test.pop('id')


predictions_xgb= model_XGB.predict_proba(OH_df_test)[:, 1]
predictions_lgb= model_LGB.predict_proba(OH_df_test)[:, 1]
predictions_cat=model_cat.predict_proba(OH_df_test)[:, 1]


predictions= predictions_xgb * ensemble_weights['xgb'] + predictions_lgb * ensemble_weights['lgb'] + predictions_cat * ensemble_weights['cat']


predictions = predictions.flatten()


output = pd.DataFrame({ 'id':id,
                       'Target': predictions})


output.set_index('id')


output.to_csv('submission.csv', index=False)

