# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,GridSearchCV,cross_val_score,cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor,ExtraTreesRegressor
from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
import time
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping
from sklearn.pipeline import make_pipeline


train=pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv",index_col='id')
test=pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv",index_col='id')
submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")



best={'max_depth': 5, 'min_child_weight': 3, 'gamma': 0.3, 'subsample': 1.0, 'colsample_bytree': 0.7}


train.head()


fig,ax=plt.subplots(3,3,figsize=(12,8))
ax=ax.flatten()
for i,col in enumerate(train.columns[:-1]):
    sns.histplot(data=train,x=col,ax=ax[i])
plt.tight_layout()



palette = sns.color_palette("husl")
plt.figure(figsize=(12,8))
mask=np.triu(train.corr())
sns.heatmap(train.corr(),mask=mask,cmap=palette,annot=True)


def cross_term(df,test=False):
    eps = 1e-6
    df=df.copy()
    if not test:
        columns=df.columns[:-1]
    else:
        columns=df.columns
    cross_term={}
    for i in range (len(columns)):
        for j in range(i+1,len(columns)):
            feat1=columns[i]
            feat2=columns[j]
            cross_term_div=f"{feat1}/{feat2}"
            cross_term_mul=f"{feat1}_x_{feat2}"
            df[cross_term_div]=df[feat1]/(df[feat2]+eps)
            df[cross_term_mul]=df[feat1]*df[feat2]
    return df
train=cross_term(train)
test=cross_term(test,test=True)
    


train.columns


y=train['BeatsPerMinute']
X=train.drop('BeatsPerMinute',axis=1)
X_test=test


RANDOM_STATE = 42
kf = KFold(n_splits=20, shuffle=True, random_state=RANDOM_STATE)

# ---- LightGBM ----
oof_lgb = np.zeros(len(X))
preds_lgb = np.zeros(len(X_test))
best_iters_lgb = []

for fold, (trn_idx, val_idx) in enumerate(kf.split(X), 1):
    X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    model_lgb = LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=128,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1
    )

    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=200, verbose=False)]
    )

    best_iters_lgb.append(model_lgb.best_iteration_)

    oof_lgb[val_idx] = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration_)
    preds_lgb += model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration_) / kf.n_splits

    rmse_fold = mean_squared_error(y_val, oof_lgb[val_idx], squared=False)
    print(f"[LGBM] Fold {fold} RMSE: {rmse_fold:.6f} | Best Iter: {model_lgb.best_iteration_}")

rmse_lgb = mean_squared_error(y, oof_lgb, squared=False)
print(f"[LGBM] OOF RMSE: {rmse_lgb:.6f}")
final_iter_lgb = int(np.median(best_iters_lgb))
print(f"[LGBM] Median best_iteration = {final_iter_lgb}")

final_model_lgb = LGBMRegressor(
    n_estimators=final_iter_lgb,
    learning_rate=0.02,
    num_leaves=128,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
final_model_lgb.fit(X, y)
final_preds_lgb = final_model_lgb.predict(X_test)


# ---- XGBoost ----
oof_xgb = np.zeros(len(X))
preds_xgb = np.zeros(len(X_test))
best_iters_xgb = []

for fold, (trn_idx, val_idx) in enumerate(kf.split(X), 1):
    X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    model_xgb = XGBRegressor(
        n_estimators=5000,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse",            # move here
        early_stopping_rounds=200  
    )

    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    best_iters_xgb.append(model_xgb.best_iteration)

    oof_xgb[val_idx] = model_xgb.predict(X_val, iteration_range=(0, model_xgb.best_iteration))
    preds_xgb += model_xgb.predict(X_test, iteration_range=(0, model_xgb.best_iteration)) / kf.n_splits

    rmse_fold = mean_squared_error(y_val, oof_xgb[val_idx], squared=False)
    print(f"[XGB] Fold {fold} RMSE: {rmse_fold:.6f} | Best Iter: {model_xgb.best_iteration}")

rmse_xgb = mean_squared_error(y, oof_xgb, squared=False)
print(f"[XGB] OOF RMSE: {rmse_xgb:.6f}")
final_iter_xgb = int(np.median(best_iters_xgb))
print(f"[XGB] Median best_iteration = {final_iter_xgb}")

final_model_xgb = XGBRegressor(
    n_estimators=final_iter_xgb,
    learning_rate=0.02,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist"
)
final_model_xgb.fit(X, y)
final_preds_xgb = final_model_xgb.predict(X_test)


weights = np.linspace(0, 1, 101)  # try 0.00, 0.01, ..., 1.00
rmse_scores = []

for w in weights:
    oof_blend = w * oof_lgb + (1 - w) * oof_xgb
    rmse = mean_squared_error(y, oof_blend, squared=False)
    rmse_scores.append(rmse)

best_w = weights[np.argmin(rmse_scores)]
best_rmse = min(rmse_scores)

print(f"Best weight: {best_w:.2f} | RMSE: {best_rmse:.6f}")


ensemble_preds = best_w * preds_lgb + (1 - best_w) * preds_xgb
final_ensemble = best_w * final_preds_lgb + (1 - best_w) * final_preds_xgb


sub=pd.DataFrame({'id':test.index,'BeatsPerMinute':final_ensemble})
sub.to_csv('submission.csv',index=False)




