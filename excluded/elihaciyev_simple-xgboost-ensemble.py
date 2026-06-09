import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score


tr = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv',              index_col='id')
ts = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv',               index_col='id')
org = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv',    sep=';'       )
sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv', index_col='id')

display(tr.head())
display(ts.head())
display(org.head())
display(sub.head())


print(tr['y'].value_counts(normalize=True) * 100)


print(f"Shape of Train: {tr.shape}")
print(f"Shape of Original: {org.shape}")
print(f"Shape of Train: {ts.shape}")
print(f"Null of Train: {tr.isnull().sum().sum()}")
print(f"Null of Original: {org.isnull().sum().sum()}")
print(f"Null of Test: {ts.isnull().sum().sum()}")


cat_col = tr.select_dtypes(include=['object', 'category']).columns

encod = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
tr[cat_col] = encod.fit_transform(tr[cat_col])
org[cat_col] = encod.transform(org[cat_col])
ts[cat_col] = encod.transform(ts[cat_col])


org['y'] = org['y'].map({'no': 0, 'yes': 1})


#X = tr.drop(['y'], axis=1)
#y = tr['y']


params = {'n_estimators': 846, 'max_depth': 12, 'learning_rate': 0.048065112801328386, 'subsample': 0.7671297466933938,
          'colsample_bytree': 0.6830189748973862, 'gamma': 1.4133367677245654, 'min_child_weight': 10}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores, models = [], []

for fold, (tr_idx, vl_idx) in enumerate(skf.split(tr, tr['y'])):
    # 1) Ğ¡Ğ¾Ğ±Ğ¸Ñ€Ğ°ĞµĞ¼ Ğ¾Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ‘Ğ½Ğ½Ñ‹Ğ¹ train + org
    M_tr = pd.concat([ tr.iloc[tr_idx], org ], ignore_index=True)
    M_tr = M_tr.drop_duplicates(subset=ts.columns, keep="first", ignore_index=True)

    # 2) Ğ�Ñ‚Ğ´ĞµĞ»Ñ�ĞµĞ¼ Ñ‚Ğ°Ñ€Ğ³ĞµÑ‚
    ytr = M_tr['y']
    Xtr = M_tr.drop('y', axis=1)

    # 3) Ğ“Ğ¾Ñ‚Ğ¾Ğ²Ğ¸Ğ¼ Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¾Ğ½Ğ½Ñ‹Ğ¹ Ğ½Ğ°Ğ±Ğ¾Ñ€
    V = tr.loc[vl_idx].reset_index(drop=True)
    yvl = V['y']
    Xvl = V.drop('y', axis=1)

    # 4) Ğ�Ğ±ÑƒÑ‡Ğ°ĞµĞ¼ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ
    model = xgb.XGBClassifier(**params)
    model.fit(Xtr, ytr)

    # 5) Ğ�Ñ†ĞµĞ½ĞºĞ°
    pred = model.predict_proba(Xvl)[:, 1]
    score = roc_auc_score(yvl, pred)

    print(f'Fold {fold} ROC AUC: {score:.4f}')
    scores.append(score)
    models.append(model)

print(f"Final Score: {np.mean(scores):.4f}")



import optuna
def objective(trial):
    params = {
        "n_estimators": 20000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "random_state": 42,
        "n_jobs": -1
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for tr_idx, vl_idx in skf.split(X, y):
        Xtr, Xvl = X.iloc[tr_idx], X.iloc[vl_idx]
        ytr, yvl = y.iloc[tr_idx], y.iloc[vl_idx]

        model = lgb.LGBMClassifier(**params)

        model.fit(
        Xtr, ytr,
        eval_set=[(Xvl, yvl)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0)
    ]
)

        preds = model.predict_proba(Xvl)[:, 1]
        score = roc_auc_score(yvl, preds)
        scores.append(score)

    return np.mean(scores)

# ğŸ”¹ Ğ�Ğ¿Ñ‚Ğ¸Ğ¼Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ�
#study = optuna.create_study(direction="maximize")
#study.optimize(objective, n_trials=50)

#print("ğŸ�¯ Ğ›ÑƒÑ‡ÑˆĞ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹:")
#print(study.best_params)


xgb_pred=0
for model in models:
    xgb_pred += model.predict_proba(ts)[:, 1]
xgb_pred/=5

sub['y'] = xgb_pred
sub.to_csv('submission.csv')
sub.head(10)

