# Ğ”Ğ°Ğ½Ğ½Ñ‹Ğµ
import pandas as pd
import numpy as np

## ĞšĞ¾Ğ½Ñ„Ğ¸Ğ³
from dataclasses import dataclass, asdict

# ML
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

# Ğ�Ñ�Ñ‚Ğ°Ğ»ÑŒĞ½Ğ¾Ğµ
from pathlib import Path
from tqdm.auto import tqdm


@dataclass
class CFG:
    folder_path = Path("/kaggle/input/home-credit-default-risk")
    
    train_name = 'application_train.csv'
    test_name = 'application_test.csv'
    subm_name = 'sample_submission.csv'

    debug = True

    seed = 2025


cfg = CFG()


train_df = pd.read_csv(str(cfg.folder_path / cfg.train_name))
test_df = pd.read_csv(str(cfg.folder_path / cfg.test_name))


train_df.head()


train_df.shape, test_df.shape


train_df.TARGET.value_counts()


train_df.isna().sum()


test_df.isna().sum()


if cfg.debug:
    train_df = train_df[: 30_000]


X = train_df.drop(columns='TARGET')
y = train_df['TARGET']


cat_features = X.select_dtypes('object').columns.tolist()


X[cat_features].isna().sum()


X[cat_features] = X[cat_features].astype(str)


cv = StratifiedKFold(n_splits=3)


models_list = []
metrics_list = []

for train_idx, valid_idx in (bar := tqdm(cv.split(X, y), total=cv.get_n_splits())):
    X_train, X_valid = X.loc[train_idx, :], X.loc[valid_idx, :]
    y_train, y_valid = y[train_idx], y[valid_idx]

    model = CatBoostClassifier(
        learning_rate=0.2,
        eval_metric='AUC',
        random_state=cfg.seed
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        cat_features=cat_features,
        early_stopping_rounds=30,
        verbose=100
    )

    y_pred = model.predict_proba(X_valid)[:, 1]
    metric = roc_auc_score(y_valid, y_pred)

    models_list.append(model)
    metrics_list.append(metric)

    bar.set_postfix_str(f"ROC-AUC: {metric:.5f}")

print(f"All metrics: {metrics_list}")
print(f"Corrected AUC: {np.mean(metrics_list) - np.std(metrics_list)}")


test_df = pd.read_csv(str(cfg.folder_path / cfg.test_name))


test_df[cat_features] = test_df[cat_features].astype(str)


weights = np.array(metrics_list)
weights = weights / weights.sum()
weights


y_pred = np.zeros(test_df.shape[0])
# Ğ£Ñ‡Ğ¸Ñ‚Ñ‹Ğ²Ğ°ĞµĞ¼ Ğ²ĞµÑ�Ğ° ĞºĞ°Ğ¶Ğ´Ğ¾Ğ³Ğ¾ Ğ°Ğ»Ğ³Ğ¾Ñ€Ğ¸Ñ‚Ğ¼Ğ° Ğ¿Ğ¾ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞµ
for model, weight in zip(models_list, weights):
    y_pred += weight * model.predict_proba(test_df)[:, 1]
y_pred


subm = pd.read_csv(str(cfg.folder_path / cfg.subm_name))
subm.head()


subm['TARGET'] = y_pred
subm.to_csv('submission.csv', index=False)


subm.head()




