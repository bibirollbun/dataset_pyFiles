import numpy as np 
import pandas as pd 
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


train.info()


num_cols = ['age','balance','day','duration','campaign','pdays','previous']
cat_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome']
target_col = 'y'
id_col = 'id'


# preprocess func
def preprocess(df):
    for col in num_cols:
        q_low = df[col].quantile(0.005)
        q_high = df[col].quantile(0.995)
        df[col] = df[col].clip(q_low, q_high)

    df['duration_log'] = np.log(df['duration'] + 1)
    df['campaign_log'] = np.log(df['campaign'] + 1)
    df['pdays_log'] = np.log(df['pdays'] + 2)
    df['previous_log'] = np.log(df['previous'] + 1)
    
    return df


def label_encode(df_train, df_test, cat_cols): # for lightgbm only
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([df_train[col], df_test[col]], axis=0).astype(str)
        le.fit(combined)
        df_train[col] = le.transform(df_train[col].astype(str))
        df_test[col] = le.transform(df_test[col].astype(str))
    return df_train, df_test


train = preprocess(train)
test = preprocess(test)

X = train.drop(columns=[id_col, target_col])
y = train[target_col]
X_test = test.drop(columns=[id_col])


# encoding data for lightgbm
X_lgb, X_test_lgb = X.copy(), X_test.copy()

X_lgb[cat_cols] = X_lgb[cat_cols].astype(str)
X_test_lgb[cat_cols] = X_test_lgb[cat_cols].astype(str)
X_lgb, X_test_lgb = label_encode(X_lgb, X_test_lgb, cat_cols)


FOLDS = 5
SEED = 42


# training both models and then averaging them
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nğŸ”� Fold {fold + 1}")
    # Split for LGBM
    X_train_lgb, X_val_lgb = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]
    # Split for CatBoost
    X_train_cb, X_val_cb = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        learning_rate=0.05,
        n_estimators=2000,
        random_state=SEED
    )
    lgb_model.fit(
        X_train_lgb, y_train,
        eval_set=[(X_val_lgb, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
    )
    lgb_val = lgb_model.predict_proba(X_val_lgb)[:, 1]
    lgb_test = lgb_model.predict_proba(X_test_lgb)[:, 1]

    # CatBoost
    cat_idxs = [X.columns.get_loc(c) for c in cat_cols]
    cb_model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        random_seed=SEED,
        cat_features=cat_idxs,
        verbose=False,
        early_stopping_rounds=100,
        eval_metric='AUC'
    )
    cb_model.fit(X_train_cb, y_train, eval_set=(X_val_cb, y_val), use_best_model=True)
    cb_val = cb_model.predict_proba(X_val_cb)[:, 1]
    cb_test = cb_model.predict_proba(X_test)[:, 1]

    # Average predictions
    oof_preds[val_idx] = (lgb_val + cb_val) / 2
    test_preds += (lgb_test + cb_test) / 2 / FOLDS


val_roc_auc = roc_auc_score(y, oof_preds)
print(f"\n Overall ROC AUC from K-Fold Ensemble: {val_roc_auc:.5f}")


submission['y'] = test_preds
submission.to_csv("submission.csv", index=False)


submission.head(10)

