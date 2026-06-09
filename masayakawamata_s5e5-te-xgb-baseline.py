import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.simplefilter('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
original = pd.read_csv('/kaggle/input/d/ruchikakumbhar/calories-burnt-prediction/calories.csv')
print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
print("Original Shape", original.shape)
display(train.head(3))
display(original.head(3))


original.drop(columns=['User_ID'], inplace=True)
original.rename(columns={'Gender': 'Sex'}, inplace=True)
train = pd.concat([train, original], ignore_index=True)
train.shape


TARGET = "Calories"
FEATURES = [col for col in train.columns if col not in ['id', TARGET]]


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import gc

encode_stats = ['std']


def target_encode(df_train, df_val, col, target, stats='mean', prefix='TE'):
    df_val = df_val.copy()
    agg = df_train.groupby(col)[target].agg(stats)    
    if isinstance(stats, (list, tuple)):
        for s in stats:
            colname = f"{prefix}_{col}_{s}"
            df_val[colname] = df_val[col].map(agg[s]).astype(float)
            df_val[colname].fillna(agg[s].mean(), inplace=True)
    else:
        suffix = stats if isinstance(stats, str) else stats.__name__
        colname = f"{prefix}_{col}_{suffix}"
        df_val[colname] = df_val[col].map(agg).astype(float)
        df_val[colname].fillna(agg.mean(), inplace=True)
    return df_val


train['Sex'] = train['Sex'].astype('category')
test['Sex'] = test['Sex'].astype('category')


FOLDS = 5
outer_kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (tr_idx, vl_idx) in enumerate(outer_kf.split(train), 1):
    print(f"--- Fold {fold} / {FOLDS} ---")
    
    X_tr_raw = train.loc[tr_idx, FEATURES].reset_index(drop=True)
    y_tr_raw = train.loc[tr_idx, TARGET].reset_index(drop=True)
    X_vl_raw = train.loc[vl_idx, FEATURES].reset_index(drop=True)
    y_vl_raw = train.loc[vl_idx, TARGET].reset_index(drop=True)
    X_ts_raw = test[FEATURES].copy()
    
    y_tr = np.log1p(y_tr_raw)
    
    X_tr, X_vl, X_ts = X_tr_raw.copy(), X_vl_raw.copy(), X_ts_raw.copy()
    inner_kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    for _, (in_tr_idx, in_vl_idx) in enumerate(inner_kf.split(X_tr_raw), 1):
        in_tr = pd.concat([X_tr_raw.loc[in_tr_idx], y_tr.loc[in_tr_idx]], axis=1)
        in_vl = X_tr_raw.loc[in_vl_idx].reset_index(drop=True)
        for col in FEATURES:
            for stat in encode_stats:
                te_tmp = target_encode(
                    in_tr, in_vl.copy(),
                    col, TARGET,
                    stats=stat, prefix="TE"
                )
                te_col = f"TE_{col}_{stat}"
                X_tr.loc[in_vl_idx, te_col] = te_tmp[te_col].values

    tr_with_y = pd.concat([X_tr_raw, pd.Series(y_tr, name=TARGET)], axis=1)
    for col in FEATURES:
        for stat in encode_stats:
            X_vl = target_encode(tr_with_y, X_vl, col, TARGET, stats=stat, prefix="TE")
            X_ts = target_encode(tr_with_y, X_ts, col, TARGET, stats=stat, prefix="TE")
    
    model = XGBRegressor(
        device='cuda',
        max_depth=6,
        colsample_bytree=0.8,
        subsample=0.8,
        n_estimators=50_000,
        learning_rate=0.01,
        enable_categorical=True,
        early_stopping_rounds=500,
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, np.log1p(y_vl_raw))],
        verbose=2000
    )

    vl_pred_log = model.predict(X_vl)
    vl_pred = np.expm1(vl_pred_log)
    oof_preds[vl_idx] = vl_pred
    
    ts_pred_log = model.predict(X_ts)
    test_preds += np.expm1(ts_pred_log)
    
    fold_mse = mean_squared_error(
        np.log1p(y_vl_raw),
        np.log1p(vl_pred),
        squared=True
    )
    fold_rmsle = np.sqrt(fold_mse)
    print(f"Fold {fold} RMSLE: {fold_rmsle:.5f}\n")
    
    del X_tr_raw, X_vl_raw, X_ts_raw, X_tr, X_vl, X_ts, y_tr, y_vl_raw
    if fold != FOLDS:
        del model
    gc.collect()

test_preds /= FOLDS

overall_mse = mean_squared_error(
    np.log1p(train[TARGET]),
    np.log1p(oof_preds),
    squared=True
)
overall_rmsle = np.sqrt(overall_mse)
print(f"Final OOF RMSLE (XGB): {overall_rmsle:.5f}")


importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']

booster = model.get_booster()

for itype in importance_types:
    score = booster.get_score(importance_type=itype)
    score_series = pd.Series(score).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    score_series.head(30).plot(kind='bar')
    plt.title(f"Feature Importance - {itype}")
    plt.ylabel(itype)
    plt.xlabel("Features")
    plt.tight_layout()
    plt.show()


test_preds = np.clip(test_preds,1,314)


sub = pd.read_csv('//kaggle/input/playground-series-s5e5/sample_submission.csv')
sub[TARGET] = test_preds
sub.to_csv('submission.csv', index=False)
sub.head(3)


import pickle

with open(f'test_pred.pkl', 'wb') as f:
    pickle.dump(test_preds, f)

with open(f'oof_pred.pkl', 'wb') as f:
    pickle.dump(oof_preds, f)

