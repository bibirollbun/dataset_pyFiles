import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


print(train_df.isnull().sum())
print(test_df.isnull().sum())


X = train_df.drop(["BeatsPerMinute", "id"], axis=1)
y = train_df["BeatsPerMinute"]
X_test = test_df.drop(["id"], axis=1)


num_cols = X.columns.tolist()
num_cols


pd.set_option("display.float_format", "{:.6f}".format)
def check_transforms(X, y, num_cols, skew_threshold=1.0):
    results = []
    for col in num_cols:
        col_data = X[col]
        
        if col_data.nunique() <= 1:
            continue

        orig_corr = col_data.corr(y)
        
        log_corr = np.nan
        if (col_data >= 0).all():
            try:
                log_corr = np.log1p(col_data).corr(y)
            except Exception:
                log_corr = np.nan

        sqrt_corr = np.nan
        if (col_data >= 0).all():
            sqrt_corr = np.sqrt(col_data).corr(y)

        sq_corr = np.square(col_data).corr(y)
        
        candidates = {
            "orig": orig_corr,
            "log1p": log_corr,
            "sqrt": sqrt_corr,
            "square": sq_corr
        }
        best = max(candidates.items(), key=lambda x: abs(x[1]) if pd.notna(x[1]) else -1)
        
        results.append({
            "feature": col,
            "skew": col_data.skew(),
            "orig_corr": orig_corr,
            "log1p_corr": log_corr,
            "sqrt_corr": sqrt_corr,
            "square_corr": sq_corr,
            "best": best[0],
            "best_corr": best[1]
        })
    
    return pd.DataFrame(results).sort_values(by="best_corr", ascending=False)

transform_check = check_transforms(X, y, num_cols)
transform_check


import itertools

def add_combinations(df, c1, c2):
    df = df.copy()
    df[f"{c1}_plus_{c2}"] = df[c1] + df[c2]
    df[f"{c1}_minus_{c2}"] = df[c1] - df[c2]
    df[f"{c2}_minus_{c1}"] = df[c2] - df[c1]
    df[f"{c1}_times_{c2}"] = df[c1] * df[c2]
    df[f"{c1}_div_{c2}"] = df[c1] / (df[c2] + 1e-6)
    df[f"{c2}_div_{c1}"] = df[c2] / (df[c1] + 1e-6)
    return df

def add_transform(df):
    df = df.copy()
    # sqrt
    df["RhythmScore_sqrt"] = np.sqrt(df["RhythmScore"])
    df["LivePerformanceLikelihood_sqrt"] = np.sqrt(df["LivePerformanceLikelihood"])
    df["InstrumentalScore_sqrt"] = np.sqrt(df["InstrumentalScore"])
    # square
    df["MoodScore_square"] = np.square(df["MoodScore"])
    df["VocalContent_square"] = np.square(df["VocalContent"])
    df["AcousticQuality_square"] = np.square(df["AcousticQuality"])
    # log1p
    df["Energy_log1p"] = np.log1p(df["Energy"])
    
    return df

for df in [X, X_test]:
    # combinations
    for c1, c2 in itertools.combinations(num_cols, 2):
        df = add_combinations(df, c1, c2)
    # transform
    df = add_transform(df)


import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

y_preds = np.zeros(len(X_test))
models = []
val_rmses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training fold {fold + 1}/{n_splits} >>>")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=20000,
        learning_rate=0.001,
        num_leaves=100,
        max_depth=10,
        min_child_samples=10,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=2.0,
        reg_lambda=1.5,
        random_state=42,
        verbosity=-1,
        boosting_type='gbdt',
        metric='rmse'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(period=100)
        ]
    )
    
    models.append(model)
    
    y_preds += model.predict(X_test) / n_splits
    
    val_pred = model.predict(X_val)
    val_rmse = mean_squared_error(y_val, val_pred, squared=False)
    val_rmses.append(val_rmse)

print(f"Mean RMSE: {np.mean(val_rmses):.6f}")


all_importances = []

for model in models:
    all_importances.append(model.feature_importances_)

avg_importances = pd.Series(np.mean(all_importances, axis=0), index=X.columns)
avg_importances.sort_values().tail(25).plot(kind='barh')


submission = pd.DataFrame({"id": test_df["id"], "BeatsPerMinute": y_preds})
submission.to_csv("submission.csv", index=False)
submission.head()

