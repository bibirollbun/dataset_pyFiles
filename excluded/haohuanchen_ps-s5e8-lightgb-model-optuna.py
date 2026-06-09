import pandas as pd

train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


# Add new features (More attempts can be made)
import numpy as np

def add_features(df):
    df = df.copy()
    
    def credit(x):
        credit_level = (x['default'] == 'no') + (x['housing'] == 'no') + (x['loan'] == 'no')
        return {3: 27, 2: 9, 1: 3}.get(credit_level, 0)
    df['credit'] = df.apply(lambda x: credit(x), axis=1)
    
    def risk(x):
        risk_level = (x['default'] == 'yes') + (x['housing'] == 'yes') + (x['loan'] == 'yes')
        return {3: 27, 2: 9, 1: 3}.get(risk_level, 0)
    df['risk'] = df.apply(lambda x: risk(x), axis=1)
    
    df['balance_log'] = df['balance'].where(df['balance'] > -1, 0).apply(np.log1p)
    df['duration_log'] = df['duration'].where(df['duration'] > -1, 0).apply(np.log1p)
    
    month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6, 'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
    df['month'] = df['month'].map(month_map)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['age_bin'] = pd.cut(df['age'], bins=[0, 40, 70, 100], labels=[1, 2, 0], right=False)
    
    return df

train_df = add_features(train_df)
test_df = add_features(test_df)


cat_cols = train_df.select_dtypes(include="object").columns.tolist()
cat_cols


from collections import Counter

for col in cat_cols:
    counts = train_df[col].value_counts()
    train_df[col + "_count"] = train_df[col].map(counts)
    test_df[col + "_count"] = test_df[col].map(counts).fillna(0)


X = train_df.drop(["y", "id"], axis=1)
y = train_df["y"]
X_test = test_df.drop(["id"], axis=1)


from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])


def add_target_encoding(X_train, y_train, X_val):
    X_train_te = X_train.copy()
    X_val_te = X_val.copy()
    
    for col in cat_cols:
        mapping = (
            pd.DataFrame({col: X_train[col], "y": y_train})
            .groupby(col)["y"]
            .mean()
        )
        new_col = f"{col}_te"
        X_train_te[new_col] = X[col].map(mapping)
        X_val_te[new_col] = X_val[col].map(mapping)
    
    return X_train_te, X_val_te


import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

def objective(trial):
    param = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 20000,
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.05),
        "num_leaves": trial.suggest_int("num_leaves", 110, 130),
        "max_depth": trial.suggest_int("max_depth", 14, 18),
        "min_child_samples": trial.suggest_int("min_child_samples", 4, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.4),
        "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 2.0),
        "max_bin": trial.suggest_int("max_bin", 8000, 9000),
        "random_state": 42,
        "verbosity": -1,
    }

    val_scores = []
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train, X_val = add_target_encoding(X_train, y_train, X_val)
        
        model = lgb.LGBMClassifier(**param)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(500)]
        )
        
        val_pred = model.predict_proba(X_val)[:, 1]
        val_score = roc_auc_score(y_val, val_pred)
        val_scores.append(val_score)

    return np.mean(val_scores)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, timeout=30600)


print("Best trial:")
trial = study.best_trial

print(f"  AUC: {trial.value:.6f}")
print("  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

