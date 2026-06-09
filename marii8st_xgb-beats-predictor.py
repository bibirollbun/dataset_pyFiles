import seaborn as sns
import numpy as np
import polars as pl
import optuna
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler


from itertools import combinations
from typing import List


train_df = pl.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pl.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


train_df


def loging_df(df: pl.DataFrame, bad_cols: List) -> pl.DataFrame:
    for col in bad_cols:
        df = df.with_columns(pl.col(col).log1p())

    return df

def windsorizing_df(df: pl.DataFrame, bad_cols: List, low=0.01, high=0.99) -> pl.DataFrame:
    for col in bad_cols:
        lo = df[col].quantile(low)
        hi = df[col].quantile(high)
        
        df = df.with_columns(
            pl.col(col).clip(lower_bound=lo, upper_bound=hi).alias(col)
        )
    
    return df

def robust_scaling(df: pl.DataFrame, bad_cols: List) -> pl.DataFrame:
    scaler = RobustScaler()
    
    for col in bad_cols:
        values = df[col].to_numpy().reshape(-1, 1)
        
        scaled = scaler.fit_transform(values).flatten()
        df = df.with_columns(
            pl.Series(name=col, values=scaled)
        )
    return df


bad_cols = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore']

train_df = loging_df(train_df, bad_cols)
train_df = windsorizing_df(train_df, bad_cols)
train_df = robust_scaling(train_df, bad_cols)

test_df = loging_df(test_df, bad_cols)
test_df = windsorizing_df(test_df, bad_cols)
test_df = robust_scaling(test_df, bad_cols)


features_to_combine = ['RhythmScore', 'AudioLoudness', 'AcousticQuality', 'MoodScore']

for col1, col2 in combinations(features_to_combine, 2):
    train_df = train_df.with_columns(
        (pl.col(col1) * pl.col(col2)).alias(f'{col1}_x_{col2}'),
        (pl.col(col1) / (pl.col(col2) + 1e-6)).alias(f'{col1}_div_{col2}')
    )
    test_df = test_df.with_columns(
        (pl.col(col1) * pl.col(col2)).alias(f'{col1}_x_{col2}'),
        (pl.col(col1) / (pl.col(col2) + 1e-6)).alias(f'{col1}_div_{col2}')
    )


X = train_df.drop(["id", "BeatsPerMinute"])
y = train_df["BeatsPerMinute"]
X_test = test_df.drop("id")

FOLDS = 20
FEATURES = X.columns

# KFold setup
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Arrays to store predictions
oof = np.zeros(len(train_df))
pred = np.zeros(len(test_df))

X_np = X.to_numpy()
y_np = y.to_numpy()
X_test_np = X_test.to_numpy()


def objective(trial):
    params = {
        "device": "cpu",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "gamma": trial.suggest_float("gamma", 0.0, 20.0),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "eval_metric": "rmse",
        "enable_categorical": True,
        "tree_method": "hist",
        "random_state": 42
    }
    
    inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_rmses = []
    
    for train_idx, val_idx in inner_kf.split(X_np):
        x_tr, y_tr = X_np[train_idx], y_np[train_idx]
        x_val, y_val = X_np[val_idx], y_np[val_idx]
        
        early_stopping = EarlyStopping(
            rounds=100,
            min_delta=1e-5,
            save_best=True
        )
        
        model = XGBRegressor(**params, callbacks=[early_stopping])
        model.fit(
            x_tr, y_tr,
            eval_set=[(x_val, y_val)],
            verbose=False
        )
        
        val_pred = model.predict(x_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        inner_rmses.append(rmse)
    
    return np.mean(inner_rmses)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_params.update({
    "device": "cpu",
    "eval_metric": "rmse",
    "enable_categorical": True,
    "tree_method": "hist",
    "random_state": 42
})

print("Best params from Optuna:", best_params)

for i, (train_indices, valid_indices) in enumerate(kf.split(X_np, y_np)):
    print(f"\n{'#'*5} Fold {i+1} {'#'*5}")
    x_train = X_np[train_indices]
    y_train = y_np[train_indices]
    x_valid = X_np[valid_indices]
    y_valid = y_np[valid_indices]
    early_stopping = EarlyStopping(
        rounds=100,
        min_delta=1e-5,
        save_best=True
    )
    model = XGBRegressor(**best_params, callbacks=[early_stopping])
   
    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=100
    )
   
    oof[valid_indices] = model.predict(x_valid)
    pred += model.predict(X_test_np)
   
    rmse = np.sqrt(mean_squared_error(y_valid, oof[valid_indices]))
    print(f"Fold {i+1} RMSE: {rmse:.4f}")
   
pred /= FOLDS
       
full_rmse = np.sqrt(mean_squared_error(y_np, oof))
print(f"\nFinal CV RMSE: {full_rmse:.4f}")


submission = pl.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

print('predict mean :',pred.mean())
print('predict median :',np.median(pred))

y_pred_after = np.clip(pred, 46.718, 206.037)

submission = submission.with_columns(
    pl.Series(name="BeatsPerMinute", values=y_pred_after)
)

submission.write_csv("submission.csv")
submission.head()

