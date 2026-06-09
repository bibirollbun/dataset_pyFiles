import pandas as pd
import numpy as np

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_log_error

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from lightgbm import early_stopping

# === Load Data ===
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

# === Basic Preprocessing ===
train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})
test['Sex'] = test['Sex'].map({'male': 1, 'female': 0})

train = train.drop_duplicates(subset=train.columns).reset_index(drop=True)
train = train.groupby(['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])['Calories'].min().reset_index()

# === Feature Engineering ===
def add_features(df):
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['Intensity'] = df['Heart_Rate'] / df['Duration']
    df['Sex_Reversed'] = 1 - df['Sex']

    for f1 in ['Duration', 'Heart_Rate', 'Body_Temp']:
        for f2 in ['Sex', 'Sex_Reversed']:
            df[f'{f1}_x_{f2}'] = df[f1] * df[f2]

    for col in ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']:
        for agg in ['min', 'max', 'mean', 'std']:
            agg_val = train.groupby('Sex')[col].agg(agg).rename(f'Sex_{col}_{agg}')
            df = df.merge(agg_val, on='Sex', how='left')

    df.drop(columns=['Sex_Reversed'], inplace=True)
    return df

train = add_features(train)
test = add_features(test)

# === Align Columns ===
common_cols = [col for col in test.columns if col in train.columns and col != 'Calories']
X = train[common_cols]
y = np.log1p(train['Calories'])  # log1p for stability
X_test = test[common_cols]

# === Stratified KFold Setup (using Duration binning for stratification) ===
duration_bins = pd.qcut(X['Duration'], q=5, labels=False)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)




# === CatBoost ===
cat_params = {
    'iterations': 1500,
    'learning_rate': 0.01,
    'depth': 8,
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'loss_function': 'RMSE',
    'verbose': 0,
    'task_type': 'GPU'
}

cat_preds = np.zeros(len(X_test))
oof_cb = np.zeros(len(X))
for fold, (train_idx, val_idx) in enumerate(skf.split(X, duration_bins)):
    model = CatBoostRegressor(**cat_params)
    model.fit(X.iloc[train_idx], y.iloc[train_idx],
              eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
              use_best_model=True)
    oof_cb[val_idx] = model.predict(X.iloc[val_idx])
    cat_preds += model.predict(X_test) / skf.n_splits




# === XGBoost ===
xgb_params = {
    'n_estimators': 1500,
    'learning_rate': 0.01,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'gpu_hist'
}

xgb_preds = np.zeros(len(X_test))
oof_xgb = np.zeros(len(X))
for fold, (train_idx, val_idx) in enumerate(skf.split(X, duration_bins)):
    model = XGBRegressor(**xgb_params)
    model.fit(X.iloc[train_idx], y.iloc[train_idx],
              eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
              early_stopping_rounds=100, verbose=0)
    oof_xgb[val_idx] = model.predict(X.iloc[val_idx])
    xgb_preds += model.predict(X_test) / skf.n_splits




# === LightGBM ===
lgb_params = {
    'n_estimators': 1500,
    'learning_rate': 0.01,
    'max_depth': 9,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'regression',
    'random_state': 42,
    'device': 'gpu',
    'verbosity': -1
}

lgb_preds = np.zeros(len(X_test))
oof_lgb = np.zeros(len(X))
for fold, (train_idx, val_idx) in enumerate(skf.split(X, duration_bins)):
    model = LGBMRegressor(**lgb_params)
    model.fit(X.iloc[train_idx], y.iloc[train_idx],
              eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
              callbacks=[early_stopping(stopping_rounds=100)])
    oof_lgb[val_idx] = model.predict(X.iloc[val_idx])
    lgb_preds += model.predict(X_test) / skf.n_splits

# === Stacking with RidgeCV ===
stack_X = np.vstack([oof_cb, oof_xgb, oof_lgb]).T
stack_test = np.vstack([cat_preds, xgb_preds, lgb_preds]).T

ridge = RidgeCV(alphas=np.logspace(-3, 3, 7))
ridge.fit(stack_X, y)
final_log_preds = ridge.predict(stack_test)
final_preds = np.expm1(final_log_preds)  # Convert back from log scale

# === Clipping and Submission ===
final_preds = np.clip(final_preds, 1, 314)
sub = pd.DataFrame({
    'id': test['id'],
    'Calories': final_preds
})
sub.to_csv("submission.csv", index=False)






from sklearn.metrics import mean_squared_log_error

# Predict on training (OOF) set using Ridge stacker
stack_oof_preds = ridge.predict(stack_X)
rmsle = np.sqrt(mean_squared_log_error(y, stack_oof_preds))

print(f"Final Stacked RMSLE: {rmsle:.6f}")





