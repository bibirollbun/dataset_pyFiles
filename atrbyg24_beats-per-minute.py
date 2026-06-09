import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import combinations

import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_squared_error
import optuna

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

%matplotlib inline


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv',index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv',index_col='id')


train.head()


train.info()


train.describe()


for col in train.columns:
    plt.figure(figsize=(8,6))
    sns.histplot(data=train,x=col)
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


for col in train.columns:
    plt.figure(figsize=(8,6))
    sns.boxplot(data=train,x=col)
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(12, 8))
sns.heatmap(
    train.corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Between Numerical Features", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


def create_features(df):
    new_df = df.copy()
    
    for col1, col2 in combinations(df.columns,2):
        if col1 != 'BeatsPerMinute' and col2 != 'BeatsPerMinute':
            new_df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
            new_df[f'{col1}_{col2}_ratio'] = df[col1]/(df[col2] + 1e-6)
    return new_df


train_processed = create_features(train)
test_processed = create_features(test)


X = train_processed.drop('BeatsPerMinute', axis=1).copy()
y = train_processed['BeatsPerMinute']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=17)


model = lgb.LGBMRegressor(objective = 'regression',random_state=42,verbose=-1)


model.fit(X_train,y_train)


y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_train,y_train)
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2: {r2}")


lgb.plot_importance(model, importance_type='split', figsize=(10, 40))
plt.title("Feature Importance (Split)")
plt.show()


lgb.plot_importance(model, importance_type='gain', figsize=(10, 40))
plt.title("Feature Importance (Gain)")
plt.show()


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "num_leaves": trial.suggest_int("num_leaves", 15, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "device": "gpu",           
        "gpu_platform_id": 0,      
        "gpu_device_id": 0         
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, valid_idx in kf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(5, verbose=False)],
        )
        
        preds = model.predict(X_valid, num_iteration=model.best_iteration_)
        rmse = mean_squared_error(y_valid, preds, squared=False)
        rmse_scores.append(rmse)
    
    return np.mean(rmse_scores)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best trial:")
print("  Value (mean RMSE):", study.best_trial.value)
print("  Params:", study.best_trial.params)


best_params = study.best_trial.params
best_params.update({
    "device": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "random_state": 42
})

best_lgb = lgb.LGBMRegressor(**best_params)
best_lgb.fit(X_train,y_train)


y_pred = best_lgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = best_lgb.score(X_train,y_train)
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2: {r2}")


submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
submission['BeatsPerMinute'] = best_lgb.predict(test_processed)
submission.to_csv("submission.csv", index=False)




