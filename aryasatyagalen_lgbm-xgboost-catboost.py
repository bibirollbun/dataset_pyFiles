import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


!nvidia-smi
!nvcc --version


!pip install scikit-learn==1.6.1 xgboost==3.0.2 lightgbm==4.6.0 catboost==1.2.8 optuna==4.5.0


import pandas as pd
import numpy as np
import optuna
import sklearn.metrics 
from sklearn.ensemble import StackingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import math


train = pd.DataFrame(pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv'))
test = pd.DataFrame(pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv'))
X_train = train.drop(columns=['id', 'BeatsPerMinute'], axis=1)
y_train = train['BeatsPerMinute']
X_test = test.drop(columns=['id'], axis=1)


display(X_train)


numeric_df = X_train.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Matrix", fontsize=16)
plt.tight_layout()
plt.show()


X_train["AcousticVsEnergy"] = X_train["AcousticQuality"] / (X_train["Energy"] + 1e-6)
X_train["LiveVsEnergy"] = X_train["LivePerformanceLikelihood"] / (X_train["Energy"] + 1e-6)
X_train["MoodEnergyDiff"] = X_train["MoodScore"] - X_train["Energy"]
X_test["AcousticVsEnergy"] = X_test["AcousticQuality"] / (X_test["Energy"] + 1e-6)
X_test["LiveVsEnergy"] = X_test["LivePerformanceLikelihood"] / (X_test["Energy"] + 1e-6)
X_test["MoodEnergyDiff"] = X_test["MoodScore"] - X_test["Energy"]


numeric_df = X_train.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Matrix (after feature engineering)", fontsize=16)
plt.tight_layout()
plt.show()


scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

kf = KFold(n_splits=5, shuffle=True, random_state=26)


def objective(trial):
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 200, 800),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 8),
        "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample", 0.6, 1.0),
        "objective": 'reg:squarederror',
        "eval_metric": 'rmse',
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 26
    }

    lgbm_params = {
        "n_estimators": trial.suggest_int("lgbm_n_estimators", 200, 800),
        "num_leaves": trial.suggest_int("lgbm_num_leaves", 31, 127),
        "learning_rate": trial.suggest_float("lgbm_lr", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("lgbm_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("lgbm_colsample", 0.6, 1.0),
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1,
        'device': 'gpu',
        'random_state': 26
    }

    cat_params = {
        "iterations": trial.suggest_int("cat_iterations", 200, 800),
        "depth": trial.suggest_int("cat_depth", 4, 8),
        "learning_rate": trial.suggest_float("cat_lr", 0.01, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("cat_l2", 1.0, 5.0),
        "devices": "0",
        "verbose": 0,
        'loss_function': 'RMSE',
        'task_type': 'GPU',
        'random_seed': 26
    }


    estimators = [
        ("xgb", XGBRegressor(**xgb_params)),
        ("lgb", LGBMRegressor(**lgbm_params)),
        ("cat", CatBoostRegressor(**cat_params))
    ]

    final_estimator = Ridge(alpha=trial.suggest_float("ridge_alpha", 0.1, 10.0, log=True))

    stack = Pipeline([
        ("scaler", StandardScaler()),
        ("stack", StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=1,
           
        ))
    ])

    score = cross_val_score(stack, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=1)
    return -np.mean(score)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

print("Best params:", study.best_params)
print("Best RMSE:", study.best_value)


best_params = study.best_params

xgb_params = {
    "n_estimators": best_params["xgb_n_estimators"],
    "max_depth": best_params["xgb_max_depth"],
    "learning_rate": best_params["xgb_lr"],
    "subsample": best_params["xgb_subsample"],
    "colsample_bytree": best_params["xgb_colsample"],
    "random_state" : 26
}

lgbm_params = {
    "n_estimators": best_params["lgbm_n_estimators"],
    "num_leaves": best_params["lgbm_num_leaves"],
    "learning_rate": best_params["lgbm_lr"],
    "subsample": best_params["lgbm_subsample"],
    "colsample_bytree": best_params["lgbm_colsample"],
    "random_state" : 26
}

cat_params = {
    "iterations": best_params["cat_iterations"],
    "depth": best_params["cat_depth"],
    "learning_rate": best_params["cat_lr"],
    "l2_leaf_reg": best_params["cat_l2"],
    "verbose": 0,
    "random_seed" : 26
}

estimators = [
    ("xgb", XGBRegressor(**xgb_params)),
    ("lgb", LGBMRegressor(**lgbm_params)),
    ("cat", CatBoostRegressor(**cat_params))
]

final_estimator = Ridge(alpha=best_params["ridge_alpha"])

stack = Pipeline([
    ("scaler", StandardScaler()),
    ("stack", StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=kf,
        n_jobs=-1
    ))
])

stack.fit(X_train, y_train)


test_preds = stack.predict(X_test)
submission = pd.DataFrame({
    "id": test['id'],  # or the column name Kaggle requires (check sample_submission.csv)
    "y": test_preds
})


submission.to_csv('submission_feature_engineered.csv', index=False)
print("Submission file created")

