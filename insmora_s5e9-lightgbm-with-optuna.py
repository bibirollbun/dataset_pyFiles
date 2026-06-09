import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv", index_col="id")
test  = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv", index_col="id")



train.head()


train.info()


train.describe()


import matplotlib.pyplot as plt
# histograms for each column
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
train.hist(ax=axes, edgecolor='black', grid=False)
plt.tight_layout()
plt.show()



X = train.drop(columns=["BeatsPerMinute"])
y = train["BeatsPerMinute"]

# train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_valid, y_valid)

# ------------------------------
# Optuna objective function
# ------------------------------
def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": 42,
        "feature_pre_filter": False,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", -1, 16),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_valid],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    preds = model.predict(X_valid, num_iteration=model.best_iteration)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    trial.set_user_attr("best_iteration", model.best_iteration)
    return rmse

# ---------------------------------
# Optuna study
# ------------------------------------
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best RMSE:", study.best_value)
print("Best params:", study.best_params)

# ------------------------------------
# - final model on full dataset with best params
# -------------------------------------------
best_params = study.best_params
best_params.update({
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "seed": 42,
    "feature_pre_filter": False
})

final_dataset = lgb.Dataset(X, y)
final_model = lgb.train(
    best_params,
    final_dataset,
    num_boost_round=study.best_trial.user_attrs.get("best_iteration", 1000)
)

# -----------------------------
# prediction on test set
# ---------------------------
y_test_pred = final_model.predict(test)

submission = pd.DataFrame({
    "id": test.index,
    "BeatsPerMinute": y_test_pred
})

submission.to_csv("submission_lightgbm_optuna.csv", index=False)



