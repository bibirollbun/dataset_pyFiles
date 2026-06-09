import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import catboost
from catboost import CatBoostRegressor, Pool
import optuna
import warnings
warnings.filterwarnings("ignore")

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
test_ids = test['id']

train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)


def preprocess(df):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['HR/Temp'] = df['Heart_Rate'] / df['Body_Temp']
    df['HR*Temp'] = df['Heart_Rate'] * df['Body_Temp']
    df['Duration*Age'] = df['Duration'] * df['Age']
    df['HR/Weight'] = df['Heart_Rate'] / df['Weight']
    df['Age^2'] = df['Age'] ** 2
    df['Log_Weight'] = np.log1p(df['Weight'])
    return df

train = preprocess(train)
test = preprocess(test)

X = train.drop("Calories", axis=1)
y = train["Calories"]
X_test = test


y_log = np.log1p(y)


def objective(trial):
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("lr", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 1, 10),
        "loss_function": "RMSE",
        "eval_metric": "MAE",
        "verbose": 0,
        "early_stopping_rounds": 50
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        preds = model.predict(X_val)
        score = mean_absolute_error(np.expm1(y_val), np.expm1(preds))
        scores.append(score)

    return np.mean(scores)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=25)

best_params = study.best_params
print("Best Hyperparameters:", best_params)


final_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=best_params['lr'],
    depth=best_params['depth'],
    l2_leaf_reg=best_params['l2_leaf_reg'],
    random_strength=best_params['random_strength'],
    loss_function="RMSE",
    eval_metric="MAE",
    verbose=0,
    early_stopping_rounds=50
)

final_model.fit(X, y_log)
final_preds = np.expm1(final_model.predict(X_test))


submission = pd.DataFrame({
    "id": test_ids,
    "Calories": final_preds
})
submission.to_csv("submission.csv", index=False)
print("✅ Saved: best_optuna_catboost_submission.csv")

