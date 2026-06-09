import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")

df.drop(columns=["id"], inplace=True)
df["Sex"] = df["Sex"].map({"female": 0, "male": 1})

df.head(5)


sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Pearson Correlation Heatmap")
plt.show()


df["Intensity"] = df["Heart_Rate"] / df["Duration"]
df["Effort"] = df["Heart_Rate"] * df["Duration"]
df["Body_Temp"] = df["Heart_Rate"] * df["Body_Temp"]
df["HR_pct_max"] = df["Heart_Rate"] / (220 - df["Age"])

df["HR_pct_max_Effort_Index"] = df["HR_pct_max"] * df["Effort"]
df["Heart_Rate_Effort_Index"] = df["Heart_Rate"] * df["Effort"]
df["Body_Temp_Effort_Index"] = df["Body_Temp"] * df["Effort"]


import optuna
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

X = df.copy()
X.drop(columns=["Calories"], inplace=True)

y = df["Calories"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "verbosity": 0,
    }

    model = XGBRegressor(**params)

    model.fit(
        X_train,
        y_train,
        verbose=False
    )

    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds, squared=False)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10, show_progress_bar=True)


print("Best Parameters:")
for param, val in study.best_params.items():
    print(f"{param}: {val}")

best_model = XGBRegressor(**study.best_params)
best_model.fit(X_train, y_train)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE:{rmse:.2f}")
print(f"R^2: {r2:.4f}")


eval_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

df = eval_df.drop(columns=["id"])
df["Sex"] = df["Sex"].map({"female"
                           : 0, "male": 1})
df["Intensity"] = df["Heart_Rate"] / df["Duration"]
df["Effort"] = df["Heart_Rate"] * df["Duration"]
df["Body_Temp"] = df["Heart_Rate"] * df["Body_Temp"]
df["HR_pct_max"] = df["Heart_Rate"] / (220 - df["Age"])

df["HR_pct_max_Effort_Index"] = df["HR_pct_max"] * df["Effort"]
df["Heart_Rate_Effort_Index"] = df["Heart_Rate"] * df["Effort"]
df["Body_Temp_Effort_Index"] = df["Body_Temp"] * df["Effort"]

predictions = best_model.predict(df)

submission = pd.DataFrame({
    "id": eval_df["id"],
    "Calories": predictions
})

submission.to_csv("submission.csv", index=False)

