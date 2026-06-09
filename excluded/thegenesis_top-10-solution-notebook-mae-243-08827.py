# To ignore warnings:
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("coolwarm")

%matplotlib inline

from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

import optuna
from optuna.visualization import plot_param_importances


df_train = pd.read_csv("/kaggle/input/thapar-summer-school-2025-hack-ii/train.csv", index_col="id")
df_train.head()


df_test = pd.read_csv("/kaggle/input/thapar-summer-school-2025-hack-ii/test.csv", index_col="id")
df_test.head()


# Combining into one dataframe for easier preprocessing
df = pd.concat([df_train, df_test], axis=0)
df.iloc[14998:15003, :] # Printing the intersection between train and test sets


df.shape


# Checking if there is any nulls
df.drop("yield", axis=1).isnull().any().any()


# Checking duplicate rows
df.duplicated().any()


# Checking if id is unique
df.index.is_unique


df["Row#"].duplicated().any()


df["Row#"] = df["Row#"].astype(int)
df.info()


df.describe()


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.show()


df.drop("Row#", axis=1, inplace=True)
df.head()


UpperTRange = df[["MaxOfUpperTRange", "MinOfUpperTRange", "AverageOfUpperTRange"]].mean(axis=1).round(1)
LowerTRange = df[["MaxOfLowerTRange", "MinOfLowerTRange", "AverageOfLowerTRange"]].mean(axis=1).round(1)
RainingDiff = df["RainingDays"] - df["AverageRainingDays"]

df.insert(5, "UpperTRange", UpperTRange)
df.insert(6, "LowerTRange", LowerTRange)
df.insert(7, "RainingDiff", RainingDiff)

df.drop(columns=[
    "MaxOfUpperTRange", "MinOfUpperTRange", "AverageOfUpperTRange",
    "MaxOfLowerTRange", "MinOfLowerTRange", "AverageOfLowerTRange",
    "RainingDays", "AverageRainingDays",
], inplace=True)

df.head()


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.show()


sns.pairplot(df)
plt.show()


df_train = df[~(df["yield"].isna())]
df_test = df[df["yield"].isna()].drop("yield", axis=1)


X = df_train.drop("yield", axis=1)
y = df_train["yield"]


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


def objective_lgb(trial):
    model = LGBMRegressor(
        n_estimators=trial.suggest_int("n_estimators", 1000, 2500),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        reg_lambda=trial.suggest_float("reg_lambda", 1, 10),
        objective="mae",
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    return mean_absolute_error(y_valid, y_pred)


study_lgb = optuna.create_study(study_name="LGBM Hyperparameter Tuning", direction="minimize")
study_lgb.optimize(objective_lgb, n_trials=50)


lgb_best = LGBMRegressor(objective="MAE", random_state=42, **study_lgb.best_params, verbosity=-1)


df_train = df[~(df["yield"].isna())]
df_test = df[df["yield"].isna()].drop("yield", axis=1)


X = df_train.drop("yield", axis=1)
y = df_train["yield"]


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


def objective_cat(trial):
    model = CatBoostRegressor(
        iterations=trial.suggest_int("iterations", 1000, 2500),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 10),
        loss_function="MAE",
        verbose=0,
        random_state=42,
    )

    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100, verbose=0)
    y_pred = model.predict(X_valid)
    return mean_absolute_error(y_valid, y_pred)


study_cat = optuna.create_study(study_name="CatBoost Hyperparameter Tuning", direction="minimize")
study_cat.optimize(objective_cat, n_trials=50)


cat_best = CatBoostRegressor(loss_function="MAE", random_state=42, **study_cat.best_params, verbose=0)


lgb_best.fit(X_train, y_train)


cat_best.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100, verbose=0)


ensemble_preds = 0.7 * lgb_best.predict(X_valid) + 0.3 * cat_best.predict(X_valid)  # type: ignore
mae_ensemble = mean_absolute_error(y_valid, ensemble_preds)
print(f"MAE: {mae_ensemble:.6f}")


lgb_best.fit(X, y)
cat_best.fit(X, y, verbose=0)


final_preds = 0.5 * lgb_best.predict(df_test) + 0.5 * cat_best.predict(df_test)
final_preds


df_test["yield"] = final_preds
df_test["yield"].to_csv("submission.csv")
print("Submission file generated! Head on over to the sidebar and Submit to Competition!")

