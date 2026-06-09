import warnings
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import datetime as dt
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    RepeatedKFold,
    cross_val_score,
    train_test_split,
)

import shap
warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use("fast")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv", parse_dates=['date']).drop(columns = "id")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv", parse_dates=['date']).drop(columns = "id")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


train_df.isna().sum()


test_df.isna().sum()


def exracting_date(df):
    df["Weekday_sv"] = df["date"].dt.strftime("%a").astype("category")
    df["Weekday_num"] = df["date"].dt.strftime("%w").astype("int")
    df["Day_of_month"] = df["date"].dt.strftime("%d").astype("category")
    df["Month_name_sv"] = df["date"].dt.strftime("%b").astype("category")
    df["Month_num"] = df["date"].dt.strftime("%m").astype("int")
    df["Year_fv"] = df["date"].dt.strftime("%Y").astype("int")
    df["Day_number_year"] = df["date"].dt.strftime("%j").astype("int")
    df["Week_number_year"] = df["date"].dt.strftime("%W").astype("int")
    df["country"] = df["country"].astype("category")
    df["store"] = df["store"].astype("category")
    df["product"] = df["product"].astype("category")
    df['year_sin'] = np.sin(2 * np.pi * df['Year_fv'])
    df['year_cos'] = np.cos(2 * np.pi * df['Year_fv'])
    df['month_sin'] = np.sin(2 * np.pi * df['Month_num'] / 12) 
    df['month_cos'] = np.cos(2 * np.pi * df['Month_num'] / 12) 
    return df


train_df = exracting_date(train_df)
train_df = train_df.dropna()


test_df = exracting_date(test_df)
test_df = test_df.dropna()


train_df.describe().T


def plot_feature_importance(model, feature_names, top_n=10):
    # Extract feature importance values
    importance = model.feature_importances_
    
    # Create a DataFrame for easy sorting and visualization
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })
    
    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False).head(top_n)
    
    # Plot the feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color="skyblue")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Top {} Feature Importances".format(top_n))
    plt.gca().invert_yaxis()  # Invert y-axis to show the most important features on top
    plt.tight_layout()
    plt.show()

# features and targets
feature_columns = [
    "Weekday_num", "Month_num", "Year_fv", "Day_number_year", 
    "Week_number_year", "year_sin", "year_cos", "month_sin", "month_cos"
]
X = train_df[feature_columns]  # Model girdileri
y = train_df["num_sold"]      # Hedef değişken

model = LGBMRegressor(random_state=42)
model.fit(X, y)

plot_feature_importance(model, feature_columns)


train_df = train_df.drop(columns=['year_cos', 'month_sin', 'month_cos', 'Month_num'])
test_df = test_df.drop(columns=['year_cos', 'month_sin', 'month_cos', 'Month_num'])


train_df.info()


test_df.info()


X = train_df.drop(columns=["date", "num_sold"], axis = "columns")
y = np.log(train_df["num_sold"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

test_df = test_df.drop(columns=["date"])


def lgbm_objective(trial):

    lgbm_params = {
        "n_estimators": 700,
        "subsample": trial.suggest_float("subsample", 0.3, 0.9),
        "bagging_freq": trial.suggest_int("bagging_freq", 4, 20),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 30, 75),
        "max_depth": trial.suggest_int("max_depth", 4, 25),
        "num_leaves": trial.suggest_int("num_leaves", 300, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.1),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 0.88),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.3, 0.9),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.001, 0.1),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.001, 0.1),
        'min_child_weight': trial.suggest_int('min_child_weight',5, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0)

    }

    lgbm_model = LGBMRegressor(**lgbm_params, random_state=42, verbose=-1)

    lgbm_model.fit(X_train, y_train)
    y_pred = np.exp(lgbm_model.predict(X_test))
    return mean_absolute_percentage_error(np.exp(y_test), y_pred)


study_LGBM = optuna.create_study(study_name="LGBM_Kaggle", direction="minimize")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_LGBM.optimize(lgbm_objective, n_trials=100, show_progress_bar=True)


print("Best trial:", study_LGBM.best_trial)


print("Best parameters:", study_LGBM.best_params)


lgbm_final = LGBMRegressor(
    **study_LGBM.best_params,
    n_estimators= 500,
    random_state=42,
    verbose=-1
)
lgbm_final.fit(X_train, y_train)
y_pred = np.exp(lgbm_final.predict(X_test))
print("MAPE:",mean_absolute_percentage_error(np.exp(y_test), y_pred))


y_pred_test = lgbm_final.predict(test_df)
sample_submission["num_sold"] = np.exp(y_pred_test)

sample_submission.to_csv("submission.csv", index=False)
sample_submission

