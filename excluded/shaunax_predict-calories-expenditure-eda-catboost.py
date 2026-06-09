# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


def check_data(dataframe):
    print(dataframe.head())
    print("-"*20)
    print(dataframe.describe().T)
    print("-"*20)
    print(dataframe.info())

check_data(train)


train.duplicated().sum()


train["Sex"].value_counts(ascending=False)


plt.figure(figsize = (10,6))
sns.countplot(x = "Sex", data=train)
plt.show()


def hist_box_plot(data):
    plt.figure(figsize = (12,2))
    sns.histplot(train[data], kde = True)
    plt.title(f"Histogram Grafiği - {data}")
    plt.show()

    plt.figure(figsize = (12,6))
    sns.boxplot(train[data])
    plt.title(f"BoxPlot Grafiği - {data}")
    plt.show()


num_features = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp","Calories"]
for i in num_features:
    hist_box_plot(i)


burn_calories = train.groupby("Sex")["Calories"].mean()
print(burn_calories)


plt.figure(figsize=(10,4))
sns.lmplot(x = "Heart_Rate", y = "Calories", hue = "Sex", data = train)
plt.show()


def full_feature_engineering(df, cross_features=None):
    df = df.copy()

    df["Bmi"] = df["Weight"] / (df["Height"] / 100)**2
    df["HR_per_min"] = df["Heart_Rate"] / df["Duration"]
    df["Temp_diff_from_norm"] = df["Body_Temp"] - 37.0
    df["Temp_Heart_Interaction"] = df["Body_Temp"] * df["Heart_Rate"]

    df = df.loc[:, ~df.columns.duplicated()]  

    if cross_features is not None:
        for i in range(len(cross_features)):
            for j in range(i + 1, len(cross_features)):
                f1 = cross_features[i]
                f2 = cross_features[j]
                new_col = f"{f1}_x_{f2}"
                df[new_col] = df[f1] * df[f2]

    return df


cross_features = ["Bmi", "HR_per_min", "Temp_diff_from_norm", "Temp_Heart_Interaction"]
train = full_feature_engineering(train, cross_features)
test = full_feature_engineering(test, cross_features)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train["Sex"] = le.fit_transform(train["Sex"])
test["Sex"] = le.transform(test["Sex"])


X = train.drop(columns=["Calories","id"])
y = np.log1p(train['Calories'])  


from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_log_error
import optuna

def rmsle(y_true, y_pred):
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 250, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 3, 6),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3, 5),
        'loss_function': 'RMSE',
        'random_seed': 42,
        'eval_metric': 'RMSE',
        'early_stopping_rounds': 200,
        'verbose': 100,
        'task_type': 'GPU'
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmsle_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, use_best_model=True)

        val_preds_log = model.predict(X_val)
        val_preds = np.expm1(val_preds_log)
        y_val_actual = np.expm1(y_val)

        score = rmsle(y_val_actual, val_preds)
        rmsle_scores.append(score)

    return np.mean(rmsle_scores)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

print("\nEn iyi Trial:", study.best_trial.number)
print("En düşük RMSLE:", study.best_trial.value)
print("\nEn iyi Hiperparametreler:")
for key, value in study.best_trial.params.items():
    print(f"{key}: {value}")


best_params = {

    "iterations": 1107,
    "learning_rate": 0.048471029694328364,
    "depth": 9,
    "l2_leaf_reg": 5.172606554303554,
    "loss_function": "RMSE",
    "random_seed": 42,
    "verbose": 0,
    "task_type" : "GPU"
}


final_model = CatBoostRegressor(**best_params)

final_model.fit(X, y)

test_pred_log = final_model.predict(test.drop(columns=["id"]))
test_pred = np.expm1(test_pred_log)

test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame(
    {
    "id": test["id"],
    "Calories": test_pred
    }
)

submission.to_csv("submission.csv", index=False)


def sub_df(df):
    print("Min : ",submission["Calories"].min())
    print("Max : ",submission["Calories"].max())
    print("Medyan : ",submission["Calories"].median())
    print("Ortalama : ",submission["Calories"].mean())
    print("Standard Sapma : ",submission["Calories"].std())

sub_df(submission)


"""
Min :  0.9692090425897408
Max :  297.6447337177263
Medyan :  76.46314879462071
Ortalama :  88.1795023674211
Standard Sapma :  62.280399593896405
"""

