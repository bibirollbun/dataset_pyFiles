import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.impute import KNNImputer, SimpleImputer

from catboost import CatBoostRegressor, Pool

import warnings


warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

pd.set_option('display.max_columns',None)
# pd.set_option('display.max_rows',None)

# sns.set(style="whitegrid", palette="muted", font_scale=1.1)
pd.plotting.register_matplotlib_converters()
%matplotlib inline


filepath = "/kaggle/input/playground-series-s5e9/train.csv"
filepath_test = "/kaggle/input/playground-series-s5e9/test.csv"
df = pd.read_csv(filepath, index_col="id")
df_test = pd.read_csv(filepath_test)
df.head()


print(f"Shape:- rows: {df.shape[0]} cols: {df.shape[1]}")
print("=" * 30)
print(df.info())

df.describe()


print("Missing values: ", df.duplicated().sum())
print("=" * 30)
print("\nMissing values:\n", df.isnull().sum())


# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".3f")
plt.show()


print(len(df.columns))
df.hist(figsize=(15, 15))
plt.show()


# for col in df.columns:
#     sns.scatterplot(data=df, x=col, y="BeatsPerMinute")
#     plt.show()


df["VocalContent"]=np.sqrt(df["VocalContent"])
df["AcousticQuality"]=np.sqrt(df["AcousticQuality"])
df["InstrumentalScore"]=np.sqrt(df["InstrumentalScore"])
df["LivePerformanceLikelihood"]=np.sqrt(df["LivePerformanceLikelihood"])
df["AudioLoudness"]=np.sqrt(np.max(df["AudioLoudness"]) + 1 - df["AudioLoudness"])



print(len(df.columns))
df.hist(figsize=(15, 15))
plt.show()


target = "BeatsPerMinute"
X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def cbr_objective(trial):
    print("="*40)
    params = {
        "iterations": trial.suggest_int("iterations", 100, 900),
        "depth": 3,
        "learning_rate": 0.07,
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg",  1, 5),
        "task_type": "GPU",
        "loss_function": "RMSE",
        "verbose": 0,
    }

    model = CatBoostRegressor(**params)
    pip = Pipeline([("scale", StandardScaler()), ("model", model)])

    scores = cross_val_score(pip, X, y, cv=5, scoring="neg_root_mean_squared_error")
    return -np.mean(scores)


def get_best_params(objectiveFun):
    study = optuna.create_study(direction="minimize")
    study.optimize(objectiveFun, n_trials=50)
    print("=" * 20)
    print("Best params:", study.best_params)
    print("Best score:", study.best_value)
    return study.best_params


# params=get_best_params(cbr_objective)


params = {
    "iterations": 302,
    "l2_leaf_reg": 1.9749126339502194,
    "depth": 3,
    "learning_rate": 0.07,
    "task_type": "GPU",
    "loss_function": "RMSE",
    "verbose": 0,
}


model = CatBoostRegressor(**params)
pip = Pipeline([("scale", StandardScaler()), ("model", model)])
pip.fit(X,y)


X_test = df_test[X.columns]
test_prd = pip.predict(X_test)

submission = pd.DataFrame({"id": df_test["id"], "BeatsPerMinute": test_prd})
submission.to_csv("submission.csv", index=False)

print(submission.head())

