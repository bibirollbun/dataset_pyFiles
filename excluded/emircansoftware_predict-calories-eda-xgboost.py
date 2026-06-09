# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df=pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


df.head()


df.info()


df.describe()


df=df.drop("id",axis=1)


df["Sex"]=pd.get_dummies(df["Sex"],dtype=int, drop_first=True)


df.duplicated().value_counts()


df=df.drop_duplicates()


import seaborn as sns
import matplotlib.pyplot as plt


corr_matrix=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=True)
plt.title("Corr matrix")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=df["Age"])
plt.title("Outlier Detection via Boxplot")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=df["Height"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df["Height"].sort_values().unique()


df=df[df["Height"]<217]


df=df[df["Height"]>129]


df.info()


plt.figure(figsize=(10, 6))
sns.boxplot(x=df["Weight"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df["Weight"].sort_values().unique()


df=df[df["Weight"]<124]


plt.figure(figsize=(10, 6))
sns.boxplot(x=df["Duration"])
plt.title("Outlier Detection via Boxplot")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=df["Heart_Rate"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df["Heart_Rate"].sort_values().unique()


df=df[df["Heart_Rate"]<126]


plt.figure(figsize=(10, 6))
sns.boxplot(x=df["Body_Temp"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df["Body_Temp"].sort_values().unique()


df=df[df["Body_Temp"]>37.9]


plt.figure(figsize=(10, 6))
sns.boxplot(x=df["Calories"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df["Calories"].sort_values().unique()


df=df[df["Calories"]<289]


df.info()


test=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


test.set_index("id", inplace=True)
test["Sex"]=pd.get_dummies(test["Sex"],dtype=int,drop_first=True)


X=df.drop("Calories",axis=1)
y=df["Calories"]
X_test = test.copy()


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

FOLDS = 3
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)


models = {
    "cat": CatBoostRegressor(
        iterations=3500,
        learning_rate=0.02,
        depth=10,
        loss_function='RMSE',
        l2_leaf_reg=3,
        random_seed=42,
        eval_metric='RMSE',
        early_stopping_rounds=200,
        verbose=False,
        task_type='GPU'
    ),
    "xgb": XGBRegressor(
        max_depth=10,
        colsample_bytree=0.75,
        subsample=0.9,
        n_estimators=2000,
        learning_rate=0.01,
        gamma=0.01,
        max_delta_step=2,
        eval_metric="rmse",
        enable_categorical=True,
        verbosity=0
    ),
    "lgb": LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),
    "rf": RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),
    "gbr": GradientBoostingRegressor(
        n_estimators=800,
        learning_rate=0.01,
        max_depth=6,
        random_state=42
    )
}


oof_preds = {name: np.zeros(len(X)) for name in models}
test_preds = {name: np.zeros(len(X_test)) for name in models}

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"##### Fold {fold+1} #####")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    y_train_log = np.log1p(y_train)
    y_valid_log = np.log1p(y_valid)

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train_log)

        val_preds = model.predict(X_valid)
        test_pred = model.predict(X_test)

        oof_preds[name][valid_idx] = val_preds
        test_preds[name] += test_pred / FOLDS

# Stack giriş verisi oluştur
X_stack = np.column_stack([oof_preds[name] for name in models])
X_stack_test = np.column_stack([test_preds[name] for name in models])

# Stacking modeli (meta-model)
stack_model = Ridge(alpha=1.0)
stack_model.fit(X_stack, np.log1p(y))

final_preds_log = stack_model.predict(X_stack_test)
final_preds = np.expm1(final_preds_log)

# İsteğe bağlı: OOF değerlendirme
for name in models:
    rmse = mean_squared_error(np.log1p(y), oof_preds[name]) ** 0.5
    print(f"{name.upper()} OOF RMSE: {rmse:.4f}")

stack_oof = stack_model.predict(X_stack)
stack_rmse = mean_squared_error(np.log1p(y), stack_oof) ** 0.5
print(f"STACKING MODEL OOF RMSE: {stack_rmse:.4f}")


submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


submission["Calories"]=final_preds



submission.head()


submission.to_csv("submission.csv", index=False)




