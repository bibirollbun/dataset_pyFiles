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


df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv").drop(columns=["id"])
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
id_ts = df_test["id"]
df_test = df_test.drop(columns=["id"])


df_train.info()


cols = df_test.columns
num_cols = df_test.select_dtypes(include=["number"]).columns
cat_cols = [col for col in cols if col not in num_cols]

y_train = df_train["Personality"]


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Define imputers
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

preprocessor = ColumnTransformer(
    transformers = [
        ('num', num_imputer, num_cols),
        ('cat', cat_imputer, cat_cols),
    ]
)

pipeline = Pipeline(
    [('preprocessing', preprocessor)]
)
pipeline.set_output(transform="pandas")

df_train = pipeline.fit_transform(df_train)
df_test = pipeline.fit_transform(df_test)


df_train.info(), df_test.info()


# import optuna
# from catboost import CatBoostClassifier, Pool
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import accuracy_score
# import numpy as np

# # Objective function
# def objective(trial):
#     params = {
#         "iterations": trial.suggest_int("iterations", 100, 1000),
#         "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2),
#         "depth": trial.suggest_int("depth", 6, 10),
#         "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
#         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
#         "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1),
#         "od_type" : "Iter",
#         "od_wait" : 100,
#         "cat_features" : ["cat__"+col for col in cat_cols],
#         "eval_metric": "Accuracy",
#         "verbose": 0
#     }

#     model = CatBoostClassifier(**params)

#     # 5-fold cross-validation on training data
#     scores = cross_val_score(
#         model,
#         df_train,
#         y_train,
#         cv=5,
#         scoring='accuracy',
#     )
#     return np.mean(scores)

# # Run Optuna study
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)  # Increase n_trials for better results

# # Best parameters
# print("Best hyperparameters:", study.best_params)



from catboost import CatBoostClassifier
params = {
        'iterations': 101, 
        'learning_rate': 0.14675663939459652, 
        'depth': 8, 
        'l2_leaf_reg': 4, 
        'min_data_in_leaf': 58, 
        'colsample_bylevel': 0.7388876927311806,
        "od_type" : "Iter",
        "od_wait" : 100,
        "cat_features" : ["cat__" + col for col in cat_cols],
        "eval_metric": "Accuracy",
        "verbose": 0,
    }
model = CatBoostClassifier(**params)
model.fit(df_train,y_train)

preds = model.predict(df_test)



out_df = pd.DataFrame(preds, columns=["Personality"], index = id_ts)
out_df.to_csv("submission.csv")


out_df.head()


out_df["Personality"].value_counts()

