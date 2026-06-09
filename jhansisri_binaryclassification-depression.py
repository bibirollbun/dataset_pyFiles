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


import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


df_train = pd.read_csv("/kaggle/input/mentally-stability-of-the-person/train.csv")
df_test = pd.read_csv("/kaggle/input/mentally-stability-of-the-person/test.csv")
df_sample_submission = pd.read_csv("/kaggle/input/mentally-stability-of-the-person/sample_submission.csv")


df_train.head()


X = df_train.drop(["Depression","id","Name"], axis = 1)
y = df_train["Depression"]


cat_cols = X.select_dtypes("O").columns
num_cols = [col for col in X.columns if col not in cat_cols]


print("Categorical columns:", cat_cols)
print("Numeric columns:", num_cols)


for col in cat_cols:
    print("---------------" + col + "-----------------------")
    X[col].value_counts(normalize = True).plot(kind = "bar")
    plt.show()


label_cols = [col for col in cat_cols if len(X[col].unique()) > 10]
ohe_cols = [col for col in cat_cols if col not in label_cols]


num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])









cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])



preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])



pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)

print(f"Model Accuracy: {accuracy:.2f}")


pred_array_baseline = pipeline.predict(df_test.iloc[:,2:])
df_test["Depression"] = pred_array_baseline
submission = pd.DataFrame({"id": df_test["id"],
                        "Depression":df_test["Depression"]})
submission.to_csv("submission.csv", index=False)


from tqdm import tqdm
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5,10],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [2, 4]
}

# **Perform GridSearchCV**
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Monitor grid search progress
results = grid_search.cv_results_

# Print out grid search progress from cv_results_
for i in tqdm(range(len(results["params"])), desc="GridSearch Progress"):
    print(f"Testing configuration {i + 1}: {results['params'][i]}")




print("Best Hyperparameters:", grid_search.best_params_)


best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")


pred_array_tuned = best_model.predict(df_test.iloc[:,2:])
df_test["Depression"] = pred_array_tuned
submission = pd.DataFrame({"id": df_test["id"],
                        "Depression":df_test["Depression"]})
submission.to_csv("submission_tuned.csv", index=False)

