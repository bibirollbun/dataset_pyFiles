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


import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


train = pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')

test = pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')

sample = pd.read_csv('/kaggle/input/playground-series-s4e1/sample_submission.csv')


train.head()


train.shape


test.head()


test.shape


sample.head()


train.info()


test.info()


train.isnull().sum()


test.isnull().sum()


train.duplicated().sum()


test.duplicated().sum()


# Save test ids
test_ids = test["id"].copy()


# Drop useless cols once at the beginning
test_ids = test["id"].copy()
train = train.drop(columns=["id", "CustomerId", "Surname"])
test = test.drop(columns=["id", "CustomerId", "Surname"])


train.head()


# Numeric columns
num_cols = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

# Categorical columns
cat_cols = [
    "Geography", "Gender"
]





# Identify numeric and categorical features automatically
# Identify numeric and categorical columns
#num_cols = train.select_dtypes(include=["int64", "float64"]).drop(columns=['Exited']).columns.tolist()
#cat_cols = train.select_dtypes(include=["object"]).columns.tolist()



train.head()


# Transformers
num_transformer = Pipeline(steps=[
    #("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    #("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
])

# Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)


# Split into features + target 
# Features / Target

X = train.drop(columns=["Exited"])
y = train["Exited"]
X_test_final = test.copy()


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


def run_search_rand(pipe, param_dist, name):
    rand = RandomizedSearchCV(pipe, param_dist, n_iter=20, cv=5, n_jobs=-1, random_state=42, scoring="accuracy")
    rand.fit(X_train, y_train)
    print(f"Best model for {name}:")
    print("Best Params:", rand.best_params_)
    return rand


def run_search_grid(pipe, params, name):
    grid = GridSearchCV(pipe, params, cv=5, n_jobs=-1, scoring="accuracy")
    grid.fit(X_train, y_train)
    print(f"\nâœ… Finished GridSearch for {name}")
    print("Best Params:", grid.best_params_)
    return grid


def evaluate_model(search, model_name, X, y, X_valid, y_valid):
    best_model = search.best_estimator_
    val_score = cross_val_score(best_model, X, y, cv=5).mean()
    test_score = accuracy_score(y_valid, best_model.predict(X_valid))
    gap = val_score - test_score

    if gap > 0.05 and val_score >= 0.85:
        fit_msg = "ğŸš¨ Overfitting"
    elif val_score < 0.70 and test_score < 0.70:
        fit_msg = "âš ï¸� Underfitting"
    elif abs(gap) <= 0.05 and test_score >= 0.75:
        fit_msg = "âœ… Good Fit"
    else:
        fit_msg = "â„¹ï¸� Borderline"

    print(f"\n--- {model_name} ---")
    print("Validation Accuracy:", round(val_score, 3))
    print("Test Accuracy:", round(test_score, 3))
    print("Fit:", fit_msg)

    return {
        "Model": model_name,
        "Validation_Accuracy": round(val_score, 3),
        "Test_Accuracy": round(test_score, 3),
        "Fit": fit_msg
    }



rf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])
rf_param_dist = {
    "model__n_estimators": np.linspace(200, 1000, 5, dtype=int),
    "model__max_depth": [None, 10, 20, 30]
    #"model__min_samples_split": np.linspace(2, 10, 5, dtype=int),
    #"model__min_samples_leaf": np.linspace(1, 5, 5, dtype=int)
}

rf_params = {
    "model__n_estimators": [200, 500, 800],
    "model__max_depth": [None, 10, 20]
    #"model__min_samples_split": np.linspace(2, 10, 5, dtype=int),
    #"model__min_samples_leaf": np.linspace(1, 5, 5, dtype=int)
}


 #grid_rf = run_search_grid(rf_pipe, rf_params, "Random Forest")


#rf_search = run_search_rand(rf_pipe, rf_param_dist, "Random Forest")


xgb_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(random_state=42, eval_metric="logloss"))
])

xgb_param_dist = {
    "model__n_estimators": np.linspace(200, 800, 5, dtype=int),
    "model__learning_rate": np.linspace(0.01, 0.3, 10),
    "model__max_depth": np.linspace(3, 10, 5, dtype=int)
    #"model__subsample": np.linspace(0.7, 1.0, 5),
    #"model__colsample_bytree": np.linspace(0.7, 1.0, 5)
}

xgb_params = {
    "model__n_estimators": [200, 500, 800],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [3, 5, 7]
    #"model__subsample": np.linspace(0.7, 1.0, 5),
    #"model__colsample_bytree": np.linspace(0.7, 1.0, 5)
}



grid_xgb = run_search_grid(xgb_pipe, xgb_params, "XG Boost")


xgb_search = run_search_rand(xgb_pipe, xgb_param_dist, "XGBoost")


grids = [
    #("Random Forest", grid_rf),
    #("Decision Tree", grid_dt),
    #("Logistic Regression", grid_lr),
    #("Gradient Boosting", grid_gb),
    ("XGBoost", grid_xgb)
    #("LightGBM", grid_lgb)
]

results = []
# Loop through each model and collect results
for name, grid in grids:
    res = evaluate_model(grid, name, X_train, y_train, X_valid, y_valid)
    results.append(res)

# Convert results to DataFrame
df_results = pd.DataFrame(results)


best_model = grid_xgb.best_estimator_
final_preds = best_model.predict(X_test_final)


submission = pd.DataFrame({
    "id": test_ids,  # restore ids
    "Exited": final_preds
})
submission.to_csv("submission.csv", index=False)




