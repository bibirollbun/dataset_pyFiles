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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score


import warnings
warnings.filterwarnings('ignore')



train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train.head()


test.head()


train.shape


test.shape


train['Personality'].unique()


train.info()


test.info()


train.isnull().sum()


test.isnull().sum()


train['Stage_fear'].unique()


train.columns


train.duplicated().sum()


test.duplicated().sum()


num_cols = ['Time_spent_Alone','Social_event_attendance', 'Going_outside', 'Friends_circle_size' ,'Post_frequency']

cat_cols = ['Stage_fear', 'Drained_after_socializing']


num_transformer = Pipeline(steps=[
    
    ('imputer', SimpleImputer(strategy='median')),
    ('scalar', StandardScaler())

])

cat_transformer = Pipeline(steps=[
    
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))

])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ]
)


# Encode Target (LabelEncoder)

le = LabelEncoder()

y = le.fit_transform(train['Personality']) #  0 = Extrovert, 1 = Introvert


X = train.drop(columns=['Personality', 'id'])

X_test_final = test.drop(columns=['id'])


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


def run_search_grid(pipe, params, name):
    # Grid Search
    grid = GridSearchCV(pipe, params, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"\nâœ… Finished GridSearch for {name}")
    print("Best Params:", grid.best_params_)
    #print("Best CV Score:", round(grid.best_score_, 3))

    return grid


# LightGBM
lgb_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LGBMClassifier(random_state=42, verbose=-1))
])
lgb_params = {
    "model__n_estimators": [200, 400, 500],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [5, 10, 20, 30],
    #"model__num_leaves": [31, 52, 63, 127]
}


grid_lgb = run_search_grid(lgb_pipe, lgb_params, 'LightGBM')


# Decision Tree
dt_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42))
])
dt_params = {
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5, 10],
    #"model__min_samples_leaf": [1, 2, 4]
}


grid_dt = run_search_grid(dt_pipe, dt_params, "Decision Tree")


# Evaluation Fuction

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
    print("Best Params:", search.best_params_)
    print("Validation Accuracy:", round(val_score, 3))
    print("Test Accuracy:", round(test_score, 3))
    print("Fit Assessment:", fit_msg)

    return {
        "Model": model_name,
        "Validation_Accuracy": round(val_score, 3),
        "Test_Accuracy": round(test_score, 3),
        "Fit_Assessment": fit_msg
    }


grids = [
    #("Random Forest", grid_rf),
    ("Decision Tree", grid_dt),
    #("Logistic Regression", grid_lr),
    #("Gradient Boosting", grid_gb),
    #("XGBoost", grid_xgb),
    ("LightGBM", grid_lgb)
]

results = []
# Loop through each model and collect results
for name, grid in grids:
    res = evaluate_model(grid, name, X_train, y_train, X_valid, y_valid)
    results.append(res)

# Convert results to DataFrame
df_results = pd.DataFrame(results)


best_lgb_model = grid_lgb.best_estimator_

test_predictions = best_lgb_model.predict(X_test_final)


test_predictions.shape



X_test_final.shape


print(test_predictions[:10])



single_test = X_test_final.iloc[[0]]  # first row as DataFrame

single_pred = best_lgb_model.predict(single_test)
print(single_pred)


# Example: create a submission DataFrame

submission = pd.DataFrame({
    "id": test["id"],
    "Personality": test_predictions
})


submission.head()


submission.info()


submission['Personality'] = submission['Personality'].map({0: 'Extrovert', 1: 'Introvert'})


submission.head()


# Save submission
submission.to_csv('submission.csv', index=False)
print("âœ… Predictions saved to submission.csv")




