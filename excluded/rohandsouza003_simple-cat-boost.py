# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_path  = "/kaggle/input/playground-series-s5e12/train.csv"
test_path = "/kaggle/input/playground-series-s5e12/test.csv"


# Load Data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


# Display first 5 rows
train.head()


# Check for NaN values
train.isna().sum()


# Summary Statistics of numeric data
train.describe()


train.columns


# Id,s for submission file
labs = test['id']


# Split features and target variable
X = train.drop(columns=['diagnosed_diabetes'])
y = train['diagnosed_diabetes']


def add_interaction_features(df):
    df = df.copy()
    df["income_employment"] = (
        df["income_level"].astype(str) + "_" +
        df["employment_status"].astype(str)
    )
    df["education_income"] = (
        df["education_level"].astype(str) + "_" +
        df["income_level"].astype(str)
    )
    return df


# Split Dataset into train and validation for training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


X_train = add_interaction_features(X_train)
X_val   = add_interaction_features(X_val)


# Get the categorical columns for CatBoost
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()


cat_cols = cat_cols + ["income_employment", "education_income"]



# Checking train and test feature columns are in same orser
test_df = test[X.columns]


test_df   = add_interaction_features(test_df)


# Define the CatBoost model
cat_model = CatBoostClassifier(
    iterations = 600,
    learning_rate= 0.07,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    cat_features=cat_cols,
    random_seed=42,
    verbose=100
)


# Train the model on the data
cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True
)


# Calculate the AUC
from sklearn.metrics import roc_auc_score

y_val_prob = cat_model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_val_prob)

print("Validation AUC:", auc)



# Predict the model 
test_probs = cat_model.predict_proba(test_df)[:, 1]


submission = pd.DataFrame({
    "id": labs,      
    "target": test_probs
})

submission.head()
submission.to_csv("submission.csv", index=False)





















