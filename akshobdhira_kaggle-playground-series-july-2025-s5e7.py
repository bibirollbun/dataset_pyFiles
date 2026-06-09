# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Load datasets
train = pd.read_csv("/kaggle/input/kaggle-competion-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/kaggle-competion-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/kaggle-competion-s5e7/sample_submission.csv")


# Split features and target
X = train.drop(columns=["Personality", "id"])
y = train["Personality"]
X_test = test.drop(columns=["id"])

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)



# Identify column types
numerical_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(include="object").columns.tolist()

# Impute missing values
num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")

X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])



# One-hot encoding
all_data = pd.concat([X, X_test], axis=0)
all_encoded = pd.get_dummies(all_data, columns=categorical_cols)

X_encoded = all_encoded.iloc[:len(X)]
X_test_encoded = all_encoded.iloc[len(X):]



# Define models
lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
catboost = CatBoostClassifier(n_estimators=300, learning_rate=0.05, verbose=0, random_state=42)

voting_clf = VotingClassifier(
    estimators=[("lgbm", lgbm), ("catboost", catboost)],
    voting="soft"
)



# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in cv.split(X_encoded, y_encoded):
    X_train_fold, X_val_fold = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
    y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]

    voting_clf.fit(X_train_fold, y_train_fold)
    preds = voting_clf.predict(X_val_fold)
    acc = accuracy_score(y_val_fold, preds)
    scores.append(acc)

print("CV Accuracy Scores:", scores)
print("Mean CV Accuracy:", np.mean(scores))



# Train on full data and predict
voting_clf.fit(X_encoded, y_encoded)
final_preds = voting_clf.predict(X_test_encoded)
final_labels = le.inverse_transform(final_preds)

submission = pd.DataFrame({
    "id": test["id"],
    "Personality": final_labels
})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("submission.csv is generated.")


