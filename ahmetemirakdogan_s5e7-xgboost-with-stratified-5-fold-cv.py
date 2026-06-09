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


# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
from xgboost import XGBClassifier

# Model evaluation
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Encoding & preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Warnings
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("../input/playground-series-s5e7/train.csv")
test = pd.read_csv("../input/playground-series-s5e7/test.csv")
submission = pd.read_csv("../input/playground-series-s5e7/sample_submission.csv")


train


train.info()


test


submission


missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("Missing values in train set:")
print(missing)


# Fill missing values
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
for col in numerical_cols:
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(train[col].median(), inplace=True)

categorical_cols = ['Stage_fear', 'Drained_after_socializing']
for col in categorical_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(train[col].mode()[0], inplace=True)


categorical_cols = ['Stage_fear', 'Drained_after_socializing']

le = LabelEncoder()
for col in categorical_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


target_map = {'Introvert': 0, 'Extrovert': 1}
train['Personality'] = train['Personality'].map(target_map)



# Prepare features and target
X = train.drop(columns=["Personality"])
y = train["Personality"]

# Set up Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# To store fold scores
accuracy_scores = []
f1_scores = []

# Best Parameters from Optuna
best_params = {
    'n_estimators': 732,
    'max_depth': 12,
    'learning_rate': 0.05975643376140525,
    'subsample': 0.7428796261522145,
    'colsample_bytree': 0.770123237709161,
    'gamma': 4.003636019950246,
    'reg_lambda': 0.001906896007478168,
    'reg_alpha': 0.0010983164211461024,
    'min_child_weight': 4,
    'objective': 'binary:logistic',
    'eval_metric': 'error',
    'tree_method': 'hist',
    'device': 'cuda',
    'use_label_encoder': False,
}

# Loop through each fold
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Use optimized XGBoost model
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    accuracy_scores.append(acc)
    f1_scores.append(f1)

    print(f"Fold {fold + 1}: Accuracy = {acc:.4f} | F1 Score = {f1:.4f}")

# Mean scores
print("\nOptimized Model CV Results:")
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Mean F1 Score: {np.mean(f1_scores):.4f}")


# Train final model on all training data
X_test = test.copy()
final_model = XGBClassifier(**best_params)
final_model.fit(X, y)

test_preds = final_model.predict(X_test)


reverse_map = {0: "Introvert", 1: "Extrovert"}
submission["Personality"] = [reverse_map[p] for p in test_preds]

# Save submission file
submission.to_csv("submission.csv", index=False)

# Preview
submission.head()

