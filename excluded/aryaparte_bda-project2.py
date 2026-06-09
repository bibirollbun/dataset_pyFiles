import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report




# Load the dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# Drop 'maxtemp' and 'mintemp' columns
train.drop(columns=['maxtemp', 'mintemp'], inplace=True)
test.drop(columns=['maxtemp', 'mintemp'], inplace=True)

# Fill missing values with mean
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

# Splitting features and target
X = train.drop(columns=['rainfall'])
y = train['rainfall']
X_test = test

# Train-test split (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define models with hyperparameter tuning
param_grids = {
    "RandomForest": {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 20, 30],
        "min_samples_split": [2, 5, 10]
    },
    "ExtraTrees": {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 20, 30],
        "min_samples_split": [2, 5, 10]
    },
    "GradientBoosting": {
        "n_estimators": [200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7]
    },
    "XGBoost": {
        "n_estimators": [200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7]
    },
    "LightGBM": {
        "n_estimators": [200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [20, 31, 40]
    },
    "CatBoost": {
        "iterations": [500, 1000],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [3, 5, 7]
    },
    "LogisticRegression": {
        "C": [0.1, 1, 10],
        "penalty": ["l2"]
    }
}

best_model = None
best_val_accuracy = 0

for name, params in param_grids.items():
    print(f"Tuning {name}...")
    if name == "RandomForest":
        model = RandomForestClassifier(random_state=42, class_weight="balanced")
    elif name == "ExtraTrees":
        model = ExtraTreesClassifier(random_state=42, class_weight="balanced")
    elif name == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
    elif name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif name == "LightGBM":
        model = LGBMClassifier(random_state=42)
    elif name == "CatBoost":
        model = CatBoostClassifier(verbose=0, random_state=42)
    elif name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, random_state=42)

    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model_instance = grid_search.best_estimator_

    train_acc = accuracy_score(y_train, best_model_instance.predict(X_train))
    val_acc = accuracy_score(y_val, best_model_instance.predict(X_val))

    print(f"{name} Best Params: {grid_search.best_params_}")
    print(f"{name} Train Accuracy: {train_acc}")
    print(f"{name} Validation Accuracy: {val_acc}\n")
    print(classification_report(y_val, best_model_instance.predict(X_val)))

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_model = best_model_instance

# Predict probabilities on test set using best model
rainfall_prob = best_model.predict_proba(X_test)[:, 1]

# Prepare submission file
# submission = pd.DataFrame({'id': test.index, 'rainfall': rainfall_prob})
# submission.to_csv("submission_optimized.csv", index=False)

print(f"\nBest Model: {best_model}")











