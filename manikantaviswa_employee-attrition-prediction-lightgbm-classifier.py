import os


# STEP 1: Load Libraries

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

import pickle
import joblib


# STEP 2: Load Dataset
df = pd.read_csv("/kaggle/input/playground-series-s3e3/train.csv")
df.head()


# STEP 3: Basic Inspection
print(df.shape)
print(df.isnull().sum())


# STEP 4: Drop Unwanted Columns
df.drop(columns=["id"], inplace=True, errors="ignore")


# STEP 5: Encode the Categorical values
import pickle

# Dictionary to store encoders
encoders = {}

# Encode categorical columns
for col in df.columns:
    if df[col].dtype == "object":
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder   # save encoder for that column

# Save encoders to a file
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("Label Encoders saved successfully!")


# STEP 6: Fill the missing values with the mean
df = df.fillna(df.mean(numeric_only=True))

df.isnull().sum()


# STEP 7: Feature–Target Split
X = df.drop(columns=["Attrition"])
y = df["Attrition"]


# STEP 8: Train–Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# STEP 9: Check if the target variable is balnced or not

# Check shape
print("Dataset Shape:", df.shape)

# Target distribution
class_counts = df['Attrition'].value_counts()
print(class_counts)

# Visualization
plt.figure(figsize=(5,4))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title("Class Distribution (0-Stayed, 1-Left)")
plt.ylabel("Count")
plt.xlabel("Attrition")
plt.show()


# STEP 10: Balance the target variable

df_majority = df[df['Attrition'] == 0]   # Stayed
df_minority = df[df['Attrition'] == 1]   # Left

df_minority_oversampled = df_minority.sample(
    n=len(df_majority),
    replace=True,
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_oversampled])

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("Balanced Dataset Shape:", df_balanced.shape)
print(df_balanced['Attrition'].value_counts())

plt.figure(figsize=(5,4))
sns.barplot(
    x=df_balanced['Attrition'].value_counts().index,
    y=df_balanced['Attrition'].value_counts().values
)
plt.title("Balanced Class Distribution (Random Oversampling)")
plt.ylabel("Count")
plt.xlabel("Attrition")
plt.show()


# STEP 11: Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")


# STEP 12: Outlier detection and removal

from sklearn.ensemble import IsolationForest

# Print shape BEFORE outlier removal
print("Training data shape BEFORE outlier removal:")
print("X_train:", X_train_scaled.shape)
print("y_train:", y_train.shape)
print("-" * 50)

# Initialize Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

# Fit and predict outliers
outlier_preds = iso_forest.fit_predict(X_train_scaled)

# Keep only inliers
mask = outlier_preds == 1
X_train_clean = X_train_scaled[mask]
y_train_clean = y_train.iloc[mask]

# Print shape AFTER outlier removal
print("Training data shape AFTER outlier removal:")
print("X_train_clean:", X_train_clean.shape)
print("y_train_clean:", y_train_clean.shape)
print("-" * 50)


# Step 13: Principle Component Analysis(PCA)

# BEFORE PCA 
print("BEFORE PCA")
print("X_train_clean shape:", X_train_clean.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
print("-" * 50)

# APPLY PCA 
pca = PCA(n_components=0.95, random_state=42)

X_train_pca = pca.fit_transform(X_train_clean)
X_test_pca = pca.transform(X_test_scaled)

joblib.dump(pca, "pca.pkl")

# AFTER PCA 
print("AFTER PCA")
print("X_train_pca shape:", X_train_pca.shape)
print("X_test_pca shape:", X_test_pca.shape)
print("-" * 50)

# PCA DETAILS
print("Number of components selected:", pca.n_components_)
print("Total explained variance ratio:", sum(pca.explained_variance_ratio_))


import lightgbm as lgb
import optuna

def objective(trial):

    params = {
        # Core parameters
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,

        # GPU parameters
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,

        # Tree structure
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", -1, 15),

        # Learning
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),

        # Sampling
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),

        # Regularization
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),

        # Other controls
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 5.0),

        "random_state": 42
    }

    model = lgb.LGBMClassifier(**params)

    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)

    return acc


# Run Optuna Study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("Best Accuracy:", study.best_value)
print("Best Parameters:")
print(study.best_params)


# Step 15: Train with best model

# Best parameters from Optuna
best_params = study.best_params
print("Best LightGBM Parameters:", best_params)

# Add required fixed parameters
best_params.update({
    "objective": "binary",
    "metric": "binary_logloss",
    "device": "gpu",        # remove if GPU not enabled
    "verbosity": -1,
    "random_state": 42
})

# Train best model
best_lightgbm = lgb.LGBMClassifier(**best_params)

best_lightgbm.fit(X_train_scaled, y_train)

# Predictions
y_pred = best_lightgbm.predict(X_test_scaled)
y_prob = best_lightgbm.predict_proba(X_test_scaled)[:, 1]

# Save model
joblib.dump(best_lightgbm, "employee_attrition_model.pkl")


# STEP 16: Model Evaluation

#from sklearn.metrics import accuracy_score, classification_report

# TRAIN ACCURACY 
y_train_pred = best_lightgbm.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)

# TEST ACCURACY 
y_test_pred = best_lightgbm.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

# PRINT RESULTS 
print("TRAIN Accuracy:", train_accuracy)
print("TEST Accuracy:", test_accuracy)

print("\nClassification Report (Test Data):\n")
print(classification_report(y_test, y_test_pred))


# STEP 17: Visulizations of train and test data
from sklearn.metrics import roc_curve, roc_auc_score

# BAR GRAPH 
accuracies = [train_accuracy, test_accuracy]
labels = ["Train Accuracy", "Test Accuracy"]

plt.figure(figsize=(5,4))
plt.bar(labels, accuracies)
plt.ylim(0, 1)

# Add value labels on bars
for i, value in enumerate(accuracies):
    plt.text(i, value + 0.01, f"{value:.2f}", ha='center', fontsize=10)

plt.title("LightGBM Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Dataset")
plt.show()

# Probability scores(ROC-AUC Scores)
y_test_prob = best_lightgbm.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_test_prob)
auc_score = roc_auc_score(y_test, y_test_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"SVM (AUC = {auc_score:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - LightGBM")
plt.legend()
plt.show()


# STEP 18: Confusion Matrix (Actual vs Predicted)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# STEP 19: Actual vs Predicted Graph

comparison = pd.DataFrame({
    "Actual": y_test.values[:100],
    "Predicted": y_pred[:100]
})

comparison.plot(kind="bar", figsize=(12,4))
plt.title("Actual vs Predicted Employee Attrition (Sample)")
plt.show()


# STEP 20: User Mushroom Poisonous Prediction

def predict_diagnosis(user_data):
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("employee_attrition_model.pkl")

    # Scale user input
    user_scaled = scaler.transform([user_data])

    # Prediction
    pred = model.predict(user_scaled)[0]
    prob = model.predict_proba(user_scaled)[0]

    label = "Left" if pred == 1 else "Stayed"

    return label, prob


# STEP 21: Example User Test + Visualization

# Example: take one real sample as user input
sample_patient = X.iloc[0].values  

label, probability = predict_diagnosis(sample_patient)

print("Predicted Employee Attrition:", label)
print("Probability [Stayed, Left]:", probability)

# Visualization
plt.figure(figsize=(5,4))
plt.bar(["Stayed", "Left"], probability)
plt.ylabel("Probability")
plt.title("Employee Attrition Prediction - LightGBM")
plt.ylim(0, 1)

# Add values on bars
for i, v in enumerate(probability):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

plt.show()


# STEP 22: Save Full Pipeline

joblib.dump({
    "model": best_lightgbm,
    "scaler": scaler,
    "label_encoders": encoders
}, "employee_attrition_full_pipeline.pkl")

print("Full pipeline saved successfully")

