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
df = pd.read_csv("/kaggle/input/playground-series-s3e23/train.csv")
df.head()


df.columns


# STEP 4: Drop Unwanted Columns
df.drop(columns=["Unnamed: 32"], inplace=True, errors="ignore")


# STEP 5: Encoding the data

# Identify object / categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
print("Categorical columns to encode:", categorical_cols)

# Initialize LabelEncoder
label_encoders = {}

# Encode each categorical column
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # convert to string just in case
    label_encoders[col] = le

# Save each LabelEncoder
for col, le in label_encoders.items():
    joblib.dump(le, f"label_encoder_{col}.pkl")

print("\nAll categorical columns encoded.")


# STEP 6: Feature–Target Split
X = df.drop(columns=["defects"])
y = df["defects"]


# STEP 7: Train–Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# STEP 8: Check if the target variable is balnced or not

# Check shape
print("Dataset Shape:", df.shape)

# Target distribution
class_counts = df['defects'].value_counts()
print(class_counts)

# Visualization
plt.figure(figsize=(5,4))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title("Class Distribution")
plt.ylabel("Count")
plt.xlabel("Defects")
plt.show()


# STEP 9: Balance the target variable

df_majority = df[df['defects'] == 0]   # False
df_minority = df[df['defects'] == 1]   # True

df_minority_oversampled = df_minority.sample(
    n=len(df_majority),
    replace=True,
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_oversampled])

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("Balanced Dataset Shape:", df_balanced.shape)
print(df_balanced['defects'].value_counts())

plt.figure(figsize=(5,4))
sns.barplot(
    x=df_balanced['defects'].value_counts().index,
    y=df_balanced['defects'].value_counts().values
)
plt.title("Balanced Class Distribution (Random Oversampling)")
plt.ylabel("Count")
plt.xlabel("Defects")
plt.show()


# STEP 10: Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")


# STEP 11: Outlier detection and removal

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


# Step 12: Principle Component Analysis(PCA)

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


# STEP 13: Optuna Optimization

from catboost import CatBoostClassifier
import optuna

def objective(trial):
    # Choose bootstrap type first
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "Poisson"])

    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 0, 10),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
        "verbose": 0,
        "random_state": 42,
        "task_type": "GPU",
        "bootstrap_type": bootstrap_type
    }

    # Conditional parameters based on bootstrap_type
    if bootstrap_type == "Bayesian":
        # Bayesian uses bagging_temperature
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 1)
    else:
        # Bernoulli or Poisson use subsample
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

    model = CatBoostClassifier(**params)
    model.fit(X_train_scaled, y_train, eval_set=(X_test_scaled, y_test), verbose=0)

    preds = model.predict(X_test_scaled)
    return accuracy_score(y_test, preds)

# Run Optuna Study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("Best Parameters:")
print(study.best_params)


# Step 14: Train with best model

best_params = study.best_params
print("Best CatGBM Parameters:", best_params)

best_catboost = CatBoostClassifier(
    **best_params
)

best_catboost.fit(X_train_scaled, y_train)

y_pred = best_catboost.predict(X_test_scaled)
y_prob = best_catboost.predict_proba(X_test_scaled)[:, 1]

joblib.dump(best_catboost, "software_defects_model.pkl")


# STEP 15: Model Evaluation

#from sklearn.metrics import accuracy_score, classification_report

# TRAIN ACCURACY 
y_train_pred = best_catboost.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)

# TEST ACCURACY 
y_test_pred = best_catboost.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

# PRINT RESULTS 
print("TRAIN Accuracy:", train_accuracy)
print("TEST Accuracy:", test_accuracy)

print("\nClassification Report (Test Data):\n")
print(classification_report(y_test, y_test_pred))


# STEP 16: Visulizations of train and test data
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

plt.title("CatGBM Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Dataset")
plt.show()

# Probability scores(ROC-AUC Scores)
y_test_prob = best_catboost.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_test_prob)
auc_score = roc_auc_score(y_test, y_test_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"SVM (AUC = {auc_score:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - CatGBM")
plt.legend()
plt.show()


# STEP 17: Confusion Matrix (Actual vs Predicted)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# STEP 18: Actual vs Predicted Graph

le = LabelEncoder()

y_test_enc = le.fit_transform(y_test)
y_pred_enc = le.transform(y_pred)

comparison = pd.DataFrame({
    "Actual": y_test_enc[:100],
    "Predicted": y_pred_enc[:100]
})

comparison.plot(kind="bar", figsize=(12,4))
plt.title("Actual vs Predicted (Sample)")
plt.show()

plt.figure(figsize=(12,4))
plt.plot(y_test_enc[:100], label="Actual", marker='o')
plt.plot(y_pred_enc[:100], label="Predicted", marker='x')
plt.title("Actual vs Predicted (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Class (0 = e, 1 = p)")
plt.legend()
plt.show()


# STEP 19: User Transaction Prediction

def predict_diagnosis(user_data):
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("software_defects_model.pkl")

    # Scale user input
    user_scaled = scaler.transform([user_data])

    # Prediction
    pred = model.predict(user_scaled)[0]
    prob = model.predict_proba(user_scaled)[0]

    label = "No" if pred == 1 else "Yes"

    return label, prob


# STEP 20: Example User Test + Visualization

# Example: take one real sample as user input
sample_patient = X.iloc[0].values  

label, probability = predict_diagnosis(sample_patient)

print("Predicted Defects:", label)
print("Probability [Yes, No]:", probability)

# Visualization
plt.figure(figsize=(5,4))
plt.bar(["Yes", "No"], probability)
plt.ylabel("Probability")
plt.title("Software Defects Prediction - CatGBM")
plt.ylim(0, 1)

# Add values on bars
for i, v in enumerate(probability):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

plt.show()


# STEP 21: Save Full Pipeline

joblib.dump({
    "model": best_catboost,
    "scaler": scaler,
    "label_encoders": label_encoders
}, "software_defects_full_pipeline.pkl")

print("Full pipeline saved successfully")

