import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna  # Hyperparameter tuning
import time
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv("../input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("../input/playground-series-s5e3/test.csv")
submission_df = pd.read_csv("../input/playground-series-s5e3/sample_submission.csv")


# Dataset Summary
print("ðŸ“Š Dataset Summary")
print("Train Dataset:")
print("Features: day, pressure, maxtemp, temperature, mintemp, dewpoint, humidity, cloud, sunshine, winddirection, windspeed")
print("Target: rainfall (1 = rain, 0 = no rain)")
print("\nTest Dataset:")
print("Same features, but without the rainfall column (to be predicted)")


# Data exploration
print("Train Dataset Info:")
print(train_df.info(), "\n")
print(train_df.describe())

print("\nTest Dataset Info:")
print(test_df.info(), "\n")
print(test_df.describe())


# Visualize target distribution
plt.figure(figsize=(6,4))
sns.countplot(x=train_df['rainfall'])
plt.title("Target Distribution (Rainfall)")
plt.show()


# Feature correlations
plt.figure(figsize=(12,6))
sns.heatmap(train_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


# Check missing values
print("\nMissing values in train dataset:\n", train_df.isnull().sum())
print("\nMissing values in test dataset:\n", test_df.isnull().sum())


# Handle missing values in test set
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X_test_imputed = imputer.fit_transform(test_df.drop(columns=["id"]))


# Feature selection
X = train_df.drop(columns=["id", "rainfall"])
y = train_df["rainfall"]
X_test = X_test_imputed


# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)


# Apply PCA for feature reduction
pca = PCA(n_components=7)
X_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled)


# Visualize PCA explained variance
plt.figure(figsize=(8,5))
plt.plot(range(1,8), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('PCA Explained Variance Ratio')
plt.show()


# Split data for training and validation
X_train, X_valid, y_train, y_valid = train_test_split(X_pca, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)



# Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_valid)[:, 1]
    results[name] = roc_auc_score(y_valid, y_pred)
    print(f"ROC AUC Score - {name}: {results[name]}")

print("âœ… Logistic Regression Model Trained!")
print("Initial evaluation score based on ROC-AUC:", results["Logistic Regression"], "ðŸŽ¯")


# Visualize model performance
plt.figure(figsize=(10,6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.xticks(rotation=45)
plt.ylabel("ROC AUC Score")
plt.title("Model Performance Comparison")
plt.show()


# Hyperparameter tuning for Random Forest with Optuna
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_valid)[:, 1]
    return roc_auc_score(y_valid, y_pred)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

best_params = study.best_params
print("\nBest Hyperparameters:", best_params)


# Train Random Forest with best parameters
final_rf = RandomForestClassifier(**best_params, random_state=42)
final_rf.fit(X_train, y_train)
y_pred_final_rf = final_rf.predict_proba(X_valid)[:, 1]
score_final_rf = roc_auc_score(y_valid, y_pred_final_rf)
print("ROC AUC Score - Tuned Random Forest:", score_final_rf)



# Final test predictions
y_test_pred = final_rf.predict_proba(X_test_pca)[:, 1]



# Create submission file
submission = pd.DataFrame({"id": test_df["id"], "rainfall": y_test_pred})
submission.to_csv("submission.csv", index=False)

print("\nSubmission file 'submission.csv' successfully created âœ…")


