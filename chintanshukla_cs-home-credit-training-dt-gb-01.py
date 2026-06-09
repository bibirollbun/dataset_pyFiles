# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score,  precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score


# Load processed dataset
data_path = "/kaggle/input/home-credit-defualt-processed-training-data/application_train_processed.csv"
df_application_train = pd.read_csv(data_path)

# Quick inspection
print("Shape:", df_application_train.shape)
print("Columns:", df_application_train.columns[:10])  # first 10 columns
df_application_train.head()


# Features and target
X = df_application_train.drop(columns=['SK_ID_CURR', 'TARGET'])
y = df_application_train['TARGET']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% test set
    random_state=42,      # reproducibility
    stratify=y            # preserve class distribution
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)


# Initialize and train Decision Tree
dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")
dt.fit(X_train, y_train)


# Predictions
y_pred_dt = dt.predict(X_test)
y_proba_dt = dt.predict_proba(X_test)[:,1]

# Evaluation
print(classification_report(y_test, y_pred_dt))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_dt))


# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'class_weight': [None, 'balanced']  # handle imbalance
}

# Initialize GridSearchCV
grid_dt = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,                # 5-fold cross-validation
    scoring='roc_auc',   # optimize for ROC-AUC
    n_jobs=-1            # use all cores for speed
)


# Fit model
grid_dt.fit(X_train, y_train)


# Best parameters
print("Best parameters:", grid_dt.best_params_)


# Best estimator
best_dt = grid_dt.best_estimator_


# Predictions
y_pred_dt = best_dt.predict(X_test)
y_proba_dt = best_dt.predict_proba(X_test)[:,1]

# Evaluation
print(classification_report(y_test, y_pred_dt))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_dt))


# Probabilities from baseline and tuned Decision Tree
y_proba_dt_base = dt.predict_proba(X_test)[:,1]          # baseline DT
y_proba_dt_tuned = best_dt.predict_proba(X_test)[:,1]    # tuned DT (from GridSearchCV)

# Generate thresholds
thresholds = np.linspace(0, 1, 100)

# Accuracy lists
accuracies_base = []
accuracies_tuned = []

# Calculate accuracy for each threshold
for thresh in thresholds:
    y_pred_base = (y_proba_dt_base >= thresh).astype(int)
    y_pred_tuned = (y_proba_dt_tuned >= thresh).astype(int)
    
    accuracies_base.append(accuracy_score(y_test, y_pred_base))
    accuracies_tuned.append(accuracy_score(y_test, y_pred_tuned))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, accuracies_base, color='blue', linewidth=2, label='Baseline Decision Tree')
plt.plot(thresholds, accuracies_tuned, color='orange', linewidth=2, label='Tuned Decision Tree')
plt.title('Accuracy vs Classification Threshold (Decision Tree)')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()



# --- Baseline Decision Tree ---
prec_dt_base, rec_dt_base, _ = precision_recall_curve(y_test, y_proba_dt_base)
ap_dt_base = average_precision_score(y_test, y_proba_dt_base)

# --- Tuned Decision Tree ---
prec_dt_tuned, rec_dt_tuned, _ = precision_recall_curve(y_test, y_proba_dt_tuned)
ap_dt_tuned = average_precision_score(y_test, y_proba_dt_tuned)

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(rec_dt_base, prec_dt_base, label=f'Baseline Decision Tree (AP={ap_dt_base:.2f})', color='blue')
plt.plot(rec_dt_tuned, prec_dt_tuned, label=f'Tuned Decision Tree (AP={ap_dt_tuned:.2f})', color='orange')

plt.title('Precision-Recall Curves (Decision Tree)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.show()



# --- Baseline Decision Tree ---
fpr_base, tpr_base, _ = roc_curve(y_test, y_proba_dt_base)
auc_base = roc_auc_score(y_test, y_proba_dt_base)

# --- Tuned Decision Tree ---
fpr_tuned, tpr_tuned, _ = roc_curve(y_test, y_proba_dt_tuned)
auc_tuned = roc_auc_score(y_test, y_proba_dt_tuned)

# --- Plot ---
plt.figure(figsize=(8,6))
plt.plot(fpr_base, tpr_base, label=f'Baseline Decision Tree (AUC={auc_base:.2f})', color='blue')
plt.plot(fpr_tuned, tpr_tuned, label=f'Tuned Decision Tree (AUC={auc_tuned:.2f})', color='orange')
plt.plot([0,1],[0,1],'k--')  # diagonal line for random guessing

plt.title('ROC Curve Comparison (Decision Tree)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()



param_dist_gbm = {
    'n_estimators': np.arange(50, 301, 50),      # 50 to 300 in steps of 50
    'learning_rate': [0.01, 0.05, 0.1, 0.2],     # shrinkage rate
    'max_depth': [3, 4, 5],                      # depth of individual trees
    'subsample': [0.7, 0.8, 1.0]                 # fraction of samples
}


# Initialize RandomizedSearchCV
rand_gbm = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=param_dist_gbm,
    n_iter=20,             # number of random combinations to try
    cv=5,                  # 5-fold cross-validation
    scoring='roc_auc',     # optimize for ROC-AUC
    n_jobs=-1,             # parallelize across cores
    random_state=42
)


# Fit model
rand_gbm.fit(X_train, y_train)


# Best parameters
print("Best parameters:", rand_gbm.best_params_)


# Best estimator
best_gbm = rand_gbm.best_estimator_


# Predictions
y_pred_gbm = best_gbm.predict(X_test)
y_proba_gbm = best_gbm.predict_proba(X_test)[:,1]

# Evaluation
print(classification_report(y_test, y_pred_gbm))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_gbm))


# --- Baseline Decision Tree ---
fpr_base, tpr_base, _ = roc_curve(y_test, y_proba_dt_base)
auc_base = roc_auc_score(y_test, y_proba_dt_base)

# --- Tuned Decision Tree ---
fpr_tuned, tpr_tuned, _ = roc_curve(y_test, y_proba_dt_tuned)
auc_tuned = roc_auc_score(y_test, y_proba_dt_tuned)

# --- GBM ---
fpr_gbm, tpr_gbm, _ = roc_curve(y_test, y_proba_gbm)
auc_gbm = roc_auc_score(y_test, y_proba_gbm)

# --- Plot ---
plt.figure(figsize=(8,6))
plt.plot(fpr_base, tpr_base, label=f'DT Baseline (AUC={auc_base:.2f})', color='blue', linestyle='--')
plt.plot(fpr_tuned, tpr_tuned, label=f'DT Tuned (AUC={auc_tuned:.2f})', color='orange', linestyle='--')
plt.plot(fpr_gbm, tpr_gbm, label=f'GBM (AUC={auc_gbm:.2f})', color='green')
plt.plot([0,1],[0,1],'k--')

plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()



# Define parameter distributions
param_dist_lgbm = {
    'n_estimators': np.arange(100, 501, 100),     # number of boosting rounds
    'learning_rate': [0.01, 0.05, 0.1, 0.2],      # shrinkage rate
    'num_leaves': [31, 63, 127],                  # complexity of trees
    'max_depth': [-1, 5, 10],                     # -1 means no limit
    'subsample': [0.7, 0.8, 1.0],                 # fraction of samples
    'colsample_bytree': [0.7, 0.8, 1.0]           # fraction of features
}


# Initialize LightGBM classifier with imbalance handling
lgbm_clf = lgb.LGBMClassifier(
    random_state=42,
    class_weight='balanced'  # handles class imbalance automatically
)


# RandomizedSearchCV
rand_lgbm = RandomizedSearchCV(
    estimator=lgbm_clf,
    param_distributions=param_dist_lgbm,
    n_iter=20,             # number of random combinations
    cv=5,                  # 5-fold CV
    scoring='roc_auc',     # optimize for ROC-AUC
    n_jobs=-1,
    random_state=42
)


# Fit model
rand_lgbm.fit(X_train, y_train)


# Best parameters
print("Best parameters:", rand_lgbm.best_params_)


# Best estimator
best_lgbm = rand_lgbm.best_estimator_


# Predictions
y_pred_lgbm = best_lgbm.predict(X_test)
y_proba_lgbm = best_lgbm.predict_proba(X_test)[:,1]

# Evaluation
print(classification_report(y_test, y_pred_lgbm))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_lgbm))


# Probabilities from LightGBM
y_proba = y_proba_lgbm  # predicted probabilities for class 1
thresholds = np.arange(0.1, 0.91, 0.05)  # range of thresholds

precisions, recalls, f1s = [], [], []

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))

# --- Plot Precision, Recall, F1 vs Threshold ---
plt.figure(figsize=(10,6))
plt.plot(thresholds, precisions, label="Precision", marker='o')
plt.plot(thresholds, recalls, label="Recall", marker='o')
plt.plot(thresholds, f1s, label="F1-score", marker='o')

plt.title("Threshold Tuning for LightGBM")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()

# --- Best threshold by F1 ---
best_idx = np.argmax(f1s)
print(f"Best threshold = {thresholds[best_idx]:.2f}")
print(f"Precision = {precisions[best_idx]:.2f}, Recall = {recalls[best_idx]:.2f}, F1 = {f1s[best_idx]:.2f}")



import joblib

# Save the tuned LightGBM model
joblib.dump(best_lgbm, "best_lgbm_model.pkl")

