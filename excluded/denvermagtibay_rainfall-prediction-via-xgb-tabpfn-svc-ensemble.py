import pandas as pd
import numpy as np

# Load the synthetic train/test sets
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Load the original dataset (assuming it's available in this path)
orig = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Original shape:", orig.shape)
print("Train sample:\n", train.head())
print("Test sample:\n", test.head())
print("Orig sample:\n", orig.head())


print("train columns:", train.columns.tolist())
print("orig columns:", orig.columns.tolist())


# Fix all specific column name issues in one go
orig = orig.rename(columns={
    'pressure ': 'pressure',
    'temparature': 'temperature',
    'humidity ': 'humidity',
    'cloud ': 'cloud',
    '         winddirection': 'winddirection'
})
train = train.rename(columns={'temparature': 'temperature'})


feature_cols = [col for col in train.columns if col not in ['id', 'group']]
orig = orig[feature_cols]


# âœ… Step 0: Fix typo in train BEFORE everything else
train = train.rename(columns={'temparature': 'temperature'})

# Step 1: Clean original column names
orig.columns = orig.columns.str.strip()

# Step 2: Convert rainfall labels
orig['rainfall'] = orig['rainfall'].map({'yes': 1, 'no': 0})

# Step 3: Select aligned feature columns
feature_cols = [col for col in train.columns if col not in ['id', 'group']]
orig = orig[feature_cols]

# Step 4: Add ID and group
orig['id'] = range(train['id'].max() + 1, train['id'].max() + 1 + len(orig))
orig['group'] = 6

# Step 5: Combine train + original
train_plus_orig = pd.concat([train, orig], ignore_index=True)
train_plus_orig.columns = train_plus_orig.columns.str.strip()

# Step 6: Assign missing group values
train_plus_orig.loc[train_plus_orig['group'].isnull(), 'group'] = train_plus_orig.loc[train_plus_orig['group'].isnull(), 'id'] // 365

# Step 7: Fill missing rainfall in group 6
train_plus_orig.loc[train_plus_orig['group'] == 6, 'rainfall'] = train_plus_orig.loc[train_plus_orig['group'] == 6, 'rainfall'].fillna(1)

# Step 8: Fill any NaNs in key features
for col in ['winddirection', 'windspeed']:
    if train_plus_orig[col].isnull().sum() > 0:
        train_plus_orig[col] = train_plus_orig[col].fillna(train_plus_orig[col].median())

# Final check
print("âœ… Final shape:", train_plus_orig.shape)
print("âœ… Missing values:\n", train_plus_orig.isnull().sum())
print("âœ… Columns:\n", train_plus_orig.columns.tolist())


import seaborn as sns
import matplotlib.pyplot as plt

# Class distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=train_plus_orig, x='rainfall')
plt.title("Rainfall Class Distribution")
plt.xlabel("Rainfall (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.grid(True)
plt.show()


# Only numeric features
plt.figure(figsize=(12, 8))
sns.heatmap(train_plus_orig.corr(), cmap="coolwarm", annot=True, fmt=".2f", square=True)
plt.title("Correlation Heatmap")
plt.show()


features_to_plot = ['temperature', 'humidity', 'pressure', 'windspeed']
for col in features_to_plot:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=train_plus_orig, x='rainfall', y=col)
    plt.title(f"{col} Distribution by Rainfall")
    plt.grid(True)
    plt.show()


import xgboost as xgb

# Prepare data with feature names
X = train_plus_orig.drop(columns=['id', 'rainfall', 'group'])
y = train_plus_orig['rainfall'].astype(int)

# Train XGBoost
xgb_model = xgb.XGBClassifier(
    max_depth=3,
    colsample_bytree=0.9,
    subsample=0.9,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X, y)


importances = xgb_model.feature_importances_
feature_names = X.columns

# Create DataFrame
feat_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='viridis')
plt.title("XGBoost Feature Importance")
plt.grid(True)
plt.tight_layout()
plt.show()


from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib.pyplot as plt

# Predict probabilities on full training data
y_pred_prob = xgb_model.predict_proba(X)[:, 1]
y_true = y

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("XGBoost ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Classification report
y_pred_binary = (y_pred_prob >= 0.5).astype(int)
print("âœ… Classification Report:\n")
print(classification_report(y_true, y_pred_binary))


from sklearn.model_selection import GroupKFold, cross_val_predict

# Data prep
X = train_plus_orig.drop(columns=['id', 'rainfall', 'group'])
y = train_plus_orig['rainfall'].astype(int)
groups = train_plus_orig['group'].astype(int)

# Re-initialize XGBoost model
xgb_model = xgb.XGBClassifier(
    max_depth=3,
    colsample_bytree=0.9,
    subsample=0.9,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

# Get cross-validated prediction probabilities
preds = cross_val_predict(
    xgb_model, X, y,
    cv=GroupKFold(n_splits=6).split(X, y, groups),
    method='predict_proba'
)


from sklearn.metrics import roc_auc_score, classification_report, roc_curve, auc

# AUC
cv_auc = roc_auc_score(y, preds[:, 1])
print(f"âœ… XGBoost CV AUC: {cv_auc:.5f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y, preds[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {cv_auc:.4f})", color='orange')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("XGBoost Cross-Validated ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend()
plt.show()

# Classification report
y_pred_class = (preds[:, 1] >= 0.5).astype(int)
print("âœ… XGBoost CV Classification Report:\n")
print(classification_report(y, y_pred_class))


# Install and import

!pip install tabpfn
from tabpfn import TabPFNClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, classification_report


# Prepare data

X = train_plus_orig.drop(columns=['id', 'rainfall', 'group']).values
y = train_plus_orig['rainfall'].astype(int).values
groups = train_plus_orig['group'].values


# Initialize TabPFN with GPU and large dataset support
tpfn = TabPFNClassifier(ignore_pretraining_limits=True)


# CV prediction
y_proba = cross_val_predict(
    tpfn, X, y,
    cv=GroupKFold(n_splits=6).split(X, y, groups),
    method='predict_proba'
)


# AUC Score
auc_tpfn = roc_auc_score(y, y_proba[:, 1])
print(f"âœ… TabPFN CV AUC: {auc_tpfn:.5f}")

# Classification report
y_pred = (y_proba[:, 1] >= 0.5).astype(int)
print("âœ… TabPFN Classification Report:\n")
print(classification_report(y, y_pred))


from sklearn.metrics import roc_auc_score, classification_report

# Blend predictions (equal weights)
blend_proba = (preds[:, 1] + y_proba[:, 1]) / 2

# AUC score
auc_blend = roc_auc_score(y, blend_proba)
print(f"âœ… Blended XGBoost + TabPFN CV AUC: {auc_blend:.5f}")

# Classification report at threshold 0.5
blend_pred = (blend_proba >= 0.5).astype(int)
print("âœ… Blended Classification Report:\n")
print(classification_report(y, blend_pred))


from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, classification_report

# Prepare data once more
X = train_plus_orig.drop(columns=['id', 'rainfall', 'group']).values
y = train_plus_orig['rainfall'].astype(int).values
groups = train_plus_orig['group'].astype(int).values

# Initialize SVC
svc = SVC(C=0.1, kernel='poly', degree=1, probability=True, random_state=42)

# OOF probability predictions
svc_proba = cross_val_predict(
    svc, X, y,
    cv=GroupKFold(n_splits=6).split(X, y, groups),
    method='predict_proba'
)

# SVC performance
auc_svc = roc_auc_score(y, svc_proba[:, 1])
print(f"âœ… SVC CV AUC: {auc_svc:.5f}")
print("\nâœ… SVC Classification Report:\n")
print(classification_report(y, (svc_proba[:, 1] >= 0.5).astype(int)))


# Blend all three model predictions
blend_all = (preds[:, 1] + y_proba[:, 1] + svc_proba[:, 1]) / 3

# AUC and classification
auc_blend_all = roc_auc_score(y, blend_all)
print(f"âœ… Final Blended CV AUC (XGB + TabPFN + SVC): {auc_blend_all:.5f}")

# Threshold at 0.5
y_pred_blend_all = (blend_all >= 0.5).astype(int)
print("\nâœ… Final Blended Classification Report:\n")
print(classification_report(y, y_pred_blend_all))


# Rename 'temparature' to 'temperature' in test
test = test.rename(columns={'temparature': 'temperature'})

# Reuse feature columns
X_test = test.drop(columns=['id']).values


# Ensure all column names are clean
test.columns = test.columns.str.strip()

# Fix column name
test = test.rename(columns={'temparature': 'temperature'})

# Fill NaNs with median values (consistent with train handling)
for col in test.columns:
    if test[col].isnull().sum() > 0:
        test[col] = test[col].fillna(test[col].median())

# Final test data
X_test = test.drop(columns=['id']).values


# 1. XGBoost
xgb_model = xgb.XGBClassifier(
    max_depth=3,
    colsample_bytree=0.9,
    subsample=0.9,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X, y)
xgb_preds_test = xgb_model.predict_proba(X_test)[:, 1]

# 2. TabPFN
tpfn_final = TabPFNClassifier(ignore_pretraining_limits=True)
tpfn_final.fit(X, y)
tabpfn_preds_test = tpfn_final.predict_proba(X_test)[:, 1]

# 3. SVC
svc_final = SVC(C=0.1, kernel='poly', degree=1, probability=True, random_state=42)
svc_final.fit(X, y)
svc_preds_test = svc_final.predict_proba(X_test)[:, 1]


# Final blend
final_preds = (xgb_preds_test + tabpfn_preds_test + svc_preds_test) / 3

# Submission
submission = pd.DataFrame({'id': test['id'], 'rainfall': final_preds})
submission.to_csv('submission.csv', index=False)
print("âœ… submission.csv saved!")

