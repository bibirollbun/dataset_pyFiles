# Basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Preprocessing
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
import category_encoders as ce

# Model selection & evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, 
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# Machine Learning models
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
train.head()


test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
test.head()


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
sample_submission.head()


TARGET = "y"
ID_COL = "id"

# save test ids for submission
test_ids = test[ID_COL].copy()


print("train shape:", train.shape, "test shape:", test.shape)
print("target distribution:\n", train[TARGET].value_counts(normalize=True))


plt.figure(figsize=(6,4))
sns.countplot(x=train[TARGET])
plt.title("Target Distribution")
plt.xlabel("Target")
plt.ylabel("Count")
plt.show()


train.info()


print("Duplicates:", train.duplicated().sum())
print("\nMissing Values:", train.isna().sum())



train.describe()


# Drop 'duration' from training and test datasets
#train = train.drop(columns=['duration'])
#test = test.drop(columns=['duration'])


num_features = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous','duration']
train[num_features].hist(bins=30, figsize=(12,8),color='Purple')
plt.tight_layout()
plt.show()


# Include target 'y' in numeric features
num_features = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous', 'duration', 'y']

corr_matrix = train[num_features].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='viridis')
plt.title("Correlation Heatmap")
plt.show()


X = train.drop(columns=[ID_COL, TARGET])
X_test = test.drop(columns=[ID_COL])
y = train[TARGET]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


cat_features = [col for col in X.columns if col not in num_features]

for col in cat_features:
    print(f"Value counts for {col}:")
    print(train[col].value_counts())
    print("-"*30)


# Encoding

# Frequency Encoding: for high-cardinality categorical features
freq_encode= ['job', 'month', 'poutcome'] # mostly unknown
for col in freq_encode:
    freq = X_train[col].value_counts()/ len(X_train)
    X_train[col] = X_train[col].map(freq)
    X_val[col] = X_val[col].map(freq)
    X_test[col] = X_test[col].map(freq)

#Label Encoding: for ordinal categorical columns
label_encode = ['default', 'housing', 'loan']
le = LabelEncoder()
for col in label_encode:
    X_train[col] = le.fit_transform(X_train[col])
    X_val[col] = le.transform(X_val[col])
    X_test[col] = le.transform(X_test[col])

#One-Hot Encoding: for nominal categorical features
one_hot_cols = ['marital', 'education', 'contact']
X_train = pd.get_dummies(X_train, columns=one_hot_cols, drop_first=True)
X_val = pd.get_dummies(X_val, columns=one_hot_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=one_hot_cols, drop_first=True)

# Align columns to avoid mismatch between train/test/val
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# Scaling
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

scaler = RobustScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols]   = scaler.transform(X_val[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


print("X_train:", X_train.shape, " X_val:", X_val.shape, "X_test", X_test.shape)


# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train, y_train)

# Random Forest validation predictions
y_val_proba_rf = rf.predict_proba(X_val)[:,1]
y_val_pred_rf = (y_val_proba_rf >= 0.5).astype(int)

# Evaluation
val_auc_rf = roc_auc_score(y_val, y_val_proba_rf)
print("Random Forest Validation AUC:", val_auc_rf)
print("Random Forest Classification Report:\n", classification_report(y_val, y_val_pred_rf))


# Train XGBoost

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='auc',
    n_jobs= -1,
    early_stopping_rounds=50
)

# Fit with early stopping using validation set
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# XGBoost validation predictions
y_val_proba_xgb = xgb_model.predict_proba(X_val)[:,1]
y_val_pred_xgb = (y_val_proba_xgb >= 0.5).astype(int)

# Evaluation
val_auc_xgb = roc_auc_score(y_val, y_val_proba_xgb)
print("XGBoost Validation AUC:", val_auc_xgb)
print("XGBoost Classification Report:\n", classification_report(y_val, y_val_pred_xgb))


# Plot ROC Curves
fpr_rf, tpr_rf, _ = roc_curve(y_val, y_val_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_val, y_val_proba_xgb)

plt.figure(figsize=(8,6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest AUC = {val_auc_rf:.3f}')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost AUC = {val_auc_xgb:.3f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()


# Confusion Matrix

cm_xgb = confusion_matrix(y_val, y_val_pred_xgb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_xgb)
disp.plot(cmap=plt.cm.Blues)
plt.title('XGBoost Confusion Matrix')
plt.show()


# Test Set Predictions
xgb_test_probs = xgb_model.predict_proba(X_test)[:,1]
xgb_test_pred = (xgb_test_probs >= 0.5).astype(int)


# Create submission file
sub_xgb = pd.DataFrame({ID_COL: test_ids, TARGET: xgb_test_probs})
sub_xgb.to_csv("submission.csv", index=False)
print(sub_xgb.head())

