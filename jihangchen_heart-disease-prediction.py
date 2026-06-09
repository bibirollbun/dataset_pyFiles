import os
import warnings 
import pandas as pd
import numpy as np
import xgboost as xgb

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score


df = pd.read_csv('/kaggle/input/heart-disease-prediction-dataquest/heart_train.csv')
test = pd.read_csv('/kaggle/input/heart-disease-prediction-dataquest/heart_test.csv')


df.head()


df.info()


# Check missing data

df.isnull().sum()


# Descriptive statistics

df.describe()


# Check whether dataset is balance

sns.countplot(x='HeartDisease', data=df)
plt.title("Heart Disease vs. No Disease")


# Feature distribution

numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

df[numeric_cols].hist(bins=20, figsize=(12, 8))
plt.suptitle("Histograms of Numeric Features")


# Check the skewness

categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, hue='HeartDisease', data=df, ax=axes[i])
    axes[i].set_title(f"{col} vs Heart Disease")
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.suptitle("Categorical Features vs Heart Disease", fontsize=16, y=1.03)
plt.show()


df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
slope_mapping = {'Up': 0, 'Flat': 1, 'Down': 2}
df['ST_Slope'] = df['ST_Slope'].map(slope_mapping)
df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG'], drop_first=True).astype(int)


df.head()


X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


cols_with_outliers = ['RestingBP', 'Cholesterol', 'MaxHR']

# Copy the data to avoid changing original
X_train_clean = X_train.copy()
y_train_clean = y_train.copy()

# Loop through each column and filter out outliers
for col in cols_with_outliers:
    Q1 = X_train_clean[col].quantile(0.2)
    Q3 = X_train_clean[col].quantile(0.8)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Keep only rows within bounds
    mask = (X_train_clean[col] >= lower_bound) & (X_train_clean[col] <= upper_bound)
    X_train_clean = X_train_clean[mask]
    y_train_clean = y_train_clean[mask]

# Final shapes after outlier removal
print(f"Original X_train shape: {X_train.shape}")
print(f"Cleaned X_train shape: {X_train_clean.shape}")



from statsmodels.stats.outliers_influence import variance_inflation_factor

# Check multicolinearity
plt.figure(figsize=(10, 6))
sns.heatmap(X_train_clean.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

vif = pd.DataFrame()
vif["feature"] = X_train_clean.columns
vif["VIF"] = [variance_inflation_factor(X_train_clean.values, i) for i in range(X_train_clean.shape[1])]
vif


scaler = StandardScaler()
X_train_clean_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, random_state=42)

# Fit on scaled data (already done earlier)
logreg.fit(X_train_clean_scaled, y_train_clean)

# Predict
y_pred = logreg.predict(X_test_scaled)
y_prob = logreg.predict_proba(X_test_scaled)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision_sco = average_precision_score(y_test, y_pred)
recall_sco = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
precision, recall_pr, _ = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

# Axes
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# 2. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
axes[0, 1].plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}', color='blue')
axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random guess')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].legend(loc='lower right')
axes[0, 1].grid(True)

# 3. Feature Importance (via coefficient magnitude)
coefs = pd.Series(logreg.coef_[0], index=X.columns)
coefs.abs().sort_values().plot(kind='barh', ax=axes[1, 0])
axes[1, 0].set_title('Feature Importance (|Coefficient|)')
axes[1, 0].set_xlabel('Absolute Coefficient Value')
axes[1, 0].grid(True)

# 4. Precision-Recall Curve
axes[1, 1].plot(recall_pr, precision, label=f'AP = {avg_precision:.4f}', color='green')
axes[1, 1].set_title('Precision-Recall Curve')
axes[1, 1].set_xlabel('Recall')
axes[1, 1].set_ylabel('Precision')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Print key metrics
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision_sco:.4f}")
print(f"Recall       : {recall_sco:.4f}")
print(f"F1-score     : {f1:.4f}")
print(f"ROC-AUC      : {roc_auc:.4f}")


# Train the model
modelRF = RandomForestClassifier(n_estimators=100, random_state=42)
modelRF.fit(X_train_clean, y_train_clean)

# Predictions
y_pred = modelRF.predict(X_test)
y_prob = modelRF.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision_val = precision_score(y_test, y_pred)
recall_val = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

# Precision-Recall curve values
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)

# ROC values
fpr, tpr, _ = roc_curve(y_test, y_prob)

# Visualization
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# 2. ROC Curve
axes[0, 1].plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}', color='blue')
axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random guess')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].legend(loc='lower right')
axes[0, 1].grid(True)

# 3. Feature Importance (for Random Forest)
importances = pd.Series(modelRF.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', ax=axes[1, 0])
axes[1, 0].set_title('Feature Importance')
axes[1, 0].set_xlabel('Importance Score')
axes[1, 0].grid(True)

# 4. Precision-Recall Curve
axes[1, 1].plot(recall_curve, precision_curve, label=f'AP = {avg_precision:.4f}', color='green')
axes[1, 1].set_title('Precision-Recall Curve')
axes[1, 1].set_xlabel('Recall')
axes[1, 1].set_ylabel('Precision')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Print key metrics
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision_val:.4f}")
print(f"Recall       : {recall_val:.4f}")
print(f"F1-score     : {f1:.4f}")
print(f"ROC-AUC      : {roc_auc:.4f}")


from xgboost import XGBClassifier
# Train the XGBoost model
modelxgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
modelxgb.fit(X_train_clean, y_train_clean)

# Predictions
y_pred = modelxgb.predict(X_test)
y_prob = modelxgb.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision_val = precision_score(y_test, y_pred)
recall_val = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

# Curves
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

# Visualization
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# 2. ROC Curve
axes[0, 1].plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}', color='blue')
axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random guess')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].legend(loc='lower right')
axes[0, 1].grid(True)

# 3. Feature Importance
importances = pd.Series(modelxgb.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', ax=axes[1, 0])
axes[1, 0].set_title('Feature Importance (XGBoost)')
axes[1, 0].set_xlabel('Importance Score')
axes[1, 0].grid(True)

# 4. Precision-Recall Curve
axes[1, 1].plot(recall_curve, precision_curve, label=f'AP = {avg_precision:.4f}', color='green')
axes[1, 1].set_title('Precision-Recall Curve')
axes[1, 1].set_xlabel('Recall')
axes[1, 1].set_ylabel('Precision')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Print key metrics
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision_val:.4f}")
print(f"Recall       : {recall_val:.4f}")
print(f"F1-score     : {f1:.4f}")
print(f"ROC-AUC      : {roc_auc:.4f}")


test['Sex'] = test['Sex'].map({'M': 1, 'F': 0})
test['ExerciseAngina'] = test['ExerciseAngina'].map({'Y': 1, 'N': 0})
test['ST_Slope'] = test['ST_Slope'].map(slope_mapping)
test = pd.get_dummies(test, columns=['ChestPainType', 'RestingECG'], drop_first=True).astype(int)
test.head()


test_scaled = scaler.transform(test)
pred_logreg = logreg.predict(test_scaled)
pred_RF = modelRF.predict(test)
pred_xgb = modelxgb.predict(test)


# Combine predictions into a DataFrame
pred_df = pd.DataFrame({
    'Logistic Regression': pred_logreg,
    'Random Forest': pred_RF,
    'XGBoost': pred_xgb
})

# Compute correlation matrix
corr_matrix = pred_df.corr()

# Plot heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Model Predictions")
plt.tight_layout()
plt.show()


pred_stack = np.vstack([pred_logreg, pred_RF, pred_xgb]).T

# Apply majority voting across axis=1 (rows)
final_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=pred_stack)


submission = pd.DataFrame({
    'id': range(0, 184),
    'target': final_vote
})
submission.to_csv('submission.csv', index=False)

