import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import joblib 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
import shap 
import os
import warnings 



warnings.filterwarnings("ignore")


# Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv") 
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")



# Define Categoricals
cat_cols = ["employment_status",
    "gender",
    "marital_status",
    "education_level",
    "loan_purpose",
    "grade_subgrade"]


# Handle Missing Values

for col in train.columns:
    if col == "id" or col == "loan_paid_back":
        continue
    if train[col].dtype == "object":
        train[col] = train[col].fillna("Missing")
    else:
        train[col] = train[col].fillna(train[col].median())

for col in test.columns:
    if col == "id":
        continue
    if test[col].dtype == "object":
        test[col] = test[col].fillna("Missing")
    else:
        median_val = train[col].median()
        test[col] = test[col].fillna(median_val)



# Encode Categoricals
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = test[col].astype(str).map(lambda s: s if s in le.classes_ else "Unknown")
    if "Unknown" not in le.classes_:
        le.classes_ = np.append(le.classes_, "Unknown")
    test[col] = le.transform(test[col])
    encoders[col] = le
    

# Feature Engineering
print("Features engineered:")
train['monthly_income'] = (train['annual_income'] / 12)
train['income_to_loan_ratio'] = 1/ train['debt_to_income_ratio']
train['credit_to_dti'] = train ['credit_score'] * np.log1p(train ['debt_to_income_ratio'])
train['monthly_loan_payment'] = (train['loan_amount']/12) + ((train['loan_amount'] * train['interest_rate'])/12)
train['credit_income_power'] = train['credit_score'] * np.log1p(train['annual_income'])
train['log_income'] = np.log1p(train['annual_income'])
train['log_loan'] = np.log1p(train['loan_amount'])
train['debt_interest_pressure'] = train['debt_to_income_ratio'] * train['interest_rate']
train['income_leverage'] = np.log1p(train['annual_income']) / np.log1p(train['loan_amount'])
train['interest_income_ratio'] = train['interest_rate'] / (train['annual_income'] + 1e-6)
train['loan_to_credit_ratio'] = train['loan_amount'] / (train['credit_score'] + 1e-6)
train['stress_factor'] = train['debt_to_income_ratio'] * train['loan_to_credit_ratio']





test['monthly_income'] = (test ['annual_income']/12)
test['income_to_loan_ratio'] = 1/ test['debt_to_income_ratio']
test['credit_to_dti'] = test ['credit_score'] * np.log1p(test['debt_to_income_ratio'])
test['monthly_loan_payment'] = (test['loan_amount']/12) + ((test['loan_amount'] * test['interest_rate'])/12)
test['credit_income_power'] = test['credit_score'] * np.log1p(test['annual_income'])
test['log_income'] = np.log1p(test['annual_income'])
test['log_loan'] = np.log1p(test['loan_amount'])
test['debt_interest_pressure'] = test['debt_to_income_ratio'] * test['interest_rate']
test['income_leverage'] = np.log1p(test['annual_income']) / np.log1p(test['loan_amount'])
test['interest_income_ratio'] = test['interest_rate'] / (test['annual_income'] + 1e-6)
test['loan_to_credit_ratio'] = test['loan_amount'] / (test['credit_score'] + 1e-6)
test['stress_factor'] = test['debt_to_income_ratio'] * test['loan_to_credit_ratio']



# Feature and Target Separation
X = X = train.drop(columns=["id", "loan_paid_back"])
y = train["loan_paid_back"]


# Test features
X_test = test.drop(columns=["id"])

# Stratified K fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"\n===== Fold {fold + 1} =====")
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    print(f"Training set: {X_train.shape}, Validation set: {X_valid.shape}")

    model = xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=4,
        gamma=0.2,
        reg_alpha=0.3,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        tree_method="hist"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=200,
        verbose=False
    )

    # Store fold predictions
    oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

# Evaluate cross-validated performance
cv_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall CV ROC-AUC: {cv_auc:.4f}")

# Precision-Recall AUC
cv_pr = average_precision_score(y, oof_preds)
print(f"Overall CV PR-AUC: {cv_pr:.4f}")






# Feature Importance
plt.figure(figsize=(10, 8))
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

sns.barplot(data=feature_importance, y='feature', x='importance', orient='h')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()



#SHAP Analysis
print("Running SHAP analysis...")
sample = X_valid.sample(n=min(1000, len(X_valid)), random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer(sample)

shap.summary_plot(shap_values, sample, plot_type="bar")
shap.summary_plot(shap_values, sample)

# Save Model
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/xgb_model.pkl")
joblib.dump(encoders, "artifacts/label_encoders.pkl")
print("XGBoost model and encoders saved to artifacts/")

# Predict on Test Data
X_test = test.drop(columns=["id"])
test_preds = model.predict_proba(X_test)[:, 1]

# Create Submission
submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": test_preds
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully!")

