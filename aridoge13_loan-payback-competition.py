import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import shap
import os

# Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")



# Define Categoricals
cat_cols = [
    "employment_status",
    "gender",
    "marital_status",
    "education_level",
    "loan_purpose",
    "grade_subgrade"
]

# Handle Missing Values
for col in train.columns:
    if col == "id" or col == "loan_paid_back":
        continue  # Skip ID and target column
        
    if train[col].dtype == "object":
        train[col] = train[col].fillna("Missing")
    else:
        train[col] = train[col].fillna(train[col].median())

for col in test.columns:
    if col == "id":
        continue  # Skip ID column
        
    if test[col].dtype == "object":
        test[col] = test[col].fillna("Missing")
    else:
        # Use median from training data to avoid data leakage
        median_val = train[col].median()
        test[col] = test[col].fillna(median_val)


# Encode Categoricals
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    # handle unseen categories in test
    test[col] = test[col].astype(str).map(lambda s: s if s in le.classes_ else "Unknown")
    if "Unknown" not in le.classes_:
        le.classes_ = np.append(le.classes_, "Unknown")
    test[col] = le.transform(test[col])
    encoders[col] = le

# Separate Features and Target 
X = train.drop(columns=["id", "loan_paid_back"])
y = train["loan_paid_back"] 

# Train / Validation Split 
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
) 


# Random Forest Classifier 
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Validation Performance
preds = rf.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, preds)
ap = average_precision_score(y_valid, preds)

print(f"Validation ROC-AUC: {auc:.4f}")
print(f"Validation PR-AUC:  {ap:.4f}")


# Visualisation of the data
sns.countplot(x="loan_paid_back", data=train)
plt.title("Distribution of Loan Repayment vs Default")
plt.show() 

sns.histplot(data=train, x="credit_score", hue="loan_paid_back", kde=True, bins=30)
plt.title("Credit Score Distribution by Repayment Status")
plt.show()

# Feature interpretibility
sample = X_valid.sample(n=min(100, len(X_valid)), random_state=42)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(sample)
shap.summary_plot(shap_values[1], sample, plot_type="bar")
shap.summary_plot(shap_values[1], sample)

# Save
os.makedirs("artifacts", exist_ok=True)
joblib.dump(rf, "artifacts/rf_model.pkl")
joblib.dump(encoders, "artifacts/label_encoders.pkl")
print("Model and encoders saved to artifacts/")


# Predict on test data for submission
X_test = test.drop(columns=["id"])
test_preds = rf.predict_proba(X_test)[:, 1]

# Build submission
submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": test_preds
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully")



