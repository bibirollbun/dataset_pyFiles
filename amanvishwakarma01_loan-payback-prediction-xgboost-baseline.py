#ğŸ�¯ Goal: Predict probability that a borrower pays back their loan
# ğŸ§  Model: XGBoost (Baseline)


# 1ï¸�âƒ£ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


# 2ï¸�âƒ£ Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


print("Train shape:", train.shape)
print("Test shape:", test.shape)


train.head()


# 3ï¸�âƒ£ Check for Missing Values
missing = train.isnull().sum().sort_values(ascending=False)
print("\nMissing values:\n", missing[missing > 0])


# 4ï¸�âƒ£ Target Distribution
plt.figure(figsize=(4,3))
sns.countplot(x='loan_paid_back', data=train, palette='Set2')
plt.title("Target Distribution (Loan Paid Back)")
plt.show()


# 5ï¸�âƒ£ Encode Categorical Features
label_encoders = {}
categorical_cols = ['gender', 'marital_status', 'education_level', 
                    'employment_status', 'loan_purpose', 'grade_subgrade']

for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le


# 6ï¸�âƒ£ Define Features and Target
X = train.drop(columns=['id', 'loan_paid_back'])
y = train['loan_paid_back']


# 7ï¸�âƒ£ Train XGBoost Model (Fast Bronze Setup)
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='auc',
    random_state=42,
    use_label_encoder=False
)

xgb_model.fit(X, y)


# 8ï¸�âƒ£ Predict Probabilities on Test Set
test_pred = xgb_model.predict_proba(test.drop(columns=['id']))[:, 1]


# 9ï¸�âƒ£ Create Submission File
submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': test_pred
})
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as submission.csv")


# ğŸ”Ÿ Feature Importance
plt.figure(figsize=(8,5))
sns.barplot(
    x=xgb_model.feature_importances_, 
    y=X.columns, 
    palette="viridis"
)
plt.title("Feature Importance (XGBoost)")
plt.show()




