import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Load datasets
train_df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")
test_df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/test_dataset_exam.csv")

# Drop Unnamed columns
train_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
test_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')

# Combine for label encoding
combined = pd.concat([train_df.drop(columns=['satisfaction']), test_df], axis=0)
categorical_cols = combined.select_dtypes(include='object').columns

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
    label_encoders[col] = le

# Restore processed columns
train_df[categorical_cols] = combined.iloc[:len(train_df)][categorical_cols]
test_df[categorical_cols] = combined.iloc[len(train_df):][categorical_cols]

# Encode target variable
le_target = LabelEncoder()
train_df['satisfaction'] = le_target.fit_transform(train_df['satisfaction'])
label_encoders['satisfaction'] = le_target

# Split training data
X = train_df.drop(columns=['satisfaction', 'id'], errors='ignore')
y = train_df['satisfaction']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Validation predictions
val_preds = model.predict(X_val)
val_proba = model.predict_proba(X_val)
acc = accuracy_score(y_val, val_preds)
print(f"\nAccuracy: {acc:.4f}")


X_test = test_df.drop(columns=['id'], errors='ignore')
test_preds = model.predict(X_test)
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'satisfaction': le_target.inverse_transform(test_preds)
})
submission_df.to_csv("submission.csv", index=False)
print("✅ Submission file saved as submission.csv")


# Create solution DataFrame for submission
solution = test_df.copy()  # keep 'id' column

# Inverse transform integer predictions to original labels
solution['satisfaction'] = le_target.inverse_transform(test_preds)

# Rename 'id' column to 'ID'
solution.rename(columns={'id': 'ID'}, inplace=True)

# Save only required columns
solution[['ID', 'satisfaction']].to_csv("submission.csv", index=False)

# Display first few rows to verify
print(solution.head())

