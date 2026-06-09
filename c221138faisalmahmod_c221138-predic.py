import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import xgboost as xgb


# Step 2: Load datasets
train_df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")
test_df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/test_dataset_exam.csv")
id_column = test_df['id'].copy()


# Step 3: Drop irrelevant columns
drop_cols = ['Unnamed: 0', 'id']
train_df.drop(columns=drop_cols, errors='ignore', inplace=True)
test_df.drop(columns=drop_cols, errors='ignore', inplace=True)
test_df.drop(columns=['Unnamed: 0', 'id'], errors='ignore', inplace=True)


# Step 4: Label encode categorical columns
categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    label_encoders[col] = le

# Encode test set categorical columns (excluding target)
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    if col in label_encoders:
        test_df[col] = label_encoders[col].transform(test_df[col].astype(str))


# Step 5: Separate features and target
X = train_df.drop('satisfaction', axis=1)
y = train_df['satisfaction']


# Step 6: Impute missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(test_df), columns=test_df.columns)


# Step 7: Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)


# Step 9: Evaluate on validation set
val_preds = model.predict(X_val)
accuracy = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {accuracy:.4f}")


# Step 10: Predict on test set
test_preds = model.predict(X_test)


# Step 11: Convert predictions back to original labels
test_df['satisfaction'] = label_encoders['satisfaction'].inverse_transform(test_preds)


# Create submission DataFrame using saved 'id' column
submission = pd.DataFrame({
    'ID': id_column,
    'satisfaction': test_df['satisfaction']
})

submission.to_csv("submission.csv", index=False)
print(submission.head())

