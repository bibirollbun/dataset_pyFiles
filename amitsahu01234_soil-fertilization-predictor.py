import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# Save test IDs
test_ids = test['id']

# -----------------------------
# Encode categorical features
# -----------------------------
label_encoders = {}
categorical_cols = ['Soil Type', 'Crop Type']  # known from your dataset

for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])  # test has same columns

# Encode target column
target_encoder = LabelEncoder()
train['Fertilizer Name'] = target_encoder.fit_transform(train['Fertilizer Name'])

# -----------------------------
# Prepare training data
# -----------------------------
X = train.drop(columns=['id', 'Fertilizer Name'])
y = train['Fertilizer Name']
X_test = test.drop(columns=['id'])

# Split for validation (optional)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train XGBoost model
# -----------------------------
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Predict top 3 fertilizers
# -----------------------------
probs = model.predict_proba(X_test)
top_3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]

# Convert label indices to fertilizer names
top_3_names = [target_encoder.inverse_transform(row).tolist() for row in top_3]
predictions = [' '.join(row) for row in top_3_names]

# -----------------------------
# Create submission file
# -----------------------------
submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': predictions
})

submission.to_csv('submission.csv', index=False)
print("✅ Submission file 'submission.csv' created.")





