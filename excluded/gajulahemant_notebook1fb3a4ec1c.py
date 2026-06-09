import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd

train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

print(train.shape)
print(test.shape)


print(train.info())
print(train.describe())
print(train.isnull().sum())


from sklearn.preprocessing import LabelEncoder

# ✅ Fill missing numeric values with median
for col in train.select_dtypes(include=['int64', 'float64']).columns:
    if col in test.columns:   # check column exists in test
        train[col] = train[col].fillna(train[col].median())
        test[col]  = test[col].fillna(train[col].median())  # use train's median

# ✅ Encode categorical (object) columns safely
label_encoders = {}
for col in train.select_dtypes(include=['object']).columns:
    if col in test.columns:   # make sure column exists in test
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))
        label_encoders[col] = le

print("✅ Preprocessing Done (no more errors)")


from sklearn.model_selection import train_test_split

# ✅ Use "y" as target column (if your dataset has a different name, we will adjust)
X = train.drop("y", axis=1)
y = train["y"]

# Split into training (80%) and validation (20%)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Data Split Done")
print("Train shape:", X_train.shape, y_train.shape)
print("Validation shape:", X_valid.shape, y_valid.shape)


import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create model
model = lgb.LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=500,
    random_state=42
)

# Train with early stopping using callback
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric='binary_error',
    callbacks=[lgb.early_stopping(stopping_rounds=50), 
               lgb.log_evaluation(50)]
)

# Predict on validation set
y_pred = model.predict(X_valid)

# Evaluate
print("✅ Accuracy:", accuracy_score(y_valid, y_pred))
print("\nClassification Report:\n", classification_report(y_valid, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_valid, y_pred))


# Predict on test data
test_preds = model.predict(test)

# Create submission DataFrame
# ⚠ Replace "id" with the actual ID column name from your test.csv
submission = pd.DataFrame({
    "id": test["id"],       # unique ID column
    "target": test_preds    # predicted target
})

# Save to CSV
submission.to_csv("submission.csv", index=False)

print("✅ Submission file created: submission.csv")

