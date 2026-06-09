import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

# Quick look at the data
train.head()

# Check for missing values
train.isnull().sum()


# Convert yes/no to 0/1
binary_cols = ['default', 'housing', 'loan']
for col in binary_cols:
    train[col] = train[col].map({'no': 0, 'yes': 1})
    test[col] = test[col].map({'no': 0, 'yes': 1})

# Convert month names to numbers
month_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
train['month'] = train['month'].map(month_map)
test['month'] = test['month'].map(month_map)

# Encode remaining categorical columns
cat_cols = ['job', 'marital', 'education', 'contact', 'poutcome']
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train[cat_cols] = encoder.fit_transform(train[cat_cols])
test[cat_cols] = encoder.transform(test[cat_cols])

print(train.head())


features = ['id', 'age', 'job', 'marital', 'education', 'default', 
            'balance', 'housing', 'loan', 'contact', 'day', 'month', 
            'duration', 'campaign', 'pdays', 'previous', 'poutcome']

X = train[features]
y = train['y']
X_test = test[features]


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Random Forest Validation Accuracy: {acc:.4f}")


test_predictions = model.predict(X_test)


submission = pd.DataFrame({
    'id': test['id'],
    'y': test_predictions
})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'.")

