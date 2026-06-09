import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

# Check the first few rows
train.head()


train.isnull().sum()


# Convert yes/no columns to 0/1
binary_cols = ['default', 'housing', 'loan']
for col in binary_cols:
    train[col] = train[col].map({'no': 0, 'yes': 1})
    test[col] = test[col].map({'no': 0, 'yes': 1})

# Convert month names to numeric
month_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
train['month'] = train['month'].map(month_map)
test['month'] = test['month'].map(month_map)

# Encode categorical columns using OrdinalEncoder
# OrdinalEncoder converts categories to numbers. Unknown categories in test data are encoded as -1
cat_cols = ['job', 'marital', 'education', 'contact', 'poutcome']
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train[cat_cols] = encoder.fit_transform(train[cat_cols])
test[cat_cols] = encoder.transform(test[cat_cols])

# Check preprocessed data
train.head()


features = ['id', 'age', 'job', 'marital', 'education', 'default', 
            'balance', 'housing', 'loan', 'contact', 'day', 'month', 
            'duration', 'campaign', 'pdays', 'previous', 'poutcome']

X = train[features]        # Input features
y = train['y']             # Target variable
X_test = test[features]    # Test features for prediction


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=40,
    min_child_samples=30,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.2,
    reg_lambda=2,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)


# Training accuracy
train_predictions = model.predict(X_train)
train_acc = accuracy_score(y_train, train_predictions)
print(f"LightGBM train accuracy: {train_acc:.4f}")

# Validation accuracy
val_predictions = model.predict(X_val)
val_acc = accuracy_score(y_val, val_predictions)
print(f"LightGBM validation accuracy: {val_acc:.4f}")


test_predictions = model.predict(X_test)


submission = pd.DataFrame({
    "id": test["id"],
    "y": test_predictions
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")

