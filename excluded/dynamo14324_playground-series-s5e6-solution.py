import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the datasets
train_df = pd.read_csv("../playground-series-s5e6/train.csv")
test_df = pd.read_csv("../playground-series-s5e6/test.csv")
sample_submission_df = pd.read_csv("../playground-series-s5e6/sample_submission.csv")

# Preprocessing (simple example: fill missing values and one-hot encode categorical features)
# Identify categorical and numerical columns
categorical_cols = train_df.select_dtypes(include='object').columns
numerical_cols = train_df.select_dtypes(include=["int64", "float64"]).columns

# Fill missing numerical values with the mean
for col in numerical_cols:
    if train_df[col].isnull().any():
        train_df[col].fillna(train_df[col].mean(), inplace=True)
    if test_df[col].isnull().any():
        test_df[col].fillna(test_df[col].mean(), inplace=True)

# Fill missing categorical values with the mode
for col in categorical_cols:
    if train_df[col].isnull().any():
        train_df[col].fillna(train_df[col].mode()[0], inplace=True)
    if test_df[col].isnull().any():
        test_df[col].fillna(test_df[col].mode()[0], inplace=True)

# One-hot encode categorical features
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

# Align columns - very important for consistent feature sets
train_labels = train_df["Response"]
train_ids = train_df["id"]
test_ids = test_df["id"]

# Drop target and id columns from feature sets
train_df = train_df.drop(columns=["Response", "id"])
test_df = test_df.drop(columns=["id"])

# Ensure both dataframes have the same columns after one-hot encoding
common_cols = list(set(train_df.columns) & set(test_df.columns))
train_df = train_df[common_cols]
test_df = test_df[common_cols]

# Model Training (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_df, train_labels)

# Make predictions on the test set
predictions = model.predict(test_df)

# Create submission file
submission_df = pd.DataFrame({"id": test_ids, "Response": predictions})
submission_df.to_csv("submission.csv", index=False)

print("Submission file created successfully!")

