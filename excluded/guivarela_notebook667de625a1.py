import pandas as pd

# Load datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s4e2/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s4e2/test.csv")
sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s4e2/sample_submission.csv")


# Basic info
print(train_df.info())
print(train_df.head())

# Class distribution
print(train_df['NObeyesdad'].value_counts())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Separate features and target
X = train_df.drop(columns=["id", "NObeyesdad"])
y = train_df["NObeyesdad"]
X_test = test_df.drop(columns=["id"])

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# Model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split training data
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Evaluate on validation set
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")



# Predict on test set
test_preds = model.predict(X_test)

# Format for submission
submission_df = pd.DataFrame({
    "id": test_df["id"],
    "NObeyesdad": test_preds
})

# Save to CSV
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")


