# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# -------------------------------
# Step 1: Import Libraries
# -------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# -------------------------------
# Step 2: Load Dataset
# -------------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

print("âœ… Data Loaded Successfully")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# -------------------------------
# Step 3: Identify Target Column Automatically
# -------------------------------
print("\nAvailable Columns:\n", train.columns.tolist())

# Try to find column name that represents the target
possible_targets = ['Deposit', 'deposit', 'target', 'y', 'subscribed', 'deposit_subscribed']
target = None
for col in possible_targets:
    if col in train.columns:
        target = col
        break

if target is None:
    raise ValueError("âš ï¸� Target column not found. Please check dataset column names manually.")
else:
    print(f"\nâœ… Target column detected: '{target}'")

print("\nTarget Value Counts:")
print(train[target].value_counts())

# -------------------------------
# Step 4: Quick EDA
# -------------------------------
plt.figure(figsize=(5, 4))
sns.countplot(x=target, data=train, palette="mako")
plt.title("Target Variable Distribution")
plt.show()

# Numerical correlation heatmap
num_cols = train.select_dtypes(include=["int64", "float64"]).columns.tolist()
if "id" in num_cols:
    num_cols.remove("id")

plt.figure(figsize=(10, 7))
sns.heatmap(train[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# Step 5: Define X and y
# -------------------------------
id_col = "id"
X = train.drop(columns=[id_col, target])
y = train[target].map(lambda x: 1 if str(x).lower() in ['yes', 'y', 'true', '1'] else 0)

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumerical Columns:", num_cols)
print("Categorical Columns:", cat_cols)

# -------------------------------
# Step 6: Preprocessing
# -------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# -------------------------------
# Step 7: Model Setup
# -------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                           ("model", model)])

# -------------------------------
# Step 8: Cross Validation
# -------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
print("\nCross-Validation Accuracy Scores:", scores)
print("Mean CV Accuracy: {:.4f}".format(scores.mean()))

# -------------------------------
# Step 9: Train-Validation Split
# -------------------------------
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_valid)

print("\nValidation Accuracy: {:.4f}".format(accuracy_score(y_valid, y_pred)))
print("\nClassification Report:\n", classification_report(y_valid, y_pred))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_valid, y_pred), annot=True, fmt="d", cmap="crest")
plt.title("Confusion Matrix (Validation Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------
# Step 10: Train on Full Data
# -------------------------------
pipeline.fit(X, y)
X_test = test.drop(columns=[id_col])
test_preds = pipeline.predict(X_test)

# -------------------------------
# Step 11: Submission File
# -------------------------------
submission = pd.DataFrame({
    id_col: test[id_col],
    target: np.where(test_preds == 1, "Yes", "No")
})

submission.to_csv("submission.csv", index=False)
print("\nğŸ“� submission.csv created successfully!")
print(submission.head())

# -------------------------------
# Step 12: Feature Importance
# -------------------------------
model_final = pipeline.named_steps["model"]
encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
encoded_features = encoder.get_feature_names_out(cat_cols)
all_features = np.concatenate([num_cols, encoded_features])

importances = model_final.feature_importances_
fi = pd.DataFrame({"Feature": all_features, "Importance": importances})
fi = fi.sort_values(by="Importance", ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=fi, palette="viridis")
plt.title("Top 15 Important Features")
plt.show()

print("\nğŸ”¥ Top Important Features:")
print(fi)


