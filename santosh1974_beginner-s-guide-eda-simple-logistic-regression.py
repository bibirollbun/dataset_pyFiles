# ------------------------------
# 1. Imports
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


# ------------------------------
# 2. Load Data
# ------------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


# ------------------------------
# 3. Basic EDA
# ------------------------------

# Target distribution
plt.figure(figsize=(5,4))
sns.countplot(x='y', data=train)
plt.title("Target Variable Distribution")
plt.show()

# Check for missing values
print("\nMissing values in train:\n", train.isnull().sum())
print("\nMissing values in test:\n", test.isnull().sum())

# Numerical feature overview
num_features = train.select_dtypes(include=[np.number]).columns.tolist()
num_features.remove("y") if "y" in num_features else None
print("\nNumerical Features:", num_features)

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(train[num_features + ['y']].corr(), annot=False, cmap="coolwarm")
plt.title("Feature Correlations")
plt.show()

# Example: distribution of a numeric feature
plt.figure(figsize=(6,4))
sns.histplot(train[num_features[0]], kde=True)
plt.title(f"Distribution of {num_features[0]}")
plt.show()


# ------------------------------
# 4. Preprocessing
# ------------------------------
# Find categorical columns
cat_features = train.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical Features:", cat_features)

# Encode categoricals
lbl = LabelEncoder()
for col in cat_features:
    train[col] = lbl.fit_transform(train[col])
    test[col] = lbl.transform(test[col])

# Separate features & target
X = train.drop(['id', 'y'], axis=1)
y = train['y']
X_test = test.drop(['id'], axis=1)


# ------------------------------
# 5. Train/Validation Split
# ------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ------------------------------
# 6. Simple Model (Logistic Regression)
# ------------------------------
model = LogisticRegression(max_iter=500, solver='lbfgs')
model.fit(X_train, y_train)

# Validation ROC-AUC
val_preds = model.predict_proba(X_val)[:,1]
val_score = roc_auc_score(y_val, val_preds)
print("Validation ROC-AUC:", val_score)


# ------------------------------
# 7. Predict on Test
# ------------------------------
test_preds = model.predict_proba(X_test)[:,1]


# ------------------------------
# 8. Submission
# ------------------------------
submission = pd.DataFrame({
    'id': test['id'],
    'y': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")




