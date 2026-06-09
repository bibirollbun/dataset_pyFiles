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
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
sns.set_style("whitegrid")

# -------------------------------
# Step 2: Load Data
# -------------------------------
train = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e10/sample_submission.csv')

print("âœ… Data Loaded Successfully")
print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
print("\nColumns:", train.columns.tolist())

# -------------------------------
# Step 3: Quick Data Info
# -------------------------------
print("\nMissing Values:\n", train.isnull().sum())
print("\nTarget Distribution:")
print(train['loan_status'].value_counts())

# -------------------------------
# Step 4: Basic Visualizations
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='loan_status', data=train, palette='viridis')
plt.title('Loan Status Distribution')
plt.show()

# Numerical features distribution
num_cols = train.select_dtypes(include=['int64', 'float64']).columns.drop(['id'])
train[num_cols].hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle('Numeric Feature Distributions')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 7))
corr = train[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap (Numeric Features)')
plt.show()

# -------------------------------
# Step 5: Feature Selection
# -------------------------------
id_col = 'id'
target_col = 'loan_status'

X = train.drop(columns=[id_col, target_col])
y = train[target_col]

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

print("\nNumerical Columns:", num_cols.tolist())
print("Categorical Columns:", cat_cols.tolist())

# -------------------------------
# Step 6: Preprocessing
# -------------------------------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# -------------------------------
# Step 7: Model Definition
# -------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# -------------------------------
# Step 8: Cross-validation
# -------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
print("\nCross-validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy: {:.4f}".format(cv_scores.mean()))

# -------------------------------
# Step 9: Train-Test Split (for analysis)
# -------------------------------
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_valid)

print("\nValidation Accuracy: {:.4f}".format(accuracy_score(y_valid, y_pred)))
print("\nClassification Report:\n", classification_report(y_valid, y_pred))

# Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_valid, y_pred), annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix (Validation Set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# -------------------------------
# Step 10: Train on Full Data
# -------------------------------
pipeline.fit(X, y)
print("\nâœ… Model retrained on full training set.")

# -------------------------------
# Step 11: Predict on Test Data
# -------------------------------
X_test = test.drop(columns=[id_col])
test_preds = pipeline.predict(X_test)

submission = pd.DataFrame({
    id_col: test[id_col],
    'loan_status': test_preds
})

submission.to_csv('submission.csv', index=False)
print("\nğŸ“� submission.csv created successfully!")
print(submission.head())

# -------------------------------
# Step 12: Feature Importance
# -------------------------------
model_final = pipeline.named_steps['model']
encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
encoded_features = encoder.get_feature_names_out(cat_cols)
all_features = np.concatenate([num_cols, encoded_features])

importances = model_final.feature_importances_
fi = pd.DataFrame({'Feature': all_features, 'Importance': importances})
fi = fi.sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=fi, palette='coolwarm')
plt.title('Top 15 Important Features')
plt.show()

print("\nğŸ”¥ Top Important Features:")
print(fi)


