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


# Step 1: Basic imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Step 2: Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

# Step 3: Basic EDA
print(train.shape)
print(train.info())
sns.countplot(data=train, x='Personality')
plt.title('Target Distribution')
plt.show()
print(train.isnull().sum())

# Step 4: Feature prep (drop id + target column from train, id only from test)
train_features = train.drop(columns=['id', 'Personality'])
test_features = test.drop(columns=['id'])

# Combine train + test for uniform label encoding
combined = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)

# Encode categorical columns
for col in combined.columns:
    if combined[col].dtype == 'object':
        combined[col] = combined[col].fillna("missing")
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

# Step 5: Impute missing numeric values
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(combined.iloc[:len(train)])
X_test = imputer.transform(combined.iloc[len(train):])

# Step 6: Encode target labels (0 for Introvert, 1 for Extrovert)
target_le = LabelEncoder()
y = target_le.fit_transform(train['Personality'])

# Step 7: Train RandomForest with CV
model = RandomForestClassifier(n_estimators=100, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

# Step 8: Train model fully
model.fit(X, y)

# Step 9: Predict test set
test_preds = model.predict(X_test)
predicted_labels = target_le.inverse_transform(test_preds)

# Step 10: Create submission file
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
submission['Personality'] = predicted_labels  # ✅ Correct column name
submission = submission[['id', 'Personality']]  # Drop any extra columns
submission.to_csv("submission.csv", index=False)


