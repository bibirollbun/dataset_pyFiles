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


train = pd.read_csv("/kaggle/input/nexar-collision-prediction/train.csv")
test = pd.read_csv("/kaggle/input/nexar-collision-prediction/test.csv")
sample= pd.read_csv("/kaggle/input/nexar-collision-prediction/sample_submission.csv")


train.head()


train.info()


train.describe()


train.isnull().sum()


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


y = train['target']
X = train.drop(columns=['target'])
test_ids = test['id']

# Combine train and test for consistent preprocessing
combined = pd.concat([X, test.drop(columns=['id'])], axis=0)



# Fill numeric NaNs with median
numeric_cols = combined.select_dtypes(include=['int64', 'float64']).columns
if len(numeric_cols) > 0:
    num_imputer = SimpleImputer(strategy='median')
    combined[numeric_cols] = num_imputer.fit_transform(combined[numeric_cols])

# Fill categorical NaNs with 'MISSING' and then encode
categorical_cols = combined.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    combined[categorical_cols] = combined[categorical_cols].fillna('MISSING')
    for col in categorical_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))


print("Missing values after preprocessing:")
print(combined.isna().sum())


X_processed = combined.iloc[:len(X), :]
test_processed = combined.iloc[len(X):, :]


X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)


print("\nFinal check before training:")
print("NaN in X_train:", X_train.isna().sum().sum())
print("NaN in y_train:", y_train.isna().sum())

# If there are still NaN values, we'll drop those rows
if X_train.isna().sum().sum() > 0 or y_train.isna().sum() > 0:
    print("\nWarning: Dropping rows with remaining NaN values")
    non_nan_mask = ~X_train.isna().any(axis=1) & ~y_train.isna()
    X_train = X_train[non_nan_mask]
    y_train = y_train[non_nan_mask]


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
print(f"\nValidation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(classification_report(y_val, y_pred))


test_predictions = model.predict(test_processed)


submission = pd.DataFrame({
    'id': test_ids,
    'target': test_predictions
})
submission.to_csv('submission.csv', index=False)

print("\n✅ Submission file created: submission.csv")




