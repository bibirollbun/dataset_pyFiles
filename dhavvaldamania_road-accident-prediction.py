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


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



train.isnull().sum()



train.describe
train.head(5)


num_cols = train.select_dtypes(include=np.number).columns.tolist()
cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
bool_cols = train.select_dtypes(include='bool').columns.tolist()

print("\nNumeric Columns:", num_cols)
print("\nCategorical Columns:", cat_cols)
print("\nBoolean Columns:", bool_cols)


for col in num_cols:
    if col not in ['id']:  # skip id column
        plt.figure(figsize=(6,4))
        sns.histplot(train[col], kde=True, bins=40)
        plt.title(f"Distribution of {col}")
        plt.show()


for col in cat_cols:
    plt.figure(figsize=(8,4))
    order = train[col].value_counts().index
    sns.countplot(y=col, data=train, order=order)
    plt.title(f"Count of {col}")
    plt.show()

    # mean target per category (if target column exists)
    if 'accident_risk' in train.columns:
        plt.figure(figsize=(8,4))
        risk_means = train.groupby(col)['accident_risk'].mean().sort_values(ascending=False)
        sns.barplot(x=risk_means.values, y=risk_means.index)
        plt.title(f"Mean Accident Risk by {col}")
        plt.show()


plt.figure(figsize=(10,8))
sns.heatmap(train[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Copy to avoid modifying original data
train_encoded = train.copy()

# Identify categorical columns
cat_cols = train_encoded.select_dtypes(include=['object', 'category']).columns

# Apply Label Encoding to all categorical columns
le = LabelEncoder()
for col in cat_cols:
    train_encoded[col] = le.fit_transform(train_encoded[col].astype(str))

# Now separate features and target
X = train_encoded.drop('accident_risk', axis=1)
y = train_encoded['accident_risk']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=None)
model.fit(X_train, y_train)

# Feature importances
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(10))



from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Predictions on validation set
y_pred = model.predict(X_valid)

# Metrics
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
r2 = r2_score(y_valid, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")



# Encode categorical columns in test using the same LabelEncoder logic
test_encoded = test.copy()
for col in cat_cols:
    if col in test_encoded.columns:
        test_encoded[col] = le.fit_transform(test_encoded[col].astype(str))

# Predict
test_predictions = model.predict(test_encoded)

# Create submission
submission['accident_risk'] = test_predictions
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv file created successfully!")


