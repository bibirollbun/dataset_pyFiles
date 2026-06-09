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


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train.head()


train.info()


train.describe()


train.isnull().sum()


from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


y = train['Listening_Time_minutes']
X = train.drop(columns=['Listening_Time_minutes'])


test_ids = test['id']
combined = pd.concat([X, test.drop(columns=['id'])], axis=0)


numeric_cols = combined.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = combined.select_dtypes(include=['object', 'category']).columns


num_imputer = SimpleImputer(strategy='median')
combined[numeric_cols] = num_imputer.fit_transform(combined[numeric_cols])


cat_imputer = SimpleImputer(strategy='most_frequent')
combined[categorical_cols] = cat_imputer.fit_transform(combined[categorical_cols])


label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
    label_encoders[col] = le


X_processed = combined.iloc[:len(X), :]
test_processed = combined.iloc[len(X):, :]


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


X = train.drop(columns=['Listening_Time_minutes'])
y = train['Listening_Time_minutes']


X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_val)



rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.4f}")


test_predictions = model.predict(test_processed)


test_predictions = np.round(test_predictions, 3)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'Listening_Time_minutes': test_predictions
})


submission.to_csv('submission.csv', index=False)

print("✅ Submission file created: submission.csv")

