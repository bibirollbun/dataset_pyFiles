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


import pandas as pd
train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_data.head()


import pandas as pd
train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
train_data.head()


import pandas as pd
train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
train_data.head()


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
extra_data = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
print("Train data:")
display(train_data.head())

print("Test data:")
display(test_data.head())

print("Extra data:")
display(extra_data.head())


numerical_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
if 'Price' in numerical_cols:
    numerical_cols.remove('Price')

categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()

for col in numerical_cols:
    median_val = train_data[col].median()
    train_data[col] = train_data[col].fillna(median_val)
    test_data[col] = test_data[col].fillna(median_val)

for col in categorical_cols:
    train_data[col] = train_data[col].fillna('Unknown')
    test_data[col] = test_data[col].fillna('Unknown')



from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col]) 
    label_encoders[col] = le


X = train_data.drop(columns=["id", "Price"])
y = train_data["Price"]

X_test = test_data.drop(columns=["id"])


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error

val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print("Validation RMSE:", rmse)



test_preds = model.predict(X_test)
submission = pd.DataFrame({
    "id": test_data["id"],
    "Price": test_preds
})
submission.to_csv("baseline_submission.csv", index=False)
print("Submission file 'baseline_submission.csv' has been created.")

