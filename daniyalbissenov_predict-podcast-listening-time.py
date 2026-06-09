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


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


train_data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train_data.head()


train_data.shape, test_data.shape


train_data.isnull().sum()


test_data.isnull().sum()


test_data.dtypes


numeric_only = train_data.select_dtypes(include=['int64','float64']).columns.drop(['id', 'Listening_Time_minutes'])
train_data[numeric_only].corr()


train_data[numeric_only] = train_data[numeric_only].fillna(train_data[numeric_only].median())
test_data[numeric_only] = test_data[numeric_only].fillna(test_data[numeric_only].median())

categorical_only = train_data.select_dtypes(include=['object']).columns

for col in categorical_only:
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
    test_data[col] = test_data[col].fillna(train_data[col].mode()[0])


plt.figure(figsize=(10,10))
t_corr = train_data[numeric_only].corr()
sns.heatmap(t_corr)
plt.title('Correlation')
plt.show()


for col in categorical_only:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])


X = train_data.drop(['id', 'Listening_Time_minutes'],axis=1)
y = train_data['Listening_Time_minutes']


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)

model.fit(X_train, y_train)
preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
rmse


plt.scatter(y_test, preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual time')
plt.ylabel('Predicted time')
plt.title('Actual vs Predicted')
plt.show()


final_predict = model.predict(test_data.drop('id', axis=1))

id_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')['id']

submission = pd.DataFrame(
    {
        'id': id_test,
        'Listening_Time_minutes': final_predict
    }
)
submission.to_csv('submission.csv', index=False)
print("Succesful")


submission




