# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # df processing, CSV file I/O (e.g. pd.read_csv)

# Input df files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd 


df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')


df.info()


df.duplicated().sum()


df.isnull().sum()


df = df.dropna(subset=['num_sold'])


df.isnull().sum()


df.info()


df['date'].head()


df['date'] = pd.to_datetime(df['date'])


df['date'].head()


df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek


df = pd.get_dummies(df , columns  = ['country' , 'store' , 'product'] , drop_first = True)


df.head()


x = df.drop(columns = ['id' , 'date' , 'num_sold'])
y = df['num_sold']


from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42)


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state = 42)
model.fit(x_train , y_train)


y_pred = model.predict(x_test)


import numpy as np

def mean_absolute_percentage_error(y_true , y_pred):

    return np.mean(np.abs((y_true - y_pred) / y_true ))  * 100

mape = mean_absolute_percentage_error(y_test , y_pred)

print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal Fit")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()


importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

features = x.columns 


plt.figure(figsize=(12, 8))
plt.bar(range(x.shape[1]), importances[indices], align="center")
plt.xticks(range(x.shape[1]), [features[i] for i in indices], rotation=90)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()


test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


test_df.info()


test_df.isnull().sum()


test_df['date'] = pd.to_datetime(test_df['date'])


test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['day_of_week'] = test_df['date'].dt.dayofweek
test_df = pd.get_dummies(test_df, columns=['country', 'store', 'product'], drop_first=True)


X_test_final = test_df.drop(columns=['id', 'date'])
X_test_final = X_test_final.reindex(columns=x.columns, fill_value=0)


test_df['num_sold'] = model.predict(X_test_final)


test_df[['id', 'num_sold']].to_csv("submission.csv", index=False)

