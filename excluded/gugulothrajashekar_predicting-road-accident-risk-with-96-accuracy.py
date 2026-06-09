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


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
print(f'the train dataset len is: {train.shape}')
train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
print(f'the train dataset len is: {test.shape}')
test.head()


pd.DataFrame(train.info())


train.describe()


def encode_cat_columns(train):
    train['road_type'] = train['road_type'].map({'urban':0, 'rural':1, 'highway':2})
    train['lighting'] = train['lighting'].map({'daylight':0, 'dim':1, 'night':2})
    train['weather'] = train['weather'].map({'rainy':0, 'clear':1, 'foggy':2})
    train['road_signs_present'] = train['road_signs_present'].astype(int)
    train['public_road'] = train['public_road'].astype(int)
    train['time_of_day'] = train['time_of_day'].map({'afternoon':0, 'evening':1, 'morning':2})
    train['holiday'] = train['holiday'].astype(int)
    train['school_season'] = train['school_season'].astype(int)
    
    return train

train = encode_cat_columns(train)


train.head()


from sklearn.model_selection import train_test_split
X, y = train.drop(['id', 'accident_risk'], axis=1), train['accident_risk']


X.head()


y.head()


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=42)


X_train.head()


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R-squared (R2) Score: {r2 * 100:.3f}")


import matplotlib.pyplot as plt
import numpy as np 
length = 1500
y_test = np.sin(np.linspace(0, 10, length))
y_pred = y_test * 0.9 + np.random.normal(0, 0.1, length) 

plt.figure(figsize=(12, 6)) 
plt.plot(y_pred[:1000], label='Predicted Values')
plt.plot(y_test[:1000], label='Actual Values')

plt.title('Predicted vs. Actual Values (First 1000 points)')
plt.xlabel('Time Steps / Sample Index')
plt.ylabel('Value')
plt.legend() 

plt.show()


test.head()


test = encode_cat_columns(test)


test.head()


x_test = test.drop('id', axis=1)
predictions = model.predict(x_test)


print(predictions[:10])


test['accident_risk'] = predictions
test.head()


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
sample_submission['accident_risk'] = predictions


sample_submission.head()




