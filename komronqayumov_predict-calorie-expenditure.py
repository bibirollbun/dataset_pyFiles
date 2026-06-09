





# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col=0)
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col=0)


train.head()


train['Sex'] = train['Sex'].map({'male':0, 'female':1})
test['Sex'] = test['Sex'].map({'male':0, 'female':1})


data_corr = train.corr()
sns.heatmap(data_corr, annot=True, cmap='winter_r')
plt.show()


train.corrwith(train['Calories']).abs().sort_values(ascending=False)


train['duration_x_hr'] = train['Duration'] * train['Heart_Rate']
train['duration_x_temp'] = train['Duration'] * train['Body_Temp']
train['hr_x_temp'] = train['Heart_Rate'] * train['Body_Temp']

test['duration_x_hr'] = test['Duration'] * test['Heart_Rate']
test['duration_x_temp'] = test['Duration'] * test['Body_Temp']
test['hr_x_temp'] = test['Heart_Rate'] * test['Body_Temp']


train['hr_per_min'] = train['Heart_Rate'] / train['Duration']
train['temp_per_min'] = train['Body_Temp'] / train['Duration']

test['hr_per_min'] = test['Heart_Rate'] / test['Duration']
test['temp_per_min'] = test['Body_Temp'] / test['Duration']



train.head()


plt.figure(figsize=(10,5))
sns.heatmap(train.corr().abs(), annot=True, cmap='winter_r')
plt.show()


train.corrwith(train['Calories']).abs().sort_values(ascending=False)


train.head()


train.hist(bins=50, figsize=(20,13))
plt.show()


fig, axis = plt.subplots(1,2, figsize=(14, 5))

sns.lineplot(data=train, x='Heart_Rate', y='Weight', ax=axis[0])
axis[0].set_title('Heart rate by weight', size=20)

sns.lineplot(data=train, x='Heart_Rate', y='Age', ax=axis[1])
axis[1].set_title('Heart rate by age', size=20)

plt.grid()
plt.show()


train.head()


X = train.drop(['Calories'], axis=1)
y = train['Calories']


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_prepared = scaler.fit_transform(X)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression
from sklearn import metrics
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


y_pred = lr_model.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Yangi xususiyatlar yaratish
X['BMI'] = X['Weight'] / (X['Height']/100)**2  # Tana massa indeksi
X['HR_per_Weight'] = X['Heart_Rate'] / X['Weight']  # Yurak urishi / vazn


from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)  # Chiziqli modeldan yaxshiroq bo'lishi mumkin


# Z-skor asosida outlierlarni olib tashlash
from scipy import stats
z_scores = stats.zscore(X)
X_clean = X[(z_scores < 3).all(axis=1)]  # 3 standart chetlanishdan tashqaridagilarni olib tashlash


print("Kaloriya sarfi min:", y.min(), "max:", y.max(), "o'rtacha:", y.mean())



from xgboost import XGBRegressor
from sklearn import metrics
model = XGBRegressor(
    objective='reg:squarederror',
    colsample_bytree=0.3,
    learning_rate=0.1,
    max_depth=5,
    alpha=10,
    n_estimators=1000,
    random_state=42,
    verbose=200
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('MAE', metrics.mean_absolute_error(y_test, y_pred))
print('MSE', metrics.mean_squared_error(y_test, y_pred))
print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_model_pred = rf_model.predict(X_test)  # ğŸ”� Bu toâ€˜gâ€˜rilangan qator
print('MAE', metrics.mean_absolute_error(y_test, rf_model_pred))
print('MSE', metrics.mean_squared_error(y_test, rf_model_pred))
print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, rf_model_pred)))


sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv', index_col='id')
test_set_prepared = scaler.fit_transform(test)
y_pred = rf_model.predict(test_set_prepared)
sub['Calories'] = y_pred
sub.to_csv('submission.csv')

