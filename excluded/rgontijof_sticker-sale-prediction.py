import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

#import matplotlib.pyplot as plt
#import seaborn as sns

test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")

data_train = train 
data_test = test


#a = data_train['num_sold']
#b = data_train.drop(columns=['num_sold'])


#bar = sns.distplot(data_train['num_sold'])
#plt.show()

# handling missing values on num_sold

imputer = SimpleImputer(strategy='median')
data_train['num_sold'] = imputer.fit_transform(data_train[['num_sold']]) 

# maybe tweek this for better precision later

# handling categorical values
# labeling

'''
['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore'] -> 6

['Discount Stickers', 'Stickers for Less', 'Premium Sticker Mart'] -> 3

['Holographic Goose', 'Kaggle', 'Kaggle Tiers', 'Kerneler', 'Kerneler Dark Mode'] -> 5

2557 different days

'''



data_train = pd.get_dummies(data_train, columns=['country', 'product', 'store'])

data_test = pd.get_dummies(data_test, columns=['country', 'product', 'store'])

data_train['date'] = pd.to_datetime(data_train['date'])
data_test['date'] = pd.to_datetime(data_test['date'])

data_train['year'] = data_train['date'].dt.year
data_train['month'] = data_train['date'].dt.month
data_train['day'] = data_train['date'].dt.day
data_train['dayofweek'] = data_train['date'].dt.dayofweek

data_test['year'] = data_test['date'].dt.year
data_test['month'] = data_test['date'].dt.month
data_test['day'] = data_test['date'].dt.day
data_test['dayofweek'] = data_test['date'].dt.dayofweek



data_train = data_train.drop(columns=['date'])
data_test = data_test.drop(columns=['date'])
# splitting and scalling

scaler = StandardScaler()

X_testing = data_test.drop(columns=['id'])
y = data_train['num_sold']
X = data_train.drop(columns=['num_sold', 'id'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

X_train = scaler.fit_transform(X_train)
X_testing = scaler.transform(X_testing)
X_test = scaler.transform(X_test)

regressor_xgb = XGBRegressor()

regressor_xgb.fit(X_train, y_train)

y_pred = regressor_xgb.predict(X_test)
y_pred_testing = regressor_xgb.predict(X_testing)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"R²: {r2:.2f}")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

#R²: 0.99
#Mean Squared Error: 4566.79657804362
#Mean Absolute Error: 41.43326884284256


y_pred_testing = np.round(y_pred_testing).astype(int)
submission = pd.DataFrame({ 
                           
    "id": data_test['id'],
    "num_sold": y_pred_testing

})
submission.to_csv("submission.csv", index=False)              


#Not by anymeans final model, studying how to implement time series on this thing. Happy competition!

