import pandas as pd


df = pd.read_csv("/kaggle/input/bitcoin-price-forecast/Bitcoin_kaggle.csv", sep=';')
df


df.describe()


df['Date'] = pd.to_datetime(df['Date'])


df.info()


df.duplicated().sum()


df


# df = df.drop(['Date'], axis=True)


for col in ['Price', 'Open', 'High', 'Low']:
    df[col] = df[col].astype(str).str.replace(',', '').astype(float)


df


train = df.iloc[:-7].copy()  # Ğ’Ñ�Ñ‘, ĞºÑ€Ğ¾Ğ¼Ğµ Ğ¿Ğ¾Ñ�Ğ»ĞµĞ´Ğ½Ğ¸Ñ… 7 Ñ�Ñ‚Ñ€Ğ¾Ğº
test = df.iloc[-7:].copy()   # ĞŸĞ¾Ñ�Ğ»ĞµĞ´Ğ½Ğ¸Ğµ 7 Ñ�Ñ‚Ñ€Ğ¾Ğº (Ğ´ÑƒĞ±Ğ»Ğ¸)


train.shape


test.shape


# Ğ£Ğ´Ğ°Ğ»Ğ¸Ğ¼ Ğ·Ğ½Ğ°Ğº Ğ¿Ñ€Ğ¾Ñ†ĞµĞ½Ñ‚Ğ°
for j in ['Change %']:
    train[j] = train[j].str.replace('%', '').astype(float)
    test[j] = test[j].str.replace('%', '').astype(float)


# Ğ£Ñ�Ñ‚Ğ°Ğ½Ğ¾Ğ²Ğ¸Ğ¼ Ğ´Ğ°Ñ‚Ñƒ ĞºĞ°Ğº Ğ¸Ğ½Ğ´ĞµĞºÑ�
train = train.set_index('Date')
test = test.set_index('Date')


test


# Ğ¡ĞºĞ¾Ğ»ÑŒĞ·Ñ�Ñ‰ĞµĞµ Ñ�Ñ€ĞµĞ´Ğ½ĞµĞµ Ğ½Ğ° 7 Ğ¸ 30 Ğ´Ğ½ĞµĞ¹
train['rol_7d'] = train['Price'].rolling(window=7, min_periods=1).mean()
train['rol_30d'] = train['Price'].rolling(window=30, min_periods=1).mean()


# Ğ�Ğ°Ğ¹Ğ´ĞµĞ¼ Ñ€Ğ°Ğ·Ğ½Ğ¸Ñ†Ñƒ
train['high_diff_low'] = (train['High'] - train['Low']).shift(1)


# Ğ”ĞµĞ½ÑŒ Ğ½ĞµĞ´ĞµĞ»Ğ¸ Ğ¸ ĞºÑ€Ğ°Ğ¹Ğ½Ğ¹ Ğ»Ğ¸ Ğ´ĞµĞ½ÑŒ Ğ² Ğ¼ĞµÑ�Ñ�Ñ†Ğµ
train['dayofweek'] = train.index.dayofweek
train['last_day'] = train.index.is_month_end.astype(int)


# Ğ£Ğ´Ğ°Ğ»Ğ¸Ğ¼ Ğ½ĞµĞ½ÑƒĞ¶Ğ½Ñ‹Ğµ Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ñ‹ Ğ¸ ÑƒĞ´Ğ°Ğ»Ğ¸Ğ¼ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ‰ĞµĞ½Ğ½Ñ‹Ğµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ�
train = train.drop(['High', 'Low', 'Vol.', 'Change %'], axis=1)
train = train.dropna()


# Ğ¡ĞºĞ¾Ğ»ÑŒĞ·Ñ�Ñ‰ĞµĞµ Ñ�Ñ€ĞµĞ´Ğ½ĞµĞµ Ğ½Ğ° 7 Ğ¸ 30 Ğ´Ğ½ĞµĞ¹
test['rol_7d'] = test['Price'].rolling(window=7, min_periods=1).mean()
test['rol_30d'] = test['Price'].rolling(window=30, min_periods=1).mean()


# Ğ Ğ°Ğ·Ğ½Ğ¸Ñ†Ğ° High - Low Ğ½Ğ° Ğ¿Ğ¾Ñ�Ğ»ĞµĞ´Ğ½Ğ¸Ğ¹ Ğ´ĞµĞ½ÑŒ train
last_high = df['High'].iloc[-8] 
last_low = df['Low'].iloc[-8]
test['high_diff_low'] = last_high - last_low


# Ğ”Ğ°Ñ‚Ğ°-Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
test['dayofweek'] = test.index.dayofweek
test['last_day'] = test.index.is_month_end.astype(int)


test = test.drop(['High', 'Low', 'Vol.', 'Change %'], axis=1)


# Ğ›Ğ°Ğ³Ğ¾Ğ²Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ (Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ� Ğ·Ğ° Ğ¿Ñ€ĞµĞ´Ñ‹Ğ´ÑƒÑ‰Ğ¸Ğµ Ğ´Ğ½Ğ¸)
for lag in [1, 2, 3, 7]:
    train[f'lag_{lag}'] = train['Price'].shift(lag)
    test[f'lag_{lag}'] = train['Price'].shift(lag).iloc[-7:].values


train['Price'].plot(figsize=(25,5))


train.isnull().sum()


train.shape


train = train.dropna()


train.shape


X_train = train.drop('Price', axis=1)
y_train = train['Price']

X_test = test.drop('Price', axis=1)
y_test = test['Price']


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

model = LinearRegression()

model.fit(X_train, y_train)


print('Score on train data = ', round(model.score(X_train, y_train), 4))
print('Score on test data = ', round(model.score(X_test, y_test), 4))


y_test_pred4 = model.predict(X_test)
mae_test = mean_absolute_error(y_test, y_test_pred4)
print('MAE on test data =', round(mae_test, 4))


import xgboost as xgb
from sklearn.model_selection import GridSearchCV

xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='mae')

# Ğ—Ğ°Ğ¿Ğ¸ÑˆĞµĞ¼ Ğ½ĞµĞ¾Ğ±Ñ…Ğ¾Ğ´Ğ¸Ğ¼Ñ‹Ğµ Ğ½Ğ°Ğ¼ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹ Ğ´Ğ»Ñ� Ğ´Ğ°Ğ»ÑŒĞ½ĞµĞ¹ÑˆĞµĞ³Ğ¾ Ğ¿ĞµÑ€ĞµĞ±Ğ¾Ñ€Ğ°.
param_xgb = {
    'n_estimators': [100, 200],  
    'learning_rate': [0.05, 0.01],  
    'max_depth': [2, 3],
    'min_child_weight': [1, 3, 5]    
}

# Ğ¡Ğ´ĞµĞ»Ğ°ĞµĞ¼ Ğ¿ĞµÑ€ĞµĞ±Ğ¾Ñ€ Ğ·Ğ°Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… Ğ²Ñ‹ÑˆĞµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ², Ğ¿Ñ€Ğ¸ Ñ�Ñ‚Ğ¾Ğ¼ Ñ€Ğ°Ğ·Ğ´ĞµĞ»Ğ¸Ğ² Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºÑƒ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… Ğ½Ğ° 5 Ñ‡Ğ°Ñ�Ñ‚ĞµĞ¹.
grid_search__xgb = GridSearchCV(xgb_regressor, param_xgb, cv=5)

# Ğ�Ğ±ÑƒÑ‡Ğ¸Ğ¼ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ½Ğ° Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ñ‹Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
grid_search__xgb.fit(X_train, y_train)


grid_search__xgb.best_params_


best_gs_xgb_two = grid_search__xgb.best_estimator_


print('Score on train data = ', round(best_gs_xgb_two.score(X_train, y_train), 4))
print('Score on test data = ', round(best_gs_xgb_two.score(X_test, y_test), 4))

