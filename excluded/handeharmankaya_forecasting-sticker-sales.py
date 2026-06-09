import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')


train=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train.head()


train['date'] = pd.to_datetime(train['date'])


train.shape


train.info()


train.isnull().sum()


test.head()


test['date'] = pd.to_datetime(test['date'])


test.shape


test.info()


test.isnull().sum()


daily_sales = train.groupby('date')['num_sold'].sum()

plt.plot(daily_sales, label='Total Daily Sales', color='skyblue')
plt.title('Total Sticker Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sold')
plt.legend()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sns.barplot(x=train['country'], y=train['num_sold'], estimator=np.sum, ax=axes[0], palette='pastel')
axes[0].set_title('Total Sales by Country')
axes[0].tick_params(axis='x', rotation=45)

sns.barplot(x=train['store'], y=train['num_sold'], estimator=np.sum, ax=axes[1], palette='pastel')
axes[1].set_title('Total Sales by Stores')
axes[1].tick_params(axis='x', rotation=45)

sns.barplot(x=train['product'], y=train['num_sold'], estimator=np.sum, ax=axes[2], palette='pastel')
axes[2].set_title('Total Sales by Products')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day_of_week'] = train['date'].dt.dayofweek 
train['day_name'] = train['date'].dt.day_name()

plt.figure(figsize=(10, 5))
sns.boxplot(x=train['day_name'], y=train['num_sold'], palette='pastel',
            order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.title('Sales Distribution by Day of the Week (Weekend Effect?)')
plt.show()

plt.figure(figsize=(12, 5))
sns.lineplot(x=train['month'], y=train['num_sold'], hue=train['year'], palette='pastel', marker='o')
plt.title("Sales Trend by Year and Month (New Year's effect?)")
plt.show()


train['num_sold'] = train.groupby(['country', 'store', 'product'])['num_sold'].transform(
    lambda x: x.interpolate(method='linear'))


train.isnull().sum()


if train['num_sold'].isnull().sum() > 0:
    train['num_sold'] = train['num_sold'].fillna(method='bfill').fillna(method='ffill')


train.isnull().sum()


def create_features(df):
    df = df.copy()

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    #Cyclical Encoding
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    return df

train = create_features(train)
test = create_features(test)


train.head()


test.head()


categorical_cols = ['country', 'store', 'product']
encoders = {} 
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]])) 
    
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    encoders[col] = le 


train.head()


test.head()


features = ['country', 'store', 'product', 'year', 'month', 'day', 'day_of_week', 'day_of_year', 
            'is_weekend', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
target = 'num_sold'


split_date = '2016-10-01'

train_split = train[train['date'] < split_date]
test_split = train[train['date'] >= split_date]

x_train = train_split[features]
y_train = train_split[target]

x_test = test_split[features]
y_test = test_split[target]


model = xgb.XGBRegressor(n_estimators=1000,learning_rate=0.05,max_depth=6,early_stopping_rounds=50,
                         objective='reg:absoluteerror',n_jobs=-1,random_state=42)

model.fit(x_train, y_train,eval_set=[(x_train, y_train), (x_test, y_test)],verbose=100)


predictions = model.predict(x_test)
predictions = [max(0, x) for x in predictions]

score = mean_absolute_percentage_error(y_test, predictions)

xgb.plot_importance(model, max_num_features=10, height=0.5)
plt.title("Feature Importance")
plt.show()


x_all = train[features]
y_all = train[target]

full_model = xgb.XGBRegressor(n_estimators=1000,learning_rate=0.05,max_depth=6,objective='reg:absoluteerror',
                              n_jobs=-1,random_state=42)

full_model.fit(x_all, y_all, verbose=False)

x_test = test[features]
final_predictions = full_model.predict(x_test)
final_predictions = [max(0, x) for x in final_predictions] 


submission = pd.DataFrame({'id': test['id'],'num_sold': final_predictions})
submission.to_csv('submission.csv', index=False)


import holidays

def add_holiday_features(df, encoders):
    df = df.copy()
    df['country_name'] = encoders['country'].inverse_transform(df['country'])
    years = df['date'].dt.year.unique()
    country_codes = {'Canada': 'CA','Finland': 'FI','Italy': 'IT','Kenya': 'KE','Norway': 'NO','Singapore': 'SG'}
    
    df['is_holiday'] = 0
    
    for country_name, code in country_codes.items():
        try:
            country_holidays = holidays.Country(code, years=years)
            mask = (df['country_name'] == country_name) & (df['date'].isin(country_holidays))
            df.loc[mask, 'is_holiday'] = 1
        except:
            continue
    df = df.drop(columns=['country_name'])
    
    return df

train = add_holiday_features(train, encoders)
test = add_holiday_features(test, encoders)


train.head()


test.head()


features_holiday = features + ['is_holiday'] 

split_date = '2016-10-01'

X_train_h = train[train['date'] < split_date][features_holiday]
y_train_h = train[train['date'] < split_date][target]

X_test_h = train[train['date'] >= split_date][features_holiday]
y_test_h = train[train['date'] >= split_date][target]

model_holiday = xgb.XGBRegressor(n_estimators=1000,learning_rate=0.05,max_depth=6,early_stopping_rounds=50,
                                 objective='reg:absoluteerror',n_jobs=-1,random_state=42)

model_holiday.fit(X_train_h, y_train_h,eval_set=[(X_train_h, y_train_h), (X_test_h, y_test_h)],verbose=100)


X_all_h = train[features_holiday]
y_all_h = train[target]

full_model_holiday = xgb.XGBRegressor(n_estimators=1000,learning_rate=0.05,max_depth=6,
                                      objective='reg:absoluteerror',n_jobs=-1,random_state=42)

full_model_holiday.fit(X_all_h, y_all_h, verbose=False)


X_final_test = test[features_holiday]
final_preds_holiday = full_model_holiday.predict(X_final_test)
final_preds_holiday = [max(0, x) for x in final_preds_holiday] 


submission_h = pd.DataFrame({'id': test['id'], 'num_sold': final_preds_holiday})
submission_h.to_csv('submission_holidays.csv', index=False)


train['num_sold'] = np.log1p(train['num_sold']) 

features = ['country', 'store', 'product', 'year', 'month', 'day', 'day_of_week', 'day_of_year', 'is_weekend', 
            'day_sin', 'day_cos', 'month_sin', 'month_cos', 'is_holiday'] 
target = 'num_sold' 

X_all = train[features]
y_all = train[target]

full_model = xgb.XGBRegressor(n_estimators=1500,learning_rate=0.03,max_depth=8,objective='reg:squarederror',
                              n_jobs=-1,random_state=42)

full_model.fit(X_all, y_all, verbose=False)


X_test = test[features]
preds_log = full_model.predict(X_test)

final_predictions = np.expm1(preds_log)
final_predictions = [max(0, x) for x in final_predictions] 


submission = pd.DataFrame({'id': test['id'], 'num_sold': final_predictions})
submission.to_csv('submission_v3.csv', index=False)


import joblib

joblib.dump(full_model, 'xgb_model.joblib')
joblib.dump(encoders, 'encoders.joblib')

