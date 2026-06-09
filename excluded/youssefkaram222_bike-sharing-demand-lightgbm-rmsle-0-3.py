import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



train=pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
test=pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
submission=pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')


train.info()


train.describe()


train.head()


train['datetime']=pd.to_datetime(train['datetime'])
train['hour']=train['datetime'].dt.hour
train['day']=train['datetime'].dt.day
train['month']=train['datetime'].dt.month
train['year']=train['datetime'].dt.year
train['weekday']=train['datetime'].dt.weekday


train['is_weekend'] = train['weekday'].isin([5, 6]).astype(int)
train['is_peak_hour'] = train['hour'].isin([7,8,9,16,17,18]).astype(int)

def get_part_of_day(hour):
    if 6 <= hour < 12: return 'morning'
    elif 12 <= hour < 17: return 'afternoon'
    elif 17 <= hour < 21: return 'evening'
    else: return 'night'

train['part_of_day'] = train['hour'].apply(get_part_of_day)



train['temp_diff'] = train['atemp'] - train['temp']
train['feels_hot'] = (train['atemp'] > 30).astype(int)
train['feels_cold'] = (train['atemp'] < 10).astype(int)



train['zero_wind'] = (train['windspeed'] == 0).astype(int)
train['wind_bin'] = pd.cut(train['windspeed'], bins=[-1,5,15,30,100], labels=['calm','breeze','windy','storm'])



train['bad_weather'] = train['weather'].isin([3,4]).astype(int)
weather_map = {1: 'clear', 2: 'misty', 3: 'rainy', 4: 'extreme'}
train['weather_str'] = train['weather'].map(weather_map)



sns.boxplot(data=train, x='is_weekend', y='count')
plt.title('Bike Demand on Weekends vs Weekdays')
plt.xticks([0,1], ['Weekday', 'Weekend'])
plt.show()


sns.boxplot(data=train, x='is_peak_hour', y='count')
plt.title('Bike Demand During Peak Hours')
plt.xticks([0,1], ['Non-Peak', 'Peak Hours'])
plt.show()



sns.barplot(data=train, x='part_of_day', y='count', estimator='mean', order=['morning','afternoon','evening','night'])
plt.title('Average Count per Part of Day')
plt.show()



sns.histplot(train['temp_diff'], kde=True)
plt.title('Distribution of Temperature Difference (atemp - temp)')
plt.show()

sns.scatterplot(data=train, x='temp_diff', y='count')
plt.title('Count vs Temperature Difference')
plt.show()



sns.barplot(data=train, x='wind_bin', y='count', estimator='mean')
plt.title('Average Count by Wind Level')
plt.show()



# Select numeric features only
numeric_cols = train.select_dtypes(include=['int64', 'float64']).copy()

# Drop target and ID/date
cols_to_drop = ['casual', 'registered', 'count']
numeric_cols = numeric_cols.drop(columns=cols_to_drop, errors='ignore')

# Add log_count if used
if 'log_count' in train.columns:
    numeric_cols['log_count'] = train['log_count']

plt.figure(figsize=(14,8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Engineered Features')
plt.show()



df=train.copy()


df['log_count'] = np.log1p(df['count'])

df = df.drop(columns=['datetime', 'casual', 'registered', 'count'])



categorical_features = ['season', 'weather', 'part_of_day', 'wind_bin', 'weather_str',
                        'year', 'month', 'weekday', 'hour']

df = pd.get_dummies(df, columns=categorical_features, drop_first=True)



from sklearn.preprocessing import StandardScaler

continuous_features = ['temp', 'atemp', 'humidity', 'windspeed', 'temp_diff']
scaler = StandardScaler()
df[continuous_features] = scaler.fit_transform(df[continuous_features])



X = df.drop(columns='log_count')
y = df['log_count']



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


def evaluate_model(model, X_val, y_val, name):
    y_pred = model.predict(X_val)
    rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(y_pred)))
    print(f"{name} RMSLE: {rmsle:.4f}")
    return rmsle



lr = LinearRegression()
lr.fit(X_train, y_train)
lr_rmsle = evaluate_model(lr, X_val, y_val, "Linear Regression")



ridge = Ridge(alpha=1.0)  # You can tune alpha later
ridge.fit(X_train, y_train)
ridge_rmsle = evaluate_model(ridge, X_val, y_val, "Ridge Regression")



lasso = Lasso(alpha=0.01)  # Alpha must be small to avoid all weights going to 0
lasso.fit(X_train, y_train)
lasso_rmsle = evaluate_model(lasso, X_val, y_val, "Lasso Regression")



rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_rmsle = evaluate_model(rf, X_val, y_val, "Random Forest")



xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)
xgb_rmsle = evaluate_model(xgb_model, X_val, y_val, "XGBoost Regressor")



lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgb_model.fit(X_train, y_train)
lgb_rmsle = evaluate_model(lgb_model, X_val, y_val, "LightGBM Regressor")



models = ['Linear', 'Ridge', 'Lasso', 'Random Forest', 'XGBoost', 'LightGBM']
scores = [lr_rmsle, ridge_rmsle, lasso_rmsle, rf_rmsle, xgb_rmsle, lgb_rmsle]

plt.figure(figsize=(10,5))
sns.barplot(x=models, y=scores)
plt.title('RMSLE Comparison Across All Models')
plt.ylabel('RMSLE')
plt.show()



# Make a copy of test set
test_df = test.copy()

# Step 1: Extract datetime features
test_df['datetime'] = pd.to_datetime(test_df['datetime'])
test_df['hour'] = test_df['datetime'].dt.hour
test_df['day'] = test_df['datetime'].dt.day
test_df['month'] = test_df['datetime'].dt.month
test_df['year'] = test_df['datetime'].dt.year
test_df['weekday'] = test_df['datetime'].dt.weekday

# Step 2: Feature Engineering (same as training set)
test_df['is_weekend'] = test_df['weekday'].isin([5, 6]).astype(int)
test_df['is_peak_hour'] = test_df['hour'].isin([7,8,9,16,17,18]).astype(int)
test_df['part_of_day'] = test_df['hour'].apply(get_part_of_day)
test_df['temp_diff'] = test_df['atemp'] - test_df['temp']
test_df['feels_hot'] = (test_df['atemp'] > 30).astype(int)
test_df['feels_cold'] = (test_df['atemp'] < 10).astype(int)
test_df['zero_wind'] = (test_df['windspeed'] == 0).astype(int)
test_df['wind_bin'] = pd.cut(test_df['windspeed'], bins=[-1,5,15,30,100], labels=['calm','breeze','windy','storm'])
test_df['bad_weather'] = test_df['weather'].isin([3,4]).astype(int)
test_df['weather_str'] = test_df['weather'].map({1: 'clear', 2: 'misty', 3: 'rainy', 4: 'extreme'})
test_df['season_weather'] = test_df['season'].astype(str) + '_' + test_df['weather'].astype(str)
test_df['hour_workingday'] = test_df['hour'] * test_df['workingday']

# Step 3: Drop datetime column (LightGBM doesn't accept datetime dtype)
test_df = test_df.drop(columns=['datetime'], errors='ignore')

# Step 4: One-hot encode all categorical features
# Must include season_weather now
all_categorical = categorical_features + ['season_weather']  # reuse list from training
test_df = pd.get_dummies(test_df, columns=all_categorical, drop_first=True)

# Step 5: Align test with training feature set (X_train_full)
test_df, X_train_full = test_df.align(X_train_full, join='left', axis=1, fill_value=0)




final_model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

final_model.fit(X_train_full, y)  # y = log1p(count)



test_preds_log = final_model.predict(test_df)
test_preds = np.expm1(test_preds_log)  # Undo log1p

submission = pd.DataFrame({
    'datetime': test['datetime'],
    'count': test_preds
})

submission.to_csv('submission.csv', index=False)





