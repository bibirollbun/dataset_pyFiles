import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 读取数据
dfs = {}
for name in ['train', 'test']:
    df = pd.read_csv(f'/kaggle/input/bike-sharing-demand/{name}.csv')
    df['_data'] = name
    dfs[name] = df

# 合并数据
df = pd.concat([dfs['train'], dfs['test']], axis=0)

# 列名小写
df.columns = map(str.lower, df.columns)


# 解析时间
dt = pd.DatetimeIndex(df['datetime'])
df.set_index(dt, inplace=True)

# 对目标变量做对数变换
for col in ['casual', 'registered', 'count']:
    if col in df.columns:
        df[f'{col}_log'] = np.log(df[col] + 1)

df['date'] = dt.date
df['day'] = dt.day
df['month'] = dt.month
df['year'] = dt.year
df['hour'] = dt.hour
df['dow'] = dt.dayofweek
df['woy'] = dt.isocalendar().week


# 插值补全
df["weather"] = df["weather"].interpolate(method='time').round()
df["temp"] = df["temp"].interpolate(method='time')
df["atemp"] = df["atemp"].interpolate(method='time')
df["humidity"] = df["humidity"].interpolate(method='time').round()
df["windspeed"] = df["windspeed"].interpolate(method='time')

# 按季节统计
by_season = df[df['_data'] == 'train'].groupby('season')[['count']].agg("sum")
by_season.columns = ['count_season']
df = df.join(by_season, on='season')

print(by_season)

def get_day(day_start):
    day_end = day_start + pd.offsets.DateOffset(hours=23)
    return pd.date_range(day_start, day_end, freq="h")  # 'h' 替换 'H'

# 特殊日期修正
df.loc[get_day(pd.to_datetime('2011-04-15')), "workingday"] = 1
df.loc[get_day(pd.to_datetime('2012-04-16')), "workingday"] = 1
df.loc[get_day(pd.to_datetime('2011-11-25')), "workingday"] = 0
df.loc[get_day(pd.to_datetime('2012-11-23')), "workingday"] = 0
df.loc[get_day(pd.to_datetime('2011-04-15')), "holiday"] = 0
df.loc[get_day(pd.to_datetime('2012-04-16')), "holiday"] = 0
df.loc[get_day(pd.to_datetime('2011-11-25')), "holiday"] = 1
df.loc[get_day(pd.to_datetime('2012-11-23')), "holiday"] = 1
df.loc[get_day(pd.to_datetime('2012-05-21')), "holiday"] = 1
df.loc[get_day(pd.to_datetime('2012-06-01')), "holiday"] = 1


# 高峰时段
def is_peak(row):
    if row['workingday'] == 1 and (row['hour'] == 8 or 17 <= row['hour'] <= 18 or row['hour'] == 12):
        return 1
    if row['workingday'] == 0 and 10 <= row['hour'] <= 19:
        return 1
    return 0
df['peak'] = df.apply(is_peak, axis=1)

# 特殊节假日修正
df['holiday'] = df.apply(lambda x: 1 if (x['year'] == 2012 and x['month'] == 10 and x['day'] in [30]) else x['holiday'], axis=1)
df['holiday'] = df.apply(lambda x: 1 if (x['month'] == 12 and x['day'] in [24, 26, 31]) else x['holiday'], axis=1)
df['workingday'] = df.apply(lambda x: 0 if (x['month'] == 12 and x['day'] in [24, 31]) else x['workingday'], axis=1)

df['ideal'] = df.apply(lambda x: 1 if (x['temp'] > 27 and x['windspeed'] < 30) else 0, axis=1)
df['sticky'] = df.apply(lambda x: 1 if (x['workingday'] == 1 and x['humidity'] >= 60) else 0, axis=1)


# 对训练集做可视化
train_df = df[df['_data'] == 'train'].copy()

# 租赁数量分布
plt.figure(figsize=(8,4))
sns.histplot(train_df['count'], bins=30, kde=True)
plt.title('Distribution of Rental Count')
plt.xlabel('Count')
plt.show()

# 不同季节的租赁量
plt.figure(figsize=(8,4))
sns.boxplot(x='season', y='count', data=train_df)
plt.title('Rental Count by Season')
plt.xlabel('Season')
plt.ylabel('Count')
plt.show()

# 小时与租赁量关系
plt.figure(figsize=(12,4))
sns.boxplot(x='hour', y='count', data=train_df)
plt.title('Rental Count by Hour')
plt.xlabel('Hour')
plt.ylabel('Count')
plt.show()

# 工作日与租赁量关系
plt.figure(figsize=(8,4))
sns.boxplot(x='workingday', y='count', data=train_df)
plt.title('Rental Count by Working Day')
plt.xlabel('Working Day')
plt.ylabel('Count')
plt.show()

# 天气与租赁量关系
plt.figure(figsize=(8,4))
sns.boxplot(x='weather', y='count', data=train_df)
plt.title('Rental Count by Weather')
plt.xlabel('Weather')
plt.ylabel('Count')
plt.show()


def get_rmsle(y_pred, y_actual):
    diff = np.log(y_pred + 1) - np.log(y_actual + 1)
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)

def get_data():
    data = df[df['_data'] == 'train'].copy()
    return data

def custom_train_test_split(data, cutoff_day=15):
    train = data[data['day'] <= cutoff_day]
    test = data[data['day'] > cutoff_day]
    return train, test

def prep_data(data, input_cols):
    X = data[input_cols].values
    y_r = data['registered_log'].values
    y_c = data['casual_log'].values
    return X, y_r, y_c


def predict_on_validation_set(model, input_cols):
    data = get_data()
    train, test = custom_train_test_split(data)
    X_train, y_train_r, y_train_c = prep_data(train, input_cols)
    X_test, y_test_r, y_test_c = prep_data(test, input_cols)

    model_r = model.fit(X_train, y_train_r)
    y_pred_r = np.exp(model_r.predict(X_test)) - 1

    model_c = model.fit(X_train, y_train_c)
    y_pred_c = np.exp(model_c.predict(X_test)) - 1

    y_pred_comb = np.round(y_pred_r + y_pred_c)
    y_pred_comb[y_pred_comb < 0] = 0

    y_test_comb = np.exp(y_test_r) + np.exp(y_test_c) - 2

    score = get_rmsle(y_pred_comb, y_test_comb)
    return (y_pred_comb, y_test_comb, score)

df_test = df[df['_data'] == 'test'].copy()

def predict_on_test_set(model, x_cols):
    df_train = df[df['_data'] == 'train'].copy()
    X_train = df_train[x_cols].values
    y_train_cas = df_train['casual_log'].values
    y_train_reg = df_train['registered_log'].values

    X_test = df_test[x_cols].values

    casual_model = model.fit(X_train, y_train_cas)
    y_pred_cas = casual_model.predict(X_test)
    y_pred_cas = np.exp(y_pred_cas) - 1
    registered_model = model.fit(X_train, y_train_reg)
    y_pred_reg = registered_model.predict(X_test)
    y_pred_reg = np.exp(y_pred_reg) - 1
    return y_pred_cas + y_pred_reg

# 随机森林参数
params_rf = {
    'n_estimators': 1000,
    'max_depth': 15,
    'random_state': 0,
    'min_samples_split': 5,
    'n_jobs': -1
}
rf_model = RandomForestRegressor(**params_rf)
rf_cols = [
    'weather', 'temp', 'atemp', 'windspeed',
    'workingday', 'season', 'holiday', 'sticky',
    'hour', 'dow', 'woy', 'peak'
]

(rf_p, rf_t, rf_score) = predict_on_validation_set(rf_model, rf_cols)
print("RF RMSLE:", rf_score)

# GBDT参数（已修正loss）
params_gbm = {
    'n_estimators': 150,
    'max_depth': 5,
    'random_state': 0,
    'min_samples_leaf': 10,
    'learning_rate': 0.1,
    'subsample': 0.7,
    'loss': 'squared_error'  # 修正
}
gbm_model = GradientBoostingRegressor(**params_gbm)
gbm_cols = [
    'weather', 'temp', 'atemp', 'humidity', 'windspeed',
    'holiday', 'workingday', 'season',
    'hour', 'dow', 'year', 'ideal', 'count_season',
]

(gbm_p, gbm_t, gbm_score) = predict_on_validation_set(gbm_model, gbm_cols)
print("GBM RMSLE:", gbm_score)

y_p = np.round(.2 * rf_p + .8 * gbm_p)
print("Blended RMSLE:", get_rmsle(y_p, rf_t))


rf_pred = predict_on_test_set(rf_model, rf_cols)
gbm_pred = predict_on_test_set(gbm_model, gbm_cols)
y_pred = np.round(.2 * rf_pred + .8 * gbm_pred)

df_test['count'] = y_pred
final_df = df_test[['datetime', 'count']].copy()
final_df.to_csv('/kaggle/working/submission.csv', index=False)


importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = rf_cols

plt.figure(figsize=(10,6))
plt.title("Random Forest Feature Importance")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
plt.ylabel("Importance")
plt.xlabel("Features")
plt.show()

