import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

rseed = 71


train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
test  = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
sample = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')
#欠損値があるか確認
#print(train.isnull().sum())
#print(test.isnull().sum())
#print(sample.isnull().sum())


#特徴量の作成
train = train.copy()
test  = test.copy()

def create_time_features(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['weekday'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year.map({2011: 0, 2012: 1})
    #通勤時間
    df['peak_hour'] = df['hour'].isin([7, 8, 17, 18]).astype(int) 
    #温度差
    df['temp_atemp_diff'] = abs(df['temp'] - df['atemp']) 
    #気持ち悪さ
    df['temp+humi'] = abs(df['temp'] + df['humidity']) 
    return df

train = create_time_features(train)
test = create_time_features(test)


#特徴量リスト
features = [
    'season', 'holiday', 'workingday', 'weather',
    'temp', 'atemp', 'humidity', 'windspeed',
    'hour', 'weekday', 'year','day','month','peak_hour','temp_atemp_diff','temp+humi'
]


#カテゴリ変数
cat_cols = ['season', 'holiday', 'workingday', 'weather', 'hour', 'weekday', 'year']

all_data = pd.concat([train[cat_cols], test[cat_cols]], axis=0)

for col in cat_cols:
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col])

train[cat_cols] = all_data[:len(train)]
test[cat_cols] = all_data[len(train):]


X = train[features]
y = np.log1p(train['count'])
X_test = test[features]

dtrain = xgb.DMatrix(X, label=y)
dtest = xgb.DMatrix(X_test)
params = {
    'objective': 'reg:squarederror',
    'seed': rseed,
    'eta': 0.1,
    'max_depth': 6
}
model = xgb.train(params, dtrain, num_boost_round=100)

preds = model.predict(dtest)
preds = np.expm1(preds) 
preds = np.clip(preds, 0, None)
sample['count'] = preds
sample.to_csv('submission.csv', index=False)

