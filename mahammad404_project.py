import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])


def engineer_features(df):
    df['Duration_per_kg'] = df['Duration'] / df['Weight']
    df['HRxDuration'] = df['Heart_Rate'] * df['Duration']
    df['BMI'] = df['Weight'] / ((df['Height'] / 100)**2)
    df['BMI_squared'] = df['BMI']**2
    df['Duration_squared'] = df['Duration'] ** 2
    df['WeightxDuration'] = df['Weight'] * df['Duration']
    df['AgexHeartRate'] = df['Age'] * df['Heart_Rate']
    df['Log_Duration'] = np.log1p(df['Duration'])
    df['Log_Weight'] = np.log1p(df['Weight'])
    df['Log_HeartRate'] = np.log1p(df['Heart_Rate'])
    df['Log_BodyTemp'] = np.log1p(df['Body_Temp'])
    df['Log_Age'] = np.log1p(df['Age'])
    return df

train = engineer_features(train)
test = engineer_features(test)


train['log_Calories'] = np.log1p(train['Calories'])


features = [col for col in train.columns if col not in ['id', 'Calories', 'log_Calories']]


X = train[features]
y = train['log_Calories']
X_test = test[features]


model = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=9,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)


y_pred = model.predict(X_test)
test['Calories'] = np.expm1(y_pred)


submission = test[['id', 'Calories']]
submission.to_csv('submission.csv', index=False)

