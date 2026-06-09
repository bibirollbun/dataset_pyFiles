import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train.info()


test.info()


train.describe()


train.columns


target = 'accident_risk'


train.corr


for col in train.select_dtypes(include=['object']).columns:
    print(f'\nColumn: {col}')
    print(train[col].unique())


from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in ['road_type', 'lighting', 'weather', 'time_of_day']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le



train.head()


test.head()


bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

for col in bool_cols:
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])


train.head()


test.head()


X_train = train.drop(columns=['id', 'accident_risk'])
y_train = train[target]

X_test = test.drop(columns=['id'])


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=300, random_state=42, verbose=2)
model.fit(X_train, y_train)


test_pred = model.predict(X_test)


submission = pd.DataFrame({
    "id": test["id"],
    "accident_risk": test_pred
})

submission.to_csv("submission.csv", index=False)


submission.head()

