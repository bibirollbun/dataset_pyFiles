import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')
sample_submission = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv')


def extract_plate_features(df):
    df['region'] = df['plate'].str.extract(r'(\d{2,3})$').astype('category')  
    df['letters'] = df['plate'].str.extract(r'([A-Z]+)')  
    df['digits'] = df['plate'].str.extract(r'(\d{3})')    
    df['is_repeating'] = df['digits'].apply(lambda x: int(len(set(x)) == 1) if pd.notna(x) else 0)
    df['is_mirror'] = df['digits'].apply(lambda x: int(x == x[::-1]) if pd.notna(x) else 0)
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    return df


train = extract_plate_features(train)
test = extract_plate_features(test)


cat_features = ['region', 'letters']
encoder = LabelEncoder()
for col in cat_features:
    train[col] = encoder.fit_transform(train[col].astype(str))
    test[col] = encoder.transform(test[col].astype(str))


features = ['region', 'letters', 'is_repeating', 'is_mirror', 'dayofweek', 'month', 'year']
X = train[features]
y = train['price']
X_test = test[features]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


import lightgbm as lgb


model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)


val_preds = model.predict(X_val)
mae = mean_absolute_error(y_val, val_preds)
print(f"Validation MAE: {mae:.2f}")


test_preds = model.predict(X_test)
submission = sample_submission.copy()
submission['price'] = test_preds.astype(int) 

submission.to_csv("my_submission.csv", index=False)
print("Submission file saved as my_submission.csv.")


print(submission.head(10))  


print(submission)




