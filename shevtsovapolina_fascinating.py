import pandas as pd
import numpy as np
import re
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

# SMAPE метрика
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff) * 100

# Проверка на зеркальность
def is_mirror(digits):
    return digits == digits[::-1]

# Фичи из номера
def extract_plate_features(df):
    df['plate'] = df['plate'].astype(str)
    df['letters'] = df['plate'].str.slice(0, 1) + df['plate'].str.slice(4, 6)
    df['digits'] = df['plate'].str.slice(1, 4)
    df['region'] = df['plate'].str.extract(r'(\d{2,3})$')

    df['has_repeating_digits'] = df['digits'].apply(lambda x: len(set(x)) == 1 if x.isdigit() else False)
    df['has_001'] = df['digits'].apply(lambda x: x == '001')
    df['has_123'] = df['digits'].apply(lambda x: x == '123')
    df['has_777'] = df['digits'].apply(lambda x: x == '777')
    df['digits_sum'] = df['digits'].apply(lambda x: sum(int(i) for i in x) if x.isdigit() else 0)
    df['is_beautiful'] = df[['has_repeating_digits', 'has_001', 'has_123', 'has_777']].any(axis=1)
    df['is_mirror'] = df['digits'].apply(lambda x: is_mirror(x) if x.isdigit() else False)
    prestige_series = ['АМР', 'МОО', 'ЕКХ']
    df['is_prestige_series'] = df['letters'].isin(prestige_series)
    return df

# Фичи из даты
def extract_date_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['year_month'] = df['date'].dt.to_period('M').astype(str)
    return df

train = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')
submission = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv')


train = extract_plate_features(train)
test = extract_plate_features(test)
train = extract_date_features(train)
test = extract_date_features(test)


letter_freq = train['letters'].value_counts(normalize=True)
digit_freq = train['digits'].value_counts(normalize=True)
train['letters_freq'] = train['letters'].map(letter_freq)
test['letters_freq'] = test['letters'].map(letter_freq)
train['digits_freq'] = train['digits'].map(digit_freq)
test['digits_freq'] = test['digits'].map(digit_freq)


train['digit_uniques'] = train['digits'].apply(lambda x: len(set(x)) if x.isdigit() else 0)
test['digit_uniques'] = test['digits'].apply(lambda x: len(set(x)) if x.isdigit() else 0)


train['is_summer'] = train['month'].isin([6, 7, 8])
test['is_summer'] = test['month'].isin([6, 7, 8])
train['is_autumn'] = train['month'].isin([9, 10, 11])
test['is_autumn'] = test['month'].isin([9, 10, 11])


avg_price_by_month = train.groupby('year_month')['price'].mean().to_dict()
train['avg_price_month'] = train['year_month'].map(avg_price_by_month)
test['avg_price_month'] = test['year_month'].map(avg_price_by_month)


features = [
    'letters', 'digits', 'region',
    'year', 'month', 'day', 'year_month',
    'has_repeating_digits', 'has_001', 'has_123', 'has_777',
    'digits_sum', 'is_beautiful', 'is_mirror', 'is_prestige_series',
    'letters_freq', 'digits_freq', 'digit_uniques',
    'is_summer', 'is_autumn', 'avg_price_month'
]

target = 'price'
cat_features = ['letters', 'digits', 'region', 'year_month']

X_train, X_val, y_train, y_val = train_test_split(train[features], train[target], test_size=0.2, random_state=42)
train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)
test_pool = Pool(test[features], cat_features=cat_features)

model = CatBoostRegressor(
    iterations=8000,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=10,
    bagging_temperature=0.7,
    loss_function='MAE',
    early_stopping_rounds=100,
    verbose=100,
    random_seed=42
)

model.fit(train_pool, eval_set=val_pool)

# Предсказание и SMAPE
val_preds = model.predict(X_val)
val_smape = smape(y_val.values, val_preds)
print(f"SMAPE на валидации: {val_smape:.2f}%")

# Предсказания на тест
submission['price'] = model.predict(test_pool).astype(int)
submission.to_csv('submission_catboost_with_manual_smape.csv', index=False)

print("✅")

