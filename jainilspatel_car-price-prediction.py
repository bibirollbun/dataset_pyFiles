import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import re
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
import lightgbm as lgb
from sklearn.metrics import make_scorer


try:
    from supplemental_english import REGION_CODES, GOVERNMENT_CODES
except ImportError:
    REGION_CODES = {}
    GOVERNMENT_CODES = {}


def parse_date(date_str):
    """
    Parses a date string that can be either a single date (e.g. "2024-12-26 00:00:00")
    or a range (e.g. "02/17/2021 - 05/01/2021"). In the case of a range, the function takes the first date.
    """
    if pd.isnull(date_str):
        return np.nan
    # If the date has a range (using ' - '), take the first part.
    if ' - ' in date_str:
        date_str = date_str.split(' - ')[0].strip()
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except Exception:
        return np.nan


def extract_plate_features(plate):
    features = {}
    # Extract region (last 2 or 3 digits)
    m = re.search(r'(\d{2,3})$', plate)
    features['region'] = m.group(1) if m else np.nan
    # Extract all groups of digits
    digit_groups = re.findall(r'\d+', plate)
    if len(digit_groups) >= 1:
        # Usually the first group is the main numeric part
        features['number_part'] = digit_groups[0]
    else:
        features['number_part'] = np.nan
    # Remove digits to get the letters part
    letters = re.sub(r'\d+', '', plate)
    features['letters'] = letters
    features['plate_length'] = len(plate)
    return features


def check_government_code(letters, number_str, region):
    try:
        number_val = int(number_str)
    except:
        return (0, 0, 0) 
    for (letters_key, (num_min, num_max), region_key), info in GOVERNMENT_CODES.items():
        if letters_key == letters and region_key == region:
            if num_min <= number_val <= num_max:
                _, is_forbidden, has_advantage, significance = info
                return (is_forbidden, has_advantage, significance)
    return (0, 0, 0)


train = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')
sample_submission = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv')


for df in [train, test]:
    df['parsed_date'] = df['date'].apply(parse_date)
    df['year'] = df['parsed_date'].dt.year
    df['month'] = df['parsed_date'].dt.month
    df['day'] = df['parsed_date'].dt.day
    df['weekday'] = df['parsed_date'].dt.weekday
    reference_date = pd.to_datetime("2021-01-01")
    df['days_since_2021'] = (df['parsed_date'] - reference_date).dt.days


def add_plate_features(df):
    plate_feats = df['plate'].apply(extract_plate_features)
    plate_df = pd.DataFrame(list(plate_feats))
    for col in ['region', 'letters']:
        plate_df[col] = plate_df[col].astype(str)
    gov_features = plate_df.apply(
        lambda row: check_government_code(row['letters'], row['number_part'], row['region']),
        axis=1)
    plate_df[['is_forbidden','has_advantage','significance_level']] = pd.DataFrame(gov_features.tolist(), index=plate_df.index)
    plate_df['number_part'] = pd.to_numeric(plate_df['number_part'], errors='coerce')
    df = pd.concat([df, plate_df], axis=1)
    return df
train = add_plate_features(train)
test = add_plate_features(test)


for col in ['region', 'letters']:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col + '_enc'] = le.transform(train[col].astype(str))
    test[col + '_enc'] = le.transform(test[col].astype(str))


price_cap = train['price'].quantile(0.995)
train['price'] = np.where(train['price'] > price_cap, price_cap, train['price'])


train['price_log'] = np.log1p(train['price'])


features = [
    'days_since_2021',  
    'weekday',          
    'plate_length', 
    'number_part',
    'region_enc', 
    'letters_enc',
    'is_forbidden',
    'has_advantage',
    'significance_level']
X = train[features]
y_log = train['price_log']
X_test = test[features]


def smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.where(denominator == 0, 0, numerator / denominator)) * 100


X_train, X_valid = train_test_split(train.index, test_size=0.2, shuffle=True, random_state=42)
lgb_train = lgb.Dataset(X.loc[X_train], y_log.loc[X_train])
lgb_valid = lgb.Dataset(X.loc[X_valid], y_log.loc[X_valid], reference=lgb_train)


params = {
    'objective': 'regression',
    'metric': 'mae', 
    'verbosity': -1,
    'seed': 42}
callbacks = [
    lgb.early_stopping(stopping_rounds=50),
    lgb.log_evaluation(period=50)]
model = lgb.train(
    params,
    train_set=lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_valid],
    callbacks=callbacks)


y_valid_pred_log = model.predict(X.loc[X_valid], num_iteration=model.best_iteration)
y_valid_pred = np.expm1(y_valid_pred_log)


y_valid_true = train['price'].loc[X_valid]
val_smape = smape(y_valid_true.values, y_valid_pred)
print(f"Validation SMAPE: {val_smape:.4f}%")


full_train = lgb.Dataset(X, y_log)
model_full = lgb.train(params, full_train, num_boost_round=model.best_iteration)


test_pred_log = model_full.predict(X_test)
test_pred = np.expm1(test_pred_log)


submission = pd.DataFrame({
    'id': test['id'],
    'price': test_pred
})
submission.to_csv('submission.csv', index=False)
print("Submission file created!")


import matplotlib.pyplot as plt
avg_price_by_month = (
    train
    .set_index('parsed_date')
    .resample('M')['price']
    .mean()
    .reset_index())
plt.figure(figsize=(12,6))
plt.plot(avg_price_by_month['parsed_date'], avg_price_by_month['price'], marker='o')
plt.title('Average Price over Time (Monthly)')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.grid(True)
plt.show()




