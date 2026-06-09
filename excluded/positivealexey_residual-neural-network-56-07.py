import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
from datetime import datetime
from supplemental_english import REGION_CODES, GOVERNMENT_CODES

def unique_char_count(plate):
    return len(set(plate))

def same_letter_count(series):
    return len(set(series))

def same_digit_count(number):
    return len(set(number))

def is_palindrome(number):
    return number == number[::-1] if len(number) > 1 else False

def sum_of_digits(number):
    return sum(int(d) for d in number if d.isdigit())

def digit_range(number):
    digits = [int(d) for d in number if d.isdigit()]
    return max(digits) - min(digits) if digits else 0

def has_beautiful_pattern(number):
    beautiful = ['123', '321', '111', '222', '333', '555', '777', '999', '100', '001', '505', '808', '909', '606', '313', '212', '101']
    return any(pat in number for pat in beautiful)

def has_digit_like_letters(series):
    digit_like = set('ĞœĞ’Ğ�Ğ�Ğ¢Ğ•Ğ Ğ¡Ğ¥')
    return any(ch in digit_like for ch in series)

def count_digit_like_letters(series):
    digit_like = set('ĞœĞ’Ğ�Ğ�Ğ¢Ğ•Ğ Ğ¡Ğ¥')
    return sum(ch in digit_like for ch in series)

def count_letters(plate):
    return sum(1 for ch in plate if ch.isalpha())

def count_digits(plate):
    return sum(1 for ch in plate if ch.isdigit())

def even_sum_of_digits(number):
    return sum_of_digits(number) % 2 == 0

def odd_sum_of_digits(number):
    return sum_of_digits(number) % 2 == 1

def is_region_rare(region_code):
    count = sum(region_code in codes for codes in REGION_CODES.values())
    return count == 1

def region_rank(region_code):
    all_codes = [code for codes in REGION_CODES.values() for code in codes]
    try:
        return all_codes.index(region_code)
    except ValueError:
        return -1

def is_holiday(date_str):
    holidays = {'01-01', '01-07', '02-23', '03-08', '05-01', '05-09', '06-12', '11-04'}
    date = pd.to_datetime(date_str)
    return date.strftime('%m-%d') in holidays

def region_type(region_name):
    if 'Republic' in region_name:
        return 'republic'
    if 'Krai' in region_name:
        return 'krai'
    if 'Oblast' in region_name:
        return 'oblast'
    if 'Autonomous' in region_name:
        return 'autonomous'
    if region_name in ['Moscow', 'Saint Petersburg', 'Sevastopol']:
        return 'federal_city'
    return 'other'

def is_capital(region_name):
    return region_name in ['Moscow', 'Saint Petersburg', 'Sevastopol']

def is_disputed(region_name):
    disputed = [
        'Republic of Crimea', 'Sevastopol', "Donetsk People's Republic",
        "Luhansk People's Republic", 'Zaporizhzhia Oblast', 'Kherson Oblast',
        'Occupational Administration of Kharkiv Oblast'
    ]
    return region_name in disputed

def region_code_count(region_code):
    for codes in REGION_CODES.values():
        if region_code in codes:
            return len(codes)
    return 1

def gov_description(series, number, region_code):
    number = int(number) if number.isdigit() else 0
    for (gov_series, (num_from, num_to), gov_region), (description, _, _, _) in GOVERNMENT_CODES.items():
        if gov_series == series and num_from <= number <= num_to and gov_region == region_code:
            return description
    return 'Regular'

def gov_series_frequency(series):
    return sum(1 for (gov_series, _, _), _ in GOVERNMENT_CODES.items() if gov_series == series)

def extract_region_code(plate):
    digits = ''
    for char in reversed(plate):
        if char.isdigit():
            digits = char + digits
        else:
            break
    return digits

def get_region_info(region_code):
    for region, codes in REGION_CODES.items():
        if region_code in codes:
            return region
    return "Unknown"

def extract_series(plate):
    first_letter = plate[0] if plate[0].isalpha() else ''
    last_letters = ''
    for char in reversed(plate):
        if char.isalpha():
            last_letters = char + last_letters
            if len(last_letters) == 2:
                break
    return first_letter + last_letters

def extract_number(plate):
    if len(plate) >= 6 and plate[1:4].isdigit():
        return plate[1:4]
    digits = ''
    for char in plate[1:]:
        if char.isdigit() and len(digits) < 3:
            digits += char
        if len(digits) == 3:
            break
    return digits

def get_government_info(series, number, region_code):
    number = int(number) if number.isdigit() else 0
    for (gov_series, (num_from, num_to), gov_region), (description, forbidden, advantage, significance) in GOVERNMENT_CODES.items():
        if (gov_series == series and 
            num_from <= number <= num_to and 
            gov_region == region_code):
            return {
                'is_government': True,
                'description': description,
                'forbidden': bool(forbidden),
                'advantage': bool(advantage),
                'significance': significance
            }
    return {
        'is_government': False,
        'description': 'Regular',
        'forbidden': False,
        'advantage': False,
        'significance': 0
    }

def has_repeating_chars(text):
    for i in range(len(text)-1):
        if text[i] == text[i+1]:
            return True
    return False

def has_special_combinations(text):
    special_combinations = ['000', '111', '222', '333', '444', '555', '666', '777', '888', '999',
                          'AAA', 'XXX', 'OOO', 'MMM', 'TTT', 'HHH', 'CCC']
    for combo in special_combinations:
        if combo in text:
            return True
    return False

def is_mirror_number(number):
    if len(number) < 2:
        return False
    return all(c == number[0] for c in number)

def extract_time_features(date_str):
    date = pd.to_datetime(date_str)
    return {
        'year': date.year,
        'month': date.month,
        'day_of_week': date.dayofweek,
        'is_weekend': date.dayofweek >= 5,
        'hour': date.hour,
        'time_of_day': 'night' if 0 <= date.hour < 6 else 
                      'morning' if 6 <= date.hour < 12 else 
                      'afternoon' if 12 <= date.hour < 18 else 
                      'evening',
        'day': date.day,
        'is_holiday': is_holiday(date_str),
        'is_even_day': date.day % 2 == 0,
        'is_odd_day': date.day % 2 == 1
    }

def add_group_stats(features, df, group_col, target_col='price', prefix=None):
    if prefix is None:
        prefix = group_col
    group_stats = df.groupby(group_col)[target_col].agg(['mean', 'median', 'std', 'min', 'max', 'count']).reset_index()
    group_stats.columns = [group_col, f'{prefix}_mean', f'{prefix}_median', f'{prefix}_std', f'{prefix}_min', f'{prefix}_max', f'{prefix}_count']
    features = features.merge(group_stats, how='left', left_on=group_col, right_on=group_col)
    return features

def create_features(df):
    features = pd.DataFrame()
    
    features['region_code'] = df['plate'].apply(extract_region_code)
    features['region'] = features['region_code'].apply(get_region_info)
    features['series'] = df['plate'].apply(extract_series)
    features['number'] = df['plate'].apply(extract_number)
    features['plate_length'] = df['plate'].apply(len)
    
    features['unique_char_count'] = df['plate'].apply(unique_char_count)
    features['letter_count'] = df['plate'].apply(count_letters)
    features['digit_count'] = df['plate'].apply(count_digits)
    features['same_letter_count'] = features['series'].apply(same_letter_count)
    features['same_digit_count'] = features['number'].apply(same_digit_count)
    features['is_palindrome'] = features['number'].apply(is_palindrome)
    features['sum_of_digits'] = features['number'].apply(sum_of_digits)
    features['digit_range'] = features['number'].apply(digit_range)
    features['has_beautiful_pattern'] = features['number'].apply(has_beautiful_pattern)
    features['has_digit_like_letters'] = features['series'].apply(has_digit_like_letters)
    features['count_digit_like_letters'] = features['series'].apply(count_digit_like_letters)
    
    gov_info = df.apply(lambda x: get_government_info(
        extract_series(x['plate']), 
        extract_number(x['plate']), 
        extract_region_code(x['plate'])
    ), axis=1)
    
    features['is_government'] = gov_info.apply(lambda x: x['is_government'])
    features['plate_forbidden'] = gov_info.apply(lambda x: x['forbidden'])
    features['plate_advantage'] = gov_info.apply(lambda x: x['advantage'])
    features['plate_significance'] = gov_info.apply(lambda x: x['significance'])
    
    features['has_repeating_chars'] = df['plate'].apply(has_repeating_chars)
    features['has_special_combinations'] = df['plate'].apply(has_special_combinations)
    features['is_mirror_number'] = features['number'].apply(is_mirror_number)
    
    time_features = df['date'].apply(extract_time_features)
    features['year'] = time_features.apply(lambda x: x['year'])
    features['month'] = time_features.apply(lambda x: x['month'])
    features['day_of_week'] = time_features.apply(lambda x: x['day_of_week'])
    features['is_weekend'] = time_features.apply(lambda x: x['is_weekend'])
    features['hour'] = time_features.apply(lambda x: x['hour'])
    features['time_of_day'] = time_features.apply(lambda x: x['time_of_day'])
    features['day'] = time_features.apply(lambda x: x['day'])
    features['is_holiday'] = time_features.apply(lambda x: x['is_holiday'])
    features['is_even_day'] = time_features.apply(lambda x: x['is_even_day'])
    features['is_odd_day'] = time_features.apply(lambda x: x['is_odd_day'])
    
    features['is_region_rare'] = features['region_code'].apply(is_region_rare)
    features['region_rank'] = features['region_code'].apply(region_rank)
    
    features['region_type'] = features['region'].apply(region_type)
    features['is_capital'] = features['region'].apply(is_capital)
    features['is_disputed'] = features['region'].apply(is_disputed)
    features['region_code_count'] = features['region_code'].apply(region_code_count)
    
    features['gov_description'] = df.apply(lambda x: gov_description(
        extract_series(x['plate']),
        extract_number(x['plate']),
        extract_region_code(x['plate'])
    ), axis=1)
    features['gov_series_frequency'] = features['series'].apply(gov_series_frequency)
    
    features['even_sum_of_digits'] = features['number'].apply(even_sum_of_digits)
    features['odd_sum_of_digits'] = features['number'].apply(odd_sum_of_digits)
    
    df = df.copy()
    df['region_code'] = features['region_code']
    df['series'] = features['series']
    df['gov_description'] = features['gov_description'] if 'gov_description' in features else df['plate'].apply(lambda x: gov_description(
        extract_series(x),
        extract_number(x),
        extract_region_code(x)
    ))

    if 'price' in df.columns:
        features = add_group_stats(features, df, 'region_code', 'price', prefix='region')
        features = add_group_stats(features, df, 'series', 'price', prefix='series')
        features = add_group_stats(features, df, 'gov_description', 'price', prefix='govdesc')
    
    num_cols = ['plate_length', 'unique_char_count', 'letter_count', 'digit_count', 'sum_of_digits', 'digit_range',
                'region_rank', 'region_code_count', 'gov_series_frequency',
                'region_mean', 'region_median', 'series_mean', 'series_median', 'govdesc_mean', 'govdesc_median']
    for col in num_cols:
        if col in features.columns:
            features[f'log1p_{col}'] = np.log1p(features[col].astype(float))
    
    return features


df = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')

features = create_features(df)

features.to_csv('features.csv', index=False) 


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')


features = pd.read_csv('/kaggle/working/features.csv')
df = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')


y = np.log1p(df['price'])

categorical_features = features.select_dtypes(include=['object', 'bool']).columns.tolist()
numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()

for col in ['id', 'price', 'number']:
    if col in categorical_features:
        categorical_features.remove(col)
    if col in numeric_features:
        numeric_features.remove(col)

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)


numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

input_dim = X_train_processed.shape[1]

model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(384, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',
              metrics=[keras.metrics.RootMeanSquaredError()])

callbacks = [
    keras.callbacks.EarlyStopping(patience=24, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=6, factor=0.5)
]

history = model.fit(
    X_train_processed, y_train,
    validation_data=(X_test_processed, y_test),
    epochs=500,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)


log_loss, log_rmse = model.evaluate(X_test_processed, y_test, verbose=0)
y_pred_log = model.predict(X_test_processed).flatten()
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)
rmse_real = np.sqrt(np.mean((y_pred - y_true) ** 2))
print(f'RMSE (log target, Ğ² Ñ€ÑƒĞ±Ğ»Ñ�Ñ…): {rmse_real:.2f}')


import joblib

model.save('car_plate_price_model_tf_log.h5')
joblib.dump(preprocessor, 'preprocessor_tf_log.joblib') 


import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

TEST_PATH = '/kaggle/input/russian-car-plates-prices-prediction/test.csv'
test_df = pd.read_csv(TEST_PATH)

features_test = create_features(test_df)


preprocessor = joblib.load('preprocessor_tf_log.joblib')
model = tf.keras.models.load_model('car_plate_price_model_tf_log.h5', compile=False)


X_test_processed = preprocessor.transform(features_test)

y_pred_log = model.predict(X_test_processed).flatten()

y_pred = np.expm1(y_pred_log)

submission = pd.DataFrame({
    'id': test_df['id'],
    'predicted_price': y_pred
})

submission.to_csv('my_submission.csv', index=False)


import pandas as pd
import numpy as np

pred = pd.read_csv('/kaggle/working/my_submission.csv')
true = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv').rename(columns={'price': 'true_price'})

df = pred.merge(true, on='id', how='inner')

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

score = smape(df['true_price'], df['predicted_price'])
print(f'SMAPE: {score:.2f}%')


import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform
import joblib

features = pd.read_csv('/kaggle/working/features.csv')
df = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')


y = np.log1p(df['price'])

cat_features = features.select_dtypes(include=['object', 'bool']).columns.tolist()
for col in ['id', 'price', 'number']:
    if col in cat_features:
        cat_features.remove(col)

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

base_model = CatBoostRegressor(
    iterations=4000,
    loss_function='RMSE',
    eval_metric='RMSE',
    early_stopping_rounds=200,
    verbose=100,
    random_seed=42
)

param_dist = {
    'learning_rate': uniform(0.005, 0.05),
    'depth': randint(6, 11),
    'l2_leaf_reg': randint(2, 10),
    'bagging_temperature': uniform(0.1, 1.0),
    'subsample': uniform(0.7, 0.3),
    'grow_policy': ['Lossguide', 'SymmetricTree'],
    'min_data_in_leaf': randint(1, 6),
    'max_leaves': randint(31, 129),
    'random_strength': uniform(0.5, 2.0),
    'boosting_type': ['Plain', 'Ordered'],
    'od_type': ['Iter', 'IncToDec']
}

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=5,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train, cat_features=cat_features)


print('Ğ›ÑƒÑ‡ÑˆĞ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹:', random_search.best_params_)


print('Ğ›ÑƒÑ‡ÑˆĞ¸Ğ¹ RMSE (log1p):', -random_search.best_score_)


best_model = random_search.best_estimator_
best_model.save_model('catboost_plate_model_best.cbm')
joblib.dump(cat_features, 'catboost_cat_features.joblib')


import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor

TEST_PATH = '/kaggle/input/russian-car-plates-prices-prediction/test.csv'
test_df = pd.read_csv(TEST_PATH)

features_test = create_features(test_df)

model = CatBoostRegressor()
model.load_model('catboost_plate_model_best.cbm')
cat_features = joblib.load('catboost_cat_features.joblib')

y_pred_log = model.predict(features_test)

y_pred = np.expm1(y_pred_log)

submission = pd.DataFrame({
    'id': test_df['id'],
    'predicted_price': y_pred
})
submission.to_csv('my_submission_catboost.csv', index=False)


import pandas as pd
import numpy as np

pred = pd.read_csv('/kaggle/working/my_submission_catboost.csv')
true = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv').rename(columns={'price': 'true_price'})

df = pred.merge(true, on='id', how='inner')

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

score = smape(df['true_price'], df['predicted_price'])
print(f'SMAPE: {score:.2f}%')


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt

features = pd.read_csv('features.csv')
train_df = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')

X = features.copy()
y = train_df['price']

y = np.log1p(y)

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

joblib.dump(numeric_cols, 'numeric_cols.joblib')
joblib.dump(categorical_cols, 'categorical_cols.joblib')

X_cat = pd.get_dummies(X[categorical_cols], drop_first=True)

joblib.dump(X_cat.columns, 'categorical_columns_after_encoding.joblib')


scaler = RobustScaler()

joblib.dump(scaler, 'robust_scaler.joblib')

X_num = pd.DataFrame(scaler.fit_transform(X[numeric_cols]), columns=numeric_cols)

X_num = X_num.replace([np.inf, -np.inf], np.nan)
X_num = X_num.fillna(X_num.mean())

X_processed = pd.concat([X_num, X_cat], axis=1)

joblib.dump(X_processed.columns, 'final_columns.joblib')

X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1_l2
import joblib

def residual_block(x, units, dropout_rate=0.3, l1_factor=1e-5, l2_factor=1e-4):
    shortcut = x
    
    x = layers.Dense(units, 
                    kernel_regularizer=l1_l2(l1=l1_factor, l2=l2_factor))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(units, 
                    kernel_regularizer=l1_l2(l1=l1_factor, l2=l2_factor))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    
    if shortcut.shape[-1] == units:
        x = layers.Add()([shortcut, x])
    
    return x

def create_model(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    
    x = layers.Dense(512, 
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(inputs)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = residual_block(x, 384, dropout_rate=0.4)
    x = residual_block(x, 256, dropout_rate=0.3)
    
    branch1 = layers.Dense(128, 
                          kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    branch1 = layers.LeakyReLU(alpha=0.1)(branch1)
    branch1 = layers.BatchNormalization()(branch1)
    branch1 = layers.Dropout(0.2)(branch1)
    
    branch2 = layers.Dense(128, 
                          kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    branch2 = layers.LeakyReLU(alpha=0.1)(branch2)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.Dropout(0.2)(branch2)
    
    x = layers.Concatenate()([branch1, branch2])
    x = layers.Dense(128, 
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(64, 
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

joblib.dump(X_processed.shape[1], 'input_dim.joblib')

model = create_model(X_processed.shape[1])
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae']
)

joblib.dump(optimizer.get_config(), 'optimizer_config.joblib')

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

joblib.dump(history.history, 'training_history.joblib')

model.save('car_plates_model.h5')

model_config = {
    'input_dim': X_processed.shape[1],
    'model_architecture': model.get_config(),
    'training_params': {
        'batch_size': 32,
        'epochs': 100,
        'early_stopping_patience': 10,
        'reduce_lr_patience': 5,
        'reduce_lr_factor': 0.5,
        'min_lr': 1e-6
    }
}
joblib.dump(model_config, 'model_config.joblib')


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import RobustScaler
import joblib

TEST_PATH = '/kaggle/input/russian-car-plates-prices-prediction/test.csv'
test_df = pd.read_csv(TEST_PATH)

features_test = create_features(test_df)

numeric_cols = joblib.load('numeric_cols.joblib')
categorical_cols = joblib.load('categorical_cols.joblib')
final_columns = joblib.load('final_columns.joblib')

X_test_cat = pd.get_dummies(features_test[categorical_cols], drop_first=True)

scaler = RobustScaler()
X_test_num = pd.DataFrame(
    scaler.fit_transform(features_test[numeric_cols]), 
    columns=numeric_cols
)

X_test_processed = pd.concat([X_test_num, X_test_cat], axis=1)

missing_cols = set(final_columns) - set(X_test_processed.columns)
for col in missing_cols:
    X_test_processed[col] = 0

X_test_processed = X_test_processed[final_columns]

X_test_processed = X_test_processed.replace([np.inf, -np.inf], np.nan)
X_test_processed = X_test_processed.fillna(0)

input_dim = joblib.load('input_dim.joblib')

model = create_model(input_dim)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae']
)

model.load_weights('car_plates_model.h5')

y_pred = np.clip(y_pred, 0, None)
y_pred = y_pred.flatten()

submission = pd.DataFrame({
    'id': test_df['id'],
    'price': y_pred
})

submission.to_csv('my_submission_tensorflow.csv', index=False)


pred = pd.read_csv('/kaggle/working/my_submission_tensorflow.csv')
true = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv')

pred = pred.rename(columns={'price': 'predicted_price'})
true = true.rename(columns={'price': 'true_price'})

df = pred.merge(true, on='id', how='inner')

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

score = smape(df['true_price'], df['predicted_price'])
print(f'SMAPE: {score:.2f}%')


import os

old_path = '/kaggle/working/my_submission.csv'
new_path = '/kaggle/working/submission.csv'

os.rename(old_path, new_path)


sub = pd.read_csv('/kaggle/working/submission.csv')

sub.head()

