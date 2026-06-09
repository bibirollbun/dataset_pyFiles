import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from supplemental_russian import REGION_CODES, GOVERNMENT_CODES


df_train = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
df_test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')
df_train.head()


def separation(df):
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    df['plate_code'] = df['plate'].str.slice(0, 6)
    df['region_code'] = df['plate'].str.slice(6)
    df['numbers'] = df['plate_code'].str.slice(1, 4)
    df['first_letter'] = df['plate_code'].str.slice(0, 1)
    df['last_letters'] = df['plate_code'].str.slice(4)
    df['letters'] = df['first_letter'] + df['last_letters']

    df.drop(columns=['first_letter', 'last_letters', 'plate_code', 'plate', 'date'], inplace=True)
    return df
train = separation(df_train)
test = separation(df_test)
train.info()


def check_government_codes(row):
    letters = str(row['letters'])
    numbers = int(row['numbers'])  # Преобразуем в число для сравнения с диапазоном
    region_code = str(row['region_code'])
    
    # Проверяем каждую запись в GOVERNMENT_CODES
    for key, value in GOVERNMENT_CODES.items():
        key_letters, key_range, key_region = key
        
        # Сравниваем все компоненты номера
        if (letters == key_letters and
            key_range[0] <= numbers <= key_range[1] and
            region_code == key_region):
            return value  # Возвращаем найденное значение
    
    # Если ничего не найдено
    return ('Неизвестно', 0, 0, 0)

# Применяем функцию к каждой строке DataFrame
train[['gov_description', 'is_forbidden_to_buy', 'have_advantage_on_road', 'level_of_significance']] = train.apply(
    lambda row: pd.Series(check_government_codes(row)), 
    axis=1
)

test[['gov_description', 'is_forbidden_to_buy', 'have_advantage_on_road', 'level_of_significance']] = train.apply(
    lambda row: pd.Series(check_government_codes(row)), 
    axis=1
)


train_sorted = train.sort_values(by='level_of_significance', ascending=False)
train_sorted.head()


train.drop(columns=['gov_description', 'letters'], inplace=True)
test.drop(columns=['gov_description', 'letters'], inplace=True)
train.head()


for region in REGION_CODES.keys():
    train[region] = 0
    test[region] = 0

# Заполняем единицами соответствующие регионы
def fill_regions(df):
    for region, codes in REGION_CODES.items():
        # Преобразуем region_code в строку для сравнения
        str_codes = [str(code) for code in codes]
        # Устанавливаем 1 для строк, где region_code находится в кодах региона
        df.loc[df['region_code'].astype(str).isin(str_codes), region] = 1
    return df

train = fill_regions(train)
test = fill_regions(test)

train.head()


test.info(verbose=True, memory_usage='deep', show_counts=True)


train.drop(columns=['region_code'], inplace=True)
test.drop(columns=['region_code'], inplace=True)
train.info()


train['numbers'] = train['numbers'].astype(int)
test['numbers'] = test['numbers'].astype(int)
train.info()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from catboost import CatBoostRegressor

# Функция SMAPE
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# --- Подготовка данных ---
y = np.log1p(train['price'])  # логарифмируем цену
X = train.drop(columns=['price', 'id'], axis=1)

# Разделение на train / val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 1. RandomForest ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred_log = rf_model.predict(X_val)
rf_pred = np.expm1(rf_pred_log)

# --- 3. CatBoost ---
cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',  # мы логарифмируем, поэтому MAPE лучше не использовать
    random_seed=42,
    verbose=0
)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
cat_pred_log = cat_model.predict(X_val)
cat_pred = np.expm1(cat_pred_log)

# --- Blending ---
y_val_true = np.expm1(y_val)  # возвращаем реальные значения из логарифма
y_val_pred = (rf_pred + cat_pred) / 3

score = smape(y_val_true, y_val_pred)
print(f'SMAPE (Blended, log-transformed): {score:.4f}%')

# --- Предсказания на тест ---
submission_ID = test['id']
X_test = test.drop(columns=['price', 'id'], axis=1)

rf_test_log = rf_model.predict(X_test)
cat_test_log = cat_model.predict(X_test)

rf_test = np.expm1(rf_test_log)
cat_test = np.expm1(cat_test_log)

y_test_pred = (rf_test + cat_test) / 2

# --- Submission ---
submission = pd.DataFrame({'id': submission_ID, 'price': y_test_pred})
submission.to_csv('submit_blended_log.csv', index=False)

print("Результаты сохранены в submit_blended_log.csv")

