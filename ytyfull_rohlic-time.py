# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

# 1. Чтение файлов
calendar = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv")
test_weights = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv")
inventory = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv")
sales_train = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv")
sales_test = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv")
solution = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv")

train_merged = sales_train.merge(inventory, on=["unique_id", "warehouse"], how="left")
train_merged = train_merged.merge(calendar, on=["warehouse", "date"], how="left")

print("Train merged shape:", train_merged.shape)
print(train_merged.head(3))

test_merged = sales_test.merge(inventory, on=["unique_id", "warehouse"], how="left")
test_merged = test_merged.merge(calendar, on=["warehouse", "date"], how="left")

test_merged = test_merged.merge(test_weights, on="unique_id", how="left")

print("Test merged shape:", test_merged.shape)
print(test_merged.head(3))


!pip install lightgbm


import pandas as pd
from sklearn.preprocessing import LabelEncoder

cols_to_encode = [
    'warehouse',
    'name',
    'L1_category_name_en',
    'L2_category_name_en',
    'L3_category_name_en',
    'L4_category_name_en',
    'holiday_name'
]

train_merged['date'] = pd.to_datetime(train_merged['date'])
test_merged['date'] = pd.to_datetime(test_merged['date'])

train_merged['holiday_name'] = train_merged['holiday_name'].fillna("NoHoliday")
test_merged['holiday_name']  = test_merged['holiday_name'].fillna("NoHoliday")

for col in cols_to_encode:
    le = LabelEncoder()

    all_categories = pd.concat([
        train_merged[col].astype(str),
        test_merged[col].astype(str)
    ], axis=0).unique()

    le.fit(all_categories)

    train_merged[col] = le.transform(train_merged[col].astype(str))
    test_merged[col]  = le.transform(test_merged[col].astype(str))

print(train_merged.head(3))
print(test_merged.head(3))


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def basic_eda(df, df_name="DataFrame"):
    """
    Проводит базовый анализ данных:
     - Размер (shape)
     - Общая инфо (dtypes и т.д.)
     - Описание числовых столбцов
     - Кол-во пропусков
     - Уникальные значения для категориальных и числовых столбцов
     - Примеры простых визуализаций
    """
    
    print(f"=== {df_name} ===")
    print(f"Shape: {df.shape}\n")
    
    print("Info:")
    print(df.info(memory_usage=False))
    print("\n")
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_cols.empty:
        print("=== Numeric Columns Description ===")
        display(numeric_cols.describe().T)
    
    print("\n=== Missing Values (NaN) ===")
    missing = df.isna().sum().sort_values(ascending=False)
    display(missing[missing > 0])  
    
    print("\n=== Unique Values Count ===")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} unique values")
    
    sample_numeric_cols = numeric_cols.columns[:5]  
    if len(sample_numeric_cols) > 0:
        _ = df[sample_numeric_cols].hist(bins=30, figsize=(15, 6))
        plt.suptitle(f"Histograms of first numeric columns in {df_name}")
        plt.show()

    cat_cols = df.select_dtypes(include=['int64', 'object']).columns
    small_cat_cols = [c for c in cat_cols if df[c].nunique() <= 10 and df[c].nunique() > 1]
    if len(small_cat_cols) > 0:
        print(f"\nBarplots for columns with few categories (in {df_name}): {small_cat_cols}")
        for col in small_cat_cols:
            plt.figure(figsize=(6, 4))
            df[col].value_counts().plot(kind='bar')
            plt.title(f"{col} distribution in {df_name}")
            plt.show()
    
    print("\n=============================\n")

basic_eda(train_merged, "TRAIN_MERGED")
basic_eda(test_merged, "TEST_MERGED")



import pandas as pd
import numpy as np

def create_features(df, is_train=False):
    """
    Создаёт набор фичей из исходного df (train_merged или test_merged).
    Параметр is_train=True означает, что в df есть колонка 'sales' (целевая).
    Возвращает:
      X (DataFrame со всеми признаками),
      y (Series с целевой переменной, если is_train=True, иначе None).
    """

    data = df.copy()

    data['date'] = pd.to_datetime(data['date'])
    data['day_of_week'] = data['date'].dt.dayofweek     
    data['month']       = data['date'].dt.month
    data['year']        = data['date'].dt.year

    discount_cols = [f"type_{i}_discount" for i in range(7) if f"type_{i}_discount" in data.columns]
    data['max_discount'] = data[discount_cols].max(axis=1)

    data['has_any_discount'] = (data['max_discount'] > 0).astype(int)

    data['discounted_price'] = data['sell_price_main'] * (1 - data['max_discount'])

    feature_cols = [
        'unique_id',
        'warehouse',
        'total_orders',
        'sell_price_main',
        'max_discount',
        'has_any_discount',
        'discounted_price',
        'holiday_name',
        'holiday',
        'shops_closed',
        'winter_school_holidays',
        'school_holidays',
        'L1_category_name_en',
        'L2_category_name_en',
        'L3_category_name_en',
        'L4_category_name_en',
        'name',               
        'product_unique_id',  
        'day_of_week',
        'month',
        'year'
    ]

    feature_cols = [c for c in feature_cols if c in data.columns]

    X = data[feature_cols].copy()

    y = data['sales'] if (is_train and 'sales' in data.columns) else None

    return X, y

X_train, y_train = create_features(train_merged, is_train=True)

X_test, _ = create_features(test_merged, is_train=False)

print(X_train.head(3))


non_numeric_cols = X_train.select_dtypes(exclude=['number','bool']).columns
if len(non_numeric_cols) == 0:
    print("Все столбцы X_train числовые или булевы.")
else:
    print("В X_train есть нечисловые столбцы:", non_numeric_cols.tolist())

non_numeric_cols_test = X_test.select_dtypes(exclude=['number','bool']).columns
if len(non_numeric_cols_test) == 0:
    print("Все столбцы X_test числовые или булевы.")
else:
    print("В X_test есть нечисловые столбцы:", non_numeric_cols_test.tolist())

date_info = train_merged.groupby("warehouse")["date"].agg(
    min_date="min",
    max_date="max",
    unique_dates="nunique"
).reset_index()

print("\n=== Временной диапазон по каждому складу ===")
print(date_info)



import pandas as pd

def expanding_gap_splits(df,
                         start_date='2023-12-01',
                         end_date='2024-05-01',
                         freq='M',
                         gap_days=7,
                         horizon_months=1):
    """
    Генерирует (train_df, val_df) в стиле Expanding Window + Gap.

    Параметры:
    ----------
    df : pd.DataFrame
        Исходный датафрейм, должен содержать столбец 'date' формата datetime.
    start_date : str или datetime
        С какой даты начинаем генерировать валидационные сплиты.
        (В нашем случае выбрали '2023-12-01', можете менять.)
    end_date : str или datetime
        Конечная дата для генерации валидационных сплитов.
        (В нашем случае '2024-05-01', но max_date у вас 2024-06-02, 
         так что цикл не выйдет за реальный диапазон.)
    freq : str
        Шаг итерации во времени ('M' - месяц).
    gap_days : int
        Количество дней "зазора" (gap) между train и val.
    horizon_months : int
        На сколько месяцев вперёд идёт валидация на каждом шаге (по умолчанию 1).

    Возвращает (через yield) кортежи (train_df, val_df).
    """

    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])

    min_date = data['date'].min()
    max_date = data['date'].max()

    print(f"Даты в исходном df: {min_date.date()} -> {max_date.date()}")

    for current_date in pd.date_range(start_date, end_date, freq=freq):
        if current_date > max_date:
            break  

        train_end = current_date - pd.Timedelta(days=gap_days)

        train_mask = (data['date'] >= min_date) & (data['date'] < train_end)
        train_df = data.loc[train_mask]

        val_start = current_date
        val_end = current_date + pd.offsets.MonthEnd(horizon_months)
        if val_end > max_date:
            val_end = max_date  

        val_mask = (data['date'] >= val_start) & (data['date'] <= val_end)
        val_df = data.loc[val_mask]

        if len(val_df) == 0:
            continue

        yield train_df, val_df


train_merged['date'] = pd.to_datetime(train_merged['date'])

actual_min = train_merged['date'].min()
actual_max = train_merged['date'].max()
print(f"=== Реальный диапазон дат в train_merged ===")
print(f"Min date: {actual_min.date()}, Max date: {actual_max.date()}")

print("\n=== Генерируем Expanding Window + Gap (7 дней) ===")
gap_days = 7
horizon_months = 1

for i, (train_part, val_part) in enumerate(
    expanding_gap_splits(train_merged,
                         start_date='2023-12-01',  
                         end_date='2024-05-01',   
                         freq='ME',               
                         gap_days=gap_days,
                         horizon_months=horizon_months),
    start=1
):
    print(f"\n===== SPLIT {i} =====")
    print(f"Train period: {train_part['date'].min().date()} -> {train_part['date'].max().date()}"
          f"   (shape={train_part.shape})")
    print(f"Val   period: {val_part['date'].min().date()}   -> {val_part['date'].max().date()}"
          f"   (shape={val_part.shape})")



# import pandas as pd
# import numpy as np
# import lightgbm as lgb

# def mae(y_true, y_pred):
#     return np.mean(np.abs(y_true - y_pred))

# def train_eval_once(train_part, val_part):
#     X_train = train_part.drop(['date','sales'], axis=1, errors='ignore')
#     y_train = train_part['sales']

#     X_val = val_part.drop(['date','sales'], axis=1, errors='ignore')
#     y_val = val_part['sales']

#     dtrain = lgb.Dataset(X_train, label=y_train)
#     dval   = lgb.Dataset(X_val,   label=y_val)

#     params = {
#         'objective': 'regression_l1', 
#         'learning_rate': 0.05,
#         'num_leaves': 31,
#         'metric': 'mae',
#         'verbose': 1
#     }

#     model = lgb.train(
#         params=params,
#         train_set=dtrain,
#         num_boost_round=5000,
#         valid_sets=[dval],
#         valid_names=['val'],
#         callbacks=[
#             lgb.early_stopping(stopping_rounds=50),
#             lgb.log_evaluation(1)
#         ]
#     )

#     y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
#     score = mae(y_val, y_pred_val)
#     return model, score


# all_splits = list(expanding_gap_splits(
#     train_merged,
#     start_date='2023-12-01',
#     end_date='2024-05-01',
#     freq='ME',
#     gap_days=7,
#     horizon_months=1
# ))

# print(f"Получено {len(all_splits)} сплитов")

# if len(all_splits) < 5:
#     raise ValueError("Недостаточно сплитов (меньше 5). Увеличьте период или проверьте данные.")

# train_part_5, val_part_5 = all_splits[4]

# print("\n=== ОБУЧЕНИЕ НА ПЯТОМ СПЛИТЕ ===")
# model_5, score_val_5 = train_eval_once(train_part_5, val_part_5)
# print(f"MAE на пятом сплите: {score_val_5:.4f}")

# X_test = test_merged.drop(['date'], axis=1, errors='ignore')
# test_preds = model_5.predict(X_test, num_iteration=model_5.best_iteration)

# sub = test_merged[['unique_id','date']].copy()
# sub['sales_hat'] = test_preds
# sub['date_str'] = pd.to_datetime(sub['date']).dt.strftime('%Y-%m-%d')
# sub['id'] = sub['unique_id'].astype(str) + "_" + sub['date_str']

# submission = sub[['id','sales_hat']].copy()
# submission.to_csv("submission.csv", index=False)

# print("\nПервые строки submission.csv:")
# print(submission.head(5))



import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def train_eval_once_lstm(train_part, val_part):
    X_train = train_part.drop(['date', 'sales'], axis=1, errors='ignore').values
    y_train = train_part['sales'].values

    X_val = val_part.drop(['date', 'sales'], axis=1, errors='ignore').values
    y_val = val_part['sales'].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(64,
             activation='tanh',
             return_sequences=True,
             dropout=0.2,
             recurrent_dropout=0.2),
        LSTM(32,
             activation='tanh',
             dropout=0.2,
             recurrent_dropout=0.2),
        Dense(16),
        LeakyReLU(negative_slope=0.2),
        Dense(1)
    ])

    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-5)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1000,
        batch_size=2048,
        callbacks=[early_stop, reduce_lr, model_checkpoint],
        verbose=1,
    )

    y_pred_val = model.predict(X_val).flatten()
    score = mae(y_val, y_pred_val)
    return model, score, scaler, history

all_splits = list(expanding_gap_splits(
    train_merged,
    start_date='2023-12-01',
    end_date='2024-05-01',
    freq='ME',
    gap_days=7,
    horizon_months=1
))

print(f"Получено {len(all_splits)} сплитов")

if len(all_splits) < 5:
    raise ValueError("Недостаточно сплитов (меньше 5). Увеличьте период или проверьте данные.")

train_part_5, val_part_5 = all_splits[4]

print("\n=== ОБУЧЕНИЕ НА 5‑М СПЛИТЕ ===")
model, score, scaler, history = train_eval_once_lstm(train_part_5, val_part_5)
print(f"MAE на 5‑м сплите: {score:.4f}")

X_test = test_merged.drop(['date'], axis=1, errors='ignore').values
X_test = scaler.transform(X_test)
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

test_preds = model.predict(X_test).flatten()

sub = test_merged[['unique_id', 'date']].copy()
sub['sales_hat'] = test_preds
sub['date_str'] = pd.to_datetime(sub['date']).dt.strftime('%Y-%m-%d')
sub['id'] = sub['unique_id'].astype(str) + "_" + sub['date_str']

submission = sub[['id', 'sales_hat']].copy()
submission.to_csv("submission.csv", index=False)

print("\nПервые строки submission.csv:")
print(submission.head(5))



train_merged.fillna(train_merged.mean(), inplace=True)
test_merged.fillna(test_merged.mean(), inplace=True)


