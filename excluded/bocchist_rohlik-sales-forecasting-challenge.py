import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv')
inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
sales_test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv')
sales_train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
test_weights = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')


sales_train['date'] = pd.to_datetime(sales_train['date'])
sales_test['date'] = pd.to_datetime(sales_test['date'])
calendar['date'] = pd.to_datetime(calendar['date'])

sales_train = sales_train.drop('availability', axis = 1)


group_data = sales_train.groupby('date')[['sales']].sum().reset_index()
px.line(group_data, x='date', y='sales', title='Date Total Sales')


monthly_data = group_data.set_index('date').resample(rule='ME').sum().reset_index()

px.line(monthly_data, x='date', y='sales', title = 'Monthly Sales')


weekly_data = group_data.set_index('date').resample(rule='W').sum().reset_index()

px.line(weekly_data, x='date', y='sales', title = 'Weekly Sales')


group_data['weekday'] = group_data['date'].dt.dayofweek

# 0 - Monday
px.bar(group_data, x='weekday', y='sales', title = 'Weekday Sales')


group_data['day'] = group_data['date'].dt.day

px.bar(group_data, x='day', y='sales', title = 'Day of month Sales')


fig = px.histogram(sales_train, x='sales', nbins=50, title='Sales histogram')
fig.update_traces(opacity=0.75)
fig.show()


def generate_time_features(df, calendar, date_column, add_trend_seasonality=False):
    """
    Генерация временных признаков из временной метки.

    Параметры:
    - df: DataFrame с данными.
    - date_column: название столбца с временной меткой (должен быть типа datetime).
    - add_trend_seasonality: bool, добавить ли признаки тренда и сезонности.

    Возвращает:
    - DataFrame с добавленными признаками.
    """
    df = df.copy()

    df = df.merge(calendar, on = ['date', 'warehouse'], how = 'left')
    
    # Убедимся, что столбец с датами имеет тип datetime
    df[date_column] = pd.to_datetime(df[date_column])

    df = df.sort_values(by=['date', 'warehouse'])

    # Создаем признак макс скидки
    discount_cols = df.filter(regex=r'^type_\d+_discount$').columns
    df['max_discount'] = df[discount_cols].max(axis=1)
    
    # Основные временные признаки
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['day_of_week'] = df[date_column].dt.dayofweek  # Понедельник = 0, Воскресенье = 6
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['quarter'] = df[date_column].dt.quarter
    df['day_of_year'] = df[date_column].dt.dayofyear  # День года
    df['week_of_year'] = df[date_column].dt.isocalendar().week  # Номер недели

    # Периодические признаки (синус/косинус для циклической природы времени)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Признаки тренда и сезонности
    if add_trend_seasonality:
        df['linear_trend'] = (df[date_column] - df[date_column].min()).dt.days  # Линейный тренд
        df['seasonality_month'] = np.sin(2 * np.pi * df['month'] / 12) * np.cos(2 * np.pi * df['month'] / 12)
        df['seasonality_week'] = np.sin(2 * np.pi * df['week_of_year'] / 52)

	# Добавление полиномиального тренда
    df["squared_trend"] = np.square(np.arange(len(df)))
    df["log_trend"] = np.log1p(np.arange(len(df)))  # Логарифмический
    df["exp_trend"] = np.exp(np.arange(len(df)) / len(df))  # Экспоненциальный

    df['id'] = df['unique_id'].astype(str) + "_" + df['date'].astype(str)
    
    df = df.drop(['holiday_name'] + list(discount_cols), axis=1) 
    df = df.dropna()

    return df


def generate_lag_features(df):
    
    # Лаги
    lag_days = [1, 7, 14]
    for lag in lag_days:
        df[f'lag_{lag}'] = df.groupby('unique_id')['sales'].shift(lag)

    # Добавляем скользящее среднее
    window_sizes = [7, 14]
    for window in window_sizes:
        df[f'rolling_mean_{window}'] = df.groupby('unique_id')['sales'].shift(1).rolling(window=window).mean()


    return df


train_data = generate_time_features(sales_train, calendar, 'date', True)
train_data = generate_lag_features(train_data)


num_feat = ['total_orders', 'sell_price_main', 'squared_trend', 'lag_1', 'lag_7', 'lag_14', 'rolling_mean_7', 'rolling_mean_14']
cat_feat = ['warehouse']

X = train_data.drop(['id', 'sales', 'date'], axis = 1)
y = np.log1p(train_data['sales'])


def wmae_score(y_true, y_pred, weights):
    """
    Вычисляет взвешенную среднюю абсолютную ошибку (WMAE).
    
    y_true  - истинные значения
    y_pred  - предсказанные значения
    weights - веса (например, значимость магазинов)
    """
    return np.sum(np.abs(y_true - y_pred) * weights) / np.sum(weights)



def custom_wmae_score(estimator, X, y):
    """
    Пользовательская метрика для cross_val_score.
    
    estimator - обученная модель
    X - входные признаки (включает store_id, который нужно удалить)
    y - целевые значения
    """
    # Извлекаем веса
    store_ids = X["unique_id"]  # Достаём ID магазинов
    weights = store_ids.map(test_weights.set_index("unique_id")["weight"]).values  # Маппим веса

    # Убираем store_id перед подачей в модель
    X_model = X.drop(columns=["unique_id"])

    # Делаем предсказания
    y_pred = estimator.predict(X_model)

    # Считаем WMAE
    return -wmae_score(y, y_pred, weights)  # Берём отрицательное значение, так как Optuna максимизирует



"""
def objective(trial):
    
    
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "random_state": 42,
        "objective" : "reg:squarederror"
    }


    # Создаем препроцессор для числовых и категориальных признаков
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_feat),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feat)
        ]
    )

    # Инициализируем модель с подобранными гиперпараметрами
    model = XGBRegressor(**params)
    
    # Собираем Pipeline: сначала препроцессинг, затем регрессор
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    

    # Определяем TimeSeriesSplit для кросс-валидации (например, 3 разбиений)
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Используем отрицательную среднюю абсолютную ошибку как метрику (чем больше — тем лучше)
    scores = cross_val_score(pipeline, X, y, cv=tscv, scoring=custom_wmae_score)
    
    # Возвращаем среднее значение метрики по кросс-валидации
    return np.mean(scores)


# Создаем исследование (study) с направлением максимизации (так как метрика отрицательная, и её максимум соответствует минимальной MAE)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)


# Вывод лучших гиперпараметров
print("Лучший trial:")
best_trial = study.best_trial
print("  Значение метрики:", best_trial.value)
print("  Гиперпараметры:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")
"""


# Значение метрики Optuna: -0.33016935935972924
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

best_params = {
    "objective": "reg:squarederror",
    "n_estimators": 101,
    "max_depth": 7,
    "learning_rate": 0.05335853605410118,
    "subsample": 0.876364265110239,
    "colsample_bytree": 0.8253151666142489,
    "reg_alpha": 0.2930905580203602,
    "reg_lambda": 0.18076715508639507
}

# Создаем препроцессор для числовых и категориальных признаков
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_feat),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feat)
    ]
)

# Инициализируем модель с подобранными гиперпараметрами
model = XGBRegressor(**best_params)

# Собираем Pipeline: сначала препроцессинг, затем регрессор
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Обучаем модель
pipeline.fit(X_train, y_train)

# Предсказываем
y_pred = pipeline.predict(X_test)

# Считаем метрики
MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)

print(f'MSE : {MSE}')
print(f'MAE : {MAE}')


import numpy as np
import pandas as pd

# Максимальный лаг, который используется (1, 7, 14)
max_lag = 14

# Список для хранения предсказаний
predictions_list = []

# Получаем уникальные идентификаторы из тестовых данных
unique_ids = sales_test['unique_id'].unique()

# Для каждого unique_id выполняем итеративное прогнозирование
for uid in unique_ids:
    # Извлекаем исторические (тренировочные) данные для данного unique_id и сортируем по дате
    hist = sales_train[sales_train['unique_id'] == uid].copy()
    hist.sort_values('date', inplace=True)
    
    # Берем последние max_lag наблюдений для формирования лагов
    hist_last = hist.tail(max_lag).copy()
    
    # Извлекаем тестовые строки для текущего unique_id и сортируем по дате
    test_rows = sales_test[sales_test['unique_id'] == uid].copy()
    test_rows.sort_values('date', inplace=True)
    
    # Итеративно прогнозируем для каждой тестовой даты
    for _, row in test_rows.iterrows():
        # Объединяем историю с текущей тестовой строкой в один DataFrame
        temp_df = pd.concat([hist_last, pd.DataFrame([row])], ignore_index=True)
        
        # Для текущей тестовой строки отсутствует значение sales, 
        # поэтому заполним его фиктивным значением (например, 0) чтобы избежать удаления строки при dropna()
        temp_df.loc[temp_df.index[-1], 'sales'] = 0
        
        # Генерируем временные признаки и лаги
        temp_df = generate_time_features(temp_df, calendar, 'date', add_trend_seasonality=True)
        temp_df = generate_lag_features(temp_df)
        
        # Извлекаем строку, для которой делаем прогноз.
        # Удаляем лишние столбцы, такие как 'id', 'sales' и 'date'
        X_current = temp_df.iloc[[-1]].drop(['id', 'sales', 'date'], axis=1)
        
        # Делаем прогноз (наша модель предсказывает log1p(sales), поэтому возвращаемся в исходное пространство)
        y_pred_log = pipeline.predict(X_current)[0]
        y_pred = np.expm1(y_pred_log)
        
        # Получаем id, который был сформирован в generate_time_features (например, в виде unique_id_date)
        test_id = temp_df.iloc[-1]['id']
        predictions_list.append({'id': test_id, 'sales_hat': y_pred})
        
        # Обновляем историю: добавляем строку с текущей датой, где sales = y_pred
        new_row = row.copy()
        new_row['sales'] = y_pred
        hist_last = pd.concat([hist_last, pd.DataFrame([new_row])], ignore_index=True)
        # Если в истории стало больше наблюдений, чем требуется для формирования лагов, удаляем самые старые
        if len(hist_last) > max_lag:
            hist_last = hist_last.iloc[1:]
            
# Формируем итоговый DataFrame для сабмита
submission = pd.DataFrame(predictions_list)
submission.to_csv('submission.csv', index=False)



submission

