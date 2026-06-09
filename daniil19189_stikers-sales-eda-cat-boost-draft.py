import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool

%matplotlib inline


data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')

# Display the first few rows of the dataset to understand its structure
data.head()


# Basic overview of the dataset
data_info = data.info()
data_description = data.describe()

# Checking for missing values
missing_values = data.isnull().sum()

# Distribution of target variable (num_sold)
plt.figure(figsize=(8, 5))
plt.hist(data['num_sold'].dropna(), bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of Number of Sold Copies')
plt.xlabel('Number of Copies Sold')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

data_info, data_description, missing_values


df = data.dropna(subset=['num_sold'])

df.info()


print("Уникальные страны:", df['country'].nunique())
print("Уникальные магазины:", df['store'].nunique())
print("Уникальные продукты:", df['product'].nunique())

print("Список стран:", df['country'].unique())
print("Список магазинов:", df['store'].unique())
print("Список продуктов:", df['product'].unique())



sns.histplot(data=df, x='num_sold', kde=True)
plt.title("Распределение количества проданных единиц (num_sold)")
plt.show()



sns.histplot(data=df, x=np.log1p(df['num_sold']), kde=True)
plt.title("Распределение лог(num_sold + 1)")
plt.show()



plt.figure(figsize=(6, 4))
sns.boxplot(data=df, y='num_sold')
plt.title("Boxplot для num_sold")
plt.show()



# Выберем только числовые столбцы
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='viridis')
plt.title("Корреляционная матрица числовых признаков")
plt.show()



# Пример для 'country'
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='country', y='num_sold')
plt.title("num_sold в разных странах")
plt.xticks(rotation=45)
plt.show()



# Пример для 'store'
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='store', y='num_sold')
plt.title("num_sold в разных магазинах")
plt.xticks(rotation=45)
plt.show()



df.groupby('store')['num_sold'].agg(['count','mean','median','std']).sort_values('mean', ascending=False)


# Агрегация по дате (сумма проданных единиц за день)
daily_sales = df.groupby('date')['num_sold'].sum().reset_index()

plt.figure(figsize=(12, 5))
plt.plot(daily_sales['date'], daily_sales['num_sold'])
plt.title("Динамика суммарных продаж по дням")
plt.xlabel("Дата")
plt.ylabel("Количество проданных единиц")
plt.show()



sales_by_store = df.groupby(['date', 'store'])['num_sold'].sum().reset_index()

plt.figure(figsize=(12, 6))
for store_name in sales_by_store['store'].unique():
    store_data = sales_by_store[sales_by_store['store'] == store_name]
    plt.plot(store_data['date'], store_data['num_sold'], label=store_name)

plt.title("Динамика продаж по дням для каждого магазина")
plt.xlabel("Дата")
plt.ylabel("Суммарные продажи")
plt.legend()
plt.show()



daily_sales['rolling_mean_7d'] = daily_sales['num_sold'].rolling(window=7).mean()
daily_sales['rolling_std_7d'] = daily_sales['num_sold'].rolling(window=7).std()

plt.figure(figsize=(12, 5))
plt.plot(daily_sales['date'], daily_sales['num_sold'], label='Daily Sales')
plt.plot(daily_sales['date'], daily_sales['rolling_mean_7d'], label='7-day Rolling Mean')
plt.fill_between(
    daily_sales['date'],
    daily_sales['rolling_mean_7d'] - 2*daily_sales['rolling_std_7d'],
    daily_sales['rolling_mean_7d'] + 2*daily_sales['rolling_std_7d'],
    color='orange', alpha=0.2, label='±2 std range'
)
plt.legend()
plt.title("Дневные продажи с 7-дневным скользящим средним и интервалом ±2σ")
plt.show()



# Средние продажи по (country, store):
pd.pivot_table(
    df,
    values='num_sold',
    index='country',
    columns='store',
    aggfunc='mean'
).fillna(0)



pd.pivot_table(
    df,
    values='num_sold',
    index='store',
    columns='product',
    aggfunc='mean'
).fillna(0)



df = df.sort_values(by='date')

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values(by='date')

# Создаём временные признаки
df['day_of_week'] = df['date'].dt.dayofweek  # 0=понедельник, 6=воскресенье
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year


# создания лагов (на 1 и 7 дней) внутри каждой группы (country, store, product)
df['lag_1'] = df.groupby(['country','store','product'])['num_sold'].shift(1)
df['lag_7'] = df.groupby(['country','store','product'])['num_sold'].shift(7)

# Удаляем строки, где в лагах нет значений
df = df.dropna(subset=['lag_1','lag_7'])

# Теперь закодируем категориальные признаки (пример - One-Hot)
df = pd.get_dummies(df, columns=['country','store','product'], drop_first=True)

# Убираем лишние столбцы, которые не нужны для обучения
# Например, 'id' или 'date', если не хотим использовать её напрямую.
X = df.drop(['id','date','num_sold'], axis=1, errors='ignore')
y = df['num_sold']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False  # Чтобы не перемешивать, хотя тоже не идеально
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape: ", X_test.shape, y_test.shape)


n_estimators = 100  # скажем, хотим 100 деревьев
model_rf = RandomForestRegressor(
    n_estimators=1,
    warm_start=True,
    random_state=42
)

for i in tqdm(range(n_estimators), desc="Training Random Forest"):
    # на каждой итерации увеличиваем количество деревьев на 1
    model_rf.n_estimators = i + 1
    model_rf.fit(X_train, y_train)

# Предсказание и оценка
y_pred_rf = model_rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("RandomForest RMSE:", rmse_rf)


model_cat = CatBoostRegressor(
    iterations=100,        # количество итераций (по сути, деревьев)
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=10            # каждые 10 итераций будем видеть прогресс
)

model_cat.fit(
    X_train, y_train,
    eval_set=(X_test, y_test)
)

y_pred_cat = model_cat.predict(X_test)
rmse_cat = np.sqrt(mean_squared_error(y_test, y_pred_cat))
print("CatBoost RMSE:", rmse_cat)



print("="*40)
print("RMSE Random Forest:", rmse_rf)
print("RMSE CatBoost:     ", rmse_cat)


importances_rf = model_rf.feature_importances_
feature_names = X_train.columns

# Сортируем от самых важных к наименее
indices = np.argsort(importances_rf)[::-1]

print("Random Forest feature importances:")
for idx in indices:
    print(f"{feature_names[idx]}: {importances_rf[idx]:.4f}")



feature_importances_cat = model_cat.get_feature_importance(prettified=True)
print("\nCatBoost feature importances:")
print(feature_importances_cat)


plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.3, label="RandomForest", color='r')
plt.scatter(y_test, y_pred_cat, alpha=0.3, label="CatBoost", color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')  # линия x=y
plt.xlabel("Фактическое num_sold")
plt.ylabel("Предсказанное num_sold")
plt.title("Сравнение истинных и предсказанных значений")
plt.legend()
plt.show()



df = data.dropna(subset=['num_sold'])

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date').reset_index(drop=True)

# 3. Пример: создаём функцию для расширенной генерации фич (лаги, роллинги)
def create_features(data, group_cols=None):
    """
    data: pd.DataFrame, должен содержать как минимум ['date', 'num_sold']
    group_cols: список столбцов, по которым группируем (например, ['country','store','product'])
    """
    df_local = data.copy()
    
    # Если группа не задана — берём всё как единый временной ряд
    if group_cols is None:
        group_cols = []
    
    # Календарные фичи
    df_local['day_of_week'] = df_local['date'].dt.dayofweek  # 0=понедельник ... 6=воскресенье
    df_local['month'] = df_local['date'].dt.month
    df_local['year'] = df_local['date'].dt.year
    
    # Лаги (1, 7, 14, 30)
    for lag in [1, 7, 14, 30]:
        df_local[f'lag_{lag}'] = df_local.groupby(group_cols)['num_sold'].shift(lag)
    
    # Роллинг-метрики (среднее и std за 7 дней)
    # Для демонстрации — 7 дней; можно делать и 14, 30 и т.п.
    df_local['roll_mean_7'] = df_local.groupby(group_cols)['num_sold'].shift(1).rolling(window=7).mean()
    df_local['roll_std_7']  = df_local.groupby(group_cols)['num_sold'].shift(1).rolling(window=7).std()
    
    # Пропуски после shift/rolling убираем
    df_local = df_local.dropna(subset=[f'lag_{lag}' for lag in [1,7,14,30]] + ['roll_mean_7','roll_std_7'])
    
    return df_local


# 5. Генерируем фичи
#   Скажем, группируем по (country, store, product).
df = create_features(df, group_cols=['country','store','product'])

# 6. Лог-трансформация целевой переменной
#   Вместо y = num_sold используем y = log(num_sold + 1)
df['y_log'] = np.log1p(df['num_sold'])

# 7. Кодируем категориальные признаки
#   Для CatBoost можно передать их как cat_features, 
#   но учитывая, что у нас уже есть "групповые" логи, 
#   проще сделать OHE или оставить в сыром виде. Покажем OHE, как пример.

df = pd.get_dummies(df, columns=['country','store','product'], drop_first=True)

# 8. Формируем X, y
y = df['y_log']  # наша целевая теперь log(num_sold+1)
X = df.drop(['date','num_sold','y_log'], axis=1, errors='ignore')

# 9. TimeSeriesSplit
#   Допустим, хотим 3 фолда. 
#   В каждом сплите мы будем обучаться на раннем периоде, проверять на более позднем.
tscv = TimeSeriesSplit(n_splits=3)


# 10. Подготовим модель CatBoostRegressor
model_cat = CatBoostRegressor(
    loss_function='RMSE',  # под лог-таргетом получим RMSE по log-значениям
    random_seed=42,
    verbose=100 
)

# 11. Параметры для RandomizedSearchCV (примерный набор)
param_distributions = {
    'iterations': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5, 7]
}

# 12. Запускаем RandomizedSearchCV c TimeSeriesSplit
search = RandomizedSearchCV(
    estimator=model_cat,
    param_distributions=param_distributions,
    n_iter=5,         # сколько случайных наборов гиперпараметров попробовать
    scoring='neg_root_mean_squared_error',  # т.к. хотим минимизировать RMSE
    cv=tscv,          # TimeSeriesSplit
    verbose=1,        # чтобы видеть прогресс в консоли
    random_state=42
)

search.fit(X, y)

print("\nЛучшие гиперпараметры CatBoost (по CV):")
print(search.best_params_)
print("Лучший CV-скор (отрицательный RMSE):", search.best_score_)


# 13. Переобучим лучшую модель на всех данных (опционально, можно сразу взять best_estimator_)
best_model = search.best_estimator_
best_model.fit(X, y, verbose=False)

# 14. Оценка качества на последних «будущих» точках
#    Для демонстрации возьмём несколько последних недель/месяцев как тест.
#    (В реальном проекте лучше жёстко фиксировать тренировочный период и тестовый.)
test_size = 200  # последние 200 наблюдений как тест
X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

best_model.fit(X_train, y_train, verbose=False)
y_pred_log = best_model.predict(X_test)

# Обратно трансформируем предсказание: из log(num_sold+1) -> num_sold
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

rmse_test = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"\nRMSE на отложенных последних {test_size} точках:", rmse_test)

# 15. Короткий анализ важности признаков
feature_importances = best_model.get_feature_importance(prettified=True)
print("\nCatBoost feature importances (top-10):")
display(feature_importances.head(10))


# 16. Сравним фактические и предсказанные значения для последних точек
plt.figure(figsize=(10, 5))
plt.plot(y_true.values, label='True', marker='o')
plt.plot(y_pred, label='Predicted', marker='x')
plt.title("Сравнение фактических и предсказанных значений (на тестовой выборке)")
plt.xlabel("Индекс (временной)")
plt.ylabel("num_sold")
plt.legend()
plt.show()


test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")

# 3) Повторяем все шаги фич-инжиниринга
test_df['date'] = pd.to_datetime(test_df['date'], errors='coerce')
test_df['day_of_week'] = test_df['date'].dt.dayofweek
test_df['month'] = test_df['date'].dt.month
test_df['year'] = test_df['date'].dt.year

# One-Hot Encoding (с теми же аргументами)
test_df = pd.get_dummies(
    test_df, 
    columns=['country','store','product'],
    drop_first=True
)

# 4) Приводим колонки к тому же набору, который был при обучении (trained_columns)
# Если в тесте нет некоторых dummy-колонок из train — создадим их со значением 0
trained_columns = X.columns
X_test = test_df.drop(['id','date'], axis=1, errors='ignore')
X_test = X_test.reindex(columns=trained_columns, fill_value=0)

# 5) Делаем предсказание
y_pred = best_model.predict(X_test)

# (Если вы делали log-трансформацию при обучении, здесь нужно сделать np.expm1)

# 6) Формируем итоговый результат [id, num_sold]
submission = pd.DataFrame({
    'id': test_df['id'],       # Берём id из исходного теста
    'num_sold': y_pred         # Наше предсказание
})

print(submission.head(10))


submission.to_csv("submission.csv", index=False)

