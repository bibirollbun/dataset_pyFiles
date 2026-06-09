# Импорты
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_absolute_error

from datetime import datetime

# Убрать предупреждения
import warnings
warnings.filterwarnings('ignore') 


# Загрузка тренировочного датасета
df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")


# Вывести несколько примеров
df.head()


# Размеры датасета
df.shape


# Сводная информация по датасету
df.info()


# Информация про значения столбцов в датасете
df.describe()


# Проверка на пустоту
df.isnull().sum()


# Проверка на дубликаты
print("Дубликатов:", df.duplicated().sum())


# Списки числовых и категориальных признаков
cat_rows = ["season", "holiday", "workingday", "weather"]
num_rows = ["temp",	"atemp", "humidity", "windspeed"]


# Распределение данных в категориальных признаках
for col in cat_rows:

    plt.figure(figsize=(10, 5))
    
    # Столбчатая диаграмма
    plt.subplot(1, 2, 1)
    sbn.barplot(x=df[col].value_counts().index, y=df[col].value_counts().values)
    plt.xlabel(col)
    plt.ylabel("Count")

    # Круговая диаграмма
    plt.subplot(1, 2, 2)
    plt.pie(df[col].value_counts(), labels=df[col].value_counts().index.values)
    plt.legend()
    plt.title(col)

    plt.show()


# Распределение в числовых признаках
for col in num_rows:

    plt.figure(figsize=(10, 5))
    
    # Гистограмма
    plt.subplot(1, 2, 1)
    sbn.histplot(df[col], bins=15)

    # Ящик с усами
    plt.subplot(1, 2, 2)
    sbn.boxplot(df[col])
    plt.show()


# Распределение данных в целевой перменной
plt.figure(figsize=(10, 5))

# Гистограмма
plt.subplot(1, 2, 1)
sbn.histplot(df["count"].value_counts(), bins=15)

# Ящик с усами
plt.subplot(1, 2, 2)
sbn.boxplot(df["count"])

plt.show()


# Корреляция признаков
sbn.heatmap(df.drop("datetime", axis=1).corr(), annot=True, fmt=".2f")


# Зависимость count от категориальных признаков
i = 0 # ограничение, чтобы не рисовался count в конце
for col in cat_rows:
    sbn.barplot(data=df, x=col, y='count')
    plt.show()

    i+=1

    if i == 4:
        break


# Взаимосвзь Count и числовых признаков
for col in num_rows:
    sbn.scatterplot(df, x=col, y="count")
    plt.show()


# Убираем аномалии
df = df[df["windspeed"] <= 40]
df = df[df['humidity'] >= 25]
df = df[df['humidity'] <= 85]
df = df[df["weather"] < 4]
df = df[df["atemp"] > 5]
df = df[df["atemp"] < 40]
df = df[df["temp"] > 5]
df = df[df["temp"] < 38]
df = df[df["count"] <= 630]
# df = df[df["holiday"] == 0]

df.shape


# Удаляем ненужные столбцы
df = df.drop(["casual", "registered", "holiday", "temp"], axis=1)

df


# One-Hot Encoding
df = pd.get_dummies(df, columns=["season", "weather"], drop_first=True)

df


# Обработка времени
df["hour"] = pd.to_datetime(df['datetime']).dt.hour
df["day"] = pd.to_datetime(df['datetime']).dt.day
df["month"] = pd.to_datetime(df['datetime']).dt.month
df["year"] = pd.to_datetime(df['datetime']).dt.year
df["weekday"] = pd.to_datetime(df['datetime']).dt.weekday
df = df.drop(['datetime'], axis=1)
df


# Подготовка X и y
X = df.drop("count", axis=1).values
y = df["count"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# Метрика RMSLE
def rmsle(y_true, y_pred):
    # Добавляем 1, чтобы избежать log(0)
    y_true = np.log1p(y_true)  # log1p(x) = log(x + 1)
    y_pred = np.log1p(y_pred)
    squared_log_errors = (y_true - y_pred) ** 2
    return np.sqrt(np.mean(squared_log_errors))


# Инициализация
lr = LinearRegression()


# Обучение
lr.fit(X_train, np.log1p(y_train))


# Метрики
y_pred_train = lr.predict(X_train) # Предсказания модели на тренировочных данных
y_pred_test = lr.predict(X_test) # Предсказания модели на тестовых данных

# На случай отрицательных ответов
y_pred_train = np.where(y_pred_train<0, 0, y_pred_train)
y_pred_test = np.where(y_pred_test<0, 0, y_pred_test)

# RMSLE метрика
print(f"RMSLE на тренировочных данных: {rmsle(y_train, np.exp(y_pred_train))}")
print(f"RMSLE на тестовых данных: {rmsle(y_test, np.exp(y_pred_test))}")


# Инциализация всего нужного

# Сам лес
rf = RandomForestRegressor(random_state=42)

# Параметры для подбора
param_grid = {
    'n_estimators': [50, 100, 300, 500, 1000],
    'max_depth': [3, 5, 7, 10, 15]
}

# Класс для подбора параметров
rf_gs = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring="r2", n_jobs=-1)


# Обучение
rf_gs.fit(X_train, np.log1p(y_train))


# Лучшие результаты
print(f"Лучшие параметры: {rf_gs.best_params_}")
print(f"Лучшие метрики (R2): {rf_gs.best_score_}")


# Метрики

y_pred_train = rf_gs.best_estimator_.predict(X_train) # Предсказания лучшей модели на тренировочных данных
y_pred_test = rf_gs.best_estimator_.predict(X_test) # Предсказания лучшей модели на тренировочных данных

# На случай отрицательных ответов
y_pred_train = np.where(y_pred_train<0, 0, y_pred_train)
y_pred_test = np.where(y_pred_test<0, 0, y_pred_test)

# RMSLE метрика
print(f"RMSLE на тренировочных данных: {rmsle(y_train, np.exp(y_pred_train) )}")
print(f"RMSLE на тестовых данных: {rmsle(y_test, np.exp(y_pred_test))}")


# Инициализация всего нужного

# Конвейер
en_pipeline = make_pipeline(
    StandardScaler(),
    ElasticNet(random_state=1)
)

# Параметры для подбора
param_grid = {
    'elasticnet__alpha': [0.001, 0.01, 0.1, 1, 10],
    'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

# Класс для подбора параметров
en_gs = GridSearchCV(estimator=en_pipeline, param_grid=param_grid, cv=5, scoring="r2", n_jobs=-1)


# Обучение
en_gs.fit(X_train, np.log1p(y_train))


# Лучшие результаты
print(f"Лучшие параметры: {en_gs.best_params_}")
print(f"Лучшие метрики(R2): {en_gs.best_score_}")


# Метрики

y_pred_train =  en_gs.best_estimator_.predict(X_train) # Предсказания модели на тренировочных данных
y_pred_test =  en_gs.best_estimator_.predict(X_test) # Предсказания модели на тестовых данных

# На случай отрицательных ответов
y_pred_train = np.where(y_pred_train<0, 0, y_pred_train)
y_pred_test = np.where(y_pred_test<0, 0, y_pred_test)

# RMSLE метрика
print(f"RMSLE на тренировочных данных: {rmsle(y_train, np.exp(y_pred_train))}")
print(f"RMSLE на тестовых данных: {rmsle(y_test, np.exp(y_pred_test))}")


# Инициализация всего нужного

# Полиномиальая регрессия
pr_pipeline = make_pipeline(
    PolynomialFeatures(),
    LinearRegression()
)

# Параметры для перебора
param_grid = {
    'polynomialfeatures__degree': [2, 3, 4]  # Степень полинома
}

# Класс для подбора параметров
pr_gs = GridSearchCV(estimator=pr_pipeline, param_grid=param_grid, cv=5, refit=True, scoring="r2", n_jobs=-1)


# Обучение
pr_gs.fit(X_train, np.log1p(y_train))


# Лучшие результаты
print(f"Лучшие параметры: {pr_gs.best_params_}")
print(f"Лучшие метрики(R2): {pr_gs.best_score_}")


# Метрики

y_pred_train = pr_gs.best_estimator_.predict(X_train) # Предсказания модели на тренировочных данных
y_pred_test = pr_gs.best_estimator_.predict(X_test) # Предсказания модели на тестовых данных

# На случай отрицательных ответов
y_pred_train = np.where(y_pred_train<0, 0, y_pred_train)
y_pred_test = np.where(y_pred_test<0, 0, y_pred_test)

# RMSLE метрика
print(f"RMSLE на тренировочных данных: {rmsle(y_train, np.exp(y_pred_train)) }")
print(f"RMSLE на тестовых данных: {rmsle(y_test, np.exp(y_pred_test) )}")


# Инициализация всего нужного

# Конвейер из полиномиальных признаков и ElasticNet
pen_pipeline = make_pipeline(
    PolynomialFeatures(),
    ElasticNet()
)

# Параметры для подбора
param_grid = {
    # Параметры PolynomialFeatures
    'polynomialfeatures__degree': [2, 3, 4],                # Степень полинома (2, 3 или 4)
    
    # Параметры ElasticNet
    'elasticnet__alpha': [0.001, 0.01, 0.1, 1],  # Сила регуляризации
    'elasticnet__l1_ratio': [0.2, 0.5, 0.8],     # Соотношение L1/L2 (0.5 = баланс)
}

# Класс для подбора
pen_gs = GridSearchCV(estimator = pen_pipeline, param_grid = param_grid, cv = 5, refit = True, scoring = "r2", n_jobs = -1)


# Обучение
pen_gs.fit(X_train, np.log1p(y_train))


# Лучшие результаты
print(f"Лучшие параметры: {pen_gs.best_params_}")
print(f"Лучшие метрики(R2): {pen_gs.best_score_}")


# Метрики

y_pred_train = pen_gs.best_estimator_.predict(X_train) # Предсказания модели на тренировочных данных
y_pred_test = pen_gs.best_estimator_.predict(X_test) # Предсказания модели на тестовых данных

# На случай отрицательных ответов
y_pred_train = np.where(y_pred_train<0, 0, y_pred_train)
y_pred_test = np.where(y_pred_test<0, 0, y_pred_test)

# RMSLE метрика
print(f"RMSLE на тренировочных данных: {rmsle(y_train, np.exp(y_pred_train))}")
print(f"RMSLE на тестовых данных: {rmsle(y_test, np.exp(y_pred_test))}")


# Инициализация всего нужного

xgb = XGBRegressor(objective='reg:squarederror') # objective - функция потерь (MSE)

# Параметры для подбора
param_grid = {
    'n_estimators': [100, 200, 300, 500], # кол-во деревьев
    'max_depth': [ 5, 7, 10, 15], # макс. глубина деревьев
    'learning_rate': [ 0.03, 0.05, 0.07, 0.1], # скорость обучения
    'reg_lambda': [50, 100, 300] # Степень L2-регуляризации
}

# Класс для подбора параметров
xgb_gs = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)


# Обучение
xgb_gs.fit(X_train, np.log1p(y_train))


# Лучшие результаты
print(f"Лучшие параметры: {xgb_gs.best_params_}")
print(f"Лучшие метрики (R2): {xgb_gs.best_score_}")


# Метрики

y_pred_train = xgb_gs.best_estimator_.predict(X_train) # Предсказания модели на тренировочных данных
y_pred_test = xgb_gs.best_estimator_.predict(X_test) # Предсказания модели на тестовых данных

# RMSLE метрика
print(f"RMSLE для тренировочных данных: { rmsle(y_train, np.exp(y_pred_train) ) }")
print(f"RMSLE для тестовых данных: { rmsle(y_test, np.exp(y_pred_test) ) }")


# Инициализируем лучшую модель
best_model = XGBRegressor(learning_rate=0.1, max_depth=15, n_estimators=500, reg_lambda=300)
best_model = RandomForestRegressor(random_state=42,n_estimators=140)


# Обучение на всех данных
best_model.fit(X, np.log1p(y))


# RMSLE
rmsle(y, np.exp(best_model.predict(X)))


# Подготовка X из задания

# Достаем данные задания
df_sub = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

# Удаляем ненужные столбцы
df_sub = df_sub.drop(["holiday", "temp"], axis=1)

# One-Hot Encoding
df_sub = pd.get_dummies(df_sub, columns=["season", "weather"], drop_first=True)
df_sub = df_sub.drop("weather_4", axis=1)

# Обработка времени
df_sub["hour"] = pd.to_datetime(df_sub['datetime']).dt.hour
df_sub["day"] = pd.to_datetime(df_sub['datetime']).dt.day
df_sub["month"] = pd.to_datetime(df_sub['datetime']).dt.month
df_sub["year"] = pd.to_datetime(df_sub['datetime']).dt.year
df_sub["weekday"] = pd.to_datetime(df_sub['datetime']).dt.weekday
df_sub = df_sub.drop("datetime", axis=1)

# Создаем X
X_sub = df_sub.values

df_sub


# Предсказания модели
y_sub_pred = best_model.predict(X_sub)

y_sub_pred




