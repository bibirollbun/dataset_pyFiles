# Импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Улучшим стиль графиков
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Загружаем данные
# Путь к файлу на Kaggle
file_path = '/kaggle/input/physical-activity-clustering/Physical_Activity_Monitoring_unlabeled.csv'
df = pd.read_csv(file_path)

# 1. Первоначальный осмотр данных
print("Размер датасета:", df.shape)
print("\nПервые 5 строк:")
print(df.head())
print("\nИнформация о типах данных:")
df.info()


# Создаем осмысленные имена столбцов
columns = ['timestamp', 'heartrate']
sensor_locs = ['hand', 'chest', 'ankle']
imu_features = ['temp', 
                'acc1_x', 'acc1_y', 'acc1_z', 
                'acc2_x', 'acc2_y', 'acc2_z', 
                'gyro_x', 'gyro_y', 'gyro_z',
                'magn_x', 'magn_y', 'magn_z',
                'orient_1', 'orient_2', 'orient_3', 'orient_4'] # Добавляем 4 столбца ориентации

for loc in sensor_locs:
    for feature in imu_features:
        columns.append(f'{loc}_{feature}')

# Присваиваем новые имена
df.columns = columns

print("Новые названия столбцов. Первые 5 строк:")
print(df.head())


# Проверяем количество пропусков в каждом столбце
missing_values = df.isnull().sum()
print("\nКоличество пропущенных значений до обработки:")
print(missing_values[missing_values > 0])


# Заполняем пропуски методом forward fill
# Этот метод хорошо подходит для временных рядов
df.fillna(method='ffill', inplace=True)

# После ffill могут остаться пропуски в самом начале файла. Заполним их bfill.
df.fillna(method='bfill', inplace=True)

print("\nКоличество пропущенных значений после обработки:")
print(df.isnull().sum().sum()) # Должен быть 0


plt.figure(figsize=(12, 6))
sns.histplot(df['heartrate'], bins=50, kde=True)
plt.title('Распределение частоты сердечных сокращений (Heart Rate)')
plt.xlabel('Пульс (уд/мин)')
plt.ylabel('Частота')
plt.show()


# Возьмем данные с акселерометра на руке
plt.figure(figsize=(15, 5))
sns.lineplot(x=df.index[:5000], y=df['hand_acc1_x'][:5000]) # Посмотрим на первые 5000 точек
plt.title('Показания акселерометра по оси X (рука)')
plt.xlabel('Индекс (время)')
plt.ylabel('Ускорение (м/с^2)')
plt.show()


pip install pycaret


import pandas as pd


import numpy as np
import scipy
try:
    scipy.interp = np.interp
except AttributeError:
    pass

from pycaret.clustering import *


# ==============================================================================
# ШАГ 1: ПОДГОТОВКА ДАННЫХ (НАШ ЛУЧШИЙ ПОДХОД)
# ==============================================================================
print("1. Загрузка данных и выбор 3 ключевых признаков...")
df = pd.read_csv('/kaggle/input/physical-activity-clustering/Physical_Activity_Monitoring_unlabeled.csv')
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# Оставляем только "золотые" признаки
features_df = df[['chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3']]

# ==============================================================================
# ШАГ 2: ЗАПУСК AutoML (PYCARET) - ИСПРАВЛЕННАЯ ВЕРСИЯ
# ==============================================================================
print("\n2. Настройка окружения PyCaret...")
clu_setup = setup(data=features_df, 
                  normalize=True, 
                  session_id=42, 
                  verbose=False,
                  profile=False)

print("\n3. Создаем и оцениваем несколько моделей...")

# Мы будем смотреть на метрику 'Silhouette'. Чем ближе она к 1, тем лучше.
# num_clusters=5 - наш самый обоснованный выбор k.

print("\n--- Модель: K-Means ---")
kmeans1 = create_model('kmeans', num_clusters=5)
print(kmeans1)
kmeans2 = create_model('kmeans', num_clusters=6)
print(kmeans2)
# print("\n--- Модель: Hierarchical Clustering ---")
# # agglomerative clustering
# hclust = create_model('hclust', num_clusters=5)
# print(hclust)

print("\n--- Модель: Birch ---")
birch = create_model('birch', num_clusters=5)
print(birch)

print("\n--- Модель: Birch ---")
birch = create_model('birch', num_clusters=6)
print(birch)
print("\n--- Модель: DBSCAN ---")

# DBSCAN сам находит количество кластеров, поэтому num_clusters не указываем
# Он может быть очень медленным на таких данных
dbscan = create_model('dbscan') 
print(dbscan)

# ==============================================================================
# ШАГ 3: АНАЛИЗ РЕЗУЛЬТАТОВ И СОЗДАНИЕ SUBMISSION
# ==============================================================================
print("\nПроанализируйте таблицы выше. Выберите модель с лучшим Silhouette Score.")
print("Поскольку мы уже знаем, что KMeans - наш фаворит, будем использовать его для submission.")

# Получаем предсказания от нашей модели K-Means
predictions = assign_model(kmeans)
print("\nПример предсказаний от K-Means:")
print(predictions.head())

# Формируем итоговый файл
print("\n5. Формирование submission файла...")
submission = pd.DataFrame({'Index': df.index})
submission['raw_activityID'] = predictions['Cluster'].str.replace('Cluster ', '').astype(int)

# Стандартная процедура преобразования меток
unique_clusters = sorted(np.unique(submission['raw_activityID']))
mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(unique_clusters)}
submission['activityID'] = submission['raw_activityID'].map(mapping)
final_submission = submission[['Index', 'activityID']]

final_submission.to_csv('submission_pycaret_kmeans_k5.csv', index=False)
print("\nФайл 'submission_pycaret_kmeans_k5.csv' создан.")

