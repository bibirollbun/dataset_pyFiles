import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


train_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
test_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')

train_demo_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
test_demo_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')

train_df.head()


test_df.head()


# Функция для вывода базовой информации о датафрейме
def analyze_dataframe(df, name):
    print(f"Анализ датафрейма {name}:")
    print(f"Форма: {df.shape}")
    print(f"Количество пропущенных значений: {df.isna().sum().sum()}")
    print("\nТипы данных:")
    print(df.dtypes.value_counts())
    print("\nПримеры уникальных значений в некоторых столбцах:")
    
    # Вывод уникальных значений для категориальных столбцов
    for col in ['subject', 'behavior']:
        if col in df.columns:
            print(f"{col}: {df[col].nunique()} уникальных значений")
            print(f"Примеры: {df[col].unique()[:5]}")
    
    # Дополнительно для тренировочного датасета
    if 'gesture' in df.columns:
        print(f"gesture: {df['gesture'].nunique()} уникальных значений")
        print(f"Распределение: \n{df['gesture'].value_counts()}")
    
    if 'sequence_type' in df.columns:
        print(f"sequence_type: {df['sequence_type'].nunique()} уникальных значений")
        print(f"Распределение: \n{df['sequence_type'].value_counts()}")
    
    if 'orientation' in df.columns:
        print(f"orientation: {df['orientation'].nunique()} уникальных значений")
        print(f"Примеры: {df['orientation'].unique()[:5]}")

# Анализ обоих датасетов
analyze_dataframe(train_df, "train_df")
analyze_dataframe(test_df, "test_df")


# Визуализация распределения целевой переменной (только для train_df)
plt.figure(figsize=(14, 6))
sns.countplot(y='gesture', data=train_df, order=train_df['gesture'].value_counts().index)
plt.title('Распределение жестов (целевая переменная)')
plt.tight_layout()
plt.show()


# Распределение по ориентации
if 'orientation' in train_df.columns:
    plt.figure(figsize=(14, 5))
    sns.countplot(y='orientation', data=train_df, order=train_df['orientation'].value_counts().index)
    plt.title('Распределение по ориентации')
    plt.tight_layout()
    plt.show()

# Распределение по поведению
plt.figure(figsize=(14, 5))
sns.countplot(y='behavior', data=train_df, order=train_df['behavior'].value_counts().index)
plt.title('Распределение по поведению')
plt.tight_layout()
plt.show()

# Распределение по типу последовательности
if 'sequence_type' in train_df.columns:
    plt.figure(figsize=(14, 4))
    sns.countplot(x='sequence_type', data=train_df)
    plt.title('Распределение по типу последовательности')
    plt.show()

# Распределение по субъектам
plt.figure(figsize=(14, 4))
sns.countplot(x='subject', data=train_df)
plt.title('Распределение по субъектам')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Анализ числовых данных акселерометра
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
sns.histplot(train_df['acc_x'], kde=True)
plt.title('Распределение acc_x')

plt.subplot(1, 3, 2)
sns.histplot(train_df['acc_y'], kde=True)
plt.title('Распределение acc_y')

plt.subplot(1, 3, 3)
sns.histplot(train_df['acc_z'], kde=True)
plt.title('Распределение acc_z')

plt.tight_layout()
plt.show()

# Анализ данных вращения
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
sns.histplot(train_df['rot_w'], kde=True)
plt.title('Распределение rot_w')

plt.subplot(1, 4, 2)
sns.histplot(train_df['rot_x'], kde=True)
plt.title('Распределение rot_x')

plt.subplot(1, 4, 3)
sns.histplot(train_df['rot_y'], kde=True)
plt.title('Распределение rot_y')

plt.subplot(1, 4, 4)
sns.histplot(train_df['rot_z'], kde=True)
plt.title('Распределение rot_z')

plt.tight_layout()
plt.show()


# Анализ данных термопилей
thm_cols = [col for col in train_df.columns if col.startswith('thm_')]
plt.figure(figsize=(18, 6))

for i, col in enumerate(thm_cols, 1):
    plt.subplot(1, 5, i)
    sns.histplot(train_df[col], kde=True)
    plt.title(f'Распределение {col}')

plt.tight_layout()
plt.show()


# Анализ зависимости acc_x, acc_y, acc_z от жеста
plt.figure(figsize=(20, 15))

plt.subplot(1, 3, 1)
sns.boxplot(x='gesture', y='acc_x', data=train_df)
plt.title('acc_x по жестам')
plt.xticks(rotation=90)

plt.subplot(1, 3, 2)
sns.boxplot(x='gesture', y='acc_y', data=train_df)
plt.title('acc_y по жестам')
plt.xticks(rotation=90)

plt.subplot(1, 3, 3)
sns.boxplot(x='gesture', y='acc_z', data=train_df)
plt.title('acc_z по жестам')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()



# Анализ ToF сенсоров (выборка первых пикселей для каждого сенсора)
tof_sample_cols = ['tof_1_v0', 'tof_2_v0', 'tof_3_v0', 'tof_4_v0', 'tof_5_v0']
plt.figure(figsize=(18, 6))

for i, col in enumerate(tof_sample_cols, 1):
    plt.subplot(1, 5, i)
    # Исключаем -1 (отсутствие отражения)
    valid_data = train_df[train_df[col] != -1][col]
    sns.histplot(valid_data, kde=True)
    plt.title(f'Распределение {col} (без -1)')

plt.tight_layout()
plt.show()


# Визуализация данных ToF для одного примера
def visualize_tof_sensor(df, sequence_id, sensor_num, frame_idx=0):
    """
    Визуализирует данные ToF сенсора для конкретного примера
    
    Args:
        df: DataFrame с данными
        sequence_id: ID последовательности
        sensor_num: Номер сенсора (1-5)
        frame_idx: Индекс кадра в последовательности
    """
    # Выбираем данные для конкретной последовательности и кадра
    seq_data = df[(df['sequence_id'] == sequence_id)]
    if len(seq_data) == 0:
        print(f"Последовательность {sequence_id} не найдена")
        return
    
    frame_data = seq_data.iloc[frame_idx]
    
    # Получаем все пиксели для выбранного сенсора
    tof_cols = [f'tof_{sensor_num}_v{i}' for i in range(64)]
    tof_values = frame_data[tof_cols].values
    
    # Преобразуем в числовой формат и заменяем -1 на NaN
    tof_values = tof_values.astype(float)
    tof_values = np.where(tof_values == -1, np.nan, tof_values)
    
    # Формируем матрицу 8x8
    tof_data = tof_values.reshape(8, 8)
    
    plt.figure(figsize=(8, 6))
    
    # Используем imshow вместо heatmap для лучшей работы с NaN значениями
    im = plt.imshow(tof_data, cmap='viridis')
    plt.colorbar(im, label='Значение')
    
    # Добавляем аннотации (только для непустых значений)
    for i in range(8):
        for j in range(8):
            value = tof_data[i, j]
            if not np.isnan(value):
                plt.text(j, i, f'{int(value)}', ha='center', va='center', 
                         color='white' if value > 100 else 'black')
    
    # Добавляем информацию о жесте, если доступна
    title = f'ToF сенсор {sensor_num}, последовательность {sequence_id}, кадр {frame_idx}'
    if 'gesture' in df.columns:
        gesture = frame_data.get('gesture', 'Н/Д')
        title += f', жест: {gesture}'
    
    plt.title(title)
    plt.show()

# Визуализируем данные ToF для первой последовательности в train_df
first_sequence = train_df['sequence_id'].iloc[0]
for sensor in range(1, 6):
    visualize_tof_sensor(train_df, first_sequence, sensor)


# Анализ временных рядов для одной последовательности
def plot_sequence_timeseries(df, sequence_id):
    """
    Строит временные ряды для одной последовательности
    
    Args:
        df: DataFrame с данными
        sequence_id: ID последовательности
    """
    seq_data = df[df['sequence_id'] == sequence_id].copy()
    if len(seq_data) == 0:
        print(f"Последовательность {sequence_id} не найдена")
        return
    
    seq_data = seq_data.sort_values('sequence_counter')
    
    # Информация о последовательности
    if 'gesture' in seq_data.columns:
        gesture = seq_data['gesture'].iloc[0]
        print(f"Последовательность {sequence_id}, жест: {gesture}")
    
    # График акселерометра
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(seq_data['sequence_counter'], seq_data['acc_x'], label='acc_x')
    plt.plot(seq_data['sequence_counter'], seq_data['acc_y'], label='acc_y')
    plt.plot(seq_data['sequence_counter'], seq_data['acc_z'], label='acc_z')
    plt.title(f'Данные акселерометра для последовательности {sequence_id}')
    plt.xlabel('Счетчик последовательности')
    plt.ylabel('Ускорение (м/с²)')
    plt.legend()
    
    # График вращения
    plt.subplot(2, 1, 2)
    plt.plot(seq_data['sequence_counter'], seq_data['rot_w'], label='rot_w')
    plt.plot(seq_data['sequence_counter'], seq_data['rot_x'], label='rot_x')
    plt.plot(seq_data['sequence_counter'], seq_data['rot_y'], label='rot_y')
    plt.plot(seq_data['sequence_counter'], seq_data['rot_z'], label='rot_z')
    plt.title(f'Данные вращения для последовательности {sequence_id}')
    plt.xlabel('Счетчик последовательности')
    plt.ylabel('Вращение')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # График термопилей
    plt.figure(figsize=(14, 6))
    for col in thm_cols:
        plt.plot(seq_data['sequence_counter'], seq_data[col], label=col)
    plt.title(f'Данные термопилей для последовательности {sequence_id}')
    plt.xlabel('Счетчик последовательности')
    plt.ylabel('Температура (°C)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Визуализируем временные ряды для нескольких последовательностей
if 'gesture' in train_df.columns:
    # Выберем по одной последовательности для каждого жеста
    for gesture in train_df['gesture'].unique():
        seq_id = train_df[train_df['gesture'] == gesture]['sequence_id'].iloc[0]
        plot_sequence_timeseries(train_df, seq_id)

# Анализ зависимости между поведением и жестом
if 'gesture' in train_df.columns and 'behavior' in train_df.columns:
    plt.figure(figsize=(16, 12))
    behavior_gesture = pd.crosstab(train_df['behavior'], train_df['gesture'], normalize='index')
    sns.heatmap(behavior_gesture, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Зависимость между поведением и жестом')
    plt.tight_layout()
    plt.show()


# PCA для визуализации данных ToF
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def visualize_tof_pca(df, n_samples=1000):
    """
    Применяет PCA к данным ToF для визуализации
    
    Args:
        df: DataFrame с данными
        n_samples: Количество образцов для анализа
    """
    if len(df) > n_samples:
        df_sample = df.sample(n_samples, random_state=42)
    else:
        df_sample = df
    
    # Собираем все данные ToF
    tof_cols = [col for col in df.columns if col.startswith('tof_')]
    
    # Заменяем -1 на 0 для PCA
    tof_data = df_sample[tof_cols].replace(-1, 0)
    
    # Проверяем наличие NaN и заменяем их на 0
    if tof_data.isna().any().any():
        print("Обнаружены пропущенные значения в данных ToF. Заменяем их на 0.")
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        tof_data = pd.DataFrame(
            imputer.fit_transform(tof_data),
            columns=tof_data.columns
        )
    
    # Нормализация данных
    scaler = StandardScaler()
    tof_scaled = scaler.fit_transform(tof_data)
    
    # Применяем PCA
    pca = PCA(n_components=2)
    tof_pca = pca.fit_transform(tof_scaled)
    
    # Создаем DataFrame для визуализации
    pca_df = pd.DataFrame({
        'PC1': tof_pca[:, 0],
        'PC2': tof_pca[:, 1]
    })
    
    # Добавляем информацию о жесте, если доступна
    if 'gesture' in df.columns:
        pca_df['gesture'] = df_sample['gesture'].values
        
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='PC1', y='PC2', hue='gesture', data=pca_df, palette='viridis', alpha=0.7)
        plt.title('PCA данных ToF по жестам')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.7)
        plt.title('PCA данных ToF')
        plt.tight_layout()
        plt.show()
    
    print(f"Объясненная дисперсия: {pca.explained_variance_ratio_}")


# Применяем PCA к данным ToF
visualize_tof_pca(train_df)


# Статистика по длине последовательностей
sequence_lengths = train_df.groupby('sequence_id').size()
plt.figure(figsize=(12, 6))
sns.histplot(sequence_lengths, kde=True)
plt.title('Распределение длин последовательностей')
plt.xlabel('Длина последовательности')
plt.ylabel('Количество')
plt.show()

print(f"Средняя длина последовательности: {sequence_lengths.mean():.2f}")
print(f"Минимальная длина последовательности: {sequence_lengths.min()}")
print(f"Максимальная длина последовательности: {sequence_lengths.max()}")

# Анализ зависимости длины последовательности от жеста
if 'gesture' in train_df.columns:
    # Получаем длину каждой последовательности и соответствующий жест
    sequence_info = train_df.groupby('sequence_id').agg({
        'gesture': 'first',
        'sequence_id': 'count'
    }).rename(columns={'sequence_id': 'length'})
    
    plt.figure(figsize=(14, 12))
    sns.boxplot(x='gesture', y='length', data=sequence_info)
    plt.title('Длина последовательности по жестам')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Сводная статистика по числовым признакам
print("\nСводная статистика по числовым признакам:")
numeric_cols = train_df.select_dtypes(include=['number']).columns
print(train_df[numeric_cols].describe().T)

# Анализ пропущенных значений
plt.figure(figsize=(12, 6))
missing_values = train_df.isna().sum().sort_values(ascending=False)
missing_values = missing_values[missing_values > 0]
if len(missing_values) > 0:
    sns.barplot(x=missing_values.index, y=missing_values.values)
    plt.title('Количество пропущенных значений по столбцам')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
else:
    print("В датасете нет пропущенных значений.")




