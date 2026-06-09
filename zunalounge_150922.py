import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import requests
from io import StringIO




data_test = pd.read_csv("/kaggle/input/playground-series-s3e17/test.csv")

df = pd.read_csv("/kaggle/input/playground-series-s3e17/train.csv")


# Посмотрим на столбцы и типы train
df.info()


duplicate_rows_data = df[df.duplicated()]
print("Number of duplicate rows: ", duplicate_rows_data.shape)


df.sample(15) #  *Структура данных:*


# Анализ причин отказов оборудования с учетом общего флага Machine failure

# Считаем частоту каждого типа отказов ТОЛЬКО когда Machine failure = 1
failure_counts = df[df['Machine failure'] == 1][['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum().sort_values(ascending=False)

print("Частота отказов по типам (только при Machine failure = 1):")
print(failure_counts)

# Визуализация
plt.figure(figsize=(12, 6))
bars = plt.bar(failure_counts.index, failure_counts.values, color='lightcoral', edgecolor='black', alpha=0.8)

# Добавляем подписи значений на столбцы
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.title('Частота отказов оборудования по типам\n(только случаи с Machine failure = 1)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Тип отказа')
plt.ylabel('Количество случаев')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Детальная статистика
print(f"\n=== ДЕТАЛЬНАЯ СТАТИСТИКА ОТКАЗОВ ===")
print(f"Всего записей в датасете: {len(df):,}")
print(f"Всего отказов (Machine failure = 1): {df['Machine failure'].sum():,}")
print(f"Процент отказов от общего числа: {df['Machine failure'].mean():.2%}")

print(f"\nРаспределение по типам отказов:")
for failure_type, count in failure_counts.items():
    percentage = (count / df['Machine failure'].sum()) * 100
    print(f"- {failure_type}: {count:,} случаев ({percentage:.1f}% от всех отказов)")

# Анализ комбинаций отказов
failure_combinations = df[df['Machine failure'] == 1][['TWF', 'HDF', 'PWF', 'OSF', 'RNF']]
multiple_failures = (failure_combinations.sum(axis=1) > 1).sum()
single_failures = (failure_combinations.sum(axis=1) == 1).sum()

print(f"\nАнализ комбинаций отказов:")
print(f"- Одиночные отказы: {single_failures:,} случаев")
print(f"- Множественные отказы: {multiple_failures:,} случаев")
print(f"- Среднее количество причин на один отказ: {failure_combinations.sum(axis=1).mean():.2f}")


# Проверим, есть ли отказы при Machine failure = 0

# Случаи, когда Machine failure = 0, но есть флаги отказов
false_failures = df[df['Machine failure'] == 0][['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum()

print("Флаги отказов при Machine failure = 0:")
print(false_failures)

# Проверим, есть ли вообще случаи с установленными флагами при Machine failure = 0
has_failure_flags = (df[df['Machine failure'] == 0][['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum(axis=1) > 0).sum()

print(f"\nКоличество записей с Machine failure = 0, но с установленными флагами отказов: {has_failure_flags}")

# Детальный анализ противоречивых случаев
if has_failure_flags > 0:
    print("\nДетали противоречивых записей:")
    contradictory_cases = df[(df['Machine failure'] == 0) & 
                           (df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum(axis=1) > 0)]
    print(contradictory_cases[['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']].head(10))
    
    # Визуализация противоречий
    plt.figure(figsize=(10, 6))
    false_failures.plot(kind='bar', color='orange', edgecolor='black')
    plt.title('Флаги отказов при Machine failure = 0\n(Противоречивые случаи)', fontsize=14, fontweight='bold')
    plt.xlabel('Тип отказа')
    plt.ylabel('Количество случаев')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print(" Нет противоречий: при Machine failure = 0 все флаги отказов = 0")




cat_df_cols = df.select_dtypes(include=["object"]).columns
num_df_cols = df.select_dtypes(include=["float64", "int64"]).columns


import matplotlib.pyplot as plt
import seaborn as sns

# Настройка отображения графиков
plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Распределение целевой переменной (Machine failure)
df['Machine failure'].value_counts().plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightcoral'])
axes[0,0].set_title('Распределение отказов станков')
axes[0,0].set_xlabel('Отказ (0-нет, 1-да)')
axes[0,0].set_ylabel('Количество')

# 2. Типы продуктов
df['Type'].value_counts().plot(kind='bar', ax=axes[0,1], color='lightgreen')
axes[0,1].set_title('Распределение по типам продукции')
axes[0,1].set_xlabel('Тип')
axes[0,1].set_ylabel('Количество')

# 3. Износ инструмента
axes[0,2].hist(df['Tool wear [min]'], bins=30, alpha=0.7, color='orange')
axes[0,2].set_title('Распределение износа инструмента')
axes[0,2].set_xlabel('Износ (мин)')
axes[0,2].set_ylabel('Частота')

# 4. Температура процесса
axes[1,0].hist(df['Process temperature [K]'], bins=30, alpha=0.7, color='red')
axes[1,0].set_title('Температура процесса')
axes[1,0].set_xlabel('Температура (K)')
axes[1,0].set_ylabel('Частота')

# 5. Скорость вращения
axes[1,1].hist(df['Rotational speed [rpm]'], bins=30, alpha=0.7, color='purple')
axes[1,1].set_title('Скорость вращения')
axes[1,1].set_xlabel('Скорость (rpm)')
axes[1,1].set_ylabel('Частота')

# 6. Крутящий момент
axes[1,2].hist(df['Torque [Nm]'], bins=30, alpha=0.7, color='brown')
axes[1,2].set_title('Крутящий момент')
axes[1,2].set_xlabel('Момент (Nm)')
axes[1,2].set_ylabel('Частота')

plt.tight_layout()
plt.show()


print("=== ОСНОВНЫЕ СТАТИСТИКИ ===")
print(df.describe(include='all'))

print("\n=== ИНФОРМАЦИЯ О ТИПАХ ДАННЫХ ===")
print(df.info())

print("\n=== ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ ===")
print(df.isnull().sum())

print("\n=== УНИКАЛЬНЫЕ ЗНАЧЕНИЯ В КАТЕГОРИАЛЬНЫХ СТОЛБЦАХ ===")
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    print(f"{col}: {df[col].nunique()} уникальных значений")
    print(df[col].value_counts().head())
    print()


# Тепловая карта корреляций
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Матрица корреляций числовых признаков')
plt.tight_layout()
plt.show()

# Корреляции с целевой переменной
print("=== КОРРЕЛЯЦИЯ С ОТКАЗАМИ СТАНКОВ ===")
failure_correlations = correlation_matrix['Machine failure'].sort_values(ascending=False)
print(failure_correlations)


# Найдите критические значения для основных параметров
print("=== СТАТИСТИКИ ПРИ ОТКАЗАХ ===")
failure_data = df[df['Machine failure'] == 1]

for col in ['Torque [Nm]', 'Air temperature [K]', 'Tool wear [min]', 'Process temperature [K]']:
    print(f"\n{col}:")
    print(f"  Медиана при отказах: {failure_data[col].median():.2f}")
    print(f"  Медиана в норме: {df[df['Machine failure'] == 0][col].median():.2f}")
    print(f"  Разница: {failure_data[col].median() - df[df['Machine failure'] == 0][col].median():.2f}")


# Топ-5 самых значимых факторов
top_factors = failure_correlations[1:6]  # исключаем сам Machine failure

plt.figure(figsize=(10, 6))
top_factors.plot(kind='barh', color='lightcoral')
plt.title('Топ-5 факторов, влияющих на отказы станков')
plt.xlabel('Коэффициент корреляции')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# Анализ выбросов для числовых параметров
numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols):
    # Box plot для визуализации выбросов
    axes[idx].boxplot(df[col], vert=True)
    axes[idx].set_title(f'Распределение {col}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Значение')
    axes[idx].grid(True, alpha=0.3)
    
    # Подсчет выбросов (IQR метод)
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}:")
    print(f"  Выбросов: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    print(f"  Границы: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print()

# Удаляем последний пустой subplot
fig.delaxes(axes[5])
plt.tight_layout()
plt.show()



# Box plots для сравнения параметров при отказах и без отказов
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols):
    # Разделяем данные на группы
    normal_data = df[df['Machine failure'] == 0][col]
    failure_data = df[df['Machine failure'] == 1][col]
    
    # Создаем box plot
    bp = axes[idx].boxplot([normal_data, failure_data], 
                           labels=['Норма', 'Отказ'],
                           patch_artist=True)
    
    # Цвета для box plots
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[idx].set_title(f'{col}\nСравнение при отказах и без', 
                       fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Значение')
    axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Статистика
    print(f"\n{col}:")
    print(f"  Норма - Медиана: {normal_data.median():.2f}, Среднее: {normal_data.mean():.2f}")
    print(f"  Отказ - Медиана: {failure_data.median():.2f}, Среднее: {failure_data.mean():.2f}")
    print(f"  Разница медиан: {failure_data.median() - normal_data.median():.2f}")

fig.delaxes(axes[5])
plt.tight_layout()
plt.show()



# Анализ отказов по типам оборудования
print("=== АНАЛИЗ ОТКАЗОВ ПО ТИПАМ ОБОРУДОВАНИЯ ===\n")

for equipment_type in df['Type'].unique():
    type_data = df[df['Type'] == equipment_type]
    total = len(type_data)
    failures = type_data['Machine failure'].sum()
    failure_rate = (failures / total) * 100
    
    print(f"Тип {equipment_type}:")
    print(f"  Всего записей: {total:,}")
    print(f"  Отказов: {failures:,}")
    print(f"  Процент отказов: {failure_rate:.2f}%")
    print()

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 1. Распределение отказов по типам
failure_by_type = df.groupby('Type')['Machine failure'].agg(['sum', 'count'])
failure_by_type['rate'] = (failure_by_type['sum'] / failure_by_type['count']) * 100

axes[0].bar(failure_by_type.index, failure_by_type['sum'], 
            color=['lightblue', 'lightgreen', 'lightcoral'], 
            edgecolor='black', alpha=0.8)
axes[0].set_title('Количество отказов по типам оборудования', 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Тип оборудования')
axes[0].set_ylabel('Количество отказов')
axes[0].grid(True, alpha=0.3, axis='y')

# Добавляем подписи
for i, (idx, row) in enumerate(failure_by_type.iterrows()):
    axes[0].text(i, row['sum'] + 10, f"{int(row['sum'])}", 
                 ha='center', va='bottom', fontweight='bold')

# 2. Процент отказов по типам
axes[1].bar(failure_by_type.index, failure_by_type['rate'],
            color=['lightblue', 'lightgreen', 'lightcoral'],
            edgecolor='black', alpha=0.8)
axes[1].set_title('Процент отказов по типам оборудования',
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Тип оборудования')
axes[1].set_ylabel('Процент отказов (%)')
axes[1].grid(True, alpha=0.3, axis='y')

# Добавляем подписи
for i, (idx, row) in enumerate(failure_by_type.iterrows()):
    axes[1].text(i, row['rate'] + 0.05, f"{row['rate']:.2f}%",
                 ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()



# Анализ параметров для каждого типа отказа
failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
failure_names = {
    'TWF': 'Tool Wear Failure',
    'HDF': 'Heat Dissipation Failure',
    'PWF': 'Power Failure',
    'OSF': 'Overstrain Failure',
    'RNF': 'Random Failures'
}

print("=== СТАТИСТИКА ПАРАМЕТРОВ ПО ТИПАМ ОТКАЗОВ ===\n")

for failure_type in failure_types:
    if failure_type == 'RNF':
        continue  # Пропускаем RNF из-за малого количества
    
    failure_data = df[df[failure_type] == 1]
    normal_data = df[df[failure_type] == 0]
    
    print(f"\n{failure_names[failure_type]} ({failure_type}):")
    print(f"  Количество случаев: {len(failure_data)}")
    
    for param in ['Torque [Nm]', 'Tool wear [min]', 'Air temperature [K]', 
                  'Process temperature [K]', 'Rotational speed [rpm]']:
        failure_median = failure_data[param].median()
        normal_median = normal_data[param].median()
        diff = failure_median - normal_median
        diff_pct = (diff / normal_median) * 100
        
        print(f"  {param}:")
        print(f"    При отказе: {failure_median:.2f}")
        print(f"    В норме: {normal_median:.2f}")
        print(f"    Разница: {diff:.2f} ({diff_pct:+.1f}%)")

# Визуализация: средние значения параметров для каждого типа отказа
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, param in enumerate(['Torque [Nm]', 'Tool wear [min]', 'Air temperature [K]',
                            'Process temperature [K]', 'Rotational speed [rpm]']):
    failure_means = []
    failure_labels = []
    
    for failure_type in ['TWF', 'HDF', 'PWF', 'OSF']:
        failure_data = df[df[failure_type] == 1]
        if len(failure_data) > 0:
            failure_means.append(failure_data[param].mean())
            failure_labels.append(failure_type)
    
    if failure_means:
        bars = axes[idx].bar(failure_labels, failure_means, 
                            color=['lightcoral', 'lightblue', 'lightgreen', 'orange'],
                            edgecolor='black', alpha=0.8)
        axes[idx].axhline(y=df[param].mean(), color='red', linestyle='--', 
                         label='Общее среднее', linewidth=2)
        axes[idx].set_title(f'Среднее значение {param}\nпо типам отказов',
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Значение')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Подписи на столбцах
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.1f}', ha='center', va='bottom', fontsize=9)

fig.delaxes(axes[5])
plt.tight_layout()
plt.show()



# Определение пороговых значений для ключевых параметров
print("=== ПОРОГОВЫЕ ЗНАЧЕНИЯ ДЛЯ ПРЕДСКАЗАНИЯ ОТКАЗОВ ===\n")

# Анализ квантилей при отказах
key_params = ['Torque [Nm]', 'Tool wear [min]', 'Air temperature [K]', 
              'Process temperature [K]']

failure_data = df[df['Machine failure'] == 1]
normal_data = df[df['Machine failure'] == 0]

for param in key_params:
    print(f"\n{param}:")
    
    # Статистика для нормальных данных
    normal_q25 = normal_data[param].quantile(0.25)
    normal_q75 = normal_data[param].quantile(0.75)
    normal_median = normal_data[param].median()
    
    # Статистика для данных с отказами
    failure_q25 = failure_data[param].quantile(0.25)
    failure_q75 = failure_data[param].quantile(0.75)
    failure_median = failure_data[param].median()
    
    print(f"  Нормальные значения:")
    print(f"    25%: {normal_q25:.2f}, Медиана: {normal_median:.2f}, 75%: {normal_q75:.2f}")
    print(f"  При отказах:")
    print(f"    25%: {failure_q25:.2f}, Медиана: {failure_median:.2f}, 75%: {failure_q75:.2f}")
    
    # Предлагаемый порог (75-й перцентиль нормальных значений)
    threshold = normal_data[param].quantile(0.75)
    failure_rate_above_threshold = (failure_data[param] > threshold).sum() / len(failure_data) * 100
    normal_rate_above_threshold = (normal_data[param] > threshold).sum() / len(normal_data) * 100
    
    print(f"  Предлагаемый порог (75% нормальных): {threshold:.2f}")
    print(f"    Процент отказов выше порога: {failure_rate_above_threshold:.1f}%")
    print(f"    Процент нормальных выше порога: {normal_rate_above_threshold:.1f}%")

# Визуализация распределений с пороговыми значениями
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, param in enumerate(key_params[:4]):
    ax = axes[idx]
    
    # Гистограммы
    ax.hist(normal_data[param], bins=50, alpha=0.6, label='Норма', 
           color='lightblue', density=True)
    ax.hist(failure_data[param], bins=50, alpha=0.6, label='Отказ',
           color='lightcoral', density=True)
    
    # Вертикальные линии для медиан
    ax.axvline(normal_data[param].median(), color='blue', linestyle='--', 
              linewidth=2, label=f'Медиана норма: {normal_data[param].median():.1f}')
    ax.axvline(failure_data[param].median(), color='red', linestyle='--',
              linewidth=2, label=f'Медиана отказ: {failure_data[param].median():.1f}')
    
    # Пороговое значение
    threshold = normal_data[param].quantile(0.75)
    ax.axvline(threshold, color='green', linestyle=':', linewidth=2,
              label=f'Порог (75%): {threshold:.1f}')
    
    ax.set_title(f'Распределение {param}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Значение')
    ax.set_ylabel('Плотность')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()



# Корреляции между параметрами и конкретными типами отказов
failure_types = ['TWF', 'HDF', 'PWF', 'OSF']
param_cols = ['Air temperature [K]', 'Process temperature [K]',
              'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# Создаем матрицу корреляций
corr_data = df[param_cols + failure_types]
correlation_matrix = corr_data.corr()

# Извлекаем корреляции между параметрами и типами отказов
failure_correlations = correlation_matrix.loc[param_cols, failure_types]

print("=== КОРРЕЛЯЦИИ ПАРАМЕТРОВ С ТИПАМИ ОТКАЗОВ ===\n")
print(failure_correlations.round(3))

# Визуализация
plt.figure(figsize=(12, 8))
sns.heatmap(failure_correlations, annot=True, cmap='RdYlBu_r', center=0,
            square=True, linewidths=0.5, fmt='.3f', cbar_kws={'label': 'Корреляция'})
plt.title('Корреляции между параметрами и типами отказов', 
          fontsize=14, fontweight='bold')
plt.xlabel('Тип отказа')
plt.ylabel('Параметр')
plt.tight_layout()
plt.show()

# Находим наиболее сильные связи
print("\n=== НАИБОЛЕЕ СИЛЬНЫЕ СВЯЗИ ===")
for failure_type in failure_types:
    max_corr_param = failure_correlations[failure_type].abs().idxmax()
    max_corr_value = failure_correlations.loc[max_corr_param, failure_type]
    print(f"{failure_type}: {max_corr_param} (корреляция: {max_corr_value:.3f})")



# Итоговая сводка анализа
print("=" * 60)
print("ИТОГОВАЯ СВОДКА АНАЛИЗА ДАННЫХ ОБ ОТКАЗАХ ОБОРУДОВАНИЯ")
print("=" * 60)

print("\n1. ОБЩАЯ СТАТИСТИКА:")
print(f"   - Всего записей: {len(df):,}")
print(f"   - Всего отказов: {df['Machine failure'].sum():,} ({df['Machine failure'].mean():.2%})")
print(f"   - Дубликатов: {df.duplicated().sum()}")
print(f"   - Пропущенных значений: {df.isnull().sum().sum()}")

print("\n2. РАСПРЕДЕЛЕНИЕ ОТКАЗОВ ПО ТИПАМ:")
failure_counts = df[df['Machine failure'] == 1][['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum().sort_values(ascending=False)
for failure_type, count in failure_counts.items():
    pct = (count / df['Machine failure'].sum()) * 100
    print(f"   - {failure_type}: {count} случаев ({pct:.1f}%)")

print("\n3. КЛЮЧЕВЫЕ ФАКТОРЫ РИСКА (по корреляции):")
# Вычисляем корреляции заново
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
top_factors = corr_matrix['Machine failure'].abs().sort_values(ascending=False)[1:6]
for param, corr in top_factors.items():
    print(f"   - {param}: {corr:.3f}")

print("\n4. КРИТИЧЕСКИЕ ПАРАМЕТРЫ ПРИ ОТКАЗАХ:")
failure_data = df[df['Machine failure'] == 1]
normal_data = df[df['Machine failure'] == 0]
for param in ['Torque [Nm]', 'Tool wear [min]']:
    diff = failure_data[param].median() - normal_data[param].median()
    diff_pct = (diff / normal_data[param].median()) * 100
    print(f"   - {param}: +{diff:.1f} ({diff_pct:+.1f}%) при отказах")

print("\n5. РЕКОМЕНДАЦИИ ДЛЯ МОДЕЛИРОВАНИЯ:")
print("   - Использовать признаки: Torque, Tool wear, HDF, OSF, PWF, TWF")
print("   - Учесть тип оборудования (L, M, H) как категориальный признак")
print("   - Нормализовать числовые признаки")
print("   - Удалить или обработать аномальные записи (315 случаев)")
print("   - Учесть дисбаланс классов (1.57% отказов)")

print("\n" + "=" * 60)


