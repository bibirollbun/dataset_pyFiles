import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
from tqdm import tqdm
import dask.dataframe as dd
from sklearn.preprocessing import LabelEncoder
from scipy import stats

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Настройка для работы с большими данными
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.3f}'.format)

# Функция для очистки памяти
def clear_memory():
    gc.collect()


train_labels = pd.read_csv('/kaggle/input/amex-default-prediction/train_labels.csv')
print(f"Размер train_labels: {train_labels.shape}")
print(f"Количество уникальных клиентов: {train_labels['customer_ID'].nunique()}")
print("\nПервые 5 строк:")
print(train_labels.head())
print(f"\nДубликатов customer_ID: {train_labels['customer_ID'].duplicated().sum()}")


print("Распределение классов:")
class_dist = train_labels['target'].value_counts()
print(class_dist)
print(f"\nДоля класса 1 (дефолт): {class_dist[1]/len(train_labels)*100:.2f}%")

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.pie(class_dist.values, labels=['Не дефолт (0)', 'Дефолт (1)'], 
            autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
ax.set_title('Процентное распределение')

plt.tight_layout()
plt.show()


train_sample = pd.read_csv('/kaggle/input/amex-default-prediction/train_data.csv', nrows=100000)
print(f"Размер сэмпла: {train_sample.shape}")
print("\nТипы данных:")
print(train_sample.dtypes.value_counts())

print("\nПервые 5 строк:")
print(train_sample.head())
print("\nИнформация о датасете:")
print(train_sample.info(memory_usage='deep'))


print("Анализ временных меток:")
train_sample['S_2'] = pd.to_datetime(train_sample['S_2'])
print(f"Уникальные значения S_2 (дата): {train_sample['S_2'].nunique()}")
print(f"Диапазон дат: от {train_sample['S_2'].min()} до {train_sample['S_2'].max()}")

# Анализ количества записей на клиента
records_per_customer = train_sample.groupby('customer_ID').size()
print(f"\nСтатистика записей на клиента:")
print(f"Среднее: {records_per_customer.mean():.2f}")
print(f"Медиана: {records_per_customer.median():.2f}")
print(f"Мин: {records_per_customer.min()}")
print(f"Макс: {records_per_customer.max()}")
print(f"Стандартное отклонение: {records_per_customer.std():.2f}")

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Распределение количества записей на клиента
axes[0].hist(records_per_customer, bins=50, edgecolor='black')
axes[0].set_title('Распределение количества записей на клиента')
axes[0].set_xlabel('Количество записей')
axes[0].set_ylabel('Частота')

# Временной анализ (количество записей по месяцам)
monthly_counts = train_sample['S_2'].dt.to_period('M').value_counts().sort_index()
axes[1].bar(range(len(monthly_counts)), monthly_counts.values)
axes[1].set_title('Количество записей по месяцам')
axes[1].set_xlabel('Месяц')
axes[1].set_ylabel('Количество записей')
axes[1].set_xticks(range(len(monthly_counts)))
axes[1].set_xticklabels([str(p) for p in monthly_counts.index], rotation=45)

plt.tight_layout()
plt.show()


chunk_size = 100000
missing_stats = []
total_rows = 0

print("Анализ пропущенных значений по частям...")
for chunk in tqdm(pd.read_csv('/kaggle/input/amex-default-prediction/train_data.csv', chunksize=chunk_size)):
    missing_percent = chunk.isnull().sum() / len(chunk) * 100
    missing_stats.append(missing_percent)
    total_rows += len(chunk)

# Объединяем статистику
missing_df = pd.DataFrame(missing_stats).mean().sort_values(ascending=False)
missing_df = pd.DataFrame({'feature': missing_df.index, 'missing_percent': missing_df.values})

print("\nТоп признаков с наибольшим количеством пропусков:")
print(missing_df.head(35))

# Визуализация
plt.figure(figsize=(12, 12))
top_35 = missing_df.head(35)
plt.barh(top_35['feature'], top_35['missing_percent'])
plt.xlabel('Процент пропусков (%)')
plt.ylabel('Признаки')
plt.title('Топ признаков с наибольшим количеством пропусков')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

clear_memory()


categorical_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 
                        'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
train_sample = pd.read_csv('/kaggle/input/amex-default-prediction/train_data.csv', nrows=500000)

print("Анализ типов признаков:")
print(f"Всего признаков: {train_sample.shape[1]}")

# Разделяем признаки по категориям
numeric_features = []
cat_features_found = []
for col in train_sample.columns:
    if col in categorical_features:
        cat_features_found.append(col)
    elif col not in ['customer_ID', 'S_2', 'target']:
        numeric_features.append(col)

print(f"\nКатегориальные признаки (из описания): {len(cat_features_found)}")
print(f"Числовые признаки: {len(numeric_features)}")
print(f"Служебные колонки: customer_ID, S_2")

print("\nАнализ категориальных признаков:")
for col in cat_features_found:
    if col in train_sample.columns:
        unique_vals = train_sample[col].nunique()
        print(f"{col}: {unique_vals} уникальных значений")
        
        plt.figure(figsize=(10, 4))
        train_sample[col].value_counts().head(20).plot(kind='bar')
        plt.title(f'Распределение признака {col}')
        plt.xlabel('Значение')
        plt.ylabel('Частота')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

clear_memory()


print("Анализ распределения числовых признаков...")
sample_features = numeric_features[:20]  # Первые 20 числовых признаков

fig, axes = plt.subplots(5, 4, figsize=(20, 15))
axes = axes.flatten()

for i, col in enumerate(sample_features):
    axes[i].hist(train_sample[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[i].set_title(f'{col}')
    axes[i].set_xlabel('Значение')
    axes[i].set_ylabel('Частота')
    
    mean_val = train_sample[col].mean()
    median_val = train_sample[col].median()
    axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
    axes[i].axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'Median: {median_val:.2f}')
    axes[i].legend(fontsize=8)

plt.tight_layout()
plt.show()

print("\nБазовые статистики для числовых признаков:")
stats_df = pd.DataFrame({
    'mean': train_sample[sample_features].mean(),
    'std': train_sample[sample_features].std(),
    'min': train_sample[sample_features].min(),
    '25%': train_sample[sample_features].quantile(0.25),
    '50%': train_sample[sample_features].quantile(0.50),
    '75%': train_sample[sample_features].quantile(0.75),
    'max': train_sample[sample_features].max(),
    'missing': train_sample[sample_features].isnull().sum() / len(train_sample) * 100
})
print(stats_df)


print("Анализ выбросов...")
selected_features = numeric_features[:20]

fig, axes = plt.subplots(2, 10, figsize=(20, 15))
axes = axes.flatten()

for i, col in enumerate(selected_features):
    train_sample[[col]].boxplot(ax=axes[i])
    axes[i].set_title(f'{col}')
    axes[i].set_ylabel('Значение')
    
    # Правило 1.5*IQR
    Q1 = train_sample[col].quantile(0.25)
    Q3 = train_sample[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = train_sample[(train_sample[col] < lower_bound) | (train_sample[col] > upper_bound)]
    outlier_percent = len(outliers) / len(train_sample) * 100
    
    axes[i].text(0.05, 0.95, f'Выбросов: {outlier_percent:.1f}%', 
                transform=axes[i].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()


print("Анализ корреляций...")
customer_target_map = train_labels.set_index('customer_ID')['target'].to_dict()
train_sample['target'] = train_sample['customer_ID'].map(customer_target_map)
corr_features = numeric_features[:20] + ['target']

corr_matrix = train_sample[corr_features].corr(method='spearman')
target_correlations = corr_matrix['target'].drop('target').sort_values(ascending=False)

print("\nТоп признаков с наибольшей корреляцией с target:")
print(target_correlations.head(20))

# Визуализация корреляций с целевой переменной
plt.figure(figsize=(12, 6))
top_corr = target_correlations.head(20)
colors = ['red' if x < 0 else 'blue' for x in top_corr.values]
plt.barh(top_corr.index, top_corr.values, color=colors)
plt.xlabel('Корреляция с target (Spearman)')
plt.title('Топ признаков по корреляции с целевой переменной')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()

top_features = list(target_correlations.head(15).index) + ['target']
plt.figure(figsize=(12, 10))
sns.heatmap(train_sample[top_features].corr(method='spearman'), 
            cmap='coolwarm', center=0, annot=True, fmt='.2f')
plt.title('Корреляционная матрица (топ-15 признаков + target)')
plt.tight_layout()
plt.show()

clear_memory()




