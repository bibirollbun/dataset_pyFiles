!pip install anytree


import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from anytree import Node, RenderTree

import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', parse_dates=['date'])
solution = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')
inventory = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv")
calendar = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv")
test_weights = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')


display(train.head())
display(test.head())
display(inventory.head())


null_elements = train.isna().sum()
null_elements[null_elements > 0].sort_values(ascending=False)


null_elements = test.isna().sum()
null_elements[null_elements > 0].sort_values(ascending=False)


null_elements = inventory.isna().sum()
null_elements[null_elements > 0].sort_values(ascending=False)


train.nunique()


"""
Ячейка, выводящая иерархию продуктов
"""
traint_inventory = train.merge(inventory,on=['warehouse','unique_id'],how='left')
root = Node("Products")
nodes = {"root": root}

for _, row in traint_inventory[['L1_category_name_en', 'L2_category_name_en', 'L3_category_name_en', 'L4_category_name_en']].drop_duplicates().iterrows():
    level1 = row['L1_category_name_en']
    level2 = row['L2_category_name_en']
    level3 = row['L3_category_name_en']
    level4 = row['L4_category_name_en']

    if level1 not in nodes:
        nodes[level1] = Node(level1, parent=root)
    if level2 not in nodes:
        nodes[level2] = Node(level2, parent=nodes[level1])
    if level3 not in nodes:
        nodes[level3] = Node(level3, parent=nodes[level2])
    if level4 not in nodes:
        nodes[level4] = Node(level4, parent=nodes[level3])

for pre, _, node in RenderTree(root):
    print(f"{pre}{node.name}")


train.dtypes


numeric_columns = [
    'total_orders', 'sales', 'sell_price_main', 'availability', 'type_0_discount', 'type_1_discount',
    'type_2_discount', 'type_3_discount', 'type_4_discount', 'type_5_discount', 'type_6_discount'
]

id_level_stats = (
    train.groupby('unique_id')[numeric_columns]
    .agg(['min', 'max', 'mean', 'std'])
    .reset_index()
)

# Переименовываем столбцы для удобства
id_level_stats.columns = ['_'.join(col).rstrip('_') for col in id_level_stats.columns]
id_level_stats.rename(columns={'unique_id_': 'unique_id'}, inplace=True)

# Агрегирование по всем unique_id
aggregated_stats = id_level_stats.drop(columns='unique_id').max().reset_index()
aggregated_stats.columns = ['feature_stat', 'value']

# Разбиваем 'feature_stat' на фичу и статистику для построения таблицы
aggregated_stats[['feature', 'stat']] = aggregated_stats['feature_stat'].str.rsplit('_', n=1, expand=True)
final_table = aggregated_stats.pivot(index='stat', columns='feature', values='value')

print("Результаты на уровне ID:")
display(id_level_stats.head())
print("\nИтоговая таблица:")
display(final_table.head())


inventory.dtypes


def draw_boxplot(df: pd.DataFrame, category: str, threshold: int = 10000) -> None:
    """
    Распределение категорийных данных с объединением редких категорий в "Other" для улучшения читаемости.
    
    :param data: pd.DataFrame - Исходные данные.
    :param category: str - Имя категориального признака.
    :param threshold: int - Порог для объединения редких категорий в "Other".
    """
    data = df.copy()
    # Считаем количество каждого значения категории
    category_counts = data[category].value_counts()

    # Категории, которые будут объединены в "Other"
    rare_categories = category_counts[category_counts < threshold].index

    # Заменяем редкие категории на "Other"
    data[category] = data[category].apply(lambda x: 'Other' if x in rare_categories else x)

    # Строим график
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=category, palette='Set2')
    plt.title(f'Частота категорий в {category}', fontsize=14)
    plt.xlabel('Категории', fontsize=12)
    plt.ylabel('Количество', fontsize=12)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


cat_features = {'L1_category_name_en': 0, 'L2_category_name_en': 100, 
                'L3_category_name_en': 17000,
                'L4_category_name_en': 10000, 'warehouse': 0}

for cat in cat_features:
    draw_boxplot(traint_inventory, cat, cat_features[cat])


def plot_numeric_feature_distributions(df: pd.DataFrame, bins: int = 30):
    """
    Визуализирует распределение числовых признаков в DataFrame.
    
    :param data: pd.DataFrame - Данные для анализа.
    :param bins: int - Количество бинов (корзин) для гистограмм.
    """
    data = df.replace([float('inf'), float('-inf')], float('nan')).dropna()
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    num_features = len(numeric_cols)
    cols = 3  # Количество столбцов в сетке
    rows = (num_features + cols - 1) // cols 
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        sns.histplot(data[col], bins=bins, kde=True, ax=axes[i], color='blue')
        axes[i].set_title(f'Distribution of {col}', fontsize=12)
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
    
    # Удаляем пустые графики, если их больше, чем числовых колонок
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
plot_numeric_feature_distributions(train)


def get_rare_categories(data: pd.DataFrame, category: str, threshold: int = 10) -> None:
    """
    Выводит количество редких категорий, которые встречаются менее чем в threshold от общего числа.
    
    :param data: pd.DataFrame - Исходные данные.
    :param category: str - Имя категориального признака.
    :param threshold: int - Порог для определения редких категорий (например, 10).
    """
    # Считаем количество каждого значения категории
    category_counts = data[category].value_counts()
    total_count = len(data)

    # Находим редкие категории, которые встречаются меньше чем порог
    rare_categories = category_counts[category_counts < threshold]

    # Выводим количество редких категорий
    print(f"Общее количество записей: {total_count}")
    print(f"Порог для редких категорий (кол-во): {threshold}")
    print(f"Количество редких категорий: {len(rare_categories)}")
    print("Редкие категории:")
    print(rare_categories)

for cat in cat_features:
    print(cat)
    get_rare_categories(traint_inventory, cat, threshold=50)
    print()


def draw_raw_ts(
    data: pd.DataFrame,
    id_column: str,
    target_column: str,
    date_column: str,
    ncols: int = 2,
):
    """
    Код из примеров по LAMA
    Draw graphs of time series with specified parameters.
    Args:
        - data: pd.DataFrame with time series data
        - id_column: id column name in dataset
        - target_column: target column name in dataset
        - date_column: date column name in dataset
        - ncols: number of columns for subplot's grid
    """
    # Initialize grid's shape
    data = data.sort_values(by=[id_column, date_column])
    num_ts = data[id_column].nunique()
    nrows = num_ts // ncols + num_ts % ncols
    fig, ax = plt.subplots(
        nrows, 
        ncols, 
        figsize=(24, 5 * nrows)
    )
    axes_to_del = nrows * ncols - num_ts
    for i in range(axes_to_del):
        i_row = (nrows - 1) - i // ncols
        i_col = (ncols - 1) - i % ncols
        fig.delaxes(ax[i_row][i_col])
    
    # Draw graphs
    for i, ts_id in enumerate(data[id_column].unique()):
        i_row = i // ncols
        i_col = i % ncols
        
        ts_df = data[data[id_column] == ts_id]
        ax = ax.reshape(nrows, ncols)
        ax[i_row, i_col].plot(ts_df[date_column], ts_df[target_column])
        ax[i_row, i_col].title.set_text(f"TS with ID {ts_id}")

draw_raw_ts(train[train.unique_id < 7], 'unique_id', 'sales', 'date')


# Функция для анализа последовательности дней и пропусков
def analyze_time_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляем периодичность поступления данных и наличие пропусков для каждого ряда.
    """
    results = []
    
    for id_, group in tqdm(df.groupby('unique_id'), 
                           total=len(df.groupby('unique_id'))):
        # Сортируем данные по времени
        group = group.sort_values('date')
        
        # Вычисляем разницу между соседними датами
        group['time_diff'] = group['date'].diff().dt.days
        
        # Определяем наиболее частую разницу
        most_common_gap = group['time_diff'].mode().iloc[0] if not group['time_diff'].isnull().all() else None
        
        # Количество пропусков (если разница больше ожидаемого интервала)
        expected_dates = pd.date_range(start=group['date'].min(), 
                                       end=group['date'].max(), 
                                       freq=f'{most_common_gap}D' if most_common_gap else 'D')
        actual_dates = group['date']
        missing_dates = set(expected_dates) - set(actual_dates)
        
        results.append({
            'id': id_,
            'most_common_gap': most_common_gap,
            'missing_count': len(missing_dates),
            'total_entries': len(group),
        })
    
    return pd.DataFrame(results)

# Анализируем пропуски
analysis_results = analyze_time_gaps(train)
analysis_results.head()


analysis_results[['id', 'most_common_gap']].nunique()


data = analysis_results.missing_count / analysis_results.total_entries * 100
data[data < 1].shape[0], data[data >= 1].shape[0]


print(data.isnull().sum())  # Проверяем на наличие NaN
print((data == float('inf')).sum())  # Проверяем на наличие inf


plt.figure(figsize=(12, 6))
sns.histplot(data[data < 1], bins=30, kde=False, color="blue")
plt.xlabel("% of gaps")
plt.ylabel("Frequency")
plt.title("% of gaps vs total entriess")
plt.show()  # Возникае ворнинг, но я не знаю почему


plt.figure(figsize=(12, 6))
sns.histplot(data[data >= 1], bins=30, kde=False, color="blue")
plt.xlabel("% of gaps")
plt.ylabel("Frequency")
plt.title("% of gaps vs total entriess")
plt.show()  # Возникае ворнинг, но я не знаю почему


# Визуализация зависимости sales от признаков
discount_cols = [
    'total_orders', 'sell_price_main', 'type_0_discount', 'type_1_discount',
    'type_2_discount', 'type_3_discount', 'type_4_discount', 'type_5_discount', 'type_6_discount'
]

for col in discount_cols:
    sns.scatterplot(x=train[col], y=train["sales"])
    plt.title(f"Sales vs {col}")
    plt.xlabel(col)
    plt.ylabel("Sales")
    plt.show()


all_dates = pd.date_range(start=train['date'].min(), end=train['date'].max())
unique_ids = train['unique_id'].unique()

# Создаем DataFrame с унифицированными временными рамками
full_data = pd.DataFrame({
    'date': np.repeat(all_dates, len(unique_ids)),
    'unique_id': np.tile(unique_ids, len(all_dates))
})

# Объединяем с исходным датасетом
full_data = full_data.merge(train[['date', 'unique_id']+numeric_columns], on=['date', 'unique_id'], how='left')
full_data[numeric_columns] = full_data[numeric_columns].fillna(0)

# Рассчитываем корреляцию для каждого unique_id
correlation_results = []

for uid in tqdm(unique_ids):
    subset = full_data[full_data['unique_id'] == uid]
    subset = subset.sort_values('date')[numeric_columns]
    if subset.dropna().shape[0] > 1:  # Только если достаточно данных
        corr = subset.corr().values  # Сохраняем корреляционную матрицу
        correlation_results.append(corr)

# Усредняем корреляции по всем unique_id
mean_corr = np.nanmean(correlation_results, axis=0)

plt.figure(figsize=(10, 8))
sns.heatmap(mean_corr, annot=True, cmap='coolwarm', fmt='.2f', 
            xticklabels=numeric_columns, yticklabels=numeric_columns, linewidths=0.5)
plt.title('Средняя корреляция между признаками для всех unique_id')
plt.show()


max_data_start = train.groupby('unique_id').date.min().max()
max_data_end = train.date.max()

print(f"Максимальная дата старта временного ряда: {max_data_start}, крайняя дата актуальности данных {max_data_end}")


# Считаем у скольких рядов представленная выше дата - крайняя?

sum(train.groupby('unique_id').date.max() == max_data_end)


train.groupby('unique_id').date.max().min()  # у каких-то рядов вообще очень рано заканчивается информация


len(set(test.unique_id)), max(set(test.unique_id))  # Видно, что не все, так как в трейне #id = 5930


max(set(train.unique_id))  


uni_id_test = set(test.unique_id)
df = train[train.unique_id.isin(uni_id_test)]
df.groupby('unique_id').date.max().min()  # Да, есть


calendar['holiday_name'].isna().mean() * 100  # много пропусков из-за специфики таблицы, надо будет обработать


import numpy as np
import pandas as pd
from datetime import datetime
from typing import List
import scipy.stats as stats


train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', parse_dates=['date'])
solution = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')
inventory = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv")
calendar = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv")
test_weights = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')


czech_holiday = [ 
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"),
]
brno_holiday = [
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"),
]

budapest_holidays = []
munich_holidays = [
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),
]

frank_holidays = [
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),
]

def fill_loss_holidays(df_fill, warehouses, holidays):
    """
    Заполнение дополнительных выходных для календаря
    """
    df = df_fill.copy()
    for item in holidays:
        dates, holiday_name = item
        generated_dates = [datetime.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d') for date in dates]
        for generated_date in generated_dates:
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday'] = 1
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday_name'] = holiday_name
    return df

calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Prague_1', 'Prague_2', 'Prague_3'], holidays=czech_holiday)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Brno_1'], holidays=brno_holiday)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Munich_1'], holidays=munich_holidays)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Frankfurt_1'], holidays=frank_holidays)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Budapest_1'], holidays=budapest_holidays)


def process_calendar(calendar: pd.DataFrame, warehouses: List[str]) -> pd.DataFrame:
    """
    Обрабатывает календарные данные для указанных складов и добавляет новые признаки, исключая пересечение данных между складами.
    
    Аргументы:
        calendar (pd.DataFrame): Датафрейм с колонками "warehouse", "date", "holiday", "shops_closed".
        warehouses (List[str]): Список складов, для которых нужно обработать данные.
    
    Возвращает:
        pd.DataFrame: Обработанный датафрейм с дополнительными признаками.
    """
    calendar = calendar[calendar["warehouse"].isin(warehouses)].copy()
    if calendar["date"].dtype != "datetime64[ns]":
        calendar["date"] = pd.to_datetime(calendar["date"])

    calendar = calendar.sort_values(["warehouse", "date"]).reset_index(drop=True)

    # Группируем по складам и применяем логику обработки
    def process_group(df: pd.DataFrame) -> pd.DataFrame:
        # Признак: дни до следующего праздника
        df["next_holiday_date"] = df.loc[df["holiday"] == 1, "date"].shift(-1)
        df["next_holiday_date"] = df["next_holiday_date"].bfill()
        df["days_to_holiday"] = (df["next_holiday_date"] - df["date"]).dt.days
        df.drop(columns=["next_holiday_date"], inplace=True)

        # Признак: дни до следующего закрытия магазинов
        df["next_shops_closed_date"] = df.loc[df["shops_closed"] == 1, "date"].shift(-1)
        df["next_shops_closed_date"] = df["next_shops_closed_date"].bfill()
        df["days_to_shops_closed"] = (df["next_shops_closed_date"] - df["date"]).dt.days
        df.drop(columns=["next_shops_closed_date"], inplace=True)

        # Признак: день после закрытия магазинов
        df["day_after_closing"] = (
            (df["shops_closed"] == 0) & (df["shops_closed"].shift(1) == 1)
        ).astype(int)

        # Признак: длинный выходной
        df["long_weekend"] = (
            (df["shops_closed"] == 1) & (df["shops_closed"].shift(1) == 1)
        ).astype(int)

        # Признак: день недели
        df["weekday"] = df["date"].dt.weekday

        return df

    # Применяем обработку по каждой группе (складу)
    calendar = calendar.groupby("warehouse", group_keys=False).apply(process_group)

    return calendar
    
warehouses = ["Frankfurt_1", "Prague_2", "Brno_1", "Munich_1", "Prague_3", "Prague_1", "Budapest_1"]
calendar_extended = process_calendar(calendar, warehouses)


def fill_missing_values(data: pd.DataFrame, date_col: str, group_col: str, target_cols: list[str]) -> pd.DataFrame:
    """
    Заполняет пропущенные значения в указанных колонках, используя среднее значение по временным рядам.

    :param data: pd.DataFrame - Исходные данные.
    :param date_col: str - Имя колонки с датой.
    :param group_col: str - Имя колонки, по которой группировать данные (например, склад или уникальный идентификатор).
    :param target_cols: list[str] - Список колонок, пропущенные значения которых нужно заполнить.
    :return: pd.DataFrame - Данные с заполненными значениями.
    """
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(by=[group_col, date_col])

    for col in target_cols:
        if col in data.columns:
            # Применяем заполнение вперед-назад с помощью transform
            data[col] = data.groupby(group_col)[col].transform(lambda x: x.ffill().bfill())

            # Если всё ещё остались пропуски, заполняем средним значением по группе
            data[col] = data[col].fillna(data.groupby(group_col)[col].transform('mean'))

    return data

train = fill_missing_values(train, date_col="date", group_col="unique_id", target_cols=["total_orders", "sales"])
test = fill_missing_values(test, date_col="date", group_col="unique_id", target_cols=["total_orders"])


null_elements = train.isna().sum()
null_elements[null_elements > 0].sort_values(ascending=False)


null_elements = test.isna().sum()
null_elements[null_elements > 0].sort_values(ascending=False)


def create_new_features(dataset: pd.DataFrame, calendar_extended: pd.DataFrame,
                        inventory: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """
    Создает новые признаки на основе исходного датасета и дополнительных данных.

    :param dataset: pd.DataFrame - Основной датасет с историей продаж.
    :param calendar_extended: pd.DataFrame - Данные календаря с дополнительной информацией (праздники, выходные).
    :param inventory: pd.DataFrame - Данные о складе и уникальных идентификаторах.
    :param weights: pd.DataFrame - Данные о весах для расчета целевых переменных.
    :return: pd.DataFrame - Датасет с добавленными признаками.

    Особенности признаков:
    1. Удаление `availability`: Этот столбец удаляется, так как информация о доступности будет отсутствовать в тестовом наборе.
    2. Признаки календаря: Добавляются временные признаки для анализа сезонности и временных закономерностей.
    3. Информация о скидках: Рассчитываются общие и индивидуальные метрики на основе типа скидок.
    4. Нормализация: Вычисляются отношения продаж к цене, скидкам и их комбинациям.
    5. Категориальные признаки: Преобразуются в тип `category` для уменьшения размера и оптимизации обработки.
    """
    # Копируем датасет, чтобы избежать изменения исходных данных
    df = dataset.copy()

    # Удаляем колонку "availability", так как она не используется
    if 'availability' in df.columns:
        df.drop(['availability'], axis=1, inplace=True)

    # Объединение с дополнительными данными
    df = df.merge(calendar_extended, on=['date', 'warehouse'], how='left')
    df = df.merge(inventory, on=['unique_id', 'warehouse'], how='left')
    
    # Добавляем веса только если столбец "sales" присутствует
    if 'sales' in df.columns:
        df = df.merge(weights, on=['unique_id'], how='left')
        
    # Преобразование даты в различные временные признаки
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year  # Год
    df['month'] = df['date'].dt.month  # Месяц
    df['day'] = df['date'].dt.day  # День месяца
    df['weekday'] = df['date'].dt.weekday  # День недели (0 - понедельник)
    df['weekofyear'] = df['date'].dt.isocalendar().week  # Номер недели
    df['dayofyear'] = df['date'].dt.dayofyear  # День года
    df['is_month_start'] = df['date'].dt.is_month_start  # Начало месяца
    df['is_month_end'] = df['date'].dt.is_month_end  # Конец месяца
    df['quarter'] = df['date'].dt.quarter  # Квартал

    # Общий размер скидки по всем типам
    df["total_dic"] = (
        df['type_0_discount'] + df['type_1_discount'] +
        df['type_2_discount'] + df['type_3_discount'] +
        df['type_4_discount'] + df['type_5_discount'] +
        df['type_6_discount']
    )

    # Нормализованные продажи
    df['total_orders_'] = df['total_orders'] / df['sell_price_main']  # Отношение продаж к цене
    df['total_orders_dic'] = df['total_orders_'] / df["total_dic"]  # Продажи с учетом скидок
    df['total_orders_sell_price_main'] = df['sell_price_main'] / df["total_dic"]  # Цена с учетом скидок
    
    df.fillna(0, inplace=True)

    categorical_columns = ['unique_id'] + list(df.select_dtypes("object").columns)
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    return df


train = create_new_features(train, calendar_extended, inventory, test_weights)
test = create_new_features(test, calendar_extended, inventory, test_weights)


train.shape


def detect_and_smooth_anomalies(data: pd.DataFrame, date_col: str, target_col: str, group_col: str, 
                                confidence_level: float = 0.95, window: int = 10) -> pd.DataFrame:
    """
    Выявляет и сглаживает аномалии в данных временных рядов локально в пределах окон.
    
    :param data: pd.DataFrame - Исходные данные.
    :param date_col: str - Имя колонки с датой.
    :param target_col: str - Имя колонки, где ищем аномалии.
    :param group_col: str - Имя колонки для группировки (например, склад или уникальный ID).
    :param confidence_level: float - Уровень доверия.
    :param window: int - Кол-во окон для локального анализа.
    :return: pd.DataFrame - Данные с обработанными аномалиями.
    """
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(by=[group_col, date_col])
    anomaly_count = 0  # Счетчик аномалий
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    def smooth_group(group):
        nonlocal anomaly_count, z_score
        window_size = int(len(group) / window)
        prev = 0
        
        for i in range(1, window + 1):
            current = min(i * window_size, len(group) - 1)
            
            window_values = group[target_col].iloc[prev:current].to_numpy()
            local_mean = window_values.mean()
            local_std = window_values.std()
            
            if local_std == 0 or np.isnan(local_std):
                continue
            
            # Доверительный интервал
            lower_bound = local_mean - z_score * local_std
            upper_bound = local_mean + z_score * local_std
            anomalies_lower = (window_values < lower_bound)
            
            # в продажах нули значимы, не аномалии
            all_close = np.abs(window_values) < 1e-6
            anomalies_lower[all_close] = False 
            anomalies_upper = (window_values > upper_bound)

            anomaly_count += anomalies_upper.sum() + anomalies_lower.sum()

            window_values[anomalies_upper] = upper_bound
            window_values[anomalies_lower] = lower_bound

            group.loc[group.index[prev:current], target_col] = window_values
            prev = current

        return group

    # Применяем сглаживание по группам
    data = data.groupby(group_col, observed=False).apply(smooth_group).reset_index(drop=True)
    
    print(f"Кол-во найденных аномалий: {anomaly_count}")
    
    return data

train = detect_and_smooth_anomalies(
    train, 
    date_col="date", 
    target_col="sales", 
    group_col="unique_id", 
    window=5, 
    confidence_level=0.95
)


# Сохраним для следующей быстрой загрузки
train.to_csv("prepared_train.csv", index=False)
test.to_csv("prepared_test.csv", index=False)


!pip install lightautoml


import torch
import numpy as np
import pandas as pd

import joblib
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


train = pd.read_csv('rohlik-sales-forecasting-challenge-v2/prepared_train.csv', parse_dates=['date'])
test = pd.read_csv('rohlik-sales-forecasting-challenge-v2/prepared_test.csv', parse_dates=['date'])


train.dtypes


train = train.sort_values(['unique_id', 'date'])
test = test.sort_values(['unique_id', 'date'])


N_THREADS = 8
N_FOLDS = 5
RANDOM_STATE = 42
TIMEOUT = 3600 * 7  # 7 часов

np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)


roles = {
    'target': 'sales',
    'weights': 'weight'
}
tuning_params = {'max_tuning_time': 3600}


task = Task('reg', loss = 'mae', metric = 'mae')

automl = TabularAutoML(
    task = task,
    timeout = TIMEOUT,
    cpu_limit = N_THREADS,
    selection_params={'mode' : 0},
    tuning_params = tuning_params,
    reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE}
)


out_of_fold_predictions = automl.fit_predict(
    train,
    roles = roles, 
    verbose = 3
)


test_predictions = automl.predict(test)

sub = test.copy()
sub['sales_hat'] = test_predictions.data[:, 0]
sub['id'] = sub['unique_id'].astype(str) + "_" + sub['date'].astype(str)
sub[['id','sales_hat']].to_csv("autoML-1.csv", index=False)


joblib.dump(automl, 'autoML-1.pkl')


%%time
accurate_fi = automl.get_feature_scores('accurate', train, silent = False)


accurate_fi.set_index('Feature')['Importance'].plot.bar(figsize = (30, 10), grid = True)


import torch
import numpy as np
import pandas as pd

import joblib
from lightautoml.automl.base import AutoML
from lightautoml.utils.timer import PipelineTimer
from lightautoml.tasks import Task
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.automl.blend import BestModelSelector, WeightedBlender
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector, ModelBasedImportanceEstimator


train = pd.read_csv('rohlik-sales-forecasting-challenge-v2/prepared_train.csv', parse_dates=['date'])
test = pd.read_csv('rohlik-sales-forecasting-challenge-v2/prepared_test.csv', parse_dates=['date'])

train = train.sort_values(['unique_id', 'date'])
test = test.sort_values(['unique_id', 'date'])


N_THREADS = 8
N_FOLDS = 5
RANDOM_STATE = 42
TIMEOUT = 3600 * 10  # 10 часов
timer = PipelineTimer(timeout=TIMEOUT)

np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)

roles = {
    'target': 'sales',
    'weights': 'weight',
    # 'datetime': ['date']
}


task = Task('reg', loss = 'mae', metric = 'mae')
reader = PandasToPandasReader(task, cv=N_FOLDS, random_state=RANDOM_STATE)


model0 = BoostLGBM(
    default_params={
        "learning_rate": 0.05,
        "num_leaves": 90,
        "seed": RANDOM_STATE,
        "num_threads": N_THREADS,
        "objective": 'regression',
        "metric": 'mae',
        "boosting_type": 'gbdt'
    }
)
pipe0 = LGBSimpleFeatures()
mbie = ModelBasedImportanceEstimator()
selector = ImportanceCutoffSelector(pipe0, model0, mbie, cutoff=0)

pipe = LGBSimpleFeatures()
params_tuner1 = OptunaTuner(n_trials=5, timeout=45*60*5)
model1 = BoostLGBM(
    default_params={'learning_rate': 0.05, 'num_leaves': 90, 
                    'seed': RANDOM_STATE + 1, 'num_threads': N_THREADS,
                    "objective": 'regression', "metric": 'mae',
                    "boosting_type": 'gbdt'}
)

model2 = BoostLGBM(
    default_params={'learning_rate': 0.025, 'num_leaves': 64, 
                    'seed': RANDOM_STATE + 2, 'num_threads': N_THREADS,
                    "metric": 'mae'}
)

pipeline_lvl1 = MLPipeline([
    (model1, params_tuner1),
    model2
], pre_selection=selector, features_pipeline=pipe, post_selection=None)

pipe1 = LGBSimpleFeatures()

model = BoostLGBM(
    default_params={'learning_rate': 0.05, 'num_leaves': 64, 'max_bin': 1024, 
                    'seed': RANDOM_STATE + 3, 'num_threads': N_THREADS},
    freeze_defaults=True
)

pipeline_lvl2 = MLPipeline([model], pre_selection=None, features_pipeline=pipe1, post_selection=None)
blender = WeightedBlender()
automl = AutoML(reader, [
                        [pipeline_lvl1],
                        [pipeline_lvl2],
                    ], 
                    skip_conn=False,
                    timer = timer,
                    blender = blender)


out_of_fold_predictions = automl.fit_predict(train, 
                                             roles=roles, 
                                             verbose = 3)


test_predictions = automl.predict(test)

sub = test.copy()
sub['sales_hat'] = test_predictions.data[:, 0]
sub['id'] = sub['unique_id'].astype(str) + "_" + sub['date'].astype(str)
sub[['id','sales_hat']].to_csv("autoML-2.csv", index=False)


joblib.dump(automl, 'autoML-2.pkl')


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor
from category_encoders import TargetEncoder
from lightgbm import LGBMRegressor


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path, parse_dates=['date'])
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Failed to load data: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocess the dataset and split it into train and test sets."""
    if not np.all(np.isfinite(df.select_dtypes(include=[np.number]))):
        print("Dataset contains infinite or NaN values. Replacing with 0.")
        df.replace([np.inf, -np.inf], 0, inplace=True)
    
    train_start_date = '2020-08-01'
    train_end_date = '2024-03-18'
    test_start_date = '2024-03-18'
    test_end_date = '2024-06-01'
    """
    Валидируемся на ближайших данных, если смогли аппроксимировать их, то тогда 
    и обучившись с ними мы сможем точнее предсказывать таргет 
    Больше 14 дней, так как следует рассмотреть, что модель выучила ассоциации с
    праздниками и другими событиями.
    """

    train_df = df[(df['date'] >= train_start_date) & (df['date'] < train_end_date)]
    test_df = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)]

    X_train = train_df.drop(columns=['sales', 'date', 'weight'])
    y_train = train_df['sales']
    weights_train = train_df['weight']

    X_test = test_df.drop(columns=['sales', 'date'])
    y_test = test_df['sales']

    return X_train, X_test, y_train, y_test, weights_train


train = load_data('/kaggle/input/rohlik-sales-forecasting-challenge-v2-prepared/rohlik-sales-forecasting-challenge-v2 — копия/prepared_train.csv')
X_train, X_test, y_train, y_test, weights_train = preprocess_data(train)


numeric_features = train.select_dtypes(include=['int64', 'float64']).columns.to_list()
categorical_features = train.select_dtypes(include=['object', 'category']).columns.to_list()
numeric_features.remove('sales')
numeric_features.remove('weight')

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('target_encoder', TargetEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Initialize models
lgbm_regressor = LGBMRegressor(random_state=42, num_leaves=93, max_depth=10)
gb_regressor = GradientBoostingRegressor(random_state=42)
xgb_regressor = XGBRegressor(random_state=42, verbosity=0)

# GridSearch to find the best algorithm
param_grid = [
    {'model': [lgbm_regressor]},
    {'model': [gb_regressor]},
    {'model': [xgb_regressor]}
]

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', lgbm_regressor)
])



def grid_search_optimization(X_train, y_train, weights_train, param_grid):
    """Perform GridSearch to find the best model."""
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_mean_absolute_error',
        verbose=4,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train, model__sample_weight=weights_train)
    return grid_search


grid_search = grid_search_optimization(X_train, y_train, weights_train, param_grid)
best_model = grid_search.best_estimator_
print(f"Best model: {grid_search.best_params_}")
best_model


import joblib
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

import shap


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path, parse_dates=['date'])
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Failed to load data: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocess the dataset and split it into train and test sets."""
    if not np.all(np.isfinite(df.select_dtypes(include=[np.number]))):
        print("Dataset contains infinite or NaN values. Replacing with 0.")
        df.replace([np.inf, -np.inf], 0, inplace=True)
    
    train_start_date = '2020-08-01'
    train_end_date = '2024-03-18'
    test_start_date = '2024-03-18'
    test_end_date = '2024-06-01'

    train_df = df[(df['date'] >= train_start_date) & (df['date'] < train_end_date)]
    test_df = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)]

    X_train = train_df.drop(columns=['sales', 'date', 'weight'])
    y_train = train_df['sales']
    weights_train = train_df['weight']

    X_test = test_df.drop(columns=['sales', 'date', 'weight'])
    y_test = test_df['sales']
    weights_test = test_df['weight']

    return X_train, X_test, y_train, y_test, weights_train, weights_test


train = load_data('rohlik-sales-forecasting-challenge-v2/prepared_train.csv')
X_train, X_test, y_train, y_test, weights_train, weights_test = preprocess_data(train)

categorical_columns = ['unique_id'] + list(X_train.select_dtypes("object").columns)
for col in categorical_columns:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')
categorical_feature_indices = [X_train.columns.get_loc(col) for col in categorical_columns if col in X_train.columns]

train_dataset = lgb.Dataset(X_train, label=y_train, 
                            categorical_feature=categorical_feature_indices, 
                            weight=weights_train, free_raw_data=False)
val_dataset = lgb.Dataset(X_test, label=y_test, 
                          categorical_feature=categorical_feature_indices, 
                          weight=weights_test, free_raw_data=False)


def objective_lgbm(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True), 
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0), 
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), 
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True), 
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        
        # Fixed parameters
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_threads': 30,
        'verbose': -1,
        'feature_pre_filter': False,
        'free_raw_data': False
    }

    num_boost_round = trial.suggest_int('num_boost_round', 1000, 10000)

    model = lgb.train(
        params,
        train_dataset,
        valid_sets=[val_dataset],
        num_boost_round=num_boost_round
        )

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    
    return mae



study = optuna.create_study(
    study_name="lgbm_study",
    direction="minimize",
    storage="sqlite:///lgbm_study.db",
    load_if_exists=True,
)
study.optimize(objective_lgbm, n_trials=30)


print('Best trial:')
trial = study.best_trial
print(f'  Value (MAE): {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')


best_params = study.best_params
best_params.update({
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_threads': 30,
    'verbose': -1
})
num_boost_round = best_params['num_boost_round']
best_params.pop('num_boost_round', None)

# Train final model
final_model = lgb.train(
    best_params, 
    train_dataset, 
    num_boost_round=num_boost_round,
    valid_sets=val_dataset,
    callbacks=[lgb.log_evaluation(period=200)]
)


preds = final_model.predict(X_test)
mean_absolute_error(y_test, preds)


def prepare_data_lgbm(path_to_data):
    data = pd.read_csv(path_to_data, parse_dates=['date'])
    if not np.all(np.isfinite(data.select_dtypes(include=[np.number]))):
        print("Dataset contains infinite or NaN values. Replacing with 0.")
        data.replace([np.inf, -np.inf], 0, inplace=True)

    return data

def prepare_test_prediction(path_to_test, model, name_of_predict):
    test = prepare_data_lgbm(path_to_test)
    for col in categorical_columns:
        test[col] = test[col].astype('category')
    test_predictions = model.predict(test.drop(columns=['date'])[X_train.columns])

    sub = test.copy()
    sub['sales_hat'] = test_predictions
    sub['id'] = sub['unique_id'].astype(str) + "_" + sub['date'].astype(str)
    sub[['id','sales_hat']].to_csv(name_of_predict, index=False)


prepare_test_prediction('rohlik-sales-forecasting-challenge-v2/prepared_test.csv', 
                        final_model, 
                        'lgbm-tuned-1.csv')


joblib.dump(final_model, 'lgbm-tuned-1.pkl')


shap_values = shap.TreeExplainer(final_model).shap_values(X_test.iloc[:1000])
shap.summary_plot(shap_values, X_test.iloc[:1000])


def prepare_train(train):
    if not np.all(np.isfinite(train.select_dtypes(include=[np.number]))):
        print("Dataset contains infinite or NaN values. Replacing with 0.")
        train.replace([np.inf, -np.inf], 0, inplace=True)

    for col in categorical_columns:
        train[col] = train[col].astype('category')

    return train.drop(columns=['sales', 'date', 'weight']), train['sales'], train['weight']
train_p, y_t, w_t = prepare_train(train)

train_dataset_full = lgb.Dataset(train_p, label=y_t, 
                            categorical_feature=categorical_feature_indices, 
                            weight=w_t, free_raw_data=False)
# Train final model
final_model = lgb.train(
    best_params, 
    train_dataset_full, 
    num_boost_round=num_boost_round,
    callbacks=[lgb.log_evaluation(period=200)]
)

prepare_test_prediction('rohlik-sales-forecasting-challenge-v2/prepared_test.csv', 
                        final_model, 
                        'lgbm-tuned-2.csv')
joblib.dump(final_model, 'lgbm-tuned-2.pkl')


shap_values = shap.TreeExplainer(final_model).shap_values(X_test.iloc[:1000])
shap.summary_plot(shap_values, X_test.iloc[:1000])


import joblib
import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

import shap


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path, parse_dates=['date'])
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Failed to load data: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocess the dataset and split it into train and test sets."""
    if not np.all(np.isfinite(df.select_dtypes(include=[np.number]))):
        print("Dataset contains infinite or NaN values. Replacing with 0.")
        df.replace([np.inf, -np.inf], 0, inplace=True)
    
    train_start_date = '2020-08-01'
    train_end_date = '2024-03-18'
    test_start_date = '2024-03-18'
    test_end_date = '2024-06-01'

    train_df = df[(df['date'] >= train_start_date) & (df['date'] < train_end_date)]
    test_df = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)]

    X_train = train_df.drop(columns=['sales', 'date', 'weight'])
    y_train = train_df['sales']
    weights_train = train_df['weight']

    X_test = test_df.drop(columns=['sales', 'date', 'weight'])
    y_test = test_df['sales']
    weights_test = test_df['weight']

    return X_train, X_test, y_train, y_test, weights_train, weights_test


train = load_data('rohlik-sales-forecasting-challenge-v2/prepared_train.csv')
X_train, X_test, y_train, y_test, weights_train, weights_test = preprocess_data(train)

numeric_features = train.select_dtypes(include=['int64', 'float64']).columns.to_list()
categorical_features = train.select_dtypes(include=['object', 'category']).columns.to_list()
numeric_features.remove('sales')
numeric_features.remove('weight')

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('target_encoder', TargetEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


def objective_gb(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True), 
        'n_estimators': trial.suggest_int('n_estimators', 70, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-3, 10.0, log=True), 
        
        # Fixed parameters
        'loss': 'absolute_error',
        'verbose': 0,
        'validation_fraction': 0.2,
        'random_state': 42,
        'n_iter_no_change': 100
    }

    model = GradientBoostingRegressor(**params)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    
    return mae


study = optuna.create_study(
    study_name="gb_study",
    direction="minimize",
    storage="sqlite:///gb_study.db",
    load_if_exists=True,
)
study.optimize(objective_gb, n_trials=15)




