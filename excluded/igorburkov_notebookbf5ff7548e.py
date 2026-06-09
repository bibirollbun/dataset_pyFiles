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


# Указываем пути к файлам (они были напечатаны в выводе предыдущей ячейки)
train_path = '/kaggle/input/playground-series-s5e4/train.csv'
test_path = '/kaggle/input/playground-series-s5e4/test.csv'
submission_path = '/kaggle/input/playground-series-s5e4/sample_submission.csv'

# Загружаем данные в DataFrame'ы pandas
# (Убедись, что pandas импортирован в первой ячейке: import pandas as pd)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission_df = pd.read_csv(submission_path)


# Проверим, что данные загрузились, посмотрев на первые несколько строк
print("Тренировочные данные (train_df):")
print(train_df.head()) # .head() показывает первые 5 строк
print("\nТестовые данные (test_df):")
print(test_df.head())
print("\nПример файла отправки (sample_submission_df):")
print(sample_submission_df.head())

# Посмотрим на размеры таблиц
print(f'\nРазмер train: {train_df.shape}')
print(f'Размер test: {test_df.shape}')


print("Столбцы в train_df СРАЗУ ПОСЛЕ ЗАГРУЗКИ:")
print(train_df.columns)


# --- Ячейка 3: Feature Engineering (Исправленная) ---
import pandas as pd
import numpy as np

# Убедись, что train_df и test_df загружены из Ячейки 2

print("Начинаем Feature Engineering...")

# --- 1. Агрегации ВСЕХ нужных признаков по Podcast_Name ---
print("... Агрегации по Podcast_Name")

# Список числовых признаков для агрегации
numeric_cols_to_agg = [
    'Listening_Time_minutes', # Наш таргет (считаем ТОЛЬКО на трейне)
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage', # Будет содержать NaN до fillna
    'Episode_Length_minutes',      # Будет содержать NaN до fillna
    'Number_of_Ads'                # Будет содержать NaN до fillna
]
# Функции агрегации
agg_funcs = ['mean', 'median', 'std'] # Можно добавить 'min', 'max', 'nunique' позже

# Словарь для агрегации (для таргета считаем на трейне, для остальных - тоже, но NaN обработаем)
agg_dict_podcast = {col: agg_funcs for col in numeric_cols_to_agg}

# Считаем статистики ТОЛЬКО на трейне
# Сначала обработаем таргет отдельно
podcast_target_stats = train_df.groupby('Podcast_Name')['Listening_Time_minutes'].agg(agg_funcs).reset_index()
# Переименовываем колонки для таргета
target_rename_map = {f: f"{'Listening_Time_minutes'}_{f}_by_podcast" for f in agg_funcs}
podcast_target_stats.rename(columns=target_rename_map, inplace=True)

# Теперь считаем для остальных признаков (тоже на трейне)
other_numeric_cols = [col for col in numeric_cols_to_agg if col != 'Listening_Time_minutes']
podcast_other_stats = train_df.groupby('Podcast_Name')[other_numeric_cols].agg(agg_funcs).reset_index()
# Переименовываем колонки для остальных признаков
podcast_other_stats.columns = ['Podcast_Name'] + ['_'.join(col).strip() + '_by_podcast' for col in podcast_other_stats.columns[1:]]

# Объединяем статистики
podcast_stats_combined = pd.merge(podcast_target_stats, podcast_other_stats, on='Podcast_Name', how='left')

# Добавляем все статистики к train и test
train_df = pd.merge(train_df, podcast_stats_combined, on='Podcast_Name', how='left')
test_df = pd.merge(test_df, podcast_stats_combined, on='Podcast_Name', how='left')

# --- 2. Агрегации ВСЕХ нужных признаков по Genre ---
print("... Агрегации по Genre")
# Используем тот же список признаков и функций
agg_dict_genre = {col: agg_funcs for col in numeric_cols_to_agg}

# Считаем статистики ТОЛЬКО на трейне
# Таргет отдельно
genre_target_stats = train_df.groupby('Genre')['Listening_Time_minutes'].agg(agg_funcs).reset_index()
target_rename_map_genre = {f: f"{'Listening_Time_minutes'}_{f}_by_genre" for f in agg_funcs}
genre_target_stats.rename(columns=target_rename_map_genre, inplace=True)

# Остальные признаки
genre_other_stats = train_df.groupby('Genre')[other_numeric_cols].agg(agg_funcs).reset_index()
genre_other_stats.columns = ['Genre'] + ['_'.join(col).strip() + '_by_genre' for col in genre_other_stats.columns[1:]]

# Объединяем
genre_stats_combined = pd.merge(genre_target_stats, genre_other_stats, on='Genre', how='left')

# Добавляем все статистики к train и test
train_df = pd.merge(train_df, genre_stats_combined, on='Genre', how='left')
test_df = pd.merge(test_df, genre_stats_combined, on='Genre', how='left')


# --- 3. Частотное кодирование (Frequency Encoding) ---
print("... Частотное кодирование")
# Считаем частоты ТОЛЬКО на трейне
podcast_freq = train_df['Podcast_Name'].value_counts(normalize=True)
episode_freq = train_df['Episode_Title'].value_counts(normalize=True)
genre_freq = train_df['Genre'].value_counts(normalize=True)

# Применяем частоты к train и test
train_df['podcast_freq'] = train_df['Podcast_Name'].map(podcast_freq)
test_df['podcast_freq'] = test_df['Podcast_Name'].map(podcast_freq)

train_df['episode_freq'] = train_df['Episode_Title'].map(episode_freq)
test_df['episode_freq'] = test_df['Episode_Title'].map(episode_freq)

train_df['genre_freq'] = train_df['Genre'].map(genre_freq)
test_df['genre_freq'] = test_df['Genre'].map(genre_freq)

# Заполняем возможные NaN в test (если название из теста не было в трейне) нулями
# Делаем это здесь, чтобы избежать проблем с fillna в Ячейке 4
test_df['podcast_freq'].fillna(0, inplace=True)
test_df['episode_freq'].fillna(0, inplace=True)
test_df['genre_freq'].fillna(0, inplace=True) # Вряд ли понадобится для жанров


# Добавить в Ячейку 3 (Feature Engineering)
train_df['is_long_episode'] = (train_df['Episode_Length_minutes'] > 60).astype(int)
test_df['is_long_episode'] = (test_df['Episode_Length_minutes'] > 60).astype(int)
# Важно: Сделать это ПОСЛЕ заполнения NaN в Episode_Length_minutes!
# Поэтому лучше этот код добавить в Ячейку 4 после fillna для длины.
# --- 4. Агрегации по listen_ratio (сначала считаем для train) ---
# Этот блок остается, но убедись, что Episode_Length_minutes будет заполнено *до* этого блока
# или используй .fillna() прямо здесь при расчете listen_ratio_temp
# Безопаснее перенести расчет ratio ПОСЛЕ fillna в Ячейке 4, но пока оставим так

print("... Агрегации по listen_ratio (сначала считаем для train)")
# Получим медиану длины ИЗ ОРИГИНАЛЬНОГО train_df ДО fillna
# Это важно, чтобы не использовать уже заполненные значения для расчета ratio
# Предполагаем, что train_df еще не изменен fillna
median_length_orig = train_df['Episode_Length_minutes'].median() # Рассчитываем ДО fillna

# Заполняем NaN во временной колонке для расчета listen_ratio_temp
train_df['Episode_Length_minutes_temp'] = train_df['Episode_Length_minutes'].fillna(median_length_orig)
# Добавим небольшое значение к знаменателю для безопасности
train_df['listen_ratio_temp'] = train_df['Listening_Time_minutes'] / (train_df['Episode_Length_minutes_temp'] + 1e-6)

# Считаем среднее отношение по подкасту
podcast_ratio_mean = train_df.groupby('Podcast_Name')['listen_ratio_temp'].mean()
train_df['podcast_mean_ratio'] = train_df['Podcast_Name'].map(podcast_ratio_mean)
test_df['podcast_mean_ratio'] = test_df['Podcast_Name'].map(podcast_ratio_mean)

# Считаем среднее отношение по жанру
genre_ratio_mean = train_df.groupby('Genre')['listen_ratio_temp'].mean()
train_df['genre_mean_ratio'] = train_df['Genre'].map(genre_ratio_mean)
test_df['genre_mean_ratio'] = test_df['Genre'].map(genre_ratio_mean)

# Заполняем NaN для ratio в test средним значением ratio из train
mean_train_ratio = train_df['listen_ratio_temp'].mean()
test_df['podcast_mean_ratio'].fillna(mean_train_ratio, inplace=True)
test_df['genre_mean_ratio'].fillna(mean_train_ratio, inplace=True)

# Удаляем временные столбцы
train_df = train_df.drop(columns=['listen_ratio_temp', 'Episode_Length_minutes_temp'])


# --- 5. Простые взаимодействия топ-признаков ---
# Этот блок можно оставить, но он будет использовать значения с NaN до этапа fillna
# Безопаснее перенести его в Ячейку 4 ПОСЛЕ fillna
# Пока оставим здесь для простоты, но NaN могут повлиять на результат взаимодействия
print("... Создание Interaction Features (до fillna)")
epsilon = 1e-6

train_df['host_x_guest_pop'] = train_df['Host_Popularity_percentage'] * train_df['Guest_Popularity_percentage'] # NaN * число = NaN
test_df['host_x_guest_pop'] = test_df['Host_Popularity_percentage'] * test_df['Guest_Popularity_percentage']

train_df['host_pop_x_length'] = train_df['Host_Popularity_percentage'] * train_df['Episode_Length_minutes'] # NaN * число = NaN
test_df['host_pop_x_length'] = test_df['Host_Popularity_percentage'] * test_df['Episode_Length_minutes']

# Отношения будут иметь NaN там, где Episode_Length_minutes был NaN
train_df['host_pop_div_length'] = train_df['Host_Popularity_percentage'] / (train_df['Episode_Length_minutes'] + epsilon)
test_df['host_pop_div_length'] = test_df['Host_Popularity_percentage'] / (test_df['Episode_Length_minutes'] + epsilon)

train_df['guest_pop_div_length'] = train_df['Guest_Popularity_percentage'] / (train_df['Episode_Length_minutes'] + epsilon)
test_df['guest_pop_div_length'] = test_df['Guest_Popularity_percentage'] / (test_df['Episode_Length_minutes'] + epsilon)


print("Feature Engineering завершен.")

# Посмотрим на новые колонки
print("\nНовые колонки в train_df:")
print(train_df.columns)
print(f"\nРазмер train_df после FE: {train_df.shape}")
print(f"Размер test_df после FE: {test_df.shape}")

# Проверим NaN ПЕРЕД Ячейкой 4
print("\nПропуски в train_df ПОСЛЕ Feature Engineering:")
print(train_df.isnull().sum().sort_values(ascending=False).head(20)) # Показать топ 20 с пропусками
print("\nПропуски в test_df ПОСЛЕ Feature Engineering:")
print(test_df.isnull().sum().sort_values(ascending=False).head(20)) # Показать топ 20 с пропусками


print("Пропуски в train_df:")
print(train_df.isnull().sum())

print("\nПропуски в test_df:")
print(test_df.isnull().sum())


import matplotlib.pyplot as plt
import seaborn as sns

# Гистограмма распределения времени прослушивания
plt.figure(figsize=(10, 5))
sns.histplot(train_df['Listening_Time_minutes'], bins=50, kde=True) # kde=True добавляет сглаженную кривую
plt.title('Распределение Listening_Time_minutes')
plt.xlabel('Время прослушивания (минуты)')
plt.ylabel('Частота')
plt.grid(True)
plt.show()

# Ящик с усами (Boxplot) для выявления выбросов
plt.figure(figsize=(10, 2))
sns.boxplot(x=train_df['Listening_Time_minutes'])
plt.title('Boxplot Listening_Time_minutes')
plt.xlabel('Время прослушивания (минуты)')
plt.show()


# Посмотрим на количество уникальных значений в некоторых категориальных столбцах
print(f"Уникальных Podcast_Name: {train_df['Podcast_Name'].nunique()}")
print(f"Уникальных Episode_Title: {train_df['Episode_Title'].nunique()}") # Скорее всего ОЧЕНЬ много
print(f"Уникальных Genre: {train_df['Genre'].nunique()}")
print(f"Уникальных Publication_Day: {train_df['Publication_Day'].nunique()}")
print(f"Уникальных Publication_Time: {train_df['Publication_Time'].nunique()}")
print(f"Уникальных Episode_Sentiment: {train_df['Episode_Sentiment'].nunique()}")

# Посмотрим на сами уникальные значения для низкокардинальных признаков
print("\nУникальные Genre:")
print(train_df['Genre'].unique())
print("\nУникальные Publication_Day:")
print(train_df['Publication_Day'].unique())
print("\nУникальные Publication_Time:")
print(train_df['Publication_Time'].unique())
print("\nУникальные Episode_Sentiment:")
print(train_df['Episode_Sentiment'].unique())


# --- 1. Заполнение пропусков ---

# Заполняем Episode_Length_minutes медианой
median_length = train_df['Episode_Length_minutes'].median()
train_df['Episode_Length_minutes'].fillna(median_length, inplace=True)
test_df['Episode_Length_minutes'].fillna(median_length, inplace=True) # Используем медиану из train!

# Заполняем Guest_Popularity_percentage нулем
train_df['Guest_Popularity_percentage'].fillna(0, inplace=True)
test_df['Guest_Popularity_percentage'].fillna(0, inplace=True)

# Заполняем Number_of_Ads медианой (или модой)
median_ads = train_df['Number_of_Ads'].median()
train_df['Number_of_Ads'].fillna(median_ads, inplace=True)
# В test нет пропусков в Number_of_Ads, но на всякий случай можно добавить:
# test_df['Number_of_Ads'].fillna(median_ads, inplace=True)

# Проверим, остались ли пропуски в этих колонках
print("Пропуски в train_df после заполнения:")
print(train_df[['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']].isnull().sum())
print("\nПропуски в test_df после заполнения:")
print(test_df[['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']].isnull().sum())


# --- 2. Кодирование низкокардинальных категорий ---

# Список столбцов для One-Hot Encoding
low_card_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# Применяем One-Hot Encoding
train_df = pd.get_dummies(train_df, columns=low_card_cols, drop_first=True) # drop_first=True убирает одну категорию из каждой группы, чтобы избежать мультиколлинеарности
test_df = pd.get_dummies(test_df, columns=low_card_cols, drop_first=True)

# Посмотрим на результат
print("\nПервые строки train_df после OHE:")
print(train_df.head())

print("\nНовые столбцы в train_df:")
print(train_df.columns)

# Убедимся, что в test те же колонки (могут быть расхождения, если в тесте не все категории из трейна есть)
# Простой способ синхронизировать колонки:
train_labels = train_df['Listening_Time_minutes']
train_ids = train_df['id']
test_ids = test_df['id']

# Удалим таргет и ID перед выравниванием колонок и обработкой остальных
train_df = train_df.drop(columns=['Listening_Time_minutes', 'id'])
test_df = test_df.drop(columns=['id'])

# Выравниваем колонки (важно для OHE, если категории в train/test не совпадают)
common_cols = list(set(train_df.columns) & set(test_df.columns))
train_df = train_df[common_cols]
test_df = test_df[common_cols]


print(f'\nРазмер train после OHE и выравнивания: {train_df.shape}')
print(f'Размер test после OHE и выравнивания: {test_df.shape}')

# !!! ДОБАВЬ ЭТИ СТРОКИ В КОНЕЦ ЯЧЕЙКИ 4 !!!
# Присваиваем финальные данные переменным X, y, X_test

# ... (код fillna, get_dummies) ...



# !!! ВАЖНО: УДАЛЯЕМ ТЕКСТОВЫЕ КОЛОНКИ ЗДЕСЬ !!!
text_cols_to_drop = ['Podcast_Name', 'Episode_Title']
train_df = train_df.drop(columns=text_cols_to_drop, errors='ignore')
test_df = test_df.drop(columns=text_cols_to_drop, errors='ignore')
# !!! КОНЕЦ ВАЖНОГО БЛОКА !!!

# Выравниваем колонки (важно для OHE, если категории в train/test не совпадают)
common_cols = list(set(train_df.columns) & set(test_df.columns))
train_df = train_df[common_cols]
test_df = test_df[common_cols]


print(f'\nРазмер train после OHE и выравнивания: {train_df.shape}')
print(f'Размер test после OHE и выравнивания: {test_df.shape}')

# ... (код fillna, get_dummies, удаление id/таргета/текстовых, выравнивание common_cols) ...

print(f'\nРазмер train после OHE и выравнивания: {train_df.shape}')
print(f'Размер test после OHE и выравнивания: {test_df.shape}')

# !!! УДАЛЯЕМ ИЗБЫТОЧНЫЙ ПРИЗНАК !!!
cols_to_remove_redundant = ['genre_mean_ratio']
common_cols = [col for col in common_cols if col not in cols_to_remove_redundant]
print(f"Удалены избыточные колонки: {cols_to_remove_redundant}")
print(f"Новое количество общих колонок: {len(common_cols)}")
# !!! КОНЕЦ УДАЛЕНИЯ !!!



# ... (финальная проверка) ...

# Присваиваем финальные данные переменным X, y, X_test
X = train_df[common_cols].copy() # Важно использовать common_cols
y = train_labels.copy()         # Мы сохранили ее ранее
X_test = test_df[common_cols].copy()

# Опциональная проверка, что все создано
print("\n--- Финальная проверка в конце Ячейки 4 ---")
print(f"Создана X с размером: {X.shape}")
print(f"Создана y с размером: {y.shape}")
print(f"Создана X_test с размером: {X_test.shape}")
print("Типы данных в X:")
print(X.dtypes.value_counts()) # Проверим типы данных
print("--- Конец проверки ---")


# --- Обучение XGBoost ---

# Убедись, что нужные библиотеки импортированы
import xgboost as xgb # <--- Импортируем XGBoost
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import gc

# Убедись, что X, y, X_test, test_ids определены из предыдущих шагов

# Параметры XGBoost (стартовые значения, можно потом тюнить)
# Многие параметры похожи на LightGBM, но называются чуть иначе
xgb_params = {
    'objective': 'reg:squarederror', # Цель - минимизация MSE (стандарт для регрессии в XGBoost)
    'eval_metric': 'rmse',          # Метрика для оценки и early stopping
    'eta': 0.05,                    # learning_rate
    'max_depth': 7,                 # Максимальная глубина дерева (обычно чуть меньше, чем у LGBM)
    'subsample': 0.8,               # Аналог subsample в LGBM
    'colsample_bytree': 0.8,        # Аналог colsample_bytree в LGBM
    'min_child_weight': 1,          # Минимальная сумма весов экземпляров в листе
    'gamma': 0.0,                   # Минимальное снижение потерь для дальнейшего разбиения (регуляризация)
    'lambda': 1.0,                  # L2 регуляризация (reg_lambda)
    'alpha': 0.1,                   # L1 регуляризация (reg_alpha)
    'seed': 42,
    'nthread': -1,                  # Использовать все потоки CPU
    # 'tree_method': 'hist',        # Используем 'hist' для ускорения на больших данных
    # 'tree_method': 'gpu_hist',    # Раскомментируй для GPU (если XGBoost собран с поддержкой GPU)
}

# Количество раундов бустинга (аналог n_estimators)
N_ESTIMATORS_XGB = 10000
EARLY_STOPPING_ROUNDS_XGB = 100

# --- Кросс-валидация ---
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42) # Те же фолды

# Списки для хранения результатов XGBoost
oof_preds_xgb = np.zeros(X.shape[0])
sub_preds_xgb = np.zeros(X_test.shape[0])
scores_xgb = []
feature_importance_df_xgb = pd.DataFrame() # Важность признаков XGBoost

print("\n--- Запускаем обучение XGBoost ---")
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    print(f"--- XGBoost Фолд {n_fold + 1}/{NFOLDS} ---")

    # Используем API XGBoost (не sklearn wrapper для большей гибкости с early stopping)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test) # Для предсказаний на тесте

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=N_ESTIMATORS_XGB,
        evals=watchlist,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS_XGB,
        verbose_eval=False # Не выводить RMSE на каждой итерации
    )

    # Предсказания
    # Важно: model.best_iteration дает оптимальное количество деревьев
    oof_preds_xgb[valid_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))
    sub_preds_xgb += model.predict(dtest, iteration_range=(0, model.best_iteration)) / folds.n_splits

    # Оценка
    fold_rmse = mean_squared_error(y_valid, oof_preds_xgb[valid_idx], squared=False)
    scores_xgb.append(fold_rmse)
    print(f"Фолд {n_fold + 1} RMSE: {fold_rmse:.5f}")

    # Важность признаков (может отличаться от LGBM/CatBoost)
    fold_importance_df = pd.DataFrame()
    # Получаем важность (разные типы: 'weight', 'gain', 'cover')
    importance = model.get_score(importance_type='gain') # 'gain' часто наиболее информативен
    fold_importance_df["feature"] = importance.keys()
    fold_importance_df["importance"] = importance.values()
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df_xgb = pd.concat([feature_importance_df_xgb, fold_importance_df], axis=0)


    # Очистка памяти (dmatrix не удаляем явно)
    del X_train, y_train, X_valid, y_valid, model, dtrain, dvalid, dtest, watchlist
    gc.collect()

# --- Итоговая Оценка XGBoost ---
mean_rmse_xgb = np.mean(scores_xgb)
print(f"\nСредний RMSE (XGBoost): {mean_rmse_xgb:.5f}")
oof_rmse_xgb = mean_squared_error(y, oof_preds_xgb, squared=False)
print(f"Общий OOF RMSE (XGBoost): {oof_rmse_xgb:.5f}")


# --- Создание файла для отправки XGBoost (С ОБРАТНЫМ ПРЕОБРАЗОВАНИЕМ, если y был логарифмирован!) ---
print("\n--- Подготовка файла submission_xgb_final.csv ---")

final_predictions_xgb = sub_preds_xgb # Предсказания XGBoost

# !!! ВСТАВЬ СЮДА КОД ДЛЯ ОБРАТНОГО ПРЕОБРАЗОВАНИЯ И КЛИППИНГА, ЕСЛИ y БЫЛ ЛОГАРИФМИРОВАН !!!
# Например:
# final_predictions_xgb = np.expm1(final_predictions_xgb)
# final_predictions_xgb[final_predictions_xgb < 0] = 0
# (код для клиппинга по длине эпизода)
# !!! КОНЕЦ БЛОКА ПРЕОБРАЗОВАНИЯ !!!


submission_df_xgb = pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': final_predictions_xgb})
submission_df_xgb.to_csv('submission_xgb_final.csv', index=False)
print("\nФайл submission_xgb_final.csv готов!")

# --- (Опционально) Блендинг трех моделей ---
# Убедись, что предсказания LGBM и CatBoost доступны (oof_preds_final, sub_preds_final, oof_preds_cb_final, sub_preds_cb_final)
# или загрузи их из .npy файлов
if ('oof_preds_final' in locals() and 'sub_preds_final' in locals() and
    'oof_preds_cb_final' in locals() and 'sub_preds_cb_final' in locals()):

    print("\n--- Блендинг LightGBM + CatBoost + XGBoost ---")
    # Простое среднее
    blend_oof_3 = (oof_preds_final + oof_preds_cb_final + oof_preds_xgb) / 3
    blend_sub_3 = (sub_preds_final + sub_preds_cb_final + sub_preds_xgb) / 3

    blend_rmse_3 = mean_squared_error(y, blend_oof_3, squared=False)
    print(f"OOF RMSE (Blend LGBM+CB+XGB): {blend_rmse_3:.5f}")

    # !!! ПРИМЕНИ ОБРАТНОЕ ПРЕОБРАЗОВАНИЕ И КЛИППИНГ К blend_sub_3 ПЕРЕД СОХРАНЕНИЕМ !!!
    # blend_sub_3_orig_scale = np.expm1(blend_sub_3)
    # blend_sub_3_orig_scale[blend_sub_3_orig_scale < 0] = 0
    # (клиппинг по длине)
    # final_blend_predictions = blend_sub_3_orig_scale
    final_blend_predictions = blend_sub_3 # Используй это, если y НЕ был логарифмирован

    submission_df_blend_3 = pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': final_blend_predictions})
    submission_df_blend_3.to_csv('submission_blend_3_models.csv', index=False)
    print("\nФайл submission_blend_3_models.csv готов!")
else:
    print("\nНе найдены предсказания от LGBM или CatBoost. Блендинг трех моделей не выполнен.")


# --- Optuna + Финальное обучение XGBoost ---

import optuna
import xgboost as xgb # <--- Используем XGBoost
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import gc

# --- Убедись, что X, y, X_test, test_ids определены ---

# --- 1. Определение Целевой Функции для Optuna (для XGBoost) ---

def objective_xgboost(trial, X, y): # Новая функция для XGBoost
    # --- Определяем пространство поиска гиперпараметров XGBoost ---
    xgb_params = {
        'objective': 'reg:squarederror', # Оставляем
        'eval_metric': 'rmse',          # Оставляем
        'seed': 42,
        'nthread': -1,
        'tree_method': 'hist',        # Можно оставить hist для скорости
        # 'tree_method': 'gpu_hist',    # Раскомментируй для GPU

        # --- Параметры, которые Optuna будет подбирать ---
        'eta': trial.suggest_float('eta', 0.01, 0.1, log=True), # learning_rate
        'max_depth': trial.suggest_int('max_depth', 3, 10), # Глубина дерева
        'subsample': trial.suggest_float('subsample', 0.5, 1.0), # Доля данных
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), # Доля признаков
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10), # Мин. вес в листе
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True), # Регуляризация (минимальное снижение потерь)
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True), # L2 регуляризация
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True), # L1 регуляризация
    }

    N_ESTIMATORS_XGB = 10000
    EARLY_STOPPING_ROUNDS_XGB = 100

    # --- Кросс-валидация внутри objective ---
    NFOLDS = 5
    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    oof_preds = np.zeros(X.shape[0])
    scores = []

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        # Используем API XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

        try: # Добавим try-except на случай невалидных параметров
            model = xgb.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=N_ESTIMATORS_XGB,
                evals=watchlist,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS_XGB,
                verbose_eval=False
            )
            oof_preds[valid_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))
            fold_rmse = mean_squared_error(y_valid, oof_preds[valid_idx], squared=False)
            scores.append(fold_rmse)
        except Exception as e:
            print(f"Ошибка в Trial {trial.number}, Fold {n_fold+1}: {e}")
            # Если ошибка на одном фолде, можно присвоить высокий RMSE или проигнорировать фолд
            # Проще всего - вернуть высокий RMSE для всей попытки
            return float('inf') # Плохой результат для этой попытки

        del X_train, y_train, X_valid, y_valid, model, dtrain, dvalid, watchlist
        gc.collect()

    mean_rmse = np.mean(scores)
    print(f"Попытка {trial.number}: Средний RMSE = {mean_rmse:.5f}")

    if np.isnan(mean_rmse):
        return float('inf')

    return mean_rmse

# --- 2. Запуск Оптимизации Optuna для XGBoost ---
N_TRIALS_XGB = 50 # Задаем количество попыток для XGBoost (начни с 30-50)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_xgb = optuna.create_study(direction='minimize') # Новое исследование

print("--- Проверка перед study_xgb.optimize ---")
print(f"Существует ли X? {'X' in locals() or 'X' in globals()}")
if 'X' in locals() or 'X' in globals(): print(f"Размер X: {X.shape}")
print(f"Существует ли y? {'y' in locals() or 'y' in globals()}")
if 'y' in locals() or 'y' in globals(): print(f"Размер y: {y.shape}")
print("--- Конец проверки ---")

# Используем lambda для передачи X и y в objective_xgboost
study_xgb.optimize(lambda trial: objective_xgboost(trial, X, y), n_trials=N_TRIALS_XGB)

# --- 3. Вывод Лучших Результатов для XGBoost ---
print("\nОптимизация XGBoost завершена!")
print(f"Количество завершенных попыток: {len(study_xgb.trials)}")
print(f"Лучшая попытка (XGBoost):")
best_trial_xgb = study_xgb.best_trial
print(f"  Значение (минимальный RMSE): {best_trial_xgb.value:.5f}")
print(f"  Лучшие гиперпараметры (XGBoost):")
# Сохраняем лучшие параметры XGBoost
best_params_xgb = best_trial_xgb.params
# Добавляем обратно фиксированные параметры
best_params_xgb['objective'] = 'reg:squarederror'
best_params_xgb['eval_metric'] = 'rmse'
best_params_xgb['seed'] = 42
best_params_xgb['nthread'] = -1
# best_params_xgb['tree_method'] = 'hist' # или 'gpu_hist'

for key, value in best_params_xgb.items(): # Печатаем все параметры, включая добавленные
    print(f"    {key}: {value}")


# --- 4. Финальное обучение XGBoost с лучшими параметрами ---
print("\n--- Запускаем финальное обучение XGBoost с лучшими параметрами ---")

oof_preds_xgb_final = np.zeros(X.shape[0])
sub_preds_xgb_final = np.zeros(X_test.shape[0])
final_scores_xgb = []
feature_importance_df_xgb_final = pd.DataFrame()

NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    print(f"--- Финальный Фолд XGBoost {n_fold + 1}/{NFOLDS} ---")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test)
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

    # Используем лучшие параметры!
    model = xgb.train(
        params=best_params_xgb,
        dtrain=dtrain,
        num_boost_round=10000, # Используем большое число и early stopping
        evals=watchlist,
        early_stopping_rounds=100,
        verbose_eval=False
    )

    oof_preds_xgb_final[valid_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))
    sub_preds_xgb_final += model.predict(dtest, iteration_range=(0, model.best_iteration)) / folds.n_splits

    fold_rmse = mean_squared_error(y_valid, oof_preds_xgb_final[valid_idx], squared=False)
    final_scores_xgb.append(fold_rmse)
    print(f"Фолд {n_fold + 1} RMSE: {fold_rmse:.5f}")

    # Важность признаков (опционально)
    # ... (код для feature importance, если нужен) ...

    del X_train, y_train, X_valid, y_valid, model, dtrain, dvalid, dtest, watchlist
    gc.collect()

mean_rmse_xgb_final = np.mean(final_scores_xgb)
print(f"\nСредний RMSE (финальная модель XGBoost): {mean_rmse_xgb_final:.5f}")
oof_rmse_xgb_final = mean_squared_error(y, oof_preds_xgb_final, squared=False)
print(f"Общий OOF RMSE (финальная модель XGBoost): {oof_rmse_xgb_final:.5f}")

# --- 5. Создание файла для отправки (XGBoost Tuned) ---
# !!! НЕ ЗАБУДЬ np.expm1() И КЛИППИНГ, ЕСЛИ y БЫЛ ЛОГАРИФМИРОВАН !!!
final_predictions_xgb_tuned = sub_preds_xgb_final # Замени, если нужно преобразование

submission_df_xgb_tuned = pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': final_predictions_xgb_tuned})
submission_df_xgb_tuned.to_csv('submission_xgb_tuned.csv', index=False)
print("\nФайл submission_xgb_tuned.csv готов для отправки!")

# --- Блендинг (если есть предсказания LGBM и CB) ---
# ... (код для блендинга трех моделей, используя _final предсказания) ...


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Убедись, что датафрейм с признаками X существует ---
# X должен содержать финальный набор числовых признаков (71 столбец)
# !!! Проверка перед corr() !!!
print("--- Диагностика корреляции genre_mean_ratio и Genre_Lifestyle ---")

# 1.1. Проверяем наличие и типы столбцов
if 'genre_mean_ratio' in X.columns and 'Genre_Lifestyle' in X.columns:
    print("Столбцы существуют.")
    print(f"Тип genre_mean_ratio: {X['genre_mean_ratio'].dtype}")
    print(f"Тип Genre_Lifestyle: {X['Genre_Lifestyle'].dtype}") # Ожидаем int или bool

    # 1.2. Считаем корреляцию напрямую
    specific_corr = X[['genre_mean_ratio', 'Genre_Lifestyle']].corr().iloc[0, 1]
    print(f"\nПрямая корреляция: {specific_corr:.4f}")

    # 1.3. Анализ уникальных значений
    print(f"\nКоличество уникальных genre_mean_ratio: {X['genre_mean_ratio'].nunique()}")
    print("Топ 10 значений genre_mean_ratio:")
    print(X['genre_mean_ratio'].value_counts().head(10))

    print(f"\nРаспределение Genre_Lifestyle:")
    print(X['Genre_Lifestyle'].value_counts())

    # 1.4. Группировка для проверки связи
    print("\nЗначения genre_mean_ratio сгруппированные по Genre_Lifestyle:")
    print(X.groupby('Genre_Lifestyle')['genre_mean_ratio'].agg(['min', 'max', 'mean', 'nunique']))

else:
    print("Ошибка: Один или оба столбца ('genre_mean_ratio', 'Genre_Lifestyle') отсутствуют в X.")

print("--- Конец диагностики ---")

if 'X' in locals() or 'X' in globals():
    print(f"Анализируем корреляции для X с размером: {X.shape}")
    
    
    corr_matrix = X.corr()

    # --- 2. Визуализация матрицы корреляций (Тепловая карта) ---
    plt.figure(figsize=(20, 18)) # Размер можно подстроить, для 71 признака нужно много места
    sns.heatmap(corr_matrix,
                cmap='coolwarm', # Цветовая схема (красный - полож., синий - отриц.)
                annot=False,     # Не выводить значения в ячейках (слишком много для 71 признака)
                fmt=".2f",       # Формат чисел (если бы annot=True)
                linewidths=.5,
                cbar=True)       # Показать цветовую шкалу
    plt.title('Матрица корреляций признаков', fontsize=16)
    # plt.xticks(rotation=90) # Можно повернуть подписи оси X, если они накладываются
    # plt.yticks(rotation=0)  # Подписи оси Y
    plt.tight_layout() # Попробовать уместить все
    plt.show()

    # --- 3. Поиск и вывод сильно коррелирующих пар ---
    # Часто удобнее смотреть не на всю матрицу, а на список пар с высокой корреляцией

    # Порог для высокой корреляции (абсолютное значение)
    threshold = 0.90 # Можно изменить (например, 0.85 или 0.95)

    # Создаем маску для верхнего треугольника матрицы (чтобы избежать дублирования пар A-B и B-A)
    upper_tri_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_tri = corr_matrix.where(upper_tri_mask)

    # Находим пары с абсолютной корреляцией выше порога
    high_corr_pairs = upper_tri[abs(upper_tri) > threshold].stack().reset_index()
    high_corr_pairs.columns = ['Признак 1', 'Признак 2', 'Корреляция']

    if not high_corr_pairs.empty:
        print(f"\nПары признаков с абсолютной корреляцией > {threshold}:")
        # Сортируем по убыванию абсолютного значения корреляции
        high_corr_pairs = high_corr_pairs.sort_values(by='Корреляция', key=abs, ascending=False)
        print(high_corr_pairs.to_string()) # to_string(), чтобы увидеть все строки, если их много
    else:
        print(f"\nНе найдено пар признаков с абсолютной корреляцией > {threshold}.")

else:
    print("Ошибка: Датафрейм с признаками 'X' не найден в памяти.")


# --- Обучение CatBoost ---

# Убедись, что нужные библиотеки импортированы
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import gc

# Убедись, что X, y, X_test, test_ids определены из предыдущих шагов

# Параметры CatBoost (стартовые значения, можно потом тюнить с Optuna)
cb_params = {
    'loss_function': 'RMSE',       # Функция потерь и метрика по умолчанию
    'iterations': 10000,           # Большое число для early stopping
    'learning_rate': 0.05,        # Типичное стартовое значение
    'depth': 8,                   # Глубина деревьев (6-10 обычно хорошо)
    'l2_leaf_reg': 3,             # L2 регуляризация
    'random_seed': 42,
    'verbose': 0,                 # Не выводить лог обучения в цикле
    'task_type': 'GPU',           # Раскомментируй, если хочешь использовать GPU (и он включен)
    'early_stopping_rounds': 100  # Для ранней остановки в .fit()
}

# --- Кросс-валидация ---
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42) # Те же фолды, что и для LGBM

# Списки для хранения результатов CatBoost
oof_preds_cb = np.zeros(X.shape[0])
sub_preds_cb = np.zeros(X_test.shape[0])
scores_cb = []
feature_importance_df_cb = pd.DataFrame()

print("\n--- Запускаем обучение CatBoost ---")
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    print(f"--- CatBoost Фолд {n_fold + 1}/{NFOLDS} ---")

    # Инициализируем модель (early_stopping_rounds передается в fit)
    model = cb.CatBoostRegressor(
        loss_function=cb_params['loss_function'],
        iterations=cb_params['iterations'],
        learning_rate=cb_params['learning_rate'],
        depth=cb_params['depth'],
        l2_leaf_reg=cb_params['l2_leaf_reg'],
        random_seed=cb_params['random_seed'],
        verbose=cb_params['verbose'],
        task_type=cb_params['task_type'] # Раскомментируй для GPU
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=cb_params['early_stopping_rounds'],
        verbose=0 # Подавляем вывод внутри цикла fit
    )

    # Предсказания
    oof_preds_cb[valid_idx] = model.predict(X_valid)
    sub_preds_cb += model.predict(X_test) / folds.n_splits

    # Оценка
    fold_rmse = mean_squared_error(y_valid, oof_preds_cb[valid_idx], squared=False)
    scores_cb.append(fold_rmse)
    print(f"Фолд {n_fold + 1} RMSE: {fold_rmse:.5f}")

    # Важность признаков
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = X.columns
    fold_importance_df["importance"] = model.get_feature_importance()
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df_cb = pd.concat([feature_importance_df_cb, fold_importance_df], axis=0)

    # Очистка памяти
    del X_train, y_train, X_valid, y_valid, model
    gc.collect()

# --- Итоговая Оценка CatBoost ---
mean_rmse_cb = np.mean(scores_cb)
print(f"\nСредний RMSE (CatBoost): {mean_rmse_cb:.5f}")
oof_rmse_cb = mean_squared_error(y, oof_preds_cb, squared=False)
print(f"Общий OOF RMSE (CatBoost): {oof_rmse_cb:.5f}")


# --- (Опционально) Создание файла для отправки CatBoost ---
submission_df_cb = pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': sub_preds_cb})
submission_df_cb.to_csv('submission_catboost.csv', index=False)
print("\nФайл submission_catboost.csv готов!")

# --- (Опционально) Простое Усреднение (Блендинг) ---
# Если у тебя есть финальные предсказания от LightGBM (назовем их oof_preds_lgb и sub_preds_lgb)
# Предположим, они были сохранены в переменных oof_preds_final и sub_preds_final из предыдущего шага
# Проверим их существование
if 'oof_preds_final' in locals() and 'sub_preds_final' in locals():
    print("\n--- Блендинг LightGBM + CatBoost ---")
    blend_oof_preds = (oof_preds_final + oof_preds_cb) / 2
    blend_oof_rmse = mean_squared_error(y, blend_oof_preds, squared=False)
    print(f"OOF RMSE (Blend LGBM+CB): {blend_oof_rmse:.5f}")

    # Создаем файл для бленда
    blend_sub_preds = (sub_preds_final + sub_preds_cb) / 2
    submission_df_blend = pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': blend_sub_preds})
    submission_df_blend.to_csv('submission_blend.csv', index=False)
    print("\nФайл submission_blend.csv готов!")
else:
    print("\nПеременные oof_preds_final/sub_preds_final от LightGBM не найдены. Блендинг не выполнен.")



# --- Ячейка 5: Optuna + Финальное обучение CatBoost ---

import optuna
import catboost as cb # <--- ИЗМЕНЕНИЕ: Импортируем CatBoost
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import gc

# --- Убедись, что X, y, X_test, test_ids определены ---

# --- 1. Определение Целевой Функции для Optuna (для CatBoost) ---

def objective_catboost(trial, X, y): # Дадим функции другое имя для ясности
    # --- Определяем пространство поиска гиперпараметров CatBoost ---
    cb_params = {
        'loss_function': 'RMSE',      # Оставляем RMSE
        'iterations': 10000,          # Большое для early stopping
        'random_seed': 42,
        'verbose': 0,                 # Подавляем вывод внутри fit
        'task_type': 'GPU',           # Раскомментируй для GPU
        'early_stopping_rounds': 100, # Передаем в fit

        # --- Параметры, которые Optuna будет подбирать ---
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10), # Типичный диапазон для CB
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True), # L2 регуляризация
        'subsample': trial.suggest_float('subsample', 0.5, 1.0), # Доля объектов
        # 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0), # Доля признаков (можно добавить)
        # 'border_count': trial.suggest_int('border_count', 32, 255), # Количество бинов (можно добавить)
        # 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100), # Мин. объектов в листе (можно добавить)
    }

    # --- Кросс-валидация внутри objective ---
    NFOLDS = 5
    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    oof_preds = np.zeros(X.shape[0])
    scores = []

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        # !!! ИЗМЕНЕНИЕ: Создаем CatBoostRegressor !!!
        model = cb.CatBoostRegressor(
            loss_function=cb_params['loss_function'],
            iterations=cb_params['iterations'],
            learning_rate=cb_params['learning_rate'],
            depth=cb_params['depth'],
            l2_leaf_reg=cb_params['l2_leaf_reg'],
            random_seed=cb_params['random_seed'],
            verbose=cb_params['verbose'],
            task_type=cb_params.get('task_type', 'CPU') # Используем .get для опционального параметра
        )

        # !!! ИЗМЕНЕНИЕ: Обучаем CatBoostRegressor !!!
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=cb_params['early_stopping_rounds'], # Передаем сюда
            verbose=0 # Подавляем вывод внутри fit
        )

        oof_preds[valid_idx] = model.predict(X_valid)
        fold_rmse = mean_squared_error(y_valid, oof_preds[valid_idx], squared=False)
        scores.append(fold_rmse)

        del X_train, y_train, X_valid, y_valid, model
        gc.collect()

    mean_rmse = np.mean(scores)
    print(f"Попытка {trial.number}: Средний RMSE = {mean_rmse:.5f}")

    if np.isnan(mean_rmse):
        return float('inf')

    return mean_rmse

# --- 2. Запуск Оптимизации Optuna для CatBoost ---
N_TRIALS_CB = 50 # Задаем количество попыток для CatBoost (можно начать с 30-50)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_cb = optuna.create_study(direction='minimize') # Создаем новое исследование

print("--- Проверка перед study_cb.optimize ---") # Проверка для нового запуска
print(f"Существует ли X? {'X' in locals() or 'X' in globals()}")
if 'X' in locals() or 'X' in globals(): print(f"Размер X: {X.shape}")
print(f"Существует ли y? {'y' in locals() or 'y' in globals()}")
if 'y' in locals() or 'y' in globals(): print(f"Размер y: {y.shape}")
print("--- Конец проверки ---")

# Используем lambda для передачи X и y в objective_catboost
study_cb.optimize(lambda trial: objective_catboost(trial, X, y), n_trials=N_TRIALS_CB)

# --- 3. Вывод Лучших Результатов для CatBoost ---
print("\nОптимизация CatBoost завершена!")
print(f"Количество завершенных попыток: {len(study_cb.trials)}")
print(f"Лучшая попытка (CatBoost):")
best_trial_cb = study_cb.best_trial
print(f"  Значение (минимальный RMSE): {best_trial_cb.value:.5f}")
print(f"  Лучшие гиперпараметры (CatBoost):")
for key, value in best_trial_cb.params.items():
    print(f"    {key}: {value}")

# Сохраняем лучшие параметры CatBoost
best_params_cb = best_trial_cb.params
# Добавляем обратно фиксированные параметры
best_params_cb['loss_function'] = 'RMSE'
best_params_cb['iterations'] = 10000
best_params_cb['random_seed'] = 42
best_params_cb['verbose'] = 0
best_params_cb['early_stopping_rounds'] = 100 # Передаем в fit

# --- 4. Финальное обучение CatBoost с лучшими параметрами ---
print("\n--- Запускаем финальное обучение CatBoost с лучшими параметрами ---")

oof_preds_cb_final = np.zeros(X.shape[0])
sub_preds_cb_final = np.zeros(X_test.shape[0])
final_scores_cb = []
feature_importance_df_cb_final = pd.DataFrame()

NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    print(f"--- Финальный Фолд CatBoost {n_fold + 1}/{NFOLDS} ---")
    # !!! ИЗМЕНЕНИЕ: Используем лучшие параметры CatBoost !!!
    model = cb.CatBoostRegressor(**best_params_cb)
    # Удаляем early_stopping_rounds из конструктора, передадим в fit
    if 'early_stopping_rounds' in best_params_cb:
        del best_params_cb['early_stopping_rounds']

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=100, # Передаем здесь
        verbose=0
    )

    oof_preds_cb_final[valid_idx] = model.predict(X_valid)
    sub_preds_cb_final += model.predict(X_test) / folds.n_splits

    fold_rmse = mean_squared_error(y_valid, oof_preds_cb_final[valid_idx], squared=False)
    final_scores_cb.append(fold_rmse)
    print(f"Фолд {n_fold + 1} RMSE: {fold_rmse:.5f}")

    # Важность признаков (опционально)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = X.columns
    fold_importance_df["importance"] = model.get_feature_importance()
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df_cb_final = pd.concat([feature_importance_df_cb_final, fold_importance_df], axis=0)

    del X_train, y_train, X_valid, y_valid, model
    gc.collect()

mean_rmse_cb_final = np.mean(final_scores_cb)
print(f"\nСредний RMSE (финальная модель CatBoost): {mean_rmse_cb_final:.5f}")
oof_rmse_cb_final = mean_squared_error(y, oof_preds_cb_final, squared=False)
print(f"Общий OOF RMSE (финальная модель CatBoost): {oof_rmse_cb_final:.5f}")

# --- 5. Создание файла для отправки (CatBoost Tuned) ---
submission_df_cb_tuned = pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': sub_preds_cb_final})
submission_df_cb_tuned.to_csv('submission_catboost_tuned.csv', index=False)
print("\nФайл submission_catboost_tuned.csv готов для отправки!")

# --- (Опционально) Блендинг с лучшим LGBM ---
# Убедись, что предсказания лучшего LGBM (oof_preds_final, sub_preds_final) доступны в памяти
# или загрузи их из .npy файлов, если сохранял
if 'oof_preds_final' in locals() and 'sub_preds_final' in locals():
    print("\n--- Блендинг Tuned LightGBM + Tuned CatBoost ---")
    blend_oof_preds_tuned = (oof_preds_final + oof_preds_cb_final) / 2
    blend_oof_rmse_tuned = mean_squared_error(y, blend_oof_preds_tuned, squared=False)
    print(f"OOF RMSE (Blend Tuned LGBM+CB): {blend_oof_rmse_tuned:.5f}")

    blend_sub_preds_tuned = (sub_preds_final + sub_preds_cb_final) / 2
    submission_df_blend_tuned = pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': blend_sub_preds_tuned})
    submission_df_blend_tuned.to_csv('submission_blend_tuned.csv', index=False)
    print("\nФайл submission_blend_tuned.csv готов!")
else:
    print("\nПеременные oof_preds_final/sub_preds_final от LightGBM не найдены. Блендинг не выполнен.")


import matplotlib.pyplot as plt
import seaborn as sns

# Группируем по признакам и считаем среднюю важность
mean_importance = feature_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)

# Визуализируем топ N признаков
plt.figure(figsize=(10, 8)) # Можно подстроить размер
sns.barplot(x=mean_importance.values, y=mean_importance.index)
plt.title('Средняя важность признаков по фолдам (LightGBM)')
plt.xlabel('Важность')
plt.ylabel('Признак')
plt.grid(axis='x')
plt.show()

# Посмотрим на топ-20 признаков
print("\nТоп 30 признаков по средней важности:")
print(mean_importance.head(30))

