import pandas as pd
import numpy as np
# import lightgbm as lgb # Удалено
# import xgboost as xgb # Удалено
from catboost import CatBoostRegressor # Оставлено
from sklearn.model_selection import KFold # Оставлено
# from sklearn.preprocessing import LabelEncoder # Не нужен при OHE или нативной обработке CatBoost
from sklearn.metrics import mean_squared_error # Оставлено
import gc # Оставлено
import warnings # Оставлено

warnings.filterwarnings('ignore') # Подавление некоторых предупреждений

# --- Конфигурация ---

DATA_PATH = '/kaggle/input/playground-series-s5e4/' # Укажите путь к данным
N_FOLDS = 5 # Количество фолдов для кросс-валидации
RANDOM_SEED = 42 # Для воспроизводимости результатов

# --- Загрузка данных ---

print("Загрузка данных...")
try:
    train_df = pd.read_csv(DATA_PATH + 'train.csv')
    test_df = pd.read_csv(DATA_PATH + 'test.csv')
    sample_submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')
except FileNotFoundError as e:
    print(f"Ошибка при загрузке файлов: {e}")
    print("Убедитесь, что train.csv, test.csv и sample_submission.csv находятся в указанной директории DATA_PATH.")
    exit()

print(f"Размер тренировочных данных: {train_df.shape}")
print(f"Размер тестовых данных: {test_df.shape}")

# --- Целевая переменная ---

TARGET = 'Listening_Time_minutes'

# --- Предобработка и создание признаков ---

print("\nНачало предобработки...")

# Сохраняем ID для файла отправки
test_ids = test_df['id']
# train_ids = train_df['id'] # Удалено, т.к. не используются

# Объединяем train и test для единообразной обработки (кроме целевой переменной и ID)
# Это позволяет обрабатывать пропуски и создавать признаки одинаково для обоих наборов
train_df_processed = train_df.drop([TARGET, 'id'], axis=1)
test_df_processed = test_df.drop('id', axis=1)

combined_df_initial = pd.concat([train_df_processed, test_df_processed], axis=0, ignore_index=True)
print(f"Размер объединенного датасета перед обработкой: {combined_df_initial.shape}")

# Трансформируем целевую переменную (для CatBoost тоже полезно)
y_train = np.log1p(train_df[TARGET]) # Применяем log1p к целевой переменной
print(f"Целевая переменная '{TARGET}' трансформирована с помощью log1p.")


# Идентифицируем типы признаков ДО создания новых и OHE
numerical_features = combined_df_initial.select_dtypes(include=np.number).columns.tolist()
categorical_features = combined_df_initial.select_dtypes(exclude=np.number).columns.tolist() # Должна быть только 'Join_Date' и возможно 'Gender', etc.

print(f"\nИдентифицированы числовые признаки (до доп. обработки): {numerical_features}")
print(f"Идентифицированы изначальные категориальные признаки: {categorical_features}")


# Создание признаков из Join_Date
# Проверяем наличие колонки Join_Date и преобразуем ее в datetime
if 'Join_Date' in combined_df_initial.columns:
    print("\nСоздание признаков из 'Join_Date'...")
    combined_df_initial['Join_Date'] = pd.to_datetime(combined_df_initial['Join_Date'])

    # Извлечение признаков времени
    combined_df_initial['join_year'] = combined_df_initial['Join_Date'].dt.year
    combined_df_initial['join_month'] = combined_df_initial['Join_Date'].dt.month
    combined_df_initial['join_day'] = combined_df_initial['Join_Date'].dt.day
    combined_df_initial['join_dayofweek'] = combined_df_initial['Join_Date'].dt.dayofweek
    combined_df_initial['join_dayofyear'] = combined_df_initial['Join_Date'].dt.dayofyear
    combined_df_initial['join_quarter'] = combined_df_initial['Join_Date'].dt.quarter
    # combined_df_initial['join_weekofyear'] = combined_df_initial['Join_Date'].dt.isocalendar().week.astype(int) # Удалено, т.к. isocalendar().week может вернуть float/NA

    # Возможно, количество дней с самой ранней даты в датасете
    earliest_date = combined_df_initial['Join_Date'].min()
    combined_df_initial['days_since_earliest_join'] = (combined_df_initial['Join_Date'] - earliest_date).dt.days

    # Удаляем исходную колонку Join_Date
    combined_df_initial = combined_df_initial.drop('Join_Date', axis=1)

    # Обновляем списки признаков после создания новых и удаления старой
    numerical_features = combined_df_initial.select_dtypes(include=np.number).columns.tolist()
    categorical_features = combined_df_initial.select_dtypes(exclude=np.number).columns.tolist() # Теперь Join_Date тут не будет


print(f"Числовые признаки после создания признаков из даты: {numerical_features}")
print(f"Категориальные признаки после создания признаков из даты: {categorical_features}")


# Обработка пропусков (Простая импутация)
print("\nОбработка пропусков...")
for col in numerical_features:
    if combined_df_initial[col].isnull().any():
        median_val = combined_df_initial[col].median()
        print(f"Импутация числового признака '{col}' медианой: {median_val}")
        combined_df_initial[col] = combined_df_initial[col].fillna(median_val)

for col in categorical_features:
    if combined_df_initial[col].isnull().any():
        # Для категориальных признаков - мода или специальное значение 'Missing'/'Unknown'
        mode_val = combined_df_initial[col].mode()[0]
        print(f"Импутация категориального признака '{col}' модой: {mode_val}")
        combined_df_initial[col] = combined_df_initial[col].fillna(mode_val)


# --- Разделение данных для CatBoost ---

print("\nПодготовка данных для CatBoost (нативная обработка категориальных)...")
# Используем данные ПОСЛЕ импутации, но ДО OHE для CatBoost
combined_df_cat = combined_df_initial.copy()

# Разделяем обратно на train и test для CatBoost
# Количество строк в train_df_processed - это исходное количество тренировочных строк
X_train_cat = combined_df_cat.iloc[:len(train_df_processed)]
X_test_cat = combined_df_cat.iloc[len(train_df_processed):]

# Идентифицируем индексы категориальных признаков для CatBoost (на основе колонок X_train_cat)
# Это индексы ОРИГИНАЛЬНЫХ категориальных колонок в датафрейме БЕЗ OHE
cat_features_indices = [X_train_cat.columns.get_loc(col) for col in categorical_features if col in X_train_cat.columns]
print(f"CatBoost будет использовать оригинальные категориальные признаки по индексам: {cat_features_indices}")


print(f"Размер CatBoost Train: {X_train_cat.shape}")
print(f"Размер CatBoost Test: {X_test_cat.shape}")


# Очистка памяти
del train_df, test_df, combined_df_initial, combined_df_cat, train_df_processed, test_df_processed
gc.collect()

# --- Обучение модели CatBoost с кросс-валидацией ---

print("\nНачало обучения модели CatBoost с KFold CV...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# Инициализируем массивы для OOF (Out-of-Fold) предсказаний и предсказаний на тесте
# OOF предсказания используются для оценки модели
# Test предсказания накапливаются и усредняются
oof_predictions_cat = np.zeros(len(X_train_cat)) # Размер соответствует train (без OHE)
test_predictions_cat = np.zeros(len(X_test_cat)) # Размер соответствует test (без OHE)


# Параметры CatBoost
# Подобраны стандартные хорошие параметры
cat_params = {
    'objective': 'RMSE',            # Целевая функция
    'eval_metric': 'RMSE',          # Метрика оценки
    'iterations': 20000,            # Эквивалент n_estimators (увеличиваем)
    'learning_rate': 0.005,         # Чуть уменьшаем learning_rate
    'depth': 8,                     # Глубина дерева (немного увеличиваем)
    'l2_leaf_reg': 3,               # L2 регуляризация
    'loss_function': 'RMSE',
    'verbose': 1,                   # Подавляем вывод хода обучения
    'random_seed': RANDOM_SEED,
    'thread_count': -1,             # Использовать все ядра
    'early_stopping_rounds': 200,   # Ранняя остановка (увеличиваем терпение)
    'allow_writing_files': False # Отключаем создание вспомогательных файлов
}

# Запускаем кросс-валидацию для CatBoost
for fold, (train_index, val_index) in enumerate(kf.split(X_train_cat, y_train)): # Используем индексы из CatBoost набора
    print(f"\n--- Фол1 {fold+1}/{N_FOLDS} ---")

    # --- Данные для текущего фолда ---
    # Данные для CatBoost (нативная обработка категориальных) - используем те же индексы
    X_train_fold_cat, X_val_fold_cat = X_train_cat.iloc[train_index], X_train_cat.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index] # y_train уже трансформирована

    # --- CatBoost ---
    print("Обучение CatBoost...")
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_train_fold_cat, y_train_fold,  # Используем CatBoost-специфичные тренировочные данные (без OHE)
                  eval_set=[(X_val_fold_cat, y_val_fold)], # Используем CatBoost-специфичные валидационные данные (без OHE)
                  cat_features=cat_features_indices,   # Передаем индексы оригинальных категориальных признаков
                  use_best_model=True, # Автоматически использует лучшую модель по ранней остановке
                  verbose=0) # Подавляем вывод хода обучения

    val_preds_cat = cat_model.predict(X_val_fold_cat) # Предсказываем на CatBoost-специфичных валидационных данных
    test_preds_cat_fold = cat_model.predict(X_test_cat) # Предсказываем на CatBoost-специфичных тестовых данных

    # Накапливаем OOF и тестовые предсказания
    oof_predictions_cat[val_index] = val_preds_cat
    test_predictions_cat += test_preds_cat_fold / N_FOLDS # Усредняем предсказания по фолдам

    # Оценка RMSE на валидационном наборе этого фолда (после обратного преобразования)
    fold_rmse_cat = mean_squared_error(np.expm1(y_val_fold), np.expm1(val_preds_cat), squared=False)
    print(f"CatBoost Фол1 {fold+1} Validation RMSE (обратное преобразование): {fold_rmse_cat:.5f}")

    del cat_model, val_preds_cat, test_preds_cat_fold, X_train_fold_cat, X_val_fold_cat, y_train_fold, y_val_fold
    gc.collect()


# --- Оценка CatBoost ---

print("\n--- Оценка CatBoost ---")

# Применяем обратное преобразование к OOF предсказаниям перед расчетом общей метрики
oof_predictions_cat_orig_scale = np.expm1(oof_predictions_cat)

# Исходная целевая переменная для финальной оценки OOF
y_train_orig_scale = np.expm1(y_train)

overall_oof_rmse_cat = mean_squared_error(y_train_orig_scale, oof_predictions_cat_orig_scale, squared=False)

print(f"\nОбщий OOF RMSE (CatBoost): {overall_oof_rmse_cat:.5f}")


# --- Отправка (Submission) ---

print("\nСоздание файла отправки с использованием предсказаний CatBoost...")

# Применяем обратное преобразование к тестовым предсказаниям CatBoost
test_predictions_cat_orig_scale = np.expm1(test_predictions_cat)

# Опционально: Убедимся, что финальные предсказания неотрицательны
test_predictions_cat_orig_scale[test_predictions_cat_orig_scale < 0] = 0


submission_df = pd.DataFrame({'id': test_ids, TARGET: test_predictions_cat_orig_scale})

# Опционально: Убедимся, что финальные предсказания неотрицательны
submission_df[TARGET] = submission_df[TARGET].clip(lower=0)

submission_df.to_csv('submission_catboost_only.csv', index=False)

print("Файл отправки 'submission_catboost_only.csv' создан успешно!")
print(submission_df.head())

print("\nСкрипт завершен.")

