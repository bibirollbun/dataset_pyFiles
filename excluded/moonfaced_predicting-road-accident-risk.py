import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Установка стиля для графиков
sns.set(style='whitegrid')


# Загрузка данных
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
    sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
except FileNotFoundError:
    print("Файлы не найдены. Укажите правильный путь к данным.")
    # Завершаем выполнение, если файлы не найдены
    exit()

# Удаление столбца 'id', так как он не является признаком
train_df = train_df.drop('id', axis=1)
test_ids = test_df['id'] # Сохраняем id для файла с результатами
test_df = test_df.drop('id', axis=1)


print("Первые 5 строк обучающего набора:")
print(train_df.head())
print("\n" + "="*50 + "\n")

print("Размер обучающего набора:", train_df.shape)
print("Размер тестового набора:", test_df.shape)
print("\n" + "="*50 + "\n")


print("Информация об обучающем наборе:")
train_df.info()
print("\n" + "="*50 + "\n")

print("Информация о тестовом наборе:")
test_df.info()


print("Описательная статистика для обучающего набора:")
pd.set_option('display.max_columns', None) # Показать все столбцы
print(train_df.describe())


plt.figure(figsize=(14, 6))

# Гистограмма
plt.subplot(1, 2, 1)
sns.histplot(train_df['accident_risk'], kde=True, bins=50)
plt.title('Распределение Accident Risk')
plt.xlabel('Accident Risk')
plt.ylabel('Частота')

# Ящик с усами
plt.subplot(1, 2, 2)
sns.boxplot(x=train_df['accident_risk'])
plt.title('Ящик с усами для Accident Risk')
plt.xlabel('Accident Risk')

plt.tight_layout()
plt.show()


# Предполагаем, что train_df уже загружен
import seaborn as sns
import matplotlib.pyplot as plt

numerical_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

# Визуализация распределений
plt.figure(figsize=(15, 8))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(train_df[feature], bins=20, kde=True)
    plt.title(f'Распределение {feature}')
plt.tight_layout()
plt.show()


# Рассчитываем корреляционную матрицу только для числовых столбцов
corr_matrix = train_df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Тепловая карта корреляций числовых признаков')
plt.show()


categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day']

plt.figure(figsize=(16, 12))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=feature, y='accident_risk', data=train_df)
    plt.title(f'Accident Risk в зависимости от {feature}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# =============================================================================
# FINAL SCRIPT 2: LightGBM with Optuna, Feature Engineering, CV, GPU, 
# Model Saving, OOF and Test Predictions
# =============================================================================

import pandas as pd
import lightgbm as lgb
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
import joblib
import optuna # <--- ИМПОРТ OPTUNA

warnings.filterwarnings('ignore')

# --- Функция для создания признаков (Версия 2) ---
def create_features_v2(df):
    """Создает полный набор инженерных признаков."""
    df['weather_lighting_interaction'] = df['weather'].astype(str) + '_' + df['lighting'].astype(str)
    df['road_type_lanes_interaction'] = df['road_type'].astype(str) + '_lanes_' + df['num_lanes'].astype(str)
    curvature_stats = df.groupby('speed_limit')['curvature'].agg(['mean', 'std']).reset_index()
    curvature_stats.columns = ['speed_limit', 'curvature_by_speed_mean', 'curvature_by_speed_std']
    df = df.merge(curvature_stats, on='speed_limit', how='left')
    df['curvature_by_speed_std'].fillna(0, inplace=True)
    df['curvature_per_lane'] = df['curvature'] / (df['num_lanes'] + 1e-6)
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1e-6)
    df['curvature_freq'] = df.groupby('curvature')['curvature'].transform('count')
    df['lanes_freq'] = df.groupby('num_lanes')['num_lanes'].transform('count')
    df['curvature_sq'] = df['curvature'] ** 2
    return df

# --- Шаг 1: Подготовка данных ---
print("LGBM Pipeline: 1. Загрузка данных...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv').drop('id', axis=1)
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test_ids = test_df['id']
test_df = test_df.drop('id', axis=1)

X = train_df.drop('accident_risk', axis=1)
y = train_df['accident_risk']

print("LGBM Pipeline: 2. Создание признаков (v2)...")
combined_df = pd.concat([X, test_df], axis=0, ignore_index=True)
combined_df_featured = create_features_v2(combined_df)

print("LGBM Pipeline: 3. One-Hot Encoding...")
categorical_features = [
    'road_type', 'lighting', 'weather', 'time_of_day', 
    'num_lanes', 'speed_limit', 'weather_lighting_interaction', 'road_type_lanes_interaction'
]
combined_df_encoded = pd.get_dummies(combined_df_featured, columns=categorical_features, drop_first=True)

split_point = len(X)
X_processed = combined_df_encoded.iloc[:split_point]
X_test_processed = combined_df_encoded.iloc[split_point:]

# --- Шаг 2: Оптимизация с Optuna ---
print("\nLGBM Pipeline: 4. Запуск оптимизации гиперпараметров с Optuna...")

N_SPLITS_OPTUNA = 3 # Можно использовать меньше фолдов для ускорения оптимизации

def objective(trial):
    """
    Целевая функция для Optuna.
    Она принимает объект 'trial', предлагает гиперпараметры,
    обучает модель на кросс-валидации и возвращает средний RMSE.
    """
    # Определение пространства поиска гиперпараметров
    params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': 10000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'n_jobs': -1,
        'random_state': 42,
        'device': 'gpu', # Используем GPU
        'verbosity': -1
    }
    
    kf = KFold(n_splits=N_SPLITS_OPTUNA, shuffle=True, random_state=42)
    oof_rmse_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_processed, y)):
        X_train, X_val = X_processed.iloc[train_index], X_processed.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)], 
                  eval_metric='rmse', 
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        oof_rmse_scores.append(rmse)

    return np.mean(oof_rmse_scores)

# Создание и запуск исследования Optuna
study = optuna.create_study(direction='minimize') # Мы хотим минимизировать RMSE
study.optimize(objective, n_trials=50) # Запускаем 50 итераций подбора. Можно изменить.

print("Оптимизация завершена!")
print("Лучший результат (RMSE):", study.best_value)
print("Лучшие параметры:", study.best_params)

# --- Шаг 3: Кросс-валидация и обучение с лучшими параметрами ---
print("\nLGBM Pipeline: 5. Запуск кросс-валидации с лучшими параметрами...")
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Инициализация массивов для результатов
oof_preds_lgbm = np.zeros(split_point)
test_predictions_lgbm = []
os.makedirs('lgbm_models', exist_ok=True) 

# Параметры модели (обновляем лучшими из Optuna)
best_params_from_optuna = study.best_params
params_lgbm = {
    'n_estimators': 10000, # Оставляем большим, т.к. есть early stopping
    'n_jobs': -1,
    'device': 'gpu',
    **best_params_from_optuna # <--- ИСПОЛЬЗУЕМ ЛУЧШИЕ ПАРАМЕТРЫ
}

for fold, (train_index, val_index) in enumerate(kf.split(X_processed, y)):
    print(f"--- Фолд {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X_processed.iloc[train_index], X_processed.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Инициализация и обучение
    model_lgbm = lgb.LGBMRegressor(**params_lgbm, random_state=42 + fold)
    model_lgbm.fit(X_train, y_train, 
                   eval_set=[(X_val, y_val)], 
                   eval_metric='rmse', 
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    
    # Сохранение модели
    model_path = f'lgbm_models/lgbm_fold_{fold+1}_optuna.joblib' # Обновим имя файла
    joblib.dump(model_lgbm, model_path)
    print(f"Модель сохранена в: {model_path}")
    
    # Сохранение OOF-предсказаний
    val_preds = model_lgbm.predict(X_val)
    oof_preds_lgbm[val_index] = val_preds
    
    # Сохранение предсказаний для теста
    test_predictions_lgbm.append(model_lgbm.predict(X_test_processed))

# --- Шаг 4: Пост-обработка и сохранение результатов ---
print("\nLGBM Pipeline: 6. Обработка и сохранение результатов...")

# Оценка по OOF
final_oof_rmse_lgbm = np.sqrt(mean_squared_error(y, oof_preds_lgbm))
print(f"Итоговый OOF RMSE для LGBM (после Optuna): {final_oof_rmse_lgbm:.5f}")

# Усреднение предсказаний для теста
final_test_preds_lgbm = np.mean(test_predictions_lgbm, axis=0)

# Сохранение артефактов
np.save('oof_preds_lgbm_optuna.npy', oof_preds_lgbm)
np.save('test_preds_lgbm_optuna.npy', final_test_preds_lgbm)
print("OOF и тестовые предсказания LGBM (Optuna) сохранены в .npy файлы.")

# Создание submission файла
submission_lgbm = pd.DataFrame({'id': test_ids, 'accident_risk': final_test_preds_lgbm})
submission_lgbm['accident_risk'] = submission_lgbm['accident_risk'].clip(0, 1)
submission_lgbm.to_csv('submission_lgbm_optuna_final.csv', index=False)
print("Файл 'submission_lgbm_optuna_final.csv' успешно создан!")


import matplotlib.pyplot as plt
import seaborn as sns


# Выводим топ-20 признаков
print("\n--- Топ-20 самых важных признаков (в среднем по 5 фолдам) ---")
print(mean_importance.head(1000))
plt.figure(figsize=(12, 10))
sns.barplot(x=mean_importance.head(20).values, y=mean_importance.head(20).index)
plt.title('Топ-20 самых важных признаков')
plt.xlabel('Средняя важность')
plt.ylabel('Признак')
plt.grid(True)
plt.show()



# =============================================================================
# FINAL SCRIPT 3: CatBoost with Optuna, Feature Engineering, CV, GPU, 
# Model Saving, OOF and Test Predictions
# =============================================================================

import pandas as pd
import catboost as cb
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
import optuna # <--- ИМПОРТ OPTUNA

warnings.filterwarnings('ignore')

# --- Функция для создания признаков (Версия 2) ---
def create_features_v2(df):
    """Создает полный набор инженерных признаков."""
    df['weather_lighting_interaction'] = df['weather'].astype(str) + '_' + df['lighting'].astype(str)
    df['road_type_lanes_interaction'] = df['road_type'].astype(str) + '_lanes_' + df['num_lanes'].astype(str)
    curvature_stats = df.groupby('speed_limit')['curvature'].agg(['mean', 'std']).reset_index()
    curvature_stats.columns = ['speed_limit', 'curvature_by_speed_mean', 'curvature_by_speed_std']
    df = df.merge(curvature_stats, on='speed_limit', how='left')
    df['curvature_by_speed_std'].fillna(0, inplace=True)
    df['curvature_per_lane'] = df['curvature'] / (df['num_lanes'] + 1e-6)
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1e-6)
    df['curvature_freq'] = df.groupby('curvature')['curvature'].transform('count')
    df['lanes_freq'] = df.groupby('num_lanes')['num_lanes'].transform('count')
    df['curvature_sq'] = df['curvature'] ** 2
    return df

# --- Шаг 1: Подготовка данных ---
print("CatBoost Pipeline: 1. Загрузка данных...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv').drop('id', axis=1)
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test_ids = test_df['id']
test_df = test_df.drop('id', axis=1)

X = train_df.drop('accident_risk', axis=1)
y = train_df['accident_risk']

print("CatBoost Pipeline: 2. Создание признаков (v2)...")
combined_df = pd.concat([X, test_df], axis=0, ignore_index=True)
combined_df_featured = create_features_v2(combined_df)

split_point = len(X)
X_processed = combined_df_featured.iloc[:split_point]
X_test_processed = combined_df_featured.iloc[split_point:]

# Определение категориальных признаков для CatBoost
categorical_features = [
    'road_type', 'lighting', 'weather', 'time_of_day', 'road_signs_present', 
    'public_road', 'holiday', 'school_season', 'num_lanes', 'speed_limit',
    'weather_lighting_interaction', 'road_type_lanes_interaction'
]
for col in categorical_features:
    X_processed[col] = X_processed[col].astype(str)
    X_test_processed[col] = X_test_processed[col].astype(str)
    
# --- Шаг 2: Оптимизация с Optuna ---
print("\nCatBoost Pipeline: 3. Запуск оптимизации гиперпараметров с Optuna...")

N_SPLITS_OPTUNA = 3 # Можно использовать меньше фолдов для ускорения оптимизации

def objective(trial):
    """
    Целевая функция для Optuna.
    Она принимает объект 'trial', предлагает гиперпараметры,
    обучает модель на кросс-валидации и возвращает средний RMSE.
    """
    # Определение пространства поиска гиперпараметров
    params = {
        'iterations': 10000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'task_type': 'GPU',
        'verbose': 0,
        'random_seed': 42,
        'early_stopping_rounds': 100,
    }
    
    kf = KFold(n_splits=N_SPLITS_OPTUNA, shuffle=True, random_state=42)
    oof_rmse_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_processed, y)):
        X_train, X_val = X_processed.iloc[train_index], X_processed.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model = cb.CatBoostRegressor(**params)
        model.fit(X_train, y_train, 
                  eval_set=(X_val, y_val), 
                  cat_features=categorical_features,
                  verbose=0) # verbose=0 здесь, чтобы не засорять лог Optuna
        
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        oof_rmse_scores.append(rmse)

    return np.mean(oof_rmse_scores)

# Создание и запуск исследования Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50) # Запускаем 50 итераций подбора.

print("Оптимизация завершена!")
print("Лучший результат (RMSE):", study.best_value)
print("Лучшие параметры:", study.best_params)

# --- Шаг 3: Кросс-валидация и обучение с лучшими параметрами ---
print("\nCatBoost Pipeline: 4. Запуск кросс-валидации с лучшими параметрами...")
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Инициализация массивов для результатов
oof_preds_catboost = np.zeros(split_point)
test_predictions_catboost = []
os.makedirs('catboost_models', exist_ok=True) 

# Параметры модели (обновляем лучшими из Optuna)
best_params_from_optuna = study.best_params
params_catboost = {
    'iterations': 10000,
    'loss_function': 'RMSE', 
    'eval_metric': 'RMSE',
    'verbose': 0, 
    'early_stopping_rounds': 100,
    'task_type': 'GPU',
    **best_params_from_optuna # <--- ИСПОЛЬЗУЕМ ЛУЧШИЕ ПАРАМЕТРЫ
}

for fold, (train_index, val_index) in enumerate(kf.split(X_processed, y)):
    print(f"--- Фолд {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X_processed.iloc[train_index], X_processed.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Инициализация и обучение
    model_cb = cb.CatBoostRegressor(**params_catboost, random_seed=42 + fold)
    model_cb.fit(X_train, y_train, 
                 eval_set=(X_val, y_val), 
                 cat_features=categorical_features)
    
    # Сохранение модели
    model_path = f'catboost_models/catboost_fold_{fold+1}_optuna.cbm' # Обновим имя
    model_cb.save_model(model_path)
    print(f"Модель сохранена в: {model_path}")
    
    # Сохранение OOF-предсказаний
    val_preds = model_cb.predict(X_val)
    oof_preds_catboost[val_index] = val_preds
    
    # Сохранение предсказаний для теста
    test_predictions_catboost.append(model_cb.predict(X_test_processed))

# --- Шаг 4: Пост-обработка и сохранение результатов ---
print("\nCatBoost Pipeline: 5. Обработка и сохранение результатов...")

# Оценка по OOF
final_oof_rmse_catboost = np.sqrt(mean_squared_error(y, oof_preds_catboost))
print(f"Итоговый OOF RMSE для CatBoost (после Optuna): {final_oof_rmse_catboost:.5f}")

# Усреднение предсказаний для теста
final_test_preds_catboost = np.mean(test_predictions_catboost, axis=0)

# Сохранение артефактов
np.save('oof_preds_catboost_optuna.npy', oof_preds_catboost)
np.save('test_preds_catboost_optuna.npy', final_test_preds_catboost)
print("OOF и тестовые предсказания CatBoost (Optuna) сохранены в .npy файлы.")

# Создание submission файла
submission_catboost = pd.DataFrame({'id': test_ids, 'accident_risk': final_test_preds_catboost})
submission_catboost['accident_risk'] = submission_catboost['accident_risk'].clip(0, 1)
submission_catboost.to_csv('submission_catboost_optuna_final.csv', index=False)
print("Файл 'submission_catboost_optuna_final.csv' успешно создан!")





# =============================================================================
# ФИНАЛЬНЫЙ СКРИПТ: Генерация OOF и Test предсказаний для LGBM и CatBoost
# Версия 3: Добавлена генерация предсказаний для тестовой выборки
# =============================================================================

import pandas as pd
import lightgbm as lgb
import catboost as cb
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# --- Функция для создания признаков (Версия 2) ---



# --- Шаг 3: Обучение CatBoost ---
print("\nCatBoost Pipeline: 3. Запуск кросс-валидации...")

X_processed_cat = combined_df_featured.iloc[:split_point]
X_test_processed_cat = combined_df_featured.iloc[split_point:]
categorical_features_cat = [
    'road_type', 'lighting', 'weather', 'time_of_day', 'road_signs_present', 
    'public_road', 'holiday', 'school_season', 'num_lanes', 'speed_limit',
    'weather_lighting_interaction', 'road_type_lanes_interaction'
]
for col in categorical_features_cat:
    X_processed_cat[col] = X_processed_cat[col].astype(str)
    X_test_processed_cat[col] = X_test_processed_cat[col].astype(str)

best_params_catboost = {
    'learning_rate': 0.02570501026298664, 'depth': 9, 'l2_leaf_reg': 1.2403048474143457, 
    'bagging_temperature': 0.09783094822547894, 'random_strength': 0.6352174046846455,
    'iterations': 10000, 'loss_function': 'RMSE', 'eval_metric': 'RMSE', 'task_type': 'GPU',
    'verbose': 0, 'early_stopping_rounds': 200
}

oof_preds_catboost = np.zeros(split_point)
test_predictions_catboost = [] # <--- ДОБАВЛЕНО: Список для хранения предсказаний на тесте

for fold, (train_index, val_index) in enumerate(kf.split(X_processed_cat, y)):
    print(f"--- CatBoost Фолд {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X_processed_cat.iloc[train_index], X_processed_cat.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model = cb.CatBoostRegressor(**best_params_catboost, random_seed=42 + fold)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=categorical_features_cat)
    
    oof_preds_catboost[val_index] = model.predict(X_val)
    test_predictions_catboost.append(model.predict(X_test_processed_cat)) # <--- ДОБАВЛЕНО: Предсказание на тесте

final_oof_rmse_catboost = np.sqrt(mean_squared_error(y, oof_preds_catboost))
print(f"\nИтоговый OOF RMSE для CatBoost: {final_oof_rmse_catboost:.5f}")
np.save('oof_preds_catboost.npy', oof_preds_catboost)
print("Файл 'oof_preds_catboost.npy' сохранен.")

# <--- ДОБАВЛЕНО: Усреднение и сохранение тестовых предсказаний
final_test_preds_catboost = np.mean(test_predictions_catboost, axis=0)
np.save('test_preds_catboost.npy', final_test_preds_catboost)
print("Файл 'test_preds_catboost.npy' сохранен.")



# =============================================================================
# FINAL SCRIPT 3: XGBoost with Feature Engineering, CV, GPU, 
# Model Saving, OOF and Test Predictions
# =============================================================================

import pandas as pd
import xgboost as xgb
import numpy as np
import os
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# --- Функция для создания признаков (Версия 2) ---
def create_features_v2(df):
    """Создает полный набор инженерных признаков."""
    df['weather_lighting_interaction'] = df['weather'].astype(str) + '_' + df['lighting'].astype(str)
    df['road_type_lanes_interaction'] = df['road_type'].astype(str) + '_lanes_' + df['num_lanes'].astype(str)
    curvature_stats = df.groupby('speed_limit')['curvature'].agg(['mean', 'std']).reset_index()
    curvature_stats.columns = ['speed_limit', 'curvature_by_speed_mean', 'curvature_by_speed_std']
    df = df.merge(curvature_stats, on='speed_limit', how='left')
    df['curvature_by_speed_std'].fillna(0, inplace=True)
    df['curvature_per_lane'] = df['curvature'] / (df['num_lanes'] + 1e-6)
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1e-6)
    df['curvature_freq'] = df.groupby('curvature')['curvature'].transform('count')
    df['lanes_freq'] = df.groupby('num_lanes')['num_lanes'].transform('count')
    df['curvature_sq'] = df['curvature'] ** 2
    return df

# --- Шаг 1: Подготовка данных ---
print("XGBoost Pipeline: 1. Загрузка данных...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv').drop('id', axis=1)
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test_ids = test_df['id']
test_df = test_df.drop('id', axis=1)

X = train_df.drop('accident_risk', axis=1)
y = train_df['accident_risk']

print("XGBoost Pipeline: 2. Создание признаков (v2)...")
combined_df = pd.concat([X, test_df], axis=0, ignore_index=True)
combined_df_featured = create_features_v2(combined_df)

print("XGBoost Pipeline: 3. One-Hot Encoding...")
categorical_features = [
    'road_type', 'lighting', 'weather', 'time_of_day', 
    'num_lanes', 'speed_limit', 'weather_lighting_interaction', 'road_type_lanes_interaction'
]
combined_df_encoded = pd.get_dummies(combined_df_featured, columns=categorical_features, drop_first=True)

split_point = len(X)
X_processed = combined_df_encoded.iloc[:split_point]
X_test_processed = combined_df_encoded.iloc[split_point:]

# --- Шаг 2: Кросс-валидация и обучение ---
print("\nXGBoost Pipeline: 4. Запуск кросс-валидации...")
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Инициализация массивов для результатов
oof_preds_xgb = np.zeros(split_point)
test_predictions_xgb = []
os.makedirs('xgb_models', exist_ok=True) # Создаем папку для моделей

# Параметры модели (включая GPU)
# tree_method='hist' и device='cuda' - стандартные параметры для включения GPU в XGBoost
params_xgb = {
    'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'n_estimators': 10000,
    'learning_rate': 0.03, 'max_depth': 8, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'tree_method': 'hist', 'device': 'cuda'
}

for fold, (train_index, val_index) in enumerate(kf.split(X_processed, y)):
    print(f"--- Фолд {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X_processed.iloc[train_index], X_processed.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Инициализация и обучение
    model_xgb = xgb.XGBRegressor(**params_xgb, random_state=42 + fold)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                  early_stopping_rounds=100, verbose=False)
    
    # Сохранение модели
    # XGBoost имеет метод .save_model(), стандартное расширение .json
    model_path = f'xgb_models/xgb_fold_{fold+1}.json'
    model_xgb.save_model(model_path)
    print(f"Модель сохранена в: {model_path}")
    
    # Сохранение OOF-предсказаний
    val_preds = model_xgb.predict(X_val)
    oof_preds_xgb[val_index] = val_preds
    
    # Сохранение предсказаний для теста
    test_predictions_xgb.append(model_xgb.predict(X_test_processed))

# --- Шаг 3: Пост-обработка и сохранение результатов ---
print("\nXGBoost Pipeline: 5. Обработка и сохранение результатов...")

# Оценка по OOF
final_oof_rmse_xgb = np.sqrt(mean_squared_error(y, oof_preds_xgb))
print(f"Итоговый OOF RMSE для XGBoost: {final_oof_rmse_xgb:.5f}")

# Усреднение предсказаний для теста
final_test_preds_xgb = np.mean(test_predictions_xgb, axis=0)

# Сохранение артефактов
np.save('oof_preds_xgb.npy', oof_preds_xgb)
np.save('test_preds_xgb.npy', final_test_preds_xgb)
print("OOF и тестовые предсказания XGBoost сохранены в .npy файлы.")

# Создание submission файла
submission_xgb = pd.DataFrame({'id': test_ids, 'accident_risk': final_test_preds_xgb})
submission_xgb['accident_risk'] = submission_xgb['accident_risk'].clip(0, 1)
submission_xgb.to_csv('submission_xgb_final.csv', index=False)
print("Файл 'submission_xgb_final.csv' успешно создан!")


# =============================================================================
# FINAL SCRIPT 4: XGBoost with Optuna, Feature Engineering, CV, GPU, 
# Model Saving, OOF and Test Predictions
# =============================================================================

import pandas as pd
import xgboost as xgb
import numpy as np
import os
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
import optuna # <--- ИМПОРТ OPTUNA

warnings.filterwarnings('ignore')

# --- Функция для создания признаков (Версия 2) ---
def create_features_v2(df):
    """Создает полный набор инженерных признаков."""
    df['weather_lighting_interaction'] = df['weather'].astype(str) + '_' + df['lighting'].astype(str)
    df['road_type_lanes_interaction'] = df['road_type'].astype(str) + '_lanes_' + df['num_lanes'].astype(str)
    curvature_stats = df.groupby('speed_limit')['curvature'].agg(['mean', 'std']).reset_index()
    curvature_stats.columns = ['speed_limit', 'curvature_by_speed_mean', 'curvature_by_speed_std']
    df = df.merge(curvature_stats, on='speed_limit', how='left')
    df['curvature_by_speed_std'].fillna(0, inplace=True)
    df['curvature_per_lane'] = df['curvature'] / (df['num_lanes'] + 1e-6)
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1e-6)
    df['curvature_freq'] = df.groupby('curvature')['curvature'].transform('count')
    df['lanes_freq'] = df.groupby('num_lanes')['num_lanes'].transform('count')
    df['curvature_sq'] = df['curvature'] ** 2
    return df

# --- Шаг 1: Подготовка данных ---
print("XGBoost Pipeline: 1. Загрузка данных...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv').drop('id', axis=1)
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test_ids = test_df['id']
test_df = test_df.drop('id', axis=1)

X = train_df.drop('accident_risk', axis=1)
y = train_df['accident_risk']

print("XGBoost Pipeline: 2. Создание признаков (v2)...")
combined_df = pd.concat([X, test_df], axis=0, ignore_index=True)
combined_df_featured = create_features_v2(combined_df)

print("XGBoost Pipeline: 3. One-Hot Encoding...")
categorical_features = [
    'road_type', 'lighting', 'weather', 'time_of_day', 
    'num_lanes', 'speed_limit', 'weather_lighting_interaction', 'road_type_lanes_interaction'
]
combined_df_encoded = pd.get_dummies(combined_df_featured, columns=categorical_features, drop_first=True)

split_point = len(X)
X_processed = combined_df_encoded.iloc[:split_point]
X_test_processed = combined_df_encoded.iloc[split_point:]

# --- Шаг 2: Оптимизация с Optuna ---
print("\nXGBoost Pipeline: 4. Запуск оптимизации гиперпараметров с Optuna...")

N_SPLITS_OPTUNA = 3 # Меньше фолдов для быстрой оптимизации

def objective(trial):
    """
    Целевая функция для Optuna для подбора параметров XGBoost.
    """
    # Определение пространства поиска гиперпараметров
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 10000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True), # L2 регуляризация
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),   # L1 регуляризация
        'tree_method': 'hist', 
        'device': 'cuda',
        'n_jobs': -1
    }
    
    kf = KFold(n_splits=N_SPLITS_OPTUNA, shuffle=True, random_state=42)
    oof_rmse_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_processed, y)):
        X_train, X_val = X_processed.iloc[train_index], X_processed.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model = xgb.XGBRegressor(**params, random_state=42)
        model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)], 
                  early_stopping_rounds=100, 
                  verbose=False)
        
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        oof_rmse_scores.append(rmse)

    return np.mean(oof_rmse_scores)

# Создание и запуск исследования Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50) # Запускаем 50 итераций подбора.

print("Оптимизация завершена!")
print("Лучший результат (RMSE):", study.best_value)
print("Лучшие параметры:", study.best_params)


# --- Шаг 3: Кросс-валидация и обучение с лучшими параметрами ---
print("\nXGBoost Pipeline: 5. Запуск кросс-валидации с лучшими параметрами...")
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Инициализация массивов для результатов
oof_preds_xgb = np.zeros(split_point)
test_predictions_xgb = []
os.makedirs('xgb_models', exist_ok=True)

# Параметры модели (обновляем лучшими из Optuna)
best_params_from_optuna = study.best_params
params_xgb = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 10000,
    'tree_method': 'hist',
    'device': 'cuda',
    **best_params_from_optuna # <--- ИСПОЛЬЗУЕМ ЛУЧШИЕ ПАРАМЕТРЫ
}

for fold, (train_index, val_index) in enumerate(kf.split(X_processed, y)):
    print(f"--- Фолд {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X_processed.iloc[train_index], X_processed.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Инициализация и обучение
    model_xgb = xgb.XGBRegressor(**params_xgb, random_state=42 + fold)
    model_xgb.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)], 
                  early_stopping_rounds=100, 
                  verbose=False)
    
    # Сохранение модели
    model_path = f'xgb_models/xgb_fold_{fold+1}_optuna.json'
    model_xgb.save_model(model_path)
    print(f"Модель сохранена в: {model_path}")
    
    # Сохранение OOF-предсказаний
    val_preds = model_xgb.predict(X_val)
    oof_preds_xgb[val_index] = val_preds
    
    # Сохранение предсказаний для теста
    test_predictions_xgb.append(model_xgb.predict(X_test_processed))

# --- Шаг 4: Пост-обработка и сохранение результатов ---
print("\nXGBoost Pipeline: 6. Обработка и сохранение результатов...")

# Оценка по OOF
final_oof_rmse_xgb = np.sqrt(mean_squared_error(y, oof_preds_xgb))
print(f"Итоговый OOF RMSE для XGBoost (после Optuna): {final_oof_rmse_xgb:.5f}")

# Усреднение предсказаний для теста
final_test_preds_xgb = np.mean(test_predictions_xgb, axis=0)

# Сохранение артефактов
np.save('oof_preds_xgb_optuna.npy', oof_preds_xgb)
np.save('test_preds_xgb_optuna.npy', final_test_preds_xgb)
print("OOF и тестовые предсказания XGBoost (Optuna) сохранены в .npy файлы.")

# Создание submission файла
submission_xgb = pd.DataFrame({'id': test_ids, 'accident_risk': final_test_preds_xgb})
submission_xgb['accident_risk'] = submission_xgb['accident_risk'].clip(0, 1)
submission_xgb.to_csv('submission_xgb_optuna_final.csv', index=False)
print("Файл 'submission_xgb_optuna_final.csv' успешно создан!")


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# --- Функция для создания признаков (Версия 2) ---
def create_features_v2(df):
    """Создает полный набор инженерных признаков."""
    # ... (код функции create_features_v2 без изменений) ...
    df['weather_lighting_interaction'] = df['weather'].astype(str) + '_' + df['lighting'].astype(str)
    df['road_type_lanes_interaction'] = df['road_type'].astype(str) + '_lanes_' + df['num_lanes'].astype(str)
    curvature_stats = df.groupby('speed_limit')['curvature'].agg(['mean', 'std']).reset_index()
    curvature_stats.columns = ['speed_limit', 'curvature_by_speed_mean', 'curvature_by_speed_std']
    df = df.merge(curvature_stats, on='speed_limit', how='left')
    df['curvature_by_speed_std'].fillna(0, inplace=True)
    df['curvature_per_lane'] = df['curvature'] / (df['num_lanes'] + 1e-6)
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1e-6)
    df['curvature_freq'] = df.groupby('curvature')['curvature'].transform('count')
    df['lanes_freq'] = df.groupby('num_lanes')['num_lanes'].transform('count')
    df['curvature_sq'] = df['curvature'] ** 2
    return df

# --- Шаг 1: Подготовка данных ---
print("RandomForest Pipeline: 1. Загрузка данных...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv').drop('id', axis=1)
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test_ids = test_df['id']
test_df = test_df.drop('id', axis=1)
X = train_df.drop('accident_risk', axis=1)
y = train_df['accident_risk']

print("RandomForest Pipeline: 2. Создание признаков (v2)...")
combined_df = pd.concat([X, test_df], axis=0, ignore_index=True)
combined_df_featured = create_features_v2(combined_df)

print("RandomForest Pipeline: 3. One-Hot Encoding...")
categorical_features = [
    'road_type', 'lighting', 'weather', 'time_of_day', 
    'num_lanes', 'speed_limit', 'weather_lighting_interaction', 'road_type_lanes_interaction'
]
combined_df_encoded = pd.get_dummies(combined_df_featured, columns=categorical_features, drop_first=True)
split_point = len(X)
X_processed = combined_df_encoded.iloc[:split_point]
X_test_processed = combined_df_encoded.iloc[split_point:]

# --- Шаг 2: Кросс-валидация и обучение ---
print("\nRandomForest Pipeline: 4. Запуск кросс-валидации...")
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds_rf = np.zeros(split_point)
test_predictions_rf = []
os.makedirs('rf_models', exist_ok=True)

# Параметры для RandomForest. n_jobs=-1 использует все ядра CPU.
# Random Forest не имеет ранней остановки, он просто строит заданное количество деревьев.
params_rf = {
    'n_estimators': 1000,     # 200 деревьев - хороший компромисс между скоростью и качеством
    'max_depth': 15,         # Ограничиваем глубину для предотвращения переобучения
    'min_samples_leaf': 10,  # Увеличиваем, чтобы сделать модель более робастной
    'random_state': 42,
    'n_jobs': -1
}

for fold, (train_index, val_index) in enumerate(kf.split(X_processed, y)):
    print(f"--- Фолд {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X_processed.iloc[train_index], X_processed.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model_rf = RandomForestRegressor(**params_rf)
    print("Обучение модели...")
    model_rf.fit(X_train, y_train)
    
    model_path = f'rf_models/rf_fold_{fold+1}.joblib'
    joblib.dump(model_rf, model_path)
    print(f"Модель сохранена в: {model_path}")
    
    val_preds = model_rf.predict(X_val)
    oof_preds_rf[val_index] = val_preds
    
    print("Предсказание на тестовом наборе...")
    test_predictions_rf.append(model_rf.predict(X_test_processed))

# --- Шаг 3: Пост-обработка и сохранение результатов ---
print("\nRandomForest Pipeline: 5. Обработка и сохранение результатов...")
final_oof_rmse_rf = np.sqrt(mean_squared_error(y, oof_preds_rf))
print(f"Итоговый OOF RMSE для RandomForest: {final_oof_rmse_rf:.5f}")

final_test_preds_rf = np.mean(test_predictions_rf, axis=0)
np.save('oof_preds_rf.npy', oof_preds_rf)
np.save('test_preds_rf.npy', final_test_preds_rf)
print("OOF и тестовые предсказания RandomForest сохранены в .npy файлы.")

submission_rf = pd.DataFrame({'id': test_ids, 'accident_risk': final_test_preds_rf})
submission_rf['accident_risk'] = submission_rf['accident_risk'].clip(0, 1)
submission_rf.to_csv('submission_rf_final.csv', index=False)
print("Файл 'submission_rf_final.csv' успешно создан!")



import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# --- Шаг 1: Загрузка всех необходимых данных ---
print("1. Загрузка OOF и тестовых предсказаний от 4 моделей...")

try:
    y_true = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')['accident_risk']
    test_ids = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')['id']

    oof_lgbm = np.load('oof_preds_lgbm.npy')
    oof_catboost = np.load('oof_preds_catboost.npy')
    oof_xgb = np.load('oof_preds_xgb.npy')
    oof_rf = np.load('oof_preds_rf.npy')

    test_lgbm = np.load('test_preds_lgbm.npy')
    test_catboost = np.load('test_preds_catboost.npy')
    test_xgb = np.load('test_preds_xgb.npy')
    test_rf = np.load('test_preds_rf.npy')

except FileNotFoundError as e:
    print(f"Ошибка: Не найден необходимый файл. Убедитесь, что все 12 файлов (.npy и .csv) находятся в папке.")
    print(f"Детали ошибки: {e}")
    exit()

print("Все файлы успешно загружены.")

# --- Шаг 2: Поиск оптимальных весов с помощью оптимизатора ---
print("\n2. Поиск оптимальных весов для ансамбля из 4 моделей...")

# Функция, которую мы будем минимизировать. Она принимает веса и возвращает RMSE.
def rmse_func(weights, oof_preds, true_labels):
    # Применяем softmax, чтобы веса всегда суммировались в 1
    weights = np.exp(weights) / np.sum(np.exp(weights))
    
    # Смешиваем предсказания
    final_prediction = np.zeros_like(true_labels, dtype=float)
    for weight, prediction in zip(weights, oof_preds):
        final_prediction += weight * prediction
    
    return np.sqrt(mean_squared_error(true_labels, final_prediction))

# Собираем все OOF-предсказания в один список
all_oof_preds = [oof_lgbm, oof_catboost, oof_xgb, oof_rf]
initial_weights = np.array([0.25] * len(all_oof_preds)) # Начальные веса (равные)

# Запускаем оптимизатор
result = minimize(
    fun=rmse_func,
    x0=initial_weights,
    args=(all_oof_preds, y_true),
    method='Nelder-Mead'
)

# Получаем оптимальные веса
best_weights_raw = result.x
best_weights = np.exp(best_weights_raw) / np.sum(np.exp(best_weights_raw))
best_rmse = result.fun

print("-" * 40)
print(f"Поиск завершен!")
print(f"Лучший локальный RMSE (ожидаемый): {best_rmse:.5f}")
print(f"Оптимальные веса (LGBM, CatBoost, XGBoost, RF):")
print(f"w_lgbm = {best_weights[0]:.3f}")
print(f"w_catboost = {best_weights[1]:.3f}")
print(f"w_xgb = {best_weights[2]:.3f}")
print(f"w_rf = {best_weights[3]:.3f}")
print("-" * 40)

# --- Шаг 3: Создание финального submission-файла ---
print("\n3. Создание финального submission-файла с оптимальными весами...")

final_predictions = (test_lgbm * best_weights[0]) + (test_catboost * best_weights[1]) + \
                    (test_xgb * best_weights[2]) + (test_rf * best_weights[3])

submission_ensemble = pd.DataFrame({'id': test_ids, 'accident_risk': final_predictions})
submission_ensemble['accident_risk'] = submission_ensemble['accident_risk'].clip(0, 1)
submission_ensemble.to_csv('submission_final_4model_ensemble.csv', index=False)

print("\nФайл 'submission_final_4model_ensemble.csv' успешно создан!")
print("Это ваш самый-самый сильный кандидат для отправки.")



print(1)


# =================================================================================
# FINAL SCRIPT 5 (Advanced): Finding Optimal Weights with a Linear Model (Stacking)
# =================================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

# --- Шаг 1: Загрузка всех необходимых данных ---
print("1. Загрузка OOF и тестовых предсказаний от 4 моделей...")

try:
    y_true = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')['accident_risk']
    test_ids = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')['id']

    oof_lgbm = np.load('oof_preds_lgbm.npy')
    oof_catboost = np.load('oof_preds_catboost.npy')
    oof_xgb = np.load('/kaggle/working/oof_preds_xgb_optuna.npy')
    # oof_rf = np.load('oof_preds_rf.npy')
    # oof_cnn = np.load('oof_preds_cnn.npy')
    # oof_mlp = np.load('oof_preds_mlp.npy')

    # test_cnn = np.load('test_preds_cnn.npy')
    test_lgbm = np.load('test_preds_lgbm.npy')
    test_catboost = np.load('test_preds_catboost.npy')
    test_xgb = np.load('/kaggle/working/test_preds_xgb_optuna.npy')
    # test_rf = np.load('test_preds_rf.npy')
    # test_mlp = np.load('test_preds_mlp.npy')
    
except FileNotFoundError as e:
    print(f"Ошибка: Не найден необходимый файл. Убедитесь, что все 12 файлов (.npy и .csv) находятся в папке.")
    print(f"Детали ошибки: {e}")
    exit()

print("Все файлы успешно загружены.")

# --- Шаг 2: Создание "мета-признаков" для обучения блендера ---

# Собираем OOF-предсказания в единую матрицу. Каждый столбец - это предсказания одной модели.
# X_oof = np.vstack([oof_lgbm, oof_catboost, oof_xgb, oof_rf, oof_cnn]).T

# Собираем тестовые предсказания в такую же матрицу
# X_test = np.vstack([test_lgbm, test_catboost, test_xgb, test_rf, test_cnn]).T

X_oof = np.vstack([oof_lgbm, oof_catboost, oof_xgb]).T
X_test = np.vstack([test_lgbm, test_catboost, test_xgb]).T
# --- Шаг 3: Обучение линейной модели (блендера) для поиска весов ---
print("\n2. Обучение Ridge-регрессии для поиска оптимальных весов...")

# Мы используем Ridge, а не простую LinearRegression, т.к. регуляризация делает веса более стабильными.
# fit_intercept=False означает, что мы не хотим добавлять свободный член, нас интересуют только веса моделей.
blender = Ridge(fit_intercept=False, random_state=42)
blender.fit(X_oof, y_true)

# Получаем найденные веса
best_weights = blender.coef_
oof_ensemble = blender.predict(X_oof)
best_rmse = np.sqrt(mean_squared_error(y_true, oof_ensemble))

print("-" * 40)
print(f"Поиск завершен!")
print(f"Лучший локальный RMSE (ожидаемый): {best_rmse:.5f}")
print(f"Оптимальные веса (LGBM, CatBoost, XGBoost, RF):")
print(f"w_lgbm = {best_weights[0]:.4f}")
print(f"w_catboost = {best_weights[1]:.4f}")
print(f"w_xgb = {best_weights[2]:.4f}")
# print(f"w_mlp = {best_weights[3]:.4f}")
# print(f"w_rf = {best_weights[3]:.4f}")
# print(f"cnn = {best_weights[4]:.4f}")
print(f"Сумма весов: {np.sum(best_weights):.4f}") # Интересно посмотреть, близка ли она к 1
print("-" * 40)

# --- Шаг 4: Создание финального submission-файла ---
print("\n3. Создание финального submission-файла с оптимальными весами...")

# Применяем обученный блендер к тестовым предсказаниям
final_predictions = blender.predict(X_test)

submission_ensemble = pd.DataFrame({'id': test_ids, 'accident_risk': final_predictions})
submission_ensemble['accident_risk'] = submission_ensemble['accident_risk'].clip(0, 1)
submission_ensemble.to_csv('submission_final_stacked_ensemble.csv', index=False)

print("\nФайл 'submission_final_stacked_ensemble.csv' успешно создан!")
print("Это ваш самый продвинутый кандидат для отправки.")


# =============================================================================
# ПОЛНЫЙ КОД: Ансамбль CatBoost через усреднение НЕСКОЛЬКИХ кросс-валидаций
# Максимально надежный, но вычислительно затратный подход.
# =============================================================================

import pandas as pd
import catboost as cb
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# --- Функция для создания признаков (Версия 2) ---
def create_features_v2(df):
    """Создает полный набор инженерных признаков."""
    df['weather_lighting_interaction'] = df['weather'].astype(str) + '_' + df['lighting'].astype(str)
    df['road_type_lanes_interaction'] = df['road_type'].astype(str) + '_lanes_' + df['num_lanes'].astype(str)
    curvature_stats = df.groupby('speed_limit')['curvature'].agg(['mean', 'std']).reset_index()
    curvature_stats.columns = ['speed_limit', 'curvature_by_speed_mean', 'curvature_by_speed_std']
    df = df.merge(curvature_stats, on='speed_limit', how='left')
    df['curvature_by_speed_std'].fillna(0, inplace=True)
    df['curvature_per_lane'] = df['curvature'] / (df['num_lanes'] + 1e-6)
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1e-6)
    df['curvature_freq'] = df.groupby('curvature')['curvature'].transform('count')
    df['lanes_freq'] = df.groupby('num_lanes')['num_lanes'].transform('count')
    df['curvature_sq'] = df['curvature'] ** 2
    return df

# --- Шаг 1: Подготовка данных (выполняется один раз) ---
print("Pipeline: 1. Загрузка и подготовка данных...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv').drop('id', axis=1)
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test_ids = test_df['id']
test_df = test_df.drop('id', axis=1)

X = train_df.drop('accident_risk', axis=1)
y = train_df['accident_risk']

combined_df = pd.concat([X, test_df], axis=0, ignore_index=True)
combined_df_featured = create_features_v2(combined_df)

split_point = len(X)
X_processed = combined_df_featured.iloc[:split_point]
X_test_processed = combined_df_featured.iloc[split_point:]

categorical_features = [
    'road_type', 'lighting', 'weather', 'time_of_day', 'road_signs_present', 
    'public_road', 'holiday', 'school_season', 'num_lanes', 'speed_limit',
    'weather_lighting_interaction', 'road_type_lanes_interaction'
]
for col in categorical_features:
    X_processed[col] = X_processed[col].astype(str)
    X_test_processed[col] = X_test_processed[col].astype(str)

# --- Шаг 2: Запуск нескольких сессий кросс-валидации ---
print("\nPipeline: 2. Запуск нескольких сессий кросс-валидации...")

# Список сидов для KFold. Каждый сид создаст уникальное разбиение на фолды.
SEEDS = [42, 101, 2023] 
N_SPLITS = 5

# Списки для хранения результатов КАЖДОГО полного прогона CV
all_oof_runs = []
all_test_runs = []

# Используем лучшие параметры, найденные ранее
best_params_catboost = {
    'learning_rate': 0.02570501026298664, 'depth': 9, 'l2_leaf_reg': 1.2403048474143457, 
    'bagging_temperature': 0.09783094822547894, 'random_strength': 0.6352174046846455,
    'iterations': 10000, 'loss_function': 'RMSE', 'eval_metric': 'RMSE', 'task_type': 'GPU',
    'verbose': 0, 'early_stopping_rounds': 100
}

# Внешний цикл по сидам
for seed in SEEDS:
    print(f"\n--- ЗАПУСК КРОСС-ВАЛИДАЦИИ С SEED = {seed} ---")
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    
    # Временные хранилища для текущего прогона CV
    oof_preds_this_run = np.zeros(split_point)
    test_predictions_this_run = []

    # Внутренний цикл по фолдам (как и раньше)
    for fold, (train_index, val_index) in enumerate(kf.split(X_processed, y)):
        print(f"  --- Обучение модели для фолда {fold+1}/{N_SPLITS} ---")
        
        X_train, X_val = X_processed.iloc[train_index], X_processed.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Сид для модели тоже делаем зависимым от внешнего сида и фолда
        model = cb.CatBoostRegressor(**best_params_catboost, random_seed=seed + fold)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=categorical_features)
        
        oof_preds_this_run[val_index] = model.predict(X_val)
        test_predictions_this_run.append(model.predict(X_test_processed))

    # --- Пост-обработка для текущего прогона CV ---
    # 1. Добавляем OOF-предсказания этого прогона в общий список
    all_oof_runs.append(oof_preds_this_run)
    
    # 2. Усредняем тестовые предсказания этого прогона и добавляем в общий список
    avg_test_preds_this_run = np.mean(test_predictions_this_run, axis=0)
    all_test_runs.append(avg_test_preds_this_run)
    
    # Опционально: выводим RMSE для этого конкретного сида
    run_rmse = np.sqrt(mean_squared_error(y, oof_preds_this_run))
    print(f"--- OOF RMSE для SEED = {seed}: {run_rmse:.5f} ---")


# --- Шаг 3: Финальное усреднение и сохранение ---
print(f"\nPipeline: 3. Финальное усреднение результатов по {len(SEEDS)} запускам...")

# Усредняем OOF-предсказания со всех прогонов
final_oof_preds = np.mean(all_oof_runs, axis=0)

# Усредняем тестовые предсказания со всех прогонов
final_test_preds = np.mean(all_test_runs, axis=0)

# Считаем итоговую, самую стабильную OOF-метрику
final_ensemble_rmse = np.sqrt(mean_squared_error(y, final_oof_preds))
print(f"\nИтоговый OOF RMSE для ансамбля (усреднение по {len(SEEDS)} CV-запускам): {final_ensemble_rmse:.5f}")

# Сохранение артефактов
np.save(f'oof_preds_catboost_{len(SEEDS)}_seeds.npy', final_oof_preds)
np.save(f'test_preds_catboost_{len(SEEDS)}_seeds.npy', final_test_preds)
print("Финальные OOF и тестовые предсказания ансамбля сохранены в .npy файлы.")

# Создание submission файла
submission = pd.DataFrame({'id': test_ids, 'accident_risk': final_test_preds})
submission['accident_risk'] = submission['accident_risk'].clip(0, 1)
submission.to_csv(f'submission_catboost_{len(SEEDS)}_seeds_ensemble.csv', index=False)
print(f"Файл 'submission_catboost_{len(SEEDS)}_seeds_ensemble.csv' успешно создан!")

print("\nВсе готово!")

