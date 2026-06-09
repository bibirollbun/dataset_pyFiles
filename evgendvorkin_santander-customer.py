import pandas as pd  # Для работы с табличными данными (чтение/запись CSV, манипуляции с DataFrame)
import lightgbm as lgb  # Библиотека LightGBM — быстрый градиентный бустинг на деревьях
import xgboost as xgb  # Библиотека XGBoost — ещё один градиентный бустинг
from catboost import CatBoostClassifier  # Библиотека CatBoost — градиентный бустинг от Яндекса
from sklearn.metrics import roc_auc_score  # Метрика AUC ROC для оценки качества классификации
from sklearn.model_selection import train_test_split  # Разделение данных на train/val


# 1. Загрузка данных (как в оригинале)
train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')  # Читаем обучающий набор
test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')  # Читаем тестовый набор
sample_submission = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')  # Шаблон сабмишена

# 2. Подготовка данных
X = train.drop('target', axis=1)  # X — все признаки, кроме целевой переменной
y = train['target']  # y — целевая переменная (0 или 1)
X_test = test  # Тестовые данные без целевой переменной (её нужно предсказать)

# 3. Разбиение
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,  # 20% данных — валидация
    random_state=42,  # Фиксируем случайность для воспроизводимости
    stratify=y  # Сохраняем долю классов 0/1 в train и val
)

# 4. Удаление ID_code
X_train_clean = X_train.drop('ID_code', axis=1)  # Удаляем ID — он не несёт информации для модели
X_val_clean = X_val.drop('ID_code', axis=1)  # То же для валидации
X_test_clean = X_test.drop('ID_code', axis=1)  # То же для теста


# 5. Модель LightGBM (как было)
lgb_model = lgb.LGBMClassifier(
    n_estimators=600,              # Кол-во деревьев
    learning_rate=0.1,             # Шаг обучения
    max_depth=3,                 # Глубина деревьев
    num_leaves=8,               # Макс. листьев в дереве
    subsample=0.8,             # Доля объектов для обучения каждого дерева
    colsample_bytree=0.8,     # Доля признаков для каждого дерева
    random_state=42,           # Воспроизводимость
    verbose=-1,                # Меньше логов
    scale_pos_weight=2.0       # Вес класса 1 (корректный: N_0 / N_1)
)

# Callback для ранней остановки в LightGBM
lgb_callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]


# Обучение LightGBM
lgb_model.fit(
    X_train_clean, y_train,
    eval_set=[(X_val_clean, y_val)],
    callbacks=lgb_callbacks
)

# Прогноз и оценка LightGBM
y_val_lgb = lgb_model.predict_proba(X_val_clean)[:, 1]
print(f"AUC LightGBM на валидации: {roc_auc_score(y_val, y_val_lgb):.4f}")


# 6. Модель XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
    scale_pos_weight=8.95,
    early_stopping_rounds=50
)


# Обучение XGBoost
xgb_model.fit(
    X_train_clean, y_train,
    eval_set=[(X_val_clean, y_val)],
    verbose=False
)

# Прогноз и оценка XGBoost
y_val_xgb = xgb_model.predict_proba(X_val_clean)[:, 1]
print(f"AUC XGBoost на валидации: {roc_auc_score(y_val, y_val_xgb):.4f}")

# 7. Модель CatBoost
cat_model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    subsample=0.8,
    colsample_bylevel=0.8,
    random_state=42,
    verbose=0,
    scale_pos_weight=8.95,
    early_stopping_rounds=50
)

# Обучение CatBoost
cat_model.fit(
    X_train_clean, y_train,
    eval_set=(X_val_clean, y_val),
    verbose=False
)

# Прогноз и оценка CatBoost
y_val_cat = cat_model.predict_proba(X_val_clean)[:, 1]
print(f"AUC CatBoost на валидации: {roc_auc_score(y_val, y_val_cat):.4f}")


# 8. Ансамбль (усреднение прогнозов трёх моделей)
y_val_ensemble = (y_val_lgb + y_val_xgb + y_val_cat) / 3
print(f"AUC ансамбля на валидации: {roc_auc_score(y_val, y_val_ensemble):.4f}")

# 9. Сабмишен для каждой модели и ансамбля
# LightGBM
y_test_lgb = lgb_model.predict_proba(X_test_clean)[:, 1]
submission_lgb = sample_submission.copy()
submission_lgb['target'] = y_test_lgb
submission_lgb.to_csv('submission_lgb.csv', index=False)

# XGBoost
y_test_xgb = xgb_model.predict_proba(X_test_clean)[:, 1]
submission_xgb = sample_submission.copy()
submission_xgb['target'] = y_test_xgb
submission_xgb.to_csv('submission_xgb.csv', index=False)

# CatBoost
y_test_cat = cat_model.predict_proba(X_test_clean)[:, 1]
submission_cat = sample_submission.copy()
submission_cat['target'] = y_test_cat
submission_cat.to_csv('submission_cat.csv', index=False)

# Ансамбль
y_test_ensemble = (y_test_lgb + y_test_xgb + y_test_cat) / 3
submission_ensemble = sample_submission.copy()
submission_ensemble['target'] = y_test_ensemble
submission_ensemble.to_csv('submission_ensemble.csv', index=False)

print("Сабмишены сохранены: submission_lgb.csv, submission_xgb.csv, submission_cat.csv, submission_ensemble.csv")



import pandas as pd  # Для работы с табличными данными (чтение/запись CSV, манипуляции с DataFrame)
import lightgbm as lgb  # Библиотека LightGBM — быстрый градиентный бустинг на деревьях
import xgboost as xgb  # Библиотека XGBoost — ещё один градиентный бустинг
from catboost import CatBoostClassifier  # Библиотека CatBoost — градиентный бустинг от Яндекса
from sklearn.metrics import roc_auc_score  # Метрика AUC ROC для оценки качества классификации
from sklearn.model_selection import train_test_split  # Разделение данных на train/val


# 1. Загрузка данных (как в оригинале)
train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')  # Читаем обучающий набор
test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')  # Читаем тестовый набор
sample_submission = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')  # Шаблон сабмишена

# 2. Подготовка данных
X = train.drop('target', axis=1)  # X — все признаки, кроме целевой переменной
y = train['target']  # y — целевая переменная (0 или 1)
X_test = test  # Тестовые данные без целевой переменной (её нужно предсказать)

# 3. Разбиение
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,  # 20% данных — валидация
    random_state=42,  # Фиксируем случайность для воспроизводимости
    stratify=y  # Сохраняем долю классов 0/1 в train и val
)

# 4. Удаление ID_code
X_train_clean = X_train.drop('ID_code', axis=1)  # Удаляем ID — он не несёт информации для модели
X_val_clean = X_val.drop('ID_code', axis=1)  # То же для валидации
X_test_clean = X_test.drop('ID_code', axis=1)  # То же для теста


lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.07,
    max_depth=5,
    num_leaves=16,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    verbose=-1,
    scale_pos_weight=8.95,
    is_unbalance=False,
    early_stopping_rounds=15,
    eval_metric='auc',
    reg_lambda=0.1,
    min_child_samples=5,
    min_split_gain=0.0
)



lgb_model.fit(
    X_train_clean, y_train,
    eval_set=[(X_val_clean, y_val)],
)



# Прогноз и оценка LightGBM
y_val_lgb = lgb_model.predict_proba(X_val_clean)[:, 1]
print(f"AUC LightGBM на валидации: {roc_auc_score(y_val, y_val_lgb):.4f}")


# 6. Модель XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
    scale_pos_weight=8.95,
    early_stopping_rounds=50
)


# Обучение XGBoost
xgb_model.fit(
    X_train_clean, y_train,
    eval_set=[(X_val_clean, y_val)],
    verbose=False
)

# Прогноз и оценка XGBoost
y_val_xgb = xgb_model.predict_proba(X_val_clean)[:, 1]
print(f"AUC XGBoost на валидации: {roc_auc_score(y_val, y_val_xgb):.4f}")

# 7. Модель CatBoost
cat_model = CatBoostClassifier(
    iterations=2500,
    learning_rate=0.03,
    depth=6,
    subsample=0.8,
    colsample_bylevel=0.8,
    random_state=42,
    verbose=0,
    scale_pos_weight=8.95,
    early_stopping_rounds=50
)

# Обучение CatBoost
cat_model.fit(
    X_train_clean, y_train,
    eval_set=(X_val_clean, y_val),
    verbose=False
)

# Прогноз и оценка CatBoost
y_val_cat = cat_model.predict_proba(X_val_clean)[:, 1]
print(f"AUC CatBoost на валидации: {roc_auc_score(y_val, y_val_cat):.4f}")


# 8. Ансамбль (усреднение прогнозов трёх моделей)
y_val_ensemble = (y_val_lgb + y_val_xgb + y_val_cat) / 3
print(f"AUC ансамбля на валидации: {roc_auc_score(y_val, y_val_ensemble):.4f}")

# 9. Сабмишен для каждой модели и ансамбля
# LightGBM
y_test_lgb = lgb_model.predict_proba(X_test_clean)[:, 1]
submission_lgb = sample_submission.copy()
submission_lgb['target'] = y_test_lgb
submission_lgb.to_csv('submission_lgb.csv', index=False)

# XGBoost
y_test_xgb = xgb_model.predict_proba(X_test_clean)[:, 1]
submission_xgb = sample_submission.copy()
submission_xgb['target'] = y_test_xgb
submission_xgb.to_csv('submission_xgb.csv', index=False)

# CatBoost
y_test_cat = cat_model.predict_proba(X_test_clean)[:, 1]
submission_cat = sample_submission.copy()
submission_cat['target'] = y_test_cat
submission_cat.to_csv('submission_cat.csv', index=False)

# Ансамбль
y_test_ensemble = (y_test_lgb + y_test_xgb + y_test_cat) / 3
submission_ensemble = sample_submission.copy()
submission_ensemble['target'] = y_test_ensemble
submission_ensemble.to_csv('submission_ensemble.csv', index=False)

print("Сабмишены сохранены: submission_lgb.csv, submission_xgb.csv, submission_cat.csv, submission_ensemble.csv")



from scipy.optimize import minimize

# Прогнозы на валидации
y_val_xgb = xgb_model.predict_proba(X_val_clean)[:, 1]
y_val_cat = cat_model.predict_proba(X_val_clean)[:, 1]


# Функция для оптимизации весов
def weighted_auc(weights):
    y_pred = weights[0] * y_val_xgb + weights[1] * y_val_cat
    return -roc_auc_score(y_val, y_pred)  # минус, т.к. minimize ищет минимум


# Оптимизация
result = minimize(
    weighted_auc,
    x0=[0.5, 0.5],
    method='SLSQP',
    bounds=[(0, 1), (0, 1)],
    constraints={'type': 'eq', 'fun': lambda w: 1 - sum(w)}
)

optimal_weights = result.x
print(f"Оптимальные веса: XGBoost={optimal_weights[0]:.3f}, CatBoost={optimal_weights[1]:.3f}")


# Финальный прогноз на валидации
y_val_ensemble = optimal_weights[0] * y_val_xgb + optimal_weights[1] * y_val_cat
print(f"AUC ансамбля (взвешенный): {roc_auc_score(y_val, y_val_ensemble):.4f}")



import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


# Тренировочные "признаки" — прогнозы моделей на валидации
X_stack_train = np.column_stack([y_val_xgb, y_val_cat])

y_stack_train = y_val  # истинные метки

# Тестовые "признаки" — прогнозы на test
X_stack_test = np.column_stack([
    xgb_model.predict_proba(X_test_clean)[:, 1],
    cat_model.predict_proba(X_test_clean)[:, 1]
])

# Мета‑модель с калибровкой (улучшает качество вероятностей)
meta_model = CalibratedClassifierCV(
    LogisticRegression(random_state=42),
    cv=3,
    method='isotonic'
)

# Обучение мета‑модели
meta_model.fit(X_stack_train, y_stack_train)

# Прогноз на валидации
y_val_stacked = meta_model.predict_proba(X_stack_train)[:, 1]
print(f"AUC стэкинга: {roc_auc_score(y_val, y_val_stacked):.4f}")


# Прогноз на тесте
y_test_stacked = meta_model.predict_proba(X_stack_test)[:, 1]



# Сабмишен для XGBoost
submission_xgb = sample_submission.copy()
submission_xgb['target'] = xgb_model.predict_proba(X_test_clean)[:, 1]
submission_xgb.to_csv('submission_xgb.csv', index=False)


# Сабмишен для CatBoost
submission_cat = sample_submission.copy()
submission_cat['target'] = cat_model.predict_proba(X_test_clean)[:, 1]
submission_cat.to_csv('submission_cat.csv', index=False)


# Сабмишен для стэкинга
submission_stacked = sample_submission.copy()
submission_stacked['target'] = y_test_stacked
submission_stacked.to_csv('submission.csv', index=False)


print("Сабмишены сохранены:")
print("- submission_xgb.csv")
print("- submission_cat.csv")
print("- submission.csv")


