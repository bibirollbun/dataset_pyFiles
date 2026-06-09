# Импорт библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, RepeatedKFold


sns.set(style="whitegrid")  
%matplotlib inline

# Загрузка данных
train_df = pd.read_csv("/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip")
test_df  = pd.read_csv("/kaggle/input/mercedes-benz-greener-manufacturing/test.csv.zip")

# Выведем размерность датасетов и первые 5 строк обучающего набора
print(f"Размер train: {train_df.shape}")
print(f"Размер test: {test_df.shape}")
train_df.head(5)



# Распределение целевой переменной
plt.figure(figsize=(8,4))
sns.histplot(train_df['y'], bins=20, kde=True)
plt.xlabel('Значение целевой переменной (y), сек.')
plt.ylabel('Частота')
plt.title('Распределение целевой переменной y')
plt.show()

# Cтатистики целевой переменной
y_min = train_df['y'].min()
y_max = train_df['y'].max()
y_mean = train_df['y'].mean()
y_std = train_df['y'].std()
print(f"min: {y_min:.2f}, max: {y_max:.2f}, mean: {y_mean:.2f}, std: {y_std:.2f}")
print(f"Количество значений y > 180: {np.sum(train_df['y'] > 180)}")



plt.figure(figsize=(4,6))
sns.boxplot(y=train_df['y'])
plt.title('Распределение y')
plt.ylabel('y (секунды)')
plt.show()



# Проверка пропусков в данных
missing_train = train_df.isnull().sum().sum()
missing_test = test_df.isnull().sum().sum()
print(f"Пропущенные значения в train: {missing_train}, в test: {missing_test}")

# Определим категориальные признаки 
cat_cols = [col for col in train_df.columns if train_df[col].dtype == 'object']
print("Категориальные признаки:", cat_cols)
# Количество уникальных значений в каждом категориальном признаке
for col in cat_cols:
    print(f"{col}: {train_df[col].nunique()} уникальных значений")



# Поиск константных признаков (не включая ID и y)
constant_feats = [col for col in train_df.columns if col not in ['ID','y'] and train_df[col].nunique() == 1]
print("Константные признаки:", constant_feats)

# Сформируем список бинарных признаков (исключая ID, y, категориальные и константные)
binary_feats = [col for col in train_df.columns 
                if col not in (['ID','y'] + cat_cols) and col not in constant_feats]

# Посчитаем долю единиц для каждого бинарного признака
ones_fraction = train_df[binary_feats].mean()

# 5 признаков с минимальной долей единиц (большинство значений 0)
rare_feats = ones_fraction.nsmallest(5)
# 5 признаков с максимальной долей единиц (большинство значений 1)
freq_feats = ones_fraction.nlargest(5)

print("Константных признаков найдено:", len(constant_feats))

print("\nТОП-5 признаков с наименьшей долей единиц:")
for feat, frac in rare_feats.items():
    print(f"{feat}: доля 1 = {frac:.4f}")

print("\nТОП-5 признаков с наибольшей долей единиц:")
for feat, frac in freq_feats.items():
    print(f"{feat}: доля 1 = {frac:.4f}")



plt.figure(figsize=(6,4))
sns.histplot(ones_fraction, bins=20)
plt.xlabel('Доля единиц в бинарном признаке')
plt.ylabel('Количество признаков')
plt.title('Распределение долей единиц по бинарным признакам')
plt.show()



# Группируем данные по X0, вычисляем среднее и стандартное отклонение целевой переменной y
x0_stats = train_df.groupby("X0")["y"].agg(["mean", "std"]).reset_index()

plt.figure(figsize=(14, 6))
sns.barplot(x="X0", y="mean", data=x0_stats, palette="viridis", edgecolor='black', capsize=0.2)
plt.errorbar(x=range(len(x0_stats["X0"])), y=x0_stats["mean"], yerr=x0_stats["std"], fmt='none', c='black', capsize=5)
plt.xlabel("Категория X0")
plt.ylabel("Среднее значение y (сек)")
plt.title("Среднее значение y и разброс по категориям X0")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Список всех признаков, кроме исключая ID и y
features = [col for col in train_df.columns if col not in ['ID','y']]
# Исключим категориальные и константные признаки, оставим числовые 
numeric_feats = [col for col in features if col not in cat_cols + constant_feats]

# Корреляция числовых признаков с y
corr_with_y = train_df[numeric_feats].corrwith(train_df['y']).abs().sort_values(ascending=False)
print("Топ-5 признаков по корреляции с y:")
print(corr_with_y.head(5))



!pip install -U lightautoml
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Удаляем константные признаки перед созданием X_full
features = [col for col in train_df.columns if col not in ['ID','y'] and col not in constant_feats]

# Настраиваем задачу регрессии для LightAutoML
task = Task('reg', metric='r2')  

# Формируем полные данные без константных признаков
X_full = train_df[features]
y_full = train_df['y']

# Разделяем данные на обучающую и валидационную выборки (80/20)
X_train, X_valid, y_train, y_valid = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# Объединяем X_train и y_train для передачи в AutoML
train_data = X_train.copy()
train_data['y'] = y_train

# Указываем роли: целевой столбец и столбцы, которые нужно исключить из обучения
roles = {
    'target': 'y',
    'drop': []
}


# Конфигурация 1: базовый AutoML (таймаут 300 секунд)
automl_1 = TabularAutoML(task=task, timeout=300)
oof_pred_1 = automl_1.fit_predict(train_data, roles=roles)   
val_pred_1 = automl_1.predict(X_valid)                       
r2_1 = r2_score(y_valid, val_pred_1.data[:, 0])
print(f"Конфигурация 1 (timeout=300 сек) LightAutoML R² на валидации: {r2_1:.4f}")

# Конфигурация 2: расширенный AutoML (таймаут 600 секунд, используем 3 алгоритма)
automl_2 = TabularAutoML(
    task=task, 
    timeout=600, 
    general_params={"use_algos": [["lgb", "linear_l2", "cb"]]}
)
oof_pred_2 = automl_2.fit_predict(train_data, roles=roles)
val_pred_2 = automl_2.predict(X_valid)
r2_2 = r2_score(y_valid, val_pred_2.data[:, 0])
print(f"Конфигурация 2 (timeout=600 сек) LightAutoML R² на валидации: {r2_2:.4f}")



# Выбираем лучшую конфигурацию LAMA 
best_automl = automl_1
# Прогноз для тестового набора 
test_pred = best_automl.predict(test_df.drop(columns=['ID']))
# Формируем файл сабмита
submission_lama = pd.DataFrame({'ID': test_df['ID'], 'y': test_pred.data[:, 0]})
submission_lama.to_csv('submission_lama_final.csv', index=False)
print("Результаты AutoML на тестовом наборе сохранены в файл submission_lama.csv")



from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Списки признаков для трансформаций
categorical_features = cat_cols  
# Исключаем константные признаки из всех остальных
numeric_features = [col for col in features if col not in categorical_features and col not in constant_feats]

# Определяем трансформеры для столбцов и one-hot encoding
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')  

# Комбинируем трансформеры в ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)



from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

# Инициализация моделей с базовыми параметрами
rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(random_state=42, n_jobs=-1)
cat = CatBoostRegressor(verbose=0, random_state=42)

# Создаем пайплайны, объединяющие предобработку и модель
rf_pipeline = Pipeline([('preproc', preprocessor), ('model', rf)])
xgb_pipeline = Pipeline([('preproc', preprocessor), ('model', xgb)])
cat_pipeline = Pipeline([('preproc', preprocessor), ('model', cat)])

# Обучение моделей и предикт
rf_pipeline.fit(X_train, y_train)
xgb_pipeline.fit(X_train, y_train)
cat_pipeline.fit(X_train, y_train)

pred_rf  = rf_pipeline.predict(X_valid)
pred_xgb = xgb_pipeline.predict(X_valid)
pred_cat = cat_pipeline.predict(X_valid)

# Оценка метрики R^2 на валидационной выборке
r2_rf  = r2_score(y_valid, pred_rf)
r2_xgb = r2_score(y_valid, pred_xgb)
r2_cat = r2_score(y_valid, pred_cat)

print(f"Random Forest R² на валидации: {r2_rf:.4f}")
print(f"XGBoost R² на валидации: {r2_xgb:.4f}")
print(f"CatBoost R² на валидации: {r2_cat:.4f}")



from sklearn.model_selection import RandomizedSearchCV

# Параметры для XGB для поиска
param_dist = {
    'model__n_estimators': [100, 300, 500],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.1, 0.05, 0.01],
    'model__subsample': [0.7, 1.0],
    'model__colsample_bytree': [0.7, 1.0]
}

# RandomizedSearchCV для XGBoost 
xgb_search = RandomizedSearchCV(
    xgb_pipeline, 
    param_dist, 
    n_iter=10, 
    cv=5, 
    scoring='r2', 
    n_jobs=-1, 
    random_state=42, 
    verbose=1
)
xgb_search.fit(X_train, y_train)

print(f"Лучшие параметры XGBoost: {xgb_search.best_params_}")
print(f"Лучший CV R² (XGB): {xgb_search.best_score_:.4f}")

# Применим лучшую модель на валидации
best_xgb_model = xgb_search.best_estimator_
val_pred_xgb_tuned = best_xgb_model.predict(X_valid)
val_r2_xgb_tuned = r2_score(y_valid, val_pred_xgb_tuned)
print(f"XGBoost с подобранными параметрами R² на валидации: {val_r2_xgb_tuned:.4f}")


from sklearn.model_selection import GridSearchCV

rf_params = {
    'model__n_estimators': [100, 300],
    'model__max_depth': [None, 5, 10]
}
rf_search = GridSearchCV(rf_pipeline, rf_params, cv=5, scoring='r2', n_jobs=-1)
rf_search.fit(X_train, y_train)
print(f"Лучшие параметры RF: {rf_search.best_params_}, лучший CV R²: {rf_search.best_score_:.4f}")

best_rf_model = rf_search.best_estimator_
val_r2_rf_tuned = r2_score(y_valid, best_rf_model.predict(X_valid))
print(f"Random Forest с подобранными параметрами R² на валидации: {val_r2_rf_tuned:.4f}")



from sklearn.model_selection import RandomizedSearchCV

params_cat = {
    'model__iterations': [300, 500, 700],
    'model__learning_rate': [0.01, 0.03, 0.05],
    'model__depth': [4, 6, 8]
}

cat_search = RandomizedSearchCV(
    cat_pipeline, 
    params_cat, 
    n_iter=10, 
    cv=3,  
    scoring='r2', 
    random_state=42, 
    n_jobs=-1,
    verbose=1
)
cat_search.fit(X_train, y_train)
print("Лучшие параметры для CatBoost:", cat_search.best_params_)
print("Лучший CV R² для CatBoost:", cat_search.best_score_)

# Проверим качество подобранной модели на валидации
best_cat = cat_search.best_estimator_
pred_cat_tuned = best_cat.predict(X_valid)
r2_cat_tuned = r2_score(y_valid, pred_cat_tuned)
print(f"CatBoost (настроенный) R² на валидации: {r2_cat_tuned:.4f}")



# Ансамбль лучших моделей – усреднение прогнозов XGBoost и CatBoost
ensemble_pred = 0.5 * best_xgb_model.predict(X_valid) + 0.5 * best_cat.predict(X_valid)
ensemble_r2 = r2_score(y_valid, ensemble_pred)
print(f"Ансамбль XGBoost + CatBoost R² на валидации: {ensemble_r2:.4f}")



# Сначала подготовим очищенный train без константных признаков
train_clean = train_df.drop(columns=constant_feats)
X_full = train_clean.drop(columns=['ID', 'y'])
y_full = train_clean['y']

# Обучаем лучшие модели на полном наборе данных
best_xgb_model.fit(X_full, y_full)
best_cat.fit(X_full, y_full)

# Предсказания на тестовом наборе
X_test = test_df.drop(columns=['ID'] + constant_feats)  
preds_xgb = best_xgb_model.predict(X_test)
preds_cat = best_cat.predict(X_test)

# Ансамбль (усреднение)
final_predictions = 0.5 * preds_xgb + 0.5 * preds_cat

# Формируем сабмишен
submission_ensemble = pd.DataFrame({
    'ID': test_df['ID'],
    'y': final_predictions
})
submission_ensemble.to_csv('submission_ensemble.csv', index=False)
print("Файл submission_ensemble.csv сохранен.")



from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.base import clone
from tqdm import tqdm
from scipy.stats import ttest_rel


def prepare_dataset_from_df(df):
    """
    Подготавливает датасет на основе уже загруженного DataFrame.
    Удаляет столбец ID, отделяет целевую переменную и выполняет one-hot encoding для категориальных признаков.
    
    Параметры:
        df (pd.DataFrame): исходный датасет.
        
    Возвращает:
        X (np.ndarray): массив признаков.
        y (np.ndarray): целевая переменная.
    """
    df_copy = df.copy()
    df_copy = df_copy.drop("ID", axis=1)
    y = df_copy.pop("y").values

    # Выбираем числовые признаки
    X_num = df_copy.select_dtypes(include="number")
    # Выбираем категориальные признаки и выполняем one-hot encoding
    X_cat = df_copy.select_dtypes(exclude="number")
    X_cat = pd.get_dummies(X_cat)
    # Объединяем признаки и заполняем пропуски
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.fillna(0).values
    return X, y


def cross_val_score(model, X, y, cv, params_list, scoring, random_state=42, show_progress=False):
    """
    Вычисляет метрики кросс-валидации для набора моделей с разными параметрами.
    
    Параметры:
        model: базовая модель (например, RandomForestRegressor).
        X (np.ndarray): массив признаков.
        y (np.ndarray): целевая переменная.
        cv: число фолдов или кортеж (n_folds, n_repeats).
        params_list (List[Dict]): список параметров для модели.
        scoring: функция оценки (например, r2_score).
        random_state (int): random_state для кросс-валидации.
        show_progress (bool): вывод прогресса с помощью tqdm.
        
    Возвращает:
        np.ndarray: матрица метрик размером [n_models x n_splits].
    """
    if isinstance(cv, tuple):
        n_folds, n_repeats = cv
        cv_generator = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
        total_splits = n_folds * n_repeats
    else:
        cv_generator = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        total_splits = cv

    n_models = len(params_list)
    metrics = np.zeros((n_models, total_splits))
    iterator = tqdm(enumerate(params_list), total=n_models, desc="Models") if show_progress else enumerate(params_list)

    for i, params in iterator:
        fold_scores = []
        for train_idx, test_idx in cv_generator.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model_clone = clone(model)
            model_clone.set_params(**params)

            # Обучение на log1p таргете и обратное преобразование (expm1)
            model_clone.fit(X_train, np.log1p(y_train))
            y_pred = np.expm1(model_clone.predict(X_test))
            score = scoring(y_test, y_pred)
            fold_scores.append(score)

        metrics[i, :] = fold_scores

    return metrics


def compare_models(cv, model, params_list, X, y, random_state=42, alpha=0.05, show_progress=False):
    """
    Сравнивает модели с использованием кросс-валидации и парного t-теста.
    
    Параметры:
        cv: число фолдов или кортеж (n_folds, n_repeats).
        model: базовая модель (например, RandomForestRegressor).
        params_list (List[Dict]): список параметров для модели.
        X (np.ndarray): массив признаков.
        y (np.ndarray): целевая переменная.
        random_state (int): random_state для кросс-валидации.
        alpha (float): уровень значимости для t-теста.
        show_progress (bool): вывод прогресса с помощью tqdm.
        
    Возвращает:
        List[Dict]: список результатов сравнения моделей (для каждой модели, кроме бейзлайна),
        содержащий model_index, avg_score, p_value и effect_sign.
    """
    metrics = cross_val_score(
        model=model,
        X=X,
        y=y,
        cv=cv,
        params_list=params_list,
        scoring=r2_score,
        random_state=random_state,
        show_progress=show_progress,
    )
    avg_scores = metrics.mean(axis=1)
    baseline_scores = metrics[0, :]
    results = []

    for i in range(1, len(params_list)):
        current_scores = metrics[i, :]
        avg_score = avg_scores[i]
        t_stat, p_value = ttest_rel(baseline_scores, current_scores)
        effect_sign = 1 if (p_value < alpha and avg_score > avg_scores[0]) else (-1 if (p_value < alpha and avg_score < avg_scores[0]) else 0)
        results.append({
            "model_index": i,
            "avg_score": avg_score,
            "p_value": p_value,
            "effect_sign": effect_sign,
        })

    return sorted(results, key=lambda x: x["avg_score"], reverse=True)


# Используем уже загруженный train_df для подготовки данных
X_alt, y_alt = prepare_dataset_from_df(train_df)

# Задаем параметры моделей (базовая модель и варианты с разным max_depth)
params_list = [
    {"max_depth": 10},  # бейзлайн
    {"max_depth": 2},
    {"max_depth": 3},
    {"max_depth": 4},
    {"max_depth": 5},
    {"max_depth": 9},
    {"max_depth": 11},
    {"max_depth": 12},
    {"max_depth": 15},
]

# Определяем модель
model_alt = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)

# Сравнение моделей с использованием KFold (5 фолдов)
result_kfold_alt = compare_models(
    cv=5,
    model=model_alt,
    params_list=params_list,
    X=X_alt,
    y=y_alt,
    random_state=42,
    alpha=0.05,
    show_progress=False,
)
print("Результаты для KFold:")
print(pd.DataFrame(result_kfold_alt))

# Сравнение моделей с использованием RepeatedKFold (5 фолдов, 3 повтора)
result_rkfold_alt = compare_models(
    cv=(5, 3),
    model=model_alt,
    params_list=params_list,
    X=X_alt,
    y=y_alt,
    random_state=42,
    alpha=0.05,
    show_progress=False,
)
print("\nРезультаты для RepeatedKFold (5 фолдов, 3 повтора):")
print(pd.DataFrame(result_rkfold_alt))





