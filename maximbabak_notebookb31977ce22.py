import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ML библиотеки
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, ndcg_score
import lightgbm as lgb

# Визуализация
import matplotlib.pyplot as plt

# Утилиты
import os
import glob

# =============================================================================
# КОНСТАНТЫ И НАСТРОЙКИ
# =============================================================================

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============================================================================
# ФУНКЦИИ МЕТРИК
# =============================================================================

def compute_ndcg_per_session(relevance_scores, predicted_scores, k=20):
    """Вычисляет NDCG@K для одной сессии"""
    if len(relevance_scores) == 0 or len(predicted_scores) == 0:
        return 0.0
    
    min_len = min(len(relevance_scores), len(predicted_scores))
    relevance_scores = np.array(relevance_scores[:min_len])
    predicted_scores = np.array(predicted_scores[:min_len])
    
    y_true = relevance_scores.reshape(1, -1)
    y_score = predicted_scores.reshape(1, -1)
    
    return ndcg_score(y_true, y_score, k=k)

# =============================================================================
# ПРЕДОБРАБОТКА ДАННЫХ
# =============================================================================

def create_session_id(df):
    """Создает session_id из user_number и search_number"""
    df = df.copy()
    df['session_id'] = df['user_number'].astype(str) + '_' + df['search_number'].astype(str)
    return df

def preprocess_data(df):
    """Предобработка данных"""
    df = df.copy()
    
    # Создаем session_id
    df = create_session_id(df)
    
    # Обработка NaN и Inf значений
    if 'relevance' in df.columns:
        df['relevance'] = df['relevance'].fillna(0)
        df['relevance'] = df['relevance'].replace([np.inf, -np.inf], 0)
    
    return df

# =============================================================================
# ГЕНЕРАЦИЯ ФИЧЕЙ (УПРОЩЕННАЯ)
# =============================================================================

def generate_features(df):
    """Генерирует основные фичи"""
    df = df.copy()
    
    # Основные фичи
    df['cnt_products_in_session'] = df.groupby('session_id')['product_id'].transform('count')
    df['position_in_session'] = df.groupby('session_id')['product_position'].rank(method='dense')
    
    # Популярность товара
    product_views = df.groupby('product_id').size()
    df['product_popularity'] = df['product_id'].map(product_views)
    df['log_product_popularity'] = np.log1p(df['product_popularity'])
    
    # Фичи цены
    if 'price' in df.columns:
        df['log_price'] = np.log1p(df['price'])
        df['price_rank_in_session'] = df.groupby('session_id')['price'].rank(method='dense')
    
    # Фичи скидки
    if 'price_discount' in df.columns:
        df['has_discount'] = (df['price_discount'] > 0).astype(int)
    
    # Фичи устройства
    if 'device' in df.columns:
        df['is_mobile'] = (df['device'] == 'mobile').astype(int)
    
    # Временные фичи
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df

# =============================================================================
# ЗАГРУЗКА ДАННЫХ
# =============================================================================

def load_data():
    """Загружает данные соревнования"""
    csv_files = glob.glob('/kaggle/input/**/*.csv', recursive=True)
    
    train_file = None
    test_file = None
    
    for file in csv_files:
        filename = file.lower().split('/')[-1]
        if 'train' in filename:
            train_file = file
        elif 'test' in filename:
            test_file = file
    
    if train_file and test_file:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        print(f"✅ Данные загружены: Train {train_df.shape}, Test {test_df.shape}")
        return train_df, test_df
    else:
        print("❌ Файлы не найдены")
        return None, None

# =============================================================================
# ОСНОВНОЙ ПАЙПЛАЙН
# =============================================================================

def main():
    """Основная функция - БЫСТРАЯ ВЕРСИЯ"""
    
    print("="*60)
    print("РЕШЕНИЕ ЗАДАЧИ РАНЖИРОВАНИЯ 21VEK.BY (БЫСТРАЯ ВЕРСИЯ)")
    print("="*60)
    
    # 1. ЗАГРУЗКА ДАННЫХ
    print("\n1. ЗАГРУЗКА ДАННЫХ")
    print("-" * 30)
    
    train_df, test_df = load_data()
    if train_df is None:
        return
    
    # Берем только часть данных для скорости
    print("Используем только 10% данных для скорости...")
    train_df = train_df.sample(frac=0.1, random_state=RANDOM_STATE)
    print(f"Train данные (10%): {train_df.shape}")
    
    # 2. ПРЕДОБРАБОТКА
    print("\n2. ПРЕДОБРАБОТКА ДАННЫХ")
    print("-" * 30)
    
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)
    
    # 3. ГЕНЕРАЦИЯ ФИЧЕЙ
    print("\n3. ГЕНЕРАЦИЯ ФИЧЕЙ")
    print("-" * 30)
    
    train_df = generate_features(train_df)
    test_df = generate_features(test_df)
    
    # Получаем фичи
    feature_columns = [col for col in train_df.columns 
                      if col not in ['session_id', 'product_id', 'relevance', 'date', 'query', 
                                   'user_number', 'search_number', 'device']]
    
    print(f"Количество фичей: {len(feature_columns)}")
    
    # 4. ОБУЧЕНИЕ МОДЕЛИ
    print("\n4. ОБУЧЕНИЕ МОДЕЛИ")
    print("-" * 30)
    
    # Простая модель для скорости
    model = lgb.LGBMRegressor(
        n_estimators=100,  # Меньше итераций
        learning_rate=0.1,
        max_depth=4,       # Меньше глубина
        random_state=RANDOM_STATE,
        verbose=-1
    )
    
    X_train = train_df[feature_columns].values
    y_train = train_df['relevance'].values
    
    print("Обучаем модель...")
    model.fit(X_train, y_train)
    print("✅ Модель обучена!")
    
    # 5. ПРЕДСКАЗАНИЯ
    print("\n5. ПРЕДСКАЗАНИЯ")
    print("-" * 30)
    
    X_test = test_df[feature_columns].values
    test_predictions = model.predict(X_test)
    
    print(f"Сгенерировано {len(test_predictions)} предсказаний")
    
    # 6. СОЗДАНИЕ SUBMISSION
    print("\n6. СОЗДАНИЕ SUBMISSION")
    print("-" * 30)
    
    # Создаем submission
    submission_df = test_df[['session_id', 'product_id']].copy()
    submission_df['session_product_id'] = (
        submission_df['session_id'].astype(str) + '_' + 
        submission_df['product_id'].astype(str)
    )
    submission_df['relevance'] = test_predictions
    
    # Нормализуем предсказания внутри сессий
    submission_df['relevance'] = submission_df.groupby('session_id')['relevance'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    )
    
    # Финальный submission
    final_submission = submission_df[['session_product_id', 'relevance']].copy()
    final_submission = final_submission.sort_values('session_product_id')
    
    # Сохраняем
    final_submission.to_csv('submission.csv', index=False)
    
    print(f"✅ Submission сохранен: submission.csv")
    print(f"Размер: {final_submission.shape}")
    print(f"Пример:")
    print(final_submission.head(10))
    
    # 7. АНАЛИЗ РЕЗУЛЬТАТОВ
    print("\n7. АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("-" * 30)
    
    print(f"Статистики предсказаний:")
    print(f"  Среднее: {final_submission['relevance'].mean():.4f}")
    print(f"  Медиана: {final_submission['relevance'].median():.4f}")
    print(f"  Минимум: {final_submission['relevance'].min():.4f}")
    print(f"  Максимум: {final_submission['relevance'].max():.4f}")
    
    # Важность фичей
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nТоп-10 важных фичей:")
        print(feature_importance.head(10))
    
    print("\n" + "="*60)
    print("ЗАДАЧА ВЫПОЛНЕНА УСПЕШНО! (БЫСТРАЯ ВЕРСИЯ)")
    print("="*60)
    print("Файл submission.csv готов для отправки!")

# =============================================================================
# ЗАПУСК
# =============================================================================

if __name__ == "__main__":
    main()


