!pip3 install rapidfuzz


import os
import re
import sys
import time
import shap
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from rapidfuzz import fuzz, process
from catboost import CatBoostRanker, Pool
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, jaccard_score
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.feature_extraction.text import CountVectorizer





train_df = pd.read_csv('/kaggle/input/21-vek-by-searched-products-ranking/train.csv')
train_df.head(3)


test_df = pd.read_csv('/kaggle/input/21-vek-by-searched-products-ranking/test.csv')
test_df.head(3)


product_df = pd.read_csv('/kaggle/input/21-vek-by-searched-products-ranking/products.csv')
product_df.head(3)


def data_validation(df, dataset_name):
    print(f"{dataset_name.upper()}")
    print(f"Размеры данных: {df.shape[0]:_} строк × {df.shape[1]} колонок")
    print(f"Колличество пропусков в данных: {df.isnull().sum().sum()}")
    print(f"Колличество дубликатов в данных: {df.duplicated().sum()}")

data_validation(train_df, 'train')
data_validation(test_df, 'test')
data_validation(product_df, 'product')


product_df = product_df.drop_duplicates()


train_df.info()


# Анализируем распределение ключевых числовых фич
numeric_cols = ['price', 'price_discount', 'browse_time', 'response_time', 'response_products', 'product_position']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, col in enumerate(numeric_cols):
    if col in train_df.columns:
        axes[i].boxplot(train_df[col].dropna())
        axes[i].set_title(f"Boxplot {col}")
        axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()


browse_upper = train_df['browse_time'].quantile(0.9999)
train_df.drop(train_df[train_df['browse_time'] > browse_upper].index, inplace=True)


price_upper = 20_000_00
train_df.drop(train_df[train_df['price'] > price_upper].index, inplace=True)


product_position_upper = train_df['product_position'].quantile(0.99)
train_df.drop(train_df[train_df['product_position'] > product_position_upper].index, inplace=True)


train_df = train_df.merge(product_df, on='product_id', how='left')
test_df = test_df.merge(product_df, on='product_id', how='left')


def add_session_features(df):

    df = df.merge(df.groupby('search_number')
          .agg(session_price_mean=('price', 'mean'),
               session_price_std=('price', 'std'),
               session_size=('price', 'count'))
          .round(4).reset_index()
    )

    df = df.merge(df.groupby('product_id')
          .agg(product_search_count=('search_number', 'nunique'))
          .reset_index(), on='product_id'
    )
    
    return df

train_df = add_session_features(train_df)
test_df = add_session_features(test_df)


def add_general_features(df):
    avg_session_size = train_df['session_size'].mean()
    avg_session_time = train_df['browse_time'].mean()
    
    # время на товар
    df['time_per_product'] = df['browse_time'] / df['session_size']

    # процентный ранг позиции товара внутри каждой поисковой сессии (чем меньше, тем товар выше)
    df['position_rank'] = df.groupby('search_number')['product_position'].rank(pct=True)

    # возраст продукта
    df['product_age_days'] = (pd.to_datetime('today') - pd.to_datetime(df['date_of_create'])).dt.days.astype(int)

    # насколько товар дороже/дешевле средней цены в сессии
    df['price_vs_session_mean'] = df['price'] / df['session_price_mean']

    # зависимость позиции от времени
    df['rank_weighted_engagement'] = df['browse_time'] * df['position_rank']
    
    # коэффициент вариации для цен в сессии
    df['price_cv'] = df['session_price_std'] / df['session_price_mean']

    # продуктивность
    df['composite_score'] = (df['browse_time'] / avg_session_time) * (df['session_size'] / avg_session_size)

    # популярность товаров
    df = df.merge(df.groupby('product_id').size().reset_index(name='product_click_count'), on='product_id')
    
    # логарифмированная популярность товара (сглаживает выбросы)
    df['product_popularity_log'] = np.log1p(df['product_click_count'])

    # показатель кликабельности
    df['product_ctr'] = df['product_click_count'] / df['product_search_count']
        
    for col in ['price', 'price_discount', 'browse_time', 'response_time']:
        df[col] = np.log1p(df[col])

    return df

train_df = add_general_features(train_df)
test_df = add_general_features(test_df)


def add_query_text_features(df):
    tqdm.pandas()
    
    # очистка текста от лишнего
    def clean_text(text):
        if pd.isna(text):
            return ''
        text = text.lower()
        text = re.sub(r'[^a-zа-я0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # коэффициент Жаккара (мера схожести множеств слов)
    def jaccard_similarity(q, n):
        q_words = set(q.split())
        n_words = set(n.split())
        if not q_words or not n_words:
            return 0
        return len(q_words & n_words) / len(q_words | n_words)
        
    df['query_clean'] = df['query'].progress_apply(clean_text)
    df['model_clean'] = df['model'].progress_apply(clean_text)

    # длина и количество слов
    df['query_len'] = df['query_clean'].str.len()

    df['model_token_jaccard'] = df.progress_apply(lambda x: jaccard_similarity(x['query_clean'], x['model_clean']), axis=1)

    return df

train_df = add_query_text_features(train_df)
test_df = add_query_text_features(test_df)


# проверяем наличие GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используется устройство: {device}")
cache_dir = Path('/kaggle/input/embeddings-cache/')
cache_dir.mkdir(exist_ok=True)

# инициализация модели
model_sbert = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)

# Функция для батчевой векторизации
def get_embeddings(texts, batch_size=256):
    vectors = []
    texts_list = texts.fillna('').tolist()
    
    for i in tqdm(range(0, len(texts_list), batch_size), desc='Генерация эмбеддингов'):
        batch = texts_list[i:i + batch_size]
        vecs = model_sbert.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        vectors.append(vecs)
    
    return np.vstack(vectors)

# загрузка или вычисление эмбеддингов
def load_or_compute_embeddings(df, column_name, dataset_type):
    file_path = cache_dir / f"{dataset_type}_{column_name}.npy"
    
    if file_path.exists():
        print(f"Загружаем кэш: {file_path.name}")
        return np.load(file_path)
    
    print(f"Генерируем эмбеддинги для {column_name}...")
    vectors = get_embeddings(df[column_name])
    np.save(file_path, vectors.astype(np.float16))
    print(f"Сохранено: {file_path.name}")
    
    return vectors

# вычисление косинусного сходства
def compute_cosine_similarity(a, b, batch_size=100000):

    assert len(a) == len(b), 'Массивы должны быть одинаковой длины!'
    
    n = len(a)
    results = np.empty(n, dtype=np.float32)
    
    for i in tqdm(range(0, n, batch_size), desc='Косинусное сходство'):
        end = min(i + batch_size, n)
        
        # конвертируем в тензоры
        a_t = torch.tensor(a[i:end], dtype=torch.float32, device=device)
        b_t = torch.tensor(b[i:end], dtype=torch.float32, device=device)
        
        # вычисляем сходство
        with torch.no_grad():
            batch_sim = torch.nn.functional.cosine_similarity(a_t, b_t)
            results[i:end] = batch_sim.cpu().numpy()
        
        # очистка памяти
        del a_t, b_t, batch_sim
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    return results

# обработка датафрейма
def process_dataframe(df, dataset_type):
    print(f"\nОбрабатка {dataset_type.upper()}")
    
    query_emb = load_or_compute_embeddings(df, 'query', dataset_type)
    
    for column in ['brand', 'model', 'type']:
        print(f"Вычисление сходства query-{column}...")
        
        col_emb = load_or_compute_embeddings(df, column, dataset_type)
        similarity = compute_cosine_similarity(query_emb, col_emb)
        
        df[f"query_{column}_cosine"] = similarity
        
    return df

train_df = process_dataframe(train_df, 'train')
test_df = process_dataframe(test_df, 'test')


def add_fuzzy_features(df, text_columns=['brand', 'model', 'type'], query_col='query'):
    tqdm.pandas()

    # схожесть между двумя строками
    def fuzzy_similarity(a, b):
        if not isinstance(a, str) or not isinstance(b, str):
            return 0
        return fuzz.partial_ratio(a.lower(), b.lower())
    
    # фичи для каждой указанной колонки
    for col in text_columns:
        feature_name = f"{col}_fuzzy"
        df[feature_name] = df.progress_apply(
            lambda x: fuzzy_similarity(x[query_col], x[col]), axis=1
        )
        # нормализация до [0, 1]
        df[feature_name] = df[feature_name] / 100
    
    return df

train_df = add_fuzzy_features(train_df, ['brand', 'model', 'type'])
test_df = add_fuzzy_features(test_df, ['brand', 'model', 'type'])


features = [
    'response_products', 'browse_time', 'price', 'price_discount',

    'device', 'selected_category_id', 'filters_applied', 'available',

    'session_price_mean', 'session_size', 'product_search_count', 'time_per_product', 'position_rank', 'product_age_days', 'price_vs_session_mean',
    'rank_weighted_engagement', 'price_cv', 'product_popularity_log',

    'query_len', 'model_token_jaccard',

    'query_brand_cosine', 'query_model_cosine', 'query_type_cosine', 'product_ctr', 'composite_score' ,'brand_fuzzy', 'model_fuzzy', 'type_fuzzy'
]

cat_features = ['device', 'selected_category_id', 'filters_applied', 'available']


train_df = train_df.sort_values('search_number').reset_index(drop=True)
test_df = test_df.sort_values('search_number').reset_index(drop=True)

splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, valid_idx = next(splitter.split(train_df, groups=train_df['search_number']))

train_data = train_df.iloc[train_idx]
valid_data = train_df.iloc[valid_idx]

train_pool = Pool(
    data=train_data[features],
    label=train_data['relevance'],
    cat_features=cat_features,
    group_id=train_data['search_number']
)

valid_pool = Pool(
    data=valid_data[features],
    label=valid_data['relevance'],
    cat_features=cat_features,
    group_id=valid_data['search_number']
)

test_pool = Pool(
    data=test_df[features],
    cat_features=cat_features,
    group_id=test_df['search_number']
)


model = CatBoostRanker(
    # Основные параметры
    iterations=7_000,
    learning_rate=0.05,
    depth=8,
    
    # Регуляризация
    l2_leaf_reg=5,
    
    # Функция потерь и метрика
    loss_function='YetiRank',
    eval_metric='NDCG:top=20',
    
    # Бутстрап для GPU
    bootstrap_type='Bernoulli',
    
    # Мониторинг и логи
    metric_period=50,
    verbose=100,
    
    # Ранняя остановка
    od_type='Iter',
    od_wait=100,
    
    # Воспроизводимость и ускорение
    random_seed=42,
    #task_type='GPU'
)

model.fit(train_pool, eval_set=valid_pool, verbose = 100, use_best_model=True, plot = True)


# важность признаков
feature_importance_values = model.get_feature_importance(train_pool)

feature_importance = pd.DataFrame({
    'feature': model.feature_names_,
    'importance': feature_importance_values
}).sort_values(by='importance', ascending=False)

print('Топ признаков по важности:')
print(feature_importance.head(55))

plt.figure(figsize=(10,6))
plt.barh(feature_importance['feature'].head(15)[::-1],
          feature_importance['importance'].head(15)[::-1])
plt.title('Feature Importance (CatBoostRanker)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()


top_features = feature_importance.head(20)['feature'].tolist()

corr_df = train_df[top_features].copy()

corr_df = corr_df.select_dtypes(include=['int', 'float'])

corr_matrix = corr_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    cmap='coolwarm',
    annot=True,
    fmt='.2f',
    linewidths=0.5
)
plt.title('Корреляции между топ-20 признаками', fontsize=14)
plt.show()


sample_df = train_df.sample(500, random_state=42)  # берем 500 наблюдений

sample_df = sample_df.sort_values('search_number').reset_index(drop=True)

sample_pool = Pool(
    data=sample_df[features],
    cat_features=cat_features,
    label=sample_df['relevance'],
    group_id=sample_df['search_number']
)

shap_values = model.get_feature_importance(sample_pool, type='ShapValues')[:, :-1]

shap_df = pd.DataFrame(shap_values, columns=features)

shap.summary_plot(shap_values, sample_df[features])


preds = model.predict(test_pool)

submission = pd.DataFrame({
    'session_product_id': test_df['session_product_id'].astype(str),
    'relevance': preds.astype(float)
})

print(submission.head())


submission.to_csv('my_test_47.csv', index=False)
print('Файл my_test_47.csv сохранён!')










