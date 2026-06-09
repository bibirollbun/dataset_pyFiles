# Установка и импорт библиотек
!pip install implicit scikit-surprise --quiet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
def load_data():
    try:
        trans = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv', 
                          dtype={'article_id': str}, parse_dates=['t_dat'])
        cust = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv')
        arts = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv',
                         dtype={'article_id': str})
        print("Данные загружены")
        return trans, cust, arts
    except:
        trans = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv',
                          dtype={'article_id': str}, parse_dates=['t_dat'])
        cust = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/customers.csv')
        arts = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/articles.csv',
                         dtype={'article_id': str})
        print("Данные загружены через относительный путь")
        return trans, cust, arts

transactions, customers, articles = load_data()

# Предобработка данных
def preprocess_data(trans, last_n_days=180, min_purchases=3):
    # Фильтрация по дате
    last_date = trans['t_dat'].max()
    mask = trans['t_dat'] > (last_date - pd.Timedelta(days=last_n_days))
    recent_trans = trans[mask].copy()
    
    # Фильтрация неактивных пользователей
    user_counts = recent_trans['customer_id'].value_counts()
    active_users = user_counts[user_counts >= min_purchases].index
    filtered_trans = recent_trans[recent_trans['customer_id'].isin(active_users)]
    
    # Топ товаров для новых пользователей
    popular_items = filtered_trans['article_id'].value_counts().head(12).index.tolist()
    
    return filtered_trans, popular_items

processed_trans, popular_items = preprocess_data(transactions)

# Подготовка матрицы взаимодействий
def build_interaction_matrix(df):
    # Создание mappings
    user_ids = df['customer_id'].unique()
    item_ids = df['article_id'].unique()
    
    user_map = {u: i for i, u in enumerate(user_ids)}
    item_map = {a: i for i, a in enumerate(item_ids)}
    
    # Построение матрицы
    rows = [user_map[u] for u in df['customer_id']]
    cols = [item_map[a] for a in df['article_id']]
    data = np.ones(len(df))
    
    return csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids))), user_map, item_map

interaction_matrix, user_map, item_map = build_interaction_matrix(processed_trans)

# Обучение ALS модели
print("Обучение модели...")
model = AlternatingLeastSquares(
    factors=64,
    iterations=15,
    regularization=0.1,
    random_state=42
)
model.fit(interaction_matrix)

# Генерация рекомендаций
def generate_recommendations(customer_list, model, user_map, item_map, popular_items, k=12):
    reverse_item_map = {v: k for k, v in item_map.items()}
    recommendations = []
    
    for customer in tqdm(customer_list, desc="Генерация рекомендаций"):
        if customer in user_map:
            user_idx = user_map[customer]
            scores = model.user_factors[user_idx] @ model.item_factors.T
            top_items = np.argsort(-scores)[:k]
            recs = [reverse_item_map[i] for i in top_items]
        else:
            recs = popular_items[:k]
        
        recommendations.append(' '.join(recs))
    
    return pd.DataFrame({'customer_id': customer_list, 'prediction': recommendations})

# Для демонстрации возьмем 10% пользователей
sample_customers = customers.sample(frac=0.1, random_state=42)['customer_id']
submission = generate_recommendations(sample_customers, model, user_map, item_map, popular_items)

# Сохранение и анализ результатов
submission.to_csv('submission.csv', index=False)
print("Сабмит сохранен как submission.csv")

# Пример рекомендаций
print("\nПример рекомендаций для 3 пользователей:")
for i in range(3):
    cust_id = submission.iloc[i]['customer_id']
    items = submission.iloc[i]['prediction'].split()
    print(f"\nПользователь {cust_id}:")
    print(articles[articles['article_id'].isin(items)][['article_id', 'prod_name']].to_string(index=False))

# Визуализация
plt.figure(figsize=(10, 6))
top_recs = submission['prediction'].str.split().explode().value_counts().head(20)
sns.barplot(y=top_recs.index, x=top_recs.values)
plt.title('20 самых рекомендуемых товаров')
plt.xlabel('Количество рекомендаций')
plt.show()

