!pip install implicit scikit-surprise --quiet

# Ğ˜Ğ¼Ğ¿Ğ¾Ñ€Ñ‚ Ğ±Ğ¸Ğ±Ğ»Ğ¸Ğ¾Ñ‚ĞµĞº
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Ğ—Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ° Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
def load_data():
    paths = [
        '/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv',
        '../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv'
    ]
    for path in paths:
        try:
            transactions = pd.read_csv(path, dtype={'article_id': str}, parse_dates=['t_dat'])
            customers = pd.read_csv(path.replace('transactions_train', 'customers'))
            articles = pd.read_csv(path.replace('transactions_train', 'articles'), dtype={'article_id': str})
            print(f"Ğ”Ğ°Ğ½Ğ½Ñ‹Ğµ Ğ·Ğ°Ğ³Ñ€ÑƒĞ¶ĞµĞ½Ñ‹ Ğ¸Ğ· {path}")
            return transactions, customers, articles
        except:
            continue
    raise FileNotFoundError("Ğ�Ğµ ÑƒĞ´Ğ°Ğ»Ğ¾Ñ�ÑŒ Ğ·Ğ°Ğ³Ñ€ÑƒĞ·Ğ¸Ñ‚ÑŒ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ")

transactions, customers, articles = load_data()

# Ğ‘Ñ‹Ñ�Ñ‚Ñ€Ñ‹Ğ¹ EDA Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ·
print("â•�"*50)
print(f"ğŸ“Š Ğ¢Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸: {len(transactions):,} Ğ·Ğ°Ğ¿Ğ¸Ñ�ĞµĞ¹")
print(f"ğŸ‘¥ ĞšĞ»Ğ¸ĞµĞ½Ñ‚Ñ‹: {transactions.customer_id.nunique():,} ÑƒĞ½Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ñ…")
print(f"ğŸ‘• Ğ¢Ğ¾Ğ²Ğ°Ñ€Ñ‹: {transactions.article_id.nunique():,} ÑƒĞ½Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ñ…")

# Ğ¢Ğ¾Ğ¿-10 Ğ¿Ğ¾Ğ¿ÑƒĞ»Ñ�Ñ€Ğ½Ñ‹Ñ… Ñ‚Ğ¾Ğ²Ğ°Ñ€Ğ¾Ğ²
top_items = transactions.article_id.value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_items.values, y=top_items.index)
plt.title('Ğ¢Ğ¾Ğ¿-10 Ğ¿Ğ¾Ğ¿ÑƒĞ»Ñ�Ñ€Ğ½Ñ‹Ñ… Ñ‚Ğ¾Ğ²Ğ°Ñ€Ğ¾Ğ²')
plt.show()

# ĞšĞ»Ğ°Ñ�Ñ� Ğ´Ğ»Ñ� Ğ¿Ñ€ĞµĞ´Ğ¾Ğ±Ñ€Ğ°Ğ±Ğ¾Ñ‚ĞºĞ¸ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
class HMDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, last_n_days=90, min_purchases=3):
        self.last_n_days = last_n_days
        self.min_purchases = min_purchases
        self.popular_items = None
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, transactions):
        # Ğ¤Ğ¸Ğ»ÑŒÑ‚Ñ€Ğ°Ñ†Ğ¸Ñ� Ğ¿Ğ¾ Ğ´Ğ°Ñ‚Ğµ
        last_date = transactions.t_dat.max()
        mask = transactions.t_dat > (last_date - pd.Timedelta(days=self.last_n_days))
        recent_trans = transactions[mask].copy()
        
        # Ğ¤Ğ¸Ğ»ÑŒÑ‚Ñ€Ğ°Ñ†Ğ¸Ñ� Ğ½ĞµĞ°ĞºÑ‚Ğ¸Ğ²Ğ½Ñ‹Ñ… Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ñ‚ĞµĞ»ĞµĞ¹
        user_counts = recent_trans.customer_id.value_counts()
        active_users = user_counts[user_counts >= self.min_purchases].index
        filtered_trans = recent_trans[recent_trans.customer_id.isin(active_users)]
        
        # Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ¿Ğ¾Ğ¿ÑƒĞ»Ñ�Ñ€Ğ½Ñ‹Ğµ Ñ‚Ğ¾Ğ²Ğ°Ñ€Ñ‹ Ğ´Ğ»Ñ� Ñ…Ğ¾Ğ»Ğ¾Ğ´Ğ½Ñ‹Ñ… Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ñ‚ĞµĞ»ĞµĞ¹
        self.popular_items = filtered_trans.article_id.value_counts().head(12).index.tolist()
        
        return filtered_trans

# ĞšĞ»Ğ°Ñ�Ñ� Ğ´Ğ»Ñ� ALS Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸
class ALSRecommender(BaseEstimator, TransformerMixin):
    def __init__(self, factors=50, iterations=15, regularization=0.01):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.model = None
        self.user_map = None
        self.item_map = None
        self.popular_items = None  # Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ğ°Ñ‚Ñ€Ğ¸Ğ±ÑƒÑ‚ Ğ´Ğ»Ñ� Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ñ� Ğ¿Ğ¾Ğ¿ÑƒĞ»Ñ�Ñ€Ğ½Ñ‹Ñ… Ñ‚Ğ¾Ğ²Ğ°Ñ€Ğ¾Ğ²
        
    def fit(self, transactions, popular_items=None):  # Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€ popular_items
        # Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ¿Ğ¾Ğ¿ÑƒĞ»Ñ�Ñ€Ğ½Ñ‹Ğµ Ñ‚Ğ¾Ğ²Ğ°Ñ€Ñ‹
        self.popular_items = popular_items
        
        # Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¸Ğµ mappings
        users = transactions.customer_id.unique()
        items = transactions.article_id.unique()
        
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {a: i for i, a in enumerate(items)}
        
        # ĞŸĞ¾Ñ�Ñ‚Ñ€Ğ¾ĞµĞ½Ğ¸Ğµ sparse Ğ¼Ğ°Ñ‚Ñ€Ğ¸Ñ†Ñ‹
        rows = [self.user_map[u] for u in transactions.customer_id]
        cols = [self.item_map[a] for a in transactions.article_id]
        data = np.ones(len(transactions))
        
        interactions = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        
        # Ğ�Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=42
        )
        self.model.fit(interactions)
        
        return self
    
    def recommend(self, customer_ids, k=12):
        """Ğ“ĞµĞ½ĞµÑ€Ğ°Ñ†Ğ¸Ñ� Ñ€ĞµĞºĞ¾Ğ¼ĞµĞ½Ğ´Ğ°Ñ†Ğ¸Ğ¹ Ğ´Ğ»Ñ� Ñ�Ğ¿Ğ¸Ñ�ĞºĞ° Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ñ‚ĞµĞ»ĞµĞ¹"""
        if self.popular_items is None:
            raise ValueError("Popular items not set. Please provide popular items during fit.")
            
        all_recs = []
        item_ids = np.arange(len(self.item_map))
        
        for customer_id in tqdm(customer_ids, desc="Ğ“ĞµĞ½ĞµÑ€Ğ°Ñ†Ğ¸Ñ� Ñ€ĞµĞºĞ¾Ğ¼ĞµĞ½Ğ´Ğ°Ñ†Ğ¸Ğ¹"):
            if customer_id in self.user_map:
                user_idx = self.user_map[customer_id]
                scores = self.model.user_factors[user_idx] @ self.model.item_factors.T
                top_items = np.argsort(-scores)[:k]
                recs = [list(self.item_map.keys())[i] for i in top_items]
            else:
                recs = self.popular_items[:k]  # Ğ˜Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·ÑƒĞµĞ¼ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ½Ñ‹Ğµ Ğ¿Ğ¾Ğ¿ÑƒĞ»Ñ�Ñ€Ğ½Ñ‹Ğµ Ñ‚Ğ¾Ğ²Ğ°Ñ€Ñ‹
            
            all_recs.append(' '.join(recs))
        
        return pd.DataFrame({'customer_id': customer_ids, 'prediction': all_recs})

# Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¸Ğµ Ğ¸ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ Ğ¿Ğ°Ğ¹Ğ¿Ğ»Ğ°Ğ¹Ğ½Ğ°
print("ğŸ›  ĞŸĞ¾Ğ´Ğ³Ğ¾Ñ‚Ğ¾Ğ²ĞºĞ° Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… Ğ¸ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸...")
preprocessor = HMDataPreprocessor(last_n_days=180)
filtered_trans = preprocessor.fit_transform(transactions)

# ĞŸĞ¾Ğ»ÑƒÑ‡Ğ°ĞµĞ¼ Ğ¿Ğ¾Ğ¿ÑƒĞ»Ñ�Ñ€Ğ½Ñ‹Ğµ Ñ‚Ğ¾Ğ²Ğ°Ñ€Ñ‹ Ğ¸Ğ· Ğ¿Ñ€ĞµĞ¿Ñ€Ğ¾Ñ†ĞµÑ�Ñ�Ğ¾Ñ€Ğ°
popular_items = preprocessor.popular_items

model = ALSRecommender(factors=64, iterations=20)
model.fit(filtered_trans, popular_items=popular_items)  # ĞŸĞµÑ€ĞµĞ´Ğ°ĞµĞ¼ Ğ¿Ğ¾Ğ¿ÑƒĞ»Ñ�Ñ€Ğ½Ñ‹Ğµ Ñ‚Ğ¾Ğ²Ğ°Ñ€Ñ‹

# Ğ“ĞµĞ½ĞµÑ€Ğ°Ñ†Ğ¸Ñ� Ñ€ĞµĞºĞ¾Ğ¼ĞµĞ½Ğ´Ğ°Ñ†Ğ¸Ğ¹ (Ğ´Ğ»Ñ� Ğ´ĞµĞ¼Ğ¾Ğ½Ñ�Ñ‚Ñ€Ğ°Ñ†Ğ¸Ğ¸ 10% Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ñ‚ĞµĞ»ĞµĞ¹)
sample_customers = customers.sample(frac=0.1, random_state=42).customer_id
submission = model.recommend(sample_customers)

# Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ğµ Ğ¸ Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ· Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ğ¾Ğ²
submission.to_csv('submission_als.csv', index=False)
print("âœ… Ğ¡Ğ°Ğ±Ğ¼Ğ¸Ñ‚ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½ ĞºĞ°Ğº submission_als.csv")

# Ğ’Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ� Ñ€ĞµĞºĞ¾Ğ¼ĞµĞ½Ğ´Ğ°Ñ†Ğ¸Ğ¹
plt.figure(figsize=(10, 6))
rec_counts = submission.prediction.str.split().explode().value_counts().head(20)
sns.barplot(y=rec_counts.index, x=rec_counts.values)
plt.title('Ğ¡Ğ°Ğ¼Ñ‹Ğµ Ñ‡Ğ°Ñ�Ñ‚Ğ¾ Ñ€ĞµĞºĞ¾Ğ¼ĞµĞ½Ğ´ÑƒĞµĞ¼Ñ‹Ğµ Ñ‚Ğ¾Ğ²Ğ°Ñ€Ñ‹')
plt.xlabel('ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ€ĞµĞºĞ¾Ğ¼ĞµĞ½Ğ´Ğ°Ñ†Ğ¸Ğ¹')
plt.show()

# ĞŸÑ€Ğ¸Ğ¼ĞµÑ€ Ñ€ĞµĞºĞ¾Ğ¼ĞµĞ½Ğ´Ğ°Ñ†Ğ¸Ğ¹
print("\nĞŸÑ€Ğ¸Ğ¼ĞµÑ€ Ñ€ĞµĞºĞ¾Ğ¼ĞµĞ½Ğ´Ğ°Ñ†Ğ¸Ğ¹ Ğ´Ğ»Ñ� 3 Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ñ‚ĞµĞ»ĞµĞ¹:")
for i in range(3):
    user_id = submission.iloc[i].customer_id
    items = submission.iloc[i].prediction.split()
    print(f"\nğŸ‘¤ {user_id}:")
    print(articles[articles.article_id.isin(items)][['article_id', 'prod_name']].to_string(index=False))

