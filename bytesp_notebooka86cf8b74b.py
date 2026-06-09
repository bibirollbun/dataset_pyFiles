import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


transactions = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv',dtype={
        'article_id': 'category',
        'customer_id': 'category',
        'price': 'float32'
    },
    parse_dates=['t_dat'])
articles = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv',dtype={'customer_id': 'category'})
customers = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv',dtype={'article_id': 'category'})

#Verify loading
print("Transactions shape:", transactions.shape)
print("Customers shape:", customers.shape)
print("Articles shape:", articles.shape)


# Handle missing values in customers
customers['FN'] = customers['FN'].fillna(0).astype('int8')
customers['Active'] = customers['Active'].fillna(0).astype('int8')
customers['club_member_status'] = customers['club_member_status'].fillna('UNKNOWN')
customers['fashion_news_frequency'] = customers['fashion_news_frequency'].replace('None', 'NONE').fillna('NONE')

# Clean articles data
articles['detail_desc'] = articles['detail_desc'].fillna('Not Available')

# Merge datasets
merged_data = transactions.merge(customers, on='customer_id', how='left')
merged_data = merged_data.merge(articles, on='article_id', how='left')

# Check merged data
print("\nMerged data shape:", merged_data.shape)
print("Missing values after merge:")
print(merged_data.isnull().sum())


# Product popularity features
product_popularity = transactions.groupby('article_id', observed=True).agg(
    total_purchases=('customer_id', 'count'),
    unique_customers=('customer_id', 'nunique')
).reset_index()
product_popularity['popularity_rank'] = product_popularity['total_purchases'].rank(ascending=False, method='dense')

# Customer activity features
customer_activity = transactions.groupby('customer_id', observed=True).agg(
    purchase_count=('t_dat', 'count'),
    first_purchase=('t_dat', 'min'),
    last_purchase=('t_dat', 'max'),
    avg_price=('price', 'mean')
).reset_index()
customer_activity['purchase_frequency'] = customer_activity['purchase_count'] / (customer_activity['last_purchase'] - customer_activity['first_purchase']).dt.days

# Product age features
current_date = transactions['t_dat'].max()
product_age = transactions.groupby('article_id', observed=True)['t_dat'].min().reset_index()
product_age['product_age_days'] = (current_date - product_age['t_dat']).dt.days

print("\nFeature engineering completed:")
print("- Product features:", product_popularity.shape)
print("- Customer features:", customer_activity.shape)
print("- Product age features:", product_age.shape)


#Prepare Data for Modeling
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

#Create user-item interaction matrix (binary)
user_ids = transactions['customer_id'].astype('category').cat.codes
item_ids = transactions['article_id'].astype('category').cat.codes

interaction_matrix = csr_matrix(
    (np.ones(len(transactions), (user_ids, item_ids)),
    shape=(len(user_ids.unique()), len(item_ids.unique()))
)

#Split data
train_matrix, test_matrix = train_test_split(
    interaction_matrix, 
    test_size=0.2,
    random_state=42
)

#Baseline Models
from collections import defaultdict

#Popularity Model
def popularity_model(transactions, n_recommendations=12):
    top_items = transactions['article_id'].value_counts().head(n_recommendations).index.tolist()
    return {user: top_items for user in transactions['customer_id'].unique()}

popularity_recs = popularity_model(transactions)

#Recent Popularity Model
def recent_popularity_model(transactions, days=30, n_recommendations=12):
    recent_date = transactions['t_dat'].max()
    recent_trans = transactions[transactions['t_dat'] >= (recent_date - pd.Timedelta(days=days)]
    top_items = recent_trans['article_id'].value_counts().head(n_recommendations).index.tolist()
    return {user: top_items for user in transactions['customer_id'].unique()}

recent_recs = recent_popularity_model(transactions)

#Collaborative Filtering Models
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k

#Initialize model
model = LightFM(
    loss='warp',  # Weighted Approximate-Rank Pairwise
    no_components=30,
    user_alpha=0.0001,
    item_alpha=0.0001
)

#Train model
model.fit(
    train_matrix,
    epochs=20,
    num_threads=4,
    verbose=True
)

#Model Evaluation
train_precision = precision_at_k(model, train_matrix, k=12).mean()
test_precision = precision_at_k(model, test_matrix, k=12).mean()

print(f"\nModel Performance:")
print(f"Train Precision@12: {train_precision:.4f}")
print(f"Test Precision@12: {test_precision:.4f}")

#Generate Recommendations
def generate_recommendations(model, user_ids, item_ids, n=12):
    all_items = np.arange(interaction_matrix.shape[1])
    user_codes = {user: code for code, user in enumerate(user_ids.cat.categories)}
    
    recommendations = {}
    for user, user_code in tqdm(user_codes.items()):
        scores = model.predict(user_code, all_items)
        top_items = np.argsort(-scores)[:n]
        recommendations[user] = item_ids.cat.categories[top_items].tolist()
    
    return recommendations

cf_recs = generate_recommendations(model, user_ids, item_ids)

#Hybrid Recommendations
def hybrid_recommendation(user, cf_recs, pop_recs, weight=0.7):
    """Combine collaborative filtering and popularity"""
    cf_items = cf_recs.get(user, [])
    pop_items = pop_recs.get(user, [])
    
    #Take top from CF, fill remainder with popular items
    n_cf = int(12 * weight)
    recommendations = cf_items[:n_cf] + pop_items[:12-n_cf]
    return recommendations[:12]  # Ensure exactly 12

#Create final recommendations
final_recs = {
    user: hybrid_recommendation(user, cf_recs, popularity_recs)
    for user in transactions['customer_id'].unique()[:10000]  # Sample for demo
}

#Save Results
import json

with open('recommendations.json', 'w') as f:
    json.dump(final_recs, f)

#Create Kaggle submission format
submission = pd.DataFrame({
    'customer_id': final_recs.keys(),
    'prediction': [' '.join(map(str, items)) for items in final_recs.values()]
})
submission.to_csv('submission.csv', index=False)

print("\nRecommendations generated and saved!")
print(f"Total users processed: {len(final_recs)}")
print(f"Sample recommendation: {next(iter(final_recs.items()))}")

