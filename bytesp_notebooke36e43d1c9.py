!pip install scikit-surprise


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict


#Helper: Safe memory optimizer
def optimize(df):
    for col in df.columns:
        col_type = df[col].dtypes

        if pd.api.types.is_object_dtype(col_type):
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype("category")

        elif pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")

        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")
    
    return df

#Load datasets
transactions = pd.read_csv(
    "/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv",
    parse_dates=["t_dat"],
    dtype={
        "customer_id": "category",
        "article_id": "int32",
        "price": "float32",
        "sales_channel_id": "int8"
    }
)

customers = pd.read_csv("/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv")
articles = pd.read_csv("/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv")

#Clean and optimize
transactions = optimize(transactions).dropna().drop_duplicates()
customers = optimize(customers).dropna().drop_duplicates()
articles = optimize(articles).dropna().drop_duplicates()

#Merge datasets
merged = transactions.merge(articles, on="article_id", how="inner")
merged = merged.merge(customers, on="customer_id", how="inner")

#Feature Engineering

#1. Most popular articles
popular_articles = transactions["article_id"].value_counts().head(10)
print("Top 10 most bought articles:\n", popular_articles)

#2. Purchase frequency per customer
purchase_freq = transactions.groupby("customer_id")["article_id"].count()
print("\nCustomer purchase frequency stats:\n", purchase_freq.describe())

#3. Product age estimate
articles["product_code"] = articles["product_code"].astype(str)
articles["product_age"] = 2025 - articles["prod_name"].str.extract(r"(\d{4})").fillna(2020).astype(int)

#4. User-Item Interaction Matrix
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

transactions["user"] = user_encoder.fit_transform(transactions["customer_id"])
transactions["item"] = item_encoder.fit_transform(transactions["article_id"])

interaction_matrix = csr_matrix(
    (np.ones(len(transactions), dtype=np.float32),
     (transactions["user"], transactions["item"]))
)

print(f"\nInteraction matrix shape: {interaction_matrix.shape}")





#Filter active users and popular items
min_items_per_user = 5
min_users_per_item = 5

user_counts = transactions["user"].value_counts()
item_counts = transactions["item"].value_counts()

filtered = transactions[
    transactions["user"].isin(user_counts[user_counts >= min_items_per_user].index) &
    transactions["item"].isin(item_counts[item_counts >= min_users_per_item].index)
]

#Optional: further subsample for performance
top_users = filtered["user"].value_counts().head(10000).index
filtered = filtered[filtered["user"].isin(top_users)]

#Rebuild sparse interaction matrix
interaction_matrix = csr_matrix(
    (np.ones(len(filtered), dtype=np.float32),
     (filtered["user"], filtered["item"]))
)

#Efficient cosine similarity (user-based)
user_sim = cosine_similarity(interaction_matrix, dense_output=False)

#Recommendation Function
def recommend_cf(user_id, k=12):
    try:
        user_idx = user_encoder.transform([user_id])[0]
        if user_idx >= user_sim.shape[0]:
            return []  # user out of range
        sim_scores = user_sim[user_idx].toarray().flatten()

        # Top similar users (exclude self)
        top_similar = sim_scores.argsort()[::-1][1:6]

        # Sum item interactions of top similar users
        top_user_items = interaction_matrix[top_similar].sum(axis=0).A1
        recommended_idx = top_user_items.argsort()[::-1][:k]
        return item_encoder.inverse_transform(recommended_idx)
    except:
        return []

#Test
test_user = filtered["customer_id"].iloc[0]
recommendations = recommend_cf(test_user)
print(f"Recommendations for user {test_user}:\n{recommendations}")



#Prepare data
filtered_df = transactions.groupby("customer_id").filter(lambda x: len(x) >= 5)

# Treat each purchase as an interaction score of 1
df_surprise = filtered_df[["customer_id", "article_id"]].copy()
df_surprise["interaction"] = 1

#Further filter: top 5000 most active users
top_users = df_surprise["customer_id"].value_counts().head(5000).index
df_surprise = df_surprise[df_surprise["customer_id"].isin(top_users)]

#Filter low-frequency articles
top_articles = df_surprise["article_id"].value_counts().head(5000).index
df_surprise = df_surprise[df_surprise["article_id"].isin(top_articles)]

#Load Surprise Dataset
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(df_surprise[["customer_id", "article_id", "interaction"]], reader)
trainset, testset = train_test_split(data, test_size=0.2)

#Train SVD
model = SVD(n_factors=20, n_epochs=5, verbose=True, random_state=42)
model.fit(trainset)



#Get top-N predictions
def get_top_n(predictions, n=12):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    
    for uid in top_n:
        top_n[uid] = sorted(top_n[uid], key=lambda x: x[1], reverse=True)[:n]
        top_n[uid] = [iid for iid, _ in top_n[uid]]
    
    return top_n

#Compute Precision at k
def precision_at_k(predictions, top_n, k=12):
    hits = 0
    total = 0
    
    for uid, iid, true_r, est, _ in predictions:
        if iid in top_n.get(uid, []):
            hits += 1
        total += 1
    
    return hits / total if total > 0 else 0



#Predict on test set
predictions = model.test(testset)

#Generate top-N for each user
top_n = get_top_n(predictions, n=12)

#Evaluate
p_at_12 = precision_at_k(predictions, top_n, k=12)
print(f"Precision@12 using optimized SVD: {p_at_12:.4f}")



#1. Get test users (first 1000 unique customer IDs)
test_users = transactions['customer_id'].unique()[:1000]

#2. Generate recommendations using your CF function
recommendations = [recommend_cf(user) for user in test_users]

#3. Create DataFrame and save to CSV
recommendations_df = pd.DataFrame({
    "customer_id": test_users,
    "recommended_articles": recommendations
})

output = '/kaggle/working/recommendations.csv'
recommendations_df.to_csv(output, index=False)

print(f"Recommendations saved to {output}")
print(f"First 5 recommendations:\n{recommendations_df.head()}")

