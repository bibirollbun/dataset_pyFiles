from copy import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


df_amazon_ratings = pd.read_csv('../input/amazon-ratings/ratings_Beauty.csv')
df_amazon_ratings = df_amazon_ratings.dropna()
df_amazon_ratings.head()


df_amazon_ratings.shape


popular_products = df_amazon_ratings.groupby('ProductId')['Rating'].agg(
    total_rating='sum',
    count_rating='count'
)

popular_products.head()


most_popular_by_total_rating = popular_products.sort_values(by=['total_rating'], ascending=False)
most_popular_by_total_rating.head(10)


most_popular_by_count_rating = popular_products.sort_values(by=['count_rating'], ascending=False)
most_popular_by_count_rating.head(10)


most_popular_by_count_then_total = popular_products.sort_values(by=['count_rating', 'total_rating'], ascending=False)
most_popular_by_count_then_total.head(10)


most_popular_by_count_then_total.head(30).plot(kind='bar')


df_product_descriptions = pd.read_csv('/kaggle/input/home-depot-product-search-relevance/product_descriptions.csv.zip')
df_product_descriptions.head()


df_product_descriptions.shape


df_product_descriptions = df_product_descriptions.dropna()
display(df_product_descriptions.shape)


df_product_descriptions_chunk1 = df_product_descriptions.head(500)
df_product_descriptions_chunk1.head()


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df_product_descriptions_chunk1['product_description'])
X


kmeans_model = KMeans(n_clusters=10, init='k-means++', n_init='auto', random_state=42)
kmeans_model.fit(X)
y_kmeans = kmeans_model.predict(X)
plt.plot(y_kmeans, ".")
plt.show()


def print_cluster(i):
    print(f"Cluster {i}:"),
    for ind in order_centroids[i, :10]:
        print(f"{terms[ind]}"),
    print()


print("Top terms per cluster:\n")
order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(10):
    print_cluster(i)


def show_recommendations(product):
    Y = vectorizer.transform([product])
    prediction = kmeans_model.predict(Y)
    print_cluster(prediction[0])


show_recommendations("cutting tool")


show_recommendations("spray paint")


show_recommendations("water")


df_amazon_ratings = pd.read_csv('../input/amazon-ratings/ratings_Beauty.csv').head(10000)
df_amazon_ratings = df_amazon_ratings.dropna()
df_amazon_ratings.head()


df_amazon_ratings.shape


ratings_matrix = pd.pivot_table(df_amazon_ratings, values='Rating', index='UserId', columns='ProductId', fill_value=0, dropna=True)
ratings_matrix.head()


ratings_matrix.shape


X1 = copy(ratings_matrix)
X1.head()


SVD = TruncatedSVD(n_components=10)
U = SVD.fit_transform(X1)
U.shape


U[:5]


X2 = copy(ratings_matrix.T)
X2.head()


SVD = TruncatedSVD(n_components=10)
V = SVD.fit_transform(X2)
V.shape


V[:5]


product_idx = 200
user_idx = 150
product_id = X1.index[product_idx]
user_id = X2.index[user_idx]
print(f"Product ID: {product_id}")
print(f"User ID: {user_id}")


ratings_matrix.iloc[user_idx, product_idx]


print(f"user latent factors: {U[user_idx]}\n")
print(f"product latent factors: {V[product_idx]}")


predicted_rating = U[user_idx].dot(V[product_idx].T)
predicted_rating


R = U.dot(V.T)
R.shape


user_ratings = R[user_idx, :]
print(f"Top 10 Ratings for user: {user_idx}\n")
user_ratings_sorted_indices = np.argsort(user_ratings)[::-1]
user_ratings_sorted = user_ratings[user_ratings_sorted_indices]
user_ratings_sorted[:10]


top_10_indices = user_ratings_sorted_indices[:10]
top_10_product_ids = ratings_matrix.columns[top_10_indices]

print("Top 10 Product IDs:", top_10_product_ids)


for pid, rating in zip(top_10_product_ids, user_ratings[top_10_indices]):
    print(f"Product: {pid}, Rating: {rating}")

