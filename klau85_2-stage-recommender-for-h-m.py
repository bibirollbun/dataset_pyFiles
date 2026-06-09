# there is a bug in tfrs and i need to use legacy keras until the PR is merged
# https://github.com/tensorflow/recommenders/pull/717
# https://github.com/tensorflow/recommenders/issues/712
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'


!pip install -q tensorflow_recommenders lightgbm


import tensorflow as tf
from tensorflow import keras
import tensorflow_recommenders as tfrs
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRanker
from sklearn.model_selection import train_test_split


pd.set_option('display.max_colwidth', 70)


# Load transactions and articles
path = '/kaggle/input/h-and-m-personalized-fashion-recommendations/'
transactions = pd.read_csv(path + "transactions_train.csv", parse_dates=["t_dat"])

articles = pd.read_csv(path + "articles.csv")
articles["article_id"] = articles["article_id"].astype(str)

customers = pd.read_csv(path + "customers.csv")
customers["customer_id"] = customers["customer_id"].astype(str)


# Reduce size for prototyping
transactions = transactions[transactions['t_dat'] >= '2020-08-01']
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
transactions.sort_values('t_dat', inplace=True)


transactions = transactions.dropna()
transactions["customer_id"] = transactions["customer_id"].astype(str)
transactions["article_id"] = transactions["article_id"].astype(str)


# Merge metadata
transactions = transactions.merge(articles[["article_id", "index_group_name"]], on="article_id", how="left")
transactions = transactions.merge(customers[["customer_id", "age"]], on="customer_id", how="left")


# Add age group
def age_to_group(age):
    if age < 20: return "<20"
    elif age < 25: return '20-25'
    elif age < 30: return '25-30'
    elif age < 40: return '30-40'
    elif age < 50: return '40-50'
    else: return '50+'   


transactions['age_group'] = transactions['age'].apply(age_to_group)


cutoff_date = transactions["t_dat"].max() - pd.DateOffset(days=7)
train = transactions[transactions["t_dat"] < cutoff_date]
test = transactions[transactions["t_dat"] >= cutoff_date]


cols = ["customer_id", "article_id", "age_group", "index_group_name"]
train = train[cols]
test = test[cols]


# Convert to TensorFlow Datasets
train_tf = tf.data.Dataset.from_tensor_slices(dict(train))
test_tf = tf.data.Dataset.from_tensor_slices(dict(test))


unique_customers = transactions["customer_id"].unique()
unique_articles = transactions["article_id"].unique() 


class RetrievalModel(tfrs.Model):
    def __init__(self):
        super().__init__()

        embedding_dim = 32
        l2_regularizer = 0.000001
        dropout = 0.2
        dense = 32

        # Embedding layers
        self.customer_lookup = tf.keras.layers.StringLookup(vocabulary=unique_customers, mask_token=None)
        self.article_lookup = tf.keras.layers.StringLookup(vocabulary=unique_articles, mask_token=None)

        # User tower (Functional API)
        user_customer_id = keras.Input(shape=(), dtype=tf.string, name='customer_id')
        customer_emb = keras.layers.Embedding(
            len(unique_customers)+1, 
            embedding_dim, 
            embeddings_regularizer=keras.regularizers.l2(l2_regularizer))(self.customer_lookup(user_customer_id))
        customer_emb = keras.layers.Dropout(dropout)(customer_emb)
        user_output = keras.layers.Dense(dense, activation='relu')(customer_emb)
        self.user_tower = keras.Model(
            inputs={'customer_id': user_customer_id}, 
            outputs=user_output, 
            name='user_tower')

        # Item tower (Functional API)
        item_article_id = keras.Input(shape=(), dtype=tf.string, name='article_id')
        article_emb = keras.layers.Embedding(
            len(unique_articles)+1, 
            embedding_dim, 
            embeddings_regularizer=keras.regularizers.l2(l2_regularizer))(self.article_lookup(item_article_id))
        article_emb = keras.layers.Dropout(dropout)(article_emb)
        item_output = keras.layers.Dense(dense, activation='relu')(article_emb)
        self.item_tower = keras.Model(
            inputs={'article_id': item_article_id}, 
            outputs=item_output, 
            name='item_tower')

        candidate_dataset = train_tf.shuffle(10000).take(100).batch(50).map(
            lambda x: {
                "article_id": x["article_id"],
            }
        )
        
        self.task = tfrs.tasks.Retrieval(
            metrics=[
                tfrs.metrics.FactorizedTopK(
                    candidates=candidate_dataset.map(self.item_tower)
                )
            ]             
        )

    def compute_loss(self, features, training=False):
        user_embedding = self.user_tower({
            "customer_id": features["customer_id"],
        })
        
        item_embedding = self.item_tower({
            "article_id": features["article_id"],
        })
        
        return self.task(user_embedding, item_embedding, compute_metrics=not training)


model = RetrievalModel()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

# batch and cache
cached_train = train_tf.batch(1024).cache()
cached_test = test_tf.batch(1024).cache()

epochs=40
batch_size = 128

# train
retrieval_history = model.fit(cached_train, validation_data=cached_test, epochs=epochs, batch_size=batch_size)


epochs_list = [x for x in range(epochs)]


plt.plot(epochs_list, retrieval_history.history["val_factorized_top_k/top_1_categorical_accuracy"], label="top 1")
plt.plot(epochs_list, retrieval_history.history["val_factorized_top_k/top_10_categorical_accuracy"], label="top 10")
plt.plot(epochs_list, retrieval_history.history["val_factorized_top_k/top_50_categorical_accuracy"], label="top 50")

plt.title("top-n categorical accuracy")
plt.xlabel("epoch")
plt.ylabel("Top-n accuracy");
plt.legend()


index = tfrs.layers.factorized_top_k.BruteForce(model.user_tower, k = 50)


unique_candidates = train[["article_id"]].drop_duplicates()
candidate_dataset = tf.data.Dataset.from_tensor_slices(dict(unique_candidates)).batch(128)
candidates = tf.data.Dataset.zip(
    (candidate_dataset.map(lambda x: x["article_id"]), candidate_dataset.map(model.item_tower))
)

index.index_from_dataset(candidates)


# Example query
index({
    "customer_id": np.array(["fff969b13a1c848d53ae3f08f111bfebcdcf6cd27e3815235db95f1e99524c79"])
})  


users = train[["customer_id", 'age_group']].drop_duplicates()
users = users[:10000]


users.shape


# get recommendations from the retrieval model
ranking_data = []
for _, row in users.iterrows():
    query = {
        "customer_id": tf.constant([row['customer_id']])
    }
    scores, articles = index(query)
    
    uid = row['customer_id']
    for score, aid in zip(scores[0].numpy(), articles[0].numpy()):
        ranking_data.append({
            "customer_id": str(uid), 
            "article_id": aid.decode("utf-8"), 
            'age_group': row['age_group'],
            "retrieval_score": float(score)
        })

ranking_df = pd.DataFrame(ranking_data)


train_articles_df = train[["article_id", "index_group_name"]].drop_duplicates()
ranking_df = ranking_df.merge(train_articles_df, on=["article_id"], how="left")


ranking_df


positive_labels = train[["customer_id", "article_id", 'age_group', 'index_group_name']].drop_duplicates()
positive_labels["label"] = 1

ranking_df = ranking_df.merge(positive_labels, on=["customer_id", "article_id", 'age_group', 'index_group_name'], how="left")
ranking_df["label"] = ranking_df["label"].fillna(0)

ranking_df = ranking_df.sort_values(by=["customer_id"], ascending=[True])


ranking_df['label'].value_counts()


# Encode categorical columns for the ranking data
col_encoders = {}
for col in cols:
    col_encoders[col] = LabelEncoder()
    for col, encoder in col_encoders.items():
        ranking_df[col] = encoder.fit_transform(ranking_df[col].astype(str))


# Split by user for group-based ranking
target_users = ranking_df["customer_id"].unique()
train_users, test_users = train_test_split(target_users, test_size=0.2, random_state=42)


train_df = ranking_df[ranking_df["customer_id"].isin(train_users)]
test_df = ranking_df[ranking_df["customer_id"].isin(test_users)]


features = ["customer_id", "article_id", "retrieval_score", "index_group_name", "age_group"]

X_train = train_df[features].fillna(0)
y_train = train_df["label"]
group_train = train_df.groupby("customer_id").size().values

X_test = test_df[features].fillna(0)
y_test = test_df["label"]
group_test = test_df.groupby("customer_id").size().values


lgbm_ranker = LGBMRanker(objective='lambdarank', n_estimators=100)
lgbm_ranker.fit(X_train, y_train, group=group_train)


importances = lgbm_ranker.feature_importances_
feature_names = X_train.columns

plt.figure(figsize=(8, 4))
plt.barh(feature_names, importances)
plt.xlabel("Importance")
plt.title("LGBMRanker Feature Importances")
plt.show()


y_test_pred = lgbm_ranker.predict(X_test)


test_df = test_df.copy()
test_df["pred_score"] = y_test_pred


# Sort by predicted score per customer
test_df_sorted = test_df.sort_values(by=["customer_id", "pred_score"], ascending=[True, False])

# Take top 12 predictions per user
top12 = test_df_sorted.groupby("customer_id").head(12)


from collections import defaultdict

# Store ground-truth relevant (positive) article IDs for each customer in the dict "truth"
truth = defaultdict(set)
for cid, aid, label in zip(test_df["customer_id"], test_df["article_id"], test_df["label"]):
    if label == 1:
        truth[cid].add(aid)

# Compute MAP@12
def compute_map_12(actual, predicted, k=12):
    if not actual:
        return 0.0
    
    predicted = predicted[:k]
    score, num_hits = 0.0, 0.0
    
    # Look at their top-12 predicted items
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            # Award more score for correct items that appear earlier in the list
            score += num_hits / (i + 1.0) #  weights precision by position
    
    return score / min(len(actual), k)

map12_scores = []
for cid, group in top12.groupby("customer_id"):
    # Retrieve customer's predicted top 12 articles
    pred_articles = group["article_id"].tolist()
    
    # Retrieve customer's true purchased articles.
    actual_articles = truth[cid]

    # calculate customer's map12 score
    map12_scores.append(compute_map_12(actual_articles, pred_articles))

mean_map12 = np.mean(map12_scores)
print(f"MAP@12: {mean_map12:.4f}")




