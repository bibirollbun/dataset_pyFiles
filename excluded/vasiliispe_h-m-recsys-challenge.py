!pip install --upgrade implicit


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error

from scipy.sparse import coo_matrix, csr_matrix
import implicit
import time
from tqdm import tqdm
from implicit.als import AlternatingLeastSquares


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', '{:.4f}'.format)


base_path = '/kaggle/input/h-and-m-personalized-fashion-recommendations/'
csv_transactions = f'{base_path}transactions_train.csv'
csv_sample_submission = f'{base_path}sample_submission.csv'
csv_customers = f'{base_path}customers.csv'
csv_articles = f'{base_path}articles.csv'

df_transactions = pd.read_csv(csv_transactions, dtype={'article_id': str}, parse_dates=['t_dat'])
df_sample_submission = pd.read_csv(csv_sample_submission)
df_customers = pd.read_csv(csv_customers)
df_articles = pd.read_csv(csv_articles, dtype={'article_id': str})


df_transactions.head()


df_sample_submission.head()


df_customers.head()


df_articles.head()


print(f"Number of rows and columns of sample submission: {df_customers.shape}")
print(f"Number of rows and columns of articles: {df_articles.shape,}")
print(f"Number of rows and columns of users: {df_customers.shape}")
print(f"Number of rows and columns of users: {df_transactions.shape}")


print(f"Start date: {df_transactions['t_dat'].min()}")
print(f"End date: {df_transactions['t_dat'].max()}")
print(f"Timespan: {(df_transactions['t_dat'].max() - df_transactions['t_dat'].min()).days} days")


df_transactions['month_year'] = df_transactions['t_dat'].dt.to_period('M')
monthly_sales = df_transactions.groupby('month_year').size().reset_index(name='sales_count')

plt.figure(figsize=(12, 6))
sns.lineplot(x=[str(p) for p in monthly_sales['month_year']], y=monthly_sales['sales_count'])
plt.title('monthly sales')
plt.xlabel('month')
plt.ylabel('transactions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


sales_channels = df_transactions['sales_channel_id'].value_counts().reset_index()
sales_channels.columns = ['channel', 'count']

plt.figure(figsize=(8, 6))
sns.barplot(x='channel', y='count', data=sales_channels, palette='viridis')
plt.title('Channel sales distribution')
plt.xlabel('Sales Channel')
plt.ylabel('transactions')
plt.show()


df_transactions['week'] = df_transactions['t_dat'].dt.isocalendar().week
weekly_avg_price = df_transactions.groupby(['week'])['price'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(x='week', y='price', data=weekly_avg_price, color='green')
plt.title('avg week price trend')
plt.xlabel('week')
plt.ylabel('avg price')
plt.grid(True, alpha=0.3)
plt.show()


plt.figure(figsize=(12, 6))
sns.histplot(df_customers['age'].dropna(), bins=70, kde=True, color='orange')
plt.title('age distribution')
plt.xlabel('age')
plt.ylabel('num of customers')
plt.show()


fashion_news = df_customers['fashion_news_frequency'].value_counts().reset_index()
fashion_news.columns = ['frequency', 'count']

plt.figure(figsize=(10, 6))
sns.barplot(x='frequency', y='count', data=fashion_news, palette='Set2')
plt.title('fashion news distribution')
plt.xlabel('frequency')
plt.ylabel('num of news')
plt.xticks(rotation=45)
plt.show()


club_status = df_customers['club_member_status'].value_counts().reset_index()
club_status.columns = ['status', 'count']

plt.figure(figsize=(10, 6))
plt.pie(club_status['count'], labels=club_status['status'], shadow=True, startangle=90, colors=sns.color_palette('pastel'))
plt.title('club status distribution')
plt.axis('equal')
plt.show()


product_types = df_articles['product_type_name'].value_counts().head(15).reset_index()
product_types.columns = ['product_type', 'count']

plt.figure(figsize=(12, 8))
sns.barplot(x='count', y='product_type', data=product_types)
plt.title('top 15 Products')
plt.xlabel('num of each type')
plt.ylabel('type')
plt.show()


product_groups = df_articles['product_group_name'].value_counts().reset_index()
product_groups.columns = ['product_group', 'count']

plt.figure(figsize=(12, 8))
sns.barplot(x='count', y='product_group', data=product_groups)
plt.title('groups distribution')
plt.xlabel('num of groups')
plt.ylabel('group')
plt.show()


colors = df_articles['perceived_colour_master_name'].value_counts().reset_index()
colors.columns = ['color', 'count']

plt.figure(figsize=(12, 8))
sns.barplot(x='count', y='color', data=colors)
plt.title('colour distribution')
plt.xlabel('num of colour')
plt.ylabel('colour')
plt.show()


# distribution is highly scewed. Some customres make way too many transactions.
customer_purchase_counts = df_transactions.groupby('customer_id').size().reset_index(name='purchase_count')

plt.figure(figsize=(10, 6))
sns.histplot(customer_purchase_counts['purchase_count'].clip(upper=50), bins=50, kde=True)
plt.title('number of transations per customer')
plt.xlabel('num of purchases')
plt.ylabel('num of customers')
plt.show()


# top 10 customers
customer_purchase_counts.sort_values('purchase_count', ascending=False).head(10)


# analogously top 10 items
popular_items = df_transactions.groupby('article_id').size().reset_index(name='purchase_count')
popular_items = popular_items.sort_values('purchase_count', ascending=False).head(10)

popular_items


# tyoe of products and name for top items. Mostly a little purchases, like trousers and socks
top_items_details = pd.merge(popular_items, df_articles[['article_id', 'prod_name', 'product_type_name']], on='article_id', how='left')

top_items_details


# What's the most and least popular in terms of price? 
# Wow women tops and bra among the most popular choices. Some of the most unpopular are 
item_price_popularity = df_transactions.groupby('article_id').agg({'price': 'mean', 'article_id': 'count'})
item_price_popularity.columns = ['avg_price', 'popularity']
item_price_popularity = item_price_popularity.reset_index().merge(df_articles[['article_id', 'prod_name', 'product_type_name']], on='article_id', how='left')
item_price_popularity




plt.figure(figsize=(10, 6))
sns.scatterplot(x='avg_price', y='popularity', data=item_price_popularity.sample(10000), alpha=0.5)
plt.xscale('log')
plt.yscale('log')
plt.title('log of Price vs Popularity')
plt.xlabel('avg price')
plt.ylabel('num of purchases')
plt.grid(True, alpha=0.3)
plt.show()


# price analysis by product category
merged_data = pd.merge(df_transactions, df_articles[['article_id', 'product_group_name', 'product_type_name']], 
                      on='article_id', how='left')

avg_price_by_group = merged_data.groupby('product_group_name')['price'].mean().reset_index()
avg_price_by_group = avg_price_by_group.sort_values('price', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='price', y='product_group_name', data=avg_price_by_group)
plt.title('avg price by product group')
plt.xlabel('avg price')
plt.ylabel('group')
plt.show()


# price distributions within 5 top groups
top_5_groups = avg_price_by_group.head(5)['product_group_name'].tolist()
top_groups_data = merged_data[merged_data['product_group_name'].isin(top_5_groups)]

plt.figure(figsize=(14, 8))
sns.boxplot(x='product_group_name', y='price', data=top_groups_data)
plt.title('price distribution for 5 top groups')
plt.xlabel('product group')
plt.ylabel('price')
plt.grid()
plt.tight_layout()
plt.show()


# purchase patterns search

merged_cust_data = pd.merge(df_transactions, df_customers[['customer_id', 'age']], on='customer_id', how='left')
merged_cust_data = merged_cust_data.dropna(subset=['age'])

bins_list = [0, 20, 30, 40, 50, 60, 100]
names_list = ['<20', '20-30', '30-40', '40-50', '50-60', '60+']

merged_cust_data['age_bin'] = pd.cut(merged_cust_data['age'], bins=bins_list, labels=names_list)

age_group_spending = merged_cust_data.groupby('age_bin')['price'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='age_bin', y='price', data=age_group_spending, palette='viridis')
plt.title('avg purchise Price per age group')
plt.xlabel('age group')
plt.ylabel('avg price')
plt.grid(True, alpha=0.3)
plt.show()


# Most selling product per age group

key_age_groups = ['20-30', '30-40', '50-60']
age_product_data = pd.merge(merged_cust_data, df_articles[['article_id', 'product_type_name']], on='article_id', how='left')

for age_group in key_age_groups:
    group_data = age_product_data[age_product_data['age_bin'] == age_group]
    top_products = group_data['product_type_name'].value_counts().head(5)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_products.values, y=top_products.index)
    plt.title(f'top 5 products per age group: {age_group}')
    plt.xlabel('how many purchases')
    plt.ylabel('type')
    plt.tight_layout()
    plt.show()


# Transactions per customer
transactions_per_customer = df_transactions.groupby('customer_id').size().reset_index(name='transaction_count')

plt.figure(figsize=(12, 6))
sns.histplot(data=transactions_per_customer, x='transaction_count', bins=50, kde=True, color='green')
plt.title('distribution of transactions for customer', fontsize=14)
plt.xlabel('num of transactions', fontsize=12)
plt.ylabel('num of customers', fontsize=12)
plt.xlim(0, transactions_per_customer['transaction_count'].quantile(0.99))
plt.show()


# convert customer_id to numeric format
df_transactions['customer_id_int'] = df_transactions['customer_id'].apply(lambda x: int(x[-16:], 16))

# let's filter unnecesary columns
df_transactions = df_transactions[['t_dat', 'customer_id', 'customer_id_int', 'article_id']]

df_sample_submission['customer_id_int'] = df_sample_submission['customer_id'].apply(lambda x: int(x[-16:], 16)) # same for submission dataframe


# get last date from all transactions
last_date = df_transactions['t_dat'].max()
print(f'Last date in all transactions: {last_date}')

# define test start date as 7 days before the last date
test_start_date = last_date - timedelta(days=7)
print(f'Test start date: {test_start_date}')

# test transactions are transactions from the last week
df_transactions_test = df_transactions[df_transactions['t_dat'] >= test_start_date].copy()
print(f'Test transactions dates from {df_transactions_test["t_dat"].min()} to {df_transactions_test["t_dat"].max()}, shape: {df_transactions_test.shape}')

# train transactions are all transactions before the test week
df_transactions_train = df_transactions[df_transactions['t_dat'] < test_start_date].copy()
print(f'Train transactions dates from {df_transactions_train["t_dat"].min()} to {df_transactions_train["t_dat"].max()}, shape: {df_transactions_train.shape}')

# get the last date of the training period
last_date_train = df_transactions_train['t_dat'].max()
print(f'Last date in train transactions: {last_date_train}')


def evaluate_recommendations(actual_df, pred_dfs, k=12):
    """
    Main metrics evaluation function. Can receive list of predictions

    Input: actual_df - dataframe with actual purchases
    pred_dfs : list of dataframes with predicted recommendations
    k : num of K at MAP
    Returns: metrics dataframe
    """

    # Actual purchases to dict
    actual_purchases = defaultdict(set)
    for _, row in actual_df.iterrows():
        actual_purchases[row['customer_id_int']].add(row['article_id']) # Используем 'customer_id_int'

    results = {
        f'map@{k}': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'rmse': [],
    }

    total_items = len(actual_df['article_id'].unique())

    for pred_df in pred_dfs:
        # track metrics
        ap_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        rmse_scores = []
        predicted_items_set = set()
        recommendation_count = 0
        pop_scores = []

        # loop for all predictions
        for _, row in pred_df.iterrows():
            customer_id = row['customer_id_int']

            # no purchases
            if customer_id not in actual_purchases:
                continue

            # Get items
            if isinstance(row['prediction'], str):
                pred_items = row['prediction'].strip().split()[:k]
            else:
                pred_items = []

            for item in pred_items:
                predicted_items_set.add(item)
            recommendation_count += len(pred_items)

            # no predictions
            if not pred_items:
                continue

            # get items for customer
            actual_items = actual_purchases[customer_id]

            hits = 0
            sum_precisions = 0

            for i, item in enumerate(pred_items):
                if item in actual_items:
                    hits += 1
                    precision_at_i = hits / (i + 1)
                    sum_precisions += precision_at_i

            if hits > 0:
                ap = sum_precisions / min(len(actual_items), k)
            else:
                ap = 0
            ap_scores.append(ap)

            # Calculate metrics
            true_positives = len(set(pred_items) & actual_items)
            precision = true_positives / len(pred_items) if pred_items else 0
            recall = true_positives / len(actual_items) if actual_items else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

            # RMSE for binary variant, when 1 if there is a hit, 0 otherwise
            actual_vector = np.zeros(total_items)
            pred_vector = np.zeros(total_items)

            for item in actual_items:
                try:
                    item_idx = int(item) % total_items
                except ValueError:
                    item_idx = hash(item) % total_items
                actual_vector[item_idx] = 1

            for item in pred_items:
                try:
                    item_idx = int(item) % total_items
                except ValueError:
                    item_idx = hash(item) % total_items
                pred_vector[item_idx] = 1

            rmse = np.sqrt(mean_squared_error(actual_vector, pred_vector))
            rmse_scores.append(rmse)

        results[f'map@{k}'].append(np.mean(ap_scores) if ap_scores else 0)
        results['precision'].append(np.mean(precision_scores) if precision_scores else 0)
        results['recall'].append(np.mean(recall_scores) if recall_scores else 0)
        results['f1_score'].append(np.mean(f1_scores) if f1_scores else 0)
        results['rmse'].append(np.mean(rmse_scores) if rmse_scores else 0)

    metrics_df = pd.DataFrame(results)
    return metrics_df


# function to ensure, we have exactly 12 predictions for each customer

def ensure_12_items(predicted_items, popular_items):
    if not isinstance(predicted_items, list):
        predicted_items = []  

    if len(predicted_items) >= 12:
        return predicted_items[:12]
    else:
        remaining = 12 - len(predicted_items)
        popular_to_add = [item for item in popular_items if item not in predicted_items] 
        return predicted_items + popular_to_add[:remaining]


cutoff_date = last_date_train - timedelta(days=30)
print(f'Cutoff date for recent transactions: {cutoff_date}')

# get transactions for last 30 days
df_recent_transactions_train = df_transactions_train[df_transactions_train['t_dat'] >= cutoff_date].copy()
print(f"Transactions for recent period from {cutoff_date} to {last_date_train}: {df_recent_transactions_train.shape}")

# how many days ago each purchase was made
df_recent_transactions_train['recency'] = (last_date_train - df_recent_transactions_train['t_dat']).dt.days

df_recent_transactions_train.head()


# we need to know the latest min recency for each cusomter_id and article id pairs
customer_article_metrics_baseline_1 = (df_recent_transactions_train
                                    .groupby(['customer_id_int', 'article_id'])
                                    .agg(purchase_count=('t_dat', 'count'),
                                         last_purchase=('t_dat', 'max'),
                                         min_recency=('recency', 'min'))
                                    .reset_index()
                                   )


customer_article_metrics_baseline_1.head()


customer_article_metrics_baseline_1.head()


# sorting the metrics. Purchase sorting is irrelevant. We want to know only customer id and it's most recent number of days
customer_article_metrics_baseline_1 = customer_article_metrics_baseline_1.sort_values(
    ['customer_id_int', 'purchase_count', 'min_recency'],
    ascending=[True, False, True]
)

# previous sorting was neeeded for this drop of duplicates. Only top recensy remain
customer_article_metrics_baseline_1 = customer_article_metrics_baseline_1.drop_duplicates(['customer_id_int', 'article_id'])


customer_article_metrics_baseline_1.head()



# Convert customer_article_metrics to dictionary for faster search- hashtable is fast

customer_items_baseline_1 = {}
for _, row in customer_article_metrics_baseline_1.iterrows():
    customer_id = row['customer_id_int']
    article_id = row['article_id']

    if customer_id not in customer_items_baseline_1:
        customer_items_baseline_1[customer_id] = []

    if len(customer_items_baseline_1[customer_id]) < 12:  # need 12 items for customer
        customer_items_baseline_1[customer_id].append(article_id)


# solving issues with cold start and filling missing predictions
# popular items from last 2 weeks

last_2weeks_date = last_date_train - timedelta(days=14)
last_2weeks_transactions_train = df_transactions_train[df_transactions_train['t_dat'] >= last_2weeks_date] 
popular_items_baseline_1 = last_2weeks_transactions_train['article_id'].value_counts().head(12).index.tolist()

print(popular_items_baseline_1)


# Create predictions dataframe
predictions_baseline_1 = []
for customer_id, items in customer_items_baseline_1.items():
    items_str = ' '.join(items)
    predictions_baseline_1.append((customer_id, items_str))

df_predictions_baseline_1 = pd.DataFrame(predictions_baseline_1, columns=['customer_id_int', 'prediction'])



predictions_list_baseline_1 = []
for customer_id, items in customer_items_baseline_1.items():
    items_str = ' '.join(map(str, items)) # items to strings
    predictions_list_baseline_1.append((customer_id, items_str))

df_predictions_baseline_1 = pd.DataFrame(predictions_list_baseline_1, columns=['customer_id_int', 'prediction'])

print(df_predictions_baseline_1.head())


# # submission template
# submission_baseline_1 = df_sample_submission.merge(
#     df_predictions_baseline_1, on='customer_id_int', how='left'
# ).fillna('')

# # filling na
# submission_baseline_1['prediction'] = submission_baseline_1['prediction'].apply(
#     lambda x: x if x != '' else ' '.join(map(str, popular_items_baseline_1[:12])) 
# )

# submission_baseline_1['prediction'] = submission_baseline_1['prediction'].apply(lambda x: ensure_12_items(x, popular_items_baseline_1))

# submission_baseline_1_final = submission_baseline_1[['customer_id', 'prediction']]

# print(submission_baseline_1_final.head())


metrics_baseline_one = evaluate_recommendations(
    df_transactions_test, 
    [df_predictions_baseline_1],
    k=12
)

metrics_baseline_one


# like for first baseline
cutoff_date_baseline_2 = last_date_train - timedelta(days=30)
df_transactions_train_baseline_2 = df_transactions[df_transactions['t_dat'] < test_start_date].copy()
last_date_train_baseline_2 = df_transactions_train_baseline_2['t_dat'].max() 
df_recent_transactions_train_baseline_2 = df_transactions_train_baseline_2[df_transactions_train_baseline_2['t_dat'] >= cutoff_date_baseline_2].copy() 
df_recent_transactions_train_baseline_2['recency'] = (last_date_train_baseline_2 - df_recent_transactions_train_baseline_2['t_dat']).dt.days



customer_article_metrics_baseline_2 = (df_recent_transactions_train_baseline_2 # Specific name for baseline 2
                                    .groupby(['customer_id_int', 'article_id'])
                                    .agg(purchase_count=('t_dat', 'count'),
                                         last_purchase=('t_dat', 'max'),
                                         min_recency=('recency', 'min'))
                                    .reset_index()
                                   )

customer_article_metrics_baseline_2.head()


# remove duplicates - like for first baseline

customer_article_metrics_baseline_2 = customer_article_metrics_baseline_2.sort_values(['customer_id_int', 'purchase_count', 'min_recency'],
    ascending=[True, False, True])

customer_article_metrics_baseline_2 = customer_article_metrics_baseline_2.drop_duplicates(['customer_id_int', 'article_id']) 

customer_article_metrics_baseline_2.head()



# convert for fast search

customer_items_baseline_2_strategy_1 = {} 
for _, row in customer_article_metrics_baseline_2.iterrows(): 
    customer_id = row['customer_id_int']
    article_id = row['article_id']

    if customer_id not in customer_items_baseline_2_strategy_1:
        customer_items_baseline_2_strategy_1[customer_id] = [] 

    if len(customer_items_baseline_2_strategy_1[customer_id]) < 12: 
        customer_items_baseline_2_strategy_1[customer_id].append(article_id) 


temp_df_baseline_2 = df_transactions_train_baseline_2.sort_values(['customer_id_int', 't_dat'])

pairs_dict_baseline_2 = {} 
window_days = 7

sample_size = min(10000, len(temp_df_baseline_2['customer_id_int'].unique())) 
customer_sample = np.random.choice(temp_df_baseline_2['customer_id_int'].unique(), size=sample_size, replace=False) 

for customer_id in customer_sample:
    customer_purchases = temp_df_baseline_2[temp_df_baseline_2['customer_id_int'] == customer_id] 

    for i, row1 in customer_purchases.iterrows():
        item1 = row1['article_id']
        purchase_date = row1['t_dat']

        window_purchases = customer_purchases[
            (customer_purchases['t_dat'] >= purchase_date) &
            (customer_purchases['t_dat'] <= purchase_date + timedelta(days=window_days)) &
            (customer_purchases['article_id'] != item1)
        ]

        for item2 in window_purchases['article_id'].unique():
            pair = (item1, item2)
            if pair not in pairs_dict_baseline_2: 
                pairs_dict_baseline_2[pair] = 0 
            pairs_dict_baseline_2[pair] += 1 

# Convert dict to dataframe
pairs_df_baseline_2 = pd.DataFrame([(k[0], k[1], v) for k, v in pairs_dict_baseline_2.items()], columns=['item1', 'item2', 'count'])
pairs_df_baseline_2 = pairs_df_baseline_2.sort_values('count', ascending=False)  

item_to_pair_baseline_2 = {} 
for _, row in pairs_df_baseline_2.iterrows(): 
    item1 = row['item1']
    item2 = row['item2']
    if item1 not in item_to_pair_baseline_2: 
        item_to_pair_baseline_2[item1] = item2 



customer_items_baseline_2 = customer_items_baseline_2_strategy_1.copy() 

for customer_id, items in customer_items_baseline_2.items():
    paired_items = []
    for item in items:
        if item in item_to_pair_baseline_2: 
            paired_items.append(item_to_pair_baseline_2[item]) 

    for item in paired_items:
        if item not in customer_items_baseline_2[customer_id] and len(customer_items_baseline_2[customer_id]) < 12:
            customer_items_baseline_2[customer_id].append(item)


# predictions 
predictions_list_baseline_2 = []
for customer_id, items in customer_items_baseline_2.items():
    items_str = ' '.join(map(str, items)) 
    predictions_list_baseline_2.append((customer_id, items_str))

df_predictions_baseline_2 = pd.DataFrame(predictions_list_baseline_2, columns=['customer_id_int', 'prediction'])

df_predictions_baseline_2.head()


# submission_baseline_2 = df_sample_submission.merge(df_predictions_baseline_2, on='customer_id_int', how='left').fillna('')

# submission_baseline_2['prediction'] = submission_baseline_2['prediction'].apply(lambda x: x if x != '' else ' '.join(map(str, popular_items_baseline_2[:12])) )

# submission_baseline_2['prediction'] = submission_baseline_2['prediction'].apply(lambda x: ensure_12_items(x, popular_items_baseline_2)) 

# submission_baseline_2_final = submission_baseline_2[['customer_id', 'prediction']]

# print(submission_baseline_2_final.head())


metrics_baseline_two = evaluate_recommendations(
    df_transactions_test,
    [df_predictions_baseline_2],
    k=12
)

metrics_baseline_two





# This function will create sparse matrix - need this for collaborative filtering
def create_user_item_matrix(df, all_users, all_items):
    # Need integer indices for sparse matrix
    row = df['user_idx'].astype(int).values
    col = df['item_idx'].astype(int).values
    data = np.ones(df.shape[0]) 
    
    coo = coo_matrix((data, (row, col)), shape=(len(all_users), len(all_items)))
    return coo


# function returning recommendations. There is several datasets, like submission and test datasets
def generate_recommendations(model, user_indices, csr_train, item_ids, num_recommendations=12):
    recommendations = {}
    
    # Loop through users to get recommendations
    for user_idx in tqdm(user_indices):
        user_idx = int(user_idx)  # Convert to int just to be safe
        
        user_items = csr_train[user_idx] if user_idx < csr_train.shape[0] else None
        
        try:
            recommended_items, _ = model.recommend(
                user_idx, 
                user_items,
                N=num_recommendations,
                filter_already_liked_items=True  # do not recommend items already existed
            )
            
            # Need indexes for further metrics evaluation
            rec_items = [item_ids[int(item_idx)] for item_idx in recommended_items]
            recommendations[user_idx] = rec_items
        except Exception as e: # if error encountered. In some tests before fixes has been a case
            recommendations[user_idx] = []
    
    return recommendations


def prepare_submission(recommendations, user_id_mapping, submission_df):
    submission = submission_df.copy()
    
    id_to_recs = {user_id_mapping[user_idx]: recs 
                  for user_idx, recs in recommendations.items() 
                  if user_idx in user_id_mapping}
    
    submission['prediction'] = submission['customer_id_int'].map(
        lambda x: ' '.join(id_to_recs.get(x, [])) if x in id_to_recs else ''
    )
    
    return submission


def train_model(matrices, factors=100, iterations=15, regularization=0.01):
    coo_train = matrices['coo_train']
    
    model_als = AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=42
    )
    
    model_als.fit(coo_train)
    
    return model_als


# Data preparation - let's focus on recent purchases only
print("Checking original data size:", df_transactions.shape)

# Filter to recent data (last 60 days) - older data isn't that useful anyway
cutoff_date = df_transactions['t_dat'].max() - timedelta(days=60)
filtered_transactions = df_transactions[df_transactions['t_dat'] > cutoff_date].copy()
print("Using only recent transactions:", filtered_transactions.shape)


# get unique users and items
als_users = filtered_transactions['customer_id_int'].unique().tolist()
als_items = filtered_transactions['article_id'].unique().tolist()

# ids to indexes
als_user_map = {user_id: idx for idx, user_id in enumerate(als_users)}
als_item_map = {item_id: idx for idx, item_id in enumerate(als_items)}

# Add index columns to the dataframe
als_df = filtered_transactions.copy()
als_df['user_idx'] = als_df['customer_id_int'].map(als_user_map).astype(int)
als_df['item_idx'] = als_df['article_id'].map(als_item_map).astype(int)

als_user_ids = {idx: user_id for user_id, idx in als_user_map.items()}
als_item_ids = {idx: item_id for item_id, idx in als_item_map.items()}

print(len(als_users), len(als_items))


# train test split for als model
validation_days = 7
validation_cutoff = als_df['t_dat'].max() - pd.Timedelta(days=validation_days)
df_train = als_df[als_df['t_dat'] < validation_cutoff]
df_val = als_df[als_df['t_dat'] >= validation_cutoff]

# training matrix for als
coo_train = create_user_item_matrix(df_train, als_users, als_items)
csr_train = coo_train.tocsr()  # CSR format is faster for some operations

# test matrix for als
coo_val = create_user_item_matrix(df_val, als_users, als_items)
csr_val = coo_val.tocsr()

# dict of metrics
als_matrices = {
    'coo_train': coo_train,
    'csr_train': csr_train,
    'csr_val': csr_val,
    'df_train': df_train,
    'df_val': df_val
}


# Search for best latent factors
factors_to_test = list(range(10, 211, 20)) 
iterations = 20  # will leave as it is iterations and regularization
regularization = 0.01  

results = []

# loop over factors
for factors in factors_to_test:
    print(f"current factors ={factors}...")
    try:
        model = train_model(
            als_matrices,
            factors=factors,
            iterations=iterations,
            regularization=regularization
        )
        
        # predictions
        val_users = df_val['user_idx'].unique()[:100] 
        
        val_recommendations = {}
        for user_idx in val_users:
            user_idx = int(user_idx)
            try:
                rec_items, _ = model.recommend(
                    user_idx,
                    csr_train[user_idx],
                    N=12,
                    filter_already_liked_items=True
                )
                val_recommendations[als_user_ids[user_idx]] = [als_item_ids[int(item)] for item in rec_items]
            except:
                val_recommendations[als_user_ids[user_idx]] = []
        
        # put predictions to dataframe
        val_pred_df = pd.DataFrame({
            'customer_id_int': list(val_recommendations.keys()),
            'prediction': [' '.join(items) for items in val_recommendations.values()]
        })
        
        # get metrics
        metrics = evaluate_recommendations(df_val, [val_pred_df], k=12)
        
        print(f"Factors: {factors} - MAP@12: {metrics['map@12'][0]:.4f} - "
              f"Precision: {metrics['precision'][0]:.4f} - Recall: {metrics['recall'][0]:.4f}")
        
        results.append({
            'factors': factors,
            'map': metrics['map@12'][0],
            'precision': metrics['precision'][0],
            'recall': metrics['recall'][0],
            'f1_score': metrics['f1_score'][0],
            'rmse': metrics['rmse'][0]
        })
    except Exception as e:
        print(f"there is error")
        print(e)


# Find best parameters
results_df = pd.DataFrame(results)
print(results_df)

# best model and MAP@12
best_params = results_df.loc[results_df['map'].idxmax()].to_dict()
best_params


# train best model on full dataset
best_factors = int(best_params['factors'])

full_coo_train = create_user_item_matrix(als_df, als_users, als_items)
full_csr_train = full_coo_train.tocsr()

final_model = train_model(
    {'coo_train': full_coo_train},
    factors=best_factors,
    iterations=iterations,
    regularization=regularization
)


# Generate recommendations for all users in the submission dataframe
submission_users = df_sample_submission['customer_id_int'].unique()

submission_user_indices = []
for user_id in submission_users:
    if user_id in als_user_map:
        submission_user_indices.append(als_user_map[user_id])

recommendations = generate_recommendations(
    final_model,
    submission_user_indices,
    full_csr_train,
    als_item_ids,
    num_recommendations=12
)





# # Prepare submission
# als_submission = prepare_submission(
#     recommendations,
#     als_user_ids,
#     df_sample_submission
# )

# # handling missings
# popular_items = als_df['article_id'].value_counts().index.tolist()[:12]

# empty_recs = als_submission[als_submission['prediction'] == ''] # no recommendations
# print(len(empty_recs))

# for idx in empty_recs.index:
#     als_submission.loc[idx, 'prediction'] = ' '.join(ensure_12_items([], popular_items))


# # Save submission file
# als_submission[['customer_id', 'prediction']].to_csv('als_recommendations.csv', index=False)
# print("Saved submission file as 'als_recommendations.csv'")

# # Run full evaluation on our validation set
# # Create predictions dataframe for validation set
# val_users = df_val['customer_id_int'].unique()

# # Match format for evaluation
# val_preds = []
# for user_id in val_users:
#     if user_id in als_user_map:
#         user_idx = als_user_map[user_id]
#         try:
#             rec_items, _ = final_model.recommend(
#                 user_idx,
#                 full_csr_train[user_idx],
#                 N=12,
#                 filter_already_liked_items=True
#             )
#             val_preds.append({
#                 'customer_id_int': user_id,
#                 'prediction': ' '.join([als_item_ids[int(item)] for item in rec_items])
#             })
#         except:
#             val_preds.append({
#                 'customer_id_int': user_id,
#                 'prediction': ' '.join(popular_items)
#             })
#     else:
#         val_preds.append({
#             'customer_id_int': user_id,
#             'prediction': ' '.join(popular_items)
#         })

# val_pred_df = pd.DataFrame(val_preds)

# # Calculate final metrics
# final_metrics = evaluate_recommendations(df_val, [val_pred_df], k=12)
# print("\nFinal model performance:")
# print(f"MAP@12: {final_metrics['map@12'][0]:.4f}")
# print(f"Precision: {final_metrics['precision'][0]:.4f}")
# print(f"Recall: {final_metrics['recall'][0]:.4f}")
# print(f"F1 Score: {final_metrics['f1_score'][0]:.4f}")
# print(f"RMSE: {final_metrics['rmse'][0]:.4f}")




