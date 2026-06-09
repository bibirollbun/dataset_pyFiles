# """
# H&M Personalized Fashion Recommendations - Baseline Implementation
# Following the baseline plan for MLE-Bench competition
# Expected MAP@12: 0.012-0.015
# Runtime: 30-60 minutes
# """

# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from collections import defaultdict
# import warnings
# warnings.filterwarnings('ignore')

# def reduce_mem_usage(df):
#     """
#     Reduce memory usage by downcasting numeric dtypes.
#     This is critical for handling the large H&M dataset.
#     """
#     start_mem = df.memory_usage().sum() / 1024**2
#     print(f'Memory usage: {start_mem:.2f} MB')

#     for col in df.columns:
#         col_type = df[col].dtype

#         if col_type != object:
#             c_min = df[col].min()
#             c_max = df[col].max()

#             if str(col_type)[:3] == 'int':
#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                     df[col] = df[col].astype(np.int8)
#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                     df[col] = df[col].astype(np.int16)
#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                     df[col] = df[col].astype(np.int32)
#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
#                     df[col] = df[col].astype(np.int64)
#             else:
#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
#                     df[col] = df[col].astype(np.float32)
#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                     df[col] = df[col].astype(np.float32)
#                 else:
#                     df[col] = df[col].astype(np.float64)

#     end_mem = df.memory_usage().sum() / 1024**2
#     print(f'Memory usage after optimization: {end_mem:.2f} MB')
#     print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')

#     return df


# def calculate_popular_items(transactions_df, cutoff_date, n=20):
#     """
#     Calculate the most popular items in the last 7 days before cutoff_date.
#     These serve as fallback recommendations for customers with limited history.
#     """
#     last_week = cutoff_date - timedelta(days=7)
#     recent_transactions = transactions_df[transactions_df['t_dat'] >= last_week].copy()

#     popular_items = (recent_transactions['article_id']
#                     .value_counts()
#                     .head(n)
#                     .index.tolist())

#     print(f"Calculated {len(popular_items)} popular items from last 7 days")
#     return popular_items


# def get_customer_recommendations(customer_id, transactions_df, cutoff_date, 
#                                  popular_items, n=12, lookback_days=30):
#     """
#     Generate recommendations for a single customer using time-weighted purchase history.

#     Strategy:
#     1. Extract customer's purchases from last 30 days before cutoff_date
#     2. Apply time decay weighting: weight = 1 / (1 + days_ago)
#     3. Aggregate weights by article_id
#     4. Select top 12 by weight
#     5. Fill gaps with popular items if < 12 recommendations

#     Args:
#         customer_id: Customer identifier
#         transactions_df: Transactions dataframe
#         cutoff_date: Date to predict from (e.g., validation start date)
#         popular_items: List of popular item fallbacks
#         n: Number of recommendations to generate (default 12)
#         lookback_days: How far back to look for purchase history (default 30)

#     Returns:
#         List of article_ids (length = n)
#     """
#     # Extract customer's recent purchase history
#     lookback_date = cutoff_date - timedelta(days=lookback_days)
#     customer_history = transactions_df[
#         (transactions_df['customer_id'] == customer_id) &
#         (transactions_df['t_dat'] >= lookback_date) &
#         (transactions_df['t_dat'] < cutoff_date)
#     ].copy()

#     if len(customer_history) == 0:
#         # No purchase history - return popular items
#         return popular_items[:n]

#     # Calculate time decay weights
#     customer_history['days_ago'] = (cutoff_date - customer_history['t_dat']).dt.days
#     customer_history['weight'] = 1.0 / (1.0 + customer_history['days_ago'])

#     # Aggregate weights by article_id
#     article_weights = (customer_history.groupby('article_id')['weight']
#                       .sum()
#                       .sort_values(ascending=False))

#     # Get top recommendations
#     recommendations = article_weights.head(n).index.tolist()

#     # Fill with popular items if we have fewer than n recommendations
#     if len(recommendations) < n:
#         # Add popular items that aren't already in recommendations
#         for item in popular_items:
#             if item not in recommendations:
#                 recommendations.append(item)
#                 if len(recommendations) == n:
#                     break

#     return recommendations[:n]


# def generate_all_recommendations(transactions_df, customer_ids, cutoff_date, 
#                                 popular_items, n=12):
#     """
#     Generate recommendations for all customers.

#     Args:
#         transactions_df: Transactions dataframe
#         customer_ids: List of customer IDs to generate predictions for
#         cutoff_date: Date to predict from
#         popular_items: List of popular item fallbacks
#         n: Number of recommendations per customer

#     Returns:
#         Dictionary mapping customer_id to list of article_ids
#     """
#     recommendations = {}
#     total_customers = len(customer_ids)

#     print(f"Generating recommendations for {total_customers} customers...")

#     for idx, customer_id in enumerate(customer_ids):
#         if (idx + 1) % 10000 == 0:
#             print(f"Processed {idx + 1}/{total_customers} customers")

#         recs = get_customer_recommendations(
#             customer_id, transactions_df, cutoff_date, 
#             popular_items, n=n
#         )
#         recommendations[customer_id] = recs

#     print(f"Completed recommendations for all {total_customers} customers")
#     return recommendations


# def create_submission(recommendations, output_file='submission.csv'):
#     """
#     Create submission file in the required format.

#     Format: customer_id,prediction
#     where prediction is space-separated list of 12 article_ids
#     """
#     submission_data = []

#     for customer_id, article_list in recommendations.items():
#         # Convert article_ids to strings and join with spaces
#         prediction = ' '.join([str(article_id).zfill(10) for article_id in article_list])
#         submission_data.append({
#             'customer_id': customer_id,
#             'prediction': prediction
#         })

#     submission_df = pd.DataFrame(submission_data)
#     submission_df.to_csv(output_file, index=False)
#     print(f"Submission saved to {output_file}")
#     print(f"Total predictions: {len(submission_df)}")

#     return submission_df


# def calculate_map_at_k(actual, predicted, k=12):
#     """
#     Calculate Mean Average Precision at K for validation.

#     Args:
#         actual: Dictionary mapping customer_id to list of actual purchased article_ids
#         predicted: Dictionary mapping customer_id to list of predicted article_ids
#         k: Number of recommendations (default 12)

#     Returns:
#         MAP@K score
#     """
#     aps = []

#     for customer_id, pred_items in predicted.items():
#         if customer_id not in actual:
#             continue

#         actual_items = set(actual[customer_id])
#         if len(actual_items) == 0:
#             continue

#         pred_items = pred_items[:k]

#         score = 0.0
#         num_hits = 0.0

#         for i, item in enumerate(pred_items):
#             if item in actual_items:
#                 num_hits += 1.0
#                 score += num_hits / (i + 1.0)

#         if len(actual_items) > 0:
#             aps.append(score / min(len(actual_items), k))

#     return np.mean(aps) if len(aps) > 0 else 0.0


# def main():
#     """
#     Main execution function for H&M baseline recommendations.
#     """
#     print("="*80)
#     print("H&M Personalized Fashion Recommendations - Baseline")
#     print("="*80)

#     # 1. Load Data
#     print("\n1. Loading data...")
#     transactions = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv', parse_dates=['t_dat'])
#     customers = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv')
#     articles = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv')

#     print(f"Transactions shape: {transactions.shape}")
#     print(f"Customers shape: {customers.shape}")
#     print(f"Articles shape: {articles.shape}")

#     # 2. Memory Optimization
#     print("\n2. Optimizing memory usage...")
#     transactions = reduce_mem_usage(transactions)
#     customers = reduce_mem_usage(customers)
#     articles = reduce_mem_usage(articles)

#     # 3. Create Validation Split
#     print("\n3. Creating validation split...")
#     SPLIT_DATE = pd.to_datetime('2020-09-16')
#     VALIDATION_END = pd.to_datetime('2020-09-23')

#     train_df = transactions[transactions['t_dat'] < SPLIT_DATE].copy()
#     valid_df = transactions[
#         (transactions['t_dat'] >= SPLIT_DATE) &
#         (transactions['t_dat'] < VALIDATION_END)
#     ].copy()

#     print(f"Train size: {len(train_df):,} transactions")
#     print(f"Valid size: {len(valid_df):,} transactions")
#     print(f"Train date range: {train_df['t_dat'].min()} to {train_df['t_dat'].max()}")
#     print(f"Valid date range: {valid_df['t_dat'].min()} to {valid_df['t_dat'].max()}")

#     # 4. Calculate Popular Items
#     print("\n4. Calculating popular items...")
#     popular_items = calculate_popular_items(train_df, SPLIT_DATE, n=20)
#     print(f"Top 5 popular items: {popular_items[:5]}")

#     # 5. Generate Validation Recommendations
#     print("\n5. Generating validation recommendations...")
#     validation_customers = valid_df['customer_id'].unique()
#     print(f"Validation customers: {len(validation_customers)}")

#     validation_recommendations = generate_all_recommendations(
#         train_df, validation_customers, SPLIT_DATE, popular_items, n=12
#     )

#     # 6. Calculate MAP@12 on Validation Set
#     print("\n6. Calculating MAP@12 on validation set...")
#     validation_actual = defaultdict(list)
#     for _, row in valid_df.iterrows():
#         validation_actual[row['customer_id']].append(row['article_id'])

#     validation_map12 = calculate_map_at_k(validation_actual, validation_recommendations, k=12)
#     print(f"Validation MAP@12: {validation_map12:.6f}")

#     # 7. Generate Test Predictions
#     print("\n7. Generating test predictions...")

#     # For test set, use all training data up to max date
#     TEST_CUTOFF = transactions['t_dat'].max()

#     # Load or infer test customer IDs
#     # If sample_submission.csv exists, use those customer IDs
#     try:
#         sample_submission = pd.read_csv('./data/sample_submission.csv')
#         test_customers = sample_submission['customer_id'].unique()
#         print(f"Loaded {len(test_customers)} test customers from sample_submission.csv")
#     except:
#         # Otherwise, use all unique customers from transactions
#         test_customers = transactions['customer_id'].unique()
#         print(f"Using {len(test_customers)} unique customers from transactions")

#     # Recalculate popular items using all training data
#     popular_items_test = calculate_popular_items(transactions, TEST_CUTOFF, n=20)

#     test_recommendations = generate_all_recommendations(
#         transactions, test_customers, TEST_CUTOFF, popular_items_test, n=12
#     )

#     # 8. Create Submission File
#     print("\n8. Creating submission file...")
#     submission_df = create_submission(test_recommendations, output_file='submission.csv')

#     # 9. Summary
#     print("\n" + "="*80)
#     print("BASELINE COMPLETE")
#     print("="*80)
#     print(f"Validation MAP@12: {validation_map12:.6f}")
#     print(f"Submission created: submission.csv")
#     print(f"Total predictions: {len(submission_df)}")
#     print("="*80)

#     return validation_map12


# if __name__ == "__main__":
#     validation_score = main()



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Load and explore the customers dataset
import pandas as pd

# Read the customers CSV file
customers_df = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv')

# Display basic information
print("Customers Dataset Shape:", customers_df.shape)
print("\nFirst 5 rows:")
print(customers_df.head())
print("\nColumn names and types:")
print(customers_df.dtypes)
print("\nMissing values:")
print(customers_df.isnull().sum())





import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
import gc
import time
warnings.filterwarnings('ignore')



def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == object or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif str(col_type)[:5] == 'float':
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory: {start_mem:.1f}MB → {end_mem:.1f}MB')
    return df

def memory_cleanup():
    gc.collect()



def load_transactions_chunked(filepath, chunksize=500000, usecols=None, parse_dates=None):
    print(f"Loading {filepath}...")
    chunks = []
    for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunksize, usecols=usecols, parse_dates=parse_dates)):
        chunk = reduce_mem_usage(chunk)
        chunks.append(chunk)
        if (i+1) % 20 == 0:
            print(f"  {i+1} chunks loaded...")
    transactions = pd.concat(chunks, ignore_index=True)
    del chunks
    memory_cleanup()
    print(f"✓ Loaded {len(transactions):,} rows")
    return transactions



def compute_popular_items(df, cutoff_date, n=20):
    cutoff = cutoff_date - timedelta(days=7)
    recent = df[df['t_dat'] >= cutoff]
    items = recent['article_id'].value_counts().head(n).index.values
    del recent
    memory_cleanup()
    return items

def get_recommendations(customer_id, customer_index, cutoff_date, popular_items, n=12):
    if customer_id not in customer_index:
        return popular_items[:n].tolist()
    
    data = customer_index[customer_id]
    cutoff = cutoff_date - timedelta(days=30)
    mask = data['dates'] >= cutoff
    
    if not mask.any():
        return popular_items[:n].tolist()
    
    recent_articles = data['article_ids'][mask]
    recent_dates = data['dates'][mask]
    
    # Fixed datetime calculation
    cutoff_ts = pd.Timestamp(cutoff_date)
    days_ago = (cutoff_ts - pd.Series(recent_dates)).dt.days.values
    weights = 1.0 / (1.0 + days_ago)
    
    df = pd.DataFrame({'article_id': recent_articles, 'weight': weights})
    top = df.groupby('article_id')['weight'].sum().nlargest(n).index.values.tolist()
    
    # Fill with popular items
    while len(top) < n:
        for item in popular_items:
            if item not in top:
                top.append(item)
                if len(top) >= n:
                    break
        break
    
    return top[:n]



# Load data
transactions = load_transactions_chunked(
    '/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv',
    chunksize=500000,
    usecols=['customer_id', 'article_id', 't_dat'],
    parse_dates=['t_dat']
)

# Compute popular items
cutoff_date = transactions['t_dat'].max()
popular_items = compute_popular_items(transactions, cutoff_date)

# Build customer index
print("Building customer index...")
transactions = transactions.sort_values(['customer_id', 't_dat'])
customer_index = {}
for cust_id, group in transactions.groupby('customer_id'):
    customer_index[cust_id] = {
        'article_ids': group['article_id'].values,
        'dates': group['t_dat'].values
    }

# Load customer IDs
sample = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/sample_submission.csv')
customer_ids = sample['customer_id'].values

# Generate recommendations
recommendations = {}
start = time.time()
for i, cid in enumerate(customer_ids):
    recommendations[cid] = get_recommendations(cid, customer_index, cutoff_date, popular_items)
    if (i+1) % 10000 == 0:
        print(f"  {i+1:,} / {len(customer_ids):,}")

# Create submission
submission = pd.DataFrame({
    'customer_id': list(recommendations.keys()),
    'prediction': [' '.join([str(int(x)).zfill(10) for x in v]) for v in recommendations.values()]
})
submission.to_csv('submission.csv', index=False)
print(f"✓ Saved vedant_submission.csv")
print(submission.head())



submission = pd.DataFrame({
    'customer_id': list(recommendations.keys()),
    'prediction': [' '.join([str(int(x)).zfill(10) for x in v]) for v in recommendations.values()]
})
submission.to_csv('submission.csv', index=False)
print(f"✓ Saved submission.csv")
print(submission.head())













