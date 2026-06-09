# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


transections = "/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv"
articals = "/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv"
sample_submission = "/kaggle/input/h-and-m-personalized-fashion-recommendations/sample_submission.csv"
customers = "/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv"

transactions_df = pd.read_csv(transections)
articles_df = pd.read_csv(articals)
customers_df = pd.read_csv(customers)


print(transactions_df.head())
print(articles_df.head())
print(customers_df.head())


# Convert `t_dat` to a datetime type for easy date manipulation
transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'])

# Get a high-level overview of the data
print("Transactions DataFrame Info:")
print(transactions_df.info())

# Count the number of unique customers and articles
num_unique_customers = transactions_df['customer_id'].nunique()
num_unique_articles = transactions_df['article_id'].nunique()

# Find the date range of the transactions
start_date = transactions_df['t_dat'].min()
end_date = transactions_df['t_dat'].max()

# Print the key statistics
print("\nBasic Statistics")
print(f"Number of unique customers: {num_unique_customers}")
print(f"Number of unique articles: {num_unique_articles}")
print(f"Date range of transactions: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")


print("--- Customers Dataframe EDA ---")
print("\nMissing values in customers data:")
print(customers_df.isnull().sum())
print("\nDistribution of club_member_status:")
print(customers_df['club_member_status'].value_counts())
print("\nDistribution of fashion_news_frequency:")
print(customers_df['fashion_news_frequency'].value_counts())


print("\n--- Articles Dataframe EDA ---")
print("\nMissing values in articles data:")
print(articles_df.isnull().sum())
print("\nDistribution of product_group_name:")
print(articles_df['product_group_name'].value_counts())
print("\nDistribution of garment_group_name:")
print(articles_df['garment_group_name'].value_counts())


# Convert `t_dat` to datetime
transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'])

# Filter transactions for the last 3 months
end_date = transactions_df['t_dat'].max()
start_date_filtered = end_date - pd.DateOffset(months=3)
recent_transactions = transactions_df[transactions_df['t_dat'] >= start_date_filtered]

# Merge the dataframes
merged_df = pd.merge(recent_transactions, customers_df, on='customer_id', how='left')
merged_df = pd.merge(merged_df, articles_df, on='article_id', how='left')

# Display the information of the new, merged dataframe
print("Merged DataFrame Info (last 3 months):")
print(merged_df.info())

print("\nMerged DataFrame Head:")
merged_df.head()


# Let's fill null with the median age of the customers
median_age = merged_df['age'].median()
merged_df['age'].fillna(median_age, inplace=True)

# Impute missing categorical values with a placeholder
merged_df['club_member_status'] = merged_df['club_member_status'].fillna('Unknown')
merged_df['fashion_news_frequency'] = merged_df['fashion_news_frequency'].fillna('Unknown')
merged_df['FN'] = merged_df['FN'].fillna(0)
merged_df['Active'] = merged_df['Active'].fillna(0)

print(merged_df[['club_member_status', 'fashion_news_frequency', 'FN', 'Active']].isna().sum())

# Create temporal features
merged_df['week'] = merged_df['t_dat'].dt.isocalendar().week.astype(int)
merged_df['day_of_week'] = merged_df['t_dat'].dt.dayofweek.astype(int)

merged_df[['t_dat', 'age', 'FN', 'Active', 'club_member_status', 'week', 'day_of_week']].head()



# calculate recency, frequency, and monetary value for each customer
customer_features = merged_df.groupby('customer_id').agg(
    total_purchase = ('article_id', 'count'),
    last_purchase_date = ('t_dat', 'max')
)
customer_features["recency_days"] = (merged_df['t_dat'].max() - customer_features['last_purchase_date']).dt.days
customer_features.head()


# calculate popularity and average price for each article
artical_features = merged_df.groupby('article_id').agg(
    purchase_count = ('customer_id', 'count'),
    average_price = ('price', 'mean')
)
artical_features.head()


# To make the process faster, let's work with a small sample of the merged data
sample_merged_df = merged_df.sample(n=100000, random_state=42).reset_index(drop=True)


# Create positive samples with a label of 1
positive_samples = sample_merged_df[['customer_id', 'article_id']].copy()
positive_samples['label'] = 1


# Negative Sampling: Get all unique article IDs
all_article_ids = sample_merged_df['article_id'].unique()


# generate negetive sample list
negative_samples_list = []
for customer in positive_samples['customer_id'].unique():
    customer_purchases = set(positive_samples[positive_samples['customer_id'] == customer]['article_id'])
    
    # Get articles not purchased by the customer
    non_purchased_articles = np.setdiff1d(all_article_ids, list(customer_purchases))
    
    # Randomly sample a few non-purchased items for each purchase
    num_neg_samples = min(len(non_purchased_articles), 4) # Take 4 negative samples for each positive
    if num_neg_samples > 0:
        neg_articles = np.random.choice(non_purchased_articles, num_neg_samples, replace=False)
        for neg_article in neg_articles:
            negative_samples_list.append([customer, neg_article, 0])

negative_samples = pd.DataFrame(negative_samples_list, columns=['customer_id', 'article_id', 'label'])

negative_samples.head()


# Now combine positive and negative samples
final_data = pd.concat([positive_samples, negative_samples], ignore_index=True)

# Merge our aggregated features
final_data = pd.merge(final_data, customer_features, on='customer_id', how='left')
final_data = pd.merge(final_data, artical_features, on='article_id', how='left')

# Show the final dataset structure
print("\nFinal Dataset Shape:")
print(final_data.shape)
print("\nDistribution of Labels:")
print(final_data['label'].value_counts())

print("Final Dataset for Modeling Head:")
final_data.head()









import lightgbm as lgb
from sklearn.model_selection import train_test_split


# Define features and target
features = ['total_purchase', 'recency_days', 'purchase_count', 'average_price']
target = 'label'

# Drop any NaN values that might have been created during merging the datasets
final_data.dropna(subset=features, inplace=True)

X = final_data[features]
y = final_data[target]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the LightGBM model
lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(X_train, y_train)

# Print a confirmation message
print("LightGBM model training complete!")
print("Model accuracy on validation set:", lgb_model.score(X_val, y_val))




