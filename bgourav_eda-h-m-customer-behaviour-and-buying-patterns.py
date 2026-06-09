import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8-whitegrid")


try:
    transactions_df = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')
    articles_df = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv')
    customers_df = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv')
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print("Error: One or more files not found. Please check the file paths: ", e)


print("\n--- Initial Data Inspection (transactions_df) ---")
print("Shape of the dataframe:", transactions_df.shape)
print("\nFirst 5 rows:")
print(transactions_df.head())
print("\nColumn information:")
transactions_df.info()

print("\n--- Initial Data Inspection (customers_df) ---")
print("Shape of the dataframe:", customers_df.shape)
print("\nFirst 5 rows:")
print(customers_df.head())
print("\nColumn information:")
customers_df.info()

print("\n--- Initial Data Inspection (articles_df) ---")
print("Shape of the dataframe:", articles_df.shape)
print("\nFirst 5 rows:")
print(articles_df.head())
print("\nColumn information:")
articles_df.info()


transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'])

print("\n--- Missing Value Check ---")
print("Missing values in transactions_df:\n", transactions_df.isnull().sum())
print("\nMissing values in customers_df:\n", customers_df.isnull().sum())
print("\nMissing values in articles_df:\n", articles_df.isnull().sum())

customers_df['club_member_status'].fillna('UNKNOWN', inplace=True)
customers_df['fashion_news_frequency'].fillna('None', inplace=True)

customers_df['age'].fillna(customers_df['age'].median(), inplace=True)


print("\n--- Merging Dataframes ---")
merged_df = pd.merge(transactions_df, customers_df, on='customer_id', how='left')
merged_df = pd.merge(merged_df, articles_df, on='article_id', how='left')

print("Shape of the merged dataframe:", merged_df.shape)
print("First 5 rows of the merged dataframe:")
print(merged_df.head())
print("Column information of the merged dataframe:")
merged_df.info()

# Create new features
merged_df['sales_month'] = merged_df['t_dat'].dt.to_period('M')
merged_df['sales_dayofweek'] = merged_df['t_dat'].dt.day_name()


print("\n--- EDA on Customer Behavior & Buying Patterns ---")

print("\n--- Overall Sales Trend ---")
sales_by_date = merged_df.groupby('t_dat')['price'].sum()
plt.figure(figsize=(15, 6))
sales_by_date.plot(title='Total Sales Over Time', color='skyblue')
plt.xlabel('Date')
plt.ylabel('Total Sales (in millions)')
plt.show()


print("\n--- Top 10 Bestselling Products ---")
top_articles = merged_df['article_id'].value_counts().head(10)
top_articles_info = pd.merge(top_articles, articles_df[['article_id', 'prod_name']], on='article_id')
print(top_articles_info)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_articles_info['count'], y=top_articles_info['prod_name'], palette='viridis')
plt.title('Top 10 Bestselling Products')
plt.xlabel('Number of Transactions')
plt.ylabel('Product Name')
plt.show()


print("\n--- Top 10 Bestselling Product Groups ---")
top_prod_groups = merged_df['product_group_name'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_prod_groups.values, y=top_prod_groups.index, palette='magma')
plt.title('Top 10 Bestselling Product Groups')
plt.xlabel('Number of Transactions')
plt.ylabel('Product Group')
plt.show()


print("\n--- Customer Segmentation: RFM Analysis ---")


latest_date = transactions_df['t_dat'].max()
rfm_df = transactions_df.groupby('customer_id').agg(
    Recency=('t_dat', lambda x: (latest_date - x.max()).days),
    Frequency=('t_dat', 'count'),
    Monetary=('price', 'sum')
)
print("RFM DataFrame Head:")
print(rfm_df.head())


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(rfm_df['Recency'], bins=50, kde=True, ax=axes[0])
axes[0].set_title('Recency Distribution')
sns.histplot(rfm_df['Frequency'], bins=50, kde=True, ax=axes[1])
axes[1].set_title('Frequency Distribution')
sns.histplot(rfm_df['Monetary'], bins=50, kde=True, ax=axes[2])
axes[2].set_title('Monetary Distribution')
plt.tight_layout()
plt.show()


print("\n--- Age and Spending Behavior ---")
plt.figure(figsize=(10, 6))
sns.histplot(customers_df['age'], bins=50, kde=True, color='purple')
plt.title('Distribution of Customer Ages')
plt.xlabel('Age')
plt.ylabel('Number of Customers')
plt.show()


bins = [0, 18, 25, 35, 45, 55, 65, 100]
labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
merged_df['age_group'] = pd.cut(merged_df['age'], bins=bins, labels=labels, right=False)

avg_spending_by_age = merged_df.groupby('age_group')['price'].mean().sort_index()
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_spending_by_age.index, y=avg_spending_by_age.values, palette='coolwarm')
plt.title('Average Transaction Price by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Price')
plt.show()


print("\n--- Seasonality and Day of the Week Analysis ---")
# Sales by month
sales_by_month = merged_df.groupby('sales_month')['price'].sum()
plt.figure(figsize=(15, 6))
sales_by_month.plot(kind='bar', title='Total Sales by Month', color='teal')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()


day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sales_by_day = merged_df.groupby('sales_dayofweek')['price'].sum().reindex(day_order)
plt.figure(figsize=(10, 6))
sns.barplot(x=sales_by_day.index, y=sales_by_day.values, palette='pastel')
plt.title('Total Sales by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Total Sales')
plt.show()


print("\n--- Impact of Fashion News & Club Membership ---")
spending_by_news = merged_df.groupby('fashion_news_frequency')['price'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=spending_by_news.index, y=spending_by_news.values, palette='dark')
plt.title('Average Transaction Price by Fashion News Frequency')
plt.xlabel('Fashion News Frequency')
plt.ylabel('Average Price')
plt.show()


spending_by_club = merged_df.groupby('club_member_status')['price'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=spending_by_club.index, y=spending_by_club.values, palette='cubehelix')
plt.title('Average Transaction Price by Club Member Status')
plt.xlabel('Club Member Status')
plt.ylabel('Average Price')
plt.show()


print("\n--- Summary of Key Findings ---")
print("Based on the EDA, we can draw the following insights:")
print("- The total sales show a clear trend over time, likely with seasonal peaks.")
print("- Bestselling products and product groups can be identified, guiding inventory and marketing efforts.")
print("- RFM analysis helps segment customers into different tiers (e.g., high-value, recent, frequent).")
print("- Customer age has a visible impact on average spending, which can inform targeted advertising.")
print("- Certain days of the week or months show higher sales, useful for staffing and promotion planning.")
print("- Fashion news subscription and club membership appear to be correlated with customer spending, suggesting these are effective engagement tools.")

