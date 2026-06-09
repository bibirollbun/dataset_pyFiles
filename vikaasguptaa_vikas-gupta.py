import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm



articles = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/articles.csv")
customers = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/customers.csv")
transactions = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv")



articles.head()


print("Articles:", articles.shape)
print("Customers:", customers.shape)
print("Transactions:", transactions.shape)


# Check for missing values
print("Missing Values:\n", transactions.isnull().sum())


# Display first few rows
print(articles.head())
print(customers.head())
print(transactions.head())

# Identify NaN or infinite values
print("Articles NaN values:\n", articles.isna().sum())
print("Customers NaN values:\n", customers.isna().sum())
print("Transactions NaN values:\n", transactions.isna().sum())

# Handle NaN values (e.g., fill NaNs with 0 or drop rows)
articles = articles.fillna(0)
customers = customers.fillna(0)
transactions = transactions.fillna(0)

# Remove infinite values (if any)
articles = articles.replace([np.inf, -np.inf], np.nan).dropna()
customers = customers.replace([np.inf, -np.inf], np.nan).dropna()
transactions = transactions.replace([np.inf, -np.inf], np.nan).dropna()

# Display first few rows again after cleaning
print(articles.head())
print(customers.head())
print(transactions.head())



transactions['t_dat'] = pd.to_datetime(transactions['t_dat']) # Convert date
customers.fillna(0, inplace=True) # Fill missing values


import matplotlib.pyplot as plt

# Extract month-year from date
transactions['month'] = transactions['t_dat'].dt.to_period('M')

# Aggregate sales by month
monthly_sales = transactions.groupby('month').size()

# Plot sales trend
plt.figure(figsize=(12,5))
monthly_sales.plot(kind='line', marker='o', color='blue')
plt.xlabel("Month")
plt.ylabel("Number of Purchases")
plt.title("Purchase Trend Over Time")
plt.grid(True)
plt.show()


# import seaborn as sns
# import matplotlib.pyplot as plt

# # Check and print column names
# print(customers.columns)

# # Verify and strip column names
# customers.columns = customers.columns.str.strip().str.lower()

# # Plot for 'active' column
# if 'active' in customers.columns:
#     sns.countplot(x=customers['active'])
#     plt.title("Active Customers Distribution")
#     plt.show()
# else:
#     print("Column 'active' does not exist in the DataFrame.")

# # Plot for 'FN' column
# if 'fn' in customers.columns:
#     sns.countplot(x=customers['fn'])
#     plt.title("Fashion News Subscription")
#     plt.show()
# else:
#     print("Column 'FN' does not exist in the DataFrame.")



import matplotlib.pyplot as plt

# Analyze sales_channel_id (1 = Store, 2 = Online)
sales_channel = transactions['sales_channel_id'].value_counts()
sales_channel.plot(kind='pie', autopct='%1.1f%%', title="Online vs Store Sales")
plt.show()



# Summary of numerical columns
print("\nSummary Statistics:\n")
print(transactions.describe())

# Unique values in categorical columns
print("\nUnique Categories in 'Articles':\n", articles.nunique())
print("\nUnique Categories in 'Customers':\n", customers.nunique())

# Distribution of sales channels
plt.figure(figsize=(6,4))
sns.countplot(x="sales_channel_id", data=transactions, palette="coolwarm")
plt.title("Sales Channel Distribution (1 = Store, 2 = Online)")
plt.xlabel("Sales Channel")
plt.ylabel("Count")
plt.show()


# Check column names
print(customers.columns)

# Age distribution of customers
plt.figure(figsize=(8,5))
sns.histplot(customers["age"].dropna(), bins=30, kde=True, color="blue")
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Impact of newsletter subscription (fn) and Active status
# Replace 'fn' with the correct column name if it's different
plt.figure(figsize=(10,5))
sns.countplot(x="fn", data=customers, palette="Set1")
plt.title("Customers Receiving Fashion News")
plt.xlabel("Fashion News Subscription (fn)")
plt.ylabel("Count")
plt.show()

# Replace 'active' with the correct column name if it's different
plt.figure(figsize=(10,5))
sns.countplot(x="active", data=customers, palette="Set2")
plt.title("Active Customers")
plt.xlabel("Active Status")
plt.ylabel("Count")
plt.show()



# Identify top-selling product categories and colors.

# Most purchased products
top_products = transactions["article_id"].value_counts().head(10).sort_values(ascending=False)
top_products.plot(kind='bar', title='Top 10 Most Purchased Products')
plt.xlabel("Article ID")
plt.ylabel("Number of Purchases")
plt.xticks(rotation=90)
plt.show()

# Top garment groups
top_garments = articles["product_group_name"].value_counts().head(10).sort_values(ascending=False)
top_garments.plot(kind='bar', title='Top 10 Product Groups')
plt.xlabel("Product Group")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()



#Temporal Analysis
#Analyze purchase trends over time.

# Convert date column to datetime format
transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])

# Daily sales trends
daily_sales = transactions.groupby("t_dat").size()

plt.figure(figsize=(12,5))
sns.lineplot(x=daily_sales.index, y=daily_sales.values, color="green")
plt.title("Daily Purchase Trends")
plt.xlabel("Date")
plt.ylabel("Number of Purchases")
plt.show()


import pandas as pd
import datetime as dt

# Load dataset
transactions = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv")

# Display first few rows
print(transactions.head())



import pandas as pd
import datetime as dt

# Load dataset
transactions = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv")

# Display column names
print(transactions.columns)



import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
articles = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/articles.csv")
customers = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/customers.csv")
transactions = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv")

# Convert 't_dat' to datetime (assuming 't_dat' is the correct column name for transaction date)
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])

# Handle missing values if any
transactions.dropna(inplace=True)

# Set reference date (e.g., one day after the last transaction date)
reference_date = transactions['t_dat'].max() + dt.timedelta(days=1)

# Calculate Recency, Frequency, and Monetary values
rfm = transactions.groupby('customer_id').agg({
    't_dat': lambda x: (reference_date - x.max()).days,  # Recency
    'customer_id': 'count',                             # Frequency
    'price': 'sum'                                      # Monetary
}).rename(columns={'t_dat': 'Recency', 'customer_id': 'Frequency', 'price': 'Monetary'})

# Define RFM score function
def rfm_score(x, quantiles, metric):
    if x <= quantiles[metric][0.25]:
        return 1
    elif x <= quantiles[metric][0.50]:
        return 2
    elif x <= quantiles[metric][0.75]:
        return 3
    else:
        return 4

# Calculate quantiles
quantiles = rfm.quantile(q=[0.25, 0.50, 0.75]).to_dict()

# Assign RFM scores
rfm['R_Score'] = rfm['Recency'].apply(rfm_score, args=(quantiles, 'Recency'))
rfm['F_Score'] = rfm['Frequency'].apply(rfm_score, args=(quantiles, 'Frequency'))
rfm['M_Score'] = rfm['Monetary'].apply(rfm_score, args=(quantiles, 'Monetary'))

# Combine RFM scores into a single score
rfm['RFM_Score'] = rfm['R_Score'].map(str) + rfm['F_Score'].map(str) + rfm['M_Score'].map(str)

# Define customer segments
rfm['Segment'] = 'Other'
rfm.loc[rfm['RFM_Score'].str.startswith('1'), 'Segment'] = 'Best Customers'
rfm.loc[rfm['RFM_Score'].str.startswith('4'), 'Segment'] = 'Lost Customers'

# Display RFM scores and segments
print(rfm.head())

# Calculate average RFM values per segment
segment_analysis = rfm.groupby('Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']
}).round(1)

# Display segment analysis
print(segment_analysis)

# Plot segment distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=rfm, x='Segment', order=rfm['Segment'].value_counts().index, palette='viridis')
plt.title('Customer Segments Distribution')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.show()



import pandas as pd
import datetime as dt

# Load Data
transactions = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv")

# Convert date column to datetime
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])

# Data Preparation
now = dt.datetime(2022, 1, 1)  # assuming the current date for analysis
rfm = transactions.groupby('customer_id').agg({
    't_dat': lambda x: (now - x.max()).days,  # Recency
    'customer_id': 'count',  # Frequency
    'price': 'sum'  # Monetary
}).rename(columns={'t_dat': 'Recency', 'customer_id': 'Frequency', 'price': 'Monetary'})

# Normalize Scores
rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'], 5, labels=[1, 2, 3, 4, 5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

# Calculate RFM Score
rfm['RFM_Score'] = rfm.R_Score.astype(str) + rfm.F_Score.astype(str) + rfm.M_Score.astype(str)

# Segment customers based on RFM score
rfm['Segment'] = rfm.apply(lambda row: 'Champion' if row['RFM_Score'] == '555' else 'Others', axis=1)

# Show the segmented customers
print(rfm.head())



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample Data for Visualization (replace with your RFM DataFrame)
# Assuming 'rfm' DataFrame contains columns: 'Recency', 'Frequency', 'Monetary'
# rfm = pd.DataFrame({ ... })

# Create customer segments
def segment_customer(row):
    if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
        return 'High Value'
    elif row['R_Score'] >= 2 and row['F_Score'] >= 2 and row['M_Score'] >= 2:
        return 'Medium Value'
    else:
        return 'Low Value'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

# Bar Chart of Customer Segments
rfm['Segment'].value_counts().plot(kind='bar')
plt.title('Customer Segments Distribution')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.show()

# Heatmap of R, F, M Scores
rfm_corr = rfm[['R_Score', 'F_Score', 'M_Score']].astype(int).corr()
sns.heatmap(rfm_corr, annot=True, cmap='coolwarm')
plt.title('Correlation of R, F, M Scores')
plt.show()

# Scatter Plot of Recency vs Frequency
sns.scatterplot(x='Recency', y='Frequency', hue='Segment', data=rfm)
plt.title('Recency vs Frequency by Customer Segment')
plt.xlabel('Recency (Days)')
plt.ylabel('Frequency (Purchases)')
plt.legend()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset files
articles = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/articles.csv")
customers = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/customers.csv")
transactions = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv")

# Check the column names and inspect the data
print(articles.columns)
print(customers.columns)
print(transactions.columns)

# Ensure 'date' is in datetime format
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])

# Calculate Recency: Number of days since the most recent purchase for each customer
most_recent_date = transactions['t_dat'].max()

# Calculate Recency for each customer
transactions['recency'] = (most_recent_date - transactions.groupby('customer_id')['t_dat'].transform('max')).dt.days

# Calculate Monetary: Total spending for each customer (using item count as proxy)
monetary_per_customer = transactions.groupby('customer_id')['article_id'].count()

# Calculate Frequency: Number of transactions per customer
frequency_per_customer = transactions.groupby('customer_id').size()

# Combine Recency, Monetary, and Frequency into a DataFrame (RFM model)
rfm = pd.DataFrame({
    'Recency': transactions.groupby('customer_id')['recency'].max(),
    'Monetary': monetary_per_customer,
    'Frequency': frequency_per_customer
})

# 1. Average Visit per Customer (Average Frequency)
average_visit = rfm['Frequency'].mean()

# 2. Average Sales per Customer (Average Monetary)
average_sales = rfm['Monetary'].mean()

# 3. Average Recency
average_recency = rfm['Recency'].mean()

# Print the results
print(f"Average Visit per Customer (Frequency): {average_visit}")
print(f"Average Sales per Customer (Monetary): {average_sales}")
print(f"Average Recency (Days since last purchase): {average_recency}")

# Visualize Recency Distribution
plt.figure(figsize=(8, 6))
sns.histplot(rfm['Recency'], bins=50, kde=True, color='skyblue')
plt.title('Recency Distribution (Days since Last Purchase)')
plt.xlabel('Days since Last Purchase')
plt.ylabel('Frequency')
plt.show()

# Visualize Frequency Distribution (Average Visits per Customer)
plt.figure(figsize=(8, 6))
sns.histplot(rfm['Frequency'], bins=50, kde=True, color='lightgreen')
plt.title('Purchase Frequency Distribution (Number of Purchases per Customer)')
plt.xlabel('Number of Purchases')
plt.ylabel('Frequency')
plt.show()

# Visualize Monetary Distribution (Total Spending per Customer)
plt.figure(figsize=(8, 6))
sns.histplot(rfm['Monetary'], bins=50, kde=True, color='lightcoral')
plt.title('Monetary Distribution (Total Spending per Customer)')
plt.xlabel('Total Spending (Item Count)')
plt.ylabel('Frequency')
plt.show()

# Optional: RFM Segmentation
def segment_customer(row):
    if row['Recency'] <= 30 and row['Frequency'] >= 5 and row['Monetary'] >= 100:
        return 'High Value'
    elif row['Recency'] <= 60 and row['Frequency'] >= 3 and row['Monetary'] >= 50:
        return 'Medium Value'
    else:
        return 'Low Value'

# Apply segmentation
rfm['Segment'] = rfm.apply(segment_customer, axis=1)

# Visualize Customer Segments
plt.figure(figsize=(8, 6))
rfm['Segment'].value_counts().plot(kind='bar', color='lightblue')
plt.title('Customer Segments Distribution')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.show()


