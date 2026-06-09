import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
aisles = pd.read_csv("/kaggle/input/instacart-market-basket-analysis-nonzip-dataset/aisles.csv")
departments = pd.read_csv("/kaggle/input/instacart-market-basket-analysis-nonzip-dataset/departments.csv")
odpp = pd.read_csv("/kaggle/input/instacart-market-basket-analysis-nonzip-dataset/order_products__prior.csv")
odtr = pd.read_csv("/kaggle/input/instacart-market-basket-analysis-nonzip-dataset/order_products__train.csv")
products = pd.read_csv("/kaggle/input/instacart-market-basket-analysis-nonzip-dataset/products.csv")
orders = pd.read_csv("/kaggle/input/instacart-market-basket-analysis-nonzip-dataset/orders.csv")

# To ignore warinings
import warnings
warnings.filterwarnings('ignore')


# List of dataset names and corresponding DataFrames
file_names = ['aisles', 'departments', 'odpp', 'odtr', 'orders', 'products']
data = [aisles, departments, odpp, odtr, orders, products]

# Loop through and display information
for i in range(len(data)):
    print(f"Quick Summary of the Data: {file_names[i]}")
    print("-" * 50)
    print(data[i].info())
    print()
    print(f"Number of duplicate rows in {file_names[i]}: {data[i].duplicated().sum()}")
    missing_values = data[i].isnull().sum()
    total_missing = missing_values.sum()
    print(f"\nTotal missing values in {file_names[i]}: {total_missing}")
    print("-" * 50)
    print('\n')
    print("=" * 50)
    print('\n')


# Loop through each DataFrame to display missing values
for i in range(len(data)):
    print(f"Missing Values Summary for: {file_names[i]}")
    print("-" * 50)
    
    # Check for missing values in each column
    missing_values = data[i].isna().sum()
    print(missing_values)
    
    # Optionally, show the total number of missing values
    total_missing = missing_values.sum()
    print(f"\nTotal missing values in {file_names[i]}: {total_missing}")
    print("=" * 50)


# Dealing with missing values
# There are missing values in the 'days_since_prior_order' column in the 'orders' DataFrame
orders['days_since_prior_order'] = orders['days_since_prior_order'].fillna(0)
orders.isna().sum()


# Merge the DataFrames
# Merge the 'orders' and 'odpp' DataFrames
merge_orders_products = odpp.merge(products, on='product_id', how='left')
product_details = products.merge(aisles, on='aisle_id', how='left').merge(departments, on='department_id', how='left')
order_detailes = orders.merge(merge_orders_products, on='order_id', how='left')


# Statistical Summary

print(orders["days_since_prior_order"].describe().apply(lambda x: "%0.2f"%x))


# outlier detection
order_number_max = orders.groupby("user_id")["order_number"].max()
order_number_max_count = order_number_max.value_counts()

# Descriptive Statistics of number of orders done by users
order_number_max.describe().apply(lambda x: "%0.2f"%x)


sns.boxplot(data=order_number_max.values)


# Finding the upper whisker
IQR = order_number_max.quantile(0.75) - order_number_max.quantile(0.25)
upper_whisker = order_number_max.quantile(0.75) + (1.5 * (IQR))
print(f"upper_whisker = {upper_whisker}")


# Customer Rush Hours/Day of the Week
fig = plt.figure(figsize=(10, 15))
fig.suptitle("Bar plot for all columns of orders file, which have nunique <= 31")
k = 1

for i in ["order_hour_of_day", "order_dow", "days_since_prior_order"]:
    plt.subplot(3, 1, k)
    plt.title(f"Bar plot for {i} Vs Number of users")
    
    # Grouping and counting unique users
    data = orders.groupby(i)["user_id"].nunique()
    
    # Plot bar chart
    ax = sns.barplot(x=data.index, y=data.values)
    plt.ylabel("Number of unique users")
    plt.xticks(rotation=90)

    k += 1  # Move to the next subplot

plt.tight_layout()
plt.show()


data = orders.groupby(["order_dow","order_hour_of_day"])["user_id"].nunique().reset_index()
data = data.pivot(index = "order_dow",columns = "order_hour_of_day",values = "user_id")
fig = plt.figure(figsize=(10,5))
sns.heatmap(data = data,cmap = "crest")
plt.title("Number of unique users heat map between order day of week and order hour of day")
plt.xlabel("order hour of day")
plt.ylabel("order day of week")
plt.show()


# Order Frequency

fig = plt.figure(figsize = (15,15))
fig.suptitle("Count plot for all columns of orders file, which have nunique <= 31")
k = 1
for i in orders.columns:
    if orders[i].nunique() <= 31:
        plt.subplot(2,2,k)
        plt.title(f"Count plot for {i} column")
        sns.countplot(data = orders, x = i)
        plt.ylabel("Number of orders")
        plt.xticks(rotation = 90)
        k += 1
plt.tight_layout()
plt.show()


data = orders.groupby(["order_dow","order_hour_of_day"])["user_id"].agg("count").reset_index()
data = data.pivot(index = "order_dow",columns = "order_hour_of_day",values = "user_id")
fig = plt.figure(figsize=(10,5))
sns.heatmap(data = data,cmap = "crest")
plt.title("Number of unique users heat map between order day of week and order hour of day")
plt.xlabel("order hour of day")
plt.ylabel("order day of week")
plt.show()


# Reorder Frequency

#Filter for reordered products only
reordered_products = odpp[odpp['reordered'] == 1]

# Count the number of times each product was reordered
reordered_counts = reordered_products['product_id'].value_counts().head(10)

# Get product names by merging with the 'products' DataFrame
top_reordered_products = products[products['product_id'].isin(reordered_counts.index)]
top_reordered_products = top_reordered_products.merge(
    reordered_counts.rename('reorder_count'), 
    left_on='product_id', 
    right_index=True
)

# Sort the result by reorder_count
top_reordered_products = top_reordered_products.sort_values(by='reorder_count', ascending=False)
print(top_reordered_products[['product_id', 'product_name', 'reorder_count']])


# Plot the most reordered products

# Create figure and axis
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='reorder_count', y='product_name', data=top_reordered_products)

# Labels and Title
plt.xlabel('Number of Reorders')
plt.ylabel('Product Name')
plt.title('Top 10 Most Reordered Products')

# Add text on bars (bar values)
for p in ax.patches:
    ax.annotate(f'{int(p.get_width())}',  # Get bar width for horizontal bars
                (p.get_width(), p.get_y() + p.get_height() / 2),  # Position text at the end of the bar
                ha='left', va='center', fontsize=10, color='black', xytext=(5, 0), textcoords='offset points')

plt.show()


# Most Ordered Products
top_product_id = order_detailes['product_id'].value_counts(ascending=False).head(10)
top_product_names = products[products['product_id'].isin(top_product_id.index)]
topic_product_final = top_product_names.merge(top_product_id, on='product_id', how='left')
topic_product_final = topic_product_final.sort_values('count', ascending=False)
print(topic_product_final)


# Plot the most ordered products

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='count', y='product_name', data=topic_product_final)

# Labels and Title
plt.xlabel('Number of Orders')
plt.ylabel('Product Name')
plt.title('Top 10 Most Ordered Products')

plt.show()


# Most Popular Aisles
popular_aisle_counts = merge_orders_products['aisle_id'].value_counts().head(20)
popular_aisles = aisles[aisles['aisle_id'].isin(popular_aisle_counts.index)]
popular_aisles = popular_aisles.merge(popular_aisle_counts.rename('count'), on='aisle_id', how='left')
popular_aisles = popular_aisles.sort_values(by='count', ascending=False)
print(popular_aisles)


# Plot the most popular aisles
sns.barplot(x='count', y='aisle', data=popular_aisles)
plt.xlabel('Number of Orders')
plt.ylabel('Aisle Name')
plt.title('Top 10 Most Popular Aisles')
plt.show()



# Most Popular Departments
popular_department_counts = merge_orders_products['department_id'].value_counts().head(10)
popular_departments = departments[departments['department_id'].isin(popular_department_counts.index)]
popular_departments = popular_departments.merge(popular_department_counts.rename('count'), on='department_id', how='left')
popular_departments = popular_departments.sort_values(by='count', ascending=False)
print(popular_departments)


# Plot the most popular department
sns.barplot(x='count', y='department', data=popular_departments)
plt.xlabel('Number of Orders')
plt.ylabel('Department Name')
plt.title('Top 10 Most Popular Department')
plt.show()



# Total Products in Aisle

# Get total unique products per aisle, sorted in descending order
tot_prd_aisle = product_details.groupby("aisle")["product_id"].nunique().sort_values(ascending=False)

# Create figure and axis
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=tot_prd_aisle.values[:10], y=tot_prd_aisle.index[:10])

# Labels and Title
plt.xlabel("Total Number of Products in Aisle")
plt.ylabel("Aisle")
plt.title("Total Products in Aisle Vs Aisle")


plt.show()


# Total Products in Department
tot_prd_department = product_details.groupby("department")["product_id"].nunique().sort_values(ascending=False)

plt.figure(figsize=(10,6))
ax = sns.barplot(x=tot_prd_department.values, y=tot_prd_department.index)

plt.ylabel("Department")
plt.xlabel("Total number of products in department")
plt.title("Total Products in Department Vs Department")

plt.show()


# Total Aisles in Department
tot_aisle_department = product_details.groupby("department")["aisle"].nunique().sort_values(ascending=False)

plt.figure(figsize=(10,6))
ax = sns.barplot(x=tot_aisle_department.values, y=tot_aisle_department.index)

plt.ylabel("Department")
plt.xlabel("Total number of aisles in department")
plt.title("Total Aisles in Department Vs Department")

plt.show()


order_number_max = orders.groupby("user_id")["order_number"].max()
order_number_max_count = order_number_max.value_counts()
order_number_max


tot_reod_pur = order_detailes.groupby("product_name")["reordered"].agg(['count', 'sum']).rename(columns = {'count':'total','sum':'reorders'})
tot_reod_pur = tot_reod_pur.sort_values('total', ascending=False).reset_index()
tot_reod_pur["product_reorder_percentage"] = tot_reod_pur["reorders"]*100/tot_reod_pur["total"]
tot_reod_pur


fig = plt.figure(figsize=(20,5))
sns.scatterplot(x = 'total',y = 'product_reorder_percentage',data = tot_reod_pur)
plt.xlabel("total product purchases")
plt.ylabel("Product reorder percentage")
plt.title("Product reorder percentage with respect to total product purchases")
plt.xticks(rotation = 90)
for i in range(10):
    plt.annotate(tot_reod_pur["product_name"][i],(tot_reod_pur["total"][i],tot_reod_pur["product_reorder_percentage"][i]+0.5),size = 10)
plt.show()


import pandas as pd
import numpy as np
import sys
from itertools import combinations, groupby
from collections import Counter
from IPython.display import display


# Convert from DataFrame to a Series, with order_id as index and item_id as value
odpp = odpp.set_index('order_id')['product_id'].rename('item_id')
display(odpp.head(10))
type(odpp)


# Returns frequency counts for items and item pairs
def freq(iterable):
    if type(iterable) == pd.core.series.Series:
        return iterable.value_counts().rename("freq")
    else: 
        return pd.Series(Counter(iterable)).rename("freq")

    
# Returns number of unique orders
def order_count(order_item):
    return len(set(order_item.index))


# Returns generator that yields item pairs, one at a time
def get_item_pairs(order_item):
    # Convert DataFrame to NumPy array (fixed)
    order_item = order_item.reset_index().values  
    
    # Group by order_id (first column)
    for order_id, order_object in groupby(order_item, key=lambda x: x[0]):
        item_list = [item[1] for item in order_object]  # Extract items
        
        # Generate all possible item pairs
        for item_pair in combinations(item_list, 2):
            yield item_pair
            

# Returns frequency and support associated with item
def merge_item_stats(item_pairs, item_stats):
    return (item_pairs
                .merge(item_stats.rename(columns={'freq': 'freqA', 'support': 'supportA'}), left_on='item_A', right_index=True)
                .merge(item_stats.rename(columns={'freq': 'freqB', 'support': 'supportB'}), left_on='item_B', right_index=True))


# Returns name associated with item
def merge_item_name(rules, item_name):
    columns = ['itemA','itemB','freqAB','supportAB','freqA','supportA','freqB','supportB', 
               'confidenceAtoB','confidenceBtoA','lift']
    rules = (rules
                .merge(item_name.rename(columns={'item_name': 'itemA'}), left_on='item_A', right_on='item_id')
                .merge(item_name.rename(columns={'item_name': 'itemB'}), left_on='item_B', right_on='item_id'))
    return rules[columns]               


def association_rules(order_item, min_support):

    print("Starting order_item: {:22d}".format(len(order_item)))


    # Calculate item frequency and support
    item_stats             = freq(order_item).to_frame("freq")
    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100


    # Filter from order_item items below min support 
    qualifying_items       = item_stats[item_stats['support'] >= min_support].index
    order_item             = order_item[order_item.isin(qualifying_items)]

    print("Items with support >= {}: {:15d}".format(min_support, len(qualifying_items)))
    print("Remaining order_item: {:21d}".format(len(order_item)))


    # Filter from order_item orders with less than 2 items
    order_size             = freq(order_item.index)
    qualifying_orders      = order_size[order_size >= 2].index
    order_item             = order_item[order_item.index.isin(qualifying_orders)]

    print("Remaining orders with 2+ items: {:11d}".format(len(qualifying_orders)))
    print("Remaining order_item: {:21d}".format(len(order_item)))


    # Recalculate item frequency and support
    item_stats             = freq(order_item).to_frame("freq")
    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100


    # Get item pairs generator
    item_pair_gen          = get_item_pairs(order_item)


    # Calculate item pair frequency and support
    item_pairs              = freq(item_pair_gen).to_frame("freqAB")
    item_pairs['supportAB'] = item_pairs['freqAB'] / len(qualifying_orders) * 100

    print("Item pairs: {:31d}".format(len(item_pairs)))


    # Filter from item_pairs those below min support
    item_pairs              = item_pairs[item_pairs['supportAB'] >= min_support]

    print("Item pairs with support >= {}: {:10d}\n".format(min_support, len(item_pairs)))


    # Create table of association rules and compute relevant metrics
    item_pairs = item_pairs.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
    item_pairs = merge_item_stats(item_pairs, item_stats)
    
    item_pairs['confidenceAtoB'] = item_pairs['supportAB'] / item_pairs['supportA']
    item_pairs['confidenceBtoA'] = item_pairs['supportAB'] / item_pairs['supportB']
    item_pairs['lift']           = item_pairs['supportAB'] / (item_pairs['supportA'] * item_pairs['supportB'])
    
    
    # Return association rules sorted by lift in descending order
    return item_pairs.sort_values('lift', ascending=False)


%%time
rules = association_rules(odpp, 0.01)  


# Replace item ID with item name and display association rules
item_name   = pd.read_csv('/kaggle/input/instacart-market-basket-analysis-nonzip-dataset/products.csv')
item_name   = item_name.rename(columns={'product_id':'item_id', 'product_name':'item_name'})
rules_final = merge_item_name(rules, item_name).sort_values('lift', ascending=False)
display(rules_final)

