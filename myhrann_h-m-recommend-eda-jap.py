import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

import warnings
warnings.simplefilter('ignore')


articles = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/articles.csv")
customers = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/customers.csv")
transactions = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv")
sample_submission = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv")


print("\n--- 1.1. articles ã�®æ¦‚è¦� ---")
print(articles.info())
print("\n--- articles ã�®æœ€åˆ�ã�®5è¡Œ ---")
display(articles.head())
print("\n--- articles ã�®æ¬ æ��å€¤ ---")
print(articles.isnull().sum())
print("\n--- articles ã�®ãƒ¦ãƒ‹ãƒ¼ã‚¯ã�ªã‚«ãƒ†ã‚´ãƒªã‚«ãƒ«ã�ªå€¤ ---")
for col in ['product_type_name', 'product_group_name', 'graphical_appearance_name',
            'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name',
            'department_name', 'index_name', 'index_group_name', 'section_name', 'garment_group_name']:
    print(f"{col}: {articles[col].nunique()} ãƒ¦ãƒ‹ãƒ¼ã‚¯ã�ªå€¤")

print("\n--- 1.2. customers ã�®æ¦‚è¦� ---")
print(customers.info())
print("\n--- customers ã�®æœ€åˆ�ã�®5è¡Œ ---")
display(customers.head())
print("\n--- customers ã�®æ¬ æ��å€¤ ---")
print(customers.isnull().sum())
print("\n--- customers ã�®ãƒ¦ãƒ‹ãƒ¼ã‚¯ã�ªã‚«ãƒ†ã‚´ãƒªã‚«ãƒ«ã�ªå€¤ ---")
for col in ['FN', 'Active', 'club_member_status', 'fashion_news_frequency']:
    print(f"{col}: {customers[col].nunique()} ãƒ¦ãƒ‹ãƒ¼ã‚¯ã�ªå€¤")

print("\n--- 1.3. transactions ã�®æ¦‚è¦� ---")
print(transactions.info())
print("\n--- transactions ã�®æœ€åˆ�ã�®5è¡Œ ---")
display(transactions.head())
print("\n--- transactions ã�®æ¬ æ��å€¤ ---")
print(transactions.isnull().sum())
print("\n--- transactions ã�®æœŸé–“ ---")
print(f"æœ€å°�å�–å¼•æ—¥: {transactions['t_dat'].min()}")
print(f"æœ€å¤§å�–å¼•æ—¥: {transactions['t_dat'].max()}")


# print("\n--- 2.1. customers ã�®æ¬ æ��å€¤ ('FN', 'Active', 'age', 'fashion_news_frequency') ã�®åˆ†æ�� ---")
# 'FN'ã�¨'Active'ã�®æ¬ æ��å€¤ã�¯NaNã�§ã�‚ã‚Šã€�0.0ã�¨ã�—ã�¦æ‰±ã�ˆã‚‹ã�‹æ¤œè¨�
# customers_df['FN'] = customers_df['FN'].fillna(0).astype(int)
# customers_df['Active'] = customers_df['Active'].fillna(0).astype(int)
# 'fashion_news_frequency'ã�®æ¬ æ��å€¤ã�¯ä¸�æ˜�ã�¨ã�—ã�¦æ‰±ã�†ã�‹ã€�æœ€é »å€¤ã�ªã�©ã�§è£œå®Œã�™ã‚‹ã�‹æ¤œè¨�
# 'age'ã�®æ¬ æ��å€¤ã�¯å°‘ã�ªã�„ã�®ã�§ã€�å¹³å�‡å€¤ã‚„ä¸­å¤®å€¤ã�§è£œå®Œã�™ã‚‹ã�‹ã€�æ¬ æ��ãƒ¦ãƒ¼ã‚¶ãƒ¼ã‚’å‰Šé™¤ã�™ã‚‹ã�‹æ¤œè¨�

print("\n--- 2.2. articles ã�®è©³ç´°èª¬æ˜� (detail_desc) ã�®æ¬ æ��å€¤ ---")
# detail_descã�®æ¬ æ��å€¤ã�¯å•†å“�ã�«ã‚ˆã�£ã�¦ç•°ã�ªã‚‹å�¯èƒ½æ€§ã�Œã�‚ã‚‹ã�Ÿã‚�ã€�ãƒ†ã‚­ã‚¹ãƒˆåˆ†æ��æ™‚ã�«è€ƒæ…®
print(f"detail_desc ã�®æ¬ æ��å€¤æ•°: {articles['detail_desc'].isnull().sum()}")

print("\n--- 2.3. é‡�è¤‡ãƒ‡ãƒ¼ã‚¿ã�®ç¢ºèª� ---")
print(f"transactions ã�®é‡�è¤‡è¡Œæ•°: {transactions.duplicated().sum()}")
# é‡�è¤‡è¡Œã�Œå­˜åœ¨ã�™ã‚‹å ´å�ˆã€�ã��ã‚Œã�Œæ„�å‘³ã�®ã�‚ã‚‹é‡�è¤‡ï¼ˆä¾‹ï¼šå�Œã�˜æ—¥ã�«å�Œã�˜å•†å“�ã‚’è¤‡æ•°è³¼å…¥ï¼‰ã�ªã�®ã�‹ã€�
# ãƒ‡ãƒ¼ã‚¿å…¥åŠ›ã‚¨ãƒ©ãƒ¼ã�ªã�®ã�‹ã‚’ç¢ºèª�ã�—ã€�å¿…è¦�ã�«å¿œã�˜ã�¦å‰Šé™¤ã�¾ã�Ÿã�¯é›†ç´„ã�—ã�¾ã�™ã€‚


print("\n--- 3.1. ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�”ã�¨ã�®ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³æ•° ---")
user_interactions = transactions.groupby('customer_id').size().reset_index(name='interaction_count')

plt.figure(figsize=(10, 6))
sns.histplot(user_interactions['interaction_count'], bins=50, log_scale=True)
plt.title('Distribution of Interactions per User (Log Scale)') #ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�”ã�¨ã�®ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³æ•°åˆ†å¸ƒ (Log Scale)
plt.xlabel('Number of Interactions') #ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³æ•°
plt.ylabel('Number of Users') #ãƒ¦ãƒ¼ã‚¶ãƒ¼æ•°
plt.grid(True, which="both", ls="--", c="0.7")
plt.show()

print(f"å¹³å�‡ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³æ•°: {user_interactions['interaction_count'].mean():.2f}")
print(f"ä¸­å¤®å€¤ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³æ•°: {user_interactions['interaction_count'].median():.2f}")
print(f"ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³æ•°1ã�®ãƒ¦ãƒ¼ã‚¶ãƒ¼æ•°: {user_interactions[user_interactions['interaction_count'] == 1].shape[0]}")
print(f"ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³æ•°1ã�®ãƒ¦ãƒ¼ã‚¶ãƒ¼å‰²å�ˆ: {user_interactions[user_interactions['interaction_count'] == 1].shape[0] / user_interactions.shape[0] * 100:.2f}%")

print("\n--- 3.2. ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�®å¹´é½¢åˆ†å¸ƒ ---")
plt.figure(figsize=(10, 6))
sns.histplot(customers['age'].dropna(), bins=30, kde=True)
plt.title('Customer Age Distribution') #é¡§å®¢ã�®å¹´é½¢åˆ†å¸ƒ
plt.xlabel('Age') #å¹´é½¢
plt.ylabel('Number of Customers') #é¡§å®¢æ•°
plt.show()

print("\n--- 3.3. ã‚¯ãƒ©ãƒ–ãƒ¡ãƒ³ãƒ�ãƒ¼ã‚·ãƒƒãƒ—ã�®ã‚¹ãƒ†ãƒ¼ã‚¿ã‚¹ã�¨ãƒ•ã‚¡ãƒƒã‚·ãƒ§ãƒ³ãƒ‹ãƒ¥ãƒ¼ã‚¹å�—ä¿¡é »åº¦ ---")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(data=customers, x='club_member_status')
plt.title('Club Member Status') #ã‚¯ãƒ©ãƒ–ãƒ¡ãƒ³ãƒ�ãƒ¼ã�®ã‚¹ãƒ†ãƒ¼ã‚¿ã‚¹
plt.subplot(1, 2, 2)
sns.countplot(data=customers, x='fashion_news_frequency')
plt.title('Fashion News Frequency') #ãƒ•ã‚¡ãƒƒã‚·ãƒ§ãƒ³ãƒ‹ãƒ¥ãƒ¼ã‚¹ã�®å�—ä¿¡é »åº¦
plt.tight_layout()
plt.show()


print("\n--- 4.1. ã‚¢ã‚¤ãƒ†ãƒ ã�”ã�¨ã�®ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³æ•° ---")
item_interactions = transactions.groupby('article_id').size().reset_index(name='interaction_count')

plt.figure(figsize=(10, 6))
sns.histplot(item_interactions['interaction_count'], bins=50, log_scale=True)
plt.title('Distribution of Interactions per Item (Log Scale)') #ã‚¢ã‚¤ãƒ†ãƒ ã�”ã�¨ã�®ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³æ•°åˆ†å¸ƒ (Log Scale)
plt.xlabel('Number of Interactions') #ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³æ•°
plt.ylabel('Number of Items') #ã‚¢ã‚¤ãƒ†ãƒ æ•°
plt.grid(True, which="both", ls="--", c="0.7")
plt.show()

print(f"å¹³å�‡ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³æ•°: {item_interactions['interaction_count'].mean():.2f}")
print(f"ä¸­å¤®å€¤ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³æ•°: {item_interactions['interaction_count'].median():.2f}")
print(f"ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³æ•°1ã�®ã‚¢ã‚¤ãƒ†ãƒ æ•°: {item_interactions[item_interactions['interaction_count'] == 1].shape[0]}")
print(f"ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³æ•°1ã�®ã‚¢ã‚¤ãƒ†ãƒ å‰²å�ˆ: {item_interactions[item_interactions['interaction_count'] == 1].shape[0] / item_interactions.shape[0] * 100:.2f}%")

print("\n--- 4.2. äººæ°—ã�®ã�‚ã‚‹å•†å“�ã‚¿ã‚¤ãƒ—/ã‚°ãƒ«ãƒ¼ãƒ— ---")
# articles_df ã�¨ transactions_df ã‚’çµ�å�ˆã�—ã�¦åˆ†æ��
merged_transactions_articles = pd.merge(transactions, articles, on='article_id', how='left')

plt.figure(figsize=(12, 6))
top_product_types = merged_transactions_articles['product_type_name'].value_counts().head(10)
sns.barplot(x=top_product_types.index, y=top_product_types.values)
plt.title('Top 10 Product Types by Purchase Count') #ä¸Šä½�10å•†å“�ã‚¿ã‚¤ãƒ—åˆ¥ã�®è³¼å…¥æ•°
plt.xlabel('Product Type') #å•†å“�ã‚¿ã‚¤ãƒ—
plt.ylabel('Purchase Count') #è³¼å…¥æ•°
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
top_product_groups = merged_transactions_articles['product_group_name'].value_counts().head(10)
sns.barplot(x=top_product_groups.index, y=top_product_groups.values)
plt.title('Top 10 Product Groups by Purchase Count') #ä¸Šä½�10å•†å“�ã‚°ãƒ«ãƒ¼ãƒ—åˆ¥ã�®è³¼å…¥æ•°
plt.xlabel('Product Group') #å•†å“�ã‚°ãƒ«ãƒ¼ãƒ—
plt.ylabel('Purchase Count') #è³¼å…¥æ•°
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n--- 4.3. è‰²ã‚°ãƒ«ãƒ¼ãƒ—ã�®äººæ°—åº¦ ---")
plt.figure(figsize=(12, 6))
top_colors = merged_transactions_articles['colour_group_name'].value_counts().head(10)
sns.barplot(x=top_colors.index, y=top_colors.values)
plt.title('Top 10 Color Groups by Purchase Count') #ä¸Šä½�10è‰²ã‚°ãƒ«ãƒ¼ãƒ—åˆ¥ã�®è³¼å…¥æ•°
plt.xlabel('Color Group') #è‰²ã‚°ãƒ«ãƒ¼ãƒ—
plt.ylabel('Purchase Count') #è³¼å…¥æ•°
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# t_datã‚’datetimeå�‹ã�«å¤‰æ�›
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])


print("\n--- 5.1. æ™‚é–“çµŒé��ã�«ä¼´ã�†ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³æ•° ---")
transactions['transaction_date'] = transactions['t_dat'].dt.date
daily_transactions = transactions.groupby('transaction_date').size()

plt.figure(figsize=(15, 7))
daily_transactions.plot()
plt.title('Daily Transaction Count') #æ—¥ã�”ã�¨ã�®å�–å¼•æ•°
plt.xlabel('Date') #æ—¥ä»˜
plt.ylabel('Transaction Count') #å�–å¼•æ•°
plt.show()

print("\n--- 5.2. ä¾¡æ ¼åˆ†å¸ƒ ---")
plt.figure(figsize=(10, 6))
sns.histplot(transactions['price'], bins=50)
plt.title('Distribution of Item Prices') #å•†å“�ã�®ä¾¡æ ¼åˆ†å¸ƒ
plt.xlabel('Price') #ä¾¡æ ¼
plt.ylabel('Number of Transactions') #å�–å¼•æ•°
plt.show()

print("\n--- 5.3. è²©å£²ãƒ�ãƒ£ãƒ�ãƒ«åˆ¥ã�®å�–å¼•æ•° ---")
plt.figure(figsize=(7, 5))
sns.countplot(data=transactions, x='sales_channel_id')
plt.title('Transaction Count by Sales Channel') #è²©å£²ãƒ�ãƒ£ãƒ�ãƒ«åˆ¥ã�®å�–å¼•æ•°
plt.xlabel('Sales Channel ID') #è²©å£²ãƒ�ãƒ£ãƒ�ãƒ«ID
plt.ylabel('Transaction Count') #å�–å¼•æ•°
plt.show()


print("\n--- 6.1. ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�‚ã�Ÿã‚Šã�®ãƒ¦ãƒ‹ãƒ¼ã‚¯ã‚¢ã‚¤ãƒ†ãƒ è³¼å…¥æ•° ---")
user_unique_items = transactions.groupby('customer_id')['article_id'].nunique().reset_index(name='unique_item_count')

plt.figure(figsize=(10, 6))
sns.histplot(user_unique_items['unique_item_count'], bins=50, log_scale=True)
plt.title('Distribution of Unique Items Purchased per User (Log Scale)') #ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�”ã�¨ã�®ãƒ¦ãƒ‹ãƒ¼ã‚¯ã‚¢ã‚¤ãƒ†ãƒ è³¼å…¥æ•°åˆ†å¸ƒ (Log Scale)
plt.xlabel('Number of Unique Items') #ãƒ¦ãƒ‹ãƒ¼ã‚¯ã‚¢ã‚¤ãƒ†ãƒ æ•°
plt.ylabel('Number of Users') #ãƒ¦ãƒ¼ã‚¶ãƒ¼æ•°
plt.grid(True, which="both", ls="--", c="0.7")
plt.show()

print(f"å¹³å�‡ãƒ¦ãƒ‹ãƒ¼ã‚¯ã‚¢ã‚¤ãƒ†ãƒ è³¼å…¥æ•°: {user_unique_items['unique_item_count'].mean():.2f}")
print(f"ä¸­å¤®å€¤ãƒ¦ãƒ‹ãƒ¼ã‚¯ã‚¢ã‚¤ãƒ†ãƒ è³¼å…¥æ•°: {user_unique_items['unique_item_count'].median():.2f}")
print(f"ãƒ¦ãƒ‹ãƒ¼ã‚¯ã‚¢ã‚¤ãƒ†ãƒ è³¼å…¥æ•°1ã�®ãƒ¦ãƒ¼ã‚¶ãƒ¼æ•°: {user_unique_items[user_unique_items['unique_item_count'] == 1].shape[0]}")
print(f"ãƒ¦ãƒ‹ãƒ¼ã‚¯ã‚¢ã‚¤ãƒ†ãƒ è³¼å…¥æ•°1ã�®ãƒ¦ãƒ¼ã‚¶ãƒ¼å‰²å�ˆ: {user_unique_items[user_unique_items['unique_item_count'] == 1].shape[0] / user_unique_items.shape[0] * 100:.2f}%")

print("\n--- 6.2. é¡§å®¢ã�Œæœ€ã‚‚è³¼å…¥ã�—ã�Ÿå•†å“�ã�®ç¨®é¡� ---")
# å�„é¡§å®¢ã�Œæœ€ã‚‚å¤šã��è³¼å…¥ã�—ã�Ÿproduct_group_nameã‚’ç‰¹å®š
customer_top_product_group = merged_transactions_articles.groupby('customer_id')['product_group_name'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
plt.figure(figsize=(12, 6))
customer_top_product_group.value_counts().head(10).plot(kind='bar')
plt.title('Distribution of Most Purchased Product Groups by Customer (Top 10)') #é¡§å®¢ã�Œæœ€ã‚‚è³¼å…¥ã�—ã�Ÿå•†å“�ã‚°ãƒ«ãƒ¼ãƒ—ã�®åˆ†å¸ƒ (ä¸Šä½�10)
plt.xlabel('Product Group') #å•†å“�ã‚°ãƒ«ãƒ¼ãƒ—
plt.ylabel('Number of Customers') #é¡§å®¢æ•°
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n--- 6.3. æœ€æ–°ã�®ãƒˆãƒ©ãƒ³ã‚¶ã‚¯ã‚·ãƒ§ãƒ³æ—¥ã�®ç¢ºèª�ã�¨ã€�è©•ä¾¡æœŸé–“ã�®è¨­å®š ---")
latest_transaction_date = transactions['t_dat'].max()
print(f"Latest transaction date in the dataset: {latest_transaction_date}") #ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®æœ€æ–°å�–å¼•æ—¥
# ä¾‹ã�¨ã�—ã�¦ã€�æœ€å¾Œã�®Né€±é–“/æ—¥ã‚’ãƒ†ã‚¹ãƒˆã‚»ãƒƒãƒˆã�¨ã�—ã�¦ä½¿ç”¨ã�™ã‚‹å ´å�ˆã�®è€ƒæ…®
# ä¾‹ã�ˆã�°ã€�é��å�»2é€±é–“ã�®ãƒ‡ãƒ¼ã‚¿ã�§å­¦ç¿’ã�—ã€�ã��ã�®å¾Œã�®1é€±é–“ã�®è³¼è²·ã‚’äºˆæ¸¬ã�™ã‚‹ã�ªã�©ã€‚
# ãƒ¢ãƒ‡ãƒ«ã�®è©•ä¾¡æ™‚ã�«ã€�æ�¨è–¦ãƒªã‚¹ãƒˆã�®12å€‹ã�«ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�®å®Ÿéš›ã�®è³¼è²·è¡Œå‹•ã‚’ã�©ã‚Œã� ã�‘å�«ã‚�ã‚‰ã‚Œã‚‹ã�‹ã‚’ç¢ºèª�ã�—ã�¾ã�™ã€‚


# customer_idã�¨article_idã‚’çµ�å�ˆ
# merged_transactions_articles = pd.merge(transactions, articles, on='article_id', how='left')
# merged_transactions_full = pd.merge(merged_transactions_articles, customers, on='customer_id', how='left')


# --- ãƒ¦ãƒ¼ã‚¶ãƒ¼å�´ã�®è¦³ç‚¹ ---
print("\n--- ãƒ¦ãƒ¼ã‚¶ãƒ¼å�´ã�®è¦³ç‚¹ã�‹ã‚‰ã�®EDA ---")

# 1. è³¼å…¥é »åº¦ï¼ˆé€±ã�”ã�¨ã€�æœˆã�”ã�¨ï¼‰
print("\n--- 1.1. ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�”ã�¨ã�®è³¼å…¥é »åº¦ï¼ˆé€±ã�”ã�¨ã€�æœˆã�”ã�¨ï¼‰ ---")
# ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�”ã�¨ã�®æœ€åˆ�ã�®è³¼å…¥æ—¥ã�¨æœ€å¾Œã�®è³¼å…¥æ—¥ã‚’è¨ˆç®—
user_activity = transactions.groupby('customer_id')['t_dat'].agg(['min', 'max']).reset_index()
user_activity['duration_days'] = (user_activity['max'] - user_activity['min']).dt.days

# ç·�ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³æ•°
user_total_interactions = transactions.groupby('customer_id').size().reset_index(name='total_interactions')
user_activity = pd.merge(user_activity, user_total_interactions, on='customer_id', how='left')

# é€±ã�”ã�¨ã€�æœˆã�”ã�¨ã�®å¹³å�‡è³¼å…¥é »åº¦ã‚’æ¦‚ç®— (ã‚¢ã‚¯ãƒ†ã‚£ãƒ–æœŸé–“ã�Œ0ã�®ãƒ¦ãƒ¼ã‚¶ãƒ¼ã‚’é™¤ã��)
user_activity['avg_weekly_purchase'] = user_activity.apply(
    lambda row: (row['total_interactions'] / (row['duration_days'] / 7)) if row['duration_days'] > 0 else row['total_interactions'], axis=1
)
user_activity['avg_monthly_purchase'] = user_activity.apply(
    lambda row: (row['total_interactions'] / (row['duration_days'] / 30.4375)) if row['duration_days'] > 0 else row['total_interactions'], axis=1
)

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.histplot(user_activity['avg_weekly_purchase'].dropna(), bins=50, log_scale=True)
plt.title('Average Weekly Purchase Frequency per User') #ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�”ã�¨ã�®é€±å¹³å�‡è³¼å…¥é »åº¦
plt.xlabel('Average Weekly Purchases') #é€±å¹³å�‡è³¼å…¥æ•°
plt.ylabel('Number of Users') #ãƒ¦ãƒ¼ã‚¶ãƒ¼æ•°

plt.subplot(1, 2, 2)
sns.histplot(user_activity['avg_monthly_purchase'].dropna(), bins=50, log_scale=True)
plt.title('Average Monthly Purchase Frequency per User') #ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�”ã�¨ã�®æœˆå¹³å�‡è³¼å…¥é »åº¦
plt.xlabel('Average Monthly Purchases') #æœˆå¹³å�‡è³¼å…¥æ•°
plt.ylabel('Number of Users') #ãƒ¦ãƒ¼ã‚¶ãƒ¼æ•°
plt.tight_layout()
plt.show()

print("é€±å¹³å�‡è³¼å…¥é »åº¦ã�®è¦�ç´„çµ±è¨ˆé‡�:\n", user_activity['avg_weekly_purchase'].describe())
print("æœˆå¹³å�‡è³¼å…¥é »åº¦ã�®è¦�ç´„çµ±è¨ˆé‡�:\n", user_activity['avg_monthly_purchase'].describe())


# 2. æœ€å¾Œã�®è³¼å…¥ã�‹ã‚‰ã�®çµŒé��æ—¥æ•°
print("\n--- 2. æœ€å¾Œã�®è³¼å…¥ã�‹ã‚‰ã�®çµŒé��æ—¥æ•° ---")
# ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®æœ€æ–°æ—¥ã‚’å�–å¾—
latest_data_date = transactions['t_dat'].max()
user_last_purchase = transactions.groupby('customer_id')['t_dat'].max().reset_index()
user_last_purchase['days_since_last_purchase'] = (latest_data_date - user_last_purchase['t_dat']).dt.days

plt.figure(figsize=(10, 6))
sns.histplot(user_last_purchase['days_since_last_purchase'], bins=50, kde=True)
plt.title('Days Since Last Purchase per User') #ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�”ã�¨ã�®æœ€å¾Œã�®è³¼å…¥ã�‹ã‚‰ã�®çµŒé��æ—¥æ•°
plt.xlabel('Days Elapsed') #çµŒé��æ—¥æ•°
plt.ylabel('Number of Users') #ãƒ¦ãƒ¼ã‚¶ãƒ¼æ•°
plt.show()

print("æœ€å¾Œã�®è³¼å…¥ã�‹ã‚‰ã�®çµŒé��æ—¥æ•°ã�®è¦�ç´„çµ±è¨ˆé‡�:\n", user_last_purchase['days_since_last_purchase'].describe())


# 3. å¹³å�‡è³¼å…¥ä¾¡æ ¼ã€�è³¼å…¥å�ˆè¨ˆé‡‘é¡�
print("\n--- 3. å¹³å�‡è³¼å…¥ä¾¡æ ¼ã€�è³¼å…¥å�ˆè¨ˆé‡‘é¡� ---")
user_purchase_stats = transactions.groupby('customer_id')['price'].agg(['mean', 'sum']).reset_index()
user_purchase_stats.rename(columns={'mean': 'avg_purchase_price', 'sum': 'total_purchase_amount'}, inplace=True)

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.histplot(user_purchase_stats['avg_purchase_price'], bins=50, kde=True)
plt.title('Average Purchase Price per User') #ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�”ã�¨ã�®å¹³å�‡è³¼å…¥ä¾¡æ ¼
plt.xlabel('Average Purchase Price') #å¹³å�‡è³¼å…¥ä¾¡æ ¼
plt.ylabel('Number of Users') #ãƒ¦ãƒ¼ã‚¶ãƒ¼æ•°

plt.subplot(1, 2, 2)
sns.histplot(user_purchase_stats['total_purchase_amount'], bins=50, log_scale=True)
plt.title('Total Purchase Amount per User (Log Scale)') #ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�”ã�¨ã�®è³¼å…¥å�ˆè¨ˆé‡‘é¡� (Log Scale)
plt.xlabel('Total Purchase Amount') #è³¼å…¥å�ˆè¨ˆé‡‘é¡�
plt.ylabel('Number of Users') #ãƒ¦ãƒ¼ã‚¶ãƒ¼æ•°
plt.tight_layout()
plt.show()

print("ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�”ã�¨ã�®å¹³å�‡è³¼å…¥ä¾¡æ ¼ã�®è¦�ç´„çµ±è¨ˆé‡�:\n", user_purchase_stats['avg_purchase_price'].describe())
print("ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�”ã�¨ã�®è³¼å…¥å�ˆè¨ˆé‡‘é¡�ã�®è¦�ç´„çµ±è¨ˆé‡�:\n", user_purchase_stats['total_purchase_amount'].describe())


# 4. è³¼å…¥ã�—ã�Ÿã‚¢ã‚¤ãƒ†ãƒ ã�®ã‚«ãƒ†ã‚´ãƒªã€�è‰²ã€�ã‚°ãƒ«ãƒ¼ãƒ—ã�®å¤šæ§˜æ€§
print("\n--- 4. è³¼å…¥ã�—ã�Ÿã‚¢ã‚¤ãƒ†ãƒ ã�®ã‚«ãƒ†ã‚´ãƒªã€�è‰²ã€�ã‚°ãƒ«ãƒ¼ãƒ—ã�®å¤šæ§˜æ€§ ---")
# product_group_nameã�®å¤šæ§˜æ€§
user_product_group_diversity = merged_transactions_full.groupby('customer_id')['product_group_name'].nunique().reset_index(name='unique_product_groups')
plt.figure(figsize=(10, 6))
sns.histplot(user_product_group_diversity['unique_product_groups'], bins=range(1, user_product_group_diversity['unique_product_groups'].max() + 2), kde=False)
plt.title('Diversity of Purchased Product Groups per User') #ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�”ã�¨ã�®è³¼å…¥å•†å“�ã‚°ãƒ«ãƒ¼ãƒ—ã�®å¤šæ§˜æ€§
plt.xlabel('Number of Unique Product Groups') #ãƒ¦ãƒ‹ãƒ¼ã‚¯ã�ªå•†å“�ã‚°ãƒ«ãƒ¼ãƒ—æ•°
plt.ylabel('Number of Users') #ãƒ¦ãƒ¼ã‚¶ãƒ¼æ•°
plt.xticks(rotation=45, ha='right')
plt.show()
print("ãƒ¦ãƒ‹ãƒ¼ã‚¯ã�ªå•†å“�ã‚°ãƒ«ãƒ¼ãƒ—æ•°ã�®è¦�ç´„çµ±è¨ˆé‡�:\n", user_product_group_diversity['unique_product_groups'].describe())

# colour_group_nameã�®å¤šæ§˜æ€§
user_color_diversity = merged_transactions_full.groupby('customer_id')['colour_group_name'].nunique().reset_index(name='unique_color_groups')
plt.figure(figsize=(10, 6))
sns.histplot(user_color_diversity['unique_color_groups'], bins=range(1, user_color_diversity['unique_color_groups'].max() + 2), kde=False)
plt.title('Diversity of Purchased Color Groups per User') #ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�”ã�¨ã�®è³¼å…¥è‰²ã‚°ãƒ«ãƒ¼ãƒ—ã�®å¤šæ§˜æ€§
plt.xlabel('Number of Unique Color Groups') #ãƒ¦ãƒ‹ãƒ¼ã‚¯ã�ªè‰²ã‚°ãƒ«ãƒ¼ãƒ—æ•°
plt.ylabel('Number of Users') #ãƒ¦ãƒ¼ã‚¶ãƒ¼æ•°
plt.xticks(rotation=45, ha='right')
plt.show()
print("ãƒ¦ãƒ‹ãƒ¼ã‚¯ã�ªè‰²ã‚°ãƒ«ãƒ¼ãƒ—æ•°ã�®è¦�ç´„çµ±è¨ˆé‡�:\n", user_color_diversity['unique_color_groups'].describe())


# 5. å¹´é½¢ã€�ä¼šå“¡ã‚¹ãƒ†ãƒ¼ã‚¿ã‚¹ã€�ãƒ•ã‚¡ãƒƒã‚·ãƒ§ãƒ³ãƒ‹ãƒ¥ãƒ¼ã‚¹è³¼èª­çŠ¶æ³�
print("\n--- 5. å¹´é½¢ã€�ä¼šå“¡ã‚¹ãƒ†ãƒ¼ã‚¿ã‚¹ã€�ãƒ•ã‚¡ãƒƒã‚·ãƒ§ãƒ³ãƒ‹ãƒ¥ãƒ¼ã‚¹è³¼èª­çŠ¶æ³� ---")
# å¹´é½¢åˆ†å¸ƒã�¯ã�™ã�§ã�«ç¢ºèª�æ¸ˆã�¿ã� ã�Œã€�ã�“ã�“ã�§å†�ç¢ºèª�
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(customers['age'].dropna(), bins=30, kde=True)
plt.title('Customer Age Distribution') #é¡§å®¢ã�®å¹´é½¢åˆ†å¸ƒ
plt.xlabel('Age') #å¹´é½¢
plt.ylabel('Number of Customers') #é¡§å®¢æ•°

plt.subplot(1, 3, 2)
sns.countplot(data=customers, x='club_member_status', palette='viridis')
plt.title('Club Member Status') #ã‚¯ãƒ©ãƒ–ãƒ¡ãƒ³ãƒ�ãƒ¼ã�®ã‚¹ãƒ†ãƒ¼ã‚¿ã‚¹
plt.xlabel('Status') #ã‚¹ãƒ†ãƒ¼ã‚¿ã‚¹
plt.ylabel('Number of Customers') #é¡§å®¢æ•°

plt.subplot(1, 3, 3)
sns.countplot(data=customers, x='fashion_news_frequency', palette='magma')
plt.title('Fashion News Subscription Frequency') #ãƒ•ã‚¡ãƒƒã‚·ãƒ§ãƒ³ãƒ‹ãƒ¥ãƒ¼ã‚¹ã�®å�—ä¿¡é »åº¦
plt.xlabel('Frequency') #å�—ä¿¡é »åº¦
plt.ylabel('Number of Customers') #é¡§å®¢æ•°
plt.tight_layout()
plt.show()

# FN, Active ã�®å€¤ã�®åˆ†å¸ƒ
print("\nFN (ãƒ•ã‚¡ãƒƒã‚·ãƒ§ãƒ³ãƒ‹ãƒ¥ãƒ¼ã‚¹ãƒ¬ã‚¿ãƒ¼è³¼èª­) ã�®åˆ†å¸ƒ:\n", customers['FN'].value_counts(dropna=False))
print("\nActive (ã‚³ãƒŸãƒ¥ãƒ‹ã‚±ãƒ¼ã‚·ãƒ§ãƒ³ã‚¢ã‚¯ãƒ†ã‚£ãƒ–) ã�®åˆ†å¸ƒ:\n", customers['Active'].value_counts(dropna=False))


# --- ã‚¢ã‚¤ãƒ†ãƒ å�´ã�®è¦³ç‚¹ ---
print("\n--- ã‚¢ã‚¤ãƒ†ãƒ å�´ã�®è¦³ç‚¹ã�‹ã‚‰ã�®EDA ---")

# 1. äººæ°—åº¦ï¼ˆè³¼å…¥å›�æ•°ã€�ãƒ¦ãƒ‹ãƒ¼ã‚¯ãƒ¦ãƒ¼ã‚¶ãƒ¼æ•°ï¼‰
print("\n--- 1. äººæ°—åº¦ï¼ˆè³¼å…¥å›�æ•°ã€�ãƒ¦ãƒ‹ãƒ¼ã‚¯ãƒ¦ãƒ¼ã‚¶ãƒ¼æ•°ï¼‰ ---")
item_popularity = transactions.groupby('article_id').agg(
    purchase_count=('customer_id', 'size'),
    unique_user_count=('customer_id', 'nunique')
).reset_index()

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.histplot(item_popularity['purchase_count'], bins=50, log_scale=True)
plt.title('Number of Purchases per Item (Log Scale)') #ã‚¢ã‚¤ãƒ†ãƒ ã�”ã�¨ã�®è³¼å…¥å›�æ•° (Log Scale)
plt.xlabel('Purchase Count') #è³¼å…¥å›�æ•°
plt.ylabel('Number of Items') #ã‚¢ã‚¤ãƒ†ãƒ æ•°

plt.subplot(1, 2, 2)
sns.histplot(item_popularity['unique_user_count'], bins=50, log_scale=True)
plt.title('Number of Unique Users per Item (Log Scale)') #ã‚¢ã‚¤ãƒ†ãƒ ã�”ã�¨ã�®ãƒ¦ãƒ‹ãƒ¼ã‚¯ãƒ¦ãƒ¼ã‚¶ãƒ¼æ•° (Log Scale)
plt.xlabel('Unique User Count') #ãƒ¦ãƒ‹ãƒ¼ã‚¯ãƒ¦ãƒ¼ã‚¶ãƒ¼æ•°
plt.ylabel('Number of Items') #ã‚¢ã‚¤ãƒ†ãƒ æ•°
plt.tight_layout()
plt.show()

print("ã‚¢ã‚¤ãƒ†ãƒ ã�”ã�¨ã�®è³¼å…¥å›�æ•°ã�®è¦�ç´„çµ±è¨ˆé‡�:\n", item_popularity['purchase_count'].describe())
print("ã‚¢ã‚¤ãƒ†ãƒ ã�”ã�¨ã�®ãƒ¦ãƒ‹ãƒ¼ã‚¯ãƒ¦ãƒ¼ã‚¶ãƒ¼æ•°ã�®è¦�ç´„çµ±è¨ˆé‡�:\n", item_popularity['unique_user_count'].describe())


# 2. ç™ºå£²æ—¥ã�‹ã‚‰ã�®çµŒé��æ—¥æ•° (articles.csvã�«ç™ºå£²æ—¥æƒ…å ±ã�Œã�ªã�„ã�Ÿã‚�ã€�ã�“ã�“ã�§ã�¯å‰²æ„›)
# ã‚‚ã�—ç™ºå£²æ—¥æƒ…å ±ã�Œåˆ¥é€”ã�‚ã‚Œã�°ã€�ä»¥ä¸‹ã�®è¨ˆç®—ã‚’è¡Œã�†
# print("\n--- 2. ç™ºå£²æ—¥ã�‹ã‚‰ã�®çµŒé��æ—¥æ•° ---")
# # ä»®ã�«'release_date'ã‚«ãƒ©ãƒ ã�Œã�‚ã‚‹ã�¨ã�—ã�¦
# articles_df['release_date'] = pd.to_datetime(articles_df['release_date'])
# current_date = pd.to_datetime('2024-01-01') # ã�¾ã�Ÿã�¯transactions_dfã�®æœ€æ–°æ—¥ã�ªã�©
# articles_df['days_since_release'] = (current_date - articles_df['release_date']).dt.days
# plt.figure(figsize=(10, 6))
# sns.histplot(articles_df['days_since_release'].dropna(), bins=50, kde=True)
# plt.title('ã‚¢ã‚¤ãƒ†ãƒ ã�®ç™ºå£²æ—¥ã�‹ã‚‰ã�®çµŒé��æ—¥æ•°')
# plt.xlabel('çµŒé��æ—¥æ•°')
# plt.ylabel('ã‚¢ã‚¤ãƒ†ãƒ æ•°')
# plt.show()
# print("ç™ºå£²æ—¥ã�‹ã‚‰ã�®çµŒé��æ—¥æ•°ã�®è¦�ç´„çµ±è¨ˆé‡�:\n", articles_df['days_since_release'].describe())


# 3. å¹³å�‡ä¾¡æ ¼ (å•†å“�ã�®ä¾¡æ ¼ã�¯transactions_dfã�«ã�‚ã‚‹ã�Ÿã‚�ã€�ã�“ã�“ã�§å†�ç¢ºèª�)
print("\n--- 3. å¹³å�‡ä¾¡æ ¼ ---")
# article_idã�”ã�¨ã�®å¹³å�‡ä¾¡æ ¼ã‚’è¨ˆç®— (è¤‡æ•°ã�®ãƒˆãƒ©ãƒ³ã‚¶ã‚¯ã‚·ãƒ§ãƒ³ã�§ä¾¡æ ¼ã�Œç•°ã�ªã‚‹å�¯èƒ½æ€§ã‚’è€ƒæ…®)
item_avg_price = transactions.groupby('article_id')['price'].mean().reset_index(name='avg_item_price')
plt.figure(figsize=(10, 6))
sns.histplot(item_avg_price['avg_item_price'], bins=50, kde=True)
plt.title('Average Price per Item') #ã‚¢ã‚¤ãƒ†ãƒ ã�”ã�¨ã�®å¹³å�‡ä¾¡æ ¼
plt.xlabel('Average Price') #å¹³å�‡ä¾¡æ ¼
plt.ylabel('Number of Items') #ã‚¢ã‚¤ãƒ†ãƒ æ•°
plt.show()
print("ã‚¢ã‚¤ãƒ†ãƒ ã�”ã�¨ã�®å¹³å�‡ä¾¡æ ¼ã�®è¦�ç´„çµ±è¨ˆé‡�:\n", item_avg_price['avg_item_price'].describe())


# 4. ã‚«ãƒ†ã‚´ãƒªã€�è‰²ã€�ã‚°ãƒ«ãƒ¼ãƒ—ã€�è©³ç´°èª¬æ˜�ã�‹ã‚‰ã�®ãƒ†ã‚­ã‚¹ãƒˆç‰¹å¾´é‡�ï¼ˆTF-IDF, Word2Vecã�ªã�©ï¼‰
print("\n--- 4. ã‚«ãƒ†ã‚´ãƒªã€�è‰²ã€�ã‚°ãƒ«ãƒ¼ãƒ—ã€�è©³ç´°èª¬æ˜�ã�‹ã‚‰ã�®ãƒ†ã‚­ã‚¹ãƒˆç‰¹å¾´é‡� ---")
# ã‚«ãƒ†ã‚´ãƒªã€�è‰²ã€�ã‚°ãƒ«ãƒ¼ãƒ—ã�®åˆ†å¸ƒã�¯ã�™ã�§ã�«ç¢ºèª�æ¸ˆã�¿ã� ã�Œã€�ã�“ã�“ã�§æ”¹ã‚�ã�¦ãƒ¦ãƒ‹ãƒ¼ã‚¯æ•°ã‚’ç¢ºèª�
print(articles[['product_type_name', 'product_group_name', 'colour_group_name', 'detail_desc']].head())
print("\nproduct_type_name ã�®ãƒ¦ãƒ‹ãƒ¼ã‚¯æ•°:", articles['product_type_name'].nunique())
print("product_group_name ã�®ãƒ¦ãƒ‹ãƒ¼ã‚¯æ•°:", articles['product_group_name'].nunique())
print("colour_group_name ã�®ãƒ¦ãƒ‹ãƒ¼ã‚¯æ•°:", articles['colour_group_name'].nunique())

# detail_desc ã�®æ¬ æ��å€¤ç¢ºèª�ï¼ˆãƒ†ã‚­ã‚¹ãƒˆåˆ†æ��ã�®å‰�ã�«å‡¦ç�†ã�Œå¿…è¦�ï¼‰
print(f"detail_desc ã�®æ¬ æ��å€¤æ•°: {articles['detail_desc'].isnull().sum()}")

# detail_desc ã�®ãƒ¯ãƒ¼ãƒ‰ã‚¯ãƒ©ã‚¦ãƒ‰ã‚„é »å‡ºå�˜èª�åˆ†æ��ã�¯ã€�åˆ¥é€”NLPãƒ©ã‚¤ãƒ–ãƒ©ãƒª (NLTK, spaCy, scikit-learn) ã‚’ç”¨ã�„ã�¦è¡Œã�†
# ä¾‹: æœ€åˆ�ã�®ã�„ã��ã�¤ã�‹ã�®è©³ç´°èª¬æ˜�ã‚’è¡¨ç¤º
print("\næœ€åˆ�ã�®5ã�¤ã�®è©³ç´°èª¬æ˜�ã�®ä¾‹:\n")
for i, desc in enumerate(articles['detail_desc'].head()):
    print(f"Article {articles['article_id'].iloc[i]}: {desc}")

# ç°¡å�˜ã�ªå�˜èª�é »åº¦åˆ†æ��ï¼ˆä¾‹ï¼šæœ€ã‚‚ä¸€èˆ¬çš„ã�ªå�˜èª�ï¼‰
from collections import Counter
import re

# å°�æ–‡å­—ã�«å¤‰æ�›ã�—ã€�æ•°å­—ã�¨å�¥èª­ç‚¹ã‚’å‰Šé™¤
articles['cleaned_desc'] = articles['detail_desc'].fillna('').astype(str).apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x).lower())
all_words = ' '.join(articles['cleaned_desc']).split()
word_counts = Counter(all_words)
print("\nè©³ç´°èª¬æ˜�ã�§æœ€ã‚‚é »ç¹�ã�«ç�¾ã‚Œã‚‹å�˜èª� (Top 20):\n", word_counts.most_common(20))
# ã�“ã�“ã�§'a', 'the', 'is'ã�ªã�©ã�®ã‚¹ãƒˆãƒƒãƒ—ãƒ¯ãƒ¼ãƒ‰ã�Œè¦‹ã‚‰ã‚Œã‚‹ã€‚å®Ÿéš›ã�®åˆ†æ��ã�§ã�¯ã�“ã‚Œã‚‰ã‚’é™¤å¤–ã�™ã‚‹ã€‚


# --- ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³å�´ã�®è¦³ç‚¹ ---
print("\n--- ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³å�´ã�®è¦³ç‚¹ã�‹ã‚‰ã�®EDA ---")

# 1. è³¼å…¥å›�æ•°ï¼ˆãƒªãƒ”ãƒ¼ãƒˆè³¼å…¥ï¼‰
print("\n--- 1. è³¼å…¥å›�æ•°ï¼ˆãƒªãƒ”ãƒ¼ãƒˆè³¼å…¥ï¼‰ ---")
# ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�¨ã‚¢ã‚¤ãƒ†ãƒ ã�®çµ„ã�¿å�ˆã‚�ã�›ã�”ã�¨ã�®è³¼å…¥å›�æ•°
repeat_purchases = transactions.groupby(['customer_id', 'article_id']).size().reset_index(name='purchase_count')
repeat_purchases_gt_1 = repeat_purchases[repeat_purchases['purchase_count'] > 1]

print(f"ãƒªãƒ”ãƒ¼ãƒˆè³¼å…¥ã�•ã‚Œã�Ÿ (å�Œã�˜ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�Œå�Œã�˜å•†å“�ã‚’è¤‡æ•°å›�è³¼å…¥) ãƒ¦ãƒ‹ãƒ¼ã‚¯ã�ªçµ„ã�¿å�ˆã‚�ã�›ã�®æ•°: {repeat_purchases_gt_1.shape[0]}")
print(f"ç·�ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ã‚·ãƒ§ãƒ³ã�«å¯¾ã�™ã‚‹ãƒªãƒ”ãƒ¼ãƒˆè³¼å…¥ã�®å‰²å�ˆ: {repeat_purchases_gt_1['purchase_count'].sum() / transactions.shape[0] * 100:.2f}%")

plt.figure(figsize=(14, 8))
sns.histplot(repeat_purchases_gt_1['purchase_count'], bins=range(2, repeat_purchases['purchase_count'].max() + 2), discrete=True)
plt.title('Repeat Purchase Count per User-Item Combination') #ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�¨ã‚¢ã‚¤ãƒ†ãƒ ã�®çµ„ã�¿å�ˆã‚�ã�›ã�”ã�¨ã�®ãƒªãƒ”ãƒ¼ãƒˆè³¼å…¥å›�æ•°
plt.xlabel('Purchase Count') #è³¼å…¥å›�æ•°
plt.ylabel('Number of Combinations') #çµ„ã�¿å�ˆã‚�ã�›æ•°

# Xè»¸ã�®ç›®ç››ã‚Šã‚’èª¿æ•´ã�—ã�¦ã€�é‡�è¤‡ã‚’æ¸›ã‚‰ã�™
max_purchase_count = repeat_purchases['purchase_count'].max()
if max_purchase_count < 10: # ä¾‹: æœ€å¤§9å›�ã�¾ã�§ã�ªã‚‰å…¨ã�¦è¡¨ç¤º
    tick_interval = 1
elif max_purchase_count < 25: # ä¾‹: æœ€å¤§24å›�ã�¾ã�§ã�ªã‚‰2å›�ã�Šã��ã�«è¡¨ç¤º
    tick_interval = 2
elif max_purchase_count < 50: # ä¾‹: æœ€å¤§49å›�ã�¾ã�§ã�ªã‚‰5å›�ã�Šã��ã�«è¡¨ç¤º
    tick_interval = 5
else: # ã��ã‚Œä»¥ä¸Šã�ªã‚‰10å›�ã�Šã��ã�«è¡¨ç¤º
    tick_interval = 10

plt.xticks(np.arange(2, max_purchase_count + 1, tick_interval), rotation=45, ha='right') # ç›®ç››ã‚Šé–“éš”ã‚’èª¿æ•´ã�—ã€�45åº¦å›�è»¢ã�•ã�›ã‚‹

plt.grid(axis='y', linestyle='--', alpha=0.7) # ã‚°ãƒªãƒƒãƒ‰ç·šã‚’è¿½åŠ ã�—ã�¦èª­ã�¿ã‚„ã�™ã��ã�™ã‚‹
plt.tight_layout() # ãƒ¬ã‚¤ã‚¢ã‚¦ãƒˆã‚’è‡ªå‹•èª¿æ•´


# 2. è³¼å…¥ã�®æ›œæ—¥ã€�æ™‚é–“å¸¯
print("\n--- 2. è³¼å…¥ã�®æ›œæ—¥ã€�æ™‚é–“å¸¯ ---")
transactions['day_of_week'] = transactions['t_dat'].dt.day_name()
transactions['hour_of_day'] = transactions['t_dat'].dt.hour

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.countplot(data=transactions, x='day_of_week', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], palette='coolwarm')
plt.title('Purchase Count by Day of Week') #æ›œæ—¥ã�”ã�¨ã�®è³¼å…¥æ•°
plt.xlabel('Day of Week') #æ›œæ—¥
plt.ylabel('Purchase Count') #è³¼å…¥æ•°

plt.subplot(1, 2, 2)
sns.histplot(transactions['hour_of_day'], bins=24, kde=False)
plt.title('Purchase Count by Hour of Day') #æ™‚é–“å¸¯ã�”ã�¨ã�®è³¼å…¥æ•°
plt.xlabel('Hour of Day (24-hour format)') #æ™‚é–“ï¼ˆ24æ™‚é–“è¡¨è¨˜)
plt.ylabel('Purchase Count') #è³¼å…¥æ•°
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()


# 3. è²©å£²ãƒ�ãƒ£ãƒ�ãƒ«ï¼ˆåº—èˆ— vs. ã‚ªãƒ³ãƒ©ã‚¤ãƒ³ï¼‰
print("\n--- 3. è²©å£²ãƒ�ãƒ£ãƒ�ãƒ«ï¼ˆåº—èˆ— vs. ã‚ªãƒ³ãƒ©ã‚¤ãƒ³ï¼‰ ---")
plt.figure(figsize=(7, 5))
sns.countplot(data=transactions, x='sales_channel_id', palette='pastel')
plt.title('Transaction Count by Sales Channel') #è²©å£²ãƒ�ãƒ£ãƒ�ãƒ«åˆ¥ã�®å�–å¼•æ•°
plt.xlabel('Sales Channel ID (1=Store, 2=Online)') #è²©å£²ãƒ�ãƒ£ãƒ�ãƒ«ID (1=åº—èˆ—, 2=ã‚ªãƒ³ãƒ©ã‚¤ãƒ³)
plt.ylabel('Transaction Count') #å�–å¼•æ•°
plt.show()

print("è²©å£²ãƒ�ãƒ£ãƒ�ãƒ«åˆ¥ã�®å�–å¼•æ•°:\n", transactions['sales_channel_id'].value_counts())
print(f"ã‚ªãƒ³ãƒ©ã‚¤ãƒ³è³¼å…¥ã�®å‰²å�ˆ: {transactions['sales_channel_id'].value_counts(normalize=True).get(2, 0) * 100:.2f}%")
print(f"åº—èˆ—è³¼å…¥ã�®å‰²å�ˆ: {transactions['sales_channel_id'].value_counts(normalize=True).get(1, 0) * 100:.2f}%")

