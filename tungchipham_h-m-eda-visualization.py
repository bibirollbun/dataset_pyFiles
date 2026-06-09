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


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm


articles = pd.read_csv("/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv")
customers = pd.read_csv("/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv")
transactions = pd.read_csv("/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv")


articles.head()


articles.info()


numeric_cols = articles.select_dtypes(include='int64')
negative_ones_count = (numeric_cols == -1).sum()

print(negative_ones_count)



# Làm sạch product_name
def clean_name(name):
    name = str(name).lower().strip()  
    name = name.replace('(1)', '')  
    name = ' '.join(name.split())  
    return name
    
articles['cleaned_name'] = articles['prod_name'].apply(clean_name)



standard_names = articles.groupby('product_code')['cleaned_name'] \
                            .agg(lambda x: x.value_counts().idxmax()) \
                            .reset_index().rename(columns={'cleaned_name': 'standard_name'})



articles = articles.merge(standard_names, on='product_code', how='left')
articles['prod_name'] = articles['standard_name']

articles.drop(columns=['cleaned_name', 'standard_name'], inplace=True)



# Fill giá trị -1 của trường product_type_no dựa trên trường product_no,prod_name
valid_reference = articles[
    (articles['product_type_no'] != -1) &
    (articles['product_type_name'].str.lower() != 'unknown') &
    (articles['product_group_name'].str.lower() != 'unknown')
].drop_duplicates(subset=['product_code'])

# Chỉ giữ các cột cần thiết
valid_reference = valid_reference[['product_code', 'product_type_no', 'product_type_name', 'product_group_name']]



# Gộp dữ liệu gốc với dữ liệu hợp lệ theo product_code
articles = articles.merge(valid_reference, on='product_code', how='left', suffixes=('', '_ref'))



# Với product_type_no
articles['product_type_no'] = articles.apply(
    lambda row: row['product_type_no_ref'] if row['product_type_no'] == -1 else row['product_type_no'],
    axis=1
)

# Với product_type_name
articles['product_type_name'] = articles.apply(
    lambda row: row['product_type_name_ref'] if row['product_type_name'].lower() == 'unknown' else row['product_type_name'],
    axis=1
)

# Với product_group_name
articles['product_group_name'] = articles.apply(
    lambda row: row['product_group_name_ref'] if row['product_group_name'].lower() == 'unknown' else row['product_group_name'],
    axis=1
)



# Xóa cột tham chiếu
articles.drop(columns=['product_type_no_ref', 'product_type_name_ref', 'product_group_name_ref'], inplace=True)

# Và chuyển product_type_no về kiểu int
articles['product_type_no'] = articles['product_type_no'].astype('Int64')  # hoặc int nếu chắc chắn không có NaN



articles.head()


type_counts = articles['product_type_name'].value_counts()
type_counts


type_percent = (type_counts / type_counts.sum() * 100).round(2)


top10 = type_percent.head(10)


import matplotlib.cm as cm


labels = top10.index[::-1]
values = top10.values[::-1]
norm = plt.Normalize(values.min(), values.max())
colors = cm.magma(norm(values)) 
# Vẽ biểu đồ
plt.figure(figsize=(8, 5))
bars = plt.barh(top10.index[::-1], top10.values[::-1], color=colors)

# Thêm giá trị phần trăm lên đầu cột
for i, v in enumerate(top10.values[::-1]):
    plt.text(v + 0.2, i, f'{v:.2f}%', va='center')

plt.xlabel('% Quantity')
plt.title('% Quantity by Product Type')
plt.tight_layout()
plt.show()



# Giả sử df_t là transactions có article_id, price
# df_a là articles có article_id, prod_name

# Kết hợp bảng để có tên sản phẩm
merged = transactions.merge(articles[['article_id', 'prod_name']], on='article_id', how='left')

# Tìm giá cao nhất cho mỗi article_id
max_price = merged.groupby(['article_id', 'prod_name'])['price'].max().reset_index()

# Lấy top 10 sản phẩm theo giá
top10_price = max_price.sort_values(by='price', ascending=False).head(10)

print(top10_price)



articles.groupby(['index_group_name', 'index_name']).count()['article_id']


customers.head()


customers.info()


import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style("darkgrid")
f, ax = plt.subplots(figsize=(10,5))
ax = sns.histplot(data=customers, x='age', bins=50, color='orange')
ax.set_xlabel('Distribution of the customers age')
plt.show()


# Tính tổng tiền chi tiêu của mỗi khách hàng.
df_cust_prices = transactions[["customer_id", "price"]].groupby("customer_id").sum()
df_cust_prices.head()


# Tính số lượng món hàng đã mua của mỗi khách hàng
df_cust_qty = transactions[["customer_id", "article_id"]].groupby("customer_id").count()
df_cust_qty.head()


# Kết hợp tổng tiền chi tiêu và số lượng sản phẩm mua thành một bảng.
cust_qty_price = pd.merge(df_cust_prices, df_cust_qty, on='customer_id', how='inner')
cust_qty_price.head()


# Ghép thêm thông tin chi tiết về khách hàng vào bảng đã tổng hợp.
cust_details = pd.merge(cust_qty_price, customers, on='customer_id', how='inner')
cust_details.head()


# Gán nhóm tuổi cho từng khách hàng để phân tích dễ hơn.
cust_details['age_groups'] = pd.cut(cust_details['age'], bins=[16, 20, 30, 40,50, 60, 70, float('Inf')], labels=['16-20', '20-30','30-40','40-50','50-60','60-70' , '70+'])


plt.figure(figsize=(8,5))
plt.title("Purchased quantity by age group\n", fontweight="bold", size=28)
g = sns.barplot(x="age_groups", y="Purchased Quantity(%)", data=cust_details.groupby("age_groups")["article_id"].sum() \
            .transform(lambda x: (x / x.sum() * 100)).rename('Purchased Quantity(%)').reset_index(), palette="icefire", edgecolor="black")
plt.xlabel("Age Group",fontweight="bold", size=22)
plt.ylabel("Purchased Quantity (%)",fontweight="bold", size=19)
for container in g.containers:
    g.bar_label(container, padding = 5, fmt='%.1f', fontsize=18, color="black")
plt.grid(axis="y",color = 'grey', linestyle = '--', linewidth = 1.5)
plt.show()


plt.figure(figsize=(8,5))
plt.title("Company Earnings by age group\n", fontweight="bold", size=28)
g = sns.barplot(x="age_groups", y="earning(%)", data=cust_details.groupby("age_groups")["price"].sum() \
            .transform(lambda x: (x / x.sum() * 100)).rename('earning(%)').reset_index(), palette="icefire",edgecolor="black")
plt.xlabel("Age Group",fontweight="bold", size=22)
plt.ylabel("Earnings (%)",fontweight="bold", size=25)
for container in g.containers:
    g.bar_label(container, padding = 5, fmt='%.1f', fontsize=18, color="black")
plt.grid(axis="y",color = 'grey', linestyle = '--', linewidth = 1.5)
plt.show()


df_qty_by_age_news = (
    cust_details
    .groupby(['age_groups', 'fashion_news_frequency'])['article_id']
    .count()
    .reset_index()
    .rename(columns={'article_id': 'purchased_qty'})
)
df_qty_by_age_news = df_qty_by_age_news[df_qty_by_age_news['fashion_news_frequency'].isin(['Regularly', 'NONE'])]
df_qty_by_age_news['pct'] = df_qty_by_age_news.groupby('age_groups')['purchased_qty'].transform(lambda x: (x / x.sum()) * 100)


plt.figure(figsize=(14, 7))
g = sns.barplot(
    data=df_qty_by_age_news,
    x='age_groups',
    y='pct',
    hue='fashion_news_frequency',
    palette={'Regularly': '#C68EFD', 'NONE': '#0118D8'}
)

plt.title("Purchased Quantity (%) by Fashion News Frequency & Age Group", fontsize=20, fontweight='bold')
plt.xlabel("Age Group", fontsize=16, fontweight='bold')
plt.ylabel("Purchased Quantity (%)", fontsize=16, fontweight='bold')
plt.legend(title='Fashion News Frequency', fontsize=12, title_fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# ✅ Thêm số % trên từng cột
for container in g.containers:
    g.bar_label(container, fmt='%.1f%%', fontsize=12, padding=3, color='black')

plt.show()


transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
transactions['day_of_week'] = transactions['t_dat'].dt.weekday + 1



# Đếm số lượt mua hàng theo từng ngày trong tuần
day_counts = transactions['day_of_week'].value_counts().sort_index()



plt.figure(figsize=(8, 5))
bars = plt.bar(day_counts.index, day_counts.values, color='dodgerblue')

# Thêm nhãn
plt.xticks(ticks=range(1, 8), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.xlabel('Day of Week')
plt.ylabel('Number of Purchases')
plt.title('Customer Purchase Frequency by Day of Week')
plt.tight_layout()
plt.show()


