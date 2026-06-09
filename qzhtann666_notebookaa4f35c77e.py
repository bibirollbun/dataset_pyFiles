import numpy as np 
import pandas as pd
from pandasql import sqldf

from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập style đồ thị (nền trắng, grid trắng, bố cục tự động).
plt.style.use('seaborn-white')
sns.set_style("whitegrid")
sns.despine()
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10)

import matplotlib as mpl

mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = False
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


df_a = pd.read_csv("/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv")
df_t = pd.read_csv("/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv")
df_c = pd.read_csv("/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv")

# Xóa cột 'postal_code' khỏi DataFrame
df_c = df_c.drop(columns=['postal_code'])


# Tính tổng tiền chi tiêu của mỗi khách hàng.
df_cust_prices = df_t[["customer_id", "price"]].groupby("customer_id").sum()
df_cust_prices.head()


# Tính số lượng món hàng đã mua của mỗi khách hàng
df_cust_qty = df_t[["customer_id", "article_id"]].groupby("customer_id").count()
df_cust_qty.head()


# Kết hợp tổng tiền chi tiêu và số lượng sản phẩm mua thành một bảng.
cust_qty_price = pd.merge(df_cust_prices, df_cust_qty, on='customer_id', how='inner')
cust_qty_price.head()


# Ghép thêm thông tin chi tiết về khách hàng vào bảng đã tổng hợp.
cust_details = pd.merge(cust_qty_price, df_c, on='customer_id', how='inner')
cust_details.head()


# Gán nhóm tuổi cho từng khách hàng để phân tích dễ hơn.
cust_details['age_groups'] = pd.cut(cust_details['age'], bins=[16, 20, 30, 40,50, 60, 70, float('Inf')], labels=['16-20', '20-30','30-40','40-50','50-60','60-70' , '70+'])

# Gán nhóm "Unknown" cho các dòng bị NaN ở age
cust_details['age_groups'] = cust_details['age_groups'].cat.add_categories('Unknown')
cust_details['age_groups'] = cust_details['age_groups'].fillna('Unknown')


# Tổng số khách hàng mỗi nhóm tuổi
age_counts = cust_details.groupby('age_groups').agg(total_customers=('customer_id', 'count')).reset_index()

# Số khách hàng ACTIVE mỗi nhóm tuổi
active_counts = (
    cust_details[cust_details['club_member_status'] == 'ACTIVE']
    .groupby('age_groups')
    .agg(active_customers=('customer_id', 'count'))
    .reset_index()
)

# Gộp lại
age_summary = pd.merge(age_counts, active_counts, on='age_groups', how='left')
age_summary['active_customers'] = age_summary['active_customers'].fillna(0).astype(int)

age_summary_melted = age_summary.melt(
    id_vars='age_groups',
    value_vars=['total_customers', 'active_customers'],
    var_name='Type', value_name='Count'
)


plt.figure(figsize=(12, 6))
g = sns.barplot(data=age_summary_melted, x='age_groups', y='Count', hue='Type',
                palette={'total_customers': '#4c72b0', 'active_customers': '#55a868'}, )

plt.title("Customer Count and ACTIVE Count by Age Group", fontsize=18, fontweight='bold')
plt.xlabel("Age Group", fontsize=14, fontweight='bold')
plt.ylabel("Number of Customers", fontsize=14, fontweight='bold')
plt.legend(title='Customer Type')
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Thêm số lượng trên đầu cột
for container in g.containers:
    g.bar_label(container, fmt='%.0f', fontsize=10, padding=3)

plt.tight_layout()
plt.show()


plt.figure(figsize=(8,5))
plt.title("Purchased quantity by age group\n", fontweight="bold", size=18)
g = sns.barplot(x="age_groups", y="Purchased Quantity(%)", data=cust_details.groupby("age_groups")["article_id"].sum() \
            .transform(lambda x: (x / x.sum() * 100)).rename('Purchased Quantity(%)').reset_index(), palette="Blues_r", edgecolor="black")
plt.xlabel("Age Group",fontweight="bold", size=14)
plt.ylabel("Purchased Quantity (%)",fontweight="bold", size=14)
for container in g.containers:
    g.bar_label(container, padding = 5, fmt='%.1f', fontsize=10, color="black")
plt.grid(axis="y",color = 'grey', linestyle = '--', linewidth = 0.5)
plt.show()


plt.figure(figsize=(8,5))
plt.title("Company Earnings by age group\n", fontweight="bold", size=18)
g = sns.barplot(x="age_groups", y="earning(%)", data=cust_details.groupby("age_groups")["price"].sum() \
            .transform(lambda x: (x / x.sum() * 100)).rename('earning(%)').reset_index(), palette="Blues_r",edgecolor="black")
plt.xlabel("Age Group",fontweight="bold", size=14)
plt.ylabel("Earnings (%)",fontweight="bold", size=14)
for container in g.containers:
    g.bar_label(container, padding = 5, fmt='%.1f', fontsize=10, color="black")
plt.grid(axis="y",color = 'grey', linestyle = '--', linewidth = 0.5)
plt.show()


# So sánh số lượng mua giữa các nhóm theo dõi/thường xuyên/thỉnh thoảng/không theo dõi.
plt.figure(figsize=(9,5))
plt.title("Purchased quantity by Fashion News Frequency by Age Group\n", fontweight="bold", size=18)
g = sns.barplot(x="fashion_news_frequency", y="Purchased Quantity(%)", data=cust_details.groupby("fashion_news_frequency")["article_id"].sum() \
            .transform(lambda x: (x / x.sum() * 100)).rename('Purchased Quantity(%)').reset_index(), palette={'Monthly' : 'gray', 'NONE': '#4c72b0', 'Regularly': '#55a868'}, edgecolor="black")
plt.xlabel("Fashion News Frequency",fontweight="bold", size=14)
plt.ylabel("Purchased Quantity (%)",fontweight="bold", size=14)
for container in g.containers:
    g.bar_label(container, padding = 5, fmt='%.3f', fontsize=10, color="black")
plt.grid(axis="y",color = 'grey', linestyle = '--', linewidth = 0.5)
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
    palette={'Regularly': '#55a868', 'NONE': '#4c72b0'}
)

plt.title("Purchased Quantity (%) by Fashion News Frequency & Age Group", fontsize=18, fontweight='bold')
plt.xlabel("Age Group", fontsize=14, fontweight='bold')
plt.ylabel("Purchased Quantity (%)", fontsize=14, fontweight='bold')
plt.legend(title='Fashion News Frequency', fontsize=12, title_fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# ✅ Thêm số % trên từng cột
for container in g.containers:
    g.bar_label(container, fmt='%.1f%%', fontsize=10, padding=3, color='black')

plt.show()


# Tỉ lệ người theo dõi theo từng nhóm tuổi
x, y = 'age_groups', 'fashion_news_frequency'
df_age_news = cust_details.groupby(x)[y].value_counts(normalize=True)
df_age_news = df_age_news.mul(100)
df_age_news = df_age_news.rename('percent(%)').reset_index()
df_age_news = df_age_news[df_age_news["fashion_news_frequency"].isin(["Regularly","NONE"])]


plt.figure(figsize=(13,6))
plt.title("Fashion News Frequency by age group\n",fontweight="bold", size=18)
g=sns.barplot(x="age_groups", y="percent(%)",data=df_age_news, hue="fashion_news_frequency", palette={'Regularly': '#55a868', 'NONE': '#4c72b0'})
plt.xlabel("Age group",fontweight="bold", size=14)
plt.ylabel("Percentage (%)",fontweight="bold", size=14)
for container in g.containers:
    g.bar_label(container, padding = 5, fmt='%.1f%%', fontsize=10, color="black")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='News Frequency',bbox_to_anchor=(1.0, 1.0), ncol=1, fancybox=True, shadow=True, fontsize=12,title_fontsize=13)
plt.show()


cust_details["club_member_status"].value_counts(normalize=True)


# Trung bình số hàng mua cho từng nhóm: ACTIVE, LEFT CLUB, PRE-CREATE
cust_details.groupby("club_member_status")["article_id"].sum()


print("The average quantity of purchased products by the customers is {:.0f} products ".format(cust_details["article_id"].mean()))


print("The average quantity of purchased products by the ACTIVE customers is {:.0f} products ".format(cust_details.groupby("club_member_status")["article_id"].mean()["ACTIVE"]))
print("The average quantity of purchased products by the LEFT-CLUB customers is {:.0f} products ".format(cust_details.groupby("club_member_status")["article_id"].mean()["LEFT CLUB"]))
print("The average quantity of purchased products by the PRE-CREATE customers is {:.0f} products ".format(cust_details.groupby("club_member_status")["article_id"].mean()["PRE-CREATE"]))


plt.figure(figsize=(9,5))
plt.title("Average Purchased Quantity by Club Member Status\n", fontweight="bold", size=18)
g = sns.barplot(x="club_member_status", y="article_id", data=cust_details.groupby("club_member_status")["article_id"].mean().astype(int).reset_index(), palette="Blues_r", edgecolor="black")
plt.axhline(y = cust_details["article_id"].mean(), color = 'r', linestyle = '-', linewidth = 0.5)
plt.text(0.76, 23.7, 'Mean Purchased Quantity: {:.0f}'.format(cust_details["article_id"].mean()), size=10, color="red",fontweight="bold")
plt.xlabel("Club Member Status",fontweight="bold", size=14)
plt.ylabel("Average Purchased Quantity",fontweight="bold", size=14)
for container in g.containers:
    g.bar_label(container, padding = 5, fmt='%.0f', fontsize=10, color="black")
plt.grid(axis="y",color = 'grey', linestyle = '--', linewidth = 0.5)
plt.show()


plt.figure(figsize=(9,5))
plt.title("Median Purchased Quantity by Club Member Status\n", fontweight="bold", size=18)
g = sns.barplot(x="club_member_status", y="article_id", data=cust_details.groupby("club_member_status")["article_id"].median().reset_index(), palette="Blues_r", edgecolor="black")
plt.axhline(y = cust_details["article_id"].median(), color = 'r', linestyle = '-', linewidth = 0.5)
plt.text(0.76, 9.3, 'Median Purchased Quantity: {:.2f}'.format(cust_details["article_id"].median()), size=10, color="red",fontweight="bold")
plt.xlabel("Club Member Status",fontweight="bold", size=14)
plt.ylabel("Median Purchaed Quantity",fontweight="bold", size=14)
for container in g.containers:
    g.bar_label(container, padding = 5, fmt='%.0f', fontsize=10, color="black")
plt.grid(axis="y",color = 'grey', linestyle = '--', linewidth = 0.5)
plt.show()


# Đếm số lượng theo index_name
index_counts = df_a['index_name'].value_counts().reset_index()
index_counts.columns = ['index_name', 'count']

# Vẽ biểu đồ ngang
plt.figure(figsize=(15, 7))
ax = sns.barplot(data=index_counts, y='index_name', x='count', palette='Blues_r')

# Tiêu đề & nhãn trục
plt.title('Number of Articles by Index Name', fontsize=18, fontweight='bold')
ax.set_xlabel('Count by Index Name', fontsize=12)
ax.set_ylabel('Index Name', fontsize=12)

# Hiện số lượng ở đầu bên phải mỗi thanh
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', label_type='edge', fontsize=10, padding=5)

plt.tight_layout()
plt.show()


# Baby/Children     4  
# Ladieswear        1
# Divided           2
# Menswear          3
# Sport             26

# Merge thêm index_group_no vào giao dịch
df_t = df_t.merge(df_a[['article_id', 'index_group_no']], on='article_id', how='left')

# Giữ lại chỉ sản phẩm thuộc nhóm nam/nữ (Menswear: 3, Ladieswear: 1)
trans_gender = df_t[df_t['index_group_no'].isin([1, 3])]

# Gom nhóm và gán nhóm phổ biến nhất
from collections import Counter

customer_index_group = trans_gender.groupby('customer_id')['index_group_no'].agg(list).reset_index()
customer_index_group['gender_calc'] = customer_index_group['index_group_no'].apply(lambda x: Counter(x).most_common(1)[0][0])

# Gộp vào bảng khách hàng
df_c = df_c.merge(customer_index_group[['customer_id', 'gender_calc']], on='customer_id', how='left')
df_c['gender_calc'] = df_c['gender_calc'].fillna(0).astype('int8')  # 0: không xác định


# Đếm số lượng theo giới tính
gender_counts = df_c['gender_calc'].value_counts().sort_index()
gender_labels = ['Unidentified', 'Female (1)', 'Male (3)']

# Chuẩn bị dữ liệu phần trăm
gender_percent = gender_counts / gender_counts.sum()

# Vẽ biểu đồ tròn
plt.figure(figsize=(6, 6))
plt.pie(
    gender_percent.values,
    labels=gender_labels[:len(gender_percent)],
    autopct='%1.1f%%',
    startangle=90,
    colors=['lightgrey', '#FFB6C1', '#87CEFA'],  # xám, hồng, xanh
    wedgeprops=dict(edgecolor='k')
)

plt.title("Customer Gender Distribution (Predicted)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


df_t = df_t.merge(df_a[['article_id', 'index_group_name']], on='article_id', how='left')
df_t = df_t.merge(cust_details[['customer_id', 'age_groups']], on='customer_id', how='left')

# Tạo bảng tổng hợp số lượng sản phẩm mua theo nhóm tuổi và nhóm sản phẩm
age_group_pref = (
    df_t.groupby(['age_groups', 'index_group_name'])['article_id']
    .count()
    .reset_index()
    .rename(columns={'article_id': 'purchased_count'})
)
plt.figure(figsize=(16, 7))
sns.barplot(
    data=age_group_pref,
    x='age_groups',
    y='purchased_count',
    hue='index_group_name',
    palette='tab10'
)

plt.title("Purchase Behavior by Age Group and Product Category", fontsize=18, fontweight='bold')
plt.xlabel("Age Group", fontsize=14)
plt.ylabel("Number of Items Purchased", fontsize=14)
plt.legend(title='Product Category', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


df_t = df_t.merge(df_a[['article_id', 'prod_name', 'colour_group_name']], on='article_id', how='left')

# Đếm tổng số sản phẩm theo màu đã mua
color_counts = (
    df_t.groupby('colour_group_name')['article_id']
    .count()
    .reset_index()
    .rename(columns={'article_id': 'count'})
    .sort_values(by='count', ascending=False)
)


color_map = {
    'Black': '#000000',
    'White': '#FFFFFF',
    'Dark Blue': '#00008B',
    'Light Beige': '#D8CAB8',
    'Blue': '#0000FF',
    'Beige': '#F5F5DC',
    'Light Blue': '#ADD8E6',
    'Light Pink': '#FFB6C1',
    'Off White': '#F8F8FF',
    'Grey': '#808080'
}


# Lấy top 10 màu phổ biến
top_colors = color_counts.head(10).copy()

# Tạo danh sách mã màu đúng theo thứ tự
top_colors_palette = [color_map.get(c, 'lightgrey') for c in top_colors['colour_group_name']]

# Vẽ biểu đồ
plt.figure(figsize=(14, 6))
sns.barplot(data=top_colors, x='colour_group_name', y='count', palette=top_colors_palette, edgecolor="black")

plt.title("Top 10 Most Frequently Purchased Colors", fontsize=16, fontweight='bold')
plt.xlabel("Color Group")
plt.ylabel("Purchase Count")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



# Tập hợp các màu một khách hàng từng mua
color_combinations = (
    df_t.groupby('customer_id')['colour_group_name']
    .apply(lambda x: list(set(x.dropna())))
    .apply(lambda x: list(combinations(sorted(x), 2)))  # tạo cặp tổ hợp màu
)

# Đếm tổ hợp màu phổ biến
color_pair_counter = Counter([pair for sublist in color_combinations for pair in sublist])
common_color_pairs = pd.DataFrame(color_pair_counter.most_common(10), columns=['Color Pair', 'Count'])

# Hiển thị
print(common_color_pairs)


# Tạo dữ liệu
pairs = common_color_pairs.copy()
pairs['Color A'] = pairs['Color Pair'].apply(lambda x: x[0])
pairs['Color B'] = pairs['Color Pair'].apply(lambda x: x[1])
pairs['Label'] = pairs['Color A'] + ' & ' + pairs['Color B']

# Vẽ dotplot
fig, ax = plt.subplots(figsize=(10, 6))

for i, row in pairs.iterrows():
    y = len(pairs) - 1 - i  # từ trên xuống
    # Vẽ 2 dấu chấm
    ax.scatter(0.5, y, s=500, color=color_map.get(row['Color A'], 'gray'), edgecolor='k')
    ax.scatter(1.5, y, s=500, color=color_map.get(row['Color B'], 'gray'), edgecolor='k')
    # Ghi số lượng
    ax.text(2.1, y, f"{row['Count']:,}", va='center', fontsize=11)

# Cấu hình trục
ax.set_yticks(range(len(pairs))[::-1])
ax.set_yticklabels(pairs['Label'])
ax.set_xticks([0.5, 1.5])
ax.set_xticklabels(['Color A', 'Color B'])
ax.set_xlim(0, 2.5)
ax.set_title("Top 10 Most Frequently Co-Purchased Color Pairs", fontsize=16, fontweight='bold')
ax.set_xlabel("Colors in Pair")
ax.set_ylabel("Color Pair")
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


# Mỗi giao dịch là tổ hợp: customer_id + t_dat + article_id
df_t['cnt_articles'] = df_t.groupby(['customer_id', 't_dat', 'article_id'])['article_id'].transform('count')
# Loại bỏ trùng dòng, chỉ lấy 1 dòng cho mỗi giao dịch sản phẩm
article_freq = (
    df_t[['customer_id', 't_dat', 'article_id', 'cnt_articles']]
    .drop_duplicates()
    .groupby('article_id')['cnt_articles']
    .sum()
    .reset_index()
    .sort_values(by='cnt_articles', ascending=False)
)
top_articles = article_freq.merge(df_a[['article_id', 'prod_name']], on='article_id', how='left')
top_articles = top_articles.drop_duplicates(subset='article_id').head(10)



plt.figure(figsize=(12, 6))
ax = sns.barplot(
    data=top_articles,
    y='prod_name',
    x='cnt_articles',
    palette='Blues_r',
    ci=None 
)
plt.title("Top 10 Most Frequently Bulk-Purchased Products", fontsize=16, fontweight='bold')
plt.xlabel("Total Quantity Purchased")
plt.ylabel("Product Name")
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', fontsize=10, padding=3)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

