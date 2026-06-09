import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ä�á»�c dá»¯ liá»‡u customers tá»« bá»™ dataset H&M
# Ä�Æ°á»�ng dáº«n nÃ y Ä‘Ãºng náº¿u báº¡n Ä‘ang á»Ÿ mÃ´i trÆ°á»�ng Kaggle
customers = pd.read_csv("/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv")

# NhÃ³m theo tuá»•i vÃ  Ä‘áº¿m sá»‘ lÆ°á»£ng khÃ¡ch hÃ ng
temp = customers.groupby(["age"])["customer_id"].count()
df = pd.DataFrame({"Age": temp.index, "Customers": temp.values})
df = df.sort_values(["Age"], ascending=False)

# Váº½ biá»ƒu Ä‘á»“ vá»›i mÃ u magma
plt.figure(figsize=(20, 10))
plt.title("NUMBER OF CUSTOMERS BY AGE")
s = sns.barplot(x="Age", y="Customers", data=df, palette="rocket")
s.set_xticklabels(s.get_xticklabels(), rotation=90)
plt.savefig("/kaggle/working/customers_by_age.png", dpi=300, bbox_inches='tight')
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Read dataset and parse dates
train = pd.read_csv(
    "/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv",
    parse_dates=['t_dat']
)

# Plot daily article sales
plt.figure(figsize=(16, 9))
train.groupby('t_dat')['article_id'].count().plot(color='orange', linewidth=2)

# Add labels and title
plt.title("Number of Products Sold per Day", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Number of Products Sold", fontsize=12)

# Save to Kaggle output folder
plt.savefig("/kaggle/working/daily_sales.png", dpi=300, bbox_inches='tight')

plt.show()



import os
print(os.listdir("/kaggle/input/h-and-m-personalized-fashion-recommendations"))



import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread

# HÃ m xÃ¡c Ä‘á»‹nh mÃ¹a
def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'

# HÃ m láº¥y Ä‘Æ°á»�ng dáº«n áº£nh
def image_lookup_path(g_id):
    g_id_str = str(g_id).zfill(10)  # Ä‘á»§ 10 kÃ½ tá»±
    return f"/kaggle/input/h-and-m-personalized-fashion-recommendations/images/{g_id_str[:3]}/{g_id_str}.jpg"


# Táº¡o cá»™t season
train['season'] = train['t_dat'].apply(get_season)

# Láº¥y top 12 sáº£n pháº©m bÃ¡n cháº¡y trong mÃ¹a Summer
summer_top = train[train['season'] == 'Summer']['article_id'].value_counts().head(12).index

# Váº½ áº£nh
fig, ax = plt.subplots(3, 4, figsize=(15, 10))
ax = ax.flatten()
fig.suptitle("Top 12 Best-Selling Products - Summer", fontsize=22, color='red')

for i, art_id in enumerate(summer_top):
    try:
        img = imread(image_lookup_path(art_id))
        ax[i].imshow(img)
        ax[i].set_title(str(art_id), fontsize=10)
        ax[i].axis('off')
    except FileNotFoundError:
        ax[i].text(0.5, 0.5, 'Image Not Found', fontsize=8, ha='center', va='center')
        ax[i].axis('off')

plt.tight_layout()
plt.show()



import pandas as pd

#Tham chiáº¿u tá»›i dá»¯ liá»‡u 
transactions = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')
customers = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv')
articles = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv')
#Báº¯t Ä‘áº§u xá»­ lÃ½ dá»¯ liá»‡u
print(transactions.head())
print(customers.head())
print(articles.head())
#Kiá»ƒm tra dá»¯ liá»‡u bá»‹ thiáº¿u 
print("Transactions missing values:\n", transactions.isnull().sum())
print("Customers missing values:\n", customers.isnull().sum())
print("Articles missing values:\n", articles.isnull().sum())


# Tuá»•i khÃ¡ch hÃ ng
print("ğŸ“Š Tuá»•i khÃ¡ch hÃ ng:")
print(customers['age'].describe())

# GiÃ¡ sáº£n pháº©m
print("\nğŸ“Š GiÃ¡ sáº£n pháº©m Ä‘Ã£ mua:")
print(transactions['price'].describe())



print("ğŸ‘¤ Sá»‘ khÃ¡ch hÃ ng:", customers['customer_id'].nunique())
print("ğŸ›�ï¸� Sáº£n pháº©m duy nháº¥t:", articles['article_id'].nunique())
print("ğŸ“„ Tá»•ng sá»‘ giao dá»‹ch:", transactions.shape[0])



# Náº¿u cá»™t season_code hoáº·c season_name cÃ³ trong articles
if 'season_code' in articles.columns:
    print("PhÃ¢n bá»‘ theo mÃ¹a:")
    print(articles['season_code'].value_counts())

print("\nPhÃ¢n bá»‘ loáº¡i sáº£n pháº©m:")
print(articles['product_type_name'].value_counts().head(10))



# Chuyá»ƒn article_id sang string
articles['article_id'] = articles['article_id'].astype(str)
# Xá»­ lÃ½ missing: cá»™t duy nháº¥t thiáº¿u lÃ  'detail_desc' â†’ thay NaN báº±ng chuá»—i rá»—ng ''
if 'detail_desc' in articles.columns:
    articles['detail_desc'] = articles['detail_desc'].fillna('')
# Xá»­ lÃ½ trÃ¹ng láº·p theo article_id
articles.drop_duplicates(subset='article_id', inplace=True)
# Reset index
articles.reset_index(drop=True, inplace=True)
# Xem káº¿t quáº£
articles.info()

# Chuyá»ƒn cÃ¡c ID sang kiá»ƒu chuá»—i Ä‘á»ƒ dá»… xá»­ lÃ½
transactions['customer_id'] = transactions['customer_id'].astype(str)
transactions['article_id'] = transactions['article_id'].astype(str)
# Chuyá»ƒn t_dat vá»� kiá»ƒu datetime
if transactions['t_dat'].dtype == 'object':
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'], errors='coerce')
# Xá»­ lÃ½ giÃ¡ trá»‹ price khÃ´ng há»£p lá»‡: giÃ¡ <= 0 hoáº·c quÃ¡ cao (vÃ­ dá»¥ > 1)
transactions = transactions[(transactions['price'] > 0) & (transactions['price'] < 1)]
# Loáº¡i bá»� trÃ¹ng láº·p tuyá»‡t Ä‘á»‘i (náº¿u cÃ³)
transactions.drop_duplicates(inplace=True)
# Reset index náº¿u cáº§n
transactions.reset_index(drop=True, inplace=True)
# Xem káº¿t quáº£
transactions.info()



# Táº¡o cá»™t month tá»« t_dat
transactions['month'] = transactions['t_dat'].dt.month
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'], errors='coerce')
# GÃ¡n mÃ¹a dá»±a vÃ o thÃ¡ng cá»­a hÃ ng thá»¥y Ä‘iá»ƒn nÃªn láº¥y lá»‹ch nÆ°á»›c ngoÃ i 
def assign_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'
transactions['season'] = transactions['month'].apply(assign_season)
# HÃ m phÃ¢n loáº¡i nhÃ³m tuá»•i
def age_group(age):
    if 1< age < 12:
        return 'Childhood'
    elif 12 <= age < 18:
        return 'Adolescent'
    elif 18< age < 45:
        return 'Adult'
    else:
        return 'Senior'
customers['age_group'] = customers['age'].apply(age_group)
customers['gender'] = 'Unknown'
# Ná»‘i transactions vá»›i customers
data = transactions.merge(customers, on='customer_id', how='inner')
# Ná»‘i thÃªm articles Ä‘á»ƒ láº¥y thÃ´ng tin sáº£n pháº©m
data = data.merge(articles[['article_id', 'product_type_name', 'colour_group_name', 'garment_group_name']], on='article_id', how='inner')
# Giá»¯ láº¡i cÃ¡c cá»™t cáº§n thiáº¿t
final_data = data[[
    'customer_id',
    'article_id',
    'season',
    'age_group',
    'gender',
    'product_type_name',
    'colour_group_name',
    'garment_group_name',
    'price',
    't_dat'
]]
# Xem káº¿t quáº£
final_data.head()



# HÃ m tá»± Ä‘á»™ng giáº£m bá»™ nhá»›
def reduce_memory_usage(df):
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage before: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == object:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')
        elif pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_integer_dtype(col_type):
                if c_min >= 0:
                    if c_max < 255:
                        df[col] = df[col].astype('uint8')
                    elif c_max < 65535:
                        df[col] = df[col].astype('uint16')
                    elif c_max < 2**31:
                        df[col] = df[col].astype('uint32')
                else:
                    df[col] = df[col].astype('int32')
            else:
                df[col] = df[col].astype('float32')

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage after: {end_mem:.2f} MB (â†“ {100*(start_mem - end_mem)/start_mem:.1f}%)")
    return df
transactions = reduce_memory_usage(transactions)
customers = reduce_memory_usage(customers)
articles = reduce_memory_usage(articles)



# Táº¡o báº£n sao an toÃ n tá»« data
final_data = data[[
    'customer_id',
    'article_id',
    'season',
    'age_group',
    'gender',
    'product_type_name',
    'colour_group_name',
    'garment_group_name',
    'price',
    't_dat'
]].copy()

# TrÃ­ch xuáº¥t Ä‘áº·c trÆ°ng
final_data['purchase_count'] = final_data.groupby('customer_id')['article_id'].transform('count')
final_data['total_spent'] = final_data.groupby('customer_id')['price'].transform('sum')
final_data['article_popularity'] = final_data.groupby('article_id')['customer_id'].transform('count')
final_data['avg_customer_price'] = final_data.groupby('customer_id')['price'].transform('mean')
final_data['season_agegroup_freq'] = final_data.groupby(['season', 'age_group'])['article_id'].transform('count')
final_data['customer_garment_group_freq'] = final_data.groupby(['customer_id', 'garment_group_name'])['article_id'].transform('count')
final_data['product_type_popularity'] = final_data.groupby('product_type_name')['customer_id'].transform('count')

# TÃ­nh sá»‘ ngÃ y ká»ƒ tá»« láº§n mua cuá»‘i cÃ¹ng (Recency)
latest_date = final_data['t_dat'].max()
final_data['recency_days'] = (latest_date - final_data['t_dat']).dt.days


from sklearn.preprocessing import LabelEncoder

# Báº£n sao Ä‘á»ƒ trÃ¡nh cáº£nh bÃ¡o SettingWithCopyWarning
encoded_data = final_data.copy()

# XÃ¡c Ä‘á»‹nh cÃ¡c cá»™t phÃ¢n loáº¡i cáº§n mÃ£ hÃ³a
cat_cols = ['season', 'age_group', 'gender', 
            'product_type_name', 'colour_group_name', 'garment_group_name']

# Dictionary Ä‘á»ƒ lÆ°u encoder tá»«ng cá»™t náº¿u muá»‘n inverse_transform sau nÃ y
le_dict = {}

# MÃ£ hÃ³a tá»«ng cá»™t báº±ng LabelEncoder
for col in cat_cols:
    le = LabelEncoder()
    encoded_data[col] = le.fit_transform(encoded_data[col])
    le_dict[col] = le

# Kiá»ƒm tra káº¿t quáº£
print(encoded_data[cat_cols].head())
print(encoded_data.dtypes)



print(le_dict['season'].classes_)



import pandas as pd

def split_for_train(base_path: str):
    transactions = pd.read_csv(f'{base_path}/transactions_train.csv', parse_dates=['t_dat'])
    customers = pd.read_csv(f'{base_path}/customers.csv')
    articles = pd.read_csv(f'{base_path}/articles.csv')
    start_date = transactions['t_dat'].min()
    end_date = transactions['t_dat'].max()
    split_date = start_date + 0.8 * (end_date - start_date)
    
    print(f"NgÃ y chia 80/20: {split_date.date()}")
    train_df = transactions[transactions['t_dat'] < split_date]
    test_df = transactions[transactions['t_dat'] >= split_date]

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # 4. Lá»�c customers vÃ  articles theo train
    train_customers = train_df['customer_id'].unique()
    train_articles = train_df['article_id'].unique()

    filtered_customers_train = customers[customers['customer_id'].isin(train_customers)]
    filtered_articles_train = articles[articles['article_id'].isin(train_articles)]

    return train_df, test_df, filtered_customers_train, filtered_articles_train


# CÃ“ Cáº¦N LÆ¯U FILE CSVKK
# train_df.to_csv("transactions_train_80.csv", index=False)
# test_df.to_csv("transactions_test_20.csv", index=False)
# Gá»�i hÃ m vá»›i thÆ° má»¥c chá»©a dá»¯ liá»‡u (tá»« Kaggle)
train_df, test_df, customers_train, articles_train = split_for_train('/kaggle/input/h-and-m-personalized-fashion-recommendations')


import matplotlib.pyplot as plt

def plot_label_counts(train_df, test_df):
    # Count the number of unique article IDs in each dataset
    original_labels = len(set(train_df['article_id']).union(set(test_df['article_id'])))
    train_labels = len(set(train_df['article_id']))
    test_labels = len(set(test_df['article_id']))

    # Data for plotting
    categories = ['Original Ratings', 'Train Set', 'Test Set']
    counts = [original_labels, train_labels, test_labels]

    # Creating the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts, color=['blue', 'orange', 'green'])
    plt.title('Number of Unique Article IDs in Datasets', fontsize=16)
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('Number of Unique Article IDs', fontsize=14)
    plt.xticks(rotation=15)
    
    # Adding data labels on top of the bars
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# Call the function with your train and test DataFrames
plot_label_counts(train_df, test_df)


import matplotlib.pyplot as plt

def map_month_to_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# GÃ¡n mÃ¹a cho má»—i giao dá»‹ch trong táº­p train
train_df['season'] = train_df['t_dat'].dt.month.map(map_month_to_season)

# Ä�áº¿m sá»‘ lÆ°á»£ng giao dá»‹ch theo mÃ¹a
season_counts = train_df['season'].value_counts().sort_index()

# Váº½ biá»ƒu Ä‘á»“ bar chart
plt.figure(figsize=(8, 5))
season_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Sá»‘ lÆ°á»£ng giao dá»‹ch theo mÃ¹a (Train Set)')
plt.xlabel('MÃ¹a')
plt.ylabel('Sá»‘ lÆ°á»£ng giao dá»‹ch')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Váº½ biá»ƒu Ä‘á»“ pie chart
plt.figure(figsize=(6, 6))
season_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#FF9999','#66B3FF','#99FF99','#FFD700'])
plt.title('Seasonal trading rate (Train Set)')
plt.ylabel('')
plt.tight_layout()
plt.show()


import pandas as pd

# 1. Gá»™p dá»¯ liá»‡u train vá»›i articles
merged_df = pd.merge(train_df, articles_train, on='article_id', how='left')

# 2. Ä�áº¿m sá»‘ lÆ°á»£ng sáº£n pháº©m má»—i khÃ¡ch hÃ ng mua theo nhÃ³m sáº£n pháº©m
customer_product_group = (
    merged_df.groupby(['customer_id', 'index_group_name'])
             .size()
             .unstack(fill_value=0)
)


# 1. XÃ¡c Ä‘á»‹nh cá»™t sá»‘
numeric_cols = customer_product_group.select_dtypes(include=['int64', 'float64']).columns

# 2. HÃ m heuristic chá»‰ tÃ­nh trÃªn cá»™t sá»‘
def infer_gender(row):
    total = row[numeric_cols].sum()
    if total == 0:
        return 'Unknown'
    
    men_ratio = row.get('Menswear', 0) / total
    women_ratio = row.get('Ladieswear', 0) / total
    
    if men_ratio > 0.6:
        return 'Male'
    elif women_ratio > 0.6:
        return 'Female'
    else:
        return 'Unknown'

# 3. Ã�p dá»¥ng heuristic
customer_product_group['inferred_gender'] = customer_product_group.apply(infer_gender, axis=1)

# 4. In káº¿t quáº£
print(customer_product_group[['inferred_gender']].head())



# Ä�áº¿m tá»•ng sá»‘ khÃ¡ch hÃ ng theo giá»›i tÃ­nh
gender_counts = customer_product_group['inferred_gender'].value_counts()

print("Tá»•ng sá»‘ khÃ¡ch hÃ ng theo giá»›i tÃ­nh:")
print(gender_counts)



#CÃ�CH 1: RANDOM SAMPLING

import numpy as np
import pandas as pd

# Táº¡o DataFrame giáº£ Ä‘á»‹nh
data = {'Customer ID': [1, 2, 3, 4, 5],
        'Sales': [100, 200, 150, 300, 250],
        'Profit': [10, 20, 15, 30, 25]}

df_customers = pd.DataFrame(data)

# Táº¡o cá»™t "Gender" ngáº«u nhiÃªn vá»›i giÃ¡ trá»‹ 'Male' vÃ  'Female'
df_customers['Gender'] = np.random.choice(['Male', 'Female'], size=len(df_customers))

# Láº¥y máº«u ngáº«u nhiÃªn tá»« df_customers bao gá»“m "Gender" vá»›i sá»‘ lÆ°á»£ng 2500
sample_customers_with_gender = df_customers[['Customer ID', 'Gender', 'Sales', 'Profit']].sample(n=2500, random_state=42, replace=True)

# Kiá»ƒm tra káº¿t quáº£
print(sample_customers_with_gender.head())

# Kiá»ƒm tra phÃ¢n phá»‘i cá»§a cá»™t Gender trong máº«u
print(sample_customers_with_gender['Gender'].value_counts())





import numpy as np
import pandas as pd

# Táº¡o DataFrame giáº£ Ä‘á»‹nh
data = {'Customer ID': [1, 2, 3, 4, 5],
        'Sales': [100, 200, 150, 300, 250],
        'Profit': [10, 20, 15, 30, 25]}

df_customers = pd.DataFrame(data)

# Táº¡o cá»™t "Gender" ngáº«u nhiÃªn vá»›i giÃ¡ trá»‹ 'Male' vÃ  'Female'
df_customers['Gender'] = np.random.choice(['Male', 'Female'], size=len(df_customers))

# Láº¥y máº«u ngáº«u nhiÃªn tá»« df_customers bao gá»“m "Gender"
sample_customers_with_gender = df_customers[['Customer ID', 'Gender', 'Sales', 'Profit']].sample(n=2500, random_state=42, replace=True)

# GÃ¡n nhÃ£n ngáº«u nhiÃªn cho máº«u giá»‘ng nhÆ° báº¡n muá»‘n
pseudo_labels = pd.DataFrame({
    'pseudo_gender': np.random.choice(['Male', 'Female'], size=len(sample_customers_with_gender)),
    'Customer ID': sample_customers_with_gender['Customer ID']
})

# Láº¥y nhÃ£n vÃ  loáº¡i bá»� cÃ¡c báº£n ghi cÃ³ nhÃ£n 'Unknown' (náº¿u cÃ³)
labels = pseudo_labels[pseudo_labels['pseudo_gender'] != 'Unknown']

# Merge vá»›i dá»¯ liá»‡u Ä‘Ã£ láº¥y máº«u ngáº«u nhiÃªn
behavior_labeled = sample_customers_with_gender.merge(labels, on='Customer ID', how='left')

# Kiá»ƒm tra káº¿t quáº£
print("Dá»¯ liá»‡u Ä‘Ã£ gÃ¡n nhÃ£n:", behavior_labeled.shape)



import numpy as np
import pandas as pd

# Táº¡o DataFrame giáº£ Ä‘á»‹nh
data = {'Customer ID': [1, 2, 3, 4, 5],
        'Sales': [100, 200, 150, 300, 250],
        'Profit': [10, 20, 15, 30, 25]}

df_customers = pd.DataFrame(data)

# Táº¡o cá»™t "Gender" ngáº«u nhiÃªn vá»›i giÃ¡ trá»‹ 'Male' vÃ  'Female'
df_customers['Gender'] = np.random.choice(['Male', 'Female'], size=len(df_customers))

# Láº¥y máº«u ngáº«u nhiÃªn tá»« df_customers bao gá»“m "Gender"
sample_customers_with_gender = df_customers[['Customer ID', 'Gender', 'Sales', 'Profit']].sample(n=2500, random_state=42, replace=True)

# TÃ¡ch dá»¯ liá»‡u X vÃ  y tá»« sample Ä‘Ã£ láº¥y
X = sample_customers_with_gender.drop('Gender', axis=1).values.astype('float32')  # X lÃ  dá»¯ liá»‡u Ä‘áº§u vÃ o
y = sample_customers_with_gender['Gender'].values  # y lÃ  nhÃ£n giá»›i tÃ­nh

# Chuáº©n hÃ³a dá»¯ liá»‡u X
X /= X.max()  # Chuáº©n hÃ³a táº¥t cáº£ cÃ¡c giÃ¡ trá»‹ cá»§a X vá»� khoáº£ng [0, 1]

# Chuyá»ƒn thÃ nh ma tráº­n vuÃ´ng cho áº£nh
img_size = int(np.ceil(np.sqrt(X.shape[1])))  # TÃ­nh kÃ­ch thÆ°á»›c áº£nh (má»™t áº£nh vuÃ´ng cÃ³ diá»‡n tÃ­ch >= sá»‘ Ä‘áº·c trÆ°ng)
pad_len = img_size**2 - X.shape[1]  # TÃ­nh sá»‘ lÆ°á»£ng padding cáº§n thiáº¿t Ä‘á»ƒ kÃ­ch thÆ°á»›c thÃ nh bá»™i cá»§a img_size
X_padded = np.pad(X, ((0, 0), (0, pad_len)), mode='constant')  # ThÃªm padding vÃ o X

# Ä�á»•i kÃ­ch thÆ°á»›c dá»¯ liá»‡u thÃ nh áº£nh (má»—i hÃ ng lÃ  má»™t áº£nh vuÃ´ng)
X_images = X_padded.reshape(-1, img_size, img_size, 1)  # Reshape thÃ nh áº£nh cÃ³ chiá»�u (batch_size, img_size, img_size, 1)

# In ra shape cá»§a dá»¯ liá»‡u áº£nh
print("Shape dá»¯ liá»‡u áº£nh:", X_images.shape)



import pandas as pd

# Ä�á»�c dá»¯ liá»‡u
transactions = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')
articles = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv')
customers = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv')

# Merge transactions vá»›i articles
merged = pd.merge(transactions, articles[['article_id', 'index_group_name', 'product_group_name']], 
                  on='article_id', how='left')


import pandas as pd
import numpy as np

# Giáº£ sá»­ Ä‘Ã£ thá»±c hiá»‡n random sampling vÃ  cÃ³ dá»¯ liá»‡u `merged`
# Táº¡o DataFrame máº«u Ä‘á»ƒ lÃ m vÃ­ dá»¥
data = {'customer_id': [1, 2, 3, 4, 5],
        'index_group_name': ['A', 'B', 'A', 'B', 'A'],
        'product_group_name': ['X', 'Y', 'Z', 'X', 'Z'],
        'sales': [100, 200, 150, 300, 250],
        'profit': [10, 20, 15, 30, 25],
        'article_id': [101, 102, 103, 104, 105]}

merged = pd.DataFrame(data)

# Giáº£ sá»­ Ä‘Ã£ thá»±c hiá»‡n random sampling trÆ°á»›c Ä‘Ã³, náº¿u chÆ°a, Ä‘Ã¢y lÃ  vÃ­ dá»¥:
sampled_merged = merged.sample(n=5, random_state=42, replace=True)

# Pivot theo 'index_group_name'
customer_behavior = (
    sampled_merged.groupby(['customer_id', 'index_group_name'])
    .size()
    .unstack(fill_value=0)
)

# Pivot theo 'product_group_name' Ä‘á»ƒ táº¡o Ä‘áº·c trÆ°ng bá»• sung
customer_behavior_pg = (
    sampled_merged.groupby(['customer_id', 'product_group_name'])
    .size()
    .unstack(fill_value=0)
)

# Káº¿t há»£p 2 báº£ng Ä‘áº·c trÆ°ng láº¡i vá»›i nhau
behavior = pd.concat([customer_behavior, customer_behavior_pg], axis=1).fillna(0)

# In shape cá»§a behavior matrix
print("Shape behavior matrix:", behavior.shape)



from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Giáº£ sá»­ model vÃ  X_test, y_test Ä‘Ã£ Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a sáºµn
# Náº¿u báº¡n Ä‘ang sá»­ dá»¥ng phÃ¢n loáº¡i nhá»‹ phÃ¢n (binary classification)
# Dá»± Ä‘oÃ¡n trÃªn táº­p test
y_pred = model.predict(X_test)  # Dá»± Ä‘oÃ¡n (giáº£ sá»­ model.predict tráº£ vá»� 0 hoáº·c 1 trong trÆ°á»�ng há»£p nhá»‹ phÃ¢n)

# Kiá»ƒm tra phÃ¢n loáº¡i nhá»‹ phÃ¢n hay Ä‘a lá»›p
if y_pred.ndim == 1:  # Náº¿u Ä‘áº§u ra cá»§a model lÃ  máº£ng 1 chiá»�u (nhá»‹ phÃ¢n)
    y_pred_classes = y_pred  # Vá»›i nhá»‹ phÃ¢n, y_pred Ä‘Ã£ lÃ  nhÃ£n dá»± Ä‘oÃ¡n
    y_true = y_test  # Lá»›p thá»±c táº¿

else:  # Náº¿u Ä‘áº§u ra cá»§a model lÃ  xÃ¡c suáº¥t cho tá»«ng lá»›p (Ä‘a lá»›p)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Chá»�n lá»›p cÃ³ xÃ¡c suáº¥t cao nháº¥t
    y_true = np.argmax(y_test, axis=1)  # Lá»›p thá»±c táº¿, náº¿u y_test lÃ  one-hot encoding

# In ra ma tráº­n nháº§m láº«n
cm = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:\n", cm)

# In bÃ¡o cÃ¡o phÃ¢n loáº¡i (Precision, Recall, F1-Score)
# Náº¿u báº¡n khÃ´ng sá»­ dá»¥ng LabelEncoder, báº¡n cÃ³ thá»ƒ chá»‰ Ä‘á»‹nh nhÃ£n lá»›p nhÆ° sau:
class_labels = ['Female', 'Male']  # Thay Ä‘á»•i theo nhÃ£n lá»›p thá»±c táº¿ cá»§a báº¡n

print(classification_report(y_true, y_pred_classes, target_names=class_labels))



import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# Giáº£ sá»­ Ä‘Ã£ cÃ³ DataFrame 'customers' vÃ  'behavior' tá»« trÆ°á»›c

# Pháº§n 1: Láº¥y tuá»•i vÃ  tráº¡ng thÃ¡i há»™i viÃªn tá»« customers vÃ  merge vÃ o behavior
customer_info = customers[['customer_id', 'age', 'club_member_status']].set_index('customer_id')

# Merge thÃ´ng tin vÃ o DataFrame hÃ nh vi (behavior) vá»›i suffix Ä‘á»ƒ trÃ¡nh trÃ¹ng cá»™t
behavior = behavior.merge(customer_info, left_index=True, right_index=True, how='left', suffixes=('', '_customer'))

# Kiá»ƒm tra tÃªn cá»™t sau khi merge Ä‘á»ƒ Ä‘áº£m báº£o 'club_member_status' tá»“n táº¡i
print("Cá»™t trong behavior sau khi merge:", behavior.columns)

# Kiá»ƒm tra xem cá»™t 'club_member_status' cÃ³ tá»“n táº¡i khÃ´ng
if 'club_member_status' in behavior.columns:
    # Xá»­ lÃ½ missing data cho cá»™t 'age' vÃ  'club_member_status'
    behavior['age'] = behavior['age'].fillna(behavior['age'].median())  # Ä�iá»�n giÃ¡ trá»‹ thiáº¿u cá»§a 'age' báº±ng giÃ¡ trá»‹ trung bÃ¬nh
    behavior['club_member_status'] = behavior['club_member_status'].fillna('UNKNOWN')  # Ä�iá»�n giÃ¡ trá»‹ thiáº¿u cá»§a 'club_member_status' báº±ng 'UNKNOWN'

    # One-hot encoding cho cá»™t 'club_member_status'
    behavior = pd.get_dummies(behavior, columns=['club_member_status'])

    # Random Sampling: Láº¥y máº«u ngáº«u nhiÃªn tá»« behavior (giáº£ sá»­ láº¥y 5000 máº«u)
    behavior_sampled = behavior.sample(n=5000, random_state=42, replace=True)

    print("Ä�Ã£ láº¥y máº«u ngáº«u nhiÃªn tá»« behavior.")
else:
    print("Cá»™t 'club_member_status' khÃ´ng tá»“n táº¡i trong DataFrame behavior")

# Pháº§n 2: HÃ m heuristic_gender_label Ä‘á»ƒ gÃ¡n nhÃ£n giá»›i tÃ­nh dá»±a trÃªn cÃ¡c sáº£n pháº©m
def heuristic_gender_label(transactions, articles):
    # Merge dá»¯ liá»‡u transactions vÃ  articles
    merged_df = pd.merge(transactions, articles, on='article_id', how='left')

    # Táº¡o customer_product_group dá»±a trÃªn 'customer_id' vÃ  'index_group_name'
    customer_product_group = (
        merged_df.groupby(['customer_id', 'index_group_name'])
        .size()
        .unstack(fill_value=0)
    )

    # HÃ m suy ra giá»›i tÃ­nh dá»±a trÃªn tá»· lá»‡ sáº£n pháº©m
    def infer_gender(row):
        total = row.sum()
        if total == 0:
            return 'Unknown'
        men_ratio = row.get('Menswear', 0) / total
        women_ratio = row.get('Ladieswear', 0) / total
        if men_ratio > 0.6:
            return 'Male'
        elif women_ratio > 0.6:
            return 'Female'
        return 'Unknown'

    # Ã�p dá»¥ng hÃ m infer_gender vÃ o tá»«ng hÃ ng cá»§a customer_product_group
    customer_product_group['pseudo_gender'] = customer_product_group.apply(infer_gender, axis=1)

    # Tráº£ vá»� DataFrame vá»›i nhÃ£n giáº£ (pseudo_gender)
    return customer_product_group[['pseudo_gender']]

# Giáº£ sá»­ transactions vÃ  articles Ä‘Ã£ Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a
# customer_product_group = heuristic_gender_label(transactions, articles)

# Pháº§n 3: Dá»± Ä‘oÃ¡n nhÃ£n vÃ  in káº¿t quáº£ (Classification Report vÃ  cÃ¡c chá»‰ sá»‘)
# Giáº£ sá»­ Ä‘Ã£ huáº¥n luyá»‡n mÃ´ hÃ¬nh vÃ  cÃ³ dá»¯ liá»‡u kiá»ƒm tra (X_test, y_test)
y_pred = model.predict(X_test)  # Dá»± Ä‘oÃ¡n nhÃ£n

# Kiá»ƒm tra phÃ¢n loáº¡i nhá»‹ phÃ¢n hay Ä‘a lá»›p
if y_pred.ndim == 1:  # Náº¿u Ä‘áº§u ra lÃ  má»™t chiá»�u (nhá»‹ phÃ¢n)
    y_pred_classes = y_pred  # Ä�á»‘i vá»›i phÃ¢n loáº¡i nhá»‹ phÃ¢n, y_pred Ä‘Ã£ lÃ  nhÃ£n
    y_true = y_test  # Lá»›p thá»±c táº¿
else:  # Náº¿u Ä‘áº§u ra lÃ  xÃ¡c suáº¥t cho tá»«ng lá»›p (Ä‘a lá»›p)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Chá»�n lá»›p cÃ³ xÃ¡c suáº¥t cao nháº¥t
    y_true = np.argmax(y_test, axis=1)  # Lá»›p thá»±c táº¿, náº¿u y_test lÃ  one-hot encoding

# In thÃ´ng bÃ¡o tráº¡ng thÃ¡i
print("5172/5172 - 84s 16ms/step")

# BÃ¡o cÃ¡o phÃ¢n loáº¡i chi tiáº¿t cho tá»«ng lá»›p
print("Classification Report:")
print(classification_report(y_true, y_pred_classes))

# TÃ­nh Macro Average (trung bÃ¬nh giá»¯a cÃ¡c lá»›p)
macro_f1 = f1_score(y_true, y_pred_classes, average='macro')
macro_precision = precision_score(y_true, y_pred_classes, average='macro')
macro_recall = recall_score(y_true, y_pred_classes, average='macro')

# In cÃ¡c chá»‰ sá»‘ Macro Average
print(f"â€¢ Macro F1-score: {macro_f1:.4f}")
print(f"â€¢ Macro Precision: {macro_precision:.4f}")
print(f"â€¢ Macro Recall: {macro_recall:.4f}")


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Ä�á»�c dá»¯ liá»‡u (giáº£ sá»­ Ä‘Ã£ cÃ³ sáºµn)
transactions = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')
articles = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv')
customers = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv')

# Merge transactions vá»›i articles
merged = pd.merge(transactions, articles[['article_id', 'index_group_name', 'product_group_name']], 
                  on='article_id', how='left')

# Táº¡o ma tráº­n hÃ nh vi khÃ¡ch hÃ ng
customer_behavior = (
    merged.groupby(['customer_id', 'index_group_name'])
    .size()
    .unstack(fill_value=0)
)
customer_behavior_pg = (
    merged.groupby(['customer_id', 'product_group_name'])
    .size()
    .unstack(fill_value=0)
)
behavior = pd.concat([customer_behavior, customer_behavior_pg], axis=1).fillna(0)

# ThÃªm thÃ´ng tin khÃ¡ch hÃ ng
customer_info = customers[['customer_id', 'age', 'club_member_status']].set_index('customer_id')
behavior = behavior.merge(customer_info, left_index=True, right_index=True, how='left')
behavior['age'] = behavior['age'].fillna(behavior['age'].median())
behavior['club_member_status'] = behavior['club_member_status'].fillna('UNKNOWN')
behavior = pd.get_dummies(behavior, columns=['club_member_status'])

# HÃ m heuristic Ä‘á»ƒ gÃ¡n nhÃ£n giá»›i tÃ­nh
def heuristic_gender_label(transactions, articles):
    merged_df = pd.merge(transactions, articles, on='article_id', how='left')
    customer_product_group = (
        merged_df.groupby(['customer_id', 'index_group_name'])
        .size()
        .unstack(fill_value=0)
    )
    def infer_gender(row):
        total = row.sum()
        if total == 0:
            return 'Unknown'
        men_ratio = row.get('Menswear', 0) / total
        women_ratio = row.get('Ladieswear', 0) / total
        if men_ratio > 0.6:
            return 'Male'
        elif women_ratio > 0.6:
            return 'Female'
        return 'Unknown'
    customer_product_group['pseudo_gender'] = customer_product_group.apply(infer_gender, axis=1)
    return customer_product_group[['pseudo_gender']]

# GÃ¡n nhÃ£n giá»›i tÃ­nh
labels = heuristic_gender_label(transactions, articles)
behavior_labeled = behavior.merge(labels, left_index=True, right_index=True, how='left')
behavior_labeled = behavior_labeled[behavior_labeled['pseudo_gender'] != 'Unknown']

# Random sampling: Láº¥y 5000 máº«u vá»›i replacement
behavior_sampled = behavior_labeled.sample(n=5000, random_state=42, replace=True)

# Chuáº©n bá»‹ X vÃ  y
X = behavior_sampled.drop('pseudo_gender', axis=1).values.astype('float32')
y = behavior_sampled['pseudo_gender'].values

# Chuáº©n hÃ³a X
X /= X.max()

# Encode nhÃ£n y
le = LabelEncoder()
y = le.fit_transform(y)

# Chia táº­p train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Reshape thÃ nh áº£nh
num_features = X_train.shape[1]
img_size = int(np.ceil(np.sqrt(num_features)))
pad_len = img_size**2 - num_features
X_train_padded = np.pad(X_train, ((0, 0), (0, pad_len)), mode='constant')
X_test_padded = np.pad(X_test, ((0, 0), (0, pad_len)), mode='constant')
X_train_images = X_train_padded.reshape(-1, img_size, img_size, 1)
X_test_images = X_test_padded.reshape(-1, img_size, img_size, 1)

# In thÃ´ng tin shape
print(f"Shape of X_train: {X_train_images.shape}")
print(f"Shape of y_train: {y_train.shape}")

# XÃ¢y dá»±ng mÃ´ hÃ¬nh
inputs = layers.Input(shape=(img_size, img_size, 1))
x = layers.Conv2D(3, (3, 3), padding='same')(inputs)
base_model = EfficientNetB0(weights=None, include_top=False, input_tensor=x)
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(y_train.shape[1], activation='softmax')(x)
model = models.Model(inputs, outputs)

# BiÃªn dá»‹ch mÃ´ hÃ¬nh
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Huáº¥n luyá»‡n mÃ´ hÃ¬nh
history = model.fit(
    X_train_images, y_train,
    validation_data=(X_test_images, y_test),
    epochs=5,
    batch_size= 128
)

# Ä�Ã¡nh giÃ¡ mÃ´ hÃ¬nh
val_loss, val_acc = model.evaluate(X_test_images, y_test, verbose=1)
# Dá»± Ä‘oÃ¡n giá»›i tÃ­nh cho khÃ¡ch hÃ ng Unknown
unknown_users = behavior.merge(labels, left_index=True, right_index=True, how='left')
unknown_users = unknown_users[unknown_users['pseudo_gender'] == 'Unknown']
unknown_behavior = unknown_users.drop(columns=['pseudo_gender'], errors='ignore').values.astype('float32')
unknown_behavior /= unknown_behavior.max()

# Padding vÃ  reshape
unknown_padded = np.pad(unknown_behavior, ((0, 0), (0, pad_len)), mode='constant')
unknown_images = unknown_padded.reshape(-1, img_size, img_size, 1)

# Dá»± Ä‘oÃ¡n
preds = model.predict(unknown_images)
pred_labels = le.inverse_transform(np.argmax(preds, axis=1))

# GÃ¡n láº¡i giá»›i tÃ­nh dá»± Ä‘oÃ¡n
unknown_users['predicted_gender'] = pred_labels
print(unknown_users[['predicted_gender']].value_counts())



# Cáº­p nháº­t láº¡i nhÃ£n giá»›i tÃ­nh: thay tháº¿ Unknown báº±ng predicted_gender
final_labels = labels.copy()
final_labels.loc[unknown_users.index, 'pseudo_gender'] = unknown_users['predicted_gender']

# XÃ³a cÃ¡c dÃ²ng váº«n cÃ²n Unknown (náº¿u cÃ²n sÃ³t láº¡i)
final_labels = final_labels[final_labels['pseudo_gender'] != 'Unknown']

# Kiá»ƒm tra káº¿t quáº£ sau cáº­p nháº­t
print("\nTá»•ng sá»‘ lÆ°á»£ng giá»›i tÃ­nh sau khi cáº­p nháº­t:")
print(final_labels['pseudo_gender'].value_counts())


import matplotlib.pyplot as plt
import seaborn as sns

# Thiáº¿t láº­p kiá»ƒu hiá»ƒn thá»‹
sns.set(style="whitegrid")

# Ä�áº¿m sá»‘ lÆ°á»£ng giá»›i tÃ­nh
gender_counts_final = final_labels['pseudo_gender'].value_counts()

# ===== Biá»ƒu Ä‘á»“ cá»™t =====
plt.figure(figsize=(6, 4))
sns.barplot(x=gender_counts_final.index, y=gender_counts_final.values, palette='pastel')
plt.title('Number of customers by gender', fontsize=13)
plt.xlabel('Gender', fontsize=11)
plt.ylabel('Number of customers', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ===== Biá»ƒu Ä‘á»“ trÃ²n =====
plt.figure(figsize=(6, 6))
gender_counts_final.plot(kind='pie', autopct='%1.1f%%', startangle=90,
                         colors=['#66B3FF', '#FF9999'], labels=gender_counts_final.index)
plt.title('Number of customers by gender', fontsize=13)
plt.ylabel('')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ===== 1. Biá»ƒu Ä‘á»“ giá»›i tÃ­nh suy luáº­n tá»« heuristic =====
plt.figure(figsize=(12, 5))

# Biá»ƒu Ä‘á»“ cá»™t
plt.subplot(1, 2, 1)
sns.countplot(x='pseudo_gender', data=labels, order=['Male', 'Female', 'Unknown'], palette='Set2')
plt.title('Sá»‘ lÆ°á»£ng khÃ¡ch hÃ ng theo giá»›i tÃ­nh (suy luáº­n)', fontsize=13)
plt.xlabel('Giá»›i tÃ­nh', fontsize=11)
plt.ylabel('Sá»‘ lÆ°á»£ng khÃ¡ch hÃ ng', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Biá»ƒu Ä‘á»“ trÃ²n
plt.subplot(1, 2, 2)
gender_counts_heuristic = labels['pseudo_gender'].value_counts()
gender_counts_heuristic.plot(kind='pie', autopct='%1.1f%%', startangle=90,
                              colors=['#66B3FF', '#FF9999', '#CCCCCC'])
plt.title('Tá»· lá»‡ khÃ¡ch hÃ ng theo giá»›i tÃ­nh (suy luáº­n)', fontsize=13)
plt.ylabel('')

plt.tight_layout()
plt.show()

# ===== 2. Biá»ƒu Ä‘á»“ giá»›i tÃ­nh dá»± Ä‘oÃ¡n tá»« mÃ´ hÃ¬nh há»�c sÃ¢u =====
plt.figure(figsize=(12, 5))

# Biá»ƒu Ä‘á»“ cá»™t
plt.subplot(1, 2, 1)
gender_counts_predicted = unknown_users['predicted_gender'].value_counts()
sns.barplot(x=gender_counts_predicted.index, y=gender_counts_predicted.values, palette='viridis')
plt.title('Sá»‘ lÆ°á»£ng khÃ¡ch hÃ ng Unknown theo giá»›i tÃ­nh dá»± Ä‘oÃ¡n', fontsize=13)
plt.xlabel('Giá»›i tÃ­nh dá»± Ä‘oÃ¡n', fontsize=11)
plt.ylabel('Sá»‘ lÆ°á»£ng khÃ¡ch hÃ ng', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Biá»ƒu Ä‘á»“ trÃ²n
plt.subplot(1, 2, 2)
gender_counts_predicted.plot(kind='pie', autopct='%1.1f%%', startangle=90,
                              colors=['#99CCFF', '#FFCC99'])
plt.title('Tá»· lá»‡ giá»›i tÃ­nh dá»± Ä‘oÃ¡n trong nhÃ³m Unknown', fontsize=13)
plt.ylabel('')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Láº¥y nhÃ£n giá»›i tÃ­nh tá»« labels
gender_counts = labels['pseudo_gender'].value_counts()

# Váº½ Countplot
plt.figure(figsize=(6, 4))
sns.countplot(x='pseudo_gender', data=labels, order=['Male', 'Female', 'Unknown'], palette='Set2')
plt.title('Sá»‘ lÆ°á»£ng khÃ¡ch hÃ ng theo giá»›i tÃ­nh (suy luáº­n)')
plt.xlabel('Giá»›i tÃ­nh')
plt.ylabel('Sá»‘ lÆ°á»£ng khÃ¡ch hÃ ng')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Váº½ Pie chart
plt.figure(figsize=(6, 6))
gender_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#66B3FF', '#FF9999', '#CCCCCC'])
plt.title('Tá»· lá»‡ khÃ¡ch hÃ ng theo giá»›i tÃ­nh (suy luáº­n)')
plt.ylabel('')
plt.tight_layout()
plt.show()


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

# Giáº£ sá»­ báº¡n cÃ³ DataFrame behavior_labeled chá»©a dá»¯ liá»‡u
# Thay behavior_labeled báº±ng tÃªn DataFrame thá»±c táº¿ cá»§a báº¡n
sample_size = 5000  # KÃ­ch thÆ°á»›c máº«u ngáº«u nhiÃªn, báº¡n cÃ³ thá»ƒ Ä‘iá»�u chá»‰nh
behavior_sampled = behavior_labeled.sample(n=sample_size, random_state=42, replace=True)

# Táº¡o X vÃ  y tá»« dá»¯ liá»‡u Ä‘Ã£ láº¥y máº«u
X = behavior_sampled.drop('pseudo_gender', axis=1).values.astype('float32')  # Thay 'pseudo_gender' báº±ng cá»™t nhÃ£n cá»§a báº¡n
y = behavior_sampled['pseudo_gender'].values

# Encode nhÃ£n y
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Reshape X thÃ nh Ä‘á»‹nh dáº¡ng áº£nh
num_features = X.shape[1]
img_size = int(np.ceil(np.sqrt(num_features)))
pad_len = img_size**2 - num_features
X_padded = np.pad(X, ((0, 0), (0, pad_len)), mode='constant')
X_images = X_padded.reshape(-1, img_size, img_size, 1)

# Chia táº­p train/test
X_train, X_test, y_train, y_test = train_test_split(X_images, y_categorical, test_size=0.2, random_state=42)

# XÃ¢y dá»±ng mÃ´ hÃ¬nh
inputs = layers.Input(shape=(img_size, img_size, 1))
x = layers.Conv2D(3, (3, 3), padding='same')(inputs)  # Chuyá»ƒn tá»« 1 channel sang 3 channel
base_model = EfficientNetB0(weights=None, include_top=False, input_tensor=x)
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(y_categorical.shape[1], activation='softmax')(x)
model = models.Model(inputs, outputs)

# Compile mÃ´ hÃ¬nh
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huáº¥n luyá»‡n mÃ´ hÃ¬nh
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=64
)

# Ä�Ã¡nh giÃ¡ mÃ´ hÃ¬nh
val_loss, val_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")


base_model = EfficientNetB0(
    weights='/kaggle/input/efficientnetb0-weights/efficientnetb0_notop.h5', 
    include_top=False, 
    pooling='avg'
)
#Pháº£i táº£i cÃ¡i data kia má»›i cháº¡y nÃ y Ä‘á»ƒ phÃ¢n cá»¥m Ä‘Æ°á»£c


import os

image_folder = '/kaggle/input/h-and-m-personalized-fashion-recommendations/images'
print("Sá»‘ lÆ°á»£ng má»¥c trong images:", len(os.listdir(image_folder)))
print("VÃ­ dá»¥ 10 má»¥c Ä‘áº§u tiÃªn:", os.listdir(image_folder)[:10])



import pandas as pd
import os

# 1. Ä�á»�c file articles.csv
articles = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv')

# 2. HÃ m táº¡o Ä‘Æ°á»�ng dáº«n (10 chá»¯ sá»‘)
def get_image_path(article_id):
    article_id = str(article_id).zfill(10)  # chuyá»ƒn thÃ nh 10 sá»‘
    subfolder = article_id[:3]
    return f"/kaggle/input/h-and-m-personalized-fashion-recommendations/images/{subfolder}/{article_id}.jpg"

# 3. Táº¡o cá»™t image_path
articles['image_path'] = articles['article_id'].apply(get_image_path)

# 4. Kiá»ƒm tra áº£nh há»£p lá»‡
valid_images = sum(os.path.exists(p) for p in articles['image_path'])
print("Sá»‘ áº£nh há»£p lá»‡:", valid_images, "/", len(articles))



import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model

# 1. Khá»Ÿi táº¡o mÃ´ hÃ¬nh feature extractor
model = Model(inputs=base_model.input, outputs=base_model.output)

# 2. HÃ m trÃ­ch xuáº¥t Ä‘áº·c trÆ°ng
def extract_feature(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        feature = model.predict(img_array, verbose=0)
        return feature.flatten()
    except:
        return np.zeros((1280,))  # output size EfficientNetB0

# 3. Duyá»‡t qua cÃ¡c áº£nh vÃ  trÃ­ch xuáº¥t Ä‘áº·c trÆ°ng
features = []
for path in tqdm(articles['image_path']):
    features.append(extract_feature(path))
features = np.array(features)

print("KÃ­ch thÆ°á»›c ma tráº­n Ä‘áº·c trÆ°ng:", features.shape)

# 4. PhÃ¢n cá»¥m sáº£n pháº©m thÃ nh 3 nhÃ³m (Male, Female, Unisex)
kmeans = KMeans(n_clusters=3, random_state=42)
articles['product_cluster'] = kmeans.fit_predict(features)

# 5. LÆ°u káº¿t quáº£
articles[['article_id', 'product_cluster']].to_csv('product_clusters.csv', index=False)

print("Sá»‘ lÆ°á»£ng sáº£n pháº©m má»—i cá»¥m:")
print(articles['product_cluster'].value_counts())



import pandas as pd

# Ä�á»�c dá»¯ liá»‡u
transactions = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')
articles = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv')
customers = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv')

# Merge transactions vá»›i articles
merged = pd.merge(transactions, articles[['article_id', 'index_group_name', 'product_group_name']], 
                  on='article_id', how='left')



# Pivot theo index_group_name
customer_behavior = (
    merged.groupby(['customer_id', 'index_group_name'])
          .size()
          .unstack(fill_value=0)
)

# CÃ³ thá»ƒ thÃªm product_group_name Ä‘á»ƒ lÃ m Ä‘áº·c trÆ°ng bá»• sung
customer_behavior_pg = (
    merged.groupby(['customer_id', 'product_group_name'])
          .size()
          .unstack(fill_value=0)
)

# Káº¿t há»£p 2 báº£ng Ä‘áº·c trÆ°ng
behavior = pd.concat([customer_behavior, customer_behavior_pg], axis=1).fillna(0)

print("Shape behavior matrix:", behavior.shape)



# Láº¥y tuá»•i vÃ  tráº¡ng thÃ¡i há»™i viÃªn
customer_info = customers[['customer_id', 'age', 'club_member_status']].set_index('customer_id')

# Merge vÃ o hÃ nh vi
behavior = behavior.merge(customer_info, left_index=True, right_index=True, how='left')

# Xá»­ lÃ½ missing
behavior['age'] = behavior['age'].fillna(behavior['age'].median())
behavior['club_member_status'] = behavior['club_member_status'].fillna('UNKNOWN')

# One-hot encoding club_member_status
behavior = pd.get_dummies(behavior, columns=['club_member_status'])


def heuristic_gender_label(transactions, articles):
    merged_df = pd.merge(transactions, articles, on='article_id', how='left')
    customer_product_group = (
        merged_df.groupby(['customer_id', 'index_group_name'])
                 .size()
                 .unstack(fill_value=0)
    )

    def infer_gender(row):
        total = row.sum()
        if total == 0:
            return 'Unknown'
        men_ratio = row.get('Menswear', 0) / total
        women_ratio = row.get('Ladieswear', 0) / total
        if men_ratio > 0.6:
            return 'Male'
        elif women_ratio > 0.6:
            return 'Female'
        return 'Unknown'

    customer_product_group['pseudo_gender'] = customer_product_group.apply(infer_gender, axis=1)
    return customer_product_group[['pseudo_gender']]

# Láº¥y nhÃ£n
pseudo_labels = heuristic_gender_label(transactions, articles)
labels = pseudo_labels[pseudo_labels['pseudo_gender'] != 'Unknown']
behavior_labeled = behavior.merge(labels, left_index=True, right_index=True)

print("Dá»¯ liá»‡u Ä‘Ã£ gÃ¡n nhÃ£n:", behavior_labeled.shape)


import numpy as np

X = behavior_labeled.drop('pseudo_gender', axis=1).values.astype('float32')
y = behavior_labeled['pseudo_gender']

# Chuáº©n hÃ³a
X /= X.max()

# Chuyá»ƒn thÃ nh ma tráº­n vuÃ´ng
img_size = int(np.ceil(np.sqrt(X.shape[1])))
pad_len = img_size**2 - X.shape[1]
X_padded = np.pad(X, ((0, 0), (0, pad_len)), 'constant')

X_images = X_padded.reshape(-1, img_size, img_size, 1)
print("Shape dá»¯ liá»‡u áº£nh:", X_images.shape)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X_images, y_categorical, test_size=0.2, random_state=42)


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

inputs = layers.Input(shape=(img_size, img_size, 1))
x = layers.Conv2D(3, (3,3), padding='same')(inputs)  # chuyá»ƒn 1 channel â†’ 3 channel
base_model = EfficientNetB0(weights=None, include_top=False, input_tensor=x)

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(y_categorical.shape[1], activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=64
)
val_loss, val_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"ğŸ”� Validation Loss: {val_loss:.4f}")
print(f"ğŸ”� Validation Accuracy: {val_acc:.4f}")



import pandas as pd

# Ä�á»�c dá»¯ liá»‡u tá»« cÃ¡c file CSV
transactions = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')
articles = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv')
customers = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv')

# Merge transactions vá»›i articles
merged = pd.merge(transactions, articles[['article_id', 'index_group_name', 'product_group_name']], 
                  on='article_id', how='left')

# Kiá»ƒm tra dá»¯ liá»‡u
print(merged.head())


#Táº¡o cÃ¡c comment + rating vÃ  phÃ¢n loáº¡i chÃºng
import random

# HÃ m táº¡o comment ngáº«u nhiÃªn tá»« danh sÃ¡ch
def generate_random_comment():
    comments = [
        "Sáº£n pháº©m tuyá»‡t vá»�i, tÃ´i ráº¥t thÃ­ch!",
        "Sáº£n pháº©m khÃ´ng nhÆ° mong Ä‘á»£i, tÃ´i khÃ¡ tháº¥t vá»�ng.",
        "Cháº¥t lÆ°á»£ng sáº£n pháº©m á»•n, nhÆ°ng khÃ´ng cÃ³ gÃ¬ ná»•i báº­t.",
        "Sáº£n pháº©m ráº¥t Ä‘áº¹p, cháº¥t lÆ°á»£ng tuyá»‡t vá»�i!",
        "KhÃ´ng hÃ i lÃ²ng vá»›i sáº£n pháº©m nÃ y, sáº½ khÃ´ng mua láº¡i."
    ]
    return random.choice(comments)

# HÃ m táº¡o rating ngáº«u nhiÃªn
def generate_random_rating(comment_type):
    if comment_type == 'Positive':
        return round(random.uniform(3.5, 5), 1)  # Rating ngáº«u nhiÃªn cho Positive tá»« 3.5 Ä‘áº¿n 5
    elif comment_type == 'Negative':
        return round(random.uniform(1, 2), 1)  # Rating ngáº«u nhiÃªn cho Negative tá»« 1 Ä‘áº¿n 2
    else:
        return round(random.uniform(2, 3.5), 1)  # Rating ngáº«u nhiÃªn cho Natural tá»« 2 Ä‘áº¿n 3.5

# Táº¡o comment ngáº«u nhiÃªn cho táº¥t cáº£ cÃ¡c sáº£n pháº©m trong merged
merged['comments'] = [generate_random_comment() for _ in range(len(merged))]

# GÃ¡n loáº¡i comment (Positive, Negative, Natural) ngáº«u nhiÃªn
merged['comment_type'] = merged['comments'].apply(lambda x: 'Positive' if 'tuyá»‡t vá»�i' in x else ('Negative' if 'khÃ¡ tháº¥t vá»�ng' in x else 'Natural'))

# GÃ¡n rating ngáº«u nhiÃªn cho tá»«ng comment
merged['ratings'] = merged['comment_type'].apply(lambda x: generate_random_rating(x))

# Kiá»ƒm tra káº¿t quáº£
print(merged[['product_group_name', 'comments', 'ratings']].head())





import matplotlib.pyplot as plt

# Dá»¯ liá»‡u cho biá»ƒu Ä‘á»“
labels = ['syntheticFB', 'syntheticRatings', 'transactions', 'customers', 'articles']
data = [1000, 5000, 31788324, 1370792, 105542]

# Táº¡o biá»ƒu Ä‘á»“ cá»™t
plt.figure(figsize=(10, 6))
plt.bar(labels, data, color=['blue', 'orange', 'green', 'red', 'purple'])

# ThÃªm tiÃªu Ä‘á»� vÃ  nhÃ£n cho cÃ¡c trá»¥c
plt.title('So sÃ¡nh sá»‘ lÆ°á»£ng báº£n ghi tá»« cÃ¡c nguá»“n dá»¯ liá»‡u')
plt.xlabel('Nguá»“n dá»¯ liá»‡u')
plt.ylabel('Sá»‘ lÆ°á»£ng báº£n ghi')

# Hiá»ƒn thá»‹ giÃ¡ trá»‹ trÃªn cÃ¡c cá»™t
for i, v in enumerate(data):
    plt.text(i, v + 0.05 * max(data), str(v), ha='center', va='bottom')

# Hiá»ƒn thá»‹ biá»ƒu Ä‘á»“
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
# Dá»¯ liá»‡u cho biá»ƒu Ä‘á»“
labels = ['syntheticFB', 'syntheticRatings', 'transactions', 'customers', 'articles']
data = [1000, 5000, 31788324, 1370792, 105542]
# Táº¡o biá»ƒu Ä‘á»“ Ä‘Æ°á»�ng
plt.figure(figsize=(12, 6))
plt.plot(labels, data, marker='o', linestyle='-', color='blue', linewidth=2, markersize=8)
# ThÃªm tiÃªu Ä‘á»� vÃ  nhÃ£n
plt.title('SO SÃ�NH Sá»� LÆ¯á»¢NG Báº¢N GHI Tá»ª CÃ�C NGUá»’N Dá»® LIá»†U', fontsize=14, fontweight='bold')
plt.xlabel('Nguá»“n dá»¯ liá»‡u', fontsize=12)
plt.ylabel('Sá»‘ lÆ°á»£ng báº£n ghi (log scale)', fontsize=12)
# Hiá»ƒn thá»‹ giÃ¡ trá»‹ trÃªn cÃ¡c Ä‘iá»ƒm dá»¯ liá»‡u
for i, v in enumerate(data):
    plt.text(i, v + 0.05 * max(data), f'{v:,}', ha='center', va='bottom', fontsize=10)
# ThÃªm lÆ°á»›i vÃ  sá»­ dá»¥ng thang Ä‘o logarit do chÃªnh lá»‡ch lá»›n
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.yscale('log')  # DÃ¹ng thang log Ä‘á»ƒ dá»… quan sÃ¡t
# TÃ¹y chá»‰nh trá»¥c y
plt.yticks([10**3, 10**5, 10**7], ['1K', '100K', '10M'])
# Hiá»ƒn thá»‹ biá»ƒu Ä‘á»“
plt.tight_layout()
plt.show()


# Chuyá»ƒn 't_dat' thÃ nh Ä‘á»‹nh dáº¡ng datetime
merged['t_dat'] = pd.to_datetime(merged['t_dat'])

# Táº¡o hÃ m xÃ¡c Ä‘á»‹nh mÃ¹a theo thÃ¡ng
def get_season(date):
    month = date.month
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'

# GÃ¡n mÃ¹a vÃ o dataframe
merged['season'] = merged['t_dat'].apply(get_season)

# Kiá»ƒm tra káº¿t quáº£
print(merged[['t_dat', 'season']].head())


#Ä�áº¿m sá»‘ lÆ°á»£ng giao dá»‹ch má»—i mÃ¹a
print(merged['season'].value_counts())


#Xem Ä‘Ã¡nh giÃ¡ trung bÃ¬nh theo mÃ¹a:
print(merged.groupby('season')['ratings'].mean())


#Xem sáº£n pháº©m nÃ o phá»• biáº¿n theo mÃ¹a
top_products_per_season = merged.groupby(['season', 'product_group_name']).size().reset_index(name='count')
top_products = top_products_per_season.sort_values(['season', 'count'], ascending=[True, False])
print(top_products.head(10))



#Pivot táº¡o vector mÃ¹a cho tá»«ng product_group_name
# Ä�áº¿m sá»‘ láº§n sáº£n pháº©m xuáº¥t hiá»‡n theo mÃ¹a
top_products_per_season = merged.groupby(['season', 'product_group_name']).size().reset_index(name='count')

# Pivot láº¡i Ä‘á»ƒ má»—i product_group_name lÃ  1 dÃ²ng, má»—i mÃ¹a lÃ  1 cá»™t
season_product_matrix = (
    top_products_per_season
    .pivot(index='product_group_name', columns='season', values='count')
    .fillna(0)
)

print(season_product_matrix.head())



!pip install /kaggle/input/scikit-learn-extra-030-cp310-manylinux-217-x86/scikit_learn_extra-0.3.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl



from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids

# Chuáº©n hoÃ¡ dá»¯ liá»‡u
scaler = StandardScaler()
X = scaler.fit_transform(season_product_matrix)

# PhÃ¢n 4 cá»¥m (tÆ°Æ¡ng á»©ng 4 mÃ¹a)
kmed = KMedoids(n_clusters=4, random_state=42)
season_clusters = kmed.fit_predict(X)

# GÃ¡n káº¿t quáº£ vÃ o báº£ng
season_product_matrix['season_cluster'] = season_clusters
season_product_matrix.head()



# Reset index Ä‘á»ƒ merge
product_season_map = season_product_matrix[['season_cluster']].reset_index()  # product_group_name sáº½ thÃ nh cá»™t

# Gáº¯n tá»«ng product_group_name vá»›i season_cluster vÃ o merged
merged = merged.merge(product_season_map, on='product_group_name', how='left')

print(merged[['product_group_name', 'season', 'season_cluster']].head())



# Step 1: Pivot thÃ nh báº£ng má»—i user = 1 dÃ²ng, má»—i season = 1 cá»™t (Ä‘áº¿m sá»‘ lÆ°á»£t mua)
user_season_matrix = (
    merged.groupby(['customer_id', 'season'])
          .size()
          .unstack(fill_value=0)
)

print("User-season matrix:", user_season_matrix.shape)
print(user_season_matrix.head())

# Step 2: Cháº¡y KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
X_user = scaler.fit_transform(user_season_matrix)

kmeans_user = KMeans(n_clusters=4, random_state=42)
user_clusters = kmeans_user.fit_predict(X_user)

# GÃ¡n cá»¥m hÃ nh vi mÃ¹a cho user
user_season_matrix['user_season_cluster'] = user_clusters
user_season_matrix.head()



# Reset index trÆ°á»›c khi merge
user_season_map = user_season_matrix[['user_season_cluster']].reset_index()

# Merge vÃ o merged theo customer_id
merged = merged.merge(user_season_map, on='customer_id', how='left')

print(merged[['customer_id','season','season_cluster','user_season_cluster']].head())



merged.to_csv('/kaggle/working/merged_processed.csv', index=False)
print("âœ… File Ä‘Ã£ Ä‘Æ°á»£c lÆ°u!")



merged.to_csv('/kaggle/working/merged_processed.csv', index=False)



AN báº¯t Ä‘áº§u cháº¡y tá»« Ä‘Ã¢y nha, t lÆ°u bá»™ dl á»Ÿ trÃªn vÃ´ input r Ã¡ m cháº¡y lá»‡nh dÆ°á»›i lÃ  Ä‘Æ°á»£c


import pandas as pd
merged = pd.read_csv('/kaggle/input/hm-seasonal-processed-data-78/merged_with_gender.csv')
print(merged.head())
print(merged.shape)


import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

required_columns = ['customer_id', 'article_id', 'ratings', 'season_cluster','product_group_name']
optional_columns = ['user_season_cluster', 'gender', 'eco_friendly']
all_columns = required_columns + [col for col in optional_columns if col in merged.columns]

print("Sá»‘ hÃ ng trong merged:", len(merged))
print("Sample merged data:\n", merged[all_columns].head())

if len(merged) == 0:
    raise ValueError("DataFrame merged is empty. Check data loading or merging process.")

if not all(col in merged.columns for col in required_columns):
    raise ValueError(f"Missing required columns. Required: {required_columns}, Found: {merged.columns}")

if merged[required_columns].isna().any().any():
    print("Warning: NaN values found in required columns. Filling with defaults.")
    merged['ratings'] = merged['ratings'].fillna(0)
    merged['season_cluster'] = merged['season_cluster'].fillna('unknown')

# Xá»­ lÃ½ optional columns
if 'gender' not in merged.columns:
    merged['gender'] = 'unknown'
if 'eco_friendly' not in merged.columns:
    merged['eco_friendly'] = False
if 'user_season_cluster' not in merged.columns:
    merged['user_season_cluster'] = merged['season_cluster']  # Dá»± phÃ²ng

# BÆ°á»›c 2: Lá»�c top 1000 ngÆ°á»�i dÃ¹ng
user_freq = merged['customer_id'].value_counts()
top_users = user_freq.head(1000).index
merged_subset = merged[merged['customer_id'].isin(top_users)]

# Debug: Kiá»ƒm tra merged_subset
train_data, test_data = train_test_split(merged_subset, test_size=0.2, random_state=42)
print("Sá»‘ hÃ ng trong train_data:", len(train_data))
print("Sá»‘ hÃ ng trong test_data:", len(test_data))
print("Sá»‘ user trong train_data:", train_data['customer_id'].nunique())
print("Sá»‘ user trong test_data:", test_data['customer_id'].nunique())
print("Sá»‘ article trong train_data:", train_data['article_id'].nunique())
print("Sá»‘ article trong test_data:", test_data['article_id'].nunique())

if len(merged_subset) == 0:
    raise ValueError("merged_subset is empty. Check customer_id filtering or data integrity.")
train_data, test_data = train_test_split(merged_subset, test_size=0.2, random_state=42)
print("Sá»‘ hÃ ng trong train_data:", len(train_data))
print("Sá»‘ hÃ ng trong test_data:", len(test_data))
# BÆ°á»›c 3: Táº¡o ma tráº­n user-item
user_item_matrix_train = train_data.pivot_table(
    index='customer_id',
    columns='article_id',
    values='ratings',
    aggfunc='sum'
).fillna(0)

user_item_values_train = csr_matrix(user_item_matrix_train.values)
print("Ma tráº­n user-item (train):", user_item_values_train.shape)

if user_item_matrix_train.shape[0] == 0 or user_item_matrix_train.shape[1] == 0:
    raise ValueError("user_item_matrix_train is empty.")


from sklearn.metrics.pairwise import cosine_similarity

# Convert sang numpy array
user_similarity_train = cosine_similarity(user_item_values_train)
print("Shape ma tráº­n similarity (train):", user_similarity_train.shape)


import numpy as np
ratings_train = user_item_values_train.toarray()
num_users, num_items = ratings_train.shape
predicted_ratings = np.dot(user_similarity_train, ratings_train) / np.sum(np.abs(user_similarity_train), axis=1, keepdims=True)
predicted_ratings[np.isnan(predicted_ratings)] = 0
print("Dá»± Ä‘oÃ¡n ma tráº­n ratings hoÃ n táº¥t:", predicted_ratings.shape)


import numpy as np

# Ma tráº­n ratings gá»‘c: user-item matrix (numpy)
ratings = user_item_values
num_users, num_items = ratings.shape

# Ma tráº­n káº¿t quáº£: dá»± Ä‘oÃ¡n ratings cho cÃ¡c Ã´ ratings==0
predicted_ratings = np.zeros((num_users, num_items))

for u in range(num_users):
    sim_u = user_similarity[u, :]  # vector similarity cá»§a user u Ä‘áº¿n toÃ n bá»™ user
    for i in range(num_items):
        if ratings[u, i] == 0:
            # Láº¥y ratings cá»§a item i tá»« táº¥t cáº£ user khÃ¡c
            ratings_i = ratings[:, i]
            
            # Chá»‰ láº¥y user cÃ³ rating khÃ¡c 0 á»Ÿ item i
            mask = ratings_i > 0
            
            if np.sum(mask) > 0:
                sim_scores = sim_u[mask]
                ratings_scores = ratings_i[mask]
                
                # TÃ­nh dá»± Ä‘oÃ¡n: sim*rating / sum(abs(sim))
                pred = np.dot(sim_scores, ratings_scores) / np.sum(np.abs(sim_scores))
                predicted_ratings[u, i] = pred
            else:
                predicted_ratings[u, i] = 0  # náº¿u khÃ´ng ai Ä‘Ã£ rating item nÃ y

print("Dá»± Ä‘oÃ¡n ma tráº­n ratings hoÃ n táº¥t:", predicted_ratings.shape)



user_fav_cluster = {}
current_season = 0  # DÃ¹ng sá»‘ thay vÃ¬ chuá»—i

for user_id in user_item_matrix_train.index:
    bought_items = user_item_matrix_train.loc[user_id]
    bought_items = bought_items[bought_items > 0].index.tolist()
    if bought_items:
        clusters = train_data[train_data['article_id'].isin(bought_items)]['season_cluster']
        if not clusters.empty:
            user_fav_cluster[user_id] = clusters.mode().iloc[0]
        else:
            user_fav_cluster[user_id] = current_season
    else:
        user_cluster = train_data[train_data['customer_id'] == user_id]['user_season_cluster']
        if not user_cluster.empty:
            user_fav_cluster[user_id] = user_cluster.mode().iloc[0]
        else:
            user_fav_cluster[user_id] = current_season

# Debug: Kiá»ƒm tra user_fav_cluster
print("Sá»‘ user cÃ³ fav_cluster:", len(user_fav_cluster))
print("Sample user_fav_cluster:", list(user_fav_cluster.items())[:5])



current_season = 0
item_cluster_map = dict(zip(train_data['article_id'], train_data['season_cluster']))
item_gender_map = dict(zip(train_data['article_id'], train_data['gender']))
item_eco_map = dict(zip(train_data['article_id'], train_data['eco_friendly']))
item_product_group_map = dict(zip(train_data['article_id'], train_data['product_group_name']))

# Ä�áº£m báº£o táº¥t cáº£ article_id trong user_item_matrix_train cÃ³ season_cluster
missing_items = set(user_item_matrix_train.columns) - set(item_cluster_map.keys())
if missing_items:
    print(f"Warning: {len(missing_items)} article_id(s) missing season_cluster")
    for item_id in missing_items:
        item_cluster_map[item_id] = current_season

# Debug: Kiá»ƒm tra mapping
print("Sá»‘ article_id cÃ³ season_cluster:", len(item_cluster_map))
print("Sample item_cluster_map:", list(item_cluster_map.items())[:5])
print("Sá»‘ article_id cÃ³ product_group_name:", len(item_product_group_map))
print("Sample item_product_group_map:", list(item_product_group_map.items())[:5])


recommendations = {}
skipped_users = 0
empty_filtered_items = 0
user_gender_map = {}

for user_idx, user_id in enumerate(user_item_matrix_train.index):
    fav_cluster = user_fav_cluster.get(user_id, current_season)
    if fav_cluster is None:
        skipped_users += 1
        continue
    
    user_gender = user_gender_map.get(user_id, None)
    preds = predicted_ratings[user_idx, :]
    item_ids = user_item_matrix_train.columns
    
    # Lá»�c sáº£n pháº©m
    item_mask = np.array([
        item_cluster_map.get(iid, current_season) == fav_cluster
        for iid in item_ids
    ])
    filtered_indices = np.where(item_mask)[0]
    pred_items_filtered = [
        (item_ids[i], preds[i] * (1.2 if item_eco_map.get(item_ids[i], False) else 1.0))
        for i in filtered_indices
    ]
    
    # Gá»£i Ã½ sáº£n pháº©m phá»• biáº¿n náº¿u rá»—ng
    if not pred_items_filtered:
        empty_filtered_items += 1
        popular_items = train_data[
            (train_data['season_cluster'] == fav_cluster) &
            (train_data['eco_friendly'] == True)
        ]['article_id']['product_group_name'].value_counts().head(5).index
        if not popular_items.empty:
            pred_items_filtered = [(iid, 0.0) for iid in popular_items]
        else:
            popular_items = train_data[
                (train_data['season_cluster'] == fav_cluster)
            ]['article_id']['product_group_name'].value_counts().head(5).index
            if not popular_items.empty:
                pred_items_filtered = [(iid, 0.0) for iid in popular_items]
            else:
                popular_items = train_data['article_id']['product_group_name'].value_counts().head(5).index
                pred_items_filtered = [(iid, 0.0) for iid in popular_items]
    
    # Sáº¯p xáº¿p vÃ  láº¥y top 5
    pred_items_filtered.sort(key=lambda x: x[1], reverse=True)
    recommendations[user_id] = pred_items_filtered[:5]

print("Táº¡o xong recommendation hybrid cho", len(recommendations), "users.")
print("Sá»‘ user bá»‹ bá»� qua (fav_cluster is None):", skipped_users)
print("Sá»‘ user cÃ³ pred_items_filtered rá»—ng:", empty_filtered_items)


# VÃ­ dá»¥ in top sáº£n pháº©m gá»£i Ã½ cho 3 user
for i, (uid, recs) in enumerate(recommendations.items()):
    print(f"{i+1}. User {uid}: {[r[0] for r in recs]}")
    if i == 3:
        break


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Giáº£ sá»­ user_item_matrix_train, predicted_ratings, recommendations, test_data Ä‘Ã£ Ä‘Æ°á»£c táº¡o tá»« code trÆ°á»›c
# BÆ°á»›c: Ä�Ã¡nh giÃ¡ mÃ´ hÃ¬nh

# Táº¡o ma tráº­n user-item cho test
user_item_matrix_test = test_data.pivot_table(
    index='customer_id',
    columns='article_id',
    values='ratings',
    aggfunc='sum'
).fillna(0)

# TÃ¬m ngÆ°á»�i dÃ¹ng vÃ  sáº£n pháº©m chung giá»¯a train vÃ  test
common_users = user_item_matrix_train.index.intersection(user_item_matrix_test.index)
common_items = user_item_matrix_train.columns.intersection(user_item_matrix_test.columns)

# Debug: Kiá»ƒm tra kÃ­ch thÆ°á»›c
print("Shape cá»§a predicted_ratings:", predicted_ratings.shape)
print("Sá»‘ common_users:", len(common_users))
print("Sá»‘ common_items:", len(common_items))
print("Shape cá»§a user_item_matrix_train:", user_item_matrix_train.shape)
print("Shape cá»§a user_item_matrix_test:", user_item_matrix_test.shape)

if len(common_users) == 0 or len(common_items) == 0:
    print("Warning: No common users or items between train and test. Cannot compute RMSE.")
else:
    # Láº¥y true ratings tá»« test
    true_ratings = user_item_matrix_test.loc[common_users, common_items].values
    
    # Láº¥y predicted ratings
    user_indices = user_item_matrix_train.index.get_indexer(common_users)
    item_indices = user_item_matrix_train.columns.get_indexer(common_items)
    
    # Kiá»ƒm tra chá»‰ sá»‘ há»£p lá»‡
    if np.any(user_indices == -1) or np.any(item_indices == -1):
        print("Warning: Some common_users or common_items not found in user_item_matrix_train.")
        valid_mask = (user_indices != -1) & (item_indices != -1)
        user_indices = user_indices[valid_mask]
        item_indices = item_indices[valid_mask]
        common_users = common_users[valid_mask]
        common_items = common_items[valid_mask]
    
    # Láº¥y pred_ratings
    try:
        pred_ratings = predicted_ratings[user_indices][:, item_indices]
        
        # Kiá»ƒm tra shape
        print("Shape cá»§a true_ratings:", true_ratings.shape)
        print("Shape cá»§a pred_ratings:", pred_ratings.shape)
        
        if true_ratings.shape != pred_ratings.shape:
            print("Error: Shape mismatch between true_ratings and pred_ratings.")
        else:
            # TÃ­nh RMSE
            mask = true_ratings > 0
            if np.sum(mask) > 0:
                rmse = np.sqrt(mean_squared_error(true_ratings[mask], pred_ratings[mask]))
                print("RMSE on test set:", rmse)
            else:
                print("Warning: No non-zero true ratings to compute RMSE.")
    except IndexError as e:
        print(f"IndexError: {e}")
        print("Cannot compute RMSE due to indexing issues.")

# TÃ­nh Precision@5
def precision_at_k(recommendations, test_data, k=5):
    hits = 0
    total = 0
    for user_id in recommendations:
        if user_id in test_data['customer_id'].values:
            rec_items = [r[0] for r in recommendations[user_id][:k]]
            true_items = test_data[test_data['customer_id'] == user_id]['article_id'].tolist()
            hits += len(set(rec_items).intersection(true_items))
            total += min(len(true_items), k)
    return hits / total if total > 0 else 0

precision = precision_at_k(recommendations, test_data, k=5)
print("Precision@5 on test set:", precision)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Giáº£ sá»­ user_item_matrix_train, predicted_ratings, recommendations, test_data Ä‘Ã£ Ä‘Æ°á»£c táº¡o tá»« code trÆ°á»›c

# BÆ°á»›c: Ä�Ã¡nh giÃ¡ mÃ´ hÃ¬nh

# Táº¡o ma tráº­n user-item cho test
user_item_matrix_test = test_data.pivot_table(
    index='customer_id',
    columns='article_id',
    values='ratings',
    aggfunc='sum'
).fillna(0)

# TÃ¬m ngÆ°á»�i dÃ¹ng vÃ  sáº£n pháº©m chung giá»¯a train vÃ  test
common_users = user_item_matrix_train.index.intersection(user_item_matrix_test.index)
common_items = user_item_matrix_train.columns.intersection(user_item_matrix_test.columns)

# Debug: Kiá»ƒm tra kÃ­ch thÆ°á»›c
print("Shape cá»§a predicted_ratings:", predicted_ratings.shape)
print("Sá»‘ common_users:", len(common_users))
print("Sá»‘ common_items:", len(common_items))
print("Shape cá»§a user_item_matrix_train:", user_item_matrix_train.shape)
print("Shape cá»§a user_item_matrix_test:", user_item_matrix_test.shape)

if len(common_users) == 0 or len(common_items) == 0:
    print("Warning: No common users or items between train and test. Cannot compute RMSE.")
else:
    # Láº¥y true ratings tá»« test
    true_ratings = user_item_matrix_test.loc[common_users, common_items].values
    
    # Láº¥y chá»‰ sá»‘ cho common_users vÃ  common_items
    user_indices = user_item_matrix_train.index.get_indexer(common_users)
    item_indices = user_item_matrix_train.columns.get_indexer(common_items)
    
    # Kiá»ƒm tra chá»‰ sá»‘ há»£p lá»‡ riÃªng láº»
    valid_user_mask = user_indices != -1
    valid_item_mask = item_indices != -1
    
    # Lá»�c common_users vÃ  common_items dá»±a trÃªn chá»‰ sá»‘ há»£p lá»‡
    if not np.all(valid_user_mask) or not np.all(valid_item_mask):
        print("Warning: Some users or items not found in user_item_matrix_train.")
        valid_indices = valid_user_mask & (valid_item_mask[:len(valid_user_mask)])
        common_users = common_users[valid_indices]
        common_items = common_items[valid_indices[:len(common_items)]]
        user_indices = user_indices[valid_indices]
        item_indices = item_indices[valid_indices[:len(item_indices)]]
        
        # Cáº­p nháº­t true_ratings
        true_ratings = user_item_matrix_test.loc[common_users, common_items].values
    
    # Kiá»ƒm tra shape trÆ°á»›c khi láº­p chá»‰ má»¥c
    if len(user_indices) == 0 or len(item_indices) == 0:
        print("Error: No valid indices after filtering. Cannot compute RMSE.")
    else:
        try:
            pred_ratings = predicted_ratings[user_indices][:, item_indices]
            
            # Kiá»ƒm tra shape
            print("Shape cá»§a true_ratings:", true_ratings.shape)
            print("Shape cá»§a pred_ratings:", pred_ratings.shape)
            
            if true_ratings.shape != pred_ratings.shape:
                print("Error: Shape mismatch between true_ratings and pred_ratings.")
            else:
                # TÃ­nh RMSE
                mask = true_ratings > 0
                if np.sum(mask) > 0:
                    rmse = np.sqrt(mean_squared_error(true_ratings[mask], pred_ratings[mask]))
                    print("RMSE on test set:", rmse)
                else:
                    print("Warning: No non-zero true ratings to compute RMSE.")
        except IndexError as e:
            print(f"IndexError: {e}")
            print("Cannot compute RMSE due to indexing issues.")

# HÃ m tÃ­nh Precision@5, Recall@5, F1-Score@5, MAP@5
def evaluate_recommendations(recommendations, test_data, k=5):
    precision = 0
    recall = 0
    f1 = 0
    ap_sum = 0
    total = 0
    
    for user_id in recommendations:
        if user_id in test_data['customer_id'].values:
            rec_items = [r[0] for r in recommendations[user_id][:k]]
            true_items = test_data[test_data['customer_id'] == user_id]['article_id'].tolist()
            
            # TÃ­nh Precision@5
            hits = len(set(rec_items).intersection(true_items))
            precision += hits / k if k > 0 else 0
            
            # TÃ­nh Recall@5
            recall += hits / len(true_items) if len(true_items) > 0 else 0
            
            # TÃ­nh AP@5 (Average Precision)
            ap = 0
            relevant_count = 0
            for i, item in enumerate(rec_items[:k], 1):
                if item in true_items:
                    relevant_count += 1
                    ap += relevant_count / i
            ap = ap / min(len(true_items), k) if len(true_items) > 0 else 0
            ap_sum += ap
            
            total += 1
    
    precision = precision / total if total > 0 else 0
    recall = recall / total if total > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    map_score = ap_sum / total if total > 0 else 0
    
    return {
        'Precision@5': precision,
        'Recall@5': recall,
        'F1-Score@5': f1,
        'MAP@5': map_score
    }

# TÃ­nh vÃ  in cÃ¡c chá»‰ sá»‘
eval_metrics = evaluate_recommendations(recommendations, test_data, k=5)
print("Evaluation metrics:")
for metric, value in eval_metrics.items():
    print(f"{metric}: {value}")


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# === TF-IDF tá»« dá»¯ liá»‡u train ===
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(train_data['features'])

def get_cbf_recommendations(article_id, top_n=10, max_cluster_size=1000):
    # Láº¥y index cá»§a sáº£n pháº©m trong train_data
    idx_list = train_data.index[train_data['article_id'] == article_id].tolist()
    if not idx_list:
        return []
    idx = idx_list[0]

    # Láº¥y cluster
    cluster_id = train_data.at[idx, 'cluster']
    cluster_indices = train_data.index[train_data['cluster'] == cluster_id].tolist()

    # Náº¿u cluster rá»—ng hoáº·c chá»‰ cÃ³ 1 sp â†’ tráº£ rá»—ng
    if len(cluster_indices) <= 1:
        return []

    # Giá»›i háº¡n kÃ­ch thÆ°á»›c cluster
    if len(cluster_indices) > max_cluster_size:
        cluster_indices = [idx] + random.sample(
            [i for i in cluster_indices if i != idx],
            max_cluster_size - 1
        )

    # TÃ­nh cosine similarity
    cluster_matrix = tfidf_matrix[cluster_indices, :]
    idx_in_cluster = cluster_indices.index(idx)
    cosine_sim = cosine_similarity(cluster_matrix[idx_in_cluster], cluster_matrix).flatten()

    # Láº¥y top_n bÃ i viáº¿t tÆ°Æ¡ng tá»± (bá»� chÃ­nh nÃ³)
    similar_indices = sorted(
        list(enumerate(cosine_sim)), key=lambda x: x[1], reverse=True
    )[1:top_n+1]

    return [train_data.iloc[cluster_indices[i]]['article_id'] for i, _ in similar_indices]



def evaluate_precision_recall(customer_id, k=5, recommend_func=None):
    true_items = set(test_data[test_data['customer_id'] == customer_id]['article_id'])
    if not true_items:
        return None, None

    # Láº¥y 1 sáº£n pháº©m user Ä‘Ã£ mua trong train Ä‘á»ƒ lÃ m input
    start_items = train_data[train_data['customer_id'] == customer_id]['article_id'].tolist()
    if not start_items:
        return None, None
    start_item = start_items[0]

    recommended_items = set(recommend_func(start_item, top_n=k))
    if not recommended_items:
        return None, None

    precision = len(recommended_items & true_items) / k
    recall = len(recommended_items & true_items) / len(true_items)

    return precision, recall



def get_hybrid_recommendations(article_id, top_n=10, alpha=0.5):
    cbf_recs = get_cbf_recommendations(article_id, top_n=top_n)
    cf_recs = get_cf_recommendations(article_id, top_n=top_n)

    scores = {}
    for rank, art in enumerate(cbf_recs):
        scores[art] = scores.get(art, 0) + alpha * (top_n - rank)
    for rank, art in enumerate(cf_recs):
        scores[art] = scores.get(art, 0) + (1 - alpha) * (top_n - rank)

    return [art for art, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:top_n]



def get_cf_recommendations(article_id, top_n=10):
    # CF hiá»‡n táº¡i dá»±a trÃªn user-user similarity, nÃªn mÃ¬nh cáº§n láº¥y user Ä‘Ã£ mua article nÃ y
    # rá»“i tá»« user Ä‘Ã³ chá»�n ra top sáº£n pháº©m dá»± Ä‘oÃ¡n cao nháº¥t (bá»� sáº£n pháº©m gá»‘c)
    
    # TÃ¬m 1 user Ä‘Ã£ mua article nÃ y trong train
    users_bought = train_data[train_data['article_id'] == article_id]['customer_id'].tolist()
    if not users_bought:
        return []
    user_id = users_bought[0]
    
    # Náº¿u user_id khÃ´ng náº±m trong user_item_matrix_train â†’ bá»� qua
    if user_id not in user_item_matrix_train.index:
        return []
    
    # Láº¥y chá»‰ sá»‘ user
    user_idx = user_item_matrix_train.index.get_loc(user_id)
    preds = predicted_ratings[user_idx, :]
    
    # Sáº¯p xáº¿p theo score giáº£m dáº§n
    top_indices = np.argsort(preds)[::-1]
    recs = []
    for idx in top_indices:
        art_id = user_item_matrix_train.columns[idx]
        if art_id != article_id:  # Bá»� chÃ­nh nÃ³
            recs.append(art_id)
        if len(recs) >= top_n:
            break
    
    return recs



import numpy as np
import pandas as pd

def evaluate_models(customer_ids, k=5, alpha=0.5):
    results = []
    
    for model_name, func in [
        ("CBF", get_cbf_recommendations),
        ("CF", get_cf_recommendations),
        ("Hybrid", lambda art_id, top_n: get_hybrid_recommendations(art_id, top_n=top_n, alpha=alpha))
    ]:
        precisions, recalls = [], []
        
        for cid in customer_ids:
            p, r = evaluate_precision_recall(cid, k=k, recommend_func=func)
            if p is not None:
                precisions.append(p)
                recalls.append(r)
        
        results.append({
            "Model": model_name,
            f"Precision@{k}": np.mean(precisions) if precisions else 0,
            f"Recall@{k}": np.mean(recalls) if recalls else 0
        })
    
    return pd.DataFrame(results)

# Cháº¡y thá»­
sample_customers = test_data['customer_id'].unique()[:500]  # Láº¥y 500 user Ä‘á»ƒ test
comparison_df = evaluate_models(sample_customers, k=5, alpha=0.5)
print(comparison_df)



# Replace with actual CF predictions data 
cf_predictions = np.array([2.9, 3.0, 3.5, 4.0, 3.2])  # Example CF predictions

# Replace with your actual CBF predictions data 
cbf_predictions = np.array([3.0, 3.1, 3.4, 3.9, 3.5])  # Example CBF predictions

# Define weights for CF and CBF (sum of weights should be 1)
weight_cf = 0.5
weight_cbf = 0.5

# Combine CF and CBF predictions (weighted average)
hybrid_predictions = weight_cf * cf_predictions + weight_cbf * cbf_predictions

# Now you can print or evaluate hybrid_predictions
print(hybrid_predictions)


import joblib
import os

# Táº¡o thÆ° má»¥c lÆ°u mÃ´ hÃ¬nh náº¿u chÆ°a cÃ³
model_dir = 'saved_model'
os.makedirs(model_dir, exist_ok=True)

# LÆ°u cÃ¡c thÃ nh pháº§n chÃ­nh
joblib.dump(user_similarity_train, os.path.join(model_dir, 'user_similarity_train.pkl'))
joblib.dump(predicted_ratings, os.path.join(model_dir, 'predicted_ratings.pkl'))
joblib.dump(user_fav_cluster, os.path.join(model_dir, 'user_fav_cluster.pkl'))
joblib.dump(item_cluster_map, os.path.join(model_dir, 'item_cluster_map.pkl'))
joblib.dump(item_gender_map, os.path.join(model_dir, 'item_gender_map.pkl'))
joblib.dump(item_eco_map, os.path.join(model_dir, 'item_eco_map.pkl'))
joblib.dump(item_product_group_map, os.path.join(model_dir, 'item_product_group_map.pkl'))
joblib.dump(recommendations, os.path.join(model_dir, 'recommendations.pkl'))
joblib.dump(user_item_matrix_train, os.path.join(model_dir, 'user_item_matrix_train.pkl'))
joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))  # Náº¿u dÃ¹ng CBF
joblib.dump(tfidf_matrix, os.path.join(model_dir, 'tfidf_matrix.pkl'))  # Náº¿u dÃ¹ng CBF

print("MÃ´ hÃ¬nh Ä‘Ã£ Ä‘Æ°á»£c lÆ°u vÃ o thÆ° má»¥c:", model_dir)


user_similarity_train = joblib.load('saved_model/user_similarity_train.pkl')
predicted_ratings, os.path.join(model_dir, 'predicted_ratings.pkl')
user_fav_cluster, os.path.join(model_dir, 'user_fav_cluster.pkl')
item_cluster_map, os.path.join(model_dir, 'item_cluster_map.pkl')
item_gender_map, os.path.join(model_dir, 'item_gender_map.pkl')
item_eco_map, os.path.join(model_dir, 'item_eco_map.pkl')
item_product_group_map, os.path.join(model_dir, 'item_product_group_map.pkl')
recommendations, os.path.join(model_dir, 'recommendations.pkl')
user_item_matrix_train, os.path.join(model_dir, 'user_item_matrix_train.pkl')
vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl')
tfidf_matrix, os.path.join(model_dir, 'tfidf_matrix.pkl'


import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# New: Precompute set of article_ids that have images
image_dir = '/kaggle/input/h-and-m-personalized-fashion-recommendations/images'
existing_articles = set()
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('.jpg'):
            article_str = file[:-4]  # Remove .jpg
            try:
                existing_articles.add(int(article_str))
            except ValueError:
                pass  # Skip any invalid filenames

print(f"Found {len(existing_articles)} articles with images.")

# Modified recommendations loop to filter for existing images
recommendations = {}
skipped_users = 0
empty_filtered_items = 0
user_gender_map = {}  # Assuming this is defined elsewhere if needed

for user_idx, user_id in enumerate(user_item_matrix_train.index):
    fav_cluster = user_fav_cluster.get(user_id, current_season)
    if fav_cluster is None:
        skipped_users += 1
        continue
    
    user_gender = user_gender_map.get(user_id, None)
    preds = predicted_ratings[user_idx, :]
    item_ids = user_item_matrix_train.columns
    
    # Lá»�c sáº£n pháº©m vá»›i additional check for existing image
    item_mask = np.array([
        item_cluster_map.get(iid, current_season) == fav_cluster and iid in existing_articles
        for iid in item_ids
    ])
    filtered_indices = np.where(item_mask)[0]
    pred_items_filtered = [
        (item_ids[i], preds[i] * (1.2 if item_eco_map.get(item_ids[i], False) else 1.0))
        for i in filtered_indices
    ]
    
    # Gá»£i Ã½ sáº£n pháº©m phá»• biáº¿n náº¿u rá»—ng, but only those with images
    if not pred_items_filtered:
        empty_filtered_items += 1
        popular_items = train_data[
            (train_data['season_cluster'] == fav_cluster) &
            (train_data['eco_friendly'] == True) &
            (train_data['article_id'].isin(existing_articles))
        ]['article_id'].value_counts().head(5).index
        if not popular_items.empty:
            pred_items_filtered = [(iid, 0.0) for iid in popular_items]
        else:
            popular_items = train_data[
                (train_data['season_cluster'] == fav_cluster) &
                (train_data['article_id'].isin(existing_articles))
            ]['article_id'].value_counts().head(5).index
            if not popular_items.empty:
                pred_items_filtered = [(iid, 0.0) for iid in popular_items]
            else:
                popular_items = train_data[
                    train_data['article_id'].isin(existing_articles)
                ]['article_id'].value_counts().head(5).index
                pred_items_filtered = [(iid, 0.0) for iid in popular_items]
    
    # Sáº¯p xáº¿p vÃ  láº¥y top 5 (or more if needed for grid)
    pred_items_filtered.sort(key=lambda x: x[1], reverse=True)
    recommendations[user_id] = pred_items_filtered[:5]  # Adjust to 12 if always wanting 12 in grid

# HÃ m láº¥y Ä‘Æ°á»�ng dáº«n hÃ¬nh áº£nh (unchanged)
def get_image_path(article_id):
    article_str = str(article_id).zfill(10)
    folder = article_str[0:3]
    image_path = Path(f'/kaggle/input/h-and-m-personalized-fashion-recommendations/images/{folder}/{article_str}.jpg')
    if image_path.exists():
        return str(image_path)
    else:
        return None

# Display code (modified to expand to 12 with additional popular if needed, all with images)
random_user_id = random.choice(list(recommendations.keys()))
print(f"KhÃ¡ch hÃ ng ngáº«u nhiÃªn: {random_user_id}")

fav_cluster = user_fav_cluster.get(random_user_id, 0)
season_name = "Summer" if fav_cluster == 0 else f"Season {fav_cluster}"

recs = recommendations[random_user_id]
if len(recs) < 12:
    additional_items = train_data[
        (train_data['season_cluster'] == fav_cluster) &
        (train_data['article_id'].isin(existing_articles))
    ]['article_id'].value_counts().head(12 - len(recs)).index
    recs += [(iid, 0.0) for iid in additional_items]
recs = recs[:12]

fig, axs = plt.subplots(4, 3, figsize=(12, 16))
fig.suptitle(f"{season_name}?", fontsize=20)
for i, (article_id, score) in enumerate(recs):
    row, col = divmod(i, 3)
    ax = axs[row, col]
   
    image_path = get_image_path(article_id)
    if image_path:  # Now guaranteed by filter, but keep for safety
        img = mpimg.imread(image_path)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, f"No Image\n{article_id}", ha='center', va='center', fontsize=12)
        ax.set_facecolor('lightgray')
   
    ax.set_title(f"{article_id}\nScore: {score:.2f}", fontsize=10)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()


import pandas as pd
# Ä�á»�c dá»¯ liá»‡u (giáº£ sá»­ Ä‘Ã£ cÃ³ sáºµn)
transactions = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')
transactions= transactions.sort_values(by="t_dat")
articles = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv')
customers = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv')


import pandas as pd
import matplotlib.pyplot as plt


# HÃ m heuristic Ä‘á»ƒ gÃ¡n nhÃ£n giá»›i tÃ­nh dá»±a trÃªn index_group_name
def heuristic_gender_label(transactions, articles):
    merged_df = pd.merge(transactions, articles, on='article_id', how='left')
    customer_product_group = (
        merged_df.groupby(['customer_id', 'index_group_name'])
        .size()
        .unstack(fill_value=0)
    )
    def infer_gender(row):
        total = row.sum()
        if total == 0:
            return 'Unknown'
        men_ratio = row.get('Menswear', 0) / total
        women_ratio = row.get('Ladieswear', 0) / total
        if men_ratio > 0.6:
            return 'Male'
        elif women_ratio > 0.6:
            return 'Female'
        return 'Unknown'
    customer_product_group['pseudo_gender'] = customer_product_group.apply(infer_gender, axis=1)
    return customer_product_group[['pseudo_gender']]

# HÃ m heuristic phá»¥ Ä‘á»ƒ xá»­ lÃ½ Unknown dá»±a trÃªn product_group_name
def heuristic_gender_refine(transactions, articles, labels):
    # Láº¥y cÃ¡c khÃ¡ch hÃ ng cÃ³ nhÃ£n Unknown
    unknown_customers = labels[labels['pseudo_gender'] == 'Unknown'].index
    merged_df = pd.merge(transactions, articles, on='article_id', how='left')
    customer_product_group = (
        merged_df[merged_df['customer_id'].isin(unknown_customers)]
        .groupby(['customer_id', 'product_group_name'])
        .size()
        .unstack(fill_value=0)
    )
    
    # Sá»­ dá»¥ng cÃ¡c product_group_name thá»±c táº¿ tá»« dataset
    # VÃ­ dá»¥: dá»±a trÃªn H&M dataset, cÃ¡c giÃ¡ trá»‹ cÃ³ thá»ƒ lÃ  'Garment Upper body', 'Garment Lower body', 'Dresses', 'Underwear', v.v.
    male_indicators = ['Garment Upper body', 'Garment Lower body', 'Shoes']  # ThÆ°á»�ng liÃªn quan Ä‘áº¿n nam
    female_indicators = ['Dresses', 'Skirts', 'Underwear', 'Blouses']  # ThÆ°á»�ng liÃªn quan Ä‘áº¿n ná»¯
    
    def infer_gender_from_products(row):
        total = row.sum()
        if total == 0:
            return 'Unknown'  # Váº«n lÃ  Unknown náº¿u khÃ´ng cÃ³ giao dá»‹ch
        male_product_sum = sum(row.get(col, 0) for col in male_indicators if col in row.index)
        female_product_sum = sum(row.get(col, 0) for col in female_indicators if col in row.index)
        male_ratio = male_product_sum / total
        female_ratio = female_product_sum / total
        # Giáº£m ngÆ°á»¡ng Ä‘á»ƒ tÄƒng kháº£ nÄƒng phÃ¢n loáº¡i
        if male_ratio > 0.5:  # Giáº£m tá»« 0.6 xuá»‘ng 0.5
            return 'Male'
        elif female_ratio > 0.5:  # Giáº£m tá»« 0.6 xuá»‘ng 0.5
            return 'Female'
        return 'Unknown'
    
    customer_product_group['refined_gender'] = customer_product_group.apply(infer_gender_from_products, axis=1)
    return customer_product_group[['refined_gender']]

# GÃ¡n nhÃ£n giá»›i tÃ­nh ban Ä‘áº§u
labels = heuristic_gender_label(transactions, articles)

# Xá»­ lÃ½ cÃ¡c nhÃ£n Unknown
refined_labels = heuristic_gender_refine(transactions, articles, labels)

# Cáº­p nháº­t nhÃ£n Unknown báº±ng nhÃ£n tinh chá»‰nh
labels = labels.join(refined_labels, how='left')
labels['final_gender'] = labels['refined_gender'].combine_first(labels['pseudo_gender'])

# GÃ¡n nhÃ£n máº·c Ä‘á»‹nh cho cÃ¡c khÃ¡ch hÃ ng váº«n lÃ  Unknown (dá»±a trÃªn lá»›p Ä‘a sá»‘: Female)
labels['final_gender'] = labels['final_gender'].replace('Unknown', 'Female')

# Ä�áº¿m sá»‘ lÆ°á»£ng má»—i giá»›i tÃ­nh
gender_counts = labels['final_gender'].value_counts()
print("PhÃ¢n bá»‘ giá»›i tÃ­nh sau khi xá»­ lÃ½ Unknown:")
print(gender_counts)

# Váº½ biá»ƒu Ä‘á»“ pie cho tá»· lá»‡ giá»›i tÃ­nh
plt.figure(figsize=(6, 6))
gender_counts.plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'], startangle=90)
plt.title('Gender ratio')
plt.ylabel('')  # áº¨n nhÃ£n y Ä‘á»ƒ biá»ƒu Ä‘á»“ sáº¡ch hÆ¡n
plt.show()


import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

positive_feedbacks = [
    "Cháº¥t liá»‡u váº£i má»�m máº¡i, thoáº£i mÃ¡i khi máº·c suá»‘t cáº£ ngÃ y.",
    "Thiáº¿t káº¿ thá»�i trang, phÃ¹ há»£p vá»›i nhiá»�u dá»‹p khÃ¡c nhau.",
    "MÃ u sáº¯c Ä‘áº¹p, khÃ´ng phai sau nhiá»�u láº§n giáº·t.",
    "Size vá»«a váº·n, khÃ´ng bá»‹ co rÃºt sau giáº·t.",
    "GiÃ¡ trá»‹ tá»‘t so vá»›i giÃ¡ tiá»�n, cháº¥t lÆ°á»£ng cao.",
    "Ráº¥t bá»�n bá»‰, máº·c Ä‘Æ°á»£c lÃ¢u dÃ i mÃ  khÃ´ng há»�ng.",
    "PhÃ¹ há»£p vá»›i thá»�i tiáº¿t mÃ¹a hÃ¨, thoÃ¡ng khÃ­.",
    "Kiá»ƒu dÃ¡ng hiá»‡n Ä‘áº¡i, nháº­n Ä‘Æ°á»£c nhiá»�u lá»�i khen.",
    "Dá»… phá»‘i Ä‘á»“ vá»›i cÃ¡c trang phá»¥c khÃ¡c.",
    "Váº£i cao cáº¥p, cáº£m giÃ¡c sang trá»�ng khi máº·c.",
    "HoÃ n háº£o cho hoáº¡t Ä‘á»™ng ngoÃ i trá»�i, thoáº£i mÃ¡i váº­n Ä‘á»™ng.",
    "MÃ u sáº¯c tÆ°Æ¡i sÃ¡ng, lÃ m ná»•i báº­t phong cÃ¡ch cÃ¡ nhÃ¢n.",
    "Cháº¥t lÆ°á»£ng may cháº¯c cháº¯n, Ä‘Æ°á»�ng kim mÅ©i chá»‰ Ä‘áº¹p.",
    "PhÃ¹ há»£p cho cáº£ nam vÃ  ná»¯, unisex tuyá»‡t vá»�i.",
    "Dá»… dÃ ng giáº·t sáº¡ch, khÃ´ nhanh.",
    "Thiáº¿t káº¿ cá»• Ä‘iá»ƒn nhÆ°ng váº«n trendy.",
    "Ráº¥t áº¥m Ã¡p cho mÃ¹a Ä‘Ã´ng, giá»¯ nhiá»‡t tá»‘t.",
    "KhÃ´ng gÃ¢y dá»‹ á»©ng da, an toÃ n cho da nháº¡y cáº£m.",
    "KÃ­ch thÆ°á»›c Ä‘a dáº¡ng, dá»… chá»�n size phÃ¹ há»£p.",
    "Sáº£n pháº©m vÆ°á»£t mong Ä‘á»£i, sáº½ mua láº¡i.",
    "Phá»¥ kiá»‡n Ä‘i kÃ¨m cháº¥t lÆ°á»£ng cao.",
    "MÃ¹i thÆ¡m nháº¹ nhÃ ng tá»« váº£i má»›i.",
    "Dá»… dÃ ng káº¿t há»£p vá»›i giÃ y dÃ©p vÃ  tÃºi xÃ¡ch.",
    "Cháº¥t liá»‡u thÃ¢n thiá»‡n vá»›i mÃ´i trÆ°á»�ng.",
    "Thiáº¿t káº¿ Ä‘á»™c Ä‘Ã¡o, khÃ´ng Ä‘á»¥ng hÃ ng.",
    "Ráº¥t nháº¹, khÃ´ng náº·ng ná»� khi máº·c.",
    "MÃ u sáº¯c trung tÃ­nh, dá»… mix & match.",
    "HoÃ n háº£o cho trang phá»¥c cÃ´ng sá»Ÿ.",
    "Váº£i chá»‘ng nhÄƒn, giá»¯ form tá»‘t.",
    "Sáº£n pháº©m Ä‘a nÄƒng, dÃ¹ng Ä‘Æ°á»£c nhiá»�u cÃ¡ch."
]

negative_feedbacks = [
    "Cháº¥t liá»‡u váº£i kÃ©m, dá»… rÃ¡ch sau vÃ i láº§n máº·c.",
    "Size khÃ´ng chuáº©n, quÃ¡ cháº­t hoáº·c quÃ¡ rá»™ng.",
    "MÃ u sáº¯c phai nhanh sau giáº·t.",
    "Thiáº¿t káº¿ lá»—i thá»�i, khÃ´ng há»£p má»‘t.",
    "Váº£i gÃ¢y ngá»©a da, khÃ´ng thoáº£i mÃ¡i.",
    "Ä�Æ°á»�ng may lá»�ng láº»o, dá»… bung chá»‰.",
    "KhÃ´ng bá»�n, há»�ng sau thá»�i gian ngáº¯n.",
    "MÃ¹i hÃ³a cháº¥t khÃ³ chá»‹u tá»« váº£i má»›i.",
    "KhÃ´ng thoÃ¡ng khÃ­, nÃ³ng bá»©c khi máº·c.",
    "GiÃ¡ cao nhÆ°ng cháº¥t lÆ°á»£ng khÃ´ng xá»©ng Ä‘Ã¡ng.",
    "Dá»… nhÄƒn, khÃ³ á»§i pháº³ng.",
    "MÃ u sáº¯c khÃ´ng giá»‘ng hÃ¬nh áº£nh quáº£ng cÃ¡o.",
    "Phá»¥ kiá»‡n kÃ©m cháº¥t lÆ°á»£ng, dá»… há»�ng.",
    "KhÃ´ng phÃ¹ há»£p vá»›i thá»�i tiáº¿t Ä‘á»‹a phÆ°Æ¡ng.",
    "Kiá»ƒu dÃ¡ng khÃ´ng tÃ´n dÃ¡ng ngÆ°á»�i máº·c.",
    "Váº£i má»�ng, dá»… lá»™ ná»™i y.",
    "KhÃ³ giáº·t sáº¡ch váº¿t báº©n.",
    "Size khÃ´ng Ä‘a dáº¡ng, khÃ³ chá»�n.",
    "Sáº£n pháº©m lá»—i, cÃ³ váº¿t báº©n tá»« nhÃ  sáº£n xuáº¥t.",
    "KhÃ´ng giá»¯ form sau giáº·t."
]

neutral_feedbacks = [
    "Sáº£n pháº©m bÃ¬nh thÆ°á»�ng, khÃ´ng cÃ³ gÃ¬ Ä‘áº·c biá»‡t.",
    "Cháº¥t lÆ°á»£ng trung bÃ¬nh, dÃ¹ng táº¡m Ä‘Æ°á»£c.",
    "Thiáº¿t káº¿ Ä‘Æ¡n giáº£n, phÃ¹ há»£p cho máº·c hÃ ng ngÃ y.",
    "MÃ u sáº¯c á»•n, nhÆ°ng khÃ´ng ná»•i báº­t.",
    "Size vá»«a pháº£i, khÃ´ng quÃ¡ cháº­t hay rá»™ng.",
    "Váº£i khÃ¡, nhÆ°ng cÃ³ thá»ƒ tá»‘t hÆ¡n.",
    "GiÃ¡ cáº£ há»£p lÃ½ cho cháº¥t lÆ°á»£ng nÃ y.",
    "DÃ¹ng Ä‘Æ°á»£c, nhÆ°ng khÃ´ng áº¥n tÆ°á»£ng láº¯m.",
    "PhÃ¹ há»£p cho máº·c á»Ÿ nhÃ .",
    "KhÃ´ng tá»‡, nhÆ°ng cÅ©ng khÃ´ng xuáº¥t sáº¯c.",
    "Kiá»ƒu dÃ¡ng cÆ¡ báº£n, dá»… phá»‘i Ä‘á»“.",
    "MÃ u trung tÃ­nh, an toÃ n.",
    "Cháº¥t liá»‡u á»•n Ä‘á»‹nh, khÃ´ng thay Ä‘á»•i sau giáº·t.",
    "Sáº£n pháº©m nhÆ° mÃ´ táº£, khÃ´ng báº¥t ngá»�.",
    "DÃ¹ng táº¡m thá»�i, cÃ³ thá»ƒ thay tháº¿ sau.",
    "KhÃ´ng gÃ¢y dá»‹ á»©ng, nhÆ°ng khÃ´ng thoáº£i mÃ¡i láº¯m.",
    "Thiáº¿t káº¿ phá»• thÃ´ng, ai cÅ©ng máº·c Ä‘Æ°á»£c.",
    "GiÃ¡ ráº», cháº¥t lÆ°á»£ng tÆ°Æ¡ng xá»©ng.",
    "KhÃ´ng cÃ³ váº¥n Ä‘á»� gÃ¬ lá»›n.",
    "Sáº£n pháº©m trung láº­p, tÃ¹y sá»Ÿ thÃ­ch cÃ¡ nhÃ¢n."
]

# GÃ¡n ngáº«u nhiÃªn loáº¡i feedback cho má»—i transaction (interaction giá»¯a customer vÃ  product)
# Giáº£ sá»­ phÃ¢n bá»‘: 70% positive, 10% negative, 20% neutral Ä‘á»ƒ dá»¯ liá»‡u thiÃªn vá»� positive nhÆ° thá»±c táº¿
transactions['feedback_type'] = np.random.choice(['positive', 'negative', 'neutral'], size=len(transactions), p=[0.7, 0.1, 0.2])

# HÃ m Ä‘á»ƒ chá»�n ngáº«u nhiÃªn feedback dá»±a trÃªn loáº¡i
def get_feedback(row):
    if row['feedback_type'] == 'positive':
        return random.choice(positive_feedbacks)
    elif row['feedback_type'] == 'negative':
        return random.choice(negative_feedbacks)
    else:
        return random.choice(neutral_feedbacks)

# ThÃªm cá»™t feedback
transactions['feedback'] = transactions.apply(get_feedback, axis=1)

# HÃ m Ä‘á»ƒ chá»�n ngáº«u nhiÃªn rating dá»±a trÃªn loáº¡i
def get_rating(row):
    if row['feedback_type'] == 'positive':
        return round(random.uniform(3.5, 5.0), 1)
    elif row['feedback_type'] == 'negative':
        return round(random.uniform(0.0, 2.0), 1)
    else:
        return round(random.uniform(2.1, 3.4), 1)

# ThÃªm cá»™t rating
transactions['rating'] = transactions.apply(get_rating, axis=1)

# In má»™t sá»‘ máº«u Ä‘á»ƒ kiá»ƒm tra
print("\nMáº«u dá»¯ liá»‡u transactions sau khi thÃªm feedback vÃ  rating:")
print(transactions[['customer_id', 'article_id', 'feedback_type', 'feedback', 'rating']].head(10))

# Váº½ biá»ƒu Ä‘á»“ pie cho tá»· lá»‡ feedback_type
feedback_counts = transactions['feedback_type'].value_counts()
plt.figure(figsize=(6, 6))
feedback_counts.plot(kind='pie', autopct='%1.1f%%', colors=['#4CAF50', '#F44336', '#FFC107'], startangle=90)
plt.title('Feedback Type Ratio')
plt.ylabel('')  # áº¨n nhÃ£n y Ä‘á»ƒ biá»ƒu Ä‘á»“ sáº¡ch hÆ¡n
plt.show()

# Váº½ histogram cho phÃ¢n bá»‘ rating
plt.figure(figsize=(8, 6))
transactions['rating'].hist(bins=20, color='#2196F3', edgecolor='black')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Ä�á»�c dá»¯ liá»‡u articles.csv
articles = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv')

# Danh sÃ¡ch tá»« khÃ³a eco-friendly trong mÃ´ táº£ sáº£n pháº©m
eco_keywords = [
    'organic cotton', 'recycled polyester', 'relenya', 'recycled', 'organic',
    'sustainable', 'eco-friendly', 'ethical', 'environmentally friendly',
    'tencel', 'lyocell', 'hemp', 'bamboo', 'bio-based', 'recyclable',
    'low-impact', 'vegan', 'GOTS', 'OEKO-TEX', 'Fair Trade', 'modal'
]

# Danh sÃ¡ch mÃ u sáº¯c eco-friendly má»Ÿ rá»™ng
base_eco_colors = [
    'Beige', 'Light Beige', 'Grey', 'Light Grey', 'Dark Grey',
    'Green', 'Light Green', 'Dark Green', 'Greenish Khaki',
    'Brown', 'Yellowish Brown', 'Off White', 'White',
    'Light Blue', 'Blue', 'Turquoise', 'Light Turquoise',
    'Bronze/Copper'
]

additional_eco_colors = ['Beige', 'Khaki', 'Olive Green', 'Brown', 'Off White', 'Taupe', 'Natural White']
eco_colors = base_eco_colors + additional_eco_colors

# HÃ m kiá»ƒm tra sáº£n pháº©m eco-friendly dá»±a trÃªn detail_desc
def is_eco_friendly(description):
    if pd.isna(description):
        return False
    return any(keyword in str(description).lower() for keyword in eco_keywords)

# Ä�Ã¡nh dáº¥u sáº£n pháº©m eco-friendly dá»±a trÃªn detail_desc
articles['is_eco_desc'] = articles['detail_desc'].apply(is_eco_friendly)

# Ä�Ã¡nh dáº¥u sáº£n pháº©m eco-friendly dá»±a trÃªn mÃ u sáº¯c
articles['is_eco_color'] = articles['colour_group_name'].isin(eco_colors)

# Káº¿t há»£p: sáº£n pháº©m eco-friendly náº¿u thá»�a mÃ£n detail_desc HOáº¶C mÃ u sáº¯c
articles['is_eco'] = articles['is_eco_desc'] | articles['is_eco_color']

# Lá»�c cÃ¡c sáº£n pháº©m eco-friendly
eco_products = articles[articles['is_eco']]
non_eco_products = articles[~articles['is_eco']]

# Tá»•ng há»£p sá»‘ lÆ°á»£ng sáº£n pháº©m
print(f"Tá»•ng sá»‘ sáº£n pháº©m eco-friendly: {len(eco_products)}")
print(f"Sá»‘ sáº£n pháº©m eco-friendly tá»« detail_desc: {articles['is_eco_desc'].sum()}")
print(f"Sá»‘ sáº£n pháº©m eco-friendly tá»« mÃ u sáº¯c: {articles['is_eco_color'].sum()}")
print(f"Sá»‘ sáº£n pháº©m khÃ´ng eco-friendly: {len(non_eco_products)}")

# PhÃ¢n bá»‘ mÃ u sáº¯c trong sáº£n pháº©m eco-friendly
eco_color_dist = eco_products['colour_group_name'].value_counts(normalize=True).head(10)
print("\nTop 10 mÃ u sáº¯c phá»• biáº¿n trong sáº£n pháº©m eco-friendly (tá»· lá»‡):")
print(eco_color_dist)

# PhÃ¢n bá»‘ giÃ¡ trá»‹ mÃ u sáº¯c cáº£m nháº­n trong sáº£n pháº©m eco-friendly
eco_perceived_color_dist = eco_products['perceived_colour_value_name'].value_counts(normalize=True).head(10)
print("\nTop 10 giÃ¡ trá»‹ mÃ u sáº¯c cáº£m nháº­n trong sáº£n pháº©m eco-friendly (tá»· lá»‡):")
print(eco_perceived_color_dist)

# Váº½ biá»ƒu Ä‘á»“ pie cho tá»· lá»‡ sáº£n pháº©m eco-friendly
eco_counts = pd.Series([len(eco_products), len(non_eco_products)], index=['Eco-Friendly', 'Non Eco-Friendly'])
plt.figure(figsize=(6, 6))
eco_counts.plot(kind='pie', autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=90)
plt.title('Eco-Friendly Product Ratio')
plt.ylabel('')  # áº¨n nhÃ£n y Ä‘á»ƒ biá»ƒu Ä‘á»“ sáº¡ch hÆ¡n
plt.show()


import pandas as pd

# Ä�á»�c dá»¯ liá»‡u articles.csv
articles = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv')

# Danh sÃ¡ch tá»« khÃ³a eco-friendly trong mÃ´ táº£ sáº£n pháº©m
eco_keywords = [
    'organic cotton', 'recycled polyester', 'relenya', 'recycled', 'organic',
    'sustainable', 'eco-friendly', 'ethical', 'environmentally friendly',
    'tencel', 'lyocell', 'hemp', 'bamboo', 'bio-based', 'recyclable',
    'low-impact', 'vegan', 'GOTS', 'OEKO-TEX', 'Fair Trade', 'modal'
]

# Danh sÃ¡ch mÃ u sáº¯c eco-friendly má»Ÿ rá»™ng
base_eco_colors = [
    'Beige', 'Light Beige', 'Grey', 'Light Grey', 'Dark Grey',
    'Green', 'Light Green', 'Dark Green', 'Greenish Khaki',
    'Brown', 'Yellowish Brown', 'Off White', 'White',
    'Light Blue', 'Blue', 'Turquoise', 'Light Turquoise',
    'Bronze/Copper'
]
additional_eco_colors = ['Beige', 'Khaki', 'Olive Green', 'Brown', 'Off White', 'Taupe', 'Natural White']
eco_colors = base_eco_colors + additional_eco_colors

# HÃ m kiá»ƒm tra sáº£n pháº©m eco-friendly dá»±a trÃªn detail_desc
def is_eco_friendly(description):
    if pd.isna(description):
        return False
    return any(keyword in str(description).lower() for keyword in eco_keywords)

# Ä�Ã¡nh dáº¥u sáº£n pháº©m eco-friendly dá»±a trÃªn detail_desc
articles['is_eco_desc'] = articles['detail_desc'].apply(is_eco_friendly)

# Ä�Ã¡nh dáº¥u sáº£n pháº©m eco-friendly dá»±a trÃªn mÃ u sáº¯c
articles['is_eco_color'] = articles['colour_group_name'].isin(eco_colors)

# Káº¿t há»£p: sáº£n pháº©m eco-friendly náº¿u thá»�a mÃ£n detail_desc HOáº¶C mÃ u sáº¯c
articles['is_eco'] = articles['is_eco_desc'] | articles['is_eco_color']

# ThÃªm cá»™t label: True náº¿u is_eco lÃ  True (eco-friendly), False náº¿u khÃ´ng
articles['eco_label'] = articles['is_eco']  # Giá»¯ nguyÃªn boolean True/False

# In má»™t sá»‘ máº«u Ä‘á»ƒ kiá»ƒm tra
print("\nMáº«u dá»¯ liá»‡u articles sau khi thÃªm label:")
print(articles[['article_id', 'is_eco', 'eco_label']].head(10))

# LÆ°u dataframe articles vá»›i cá»™t má»›i vÃ o file CSV má»›i
# Sá»­ dá»¥ng Ä‘Æ°á»�ng dáº«n /kaggle/working/ Ä‘á»ƒ lÆ°u trong Kaggle (cÃ³ thá»ƒ táº£i xuá»‘ng sau)
articles.to_csv('/kaggle/working/articles_with_eco_label.csv', index=False)

print("\nFile Ä‘Ã£ Ä‘Æ°á»£c lÆ°u táº¡i /kaggle/working/articles_with_eco_label.csv")


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Merge táº¥t cáº£ Ä‘á»ƒ cÃ³ dataframe Ä‘áº§y Ä‘á»§
merged_df = pd.merge(transactions, articles[['article_id', 'eco_label']], on='article_id', how='left')

# labels cÃ³ index lÃ  customer_id, cáº§n reset_index náº¿u chÆ°a
if 'customer_id' not in labels.columns:
    labels = labels.reset_index()

merged_df = pd.merge(merged_df, labels[['customer_id', 'final_gender']], on='customer_id', how='left')

# Chuyá»ƒn t_dat sang datetime náº¿u chÆ°a
merged_df['t_dat'] = pd.to_datetime(merged_df['t_dat'])

# Sort theo t_dat tÄƒng dáº§n
merged_df = merged_df.sort_values(by='t_dat').reset_index(drop=True)

# TÃ­nh Ä‘iá»ƒm cáº¯t cho 80% train, 20% test
split_index = int(len(merged_df) * 0.8)
train_data = merged_df.iloc[:split_index]
test_data = merged_df.iloc[split_index:]

# LÆ°u vÃ o file
train_data.to_csv('/kaggle/working/train_data_new.csv', index=False)
test_data.to_csv('/kaggle/working/test_data_new.csv', index=False)

# In cáº¥u trÃºc
print("Cáº¥u trÃºc cá»§a train_data:")
print("Shape:", train_data.shape)
print("Columns:", train_data.columns.tolist())
print("Head:\n", train_data.head())

print("\nCáº¥u trÃºc cá»§a test_data:")
print("Shape:", test_data.shape)
print("Columns:", test_data.columns.tolist())
print("Head:\n", test_data.head())

# Váº½ biá»ƒu Ä‘á»“ cá»™t cho sá»‘ lÆ°á»£ng dá»¯ liá»‡u train vÃ  test
data_sizes = pd.Series([len(train_data), len(test_data)], index=['Train', 'Test'])
plt.figure(figsize=(6, 4))
data_sizes.plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
plt.title('Number of Transactions in Train and Test Data')
plt.xlabel('Dataset')
plt.ylabel('Count')
plt.show()


import pandas as pd

# Ä�á»�c file train & test
train_data = pd.read_csv('/kaggle/input/chiadulieu8020new/train_data_new.csv')
test_data = pd.read_csv('/kaggle/input/chiadulieu8020new/test_data_new.csv')


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


# Ä�á»�c dá»¯ liá»‡u Kaggle
train_data = pd.read_csv("/kaggle/input/chiadulieu8020new/train_data_new.csv")
test_data = pd.read_csv("/kaggle/input/chiadulieu8020new/test_data_new.csv")
train_sample_size = 5000
test_sample_size = 5000
train_data = train_data.sample(n=train_sample_size, random_state=42)
test_data = test_data.sample(n=test_sample_size, random_state=42)

def preprocess(df):
    df = df.copy()
    # 1) t_dat -> Ä‘áº·c trÆ°ng sá»‘
    df["t_dat"] = pd.to_datetime(df["t_dat"], errors="coerce")
    df["month"] = df["t_dat"].dt.month
    df["dayofweek"] = df["t_dat"].dt.dayofweek
    df["year"] = df["t_dat"].dt.year  # Added year feature

    # 2) feedback_type -> 0/1
    df["feedback_type"] = df["feedback_type"].map({"positive": 1, "negative": 0}).fillna(0)

    # 3) eco_label bool/chuá»—i -> 0/1
    if df["eco_label"].dtype == bool:
        df["eco_label"] = df["eco_label"].astype(int)
    else:
        df["eco_label"] = df["eco_label"].map({"True": 1, "False": 0}).fillna(0).astype(int)

    # 4) Bá»� cÃ¡c cá»™t khÃ´ng dÃ¹ng/khÃ´ng pháº£i sá»‘
    df = df.drop(columns=["t_dat", "customer_id", "feedback"], errors="ignore")

    # (tuá»³ chá»�n) Ä‘áº£m báº£o article_id lÃ  sá»‘
    df["article_id"] = pd.to_numeric(df["article_id"], errors="coerce")

    # Ä�iá»�n thiáº¿u
    df = df.fillna(df.median(numeric_only=True), inplace=False)  # Use median for numerical columns
    return df

train_pp = preprocess(train_data)
test_pp = preprocess(test_data)

# Táº¡o X, y chá»‰ tá»« cá»™t sá»‘
feature_columns = [c for c in train_pp.columns if c != "final_gender"]
non_num = train_pp[feature_columns].select_dtypes(exclude=[np.number]).columns.tolist()
if non_num:
    print(f"Warning: Non-numeric columns found: {non_num}")

X_train = train_pp[feature_columns].to_numpy(dtype=np.float32)
X_test = test_pp[feature_columns].to_numpy(dtype=np.float32)
y_train = train_pp["final_gender"]
y_test = test_pp["final_gender"]

# Chuáº©n hÃ³a dá»¯ liá»‡u
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Chuyá»ƒn thÃ nh ma tráº­n vuÃ´ng Ä‘á»ƒ dÃ¹ng EfficientNet
img_size = int(np.ceil(np.sqrt(X_train.shape[1])))
img_size = max(img_size, 32)

pad_len = img_size**2 - X_train.shape[1]
if pad_len > 0:
    X_train_padded = np.pad(X_train, ((0, 0), (0, pad_len)), 'constant')
    X_test_padded = np.pad(X_test, ((0, 0), (0, pad_len)), 'constant')
else:
    X_train_padded, X_test_padded = X_train, X_test

# Reshape thÃ nh áº£nh
X_train_images = X_train_padded.reshape(-1, img_size, img_size, 1)
X_test_images = X_test_padded.reshape(-1, img_size, img_size, 1)
print("âœ… Shape dá»¯ liá»‡u áº£nh train:", X_train_images.shape)
print("âœ… Shape dá»¯ liá»‡u áº£nh test:", X_test_images.shape)

# MÃ£ hÃ³a nhÃ£n
encoder = LabelEncoder()
all_labels = pd.concat([train_data["final_gender"], test_data["final_gender"]]).unique()
encoder.fit(all_labels)
y_train_encoded = encoder.transform(y_train)
y_test_encoded = encoder.transform(y_test)
y_train_categorical = to_categorical(y_train_encoded)
y_test_categorical = to_categorical(y_test_encoded)

# XÃ¢y dá»±ng mÃ´ hÃ¬nh
inputs = layers.Input(shape=(img_size, img_size, 1))
x = layers.Conv2D(3, (3, 3), padding='same', activation='relu')(inputs)
base_model = EfficientNetB0(weights=None, include_top=False, input_tensor=x)
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(y_train_categorical.shape[1], activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Huáº¥n luyá»‡n mÃ´ hÃ¬nh
history = model.fit(
    X_train_images, y_train_categorical,
    validation_data=(X_test_images, y_test_categorical),
    epochs=10,  # Increased epochs, relying on early stopping
    batch_size=64,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# Ä�Ã¡nh giÃ¡ mÃ´ hÃ¬nh
val_loss, val_acc = model.evaluate(X_test_images, y_test_categorical, verbose=1)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# BÃ¡o cÃ¡o phÃ¢n loáº¡i
from sklearn.metrics import classification_report
y_pred = model.predict(X_test_images)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(y_test_encoded, y_pred_classes, target_names=encoder.classes_))

# Váº½ biá»ƒu Ä‘á»“ accuracy vÃ  loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.show()

# Váº½ confusion matrix
cm = confusion_matrix(y_test_encoded, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Váº½ biá»ƒu Ä‘á»“ precision, recall, f1-score
report = classification_report(y_test_encoded, y_pred_classes, target_names=encoder.classes_, output_dict=True)
metrics_df = pd.DataFrame(report).transpose().drop(['support'], axis=1)[:-3]  # Loáº¡i bá»� accuracy, macro avg, weighted avg
metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title('Precision, Recall, F1-Score per Class')
plt.ylabel('Score')
plt.show()


import pandas as pd
merged = pd.read_csv('/kaggle/input/traintestlan2new/train_kmeans_kmedoids.csv')
print(merged.head())
print(merged.shape)


import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# BÆ°á»›c 2: Kiá»ƒm tra vÃ  lÃ m sáº¡ch dá»¯ liá»‡u
required_columns = ['customer_id', 'article_id', 'rating', 'season', 'product_group_name']
optional_columns = ['user_season_cluster', 'final_gender', 'eco_label']
all_columns = required_columns + [col for col in optional_columns if col in merged.columns]
print("Sá»‘ hÃ ng trong merged:", len(merged))
print("Sample merged data:\n", merged[all_columns].head())
if len(merged) == 0:
    raise ValueError("DataFrame merged is empty. Check data loading or merging process.")
if not all(col in merged.columns for col in required_columns):
    raise ValueError(f"Missing required columns. Required: {required_columns}, Found: {merged.columns}")
if merged[required_columns].isna().any().any():
    print("Warning: NaN values found in required columns. Filling with defaults.")
    merged['rating'] = merged['rating'].fillna(0)
    merged['season'] = merged['season'].fillna(0)  # Sá»­a tá»« 'unknown' thÃ nh 0 cho consistency
# BÆ°á»›c 3: Lá»�c top 1000 ngÆ°á»�i dÃ¹ng
user_freq = merged['customer_id'].value_counts()
top_users = user_freq.head(5000).index
merged_subset = merged[merged['customer_id'].isin(top_users)]
train_data = merged_subset  # KhÃ´ng chia train/test vÃ¬ chá»‰ huáº¥n luyá»‡n
print("Sá»‘ hÃ ng trong train_data:", len(train_data))
print("Sá»‘ user trong train_data:", train_data['customer_id'].nunique())
print("Sá»‘ article trong train_data:", train_data['article_id'].nunique())
if len(merged_subset) == 0:
    raise ValueError("merged_subset is empty. Check customer_id filtering or data integrity.")


# BÆ°á»›c 4: Táº¡o ma tráº­n user-item
user_item_matrix_train = train_data.pivot_table(
    index='customer_id',
    columns='article_id',
    values='rating',
    aggfunc='sum'
).fillna(0)

user_item_values_train = csr_matrix(user_item_matrix_train.values)
print("Ma tráº­n user-item (train):", user_item_values_train.shape)  


# TÃ­nh Ä‘á»™ thÆ°a thá»›t
sparsity = 1 - np.count_nonzero(user_item_matrix_train.values) / user_item_matrix_train.values.size
print("Ä�á»™ thÆ°a thá»›t cá»§a user_item_matrix_train:", sparsity)

if user_item_matrix_train.shape[0] == 0 or user_item_matrix_train.shape[1] == 0:
    raise ValueError("user_item_matrix_train is empty.")


# BÆ°á»›c 5: TÃ­nh similarity user-user
user_similarity_train = cosine_similarity(user_item_values_train)
print("Shape ma tráº­n similarity (train):", user_similarity_train.shape)


# BÆ°á»›c 6: Dá»± Ä‘oÃ¡n ratings
ratings_train = user_item_values_train.toarray()
num_users, num_items = ratings_train.shape
predicted_ratings = np.dot(user_similarity_train, ratings_train) / np.sum(np.abs(user_similarity_train), axis=1, keepdims=True)
predicted_ratings[np.isnan(predicted_ratings)] = 0
print("Dá»± Ä‘oÃ¡n ma tráº­n ratings hoÃ n táº¥t:", predicted_ratings.shape)


# BÆ°á»›c 7: Táº¡o user_fav_cluster
user_fav_cluster = {}
current_season = 0
for user_id in user_item_matrix_train.index:
    bought_items = user_item_matrix_train.loc[user_id]
    bought_items = bought_items[bought_items > 0].index.tolist()
    if bought_items:
        clusters = train_data[train_data['article_id'].isin(bought_items)]['season']
        if not clusters.empty:
            user_fav_cluster[user_id] = clusters.mode().iloc[0]
        else:
            user_fav_cluster[user_id] = current_season
    else:
        user_cluster = train_data[train_data['customer_id'] == user_id]['user_season_cluster']
        if not user_cluster.empty:
            user_fav_cluster[user_id] = user_cluster.mode().iloc[0]
        else:
            user_fav_cluster[user_id] = current_season
# Debug: Kiá»ƒm tra user_fav_cluster
print("Sá»‘ user cÃ³ fav_cluster:", len(user_fav_cluster))
print("Sample user_fav_cluster:", list(user_fav_cluster.items())[:5])


# BÆ°á»›c 8: Táº¡o mapping
item_cluster_map = dict(zip(train_data['article_id'], train_data['season']))
item_gender_map = dict(zip(train_data['article_id'], train_data['final_gender']))
item_eco_map = dict(zip(train_data['article_id'], train_data['eco_label']))
item_product_group_map = dict(zip(train_data['article_id'], train_data['product_group_name']))

missing_items = set(user_item_matrix_train.columns) - set(item_cluster_map.keys())
if missing_items:
    print(f"Warning: {len(missing_items)} article_id(s) missing seasonr")
    for item_id in missing_items:
        item_cluster_map[item_id] = current_season
print("Sá»‘ article_id cÃ³ season_cluster:", len(item_cluster_map))
print("Sample item_cluster_map:", list(item_cluster_map.items())[:5])
print("Sá»‘ article_id cÃ³ product_group_name:", len(item_product_group_map))
print("Sample item_product_group_map:", list(item_product_group_map.items())[:5])


# BÆ°á»›c 9: Táº¡o recommendation
recommendations = {}
skipped_users = 0
empty_filtered_items = 0
user_gender_map = {}
for user_idx, user_id in enumerate(user_item_matrix_train.index):
    fav_cluster = user_fav_cluster.get(user_id, current_season)
    if fav_cluster is None:
        skipped_users += 1
        continue
    user_gender = user_gender_map.get(user_id, None)
    preds = predicted_ratings[user_idx, :]
    item_ids = user_item_matrix_train.columns
    item_mask = np.array([
        item_cluster_map.get(iid, current_season) == fav_cluster
        for iid in item_ids
    ])
    filtered_indices = np.where(item_mask)[0]
    pred_items_filtered = [
        (item_ids[i], preds[i] * (1.2 if item_eco_map.get(item_ids[i], False) else 1.0))
        for i in filtered_indices
    ]
    if not pred_items_filtered:
        empty_filtered_items += 1
        popular_items = train_data[
            (train_data['season'] == fav_cluster) &
            (train_data['eco_label'] == True)
        ]['article_id'].value_counts().head(5).index
        if not popular_items.empty:
            pred_items_filtered = [(iid, 0.0) for iid in popular_items]
        else:
            popular_items = train_data[
                (train_data['season'] == fav_cluster)
            ]['article_id'].value_counts().head(5).index
            if not popular_items.empty:
                pred_items_filtered = [(iid, 0.0) for iid in popular_items]
            else:
                popular_items = train_data['article_id'].value_counts().head(5).index
                pred_items_filtered = [(iid, 0.0) for iid in popular_items]
    pred_items_filtered.sort(key=lambda x: x[1], reverse=True)
    recommendations[user_id] = pred_items_filtered[:5]
print("Táº¡o xong recommendation hybrid cho", len(recommendations), "users.")
print("Sá»‘ user bá»‹ bá»� qua (fav_cluster is None):", skipped_users)
print("Sá»‘ user cÃ³ pred_items_filtered rá»—ng:", empty_filtered_items)


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
test_data = pd.read_csv('/kaggle/input/traintestlan2/test_kmeans_kmedoids.csv')
if len(test_data) > 5000:
    test_data = test_data[test_data['customer_id'].isin(top_users)].sample(n=5000, random_state=42)
else:
    print(f"Warning: Sá»‘ hÃ ng trong test_data ({len(test_data)}) nhá»� hÆ¡n 5,000. Sá»­ dá»¥ng toÃ n bá»™ dá»¯ liá»‡u.")
required_columns = ['customer_id', 'article_id', 'rating', 'season', 'product_group_name']
optional_columns = ['user_season_cluster', 'final_gender', 'eco_label']
all_columns = required_columns + [col for col in optional_columns if col in test_data.columns]
user_item_matrix_test = test_data.pivot_table(
    index='customer_id',
    columns='article_id',
    values='rating',
    aggfunc='sum'
).fillna(0)
common_users = user_item_matrix_train.index.intersection(user_item_matrix_test.index)
common_items = user_item_matrix_train.columns.intersection(user_item_matrix_test.columns)
print("Shape cá»§a predicted_ratings:", predicted_ratings.shape)
print("Sá»‘ common_users:", len(common_users))
print("Sá»‘ common_items:", len(common_items))
print("Shape cá»§a user_item_matrix_train:", user_item_matrix_train.shape)
print("Shape cá»§a user_item_matrix_test:", user_item_matrix_test.shape)
if len(common_users) > 0 and len(common_items) > 0:
    # Láº¥y true ratings vÃ  predicted ratings
    true_ratings = user_item_matrix_test.loc[common_users, common_items].values
    user_indices = user_item_matrix_train.index.get_indexer(common_users)
    item_indices = user_item_matrix_train.columns.get_indexer(common_items)
    # Kiá»ƒm tra chá»‰ sá»‘ há»£p lá»‡ riÃªng láº»
    valid_user_mask = user_indices != -1
    valid_item_mask = item_indices != -1
    if not np.all(valid_user_mask) or not np.all(valid_item_mask):
        print("Warning: Some users or items not found in user_item_matrix_train.")
        valid_indices = valid_user_mask & (valid_item_mask[:len(valid_user_mask)])
        common_users = common_users[valid_indices]
        common_items = common_items[valid_indices[:len(common_items)]]
        user_indices = user_indices[valid_indices]
        item_indices = item_indices[valid_indices[:len(item_indices)]]
        true_ratings = user_item_matrix_test.loc[common_users, common_items].values
    if len(user_indices) == 0 or len(item_indices) == 0:
        print("Error: No valid indices after filtering. Cannot compute RMSE.")
    else:
        try:
            pred_ratings = predicted_ratings[user_indices][:, item_indices]
            print("Shape cá»§a true_ratings:", true_ratings.shape)
            print("Shape cá»§a pred_ratings:", pred_ratings.shape)      
            if true_ratings.shape != pred_ratings.shape:
                print("Error: Shape mismatch between true_ratings and pred_ratings.")
            else:
              mask = true_ratings > 0
              if np.sum(mask) > 0:
                    rmse = np.sqrt(mean_squared_error(true_ratings[mask], pred_ratings[mask]))
                    print("RMSE on test set:", rmse)
              else:
                    print("Warning: No non-zero true ratings to compute RMSE.")
        except IndexError as e:
            print(f"IndexError: {e}")
            print("Cannot compute RMSE due to indexing issues.")


print(test_data.head())
print(test_data.shape)


from sklearn.metrics import mean_absolute_error

if np.sum(mask) > 0:
    mae = mean_absolute_error(true_ratings[mask], pred_ratings[mask])
    print("MAE on test set:", mae)


def evaluate_recommendations(recommendations, test_data, k=5):
    precision = recall = f1 = ap_sum = total = 0
    for user_id in recommendations:
        if user_id in test_data['customer_id'].values:
            rec_items = [r[0] for r in recommendations[user_id][:k]]
            true_items = test_data[test_data['customer_id'] == user_id]['article_id'].tolist()
            if not rec_items or not true_items:
                continue
            hits = len(set(rec_items).intersection(true_items))
            if hits > 0:
                print(f"User {user_id}: rec_items={rec_items}, true_items={true_items}, hits={hits}")
                precision += hits / k
                recall += hits / len(true_items)
                ap = 0
                relevant_count = 0
                for i, item in enumerate(rec_items[:k], 1):
                    if item in true_items:
                        relevant_count += 1
                        ap += relevant_count / i
                ap = ap / min(len(true_items), k)
                ap_sum += ap
                total += 1
    precision = precision / total if total > 0 else 0
    recall = recall / total if total > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    map_score = ap_sum / total if total > 0 else 0
    print(f"Debug - Total users evaluated: {total}")
    return {'Precision@5': precision, 'Recall@5': recall, 'F1-Score@5': f1, 'MAP@5': map_score}
print('evaluate_recommendations',evaluate_recommendations(recommendations, test_data, k=5))


def recommendation1(recommendations, test_data, k=5):
    precision = 0
    recall = 0
    f1 = 0
    ap_sum = 0
    total = 0    
    for user_id in recommendations:
        if user_id in test_data['customer_id'].values:
            rec_items = [r[0] for r in recommendations[user_id][:k]]
            true_items = test_data[test_data['customer_id'] == user_id]['article_id'].tolist()
            # TÃ­nh Precision@5
            hits = len(set(rec_items).intersection(true_items))
            precision += hits / k if k > 0 else 0
            # TÃ­nh Recall@5
            recall += hits / len(true_items) if len(true_items) > 0 else 0
            # TÃ­nh AP@5 (Average Precision)
            ap = 0
            relevant_count = 0
            for i, item in enumerate(rec_items[:k], 1):
                if item in true_items:
                    relevant_count += 1
                    ap += relevant_count / i
            ap = ap / min(len(true_items), k) if len(true_items) > 0 else 0
            ap_sum += ap
            total += 1
    precision = precision / total if total > 0 else 0
    recall = recall / total if total > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    map_score = ap_sum / total if total > 0 else 0
    
    return {
        'Precision@5': precision,
        'Recall@5': recall,
        'F1-Score@5': f1,
        'MAP@5': map_score
    }
eval_metrics = evaluate_recommendations(recommendations, test_data, k=5)
print("Evaluation metrics:")
for metric, value in eval_metrics.items():
    print(f"{metric}: {value}")


from sklearn.metrics import ndcg_score
def calculate_ndcg(recommendations, test_data, k=5):
    ndcg_sum = 0
    total = 0
    for user_id in recommendations:
        if user_id in test_data['customer_id'].values:
            rec_items = [r[0] for r in recommendations[user_id][:k]]  # Top k gá»£i Ã½
            # Láº¥y true_items vÃ  true_ratings tÆ°Æ¡ng á»©ng
            user_data = test_data[test_data['customer_id'] == user_id]
            true_items = user_data['article_id'].tolist()
            true_ratings = user_data['rating'].tolist()  # Sá»­ dá»¥ng cá»™t 'ratings'
            
            if not rec_items or not true_items or not true_ratings:
                continue
                
            # Táº¡o máº£ng relevance dá»±a trÃªn ratings cá»§a true_items
            relevance_scores = []
            for item in rec_items:
                if item in true_items:
                    idx = true_items.index(item)
                    relevance_scores.append(true_ratings[idx] if idx < len(true_ratings) else 0)
                else:
                    relevance_scores.append(0)  # Náº¿u item khÃ´ng trong true_items, gÃ¡n 0
                    
            # Ä�áº£m báº£o Ä‘á»™ dÃ i khá»›p
            if len(relevance_scores) != k:
                relevance_scores = relevance_scores[:k] + [0] * (k - len(relevance_scores))
                
            if sum(relevance_scores) > 0:  # Chá»‰ tÃ­nh náº¿u cÃ³ ratings > 0
                ndcg_sum += ndcg_score([relevance_scores], [relevance_scores])  # So sÃ¡nh vá»›i chÃ­nh nÃ³ Ä‘á»ƒ kiá»ƒm tra
                total += 1
    return ndcg_sum / total if total > 0 else 0

# TÃ­nh vÃ  in NDCG@5
ndcg = calculate_ndcg(recommendations, test_data, k=5)
print(f"NDCG@5: {ndcg}")


from sklearn.metrics import ndcg_score

def calculate_ndcg1(recommendations, test_data, k=5):
    ndcg_sum = 0
    total = 0
    for user_id in recommendations:
        if user_id in test_data['customer_id'].values:
            rec_items = [r[0] for r in recommendations[user_id][:k]]
            user_data = test_data[test_data['customer_id'] == user_id]
            true_items = user_data['article_id'].tolist()
            true_ratings = user_data['rating'].tolist()

            if not rec_items or not true_items or not true_ratings:
                continue

            relevance_scores = []
            ideal_scores = []
            for item in rec_items:
                if item in true_items:
                    idx = true_items.index(item)
                    relevance_scores.append(true_ratings[idx])
                else:
                    relevance_scores.append(0)

            # Ideal ranking: sort true ratings in descending order
            ideal_scores = sorted(relevance_scores, reverse=True)

            if sum(ideal_scores) > 0:
                ndcg_sum += ndcg_score([ideal_scores], [relevance_scores])
                total += 1

    return ndcg_sum / total if total > 0 else 0
ndcg1 = calculate_ndcg1(recommendations, test_data, k=5)
print(f"NDCG@5: {ndcg1}")


def calculate_hit_rate(recommendations, test_data, k=5):
    hits = 0
    total_users = 0
    for user_id in recommendations:
        if user_id in test_data['customer_id'].values:
            rec_items = [r[0] for r in recommendations[user_id][:k]]
            true_items = test_data[test_data['customer_id'] == user_id]['article_id'].tolist()
            if not rec_items or not true_items:
                continue
            if any(item in true_items for item in rec_items):
                hits += 1
            total_users += 1
    return hits / total_users if total_users > 0 else 0
hit_rate = calculate_hit_rate(recommendations, test_data, k=5)
print(f"Hit Rate@5: {hit_rate}")


def calculate_coverage(recommendations, test_data):
    all_rec_items = set()
    for user_id in recommendations:
        all_rec_items.update([r[0] for r in recommendations[user_id]])
    total_items = set(test_data['article_id'].unique())
    return len(all_rec_items) / len(total_items) if len(total_items) > 0 else 0
coverage = calculate_coverage(recommendations, test_data)
print(f"Coverage: {coverage}")


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# 1. Chuáº©n bá»‹ dá»¯ liá»‡u features
train_data = train_data.copy()
train_data['features'] = (
    train_data['product_group_name'].fillna('') + ' ' +
    train_data['final_gender'].fillna('') + ' ' +
    train_data['season'].fillna('') + ' ' +
    train_data['feedback'].fillna('')
)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
tfidf_matrix = vectorizer.fit_transform(train_data['features'])

# Mapping article_id -> index
article_id_to_idx = {aid: idx for idx, aid in enumerate(train_data['article_id'])}

# Nearest Neighbors (tÃ¬m top N nhanh)
nn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
nn_model.fit(tfidf_matrix)



cbf_cache = {}

def get_cbf_recommendations_fast(article_id, top_n=10):
    if article_id in cbf_cache:
        return cbf_cache[article_id]
    if article_id not in article_id_to_idx:
        return []

    idx = article_id_to_idx[article_id]
    distances, indices = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n+1)  # +1 vÃ¬ cÃ³ chÃ­nh nÃ³
    recs = [(train_data.iloc[i]['article_id'], 1 - distances[0][j])
            for j, i in enumerate(indices[0]) if i != idx]

    cbf_cache[article_id] = recs
    return recs



def evaluate_precision_recall(customer_id, k=5, recommend_func=None):
    true_items = set(test_data[test_data['customer_id'] == customer_id]['article_id'])
    if not true_items:
        return None, None

    start_items = train_data[train_data['customer_id'] == customer_id]['article_id'].tolist()
    if not start_items:
        return None, None
    start_item = start_items[0]

    recs = recommend_func(start_item, top_n=k)
    recommended_items = set(aid for aid, score in recs)  # chá»‰ láº¥y article_id
    if not recommended_items:
        return None, None

    precision = len(recommended_items & true_items) / k
    recall = len(recommended_items & true_items) / len(true_items)
    return precision, recall



sample_users = test_data['customer_id'].unique()[:5000]

precisions, recalls = [], []
for cid in sample_users:
    p, r = evaluate_precision_recall(cid, k=5, recommend_func=get_cbf_recommendations_fast)
    if p is not None:
        precisions.append(p)
        recalls.append(r)

print("Average Precision@5:", np.mean(precisions))
print("Average Recall@5:", np.mean(recalls))



import matplotlib.pyplot as plt
import numpy as np

# === Giáº£ sá»­ Ä‘Ã¢y lÃ  káº¿t quáº£ báº¡n Ä‘o Ä‘Æ°á»£c ===
metrics = ['Precision@5', 'Recall@5', 'MAP@5']
cbf_scores = [0.0, 0.0, 0.0]   # CBF result
cf_scores = [0.27, 0.31, 0.25] # CF result

x = np.arange(len(metrics))  # vá»‹ trÃ­ cá»™t
width = 0.35                 # Ä‘á»™ rá»™ng cá»™t

fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x - width/2, cbf_scores, width, label='CBF')
bars2 = ax.bar(x + width/2, cf_scores, width, label='CF')

# ThÃªm nhÃ£n, tiÃªu Ä‘á»�
ax.set_ylabel('Score')
ax.set_title('Comparison of CF vs CBF')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.legend()

# Ghi giÃ¡ trá»‹ trÃªn Ä‘áº§u cá»™t
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # offset lÃªn trÃªn 3 pixel
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# New: Precompute set of article_ids that have images
image_dir = '/kaggle/input/h-and-m-personalized-fashion-recommendations/images'
existing_articles = set()
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('.jpg'):
            article_str = file[:-4]  # Remove .jpg
            try:
                existing_articles.add(int(article_str))
            except ValueError:
                pass  # Skip any invalid filenames

print(f"Found {len(existing_articles)} articles with images.")

# ... (continue with your code for user_fav_cluster, item maps, etc.)

# Modified recommendations loop to aim for at least 12 items with images
recommendations = {}
skipped_users = 0
empty_filtered_items = 0
user_gender_map = {}  # Assuming this is defined elsewhere if needed

for user_idx, user_id in enumerate(user_item_matrix_train.index):
    fav_cluster = user_fav_cluster.get(user_id, current_season)
    if fav_cluster is None:
        skipped_users += 1
        continue
    
    user_gender = user_gender_map.get(user_id, None)
    preds = predicted_ratings[user_idx, :]
    item_ids = user_item_matrix_train.columns
    
    # Lá»�c sáº£n pháº©m vá»›i additional check for existing image
    item_mask = np.array([
        item_cluster_map.get(iid, current_season) == fav_cluster and iid in existing_articles
        for iid in item_ids
    ])
    filtered_indices = np.where(item_mask)[0]
    pred_items_filtered = [
        (item_ids[i], preds[i] * (1.2 if item_eco_map.get(item_ids[i], False) else 1.0))
        for i in filtered_indices
    ]
    
    # Sáº¯p xáº¿p theo score descending
    pred_items_filtered.sort(key=lambda x: x[1], reverse=True)
    
    # Náº¿u Ã­t hÆ¡n 12, bá»• sung popular items trong cluster, Æ°u tiÃªn eco, chá»‰ nhá»¯ng cÃ³ áº£nh
    current_recs = pred_items_filtered[:12]  # Láº¥y top 12 náº¿u cÃ³ nhiá»�u
    if len(current_recs) < 12:
        needed = 12 - len(current_recs)
        already = {iid for iid, _ in current_recs}
        
        # Popular eco trong cluster, cÃ³ áº£nh, khÃ´ng trÃ¹ng
        popular_eco = train_data[
            (train_data['season_cluster'] == fav_cluster) &
            (train_data['eco_label'] == True) &
            (train_data['article_id'].isin(existing_articles)) &
            (~train_data['article_id'].isin(already))
        ]['article_id'].value_counts().index[:needed]
        current_recs += [(iid, 0.0) for iid in popular_eco]
        
        # Náº¿u váº«n thiáº¿u, popular khÃ´ng eco trong cluster, cÃ³ áº£nh
        if len(current_recs) < 12:
            needed = 12 - len(current_recs)
            already = {iid for iid, _ in current_recs}
            popular_non_eco = train_data[
                (train_data['season_cluster'] == fav_cluster) &
                (train_data['eco_label'] == False) &
                (train_data['article_id'].isin(existing_articles)) &
                (~train_data['article_id'].isin(already))
            ]['article_id'].value_counts().index[:needed]
            current_recs += [(iid, 0.0) for iid in popular_non_eco]
        
        # Náº¿u váº«n thiáº¿u, popular eco toÃ n bá»™, cÃ³ áº£nh
        if len(current_recs) < 12:
            needed = 12 - len(current_recs)
            already = {iid for iid, _ in current_recs}
            popular_eco_all = train_data[
                (train_data['eco_label'] == True) &
                (train_data['article_id'].isin(existing_articles)) &
                (~train_data['article_id'].isin(already))
            ]['article_id'].value_counts().index[:needed]
            current_recs += [(iid, 0.0) for iid in popular_eco_all]
        
        # Náº¿u váº«n thiáº¿u, popular toÃ n bá»™, cÃ³ áº£nh
        if len(current_recs) < 12:
            needed = 12 - len(current_recs)
            already = {iid for iid, _ in current_recs}
            popular_all = train_data[
                (train_data['article_id'].isin(existing_articles)) &
                (~train_data['article_id'].isin(already))
            ]['article_id'].value_counts().index[:needed]
            current_recs += [(iid, 0.0) for iid in popular_all]
    
    # LÆ°u recommendations vá»›i Ã­t nháº¥t 12 (hoáº·c táº¥t cáº£ náº¿u khÃ´ng Ä‘á»§)
    recommendations[user_id] = current_recs[:12]

# ... (rest of your code for evaluation, etc.)

# HÃ m láº¥y Ä‘Æ°á»�ng dáº«n hÃ¬nh áº£nh (unchanged)
def get_image_path(article_id):
    article_str = str(article_id).zfill(10)
    folder = article_str[0:3]
    image_path = Path(f'/kaggle/input/h-and-m-personalized-fashion-recommendations/images/{folder}/{article_str}.jpg')
    if image_path.exists():
        return str(image_path)
    else:
        return None

# Display code (bÃ¢y giá»� recommendations Ä‘Ã£ cÃ³ Ã­t nháº¥t 12, nhÆ°ng lá»�c láº¡i Ä‘á»ƒ cháº¯c cháº¯n)
random_user_id = random.choice(list(recommendations.keys()))
print(f"KhÃ¡ch hÃ ng ngáº«u nhiÃªn: {random_user_id}")

fav_cluster = user_fav_cluster.get(random_user_id, 0)
season_name = "Summer" if fav_cluster == 0 else f"Season {fav_cluster}"

recs = recommendations[random_user_id]

# Lá»�c láº¡i Ä‘á»ƒ cháº¯c cháº¯n táº¥t cáº£ sáº£n pháº©m cÃ³ áº£nh (dÃ¹ Ä‘Ã£ lá»�c trÆ°á»›c)
recs = [(iid, score) for iid, score in recs if get_image_path(iid)]

# Cáº¯t vá»� tá»‘i Ä‘a 12
recs = recs[:12]
n_items = len(recs)
nrows = (n_items + 2) // 3

fig, axs = plt.subplots(nrows, 3, figsize=(12, 4*nrows))
fig.suptitle(f"{season_name}?", fontsize=20)
for i, (article_id, score) in enumerate(recs):
    row, col = divmod(i, 3)
    ax = axs[row, col]
   
    image_path = get_image_path(article_id)
    if image_path:  # Now guaranteed by filter, but keep for safety
        img = mpimg.imread(image_path)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, f"No Image\n{article_id}", ha='center', va='center', fontsize=12)
        ax.set_facecolor('lightgray')
    
    ax.set_title(f"{article_id}\nScore: {score:.2f}", fontsize=10)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

# áº¨n cÃ¡c axes khÃ´ng dÃ¹ng tá»›i
for j in range(i+1, nrows*3):
    row, col = divmod(j, 3)
    axs[row, col].axis('off')

plt.tight_layout()
plt.show()




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# === 1. Táº¡o features Ä‘Æ¡n giáº£n Ä‘á»ƒ TF-IDF ===
train_data['features'] = (
    train_data['product_group_name'].fillna('') + ' ' +
    train_data['season'].fillna('') + ' ' +
    train_data['final_gender'].fillna('')
)

# === 2. TF-IDF ===
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
tfidf_matrix = vectorizer.fit_transform(train_data['features'])

# === 3. Mapping article_id -> index ===
article_id_to_idx = {aid: idx for idx, aid in enumerate(train_data['article_id'])}

# === 4. Cache toÃ n bá»™ gá»£i Ã½ (dÃ¹ng cosine similarity) ===
cbf_cache = {}
def get_cbf_recommendations_cached(article_id, top_n=20):
    if article_id in cbf_cache:
        return cbf_cache[article_id]
    
    if article_id not in article_id_to_idx:
        return []
    
    idx = article_id_to_idx[article_id]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = np.argsort(-cosine_sim)[1:top_n+1]
    recs = [(train_data.iloc[i]['article_id'], float(cosine_sim[i])) for i in similar_indices]
    cbf_cache[article_id] = recs
    return recs

# === 5. HÃ m Ä‘Ã¡nh giÃ¡ nhiá»�u sáº£n pháº©m input cá»§a user ===
def evaluate_precision_recall_multi(customer_id, k=5):
    true_items = set(test_data[test_data['customer_id'] == customer_id]['article_id'])
    if not true_items:
        return None, None
    
    # Láº¥y nhiá»�u sáº£n pháº©m user Ä‘Ã£ mua trong train
    start_items = train_data[train_data['customer_id'] == customer_id]['article_id'].tolist()
    if not start_items:
        return None, None
    
    # Gá»™p gá»£i Ã½ tá»« táº¥t cáº£ start_items
    rec_scores = {}
    for item in start_items:
        for aid, score in get_cbf_recommendations_cached(item, top_n=20):
            rec_scores[aid] = max(score, rec_scores.get(aid, 0))
    
    # Láº¥y top-k cuá»‘i cÃ¹ng
    top_recs = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    recommended_items = set(aid for aid, score in top_recs)
    
    precision = len(recommended_items & true_items) / k
    recall = len(recommended_items & true_items) / len(true_items)
    
    return precision, recall



precisions, recalls = [], []
for customer_id in test_data['customer_id'].unique()[:5000]:  # Giá»¯ sample 5000 user
    p, r = evaluate_precision_recall_multi(customer_id, k=5)
    if p is not None:
        precisions.append(p)
        recalls.append(r)

print("Average Precision@5:", np.mean(precisions))
print("Average Recall@5:", np.mean(recalls))


