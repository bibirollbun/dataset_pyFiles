# Libraries
import os
import gc
import wandb
import time
import random
import math
import glob
from scipy import spatial
from tqdm import tqdm
import warnings
import cv2
import pandas as pd
import numpy as np
from numpy import dot, sqrt
import seaborn as sns
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from IPython.display import display_html
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
plt.rcParams.update({'font.size': 16})

# Environment check
warnings.filterwarnings("ignore")
os.environ["WANDB_SILENT"] = "true"
CONFIG = {'competition': 'HandM', '_wandb_kernel': 'aot'}

# Custom colors
class clr:
    S = '\033[1m' + '\033[95m'
    E = '\033[0m'
    
my_colors = ["#AF0848", "#E90B60", "#CB2170", "#954E93", "#705D98", "#5573A8", "#398BBB", "#00BDE3"]


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


transactions = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')
# Let's convert back to parquet and load it in parquet format
transactions.to_parquet('transactions.parquet')
transactions_parquet = pd.read_parquet('./transactions.parquet')
del transactions # to save space

articles = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv')
customers = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv')


transactions_parquet['customer_id2'] =\
    transactions_parquet['customer_id'].apply(lambda x: int(x[-16:],16) ).astype('int64')

customers['customer_id'] = customers['customer_id'].apply(lambda x: int(x[-16:], 16) ).astype('int64')

print(transactions_parquet['customer_id2'].nunique())
print(transactions_parquet['customer_id'].nunique())

transactions_parquet.drop(['customer_id'], axis=1, inplace=True)
transactions_parquet.rename(columns={'customer_id2': 'customer_id'}, inplace=True)
transactions_parquet.head(3)


transactions_parquet['article_id2'] = transactions_parquet['article_id'].astype('int32')
# Quick check to ensure 1-1 mapping after converting to int32
print(transactions_parquet['article_id2'].nunique())
print(transactions_parquet['article_id'].nunique())

transactions_parquet.rename({'article_id2':'article_id'}, inplace=True)
articles['article_id'] = articles['article_id'].astype('int32')


print(clr.S+"ARTICLES:"+clr.E, articles.shape)
display_html(articles.head(3))
print("\n", clr.S+"CUSTOMERS:"+clr.E, customers.shape)
display_html(customers.head(3))
print("\n", clr.S+"TRANSACTIONS:"+clr.E, transactions_parquet.shape)
display_html(transactions_parquet.head(3))

print("\n", clr.S + "Number of unique customers =" + clr.E, transactions_parquet['customer_id'].nunique())
print("\n", clr.S + "Number of unique articles purchased = " + clr.E, transactions_parquet['article_id'].nunique())
print("\n", clr.S + "Number of transactions = " + clr.E, transactions_parquet.shape[0])


a = transactions_parquet['sales_channel_id'].value_counts()
print("\n", clr.S + "Pct of Channel 1 purchases = " + clr.E, round((a[1]*100/(a[1]+a[2])), 3), "%")
print("\n", clr.S + "Pct of Channel 2 purchases = " + clr.E, round((a[2]*100.0/(a[1]+a[2])), 3), "%")

print("\n", clr.S + "Start Date:" + clr.E, transactions_parquet['t_dat'].min())
print("\n", clr.S + "End Date:" + clr.E, transactions_parquet['t_dat'].max())


def adjust_id(x):
    '''Adjusts article ID code.'''
    x = str(x)
    if len(x) == 9:
        x = "0"+x
    
    return x

def insert_image(path, zoom, xybox, ax):
    '''Insert an image within matplotlib'''
    imagebox = OffsetImage(mpimg.imread(path), zoom=zoom)
    ab = AnnotationBbox(imagebox, xy=(0.5, 0.7), frameon=False, pad=1, xybox=xybox)
    ax.add_artist(ab)

def show_values_on_bars(axs, h_v="v", space=0.4):
    '''Plots the value at the end of the a seaborn barplot.
    axs: the ax of the plot
    h_v: whether or not the barplot is vertical/ horizontal'''
    
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, format(value, ','), ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, format(value, ','), ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

def convert_to_date(s):
    """
    Memoization technique - very fast conversion to pure python dates
    """
    dates = {date:datetime.datetime.strptime(date,'%Y-%m') for date in s.unique()}
    return s.map(dates)


def plot_values( df, col):

    print(clr.S+ f"Total Number of unique {col} values:"+clr.E, df[col].nunique())

    # Data
    val_cnt = df[col].value_counts().reset_index().head(15)
    clrs = ['#954E93' for x in val_cnt[col]]
    # clrs = ["#CB2170" if x==max(val_cnt[col]) else '#954E93' for x in val_cnt[col]]
    
    
    # Plot
    fig, ax = plt.subplots(figsize=(25, 13))
    plt.title(f'- Most Frequent {col} values -', size=22, weight="bold")
    
    sns.barplot(data=val_cnt, x="count", y=col, ax=ax,
                palette=clrs)

    show_values_on_bars(ax, h_v="h")
    
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    plt.show()


articles.isna().sum()


print(clr.S+"There are no missing values in any columns but 'Detail Description':"+clr.E,
      articles.isna().sum()[-1], "total missing values")
# Replace missing values
articles.fillna(value="No Description", inplace=True)


analyse_cols = ['prod_name', 'product_type_name', 'graphical_appearance_name', 'colour_group_name', 'perceived_colour_value_name',
                    'department_name', 'index_name', 'index_group_name', 'section_name', 'garment_group_name']

for col in analyse_cols:
    plot_values( articles, col)


text = ' '.join(articles['detail_desc'].tolist())

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)

# Display the word cloud
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


transactions_parquet.head()


tran_count = transactions_parquet.groupby('t_dat')['customer_id'].count().reset_index().rename(columns = {'customer_id' : 'count'})

fig, ax = plt.subplots(figsize=(25, 13))
plt.title('Number of Transactions by Date', size=22, weight="bold")

sns.lineplot(x='t_dat', y='count', data=tran_count)
plt.xticks(np.arange(0, len(tran_count), 30), rotation=90)
plt.show()


tran_count['t_dat'] = pd.to_datetime(tran_count['t_dat'])

tran_count['year_month'] = tran_count['t_dat'].dt.to_period('M')
monthly_vol = tran_count.groupby('year_month')['count'].sum().reset_index()
# monthly_vol['year_month'] = monthly_vol['year_month'].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(25, 13))
sns.barplot(x='year_month', y='count', data=monthly_vol, ax=ax)
plt.title('Monthly Transaction Volume', fontsize=18, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


purchase_counts = transactions_parquet['customer_id'].value_counts().reset_index()

plt.figure(figsize=(10,6))
sns.histplot(purchase_counts, bins=range(1, purchase_counts['count'].max()+2), kde=False)
plt.xlabel('Number of Purchases per Customer')
plt.ylabel('Number of Customers')
plt.title('Distribution of Purchase Frequency per Customer')
plt.show()


tmp = purchase_counts[purchase_counts['count']<=20].groupby('count')['customer_id'].count().reset_index().rename(columns={'count':'num_purchases', 'customer_id':'num_customers'})
plt.figure(figsize=(10,6))
sns.barplot(tmp, x='num_purchases', y='num_customers')
plt.xlabel('Number of < 20 Purchases per Customer')
plt.ylabel('Number of Customers')
plt.title('Distribution of Purchase Frequency per Customer')
plt.show()


plt.figure(figsize=(10,6))
ax = sns.countplot(x = 'sales_channel_id', data=transactions_parquet)
plt.xlabel('Sales Channel')
plt.ylabel('Number of Customers')
plt.title('Transactions by Sales Channel')

for container in ax.containers:
    ax.bar_label(container, labels=[f'{int(v):,}' for v in container.datavalues])
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(data=transactions_parquet, x='price', bins=50, kde=True)
plt.title('Distribution of Transaction Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')

plt.show()





customers.head()


customers.isna().sum()


print(customers['FN'].value_counts(dropna=False))
print(customers['Active'].value_counts(dropna=False))
print(customers['club_member_status'].value_counts(dropna=False))
print(customers['fashion_news_frequency'].value_counts(dropna=False))


# Fill FN and Active - the only available value is "1"
customers["FN"].fillna(0, inplace=True)
customers["Active"].fillna(0, inplace=True)

# Set unknown the club member status & news frequency
customers["club_member_status"].fillna("UNKNOWN", inplace=True)

customers["fashion_news_frequency"] = customers["fashion_news_frequency"].replace({"None":"NONE"})
customers["fashion_news_frequency"].fillna("UNKNOWN", inplace=True)

# Set missing values in age with the median
customers["age"].fillna(customers["age"].median(), inplace=True)

print(customers.isna().sum())


def plot_values( df, col):

    print(clr.S+ f"Total Number of unique {col} values:"+clr.E, df[col].nunique())

    # Data
    val_cnt = df[col].value_counts().reset_index().head(15)
    clrs = ['#954E93' for x in val_cnt[col]]
    # clrs = ["#CB2170" if x==max(val_cnt[col]) else '#954E93' for x in val_cnt[col]]
    
    
    # Plot
    fig, ax = plt.subplots(figsize=(25, 13))
    plt.title(f'- Most Frequent {col} values -', size=22, weight="bold")
    
    sns.barplot(data=val_cnt, x="count", y=col, ax=ax,
                palette=clrs)

    show_values_on_bars(ax, h_v="h")
    
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    plt.show()


customers.dtypes


cust2 = customers.drop(['customer_id', 'postal_code', 'age'], axis=1)
cust2.value_counts()


articles['article_id'] = articles['article_id'].astype('int64')


combined_df1 = transactions_parquet.merge(articles, on='article_id', how='left')
display_html(combined_df1.head(3))
del articles
del transactions_parquet


combined_df1 = combined_df1[['t_dat', 'customer_id', 'article_id', 'price', 'sales_channel_id',
       'product_code', 'prod_name', 'product_type_name',
       'product_group_name',
       'graphical_appearance_name', 'colour_group_name',
       'perceived_colour_value_name',
       'perceived_colour_master_name',
       'department_name', 'index_name',
       'index_group_name', 'section_name',
       'garment_group_name', 'detail_desc']]


combined_df = combined_df1.merge(customers, on='customer_id', how='left')
display_html(combined_df.head(3))
del combined_df1


print(clr.S+"Total Number of Transacting Customers"+clr.E, \
          f"{combined_df['customer_id'].nunique():,}")
print(clr.S+"Total Number of Unique Articles purchased"+clr.E, \
          f"{combined_df['article_id'].nunique():,}")
print(clr.S+"Min Price of items purchased"+clr.E, \
          combined_df['price'].min())
print(clr.S+"Max Price of items purchased"+clr.E, \
          combined_df['price'].max())
print(clr.S+"Total Number of unique Addresses:"+clr.E, \
          f"{combined_df['postal_code'].nunique():,}")


print(clr.S+"Missing values within customers dataset:"+clr.E)
print(combined_df.isna().sum())


print(combined_df['FN'].value_counts(dropna=False))
print(combined_df['Active'].value_counts(dropna=False))
print(combined_df['club_member_status'].value_counts(dropna=False))
print(combined_df['fashion_news_frequency'].value_counts(dropna=False))


# Fill FN and Active - the only available value is "1"
combined_df["FN"].fillna(0, inplace=True)
combined_df["Active"].fillna(0, inplace=True)

# Set unknown the club member status & news frequency
combined_df["club_member_status"].fillna("UNKNOWN", inplace=True)

combined_df["fashion_news_frequency"] = combined_df["fashion_news_frequency"].replace({"None":"NONE"})
combined_df["fashion_news_frequency"].fillna("UNKNOWN", inplace=True)

# Set missing values in age with the median
combined_df["age"].fillna(customers["age"].median(), inplace=True)

combined_df['detail_desc'].fillna(" ", inplace=True)

print(combined_df.isna().sum())


combined_df['Month-Year'] = combined_df['t_dat'].str.slice(0, 7)

prod_cnt_df = combined_df.groupby(['Month-Year', 'product_type_name'])\
                ['product_type_name'].count().rename('Count').reset_index()
most_purchased = prod_cnt_df.groupby('Month-Year').apply(lambda x: x.nlargest(10, 'Count'))


sweater_cnt_df = prod_cnt_df[(prod_cnt_df['product_type_name'] == 'Sweater') | (prod_cnt_df['product_type_name'] == 'Bikini top')]

# Plot
fig, ax = plt.subplots(figsize=(20, 10))
plt.title('- No. of Sweaters / Bikinis Sold -', size=22, weight="bold")


sns.lineplot(x='Month-Year', y='Count', data=sweater_cnt_df, ax=ax, hue = 'product_type_name', markers='*')
x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()

ticks = range(0, len(sweater_cnt_df), 6)
plt.xticks(ticks)
plt.show()


combined_df['age'].describe()


def create_age_interval(x):
    if x <= 25:
        return [16, 25]
    elif x <= 35:
        return [26, 35]
    elif x <= 45:
        return [36, 45]
    elif x <= 55:
        return [46, 55]
    elif x <= 65:
        return [56, 65]
    else:
        return [66, 99]

combined_df["age_interval"] = combined_df["age"].apply(lambda x: create_age_interval(x))


plt.figure(figsize=(24, 10))
plt.suptitle('- Customer Profile -', size=22, weight="bold")

ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,1,2)

sns.countplot(data=customers, x="club_member_status", ax=ax1,
              order=customers['club_member_status'].value_counts().index,
              palette=my_colors[2:])
show_values_on_bars(axs=ax1, h_v="v", space=0.4)
ax1.set_title("Club Member Status", size=18, weight="bold")
ax1.set_yticks([])
ax1.set_xlabel("")
ax1.set_ylabel("")

sns.countplot(data=customers, x="fashion_news_frequency", ax=ax2,
              order=customers['fashion_news_frequency'].value_counts().index,
              palette=my_colors[2:])
show_values_on_bars(axs=ax2, h_v="v", space=0.4)
ax2.set_title("Fashion News frequency", size=18, weight="bold")
ax2.set_yticks([])
ax2.set_xlabel("")
ax2.set_ylabel("")

sns.distplot(customers["age"], color=my_colors[-3], ax=ax3,
             hist_kws=dict(edgecolor=my_colors[-3]))
ax3.set_title("Age Distribution", size=18, weight="bold")
ax3.set_ylabel("")

for ax in [ax1, ax2]:
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    # ax.imshow(bk_image, zorder=0, extent=[x0, x1, y0, y1], alpha=0.35, aspect='auto')
    
# insert_image(path='../input/hm-fashion-recommender-dataset/pics/vans.jpg', zoom=0.5, xybox=(60, 0.00), ax=ax3)

sns.despine(left=True, bottom=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.99);


def create_age_interval2(x):
    if x <= 25:
        return '16-25'
    elif x <= 35:
        return '26-35'
    elif x <= 45:
        return '36-45'
    elif x <= 55:
        return '46-55'
    elif x <= 65:
        return '56-65'
    else:
        return '66-99'

combined_df["age_interval"] = combined_df["age"].apply(lambda x: create_age_interval2(x))


age_prod_df = combined_df.groupby(['age_interval', 'sales_channel_id'])['sales_channel_id'].count().rename('Count').reset_index()
# most_purchased = age_prod_df.groupby('age_interval').apply(lambda x: x.nlargest(6, 'Count'))
age_prod_df['channel-wise %'] = age_prod_df['Count'] / age_prod_df.groupby('age_interval')['Count'].transform('sum') * 100

age_prod_df.head(1000)


age_prod_df = combined_df.groupby(['age_interval', 'prod_name'])['perceived_colour_value_name'].count().rename('Count').reset_index()
most_purchased = age_prod_df.groupby('age_interval').apply(lambda x: x.nlargest(10, 'Count'))

most_purchased.head(1000)




