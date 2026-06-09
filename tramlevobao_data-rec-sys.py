# import packages
import numpy as np
import pandas as pd

import matplotlib 
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.lines as lines

import seaborn as sns

from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_pacf
import random

from pathlib import Path

import time
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_colwidth', False)


# runtime configuration of matplotlib
plt.style.use("Solarize_Light2")
plt.rc("figure", 
    autolayout=True, 
    figsize=(20, 10)
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=20,
    titlepad=10,
)

# periodogram
def plot_periodogram(ts, detrend='linear', ax=None, title="Periodogram"):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title(title)
    return ax


data_path = Path('/kaggle/input/h-and-m-personalized-fashion-recommendations/')

# This you have to do just once.
start = time.time()

transactions = pd.read_csv(data_path/'transactions_train.csv', 
                           dtype={'article_id': str} ,low_memory=True)

articles = pd.read_csv(data_path / 'articles.csv', dtype={'article_id': str},low_memory=True)

end = time.time()

print(f"Loading time for csv format: {np.round(end - start)}")


# Save the csv file to parquet. This you have to do just once.
transactions.to_parquet('transactions_parquet.parquet')
articles.to_parquet('articles_parquet.parquet')



customer = pd.read_csv(data_path / 'customers.csv')


sample = pd.read_csv(data_path / 'sample_submission.csv')


# Read the parquet data.
start = time.time()

transactions_parquet = pd.read_parquet('transactions_parquet.parquet')
articles_parquet = pd.read_parquet('articles_parquet.parquet')


end = time.time()

print(f"Loading time for parquet format: {np.round((end - start),2)}")


transactions = transactions_parquet.copy()


transactions


articles = articles_parquet.copy()


articles





# Convert date to date object
transactions = transactions_parquet.copy()
transactions["date"] = pd.to_datetime(transactions["t_dat"]).dt.date
transactions.drop(columns=["t_dat"], inplace=True)

# Extract product name
df_plot = transactions.merge(articles[["article_id", "prod_name"]],\
     how='left', on=None, left_on='article_id', right_on='article_id', suffixes=('_x', '_y'))

# Calculate the order of the transcation per customer
order_number = df_plot[["date", "customer_id"]].groupby(["date", "customer_id"]).count()
order_number.reset_index(["date", "customer_id"], inplace=True)

order_number['nth_order'] = order_number.sort_values(["customer_id",'date'], ascending=True)\
             .groupby(['customer_id'])\
             .cumcount() + 1
order_number.loc[order_number["customer_id"]=="000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318",:]

# Join transaction with order of transaction from previous step
df_plot = df_plot.merge(order_number[["date", "customer_id", "nth_order"]],\
     how='left', on=None, left_on=["date", "customer_id"], right_on=["date", "customer_id"], suffixes=('_x', '_y'))

# prepare data for eda
df_plot["flag"] = 1

y_channel= df_plot[["date","sales_channel_id", "customer_id","flag"]]\
    .groupby(["date","sales_channel_id", "customer_id"]).max("flag")
y_channel.reset_index(level="sales_channel_id", inplace=True)

y_channel_2= df_plot.groupby(["date","sales_channel_id"]).agg({"customer_id": lambda num: num.nunique()}) #total price per customer
y_channel_2.columns = ['nb_visitors']
y_channel_2.reset_index(level=["date","sales_channel_id"], inplace=True)


# placeholders for min and max of the axis
xmin = y_channel_2["date"].min()
xmax = y_channel_2["date"].max()

ymin = y_channel_2["nb_visitors"].min() - 1000
ymax = y_channel_2["nb_visitors"].max() + 1000


top12= pd.value_counts(df_plot["prod_name"]).iloc[:12]

# plot
fig = plt.figure(constrained_layout=False)
spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)

ax1 = fig.add_subplot(spec[:2, 0])
sns.countplot(y="prod_name", ax=ax1, data=df_plot, order=top12.index);
ax1.set(xlabel="Quantity Sold", ylabel = "")
plt.setp(ax1.get_yticklabels(), rotation=45);
ax1.set_title('Top 12 popular articles')

ax3 = fig.add_subplot(spec[:2, 1]);
sns.countplot(x="sales_channel_id", ax=ax3, data=y_channel, palette=["orange", "green"]);
ax3.set(xlabel="Channel", ylabel = "Visitors");
ax3.ticklabel_format(style='plain', useOffset=False, axis='y');
ax3.set_xticklabels(["offline", "online"])
ax3.set_title('Visitors per channel');

ax5 = fig.add_subplot(spec[2, :]);
ax5.set(xlabel="Date", ylabel = "Visitors");
sns.scatterplot(data=y_channel_2.loc[y_channel_2["sales_channel_id"]==1,:],  x="date", y="nb_visitors", color=['orange'], label="offline", ax=ax5)
sns.scatterplot(data=y_channel_2.loc[y_channel_2["sales_channel_id"]==2,:],  x="date", y="nb_visitors", color=['green'], label="online", ax=ax5)
ax5.ticklabel_format(style='plain', useOffset=False, axis='y');
ax5.legend(title="Channel")
ax5.set_title('Daily Visitors');
ax5.set_ylim(ymin, ymax)
ax5.set_xlim(xmin, xmax)

ax5.fill_betweenx([ymin,ymax],18343, 18384, color="gray", alpha=0.3)

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax5.annotate("Missing offline \ntransaction period", (18342, 15000), (18270, 27000), \
    arrowprops={"arrowstyle": "->", "color":"C1"},
    bbox=props,
    fontproperties='italic'
    );


# 
palette_dict = ["#003f5c", "#2f4b7c", "#665191", "#a05195", "#d45087", "#f95d6a", "#ff7c43", "#ffa600"]

online_transactions = df_plot.loc[df_plot["sales_channel_id"]==2,:]

first_order = pd.value_counts(online_transactions.loc[online_transactions["nth_order"]==1,"prod_name"]).iloc[:12].index

palette_1 = [
'lightblue'for d in first_order if d in ('Luna skinny RW', 'Jade HW Skinny Denim TRS', 'Timeless Midrise Brief','Tilly (1)') 
] + random.sample(palette_dict, 8)

not_first_order = pd.value_counts(df_plot.loc[df_plot["nth_order"]!=1,"prod_name"]).iloc[:12].index

palette_2 = [
'lightblue'for d in not_first_order if d in ('Luna skinny RW', 'Jade HW Skinny Denim TRS', 'Timeless Midrise Brief','Tilly (1)') 
] + random.sample(palette_dict, 8)


fig = plt.figure(constrained_layout=False)
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

ax1 = fig.add_subplot(spec[0])
sns.countplot(y="prod_name", ax=ax1, \
     data=online_transactions.loc[online_transactions["nth_order"]==1,:],\
     order=first_order,
     palette=palette_1
     )
ax1.set(xlabel="Quantity Sold", ylabel = "")
plt.setp(ax1.get_xticklabels(), rotation=90)
ax1.set_title('Popular Products - New visitors Online')

ax2 = fig.add_subplot(spec[1])
sns.countplot(y="prod_name", ax=ax2, \
     data=df_plot.loc[df_plot["nth_order"]!=1,:],\
     order=not_first_order,
     palette=palette_2)
ax2.set(xlabel="Quantity Sold", ylabel = "")
plt.setp(ax2.get_xticklabels(), rotation=90)
ax2.set_title('Popular Products - Returning visitors')

fig.add_artist(lines.Line2D([0, 1], [0.667, 0.667], color="black", linestyle="-."));


new_customers_online = df_plot.loc[(df_plot["nth_order"]==1)&(df_plot["sales_channel_id"]==2),:]
new_y_online= new_customers_online.groupby("date").agg({"article_id": lambda num: num.nunique()})
new_y_online.columns = ['products_sold']
new_y_online.index = pd.to_datetime(new_y_online.index)


fig = plt.figure(constrained_layout=False, tight_layout=True)
spec = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

ax1 = fig.add_subplot(spec[0,:])
sns.scatterplot(data=new_y_online,  x="date", y="products_sold", color=['green'], ax=ax1);
ax1.set_title('Daily Product Sales');
ax1.set(xlabel="Date", ylabel = "Number Sold");

ax5 = fig.add_axes([0.77, 0.75, 0.2, 0.2]);
ax5.hist(new_y_online["products_sold"]);

# place a text box in upper left in axes coords
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax5.text(6000, 250, "Histogram daily product sales",fontsize=8, verticalalignment='top', bbox=props)

ax2 = fig.add_subplot(spec[1,0])
plot_periodogram(new_y_online["products_sold"], 'linear', ax2, title="Periodogram");

ax3 = fig.add_subplot(spec[1,1])
plot_pacf(new_y_online["products_sold"], ax3, lags=10, title="PACF");
ax3.set_xlabel("Lags");

