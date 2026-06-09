# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Read One parquet file. Obviously, it's big.

train = pd.read_parquet("../input/aeroclub-recsys-2025/train.parquet")
train.tail()


!pip install duckdb


#By Andrew D. Blevins https://www.kaggle.com/code/andrewdblevins/leash-tutorial-ecfps-and-random-forest

import duckdb
import pandas as pd

train_path = '/kaggle/input/aeroclub-recsys-2025/train.parquet'
test_path = '/kaggle/input/aeroclub-recsys-2025/test.parquet'

con = duckdb.connect()

df = con.query(f"""(SELECT *
                        FROM parquet_scan('{train_path}')
                        WHERE selected = 0
                        ORDER BY random()
                        LIMIT 30000)
                        UNION ALL
                        (SELECT *
                        FROM parquet_scan('{train_path}')
                        WHERE selected = 1
                        ORDER BY random()
                        LIMIT 30000)""").df()

con.close()


df.tail()


#By Iqbal Syah Akbar https://www.kaggle.com/code/iqbalsyahakbar/ps3e22-multi-class-classification-for-beginners

desc = pd.DataFrame(index = list(df))
desc['count'] = df.count()
desc['nunique'] = df.nunique()
desc['%unique'] = desc['nunique'] / len(df) * 100
desc['null'] = df.isnull().sum()
desc['type'] = df.dtypes
desc = pd.concat([desc, df.describe().T], axis = 1)
desc


df.info()


# Check if there are any missing values left
train_na = (df.isnull().sum() / len(df)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :train_na})
missing_data.head()


#https://stackoverflow.com/questions/43816122/how-to-represent-boolean-data-in-graph

# Set up a grid of plots
fig = plt.figure(figsize=(10,10)) 
fig_dims = (3, 2)


# Plot accidents depending on type
plt.subplot2grid(fig_dims, (0, 0))
df['sex'].value_counts().plot(kind='bar', 
                                     title='Gender')
plt.subplot2grid(fig_dims, (0, 1))
df['bySelf'].value_counts().plot(kind='bar', 
                                     title='Travel by Self')
plt.subplot2grid(fig_dims, (1, 0))
df['isVip'].value_counts().plot(kind='bar', 
                                     title='Vip')
plt.subplot2grid(fig_dims, (1, 1))
df['selected'].value_counts().plot(kind='bar', 
                                     title='selected');


#https://stackoverflow.com/questions/64791405/log-scale-for-multiple-subplot-histograms-in-pandas

# no need to initiate `fig,ax` to avoid the warning
axes = df.hist(bins=25, figsize=(20,30), layout=(-1, 4), edgecolor="black")
plt.tight_layout()
#plt.yscale('log') or xscale('log') Both didn't help

# set log scale
#for a in axes.ravel(): a.set_yscale('log')#xscale didn't change the size of the bars


plt.scatter(df["totalPrice"], df["selected"], color="yellow", edgecolor="black")
plt.ylabel("Selected")
plt.xlabel("Total Price")
plt.title("Relationship between Total Price vs Selected flight", fontsize=14)
#plt.xscale('log')
#plt.yscale('log')
plt.show()


#By Karnika Kapoor https://www.kaggle.com/code/karnikakapoor/diamond-price-prediction

ax = sns.regplot(x="totalPrice", y="selected", data=df, fit_reg=True, scatter_kws={"color": "#006400"}, line_kws={"color": "#FF1493"})
ax.set_title("Regression Line on Total Price vs Selected flight", color="#4e4c39");


from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import pearsonr

df.plot("selected","totalPrice",style='o') 
print("Pearson correlation:",df["selected"].corr(df["totalPrice"]))
print("T Test and P value:",stats.ttest_ind(df["selected"],df["totalPrice"]))


plt.figure(figsize=(7,2))
df['isVip'].value_counts().plot(kind='barh', color='black')
plt.title('Is a Vip Passenger?');

