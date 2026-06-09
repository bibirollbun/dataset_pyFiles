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


import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense, GRU, Input
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
df_train.info()


missing_values_count = df_train.isnull().sum()
missing_values_count


df_train.head()


def plot_data(name, data):
    plt.figure(figsize=(5, 5))
    palette_color = sns.color_palette('pastel')
    explode = [0.1 for _ in range(data.nunique())]

    target_counts = df_train.groupby(name)[name].count()

    target_counts.plot.pie(
    colors=palette_color,
    explode=explode,
    autopct="%1.1f%%",
    shadow=True,
    startangle=140,
    textprops={'fontsize': 14},
    wedgeprops={'edgecolor': 'black', 'linewidth': 1.5} 
    )

    plt.title(name, fontsize=18, weight='bold')
    plt.axis('equal')
    plt.show()


print('country :',df_train['country'].unique())
print('store :',df_train['store'].unique())
print('product :',df_train['product'].unique())


plot_data('country', df_train['country'])


plot_data('store', df_train['store'])


plot_data('product', df_train['product'])


fig, axes = plt.subplots(1, 3, sharex=True, figsize=(15,5))

plt.suptitle("Sticker Sales")

sns.histplot(data=df_train, x='num_sold', hue='country', ax=axes[0]);

sns.histplot(data=df_train, x='num_sold', hue='store', ax=axes[1]);

sns.histplot(data=df_train, x='num_sold', hue='product', ax=axes[2]);


print(len(df_train.loc[(df_train['country']=='Canada')&(df_train['num_sold'].isnull())]))
print(len(df_train.loc[(df_train['country']=='Canada')&(df_train['product']=='Holographic Goose')&(df_train['num_sold'].isnull())]))
print(len(df_train.loc[(df_train['country']=='Canada')&(df_train['product']=='Kaggle')&(df_train['num_sold'].isnull())]))
print(len(df_train.loc[(df_train['country']=='Canada')&(df_train['product']=='Kaggle Tiers')&(df_train['num_sold'].isnull())]))
print(len(df_train.loc[(df_train['country']=='Canada')&(df_train['product']=='Kerneler')&(df_train['num_sold'].isnull())]))
print(len(df_train.loc[(df_train['country']=='Canada')&(df_train['product']=='Kerneler Dark Mode')&(df_train['num_sold'].isnull())]))


#Example
print('store: Discount Stickers, product: Kaggle ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kaggle')]['num_sold'])[0])
print('store: Stickers for Less, product: Kaggle ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Kaggle')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Kaggle ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Kaggle')]['num_sold'])[0])


#Example
print('store: Discount Stickers, product: Kaggle ==>',list(df_train.loc[(df_train['date']=='2011-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kaggle')]['num_sold'])[0])
print('store: Stickers for Less, product: Kaggle ==>',list(df_train.loc[(df_train['date']=='2011-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Kaggle')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Kaggle ==>',list(df_train.loc[(df_train['date']=='2011-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Kaggle')]['num_sold'])[0])


#Example
print('store: Discount Stickers, product: Kaggle ==>',list(df_train.loc[(df_train['date']=='2012-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kaggle')]['num_sold'])[0])
print('store: Stickers for Less, product: Kaggle ==>',list(df_train.loc[(df_train['date']=='2012-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Kaggle')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Kaggle ==>',list(df_train.loc[(df_train['date']=='2012-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Kaggle')]['num_sold'])[0])


#Example
print('store: Discount Stickers, product: Kaggle ==>',list(df_train.loc[(df_train['date']=='2013-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kaggle')]['num_sold'])[0])
print('store: Stickers for Less, product: Kaggle ==>',list(df_train.loc[(df_train['date']=='2013-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Kaggle')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Kaggle ==>',list(df_train.loc[(df_train['date']=='2013-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Kaggle')]['num_sold'])[0])


#Example
print('store: Discount Stickers, product: Kaggle ==>',list(df_train.loc[(df_train['date']=='2014-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kaggle')]['num_sold'])[0])
print('store: Stickers for Less, product: Kaggle ==>',list(df_train.loc[(df_train['date']=='2014-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Kaggle')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Kaggle ==>',list(df_train.loc[(df_train['date']=='2014-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Kaggle')]['num_sold'])[0])


#Example
print('store: Discount Stickers, product: Kaggle ==>',list(df_train.loc[(df_train['date']=='2015-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kaggle')]['num_sold'])[0])
print('store: Stickers for Less, product: Kaggle ==>',list(df_train.loc[(df_train['date']=='2015-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Kaggle')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Kaggle ==>',list(df_train.loc[(df_train['date']=='2015-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Kaggle')]['num_sold'])[0])


#Example
print('store: Discount Stickers, product: Kaggle Tiers ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kaggle Tiers')]['num_sold'])[0])
print('store: Stickers for Less, product: Kaggle Tiers ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Kaggle Tiers')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Kaggle Tiers ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Kaggle Tiers')]['num_sold'])[0])


#Example
print('store: Discount Stickers, product: Kaggle Tiers ==>',list(df_train.loc[(df_train['date']=='2012-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kaggle Tiers')]['num_sold'])[0])
print('store: Stickers for Less, product: Kaggle Tiers ==>',list(df_train.loc[(df_train['date']=='2012-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Kaggle Tiers')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Kaggle Tiers ==>',list(df_train.loc[(df_train['date']=='2012-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Kaggle Tiers')]['num_sold'])[0])


#Example
print('store: Discount Stickers, product: Kaggle Tiers ==>',list(df_train.loc[(df_train['date']=='2015-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kaggle Tiers')]['num_sold'])[0])
print('store: Stickers for Less, product: Kaggle Tiers ==>',list(df_train.loc[(df_train['date']=='2015-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Kaggle Tiers')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Kaggle Tiers ==>',list(df_train.loc[(df_train['date']=='2015-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Kaggle Tiers')]['num_sold'])[0])


#Example
print('store: Discount Stickers, product: Kerneler ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kerneler')]['num_sold'])[0])
print('store: Stickers for Less, product: Kerneler ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Kerneler')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Kerneler ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Kerneler')]['num_sold'])[0])


#Example
print('store: Discount Stickers, product: Kerneler ==>',list(df_train.loc[(df_train['date']=='2012-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kerneler')]['num_sold'])[0])
print('store: Stickers for Less, product: Kerneler ==>',list(df_train.loc[(df_train['date']=='2012-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Kerneler')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Kerneler ==>',list(df_train.loc[(df_train['date']=='2012-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Kerneler')]['num_sold'])[0])


#Example
print('store: Discount Stickers, product: Kerneler ==>',list(df_train.loc[(df_train['date']=='2015-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kerneler')]['num_sold'])[0])
print('store: Stickers for Less, product: Kerneler ==>',list(df_train.loc[(df_train['date']=='2015-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Kerneler')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Kerneler ==>',list(df_train.loc[(df_train['date']=='2015-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Kerneler')]['num_sold'])[0])


#Example
print('store: Discount Stickers, product: Kerneler Dark Mode ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kerneler Dark Mode')]['num_sold'])[0])
print('store: Stickers for Less, product: Kerneler Dark Mode ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Kerneler Dark Mode')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Kerneler Dark Mode ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Kerneler Dark Mode')]['num_sold'])[0])


#Example
print('store: Discount Stickers, product: Kerneler Dark Mode ==>',list(df_train.loc[(df_train['date']=='2012-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kerneler Dark Mode')]['num_sold'])[0])
print('store: Stickers for Less, product: Kerneler Dark Mode ==>',list(df_train.loc[(df_train['date']=='2012-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Kerneler Dark Mode')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Kerneler Dark Mode ==>',list(df_train.loc[(df_train['date']=='2012-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Kerneler Dark Mode')]['num_sold'])[0])


#Example
print('store: Discount Stickers, product: Kerneler Dark Mode ==>',list(df_train.loc[(df_train['date']=='2015-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kerneler Dark Mode')]['num_sold'])[0])
print('store: Stickers for Less, product: Kerneler Dark Mode ==>',list(df_train.loc[(df_train['date']=='2015-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Kerneler Dark Mode')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Kerneler Dark Mode ==>',list(df_train.loc[(df_train['date']=='2015-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Kerneler Dark Mode')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2010-02-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2010-02-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2010-02-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2011-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2011-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2011-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2011-02-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2011-02-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2011-02-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2012-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2012-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2012-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2012-02-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2012-02-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2012-02-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2013-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2013-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2013-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2013-02-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2013-02-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2013-02-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2014-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2014-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2014-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2014-02-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2014-02-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2014-02-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2015-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2015-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2015-10-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2015-02-01')&(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2015-02-01')&(df_train['country']=='Canada')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2015-02-01')&(df_train['country']=='Canada')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])


x = list(df_train.loc[(df_train['country']=='Canada')&(df_train['product']=='Holographic Goose')&(df_train['num_sold'].notnull())]['num_sold'].mode())[0]
x


x = list(df_train.loc[(df_train['country']=='Canada')&(df_train['product']=='Holographic Goose')&(df_train['num_sold'].notnull())]['num_sold'].mode())[0]
ind = df_train.loc[(df_train['country']=='Canada')&(df_train['product']=='Holographic Goose')&(df_train['num_sold'].isnull())].index

for i in ind:
    df_train.loc[i, 'num_sold'] = x


print(len(df_train.loc[(df_train['country']=='Canada')&(df_train['num_sold'].isnull())]))
print(len(df_train.loc[(df_train['country']=='Canada')&(df_train['product']=='Holographic Goose')&(df_train['num_sold'].isnull())]))
print(len(df_train.loc[(df_train['country']=='Canada')&(df_train['product']=='Kaggle')&(df_train['num_sold'].isnull())]))
print(len(df_train.loc[(df_train['country']=='Canada')&(df_train['product']=='Kaggle Tiers')&(df_train['num_sold'].isnull())]))
print(len(df_train.loc[(df_train['country']=='Canada')&(df_train['product']=='Kerneler')&(df_train['num_sold'].isnull())]))
print(len(df_train.loc[(df_train['country']=='Canada')&(df_train['product']=='Kerneler Dark Mode')&(df_train['num_sold'].isnull())]))


fig, axes = plt.subplots(1, 3, sharex=True, figsize=(15,5))

plt.suptitle("Sticker Sales")

sns.histplot(data=df_train, x='num_sold', hue='country', ax=axes[0]);

sns.histplot(data=df_train, x='num_sold', hue='store', ax=axes[1]);

sns.histplot(data=df_train, x='num_sold', hue='product', ax=axes[2]);


missing_values_count = df_train.isnull().sum()
missing_values_count


df_train.loc[df_train['num_sold'].isnull()]


print(len(df_train.loc[(df_train['country']=='Kenya')&(df_train['num_sold'].isnull())]))
print(len(df_train.loc[(df_train['country']=='Kenya')&(df_train['product']=='Holographic Goose')&(df_train['num_sold'].isnull())]))
print(len(df_train.loc[(df_train['country']=='Kenya')&(df_train['product']=='Kaggle')&(df_train['num_sold'].isnull())]))
print(len(df_train.loc[(df_train['country']=='Kenya')&(df_train['product']=='Kaggle Tiers')&(df_train['num_sold'].isnull())]))
print(len(df_train.loc[(df_train['country']=='Kenya')&(df_train['product']=='Kerneler')&(df_train['num_sold'].isnull())]))
print(len(df_train.loc[(df_train['country']=='Kenya')&(df_train['product']=='Kerneler Dark Mode')&(df_train['num_sold'].isnull())]))


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2011-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2011-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2011-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2011-12-01')&(df_train['country']=='Kenya')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2011-12-01')&(df_train['country']=='Kenya')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2011-12-01')&(df_train['country']=='Kenya')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2013-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2013-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2013-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2014-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2014-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2014-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2015-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2015-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2015-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Holographic Goose')]['num_sold'])[0])


x = list(df_train.loc[(df_train['country']=='Kenya')&(df_train['product']=='Holographic Goose')&(df_train['num_sold'].notnull())]['num_sold'].mode())[0]
print(x)
ind = df_train.loc[(df_train['country']=='Kenya')&(df_train['product']=='Holographic Goose')&(df_train['num_sold'].isnull())].index

for i in ind:
    df_train.loc[i, 'num_sold'] = x


fig, axes = plt.subplots(1, 3, sharex=True, figsize=(15,5))

plt.suptitle("Sticker Sales")

sns.histplot(data=df_train, x='num_sold', hue='country', ax=axes[0]);

sns.histplot(data=df_train, x='num_sold', hue='store', ax=axes[1]);

sns.histplot(data=df_train, x='num_sold', hue='product', ax=axes[2]);


missing_values_count = df_train.isnull().sum()
missing_values_count


df_train.loc[df_train['num_sold'].isnull()]


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kerneler')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Kerneler')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2010-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Kerneler')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2011-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kerneler')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2011-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Kerneler')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2011-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Kerneler')]['num_sold'])[0])


print('store: Discount Stickers, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2012-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kerneler')]['num_sold'])[0])
print('store: Stickers for Less, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2012-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Stickers for Less')&(df_train['product']=='Kerneler')]['num_sold'])[0])
print('store: Premium Sticker Mart, product: Holographic Goose ==>',list(df_train.loc[(df_train['date']=='2012-10-01')&(df_train['country']=='Kenya')&(df_train['store']=='Premium Sticker Mart')&(df_train['product']=='Kerneler')]['num_sold'])[0])


df_train.loc[(df_train['country']=='Kenya')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kerneler')]['num_sold']


len(df_train.loc[(df_train['country']=='Kenya')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kerneler')&(df_train['num_sold'].isnull())])


x = list(df_train.loc[(df_train['country']=='Kenya')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kerneler')&(df_train['num_sold'].notnull())]['num_sold'].mode())[0]
print(x)
ind = df_train.loc[(df_train['country']=='Kenya')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kerneler')&(df_train['num_sold'].isnull())].index

for i in ind:
    df_train.loc[i, 'num_sold'] = x


fig, axes = plt.subplots(1, 3, sharex=True, figsize=(15,5))

plt.suptitle("Sticker Sales")

sns.histplot(data=df_train, x='num_sold', hue='country', ax=axes[0]);

sns.histplot(data=df_train, x='num_sold', hue='store', ax=axes[1]);

sns.histplot(data=df_train, x='num_sold', hue='product', ax=axes[2]);


missing_values_count = df_train.isnull().sum()
missing_values_count


df_train.loc[df_train['num_sold'].isnull()]


x = list(df_train.loc[(df_train['country']=='Kenya')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kerneler Dark Mode')&(df_train['num_sold'].notnull())]['num_sold'].mode())[0]
print(x)
ind = df_train.loc[(df_train['country']=='Kenya')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kerneler Dark Mode')&(df_train['num_sold'].isnull())].index
df_train.loc[ind, 'num_sold'] = x


x = list(df_train.loc[(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kerneler')&(df_train['num_sold'].notnull())]['num_sold'].mode())[0]
print(x)
ind = df_train.loc[(df_train['country']=='Canada')&(df_train['store']=='Discount Stickers')&(df_train['product']=='Kerneler')&(df_train['num_sold'].isnull())].index
df_train.loc[ind, 'num_sold'] = x


fig, axes = plt.subplots(1, 3, sharex=True, figsize=(15,5))

plt.suptitle("Sticker Sales")

sns.histplot(data=df_train, x='num_sold', hue='country', ax=axes[0]);

sns.histplot(data=df_train, x='num_sold', hue='store', ax=axes[1]);

sns.histplot(data=df_train, x='num_sold', hue='product', ax=axes[2]);


missing_values_count = df_train.isnull().sum()
missing_values_count


df_train.head(10)


df_test.head()


missing_values_count = df_test.isnull().sum()
missing_values_count


df_train.head(2)


encoder=LabelEncoder()
df_train['country']=encoder.fit_transform(df_train['country'])
df_train['store']=encoder.fit_transform(df_train['store'])
df_train['product']=encoder.fit_transform(df_train['product'])


X = df_train.drop(['num_sold','date','id'], axis=1)
y = df_train['num_sold']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("The size of the input train data is: {}".format(X_train.shape))
print("The size of the output train data is: {}".format(y_train.shape))
print("The size of the input test data is: {}".format(X_test.shape))
print("The size of the output test data is: {}".format(y_test.shape))


scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_new = scaler.transform(X_train)
X_test_new = scaler.transform(X_test)


train_sample_size = X_train_new.shape[0]
train_time_steps  = X_train_new.shape[1] 

test_sample_size = X_test_new.shape[0]
test_time_steps  = X_test_new.shape[1]
input_dimension = 1               

train_data_reshaped = X_train_new.reshape(train_sample_size,train_time_steps,input_dimension)
test_data_reshaped = X_test_new.reshape(test_sample_size,test_time_steps,input_dimension)

print("After reshape train data set shape:\n", train_data_reshaped.shape)
print("1 Sample shape:\n",train_data_reshaped[0].shape)

print("After reshape test data set shape:\n", test_data_reshaped.shape)
print("1 Sample shape:\n",test_data_reshaped[0].shape)


n_timesteps = train_data_reshaped.shape[1]
n_features  = train_data_reshaped.shape[2]


model = Sequential()
model.add(GRU(50,return_sequences=True,input_shape=(n_timesteps, n_features)))
model.add(Dropout(0.2))

model.add(GRU(50,return_sequences=True,input_shape=(n_timesteps, n_features)))
model.add(Dropout(0.4))

model.add(LSTM(50,input_shape=(n_timesteps, n_features)))
model.add(Dropout(0.5))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


callbacks = [
        EarlyStopping(monitor='val_loss', patience=5),
    ]


history = model.fit(train_data_reshaped, y_train, epochs=50, validation_data =(test_data_reshaped, y_test), callbacks=callbacks, verbose=1)


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.plot(history.history['mae'], label='Training mae')
plt.plot(history.history['val_mae'], label='Validation mae')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


y_predict = model.predict(X_test)
y_predict = pd.DataFrame(y_predict, columns = ['Predicted_num_sold'])
results = pd.concat([y_predict, y_test.to_frame().reset_index(drop = True)], axis = 1, ignore_index = False)
results.head()
print(results.head(),'\n')

[loss, mae] = model.evaluate(test_data_reshaped, y_test, verbose=0)
print("Testing set MAE: ", mae)


df_test=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
df_test.head(10)


encoder=LabelEncoder()
df_test['country']=encoder.fit_transform(df_test['country'])
df_test['store']=encoder.fit_transform(df_test['store'])
df_test['product']=encoder.fit_transform(df_test['product'])


X=df_test.drop(columns=['id', 'date'])
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)


y_pred = model.predict(X)


df_target = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
df_target.head()


test_preds_final = y_pred.copy()
submission_file = df_test.reset_index()[['id']]
submission_file['Predicted num_sold'] = test_preds_final
submission_file = submission_file.set_index("id")
submission_file.head()


submission_file.to_csv("/kaggle/working/submission.csv")

