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
import re
%matplotlib inline


df=pd.read_csv("/kaggle/input/web-traffic-time-series-forecasting/train_1.csv.zip")
df.head()


# to get information about columns
df.info()


key1=pd.read_csv("/kaggle/input/web-traffic-time-series-forecasting/key_1.csv.zip")
key1.head()


#preprocessing, fill missing values
df = df.fillna(0)


#using a simple regular expression to search for the language code in 
#the wikipedia URL.
def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        return res[0][0:2]
    return 'na'

df['lang'] = df.Page.map(get_language)

from collections import Counter

print(Counter(df.lang))


lang_sets = {}
lang_sets['en'] = df[df.lang=='en'].iloc[:,0:-1]
lang_sets['ja'] = df[df.lang=='ja'].iloc[:,0:-1]
lang_sets['de'] = df[df.lang=='de'].iloc[:,0:-1]
lang_sets['na'] = df[df.lang=='na'].iloc[:,0:-1]
lang_sets['fr'] = df[df.lang=='fr'].iloc[:,0:-1]
lang_sets['zh'] = df[df.lang=='zh'].iloc[:,0:-1]
lang_sets['ru'] = df[df.lang=='ru'].iloc[:,0:-1]
lang_sets['es'] = df[df.lang=='es'].iloc[:,0:-1]


lang_sets['en']


# to see total number of view
sums = {}
for key in lang_sets:
    sums[key] = lang_sets[key].iloc[:,1:].sum(axis=0) / lang_sets[key].shape[0]


#plot all the different sets on the same plot to see changes over time
days = [r for r in range(sums['en'].shape[0])]

fig = plt.figure(1,figsize=[10,10])
plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title('Pages in Different Languages')
labels={'en':'English','ja':'Japanese','de':'German',
        'na':'Media','fr':'French','zh':'Chinese',
        'ru':'Russian','es':'Spanish'
       }

for key in sums:
    plt.plot(days,sums[key],label = labels[key] )
    
plt.legend()
plt.show()


# For each language get highest few pages
npages = 5
top_pages = {}
for key in lang_sets:
    print(key)
    sum_set = pd.DataFrame(lang_sets[key][['Page']])
    sum_set['total'] =lang_sets[key].select_dtypes(include='number').sum(axis=1)
    sum_set = sum_set.sort_values('total',ascending=False)
    print(sum_set.head(10))
    top_pages[key] = sum_set.index[0]
    print('\n\n')


for key in top_pages:
    fig = plt.figure(1,figsize=(10,5))
    cols = df.columns
    cols = cols[1:-1]
    data = df.loc[top_pages[key],cols]
    plt.plot(days,data)
    plt.xlabel('Days')
    plt.ylabel('Views')
    plt.title(df.loc[top_pages[key],'Page'])
    plt.show()


#statsmodels package includes quite a few tools for doing time series analysis. Here, I show the autocorrelation and 
#partial autocorrelation for the most-viewed page for each language.

from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf


for key in top_pages:
    fig = plt.figure(1,figsize=[10,5])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    cols = df.columns[1:-1]
    data = np.array(df.loc[top_pages[key],cols])
    data_diff = [data[i] - data[i-1] for i in range(1,len(data))]
    autocorr = acf(data_diff)
    pac = pacf(data_diff)
    x = [x for x in range(len(pac))]
    ax1.plot(x[1:],autocorr[1:])

    ax2.plot(x[1:],pac[1:])
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelation')
    ax1.set_title(df.loc[top_pages[key],'Page'])

    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Partial Autocorrelation')
    plt.show()



from statsmodels.tsa.arima.model import ARIMA  # ✅ NEW version
import warnings
import numpy as np
import matplotlib.pyplot as plt

cols = df.columns[1:-1]

for key in top_pages:
    page_name = df.loc[top_pages[key], 'Page']
    data = np.array(df.loc[top_pages[key], cols], dtype='f')

    if len(data) < 10 or np.all(data == data[0]):
        print(page_name)
        print('\tSkipped: too short or constant')
        continue

    result = None
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            arima = ARIMA(data, order=(2, 1, 4))
            result = arima.fit()  # ✅ NO disp
        except Exception as e1:
            try:
                arima = ARIMA(data, order=(2, 1, 2))
                result = arima.fit()  # ✅ NO disp
            except Exception as e2:
                print(page_name)
                print(f'\tARIMA failed: {e2}')
                continue

    if result is not None:
        try:
            pred = result.predict(start=2, end=599)
            x_data = list(range(2, len(data)))
            x_pred = list(range(2, 2 + len(pred)))

            plt.plot(x_data, data[2:], label='Data')
            plt.plot(x_pred, pred, label='ARIMA Model')
            plt.title(page_name)
            plt.legend()
            plt.show()
        except Exception as e3:
            print(page_name)
            print(f'\tPrediction/plotting failed: {e3}')



def plot_entry(key,idx):
    data = lang_sets[key].iloc[idx,1:]
    fig = plt.figure(1,figsize=(10,5))
    plt.plot(days,data)
    plt.xlabel('day')
    plt.ylabel('views')
    plt.title(df.iloc[lang_sets[key].index[idx],0])
    
    plt.show()


# for english data
idx = [1, 5, 10, 50, 100]
for i in idx:
    plot_entry('en',i)


#submission
submission=pd.read_csv("/kaggle/input/web-traffic-time-series-forecasting/sample_submission_1.csv.zip")
submission


submission.shape


submission ['Visits']= y_pred


# save prediction
submission.to_csv("submission.csv", index=None)

