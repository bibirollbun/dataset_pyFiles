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


test=pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv")
a = pd.DataFrame(test.unique_id.value_counts()).reset_index()
aa=a[a['count']>13]['unique_id'].to_list()


rohlik=pd.read_csv("/kaggle/input/bvn-rohlik/valid_sales.csv")
rohlik['date'] = pd.to_datetime(rohlik['date'])



rolik_sales=pd.DataFrame(rohlik.sales.describe())
rolik_sales


last_date = train['date'].max()

cutoff_date = last_date - pd.Timedelta(days=14)
cutoff_date


valid = train[train['date'] >= cutoff_date]
valid[valid['unique_id'].isin(aa)]
valid.to_csv('valid_sales.csv', index=False)


valid['date'].max()


train = train[train['date'] < cutoff_date]
train.to_csv('train_sales.csv', index=False)


train['date'].max()




