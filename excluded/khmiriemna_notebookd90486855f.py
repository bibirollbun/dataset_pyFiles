# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


training = pd.read_csv('/kaggle/input/orbyx-ml-challenge-star-system-classification/train.csv')
test = pd.read_csv('/kaggle/input/orbyx-ml-challenge-star-system-classification/test.csv')

training['train_test'] = 1
test['train_test'] = 0
test['galaxy_type'] = np.NaN
all_data = pd.concat([training,test])

%matplotlib inline
all_data.columns


training.info()


training.describe().columns


df_num = training[['star_size','star_brightness','distance_from_earth','star_mass','metallicity']]
df_cat = training[['galaxy_region','galaxy_type','system_type','star_spectral_class', 'planet_configuration','stellar_activity_class']]


df_num = df_num.replace([np.inf, -np.inf], np.nan).dropna()
for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()


print(df_num.corr())
sns.heatmap(df_num.corr())


pd.pivot_table(training, index = 'galaxy_type', values = ['star_size','star_brightness','distance_from_earth','star_mass','metallicity'])


for i in df_cat.columns:
    counts = df_cat[i].value_counts()
    sns.barplot(x=counts.index, y=counts.values)
    plt.title(i)
    plt.show()



mapping = {'low': 0, 'medium': 1, 'high': 2}
training['stellar_activity_num'] = training['stellar_activity_class'].map(mapping)

