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


import pandas as pd
import matplotlib.pyplot as plt


train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv', index_col='ID')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv', index_col='ID')


#1.race_group
vc = train.race_group.value_counts()
plt.pie(vc, labels=vc.index)
plt.show()


#2.donor age and age at hct


plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.hist(train.donor_age, bins=50, color='skyblue')
plt.title('Donor age histogram')
plt.xlabel('donor_age')
plt.ylabel('count')
plt.subplot(1, 2, 2)
plt.title('Patient age histogram')
plt.hist(train.age_at_hct, bins=50, color='skyblue')
plt.xlabel('age_at_hct')
plt.tight_layout()
plt.savefig('a.png')
plt.show()


train.age_at_hct.value_counts().sort_values(ascending=False).head()


train.donor_age.value_counts().sort_values(ascending=False).head()


import matplotlib.pyplot as plt
import seaborn as sns

# 以 efs_time 为例，查看不同种族的分布
sns.kdeplot(train[train['race_group'] == 'White']['efs_time'], label='White', shade=True)
sns.kdeplot(train[train['race_group'] == 'More than one race']['efs_time'], label='Black', shade=True)
sns.kdeplot(train[train['race_group'] == 'Asian']['efs_time'], label='Asian', shade=True)
sns.kdeplot(train[train['race_group'] == 'Black or African-American']['efs_time'], label='Black or African-American', shade=True)
sns.kdeplot(train[train['race_group'] == 'Native Hawaiian or other Pacific Islander']['efs_time'], label='American Indian or Alaska native', shade=True)
sns.kdeplot(train[train['race_group'] == 'American Indian or Alaska native']['efs_time'], label='American Indian or Alaska native', shade=True)
plt.legend()
plt.title("EFS Time Distribution by Race")
plt.show()


