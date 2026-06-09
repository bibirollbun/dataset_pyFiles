# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline


import warnings
warnings. filterwarnings('ignore')

from sklearn.utils import resample

#import models
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train_df.head()


train_df.info()


def engineer_features(df):
    df['Stage_fear'] = df['Stage_fear'].fillna(
    pd.Series(
        np.where(df['Post_frequency'] > 5, 'No', 'Yes'),
        index=df.index)
            )

    #drop null rows
    df.dropna(inplace = True)

    df['Stage_fear'] = (df['Stage_fear'] == 'Yes').astype(int)
    df['Drained_after_socializing'] = (df['Drained_after_socializing'] == 'Yes').astype(int)
    
    df['is_social_media_active'] = (df['Post_frequency'] > 5).astype(int)

    df = df.reset_index(drop=True)

    return df


train_df = engineer_features(train_df)


map_target = {'Extrovert' : 1, 'Introvert' : 0}

train_df['Personality'] = train_df['Personality'].replace(map_target)


cols = train_df.drop(['Personality','id'], axis=1).columns

# set rows and cols
n_rows = (len(cols) + 1) // 2   # adjust rows to fit all features
n_cols = 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 25))

# flatten axes for easy iteration
axes = axes.flatten()

for i, col in enumerate(cols):
    sb.histplot(train_df[col], kde=True, bins=20, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')

# remove any empty axes (if number of features not multiple of 2)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



plt.figure(figsize = (10,10))
sb.heatmap(train_df.corr(),annot = True);


train_df['Personality'].value_counts()


#choose class size

class_size = 6000 


extrovert = train_df[train_df['Personality'] == 1]
introvert = train_df[train_df['Personality'] == 0]


downsampled_ = resample(extrovert,
                          replace = False,
                          n_samples = class_size,
                          random_state = 27)


upsampled_ = resample(introvert,
                              replace = True,  #sample with replacement(we want to duplicate observation)
                              n_samples = class_size,
                              random_state = 27)


# Combine both to get a balanced dataset
balanced_df = pd.concat([downsampled_, upsampled_])


balanced_df = balanced_df.sort_values(by = 'id')


X = train_df.drop('Personality', axis = 1)
y = train_df['Personality']


sc = StandardScaler()
X_scaled = sc.fit_transform(X)


X_train,X_val,y_train, y_val = train_test_split(X_scaled,y, test_size = 0.2, random_state = 42)


rf= RandomForestClassifier()
rf.fit(X_train, y_train)

rf_model = rf.predict(X_val)


xg = XGBClassifier()
xg.fit(X_train,y_train)

xg_model = xg.predict(X_val)


lr = LogisticRegression()
lr.fit(X_train,y_train)

lr_model = lr.predict(X_val)


lgbm = LGBMClassifier()
lgbm.fit(X_train,y_train)
lgbm_model = lgbm.predict(X_val)


accuracy_score(y_val, lgbm_model)


test_df = engineer_features(test_df)


X_test_scaled = sc.transform(test_df)


final_model = lgbm.predict(X_test_scaled)


test_df['Personality'] = final_model


submission_df = test_df[['id','Personality']]


remap_target = {1 : 'Extrovert', 0 : 'Introvert'}
submission_df['Personality'] = submission_df['Personality'].replace(remap_target)


submission_df.to_csv('submission.csv', index=False)







