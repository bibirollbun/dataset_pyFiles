# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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


data=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
data



data.info()


data.describe()


import matplotlib.pyplot as plt 
import seaborn as sns


# a=data.columns
# for i in a:
#     print(f'boxplot for {i}')
#     sns.boxplot(data[i])
#     plt.show()
    


# fig ,ax=plt.subplots(4,3,figsize=(15,10))
# ax = ax.flatten()

# for i, col in enumerate(a):
#     sns.histplot(data[col], ax=ax[i],kde=True)
#     ax[i].set_title(col)

# plt.tight_layout()
# plt.show()


# data2=data.sample(200)
# fig ,ax=plt.subplots(4,3,figsize=(15,10))
# ax = ax.flatten()

# for i, col in enumerate(a):
#     sns.scatterplot(data=data2,x='BeatsPerMinute',y=data2[col], ax=ax[i])
#     ax[i].set_title(col)

# plt.tight_layout()
# plt.show()


# plt.hexbin(data=data,x='Energy',y='BeatsPerMinute', gridsize=25, cmap='Blues')


plt.figure(figsize=(15,5))
sns.heatmap(data.corr(),annot=True)


from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

t=DecisionTreeRegressor(max_depth=5,min_impurity_decrease=0.01)


col=['TrackDurationMs','AudioLoudness']
p=ColumnTransformer(transformers=[
    ('std',StandardScaler(),col)
],remainder='passthrough')
pipe=Pipeline(steps=[
    ('pre',p),
    ('model',t)
])
pipe

model1_data=data.drop(columns=['id','BeatsPerMinute'])
test=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
model1_test=test.drop(columns=['id'])
y=data['BeatsPerMinute']

pipe.fit(model1_data,y)
pred=pipe.predict(model1_test)
pred






# num_cols = ['TrackDurationMs','AudioLoudness','MoodScore']  # numeric features
# target = 'BeatsPerMinute'

# for col in num_cols:
#     plt.figure(figsize=(6,4))
#     sns.regplot(x=data[col], y=data[target], line_kws={"color":"red"})
#     plt.title(f'{col} vs {target}')
#     plt.show()


importance = pd.DataFrame({'Feature': model1_data.columns, 'Importance': t.feature_importances_})
importance.sort_values(by='Importance', ascending=False)


temp_df1=data[['TrackDurationMs','AcousticQuality','RhythmScore','Energy','VocalContent']]
temp_df2=test[['TrackDurationMs','AcousticQuality','RhythmScore','Energy','VocalContent']]


t=DecisionTreeRegressor(max_depth=5,min_impurity_decrease=0.01)
t.fit(temp_df1,data['BeatsPerMinute'])
pred=t.predict(temp_df2)


pred


from sklearn.neighbors import KNeighborsRegressor
k=KNeighborsRegressor(n_neighbors=5)
k.fit(temp_df1,data['BeatsPerMinute'])
pred=k.predict(temp_df2)
pd.DataFrame({
    'id':test['id'],
    'BeatsPerMinute':pred
}).to_csv('submission.csv',index=False)

