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


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


train.tail(3)


train.info()


train.describe().loc[['mean','min','max']].T


plt.figure(figsize=(12,7))
sns.heatmap(train.corr(),annot=True,cmap='summer');


from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import pearsonr

train.plot("temparature","maxtemp",style='o') 
print("Pearson correlation:",train["temparature"].corr(train["maxtemp"]))
print("T Test and P value:",stats.ttest_ind(train["temparature"],train["maxtemp"]))


train.plot("winddirection","maxtemp",style='o') 
print("Pearson correlation:",train["winddirection"].corr(train["maxtemp"]))
print("T Test and P value:",stats.ttest_ind(train["winddirection"],train["maxtemp"]))


sns.set(style="darkgrid")
fig,axs=plt.subplots(2,2,figsize=(10,8))
sns.histplot(data=train,x="temparature",kde=True,ax=axs[0,0],color='green')
sns.histplot(data=train,x="maxtemp",kde=True,ax=axs[0,1],color='red')
sns.histplot(data=train,x="mintemp",kde=True,ax=axs[1,0],color='skyblue')
sns.histplot(data=train,x="dewpoint",kde=True,ax=axs[1,1],color='orange');


sns.set(style="darkgrid")
fig,axs=plt.subplots(2,2,figsize=(10,8))
sns.violinplot(data=train,x="temparature",kde=True,ax=axs[0,0],color='green')
sns.violinplot(data=train,x="maxtemp",kde=True,ax=axs[0,1],color='red')
sns.violinplot(data=train,x="mintemp",kde=True,ax=axs[1,0],color='skyblue')
sns.violinplot(data=train,x="dewpoint",kde=True,ax=axs[1,1],color='yellow');


plt.figure(figsize=(7,2))
train['rainfall'].value_counts().plot(kind='barh', color='black')
plt.title('Is it going to rain?');


from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# Prepare train (features and target)
x = train.drop("rainfall", axis=1) 
y = train["rainfall"] 


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=2)


knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
print("KNN Accuracy:{:.2f}%".format(knn.score(x_test,y_test)*100))


svm=SVC()
svm.fit(x_train,y_train)
print("SVM Accuracy:{:.2f}%".format(svm.score(x_test,y_test)*100))


gbc=GradientBoostingClassifier(subsample=0.5,n_estimators=450,max_depth=5,max_leaf_nodes=25)
gbc.fit(x_train,y_train)
print("Gradient Boosting Accuracy:{:.2f}%".format(gbc.score(x_test,y_test)*100))


xgb=XGBClassifier()
xgb.fit(x_train,y_train)
print("XGB Accuracy:{:.2f}%".format(xgb.score(x_test,y_test)*100))

