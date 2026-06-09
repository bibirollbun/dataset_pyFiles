# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

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


df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


df.tail()


df.info()


# checking dataset

print ("Rows     : " ,df.shape[0])
print ("Columns  : " ,df.shape[1])
print ("\nFeatures : \n" ,df.columns.tolist())
print ("\nMissing values :  ", df.isnull().sum().values.sum())
print ("\nUnique values :  \n",df.nunique())


#By Lokesh https://www.kaggle.com/code/lokeshbabukolamala/predicting-human-personality-94-acc-with-eda/notebook

cols=df.columns
num_cols=[x for x in df.columns if df[x].dtypes!='O']
cat_cols=[y for y in cols if y not in num_cols]


#By Lokesh https://www.kaggle.com/code/lokeshbabukolamala/predicting-human-personality-94-acc-with-eda/notebook

for col in num_cols:
    x=df[df.Personality=='Extrovert'][col].mean()
    df.loc[df.Personality == 'Extrovert', col] = df.loc[df.Personality == 'Extrovert', col].fillna(x)
    y=df[df.Personality=='Introvert'][col].mean()
    df.loc[df.Personality == 'Introvert', col] = df.loc[df.Personality == 'Introvert', col].fillna(y)


#By Lokesh https://www.kaggle.com/code/lokeshbabukolamala/predicting-human-personality-94-acc-with-eda/notebook

for col in cat_cols:
    x=df[df.Personality=='Extrovert'][col].mode()[0]
    df.loc[df.Personality == 'Extrovert', col] = df.loc[df.Personality == 'Extrovert', col].fillna(x)
    y=df[df.Personality=='Introvert'][col].mode()[0]
    df.loc[df.Personality == 'Introvert', col] = df.loc[df.Personality == 'Introvert', col].fillna(y)


df.isnull().sum()


#By Lokesh https://www.kaggle.com/code/lokeshbabukolamala/predicting-human-personality-94-acc-with-eda/notebook

for col in cat_cols:
    ax=sns.countplot(x=col,data=df,hue='Personality',palette='dark')
    for container in ax.containers:
        ax.bar_label(container)
    plt.show()


#By Lokesh https://www.kaggle.com/code/lokeshbabukolamala/predicting-human-personality-94-acc-with-eda/notebook

sns.set(style="darkgrid")
for col in num_cols:
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    sns.histplot(x=col,kde=True,data=df[df.Personality=="Introvert"],bins=df[col].nunique())
    plt.title(f'Distribution of {col} w.r.t  Introvert')
    plt.subplot(1,2,2)
    sns.histplot(x=col,kde=True,data=df[df.Personality=="Extrovert"],bins=df[col].nunique())
    plt.title(f'Distribution of {col} w.r.t  Extrovert')
    plt.show()


plt.figure(figsize=(16,8))
sns.heatmap(df.corr(numeric_only=True),cmap='summer',annot=True,linewidths=0.75)
plt.show()


#By Lokesh https://www.kaggle.com/code/lokeshbabukolamala/predicting-human-personality-94-acc-with-eda/notebook

stage_fear_map={'Yes':1,'No':0}
df['Stage_fear']=df['Stage_fear'].map(stage_fear_map)
soc_map={'Yes':1,'No':0}
df['Drained_after_socializing']=df['Drained_after_socializing'].map(soc_map)
personality_map={'Introvert':1,'Extrovert':0}
df['Personality']=df['Personality'].map(personality_map)


#By Lokesh https://www.kaggle.com/code/lokeshbabukolamala/predicting-human-personality-94-acc-with-eda/notebook

for col in cols:
    print(df[col].value_counts())
    print(f'\n{ "-" * 20 }\n')


#By Lokesh https://www.kaggle.com/code/lokeshbabukolamala/predicting-human-personality-94-acc-with-eda/notebook

for col in num_cols:
    df[col]=df[col].astype(int)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


#By Lokesh https://www.kaggle.com/code/lokeshbabukolamala/predicting-human-personality-94-acc-with-eda/notebook

X=df.drop('Personality',axis=1)
y=df.Personality


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=9)


#By Lokesh https://www.kaggle.com/code/lokeshbabukolamala/predicting-human-personality-94-acc-with-eda/notebook

model=LogisticRegression()
model.fit(X_train,y_train)
y_train_pred=model.predict(X_train)
accuracy_score(y_train,y_train_pred)


#By Lokesh https://www.kaggle.com/code/lokeshbabukolamala/predicting-human-personality-94-acc-with-eda/notebook

y_test_pred=model.predict(X_test)
accuracy_score(y_test,y_test_pred)


#By Lokesh https://www.kaggle.com/code/lokeshbabukolamala/predicting-human-personality-94-acc-with-eda/notebook

sns.heatmap(confusion_matrix(y_test,y_test_pred),annot=True,linewidth=0.75,fmt='d')
plt.show()


print(classification_report(y_test,y_test_pred))

