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


df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df.info()
df.head()
df.describe().T


print(df.head())
df.shape


df.isnull().sum()



#outlier detection
import seaborn as sns
import matplotlib.pyplot as plt
#sns.boxplot(df['pressure'])
for column in df.columns:
    plt.figure(figsize= (6,4))
    sns.boxplot(df[column])
    plt.title(f'{column}')
    plt.show()
    


plt.figure(figsize =(14,7))
sns.boxplot(df)
plt.show()


co_relation = df.corr()
plt.figure(figsize= (15,7))
sns.heatmap(co_relation,cmap="Blues",vmin = -1, vmax = 1, center = 0, annot=True, fmt=".2f", square=True, linewidths=.5)
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier

x = df.drop(columns='rainfall')
y = df['rainfall']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=42)
x_train.shape
x_test.shape

#using decision stumps and adaboost
ada = AdaBoostClassifier(base_estimator= DecisionTreeClassifier(max_depth=1),n_estimators=100)
ada.fit(x_train,y_train)

y_pred = ada.predict(x_test)
print("accuracy:",accuracy_score(y_test,y_pred))


kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(x_train,y_train)
y_pred4 =kn.predict(x_test)
print("accuracy: ",accuracy_score(y_test,y_pred))


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(max_iter= 100000, class_weight= "balanced" ,random_state= 42,solver='liblinear')
lg.fit(x_train,y_train)
y_pred2 =lg.predict(x_test)
print("accuracy: ", accuracy_score(y_test,y_pred2))


from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier()
#rf.fit(x_train,y_train)

param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}


rf = RandomForestClassifier()

rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)
rand_search.fit(x_train, y_train)


best_rf = rand_search.best_estimator_

print('Best hyperparameters:',  rand_search.best_params_)


y_pred1 = rand_search.predict(x_test)
print("accuracy: ", accuracy_score(y_test,y_pred1))


#ensemble using the above 3 ada, lg and rf
from sklearn.ensemble import VotingClassifier
em = VotingClassifier(estimators=[('lg',lg),('rf',rand_search),('ada',ada),('kn',kn)],
                     voting='hard')
em.fit(x_train,y_train)



y_pred3 = em.predict(x_test)
print("accuracy: ", accuracy_score(y_test,y_pred3))


df1 = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


df1.isnull().sum()


df1.describe().T


gf = df1.plot.scatter(x="id",y="winddirection")


#let us add the previous value as the wind direction missing value_ff fill
df1['winddirection'].fillna(method="bfill",inplace=True)
df1.isnull().sum()


y1 = em.predict(df1)
y1 = pd.DataFrame(y1,columns=["rainfall"])
final =pd.concat([df1.id,y1],axis=1)
final.set_index('id',inplace=True)
final.to_csv(f"final_pred_em.csv")

