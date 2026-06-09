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


data=pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
data


data.info()


data.corr()


import seaborn as sns
import matplotlib.pyplot as plt 
feature=data.columns
for i in feature:
    plt.figure(figsize=(4,2))
    sns.boxplot(data=data,x=i)
    plt.show()




for i in feature:
    plt.figure(figsize=(6,4))
    sns.histplot(data=data,x=i,kde=True)
    plt.show()


from scipy.stats import normaltest


feature = data.columns
for i in feature:
    stat, p = normaltest(data[i])
    print(f"D'Agostino p-value {i}: {p:.10f}") 



test=pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
test.info()


data



from sklearn.preprocessing import MinMaxScaler
m=MinMaxScaler()
a=data[["maxtemp", 'temparature',"sunshine","winddirection"]]
b=test[["maxtemp", 'temparature',"sunshine","winddirection"]]
a_scaled=m.fit_transform(a)
b_scaled=m.transform(b)



data[["maxtemp", 'temparature',"sunshine","winddirection"]]=a_scaled
test[["maxtemp", 'temparature',"sunshine","winddirection"]]=b_scaled


data



data.columns


from sklearn.preprocessing import RobustScaler
r=RobustScaler()
a1=data[['pressure','temparature', 'mintemp','dewpoint', 'humidity', 'cloud','windspeed']]
b1=test[['pressure','temparature', 'mintemp','dewpoint', 'humidity', 'cloud','windspeed']]
a1_scaled=r.fit_transform(a1)
b1_scaled=r.transform(b1)


data[['pressure','temparature', 'mintemp','dewpoint', 'humidity', 'cloud','windspeed']]=a1_scaled


test[['pressure','temparature', 'mintemp','dewpoint', 'humidity', 'cloud','windspeed']]=b1_scaled


data


x=data.drop(columns=['id','day','rainfall'])
y=data["rainfall"]
testing=test.drop(columns=['id','day'])


from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(x, y, discrete_features=False)
print(mi_scores)



from sklearn.feature_selection import SelectKBest, f_classif


selector = SelectKBest(score_func=f_classif, k='all')  # or choose k best features
X_new = selector.fit_transform(x, y)

print(selector.scores_)
x.columns[selector.get_support()]


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


lr=LogisticRegression(C=0.1, max_iter=1000)
k=KNeighborsClassifier(n_neighbors=3)
X=x[[ 'temparature', 'mintemp',  'humidity',
       'cloud', 'sunshine', 'winddirection', 'windspeed']]
Y=data["rainfall"]
Testing=testing[['temparature', 'mintemp', 'humidity',
       'cloud', 'sunshine', 'winddirection', 'windspeed']]
Testing['winddirection'].fillna(Testing['winddirection'].mean(),inplace=True)


lr.fit(X,Y)
k.fit(X,Y)
y_pred_lr=lr.predict(Testing)
y_pred_k=k.predict(Testing)
y_pred_lr



y_pred_k


# pd.DataFrame({
#     "id":test["id"],
#     'rainfall':y_pred_lr
# }).to_csv("submission.csv",index=False)


# pd.DataFrame({
#     "id":test["id"],
#     'rainfall':y_pred_k
# }).to_csv("submission.csv",index=False)





!pip install mlxtend


# from mlxtend.feature_selection import ExhaustiveFeatureSelector as efs
# model =efs(
#      lr,
#      min_features=1,
#      max_features=x.shape[1],
#      scoring='accuracy',
#      cv=5
# )
# model=model.fit(x,y)
# print("Best Accuracy: %.3f" % model.best_score_)
# print("Best Feature Indices:", model.best_idx_)
# print("Best Feature Names:", model.best_feature_names_)








# a= list(model.best_feature_names_)


# a



x=x[['maxtemp',
 'temparature',
 'dewpoint',
 'humidity',
 'cloud',
 'sunshine',
 'winddirection',
 'windspeed']]
y=data["rainfall"]
testing=testing[['maxtemp',
 'temparature',
 'dewpoint',
 'humidity',
 'cloud',
 'sunshine',
 'winddirection',
 'windspeed']]


testing['winddirection'].fillna(testing['winddirection'].mean(),inplace=True)


lr.fit(x,y)
y_pred=lr.predict(testing)
y_pred_lr
y_pred_lr2=lr.predict(x)
from sklearn.metrics import accuracy_score
accuracy_score(y,y_pred_lr2)



# pd.DataFrame({
#     "id":test["id"],
#     'rainfall':y_pred
# }).to_csv("submission.csv",index=False)


from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0, gamma='scale')

# 4. Train the model
model.fit(x,y)

# 5. Make predictions
y_pred_svm = model.predict(testing)
y_pred_svm2=model.predict(x)
accuracy_score(y,y_pred_svm2)



k=KNeighborsClassifier(n_neighbors=3)
k.fit(x,y)
y_pred_knn=k.predict(testing)
y_pred_knn2=k.predict(x)
accuracy_score(y,y_pred_knn2)


train=pd.DataFrame({
    "lr":y_pred_lr2,
    "knn":y_pred_knn2,
    
    "output":data['rainfall']
})


test=pd.DataFrame({
    "lr":y_pred_lr,
    "knn":y_pred_knn,
   
  
    
})


x=train.drop(columns=['output'])
y=train['output']
model.fit(x,y)
y_pred=model.predict(test)
y_pred2=model.predict(x)
accuracy_score(y,y_pred2)


k.fit(x,y)
y_pred=k.predict(test)
y_pred2=k.predict(x)
accuracy_score(y,y_pred2)


t=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


t


test


lr.fit(x,y)
y_pred=k.predict(test)
y_pred2=k.predict(x)
accuracy_score(y,y_pred2)
# kk=pd.DataFrame({
#     'id':t['id'],
#     'rainfall':y_pred
# })


train[(train['lr']==1) & (train['knn']==0)]


test['output']=0
test.loc[(train['lr']==1) & (train['knn']==0),'output']=0
test.loc[(train['lr']==0) & (train['knn']==1),'output']=1
test.loc[(train['lr']==0) & (train['knn']==0),'output']=0
test.loc[(train['lr']==1) & (train['knn']==1),'output']=1


test['output']=kk['rainfall']


test



test.loc[(test['lr']==0) & (test['knn']==1),'output']=1


test


pd.DataFrame({
    'id':t['id'],
    'rainfall':test['output']
}).to_csv('submission.csv',index=False)

