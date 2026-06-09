import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


train=pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv',header=None)
test=pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv',header=None)
train_labels=pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv',header=None)
train.columns=['Col'+str(i) for i in range(1,train.shape[1]+1)]
test.columns=['Col'+str(i) for i in range(1,train.shape[1]+1)]
train['Label']=train_labels
train


train.info()


train.describe()


sns.countplot(data=train,x='Label',palette='viridis')


train.corr()['Label'].sort_values(ascending=False).iloc[1:].head()


sns.distplot(train['Col15'],color='red')


sns.boxplot(data=train,x='Label',y='Col15',palette='rainbow')


plt.scatter(data=train,x='Col13',y='Col15',c='Label')


sns.jointplot(data=train,x='Col13',y='Col15',kind='hex',color='purple')


plt.figure(figsize=(12,10))
sns.heatmap(train.corr(),cmap='coolwarm')


from sklearn.model_selection import train_test_split


X=train.drop('Label',axis=1)
y=train['Label']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)


from sklearn.linear_model import LogisticRegression


log_model=LogisticRegression()
log_model.fit(X_train,y_train)
pred=log_model.predict(X_test)


from sklearn.metrics import classification_report,confusion_matrix


print(confusion_matrix(y_test,pred),'\n',classification_report(y_test,pred))


from sklearn.neighbors import KNeighborsClassifier


error=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred=knn.predict(X_test)
    error.append(np.mean(pred!=y_test))


plt.plot(range(1,40),error,'r--o',markerfacecolor='blue')
plt.ylabel('Error')
plt.xlabel('No. of neighbors')


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)


print(confusion_matrix(y_test,pred),'\n',classification_report(y_test,pred))


from sklearn.tree import DecisionTreeClassifier


dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
pred=dtree.predict(X_test)


print(confusion_matrix(y_test,pred),'\n',classification_report(y_test,pred))


from sklearn.ensemble import RandomForestClassifier


rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
pred=rfc.predict(X_test)


print(confusion_matrix(y_test,pred),'\n',classification_report(y_test,pred))


from sklearn.svm import SVC


svc=SVC()
svc.fit(X_train,y_train)
pred=svc.predict(X_test)


print(confusion_matrix(y_test,pred),'\n',classification_report(y_test,pred))


from xgboost import XGBClassifier


xgb=XGBClassifier()
xgb.fit(X_train,y_train)
pred=xgb.predict(X_test)


print(confusion_matrix(y_test,pred),'\n',classification_report(y_test,pred))


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
pred=knn.predict(test)


final=pd.DataFrame({'Id':np.arange(1,pred.shape[0]+1),'Solution':pred}).set_index('Id')


final.to_csv('final.csv')


from sklearn.decomposition import PCA


pca=PCA(n_components=2)
X_train_pca=pca.fit_transform(X_train)
X_test_pca=pca.transform(X_test)


plt.scatter(X_train_pca[:,0],X_train_pca[:,1],c=y_train)
plt.xlabel('Component - 1')
plt.ylabel('Component - 2')

