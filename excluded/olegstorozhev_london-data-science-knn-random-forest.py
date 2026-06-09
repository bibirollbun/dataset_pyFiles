import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings('ignore')


train=pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv', header=None)
test=pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv', header=None)
train_labels=pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv', header=None)
train


train_labels


train.columns=['col_'+str(i) for i in range(1,train.shape[1]+1)]
test.columns=['col_'+str(i) for i in range(1,train.shape[1]+1)]
train['label']=train_labels
train


train.describe().T


plt.figure(figsize=(10,8))
sns.heatmap(train.corr(),cmap='coolwarm')


plt.figure(figsize=(10,8))
plt.grid(True)
sns.countplot(data=train,x='label')


plt.figure(figsize=(10,8))
plt.grid(True)
sns.violinplot(data=train,x='label',y='col_13')


plt.figure(figsize=(10,8))
plt.grid(True)
sns.violinplot(data=train,x='label',y='col_15')


plt.figure(figsize=(10,8))
plt.grid(True)
sns.scatterplot(data=train, x='col_13',y='col_29', hue='label')


plt.figure(figsize=(10,8))
plt.grid(True)
sns.scatterplot(data=train, x='col_13',y='col_15', hue='label')


plt.figure(figsize=(10,8))
plt.grid(True)
sns.scatterplot(data=train, x='col_13',y='col_23', hue='label')


from sklearn.ensemble import RandomForestClassifier


X=train.drop('label',axis=1)
y=train['label']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)


errors = list()
for i in range(50,501,50):
    clf = RandomForestClassifier(n_estimators=i, max_depth=5, random_state=42)
    clf.fit(X_train,y_train)
    pred=clf.predict(X_test)
    errors.append(np.mean(pred!=y_test))


plt.figure(figsize=(10,8))
plt.plot(range(50,501,50),errors, '-o')
plt.grid(True)
plt.ylabel('Error')
plt.xlabel('n_estimators')


clf = RandomForestClassifier(n_estimators=250, max_depth=5, random_state=42)
clf.fit(X_train,y_train)
pred=clf.predict(X_test)


print(confusion_matrix(y_test,pred),'\n',classification_report(y_test,pred))


from sklearn.neighbors import KNeighborsClassifier


errors=[]
for i in range(1,50):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred=knn.predict(X_test)
    errors.append(np.mean(pred!=y_test))


plt.figure(figsize=(10,8))
plt.plot(range(1,50),errors, '-o')
plt.grid(True)
plt.ylabel('Error')
plt.xlabel('Number of neighbors')


knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
print(confusion_matrix(y_test,pred),'\n',classification_report(y_test,pred))


pred=knn.predict(test)

submission = pd.DataFrame({
    "Id": range(1, len(pred)+1),
    "Solution": pred
})

submission.to_csv("submission.csv", index=False)




