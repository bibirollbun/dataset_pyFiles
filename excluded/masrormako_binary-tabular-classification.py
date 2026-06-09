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


train=pd.read_csv("/kaggle/input/playground-series-s3e3/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s3e3/test.csv")
submission=pd.read_csv("/kaggle/input/playground-series-s3e3/sample_submission.csv")


train


test


submission


sns.histplot(train['Attrition'])


train.info()


train.isnull().sum().sum()


target=train['Attrition']
comb=pd.concat([train.drop('Attrition',axis=1),test],ignore_index=True)
comb


comb.isnull().sum().sum()


comb.drop('id',axis=1)


for col in comb:
    if comb[col].dtype=='O':
        print(col,comb[col].unique())


for col in comb:
    if comb[col].dtype=='O':
        print(col,comb[col].nunique())


from sklearn.preprocessing import OrdinalEncoder
encode=OrdinalEncoder()

for col in comb:
    if comb[col].dtype=='O':
        comb[col]=encode.fit_transform(comb[col].values.reshape(-1,1))
    
comb


cor=comb.corr()
sns.heatmap(cor)


cor



columns=np.full((cor.shape[0],),True,dtype=bool)
for i in range(cor.shape[0]):
    for j in range (i+1,cor.shape[0]):
        if cor.iloc[i,j]>=0.5:
            if columns[j]:
                columns[j]=False
sel_col=comb.columns[columns]
comb=comb[sel_col]
comb


comb=(comb-comb.min())/(comb.max()-comb.min())
comb


comb.isnull().sum()


comb=comb.dropna(axis=1)
comb


y=target
X=comb[:len(train)]
X_test=comb[len(train):]


X['target']=target
train=X.sample(frac=0.9,random_state=42)
val=X.drop(train.index)

y_train=train.iloc[:,-1]
X_train=train.iloc[:,:-1]

y_val=val.iloc[:,-1]
X_val=val.iloc[:,:-1]


X_train.shape,y_train.shape,X_val.shape,y_val.shape,X_test.shape


X_train.isnull().sum()


from sklearn.linear_model import LogisticRegression
model=LogisticRegression(class_weight='balanced',random_state=42).fit(X_train,y_train)
print(model.score(X_train,y_train))


y_pred=model.predict(X_val)
y_prob=model.predict_proba(X_val)
print(model.score(X_val,y_val))
y_prob[:,1]


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_val,y_pred))


predictions=model.predict(X_test)
y_test_prob=model.predict_proba(X_test)
y_test_prob[:,1]


submission.Attrition=y_test_prob[:,1]
submission.to_csv('submission.csv',index=False)
submission=pd.read_csv('submission.csv')
submission

