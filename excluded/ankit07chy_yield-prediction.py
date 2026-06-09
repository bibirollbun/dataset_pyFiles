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


import seaborn as sns
import matplotlib.pyplot as plt


sample_submission = pd.read_csv('/kaggle/input/pascmlsig/sample_submission.csv')
train = pd.read_csv('/kaggle/input/pascmlsig/train.csv')
test = pd.read_csv('/kaggle/input/pascmlsig/test.csv')


train.head()


sns.lineplot(train,x='clonesize',y='yield')


train.shape


train.info()


plt.figure(figsize=(20,20))
sns.heatmap(train.corr(),annot=True)
plt.show()


# here I run a model 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error


y = train.iloc[:,18:]
X = train.iloc[:,1:18].values


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


X_train.shape


lr = LinearRegression()


lr.fit(X_train,y_train)


import statsmodels.api as sm


# X_train = sm.add_constant(X_train)
# model = sm.OLS(y_train,X_train).fit()
# print(model.summary())


y_predict = lr.predict(X_test)


r2 = r2_score(y_test,y_predict)


X_train.shape


#adjusted r2 score
1 - ((1-r2)*((18-1)/(18-2)))


mean_absolute_error(y_test,y_predict)


X = test.iloc[:,1:18].values


y_predict = lr.predict(X)



sample_submission


id = test['id']


id


y_predict


print("id shape:", np.shape(id))
print("y_predict shape:", np.shape(y_predict))



from sklearn.linear_model import Ridge
r = Ridge(alpha=2)


r.fit(X_train,y_train)


r.coef_


X_train.shape


X_test.shape


y_pred1 = r.predict(X_test)


y_pred1


y_predict


print(len(y_test))       # Should be the same as below
print(len(y_predict)) 


y_pred_L = lr.predict(X_test)


print(r2_score(y_test,y_pred1))
print(r2_score(y_test,y_pred_L))


print(mean_absolute_error(y_test,y_pred1))
print(mean_absolute_error(y_test,y_pred_L))


from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train,y_train)
y_pred2 = lasso.predict(X_test)
print(mean_absolute_error(y_test,y_pred1))
print(mean_absolute_error(y_test,y_pred_L))
print(mean_absolute_error(y_test,y_pred2))
print('For r2 score')
print(r2_score(y_test,y_pred1))
print(r2_score(y_test,y_pred_L))
print(r2_score(y_test,y_pred2))


test


id = test['id']


X_test.shape


testing = test.iloc[:,1:]


price = lasso.predict(testing)


price = price.ravel()


id.shape


final = pd.DataFrame({
    'id':id,
    'price':price,
})


final.to_csv('submission.csv',index=False)




