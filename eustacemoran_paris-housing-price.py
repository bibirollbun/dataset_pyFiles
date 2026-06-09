import numpy as np
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('/kaggle/input/playground-series-s3e6/train.csv')


df.head()


df.isna().sum()


X= df.drop(['price'],axis =1)
y = df['price']
X


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)


test = pd.read_csv('/kaggle/input/playground-series-s3e6/test.csv')


X1 = df.drop(['price'],axis =1)
y1 = df['price']


lr.score(X1,y1)


from sklearn.ensemble import RandomForestRegressor

model= RandomForestRegressor(max_depth=15) 
model.fit(X,y)


model.score(X1,y1)


model.predict(X1)


test_predict=model.predict(test) 


prediction=pd.DataFrame({
    'id':test['id'],
    'price':test_predict
})
prediction.head()


prediction.to_csv('submission.csv',index=False)

