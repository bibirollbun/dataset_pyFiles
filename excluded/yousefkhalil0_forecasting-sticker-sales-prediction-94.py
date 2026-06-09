import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv(r'/kaggle/input/playground-series-s5e1/train.csv').dropna()
test=pd.read_csv(r'/kaggle/input/playground-series-s5e1/test.csv').dropna()



df=df.drop(columns=['id','date'])
test=test.drop(columns=['date'])


df.head()


encode=LabelEncoder()
for i in df.select_dtypes(exclude=[np.number]):
    df[i]=encode.fit_transform(df[i])
    test[i]=encode.fit_transform(test[i])

df.head()


x=df.iloc[:,:-1]
y=df.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=44)

model=RandomForestRegressor(n_estimators=200,max_depth=50)
model.fit(x_train,y_train)
predict=model.predict(x_test)

print('Score For Training : ',model.score(x_train,y_train))
print('Score For Testing : ',model.score(x_test,y_test))
print("Accuracy : ",r2_score(y_test,predict))
print('mean_squared_error : ',mean_squared_error(y_test,predict))


test_f=test.drop(columns=['id'])
test_pred=model.predict(test_f)


test_pred=pd.Series(test_pred)
test_pred


test['num_sold']=test_pred
test.head()


submission = test[['id', 'num_sold']]
submission.to_csv('submission.csv')

