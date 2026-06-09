train="/content/simple_regression_train.csv"
test="/content/simple_regression_test.csv"
sample="/content/simple_regression_submission.csv"


import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv(train)
df.head(10)


df.info()


plt.plot(df.loc[0:20],df.loc[0:20],color="blue")
plt.scatter(df.iloc[0:20],df.iloc[0:20],color="orange",alpha=0.6)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simple Regression")
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


X=df['t']
y=df['y']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


lr=LinearRegression()
lr.fit(X_train.values.reshape(-1,1),y_train)


y_pred=lr.predict(X_test.values.reshape(-1,1))
mean_squared_error(y_test,y_pred)
# from sklearn.metrics import r2_score
# r2_score(y_test,y_pred)


plt.scatter(X_test,y_test,color="blue")
plt.scatter(X_test,y_pred,color="orange",alpha=0.6)


# y=mx+c
import numpy as np
df1=pd.read_csv(train)


x_sum=np.sum(df1['t'])
y_sum=np.sum(df1['y'])
xy_sum=np.sum(df1['t']*df1['y'])
x_sqr_sum=np.sum(df1['t']**2)


m=(xy_sum-((x_sum*y_sum)/len(df1)))/(x_sqr_sum-((x_sum**2)/len(df1)))
c=(y_sum-m*x_sum)/len(df1)


df1_test=pd.read_csv(test)
df1_test['y']=m*df1_test['t']+c


sub=pd.DataFrame()
sub['t']=df1_test['t']
sub['y']=df1_test['y']
sub.to_csv('sub.csv',index=False)


from sklearn.ensemble import RandomForestRegressor


rf=RandomForestRegressor()
rf.fit(X_train.values.reshape(-1,1),y_train)


y_pred_rf=lr.predict(X_test.values.reshape(-1,1))
mean_squared_error(y_test,y_pred_rf)


df_test=pd.read_csv(test)
df_test.head()


y_new=lr.predict(df_test['t'].values.reshape(-1,1))


submission=pd.DataFrame()
submission['t']=df_test['t']
submission['y']=y_new
submission.to_csv('submission.csv',index=False)




