path="/content/dataset.csv"
test="/content/test.csv"
sample="/content/sample_submission.csv"


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


df=pd.read_csv(path)
df.head()


df.info()


plt.hist(df['sale_price'],alpha=0.5)
plt.show()


# Drop attributes which do not have any affect on prediction
def drop_attributes(df):
  return df.drop(['id','sale_nbr','sale_warning','join_status','join_year','subdivision'],axis=1)
df=drop_attributes(df)


# There are some null values, lets work on them
df.isna().sum()


df['submarket'].unique()


df['submarket']=df['submarket'].fillna(df['submarket'].mode()[0])
df['submarket'].unique()


column=['sale_date','city','zoning','submarket']
oe=OrdinalEncoder()
def ordinal_encode(df,column):
  df[column]=oe.fit_transform(df[column])
ordinal_encode(df,column)


df['total_val']=df['land_val']+df['imp_val']
df=df.drop(['land_val','imp_val'],axis=1)


ss=StandardScaler()
df['total_val']=ss.fit_transform(df[['total_val']])


df.head()


X=df.drop('sale_price',axis=1)
y=df['sale_price']


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# train_data = lgb.Dataset(x_train, label=y_train)
# # Train Mean Model
# params_mean = {
#     "objective": "regression",
#     "metric": "mae",
#     "verbosity": -1
# }
# model_mean = lgb.train(params_mean, train_data)

train_data = lgb.Dataset(x_train, label=y_train)
params = {
    "objective": "quantile",
    "alpha": 0.1,
    "verbosity": -1
}
lgb_lower= lgb.train(params, train_data)

params["alpha"] = 0.9
lgb_upper = lgb.train(params, train_data)


y_pred=model_mean.predict(x_test)
mean_absolute_error(y_test,y_pred)





df_test=pd.read_csv(test)
df_test.head()


df_test=drop_attributes(df_test)


df_test.isna().sum()


df_test['submarket']=df_test['submarket'].fillna(df_test['submarket'].mode()[0])


ordinal_encode(df_test,column)


df_test['total_val']=df_test['land_val']+df_test['imp_val']
df_test=df_test.drop(['land_val','imp_val'],axis=1)


pi_lower=lgb_lower.predict(df_test)


pi_higher=lgb_upper.predict(df_test)


submission = pd.DataFrame({"id":df_test['id'],"pi_lower": pi_lower, "pi_upper": pi_higher})
submission.to_csv("submission.csv", index=False)







