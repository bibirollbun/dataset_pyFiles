dataset=r"prediction-interval-competition-ii-house-price\dataset.csv"


test=r"prediction-interval-competition-ii-house-price\test.csv"


import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


test=pd.read_csv(test)
dataset=pd.read_csv(dataset)


dataset.head()



(dataset.isnull().sum()).sort_values(ascending=False)


df=dataset.drop(['id','sale_nbr','sale_warning','join_status','join_year','subdivision'],axis=1)


(df["submarket"].isnull().sum()/df["submarket"].shape[0])*100


df["submarket"]=df["submarket"].fillna(df["submarket"].mode()[0])


df.info()


columns=[]
for i in df.select_dtypes("object").columns:
    columns.append(i)



columns


from sklearn.preprocessing import OrdinalEncoder
Or=OrdinalEncoder()



df[columns]=Or.fit_transform(df[columns])


df.describe()


df['total_val']=df['land_val']+df['imp_val']
df=df.drop(['land_val','imp_val'],axis=1)


from sklearn.preprocessing import StandardScaler


ss=StandardScaler()
df['total_val']=ss.fit_transform(df[['total_val']])


X=df.drop('sale_price',axis=1)
y=df['sale_price']


plt.hist(df['sale_price'],alpha=0.8)
plt.show()


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


from sklearn.ensemble import GradientBoostingRegressor

# Lower Quantile Model
model_lower = GradientBoostingRegressor(loss='quantile', alpha=0.1)
model_lower.fit(x_train, y_train)

# Upper Quantile Model
model_upper = GradientBoostingRegressor(loss='quantile', alpha=0.9)
model_upper.fit(x_train, y_train)

# Predictions
y_pred_lower = model_lower.predict(x_test)
y_pred_upper = model_upper.predict(x_test)




mean_absolute_error(y_test,y_pred_lower)


mean_absolute_error(y_test,y_pred_upper)



test.head()


df_test=test.drop(['id','sale_nbr','sale_warning','join_status','join_year','subdivision'],axis=1)


df_test['submarket']=df_test['submarket'].fillna(df_test['submarket'].mode()[0])


from sklearn.preprocessing import OrdinalEncoder
Or=OrdinalEncoder()
df_test[columns]=Or.fit_transform(df_test[columns])


df_test['total_val']=df_test['land_val']+df_test['imp_val']
df_test=df_test.drop(['land_val','imp_val'],axis=1)


pi_lower= model_lower.predict(df_test)



pi_upper=model_upper.predict(df_test)



submission = pd.DataFrame({"id":test['id'],"pi_lower": pi_lower, "pi_upper": pi_upper})
submission.to_csv("submission.csv", index=False)





