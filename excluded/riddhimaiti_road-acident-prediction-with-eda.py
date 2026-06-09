import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


train=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv',index_col=0)
train.head()


test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv',index_col=0)


train.info()


train.describe()


sns.distplot(train['accident_risk'])


nonnum_col=[col for col in train.columns if train[col].dtype=='object']
nonnum_col


cat_cols=nonnum_col.copy()


cat_cols.append('num_lanes')


cat_cols.append('speed_limit')


cat_cols


n_cols = 2
n_rows = len(cat_cols) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ax = axes[i]
    sns.countplot(data=train, x=col, ax=ax,palette='viridis')
    ax.set_title(f"{col.capitalize()} Distribution", fontsize = 16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize = 14)

        
# Turn off any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


sns.barplot(data=train,x='road_type',y='accident_risk',palette='rainbow')


sns.barplot(data=train,x='num_lanes',y='accident_risk',palette='coolwarm')


sns.jointplot(data=train,x='curvature',y='accident_risk',color='red')


sns.barplot(data=train,x='speed_limit',y='accident_risk',palette='rainbow')


sns.barplot(data=train,x='lighting',y='accident_risk',palette='viridis')


sns.barplot(data=train,x='weather',y='accident_risk',palette='coolwarm')


sns.barplot(data=train,x='time_of_day',y='accident_risk',palette='rainbow')


plt.figure(figsize=(12,10))
sns.heatmap(train.corr(numeric_only=True),cmap='coolwarm',annot=True,linewidths=1)


from sklearn.model_selection import train_test_split


X=train.drop('accident_risk',axis=1)
y=train['accident_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


from sklearn.preprocessing import OneHotEncoder


encoder=OneHotEncoder(sparse_output=False,drop='first')
encoded_X_train=encoder.fit_transform(X_train[nonnum_col])
encoded_X_test=encoder.transform(X_test[nonnum_col])


encoded_X_train=pd.DataFrame(encoded_X_train,columns=encoder.get_feature_names_out(),index=X_train.index)
encoded_X_test=pd.DataFrame(encoded_X_test,columns=encoder.get_feature_names_out(),index=X_test.index)


encoded_X_train=pd.concat([X_train,encoded_X_train],axis=1)
encoded_X_train.drop(nonnum_col,axis=1,inplace=True)
encoded_X_test=pd.concat([X_test,encoded_X_test],axis=1)
encoded_X_test.drop(nonnum_col,axis=1,inplace=True)


from sklearn.preprocessing import StandardScaler


scaler=StandardScaler()
scaled_X_train=scaler.fit_transform(encoded_X_train)
scaled_X_test=scaler.transform(encoded_X_test)


from sklearn.linear_model import LinearRegression


linreg=LinearRegression()
linreg.fit(scaled_X_train,y_train)
pred=linreg.predict(scaled_X_test)


from sklearn.metrics import mean_absolute_error,mean_squared_error


mean_absolute_error(y_test,pred)


mean_squared_error(y_test,pred)**0.5


from sklearn.tree import DecisionTreeRegressor


dtree=DecisionTreeRegressor()
dtree.fit(scaled_X_train,y_train)
pred=dtree.predict(scaled_X_test)


mean_absolute_error(y_test,pred)


mean_squared_error(y_test,pred)**0.5


from sklearn.ensemble import RandomForestRegressor


rfr=RandomForestRegressor()
rfr.fit(scaled_X_train,y_train)
pred=rfr.predict(scaled_X_test)


mean_absolute_error(y_test,pred)


mean_squared_error(y_test,pred)**0.5


from xgboost import XGBRegressor


xgb=XGBRegressor()
xgb.fit(scaled_X_train,y_train)
pred=xgb.predict(scaled_X_test)


mean_absolute_error(y_test,pred)


mean_squared_error(y_test,pred)**0.5


from lightgbm import LGBMRegressor


lgbm=LGBMRegressor()
lgbm.fit(scaled_X_train,y_train)
pred=lgbm.predict(scaled_X_test)


mean_absolute_error(y_test,pred)


mean_squared_error(y_test,pred)**0.5


error=np.zeros(10)
for i in range(10):
    xgb=XGBRegressor(n_estimators=50+10*i,random_state=101)
    xgb.fit(scaled_X_train,y_train)
    pred=xgb.predict(scaled_X_test)
    error[i]=mean_squared_error(y_test,pred)**0.5
plt.plot([50+10*i for i in range(10)],error,'ro-')
plt.plot(50+10*error.argmin(),error.min(),'b*')


best_n=50+10*error.argmin()
print(best_n)


xgb=XGBRegressor(n_estimators=best_n,random_state=101)
xgb.fit(scaled_X_train,y_train)
pred=xgb.predict(scaled_X_test)


mean_absolute_error(y_test,pred)


mean_squared_error(y_test,pred)**0.5


from sklearn.metrics import r2_score


r2_score(y_test,pred)


encoded_train_X=encoder.transform(X[nonnum_col])
encoded_test_X=encoder.transform(test[nonnum_col])


encoded_train_X=pd.DataFrame(encoded_train_X,columns=encoder.get_feature_names_out(),index=X.index)
encoded_test_X=pd.DataFrame(encoded_test_X,columns=encoder.get_feature_names_out(),index=test.index)


encoded_train_X=pd.concat([X,encoded_train_X],axis=1)
encoded_train_X.drop(nonnum_col,axis=1,inplace=True)
encoded_test_X=pd.concat([test,encoded_test_X],axis=1)
encoded_test_X.drop(nonnum_col,axis=1,inplace=True)


scaler=StandardScaler()
scaled_train_X=scaler.fit_transform(encoded_train_X)
scaled_test_X=scaler.transform(encoded_test_X)


xgb=XGBRegressor(n_estimators=best_n,random_state=101)
xgb.fit(scaled_train_X,y)
pred=xgb.predict(scaled_test_X)


final=pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv',index_col=0)


final['accident_risk']=pred


final.to_csv('final.csv')

