import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")


df.head()


print(df.info())


df.isna().sum()


df.isnull().sum()


9705/30000


df['Brand'] = df['Brand'].fillna('Unknown')



df.isnull().sum()


avg_price = df.groupby('Color')['Price'].mean().reset_index()

# Bar Chart (Average Price per Color)
plt.figure(figsize=(8,5))
sns.barplot(x='Color', y='Price', data=avg_price, palette='viridis')

plt.title('Backpack Color vs Average Price')
plt.xlabel('Backpack Color')
plt.ylabel('Average Price ($)')
plt.show()


df = df.drop('Color', axis=1)



df


from sklearn.preprocessing import LabelEncoder


le=LabelEncoder()


from sklearn.preprocessing import OneHotEncoder


encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False).set_output(transform='pandas')


encoded_brands = encoder.fit_transform(df[['Brand']])


encoded_brands


df=pd.concat([df,encoded_brands],axis=1).drop(columns=['Brand'])


df


df.drop(columns=['id'])


df['Material'].value_counts()



le=LabelEncoder()


for col in ['Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style']:
    df[col] = le.fit_transform(df[col])



df


df.describe()


df=df.drop('id',axis=1)


df


df.isna().sum()


df['Weight Capacity (kg)']= df['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].mean())



df


corr_matrix=df.corr()


plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Display the heatmap
plt.title("Correlation Heatmap")
plt.show()




#Inference -------------> not muchh corelation.


from sklearn.model_selection import train_test_split


X=pd.concat([df.iloc[:,0:7],df.iloc[:,8:14]],axis=1)
X


y=df['Price']



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestRegressor


model_1=RandomForestRegressor(n_estimators=20,random_state=42,oob_score=True)


model_1.fit(X_train,y_train)





y_pred = model_1.predict(X_test)


y_pred


from sklearn.metrics import mean_squared_error as MSE 

rmse = np.sqrt(MSE(y_test, y_pred)) 
print("RMSE : % f" %(rmse)) 


model_2=RandomForestRegressor(n_estimators=200,random_state=42,oob_score=True)


model_2.fit(X_train,y_train)


from sklearn.metrics import mean_squared_error as MSE 

rmse = np.sqrt(MSE(y_test, y_pred)) 
print("RMSE : % f" %(rmse)) 


!pip install xgboost




import xgboost as xg 

reg = xg.XGBRegressor(
    eval_metric='rmsle',
)


from sklearn.model_selection import GridSearchCV

param_grid={"max_depth":[7,8,9],
            "n_estimators":[500,600,900],
            "learning_rate":[0.01,0.015]}

search = GridSearchCV(reg, param_grid, cv=5).fit(X_train, y_train)

print("The best hyperparameters are ",search.best_params_)


regressor=xg.XGBRegressor(learning_rate = search.best_params_["learning_rate"],
                           n_estimators  = search.best_params_["n_estimators"],
                           max_depth     = search.best_params_["max_depth"],
                           eval_metric='rmsle')

regressor.fit(X_train, y_train)


predictions = regressor.predict(X_test)
print(predictions)


from sklearn.metrics import mean_squared_log_error
RMSLE = np.sqrt( mean_squared_log_error(y_test, y_pred) )
print("The score is %.5f" % RMSLE )


from sklearn.metrics import mean_squared_error as MSE 

rmse = np.sqrt(MSE(y_test, y_pred)) 
print("RMSE : % f" %(rmse))


df_test=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
df_test


df_test=df_test.drop('Color',axis=1)



df_test


df_test['Brand'] = df_test['Brand'].fillna('Unknown')



encoded_brands = encoder.fit_transform(df_test[['Brand']])


df_test=pd.concat([df_test,encoded_brands],axis=1).drop(columns=['Brand'])
for col in ['Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style']:
    df_test[col] = le.fit_transform(df_test[col])
df_test=df_test.drop('id',axis=1)
df_test['Weight Capacity (kg)']= df_test['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].mean())


df_test


test_prediction=regressor.predict(df_test)


test_prediction


len(test_prediction)


df_pred = pd.DataFrame(test_prediction, columns=['Price'])



df_pred


df_pred['id'] = range(300000, 300000 + len(df_pred))


df_pred


df_pred=df_pred[['id','Price']]





df_pred


df_pred.to_csv("submission.csv",index=False)

