import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',None)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import normalize, scale


df=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')


df.head()


df.shape


df.isnull().sum()


df.info()


df.describe().T


df['Brand'].value_counts()


df['Material'].value_counts()


df['Size'].value_counts()


df['Laptop Compartment'].value_counts()


df['Style'].value_counts()


df['Color'].value_counts()


plt.hist(df['Price'], bins=20, color='skyblue', edgecolor='black')
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency');


plt.figure(figsize=(10, 6))
sns.boxplot(x='Brand', y='Price', data=df)
plt.title('Price Distribution by Brand')
plt.xticks(rotation=90);


material_price = df.groupby('Material')['Price'].mean().sort_values()
material_price.plot(kind='bar', cmap='viridis')
plt.title('Average Price by Material')
plt.xlabel('Material')
plt.ylabel('Average Price');


plt.figure(figsize=(10, 6))
sns.boxplot(x='Compartments', y='Price', data=df)
plt.title('Price by Number of Compartments');


laptop_compartment_price = df.groupby('Laptop Compartment')['Price'].mean()
laptop_compartment_price.plot(kind='bar', color='lightcoral')
plt.title('Average Price by Laptop Compartment')
plt.xlabel('Laptop Compartment')
plt.ylabel('Average Price');


waterproof_price = df.groupby('Waterproof')['Price'].mean()
waterproof_price.plot(kind='bar', color='lightblue')
plt.title('Average Price by Waterproof Feature')
plt.xlabel('Waterproof')
plt.ylabel('Average Price');


plt.figure(figsize=(10, 6))
sns.boxplot(x='Style', y='Price', data=df)
plt.title('Price by Style');


sns.boxplot(x='Color', y='Price', data=df)
plt.title('Price Distribution by Color')
plt.xticks(rotation=45);


df.dropna(inplace=True)


from sklearn.impute import SimpleImputer

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

numerical_imputer = SimpleImputer(strategy='mean')


categorical_imputer = SimpleImputer(strategy='most_frequent')

df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])


def perform_future_engineering(df):
    df['Brand_Material']=df['Brand']+'_'+df['Material']
    df['Brand_Size']=df['Brand']+'_'+df['Size']
    df['Has_Laptop_Compartment']=df['Laptop Compartment'].map({'Yes':1,'No':0})
    df['Is_Waterproof']=df['Waterproof'].map({'Yes':1,'No':0})
    df['Compartments_Category']=pd.cut(df['Compartments'], 
                                         bins=[0, 2, 5, 10, np.inf], 
                                         labels=['Few', 'Moderate', 'Many', 'Very Many'])
    df['Weight_Capacity_Ratio']=df['Weight Capacity (kg)']/df['Weight Capacity (kg)'].max()
    df['Weight_to_Compartments']=df['Weight Capacity (kg)']/(df['Compartments']+1)
    df['Style_Size']=df['Style']+'_'+df['Size']
    return df


df=perform_future_engineering(df)


abs(df.corr(numeric_only=True)['Price'].sort_values(ascending=False))


df.columns


x=df[['Brand', 'Material', 'Size', 'Laptop Compartment','Waterproof', 
      'Style', 'Color','Brand_Material', 'Brand_Size', 'Has_Laptop_Compartment',
      'Is_Waterproof', 'Compartments_Category',
      'Style_Size','Weight Capacity (kg)','Weight_Capacity_Ratio','Weight_to_Compartments']]
y=df[['Price']]


x=pd.get_dummies(x,drop_first=True)


def regression_algo(x,y,plot=False):
    L=LinearRegression()
    R=Ridge()
    Lass=Lasso()
    E=ElasticNet()
    ETR=ExtraTreeRegressor()
    GBR=GradientBoostingRegressor()
    kn=KNeighborsRegressor()
    dt=DecisionTreeRegressor()
    xgb=XGBRegressor()
    rf=RandomForestRegressor()

    algos=[L,R,Lass,E,ETR,GBR,kn,dt,xgb,rf]
    algo_names=['Linear','Ridge','Lasso','ElasticNet','Extra Tree','Gradient Bossting','KNeighborRegressor','DecisionTreeRegressor','XGBReggressor','Random Forest Classifier']

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)

    r_squared=[]
    rmse=[]
    mae=[]

    result=pd.DataFrame(columns=['R_Squared','RMSE','MAE'],index=algo_names)

    for algo in algos:
        p=algo.fit(x_train,y_train).predict(x_test)
        r_squared.append(r2_score(y_test,p))
        rmse.append(mean_squared_error(y_test,p)**0.5)
        mae.append(mean_absolute_error(y_test,p))

    result.R_Squared=r_squared
    result.RMSE=rmse
    result.MAE=mae

    r_table=result.sort_values('R_Squared',ascending=False)
    if plot:
        best_model = algos[r_squared.index(max(r_squared))]
        y_pred = best_model.predict(x_test)
        
        plt.figure(figsize=(10,6))
        plt.plot(y_test.reset_index(drop=True),label='Acutal',color='green',linestyle='--')
        plt.plot(pd.Series(y_pred),label='Predicted',color='red',linestyle='--')
        plt.xlabel('Predicted Price')
        plt.ylabel('Actual Price')
        plt.title('Actual vs Predicted Price for Best Model')
        plt.legend()
        plt.show()
    return r_table


regression_algo(x,y,plot=False)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
R=Ridge()
model=R.fit(x_train,y_train)


import joblib
joblib.dump(model, 'best_regression_model.pkl')


df_test=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


df_test.shape


df_test.isnull().sum()


numerical_columns_test = df_test.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns_test = df_test.select_dtypes(include=['object']).columns.tolist()

numerical_columns_test = [col for col in numerical_columns_test if col not in ['id', 'Price']]
categorical_columns_test = [col for col in categorical_columns_test if col not in ['id', 'Price']]
numerical_imputer.fit(df[numerical_columns_test])

categorical_imputer.fit(df[categorical_columns_test])  

df_test[numerical_columns_test] = numerical_imputer.transform(df_test[numerical_columns_test])

df_test[categorical_columns_test] = categorical_imputer.transform(df_test[categorical_columns_test])


df_test=perform_future_engineering(df_test)


submission=pd.DataFrame({
    'id':df_test['id']
})


df_test=df_test[['Brand', 'Material', 'Size', 'Laptop Compartment','Waterproof', 
      'Style', 'Color','Brand_Material', 'Brand_Size', 'Has_Laptop_Compartment',
      'Is_Waterproof', 'Compartments_Category',
      'Style_Size','Weight Capacity (kg)','Weight_Capacity_Ratio','Weight_to_Compartments']]


df_test=pd.get_dummies(df_test,drop_first=True)


predictions=model.predict(df_test)


submission['Price']=predictions


submission.to_csv('submission.csv',index=False)


coefficients = np.abs(R.coef_).ravel()  # Use .ravel() to flatten the coefficients

feature_importance = pd.DataFrame({'Feature': x_train.columns, 'Importance': coefficients})

feature_importance = feature_importance.sort_values(by='Importance', ascending=False).head(30)

plt.figure(figsize=(6,15))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.gca().invert_yaxis()
plt.title('Top 30 Features by Importance (Ridge Model)')
plt.show()


x=df[['Brand', 'Material', 'Size', 'Laptop Compartment','Waterproof', 
      'Style', 'Color','Brand_Material', 'Brand_Size', 'Has_Laptop_Compartment',
      'Is_Waterproof', 'Compartments_Category',
      'Style_Size','Weight Capacity (kg)','Weight_Capacity_Ratio','Weight_to_Compartments']]
y=df[['Price']]


x=pd.get_dummies(x,drop_first=True)


model = Sequential()
model.add(Dense(120, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss='mse', optimizer='adam')


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, verbose=1)


predictions=model.predict(x_test)
r2_score(y_test,predictions)


mean_squared_error(y_test,predictions)**0.5


loss_f=pd.DataFrame(history.history)
loss_f.plot()


predictions=model.predict(df_test)


submission['Price']=predictions


submission.to_csv('submission.csv',index=False)

