import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor


train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
train_df.head()


# impute_col = train_df['num_sold']


# imputer = SimpleImputer(strategy='mean')
# train_df['num_sold'] = imputer.fit_transform(train_df['num_sold'].values.reshape(-1,1))


train_df.isna().sum()


train_df.info()


train_df.dropna(axis=0,subset=["num_sold"],inplace=True)


train_df.head()


train_df.info()


train_df['date'] = pd.to_datetime(train_df['date'])


validation_df = train_df[train_df['date'].dt.year >= 2016]
train_df = train_df[train_df['date'].dt.year < 2016]


train_df.head()


validation_df.head()


columns_of_interest = train_df.iloc[:,2:].columns.tolist()
columns_of_interest


train_df = train_df[columns_of_interest]
validation_df = validation_df[columns_of_interest]


validation_df.head()


train_df['product'].unique()


px.histogram(train_df,x='num_sold',y='product',color="product")


train_df['store'].unique()


px.histogram(train_df,x='num_sold',y='store',color="store")


px.histogram(train_df,x='num_sold',y='country',color="country")



categorical_columns = ['country','store','product']


le = LabelEncoder()


for col in categorical_columns:
    train_df[col] = le.fit_transform(train_df[col])


train_df.head()


sns.heatmap(train_df.corr(),annot=True,cmap='Reds')


scaler = MinMaxScaler()
# scaler = StandardScaler()



train_df[categorical_columns] = scaler.fit_transform(train_df[categorical_columns])


train_df.head(10)


X_train = train_df.drop('num_sold',axis=1)
y_train = train_df['num_sold']


# model = LinearRegression()
model = RandomForestRegressor(criterion="absolute_error",n_estimators=200)
model.fit(X_train,y_train)


pred_sales = model.predict(X_train)


import math


mse = mean_squared_error(pred_sales,y_train)
print(f'MSE: ',mse)
print(f'RMSE: ',math.sqrt(mse))


pred_sales


mape = mean_absolute_percentage_error(pred_sales,y_train)
print(f'MAPE: ',mape)


def validate_data(df):
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])

    df[categorical_columns] = scaler.fit_transform(df[categorical_columns])
    X_val = df.drop('num_sold',axis=1)
    y_val = df['num_sold']
    pred_sales = model.predict(X_val)
    mape = mean_absolute_percentage_error(np.round(pred_sales),y_val)
    
    return f'Validation Mean Absolute Percentage Error: ',mape
    
    


validate_data(validation_df)


def make_predictions_for_test(test_df):
    for col in categorical_columns:
        test_df[col] = le.fit_transform(test_df[col])

    test_df[categorical_columns] = scaler.fit_transform(test_df[categorical_columns])
    X_test = test_df[categorical_columns]
    pred_sales = model.predict(X_test)

    return np.round(pred_sales,0)
    
    


test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
test_df.head()


test_predictions = make_predictions_for_test(test_df)
print(test_predictions)


final_compiled = pd.DataFrame({
    "id":test_df['id'],
    "num_sold":test_predictions
}) 
final_compiled.head(20)



final_compiled.to_csv("submission.csv",index=False)




