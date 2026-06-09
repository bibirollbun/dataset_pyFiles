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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style('whitegrid')

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

import warnings
warnings.filterwarnings('ignore')


train_path = "/kaggle/input/playground-series-s5e1/train.csv"
test_path = "/kaggle/input/playground-series-s5e1/test.csv"


train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


train_df= pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)


train_df.head()


test_df.head()


train_df.info()


test_df.info()


train_df.isnull().sum()


test_df.isnull().sum()


train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])


for col in train_df.columns:
    print(col,'---->',train_df[col].nunique())


plt.figure(figsize=(12,8))
sns.histplot(train_df["num_sold"], kde=True)
plt.show()


plt.figure(figsize=(12,8))
sns.boxplot(x=train_df["num_sold"])
plt.show()


train_df["country"].value_counts()


categorical_columns = ['country','store','product']

plt.figure(figsize=(15,10))
for  i,column in enumerate(categorical_columns,1):
    plt.subplot(3,1,i)
    sns.countplot(y = column, data = train_df, order = train_df[column].value_counts().index)
    plt.title(f'Distribution of {column}')
    plt.xlabel('Count')
    plt.ylabel(column)

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 5))

for i,column in enumerate(categorical_columns, 1):
    plt.subplot(1,3,i)
    sns.histplot(data = train_df, x = "num_sold", hue =column , element ='step', bins=30)
    plt.title(f'{column} Distribution')
plt.tight_layout()
plt.show()


plt.figure(figsize=(15,5))
for i,column in enumerate(categorical_columns, 1):
    plt.subplot(1,3,i)
    sns.boxplot(y = train_df["num_sold"], x =train_df[column] ,hue =train_df[column])
    plt.title(f'Distribution of {column}')

plt.tight_layout()
plt.show()


train_df.groupby('country')['num_sold'].mean()


train_df.groupby('store')['num_sold'].mean()


train_df.groupby('product')['num_sold'].mean()


train_df['num_sold'] = train_df.groupby('country')['num_sold'].transform(lambda x: x.fillna(x.mean()))


import plotnine as p9 
from plotnine import *


ggplot(train_df, aes(x='date', y='num_sold')) + geom_line()


ggplot(train_df, aes(x='date', y='num_sold',group='country' ,color='country'))+geom_point()+ theme_minimal()


ggplot(train_df, aes(x='date', y='num_sold',group='store' ,color='store'))+geom_point()+ theme_minimal()


ggplot(train_df, aes(x='date', y='num_sold',group='product' ,color='product'))+geom_point()+ theme_minimal()


train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['day_of_week'] = train_df['date'].dt.dayofweek


test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['day_of_week'] = test_df['date'].dt.dayofweek


ggplot(train_df, aes(x='month', y='num_sold',group='year' ,color='year'))+geom_point()+scale_x_continuous(breaks=range(1, 13))+ theme_minimal() 


train_df.head()


train_df["holiday"] = 0
test_df["holiday"] = 0


train_df["country"].unique()
test_df["country"].unique()


import holidays

ca_holidays = holidays.country_holidays('CA') # Canada
fi_holidays = holidays.country_holidays('FI') # Finland
it_holidays = holidays.country_holidays('IT') # Italy
ke_holidays = holidays.country_holidays('KE') # Kenya
no_holidays = holidays.country_holidays('NO') # Norway
sg_holidays = holidays.country_holidays('SG') # Singapore


def set_holiday(row):
    VAL_HOLIDAY = 1
    if row["country"] == "Canada" and row["date"] in ca_holidays:
        row["holiday"] = VAL_HOLIDAY

    elif row["country"] == "Finland" and row["date"] in fi_holidays:
        row["holiday"] = VAL_HOLIDAY

    elif row["country"] == "Italy" and row["date"] in it_holidays:
        row["holiday"] = VAL_HOLIDAY

    elif row["country"] == "Kenya" and row["date"] in ke_holidays:
        row["holiday"] = VAL_HOLIDAY
    
    elif row["country"] == "Norway" and row["date"] in no_holidays:
        row["holiday"] = VAL_HOLIDAY

    elif row["country"] == "Singapore" and row["date"] in sg_holidays:
        row["holiday"] = VAL_HOLIDAY

    return row


df_train = train_df.apply(set_holiday, axis=1)
df_test = test_df.apply(set_holiday, axis=1)


df_train


df_train_encoded = pd.get_dummies(df_train, columns=['country','store','product'])
df_test_encoded = pd.get_dummies(df_test, columns=['country','store','product'])


def periodic_transform(dff,variable):
    dff[f"{variable}_SIN"] = np.sin(dff[variable] / dff[variable].max()*2*np.pi)
    dff[f"{variable}_COS"] = np.cos(dff[variable] / dff[variable].max()*2*np.pi)
    return dff


cyclic_col = ['month','day','day_of_week']

for col in cyclic_col:
    df_train_final = periodic_transform(df_train_encoded, col)
    df_test_final = periodic_transform(df_test_encoded, col)


df_train_final.columns


df_test_final.columns


df_train_final = df_train_final.drop(['month', 'day', 'day_of_week', 'date', 'id'], axis = 1)
df_test_final = df_test_final.drop(['month', 'day', 'day_of_week', 'date', 'id'], axis = 1)


df_train_final.columns


numeric_df = df_train_final.select_dtypes(include = ['number'])
corr_matrix = numeric_df.corr()


print(corr_matrix['num_sold'].sort_values(ascending = False).to_string())


plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix,annot=True,cmap = 'coolwarm', fmt = ".2f")
plt.show()


x = df_train_final.drop(['num_sold'],axis =1)
y = df_train_final['num_sold']


from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state=42)


from sklearn.preprocessing import MinMaxScaler


mm = MinMaxScaler()
x_train_scaled = mm.fit_transform(x_train)
x_test_scaled = mm.transform(x_test)


df_test_scaled_final = mm.transform(df_test_final)


def model_acc(model):
    model.fit(x_train_scaled,y_train)
    acc = model.score(x_test_scaled,y_test)
    print(str(model)+'-->'+str(acc))


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
model_acc(dt)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
model_acc(rf)


y_test_pred = dt.predict(x_test_scaled)
y_test = y_test.values.flatten()
y_test_pred = y_test_pred.flatten()
final_df1 = pd.DataFrame(np.hstack((y_test_pred[:, np.newaxis], y_test[:, np.newaxis])), columns=['Prediction', 'Real'])


from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error


acc_train_dt = dt.score(x_train_scaled,y_train)
print("Model Score on Train set :",acc_train_dt)
print("Model Score on Test set :",dt.score(x_test_scaled,y_test))


print(f'MAE: {mean_absolute_error(final_df1["Prediction"],final_df1["Real"])}')
print(f'MSE: {mean_squared_error(final_df1["Prediction"],final_df1["Real"])}')
print(f'RMSE: {np.sqrt(mean_squared_error(final_df1["Prediction"],final_df1["Real"]))}')
print(f'MAPE: {mean_absolute_percentage_error(y_test, y_test_pred)}')


fig, ax = plt.subplots(figsize=(20, 5))
sns.lineplot(x=range(len(final_df1['Real'])) ,y=final_df1['Real'],color='black',label='Real')
sns.lineplot(x=range(len(final_df1['Prediction'])),y=final_df1['Prediction'],color='red',label='Prediction')
ax.set_xlim([3000,3100])
plt.title('Real vs. Predictions for Decision Tree')
plt.show()


y_test_pred = rf.predict(x_test_scaled)
y_test_pred = y_test_pred.flatten()
final_df2 = pd.DataFrame(np.hstack((y_test_pred[:, np.newaxis], y_test[:, np.newaxis])), columns=['Prediction', 'Real'])


acc_train = rf.score(x_train_scaled,y_train)
print(acc_train)


print(f'MAE: {mean_absolute_error(final_df2["Prediction"],final_df2["Real"])}')
print(f'MSE: {mean_squared_error(final_df2["Prediction"],final_df2["Real"])}')
print(f'RMSE: {np.sqrt(mean_squared_error(final_df1["Prediction"],final_df1["Real"]))}')
print(f'MAPE: {mean_absolute_percentage_error(y_test, y_test_pred)}')


fig, ax = plt.subplots(figsize=(20, 5))
sns.lineplot(x=range(len(final_df2['Real'])) ,y=final_df2['Real'],color='black',label='Real')
sns.lineplot(x=range(len(final_df2['Prediction'])),y=final_df2['Prediction'],color='red',label='Prediction')
ax.set_xlim([3000,3100])
plt.title('Real vs. Predictions Random Forest')
plt.show()


import xgboost as xgb


train_data = xgb.DMatrix(x_train_scaled, label=y_train)
test_data = xgb.DMatrix(x_test_scaled, label=y_test)


params = {
    'objective': 'reg:squarederror',  # For regression tasks
    'learning_rate': 0.1,  # Step size shrinkage
    'max_depth': 5,  # Maximum depth of a tree
    'alpha': 10,  # L1 regularization term on weights
    'n_estimators': 100  # Number of boosting rounds (trees)
}


model_xgb = xgb.train(params, train_data, num_boost_round=100)


y_pred = model_xgb.predict(test_data)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f'MAPE: {mean_absolute_percentage_error(y_test, y_pred)}')


y_test_pred = y_pred.flatten()
final_df2 = pd.DataFrame(np.hstack((y_test_pred[:, np.newaxis], y_test[:, np.newaxis])), columns=['Prediction', 'Real'])


fig, ax = plt.subplots(figsize=(20, 5))
sns.lineplot(x=range(len(final_df2['Real'])) ,y=final_df2['Real'],color='black',label='Real')
sns.lineplot(x=range(len(final_df2['Prediction'])),y=final_df2['Prediction'],color='red',label='Prediction')
ax.set_xlim([3000,3100])
plt.title('Real vs. Predictions XGBoost')
plt.show()


y_test_pred_rf = rf.predict(df_test_scaled_final)


submission_df = pd.DataFrame({
    'id': df_test_encoded['id'],  # Extract 'id' column from the test DataFrame
    'Premium Amount': y_test_pred_rf  # Use the predictions from your model
})


submission_df

