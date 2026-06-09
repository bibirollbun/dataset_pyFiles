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


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


sample_submission


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')


train


test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


test


train.head()


train.shape


## list of columns in train df
train.columns


## check is there any null values -->
train.isnull().sum()


train.describe()


train.info()


## filling the null values in 'num_sold' feature using mean value imputation -->

# train['num_sold'] = train['num_sold'].fillna(train['num_sold'].mean()) ## replacing all null values with the mean of 'num_sold' feature..


## trying by droping all the rows which contain null values -->
train.dropna(inplace=True)


train['num_sold'].isnull().sum()


train.isnull().sum()


train['date'].dtype


## here the dtype of 'date' feature is object so I need to change it
train['date'] = pd.to_datetime(train['date'])


train['date'].dtype


train.info()


train['year'] = train['date'].dt.year ## extracting year from date

train['month_num'] = train['date'].dt.month ## extracting month from date

train['date_day'] = train['date'].dt.day ## extracting day from date


train.sample(3)


## day of week -->
train['date_dow'] = train['date'].dt.dayofweek


train.sample(4)


train['is_weekend'] = np.where(train['date_dow'].isin([5,6]), 1,0)  
## if day is either 'Sunday' which is 6 or 'Saturday' which is 5, it is weekend else not weekend...


train


## drop 'id' feature because there is no use it
train.drop('id', axis=1 ,inplace=True)


train.columns


## Now just drop initial 'date' feature
train.drop('date', axis=1 ,inplace=True)


train['country'].value_counts()


import matplotlib.pyplot as plt


country_names = train.country.value_counts().index
country_count = train.country.value_counts().values


plt.pie(country_count,labels=country_names,autopct='%1.2f%%')
plt.show()


train.columns


store_names = train.store.value_counts().index
store_count = train.store.value_counts().values


plt.pie(store_count,labels=store_names,autopct='%1.2f%%')
plt.show()


product_names = train['product'].value_counts().index
product_count = train['product'].value_counts().values


plt.pie(product_count,labels=product_names,autopct='%1.2f%%')
plt.show()


train['store'].value_counts()


train['product'].value_counts()


train['num_sold']


print("Maximum values in num_sold: ",max(train['num_sold']))
print("Minimum values in num_sold: ",min(train['num_sold']))
print("Average values in num_sold: ",train['num_sold'].mean())


import seaborn as sns


plt.figure(figsize=(16,6))
sns.distplot(train['num_sold'])
plt.show()


train[train['num_sold']==5]


train.shape


train[train['num_sold']>2000]


train[train['num_sold']>2000].shape


train[train['num_sold']>3000].shape


train[train['num_sold']>2500].shape


train.shape


train = train[train['num_sold']<2500]


train.shape


train[['num_sold']]


train['num_sold'].skew()


## so it is right skewed 


train['num_sold'].describe()


sns.boxplot(train[['num_sold']])


## finding the IQR -->
percentile25 = train['num_sold'].quantile(0.25) ## Q1
percentile75 = train['num_sold'].quantile(0.75) ## Q3

iqr = percentile75 - percentile25


print('Q3: ',percentile75)
print('Q1: ',percentile25)
print('IQR: ',iqr)


upper_limit = percentile75 + 1.5*iqr
lower_limit = percentile25 - 1.5*iqr


print('Lower Limit:',lower_limit)
print('Upper Limit:',upper_limit)


train[train['num_sold']>upper_limit] ## values where it is greater than upper limit


train.shape


train[train['num_sold']<lower_limit] ## values where it is less than lower limit


## there is no any values lower than lower_limit


## Removing all the rows where the values is less than lower limit
train = train[train['num_sold']<upper_limit]


train.shape


plt.figure(figsize=(16,6))
sns.distplot(train['num_sold'])
plt.show()


train[train['num_sold']<500].shape


train_greater_200 = train[train['num_sold']>200]


train_greater_200.shape


plt.figure(figsize=(16,6))
sns.distplot(train_greater_200['num_sold'])
plt.show()


train.dtypes


cat_col_list = [feature for feature in train.columns if train[feature].dtype=='O']


cat_col_list


train_encoded = pd.get_dummies(train, columns=['country', 'store', 'product'], dtype=int)


train_encoded


# train_greater_200_encoded = pd.get_dummies(train, columns=['country', 'store', 'product'], dtype=int)


## Independent and Target feature --.
X = train_encoded.drop(columns=['num_sold'])
y = train_encoded['num_sold']


X


y


y = y.astype('int')


y


# training model on complete data -->

## train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


## Applying the OneHotEncoding on these categorical columns -->
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train_scaled


pd.DataFrame(X_train_scaled)


X_train


## Model Training and Model Selection
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

## ensembles 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor


## Creating a function to evaluat model
def evaluate_model(true, predicted):
    mae=mean_absolute_error(true,predicted)
    mse=mean_squared_error(true,predicted)
    rmse=np.sqrt(mse)
    r2=r2_score(true,predicted)
    print("R2 Score:{:.4f}".format(r2))
    print("MSE:{:.4f}".format(mse))
    print("RMSE:{:.4f}".format(rmse))
    print("MAE:{:.4f}".format(mae))


## Model training

models={
    "Linear Regression":LinearRegression(),
    "Lasso":Lasso(),
    "Ridge":Ridge()
}

for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(X_train_scaled,y_train) ## Train Model

    ## Make Predictions
    y_train_pred=model.predict(X_train_scaled)
    y_test_pred=model.predict(X_test_scaled)

    print(list(models.keys())[i],"=============>")
    print("Evaluating Train Dataset")
    evaluate_model(y_train,y_train_pred)
    print(f"\n{'-'*50}\n")
    print("Evaluating Test Dataset")
    evaluate_model(y_test,y_test_pred)
    print("="*50)
    print("\n")


## using X_train not X_train_scaled to train models because here using ensemble techniques...


dt = RandomForestRegressor()

dt.fit(X_train,y_train)

## Make Predictions
y_train_pred=dt.predict(X_train)
y_test_pred=dt.predict(X_test)

print("Random Forest Regressor","=============>")
print("Evaluating Train Dataset")
evaluate_model(y_train,y_train_pred)
print(f"\n{'-'*50}\n")
print("Evaluating Test Dataset")
evaluate_model(y_test,y_test_pred)
print("="*50)
print("\n")


rf = RandomForestRegressor()

rf.fit(X_train,y_train)

## Make Predictions
y_train_pred=rf.predict(X_train)
y_test_pred=rf.predict(X_test)

print("Random Forest Regressor","=============>")
print("Evaluating Train Dataset")
evaluate_model(y_train,y_train_pred)
print(f"\n{'-'*50}\n")
print("Evaluating Test Dataset")
evaluate_model(y_test,y_test_pred)
print("="*50)
print("\n")


rf


lr = LinearRegression()

lr.fit(X_train_scaled,y_train)

## Make Predictions
y_train_pred=lr.predict(X_train_scaled)
y_test_pred=lr.predict(X_test_scaled)

print("Random Forest Regressor","=============>")
print("Evaluating Train Dataset")
evaluate_model(y_train,y_train_pred)
print(f"\n{'-'*50}\n")
print("Evaluating Test Dataset")
evaluate_model(y_test,y_test_pred)
print("="*50)
print("\n")


test


## check for null values
test.isnull().sum()


## So there is no any null value 


test.info()


## here the dtype of 'date' feature is object so I need to change it
test['date'] = pd.to_datetime(test['date'])


test['year'] = test['date'].dt.year ## extracting year from date

test['month_num'] = test['date'].dt.month ## extracting month from date

test['date_day'] = test['date'].dt.day ## extracting day from date


## day of week -->
test['date_dow'] = test['date'].dt.dayofweek


test


test['is_weekend'] = np.where(test['date_dow'].isin([5,6]), 1,0)  


## dropping the 'id' feature and initial 'date' feature -->
test.drop(columns = ['id','date'],inplace=True)


test.columns


test_encoded = pd.get_dummies(test, columns=['country', 'store', 'product'], dtype=int)


test_encoded


# applying Standard Scaling for linear regression
test_encoded_scaled = scaler.transform(test_encoded)


test_encoded


pd.DataFrame(test_encoded)


pd.DataFrame(test_encoded_scaled)


rf_predictions = rf.predict(test_encoded)


rf_predictions


sample_submission


id_column = sample_submission['id']


id_column


rf_result = pd.DataFrame(
    {
        'id':id_column,
        'num_sold':rf_predictions
    }
)


rf_result


rf_result['num_sold'] = rf_result['num_sold'].astype('int')


# rf_result['num_sold'] = rf_result['num_sold'].apply(lambda x: np.round(x,2))


rf_result


rf_result.to_csv('rf_prediction.csv',index=False)
print("File saved as rf_prediction.csv")


dt_predictions = dt.predict(test_encoded)


dt_predictions


dt_result = pd.DataFrame(
    {
        'id':id_column,
        'num_sold':dt_predictions
    }
)


dt_result['num_sold'] = dt_result['num_sold'].astype('int')


dt_result


dt_result.to_csv('dt_prediction.csv',index=False)
print("File saved as dt_prediction.csv")


lr


lr_predictions = lr.predict(test_encoded_scaled)


lr_predictions


lr_result = pd.DataFrame(
    {
        'id':id_column,
        'num_sold':lr_predictions
    }
)


lr_result['num_sold'] = lr_result['num_sold'].astype('int')


lr_result


lr_result.to_csv('lr_prediction.csv',index=False)
print("File saved as lr_prediction.csv")




