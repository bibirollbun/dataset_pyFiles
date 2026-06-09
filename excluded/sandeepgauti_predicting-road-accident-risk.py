import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

import warnings 
warnings.filterwarnings('ignore')


Data=pd.read_csv('data.csv')


Data.shape


Data.head(5)


Data.tail(5)


Data.columns


Data.info()


Data.describe()


Data[Data['num_reported_accidents']==7]


Data[Data['speed_limit']==70]


# checking null values

Data.isnull().sum()


# checking for duplicates

Data.duplicated().sum()


# Checking for unique values

Data.nunique()


plt.figure(figsize=(15,8))

sns.pairplot(Data)


cat_columns=[feature for feature in Data.columns if Data[feature].dtype=='object']
num_columns=[feature for feature in Data.columns if Data[feature].dtype!='object']


cat_columns


num_columns


Data['road_type'].value_counts()


Data['lighting'].value_counts()


Data['weather'].value_counts()


Data['time_of_day'].value_counts()


for i in cat_columns:
    ax=sns.countplot(Data, x=i)
    for container in ax.containers:
        ax.bar_label(container)

    plt.xlabel(i)
    plt.show()


Data[cat_columns].head(5)


Data[num_columns].head(5)


plt.pie(Data['road_type'].value_counts(), labels=['highway','rural','urban'], colors=['violet', 'blue', 'green'], autopct='%1.1f', explode=[0.2, 0, 0], shadow=True)


plt.pie(Data['lighting'].value_counts(), labels=['dim','daylight', 'night'], colors=['yellow', 'orange', 'pink'], explode=[0.2, 0, 0], autopct='%1.1f', shadow=True)


plt.pie(Data['weather'].value_counts(), labels=['foggy','clear','rain'], colors=['blue', 'gray', 'green'], explode=[0.2, 0, 0], autopct='%1.1f', shadow=True)


plt.pie(Data['time_of_day'].value_counts(), labels=['morning','evening', 'afternoon'], colors=['yellow', 'gray', 'orange'], explode=[0.2, 0, 0], autopct='%1.1f', shadow=True)


for i in num_columns:
    sns.distplot(x=Data[i])
    plt.xlabel(i)
    plt.show()


for i in num_columns:
    sns.histplot(x=Data[i], kde=True)

    plt.xlabel(i)
    plt.show()


for i in num_columns:
    sns.boxplot(x=Data[i])

    plt.xlabel(i)
    plt.show()


for i in num_columns:
    sns.histplot(x=Data[i], kde=True, hue=Data['road_type'])
    plt.xlabel(i)

    plt.show()
                 


for i in num_columns:
    sns.histplot(x=Data[i], kde=True, hue=Data['lighting'])
    plt.xlabel(i)

    plt.show()


for i in num_columns:
    sns.histplot(x=Data[i], kde=True, hue=Data['weather'])
    plt.xlabel(i)

    plt.show()


for i in num_columns:
    sns.histplot(x=Data[i], kde=True, hue=Data['time_of_day'])
    plt.xlabel(i)

    plt.show()


for i in cat_columns:
    ax=sns.countplot(x=Data[i], hue=Data['road_type'])
    for container in ax.containers:
        ax.bar_label(container)

    plt.xlabel(i)
    plt.show()


for i in cat_columns:
    ax=sns.countplot(x=Data[i], hue=Data['weather'])

    for container in ax.containers:
        ax.bar_label(container)

    plt.xlabel(i)
    plt.show()


for i in cat_columns:
    ax=sns.countplot(x=Data[i], hue=Data['lighting'])
    for container in ax.containers:
        ax.bar_label(container)

    plt.xlabel(i)
    plt.show()


for i in cat_columns:
    ax=sns.countplot(x=Data[i], hue=Data['time_of_day'])
    for container in ax.containers:
        ax.bar_label(container)

    plt.xlabel(i)
    plt.show()


plt.figure(figsize=(18,9))

sns.heatmap(Data[num_columns].corr(), linewidth=0.2, fmt='.2f', annot=True)


Data[num_columns].corr()


Data[num_columns].corr().style.applymap(lambda v: 'background-color: orange' if abs(v) > 0.4 else '')


X=Data.drop('accident_risk', axis=1)
y=Data['accident_risk']


X


y


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=23)


X_train.shape, y_train.shape


X_test.shape, y_test.shape


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


scaler=StandardScaler()
OHE=OneHotEncoder(drop='first')


preprocessor=ColumnTransformer([
    ('StandardScaler', scaler, ['id','num_lanes','curvature','speed_limit','road_signs_present','public_road','holiday','school_season','num_reported_accidents']),
    ('OneHotEncoder', OHE, ['road_type', 'lighting', 'weather', 'time_of_day'])
])


preprocessor


X_train=preprocessor.fit_transform(X_train)
X_test=preprocessor.transform(X_test)


X_train


X_test


import xgboost

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import tensorflow as tf
print("TensorFlow version:", tf.__version__)


from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor



def Build_ANN(batch_size=None, optimizer='adam', **kwargs):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=optimizer, loss='mse')
    return model


from scikeras.wrappers import KerasRegressor

ann_model = KerasRegressor(
    model=Build_ANN,
    epochs=50,
    batch_size=32,
    verbose=0
)


def model_evaluate(true, predicted):
    mse=mean_squared_error(true, predicted)
    mae=mean_absolute_error(true, predicted)
    rmse=np.sqrt(mean_squared_error(true, predicted))
    r2=r2_score(true, predicted)
    
    return mse, mae, rmse, r2


models={
    'Linear Regression' : LinearRegression(),
    'Ridge' : Ridge(),
    'Lasso' : Lasso(alpha=0.001),
    'Decision Tree Regressor' : DecisionTreeRegressor(),
    'Ada Boost Regressor' : AdaBoostRegressor(),
    'Gradient Boosting Regressor' : GradientBoostingRegressor(),
    'Random Forest Regressor' : RandomForestRegressor(),
    'XGB Regressor' : XGBRegressor(),
    'KNeighbors Regressor' : KNeighborsRegressor(),
    'Artificial Neural Network': ann_model
}

model_list=[]
r2_list=[]

for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(X_train, y_train)

    y_train_pred=model.predict(X_train)
    y_test_pred=model.predict(X_test)

    model_train_mse, model_train_mae, model_train_rmse, model_train_r2=model_evaluate(y_train, y_train_pred)
    model_test_mse, model_test_mae, model_test_rmse, model_test_r2=model_evaluate(y_test, y_test_pred)

    print(list(models.keys())[i])

    model_list.append(list(models.keys())[i])

    print("Model Performance for Training Data : ")
    print("- Mean Squared Error : {:.4f}".format(model_train_mse))
    print("- Mean Absolute Error : {:.4f}".format(model_train_mae))
    print("- Root Mean Squared Error : {:.4f}".format(model_train_rmse))
    print("- R2 Score : {:.4f}".format(model_train_r2))
    
    
    print("---------------------------------------------------------------")
    
    print("Model Performance for Testing Data : ")
    
    print("- Mean Squared Error : {:.4f}".format(model_test_mse))
    print("- Mean Absolute Error : {:.4f}".format(model_test_mae))
    print("- Root Mean Squared Error : {:.4f}".format(model_test_rmse))
    print("- R2 Score : {:.4f}".format(model_test_r2))
    
    r2_list.append(model_test_r2)
    
    print("="*35)
    print("\n")


pd.DataFrame(list(zip(model_list, r2_list)), columns=['Model Name', 'Accuracy Score']).sort_values(by=["Accuracy Score"],ascending=False)































