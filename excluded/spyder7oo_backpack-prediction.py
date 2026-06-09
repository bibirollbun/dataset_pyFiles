# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_pred = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


df_pred.columns


print(df.shape)
df.head(10)


df.nunique()


df.isnull().sum()


def clean_backpack_data(df_clean):

    df_clean.drop('id', axis = 1)
    print(df_clean.head())
    onehot_features = ['Material','Style']
    le_features = ['Brand','Color']
    oe_features = ['Size','Laptop Compartment','Waterproof']
    
    onehot = OneHotEncoder()
    le = LabelEncoder()
    oe = OrdinalEncoder()

    df_clean[onehot_features] = df_clean[onehot_features].fillna('Unknown')
    df_clean[le_features] = df_clean[le_features].fillna('Unknown')
    df_clean[oe_features] = df_clean[oe_features].fillna('Unknown')

    df_clean = pd.get_dummies(df_clean, columns = onehot_features, drop_first = True)
    
    for feature in le_features:
        df_clean[feature] = le.fit_transform(df_clean[feature])
        
    for feature in oe_features:
        df_clean[feature] = oe.fit_transform(df_clean[feature].values.reshape(-1,1))
    
    df_clean['Weight Capacity (kg)']=df_clean['Weight Capacity (kg)'].fillna(df_clean['Weight Capacity (kg)'].median())
    df_clean=df_clean.reset_index(drop=True)
    
    return df_clean


# aryaman bro, ye function correct kr dena
# isse encoding ka code remove krna hai and bas correlation matrix ka code rakhna hai

def create_correlation_matrix(df_cleaned):
    
    categorical_cols=['Size','Brand','Material','Style','Color']
    
    
    df_cleaned['Size']=le.fit_transform(df_cleaned['Size'])

    df_cleaned=pd.get_dummies(df_cleaned,columns=["Material","Style"],drop_first=True)

    df_cleaned["Brand-freq"]=df_cleaned["Brand"].map(df_cleaned["Brand"].value_counts()/len(df_cleaned))
    df_cleaned["Color-freq"]=df_cleaned["Color"].map(df_cleaned["Color"].value_counts()/len(df_cleaned))
    df_cleaned.drop(columns=["Brand","Color"],inplace=True)
    
    numeric_cols=df_cleaned.select_dtypes(include=[np.number,bool]).columns
    corr_matrix = df_cleaned[numeric_cols].corr()
    
    plt.figure(figsize=(12,8))
    sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',center=0,fmt='.2f')
    plt.title('Feature correlation matrix')
    plt.tight_layout()
    return corr_matrix

correlation_matrix=create_correlation_matrix(df_cleaned)
price_correlations=correlation_matrix['Price'].sort_values(ascending=False)
print(price_correlations)


df = clean_backpack_data(df)
#df.head(10)


y = df['Price']
df.drop('Price', axis = 1, inplace = True)


xtrain, xtest, ytrain, ytest = train_test_split(df,y,random_state = 42, test_size = 0.2)


model=LinearRegression()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
mse = mean_squared_error(ytest,ypred)
rmse = np.sqrt(mse)
print(rmse)

