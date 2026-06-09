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


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
print(train.shape)
train.head()


test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
print(test.shape)
test.head()


# concating test and train for EDA
df = pd.concat([train,test],ignore_index=True)


print(train.shape)
print(test.shape)
print(df.shape)


df.iloc[300000]


pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.width",None)


## Null treatment

# Test set doesnt have saleprice col, since we concated train and test sale price would be null, so imputed with zero
df.loc[300000:,'Price'] = int(0)


df.isnull().sum()


# filtering columns with null values greater than 20% 
null= df.isnull().sum()
null_20perc = null[(null/df.shape[0] * 100)>20]/df.shape[0]
print(null_20perc) 
cols_to_drop = null_20perc.index.to_list()
cols_to_drop


# Drop columns with more than 20% null values
print(df.shape, " before dropping")
df = df.drop(columns=cols_to_drop,axis=1)
print(df.shape, " after dropping")


others = null[(null.values>0) & (~null.index.isin(null_20perc.index))]
other_null_cols = others.index.to_list()
print(other_null_cols)


# Impute with median and mode for other null columns
for col in other_null_cols:
    if np.issubdtype(df[col].dtype,np.number):
        df[col]=df[col].fillna(df[col].median())
    else:
        df[col]=df[col].fillna(df[col].mode()[0])


df.isnull().sum()


df.describe()


df.describe(include='O')


## Univariate Analysis


num_cols = df.select_dtypes(include=np.number).columns
num_cols


cat_cols=df.select_dtypes(exclude=np.number).columns
cat_cols


import matplotlib.pyplot as plt
import seaborn as sns


n_cols = 4  # Number of columns in subplot layout
n_rows = (len(num_cols) + n_cols - 1) // n_cols  # Calculate rows needed

plt.figure(figsize=(12, n_rows * 2))  # Adjust figure size as needed

for k, i in enumerate(num_cols[1:]): #num_cols from 1st index to ignore 'id' column as it is identifier
    plt.subplot(n_rows, n_cols, k + 1)  # Create subplots
    df[i].plot(kind='kde')
    plt.title(i)

plt.tight_layout()  
plt.show() 


# All the numerical variable is not normally distributed, so we can proceed with tree / ensemble models


# Outlier Detection using Box Plot
n_cols = 4  
n_rows = (len(num_cols) + n_cols - 1) // n_cols  

plt.figure(figsize=(12, n_rows * 2))

for k, i in enumerate(num_cols[1:]): #num_cols from 1st index to ignore 'id'  column as it is identifier
    plt.subplot(n_rows, n_cols, k + 1) 
    df[i].plot(kind='box')
    plt.title(i)

plt.tight_layout()  
plt.show() 


## Bivariate Analysis

n_cols = 2
n_rows = (len(num_cols) + n_cols - 1) // n_cols  

plt.figure(figsize=(12, n_rows * 3))  
for k,col in enumerate(cat_cols[0:4]):
    plt.subplot(n_rows, n_cols, k+1)
    sns.barplot(x=col, y='Price',data=df,estimator=np.mean)
    plt.xticks(rotation=65)
    plt.title(f"Price vs {col}")
    
plt.tight_layout()  
plt.show() 


n_cols = 2
n_rows = (len(num_cols) + n_cols - 1) // n_cols  

plt.figure(figsize=(12, n_rows * 3))  
for k,col in enumerate(cat_cols[4:8]):
    plt.subplot(n_rows, n_cols, k+1)
    sns.barplot(x=col, y='Price',data=df,estimator=np.mean)
    plt.xticks(rotation=65)
    plt.title(f"SalePrice vs {col}")
    
plt.tight_layout()  
plt.show() 


corr=df[num_cols].corr()
sns.heatmap(data=corr,annot=False, cmap = "RdBu")


df['Size'].value_counts()


## Encoding
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
freq_code = []
dummy_code = []
for col in cat_cols:
    length=len(df[col].value_counts().index)
    if length>4:
        freq_code.append(col)
    else:
        dummy_code.append(col)
freq_code,dummy_code


dummy = ce.OneHotEncoder(cols=dummy_code)
freq = ce.CountEncoder(cols=freq_code)
targ = ce.TargetEncoder()
le = LabelEncoder()
df_new = df.copy()
df_new = df_new.drop(columns=cat_cols,axis=1)
df_dummy = dummy.fit_transform(df[['Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style']])
df_new = pd.concat([df_new,df_dummy],axis=1)


for col in ['Brand','Color']:
    df_new[col] = le.fit_transform(df[col])
df_new.head()


print(train.shape)
print(test.shape)


train_df = df_new.iloc[0:299999]
test_df = df_new.iloc[300000:]
test_df = test_df.drop(columns='Price',axis=1)


test_df.reset_index(inplace=True,drop=True)


print(train_df.shape)
print(test_df.shape)


from sklearn.model_selection import cross_val_predict, RandomizedSearchCV, GridSearchCV
from scipy.stats import randint,uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from xgboost import XGBRegressor 



xtrain = train_df.drop(columns=['id','Price'],axis=1)
ytrain = train_df['Price']


xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100)

xgb.fit(xtrain,ytrain)


ypred=cross_val_predict(xgb,xtrain,ytrain,cv=5)
print('cross-validation-rmse of XGB',np.sqrt(mse(ytrain,ypred)))


xgb = XGBRegressor()

xgb_param_grid = {
    'n_estimators':[100,200],
    'max_depth':randint(3,10),
    'learning_rate':uniform(0.01,0.3),
    'subsample': uniform(0.7,0.9)
}

xgb_search = RandomizedSearchCV(estimator=xgb,param_distributions=xgb_param_grid,
                                scoring='neg_mean_squared_error',cv=3,
                               n_iter=50,random_state=11,verbose=1)

xgb_search.fit(xtrain,ytrain)


best_xgb_model = xgb_search.best_estimator_
best_ypred=cross_val_predict(best_xgb_model,xtrain,ytrain,cv=5)
print('cross-validation-rmse of Tuned XGB ',np.sqrt(mse(ytrain,best_ypred)))


model = XGBRegressor()

param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'n_estimators': [100, 200]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5)
grid_search.fit(xtrain, ytrain)


xgb_gridcv = grid_search.best_estimator_
print("Best Score:",grid_search.best_score_)


preds = xgb_gridcv.predict(test_df.drop(columns='id',axis=1))
preds


submission = pd.concat([test_df['id'],pd.Series(preds)],axis=1,ignore_index=True)
submission.columns = ['id','Price']
submission.head()


submission.to_csv('submission.csv',index=False)




