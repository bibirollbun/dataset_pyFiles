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


# read train data. Size and shape

train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
print(train.shape)
train_extra.shape


# Our needed libraries.
# Regression tools
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV, LassoCV

# Graphing tools
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
# imputation and pipeline imports
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder as LE


train.head()


for col in train.columns:
    print(col, train[col].dtype)


new_train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
new_train_extra= pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
df_train =pd.concat([new_train,new_train_extra])
df_train.head()


df_train.describe()


obj_columns = train.select_dtypes(['object']).columns
for col in obj_columns:
    fig, ax = plt.subplots(1,2,sharex = True, figsize = (20,20))
    fig.suptitle(col)
    ax[0].set_title("Smaller dataset")
    ax[1].set_title("Larger dataset")
    sns.countplot(ax = ax[0],x=col, data= new_train)
    sns.countplot(ax = ax[1],x=col, data= df_train)
    plt.show()


no_obj_col = [col for col in new_train.columns if col not in obj_columns]
no_obj_col.remove('id')
no_obj_col


fig, ax = plt.subplots(1,2,sharex = True, figsize = (20,20))
fig.suptitle("Compartments")
ax[0].set_title("Smaller dataset")
ax[1].set_title("Larger dataset")
sns.countplot(ax = ax[0],x="Compartments", data= new_train)
sns.countplot(ax = ax[1],x="Compartments", data= df_train)
plt.show()


# fig, ax = plt.subplots(1,2, figsize = (20,20))
# fig.suptitle("Weight Capacity (kgs)")
# ax[0].set_title("Smaller dataset")
# ax[1].set_title("Larger dataset")
sns.displot(x="Weight Capacity (kg)", data= new_train,kind= "kde")
plt.show()



df_train = df_train.reset_index()


# fig, ax = plt.subplots(1,2, figsize = (20,20))
# fig.suptitle("Weight Capacity (kgs)")
ax[0].set_title("Smaller dataset")
ax[1].set_title("Larger dataset")
sns.displot(ax = ax[0],x="Weight Capacity (kg)", data= new_train,kind= "kde")
sns.displot(ax = ax[0],x="Weight Capacity (kg)", data= df_train,kind= "kde")
plt.show()


data_list = [new_train,df_train]
colX,colY = 'Weight Capacity (kg)','Price'
for data in data_list:
    df = data[['Weight Capacity (kg)','Price']].dropna()
    X,y= df['Weight Capacity (kg)'],df['Price']
    model = sm.OLS(y,X)
    results = model.fit()
    print(data.shape[0])
    print("paramaters: ",results.params)
    print("Rsquared: ",results.rsquared )


print(new_train['Weight Capacity (kg)'].describe())
print(df_train['Weight Capacity (kg)'].describe())


MissData = {}
for col in df_train.columns:
    MissData[col] = df_train[col].isna().sum()/df_train.shape[0]
MissData


for col in obj_columns:
    numeric = df_train[col].value_counts()
    print(numeric)
    print('Missing Data: ',df_train[col].isna().sum())


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoded_total = encoder.fit_transform(df_train[obj_columns]).toarray()
one_hot_df = pd.DataFrame(encoded_total,columns= encoder.get_feature_names_out(obj_columns.to_list()))
encoded_df= pd.concat([df_train,one_hot_df],axis =1)
encoded_df.head()


df = encoded_df.drop(obj_columns,axis =1)
df.head()


from sklearn.linear_model import RidgeCV
def ridgebp(df,col = None):
    if col == None:
        X,y = df.drop(['Price','id'], axis =1), df['Price']
    else:
        X,y = df.drop(['Price','id',col], axis =1), df['Price']
    cols = np.arange(0,X.shape[1])
    model = RidgeCV()
    ct = ColumnTransformer([('mean',SimpleImputer(),cols)])
    pipe = Pipeline([('trans',ct),('mdl',model)])
    pipe.fit(X,y)
    print(pipe.named_steps['mdl'].coef_)
    return(pipe.named_steps['mdl'].best_score_)


from sklearn.linear_model import LassoCV
def lassobp(df,col = None):
    if col == None:
        X,y = df.drop(['Price','id'], axis =1), df['Price']
    else:
        X,y = df.drop(['Price','id',col], axis =1), df['Price']
    cols = np.arange(0,X.shape[1])
    model = LassoCV()
    ct = ColumnTransformer([('mean',SimpleImputer(),cols)])
    pipe = Pipeline([('trans',ct),('mdl',model)])
    pipe.fit(X,y)
    print(pipe.named_steps['mdl'].coef_)
    return(np.min(pipe.named_steps['mdl'].mse_path_.mean(1)))


score = ridgebp(df)
score


score = lassobp(df)
score


display = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
display.head()


test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
for col in test.select_dtypes(['object']).columns:
    test[col]= LE().fit_transform(test[col])
test.head()


df.to_csv('submission.csv',index=False)
print('Submission file created.')


col_vals= {}
for col in obj_columns:
    col_vals[col] = np.max(train[col])
col_vals


col_scores={}
for col in obj_columns:
    n = col_vals[col]
    scores = {}
    for i in range(n):
        df=train[train[col]==i]
        scores[i]= ridgebp(df)
        print(i,df.shape[0])
    col_scores[col] = scores
col_scores
    


min_vals = {}
for col in obj_columns:
    num = 0
    for i in col_scores[col]:
        if col_scores[col][i]<num:
            num = col_scores[col][i]
    min_vals[col] = num
min_vals


import seaborn as sns


sns.boxplot(x= 'Brand', y= 'Price', data = train)


for col in obj_columns:
    sns.boxplot(x= col, y= 'Price', data = train)
    plt.show()
    


Enter

