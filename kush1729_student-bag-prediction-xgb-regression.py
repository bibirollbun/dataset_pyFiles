# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
import xgboost as xgb
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv(r'/kaggle/input/playground-series-s5e2/train.csv')
vdf  = pd.read_csv(r'/kaggle/input/playground-series-s5e2/training_extra.csv') #validation dataset
test = pd.read_csv(r'/kaggle/input/playground-series-s5e2/test.csv')
submission_csv = pd.read_csv(r'/kaggle/input/playground-series-s5e2/sample_submission.csv')
train.head()


train.info()


train.shape


# Check null percentage in each feature 
train.isnull().sum()/ train.shape[0] *100


df = train.dropna(axis=0)


# Avoid exaggerating by filling missing values.
# We still have huge data for EDA and Training.
df.shape


df.describe()


df.head()


plt.figure(figsize= (8,5))
sns.histplot(df['Brand'])
plt.title("Frequency Distribution of Brand")
plt.show()
print(df['Brand'].value_counts())


sns.histplot(df['Material'])
plt.show()
print(df['Material'].value_counts())


df.head()


sns.histplot(df['Size'])
plt.show()
print(df['Size'].value_counts())


sns.histplot(df['Laptop Compartment'])
plt.show()
print(df['Laptop Compartment'].value_counts())


sns.histplot(df['Color'])
plt.show()
print(df['Color'].value_counts())


sns.histplot(df['Style'])
plt.show()
print(df['Style'].value_counts())


df.head()


sns.displot(df['Compartments'])
plt.xticks(np.arange(1,11))
plt.show()
print(df['Compartments'].value_counts())


sns.boxplot(df['Weight Capacity (kg)'])
plt.show()


sns.distplot(df['Weight Capacity (kg)'])


df.head()


# As we know price is target variable so, we will try to find how feature impacting price

df.groupby(['Brand'])['Price'].median().plot(kind='bar')
plt.ylabel('Average Price')
plt.show()


brand_by_weight =  df.groupby(['Brand'])['Price'].sum()

brand_by_weight.plot(kind='bar')
plt.show()
print(brand_by_weight)


segm =  df.groupby(['Style'])['Price'].median()

segm.plot(kind='bar')
plt.show()
print(segm)


segm =  df.groupby(['Material'])['Price'].sum()

segm.plot(kind='bar')
plt.show()
print(segm)


df.head()


plt.scatter(df['Compartments'], df['Price'])


pv1 = pd.pivot_table(data=df,index = 'Brand',columns = 'Size', values = 'Price')
pv1.head()


sns.heatmap(pv1, annot=True)


df.head()


sns.heatmap(df[['Compartments','Weight Capacity (kg)','Price']].corr(), annot=True)


def create_dummies(df, col):
    dum = pd.get_dummies(drop_first=True, data= df[col], dtype='int')
    
    conc = pd.concat([df,dum], axis=1)
    conc.drop(col, axis =1, inplace=True)
    return conc


df = create_dummies(df,'Brand')
df = create_dummies(df,'Material')
df = create_dummies(df, 'Size')
df = create_dummies(df, 'Style')
df = create_dummies(df,'Color')
df.head()



df['Waterproof'] = df['Waterproof'].apply(lambda x : 0 if x == "No" else 1)
df['Laptop Compartment'] = df['Laptop Compartment'].apply(lambda x : 0 if x == "No" else 1)


df.head()


df.corr()


df.head()


Y_train = df.pop('Price')
X_train = df


X_train.pop('id')
X_train.head()


# Create XGBoost regressor
xgb_regressor = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100,random_state=42)

# Train the model
xgb_regressor.fit(X_train, Y_train)
rfe = RFE(estimator=xgb_regressor,n_features_to_select=15)

rfe = rfe.fit(X_train, Y_train)

print(rfe.support_)


x_train = X_train.loc[:,list(rfe.support_)]
x_train.head()


# Create XGBoost regressor
xgb_regressor = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100,random_state=42)

# Train the model
xgb_regressor.fit(X_train, Y_train)



vdf.head()


vdf = create_dummies(vdf,'Brand')
vdf = create_dummies(vdf,'Material')
vdf = create_dummies(vdf, 'Size')
vdf = create_dummies(vdf, 'Style')
vdf = create_dummies(vdf,'Color')
vdf.head()

test = create_dummies(test,'Brand')
test = create_dummies(test,'Material')
test = create_dummies(test, 'Size')
test = create_dummies(test, 'Style')
test = create_dummies(test,'Color')



vdf['Waterproof'] = vdf['Waterproof'].apply(lambda x : 0 if x == "No" else 1)
vdf['Laptop Compartment'] = vdf['Laptop Compartment'].apply(lambda x : 0 if x == "No" else 1)

test['Waterproof'] = test['Waterproof'].apply(lambda x : 0 if x == "No" else 1)
test['Laptop Compartment'] = test['Laptop Compartment'].apply(lambda x : 0 if x == "No" else 1)


vdf.head()


vdf.pop('id')
test.pop('id')
Y_vtrain = vdf.pop('Price')
X_vtrain = vdf








X_vtrain = X_vtrain[list(X_train.columns)]

X_test = test[list(test.columns)]


X_vtrain.head()


y_vpredict = xgb_regressor.predict(X_vtrain)


# Evaluation for Validation Set
mean_squared_error(y_vpredict, Y_vtrain, squared=False)


y_pred = xgb_regressor.predict(X_test)


submission_csv['Price'] = y_pred
submission_csv.head()


submission_csv.to_csv("submission.csv", index=False)




