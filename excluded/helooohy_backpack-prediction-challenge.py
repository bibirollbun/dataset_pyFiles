# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train.head()


print("Data shape : ", train.shape)


#  Data Information
train.info()


# cheking the null values 

train.isnull().sum()


# dublicated Row Sum
train.duplicated().sum()



# stastical Summary

train.describe(include='all')



train.head()


# checking null vaules using seaborn 
plt.figure(figsize=(10,6))

sns.heatmap(train.isnull(), cbar=False, yticklabels=False)
plt.show()



sns.histplot(x='Price', data=train, kde=True)
plt.title("Histplot Price")


train['Material'].value_counts().plot(kind='pie', autopct='1%.1f%%')
plt.title("what kind of Material Used in percentage")


plt.figure(figsize=(12, 6))
sns.boxplot(x=train['Price'])
plt.title('Boxplot of Target Variable')
plt.show()


train['Waterproof'].value_counts().plot(kind='pie', autopct='1%.1f%%')
plt.title("How many have WaterProof in dataset ")


sns.barplot(x='Compartments', y='Price', data=train)
plt.title("cheking the Bar Plot  of Compartments and Price")


plt.figure(figsize=(10,6))
sns.barplot(x='Brand', y='Price', data=train)
plt.title("Brand Price")


plt.figure(figsize=(10,6))
sns.barplot(x='Color', y='Price', data=train)
plt.title("color Price")


plt.figure(figsize=(10,6))
sns.barplot(x='Brand', y='Weight Capacity (kg)', data=train)
plt.title("Weight Capacity (kg) Brand")


corr_matrix = train.select_dtypes(include='number').corr()


sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()


sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")   # Diverging
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")     # Sequential
sns.heatmap(corr_matrix, annot=True, cmap="viridis")    # Perceptually uniform
plt.show()


sns.pairplot(train)
plt.show()


train.isnull().sum()


cat_cols=train.select_dtypes(include='object')
cat_cols_test=test.select_dtypes(include='object')


# fill the calgorical Value  fill  mode 

for col in cat_cols:
    train[col]=train[col].fillna(train[col].mode()[0])


for col in cat_cols_test:
    test[col]=test[col].fillna(test[col].mode()[0])


train['Weight Capacity (kg)']=train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].mean())


test['Weight Capacity (kg)']=test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].mean())


train.head()


train['Laptop Compartment'].value_counts()


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, r2_score



le=LabelEncoder()
train['Style']=le.fit_transform(train['Style'])
train['Color']=le.fit_transform(train['Color'])


test['Style']=le.fit_transform(test['Style'])
test['Color']=le.fit_transform(test['Color'])


train.drop(columns=['id'], axis=1, inplace=True)





categorical_cols=['Brand','Material','Size','Laptop Compartment','Waterproof']

for col in categorical_cols:
    dummies=pd.get_dummies(train[col],prefix=col).astype(int)
    train=train.drop(col,axis=1).join(dummies)


categorical_cols_test=['Brand','Material','Size','Laptop Compartment','Waterproof']

for col in categorical_cols_test:
    dummies=pd.get_dummies(test[col],prefix=col).astype(int)
    test=test.drop(col,axis=1).join(dummies)


x=train.drop(columns=['Price'])
y=train['Price']


x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
print("RMSE",np.sqrt(mse))



model_dt= DecisionTreeRegressor(max_depth=3, random_state=42)
model_dt.fit(x_train,y_train)
y_pred_dt=model_dt.predict(x_test)
mse=mean_squared_error(y_test,y_pred_dt)
print("RMSE",np.sqrt(mse))


X_test=test.drop(columns=['id'],axis=1)


model_r= RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
model_r.fit(x_train,y_train)
y_pred_r=model_r.predict(x_test)
mse=mean_squared_error(y_test,y_pred_r)
print("RMSE",np.sqrt(mse))
y_pred_test=model_r.predict(X_test)


submission = pd.DataFrame({
    'id': test['id'],
    'Price':y_pred_test
})

submission.to_csv('submission.csv', index=False)




