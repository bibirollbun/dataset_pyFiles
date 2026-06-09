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


from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder


train_df=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')


train_df.head()


train_extra_df=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


merged_df = pd.concat([train_df, train_extra_df], ignore_index=True)


merged_df.head()


merged_df.info()


merged_df.isnull().sum()


df=merged_df.copy()


df.head()


from sklearn.impute import SimpleImputer


merged_df.isnull().sum()


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')

# Apply imputer and flatten the output
merged_df['Brand'] = imputer.fit_transform(merged_df[['Brand']]).ravel()
merged_df['Material'] = imputer.fit_transform(merged_df[['Material']]).ravel()
merged_df['Size'] = imputer.fit_transform(merged_df[['Size']]).ravel()
merged_df['Laptop Compartment'] = imputer.fit_transform(merged_df[['Laptop Compartment']]).ravel()
merged_df['Waterproof'] = imputer.fit_transform(merged_df[['Waterproof']]).ravel()
merged_df['Style'] = imputer.fit_transform(merged_df[['Style']]).ravel()
merged_df['Color'] = imputer.fit_transform(merged_df[['Color']]).ravel()



merged_df.isnull().sum()


imputer2=SimpleImputer(strategy='mean')
merged_df['Weight Capacity (kg)']=imputer2.fit_transform(merged_df[['Weight Capacity (kg)']])


merged_df.isnull().sum()


merged_df.head()


label_encoder=LabelEncoder()
merged_df['Brand']=label_encoder.fit_transform(merged_df['Brand'])
merged_df['Material']=label_encoder.fit_transform(merged_df['Material'])
merged_df['Laptop Compartment']=label_encoder.fit_transform(merged_df['Laptop Compartment'])
merged_df['Waterproof']=label_encoder.fit_transform(merged_df['Waterproof'])
merged_df['Style']=label_encoder.fit_transform(merged_df['Style'])
merged_df['Color']=label_encoder.fit_transform(merged_df['Color'])


ordinal_encoder=OrdinalEncoder()
merged_df['Size']=ordinal_encoder.fit_transform(merged_df[['Size']])


merged_df.head()


X = merged_df.iloc[:, :-1]
Y = merged_df.iloc[:, -1]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score


model1=LinearRegression()
model1.fit(X_train, y_train)


y_pred=model1.predict(X_test)


mse=mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"mean_squared_error: {mse}")
print(f"R-squared score: {r2}")
print(f"RMSE: {rmse}")


test_df=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


test_df.head()


test_df.isnull().sum()


test_df['Brand'] = imputer.fit_transform(test_df[['Brand']]).ravel()
test_df['Material'] = imputer.fit_transform(test_df[['Material']]).ravel()
test_df['Size'] = imputer.fit_transform(test_df[['Size']]).ravel()
test_df['Laptop Compartment'] = imputer.fit_transform(test_df[['Laptop Compartment']]).ravel()
test_df['Waterproof'] = imputer.fit_transform(test_df[['Waterproof']]).ravel()
test_df['Style'] = imputer.fit_transform(test_df[['Style']]).ravel()
test_df['Color'] = imputer.fit_transform(test_df[['Color']]).ravel()


test_df['Weight Capacity (kg)']=imputer2.fit_transform(test_df[['Weight Capacity (kg)']])


test_df['Brand']=label_encoder.fit_transform(test_df['Brand'])
test_df['Material']=label_encoder.fit_transform(test_df['Material'])
test_df['Laptop Compartment']=label_encoder.fit_transform(test_df['Laptop Compartment'])
test_df['Waterproof']=label_encoder.fit_transform(test_df['Waterproof'])
test_df['Style']=label_encoder.fit_transform(test_df['Style'])
test_df['Color']=label_encoder.fit_transform(test_df['Color'])


test_df['Size']=ordinal_encoder.fit_transform(test_df[['Size']])


test_df.isnull().sum()


test_df.info()


test_df.head()


submission=pd.DataFrame()


submission['id']=test_df['id']


test_pred=model1.predict(test_df)





submission['Price']=test_pred


submission.head()


model2=ElasticNet()


model2.fit(X_train, y_train)


y_pred2=model2.predict(X_test)


mse2=mean_squared_error(y_test, y_pred2)
r22=r2_score(y_test, y_pred2)
rmse2 = np.sqrt(mean_squared_error(y_test, y_pred2))


print(mse2)
print(r22)
print(rmse2)


submission.to_csv('submission.csv', index=False)


from IPython.display import FileLink
FileLink('submission.csv')


test_df.shape




