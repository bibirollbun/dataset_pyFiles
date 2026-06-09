import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import seaborn as sns
import matplotlib.pyplot as plt


train_data=pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')
test_data=pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')
store_data=pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')


test_data.head()


store_data.head()


train_df=train_data.merge(store_data,on='Store',how='left')


train_df.head()


test_df=test_data.merge(store_data,on='Store',how='left')


test_df.head()


train_df.info()


test_df.info()


#checking for unique values in columns with nan
columns=['CompetitionOpenSinceMonth',
         'CompetitionOpenSinceYear',
         'CompetitionDistance',
         'Promo2SinceWeek',
         'Promo2SinceYear',
         'PromoInterval'
        ]
for col in columns:
    print(f"{train_df[col].value_counts()}")


train_df.duplicated().sum()


round(train_df.describe().T,2)


train_df['Date']=pd.to_datetime(train_df['Date'])
test_df['Date']=pd.to_datetime(test_df['Date'])


print(train_df['Date'].dt.year.min())
print(test_df['Date'].dt.year.max())


# sns.histplot(train_df,x='Sales',kde=True,color='r')


train_df=train_df[train_df['Open']==1].copy()
test_df=test_df[test_df['Open']==1].copy()


train_df['Open'].value_counts()


# sns.histplot(train_df,x='Sales',kde=True,color='g')


# plt.figure(figsize=(10,8))
# sns.scatterplot(x='Sales',y='Customers',data=train_df.sample(5000),hue=train_df['Date'].dt.year,alpha=0.7)
# plt.xlabel('Sales')
# plt.ylabel('Customers')
# plt.title('Sales vs Cusomers')
# plt.legend()
# plt.show()


# sns.scatterplot(x=train_df['CompetitionDistance'],y=train_df['Sales'],data=train_df.sample(10000),hue=train_df['Date'].dt.year)
# plt.xlabel('CompetitionDistance')
# plt.ylabel('Sales')
# plt.title('Sales vs CompetitionDistance')
# plt.legend()
# plt.show()


# plt.figure(figsize=(10,8))
# sns.barplot(data=train_df,x='Promo',y='Sales')


# #yearly sales
# yearly_sales=train_df.groupby(train_df['Date'].dt.year)['Sales'].sum().reset_index()
# sns.barplot(x=yearly_sales['Date'],y=yearly_sales['Sales'])


train_df.head()


# #histogram of columns with missing values
# missing_cols=['CompetitionDistance',
#               'CompetitionOpenSinceMonth',
#              'CompetitionOpenSinceYear',
#              'Promo2',
#              'Promo2SinceWeek',
#              'Promo2SinceYear',
#              'PromoInterval']
# for col in missing_cols:
#     plt.figure(figsize=(10,8))
#     sns.histplot(data=train_df,x=col,kde=True)
#     plt.xlabel(col)


#feature engineering
train_df['Month']=train_df.Date.dt.month
train_df['Year']=train_df.Date.dt.year
train_df['Day']=train_df.Date.dt.day
test_df['Month']=test_df.Date.dt.month
test_df['Year']=test_df.Date.dt.year
test_df['Day']=test_df.Date.dt.day
train_df=train_df.drop('Date',axis=1)
test_df=test_df.drop('Date',axis=1)


train_df.head()


numerical_cols=train_df.select_dtypes(include=[np.number])
numerical_cols=numerical_cols.drop(['Sales','Customers'],axis=1).columns.tolist()
print(numerical_cols)
categorical_cols=train_df.select_dtypes(include='object').columns.tolist()
print(categorical_cols)


#impute missing values
from sklearn.impute import SimpleImputer
features_to_impute=['CompetitionDistance',
                    'CompetitionOpenSinceYear',
                   'Promo2',
                   'Promo2SinceWeek',
                   'Promo2SinceYear',
                   'CompetitionOpenSinceMonth',
                   'SchoolHoliday']
imputer=SimpleImputer(strategy='median').fit(train_df[features_to_impute])
train_df[features_to_impute]=imputer.transform(train_df[features_to_impute]).copy()
test_df[features_to_impute]=imputer.transform(test_df[features_to_impute]).copy()


#scale values
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler().fit(train_df[numerical_cols])
train_df[numerical_cols]=scaler.transform(train_df[numerical_cols])
test_df[numerical_cols]=scaler.transform(test_df[numerical_cols])


round(train_df.describe().T,2)


train_df['StateHoliday'].value_counts()


train_df['StateHoliday'] = train_df['StateHoliday'].astype('str')


#onehotencoding for categorical columns
train_df[categorical_cols]=train_df[categorical_cols].fillna('unknown')
test_df[categorical_cols]=test_df[categorical_cols].fillna('unknown')
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
encoder.fit(train_df[categorical_cols])
encoded_cols=encoder.get_feature_names_out(categorical_cols)
encoded_array=encoder.transform(train_df[categorical_cols])
encoded_array_test=encoder.transform(test_df[categorical_cols])
encoded_df=pd.DataFrame(encoded_array,columns=encoded_cols,index=train_df.index)
encoded_df_test=pd.DataFrame(encoded_array_test,columns=encoded_cols,index=test_df.index)
x_train=pd.concat([train_df[numerical_cols],encoded_df],axis=1)
y_train=train_df['Sales']
test_df_data=pd.concat([test_df[numerical_cols],encoded_df_test],axis=1)


test_df_data.head()


#train valid split
targets=train_df['Sales']
size=int(0.75*len(train_df))
train_inputs=x_train[:size]
valid_inputs=x_train[size:]
train_targets=targets[:size]
valid_targets=targets[size:]

train_inputs.isna().sum()



# from sklearn.ensemble import RandomForestClassifier

# # Initial training with 10 trees
# model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=2, warm_start=True, max_depth=10)
# model.fit(train_inputs, train_targets)

# # Incrementally add more trees
# for i in range(20, 81, 10):  # Start from 20 to avoid duplicate 10
#     model.n_estimators = i  # Increase the number of trees
#     model.fit(train_inputs, train_targets)  # Fit the model again



# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# model=LinearRegression()
# model.fit(train_inputs,train_targets)
# pred=model.predict(train_inputs)
# print(mean_squared_error(train_targets,pred))


from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor(random_state=42,max_depth=37)
model.fit(train_inputs,train_targets)
pred=model.predict(train_inputs)
model.score(train_inputs,train_targets)
print(model.score(valid_inputs,valid_targets))


# from sklearn.ensemble import RandomForestClassifier
# model=RandomForestClassifier(random_state=42,n_estimators=40,max_features='sqrt')
# model.fit(train_inputs,train_targets)


prediction=model.predict(test_df_data)
prediction


output=pd.DataFrame({
    'Id':test_df['Id'],
    'Sales':prediction
})


importance_df=pd.DataFrame({
    'feature':train_inputs.columns,
    'importance':model.feature_importances_
})
importance_df.sort_values(by='importance',ascending=False)


output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


