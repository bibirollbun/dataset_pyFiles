# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer,make_column_selector,ColumnTransformer

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train_df=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')

train_df=pd.concat([train_df,train_extra],ignore_index=True)


train_df.shape


train_df.shape


train_df.columns


train_df.head()


train_df.info()


train_df.describe()


ax=((train_df.isna().sum()/train_df.shape[0])*100).round(2)
plt.barh(ax.index,ax.values,color='pink')


train_df.head(10)


brand=train_df['Brand'].value_counts()
brand



palette_color=sns.color_palette('dark')
explode=[0,0,0.1,0,0]
plt.pie(brand.values,labels=brand.index,explode=explode,colors=palette_color,autopct='%.2f%%')
plt.title('Distribution of Brands')


size=train_df['Size'].value_counts()
size


palette_color=sns.color_palette('dark')
explode=[0,0.1,0]
plt.pie(size.values,labels=size.index,explode=explode,colors=palette_color,autopct='%.2f%%')
plt.title('Distribution of Sizes')


mat=train_df['Material'].value_counts()
mat


palette_color=sns.color_palette('dark')
explode=[0,0.1,0,0]
plt.pie(mat.values,labels=mat.index,explode=explode,colors=palette_color,autopct='%.2f%%')
plt.title('Distribution of Materials')


train_df.columns


X=train_df.drop(columns=['id','Price'])
y=train_df['Price']


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=42)


X_train.isna().sum()


# outlier detection in Compartment in training data
sns.boxplot(X_train['Compartments'].values)
plt.title('Outliers detection in Compartments')


# outlier detection in Compartment in test data
sns.boxplot(X_test['Compartments'].values)
plt.title('Outliers detection in Compartments')


# outlier detection in Weight in training data
sns.boxplot(X_train['Weight Capacity (kg)'].values)
plt.title('Outliers detection in Weight')


# outlier detection in Weight in testing data
sns.boxplot(X_test['Weight Capacity (kg)'].values)
plt.title('Outliers detection in Weight')


X_train.columns


imputer=SimpleImputer(strategy='mean')
X_train['Weight Capacity (kg)']=imputer.fit_transform(X_train[['Weight Capacity (kg)']])

imputer=SimpleImputer(strategy='most_frequent')
X_train[['Brand','Material','Size','Laptop Compartment','Waterproof','Style','Color']]=imputer.fit_transform(X_train[['Brand','Material','Size','Laptop Compartment','Waterproof','Style','Color']])


trans=OrdinalEncoder()
X_train[['Laptop Compartment','Waterproof']]=trans.fit_transform(X_train[['Laptop Compartment','Waterproof']])


trans=LabelEncoder()
X_train['Brand']=trans.fit_transform(X_train['Brand'])
X_train['Material']=trans.fit_transform(X_train['Material'])
X_train['Size']=trans.fit_transform(X_train['Size'])
X_train['Style']=trans.fit_transform(X_train['Style'])
X_train['Color']=trans.fit_transform(X_train['Color'])



imputer=SimpleImputer(strategy='mean')
X_test['Weight Capacity (kg)']=imputer.fit_transform(X_test[['Weight Capacity (kg)']])
imputer=SimpleImputer(strategy='most_frequent')
X_test[['Brand','Material','Size','Laptop Compartment','Waterproof','Style','Color']]=imputer.fit_transform(X_test[['Brand','Material','Size','Laptop Compartment','Waterproof','Style','Color']])
trans=OrdinalEncoder()
X_test[['Laptop Compartment','Waterproof']]=trans.fit_transform(X_test[['Laptop Compartment','Waterproof']])
trans=LabelEncoder()
X_test['Brand']=trans.fit_transform(X_test['Brand'])
X_test['Material']=trans.fit_transform(X_test['Material'])
X_test['Size']=trans.fit_transform(X_test['Size'])
X_test['Style']=trans.fit_transform(X_test['Style'])
X_test['Color']=trans.fit_transform(X_test['Color'])


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)


y_linear=lr.predict(X_test)


r2=r2_score(y_test,y_linear)
print(r2)


np.sqrt(mean_squared_error(y_test,y_linear))


X_test1=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
X_test=X_test1.drop(columns='id')
imputer=SimpleImputer(strategy='mean')
X_test['Weight Capacity (kg)']=imputer.fit_transform(X_test[['Weight Capacity (kg)']])
imputer=SimpleImputer(strategy='most_frequent')
X_test[['Brand','Material','Size','Laptop Compartment','Waterproof','Style','Color']]=imputer.fit_transform(X_test[['Brand','Material','Size','Laptop Compartment','Waterproof','Style','Color']])
trans=OrdinalEncoder()
X_test[['Laptop Compartment','Waterproof']]=trans.fit_transform(X_test[['Laptop Compartment','Waterproof']])
trans=LabelEncoder()
X_test['Brand']=trans.fit_transform(X_test['Brand'])
X_test['Material']=trans.fit_transform(X_test['Material'])
X_test['Size']=trans.fit_transform(X_test['Size'])
X_test['Style']=trans.fit_transform(X_test['Style'])
X_test['Color']=trans.fit_transform(X_test['Color'])


X_test1['Price']=lr.predict(X_test)


X_test1.head()


X_test1[['id','Price']].to_csv('/kaggle/working/submission.csv',index=False)




