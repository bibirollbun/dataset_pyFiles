#!pip install xgboost


import pandas as pd
import numpy as np
import matplotlib as mp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


data=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
data2=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


df=pd.DataFrame(data)
df2=pd.DataFrame(data2)


df.head()


df2.head()


df.describe()


df2.describe()


df.info()


df2.info()


Pie=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
x=Pie['Brand'].value_counts().sort_values(ascending=False)
label=['Adidas','Under Armour', 'Nike', 'Puma', 'Jansport']
plt.pie(x,labels=label)
plt.show()
x


grp=Pie.groupby('Brand').agg({'Price':['mean']})
grp


Pie.hist(figsize=(20,20),bins=20)
plt.show()


missing=df.isnull().sum().sort_values(ascending=False)
missing


#graph to judge the quantity of missing values
miss= missing[missing>100]
sns.barplot(x=miss,y=miss.index,palette='pastel')


df['Color'].value_counts()
df['Color'].mode()
df=df.fillna(value={'Color':'empty'})
df=df.fillna(value={'Brand':'empty'})
df=df.fillna(value={'Material':'empty'})
df=df.fillna(value={'Style':'empty'})
df=df.fillna(value={'Laptop Compartment':'empty'})
df=df.fillna(value={'Waterproof':'empty'})
df=df.fillna(value={'Size':'empty'})
df=df.fillna(value=df2['Weight Capacity (kg)'].mean())
df


df2['Color'].value_counts()
df2['Color'].mode()
df2=df2.fillna(value={'Color':'empty'})
df2=df2.fillna(value={'Brand':'empty'})
df2=df2.fillna(value={'Material':'empty'})
df2=df2.fillna(value={'Laptop Compartment':'empty'})
df2=df2.fillna(value={'Waterproof':'empty'})
df2=df2.fillna(value={'Size':'empty'})
df2=df2.fillna(value={'Style':'empty'})
df2=df2.fillna(value=df2['Weight Capacity (kg)'].mean())
df2.head(20)


label_enc=LabelEncoder()
cols=df.select_dtypes(['object','category']).columns
cols


df['Material']=label_enc.fit_transform(df['Material'])
df['Brand']=label_enc.fit_transform(df['Brand'])
df['Waterproof']=label_enc.fit_transform(df['Waterproof'])
df['Style']=label_enc.fit_transform(df['Style'])
df['Color']=label_enc.fit_transform(df['Color'])
df['Size']=label_enc.fit_transform(df['Size'])
df['Laptop Compartment']=label_enc.fit_transform(df['Laptop Compartment'])
df


df2['Material']=label_enc.fit_transform(df2['Material'])
df2['Brand']=label_enc.fit_transform(df2['Brand'])
df2['Waterproof']=label_enc.fit_transform(df2['Waterproof'])
df2['Style']=label_enc.fit_transform(df2['Style'])
df2['Color']=label_enc.fit_transform(df2['Color'])
df2['Size']=label_enc.fit_transform(df2['Size'])
df2['Laptop Compartment']=label_enc.fit_transform(df2['Laptop Compartment'])
df2


#copying data to another datafram
df_train=df.drop(columns=['id'])
corr=df_train.corr()
corr_price=corr['Price'].sort_values(ascending=False)
corr_price


sns.heatmap(df[corr_price.index].corr(),annot=True, cmap='Blues')
plt.show()


#training the model
x=df_train.drop(columns='Price')
y=df_train['Price']


#train-test-split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42 )



model=XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.01,
    max_depth=10,
    n_estimators=100,
    subsample=0.65,
    random_state=42,
    use_label_encoder=False
)
x


model.fit(x_train,y_train)


y_pred=model.predict(x_test)
y_pred


df_test=df2.drop(columns=['id'])
df_test
y_pred2=model.predict(df_test)
y_pred2


df2['Price']=y_pred2
df2


sub=df2.drop(columns=['Brand','Material','Size','Compartments','Laptop Compartment','Waterproof','Style','Color','Weight Capacity (kg)'])
df2.info()


sub

