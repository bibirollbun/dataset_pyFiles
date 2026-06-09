import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv',index_col=0)


df.head()


df.shape


df['Podcast_Name'].value_counts()


df['Episode_Title'].value_counts()


df['Episode_Length_minutes'].value_counts()


df['Number_of_Ads'].value_counts()


df.info()


df.describe()


df.columns


df.isnull().sum()


df['Episode_Length_minutes'].fillna(value=df['Episode_Length_minutes'].median(), inplace=True)


df['Guest_Popularity_percentage'].fillna(value=df['Guest_Popularity_percentage'].median(),inplace=True)


df['Number_of_Ads'].fillna(value=df['Number_of_Ads'].median(),inplace=True)


numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()


df_numric=df[numerical_columns]


df_numric.head()


df_numric.corr()


df.boxplot(figsize=(20,10))
plt.show()


sns.boxplot(df['Number_of_Ads'])
plt.show()


q1=df['Number_of_Ads'].quantile(0.25)
q3=df['Number_of_Ads'].quantile(0.75)
iqr=q3-q1
lower_bound=q1-(1.5*iqr)
upper_bound=q3+(1.5*iqr)
print(f'lower bound: {lower_bound}')
print(f'upper_bound: {upper_bound}')


outliers=df['Number_of_Ads'][(df['Number_of_Ads']<lower_bound)|(df['Number_of_Ads']>upper_bound)]
print(outliers)


# handling outliers
df['Number_of_Ads']=df['Number_of_Ads'].apply(lambda x:lower_bound if x<lower_bound else (upper_bound if x>upper_bound else x))


sns.boxplot(df['Number_of_Ads'])
plt.show()


numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()


numerical_columns


sns.displot(df['Episode_Length_minutes'],kde=True)
plt.show()





sns.displot(df['Host_Popularity_percentage'],kde=True)
plt.show()


sns.displot(df['Guest_Popularity_percentage'],kde=True)
plt.show()


sns.displot(df['Number_of_Ads'],kde=True)
plt.show()


df_cat=df.select_dtypes(include=['object']).columns.tolist()
df_cat


categorical_df = df[df_cat]
categorical_df.head()


categorical_df.isnull().sum()


categorical_df.describe()


categorical_df.nunique()


categorical_df.shape


plt.figure(figsize=(10,5))
sns.countplot(data=df, x='Genre', order=df['Genre'].value_counts().index, palette='viridis')
plt.xticks(rotation=45)
plt.title('Distribution of podcast genres')
plt.show()


plt.figure(figsize=(10,5))
sns.countplot(data=df,x='Publication_Day',order=df['Publication_Day'].value_counts().index,palette='viridis')
plt.xticks(rotation=45)
plt.title('Distribution of Publicaiton Days')
plt.show()


# plot of publication time
plt.figure(figsize=(10,5))
sns.countplot(data=df,x='Publication_Time',order=df['Publication_Time'].value_counts().index,palette='viridis')
plt.xticks(rotation=45)
plt.title('Distribution of publication_time')
plt.show()


# Plot distribution of Episode_sentiment
plt.figure(figsize=(10,5))
sns.countplot(data=df,x='Episode_Sentiment',order=df['Episode_Sentiment'].value_counts().index,palette='viridis')
plt.xticks(rotation=45)
plt.title('Distribution of Episode_Sentiment')
plt.show()


from sklearn.preprocessing import LabelEncoder


label_encoder=LabelEncoder()


df_cat


for column in df_cat:
    df[column]=label_encoder.fit_transform(df[column])


correlation_matrix = df.corr()




# Plot the heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# Focus on correlation with Lisening Time
listening_time_corr=correlation_matrix['Listening_Time_minutes'].sort_values(ascending=False)
print(listening_time_corr)


x=df_rem.drop([ 'Listening_Time_minutes'],axis=1)
y=df_rem['Listening_Time_minutes']


x.head()


y.head()


from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


x_train.head()


x_train.shape


x_test.head()


x_test.shape


from sklearn.preprocessing import StandardScaler


ss=StandardScaler()


x_train_transform=ss.fit_transform(x_train)
x_test_transform=ss.transform(x_test)


x_train


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


model=LinearRegression()


model.fit(x_train,y_train)


y_pred=model.predict(x_test)


mse=mean_squared_error(y_test,y_pred)


mse


from math import sqrt

sqrt(mse)


r2=r2_score(y_test,y_pred)


r2


from sklearn.ensemble import  RandomForestRegressor



rf_model = RandomForestRegressor(n_estimators=100,random_state=42)


rf_model.fit(x_train,y_train)


y_pred_rf=rf_model.predict(x_test)


y_pred_rf


mse_rf=mean_squared_error(y_test,y_pred_rf)


mse_rf


sqrt(mse_rf)


r2_score(y_test,y_pred_rf)


residuals=y_test-y_pred_rf
plt.scatter(y_test,residuals,alpha=0.7,color='blue')
plt.axhline(y=0,color='red',linestyle='--')
plt.xlabel('Actual Listening Time')
plt.ylabel('Residuals')
plt.title('Residual Analysis - Random Forest')
plt.show()


import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense


model=Sequential()


x_train_transform.shape


model.add(Dense(16,activation='relu',input_dim=8))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='linear'))


model.summary()


model.compile(loss='mean_squared_error',optimizer='Adam')


model.fit(x_train_transform,y_train,epochs=10,validation_split=0.2)



y_pred_nn=model.predict(x_test)
mse_nn=mean_squared_error(y_test,y_pred)




sqrt(mse_nn)


from sklearn.ensemble import GradientBoostingRegressor
gb_model=GradientBoostingRegressor(n_estimators=200,learning_rate=0.1,max_depth=5,random_state=42)



gb_model.fit(x_train_transform,y_train)


y_test=gb_model.predict(x_test)


from sklearn.metrics import mean_squared_error

mse=mean_squared_error(y_test, y_pred_gb)
print(mse)


sqrt(mse)


df_test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv',index_col=0)


df_test.head()


df_test.shape


df_test.isnull().sum()


df_test.info()


df_test['Episode_Length_minutes'].fillna(value=df_test['Episode_Length_minutes'].median(),inplace=True)


df_test['Guest_Popularity_percentage'].fillna(value=df_test['Guest_Popularity_percentage'].median(),inplace=True)


df_test.info()


df_test.boxplot(figsize=(20,10))
plt.show()


q1=df_test['Episode_Length_minutes'].quantile(0.25)
q3=df_test['Episode_Length_minutes'].quantile(0.75)
iqr=q3-q1
lower_bound=q1-(1.5*iqr)
upper_bound=q3+(1.5*iqr)
print(f'lower bound: {lower_bound}')
print(f'upper_bound: {upper_bound}')



outliers=df_test['Episode_Length_minutes'][(df_test['Episode_Length_minutes']<lower_bound)|(df_test['Episode_Length_minutes']>upper_bound)]
print(outliers)


# Handling the outlier
df_test['Episode_Length_minutes']=df_test['Episode_Length_minutes'].apply(lambda x:lower_bound if x<lower_bound else (upper_bound if x>upper_bound else x))


sns.boxplot(df_test['Episode_Length_minutes'])
plt.show()


# handling the outlier of Number_of_Ads
q1=df_test['Number_of_Ads'].quantile(0.25)
q3=df_test['Number_of_Ads'].quantile(0.75)
iqr=q3-q1
lower_bound=q1-(1.5*iqr)
upper_bound=q3+(1.5*iqr)
print(f'lower bound: {lower_bound}')
print(f'upper_bound: {upper_bound}')



outliers=df_test['Number_of_Ads'][(df_test['Number_of_Ads']<lower_bound)|(df_test['Number_of_Ads']>upper_bound)]
print(outliers)


df_test['Number_of_Ads']=df_test['Number_of_Ads'].apply(lambda x:lower_bound if x<lower_bound else (upper_bound if x>upper_bound else x))


sns.boxplot(df_test['Number_of_Ads'])
plt.show()



numerical_columns=df_test.select_dtypes(include=[np.number]).columns.tolist()


numerical_columns


sns.displot(df_test['Episode_Length_minutes'],kde=True)
plt.show()


sns.displot(df_test['Host_Popularity_percentage'],kde=True)
plt.show()


sns.displot(df_test['Guest_Popularity_percentage'],kde=True)
plt.show()


sns.displot(df_test['Number_of_Ads'],kde=True)
plt.show()


sns.displot(df_test['Episode_Length_minutes'],kde=True)
plt.show()


df_test_cat=df_test.select_dtypes(include=['object']).columns.tolist()
df_test_cat


categorical_df_test=df_test[df_test_cat]
categorical_df_test.head()


categorical_df_test.isnull().sum()


categorical_df_test.describe()


categorical_df_test.nunique()


categorical_df_test.shape


plt.figure(figsize=(10,5))
sns.countplot(data=categorical_df_test, x='Genre', order=categorical_df_test['Genre'].value_counts().index, palette='viridis')
plt.xticks(rotation=45)
plt.title('Distribution of podcast genres')
plt.show()


plt.figure(figsize=(10,5))
sns.countplot(data=categorical_df_test,x='Publication_Day',order=categorical_df_test['Publication_Day'].value_counts().index,palette='viridis')
plt.xticks(rotation=45)
plt.title('Distribution of Publicaiton Days')


plt.figure(figsize=(10,5))
sns.countplot(data=categorical_df_test,x='Publication_Time',order=categorical_df_test['Publication_Time'].value_counts().index,palette='viridis')
plt.xticks(rotation=45)
plt.title('Distribution of publication_time')
plt.show()


plt.figure(figsize=(10,5))
sns.countplot(data=categorical_df_test,x='Episode_Sentiment',order=categorical_df_test['Episode_Sentiment'].value_counts().index,palette='viridis')
plt.xticks(rotation=45)
plt.title('Distribution of Episode_Sentiment')
plt.show()


for column in df_test_cat:
    df_test[column]=label_encoder.fit_transform(df_test[column])


df_test.head()



df_test_scaled=ss.transform(df_test)


df_test_pred=rf_model.predict(df_test_scaled)


df_test_pred


submission_rf=pd.DataFrame({id:df_test.index,'Listening_Time_minutes':df_test_pred})


submission_rf.head()


submission_rf.to_csv('submission_rf+1.csv',index=False)

