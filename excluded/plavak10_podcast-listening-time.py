import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action="ignore")


df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df


df.drop('id',axis=1,inplace=True)
df


df.describe()


df.info()


df['Podcast_Name'].value_counts()


df['Episode_Title'].value_counts()


plt.hist(x='Episode_Length_minutes',data=df,bins=30)
plt.show()


df['Genre'].value_counts()


plt.hist(x='Host_Popularity_percentage',data=df,bins=30)
plt.show()


sns.countplot(x='Publication_Day',data=df)
plt.show()


sns.countplot(x='Publication_Time',data=df)
plt.show()


plt.hist(x='Guest_Popularity_percentage',data=df,bins=30)
plt.show()


df['Number_of_Ads'].value_counts()


sns.countplot(x='Episode_Sentiment',data=df)
plt.show()


plt.hist(x='Listening_Time_minutes',data=df,bins=30)
plt.show()


df


import re
df['Episode_number'] = df['Episode_Title'].str.extract(r'(\d+)').astype(int)
df


df.drop('Episode_Title',axis=1,inplace=True)
df


podcast_avg_length = df.groupby('Podcast_Name')['Episode_Length_minutes'].transform('mean')
df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(podcast_avg_length)
df


df.info()


df['Episode_Length_minutes'].max()	


df = df.query("Number_of_Ads <= 5")
df


df['Number_of_Ads'].value_counts()


avg_guest_popularity_percent = df.groupby('Podcast_Name')['Guest_Popularity_percentage'].transform('mean')
df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(avg_guest_popularity_percent)
df.info()


df


cat_columns = df.select_dtypes(include=['object','category']).columns.tolist()
cat_columns


for col in cat_columns:
    encodeded = pd.get_dummies(df[col],dtype='int',prefix=col)
    df = pd.concat([df,encodeded],axis=1)
    df.drop(col,axis=1,inplace=True)

df


from sklearn.preprocessing import StandardScaler, MinMaxScaler
mm = MinMaxScaler()
sc = StandardScaler()


y = df['Listening_Time_minutes']
X = df.drop('Listening_Time_minutes',axis=1)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)


from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers =[
        ('minmax', mm, ['Host_Popularity_percentage','Guest_Popularity_percentage']),
        ('standard',sc,['Episode_Length_minutes'])
    ],
    remainder='passthrough'
)

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)


X_train_scaled


X_test_scaled


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score


# rf = RandomForestRegressor()
# rf.fit(X_train,y_train)
# pred  = rf.predict(X_test)

# print("Random Forest RMSE",np.sqrt(mean_squared_error(y_test,pred)))
# print("Random Forest R2",r2_score(y_test,pred))


# xgb = XGBRegressor()
# xgb.fit(X_train,y_train)
# pred  = xgb.predict(X_test)

# print("XGB RMSE",np.sqrt(mean_squared_error(y_test,pred)))
# print("XGB R2",r2_score(y_test,pred))


# cat = CatBoostRegressor()
# cat.fit(X_train,y_train)
# pred  = cat.predict(X_test)

# print("Catboost RMSE",np.sqrt(mean_squared_error(y_test,pred)))
# print("Catboost R2",r2_score(y_test,pred))


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


model = Sequential()
model.add(Dense(256,activation='relu',input_shape=(77,)))
model.add(Dropout(0.4))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1))

model.summary()


model.compile(optimizer=Adam(learning_rate=0.0001),loss=Huber(delta=1.0),metrics=['mse','mae'])


# history = model.fit(X_train_scaled,y_train,epochs=100,validation_data=(X_test_scaled,y_test),verbose=1)













