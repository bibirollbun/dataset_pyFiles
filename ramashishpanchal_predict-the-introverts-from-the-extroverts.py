# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


data=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

target=data['Personality']

data.shape


data.head()


data.info()


data.describe()


numerical_columns=data.select_dtypes(include=['float64','int64']).columns

table=data[numerical_columns]
table['Personality']=data['Personality']
table['Personality']=table['Personality'].map(
    {'Introvert':0,
     'Extrovert':1
    })
correlation_table=table.corr()

plt.figure(figsize=(12,6))
sns.heatmap(correlation_table,annot=True)
table


fig,axis=plt.subplots(5,3,figsize=(15,20))
i=0
for col in numerical_columns:
    if col=='id' or col=='Personality':
        continue
    sns.kdeplot(x=col,data=table,ax=axis[i,0])
    sns.boxplot(x=col,data=table,ax=axis[i,1])
    sns.kdeplot(x=col,data=table,hue='Personality',ax=axis[i,2])
    i=i+1


data[data['Time_spent_Alone']>8]


categorical_columns=data.select_dtypes(include='object').columns

frame1=data['Stage_fear'].map({
    'No':0,
    'Yes':1
})
frame2=data['Drained_after_socializing'].map({
    'No':0,
    'Yes':1
})
frame3=data['Personality'].map({
    'Introvert':0,
    'Extrovert':1
})
cat_table=pd.DataFrame({
    'Stage_fear':frame1,
    'Drained_after_socializing':frame2,
    'Personality':frame3
})
cat_table_correlation=cat_table.corr()
sns.heatmap(cat_table_correlation,annot=True)


#trying to fill 'Drained_after_socializing' column using 'Stage_fear' column

print(f"Before Null values : {data['Drained_after_socializing'].isnull().sum()}")

mask1=data['Drained_after_socializing'].isna()
mask2=data['Stage_fear']=='No'

data.loc[mask1 & mask2,'Drained_after_socializing']='No'

mask3=data['Drained_after_socializing'].isna()
mask4=data['Stage_fear']=='Yes'

data.loc[mask1 & mask2,'Drained_after_socializing']='Yes'

print(f"Before Null values : {data['Drained_after_socializing'].isnull().sum()}")


data.info()



data[numerical_columns].info()


#removing useless columns
data.drop(columns=['id','Stage_fear','Personality'],inplace=True)

test_id=test['id']
test.drop(columns=['id','Stage_fear'],inplace=True)

data.head()


#filling time_spent_alone

mask1=(~data['Social_event_attendance'].isna()) & (data['Social_event_attendance'] >= data['Social_event_attendance'].mean())
mask2=(~data['Going_outside'].isna()) & (data['Going_outside'] >= data['Going_outside'].mean())
mask3=(~data['Friends_circle_size'].isna()) & (data['Friends_circle_size'] >= data['Friends_circle_size'].mean())
mask4=(~data['Post_frequency'].isna()) & (data['Post_frequency'] >= data['Post_frequency'].mean())

condition=(mask1 & mask2 & mask3 & mask4) | (mask1 & mask2 & mask3 ) | (mask1 & mask2 & mask4 ) | (mask1 & mask3 & mask4 ) | (mask2 & mask3 & mask4 ) | (mask1 & mask2) | (mask1 & mask3) | (mask1 & mask4) | (mask2 & mask3) |(mask2 & mask4) | (mask3 & mask4)

data.loc[condition,'Time_spent_Alone']=data['Time_spent_Alone'].min()

data.info()


mask1=(~test['Social_event_attendance'].isna()) & (test['Social_event_attendance'] >= test['Social_event_attendance'].mean())
mask2=(~test['Going_outside'].isna()) & (test['Going_outside'] >= test['Going_outside'].mean())
mask3=(~test['Friends_circle_size'].isna()) & (test['Friends_circle_size'] >= test['Friends_circle_size'].mean())
mask4=(~test['Post_frequency'].isna()) & (test['Post_frequency'] >= test['Post_frequency'].mean())

condition=(mask1 & mask2 & mask3 & mask4) | (mask1 & mask2 & mask3 ) | (mask1 & mask2 & mask4 ) | (mask1 & mask3 & mask4 ) | (mask2 & mask3 & mask4 ) | (mask1 & mask2) | (mask1 & mask3) | (mask1 & mask4) | (mask2 & mask3) |(mask2 & mask4) | (mask3 & mask4)

test.loc[condition,'Time_spent_Alone']=test['Time_spent_Alone'].min()

test.info()


#trying to fill post_frequency,Social_event_attendance,Going_outside,Friends_circle_size using time_spent alone column

mask1=(~data['Time_spent_Alone'].isna()) & (data['Time_spent_Alone'] >= data['Post_frequency'].mean())
data.loc[mask1,'Post_frequency']=data['Post_frequency'].min()

mask1=(~data['Time_spent_Alone'].isna()) & (data['Time_spent_Alone'] >= data['Social_event_attendance'].mean())
data.loc[mask1,'Social_event_attendance']=data['Social_event_attendance'].min()

mask1=(~data['Time_spent_Alone'].isna()) & (data['Time_spent_Alone'] >= data['Going_outside'].mean())
data.loc[mask1,'Going_outside']=data['Going_outside'].min()

mask1=(~data['Time_spent_Alone'].isna()) & (data['Time_spent_Alone'] >= data['Friends_circle_size'].mean())
data.loc[mask1,'Friends_circle_size']=data['Friends_circle_size'].min()

data.info()


mask1=(~test['Time_spent_Alone'].isna()) & (test['Time_spent_Alone'] >= test['Post_frequency'].mean())
test.loc[mask1,'Post_frequency']=test['Post_frequency'].min()

mask1=(~test['Time_spent_Alone'].isna()) & (test['Time_spent_Alone'] >= test['Social_event_attendance'].mean())
test.loc[mask1,'Social_event_attendance']=test['Social_event_attendance'].min()

mask1=(~test['Time_spent_Alone'].isna()) & (test['Time_spent_Alone'] >= test['Going_outside'].mean())
test.loc[mask1,'Going_outside']=test['Going_outside'].min()

mask1=(~test['Time_spent_Alone'].isna()) & (test['Time_spent_Alone'] >= test['Friends_circle_size'].mean())
test.loc[mask1,'Friends_circle_size']=test['Friends_circle_size'].min()


from sklearn.impute import SimpleImputer
imputer={}
for col in ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside','Friends_circle_size', 'Post_frequency']:
    imputer[col]=SimpleImputer(strategy='mean')
    data[col]=imputer[col].fit_transform(data[[col]]).ravel()
    test[col]=imputer[col].transform(test[[col]]).ravel()
    
imputer['Drained_after_socializing']=SimpleImputer(strategy='most_frequent')
data['Drained_after_socializing']=imputer['Drained_after_socializing'].fit_transform(data[['Drained_after_socializing']]).ravel()
test['Drained_after_socializing']=imputer['Drained_after_socializing'].transform(test[['Drained_after_socializing']]).ravel()

data.info()


test.info()


data.columns


#encoding columns
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()
data['Drained_after_socializing']=encoder.fit_transform(data['Drained_after_socializing'])
test['Drained_after_socializing']=encoder.transform(test['Drained_after_socializing'])

target=target.map(
    {
        'Introvert':0,
        'Extrovert':1
    }
)

data.head()


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

data_scaled=scaler.fit_transform(data)
data=pd.DataFrame(data_scaled,columns=data.columns)

test_scaled=scaler.transform(test)
test=pd.DataFrame(test_scaled,columns=test.columns)

test.head()


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization

model=Sequential()
model.add(Dense(64,activation='relu',input_dim=6))
model.add(BatchNormalization())

model.add(Dense(32,activation='relu'))
model.add(BatchNormalization())

model.add(Dense(16,activation='relu'))
model.add(BatchNormalization())

model.add(Dense(1,activation='sigmoid'))


model.summary()


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


model.fit(data,target,epochs=20,batch_size=64,validation_split=0.2)



pred=model.predict(test)
pred=pred.ravel()

for i in range(len(pred)):
    if pred[i] < 0.5 :
        pred[i]=0
    else :
        pred[i]=1

pred=pd.Series(pred,name='Personality')

pred=pred.map({
    0 : 'Introvert',
    1 : 'Extrovert'
})

pred


result={
    'id':test_id,
    'Personality':pred
}
result=pd.DataFrame(result)
result.to_csv('fourth_submission.csv',index=False)




