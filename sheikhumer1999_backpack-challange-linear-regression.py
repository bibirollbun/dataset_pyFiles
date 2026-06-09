import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np


df_train=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_train_extra=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
df_train=pd.concat([df_train,df_train_extra], ignore_index=True)


df_train.info()


def fill_na(df):
    df_train['Brand'] = df_train['Brand'].fillna('Unknown')
    df['Material'] = df['Material'].fillna('Unknown')
    df['Size'] = df['Size'].fillna('Unknown')
    df['Laptop Compartment'] = df['Laptop Compartment'].fillna('Unknown')
    df['Waterproof'] = df['Waterproof'].fillna('Unknown')
    df['Style'] = df['Style'].fillna('Unknown')
    df['Color'] = df['Color'].fillna('Unknown')
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].mean())
    return df


df_train=fill_na(df_train)
df_test=fill_na(df_test)

print(df_train.isnull().sum())
print(df_test.isnull().sum())



def label_encode(df):
    label_encoder = LabelEncoder()
    df['Brand'] = label_encoder.fit_transform(df['Brand'])
    df['Material'] = label_encoder.fit_transform(df['Material'])
    df['Size'] = label_encoder.fit_transform(df['Size'])
    df['Laptop Compartment'] = label_encoder.fit_transform(df['Laptop Compartment'])
    df['Waterproof'] = label_encoder.fit_transform(df['Waterproof'])
    df['Style'] = label_encoder.fit_transform(df['Style'])
    df['Color'] = label_encoder.fit_transform(df['Color'])

    return df


df_train=label_encode(df_train)
df_test=label_encode(df_test)



x_train,x_test,y_train,y_test=train_test_split(df_train.drop('Price',axis=1),df_train['Price'],test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
test_pred=model.predict(df_test)


mse=mean_squared_error(y_test,y_pred)
print(np.sqrt(mse))



submission = pd.DataFrame({'id': df_test.index, 'Price': test_pred})
submission.to_csv('submission.csv', index=False)

print(submission)

