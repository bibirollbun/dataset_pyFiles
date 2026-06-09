import warnings
warnings.simplefilter('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from collections import Counter

import optuna

from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np


# 4-6
delay_4_6 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv', encoding='utf-8')
not_delay_4_6 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv', encoding='utf-8', low_memory=False)

#7-9
delay_7_9 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv', encoding='utf-8')
not_delay_7_9 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv', encoding='utf-8', low_memory=False)

#pilot
pilot = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/PILOT_10.csv', encoding= 'utf-8')

# Data concat
data_4_6 = pd.concat([delay_4_6, not_delay_4_6],join='inner', ignore_index=True)
data_7_9 = pd.concat([delay_7_9, not_delay_7_9], join='inner', ignore_index=True)
full_data = pd.concat([data_4_6, data_7_9], join='inner', ignore_index=True)
print(data_4_6.shape, data_7_9.shape,full_data.shape, pilot.shape)


def preprocessing(df):
    # drop data leak feature - ['REASON_CD', 'QTUF_RCV_NO', 'SOUF_RCV_NO']
    df = df.drop(columns=['REASON_CD', 'QTUF_RCV_NO', 'SOUF_RCV_NO'])
    # drop feature has correlation = 1 - ['SUPPLIER INV AMOUNT', 'Stock class', 'ALLOCATION QTY', 'SPECIAL DIV']
    df = df.drop(columns=['SUPPLIER INV AMOUNT', 'Stock class', 'ALLOCATION QTY', 'SPECIAL DIV'])
    # drop SUBSIDIARY_CD
    df = df.drop(columns='SUBSIDIARY_CD')
    
    # Ensure date and add new features
    df['Order date'] = pd.to_datetime(df['Order date'], format='mixed', errors='raise')
    df['VSD'] = pd.to_datetime(df['VSD'], format='mixed', errors='raise')
    
    df['Order_day'] = df['Order date'].dt.day
    df['Order_month'] = df['Order date'].dt.month
    df['Order_year'] = df['Order date'].dt.year
    
    df['VSD_day'] = df['VSD'].dt.day
    df['VSD_month'] = df['VSD'].dt.month
    df['VSD_year'] = df['VSD'].dt.year
    
    df['Day_range'] = (df['VSD'] - df['Order date']).dt.days
    
    df = df.drop(columns=['Order date', 'VSD'])
    
    # Handle missing value
    numerical_cols = df.select_dtypes(include=['number']).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].mean()) #impute mean

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:  
        df[col] = df[col].fillna('missing') #impute missing
        
    #df[categorical_cols] =  df[categorical_cols].astype('category')

    return df


train = preprocessing(full_data)
test = preprocessing(pilot)

cat_cols = train.select_dtypes(include='object').columns
train[cat_cols] = train[cat_cols].astype('category')
test[cat_cols] = test[cat_cols].astype('category')

X_train = train.drop(columns= 'label')
y_train = train['label']
test = test.drop(columns= 'ID')
print(X_train.shape, test.shape)


LGBM = LGBMClassifier()
LGBM.fit(X_train, y_train)


result = LGBM.predict(test)
print(Counter(result))


output = pd.DataFrame(result)
output.insert(0, 'ID', range(1, len(output) + 1))


output.to_csv('submission.csv', index=False)

