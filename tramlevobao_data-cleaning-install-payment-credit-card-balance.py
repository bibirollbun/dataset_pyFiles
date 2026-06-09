!pip install optbinning


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import numpy as np
import gc
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import math
import pickle
import os

from collections import Counter
from scipy.sparse import hstack
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from datetime import datetime



from optbinning import BinningProcess

# set col width
pd.set_option('display.max_colwidth', 100)

# set plot style
sns.set_style('whitegrid')

# set palette
sns.set_palette('Set1')


def reduce_memory_usage(df):
  
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def one_hot_encode(df):
    original_columns = list(df.columns)
    categories = [cat for cat in df.columns if df[cat].dtype == 'object']
    df = pd.get_dummies(df, columns= categories, dummy_na= True) #one_hot_encode the categorical features
    categorical_columns = [cat for cat in df.columns if cat not in original_columns]
    return df, categorical_columns


train_data = reduce_memory_usage(pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv'))
print('Number of data points : ', train_data.shape[0])
print('Number of features : ', train_data.shape[1])
train_data.head()


df = train_data[["SK_ID_CURR", "TARGET"]]


installments_payments = reduce_memory_usage(pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv'))
print('Number of data points : ', installments_payments.shape[0])
print('Number of features : ', installments_payments.shape[1])
installments_payments.head()


installments_payments.columns


def FE_installments_payments(installments_payments):
    
    pay1 = installments_payments[['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']+ ['AMT_PAYMENT']]
    pay2 = pay1.groupby(['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'])['AMT_PAYMENT'].sum().reset_index()
    pay_final = pay2.rename(columns={'AMT_PAYMENT': 'AMT_PAYMENT_GROUPED'})
    payments_final = installments_payments.merge(pay_final,\
                            on=['SK_ID_PREV','NUM_INSTALMENT_NUMBER'], how='left')

    payments_final['PAYMENT_DIFFERENCE'] = payments_final['AMT_INSTALMENT'] - \
                                       payments_final['AMT_PAYMENT_GROUPED']
    payments_final['PAYMENT_RATIO'] = payments_final['AMT_INSTALMENT'] / payments_final['AMT_PAYMENT_GROUPED']

    payments_final['PAID_OVER_AMOUNT'] = payments_final['AMT_PAYMENT'] - \
                                     payments_final['AMT_INSTALMENT']
    payments_final['PAID_OVER'] = (payments_final['PAID_OVER_AMOUNT'] > 0).astype(int)
   
    payments_final['DPD'] = payments_final['DAYS_ENTRY_PAYMENT'] - \
                        payments_final['DAYS_INSTALMENT']
    payments_final['DPD'] = payments_final['DPD'].apply(lambda x: 0 if x <= 0 else x)

    payments_final['DBD'] = payments_final['DAYS_INSTALMENT'] - \
                        payments_final['DAYS_ENTRY_PAYMENT']
    payments_final['DBD'] = payments_final['DBD'].apply(lambda x: 0 if x <= 0 else x)
    payments_final['LATE_PAYMENT'] = payments_final['DBD'].apply(lambda x: 1 if x > 0 else 0)
    
    payments_final['INSTALMENT_PAYMENT_RATIO'] = payments_final['AMT_PAYMENT'] / payments_final['AMT_INSTALMENT']
    payments_final['LATE_PAYMENT_RATIO'] = payments_final.apply(lambda x: x['INSTALMENT_PAYMENT_RATIO'] if x['LATE_PAYMENT'] == 1 else 0, axis=1)

    payments_final['SIGNIFICANT_LATE_PAYMENT'] = payments_final['LATE_PAYMENT_RATIO'].apply(lambda x: 1 if x > 0.05 else 0)

    payments_final['DPD_7'] = payments_final['DPD'].apply(lambda x: 1 if x >= 7 else 0)
    payments_final['DPD_15'] = payments_final['DPD'].apply(lambda x: 1 if x >= 15 else 0)
    payments_final['DPD_30'] = payments_final['DPD'].apply(lambda x: 1 if x >= 30 else 0)
    payments_final['DPD_60'] = payments_final['DPD'].apply(lambda x: 1 if x >= 60 else 0)
    payments_final['DPD_90'] = payments_final['DPD'].apply(lambda x: 1 if x >= 90 else 0)
    payments_final['DPD_180'] = payments_final['DPD'].apply(lambda x: 1 if x >= 180 else 0)
    payments_final['DPD_WOF'] = payments_final['DPD'].apply(lambda x: 1 if x >= 720 else 0)
    
    payments_final, pay_final_columns = one_hot_encode(payments_final)

    numeric_agg_payments = {'LATE_PAYMENT': ['max','mean','min'],'AMT_PAYMENT': ['min', 'max',\
                      'mean', 'sum'], 'NUM_INSTALMENT_VERSION': ['nunique'], \
                      'NUM_INSTALMENT_NUMBER':['max'], 'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'PAYMENT_DIFFERENCE': ['max','mean','min','sum'],'DAYS_ENTRY_PAYMENT': ['max', \
        'mean', 'sum'],  'PAID_OVER_AMOUNT': ['max','mean','min']
               }

    for col in pay_final_columns:
        numeric_agg_payments[col] = ['mean']
    
    payments_final_agg = payments_final.groupby('SK_ID_CURR').agg(numeric_agg_payments)
    col_list_9=[]

    for col in payments_final_agg.columns.tolist():
        col_list_9.append('INS_'+col[0]+'_'+col[1].upper())

    payments_final_agg.columns = pd.Index(col_list_9)
    payments_final_agg['INSTALLATION_COUNT'] = payments_final.groupby('SK_ID_CURR').size()
    
    del payments_final
    gc.collect()
    
    return payments_final_agg


payments_final = FE_installments_payments(reduce_memory_usage(installments_payments))


payments_final


payments_final = df.merge(payments_final, on='SK_ID_CURR', how='left')


payments_final


# Bar plot number of missing values column ascending 
cols_with_null = payments_final.isnull().mean() 
cols_with_null 


num_cols = payments_final.select_dtypes(include=np.number).columns.tolist()
cat_cols = payments_final.select_dtypes(include='object').columns.tolist()

cols_to_plot = payments_final[num_cols].var()[payments_final[num_cols].var() > 0.1].index.tolist()
print(f'Plotting {len(cols_to_plot)} columns: {cols_to_plot}')
ncol, nrow = 3, len(cols_to_plot)//3 + 1


# Plot correlation matrix
corr = payments_final[num_cols].corr()
plt.figure(figsize=(20, 15))
sns.heatmap(corr, cmap='coolwarm')
plt.show()


# Filter out highly correlated features
high_corr_cols = []
for i in range(len(corr)):
    for j in range(i+1, len(corr)):
        if 1 > abs(corr.iloc[i, j]) > 0.9:
            high_corr_cols.append(corr.columns[j])
            print(f'{corr.columns[i]}, {corr.columns[j]}: {corr.iloc[i, j]}')


len(high_corr_cols)


# Plot feature importance by IV and WoE
optb = BinningProcess(variable_names=payments_final.columns.to_list())
optb.fit(payments_final, payments_final['TARGET'])
optb.summary()


summary = optb.summary().set_index('name')
summary.sort_values(by='iv', ascending=False, inplace=True)
summary.head(20)


top10 = summary.copy()

# Plot feature importance by IV and WoE
plt.figure(figsize=(2, len(top10)//5))
sns.barplot(x='iv', y=top10.index, data=top10)
plt.title('Top 10 features by IV')
plt.show()


# Prepare data for modelling
# Union cols with null and highly correlated cols
var_0_cols = payments_final.select_dtypes('number').var()[payments_final.select_dtypes('number').var() < 1e-3].index.tolist()
cols_with_null = payments_final[payments_final.columns[payments_final.isnull().mean() > 0.6]].columns.tolist()
cols_to_drop = var_0_cols + high_corr_cols + cols_with_null
cols_to_drop = set(cols_to_drop)
len(cols_to_drop)


# Drop cols
payments_final.drop(cols_to_drop, axis=1, inplace=True)
payments_final.shape


percent_null = payments_final.isnull().mean() * 100
print(percent_null)


payments_final.to_csv('installments_payments_df.csv')


credit_card_balance = reduce_memory_usage(pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv'))
print('Number of data points : ', credit_card_balance.shape[0])
print('Number of features : ', credit_card_balance.shape[1])
credit_card_balance.head()


credit_card_balance.columns


credit_card_balance.dtypes


def FE_credit_card_balance(credit_card_balance):
    
    cc_balance_data, cc_balance_columns = one_hot_encode(credit_card_balance)
    cc_balance_data.rename(columns={'AMT_RECIVABLE': 'AMT_RECEIVABLE'}, inplace=True)

    cc_balance_data['LIMIT_USE'] = cc_balance_data['AMT_BALANCE'] / cc_balance_data['AMT_CREDIT_LIMIT_ACTUAL']
    cc_balance_data['PAYMENT_DIV_MIN'] = cc_balance_data['AMT_PAYMENT_CURRENT'] / cc_balance_data['AMT_INST_MIN_REGULARITY']
    cc_balance_data['LATE_PAYMENT'] = cc_balance_data['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    
    cc_balance_data['DRAWING_LIMIT_RATIO'] = cc_balance_data['AMT_DRAWINGS_ATM_CURRENT'] / cc_balance_data['AMT_CREDIT_LIMIT_ACTUAL']

    cc_balance_data.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_balance_data_agg = cc_balance_data.groupby('SK_ID_CURR').agg(['sum'])
    
    col_list_9=[]

    for col in cc_balance_data_agg.columns.tolist():
        col_list_9.append('CR_'+col[0]+'_'+col[1].upper())
    
    cc_balance_data_agg.columns = pd.Index(col_list_9)

    cc_balance_data_agg['CREDIT_COUNT'] = cc_balance_data.groupby('SK_ID_CURR').size()
    
    del cc_balance_data, cc_balance_columns
    gc.collect()
    
    return cc_balance_data_agg


credit_card_balance = FE_credit_card_balance(credit_card_balance)


credit_card_balance


credit_card_balance = df.merge(credit_card_balance, on='SK_ID_CURR', how='left')


credit_card_balance


# Bar plot number of missing values column ascending 
cols_with_null = credit_card_balance.isnull().mean() 
cols_with_null 


num_cols = credit_card_balance.select_dtypes(include=np.number).columns.tolist()
cat_cols = credit_card_balance.select_dtypes(include='object').columns.tolist()

cols_to_plot = credit_card_balance[num_cols].var()[credit_card_balance[num_cols].var() > 0.1].index.tolist()
print(f'Plotting {len(cols_to_plot)} columns: {cols_to_plot}')
ncol, nrow = 3, len(cols_to_plot)//3 + 1


# Plot correlation matrix
corr = credit_card_balance[num_cols].corr()
plt.figure(figsize=(20, 15))
sns.heatmap(corr, cmap='coolwarm')
plt.show()


# Filter out highly correlated features
high_corr_cols = []
for i in range(len(corr)):
    for j in range(i+1, len(corr)):
        if 1 > abs(corr.iloc[i, j]) > 0.9:
            high_corr_cols.append(corr.columns[j])
            print(f'{corr.columns[i]}, {corr.columns[j]}: {corr.iloc[i, j]}')


# Prepare data for modelling
# Union cols with null and highly correlated cols
var_0_cols = credit_card_balance.select_dtypes('number').var()[credit_card_balance.select_dtypes('number').var() < 1e-3].index.tolist()
cols_with_null = credit_card_balance[credit_card_balance.columns[credit_card_balance.isnull().mean() > 0.6]].columns.tolist()
cols_to_drop = var_0_cols + high_corr_cols + cols_with_null
cols_to_drop = set(cols_to_drop)
len(cols_to_drop)


# Drop cols
credit_card_balance.drop(cols_to_drop, axis=1, inplace=True)
credit_card_balance.shape




