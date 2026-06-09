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


previous_application = reduce_memory_usage(pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv'))
print('Number of data points : ', previous_application.shape[0])
print('Number of features : ', previous_application.shape[1])
previous_application.head()


def FE_previous_application(previous_application):
    
    prev_app, previous_application_columns = one_hot_encode(previous_application)
    
    prev_app['APPLICATION_CREDIT_DIFF'] = prev_app['AMT_APPLICATION'] - prev_app['AMT_CREDIT']
    prev_app['APPLICATION_CREDIT_RATIO'] = prev_app['AMT_APPLICATION'] / prev_app['AMT_CREDIT']
    prev_app['CREDIT_TO_ANNUITY_RATIO'] = prev_app['AMT_CREDIT']/prev_app['AMT_ANNUITY']
    prev_app['DOWN_PAYMENT_TO_CREDIT'] = prev_app['AMT_DOWN_PAYMENT'] / prev_app['AMT_CREDIT']

    total_payment = prev_app['AMT_ANNUITY'] * prev_app['CNT_PAYMENT']
    prev_app['SIMPLE_INTERESTS'] = (total_payment/prev_app['AMT_CREDIT'] - 1)/prev_app['CNT_PAYMENT']

    prev_app['DAYS_LAST_DUE_DIFF'] = prev_app['DAYS_LAST_DUE_1ST_VERSION'] - prev_app['DAYS_LAST_DUE']

    numerical_agg_prev = {'AMT_ANNUITY': ['max', 'mean'], 'AMT_APPLICATION': ['max','mean'],\
                     'AMT_CREDIT':['max','mean'], 'AMT_DOWN_PAYMENT': ['max','mean'],\
                      'AMT_GOODS_PRICE':['mean','sum'], 'HOUR_APPR_PROCESS_START' :\
                      ['max','mean'], 'RATE_DOWN_PAYMENT':['max','mean'], 'RATE_INTEREST_PRIMARY':\
                      ['max','mean'],'RATE_INTEREST_PRIVILEGED':['max','mean'], \
                      'DAYS_DECISION': ['max','mean'], 'CNT_PAYMENT' :['mean','sum'], \
                      'DAYS_FIRST_DRAWING': ['max','mean'], 'DAYS_TERMINATION' : ['max','mean'],\
                      'APPLICATION_CREDIT_RATIO': ['max','mean'], 'DOWN_PAYMENT_TO_CREDIT' : \
                      ['max','mean'], 'DAYS_LAST_DUE_DIFF': ['max','mean']}

    categorical_agg_prev = {}
    
    for column in previous_application_columns:
        categorical_agg_prev[column] = ['mean']
    
    prev_app_agg1 = prev_app.groupby('SK_ID_CURR').agg({**numerical_agg_prev, \
                                                    **categorical_agg_prev})
    col_list_5 =[]
    
    for col in prev_app_agg1.columns.tolist():
        col_list_5.append('PREV_'+col[0]+'_'+col[1].upper())

    prev_app_agg1.columns = pd.Index(col_list_5)
    
    prev_app_cs_approved = prev_app[prev_app['NAME_CONTRACT_STATUS_Approved']==1]
    prev_app_agg2 = prev_app_cs_approved.groupby('SK_ID_CURR').agg(numerical_agg_prev)

    col_list_6 = []

    for col in prev_app_agg2.columns.tolist():
        col_list_6.append('CS_APP_' + col[0] + '_' + col[1].upper())
    
    prev_app_agg2.columns = pd.Index(col_list_6)
    
    prev_app_agg1_join = prev_app_agg1.join(prev_app_agg2, how='left', on='SK_ID_CURR')

    prev_app_cs_refused = prev_app[prev_app['NAME_CONTRACT_STATUS_Refused']==1]
    prev_app_agg3 = prev_app_cs_refused.groupby('SK_ID_CURR').agg(numerical_agg_prev)
    
    col_list_7 =[]

    for col in prev_app_agg3.columns.tolist():
        col_list_7.append('CS_REF_' + col[0] + '_' + col[1].upper())

    prev_app_agg3.columns = pd.Index(col_list_7)
    prev_app_agg_final = prev_app_agg1_join.join(prev_app_agg3,how='left', on='SK_ID_CURR')
    
    del prev_app_agg1_join, prev_app_agg3, prev_app_cs_refused, prev_app_agg1, prev_app_agg2,\
        prev_app_cs_approved
    gc.collect()
    return prev_app_agg_final


previous_application = FE_previous_application(reduce_memory_usage(previous_application))


previous_application


previous_application = df.merge(previous_application, on='SK_ID_CURR', how='left')


# Bar plot number of missing values column ascending 
cols_with_null = previous_application.isnull().mean() 
cols_with_null 


num_cols = previous_application.select_dtypes(include=np.number).columns.tolist()
cat_cols = previous_application.select_dtypes(include='object').columns.tolist()


# Plot correlation matrix
corr = previous_application[num_cols].corr()
plt.figure(figsize=(20, 15))
sns.heatmap(corr, cmap='coolwarm')
plt.show()


# Filter out highly correlated features
high_corr_cols = []
for i in range(len(corr)):
    for j in range(i+1, len(corr)):
        if 1 > abs(corr.iloc[i, j]) > 0.8:
            high_corr_cols.append(corr.columns[j])
            print(f'{corr.columns[i]}, {corr.columns[j]}: {corr.iloc[i, j]}')


len(high_corr_cols)


# Prepare data for modelling
# Union cols with null and highly correlated cols
var_0_cols = previous_application.select_dtypes('number').var()[previous_application.select_dtypes('number').var() < 1e-3].index.tolist()
cols_with_null = previous_application[previous_application.columns[previous_application.isnull().mean() > 0.6]].columns.tolist()
cols_to_drop = var_0_cols + high_corr_cols + cols_with_null
cols_to_drop = set(cols_to_drop)
len(cols_to_drop)


# Drop cols
previous_application.drop(cols_to_drop, axis=1, inplace=True)
previous_application.shape


percent_null = previous_application.isnull().mean() * 100
print(percent_null)


previous_application.to_csv("previous_application_df.csv")


pos_cash_balance = reduce_memory_usage(pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv'))
print('Number of data points : ', pos_cash_balance.shape[0])
print('Number of features : ', pos_cash_balance.shape[1])
pos_cash_balance.head()


pos_cash_balance.columns


def FE_pos_cash_balance(pos_cash_balance):
    
    pos_balance_data, pos_balance_columns = one_hot_encode(pos_cash_balance)
    
    pos_balance_data['LATE_PAYMENT'] = pos_balance_data['SK_DPD'].apply(lambda x:1 \
                                                if x>0 else 0)
    numerical_agg_pos_balance = {'SK_DPD_DEF': ['max', 'mean','min'],'SK_DPD': ['max', 'mean','min'],
        'MONTHS_BALANCE': ['max', 'mean', 'size'], 'CNT_INSTALMENT': ['max','size'],
        'CNT_INSTALMENT_FUTURE': ['max','size','sum']}

    categorical_agg_pos_balance = {}

    for col in pos_balance_columns:
        categorical_agg_pos_balance[col] = ['mean']

    pos_balance_agg = pos_balance_data.groupby('SK_ID_CURR').agg({**numerical_agg_pos_balance, \
                                                    **categorical_agg_pos_balance})
    col_list_8=[]
    for col in pos_balance_agg.columns.tolist():
        col_list_8.append('POS_'+col[0] + '_' + col[1].upper())

    pos_balance_agg.columns = pd.Index(col_list_8)

    sort_pos_balance = pos_balance_data.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'])
    pos_group = sort_pos_balance.groupby('SK_ID_PREV')
    
    pos_final_df = pd.DataFrame()
    pos_final_df['SK_ID_CURR'] = pos_group['SK_ID_CURR'].first()
    pos_final_df['MONTHS_BALANCE_MAX'] = pos_group['MONTHS_BALANCE'].max()
    
    pos_final_df['POS_LOAN_COMPLETED_MEAN'] = pos_group['NAME_CONTRACT_STATUS_Completed'].mean()
    pos_final_df['POS_COMPLETED_BEFORE_MEAN'] = pos_group['CNT_INSTALMENT'].first() - \
                                            pos_group['CNT_INSTALMENT'].last()
    pos_final_df['POS_COMPLETED_BEFORE_MEAN'] = pos_final_df.apply(lambda x: 1 if x['POS_COMPLETED_BEFORE_MEAN'] > 0
                                                and x['POS_LOAN_COMPLETED_MEAN'] > 0 else 0, axis=1)
    
    pos_final_df['POS_REMAINING_INSTALMENTS'] = pos_group['CNT_INSTALMENT_FUTURE'].last()
    pos_final_df['POS_REMAINING_INSTALMENTS_RATIO'] = pos_group['CNT_INSTALMENT_FUTURE'].last()/pos_group['CNT_INSTALMENT'].last()
    
    pos_final_df_groupby = pos_final_df.groupby('SK_ID_CURR').sum().reset_index()
    pos_final_df_groupby.drop(['MONTHS_BALANCE_MAX'], axis=1, inplace= True)
    pos_final_agg = pd.merge(pos_balance_agg, pos_final_df_groupby, on= 'SK_ID_CURR',\
                         how= 'left')
    
    del pos_balance_agg, pos_final_df_groupby, pos_group, sort_pos_balance
    gc.collect()
    return pos_final_agg


pos_cash_balance = FE_pos_cash_balance(reduce_memory_usage(pos_cash_balance))


pos_cash_balance


# Bar plot number of missing values column ascending 
cols_with_null = pos_cash_balance.isnull().mean() 
cols_with_null 


num_cols = pos_cash_balance.select_dtypes(include=np.number).columns.tolist()
cat_cols = pos_cash_balance.select_dtypes(include='object').columns.tolist()


# Plot correlation matrix
corr = pos_cash_balance[num_cols].corr()
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
var_0_cols = pos_cash_balance.select_dtypes('number').var()[pos_cash_balance.select_dtypes('number').var() < 1e-3].index.tolist()
cols_with_null = pos_cash_balance[pos_cash_balance.columns[pos_cash_balance.isnull().mean() > 0.6]].columns.tolist()
cols_to_drop = var_0_cols + high_corr_cols + cols_with_null
cols_to_drop = set(cols_to_drop)
len(cols_to_drop)


percent_null = pos_cash_balance.isnull().mean() * 100
print(percent_null)


pos_cash_balance.to_csv('pos_cash_balance_df.csv')

