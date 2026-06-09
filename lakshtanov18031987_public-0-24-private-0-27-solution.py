import numpy as np
import pandas as pd
import os
import zipfile
import glob
import seaborn as sns
from matplotlib import pyplot as plt
import catboost
from catboost import CatBoostClassifier
from sklearn import model_selection
from sklearn import metrics
import random
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV


#Read train and test data
file_pattern = '/kaggle/input/neo-bank-non-sub-churn-prediction/train_*.parquet' 

all_files = glob.glob(file_pattern)
df = pd.concat([pd.read_parquet(file) for file in all_files], ignore_index=True)

df_test = pd.read_parquet('/kaggle/input/neo-bank-non-sub-churn-prediction/test.parquet')


#Bring test data columns order to the one in train; concat train and test
df['Usage'] = 'Train' #to separate train from test

col_order = list(df_test.columns)[0:25]+list(df_test.columns)[26:27]+list(df_test.columns)[25:26]
df_test = df_test[col_order]

df.columns == df_test.columns

df = pd.concat([df,df_test],axis = 0,ignore_index = True)


#change int&float 64 to 32 bit 
d1 = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
d2 = dict.fromkeys(df.select_dtypes(np.float64).columns, np.float32)

df = df.astype(d1)
df = df.astype(d2)


# add age feature
df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
#df['age_at_tnx_date'] = ((df['date'] - df['date_of_birth'])/np.timedelta64(1,'Y')).astype('float16')
df['age_at_tnx_date'] = ((df['date'] - df['date_of_birth'])/np.timedelta64(1,'D')/365.25).astype('float16')

# add csat features
df['csat_appointment'] = df['csat_scores'].apply(lambda x: x['appointment']).astype('float16')
df['csat_email'] = df['csat_scores'].apply(lambda x: x['email']).astype('float16')
df['csat_phone'] = df['csat_scores'].apply(lambda x: x['phone']).astype('float16')
df['csat_whatsapp'] = df['csat_scores'].apply(lambda x: x['whatsapp']).astype('float16')

df['csat_agg'] = df[['csat_appointment','csat_email','csat_phone','csat_whatsapp']].mean(axis = 1,skipna=True).astype('float16')

# add last csat responce by customer
df['csat_appointment_acc'] = (df.sort_values(by=['customer_id','date'], 
                                             ascending=[True,True])).groupby("customer_id")['csat_appointment'].ffill().astype('float16').fillna(-1)
df['csat_email_acc'] = (df.sort_values(by=['customer_id','date'], 
                                             ascending=[True,True])).groupby("customer_id")['csat_email'].ffill().astype('float16').fillna(-1)
df['csat_phone_acc'] = (df.sort_values(by=['customer_id','date'], 
                                             ascending=[True,True])).groupby("customer_id")['csat_phone'].ffill().astype('float16').fillna(-1)
df['csat_whatsapp_acc'] = (df.sort_values(by=['customer_id','date'], 
                                             ascending=[True,True])).groupby("customer_id")['csat_whatsapp'].ffill().astype('float16').fillna(-1)
df['csat_agg_acc'] = (df.sort_values(by=['customer_id','date'], 
                                             ascending=[True,True])).groupby("customer_id")['csat_agg'].ffill().astype('float16').fillna(-1)
#drop initial columns
df.drop(['csat_appointment','csat_email','csat_phone','csat_whatsapp','csat_agg'], axis = 1, inplace = True)


# Calc tenure, period from last & till next action
df['tenure_man'] = (df['date'] - df.groupby("customer_id")['date'].transform('min')).dt.days.astype('int32')
df['days_since_la'] = (df['tenure_man'] - (df.sort_values(by=['customer_id','date'], 
                                              ascending=[True,True])).groupby("customer_id")['tenure_man'].shift(1)).astype('float32').fillna(0)
df['days_till_na'] = df.sort_values(by=['customer_id','date'], 
                                    ascending=[True,True]).groupby("customer_id")['tenure_man'].shift(-1) - df['tenure_man']

df['days_till_na'] = df['days_till_na'].astype('float32')


#add churn (target) variable
days_till_churn = [500]
def add_churn_var(df, days_till_churn):
    for value in days_till_churn:
         df['churn'+str(value)] = ((df['days_till_na'] > value)*1).astype('int16').fillna(1)
    return df
# def add_churn_var_2(df, days_till_churn):
#     max_date = df[df['Usage'] == 'Train']['date'].max()
#     for value in days_till_churn:
#         date_thresh = max_date - pd.to_timedelta(value, unit='d')
#         df['churn'+str(value)+'_2'] = np.nan
#         df.loc[~df['days_till_na'].isna(),'churn'+str(value)+'_2'] = (df.loc[~df['days_till_na'].isna(),'days_till_na'] > value)*1
#         df.loc[(df['days_till_na'].isna())&(df['date'] <= date_thresh),'churn'+str(value)] = 1
# #        df.loc[(df['Usage'] == 'Train')&(df['churn_due_to_fraud'] == True),'churn'+str(value)] = 1
#         df['churn'+str(value)] = df['churn'+str(value)].astype('float32')
#     return df

df = add_churn_var(df, days_till_churn)
#df = add_churn_var_2(df, days_till_churn)


#add date&trend features
df['date_block_num'] = (df['date'] - df['date'].min()).dt.days.astype('int32')
#df['month_block_num'] = ((df['date'] - df['date'].min())/np.timedelta64(1,'M')).astype(int)
df['month_block_num'] = ((df['date'] - df['date'].min())/np.timedelta64(1,'D')/30.5).astype('int32')
#df['year_block_num'] = ((df['date'] - df['date'].min())/np.timedelta64(1,'Y')).astype(int)
df['year_block_num'] = ((df['date'] - df['date'].min())/np.timedelta64(1,'D')/365.25).astype('int32')

df['date_year'] = df['date'].dt.year
df['date_month'] = df['date'].dt.month
df['date_day'] = df['date'].dt.day

df['acq_date_year'] = df.groupby("customer_id")['date'].transform('min').dt.year
df['acq_date_month'] = df.groupby("customer_id")['date'].transform('min').dt.month


#add net tnx cnt&volume features
df['atm_transfer_net'] = df['atm_transfer_in'] - df['atm_transfer_out']
df['bank_transfer_net'] = df['bank_transfer_in'] - df['bank_transfer_out']
df['crypto_net'] = df['crypto_in'] - df['crypto_out']
df['bank_transfer_net_volume'] = df['bank_transfer_in_volume'] - df['bank_transfer_out_volume']
df['crypto_net_volume'] = df['crypto_in_volume'] - df['crypto_out_volume']

# add chronological running totals at tnx date by customer
value_cols = [
    'atm_transfer_net',
    'bank_transfer_net',
    'crypto_net',
    'bank_transfer_net_volume',
    'crypto_net_volume'
]

for col in value_cols:
    df[str(col)+'_cs_at_td'] = (df.sort_values(by=['customer_id','date'], 
                                             ascending=[True,True]).groupby("customer_id")[col].cumsum() - df[col]).astype('float32')
df['num_days_cs_at_td'] = df.sort_values(by=['customer_id','date'], 
                                             ascending=[True,True]).groupby("customer_id")['date'].cumcount().astype('int32') 
df['days_s_la_cm_at_td'] = df.sort_values(by=['customer_id','date'], 
                                             ascending=[True,True]).groupby("customer_id")['days_since_la'].cummax().astype('int32')


# cerate lag features
period_days = [10,30,90,360,500]

def create_lag_features(df, value_cols, period_days, func):
    for col in value_cols:
        vals_to_deduct = pd.Series()
        for i in range(len(period_days)):
            if i == 0:
#                print(col, period_days[i])
                roll_vals = df.sort_values(by=['customer_id','date'], 
                                           ascending=[True,True]).groupby('customer_id').rolling(str(period_days[i])+'D', 
                                                                                                 on='date',
                                                                                                 closed = 'left')[col].agg(func).fillna(0).astype('float32').rename(str(col)+str(period_days[i])+str(func))
                df = df.merge(roll_vals, how = 'left', left_on=['customer_id','date'], right_index=True )
                vals_to_deduct = roll_vals.copy()
            else:
 #               print(col, period_days[i])
                roll_vals = df.sort_values(by=['customer_id','date'], 
                                            ascending=[True,True]).groupby("customer_id").rolling(str(period_days[i])+'D', 
                                                                                                      on='date',
                                                                                                      closed = 'left')[col].agg(func).fillna(0).astype('float32')
                vals_to_fil = roll_vals - vals_to_deduct
                vals_to_fil.rename(str(col)+str(period_days[i])+'-'+str(period_days[i-1])+str(func), inplace = True)
                df = df.merge(vals_to_fil, how = 'left', left_on=['customer_id','date'], right_index=True )
                vals_to_deduct = roll_vals.copy()
    return df

def create_lag_features_2(df, value_cols, period_days, func):
    for col in value_cols:
        for i in range(len(period_days)):
#                print(col, period_days[i])
                roll_vals = df.sort_values(by=['customer_id','date'], 
                                           ascending=[True,True]).groupby('customer_id').rolling(str(period_days[i])+'D', 
                                                                                                 on='date',
                                                                                                 closed = 'left')[col].agg(func).fillna(0).astype('float32').rename(str(col)+str(period_days[i])+str(func))
                df = df.merge(roll_vals, how = 'left', left_on=['customer_id','date'], right_index=True )

    return df




%%time
df = create_lag_features(df, value_cols, period_days,'sum')

df.drop(value_cols, axis = 1, inplace = True)


%%time
df = create_lag_features(df, ['date'], period_days, 'count')


%%time
df = create_lag_features_2(df, ['days_since_la'], period_days, 'mean')


%%time
df = create_lag_features_2(df, ['days_since_la'], period_days, 'max')


df['avg_days_betw_cs_at_td'] = ((df.sort_values(by=['date'], 
                                             ascending=[True]).groupby("customer_id")['days_since_la'].transform('cumsum') - df['days_since_la']) \
                            / \
                                (df.sort_values(by=['date'], 
                                             ascending=[True]).groupby("customer_id")['days_since_la'].transform('cumcount')-1)).astype('float32').fillna(0)


df = df.copy()


#grouping features for modeling and removing unnecessary ones
cols_to_drop = [
#                'Id',
#                'customer_id',
                'name',
                'date_of_birth',
                'address',
#                'date',
                'touchpoints',
                'csat_scores',
                'tenure',
#                'Usage',
#                'test_not_train_cid',
#                'days_till_na'
                ]
cols_to_drop2 = [
                 'Id',
                 'customer_id',
                 'date','Usage',
                 'days_till_na',
                 'atm_transfer_in'
                ]
cat_feat = ['country',
            'job',
            'from_competitor',
            'churn_due_to_fraud',
            'model_predicted_fraud']

y_feat = ['churn500']

df.drop(cols_to_drop, axis = 1, inplace = True)


# cat_classifier2 = CatBoostClassifier(n_estimators = 500,
# #                                  reg_lambda = 12, 
#                                     eta = 0.3,
#                                   loss_function = 'Logloss', 
#                                   eval_metric = 'Logloss', 
#                                   early_stopping_rounds = 10,
#                                   auto_class_weights = 'Balanced',
# #                                    random_seed = 4567
# #                                    'Balanced'
#                                    )

# X_train, X_test, y_train, y_test = train_test_split(df.drop(cols_to_drop2+y_feat,axis = 1)[(df['Usage'] == 'Train')&(~df['days_till_na'].isna())], 
#                                                     df[(df['Usage'] == 'Train')&(~df['days_till_na'].isna())]['churn500'],
#                                                     stratify = df[(df['Usage'] == 'Train')&(~df['days_till_na'].isna())]['churn500'], 
#                                                     test_size=0.2,
#                                                     random_state= 3456)

# cat_classifier2.fit(X = X_train, 
#                    y = y_train,
#                    cat_features = cat_feat,
#                    eval_set = (X_test,
#                                y_test),
#                    verbose = 10)


# Modeling. As classes are highly unbalanced it is reasonable to give a greater weight to the minor class.
cat_classifier = CatBoostClassifier(
                                    n_estimators = 350,
#                                    reg_lambda = 12, 
                                    eta = 0.05,
                                    loss_function = 'Logloss', 
                                    eval_metric = 'Logloss', 
                                    early_stopping_rounds = 10,
                                    class_weights = {0: 1, 1: 7},
                                   )
# Better result is achieved using time-series based validation approach, rather than train test split. Train data dated earlier 2022-01-01 is for training, rest of the train data is for validation.
random.seed(4567)
cat_classifier.fit(X = df.drop(cols_to_drop2+y_feat,axis = 1)[(df['Usage'] == 'Train')&(df['date']<pd.to_datetime('2022-01-01'))], 
                   y = df[(df['Usage'] == 'Train')&(df['date']<pd.to_datetime('2022-01-01'))]['churn500'],
                   cat_features = cat_feat,
                   eval_set = (df.drop(cols_to_drop2+y_feat,axis = 1)[(df['Usage'] == 'Train')&(df['date']>=pd.to_datetime('2022-01-01'))],
                               df[(df['Usage'] == 'Train')&(df['date']>=pd.to_datetime('2022-01-01'))]['churn500']),
                   verbose = 10)


# creating submission file
df['y_pred'] = df['churn500'].astype('float16')
df.loc[(df['days_till_na'].isna())&(df['Usage'] != 'Train'),'y_pred'] = cat_classifier.predict_proba(df.drop(cols_to_drop2+y_feat+['y_pred'],axis = 1)[(df['days_till_na'].isna())&(df['Usage'] != 'Train')])[:,1]
subm_df = df[df['Usage']!='Train'][['Id', 'y_pred']]
subm_df.rename({'y_pred':'proba'}).to_csv('/kaggle/working/submission.csv', index=False)
df.drop('y_pred',axis = 1,inplace = True)




