import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
from itertools import product
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn import metrics
import lightgbm as lgb

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)


# import the data
DATA_FOLDER = '/kaggle/input/competitive-data-science-predict-future-sales'

sales           = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv'))
items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
shops           = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))
test            = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'))


sales['date'] = pd.to_datetime(sales['date'], format = '%d.%m.%Y')


# exclude shops not in test
sales = sales[sales['shop_id'].isin(test['shop_id'].unique())]


# remove outliers 
sales = sales[ (sales['item_price'] < 100000 )]
sales = sales[ (sales['item_cnt_day'] < 1001)]
sales = sales[ (sales['item_price'] > 0 )]
sales = sales[ (sales['item_cnt_day'] > 0)]


# fixing ids of the same (presumably) shops

# Якутск Орджоникидзе, 56
sales.loc[sales.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
sales.loc[sales.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
sales.loc[sales.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11

sales.loc[sales.shop_id == 39, 'shop_id'] = 40
test.loc[test.shop_id == 39, 'shop_id'] = 40


sales['revenue'] = sales['item_price']*sales['item_cnt_day']


# cerate test-like train - add rows for all shops&items&periods product. For combinations not in original data fill 0

def create_testlike_train(df):
    matrix = []
    min_date = df['date'].min()
    for i in range(df['date_block_num'].min(), df['date_block_num'].max()+1):
        shops = df[df['date_block_num'] == i]['shop_id'].unique()
        items = df[df['date_block_num'] == i]['item_id'].unique()
        month_start = min_date + pd.tseries.offsets.DateOffset(months = i)
        matrix.append( np.array( list(product([i],[month_start],shops,items))))
    df_new = pd.DataFrame(np.vstack(matrix),columns = ['date_block_num','month_start','shop_id','item_id'])
    pivot = pd.pivot_table(df, 
                            values = ['item_cnt_day','revenue'], 
                            index = ['date_block_num','shop_id','item_id'], 
                            aggfunc = 'sum').reset_index()
    pivot2 = pd.pivot_table(df[df['item_cnt_day']>0], 
                            values = ['item_cnt_day'], 
                            index = ['date_block_num','shop_id','item_id'], 
                            aggfunc = 'count').reset_index()
    pivot2.rename(columns={'item_cnt_day': 'purch_cnt_month'}, inplace=True)
    
    df_new = df_new.merge(right = pivot, how = 'left', on = ['date_block_num','shop_id','item_id'], sort = False)
    df_new = df_new.merge(right = pivot2, how = 'left', on = ['date_block_num','shop_id','item_id'], sort = False)
    
    df_new.rename(columns={'item_cnt_day': 'item_cnt_month_uncl'}, inplace=True)
    df_new['item_cnt_month_uncl'] = df_new['item_cnt_month_uncl'].fillna(0)
    df_new['item_cnt_month'] = df_new['item_cnt_month_uncl'].clip(0,20)
    df_new['revenue'] = df_new['revenue'].fillna(0)
    df_new['purch_cnt_month'] = df_new['purch_cnt_month'].fillna(0)
    df_new['ID'] = -1
    return df_new


%%time
df = create_testlike_train(sales)


test['date_block_num'] = df['date_block_num'].max()+1
test['month_start'] = df['month_start'].max() + pd.tseries.offsets.DateOffset(months = 1)
test['item_cnt_month'] = 0
test['item_cnt_month_uncl'] = 0
test['revenue'] = 0
test['purch_cnt_month'] = 0

#test['sh_it_key'] = test['shop_id'].astype(str) + ['-']*len(test['shop_id']) + test['item_id'].astype(str)
#train_key = list(set(sales['shop_id'].astype(str) + ['-']*len(sales['shop_id']) + sales['item_id'].astype(str)))
#test['was_in_s_it_sh'] = test['sh_it_key'].apply(lambda x: 1 if x in train_key else 0)
#test.drop('sh_it_key', inplace=True, axis=1)


# concat train and test to a single df
df = df[['ID','date_block_num','month_start','shop_id','item_id','item_cnt_month_uncl','item_cnt_month','purch_cnt_month','revenue']]
test = test[['ID','date_block_num','month_start','shop_id','item_id','item_cnt_month_uncl','item_cnt_month','purch_cnt_month','revenue']]
df = pd.concat([df,test], ignore_index=True, join = 'inner')
#del(test)
df.shape


df['ID'] = df['ID'].astype('int32')
df['date_block_num'] = df['date_block_num'].astype('int8')
df['shop_id'] = df['shop_id'].astype('int8')
df['item_id'] = df['item_id'].astype('int16')
df['item_cnt_month'] = df['item_cnt_month'].astype('float32')
df['item_cnt_month_uncl'] = df['item_cnt_month_uncl'].astype('float32')
df['revenue'] = df['revenue'].astype('float32')
df['purch_cnt_month'] = df['purch_cnt_month'].astype('float32')
#df['was_in_s_it_sh'] = df['was_in_s_it_sh'].astype('int8')


#add months and days in month features
df['month'] = df['month_start'].dt.month.astype('int8')
#df['year'] = df['month_start'].dt.year.astype('int16')
df.drop(['month_start'], axis = 1, inplace = True)

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
df['days_in_m'] = (df['month']-1).map(days).astype('int8')


# add city and type features
shops['shop_city'] = shops['shop_name'].apply(lambda x: x.split()[0])
shops['shop_type'] = shops['shop_name'].apply(lambda x: x.split()[1])


# add item categories features
item_categories['split'] = item_categories['item_category_name'].str.split('-')
item_categories['item_category_type'] = item_categories['split'].map(lambda x: x[0].strip())
item_categories['item_category_subtype'] = item_categories['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
item_categories.drop('split', axis = 1, inplace = True)

df = df.merge(items, 
              how='left', 
              on='item_id').merge(item_categories, 
                                  how ='left', 
                                  on='item_category_id').merge(shops, how = 'left', on='shop_id')

df['item_category_id'] = df['item_category_id'].astype('int8')


# encode categorical features
features_to_encode=['shop_city',
                    'shop_type',
                    'item_category_type',
                    'item_category_subtype']
def encode_cat_features(df,features_to_encode):
    for feat in features_to_encode:
        df[feat+'_encoded'] = LabelEncoder().fit_transform( df[feat] )
    df.drop(features_to_encode, axis = 1, inplace = True)
    return df

df = encode_cat_features(df,features_to_encode)

df.drop(['item_category_name','shop_name','item_name'], axis = 1, inplace = True)


# add features for months since shop, item, item&shop first and last sale

def add_col_months_from_sh_it_last_s(df):
    vect_months_from_last_s = []
    dict_ = {}
    for ind, row in df.iterrows():
        key = str(row['shop_id']) + ' ' + str(row['item_id'])
        if key not in dict_:
            if row['item_cnt_month'] > 0:
                dict_[key] = row['date_block_num']
                vect_months_from_last_s.append(0)
            else:
                vect_months_from_last_s.append(0)
        else:
            last_b_1 = dict_[key]
            last = row['date_block_num']
            vect_months_from_last_s.append(last-last_b_1)
            dict_[key] = row['date_block_num']
    df['months_from_sh_it_last_s'] = vect_months_from_last_s
    df['months_from_sh_it_last_s'] = df['months_from_sh_it_last_s'].astype('int16')
    return df

def add_col_months_from_it_last_s(df):
    vect_months_from_last_s = []
    dict_ = {}
    dict_2 = {}
    for ind, row in df.iterrows():
        key = str(row['item_id'])
        if key not in dict_:
            if row['item_cnt_month'] > 0:
                dict_[key] = row['date_block_num']
                dict_2[key] = row['date_block_num']
                vect_months_from_last_s.append(0)
            else:
                vect_months_from_last_s.append(0)
        else:
            last_b_1 = dict_[key]
            last = row['date_block_num']
            if last > last_b_1:      
                vect_months_from_last_s.append(last-last_b_1)
                dict_2[key] = last_b_1
                dict_[key] = row['date_block_num']
            elif last == last_b_1:
                last_b_1 = dict_2[key]
                vect_months_from_last_s.append(last-last_b_1)          
    df['months_from_it_last_s'] = vect_months_from_last_s
    df['months_from_it_last_s'] = df['months_from_it_last_s'].astype('int16')
    return df

def add_col_months_since_it_first_s(df):
    pivot = pd.pivot_table(df[df['item_cnt_month']>0], values = 'date_block_num', index = 'item_id', aggfunc = 'min').reset_index()
    pivot.rename(columns={'date_block_num': 'dt_block_it_first_sale'}, inplace=True)
    df =  df.merge(right = pivot, how = 'left', on = 'item_id', sort = False) 
#    df['is_it_first_s'] = (df['date_block_num'] == df['dt_block_it_first_sale']).astype('int8')
#    df['it_had_sales_before'] = (df['date_block_num'] > df['dt_block_it_first_sale']).astype('int8').fillna(0)
#    df['months_since_it_first_s'] = (df['date_block_num'] - df['dt_block_it_first_sale']).fillna(0)
#    df['months_since_it_first_s'] = df['months_since_it_first_s'].astype('int16')
    df['dt_block_it_first_sale'] = df['dt_block_it_first_sale'].fillna(34).astype('int16')
    df['months_since_it_first_s'] = df['date_block_num'] - df['dt_block_it_first_sale']
    df.drop('dt_block_it_first_sale', inplace=True, axis=1)
    return df

def add_col_months_since_sh_it_first_sale(df):
    pivot = pd.pivot_table(df[df['item_cnt_month']>0], values = 'date_block_num', index = ['shop_id','item_id'], aggfunc = 'min').reset_index()
    pivot.rename(columns={'date_block_num': 'dt_block_sh_it_first_sale'}, inplace=True)
    df =  df.merge(right = pivot, how = 'left', on = ['shop_id','item_id'], sort = False)
#    df['is_sh_it_first_s'] = (df['date_block_num'] == df['dt_block_sh_it_first_sale']).astype('int8')
#    df['sh_it_had_sales_before'] = (df['date_block_num'] > df['dt_block_sh_it_first_sale']).astype('int8').fillna(0)
#    df['months_since_sh_it_first_s'] = (df['date_block_num'] - df['dt_block_sh_it_first_sale']).fillna(0)
#    df['months_since_sh_it_first_s'] = df['months_since_sh_it_first_s'].astype('int16')
    df['dt_block_sh_it_first_sale'] = df['dt_block_sh_it_first_sale'].fillna(34).astype('int16')
    df['months_since_sh_it_first_s'] = df['date_block_num'] - df['dt_block_sh_it_first_sale']
    df.drop('dt_block_sh_it_first_sale', inplace=True, axis=1)
    return df

def add_col_months_since_sh_first_s(df):
    pivot = pd.pivot_table(df[df['item_cnt_month']>0], values = 'date_block_num', index = 'shop_id', aggfunc = 'min').reset_index()
    pivot.rename(columns={'date_block_num': 'dt_block_sh_first_sale'}, inplace=True)
    df =  df.merge(right = pivot, how = 'left', on = 'shop_id', sort = False) 
#    df['is_sh_first_s'] = (df['date_block_num'] == df['dt_block_sh_first_sale']).astype('int8')
#    df['sh_had_sales_before'] = (df['date_block_num'] > df['dt_block_sh_first_sale']).astype('int8').fillna(0)
#    df['months_since_sh_first_s'] = (df['date_block_num'] - df['dt_block_sh_first_sale']).fillna(0)
#    df['months_since_sh_first_s'] = df['months_since_sh_first_s'].astype('int16')
    df['dt_block_sh_first_sale'] = df['dt_block_sh_first_sale'].fillna(34).astype('int16')
    df['months_since_sh_first_s'] = df['date_block_num'] - df['dt_block_sh_first_sale']
    df.drop('dt_block_sh_first_sale', inplace=True, axis=1)
    return df


%%time
df = add_col_months_from_sh_it_last_s(df)
df = add_col_months_from_it_last_s(df)


%%time
df = add_col_months_since_it_first_s(df)
df = add_col_months_since_sh_it_first_sale(df)
df = add_col_months_since_sh_first_s(df)


# add mean encoded features
def add_mean_encoded_feat(df):
    pivot_it = pd.pivot_table(df, values = ['item_cnt_month',], index = ['item_id','date_block_num'], aggfunc = ['sum','count']).reset_index()
    pivot_it.columns = ['item_id','date_block_num','item_cnt_month_sum','item_cnt_month_cnt']
    pivot_it['lagged_it_mean'] = ((pivot_it.groupby(['item_id'])['item_cnt_month_sum'].cumsum() - pivot_it['item_cnt_month_sum'])/(pivot_it.groupby(['item_id'])['item_cnt_month_cnt'].cumsum() - pivot_it['item_cnt_month_cnt'])).fillna(0)
    pivot_it.drop(['item_cnt_month_sum','item_cnt_month_cnt'], axis = 1, inplace = True)
    df =  df.merge(right = pivot_it, how = 'left', on = ['item_id','date_block_num'], sort = False)
    
    pivot_sh_it = pd.pivot_table(df, values = 'item_cnt_month', index = ['shop_id','item_id','date_block_num'], aggfunc = 'sum').reset_index()
    pivot_sh_it['lagged_sh_it_mean'] = ((pivot_sh_it.groupby(['shop_id','item_id'])['item_cnt_month'].cumsum() - pivot_sh_it['item_cnt_month'])/(pivot_sh_it.groupby(['shop_id','item_id'])['item_cnt_month'].cumcount())).fillna(0)
    pivot_sh_it.drop(['item_cnt_month'], axis = 1, inplace = True)
    df =  df.merge(right = pivot_sh_it, how = 'left', on = ['shop_id','item_id','date_block_num'], sort = False)
    df['lagged_it_mean'] = df['lagged_it_mean'].astype('float32')
    df['lagged_sh_it_mean'] = df['lagged_sh_it_mean'].astype('float32')
    return df


%%time
df = add_mean_encoded_feat(df)


#add lag features
def add_lag_feat(df, col_to_agg, group_levels, n_lags, aggfunc = 'mean', clip = False):
    new_col_title_code = '_'.join([x for x in group_levels if x != 'date_block_num'])
    pivot = pd.pivot_table(df, values = col_to_agg, index = group_levels, aggfunc = aggfunc).reset_index()        
    pivot.rename(columns = {col_to_agg[0] : col_to_agg[0] + '_' + aggfunc}, inplace = True)
    idx_cols = ['date_block_num','shop_id','item_id']
    cols = list(set(idx_cols+group_levels))
    df_tech = df[cols].copy()
    df_tech = df_tech.merge(right = pivot, how = 'left', on = group_levels, sort = False)
    list_of_new_col = [] 
    for lag in n_lags:
        df_to_shift = df_tech[idx_cols+[col_to_agg[0] + '_' + aggfunc]].copy()
        df_to_shift['date_block_num'] = df_to_shift['date_block_num'] + lag
        df_to_shift.rename(columns={col_to_agg[0] + '_' + aggfunc : col_to_agg[0]+'_'+new_col_title_code + '_' + aggfunc+ '_lag_'+str(lag)}, inplace=True)
        list_of_new_col.append(col_to_agg[0]+'_'+new_col_title_code + '_' + aggfunc+ '_lag_'+str(lag))
        df= df.merge(right = df_to_shift, how = 'left', on = idx_cols, sort = False)
    for col in list_of_new_col:
        df[col] = df[col].fillna(0).astype('float32')
        if clip:
            df[col] = df[col].clip(0,20)
    return df


%%time
df = add_lag_feat(df, 
                  col_to_agg = ['item_cnt_month'],
                  group_levels = ['date_block_num','shop_id','item_id'], 
                  n_lags = [1,2,3], 
                  aggfunc = 'sum')

df = add_lag_feat(df, 
                  col_to_agg = ['purch_cnt_month'],
                  group_levels = ['date_block_num','shop_id','item_id'], 
                  n_lags = [1,2], 
                  aggfunc = 'sum')


df['it_cnt_sh_it_lag_avg'] = df[['item_cnt_month_shop_id_item_id_sum_lag_1', 
                                 'item_cnt_month_shop_id_item_id_sum_lag_2', 
                                 'item_cnt_month_shop_id_item_id_sum_lag_3']].mean(skipna=True, axis=1)

df['it_cnt_sh_it_lag_grad'] = df['item_cnt_month_shop_id_item_id_sum_lag_1']/df['item_cnt_month_shop_id_item_id_sum_lag_2']

df['it_cnt_sh_it_lag_avg'] = df['it_cnt_sh_it_lag_avg'].astype('float32')
df['it_cnt_sh_it_lag_grad'] = df['it_cnt_sh_it_lag_grad'].replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')


%%time
df = add_lag_feat(df, 
                  col_to_agg = ['item_cnt_month'],
                  group_levels = ['date_block_num'], 
                  n_lags = [1])

df = add_lag_feat(df, 
                  col_to_agg = ['item_cnt_month'],
                  group_levels = ['date_block_num','item_id'], 
                  n_lags = [1,2])
                   
df = add_lag_feat(df, 
                  col_to_agg = ['item_cnt_month'],
                  group_levels = ['date_block_num','shop_id'], 
                  n_lags = [1,2])
# df = add_lag_feat(df, 
#                   col_to_agg = ['revenue'],
#                   group_levels = ['date_block_num','shop_id'], 
#                   n_lags = [1])
df = add_lag_feat(df, 
                  col_to_agg = ['item_cnt_month'],
                  group_levels = ['date_block_num','item_category_id'], 
                  n_lags = [1])
df = add_lag_feat(df, 
                  col_to_agg = ['item_cnt_month'],
                  group_levels = ['date_block_num','shop_id','item_category_id'], 
                  n_lags = [1])
df = add_lag_feat(df, 
                  col_to_agg = ['item_cnt_month'],
                  group_levels = ['date_block_num','shop_id','item_category_type_encoded'], 
                  n_lags = [1])
df = add_lag_feat(df, 
                  col_to_agg = ['item_cnt_month'],
                  group_levels = ['date_block_num','shop_id','item_category_subtype_encoded'], 
                  n_lags = [1])
# df = add_lag_feat(df, 
#                   col_to_agg = ['item_cnt_month'],
#                   group_levels = ['date_block_num','shop_city_encoded'], 
#                   n_lags = [1])
df = add_lag_feat(df, 
                  col_to_agg = ['item_cnt_month'],
                  group_levels = ['date_block_num','shop_city_encoded','item_id'], 
                  n_lags = [1])
# df = add_lag_feat(df, 
#                   col_to_agg = ['item_cnt_month'],
#                   group_levels = ['date_block_num','shop_city_encoded','item_category_id'], 
#                   n_lags = [1])
# df = add_lag_feat(df, 
#                   col_to_agg = ['item_cnt_month'],
#                   group_levels = ['date_block_num','shop_city_encoded','item_category_type_encoded'], 
#                   n_lags = [1])
# df = add_lag_feat(df, 
#                   col_to_agg = ['item_cnt_month'],
#                   group_levels = ['date_block_num','shop_city_encoded','item_category_subtype_encoded'], 
#                   n_lags = [1])
# df = add_lag_feat(df, 
#                   col_to_agg = ['item_cnt_month'],
#                   group_levels = ['date_block_num','shop_type_encoded'], 
#                   n_lags = [1])
df = add_lag_feat(df, 
                  col_to_agg = ['item_cnt_month'],
                  group_levels = ['date_block_num','shop_type_encoded','item_id'], 
                  n_lags = [1])
# df = add_lag_feat(df, 
#                   col_to_agg = ['item_cnt_month'],
#                   group_levels = ['date_block_num','shop_type_encoded','item_category_subtype_encoded'], 
#                   n_lags = [1])
df = add_lag_feat(df, 
                  col_to_agg = ['item_cnt_month'],
                  group_levels = ['date_block_num','item_category_type_encoded'], 
                  n_lags = [1])
df = add_lag_feat(df, 
                  col_to_agg = ['item_cnt_month'],
                  group_levels = ['date_block_num','item_category_subtype_encoded'], 
                  n_lags = [1])


# add price lag features
def add_price_lag_feat(df):
    pivot_global = pd.pivot_table(df[(df['item_cnt_month_uncl']>0)|(df['date_block_num']==34)], 
                                  values = ['item_cnt_month_uncl','revenue'], 
                                  index = ['item_id'], 
                                  aggfunc = 'sum').reset_index()
    pivot_global['avg_price_global'] = pivot_global['revenue']/pivot_global['item_cnt_month_uncl']
    pivot_global.drop(['item_cnt_month_uncl','revenue'],axis = 1,inplace = True)
    
    pivot_avg_mnth = pd.pivot_table(df[(df['item_cnt_month_uncl']>0)|(df['date_block_num']==34)], 
                                    values = ['item_cnt_month_uncl','revenue'], 
                                    index = ['item_id','date_block_num'], 
                                    aggfunc = 'sum').reset_index()
    pivot_avg_mnth['item_id'] = pivot_avg_mnth['item_id'].astype('int16')
    pivot_avg_mnth['date_block_num'] = pivot_avg_mnth['date_block_num'].astype('int8')
    pivot_avg_mnth['avg_price_mnth'] = pivot_avg_mnth['revenue']/pivot_avg_mnth['item_cnt_month_uncl']
    pivot_avg_mnth['avg_price_mnth_lag1'] = pivot_avg_mnth.groupby(['item_id'])['avg_price_mnth'].shift(1)
    pivot_avg_mnth['avg_price_mnth_lag2'] = pivot_avg_mnth.groupby(['item_id'])['avg_price_mnth'].shift(2)
    pivot_avg_mnth.drop(['item_cnt_month_uncl','revenue','avg_price_mnth'],axis = 1,inplace = True)
    
    pivot_avg_mnth_sh = pd.pivot_table(df[(df['item_cnt_month_uncl']>0)|(df['date_block_num']==34)], 
                                       values = ['item_cnt_month_uncl','revenue'], 
                                        index = ['shop_id','item_id','date_block_num'], 
                                       aggfunc = 'sum').reset_index()
    pivot_avg_mnth_sh['shop_id'] = pivot_avg_mnth_sh['shop_id'].astype('int8')
    pivot_avg_mnth_sh['item_id'] = pivot_avg_mnth_sh['item_id'].astype('int16')
    pivot_avg_mnth_sh['date_block_num'] = pivot_avg_mnth_sh['date_block_num'].astype('int8')
    pivot_avg_mnth_sh['avg_price_mnth_sh'] = pivot_avg_mnth_sh['revenue']/pivot_avg_mnth_sh['item_cnt_month_uncl']
    pivot_avg_mnth_sh['avg_price_mnth_sh_lag1'] = pivot_avg_mnth_sh.groupby(['shop_id','item_id'])['avg_price_mnth_sh'].shift(1)
    pivot_avg_mnth_sh['avg_price_mnth_sh_lag2'] = pivot_avg_mnth_sh.groupby(['shop_id','item_id'])['avg_price_mnth_sh'].shift(2)
    pivot_avg_mnth_sh.drop(['item_cnt_month_uncl','revenue','avg_price_mnth_sh'],axis = 1,inplace = True)
    
    pivot_avg_mnth.sort_values(by = 'date_block_num', inplace = True)
    pivot_avg_mnth_sh.sort_values(by = 'date_block_num', inplace = True)
    
    df = df.merge(right = pivot_global, how = 'left', on = ['item_id'], sort = False)
    df = pd.merge_asof(left = df, 
                       right = pivot_avg_mnth, 
                       on = ['date_block_num'], 
                       by = ['item_id'], 
                       allow_exact_matches = True, 
                       direction ='backward')
    df = pd.merge_asof(left = df, 
                       right = pivot_avg_mnth_sh, 
                       on = ['date_block_num'], 
                       by = ['shop_id','item_id'], 
                       allow_exact_matches = True, 
                       direction ='backward')
    
    df['avg_price_global'] = df['avg_price_global'].replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
    df['avg_price_mnth_lag1'] = df['avg_price_mnth_lag1'].replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
    df['avg_price_mnth_lag2'] = df['avg_price_mnth_lag2'].replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
    df['avg_price_mnth_sh_lag1'] = df['avg_price_mnth_sh_lag1'].replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
    df['avg_price_mnth_sh_lag2'] = df['avg_price_mnth_sh_lag2'].replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
    
    df['avg_price_mnth_grad'] = (df['avg_price_mnth_lag1']/df['avg_price_mnth_lag2']).replace([np.inf, -np.inf], 
                                                                                             np.nan).fillna(0).astype('float32')
    df['avg_price_mnth_to_gl'] = (df['avg_price_mnth_lag1']/df['avg_price_global']).replace([np.inf, -np.inf], 
                                                                                             np.nan).fillna(0).astype('float32')
    df['avg_price_mnth_sh_grad'] = (df['avg_price_mnth_sh_lag1']/df['avg_price_mnth_sh_lag2']).replace([np.inf, -np.inf], 
                                                                                             np.nan).fillna(0).astype('float32')
    df['avg_price_mnth_sh_to_gl'] = (df['avg_price_mnth_sh_lag1']/df['avg_price_global']).replace([np.inf, -np.inf], 
                                                                                             np.nan).fillna(0).astype('float32')
#     df['avg_price_mnth_sh_to_gl_2'] = (df['avg_price_mnth_sh_lag2']/df['avg_price_global']).replace([np.inf, -np.inf], 
#                                                                                              np.nan).fillna(0).astype('float32')
    df['avg_price_mnth_sh_to_mnth'] = (df['avg_price_mnth_sh_lag1']/df['avg_price_mnth_lag1']).replace([np.inf, -np.inf], 
                                                                                             np.nan).fillna(0).astype('float32')
#     df['avg_price_mnth_sh_to_mnth_2'] = (df['avg_price_mnth_sh_lag2']/df['avg_price_mnth_lag2']).replace([np.inf, -np.inf], 
#                                                                                              np.nan).fillna(0).astype('float32')
#    df.drop(['avg_price_global','avg_price_mnth_lag2','avg_price_mnth_sh_lag2'], axis = 1, inplace = True)
    return df


%%time
df = add_price_lag_feat(df)


#df.info()


#df.isna().sum().sum()


df = df[(df['month']!=12)&(df['month']!=1)]


columns_to_exclude = ['ID',
                      'item_cnt_month',
                      'item_cnt_month_uncl',
                      'revenue',
                      'purch_cnt_month',
                      'months_since_sh_it_first_s',
                      'months_since_it_first_s',
                      'months_since_sh_first_s', 
                      'avg_price_global', 
#                      'avg_price_mnth_lag1', 
#                      'avg_price_mnth_lag2',
#                      'avg_price_mnth_sh_lag1', 
#                      'avg_price_mnth_sh_lag2'
#                      'avg_price_mnth_to_gl', 
#                      'avg_price_mnth_sh_to_gl', 
#                      'months_from_sh_it_last_s',
#                      'months_from_it_last_s',
                      'lagged_sh_it_mean',
                      'lagged_it_mean',
#                      'it_had_sales_before'
#                      'sh_it_had_sales_before'
                     ]
cat_features = ['month',
#                'year',
                'shop_id',
                'shop_city_encoded',
                'shop_type_encoded',
                'item_category_id',
                'item_category_type_encoded',
                'item_category_subtype_encoded',
                'days_in_m'
               ]


# modeling
params = {'metric': 'rmse',
          'objective': 'mse',
          'num_leaves': 255,
          'learning_rate': 0.005,
          'feature_fraction': 0.75,
          'bagging_fraction': 0.75,
          'bagging_freq': 5,
          'force_col_wise' : True,
          'random_state': 10}

lgb_model = lgb.train(params=params,
                      train_set=(lgb.Dataset(df[(df['date_block_num']>=19)&(df['date_block_num']<33)].drop(columns_to_exclude, axis = 1),
                                             df[(df['date_block_num']>=19)&(df['date_block_num']<33)]['item_cnt_month'])),
                      num_boost_round=1500,
                      valid_sets=((lgb.Dataset(df[(df['date_block_num']>=19)&(df['date_block_num']<33)].drop(columns_to_exclude, axis = 1),
                                               df[(df['date_block_num']>=19)&(df['date_block_num']<33)]['item_cnt_month'])),
                                  (lgb.Dataset(df[(df['date_block_num'] == 33)].drop(columns_to_exclude,axis = 1),
                                                df[(df['date_block_num'] == 33)]['item_cnt_month']))),
                      callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)],
                      categorical_feature = cat_features)


# create a submition df 
def creagte_submition(df,model):
    df.loc[df['date_block_num'] == 34,'item_cnt_month'] = model.predict(df[df['date_block_num'] == 34].drop(columns_to_exclude, axis = 1))
    submition_df = df[df['date_block_num'] == 34][['ID','item_cnt_month']]
    submition_df['item_cnt_month'] = submition_df['item_cnt_month'].clip(0,20) 
    return submition_df


creagte_submition(df,lgb_model).to_csv('/kaggle/working/subm.csv', index = False)




