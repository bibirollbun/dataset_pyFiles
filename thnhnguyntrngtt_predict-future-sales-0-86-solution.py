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


# import time
# import psutil # Thư viện để kiểm tra tài nguyên hệ thống
# import os     # Thư viện để tương tác với hệ điều hành (lấy process ID)

# # modeling
# params = {'metric': 'rmse',
#           'objective': 'mse', # Hoặc 'regression' cho bài toán hồi quy
#           'num_leaves': 255,
#           'learning_rate': 0.005,
#           'feature_fraction': 0.75,
#           'bagging_fraction': 0.75,
#           'bagging_freq': 5,
#           'force_col_wise' : True,
#           'random_state': 10,
#           'verbosity': -1 # Giảm bớt log không cần thiết, vì đã có log_evaluation
#          }

# # Lấy process ID hiện tại của tiến trình Python này
# pid = os.getpid()
# process = psutil.Process(pid)

# # --- Ghi lại thông tin bộ nhớ TRƯỚC KHI tạo Dataset và huấn luyện ---
# # Lưu ý: Việc tạo lgb.Dataset cũng sẽ tiêu tốn bộ nhớ.
# # Nếu bạn muốn đo lường chính xác hơn bộ nhớ chỉ riêng cho hàm lgb.train,
# # bạn có thể di chuyển phần đo mem_before_train xuống sau khi tạo train_data và valid_data.
# # Tuy nhiên, để đơn giản, chúng ta đo tổng bộ nhớ tăng thêm cho cả việc chuẩn bị data và training.

# mem_info_before_dataload = process.memory_info()
# rss_before_dataload_MB = mem_info_before_dataload.rss / (1024 * 1024) # Chuyển byte sang MB
# print(f"Bộ nhớ RSS của tiến trình trước khi tải dữ liệu vào lgb.Dataset: {rss_before_dataload_MB:.2f} MB")


# # Prepare training and validation datasets with categorical features
# # Giả định các biến df, columns_to_exclude, và cat_features đã được định nghĩa ở các cell trước
# # và cat_features chỉ chứa tên các cột có trong dữ liệu sau khi drop columns_to_exclude.
# print("Đang chuẩn bị lgb.Dataset...")
# train_data = lgb.Dataset(
#     df[(df['date_block_num'] >= 19) & (df['date_block_num'] < 33)].drop(columns_to_exclude, axis=1),
#     label=df[(df['date_block_num'] >= 19) & (df['date_block_num'] < 33)]['item_cnt_month'],
#     categorical_feature=cat_features # Đảm bảo các cột trong cat_features tồn tại sau khi drop
# )

# valid_data = lgb.Dataset(
#     df[df['date_block_num'] == 33].drop(columns_to_exclude, axis=1),
#     label=df[df['date_block_num'] == 33]['item_cnt_month'],
#     categorical_feature=cat_features, # Đảm bảo các cột trong cat_features tồn tại sau khi drop
#     reference=train_data
# )
# print("Chuẩn bị lgb.Dataset hoàn tất.")

# # --- Ghi lại thông tin bộ nhớ TRƯỚC KHI huấn luyện (sau khi tạo Dataset) ---
# mem_info_before_train = process.memory_info()
# rss_before_train_MB = mem_info_before_train.rss / (1024 * 1024)
# print(f"Bộ nhớ RSS của tiến trình trước khi huấn luyện: {rss_before_train_MB:.2f} MB")


# print("Bắt đầu huấn luyện mô hình LightGBM...")
# start_time = time.time() # Ghi lại thời điểm bắt đầu

# # Train model
# lgb_model = lgb.train(
#     params=params,
#     train_set=train_data,
#     num_boost_round=1500,
#     valid_sets=[train_data, valid_data],
#     callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True), lgb.log_evaluation(100)]
# )

# end_time = time.time() # Ghi lại thời điểm kết thúc
# training_time = end_time - start_time # Tính toán thời gian huấn luyện

# # --- Ghi lại thông tin bộ nhớ SAU KHI huấn luyện ---
# mem_info_after_train = process.memory_info()
# rss_after_train_MB = mem_info_after_train.rss / (1024 * 1024)


# print(f"Hoàn tất huấn luyện mô hình.")
# print(f"Thời gian huấn luyện: {training_time:.2f} giây")

# print(f"Bộ nhớ RSS của tiến trình sau khi huấn luyện: {rss_after_train_MB:.2f} MB")

# # Tính toán lượng bộ nhớ sử dụng thêm cho quá trình huấn luyện (và mô hình được tạo ra)
# memory_used_for_training_MB = rss_after_train_MB - rss_before_train_MB
# print(f"Lượng bộ nhớ RSS sử dụng thêm trong quá trình huấn luyện (và lưu mô hình): {memory_used_for_training_MB:.2f} MB")

# # Tính toán tổng lượng bộ nhớ sử dụng thêm từ lúc trước khi load data vào Dataset
# total_memory_increase_MB = rss_after_train_MB - rss_before_dataload_MB
# print(f"Tổng lượng bộ nhớ RSS tăng thêm (bao gồm tải data và huấn luyện): {total_memory_increase_MB:.2f} MB")


# create a submition df 
def creagte_submition(df,model):
    df.loc[df['date_block_num'] == 34,'item_cnt_month'] = model.predict(df[df['date_block_num'] == 34].drop(columns_to_exclude, axis = 1))
    submition_df = df[df['date_block_num'] == 34][['ID','item_cnt_month']]
    submition_df['item_cnt_month'] = submition_df['item_cnt_month'].clip(0,20) 
    return submition_df


creagte_submition(df,lgb_model).to_csv('/kaggle/working/subm.csv', index = False)


# import xgboost as xgb

# # (Giữ nguyên phần code chuẩn bị df của bạn)

# # --- Bắt đầu phần code cho XGBoost ---

# # Loại bỏ các cột không cần thiết (tương tự như với LightGBM)
# # columns_to_exclude đã được định nghĩa ở cell [36] trong notebook của bạn
# # cat_features cũng đã được định nghĩa, XGBoost có thể xử lý các feature dạng số
# # tuy nhiên, để đơn giản, ta có thể không cần chỉ định rõ categorical_feature cho XGBoost như LightGBM
# # nếu các feature đó đã được LabelEncoded.

# X_train = df[(df['date_block_num'] >= 19) & (df['date_block_num'] < 33)].drop(columns_to_exclude, axis=1)
# Y_train = df[(df['date_block_num'] >= 19) & (df['date_block_num'] < 33)]['item_cnt_month']
# X_valid = df[df['date_block_num'] == 33].drop(columns_to_exclude, axis=1)
# Y_valid = df[df['date_block_num'] == 33]['item_cnt_month']
# X_test = df[df['date_block_num'] == 34].drop(columns_to_exclude, axis=1)

# # Định nghĩa tham số cho XGBoost
# xgb_params = {
#     'objective': 'reg:squarederror', # Hàm mục tiêu cho bài toán hồi quy
#     'eval_metric': 'rmse',           # Thước đo đánh giá
#     'eta': 0.03,                     # Tốc độ học (learning_rate)
#     'max_depth': 10,                 # Độ sâu tối đa của cây
#     'subsample': 0.8,                # Tỷ lệ mẫu dùng để huấn luyện mỗi cây
#     'colsample_bytree': 0.8,         # Tỷ lệ feature dùng để huấn luyện mỗi cây
#     'seed': 42                       # Random seed để có thể tái tạo kết quả
# }

# # Tạo DMatrix cho XGBoost (định dạng dữ liệu tối ưu cho XGBoost)
# dtrain = xgb.DMatrix(X_train, label=Y_train)
# dvalid = xgb.DMatrix(X_valid, label=Y_valid)
# dtest = xgb.DMatrix(X_test)

# # Huấn luyện mô hình XGBoost
# num_boost_round = 1000 # Số lượng cây ( vòng lặp boosting)
# early_stopping_rounds = 50 # Dừng sớm nếu không cải thiện sau số vòng này

# xgb_model = xgb.train(
#     xgb_params,
#     dtrain,
#     num_boost_round=num_boost_round,
#     evals=[(dtrain, 'train'), (dvalid, 'valid')],
#     early_stopping_rounds=early_stopping_rounds,
#     verbose_eval=100 # In ra kết quả mỗi 100 vòng
# )

# # Dự đoán trên tập test
# Y_test_xgb_pred = xgb_model.predict(dtest).clip(0, 20)

# # Tạo dataframe submission cho XGBoost (tương tự hàm creagte_submition của bạn)
# submission_df_xgb = pd.DataFrame({
#     "ID": df[df['date_block_num'] == 34]['ID'],
#     "item_cnt_month": Y_test_xgb_pred
# })

# # Lưu file submission
# # submission_df_xgb.to_csv('/kaggle/working/submission_xgb.csv', index=False)
# print("Đã tạo xong file submission cho XGBoost (chưa lưu).")
# print(submission_df_xgb.head())


# import xgboost as xgb
# import pandas as pd # Assuming pandas is used for df
# import numpy as np  # Assuming numpy might be used indirectly
# import time
# import psutil
# import os

# # --- Get process information for memory measurement ---
# pid = os.getpid()
# process = psutil.Process(pid)

# # --- (Assume df, columns_to_exclude are already defined from your notebook) ---
# # Example placeholders if not defined in this specific cell run,
# # ensure these are correctly populated from your notebook's context.
# # if 'df' not in locals():
# #     print("Warning: 'df' not defined. Please ensure it's loaded and preprocessed.")
# #     # df = pd.DataFrame() # Placeholder
# # if 'columns_to_exclude' not in locals():
# #     print("Warning: 'columns_to_exclude' not defined.")
# #     # columns_to_exclude = [] # Placeholder

# # --- Memory before data preparation for XGBoost ---
# mem_info_before_xgb_data = process.memory_info()
# rss_before_xgb_data_MB = mem_info_before_xgb_data.rss / (1024 * 1024)
# print(f"Bộ nhớ RSS của tiến trình trước khi chuẩn bị dữ liệu XGBoost: {rss_before_xgb_data_MB:.2f} MB")

# # --- Data preparation for XGBoost ---
# print("Đang chuẩn bị dữ liệu X_train, Y_train, X_valid, Y_valid, X_test...")
# X_train = df[(df['date_block_num'] >= 19) & (df['date_block_num'] < 33)].drop(columns_to_exclude, axis=1)
# Y_train = df[(df['date_block_num'] >= 19) & (df['date_block_num'] < 33)]['item_cnt_month']
# X_valid = df[df['date_block_num'] == 33].drop(columns_to_exclude, axis=1)
# Y_valid = df[df['date_block_num'] == 33]['item_cnt_month']
# X_test = df[df['date_block_num'] == 34].drop(columns_to_exclude, axis=1)
# print("Chuẩn bị dữ liệu X, Y hoàn tất.")

# # --- Memory after X, Y data prep, before DMatrix creation ---
# mem_info_before_dmatrix = process.memory_info()
# rss_before_dmatrix_MB = mem_info_before_dmatrix.rss / (1024 * 1024)
# print(f"Bộ nhớ RSS của tiến trình sau khi tạo X/Y, trước khi tạo DMatrix: {rss_before_dmatrix_MB:.2f} MB")
# mem_used_for_xy_prep_MB = rss_before_dmatrix_MB - rss_before_xgb_data_MB
# print(f"Bộ nhớ RSS sử dụng thêm cho việc chuẩn bị X/Y: {mem_used_for_xy_prep_MB:.2f} MB")


# # --- Tạo DMatrix cho XGBoost ---
# print("Đang tạo DMatrix cho XGBoost...")
# # Enable GPUTree explainer for GPU execution if you have a GPU and XGBoost compiled with GPU support
# # X_train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_train.columns]
# # X_valid.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_valid.columns]
# # X_test.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_test.columns]


# dtrain = xgb.DMatrix(X_train, label=Y_train, feature_names=X_train.columns.to_list())
# dvalid = xgb.DMatrix(X_valid, label=Y_valid, feature_names=X_valid.columns.to_list())
# dtest = xgb.DMatrix(X_test, feature_names=X_test.columns.to_list())
# print("Tạo DMatrix hoàn tất.")

# # --- Memory after DMatrix creation, before training ---
# mem_info_after_dmatrix_before_train = process.memory_info()
# rss_after_dmatrix_before_train_MB = mem_info_after_dmatrix_before_train.rss / (1024 * 1024)
# print(f"Bộ nhớ RSS của tiến trình sau khi tạo DMatrix, trước khi huấn luyện: {rss_after_dmatrix_before_train_MB:.2f} MB")
# mem_used_for_dmatrix_MB = rss_after_dmatrix_before_train_MB - rss_before_dmatrix_MB
# print(f"Bộ nhớ RSS sử dụng thêm cho việc tạo DMatrix: {mem_used_for_dmatrix_MB:.2f} MB")


# # --- Định nghĩa tham số và Huấn luyện mô hình XGBoost ---
# xgb_params = {
#     'objective': 'reg:squarederror',
#     'eval_metric': 'rmse',
#     'eta': 0.03,
#     'max_depth': 10,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'seed': 42,
#     # 'tree_method': 'gpu_hist' # Bỏ comment dòng này nếu bạn muốn sử dụng GPU và đã cài đặt XGBoost hỗ trợ GPU
# }

# num_boost_round = 1000
# early_stopping_rounds = 50

# print("Bắt đầu huấn luyện mô hình XGBoost...")
# start_time = time.time()

# xgb_model = xgb.train(
#     xgb_params,
#     dtrain,
#     num_boost_round=num_boost_round,
#     evals=[(dtrain, 'train'), (dvalid, 'valid')],
#     early_stopping_rounds=early_stopping_rounds,
#     verbose_eval=100
# )

# end_time = time.time()
# training_time_xgb = end_time - start_time

# # --- Memory after training ---
# mem_info_after_train_xgb = process.memory_info()
# rss_after_train_xgb_MB = mem_info_after_train_xgb.rss / (1024 * 1024)

# print("Hoàn tất huấn luyện mô hình XGBoost.")
# print(f"Thời gian huấn luyện XGBoost: {training_time_xgb:.2f} giây")

# print(f"Bộ nhớ RSS của tiến trình sau khi huấn luyện XGBoost: {rss_after_train_xgb_MB:.2f} MB")
# memory_used_for_xgb_training_MB = rss_after_train_xgb_MB - rss_after_dmatrix_before_train_MB
# print(f"Lượng bộ nhớ RSS sử dụng thêm trong quá trình huấn luyện XGBoost (và lưu mô hình): {memory_used_for_xgb_training_MB:.2f} MB")

# total_memory_increase_xgb_MB = rss_after_train_xgb_MB - rss_before_xgb_data_MB
# print(f"Tổng lượng bộ nhớ RSS tăng thêm cho XGBoost (bao gồm chuẩn bị data, DMatrix và huấn luyện): {total_memory_increase_xgb_MB:.2f} MB")


# # --- Dự đoán và tạo submission (giữ nguyên như code của bạn) ---
# print("Đang dự đoán trên tập test...")
# Y_test_xgb_pred = xgb_model.predict(dtest).clip(0, 20)
# print("Dự đoán hoàn tất.")

# submission_df_xgb = pd.DataFrame({
#     "ID": df[df['date_block_num'] == 34]['ID'], # Ensure this ID alignment is correct
#     "item_cnt_month": Y_test_xgb_pred
# })

# # submission_df_xgb.to_csv('/kaggle/working/submission_xgb.csv', index=False)
# print("Đã tạo xong file submission cho XGBoost (chưa lưu).")
# print(submission_df_xgb.head())


# import time
# import psutil # Thư viện để kiểm tra tài nguyên hệ thống
# import os     # Thư viện để tương tác với hệ điều hành (lấy process ID)
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from sklearn.preprocessing import MinMaxScaler

# # Giả định df đã được nạp và xử lý ở các bước trước
# # Ví dụ: df = pd.read_csv('your_data.csv')
# # và các cột lag đã được tạo.

# # --- Bắt đầu phần code cho LSTM ---

# # Chọn các features bạn muốn sử dụng cho LSTM.
# features_for_lstm = [
#     'item_cnt_month_shop_id_item_id_sum_lag_1',
#     'item_cnt_month_shop_id_item_id_sum_lag_2',
#     'item_cnt_month_shop_id_item_id_sum_lag_3',
#     'item_cnt_month_item_id_mean_lag_1',
#     'item_cnt_month_shop_id_mean_lag_1',
#     'avg_price_mnth_lag1',
#     # Thêm các features bạn cho là quan trọng khác ở đây
#     # Ví dụ: 'month', 'shop_id_encoded', 'item_id_encoded', 'item_category_id_encoded'
# ]

# # Lấy dữ liệu mục tiêu (item_cnt_month)
# target_lstm = 'item_cnt_month'

# # Tạo một bản sao của dataframe để tránh thay đổi df gốc
# # Giả định df đã được nạp. Nếu chưa, bạn cần nạp nó.
# # Ví dụ:
# # if 'df' not in locals():
# #     print("Biến 'df' chưa được định nghĩa. Vui lòng nạp dữ liệu.")
# #     # df = pd.read_csv('path_to_your_data.csv') # Hoặc cách bạn nạp df
# #     # Sau đó thực hiện các bước tạo features lag nếu cần
# # else:
# #     print("'df' đã được định nghĩa.")
# df_lstm = df.copy()


# # LSTM thường hoạt động tốt hơn với dữ liệu được chuẩn hóa (scale)
# scaler_features = MinMaxScaler(feature_range=(0, 1))
# scaler_target = MinMaxScaler(feature_range=(0, 1))

# # Chuẩn hóa các features đã chọn
# # Lọc bỏ các dòng không có đủ lag hoặc fillna(0)
# # Giả sử các lagged features đã được tạo và fillna(0) ở các bước trước.
# # Và df_lstm chỉ chứa các dòng có date_block_num phù hợp để huấn luyện/kiểm định/test

# # Để ví dụ chạy được, giả sử các cột lag đã được fillna(0) cho các tháng đầu
# for col in features_for_lstm:
#     if col not in df_lstm.columns:
#         print(f"Cảnh báo: Cột '{col}' không tồn tại trong df_lstm. Sẽ tạo cột giả với giá trị 0.")
#         df_lstm[col] = 0 # Tạo cột giả nếu thiếu để code chạy
# if target_lstm not in df_lstm.columns:
#     print(f"Cảnh báo: Cột target '{target_lstm}' không tồn tại. Sẽ tạo cột giả với giá trị 0.")
#     df_lstm[target_lstm] = 0


# print("Lưu ý: Phần chuẩn bị dữ liệu cho LSTM dưới đây là một ví dụ đơn giản hóa.")
# print("Bạn cần điều chỉnh cách tạo sequence cho phù hợp với bài toán dự đoán doanh số theo từng item-shop.")

# train_start_block = 19
# train_end_block = 32 # Dữ liệu huấn luyện đến hết block 32
# valid_block = 33
# test_block = 34 # Dữ liệu test là block 34

# # Lấy dữ liệu dựa trên date_block_num
# # Đảm bảo rằng df_lstm có cột 'date_block_num'
# if 'date_block_num' not in df_lstm.columns:
#     print("Cảnh báo: Cột 'date_block_num' không tồn tại trong df_lstm. Sẽ tạo cột giả.")
#     # Tạo cột giả để code chạy, bạn cần đảm bảo cột này có thật trong dữ liệu của mình
#     df_lstm['date_block_num'] = np.random.randint(0, 35, size=len(df_lstm))


# X_train_lstm_flat = df_lstm[ (df_lstm['date_block_num'] >= train_start_block) & (df_lstm['date_block_num'] <= train_end_block) ][features_for_lstm].values
# Y_train_lstm_flat = df_lstm[ (df_lstm['date_block_num'] >= train_start_block) & (df_lstm['date_block_num'] <= train_end_block) ][target_lstm].values

# X_valid_lstm_flat = df_lstm[df_lstm['date_block_num'] == valid_block][features_for_lstm].values
# Y_valid_lstm_flat = df_lstm[df_lstm['date_block_num'] == valid_block][target_lstm].values

# X_test_lstm_flat = df_lstm[df_lstm['date_block_num'] == test_block][features_for_lstm].values
# # Y_test_lstm_flat (nếu có để đánh giá offline)
# # Y_test_lstm_flat = df_lstm[df_lstm['date_block_num'] == test_block][target_lstm].values


# # Kiểm tra nếu dữ liệu rỗng trước khi scale
# if X_train_lstm_flat.shape[0] == 0:
#     print("Lỗi: X_train_lstm_flat rỗng. Kiểm tra lại điều kiện lọc date_block_num và dữ liệu đầu vào.")
#     # Có thể dừng hoặc xử lý lỗi ở đây
#     # exit()
# else:
#     print(f"Kích thước X_train_lstm_flat: {X_train_lstm_flat.shape}")


# # Chuẩn hóa
# # Fit scaler_features CHỈ trên dữ liệu training
# scaler_features.fit(X_train_lstm_flat)
# X_train_lstm_scaled = scaler_features.transform(X_train_lstm_flat)
# X_valid_lstm_scaled = scaler_features.transform(X_valid_lstm_flat)
# X_test_lstm_scaled = scaler_features.transform(X_test_lstm_flat)

# # Fit scaler_target CHỈ trên dữ liệu training
# scaler_target.fit(Y_train_lstm_flat.reshape(-1, 1))
# Y_train_lstm_scaled = scaler_target.transform(Y_train_lstm_flat.reshape(-1, 1))
# Y_valid_lstm_scaled = scaler_target.transform(Y_valid_lstm_flat.reshape(-1, 1))


# # Reshape dữ liệu cho LSTM [samples, time_steps, features]
# # Giả sử mỗi feature trong features_for_lstm là một feature tại một time_step.
# # Và chúng ta có 1 time_step (tháng hiện tại mà ta đang dùng lags của nó)
# # với `len(features_for_lstm)` features.
# X_train_lstm = X_train_lstm_scaled.reshape((X_train_lstm_scaled.shape[0], 1, X_train_lstm_scaled.shape[1]))
# X_valid_lstm = X_valid_lstm_scaled.reshape((X_valid_lstm_scaled.shape[0], 1, X_valid_lstm_scaled.shape[1]))
# X_test_lstm = X_test_lstm_scaled.reshape((X_test_lstm_scaled.shape[0], 1, X_test_lstm_scaled.shape[1]))

# print(f"Hình dạng X_train_lstm: {X_train_lstm.shape}")
# print(f"Hình dạng Y_train_lstm_scaled: {Y_train_lstm_scaled.shape}")
# if X_valid_lstm.shape[0] > 0:
#     print(f"Hình dạng X_valid_lstm: {X_valid_lstm.shape}")
#     print(f"Hình dạng Y_valid_lstm_scaled: {Y_valid_lstm_scaled.shape}")
# else:
#     print("Cảnh báo: Tập validation rỗng.")


# # 3. Xây dựng mô hình LSTM
# if X_train_lstm.shape[0] > 0: # Chỉ xây dựng mô hình nếu có dữ liệu huấn luyện
#     n_features_lstm = X_train_lstm.shape[2] # Số lượng features mỗi time step

#     lstm_model = Sequential()
#     # input_shape=(time_steps, n_features) -> (1, n_features_lstm)
#     lstm_model.add(LSTM(units=50, activation='relu', input_shape=(X_train_lstm.shape[1], n_features_lstm), return_sequences=True))
#     lstm_model.add(Dropout(0.2))
#     lstm_model.add(LSTM(units=50, activation='relu'))
#     lstm_model.add(Dropout(0.2))
#     lstm_model.add(Dense(units=1)) # Output layer: dự đoán 1 giá trị (item_cnt_month)

#     lstm_model.compile(optimizer='adam', loss='mse', metrics=['RootMeanSquaredError'])
#     lstm_model.summary()

#     # 4. Huấn luyện mô hình LSTM
#     epochs = 20 # Số epochs, có thể cần nhiều hơn
#     batch_size = 256 # Kích thước batch

#     print("Đã định nghĩa mô hình LSTM.")
#     print("LƯU Ý: Huấn luyện LSTM có thể mất nhiều thời gian và tài nguyên")

#     # --- BẮT ĐẦU THEO DÕI TÀI NGUYÊN ---
#     pid = os.getpid()
#     process = psutil.Process(pid)

#     mem_info_before_lstm_train = process.memory_info()
#     rss_before_lstm_train_MB = mem_info_before_lstm_train.rss / (1024 * 1024)
#     print(f"Bộ nhớ RSS của tiến trình TRƯỚC KHI huấn luyện LSTM: {rss_before_lstm_train_MB:.2f} MB")

#     print("Bắt đầu huấn luyện mô hình LSTM...")
#     start_time_lstm = time.time() # Ghi lại thời điểm bắt đầu

#     # Kiểm tra xem tập validation có dữ liệu không
#     validation_data_lstm = None
#     if X_valid_lstm.shape[0] > 0 and Y_valid_lstm_scaled.shape[0] > 0:
#         validation_data_lstm = (X_valid_lstm, Y_valid_lstm_scaled)
#         print("Sử dụng tập validation để đánh giá.")
#     else:
#         print("Cảnh báo: Tập validation rỗng, sẽ không sử dụng để đánh giá trong quá trình fit.")


#     history_lstm = lstm_model.fit(
#         X_train_lstm, Y_train_lstm_scaled,
#         epochs=epochs,
#         batch_size=batch_size,
#         validation_data=validation_data_lstm,
#         verbose=1
#     )

#     end_time_lstm = time.time() # Ghi lại thời điểm kết thúc
#     training_time_lstm = end_time_lstm - start_time_lstm # Tính toán thời gian huấn luyện

#     # --- KẾT THÚC THEO DÕI TÀI NGUYÊN ---
#     mem_info_after_lstm_train = process.memory_info()
#     rss_after_lstm_train_MB = mem_info_after_lstm_train.rss / (1024 * 1024)

#     print("\nHoàn tất huấn luyện mô hình LSTM.")
#     print(f"Thời gian huấn luyện LSTM: {training_time_lstm:.2f} giây ({training_time_lstm/60:.2f} phút)")
#     print(f"Bộ nhớ RSS của tiến trình SAU KHI huấn luyện LSTM: {rss_after_lstm_train_MB:.2f} MB")

#     # Tính toán lượng bộ nhớ sử dụng thêm
#     memory_used_for_lstm_training_MB = rss_after_lstm_train_MB - rss_before_lstm_train_MB
#     print(f"Lượng bộ nhớ RSS sử dụng thêm trong quá trình huấn luyện LSTM: {memory_used_for_lstm_training_MB:.2f} MB")

#     # (Tùy chọn) Dự đoán và đảo ngược scale nếu cần
#     # predictions_scaled = lstm_model.predict(X_test_lstm)
#     # predictions_lstm = scaler_target.inverse_transform(predictions_scaled)
#     # print("\nDự đoán trên tập test (đã đảo ngược scale):")
#     # print(predictions_lstm[:5])

# else:
#     print("Lỗi: Không có dữ liệu huấn luyện (X_train_lstm rỗng). Mô hình LSTM sẽ không được xây dựng hoặc huấn luyện.")







# import time
# import psutil # Thư viện để kiểm tra tài nguyên hệ thống
# import os     # Thư viện để tương tác với hệ điều hành (lấy process ID)
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_squared_error

# # Giả định các biến df, columns_to_exclude, cat_features đã được định nghĩa ở các cell trước.
# # Ví dụ:
# # if 'df' not in locals():
# #     print("Biến 'df' chưa được định nghĩa. Vui lòng nạp và xử lý dữ liệu trước.")
# #     # df = pd.read_csv('your_processed_data.csv') # Hoặc cách bạn có df
# # if 'columns_to_exclude' not in locals():
# #     columns_to_exclude = [] # Khởi tạo nếu chưa có
# # if 'cat_features' not in locals():
# #     cat_features = [] # Khởi tạo nếu chưa có


# # --- Chuẩn bị dữ liệu cho Ridge Regression ---
# print("="*50)
# print("Bắt đầu chuẩn bị cho mô hình Ridge Regression")

# # Mô hình tuyến tính không xử lý trực tiếp categorical features
# # Chúng ta sẽ loại bỏ chúng cùng với các cột khác
# # Đảm bảo columns_to_exclude và cat_features là list
# if not isinstance(columns_to_exclude, list):
#     columns_to_exclude = list(columns_to_exclude) if hasattr(columns_to_exclude, '__iter__') else []
# if not isinstance(cat_features, list):
#     cat_features = list(cat_features) if hasattr(cat_features, '__iter__') else []

# features_to_drop_ridge = columns_to_exclude + cat_features
# # Loại bỏ các cột trùng lặp nếu có
# features_to_drop_ridge = sorted(list(set(features_to_drop_ridge)))


# # Tạo tập dữ liệu train và validation
# # Đảm bảo 'date_block_num' và 'item_cnt_month' tồn tại trong df
# if 'date_block_num' not in df.columns or 'item_cnt_month' not in df.columns:
#     print("Lỗi: df thiếu cột 'date_block_num' hoặc 'item_cnt_month'.")
#     # exit() # Hoặc xử lý lỗi phù hợp

# # Loại bỏ các cột không tồn tại khỏi features_to_drop_ridge để tránh lỗi khi drop
# actual_features_to_drop_ridge = [col for col in features_to_drop_ridge if col in df.columns]


# X_train_ridge = df[(df['date_block_num'] >= 19) & (df['date_block_num'] < 33)].drop(actual_features_to_drop_ridge, axis=1, errors='ignore')
# y_train_ridge = df[(df['date_block_num'] >= 19) & (df['date_block_num'] < 33)]['item_cnt_month']

# X_valid_ridge = df[df['date_block_num'] == 33].drop(actual_features_to_drop_ridge, axis=1, errors='ignore')
# y_valid_ridge = df[df['date_block_num'] == 33]['item_cnt_month']

# # Chuẩn bị dữ liệu test (date_block_num == 34)
# X_test_ridge = df[df['date_block_num'] == 34].drop(actual_features_to_drop_ridge, axis=1, errors='ignore')
# test_ids = df[df['date_block_num'] == 34]['ID'] # Lấy ID cho submission

# print("Chuẩn bị dữ liệu cho Ridge hoàn tất.")
# print(f"Kích thước X_train_ridge: {X_train_ridge.shape}")
# print(f"Kích thước X_valid_ridge: {X_valid_ridge.shape}")
# print(f"Kích thước X_test_ridge: {X_test_ridge.shape}")


# # Lấy process ID và theo dõi bộ nhớ
# pid = os.getpid()
# process = psutil.Process(pid)

# # --- Ghi lại thông tin bộ nhớ TRƯỚC KHI huấn luyện ---
# mem_info_before_ridge = process.memory_info()
# rss_before_ridge_MB = mem_info_before_ridge.rss / (1024 * 1024)
# print(f"Bộ nhớ RSS trước khi huấn luyện Ridge: {rss_before_ridge_MB:.2f} MB")

# # --- Huấn luyện mô hình ---
# print("Bắt đầu huấn luyện mô hình Ridge Regression...")
# start_time_ridge = time.time()

# # Khởi tạo và huấn luyện mô hình
# ridge_model = Ridge(alpha=1.0, random_state=10, solver='auto')
# # Kiểm tra X_train_ridge không rỗng
# if X_train_ridge.empty:
#     print("Lỗi: X_train_ridge rỗng, không thể huấn luyện mô hình.")
#     # exit()
# else:
#     ridge_model.fit(X_train_ridge, y_train_ridge)

#     end_time_ridge = time.time()
#     training_time_ridge = end_time_ridge - start_time_ridge

#     # --- Ghi lại thông tin bộ nhớ SAU KHI huấn luyện ---
#     mem_info_after_ridge = process.memory_info()
#     rss_after_ridge_MB = mem_info_after_ridge.rss / (1024 * 1024)

#     print("Hoàn tất huấn luyện mô hình Ridge.")
#     print(f"Thời gian huấn luyện: {training_time_ridge:.4f} giây")
#     print(f"Bộ nhớ RSS sau khi huấn luyện: {rss_after_ridge_MB:.2f} MB")

#     # Tính toán lượng bộ nhớ sử dụng thêm
#     memory_used_ridge_MB = rss_after_ridge_MB - rss_before_ridge_MB
#     print(f"Lượng bộ nhớ RSS sử dụng thêm cho việc huấn luyện Ridge: {memory_used_ridge_MB:.2f} MB")

#     # Đánh giá mô hình trên tập validation
#     if not X_valid_ridge.empty:
#         preds_valid_ridge = ridge_model.predict(X_valid_ridge).clip(0, 20)
#         rmse_ridge = np.sqrt(mean_squared_error(y_valid_ridge, preds_valid_ridge))
#         print(f"RMSE trên tập validation của Ridge: {rmse_ridge:.4f}")
#     else:
#         print("Cảnh báo: X_valid_ridge rỗng, không thể đánh giá trên tập validation.")
#     print("="*50)

#     # --- Tạo Submission File ---
#     print("\nBắt đầu tạo file submission cho Ridge Regression...")
#     if not X_test_ridge.empty:
#         # Dự đoán trên tập test
#         preds_test_ridge = ridge_model.predict(X_test_ridge).clip(0, 20)

#         # Kiểm tra độ dài của test_ids và preds_test_ridge
#         if len(test_ids) == len(preds_test_ridge):
#             submission_df_ridge = pd.DataFrame({
#                 "ID": test_ids,
#                 "item_cnt_month": preds_test_ridge
#             })
#             submission_df_ridge.to_csv('/kaggle/working/submission_ridge.csv', index=False)
#             print(f"File submission_ridge.csv đã được tạo thành công với {len(submission_df_ridge)} dòng.")
#             print(submission_df_ridge.head())
#         else:
#             print(f"Lỗi: Độ dài của test_ids ({len(test_ids)}) không khớp với độ dài của predictions ({len(preds_test_ridge)}).")
#             print("Không thể tạo file submission.")
#     else:
#         print("Cảnh báo: X_test_ridge rỗng, không thể tạo file submission.")

#     print("="*50)



import time
import psutil
import os
import pandas as pd
from catboost import CatBoostRegressor, Pool

# ==================================================================
# PHẦN MÃ GỐC CỦA BẠN (GIỮ NGUYÊN)
# ==================================================================

# Giả định các biến df, columns_to_exclude, cat_features đã được định nghĩa
# và df đã được tạo ra từ các bước trước đó.
# Ví dụ:
# df = pd.read_pickle('df_processed.pkl')
# columns_to_exclude = ['ID', 'item_cnt_month', 'date_block_num'] 
# # (Thực tế có thể nhiều cột hơn)
# cat_features = ['shop_id', 'item_id', 'item_category_id', 'city_code', 'month'] 
# # (Ví dụ các cột category)


# --- Chuẩn bị dữ liệu cho CatBoost ---
print("\n" + "="*50)
print("Bắt đầu chuẩn bị cho mô hình CatBoost")

# Giả định các biến df, columns_to_exclude, cat_features đã được định nghĩa
X_train_cat = df[(df['date_block_num'] >= 19) & (df['date_block_num'] < 33)].drop(columns_to_exclude, axis=1)
y_train_cat = df[(df['date_block_num'] >= 19) & (df['date_block_num'] < 33)]['item_cnt_month']

X_valid_cat = df[df['date_block_num'] == 33].drop(columns_to_exclude, axis=1)
y_valid_cat = df[df['date_block_num'] == 33]['item_cnt_month']

train_pool = Pool(data=X_train_cat, label=y_train_cat, cat_features=cat_features)
valid_pool = Pool(data=X_valid_cat, label=y_valid_cat, cat_features=cat_features)

print("Chuẩn bị CatBoost Pool hoàn tất.")

pid = os.getpid()
process = psutil.Process(pid)
mem_info_before_cat = process.memory_info()
rss_before_cat_MB = mem_info_before_cat.rss / (1024 * 1024)
print(f"Bộ nhớ RSS trước khi huấn luyện CatBoost: {rss_before_cat_MB:.2f} MB")

# --- Huấn luyện mô hình ---
print("Bắt đầu huấn luyện mô hình CatBoost...")
start_time_cat = time.time()

cat_model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.05,
    eval_metric='RMSE',
    random_seed=10,
    verbose=100,
    early_stopping_rounds=100
)

cat_model.fit(
    train_pool,
    eval_set=valid_pool
)

end_time_cat = time.time()
training_time_cat = end_time_cat - start_time_cat
mem_info_after_cat = process.memory_info()
rss_after_cat_MB = mem_info_after_cat.rss / (1024 * 1024)

print("Hoàn tất huấn luyện mô hình CatBoost.")
print(f"Thời gian huấn luyện: {training_time_cat:.2f} giây")
print(f"Bộ nhớ RSS của tiến trình sau khi huấn luyện: {rss_after_cat_MB:.2f} MB")
memory_used_cat_MB = rss_after_cat_MB - rss_before_cat_MB
print(f"Lượng bộ nhớ RSS sử dụng thêm cho việc huấn luyện CatBoost: {memory_used_cat_MB:.2f} MB")
best_score = cat_model.get_best_score()
print(f"RMSE tốt nhất trên tập validation của CatBoost: {best_score['validation']['RMSE']:.4f}")
print("="*50)


# ==================================================================
# PHẦN BỔ SUNG: DỰ ĐOÁN VÀ TẠO FILE SUBMISSION
# ==================================================================
print("\nBắt đầu quá trình dự đoán và tạo file submission...")

# --- BẮT ĐẦU ĐO LƯỜNG ---
start_time_pred = time.time()
mem_info_before_pred = process.memory_info()
rss_before_pred_MB = mem_info_before_pred.rss / (1024 * 1024)
print(f"Bộ nhớ RSS trước khi dự đoán: {rss_before_pred_MB:.2f} MB")
# -------------------------

# 1. Chuẩn bị dữ liệu test (thường là date_block_num == 34)
# Đảm bảo các cột trong X_test khớp với các cột đã dùng để huấn luyện
X_test = df[df['date_block_num'] == 34].drop(columns_to_exclude, axis=1)

# 2. Thực hiện dự đoán trên tập test
print("Đang dự đoán trên tập test...")
test_predictions = cat_model.predict(X_test)

# 3. Giới hạn giá trị dự đoán trong khoảng [0, 20]
# Đây là bước xử lý phổ biến cho bài toán này
test_predictions_clipped = test_predictions.clip(0, 20)

# 4. Tạo DataFrame cho file submission
# Giả định bạn có dataframe `test_df` gốc chứa cột 'ID'
# Nếu test_df chưa được định nghĩa, bạn cần nạp nó từ file test.csv
# Ví dụ: test_df = pd.read_csv('test.csv')
# Ở đây, chúng ta lấy ID từ phần test của df tổng hợp
test_ids = df[df['date_block_num'] == 34]['ID']
submission_df = pd.DataFrame({
    'ID': test_ids,
    'item_cnt_month': test_predictions_clipped
})

# 5. Lưu file submission
output_path = '/kaggle/working/submission_catboost.csv'
submission_df.to_csv(output_path, index=False)

# --- KẾT THÚC ĐO LƯỜNG VÀ IN KẾT QUẢ ---
end_time_pred = time.time()
prediction_time = end_time_pred - start_time_pred
mem_info_after_pred = process.memory_info()
rss_after_pred_MB = mem_info_after_pred.rss / (1024 * 1024)
memory_used_pred_MB = rss_after_pred_MB - rss_before_pred_MB

print(f"\nTạo file submission thành công!")
print(f"File được lưu tại: {output_path}")
print("\n--- Thống kê hiệu năng cho quá trình dự đoán ---")
print(f"Thời gian chạy: {prediction_time:.2f} giây")
print(f"Bộ nhớ RSS sau khi hoàn tất: {rss_after_pred_MB:.2f} MB")
print(f"Lượng bộ nhớ RSS sử dụng thêm: {memory_used_pred_MB:.2f} MB")
print("-------------------------------------------------")

print("\nNăm dòng đầu tiên của file submission:")
print(submission_df.head())
print("="*50)


# # 5. Dự đoán và tạo submission
# Y_test_lstm_pred_scaled = lstm_model.predict(X_test_lstm)
# Y_test_lstm_pred = scaler_target.inverse_transform(Y_test_lstm_pred_scaled).clip(0, 20)

# submission_df_lstm = pd.DataFrame({
#     "ID": df[df['date_block_num'] == 34]['ID'], # Đảm bảo ID này khớp với X_test_lstm
#     "item_cnt_month": Y_test_lstm_pred.flatten()
# })

# # Lưu file submission
# submission_df_lstm.to_csv('/kaggle/working/submission_lstm.csv', index=False)
# print("Đã tạo xong file submission cho LSTM (chưa lưu và chưa huấn luyện).")
# print(submission_df_lstm.head())


# submission_df_lstm.to_csv('/kaggle/working/submission.csv', index=False)
# submission_df_xgb.to_csv('/kaggle/working/submission.csv', index=False)
# creagte_submition(df,lgb_model).to_csv('/kaggle/working/submission.csv', index = False)







