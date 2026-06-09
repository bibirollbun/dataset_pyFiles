import numpy as np 
import pandas as pd 


# Load parquet files
train = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet')
test = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet')

train['purpose'] = 'train'

test['selected'] = 0
test['purpose'] = 'test'

df = pd.concat([train,test])

del(train,test)


# Create functions to calculate the number of segments in each leg

def flight_segment_num_leg0(row):
    if pd.notna(row['legs0_segments3_flightNumber']):
        return 4
    elif pd.notna(row['legs0_segments2_flightNumber']):
        return 3
    elif pd.notna(row['legs0_segments1_flightNumber']):
        return 2
    elif pd.notna(row['legs0_segments0_flightNumber']):
       return 1

def flight_segment_num_leg1(row):
    if pd.notna(row['legs1_segments3_flightNumber']):
        return 4
    elif pd.notna(row['legs1_segments2_flightNumber']):
        return 3
    elif pd.notna(row['legs1_segments1_flightNumber']):
        return 2
    elif pd.notna(row['legs1_segments0_flightNumber']):
        return 1
    else:
        return 0


# Craete features with number of segments in each leg

df['leg0_segment_num'] = df[['legs0_segments3_flightNumber',
                                 'legs0_segments2_flightNumber',
                                 'legs0_segments1_flightNumber',
                                 'legs0_segments0_flightNumber']].apply(flight_segment_num_leg0, axis = 1)

df['leg1_segment_num'] = df[['legs1_segments3_flightNumber',
                                 'legs1_segments2_flightNumber',
                                 'legs1_segments1_flightNumber',
                                 'legs1_segments0_flightNumber']].apply(flight_segment_num_leg1, axis = 1)


# Drop columns with 98.9%+ null values

null_col = df.columns[df.isna().sum()/len(df)>0.989]

df.drop(null_col, axis = 1, inplace = True)


# Convert to datetime format
df['legs0_arrivalAt'] = pd.to_datetime(df['legs0_arrivalAt'])
df['legs0_departureAt'] = pd.to_datetime(df['legs0_departureAt'])
df['legs1_arrivalAt'] = pd.to_datetime(df['legs1_arrivalAt'])
df['legs1_departureAt'] = pd.to_datetime(df['legs1_departureAt'])


# Fill nan values
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

df.loc[:,num_cols] = df.loc[:,num_cols].fillna(0)
df.loc[:,cat_cols] = df.loc[:,cat_cols].fillna('missing')


# Downcast numeric columns format (64 to 32 bit)
d1 = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
d2 = dict.fromkeys(df.select_dtypes(np.float64).columns, np.float32)

df = df.astype(d1)
df = df.astype(d2)
df[cat_cols] = df[cat_cols].astype('category')


df['init_order'] = range(df.shape[0])
df['init_order'] = df['init_order'].astype('int32')

# init_order_idx = df.index

df = df.sort_values(by = 'requestDate')


# Cast 'object' columns considered to be categorical to 'category' data type

df['companyID'] = df['companyID'].astype('category')
df['corporateTariffCode'] = df['corporateTariffCode'].astype('category')
df['nationality'] = df['nationality'].astype('category')

meas_cols = df.columns[df.columns.str.contains('MeasurementType')].tolist()

for col in meas_cols:
    df[col] = df[col].astype('category')

df['miniRules0_statusInfos'] = df['miniRules0_statusInfos'].astype('category')
df['miniRules1_statusInfos'] = df['miniRules1_statusInfos'].astype('category')
df['pricingInfo_isAccessTP'] = df['pricingInfo_isAccessTP'].astype('category')


%%time
# check if each filled '..marketingCarrier_code' contains Carrier from 'frequentFlyer' carrier list

mcc_cols = df.columns[df.columns.str.contains('marketingCarrier_code')].tolist()

for col in mcc_cols:
    df[col+'_in_ff'] = [mcc in ff if mcc !='missing' else False for ff, mcc in zip(df['frequentFlyer'],df[col]) ]

_in_ff_cols = df.columns[df.columns.str.contains('_in_ff')].tolist()

df['avg_in_ff'] = df[_in_ff_cols].where((df[mcc_cols].astype('str')!='missing').to_numpy()).mean(axis = 1).astype('float16')


# Calc approximate baggage weight in kg for rows where it is measured in pieces count (using typical 23kg limit for 1 piece)

df['l_0_s_0_approx_bagg_weight'] = np.where(df['legs0_segments0_baggageAllowance_weightMeasurementType'] == 0,
                                            df['legs0_segments0_baggageAllowance_quantity']*23,
                                            df['legs0_segments0_baggageAllowance_quantity']).astype('int32')

df['l_1_s_0_approx_bagg_weight'] = np.where(df['legs1_segments0_baggageAllowance_weightMeasurementType'] == 0,
                                            df['legs1_segments0_baggageAllowance_quantity']*23,
                                            df['legs1_segments0_baggageAllowance_quantity'])

df['l_1_s_0_approx_bagg_weight'] = df['l_1_s_0_approx_bagg_weight'].fillna(0).astype('int32')


%%time
# Create additional date&time features

df['date_block_num'] = (df['requestDate'] - df['requestDate'].min()).dt.days.astype('int32')

df['requestDate_day'] = df['requestDate'].dt.day.astype('int32')
df['requestDate_dow'] = df['requestDate'].dt.day_name().astype('category')
df['requestDate_month'] = df['requestDate'].dt.month_name().astype('category')

df['days_till_flight'] = (df['legs0_departureAt'] - df['requestDate']).dt.days.astype('int32')

df['legs0_departureAt_day'] = df['legs0_departureAt'].dt.day.astype('int32')
df['legs0_departureAt_dow'] = df['legs0_departureAt'].dt.day_name().astype('category')
df['legs0_departureAt_month'] = df['legs0_departureAt'].dt.month_name().astype('category')
df['legs0_departureAt_hour'] = df['legs0_departureAt'].dt.hour.astype('int32')

df['legs0_arrivalAt_day'] = df['legs0_arrivalAt'].dt.day.astype('int32')
df['legs0_arrivalAt_dow'] = df['legs0_arrivalAt'].dt.day_name().astype('category')
df['legs0_arrivalAt_month'] = df['legs0_arrivalAt'].dt.month_name().astype('category')
df['legs0_arrivalAt_hour'] = df['legs0_arrivalAt'].dt.hour.astype('int32')

df['legs1_departureAt_day'] = df['legs1_departureAt'].dt.day.fillna(-1).astype('int32')
df['legs1_departureAt_dow'] = df['legs1_departureAt'].dt.day_name().fillna('missing').astype('category')
df['legs1_departureAt_month'] = df['legs1_departureAt'].dt.month_name().fillna('missing').astype('category')
df['legs1_departureAt_hour'] = df['legs1_departureAt'].dt.hour.fillna(-1).astype('int32')

df['legs1_arrivalAt_day'] = df['legs1_arrivalAt'].dt.day.fillna(-1).astype('int32')
df['legs1_arrivalAt_dow'] = df['legs1_arrivalAt'].dt.day_name().fillna('missing').astype('category')
df['legs1_arrivalAt_month'] = df['legs1_arrivalAt'].dt.month_name().fillna('missing').astype('category')
df['legs1_arrivalAt_hour'] = df['legs1_arrivalAt'].dt.hour.fillna(-1).astype('int32')


# Create a manual (based on info from internet) rule for rating of flight hour of day
def depart_hour_rat(hour):
    if hour in [23,0,1,2,3,4] : 
        return 1
    elif hour in [5,6,7,21,22] :
        return 2
    elif hour in [8,12,13,14,15,20] :
        return 3
    elif hour == -1:
        return -1
    else :
        return 4


%%time
df['legs0_departureAt_hour_rat'] = df['legs0_departureAt_hour'].apply(depart_hour_rat).astype('int32')
# df['legs0_departureAt_hour_rat'].value_counts()
df['legs0_arrivalAt_hour_rat'] = df['legs0_arrivalAt_hour'].apply(depart_hour_rat).astype('int32')
df['legs1_departureAt_hour_rat'] = df['legs1_departureAt_hour'].apply(depart_hour_rat).astype('int32').fillna(-1)
df['legs1_arrivalAt_hour_rat'] = df['legs1_arrivalAt_hour'].apply(depart_hour_rat).astype('int32').fillna(-1)


df['hour_rat_avg'] = df[['legs0_departureAt_hour_rat',
                        'legs0_arrivalAt_hour_rat',
                        'legs1_departureAt_hour_rat',
                        'legs1_arrivalAt_hour_rat']][df[['legs0_departureAt_hour_rat',
                                                        'legs0_arrivalAt_hour_rat',
                                                        'legs1_departureAt_hour_rat',
                                                        'legs1_arrivalAt_hour_rat']] != -1].mean(axis = 1).astype('float32')


%%time
# Calc duration in minutes

dur_cols = df.columns[df.columns.str.contains('duration')].tolist()

for col in dur_cols:
    df[col+'_min'] = (pd.to_timedelta(df[col].astype('object').str.replace('.',' days '), 
                                      errors = 'coerce').dt.total_seconds().fillna(0)/60).astype('int32')



# Calc wait time in minutes

df['legs0_wait_time'] = df['legs0_duration_min'] - df['legs0_segments0_duration_min'] - df['legs0_segments1_duration_min']
df['legs1_wait_time'] = df['legs1_duration_min'] - df['legs1_segments0_duration_min'] - df['legs1_segments1_duration_min']


df['tax_rate'] = df['taxes'] / df['totalPrice']
df['is_return'] = df['searchRoute'].astype('object').str.contains('/')


# Calc the number of segments, total duration, wait time and cabin class characteristics

#df['total_segments'] = df['leg0_segment_num'] + df['leg1_segment_num']
df['avg_segments'] = df[['leg0_segment_num',
                         'leg1_segment_num']][df[['leg0_segment_num',
                                                  'leg1_segment_num']]>0].mean(axis = 1).astype('float32')

df['total_dur_min'] = df['legs0_duration_min'] + df['legs1_duration_min']
df['seg_dur_rat'] = (df['legs0_duration_min']/df['legs1_duration_min']).replace([np.inf, -np.inf], 0).astype('float32')
df['total_wait_time'] = df['legs0_wait_time'] + df['legs1_wait_time']

df['avg_cabinClass'] = df[['legs0_segments0_cabinClass',
                           'legs0_segments1_cabinClass',
                           'legs1_segments0_cabinClass',
                           'legs1_segments1_cabinClass']][df[['legs0_segments0_cabinClass',
                                                               'legs0_segments1_cabinClass',
                                                               'legs1_segments0_cabinClass',
                                                               'legs1_segments1_cabinClass']]>0].mean(axis = 1)


%%time
# Add Carrier rating features

oper_car_rat = pd.concat([df[(df['purpose'] != 'test')&(df['selected'] == 1)]['legs0_segments0_operatingCarrier_code'],
                          df[(df['purpose'] != 'test')&(df['selected'] == 1)]['legs0_segments1_operatingCarrier_code'],
                          df[(df['purpose'] != 'test')&(df['selected'] == 1)]['legs1_segments0_operatingCarrier_code'],
                          df[(df['purpose'] != 'test')&(df['selected'] == 1)]['legs1_segments1_operatingCarrier_code']]).value_counts().to_frame().rename({'count':'rating'},
                                                                                                                                                          axis = 1).drop(labels = 'missing', axis = 0)

for col in ['legs0_segments0_operatingCarrier_code',
            'legs0_segments1_operatingCarrier_code',
            'legs1_segments0_operatingCarrier_code',
            'legs1_segments1_operatingCarrier_code']:
    df[col+'_rating'] = df[[col]].merge(oper_car_rat, 
                                      how='left', 
                                      left_on = col, 
                                      right_index = True).rename({'rating':col+'_rating'}, axis = 1)[col+'_rating']
    df[col+'_rating'] = df[col+'_rating'].fillna(0).astype('int32')

df['avg_operatingCarrier_code_rating'] = df[['legs0_segments0_operatingCarrier_code_rating',
                                              'legs0_segments1_operatingCarrier_code_rating',
                                              'legs1_segments0_operatingCarrier_code_rating',
                                              'legs1_segments1_operatingCarrier_code_rating']][df[['legs0_segments0_operatingCarrier_code_rating',
                                                                                                    'legs0_segments1_operatingCarrier_code_rating',
                                                                                                    'legs1_segments0_operatingCarrier_code_rating',
                                                                                                    'legs1_segments1_operatingCarrier_code_rating']]>0].mean(axis = 1).fillna(0).astype('float32')


df['final_dest_city'] = np.where(df['legs0_segments1_arrivalTo_airport_city_iata'] != 'missing',
                                 df['legs0_segments1_arrivalTo_airport_city_iata'],
                                 df['legs0_segments0_arrivalTo_airport_city_iata'])

df['final_dest_city'] = df['final_dest_city'].astype('category')


# Create additional tables for route mean duration,price considering segments number, cabin class; route popularity

route_avg_dur = df[df['purpose'] != 'test'].groupby(['searchRoute',
                                                      'legs0_segments0_departureFrom_airport_iata',
                                                      'final_dest_city'],
                                                     observed=True)['total_dur_min'].mean().to_frame().rename({'total_dur_min':'route_dur_avg'}, 
                                                                                                                axis = 1)
route_avg_dur_seg = df[df['purpose'] != 'test'].groupby(['searchRoute',
                                                          'legs0_segments0_departureFrom_airport_iata',
                                                          'final_dest_city',
                                                          'leg0_segment_num',
                                                          'leg1_segment_num'],
                                                         observed=True)['total_dur_min'].mean().to_frame().rename({'total_dur_min':'route_seg_dur_avg'}, 
                                                                                                                   axis = 1)
route_cc_price = df[df['purpose'] != 'test'].groupby(['searchRoute',
                                                       'legs0_segments0_departureFrom_airport_iata',
                                                       'final_dest_city',
                                                       'legs0_segments0_cabinClass',
                                                        'legs0_segments1_cabinClass',
                                                        'legs1_segments0_cabinClass',
                                                        'legs1_segments1_cabinClass'],
                                                       observed=True)['totalPrice'].mean().to_frame().rename({'totalPrice':'route_cc_avgPrice'}, 
                                                                                                            axis = 1)
route_popularity = df[(df['purpose'] != 'test')&(df['selected']==1)].groupby(['searchRoute'],
                                                          observed=True)['searchRoute'].count().to_frame().rename({'searchRoute':'route_pop'}, 
                                                                                                                axis = 1)  


%%time
# Merging data from the cell abobe to the main df

df['route_dur_avg'] = df[['searchRoute',
                          'legs0_segments0_departureFrom_airport_iata',
                          'final_dest_city']].merge(route_avg_dur, 
                                                how='left', 
                                                left_on = (['searchRoute',
                                                           'legs0_segments0_departureFrom_airport_iata',
                                                            'final_dest_city']),
                                                right_index = True)['route_dur_avg']
df['route_dur_avg'] = df['route_dur_avg'].fillna(0).astype('float32')

df['route_seg_dur_avg'] = df[['searchRoute',
                              'legs0_segments0_departureFrom_airport_iata',
                              'final_dest_city',
                              'leg0_segment_num',
                              'leg1_segment_num']].merge(route_avg_dur_seg, 
                                                        how='left', 
                                                        left_on = (['searchRoute',
                                                                    'legs0_segments0_departureFrom_airport_iata',
                                                                    'final_dest_city',
                                                                    'leg0_segment_num',
                                                                    'leg1_segment_num']), 
                                                        right_index = True)['route_seg_dur_avg']
df['route_seg_dur_avg'] = df['route_seg_dur_avg'].fillna(0).astype('float32')

df['route_cc_avgPrice'] = df[['searchRoute',
                              'legs0_segments0_departureFrom_airport_iata',
                              'final_dest_city',
                              'legs0_segments0_cabinClass',
                              'legs0_segments1_cabinClass',
                              'legs1_segments0_cabinClass',
                              'legs1_segments1_cabinClass']].merge(route_cc_price, 
                                                                    how='left', 
                                                                    left_on = (['searchRoute',
                                                                                'legs0_segments0_departureFrom_airport_iata',
                                                                                'final_dest_city',
                                                                                'legs0_segments0_cabinClass',
                                                                                'legs0_segments1_cabinClass',
                                                                                'legs1_segments0_cabinClass',
                                                                                'legs1_segments1_cabinClass']), 
                                                                    right_index = True)['route_cc_avgPrice']
df['route_cc_avgPrice'] = df['route_cc_avgPrice'].fillna(0).astype('float32')

df['route_pop'] = df[['searchRoute']].merge(route_popularity, 
                                                how='left', 
                                                left_on = (['searchRoute']),
                                                right_index = True)['route_pop']
df['route_pop'] = df['route_pop'].fillna(0).astype('int32')


%%time
# Add rank and scaled features per each ranker_id

cols_to_rank = ['avg_segments','total_dur_min','total_wait_time','totalPrice','hour_rat_avg','avg_in_ff']

for col in cols_to_rank:
    df[col+'_rank'] = df.groupby('ranker_id',
                                observed=True)[col].rank(method = 'dense').astype('float32')
    df[col+'_std'] = df.groupby('ranker_id',
                                observed=True)[col].transform('std').fillna(0).astype('float32')
    df[col+'_nunique'] = df.groupby('ranker_id',
                                    observed=True)[col].transform('nunique').astype('float32')
    

cols_to_scale = ['total_dur_min','total_wait_time','totalPrice']

for col in cols_to_scale:    
    df[col+'_scaled'] = ((df[col] - df.groupby('ranker_id',
                                             observed=True)[col].transform('min')) \
                       / \
                       (df.groupby('ranker_id',
                                    observed=True)[col].transform('max') - df.groupby('ranker_id',
                                                                                      observed=True)[col].transform('min'))*(10-1)+1).replace([np.inf, -np.inf], 1).fillna(1).astype('float32')


df['dur2price_ratio'] = ((1/df['total_dur_min_scaled']) / df['totalPrice_scaled']).astype('float32')


df['id_per_ranker_id'] = df.groupby('ranker_id',
                                    observed=True)['Id'].transform('count').astype('int32')


df['condition_flag'] = df['selected']==1


df['prof_d_since_acq'] = (df['requestDate'] \
                          - \
                          df.groupby('profileId')['requestDate'].transform('min')).dt.days.astype('int32')


%%time
df['prof_tot_purch'] = df.groupby(['profileId'],
                                  observed = True)['selected'].transform('cumsum') \
                     - \
                       df.groupby(['profileId','ranker_id'],
                                  observed = True)['selected'].transform('cumsum')


%%time
# Add expanding mean features based on profile historic data - previuos to a given ranker_id

cols_exp_mean = ['avg_in_ff',
                 'avg_segments',
                 'avg_segments_rank',
                 'total_dur_min_rank',
                 'total_dur_min_scaled',
                 'totalPrice_rank',
                 'totalPrice_scaled',
                 'dur2price_ratio',
                 'avg_cabinClass',
                 'legs0_departureAt_hour_rat',
                 'legs0_arrivalAt_hour_rat',
                 'hour_rat_avg',
                 'avg_operatingCarrier_code_rating',
                 'l_0_s_0_approx_bagg_weight']

for col in cols_exp_mean:
    df[col+'_exp_mean'] = (df.groupby(['profileId'],
                                     observed = True)[col].transform(lambda x: x[df.loc[x.index,'condition_flag']].cumsum()) \
                         - \
                         df.groupby(['profileId','ranker_id'],
                                     observed = True)[col].transform(lambda x: x[df.loc[x.index,'condition_flag']].cumsum()) ) \
                         / df['prof_tot_purch']
    # df[col+'_exp_mean'] = df.groupby(['profileId','ranker_id'],
    #                                   observed = True)[col+'_exp_mean'].transform(lambda x: x.ffill().bfill())
    df[col+'_exp_mean'] = df.groupby(['profileId','ranker_id'],
                                      observed = True)[col+'_exp_mean'].ffill()
    df[col+'_exp_mean'] = df.groupby(['profileId','ranker_id'],
                                      observed = True)[col+'_exp_mean'].bfill()
    df[col+'_exp_mean'] = df[col+'_exp_mean'].fillna(0).astype('float32')    


# Drop unnecessary columns

col_to_excl = ['Id',
             'pricingInfo_passengerCount',
             # 'requestDate',
             'legs0_departureAt',
             'legs0_arrivalAt',
             'legs1_departureAt',
             'legs1_arrivalAt',
             'legs0_duration',
             'legs0_segments0_duration',
             'legs0_segments1_duration',
             'legs1_duration',
             'legs1_segments0_duration',
             'legs1_segments1_duration',
             'condition_flag']

df = df.drop(col_to_excl, axis = 1)

df = df.sort_values(by = 'init_order')


# Creating a list of features to exclude from training

spr_feat = ['selected','purpose','ranker_id','profileId','init_order','requestDate']
fl_rout  = ['searchRoute'] + df.columns[df.columns.str.contains('flightNumber')].tolist()
aircr_code = df.columns[df.columns.str.contains('aircraft_code')].tolist()

cols_no_use_all = [
                    # 'bySelf',
                     # 'is_return',
                     'legs0_segments0_baggageAllowance_weightMeasurementType',
                     'legs0_segments1_baggageAllowance_weightMeasurementType',
                     'legs1_segments0_baggageAllowance_weightMeasurementType',
                     'legs1_segments1_baggageAllowance_weightMeasurementType',
                     # 'miniRules1_percentage'
                  ]



cols_to_drop = spr_feat + fl_rout + cols_no_use_all
# + aircr_code
cat_cols = df.drop(columns = cols_to_drop).select_dtypes(include=['category']).columns.tolist()


# Split and allocate a part of data for validation

val_perc = 90

val_threshold = np.percentile(df[df['purpose'] != 'test']['requestDate'].unique(),val_perc)

df['purpose'] = df['purpose'].astype('object')

df.loc[(df['purpose'] != 'test')&(df['requestDate']>val_threshold),['purpose']] = 'valid'
df.loc[(df['purpose'] != 'test')&(df['requestDate']<=val_threshold),['purpose']] = 'train'

df['purpose'] = df['purpose'].astype('category')


# Define a func to calculate the target copetition 'hitrate@3' metric

def hit_rate_n(y_true, y_pred_score, ranker_id, n):
    df = pd.DataFrame({'y_true':y_true,'y_pred_score':y_pred_score,'ranker_id':ranker_id})
    df_sorted = df.sort_values(['ranker_id', 'y_pred_score'], ascending=[True, False])
    df_top_n = df_sorted.groupby('ranker_id', observed = True).head(n).reset_index(drop=True)
    return df_top_n['y_true'].sum() / df_top_n['ranker_id'].nunique()


!pip install -U xgboost
import xgboost as xgb


# Crating 'group_sizes' objects for group and eval_group parameters of XGB Ranker model

train_group_sizes = df[(df['purpose'] == 'train')].groupby('ranker_id', 
                                                         observed = True, 
                                                         sort = False)['ranker_id'].count()
val_group_sizes = df[(df['purpose'] == 'valid')].groupby('ranker_id', 
                                                         observed = True, 
                                                         sort = False)['ranker_id'].count()
# tval_group_sizes = df[(df['purpose'] != 'test')].groupby('ranker_id', 
#                                                         observed = True, 
#                                                         sort = False)['ranker_id'].count()
test_group_sizes = df[df['purpose'] == 'test'].groupby('ranker_id', 
                                                         observed = True, 
                                                         sort = False)['ranker_id'].count()


# Parameters were determined with manual parameters search loops

xgb_params = {
    'max_depth': 8,
    'min_child_weight': 11,
    # 'subsample': 0.8,
    # 'colsample_bytree': 0.8,
    'lambda': 2.0,
    'learning_rate': 0.05
}

xgbr = xgb.XGBRanker(
                     **xgb_params,
                     objective = 'rank:pairwise',
                     # objective = 'rank:ndcg',
                     eval_metric = 'ndcg@3',
                     enable_categorical = True,
                       n_estimators = 1000,
                       early_stopping_rounds = 85,
                       random_state = 42)


xgbr.fit(df[df['purpose'] == 'train'].drop(columns = cols_to_drop), 
         df[df['purpose'] == 'train']['selected'], 
         group = train_group_sizes,
         eval_set = [(df[df['purpose'] == 'train'].drop(columns = cols_to_drop),
                      df[df['purpose'] == 'train']['selected']),
                     (df[df['purpose'] == 'valid'].drop(columns = cols_to_drop),
                      df[df['purpose'] == 'valid']['selected'])],
         eval_group = [train_group_sizes,
                       val_group_sizes],
         verbose = 10)


# Check target metric on validation dataset

hit_rate_n(df[(df['purpose'] == 'valid')&(df['id_per_ranker_id']>10)]['selected'], 
           xgbr.predict(df[(df['purpose'] == 'valid')&(df['id_per_ranker_id']>10)].drop(columns = cols_to_drop)), 
           df[(df['purpose'] == 'valid')&(df['id_per_ranker_id']>10)]['ranker_id'],
           3 )


# Check feature importance

xgbr_feat_df = pd.DataFrame({'Feature': df[df['purpose'] == 'train'].drop(columns=cols_to_drop).columns, 
                              'Importance': xgbr.feature_importances_}).sort_values(by='Importance', ascending=False)
# xgbr_feat_no_imp = set(xgbr_feat_df['Feature'][xgbr_feat_df['Importance'] == 0])
xgbr_feat_df[:50]


# Create submission

subm_xgbr = df[df['purpose'] == 'test'][[ 'ranker_id']].copy()
subm_xgbr['Id'] = subm_xgbr.index
subm_xgbr['pred'] = xgbr.predict(df[df['purpose'] == 'test'].drop(columns = cols_to_drop))
subm_xgbr['selected'] = subm_xgbr.groupby(['ranker_id'], 
                                        observed = True)['pred'].rank(ascending=False, method='first').astype('int32')
subm_xgbr[['Id','ranker_id','selected']].to_csv('submission.csv', index=False)
subm_xgbr[['Id','ranker_id','selected']].head(20)

