import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import joblib
import warnings
import gc
import joblib

warnings.filterwarnings('ignore')


df=pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')
df.sort_values(by='date_id',ascending=True)
df.reset_index(drop=True,inplace=True)
df.head()


# df.shape (5237980, 17)
df= df.dropna(subset=['target'],axis=0) # drop rows where taget is nan
df.drop_duplicates(inplace=True)
# df.shape (5237980, 17)
df.reset_index(drop=True,inplace=True)
# Reduce Memory Usage 

def mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2  # Total memory in MB

    for col in df.columns:
        print(f"Processing column: {col}, dtype: {df[col].dtype}")  # Debugging output
        col_type = df[col].dtype
        
        # Check for integer types
        if np.issubdtype(col_type, np.integer):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)

        # Check for float types
        elif np.issubdtype(col_type, np.floating):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float32) # numba is not supporting float16
            elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB")
    print(f"Decrease: {100 * (start_mem - end_mem) / start_mem:.2f}%")
    
    return df

df = mem_usage(df)


from numba import njit, prange
from itertools import combinations

@njit(parallel=True)

def compute_triplet_imbalance(df_values,comb_indices):
    num_row=df_values.shape[0]
    num_comb=len(comb_indices)
    imbalance_features=np.empty((num_row,num_comb))
    
    for i in prange(num_comb):
        a,b,c=comb_indices[i]
        
        for j in range(num_row):
            max_val=max(df_values[j,a],df_values[j,b],df_values[j,c])
            min_val=min(df_values[j,a],df_values[j,b],df_values[j,c])
            mid_val=df_values[j,a]+df_values[j,b]+df_values[j,c]-min_val-max_val
            
            if min_val==mid_val:
                imbalance_features[j,i]=np.nan
            else:
                imbalance_features[j,i]=(max_val-mid_val)/(mid_val-min_val)
            
    return imbalance_features
        

def calcul_triplet_imbalance(price,df):
    df_values=df[price].values
    comb_indices=np.array([(price.index(a),price.index(b),price.index(c)) for a,b,c in combinations(price,3)],dtype=np.int64)
#     print(comb_indices)
    feature_array=compute_triplet_imbalance(df_values,comb_indices)

    columns=[f'{a}_{b}_{c}_imb' for a,b,c in combinations(price,3)]
    features=pd.DataFrame(feature_array,columns=columns)
    return features
    


def imbalance_features(df):
    prices=["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    
    df['volume']= df.eval('ask_size + bid_size')
    df['mid_price']= df.eval('(ask_price + bid_price)/2')
    df['liquidity_imbalance']= df.eval('(bid_size - ask_size)/(bid_size + ask_size)')
    df['matched_imbalance']= df.eval('(imbalance_size - matched_size)/(imbalance_size + matched_size)')
    df['size_imbalance']= df.eval('bid_size/ask_size')
    
    for c in combinations(prices,2):
        df[f'{c[0]}_{c[1]}_imb']= df.eval(f'({c[0]}-{c[1]})/({c[0]}+{c[1]})')
# considering 3 features at a time and finding their relationships
    for c in [['ask_price', 'bid_price', 'wap', 'reference_price'], sizes]:
        triplet_feature=calcul_triplet_imbalance(c,df)
        df[triplet_feature.columns]=triplet_feature.values
      # new features
        df['imbalance_momentum']=df.groupby(['stock_id'])['imbalance_size'].diff(periods=1)/df['matched_size']
        df['price_spread']=df['ask_price']-df['bid_size']
        df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
        df['price_pressure'] = df['imbalance_size'] * (df['ask_price'] - df['bid_price'])
        df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
        df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])
    
    #measures of all cols of single row
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)
        
    # lag features and % change in values over the given time frame
    for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_buy_sell_flag']:
        for window in [1,2,5,10]:
            df[f'{col}_shift_{window}']=df.groupby(['stock_id'])[col].shift(window)
            df[f'{col}_return_{window}']=df.groupby(['stock_id'])[col].pct_change(window)
    
    for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size']:
        for window in [1,2,5,10]:
            df[f'{col}_diff_{window}']=df.groupby(['stock_id'])[col].diff(window)
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    return df


def other_features(df):
    df['day_of_wk']=(df['date_id'])%5
    df['seconds']=df['seconds_in_bucket']%60
    df['minute']=df['seconds_in_bucket']//60
    
#     for k,v in global_stock_id_feats.items():
#         df[f'global_{key}']=df['stock_id'].map(value.to_dict())
    return df

def generate_all_features(df):
    cols = [c for c in df.columns if c not in ['row_id','time_id','target']]
    df=df[cols]
    df=imbalance_features(df)
    df=other_features(df)
    gc.collect()
    feature_name=[i for i in df.columns if i not in ['row_id','date_id','target','time_id']]
    return df[feature_name]


# Splitting data
split_day=435
date_ids=df['date_id'].values
df_train=df[df['date_id']<=split_day]
df_test=df[df['date_id']>split_day]


# df[df['date_id']>435].count() #494999
# df[df['date_id']<=435].count() #4742893
start=0
end=480//5
purged_set =(
(date_ids>= start-2) & (date_ids<=start+2)|
(date_ids>= end-2) & (date_ids<=end+2)
)

test_indices = (date_ids>=start) & (date_ids<end) & (date_ids>split_day) &  ~purged_set
train_indices= ~test_indices & ~purged_set & date_ids<=split_day
print(len(test_indices),len(train_indices))


df_train_feats=generate_all_features(df_train)
df_test_feats=generate_all_features(df_test)
# len(df_test_feats)


import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

lgb_params={
    'objective':'mae',
    'n_estimators':5000,
    'num_leaves':250,
    'learning_rate':0.008,
    'n_jobs':4,
    'verbosity':-1,
    'importance_type':'gain'
    
}
feature_name=list(df_train_feats.columns)
# print(feature_name)

models=[]
score=[]

# saving models 
# model_save_path='/kaggle/input/optiver-trading-at-the-close/'
# if not model_save_path:
#     os.makedirs(model_save_path)


    
df_fold_train= df_train_feats
df_fold_train_target=df_train['target']


df_fold_test=df_test_feats
df_fold_test_target=df_test['target']


lgb_model=lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(df_fold_train[feature_name],df_fold_train_target,
             eval_set=[(df_fold_test[feature_name],df_fold_test_target)],
              callbacks=[lgb.callback.early_stopping(stopping_rounds=100),
                        lgb.callback.log_evaluation(period=100)]
             )

models.append(lgb_model)
# model_filename=os.path.join(model_save_path,f'1.txt')
# lgb_model.booster_.save_model(model_filename)

fold_predictions = lgb_model.predict(df_fold_test[feature_name])
fold_score=mean_absolute_error(fold_predictions,df_fold_test_target)
score.append(fold_score)
print(f'MAE : {fold_score}')


# Submission

import optiver2023
env = optiver2023.make_env()
iter_test = env.iter_test()



counter = 0



counter = 0
for (test, revealed_targets, sample_prediction) in iter_test:
    if counter == 0:
        print(test.head(3))
        print(revealed_targets.head(3))
        print(sample_prediction.head(3))
    test=test.drop('currently_scored',axis=1)
    test=generate_all_features(test)
    sample_prediction['target'] = lgb_model.predict(test)
    env.predict(sample_prediction)
    counter += 1

print('Submision is completed')

