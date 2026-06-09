# lib import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
TRAINING = True


def generate_features(df):
    features = ['seconds_in_bucket', 'imbalance_buy_sell_flag',
               'imbalance_size', 'matched_size', 'bid_size', 'ask_size',
                'reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap',
                'imb_s1', 'imb_s2'
               ]

    df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)')
    df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)')

    prices = ['reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap']

    for i,a in enumerate(prices):
        for j,b in enumerate(prices):
            if i>j:
                df[f'{a}_{b}_imb'] = df.eval(f'({a}-{b})/({a}+{b})')
                features.append(f'{a}_{b}_imb')

    for i,a in enumerate(prices):
        for j,b in enumerate(prices):
            for k,c in enumerate(prices):
                if i>j and j>k:
                    max_ = df[[a,b,c]].max(axis=1)
                    min_ = df[[a,b,c]].min(axis=1)
                    mid_ = df[[a,b,c]].sum(axis=1)-min_-max_

                    df[f'{a}_{b}_{c}_imb2'] = (max_-mid_)/(mid_-min_)
                    features.append(f'{a}_{b}_{c}_imb2')
                    
    #There are lots of feature generated from the for loops above
    #--------------------------------------My attemptions ----------------------------------------
    # Create 3 lag features for the 'target' column
    df['target_lag_1'] = df['target'].shift(1)
    df['target_lag_2'] = df['target'].shift(2)
    df['target_lag_3'] = df['target'].shift(3)
    df[['target_lag_1', 'target_lag_2', 'target_lag_3']] = df[['target_lag_1', 'target_lag_2', 'target_lag_3']].fillna(0)
    # Add these lag features to the features list
    features.extend(['target_lag_1', 'target_lag_2', 'target_lag_3'])

    '''
    #Ref from 6th place solution
    # （1）集合竞价前n秒数据分桶
    df['seconds_in_bucket_flag_1'] = df['seconds_in_bucket'] >= 300 - 60
    df['seconds_in_bucket_flag_2'] = df['seconds_in_bucket'] >= 300
    df['seconds_in_bucket_flag_3'] = df['seconds_in_bucket'] >= 480 - 60
    df['seconds_in_bucket_flag_4'] = df['seconds_in_bucket'] >= 480
    features.extend(['seconds_in_bucket_flag_1','seconds_in_bucket_flag_2','seconds_in_bucket_flag_3','seconds_in_bucket_flag_4'])
    
    
    # （2）常规量化因子
    df["volume"] = df['ask_size'] + df['bid_size']
    df["mid_price"] = (df['ask_price'] + df['bid_price']) / 2
    df["liquidity_imbalance"] = (df['bid_size'] - df['ask_size']) / df["volume"]
    df["matched_imbalance"] = (df['imbalance_size'] - df['matched_size']) / (df['imbalance_size'] + df['matched_size'])
    df["size_imbalance"] = df['bid_size'] / df['ask_size']
    df['harmonic_imbalance'] = 2 / ((1 / df['bid_size']) + (1 / df['ask_size'] ))
    features.extend(["volume","mid_price","liquidity_imbalance","matched_imbalance","size_imbalance",'harmonic_imbalance'])
    
    '''


    return df[features]



def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
 
            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    print(type(props))  
    return props, NAlist


import gc

if TRAINING:
    # load dataset 
    df_train = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')
    print('file read done')
    #df_train,NAlist= reduce_mem_usage(df_train)
    #print('reduced memory usage')

    df_features = generate_features(df_train)
    print('feature generated')
    
    X = df_features.values
    Y = df_train['target'].values
    
    #X = X[np.isfinite(Y)] 
    #Y = Y[np.isfinite(Y)]
    # if X has Nan value but its Y has not, this method cannot clean the Nan value 

    # Find rows where both X and Y are finite
    valid_mask = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
    #this method work for that case
    X = X[valid_mask]
    Y = Y[valid_mask]
    index = np.arange(len(X))
    print("DONE")


import lightgbm as lgb 
import catboost as cat
import joblib
import os


models = []

# test to train ratio
N_fold = 5

os.system('mkdir models')

#model_path ='/kaggle/input/testing_data_set/models'
model_path ='/kaggle/input/optimize'

def train(model_dict, modelname='cat'):
    if TRAINING:
        model = model_dict[modelname]
        model.fit(X[index%N_fold!=i], Y[index%N_fold!=i],
          eval_set=(X[index%N_fold==i], Y[index%N_fold==i]),
          early_stopping_rounds=100,
          verbose = 10
         )
        models.append(model)
        joblib.dump(model, f'./models/{modelname}_{i}.model')
    else:
        models.append(joblib.load(f'{model_path}/{modelname}_{i}.model'))


optimized_catboost_params = {
    'iterations': 100,                  # Increased for better convergence
    'learning_rate': 0.03,               # Lower for financial data precision
    'depth': 8,                         # Slightly deeper for complex relationships
    'l2_leaf_reg': 5,                   # Stronger regularization
    'random_strength': 0.5,             # Reduced randomness for stability
    'early_stopping_rounds': 150,       # More patience for financial data
    'grow_policy': 'Lossguide',         # Better for heterogeneous features
    'min_data_in_leaf': 50,             # Prevent overfitting
    'max_leaves': 64,                   # Control complexity
    'nan_mode': 'Min',                  # Handle missing values
    'fold_len_multiplier': 1.1,         # Slightly more conservative
    'verbose': 100,
    #'cat_features': ['stock_id', 'imbalance_buy_sell_flag']  # Explicit categoricals
}

model_dict = {
    'cat': cat.CatBoostRegressor(**optimized_catboost_params)
}
for i in range(N_fold):
    train(model_dict, 'cat')


import optiver2023
env = optiver2023.make_env()
iter_test = env.iter_test()


for (test, revealed_targets, sample_prediction) in iter_test:
    feat = generate_features(test)
    
    sample_prediction['target'] = np.mean([model.predict(feat) for model in models], 0)
    env.predict(sample_prediction)

