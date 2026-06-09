import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss 
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from lightgbm import early_stopping, log_evaluation
from itertools import combinations
import gc
from sklearn.base import BaseEstimator, TransformerMixin

import warnings 
warnings.filterwarnings('ignore')


competition_data_path = '/kaggle/input/solana-skill-sprint-memcoin-graduation'
datasets_path = '/kaggle/input/pump-fun-graduation-february-2025'

def deta_prep(data_path1,data_path2):
    # test and train data 
    train = pd.read_csv(os.path.join(data_path1,'train.csv'))
    test = pd.read_csv(os.path.join(data_path2,'test_unlabeled.csv'))
    train = train[['mint','slot_min','has_graduated']]
    test = test[['mint','slot_min']]
    train['is_train'] = 1 
    train['has_graduated'] = train['has_graduated'].astype('int')
    test['is_train'] = 0 

    # Helping data 
    chunks = os.path.join(data_path2,'chunk*.csv')
    dune_info = os.path.join(data_path2,'dune_token_info_v2.csv')
    on_chain_info = os.path.join(data_path2,'token_info_onchain_divers_v2.csv')
    
    dune_info_temp = pd.read_csv(dune_info)
    dune_info_temp.rename(columns = {'token_mint_address':'mint'},inplace=True)
    on_chain_df = pd.read_csv(on_chain_info)

    chunk_file_path_list = glob.glob(chunks)
    df_list = [pd.read_csv(file) for file in chunk_file_path_list] 
    all_chunk = pd.concat(df_list,ignore_index = True)
    del df_list
    
    return train,test,dune_info_temp,on_chain_df,all_chunk

train,test,dune_info_temp,on_chain_df,all_chunk = deta_prep(competition_data_path,datasets_path)


def combined_dataframe(train,test,on_chain_df,dune_info_df):
    
    combined = pd.concat([train,test],axis=0,ignore_index=True).reset_index(drop=True)
    on_chain_temp = on_chain_df.drop_duplicates()
    combined = combined.merge(dune_info_temp,on='mint',how='left').merge(on_chain_temp,on='mint',how='left')
    
    combined['created_at'] = pd.to_datetime(combined['created_at'], errors='coerce').dt.tz_localize(None)
    combined['block_time'] = pd.to_datetime(combined['block_time'], errors='coerce')
    combined['version'] = pd.to_datetime(combined['version'], errors='coerce')

    cols_to_drop = ['curve_address','init_tx','decimals','token_uri','name_y','symbol_y','url','bundle_size']
    combined = combined.drop(cols_to_drop,axis=1)

    return combined
combined = combined_dataframe(train,test,on_chain_df,dune_info_temp)


def transection_data(all_chunk,train,test):
    train_test_df = pd.concat([train,test],axis=0,ignore_index=True).reset_index(drop=True)
    transectional_df = all_chunk.merge(train_test_df,left_on = 'base_coin',right_on = 'mint',how='left')
    transectional_df = transectional_df[transectional_df['slot']<=transectional_df['slot_min']+100].reset_index(drop=True)
    del train_test_df
    return transectional_df
transectional_df = transection_data(all_chunk,train,test)


transectional_df.head(3)


target = 'has_graduated'


del dune_info_temp,on_chain_df, train,all_chunk
gc.collect()


def transection_fe(data):
    df = data.copy()
    
    # Prepare flags and datetime
    df['buy'] = (df['direction'] == 'buy').astype(int)
    df['sell'] = (df['direction'] == 'sell').astype(int)
    df['block_time'] = pd.to_datetime(df['block_time'], errors='coerce')

    # Standard Aggregation
    agg_func = {
        'signature': 'count',
        'buy': 'sum',
        'sell': 'sum',
        'slot': 'max',
        'base_coin_amount': ['sum', 'mean', 'max', 'min', 'last'],
        'quote_coin_amount': ['sum', 'mean', 'max', 'min', 'last'],
        'fee': ['sum', 'mean', 'std', 'max', 'min', 'last'],
        'provided_gas_fee': ['sum', 'mean', 'std'],
        'provided_gas_limit': ['sum', 'mean'],
        'consumed_gas': ['sum', 'mean', 'std', 'max', 'min', 'last'],
        'block_time': ['min', 'max'],
        'signing_wallet': 'nunique',
        'virtual_token_balance_after': ['sum', 'mean'],
        'virtual_sol_balance_after': ['sum', 'mean', 'max', 'min', 'last'],
    }

    # Main aggregation
    main_agg = df.groupby('mint').agg(agg_func)
    main_agg.columns = ['_'.join(col).strip() for col in main_agg.columns.values]
    main_agg.reset_index(inplace=True)

    # Define columns for conditional aggregation
    conditional_cols = [
        'base_coin_amount', 'quote_coin_amount',
        'consumed_gas', 'virtual_token_balance_after',
        'virtual_sol_balance_after'
    ]
    cond_aggs = ['sum', 'mean', 'max', 'min']

    # Buy-only aggregation
    df_buy = df[df['buy'] == 1]
    buy_agg = df_buy.groupby('mint')[conditional_cols].agg(cond_aggs)
    buy_agg.columns = [f"{col}_buy_{agg}" for col in conditional_cols for agg in cond_aggs]
    buy_agg.reset_index(inplace=True)

    # Sell-only aggregation
    df_sell = df[df['sell'] == 1]
    sell_agg = df_sell.groupby('mint')[conditional_cols].agg(cond_aggs)
    sell_agg.columns = [f"{col}_sell_{agg}" for col in conditional_cols for agg in cond_aggs]
    sell_agg.reset_index(inplace=True)

    # Merge all
    df_final = main_agg.merge(buy_agg, on='mint', how='left')
    df_final = df_final.merge(sell_agg, on='mint', how='left')

    return df_final
transect_df = transection_fe(transectional_df)


del transectional_df 
gc.collect()


df = combined.merge(transect_df,on='mint',how='left')


# add more transectional behvaior features 
df['buy_sell_ratio'] = df['buy_sum']/(df['sell_sum']+1e-6)
df['buy_ratio'] = df['buy_sum']/(df['buy_sum']+df['sell_sum']+1e-6)
df['sell_ratio'] = df['sell_sum']/(df['buy_sum']+df['sell_sum']+1e-6)
df['active_days'] = (df['block_time_max'] - df['block_time_min']).dt.total_seconds()/86400
df['active_hours'] = (df['block_time_max'] - df['block_time_min']).dt.total_seconds()/3600
df['active_seconds'] =(df['block_time_max'] - df['block_time_min']).dt.total_seconds()

df['avg_transection_per_day'] = df['signature_count']/(df['active_days']+1e-6) # here signature count is total transection
df['avg_base_coin_per_tx'] = df['base_coin_amount_sum']/(df['signature_count']+1e-6)
df['avg_quote_coin_per_tx'] = df['quote_coin_amount_sum']/(df['signature_count']+1e-6)
df[['name_x', 'symbol_x', 'creator']]=df[['name_x', 'symbol_x', 'creator']].astype('category')
df[['bundled_buys', 'dev_balance']] = df[['bundled_buys', 'dev_balance']].astype('float64')

df['fee_per_signature'] = df['fee_sum'] / (df['signature_count'] + 1e-5)
df['gas_used_per_instruction'] = df['gas_used'] / (df['amount_of_instructions'] + 1e-5)
df['virtual_balance_per_signer'] = df['virtual_token_balance_after_sum'] / (df['signing_wallet_nunique'] + 1e-5)
df['base_to_quote_ratio'] = df['base_coin_amount_sum'] / (df['quote_coin_amount_sum'] + 1e-5)
df['provided_to_consumed_gas_ratio'] = df['provided_gas_fee_sum'] / (df['consumed_gas_sum'] + 1e-5)

df['gas_signature'] = df['gas_used'] * df['signature_count']
df['buy_volume'] = df['base_coin_amount_mean'] * df['buy_sum']
df['fee_signal'] = df['fee_mean'] * df['amount_of_instructions']
df['complexity'] = df['amount_of_instructions'] * df['bundled_buys_count']

df['net_fee'] = df['provided_gas_fee_sum'] - df['fee_sum']
df['net_token_balance'] = df['virtual_token_balance_after_sum'] - df['consumed_gas_sum']
df['gas_margin'] = df['provided_gas_limit_sum'] - df['gas_used']
df['slot_diff'] = df['slot_max']-df['slot_min']
df['bundle_structure_len'] = df['bundle_structure'].str.len()
df['bundle_structure'] = df['bundle_structure'].astype('category')

drop_cols1 = ['version','created_at','block_time','slot','block_time_min','block_time_max','slot_max','slot_min']
df = df.drop(columns = drop_cols1,index=1)
df.columns


freq_encode_cols = [
             'creator', 'name_x', 'symbol_x', 'bundle_structure',
            'tx_idx', 'amount_of_instructions', 'amount_of_lookup_reads',
            'amount_of_lookup_writes', 'bundled_buys_count', 'creation_ix_index',
            'pf_program_index', 'direct_pf_invocation', 'signature_count', 'buy_sum',
            'sell_sum', 'signing_wallet_nunique', 'active_days', 'avg_transection_per_day',
            'complexity', 'slot_diff'
        ]
        
label_encode_cols = [
             'creator', 'name_x', 'symbol_x', 'bundle_structure'
        ]
        
agg_main_cols = [
            'gas_used', 'fee_sum', 'base_coin_amount_sum', 'buy_sum', 'sell_sum',
            'consumed_gas_sum', 'provided_gas_fee_sum', 'bundled_buys',
            'bundled_buys_count', 'dev_balance', 'virtual_balance_per_signer',
            'buy_sell_ratio', 'consumed_gas_mean', 'fee_signal', 'gas_signature',
            'virtual_token_balance_after_sum', 'virtual_sol_balance_after_sum',
            'virtual_token_balance_after_mean','base_to_quote_ratio',
            'virtual_sol_balance_after_mean',
            'gas_used_per_instruction', 'gas_margin', 'sell_ratio', 'buy_volume',
            'provided_to_consumed_gas_ratio', 'avg_quote_coin_per_tx',
            'provided_gas_limit_sum', 'net_fee', 'fee_per_signature',
            'net_token_balance', 'pf_program_index','active_hours','active_seconds'
        ]
        
agg_uids = [
            'creator', 'symbol_x', 'bundle_structure', 'name_x'
        ]
        
combined_cols = [
            ('creator', 'name_x'),
 ('creator', 'symbol_x'),
 ('creator', 'bundle_structure'),
 ('creator', 'signature_count'),
 ('creator', 'buy_sum'),
 ('creator', 'sell_sum'),
 ('creator', 'amount_of_lookup_reads'),
 ('creator', 'amount_of_lookup_writes'),
 ('creator', 'amount_of_instructions'),
 ('creator', 'fee_std'),
 ('creator', 'signing_wallet_nunique'),
 ('creator', 'complexity'),
 ('name_x', 'symbol_x'),
 ('name_x', 'signature_count'),
 ('name_x', 'buy_sum'),
 ('name_x', 'sell_sum'),
 ('name_x', 'amount_of_lookup_reads'),
 ('name_x', 'amount_of_lookup_writes'),
 ('name_x', 'amount_of_instructions'),
 ('name_x', 'provided_gas_fee_std'),
 ('name_x', 'signing_wallet_nunique'),
 ('name_x', 'complexity'),
 ('symbol_x', 'signature_count'),
 ('symbol_x', 'buy_sum'),
 ('symbol_x', 'sell_sum'),
 ('symbol_x', 'amount_of_lookup_reads'),
 ('symbol_x', 'amount_of_lookup_writes'),
 ('symbol_x', 'amount_of_instructions'),
 ('symbol_x', 'fee_std'),
 ('symbol_x', 'signing_wallet_nunique'),
 ('bundle_structure', 'signature_count'),
 ('bundle_structure', 'buy_sum'),
 ('bundle_structure', 'sell_sum'),
 ('bundle_structure', 'provided_gas_fee_std'),
 ('signature_count', 'buy_sum'),
 ('signature_count', 'amount_of_instructions'),
 ('signature_count', 'fee_std'),
 ('signature_count', 'signing_wallet_nunique'),
 ('buy_sum', 'fee_std'),
 ('amount_of_lookup_reads', 'amount_of_instructions'),
 ('amount_of_lookup_writes', 'signing_wallet_nunique'),
 ('amount_of_instructions', 'fee_std'),
 ('amount_of_instructions', 'signing_wallet_nunique')
        ]

nunique_cols = [
            'signature_count', 'pf_program_index', 'tx_idx',
            'amount_of_instructions', 'amount_of_lookup_reads',
            'amount_of_lookup_writes', 'bundled_buys_count', 'creation_ix_index',
            'direct_pf_invocation', 'buy_sum', 'sell_sum', 'signing_wallet_nunique',
            'active_days', 'avg_transection_per_day', 'complexity', 'slot_diff'
        ]

nunique_uids = ['creator','bundle_structure','name_x','symbol_x']



def apply_frequency_encoding(X_train, X_test,df_test, cols):
    for col in cols:
        freq_map = X_train[col].value_counts(normalize=True).to_dict()
        freq_map[-1] = -1

        X_train[col + '_FE'] = X_train[col].map(freq_map).astype('float32')
        X_test[col + '_FE'] = X_test[col].map(freq_map).astype('float32')
        df_test[col + '_FE'] = df_test[col].map(freq_map).astype('float32')
        X_train[col + '_FE'].fillna(-1, inplace=True)
        X_test[col + '_FE'].fillna(-1, inplace=True)
        df_test[col + '_FE'].fillna(-1, inplace=True)
    
    return X_train, X_test,df_test

def apply_label_encoding(X_train, X_test,df_test, cols):
    for col in cols:
        unique_vals = X_train[col].dropna().astype(str).unique()
        label_map = {val: idx for idx, val in enumerate(sorted(unique_vals))}
        
        X_train[col] = X_train[col].astype(str).map(label_map).fillna(-1).astype('category')
        X_test[col] = X_test[col].astype(str).map(label_map).fillna(-1).astype('category')
        df_test[col] = df_test[col].astype(str).map(label_map).fillna(-1).astype('category')
        
        
    return X_train, X_test,df_test

def apply_agg_encoding(X_train, X_test,df_test, agg_main_cols, agg_uids):
    for main_col in agg_main_cols:
        for uid in agg_uids:
            for agg_type in ['mean', 'std']:
                new_col = f"{main_col}_{uid}_{agg_type}"
                
                # Compute aggregation on train only
                agg_map = X_train.groupby(uid)[main_col].agg(agg_type).to_dict()
                
                # Map to train and test
                X_train[new_col] = X_train[uid].map(agg_map).astype('float32')
                X_test[new_col] = X_test[uid].map(agg_map).astype('float32')
                df_test[new_col] = df_test[uid].map(agg_map).astype('float32')
                
                # Handle missing values
                X_train[new_col].fillna(-1, inplace=True)
                X_test[new_col].fillna(-1, inplace=True)
                df_test[new_col].fillna(-1, inplace=True)
    
    return X_train, X_test,df_test

def apply_combined_label_encoding(X_train, X_test,df_test, combined_cols):
    for col1, col2 in combined_cols:
        new_col = f"{col1}_{col2}"

        # Create the combined column
        X_train[new_col] = X_train[col1].astype(str) + "_" + X_train[col2].astype(str)
        X_test[new_col] = X_test[col1].astype(str) + "_" + X_test[col2].astype(str)
        df_test[new_col] = df_test[col1].astype(str) + "_" + df_test[col2].astype(str)

        # Get unique values from train only
        unique_vals = X_train[new_col].unique()
        label_mapping = {val: i for i, val in enumerate(unique_vals)}

        # Map to int labels
        X_train[new_col] = X_train[new_col].map(label_mapping).fillna(-1).astype('category')
        X_test[new_col] = X_test[new_col].map(label_mapping).fillna(-1).astype('category')
        df_test[new_col] = df_test[new_col].map(label_mapping).fillna(-1).astype('category')

    return X_train, X_test,df_test

def apply_combined_aggregations(X_train, X_test,df_test, combined_cols, agg_main_cols):
    for col1, col2 in combined_cols:
        combo_col = f"{col1}_{col2}"
        
        # Create combo col
        X_train[combo_col] = X_train[col1].astype(str) + "_" + X_train[col2].astype(str)
        X_test[combo_col] = X_test[col1].astype(str) + "_" + X_test[col2].astype(str)
        df_test[combo_col] = df_test[col1].astype(str) + "_" + df_test[col2].astype(str)
        
        for target in agg_main_cols:
            # Create mapping on train
            agg_map = X_train.groupby(combo_col)[target].mean().to_dict()
            new_col = f"{combo_col}_{target}_mean"
            
            # Map to train and test
            X_train[combo_col] = X_train[combo_col].map(agg_map).astype('float32').fillna(-1)
            X_test[combo_col] = X_test[combo_col].map(agg_map).astype('float32').fillna(-1)
            df_test[combo_col] = df_test[combo_col].map(agg_map).astype('float32').fillna(-1)
            
    return X_train, X_test, df_test

def apply_nunique_aggregations(X_train, X_test,df_test, nunique_cols, nunique_uids):
    for main_col in nunique_cols:
        for uid_col in nunique_uids:
            new_col = f"{uid_col}_{main_col}_ct"
            
            # Create mapping from train only
            agg_map = X_train.groupby(uid_col)[main_col].nunique().to_dict()
            
            # Map to train and test
            X_train[new_col] = X_train[uid_col].map(agg_map).astype('float32').fillna(-1)
            X_test[new_col] = X_test[uid_col].map(agg_map).astype('float32').fillna(-1)
            df_test[new_col] = df_test[uid_col].map(agg_map).astype('float32').fillna(-1)
    
    return X_train, X_test,df_test



def run_all_transformations(X_train, X_test,df_test,
                             freq_encode_cols,
                             label_encode_cols,
                             agg_main_cols,
                             agg_uids,
                             combined_cols,
                             nunique_cols,
                             nunique_uids):
    
    X_train, X_test, df_test = apply_frequency_encoding(X_train, X_test,df_test, freq_encode_cols)
    X_train, X_test, df_test = apply_label_encoding(X_train, X_test,df_test, label_encode_cols)
    X_train, X_test,df_test = apply_agg_encoding(X_train, X_test,df_test, agg_main_cols, agg_uids)
    X_train, X_test,df_test = apply_combined_label_encoding(X_train, X_test,df_test, combined_cols)
    X_train, X_test,df_test = apply_combined_aggregations(X_train, X_test,df_test, combined_cols, agg_main_cols)
    X_train, X_test,df_test = apply_nunique_aggregations(X_train, X_test,df_test, nunique_cols, nunique_uids)

    return X_train, X_test,df_test



df_train = df[df['is_train']==1].drop(columns=['is_train']).reset_index(drop=True)
df_test = df[df['is_train']!=1].drop(columns=['has_graduated','is_train']).reset_index(drop=True)

df_test = df_test.drop_duplicates(subset='mint',keep='first')

cols_to_remove = ['mint']


df_train = df_train.drop(columns = cols_to_remove,axis=1)
df_test = df_test.drop(columns = cols_to_remove,axis=1)


df_train.shape,df_test.shape


top_300 = ['virtual_sol_balance_after_max',
 'virtual_sol_balance_after_last',
 'symbol_x',
 'name_x',
 'quote_coin_amount_buy_sum',
 'base_to_quote_ratio',
 'virtual_token_balance_after_mean',
 'creator',
 'bundle_structure',
 'base_coin_amount_buy_sum',
 'bundled_buys_name_x_std',
 'virtual_balance_per_signer',
 'virtual_sol_balance_after_buy_max',
 'gas_used_per_instruction_name_x_std',
 'dev_balance_name_x_std',
 'virtual_token_balance_after_buy_min',
 'pf_program_index_symbol_x_std',
 'virtual_sol_balance_after_mean',
 'virtual_token_balance_after_sell_mean',
 'gas_used_per_instruction_name_x_mean',
 'virtual_token_balance_after_mean_creator_mean',
 'quote_coin_amount_sell_max',
 'pf_program_index_creator_mean',
 'bundle_structure_FE',
 'virtual_token_balance_after_mean_name_x_std',
 'virtual_token_balance_after_mean_bundle_structure_mean',
 'symbol_x_FE',
 'quote_coin_amount_sum',
 'provided_gas_fee_sum_creator_mean',
 'base_coin_amount_sum',
 'buy_sell_ratio',
 'name_x_FE',
 'base_coin_amount_buy_min',
 'buy_sell_ratio_creator_std',
 'virtual_sol_balance_after_mean_name_x_std',
 'name_x_creation_ix_index_ct',
 'virtual_sol_balance_after_buy_mean',
 'fee_signal_creator_mean',
 'virtual_sol_balance_after_min',
 'gas_used_per_instruction_symbol_x_std',
 'virtual_token_balance_after_sell_max',
 'symbol_x_tx_idx_ct',
 'virtual_sol_balance_after_mean_creator_mean',
 'sell_sum_name_x_mean',
 'pf_program_index_name_x_std',
 'name_x_tx_idx_ct',
 'name_x_active_days_ct',
 'base_to_quote_ratio_creator_mean',
 'virtual_sol_balance_after_sell_min',
 'bundle_structure_tx_idx_ct',
 'pf_program_index_name_x_mean',
 'gas_used_per_instruction_bundle_structure_std',
 'creator_FE',
 'bundle_structure_avg_transection_per_day_ct',
 'gas_used_bundle_structure_std',
 'buy_sell_ratio_name_x_std',
 'base_coin_amount_sum_name_x_mean',
 'active_hours_creator_std',
 'symbol_x_signature_count_ct',
 'quote_coin_amount_buy_min',
 'name_x_complexity_ct',
 'pf_program_index',
 'provided_gas_limit_mean',
 'name_x_avg_transection_per_day_ct',
 'virtual_balance_per_signer_name_x_std',
 'consumed_gas_mean_creator_std',
 'pf_program_index_symbol_x_mean',
 'name_x_buy_sum_ct',
 'symbol_x_avg_transection_per_day_ct',
 'buy_volume',
 'name_x_amount_of_instructions_ct',
 'fee_signal_bundle_structure_mean',
 'name_x_signature_count_ct',
 'name_x_slot_diff_ct',
 'creator_slot_diff_ct',
 'active_hours_creator_mean',
 'symbol_x_active_days_ct',
 'provided_gas_limit_sum_name_x_mean',
 'virtual_token_balance_after_buy_mean',
 'net_fee_creator_mean',
 'fee_signal_name_x_mean',
 'avg_quote_coin_per_tx_creator_mean',
 'dev_balance_name_x_mean',
 'bundle_structure_signature_count_ct',
 'dev_balance',
 'provided_to_consumed_gas_ratio_creator_mean',
 'bundle_structure_len',
 'creator_active_days_ct',
 'buy_sell_ratio_name_x_mean',
 'symbol_x_sell_sum_ct',
 'gas_used_per_instruction_symbol_x_mean',
 'bundled_buys_symbol_x_mean',
 'base_coin_amount_sum_bundle_structure_mean',
 'buy_volume_creator_mean',
 'virtual_sol_balance_after_sum_name_x_std',
 'base_coin_amount_sum_symbol_x_mean',
 'creator_tx_idx_ct',
 'bundled_buys',
 'symbol_x_slot_diff_ct',
 'consumed_gas_mean_creator_mean',
 'slot_diff',
 'virtual_token_balance_after_buy_max',
 'bundle_structure_slot_diff_ct',
 'virtual_sol_balance_after_buy_sum',
 'virtual_sol_balance_after_buy_min',
 'dev_balance_creator_std',
 'active_seconds_creator_mean',
 'symbol_x_buy_sum_ct',
 'creator_buy_sum_ct',
 'symbol_x_signing_wallet_nunique_ct',
 'pf_program_index_FE',
 'sell_ratio',
 'provided_to_consumed_gas_ratio',
 'symbol_x_pf_program_index_ct',
 'provided_gas_fee_mean',
 'active_days',
 'virtual_token_balance_after_mean_name_x_mean',
 'creator_avg_transection_per_day_ct',
 'base_coin_amount_sum_creator_std',
 'dev_balance_symbol_x_std',
 'symbol_x_creation_ix_index_ct',
 'name_x_signing_wallet_nunique_ct',
 'fee_per_signature_name_x_mean',
 'net_fee_name_x_std',
 'buy_sum_name_x_std',
 'gas_margin_creator_mean',
 'bundle_structure_signing_wallet_nunique_ct',
 'buy_volume_name_x_mean',
 'gas_used_name_x_std',
 'active_seconds_creator_std',
 'fee_sum',
 'buy_ratio',
 'amount_of_instructions_FE',
 'virtual_token_balance_after_sum_name_x_mean',
 'fee_std',
 'quote_coin_amount_last',
 'bundled_buys_bundle_structure_mean',
 'buy_sell_ratio_symbol_x_mean',
 'base_to_quote_ratio_name_x_std',
 'bundled_buys_symbol_x_std',
 'gas_used_per_instruction',
 'gas_used_per_instruction_creator_std',
 'buy_volume_name_x_std',
 'buy_sell_ratio_symbol_x_std',
 'provided_gas_fee_sum_name_x_std',
 'sell_ratio_name_x_mean',
 'active_hours',
 'gas_used_per_instruction_bundle_structure_mean',
 'virtual_sol_balance_after_sell_max',
 'provided_gas_fee_std',
 'dev_balance_bundle_structure_mean',
 'bundled_buys_name_x_mean',
 'virtual_balance_per_signer_bundle_structure_mean',
 'virtual_token_balance_after_sell_sum',
 'creator_signature_count_ct',
 'quote_coin_amount_sell_mean',
 'dev_balance_creator_mean',
 'provided_to_consumed_gas_ratio_name_x_mean',
 'fee_signal_symbol_x_mean',
 'fee_sum_name_x_mean',
 'active_hours_symbol_x_std',
 'fee_signal',
 'sell_ratio_name_x_std',
 'active_hours_bundle_structure_mean',
 'net_token_balance_bundle_structure_mean',
 'sell_sum_symbol_x_mean',
 'creator_sell_sum',
 'virtual_balance_per_signer_creator_mean',
 'base_coin_amount_sum_creator_mean',
 'consumed_gas_mean_symbol_x_mean',
 'fee_sum_creator_mean',
 'symbol_x_complexity_ct',
 'fee_mean',
 'buy_sell_ratio_bundle_structure_mean',
 'sell_ratio_bundle_structure_mean',
 'base_coin_amount_mean',
 'bundled_buys_creator_mean',
 'base_coin_amount_buy_mean',
 'provided_gas_limit_sum_creator_std',
 'active_seconds_bundle_structure_mean',
 'virtual_token_balance_after_mean_symbol_x_std',
 'quote_coin_amount_sell_min',
 'fee_sum_name_x_std',
 'avg_quote_coin_per_tx_bundle_structure_mean',
 'virtual_balance_per_signer_name_x_mean',
 'virtual_token_balance_after_mean_bundle_structure_std',
 'pf_program_index_bundle_structure_mean',
 'creator_sell_sum_ct',
 'slot_diff_FE',
 'sell_ratio_symbol_x_mean',
 'consumed_gas_sell_max',
 'fee_per_signature_symbol_x_mean',
 'consumed_gas_mean_symbol_x_std',
 'gas_margin_name_x_std',
 'active_seconds',
 'bundle_structure_active_days_ct',
 'net_token_balance_name_x_mean',
 'sell_sum_name_x_std',
 'gas_used_name_x_mean',
 'gas_used_creator_std',
 'buy_volume_symbol_x_mean',
 'consumed_gas_buy_min',
 'base_coin_amount_sell_mean',
 'consumed_gas_last',
 'net_fee_symbol_x_mean',
 'gas_margin_symbol_x_mean',
 'dev_balance_symbol_x_mean',
 'consumed_gas_buy_mean',
 'virtual_sol_balance_after_mean_bundle_structure_mean',
 'gas_signature',
 'provided_to_consumed_gas_ratio_creator_std',
 'virtual_sol_balance_after_sum_symbol_x_std',
 'gas_signature_creator_std',
 'base_to_quote_ratio_symbol_x_mean',
 'avg_quote_coin_per_tx_name_x_mean',
 'avg_transection_per_day',
 'fee_sum_symbol_x_mean',
 'fee_per_signature_creator_mean',
 'provided_to_consumed_gas_ratio_bundle_structure_mean',
 'provided_gas_limit_sum',
 'virtual_token_balance_after_sum_bundle_structure_mean',
 'sell_ratio_symbol_x_std',
 'base_to_quote_ratio_creator_std',
 'name_x_sell_sum_ct',
 'provided_gas_limit_sum_symbol_x_mean',
 'avg_base_coin_per_tx',
 'consumed_gas_mean_name_x_std',
 'provided_gas_limit_sum_name_x_std',
 'virtual_sol_balance_after_sell_sum',
 'base_coin_amount_max',
 'gas_margin_creator_std',
 'virtual_token_balance_after_sum_symbol_x_mean',
 'consumed_gas_buy_sum',
 'virtual_balance_per_signer_symbol_x_mean',
 'provided_gas_limit_sum_creator_mean',
 'buy_sum_bundle_structure_std',
 'virtual_sol_balance_after_sell_mean',
 'net_fee',
 'sell_ratio_creator_std',
 'provided_gas_fee_sum_symbol_x_mean',
 'active_seconds_symbol_x_std',
 'avg_transection_per_day_FE',
 'quote_coin_amount_min',
 'provided_gas_fee_sum',
 'consumed_gas_sum_creator_std',
 'virtual_sol_balance_after_mean_symbol_x_mean',
 'active_hours_symbol_x_mean',
 'buy_volume_bundle_structure_std',
 'virtual_balance_per_signer_creator_std',
 'fee_signal_symbol_x_std',
 'gas_margin',
 'virtual_sol_balance_after_mean_name_x_mean',
 'fee_max',
 'provided_to_consumed_gas_ratio_symbol_x_std',
 'tx_idx_FE',
 'fee_sum_creator_std',
 'fee_per_signature',
 'quote_coin_amount_max',
 'gas_used_creator_mean',
 'quote_coin_amount_sell_sum',
 'fee_sum_symbol_x_std',
 'gas_signature_creator_mean',
 'provided_gas_fee_sum_creator_std',
 'creator_signing_wallet_nunique_ct',
 'consumed_gas_sell_mean',
 'gas_margin_name_x_mean',
 'virtual_token_balance_after_sum_symbol_x_std',
 'base_coin_amount_sum_symbol_x_std',
 'active_hours_name_x_mean',
 'consumed_gas_min',
 'consumed_gas_mean_name_x_mean',
 'base_to_quote_ratio_symbol_x_std',
 'buy_sell_ratio_creator_mean',
 'base_coin_amount_sell_max',
 'quote_coin_amount_buy_mean',
 'base_coin_amount_sell_min',
 'net_fee_bundle_structure_mean',
 'virtual_sol_balance_after_sum_symbol_x_mean',
 'amount_of_instructions_signing_wallet_nunique',
 'bundle_structure_buy_sum_ct',
 'tx_idx',
 'avg_quote_coin_per_tx_creator_std',
 'net_token_balance_creator_std',
 'gas_used_per_instruction_creator_mean',
 'base_coin_amount_sum_name_x_std',
 'consumed_gas_sum_bundle_structure_mean',
 'buy_sum_symbol_x_std',
 'virtual_balance_per_signer_symbol_x_std',
 'symbol_x_signature_count',
 'fee_per_signature_symbol_x_std',
 'fee_per_signature_name_x_std',
 'gas_signature_name_x_std',
 'dev_balance_bundle_structure_std',
 'quote_coin_amount_mean',
 'consumed_gas_std',
 'quote_coin_amount_buy_max',
 'virtual_token_balance_after_sell_min',
 'buy_sum_bundle_structure_mean',
 'symbol_x_fee_std',
 'buy_sum_symbol_x_mean']


import time
folds = 5
oof_preds = np.zeros(df_train.shape[0])
test_preds = np.zeros(test.shape[0])

SKFold = StratifiedKFold(n_splits=folds,shuffle=True,random_state = 42)

for i, (train_idx,val_idx) in enumerate(SKFold.split(df_train,df_train[target]),1):
    print(f"\n{'='*30}")
    print(f"ğŸ› ï¸�  Fold {i} training starts!")
    print(f"{'='*30}")
    start_time = time.time()
    
    X = df_train.drop(target,axis=1) 
    y = df_train[target]
    df_test_fold = df_test.copy()
    x_train,x_val = X.iloc[train_idx],X.iloc[val_idx]
    y_train,y_val = y[train_idx],y[val_idx]
    x_test = df_test.copy()

    # x_train,x_val,x_test = x_train_raw.copy(),x_val_raw.copy(),x_test_raw.copy()

    x_train,x_val,df_test_fold = run_all_transformations(x_train, x_val,x_test,
                             freq_encode_cols,
                             label_encode_cols,
                             agg_main_cols,
                             agg_uids,
                             combined_cols,
                             nunique_cols,
                             nunique_uids)
    x_train = x_train[top_300]
    x_val = x_val[top_300]
    x_test = x_test[top_300]

    
    model = LGBMClassifier(
        n_estimators=2000,
        device = "cpu",
        max_depth=16, 
        learning_rate=0.02, 
        subsample=0.8,
        colsample_bytree=0.4,  
        objective='binary',         # important: since it's binary classification
        boosting_type='gbdt',
        metric='binary_logloss',     # evaluate with logloss
        reg_alpha=0.2,   # L1 regularization (you can try 0.5 or tune it)
        reg_lambda=0.2,   # L2 regularization (you can try 0.5 or tune it)
        verbose=-1,
        enable_categorical = True)

    model.fit(x_train,y_train,
             eval_set=[(x_val, y_val)],          # give validation data
            eval_metric='binary_logloss',        # evaluation metric
            callbacks=[
            early_stopping(stopping_rounds=200),   # early stopping
            log_evaluation(500)]
             ) 

    val_preds = model.predict_proba(x_val)[:,1]
    oof_preds[val_idx] = val_preds

    preds = model.predict_proba(x_test)[:,1]
    test_preds += preds

    loss = log_loss(y_val,val_preds)
    elapsed = (time.time() - start_time) / 60  # convert to minutes

    print(f"\nâœ… Fold {i} finished.")
    print(f"Log loss for Fold {i}: {loss}")
    print(f"Time taken: {elapsed:.2f} minutes")
    print(f"{'='*30}\n")
    del  x_train,x_val,y_train,y_val,X,y,x_test
    gc.collect()
        
test_preds/=folds
oof_score = log_loss(df_train[target],oof_preds)
print()
print(f'Overall OOF log loss: {oof_score}')


# importances = model.booster_.feature_importance(importance_type='gain')
# feature_names = model.booster_.feature_name()

# # 2. Create a DataFrame of features and their importance
# feature_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

# # 3. Sort by importance and get the top 300 features
# top_300 = feature_df.sort_values(by='importance', ascending=False)['feature'].head(300).tolist()


sub_preds = test_preds


sub_df = pd.read_csv('/kaggle/input/solana-skill-sprint-memcoin-graduation/sample_submission.csv')
sub_df['has_graduated'] = sub_preds


sub_df.to_csv('/kaggle/working/pump_fun_probablies.csv',index = False)


# Column Name | Simple Meaning
# mint | Unique ID of the token (like token address)
# slot_min | Minimum blockchain slot when token activity started (block number type)
# has_graduated | Target label (1 = graduated, 0 = not graduated)
# is_train | 1 if it's part of training data, 0 if test data
# name_x | Name of the token (example: "FunnyCoin")
# symbol_x | Symbol of the token (example: "FCN")
# created_at | Time when the token was created
# block_time | Average time of transactions related to token (UNIX timestamp)
# tx_idx | Transaction index (maybe which transaction number)
# creator | Wallet address of the token creator
# gas_used | Total gas consumed in transactions (execution cost)
# amount_of_instructions | Number of blockchain instructions inside the transaction
# amount_of_lookup_reads | How many times program read blockchain data
# amount_of_lookup_writes | How many times program wrote blockchain data
# bundled_buys | Whether multiple buys were bundled together (bulk buy)
# bundled_buys_count | Count of bundled buy transactions
# dev_balance | Developer's balance (maybe after listing or transactions)
# creation_ix_index | Index of the instruction which created token
# pf_program_index | Program index related to pumpfun platform
# direct_pf_invocation | Whether pumpfun was directly called in transaction (1 = Yes, 0 = No)
# version | Version of the smart contract / program used
# signature_count | Number of signatures in transactions (how many approvals)
# buy_sum | Total amount of buy transactions
# sell_sum | Total amount of sell transactions
# base_coin_amount_sum | Sum of base coin involved (example: total SOL involved)
# base_coin_amount_mean | Average base coin amount per transaction
# quote_coin_amount_sum | Sum of quote coin involved (example: USDC maybe)
# quote_coin_amount_mean | Average quote coin amount per transaction
# fee_sum | Total fees paid (transaction fees + platform fees)
# fee_mean | Average fee per transaction
# provided_gas_fee_sum | Total gas fee provided for transactions
# provided_gas_fee_mean | Average gas fee provided
# provided_gas_limit_sum | Sum of gas limits set by transactions
# provided_gas_limit_mean | Average gas limit set
# consumed_gas_sum | Sum of actual gas consumed
# consumed_gas_mean | Average gas consumed
# block_time_min | Minimum block time (earliest transaction time)
# block_time_max | Maximum block time (latest transaction time)
# signing_wallet_nunique | Number of unique wallets signing transactions
# virtual_token_balance_after_sum | Sum of token balances after transactions
# virtual_token_balance_after_mean | Average token balance after transaction
# virtual_sol_balance_after_sum | Sum of SOL balance after transactions
# virtual_sol_balance_after_mean | Average SOL balance after transaction
# buy_sell_ratio | Ratio of buys vs sells (buy_sum / sell_sum)
# buy_ratio | Only buy side ratio (extra feature)
# sell_ratio | Only sell side ratio (extra feature)
# active_days | How many different days token was active
# avg_transection_per_day | Average number of transactions per day
# avg_base_coin_per_tx | Average base coin used per transaction (looks like typo: transection â†’ transaction)
# avg_quote_coin_per_tx | Average quote coin used per transaction




