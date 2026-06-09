# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import os
import gc

def optimize_memory(df, verbose=True):
    """
    Optimize memory usage by downcasting numeric types where possible.
    """

    if verbose:
        start_mem = df.memory_usage().sum() / 1024**2
        print(f'Memory usage before optimization: {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
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
                # For float types, we'll use float32 instead of float64
                # This is usually sufficient for ML models
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        print(f'Memory usage after optimization: {end_mem:.2f} MB')
        print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df

def feature_engineering(df):
    df['exp_302P289M125'] = np.exp(df['X302'] + df['X289'] - df['X125'])
    df['289xexp_289M125'] = df['X289'] * np.exp(df['X289'] - df['X125'])
    df['385xexp_289M125'] = df['X385'] * np.exp(df['X289'] - df['X125'])
    
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-10)
    df['log_volume'] = np.log1p(df['volume'])

    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-10)
    
    df['ask_buy_interaction_x_X293']=df['X293']*df['ask_buy_interaction']
    
     # Price Pressure Indicators
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
    df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
    
    # Liquidity Depth Measures
    df['total_depth'] = df['bid_qty'] + df['ask_qty']
    df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['log_depth'] = np.log1p(df['total_depth'])
    
    # Order Flow Toxicity Proxies
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['flow_toxicity'] = np.abs(df['order_flow_imbalance']) * df['volume']
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    
    # Market Activity Indicators
    df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + 1e-10)
    df['activity_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-10)
    df['log_buy_qty'] = np.log1p(df['buy_qty'])
    df['log_sell_qty'] = np.log1p(df['sell_qty'])
    df['log_bid_qty'] = np.log1p(df['bid_qty'])
    df['log_ask_qty'] = np.log1p(df['ask_qty'])
    
    # Microstructure Volatility Proxies
    df['realized_spread_proxy'] = 2 * np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['price_impact_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10)
    df['quote_volatility_proxy'] = np.abs(df['depth_imbalance'])
    
    # Complex Interaction Terms
    df['flow_depth_interaction'] = df['net_order_flow'] * df['total_depth']
    df['imbalance_volume_interaction'] = df['order_flow_imbalance'] * df['volume']
    df['depth_volume_interaction'] = df['total_depth'] * df['volume']
    df['buy_sell_spread'] = np.abs(df['buy_qty'] - df['sell_qty'])
    df['bid_ask_spread'] = np.abs(df['bid_qty'] - df['ask_qty'])
    
    # Information Asymmetry Measures
    df['trade_informativeness'] = df['net_order_flow'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['execution_shortfall_proxy'] = df['buy_sell_spread'] / (df['volume'] + 1e-10)
    df['adverse_selection_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10) * df['volume']
    
    # Market Efficiency Indicators
    df['fill_probability'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['execution_rate'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_efficiency'] = df['volume'] / (df['bid_ask_spread'] + 1e-10)
    
    # Non-linear Transformations
    df['sqrt_volume'] = np.sqrt(df['volume'])
    df['sqrt_depth'] = np.sqrt(df['total_depth'])
    df['volume_squared'] = df['volume'] ** 2
    df['imbalance_squared'] = df['order_flow_imbalance'] ** 2
    
    # Relative Measures
    df['bid_ratio'] = df['bid_qty'] / (df['total_depth'] + 1e-10)
    df['ask_ratio'] = df['ask_qty'] / (df['total_depth'] + 1e-10)
    df['buy_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['sell_ratio'] = df['sell_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    
    # Market Stress Indicators
    df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_stress'] = df['volume'] / (df['total_depth'] + 1e-10) * np.abs(df['order_flow_imbalance'])
    df['depth_depletion'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    
    # Directional Indicators
    df['net_buying_ratio'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['directional_volume'] = df['net_order_flow'] * np.log1p(df['volume'])
    df['signed_volume'] = np.sign(df['net_order_flow']) * df['volume']

    #etc
    # df['sqrt_volume_div_log_volume'] = df['sqrt_volume'] / (df['log_volume'] + 1e-6)
    # df['sqrt_volume_div_activity_intensity'] = df['sqrt_volume'] / (df['activity_intensity'] + 1e-6)
    # df['sqrt_volume_mul_fill_probability'] = df['sqrt_volume'] * df['fill_probability']
    # df['volume_div_sqrt_volume'] = df['volume'] / (df['sqrt_volume'] + 1e-6)
    # df['sqrt_volume_div_fill_probability'] = df['sqrt_volume'] / (df['fill_probability'] + 1e-6)
    # df['sqrt_volume_mul_activity_intensity'] = df['sqrt_volume'] * df['activity_intensity']
    # df['sqrt_volume_div_log_sell_qty'] = df['sqrt_volume'] / (df['log_sell_qty'] + 1e-6)
    # df['log_buy_qty_mul_sqrt_volume'] = df['log_buy_qty'] * df['sqrt_volume']
    # df['sqrt_volume_mul_log_buy_qty'] = df['sqrt_volume'] * df['log_buy_qty']
    # df['log_volume_mul_sqrt_volume'] = df['log_volume'] * df['sqrt_volume']
    
    # df['log_sell_qty_mul_X598'] = df['log_sell_qty'] * df['X598']
    # df['log_buy_qty_mul_X598'] = df['log_buy_qty'] * df['X598']
    # df['log_volume_mul_X598'] = df['log_volume'] * df['X598']
    
    # # df['sqrt_volume_mul_X856'] = df['sqrt_volume'] * df['X856']
    
    # df['log_sell_qty_mul_X302'] = df['log_sell_qty'] * df['X302']
    # df['log_volume_mul_X302'] = df['log_volume'] * df['X302']
    # df['log_buy_qty_mul_X302'] = df['log_buy_qty'] * df['X302']
    
    # df['log_sell_qty_mul_X292'] = df['log_sell_qty'] * df['X292']

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df

def preprocess_data_chunked(raw_df, chunk_size=10):
    """
    Preprocess data with memory-efficient chunked lag creation.
    """
    assert len(raw_df.shape) == 2

    y = raw_df['label'].to_numpy().astype(np.float32)  # Use float32 for labels
    assert y.shape == (raw_df.shape[0],)

    # Original features
    cols = [
        'X363', 'X405', 'X321',
        'X175', 'X179', 'X137', 'X197', 'X22', 'X40', 'X181',
        'X28', 'X169', 'X198', 'X173',
        'X338', 'X288', 'X385', 'X344', 'X427', 'X587', 'X450',
        'X97', 'X52', 'X444',
        'X598', 'X379', 'X696', 'X297', 'X138',
        'X572', 'X343', 'X586', 'X466', 'X438', 'X452', 'X459',
        'X435', 'X386', 'X55', 'X341', 'X683', 'X428', 'X605',
        'X445', 'X272', 'X180', 'X593', 'X680',
        'X686', 'X692', 'X695',
        "X603", "X674", "X421", "X333",
        "X415", "X345", "X174", "X302", "X178", "X168", "X612",
        'X298', 'X45', 'X46', 'X39', 'X752', 'X759', 'X41', 'X42',
        "buy_qty", "sell_qty", "volume",
        "bid_qty", "ask_qty",
        'X465','X153','X289','X125','X21',"X293",'X540','X493',
        'X425',"X292", "X532",
    ]
    
    # Add new top important features
    new_features = [
        'X758',  # Importance: 0.0260, Consistency: 83.3%
        'X296',  # Importance: 0.0170, Consistency: 66.7%
        'X611',  # Importance: 0.0133, Consistency: 66.7%
        'X780',  # Importance: 0.0084, Consistency: 100.0%
        'X451',  # Importance: 0.0449, Consistency: 16.7%
        'X25',   # Importance: 0.0148, Consistency: 50.0%
        'X591',  # Importance: 0.0138, Consistency: 50.0%
        'X363', 'X321', 'X405', 'X730', 'X523', 'X756', 'X589', 'X462', 'X779', 'log_liquidity',
        'X25', 'X532', 'X520', 'X329', 'X383', 'X751', 'X535', 'X639', 'X596', 'X761',
        'X145', 'X709', 'X173', 'X245', 'X168', 'X171', 'X241', 'X31', 'X105', 'X63',
        'X263', 'X426', 'X286', 'X357', 'X399', 'X315', 'X468', 'X131', 'X647', 'log_spread',
        'X752', 'X254', 'X592', 'X733', 'X636', 'X394', 'X527', 'X180', 'X367', 'X38',
        'X634', 'X718', 'X387', 'X429', 'X345', 'X344', 'X253', 'X469', 'X446', 'X125',
        'X760', 'X186', 'X711', 'X150', 'X661', 'X215', 'X403', 'X141', 'X771', 'X453',
        'X401', 'X629', 'X616', 'X281', 'X432', 'X283', 'X244', 'X440', 'X430', 'X382',
        'X175', 'X95', 'X444', 'X189', 'X55', 'X605', 'X663', 'X194', 'X439', 'X670',
        'X483', 'X163', 'X376', 'X71', 'X650', 'X203', 'X8', 'X624', 'X160', 'X100',
        'X14', 'X511', 'X59', 'X302', 'X81', 'X325', 'X514', 'X649', 'X447', 'X538',
        'X443', 'X39', 'X343', 'X12', 'X678', 'X775', 'X498', 'X249', 'X42', 'X384',
        'kyle_lambda', 'X349', 'X356', 'X2', 'X250', 'X397', 'X685', 'X568', 'X136', 'X496',
        'X53', 'X66', 'X374', 'X590', 'X668', 'X585', 'X677', 'X667', 'X530', 'X28',
        'X64', 'X407', 'X494', 'X770', 'X710', 'X526', 'X644', 'X167', 'X190', 'X723',
        'X33', 'X579', 'X206',
    ]

    # ADDITIONAL features requested by user
    additional_features = [
        'X525', 'X267', 'X166', 'X719', 'X489', 'X758', 'X652', 'X433', 'X778', 'X428',
        'X617', 'X259', 'X633', 'X565', 'X364', 'depth_ratio', 'X550', 'X687', 'X610', 'X599',
        'X717', 'X587', 'X143', 'X506', 'X546', 'X505', 'X159', 'X574', 'X278', 'X458',
        'X1', 'X749', 'X155', 'X651', 'X470', 'X580', 'X445', 'X373', 'X82', 'X607',
        'X298', 'X221', 'X388', 'X120', 'X391', 'X23', 'X679', 'X377', 'X767', 'X755',
        'X566', 'X424', 'X438', 'X198', 'X300', 'X268', 'X434', 'X290', 'X368', 'X464',
        'X119', 'X197', 'X597', 'X157', 'X485', 'X127', 'X101', 'X533', 'X235', 'X712',
        'X154', 'X239', 'X10', 'X420', 'X449', 'X740', 'X227', 'X36', 'X358', 'X551',
        'X528', 'X285', 'X335', 'X152', 'X110', 'X68', 'X713', 'X402', 'X370', 'X735',
        'X200', 'X331', 'X473', 'X162', 'X213', 'X322', 'X289', 'X477', 'X113', 'X560',
        'X672', 'X621', 'X682', 'X5', 'X72', 'X44', 'X419', 'buy_pressure', 'X242', 'volume',
        'X472', 'X332', 'X441', 'buy_sell_ratio', 'pressure_ratio', 'X508', 'X594', 'X191',
        'X261', 'X603', 'net_pressure', 'order_flow_imbalance', 'sell_pressure', 'X240', 'X673',
        'X608', 'X509', 'X165', 'X720', 'X314', 'X522', 'X531', 'X625', 'bid_depth_ratio',
        'X435', 'X293', 'X486', 'price_efficiency', 'X716', 'X627', 'X626', 'X169', 'X613',
        'X680', 'X544', 'X115', 'X307', 'X665', 'X465', 'X347', 'X728', 'X70', 'log_volume',
        'X340', 'X459', 'X56', 'X395', 'X354', 'X51', 'X732', 'X247', 'X324', 'X316',
        'X76', 'X341', 'X739', 'X601', 'X386', 'X683', 'X149', 'X193', 'X628', 'X309',
        'X351', 'X393'
    ]
    extended_features = [
        # 'X727', 'X427', 'X288', 'X721', 'X312', 'X421', 'X471', 'X573', 'X780', 'X255',
        # 'X144', 'X299', 'X301', 'X563', 'X737', 'X702', 'ask_qty', 'X507', 'X306', 'X501',
        # 'X303', 'amihud_illiquidity', 'X586', 'X43', 'X517', 'X248', 'X137', 'X757', 'X196',
        # 'X777', 'X280', 'X266', 'X689', 'X294', 'X492', 'X555', 'X731', 'X262', 'X576',
        # 'X13', 'X518', 'X502', 'X558', 'pin_proxy', 'X6', 'X602', 'X695', 'X703', 'X413',
        # 'X660', 'X37', 'X15', 'X310', 'X512', 'X362', 'X631', 'X214', 'X562', 'X488',
        # 'X510', 'X256', 'X35', 'X128', 'X86', 'X170', 'X30', 'X265', 'X323', 'X559',
        # 'X348', 'X130', 'X529', 'X20', 'X4', 'X90', 'X192', 'X91', 'X582', 'X99',
        # 'X24', 'X317', 'X707', 'X653', 'X519', 'X557', 'X371', 'X415', 'X84', 'X83',
        # 'order_toxicity', 'X360', 'X111', 'X699', 'X187', 'X591', 'X637', 'X567', 'X577',
        # 'X313', 'X60', 'X671', 'X698', 'X701', 'X725', 'X292', 'X638', 'X741', 'X379',
        # 'X700', 'X614', 'X676', 'X516', 'X697', 'X611', 'X311', 'X615', 'X706', 'X466',
        # 'X571', 'X451', 'X17', 'X584', 'X436', 'X305', 'liquidity_consumption', 'X34', 'X282',
        # 'X681', 'X7', 'X208', 'X41', 'X536', 'X548', 'X296', 'X776', 'X87', 'X40',
        # 'X570', 'X539', 'X474', 'X753', 'X425', 'X217', 'X199', 'X18', 'X609', 'X21',
        # 'X277', 'X279', 'X326', 'X540', 'X688', 'X553', 'X452', 'X738', 'X183', 'X759',
        # 'bid_ask_ratio', 'X495', 'volume_participation', 'X715', 'X385', 'X291', 'X409', 'X112',
        # 'X693', 'X102', 'X318', 'X705', 'X556', 'X547'
    ]

    # Combine all features and remove duplicates while preserving order
    cols = list(dict.fromkeys(cols + new_features+additional_features))
    
    # Check which features actually exist in the dataframe
    available_cols = [col for col in cols if col in raw_df.columns]
    missing_cols = [col for col in cols if col not in raw_df.columns]
    
    if missing_cols:
        print(f"Warning: The following features are not in the dataset: {missing_cols}")
    
    print(f"Using {len(available_cols)} features out of {len(cols)} requested")

    # Select and optimize base features
    # print("feature engineering...")
    # df = feature_engineering(df)
    df = raw_df[available_cols].copy()
    print(df.shape)
    # df = optimize_memory(df, verbose=True)
    
    assert df.isna().sum().sum() == 0

    # Extended lag features
    lag_periods = [
        1, 3, 5, 6, 7, 8, 9,  # Very short-term (1-10)
        12, 15, 18, 20, 25, 30,          # Short-term (12-30)
        # 40, 50, 60, 75, 90,              # Medium-term (40-90)
        # 120,150,
        45, 60, 90, 120, 180,
        # 180, 222,              # Long-term (2-4 hours if minute data)
        365,800, 1600
        # 480, 600,              # Longer-term (6-12 hours)
        # 960, 1440,                 # Very long-term (16-24 hours)
        # 2880                     # Multi-day (2-3 days)
    ]
    
    # Process lags in chunks to manage memory
    print("Creating lagged features in chunks...")
    
    # Start with base features
    result_df = df.copy()
    
    # Process lags in chunks
    for i in range(0, len(lag_periods), chunk_size):
        chunk_lags = lag_periods[i:i+chunk_size]
        print(f"  Processing lags: {chunk_lags}")
        
        # Create lagged features for this chunk
        chunk_dfs = []
        for lag in chunk_lags:
            lagged = df.shift(-lag).add_suffix(f'_lead_{lag}')
            lagged = lagged.fillna(0.0).astype(np.float32)  # Fill NaN and convert to float32
            chunk_dfs.append(lagged)
        
        # Concatenate chunk
        if chunk_dfs:
            chunk_combined = pd.concat(chunk_dfs, axis=1)
            result_df = pd.concat([result_df, chunk_combined], axis=1)
            
            # Clean up
            del chunk_dfs, chunk_combined
            gc.collect()
    
    # Final optimization
    # result_df = optimize_memory(result_df, verbose=True)
    # print("feature engineering...")
    # result_df = feature_engineering(result_df)
    
    assert 'label' not in result_df.columns
    assert raw_df.shape[0] == result_df.shape[0] and (raw_df.index == result_df.index).all()
    assert result_df.isna().sum().sum() == 0
    assert result_df.shape[0] == y.shape[0]
    
    print(f"Final feature count: {result_df.shape[1]}")
    
    return result_df, y




# Set memory-efficient options for pandas
pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning
pd.options.display.max_columns = None

# Load and preprocess training data
print("Loading training data...")
train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')

# Display available columns to verify feature existence
print(f"\nTotal columns in training data: {len(train_df.columns)}")
print(f"Sample columns: {list(train_df.columns[:20])}")

# Optimize memory for the raw training data
# print("\nOptimizing memory for raw training data...")
# train_df = optimize_memory(train_df, verbose=True)

X_train, y_train = preprocess_data_chunked(train_df, chunk_size=10)

# Clean up training dataframe
del train_df
gc.collect()

print(f"\nTraining data shape: X={X_train.shape}, y={y_train.shape}")




# XGB Model - COMMENTED OUT
# from sklearn.model_selection import KFold
# from xgboost import XGBRegressor

# LightGBM Model
from sklearn.model_selection import KFold
import lightgbm as lgb
import importlib
import pandas as pd
importlib.reload(pd)


# XGB_PARAMS = {
#     "tree_method": "hist",
#     "device": "gpu",
#     "colsample_bylevel": 0.4778,
#     "colsample_bynode": 0.3628,
#     "colsample_bytree": 0.7107,
#     "gamma": 1.7095,
#     "learning_rate": 0.02213,
#     "max_depth": 20,
#     "max_leaves": 12,
#     "min_child_weight": 16,
#     "n_estimators": 1500,
#     "subsample": 0.06567,
#     "reg_alpha": 39.3524,
#     "reg_lambda": 75.4484,
#     "verbosity": 0,
#     "random_state": 42,
#     "n_jobs": -1,
#     "early_stopping_rounds":100
# }

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
    "random_state": 42,
    "n_jobs": -1,
    "device": "gpu"
}
models = []

kf = KFold(n_splits=5, shuffle=False)
for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train), start=1):
    print(f'*** ----------- FOLD {fold} ----------- ***')
    # X_val = X_train.iloc[valid_idx]
    # X_trn = X_train.iloc[train_idx]
    # y_val = y_train[valid_idx]
    # y_trn = y_train[train_idx]
    
    # train_data = lgb.Dataset(X_trn, label=y_trn)
    # valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    # print('start train')
    # model = lgb.train(
    #     LGBM_PARAMS,
    #     train_data,
    #     valid_sets=[valid_data],
    #     num_boost_round=1000,
    #     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
    # )
    model = lgb.Booster(model_file=f'/kaggle/input/drw-lgbm-models/lgb_fold{fold}.txt')
    models.append(model)

# # Clean up training data
# del X_train, y_train
# gc.collect()



train_df


from tqdm import tqdm
# Load test data
print("\nLoading test data...")
test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

# Optimize memory for test data
# print("\nOptimizing memory for raw test data...")
# test_df = optimize_memory(test_df, verbose=True)

# Try to load precomputed timestamp reconstruction data
timestamp_recon_path = '/kaggle/input/the-order-of-the-test-rows-2/closest_rows.csv'
use_timestamp_reconstruction = os.path.exists(timestamp_recon_path)

if use_timestamp_reconstruction:
    print("Found timestamp reconstruction file, loading...")
    
    # Load precomputed timestamp reconstruction data
    t = pd.Series(pd.read_csv(timestamp_recon_path)['0'].to_numpy())
    assert t.shape == (test_df.shape[0],)
    print('Reconstructed timestamps share:', len(t[t >= 0]) / len(t))

    # Visualize the reconstructed timestamps
    plt.figure(figsize=(16, 4))
    plt.plot(t.sort_values().to_numpy())
    plt.title('Sorted Reconstructed Timestamps')
    plt.show()

    plt.figure(figsize=(16, 4))
    plt.plot(t[t >= 0].sort_values().iloc[:1000].to_numpy())
    plt.axhline(10080, color='r', linestyle='--')
    plt.title('First 1000 Valid Reconstructed Timestamps')
    plt.show()

    # Process timestamp reconstruction
    t -= 10080
    t[t < 0] = 538149

    t = t.sort_values()
    t[t <= len(t)] = np.arange(t[t <= len(t)].shape[0])
    t = t.sort_index()

    t = pd.Series(np.arange(538150), index=t.to_numpy()).sort_index()

    # Visualize test data before sorting
    if 'X656' in test_df.columns:
        plt.figure(figsize=(16, 4))
        plt.plot(test_df['X656'].to_numpy())
        plt.title('Test Data Feature X656 - Before Sorting')
        plt.show()

    # Sort test dataset by reconstructed time order
    test_df = test_df.iloc[t.to_numpy()]

    # Visualize test data after sorting
    if 'X656' in test_df.columns:
        plt.figure(figsize=(16, 4))
        plt.plot(test_df['X656'].to_numpy())
        plt.title('Test Data Feature X656 - After Sorting')
        plt.show()
else:
    print("WARNING: Timestamp reconstruction file not found!")
    print(f"Expected path: {timestamp_recon_path}")
    print("Proceeding without timestamp reconstruction...")
    print("This may significantly impact model performance since lagged features assume temporal order.")
    
    t = pd.Series(np.arange(len(test_df)))

# Preprocess test data
print("\nPreprocessing test data...")

X_test, _ = preprocess_data_chunked(test_df, chunk_size=10)

# Clean up test dataframe
del test_df
gc.collect()

print(f"Test data shape: {X_test.shape}")

# LightGBM Make predictions in batches to save memory
print("\nMaking predictions...")
batch_size = 100000
n_samples = X_test.shape[0]
y_pred = np.zeros(n_samples, dtype=np.float32)

for i in range(0, n_samples, batch_size):
    end_idx = min(i + batch_size, n_samples)
    print(f"  Predicting batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}")
    preds = []
    for j in tqdm([0,1,2,3,4]):
        preds.append(models[j].predict(X_test.iloc[i:end_idx], num_iteration=models[j].best_iteration).astype(np.float32))
    y_pred[i:end_idx] = np.mean(preds,0)
    # y_pred[i:end_idx] = model.predict(X_test.iloc[i:end_idx]).astype(np.float32)

# Clean up test features
del X_test
gc.collect()

# Display prediction statistics
print("\nPrediction statistics:")
print(pd.Series(y_pred).describe())

# Plot cumulative predictions
plt.figure(figsize=(16, 4))
plt.plot(np.cumsum(y_pred))
plt.title('Cumulative Predictions')
plt.xlabel('Sample Index')
plt.ylabel('Cumulative Sum')
plt.grid(True, alpha=0.3)
plt.show()

# Plot prediction distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(y_pred, bins=100, alpha=0.7, edgecolor='black')
plt.title('Prediction Distribution')
plt.xlabel('Predicted Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.plot(y_pred[:1000])
plt.title('First 1000 Predictions')
plt.xlabel('Sample Index')
plt.ylabel('Predicted Value')
plt.tight_layout()
plt.show()

# Prepare submission
print("\nPreparing submission...")
submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')

if use_timestamp_reconstruction:
    # Reorder submission to match original test order
    submission = submission.iloc[t.to_numpy()]
    submission['prediction'] = y_pred
    submission = submission.sort_index()
else:
    # If no timestamp reconstruction, just use predictions in order
    submission['prediction'] = y_pred

# Save submission
submission.to_csv('submission_lgbm.csv', index=False)
print("Submission saved to 'submission_lgbm.csv'")

# Display submission
print("\nSubmission preview:")
print(submission.head())
print(f"\nSubmission shape: {submission.shape}")
print(f"Prediction range: [{submission['prediction'].min():.6f}, {submission['prediction'].max():.6f}]")

# Final memory cleanup
gc.collect()
print("\nDone!")


import pandas as pd
# sub1 = pd.read_csv("submission_xgb.csv",index_col=None)  # Changed from XGB to LGBM
# sub2 = pd.read_csv("submission_sgd.csv",index_col=None)
# sub3 = pd.read_csv("submission_turkish.csv",index_col=None)
# sub4 = pd.read_csv("submission_iblend.csv",index_col=None)
# sub5 = pd.read_csv("submission_DL.csv",index_col=None)
# sub6 = pd.read_csv("submission_ensemble.csv",index_col=None)
sub7 = pd.read_csv("submission_lgbm.csv",index_col=None)
sub8 = pd.read_csv("/kaggle/input/drw-blend-h-v-remix-higher-changepoint/submission_prophet_enhanced.csv",index_col=None)
sub9 = pd.read_csv("")


sub1['prediction'] = sub7['prediction'] * 0.6 + sub8['prediction'] * 0.4
sub1.to_csv("submission.csv",index=False)

