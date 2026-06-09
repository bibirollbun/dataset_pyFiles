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


import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import category_encoders as ce
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_curve, roc_auc_score

import warnings
warnings.filterwarnings('ignore')


# path of files
base_path = '/kaggle/input/pumpfun-30s-september-2025/'

# list of files path
files = [
    base_path + 'september_2025_first30s_chunk_001.csv',
    base_path + 'september_2025_first30s_chunk_002.csv',
    base_path + 'september_2025_first30s_chunk_003.csv',
    base_path + 'september_2025_first30s_chunk_004.csv',
    base_path + 'september_2025_first30s_chunk_005.csv',
    base_path + 'september_2025_first30s_chunk_006.csv',
    base_path + 'september_2025_first30s_chunk_007.csv',
    base_path + 'september_2025_first30s_chunk_008.csv',
    base_path + 'september_2025_first30s_chunk_009.csv',
    base_path + 'september_2025_first30s_chunk_010.csv',
    base_path + 'september_2025_first30s_chunk_011.csv',
    base_path + 'september_2025_first30s_chunk_012.csv',
    base_path + 'september_2025_first30s_chunk_013.csv',
    base_path + 'september_2025_first30s_chunk_014.csv',
    base_path + 'september_2025_first30s_chunk_015.csv'
]

# to conact betwen all files at the same DataFrame
train_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"Shape: {train_df.shape}")


# Read Target Tokens (adjust path if needed)
target_tokens = pd.read_csv('/kaggle/input/alpha-reader/Alpha Radar Target Tokens.csv')

# Rename the column in target_tokens to match train_df
target_tokens.rename(columns={'Target Token Addresses': 'mint_token_id'}, inplace=True)

# Create 'is_target' column in the training data
train_df['is_target'] = train_df['mint_token_id'].isin(target_tokens['mint_token_id']).astype(int)


train_df.shape


train_df.info()


train_df.head(7)


train_df.describe()


train_df.isnull().sum()


def outlier_percentage(train_df):
    numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
    results = []

    for col in numeric_cols:
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # Outliers
        outliers = train_df[(train_df[col] < lower) | (train_df[col] > upper)][col].count()
        total = train_df[col].notnull().sum()
        percent = (outliers / total) * 100
        
        results.append({
            'Column': col,
            'num of Outliers':outliers,
            'Total': total,
            'Percentage of Outliers (%)': round(percent, 2)})
    
    return pd.DataFrame(results)


outlier_df = outlier_percentage(train_df)
print(outlier_df)


# Filter columns with high outlier percentage (e.g., >10%)
high_outlier_df = outlier_df[outlier_df['Percentage of Outliers (%)'] > 10]

# Display them sorted by percentage (highest first)
high_outlier_df = high_outlier_df.sort_values('Percentage of Outliers (%)', ascending=False)

# Print result
print("Columns with a high percentage of outliers (>10%):")
print(high_outlier_df[['Column', 'Percentage of Outliers (%)']])


train_df['rate_of_change'].min(),train_df['rate_of_change'].max()


train_df['volume_oscillator'].min(),train_df['volume_oscillator'].max()


train_df['token_quantity'].min(),train_df['token_quantity'].max()


train_df['token_delta'].min(),train_df['token_delta'].max()


train_df['relative_strength_index'].min(),train_df['relative_strength_index'].max()


train_df['sol_delta'].min(),train_df['sol_delta'].max()


train_df.columns


train_df['timestamp']


# Split the timestamp into minutes, seconds, and sub-seconds
timestamp_split = train_df['timestamp'].str.split(':', expand=True)
train_df['minute'] = timestamp_split[0].astype(int)

sec_split = timestamp_split[1].str.split('.', expand=True)
train_df['second'] = sec_split[0].astype(int)
train_df['sub_second'] = sec_split[1].astype(int)

# Drop the original timestamp column
train_df = train_df.drop(columns=['timestamp'])


train_df['mint_token_id']


# Length of the token ID
train_df['token_id_length'] = train_df['mint_token_id'].str.len()

# Prefix and suffix (last 4 characters)
train_df['token_suffix'] = train_df['mint_token_id'].str[-4:]

# Count of digits, uppercase letters, and lowercase letters
train_df['num_digits'] = train_df['mint_token_id'].str.count(r'\d')
train_df['num_upper'] = train_df['mint_token_id'].str.count(r'[A-Z]')
train_df['num_lower'] = train_df['mint_token_id'].str.count(r'[a-z]')

# Frequency of each token ID in the dataset
token_counts = train_df['mint_token_id'].value_counts()
train_df['token_frequency'] = train_df['mint_token_id'].map(token_counts)


# Token suffix
token_suffix_encoder = ce.TargetEncoder(cols=['token_suffix'])
train_df['token_suffix_encoded'] = token_suffix_encoder.fit_transform(train_df['token_suffix'], train_df['is_target'])

# Drop the original column if desired
train_df = train_df.drop(columns=['mint_token_id','token_suffix'])


# Length of the holder ID
train_df['holder_length'] = train_df['holder'].str.len()

# Last 4 characters (suffix)
train_df['holder_suffix'] = train_df['holder'].str[-4:]

# Count of digits, uppercase letters, and lowercase letters
train_df['num_digits_holder'] = train_df['holder'].str.count(r'\d')
train_df['num_upper_holder'] = train_df['holder'].str.count(r'[A-Z]')
train_df['num_lower_holder'] = train_df['holder'].str.count(r'[a-z]')

# Frequency of each holder in the dataset
holder_counts = train_df['holder'].value_counts()
train_df['holder_frequency'] = train_df['holder'].map(holder_counts)


# Holder suffix
holder_suffix_encoder = ce.TargetEncoder(cols=['holder_suffix'])
train_df['holder_suffix_encoded'] = holder_suffix_encoder.fit_transform(train_df['holder_suffix'], train_df['is_target'])

# Drop the original columns
train_df = train_df.drop(columns=['holder','holder_suffix'])


train_df['trade_mode'].unique()


# Encode 'trade_mode' as numeric: buy = 1, sell = 0
train_df['trade_mode_encoded'] = train_df['trade_mode'].map({'buy': 1, 'sell': 0})

# Then drop the original column
train_df = train_df.drop(columns=['trade_mode'])


train_df['creator']


# Create a Target Encoder
target_encoder = ce.TargetEncoder(cols=['creator'])

# Fit the encoder on the train data
train_df['creator_encoded'] = target_encoder.fit_transform(train_df['creator'], train_df['is_target'])

# Drop the original column
train_data = train_df['creator'].copy()
train_df = train_df.drop(columns=['creator'])


# Max
MAX_VAL = 1e10

# calculate percentage and avoid devided on zero
train_df['delta_ratio'] = train_df['token_delta'] / np.maximum(train_df['sol_delta'], 1e-6)
train_df['volume_ratio'] = train_df['token_volume'] / np.maximum(train_df['sol_volume'], 1e-6)
train_df['cap_to_fee'] = train_df['market_cap_usd'] / np.maximum(train_df['creator_fee'], 1e-6)
train_df['activity_ratio'] = train_df['total_count'] / np.maximum(train_df['token_frequency'], 1)
train_df['buy_rate'] = train_df['buy_count'] / np.maximum(train_df['total_count'], 1)
train_df['sell_rate'] = train_df['sell_count'] / np.maximum(train_df['total_count'], 1)
train_df['holder_to_token_ratio'] = train_df['holder_length'] / np.maximum(train_df['token_id_length'], 1)
train_df['bollinger_rsi_ratio'] = train_df['bollinger_relative_position'] / np.maximum(train_df['relative_strength_index'], 1)
train_df['creator_impact'] = (train_df['creator_balance'] - train_df['creator_fee_pump']) / np.maximum(train_df['market_cap_usd'], 1)
train_df['top10_ratio'] = train_df['top10_percent_total'] / np.maximum(train_df['market_cap_usd'], 1)

# safe processing (add and multiply)
train_df['time_complexity'] = train_df['minute']*60 + train_df['second'] + train_df['sub_second']/1000
train_df['is_fast_tx'] = (train_df['sub_second'] < 500).astype(int)
train_df['id_complexity'] = train_df['num_digits'] + train_df['num_upper'] + train_df['num_lower']
train_df['holder_complexity'] = train_df['num_digits_holder'] + train_df['num_upper_holder'] + train_df['num_lower_holder']
train_df['rsi_mfi_interaction'] = train_df['relative_strength_index'] * train_df['money_flow_index']
train_df['volatility_strength'] = abs(train_df['rate_of_change']) * train_df['volume_oscillator']
train_df['holder_retention'] = train_df['current_holders'] / np.maximum(train_df['total_holders'], 1)

# replace any huage value by maximum value
for col in ['delta_ratio', 'volume_ratio', 'cap_to_fee', 'holder_to_token_ratio', 
            'bollinger_rsi_ratio', 'creator_impact', 'top10_ratio']:
    train_df[col] = np.clip(train_df[col], -MAX_VAL, MAX_VAL)

# Remove any INF or NaN
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df.fillna(0, inplace=True)


train_df.shape


train_df.info()


# float64 -> float32
for col in train_df.select_dtypes(include=['float64']).columns:
    train_df[col] = train_df[col].astype(np.float32)

# int64 -> int32 or uint16 
for col in train_df.select_dtypes(include=['int64']).columns:
    if train_df[col].min() >= 0:
        if train_df[col].max() < 256:
            train_df[col] = train_df[col].astype(np.uint8)
        elif train_df[col].max() < 65536:
            train_df[col] = train_df[col].astype(np.uint16)
        else:
            train_df[col] = train_df[col].astype(np.int32)
    else:
        train_df[col] = train_df[col].astype(np.int32)

# ===== Feature Engineering =====
# Interaction Features
train_df['rsi_mfi_interaction'] = train_df['relative_strength_index'] * train_df['money_flow_index']
train_df['cap_to_fee_ratio'] = train_df['market_cap_usd'] / (train_df['creator_fee'] + 1e-9)
train_df['volume_liquidity_ratio'] = train_df['token_volume'] / (train_df['liquidity_ratio'] + 1e-9)

# Log Transform 
large_cols = ['market_cap_usd', 'token_volume', 'sol_volume', 'creator_balance']
for col in large_cols:
    train_df[col + '_log'] = np.log1p(train_df[col])

# Cyclical Encoding 
train_df['minute_sin'] = np.sin(2 * np.pi * train_df['minute']/60)
train_df['minute_cos'] = np.cos(2 * np.pi * train_df['minute']/60)
train_df['second_sin'] = np.sin(2 * np.pi * train_df['second']/60)
train_df['second_cos'] = np.cos(2 * np.pi * train_df['second']/60)


# Numerical columns
numeric_df = train_df.select_dtypes(include=[np.number])

# check NaN and INF 
print("Number of INF:", np.isinf(numeric_df).sum().sum())
print("Number of NaN:", np.isnan(numeric_df).sum().sum())


train_df.columns


train_df['is_target'].value_counts()


#show pie plot
train_df['is_target'].value_counts().plot.pie(autopct='%.2f')


zeros = train_df[train_df.is_target==0]
ones = train_df[train_df.is_target==1]
print(zeros.shape)
print(ones.shape)


zeros_sample = zeros.sample(980000)
print(zeros_sample.shape)


new_data = pd.concat([zeros_sample,ones] ,axis=0)
new_data.head(7)


target = 'is_target'

report = []

for col in train_df.columns:
    if col == target:
        continue
    
    # correlation with target ( float)
    if np.issubdtype(train_df[col].dtype, np.number):
        corr = train_df[col].corr(train_df[target])
    else:
        corr = np.nan
    
    # uniqueness
    n_unique = train_df[col].nunique()
    n_total = len(train_df)
    uniqueness = n_unique / n_total
    
    # if column is potential leak(corr or uniqueness very low/high)
    potential_leak = False
    if corr is not np.nan and abs(corr) > 0.8:
        potential_leak = True
    if uniqueness < 0.001 or uniqueness > 0.999:
        potential_leak = True

    # if time columns
    time_col = any(x in col.lower() for x in ['time','minute','second','hour'])
    
    report.append({
        'feature': col,
        'dtype': train_df[col].dtype,
        'corr_with_target': corr,
        'abs_corr': abs(corr) if corr is not np.nan else np.nan,
        'uniqueness': uniqueness,
        'is_time_feature': time_col,
        'potential_leak': potential_leak
    })

report_df = pd.DataFrame(report).sort_values(by='abs_corr', ascending=False)
print(report_df.head(20))


# feature which make data leakage
leak_cols = report_df.loc[report_df['potential_leak'] == True, 'feature'].tolist()
time_cols = report_df.loc[report_df['is_time_feature'] == True, 'feature'].tolist()

extra_drop = ['index', 'is_target']

# safe X
X_safe = new_data.drop(columns=leak_cols + time_cols + extra_drop)
feature_cols_safe = X_safe.columns.tolist()

# target
y = new_data['is_target']

print("Number of safe features:", len(feature_cols_safe))
print("Dropped leak/time features + extra:", leak_cols + time_cols + extra_drop)


X_train, X_val, y_train, y_val = train_test_split(X_safe, y, test_size=0.2, random_state=42, stratify=y)


scale_weight = 980000/397893
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    eval_metric='AUC',
    scale_pos_weight = scale_weight,
    verbose=100
)
eval_set = (X_val,y_val)
model.fit(X_train,y_train,
          eval_set = eval_set,
          use_best_model=True)

y_pred_proba = model.predict_proba(X_val)[: ,1]
auc = roc_auc_score(y_val,y_pred_proba)
print("AUC :" ,auc)


y_pred = model.predict(X_val)
precision = precision_score(y_val,y_pred)
recall = recall_score(y_val,y_pred)
print("precision :" ,precision)
print("Recall :" ,recall)

f1 =f1_score(y_val,y_pred)
print("F1-score :" ,f1)


model.save_model("catboost_model.cbm")

print("Model is saving'catboost_model.cbm'")

