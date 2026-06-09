import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway

train = pd.read_parquet('/kaggle/input/alpha-summer-challenge/train.pa')
transactions = pd.read_parquet('/kaggle/input/alpha-summer-challenge/df_transaction.pa')


# 1. Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ· Ñ‡Ğ°Ñ�Ñ‚Ğ¾Ñ‚
mcc_freq = transactions['mcc_code'].value_counts()
print("Ğ¢Ğ¾Ğ¿-20 MCC Ğ¿Ğ¾ Ñ‡Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ğµ:")
print(mcc_freq.head(20))

# 2. Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ· Ñ�Ñ€ĞµĞ´Ğ½Ğ¸Ñ… Ñ�ÑƒĞ¼Ğ¼  
mcc_amounts = transactions.groupby('mcc_code')['amount'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
print("\nĞ¢Ğ¾Ğ¿-20 MCC Ğ¿Ğ¾ Ñ�Ñ€ĞµĞ´Ğ½ĞµĞ¹ Ñ�ÑƒĞ¼Ğ¼Ğµ:")
print(mcc_amounts.head(20))

# 3. Ğ�Ğ°Ğ¹Ñ‚Ğ¸ "Ğ´Ğ¾Ñ€Ğ¾Ğ³Ğ¸Ğµ" vs "Ğ´ĞµÑˆĞµĞ²Ñ‹Ğµ" ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¸
expensive_threshold = transactions['amount'].quantile(0.8)
cheap_threshold = transactions['amount'].quantile(0.2)

expensive_mccs = transactions[transactions['amount'] > expensive_threshold]['mcc_code'].value_counts().head(10).index.tolist()
cheap_mccs = transactions[transactions['amount'] < cheap_threshold]['mcc_code'].value_counts().head(10).index.tolist()

print(f"\nĞ’ĞµÑ€Ğ¾Ñ�Ñ‚Ğ½Ğ¾ 'luxury' MCC (Ğ´Ğ¾Ñ€Ğ¾Ğ³Ğ¸Ğµ Ğ¿Ğ¾ĞºÑƒĞ¿ĞºĞ¸): {expensive_mccs}")
print(f"Ğ’ĞµÑ€Ğ¾Ñ�Ñ‚Ğ½Ğ¾ 'essential' MCC (Ğ´ĞµÑˆĞµĞ²Ñ‹Ğµ/Ñ‡Ğ°Ñ�Ñ‚Ñ‹Ğµ): {cheap_mccs}")

# 4. Data-driven ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¸
essential_categories = mcc_freq.head(10).index.tolist()  # Ğ¢Ğ¾Ğ¿ Ğ¿Ğ¾ Ñ‡Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ğµ
luxury_categories = expensive_mccs[:5]  # Ğ¢Ğ¾Ğ¿ Ğ¿Ğ¾ Ñ�ÑƒĞ¼Ğ¼Ğµ

print(f"\nData-driven essential: {essential_categories}")
print(f"Data-driven luxury: {luxury_categories}")


import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_advanced_features(transactions, train_clients):
    """
    Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¸Ğµ Ğ¿Ñ€Ğ¾Ğ´Ğ²Ğ¸Ğ½ÑƒÑ‚Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² Ğ´Ğ»Ñ� Ğ±Ğ°Ğ½ĞºĞ¾Ğ²Ñ�ĞºĞ¸Ñ… Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹
    """
    features = []
    
    # ĞŸĞ¾Ğ´Ğ³Ğ¾Ñ‚Ğ°Ğ²Ğ»Ğ¸Ğ²Ğ°ĞµĞ¼ Ğ±Ğ°Ğ·Ğ¾Ğ²Ñ‹Ğµ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸, ĞµÑ�Ğ»Ğ¸ Ğ¸Ñ… Ğ½ĞµÑ‚
    if 'hour' not in transactions.columns:
        transactions['hour'] = pd.to_datetime(transactions['date_time']).dt.hour
    if 'weekday' not in transactions.columns:
        transactions['weekday'] = pd.to_datetime(transactions['date_time']).dt.weekday
    if 'date' not in transactions.columns:
        transactions['date'] = pd.to_datetime(transactions['date_time']).dt.date
    
    # === 1. VELOCITY & MOMENTUM FEATURES ===
    # Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ· Ñ�ĞºĞ¾Ñ€Ğ¾Ñ�Ñ‚Ğ¸ Ğ¸Ğ·Ğ¼ĞµĞ½ĞµĞ½Ğ¸Ñ� Ğ¿Ğ¾Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ� ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ°
    print("ğŸš€ Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ velocity/momentum Ñ„Ğ¸Ñ‡Ğ¸...")
    
    # Ğ Ğ°Ğ·Ğ´ĞµĞ»Ğ¸Ğ¼ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ Ğ½Ğ° Ğ¿ĞµÑ€Ğ¸Ğ¾Ğ´Ñ‹ (Ğ¿ĞµÑ€Ğ²Ñ‹Ğµ 2 Ğ¼ĞµÑ�Ñ�Ñ†Ğ° vs Ğ¿Ğ¾Ñ�Ğ»ĞµĞ´Ğ½Ğ¸Ğ¹ Ğ¼ĞµÑ�Ñ�Ñ†)
    transactions['date'] = pd.to_datetime(transactions['date_time']).dt.date
    max_date = transactions['date'].max()
    split_date = max_date - timedelta(days=30)
    
    # Ğ¡Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞ¸ Ğ¿Ğ¾ Ğ¿ĞµÑ€Ğ¸Ğ¾Ğ´Ğ°Ğ¼
    early_period = transactions[transactions['date'] <= split_date]
    late_period = transactions[transactions['date'] > split_date]
    
    # Ğ¡Ñ€Ğ°Ğ²Ğ½ĞµĞ½Ğ¸Ğµ Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ¼ĞµĞ¶Ğ´Ñƒ Ğ¿ĞµÑ€Ğ¸Ğ¾Ğ´Ğ°Ğ¼Ğ¸
    early_stats = early_period.groupby('client_num').agg({
        'amount': ['sum', 'mean', 'count'],
        'mcc_code': 'nunique'
    }).reset_index()
    early_stats.columns = ['client_num', 'early_sum', 'early_mean', 'early_count', 'early_mcc']
    
    late_stats = late_period.groupby('client_num').agg({
        'amount': ['sum', 'mean', 'count'],
        'mcc_code': 'nunique'
    }).reset_index()
    late_stats.columns = ['client_num', 'late_sum', 'late_mean', 'late_count', 'late_mcc']
    
    # Momentum Ñ„Ğ¸Ñ‡Ğ¸
    momentum_df = pd.merge(early_stats, late_stats, on='client_num', how='outer').fillna(0)
    momentum_df['momentum_sum'] = (momentum_df['late_sum'] - momentum_df['early_sum']) / (momentum_df['early_sum'] + 1)
    momentum_df['momentum_mean'] = (momentum_df['late_mean'] - momentum_df['early_mean']) / (momentum_df['early_mean'] + 1)
    momentum_df['momentum_count'] = (momentum_df['late_count'] - momentum_df['early_count']) / (momentum_df['early_count'] + 1)
    momentum_df['momentum_mcc'] = (momentum_df['late_mcc'] - momentum_df['early_mcc']) / (momentum_df['early_mcc'] + 1)
    
    # Ğ¡Ñ‚Ğ°Ğ±Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¿Ğ¾Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ�
    momentum_df['behavior_stability'] = 1 / (1 + np.abs(momentum_df['momentum_sum']) + 
                                           np.abs(momentum_df['momentum_mean']) + 
                                           np.abs(momentum_df['momentum_count']))
    
    features.append(momentum_df[['client_num', 'momentum_sum', 'momentum_mean', 'momentum_count', 
                                'momentum_mcc', 'behavior_stability']])
    
    # === 2. CYCLICAL & SEASONAL FEATURES ===
    print("ğŸ”„ Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ†Ğ¸ĞºĞ»Ğ¸Ñ‡ĞµÑ�ĞºĞ¸Ğµ/Ñ�ĞµĞ·Ğ¾Ğ½Ğ½Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸...")
    
    transactions['month'] = pd.to_datetime(transactions['date_time']).dt.month
    transactions['day_of_month'] = pd.to_datetime(transactions['date_time']).dt.day
    transactions['is_weekend'] = pd.to_datetime(transactions['date_time']).dt.weekday >= 5
    transactions['is_month_start'] = transactions['day_of_month'] <= 7
    transactions['is_month_end'] = transactions['day_of_month'] >= 25
    
    # Ğ¦Ğ¸ĞºĞ»Ğ¸Ñ‡ĞµÑ�ĞºĞ¸Ğµ Ñ�Ğ½ĞºĞ¾Ğ´Ğ¸Ğ½Ğ³Ğ¸
    transactions['hour_sin'] = np.sin(2 * np.pi * transactions['hour'] / 24)
    transactions['hour_cos'] = np.cos(2 * np.pi * transactions['hour'] / 24)
    transactions['weekday_sin'] = np.sin(2 * np.pi * transactions['weekday'] / 7)
    transactions['weekday_cos'] = np.cos(2 * np.pi * transactions['weekday'] / 7)
    
    # Ğ¡ĞµĞ·Ğ¾Ğ½Ğ½Ñ‹Ğµ Ğ¿Ğ°Ñ‚Ñ‚ĞµÑ€Ğ½Ñ‹
    seasonal_features = transactions.groupby('client_num').agg({
        'is_weekend': ['sum', 'mean'],
        'is_month_start': ['sum', 'mean'],
        'is_month_end': ['sum', 'mean'],
        'hour_sin': 'mean',
        'hour_cos': 'mean',
        'weekday_sin': 'mean',
        'weekday_cos': 'mean'
    }).reset_index()
    seasonal_features.columns = ['client_num', 'weekend_txns', 'weekend_ratio', 
                               'month_start_txns', 'month_start_ratio',
                               'month_end_txns', 'month_end_ratio',
                               'avg_hour_sin', 'avg_hour_cos',
                               'avg_weekday_sin', 'avg_weekday_cos']
    
    features.append(seasonal_features)
    
    # === 3. TRANSACTION GRAPH/NETWORK FEATURES ===
    print("ğŸ•¸ï¸� Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ�ĞµÑ‚ĞµĞ²Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸...")
    
    # Ğ¡Ğ²Ñ�Ğ·Ğ¸ Ñ‡ĞµÑ€ĞµĞ· Ğ¼ĞµÑ€Ñ‡Ğ°Ğ½Ñ‚Ğ¾Ğ²
    merchant_clients = transactions.groupby('merchant_name')['client_num'].nunique().reset_index()
    merchant_clients.columns = ['merchant_name', 'merchant_popularity']
    
    # Ğ”Ğ¶Ğ¾Ğ¹Ğ½Ğ¸Ğ¼ Ğ¾Ğ±Ñ€Ğ°Ñ‚Ğ½Ğ¾ Ğº Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ñ�Ğ¼
    transactions_with_merchant = transactions.merge(merchant_clients, on='merchant_name', how='left')
    
    # Ğ¡ĞµÑ‚ĞµĞ²Ñ‹Ğµ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸ ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ¾Ğ²
    network_features = transactions_with_merchant.groupby('client_num').agg({
        'merchant_popularity': ['mean', 'max', 'min', 'std'],
        'merchant_name': 'nunique'
    }).reset_index()
    network_features.columns = ['client_num', 'avg_merchant_popularity', 'max_merchant_popularity',
                              'min_merchant_popularity', 'std_merchant_popularity', 'unique_merchants']
    
    # Ğ­ĞºÑ�ĞºĞ»Ñ�Ğ·Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚ÑŒ ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ° (Ğ¸Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·ÑƒĞµÑ‚ Ğ»Ğ¸ Ñ€ĞµĞ´ĞºĞ¸Ğµ Ğ¼ĞµÑ€Ñ‡Ğ°Ğ½Ñ‚Ñ‹)
    network_features['exclusivity_score'] = 1 / (network_features['avg_merchant_popularity'] + 1)
    
    features.append(network_features)
    
    # === 4. SPENDING PATTERN COMPLEXITY ===
    print("ğŸ“Š Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ„Ğ¸Ñ‡Ğ¸ Ñ�Ğ»Ğ¾Ğ¶Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ¿Ğ°Ñ‚Ñ‚ĞµÑ€Ğ½Ğ¾Ğ²...")
    
    # Ğ­Ğ½Ñ‚Ñ€Ğ¾Ğ¿Ğ¸Ñ� Ñ‚Ñ€Ğ°Ñ‚ Ğ¿Ğ¾ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ñ�Ğ¼
    def calculate_entropy(series):
        probs = series.value_counts(normalize=True)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    # Ğ­Ğ½Ñ‚Ñ€Ğ¾Ğ¿Ğ¸Ñ� Ğ¿Ğ¾ MCC ĞºĞ¾Ğ´Ğ°Ğ¼
    mcc_entropy = transactions.groupby('client_num')['mcc_code'].apply(calculate_entropy).reset_index()
    mcc_entropy.columns = ['client_num', 'mcc_entropy']
    
    # Ğ­Ğ½Ñ‚Ñ€Ğ¾Ğ¿Ğ¸Ñ� Ğ¿Ğ¾ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ¸ (Ñ‡Ğ°Ñ�Ñ‹)
    time_entropy = transactions.groupby('client_num')['hour'].apply(calculate_entropy).reset_index()
    time_entropy.columns = ['client_num', 'time_entropy']
    
    # Ğ­Ğ½Ñ‚Ñ€Ğ¾Ğ¿Ğ¸Ñ� Ğ¿Ğ¾ Ñ�ÑƒĞ¼Ğ¼Ğ°Ğ¼ (Ğ´Ğ¸Ñ�ĞºÑ€ĞµÑ‚Ğ¸Ğ·Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½Ñ‹Ğ¼)
    transactions['amount_bucket'] = pd.cut(transactions['amount'], bins=20, labels=False)
    amount_entropy = transactions.groupby('client_num')['amount_bucket'].apply(calculate_entropy).reset_index()
    amount_entropy.columns = ['client_num', 'amount_entropy']
    
    # Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼ Ñ�Ğ½Ñ‚Ñ€Ğ¾Ğ¿Ğ¸Ğ¸
    entropy_features = mcc_entropy.merge(time_entropy, on='client_num', how='outer')
    entropy_features = entropy_features.merge(amount_entropy, on='client_num', how='outer')
    
    # Ğ�Ğ±Ñ‰Ğ°Ñ� Ñ�Ğ»Ğ¾Ğ¶Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¿Ğ¾Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ�
    entropy_features['total_complexity'] = (entropy_features['mcc_entropy'] + 
                                          entropy_features['time_entropy'] + 
                                          entropy_features['amount_entropy'])
    
    features.append(entropy_features)
    
    # === 5. FINANCIAL HEALTH INDICATORS ===
    print("ğŸ’° Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ¸Ğ½Ğ´Ğ¸ĞºĞ°Ñ‚Ğ¾Ñ€Ñ‹ Ñ„Ğ¸Ğ½Ğ°Ğ½Ñ�Ğ¾Ğ²Ğ¾Ğ³Ğ¾ Ğ·Ğ´Ğ¾Ñ€Ğ¾Ğ²ÑŒÑ�...")
    
    # Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ· ĞºÑ€ÑƒĞ¿Ğ½Ñ‹Ñ… Ñ‚Ñ€Ğ°Ñ‚
    transactions['is_large_transaction'] = transactions['amount'] > transactions.groupby('client_num')['amount'].transform('quantile', 0.9)
    
    # ĞŸĞ°Ñ‚Ñ‚ĞµÑ€Ğ½Ñ‹ ĞºÑ€ÑƒĞ¿Ğ½Ñ‹Ñ… Ñ‚Ñ€Ğ°Ñ‚
    large_txn_features = transactions.groupby('client_num').agg({
        'is_large_transaction': ['sum', 'mean']
    }).reset_index()
    large_txn_features.columns = ['client_num', 'large_txn_count', 'large_txn_ratio']
    
    # Ğ ĞµĞ³ÑƒĞ»Ñ�Ñ€Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ñ‚Ñ€Ğ°Ñ‚
    daily_spending = transactions.groupby(['client_num', 'date'])['amount'].sum().reset_index()
    spending_regularity = daily_spending.groupby('client_num')['amount'].agg(['std', 'mean']).reset_index()
    spending_regularity['spending_cv'] = spending_regularity['std'] / (spending_regularity['mean'] + 1)
    spending_regularity = spending_regularity[['client_num', 'spending_cv']]
    
    # Ğ¢Ñ€ĞµĞ½Ğ´ Ñ‚Ñ€Ğ°Ñ‚ (Ğ»Ğ¸Ğ½ĞµĞ¹Ğ½Ğ°Ñ� Ñ€ĞµĞ³Ñ€ĞµÑ�Ñ�Ğ¸Ñ� Ğ¿Ğ¾ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ¸)
    def calculate_spending_trend(group):
        if len(group) < 3:
            return 0
        x = np.arange(len(group))
        try:
            slope = np.polyfit(x, group['amount'], 1)[0]
            return slope
        except:
            return 0
    
    spending_trend = daily_spending.groupby('client_num').apply(calculate_spending_trend).reset_index()
    spending_trend.columns = ['client_num', 'spending_trend']
    
    # Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼ Ñ„Ğ¸Ğ½Ğ°Ğ½Ñ�Ğ¾Ğ²Ñ‹Ğµ Ğ¸Ğ½Ğ´Ğ¸ĞºĞ°Ñ‚Ğ¾Ñ€Ñ‹
    financial_features = large_txn_features.merge(spending_regularity, on='client_num', how='outer')
    financial_features = financial_features.merge(spending_trend, on='client_num', how='outer')
    
    features.append(financial_features)
    
    # === 6. BEHAVIORAL ANOMALIES ===
    print("ğŸš¨ Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ„Ğ¸Ñ‡Ğ¸ Ğ°Ğ½Ğ¾Ğ¼Ğ°Ğ»Ğ¸Ğ¹ Ğ¿Ğ¾Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ�...")
    
    # Z-scores Ğ´Ğ»Ñ� Ğ¾Ñ�Ğ½Ğ¾Ğ²Ğ½Ñ‹Ñ… Ğ¼ĞµÑ‚Ñ€Ğ¸Ğº
    client_stats = transactions.groupby('client_num').agg({
        'amount': ['mean', 'std', 'count'],
        'date': 'nunique'
    }).reset_index()
    client_stats.columns = ['client_num', 'avg_amount', 'std_amount', 'txn_count', 'active_days']
    
    # Ğ“Ğ»Ğ¾Ğ±Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ z-scores
    client_stats['z_avg_amount'] = (client_stats['avg_amount'] - client_stats['avg_amount'].mean()) / client_stats['avg_amount'].std()
    client_stats['z_txn_count'] = (client_stats['txn_count'] - client_stats['txn_count'].mean()) / client_stats['txn_count'].std()
    client_stats['z_active_days'] = (client_stats['active_days'] - client_stats['active_days'].mean()) / client_stats['active_days'].std()
    
    # Ğ�Ğ½Ğ¾Ğ¼Ğ°Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ°
    client_stats['anomaly_score'] = np.sqrt(client_stats['z_avg_amount']**2 + 
                                          client_stats['z_txn_count']**2 + 
                                          client_stats['z_active_days']**2)
    
    features.append(client_stats[['client_num', 'z_avg_amount', 'z_txn_count', 
                                'z_active_days', 'anomaly_score']])
    
    # === 7. ADVANCED ROLLING FEATURES ===
    print("ğŸ“ˆ Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ¿Ñ€Ğ¾Ğ´Ğ²Ğ¸Ğ½ÑƒÑ‚Ñ‹Ğµ rolling Ñ„Ğ¸Ñ‡Ğ¸...")
    
    # Ğ­ĞºÑ�Ğ¿Ğ¾Ğ½ĞµĞ½Ñ†Ğ¸Ğ°Ğ»ÑŒĞ½Ğ¾Ğµ Ñ�ĞºĞ¾Ğ»ÑŒĞ·Ñ�Ñ‰ĞµĞµ Ñ�Ñ€ĞµĞ´Ğ½ĞµĞµ
    def calculate_ema_features(group, alpha=0.3):
        amounts = group.sort_values('date_time')['amount']
        ema = amounts.ewm(alpha=alpha).mean()
        return pd.Series({
            'ema_amount': ema.iloc[-1] if len(ema) > 0 else 0,
            'ema_volatility': amounts.ewm(alpha=alpha).std().iloc[-1] if len(amounts) > 1 else 0
        })
    
    ema_features = transactions.groupby('client_num').apply(calculate_ema_features).reset_index()
    
    # Ğ�Ğ²Ñ‚Ğ¾ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ñ� Ğ² Ñ‚Ñ€Ğ°Ñ‚Ğ°Ñ…
    def calculate_autocorr(group):
        if len(group) < 10:
            return 0
        daily_amounts = group.groupby('date')['amount'].sum()
        try:
            return daily_amounts.autocorr(lag=1)
        except:
            return 0
    
    autocorr_features = transactions.groupby('client_num').apply(calculate_autocorr).reset_index()
    autocorr_features.columns = ['client_num', 'spending_autocorr']
    
    # Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼ advanced rolling Ñ„Ğ¸Ñ‡Ğ¸
    advanced_rolling = ema_features.merge(autocorr_features, on='client_num', how='outer')
    features.append(advanced_rolling)
    
    # === 8. CATEGORY-SPECIFIC INSIGHTS ===
    print("ğŸ�ª Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ„Ğ¸Ñ‡Ğ¸ Ğ¿Ğ¾ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ñ�Ğ¼...")
    
    # DATA-DRIVEN ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¸ Ğ½Ğ° Ğ¾Ñ�Ğ½Ğ¾Ğ²Ğµ Ğ²Ğ°ÑˆĞ¸Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
    # Ğ’Ñ‹Ñ�Ğ¾ĞºĞ¾Ñ‡Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ğ½Ñ‹Ğµ (ĞµĞ¶ĞµĞ´Ğ½ĞµĞ²Ğ½Ñ‹Ğµ Ğ½ÑƒĞ¶Ğ´Ñ‹)
    frequent_categories = [5411, 5499, 5814, 4131, 3990]  # Ğ¢Ğ¾Ğ¿-5 Ğ¿Ğ¾ Ñ‡Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ğµ
    
    # Ğ”Ğ¾Ñ€Ğ¾Ğ³Ğ¸Ğµ Ğ¾Ğ¿ĞµÑ€Ğ°Ñ†Ğ¸Ğ¸ (Ğ¿Ğ¾ Ñ�Ñ€ĞµĞ´Ğ½ĞµĞ¹ Ñ�ÑƒĞ¼Ğ¼Ğµ, Ğ¸Ñ�ĞºĞ»Ñ�Ñ‡Ğ°Ñ� outliers)
    expensive_categories = [6011, 5511, 3011, 4722, 5712]  # Ğ’Ñ‹Ñ�Ğ¾ĞºĞ¸Ğµ Ñ�Ñ€ĞµĞ´Ğ½Ğ¸Ğµ Ñ�ÑƒĞ¼Ğ¼Ñ‹
    
    # Ğ¡Ñ€ĞµĞ´Ğ½Ğ¸Ğ¹ Ñ†ĞµĞ½Ğ¾Ğ²Ğ¾Ğ¹ Ñ�ĞµĞ³Ğ¼ĞµĞ½Ñ‚ (Ñ‡Ğ°Ñ�Ñ‚Ñ‹Ğµ + ÑƒĞ¼ĞµÑ€ĞµĞ½Ğ½Ñ‹Ğµ Ñ�ÑƒĞ¼Ğ¼Ñ‹)
    moderate_categories = [5541, 5912, 5921, 5812, 6536]
    
    # Ğ ĞµĞ´ĞºĞ¸Ğµ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¸ (Ğ½Ğ¸Ğ·ĞºĞ°Ñ� Ñ‡Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ğ°)
    rare_categories = [7999, 5993, 5462]  # Ğ˜Ğ· Ñ‚Ğ¾Ğ¿-20 Ğ½Ğ¾ Ñ� Ğ½Ğ¸Ğ·ĞºĞ¾Ğ¹ Ñ‡Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ğ¾Ğ¹
    
    transactions['is_frequent'] = transactions['mcc_code'].isin(frequent_categories)
    transactions['is_expensive'] = transactions['mcc_code'].isin(expensive_categories)
    transactions['is_moderate'] = transactions['mcc_code'].isin(moderate_categories)
    transactions['is_rare'] = transactions['mcc_code'].isin(rare_categories)
    
    category_features = transactions.groupby('client_num').agg({
        'is_frequent': ['sum', 'mean'],
        'is_expensive': ['sum', 'mean'],
        'is_moderate': ['sum', 'mean'],
        'is_rare': ['sum', 'mean']
    }).reset_index()
    category_features.columns = ['client_num', 'frequent_txns', 'frequent_ratio', 
                               'expensive_txns', 'expensive_ratio',
                               'moderate_txns', 'moderate_ratio',
                               'rare_txns', 'rare_ratio']
    
    # ĞŸĞ°Ñ‚Ñ‚ĞµÑ€Ğ½Ñ‹ Ğ¿Ğ¾Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ� Ğ¿Ğ¾ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ñ�Ğ¼
    category_features['frequent_expensive_ratio'] = category_features['frequent_txns'] / (category_features['expensive_txns'] + 1)
    category_features['category_diversity'] = (category_features['frequent_ratio'] + 
                                             category_features['expensive_ratio'] + 
                                             category_features['moderate_ratio'] + 
                                             category_features['rare_ratio'])
    
    # Ğ˜Ğ½Ğ´ĞµĞºÑ� "practical spending" - Ğ¼Ğ½Ğ¾Ğ³Ğ¾ Ñ‡Ğ°Ñ�Ñ‚Ñ‹Ñ…, Ğ¼Ğ°Ğ»Ğ¾ Ñ€ĞµĞ´ĞºĞ¸Ñ…
    category_features['practical_spending_index'] = (category_features['frequent_ratio'] - 
                                                   category_features['rare_ratio'] + 1) / 2
    
    features.append(category_features)
    
    # === Ğ�Ğ‘ĞªĞ•Ğ”Ğ˜Ğ�Ğ•Ğ�Ğ˜Ğ• Ğ’Ğ¡Ğ•Ğ¥ Ğ¤Ğ˜Ğ§Ğ•Ğ™ ===
    print("ğŸ”— Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼ Ğ²Ñ�Ğµ Ñ„Ğ¸Ñ‡Ğ¸...")
    
    # Ğ�Ğ°Ñ‡Ğ¸Ğ½Ğ°ĞµĞ¼ Ñ� Ğ¿ĞµÑ€Ğ²Ğ¾Ğ³Ğ¾ Ğ½Ğ°Ğ±Ğ¾Ñ€Ğ° Ñ„Ğ¸Ñ‡ĞµĞ¹
    final_features = features[0]
    
    # ĞŸĞ¾Ñ�Ğ»ĞµĞ´Ğ¾Ğ²Ğ°Ñ‚ĞµĞ»ÑŒĞ½Ğ¾ Ğ´Ğ¶Ğ¾Ğ¹Ğ½Ğ¸Ğ¼ Ğ²Ñ�Ğµ Ğ¾Ñ�Ñ‚Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ
    for i, feature_set in enumerate(features[1:], 1):
        print(f"   Ğ”Ğ¶Ğ¾Ğ¹Ğ½Ğ¸Ğ¼ Ğ½Ğ°Ğ±Ğ¾Ñ€ {i+1}/{len(features)}: {feature_set.shape}")
        final_features = final_features.merge(feature_set, on='client_num', how='outer')
    
    # Ğ—Ğ°Ğ¿Ğ¾Ğ»Ğ½Ñ�ĞµĞ¼ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¸
    final_features = final_features.fillna(0)
    
    print(f"âœ… Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¾ {final_features.shape[1]-1} Ğ½Ğ¾Ğ²Ñ‹Ñ… Ñ„Ğ¸Ñ‡ĞµĞ¹ Ğ´Ğ»Ñ� {final_features.shape[0]} ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ¾Ğ²")
    
    return final_features


def create_balance_specific_features(transactions, train_clients):
    """
    Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¸Ğµ Ñ„Ğ¸Ñ‡ĞµĞ¹ Ñ�Ğ¿ĞµÑ†Ğ¸Ğ°Ğ»ÑŒĞ½Ğ¾ Ğ´Ğ»Ñ� Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�Ğ° Ñ�Ñ‡ĞµÑ‚Ğ°
    """
    features = []
    
    # ĞŸĞ¾Ğ´Ğ³Ğ¾Ñ‚Ğ°Ğ²Ğ»Ğ¸Ğ²Ğ°ĞµĞ¼ Ğ±Ğ°Ğ·Ğ¾Ğ²Ñ‹Ğµ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸, ĞµÑ�Ğ»Ğ¸ Ğ¸Ñ… Ğ½ĞµÑ‚
    if 'hour' not in transactions.columns:
        transactions['hour'] = pd.to_datetime(transactions['date_time']).dt.hour
    if 'weekday' not in transactions.columns:
        transactions['weekday'] = pd.to_datetime(transactions['date_time']).dt.weekday
    if 'date' not in transactions.columns:
        transactions['date'] = pd.to_datetime(transactions['date_time']).dt.date
    
    # === 1. CASH FLOW PATTERNS ===
    print("ğŸ’¸ Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ„Ğ¸Ñ‡Ğ¸ Ğ´ĞµĞ½ĞµĞ¶Ğ½Ñ‹Ñ… Ğ¿Ğ¾Ñ‚Ğ¾ĞºĞ¾Ğ²...")
    
    # Ğ˜Ğ¼Ğ¸Ñ‚Ğ°Ñ†Ğ¸Ñ� Ğ²Ñ…Ğ¾Ğ´Ñ�Ñ‰Ğ¸Ñ…/Ğ¸Ñ�Ñ…Ğ¾Ğ´Ñ�Ñ‰Ğ¸Ñ… Ğ¿Ğ¾Ñ‚Ğ¾ĞºĞ¾Ğ²
    # ĞŸÑ€ĞµĞ´Ğ¿Ğ¾Ğ»Ğ¾Ğ¶Ğ¸Ğ¼, Ñ‡Ñ‚Ğ¾ ĞºÑ€ÑƒĞ¿Ğ½Ñ‹Ğµ Ğ¿Ğ¾Ğ»Ğ¾Ğ¶Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ñ‹Ğµ Ñ�ÑƒĞ¼Ğ¼Ñ‹ - Ñ�Ñ‚Ğ¾ Ğ¿Ğ¾Ñ�Ñ‚ÑƒĞ¿Ğ»ĞµĞ½Ğ¸Ñ�
    # Ğ° Ğ¾Ğ±Ñ‹Ñ‡Ğ½Ñ‹Ğµ Ñ�ÑƒĞ¼Ğ¼Ñ‹ - Ñ�Ñ‚Ğ¾ Ñ‚Ñ€Ğ°Ñ‚Ñ‹
    
    # Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ· Ñ€Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ñ� Ñ�ÑƒĞ¼Ğ¼ Ğ´Ğ»Ñ� Ğ¾Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ñ� Ğ¿Ğ¾Ñ€Ğ¾Ğ³Ğ¾Ğ²Ñ‹Ñ… Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğ¹
    amount_stats = transactions.groupby('client_num')['amount'].agg(['mean', 'std', 'quantile']).reset_index()
    
    # Ğ�Ğ¿Ñ€ĞµĞ´ĞµĞ»Ñ�ĞµĞ¼ "ĞºÑ€ÑƒĞ¿Ğ½Ñ‹Ğµ" Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸ Ğ´Ğ»Ñ� ĞºĞ°Ğ¶Ğ´Ğ¾Ğ³Ğ¾ ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ°
    def classify_transactions(group):
        q75 = group['amount'].quantile(0.75)
        q25 = group['amount'].quantile(0.25)
        
        group['is_large_inflow'] = (group['amount'] > q75 * 2)  # Ğ’Ğ¾Ğ·Ğ¼Ğ¾Ğ¶Ğ½Ñ‹Ğµ Ğ¿Ğ¾Ñ�Ñ‚ÑƒĞ¿Ğ»ĞµĞ½Ğ¸Ñ�
        group['is_regular_spend'] = (group['amount'] >= q25) & (group['amount'] <= q75)
        group['is_small_spend'] = group['amount'] < q25
        
        return group[['client_num', 'amount', 'is_large_inflow', 'is_regular_spend', 'is_small_spend']]
    
    classified_txns = transactions.groupby('client_num').apply(classify_transactions).reset_index(drop=True)
    
    # ĞŸĞ°Ñ‚Ñ‚ĞµÑ€Ğ½Ñ‹ Ğ´ĞµĞ½ĞµĞ¶Ğ½Ñ‹Ñ… Ğ¿Ğ¾Ñ‚Ğ¾ĞºĞ¾Ğ²
    cashflow_features = classified_txns.groupby('client_num').agg({
        'is_large_inflow': ['sum', 'mean'],
        'is_regular_spend': ['sum', 'mean'],
        'is_small_spend': ['sum', 'mean']
    }).reset_index()
    cashflow_features.columns = ['client_num', 'large_inflow_count', 'large_inflow_ratio',
                               'regular_spend_count', 'regular_spend_ratio',
                               'small_spend_count', 'small_spend_ratio']
    
    # Ğ‘Ğ°Ğ»Ğ°Ğ½Ñ� Ğ¼ĞµĞ¶Ğ´Ñƒ Ğ¿Ğ¾Ñ�Ñ‚ÑƒĞ¿Ğ»ĞµĞ½Ğ¸Ñ�Ğ¼Ğ¸ Ğ¸ Ñ‚Ñ€Ğ°Ñ‚Ğ°Ğ¼Ğ¸
    cashflow_features['inflow_spend_balance'] = cashflow_features['large_inflow_ratio'] - cashflow_features['regular_spend_ratio']
    
    features.append(cashflow_features)
    
    # === 2. SPENDING VELOCITY & BURN RATE ===
    print("ğŸ”¥ Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ„Ğ¸Ñ‡Ğ¸ Ñ�ĞºĞ¾Ñ€Ğ¾Ñ�Ñ‚Ğ¸ Ñ‚Ñ€Ğ°Ñ‚...")
    
    # Ğ¡Ğ¾Ñ€Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸ Ğ¿Ğ¾ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ¸
    transactions_sorted = transactions.sort_values(['client_num', 'date_time'])
    
    # ĞšÑƒĞ¼ÑƒĞ»Ñ�Ñ‚Ğ¸Ğ²Ğ½Ñ‹Ğµ Ñ�ÑƒĞ¼Ğ¼Ñ‹ Ğ¸ Ñ�ĞºĞ¾Ñ€Ğ¾Ñ�Ñ‚ÑŒ Ñ‚Ñ€Ğ°Ñ‚
    def calculate_burn_rate(group):
        if len(group) < 2:
            return pd.Series({'burn_rate': 0, 'acceleration': 0, 'days_to_spend_median': 0})
        
        # ĞšÑƒĞ¼ÑƒĞ»Ñ�Ñ‚Ğ¸Ğ²Ğ½Ğ°Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ°
        group['cumsum'] = group['amount'].cumsum()
        
        # Ğ’Ñ€ĞµĞ¼Ñ� Ğ¼ĞµĞ¶Ğ´Ñƒ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ñ�Ğ¼Ğ¸ Ğ² Ğ´Ğ½Ñ�Ñ…
        group['time_diff_days'] = group['date_time'].diff().dt.total_seconds() / (24 * 3600)
        
        # Ğ¡ĞºĞ¾Ñ€Ğ¾Ñ�Ñ‚ÑŒ Ñ‚Ñ€Ğ°Ñ‚ (Ñ�ÑƒĞ¼Ğ¼Ğ°/Ğ´ĞµĞ½ÑŒ)
        daily_spend = group['amount'].sum() / ((group['date_time'].max() - group['date_time'].min()).days + 1)
        
        # Ğ£Ñ�ĞºĞ¾Ñ€ĞµĞ½Ğ¸Ğµ Ñ‚Ñ€Ğ°Ñ‚ (Ğ¸Ğ·Ğ¼ĞµĞ½ĞµĞ½Ğ¸Ğµ Ñ�ĞºĞ¾Ñ€Ğ¾Ñ�Ñ‚Ğ¸)
        if len(group) >= 4:
            half_point = len(group) // 2
            early_rate = group.iloc[:half_point]['amount'].sum() / (half_point + 1)
            late_rate = group.iloc[half_point:]['amount'].sum() / (len(group) - half_point + 1)
            acceleration = (late_rate - early_rate) / (early_rate + 1)
        else:
            acceleration = 0
        
        # Ğ”Ğ½Ğ¸ Ğ½Ğ° Ğ¿Ğ¾Ñ‚Ñ€Ğ°Ñ‚Ğ¸Ñ‚ÑŒ Ğ¼ĞµĞ´Ğ¸Ğ°Ğ½Ğ½ÑƒÑ� Ñ�ÑƒĞ¼Ğ¼Ñƒ
        median_amount = group['amount'].median()
        days_to_spend_median = median_amount / (daily_spend + 1)
        
        return pd.Series({
            'burn_rate': daily_spend,
            'acceleration': acceleration,
            'days_to_spend_median': days_to_spend_median
        })
    
    burn_features = transactions_sorted.groupby('client_num').apply(calculate_burn_rate).reset_index()
    features.append(burn_features)
    
    # === 3. FINANCIAL DISCIPLINE INDICATORS ===
    print("ğŸ�¯ Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ¸Ğ½Ğ´Ğ¸ĞºĞ°Ñ‚Ğ¾Ñ€Ñ‹ Ñ„Ğ¸Ğ½Ğ°Ğ½Ñ�Ğ¾Ğ²Ğ¾Ğ¹ Ğ´Ğ¸Ñ�Ñ†Ğ¸Ğ¿Ğ»Ğ¸Ğ½Ñ‹...")
    
    # Ğ ĞµĞ³ÑƒĞ»Ñ�Ñ€Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ñ‚Ñ€Ğ°Ñ‚
    daily_spending = transactions.groupby(['client_num', 'date'])['amount'].sum().reset_index()
    
    def calculate_discipline_metrics(group):
        amounts = group['amount']
        
        # ĞšĞ¾Ñ�Ñ„Ñ„Ğ¸Ñ†Ğ¸ĞµĞ½Ñ‚ Ğ²Ğ°Ñ€Ğ¸Ğ°Ñ†Ğ¸Ğ¸
        cv = amounts.std() / (amounts.mean() + 1)
        
        # ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ´Ğ½ĞµĞ¹ Ğ±ĞµĞ· Ñ‚Ñ€Ğ°Ñ‚
        total_days = (group['date'].max() - group['date'].min()).days + 1
        active_days = len(group)
        inactive_days = total_days - active_days
        
        # ĞœĞ°ĞºÑ�Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ°Ñ� Ğ¿Ğ°ÑƒĞ·Ğ° Ğ² Ñ‚Ñ€Ğ°Ñ‚Ğ°Ñ…
        dates_sorted = sorted(group['date'])
        max_gap = max([(dates_sorted[i+1] - dates_sorted[i]).days for i in range(len(dates_sorted)-1)] + [0])
        
        # Ğ¡Ñ‚Ğ°Ğ±Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ Ğ½ĞµĞ´ĞµĞ»ÑŒĞ½Ñ‹Ñ… Ñ‚Ñ€Ğ°Ñ‚
        group['week'] = pd.to_datetime(group['date']).dt.isocalendar().week
        weekly_spending = group.groupby('week')['amount'].sum()
        weekly_stability = 1 / (weekly_spending.std() / (weekly_spending.mean() + 1) + 1)
        
        return pd.Series({
            'spending_cv': cv,
            'inactive_days_ratio': inactive_days / total_days,
            'max_spending_gap': max_gap,
            'weekly_stability': weekly_stability
        })
    
    discipline_features = daily_spending.groupby('client_num').apply(calculate_discipline_metrics).reset_index()
    features.append(discipline_features)
    
    # === 4. ECONOMIC CYCLE SENSITIVITY ===
    print("ğŸ“Š Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ„Ğ¸Ñ‡Ğ¸ Ñ‡ÑƒĞ²Ñ�Ñ‚Ğ²Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ğ¾Ñ�Ñ‚Ğ¸ Ğº Ñ�ĞºĞ¾Ğ½Ğ¾Ğ¼Ğ¸Ñ‡ĞµÑ�ĞºĞ¸Ğ¼ Ñ†Ğ¸ĞºĞ»Ğ°Ğ¼...")
    
    # Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ· Ñ‚Ñ€Ğ°Ñ‚ Ğ² Ğ½Ğ°Ñ‡Ğ°Ğ»Ğµ/Ñ�ĞµÑ€ĞµĞ´Ğ¸Ğ½Ğµ/ĞºĞ¾Ğ½Ñ†Ğµ Ğ¼ĞµÑ�Ñ�Ñ†Ğ°
    transactions['day_of_month'] = pd.to_datetime(transactions['date_time']).dt.day
    transactions['month_period'] = pd.cut(transactions['day_of_month'], 
                                        bins=[0, 10, 20, 31], 
                                        labels=['month_start', 'month_mid', 'month_end'])
    
    # ĞŸĞ°Ñ‚Ñ‚ĞµÑ€Ğ½Ñ‹ Ñ‚Ñ€Ğ°Ñ‚ Ğ¿Ğ¾ Ğ¿ĞµÑ€Ğ¸Ğ¾Ğ´Ğ°Ğ¼ Ğ¼ĞµÑ�Ñ�Ñ†Ğ°
    monthly_patterns = transactions.groupby(['client_num', 'month_period'])['amount'].sum().unstack(fill_value=0)
    monthly_patterns.columns = [f'spending_{col}' for col in monthly_patterns.columns]
    monthly_patterns = monthly_patterns.reset_index()
    
    # ĞšĞ¾Ñ�Ñ„Ñ„Ğ¸Ñ†Ğ¸ĞµĞ½Ñ‚ Ğ½ĞµÑ€Ğ°Ğ²Ğ½Ğ¾Ğ¼ĞµÑ€Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ñ‚Ñ€Ğ°Ñ‚ Ğ¿Ğ¾ Ğ¼ĞµÑ�Ñ�Ñ†Ñƒ
    monthly_patterns['monthly_unevenness'] = monthly_patterns.iloc[:, 1:].std(axis=1) / (monthly_patterns.iloc[:, 1:].mean(axis=1) + 1)
    
    features.append(monthly_patterns)
    
    # === 5. LOYALTY & HABIT PATTERNS ===
    print("ğŸ”„ Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ„Ğ¸Ñ‡Ğ¸ Ğ»Ğ¾Ñ�Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ¸ Ğ¿Ñ€Ğ¸Ğ²Ñ‹Ñ‡ĞµĞº...")
    
    # Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ· Ğ¿Ğ¾Ğ²Ñ‚Ğ¾Ñ€Ğ½Ñ‹Ñ… Ğ¿Ğ¾ĞºÑƒĞ¿Ğ¾Ğº Ñƒ Ğ¾Ğ´Ğ½Ğ¸Ñ… Ğ¼ĞµÑ€Ñ‡Ğ°Ğ½Ñ‚Ğ¾Ğ²
    merchant_loyalty = transactions.groupby(['client_num', 'merchant_name']).agg({
        'amount': ['count', 'sum', 'mean'],
        'date_time': ['min', 'max']
    }).reset_index()
    
    merchant_loyalty.columns = ['client_num', 'merchant_name', 'visits', 'total_spent', 'avg_spent', 'first_visit', 'last_visit']
    
    # ĞœĞµÑ‚Ñ€Ğ¸ĞºĞ¸ Ğ»Ğ¾Ñ�Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚Ğ¸
    loyalty_metrics = merchant_loyalty.groupby('client_num').agg({
        'visits': ['max', 'mean', 'std'],
        'total_spent': ['max', 'mean'],
        'merchant_name': 'nunique'
    }).reset_index()
    loyalty_metrics.columns = ['client_num', 'max_visits_merchant', 'avg_visits_merchant', 'std_visits_merchant',
                             'max_spent_merchant', 'avg_spent_merchant', 'unique_merchants']
    
    # Ğ˜Ğ½Ğ´ĞµĞºÑ� Ğ»Ğ¾Ñ�Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚Ğ¸
    loyalty_metrics['loyalty_index'] = loyalty_metrics['max_visits_merchant'] / (loyalty_metrics['unique_merchants'] + 1)
    
    # ĞŸÑ€Ğ¸Ğ²Ñ‹Ñ‡ĞºĞ¸ Ğ¿Ğ¾ Ğ´Ğ½Ñ�Ğ¼ Ğ½ĞµĞ´ĞµĞ»Ğ¸
    weekday_habits = transactions.groupby(['client_num', 'weekday'])['amount'].sum().unstack(fill_value=0)
    weekday_habits.columns = [f'weekday_{col}_spending' for col in weekday_habits.columns]
    weekday_habits = weekday_habits.reset_index()
    
    # Ğ¡Ñ‚Ğ°Ğ±Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¿Ñ€Ğ¸Ğ²Ñ‹Ñ‡ĞµĞº
    weekday_habits['weekday_habit_stability'] = 1 / (weekday_habits.iloc[:, 1:].std(axis=1) / (weekday_habits.iloc[:, 1:].mean(axis=1) + 1) + 1)
    
    features.append(loyalty_metrics)
    features.append(weekday_habits)
    
    # === 6. TRANSACTION TIMING INTELLIGENCE ===
    print("â�° Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ„Ğ¸Ñ‡Ğ¸ Ğ¸Ğ½Ñ‚ĞµĞ»Ğ»ĞµĞºÑ‚ÑƒĞ°Ğ»ÑŒĞ½Ğ¾Ğ³Ğ¾ Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ° Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ¸...")
    
    # Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ· Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ¸ Ğ¼ĞµĞ¶Ğ´Ñƒ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ñ�Ğ¼Ğ¸
    def calculate_timing_features(group):
        if len(group) < 3:
            return pd.Series({'timing_regularity': 0, 'peak_activity_hour': 12, 'activity_concentration': 0})
        
        # Ğ¡Ğ¾Ñ€Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ğ¿Ğ¾ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ¸
        sorted_group = group.sort_values('date_time')
        
        # Ğ˜Ğ½Ñ‚ĞµÑ€Ğ²Ğ°Ğ»Ñ‹ Ğ¼ĞµĞ¶Ğ´Ñƒ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ñ�Ğ¼Ğ¸
        intervals = sorted_group['date_time'].diff().dt.total_seconds() / 3600  # Ğ² Ñ‡Ğ°Ñ�Ğ°Ñ…
        intervals = intervals.dropna()
        
        # Ğ ĞµĞ³ÑƒĞ»Ñ�Ñ€Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¿Ğ¾ Ğ¸Ğ½Ñ‚ĞµÑ€Ğ²Ğ°Ğ»Ğ°Ğ¼
        if len(intervals) > 1:
            timing_regularity = 1 / (intervals.std() / (intervals.mean() + 1) + 1)
        else:
            timing_regularity = 0
        
        # ĞŸĞ¸ĞºĞ¾Ğ²Ñ‹Ğ¹ Ñ‡Ğ°Ñ� Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚Ğ¸
        hour_counts = sorted_group['hour'].value_counts()
        peak_activity_hour = hour_counts.index[0] if len(hour_counts) > 0 else 12
        
        # ĞšĞ¾Ğ½Ñ†ĞµĞ½Ñ‚Ñ€Ğ°Ñ†Ğ¸Ñ� Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚Ğ¸ (Ñ�Ğ½Ñ‚Ñ€Ğ¾Ğ¿Ğ¸Ñ� Ğ¿Ğ¾ Ñ‡Ğ°Ñ�Ğ°Ğ¼)
        hour_probs = hour_counts / len(sorted_group)
        activity_concentration = -np.sum(hour_probs * np.log2(hour_probs + 1e-10))
        
        return pd.Series({
            'timing_regularity': timing_regularity,
            'peak_activity_hour': peak_activity_hour,
            'activity_concentration': activity_concentration
        })
    
    timing_features = transactions.groupby('client_num').apply(calculate_timing_features).reset_index()
    features.append(timing_features)
    
    # === 7. RISK INDICATORS ===
    print("âš ï¸� Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ¸Ğ½Ğ´Ğ¸ĞºĞ°Ñ‚Ğ¾Ñ€Ñ‹ Ñ„Ğ¸Ğ½Ğ°Ğ½Ñ�Ğ¾Ğ²Ğ¾Ğ³Ğ¾ Ñ€Ğ¸Ñ�ĞºĞ°...")
    
    # Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ· Ñ�ĞºÑ�Ñ‚Ñ€ĞµĞ¼Ğ°Ğ»ÑŒĞ½Ñ‹Ñ… Ñ‚Ñ€Ğ°Ñ‚
    def calculate_risk_indicators(group):
        amounts = group['amount']
        
        # ĞŸÑ€Ğ¾Ñ†ĞµĞ½Ñ‚ Ñ�ĞºÑ�Ñ‚Ñ€ĞµĞ¼Ğ°Ğ»ÑŒĞ½Ñ‹Ñ… Ñ‚Ñ€Ğ°Ñ‚
        q95 = amounts.quantile(0.95)
        extreme_ratio = (amounts > q95).mean()
        
        # ĞœĞ°ĞºÑ�Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ°Ñ� Ñ‚Ñ€Ğ°Ñ‚Ğ° Ğ¾Ñ‚Ğ½Ğ¾Ñ�Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ğ¾ Ğ¼ĞµĞ´Ğ¸Ğ°Ğ½Ñ‹
        max_vs_median = amounts.max() / (amounts.median() + 1)
        
        # Ğ’Ğ¾Ğ»Ğ°Ñ‚Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ Ñ‚Ñ€Ğ°Ñ‚
        volatility = amounts.std() / (amounts.mean() + 1)
        
        # Ğ¡ĞºĞ»Ğ¾Ğ½Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğº ĞºÑ€ÑƒĞ¿Ğ½Ñ‹Ğ¼ Ñ‚Ñ€Ğ°Ñ‚Ğ°Ğ¼ Ğ² ĞºĞ¾Ğ½Ñ†Ğµ Ğ¿ĞµÑ€Ğ¸Ğ¾Ğ´Ğ°
        sorted_amounts = group.sort_values('date_time')['amount']
        if len(sorted_amounts) >= 4:
            last_quarter = sorted_amounts.iloc[-len(sorted_amounts)//4:]
            first_quarter = sorted_amounts.iloc[:len(sorted_amounts)//4]
            end_period_bias = last_quarter.mean() / (first_quarter.mean() + 1)
        else:
            end_period_bias = 1
        
        return pd.Series({
            'extreme_spending_ratio': extreme_ratio,
            'max_vs_median_ratio': max_vs_median,
            'spending_volatility': volatility,
            'end_period_spending_bias': end_period_bias
        })
    
    risk_features = transactions.groupby('client_num').apply(calculate_risk_indicators).reset_index()
    features.append(risk_features)
    
    # === 8. CATEGORY DIVERSIFICATION ===
    print("ğŸ�¯ Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ„Ğ¸Ñ‡Ğ¸ Ğ´Ğ¸Ğ²ĞµÑ€Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ğ¸ Ñ‚Ñ€Ğ°Ñ‚...")
    
    # Ğ˜Ğ½Ğ´ĞµĞºÑ� Ğ´Ğ¸Ğ²ĞµÑ€Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ğ¸ Ğ¥ĞµÑ€Ñ„Ğ¸Ğ½Ğ´Ğ°Ğ»Ñ� Ğ´Ğ»Ñ� MCC ĞºĞ¾Ğ´Ğ¾Ğ²
    def calculate_herfindahl_index(group):
        mcc_counts = group['mcc_code'].value_counts(normalize=True)
        hhi = (mcc_counts ** 2).sum()
        return 1 - hhi  # Ğ˜Ğ½Ğ²ĞµÑ€Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ğ´Ğ»Ñ� Ğ¸Ğ½Ñ‚ĞµÑ€Ğ¿Ñ€ĞµÑ‚Ğ°Ñ†Ğ¸Ğ¸ ĞºĞ°Ğº Ğ´Ğ¸Ğ²ĞµÑ€Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ğ¸
    
    # Ğ”Ğ¸Ğ²ĞµÑ€Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ñ� Ğ¿Ğ¾ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ñ�Ğ¼
    diversification_mcc = transactions.groupby('client_num').apply(calculate_herfindahl_index).reset_index()
    diversification_mcc.columns = ['client_num', 'mcc_diversification']
    
    # Ğ”Ğ¸Ğ²ĞµÑ€Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ñ� Ğ¿Ğ¾ Ğ¼ĞµÑ€Ñ‡Ğ°Ğ½Ñ‚Ğ°Ğ¼
    def calculate_merchant_diversification(group):
        merchant_counts = group['merchant_name'].value_counts(normalize=True)
        hhi = (merchant_counts ** 2).sum()
        return 1 - hhi
    
    diversification_merchant = transactions.groupby('client_num').apply(calculate_merchant_diversification).reset_index()
    diversification_merchant.columns = ['client_num', 'merchant_diversification']
    
    # Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼ Ğ´Ğ¸Ğ²ĞµÑ€Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ğ¸
    diversification_features = diversification_mcc.merge(diversification_merchant, on='client_num', how='outer')
    
    # Ğ�Ğ±Ñ‰Ğ¸Ğ¹ Ğ¸Ğ½Ğ´ĞµĞºÑ� Ğ´Ğ¸Ğ²ĞµÑ€Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ğ¸
    diversification_features['total_diversification'] = (diversification_features['mcc_diversification'] + 
                                                       diversification_features['merchant_diversification']) / 2
    
    features.append(diversification_features)
    
    # === 9. PREDICTIVE LAG FEATURES ===
    print("ğŸ“ˆ Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ¿Ñ€ĞµĞ´Ğ¸ĞºÑ‚Ğ¸Ğ²Ğ½Ñ‹Ğµ Ğ»Ğ°Ğ³Ğ¾Ğ²Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸...")
    
    # Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ´Ğ½ĞµĞ²Ğ½Ñ‹Ğµ Ğ°Ğ³Ñ€ĞµĞ³Ğ°Ñ‚Ñ‹
    daily_aggs = transactions.groupby(['client_num', 'date']).agg({
        'amount': ['sum', 'count', 'mean'],
        'mcc_code': 'nunique'
    }).reset_index()
    daily_aggs.columns = ['client_num', 'date', 'daily_amount', 'daily_txns', 'daily_avg_amount', 'daily_unique_mcc']
    
    # Ğ›Ğ°Ğ³Ğ¾Ğ²Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸ Ğ¿Ğ¾Ñ�Ğ»ĞµĞ´Ğ½Ğ¸Ñ… Ğ´Ğ½ĞµĞ¹
    def create_lag_features(group, lags=[1, 3, 7, 14]):
        group = group.sort_values('date')
        lag_features = {}
        
        for lag in lags:
            lag_features[f'amount_lag_{lag}'] = group['daily_amount'].shift(lag).fillna(0).iloc[-1] if len(group) > lag else 0
            lag_features[f'txns_lag_{lag}'] = group['daily_txns'].shift(lag).fillna(0).iloc[-1] if len(group) > lag else 0
            lag_features[f'avg_amount_lag_{lag}'] = group['daily_avg_amount'].shift(lag).fillna(0).iloc[-1] if len(group) > lag else 0
        
        return pd.Series(lag_features)
    
    lag_features = daily_aggs.groupby('client_num').apply(create_lag_features).reset_index()
    features.append(lag_features)
    
    # === 10. MCC-SPECIFIC BEHAVIORAL PATTERNS ===
    print("ğŸ�¯ Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ MCC-Ñ�Ğ¿ĞµÑ†Ğ¸Ñ„Ğ¸Ñ‡Ğ½Ñ‹Ğµ Ğ¿Ğ°Ñ‚Ñ‚ĞµÑ€Ğ½Ñ‹...")
    
    # Ğ�Ñ�Ğ½Ğ¾Ğ²Ñ‹Ğ²Ğ°ĞµĞ¼Ñ�Ñ� Ğ½Ğ° Ğ’Ğ�Ğ¨Ğ˜Ğ¥ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
    top_frequent_mccs = [5411, 5499, 5814, 4131, 3990]  # Ğ¢Ğ¾Ğ¿-5 Ğ¿Ğ¾ Ñ‡Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ğµ
    top_expensive_mccs = [6011, 5511, 3011, 4722, 5712]  # Ğ¢Ğ¾Ğ¿-5 Ğ¿Ğ¾ Ñ�Ñ€ĞµĞ´Ğ½ĞµĞ¹ Ñ�ÑƒĞ¼Ğ¼Ğµ
    
    # ĞŸĞ°Ñ‚Ñ‚ĞµÑ€Ğ½Ñ‹ Ğ¿Ğ¾ Ñ‚Ğ¾Ğ¿ MCC
    for mcc in top_frequent_mccs:
        mcc_data = transactions[transactions['mcc_code'] == mcc]
        if len(mcc_data) > 0:
            mcc_stats = mcc_data.groupby('client_num').agg({
                'amount': ['sum', 'mean', 'count']
            }).reset_index()
            mcc_stats.columns = ['client_num', f'mcc_{mcc}_total', f'mcc_{mcc}_avg', f'mcc_{mcc}_count']
            features.append(mcc_stats)
    
    # ĞšĞ¾Ğ½Ñ†ĞµĞ½Ñ‚Ñ€Ğ°Ñ†Ğ¸Ñ� Ñ‚Ñ€Ğ°Ñ‚ Ğ² Ñ‚Ğ¾Ğ¿ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ñ�Ñ…
    top_mcc_concentration = transactions[transactions['mcc_code'].isin(top_frequent_mccs)].groupby('client_num').agg({
        'amount': 'sum'
    }).reset_index()
    
    total_spending = transactions.groupby('client_num')['amount'].sum().reset_index()
    total_spending.columns = ['client_num', 'total_amount']
    
    concentration_features = top_mcc_concentration.merge(total_spending, on='client_num', how='right').fillna(0)
    concentration_features['top_mcc_concentration'] = concentration_features['amount'] / (concentration_features['total_amount'] + 1)
    concentration_features = concentration_features[['client_num', 'top_mcc_concentration']]
    features.append(concentration_features)
    
    # === 10. WEEKEND/WEEKDAY BEHAVIOR ANALYSIS ===
    print("ğŸ“… Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ· Ğ¿Ğ¾Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ� Ğ²Ñ‹Ñ…Ğ¾Ğ´Ğ½Ñ‹Ğµ/Ğ±ÑƒĞ´Ğ½Ğ¸...")
    
    transactions['is_weekend'] = transactions['weekday'] >= 5
    
    # Ğ¡Ñ€Ğ°Ğ²Ğ½ĞµĞ½Ğ¸Ğµ Ğ¿Ğ¾Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ� Ğ² Ğ²Ñ‹Ñ…Ğ¾Ğ´Ğ½Ñ‹Ğµ Ğ¸ Ğ±ÑƒĞ´Ğ½Ğ¸ - Ğ˜Ğ¡ĞŸĞ Ğ�Ğ’Ğ›Ğ•Ğ�Ğ�Ğ�Ğ¯ Ğ’Ğ•Ğ Ğ¡Ğ˜Ğ¯
    weekend_stats = transactions.groupby(['client_num', 'is_weekend']).agg({
        'amount': ['sum', 'mean', 'count'],
        'mcc_code': 'nunique'
    }).reset_index()
    
    # Ğ£Ğ±Ğ¸Ñ€Ğ°ĞµĞ¼ Ğ¼ÑƒĞ»ÑŒÑ‚Ğ¸Ğ¸Ğ½Ğ´ĞµĞºÑ� ĞºĞ¾Ğ»Ğ¾Ğ½Ğ¾Ğº
    weekend_stats.columns = ['client_num', 'is_weekend', 'amount_sum', 'amount_mean', 'amount_count', 'unique_mcc']
    
    # Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ¾Ñ‚Ğ´ĞµĞ»ÑŒĞ½Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸ Ğ´Ğ»Ñ� Ğ²Ñ‹Ñ…Ğ¾Ğ´Ğ½Ñ‹Ñ… Ğ¸ Ğ±ÑƒĞ´Ğ½ĞµĞ¹
    weekend_data = weekend_stats[weekend_stats['is_weekend'] == True][['client_num', 'amount_sum', 'amount_mean', 'amount_count', 'unique_mcc']]
    weekend_data.columns = ['client_num', 'weekend_amount_sum', 'weekend_amount_mean', 'weekend_amount_count', 'weekend_unique_mcc']
    
    weekday_data = weekend_stats[weekend_stats['is_weekend'] == False][['client_num', 'amount_sum', 'amount_mean', 'amount_count', 'unique_mcc']]
    weekday_data.columns = ['client_num', 'weekday_amount_sum', 'weekday_amount_mean', 'weekday_amount_count', 'weekday_unique_mcc']
    
    # Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼ Ğ¸ Ñ�Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ�Ğ¾Ğ¾Ñ‚Ğ½Ğ¾ÑˆĞµĞ½Ğ¸Ñ�
    weekend_combined = weekend_data.merge(weekday_data, on='client_num', how='outer').fillna(0)
    
    # Ğ¡Ğ¾Ğ¾Ñ‚Ğ½Ğ¾ÑˆĞµĞ½Ğ¸Ñ� Ğ²Ñ‹Ñ…Ğ¾Ğ´Ğ½Ñ‹Ğµ/Ğ±ÑƒĞ´Ğ½Ğ¸
    weekend_combined['weekend_weekday_amount_ratio'] = weekend_combined['weekend_amount_sum'] / (weekend_combined['weekday_amount_sum'] + 1)
    weekend_combined['weekend_weekday_count_ratio'] = weekend_combined['weekend_amount_count'] / (weekend_combined['weekday_amount_count'] + 1)
    weekend_combined['weekend_activity_preference'] = (weekend_combined['weekend_amount_count'] / 
                                                     (weekend_combined['weekend_amount_count'] + weekend_combined['weekday_amount_count'] + 1))
    
    features.append(weekend_combined)
    
    # === Ğ�Ğ‘ĞªĞ•Ğ”Ğ˜Ğ�Ğ•Ğ�Ğ˜Ğ• Ğ’Ğ¡Ğ•Ğ¥ Ğ¤Ğ˜Ğ§Ğ•Ğ™ ===
    print("ğŸ”— Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼ Ğ²Ñ�Ğµ balance-specific Ñ„Ğ¸Ñ‡Ğ¸...")
    
    # Ğ�Ğ°Ñ‡Ğ¸Ğ½Ğ°ĞµĞ¼ Ñ� Ğ¿ĞµÑ€Ğ²Ğ¾Ğ³Ğ¾ Ğ½Ğ°Ğ±Ğ¾Ñ€Ğ°
    final_features = features[0]
    
    # ĞŸĞ¾Ñ�Ğ»ĞµĞ´Ğ¾Ğ²Ğ°Ñ‚ĞµĞ»ÑŒĞ½Ğ¾ Ğ´Ğ¶Ğ¾Ğ¹Ğ½Ğ¸Ğ¼ Ğ²Ñ�Ğµ Ğ¾Ñ�Ñ‚Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ
    for i, feature_set in enumerate(features[1:], 1):
        print(f"   Ğ”Ğ¶Ğ¾Ğ¹Ğ½Ğ¸Ğ¼ Ğ½Ğ°Ğ±Ğ¾Ñ€ {i+1}/{len(features)}: {feature_set.shape}")
        final_features = final_features.merge(feature_set, on='client_num', how='outer')
    
    # Ğ—Ğ°Ğ¿Ğ¾Ğ»Ğ½Ñ�ĞµĞ¼ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¸
    final_features = final_features.fillna(0)
    
    # Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ ĞºĞ¾Ğ¼Ğ¿Ğ¾Ğ·Ğ¸Ñ‚Ğ½Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸
    print("ğŸ”„ Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ ĞºĞ¾Ğ¼Ğ¿Ğ¾Ğ·Ğ¸Ñ‚Ğ½Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸...")
    
    # Ğ¤Ğ¸Ğ½Ğ°Ğ½Ñ�Ğ¾Ğ²Ğ°Ñ� Ñ�Ñ‚Ğ°Ğ±Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ (ĞºĞ¾Ğ¼Ğ¿Ğ¾Ğ·Ğ¸Ñ‚Ğ½Ğ°Ñ� Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ°)
    stability_cols = ['weekly_stability', 'timing_regularity', 'spending_cv', 'monthly_unevenness']
    available_stability_cols = [col for col in stability_cols if col in final_features.columns]
    if available_stability_cols:
        final_features['financial_stability_score'] = final_features[available_stability_cols].mean(axis=1)
    
    # Ğ Ğ¸Ñ�Ğº Ğ±Ğ°Ğ½ĞºÑ€Ğ¾Ñ‚Ñ�Ñ‚Ğ²Ğ° (ĞºĞ¾Ğ¼Ğ¿Ğ¾Ğ·Ğ¸Ñ‚Ğ½Ğ°Ñ� Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ°)
    risk_cols = ['spending_volatility', 'extreme_spending_ratio', 'burn_rate', 'acceleration']
    available_risk_cols = [col for col in risk_cols if col in final_features.columns]
    if available_risk_cols:
        final_features['bankruptcy_risk_score'] = final_features[available_risk_cols].mean(axis=1)
    
    # Ğ˜Ğ½Ğ´ĞµĞºÑ� Ñ„Ğ¸Ğ½Ğ°Ğ½Ñ�Ğ¾Ğ²Ğ¾Ğ³Ğ¾ Ğ·Ğ´Ğ¾Ñ€Ğ¾Ğ²ÑŒÑ�
    if 'financial_stability_score' in final_features.columns and 'bankruptcy_risk_score' in final_features.columns:
        final_features['financial_health_index'] = final_features['financial_stability_score'] - final_features['bankruptcy_risk_score']
    
    print(f"âœ… Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¾ {final_features.shape[1]-1} balance-specific Ñ„Ğ¸Ñ‡ĞµĞ¹ Ğ´Ğ»Ñ� {final_features.shape[0]} ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ¾Ğ²")
    
    return final_features


# === INTEGRATION FUNCTION ===
def integrate_all_features(original_features, transactions, train_clients):
    """
    Ğ˜Ğ½Ñ‚ĞµĞ³Ñ€Ğ¸Ñ€ÑƒĞµÑ‚ Ğ¾Ñ€Ğ¸Ğ³Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸ Ñ� Ğ½Ğ¾Ğ²Ñ‹Ğ¼Ğ¸ Ğ¿Ñ€Ğ¾Ğ´Ğ²Ğ¸Ğ½ÑƒÑ‚Ñ‹Ğ¼Ğ¸ Ñ„Ğ¸Ñ‡Ğ°Ğ¼Ğ¸
    """
    print("ğŸš€ Ğ�Ğ°Ñ‡Ğ¸Ğ½Ğ°ĞµĞ¼ Ñ�Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¸Ğµ Ğ²Ñ�ĞµÑ… Ğ½Ğ¾Ğ²Ñ‹Ñ… Ñ„Ğ¸Ñ‡ĞµĞ¹...")
    
    # Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ¿Ñ€Ğ¾Ğ´Ğ²Ğ¸Ğ½ÑƒÑ‚Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸
    advanced_features = create_advanced_features(transactions, train_clients)
    
    # Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ balance-specific Ñ„Ğ¸Ñ‡Ğ¸
    balance_features = create_balance_specific_features(transactions, train_clients)
    
    # Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼ Ğ²Ñ�Ğµ Ñ„Ğ¸Ñ‡Ğ¸
    print("ğŸ”— Ğ˜Ğ½Ñ‚ĞµĞ³Ñ€Ğ¸Ñ€ÑƒĞµĞ¼ Ñ� Ğ¾Ñ€Ğ¸Ğ³Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¼Ğ¸ Ñ„Ğ¸Ñ‡Ğ°Ğ¼Ğ¸...")
    
    # Ğ”Ğ¶Ğ¾Ğ¹Ğ½Ğ¸Ğ¼ Ñ� Ğ¾Ñ€Ğ¸Ğ³Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¼Ğ¸ Ñ„Ğ¸Ñ‡Ğ°Ğ¼Ğ¸
    enhanced_features = original_features.merge(advanced_features, on='client_num', how='left')
    enhanced_features = enhanced_features.merge(balance_features, on='client_num', how='left')
    
    # Ğ—Ğ°Ğ¿Ğ¾Ğ»Ğ½Ñ�ĞµĞ¼ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¸
    enhanced_features = enhanced_features.fillna(0)
    
    print(f"ğŸ�‰ Ğ˜Ñ‚Ğ¾Ğ³Ğ¾ Ñ�Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¾ {enhanced_features.shape[1]-1} Ñ„Ğ¸Ñ‡ĞµĞ¹!")
    print(f"   Ğ�Ñ€Ğ¸Ğ³Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ñ‹Ñ…: {original_features.shape[1]-1}")
    print(f"   Ğ�Ğ¾Ğ²Ñ‹Ñ… Ğ¿Ñ€Ğ¾Ğ´Ğ²Ğ¸Ğ½ÑƒÑ‚Ñ‹Ñ…: {advanced_features.shape[1]-1}")
    print(f"   Ğ�Ğ¾Ğ²Ñ‹Ñ… balance-specific: {balance_features.shape[1]-1}")
    
    # Ğ’Ñ‹Ğ²Ğ¾Ğ´Ğ¸Ğ¼ Ğ½Ğ¾Ğ²Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸
    original_cols = set(original_features.columns)
    new_cols = [col for col in enhanced_features.columns if col not in original_cols]
    print(f"\nğŸ“‹ Ğ¡Ğ¿Ğ¸Ñ�Ğ¾Ğº Ğ½Ğ¾Ğ²Ñ‹Ñ… Ñ„Ğ¸Ñ‡ĞµĞ¹ ({len(new_cols)}):")
    for i, col in enumerate(new_cols[:20]):  # ĞŸĞ¾ĞºĞ°Ğ·Ñ‹Ğ²Ğ°ĞµĞ¼ Ğ¿ĞµÑ€Ğ²Ñ‹Ğµ 20
        print(f"   {i+1:2d}. {col}")
    if len(new_cols) > 20:
        print(f"   ... Ğ¸ ĞµÑ‰Ğµ {len(new_cols)-20} Ñ„Ğ¸Ñ‡ĞµĞ¹")
    
    return enhanced_features


import pandas as pd
import numpy as np

def create_data_driven_features(transactions, train_clients):
    """
    Ğ§Ğ˜Ğ¡Ğ¢Ğ«Ğ• data-driven Ñ„Ğ¸Ñ‡Ğ¸ Ğ±ĞµĞ· Ğ±Ğ°Ğ³Ğ¾Ğ² - Ğ½Ğ° Ğ¾Ñ�Ğ½Ğ¾Ğ²Ğµ Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ° Ğ²Ğ°ÑˆĞ¸Ñ… MCC
    """
    print("ğŸ�¯ Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ§Ğ˜Ğ¡Ğ¢Ğ«Ğ• data-driven Ñ„Ğ¸Ñ‡Ğ¸...")
    
    # ĞŸĞ¾Ğ´Ğ³Ğ¾Ñ‚Ğ°Ğ²Ğ»Ğ¸Ğ²Ğ°ĞµĞ¼ Ğ±Ğ°Ğ·Ğ¾Ğ²Ñ‹Ğµ Ğ¿Ğ¾Ğ»Ñ�
    if 'hour' not in transactions.columns:
        transactions['hour'] = pd.to_datetime(transactions['date_time']).dt.hour
    if 'weekday' not in transactions.columns:
        transactions['weekday'] = pd.to_datetime(transactions['date_time']).dt.weekday
    if 'date' not in transactions.columns:
        transactions['date'] = pd.to_datetime(transactions['date_time']).dt.date
    
    # Ğ‘Ğ°Ğ·Ğ¾Ğ²Ñ‹Ğ¹ Ñ�Ğ¿Ğ¸Ñ�Ğ¾Ğº Ğ²Ñ�ĞµÑ… ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ¾Ğ²
    all_clients = pd.DataFrame({'client_num': transactions['client_num'].unique()})
    features_list = []
    
    # === 1. Ğ¤Ğ˜Ğ§Ğ˜ Ğ”Ğ›Ğ¯ Ğ”Ğ�ĞœĞ˜Ğ�Ğ˜Ğ Ğ£Ğ®Ğ©Ğ•Ğ“Ğ� MCC 5411 ===
    print("   ğŸ“Š MCC 5411 Ñ„Ğ¸Ñ‡Ğ¸...")
    
    mcc_5411_data = transactions[transactions['mcc_code'] == 5411]
    if len(mcc_5411_data) > 0:
        mcc_5411_features = mcc_5411_data.groupby('client_num').agg({
            'amount': ['sum', 'mean', 'count', 'std'],
            'hour': 'mean',
            'weekday': 'mean'
        })
        
        # ĞŸÑ€Ğ¾Ñ�Ñ‚Ñ‹Ğµ Ğ½Ğ°Ğ·Ğ²Ğ°Ğ½Ğ¸Ñ� ĞºĞ¾Ğ»Ğ¾Ğ½Ğ¾Ğº
        mcc_5411_features.columns = [
            'mcc5411_total', 'mcc5411_avg', 'mcc5411_count', 'mcc5411_std',
            'mcc5411_avg_hour', 'mcc5411_avg_weekday'
        ]
        mcc_5411_features = mcc_5411_features.reset_index()
        
        # Ğ”Ğ¶Ğ¾Ğ¹Ğ½Ğ¸Ğ¼ ĞºĞ¾ Ğ²Ñ�ĞµĞ¼ ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ°Ğ¼ Ğ¸ Ğ·Ğ°Ğ¿Ğ¾Ğ»Ğ½Ñ�ĞµĞ¼ Ğ½ÑƒĞ»Ğ¸
        mcc_5411_final = all_clients.merge(mcc_5411_features, on='client_num', how='left').fillna(0)
        features_list.append(mcc_5411_final)
    
    # === 2. Ğ¤Ğ˜Ğ§Ğ˜ Ğ”Ğ›Ğ¯ MCC 5499 ===
    print("   ğŸ“Š MCC 5499 Ñ„Ğ¸Ñ‡Ğ¸...")
    
    mcc_5499_data = transactions[transactions['mcc_code'] == 5499]
    if len(mcc_5499_data) > 0:
        mcc_5499_features = mcc_5499_data.groupby('client_num').agg({
            'amount': ['sum', 'mean', 'count']
        })
        
        mcc_5499_features.columns = ['mcc5499_total', 'mcc5499_avg', 'mcc5499_count']
        mcc_5499_features = mcc_5499_features.reset_index()
        
        mcc_5499_final = all_clients.merge(mcc_5499_features, on='client_num', how='left').fillna(0)
        features_list.append(mcc_5499_final)
    
    # === 3. Ğ¤Ğ˜Ğ§Ğ˜ Ğ”Ğ›Ğ¯ Ğ”Ğ�Ğ Ğ�Ğ“Ğ�Ğ“Ğ� MCC 6011 ===
    print("   ğŸ’° MCC 6011 Ñ„Ğ¸Ñ‡Ğ¸...")
    
    mcc_6011_data = transactions[transactions['mcc_code'] == 6011]
    if len(mcc_6011_data) > 0:
        mcc_6011_features = mcc_6011_data.groupby('client_num').agg({
            'amount': ['sum', 'mean', 'count', 'max']
        })
        
        mcc_6011_features.columns = ['mcc6011_total', 'mcc6011_avg', 'mcc6011_count', 'mcc6011_max']
        mcc_6011_features = mcc_6011_features.reset_index()
        
        mcc_6011_final = all_clients.merge(mcc_6011_features, on='client_num', how='left').fillna(0)
        features_list.append(mcc_6011_final)
    
    # === 4. ĞšĞ�Ğ�Ğ¦Ğ•Ğ�Ğ¢Ğ Ğ�Ğ¦Ğ˜Ğ¯ Ğ�Ğ� Ğ¢Ğ�ĞŸ-3 MCC ===
    print("   ğŸ�¯ ĞšĞ¾Ğ½Ñ†ĞµĞ½Ñ‚Ñ€Ğ°Ñ†Ğ¸Ñ� Ñ„Ğ¸Ñ‡Ğ¸...")
    
    top3_mccs = [5411, 5499, 5814]
    
    # Ğ¢Ñ€Ğ°Ñ‚Ñ‹ Ğ² Ñ‚Ğ¾Ğ¿-3
    top3_spending = transactions[transactions['mcc_code'].isin(top3_mccs)].groupby('client_num')['amount'].sum().reset_index()
    top3_spending.columns = ['client_num', 'top3_spending']
    
    # Ğ�Ğ±Ñ‰Ğ¸Ğµ Ñ‚Ñ€Ğ°Ñ‚Ñ‹
    total_spending = transactions.groupby('client_num')['amount'].sum().reset_index()
    total_spending.columns = ['client_num', 'total_spending']
    
    # ĞšĞ¾Ğ½Ñ†ĞµĞ½Ñ‚Ñ€Ğ°Ñ†Ğ¸Ñ�
    concentration = total_spending.merge(top3_spending, on='client_num', how='left').fillna(0)
    concentration['top3_concentration'] = concentration['top3_spending'] / (concentration['total_spending'] + 1)
    concentration_final = concentration[['client_num', 'top3_concentration']]
    
    features_list.append(concentration_final)
    
    # === 5. Ğ’Ğ«Ğ¡Ğ�ĞšĞ˜Ğ• Ğ¢Ğ Ğ�Ğ�Ğ—Ğ�ĞšĞ¦Ğ˜Ğ˜ ===
    print("   ğŸ’� Ğ’Ñ‹Ñ�Ğ¾ĞºĞ¸Ğµ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸...")
    
    high_threshold = transactions['amount'].quantile(0.8)
    high_txns = transactions[transactions['amount'] >= high_threshold]
    
    if len(high_txns) > 0:
        high_features = high_txns.groupby('client_num').agg({
            'amount': ['sum', 'count', 'mean'],
            'mcc_code': 'nunique'
        })
        
        high_features.columns = ['high_total', 'high_count', 'high_avg', 'high_unique_mcc']
        high_features = high_features.reset_index()
        
        # Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ğ´Ğ¾Ğ»Ñ� Ğ²Ñ‹Ñ�Ğ¾ĞºĞ¸Ñ… Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹
        high_with_total = high_features.merge(total_spending, on='client_num', how='right').fillna(0)
        high_with_total['high_ratio'] = high_with_total['high_total'] / (high_with_total['total_spending'] + 1)
        
        high_final = high_with_total[['client_num', 'high_total', 'high_count', 'high_avg', 'high_unique_mcc', 'high_ratio']]
        features_list.append(high_final)
    
    # === 6. Ğ§Ğ�Ğ¡Ğ¢Ğ«Ğ• VS Ğ Ğ•Ğ”ĞšĞ˜Ğ• ĞšĞ�Ğ¢Ğ•Ğ“Ğ�Ğ Ğ˜Ğ˜ ===
    print("   ğŸ“ˆ Ğ§Ğ°Ñ�Ñ‚Ñ‹Ğµ vs Ñ€ĞµĞ´ĞºĞ¸Ğµ...")
    
    # Ğ§Ğ°Ñ�Ñ‚Ñ‹Ğµ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¸ (Ñ‚Ğ¾Ğ¿-5)
    frequent_mccs = [5411, 5499, 5814, 4131, 3990]
    frequent_spending = transactions[transactions['mcc_code'].isin(frequent_mccs)].groupby('client_num')['amount'].sum().reset_index()
    frequent_spending.columns = ['client_num', 'frequent_spending']
    
    # Ğ ĞµĞ´ĞºĞ¸Ğµ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¸ (Ğ¿Ğ¾Ñ�Ğ»ĞµĞ´Ğ½Ğ¸Ğµ 50)
    mcc_counts = transactions['mcc_code'].value_counts()
    rare_mccs = mcc_counts.tail(50).index.tolist()
    rare_spending = transactions[transactions['mcc_code'].isin(rare_mccs)].groupby('client_num')['amount'].sum().reset_index()
    rare_spending.columns = ['client_num', 'rare_spending']
    
    # Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼
    freq_rare = all_clients.merge(frequent_spending, on='client_num', how='left').fillna(0)
    freq_rare = freq_rare.merge(rare_spending, on='client_num', how='left').fillna(0)
    
    # Ğ¡Ğ¾Ğ¾Ñ‚Ğ½Ğ¾ÑˆĞµĞ½Ğ¸Ñ�
    freq_rare['frequent_rare_ratio'] = freq_rare['frequent_spending'] / (freq_rare['rare_spending'] + 1)
    freq_rare['rare_tendency'] = freq_rare['rare_spending'] / (freq_rare['frequent_spending'] + freq_rare['rare_spending'] + 1)
    
    features_list.append(freq_rare)
    
    # === 7. ĞŸĞ•Ğ Ğ•ĞšĞ›Ğ®Ğ§Ğ•Ğ�Ğ˜Ğ¯ ĞœĞ•Ğ–Ğ”Ğ£ ĞšĞ�Ğ¢Ğ•Ğ“Ğ�Ğ Ğ˜Ğ¯ĞœĞ˜ ===
    print("   ğŸ”„ ĞŸĞµÑ€ĞµĞºĞ»Ñ�Ñ‡ĞµĞ½Ğ¸Ñ� ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¹...")
    
    def calculate_switches(group):
        if len(group) < 2:
            return pd.Series({'category_switches': 0, 'switch_rate': 0})
        
        sorted_group = group.sort_values('date_time')
        mcc_sequence = sorted_group['mcc_code'].tolist()
        
        switches = sum(1 for i in range(len(mcc_sequence)-1) 
                      if mcc_sequence[i] != mcc_sequence[i+1])
        switch_rate = switches / len(mcc_sequence) if len(mcc_sequence) > 1 else 0
        
        return pd.Series({
            'category_switches': switches,
            'switch_rate': switch_rate
        })
    
    switch_features = transactions.groupby('client_num').apply(calculate_switches)
    switch_features = switch_features.reset_index()
    
    features_list.append(switch_features)
    
    # === 8. Ğ¡Ğ¢Ğ�Ğ‘Ğ˜Ğ›Ğ¬Ğ�Ğ�Ğ¡Ğ¢Ğ¬ ĞŸĞ�Ğ’Ğ•Ğ”Ğ•Ğ�Ğ˜Ğ¯ ===
    print("   ğŸ“Š Ğ¡Ñ‚Ğ°Ğ±Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¿Ğ¾Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ�...")
    
    # ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ ÑƒĞ½Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ñ… Ğ´Ğ½ĞµĞ¹ Ñ� Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ñ�Ğ¼Ğ¸
    active_days = transactions.groupby('client_num')['date'].nunique().reset_index()
    active_days.columns = ['client_num', 'unique_active_days']
    
    # Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸ Ğ² Ñ€Ğ°Ğ·Ğ½Ñ‹Ğµ Ğ´Ğ½Ğ¸ Ğ½ĞµĞ´ĞµĞ»Ğ¸
    weekday_std = transactions.groupby(['client_num', 'weekday'])['amount'].mean().groupby('client_num').std().reset_index()
    weekday_std.columns = ['client_num', 'weekday_amount_stability']
    weekday_std['weekday_amount_stability'] = weekday_std['weekday_amount_stability'].fillna(0)
    
    # Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼ Ñ�Ñ‚Ğ°Ğ±Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ
    stability = active_days.merge(weekday_std, on='client_num', how='left').fillna(0)
    features_list.append(stability)
    
    # === 9. Ğ’Ğ Ğ•ĞœĞ•Ğ�Ğ�Ğ«Ğ• ĞŸĞ�Ğ¢Ğ¢Ğ•Ğ Ğ�Ğ« ===
    print("   â�° Ğ’Ñ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ğµ Ğ¿Ğ°Ñ‚Ñ‚ĞµÑ€Ğ½Ñ‹...")
    
    # Ğ›Ñ�Ğ±Ğ¸Ğ¼Ğ¾Ğµ Ğ²Ñ€ĞµĞ¼Ñ� Ğ´Ğ»Ñ� Ñ‚Ñ€Ğ°Ñ‚
    time_patterns = transactions.groupby('client_num').agg({
        'hour': ['mean', 'std'],
        'weekday': ['mean', 'std']
    })
    
    time_patterns.columns = ['avg_hour', 'std_hour', 'avg_weekday', 'std_weekday']
    time_patterns = time_patterns.reset_index()
    time_patterns = time_patterns.fillna(0)
    
    features_list.append(time_patterns)
    
    # === 10. Ğ�Ğ‘ĞªĞ•Ğ”Ğ˜Ğ�Ğ•Ğ�Ğ˜Ğ• Ğ’Ğ¡Ğ•Ğ¥ Ğ¤Ğ˜Ğ§Ğ•Ğ™ ===
    print("   ğŸ”— Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼ Ğ²Ñ�Ğµ Ñ„Ğ¸Ñ‡Ğ¸...")
    
    # Ğ�Ğ°Ñ‡Ğ¸Ğ½Ğ°ĞµĞ¼ Ñ� all_clients
    final_features = all_clients.copy()
    
    # ĞŸĞ¾Ñ�Ğ»ĞµĞ´Ğ¾Ğ²Ğ°Ñ‚ĞµĞ»ÑŒĞ½Ğ¾ Ğ´Ğ¶Ğ¾Ğ¹Ğ½Ğ¸Ğ¼ Ğ²Ñ�Ğµ Ñ„Ğ¸Ñ‡Ğ¸
    for i, feature_df in enumerate(features_list):
        print(f"      Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ğ½Ğ°Ğ±Ğ¾Ñ€ {i+1}/{len(features_list)}: {feature_df.shape}")
        final_features = final_features.merge(feature_df, on='client_num', how='left')
    
    # Ğ¤Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ğ¾Ğµ Ğ·Ğ°Ğ¿Ğ¾Ğ»Ğ½ĞµĞ½Ğ¸Ğµ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¾Ğ²
    final_features = final_features.fillna(0)
    
    print(f"âœ… Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¾ {final_features.shape[1]-1} Ñ‡Ğ¸Ñ�Ñ‚Ñ‹Ñ… data-driven Ñ„Ğ¸Ñ‡ĞµĞ¹ Ğ´Ğ»Ñ� {final_features.shape[0]} ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ¾Ğ²")
    
    return final_features

# # === Ğ¢Ğ•Ğ¡Ğ¢Ğ˜Ğ Ğ�Ğ’Ğ�Ğ�Ğ˜Ğ• Ğ¤Ğ£Ğ�ĞšĞ¦Ğ˜Ğ˜ ===
# if __name__ == "__main__":
#     # ĞŸÑ€Ğ¾Ñ�Ñ‚Ğ¾Ğ¹ Ñ‚ĞµÑ�Ñ‚
#     print("ğŸ§ª Ğ¢ĞµÑ�Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ñ„ÑƒĞ½ĞºÑ†Ğ¸Ñ�...")
    
#     # Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ñ‹Ğµ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ
#     test_data = pd.DataFrame({
#         'client_num': [1, 1, 2, 2, 3],
#         'date_time': pd.to_datetime(['2024-01-01 10:00', '2024-01-02 15:00', 
#                                    '2024-01-01 12:00', '2024-01-03 18:00', 
#                                    '2024-01-01 09:00']),
#         'mcc_code': [5411, 5499, 5411, 6011, 5814],
#         'amount': [1000, 500, 1500, 50000, 800]
#     })
    
#     test_clients = {1, 2}
    
#     try:
#         result = create_clean_data_driven_features(test_data, test_clients)
#         print(f"âœ… Ğ¢ĞµÑ�Ñ‚ Ğ¿Ñ€Ğ¾ÑˆĞµĞ»! Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¾ {result.shape[1]-1} Ñ„Ğ¸Ñ‡ĞµĞ¹ Ğ´Ğ»Ñ� {result.shape[0]} ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ¾Ğ²")
#         print("ĞšĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸:", result.columns.tolist())
#     except Exception as e:
#         print(f"â�Œ Ğ¢ĞµÑ�Ñ‚ Ğ¿Ñ€Ğ¾Ğ²Ğ°Ğ»ĞµĞ½: {e}")




# ĞœĞµÑ‚ĞºĞ¸
train_clients = set(train['client_num'])
transactions['date_time'] = pd.to_datetime(transactions['date_time'])
transactions['date'] = transactions['date_time'].dt.date
transactions['hour'] = transactions['date_time'].dt.hour
transactions['weekday'] = transactions['date_time'].dt.weekday
transactions['part_of_day'] = pd.cut(
    transactions['hour'], bins=[-1, 5, 11, 17, 23],
    labels=['night', 'morning', 'day', 'evening']
)

# === Ğ¡Ğ�Ğ�Ğ§Ğ�Ğ›Ğ� Ğ¡Ğ�Ğ—Ğ”Ğ�Ğ•Ğœ Ğ�Ğ�Ğ’Ğ«Ğ• Ğ¤Ğ˜Ğ§Ğ˜ ===
print("ğŸš€ Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ½Ğ¾Ğ²Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸...")
advanced_features = create_advanced_features(transactions, train_clients)
balance_features = create_balance_specific_features(transactions, train_clients)
data_driven_features = create_data_driven_features(transactions, train_clients)

# === ĞŸĞ�Ğ¢Ğ�Ğœ Ğ�Ğ Ğ˜Ğ“Ğ˜Ğ�Ğ�Ğ›Ğ¬Ğ�Ğ«Ğ• Ğ¤Ğ˜Ğ§Ğ˜ ===
print("ğŸ”§ Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ¾Ñ€Ğ¸Ğ³Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸...")
features = []
lags = [3, 7, 10, 14, 30, 60, 90, 180, 360]

# === 1. Ğ‘Ğ°Ğ·Ğ¾Ğ²Ñ‹Ğµ Ğ°Ğ³Ñ€ĞµĞ³Ğ°Ñ‚Ñ‹ ===
agg_main = transactions.groupby('client_num').agg(
    total_amount=('amount', 'sum'),
    mean_amount=('amount', 'mean'),
    median_amount=('amount', 'median'),
    std_amount=('amount', 'std'),
    max_amount=('amount', 'max'),
    min_amount=('amount', 'min'),
    skew_amount=('amount', pd.Series.skew),
    kurtosis_amount=('amount', pd.Series.kurtosis),
    transaction_count=('amount', 'count'),
    unique_mcc=('mcc_code', 'nunique'),
).reset_index()
for col in ['total_amount', 'mean_amount', 'median_amount', 'std_amount']:
    agg_main['log_' + col] = np.log1p(agg_main[col])
features.append(agg_main)

# === 2. Rolling Ñ„Ğ¸Ñ‡Ğ¸ Ñ� multiprocessing ===
daily = transactions.groupby(['client_num', 'date']).agg(
    daily_amount=('amount', 'sum'),
    daily_count=('amount', 'count')
).reset_index()
daily['date'] = pd.to_datetime(daily['date'])

def compute_rolling_features(lag):
    df = daily.copy()
    df = df.sort_values(['client_num', 'date'])
    df[f'rolling_mean_amt_{lag}'] = df.groupby('client_num')['daily_amount'].transform(lambda x: x.rolling(lag, min_periods=1).mean())
    df[f'rolling_median_amt_{lag}'] = df.groupby('client_num')['daily_amount'].transform(lambda x: x.rolling(lag, min_periods=1).median())
    df[f'rolling_std_amt_{lag}'] = df.groupby('client_num')['daily_amount'].transform(lambda x: x.rolling(lag, min_periods=1).std())
    df[f'rolling_mean_cnt_{lag}'] = df.groupby('client_num')['daily_count'].transform(lambda x: x.rolling(lag, min_periods=1).mean())
    df_last = df.groupby('client_num').tail(1).drop(columns=['date', 'daily_amount', 'daily_count'])
    return df_last

with Pool(cpu_count()) as pool:
    rolling_results = list(tqdm(pool.imap(compute_rolling_features, lags), total=len(lags), desc='Rolling lags'))

rolling_df = pd.concat(rolling_results).groupby('client_num').mean().reset_index()
features.append(rolling_df)

# === 3. Temporal features ===
partofday = transactions.groupby(['client_num', 'part_of_day'])['amount'].mean().unstack().add_prefix('avg_amt_').reset_index()
features.append(partofday)

weekday_stats = transactions.groupby(['client_num', 'weekday'])['amount'].mean().unstack().add_prefix('weekday_avg_').reset_index()
features.append(weekday_stats)

transactions = transactions.sort_values(['client_num', 'date_time'])
transactions['time_diff'] = transactions.groupby('client_num')['date_time'].diff().dt.total_seconds() / 3600
intervals = transactions.groupby('client_num')['time_diff'].agg(['mean', 'std', 'min', 'max']).add_prefix('interval_').reset_index()
features.append(intervals)

life = transactions.groupby('client_num').agg(
    first_txn=('date_time', 'min'),
    last_txn=('date_time', 'max')
).reset_index()
life['active_days'] = (life['last_txn'] - life['first_txn']).dt.days
features.append(life[['client_num', 'active_days']])

# === 4. MCC features ===
top_mcc_global = transactions['mcc_code'].value_counts().nlargest(10).index.tolist()
transactions['top_mcc'] = transactions['mcc_code'].apply(lambda x: x if x in top_mcc_global else 'other')
top_mcc_stats = transactions.groupby(['client_num', 'top_mcc']).size().unstack(fill_value=0).add_prefix('mcc_').reset_index()
features.append(top_mcc_stats)

def get_top_mccs(group):
    top = group['mcc_code'].value_counts().nlargest(3).index.tolist()
    return pd.Series(top + [np.nan] * (3 - len(top)), index=['mcc1', 'mcc2', 'mcc3'])

top_mcc_client = transactions.groupby('client_num').apply(get_top_mccs).reset_index()
features.append(top_mcc_client)

# === 5. RFM ===
now = transactions['date_time'].max()
rfm = transactions.groupby('client_num').agg(
    rfm_recency=('date_time', lambda x: (now - x.max()).days),
    rfm_frequency=('amount', 'count'),
    rfm_monetary=('amount', 'sum')
).reset_index()
features.append(rfm)

# === 6. Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½ĞµĞ½Ğ¸Ğµ Ğ�Ğ Ğ˜Ğ“Ğ˜Ğ�Ğ�Ğ›Ğ¬Ğ�Ğ«Ğ¥ Ñ„Ğ¸Ñ‡ĞµĞ¹ ===
original_features = features[0]
for f in features[1:]:
    original_features = original_features.merge(f, on='client_num', how='left')

print(f"Ğ�Ñ€Ğ¸Ğ³Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ñ‹Ñ… Ñ„Ğ¸Ñ‡ĞµĞ¹: {original_features.shape[1]-1}")

# === 7. Ğ�Ğ‘ĞªĞ•Ğ”Ğ˜Ğ�Ğ•Ğ�Ğ˜Ğ• Ğ¡ Ğ�Ğ�Ğ’Ğ«ĞœĞ˜ Ğ¤Ğ˜Ğ§Ğ�ĞœĞ˜ ===
print("ğŸ”— Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼ Ñ� Ğ½Ğ¾Ğ²Ñ‹Ğ¼Ğ¸ Ñ„Ğ¸Ñ‡Ğ°Ğ¼Ğ¸...")
enhanced_features = original_features.copy()
enhanced_features = enhanced_features.merge(advanced_features, on='client_num', how='left')
enhanced_features = enhanced_features.merge(balance_features, on='client_num', how='left')
enhanced_features = enhanced_features.merge(data_driven_features, on='client_num', how='left')
enhanced_features = enhanced_features.fillna(0)

print(f"Ğ˜Ñ‚Ğ¾Ğ³Ğ¾ Ñ„Ğ¸Ñ‡ĞµĞ¹: {enhanced_features.shape[1]-1}")

# === 8. Ğ Ğ°Ğ·Ğ´ĞµĞ»ĞµĞ½Ğ¸Ğµ train/test ===
train_full = train.merge(enhanced_features, on='client_num', how='left')
train_data = train_full.drop(columns=['client_num'])

test_clients = list(set(transactions['client_num']) - set(train['client_num']))
test_data = enhanced_features[enhanced_features['client_num'].isin(test_clients)].reset_index(drop=True)


print("ğŸ”§ Ğ˜Ñ�Ğ¿Ñ€Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ñ‚Ğ¸Ğ¿Ñ‹ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…...")

# Ğ˜Ñ�Ğ¿Ñ€Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸ Ñ� MCC ĞºĞ¾Ğ´Ğ°Ğ¼Ğ¸
mcc_columns = [col for col in train_data.columns if col.startswith('mcc') and train_data[col].dtype == 'object']
for col in mcc_columns:
    train_data[col] = pd.to_numeric(train_data[col], errors='coerce').fillna(0).astype('int64')

mcc_columns_test = [col for col in test_data.columns if col.startswith('mcc') and test_data[col].dtype == 'object']
for col in mcc_columns_test:
    test_data[col] = pd.to_numeric(test_data[col], errors='coerce').fillna(0).astype('int64')

# Ğ¢ĞµĞ¿ĞµÑ€ÑŒ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼
train_data.to_parquet('SET_1_train_data.parquet', index=False)
test_data.to_parquet('SET_1_test_data.parquet', index=False)

print("âœ… Ğ“Ğ¾Ñ‚Ğ¾Ğ²Ğ¾! Enhanced Ñ„Ğ¸Ñ‡Ğ¸ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ñ‹.")


train_data


train_data.columns.to_list()


pd.set_option('display.max_columns', None)


train_data.describe()


def analyze_target_correlations(train_data):
    """
    Ğ“Ğ»ÑƒĞ±Ğ¾ĞºĞ¸Ğ¹ Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ· ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¹ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² Ñ� Ñ‚Ğ°Ñ€Ğ³ĞµÑ‚Ğ¾Ğ¼
    """
    print("ğŸ�¯ Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ¸Ñ€ÑƒĞµĞ¼ ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¸ Ñ� Ñ‚Ğ°Ñ€Ğ³ĞµÑ‚Ğ¾Ğ¼...")
    
    # Ğ Ğ°Ğ·Ğ´ĞµĞ»Ñ�ĞµĞ¼ Ñ„Ğ¸Ñ‡Ğ¸ Ğ¸ Ñ‚Ğ°Ñ€Ğ³ĞµÑ‚
    target = train_data['target']
    features = train_data.drop('target', axis=1)
    
    print(f"ğŸ“Š Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ñ‚Ğ°Ñ€Ğ³ĞµÑ‚Ğ°:")
    target_dist = target.value_counts().sort_index()
    for cls, count in target_dist.items():
        pct = count / len(target) * 100
        print(f"   ĞšĞ»Ğ°Ñ�Ñ� {cls}: {count:,} ({pct:.1f}%)")
    
    # === 1. Ğ�Ğ‘Ğ©Ğ˜Ğ• ĞšĞ�Ğ Ğ Ğ•Ğ›Ğ¯Ğ¦Ğ˜Ğ˜ ===
    print("\nğŸ”� Ğ�Ğ±Ñ‰Ğ¸Ğµ ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¸ Ñ� Ñ‚Ğ°Ñ€Ğ³ĞµÑ‚Ğ¾Ğ¼ (Ñ‚Ğ¾Ğ¿-20):")
    
    # Ğ¢Ğ¾Ğ»ÑŒĞºĞ¾ Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ğµ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸
    numeric_features = features.select_dtypes(include=[np.number])
    correlations = numeric_features.corrwith(target).abs().sort_values(ascending=False)
    
    print(correlations.head(20))
    
    # === 2. Ğ�Ğ�Ğ�Ğ›Ğ˜Ğ— ĞŸĞ� ĞšĞ›Ğ�Ğ¡Ğ¡Ğ�Ğœ ===
    print("\nğŸ�¯ Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ· Ñ€Ğ°Ğ·Ğ»Ğ¸Ñ‡Ğ¸Ğ¹ Ğ¼ĞµĞ¶Ğ´Ñƒ ĞºĞ»Ğ°Ñ�Ñ�Ğ°Ğ¼Ğ¸...")
    
    # Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ±Ğ¸Ğ½Ğ°Ñ€Ğ½Ñ‹Ğµ Ğ¸Ğ½Ğ´Ğ¸ĞºĞ°Ñ‚Ğ¾Ñ€Ñ‹ Ğ´Ğ»Ñ� ĞºĞ°Ğ¶Ğ´Ğ¾Ğ³Ğ¾ ĞºĞ»Ğ°Ñ�Ñ�Ğ°
    class_analysis = {}
    
    for target_class in sorted(target.unique()):
        print(f"\n{'='*50}")
        print(f"ğŸ“Š ĞšĞ›Ğ�Ğ¡Ğ¡ {target_class} vs Ğ¾Ñ�Ñ‚Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ")
        print(f"{'='*50}")
        
        # Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ±Ğ¸Ğ½Ğ°Ñ€Ğ½Ñ‹Ğ¹ Ñ‚Ğ°Ñ€Ğ³ĞµÑ‚ (ĞºĞ»Ğ°Ñ�Ñ� vs Ğ½Ğµ ĞºĞ»Ğ°Ñ�Ñ�)
        binary_target = (target == target_class).astype(int)
        
        # ĞšĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¸ Ğ´Ğ»Ñ� Ñ�Ñ‚Ğ¾Ğ³Ğ¾ ĞºĞ»Ğ°Ñ�Ñ�Ğ°
        class_corr = numeric_features.corrwith(binary_target).abs().sort_values(ascending=False)
        
        print(f"Ğ¢Ğ¾Ğ¿-15 Ğ¿Ñ€ĞµĞ´Ğ¸ĞºÑ‚Ğ¾Ñ€Ğ¾Ğ² Ğ´Ğ»Ñ� ĞºĞ»Ğ°Ñ�Ñ�Ğ° {target_class}:")
        for i, (feature, corr) in enumerate(class_corr.head(15).items(), 1):
            direction = "â†‘" if numeric_features[feature].corr(binary_target) > 0 else "â†“"
            print(f"   {i:2d}. {feature:<30} {corr:.4f} {direction}")
        
        class_analysis[target_class] = class_corr.head(20).to_dict()
    
    # === 3. Ğ¡Ğ¢Ğ�Ğ¢Ğ˜Ğ¡Ğ¢Ğ˜Ğ§Ğ•Ğ¡ĞšĞ�Ğ¯ Ğ—Ğ�Ğ�Ğ§Ğ˜ĞœĞ�Ğ¡Ğ¢Ğ¬ (ANOVA) ===
    print(f"\n{'='*60}")
    print("ğŸ“ˆ Ğ¡Ğ¢Ğ�Ğ¢Ğ˜Ğ¡Ğ¢Ğ˜Ğ§Ğ•Ğ¡ĞšĞ�Ğ¯ Ğ—Ğ�Ğ�Ğ§Ğ˜ĞœĞ�Ğ¡Ğ¢Ğ¬ (ANOVA F-Ñ�Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞ°)")
    print(f"{'='*60}")
    
    f_scores = []
    for feature in numeric_features.columns:
        try:
            # Ğ“Ñ€ÑƒĞ¿Ğ¿Ğ¸Ñ€ÑƒĞµĞ¼ Ğ¿Ğ¾ ĞºĞ»Ğ°Ñ�Ñ�Ğ°Ğ¼ Ñ‚Ğ°Ñ€Ğ³ĞµÑ‚Ğ°
            groups = [numeric_features[feature][target == cls].dropna() for cls in sorted(target.unique())]
            
            # Ğ£Ğ´Ğ°Ğ»Ñ�ĞµĞ¼ Ğ¿ÑƒÑ�Ñ‚Ñ‹Ğµ Ğ³Ñ€ÑƒĞ¿Ğ¿Ñ‹
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) >= 2:
                f_stat, p_value = f_oneway(*groups)
                f_scores.append({
                    'feature': feature,
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.001
                })
        except:
            continue
    
    f_scores_df = pd.DataFrame(f_scores).sort_values('f_statistic', ascending=False)
    
    print("Ğ¢Ğ¾Ğ¿-20 Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² Ğ¿Ğ¾ F-Ñ�Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞµ (Ñ€Ğ°Ğ·Ğ»Ğ¸Ñ‡Ğ¸Ñ� Ğ¼ĞµĞ¶Ğ´Ñƒ ĞºĞ»Ğ°Ñ�Ñ�Ğ°Ğ¼Ğ¸):")
    for i, row in f_scores_df.head(20).iterrows():
        sig_mark = "***" if row['significant'] else ""
        print(f"   {row.name+1:2d}. {row['feature']:<30} F={row['f_statistic']:8.2f} p={row['p_value']:.2e} {sig_mark}")
    
    # === 4. Ğ¡Ğ Ğ•Ğ”Ğ�Ğ˜Ğ• Ğ—Ğ�Ğ�Ğ§Ğ•Ğ�Ğ˜Ğ¯ ĞŸĞ� ĞšĞ›Ğ�Ğ¡Ğ¡Ğ�Ğœ ===
    print(f"\n{'='*60}")
    print("ğŸ“Š Ğ¡Ğ Ğ•Ğ”Ğ�Ğ˜Ğ• Ğ—Ğ�Ğ�Ğ§Ğ•Ğ�Ğ˜Ğ¯ Ğ¢Ğ�ĞŸ-Ğ¤Ğ˜Ğ§Ğ•Ğ™ ĞŸĞ� ĞšĞ›Ğ�Ğ¡Ğ¡Ğ�Ğœ")
    print(f"{'='*60}")
    
    # Ğ‘ĞµÑ€ĞµĞ¼ Ñ‚Ğ¾Ğ¿-10 Ñ„Ğ¸Ñ‡ĞµĞ¹ Ğ¿Ğ¾ F-Ñ�Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞµ
    top_features = f_scores_df.head(10)['feature'].tolist()
    
    for feature in top_features:
        print(f"\nğŸ”� {feature}:")
        means_by_class = train_data.groupby('target')[feature].agg(['mean', 'std', 'count'])
        
        for cls in sorted(target.unique()):
            mean_val = means_by_class.loc[cls, 'mean']
            std_val = means_by_class.loc[cls, 'std']
            count_val = means_by_class.loc[cls, 'count']
            print(f"   ĞšĞ»Ğ°Ñ�Ñ� {cls}: {mean_val:10.2f} Â± {std_val:8.2f} (n={count_val:,})")
    
    # === 5. Ğ¡ĞŸĞ•Ğ¦Ğ˜Ğ�Ğ›Ğ¬Ğ�Ğ«Ğ™ Ğ�Ğ�Ğ�Ğ›Ğ˜Ğ— Ğ”Ğ›Ğ¯ ĞšĞ›Ğ�Ğ¡Ğ¡Ğ� 0 ===
    print(f"\n{'='*60}")
    print("ğŸš¨ Ğ¡ĞŸĞ•Ğ¦Ğ˜Ğ�Ğ›Ğ¬Ğ�Ğ«Ğ™ Ğ�Ğ�Ğ�Ğ›Ğ˜Ğ— Ğ”Ğ›Ğ¯ ĞšĞ›Ğ�Ğ¡Ğ¡Ğ� 0 (Ñ�Ğ°Ğ¼Ñ‹Ğ¹ Ğ²Ğ°Ğ¶Ğ½Ñ‹Ğ¹ Ğ´Ğ»Ñ� WMAE)")
    print(f"{'='*60}")
    
    class_0_indicator = (target == 0).astype(int)
    
    # Ğ¢Ğ¾Ğ¿ Ğ¿Ñ€ĞµĞ´Ğ¸ĞºÑ‚Ğ¾Ñ€Ñ‹ Ğ´Ğ»Ñ� ĞºĞ»Ğ°Ñ�Ñ�Ğ° 0
    class_0_corr = numeric_features.corrwith(class_0_indicator).sort_values(key=abs, ascending=False)
    
    print("Ğ¢Ğ¾Ğ¿-15 Ğ¿Ñ€ĞµĞ´Ğ¸ĞºÑ‚Ğ¾Ñ€Ğ¾Ğ² Ğ¾Ğ±Ğ½ÑƒĞ»ĞµĞ½Ğ¸Ñ� Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�Ğ° (ĞºĞ»Ğ°Ñ�Ñ� 0):")
    for i, (feature, corr) in enumerate(class_0_corr.head(15).items(), 1):
        direction = "Ñ�Ğ¿Ğ¾Ñ�Ğ¾Ğ±Ñ�Ñ‚Ğ²ÑƒĞµÑ‚ Ğ¾Ğ±Ğ½ÑƒĞ»ĞµĞ½Ğ¸Ñ�" if corr > 0 else "Ğ·Ğ°Ñ‰Ğ¸Ñ‰Ğ°ĞµÑ‚ Ğ¾Ñ‚ Ğ¾Ğ±Ğ½ÑƒĞ»ĞµĞ½Ğ¸Ñ�"
        print(f"   {i:2d}. {feature:<30} {corr:7.4f} ({direction})")
    
    # Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ· Ğ·Ğ°Ñ‰Ğ¸Ñ‚Ğ½Ñ‹Ñ… Ñ„Ğ°ĞºÑ‚Ğ¾Ñ€Ğ¾Ğ² (Ğ¾Ñ‚Ñ€Ğ¸Ñ†Ğ°Ñ‚ĞµĞ»ÑŒĞ½Ğ°Ñ� ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ñ� Ñ� ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ¼ 0)
    protective_factors = class_0_corr[class_0_corr < -0.05].sort_values()
    print(f"\nğŸ›¡ï¸�  Ğ¢Ğ¾Ğ¿-10 Ğ·Ğ°Ñ‰Ğ¸Ñ‚Ğ½Ñ‹Ñ… Ñ„Ğ°ĞºÑ‚Ğ¾Ñ€Ğ¾Ğ² Ğ¾Ñ‚ Ğ¾Ğ±Ğ½ÑƒĞ»ĞµĞ½Ğ¸Ñ� Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�Ğ°:")
    for i, (feature, corr) in enumerate(protective_factors.head(10).items(), 1):
        print(f"   {i:2d}. {feature:<30} {corr:7.4f}")
    
    # Ğ¤Ğ°ĞºÑ‚Ğ¾Ñ€Ñ‹ Ñ€Ğ¸Ñ�ĞºĞ° (Ğ¿Ğ¾Ğ»Ğ¾Ğ¶Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ğ°Ñ� ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ñ� Ñ� ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ¼ 0)
    risk_factors = class_0_corr[class_0_corr > 0.05].sort_values(ascending=False)
    print(f"\nâš ï¸�  Ğ¢Ğ¾Ğ¿-10 Ñ„Ğ°ĞºÑ‚Ğ¾Ñ€Ğ¾Ğ² Ñ€Ğ¸Ñ�ĞºĞ° Ğ¾Ğ±Ğ½ÑƒĞ»ĞµĞ½Ğ¸Ñ� Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�Ğ°:")
    for i, (feature, corr) in enumerate(risk_factors.head(10).items(), 1):
        print(f"   {i:2d}. {feature:<30} {corr:7.4f}")
    
    # === 6. ĞšĞ�ĞœĞ‘Ğ˜Ğ�Ğ˜Ğ Ğ�Ğ’Ğ�Ğ�Ğ�Ğ«Ğ™ Ğ�Ğ�Ğ�Ğ›Ğ˜Ğ— ===
    print(f"\n{'='*60}")
    print("ğŸ�¯ ĞšĞ�ĞœĞ‘Ğ˜Ğ�Ğ˜Ğ Ğ�Ğ’Ğ�Ğ�Ğ�Ğ«Ğ™ Ğ¡ĞšĞ�Ğ  (ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ñ� + F-Ñ�Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞ°)")
    print(f"{'='*60}")
    
    # Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ ĞºĞ¾Ğ¼Ğ±Ğ¸Ğ½Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½Ñ‹Ğ¹ Ñ�ĞºĞ¾Ñ€
    combined_scores = []
    
    for feature in numeric_features.columns:
        # ĞšĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ñ� Ñ� ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ¼ 0
        class_0_corr_val = abs(class_0_corr.get(feature, 0))
        
        # F-Ñ�Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞ°
        f_row = f_scores_df[f_scores_df['feature'] == feature]
        f_stat_norm = f_row['f_statistic'].iloc[0] / f_scores_df['f_statistic'].max() if len(f_row) > 0 else 0
        
        # ĞšĞ¾Ğ¼Ğ±Ğ¸Ğ½Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½Ñ‹Ğ¹ Ñ�ĞºĞ¾Ñ€
        combined_score = class_0_corr_val * 0.6 + f_stat_norm * 0.4
        
        combined_scores.append({
            'feature': feature,
            'class_0_corr': class_0_corr_val,
            'f_stat_norm': f_stat_norm,
            'combined_score': combined_score
        })
    
    combined_df = pd.DataFrame(combined_scores).sort_values('combined_score', ascending=False)
    
    print("Ğ¢Ğ¾Ğ¿-20 Ñ�Ğ°Ğ¼Ñ‹Ñ… Ğ²Ğ°Ğ¶Ğ½Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² (ĞºĞ¾Ğ¼Ğ±Ğ¸Ğ½Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½Ñ‹Ğ¹ Ñ�ĞºĞ¾Ñ€):")
    for i, row in combined_df.head(20).iterrows():
        print(f"   {i+1:2d}. {row['feature']:<30} Score={row['combined_score']:.4f} "
              f"(Corr={row['class_0_corr']:.3f}, F-norm={row['f_stat_norm']:.3f})")
    
    # === 7. ĞŸĞ Ğ�ĞšĞ¢Ğ˜Ğ§Ğ•Ğ¡ĞšĞ˜Ğ• Ğ Ğ•ĞšĞ�ĞœĞ•Ğ�Ğ”Ğ�Ğ¦Ğ˜Ğ˜ ===
    print(f"\n{'='*60}")
    print("ğŸ’¡ ĞŸĞ Ğ�ĞšĞ¢Ğ˜Ğ§Ğ•Ğ¡ĞšĞ˜Ğ• Ğ Ğ•ĞšĞ�ĞœĞ•Ğ�Ğ”Ğ�Ğ¦Ğ˜Ğ˜")
    print(f"{'='*60}")
    
    print("ğŸ�¯ Ğ”Ğ»Ñ� ÑƒĞ»ÑƒÑ‡ÑˆĞµĞ½Ğ¸Ñ� Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� ĞºĞ»Ğ°Ñ�Ñ�Ğ° 0 (Ğ¾Ğ±Ğ½ÑƒĞ»ĞµĞ½Ğ¸Ğµ Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�Ğ°):")
    
    top_5_features = combined_df.head(5)['feature'].tolist()
    print(f"\n   Ğ¤Ğ¾ĞºÑƒÑ�Ğ¸Ñ€ÑƒĞ¹Ñ‚ĞµÑ�ÑŒ Ğ½Ğ° Ñ‚Ğ¾Ğ¿-5 Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ°Ñ…:")
    for i, feature in enumerate(top_5_features, 1):
        corr_val = class_0_corr.get(feature, 0)
        trend = "â†‘ ÑƒĞ²ĞµĞ»Ğ¸Ñ‡Ğ¸Ğ²Ğ°ĞµÑ‚ Ñ€Ğ¸Ñ�Ğº" if corr_val > 0 else "â†“ Ñ�Ğ½Ğ¸Ğ¶Ğ°ĞµÑ‚ Ñ€Ğ¸Ñ�Ğº"
        print(f"   {i}. {feature} ({trend})")
    
    # Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ· Ğ½Ğ¾Ğ²Ñ‹Ñ… vs Ñ�Ñ‚Ğ°Ñ€Ñ‹Ñ… Ñ„Ğ¸Ñ‡ĞµĞ¹
    new_feature_keywords = ['mcc5411', 'mcc5499', 'mcc6011', 'momentum', 'burn_rate', 'stability', 
                           'concentration', 'frequent_rare', 'high_value', 'consistency']
    
    new_features_in_top = [f for f in top_5_features if any(kw in f for kw in new_feature_keywords)]
    
    if new_features_in_top:
        print(f"\nğŸš€ Ğ�Ğ¾Ğ²Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸ Ğ² Ñ‚Ğ¾Ğ¿-5: {len(new_features_in_top)}")
        for feat in new_features_in_top:
            print(f"   âœ… {feat}")
    else:
        print(f"\nâš ï¸�  Ğ�Ğ¾Ğ²Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸ Ğ½Ğµ Ğ¿Ğ¾Ğ¿Ğ°Ğ»Ğ¸ Ğ² Ñ‚Ğ¾Ğ¿-5, Ğ²Ğ¾Ğ·Ğ¼Ğ¾Ğ¶Ğ½Ğ¾ Ğ½ÑƒĞ¶Ğ½Ğ° Ğ´Ğ¾Ñ€Ğ°Ğ±Ğ¾Ñ‚ĞºĞ°")
    
    return {
        'correlations': correlations,
        'class_analysis': class_analysis,
        'f_scores': f_scores_df,
        'class_0_analysis': class_0_corr,
        'combined_scores': combined_df,
        'top_features': top_5_features
    }

# === Ğ’Ğ˜Ğ—Ğ£Ğ�Ğ›Ğ˜Ğ—Ğ�Ğ¦Ğ˜Ğ¯ ===
def plot_correlation_analysis(train_data, analysis_results):
    """
    Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµÑ‚ Ğ²Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ğ¸ ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¾Ğ½Ğ½Ğ¾Ğ³Ğ¾ Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ°
    """
    print("ğŸ“Š Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ²Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ğ¸...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Ğ¢Ğ¾Ğ¿ ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¸ Ñ� Ñ‚Ğ°Ñ€Ğ³ĞµÑ‚Ğ¾Ğ¼
    top_corr = analysis_results['correlations'].head(15)
    axes[0,0].barh(range(len(top_corr)), top_corr.values)
    axes[0,0].set_yticks(range(len(top_corr)))
    axes[0,0].set_yticklabels(top_corr.index, fontsize=10)
    axes[0,0].set_title('Ğ¢Ğ¾Ğ¿-15 ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¹ Ñ� Ñ‚Ğ°Ñ€Ğ³ĞµÑ‚Ğ¾Ğ¼', fontsize=14)
    axes[0,0].set_xlabel('Ğ�Ğ±Ñ�Ğ¾Ğ»Ñ�Ñ‚Ğ½Ğ°Ñ� ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ñ�')
    
    # 2. F-Ñ�Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞ¸
    top_f = analysis_results['f_scores'].head(15)
    axes[0,1].barh(range(len(top_f)), top_f['f_statistic'])
    axes[0,1].set_yticks(range(len(top_f)))
    axes[0,1].set_yticklabels(top_f['feature'], fontsize=10)
    axes[0,1].set_title('Ğ¢Ğ¾Ğ¿-15 F-Ñ�Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸Ğº (Ñ€Ğ°Ğ·Ğ»Ğ¸Ñ‡Ğ¸Ñ� Ğ¼ĞµĞ¶Ğ´Ñƒ ĞºĞ»Ğ°Ñ�Ñ�Ğ°Ğ¼Ğ¸)', fontsize=14)
    axes[0,1].set_xlabel('F-Ñ�Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞ°')
    
    # 3. ĞšĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¸ Ñ� ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ¼ 0
    class_0_corr = analysis_results['class_0_analysis'].head(15)
    colors = ['red' if x > 0 else 'green' for x in class_0_corr.values]
    axes[1,0].barh(range(len(class_0_corr)), class_0_corr.values, color=colors, alpha=0.7)
    axes[1,0].set_yticks(range(len(class_0_corr)))
    axes[1,0].set_yticklabels(class_0_corr.index, fontsize=10)
    axes[1,0].set_title('Ğ¢Ğ¾Ğ¿-15 ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¹ Ñ� ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ¼ 0\n(ĞºÑ€Ğ°Ñ�Ğ½Ñ‹Ğ¹=Ñ€Ğ¸Ñ�Ğº, Ğ·ĞµĞ»ĞµĞ½Ñ‹Ğ¹=Ğ·Ğ°Ñ‰Ğ¸Ñ‚Ğ°)', fontsize=14)
    axes[1,0].set_xlabel('ĞšĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ñ� Ñ� ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ¼ 0')
    
    # 4. ĞšĞ¾Ğ¼Ğ±Ğ¸Ğ½Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½Ñ‹Ğµ Ñ�ĞºĞ¾Ñ€Ñ‹
    top_combined = analysis_results['combined_scores'].head(15)
    axes[1,1].barh(range(len(top_combined)), top_combined['combined_score'])
    axes[1,1].set_yticks(range(len(top_combined)))
    axes[1,1].set_yticklabels(top_combined['feature'], fontsize=10)
    axes[1,1].set_title('Ğ¢Ğ¾Ğ¿-15 ĞºĞ¾Ğ¼Ğ±Ğ¸Ğ½Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½Ñ‹Ñ… Ñ�ĞºĞ¾Ñ€Ğ¾Ğ²', fontsize=14)
    axes[1,1].set_xlabel('ĞšĞ¾Ğ¼Ğ±Ğ¸Ğ½Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½Ñ‹Ğ¹ Ñ�ĞºĞ¾Ñ€')
    
    plt.tight_layout()
    plt.show()
    
    # Heatmap Ñ�Ñ€ĞµĞ´Ğ½Ğ¸Ñ… Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğ¹ Ğ¿Ğ¾ ĞºĞ»Ğ°Ñ�Ñ�Ğ°Ğ¼ Ğ´Ğ»Ñ� Ñ‚Ğ¾Ğ¿ Ñ„Ğ¸Ñ‡ĞµĞ¹
    top_features = analysis_results['top_features']
    class_means = train_data.groupby('target')[top_features].mean()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(class_means.T, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0)
    plt.title('Ğ¡Ñ€ĞµĞ´Ğ½Ğ¸Ğµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ� Ñ‚Ğ¾Ğ¿-5 Ñ„Ğ¸Ñ‡ĞµĞ¹ Ğ¿Ğ¾ ĞºĞ»Ğ°Ñ�Ñ�Ğ°Ğ¼ Ñ‚Ğ°Ñ€Ğ³ĞµÑ‚Ğ°')
    plt.xlabel('ĞšĞ»Ğ°Ñ�Ñ� Ñ‚Ğ°Ñ€Ğ³ĞµÑ‚Ğ°')
    plt.ylabel('ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸')
    plt.tight_layout()
    plt.show()

# Ğ˜Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ:
analysis_results = analyze_target_correlations(train_data)
plot_correlation_analysis(train_data, analysis_results)


def create_activity_recency_features(transactions, train_clients):
    """
    Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµÑ‚ Ñ„Ğ¸Ñ‡Ğ¸ Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ¸ recency Ğ½Ğ° Ğ¾Ñ�Ğ½Ğ¾Ğ²Ğµ ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¾Ğ½Ğ½Ğ¾Ğ³Ğ¾ Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ°
    Ğ¤Ğ¾ĞºÑƒÑ� Ğ½Ğ° Ñ�Ğ°Ğ¼Ñ‹Ñ… Ñ�Ğ¸Ğ»ÑŒĞ½Ñ‹Ñ… Ğ¿Ñ€ĞµĞ´Ğ¸ĞºÑ‚Ğ¾Ñ€Ğ°Ñ… ĞºĞ»Ğ°Ñ�Ñ�Ğ° 0
    """
    print("ğŸ�¯ Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ ACTIVITY/RECENCY Ñ„Ğ¸Ñ‡Ğ¸ Ğ½Ğ° Ğ¾Ñ�Ğ½Ğ¾Ğ²Ğµ Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ° ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¹...")
    
    # ĞŸĞ¾Ğ´Ğ³Ğ¾Ñ‚Ğ°Ğ²Ğ»Ğ¸Ğ²Ğ°ĞµĞ¼ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ğµ Ğ¿Ğ¾Ğ»Ñ�
    if 'date' not in transactions.columns:
        transactions['date'] = pd.to_datetime(transactions['date_time']).dt.date
    
    # Ğ‘Ğ°Ğ·Ğ¾Ğ²Ñ‹Ğµ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ğµ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸
    now = transactions['date_time'].max()
    max_date = transactions['date'].max()
    min_date = transactions['date'].min()
    
    print(f"   ĞŸĞµÑ€Ğ¸Ğ¾Ğ´ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…: {min_date} - {max_date}")
    
    features_list = []
    all_clients = pd.DataFrame({'client_num': transactions['client_num'].unique()})
    
    # === 1. ENHANCED ACTIVITY FEATURES ===
    print("   ğŸ�ƒ Enhanced activity Ñ„Ğ¸Ñ‡Ğ¸...")
    
    # ĞŸĞ»Ğ¾Ñ‚Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚Ğ¸ (Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸/Ğ´ĞµĞ½ÑŒ)
    def calculate_activity_density(group):
        total_days = (group['date'].max() - group['date'].min()).days + 1
        return len(group) / total_days if total_days > 0 else 0
    
    activity_density = transactions.groupby('client_num').apply(calculate_activity_density).reset_index()
    activity_density.columns = ['client_num', 'activity_density']
    
    # Ğ¡Ñ‚Ğ°Ğ±Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ¿Ğ¾ Ğ½ĞµĞ´ĞµĞ»Ñ�Ğ¼
    transactions['week'] = pd.to_datetime(transactions['date']).dt.isocalendar().week
    weekly_activity = transactions.groupby(['client_num', 'week']).size().reset_index()
    weekly_activity.columns = ['client_num', 'week', 'weekly_txns']
    
    weekly_stats = weekly_activity.groupby('client_num')['weekly_txns'].agg(['mean', 'std', 'count']).reset_index()
    weekly_stats.columns = ['client_num', 'avg_weekly_txns', 'std_weekly_txns', 'active_weeks']
    weekly_stats['weekly_consistency'] = 1 / (weekly_stats['std_weekly_txns'] / (weekly_stats['avg_weekly_txns'] + 1) + 1)
    weekly_stats = weekly_stats.fillna(0)
    
    # Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼ activity Ñ„Ğ¸Ñ‡Ğ¸
    activity_features = activity_density.merge(weekly_stats, on='client_num', how='outer').fillna(0)
    features_list.append(activity_features)
    
    # === 2. ADVANCED RECENCY FEATURES ===
    print("   â�° Advanced recency Ñ„Ğ¸Ñ‡Ğ¸...")
    
    # Ğ”ĞµÑ‚Ğ°Ğ»Ğ¸Ğ·Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½Ñ‹Ğ¹ recency Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ·
    def calculate_advanced_recency(group):
        sorted_dates = sorted(group['date_time'])
        
        if len(sorted_dates) == 0:
            return pd.Series({
                'days_since_last_txn': 999,
                'days_since_penultimate': 999,
                'recency_acceleration': 0,
                'recency_trend': 0
            })
        
        # Ğ”Ğ½Ğ¸ Ñ� Ğ¿Ğ¾Ñ�Ğ»ĞµĞ´Ğ½ĞµĞ¹ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸
        days_since_last = (now - sorted_dates[-1]).days
        
        # Ğ”Ğ½Ğ¸ Ñ� Ğ¿Ñ€ĞµĞ´Ğ¿Ğ¾Ñ�Ğ»ĞµĞ´Ğ½ĞµĞ¹ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸
        days_since_penultimate = (now - sorted_dates[-2]).days if len(sorted_dates) >= 2 else days_since_last
        
        # Ğ£Ñ�ĞºĞ¾Ñ€ĞµĞ½Ğ¸Ğµ Ğ¿Ğ°Ñ�Ñ�Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚Ğ¸ (Ñ€Ğ°Ñ�Ñ‚ĞµÑ‚ Ğ»Ğ¸ gap Ğ¼ĞµĞ¶Ğ´Ñƒ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ñ�Ğ¼Ğ¸?)
        if len(sorted_dates) >= 3:
            recent_gap = (sorted_dates[-1] - sorted_dates[-2]).days
            prev_gap = (sorted_dates[-2] - sorted_dates[-3]).days
            recency_acceleration = recent_gap / (prev_gap + 1)
        else:
            recency_acceleration = 1
        
        # Ğ¢Ñ€ĞµĞ½Ğ´ recency (Ñ�Ñ‚Ğ°Ğ½Ğ¾Ğ²Ğ¸Ñ‚Ñ�Ñ� Ğ»Ğ¸ Ñ…ÑƒĞ¶Ğµ Ñ�Ğ¾ Ğ²Ñ€ĞµĞ¼ĞµĞ½ĞµĞ¼?)
        if len(sorted_dates) >= 4:
            gaps = [(sorted_dates[i] - sorted_dates[i-1]).days for i in range(1, len(sorted_dates))]
            recency_trend = np.polyfit(range(len(gaps)), gaps, 1)[0] if len(gaps) > 1 else 0
        else:
            recency_trend = 0
        
        return pd.Series({
            'days_since_last_txn': days_since_last,
            'days_since_penultimate': days_since_penultimate,
            'recency_acceleration': recency_acceleration,
            'recency_trend': recency_trend
        })
    
    recency_features = transactions.groupby('client_num').apply(calculate_advanced_recency).reset_index()
    features_list.append(recency_features)
    
    # === 3. RECENT ACTIVITY ANALYSIS ===
    print("   ğŸ“ˆ Recent activity Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ·...")
    
    # Ğ�ĞºÑ‚Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ² Ğ¿Ğ¾Ñ�Ğ»ĞµĞ´Ğ½Ğ¸Ğµ Ğ¿ĞµÑ€Ğ¸Ğ¾Ğ´Ñ‹
    periods = [7, 14, 30]  # Ğ´Ğ½Ğ¸
    
    recent_activity_features = []
    
    for period in periods:
        cutoff_date = max_date - timedelta(days=period)
        recent_data = transactions[transactions['date'] >= cutoff_date]
        
        if len(recent_data) > 0:
            recent_stats = recent_data.groupby('client_num').agg({
                'amount': ['sum', 'count', 'mean'],
                'mcc_code': 'nunique',
                'date': 'nunique'
            })
            
            recent_stats.columns = [f'recent_{period}d_{col[1]}_{col[0]}' if col[1] else f'recent_{period}d_{col[0]}' 
                                   for col in recent_stats.columns]
            recent_stats = recent_stats.reset_index()
            
            # Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ğ¸Ğ½Ñ‚ĞµĞ½Ñ�Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚Ğ¸
            recent_stats[f'recent_{period}d_intensity'] = recent_stats[f'recent_{period}d_count_amount'] / period
            
        else:
            # Ğ•Ñ�Ğ»Ğ¸ Ğ½ĞµÑ‚ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… Ğ·Ğ° Ğ¿ĞµÑ€Ğ¸Ğ¾Ğ´, Ñ�Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ¿ÑƒÑ�Ñ‚Ğ¾Ğ¹ DataFrame
            recent_stats = pd.DataFrame({'client_num': []})
        
        recent_activity_features.append(recent_stats)
    
    # Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼ recent activity
    recent_combined = all_clients.copy()
    for recent_df in recent_activity_features:
        if len(recent_df) > 0:
            recent_combined = recent_combined.merge(recent_df, on='client_num', how='left')
    
    recent_combined = recent_combined.fillna(0)
    features_list.append(recent_combined)
    
    # === 4. MOMENTUM INDICATORS ===
    print("   ğŸš€ Momentum Ğ¸Ğ½Ğ´Ğ¸ĞºĞ°Ñ‚Ğ¾Ñ€Ñ‹...")
    
    # Ğ¡Ñ€Ğ°Ğ²Ğ½ĞµĞ½Ğ¸Ğµ Ğ¿ĞµÑ€Ğ²Ğ¾Ğ¹ Ğ¸ Ğ²Ñ‚Ğ¾Ñ€Ğ¾Ğ¹ Ğ¿Ğ¾Ğ»Ğ¾Ğ²Ğ¸Ğ½Ñ‹ Ğ¿ĞµÑ€Ğ¸Ğ¾Ğ´Ğ°
    total_days = (max_date - min_date).days
    mid_date = min_date + timedelta(days=total_days//2)
    
    first_half = transactions[transactions['date'] <= mid_date]
    second_half = transactions[transactions['date'] > mid_date]
    
    # Ğ¡Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞ¸ Ğ¿Ğ¾ Ğ¿Ğ¾Ğ»Ğ¾Ğ²Ğ¸Ğ½Ğ°Ğ¼
    first_half_stats = first_half.groupby('client_num').agg({
        'amount': ['sum', 'count'],
        'date': 'nunique'
    })
    first_half_stats.columns = ['first_half_amount', 'first_half_count', 'first_half_days']
    first_half_stats = first_half_stats.reset_index()
    
    second_half_stats = second_half.groupby('client_num').agg({
        'amount': ['sum', 'count'],
        'date': 'nunique'
    })
    second_half_stats.columns = ['second_half_amount', 'second_half_count', 'second_half_days']
    second_half_stats = second_half_stats.reset_index()
    
    # Momentum Ñ„Ğ¸Ñ‡Ğ¸
    momentum_features = all_clients.merge(first_half_stats, on='client_num', how='left').fillna(0)
    momentum_features = momentum_features.merge(second_half_stats, on='client_num', how='left').fillna(0)
    
    # Ğ Ğ°Ñ�Ñ‡ĞµÑ‚ momentum
    momentum_features['activity_momentum'] = (
        momentum_features['second_half_count'] - momentum_features['first_half_count']
    ) / (momentum_features['first_half_count'] + 1)
    
    momentum_features['spending_momentum'] = (
        momentum_features['second_half_amount'] - momentum_features['first_half_amount']
    ) / (momentum_features['first_half_amount'] + 1)
    
    momentum_features['days_momentum'] = (
        momentum_features['second_half_days'] - momentum_features['first_half_days']
    ) / (momentum_features['first_half_days'] + 1)
    
    # Ğ�Ğ±Ñ‰Ğ¸Ğ¹ momentum score
    momentum_features['overall_momentum'] = (
        momentum_features['activity_momentum'] + 
        momentum_features['spending_momentum'] + 
        momentum_features['days_momentum']
    ) / 3
    
    features_list.append(momentum_features)
    
    # === 5. BANKRUPTCY RISK SCORE ===
    print("   ğŸš¨ Bankruptcy risk score...")
    
    # Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ¿Ñ€Ğ¾Ğ¼ĞµĞ¶ÑƒÑ‚Ğ¾Ñ‡Ğ½Ñ‹Ğ¹ Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚ Ğ´Ğ»Ñ� risk score
    temp_features = all_clients.copy()
    for feat_df in features_list:
        temp_features = temp_features.merge(feat_df, on='client_num', how='left')
    temp_features = temp_features.fillna(0)
    
    # ĞšĞ¾Ğ¼Ğ¿Ğ¾Ğ·Ğ¸Ñ‚Ğ½Ñ‹Ğ¹ Ñ€Ğ¸Ñ�Ğº-Ñ�ĞºĞ¾Ñ€ Ğ½Ğ° Ğ¾Ñ�Ğ½Ğ¾Ğ²Ğµ Ğ½Ğ°Ğ¹Ğ´ĞµĞ½Ğ½Ñ‹Ñ… ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¹
    risk_components = []
    
    # ĞšĞ¾Ğ¼Ğ¿Ğ¾Ğ½ĞµĞ½Ñ‚ 1: Recency risk (Ñ‡ĞµĞ¼ Ğ±Ğ¾Ğ»ÑŒÑˆĞµ Ğ´Ğ½ĞµĞ¹, Ñ‚ĞµĞ¼ Ñ…ÑƒĞ¶Ğµ)
    if 'days_since_last_txn' in temp_features.columns:
        recency_risk = temp_features['days_since_last_txn'] / temp_features['days_since_last_txn'].max()
        risk_components.append(recency_risk * 0.3)
    
    # ĞšĞ¾Ğ¼Ğ¿Ğ¾Ğ½ĞµĞ½Ñ‚ 2: Low activity risk
    if 'activity_density' in temp_features.columns:
        activity_risk = 1 / (temp_features['activity_density'] + 0.01)  # Ğ˜Ğ½Ğ²ĞµÑ€Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼
        activity_risk = activity_risk / activity_risk.max()  # Ğ�Ğ¾Ñ€Ğ¼Ğ°Ğ»Ğ¸Ğ·ÑƒĞµĞ¼
        risk_components.append(activity_risk * 0.25)
    
    # ĞšĞ¾Ğ¼Ğ¿Ğ¾Ğ½ĞµĞ½Ñ‚ 3: Negative momentum
    if 'overall_momentum' in temp_features.columns:
        momentum_risk = np.clip(-temp_features['overall_momentum'], 0, None)  # Ğ¢Ğ¾Ğ»ÑŒĞºĞ¾ Ğ½ĞµĞ³Ğ°Ñ‚Ğ¸Ğ²Ğ½Ñ‹Ğ¹ momentum
        if momentum_risk.max() > 0:
            momentum_risk = momentum_risk / momentum_risk.max()
        risk_components.append(momentum_risk * 0.25)
    
    # ĞšĞ¾Ğ¼Ğ¿Ğ¾Ğ½ĞµĞ½Ñ‚ 4: Inconsistency
    if 'weekly_consistency' in temp_features.columns:
        consistency_risk = 1 - temp_features['weekly_consistency']  # Ğ˜Ğ½Ğ²ĞµÑ€Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼
        risk_components.append(consistency_risk * 0.2)
    
    # Ğ˜Ñ‚Ğ¾Ğ³Ğ¾Ğ²Ñ‹Ğ¹ risk score
    if risk_components:
        bankruptcy_risk_score = sum(risk_components)
        bankruptcy_risk_df = pd.DataFrame({
            'client_num': temp_features['client_num'],
            'bankruptcy_risk_score': bankruptcy_risk_score,
            'risk_category': pd.cut(bankruptcy_risk_score, bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        })
        
        features_list.append(bankruptcy_risk_df)
    
    # === Ğ�Ğ‘ĞªĞ•Ğ”Ğ˜Ğ�Ğ•Ğ�Ğ˜Ğ• Ğ’Ğ¡Ğ•Ğ¥ Ğ¤Ğ˜Ğ§Ğ•Ğ™ ===
    print("   ğŸ”— Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼ activity/recency Ñ„Ğ¸Ñ‡Ğ¸...")
    
    final_features = all_clients.copy()
    
    for i, feature_df in enumerate(features_list):
        print(f"      Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ğ½Ğ°Ğ±Ğ¾Ñ€ {i+1}/{len(features_list)}: {feature_df.shape}")
        final_features = final_features.merge(feature_df, on='client_num', how='left')
    
    # Ğ¤Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ğ°Ñ� Ğ¾Ğ±Ñ€Ğ°Ğ±Ğ¾Ñ‚ĞºĞ° - Ğ˜Ğ¡ĞŸĞ Ğ�Ğ’Ğ›Ğ•Ğ�Ğ�Ğ�Ğ¯ Ğ’Ğ•Ğ Ğ¡Ğ˜Ğ¯
    # Ğ—Ğ°Ğ¿Ğ¾Ğ»Ğ½Ñ�ĞµĞ¼ Ñ‚Ğ¾Ğ»ÑŒĞºĞ¾ Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ğµ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸
    numeric_cols = final_features.select_dtypes(include=[np.number]).columns
    final_features[numeric_cols] = final_features[numeric_cols].fillna(0)
    
    # Ğ£Ğ±Ğ¸Ñ€Ğ°ĞµĞ¼ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸ Ğ´Ğ»Ñ� ĞºĞ¾Ñ€Ñ€ĞµĞºÑ‚Ğ½Ğ¾Ğ³Ğ¾ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ñ�
    categorical_cols = final_features.select_dtypes(include=['category', 'object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'client_num']
    
    if len(categorical_cols) > 0:
        print(f"      Ğ£Ğ´Ğ°Ğ»Ñ�ĞµĞ¼ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸: {categorical_cols}")
        final_features = final_features.drop(columns=categorical_cols)
    
    print(f"âœ… Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¾ {final_features.shape[1]-1} activity/recency Ñ„Ğ¸Ñ‡ĞµĞ¹ Ğ´Ğ»Ñ� {final_features.shape[0]} ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ¾Ğ²")
    
    # ĞŸĞ¾ĞºĞ°Ğ·Ñ‹Ğ²Ğ°ĞµĞ¼ Ñ�Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸
    new_features = [col for col in final_features.columns if col != 'client_num']
    print(f"\nğŸ“‹ Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸:")
    for i, feat in enumerate(new_features, 1):
        print(f"   {i:2d}. {feat}")
    
    return final_features

# === Ğ‘Ğ«Ğ¡Ğ¢Ğ Ğ�Ğ• Ğ¢Ğ•Ğ¡Ğ¢Ğ˜Ğ Ğ�Ğ’Ğ�Ğ�Ğ˜Ğ• ===
if __name__ == "__main__":
    print("ğŸ§ª Ğ¢ĞµÑ�Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ activity/recency Ñ„ÑƒĞ½ĞºÑ†Ğ¸Ñ�...")
    
    # Ğ¢ĞµÑ�Ñ‚Ğ¾Ğ²Ñ‹Ğµ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ
    test_data = pd.DataFrame({
        'client_num': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        'date_time': pd.to_datetime([
            '2024-01-01 10:00', '2024-01-15 15:00', '2024-01-30 12:00',  # Client 1
            '2024-01-01 11:00', '2024-01-02 16:00',                      # Client 2 (recent)
            '2024-01-01 09:00', '2024-01-10 14:00', '2024-01-20 13:00', '2024-01-29 18:00'  # Client 3
        ]),
        'mcc_code': [5411, 5499, 5814, 5411, 6011, 5411, 5499, 5814, 6011],
        'amount': [1000, 500, 1500, 2000, 50000, 800, 600, 1200, 30000]
    })
    
    test_clients = {1, 2, 3}
    
    try:
        result = create_activity_recency_features(test_data, test_clients)
        print(f"âœ… Ğ¢ĞµÑ�Ñ‚ Ğ¿Ñ€Ğ¾ÑˆĞµĞ»! Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¾ {result.shape[1]-1} Ñ„Ğ¸Ñ‡ĞµĞ¹ Ğ´Ğ»Ñ� {result.shape[0]} ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ¾Ğ²")
    except Exception as e:
        print(f"â�Œ Ğ¢ĞµÑ�Ñ‚ Ğ¿Ñ€Ğ¾Ğ²Ğ°Ğ»ĞµĞ½: {e}")


# Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ½Ğ¾Ğ²Ñ‹Ğµ activity/recency Ñ„Ğ¸Ñ‡Ğ¸
activity_recency_features = create_activity_recency_features(transactions, train_clients)

print("ğŸ”— Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ğº Ñ�ÑƒÑ‰ĞµÑ�Ñ‚Ğ²ÑƒÑ�Ñ‰Ğ¸Ğ¼ Ñ„Ğ¸Ñ‡Ğ°Ğ¼...")

# Ğ—Ğ°Ğ³Ñ€ÑƒĞ¶Ğ°ĞµĞ¼ enhanced Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ
enhanced_train = pd.read_parquet('/kaggle/working/SET_1_train_data.parquet')
enhanced_test = pd.read_parquet('/kaggle/working/SET_1_test_data.parquet')

# Ğ—Ğ°Ğ³Ñ€ÑƒĞ¶Ğ°ĞµĞ¼ train Ğ´Ğ»Ñ� Ğ¿Ğ¾Ğ»ÑƒÑ‡ĞµĞ½Ğ¸Ñ� client_num Ğ¸ target
train = pd.read_parquet('/kaggle/input/alpha-summer-challenge/train.pa')

# Ğ§Ğ˜Ğ¡Ğ¢Ğ�Ğ• Ğ¾Ğ±ÑŠĞµĞ´Ğ¸Ğ½ĞµĞ½Ğ¸Ğµ Ğ±ĞµĞ· Ğ´ÑƒĞ±Ğ»Ğ¸ĞºĞ°Ñ‚Ğ¾Ğ²
# Ğ”Ğ»Ñ� train
train_with_features = train[['client_num', 'target']].copy()
train_with_features = train_with_features.reset_index(drop=True)
enhanced_train = enhanced_train.reset_index(drop=True)

# Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼ Ğ¿Ğ¾ Ğ¸Ğ½Ğ´ĞµĞºÑ�Ñƒ (Ğ¾Ğ½Ğ¸ Ğ´Ğ¾Ğ»Ğ¶Ğ½Ñ‹ Ñ�Ğ¾Ğ²Ğ¿Ğ°Ğ´Ğ°Ñ‚ÑŒ)
train_combined = pd.concat([train_with_features, enhanced_train], axis=1)

# Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ğ½Ğ¾Ğ²Ñ‹Ğµ activity Ñ„Ğ¸Ñ‡Ğ¸
final_train = train_combined.merge(activity_recency_features, on='client_num', how='left').fillna(0)

# Ğ”Ğ»Ñ� test - Ğ¿Ñ€Ğ¾Ñ�Ñ‚Ğ¾ Ğ´Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ğ½Ğ¾Ğ²Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸
final_test = enhanced_test.merge(activity_recency_features, on='client_num', how='left').fillna(0)

# Ğ£Ğ±Ğ¸Ñ€Ğ°ĞµĞ¼ Ğ´ÑƒĞ±Ğ»Ğ¸ĞºĞ°Ñ‚Ñ‹ ĞºĞ¾Ğ»Ğ¾Ğ½Ğ¾Ğº ĞµÑ�Ğ»Ğ¸ ĞµÑ�Ñ‚ÑŒ
final_train = final_train.loc[:, ~final_train.columns.duplicated()]
final_test = final_test.loc[:, ~final_test.columns.duplicated()]

print(f"Train: {final_train.shape[1]-2} Ñ„Ğ¸Ñ‡ĞµĞ¹")  # -2 Ğ´Ğ»Ñ� client_num Ğ¸ target
print(f"Test: {final_test.shape[1]-1} Ñ„Ğ¸Ñ‡ĞµĞ¹")   # -1 Ğ´Ğ»Ñ� client_num

# Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼
final_train.to_parquet('/kaggle/working/SET_2_train_data.parquet', index=False)
final_test.to_parquet('/kaggle/working/SET_2_test_data.parquet', index=False)

print("âœ… Ğ“Ğ�Ğ¢Ğ�Ğ’Ğ�! Ğ¤Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚Ñ‹ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ñ‹!")


final_train


print(f"ğŸ“Š Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ¸Ñ€ÑƒĞµĞ¼ {final_train.shape[1]-2} Ñ„Ğ¸Ñ‡ĞµĞ¹ Ğ½Ğ° {final_train.shape[0]} ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ°Ñ…")

analysis_results_v2 = analyze_target_correlations(final_train)
plot_correlation_analysis(final_train, analysis_results_v2)


features_guide = """
# Ğ¡Ğ¿Ñ€Ğ°Ğ²Ğ¾Ñ‡Ğ½Ğ¸Ğº Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²

## 1. SIMPLE FEATURES

### 1.1 Ğ‘Ğ°Ğ·Ğ¾Ğ²Ñ‹Ğµ Ğ°Ğ³Ñ€ĞµĞ³Ğ°Ñ‚Ñ‹
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `total_amount` | Ğ�Ğ±Ñ‰Ğ°Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ğ²Ñ�ĞµÑ… Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ° |
| `mean_amount` | Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸ |
| `median_amount` | ĞœĞµĞ´Ğ¸Ğ°Ğ½Ğ½Ğ°Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸ |
| `std_amount` | Ğ¡Ñ‚Ğ°Ğ½Ğ´Ğ°Ñ€Ñ‚Ğ½Ğ¾Ğµ Ğ¾Ñ‚ĞºĞ»Ğ¾Ğ½ĞµĞ½Ğ¸Ğµ Ñ�ÑƒĞ¼Ğ¼ |
| `max_amount` | ĞœĞ°ĞºÑ�Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ°Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸ |
| `min_amount` | ĞœĞ¸Ğ½Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ°Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸ |
| `skew_amount` | Ğ�Ñ�Ğ¸Ğ¼Ğ¼ĞµÑ‚Ñ€Ğ¸Ñ� Ñ€Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ñ� Ñ�ÑƒĞ¼Ğ¼ |
| `kurtosis_amount` | Ğ­ĞºÑ�Ñ†ĞµÑ�Ñ� Ñ€Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ñ� Ñ�ÑƒĞ¼Ğ¼ |
| `transaction_count` | Ğ�Ğ±Ñ‰ĞµĞµ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ |
| `unique_mcc` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ ÑƒĞ½Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ñ… MCC ĞºĞ¾Ğ´Ğ¾Ğ² |
| `log_total_amount` | Ğ›Ğ¾Ğ³Ğ°Ñ€Ğ¸Ñ„Ğ¼ Ğ¾Ğ±Ñ‰ĞµĞ¹ Ñ�ÑƒĞ¼Ğ¼Ñ‹ |
| `log_mean_amount` | Ğ›Ğ¾Ğ³Ğ°Ñ€Ğ¸Ñ„Ğ¼ Ñ�Ñ€ĞµĞ´Ğ½ĞµĞ¹ Ñ�ÑƒĞ¼Ğ¼Ñ‹ |
| `log_median_amount` | Ğ›Ğ¾Ğ³Ğ°Ñ€Ğ¸Ñ„Ğ¼ Ğ¼ĞµĞ´Ğ¸Ğ°Ğ½Ğ½Ğ¾Ğ¹ Ñ�ÑƒĞ¼Ğ¼Ñ‹ |
| `log_std_amount` | Ğ›Ğ¾Ğ³Ğ°Ñ€Ğ¸Ñ„Ğ¼ Ñ�Ñ‚Ğ°Ğ½Ğ´Ğ°Ñ€Ñ‚Ğ½Ğ¾Ğ³Ğ¾ Ğ¾Ñ‚ĞºĞ»Ğ¾Ğ½ĞµĞ½Ğ¸Ñ� |

### 1.2 Rolling Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `rolling_mean_amt_{lag}` | Ğ¡ĞºĞ¾Ğ»ÑŒĞ·Ñ�Ñ‰ĞµĞµ Ñ�Ñ€ĞµĞ´Ğ½ĞµĞµ Ñ�ÑƒĞ¼Ğ¼Ñ‹ Ğ·Ğ° {lag} Ğ´Ğ½ĞµĞ¹ |
| `rolling_median_amt_{lag}` | Ğ¡ĞºĞ¾Ğ»ÑŒĞ·Ñ�Ñ‰Ğ°Ñ� Ğ¼ĞµĞ´Ğ¸Ğ°Ğ½Ğ° Ñ�ÑƒĞ¼Ğ¼Ñ‹ Ğ·Ğ° {lag} Ğ´Ğ½ĞµĞ¹ |
| `rolling_std_amt_{lag}` | Ğ¡ĞºĞ¾Ğ»ÑŒĞ·Ñ�Ñ‰ĞµĞµ Ñ�Ñ‚Ğ°Ğ½Ğ´Ğ°Ñ€Ñ‚Ğ½Ğ¾Ğµ Ğ¾Ñ‚ĞºĞ»Ğ¾Ğ½ĞµĞ½Ğ¸Ğµ Ğ·Ğ° {lag} Ğ´Ğ½ĞµĞ¹ |
| `rolling_mean_cnt_{lag}` | Ğ¡ĞºĞ¾Ğ»ÑŒĞ·Ñ�Ñ‰ĞµĞµ Ñ�Ñ€ĞµĞ´Ğ½ĞµĞµ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ° Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ·Ğ° {lag} Ğ´Ğ½ĞµĞ¹ |

*Ğ›Ğ°Ğ³Ğ¸: 3, 7, 10, 14, 30, 60, 90, 180, 360 Ğ´Ğ½ĞµĞ¹*

### 1.3 Ğ’Ñ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `avg_amt_night` | Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ğ½Ğ¾Ñ‡Ğ½Ñ‹Ñ… Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ (0-5 Ñ‡Ğ°Ñ�Ğ¾Ğ²) |
| `avg_amt_morning` | Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° ÑƒÑ‚Ñ€ĞµĞ½Ğ½Ğ¸Ñ… Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ (6-11 Ñ‡Ğ°Ñ�Ğ¾Ğ²) |
| `avg_amt_day` | Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ğ´Ğ½ĞµĞ²Ğ½Ñ‹Ñ… Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ (12-17 Ñ‡Ğ°Ñ�Ğ¾Ğ²) |
| `avg_amt_evening` | Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ğ²ĞµÑ‡ĞµÑ€Ğ½Ğ¸Ñ… Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ (18-23 Ñ‡Ğ°Ñ�Ğ°) |
| `weekday_avg_{0-6}` | Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ğ¿Ğ¾ Ğ´Ğ½Ñ�Ğ¼ Ğ½ĞµĞ´ĞµĞ»Ğ¸ (0=Ğ¿Ğ¾Ğ½ĞµĞ´ĞµĞ»ÑŒĞ½Ğ¸Ğº) |
| `interval_mean` | Ğ¡Ñ€ĞµĞ´Ğ½ĞµĞµ Ğ²Ñ€ĞµĞ¼Ñ� Ğ¼ĞµĞ¶Ğ´Ñƒ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ñ�Ğ¼Ğ¸ (Ñ‡Ğ°Ñ�Ñ‹) |
| `interval_std` | Ğ¡Ñ‚Ğ°Ğ½Ğ´Ğ°Ñ€Ñ‚Ğ½Ğ¾Ğµ Ğ¾Ñ‚ĞºĞ»Ğ¾Ğ½ĞµĞ½Ğ¸Ğµ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ¸ Ğ¼ĞµĞ¶Ğ´Ñƒ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ñ�Ğ¼Ğ¸ |
| `interval_min` | ĞœĞ¸Ğ½Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ¾Ğµ Ğ²Ñ€ĞµĞ¼Ñ� Ğ¼ĞµĞ¶Ğ´Ñƒ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ñ�Ğ¼Ğ¸ |
| `interval_max` | ĞœĞ°ĞºÑ�Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ¾Ğµ Ğ²Ñ€ĞµĞ¼Ñ� Ğ¼ĞµĞ¶Ğ´Ñƒ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ñ�Ğ¼Ğ¸ |
| `active_days` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ´Ğ½ĞµĞ¹ Ğ¾Ñ‚ Ğ¿ĞµÑ€Ğ²Ğ¾Ğ¹ Ğ´Ğ¾ Ğ¿Ğ¾Ñ�Ğ»ĞµĞ´Ğ½ĞµĞ¹ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸ |

### 1.4 MCC Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `mcc_{ĞºĞ¾Ğ´}` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ¿Ğ¾ ĞºĞ¾Ğ½ĞºÑ€ĞµÑ‚Ğ½Ğ¾Ğ¼Ñƒ MCC ĞºĞ¾Ğ´Ñƒ |
| `mcc_other` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ñ€ĞµĞ´ĞºĞ¸Ñ… MCC ĞºĞ¾Ğ´Ğ¾Ğ² |
| `mcc1`, `mcc2`, `mcc3` | Ğ¢Ğ¾Ğ¿-3 MCC ĞºĞ¾Ğ´Ğ° ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ° Ğ¿Ğ¾ Ñ‡Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ğµ |

### 1.5 RFM Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `rfm_recency` | Ğ”Ğ½Ğ¸ Ñ� Ğ¿Ğ¾Ñ�Ğ»ĞµĞ´Ğ½ĞµĞ¹ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸ |
| `rfm_frequency` | Ğ§Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ğ° Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ (Ğ¾Ğ±Ñ‰ĞµĞµ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾) |
| `rfm_monetary` | Ğ”ĞµĞ½ĞµĞ¶Ğ½Ğ°Ñ� Ñ†ĞµĞ½Ğ½Ğ¾Ñ�Ñ‚ÑŒ (Ğ¾Ğ±Ñ‰Ğ°Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ°) |

---

## 2. ADVANCED FEATURES

### 2.1 Momentum & Velocity
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `momentum_sum` | Ğ˜Ğ·Ğ¼ĞµĞ½ĞµĞ½Ğ¸Ğµ Ñ�ÑƒĞ¼Ğ¼Ñ‹ Ñ‚Ñ€Ğ°Ñ‚ (Ğ²Ñ‚Ğ¾Ñ€Ğ°Ñ� Ğ¿Ğ¾Ğ»Ğ¾Ğ²Ğ¸Ğ½Ğ° vs Ğ¿ĞµÑ€Ğ²Ğ°Ñ�) |
| `momentum_mean` | Ğ˜Ğ·Ğ¼ĞµĞ½ĞµĞ½Ğ¸Ğµ Ñ�Ñ€ĞµĞ´Ğ½ĞµĞ¹ Ñ�ÑƒĞ¼Ğ¼Ñ‹ Ğ¼ĞµĞ¶Ğ´Ñƒ Ğ¿ĞµÑ€Ğ¸Ğ¾Ğ´Ğ°Ğ¼Ğ¸ |
| `momentum_count` | Ğ˜Ğ·Ğ¼ĞµĞ½ĞµĞ½Ğ¸Ğµ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ° Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ |
| `momentum_mcc` | Ğ˜Ğ·Ğ¼ĞµĞ½ĞµĞ½Ğ¸Ğµ Ñ€Ğ°Ğ·Ğ½Ğ¾Ğ¾Ğ±Ñ€Ğ°Ğ·Ğ¸Ñ� MCC ĞºĞ¾Ğ´Ğ¾Ğ² |
| `behavior_stability` | Ğ¡Ñ‚Ğ°Ğ±Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¿Ğ¾Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ� (Ğ¾Ğ±Ñ€Ğ°Ñ‚Ğ½Ğ°Ñ� Ğº Ğ¸Ğ·Ğ¼ĞµĞ½Ñ‡Ğ¸Ğ²Ğ¾Ñ�Ñ‚Ğ¸) |

### 2.2 Ğ¦Ğ¸ĞºĞ»Ğ¸Ñ‡ĞµÑ�ĞºĞ¸Ğµ Ğ¸ Ñ�ĞµĞ·Ğ¾Ğ½Ğ½Ñ‹Ğµ
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `weekend_txns` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ² Ğ²Ñ‹Ñ…Ğ¾Ğ´Ğ½Ñ‹Ğµ |
| `weekend_ratio` | Ğ”Ğ¾Ğ»Ñ� Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ² Ğ²Ñ‹Ñ…Ğ¾Ğ´Ğ½Ñ‹Ğµ |
| `month_start_txns` | Ğ¢Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸ Ğ² Ğ½Ğ°Ñ‡Ğ°Ğ»Ğµ Ğ¼ĞµÑ�Ñ�Ñ†Ğ° (1-7 Ñ‡Ğ¸Ñ�Ğ»Ğ¾) |
| `month_start_ratio` | Ğ”Ğ¾Ğ»Ñ� Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ² Ğ½Ğ°Ñ‡Ğ°Ğ»Ğµ Ğ¼ĞµÑ�Ñ�Ñ†Ğ° |
| `month_end_txns` | Ğ¢Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸ Ğ² ĞºĞ¾Ğ½Ñ†Ğµ Ğ¼ĞµÑ�Ñ�Ñ†Ğ° (25-31 Ñ‡Ğ¸Ñ�Ğ»Ğ¾) |
| `month_end_ratio` | Ğ”Ğ¾Ğ»Ñ� Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ² ĞºĞ¾Ğ½Ñ†Ğµ Ğ¼ĞµÑ�Ñ�Ñ†Ğ° |
| `avg_hour_sin` | Ğ¦Ğ¸ĞºĞ»Ğ¸Ñ‡ĞµÑ�ĞºĞ¾Ğµ ĞºĞ¾Ğ´Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ñ�Ñ€ĞµĞ´Ğ½ĞµĞ³Ğ¾ Ñ‡Ğ°Ñ�Ğ° (sin) |
| `avg_hour_cos` | Ğ¦Ğ¸ĞºĞ»Ğ¸Ñ‡ĞµÑ�ĞºĞ¾Ğµ ĞºĞ¾Ğ´Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ñ�Ñ€ĞµĞ´Ğ½ĞµĞ³Ğ¾ Ñ‡Ğ°Ñ�Ğ° (cos) |
| `avg_weekday_sin` | Ğ¦Ğ¸ĞºĞ»Ğ¸Ñ‡ĞµÑ�ĞºĞ¾Ğµ ĞºĞ¾Ğ´Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ğ´Ğ½Ñ� Ğ½ĞµĞ´ĞµĞ»Ğ¸ (sin) |
| `avg_weekday_cos` | Ğ¦Ğ¸ĞºĞ»Ğ¸Ñ‡ĞµÑ�ĞºĞ¾Ğµ ĞºĞ¾Ğ´Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ğ´Ğ½Ñ� Ğ½ĞµĞ´ĞµĞ»Ğ¸ (cos) |

### 2.3 Ğ¡ĞµÑ‚ĞµĞ²Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `avg_merchant_popularity` | Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ğ¿Ğ¾Ğ¿ÑƒĞ»Ñ�Ñ€Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¸Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·ÑƒĞµĞ¼Ñ‹Ñ… Ğ¼ĞµÑ€Ñ‡Ğ°Ğ½Ñ‚Ğ¾Ğ² |
| `max_merchant_popularity` | ĞœĞ°ĞºÑ�Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ°Ñ� Ğ¿Ğ¾Ğ¿ÑƒĞ»Ñ�Ñ€Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¼ĞµÑ€Ñ‡Ğ°Ğ½Ñ‚Ğ° |
| `min_merchant_popularity` | ĞœĞ¸Ğ½Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ°Ñ� Ğ¿Ğ¾Ğ¿ÑƒĞ»Ñ�Ñ€Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¼ĞµÑ€Ñ‡Ğ°Ğ½Ñ‚Ğ° |
| `std_merchant_popularity` | Ğ Ğ°Ğ·Ğ±Ñ€Ğ¾Ñ� Ğ¿Ğ¾Ğ¿ÑƒĞ»Ñ�Ñ€Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ¼ĞµÑ€Ñ‡Ğ°Ğ½Ñ‚Ğ¾Ğ² |
| `unique_merchants_x` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ ÑƒĞ½Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ñ… Ğ¼ĞµÑ€Ñ‡Ğ°Ğ½Ñ‚Ğ¾Ğ² |
| `exclusivity_score` | Ğ¡ĞºĞ»Ğ¾Ğ½Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğº Ñ€ĞµĞ´ĞºĞ¸Ğ¼ Ğ¼ĞµÑ€Ñ‡Ğ°Ğ½Ñ‚Ğ°Ğ¼ |

### 2.4 Ğ­Ğ½Ñ‚Ñ€Ğ¾Ğ¿Ğ¸Ñ� Ğ¸ Ñ�Ğ»Ğ¾Ğ¶Ğ½Ğ¾Ñ�Ñ‚ÑŒ
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `mcc_entropy` | Ğ­Ğ½Ñ‚Ñ€Ğ¾Ğ¿Ğ¸Ñ� Ñ€Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ñ� MCC ĞºĞ¾Ğ´Ğ¾Ğ² |
| `time_entropy` | Ğ­Ğ½Ñ‚Ñ€Ğ¾Ğ¿Ğ¸Ñ� Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ñ… Ğ¿Ğ°Ñ‚Ñ‚ĞµÑ€Ğ½Ğ¾Ğ² |
| `amount_entropy` | Ğ­Ğ½Ñ‚Ñ€Ğ¾Ğ¿Ğ¸Ñ� Ñ€Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ñ� Ñ�ÑƒĞ¼Ğ¼ |
| `total_complexity` | Ğ�Ğ±Ñ‰Ğ°Ñ� Ñ�Ğ»Ğ¾Ğ¶Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¿Ğ¾Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ� |

### 2.5 Ğ�Ğ½Ğ¾Ğ¼Ğ°Ğ»Ğ¸Ğ¸ Ğ¿Ğ¾Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ�
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `z_avg_amount` | Z-score Ñ�Ñ€ĞµĞ´Ğ½ĞµĞ¹ Ñ�ÑƒĞ¼Ğ¼Ñ‹ |
| `z_txn_count` | Z-score ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ° Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ |
| `z_active_days` | Z-score Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ñ‹Ñ… Ğ´Ğ½ĞµĞ¹ |
| `anomaly_score` | Ğ�Ğ±Ñ‰Ğ¸Ğ¹ Ñ�ĞºĞ¾Ñ€ Ğ°Ğ½Ğ¾Ğ¼Ğ°Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚Ğ¸ |

### 2.6 ĞŸÑ€Ğ¾Ğ´Ğ²Ğ¸Ğ½ÑƒÑ‚Ñ‹Ğµ rolling
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `ema_amount` | Ğ­ĞºÑ�Ğ¿Ğ¾Ğ½ĞµĞ½Ñ†Ğ¸Ğ°Ğ»ÑŒĞ½Ğ¾Ğµ Ñ�ĞºĞ¾Ğ»ÑŒĞ·Ñ�Ñ‰ĞµĞµ Ñ�Ñ€ĞµĞ´Ğ½ĞµĞµ Ñ�ÑƒĞ¼Ğ¼ |
| `ema_volatility` | EMA Ğ²Ğ¾Ğ»Ğ°Ñ‚Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚Ğ¸ |
| `spending_autocorr` | Ğ�Ğ²Ñ‚Ğ¾ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ñ� Ğ´Ğ½ĞµĞ²Ğ½Ñ‹Ñ… Ñ‚Ñ€Ğ°Ñ‚ |

### 2.7 ĞšĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¸ Ğ¿Ğ¾ Ğ¿Ğ¾Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ�
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `frequent_txns` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ² Ñ‡Ğ°Ñ�Ñ‚Ñ‹Ñ… ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ñ�Ñ… |
| `frequent_ratio` | Ğ”Ğ¾Ğ»Ñ� Ñ‡Ğ°Ñ�Ñ‚Ñ‹Ñ… ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¹ |
| `expensive_txns` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ´Ğ¾Ñ€Ğ¾Ğ³Ğ¸Ñ… Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ |
| `expensive_ratio` | Ğ”Ğ¾Ğ»Ñ� Ğ´Ğ¾Ñ€Ğ¾Ğ³Ğ¸Ñ… Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ |
| `moderate_txns` | Ğ£Ğ¼ĞµÑ€ĞµĞ½Ğ½Ñ‹Ğµ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸ |
| `rare_txns` | Ğ ĞµĞ´ĞºĞ¸Ğµ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸ |
| `frequent_expensive_ratio` | Ğ¡Ğ¾Ğ¾Ñ‚Ğ½Ğ¾ÑˆĞµĞ½Ğ¸Ğµ Ñ‡Ğ°Ñ�Ñ‚Ñ‹Ñ… Ğº Ğ´Ğ¾Ñ€Ğ¾Ğ³Ğ¸Ğ¼ |
| `category_diversity` | Ğ Ğ°Ğ·Ğ½Ğ¾Ğ¾Ğ±Ñ€Ğ°Ğ·Ğ¸Ğµ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¹ |
| `practical_spending_index` | Ğ˜Ğ½Ğ´ĞµĞºÑ� Ğ¿Ñ€Ğ°ĞºÑ‚Ğ¸Ñ‡Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ñ‚Ñ€Ğ°Ñ‚ |

---

## 3. BALANCE-SPECIFIC FEATURES

### 3.1 Ğ”ĞµĞ½ĞµĞ¶Ğ½Ñ‹Ğµ Ğ¿Ğ¾Ñ‚Ğ¾ĞºĞ¸
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `large_inflow_count` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ ĞºÑ€ÑƒĞ¿Ğ½Ñ‹Ñ… Ğ¿Ğ¾Ñ�Ñ‚ÑƒĞ¿Ğ»ĞµĞ½Ğ¸Ğ¹ |
| `large_inflow_ratio` | Ğ”Ğ¾Ğ»Ñ� ĞºÑ€ÑƒĞ¿Ğ½Ñ‹Ñ… Ğ¿Ğ¾Ñ�Ñ‚ÑƒĞ¿Ğ»ĞµĞ½Ğ¸Ğ¹ |
| `regular_spend_count` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ€ĞµĞ³ÑƒĞ»Ñ�Ñ€Ğ½Ñ‹Ñ… Ñ‚Ñ€Ğ°Ñ‚ |
| `regular_spend_ratio` | Ğ”Ğ¾Ğ»Ñ� Ñ€ĞµĞ³ÑƒĞ»Ñ�Ñ€Ğ½Ñ‹Ñ… Ñ‚Ñ€Ğ°Ñ‚ |
| `small_spend_count` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ¼ĞµĞ»ĞºĞ¸Ñ… Ñ‚Ñ€Ğ°Ñ‚ |
| `small_spend_ratio` | Ğ”Ğ¾Ğ»Ñ� Ğ¼ĞµĞ»ĞºĞ¸Ñ… Ñ‚Ñ€Ğ°Ñ‚ |
| `inflow_spend_balance` | Ğ‘Ğ°Ğ»Ğ°Ğ½Ñ� Ğ¿Ğ¾Ñ�Ñ‚ÑƒĞ¿Ğ»ĞµĞ½Ğ¸Ğ¹ Ğ¸ Ñ‚Ñ€Ğ°Ñ‚ |

### 3.2 Ğ¡ĞºĞ¾Ñ€Ğ¾Ñ�Ñ‚ÑŒ Ñ‚Ñ€Ğ°Ñ‚
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `burn_rate` | Ğ¡ĞºĞ¾Ñ€Ğ¾Ñ�Ñ‚ÑŒ Ñ�Ğ¶Ğ¸Ğ³Ğ°Ğ½Ğ¸Ñ� Ğ´ĞµĞ½ĞµĞ³ (Ñ�ÑƒĞ¼Ğ¼Ğ°/Ğ´ĞµĞ½ÑŒ) |
| `acceleration` | Ğ£Ñ�ĞºĞ¾Ñ€ĞµĞ½Ğ¸Ğµ Ñ‚Ñ€Ğ°Ñ‚ |
| `days_to_spend_median` | Ğ”Ğ½ĞµĞ¹ Ğ½Ğ° Ñ‚Ñ€Ğ°Ñ‚Ñƒ Ğ¼ĞµĞ´Ğ¸Ğ°Ğ½Ğ½Ğ¾Ğ¹ Ñ�ÑƒĞ¼Ğ¼Ñ‹ |

### 3.3 Ğ¤Ğ¸Ğ½Ğ°Ğ½Ñ�Ğ¾Ğ²Ğ°Ñ� Ğ´Ğ¸Ñ�Ñ†Ğ¸Ğ¿Ğ»Ğ¸Ğ½Ğ°
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `spending_cv` | ĞšĞ¾Ñ�Ñ„Ñ„Ğ¸Ñ†Ğ¸ĞµĞ½Ñ‚ Ğ²Ğ°Ñ€Ğ¸Ğ°Ñ†Ğ¸Ğ¸ Ñ‚Ñ€Ğ°Ñ‚ |
| `inactive_days_ratio` | Ğ”Ğ¾Ğ»Ñ� Ğ½ĞµĞ°ĞºÑ‚Ğ¸Ğ²Ğ½Ñ‹Ñ… Ğ´Ğ½ĞµĞ¹ |
| `max_spending_gap` | ĞœĞ°ĞºÑ�Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ°Ñ� Ğ¿Ğ°ÑƒĞ·Ğ° Ğ² Ñ‚Ñ€Ğ°Ñ‚Ğ°Ñ… |
| `weekly_stability` | Ğ¡Ñ‚Ğ°Ğ±Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ Ğ½ĞµĞ´ĞµĞ»ÑŒĞ½Ñ‹Ñ… Ñ‚Ñ€Ğ°Ñ‚ |

### 3.4 Ğ­ĞºĞ¾Ğ½Ğ¾Ğ¼Ğ¸Ñ‡ĞµÑ�ĞºĞ¸Ğµ Ñ†Ğ¸ĞºĞ»Ñ‹
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `spending_month_start` | Ğ¢Ñ€Ğ°Ñ‚Ñ‹ Ğ² Ğ½Ğ°Ñ‡Ğ°Ğ»Ğµ Ğ¼ĞµÑ�Ñ�Ñ†Ğ° |
| `spending_month_mid` | Ğ¢Ñ€Ğ°Ñ‚Ñ‹ Ğ² Ñ�ĞµÑ€ĞµĞ´Ğ¸Ğ½Ğµ Ğ¼ĞµÑ�Ñ�Ñ†Ğ° |
| `spending_month_end` | Ğ¢Ñ€Ğ°Ñ‚Ñ‹ Ğ² ĞºĞ¾Ğ½Ñ†Ğµ Ğ¼ĞµÑ�Ñ�Ñ†Ğ° |
| `monthly_unevenness` | Ğ�ĞµÑ€Ğ°Ğ²Ğ½Ğ¾Ğ¼ĞµÑ€Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ñ‚Ñ€Ğ°Ñ‚ Ğ¿Ğ¾ Ğ¼ĞµÑ�Ñ�Ñ†Ñƒ |

### 3.5 Ğ›Ğ¾Ñ�Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¸ Ğ¿Ñ€Ğ¸Ğ²Ñ‹Ñ‡ĞºĞ¸
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `max_visits_merchant` | ĞœĞ°ĞºÑ�Ğ¸Ğ¼ÑƒĞ¼ Ğ¿Ğ¾Ñ�ĞµÑ‰ĞµĞ½Ğ¸Ğ¹ Ğ¾Ğ´Ğ½Ğ¾Ğ³Ğ¾ Ğ¼ĞµÑ€Ñ‡Ğ°Ğ½Ñ‚Ğ° |
| `avg_visits_merchant` | Ğ¡Ñ€ĞµĞ´Ğ½ĞµĞµ Ğ¿Ğ¾Ñ�ĞµÑ‰ĞµĞ½Ğ¸Ğ¹ Ğ¼ĞµÑ€Ñ‡Ğ°Ğ½Ñ‚Ğ¾Ğ² |
| `std_visits_merchant` | Ğ Ğ°Ğ·Ğ±Ñ€Ğ¾Ñ� Ğ¿Ğ¾Ñ�ĞµÑ‰ĞµĞ½Ğ¸Ğ¹ |
| `max_spent_merchant` | ĞœĞ°ĞºÑ�Ğ¸Ğ¼ÑƒĞ¼ Ğ¿Ğ¾Ñ‚Ñ€Ğ°Ñ‡ĞµĞ½Ğ¾ Ñƒ Ğ¾Ğ´Ğ½Ğ¾Ğ³Ğ¾ Ğ¼ĞµÑ€Ñ‡Ğ°Ğ½Ñ‚Ğ° |
| `avg_spent_merchant` | Ğ¡Ñ€ĞµĞ´Ğ½ĞµĞµ Ğ¿Ğ¾Ñ‚Ñ€Ğ°Ñ‡ĞµĞ½Ğ¾ Ñƒ Ğ¼ĞµÑ€Ñ‡Ğ°Ğ½Ñ‚Ğ¾Ğ² |
| `unique_merchants_y` | Ğ£Ğ½Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ğµ Ğ¼ĞµÑ€Ñ‡Ğ°Ğ½Ñ‚Ñ‹ |
| `loyalty_index` | Ğ˜Ğ½Ğ´ĞµĞºÑ� Ğ»Ğ¾Ñ�Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚Ğ¸ |
| `weekday_{0-6}_spending` | Ğ¢Ñ€Ğ°Ñ‚Ñ‹ Ğ¿Ğ¾ Ğ´Ğ½Ñ�Ğ¼ Ğ½ĞµĞ´ĞµĞ»Ğ¸ |
| `weekday_habit_stability` | Ğ¡Ñ‚Ğ°Ğ±Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¿Ñ€Ğ¸Ğ²Ñ‹Ñ‡ĞµĞº |

### 3.6 Ğ’Ñ€ĞµĞ¼ĞµĞ½Ğ½Ğ¾Ğ¹ Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ·
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `timing_regularity` | Ğ ĞµĞ³ÑƒĞ»Ñ�Ñ€Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¿Ğ¾ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ¸ |
| `peak_activity_hour` | ĞŸĞ¸ĞºĞ¾Ğ²Ñ‹Ğ¹ Ñ‡Ğ°Ñ� Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚Ğ¸ |
| `activity_concentration` | ĞšĞ¾Ğ½Ñ†ĞµĞ½Ñ‚Ñ€Ğ°Ñ†Ğ¸Ñ� Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚Ğ¸ |

### 3.7 Ğ˜Ğ½Ğ´Ğ¸ĞºĞ°Ñ‚Ğ¾Ñ€Ñ‹ Ñ€Ğ¸Ñ�ĞºĞ°
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `extreme_spending_ratio` | Ğ”Ğ¾Ğ»Ñ� Ñ�ĞºÑ�Ñ‚Ñ€ĞµĞ¼Ğ°Ğ»ÑŒĞ½Ñ‹Ñ… Ñ‚Ñ€Ğ°Ñ‚ |
| `max_vs_median_ratio` | Ğ�Ñ‚Ğ½Ğ¾ÑˆĞµĞ½Ğ¸Ğµ Ğ¼Ğ°ĞºÑ�Ğ¸Ğ¼ÑƒĞ¼Ğ° Ğº Ğ¼ĞµĞ´Ğ¸Ğ°Ğ½Ğµ |
| `spending_volatility` | Ğ’Ğ¾Ğ»Ğ°Ñ‚Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ Ñ‚Ñ€Ğ°Ñ‚ |
| `end_period_spending_bias` | Ğ¡Ğ¼ĞµÑ‰ĞµĞ½Ğ¸Ğµ Ğº ĞºĞ¾Ğ½Ñ†Ñƒ Ğ¿ĞµÑ€Ğ¸Ğ¾Ğ´Ğ° |

### 3.8 Ğ”Ğ¸Ğ²ĞµÑ€Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ñ�
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `mcc_diversification` | Ğ”Ğ¸Ğ²ĞµÑ€Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ñ� Ğ¿Ğ¾ MCC |
| `merchant_diversification` | Ğ”Ğ¸Ğ²ĞµÑ€Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ñ� Ğ¿Ğ¾ Ğ¼ĞµÑ€Ñ‡Ğ°Ğ½Ñ‚Ğ°Ğ¼ |
| `total_diversification` | Ğ�Ğ±Ñ‰Ğ°Ñ� Ğ´Ğ¸Ğ²ĞµÑ€Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ñ� |

### 3.9 Ğ›Ğ°Ğ³Ğ¾Ğ²Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `amount_lag_{1,3,7,14}` | Ğ¡ÑƒĞ¼Ğ¼Ğ° {N} Ğ´Ğ½ĞµĞ¹ Ğ½Ğ°Ğ·Ğ°Ğ´ |
| `txns_lag_{1,3,7,14}` | Ğ¢Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸ {N} Ğ´Ğ½ĞµĞ¹ Ğ½Ğ°Ğ·Ğ°Ğ´ |
| `avg_amount_lag_{1,3,7,14}` | Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° {N} Ğ´Ğ½ĞµĞ¹ Ğ½Ğ°Ğ·Ğ°Ğ´ |

### 3.10 Ğ’Ñ‹Ñ…Ğ¾Ğ´Ğ½Ñ‹Ğµ vs Ğ±ÑƒĞ´Ğ½Ğ¸
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `weekend_amount_sum` | Ğ�Ğ±Ñ‰Ğ°Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ğ² Ğ²Ñ‹Ñ…Ğ¾Ğ´Ğ½Ñ‹Ğµ |
| `weekend_amount_mean` | Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ğ² Ğ²Ñ‹Ñ…Ğ¾Ğ´Ğ½Ñ‹Ğµ |
| `weekend_amount_count` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ² Ğ²Ñ‹Ñ…Ğ¾Ğ´Ğ½Ñ‹Ğµ |
| `weekend_unique_mcc` | Ğ£Ğ½Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ğµ MCC Ğ² Ğ²Ñ‹Ñ…Ğ¾Ğ´Ğ½Ñ‹Ğµ |
| `weekday_amount_sum` | Ğ�Ğ±Ñ‰Ğ°Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ğ² Ğ±ÑƒĞ´Ğ½Ğ¸ |
| `weekday_amount_mean` | Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ğ² Ğ±ÑƒĞ´Ğ½Ğ¸ |
| `weekday_amount_count` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ² Ğ±ÑƒĞ´Ğ½Ğ¸ |
| `weekday_unique_mcc` | Ğ£Ğ½Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ğµ MCC Ğ² Ğ±ÑƒĞ´Ğ½Ğ¸ |
| `weekend_weekday_amount_ratio` | Ğ¡Ğ¾Ğ¾Ñ‚Ğ½Ğ¾ÑˆĞµĞ½Ğ¸Ğµ Ñ�ÑƒĞ¼Ğ¼ Ğ²Ñ‹Ñ…Ğ¾Ğ´Ğ½Ñ‹Ğµ/Ğ±ÑƒĞ´Ğ½Ğ¸ |
| `weekend_weekday_count_ratio` | Ğ¡Ğ¾Ğ¾Ñ‚Ğ½Ğ¾ÑˆĞµĞ½Ğ¸Ğµ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ° Ğ²Ñ‹Ñ…Ğ¾Ğ´Ğ½Ñ‹Ğµ/Ğ±ÑƒĞ´Ğ½Ğ¸ |
| `weekend_activity_preference` | ĞŸÑ€ĞµĞ´Ğ¿Ğ¾Ñ‡Ñ‚ĞµĞ½Ğ¸Ğµ Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ² Ğ²Ñ‹Ñ…Ğ¾Ğ´Ğ½Ñ‹Ğµ |

### 3.11 ĞšĞ¾Ğ¼Ğ¿Ğ¾Ğ·Ğ¸Ñ‚Ğ½Ñ‹Ğµ Ğ¸Ğ½Ğ´ĞµĞºÑ�Ñ‹
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `financial_stability_score` | Ğ¡ĞºĞ¾Ñ€ Ñ„Ğ¸Ğ½Ğ°Ğ½Ñ�Ğ¾Ğ²Ğ¾Ğ¹ Ñ�Ñ‚Ğ°Ğ±Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚Ğ¸ |
| `bankruptcy_risk_score_x` | Ğ Ğ¸Ñ�Ğº Ğ±Ğ°Ğ½ĞºÑ€Ğ¾Ñ‚Ñ�Ñ‚Ğ²Ğ° (Ğ²ĞµÑ€Ñ�Ğ¸Ñ� 1) |
| `financial_health_index` | Ğ˜Ğ½Ğ´ĞµĞºÑ� Ñ„Ğ¸Ğ½Ğ°Ğ½Ñ�Ğ¾Ğ²Ğ¾Ğ³Ğ¾ Ğ·Ğ´Ğ¾Ñ€Ğ¾Ğ²ÑŒÑ� |

---

## 4. DATA-DRIVEN FEATURES

### 4.1 MCC-Ñ�Ğ¿ĞµÑ†Ğ¸Ñ„Ğ¸Ñ‡Ğ½Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `mcc5411_total` | Ğ�Ğ±Ñ‰Ğ°Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ğ¿Ğ¾ MCC 5411 (Ğ¿Ñ€Ğ¾Ğ´ÑƒĞºÑ‚Ñ‹) |
| `mcc5411_avg` | Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ğ¿Ğ¾ MCC 5411 |
| `mcc5411_count` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ MCC 5411 |
| `mcc5411_std` | Ğ¡Ñ‚Ğ°Ğ½Ğ´Ğ°Ñ€Ñ‚Ğ½Ğ¾Ğµ Ğ¾Ñ‚ĞºĞ»Ğ¾Ğ½ĞµĞ½Ğ¸Ğµ MCC 5411 |
| `mcc5411_avg_hour` | Ğ¡Ñ€ĞµĞ´Ğ½Ğ¸Ğ¹ Ñ‡Ğ°Ñ� Ğ´Ğ»Ñ� MCC 5411 |
| `mcc5411_avg_weekday` | Ğ¡Ñ€ĞµĞ´Ğ½Ğ¸Ğ¹ Ğ´ĞµĞ½ÑŒ Ğ½ĞµĞ´ĞµĞ»Ğ¸ Ğ´Ğ»Ñ� MCC 5411 |
| `mcc5499_total` | Ğ�Ğ±Ñ‰Ğ°Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ğ¿Ğ¾ MCC 5499 |
| `mcc5499_avg` | Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ğ¿Ğ¾ MCC 5499 |
| `mcc5499_count` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ MCC 5499 |
| `mcc6011_total` | Ğ�Ğ±Ñ‰Ğ°Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ğ¿Ğ¾ MCC 6011 (Ğ´Ğ¾Ñ€Ğ¾Ğ³Ğ¸Ğµ Ğ¾Ğ¿ĞµÑ€Ğ°Ñ†Ğ¸Ğ¸) |
| `mcc6011_avg` | Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ğ¿Ğ¾ MCC 6011 |
| `mcc6011_count` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ MCC 6011 |
| `mcc6011_max` | ĞœĞ°ĞºÑ�Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ°Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ğ¿Ğ¾ MCC 6011 |

### 4.2 ĞšĞ¾Ğ½Ñ†ĞµĞ½Ñ‚Ñ€Ğ°Ñ†Ğ¸Ñ� Ñ‚Ñ€Ğ°Ñ‚
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `top3_concentration` | ĞšĞ¾Ğ½Ñ†ĞµĞ½Ñ‚Ñ€Ğ°Ñ†Ğ¸Ñ� Ğ½Ğ° Ñ‚Ğ¾Ğ¿-3 MCC |

### 4.3 Ğ’Ñ‹Ñ�Ğ¾ĞºĞ¸Ğµ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `high_total` | Ğ�Ğ±Ñ‰Ğ°Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ğ²Ñ‹Ñ�Ğ¾ĞºĞ¸Ñ… Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ |
| `high_count` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ²Ñ‹Ñ�Ğ¾ĞºĞ¸Ñ… Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ |
| `high_avg` | Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ğ²Ñ‹Ñ�Ğ¾ĞºĞ°Ñ� Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ñ� |
| `high_unique_mcc` | Ğ£Ğ½Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ğµ MCC Ğ² Ğ²Ñ‹Ñ�Ğ¾ĞºĞ¸Ñ… Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ñ�Ñ… |
| `high_ratio` | Ğ”Ğ¾Ğ»Ñ� Ğ²Ñ‹Ñ�Ğ¾ĞºĞ¸Ñ… Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ |

### 4.4 Ğ§Ğ°Ñ�Ñ‚Ñ‹Ğµ vs Ñ€ĞµĞ´ĞºĞ¸Ğµ
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `frequent_spending` | Ğ¢Ñ€Ğ°Ñ‚Ñ‹ Ğ² Ñ‡Ğ°Ñ�Ñ‚Ñ‹Ñ… ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ñ�Ñ… |
| `rare_spending` | Ğ¢Ñ€Ğ°Ñ‚Ñ‹ Ğ² Ñ€ĞµĞ´ĞºĞ¸Ñ… ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ñ�Ñ… |
| `frequent_rare_ratio` | Ğ¡Ğ¾Ğ¾Ñ‚Ğ½Ğ¾ÑˆĞµĞ½Ğ¸Ğµ Ñ‡Ğ°Ñ�Ñ‚Ñ‹Ñ… Ğº Ñ€ĞµĞ´ĞºĞ¸Ğ¼ |
| `rare_tendency` | Ğ¡ĞºĞ»Ğ¾Ğ½Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğº Ñ€ĞµĞ´ĞºĞ¸Ğ¼ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ñ�Ğ¼ |

### 4.5 ĞŸĞµÑ€ĞµĞºĞ»Ñ�Ñ‡ĞµĞ½Ğ¸Ñ� ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¹
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `category_switches` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ¿ĞµÑ€ĞµĞºĞ»Ñ�Ñ‡ĞµĞ½Ğ¸Ğ¹ Ğ¼ĞµĞ¶Ğ´Ñƒ MCC |
| `switch_rate` | Ğ§Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ğ° Ğ¿ĞµÑ€ĞµĞºĞ»Ñ�Ñ‡ĞµĞ½Ğ¸Ğ¹ |

### 4.6 Ğ¡Ñ‚Ğ°Ğ±Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¿Ğ¾Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ�
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `unique_active_days` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ ÑƒĞ½Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ñ… Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ñ‹Ñ… Ğ´Ğ½ĞµĞ¹ |
| `weekday_amount_stability` | Ğ¡Ñ‚Ğ°Ğ±Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ Ñ‚Ñ€Ğ°Ñ‚ Ğ¿Ğ¾ Ğ´Ğ½Ñ�Ğ¼ Ğ½ĞµĞ´ĞµĞ»Ğ¸ |

### 4.7 Ğ’Ñ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ğµ Ğ¿Ğ°Ñ‚Ñ‚ĞµÑ€Ğ½Ñ‹
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `avg_hour` | Ğ¡Ñ€ĞµĞ´Ğ½Ğ¸Ğ¹ Ñ‡Ğ°Ñ� Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ |
| `std_hour` | Ğ¡Ñ‚Ğ°Ğ½Ğ´Ğ°Ñ€Ñ‚Ğ½Ğ¾Ğµ Ğ¾Ñ‚ĞºĞ»Ğ¾Ğ½ĞµĞ½Ğ¸Ğµ Ñ‡Ğ°Ñ�Ğ¾Ğ² |
| `avg_weekday` | Ğ¡Ñ€ĞµĞ´Ğ½Ğ¸Ğ¹ Ğ´ĞµĞ½ÑŒ Ğ½ĞµĞ´ĞµĞ»Ğ¸ |
| `std_weekday` | Ğ¡Ñ‚Ğ°Ğ½Ğ´Ğ°Ñ€Ñ‚Ğ½Ğ¾Ğµ Ğ¾Ñ‚ĞºĞ»Ğ¾Ğ½ĞµĞ½Ğ¸Ğµ Ğ´Ğ½ĞµĞ¹ Ğ½ĞµĞ´ĞµĞ»Ğ¸ |

---

## 5. ACTIVITY/RECENCY FEATURES

### 5.1 Enhanced Activity
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `activity_density` | ĞŸĞ»Ğ¾Ñ‚Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚Ğ¸ (Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸/Ğ´ĞµĞ½ÑŒ) |
| `avg_weekly_txns` | Ğ¡Ñ€ĞµĞ´Ğ½ĞµĞµ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ² Ğ½ĞµĞ´ĞµĞ»Ñ� |
| `std_weekly_txns` | Ğ¡Ñ‚Ğ°Ğ½Ğ´Ğ°Ñ€Ñ‚Ğ½Ğ¾Ğµ Ğ¾Ñ‚ĞºĞ»Ğ¾Ğ½ĞµĞ½Ğ¸Ğµ Ğ½ĞµĞ´ĞµĞ»ÑŒĞ½Ñ‹Ñ… Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ |
| `active_weeks` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ñ‹Ñ… Ğ½ĞµĞ´ĞµĞ»ÑŒ |
| `weekly_consistency` | Ğ¡Ğ¾Ğ³Ğ»Ğ°Ñ�Ğ¾Ğ²Ğ°Ğ½Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ½ĞµĞ´ĞµĞ»ÑŒĞ½Ğ¾Ğ¹ Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚Ğ¸ |

### 5.2 Advanced Recency
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `days_since_last_txn` | Ğ”Ğ½Ğ¸ Ñ� Ğ¿Ğ¾Ñ�Ğ»ĞµĞ´Ğ½ĞµĞ¹ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸ |
| `days_since_penultimate` | Ğ”Ğ½Ğ¸ Ñ� Ğ¿Ñ€ĞµĞ´Ğ¿Ğ¾Ñ�Ğ»ĞµĞ´Ğ½ĞµĞ¹ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸ |
| `recency_acceleration` | Ğ£Ñ�ĞºĞ¾Ñ€ĞµĞ½Ğ¸Ğµ Ğ¿Ğ°Ñ�Ñ�Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚Ğ¸ |
| `recency_trend` | Ğ¢Ñ€ĞµĞ½Ğ´ ÑƒĞ²ĞµĞ»Ğ¸Ñ‡ĞµĞ½Ğ¸Ñ� Ğ¿Ğ°ÑƒĞ· |

### 5.3 Recent Activity Analysis
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `recent_7d_sum_amount` | Ğ¡ÑƒĞ¼Ğ¼Ğ° Ğ·Ğ° Ğ¿Ğ¾Ñ�Ğ»ĞµĞ´Ğ½Ğ¸Ğµ 7 Ğ´Ğ½ĞµĞ¹ |
| `recent_7d_count_amount` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ·Ğ° 7 Ğ´Ğ½ĞµĞ¹ |
| `recent_7d_mean_amount` | Ğ¡Ñ€ĞµĞ´Ğ½Ñ�Ñ� Ñ�ÑƒĞ¼Ğ¼Ğ° Ğ·Ğ° 7 Ğ´Ğ½ĞµĞ¹ |
| `recent_7d_nunique_mcc_code` | Ğ£Ğ½Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ğµ MCC Ğ·Ğ° 7 Ğ´Ğ½ĞµĞ¹ |
| `recent_7d_nunique_date` | Ğ£Ğ½Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ğµ Ğ´Ğ½Ğ¸ Ñ� Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ñ�Ğ¼Ğ¸ Ğ·Ğ° 7 Ğ´Ğ½ĞµĞ¹ |
| `recent_7d_intensity` | Ğ˜Ğ½Ñ‚ĞµĞ½Ñ�Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ·Ğ° 7 Ğ´Ğ½ĞµĞ¹ |
| `recent_14d_*` | Ğ�Ğ½Ğ°Ğ»Ğ¾Ğ³Ğ¸Ñ‡Ğ½Ñ‹Ğµ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸ Ğ·Ğ° 14 Ğ´Ğ½ĞµĞ¹ |
| `recent_30d_*` | Ğ�Ğ½Ğ°Ğ»Ğ¾Ğ³Ğ¸Ñ‡Ğ½Ñ‹Ğµ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸ Ğ·Ğ° 30 Ğ´Ğ½ĞµĞ¹ |

### 5.4 Momentum Indicators
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `first_half_amount` | Ğ¡ÑƒĞ¼Ğ¼Ğ° Ğ² Ğ¿ĞµÑ€Ğ²Ğ¾Ğ¹ Ğ¿Ğ¾Ğ»Ğ¾Ğ²Ğ¸Ğ½Ğµ Ğ¿ĞµÑ€Ğ¸Ğ¾Ğ´Ğ° |
| `first_half_count` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ² Ğ¿ĞµÑ€Ğ²Ğ¾Ğ¹ Ğ¿Ğ¾Ğ»Ğ¾Ğ²Ğ¸Ğ½Ğµ |
| `first_half_days` | Ğ�ĞºÑ‚Ğ¸Ğ²Ğ½Ñ‹Ğµ Ğ´Ğ½Ğ¸ Ğ² Ğ¿ĞµÑ€Ğ²Ğ¾Ğ¹ Ğ¿Ğ¾Ğ»Ğ¾Ğ²Ğ¸Ğ½Ğµ |
| `second_half_amount` | Ğ¡ÑƒĞ¼Ğ¼Ğ° Ğ²Ğ¾ Ğ²Ñ‚Ğ¾Ñ€Ğ¾Ğ¹ Ğ¿Ğ¾Ğ»Ğ¾Ğ²Ğ¸Ğ½Ğµ Ğ¿ĞµÑ€Ğ¸Ğ¾Ğ´Ğ° |
| `second_half_count` | ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ²Ğ¾ Ğ²Ñ‚Ğ¾Ñ€Ğ¾Ğ¹ Ğ¿Ğ¾Ğ»Ğ¾Ğ²Ğ¸Ğ½Ğµ |
| `second_half_days` | Ğ�ĞºÑ‚Ğ¸Ğ²Ğ½Ñ‹Ğµ Ğ´Ğ½Ğ¸ Ğ²Ğ¾ Ğ²Ñ‚Ğ¾Ñ€Ğ¾Ğ¹ Ğ¿Ğ¾Ğ»Ğ¾Ğ²Ğ¸Ğ½Ğµ |
| `activity_momentum` | Ğ˜Ğ·Ğ¼ĞµĞ½ĞµĞ½Ğ¸Ğµ Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ¼ĞµĞ¶Ğ´Ñƒ Ğ¿Ğ¾Ğ»Ğ¾Ğ²Ğ¸Ğ½Ğ°Ğ¼Ğ¸ |
| `spending_momentum` | Ğ˜Ğ·Ğ¼ĞµĞ½ĞµĞ½Ğ¸Ğµ Ñ‚Ñ€Ğ°Ñ‚ Ğ¼ĞµĞ¶Ğ´Ñƒ Ğ¿Ğ¾Ğ»Ğ¾Ğ²Ğ¸Ğ½Ğ°Ğ¼Ğ¸ |
| `days_momentum` | Ğ˜Ğ·Ğ¼ĞµĞ½ĞµĞ½Ğ¸Ğµ Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ñ‹Ñ… Ğ´Ğ½ĞµĞ¹ |
| `overall_momentum` | Ğ�Ğ±Ñ‰Ğ¸Ğ¹ momentum Ğ°ĞºÑ‚Ğ¸Ğ²Ğ½Ğ¾Ñ�Ñ‚Ğ¸ |

### 5.5 Bankruptcy Risk Score
| ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº | Ğ�Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ¸Ğµ |
|---------|----------|
| `bankruptcy_risk_score_y` | ĞšĞ¾Ğ¼Ğ¿Ğ¾Ğ·Ğ¸Ñ‚Ğ½Ñ‹Ğ¹ Ñ�ĞºĞ¾Ñ€ Ñ€Ğ¸Ñ�ĞºĞ° Ğ¾Ğ±Ğ½ÑƒĞ»ĞµĞ½Ğ¸Ñ� Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�Ğ° |
"""

# Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ² Ñ„Ğ°Ğ¹Ğ»
with open('/kaggle/working/features_guide.md', 'w', encoding='utf-8') as f:
    f.write(features_guide)

print("Ğ¡Ğ¿Ñ€Ğ°Ğ²Ğ¾Ñ‡Ğ½Ğ¸Ğº Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½ ĞºĞ°Ğº features_guide.md")

