import polars as pl
import pandas as pd


train = pl.read_csv('/kaggle/input/solana-skill-sprint-memcoin-graduation/train.csv')
train = train.filter(pl.col('is_valid'))

test = pl.read_csv('/kaggle/input/solana-skill-sprint-memcoin-graduation/test_unlabeled.csv')
test = test.filter(pl.col('is_valid'))


pump_fun_info = pl.read_parquet('/kaggle/input/pump-fun-api-solana-tokens-info/pump_fun_api_info.parquet')


train_joined = train.join(pump_fun_info, how='left', on='mint')
test_joined = test.join(pump_fun_info, how='left', on='mint')


pd.to_datetime(train_joined['created_timestamp'].min() * 1_000_000), pd.to_datetime(train_joined['created_timestamp'].max() * 1_000_000)


pd.to_datetime(test_joined['created_timestamp'].min() * 1_000_000), pd.to_datetime(test_joined['created_timestamp'].max() * 1_000_000)


train_joined['slot_min'].min(), train_joined['slot_min'].max()


test_joined['slot_min'].min(), test_joined['slot_min'].max()


import matplotlib.pyplot as plt


plt.scatter(train_joined['created_timestamp'], train_joined['slot_min']);


plt.scatter(test_joined['created_timestamp'], test_joined['slot_min']);


import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Created timestamps from Pump Fun API
train_times = pd.Series(pd.to_datetime(train_joined['created_timestamp'].to_numpy() * 1_000_000))
test_times = pd.Series(pd.to_datetime(test_joined['created_timestamp'].to_numpy() * 1_000_000))

# Build labeled DataFrames
train_df = pd.DataFrame({'datetime': train_times, 'dataset': 'train'})
test_df = pd.DataFrame({'datetime': test_times, 'dataset': 'test'})

# Combine for plotting
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Define cutoff date
cutoff = pd.to_datetime("2025-02-01")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    # Plot: Up to cutoff
    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=combined_df[combined_df['datetime'] <= cutoff],
        x='datetime',
        hue='dataset',
        element='step',
        stat='count',
        bins=50,
        common_norm=False
    )
    plt.title('Distribution of Created Timestamps (Up to 2025-02)')
    plt.xlabel('Timestamp')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot: After cutoff
    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=combined_df[combined_df['datetime'] > cutoff],
        x='datetime',
        hue='dataset',
        element='step',
        stat='count',
        bins=50,
        common_norm=False
    )
    plt.title('Distribution of Created Timestamps (After 2025-02)')
    plt.xlabel('Timestamp')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



cutoff = pd.to_datetime("2025-02-01")
print(f'Number of train and test tokens with Pump Fun API created timestamp before {cutoff}')
combined_df[combined_df['datetime'] <= cutoff]['dataset'].value_counts()


cutoff = pd.to_datetime("2025-02-01")
print(f'Number of train and test tokens with Pump Fun API created timestamp after {cutoff}')
combined_df[combined_df['datetime'] >= cutoff]['dataset'].value_counts()


cutoff_1 = pd.to_datetime("2025-02-01")
cutoff_2 = pd.to_datetime("2025-02-17")

print(f'Number of train and test tokens with Pump Fun API created timestamp between {cutoff_1} and {cutoff_2}')
filtered_df = combined_df[(combined_df['datetime'] >= cutoff_1) & (combined_df['datetime'] < cutoff_2)]
print(filtered_df['dataset'].value_counts())


cutoff = pd.to_datetime("2025-02-17")
print(f'Number of train and test tokens with Pump Fun API created timestamp after {cutoff}')
combined_df[combined_df['datetime'] >= cutoff]['dataset'].value_counts()


with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    plt.figure(figsize=(12, 5))
    
    # Plot histogram for train
    sns.histplot(train_joined['slot_min'], bins=100, color='skyblue', label='Train', kde=False)
    
    # Plot histogram for test
    sns.histplot(test_joined['slot_min'], bins=100, color='salmon', label='Test', kde=False)
    
    plt.xlabel("slot_min")
    plt.ylabel("Count")
    plt.title("Distribution of slot_min for Train and Test")
    plt.legend()
    plt.tight_layout()
    plt.show()

