import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')


train.head(3)


train.shape


train.info()


nan_count = train.isna().sum()

nan_percent = (nan_count / len(train)) * 100

nan_summary = pd.DataFrame({
    'missing_count': nan_count,
    'missing_percent': nan_percent
}).sort_values(by='missing_percent', ascending=False)

if (nan_summary['missing_count'] > 0).any():
    print("Columns with missing values:")
    print(nan_summary[nan_summary['missing_count'] > 0])
else:
    print("âœ… No missing values (NaN) found in any column.")


print("\nðŸ“ˆ Descriptive Statistics:")
train.describe().transpose().head(5)


key_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'label']
for col in key_features:
    plt.figure(figsize=(10, 4))
    sns.histplot(train[col].dropna(), bins=100, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(15, 4))
train['label'].plot(title='Trend of Target Variable (label) Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Label')
plt.tight_layout()
plt.show()


key_to_check = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'label']

for col in key_to_check:
    q_low = train[col].quantile(0.01)
    q_high = train[col].quantile(0.99)
    outliers = train[(train[col] < q_low) | (train[col] > q_high)]
    print(f"{col}: {len(outliers)} outliers (outside 1stâ€“99th percentiles)")


key_to_check = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'label']

corr_sample = train[key_to_check].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_sample, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap (Sample of 50,000 Rows)")
plt.tight_layout()
plt.show()


x_features = [col for col in train.columns if col.startswith('X')]

print("Are there any infinite values in X features?",
      np.isinf(train[x_features]).values.any())

