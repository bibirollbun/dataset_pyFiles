import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# tpu = tf.distribute.cluster_resolver.TPUClusterResolver('TPU VM v3-8')
# tf.tpu.experimental.initialize_tpu_system(tpu)
# tpu_strategy = tf.distribute.TPUStrategy(tpu)


data = pd.read_parquet(r'/kaggle/input/drw-crypto-market-prediction/train.parquet')
data.head()


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index,data['label'])
ax.set_title('label trending')
plt.tight_layout()
plt.show()


fig1, ax = plt.subplots(figsize=(18, 10))
sns.histplot(data['label'], kde=True, bins=50)
ax.set_xlabel('Frequency')
ax.set_ylabel('label value')
ax.set_title('label trending')
plt.tight_layout()
plt.show()


features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'X1', 'X2', 'X3']

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
for i, feature in enumerate(features):
    sns.scatterplot(x=data[feature], y=data['label'], ax=axes[i//4, i%4])
plt.tight_layout()
plt.show()


correlation_matrix = data[features + ['label']].corr()
print(correlation_matrix['label'].sort_values(ascending=False))
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation with Label')
plt.show()


window_size = 10  # 分钟
data['rolling_volume'] = data['volume'].rolling(window=window_size).mean()
data['rolling_label_std'] = data['label'].rolling(window=window_size).std()

fig, ax1 = plt.subplots(figsize=(18, 10))
ax1.plot(data.index, data['rolling_volume'], color='blue', label='Rolling Volume')
ax1.set_xlabel('Time')
ax1.set_ylabel('Rolling Volume', color='blue')

ax2 = ax1.twinx()
ax2.plot(data.index, data['rolling_label_std'], color='red', label='Label Volatility')
ax2.set_ylabel('Label Volatility', color='red')

plt.title('Volume vs Label Volatility')
fig.legend(loc="upper right")
plt.tight_layout()
plt.show()


data['hour'] = data.index.hour
hourly_stats = data.groupby('hour')['label'].agg(['mean', 'std']).reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(hourly_stats['hour'], hourly_stats['mean'], yerr=hourly_stats['std'], capsize=5)
plt.title('Hourly Average Label with Std Dev')
plt.xlabel('Hour of Day')
plt.ylabel('Average Label')
plt.tight_layout()
plt.show()


data['minute'] = data.index.minute
minute_volatility = data.groupby('minute')['label'].std().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='minute', y='label', data=minute_volatility)
plt.title('Minute-wise Label Volatility')
plt.xlabel('Minute of Hour')
plt.ylabel('Label Volatility')
plt.tight_layout()
plt.show()

