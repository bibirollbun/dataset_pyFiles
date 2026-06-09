import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns



train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

print(f"Train: {train.shape} | Test: {test.shape}")


print("\nğŸ“Š BPM Stats:")
print(f"Mean: {train['BeatsPerMinute'].mean():.1f}")
print(f"Range: {train['BeatsPerMinute'].min():.1f} - {train['BeatsPerMinute'].max():.1f}")


correlations = train.corr()['BeatsPerMinute'].drop('BeatsPerMinute').sort_values(key=abs, ascending=False)

print("\nğŸ”— Top Correlations with BPM:")
for feature, corr in correlations.head(5).items():
    print(f"{feature}: {corr:.3f}")


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

###BPM distribution
train['BeatsPerMinute'].hist(bins=30, ax=axes[0])
axes[0].set_title('BPM Distribution')

###Top correlation
top_feature = correlations.index[0]
axes[1].scatter(train[top_feature], train['BeatsPerMinute'], alpha=0.5)
axes[1].set_xlabel(top_feature)
axes[1].set_ylabel('BPM')
axes[1].set_title(f'{top_feature} vs BPM')

###Correlation heatmap (top 5 features only)
top_features = correlations.head(5).index.tolist() + ['BeatsPerMinute']
sns.heatmap(train[top_features].corr(), annot=True, cmap='coolwarm', center=0, ax=axes[2])
axes[2].set_title('Top Features Correlation')

plt.tight_layout()
plt.show()


print("\nğŸ’¡ Quick Insights:")
print(f"â€¢ {top_feature} has strongest correlation ({correlations.iloc[0]:.3f})")
print(f"â€¢ Energy range: {train['Energy'].min():.2f} to {train['Energy'].max():.2f}")
print(f"â€¢ Track duration: {train['TrackDurationMs'].mean()/1000/60:.1f} min average")


print(f"\nâ�Œ Missing values: {train.isnull().sum().sum()} total")




