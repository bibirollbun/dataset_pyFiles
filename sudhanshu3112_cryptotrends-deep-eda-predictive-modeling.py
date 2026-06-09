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


df_train = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
df_test = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")


submission = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")




df_train.shape


df_train.head(5)


df_test.shape


df_test.head()


print("\nTrain columns and data types:")
print(df_train.dtypes)

print("\nTest columns and data types:")
print(df_test.dtypes)


#  Check for missing values
print("\nMissing values in train data:")
print(df_train.isnull().sum())

print("\nMissing values in test data:")
print(df_test.isnull().sum())


import warnings
warnings.filterwarnings('ignore')


# statistics summary
print("\nTrain data statistics:")
print(df_train.describe())

print("\nTest data statistics:")
print(df_test.describe())


import matplotlib.pyplot as plt
import seaborn as sns


public_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(public_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df_train[feature], bins=50, kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()



plt.figure(figsize=(6,4))
sns.histplot(df_train['label'], bins=100, kde=True)
plt.title('Distribution of Target (label)')
plt.show()



plt.figure(figsize=(15, 10))
for i, feature in enumerate(public_features, 1):
    plt.subplot(2, 3, i)
    plt.hexbin(df_train[feature], df_train['label'], gridsize=50, cmap='Blues')
    plt.xlabel(feature)
    plt.ylabel('label')
    plt.title(f'{feature} vs label')
plt.tight_layout()
plt.show()



corr_matrix = df_train[public_features + ['label']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()




plt.figure(figsize=(15,5))
df_train['label'].plot()
plt.title('Target (label) over Time')
plt.xlabel('Timestamp')
plt.ylabel('label')
plt.show()



rolling_window = 60  # e.g., 60 minutes
plt.figure(figsize=(15,5))
df_train['label'].rolling(window=rolling_window).mean().plot(label='Rolling Mean')
df_train['label'].rolling(window=rolling_window).std().plot(label='Rolling Std')
plt.legend()
plt.title(f'Rolling Mean and Std of label (window={rolling_window})')
plt.show()



plt.figure(figsize=(15,10))
for i, feature in enumerate(public_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=df_train[feature])
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()


from scipy.stats import zscore

z_scores = zscore(df_train['volume'])
outliers = df_train[np.abs(z_scores) > 3]
print(f"Number of outliers in volume: {len(outliers)}")



import numpy as np
from sklearn.preprocessing import RobustScaler

# Log-transform volume to reduce skewness
df_train['volume_log'] = np.log1p(df_train['volume'])

# Create outlier flag based on z-score
from scipy.stats import zscore
z_scores = zscore(df_train['volume'])
df_train['volume_outlier'] = (np.abs(z_scores) > 3).astype(int)

# Use RobustScaler for scaling
scaler = RobustScaler()
df_train['volume_scaled'] = scaler.fit_transform(df_train[['volume']])



for col in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']:
    df_train[f'{col}_log'] = np.log1p(df_train[col])


# Scale the log-transformed features:
scaler = RobustScaler()
for col in ['bid_qty_log', 'ask_qty_log', 'buy_qty_log', 'sell_qty_log', 'volume_log']:
    df_train[f'{col}_scaled'] = scaler.fit_transform(df_train[[col]])


# Log-transform

df_test['volume_log'] = np.log1p(df_test['volume'])

# Step 2: Create outlier flag (same as before)
test_z_scores = (df_test['volume'] - df_train['volume'].mean()) / df_train['volume'].std()
df_test['volume_outlier'] = (np.abs(test_z_scores) > 3).astype(int)

# Step 3: Apply scaler on the log-transformed volume (volume_log)
df_test['volume_scaled'] = scaler.transform(df_test[['volume_log']])




window_sizes = [5, 15, 60]  # Example windows in minutes

for feature in ['volume', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty']:
    for window in window_sizes:
        df_train[f'{feature}_rollmean_{window}'] = df_train[feature].rolling(window).mean()
        df_train[f'{feature}_rollstd_{window}'] = df_train[feature].rolling(window).std()
        df_train[f'{feature}_lag_{window}'] = df_train[feature].shift(window)



df_train['label_rollstd_60'] = df_train['label'].rolling(60).std()



feature_cols = [col for col in df_train.columns if col not in ['label', 'timestamp', 'ID']]

# Filter to columns existing in both train and test
feature_cols = [col for col in feature_cols if col in df_test.columns]

X = df_train[feature_cols]
y = df_train['label']
X_test = df_test[feature_cols]




split_idx = int(len(df_train) * 0.8)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]



import lightgbm as lgb
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)



from scipy.stats import pearsonr
y_pred = model.predict(X_val)
corr, _ = pearsonr(y_val, y_pred)
print("Validation Pearson correlation:", corr)



model.fit(X, y)



test_preds = model.predict(X_test)



submission['label'] = test_preds
submission.to_csv('submission.csv', index=False)





