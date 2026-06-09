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


df = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')


df


df['label'] = df['prediction'].apply(lambda x:1 if x>0 else 0)


df


print(df[['prediction', 'label']].describe())


from scipy.stats import pearsonr
import matplotlib.pyplot as plt 
import seaborn as sns


corr,p_value = pearsonr(df['prediction'], df['label'])
print(f"Pearson Correlation: {corr:.4f}")
print(f"P-value : {p_value:.4e}")


plt.figure(figsize=(10 , 6))
sns.scatterplot(x='label', y='prediction', data=df, alpha=0.3)
sns.regplot(x='label', y='prediction', data=df, scatter=False, color='red')
plt.title(f'predictions vs labels (pearson r = {corr:.4f})')
plt.xlabel('True Label')
plt.ylabel('Prediction')
plt.grid(True)
plt.tight_layout()
plt.show()


df['SMA_20'] = df['prediction'].rolling(window=20).mean()
df['SMA_50'] = df['prediction'].rolling(window=50).mean()


df


delta = df['prediction'].diff()


delta


gain = delta.where(delta > 0, 0)
lose = delta.where(delta < 0, 0)


avg_gain = gain.rolling(window=14).mean()
avg_lose = lose.rolling(window=14).mean()


rs = avg_gain / avg_lose


rs


df


df['RSI'] = 100 - (100/(1+rs))


df


df['EMA_20'] = df['prediction'].ewm(span=20, adjust=False).mean()


df


features = ['SMA_20', 'SMA_50', 'RSI', 'EMA_20']
X = df[features].dropna()


X


y = df['prediction'].shift(-1).dropna()


y


X = X.iloc[:-1]
y = y.iloc[:-1]


X


y


from sklearn.model_selection import train_test_split


print(X.shape)


print(y.shape)


min_len = min(len(X), len(y))


min_len


X = X.iloc[:min_len].reset_index(drop=True)
y = y.iloc[:min_len].reset_index(drop=True)


X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)


model.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error


y_pred = model.predict(X_test)


mae = mean_squared_error(y_test, y_pred)
rsme = np.sqrt(mean_squared_error(y_test, y_pred))


print(f'MAE: {mae}')
print(f'RSME: {rsme}')


df['Predicted_RandomForestRegressor']= model.predict(df[features].fillna(0))


df


import seaborn as sns 
import matplotlib.pyplot as plt 


sns.histplot(df['prediction'], bins=100)
plt.title('Predictions Distribution')
plt.show()


y = df['prediction'].shift(-1).dropna()


y


plt.figure(figsize=(10, 6))
sns.scatterplot(x='prediction', y='Predicted_RandomForestRegressor', data=df, alpha=0.4)
plt.title('Actual Predictions vs RandomForestRegressor')
plt.xlabel('Raw Prediction')
plt.ylabel('Raw RandomForestRegressor')
plt.axhline(0 , color='red', linestyle='--')
plt.axvline(0 , color='Blue', linestyle='--')
plt.show()



plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlations Matrix')
plt.show()


plt.figure(figsize=(14, 6))
df['prediction'].plot(label='Prediction', alpha=0.7)
df['SMA_20'].plot(label='SMA 20')
df['SMA_50'].plot(label='SMA 50')
df['EMA_20'].plot(label='EWA_20')
plt.legend()
plt.title('Predictions VS SMA/EMA')
plt.show()


df['Target_class'] = (df['prediction']>0).astype(int)


df


df['residual'] = df['prediction'] - df['Predicted_RandomForestRegressor']


df


plt.figure(figsize=(10, 5))
sns.histplot(df['residual'], bins=100, kde=True)
plt.title('Prediction Error (Residual) Distribution')
plt.show()


sample = df.iloc[-500:]


sample


plt.figure(figsize=(14, 6))
plt.plot(sample['prediction'], label='Actual Prediction')
plt.plot(sample['Predicted_RandomForestRegressor'], label='RandomForest')
plt.title('Actual predictions vs Predicted (last Observation)')
plt.legend()
plt.show()

