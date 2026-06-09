#importing relevant libraries

import pandas as pd
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path


data_path = Path("/kaggle/input/jane-street-real-time-market-data-forecasting")

features = pd.read_csv(data_path/'features.csv')
features.head()


responders = pd.read_csv(data_path/'responders.csv')
responders.head()


train_file = pd.read_parquet(data_path/'train.parquet'/'partition_id=0'/'part-0.parquet')
train_file.head()


import missingno
feature_columns = [f"feature_{feature_num:02d}" for feature_num in range(79)]
missingno.matrix(train_file[feature_columns[0:50]].sample(1000))
missingno.matrix(train_file[feature_columns[50:]].sample(1000))
plt.show()


missingno.bar(train_file[feature_columns])


missingno.heatmap(train_file[feature_columns])


missingno.dendrogram(train_file[feature_columns])


feature_correlation = train_file[feature_columns].corr().fillna(0)
plt.figure(figsize = (15, 15))
sns.heatmap(feature_correlation, square = True, cmap = 'coolwarm')
plt.show()


from scipy.cluster.hierarchy import dendrogram, linkage

distance_matrix = 1 - np.abs(feature_correlation)
linkage_matrix = linkage(distance_matrix, method = 'ward')

plt.figure(figsize=(20, 10))
dendrogram(
    linkage_matrix,
    labels=feature_correlation.columns,
    leaf_rotation=90,
    leaf_font_size=12
)
plt.title("Dendrogram of Feature Correlations")
plt.xlabel("Features")
plt.ylabel("Distance")
plt.show()


plt.figure(figsize = (20, 20))
sns.clustermap(feature_correlation, cmap='coolwarm', annot=False, method='ward')
plt.title("Heatmap with Dendrogram")
plt.show()



plt.figure(figsize = (15, 10))
sns.boxplot(train_file, x = 'symbol_id', y = 'responder_6', whis = 2.0, showfliers = False)
plt.title('Distribution of the Different Stock Symbols')
plt.show()


import warnings
warnings.simplefilter("ignore")

plt.figure(figsize = (15, 10))
sns.histplot(train_file,y = 'responder_6', kde = True, hue = 'symbol_id', color = 'grey')
plt.title('Distribution of the Different Stock Symbols')
plt.show()


plt.figure(figsize = (15, 10))
sns.histplot(train_file,y = 'responder_6', kde = True, hue = 'symbol_id', stat = 'probability')
plt.title('Distribution of the Different Stock Symbols')
plt.show()


import warnings
warnings.simplefilter('ignore')

plt.figure(figsize = (15, 10))
sns.histplot(train_file,y = 'responder_6', kde = True, hue = 'symbol_id', stat = 'probability', weights = 'weight')
plt.title('Distribution of the Different Stock Symbols')
plt.show()


plt.figure(figsize = (15, 10))
sns.histplot(train_file,y = 'responder_6', kde = True, stat = 'probability', weights = 'weight')
plt.title('Distribution of the Different Stock Symbols')
plt.show()


responder_column = [f'responder_{response_num}' for response_num in range(8)]
responder_corr = train_file[responder_column].corr()

plt.figure(figsize = (10,10))
sns.heatmap(responder_corr, annot = True, cmap = 'jet')
plt.show()


symbol_pivot = train_file.pivot(index = ['date_id','time_id'], columns = ['symbol_id'], values = ['responder_6']).reset_index()
symbol_pivot.columns = [f"{first}_{second}" for first, second in symbol_pivot.columns]
symbol_columns = [f"responder_6_{symbol}" for symbol in train_file['symbol_id'].unique()]
symbol_corr = symbol_pivot[symbol_columns].fillna(0).corr()
plt.figure(figsize = (20, 20))
sns.heatmap(symbol_corr, cmap = 'jet', annot=True)
plt.title('Symbol Wise Corelation for Responder 6')
plt.legend()



plt.figure(figsize = (5,10))
sns.histplot(train_file, y = 'weight')
plt.show()


plt.figure(figsize = (5,10))
sns.histplot(train_file, y = 'weight', stat = 'probability')
plt.show()


daily_avg = train_file.groupby(['date_id','symbol_id'])['responder_6'].mean().reset_index()

for symbol_id in train_file['symbol_id'].unique():
    symbol_data = daily_avg[daily_avg['symbol_id']==symbol_id]
    plt.figure(figsize=(18,6))
    plt.plot(symbol_data['date_id'], symbol_data['responder_6'], label = f"Symbol_{symbol_id}")
    plt.title(f"Time Series for Symbol {symbol_id}")
    plt.xlabel("date_id")
    plt.ylabel("responder_6")
    plt.legend()
    plt.show()


from statsmodels.tsa.seasonal import STL

stl = STL(daily_avg['responder_6'], seasonal = 31, period = 7)
result = stl.fit()
fig = result.plot()
fig.set_size_inches(20,8)
plt.tight_layout()
plt.show()


from scipy.fft import fft, fftfreq

# Compute Fourier transform
symbol_zero = daily_avg[daily_avg['symbol_id']==0]
yf = fft(symbol_zero["responder_6"].values)
xf = fftfreq(len(symbol_zero), 1)  # Assume unit frequency

# Plot frequency spectrum
plt.figure(figsize = (20,8))
plt.plot(xf, np.abs(yf))
plt.title("Frequency Spectrum")
plt.xticks(np.arange(min(xf), max(xf), step = xf.std()/5))
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.show()




