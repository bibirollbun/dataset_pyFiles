import pandas as pd
import matplotlib
import matplotlib.pyplot as plt  
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")


sample_df = pd.DataFrame()
for i in range(2):
    partition = pd.read_parquet(f'/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}/part-0.parquet')
    sample_df = pd.concat([sample_df, partition])
    del partition


responder_col = [col for col in sample_df.columns if 'responder' in col]
responder_col


sample_df.info()


sample_df.describe().T


sample_df[['responder_6', 'symbol_id']].groupby('symbol_id').count()


fig, axes = plt.subplots(10, 2, figsize=(25, 50))

for i, symbol in enumerate(sample_df[['responder_6', 'symbol_id']].groupby('symbol_id').count().reset_index()['symbol_id']):
    time = sample_df[sample_df['symbol_id']==symbol]['time_id']
    target = sample_df[sample_df['symbol_id']==symbol]['responder_6']
    ax = sns.lineplot(
        data = sample_df,
        x = time, 
        y = target, 
        color='black', 
        ax = axes[i // 2, i % 2])
    ax.set_title(f'Responder 6 Over Time by Symbol {symbol}')
    ax.set(xlabel='Time', ylabel='Return')
    ax.grid(True)
    ax.axhline(0, color='red', linestyle='-', linewidth=1.2)

plt.show()


from statsmodels.tsa.stattools import adfuller
adf, pval, usedlag, nobs, crit_vals, icbest =  adfuller(sample_df[sample_df['symbol_id']==0]['responder_6'])
print('ADF test statistic:', adf)
print('ADF p-values:', pval)
print('ADF number of lags used:', usedlag)
print('ADF number of observations:', nobs)
print('ADF critical values:', crit_vals)
print('ADF best information criterion:', icbest)


fig, axes = plt.subplots(10, 2, figsize=(25, 50))

for i, symbol in enumerate(sample_df[['responder_6', 'symbol_id']].groupby('symbol_id').count().reset_index()['symbol_id']):
    time = sample_df[sample_df['symbol_id']==symbol]['date_id']
    target = sample_df[sample_df['symbol_id']==symbol]['responder_6'].cumsum()
    ax = sns.lineplot(
        data = sample_df,
        x = time, 
        y = target, 
        color='black', 
        ax = axes[i // 2, i % 2])
    ax.set_title(f'Responder 6 Over Time by Symbol {symbol}')
    ax.set(xlabel='Time', ylabel='Cumulative Return')
    ax.grid(True)
    ax.axhline(0, color='red', linestyle='-', linewidth=1.2)

plt.show()


sample_df['pct_change'] = sample_df.groupby(['time_id', 'symbol_id'])['responder_6'].pct_change()


sample_df['pct_change'] = sample_df.groupby(['time_id', 'symbol_id'])['responder_6'].pct_change()
fig, axes = plt.subplots(10, 2, figsize=(25, 100)) 
axes = axes.flatten()  

for i, symbol in enumerate(sample_df[['responder_6', 'symbol_id']].groupby('symbol_id').count().reset_index()['symbol_id']):
    ax = axes[i] 
    values, bins, bars = ax.hist(sample_df[(sample_df['symbol_id']==0) & (sample_df['pct_change'].between(-50,50))]['pct_change'], alpha=0.7, bins = 30)
    ax.set_xlabel('Percentage Change (%)Daily Returns')
    ax.set_title(f'Distribution of Daily Returns of Symbol {symbol}', weight='bold')
    ax.bar_label(bars)
    ax.grid(visible=True, color='gray', linewidth=0.7)

plt.show()


fig, axes = plt.subplots(10, 2, figsize=(25, 90))
axes = axes.flatten()  # Flatten the axes for easier indexing

for i, symbol in enumerate(sample_df[['responder_6', 'symbol_id']].groupby('symbol_id').count().reset_index()['symbol_id']):
    # Filter data for the current symbol
    symbol_data = sample_df[sample_df['symbol_id'] == symbol]['responder_6'].dropna()

    # Calculate IQR and extreme outlier bounds
    q1 = symbol_data.quantile(0.25)
    q3 = symbol_data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr  # Extreme lower outlier bound
    upper_bound = q3 + 3 * iqr  # Extreme upper outlier bound

    # Plot boxplot
    ax = axes[i]
    sns.boxplot(y=symbol_data, color='skyblue', ax=ax)  # Use x= for Series data
    ax.set_title(f'Distribution of Responder 6 by Symbol {symbol}', weight='bold')

    # Add red lines for extreme outlier boundaries
    ax.axhline(lower_bound, color='red', linestyle='--', linewidth=1, label='Extreme Outlier Lower Bound')
    ax.axhline(upper_bound, color='red', linestyle='--', linewidth=1, label='Extreme Outlier Upper Bound')

    # Add legend only once per plot
    ax.legend()

    # Enable grid for better readability
    ax.grid(True)

plt.tight_layout()
plt.show()



from statsmodels.tsa.seasonal import STL
for i, symbol in enumerate(sample_df[['responder_6', 'symbol_id']].groupby('symbol_id').count().reset_index()['symbol_id']):
    stl = STL(sample_df[sample_df['symbol_id']==symbol]['responder_6'], period=12, seasonal=13)
    result = stl.fit()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    ax1.plot(sample_df[sample_df['symbol_id']==symbol]['responder_6'])
    ax1.set_title(f'Original Responder 6 for Symbol {symbol}')
    ax2.plot(result.trend)
    ax2.set_title(f'Trend Component for Symbol {symbol}')
    ax3.plot(result.seasonal)
    ax3.set_title(f'Seasonal Component for Symbol {symbol}')
    ax4.plot(result.resid)
    ax4.set_title(f'Residual (Noise) Component for Symbol {symbol}')
    
    plt.tight_layout()
    plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose
for i, symbol in enumerate(sample_df[['responder_6', 'symbol_id']].groupby('symbol_id').count().reset_index()['symbol_id']):
    responder_6_decomp = seasonal_decompose(sample_df[sample_df['symbol_id']==symbol]['responder_6'], model='additive', period=251)
    fig = responder_6_decomp.plot()  # Plot the decomposition
    fig.suptitle(f'Time Series Decomposition for Symbol {symbol}', fontsize=12)  
    plt.tight_layout()
    plt.show()


stat_df = sample_df.copy()
fig, axes = plt.subplots(10, 2, figsize=(25, 50)) 
axes = axes.flatten() 

for i, symbol in enumerate(sample_df[['responder_6', 'symbol_id']].groupby('symbol_id').count().reset_index()['symbol_id']):
    symbol_data = stat_df[stat_df['symbol_id'] == symbol]
    ax = axes[i]
    ax.plot(symbol_data['responder_6'].rolling(window=20).mean(), alpha = 0.5, label=f'20-day Moving Average for Symbol {symbol}')
    ax.plot(symbol_data['responder_6'].rolling(window=100).mean(), alpha = 0.9, label=f'100-day Moving Average for Symbol {symbol}')
    ax.set_title(f'Moving Average of Responder 6 for Symbol {symbol}', weight='bold')
    ax.grid(visible=True, color='gray', linewidth=0.7)
    ax.legend()

plt.tight_layout()
plt.show()


stat_df = sample_df.copy()
fig, axes = plt.subplots(10, 2, figsize=(25, 50)) 
axes = axes.flatten() 

for i, symbol in enumerate(sample_df[['responder_6', 'symbol_id']].groupby('symbol_id').count().reset_index()['symbol_id']):
    symbol_data = stat_df[stat_df['symbol_id'] == symbol]
    ax = axes[i]
    ax.plot(symbol_data['responder_6'].rolling(window=20).std(), alpha = 0.5, label=f'20-day Moving Average for Symbol {symbol}')
    ax.plot(symbol_data['responder_6'].rolling(window=100).std(), alpha = 0.9, label=f'100-day Moving Average for Symbol {symbol}')
    ax.set_title(f'Moving Standard Deviation of Responder 6 for Symbol {symbol}', weight='bold')
    ax.grid(visible=True, color='gray', linewidth=0.7)
    ax.legend()

plt.tight_layout()
plt.show()


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

for i, symbol in enumerate(sample_df[['responder_6', 'symbol_id']].groupby('symbol_id').count().reset_index()['symbol_id']):
    fig = plot_acf(sample_df[sample_df['symbol_id']==symbol]['responder_6'].dropna(), lags=100)
    fig.suptitle(f'Autocorrelation Analysis for Symbol {symbol}', fontsize=12)  
    plt.tight_layout()
    plt.show()


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

for i, symbol in enumerate(sample_df[['responder_6', 'symbol_id']].groupby('symbol_id').count().reset_index()['symbol_id']):
    fig = plot_pacf(sample_df[sample_df['symbol_id']==symbol]['responder_6'].dropna())
    fig.suptitle(f'Partial Autocorrelation Analysis for Symbol {symbol}', fontsize=12)  
    plt.tight_layout()
    plt.show()


fig, axes = plt.subplots(9, 1, figsize=(25, 90)) 
axes = axes.flatten()  

for i, responder in enumerate(responder_col): 
    ax = axes[i] 
    
    if responder != 'responder_6':  
        c = 'red'
        lw = 2.5
        ax.plot(
            (sample_df[sample_df.symbol_id == 0].groupby(['date_id'])['responder_6'].mean()).cumsum(), 
            linewidth=lw, color=c, label='Responder 6'
        )
        
        # Plot the other responder
        lw = 1
        ax.plot(
            (sample_df[sample_df.symbol_id == 0].groupby(['date_id'])[responder].mean()).cumsum(), 
            linewidth=lw, label=responder
        )
        
        # Subplot settings
        ax.set_xlabel('Trade days')
        ax.set_ylabel('Cumulative response')
        ax.set_title(f'Response time series over trade days  \n Responder 6 (red) and {responder}', weight='bold')
        ax.grid(visible=True, color='gray', linewidth=0.7)
        ax.axhline(0, color='red', linestyle='-', linewidth=1)
        ax.legend()
    else:
        ax.plot(
        (sample_df[sample_df.symbol_id == 0].groupby(['date_id'])['responder_6'].mean()).cumsum(), 
        linewidth=lw, color=c, label='Responder 6'
        )
        ax.set_xlabel('Trade days')
        ax.set_ylabel('Cumulative response')
        ax.set_title('Response time series over trade days  \n Responder 6 (red)', weight='bold')
        ax.grid(visible=True, color='gray', linewidth=0.7)
        ax.axhline(0, color='red', linestyle='-', linewidth=1)
        ax.legend()
        
        
# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


plt.figure(figsize=(6, 6))
path = "/kaggle/input/jane-street-real-time-market-data-forecasting"
responders = pd.read_csv(f"{path}/responders.csv")
matrix = responders[[ f"tag_{no}" for no in range(0,5,1) ] ].T.corr()
sns.heatmap(matrix, square=True, cmap="coolwarm", alpha =0.9, vmin=-1, vmax=1, center= 0, linewidths=0.5, 
            linecolor='white', annot=True, fmt='.2f')
plt.xlabel("Responder_0 - Responder_8")
plt.ylabel("Responder_0 - Responder_8")
plt.show()


sample_df


sample_df
gridColor = 'lightgrey'
row = len(responder_col)
j = 0

fig, axs = plt.subplots(figsize=(18, 4*row))
for i in range(1, 3 * len(responder_col) + 1, 3):
    xx=sample_df[(sample_df.symbol_id==0)]['date_id']
    yy=sample_df[ (sample_df.symbol_id==0)][f'responder_{j}']
    c='black'
    if j == 6: c='red'
        
    ax1 = plt.subplot(9, 3, i)
    ax1.plot(xx,yy.cumsum(), color = c, linewidth =0.8 )
    plt.axhline(0, color='blue', linestyle='-', linewidth=0.9)
    plt.grid(color =gridColor )
    
    ax2 = plt.subplot(9, 3, i+1)
    ax2.plot(xx,yy, color = c, linewidth =0.05)
    plt.axhline(0, color='blue', linestyle='-', linewidth=1.2)
    ax2.set_title(f"responder_{j}", fontsize = 14)
    plt.grid(color = gridColor)
    
    ax3 = plt.subplot(9, 3, i+2)
    b=1000
    ax3.hist(yy, bins=b, color = c,density=True, histtype="step" )
    ax3.hist(yy, bins=b, color = 'lightgrey',density=True)
    plt.grid(color = gridColor)
    ax3.set_ylim([0, 3.5])
    ax3.set_xlim([-2.5, 2.5])
    
    j = j + 1
    
fig.patch.set_linewidth(3)
fig.patch.set_edgecolor('#000000')
fig.patch.set_facecolor('#eeeeee') 
plt.show()


dts = ['date_id', 'time_id', 'symbol_id']
features = [f'feature_{i:02}' for i in range(79)]
responders = [f'responder_{i:1}' for i in range(9)]

X = sample_df[features]
weights = sample_df['weight']
y = sample_df['responder_6']


train_size = int(len(X) * 0.8)

# Sequential split
train_data = sample_df[:train_size]
val_data = sample_df[train_size:]
X_train = X[:train_size]
X_val = X[train_size:]
y_train = y[:train_size]
y_val = y[train_size:]
weights_train = weights[:train_size]
weights_val = weights[train_size:]

print(f"Train shapes: {X_train.shape}, {y_train.shape}, {weights_train.shape}")
print(f"Validation shapes: {X_val.shape}, {y_val.shape}, {weights_val.shape}")


from sklearn.metrics import mean_squared_error, r2_score


from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import AutoReg


# Create and train the autoregressive model
lag_order = 20 # Adjust this based on the ACF plot
ar_model = AutoReg(y_train, lags=lag_order)
ar_results = ar_model.fit()
print(ar_results.summary())


y_pred_ar = ar_results.predict(start=len(train_data), end=len(train_data) + len(val_data) - 1, dynamic=False)
print(y_pred_ar)


mse = mean_squared_error(y_val, y_pred_ar, squared=False)
r2 = r2_score(y_val, y_pred_ar)
print(f"RMSE: {mse}")
print(f"R²: {r2}")


stat_df = train_data.copy()
fig, axes = plt.subplots(10, 2, figsize=(25, 50)) 
axes = axes.flatten() 

for i, symbol in enumerate(train_data[['responder_6', 'symbol_id']].groupby('symbol_id').count().reset_index()['symbol_id']):
    symbol_data = stat_df[stat_df['symbol_id'] == symbol]
    ax = axes[i]
    ax.plot(symbol_data['responder_6'], label=f'Responder 6 for Symbol {symbol}')
    ax.plot(symbol_data['responder_6'].rolling(window=20).mean(), alpha = 0.9, label=f'20-day Moving Average for Symbol {symbol}')
    ax.set_title(f'Moving Average of Responder 6 for Symbol {symbol}', weight='bold')
    ax.grid(visible=True, color='gray', linewidth=0.7)
    ax.legend()

plt.tight_layout()
plt.show()


from statsmodels.tsa.arima.model import ARIMA

# Create and train the moving average model
ma_model = ARIMA(train_data['responder_6'], order=(00, 0, 20))
ma_results = ma_model.fit()

print(ma_results.summary())


y_pred_ma = ma_results.predict(start=len(train_data), end=len(train_data) + len(val_data) - 1, dynamic=False)
print(y_pred_ma)


mse = mean_squared_error(y_val, y_pred_ma, squared=False)
r2 = r2_score(y_val, y_pred_ma)
print(f"RMSE: {mse}")
print(f"R²: {r2}")


from statsmodels.tsa.arima.model import ARIMA

# Create and train the moving average model
arma_model = ARIMA(endog=y_train, order=(20, 0, 20))
arma_results = arma_model.fit()

print(arma_results.summary())


y_pred_arma = arma_results.predict(start=len(train_data), end=len(train_data) + len(val_data) - 1, dynamic=False)
print(y_pred_arma)


import pmdarima as pm
from pmdarima.model_selection import train_test_split


train, test = train_test_split(sample_df['responder_6'], train_size = int(len(X) * 0.8))


model = pm.auto_arima(train, seasonal=True, m=52)
preds = model.predict(test.shape[0])


plt.plot(sample_df['responder_6'][:train_size], train)
plt.plot(sample_df['responder_6'][train_size:], preds)
plt.show()




