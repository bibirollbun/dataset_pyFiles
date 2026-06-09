import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import os


test_path = "/kaggle/input/stock-price-prediction-challenge/test"
indices_path = "/kaggle/input/stock-price-prediction-challenge/train/indices"

test_dfs = []
for i in range(1, 6):
    df = pd.read_csv(os.path.join(test_path, f"test_{i}.csv"))
    df = df.rename(columns={
        "Returns": f"return_{i}",
        "Close": f"close_{i}",
        "Volume": f"volume_{i}"
    })
    test_dfs.append(df)

merged_df = test_dfs[0][['Date', f'return_1', f'close_1', f'volume_1']]
for i in range(1, 5):
    df_i = test_dfs[i][['Date', f'return_{i+1}', f'close_{i+1}', f'volume_{i+1}']]
    merged_df = pd.merge(merged_df, df_i, on='Date', how='outer')

indices = {
    "dj": "Dow_Jones.csv",
    "nasdaq": "NASDAQ.csv",
    "SP500": "SP500.csv"
}

for key, filename in indices.items():
    index_df = pd.read_csv(os.path.join(indices_path, filename))
    index_df = index_df.rename(columns={"Returns": f"returns_{key}"})
    merged_df = pd.merge(merged_df, index_df[['Date', f'returns_{key}']], on='Date', how='left')

merged_df['Date'] = pd.to_datetime(merged_df['Date'])
merged_df = merged_df.sort_values('Date').reset_index(drop=True)

print(merged_df.head())


window = 10

for i in range(1, 6):
    merged_df[f'ma10_{i}'] = merged_df[f'close_{i}'].rolling(window=10).mean()
    merged_df[f'ma20_{i}'] = merged_df[f'close_{i}'].rolling(window=20).mean()
    
    merged_df[f'envelope_upper_{i}'] = merged_df[f'ma10_{i}'] * 1.05
    merged_df[f'envelope_lower_{i}'] = merged_df[f'ma10_{i}'] * 0.95

    merged_df[f'roc10_{i}'] = merged_df[f'close_{i}'].pct_change(periods=10)

    for index_key in ['dj', 'nasdaq', 'SP500']:
        returns_index_col = f'returns_{index_key}'
        
        rolling_cov = merged_df[[f'return_{i}', returns_index_col]].rolling(window=window).cov()
        
        cov = rolling_cov.loc[
            rolling_cov.index.get_level_values(1) == returns_index_col,
            f'return_{i}'
        ].reset_index(drop=True)

        var = merged_df[returns_index_col].rolling(window=window).var()
        
        merged_df[f'beta_{i}_{index_key}'] = cov / var



import matplotlib.pyplot as plt

fig, axes = plt.subplots(5, 4, figsize=(20, 20), sharex=True)

for i in range(1, 6):
    axes[i-1, 0].plot(merged_df['Date'], merged_df[f'close_{i}'], label='Close', color='blue')
    axes[i-1, 0].set_title(f'Stock {i} — Close')
    axes[i-1, 0].grid()

    axes[i-1, 1].plot(merged_df['Date'], merged_df[f'close_{i}'], label='Close', alpha=0.5)
    axes[i-1, 1].plot(merged_df['Date'], merged_df[f'ma10_{i}'], label='MA10', color='orange')
    axes[i-1, 1].plot(merged_df['Date'], merged_df[f'envelope_upper_{i}'], '--', color='green', label='Envelope +5%')
    axes[i-1, 1].plot(merged_df['Date'], merged_df[f'envelope_lower_{i}'], '--', color='red', label='Envelope -5%')
    axes[i-1, 1].set_title(f'Stock {i} — MA10 + Envelopes')
    axes[i-1, 1].legend()
    axes[i-1, 1].grid()

    axes[i-1, 2].plot(merged_df['Date'], merged_df[f'roc10_{i}'], label='ROC10', color='purple')
    axes[i-1, 2].set_title(f'Stock {i} — ROC (10 days)')
    axes[i-1, 2].grid()

    axes[i-1, 3].plot(merged_df['Date'], merged_df[f'beta_{i}_nasdaq'], label='Beta', color='black')
    axes[i-1, 3].set_title(f'Stock {i} — Beta to NASDAQ')
    axes[i-1, 3].axhline(1, color='gray', linestyle='--', linewidth=0.8)
    axes[i-1, 3].grid()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



corr_columns = [f'return_{i}' for i in range(1, 6)] + ['returns_dj', 'returns_nasdaq', 'returns_SP500']

corr_df = merged_df[corr_columns].dropna()

corr_matrix = corr_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.show()


plt.figure(figsize=(14, 10))

for i in range(1, 6):
    plt.subplot(5, 1, i)
    plt.plot(merged_df['Date'], merged_df[f'return_{i}'], label=f'Return {i}', color='tab:blue')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.ylabel('Return')
    plt.title(f'Return {i}')
    plt.grid(True)
    if i == 5:
        plt.xlabel('Date')

plt.tight_layout()
plt.show()



plt.figure(figsize=(14, 10))

for i in range(1, 6):
    plt.subplot(3, 2, i)
    sns.histplot(merged_df[f'return_{i}'].dropna(), kde=True, bins=50, color='steelblue')
    plt.title(f'Distribution of Return {i}')
    plt.xlabel('Return')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


scaler = MinMaxScaler(feature_range=(0, 1))

def prepare_data(series, time_step=100):
    X, y = [], []
    for i in range(time_step, len(series) - 10):
        X.append(series[i - time_step:i, 0])
        y.append(series[i:i + 10, 0])
    return np.array(X), np.array(y)

returns = [f'return_{i}' for i in range(1, 6)]
predictions = {}

for return_col in returns:
    series = merged_df[return_col].dropna().values.reshape(-1, 1)
    series = scaler.fit_transform(series)

    X, y = prepare_data(series)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=10)) 

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    predicted_returns = model.predict(X_test[-1].reshape(1, X_test.shape[1], 1))
    predicted_returns = scaler.inverse_transform(predicted_returns)  

    predictions[return_col] = predicted_returns.flatten()

plt.figure(figsize=(14, 10))

for i, return_col in enumerate(returns, start=1):
    plt.subplot(3, 2, i)
    plt.plot(range(1, 11), predictions[return_col], label=f'Predicted {return_col}', color='tab:orange')
    plt.title(f'Predicted Next 10 Returns for {return_col}')
    plt.xlabel('Day')
    plt.ylabel('Return')
    plt.grid(True)

plt.tight_layout()
plt.show()



sample_submission = pd.read_csv('/kaggle/input/stock-price-prediction-challenge/sample_submission.csv')

dates_from_sample = sample_submission['Date']

submission_df = pd.DataFrame(dates_from_sample, columns=['Date'])

for return_col in predictions:
    submission_df[return_col] = predictions[return_col]

submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())

