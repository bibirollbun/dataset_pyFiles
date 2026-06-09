import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression

ci = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/city_indexes.csv') # one row per year
csi = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/city_search_index.csv') # several rows per training month
sp = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/sector_POI.csv') # at most one row per sector

train_lt = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions.csv')
train_ltns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv')
train_pht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions.csv')
train_phtns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions_nearby_sectors.csv')
train_nht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
train_nhtns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions_nearby_sectors.csv')
test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')


month_codes = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12
}

test_id = test.id.str.split('_', expand=True)
test['month'] = test_id[0]
test['sector'] = test_id[1]
del test_id

for df in [train_lt, train_ltns, train_pht, train_phtns, train_nht, train_nhtns, csi, sp, test]:
    if df is not csi:
        df['sector_id'] = df.sector.str.slice(7, None).astype(int)
        df.drop(columns=['sector'],inplace=True)
    if df is not sp:
        df['year'] = df.month.str.slice(0, 4).astype(int)
        df['month'] = df.month.str.slice(5, None).map(month_codes)
        df['time'] = (df['year'] - 2019) * 12 + df['month'] - 1 # min=0, max=66

amount_new_house_transactions = train_nht.set_index(['time', 'sector_id']).amount_new_house_transactions.unstack()
# Missing values must be filled with zero:
amount_new_house_transactions = amount_new_house_transactions.fillna(0)
# We add sector 95, which has no transactions during the training period:
amount_new_house_transactions[95] = 0
amount_new_house_transactions = amount_new_house_transactions[np.arange(1, 97)]
amount_new_house_transactions.astype(int)


import matplotlib.pyplot as plt

for col in amount_new_house_transactions.columns:
    plt.figure()  
    series = amount_new_house_transactions[col].replace(0, np.nan)
    plt.plot(series)
    plt.title(f"New House Transactions - {col}")
    plt.show()



import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

np.random.seed(0)
df = amount_new_house_transactions[50:]
display(df)
# Prepare results dict
future_preds = {}

# Current time steps
X = np.arange(len(df)).reshape(-1, 1)
future_steps = np.arange(len(df), len(df) + 12).reshape(-1, 1)

for col in df.columns:
    y = df[col].values
    
    # Fit linear regression for this timeseries
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next 12 steps
    y_future = model.predict(future_steps)
    future_preds[col] = np.clip(y_future, 0, None)

# Convert predictions to DataFrame
preds_df = pd.DataFrame(future_preds)
preds_df


test['new_house_transaction_amount'] = preds_df.T.unstack().values

test[['id', 'new_house_transaction_amount']].to_csv('submission.csv', index=False)
!head submission.csv


import matplotlib.pyplot as plt
import numpy as np

# Create a large figure with subplots for all 96 time series
fig, axes = plt.subplots(12, 8, figsize=(24, 36))
fig.suptitle('New House Transactions: Historical Data and 12-Month Predictions', fontsize=16, y=0.995)

# Flatten axes for easier iteration
axes_flat = axes.flatten()

# Time indices for plotting
historical_time = np.arange(len(df))
future_time = np.arange(len(df), len(df) + 12)
combined_time = np.concatenate([historical_time, future_time])

for i, col in enumerate(df.columns):
    ax = axes_flat[i]
    
    # Historical data
    historical_data = df[col].values
    
    # Future predictions
    future_data = preds_df[col].values
    
    # Plot historical data
    ax.plot(historical_time, historical_data, 'b-', linewidth=1.5, label='Historical')
    
    # Plot predictions as dotted line
    ax.plot(future_time, future_data, 'r--', linewidth=2, label='Prediction')
    
    # Connect the last historical point with first prediction point
    ax.plot([historical_time[-1], future_time[0]], 
            [historical_data[-1], future_data[0]], 'r--', linewidth=2)
    
    # Customize the plot
    ax.set_title(f'Sector {col}', fontsize=10)
    ax.set_xlabel('Time')
    ax.set_ylabel('Transactions')
    ax.grid(True, alpha=0.3)
    
    # Add vertical line to separate historical and prediction
    ax.axvline(x=len(df)-1, color='gray', linestyle=':', alpha=0.7)
    
    # Only show legend on first subplot
    if i == 0:
        ax.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.show()

# Alternative: Create individual larger plots for better detail
print("\nCreating individual detailed plots...")

# Create individual plots with better visibility
for col in df.columns:
    plt.figure(figsize=(12, 6))
    
    # Historical data
    historical_data = df[col].values
    future_data = preds_df[col].values
    
    # Time indices
    historical_time = np.arange(len(df))
    future_time = np.arange(len(df), len(df) + 12)
    
    # Plot historical data
    plt.plot(historical_time, historical_data, 'b-', linewidth=2, label='Historical Data', marker='o', markersize=4)
    
    # Plot predictions
    plt.plot(future_time, future_data, 'r--', linewidth=2, label='12-Month Prediction', marker='s', markersize=4)
    
    # Connect last historical with first prediction
    plt.plot([historical_time[-1], future_time[0]], 
             [historical_data[-1], future_data[0]], 'r--', linewidth=2)
    
    # Customize
    plt.title(f'New House Transactions - Sector {col}\nHistorical Data and 12-Month Linear Regression Prediction', fontsize=14)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Number of Transactions', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add vertical separator
    plt.axvline(x=len(df)-1, color='gray', linestyle=':', alpha=0.7, label='Prediction Start')
    
    # Add some statistics as text
    mean_historical = np.mean(historical_data)
    mean_prediction = np.mean(future_data)
    
    plt.text(0.02, 0.98, f'Historical Mean: {mean_historical:.1f}\nPredicted Mean: {mean_prediction:.1f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Summary statistics
print("\nSummary Statistics:")
print(f"Total sectors: {len(df.columns)}")
print(f"Historical time steps: {len(df)}")
print(f"Prediction steps: 12")
print(f"\nHistorical data range: {df.min().min():.0f} - {df.max().max():.0f} transactions")
print(f"Prediction range: {preds_df.min().min():.0f} - {preds_df.max().max():.0f} transactions")

# Show sectors with highest predicted growth
growth_rates = []
for col in df.columns:
    last_historical = df[col].iloc[-1]
    first_prediction = preds_df[col].iloc[0]
    if last_historical > 0:
        growth_rate = (first_prediction - last_historical) / last_historical * 100
    else:
        growth_rate = 0 if first_prediction == 0 else float('inf')
    growth_rates.append((col, growth_rate))

growth_rates.sort(key=lambda x: x[1], reverse=True)
print(f"\nTop 10 sectors with highest predicted growth:")
for sector, growth in growth_rates[:10]:
    if growth != float('inf'):
        print(f"Sector {sector}: {growth:.1f}% growth")
    else:
        print(f"Sector {sector}: Starting from zero")

print(f"\nTop 10 sectors with lowest predicted growth:")
for sector, growth in growth_rates[-10:]:
    print(f"Sector {sector}: {growth:.1f}% growth")

