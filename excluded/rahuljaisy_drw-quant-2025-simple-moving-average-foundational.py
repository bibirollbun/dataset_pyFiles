import warnings
warnings.filterwarnings("ignore")

import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr


# Load Parquet
train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet', engine='pyarrow')

# Replace infs
train_df.replace([np.inf, -np.inf], 0, inplace=True)

# Focus on a segment of the label
y_test = train_df.iloc[500_000:550_000]['label'].reset_index(drop=True)

# Clean up
del train_df
gc.collect()


window_size = 5
y_pred = y_test.rolling(window=window_size).mean().fillna(method='bfill')


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
corr_coef, p_val = pearsonr(y_test, y_pred)

results = pd.DataFrame({
    'Model': [f'SMA-{window_size}'],
    'MAE': [mae],
    'MSE': [mse],
    'RMSE': [rmse],
    'R2 Score': [r2],
    'Pearson Corr': [corr_coef],
    'P-Value': [p_val],
})

results.style.background_gradient(cmap='YlGnBu', axis=1).format(precision=5)


np.random.seed(42)
idx = np.random.choice(len(y_test), size=100, replace=False)

plt.figure(figsize=(12, 6))
plt.plot(y_test.iloc[idx].values, label='Actual', marker='o')
plt.plot(y_pred.iloc[idx].values, label='SMA Predicted', marker='x')
plt.title("SMA vs Actual (100 Random Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Target")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


errors = y_test - y_pred

plt.figure(figsize=(10, 4))
plt.hist(errors, bins=50, color='salmon', edgecolor='black')
plt.title("Error Distribution: SMA Prediction")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()


window_metrics = []

for w in [3, 5, 10, 20, 50]:
    y_w = y_test.rolling(window=w).mean().fillna(method='bfill')
    mae = mean_absolute_error(y_test, y_w)
    r2 = r2_score(y_test, y_w)
    corr, _ = pearsonr(y_test, y_w)
    window_metrics.append((w, mae, r2, corr))

sweep_df = pd.DataFrame(window_metrics, columns=['Window Size', 'MAE', 'R2 Score', 'Pearson Corr'])
sweep_df.style.bar(subset=['R2 Score', 'Pearson Corr'], color='#5fba7d').format(precision=4)


import pandas as pd
import numpy as np

# 1. Load the full train labels to compute rolling mean
train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet', engine='pyarrow')
labels = train_df['label']

# 2. Compute SMA on the training labels and take the last value
window_size = 5
sma_series = labels.rolling(window=window_size).mean().fillna(method='bfill')
last_sma_value = sma_series.iloc[-1]

# 3. Load test DataFrame just to get its length
test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet', engine='pyarrow')
n_test = len(test_df)

# 4. Create a constant窶心MA prediction array of the same length as test set
sma_pred = np.full(shape=n_test, fill_value=last_sma_value)

# 5. Prepare and write submission
sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
sample_submission['prediction'] = sma_pred
sample_submission.to_csv('sample_submission.csv', index=False)


import pandas as pd
import numpy as np

# 1. Load the full train labels to compute rolling mean
train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet', engine='pyarrow')
labels = train_df['label']

# 2. Compute SMA on the training labels and take the last value
window_size = 5
sma_series = labels.rolling(window=window_size).mean().fillna(method='bfill')
last_sma_value = sma_series.iloc[-1]

# 3. Load test DataFrame just to get its length
test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet', engine='pyarrow')
n_test = len(test_df)

# 4. Create a constant窶心MA prediction array of the same length as test set
sma_pred = np.full(shape=n_test, fill_value=last_sma_value)

# 5. Prepare and write submission
sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
sample_submission['prediction'] = sma_pred
sample_submission.to_csv('sample_submission.csv', index=False)

