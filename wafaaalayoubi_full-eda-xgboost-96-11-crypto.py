import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

from scipy.stats import pearsonr

import xgboost as xgb
from xgboost import plot_importance

import warnings
warnings.filterwarnings("ignore")


train_path = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
test_path = '/kaggle/input/drw-crypto-market-prediction/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)



# ✅ No conversion needed, just sort by index (already datetime)
train_df = train_df.sort_index()
print(train_df.index[:5])  # Confirm it's datetime


print(train_df.head())
print(train_df.columns)


# EDA: Missing values
print(train_df.isnull().sum())
print(test_df.isnull().sum())

# EDA: Inf values
print("Inf values in train:")
print(np.isinf(train_df).sum())

print("Inf values in test:")
print(np.isinf(test_df).sum())


print(train_df['label'].describe())


print(train_df['label'].value_counts())


print(train_df['label'].head(20))


print("Any NaNs?", train_df.isna().sum().sum(), test_df.isna().sum().sum())
print("Any Infs?", np.isinf(train_df.to_numpy()).sum(), np.isinf(test_df.to_numpy()).sum())


# Count inf values per column
inf_counts = pd.DataFrame({
    'train_inf': np.isinf(train_df).sum(axis=0),
    'test_inf': np.isinf(test_df).sum(axis=0)
})

# Filter only columns with inf values
inf_cols = inf_counts[(inf_counts['train_inf'] > 0) | (inf_counts['test_inf'] > 0)]
print("Columns with inf values:")
print(inf_cols.sort_values(by='train_inf', ascending=False))


# Drop columns with inf values from both train and test
cols_to_drop = list(inf_cols.index)
train_df = train_df.drop(columns=cols_to_drop)
test_df = test_df.drop(columns=cols_to_drop)


# Count how many values are outside [-10, 10]
outliers = (train_df['label'].abs() > 10).sum()
print(f"Number of extreme outliers (|label| > 10): {outliers}")

# View a few of them
print(train_df[train_df['label'].abs() > 10].sort_values(by='label'))



plt.figure(figsize=(8, 4))
sns.histplot(train_df['label'], bins=100, kde=True)
plt.title("Distribution of Target Variable (label)")
plt.xlabel("label")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(15,5))
train_df['label'].plot()
plt.title("Label Over Time")
plt.xlabel("Timestamp")
plt.ylabel("label")
plt.show()


train_df['label'].rolling(window=60).mean().plot(figsize=(15,5))
plt.title("Rolling Mean of Label (60-minute window)")
plt.xlabel("Timestamp")
plt.ylabel("Smoothed label")
plt.show()


lags = list(range(1, 11))
autocorrs = [train_df['label'].autocorr(lag=lag) for lag in lags]

plt.figure(figsize=(8,4))
plt.bar(lags, autocorrs)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation by Lag (1 to 10)')
plt.show()


for lag in range(1, 11):
    autocorr = train_df['label'].autocorr(lag=lag)
    print(f"Autocorrelation at lag {lag}: {autocorr:.4f}")


skewness = train_df['label'].skew()
print(f"Skewness of label: {skewness:.4f}")


# # 0. Sort and lag BEFORE split
# train_df = train_df.sort_index()  # just in case
# for lag in [1, 2, 3]:
#     for col in ['volume', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty']:
#         train_df[f'{col}_lag{lag}'] = train_df[col].shift(lag)
#         test_df[f'{col}_lag{lag}'] = test_df[col].shift(lag)

# # 1. Drop rows with NaNs from lagging
# train_df.dropna(inplace=True)

# # ⚠️ DO NOT drop rows from test_df — it's the competition test set!
# # Instead, fill or leave NaNs (XGBoost handles them fine)
# #optional
# test_df.fillna(0, inplace=True)  # or use forward fill if that's logical



df_train, df_valid = train_test_split(train_df, test_size=0.2, random_state=42)

# 2. Separate features and target
X_train = df_train.drop(columns=['label'])
y_train = df_train['label']

X_valid = df_valid.drop(columns=['label'])
y_valid = df_valid['label']


Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)
IQR = Q3 - Q1
low = Q1 - 1.5 * IQR
high = Q3 + 1.5 * IQR


# For IQR clipping, you might want to handle edge cases
if low < high:  # Ensure valid bounds
    y_train_clipped = y_train.clip(low, high)
    y_valid_clipped = y_valid.clip(low, high)
else:
    # Handle case where IQR is very small
    y_train_clipped = y_train.copy()
    y_valid_clipped = y_valid.copy()


# Prepare DMatrix (optional but recommended for XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train_clipped)
dvalid = xgb.DMatrix(X_valid, label=y_valid_clipped)


import gc
del train_df

del y_train
del y_valid

del X_train
del X_valid


# Set XGBoost parameters (regression example)
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",  # fast histogram algorithm
    "seed": 42
}
evals = [(dtrain, "train"), (dvalid, "valid")]


# Train with early stopping on validation
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=10
)


# Predict on validation set
y_pred = model.predict(dvalid)


# Evaluate
rmse = mean_squared_error(y_valid_clipped, y_pred, squared=False)
r2 = r2_score(y_valid_clipped, y_pred)

print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation R2: {r2:.4f}")


# Top 20 features by average gain (most useful metric)
plot_importance(model, importance_type='gain', max_num_features=20)
plt.title("Top 20 Feature Importances")
plt.show()


X_test = test_df.drop(columns = ['label'])

dtest = xgb.DMatrix(X_test)

# Make predictions on test set
xgb_test_preds = model.predict(dtest)

# Generate row IDs (starting at 1)
row_ids = range(1, len(xgb_test_preds) + 1)

# Create submission DataFrame
submission = pd.DataFrame({
    'ID': row_ids,
    'prediction': xgb_test_preds
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved: submission.csv")





