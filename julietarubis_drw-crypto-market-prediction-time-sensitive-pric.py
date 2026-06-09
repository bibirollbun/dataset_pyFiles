import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


# === Load Data ===
train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')


# === Feature Engineering ===
# Derived features
for df in [train, test]:
    df['order_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-6)
    df['trade_aggressiveness'] = (df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-6)



# Proprietary features
X_cols = [col for col in train.columns if col.startswith("X_")]


# Final feature set
feature_cols = ['order_imbalance', 'trade_aggressiveness', 'bid_qty', 'ask_qty',
                'buy_qty', 'sell_qty', 'volume'] + X_cols


# Target
target = train['label']


# === Time-Based Split ===
split_idx = int(len(train) * 0.8)
X_train = train[feature_cols].iloc[:split_idx]
y_train = train['label'].iloc[:split_idx]
X_val = train[feature_cols].iloc[split_idx:]
y_val = train['label'].iloc[split_idx:]


# === Scale Features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test[feature_cols])


# === Train Ridge Regression ===
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)


# === Validate Using Pearson Correlation ===
val_preds = ridge.predict(X_val_scaled)
pearson = pearsonr(y_val, val_preds)[0]
print(f"Validation Pearson Correlation (Ridge): {pearson:.5f}")


# === Predict on Test Set ===
test_preds = ridge.predict(X_test_scaled)


# === Create Submission ===
# Load the sample submission to get correct format
submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')

# Replace the correct column with your predictions
# Replace 'label' below if the column name in sample_submission is different
submission['label'] = test_preds  # <- make sure test_preds is your final prediction array

# Save the submission file in proper format
submission.to_csv("submission.csv", index=False)

# Confirm the structure
print("Submission file saved. Preview:")
display(submission.head())
print("Columns:", submission.columns.tolist())
print("Shape:", submission.shape)



# === Optional: Plot Prediction Distribution ===
plt.figure(figsize=(8,4))
plt.hist(test_preds, bins=100, alpha=0.7)
plt.title("Test Predictions Distribution")
plt.xlabel("Predicted Label")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

