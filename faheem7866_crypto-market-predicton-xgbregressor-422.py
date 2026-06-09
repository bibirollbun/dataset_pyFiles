# Step 1: Libraries
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr



# Step 2: Load Data
train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)



# Step 3: Select Features
features = [col for col in train.columns if col not in ['label', 'timestamp', 'id']]
X = train[features]
y = train['label']



# Step 4: Time-Based Split (last 10% for validation)
split_idx = int(len(train) * 0.9)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]



# Step 5: Initialize Model
model = XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.01,
    max_depth=6,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist'  # faster on Kaggle
)



# Step 6: Train with Early Stopping
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    early_stopping_rounds=50,
    verbose=100
)



# Step 7: Validation Results
val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
r2 = r2_score(y_val, val_preds)
corr, _ = pearsonr(y_val, val_preds)

print(f"\nðŸ“Š Validation RMSE: {rmse:.5f}")
print(f"ðŸ“Š R2 Score: {r2:.5f}")
print(f"ðŸ“Š Pearson Correlation: {corr:.5f}")



# Step 8: Test Prediction
X_test = test[features]
test_preds = model.predict(X_test)



# Step 9: Create Submission File
submission = sample_submission.copy()
submission['label'] = test_preds
submission.to_csv("submission.csv", index=False)
print("âœ… 'submission.csv' ready for upload.")


