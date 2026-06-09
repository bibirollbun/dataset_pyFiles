import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import pearsonr


train_path = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
test_path = '/kaggle/input/drw-crypto-market-prediction/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

print("Train shape:", train.shape)
print("Test shape:", test.shape)



train.head()


test.head()


print(list(test.columns))


print(train.head())
print(train.columns)

# Check target distribution
print(train['label'].describe())



print("Any NaNs?", train.isna().sum().sum(), test.isna().sum().sum())
print("Any Infs?", np.isinf(train.to_numpy()).sum(), np.isinf(test.to_numpy()).sum())



# Count inf values per column
inf_counts = pd.DataFrame({
    'train_inf': np.isinf(train).sum(),
    'test_inf': np.isinf(test).sum()
})

# Filter only columns with inf values
inf_cols = inf_counts[(inf_counts['train_inf'] > 0) | (inf_counts['test_inf'] > 0)]
print("Columns with inf values:")
print(inf_cols.sort_values(by='train_inf', ascending=False))


# Step 2: Prepare cleaned features and target from train
X = train.drop(columns=['label'] + list(inf_cols.index))
y = train['label']


# Step 3: Prepare cleaned test features
X_test = test.drop(columns = ['label'] + list(inf_cols.index))


final_model = Ridge()
final_model.fit(X, y)


X.shape


X_test.shape


# Columns in X but not in X_test
diff_in_train = set(X.columns) - set(X_test.columns)

# Columns in X_test but not in X
diff_in_test = set(X_test.columns) - set(X.columns)

print("Columns in train but missing in test:", diff_in_train)
print("Columns in test but missing in train:", diff_in_test)



test_preds = final_model.predict(X_test)


# Generate row IDs (starting at 1)
row_ids = range(1, len(test_preds) + 1)

# Create submission DataFrame
submission = pd.DataFrame({
    'ID': row_ids,
    'prediction': test_preds
})

# Save to CSV without index
submission.to_csv('submission.csv', index=False)

print("✅ Submission file saved: submission.csv")




