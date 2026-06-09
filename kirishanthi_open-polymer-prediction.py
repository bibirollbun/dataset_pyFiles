import seaborn as sns 
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import joblib
import time
import csv
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import xgboost as xgb


# Load datasets
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')


supp1 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv')
supp2 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv')
supp3 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv')
supp4 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv')


print("ðŸ”¹ Train shape:", train.shape)
print("ðŸ”¹ Test shape:", test.shape)
print("\n Train columns:\n", train.columns)


print("Train columns:", train.columns.tolist())


# Merge supplement data
for df in [supp1, supp2, supp3, supp4]:
    train = train.merge(df, on='SMILES', how='left')
    test = test.merge(df, on='SMILES', how='left')


# Rename duplicate columns if any
if 'Tg_x' in train.columns:
    train.rename(columns={'Tg_x': 'Tg'}, inplace=True)
if 'FFV_x' in train.columns:
    train.rename(columns={'FFV_x': 'FFV'}, inplace=True)


# Drop duplicated target columns if exist
train.drop(columns=['Tg_y', 'FFV_y'], inplace=True, errors='ignore')
test.drop(columns=['Tg_y', 'FFV_y'], inplace=True, errors='ignore')


# Check no duplicates
assert train.columns.duplicated().sum() == 0, "Duplicate columns in train!"
assert test.columns.duplicated().sum() == 0, "Duplicate columns in test!"


target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


# Fill missing values with train mean
fill_cols = target_cols + list(train.columns.difference(['id', 'SMILES'] + target_cols))
for col in fill_cols:
    if col in train.columns:
        train[col] = train[col].fillna(train[col].mean())
    if col in test.columns:
        test[col] = test[col].fillna(train[col].mean())


for col in target_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(train[col], bins=40, kde=True, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(8, 6))
corr_matrix = train[target_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='viridis', linewidths=0.5)
plt.title("Correlation Matrix of Target Properties")
plt.tight_layout()
plt.show()



for col in target_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=train[col], color='lightgreen')
    plt.title(f"Boxplot of {col}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



train['smiles_len'] = train['SMILES'].apply(len)
test['smiles_len'] = test['SMILES'].apply(len)

plt.figure(figsize=(8, 4))
sns.histplot(train['smiles_len'], label='Train', kde=True, bins=40, color='blue')
sns.histplot(test['smiles_len'], label='Test', kde=True, bins=40, color='orange')
plt.title("SMILES String Length Distribution")
plt.xlabel("SMILES Length")
plt.legend()
plt.tight_layout()
plt.show()



# Pair plot with full dataset
sns.pairplot(train[target_cols])
plt.suptitle("Pair Plot of All Target Properties (Full Dataset)", y=1.02)
plt.show()


missing_counts = train[target_cols].isna().sum()

plt.figure(figsize=(6, 4))
missing_counts.plot(kind='bar', color='crimson')
plt.title("Missing Values per Target Column")
plt.ylabel("Missing Count")
plt.grid(True)
plt.tight_layout()
plt.show()



# Mean of each target column
mean_values = train[target_cols].mean()

# Plot
plt.figure(figsize=(6, 4))
mean_values.plot(kind='bar', color='salmon', edgecolor='black')
plt.title("Mean Value per Target Column")
plt.ylabel("Mean")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()



# SMILES ASCII encoding function
def featurize_smiles(smiles_series):
    features = []
    max_len = 100
    for s in smiles_series:
        vec = np.array([ord(c) for c in s])
        if len(vec) < max_len:
            vec = np.pad(vec, (0, max_len - len(vec)), constant_values=0)
        else:
            vec = vec[:max_len]
        features.append(vec)
    return np.array(features, dtype=np.float32)

X_train_full = featurize_smiles(train['SMILES'])
X_test_full = featurize_smiles(test['SMILES'])



# Manual KFold Split
def manual_kfold_split(X, n_splits=5, shuffle=True, random_seed=42):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(idx)
    fold_sizes = np.full(n_splits, n // n_splits)
    fold_sizes[:n % n_splits] += 1
    splits, cur = [], 0
    for size in fold_sizes:
        val = idx[cur:cur + size]
        trn = np.concatenate([idx[:cur], idx[cur + size:]])
        splits.append((trn, val))
        cur += size
    return splits

def manual_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}


dtest = xgb.DMatrix(X_test_full)
final_test_preds = pd.DataFrame({'id': test['id']})

X = X_train_full
all_true = []
all_pred = []

for tgt in target_cols:
    print(f"\nðŸ“Œ Training target: {tgt}")
    y = train[tgt].values.astype(np.float32)
    splits = manual_kfold_split(X, n_splits=5, shuffle=True)
    oof_preds = np.zeros_like(y)
    test_preds = np.zeros(len(test), dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(splits):
        dtrain = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
        dval = xgb.DMatrix(X[val_idx], label=y[val_idx])
        model = xgb.train(params, dtrain,
                          num_boost_round=1000,
                          evals=[(dval, 'valid')],
                          early_stopping_rounds=50,
                          verbose_eval=False)
        oof_preds[val_idx] = model.predict(dval)
        test_preds += model.predict(dtest) / len(splits)
        fold_mae = manual_mae(y[val_idx], oof_preds[val_idx])
        print(f"  Fold {fold+1} MAE: {fold_mae:.5f}")

    total_mae = manual_mae(y, oof_preds)
    print(f"âœ… Total MAE for {tgt}: {total_mae:.5f}")
    final_test_preds[tgt] = test_preds
    all_true.append(y)
    all_pred.append(oof_preds)


# Plot Actual vs Predicted for all targets
colors = ['red', 'blue', 'green', 'purple', 'orange']
markers = ['o', 's', '^', 'D', 'x']

plt.figure(figsize=(8, 8))
for i, tgt in enumerate(target_cols):
    plt.scatter(all_true[i], all_pred[i], alpha=0.4, color=colors[i], marker=markers[i], label=tgt)

min_val = min([min(a) for a in all_true])
max_val = max([max(a) for a in all_true])
plt.plot([min_val, max_val], [min_val, max_val], 'black', linestyle='--', label='Ideal')

plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.title("ðŸ“ˆ Actual vs Predicted for All Targets")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Calculate overall MAE
overall_mae = manual_mae(np.concatenate(all_true), np.concatenate(all_pred))
print(f"\nðŸ“Š Overall MAE across all targets: {overall_mae:.5f}")

# Ensure column order
final_test_preds = final_test_preds[['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]
final_test_preds.to_csv('submission.csv', index=False)

# Final check
print("âœ… submission.csv saved successfully!")
print(final_test_preds.head())


