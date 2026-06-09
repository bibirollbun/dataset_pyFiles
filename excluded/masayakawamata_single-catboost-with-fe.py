import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.simplefilter('ignore')


train = pd.read_csv("/kaggle/input/black-jack-smart-effect-of-removal-ml/train.csv")
test = pd.read_csv("/kaggle/input/black-jack-smart-effect-of-removal-ml/test.csv")
print("Train Shape:", train.shape)
print("Test Shape :", test.shape)
train.head(3)


# Define the initial counts for each card in a 6-deck shoe
# In a 6-deck shoe, assume:
#   - "1" (Ace) appears 24 times (6 decks × 4 Aces per deck)
#   - Cards "2" through "9" each appear 24 times
#   - "10" represents 10, J, Q, K and appears 96 times (6 decks × 16)
initial_counts = {
    '1': 24,
    '2': 24,
    '3': 24,
    '4': 24,
    '5': 24,
    '6': 24,
    '7': 24,
    '8': 24,
    '9': 24,
    '10': 96
}

# Total cards in a 6-deck shoe
full_total = sum(initial_counts.values())  # 312

# Compute the full deck's average card value:
# For each card, multiply its value by its count and sum over all cards.
full_total_value = sum(int(card) * count for card, count in initial_counts.items())
full_avg = full_total_value / full_total  # ~6.5385

def add_auxiliary_features(df):
    # Ensure card columns are numeric
    for card in initial_counts.keys():
        df[card] = pd.to_numeric(df[card], errors='coerce')
    
    # 1. Total number of cards removed
    card_cols = [str(i) for i in range(1, 11)]
    df['total_removed'] = df[card_cols].sum(axis=1)
    
    # 2. Weighted average value of removed cards
    # (sum(value * count) / total_removed). If total_removed==0, set to 0.
    df['weighted_avg_removed'] = df.apply(
        lambda row: sum(int(card) * row[str(card)] for card in range(1, 11)) / row['total_removed']
                    if row['total_removed'] > 0 else 0,
        axis=1
    )
    
    # 3. Remaining cards in the shoe (full deck minus removed)
    df['remaining_cards'] = full_total - df['total_removed']
    
    # 4. For each card value, compute the remaining count (initial - removed)
    for card in initial_counts.keys():
        df[f'remaining_{card}'] = initial_counts[card] - df[str(card)]
    
    # 5. Total remaining (for sanity check; should equal remaining_cards)
    remaining_cols = [f'remaining_{card}' for card in initial_counts.keys()]
    df['remaining_total'] = df[remaining_cols].sum(axis=1)
    
    # 6. Compute the average card value of the remaining deck
    def calc_remaining_avg(row):
        total = row['remaining_total']
        if total > 0:
            return sum(int(card) * row[f'remaining_{card}'] for card in initial_counts.keys()) / total
        else:
            return 0
    df['remaining_avg'] = df.apply(calc_remaining_avg, axis=1)
    
    # 7. Difference between the remaining deck's average value and the full deck's average
    df['remaining_avg_diff'] = df['remaining_avg'] - full_avg
    
    # 8. Low vs. high removal ratios:
    # Here, we define low cards as 2-6, and high cards as the "10" cards.
    df['low_removed'] = df[['2', '3', '4', '5', '6']].sum(axis=1)
    df['high_removed'] = df['10']
    # Add 1 to avoid division by zero
    df['ratio_low_high'] = (df['low_removed'] + 1) / (df['high_removed'] + 1)
    
    return df

# Apply auxiliary feature creation to both train and test DataFrames
train = add_auxiliary_features(train)
test = add_auxiliary_features(test)

display(train.head())
display(test.head())


from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


TARGET = 'ev'
X = train.drop([TARGET, "id"], axis=1).copy()
y = train[TARGET].copy()
X_test = test.drop(columns='id').copy()


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_pred_cat = np.zeros(len(X))
fold_mse = []
test_preds = np.zeros((len(X_test), FOLDS))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
    print(f"Training fold {fold} ...")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = CatBoostRegressor(
        iterations=20000,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        od_wait=100,
        rsm=1.0,
        verbose=False,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        # verbose=5000
    )
    
    y_pred = model.predict(X_val)
    mse_fold = mean_squared_error(y_val, y_pred)
    fold_mse.append(mse_fold)
    oof_pred_cat[val_idx] = y_pred
    print(f"Fold {fold} MSE: {mse_fold:.8f}")
    
    test_preds[:, fold - 1] = model.predict(X_test)

overall_mse = mean_squared_error(y, oof_pred_cat)
print(f"\nOverall OOF MSE: {overall_mse:.8f}")

final_test_pred_cat = test_preds.mean(axis=1)
print("\nFinal test predictions (first 10 samples):")
print(final_test_pred_cat[:10])


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6))

axes[0].hist(y, bins=50, edgecolor='k', alpha=0.7)
axes[0].set_title("Distribution of y (Target)")
axes[0].set_xlabel("Target Value")
axes[0].set_ylabel("Frequency")
axes[0].grid(True)

axes[1].hist(final_test_pred_cat, bins=50, edgecolor='k', alpha=0.7)
axes[1].set_title("Distribution of Final Test Predictions")
axes[1].set_xlabel("Predicted Value")
axes[1].set_ylabel("Frequency")
axes[1].grid(True)

plt.tight_layout()
plt.show()


sub = pd.read_csv("/kaggle/input/black-jack-smart-effect-of-removal-ml/sample_submission.csv")
sub.ev = final_test_pred_cat
sub.to_csv("submission.csv", index=False)
sub.head(3)

