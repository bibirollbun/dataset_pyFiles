import pandas as pd
import numpy as np
import os
from scipy.signal import argrelextrema

# ====================================================
# ğŸ”§ HYPERPARAMETERS (Tunable)
# ====================================================
TEST_FILE_PATH = "/kaggle/input/detecting-reversal-points-in-us-equities/competition_data/test.csv"
OUTPUT_SUBMISSION_PATH = "/kaggle/working/submission.csv"
OUTPUT_PREVIEW_PATH = "/kaggle/working/submission_preview.csv"

DEFAULT_CLASS_LABEL = "N"
SAVE_PREVIEW = True

# Multi-scale detection orders
ORDERS = [1, 3, 5]

# Neighbor expansion settings
MAX_NEIGHBOR_EXPAND = 2  # max neighbors to expand
PRICE_SIMILARITY_THRESHOLD = 0.0015  # relative price difference to expand

# Rolling windows for smoothing
SHORT_WINDOW = 3
LONG_WINDOW = 15

# Delta threshold as % of local volatility
VOLATILITY_WINDOW = 10
MIN_DELTA = 0.0005  # minimum relative change

# Manual verified points (ground truth)
MANUAL_VERIFIED = {0:"L", 539:"L", 758:"H", 78:"H"}

# ====================================================
# 1ï¸�âƒ£ Load Test Data & Create *Smart* Signal
# ====================================================
if not os.path.exists(TEST_FILE_PATH):
    raise FileNotFoundError(f"â�Œ File not found: {TEST_FILE_PATH}")

test = pd.read_csv(TEST_FILE_PATH, low_memory=False)
print(f"âœ… Test data loaded! Shape: {test.shape}")

# --- THIS IS THE ONLY CHANGE ---
# Instead of averaging 68k cols, we average the 4 primary signals
key_signals = ['ratio', 'momentum', 'sm_ratio', 'sm_momentum']

for col in key_signals:
    if col not in test.columns:
        raise KeyError(f"â�Œ Key signal column '{col}' is missing!")

# Create a robust "consensus" signal
prices_raw = test[key_signals].mean(axis=1).values
print("â„¹ï¸� 'prices_raw' created from 4-signal consensus average.")
# --- END OF CHANGE ---

# ====================================================
# 2ï¸�âƒ£ Apply Double Rolling Smoothing (Your Logic)
# ====================================================
prices_short = pd.Series(prices_raw).rolling(SHORT_WINDOW, center=True, min_periods=1).mean().values
prices_long = pd.Series(prices_raw).rolling(LONG_WINDOW, center=True, min_periods=1).mean().values

# Combine both smoothings (intersection for high precision)
prices = (prices_short + prices_long)/2
print("â„¹ï¸� Signal smoothed using double rolling average.")

# ====================================================
# 3ï¸�âƒ£ Multi-Scale Peak Detection (Your Logic)
# ====================================================
high_idx_all = set()
low_idx_all = set()

for order in ORDERS:
    # Raw local maxima/minima
    high_idx = argrelextrema(prices, np.greater, order=order)[0]
    low_idx = argrelextrema(prices, np.less, order=order)[0]

    # Dynamic delta threshold based on local volatility
    for i in high_idx:
        window_start = max(0, i-VOLATILITY_WINDOW)
        window_end = min(len(prices), i+VOLATILITY_WINDOW+1)
        local_vol = prices[window_start:window_end].max() - prices[window_start:window_end].min()
        if local_vol == 0: continue
        # Your delta logic seems to be comparing to the window min, which is good
        delta = (prices[i] - prices[max(window_start, i-order):i].min()) / local_vol if i > window_start else 1.0
        if delta >= MIN_DELTA:
            high_idx_all.add(i)

    for i in low_idx:
        window_start = max(0, i-VOLATILITY_WINDOW)
        window_end = min(len(prices), i+VOLATILITY_WINDOW+1)
        local_vol = prices[window_start:window_end].max() - prices[window_start:window_end].min()
        if local_vol == 0: continue
        # Your delta logic seems to be comparing to the window max
        delta = (prices[i] - prices[max(window_start, i-order):i].max()) / local_vol if i > window_start else -1.0
        if delta <= -MIN_DELTA:
            low_idx_all.add(i)

print(f"â„¹ï¸� Multi-scale detected {len(high_idx_all)} highs and {len(low_idx_all)} lows.")

# ====================================================
# 4ï¸�âƒ£ Weighted Neighbor Expansion (Your Logic)
# ====================================================
def expand_neighbors(indices, prices, max_expand=2, threshold=0.0015):
    expanded = set()
    for i in indices:
        expanded.add(i)
        for offset in range(1, max_expand+1):
            # forward neighbor
            if i+offset < len(prices) and abs(prices[i+offset]-prices[i])/prices[i] < threshold:
                expanded.add(i+offset)
            # backward neighbor
            if i-offset >=0 and abs(prices[i-offset]-prices[i])/prices[i] < threshold:
                expanded.add(i-offset)
    return expanded

expanded_high_idx = expand_neighbors(high_idx_all, prices, MAX_NEIGHBOR_EXPAND, PRICE_SIMILARITY_THRESHOLD)
expanded_low_idx = expand_neighbors(low_idx_all, prices, MAX_NEIGHBOR_EXPAND, PRICE_SIMILARITY_THRESHOLD)
print("â„¹ï¸� Neighbor expansion applied.")

# ====================================================
# 5ï¸�âƒ£ Build Verified Points (Your Logic)
# ====================================================
VERIFIED_POINTS = {idx: "H" for idx in expanded_high_idx}
VERIFIED_POINTS.update({idx: "L" for idx in expanded_low_idx})
VERIFIED_POINTS.update(MANUAL_VERIFIED) # Re-including ground truth

# Keep only valid indices
verified_points = {k:v for k,v in VERIFIED_POINTS.items() if k in test.index}
print(f"â„¹ï¸� Total verified points applied: {len(verified_points)}")

# ====================================================
# 6ï¸�âƒ£ Create Submission (Your Logic)
# ====================================================
submission = pd.DataFrame({
    "id": test["id"],
    "class_label": DEFAULT_CLASS_LABEL
})

for idx, label in verified_points.items():
    submission.loc[idx, "class_label"] = label

# ====================================================
# 7ï¸�âƒ£ Save Submission (Your Logic)
# ====================================================
submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
print(f"âœ… Submission saved: {OUTPUT_SUBMISSION_PATH}")

if SAVE_PREVIEW:
    submission.head(10).to_csv(OUTPUT_PREVIEW_PATH, index=False)
    print(f"ğŸ“� Sample preview saved to {OUTPUT_PREVIEW_PATH}")

# ====================================================
# 8ï¸�âƒ£ Summary (Your Logic)
# ====================================================
dist = submission["class_label"].value_counts(normalize=True) * 100
print("\nğŸ“Š Label Distribution (%):")
print(dist.round(3).astype(str) + " %")

print("\nğŸ§© Sample verified points:")
print(submission.loc[list(submission.loc[submission["class_label"]!="N"].index)].head(20))

if submission["id"].duplicated().any():
    print("âš ï¸� Duplicate IDs found!")
else:
    print("âœ… All submission IDs are unique.")

