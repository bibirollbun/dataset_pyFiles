# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import os

# File paths
file_paths = {
    "sample_submission": "/kaggle/input/china-real-estate-demand-prediction/sample_submission.csv",
    "test": "/kaggle/input/china-real-estate-demand-prediction/test.csv",
    "city_search_index": "/kaggle/input/china-real-estate-demand-prediction/train/city_search_index.csv",
    "land_transactions_nearby_sectors": "/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv",
    "new_house_transactions_nearby_sectors": "/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions_nearby_sectors.csv",
    "city_indexes": "/kaggle/input/china-real-estate-demand-prediction/train/city_indexes.csv",
    "pre_owned_house_transactions": "/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions.csv",
    "new_house_transactions": "/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv",
    "land_transactions": "/kaggle/input/china-real-estate-demand-prediction/train/land_transactions.csv",
    "sector_POI": "/kaggle/input/china-real-estate-demand-prediction/train/sector_POI.csv",
    "pre_owned_house_transactions_nearby_sectors": "/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions_nearby_sectors.csv"
}

# Function to load and summarize each file
def load_and_describe(name, path):
    print(f"\n{'='*80}")
    print(f"ğŸ“� FILE: {name} â€” {os.path.basename(path)}")
    
    try:
        df = pd.read_csv(path)
        print(f"âœ… Shape: {df.shape}")
        print(f"ğŸ§¾ Columns & Dtypes:\n{df.dtypes}")
        print(f"\nğŸ“Š Describe:\n{df.describe(include='all', datetime_is_numeric=True)}")
        print(f"\nğŸ”� Sample Rows:\n{df.sample(2)}")
    except Exception as e:
        print(">")

# Run through all files
for name, path in file_paths.items():
    load_and_describe(name, path)



import pandas as pd

# Define file paths with labels
file_paths = {
    "sample_submission": "/kaggle/input/china-real-estate-demand-prediction/sample_submission.csv",
    "test": "/kaggle/input/china-real-estate-demand-prediction/test.csv",
    "city_search_index": "/kaggle/input/china-real-estate-demand-prediction/train/city_search_index.csv",
    "land_transactions_nearby_sectors": "/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv",
    "new_house_transactions_nearby_sectors": "/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions_nearby_sectors.csv",
    "city_indexes": "/kaggle/input/china-real-estate-demand-prediction/train/city_indexes.csv",
    "pre_owned_house_transactions": "/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions.csv",
    "new_house_transactions": "/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv",
    "land_transactions": "/kaggle/input/china-real-estate-demand-prediction/train/land_transactions.csv",
    "sector_POI": "/kaggle/input/china-real-estate-demand-prediction/train/sector_POI.csv",
    "pre_owned_house_transactions_nearby_sectors": "/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions_nearby_sectors.csv"
}

# Loop through each file and print sample(2)
for name, path in file_paths.items():
    print(f"\n{'='*80}")
    print(f"ğŸ“� FILE: {name}")
    try:
        df = pd.read_csv(path)
        print(df.sample(2, random_state=42))  # random_state for reproducibility
    except Exception as e:
        print(f"â�Œ Failed to read {name}: {e}")



# Debug: Check if file exists and what's in it
import os
path = '/kaggle/input/china-real-estate-demand-prediction/train/city_search_index.csv'
if os.path.exists(path):
    df_temp = pd.read_csv(path)
    print("Columns:", df_temp.columns.tolist())
    print(df_temp.head())
else:
    print("File not found!")


# #!/usr/bin/env python3

# import pandas as pd
# import numpy as np
# import warnings
# warnings.filterwarnings('ignore')

# def month_str_to_time(month_str):
#     if '-' in month_str:
#         year, month = month_str.split('-')
#         year = int(year)
#         month_map = {
#             'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
#             'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
#         }
#         month_num = month_map[month]
#     else:
#         parts = month_str.split()
#         year = int(parts[0])
#         month_name = parts[1]
#         month_map = {
#             'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
#             'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
#         }
#         month_num = month_map[month_name]
#     return (year - 2019) * 12 + month_num

# def load_and_preprocess():
#     train = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
#     test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')
#     train['time'] = train['month'].apply(month_str_to_time)
#     train['sector_num'] = train['sector'].str.extract(r'(\d+)').astype(int)
#     return train, test

# def main():
#     print("=== ğŸš€ FINAL PUSH: V7 - Maximize Score, Minimize Zeros ===")
    
#     train, test = load_and_preprocess()
    
#     # Pivot: time x sector
#     hist = train.pivot(index='time', columns='sector_num', values='amount_new_house_transactions').fillna(0)
#     hist = hist.reindex(columns=range(1, 97), fill_value=0)
    
#     final_preds = []
    
#     for sector in range(1, 97):
#         series = hist[sector]
#         positive = series[series > 0]
        
#         # 1. Base: geometric mean of last 6 positive
#         recent_pos = positive.tail(6)
#         if len(recent_pos) >= 2:
#             base = np.exp(np.log(recent_pos).mean())
#         elif len(recent_pos) == 1:
#             base = recent_pos.iloc[0]
#         else:
#             base = series.mean() if series.mean() > 0 else 100  # fallback
        
#         # 2. Don't zero out unless truly dead
#         last_3_months = series.tail(3)
#         if (last_3_months == 0).all():
#             # Was dead recently â†’ predict small value, not zero
#             # Why? true might be small, not zero
#             pred = base * 0.1  # small positive
#         else:
#             pred = base
        
#         # 3. Cap extreme values (reduce MAPE)
#         if pred > 120000:
#             pred = 120000
#         if pred < 1:
#             pred = max(pred, 1.0)  # avoid near-zero
        
#         final_preds.append(pred)
    
#     # 4. Create test prediction matrix (months 67â€“78)
#     test_pred = pd.DataFrame(
#         {t: final_preds for t in range(67, 79)},
#         index=range(1, 97)
#     ).T
    
#     # 5. Generate submission
#     submission = []
#     for _, row in test.iterrows():
#         test_id = row['id']
#         month_str, sector_str = test_id.split('_')
#         sector_num = int(sector_str.split()[1])
#         time_val = month_str_to_time(month_str)
        
#         pred = test_pred.loc[time_val, sector_num] if (time_val in test_pred.index and sector_num in test_pred.columns) else 1.0
#         submission.append({'id': test_id, 'amount_new_house_transactions': pred})
    
#     submission_df = pd.DataFrame(submission)
#     submission_df.to_csv('submission.csv', index=False)
    
#     print(f"âœ… Final submission saved! Shape: {submission_df.shape}")
#     print(f"ğŸ“ˆ Min: {submission_df['amount_new_house_transactions'].min():.2f}, "
#           f"Max: {submission_df['amount_new_house_transactions'].max():.2f}")
#     print("ğŸ”� First 10 predictions:")
#     print(submission_df.head(10))
    
#     return submission_df

# if __name__ == "__main__":
#     submission = main()


#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("ğŸš€ V14: THE ULTIMATE CHAMPION â€“ ZERO-AWARE, SIGNAL-FUSED, ENSEMBLE POWERHOUSE")

# === 1. Time Conversion ===
def month_str_to_time(month_str):
    if '-' in month_str:
        year, month = month_str.split('-')
        year = int(year)
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        month_num = month_map[month]
    else:
        parts = month_str.split()
        year = int(parts[0])
        month_name = parts[1]
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        month_num = month_map[month_name]
    return (year - 2019) * 12 + month_num

# === 2. Load All Data ===
def load_data():
    base = '/kaggle/input/china-real-estate-demand-prediction/'
    train_dir = base + 'train/'
    
    train = pd.read_csv(train_dir + 'new_house_transactions.csv')
    test = pd.read_csv(base + 'test.csv')
    
    search = pd.read_csv(train_dir + 'city_search_index.csv')
    nearby_new = pd.read_csv(train_dir + 'new_house_transactions_nearby_sectors.csv')
    pre_owned = pd.read_csv(train_dir + 'pre_owned_house_transactions.csv')
    poi = pd.read_csv(train_dir + 'sector_POI.csv')
    
    return train, test, search, nearby_new, pre_owned, poi

# === 3. Preprocess Everything ===
def preprocess(train, search, nearby_new, pre_owned, poi):
    # Add time and sector_num
    train['time'] = train['month'].apply(month_str_to_time)
    train['sector_num'] = train['sector'].str.extract(r'(\d+)').astype(int)
    
    # --- Search Volume: Aggregate by time ---
    if 'search_volume' in search.columns:
        search['time'] = search['month'].apply(month_str_to_time)
        search_vol = search.groupby('time')['search_volume'].sum()
    else:
        search_vol = pd.Series(0, index=range(1, 79))
    
    # --- Nearby New Transactions ---
    nearby_new['time'] = nearby_new['month'].apply(month_str_to_time)
    nearby_new['sector_num'] = nearby_new['sector'].str.extract(r'(\d+)').astype(int)
    nearby_pivot = nearby_new.pivot(
        index='time', columns='sector_num', values='amount_new_house_transactions_nearby_sectors'
    ).fillna(0).reindex(columns=range(1, 97), fill_value=0)
    
    # --- Pre-Owned Transactions ---
    pre_owned['time'] = pre_owned['month'].apply(month_str_to_time)
    pre_owned['sector_num'] = pre_owned['sector'].str.extract(r'(\d+)').astype(int)
    pre_owned_pivot = pre_owned.pivot(
        index='time', columns='sector_num', values='amount_pre_owned_house_transactions'
    ).fillna(0).reindex(columns=range(1, 97), fill_value=0)
    
    # --- POI: Population and Density ---
    poi['sector_num'] = poi['sector'].str.extract(r'(\d+)').astype(int)
    pop_scale = poi.set_index('sector_num')['population_scale'].to_dict()
    
    return train, search_vol, nearby_pivot, pre_owned_pivot, pop_scale

# === 4. Baseline 1: Geometric Mean (Safe Core) ===
def geometric_mean_baseline(train):
    hist = train.pivot(index='time', columns='sector_num', values='amount_new_house_transactions').fillna(0)
    hist = hist.reindex(columns=range(1, 97), fill_value=0)
    
    preds = []
    for sector in range(1, 97):
        series = hist[sector]
        pos = series[series > 0].tail(6)
        if len(pos) >= 2:
            pred = np.exp(np.log(pos).mean())
        elif len(pos) == 1:
            pred = pos.iloc[0]
        else:
            pred = series.mean()
        preds.append(max(1.0, pred))
    
    return np.array(preds)

# === 5. Baseline 2: Seasonal (Same Month Last Year) ===
def seasonal_baseline(train):
    hist = train.pivot(index='time', columns='sector_num', values='amount_new_house_transactions').fillna(0)
    hist = hist.reindex(columns=range(1, 97), fill_value=0)
    
    preds = []
    for sector in range(1, 97):
        if 54 in hist.index:  # 2023 Jun = 54, 2024 Jun = 66
            pred = hist.loc[54, sector] if hist.loc[54, sector] > 0 else hist[sector].mean()
        else:
            pred = hist[sector].mean()
        preds.append(max(0.1, pred))
    
    return np.array(preds)

# === 6. Baseline 3: Signal-Enhanced Physics Model ===
def physics_baseline(train, search_vol, nearby_pivot, pre_owned_pivot, pop_scale):
    hist = train.pivot(index='time', columns='sector_num', values='amount_new_house_transactions').fillna(0)
    hist = hist.reindex(columns=range(1, 97), fill_value=0)
    
    preds = []
    for sector in range(1, 97):
        series = hist[sector]
        pos = series[series > 0].tail(6)
        
        if len(pos) == 0:
            base = series.mean()
        else:
            base = np.exp(np.log(pos).mean())
        
        # Signals
        search_growth = (search_vol.get(66, 0) - search_vol.get(54, 0)) / (search_vol.get(54, 1) + 1)
        search_factor = np.clip(1.0 + 0.2 * search_growth, 0.9, 1.2)
        
        nearby_factor = 1.0 + (nearby_pivot[sector].tail(1).values[0] / 10000) * 0.05
        pre_owned_factor = 1.0 + (pre_owned_pivot[sector].tail(1).values[0] / 10000) * 0.03
        pop_factor = np.clip(pop_scale.get(sector, 100000) / 100000, 0.9, 1.3)
        
        pred = base * search_factor * nearby_factor * pre_owned_factor * pop_factor
        pred = np.clip(pred, 1.0, 120000)
        preds.append(pred)
    
    return np.array(preds)

# === 7. Fold-Based Validation (Real Scoring) ===
def custom_score(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs((y_true - y_pred) / np.maximum(y_true, eps))
    good_mask = ape <= 1.0
    good_rate = good_mask.mean()
    if good_rate < 0.7:
        return 0
    good_ape = ape[good_mask]
    mape = np.mean(good_ape)
    scaled_mape = mape / good_rate
    return 1 - scaled_mape

def time_series_cv_ensemble(train, baselines):
    hist = train.pivot(index='time', columns='sector_num', values='amount_new_house_transactions').fillna(0)
    hist = hist.reindex(columns=range(1, 97), fill_value=0)
    
    scores = []
    for fold_time in range(54, 66):  # Validate on months 54 to 66
        if fold_time not in hist.index:
            continue
        y_true = hist.loc[fold_time].values
        fold_scores = [custom_score(y_true, b) for b in baselines]
        scores.append(fold_scores)
    
    if len(scores) == 0:
        return [0.4, 0.3, 0.3]
    
    scores = np.array(scores)
    weights = np.mean(scores, axis=0)
    weights = weights / weights.sum() if weights.sum() > 0 else [0.33, 0.33, 0.34]
    return weights.tolist()

# === 8. Final Ensemble Prediction ===
def main():
    print("ğŸ�† V14: THE ULTIMATE CHAMPION â€“ ZERO-AWARE, SIGNAL-FUSED, ENSEMBLE POWERHOUSE")
    
    # Load and preprocess
    train_raw, test_raw, search_raw, nearby_raw, pre_owned_raw, poi_raw = load_data()
    train, search_vol, nearby_pivot, pre_owned_pivot, pop_scale = preprocess(
        train_raw, search_raw, nearby_raw, pre_owned_raw, poi_raw
    )
    
    # Generate baselines
    base1 = geometric_mean_baseline(train)  # Robust
    base2 = seasonal_baseline(train)        # Seasonal
    base3 = physics_baseline(train, search_vol, nearby_pivot, pre_owned_pivot, pop_scale)  # Smart
    
    baselines = [base1, base2, base3]
    
    # Get ensemble weights via fold validation
    weights = time_series_cv_ensemble(train, baselines)
    print(f"ğŸ“Š Ensemble Weights: {weights}")
    
    # Final prediction: weighted blend
    final_pred_raw = (
        weights[0] * base1 +
        weights[1] * base2 +
        weights[2] * base3
    )
    
    # === FINAL ZERO LOGIC: Only predict >0 if sector was active in last 3 months ===
    last_3_active = train[train['time'] >= 64].groupby('sector_num')['amount_new_house_transactions'].sum() > 0
    last_3_active = last_3_active.reindex(range(1, 97), fill_value=False)
    
    # Apply zero logic
    final_pred = []
    for sector in range(1, 97):
        if last_3_active[sector]:
            final_pred.append(final_pred_raw[sector - 1])
        else:
            final_pred.append(0.0)
    
    # Create submission
    submission = []
    for _, row in test_raw.iterrows():
        test_id = row['id']
        sector_num = int(test_id.split('_')[1].split()[1])
        pred = final_pred[sector_num - 1]
        submission.append({'id': test_id, 'amount_new_house_transactions': max(0, pred)})
    
    submission_df = pd.DataFrame(submission)
    submission_df.to_csv('submission.csv', index=False)
    
    # Stats
    zeros = (submission_df['amount_new_house_transactions'] == 0).sum()
    print(f"âœ… Final submission saved! Zeros: {zeros}")
    print(f"ğŸ“ˆ Range: {submission_df['amount_new_house_transactions'].min():.2f} to "
          f"{submission_df['amount_new_house_transactions'].max():.2f}")
    print(submission_df.head(10))
    
    return submission_df

if __name__ == "__main__":
    submission = main()


df = pd.read_csv("/kaggle/working/submission.csv")


df.head(50)


df.isnull().sum()




