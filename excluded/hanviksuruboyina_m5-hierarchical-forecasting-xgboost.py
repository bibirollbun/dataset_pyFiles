# ==============================================================================
# M5 Forecasting - æœ€ç»ˆç‰ˆ v6.0 (åŸºäº�ä¸“ä¸šè§£å†³æ–¹æ¡ˆ)
#
# æ ¸å¿ƒç­–ç•¥:
# 1. é¢„å¤„ç�†: ç¨³å®šã€�å†…å­˜ä¼˜åŒ–çš„ Pandas (CPU)
# 2. ç‰¹å¾�å·¥ç¨‹: å¯¹é«˜åŸºæ•°ç‰¹å¾� 'item_id' è¿›è¡Œé¢‘ç�‡ç¼–ç �ï¼Œä»�æ ¹æœ¬ä¸Šè§£å†³ bin æ•°é‡�é—®é¢˜ã€‚
# 3. å�‚æ•°ä¼˜åŒ–: é‡‡çº³å®˜æ–¹å»ºè®®ï¼Œä¸º GPU è®¾ç½® 'max_bin': 63ã€‚
# 4. è®­ç»ƒ: åœ¨ GPU ä¸Šé«˜é€Ÿè®­ç»ƒ LightGBM æ¨¡å�‹ã€‚
# ==============================================================================

# ------------------------------------------------------------------------------
# Step 1: Setup and Data Loading
# ------------------------------------------------------------------------------
print("--- Step 1: Loading Data and Initial Setup ---")
import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore')

# Function to downcast dtypes for memory saving
def downcast(df):
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    for i, t in enumerate(types):
        if 'int' in str(t):
            if df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:
                df[cols[i]] = df[cols[i]].astype(np.int32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.int64)
        elif 'float' in str(t):
            if df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:
                df[cols[i]] = df[cols[i]].astype(np.float32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.float64)
        elif 'object' in str(t):
            if cols[i] != 'date':
                df[cols[i]] = df[cols[i]].astype('category')
    return df

# Load data
DATA_PATH = "/kaggle/input/m5-forecasting-accuracy/"
calendar = pd.read_csv(f"{DATA_PATH}calendar.csv")
prices = pd.read_csv(f"{DATA_PATH}sell_prices.csv")
sales = pd.read_csv(f"{DATA_PATH}sales_train_validation.csv")

# Perform memory optimization
calendar = downcast(calendar)
prices = downcast(prices)
sales = downcast(sales)

print("Data loaded and downcasted.")
gc.collect()




# ------------------------------------------------------------------------------
# Step 2: Data Preprocessing and Feature Engineering
# ------------------------------------------------------------------------------
print("\n--- Step 2: Creating Features ---")

# Melt sales data from wide to long format
df = pd.melt(sales,
             id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
             var_name='d',
             value_name='sales')

# Merge with calendar and prices
df = pd.merge(df, calendar, on='d', how='left')
df = pd.merge(df, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
del calendar, prices
gc.collect()

# Create time-based features
df['d'] = df['d'].str.extract(r'(\d+)').astype(np.int16)
df['wday'] = df['wday'].astype(np.int8)
df['month'] = df['month'].astype(np.int8)
df['year'] = df['year'].astype(np.int16)

# ==============================================================================
# ======================= APPLYING THE NEW STRATEGY ============================
# ==============================================================================
# Create Frequency Encoding for 'item_id'
print("  Creating Frequency Encoding for 'item_id'...")
item_id_freq_map = df['item_id'].value_counts().to_dict()
df['item_id_freq'] = df['item_id'].map(item_id_freq_map).astype(np.int32)
# ==============================================================================

# Create lag and rolling window features
print("  Creating lag and rolling window features...")
lags = [7, 14, 28]
for lag in tqdm(lags, desc="Creating Lags"):
    df[f'sales_lag_{lag}'] = df.groupby(['id'])['sales'].shift(lag).astype(np.float32)

windows = [7, 14, 28]
for window in tqdm(windows, desc="Creating Rolling Means"):
    df[f'rolling_mean_{window}'] = df.groupby(['id'])['sales'].transform(
        lambda x: x.shift(28).rolling(window).mean()
    ).astype(np.float32)

print("Feature engineering complete.")
gc.collect()





df.head()


# ==============================================================================
# Part 1: Exploratory Data Analysis (EDA) Visualizations
# Place this block after Step 2 and before Step 3.
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- Generating EDA Visualizations ---")

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('colorblind')

# --- 1.1 Overall Daily Sales Trend ---
print("  Plotting overall daily sales trend...")
# Aggregate total daily sales across all products
daily_sales = df.groupby('date')['sales'].sum()
daily_sales.index = pd.to_datetime(daily_sales.index)

plt.figure(figsize=(15, 6))
daily_sales.plot(title='Overall Daily Sales Trend', color='darkblue')
plt.ylabel('Total Units Sold')
plt.xlabel('Date')
# Highlight the Christmas period to show holiday effects
plt.axvspan(xmin=pd.to_datetime('2013-12-01'), xmax=pd.to_datetime('2013-12-31'), color='red', alpha=0.2, label='Christmas Period')
plt.axvspan(xmin=pd.to_datetime('2014-12-01'), xmax=pd.to_datetime('2014-12-31'), color='red', alpha=0.2)
plt.axvspan(xmin=pd.to_datetime('2015-12-01'), xmax=pd.to_datetime('2015-12-31'), color='red', alpha=0.2)
plt.legend()
plt.show()


# --- 1.2 Weekly Sales Seasonality ---
print("  Plotting weekly sales seasonality...")
# Aggregate average sales by day of the week
weekly_sales = df.groupby('wday')['sales'].mean()
weekday_map = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
weekly_sales.index = weekly_sales.index.map(weekday_map)
# Order the days correctly
ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_sales = weekly_sales.reindex(ordered_days)

plt.figure(figsize=(10, 5))
weekly_sales.plot(kind='bar', title='Average Sales by Day of Week', color='skyblue', edgecolor='black')
plt.ylabel('Average Sales')
plt.xlabel('Day of Week')
plt.xticks(rotation=45)
plt.show()


# --- 1.3 Sales Comparison by Store ---
print("  Plotting sales comparison by store...")
# Aggregate total sales by store ID
store_sales = df.groupby('store_id')['sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
store_sales.plot(kind='bar', title='Total Sales by Store', color='mediumseagreen', edgecolor='black')
plt.ylabel('Total Units Sold')
plt.xlabel('Store ID')
plt.xticks(rotation=45)
plt.show()


# ==============================================================================
# Step 3: Model Training (Improved XGBoost Version)
# ==============================================================================
import xgboost as xgb
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

print("\n--- Step 3: Training Models (Improved XGBoost Version) ---")

# 1. Feature list
features = [
    'item_id_freq', 'dept_id', 'cat_id', 'store_id', 'state_id',
    'wday', 'month', 'year',
    'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
    'snap_CA', 'snap_TX', 'snap_WI',
    'sell_price',
    'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28'
]

# 2. Train split
train_df = df[df['d'] <= 1913].copy()

# 3. Tuned XGBoost Parameters
params = {
    'objective': 'reg:tweedie',
    'eval_metric': 'rmse',
    'tree_method': 'gpu_hist',
    'tweedie_variance_power': 1.1,
    
    # Learning behavior
    'eta': 0.015,
    'max_depth': 8,
    'min_child_weight': 100,
    
    # Regularization
    'lambda': 0.3,  # L2
    'alpha': 0.15,  # L1
    
    # Sampling (randomness)
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.9,
    
    # Misc
    'seed': 42,
    'verbosity': 0,
    'enable_categorical': True,
    'max_bin': 128,  # finer split bins for continuous features
}

categorical_features = [
    'dept_id', 'cat_id', 'store_id', 'state_id',
    'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'
]

# 4. Per-store training
models = {}
for store in tqdm(sales['store_id'].cat.categories, desc="Training per store"):
    if store == 'CA_1':  # for example
        print(f"\nâ–¶â–¶â–¶ Training model for store: {store} ...")
        
        # Prepare data
        df_store = train_df[train_df['store_id'] == store].copy()
        X = df_store[features]
        y = df_store['sales']

        for col in categorical_features:
            X[col] = X[col].astype('category')

        # Split validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, shuffle=True
        )
        
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

        # Train with early stopping
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=4000,
            evals=[(dtrain, 'train'), (dval, 'valid')],
            early_stopping_rounds=100,
            verbose_eval=200
        )

        models[store] = model

        print(f"  âœ”ï¸� Store {store} model trained.")
        print(f"     - Best iteration: {model.best_iteration}")
        print(f"     - Best RMSE: {model.best_score:.5f}")

        # Save model
        out_path = f"/kaggle/working/model_store_{store}_xgb.json"
        model.save_model(out_path)
        print(f"     - Model saved to `{out_path}`")

print("\nâœ… All Improved XGBoost models trained and saved to /kaggle/working/.")



# ================================================================================
# Step 4: Incremental Low-Memory Forecasting & Submission Export (XGBoost Version)
# ================================================================================ 
import pandas as pd
import numpy as np
import gc
from tqdm.auto import tqdm
import xgboost as xgb # ç¡®ä¿�å¯¼å…¥äº†XGBoost

# --- A) åŠ è½½å·²è®­ç»ƒçš„XGBoostæ¨¡å�‹ ---
print("\n--- Loading XGBoost Models ---")
models = {}
for store_id in sales['store_id'].cat.categories:
    # ========================== æ ¸å¿ƒä¿®æ­£ START ==========================
    if store_id == 'CA_1':
        print(store_id)
        model_path = f"/kaggle/working/model_store_{store_id}_xgb.json"
        # ========================== æ ¸å¿ƒä¿®æ­£ END ============================
        booster = xgb.Booster()
        booster.load_model(model_path)
        models[store_id] = booster
        print(f"  Loaded model for {store_id}")


# --- B) å‡†å¤‡åˆ�å§‹DataFrameå’Œé�™æ€�ç‰¹å¾� (ä¸�å�Ÿä»£ç �é€»è¾‘ä¸€è‡´) ---
df_for_pred_init   = df[df['d'] <= 1913].copy()
static_feature_list = [
    'item_id', 'item_id_freq',
    'dept_id', 'cat_id', 'store_id', 'state_id'
]

# ç‰¹å¾�å’Œç±»åˆ«åˆ—è¡¨å¿…é¡»ä¸�è®­ç»ƒæ—¶å®Œå…¨ä¸€è‡´
features = [
    'item_id_freq','dept_id','cat_id','store_id','state_id',
    'wday','month','year',
    'event_name_1','event_type_1','event_name_2','event_type_2',
    'snap_CA','snap_TX','snap_WI',
    'sell_price',
    'sales_lag_7','sales_lag_14','sales_lag_28',
    'rolling_mean_7','rolling_mean_14','rolling_mean_28'
]

categorical_features_for_xgb = [
    'dept_id','cat_id','store_id','state_id',
    'event_name_1','event_type_1','event_name_2','event_type_2'
]


# --- C) å¸¸é‡�å’Œè·¯å¾„ (ä¸�å�Ÿä»£ç �é€»è¾‘ä¸€è‡´) ---
DATA_PATH    = "/kaggle/input/m5-forecasting-accuracy/"
MAX_LAG      = 28
VALID_START  = 1914; VALID_END = 1942
EVAL_START   = 1942; EVAL_END  = 1970
ALL_DAYS     = list(range(VALID_START, EVAL_END))


# --- D) æ�„å»ºæ��äº¤æ–‡ä»¶éª¨æ�¶ (ä¸�å�Ÿä»£ç �é€»è¾‘ä¸€è‡´) ---
sample_sub = pd.read_csv(f"{DATA_PATH}sample_submission.csv").set_index('id')
for f in [f"F{i}" for i in range(1,29)]:
    sample_sub[f] = 0
submission = sample_sub.copy()


# --- E) å‡†å¤‡æ»šåŠ¨å�†å�²å’Œé�™æ€�è¡¨ (ä¸�å�Ÿä»£ç �é€»è¾‘ä¸€è‡´) ---
hist = df_for_pred_init[['id','d','sales']].copy()
hist = hist[hist['d'] >= (VALID_START - MAX_LAG)].reset_index(drop=True)

all_ids   = df_for_pred_init['id'].unique()
static_df = df_for_pred_init[['id'] + static_feature_list].drop_duplicates()


# --- F) åŠ è½½æ—¥å�†å’Œä»·æ ¼æ•°æ�® (ä¸�å�Ÿä»£ç �é€»è¾‘ä¸€è‡´) ---
calendar = pd.read_csv(f"{DATA_PATH}calendar.csv")
calendar['d'] = calendar['d'].str.replace('d_','').astype(int)
prices   = pd.read_csv(f"{DATA_PATH}sell_prices.csv")


# --- G) æ¯�æ—¥å¢�é‡�é¢„æµ‹ (æ ¸å¿ƒé¢„æµ‹é€»è¾‘ä¿®æ”¹) ---
for d_pred in tqdm(ALL_DAYS, desc="Forecasting all days"):
    # G.1 & G.2 & G.3 & G.4: ç‰¹å¾�ç”Ÿæˆ� (ä¸�å�Ÿä»£ç �é€»è¾‘ä¸€è‡´)
    today = pd.DataFrame({'id': all_ids, 'd': d_pred})
    today = today.merge(static_df, on='id', how='left')
    
    cf = calendar[calendar['d'] == d_pred]
    today = today.merge(cf, on='d', how='left')

    wm = cf['wm_yr_wk'].iloc[0]
    pf = prices[prices['wm_yr_wk'] == wm][['store_id','item_id','sell_price']]
    today = today.merge(pf, on=['store_id','item_id'], how='left')

    for lag in (7,14,28):
        prev = hist[hist['d'] == d_pred-lag][['id','sales']].rename(columns={'sales': f'sales_lag_{lag}'})
        today = today.merge(prev, on='id', how='left')
    
    for win in (7,14,28):
        mask = (hist['d'] >= d_pred-win) & (hist['d'] < d_pred)
        rolling = (hist.loc[mask].groupby('id')['sales'].mean().rename(f'rolling_mean_{win}').reset_index())
        today = today.merge(rolling, on='id', how='left')

    # G.5: ç¡®ä¿�ç±»åˆ«ç‰¹å¾�çš„dtypeä¸�è®­ç»ƒæ—¶ä¸€è‡´
    for col in categorical_features_for_xgb:
        today[col] = today[col].astype('category')
        today[col] = today[col].cat.set_categories(train_df[col].cat.categories)

    # G.6: æŒ‰åº—é“ºè¿›è¡Œé¢„æµ‹ (ä½¿ç”¨XGBoost)
    X_pred_today = today[features]
    y_pred = np.zeros(len(today), dtype=float)
    
    for store_id, model in models.items():
        mask = today['store_id'] == store_id
        if mask.any():
            # ä¸ºå½“å‰�åº—é“ºçš„æ•°æ�®åˆ›å»ºDMatrix
            dpred = xgb.DMatrix(X_pred_today.loc[mask], enable_categorical=True)
            # è¿›è¡Œé¢„æµ‹
            y = model.predict(dpred)
            y[y < 0] = 0 # ç¡®ä¿�é¢„æµ‹å€¼é��è´Ÿ
            y_pred[mask] = y

    # G.7 & G.8: å¡«å……æ��äº¤æ–‡ä»¶å’Œæ›´æ–°å�†å�²è®°å½• (ä¸�å�Ÿä»£ç �é€»è¾‘ä¸€è‡´)
    if d_pred < VALID_END:
        col = f"F{d_pred - VALID_START + 1}"
        submission.loc[today['id'], col] = y_pred
    else:
        col = f"F{d_pred - EVAL_START + 1}"
        idx = today['id'].str.replace('_validation', '_evaluation')
        submission.loc[idx, col] = y_pred

    new_hist = pd.DataFrame({'id': all_ids, 'd': d_pred, 'sales': y_pred})
    hist = pd.concat([hist, new_hist], ignore_index=True)
    hist = hist[hist['d'] >= (d_pred - MAX_LAG)].reset_index(drop=True)
    gc.collect()

# --- H) å¯¼å‡ºæ��äº¤æ–‡ä»¶ (ä¸�å�Ÿä»£ç �é€»è¾‘ä¸€è‡´) ---
submission.reset_index().to_csv('submission.csv', index=False)
print("âœ… submission.csv generated (shape: {})".format(submission.shape))
submission.head()


# ==============================================================================
#  Visualization and final result: Forecast vs Actual & Feature Importance
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('colorblind')

# --- 1) Load ground truth (days 1914â€“1941) ---
try:
    sales_eval = pd.read_csv(Path(DATA_PATH) / "sales_train_evaluation.csv")
    print("Loaded sales_train_evaluation.csv")
except FileNotFoundError:
    sales_eval = None
    print("Warning: cannot load sales_train_evaluation.csv")

# --- 2) Prepare evaluation-version predictions ---
# å°† submission çš„ç´¢å¼• 'id' å�˜æˆ�åˆ—ï¼Œå†�å�š _validation -> _evaluation æ›¿æ�¢
preds_eval = submission.reset_index().copy()
preds_eval['id'] = preds_eval['id'].str.replace('_validation', '_evaluation')

# --- 3) Plotting function ---
def plot_forecast_vs_actual(item_id, store_id, preds_df, truth_df):
    if truth_df is None:
        print("No ground truth available.")
        return
    
    eval_id = f"{item_id}_{store_id}_evaluation"
    row = preds_df.loc[preds_df['id'] == eval_id]
    if row.empty:
        print(f"Missing forecast for {eval_id}")
        return
    y_pred = row[[f"F{i}" for i in range(1,29)]].values.flatten().astype(float)
    
    valid_id = f"{item_id}_{store_id}_validation"
    if valid_id not in truth_df['id'].values:
        print(f"Missing actuals for {valid_id}")
        return
    actual_cols = [f"d_{d}" for d in range(1914, 1942)]
    y_true = truth_df.loc[truth_df['id'] == valid_id, actual_cols].values.flatten().astype(float)
    
    dates = pd.date_range("2016-04-25", periods=28)
    plt.figure(figsize=(12,5))
    plt.plot(dates, y_true, label="Actual", marker="o")
    plt.plot(dates, y_pred, label="Forecast", marker="x")
    plt.title(f"{item_id} @ {store_id}: Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- 4) Example plots ---
print("Example Forecast vs Actual:")
plot_forecast_vs_actual("HOBBIES_1_008", "CA_1", preds_eval, sales_eval)
# plot_forecast_vs_actual("HOUSEHOLD_1_118", "CA_3", preds_eval, sales_eval)
# plot_forecast_vs_actual("FOODS_3_586",     "TX_2", preds_eval, sales_eval)

# 5) Feature Importance (ä½¿ç”¨XGBoost)
store = "CA_1"
if store in models:
    fig, ax = plt.subplots(figsize=(8, 6)) # ä¸ºXGBoostç»˜å›¾åˆ›å»ºfigå’Œax
    xgb.plot_importance(
        models[store],
        ax=ax, # ä¼ é€’ax
        max_num_features=15,
        importance_type="gain",
        title=f"Feature Importance for Store {store}"
    )
    plt.tight_layout()
    plt.show()
else:
    print(f"Model for {store} not found.")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from pathlib import Path

plt.style.use('seaborn-whitegrid')
sns.set_palette('tab10')

# --- 1) Load ground truth (validation: d_1914â€“d_1941) ---
try:
    sales_eval = pd.read_csv(Path(DATA_PATH) / "sales_train_evaluation.csv")
    print("âœ… Loaded sales_train_evaluation.csv")
except FileNotFoundError:
    sales_eval = None
    print("âš ï¸� Warning: cannot load sales_train_evaluation.csv")

# --- 2) Prepare validation-version predictions (âœ… Filter only matching IDs) ---
preds_valid = submission.reset_index().copy()

# Keep only ids that exist in ground truth
valid_ids = sales_eval['id'].values
preds_valid = preds_valid[preds_valid['id'].isin(valid_ids)].copy()

print(f"âœ… Filtered preds_valid: {len(preds_valid)} rows (should match {len(sales_eval)} truth rows)")

# --- 3) Plot function (Forecast vs Actual) ---
def plot_forecast_vs_actual(item_id, store_id, preds_df, truth_df):
    if truth_df is None:
        print("No ground truth available.")
        return
    
    valid_id = f"{item_id}_{store_id}_validation"
    row = preds_df.loc[preds_df['id'] == valid_id]
    if row.empty:
        print(f"âš ï¸� Missing forecast for {valid_id}")
        return
    y_pred = row[[f"F{i}" for i in range(1, 29)]].values.flatten().astype(float)

    if valid_id not in truth_df['id'].values:
        print(f"âš ï¸� Missing actuals for {valid_id}")
        return
    
    actual_cols = [f"d_{d}" for d in range(1914, 1942)]
    y_true = truth_df.loc[truth_df['id'] == valid_id, actual_cols].values.flatten().astype(float)
    
    dates = pd.date_range("2016-04-25", periods=28)
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="Actual", marker="o")
    plt.plot(dates, y_pred, label="Forecast", marker="x")
    plt.title(f"{item_id} @ {store_id}: Forecast vs Actual (Validation)")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- 4) Example item-level plots ---
print("ğŸ“ˆ Example Forecast vs Actual:")
plot_forecast_vs_actual("HOBBIES_1_008", "CA_1", preds_valid, sales_eval)
plot_forecast_vs_actual("HOUSEHOLD_1_118", "CA_3", preds_valid, sales_eval)
plot_forecast_vs_actual("FOODS_3_586",     "TX_2", preds_valid, sales_eval)

# --- 5) Daily Total Sales (Validation Period) ---
dates = pd.date_range('2016-04-25', periods=28)

# Daily forecast sum (F1â€“F28 across all items)
daily_pred = preds_valid[[f'F{i}' for i in range(1, 29)]].sum().values

# Daily actual sum (d_1914â€“d_1941 across all items)
daily_true = np.array([sales_eval[f'd_{d}'].sum() for d in range(1914, 1942)])

plt.figure(figsize=(12, 4))
plt.plot(dates, daily_true, label='Actual Total Sales', marker='o')
plt.plot(dates, daily_pred, label='Forecast Total Sales', marker='x')
plt.title('Daily Total Sales: Actual vs Forecast (Validation)')
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# --- 6) Optional: Residuals Plot (Forecast - Actual per day) ---
residuals = daily_pred - daily_true
plt.figure(figsize=(12, 3))
plt.bar(dates, residuals, color=np.where(residuals > 0, 'tomato', 'skyblue'))
plt.axhline(0, color='black', linewidth=1)
plt.title("Daily Forecast Bias (Forecast - Actual)")
plt.ylabel("Residual Units")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 7) XGBoost Feature Importance (optional) ---
store = 'CA_1'
if store in models:
    fig, ax = plt.subplots(figsize=(8, 6))
    xgb.plot_importance(
        models[store],
        ax=ax,
        max_num_features=10,
        importance_type='gain',
        title=f"Feature Importance for Store {store}"
    )
    plt.tight_layout()
    plt.show()
else:
    print(f"âš ï¸� No model found for {store}.")



# Check matching IDs between forecast and truth
valid_ids_pred = set(preds_valid['id'])
valid_ids_true = set(sales_eval['id'])

print("Pred IDs:", len(valid_ids_pred))
print("True IDs:", len(valid_ids_true))
print("Common IDs:", len(valid_ids_pred.intersection(valid_ids_true)))

# Should be exactly equal
if len(valid_ids_pred.intersection(valid_ids_true)) < len(valid_ids_pred):
    missing = valid_ids_pred - valid_ids_true
    print("âš ï¸� Some forecast IDs missing in truth, e.g.:", list(missing)[:5])
else:
    print("âœ… All forecast IDs have matching truth entries.")


