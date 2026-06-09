# ==============================================================================
# M5 Forecasting - 最终版 v6.0 (基于专业解决方案)
#
# 核心策略:
# 1. 预处理: 稳定、内存优化的 Pandas (CPU)
# 2. 特征工程: 对高基数特征 'item_id' 进行频率编码，从根本上解决 bin 数量问题。
# 3. 参数优化: 采纳官方建议，为 GPU 设置 'max_bin': 63。
# 4. 训练: 在 GPU 上高速训练 LightGBM 模型。
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
# Step 3: Model Training (30 Models: Store + Category)
# ==============================================================================
import xgboost as xgb
from tqdm.auto import tqdm

print("\n--- Step 3: Training 30 Models (Store-Category) ---")

# 1. 特征列表 (保持不变)
features = [
    'item_id_freq', 'dept_id', 'cat_id', 'store_id', 'state_id',
    'wday', 'month', 'year',
    'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
    'snap_CA', 'snap_TX', 'snap_WI',
    'sell_price',
    'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28'
]

# 2. 划分训练集 (保持不变)
train_df = df[df['d'] <= 1913].copy()

# 3. XGBoost 参数 (保持不变)
params = {
    'objective': 'reg:tweedie',
    'eval_metric': 'rmse',
    'tree_method': 'gpu_hist',
    'tweedie_variance_power': 1.1,
    'eta': 0.05,
    'max_depth': 8,
    'min_child_weight': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'verbosity': 0,
    'enable_categorical': True,
    'max_bin': 63,
}

# 4. 类别特征 (保持不变)
categorical_features_for_xgb = [
    'dept_id', 'cat_id', 'store_id', 'state_id',
    'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'
]

# ========================== 核心修正 START ==========================

# --> 修正: 定义类别列表，用于嵌套循环
STORES = sales['store_id'].cat.categories.tolist()
CATS = sales['cat_id'].cat.categories.tolist()

# 5. 按店铺和类别训练并保存模型
models = {}
# --> 修正: 改为嵌套循环，训练30个模型
for store in tqdm(STORES, desc="Training per store"):
    for cat in tqdm(CATS, desc=f"Training categories for {store}", leave=False):
        print(f"\n▶▶▶ Training model for: {store} - {cat} ...")
        
        # 5.1 准备训练数据 (按store和cat筛选)
        mask = (train_df['store_id'] == store) & (train_df['cat_id'] == cat)
        df_train_group = train_df[mask]
        
        # 如果某个组合没有数据，则跳过
        if df_train_group.empty:
            print(f"  --> Skipping {store}-{cat}, no data.")
            continue

        X_train = df_train_group[features]
        y_train = df_train_group['sales']
        
        # 确保数据类型正确
        for col in categorical_features_for_xgb:
            X_train[col] = X_train[col].astype("category")

        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        
        # 5.2 训练模型
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train')],
            verbose_eval=200
        )

        model_key = f"{store}_{cat}"
        models[model_key] = model
        
        # 5.3 打印摘要
        print(f"  ✔️ Model for {model_key} trained.")
        
        # 5.4 保存模型 (文件名包含store和cat)
        out_path = f"/kaggle/working/model_{model_key}.json"
        model.save_model(out_path)
        print(f"     - Model saved to `{out_path}`")

# ========================== 核心修正 END ============================

print("\n✅ All 30 models trained and saved to /kaggle/working/.")


# ==============================================================================
# Step 4: Incremental Low-Memory Forecasting & Submission (30 Models) - 修正版
# ==============================================================================
import pandas as pd
import numpy as np
import gc
from tqdm.auto import tqdm
import xgboost as xgb
import os  # <-- 核心修正：导入os模块

# --- A) 加载已训练的30个XGBoost模型 ---
print("\n--- Loading 30 XGBoost Models ---")
models = {}
STORES = df['store_id'].cat.categories.tolist()
CATS = df['cat_id'].cat.categories.tolist()

for store_id in STORES:
    for cat_id in CATS:
        model_key = f"{store_id}_{cat_id}"
        model_path = f"/kaggle/working/model_{model_key}.json"
        if os.path.exists(model_path):
            booster = xgb.Booster()
            booster.load_model(model_path)
            models[model_key] = booster
            print(f"  Loaded model for {model_key}")
        else:
            print(f"  WARNING: Model for {model_key} not found at {model_path}")

# --- B) 预测循环 (同样需要按store和cat进行) ---
# ... (之前的数据准备代码，如加载df_for_pred_init, hist, static_df, calendar, prices等保持不变) ...

all_preds_df = pd.DataFrame() # 用于收集所有预测结果

for d_pred in tqdm(range(1914, 1942), desc="Forecasting all days"):
    # ... (每天的特征生成代码，如today的构建、合并calendar/price/lags/rolling_means等保持不变) ...
    
    # --- 关键的预测部分 ---
    X_pred_today = today[features]
    y_pred = np.zeros(len(today), dtype=float)

    # --> 修正: 嵌套循环，使用对应的模型进行预测
    for store_id in STORES:
        for cat_id in CATS:
            model_key = f"{store_id}_{cat_id}"
            
            if model_key in models:
                model = models[model_key]
                
                mask = (today['store_id'] == store_id) & (today['cat_id'] == cat_id)
                
                if mask.any():
                    X_test_group = X_pred_today[mask]
                    
                    for col in categorical_features_for_xgb:
                        if col in X_test_group.columns:
                            X_test_group[col] = X_test_group[col].astype('category')

                    dpred = xgb.DMatrix(X_test_group, enable_categorical=True)
                    
                    y = model.predict(dpred)
                    y[y < 0] = 0
                    
                    y_pred[mask] = y

    # --- 后续逻辑 ---
    today_preds = today[['id']].copy()
    today_preds['sales'] = y_pred
    today_preds['d'] = d_pred
    all_preds_df = pd.concat([all_preds_df, today_preds])

    new_hist = pd.DataFrame({'id': all_ids, 'd': d_pred, 'sales': y_pred})
    hist = pd.concat([hist, new_hist])
    hist = hist[hist['d'] >= (d_pred - MAX_LAG + 1)]
    
    gc.collect()

# --- C) 生成提交文件 ---
print("\nCreating submission file...")
submission_df = pd.read_csv(f"{DATA_PATH}sample_submission.csv")

all_preds_df['F'] = 'F' + (all_preds_df['d'] - 1913).astype(str)
submission_map = all_preds_df.pivot(index='id', columns='F', values='sales').reset_index()

validation_submission = submission_df[submission_df['id'].str.contains('validation')][['id']].merge(submission_map, on='id', how='left')
evaluation_submission = validation_submission.copy()
evaluation_submission['id'] = evaluation_submission['id'].str.replace('validation', 'evaluation')

final_submission = pd.concat([validation_submission, evaluation_submission])
final_submission = final_submission[submission_df.columns]
final_submission.fillna(0, inplace=True)

final_submission.to_csv('submission.csv', index=False)
print("✅ submission.csv generated (shape: {})".format(final_submission.shape))
display(final_submission.head())


# ==============================================================================
#  Visualization and final result (XGBoost Version for 30 Models)
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from pathlib import Path

plt.style.use('seaborn-whitegrid')
sns.set_palette('colorblind')

# --- 1) 加载 ground truth ---
try:
    # 假设 sales_eval 是从原始数据加载的完整 DataFrame
    sales_eval = pd.read_csv(f"{DATA_PATH}sales_train_evaluation.csv")
    print("Loaded sales_train_evaluation.csv")
except FileNotFoundError:
    sales_eval = None
    print("Warning: cannot load sales_train_evaluation.csv")

# --- 2) 加载预测结果 ---
# submission 是您上一步生成的包含所有预测的 DataFrame
preds_eval = submission.reset_index().copy()
preds_eval['id'] = preds_eval['id'].str.replace('_validation', '_evaluation')

# --- 3) 绘图函数 (保持不变) ---
def plot_forecast_vs_actual(item_id, store_id, preds_df, truth_df):
    # ... (这个函数无需修改) ...

# --- 4) 示例图 (保持不变) ---
print("Example Forecast vs Actual:")
plot_forecast_vs_actual("HOBBIES_1_008", "CA_1", preds_eval, sales_eval)
plot_forecast_vs_actual("HOUSEHOLD_1_118", "CA_3", preds_eval, sales_eval)
plot_forecast_vs_actual("FOODS_3_586", "TX_2", preds_eval, sales_eval)


# ========================== 核心修正 START ==========================
# --- 5) 特征重要性 (为指定的 store-cat 模型) ---
# 您可以修改下面的 store_to_plot 和 cat_to_plot 来查看不同模型的重要性
store_to_plot = "CA_1"
cat_to_plot = "HOBBIES"
model_key_to_plot = f"{store_to_plot}_{cat_to_plot}"

if model_key_to_plot in models:
    fig, ax = plt.subplots(figsize=(8, 6))
    xgb.plot_importance(
        models[model_key_to_plot],
        ax=ax,
        max_num_features=15,
        importance_type="gain",
        title=f"Feature Importance for {model_key_to_plot}"
    )
    plt.tight_layout()
    plt.show()
else:
    print(f"Model for {model_key_to_plot} not found in the 'models' dictionary.")
# ========================== 核心修正 END ============================


# ==============================================================================
#  Detailed Visualizations (XGBoost Version)
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb  # 确保导入了XGBoost
from pathlib import Path

plt.style.use('seaborn-whitegrid')
sns.set_palette('tab10')

# --- 1) 加载 ground truth (days 1914–1941) ---
try:
    sales_eval = pd.read_csv(Path(DATA_PATH) / "sales_train_evaluation.csv")
    print("Loaded sales_train_evaluation.csv")
except FileNotFoundError:
    sales_eval = None
    print("Warning: cannot load sales_train_evaluation.csv")

# --- 2) 准备 evaluation-version 预测 ---
preds_eval = submission.reset_index().copy()
preds_eval['id'] = preds_eval['id'].str.replace('_validation', '_evaluation')

# --- 3) 绘图函数 (保持不变) ---
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
    # 修正: truth_df的id列已经是索引了，需要先reset_index()或者直接用index查询
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

# --- 4) 示例图 (保持不变) ---
print("Example Forecast vs Actual:")
plot_forecast_vs_actual("HOBBIES_1_008", "CA_1", preds_eval, sales_eval)
plot_forecast_vs_actual("HOUSEHOLD_1_118", "CA_3", preds_eval, sales_eval)
plot_forecast_vs_actual("FOODS_3_586",     "TX_2", preds_eval, sales_eval)


# --- 5) 绘制 Daily total sales & residuals (保持不变) ---
dates = pd.date_range('2016-04-25', periods=28)
daily_pred = preds_eval[[f'F{i}' for i in range(1,29)]].sum().values
daily_true = np.array([sales_eval[f'd_{d}'].sum() for d in range(1914,1942)])

plt.figure(figsize=(12,4))
plt.plot(dates, daily_true,  label='Actual Total Sales',  marker='o')
plt.plot(dates, daily_pred,  label='Forecast Total Sales', marker='x')
plt.title('Daily Total Sales: Actual vs Forecast (Validation)')
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# --- 7) Feature Importance (为指定的 store-cat 模型) ---
# 您可以修改下面的 store_to_plot 和 cat_to_plot 来查看不同模型的重要性
store_to_plot = 'CA_1'
cat_to_plot = 'FOODS' # 换一个类别作为示例
model_key_to_plot = f"{store_to_plot}_{cat_to_plot}"

if model_key_to_plot in models:
    fig, ax = plt.subplots(figsize=(8, 6))
    xgb.plot_importance(
        models[model_key_to_plot],
        ax=ax,
        max_num_features=10,
        importance_type='gain',
        title=f'{model_key_to_plot} Feature Importance'
    )
    plt.tight_layout()
    plt.show()
else:
    print(f"No model for {model_key_to_plot} was found in the 'models' dictionary.")


