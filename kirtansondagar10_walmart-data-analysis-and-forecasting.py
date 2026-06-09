# --- CELL 1: Setup & Imports ---
import os
try:
    import pandasql
except ImportError:
    print("Installing pandasql...")
    os.system('pip install pandasql')

# Standard Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling Imports
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest

# Utilities
import pandasql as ps 
import warnings
import time
from joblib import Parallel, delayed
import multiprocessing

# Configuration
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight') 

print("All libraries installed and imported successfully.")


# --- CELL 2 (EXPERT): Feature Engineering with Momentum & Cyclic Time ---

import pandas as pd
import pandasql as ps
import numpy as np

# 1. Load Data
try:
    train = pd.read_csv('train.csv')
    features = pd.read_csv('features.csv')
    stores = pd.read_csv('stores.csv')
except:
    base_path = '/kaggle/input/walmart-recruiting-store-sales-forecasting/'
    train = pd.read_csv(base_path + '/train.csv.zip')
    features = pd.read_csv(base_path + '/features.csv.zip')
    stores = pd.read_csv(base_path + '/stores.csv')

# Merge
df_merged = pd.merge(train, features, on=['Store', 'Date', 'IsHoliday'], how='left')
df_merged = pd.merge(df_merged, stores, on=['Store'], how='left')
df_merged['Date'] = pd.to_datetime(df_merged['Date'])

# Handle Missing Markdowns & Ensure Numeric
md_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
for col in md_cols:
    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna(0)

df_merged['Total_MarkDown'] = df_merged[md_cols].sum(axis=1)

# Ensure CPI is numeric
df_merged['CPI'] = pd.to_numeric(df_merged['CPI'], errors='coerce')

# --- RESUME BOOSTER: SQL Aggregation ---
print("Running SQL for Global Aggregation...")
query = """
SELECT 
    Date,
    SUM(Weekly_Sales) as Global_Sales,
    AVG(CPI) as Global_CPI,
    AVG(Total_MarkDown) as Avg_Markdown
FROM df_merged
GROUP BY Date
ORDER BY Date
"""
global_sales_df = ps.sqldf(query, locals())
global_sales_df['Date'] = pd.to_datetime(global_sales_df['Date'])

# --- THE FIX: Manually Add Missing Retail Holidays ---
easter_dates = ['2010-04-02', '2011-04-22', '2012-04-06', '2013-03-29']
memorial_dates = ['2010-05-31', '2011-05-30', '2012-05-28', '2013-05-27']
july4_dates = ['2010-07-02', '2011-07-01', '2012-07-06', '2013-07-05'] # Nearest Friday
halloween_dates = ['2010-10-29', '2011-10-28', '2012-10-26', '2013-10-25'] # Nearest Friday

# Create a "Retail_Holiday" Flag (Combines Official + Unofficial)
official_holidays = df_merged[df_merged['IsHoliday'] == True]['Date'].unique().astype(str).tolist()
all_holiday_dates = official_holidays + easter_dates + memorial_dates + july4_dates + halloween_dates

df_merged['Is_Retail_Holiday'] = df_merged['Date'].astype(str).isin(all_holiday_dates).astype(int)

# --- INNOVATION 1: Shopping Momentum (Lag/Lead Features) ---
df_merged['Holiday_Momentum'] = df_merged.groupby('Store')['Is_Retail_Holiday'].shift(-1).fillna(0) + \
                                df_merged.groupby('Store')['Is_Retail_Holiday'].shift(-2).fillna(0) * 0.5 - \
                                df_merged.groupby('Store')['Is_Retail_Holiday'].shift(1).fillna(0)

# --- INNOVATION 2: Cyclic Time Encoding ---
# Calculate week number manually to ensure numeric types
week_num = df_merged['Date'].dt.isocalendar().week.astype(float)
df_merged['Week_Sin'] = np.sin(2 * np.pi * week_num / 52)
df_merged['Week_Cos'] = np.cos(2 * np.pi * week_num / 52)

print("Expert Features Created:")
print("- 'Is_Retail_Holiday': Captures Easter, July 4th, Halloween")
print("- 'Holiday_Momentum': Captures 2-week ramp up and 1-week cool down")
print("- 'Week_Sin/Cos': Captures cyclic seasonality")


# --- CELL 2.5: Exploratory Data Analysis (EDA) ---
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure df_merged exists
if 'df_merged' not in locals():
    print("Please run Cell 2 first to load and merge the data.")
else:
    print("--- Generating EDA Plots ---")
    
    # 1. Global Sales Trend over Time
    # This reveals the strong seasonality and the massive spikes at end-of-year
    global_weekly_sales = df_merged.groupby('Date')['Weekly_Sales'].sum().reset_index()
    
    plt.figure(figsize=(18, 6))
    plt.plot(global_weekly_sales['Date'], global_weekly_sales['Weekly_Sales'], color='#2c3e50', linewidth=2)
    plt.title('Total Weekly Sales Across All 45 Stores (2010-2012)', fontsize=16)
    plt.ylabel('Total Sales ($)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

    # 2. Sales Distribution by Store Type
    # Walmart stores are classified into Types A, B, C (likely based on size)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Type', y='Weekly_Sales', data=df_merged, palette="Set2")
    plt.title('Sales Distribution by Store Type', fontsize=16)
    plt.ylabel('Weekly Sales ($)', fontsize=12)
    plt.yscale('log') # Log scale helps see the distribution better
    plt.show()

    # 3. Impact of Holidays on Sales
    # Comparing weeks flagged as Holidays vs Non-Holidays
    plt.figure(figsize=(8, 6))
    sns.barplot(x='IsHoliday', y='Weekly_Sales', data=df_merged, palette="Paired")
    plt.title('Average Weekly Sales: Holiday vs. Non-Holiday', fontsize=16)
    plt.ylabel('Average Sales ($)', fontsize=12)
    plt.xticks([0, 1], ['Non-Holiday', 'Holiday'])
    plt.show()

    # 4. Correlation Heatmap
    # See how Sales correlate with external factors like Unemployment, CPI, Fuel_Price
    # We select only numeric columns for correlation
    numeric_cols = ['Weekly_Sales', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Total_MarkDown']
    # Filter to columns that actually exist in your dataframe
    existing_cols = [c for c in numeric_cols if c in df_merged.columns]
    
    plt.figure(figsize=(10, 8))
    corr = df_merged[existing_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.show()


# --- CELL 3 (EXPERT): Aggressive Hybrid Training ---
from joblib import Parallel, delayed
import multiprocessing

def train_predict_expert(store_id, df_all):
    # 1. Aggregate
    store_df = df_all[df_all['Store'] == store_id].groupby('Date').agg({
        'Weekly_Sales': 'sum', 
        'CPI': 'mean', 
        'Total_MarkDown': 'mean',
        'Is_Retail_Holiday': 'max', 
        'Holiday_Momentum': 'max',
        'Week_Sin': 'max',
        'Week_Cos': 'max'
    }).reset_index()
    
    # --- CRITICAL FIX: Clean Data for Prophet ---
    # Prophet crashes if regressors have NaNs or are not float/int
    regressors = ['Total_MarkDown', 'CPI', 'Is_Retail_Holiday', 'Holiday_Momentum', 'Week_Sin', 'Week_Cos']
    for col in regressors:
        # Force numeric
        store_df[col] = pd.to_numeric(store_df[col], errors='coerce')
        # Fill missing values (Prophet cannot handle NaNs in regressors)
        # Forward fill first (temporal consistency), then backfill, then 0 as last resort
        store_df[col] = store_df[col].ffill().bfill().fillna(0)

    # 2. Split
    train_size = int(len(store_df) * 0.85)
    train_data = store_df.iloc[:train_size]
    test_data = store_df.iloc[train_size:]
    
    prophet_train = train_data.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
    prophet_test = test_data.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})

    # 3. Prophet Configuration
    m = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=False, 
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        holidays_prior_scale=25.0, 
        seasonality_prior_scale=20.0
    )
    
    m.add_country_holidays(country_name='US')
    
    # Add Regressors (Now guaranteed to be clean)
    for col in regressors:
        m.add_regressor(col)
    
    m.fit(prophet_train)
    forecast = m.predict(prophet_test)
    
    # 4. ARIMA on Residuals
    y_true = prophet_test.set_index('ds')['y']
    y_pred_p = forecast.set_index('ds')['yhat']
    residuals = y_true - y_pred_p
    
    try:
        model_arima = ARIMA(residuals, order=(4,0,1))
        model_fit = model_arima.fit()
        arima_pred = model_fit.forecast(steps=len(test_data))
    except:
        try:
            model_arima = ARIMA(residuals, order=(1,0,1))
            model_fit = model_arima.fit()
            arima_pred = model_fit.forecast(steps=len(test_data))
        except:
             arima_pred = pd.Series([0]*len(test_data), index=test_data.index)

    final_pred = y_pred_p.values + arima_pred.values
    
    result = test_data.copy()
    result['Store'] = store_id
    result['Pred_Hybrid'] = final_pred
    result['Pred_Prophet'] = y_pred_p.values 
    return result

# Execution
print(f"Training Expert Models on {multiprocessing.cpu_count()} Cores...")
unique_stores = df_merged['Store'].unique()
# reduced verbosity to keep log clean
all_predictions = Parallel(n_jobs=-1, verbose=1)(
    delayed(train_predict_expert)(s_id, df_merged) for s_id in unique_stores
)
final_df = pd.concat(all_predictions)
print("Expert Training Complete.")


# --- CELL 4 (EXPERT): Evaluation & Visualization ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- FIX: Restore IsHoliday Column ---
# The expert aggregation in Step 3 dropped 'IsHoliday'. We need it for WMAE.
if 'IsHoliday' not in final_df.columns:
    print("Restoring 'IsHoliday' column for evaluation...")
    # Pull from original merge, or reload if necessary
    try:
        holiday_ref = df_merged[['Store', 'Date', 'IsHoliday']].drop_duplicates()
    except:
        train_ref = pd.read_csv('train.csv')
        train_ref['Date'] = pd.to_datetime(train_ref['Date'])
        holiday_ref = train_ref[['Store', 'Date', 'IsHoliday']].drop_duplicates()
        
    final_df = pd.merge(final_df, holiday_ref, on=['Store', 'Date'], how='left')
    final_df['IsHoliday'] = final_df['IsHoliday'].fillna(False)

print("--- Evaluation Metrics ---\n")

# 1. Define Weighted MAE (Walmart's specific metric)
# Weights: 5x for Holiday weeks, 1x for Non-Holiday weeks
def weighted_mae(dataset, pred_col):
    weights = dataset['IsHoliday'].apply(lambda x: 5 if x else 1)
    return np.sum(weights * np.abs(dataset['Weekly_Sales'] - dataset[pred_col])) / np.sum(weights)

# 2. Calculate Global Scores
if 'Pred_Prophet' in final_df.columns:
    wmae_prophet = weighted_mae(final_df, 'Pred_Prophet')
    print(f"Global Weighted MAE (Prophet Baseline): ${wmae_prophet:,.2f}")

wmae_hybrid = weighted_mae(final_df, 'Pred_Hybrid')
print(f"Global Weighted MAE (Expert Hybrid):    ${wmae_hybrid:,.2f}")

if 'Pred_Prophet' in final_df.columns:
    print(f"Improvement: ${wmae_prophet - wmae_hybrid:,.2f} per store/week (avg)")

# --- Define Specific Holidays for Plotting ---
major_holidays = {
    # Super Bowl
    '2010-02-12': 'Super Bowl', '2011-02-11': 'Super Bowl', 
    '2012-02-10': 'Super Bowl', '2013-02-08': 'Super Bowl',
    # Labor Day
    '2010-09-10': 'Labor Day', '2011-09-09': 'Labor Day', 
    '2012-09-07': 'Labor Day', '2013-09-06': 'Labor Day',
    # Thanksgiving
    '2010-11-26': 'Thanksgiving', '2011-11-25': 'Thanksgiving', 
    '2012-11-23': 'Thanksgiving', '2013-11-29': 'Thanksgiving',
    # Christmas
    '2010-12-31': 'Christmas', '2011-12-30': 'Christmas', 
    '2012-12-28': 'Christmas', '2013-12-27': 'Christmas'
}

# 3. Visualization Function
def plot_expert_forecast(store_id):
    store_data = final_df[final_df['Store'] == store_id].sort_values('Date')
    
    plt.figure(figsize=(18, 8))
    
    # Actuals
    plt.plot(store_data['Date'], store_data['Weekly_Sales'], 
             color='#34495e', linewidth=2.5, label='Actual Sales', alpha=0.9)
    
    # Prophet Baseline (Dotted Orange)
    if 'Pred_Prophet' in store_data.columns:
        plt.plot(store_data['Date'], store_data['Pred_Prophet'], 
                 color='#e67e22', linewidth=2.0, linestyle=':', label='Prophet Baseline (Trend Only)')
        
    # Expert Hybrid (Dashed Green)
    plt.plot(store_data['Date'], store_data['Pred_Hybrid'], 
             color='#2ecc71', linewidth=2.5, linestyle='--', label='Expert Hybrid Forecast (Trend + Residuals)')
    
    # --- Highlight Specific Holidays ---
    y_max = store_data['Weekly_Sales'].max()
    
    for date_str, label in major_holidays.items():
        date_dt = pd.to_datetime(date_str)
        
        # Only plot if the holiday is within this store's date range
        if (date_dt >= store_data['Date'].min()) and (date_dt <= store_data['Date'].max()):
            # Plot ORANGE vertical line
            plt.axvline(x=date_dt, color='orange', alpha=0.8, linestyle='-', linewidth=2.0)
            
            # Add Label at the top
            plt.text(date_dt, y_max, label, 
                     rotation=90, verticalalignment='bottom', horizontalalignment='center',
                     fontsize=9, color='#d35400', fontweight='bold', # Darker orange for text readability
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.title(f'Store {store_id}: Expert Forecast vs Actuals (Major Holidays Highlighted)', fontsize=18)
    plt.ylabel('Weekly Sales', fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.show()

# Visualize for all stores (1 to 45)
for i in range(1, 46):
    plot_expert_forecast(i)


# --- CELL 5: Anomaly Detection (Robust Version) ---

print("--- Running Anomaly Detection ---")

# --- FIX 1: Restore Missing 'Unemployment' Column ---
if 'Unemployment' not in final_df.columns:
    print("Restoring 'Unemployment' column...")
    
    # Try multiple paths to find the file
    try:
        # Try standard CSV first
        feat_ref = pd.read_csv('features.csv')
    except FileNotFoundError:
        try:
            # Try zipped version
            feat_ref = pd.read_csv('features.csv.zip')
        except FileNotFoundError:
            # Try Kaggle input path
            base_path = '/kaggle/input/walmart-recruiting-store-sales-forecasting/'
            try:
                feat_ref = pd.read_csv(base_path + 'features.csv')
            except FileNotFoundError:
                 feat_ref = pd.read_csv(base_path + 'features.csv.zip')

    feat_ref['Date'] = pd.to_datetime(feat_ref['Date'])
    feat_ref = feat_ref[['Store', 'Date', 'Unemployment']]
    
    final_df = pd.merge(final_df, feat_ref, on=['Store', 'Date'], how='left')
    final_df['Unemployment'] = final_df['Unemployment'].ffill().bfill()

# --- FIX 2: Restore 'Is_Super_Holiday' ---
super_holidays = [
    '2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08', # Super Bowl
    '2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06', # Labor Day
    '2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29', # Thanksgiving
    '2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27'  # Christmas
]
final_df['Date'] = pd.to_datetime(final_df['Date'])
final_df['Is_Super_Holiday'] = final_df['Date'].astype(str).isin(super_holidays).astype(int)

# --- FIX 3: Ensure No NaNs in Input Data ---
X_cols = ['Weekly_Sales', 'Total_MarkDown', 'CPI', 'Unemployment']
final_df[X_cols] = final_df[X_cols].fillna(0)

# 1. Fit Isolation Forest
X_anomaly = final_df[X_cols]
iso = IsolationForest(contamination=0.10, random_state=42) 
final_df['Anomaly_Score'] = iso.fit_predict(X_anomaly)

# 2. Business Logic Classification
def explain_anomaly(row):
    # Check 1: Mega Holiday?
    if row['Is_Super_Holiday'] == 1:
        return "Major Holiday Event"
    
    # Check 2: Promo Spike? (High Sales + High Markdown)
    elif row['Total_MarkDown'] > 5000 and row['Weekly_Sales'] > row['Pred_Hybrid']:
        return "Promotional Spike"
    
    # Check 3: Stockout? (Sales significantly lower than Predicted)
    elif row['Weekly_Sales'] < 0.90 * row['Pred_Hybrid']: 
        return "Potential Stockout / Demand Drop"
    
    # Check 4: Unexplained Spike?
    elif row['Weekly_Sales'] > 1.10 * row['Pred_Hybrid']:
        return "Unexplained Demand Surge"
        
    else:
        return "Normal Variance"

final_df['Anomaly_Reason'] = final_df.apply(explain_anomaly, axis=1)

print("\nAnomaly Breakdown (All Weeks):")
print(final_df['Anomaly_Reason'].value_counts())


# --- CELL 6 (EXECUTIVE): Financial Impact Analysis ---

# Filter for the "Money" Table (Stockouts)
stockouts = final_df[final_df['Anomaly_Reason'] == "Potential Stockout / Demand Drop"].copy()
stockouts['Est_Lost_Revenue'] = stockouts['Pred_Hybrid'] - stockouts['Weekly_Sales']

# 1. Total Impact
total_loss = stockouts['Est_Lost_Revenue'].sum()
print(f"--- Executive Summary ---")
print(f"Total Estimated Lost Revenue (Stockouts): ${total_loss:,.2f}")

# 2. Top 10 "Leaky" Stores
store_losses = stockouts.groupby('Store')['Est_Lost_Revenue'].sum().sort_values(ascending=False).head(10)

# 3. Visualization
plt.figure(figsize=(12, 6))
barplot = sns.barplot(x=store_losses.index, y=store_losses.values, palette="Reds_r", order=store_losses.index)

plt.title('Top 10 Stores by Lost Revenue (Stockouts)', fontsize=16)
plt.xlabel('Store ID', fontsize=12)
plt.ylabel('Estimated Lost Revenue ($)', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add labels
for i, v in enumerate(store_losses.values):
    barplot.text(i, v, f'${v/1000:.1f}k', ha='center', va='bottom', fontweight='bold')
    
plt.show()

