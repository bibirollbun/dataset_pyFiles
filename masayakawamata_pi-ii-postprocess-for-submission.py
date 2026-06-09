import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

warnings.filterwarnings('ignore')
print("--- Future Model: Rule-Based Extrapolation with Optimized Global Trend ---")
print("=" * 60)

BASE_SUBMISSION_PATH = '/kaggle/input/pic-ii-l3-hill-climb/submission_hillclimb_cross_bounds.csv' 

TRAIN_DATA_PATH = '/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv'
TEST_DATA_PATH = '/kaggle/input/prediction-interval-competition-ii-house-price/test.csv'

LAST_KNOWN_DATE = pd.to_datetime('2025-01-31')
CV_START_DATE = pd.to_datetime('2010-01-01') 
VALIDATION_MONTHS = 3 
LOOKBACK_CANDIDATES = [6, 9, 12, 18, 24, 30, 36, 42, 48]

print(f"ğŸ› ï¸� Finding best lookback period and correction factor for the global market trend...")

try:
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_train['sale_date'] = pd.to_datetime(df_train['sale_date'])

    best_lookback = 0
    best_mae = float('inf')
    best_correction_factor = 1.0

    for lookback_months in tqdm(LOOKBACK_CANDIDATES, desc="Testing Lookback Periods"):
        correction_factors = []
        all_preds, all_actuals = [], []
        
        start_date = CV_START_DATE
        end_loop_date = df_train['sale_date'].max() - relativedelta(months=VALIDATION_MONTHS)
        current_date = start_date

        while current_date <= end_loop_date:
            train_end_date = current_date
            train_start_date = train_end_date - relativedelta(months=lookback_months)
            validation_start_date = train_end_date + relativedelta(days=1)
            validation_end_date = validation_start_date + relativedelta(months=VALIDATION_MONTHS) - relativedelta(days=1)

            df_train_window = df_train[(df_train['sale_date'] >= train_start_date) & (df_train['sale_date'] <= train_end_date)]
            df_validation_window = df_train[(df_train['sale_date'] >= validation_start_date) & (df_train['sale_date'] <= validation_end_date)]

            if df_train_window.empty or df_validation_window.empty:
                current_date += relativedelta(months=1)
                continue

            monthly_avg_price_train = df_train_window.set_index('sale_date')['sale_price'].resample('M').mean().dropna()
            
            if len(monthly_avg_price_train) < 2:
                current_date += relativedelta(months=1)
                continue

            monthly_growth_rates_train = monthly_avg_price_train.pct_change().dropna()
            
            if monthly_growth_rates_train.empty:
                current_date += relativedelta(months=1)
                continue

            avg_monthly_growth_train = monthly_growth_rates_train.mean()
            last_month_price = monthly_avg_price_train.iloc[-1]

            predicted_prices = [last_month_price * ((1 + avg_monthly_growth_train) ** i) for i in range(1, VALIDATION_MONTHS + 1)]
            
            monthly_avg_price_validation = df_validation_window.set_index('sale_date')['sale_price'].resample('M').mean().dropna()
            actual_prices = monthly_avg_price_validation.values

            if len(predicted_prices) == len(actual_prices) and all(p > 0 for p in predicted_prices) and len(actual_prices) > 0:
                ratios = actual_prices / predicted_prices
                correction_factors.extend(ratios)
                all_preds.extend(predicted_prices)
                all_actuals.extend(actual_prices)
                
            current_date += relativedelta(months=1)

        if not all_preds or not correction_factors:
            continue

        current_mae = mean_absolute_error(all_actuals, np.array(all_preds) * np.mean(correction_factors))
        
        if current_mae < best_mae:
            best_mae = current_mae
            best_lookback = lookback_months
            best_correction_factor = np.mean(correction_factors)

    print("\n" + "="*40)
    print("âœ… Strategy Optimization Complete.")
    print(f"ğŸ�† Best Lookback Period: {best_lookback} months")
    print(f"ğŸ�† Best Historical Correction Factor: {best_correction_factor:.4f}")
    print(f"ğŸ�† Best Historical MAE: {best_mae:.2f}")
    print("="*40)

except Exception as e:
    print(f"â�Œ Error during walk-forward validation: {e}")
    best_lookback = 12
    best_correction_factor = 1.0

print(f"\nğŸ› ï¸� Calculating final market trends using best lookback: {best_lookback} months...")
try:
    final_trend_start_date = LAST_KNOWN_DATE - relativedelta(months=best_lookback-1)
    df_recent = df_train[(df_train['sale_date'] >= final_trend_start_date) & (df_train['sale_date'] <= LAST_KNOWN_DATE)].copy()
    
    if df_recent.empty:
        raise ValueError(f"No data found for the last {best_lookback} months.")

    monthly_avg_price = df_recent.set_index('sale_date')['sale_price'].resample('M').mean().dropna()
    monthly_growth_rates = monthly_avg_price.pct_change().dropna()
    avg_monthly_growth = monthly_growth_rates.mean()
    growth_volatility = monthly_growth_rates.std()

    print(f"âœ… Calculated Final Average Monthly Growth: {avg_monthly_growth:.4%}")
    print(f"âœ… Calculated Final Growth Volatility (Std Dev): {growth_volatility:.4%}")

except (FileNotFoundError, ValueError) as e:
    print(f"â�Œ Error calculating final market trends: {e}")
    print("Using default fallback values.")
    avg_monthly_growth = 0.005
    growth_volatility = 0.01

print("\nğŸ“� Loading data for extrapolation...")
try:
    df_base = pd.read_csv(BASE_SUBMISSION_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)
    print("âœ… All files loaded successfully.")
except FileNotFoundError as e:
    print(f"â�Œ Error: File not found. Please check the path: {e}")
    exit()

df_final = pd.merge(df_base, df_test[['id', 'sale_date']], on='id')
df_final['sale_date'] = pd.to_datetime(df_final['sale_date'])
df_before_extrapolation = df_final.copy()

print("\nğŸš€ Applying rule-based extrapolation to future predictions...")
df_final['months_to_extrapolate'] = (df_final['sale_date'].dt.year - LAST_KNOWN_DATE.year) * 12 + \
                                    (df_final['sale_date'].dt.month - LAST_KNOWN_DATE.month)
future_mask = df_final['months_to_extrapolate'] > 0
future_df = df_final[future_mask].copy()

compounded_growth = (1 + avg_monthly_growth) ** future_df['months_to_extrapolate']
future_df['pi_lower'] *= compounded_growth * best_correction_factor
future_df['pi_upper'] *= compounded_growth * best_correction_factor

mid_point = (future_df['pi_lower'] + future_df['pi_upper']) / 2
half_width = (future_df['pi_upper'] - future_df['pi_lower']) / 2
volatility_factor = 1 + (growth_volatility * future_df['months_to_extrapolate'])
new_half_width = half_width * volatility_factor
future_df['pi_lower'] = mid_point - new_half_width
future_df['pi_upper'] = mid_point + new_half_width

df_final.loc[future_mask, ['pi_lower', 'pi_upper']] = future_df[['pi_lower', 'pi_upper']]
print("âœ… Extrapolation complete.")

print("\nğŸ’¾ Generating and saving final submission file...")
submission_df = df_final[['id', 'pi_lower', 'pi_upper']]
submission_df['pi_upper'] = np.maximum(submission_df['pi_lower'], submission_df['pi_upper'])
submission_df.to_csv("submission.csv", index=False)
print("\nğŸ�‰ Hybrid submission file 'submission.csv' has been created successfully!")
print("Final submission head:")
print(submission_df.head())


print("\nğŸ“ˆ Plotting the comparison...")
df_before_monthly = df_before_extrapolation.set_index('sale_date')[['pi_lower', 'pi_upper']].resample('M').mean()
df_after_monthly = df_final.set_index('sale_date')[['pi_lower', 'pi_upper']].resample('M').mean()
df_before_monthly['pi_mid'] = (df_before_monthly['pi_lower'] + df_before_monthly['pi_upper']) / 2
df_after_monthly['pi_mid'] = (df_after_monthly['pi_lower'] + df_after_monthly['pi_upper']) / 2

plt.figure(figsize=(16, 8))
plt.style.use('seaborn-v0_8-whitegrid')
plt.plot(df_before_monthly.index, df_before_monthly['pi_mid'], label='Before Extrapolation (Mid)', color='gray', linestyle='--', alpha=0.8)
plt.fill_between(
    df_before_monthly.index,
    df_before_monthly['pi_lower'],
    df_before_monthly['pi_upper'],
    color='gray',
    alpha=0.15,
    label='Before Prediction Interval'
)
plt.plot(df_after_monthly.index, df_after_monthly['pi_mid'], label='After Extrapolation (Mid)', color='dodgerblue', linewidth=2)
plt.fill_between(
    df_after_monthly.index,
    df_after_monthly['pi_lower'],
    df_after_monthly['pi_upper'],
    color='dodgerblue',
    alpha=0.2,
    label='After Prediction Interval'
)
plt.title('Before vs. After Rule-Based Extrapolation (Monthly Average)', fontsize=18)
plt.xlabel('Sale Date', fontsize=14)
plt.ylabel('Predicted Price', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlim([pd.to_datetime('2024-01-01'), pd.to_datetime('2025-05-31')])
plt.axvline(x=pd.to_datetime('2025-02-01'), color='red', linestyle=':', linewidth=2, label='Future Period Start')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('rule_based_extrapolation_comparison.png')
print("\nğŸ�‰ Comparison plot saved as 'rule_based_extrapolation_comparison.png'.")
plt.show()




