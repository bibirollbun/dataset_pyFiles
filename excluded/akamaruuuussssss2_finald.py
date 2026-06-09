import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 1. データ読み込み ---
DATA_DIR = "/kaggle/input/rossmann-store-sales"
train = pd.read_csv(f"{DATA_DIR}/train.csv", low_memory=False, parse_dates=['Date'])
test  = pd.read_csv(f"{DATA_DIR}/test.csv", low_memory=False, parse_dates=['Date'])
store = pd.read_csv(f"{DATA_DIR}/store.csv", low_memory=False)

# --- store をマージ ---
train = train.merge(store, on='Store', how='left')
test = test.merge(store, on='Store', how='left')

# --- 特徴量追加関数 ---
def add_features(df):
    # Original Date Features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['IsMonthEnd'] = (df['Day'] >= 25).astype(int)

    # New Date Features
    df['Quarter'] = df['Date'].dt.quarter
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
    df['IsQuarterStart'] = df['Date'].dt.is_quarter_start.astype(int)
    df['IsYearStart'] = df['Date'].dt.is_year_start.astype(int)
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    df['IsQuarterEnd'] = df['Date'].dt.is_quarter_end.astype(int)
    df['IsYearEnd'] = df['Date'].dt.is_year_end.astype(int)
    df['DayOfYear'] = df['Date'].dt.dayofyear # Corrected from dayofofyear
    df['WeekOfMonth'] = (df['Day'] - 1) // 7 + 1

    # PromoOpen（月単位の経過期間）
    # Convert Promo2SinceYear and Promo2SinceWeek to datetime for easier calculation
    # Handle potential conversion errors for invalid dates by setting them to NaT
    df['Promo2Start'] = pd.to_datetime(df['Promo2SinceYear'].fillna(0).astype(int).astype(str) + '-' + \
                                      df['Promo2SinceWeek'].fillna(0).astype(int).astype(str) + '-1', format='%Y-%W-%w', errors='coerce')
    df['PromoOpen'] = ((df['Date'] - df['Promo2Start']).dt.days // 30).fillna(0).astype(int)
    df['PromoOpen'] = df['PromoOpen'].apply(lambda x: x if x > 0 else 0)


    # CompetitionOpen (in months)
    # Fill NaN values for CompetitionOpenSinceYear/Month with a placeholder like current year/month
    # Or, a more robust way: If CompetitionOpenSinceYear/Month is NaN, assume no competition yet.
    # For now, let's fill with 0 and handle later.
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(df['Year'])
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(df['Month'])

    df['CompetitionOpen'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + \
                            (df['Month'] - df['CompetitionOpenSinceMonth'])
    # Pandas 3.0 FutureWarning fix
    df['CompetitionOpen'] = df['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)
    df['CompetitionOpen'] = df['CompetitionOpen'].fillna(0) # Fill any remaining NaNs, e.g., if year is less than competition year

    # Interaction Features
    df['IsPromo2Active'] = ((df['Promo2SinceYear'].notna()) & (df['Promo2SinceWeek'].notna()) & (df['Promo2SinceYear'] != 0) & (df['Promo2SinceWeek'] != 0)).astype(int)
    df['Promo2Interval_None'] = df['PromoInterval'].isna().astype(int) # Is PromoInterval NaN
    df['Promo2Interval_JanFeb'] = df['PromoInterval'].apply(lambda x: 'Jan,Feb' in str(x)).astype(int)
    df['Promo2Interval_MarApr'] = df['PromoInterval'].apply(lambda x: 'Mar,Apr' in str(x)).astype(int)
    df['Promo2Interval_MayJun'] = df['PromoInterval'].apply(lambda x: 'May,Jun' in str(x)).astype(int)
    df['Promo2Interval_JulAug'] = df['PromoInterval'].apply(lambda x: 'Jul,Aug' in str(x)).astype(int)
    df['Promo2Interval_SeptOct'] = df['PromoInterval'].apply(lambda x: 'Sept,Oct' in str(x)).astype(int)
    df['Promo2Interval_NovDec'] = df['PromoInterval'].apply(lambda x: 'Nov,Dec' in str(x)).astype(int)

    # Check if current month is in PromoInterval
    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    df['CurrentMonthInPromo2Interval'] = df.apply(lambda row: month_map[row['Month']] in str(row['PromoInterval']) if pd.notna(row['PromoInterval']) else 0, axis=1).astype(int)

    # Store Type and Assortment interaction with Promo
    df['StoreType_Promo'] = df['StoreType'].astype(str) + '_' + df['Promo'].astype(str)
    df['Assortment_Promo'] = df['Assortment'].astype(str) + '_' + df['Promo'].astype(str)


    # Holiday features
    df['IsPublicHoliday'] = ((df['StateHoliday'] == 'a') | (df['StateHoliday'] == 'b') | (df['StateHoliday'] == 'c')).astype(int)
    df['IsEasterHoliday'] = (df['StateHoliday'] == 'b').astype(int) # Assuming 'b' is Easter-related

    return df

# --- 日付ベースの特徴量追加 ---
train = add_features(train)
test = add_features(test)

# --- 店舗ごとの平均売上を特徴量として追加（trainに基づく） ---
# Original StoreAvgSales
store_avg = train.groupby('Store')['Sales'].mean().rename('StoreAvgSales')
train = train.merge(store_avg, on='Store', how='left')
test = test.merge(store_avg, on='Store', how='left')

# Additional Aggregated Features (from train data only to avoid data leakage)
# These features capture typical behavior of stores
store_promo_avg_sales = train.groupby(['Store', 'Promo'])['Sales'].mean().rename('StorePromoAvgSales').reset_index()
train = train.merge(store_promo_avg_sales, on=['Store', 'Promo'], how='left')
test = test.merge(store_promo_avg_sales, on=['Store', 'Promo'], how='left')

store_dayofweek_avg_sales = train.groupby(['Store', 'DayOfWeek'])['Sales'].mean().rename('StoreDayOfWeekAvgSales').reset_index()
train = train.merge(store_dayofweek_avg_sales, on=['Store', 'DayOfWeek'], how='left')
test = test.merge(store_dayofweek_avg_sales, on=['Store', 'DayOfWeek'], how='left')

# Fill NaNs for new test features that might not have a match in train
# This is crucial for test set where a new combination of Store/Promo or Store/DayOfWeek might appear
for col in ['StorePromoAvgSales', 'StoreDayOfWeekAvgSales']:
    if col in test.columns:
        test[col] = test[col].fillna(test[col].mean()) # Pandas 3.0 FutureWarning fix

# --- Lagged Features (requires sorting by Date and Store) ---
# Ensure data is sorted for correct lagging
train = train.sort_values(by=['Store', 'Date']).reset_index(drop=True)
test = test.sort_values(by=['Store', 'Date']).reset_index(drop=True) # Test set might need different handling if future dates are not present

# Important: If 'Open' is NaN in test, assume it's open if it's a sales day.
# In this competition, test set includes 'Open' column.
# Fill missing 'Open' values in test set based on common assumptions (e.g., if store is closed on Sunday, Open=0)
test['Open'] = test['Open'].fillna(1).astype(int) # Pandas 3.0 FutureWarning fix and ensure int type


# --- Drop original date related columns as we have extracted features ---
drop_cols = ['Id', 'Sales', 'Customers', 'Date', 'Promo2Start'] # 'Promo2Start' was temporary

# Define columns to keep after feature engineering
# Now, explicitly include the original columns used for feature creation
# if you want them in the final dataset for the model.
# PromoInterval will be label encoded.
engineered_features = [col for col in train.columns if col not in drop_cols]
# Ensure that the original columns used for calculation that you later want to fill NaNs for are in `engineered_features`
# For example, 'Promo2SinceYear', 'Promo2SinceWeek', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'PromoInterval'
# If these were not explicitly excluded earlier, they would be present.
# Let's ensure they are not accidentally dropped by the list comprehension.
# For simplicity, we'll ensure they are present by not dropping them.


# Prepare train and test datasets with engineered features
X_train_df = train[engineered_features].copy()
X_test_df = test[engineered_features].copy()

# Ensure 'Open' column is numerical and filled (already done above)
X_train_df['Open'] = X_train_df['Open'].astype(int)
X_test_df['Open'] = X_test_df['Open'].astype(int)

# --- Preprocessing Missing Values ---
# Fill remaining NaNs (e.g., for CompetitionDistance, Promo2 related)
for df in [X_train_df, X_test_df]:
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median()) # Pandas 3.0 FutureWarning fix
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0) # Pandas 3.0 FutureWarning fix
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0) # Pandas 3.0 FutureWarning fix
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0) # Pandas 3.0 FutureWarning fix
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0) # Pandas 3.0 FutureWarning fix

# Combine for consistent Label Encoding
all_data_for_encoding = pd.concat([X_train_df, X_test_df], axis=0).reset_index(drop=True)

# --- Label Encoding for all categorical object columns ---
for col in all_data_for_encoding.columns:
    if all_data_for_encoding[col].dtype == 'object':
        # Replace NaN with a string representation if any, before encoding
        all_data_for_encoding[col] = all_data_for_encoding[col].fillna('None') # Pandas 3.0 FutureWarning fix
        le = LabelEncoder()
        all_data_for_encoding[col] = le.fit_transform(all_data_for_encoding[col].astype(str))

# --- Split back ---
X_train = all_data_for_encoding.iloc[:len(train)]
X_test = all_data_for_encoding.iloc[len(train):]
y_train = train['Sales']
test_id = test['Id']

# --- XGBoost モデル学習 ---
# Hyperparameters are kept as per your request to focus on feature engineering.
# The goal is to see the impact of features, not hyperparameter tuning.
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=71,
    n_jobs=-1, # Use all available cores
    tree_method='hist', # Faster training for large datasets
    # Add an objective that suits regression with positive values better
    objective='reg:squarederror'
)
model.fit(X_train, y_train)

# --- 推論 ---
y_pred = model.predict(X_test)
y_pred = np.maximum(0, y_pred) # Ensure no negative sales predictions

# --- 提出ファイル作成 ---
submission = pd.DataFrame({'Id': test_id, 'Sales': y_pred})
submission.to_csv("submission_with_enhanced_features.csv", index=False)

print("Submission file 'submission_with_enhanced_features.csv' created successfully.")

