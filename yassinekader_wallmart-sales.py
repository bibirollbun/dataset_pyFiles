#Data handling
import pandas as pd
import numpy as np

# Viz
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn
from sklearn import model_selection, metrics, ensemble, linear_model
from sklearn.inspection import permutation_importance
# from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


try:
    features = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/features.csv.zip')
    train = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
    stores = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/stores.csv')
    test = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/test.csv.zip')
    sample_submission = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: Data files not found. Please check your file paths.")


# Merging Data
feature_store = features.merge(stores, how='inner', on = "Store")
train_df = train.merge(feature_store, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)
test_df = test.merge(feature_store, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by = ['Store','Dept','Date']).reset_index(drop=True)


train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])
feature_store['Date'] = pd.to_datetime(feature_store['Date'])

feature_store['Day'] = feature_store['Date'].dt.day
feature_store['Week'] = feature_store['Date'].dt.isocalendar().week # Updated for modern pandas
feature_store['Month'] = feature_store['Date'].dt.month
feature_store['Year'] = feature_store['Date'].dt.year


train_df = train.merge(feature_store, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)
test_df = test.merge(feature_store, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by = ['Store','Dept','Date']).reset_index(drop=True)


train_df.describe().T


print("\n--- Plotting Sales Analysis ---")

# Weekly Sales Sum
df_weeks = train_df.groupby('Week')['Weekly_Sales'].sum().reset_index()

plt.figure(figsize=(15, 6))
sns.lineplot(data=df_weeks, x='Week', y='Weekly_Sales', marker='o')
plt.title('Sales over the year across every week', fontsize=16)
plt.ylabel('Total Sales')
plt.xlabel('Week')
plt.show()


df_weeks_md = train_df.groupby('Week')[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Weekly_Sales']].sum()
# Normalize for visualization comparison
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_weeks_md), columns=df_weeks_md.columns, index=df_weeks_md.index)


plt.figure(figsize=(15, 6))
sns.lineplot(data=df_scaled)
plt.title('Normalized Markdowns vs Sales Trends', fontsize=16)
plt.ylabel('Normalized Value')
plt.show()


weekly_sales = train_df.groupby(['Year','Week'])['Weekly_Sales'].mean().reset_index()

plt.figure(figsize=(15, 6))
sns.lineplot(data=weekly_sales, x='Week', y='Weekly_Sales', hue='Year', palette='viridis', marker='o')
plt.title('Average Sales across the years by weeks', fontsize=16)
plt.show()


train_df['Temperature'] = (train_df['Temperature'] - 32) / 1.8 # Convert to Celsius
train_plt = train_df.sample(frac=0.1, random_state=42) # Sample for plotting speed

# Function to replace Plotly histograms/boxplots
def plot_feature_vs_sales(feature_name, title):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=train_plt, x=feature_name, y='Weekly_Sales', hue='IsHoliday', alpha=0.6)
    plt.title(title, fontsize=14)
    plt.show()


plot_feature_vs_sales('Temperature', 'Temperature vs Sales (Colored by Holiday)')
plot_feature_vs_sales('Fuel_Price', 'Fuel Price vs Sales (Colored by Holiday)')
plot_feature_vs_sales('CPI', 'CPI vs Sales (Colored by Holiday)')
plot_feature_vs_sales('Unemployment', 'Unemployment vs Sales (Colored by Holiday)')


# Store Sizes
sizes = train_plt.groupby('Size')['Weekly_Sales'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=sizes, x='Size', y='Weekly_Sales')
plt.title('Average Sales across Store Sizes', fontsize=14)
plt.show()


# Store Type Boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(data=stores, x='Type', y='Size', palette='viridis')
plt.title('Store Size distribution by Type')
plt.show()


plt.figure(figsize=(10, 6))

# Fix: Use train_df directly since it already contains the 'Type' column.
# Using .merge() here created 'Type_x' and 'Type_y', causing the "Could not interpret input 'Type'" error.
sns.boxplot(data=train_df, x='Type', y='Weekly_Sales', palette='viridis', showfliers=False)

plt.title('Sales distribution by Store Type')
plt.show()


# Departments
depts = train_plt.groupby('Dept')['Weekly_Sales'].mean().sort_values(ascending=False).reset_index()
plt.figure(figsize=(18, 6))
sns.barplot(data=depts, x='Dept', y='Weekly_Sales', palette='viridis')
plt.title('Average Sales across Departments', fontsize=14)
plt.xticks(rotation=90)
plt.show()


# 3.3 Correlation
print("\n--- Correlation Matrix ---")
corr = train_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(16, 12))
sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False) # Annot=False to keep it clean like original
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.show()


# Correlation with Sales Barplot
weekly_sales_corr = corr['Weekly_Sales'].drop('Weekly_Sales').sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=weekly_sales_corr.values, y=weekly_sales_corr.index, palette='viridis')
plt.title('Feature Correlation with Weekly Sales', fontsize=14)
plt.show()


# 4. Feature Engineering
print("\n--- Feature Engineering ---")
data_train = train_df.copy()
data_test = test_df.copy()

# 4.1 Holidays
def calculate_holiday_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    # Vectorized calculation for days to holidays
    df['Days_to_Thansksgiving'] = (pd.to_datetime(df["Year"].astype(str)+"-11-24") - df["Date"]).dt.days.astype(int)
    df['Days_to_Christmas'] = (pd.to_datetime(df["Year"].astype(str)+"-12-24") - df["Date"]).dt.days.astype(int)
    
    df['SuperBowlWeek'] = df['Week'].apply(lambda x: 1 if x == 6 else 0)
    df['LaborDay'] = df['Week'].apply(lambda x: 1 if x == 36 else 0)
    df['Tranksgiving'] = df['Week'].apply(lambda x: 1 if x == 47 else 0)
    df['Christmas'] = df['Week'].apply(lambda x: 1 if x == 52 else 0)
    return df


data_train = calculate_holiday_features(data_train)
data_test = calculate_holiday_features(data_test)


# 4.2 Markdowns
data_train['MarkdownsSum'] = data_train[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].sum(axis=1)
data_test['MarkdownsSum'] = data_test[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].sum(axis=1)


# 5. Preprocessing
print("\n--- Preprocessing ---")

# 5.1 Filling missing values
data_train.fillna(0, inplace=True)

data_test['CPI'] = data_test['CPI'].fillna(data_test['CPI'].mean())
data_test['Unemployment'] = data_test['Unemployment'].fillna(data_test['Unemployment'].mean())
data_test.fillna(0, inplace=True)


# 5.2 Encoding
# IsHoliday is already boolean/int in most merges, but ensuring:
data_train['IsHoliday'] = data_train['IsHoliday'].astype(int)
data_test['IsHoliday'] = data_test['IsHoliday'].astype(int)


# Type Encoding
type_mapping = {'A': 1, 'B': 2, 'C': 3}
data_train['Type'] = data_train['Type'].map(type_mapping).fillna(0) # fillna 0 just in case
data_test['Type'] = data_test['Type'].map(type_mapping).fillna(0)


# 6. Feature Selection
print("\n--- Feature Selection ---")
features_list = [col for col in data_train.columns if col not in ['Date', 'Weekly_Sales']]

X = data_train[features_list].copy()
y = data_train['Weekly_Sales'].copy()


X


y


# Sampling for speed (as per original logic)
data_sample = data_train.sample(frac=0.25, random_state=42)
X_sample = data_sample[features_list]
y_sample = data_sample['Weekly_Sales']

X_train_fs, X_valid_fs, y_train_fs, y_valid_fs = model_selection.train_test_split(X_sample, y_sample, random_state=0, test_size=0.15)


!pip install prophet


from prophet import Prophet
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])


global_sales = train_df.groupby('Date')['Weekly_Sales'].sum().reset_index()
global_sales.columns = ['ds', 'y']


m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
m.add_country_holidays(country_name='US')


m.fit(global_sales)


future = m.make_future_dataframe(periods=39, freq='W') # 39 weeks in test set
forecast = m.predict(future)


fig1 = m.plot(forecast)
plt.title("Total Walmart Weekly Sales Forecast")
plt.show()

fig2 = m.plot_components(forecast)
plt.show()





# PRE-REQUISITE: Install Prophet if you haven't already
# !pip install prophet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import logging

# Mute Prophet's excessive logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

# 1. Load Data
# Update these paths to match your folder structure
path = '../input/walmart-recruiting-store-sales-forecasting/'
try:
    train = pd.read_csv(path + 'train.csv.zip')
    test = pd.read_csv(path + 'test.csv.zip')
    features = pd.read_csv(path + 'features.csv.zip')
    stores = pd.read_csv(path + 'stores.csv')
    print("Data Loaded Successfully.")
except FileNotFoundError:
    print("Error: Files not found. Please check your paths.")

# 2. Merge Data (Store Info & Features)
# We merge to ensure we have all dates and store details aligned
feature_store = features.merge(stores, how='inner', on="Store")

train_df = train.merge(feature_store, how='inner', on=['Store','Date','IsHoliday'])
test_df = test.merge(feature_store, how='inner', on=['Store','Date','IsHoliday'])

# Convert Date columns to datetime objects
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])

# 3. Forecasting Loop
print("Starting Granular Forecast (Store + Dept)...")

submission_preds = []
unique_pairs = test_df[['Store', 'Dept']].drop_duplicates().values

# CONFIGURATION: How many plots do you want to see?
# Set to None to plot ALL (Warning: Will generate 3000+ images)
MAX_PLOTS = 5 
plot_counter = 0

total_pairs = len(unique_pairs)

for i, (store, dept) in enumerate(unique_pairs):
    
    # A. Filter Data for specific Store/Dept
    train_subset = train_df[(train_df['Store'] == store) & (train_df['Dept'] == dept)].copy()
    test_subset = test_df[(test_df['Store'] == store) & (test_df['Dept'] == dept)].copy()
    
    # Skip if we have insufficient training data
    if len(train_subset) < 2:
        # Fill with 0 if no history exists
        test_subset['Weekly_Sales'] = 0
        submission_preds.append(test_subset[['Store', 'Dept', 'Date', 'Weekly_Sales']])
        continue
    
    # B. Prepare for Prophet (Requires 'ds' and 'y' columns)
    prophet_train = train_subset[['Date', 'Weekly_Sales']].rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
    
    # C. Model Setup
    # - weekly_seasonality: Critical for retail
    # - yearly_seasonality: Critical for holidays
    # - uncertainty_samples=0: Speeds up training (disable if you want confidence intervals)
    m = Prophet(yearly_seasonality=True, 
                weekly_seasonality=True, 
                daily_seasonality=False, 
                uncertainty_samples=0)
    
    # Add US Holidays (Thanksgiving/Christmas are vital)
    m.add_country_holidays(country_name='US')
    
    # D. Fit and Predict
    m.fit(prophet_train)
    
    future_dates = pd.DataFrame({'ds': test_subset['Date']})
    forecast = m.predict(future_dates)
    
    # E. Plotting (Visualize the Forecast)
    if plot_counter < MAX_PLOTS:
        print(f"Plotting Forecast for Store {store}, Dept {dept}")
        
        # Plot the components (Trend, Yearly Seasonality, Weekly Seasonality)
        fig1 = m.plot(forecast)
        plt.title(f'Forecast: Store {store} - Dept {dept}')
        plt.xlabel('Date')
        plt.ylabel('Weekly Sales')
        plt.show()
        
        # Optional: Plot components breakdown
        # fig2 = m.plot_components(forecast)
        # plt.show()
        
        plot_counter += 1

    # F. Save Results
    test_subset['Weekly_Sales'] = forecast['yhat'].values
    submission_preds.append(test_subset[['Store', 'Dept', 'Date', 'Weekly_Sales']])
    
    # Progress Tracker
    if i % 100 == 0:
        print(f"Processed {i}/{total_pairs} pairs...")

# 4. Final Compilation
print("Compiling Submission...")
final_submission = pd.concat(submission_preds)

# Handle negative predictions (Sales generally can't be negative)
final_submission['Weekly_Sales'] = final_submission['Weekly_Sales'].apply(lambda x: 0 if x < 0 else x)

# Create ID column as required by Kaggle: Store_Dept_Date
final_submission['Id'] = final_submission['Store'].astype(str) + '_' + \
                         final_submission['Dept'].astype(str) + '_' + \
                         final_submission['Date'].astype(str)

# Save to CSV
output = final_submission[['Id', 'Weekly_Sales']]
output.to_csv('submission_prophet_granular.csv', index=False)

print("Success! Submission saved as 'submission_prophet_granular.csv'")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

# Mute Prophet logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

# 1. Load Data
path = '../input/walmart-recruiting-store-sales-forecasting/'
try:
    train = pd.read_csv(path + 'train.csv.zip')
    features = pd.read_csv(path + 'features.csv.zip')
    stores = pd.read_csv(path + 'stores.csv')
except FileNotFoundError:
    print("Error: Files not found. Please check your paths.")

# Merge Data
feature_store = features.merge(stores, how='inner', on="Store")
train_df = train.merge(feature_store, how='inner', on=['Store','Date','IsHoliday'])
train_df['Date'] = pd.to_datetime(train_df['Date'])

# ---------------------------------------------------------
# 2. CONFIGURATION: VALIDATION SPLIT
# ---------------------------------------------------------
# We hide the last 3 months of data to act as our "Test Set"
VAL_START_DATE = '2012-08-01' 

# Run on a SAMPLE of stores to get metrics quickly (running all 45 takes hours)
sample_stores = [1, 10, 20, 30, 40] 
print(f"Running Validation on Stores: {sample_stores}")
print(f"Validation Period: {VAL_START_DATE} to {train_df['Date'].max()}")

results = []

# 3. Validation Loop
for store in sample_stores:
    # Filter for current store
    store_data = train_df[train_df['Store'] == store]
    
    for dept in store_data['Dept'].unique():
        # Get Dept Data sorted by date
        dept_data = store_data[store_data['Dept'] == dept].sort_values('Date')
        
        # Skip departments with too little data
        if len(dept_data) < 20: continue 
        
        # SPLIT: Train (Before Aug) vs Test (Aug onwards)
        train_subset = dept_data[dept_data['Date'] < VAL_START_DATE]
        test_subset = dept_data[dept_data['Date'] >= VAL_START_DATE]
        
        if len(test_subset) < 1: continue

        # Prepare for Prophet
        prophet_train = train_subset[['Date', 'Weekly_Sales']].rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
        
        # Initialize Prophet
        m = Prophet(yearly_seasonality=True, 
                    weekly_seasonality=True, 
                    daily_seasonality=False, 
                    uncertainty_samples=0)
        m.add_country_holidays(country_name='US')
        
        # Fit
        m.fit(prophet_train)
        
        # Predict on "Test" (Validation) period
        future = pd.DataFrame({'ds': test_subset['Date']})
        forecast = m.predict(future)
        
        # Store Actuals vs Predicted
        temp_res = test_subset[['Date', 'Store', 'Dept', 'Weekly_Sales']].copy()
        temp_res['Predicted'] = forecast['yhat'].values
        results.append(temp_res)

# Concatenate all results
val_df = pd.concat(results)

# Remove negative predictions (Sales can't be negative)
val_df['Predicted'] = val_df['Predicted'].apply(lambda x: 0 if x < 0 else x)

# ---------------------------------------------------------
# 4. METRICS & PLOTTING
# ---------------------------------------------------------

# A. Calculate Metrics
mae = mean_absolute_error(val_df['Weekly_Sales'], val_df['Predicted'])
rmse = np.sqrt(mean_squared_error(val_df['Weekly_Sales'], val_df['Predicted']))

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mape = calculate_mape(val_df['Weekly_Sales'], val_df['Predicted'])

print("\n" + "="*40)
print("   PROPHET MODEL PERFORMANCE (VALIDATION)")
print("="*40)
print(f"Test MAE:  {mae:,.2f}")
print(f"Test RMSE: {rmse:,.2f}")
print(f"Test MAPE: {mape:.2f}%")
print("="*40 + "\n")

# B. Plots
sns.set_style("whitegrid")

# Plot 1: Actual vs Predicted Scatter
plt.figure(figsize=(10, 6))
sns.scatterplot(x=val_df['Weekly_Sales'], y=val_df['Predicted'], alpha=0.3, color='#0077B6')
plt.plot([val_df['Weekly_Sales'].min(), val_df['Weekly_Sales'].max()], 
         [val_df['Weekly_Sales'].min(), val_df['Weekly_Sales'].max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.title('Prophet: Actual vs. Predicted Sales', fontsize=16)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.legend()
plt.show()

# Plot 2: Time Series Zoom (Store 1, Dept 1)
# Visualizing how the forecast looks over time for one specific department
example = val_df[(val_df['Store'] == 1) & (val_df['Dept'] == 1)].sort_values('Date')
plt.figure(figsize=(14, 6))
plt.plot(example['Date'], example['Weekly_Sales'], label='Actual Sales', marker='o', color='gray')
plt.plot(example['Date'], example['Predicted'], label='Prophet Forecast', marker='x', color='#20BAFA', lw=2, ls='--')
plt.title('Prophet Forecast vs Actuals (Store 1, Dept 1 - Validation Period)', fontsize=16)
plt.ylabel('Weekly Sales')
plt.legend()
plt.show()

# Plot 3: Residuals
residuals = val_df['Weekly_Sales'] - val_df['Predicted']
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=100, kde=True, color='#4AC9FE')
plt.title('Distribution of Errors (Residuals)', fontsize=16)
plt.xlabel('Error (Actual - Predicted)')
plt.xlim(-5000, 5000) # Zoom to center
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import logging

# Mute Prophet logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

# 1. Load Data
path = '../input/walmart-recruiting-store-sales-forecasting/'
try:
    train = pd.read_csv(path + 'train.csv.zip')
    test = pd.read_csv(path + 'test.csv.zip')
    features = pd.read_csv(path + 'features.csv.zip')
    stores = pd.read_csv(path + 'stores.csv')
except FileNotFoundError:
    print("Error: Files not found. Please check your paths.")

# Merge Features
feature_store = features.merge(stores, how='inner', on="Store")
train_df = train.merge(feature_store, how='inner', on=['Store','Date','IsHoliday'])
test_df = test.merge(feature_store, how='inner', on=['Store','Date','IsHoliday'])

train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])

# ==========================================
# PART A: GLOBAL DATA FORECAST (All Stores)
# ==========================================
print("Running Global Forecast (Aggregated Data)...")

# 1. Aggregate Sales by Date
global_sales = train_df.groupby('Date')['Weekly_Sales'].sum().reset_index()
global_sales.columns = ['ds', 'y']

# 2. Setup & Fit Prophet
# Global data usually has strong seasonality
m_global = Prophet(yearly_seasonality=True, 
                   weekly_seasonality=True, 
                   daily_seasonality=False,
                   changepoint_prior_scale=0.5) # Flexible trend for global shifts
m_global.add_country_holidays(country_name='US')
m_global.fit(global_sales)

# 3. Create Future Dataframe (Train Dates + Test Period)
# We forecast 39 weeks into the future (size of test set)
future_global = m_global.make_future_dataframe(periods=39, freq='W')
forecast_global = m_global.predict(future_global)

# 4. Plotting Global Data
plt.figure(figsize=(16, 8))
sns.set_style("whitegrid")

# Plot Actuals (Training Data)
plt.plot(global_sales['ds'], global_sales['y'], label='Actual Sales (History)', color='black', alpha=0.6)

# Plot Forecast (Full History + Future)
# We plot the 'yhat' (prediction) line
plt.plot(forecast_global['ds'], forecast_global['yhat'], label='Prophet Forecast', color='#20BAFA', linewidth=2)

# Plot Uncertainty Interval (Shaded Area)
plt.fill_between(forecast_global['ds'], 
                 forecast_global['yhat_lower'], 
                 forecast_global['yhat_upper'], 
                 color='#20BAFA', alpha=0.2, label='Confidence Interval')

plt.title('Global Walmart Sales: History vs. Prophet Forecast', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Weekly Sales', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# Highlight the "Future" part
test_start_date = train_df['Date'].max()
plt.axvline(x=test_start_date, color='red', linestyle='--', label='Forecast Start')

plt.show()

# ==========================================
# PART B: GRANULAR LONG-TERM PLOT (Store 1, Dept 1)
# ==========================================
print("Running Granular Forecast (Store 1, Dept 1)...")

# 1. Select Data
s1_d1 = train_df[(train_df['Store'] == 1) & (train_df['Dept'] == 1)][['Date', 'Weekly_Sales']]
s1_d1.columns = ['ds', 'y']

# 2. Fit Prophet
m_gran = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
m_gran.add_country_holidays(country_name='US')
m_gran.fit(s1_d1)

# 3. Forecast
future_gran = m_gran.make_future_dataframe(periods=39, freq='W')
forecast_gran = m_gran.predict(future_gran)

# 4. Plotting Granular Data
plt.figure(figsize=(16, 8))

# Actuals
plt.scatter(s1_d1['ds'], s1_d1['y'], color='black', s=20, label='Actual Sales (Dots)')

# Forecast Line
plt.plot(forecast_gran['ds'], forecast_gran['yhat'], color='#FF5733', linewidth=2, label='Prophet Forecast')

# Confidence Interval
plt.fill_between(forecast_gran['ds'], 
                 forecast_gran['yhat_lower'], 
                 forecast_gran['yhat_upper'], 
                 color='#FF5733', alpha=0.2)

plt.title('Long-Term Forecast: Store 1, Dept 1', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Weekly Sales', fontsize=12)
plt.legend()
plt.axvline(x=test_start_date, color='red', linestyle='--', label='Forecast Start') # Divider line

plt.show()

# Optional: Component Plot to see Seasonality/Trend
# fig_comp = m_global.plot_components(forecast_global)
# plt.show()


# ==========================================
# PART B: GRANULAR LONG-TERM PLOT (Store 1, Dept 1)
# ==========================================
print("Running Granular Forecast (Store 1, Dept 1)...")

# 1. Select Data
s1_d1 = train_df[(train_df['Store'] == 1) & (train_df['Dept'] == 1)][['Date', 'Weekly_Sales']]
s1_d1.columns = ['ds', 'y']

# 2. Fit Prophet
m_gran = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
m_gran.add_country_holidays(country_name='US')
m_gran.fit(s1_d1)

# 3. Forecast
future_gran = m_gran.make_future_dataframe(periods=39, freq='W')
forecast_gran = m_gran.predict(future_gran)

# 4. Plotting Granular Data
plt.figure(figsize=(16, 8))

# --- CHANGE: Plot Actuals as a Line instead of Scatter ---
plt.plot(s1_d1['ds'], s1_d1['y'], color='black', linewidth=1.5, alpha=0.7, label='Actual Sales (History)')

# Forecast Line
plt.plot(forecast_gran['ds'], forecast_gran['yhat'], color='#FF5733', linewidth=2, label='Prophet Forecast')

# Confidence Interval
plt.fill_between(forecast_gran['ds'], 
                 forecast_gran['yhat_lower'], 
                 forecast_gran['yhat_upper'], 
                 color='#FF5733', alpha=0.2, label='Confidence Interval')

# Styling
plt.title('Long-Term Forecast: Store 1, Dept 1', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Weekly Sales', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# Divider line for Future
test_start_date = train_df['Date'].max()
plt.axvline(x=test_start_date, color='red', linestyle='--', label='Forecast Start')

plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import logging

# Mute Prophet logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

# 1. Load & Prepare Data
path = '../input/walmart-recruiting-store-sales-forecasting/'
try:
    train = pd.read_csv(path + 'train.csv.zip')
    features = pd.read_csv(path + 'features.csv.zip')
    stores = pd.read_csv(path + 'stores.csv')
    
    feature_store = features.merge(stores, how='inner', on="Store")
    train_df = train.merge(feature_store, how='inner', on=['Store','Date','IsHoliday'])
    train_df['Date'] = pd.to_datetime(train_df['Date'])

    # 2. Filter for Store 1, Dept 1
    subset = train_df[(train_df['Store'] == 1) & (train_df['Dept'] == 1)].sort_values('Date')

    # 3. Split: Train (Before Aug 2012) & Test (Aug 2012 onwards)
    split_date = '2012-04-12'
    train_data = subset[subset['Date'] < split_date].copy()
    test_data = subset[subset['Date'] >= split_date].copy()

    # 4. Fit Prophet Model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.add_country_holidays(country_name='US')
    
    # Format for Prophet
    prophet_train = train_data[['Date', 'Weekly_Sales']].rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
    model.fit(prophet_train)

    # 5. Forecast
    future = pd.DataFrame({'ds': test_data['Date']})
    forecast = model.predict(future)

    # 6. Plotting (Exact style requested)
    plt.figure(figsize=(12,6))
    
    # Plotting columns against Date to get "weeks" on x-axis automatically
    plt.plot(train_data['Date'], train_data['Weekly_Sales'], label="Train")
    plt.plot(test_data['Date'], test_data['Weekly_Sales'], label="Test")
    
    # Forecast needs to share the same x-axis (Dates)
    plt.plot(test_data['Date'], forecast['yhat'].values, label="Forecast")
    
    plt.title("Weekly Sales: Store 1, Dept 1")
    plt.xlabel("Date (Weeks)")
    plt.ylabel("Weekly Sales")
    plt.legend()
    plt.show()

except FileNotFoundError:
    print("Error: Files not found. Check your paths.")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

# Mute Prophet logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

# 1. Load Data
path = '../input/walmart-recruiting-store-sales-forecasting/'
try:
    train = pd.read_csv(path + 'train.csv.zip')
    features = pd.read_csv(path + 'features.csv.zip')
    stores = pd.read_csv(path + 'stores.csv')
    
    # Merge
    feature_store = features.merge(stores, how='inner', on="Store")
    train_df = train.merge(feature_store, how='inner', on=['Store','Date','IsHoliday'])
    train_df['Date'] = pd.to_datetime(train_df['Date'])

    # 2. Filter for Store 1, Dept 1
    subset = train_df[(train_df['Store'] == 1) & (train_df['Dept'] == 1)].sort_values('Date')

    # 3. Split Data
    split_date = '2012-04-12'
    train_data = subset[subset['Date'] < split_date].copy()
    test_data = subset[subset['Date'] >= split_date].copy()

    # 4. Fit Prophet Model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.add_country_holidays(country_name='US')
    
    prophet_train = train_data[['Date', 'Weekly_Sales']].rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
    model.fit(prophet_train)

    # 5. Forecast on Test Dates
    future = pd.DataFrame({'ds': test_data['Date']})
    forecast = model.predict(future)
    
    # Extract predicted values
    y_true = test_data['Weekly_Sales'].values
    y_pred = forecast['yhat'].values

    # 6. Compute Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE calculation (handling division by zero)
    non_zero_mask = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

    print("="*40)
    print(f"METRICS FOR STORE 1, DEPT 1 (Split: {split_date})")
    print("="*40)
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAPE: {mape:.2f}%")
    print("="*40)

    # 7. Plot (Visual Confirmation)
    plt.figure(figsize=(12,6))
    plt.plot(train_data['Date'], train_data['Weekly_Sales'], label="Train")
    plt.plot(test_data['Date'], test_data['Weekly_Sales'], label="Test (Actual)")
    plt.plot(test_data['Date'], y_pred, label="Forecast", linestyle='--')
    
    plt.title(f"Store 1, Dept 1 Forecast (Split: {split_date})")
    plt.xlabel("Date")
    plt.ylabel("Weekly Sales")
    plt.legend()
    plt.show()

except FileNotFoundError:
    print("Error: Files not found. Check your paths.")















































































