import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignore unnecessary warnings
warnings.filterwarnings('ignore')

# Set consistent visualization style
plt.style.use('seaborn-v0_8-darkgrid')



base_path = "/kaggle/input/recruit-restaurant-visitor-forecasting"
train_path = os.path.join(base_path, "air_visit_data.csv.zip")
store_info_path = os.path.join(base_path, "air_store_info.csv.zip")


df_train = pd.read_csv(train_path, compression='zip')
df_store = pd.read_csv(store_info_path, compression='zip')

print("âœ… Train Data Shape:", df_train.shape)
print("âœ… Store Info Shape:", df_store.shape)
print("\nTrain Data Sample:")
print(df_train.head())


print("\nğŸ”¹ Dataset Overview:")
print(df_train.info())


print("\nğŸ”¹ Missing Values:")
print(df_train.isnull().sum())


df_train['visit_date'] = pd.to_datetime(df_train['visit_date'])
df_train['year'] = df_train['visit_date'].dt.year
df_train['month'] = df_train['visit_date'].dt.month
df_train['day'] = df_train['visit_date'].dt.day
df_train['weekday'] = df_train['visit_date'].dt.day_name()

print("\nUnique stores:", df_train['air_store_id'].nunique())
print("Date range:", df_train['visit_date'].min(), "â†’", df_train['visit_date'].max())


print("\nVisitor Statistics:")


print(df_train['visitors'].describe())


plt.figure(figsize=(8,5))
sns.histplot(df_train['visitors'], bins=50, kde=True)
plt.title("Distribution of Visitors per Record")
plt.xlabel("Visitors")
plt.ylabel("Frequency")
plt.show()


daily_visitors = df_train.groupby('visit_date')['visitors'].sum().reset_index()

plt.figure(figsize=(12,6))
plt.plot(daily_visitors['visit_date'], daily_visitors['visitors'], color='tab:blue')
plt.title("Total Visitors Over Time (All Restaurants)")
plt.xlabel("Date")
plt.ylabel("Total Visitors")
plt.show()



plt.figure(figsize=(8,5))
order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.barplot(data=df_train, x='weekday', y='visitors', estimator=np.mean, order=order)
plt.title("Average Visitors by Day of Week")
plt.ylabel("Average Visitors")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8,5))
sns.barplot(data=df_train, x='month', y='visitors', estimator=np.mean, palette='crest')
plt.title("Average Visitors per Month")
plt.ylabel("Average Visitors")
plt.show()


top_stores = (
    df_train.groupby('air_store_id')['visitors'].mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,6))
sns.barplot(x=top_stores.values, y=top_stores.index, palette='mako')
plt.title("Top 10 Restaurants by Average Visitors")
plt.xlabel("Average Visitors")
plt.ylabel("Store ID")
plt.show()


df_merged = pd.merge(df_train, df_store, on='air_store_id', how='left')

plt.figure(figsize=(10,6))
top_genres = (
    df_merged.groupby('air_genre_name')['visitors'].mean()
    .sort_values(ascending=False)
    .head(10)
)

sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis')
plt.title("Top 10 Restaurant Genres by Average Visitors")
plt.xlabel("Average Visitors")
plt.ylabel("Genre")
plt.show()


plt.figure(figsize=(10,6))
top_areas = (
    df_merged.groupby('air_area_name')['visitors'].mean()
    .sort_values(ascending=False)
    .head(10)
)

sns.barplot(x=top_areas.values, y=top_areas.index, palette='coolwarm')
plt.title("Top 10 Areas by Average Visitors")
plt.xlabel("Average Visitors")
plt.ylabel("City / Area")
plt.show()


example_id = df_train['air_store_id'].iloc[0]
example_df = df_train[df_train['air_store_id'] == example_id].sort_values('visit_date')

plt.figure(figsize=(12,5))
plt.plot(example_df['visit_date'], example_df['visitors'], marker='o', color='tab:orange')
plt.title(f"Visitors Over Time - Example Store: {example_id}")
plt.xlabel("Date")
plt.ylabel("Visitors")
plt.show()


plt.figure(figsize=(10,5))
sns.boxplot(data=df_train, x='month', y='visitors')
plt.title("Visitor Outliers by Month")
plt.xlabel("Month")
plt.ylabel("Visitors")
plt.show()


plt.figure(figsize=(6,4))
sns.heatmap(df_train[['visitors','year','month','day']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


import os
import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


input_dir = "/kaggle/input/recruit-restaurant-visitor-forecasting"
output_dir = "./extracted_data"
os.makedirs(output_dir, exist_ok=True)

def safe_extract(zip_path, extract_dir):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        print(f"âœ… Extracted: {os.path.basename(zip_path)}")
    else:
        print(f"âš ï¸� File not found: {zip_path}")


safe_extract(f"{input_dir}/air_visit_data.csv.zip", output_dir)
safe_extract(f"{input_dir}/sample_submission.csv.zip", output_dir)


df_train = pd.read_csv(f"{output_dir}/air_visit_data.csv")
df_test = pd.read_csv(f"{output_dir}/sample_submission.csv")

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)
print("\nTrain sample:")
print(df_train.head())


# Group by store and date
df = df_train.groupby(['air_store_id', 'visit_date'])['visitors'].sum().reset_index()



# Convert visit_date to datetime
df['visit_date'] = pd.to_datetime(df['visit_date'])


# Prepare test data
import re
df_test[['air', 'ds']] = df_test['id'].str.extract(r'(air_[A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})')
df_test.drop('id', axis=1, inplace=True)
df_test.rename(columns={'air': 'id'}, inplace=True)
df_test['ds'] = pd.to_datetime(df_test['ds'])

print("\nTest sample after split:")
print(df_test.head())


!pip install prophet -q
from prophet import Prophet
import logging, time
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)


forecast_table = []

store_groups = df.groupby('air_store_id')
test_groups = df_test.groupby('id')

start = time.time()
store_ids = df['air_store_id'].unique()
print(f"\nTotal stores: {len(store_ids)}\n")

for i, store_id in enumerate(store_ids, 1):
    if store_id not in test_groups.groups:
        continue

    # Prepare training data
    store_data = store_groups.get_group(store_id).rename(columns={'visit_date': 'ds', 'visitors': 'y'})
    
    # Ensure no missing or negative values
    store_data['y'] = store_data['y'].clip(lower=0)
    
    # Initialize Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.5,
        seasonality_prior_scale=10
    )
    
    # Fit model
    model.fit(store_data)
    
    # Prepare future (test) dates
    test_dates = test_groups.get_group(store_id)[['ds']].reset_index(drop=True)
    
    # Predict
    forecast = model.predict(test_dates)[['ds', 'yhat']]
    forecast['id'] = store_id
    forecast_table.append(forecast)

    if i % 50 == 0 or i == len(store_ids):
        print(f"âœ… Processed {i}/{len(store_ids)} stores...")

end = time.time()
print(f"\nâ�±ï¸� Forecasting finished in {end - start:.2f} seconds")


forecast_df = pd.concat(forecast_table, ignore_index=True)
forecast_df['yhat'] = forecast_df['yhat'].apply(lambda x: max(0, int(x)))  # visitors canâ€™t be negative


submission = forecast_df.copy()
submission['id'] = submission['id'] + '_' + submission['ds'].dt.strftime('%Y-%m-%d')
submission.rename(columns={'yhat': 'visitors'}, inplace=True)
submission = submission[['id', 'visitors']]


# Save submission
output_file = '/kaggle/working/submission.csv'
submission.to_csv(output_file, index=False)
print(f"\nâœ… Submission file saved to {output_file}")
print(submission.head())


import optuna
from prophet import Prophet
from sklearn.metrics import mean_squared_log_error

# Step 1: Select one specific store's data for tuning
store_id = "air_00a91d42b08b08d9"
sample_df = df[df['air_store_id'] == store_id][['visit_date', 'visitors']].rename(columns={
    'visit_date': 'ds',
    'visitors': 'y'
})
sample_df['y'] = sample_df['y'].clip(lower=0)  # Ensure no negative visitor counts

# Step 2: Define the Optuna objective function
def objective(trial):
    # Suggest Prophet hyperparameters to tune
    changepoint = trial.suggest_loguniform('changepoint_prior_scale', 0.01, 10.0)
    seasonality = trial.suggest_loguniform('seasonality_prior_scale', 0.01, 10.0)
    holidays = trial.suggest_loguniform('holidays_prior_scale', 0.01, 10.0)

    # Initialize Prophet with current trial parameters
    model = Prophet(
        changepoint_prior_scale=changepoint,
        seasonality_prior_scale=seasonality,
        holidays_prior_scale=holidays,
        yearly_seasonality=True,
        weekly_seasonality=True
    )

    # Fit the Prophet model
    model.fit(sample_df)

    # Predict using the same training data for evaluation
    forecast = model.predict(sample_df[['ds']])

    # Compute Mean Squared Logarithmic Error (MSLE)
    return mean_squared_log_error(sample_df['y'], forecast['yhat'])

# Step 3: Create an Optuna study and run optimization
study = optuna.create_study(direction='minimize')
study.optimize(lambda t: objective(t), n_trials=10)



print("âœ… Best Parameters:")
print(study.best_params)

