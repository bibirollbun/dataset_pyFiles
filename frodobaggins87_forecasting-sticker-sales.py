!pip install autogluon holidays


import pandas as pd
import matplotlib.pyplot as plt
import autogluon
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import holidays as hd


!nvidia-smi


original_train_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
print(f"The training data has {original_train_data.shape[0]} rows")
original_train_data.head(10)


test_data = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
print(f"The testing data has {test_data.shape[0]} rows")
print(test_data.head())
print(test_data.tail())


submission_format = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
print(submission_format.shape)
print(submission_format.head())
print(submission_format.tail())


def check_nan(df:pd.DataFrame):
  # Check for NaN values
  has_nan = df.isnull().any().any()
  return has_nan


print(f"The training data has NaN values: {check_nan(original_train_data)}")
print(f"The testing data has NaN values: {check_nan(test_data)}")


original_train_data.isnull().any()


# Remove rows where 'num_sold' column has NaN values
#train_data = original_train_data.dropna(subset=['num_sold']).copy()
#print(f"The training data has {train_data.shape[0]} rows")
#train_data.head(10)
#REMOVED ABOVE CODE AS IT LED TO INCOMPLETE DATA
train_data=original_train_data.copy()


#checking datatypes of each column
train_data.dtypes


# Convert date column to datetime
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date']=pd.to_datetime(test_data['date'])


train_data.describe(include='all')


#make a function to check if a date is a holiday in a specific country or not
def is_holiday(row):
  #get list of holidays in specific country
  country_holidays=hd.country_holidays(row['country'])

  #Return True if date belongs in that list
  return row['date'] in country_holidays


#apply is_holiday on training and testing datasets
train_data['holiday']=train_data.apply(is_holiday, axis=1)
test_data['holiday']=test_data.apply(is_holiday, axis=1)


#make new column in training data containing day of week
train_data['day_of_week']=train_data['date'].dt.day_name()
test_data['day_of_week']=test_data['date'].dt.day_name()


#getting season on the date
def get_season(date):
  month=date.month
  season='spring' if 3<=month<=5 else 'summer' if 6<=month<=8 else 'autumn' if 9<=month<=11 else 'winter'
  return season

#applying above function on training and testing data
train_data['season']=train_data['date'].apply(get_season)
test_data['season']=test_data['date'].apply(get_season)
train_data['season']


#Make fiscal Quarter Column with 1 from April to June, 2 from July to September, 3 from October to December, 4 January to March
def get_fiscal_quarter(date):
  month = date.month
  quarter= 1 if month in [4,5,6] else 2 if month in [7,8,9] else 3 if month in [10,11,12] else 4 if month in [1,2,3] else 0
  return quarter


#apply above function to training and testing dataset
train_data['fiscal_quarter']=train_data['date'].apply(get_fiscal_quarter)
test_data['fiscal_quarter']=test_data['date'].apply(get_fiscal_quarter)


#Make column to check if date is near new years or not(Between 15 December to 15 January)
def near_new_year(date):
  month=date.month
  day=date.day
  return True if (15<=day<=31 and month==12) or (1<=day<=15 and month==1) else False

#Apply function to both the datasets
train_data['near_new_year']=train_data['date'].apply(near_new_year)
test_data['near_new_year']=test_data['date'].apply(near_new_year)


#Check if date is near start of the month (up till 5th of the month)
def near_start_of_month(date):
  day=date.day
  return True if day<=5 else False

#Apply to both datasets
train_data['near_start_of_month']=train_data['date'].apply(near_start_of_month)
test_data['near_start_of_month']=test_data['date'].apply(near_start_of_month)


# Count the number of rows for each country
country_counts = train_data['country'].value_counts()

# Display the result
print(country_counts)


# Count the number of rows for each country
product_counts = train_data['product'].value_counts()

# Display the result
print(product_counts)


# Count the number of rows for each country
store_counts = train_data['store'].value_counts()

# Display the result
print(store_counts)


print(f'countries in training dataset:{train_data["country"].unique()}')
print(f'countries in testing dataset:{test_data["country"].unique()}')


print(f'stores in training dataset:{train_data["store"].unique()}')
print(f'stores in testing dataset:{test_data["store"].unique()}')


print(f'products in training dataset:{train_data["product"].unique()}')
print(f'products in testing dataset:{test_data["product"].unique()}')


print(f' Number of Unique values of all categorical variables:Day of Week:{train_data["day_of_week"].nunique()}, Quarter: {train_data["fiscal_quarter"].nunique()}, Holiday: {train_data["holiday"].nunique()}, New Year: {train_data["near_new_year"].nunique()}, Start of Month: {train_data["near_start_of_month"].nunique()}, Season: {train_data["season"].nunique()}')


# 1. Pie Chart for `num_sold` against `store`, `country`, and `product`
categories = ['store', 'country', 'product']
for category in categories:
    aggregated = train_data.groupby(category)['num_sold'].sum()
    plt.figure(figsize=(8, 6))
    plt.pie(aggregated, labels=aggregated.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Proportion of Sales by {category.capitalize()}')
    plt.show()



# Function to plot grouped data
def plot_grouped_data(df, group_column, title):
    grouped = df.groupby([group_column, 'date'])['num_sold'].sum().reset_index()
    unique_groups = grouped[group_column].unique()

    plt.figure(figsize=(12, 6))
    for group in unique_groups:
        group_data = grouped[grouped[group_column] == group]
        plt.plot(group_data['date'], group_data['num_sold'], marker='o', label=f"{group}")

    plt.xlabel('Date')
    plt.ylabel('Number Sold')
    plt.title(title)
    plt.legend(title=group_column.capitalize())
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot for each category
plot_grouped_data(train_data, 'country', 'Sales Over Time by Country')
plot_grouped_data(train_data, 'store', 'Sales Over Time by Store')
plot_grouped_data(train_data, 'product', 'Sales Over Time by Product')


import numpy as np
print(train_data['num_sold'].skew())  # A skewness > 1 suggests log transformation might help



# Create transformed versions of num_sold
train_data['log_num_sold'] = np.log(train_data['num_sold'] + 1)  # Add 1 to avoid issues with zeros
train_data['sqrt_num_sold'] = np.sqrt(train_data['num_sold'])
train_data['cbrt_num_sold'] = np.cbrt(train_data['num_sold'])

# Calculate skewness for each column
skewness_original = train_data['num_sold'].skew()
skewness_log = train_data['log_num_sold'].skew()
skewness_sqrt = train_data['sqrt_num_sold'].skew()
skewness_cbrt = train_data['cbrt_num_sold'].skew()

# Plot distributions with skewness in titles
plt.figure(figsize=(16, 8))

# Original data
plt.subplot(2, 2, 1)
train_data['num_sold'].hist(bins=20, color='blue', alpha=0.7)
plt.title(f'Original Distribution of num_sold (Skewness: {skewness_original:.2f})')
plt.xlabel('num_sold')
plt.ylabel('Frequency')

# Log-transformed data
plt.subplot(2, 2, 2)
train_data['log_num_sold'].hist(bins=20, color='green', alpha=0.7)
plt.title(f'Log-Transformed Distribution (Skewness: {skewness_log:.2f})')
plt.xlabel('log(num_sold + 1)')
plt.ylabel('Frequency')

# Square root-transformed data
plt.subplot(2, 2, 3)
train_data['sqrt_num_sold'].hist(bins=20, color='orange', alpha=0.7)
plt.title(f'Square Root-Transformed Distribution (Skewness: {skewness_sqrt:.2f})')
plt.xlabel('sqrt(num_sold)')
plt.ylabel('Frequency')

# Cube root-transformed data
plt.subplot(2, 2, 4)
train_data['cbrt_num_sold'].hist(bins=20, color='red', alpha=0.7)
plt.title(f'Cube Root-Transformed Distribution (Skewness: {skewness_cbrt:.2f})')
plt.xlabel('cbrt(num_sold)')
plt.ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()


train_data.head()


# Ensure 'date' column is in datetime format
train_data['date'] = pd.to_datetime(train_data['date'])


train_data.dtypes


# Rename 'date' column to 'timestamp' to match AutoGluon's expected format
train_data.rename(columns={'date': 'timestamp'}, inplace=True)



# Create the 'item_id' column that combines 'country', 'store', and 'product' to uniquely identify time series, since this will be used, the 3 columns are not required to be input in model separately
train_data['item_id'] = train_data['country'] + '_' + train_data['store'] + '_' + train_data['product']



train_data['item_id'].nunique()


# Step 1: Prepare data for TimeSeriesDataFrame
# Assuming train_data already has columns: 'timestamp', 'item_id', 'holiday', and 'sqrt_num_sold'
train_data_ts = TimeSeriesDataFrame(train_data[['timestamp', 'item_id', 'holiday', 'day_of_week','season', 'fiscal_quarter', 'near_new_year', 'near_start_of_month', 'sqrt_num_sold']])

# Step 2: Handle missing values in TimeSeriesDataFrame
# This uses forward fill followed by backward fill by default
train_data_ts = train_data_ts.fill_missing_values()

# Step 3: Adjust frequency (e.g., 'D' for daily)
train_data_ts.convert_frequency(freq='D')

# Step 4: Initialize the TimeSeriesPredictor
predictor = TimeSeriesPredictor(
    target='sqrt_num_sold',
    prediction_length=test_data['date'].nunique(),  # Predicting for number of unique dates in test_data
    freq='D',
    eval_metric='MAPE'  # Set MAPE as the evaluation metric
)

# Step 5: Specify GPU usage for deep learning models
hyperparameters = {
    "DeepAR": {"use_gpu": True},  # Enable GPU for DeepAR
    "TemporalFusionTransformer": {"use_gpu": True},  # Enable GPU for TFT
    "PatchTST": {"use_gpu": True},  # Enable GPU for PatchTST
    "TiDE": {"use_gpu": True}  # Enable GPU for TiDE
}

# Step 6: Fit the model with the data
predictor.fit(train_data_ts, presets='best_quality', hyperparameters=hyperparameters)

# Step 7: Retrieve model performance and sort by MAPE
performance = predictor.leaderboard(train_data_ts, silent=True)
performance_sorted = performance[['model', 'score_val']].sort_values(by='score_val', ascending=True)

# Step 8: Print MAPE of models in increasing order
print("MAPE of models in increasing order:")
print(performance_sorted)


test_data.dtypes


#convert date to detetime
test_data['date'] = pd.to_datetime(test_data['date'])


# Rename 'date' column to 'timestamp' to match AutoGluon's expected format
test_data.rename(columns={'date': 'timestamp'}, inplace=True)
print(test_data['timestamp'].dtype)


# Create 'item_id' for future data
test_data['item_id'] = test_data['country'] + '_' + test_data['store'] + '_' + test_data['product']


test_data['item_id'] = test_data['item_id'].astype(str)


test_data.columns


# Prepare future data as TimeSeriesDataFrame
test_data_filtered = test_data[['timestamp', 'item_id', 'holiday']]
test_data_filtered = test_data_filtered.set_index(['item_id', 'timestamp'])
test_data_ts = TimeSeriesDataFrame(test_data_filtered)#filtering out required columns for prediction



print(test_data_ts.index.names)
print(test_data_ts.columns)
print(test_data.head())


# Make predictions for sqrt_num_sold
future_forecast = predictor.predict(data=train_data_ts)
future_forecast



future_forecast.columns


# Convert sqrt_num_sold predictions back to num_sold
future_forecast['num_sold'] = future_forecast['mean']**2


#left join future forecast to test_data to get correct ids with the num_sold forecast
final_forecast = future_forecast.merge(test_data[['id', 'item_id', 'timestamp']],
                                       on=['item_id', 'timestamp'],
                                       how='left')
# Display the predictions
print("Future predictions for num_sold:")
print(final_forecast[['id', 'num_sold']])


final_forecast.columns


#Check for NaN values in all columns
future_forecast.isnull().any()


# Convert both columns to lists
final_forecast_ids = final_forecast['id'].tolist()
submission_format_ids = submission_format['id'].tolist()

difference = list(set(submission_format_ids)-set(final_forecast_ids))
difference #empty list implies no difference so our output matches expected submission format


#save final_forecast['id', 'num_sold']
final_forecast[['id', 'num_sold']].to_csv('/kaggle/working/submission.csv', index=False)





'''# Get the name of the best model
best_model = predictor.model_best

# Extract the best model's object
best_model_obj = predictor._trainer.load_model(best_model)

# Save the best model separately
import pickle
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model_obj, f)

print(f"Best model '{best_model}' saved as 'best_model.pkl'")'''


'''from google.colab import files

# Download the saved best model file
files.download("best_model.pkl")'''

