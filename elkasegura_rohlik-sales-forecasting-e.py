!pip install darts[torch] scikit-learn==1.2.2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
!pip install statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from darts import TimeSeries
from darts.models import BlockRNNModel
from darts.dataprocessing.transformers import Scaler


import pandas as pd
# Load the data into pandas DataFrames
sales_train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv')
test_weights = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')
solution = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')
sales_test =pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv')


from IPython.display import display

# Display the first few rows of each file as tables
print("Sales Train Data:")
display(sales_train.head())

print("Inventory Data:")
display(inventory.head())

print("Calendar Data:")
display(calendar.head())

print("Test Weights Data:")
display(test_weights.head())


# Perform the merge using unique_id and warehouse
sales_train_inventory = pd.merge(
    sales_train,
    inventory,
    on=["unique_id", "warehouse"],
    how="inner"  # Change to "left", "right", or "outer" based on requirements
)

sales_train_inventory_calendar = pd.merge(
    sales_train_inventory,
    calendar,
    on=["warehouse", "date"],
    how="left"  # Use "left" to keep all rows from merged_data
)

final_combined_data = pd.merge(
    sales_train_inventory_calendar,
    test_weights,
    on="unique_id",
    how="left"  # Use "left" to keep all rows from final_data
)

# Display the first rows of the combined result
print(final_combined_data.columns)
print(final_combined_data.shape)
final_combined_data.head()


# Organize the final dataset by unique_id, date, and warehouse
final_combined_data = final_combined_data.sort_values(by=["product_unique_id", "date", "warehouse"])

# Check for duplicates based on unique_id, date, and warehouse
duplicates = final_combined_data.duplicated(subset=["product_unique_id", "date", "warehouse"])
if duplicates.any():
    print("Duplicates found:")
    print(final_combined_data[duplicates])
else:
    print("No duplicates found.")

# Check the first rows of the organized dataset
final_combined_data.head()

null_check = final_combined_data.isnull().sum()
print(null_check)


# Check for nulls and zeros
print(final_combined_data[['total_orders', 'sales']].isnull().sum())
final_combined_data[['total_orders', 'sales']] = final_combined_data[['total_orders', 'sales']].fillna(0)


print(final_combined_data[['total_orders', 'sales']].isnull().sum())


# Get data types of each column
data_types = final_combined_data.dtypes

# Display the results
print(data_types)



# Select the relevant columns
selected_columns = ['sales','total_orders', 'sell_price_main', 'availability',
                    'type_0_discount', 'type_1_discount', 'type_2_discount',
                    'type_3_discount', 'type_4_discount', 'type_5_discount',
                    'type_6_discount', 'holiday', 'shops_closed',
                    'winter_school_holidays', 'school_holidays', 'weight']

#Calculate the correlation matrix
correlation_matrix = final_combined_data[selected_columns].corr()

# Display the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Selected Variables')
plt.show()


# Filter relevant columns
data_filtered = final_combined_data[['date', 'sales', 'warehouse']].copy()

# Convert 'date' to datetime format
data_filtered['date'] = pd.to_datetime(data_filtered['date'])

# Filter data up to May 31, 2023 (fixed incorrect year in your code)
data_filtered = data_filtered[data_filtered['date'] <= '2023-05-31']

# Group by warehouse and date (daily sales)
daily_sales_warehouse = data_filtered.groupby(
    ['warehouse', 'date']
)['sales'].sum().reset_index()

# Set seaborn style
sns.set(style="whitegrid")

# Create figure
plt.figure(figsize=(12, 6))

# Plot daily sales for each warehouse
sns.lineplot(
    data=daily_sales_warehouse,
    x='date',
    y='sales',
    hue='warehouse',
    marker="o",
    linewidth=1
)

# Set labels and title
plt.xlabel("Date")
plt.ylabel("Daily Sales")
plt.title("Daily Sales by Warehouse")  # Fixed title
plt.xticks(rotation=45)
plt.legend(title="Warehouse")
plt.grid(True)

# Show the plot
plt.show()


import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*InterpolationWarning.*")

# Prepare a dictionary to store results ADF test
stationarity_results = {}

# Loop through each warehouse and perform for ADF test
for warehouse in daily_sales_warehouse['warehouse'].unique():
    # Filter the data for the warehouse
    warehouse_data = daily_sales_warehouse[daily_sales_warehouse['warehouse'] == warehouse]
    # Extract sales values
    sales_series = warehouse_data['sales']
    # Perform the ADF test
    adf_result = adfuller(sales_series)
    # Store the results for ADF test
    stationarity_results[warehouse] = {
        'ADF Statistic': adf_result[0],
        'ADF p-value': adf_result[1],
        'ADF Stationary': adf_result[1] < 0.05,  # Stationary if p-value < 0.05
    }
# Prepare a list to store the results
results_list = []

# Loop through each warehouse and perform ADF test
for warehouse, result in stationarity_results.items():
    # Create a dictionary for the current warehouse results
    results_list.append({
        'Warehouse': warehouse,
        'ADF Statistic': result['ADF Statistic'],
        'ADF p-value': result['ADF p-value'],
        'ADF Stationary': 'Yes' if result['ADF Stationary'] else 'No',
    })
# Convert the results into a DataFrame
results_df = pd.DataFrame(results_list)

# Display the results table
print(results_df)



# Filter and compute the first difference in one step
brno_1_diff = (
    daily_sales_warehouse[daily_sales_warehouse['warehouse'] == 'Brno_1']
    .copy()
    .assign(sales_diff=lambda df: df['sales'].diff())
    .dropna(subset=['sales_diff'])
)

# Create figure and axes
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Plot original sales
axes[0].plot(brno_1_diff['date'], brno_1_diff['sales'], color='blue')
axes[0].set_title("Original Sales - Brno_1")
axes[0].set_ylabel("Sales")

# Plot first difference of sales
axes[1].plot(brno_1_diff['date'], brno_1_diff['sales_diff'], color='red')
axes[1].set_title("First Difference - Brno_1")
axes[1].set_ylabel("Sales Difference")

# Formatting
for ax in axes:
    ax.set_xlabel("Date")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)

plt.tight_layout()
plt.show()

# Perform ADF test
adf_stat, adf_pvalue, *_ = adfuller(brno_1_diff['sales_diff'])
print(f"ADF Statistic: {adf_stat:.4f}")
print(f"ADF p-value: {adf_pvalue:.4f}")
print(f"Is the series stationary? {'Yes' if adf_pvalue < 0.05 else 'No'}")


# Plot ACF and PACF for original and differenced series
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

# ACF and PACF for original sales
plot_acf(brno_1_diff['sales'], lags=20, ax=ax[0, 0], title="ACF - Original Sales")
plot_pacf(brno_1_diff['sales'], lags=20, ax=ax[0, 1], title="PACF - Original Sales")

# ACF and PACF for first differenced sales
plot_acf(brno_1_diff['sales_diff'], lags=20, ax=ax[1, 0], title="ACF - First Difference")
plot_pacf(brno_1_diff['sales_diff'], lags=20, ax=ax[1, 1], title="PACF - First Difference")

plt.tight_layout()
plt.show()


# Filter data for the years 2021 to 2023
brno_1_filtered = brno_1_diff[(brno_1_diff['date'] >= '2021-01-01') & (brno_1_diff['date'] <= '2023-12-31')]

# Ensure there are enough observations for decomposition
print(brno_1_filtered.shape)

# Perform seasonal decomposition with the adjusted period
brno_1_series = brno_1_filtered[['date', 'sales']].set_index('date')
result = seasonal_decompose(brno_1_series['sales'], model='additive', period=30)

# Plot the decomposition
result.plot()
plt.show()

print(brno_1_filtered)



# Fit an ARIMA model
# Using p=1, d=1, q=1 as an example, but you can adjust based on ACF/PACF analysis
model_1_0_1 = ARIMA(brno_1_diff['sales'], order=(1, 0, 1))
model_1_0_1 = model_1_0_1.fit()

model_0_0_1 = ARIMA(brno_1_diff['sales'], order=(0, 0, 1))
model_0_0_1 = model_0_0_1.fit()

model_1_0_0 = ARIMA(brno_1_diff['sales'], order=(1, 0, 0))
model_1_0_0 = model_1_0_0.fit()


print(model_1_0_1.summary())
print(model_0_0_1.summary())
print(model_1_0_0.summary())


# Load the dataset
data = brno_1_diff[['date', 'sales']]
brno_1 = pd.DataFrame(data)
brno_1['date'] = pd.to_datetime(brno_1['date'])
brno_1.set_index('date', inplace=True)

# Predictions for the next 12 months
n_steps = 15
forecast = model_1_0_1.get_forecast(steps=n_steps)
forecast_index = pd.date_range(brno_1.index[-1], periods=n_steps + 1, freq="D")[1:]
forecast_values = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(brno_1.index, brno_1['sales'], label="Historical data", color="blue")
plt.plot(forecast_index, forecast_values, label="Predictions", color="orange")
plt.fill_between(
    forecast_index,
    confidence_intervals.iloc[:, 0],
    confidence_intervals.iloc[:, 1],
    color="orange",
    alpha=0.3,
    label="Confidence interval"
)
plt.title("Sales Differences Forecast (ARIMA(1,0,1))")
plt.xlabel("Date")
plt.ylabel("Sales differences")
plt.legend()
plt.grid()
plt.show()

# Forecast results
forecast_df = pd.DataFrame({
    "date": forecast_index,
    "forecast": forecast_values,
    "lower_bound": confidence_intervals.iloc[:, 0],
    "upper_bound": confidence_intervals.iloc[:, 1]
})
print(forecast_df)


final_combined_data  = final_combined_data.drop(columns=['holiday_name'])

def plot_sales_by_product(df):
    # Convert the 'date' column to datetime type if it's not already
    df['date'] = pd.to_datetime(df['date'])

    # Get the first 4 unique products
    products = df['name'].unique()[:4]

    # Create a figure with subplots (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()  # Flatten to iterate more easily

    # Plot sales for each product
    for i, product in enumerate(products):
        product_data = df[df['name'] == product]

        ax = axes[i]
        ax.plot(product_data['date'], product_data['sales'], label=product, color='b', marker='o', linestyle='-')
        ax.set_title(f'Sales Trend - {product}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

    # Remove extra axes if there are fewer than 4 products
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout and show plot
    plt.suptitle('Time Series of Sales for First 4 Products', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust spacing for the title
    plt.show()

# Call the function with your DataFrame
plot_sales_by_product(final_combined_data)


# Convert 'date' to datetime format
final_combined_data['date'] = pd.to_datetime(final_combined_data['date'])

# Select relevant columns (only the necessary features)
df = final_combined_data[['unique_id', 'date', 'sales', 'warehouse', 'total_orders', 'sell_price_main',
                          'type_0_discount', 'type_1_discount', 'type_2_discount', 'type_3_discount',
                          'type_4_discount', 'type_5_discount', 'type_6_discount']].copy()

# Add temporal features
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['day_of_year'] = df['date'].dt.dayofyear

# Encode categorical columns
categorical_columns = ['warehouse']
df[categorical_columns] = df[categorical_columns].apply(lambda x: LabelEncoder().fit_transform(x))

# Split into training and testing sets
train_data = df[df['date'] < '2023-01-01']
test_data = df[(df['date'] >= '2023-01-01')]

# Define features (X) and target (y)
X_train, y_train = train_data.drop(columns=['sales', 'date']), train_data['sales']
X_test, y_test = test_data.drop(columns=['sales', 'date']), test_data['sales']

# Train the model
model_LGBM = lgb.LGBMRegressor(learning_rate=0.1, n_estimators=150, boosting_type='gbdt', metric='mae')
model_LGBM.fit(X_train, y_train)

# Predict and calculate MAE
y_pred = model_LGBM.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Display results
#print(f'MAE: {mae}')

# Filter the data for dates >= '2023-01-01'
filtered_data = final_combined_data[final_combined_data['date'] >= '2023-01-01']

# Calculate absolute errors
absolute_errors = np.abs(y_test - y_pred)

# Ensure the length matches the filtered data
filtered_weights = filtered_data['weight'].values[:len(absolute_errors)]

# Compute WMAE
weighted_errors = filtered_weights * absolute_errors
wmae = weighted_errors.sum() / filtered_weights.sum()

# Display the result
#print(f'WMAE: {wmae}')



# Ensure 'date' is in datetime format
predict_sales = sales_test
sales_test['date'] = pd.to_datetime(sales_test['date'])

# Sort the data by 'unique_id' and 'date'
#sales_test = sales_test.sort_values(by=['unique_id', 'date'])
#predict_sales = predict_sales.sort_values(by=['unique_id', 'date'])

# Add temporal features to sales_test
sales_test['day_of_week'] = sales_test['date'].dt.dayofweek
sales_test['month'] = sales_test['date'].dt.month
sales_test['year'] = sales_test['date'].dt.year
sales_test['day_of_year'] = sales_test['date'].dt.dayofyear

# Encode categorical columns
categorical_columns = ['warehouse']
sales_test[categorical_columns] = sales_test[categorical_columns].apply(lambda x: LabelEncoder().fit_transform(x))

# Drop 'date' columns for prediction
sales_test = sales_test.drop(columns=['date'])

# Predict using the trained LGBM model
y_pred = model_LGBM.predict(sales_test)

# Print the predictions
#print(y_pred)



# Add the predictions as a new column 'sales' in the weights_test_data dataframe
predict_sales['sales'] = y_pred

# Create 'id' column by applying a function to concatenate 'unique_id' and 'date'
predict_sales['id'] = predict_sales.apply(lambda row: f"{row['unique_id']}_{pd.to_datetime(row['date']).strftime('%Y-%m-%d')}", axis=1)

# Rename the 'sales' column to 'sales_hat'
predict_sales = predict_sales.rename(columns={'sales': 'sales_hat'})

# Select 'id' and 'sales_hat' columns and save as CSV
predict_sales[['id', 'sales_hat']].to_csv('submission.csv', index=False)


# Group by 'name' and 'date', and sum 'sales' once
productos = final_combined_data.groupby(['name', 'date'], as_index=False)['sales'].sum()

# Dictionary to store processed data for each product
product_data = {}

# Filter and store the grouped data for each product
for product in productos['name'].unique():
    # Filter data for the current product
    product_data[product] = productos[productos['name'] == product][['date', 'sales']]

# Optionally, check the processed data for one product (e.g., the first one)
print(productos['name'].unique())



Croissant_9 = product_data['Croissant_9']
print(Croissant_9.head(6))


Croissant_9['date'] = pd.to_datetime(Croissant_9['date'])
fig, ax = plt.subplots()
ax.plot(Croissant_9['date'], Croissant_9['sales'], label='Croissant_9', color='b', marker='o', linestyle='-')
ax.set_title(f'Sales Trend - {"Croissant_9"}')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend()
plt.xticks(rotation=45)  # Optional: Rotate date labels for better readability
plt.tight_layout()      # Optional: To prevent clipping of labels
plt.show()


# Assuming 'product_data' is a dictionary with product names and their corresponding data
df = product_data['Croissant_9']

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Ensure the DataFrame is sorted by date
df = df.sort_values(by='date')

# Generate the complete date range from the first to the last date in the dataset
full_date_range = pd.date_range(df['date'].min(), df['date'].max(), freq='D')

# Identify missing dates
missing_dates = set(full_date_range) - set(df['date'])

# Initialize an empty list to store rows to be added
rows_to_add = []

# For each missing date, add a row with sales set to 0
for missing_date in sorted(missing_dates):
    rows_to_add.append({'date': missing_date, 'sales': 0})

# Create a DataFrame with the new rows and append it to the original dataset
if rows_to_add:
    new_rows_df = pd.DataFrame(rows_to_add)
    df = pd.concat([df, new_rows_df], ignore_index=True)

# Reorder the dataset by date
df = df.sort_values(by="date").reset_index(drop=True)

# Print the updated DataFrame with the missing dates inserted
print(df.tail())



from darts.metrics import mae
# Define a function to train and forecast using the BlockRNNModel
def train_and_forecast(
    data,
    name,
    input_chunk_length,  # Number of past time steps used as input for the model
    output_chunk_length, # Number of future time steps to predict
    model_type,          # Underlying architecture (LSTM/GRU/RNN)
    random_state,        # Seed for reproducibility
    n_epochs,            # Number of training epochs
    batch_size,          # Batch size for training
    dropout,             # Dropout rate for regularization
    lr,                  # Learning rate for the optimizer
    activation           # Activation function (e.g., 'relu', 'tanh')
):
    # Ensure the data does not contain null values
    data = pd.DataFrame(data)
    data['date'] = pd.to_datetime(data['date'])

    # Convert the series into a TimeSeries object
    ts = TimeSeries.from_dataframe(data, "date", "sales", freq='D')

    # Normalize the series (optional, improves training stability)
    scaler = Scaler()
    ts = scaler.fit_transform(ts)

    # Split the series into training and validation sets
    train, val = ts.split_after(pd.Timestamp("2024-04-30"))

    # Train the BlockRNNModel with dynamic parameters
    model = BlockRNNModel(
        input_chunk_length=input_chunk_length,  # Number of past time steps used for input
        output_chunk_length=output_chunk_length,  # Number of future time steps to predict
        model=model_type,           # Architecture: LSTM (Long Short-Term Memory), GRU, or RNN
        random_state=random_state,  # Seed for reproducibility
        n_epochs=n_epochs,          # Total number of training epochs
        batch_size=batch_size,      # Number of samples per training batch
        dropout=dropout,            # Dropout rate for regularization to prevent overfitting
        optimizer_kwargs={"lr": lr}, # Learning rate for the optimizer
        activation=activation,        # Activation function to apply in the model
        n_rnn_layers=3
    )

    # Train the model on the training data
    model.fit(train)

    # Inverse transform the training and validation sets
    train = scaler.inverse_transform(train)
    val = scaler.inverse_transform(val)

    # Forecast the future for the specified prediction horizon
    forecast = model.predict(n=output_chunk_length, series=train)

    # Inverse transform the forecasted values
    forecast = scaler.inverse_transform(forecast)

    # Generate the dates for the forecasted values
    forecast_dates = pd.date_range(start=train.time_index[-1] + pd.Timedelta(days=1), periods=output_chunk_length, freq='D')

    # Convert the forecasted values into a DataFrame
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'sales': forecast.values().flatten()
    })

    # Calculate RMSE for evaluation
    mae_value = mae(val, forecast)
    print(f"MAE for {name}: {mae_value:.2f}")

    # Plot the training, validation, and forecasted values
    plt.figure(figsize=(12, 6))
    train.plot(label=f"Train {name}", color="blue")
    val.plot(label=f"Validation {name}", color="orange")
    forecast.plot(label=f"Forecast {name}", color="green", linestyle="--")
    plt.title(f"Forecast for {name}")
    plt.legend()
    plt.show()

    # Return the forecast as a DataFrame
    return forecast_df


Croissant_9 = df 
# Apply the function to each of the series and save the predictions
forecast_Croissant_9 = train_and_forecast(
    data=Croissant_9,
    name="Croissant_9",
    input_chunk_length=365,
    output_chunk_length=60,
    model_type="LSTM",
    random_state=42,
    n_epochs=100,
    batch_size=128,
    dropout=0.5,
    lr=0.1,
    activation="relu"
)

print(forecast_Croissant_9)

