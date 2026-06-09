import polars as pl
import pandas as pd

# Define the path to the parquet file
file_path = '../input/jane-street-real-time-market-data-forecasting/train.parquet'

# Load data lazily with Polars
data = pl.scan_parquet(file_path)

# Sort data by date_id and time_id
sorted_data = data.sort(['date_id', 'time_id'])

# Define batch size for processing
batch_size = 500_000  # Size of each batch

# Define start and end indices for the first batch
start_idx = 0
end_idx = batch_size

print("Analyzing the first batch...")

# Collect the first batch using Polars
first_batch_polars = sorted_data.slice(start_idx, end_idx - start_idx).collect()

# Convert the Polars DataFrame to Pandas for detailed analysis
first_batch = first_batch_polars.to_pandas()

# Check for NaN, non-numeric, and other unexpected values
nan_columns = {}
unexpected_details = {}

for col in first_batch.columns:
    if pd.api.types.is_numeric_dtype(first_batch[col]):  # Process only numeric columns
        # Check for NaN values
        nan_count = first_batch[col].isna().sum()
        if nan_count > 0:
            nan_columns[col] = nan_count
        
        # Check for values that are neither numeric nor NaN
        unexpected_values = first_batch[~first_batch[col].apply(lambda x: pd.isna(x) or isinstance(x, (int, float)))]
        if not unexpected_values.empty:
            unexpected_details[col] = unexpected_values[col].unique().tolist()

# Display NaN results
if nan_columns:
    print("\nColumns with NaN values:")
    for col, count in nan_columns.items():
        print(f"Column '{col}' has {count} NaN values.")
else:
    print("No NaN values found in numeric columns of the first batch.")

# Display unexpected values
if unexpected_details:
    print("\nUnexpected Values Found in Numeric Columns:")
    for col, values in unexpected_details.items():
        print(f"Column '{col}': {values}")
else:
    print("No unexpected values (other than numeric or NaN) found in numeric columns of the first batch.")

# Print sample rows from the batch
print("\nSample rows from the first batch:")
print(first_batch.sample(2))
print("\n" + "=" * 80 + "\n")



import polars as pl
import pandas as pd

# Define the path to the parquet file
file_path = '../input/jane-street-real-time-market-data-forecasting/train.parquet'

# Load data lazily with Polars
data = pl.scan_parquet(file_path)

# Sort data by date_id and time_id
sorted_data = data.sort(['date_id', 'time_id'])

# Define batch size for processing
batch_size = 500_000  # Size of each batch

# Define start and end indices for the second batch
start_idx = batch_size
end_idx = start_idx + batch_size

print("Analyzing the second batch...")

# Collect the second batch using Polars
second_batch_polars = sorted_data.slice(start_idx, end_idx - start_idx).collect()

# Convert the Polars DataFrame to Pandas for detailed analysis
second_batch = second_batch_polars.to_pandas()

# Check for NaN, non-numeric, and other unexpected values
nan_columns = {}
unexpected_details = {}

for col in second_batch.columns:
    if pd.api.types.is_numeric_dtype(second_batch[col]):  # Process only numeric columns
        # Check for NaN values
        nan_count = second_batch[col].isna().sum()
        if nan_count > 0:
            nan_columns[col] = nan_count
        
        # Check for values that are neither numeric nor NaN
        unexpected_values = second_batch[~second_batch[col].apply(lambda x: pd.isna(x) or isinstance(x, (int, float)))]
        if not unexpected_values.empty:
            unexpected_details[col] = unexpected_values[col].unique().tolist()

# Display NaN results
if nan_columns:
    print("\nColumns with NaN values:")
    for col, count in nan_columns.items():
        print(f"Column '{col}' has {count} NaN values.")
else:
    print("No NaN values found in numeric columns of the second batch.")

# Display unexpected values
if unexpected_details:
    print("\nUnexpected Values Found in Numeric Columns:")
    for col, values in unexpected_details.items():
        print(f"Column '{col}': {values}")
else:
    print("No unexpected values (other than numeric or NaN) found in numeric columns of the second batch.")

# Print sample rows from the batch
print("\nSample rows from the second batch:")
print(second_batch.sample(2))
print("\n" + "=" * 80 + "\n")



import polars as pl
import pandas as pd

# Define the path to the parquet file
file_path = '../input/jane-street-real-time-market-data-forecasting/train.parquet'

# Load data lazily with Polars
data = pl.scan_parquet(file_path)

# Sort data by date_id and time_id
sorted_data = data.sort(['date_id', 'time_id'])

# Define batch size for processing
batch_size = 500_000  # Size of each batch

# Define start and end indices for the batch to analyze
start_idx = 0  # Adjust as needed for other batches
end_idx = batch_size

print(f"Analyzing statistics and correlations for batch {start_idx // batch_size + 1}...")

# Collect the batch using Polars
batch_polars = sorted_data.slice(start_idx, end_idx - start_idx).collect()

# Convert the Polars DataFrame to Pandas for detailed analysis
batch = batch_polars.to_pandas()

# if needed:
# Compute basic statistics
# print("\nBasic Statistics:")
# basic_stats = batch.describe(include='all').T
# basic_stats['NaN Count'] = batch.isna().sum()
# basic_stats['Unique Count'] = batch.nunique()
# print(basic_stats.head(10))  # Print only the first 10 rows for brevity

# Calculate correlations and filter significant ones
correlation_threshold = 0.3  # Threshold for significant correlations
print("\nSignificant Correlations:")

correlations = batch.corr(method='pearson')  # Compute the correlation matrix
if 'responder_6' in correlations.columns:
    responder_6_corr = correlations['responder_6'].sort_values(ascending=False)
    significant_corr = responder_6_corr[responder_6_corr.abs() > correlation_threshold]
    print("\nCorrelations with responder_6 (|correlation| > 0.3):")
    print(significant_corr)
else:
    print("Column 'responder_6' not found in the data.")

# if needed:
# Mask to show only significant correlations in the entire matrix
# significant_matrix = correlations.where(correlations.abs() > correlation_threshold)
# significant_matrix = significant_matrix.dropna(how='all', axis=1).dropna(how='all', axis=0)
# print("\nSignificant Correlation Matrix (|correlation| > 0.3):")
# print(significant_matrix)


# Save filtered results to CSV for further exploration
# basic_stats.to_csv('filtered_basic_statistics.csv', index=True)
# significant_corr.to_csv('responder_6_significant_correlations.csv', index=True)
# significant_matrix.to_csv('filtered_correlation_matrix.csv', index=True)
# print("\nFiltered statistics and correlations saved to:")
# print("  - 'filtered_basic_statistics.csv'")
# print("  - 'responder_6_significant_correlations.csv'")
# print("  - 'filtered_correlation_matrix.csv'")



import polars as pl
import pandas as pd

file_path = '../input/jane-street-real-time-market-data-forecasting/train.parquet'
data = pl.scan_parquet(file_path)
sorted_data = data.sort(['date_id', 'time_id'])

batch_size = 500_000
correlation_threshold = 0.3
start_batch = 2
end_batch = 3

batch_responder_6_corr = []

for batch_idx in range(start_batch, end_batch + 1):
    start_idx = (batch_idx - 1) * batch_size
    end_idx = start_idx + batch_size
    batch_polars = sorted_data.slice(start_idx, end_idx - start_idx).collect()
    batch = batch_polars.to_pandas()
    correlations = batch.corr(method='pearson')
    if 'responder_6' in correlations.columns:
        responder_6_corr = correlations['responder_6'].drop('responder_6', errors='ignore')  # Exclude self-correlation
        significant_corr = responder_6_corr[responder_6_corr.abs() > correlation_threshold]
        batch_result = significant_corr.reset_index()
        batch_result.columns = ['Feature', 'Correlation']
        batch_result['Batch'] = f'batch_{batch_idx}'
        batch_responder_6_corr.append(batch_result)

combined_results = pd.concat(batch_responder_6_corr, ignore_index=True)

aggregated_results = combined_results.groupby('Feature').agg(
    Correlation_Mean=('Correlation', 'mean'),
    Correlation_Std=('Correlation', 'std'),
    Batch_Count=('Batch', 'count')
).reset_index()

aggregated_results = aggregated_results[aggregated_results['Correlation_Mean'].abs() > correlation_threshold]
aggregated_results = aggregated_results.sort_values(by='Correlation_Mean', key=abs, ascending=False)

print("\nImportant Correlations with responder_6 Across Batches:")
print(aggregated_results)



import polars as pl
import pandas as pd

file_path = '../input/jane-street-real-time-market-data-forecasting/train.parquet'
data = pl.scan_parquet(file_path)
sorted_data = data.sort(['date_id', 'time_id'])

batch_size = 500_000
correlation_threshold = 0.3

start_batch = 2
batch_idx = start_batch  # Start processing from batch 2

aggregated_results_list = []

while True:
    start_idx = (batch_idx - 1) * batch_size
    batch = sorted_data.slice(start_idx, batch_size).collect()
    
    if batch.height == 0:  # If no data is returned, end the loop
        print(f"No more data to process after batch {batch_idx - 1}.")
        break
    
    print(f"Processing batch {batch_idx} (starting row {start_idx})...")
    
    # Convert to pandas for correlation calculation
    batch_df = batch.to_pandas()
    
    if 'responder_6' in batch_df.columns:
        correlations = batch_df.corr()
        responder_6_corr = correlations['responder_6'].drop('responder_6', errors='ignore')
        significant_corr = responder_6_corr[abs(responder_6_corr) > correlation_threshold]

        # Store and print results per batch
        batch_results = []
        for feature, corr in significant_corr.items():
            result = {
                'Feature': feature,
                'Correlation': corr
            }
            batch_results.append(result)
            aggregated_results_list.append(result)
        
        if batch_results:
            print("\nImportant Correlations with responder_6 for Batch", batch_idx)
            batch_df_results = pd.DataFrame(batch_results)
            print(batch_df_results)
    
    batch_idx += 1
    print(f"Batch {batch_idx - 1} processed.\n")

# Aggregate results
results_df = pd.DataFrame(aggregated_results_list)
aggregated_df = results_df.groupby('Feature').agg(
    Correlation_Mean=('Correlation', 'mean'),
    Correlation_Std=('Correlation', 'std')).reset_index()

aggregated_df = aggregated_df[aggregated_df['Correlation_Mean'].abs() > correlation_threshold]
aggregated_df = aggregated_df.sort_values(by='Correlation_Mean', key=abs, ascending=False)

print("\nImportant Correlations with responder_6 Across All Batches:")
print(aggregated_df)



import polars as pl
import pandas as pd

file_path = '../input/jane-street-real-time-market-data-forecasting/train.parquet'
data = pl.scan_parquet(file_path)
sorted_data = data.sort(['date_id', 'time_id'])

batch_size = 100_000
start_batch = 2
finish_batch = 5  # Define the last batch to process
correlation_threshold = 0.3  # Set a threshold for important correlations

# Define the features and the target
features = [f"feature_{str(i).zfill(2)}" for i in range(79)]
target = 'responder_6'

# Initialize correlation results storage
symbol_correlation_results = {}

for batch_idx in range(start_batch, finish_batch + 1):
    start_idx = (batch_idx - 1) * batch_size
    batch = sorted_data.slice(start_idx, batch_size).collect()

    if batch.height == 0:
        print(f"No more data to process after batch {batch_idx - 1}.")
        break

    print(f"Processing batch {batch_idx} (starting row {start_idx}, {batch.height} rows)...")

    batch_df = batch.to_pandas()
    if 'symbol_id' in batch_df.columns and set(features).issubset(batch_df.columns) and target in batch_df.columns:
        grouped = batch_df.groupby('symbol_id')
        for symbol, group in grouped:
            correlation_matrix = group[features + [target]].corr()
            responder_correlations = correlation_matrix[target].drop(target)
            important_correlations = responder_correlations[(responder_correlations > correlation_threshold) | (responder_correlations < -correlation_threshold)]
            
            if not important_correlations.empty:
                print(f"Important Correlations with responder_6 for symbol {symbol} in Batch {batch_idx}:")
                print(important_correlations.sort_values(ascending=False))
                
            if symbol not in symbol_correlation_results:
                symbol_correlation_results[symbol] = []
            symbol_correlation_results[symbol].append(important_correlations)

    print(f"Batch {batch_idx} processed.\n")

    if batch_idx == finish_batch:
        print(f"Stopped processing as per finish_batch parameter at batch {finish_batch}.")

# Aggregate correlation results across batches for each symbol
print("\nAggregated Important Correlations with responder_6 across batches:")
for symbol, correlations in symbol_correlation_results.items():
    final_correlations = pd.concat(correlations, axis=1).mean(axis=1)
    final_important_correlations = final_correlations[(final_correlations > correlation_threshold) | (final_correlations < -correlation_threshold)]
    if not final_important_correlations.empty:
        print(f"\nSymbol {symbol}:")
        print(final_important_correlations.sort_values(ascending=False))



import polars as pl
import pandas as pd

file_path = '../input/jane-street-real-time-market-data-forecasting/train.parquet'
data = pl.scan_parquet(file_path)
sorted_data = data.sort(['date_id', 'time_id'])

batch_size = 100_000
start_batch = 2
finish_batch = 5  # Define the last batch to process
correlation_threshold = 0.3  # Set a threshold for important correlations

# Define the features and the target
features = [f"feature_{str(i).zfill(2)}" for i in range(79)]
target = 'responder_6'

# Initialize correlation results storage
stock_correlation_results = {}

for batch_idx in range(start_batch, finish_batch + 1):
    start_idx = (batch_idx - 1) * batch_size
    batch = sorted_data.slice(start_idx, batch_size).collect()

    if batch.height == 0:
        print(f"No more data to process after batch {batch_idx - 1}.")
        break

    print(f"Processing batch {batch_idx} (starting row {start_idx}, {batch.height} rows)...")

    batch_df = batch.to_pandas()
    if 'symbol_id' in batch_df.columns and set(features).issubset(batch_df.columns) and target in batch_df.columns:
        # Calculate returns for price-like features
        for feature in features:
            batch_df[f'{feature}_return'] = batch_df[feature].pct_change(fill_method=None)  # Handle NA values explicitly

        # Group by symbol_id and calculate correlations
        grouped = batch_df.groupby('symbol_id')
        for symbol, group in grouped:
            return_features = [f"{feature}_return" for feature in features if f"{feature}_return" in group.columns]
            correlation_matrix = group[return_features + [target]].corr()
            responder_correlations = correlation_matrix[target].drop(target)
            important_correlations = responder_correlations[abs(responder_correlations) > correlation_threshold]

            if not important_correlations.empty:
                print(f"Important Correlations with responder_6 for symbol {symbol} in Batch {batch_idx}:")
                print(important_correlations.sort_values(ascending=False))

            # Store results by symbol
            if symbol not in stock_correlation_results:
                stock_correlation_results[symbol] = []
            stock_correlation_results[symbol].append(important_correlations)

    print(f"Batch {batch_idx} processed.\n")

    if batch_idx == finish_batch:
        print(f"Stopped processing as per finish_batch parameter at batch {finish_batch}.")

# Aggregate correlation results across batches for each symbol
print("\nAggregated Important Correlations with responder_6 across batches:")
for symbol, correlations in stock_correlation_results.items():
    if correlations:
        aggregated_correlations = pd.concat(correlations, axis=1).mean(axis=1)
        final_important_correlations = aggregated_correlations[abs(aggregated_correlations) > correlation_threshold]
        print(f"\nSymbol {symbol}:")
        print(final_important_correlations.sort_values(ascending=False))



import polars as pl
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

# Load the data
file_path = '../input/jane-street-real-time-market-data-forecasting/train.parquet'
data = pl.scan_parquet(file_path)
sorted_data = data.sort(['date_id', 'time_id'])

# Set up parameters
batch_size = 100_000
start_batch = 2
end_batch = 5  # Define the last batch to process explicitly

# Define the features and the target
expected_features = [f"feature_{str(i).zfill(2)}" for i in range(79)] + ['symbol_id']
target = 'responder_6'

# Initialize the CatBoost model, specifying that symbol_id is a categorical feature
model = CatBoostRegressor(
    iterations=100, 
    learning_rate=0.1, 
    depth=6, 
    loss_function='RMSE', 
    cat_features=['symbol_id'],
    verbose=0
)

# Prepare validation data
validation_data = sorted_data.filter(pl.col('date_id') > 950)
sorted_data = sorted_data.filter(pl.col('date_id') <= 950)
validation_df = validation_data.collect().to_pandas()

model_performance = []

for batch_idx in range(start_batch, end_batch + 1):
    start_idx = (batch_idx - 1) * batch_size
    batch = sorted_data.slice(start_idx, batch_size).collect()

    if batch.height == 0:
        print(f"No more data to process after batch {batch_idx - 1}.")
        break

    print(f"Processing batch {batch_idx} (starting row {start_idx})...")
    
    batch_df = batch.to_pandas()
    if set(expected_features).issubset(batch_df.columns) and target in batch_df.columns:
        X_train = batch_df[expected_features]
        y_train = batch_df[target]

        # Update the model incrementally
        if batch_idx == start_batch:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, init_model=model)

        # Evaluate the model on the validation set
        y_pred = model.predict(validation_df[expected_features])
        mse = mean_squared_error(validation_df[target], y_pred)
        model_performance.append(mse)
        print(f"Batch {batch_idx} processed. MSE: {mse:.4f}")

    else:
        print(f"Missing necessary features in batch {batch_idx}. Skipping...")

# Final evaluation
y_pred_final = model.predict(validation_df[expected_features])
final_mse = mean_squared_error(validation_df[target], y_pred_final)
model_performance.append(final_mse)

print("Training complete. Model performance across batches:")
for i, mse in enumerate(model_performance, start=start_batch):
    print(f"Batch {i}: MSE = {mse:.4f}")
print(f"Final MSE after all batches: {final_mse:.4f}")


