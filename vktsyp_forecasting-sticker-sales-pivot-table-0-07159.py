import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from typing import List
from prophet import Prophet

import warnings
warnings.filterwarnings("ignore")


# Load the training and test datasets

train=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv',index_col=[0],parse_dates=['date'])
test=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv',index_col=[0],parse_dates=['date'])


train.head(20)


test.head(20)


# Creating Pivot Table from DataFrame
def create_pivot_table(data):
 
    data_copy = data.copy()
    
    # Create a new column combining 'country', 'store', and 'product'.
    data_copy['country_store_product'] = data_copy['country'] + '-' + data_copy['store'] + '-' + data_copy['product']

    # Check if the 'num_sold' column exists and create a pivot table.
    if 'num_sold' in data_copy.columns:
        pivot_table = data_copy.pivot_table(
            index='date',
            columns='country_store_product',
            values='num_sold',
            aggfunc='sum'
        )
    else:
        pivot_table = data_copy.pivot_table(
            index='date',
            columns='country_store_product',
            aggfunc='size'
        )

    # Reorder the columns based on the original data order.
    original_order = data_copy[['country_store_product']].drop_duplicates()['country_store_product']
    pivot_table = pivot_table[original_order]

    return pivot_table

# Create pivot table
train20 = create_pivot_table(train)
test20 = create_pivot_table(test)
display(train20)


def plot_pivot_data(df_pivot, max_cols=5, filter_keywords=None, highlight_plots=None):   
    df_pivot_filled = df_pivot.fillna(0)
    df_pivot_copy = df_pivot_filled.copy()

    # Filter columns by the provided keywords
    if filter_keywords:
        filtered_columns = [
            col for col in df_pivot_copy.columns if any(keyword in col for keyword in filter_keywords)
        ]
        df_pivot_copy = df_pivot_copy[filtered_columns]

    # Calculate layout for subplots
    num_columns = len(df_pivot_copy.columns)
    num_rows = (num_columns // max_cols) + (num_columns % max_cols > 0)

    # Set the size of the figure
    plt.figure(figsize=(12, num_rows * 1.9))

    # Plot each column
    for i, column in enumerate(df_pivot_copy.columns):
        plt.subplot(num_rows, max_cols, i + 1)
        
        # Plot zero values as small red circles
        zero_indices = df_pivot_copy[column] == 0
        plt.scatter(df_pivot_copy.index[zero_indices], df_pivot_copy[column][zero_indices], color='red', s=5, label='Zero', zorder=5)

        # Plot the data
        plt.plot(df_pivot_copy.index, df_pivot_copy[column], label=column, alpha=0.5)

        # Split the title into 3 parts
        title_parts = column.split('-')
        title = '\n'.join(title_parts)
        plt.title(title, fontsize=9)

        # Set axis labels and formatting
        plt.xticks(rotation=30, fontsize=8)
        plt.yticks(fontsize=7)

        # Add a small blue number in the top-right corner of the plot
        plt.text(0.95, 0.95, f'{i + 1}', color='brown', ha='right', va='top', fontsize=9, transform=plt.gca().transAxes)

        # Highlight the plot if it is in the highlight_plots list
        if highlight_plots and (i + 1) in highlight_plots:
            for spine in plt.gca().spines.values():
                spine.set_edgecolor('blue')
                spine.set_linewidth(3)

        # If there are missing values, display "Missing values" on the bottom left
        if zero_indices.any():  # If there are zeros (missing values)
            missing_count = zero_indices.sum()  # Count the number of missing values
            plt.text(0.05, 0.12, f'Missing values: {missing_count}', color='red', ha='left', va='bottom', fontsize=11, transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()

# Specify the plot numbers to be highlighted with a blue border
plot_pivot_data(train20, highlight_plots=[1, 4, 6, 11, 46, 49, 50, 51, 56])



def missing_data_table(df):
    # Number of rows in the data (data length)
    data_length = len(df)

    # Calculate the number of 0.0 values
    missing_data = (df == 0.0).sum()

    # Extract only columns that have one or more 0.0 values
    missing_data = missing_data[missing_data > 0]

    # Determine the type of missing data
    missing_type = ['Completely Missing' if count == data_length else 'Partially Missing' for count in missing_data.values]

    # Convert the results into a DataFrame
    missing_data_df = pd.DataFrame({
        'Columns with Missing Data': missing_data.index, 
        'Missing Values': missing_data.values,
        'Missing Type': missing_type
    })

    # Sort by the number of missing values in ascending order
    missing_data_df = missing_data_df.sort_values(by='Missing Values')

    # Reset the index to clean up the ID
    missing_data_df = missing_data_df.reset_index(drop=True)

    return missing_data_df

# Test DataFrame for display
missing_data_df = missing_data_table(train20)
display(missing_data_df)


def plot_spectrum_without_dc(df, max_cols=5, filter_keywords=None):
    df_interpolated = df.interpolate(method='linear', axis=0)
    df_copy = df_interpolated.copy()

    # Filter columns by the provided keywords
    if filter_keywords:
        filtered_columns = [col for col in df_copy.columns if any(keyword in col for keyword in filter_keywords)]
        df_copy = df_copy[filtered_columns]

    # Calculate layout for subplots
    num_columns = len(df_copy.columns)
    num_rows = (num_columns // max_cols) + (num_columns % max_cols > 0)

    # Set the size of the figure
    plt.figure(figsize=(12, num_rows * 1.7))

    for i, column in enumerate(df_copy.columns):
        plt.subplot(num_rows, max_cols, i + 1)

        # Remove DC component
        signal = df_copy[column].values
        signal_centered = signal - np.mean(signal)

        # Fourier Transform
        fft_values = np.abs(fft(signal_centered))
        freqs = np.fft.fftfreq(len(signal_centered))

        # Plot up to the folding frequency
        half_n = len(signal_centered) // 2
        plt.plot(freqs[:half_n], fft_values[:half_n], label=column, alpha=0.5)

        title_parts = column.split('-')
        title = '\n'.join(title_parts)
        plt.title(title, fontsize=9)

        plt.ylabel('Amplitude', fontsize=8)
        plt.xticks(rotation=30, fontsize=8)
        plt.yticks(fontsize=6)

        plt.text(0.95, 0.95, f'{i + 1}', color='darkgreen', ha='right', va='top', fontsize=8, transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()

# Plot the data
plot_spectrum_without_dc(train20)


def plot_spectrum_overlay_normalized(df, filter_keywords=None):
    df_interpolated = df.interpolate(method='linear', axis=0)
    df_copy = df_interpolated.copy()

    if filter_keywords:
        filtered_columns = [col for col in df_copy.columns if any(keyword in col for keyword in filter_keywords)]
        df_copy = df_copy[filtered_columns]

    # Standard normalization
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_copy), index=df_copy.index, columns=df_copy.columns)

    plt.figure(figsize=(10, 8))  # Adjust the size of the entire figure

    # First graph: display full range on the x-axis
    plt.subplot(2, 1, 1)
    for column in df_normalized.columns:
        # Remove DC component
        signal = df_normalized[column].values
        signal_centered = signal - np.mean(signal)

        # Fourier Transform
        fft_values = np.abs(fft(signal_centered))
        freqs = np.fft.fftfreq(len(signal_centered))

        # Plot up to the folding frequency
        half_n = len(signal_centered) // 2
        plt.plot(freqs[:half_n], fft_values[:half_n], alpha=0.7, linewidth=1.5, label=column)

    fx = 1 / 7
    plt.axvline(x=fx, color='#707070', linestyle='--', linewidth=1, alpha=0.5) 
    plt.text(fx + 0.007, max(fft_values[:half_n]) / 3, '1 / 7 = 0.142 (Weekly)', color='red', fontsize=12, rotation=90)
    plt.axvline(x=fx * 2, color='#707070', linestyle='--', linewidth=1, alpha=0.5) 
    plt.text(fx * 2 + 0.007, max(fft_values[:half_n]) * 1 / 6, '2/7=2nd Harmonic of Weekly', color='red', fontsize=12, rotation=90)
    plt.axvline(x=fx * 3, color='#707070', linestyle='--', linewidth=1, alpha=0.5) 
    plt.text(fx * 3 + 0.007, max(fft_values[:half_n]) * 1 / 6, '3/7=3rd Harmonic of Weekly', color='red', fontsize=12, rotation=90)

    fx = 1 / 30
    plt.axvline(x=fx, color='#707070', linestyle='--', linewidth=1, alpha=0.5)  
    plt.text(fx + 0.007, max(fft_values[:half_n]) / 3, '1 / 30 = 0.033 (Monthly)', color='red', fontsize=12, rotation=90)

    plt.title('Overlayed Spectrum Analysis (Normalized) - Full Range')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')

    # Second graph: zoom in to 0.01 on the x-axis
    plt.subplot(2, 1, 2)
    for column in df_normalized.columns:
        # Remove DC component
        signal = df_normalized[column].values
        signal_centered = signal - np.mean(signal)

        # Fourier Transform
        fft_values = np.abs(fft(signal_centered))
        freqs = np.fft.fftfreq(len(signal_centered))

        # Plot up to the folding frequency
        half_n = len(signal_centered) // 2
        plt.plot(freqs[:half_n], fft_values[:half_n], alpha=0.7, linewidth=2, label=column) 

    fx = 1 / 365
    plt.axvline(x=fx, color='#707070', linestyle='--', linewidth=1, alpha=0.5) 
    plt.text(fx + 0.0003, max(fft_values[:half_n]) / 2.5, '1 / 365 : Annual cycle', color='red', fontsize=12, rotation=90)

    fx = 1 / (365*2)
    plt.axvline(x=fx, color='#707070', linestyle='--', linewidth=1, alpha=0.5) 
    plt.text(fx + 0.0002, max(fft_values[:half_n]) / 2.6, '1/(365*2): two-year cycle', color='red', fontsize=12, rotation=90)

    plt.title('Overlayed Spectrum Analysis (Normalized) - Zoomed In')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.xlim([0, 0.01])

    plt.tight_layout()
    plt.show()

plot_spectrum_overlay_normalized(train20)



columns_to_drop = [col for col in train20.columns if (train20[col] == 0.0).all()]
train22 = train20.drop(columns=columns_to_drop)

# Display columns that contain 0.0
missing_columns_zero = train22.columns[(train22 == 0.0).any()].tolist()
display(missing_columns_zero)

# Create a new dataframe train23 by dropping rows that contain 0.0
train23 = train22[(train22 != 0.0).all(axis=1)]
target_columns = missing_columns_zero

# Plot correlation for each column
for target_column in target_columns:
    # Extract numeric data
    numeric_data = train23.select_dtypes(include=['number']).copy()
    
    # Calculate correlation matrix
    correlation_matrix = numeric_data.corr()
    
    # Extract correlation with the target column and take the absolute value
    correlation_with_target = correlation_matrix[target_column].drop(target_column).abs()
    
    # Exclude items included in target_columns
    correlation_with_target_filtered = correlation_with_target.drop(target_columns, errors='ignore')
    
    # Sort correlation coefficients in descending order
    correlation_with_target_sorted = correlation_with_target_filtered.sort_values(ascending=False)

    # Extract top 15 correlations
    top_nn_correlation = correlation_with_target_sorted.head(15)
    
    # Plot horizontal bar graph
    plt.figure(figsize=(9, 3.5))
    ax = top_nn_correlation.plot(kind='barh', color='steelblue', edgecolor='black')
    
    # Display correlation coefficients on each bar
    for index, value in enumerate(top_nn_correlation):
        if not (pd.isna(value) or value == float('inf') or value == float('-inf')):
            plt.text(value + 0.002, index, f"{value:.3f}", va='center', fontsize=9)
    
    # Display larger values on top
    ax.invert_yaxis()
    
    # Set font size of y-axis column names to small
    ax.tick_params(axis='y', labelsize=9)
    
    # Configure graph display
    plt.xlim(0, 1)  # Set maximum value of x-axis to 1
    #plt.title(f"Top 10 Correlations with {target_column} (Absolute Values)", fontsize=10)
    plt.title(f" {target_column} ", fontsize=10)
    plt.xlabel("Correlation Coefficient (Absolute)", fontsize=9)
    plt.ylabel("Features", fontsize=9)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout(pad=2)
    plt.show()



# Estimating Partially Missing Data

numeric_data = train23.select_dtypes(include=['number']).copy()
models = {}

# Copy the dataframe to hold the filled data
train20_filled = train20.copy()

# For each target column
for target_column in target_columns:
    # Calculate the correlation matrix
    correlation_matrix = numeric_data.corr()
    
    # Extract correlation with the target column and take the absolute value
    correlation_with_target = correlation_matrix[target_column].drop(target_column).abs()
    
    # Exclude items included in target_columns
    correlation_with_target_filtered = correlation_with_target.drop(target_columns, errors='ignore')
    
    # Sort correlation coefficients in descending order and get the top 10 columns
    top_n_correlation = correlation_with_target_filtered.sort_values(ascending=False).head(10)

    # Prepare the data
    X = numeric_data[top_n_correlation.index]
    y = numeric_data[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Create the model
    model = ElasticNet(alpha=0.1, l1_ratio=0.1)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the model
    models[target_column] = model
    
    # Predict using the test data
    y_test_pred = model.predict(X_test)
    
    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_test_pred)
    print(f"Mean Squared Error for {target_column}: {mse}")
    
    # Predict the missing values (0.0)
    X_fill = train20[top_n_correlation.index]

    # Ensure the order of features is consistent
    X_fill = X_fill.loc[:, top_n_correlation.index]
    
    mask = train20[target_column] == 0.0
    
    y_pred_fill = model.predict(X_fill[mask])
    
    # Fill with predicted values
    train20_filled.loc[mask, target_column] = y_pred_fill

# Obtain the filled dataset
train27 = train20_filled
plot_pivot_data(train27, highlight_plots=[4, 6, 11, 49, 50, 51, 56])



# List of completely missing data
missing_data_list = ['Canada-Discount Stickers-Holographic Goose', 'Kenya-Discount Stickers-Holographic Goose']

for missing_data_df2 in missing_data_list:
    
    display(missing_data_df2)
 
    country_store_prefix = '-'.join(missing_data_df2.split('-')[:-1]) + '-'
    excluded_product_name = missing_data_df2.split('-')[-1]
    correlation_df = train27  # Replace with actual data

    # Function to extract product names
    def extract_product_names(df: pd.DataFrame, excluded_product_name: str) -> List[str]:
        product_names = df.columns.str.split('-').str[-1].unique()
        return product_names[product_names != excluded_product_name]

    # Extract product names
    product_names = extract_product_names(correlation_df, excluded_product_name)

    # DataFrame to store correlation results for all products
    all_correlations = pd.DataFrame()

    # Calculate correlation for each product
    for product in product_names:
        filtered_df = correlation_df[[col for col in correlation_df.columns if col.endswith(product)]]
        correlation_matrix = filtered_df.corr()
        target_column = f"{country_store_prefix}{product}"
        target_correlation = correlation_matrix[target_column].drop(target_column)
        sorted_correlation = target_correlation.sort_values(ascending=False)
        sorted_correlation.name = product
        sorted_correlation.index = sorted_correlation.index.str.rsplit('-', n=1).str[0]
        all_correlations = pd.concat([all_correlations, sorted_correlation], axis=1)

    # Calculate average correlation and add as a new column
    all_correlations['Average Correlation'] = all_correlations.mean(axis=1)
    display(all_correlations)

    # Get the top 5 columns
    top_5_columns = all_correlations.nlargest(5, 'Average Correlation').index.tolist()

    # Lists to store features and targets
    X_list = []
    y_list = []
    product_labels = []

    # Build data for each product
    for product in product_names:
        target_column = f"{country_store_prefix}{product}"
        selected_columns = [target_column] + [f"{store}-{product}" for store in top_5_columns]
        selected_data = correlation_df[selected_columns]
        X = selected_data.drop(target_column, axis=1)
        y = selected_data[target_column]
        X.columns = X.columns.str.rsplit('-', n=1).str[0]
        X_list.append(X)
        y_list.append(y)
        product_labels.extend([product] * len(y))

    # Concatenate all data vertically
    X_all = pd.concat(X_list, axis=0)
    y_all = pd.concat(y_list, axis=0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=42)

    # Create and train the model
    model = ElasticNet(alpha=0.1, l1_ratio=0.1)
    model.fit(X_train, y_train)

    # Predict using the test data and calculate the error
    y_test_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    print(f"Mean Squared Error for {missing_data_df2}: {mse}")

    # Predict with new data
    feature_columns = [f"{store}-{excluded_product_name}" for store in top_5_columns]
    new_data = correlation_df[feature_columns]
    new_data.columns = new_data.columns.str.rsplit('-', n=1).str[0]
    new_prediction = model.predict(new_data)

    # Add the prediction results to train27
    train27[f"{country_store_prefix}{excluded_product_name}"] = new_prediction

# Display results
plot_pivot_data(train27, highlight_plots=[1, 46])


# Spectrum Characteristics of Corrected Sales Data

plot_spectrum_overlay_normalized(train27)


#from fbprophet import Prophet

def predict_and_update(train, test, target_column):
    
    # Create a DataFrame for Prophet from the train dataset
    df = train[[target_column]].reset_index() 
    df['ds'] = df['date'] 
    df['y'] = df[target_column] 
    df = df[['ds', 'y']] 

    # Create and configure the Prophet model
    model = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=True, 
        daily_seasonality=False, 
    )
    
    # Add 2-year seasonality
    model.add_seasonality(name='biennial', period=365.25*2, fourier_order=1)
    model.add_seasonality(name='semiannual', period=365.25/2, fourier_order=2)
    model.add_seasonality(name='four_month', period=365.25/3, fourier_order=2)    
    model.add_seasonality(name='quarterly', period=365.25/4, fourier_order=2)

    # Train the model
    model.fit(df)

    # Prepare the test dataset for prediction
    df2 = test[[target_column]].reset_index() 
    df2['ds'] = df2['date'] 
    df2 = df2[['ds']] 

    # Perform prediction
    future = df2.copy()
    forecast = model.predict(future)

    # Write the prediction results back to the test dataset
    test[target_column] = test.index.map(
        lambda x: forecast.loc[forecast['ds'] == x, 'yhat'].values[0]
        if x in forecast['ds'].values else None
    )
    return test

# Dynamically get target columns to predict
target_columns = [col for col in test20.columns if col != 'date'] 

# Perform prediction and update for each column
test30 = test20.copy() 
for column in target_columns:
    test50 = predict_and_update(train27, test30, column)

# Round the numbers in test50 to the nearest integer
test50 = test50.round()

# Display the results
display(test50)


# Displaying Predicted Sales Values

def plot_pivot_data_overlay(df_pivot, df_secondary=None, alpha=0.7, max_cols=5, filter_keywords=None):

    df_pivot_filled = df_pivot.fillna(0)
    df_pivot_copy = df_pivot_filled.copy()

    # If secondary data is provided, replace missing values with zero
    if df_secondary is not None:
        df_secondary_filled = df_secondary.fillna(0)
        df_secondary_copy = df_secondary_filled.copy()

    # Filter columns based on keywords
    if filter_keywords:
        filtered_columns = [
            col for col in df_pivot_copy.columns if any(keyword in col for keyword in filter_keywords)
        ]
        df_pivot_copy = df_pivot_copy[filtered_columns]
        if df_secondary is not None:
            df_secondary_copy = df_secondary_copy[filtered_columns]

    # Calculate layout
    num_columns = len(df_pivot_copy.columns)
    num_rows = (num_columns // max_cols) + (num_columns % max_cols > 0)

    # Set the figure size
    plt.figure(figsize=(12, num_rows * 1.7))

    # Plot each column
    for i, column in enumerate(df_pivot_copy.columns):
        plt.subplot(num_rows, max_cols, i + 1)

        # Plot the main data
        plt.plot(df_pivot_copy.index, df_pivot_copy[column], label=f'Train {column}', alpha=alpha, color='blue')

        # If secondary data is available, plot it
        if df_secondary is not None:
            plt.plot(df_secondary_copy.index, df_secondary_copy[column], label=f'Test {column}', alpha=alpha, color='orange')

        title_parts = column.split('-')
        title = '\n'.join(title_parts)
        plt.title(title, fontsize=7)
        plt.ylabel('num_sold', fontsize=7)
        plt.xticks(rotation=35, fontsize=6)
        plt.yticks(fontsize=6)

    plt.tight_layout()
    plt.show()

plot_pivot_data_overlay(train27, df_secondary=test50)



# Converting Pivot Table to Original Format

def convert_pivot_to_original(pivot_table, store_order):

    # Reset the pivot table and convert to long format
    data_unpivoted = pivot_table.reset_index().melt(id_vars='date', var_name='country_store_product', value_name='num_sold')
    
    # Split the 'country_store_product' column into 'country', 'store', and 'product' columns
    country_store_product_split = data_unpivoted['country_store_product'].str.split('-', expand=True)
    data_unpivoted['country'] = country_store_product_split[0]
    data_unpivoted['store'] = country_store_product_split[1]
    data_unpivoted['product'] = country_store_product_split[2]

    # Rearrange the necessary columns to return to the original format
    original_format = data_unpivoted[['date', 'country', 'store', 'product', 'num_sold']]

    # Convert the 'store' column to a categorical type and specify the order
    original_format['store'] = pd.Categorical(original_format['store'], categories=store_order, ordered=True)

    # Sort by 'date', 'country', 'store', and 'product'
    original_format = original_format.sort_values(by=['date', 'country', 'store', 'product']).reset_index(drop=True)

    # Create the original index
    original_format['id'] = original_format.index
    
    return original_format

# Define the store order
store_order = ["Discount Stickers", "Stickers for Less", "Premium Sticker Mart"]

# Convert 'test50' back to the original format
test70 = convert_pivot_to_original(test50, store_order)


# Create the submission DataFrame
submission = pd.DataFrame({'id': test.index, 'num_sold': test70['num_sold']})

# Save the submission DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)

submission

