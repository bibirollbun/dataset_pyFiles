from tabulate import tabulate
import pandas as pd

def load_and_merge_data(train_file_path, test_file_path, target_column='num_sold', index_column='id', date_column='date'):
    """
    Loads train and test datasets, processes them, and merges into a single DataFrame.
    
    Args:
        train_file_path (str): Path to the training dataset.
        test_file_path (str): Path to the test dataset.
        target_column (str): Name of the target column.
        index_column (str): Column to use as the index.
        date_column (str): Column to parse as dates.
    
    Returns:
        pd.DataFrame: A combined DataFrame of train and test data.
    """
    # Load train and test datasets
    train_data = pd.read_csv(train_file_path, index_col=index_column, parse_dates=[date_column])
    test_data = pd.read_csv(test_file_path, index_col=index_column, parse_dates=[date_column])

    train_data.index.name = 'id'
    test_data.index.name = 'id'
    
    # Add an indicator column for test and train
    train_data['is_test'] = False
    test_data['is_test'] = True
    
    # Add a placeholder for the target column in test_data with the correct dtype
    # test_data[target_column] = None
    # test_data[target_column] = test_data[target_column].astype('float64')
    
    # Merge train and test datasets
    all_data = pd.concat([train_data, test_data])
    all_data.index.name = 'id'
    
    return train_data, test_data, all_data

# Call the function and process the data
train_file_path = '/kaggle/input/playground-series-s5e1/train.csv'
test_file_path = '/kaggle/input/playground-series-s5e1/test.csv'
train_data, test_data, all_raw_data = load_and_merge_data(
    train_file_path=train_file_path,
    test_file_path=test_file_path
)

# Verify the changes
print(train_data.info())
print(test_data.info())
print("\nAll Data Info:")
print(all_raw_data.info())


from ydata_profiling import ProfileReport

train_report = ProfileReport(
    train_data,
    tsmode=True,
    sortby="date",
    title="Train DS",
)


test_report = ProfileReport(
    test_data,
    tsmode=True,
    sortby="date",
    title="Test DS",
)

comparison_report = train_report.compare(test_report)

train_report.to_file("train_report.html")
test_report.to_file("test_report.html")
comparison_report.to_file("compare_train_test_report.html")

del train_report, test_report, comparison_report


import matplotlib.pyplot as plt
import seaborn as sns

def plot_data_distribution_and_trend(data, target_column='num_sold', date_column='date'):
    """
    Plots the distribution of the target column, its trend over time,
    and its relationship with categorical variables.
    
    Args:
        data (pd.DataFrame): The input dataset.
        target_column (str): The column to plot the distribution for.
        date_column (str): The date column to aggregate sales over time.
    """
    # Plot the distribution of the target column
    plt.figure(figsize=(12, 6))
    sns.histplot(data[target_column], bins=50, kde=True, color='blue')
    plt.title(f'Distribution of {target_column}', fontsize=16)
    plt.xlabel(target_column, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.show()

    # Aggregate sales over time
    sales_over_time = data.groupby(date_column)[target_column].sum().reset_index()

    # Plot sales trend over time
    plt.figure(figsize=(15, 6))
    plt.plot(sales_over_time[date_column], sales_over_time[target_column], color='green')
    plt.title(f'{target_column} Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel(f'Total {target_column}', fontsize=14)
    plt.grid()
    plt.show()

    # Relationship between num_sold and country
    plt.figure(figsize=(12, 6))
    sns.barplot(x='country', y=target_column, data=data, errorbar=None, estimator='mean', palette='viridis')
    plt.title(f'Average {target_column} by Country', fontsize=16)
    plt.xlabel('Country', fontsize=14)
    plt.ylabel(f'Average {target_column}', fontsize=14)
    plt.xticks(rotation=45)
    plt.show()

    # Relationship between num_sold and store
    plt.figure(figsize=(12, 6))
    sns.barplot(x='store', y=target_column, data=data, errorbar=None, estimator='mean', palette='cool')
    plt.title(f'Average {target_column} by Store', fontsize=16)
    plt.xlabel('Store', fontsize=14)
    plt.ylabel(f'Average {target_column}', fontsize=14)
    plt.xticks(rotation=45)
    plt.show()

    # Relationship between num_sold and product
    plt.figure(figsize=(12, 6))
    sns.barplot(x='product', y=target_column, data=data, errorbar=None, estimator='mean', palette='plasma')
    plt.title(f'Average {target_column} by Product', fontsize=16)
    plt.xlabel('Product', fontsize=14)
    plt.ylabel(f'Average {target_column}', fontsize=14)
    plt.xticks(rotation=45)
    plt.show()

# Call the function to visualize the data
plot_data_distribution_and_trend(train_data, target_column='num_sold', date_column='date')


def feature_engineering(data, target_column='num_sold', date_column='date'):
    """
    Performs feature engineering on the dataset:
    - Validates required columns.
    - Extracts date features.
    - Creates interaction features.
    - Adds lagged features and rolling mean features at multiple aggregation levels.
    - Handles NaN cases for lagged features with special indicators and imputed values.
    
    Args:
        data (pd.DataFrame): The input dataset.
        target_column (str): The target column to compute lag/rolling statistics for.
        date_column (str): The date column for sorting and feature extraction.
        
    Returns:
        pd.DataFrame: The dataset with engineered features.
    """
    # Copy the dataset to avoid modifying the original
    data = data.copy()

    # For simple backroll to original index
    data = data.reset_index()
    
    # Required columns for feature engineering
    required_columns = [target_column, date_column, 'country', 'store', 'product']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    # Check if all required columns are present
    if missing_columns:
        raise ValueError(f"Missing required columns for feature engineering: {missing_columns}")
    
    # Extract date features
    data['year'] = pd.to_datetime(data[date_column]).dt.year
    data['month'] = pd.to_datetime(data[date_column]).dt.month
    data['day'] = pd.to_datetime(data[date_column]).dt.day
    data['day_of_week'] = pd.to_datetime(data[date_column]).dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

    # Interaction features
    data['country_store'] = data['country'] + '_' + data['store']
    data['country_product'] = data['country'] + '_' + data['product']
    data['store_product'] = data['store'] + '_' + data['product']
    data['country_store_product'] = data['country'] + '_' + data['store'] + '_' + data['product']

    # Sort data for lagged calculations
    data = data.sort_values(by=['country', 'store', 'product', date_column])
    data[date_column] = pd.to_datetime(data[date_column])
    
    # Levels for aggregated lagged and rolling features
    levels = [
        ['country', 'product'],
        ['country', 'store'],
        ['country']
    ]
    
    # Create lagged and rolling mean features for each level
    for level in levels:
        level_name = "_".join(level)
        
        # Aggregate target value at the specified level
        aggregated = data.groupby(level + [date_column])[target_column].sum().reset_index()
        
        # Compute lagged features
        aggregated[f'lag_1_{level_name}'] = aggregated.groupby(level)[target_column].shift(1)
        aggregated[f'lag_7_{level_name}'] = aggregated.groupby(level)[target_column].shift(7)
        
        # Compute rolling mean features
        aggregated[f'rolling_mean_7_{level_name}'] = (
            aggregated.groupby(level)[target_column]
            .shift(1)
            .rolling(window=7, min_periods=1)
            .mean()
        )
        
        # Merge the aggregated features back into the main dataset
        data = data.merge(
            aggregated[[date_column] + level + [f'lag_1_{level_name}', f'lag_7_{level_name}', f'rolling_mean_7_{level_name}']],
            on=[date_column] + level,
            how='left'
        )
    
    # Create lag features at the individual level
    data['lag_1'] = data.groupby(['country', 'store', 'product'])[target_column].shift(1)
    data['lag_7'] = data.groupby(['country', 'store', 'product'])[target_column].shift(7)

    # Handle NaN cases for lagged features
    lag_features = ['lag_1', 'lag_7']
    for level in levels:
        level_name = "_".join(level)
        lag_features += [f'lag_1_{level_name}', f'lag_7_{level_name}']
    
    for feature in lag_features:
        # Create an indicator for NaN values
        data[f'{feature}_is_nan'] = data[feature].isnull().astype(int)
        # Impute missing values with the group mean
        data[f'{feature}_imputed'] = data.groupby(['country', 'store', 'product'])[feature].transform(
            lambda x: x.fillna(x.mean())
        )

    # Create rolling mean features at the individual level
    data['rolling_mean_7'] = (
        data.groupby(['country', 'store', 'product'])[target_column]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    # Set index back to id column
    data = data.set_index('id', verify_integrity=True).sort_index()
    return data

# Call the function for feature engineering
try:
    all_data = feature_engineering(all_raw_data, target_column='num_sold', date_column='date')
    all_data.info()
    train_data, test_data = all_data[~all_data['is_test']],all_data[all_data['is_test']]
except ValueError as e:
    print(e)


# all_aug_report = ProfileReport(
#     all_data,
#     tsmode=True,
#     sortby="date",
#     title="All Aug DS",
# )

# all_aug_report.to_file("all_aug_report.html")
# del all_aug_report


def check_data_summary(data, target_column='num_sold', correlation_threshold=0.9):
    """
    Checks the data summary, including missing values, feature-target correlation, 
    and highly correlated features.
    
    Args:
        data (pd.DataFrame): The dataset to analyze.
        target_column (str): The target column for correlation analysis.
        correlation_threshold (float): The threshold for identifying high correlations.
        
    Returns:
        None: Prints the summary using tabulate.
    """
    # Select only numerical columns for correlation
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

    # Compute correlation matrix for numeric columns
    correlation_matrix = data[numeric_columns].corr()

    # Prepare missing values summary with percentage
    total_rows = len(data)
    missing_summary_table = [
        {
            "Column": col,
            "Missing Values": val,
            "Percentage (%)": f"{(val / total_rows) * 100:.2f}%"
        }
        for col, val in data.isnull().sum().sort_values(ascending=False).items()
    ]

    # Prepare feature-target correlation (exclude target itself)
    correlation_table = [
        {"Feature": feature, "Correlation": round(corr, 4)}
        for feature, corr in correlation_matrix[target_column]
        .drop(labels=[target_column])  # Exclude target
        .sort_values(ascending=False)
        .items()
    ]

    # Prepare highly correlated features (exclude correlations involving the target)
    high_corr_features_table = [
        {
            "Feature Pair": f"{pair[0]} & {pair[1]}",
            "Correlation": round(correlation_matrix.loc[pair[0], pair[1]], 4)
        }
        for pair in correlation_matrix[
            (correlation_matrix.abs() > correlation_threshold) & (correlation_matrix != 1)
        ].stack().index.tolist()
        if target_column not in pair  # Exclude pairs involving the target
    ]

    # Display results using tabulate
    print("Missing Values Summary:")
    print(tabulate(missing_summary_table, headers="keys", tablefmt="pretty"))

    print("\nFeature-Target Correlation (Excluding Target):")
    print(tabulate(correlation_table, headers="keys", tablefmt="pretty"))

    print("\nHighly Correlated Features (Excluding Target) (Threshold > {:.1f}):".format(correlation_threshold))
    print(tabulate(high_corr_features_table, headers="keys", tablefmt="pretty"))

# Call the function
check_data_summary(train_data, target_column='num_sold', correlation_threshold=0.9)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def train_simple_model_with_features(data, target_column='num_sold'):
    """
    Trains a simple linear regression model using the provided dataset and feature-engineered columns.
    
    Args:
        data (pd.DataFrame): The input dataset with feature-engineered columns.
        target_column (str): The target column to predict.
        
    Returns:
        dict: A dictionary containing the trained model, MAPE, and MSE scores.
    """
    # Step 1: Handle missing target values
    data = data[data[target_column].notna()]  # Remove rows where the target is NaN

    # Step 2: Separate features and target
    features = data.drop(columns=[target_column, 'date'])
    target = data[target_column]

    # Step 3: Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Step 4: Define preprocessing pipeline for categorical and numerical columns
    categorical_features = [
        'country', 'store', 'product', 'country_store', 
        'country_product', 'store_product', 'country_store_product'
    ]
    numeric_features = list(
        features.drop(columns=categorical_features).columns
    )

    # Add imputed lagged features and rolling means to numeric features
    lagged_features = [
        col for col in numeric_features if "lag_" in col or "rolling_mean" in col
    ]
    numeric_features = lagged_features + [
        col for col in numeric_features if col not in lagged_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='constant', fill_value=0), numeric_features),  # Handle missing numeric
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Encode categorical
        ]
    )

    # Step 5: Create the pipeline for Linear Regression
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Step 6: Train the model
    pipeline.fit(X_train, y_train)

    # Step 7: Make predictions
    y_pred = pipeline.predict(X_val)

    # Step 8: Evaluate the model
    mape = mean_absolute_percentage_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)

    return {
        "Model": pipeline,
        "MAPE": mape,
        "MSE": mse
    }

# Call the function and display results
simple_model_results = train_simple_model_with_features(train_data)
print(f"MAPE: {simple_model_results['MAPE']:.4f}")
print(f"MSE: {simple_model_results['MSE']:.2f}")


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.impute import SimpleImputer

def train_xgboost_model_with_features(data, target_column='num_sold'):
    """
    Trains an XGBoost regression model using the provided dataset and feature-engineered columns.
    
    Args:
        data (pd.DataFrame): The input dataset with feature-engineered columns.
        target_column (str): The target column to predict.
        
    Returns:
        dict: A dictionary containing the trained model, MAPE, and MSE scores.
    """
    # Step 1: Handle missing target values
    data = data[data[target_column].notna()]  # Remove rows where the target is NaN

    # Step 2: Separate features and target
    features = data.drop(columns=[target_column, 'date'])
    target = data[target_column]

    # Step 3: Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Step 4: Define preprocessing pipeline for categorical and numerical columns
    categorical_features = [
        'country', 'store', 'product', 'country_store', 
        'country_product', 'store_product', 'country_store_product'
    ]
    numeric_features = list(
        features.drop(columns=categorical_features).columns
    )

    # Add lagged features and rolling means to numeric features
    lagged_features = [
        col for col in numeric_features if "lag_" in col or "rolling_mean" in col
    ]
    numeric_features = lagged_features + [
        col for col in numeric_features if col not in lagged_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='constant', fill_value=0), numeric_features),  # Handle missing numeric
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Encode categorical
        ]
    )

    # Step 5: Define the XGBoost model pipeline
    xgb_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        ))
    ])

    # Step 6: Train the model
    xgb_model.fit(X_train, y_train)

    # Step 7: Make predictions
    y_pred = xgb_model.predict(X_val)

    # Step 8: Evaluate the model
    mape = mean_absolute_percentage_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)

    return {
        "Model": xgb_model,
        "MAPE": mape,
        "MSE": mse
    }

# Call the function and display results
xgboost_results = train_xgboost_model_with_features(train_data)
print(f"MAPE: {xgboost_results['MAPE']:.4f}")
print(f"MSE: {xgboost_results['MSE']:.2f}")


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_absolute_percentage_error, mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np

def train_xgboost_model_with_optimization(data, target_column='num_sold'):
    """
    Trains an optimized XGBoost regression model with cross-validation using the provided dataset.
    
    Args:
        data (pd.DataFrame): The input dataset with feature-engineered columns.
        target_column (str): The target column to predict.
        
    Returns:
        dict: A dictionary containing the trained model, best hyperparameters, MAPE, and MSE scores.
    """
    # Step 1: Handle missing target values
    data = data[data[target_column].notna()]  # Remove rows where the target is NaN

    # Step 2: Separate features and target
    features = data.drop(columns=[target_column, 'date'])
    target = data[target_column]

    # Step 3: Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Step 4: Define preprocessing pipeline for categorical and numerical columns
    categorical_features = [
        'country', 'store', 'product', 'country_store', 
        'country_product', 'store_product', 'country_store_product'
    ]
    numeric_features = list(
        features.drop(columns=categorical_features).columns
    )

    # Add lagged features and rolling means to numeric features
    lagged_features = [
        col for col in numeric_features if "lag_" in col or "rolling_mean" in col
    ]
    numeric_features = lagged_features + [
        col for col in numeric_features if col not in lagged_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='constant', fill_value=0), numeric_features),  # Handle missing numeric
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Encode categorical
        ]
    )

    # Step 5: Define the base XGBoost pipeline
    xgb_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42, n_jobs=-1))
    ])

    # Step 6: Define hyperparameter grid
    param_grid = {
        'regressor__n_estimators': [100, 500, 1000],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [1, 3, 5],
        'regressor__subsample': [0.6, 0.8, 1.0],
        'regressor__colsample_bytree': [0.6, 0.8, 1.0]
    }

    # Step 7: Perform randomized search with cross-validation
    randomized_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_grid,
        n_iter=20,  # Number of random combinations
        scoring=make_scorer(mean_absolute_percentage_error, greater_is_better=False),
        cv=5,  # 5-fold cross-validation
        random_state=42,
        verbose=4,
        n_jobs=-1
    )

    randomized_search.fit(X_train, y_train)

    # Best model and hyperparameters
    best_model = randomized_search.best_estimator_
    best_params = randomized_search.best_params_

    # Step 8: Evaluate the best model
    y_pred = best_model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)

    return {
        "Model": best_model,
        "Best Params": best_params,
        "MAPE": mape,
        "MSE": mse
    }

# Call the function and display results
xgboost_complex_results = train_xgboost_model_with_optimization(train_data)
print(f"MAPE: {xgboost_complex_results['MAPE']:.4f}")
print(f"MSE: {xgboost_complex_results['MSE']:.2f}")
print(f"Best Parameters: {xgboost_complex_results['Best Params']}")


def compare_models(models_results, metric='MAPE'):
    """
    Compares multiple trained models based on a specified metric and returns the best model.
    
    Args:
        models_results (list): A list of dictionaries, each containing 'Model', 'MAPE', and 'MSE' keys.
        metric (str): The metric to use for comparison ('MAPE' or 'MSE').
        
    Returns:
        dict: The results of the best model.
    """
    if not models_results:
        raise ValueError("The models_results list cannot be empty.")
    
    if metric not in ['MAPE', 'MSE']:
        raise ValueError("Invalid metric. Choose 'MAPE' or 'MSE'.")

    # Sort models by the chosen metric in ascending order (lower is better)
    best_model_results = min(models_results, key=lambda x: x[metric])

    # Print all models' metrics for comparison
    print("Model Comparison Results:")
    for idx, result in enumerate(models_results, start=1):
        print(f"Model {idx}: {metric} = {result[metric]:.4f}")

    print(f"\nBest Model selected based on {metric}: {best_model_results[metric]:.4f}")
    return best_model_results

# Compare the models
models_results = [simple_model_results, xgboost_results, xgboost_complex_results]
best_model_results = compare_models(models_results, metric='MAPE')


def generate_submission(test_data, trained_model, output_file='submission.csv'):
    """
    Generates a submission file for a Kaggle competition.
    
    Args:
        test_data (pd.DataFrame): The test dataset.
        trained_model: The trained model for making predictions.
        output_file (str): The name of the submission file to save.
    
    Returns:
        pd.DataFrame: The submission DataFrame.
    """
    # Step 1: Copy data not to modify it
    data = test_data.copy()
    
    # Step 2: Prepare features for prediction
    features = data.drop(columns=['date', 'num_sold'])

    # Step 3: Make predictions
    data['num_sold'] = trained_model.predict(features)
    
    # Step 4: Create the submission DataFrame
    submission = data.reset_index()[['id', 'num_sold']]

    # Step 5: Save the submission file
    submission.to_csv(output_file, index=False)
    print(f"Submission file saved as {output_file}")

    # Step 6: Print the head of the submission using tabulate
    print("\nSubmission Preview:")
    print(tabulate(submission.head(), headers='keys', tablefmt='pretty'))

    return submission

# Generate submission file using the XGBoost model
submission = generate_submission(test_data, best_model_results["Model"])




