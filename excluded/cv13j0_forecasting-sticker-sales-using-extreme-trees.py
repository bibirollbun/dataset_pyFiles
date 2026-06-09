# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


%%capture
!pip install holidays


# Importing additional libraries
from sklearn.preprocessing import LabelEncoder # label encoder function from sklearn
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error, mean_absolute_percentage_error


%%time
# I like to disable my Notebook Warnings.
import warnings
warnings.filterwarnings('ignore')

# Configure notebook display settings to only use 2 decimal places, tables look nicer.
pd.options.display.float_format = '{:,.3f}'.format
pd.set_option('display.max_columns', 15) 
pd.set_option('display.max_rows', 25)

# Define some of the notebook parameters for future experiment replication.
SEED   = 548


def load_csv_to_dataframe(file_path, ignore_fields=[]):
    """
    Load a CSV file into a pandas DataFrame, optionally ignoring specified fields.

    Parameters:
    file_path (str): The file path of the CSV file to be loaded.
    ignore_fields (list): A list of field names to be ignored when loading the CSV.

    Returns:
    pandas.DataFrame: A DataFrame containing the data from the CSV file, excluding the ignored fields.
    """
    # Read the CSV file from the given file path using pandas
    df = pd.read_csv(file_path)
    
    # Drop the fields that need to be ignored, if they exist in the DataFrame
    df = df.drop(columns=ignore_fields, errors='ignore')
    
    # Return the resulting DataFrame
    return df

# Example usage:
# df = load_csv_to_dataframe('data/sample.csv', ignore_fields=['column_to_ignore'])
# print(df.head())


# Load the competiion dataset
trn_input = '/kaggle/input/playground-series-s5e1/train.csv'
tst_input = '/kaggle/input/playground-series-s5e1/test.csv'
sub_input = '/kaggle/input/playground-series-s5e1/sample_submission.csv' 
gdp_input = '/kaggle/input/gpd-gpd-per-capita-by-country/GPD By Country.csv'

trn_df = load_csv_to_dataframe(trn_input, ignore_fields=[])
tst_df = load_csv_to_dataframe(tst_input, ignore_fields=[])
sub_df = load_csv_to_dataframe(sub_input)
gdp_df = load_csv_to_dataframe(gdp_input)

#gdp_df = gdp_df.rename(columns = {'Country Name': 'country', 'Year': 'year'}) # Rename some of the columsn to align with the train dataset.
#gdp_df['year'] = gdp_df['year'].astype('int32')


trn_df.info()


tst_df.info()


gdp_df.info()


gdp_df['year'].unique()


sub_df.info()


def extensive_eda(df):
    """
    Perform exploratory data analysis (EDA) on the given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.

    Returns:
    None
    """
    from IPython.display import display

    # Display the DataFrame info
    print("Information about the DataFrame:")
    df_info = df.info()
    display(df_info)
    print(".....")
    print("\n")

    # Display the first few rows of data
    print("First few rows of the DataFrame:")
    display(df.head().T)
    print(".....")
    print("\n")
    
    # Display the number of duplicate values in each column
    print("Number of duplicate values in each column:")
    duplicate_counts = df.duplicated().sum()
    display(duplicate_counts)
    print(".....")
    print("\n")
    
    # Display the number of missing datapoints in each column
    print("Number of missing datapoints in each column:")
    missing_counts = df.isna().sum()
    display(missing_counts)
    print(".....")
    print("\n")
    
    # Display the number of outliers in each column using the IQR technique
    print("Number of outliers in each column (using IQR technique):")
    outliers = {}
    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers[column] = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))].shape[0]
    display(outliers)
    print(".....")
    print("\n")
    
    # Display basic statistics of the DataFrame
    print("Statistical summary of the DataFrame:")
    display(df.describe().T)
    print(".....")
    print("\n")
    
    # Display unique value count for each column
    print("Number of unique values in each column:")
    unique_counts = df.nunique()
    display(unique_counts)
    print(".....")
    print("\n")
    
    # Display column-wise summary in a table
    print("Column-wise summary:")
    summary_data = []
    for column in df.columns:
        column_summary = {
            "Column": column,
            "Data Type": df[column].dtype,
            "Missing Values": missing_counts[column],
            "Unique Values": unique_counts[column],
            "Outliers": outliers.get(column, 0) if df[column].dtype in ['int64', 'float64'] else "N/A",
            "Top 5 Values": df[column].value_counts().head().to_dict() if df[column].dtype == 'object' else "N/A"
        }
        summary_data.append(column_summary)
    summary_df = pd.DataFrame(summary_data)
    display(summary_df.style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector='th', props=[('text-align', 'left')])]))
    print(".....")
    print("\n")
    
    # Display correlation matrix for numerical features only
    print("Correlation matrix of numerical features:")
    numerical_df = df.select_dtypes(include=['number'])
    display(numerical_df.corr())
    print(".....")
    print("\n")
    
    # Display value counts for categorical columns in a table with widened format
    print("Value counts for categorical columns:")
    value_counts_data = []
    for column in df.select_dtypes(include=['object']).columns:
        value_counts = df[column].value_counts().head().to_dict()
        value_counts_data.append({"Column": column, "Top 5 Values": value_counts})
    value_counts_df = pd.DataFrame(value_counts_data)
    display(value_counts_df.style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector='th', props=[('text-align', 'left')])]))
    print(".....")
    print("\n")

# Example usage:
# df = load_csv_to_dataframe('data/sample.csv')
# perform_eda(df)


# EDA 
extensive_eda(trn_df)


%%time
# Create a simple function to evaluate the time-ranges of the information provided.
# It will help with the train / validation separations

def evaluate_time(df):
    min_date = df['date'].min()
    max_date = df['date'].max()
    print(f'Min Date: {min_date} /  Max Date: {max_date}')
    return None

evaluate_time(trn_df)
evaluate_time(tst_df)


# Define the function to extract time value features
def extract_time_features(df, datetime_column):
    # Ensure the column is in datetime format
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    
    # Extract various time features
    df['year'] = df[datetime_column].dt.year
    df['month'] = df[datetime_column].dt.month
    df['day'] = df[datetime_column].dt.day
    
    df['day_of_week'] = df[datetime_column].dt.dayofweek  # Monday=0, Sunday=6
    #df['day_name'] = df[datetime_column].dt.day_name()
    #df['month_name'] = df[datetime_column].dt.month_name()
    df['week_of_year'] = df[datetime_column].dt.isocalendar().week
    df['quarter'] = df[datetime_column].dt.quarter
    df['is_weekend'] = df[datetime_column].dt.dayofweek >= 5
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    #df['hour'] = df[datetime_column].dt.hour
    #df['minute'] = df[datetime_column].dt.minute
    #df['second'] = df[datetime_column].dt.second
    #df['microsecond'] = df[datetime_column].dt.microsecond

    df = df.drop(columns = [
                            #datetime_column, 
                            #'day',
                            #'month'
                           ])

    return df

# Example usage
# data = {'datetime': ['2023-12-23 15:21:39.134960', '2023-11-22 11:05:13.123456']}
# df = pd.DataFrame(data)
# df = extract_time_features(df, 'datetime')
# print(df)


# Creates time features
trn_df = extract_time_features(trn_df, 'date')
tst_df = extract_time_features(tst_df, 'date')


trn_df = trn_df.merge(gdp_df, how = 'left', on = ['country', 'year'])
tst_df = tst_df.merge(gdp_df, how = 'left', on = ['country', 'year'])


trn_df.sample(5)


trn_df['country'].unique()


gdp_df.info()


# Computing GDP Ratios for each year.
list_years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
# Filter GDP to only account for countries in the competition
gdp_filter_df = gdp_df.loc[gdp_df['country'].isin(trn_df['country'].unique()) & gdp_df['year'].isin(list_years) ,:]
# Calculates the GDP Ratios by year for each of the countries
gdp_filter_df['pct_gdp'] = gdp_filter_df['GPD per Capita'] / gdp_filter_df.groupby('year')['GPD per Capita'].transform('sum')


gdp_filter_df


print(f"Missing values remaining: {trn_df['num_sold'].isna().sum()}")


# Imputing values using Ratios for Canada
for year in trn_df['year'].unique():
    # Calculating ratios.
    target_ratio = gdp_filter_df.loc[((gdp_filter_df['year'] == year) & (gdp_filter_df['country'] == 'Norway')), 'pct_gdp'].values[0]
    current_ratio = gdp_filter_df.loc[((gdp_filter_df['year'] == year) & (gdp_filter_df['country'] == 'Canada')), 'pct_gdp'].values[0]
    canada_ratio = current_ratio / target_ratio
    # ------------------------------------------------------------
    
    # Imputing the values on the dataframe Canada.
    trn_df.loc[(trn_df['year'] == year) 
                & (trn_df['country'] == 'Canada') 
                & (trn_df['store'] == 'Discount Stickers') 
                & (trn_df['product'] == 'Holographic Goose'),
                'num_sold'] = (trn_df.loc[(trn_df['year'] == year) 
                & (trn_df['country'] == 'Norway') 
                & (trn_df['store'] == 'Discount Stickers') 
                & (trn_df['product'] == 'Holographic Goose'),
                'num_sold'] * canada_ratio).values
    # ------------------------------------------------------------
    
    # Imputing only missing values, Canada Premium Sticker Mart
    selected_time_series = trn_df.loc[(trn_df['year'] == year) 
                & (trn_df['country'] == 'Canada') 
                & (trn_df['store'] == 'Premium Sticker Mart') 
                & (trn_df['product'] == 'Holographic Goose')]

    missing_time_series = selected_time_series.loc[selected_time_series['num_sold'].isna(), 'date']
    
    trn_df.loc[(trn_df['country'] == 'Canada') 
    & (trn_df['store'] == 'Premium Sticker Mart') 
    & (trn_df['product'] == "Holographic Goose") 
    & (trn_df['year'] == year) 
    & (trn_df['date'].isin(missing_time_series)), 
    'num_sold'] = (trn_df.loc[(trn_df['country'] == 'Norway') 
                   & (trn_df['store'] == 'Premium Sticker Mart') 
                   & (trn_df['product'] == 'Holographic Goose') 
                   & (trn_df['year'] == year) 
                   & (trn_df['date'].isin(missing_time_series)), 
                   'num_sold'] * canada_ratio).values
    # ------------------------------------------------------------

    # Imputing only missing values, Canada, Stickers for Less
    selected_time_series = trn_df.loc[(trn_df['year'] == year) 
                & (trn_df['country'] == 'Canada') 
                & (trn_df['store'] == 'Stickers for Less') 
                & (trn_df['product'] == 'Holographic Goose')]

    missing_time_series = selected_time_series.loc[selected_time_series['num_sold'].isna(), 'date']
    
    trn_df.loc[(trn_df['country'] == 'Canada') 
    & (trn_df['store'] == 'Stickers for Less') 
    & (trn_df['product'] == "Holographic Goose") 
    & (trn_df['year'] == year) 
    & (trn_df['date'].isin(missing_time_series)), 
    'num_sold'] = (trn_df.loc[(trn_df['country'] == 'Norway') 
                   & (trn_df['store'] == 'Stickers for Less') 
                   & (trn_df['product'] == 'Holographic Goose') 
                   & (trn_df['year'] == year) 
                   & (trn_df['date'].isin(missing_time_series)), 
                   'num_sold'] * canada_ratio).values
    # ------------------------------------------------------------


print(f"Missing values remaining: {trn_df['num_sold'].isna().sum()}")


# Imputing values using Ratios for Kenya
for year in trn_df['year'].unique():
    # Calculating ratios.
    target_ratio = gdp_filter_df.loc[((gdp_filter_df['year'] == year) & (gdp_filter_df['country'] == 'Norway')), 'pct_gdp'].values[0]
    current_ratio = gdp_filter_df.loc[((gdp_filter_df['year'] == year) & (gdp_filter_df['country'] == 'Kenya')), 'pct_gdp'].values[0]
    kenya_ratio = current_ratio / target_ratio
    # ------------------------------------------------------------
    
    # Imputing only missing values, Kenya, Premium Sticker Mart
    selected_time_series = trn_df.loc[(trn_df['year'] == year) 
                & (trn_df['country'] == 'Kenya') 
                & (trn_df['store'] == 'Premium Sticker Mart') 
                & (trn_df['product'] == 'Holographic Goose')]

    missing_time_series = selected_time_series.loc[selected_time_series['num_sold'].isna(), 'date']
    
    trn_df.loc[(trn_df['country'] == 'Kenya') 
    & (trn_df['store'] == 'Premium Sticker Mart') 
    & (trn_df['product'] == "Holographic Goose") 
    & (trn_df['year'] == year) 
    & (trn_df['date'].isin(missing_time_series)), 
    'num_sold'] = (trn_df.loc[(trn_df['country'] == 'Norway') 
                   & (trn_df['store'] == 'Premium Sticker Mart') 
                   & (trn_df['product'] == 'Holographic Goose') 
                   & (trn_df['year'] == year) 
                   & (trn_df['date'].isin(missing_time_series)), 
                   'num_sold'] * kenya_ratio).values
    # ------------------------------------------------------------

    # Imputing only missing values, Kenya, Stickers for Less
    selected_time_series = trn_df.loc[(trn_df['year'] == year) 
                & (trn_df['country'] == 'Kenya') 
                & (trn_df['store'] == 'Stickers for Less') 
                & (trn_df['product'] == 'Holographic Goose')]

    missing_time_series = selected_time_series.loc[selected_time_series['num_sold'].isna(), 'date']
    
    trn_df.loc[(trn_df['country'] == 'Kenya') 
    & (trn_df['store'] == 'Stickers for Less') 
    & (trn_df['product'] == "Holographic Goose") 
    & (trn_df['year'] == year) 
    & (trn_df['date'].isin(missing_time_series)), 
    'num_sold'] = (trn_df.loc[(trn_df['country'] == 'Norway') 
                   & (trn_df['store'] == 'Stickers for Less') 
                   & (trn_df['product'] == 'Holographic Goose') 
                   & (trn_df['year'] == year) 
                   & (trn_df['date'].isin(missing_time_series)), 
                   'num_sold'] * kenya_ratio).values
    # ------------------------------------------------------------

    # Imputing only missing values, Kenya, Discount Stickers
    selected_time_series = trn_df.loc[(trn_df['year'] == year) 
                & (trn_df['country'] == 'Kenya') 
                & (trn_df['store'] == 'Discount Stickers') 
                & (trn_df['product'] == 'Kerneler')]

    missing_time_series = selected_time_series.loc[selected_time_series['num_sold'].isna(), 'date']
    
    trn_df.loc[(trn_df['country'] == 'Kenya') 
    & (trn_df['store'] == 'Discount Stickers') 
    & (trn_df['product'] == "Kerneler") 
    & (trn_df['year'] == year) 
    & (trn_df['date'].isin(missing_time_series)), 
    'num_sold'] = (trn_df.loc[(trn_df['country'] == 'Norway') 
                   & (trn_df['store'] == 'Discount Stickers') 
                   & (trn_df['product'] == 'Kerneler') 
                   & (trn_df['year'] == year) 
                   & (trn_df['date'].isin(missing_time_series)), 
                   'num_sold'] * kenya_ratio).values

    # Imputing only missing values, Kenya, Premium Sticker Mart
    selected_time_series = trn_df.loc[(trn_df['year'] == year) 
                & (trn_df['country'] == 'Kenya') 
                & (trn_df['store'] == 'Discount Stickers') 
                & (trn_df['product'] == 'Holographic Goose')]

    missing_time_series = selected_time_series.loc[selected_time_series['num_sold'].isna(), 'date']
    
    trn_df.loc[(trn_df['country'] == 'Kenya') 
    & (trn_df['store'] == 'Discount Stickers') 
    & (trn_df['product'] == "Holographic Goose") 
    & (trn_df['year'] == year) 
    & (trn_df['date'].isin(missing_time_series)), 
    'num_sold'] = (trn_df.loc[(trn_df['country'] == 'Norway') 
                   & (trn_df['store'] == 'Discount Stickers') 
                   & (trn_df['product'] == 'Holographic Goose') 
                   & (trn_df['year'] == year) 
                   & (trn_df['date'].isin(missing_time_series)), 
                   'num_sold'] * kenya_ratio).values
    # ------------------------------------------------------------


# Imputing the remaining values in the dataframe
missing_rows = trn_df.loc[trn_df["num_sold"].isna()]
display(missing_rows)
trn_df.loc[trn_df["id"] == 23719, "num_sold"] = 4
trn_df.loc[trn_df["id"] == 207003, "num_sold"] = 195

print(f"Missing values remaining: {trn_df['num_sold'].isna().sum()}")


print(f"Missing values remaining: {trn_df['num_sold'].isna().sum()}")


# Creating a Totals Sales Dataframe...


%%time
import holidays
years_list = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]

holiday_CA = holidays.CountryHoliday('CA', years = years_list)
holiday_FI = holidays.CountryHoliday('FI', years = years_list)
holiday_IT = holidays.CountryHoliday('IT', years = years_list)
holiday_KE = holidays.CountryHoliday('KE', years = years_list)
holiday_NO = holidays.CountryHoliday('NO', years = years_list)
holiday_SG = holidays.CountryHoliday('SG', years = years_list)

holiday_dict = holiday_CA.copy()
holiday_dict.update(holiday_FI)
holiday_dict.update(holiday_IT)
holiday_dict.update(holiday_KE)
holiday_dict.update(holiday_NO)
holiday_dict.update(holiday_SG)

def map_holydays(df, map_dict = holiday_dict):
    '''
    Describe the function...
    '''
    df['date'] = pd.to_datetime(df['date']) # Convert the date to datetime.
    df['holiday_name'] = df['date'].map(holiday_dict)
    df['is_holiday'] = np.where(df['holiday_name'].notnull(), 1, 0)
    df['holiday_name'] = df['holiday_name'].fillna('Not Holiday')

    return df
    
trn_df = map_holydays(trn_df, holiday_dict)
tst_df = map_holydays(tst_df, holiday_dict)


trn_df = trn_df.drop(columns = ['date', 'holiday_name'])
tst_df = tst_df.drop(columns = ['date', 'holiday_name'])


# Create a list of categorical variables to help the label encoding function
categorical_fields = ['country', 'store', 'product']


def label_encode_datasets(train_df, test_df, categ_fields):
    """
    Label encode the categorical variables of the train and test DataFrames.

    Parameters:
    train_df (pandas.DataFrame): The training DataFrame.
    test_df (pandas.DataFrame): The testing DataFrame.

    Returns:
    tuple: A tuple containing the label encoded training and testing DataFrames.
    """
    # Create a copy of train and test dataframes to avoid modifying original dataframes
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    # Identify categorical columns
    # categorical_columns = test_encoded.select_dtypes(include=['object']).columns
    categorical_columns = categ_fields
    
    # Initialize label encoder
    le = LabelEncoder()
    
    # Apply label encoding to each categorical column
    for column in categorical_columns:
        print(f'Encoding: {column} ...')
        # Fit the label encoder on the train data
        le.fit(train_encoded[column])
        
        # Transform both train and test data using the same encoder
        train_encoded[column] = le.transform(train_encoded[column])
        if column in test_encoded.columns:
            # Handle cases where test set may have unseen labels by using fillna
            test_encoded[column] = test_encoded[column].map(lambda s: le.transform([s])[0] if s in le.classes_ else None)
            test_encoded[column].fillna(-1, inplace=True)
            test_encoded[column] = test_encoded[column].astype(int)

    return train_encoded, test_encoded

# Example usage:
# train_df = load_csv_to_dataframe('data/train.csv')
# test_df = load_csv_to_dataframe('data/test.csv')
# train_encoded, test_encoded = label_encode_datasets(train_df, test_df)


# Encoding the train and test datasets.
trn_encoded, tst_encoded = label_encode_datasets(trn_df, tst_df, categorical_fields)


trn_encoded = trn_encoded.dropna()


tst_encoded.info()


trn_encoded['num_sold'] = np.log(trn_encoded['num_sold'])


# Function to train an XGBoost Regressor with GPU support
def train_xgboost_regressor(train_df, test_df, target_column, param_file=None, n_splits=10):
    """
    Train an XGBoost regressor using the provided training and test datasets with K-Fold cross-validation, utilizing GPU support.

    Parameters:
    train_df (pandas.DataFrame): The training DataFrame.
    test_df (pandas.DataFrame): The testing DataFrame.
    target_column (str): The name of the target column.
    param_file (dict): Dictionary of hyperparameters for the XGBoost model.
    n_splits (int): The number of folds for cross-validation.

    Returns:
    tuple: A tuple containing the model performance metrics and the predictions on the test dataset.
    """
    from xgboost import XGBRegressor
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
    import numpy as np

    # Separate features and target from the training data
    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]
    
    # Set default parameters if none are provided
    if param_file is None:
        param_file = {
            'n_estimators': 4096,
            'learning_rate': 0.01,
            'max_depth': 12,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'gpu_hist',  # Enable GPU support
            'predictor': 'gpu_predictor',
            'eval_metric': 'mape',
            'random_state': SEED
        }

    # Initialize the XGBoost model with parameters from param_file
    model = XGBRegressor(**param_file)

    # Initialize KFold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # Lists to store cross-validation results
    mse_scores = []
    mae_scores = []
    mape_scores = []
    rmsle_scores = []
    test_predictions = []

    # Perform K-Fold cross-validation

    fold_number = 0
    for train_index, val_index in kf.split(X):
        fold_number += 1
        #print('Fold:', fold_number)

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the validation set
        y_val_pred = model.predict(X_val)

        # Calculate validation metrics
        mse_scores.append(mean_squared_error(y_val, y_val_pred))
        mae_scores.append(mean_absolute_error(y_val, y_val_pred))
        mape_scores.append(mean_absolute_percentage_error(y_val, y_val_pred))
        #rmsle_scores.append(np.sqrt(mean_squared_log_error(y_val, np.maximum(y_val_pred, 0))))

        # Make predictions on the test dataset for each fold
        X_test = test_df.drop(columns=[target_column], errors='ignore')
        test_predictions.append(model.predict(X_test))

        print(f'Fold {fold_number} MAPE = {mape_scores[fold_number-1]}')

    # Calculate average metrics
    avg_mse = np.mean(mse_scores)
    avg_mae = np.mean(mae_scores)
    avg_mape = np.mean(mape_scores)

    # Print the metrics in a readable format
    print("Model Performance Metrics (Cross-Validation):")
    print("..................")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average MAPE: {avg_mape:.4f}")

    # Calculate the average predictions across all folds
    y_test_pred = np.mean(test_predictions, axis=0)

    return y_test_pred

# Example usage:
# param_file = {
#     'n_estimators': 200,
#     'learning_rate': 0.05,
#     'max_depth': 8,
#     'tree_method': 'gpu_hist',  # Enable GPU support
#     'predictor': 'gpu_predictor'
# }
# train_df = load_csv_to_dataframe('data/train.csv')
# test_df = load_csv_to_dataframe('data/test.csv')
# test_predictions = train_xgboost_regressor(train_df, test_df, 'target', param_file=param_file)
# print(test_predictions)


test_predictions = train_xgboost_regressor(trn_encoded, tst_encoded, 'num_sold')


# Average MAE: 38.3415
# Average MAPE: 0.1062
# Average MAPE: 0.0596
# Average MAPE: 0.0089 >>> Changed the Target to Log...
# Average MAPE: 0.0089


sub_df['num_sold'] = np.exp(test_predictions)

sub_df.to_csv('submission.csv', index=False)
display(sub_df.head())


sub_df




