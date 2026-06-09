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


import os  # Operating system interactions

import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Data visualization
import seaborn as sns  # High-level data visualization based on matplotlib
from scipy import stats 

# Ignore all warnings
import warnings
warnings.filterwarnings('ignore')


train_df=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
training_extra_df=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
test=test_df.copy()


pd.set_option('display.max_rows', None)  # To display all rows of the dataframe.
pd.set_option('display.max_columns', None)  # To display all columns of the dataframe.


print(train_df.shape)
print(training_extra_df.shape)
print(test_df.shape)


train_df.head()


training_extra_df.head()


test_df.head()


def analyze_missing_data(dataframe):
    """
    Analyzes missing data in the provided DataFrame.
    Parameters:
        dataframe (pd.DataFrame): The input DataFrame to analyze.
    Returns:
        pd.DataFrame: A DataFrame containing missing percentages, data types, and null counts.
    """
    # Calculate missing percentages for each column
    missing_percent = (dataframe.isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_percent = missing_percent.apply(lambda x: f"{x:.2f}%")

    # Extract DataFrame info (data types and null counts)
    info_df = pd.DataFrame({
        'DataType': dataframe.dtypes,  # Get the data type of each column
        'Null Count': dataframe.isnull().sum()  # Count the number of null values in each column
    })
    
    # Combine missing percentage and column information into a single DataFrame
    combined_df = pd.concat([missing_percent, info_df], axis=1)
    # Rename the columns of the resulting DataFrame
    combined_df.columns = ['Missing Percent', 'DataType', 'Null Count']

    # Display the total number of rows in the dataset
    print(f'Number of rows: {dataframe.shape[0]}')
    # Return the DataFrame sorted by missing percentage in descending order
    return combined_df.sort_values(by='Missing Percent', ascending=False)


analyze_missing_data(train_df)


analyze_missing_data(test_df)


analyze_missing_data(training_extra_df)


def plot_price_distributions(train, train_ex, column="Price"):
    """
    Plots the distribution of a specified column (default: "Price") 
    in the train and train_ex datasets.
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(train[column], bins=50, kde=True, color='blue')
    plt.title("Train [{}] Distribution".format(column))
    plt.xlabel(column)

    plt.subplot(1, 2, 2)
    sns.histplot(train_ex[column], bins=50, kde=True, color='green')
    plt.title("Train_ex [{}] Distribution".format(column))
    plt.xlabel(column)

    plt.tight_layout()
    plt.show()


plot_price_distributions(train_df,  training_extra_df)


def plot_numeric_distributions(train, test, train_ex):
    """
    Plots the distribution of all numeric columns in train, test, and train_ex datasets.
    """
    num_cols = test.select_dtypes(include=['number']).columns  # Get numeric columns

    plt.figure(figsize=(12, len(num_cols) * 3))

    for i, col in enumerate(num_cols):
        plt.subplot(len(num_cols), 3, i * 3 + 1)
        sns.histplot(train[col], bins=10, color='blue')
        plt.title(f"Train [{col}] Distribution")
        plt.xlabel(col)

        plt.subplot(len(num_cols), 3, i * 3 + 2)
        sns.histplot(test[col], bins=10, color='green')
        plt.title(f"Test [{col}] Distribution")
        plt.xlabel(col)

        plt.subplot(len(num_cols), 3, i * 3 + 3)
        sns.histplot(train_ex[col], bins=10, color='red')
        plt.title(f"Train_ex [{col}] Distribution")
        plt.xlabel(col)

    plt.tight_layout()
    plt.show()


plot_numeric_distributions(train_df, test_df, training_extra_df)


def plot_categorical_pie_charts(train, test, train_ex):
    """
    Plots pie chart comparisons of categorical variables in train, test, and train_ex datasets.
    """
    obj_cols = train.select_dtypes(include=['object']).columns  # Get categorical columns

    for variable in obj_cols:
        sns.set_style('whitegrid')

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(wspace=0.3)

        # Pie Chart for Train
        train[variable].value_counts().plot.pie(ax=axes[0], autopct='%1.1f%%', startangle=90)
        axes[0].set_ylabel('')
        axes[0].set_title(f"Train [{variable}]")

        # Pie Chart for Test
        test[variable].value_counts().plot.pie(ax=axes[1], autopct='%1.1f%%', startangle=90)
        axes[1].set_ylabel('')
        axes[1].set_title(f"Test [{variable}]")

        # Pie Chart for Train_ex
        train_ex[variable].value_counts().plot.pie(ax=axes[2], autopct='%1.1f%%', startangle=90)
        axes[2].set_ylabel('')
        axes[2].set_title(f"Train_ex [{variable}]")

        plt.show()


plot_categorical_pie_charts(train_df, test_df, training_extra_df)


import matplotlib.pyplot as plt

def plot_missing_data(*datasets, names=None):
    """
    Plots comparative charts (pie charts and bar plots) of missing values 
    for multiple datasets.

    Parameters:
        *datasets: Multiple Pandas DataFrames.
        names (list, optional): List of dataset names for reference.
    
    Displays:
        Pie charts and bar plots for missing values in each dataset.
    """
    if names is None:
        names = [f"Dataset {i+1}" for i in range(len(datasets))]

    missing_values = {
        name: df.isnull().sum()[df.isnull().sum() > 0]  # Only columns with missing values
        for df, name in zip(datasets, names)
    }

    # Filter datasets that have missing values
    missing_values = {name: values for name, values in missing_values.items() if not values.empty}

    if not missing_values:
        print("No missing values in the provided datasets.")
        return

    fig, axes = plt.subplots(len(missing_values), 2, figsize=(12, len(missing_values) * 5))

    if len(missing_values) == 1:  # Ensure axes is iterable when only one dataset
        axes = [axes]

    for ax, (name, missing_data) in zip(axes, missing_values.items()):
        # Pie chart
        ax[0].pie(missing_data, labels=missing_data.index, autopct='%1.1f%%', startangle=90)
        ax[0].set_title(f'Missing Values in {name} Dataset')

        # Bar plot
        ax[1].barh(missing_data.index, missing_data.values, color='skyblue')
        ax[1].set_title(f'Missing Values in {name} Dataset')
        ax[1].set_xlabel('Count')
        ax[1].invert_yaxis()

    plt.tight_layout()
    plt.show()

# Example usage:
# plot_missing_data(train, test, train_ex, names=["Train", "Test", "Train_ex"])



plot_missing_data(train_df, test_df, training_extra_df, names=["Train", "Test", "Train_ex"])


import missingno as msno


def plot_missing_data_matrix(*datasets, names=None):
    """
    Plots missing data locations using the missingno matrix for multiple datasets.

    Parameters:
        *datasets: Multiple Pandas DataFrames.
        names (list, optional): List of dataset names for reference.
    
    Displays:
        Missing data matrices for each dataset.
    """
    if names is None:
        names = [f"Dataset {i+1}" for i in range(len(datasets))]

    colors = [(0.0, 0.2, 0.4), (0.0, 0.4, 0.2), (0.6, 0.2, 0.0), (0.4, 0.0, 0.6), (0.2, 0.6, 0.2)]  # Different color options

    for i, (df, name) in enumerate(zip(datasets, names)):
        plt.figure(figsize=(12, 6))
        msno.matrix(df, color=colors[i % len(colors)])  # Cycle through colors
        plt.title(f"Missing Data Locations in {name} Dataset", fontsize=24)
        plt.xlabel("Columns", fontsize=20)
        plt.show()

# Example usage:
# plot_missing_data_matrix(train, test, train_ex, names=["Train", "Test", "Train_ex"])


plot_missing_data_matrix(train_df, test_df, training_extra_df, names=["Train", "Test", "Train_ex"])


import pandas as pd

def highlight_missing(val):
    """Highlights missing values in a DataFrame."""
    return 'background-color: SkyBlue; border: 1px solid red' if pd.isna(val) else ''

def get_representative_rows(df):
    """
    Returns representative rows that contain missing values.
    Selects the first occurrence for each column with missing data.
    """
    columns_with_issues = df.columns[df.isnull().sum() > 0]
    representative_rows = pd.concat(
        [df[df[col].isnull()].iloc[:1] for col in columns_with_issues]
    ).drop_duplicates()
    
    # Sort if 'id' column exists, else keep the default order
    return representative_rows.sort_values(by='id') if 'id' in representative_rows.columns else representative_rows

def visualize_missing_values(*datasets, names=None):
    """
    Highlights missing values in the given datasets.
    
    Parameters:
        *datasets: Multiple Pandas DataFrames to check for missing values.
        names (list, optional): List of dataset names for reference.
        
    Displays:
        Styled DataFrames with missing values highlighted.
    """
    if names is None:
        names = [f"Dataset {i+1}" for i in range(len(datasets))]

    for df, name in zip(datasets, names):
        print(f"Missing Values in {name} Dataset")
        representative_rows = get_representative_rows(df)
        display(representative_rows.style.applymap(highlight_missing))

# visualize_missing_values(train, test, train_ex, names=["Train", "Test", "Train_ex"])



visualize_missing_values(train_df, test_df, training_extra_df, names=["Train", "Test", "Train_extra"])


# Merging Train_df and Training_extra_df Data
train_mr = pd.concat([train_df, training_extra_df], axis=0, ignore_index=True)


analyze_missing_data(train_mr)


plot_missing_data_matrix(train_mr,names=['Train Merge'])


def get_missing_data_indices(*datasets, names=None, top_n=100, min_nans=1, max_nans=None):
    """
    Returns a list of row indices sorted by missing values, filtered by minimum and maximum NaNs.

    Parameters:
        *datasets: Multiple Pandas DataFrames.
        names (list, optional): List of dataset names for reference.
        top_n (int, optional): Maximum number of rows to return, default is 500.
        min_nans (int, optional): Minimum number of NaN values a row must have to be included.
        max_nans (int, optional): Maximum number of NaN values a row can have to be included.

    Returns:
        list: A list of row indices sorted by missing values across all datasets.
    """
    missing_data_info = []  # List to store row indices across all datasets

    for df in datasets:
        # Count NaNs per row
        nan_counts = df.isna().sum(axis=1)
        
        # Filter rows based on min and max NaNs
        if max_nans is not None:
            filtered_rows = nan_counts[(nan_counts >= min_nans) & (nan_counts <= max_nans)]
        else:
            filtered_rows = nan_counts[nan_counts >= min_nans]
        
        # Sort rows by NaN count (descending) and get only indices
        sorted_row_indices = filtered_rows.sort_values(ascending=False).head(top_n).index.tolist()
        
        missing_data_info.extend(sorted_row_indices)  # Add indices to the list

    return missing_data_info

# Example usage:
# result = get_missing_data_indices(train, test, names=["Train", "Test"], top_n=100, min_nans=2, max_nans=10)
# print(result)  # List of row indices with missing values




missing_list=get_missing_data_indices(train_mr, names=["Train_Merge"], top_n=100000, min_nans=2, max_nans=10)


len(missing_list)


train_mr_update = train_mr.drop(missing_list)


analyze_missing_data(train_mr_update)


missing_list_1=get_missing_data_indices(train_mr_update, names=["Train_Merge"], top_n=100000, min_nans=2, max_nans=10)


len(missing_list_1)


train_mr_update.shape


## This will print the number of rows deleted and the percentage of rows deleted:


# Get the total number of rows before deletion
total_rows_before = train_mr.shape[0]

# Remove rows with NaN values (drop rows containing any NaN)
train_mr_cleaned = train_mr.dropna()

# Get the total number of rows after deletion
total_rows_after = train_mr_cleaned.shape[0]

# Calculate the number of rows deleted
rows_deleted = total_rows_before - total_rows_after

# Calculate the percentage of rows deleted
percentage_deleted = (rows_deleted / total_rows_before) * 100

# Print the result
print(f"Rows deleted: {rows_deleted} rows")
print(f"Percentage of rows deleted: {percentage_deleted:.2f}%")


analyze_missing_data(test_df)


# Impute missing numerical data with the median values from the TRAIN dataset

num_cols = test_df.select_dtypes(include=['number']).columns

imputation_value = train_mr_update[num_cols].median()

train_mr_update[num_cols] = train_mr_update[num_cols].fillna(imputation_value)
test_df[num_cols] = test_df[num_cols].fillna(imputation_value)


analyze_missing_data(test_df)


# Impute Missing Values in Object Columns with 'None'

obj_cols = train_mr_update.select_dtypes(include=['object']).columns

train_mr_update[obj_cols] = train_mr_update[obj_cols].fillna('None')
test_df[obj_cols] = test_df[obj_cols].fillna('None')


analyze_missing_data(test_df)


analyze_missing_data(train_mr_update)


# Drop the 'id' column from both train_mr_update and test_df
train_mr_update = train_mr_update.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)


# # Converting object type data to categorical type for compatibility with CatBoost.

# obj_cols = train_mr_update.select_dtypes(include=['object']).columns

# train_mr_update[obj_cols] = train_mr_update[obj_cols].astype('string').astype('category')
# test_df[obj_cols] = test_df[obj_cols].astype('string').astype('category')


analyze_missing_data(train_mr_update)


# Set the target variable 'Price' as y and features as X for training data

X = train_mr_update.drop(['Price'], axis=1)
y = train_mr_update['Price']


from catboost import CatBoostRegressor
from sklearn.model_selection import KFold


# # Initialize variables for storing values
# val_rmse = []
# y_test_pred = []

# # Train CatBoost using K-Fold cross-validation
# for train_id, val_id in KFold(5, shuffle=True, random_state=42).split(X, y):
#     model = CatBoostRegressor(
#         iterations=3000, learning_rate=0.28, depth=4, random_seed=42
#     )

#     X_train, X_val, y_train, y_val = X.iloc[train_id], X.iloc[val_id], y.iloc[train_id], y.iloc[val_id]
#     model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=list(obj_cols), verbose=1000)

#     # Save RMSE, predictions
#     val_rmse.append(model.get_evals_result()['validation']['RMSE'])
#     y_test_pred.append(model.predict(test_df))



# from catboost import CatBoostRegressor
# from sklearn.model_selection import GridSearchCV, KFold
# from sklearn.metrics import mean_squared_error



# # List of categorical features (replace with your actual list of categorical columns)
# obj_cols = ['Brand', 'Material','Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# # Define the CatBoost model
# model = CatBoostRegressor(
#     random_seed=42,
#     task_type='CPU',  # No GPU
#     cat_features=obj_cols,  # Categorical columns
#     verbose=1000  # Adjust verbosity as needed
# )

# # Hyperparameter grid for GridSearchCV
# param_grid = {
#     'iterations': [1000, 2000],  # Number of iterations
#     'learning_rate': [0.1, 0.28],  # Learning rate
#     'depth': [4, 6],  # Tree depth
# }

# # Setup GridSearchCV with Cross-Validation
# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     cv=3,  # 3-fold cross-validation
#     n_jobs=-1,  # Use all available CPUs
#     verbose=1,
#     scoring='neg_mean_squared_error'  # Using negative MSE as the metric
# )

# # Fit GridSearchCV to find the best hyperparameters
# grid_search.fit(X, y)

# # Best parameters from GridSearchCV
# best_params = grid_search.best_params_
# print("Best parameters found: ", best_params)

# # Use best model from GridSearchCV with early stopping
# best_model = grid_search.best_estimator_

# # Train model using K-Fold cross-validation with early stopping
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# val_rmse = []  # List to store RMSE for each fold

# for train_id, val_id in kf.split(X, y):
#     # Split the data into train and validation sets
#     X_train, X_val, y_train, y_val = X.iloc[train_id], X.iloc[val_id], y.iloc[train_id], y.iloc[val_id]
    
#     # Train model with early stopping
#     best_model.fit(
#         X_train, y_train,
#         eval_set=(X_val, y_val),
#         early_stopping_rounds=100,  # Stop after 100 rounds if no improvement
#         verbose=1000
#     )
    
#     # Predict on the validation set and calculate RMSE
#     y_val_pred = best_model.predict(X_val)
#     rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
#     val_rmse.append(rmse)

# # Print average RMSE across folds
# print("Average RMSE across folds: ", np.mean(val_rmse))


# # Initialize variables for storing values
# val_rmse = []
# y_test_pred = []

# # Reduce iterations and learning rate for faster training
# params = {
#     "iterations": 500,  # Reduce from 3000
#     "learning_rate": 0.1,  # Lower value for stability
#     "depth": 4,
#     "random_seed": 42,
#     "verbose": 500,
#     "thread_count": -1  # Use all available CPU cores
# }

# kf = KFold(5, shuffle=True, random_state=42)

# for train_id, val_id in kf.split(X, y):
#     model = CatBoostRegressor(**params)

#     X_train, X_val = X.iloc[train_id], X.iloc[val_id]
#     y_train, y_val = y.iloc[train_id], y.iloc[val_id]

#     model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=list(obj_cols), verbose=500)

#     # Save RMSE, predictions
#     val_rmse.append(model.get_best_score()['validation']['RMSE'])
#     y_test_pred.append(model.predict(test_df))

# # Average RMSE across folds
# mean_rmse = np.mean(val_rmse)
# print("Mean RMSE:", mean_rmse)


# import lightgbm as lgb
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# import numpy as np

# # Initialize variables for storing values
# val_rmse = []
# y_test_pred = []

# # Define LightGBM parameters
# params = {
#     "objective": "regression",
#     "metric": "rmse",
#     "learning_rate": 0.1,  # Lower for stability
#     "num_leaves": 31,  # Controls complexity
#     "max_depth": 4,  # Same depth as CatBoost
#     "random_state": 42,
#     "n_jobs": -1  # Use all available CPU cores
# }

# kf = KFold(5, shuffle=True, random_state=42)

# for train_id, val_id in kf.split(X, y):
#     X_train, X_val = X.iloc[train_id], X.iloc[val_id]
#     y_train, y_val = y.iloc[train_id], y.iloc[val_id]

#     # Convert to LightGBM Dataset
#     train_data = lgb.Dataset(X_train, label=y_train)
#     val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

#     # Train model
#     model = lgb.train(params, train_data, valid_sets=[val_data], verbose_eval=500)

#     # Predict & compute RMSE
#     y_val_pred = model.predict(X_val)
#     rmse = mean_squared_error(y_val, y_val_pred, squared=False)
#     val_rmse.append(rmse)

#     # Predict on test data
#     y_test_pred.append(model.predict(test_df))

# # Average RMSE across folds
# mean_rmse = np.mean(val_rmse)
# print("Mean RMSE:", mean_rmse)



obj_cols


# #Trends in Validation RMSE During Model Training

# mean_val_rmse = np.mean(val_rmse, axis=0)
# min_val_rmse = np.min(mean_val_rmse)
# min_val_rmse_iteration = np.argmin(mean_val_rmse)

# plt.figure(figsize=(9, 5))
# for idx, rmse_list in enumerate(val_rmse):
#     plt.plot(rmse_list, alpha=0.5, color='green', linestyle='--', label='Individual Validation RMSE' if idx == 0 else "")
# plt.plot(mean_val_rmse, label='Validation RMSE (Mean)', color='blue')
# plt.scatter(min_val_rmse_iteration, min_val_rmse, color='red')
# plt.text(min_val_rmse_iteration, min_val_rmse+0.01, f'min Validation RMSE: {min_val_rmse:.3f}', ha='right', fontsize=11, color='red')
# plt.xlabel('Iterations', fontsize=11)
# plt.ylabel('RMSE', fontsize=11)
# plt.title('Trends in Validation RMSE During Model Training', fontsize=12)
# plt.legend(loc='upper right', fontsize=11)

# plt.tight_layout()
# plt.show()



# Convert categorical columns to integer encoding before training
X_encoded = X.copy()
for col in obj_cols:
    X_encoded[col] = X_encoded[col].astype('category').cat.codes  # Convert categorical to integer

test_encoded = test_df.copy()
for col in obj_cols:
    test_encoded[col] = test_encoded[col].astype('category').cat.codes  # Convert categorical to integer

# Initialize variables
val_rmse_sav = []
y_val_sav = []
y_val_pred_sav = []
y_test_pred_sav = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_id, val_id in kf.split(X_encoded, y):
    
    # Define Fast CatBoost Model for CPU
    model = CatBoostRegressor(
        iterations=400,  # Reduced from 1000
        learning_rate=0.1,  # Balanced for speed + accuracy
        depth=3,  # Shallower trees = faster training
        loss_function='MAE',  # MAE converges faster
        task_type='CPU',  # Ensure CPU usage
        thread_count=-1,  # Uses all CPU cores
        random_seed=42,
        border_count=32,  # Limits split points per feature
        early_stopping_rounds=50,  # Stops early if no improvement
        verbose=100,  # Logs every 100 iterations instead of 1
        allow_writing_files=False  # Prevents unnecessary disk usage
    )
    
    # Split training and validation data
    X_train, X_val = X_encoded.iloc[train_id], X_encoded.iloc[val_id]
    y_train, y_val = y.iloc[train_id], y.iloc[val_id]
    
    # Train model
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    # Predict values
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(test_encoded)

    # Save data
    y_val_sav.append(y_val.to_numpy())  # Convert to NumPy for efficiency
    y_val_pred_sav.append(y_val_pred)
    y_test_pred_sav.append(y_test_pred)
    val_rmse_sav.append(model.get_best_score()['validation']['MAE'])  # Use MAE instead of RMSE for faster convergence

# Convert to NumPy arrays for better performance
y_val_sav = np.array(y_val_sav)
y_val_pred_sav = np.array(y_val_pred_sav)
y_test_pred_sav = np.array(y_test_pred_sav)
val_rmse_sav = np.array(val_rmse_sav)



print("Comparison of Validation True and Predicted Values")

y_true = [val for sublist in y_val_sav for val in sublist]
y_pred = [pred for sublist in y_val_pred_sav for pred in sublist]

# Plot preparation
plt.figure(figsize=(7, 5))
plt.scatter(y_true, y_pred, c=y_pred, cmap='viridis', s=20, alpha=0.7, linewidth=0.5)
cb = plt.colorbar()
#cb.set_label('Prediction values')

# Plot the diagonal line
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', linewidth=1.0)
plt.axis('equal')

plt.xlabel(f'True values (y_val)', fontsize=11)
plt.ylabel(f'Predicted values (y_val_pred)', fontsize=11)
plt.title('Comparison of True and Predicted Values (Validation)', fontsize=12)
plt.grid(True)
plt.show()


# Trends in Validation RMSE During Model Training

mean_val_rmse = np.mean(val_rmse_sav, axis=0)
min_val_rmse = np.min(mean_val_rmse)
min_val_rmse_iteration = np.argmin(mean_val_rmse)

plt.figure(figsize=(9, 5))
for idx, rmse_list in enumerate(val_rmse_sav):
    plt.plot(rmse_list, alpha=0.5, color='green', linestyle='--', label='Individual Validation RMSE' if idx == 0 else "")
plt.plot(mean_val_rmse, label='Validation RMSE (Mean)', color='blue')
plt.scatter(min_val_rmse_iteration, min_val_rmse, color='red')
plt.text(min_val_rmse_iteration, min_val_rmse+0.01, f'min Validation RMSE: {min_val_rmse:.3f}', ha='right', fontsize=11, color='red')
plt.xlabel('Iterations', fontsize=11)
plt.ylabel('RMSE', fontsize=11)
plt.title('Trends in Validation RMSE During Model Training', fontsize=12)
plt.legend(loc='upper right', fontsize=11)

plt.tight_layout()
plt.show()


print("Calculating Validation RMSE")

print(f"Validation RMSE: {min_val_rmse:.4f}")


# Distribution of Test Data Prediction

y_test_pred = np.mean(y_test_pred_sav, axis=0)

plt.figure(figsize=(6, 4))
sns.histplot(y_test_pred, bins=50, kde=True, color='blue')
plt.title("Distribution of Test Data Prediction")
plt.xlabel("Price")

plt.tight_layout()
plt.show()


submission = pd.DataFrame({'id': test['id'], 'Price': y_test_pred})
submission.to_csv('submission.csv', index=False)





