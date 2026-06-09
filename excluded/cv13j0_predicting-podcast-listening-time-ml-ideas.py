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


# Importing the nesesary libraries
import warnings
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder  # Label encoder function from sklearn


def configure_notebook(seed=548, float_precision=3, max_columns=15, max_rows=25):
    """
    Configure notebook settings:
      - Disables warnings for cleaner output.
      - Sets pandas display options for better table formatting.
      - Returns a seed value for reproducibility.
    
    Parameters:
      seed (int): Random seed (default 548).
      float_precision (int): Number of decimal places for floats (default 3).
      max_columns (int): Maximum number of columns to display (default 15).
      max_rows (int): Maximum number of rows to display (default 25).

    Returns:
      int: The provided seed.
    """
    # Disable all warnings
    warnings.filterwarnings("ignore")
    
    # Set pandas display options for nicer output
    pd.options.display.float_format = f"{{:,.{float_precision}f}}".format
    pd.set_option("display.max_columns", max_columns)
    pd.set_option("display.max_rows", max_rows)

    # Set seeds for reproducibility in numpy and the standard random module
    np.random.seed(seed)
    random.seed(seed)
    
    return seed

# Apply configuration and set random seeds for reproducibility
seed = configure_notebook()


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


trn_file_path = "/kaggle/input/playground-series-s5e4/train.csv"  # Replace with your CSV file path
trn_df = load_csv_to_dataframe(trn_file_path, ignore_fields=['id'])

test_file_path = "/kaggle/input/playground-series-s5e4/test.csv"  # Replace with your CSV file path
tst_df = load_csv_to_dataframe(test_file_path, ignore_fields=['id'])

sample_file_path = "/kaggle/input/playground-series-s5e4/sample_submission.csv"  # Replace with your CSV file path
sub_df = load_csv_to_dataframe(sample_file_path)

org_file_path = "/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv"  # Replace with your CSV file path
org_df = load_csv_to_dataframe(org_file_path, ignore_fields=['id'])


trn_df.info()


org_df.info()


target = 'Listening_Time_minutes'
org_df = org_df.dropna(subset=[target]).drop_duplicates()
trn_df = pd.concat([trn_df, org_df], axis=0, ignore_index=True)


import pandas as pd
import numpy as np
from IPython.display import display  # for nicer display in notebooks

def eda_summary(df):
    # 1. Display the first few rows
    print("======== First 5 Rows ========")
    display(df.head())
    
    # 2. DataFrame information (data types, non-null counts, etc.)
    print("\n======== DataFrame Info ========")
    df.info()
    
    # 3. Descriptive statistics for numeric columns
    print("\n======== Descriptive Statistics (Numeric Columns) ========")
    display(df.describe())
    
    # 4. Descriptive statistics for categorical columns (if any)
    categorical_df = df.select_dtypes(include=['object', 'category'])
    print("\n======== Descriptive Statistics (Categorical Columns) ========")
    if not categorical_df.empty:
        display(categorical_df.describe())
    else:
        print("No categorical columns found.")
    
    # 5. Missing values summary
    print("\n======== Missing Values Summary ========")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_summary = pd.DataFrame({
        "Missing Count": missing,
        "Percentage": missing_percent
    })
    display(missing_summary)
    
    # 6. Count of duplicated rows
    print("\n======== Duplicated Rows ========")
    print(f"Total duplicated rows: {df.duplicated().sum()}")
    
    # 7. Count of each data type
    print("\n======== Data Types Count ========")
    display(df.dtypes.value_counts())
    
    # 8. Correlation matrix for numeric variables (if more than one exists)
    numeric_cols = df.select_dtypes(include=[np.number])
    if numeric_cols.shape[1] > 1:
        print("\n======== Correlation Matrix (Numeric Columns) ========")
        display(numeric_cols.corr())
    else:
        print("\n======== Correlation Matrix ========")
        print("Not enough numeric columns to compute correlation.")
    
    # 9. Value counts for categorical variables with low cardinality
    print("\n======== Value Counts for Categorical Columns (Low Cardinality) ========")
    if not categorical_df.empty:
        for col in categorical_df.columns:
            if df[col].nunique() <= 20:
                print(f"\nValue Counts for '{col}':")
                display(df[col].value_counts())
    else:
        print("No categorical columns found.")


eda_summary(trn_df)


eda_summary(tst_df)


# Exploring the data in more detail...
# Creating some features...
trn_df['Episode_Title_Number'] = trn_df['Episode_Title'].str.split().str[-1].astype('int')
tst_df['Episode_Title_Number'] = tst_df['Episode_Title'].str.split().str[-1].astype('int')

trn_df['Host_Plus_Guest_Popularity'] = trn_df['Host_Popularity_percentage'] + trn_df['Guest_Popularity_percentage']
tst_df['Host_Plus_Guest_Popularity'] = tst_df['Host_Popularity_percentage'] + tst_df['Guest_Popularity_percentage']


trn_df.head()


trn_df[trn_df.Podcast_Name == 'Mystery Matters'].describe()


import matplotlib.pyplot as plt
plt.hist(trn_df[trn_df.Podcast_Name == 'Mystery Matters']['Episode_Title_Number'])


tst_df[tst_df.Podcast_Name == 'Mystery Matters'].describe()


trn_df.head()


import pandas as pd
import matplotlib.pyplot as plt

# Define the correct order for the days of the week
week_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Group your data and compute the mean of Listening_Time_minutes
pivot_summary = trn_df.groupby('Publication_Day')['Listening_Time_minutes'].mean().reset_index()

# Convert 'Publication_Day' into an ordered categorical type using the defined order
pivot_summary['Publication_Day'] = pd.Categorical(pivot_summary['Publication_Day'], categories=week_order, ordered=True)

# Sort the DataFrame by the ordered categorical column
pivot_summary = pivot_summary.sort_values('Publication_Day')

# Plot the bar chart with rotated x-axis labels and a specified y-axis minimum
plt.bar(pivot_summary['Publication_Day'], pivot_summary['Listening_Time_minutes'])
plt.xlabel('Publication Day')
plt.ylabel('Average Listening Time (minutes)')
plt.title('Average Listening Time by Publication Day')
plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
plt.ylim(bottom=43)       # Set the minimum value for the y-axis to 0
plt.ylim(top=47) 
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Define the correct order for the days of the week
order = ['Morning', 'Afternoon', 'Evening', 'Night']

# Group your data and compute the mean of Listening_Time_minutes
pivot_summary = trn_df.groupby('Publication_Time')['Listening_Time_minutes'].mean().reset_index()

# Convert 'Publication_Day' into an ordered categorical type using the defined order
pivot_summary['Publication_Time'] = pd.Categorical(pivot_summary['Publication_Time'], categories=order, ordered=True)

# Sort the DataFrame by the ordered categorical column
pivot_summary = pivot_summary.sort_values('Publication_Time')

# Plot the bar chart with rotated x-axis labels and a specified y-axis minimum
plt.bar(pivot_summary['Publication_Time'], pivot_summary['Listening_Time_minutes'])
plt.xlabel('Publication_Time')
plt.ylabel('Average Listening Time (minutes)')
plt.title('Average Listening Time by Publication Time')
plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
plt.ylim(bottom=40)       # Set the minimum value for the y-axis to 0
plt.ylim(top=47) 
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Define the correct order for the days of the week
order = list(trn_df['Genre'].unique())

# Group your data and compute the mean of Listening_Time_minutes
pivot_summary = trn_df.groupby('Genre')['Listening_Time_minutes'].mean().reset_index()

# Convert 'Publication_Day' into an ordered categorical type using the defined order
pivot_summary['Genre'] = pd.Categorical(pivot_summary['Genre'], categories=order, ordered=True)

# Sort the DataFrame by the ordered categorical column
pivot_summary = pivot_summary.sort_values('Genre')

# Plot the bar chart with rotated x-axis labels and a specified y-axis minimum
plt.bar(pivot_summary['Genre'], pivot_summary['Listening_Time_minutes'])
plt.xlabel('Genre')
plt.ylabel('Average Listening Time (minutes)')
plt.title('Average Listening Time by Genre')
plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
plt.ylim(bottom=40)       # Set the minimum value for the y-axis to 0
plt.ylim(top=47) 
plt.show()


from sklearn.impute import SimpleImputer

def impute_missing_values(train_df, test_df, target_column):
    """
    Impute missing values for categorical and numerical columns in the training and test DataFrames.

    Parameters:
    train_df (pandas.DataFrame): The training DataFrame with missing values.
    test_df (pandas.DataFrame): The testing DataFrame with missing values.
    target_column (str): The name of the target column.

    Returns:
    tuple: A tuple containing the training and testing DataFrames with imputed values.
    """
    # Create copies of the DataFrames to avoid modifying the originals
    train_imputed = train_df.copy()
    test_imputed = test_df.copy()
    
    # Separate categorical and numerical columns, excluding the target column
    categorical_columns = train_imputed.select_dtypes(include=['object']).columns.difference([target_column])
    numerical_columns = train_imputed.select_dtypes(include=['number']).columns.difference([target_column])
    
    # Impute missing values for categorical columns if they exist
    if not categorical_columns.empty:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        train_imputed[categorical_columns] = cat_imputer.fit_transform(train_imputed[categorical_columns])
        test_imputed[categorical_columns] = cat_imputer.transform(test_imputed[categorical_columns])
    
    # Impute missing values for numerical columns if they exist
    if not numerical_columns.empty:
        num_imputer = SimpleImputer(strategy='mean')
        train_imputed[numerical_columns] = num_imputer.fit_transform(train_imputed[numerical_columns])
        test_imputed[numerical_columns] = num_imputer.transform(test_imputed[numerical_columns])
    
    return train_imputed, test_imputed


# Utilize the imputation function...
train_imputed, test_imputed = impute_missing_values(trn_df, tst_df, 'Listening_Time_minutes')


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


# Encoding the train and test datasets.
categorical_fields = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
trn_encoded, tst_encoded = label_encode_datasets(train_imputed, test_imputed, categorical_fields)


trn_encoded


import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scale_dataframes(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target_col: str, 
    method: str = "zscore"
):
    """
    Scale numeric columns in train and test dataframes using a chosen method.
    The target column is excluded from scaling.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe.
    test_df : pd.DataFrame
        Test dataframe.
    target_col : str
        The name of the target column that should not be scaled.
    method : str, optional
        Scaling method: "zscore", "minmax", or "robust". 
        Default is "zscore".
    
    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        A tuple of scaled train_df and test_df dataframes with the target
        column unchanged.
    """

    # Create a copy of the dataframes to avoid modifying originals
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    # Select columns to scale (exclude target column)
    # - Only consider numeric columns
    numeric_cols = train_scaled.select_dtypes(include=["number"]).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]

    # Choose the scaler based on user input
    if method == "zscore":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("Unknown method. Choose from 'zscore', 'minmax', or 'robust'.")

    # Fit the scaler on the training data (numeric columns)
    scaler.fit(train_scaled[numeric_cols])

    # Transform both train and test data
    train_scaled[numeric_cols] = scaler.transform(train_scaled[numeric_cols])
    test_scaled[numeric_cols] = scaler.transform(test_scaled[numeric_cols])

    return train_scaled, test_scaled


train_scaled, test_scaled = scale_dataframes(trn_encoded, tst_encoded, 'Listening_Time_minutes', 'zscore')


train_scaled


import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

def baseline_lasso_regressor_df_cv_ensemble(
    train_df,
    target_col,
    feature_cols,
    test_df=None,
    n_splits=5,
    random_state=42,
    alpha=1.0  # Regularization strength for Lasso
):
    """
    Performs K-fold cross-validation on the training DataFrame, printing 
    the mean squared error (MSE) and root mean squared error (RMSE) for each fold,
    and their mean values across all folds.
    Optionally ensembles test predictions by averaging predictions from each fold.

    Parameters:
    -----------
    train_df : pd.DataFrame
        DataFrame containing the training data (with the target column).
    target_col : str
        Name of the target variable column in `train_df`.
    feature_cols : list of str
        List of feature column names.
    test_df : pd.DataFrame, optional
        If provided, ensemble predictions for this test set are generated by
        averaging predictions from each fold model.
    n_splits : int, default=5
        Number of folds for cross-validation.
    random_state : int, default=42
        Random seed for reproducibility.
    alpha : float, default=1.0
        Regularization strength for the Lasso regressor.

    Returns:
    --------
    If test_df is None:
        mean_cv_mse : float
            Mean MSE across all folds of cross-validation.
        mean_cv_rmse : float
            Mean RMSE across all folds of cross-validation.
    Else:
        mean_cv_mse : float
            Mean MSE across all folds of cross-validation.
        mean_cv_rmse : float
            Mean RMSE across all folds of cross-validation.
        avg_test_predictions : np.ndarray
            Averaged predicted values on test_df, shape = (num_test_samples,).
    """
    X = train_df[feature_cols]
    y = train_df[target_col]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_mses = []
    cv_rmses = []

    # If test_df is provided, prepare space to store predictions for each fold
    if test_df is not None:
        X_test = test_df[feature_cols]
        test_predictions = np.zeros((len(X_test), n_splits))
    else:
        test_predictions = None

    # K-fold cross-validation loop
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train Lasso regressor on this fold
        model = Lasso(alpha=alpha, max_iter=1000)
        model.fit(X_train, y_train)

        # Measure MSE and compute RMSE on the validation fold
        y_pred_val = model.predict(X_val)
        fold_mse = mean_squared_error(y_val, y_pred_val)
        fold_rmse = np.sqrt(fold_mse)
        cv_mses.append(fold_mse)
        cv_rmses.append(fold_rmse)

        # Print the per-fold MSE and RMSE
        print(f"Fold {fold_idx + 1} MSE: {fold_mse:.4f}, RMSE: {fold_rmse:.4f}")

        # If test_df is provided, predict for test set
        if test_df is not None:
            test_predictions[:, fold_idx] = model.predict(X_test)

    # Calculate mean MSE and RMSE across folds
    mean_cv_mse = np.mean(cv_mses)
    mean_cv_rmse = np.mean(cv_rmses)
    print(f"Mean CV RMSE across {n_splits} folds: {mean_cv_rmse:.4f}")

    # Return results
    if test_df is None:
        return mean_cv_mse, mean_cv_rmse
    else:
        # Average predictions across folds for the test set
        avg_test_predictions = np.mean(test_predictions, axis=1)
        return mean_cv_mse, mean_cv_rmse, avg_test_predictions



target = 'Listening_Time_minutes'
feature_cols = [col for col in train_scaled.columns if target not in col]
print(feature_cols)

feature_cols = ['Podcast_Name', 'Episode_Length_minutes', 'Genre', 'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'Episode_Title_Number', 'Host_Plus_Guest_Popularity']


mean_cv_mse, mean_cv_rmse, avg_test_predictions = baseline_lasso_regressor_df_cv_ensemble(train_scaled, target, feature_cols, test_scaled, n_splits = 5, random_state=468)


sub_df['Listening_Time_minutes'] = avg_test_predictions
sub_df.to_csv('lr_submission.csv', index=False)
display(sub_df.head())


def train_model(train_df, test_df, target_column, feature_columns=None, model_type="xgboost", param_file=None, 
                n_splits=5, categorical_features=None, use_target_encoding=False):
    """
    Train a machine learning regressor using the provided training and test datasets with K-Fold cross-validation,
    utilizing GPU support where applicable, early stopping where supported, and return predicted target values.
    
    Optionally, target encoding for categorical features is performed within each fold so that no target leakage occurs.
    When target encoding is enabled, additional features with suffix "_te" are created, preserving the original
    categorical columns.
    
    Parameters:
        train_df (pandas.DataFrame): Training DataFrame.
        test_df (pandas.DataFrame): Testing DataFrame.
        target_column (str): Name of the target column.
        feature_columns (list): List of feature column names to use. If None, all columns except target_column are used.
        model_type (str): "xgboost", "catboost", "lgbm", or "hgb".
        param_file (dict): Dictionary of hyperparameters for the model.
        n_splits (int): Number of folds for cross-validation.
        categorical_features (list): List of categorical column names. If None, they are inferred from the features.
        use_target_encoding (bool): If True, add mean target encoded features for categorical columns during each fold.
    
    Returns:
        tuple: (test_predictions, oof_predictions, model)
            - test_predictions: Final predicted target values for the test set.
            - oof_predictions: Out-of-fold predicted target values for the training set.
            - model: The final fitted model from the last fold.
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    import numpy as np

    # Determine feature set: use specified feature_columns if provided, otherwise use all except target_column
    if feature_columns is not None:
        # Ensure target_column is not included in feature_columns
        feature_columns = [col for col in feature_columns if col != target_column]
        X = train_df[feature_columns].copy()
        X_test = test_df[feature_columns].copy()
    else:
        X = train_df.drop(columns=[target_column])
        X_test = test_df.copy()

    # Extract target values
    y = train_df[target_column]

    # Infer categorical features from the selected features if not provided
    if categorical_features is None:
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Import model-specific regressor classes
    if model_type == "xgboost":
        from xgboost import XGBRegressor as Model
    elif model_type == "catboost":
        from catboost import CatBoostRegressor as Model
    elif model_type == "lgbm":
        from lightgbm import LGBMRegressor as Model
    elif model_type == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor as Model
    else:
        raise ValueError("Unsupported model_type. Choose from 'xgboost', 'catboost', 'lgbm', or 'hgb'.")

    # Set default parameters if none are provided
    if param_file is None:
        if model_type == "xgboost":
            param_file = {
                'n_estimators': 10000,
                'learning_rate': 0.01,
                'max_depth': 12,
                'subsample': 0.75,
                'colsample_bytree': 0.80,
                'min_child_weight': 120,
                'alpha': 1.5,
                'tree_method': 'gpu_hist',
                'eval_metric': 'rmse',
                'random_state': 42
            }
        elif model_type == "catboost":
            param_file = {
                'iterations': 10000,
                'learning_rate': 0.01,
                'depth': 12,
                'task_type': 'GPU',
                'random_seed': 42,
                'verbose': False,
                'loss_function': 'RMSE'
            }
        elif model_type == "lgbm":
            param_file = {
                'n_estimators': 2048,
                'learning_rate': 0.01,
                'max_depth': -1,
                'subsample': 0.80,
                'colsample_bytree': 0.50,
                'min_child_weight': 80,
                'device': 'gpu',
                'random_state': 42
            }
        elif model_type == "hgb":
            param_file = {
                'max_iter': 1000,
                'learning_rate': 0.01,
                'max_leaf_nodes': 31,
                'early_stopping': True,
                'random_state': 42
            }

    # Initialize the model with the specified parameters
    model = Model(**param_file)

    # Set up K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Lists to store RMSE and test-set predictions
    rmse_scores = []
    test_pred_vals = []  # To collect test-set predictions from each fold
    oof_predictions = None  # Will be initialized upon first fold

    # Early stopping rounds to be used for models that support it
    early_stopping_rounds = 100

    for train_index, val_index in kf.split(X):
        # Create fold-specific training and validation data
        X_train, X_val = X.iloc[train_index].copy(), X.iloc[val_index].copy()
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        X_test_fold = X_test.copy()

        # --- PROCESS CATEGORICAL FEATURES ---
        # If mean target encoding is enabled, create additional features with target encoding.
        if use_target_encoding and categorical_features:
            for col in categorical_features:
                # Use filled values to ensure consistency
                filled_train = X_train[col].fillna('Missing')
                filled_val = X_val[col].fillna('Missing')
                filled_test = X_test_fold[col].fillna('Missing')
                # Compute the mapping from the training fold (using only training data)
                mapping = y_train.groupby(filled_train).mean()
                # Create additional features with the mean target encoding.
                new_feature = col + '_te'
                X_train[new_feature] = filled_train.map(mapping)
                X_val[new_feature] = filled_val.map(mapping)
                X_test_fold[new_feature] = filled_test.map(mapping)
                # For unseen categories, fill with the overall training target mean
                overall_mean = y_train.mean()
                X_val[new_feature].fillna(overall_mean, inplace=True)
                X_test_fold[new_feature].fillna(overall_mean, inplace=True)
        else:
            # Process categorical features based on model type while preserving the original columns.
            if model_type == "catboost":
                # CatBoost handles categorical features internally if passed as strings.
                X_train[categorical_features] = X_train[categorical_features].fillna('Missing').astype(str)
                X_val[categorical_features] = X_val[categorical_features].fillna('Missing').astype(str)
                X_test_fold[categorical_features] = X_test_fold[categorical_features].fillna('Missing').astype(str)
            elif model_type in ("xgboost", "lgbm", "hgb"):
                from sklearn.preprocessing import OrdinalEncoder
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                X_train[categorical_features] = encoder.fit_transform(
                    X_train[categorical_features].fillna('Missing').astype(str)
                )
                X_val[categorical_features] = encoder.transform(
                    X_val[categorical_features].fillna('Missing').astype(str)
                )
                X_test_fold[categorical_features] = encoder.transform(
                    X_test_fold[categorical_features].fillna('Missing').astype(str)
                )

        # --- MODEL TRAINING WITH EARLY STOPPING WHERE SUPPORTED ---
        if model_type == "catboost":
            model.fit(
                X_train, y_train,
                cat_features=categorical_features,
                eval_set=(X_val, y_val),
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        elif model_type in ("xgboost", "lgbm"):
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        else:  # For models like HistGradientBoosting that may not support early stopping via fit()
            model.fit(X_train, y_train)

        # --- EVALUATION ON VALIDATION SET ---
        y_val_pred = model.predict(X_val)
        fold_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        rmse_scores.append(fold_rmse)
        print(f"Fold RMSE: {fold_rmse:.4f}")

        # Initialize oof_predictions array on first fold
        if oof_predictions is None:
            oof_predictions = np.zeros(len(train_df))
        oof_predictions[val_index] = y_val_pred

        # --- PREDICTION ON TEST SET ---
        # Remove target column if present by mistake
        if target_column in X_test_fold.columns:
            X_test_fold = X_test_fold.drop(columns=[target_column], errors='ignore')
        test_pred_vals.append(model.predict(X_test_fold))

    # Calculate and print average RMSE across folds
    avg_rmse = np.mean(rmse_scores)
    print(f"Average RMSE across {n_splits} folds: {avg_rmse:.4f}")

    # Average the test-set predictions from each fold
    test_pred_vals = np.array(test_pred_vals)
    mean_test_predictions = np.mean(test_pred_vals, axis=0)

    # Optionally, print the feature names used in the last training fold
    print("Features used in the last fold:")
    print(X_train.columns)

    return mean_test_predictions, oof_predictions, model



train_scaled.head()


categorical_features = ['Publication_Day', 'Publication_Time', 'Episode_Sentiment', 'Genre']

xgb_mean_test_predictions, xgb_oof_predictions, xgb_model = train_model(train_scaled, 
                                                                        test_scaled, 
                                                                        target, 
                                                                        feature_columns=None, 
                                                                        model_type="xgboost", 
                                                                        param_file=None, 
                                                                        n_splits=10, 
                                                                        categorical_features=categorical_features, 
                                                                        use_target_encoding=True
                                                                       )





import seaborn as sns
feature_cols = ['Podcast_Name', 'Episode_Title', 'Episode_Length_minutes', 'Genre',
       'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time',
       'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment',
       'Episode_Title_Number', 'Host_Plus_Guest_Popularity',
       'Publication_Day_te', 'Publication_Time_te', 'Episode_Sentiment_te',
       'Genre_te']
feature_importance = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(5, 4))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.show()


# Average RMSE across 5 folds: 12.7530
# Average RMSE across 5 folds: 12.7749


sub_df['Listening_Time_minutes'] = xgb_mean_test_predictions
sub_df.to_csv('xgbt_submission_orig.csv', index=False)
display(sub_df.head())


#xgb_mean_test_predictions, xgb_oof_predictions, xgb_model = train_model(train_scaled, test_scaled, target, feature_columns=None, model_type="xgboost", param_file=None, n_splits=5, categorical_features=None, use_target_encoding=False)


#cb_mean_test_predictions, cb_oof_predictions, cb_model = train_model(train_scaled, test_scaled, target, feature_columns=None, model_type="catboost", param_file=None, n_splits=5, categorical_features=None)


#sub_df['Listening_Time_minutes'] = cb_mean_test_predictions
#sub_df.to_csv('cb_submission.csv', index=False)
#display(sub_df.head())


#hgh_mean_test_predictions, hgh_oof_predictions, hgh_model = train_model(train_scaled, test_scaled, target, feature_columns=None, model_type="hgb", param_file=None, n_splits=5, categorical_features=None)


#sub_df['Listening_Time_minutes'] = hgh_mean_test_predictions
#sub_df.to_csv('hgh_submission.csv', index=False)
#display(sub_df.head())

