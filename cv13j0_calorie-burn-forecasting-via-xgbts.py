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


# Importing the necessary libraries
import warnings  # Provides a way to control the display of warning messages (e.g., filter out deprecation warnings)
import pandas as pd  # Powerful data manipulation and analysis library; offers DataFrame and Series data structures
import numpy as np  # Fundamental package for numerical computing in Python; provides N-dimensional array objects and routines
import random  # Pythonâ€™s built-in module for generating pseudo-random numbers and selecting random elements
from sklearn.preprocessing import LabelEncoder  # Utility from scikit-learn to convert categorical labels into numeric form (e.g., â€œredâ€� â†’ 0, â€œblueâ€� â†’ 1)
from IPython.display import display  # for nicer display in notebooks


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


trn_file_path = "/kaggle/input/playground-series-s5e5/train.csv"  # Replace with your CSV file path
trn_df = load_csv_to_dataframe(trn_file_path, ignore_fields=['id'])

test_file_path = "/kaggle/input/playground-series-s5e5/test.csv"  # Replace with your CSV file path
tst_df = load_csv_to_dataframe(test_file_path, ignore_fields=['id'])

sample_file_path = "/kaggle/input/playground-series-s5e5/sample_submission.csv"  # Replace with your CSV file path
sub_df = load_csv_to_dataframe(sample_file_path)

orig_file_path = "/kaggle/input/calories-burnt-prediction/calories.csv"  # Replace with your CSV file path
org_df = load_csv_to_dataframe(orig_file_path, ignore_fields=['User_ID'])
org_df = org_df.rename({"Gender":"Sex"},axis=1)

trn_df = pd.concat([trn_df, org_df], axis=0, ignore_index=True)


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


numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new

# trn_df = add_feature_cross_terms(trn_df, numerical_features)
# tst_df = add_feature_cross_terms(tst_df, numerical_features)


def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        subset (list or str, optional): Column label or labels to consider for identifying duplicates. 
                                        If None, all columns are used.
        keep (str, optional): Which duplicates to keep:
            - 'first' (default): Keep the first occurrence.
            - 'last': Keep the last occurrence.
            - False: Drop all duplicates.

    Returns:
        pd.DataFrame: A DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

trn_df = remove_duplicates(trn_df)


import pandas as pd

def find_duplicates(train_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    target_col: str) -> pd.DataFrame:
    """
    Identify duplicate featureâ€�rows between train and test DataFrames,
    ignoring the specified target column.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training set, with target column.
    test_df : pd.DataFrame
        Test set, with target column.
    target_col : str
        Name of the target column to exclude when comparing.

    Returns
    -------
    pd.DataFrame
        A DataFrame of duplicates with:
         - all feature columns,
         - train_index: the index in train_df,
         - test_index: the index in test_df.
    """
    # 1. Determine feature columns (everything except the target)
    features = [c for c in train_df.columns if c != target_col]

    # 2. Reset index so we can keep track of original row numbers
    train_feats = train_df[features].reset_index().rename(columns={'index': 'train_index'})
    test_feats  = test_df[features].reset_index().rename(columns={'index': 'test_index'})

    # 3. Inner join on all feature columns to find exact matches
    duplicates = pd.merge(
        train_feats,
        test_feats,
        on=features,
        how='inner'
    )

    return duplicates


# assume train_df and test_df both have a column 'y' as the target
dupes = find_duplicates(trn_df, tst_df, target_col='Calories')
if not dupes.empty:
    print(f"Found {len(dupes)} duplicate feature-rows:")
    print(dupes.head())
else:
    print("No duplicates found.")


dupes.head()


tst_df.iloc[dupes.test_index]


trn_df['Calories'] = np.log1p(trn_df['Calories'])


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
categorical_fields = ['Sex']
trn_encoded, tst_encoded = label_encode_datasets(trn_df, tst_df, categorical_fields)


target = 'Calories'
feature_cols = [col for col in trn_encoded.columns if target not in col]
print(feature_cols)


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
                'n_estimators': 2048,
                'learning_rate': 0.01,
                'max_depth': 10,
                'subsample': 0.80,
                'colsample_bytree': 0.80,
                'min_child_weight': 60,
                'alpha': 1.0,
                'tree_method': 'gpu_hist',
                'eval_metric': 'rmse',
                'random_state': 42
            }
        elif model_type == "catboost":
            param_file = {
                'iterations': 2048,
                'learning_rate': 0.01,
                'depth': 10,
                'task_type': 'GPU',
                'random_seed': 42,
                'verbose': False,
                'loss_function': 'RMSE'
            }
        elif model_type == "lgbm":
            param_file = {
                'n_estimators': 2048,
                'learning_rate': 0.01,
                'max_depth': 10,
                'subsample': 0.80,
                'colsample_bytree': 0.80,
                'min_child_weight': 80,
                'device': 'gpu',
                'random_state': 42,
                'verbosity':-1,
                'silent':True
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
    early_stopping_rounds = 250

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
        elif model_type in ("xgboost"):
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



categorical_features = ['Sex']

xgb_mean_test_predictions, xgb_oof_predictions, xgb_model = train_model(trn_encoded, 
                                                                        tst_encoded, 
                                                                        target, 
                                                                        feature_columns=None, 
                                                                        model_type="xgboost", 
                                                                        param_file=None, 
                                                                        n_splits=20, 
                                                                        categorical_features=categorical_features, 
                                                                        use_target_encoding=True
                                                                       )


# Store the predictions in the Submission dataframe...
sub_df['Calories'] = np.expm1(xgb_mean_test_predictions)
sub_df['Calories'] = np.clip(sub_df['Calories'],1,314)


# Check some of the predictions on the duplicated values...
sub_df['Calories'].iloc[dupes.test_index] 


# Overwrite the predictions by true values from training set...
sub_df['Calories'].iloc[dupes.test_index] = np.expm1(trn_df['Calories'].iloc[dupes.train_index])

sub_df.to_csv('xgbt_submission.csv', index=False)
display(sub_df.head())


# Check some of the predictions on the duplicated values, after the replacement...
sub_df['Calories'].iloc[dupes.test_index] 


# Average RMSE across 20 folds: 0.0595


categorical_features = ['Sex']

cbt_mean_test_predictions, cbt_oof_predictions, cbt_model = train_model(trn_encoded, 
                                                                        tst_encoded, 
                                                                        target, 
                                                                        feature_columns=None, 
                                                                        model_type="catboost", 
                                                                        param_file=None, 
                                                                        n_splits=20, 
                                                                        categorical_features=categorical_features, 
                                                                        use_target_encoding=True
                                                                       )


sub_df['Calories'] = np.expm1(cbt_mean_test_predictions)
sub_df['Calories'] = np.clip(sub_df['Calories'],1,314)

# Overwrite the predictions by true values from training set...
sub_df['Calories'].iloc[dupes.test_index] = np.expm1(trn_df['Calories'].iloc[dupes.train_index])# Overwrite the predictions by true values from training set...

sub_df.to_csv('cbt_submission.csv', index=False)
display(sub_df.head())


# categorical_features = ['Sex']

# lgb_mean_test_predictions, lgb_oof_predictions, lgb_model = train_model(trn_encoded, 
#                                                                         tst_encoded, 
#                                                                         target, 
#                                                                         feature_columns=None, 
#                                                                         model_type="lgbm", 
#                                                                         param_file=None, 
#                                                                         n_splits=20, 
#                                                                         categorical_features=categorical_features, 
#                                                                         use_target_encoding=False
#                                                                        )


# sub_df['Calories'] = np.expm1(lgb_mean_test_predictions)
# sub_df['Calories'] = np.clip(sub_df['Calories'],1,314)
# sub_df.to_csv('lgb_submission.csv', index=False)
# display(sub_df.head())


# categorical_features = ['Sex']

# hgb_mean_test_predictions, hgb_oof_predictions, hgb_model = train_model(trn_encoded, 
#                                                                         tst_encoded, 
#                                                                         target, 
#                                                                         feature_columns=None, 
#                                                                         model_type="hgb", 
#                                                                         param_file=None, 
#                                                                         n_splits=5, 
#                                                                         categorical_features=categorical_features, 
#                                                                         use_target_encoding=False
#                                                                        )


# sub_df['Calories'] = np.expm1(hgb_mean_test_predictions)
# sub_df['Calories'] = np.clip(sub_df['Calories'],1,314)
# sub_df.to_csv('hgb_submission.csv', index=False)
# display(sub_df.head())


# xgb_ypred = np.expm1(xgb_mean_test_predictions)
# cbt_ypred = np.expm1(cbt_mean_test_predictions)
# lgb_ypred = np.expm1(lgb_mean_test_predictions)
# #hgb_ypred = np.expm1(hgb_mean_test_predictions)
# a = 0
# b = 1
# c = 0
# #d = 0
# sub_df['Calories'] = a * xgb_ypred + b * cbt_ypred + c * lgb_ypred 
# #sub_df['Calories'] = a * xgb_ypred + b * cbt_ypred + c * lgb_ypred + d * hgb_ypred
# sub_df['Calories'] = np.clip(sub_df['Calories'],1,314)
# sub_df.to_csv('blend_submission.csv', index=False)
# display(sub_df.head())


# import seaborn as sns
# import matplotlib.pyplot as plt

# feature_cols = feature_cols
# feature_importance = xgb_model.feature_importances_
# importance_df = pd.DataFrame({
#     'Feature': feature_cols,
#     'Importance': feature_importance
# }).sort_values(by='Importance', ascending=False)

# plt.figure(figsize=(5, 3))
# sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
# plt.show()

