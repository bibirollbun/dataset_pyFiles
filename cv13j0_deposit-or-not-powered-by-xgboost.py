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

import pandas as pd  # Powerful data manipulation and analysis library; offers DataFrame and Series data structures
import numpy as np  # Fundamental package for numerical computing in Python; provides N-dimensional array objects and routines
import random  # Pythonâ€™s built-in module for generating pseudo-random numbers and selecting random elements
import warnings  # Provides a way to control the display of warning messages (e.g., filter out deprecation warnings)

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


def load_csv_to_dataframe(file_path, ignore_fields=[], delimiter = ','):
    """
    Load a CSV file into a pandas DataFrame, optionally ignoring specified fields.

    Parameters:
    file_path (str): The file path of the CSV file to be loaded.
    ignore_fields (list): A list of field names to be ignored when loading the CSV.

    Returns:
    pandas.DataFrame: A DataFrame containing the data from the CSV file, excluding the ignored fields.
    """
    # Read the CSV file from the given file path using pandas
    df = pd.read_csv(file_path, delimiter = delimiter)
    
    # Drop the fields that need to be ignored, if they exist in the DataFrame
    df = df.drop(columns=ignore_fields, errors='ignore')
    
    # Return the resulting DataFrame
    return df


trn_file_path = "/kaggle/input/playground-series-s5e8/train.csv"  # Replace with your CSV file path
trn_df = load_csv_to_dataframe(trn_file_path, ignore_fields=['id'])

test_file_path = "/kaggle/input/playground-series-s5e8/test.csv"  # Replace with your CSV file path
tst_df = load_csv_to_dataframe(test_file_path, ignore_fields=['id'])

sample_file_path = "/kaggle/input/playground-series-s5e8/sample_submission.csv"  # Replace with your CSV file path
sub_df = load_csv_to_dataframe(sample_file_path)

orig_file_path = "/kaggle/input/bank-marketing-dataset-full/bank-full.csv"  # Replace with your CSV file path
org_df = load_csv_to_dataframe(orig_file_path, ignore_fields=['User_ID'], delimiter = ";")
org_df['y'] = org_df.y.map({'yes':1,'no':0})
#org_df['id'] = (np.arange(len(org_df))+1e6).astype('int')
#org_df = org_df.set_index('id')

trn_df = pd.concat([trn_df, org_df], axis=0, ignore_index=True)


trn_df.head()


org_df.head()





def eda_summary(df):
    # 1. Display the first few rows
    print("======== First 5 Rows ========")
    display(df.head().T)
    
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


target = 'y'
feature_cols = [col for col in trn_df.columns if target not in col]
categorical_fields = ['job','marital','education','default','housing','loan','contact','month','poutcome']
print(feature_cols)


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
trn_encoded, tst_encoded = label_encode_datasets(trn_df, tst_df, categorical_fields)


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


train_scaled, test_scaled = scale_dataframes(trn_encoded, tst_encoded, 'y', 'minmax')


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

def baseline_logistic_classifier_df_cv_ensemble(
    train_df,
    target_col,
    feature_cols,
    test_df=None,
    n_splits=5,
    random_state=42
):
    """
    Performs K-fold cross-validation on the training DataFrame, printing 
    the accuracy for each fold and the mean accuracy across all folds.
    Optionally ensembles test predictions by averaging fold predictions.

    Parameters:
    -----------
    train_df : pd.DataFrame
        DataFrame containing the training data (with the target column).
    target_col : str
        Name of the target variable column in `train_df`.
    feature_cols : list of str
        List of feature column names.
    test_df : pd.DataFrame, optional
        If provided, we generate ensemble predictions for this test set by
        averaging predictions from each fold model.
    n_splits : int, default=5
        Number of folds for cross-validation.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns:
    --------
    If test_df is None:
        mean_cv_accuracy : float
            Mean accuracy across all folds of cross-validation.
    Else:
        mean_cv_accuracy : float
            Mean accuracy across all folds of cross-validation.
        avg_test_predictions : np.ndarray
            Averaged predicted probabilities on test_df,
            shape = (num_test_samples, n_classes).
    """
    X = train_df[feature_cols]
    y = train_df[target_col]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_accuracies = []

    # If test_df is provided, prepare space to store predictions for each fold
    if test_df is not None:
        X_test = test_df[feature_cols]
        # Assuming binary classification -> shape: [num_test_samples, 2, n_splits]
        test_predictions = np.zeros((len(X_test), 2, n_splits))
    else:
        test_predictions = None

    # K-fold cross-validation loop
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train logistic regression on this fold
        model = LogisticRegression(penalty = 'l1', solver = 'liblinear', max_iter = 100, random_state=random_state)
        model.fit(X_train, y_train)

        # Measure accuracy on the validation fold
        y_pred_val = model.predict(X_val)
        fold_accuracy = accuracy_score(y_val, y_pred_val)
        cv_accuracies.append(fold_accuracy)

        # Print the per-fold accuracy
        print(f"Fold {fold_idx + 1} accuracy: {fold_accuracy:.4f}")

        # If test_df is provided, predict probabilities for test set
        if test_df is not None:
            test_pred_proba = model.predict_proba(X_test)
            test_predictions[:, :, fold_idx] = test_pred_proba

    # Calculate mean accuracy across folds
    mean_cv_accuracy = np.mean(cv_accuracies)

    # Print the mean cross-validation accuracy
    print(f"Mean CV accuracy across {n_splits} folds: {mean_cv_accuracy:.4f}")

    # Return results
    if test_df is None:
        return mean_cv_accuracy
    else:
        # Average predictions across folds for the test set
        avg_test_predictions = np.mean(test_predictions, axis=2)
        return mean_cv_accuracy, avg_test_predictions


mean_cv_accuracy, avg_test_predictions = baseline_logistic_classifier_df_cv_ensemble(train_scaled, target, feature_cols, test_scaled, n_splits = 5, random_state=468)


sub_df['y'] = 1 - avg_test_predictions
sub_df.to_csv('lr_submission.csv', index=False)
display(sub_df.head())


def train_model(train_df, test_df, target_column, feature_columns=None, model_type="xgboost", param_file=None, 
                n_splits=5, categorical_features=None):
    """
    Train a machine learning classifier using the provided training and test datasets with K-Fold cross-validation,
    utilizing GPU support where applicable, and return predicted probabilities.

    Parameters:
        train_df (pandas.DataFrame): Training DataFrame.
        test_df (pandas.DataFrame): Testing DataFrame.
        target_column (str): Name of the target column.
        feature_columns (list): List of feature column names to use. If None, all columns except target_column are used.
        model_type (str): "xgboost", "catboost", "lgbm", or "hgb".
        param_file (dict): Dictionary of hyperparameters for the model.
        n_splits (int): Number of folds for cross-validation.
        categorical_features (list): List of categorical column names. If None, they are inferred from the features.

    Returns:
        tuple: (test_probabilities, oof_predictions, model)
            - test_probabilities: Final predicted probabilities for the test set.
            - oof_predictions: Out-of-fold predicted probabilities for the training set.
            - model: The final fitted model from the last fold.
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
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

    # Import model-specific classifier classes
    if model_type == "xgboost":
        from xgboost import XGBClassifier as Model
    elif model_type == "catboost":
        from catboost import CatBoostClassifier as Model
    elif model_type == "lgbm":
        from lightgbm import LGBMClassifier as Model
    elif model_type == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier as Model
    else:
        raise ValueError("Unsupported model_type. Choose from 'xgboost', 'catboost', 'lgbm', or 'hgb'.")

    # Set default parameters if none are provided
    if param_file is None:
        if model_type == "xgboost":
            param_file = {
                'n_estimators': 4096,
                'learning_rate': 0.05,
                'max_depth': 12,
                'subsample': 0.90,
                
                'colsample_bytree': 0.80,
                'min_child_weight': 80,
                'alpha': 1.5,
                'tree_method': 'gpu_hist',
                'eval_metric':'auc',
                'random_state': 42
            }
        elif model_type == "catboost":
            param_file = {
                'iterations': 2048,
                'learning_rate': 0.03,
                'depth': 6,
                'task_type': 'GPU',
                'random_seed': 42,
                'verbose': False
            }
        elif model_type == "lgbm":
            param_file = {
                'n_estimators': 2048,
                'learning_rate': 0.03,
                'max_depth': -1,
                'subsample': 0.80,
                'colsample_bytree': 0.50,
                'min_child_weight': 80,
                'device': 'gpu',
                'random_state': 42
            }
        elif model_type == "hgb":
            param_file = {
                'max_iter': 200,
                'learning_rate': 0.03,
                'max_leaf_nodes': 31,
                'early_stopping': True,
                'random_state': 42
            }

    # Initialize the model with the specified parameters
    model = Model(**param_file)

    # Set up K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Lists to store metrics and test-set predictions
    accuracy_scores = []
    f1_scores = []
    auc_scores = []
    logloss_scores = []
    test_pred_probs = []  # To collect test-set probabilities from each fold
    oof_predictions = None  # Will be initialized upon first fold

    for train_index, val_index in kf.split(X):
        # Create fold-specific training and validation data
        X_train, X_val = X.iloc[train_index].copy(), X.iloc[val_index].copy()
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        X_test_fold = X_test.copy()

        # --- PROCESS CATEGORICAL FEATURES ---
        if model_type == "catboost":
            # CatBoost handles categorical features internally if passed as strings.
            X_train[categorical_features] = X_train[categorical_features].fillna('Missing').astype(str)
            X_val[categorical_features] = X_val[categorical_features].fillna('Missing').astype(str)
            X_test_fold[categorical_features] = X_test_fold[categorical_features].fillna('Missing').astype(str)
            model.fit(X_train, y_train, cat_features=categorical_features)
        else:
            # For other models, perform label encoding for categorical features.
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
            model.fit(X_train, y_train)

        # --- EVALUATION ON VALIDATION SET ---
        # Attempt to get predicted probabilities; if not available, fall back to class predictions.
        try:
            y_val_pred_proba = model.predict_proba(X_val)
        except AttributeError:
            y_val_pred_proba = None

        if y_val_pred_proba is not None:
            # Initialize oof_predictions array on first fold
            if oof_predictions is None:
                n_classes = y_val_pred_proba.shape[1]
                oof_predictions = np.zeros((len(train_df), n_classes))
            oof_predictions[val_index] = y_val_pred_proba
        else:
            # Fallback: use predicted classes (note: these are not probabilities)
            y_val_pred = model.predict(X_val)
            if oof_predictions is None:
                oof_predictions = np.empty((len(train_df),), dtype=object)
            oof_predictions[val_index] = y_val_pred

        # For computing metrics, use class predictions.
        y_val_pred_classes = model.predict(X_val)
        accuracy_scores.append(accuracy_score(y_val, y_val_pred_classes))
        f1_scores.append(f1_score(y_val, y_val_pred_classes, average='macro'))
        try:
            if y_val_pred_proba is not None:
                if len(np.unique(y)) == 2:
                    auc_scores.append(roc_auc_score(y_val, y_val_pred_proba[:, 1]))
                logloss_scores.append(log_loss(y_val, y_val_pred_proba))
        except Exception:
            pass

        # --- PREDICTION ON TEST SET ---
        if target_column in X_test_fold.columns:
            X_test_fold = X_test_fold.drop(columns=[target_column], errors='ignore')
        try:
            test_pred_probs.append(model.predict_proba(X_test_fold))
        except AttributeError:
            test_pred_probs.append(model.predict(X_test_fold))

    # Calculate and print average metrics
    avg_accuracy = np.mean(accuracy_scores)
    avg_f1 = np.mean(f1_scores)
    print("Model Performance Metrics (Cross-Validation):")
    print("..................")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average F1 Score (macro): {avg_f1:.4f}")
    if auc_scores:
        avg_auc = np.mean(auc_scores)
        print(f"Average ROC-AUC: {avg_auc:.4f}")
    if logloss_scores:
        avg_logloss = np.mean(logloss_scores)
        print(f"Average Log Loss: {avg_logloss:.4f}")

    # Average the test-set predicted probabilities from each fold
    test_pred_probs = np.array(test_pred_probs)
    mean_probs = np.mean(test_pred_probs, axis=0)

    # Optionally, print the feature names used in the last training fold
    print("Features used in the last fold:")
    print(X_train.columns)

    return mean_probs, oof_predictions, model


mean_probs, oof_predictions, model = train_model(train_scaled, test_scaled, feature_columns = feature_cols , target_column = target, model_type = "xgboost", param_file = None, n_splits = 5, categorical_features = None)


sub_df['y'] = 1 - mean_probs
sub_df.to_csv('xgbt_submission.csv', index=False)
display(sub_df.head())


# Average ROC-AUC: 0.9615
# Average ROC-AUC: 0.9600




