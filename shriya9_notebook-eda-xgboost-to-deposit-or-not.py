# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import random  # Python’s built-in module for generating pseudo-random numbers and selecting random elements
import warnings  # Provides a way to control the display of warning messages (e.g., filter out deprecation warnings)

from sklearn.preprocessing import LabelEncoder  # Utility from scikit-learn to convert categorical labels into numeric form (e.g., “red” → 0, “blue” → 1)
from IPython.display import display  # for nicer display in notebooks


def load_csv_to_dataframe(file_path, ignore_fields=[], delimiter = ','):
    """
    Load a CSV file into a pandas DataFrame, ignoring specified fields.

    Parameters:
    - file_path (str): Path to the CSV file.
    - ignore_fields (list): List of column names to ignore.
    - delimiter (str): Delimiter used in the CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing the data from the CSV file, excluding ignored fields.
    """
    df = pd.read_csv(file_path, delimiter=delimiter)
    # drop the specified columns if they exist in the DataFrame
    df = df.drop(columns=ignore_fields, errors='ignore')
    return df


train_file_path = '/kaggle/input/binary-classification-with-a-bank-dataset/train.csv' 
train_df = load_csv_to_dataframe(train_file_path, ignore_fields=['id'])
print("Train DataFrame shape:", train_df.shape)

test_file_path = '/kaggle/input/binary-classification-with-a-bank-dataset/test.csv' 
test_df = load_csv_to_dataframe(test_file_path, ignore_fields=['id'])
print("Test DataFrame shape:", test_df.shape)

sample_file_path = "/kaggle/input/playground-series-s5e8/sample_submission.csv"  # Replace with your CSV file path
sub_df = load_csv_to_dataframe(sample_file_path)

# orig_file_path = '/kaggle/input/binary-classification-with-a-bank-dataset/test.csv'
# orig_df = load_csv_to_dataframe(orig_file_path, ignore_fields=['User_ID'], delimiter=';')
# orig_df['y'] = orig_df['y'].map({'yes': 1, 'no': 0})  # Convert target variable to binary (1 for 'yes', 0 for 'no')

# print("Original DataFrame shape:", orig_df.shape)


def load_csv_to_dataframe(file_path, ignore_fields=[], delimiter = ','):
    """
    Load a CSV file into a pandas DataFrame, ignoring specified fields.

    Parameters:
    - file_path (str): Path to the CSV file.
    - ignore_fields (list): List of column names to ignore.
    - delimiter (str): Delimiter used in the CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing the data from the CSV file, excluding ignored fields.
    """
    df = pd.read_csv(file_path, delimiter=delimiter)
    # drop the specified columns if they exist in the DataFrame
    df = df.drop(columns=ignore_fields, errors='ignore')
    return df


train_df.head()


def eda_summary(df):
    #1. Display the info about the dataframe
    print("\n ===========DataFrame Info ===========")
    df.info()

    #2. Descriptive statistics for numeric columns
    print("\n ===========Descriptive Statistics for Numeric Columns ===========")
    display(df.describe().T)

    #3. Descriptive statistics for categorical columns (if any)
    # categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_df = df.select_dtypes(include=['object', 'category'])
    if not categorical_df.empty:
        print("\n ===========Descriptive Statistics for Categorical Columns ===========")
        display(categorical_df.describe().T)
    else:
        print("\nNo categorical columns found.")

    #4. Missing values summary
    print("\n ===========Missing Values Summary ===========")
    missing = df.isnull().sum()
    missing_percent = (missing/len(df))*100
    missing_summary = pd.DataFrame({'Missing Values': missing, 'Missing Percentage': missing_percent})
    display(missing_summary)

    #5. Duplicate row counts
    print("\n ===========Duplicate Row Counts ===========")
    print(f'Total duplicated rows: {df.duplicated().sum()}')


    #6. Count of each data type
    print("\n ===========Count of Each Data Type ===========")
    display(df.dtypes.value_counts())

    #7. Correlation matrix for numeric columns
    numeric_cols =df.select_dtypes(include=[np.number])
    if numeric_cols.shape[1] > 1:
        print("\n ===========Correlation Matrix for Numeric Columns ===========")
        display(numeric_cols.corr())
    else:
        print("\nNot enough numeric columns for correlation matrix.")
    
    #8. Value counts for categorical columns with low cardinality
    if not categorical_df.empty:
        print("\n ===========Value Counts for Categorical Columns (Low Cardinality) ===========")
        for col in categorical_df.columns:
            if df[col].nunique() <= 20:
                print(f"\n Value counts for '{col}':")
                display(df[col].value_counts())
            else:
                print(f"\n No categorical columns found")

    #9. Skewness and Kurtosis for numeric columns
    if not numeric_cols.empty:
        print("\n ===========Skewness and Kurtosis for Numeric Columns ===========")
        skew_kurt = pd.DataFrame({'Skewness': numeric_cols.skew(), 'Kurtosis': numeric_cols.kurt()})
        display(skew_kurt)
    else:
        print("\nNo numeric columns found.")
    print("\n ===========End of EDA Summary ===========")


eda_summary(train_df)


target = 'y'
feature_cols = [col for col in train_df.columns if target not in col]
categorical_fields = ['job','marital','education','default','housing','loan','contact','month','poutcome']
print(feature_cols)


def label_encode_columns(train_df, test_df, categorical_fields):
    """
    Label encode specified categorical columns in both training and testing DataFrames.

    Parameters:
    - train_df (pd.DataFrame): Training DataFrame.
    - test_df (pd.DataFrame): Testing DataFrame.
    - categorical_fields (list): List of column names to label encode.

    Returns:
    - tuple: (encoded_train_df, encoded_test_df)  - tuple containing label encoded training and testing dataframes
    """
    # Create a copy of the original DataFrames to avoid modifying them directly
    encoded_train_df = train_df.copy()
    encoded_test_df = test_df.copy()

    # identify categorical columns
    categorical_columns = categorical_fields

    # Initialize LabelEncoder
    le = LabelEncoder()

    # apply label encoding to each categorical column
    for col in categorical_columns:
        print(f"Label encoding : {col}...")
        # fit the label encoder on train data
        le.fit(encoded_train_df[col])

        # transform both train and test data using the fitted encoder
        encoded_train_df[col] = le.transform(encoded_train_df[col])
        
        if col in encoded_test_df.columns:
            # handle unseen labels in test data by mapping them to -1
            encoded_test_df[col] = encoded_test_df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else None)
            # for any NaN values (unseen labels), assign a new label
            encoded_test_df[col] = encoded_test_df[col].fillna(-1)
            # ensure the column is of integer type
            encoded_test_df[col] = encoded_test_df[col].astype(int)
            # alternatively, you can use the following line to transform test data
            # encoded_test_df[col] = le.transform(encoded_test_df[col])
            

    return encoded_train_df, encoded_test_df



# Encoding the train and test datasets.
trn_encoded, tst_encoded = label_encode_columns(train_df, test_df, categorical_fields)


print(trn_encoded.head())

print("\n test encoded data")
print(tst_encoded.head())


import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scale_dataframes(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        method: str ="zscore"
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
    Robust Scaler removes the median and scales the data according to the quantile range 
    (defaults to IQR: Interquartile Range). The IQR is the range 
    between the 1st quartile (25th quantile) 
    and the 3rd quartile (75th quantile). Centering and scaling happen independently 
    on each feature by computing the relevant statistics 
    on the samples in the training set. 
    Median and interquartile range are then stored to be used on later data using the transform method.
    
    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        A tuple of scaled train_df and test_df dataframes with the target
        column unchanged.
    """

    # create a copy of the dataframes to avoid modifying originals
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    # select columns to scale EXCLUDING target column
    numeric_cols = train_scaled.select_dtypes(include=['number']).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]

    # choose the scaler based on user input
    if method == 'zscore':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Unknown method: Choose scaling technique from 'zscore','minmax', or 'robust")
    
    # fit the scaler on the training data(numeric cols)
    scaler.fit(train_scaled[numeric_cols])

    # transform both train and test data
    train_scaled[numeric_cols] = scaler.transform(train_scaled[numeric_cols])
    test_scaled[numeric_cols] = scaler.transform(test_scaled[numeric_cols])

    return train_scaled, test_scaled



# using minmax scaling technique
train_scaled, test_scaled = scale_dataframes(trn_encoded, tst_encoded, 'y', method ='minmax')


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold



target_col = 'y'


def baseline_logistic_classifier_df_cv_ensemble(
        train_df,
        target_col,
        feature_cols,
        test_df = None,
        n_splits = 5,
        random_state=42
):
    """
    Performs K-fold cross-validation on the training DataFrame,
    trains logistic regression on each fold printing 
    the accuracy for each fold and the mean accuracy across all folds.
    Optionally  if test_df is provided, 
    it ensembles test predictions by averaging fold predictions.
    StratifiedKFold - 
    The folds are made by preserving the percentage of samples for each class in y in a binary or multiclass classification setting.

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

    # if test_df is provided, prepare to collect predictions for each fold
    if test_df is not None:
        X_test = test_df[feature_cols]
        # assuming binary classification -> shape: [num_test_samples, 2, n_splits]
        test_preds = np.zeros((X_test.shape[0], 2, n_splits))
    else:
        test_preds = None

    # K-fold cross-validation loop
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Initialize and train the Logistic Regression model
        model = LogisticRegression(penalty =  "l1", solver = "liblinear",
                                max_iter=100, random_state= random_state)
        model.fit(X_train, y_train)

        # Validate the model
        y_pred_val = model.predict(X_val)
        fold_accuracy = accuracy_score(y_val, y_pred_val)
        cv_accuracies.append(fold_accuracy)

        print(f"Validation {fold_idx+1} accuracy: {fold_accuracy:.4f}")
        
        # If test_df is provided, generate predictions for this fold
        if test_df is not None:
            test_pred_proba = model.predict_proba(X_test)  # shape: [num_test_samples, n_classes]
            test_preds[:, :, fold_idx] = test_pred_proba

    # After all folds, compute mean accuracy
    mean_cv_accuracy = np.mean(cv_accuracies)
    print(f"\nMean CV accuracy over {n_splits} folds: {mean_cv_accuracy:.4f}")

    # return results
    if test_df is None:
        return mean_cv_accuracy
    else:
        # Average test predictions across folds for test set
        avg_test_predictions = np.mean(test_preds, axis=2)
        return mean_cv_accuracy, avg_test_predictions


mean_cv_accuracy, avg_test_predictions = baseline_logistic_classifier_df_cv_ensemble(train_scaled, target, feature_cols,
                                                                                      test_scaled, n_splits = 5, random_state=468)


# logistic regression submission file
sub_df['y'] = 1 - avg_test_predictions
sub_df.to_csv('lr_submission.csv', index=False)
display(sub_df.head())


%pip install xgboost catboost lightgbm scikit-learn


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
                # 'tree_method': 'gpu_hist',
                'eval_metric':'auc',
                'random_state': 42
            }
        elif model_type == "catboost":
            param_file = {
                'iterations': 2048,
                'learning_rate': 0.03,
                'depth': 6,
                # 'task_type': 'GPU',
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
                # 'device': 'gpu',
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
                # 'tree_method': 'gpu_hist',
                'eval_metric':'auc',
                'random_state': 42
            }
        elif model_type == "catboost":
            param_file = {
                'iterations': 2048,
                'learning_rate': 0.03,
                'depth': 6,
                # 'task_type': 'GPU',
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
                # 'device': 'gpu',
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



# Average ROC-AUC: 0.9587
mean_probs, oof_predictions, model = train_model(train_scaled, 
                                                 test_scaled, 
                                                 feature_columns = feature_cols , 
                                                 target_column = target, 
                                                 model_type = "xgboost", 
                                                 param_file = None, 
                                                 n_splits = 5, 
                                                 categorical_features = None)


sub_df['y'] = 1 - mean_probs
sub_df.to_csv('xgbt_submission.csv', index=False)
display(sub_df.head())




