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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from IPython.display import display

def eda_summary(df, target=None, max_categories=20, plot=True, verbose=True):
    """
    Comprehensive EDA summary:
      - Shape, columns, memory usage, data types
      - Head, tail, random sample
      - Info, descriptive stats (numeric & categorical)
      - Missing values summary & patterns
      - Duplicate rows count
      - Unique value counts
      - Skewness & kurtosis
      - Histograms & boxplots
      - Correlation matrix & heatmap
      - Outlier detection (|z|>3)
      - VIF for multicollinearity
      - Datetime feature summary
      - Value counts for low-cardinality categoricals
      - Target-aware correlations & group summaries
    Params:
      df: pandas DataFrame
      target: column name for target variable (optional)
      max_categories: max unique values for categorical plots
      plot: whether to display plots
      verbose: whether to print textual summaries
    """
    # Shape & columns
    if verbose:
        print(f"======== Shape ========\nRows: {df.shape[0]}, Columns: {df.shape[1]}")
        print("======== Columns ========")
        print(df.columns.tolist())

    # Memory usage
    if verbose:
        print("\n======== Memory Usage ========")
        mem = df.memory_usage(deep=True).sort_values(ascending=False)
        display(mem)

    # Head, tail, sample
    if verbose:
        print("\n======== First 5 Rows ========")
        display(df.head())
        print("\n======== Last 5 Rows ========")
        display(df.tail())
        print("\n======== Random Sample (5 rows) ========")
        display(df.sample(min(5, len(df))))

    # Info
    if verbose:
        print("\n======== DataFrame Info ========")
        display(df.info())

    # Descriptive stats
    numeric = df.select_dtypes(include=[np.number])
    if verbose:
        print("\n======== Descriptive Stats (Numeric) ========")
        display(numeric.describe())
    categorical = df.select_dtypes(include=['object', 'category'])
    if verbose:
        print("\n======== Descriptive Stats (Categorical) ========")
        if not categorical.empty:
            display(categorical.describe())
        else:
            print("No categorical columns found.")

    # Missing values
    if verbose:
        print("\n======== Missing Values ========")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        miss_df = pd.DataFrame({ 'count': missing, 'percent': missing_pct })
        display(miss_df)
    if plot:
        # compact missing value matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        msno.matrix(df, ax=ax)
        ax.set_title("Missing Value Matrix")
        plt.tight_layout()
        plt.show()

    # Duplicate rows
    if verbose:
        dup_count = df.duplicated().sum()
        print(f"\n======== Duplicated Rows ========\nTotal duplicates: {dup_count}")

    # Unique counts & cardinality
    if verbose:
        print("\n======== Unique Values per Column ========")
        display(df.nunique().sort_values())

    # Skewness & kurtosis
    if verbose:
        print("\n======== Skewness & Kurtosis ========")
        skk = pd.DataFrame({ 'skewness': numeric.skew(), 'kurtosis': numeric.kurtosis() })
        display(skk)

    # Histograms & boxplots
    if plot and not numeric.empty:
        numeric.hist(figsize=(12, 8))
        plt.tight_layout()
        plt.show()
        numeric.plot.box(figsize=(12, 6), vert=False)
        plt.tight_layout()
        plt.show()

    # Correlation matrix & heatmap
    if numeric.shape[1] > 1:
        corr = numeric.corr()
        if verbose:
            print("\n======== Correlation Matrix ========")
            display(corr)
        if plot:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
            plt.title('Numeric Correlation Heatmap')
            plt.show()

    # Outlier detection via z-score
    if verbose and not numeric.empty:
        print("\n======== Outliers (|z|>3) per Numeric Column ========")
        z = np.abs(stats.zscore(numeric.dropna()))
        outlier_counts = (z > 3).sum(axis=0)
        out_df = pd.DataFrame({ 'outliers': outlier_counts }, index=numeric.columns)
        display(out_df)

    # VIF for multicollinearity
    if numeric.shape[1] > 1:
        if verbose:
            print("\n======== Variance Inflation Factor (VIF) ========")
        vif_data = pd.DataFrame({
            'feature': numeric.columns,
            'VIF': [variance_inflation_factor(numeric.values, i) for i in range(numeric.shape[1])]
        })
        display(vif_data)

    # Value counts for categorical
    if verbose and not categorical.empty:
        print("\n======== Value Counts (Categorical, low-cardinality) ========")
        for col in categorical.columns:
            if df[col].nunique() <= max_categories:
                print(f"-- {col} --")
                display(df[col].value_counts())
                if plot:
                    df[col].value_counts().plot.bar(figsize=(8, 4))
                    plt.title(f"Value Counts: {col}")
                    plt.tight_layout()
                    plt.show()

    # Datetime feature summary
    datetimes = df.select_dtypes(include=['datetime64'])
    if verbose and not datetimes.empty:
        for col in datetimes.columns:
            print(f"\n-- Datetime Column: {col} --")
            print(f"Range: {df[col].min()} to {df[col].max()}")
            freq = df[col].diff().mode().iloc[0] if df[col].diff().mode().any() else None
            print(f"Sample frequency (mode of diffs): {freq}")
            if plot:
                df[col].dt.year.value_counts().sort_index().plot()
                plt.title(f"Yearly counts for {col}")
                plt.tight_layout()
                plt.show()

    # Target-aware summaries
    if target is not None and target in df.columns:
        if verbose:
            print(f"\n======== Target-Aware Analysis (target = {target}) ========")
        corr_target = numeric.corrwith(df[target])
        if verbose:
            print("Numeric feature correlations with target:")
            display(corr_target)
        for col in categorical.columns:
            if df[col].nunique() <= max_categories:
                if verbose:
                    print(f"Group means of {target} by {col}:")
                display(df.groupby(col)[target].mean().sort_values())

    print("\nEDA summary complete.")



eda_summary(trn_df)


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
trn_df, tst_df = label_encode_datasets(trn_df, tst_df, categorical_fields)


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from typing import Optional, Tuple, Union, List

def train_nn_regressor(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    test_targets: Optional[Union[pd.Series, np.ndarray]] = None,
    hidden_layers: List[int] = [64, 32],
    activation: str = 'relu',
    optimizer: str = 'adam',
    learning_rate: float = 0.001,
    lr_decay_type: Optional[str] = None,      # 'exponential', 'inverse_time', or None
    decay_steps: int = 100000,
    decay_rate: float = 0.96,
    staircase: bool = False,
    dropout_rate: float = 0.0,
    epochs: int = 10,
    batch_size: int = 32,
    n_splits: int = 5,
    shuffle: bool = True,
    cv_random_seed: int = 42,
    random_seed: int = 42,
    early_stopping: bool = True,
    es_monitor: str = 'val_loss',
    es_patience: int = 5
) -> Tuple[np.ndarray, Optional[float], List[float], np.ndarray]:
    """
    Trains a NN regressor with:
      â€¢ optional dropout, LR decay, early stopping
      â€¢ n-fold CV on train set, with per-fold test inference
      â€¢ returns OOF preds for stacking

    Returns:
      y_test_pred_avg â€“ array, average of per-fold test predictions
      test_rmse       â€“ float or None, RMSE between averaged preds & test_targets
      cv_val_scores   â€“ list of RMSEs on each val fold
      oof_preds       â€“ array of out-of-fold train predictions
    """
    # reproducibility & GPU
    tf.random.set_seed(random_seed)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    # prepare data
    X = train_df.drop(columns=[target_col]).values
    y = train_df[target_col].values
    X_test = test_df.values

    # containers
    cv_val_scores    = []
    oof_preds        = np.zeros_like(y, dtype=float)
    test_preds_folds = (np.zeros((n_splits, X_test.shape[0]))
                        if n_splits > 1 else None)

    # model builder with BatchNorm & Activation
    def build_model(input_dim):
        model = tf.keras.Sequential()
        for i, units in enumerate(hidden_layers):
            if i == 0:
                model.add(tf.keras.layers.Dense(units, activation=None,
                                                input_shape=(input_dim,)))
            else:
                model.add(tf.keras.layers.Dense(units, activation=None))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Activation(activation))
            if dropout_rate > 0:
                model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        return model

    # cross-validation
    if n_splits > 1:
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=cv_random_seed)
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            # per-fold scaling
            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)

            # build & compile
            with strategy.scope():
                model = build_model(input_dim=X_tr_s.shape[1])

                # LR schedule
                if lr_decay_type == 'exponential':
                    lr = tf.keras.optimizers.schedules.ExponentialDecay(
                        learning_rate, decay_steps, decay_rate, staircase
                    )
                elif lr_decay_type == 'inverse_time':
                    lr = tf.keras.optimizers.schedules.InverseTimeDecay(
                        learning_rate, decay_steps, decay_rate, staircase
                    )
                else:
                    lr = learning_rate

                Optim = {
                    'adam':    tf.keras.optimizers.Adam,
                    'sgd':     tf.keras.optimizers.SGD,
                    'rmsprop': tf.keras.optimizers.RMSprop
                }.get(optimizer.lower(), tf.keras.optimizers.get)
                opt_instance = (Optim(learning_rate=lr)
                                if callable(Optim) else tf.keras.optimizers.get(optimizer))
                model.compile(optimizer=opt_instance, loss='mse')

            # callbacks
            cbs = []
            if early_stopping:
                cbs.append(tf.keras.callbacks.EarlyStopping(
                    monitor=es_monitor,
                    patience=es_patience,
                    restore_best_weights=True,
                    verbose=0
                ))

            # train
            model.fit(
                X_tr_s, y_tr,
                validation_data=(X_val_s, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=cbs,
                verbose=0
            )

            # validation eval
            y_val_pred = model.predict(X_val_s, verbose=0).flatten()
            rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
            cv_val_scores.append(rmse_val)
            print(f"Fold {fold}/{n_splits} â€” Val RMSE: {rmse_val:.4f}")

            # store OOF preds
            oof_preds[val_idx] = y_val_pred

            # test inference in-fold
            test_preds_folds[fold-1] = model.predict(X_test_s, verbose=0).flatten()

        # aggregate results
        print(f"CV Val RMSE â†’ mean: {np.mean(cv_val_scores):.4f}, "
              f"std: {np.std(cv_val_scores):.4f}")

        # average test preds
        y_test_pred_avg = test_preds_folds.mean(axis=0)
        test_rmse = None
        if test_targets is not None:
            test_rmse = np.sqrt(mean_squared_error(test_targets, y_test_pred_avg))
            print(f"Test RMSE (avg preds): {test_rmse:.4f}")

        return y_test_pred_avg, test_rmse, cv_val_scores, oof_preds

    # single-fit (no CV) branch
    scaler = StandardScaler().fit(X)
    X_s      = scaler.transform(X)
    X_test_s = scaler.transform(X_test)

    with strategy.scope():
        model = build_model(input_dim=X_s.shape[1])

        if lr_decay_type == 'exponential':
            lr = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps, decay_rate, staircase
            )
        elif lr_decay_type == 'inverse_time':
            lr = tf.keras.optimizers.schedules.InverseTimeDecay(
                learning_rate, decay_steps, decay_rate, staircase
            )
        else:
            lr = learning_rate

        Optim = {
            'adam':    tf.keras.optimizers.Adam,
            'sgd':     tf.keras.optimizers.SGD,
            'rmsprop': tf.keras.optimizers.RMSprop
        }.get(optimizer.lower(), tf.keras.optimizers.get)
        opt_instance = (Optim(learning_rate=lr)
                        if callable(Optim) else tf.keras.optimizers.get(optimizer))
        model.compile(optimizer=opt_instance, loss='mse')

    callbacks = []
    if early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor=es_monitor,
            patience=es_patience,
            restore_best_weights=True,
            verbose=1
        ))

    model.fit(
        X_s, y,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    y_test_pred = model.predict(X_test_s).flatten()
    test_rmse = None
    if test_targets is not None:
        test_rmse = np.sqrt(mean_squared_error(test_targets, y_test_pred))
        print(f"Test RMSE: {test_rmse:.4f}")

    # for no-CV we return empty val-scores and zero OOF preds
    return y_test_pred, test_rmse, [], oof_preds



## # Case A: you have ground-truth for test
# preds, test_rmse = train_nn_regressor(
#     train_df=trn_df,
#     test_df=tst_df,
#     target_col='Calories',
#     test_targets=df_test_target,
#     hidden_layers=[128,64],
#     epochs=50
# )
# # preds is an array of predictions, test_rmse is a float

# Case B: no ground-truth on test set yet
preds_only, _, _, oof_preds = train_nn_regressor(
    train_df=trn_df,
    test_df=tst_df,
    target_col='Calories',
    hidden_layers=[256, 128, 8],
    batch_size = 2048,
    epochs = 256,
    dropout_rate = 0.1,
    es_patience = 64,
    n_splits = 10,
    learning_rate = 0.001,
    decay_steps = 32,
    decay_rate = 0.98,
    activation = 'relu'
    
)
# preds_only is an array of predictions


# CV Val RMSE â†’ mean: 0.0745, std: 0.0038


preds_only


sub_df['Calories'] = np.expm1(preds_only)
sub_df['Calories'] = np.clip(sub_df['Calories'], 1, 314)
sub_df.to_csv('nn_submission.csv', index=False)
display(sub_df.head())




