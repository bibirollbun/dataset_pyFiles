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

    # Return the seed
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


trn_file_path = "/kaggle/input/playground-series-s5e6/train.csv"  # Replace with your CSV file path
trn_df = load_csv_to_dataframe(trn_file_path, ignore_fields=['id'])

test_file_path = "/kaggle/input/playground-series-s5e6/test.csv"  # Replace with your CSV file path
tst_df = load_csv_to_dataframe(test_file_path, ignore_fields=['id'])

sample_file_path = "/kaggle/input/playground-series-s5e6/sample_submission.csv"  # Replace with your CSV file path
sub_df = load_csv_to_dataframe(sample_file_path)


orig_file_path = "/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv"  # Replace with your CSV file path
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
categorical_fields = ['Soil Type', 'Crop Type']
trn_df, tst_df = label_encode_datasets(trn_df, tst_df, categorical_fields)


# Suppose your DataFrame is train_df and your target column is named "target"
import pandas as pd

# 1. Ensure the column is of dtype "category"
trn_df["Fertilizer Name"] = trn_df["Fertilizer Name"].astype("category")

# 2. Create a new column of integer codes
trn_df["Fertilizer Name Enc"] = trn_df["Fertilizer Name"].cat.codes

# 3. If you ever need to see the mapping:
mapping = dict(enumerate(trn_df["Fertilizer Name"].cat.categories))
# mapping might look like: {0: "ClassA", 1: "ClassB", 2: "ClassC", ...}


trn_df = trn_df.drop(columns = ['Fertilizer Name'])


trn_df.head()


target = 'Fertilizer Name Enc'
ignore = ['Fertilizer Name Enc','Fertilizer Name']
feature_cols = [col for col in trn_df.columns if col not in ignore]
print(feature_cols)


def train_model(
    train_df,
    test_df,
    target_column,
    feature_columns=None,
    model_type="xgboost",
    param_file=None,
    n_splits=5,
    categorical_features=None,
    use_target_encoding=False,
):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
    from scipy.stats import mode

    # --- Prepare features and target ---
    if feature_columns is not None:
        feature_columns = [c for c in feature_columns if c != target_column]
        X = train_df[feature_columns].copy()
        X_test = test_df[feature_columns].copy()
    else:
        X = train_df.drop(columns=[target_column]).copy()
        X_test = test_df.copy()
    y = train_df[target_column].copy()

    # Capture original indices
    train_idx = train_df.index
    test_idx = test_df.index

    # Class labels
    classes = np.sort(y.unique())
    n_classes = len(classes)

    # Infer categorical features
    if categorical_features is None:
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # --- Choose model class ---
    if model_type == "xgboost":
        from xgboost import XGBClassifier as Model
    elif model_type == "catboost":
        from catboost import CatBoostClassifier as Model
    elif model_type == "lgbm":
        from lightgbm import LGBMClassifier as Model
    elif model_type == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier as Model
    else:
        raise ValueError(f"Unsupported model_type: {model_type!r}")

    # --- Default params (omitted for brevity) ---
    if param_file is None:
        # ... your existing defaults ...
        pass

    model = Model(**param_file)

    # --- Crossâ€�validation setup ---
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds       = np.empty(len(train_df), dtype=y.dtype)
    fold_test_probas = []
    accuracy_scores  = []
    fold_aucs        = []
    fold_loglosses   = []

    for fold, (tr, val) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[tr], X.iloc[val]
        y_tr, y_val = y.iloc[tr], y.iloc[val]

        # Fit kwargs per model
        fit_kwargs = {}
        if model_type == "catboost":
            fit_kwargs = dict(
                cat_features=categorical_features,
                eval_set=(X_val, y_val),
                early_stopping_rounds=250,
                verbose=False,
            )
        elif model_type in ("xgboost", "lgbm"):
            fit_kwargs = dict(
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=250,
                verbose=False,
            )

        model.fit(X_tr, y_tr, **fit_kwargs)

        # --- Validation predictions & metrics ---
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)

        oof_preds[val] = val_pred
        accuracy_scores.append(accuracy_score(y_val, val_pred))

        ll = log_loss(y_val, val_proba)
        auc = roc_auc_score(y_val, val_proba, multi_class="ovr", average="macro")
        fold_loglosses.append(ll)
        fold_aucs.append(auc)

        print(f"Fold {fold}: AUC = {auc:.4f}, LogLoss = {ll:.4f}")

        # --- Test fold proba ---
        fold_test_probas.append(model.predict_proba(X_test))

    # Print CV averages
    print(f"Average CV Accuracy : {np.mean(accuracy_scores):.4f}")
    print(f"Average CV AUC      : {np.mean(fold_aucs):.4f}")
    print(f"Average CV LogLoss  : {np.mean(fold_loglosses):.4f}")

    # Majorityâ€�vote final class
    fold_test_preds = np.stack([np.argmax(p, axis=1) for p in fold_test_probas], axis=0)
    majority_vote = mode(fold_test_preds, axis=0).mode.flatten()

    # Average probabilities
    avg_test_proba = np.mean(np.stack(fold_test_probas, axis=0), axis=0)

    # Topâ€�3 indices & probs
    top3_idx     = np.argsort(avg_test_proba, axis=1)[:, -3:][:, ::-1]
    top3_probs   = np.take_along_axis(avg_test_proba, top3_idx, axis=1)
    top3_classes = classes[top3_idx]

    # --- Build DataFrames ---
    df_oof = pd.DataFrame(
        {"oof_pred": oof_preds},
        index=train_idx
    )
    df_majority = pd.DataFrame(
        {"majority_vote": majority_vote},
        index=test_idx
    )
    proba_cols = [f"proba_{cls}" for cls in classes]
    df_proba = pd.DataFrame(
        avg_test_proba,
        columns=proba_cols,
        index=test_idx
    )
    df_top3 = pd.DataFrame(
        {
            "top1":   top3_classes[:, 0],
            "top1_p": top3_probs[:,   0],
            "top2":   top3_classes[:, 1],
            "top2_p": top3_probs[:,   1],
            "top3":   top3_classes[:, 2],
            "top3_p": top3_probs[:,   2],
        },
        index=test_idx
    )
    df_fold_metrics = pd.DataFrame({
        "auc":      fold_aucs,
        "logloss":  fold_loglosses
    }, index=[f"fold_{i}" for i in range(1, n_splits+1)])

    return {
        "oof_predictions": df_oof,
        "majority_vote":   df_majority,
        "average_proba":   df_proba,
        "top3":            df_top3,
        "fold_metrics":    df_fold_metrics,
        "model":           model
    }



n_classes = 7

param_file = {
                "n_estimators": 2048,
                "learning_rate": 0.03,
                "max_depth": 12,
                "subsample": 0.80,
                "colsample_bytree": 0.80,
                "min_child_weight": 60,
                "alpha": 1.0,
                "tree_method": "gpu_hist",
                "use_label_encoder": False,
                "objective": "multi:softprob",
                "num_class": n_classes,
                "eval_metric": "mlogloss",
                "random_state": 42,
            }


categorical_features = ['Soil Type', 'Crop Type']

train_model_results = train_model(trn_df, 
                                  tst_df, 
                                  target, 
                                  feature_columns=None, 
                                  model_type="xgboost", 
                                  param_file=param_file, 
                                  n_splits=5, 
                                  categorical_features=categorical_features, 
                                  use_target_encoding=True
                                 )


results = train_model_results['top3']
results_array = results[['top1', 'top2', 'top3']]
results_array['top1'] = results_array['top1'].map(mapping)
results_array['top2'] = results_array['top2'].map(mapping)
results_array['top3'] = results_array['top3'].map(mapping)


results_array["results"] = results_array["top1"].astype(str) + " " + results_array["top2"].astype(str) + " " + results_array["top3"].astype(str)


mapping


sub_df['Fertilizer Name'] = results_array['results']


sub_df.to_csv('submission', index = False)


sub_df




