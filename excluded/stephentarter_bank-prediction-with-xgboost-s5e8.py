# Install necessary components
!pip -q install xgboost matplotlib plotly catboost


import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
import random 
import warnings  # Provides a way to control the display of warning messages (e.g., filter out deprecation warnings)
from IPython.display import display  # for nicer display in notebooks
from pandas.api.types import is_categorical_dtype
import matplotlib.pyplot as plt
%matplotlib inline


# Define some awesome utilities, from C4rl05/V on kaggle
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


# Configurable flag to control whether GPU is used
USE_GPU = True


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



PLAYGROUND_PATH = '/kaggle/input/playground-series-s5e8/'


training_df = pd.read_csv(PLAYGROUND_PATH + 'train.csv')
training_df


test_df = pd.read_csv(PLAYGROUND_PATH + 'test.csv')
test_df


training_df.drop('id', axis=1, inplace=True)
training_df.head()


# Save a copy with the ID for use later
test_ids = test_df['id'].copy()

test_df.drop('id', axis=1, inplace=True)
test_df.head()


eda_summary(training_df)


# Plot Target Distribution
ax = (training_df['y'].value_counts(normalize=True)
        .rename({0:'No',1:'Yes'})
        .plot(kind='bar'))
plt.title('Target Distribution'); 
plt.xlabel('Opened Account'); 
plt.ylabel('Share'); 
plt.show()


# Plot duration
training_df['duration'].clip(upper=training_df['duration'].quantile(0.99)).plot(kind='hist', bins=50)
plt.title('Duration (clipped 99th)'); 
plt.xlabel('seconds'); 
plt.show()


# Plot pdays missingness
(training_df['pdays'].eq(-1) | training_df['pdays'].isna()).value_counts(normalize=True).plot(kind='bar')
plt.title('pdays missing / never-contacted rate'); 
plt.show()


eda_summary(test_df)


# Plot duration
test_df['duration'].clip(upper=test_df['duration'].quantile(0.99)).plot(kind='hist', bins=50)
plt.title('Duration (clipped 99th)'); 
plt.xlabel('seconds'); 
plt.show()


# Plot pdays missingness
(test_df['pdays'].eq(-1) | test_df['pdays'].isna()).value_counts(normalize=True).plot(kind='bar')
plt.title('pdays missing / never-contacted rate'); 
plt.show()


def transform_job(df: pd.DataFrame) -> pd.DataFrame:
    # If the 'job' feature is already categorical, then remark about it and return
    if is_categorical_dtype(df['job']):
        print("The 'job' feature has already been transformed.  Operation cancelled.")
        return df

    out_df = df.copy()
    out_df['job'] = out_df['job'].astype('category')
        
    return out_df


def fit_marital(df: pd.DataFrame):
    global marital_encoder
    
    # If the 'marital' feature isn't present, then remark about it and return
    if 'marital' not in df.columns:
        print("The 'marital' feature has already been transformed.  Operation cancelled.")
        return
        
    marital_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    marital_encoder.fit(df[['marital']])

def transform_marital(df: pd.DataFrame, drop_original: bool) -> pd.DataFrame:
    """Idempotent one-hot transform for 'marital'.
    - Drops any previously generated marital one-hot columns.
    - Works even if called multiple times.
    - Robust to unseen categories if encoder was created with handle_unknown='ignore'.
    """
    # If the 'marital' feature isn't present, then remark about it and return
    if 'marital' not in df.columns:
        print("The 'marital' feature has already been transformed.  Operation cancelled.")
        return df
        
    # Ensure encoder is fitted
    if not hasattr(marital_encoder, "categories_"):
        marital_encoder.fit(df[["marital"]])

    # Compute consistent column names
    cols = marital_encoder.get_feature_names_out(["marital"])

    # Get a copy of the DataFrame to work upon.
    out_df = df.copy()
    
    # Transform to sparse matrix
    X = marital_encoder.transform(out_df[["marital"]])

    # Drop existing one-hot columns if present
    to_drop = [c for c in cols if c in out_df.columns]
    if to_drop:
        out_df = out_df.drop(columns=to_drop)

    # Build a (sparse) DataFrame aligned to df.index
    df_ohe = pd.DataFrame.sparse.from_spmatrix(X, index=out_df.index, columns=cols)
    df_ohe = df_ohe.astype("Int8")
    
    # Optionally drop the original column
    if drop_original and "marital" in out_df.columns:
        out_df = out_df.drop(columns=["marital"])

    # Concatenate and return
    return pd.concat([out_df, df_ohe], axis=1)

def save_marital_encoder():
    joblib.dump(marital_encoder, "marital_encoder.pkl")
    print('marital_encoder saved')


education_encoder = LabelEncoder()

def fit_education(df: pd.DataFrame):
    # If the 'education' feature isn't present, then remark about it and return
    if 'education' not in df.columns:
        print("The 'education' feature has already been transformed.  Operation cancelled.")
        return

    education_encoder.fit(df['education'])

def transform_education(df: pd.DataFrame, drop_original: bool) -> pd.DataFrame:
    # If the 'education' feature isn't present, then remark about it and return
    if 'education' not in df.columns:
        print("The 'education' feature has already been transformed.  Operation cancelled.")
        return df

    out_df = df.copy()
    out_df['encoded_education'] = education_encoder.transform(out_df['education'])
    out_df['education_unknown'] = (out_df['education'] == 'unknown').astype('int8')

    if drop_original:
        out_df.drop('education', axis=1, inplace=True)
        
    return out_df

def save_education_encoder():
    joblib.dump(education_encoder, "education_encoder.pkl")
    print('education_encoder saved')


def convert_default(df: pd.DataFrame, drop_original: bool) -> pd.DataFrame:
    # If the 'default' feature isn't present, then remark about it and return
    if 'default' not in df.columns:
        print("The 'default' feature has already been transformed.  Operation cancelled.")
        return df

    out_df = df.copy()
    
    out_df['default_bool'] = out_df['default'].str.lower().map({"yes": 1, "no": 0}).astype("Int8")

    if drop_original:
        out_df.drop('default', axis=1, inplace=True)

    return out_df


def add_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()

    out_df['balance_per_age'] = out_df['balance'] / out_df['age']
    out_df['balance_per_campaign'] = out_df['balance'] / out_df['campaign']

    return out_df


def convert_housing(df: pd.DataFrame, drop_original: bool) -> pd.DataFrame:
    # If the 'housing' feature isn't present, then remark about it and return
    if 'housing' not in df.columns:
        print("The 'housing' feature has already been transformed.  Operation cancelled.")
        return df

    out_df = df.copy()
    
    out_df['housing_bool'] = out_df['housing'].str.lower().map({"yes": 1, "no": 0}).astype("Int8")

    if drop_original:
        out_df.drop('housing', axis=1, inplace=True)

    return out_df


def convert_loan(df: pd.DataFrame, drop_original: bool) -> pd.DataFrame:
    # If the 'loan' feature isn't present, then remark about it and return
    if 'loan' not in df.columns:
        print("The 'loan' feature has already been transformed.  Operation cancelled.")
        return df

    out_df = df.copy()
    
    out_df['loan_bool'] = out_df['loan'].str.lower().map({"yes": 1, "no": 0}).astype("Int8")

    if drop_original:
        out_df.drop('loan', axis=1, inplace=True)

    return out_df


def transform_contact(df: pd.DataFrame) -> pd.DataFrame:
    # If the 'contact' feature is already categorical, then remark about it and return
    if is_categorical_dtype(df['contact']):
        print("The 'contact' feature has already been transformed.  Operation cancelled.")
        return df

    out_df = df.copy()
    
    out_df['contact'] = out_df['contact'].astype('category')

    return out_df


# First, let's see if there are any leap-year dates in the dataset
training_df[(training_df['month'] == 'feb') & (training_df['day'] == 29)]


# Alright, we found both Feb 29 as well as out of range day values.
# So, we'll choose 2020 as the year, since that's a leap year.
# Wealso take action to coerce the bad dates into good ones.
def convert_calendar_features(df, drop_originals: bool):
    # If the 'month' feature isn't present, then remark about it and return
    if 'month' not in df.columns:
        print("The 'month' feature has already been transformed.  Operation cancelled.")
        return df

    # Work on a copy so we don't mess stuff up if this fails halfway throug
    out_df = df.copy()

    # Create a numbered month series
    month_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12
    }
    month = out_df["month"].str.lower().map(month_map)

    # Create a year series that is the correct length
    year_array = np.empty(out_df["month"].size)
    year_array.fill(2020)
    year = pd.Series(data=year_array, name='year')

    # Assemble those series into a DataFrame suitable for
    raw_date_df = pd.DataFrame(
        {'year': year,
         'month': month,
         'day': out_df['day']
        }
    )

    # Look for bad dates and try to show them
    bad = pd.to_datetime(raw_date_df, errors="coerce")
    raw_date_df.loc[bad.isna()]

    # There's some bogus dats in the dataset, so we need to do some hygiene.
    # 1) month-end day for each row (handles leap years)
    month_start = pd.to_datetime(dict(year=raw_date_df["year"],
                                      month=raw_date_df["month"],
                                      day=1))
    month_end_day = (month_start + pd.offsets.MonthEnd(0)).dt.day
    
    # 2) clip day into [1, month_end_day]
    day_clipped = raw_date_df["day"].clip(lower=1, upper=month_end_day)

    # 3) build safe dates
    date_df = pd.to_datetime(dict(year=raw_date_df["year"],
                                  month=raw_date_df["month"],
                                  day=day_clipped))
    
    # Flag rows that were adjusted
    adjusted = day_clipped.ne(raw_date_df["day"])
    print(f"Adjusted {adjusted.sum()} rows where day exceeded month length.")
    
    # Now, create new features for the day of year and cyclical encoding.
    out_df["day_of_year"] = date_df.dt.dayofyear
    out_df["day_sin"] = np.sin(2 * np.pi * out_df["day_of_year"] / 366.0)
    out_df["day_cos"] = np.cos(2 * np.pi * out_df["day_of_year"] / 366.0)

    if drop_originals:
        out_df.drop('month', axis=1, inplace=True)
        out_df.drop('day', axis=1, inplace=True)

    return out_df


def add_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()

    out_df['log_duration'] = np.log1p(out_df['duration'])

    return out_df


def add_pdays_features(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()
    
    # Add bins for pdays: -1 (not contacted), 0-90 (recent), 91-180 (quarterly), etc.
    bins = [-2, 0, 90, 180, 365, 1000]
    labels = ['NotContacted', 'Recent', 'WithinHalfYear', 'WithinYear', 'LongAgo']
    out_df['pdays_binned'] = pd.cut(out_df['pdays'].fillna(-1), bins=bins, labels=labels)

    # The 'pdays' feature uses -1 to indicate nothing.  Replace all these with NaNs and add a 'pdays_missing' feature
    out_df.loc[out_df['pdays'] == -1, 'pdays'] = np.nan
    out_df["pdays_missing"] = (out_df["pdays"].isna()).astype("int8")

    return out_df


def transform_previous(df: pd.DataFrame, drop_original: bool) -> pd.DataFrame:
    # If the 'previous' feature isn't present, then remark about it and return
    if 'previous' not in df.columns:
        print("The 'previous' feature has already been transformed.  Operation cancelled.")
        return df

    out_df = df.copy()
    
    out_df['prev_any'] = (out_df['previous'] > 0).astype('int8')

    out_df['prev_log'] = np.where(out_df['previous'] > 0,
                                  np.log1p(out_df["previous"]),
                                  0)

    if drop_original:
        out_df.drop('previous', axis=1, inplace=True)

    return out_df    


def transform_poutcome(df: pd.DataFrame) -> pd.DataFrame:
    # If the 'poutcome' feature is already categorical, then remark about it and return
    if is_categorical_dtype(df['poutcome']):
        print("The 'poutcome' feature has already been transformed.  Operation cancelled.")
        return df

    out_df = df.copy()
    
    out_df['poutcome'] = out_df['poutcome'].astype('category')

    return out_df


# Fit the features
def fit_features(df: pd.DataFrame):
    fit_marital(df)
    fit_education(df)

    save_marital_encoder()
    save_education_encoder()


# Transform features
def transform_features(df: pd.DataFrame, drop_original: bool) -> pd.DataFrame:
    out_df = df.copy()

    out_df = transform_job(out_df)
    out_df = transform_marital(out_df, drop_original)
    out_df = transform_education(out_df, drop_original)
    out_df = convert_default(out_df, drop_original)
    out_df = add_balance_features(out_df)
    out_df = convert_housing(out_df, drop_original)
    out_df = convert_loan(out_df, drop_original)
    out_df = transform_contact(out_df)
    out_df = convert_calendar_features(out_df, drop_original)
    out_df = add_duration_features(out_df)
    out_df = add_pdays_features(out_df)
    out_df = transform_previous(out_df, drop_original)
    out_df = transform_poutcome(out_df)

    return out_df


fit_features(training_df)


training_df = transform_features(training_df, True)
training_df.head(10)


# Examine the transformed training dataset
eda_summary(training_df)


# Separate the predictors(X) from the target(y)
TARGET = "y"

X = training_df.drop(columns=[TARGET])
y = training_df[TARGET]


# imbalance weight
pos = y.sum()
neg = len(y) - pos
spw = neg / pos

# Store out-of-fold (OOF) and test predictions
oof_preds_xgb = np.zeros(len(X))
oof_preds_lgb = np.zeros(len(X))
oof_preds_cat = np.zeros(len(X))

# Identify categorical features for LightGBM and CatBoost
categorical_features = X.select_dtypes(include=['category']).columns.tolist()

# Define paramteres for the various models
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "enable_categorical": True,
    "tree_method": "hist" if not USE_GPU else "gpu_hist",
    "predictor": "auto" if not USE_GPU else "gpu_predictor",
    "learning_rate": 0.03,
    "max_depth": 7,               # ≈ num_leaves ~ 128
    "min_child_weight": 50,       # ≈ LGBM min_data_in_leaf
    "subsample": 0.8,             # ≈ bagging_fraction
    "colsample_bytree": 0.8,      # ≈ feature_fraction
    "alpha": 0.0,                 # L1
    "lambda": 5.0,                # L2
    "gamma": 0.0,
    "scale_pos_weight": spw,
    "max_bin": 512,
    "colsample_bylevel": 0.9,
    "seed": seed,
}

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': seed
}

cat_params = {
    'iterations': 2000,
    'learning_rate': 0.03,
    'eval_metric': 'AUC',
    'random_seed': seed,
    'verbose': 0,
    'cat_features': categorical_features
}


# fixed folds for stability
dall = xgb.DMatrix(X, label=y, enable_categorical=True)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
folds = list(kf.split(X, y))


# Let the training begin!
for fold, (train_idx, val_idx) in enumerate(folds):
    print(f"===== Fold {fold+1} =====")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # --- XGBoost ---
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    xgb_model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=3000,
        evals=[(dval, 'eval')],
        early_stopping_rounds=150,
        verbose_eval=0
    )
    # Save the trained model
    xgb_model.save_model(f'xgb_model_fold_{fold}.json')
    oof_preds_xgb[val_idx] = xgb_model.predict(dval)

    # --- LightGBM ---
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    # Save the trained model
    lgb_model.booster_.save_model(f'lgb_model_fold_{fold}.txt')
    oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]

    # --- CatBoost ---
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=150,
                  verbose=0)
    # Save the trained model
    cat_model.save_model(f'cat_model_fold_{fold}.cbm')
    oof_preds_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]


# Combine the OOF predictions into a single matrix
# Each column represents a model's predictions.
# The .T transposes the matrix to the correct shape (n_samples, n_models).
oof_preds = np.vstack([
    oof_preds_xgb, 
    oof_preds_lgb, 
    oof_preds_cat
]).T

# Define the Objective Function ---
# This is the function that Scipy's minimizer will call.
# It takes a list of weights and returns the negative AUC score.
def get_oof_auc(weights):
    # Ensure weights are normalized to sum to 1
    w = np.array(weights) / np.sum(weights)
    
    # Calculate the weighted average of the OOF predictions
    blended_oof = np.dot(oof_preds, w)
    
    # Calculate AUC and return its negative value
    auc = roc_auc_score(y, blended_oof)
    return -auc

# Set up the Optimization ---
# Initial guess for the weights (can be anything that sums to 1)
initial_weights = [1/3, 1/3, 1/3]

# Set bounds for each weight (between 0 and 1)
bounds = [(0, 1)] * 3

# Set the constraint that the weights must sum to 1
constraints = ({
    'type': 'eq',
    'fun': lambda w: np.sum(w) - 1
})

# Run the Optimizer ---
# This is where the 'result' variable is created.
# We use the SLSQP method, which is good for constrained optimization problems.
print("Finding optimal weights...")
result = minimize(
    fun=get_oof_auc,
    x0=initial_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# Extract and Display the Results ---
# The optimal weights are stored in result.x
optimal_weights = result.x
# The best score is the negative of the function's minimum value
best_auc = -result.fun

print("\n--- Optimal Ensemble Weights ---")
print(f"XGBoost Weight:  {optimal_weights[0]:.4f}")
print(f"LightGBM Weight: {optimal_weights[1]:.4f}")
print(f"CatBoost Weight: {optimal_weights[2]:.4f}")
print("---------------------------------")
print(f"Blended OOF AUC: {best_auc:.5f}")

# Save the weights for the prediction phase ---
np.save('optimal_weights.npy', optimal_weights)
print("\nOptimal weights saved to 'optimal_weights.npy'")


def predict_on_dataset(new_df):
    """
    Loads the trained ensemble and predicts on a new dataset.
    """
    # IMPORTANT: Apply the exact same feature engineering
    # This assumes you have your feature engineering steps in a function
    new_df_processed = transform_features(new_df, drop_original=True)
    
    # Load the optimal weights
    optimal_weights = np.load('optimal_weights.npy')

    # Load models and predict
    all_test_preds_xgb = []
    all_test_preds_lgb = []
    all_test_preds_cat = []

    for fold in range(len(folds)):
        # XGBoost Prediction
        xgb_model = xgb.Booster()
        xgb_model.load_model(f'xgb_model_fold_{fold}.json')
        dtest = xgb.DMatrix(new_df_processed, enable_categorical=True)
        all_test_preds_xgb.append(xgb_model.predict(dtest))

        # LightGBM Prediction
        lgb_model = lgb.Booster(model_file=f'lgb_model_fold_{fold}.txt')
        all_test_preds_lgb.append(lgb_model.predict(new_df_processed))

        # CatBoost Prediction
        cat_model = CatBoostClassifier()
        cat_model.load_model(f'cat_model_fold_{fold}.cbm')
        all_test_preds_cat.append(cat_model.predict_proba(new_df_processed)[:, 1])

    # Average the predictions from all folds for each model type
    avg_preds_xgb = np.mean(all_test_preds_xgb, axis=0)
    avg_preds_lgb = np.mean(all_test_preds_lgb, axis=0)
    avg_preds_cat = np.mean(all_test_preds_cat, axis=0)
    
    # Blend the averaged predictions using the saved weights
    blended_preds = (optimal_weights[0] * avg_preds_xgb +
                     optimal_weights[1] * avg_preds_lgb +
                     optimal_weights[2] * avg_preds_cat)
                     
    return blended_preds




ORIGINAL_PATH = '/kaggle/input/bank-marketing-dataset-full/'


import pandas as pd

csv_path = ORIGINAL_PATH + "bank-full.csv"
original_df = pd.read_csv(csv_path, sep=";", quotechar='"')
original_df.head()


eda_summary(original_df)


# Prepare the data
X_original = original_df.drop(columns=[TARGET])
y_original = original_df[TARGET].map({"yes": 1, "no": 0}).astype("int8")


# Let's see how the full dataset model does on the original dataset
y_original_pred = predict_on_dataset(X_original)
print("Validation AUC:", roc_auc_score(y_original, y_original_pred))


test_preds = predict_on_dataset(test_df)

submission = pd.DataFrame({"id": test_ids, "y": test_preds})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")


submission.head()

