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





# Importing essential libraries
import pandas as pd # For data manipulation and analysis
import numpy as np  # For numerical operations
import re           # For regular expressions (used in plate parsing)
import sys          # For system-specific parameters and functions (like path manipulation)
import os           # For interacting with the operating system (e.g., file paths)

# Scikit-learn for preprocessing and model selection utilities
from sklearn.preprocessing import StandardScaler, MinMaxScaler # For numerical scaling
from sklearn.impute import SimpleImputer # For handling missing values
from sklearn.model_selection import KFold, StratifiedKFold # For cross-validation strategies
from sklearn.ensemble import GradientBoostingRegressor # An example of a tree-based regressor
from sklearn.linear_model import Ridge # An example of a linear model
from sklearn.metrics import mean_squared_error # Metric for regression evaluation



# Load Data from Kaggle input path
print("Loading datasets...")
train = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')
print(f"Train data shape: {train.shape}")
print(f"Test data shape: {test.shape}")

# Add path to the folder containing supplemental_english.py
# This script contains external data like region codes and government plate information.
sys.path.append('kaggle/input/russian-car-plates-prices-prediction/')
import supplemental_english as supp
print("Loaded 'supplemental_english.py' for additional data.")

# Concatenate train and test datasets for homogeneous preprocessing
# A new column 'is_train' is added to differentiate between original train and test rows.
train['is_train'] = 1
test['is_train'] = 0
df = pd.concat([train, test], ignore_index=True)
print(f"Combined DataFrame shape: {df.shape}")
print("Train and Test data concatenated for consistent preprocessing.")



# Function to extract components from the car plate string
# A Russian car plate typically follows the pattern: Letter-3Digits-2Letters-RegionCode
def extract_components(plate):
    """
    Extracts the first letter, 3-digit number, last two letters, and region code
    from a Russian car plate string using regular expressions.
    """
    match = re.match(r'^([A-Z])(\d{3})([A-Z]{2})(\d{2,3})$', plate)
    if match:
        first_letter = match.group(1)
        number = match.group(2)
        last_letters = match.group(3)
        region_code = match.group(4)
        full_letters = first_letter + last_letters # Combine for a 'full letters' feature
        return first_letter, number, last_letters, region_code, full_letters
    return None, None, None, None, None

# Apply the plate extraction function to create new columns
print("Extracting plate components...")
df[['pre_lettre', 'numero', 'second_lettre', 'code_region', 'lettre_complet']] = \
    pd.DataFrame(df['plate'].apply(extract_components).tolist(), index=df.index)
print("Plate components extracted: first_letter, number, last_letters, region_code, full_letters.")

# Convert 'date' column to datetime objects
df['date'] = pd.to_datetime(df['date'])
print("Converted 'date' column to datetime objects.")

# Function to enrich temporal features from the 'date' column
def enrich_date_features(df):
    """
    Extracts various time-based features from a datetime column,
    including basic components and cyclical features.
    """
    # Basic Features: Year, month, day, day of week, week of year, quarter, total days from start, weekend flag
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int) # .isocalendar().week returns a UInt32
    df['quarter'] = df['date'].dt.quarter
    df['total_days'] = (df['date'] - df['date'].min()).dt.days # Days since the earliest date in the dataset
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int) # 5=Saturday, 6=Sunday
    df['day_name'] = df['date'].dt.day_name() # e.g., 'Monday', 'Tuesday'
    
    # Cyclical Features: Sine and cosine transformations for periodic features
    # These help models understand the cyclical nature without implying a linear relationship.
    df['weekday_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

df = enrich_date_features(df)
print("Date features enriched with basic and cyclical components.")




# Load supplemental_english.py content explicitly to parse REGION_CODES
# This ensures we can programmatically access the dictionary structure.
file_path = '/kaggle/input/russian-car-plates-prices-prediction/supplemental_english.py'
with open(file_path, 'r', encoding='utf-8') as file:
    python_content = file.readlines()

# Extracting REGION_CODES dictionary
# This logic parses the REGION_CODES variable from the Python file content.
region_start = [i for i, line in enumerate(python_content) if "REGION_CODES = {" in line][0]
bracket_count = 0
region_end = None

for i in range(region_start, len(python_content)):
    line = python_content[i]
    bracket_count += line.count("{") - line.count("}") # Track open/close brackets
    if bracket_count == 0:
        region_end = i
        break

region_lines = python_content[region_start:region_end+1]

# Create a DataFrame for regional codes by flattening the dictionary
region_codes = []
for line in region_lines[1:]: # Skip the first line which is 'REGION_CODES = {'
    if ":" in line:
        key, value = map(str.strip, line.split(":"))
        # Clean and split the value string to get individual codes
        value = value.rstrip(",").replace("[", "").replace("]", "").replace("\"", "").split(",")
        region_codes.extend([(key.strip('"'), code.strip()) for code in value if code.strip()]) # Add non-empty codes

region_codes_df = pd.DataFrame(region_codes, columns=["region_name", "region_code"])
region_codes_df['region_code'] = region_codes_df['region_code'].str.strip()
print("Parsed REGION_CODES from supplemental file.")

# Merge the main DataFrame with regional names based on 'code_region'
df = df.merge(region_codes_df, left_on='code_region', right_on='region_code', how='left')
# Drop the redundant 'region_code' from the merged df
df.drop(columns=['region_code'], inplace=True) 
print("Merged region names based on region codes.")

# Using REGION_CODES directly from 'supp' module for numeric mapping
def get_region_code_numeric(row):
    """Converts the region code to a numeric code based on the first code in REGION_CODES."""
    code_region = str(row['code_region']) # Ensure it's a string for lookup
    for region, codes in supp.REGION_CODES.items():
        if code_region in codes:
            # Return the first numeric code from the list for the region
            return int(codes[0]) 
    return -1 # Default value if code is not found

df['region_code_numeric'] = df.apply(get_region_code_numeric, axis=1)
print("Created 'region_code_numeric' feature.")

# Initialize governmental information columns with default values
df['is_government'] = 0
df['government_agency'] = None
df['forbidden_to_buy'] = False
df['road_advantage'] = False
df['significance_level'] = 0

# Improved function to extract governmental plate information using 'GOVERNMENT_CODES'
def get_government_info(row):
    """
    Retrieves information about governmental plates based on plate components
    and the 'GOVERNMENT_CODES' dictionary.
    """
    # Handle missing plate components gracefully
    if pd.isna(row['pre_lettre']) or pd.isna(row['numero']) or pd.isna(row['code_region']):
        return 0, None, False, False, 0
    
    first_letter = row['pre_lettre']
    numbers = int(row['numero']) if pd.notna(row['numero']) else -1
    region_code = row['code_region']
    
    # Iterate through the defined governmental codes
    for (letters, (start, end), code), (agency, forbidden, advantage, significance) in supp.GOVERNMENT_CODES.items():
        # Check if the plate matches any governmental pattern
        if first_letter == letters[0] and region_code == code and start <= numbers <= end:
            return 1, agency, bool(forbidden), bool(advantage), significance
    
    return 0, None, False, False, 0 # Default if not governmental

# Apply the function to each row to populate governmental features
print("Extracting governmental plate information...")
govt_info = df.apply(get_government_info, axis=1)
df['is_government'] = [info[0] for info in govt_info]
df['government_agency'] = [info[1] for info in govt_info]
df['forbidden_to_buy'] = [info[2] for info in govt_info]
df['road_advantage'] = [info[3] for info in govt_info]
df['significance_level'] = [info[4] for info in govt_info]
print("Governmental plate information extracted and new features created.")

# Rename columns for clarity and consistency
df.rename(columns={
    'pre_lettre': 'first_letter',
    'second_lettre': 'last_letters',
    'lettre_complet': 'full_letters',
    'numero': 'numbers',
    'code_region': 'region_code_original' # Renamed to avoid confusion with 'region_code_numeric'
}, inplace=True)
print("Columns renamed for better readability.")

# Convert 'numbers' to integer type, coercing errors to NaN and filling with 0
# This is crucial for numerical comparisons and calculations later.
df['numbers'] = pd.to_numeric(df['numbers'], errors='coerce').fillna(0).astype(int)
print("Converted 'numbers' column to integer type.")




# Replace missing 'government_agency' values with "Non-governmental"
df['government_agency'] = df['government_agency'].fillna('Non-governmental')
print("Missing 'government_agency' values filled with 'Non-governmental'.")

# Function to categorize agencies into more general groups
# This reduces the high cardinality of the 'government_agency' column.
def categorize_agency(agency):
    """
    Categorizes specific government agencies into broader, more general groups
    to simplify the feature.
    """
    if agency == 'Non-governmental':
        return 'Non-governmental'
    elif 'President' in agency:
        return 'Presidential'
    elif 'Police' in agency.lower() or 'Internal Affairs' in agency:
        return 'Police/Security'
    elif 'Government' in agency:
        return 'Government'
    elif 'Military' in agency or 'Army' in agency or 'Defense' in agency:
        return 'Military'
    elif 'Federal' in agency:
        return 'Federal Services'
    elif 'Judge' in agency or 'Court' in agency or 'Justice' in agency or 'prosecutor' in agency.lower():
        return 'Judicial'
    elif 'Administration' in agency:
        return 'Administration'
    else:
        return 'Other Governmental'

# Apply the categorization to create a new 'agency_category' feature
df['agency_category'] = df['government_agency'].apply(categorize_agency)
print("Government agencies categorized into broader groups.")

# Create binary variables for each agency category using one-hot encoding
# This converts the categorical feature into a numerical format suitable for models.
agency_dummies = pd.get_dummies(df['agency_category'], prefix='agency')
df = pd.concat([df, agency_dummies], axis=1)
print("One-hot encoded 'agency_category' into binary features.")

# Calculate average price per agency category for insights (only for training data)
if 'price' in df.columns: # Ensure 'price' exists before calculating
    agency_price = df[df['is_train'] == 1].groupby('agency_category')['price'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    print("\nAverage price per agency category (Training Data):")
    print(agency_price)





# Handle potential missing values in 'numbers' again just in case (though filled above)
df['numbers'] = df['numbers'].fillna(0)

# Check for repeated letters in the 'full_letters' (e.g., 'AAA', 'XXX')
# This feature indicates patterns that might be considered desirable.
df['has_repeated_letters'] = df['full_letters'].str.replace(r'(.)(?=.*\1)', '', regex=True).str.len() < df['full_letters'].str.len()
print("Created 'has_repeated_letters' feature.")

# Check for repeated digits in the 'numbers' (e.g., '111', '777')
# These are often considered "beautiful" or "prestigious" numbers.
df['has_repeated_numbers'] = df['numbers'].apply(
    lambda n: bool(re.search(r'(\d)\1', f"{int(n):03d}")) # Format to 3 digits (e.g., 7 -> 007)
)
print("Created 'has_repeated_numbers' feature.")

# Check for sequential digits (e.g., '123', '987')
# Another pattern that can indicate prestige.
df['has_sequential_numbers'] = df['numbers'].apply(
    lambda n: bool(re.search(r'123|234|345|456|567|678|789|987|876|765|654|543|432|321', f"{int(n):03d}"))
)
print("Created 'has_sequential_numbers' feature.")

# Check for mirror digits (e.g., '121', '303') or palindromic numbers (e.g., '111')
# These are also considered special patterns.
df['has_mirror_numbers'] = df['numbers'].apply(
    lambda n: (str(int(n))[0] == str(int(n))[-1]) or (str(int(n)) == str(int(n))[::-1])
)
print("Created 'has_mirror_numbers' feature.")

# Define a list of prestigious letter series (e.g., specific combinations like 'AAA', 'XXX')
prestigious_letter_series = ["AAA", "MMM", "EEE", "KKK", "OOO", "PPP", "CCC", "TTT", "XXX"]
df['is_beautiful_series'] = df['full_letters'].isin(prestigious_letter_series)
print("Created 'is_beautiful_series' feature based on prestigious letter combinations.")

# Define a list of prestigious number combinations (e.g., single digits, triple digits, hundreds)
prestigious_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 111, 222, 333, 444, 555, 666, 777, 888, 999,
                       100, 200, 300, 400, 500, 600, 700, 800, 900, 7] # 7 is often considered lucky
df['is_prestigious_number'] = df['numbers'].isin(prestigious_numbers)
print("Created 'is_prestigious_number' feature based on specific prestigious number patterns.")

# Calculate the complexity of letters based on the number of unique characters
# A lower complexity (e.g., 'AAA') might indicate simplicity and prestige.
df['letter_complexity'] = df['full_letters'].apply(
    lambda x: len(set(x)) if pd.notnull(x) else 0
)
print("Calculated 'letter_complexity' feature.")

# Create an overall prestige score by weighting different prestige-related features
# This combines multiple signals into a single numeric score.
df['prestige_score'] = (
    (df['is_beautiful_series'].astype(int) * 3) + # Higher weight for beautiful letter series
    (df['is_prestigious_number'].astype(int) * 2) + # Moderate weight for prestigious numbers
    (df['has_repeated_letters'].astype(int) * 1) +
    (df['has_repeated_numbers'].astype(int) * 1) +
    (df['has_sequential_numbers'].astype(int) * 1) +
    (df['has_mirror_numbers'].astype(int) * 1) +
    (df['significance_level'].fillna(0)) # Include governmental significance level
)
print("Calculated 'prestige_score' by combining various prestige indicators.")

# Convert 'prestige_score' to a categorical type for potential use in models/visualizations
df['prestige_score'] = df['prestige_score'].astype('category')
print("Converted 'prestige_score' to categorical type.")




# Frequency Encoding for the 'numbers' feature
# This replaces the number with its frequency of occurrence in the dataset.
freq_table = df['numbers'].value_counts().reset_index()
freq_table.columns = ['numbers', 'n']
freq_table['freq_enc'] = freq_table['n'] / freq_table['n'].sum()
freq_table['log_freq_enc'] = np.log1p(freq_table['freq_enc']) # Log transform for potential skewed distribution

# Merge frequency encodings back to the main DataFrame
df = df.merge(freq_table[['numbers', 'freq_enc', 'log_freq_enc']], 
              on='numbers', how='left')
df.rename(columns={'freq_enc': 'numbers_freq_enc', 
                  'log_freq_enc': 'numbers_log_freq_enc'}, inplace=True)
print("Applied Frequency Encoding to 'numbers' feature.")

# Target Encoding (Mean Encoding) for categorical features
# This technique replaces a categorical value with the mean of the target variable
# for that category. It's crucial to perform this only on the training data
# to avoid data leakage from the test set.
train_data = df[df['is_train'] == 1].copy()

# For regions: calculate mean price for each region name
region_mean_price = train_data.groupby('region_name')['price'].mean().reset_index()
region_mean_price.columns = ['region_name', 'region_mean_price']
df = df.merge(region_mean_price, on='region_name', how='left')
print("Target encoded 'region_name' with 'region_mean_price'.")

# For first letter: calculate mean price for each first letter
first_letter_mean_price = train_data.groupby('first_letter')['price'].mean().reset_index()
first_letter_mean_price.columns = ['first_letter', 'first_letter_mean_price']
df = df.merge(first_letter_mean_price, on='first_letter', how='left')
print("Target encoded 'first_letter' with 'first_letter_mean_price'.")

# For last letters: calculate mean price for each last letters combination
last_letters_mean_price = train_data.groupby('last_letters')['price'].mean().reset_index()
last_letters_mean_price.columns = ['last_letters', 'last_letters_mean_price']
df = df.merge(last_letters_mean_price, on='last_letters', how='left')
print("Target encoded 'last_letters' with 'last_letters_mean_price'.")

# Logarithmic transformation of the target variable 'price'
# This is a common practice in regression to make the target distribution more normal
# and reduce the impact of outliers, improving model performance.
df['log_price'] = np.log1p(df['price'])
print("Applied logarithmic transformation (log1p) to 'price' to create 'log_price'.")

# --- Newly Added Features for enhanced modeling ---

# Number Length and Uniqueness:
df['number_length'] = df['numbers'].apply(lambda x: len(str(x))) # Length of the numeric part
df['is_single_digit'] = (df['number_length'] == 1).astype(int) # Binary flag for single-digit numbers
print("Added 'number_length' and 'is_single_digit' features.")

# Frequency of letter + region combinations:
# This captures the popularity or rarity of specific plate patterns within regions.
df['letters_region'] = df['full_letters'] + "_" + df['region_code_original'].astype(str)
freq_lr = df['letters_region'].value_counts(normalize=True).to_dict()
df['letters_region_freq'] = df['letters_region'].map(freq_lr)
print("Calculated 'letters_region_freq' for letter-region combinations.")

# Relative Prestige Ranking:
# Convert prestige score to a rank, normalized between 0 and 1.
# This gives a relative measure of prestige across all plates.
from scipy.stats import rankdata
df['prestige_rank'] = rankdata(df['prestige_score'].astype(int), method='average') / len(df)
print("Created 'prestige_rank' based on 'prestige_score'.")

# Interaction Features:
df['letter_number_combo'] = df['full_letters'] + "_" + df['numbers'].astype(str)
# Interaction between 'is_government' and 'prestige_score'
df['is_gov_and_prestige'] = df['is_government'] * df['prestige_score'].astype(int)
print("Added 'letter_number_combo' and 'is_gov_and_prestige' interaction features.")

# Similarity with Known Plates (Textual Embedding using CountVectorizer):
# This attempts to capture patterns in letter sequences.
from sklearn.feature_extraction.text import CountVectorizer

# Using character n-grams to capture patterns like 'AA', 'AB', 'BA'
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,2))
# Apply to 'full_letters' (e.g., 'XAA', 'TMM')
letter_features = vectorizer.fit_transform(df['full_letters'].fillna(''))
# Note: 'letter_features' is a sparse matrix and needs to be integrated into the
# modeling pipeline if directly used. For now, it's generated for demonstration.
print(f"Generated textual features for 'full_letters' using CountVectorizer. Shape: {letter_features.shape}")

# Finer Geography:
# Flag common premium regions (e.g., major cities/oblasts) as a binary feature.
premium_regions = ['Moscow', 'Saint Petersburg', 'Moscow Oblast']
df['is_premium_region'] = df['region_name'].isin(premium_regions).astype(int)
print("Created 'is_premium_region' feature for major economic centers.")

# End of Feature Engineering section
print("\nFeature engineering complete. DataFrame is ready for model training.")
print(f"Final DataFrame shape after feature engineering: {df.shape}")




# Importing necessary libraries for modeling
import numpy as np
import pandas as pd
from sklearn.base import clone # For cloning estimators in cross-validation
from sklearn.compose import ColumnTransformer # To apply different transformers to different columns
from sklearn.pipeline import Pipeline # To chain multiple processing steps and a final estimator
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer, OneHotEncoder # Various encoding/discretization methods
from sklearn.model_selection import StratifiedKFold # Cross-validation strategy
from xgboost import XGBRegressor # Gradient Boosting Machine from XGBoost
from lightgbm import LGBMRegressor # Gradient Boosting Machine from LightGBM
from catboost import CatBoostRegressor # Gradient Boosting Machine from CatBoost
import category_encoders as ce # Advanced categorical encoders (install with: pip install category-encoders)

# General Parameters for reproducibility and control
SEED = 92       # Random seed for reproducibility
N_SPLITS = 10   # Number of folds for cross-validation

TARGET = 'log_price' # The logarithmically transformed target variable for modeling

# Columns to be dropped from the feature set (X) before training
# These include original identifiers, the original price, and engineered features
# that might be redundant or explicitly excluded from the model.
DROP_COLS = [
    'id', 'plate', 'price', 'log_price', 'is_train', # Essential IDs and target variables
    # Specific engineered features that might be dropped if they are redundant or not performing well:
    "is_number_000", "is_number_444", "is_number_222", "is_number_700", 
    "is_number_555", "quarter", "day_of_week", "is_weekend", # Time-based features
    "prestige_score", # Dropping the combined score if its components are used directly or if it leads to multicollinearity
    "is_number_300","is_number_333","is_number_400" # Potentially redundant number pattern flags
] 



# Separating the concatenated DataFrame back into original training and testing sets
# based on the 'is_train' flag.
train_df = df[df['is_train'] == 1].copy() # .copy() to avoid SettingWithCopyWarning
test_df = df[df['is_train'] == 0].copy()

# Defining the features (X) and the target (y) for the training set,
# and features for the test set (X_test).
# 'errors='ignore'' handles cases where a column in DROP_COLS might not exist, preventing errors.
X = train_df.drop(columns=DROP_COLS, errors='ignore')
y = train_df[TARGET].copy()
X_test = test_df.drop(columns=DROP_COLS, errors='ignore')

print("Data split into training and testing sets.")
print(f"Training features (X) shape: {X.shape}")
print(f"Training target (y) shape: {y.shape}")
print(f"Test features (X_test) shape: {X_test.shape}")


# This section is crucial for handling different data types dynamically.
# It automatically identifies numerical, boolean, and categorical columns,
# and further segments categorical columns by their cardinality to apply
# appropriate encoding strategies.

def detect_columns(X):
    """
    Detects and segments columns by their data type and cardinality.
    This helps in applying specific preprocessing steps to different column types.
    """
    bool_cols = [c for c in X.columns if X[c].dtype == 'bool'] # Identify boolean columns
    num_cols = [c for c in X.columns if X[c].dtype.kind in 'if' and c not in bool_cols] # Identify numerical (int/float) columns, excluding booleans
    cat_cols = [c for c in X.columns if c not in num_cols + bool_cols]  # Remaining columns are treated as categorical

    # Further segmentation of categorical columns by cardinality (number of unique values)
    # Different encoding strategies are optimal for different cardinalities.
    cat_low = [c for c in cat_cols if X[c].nunique() <= 20] # Low cardinality for One-Hot Encoding
    cat_mid = [c for c in cat_cols if 20 < X[c].nunique() <= 200] # Medium cardinality for Ordinal Encoding
    cat_high = [c for c in cat_cols if X[c].nunique() > 200] # High cardinality for Target Encoding

    print('\nColumn Summary ➜ Numerical:', len(num_cols),
          '| Boolean:', len(bool_cols),
          '| Low Cardinality Categorical:', len(cat_low),
          '| Medium Cardinality Categorical:', len(cat_mid),
          '| High Cardinality Categorical:', len(cat_high))

    return num_cols, bool_cols, cat_low, cat_mid, cat_high

# Apply the column detection function to the training features
num_cols, bool_cols, cat_low, cat_mid, cat_high = detect_columns(X)



# The `ColumnTransformer` is the core component here. It allows applying
# different transformations to different subsets of columns in parallel.
# This ensures that each column type is handled appropriately before feeding to the model.

preprocess = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols), # Numerical columns: 'passthrough' means no transformation
        ('bool', 'passthrough', bool_cols), # Boolean columns: 'passthrough' as they are already binary
        ('low', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_low),  # One-Hot Encoding for low cardinality categories
                                                                                        # 'handle_unknown='ignore'' prevents errors if new categories appear in test set
                                                                                        # 'sparse_output=False' returns a dense NumPy array
        ('mid', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_mid),  # Ordinal Encoding for medium cardinality categories
                                                                                                # Assigns a unique integer to each category.
                                                                                                # 'unknown_value=-1' handles unseen categories.
        ('high', ce.TargetEncoder(cols=cat_high, smoothing=0.2), cat_high)  # Target Encoding for high cardinality categories
                                                                            # Replaces category with mean of target. 'smoothing' helps prevent overfitting.
    ],
    remainder='drop',  # Drops any columns not explicitly specified in `transformers` (safer approach)
    n_jobs=-1  # Utilizes all available CPU cores for parallel processing during transformation
)
print("\nPreprocessing pipeline (ColumnTransformer) defined.")



# Optimized Hyperparameters for various Gradient Boosting Regressors.
# These parameters are typically found through hyperparameter optimization
# techniques like GridSearchCV, RandomizedSearchCV, or more advanced tools like Optuna.

# XGBoost Parameters
xgb_params = {
    'n_estimators': 1433,
    'max_depth': 12,
    'learning_rate': 0.01852160907217988,
    'subsample': 0.6786672470738663,
    'colsample_bytree': 0.46208650739218005,
    'reg_alpha': 0.017519138973638618,
    'reg_lambda': 0.2839310763317462,
    'gamma': 0.0033995958574628547,
    'tweedie_variance_power': 1.0869464555654937, # Tweedie objective is suitable for target variables with a skewed distribution and many zero values, which can be the case for prices.
    'objective': 'reg:tweedie',
    'n_jobs': -1,
    'random_state': SEED
}

# LightGBM Parameters
lgb_params = {
    'n_estimators': 999,
    'max_depth': 11,
    'learning_rate': 0.07607568555547708,
    'subsample': 0.6363036032688429,
    'colsample_bytree': 0.5072021102992719,
    'min_child_samples': 97,
    'reg_alpha': 0.16671454380081874,
    'reg_lambda': 0.6455320711051608,
    'n_jobs': -1,
    'random_state': SEED
}

# CatBoost Parameters
cat_params = {
    'iterations': 991,
    'depth': 10,
    'learning_rate': 0.06462213707942074,
    'l2_leaf_reg': 1.9289204888270515,
    'subsample': 0.7213225292844163,
    'bagging_temperature': 0.4361642090192932,
    'random_strength': 6.443179917768372,
    'min_data_in_leaf': 72,
    'loss_function': 'RMSE', # Root Mean Squared Error, common for regression tasks
    'verbose': 0, # Suppress training output for cleaner logs
    'random_state': SEED
}

# Dictionary of models to be trained. Easily extensible to include more models.
# Uncomment LGBM and CatBoost to include them in the ensemble.
models = {
    'XGB': XGBRegressor(**xgb_params),
    #'LGBM': LGBMRegressor(**lgb_params),
    #'CatBoost': CatBoostRegressor(**cat_params)
}
print("\nModels and their optimized hyperparameters defined.")

# Construct the full pipeline for each model: preprocessing + estimator
# Each pipeline handles all necessary data transformations before training the model.
pipelines = {name: Pipeline(steps=[('prep', preprocess), ('model', model)]) for name, model in models.items()}
print("Pipelines constructed: Preprocessing -> Model.")



# The Symmetric Mean Absolute Percentage Error (SMAPE) is often used in forecasting
# and is robust to zero values in the actuals. It's defined once to ensure consistency.

def smape(y_true, y_pred):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).
    Formula: (1/n) * Sum(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2)) * 100
    This metric handles cases where y_true or y_pred (or both) are zero.
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    
    # Handle division by zero: if denominator is zero (i.e., both y_true and y_pred are zero),
    # the corresponding term for SMAPE is defined as 0.0 to avoid NaN/Inf.
    smape_term = np.zeros_like(diff, dtype=float)
    non_zero_denom = denominator != 0 # Identify where denominator is not zero
    smape_term[non_zero_denom] = diff[non_zero_denom] / denominator[non_zero_denom]
    
    return np.mean(smape_term) * 100

print("\nSMAPE evaluation metric defined.")



# Stratified K-Fold cross-validation is used to ensure that each fold has a
# representative distribution of the target variable. This is especially important
# for skewed targets or when specific target ranges are more critical.

# Bin the target variable ('log_price') to create "strata" for StratifiedKFold.
# This effectively treats the regression problem as a classification problem for splitting purposes,
# ensuring similar target distributions across folds.
y_bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile') \
    .fit_transform(y.values.reshape(-1, 1)).astype(int).ravel()
print(f"\nTarget variable ('{TARGET}') binned into {y_bins.max() + 1} strata for stratification.")

# Initialize StratifiedKFold with the specified number of splits and random state.
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# Dictionaries to store out-of-fold (OOF) predictions and test predictions for each model.
# OOF predictions are used for ensemble weighting and final CV evaluation.
# Test predictions are accumulated across folds for final submission.
oof_preds = {name: np.zeros(len(y)) for name in models}
test_preds = {name: np.zeros(len(X_test)) for name in models}
feature_importances = {} # To store feature importances for each model (if available)

print('\n===== CROSS-VALIDATION TRAINING =====')
# Iterate through each defined model and perform cross-validation
for model_name, pipeline in pipelines.items():
    print(f"\nInitiating training for model: {model_name}...")
    try:
        # Loop through each fold generated by StratifiedKFold
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_bins), 1):
            print(f"  Fold {fold:02d}/{N_SPLITS}")
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx] # Training data for the current fold
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx] # Validation data for the current fold

            # Train the pipeline on the training data
            pipeline.fit(X_tr, y_tr)
            
            # Store out-of-fold (OOF) predictions for the validation set
            oof_preds[model_name][val_idx] = pipeline.predict(X_val)
            
            # Accumulate test predictions by averaging predictions from each fold
            # This is a simple form of ensemble averaging.
            test_preds[model_name] += pipeline.predict(X_test) / N_SPLITS

        # Calculate the overall cross-validation SMAPE for the current model
        # Predictions are converted back from log-price scale to original price scale for SMAPE calculation.
        cv_smape = smape(np.exp(y), np.exp(oof_preds[model_name]))
        print(f'⮕  Overall CV SMAPE for {model_name}: {cv_smape:.2f}%')

        # Extract feature importances from the trained model (if the model supports it)
        if hasattr(pipeline['model'], 'feature_importances_'):
            # For models like LightGBM, scikit-learn tree models
            feature_importances[model_name] = pipeline['model'].feature_importances_
        elif hasattr(pipeline['model'], 'get_booster'): 
            # For XGBoost, get_booster() allows accessing internal booster attributes
            feature_importances[model_name] = pipeline['model'].get_booster().get_score(importance_type='weight') 
            # 'importance_type' can be 'weight' (number of times a feature is used in splits), 'gain' (average gain across splits), 'cover', etc.
        else:
            feature_importances[model_name] = None # No direct importance available

    except Exception as e:
        print(f"Error during training of model {model_name}: {str(e)}")
        continue  # Skip to the next model in case of an error



# This ensembling method is simple yet effective: models that perform better
# (i.e., have a lower SMAPE on the validation set) are given higher weights
# in the final blended prediction.

# Calculate errors (SMAPE) for each model based on their OOF predictions
errors = {name: smape(np.exp(y), np.exp(oof_preds[name])) for name in models}

# Calculate inverse errors (1 / SMAPE). A lower SMAPE means a higher inverse error.
inv_errors = {k: 1 / v for k, v in errors.items()}

# Normalize inverse errors to get weights that sum to 1.
# These normalized weights determine each model's contribution to the final ensemble.
norm_weights = {k: v / sum(inv_errors.values()) for k, v in inv_errors.items()}

# Combine test predictions from individual models using the calculated normalized weights.
# The predictions are combined in the log-price space, then converted back to original price.
ensemble_preds_log = sum(test_preds[k] * norm_weights[k] for k in test_preds)
ensemble_preds = np.exp(ensemble_preds_log) # Convert back from log-price to original price scale

# Calculate the SMAPE of the ensemble model on the out-of-fold validation set.
# This gives an estimate of the ensemble's performance on unseen data.
ensemble_oof_log = sum(oof_preds[k] * norm_weights[k] for k in oof_preds)
ensemble_smape = smape(np.exp(y), np.exp(ensemble_oof_log))
print(f"\n⮕  Final Ensemble Model CV SMAPE: {ensemble_smape:.2f}%")

# Display the weights of each model in the ensemble, along with their individual SMAPE.
print("\nModel Weights in Ensemble:")
for model_name, weight in norm_weights.items():
    print(f"- {model_name}: {weight:.4f} (Individual OOF SMAPE: {errors[model_name]:.2f}%)")





# Final steps to prepare the predictions for submission to Kaggle.

# Apply clipping to the ensemble predictions to prevent unrealistic values.
# 'a_min=0' ensures no negative prices, while 'a_max=None' means no upper limit is applied here.
# Clipping helps to make predictions more robust to potential outliers or model errors.
ensemble_preds = np.clip(ensemble_preds, a_min=0, a_max=None)

# Create the submission DataFrame with 'id' and the final 'price' predictions.
submission = pd.DataFrame({
    'id': test_df['id'], # Ensure 'id' column is taken from the original test_df
    'price': ensemble_preds
})

# Save the submission file to a CSV in the required format.
submission.to_csv('submission.csv', index=False)
print('\n✅  Submission file "submission.csv" saved successfully.')




