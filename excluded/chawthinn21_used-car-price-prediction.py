from datetime import date
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import math
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Define file paths of the datasets
train_path = "/kaggle/input/playground-series-s4e9/train.csv"
test_path = "/kaggle/input/playground-series-s4e9/test.csv"

# Load datasets into pandas DataFrames
train_df_raw = pd.read_csv(train_path)
test_df_raw = pd.read_csv(test_path)
submission = pd.read_csv("/kaggle/input/playground-series-s4e9/sample_submission.csv")


# Display first 5 rows
train_df_raw.head()


# Display first 5 rows
test_df_raw.head()


# Check data types and non-null counts
train_df_raw.info()


# Check data types and non-null counts
test_df_raw.info()


# Check statistics summary
train_df_raw.describe()


# Check statistics summary
test_df_raw.describe()


# Check missing values for each column
train_df_raw.isna().sum()


# Check missing values for each column
test_df_raw.isna().sum()


# Visualize missing values
msno.matrix(train_df_raw, figsize=(12, 6), fontsize=10)


msno.heatmap(train_df_raw, figsize=(12, 6), fontsize=10)


# Verify unique values in the missing columns
missing_columns = train_df_raw.columns[train_df_raw.isna().any()]

for col in missing_columns:
    print(f"Unique values count of {col}: {train_df_raw[col].nunique()}")
    print(f"{train_df_raw[col].unique()}\n")


for col in missing_columns:
    print(f"Value counts of {col}: {train_df_raw[col].value_counts()}\n")


# Copy train and test df
train_df = train_df_raw.copy()
test_df = test_df_raw.copy()


# Strip and lower model values
original_n_unique = train_df['model'].nunique()
cleaned_n_unique = train_df['model'].astype(str).str.strip().str.lower().nunique()

print(f"Before cleaning: {original_n_unique} unique values")
print(f"After cleaning:  {cleaned_n_unique} unique values")


# Create function to clean and standardize columns
def clean_categorical_columns(df):
    df = df.copy()

    # Define cat cols
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns

    # Strip, lowercase, and clean string 'nan'
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace('nan', np.nan)  # Convert string 'nan' to real np.nan

    # Define valid fuel types
    valid_fuels = {'gasoline', 'hybrid', 'e85 flex fuel', 'plug-in hybrid'}

    # Replace invalid or missing fuel_type with 'unknown'
    if 'fuel_type' in df.columns:
        df['fuel_type'] = df['fuel_type'].apply(lambda x: x if x in valid_fuels else 'unknown')

    # Replace missing values in 'accident'
    if 'accident' in df.columns:
        df['accident'] = df['accident'].fillna('unknown')
        df['accident'] = df['accident'].replace({
            'at least 1 accident or damage reported': 'reported',
            'none reported': 'none'
        })

    # Replace missing values in 'clean_title'
    if 'clean_title' in df.columns:
        df['clean_title'] = df['clean_title'].fillna('unknown')

    return df


# Call function to clean both datasets
train_df = clean_categorical_columns(train_df)
test_df = clean_categorical_columns(test_df)


# Verify again
print(train_df.isna().sum())
print(test_df.isna().sum())


# Print unique values of each cat col
cat_cols = train_df.select_dtypes(include=['object', 'string']).columns

for col in cat_cols:
    print(f"\nTop unique values in '{col}':")
    print(train_df[col].value_counts(dropna=False).to_frame(name='count').head(10))
    print(f"\Bottom unique values in '{col}':")
    print(train_df[col].value_counts(dropna=False).to_frame(name='count').tail(10))


# Check full row duplicates
print(train_df.duplicated().sum())

# Check full row duplicates
print(test_df.duplicated().sum())


# Define columns to check (except id and price)
to_check = train_df.columns.difference(['id', 'price'])

# Check count of every duplicated row including first occurrence
print(train_df.duplicated(subset=to_check, keep=False).sum())

# Print duplicated rows
train_df[train_df.duplicated(subset=to_check, keep=False)]


# Keep the first unique row, and drop the rest
train_df = train_df[~train_df.duplicated(subset=to_check, keep='first')]

# Check duplicate counts again
print(train_df.duplicated(subset=to_check, keep=False).sum())


# Get current year using date.today().year
current_year = date.today().year
print(current_year)


# Check unique values of model year
train_df['model_year'].unique()


# Create new column to calculate car age
train_df['car_age'] = current_year - train_df['model_year']
train_df.head()


# Do the same for test_df
test_df['car_age'] = current_year - test_df['model_year']
test_df.head()


# Create new feature mileage per car_age
train_df['car_age'] = current_year - train_df['model_year']
train_df['car_age'] = train_df['car_age'].clip(lower=1)  # minimum 1 year

# Calculate mileage per year
train_df['mileage_per_year'] = train_df['milage'] / train_df['car_age']
print(train_df['mileage_per_year'].head())


# Do the same for test_df
test_df['car_age'] = current_year - test_df['model_year']
test_df['car_age'] = test_df['car_age'].clip(lower=1)
test_df['mileage_per_year'] = test_df['milage'] / test_df['car_age']
print(test_df['mileage_per_year'].head())


# Check the column names
print(train_df.columns)
print(test_df.columns)


# Drop irrelevant columns
train_df.drop(['model_year', 'id'], axis=1, inplace=True, errors='ignore')
test_df.drop(['model_year', 'id'], axis=1, inplace=True, errors='ignore')


# Verify columns again
print(train_df.columns)
print(test_df.columns)


categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns

print(train_df[categorical_cols].nunique().sort_values(ascending=False))


print("Top values (brand):")
print(train_df['brand'].value_counts().head(10))

print("\nRare or weird values (brand):")
print(train_df['brand'].value_counts().tail(10))


# Define threshold for filtering common brands
threshold = 1000

# Get frequency count of each brand in training set
brand_counts = train_df['brand'].value_counts()

# Get list of brands that meet or exceed the threshold
common_brands = brand_counts[brand_counts >= threshold].index.tolist()

# Replace rare brands with 'other' in both train and test sets
train_df['brand_grp'] = train_df['brand'].apply(lambda x: x if x in common_brands else 'other')
test_df['brand_grp'] = test_df['brand'].apply(lambda x: x if x in common_brands else 'other')


# Verify the new cols
print(train_df['brand_grp'].value_counts())
print("------------------")
print(test_df['brand_grp'].value_counts())


# Extract first two tokens as model_base
train_df['model_base'] = train_df['model'].str.split().str[:2].str.join(' ')
test_df['model_base'] = test_df['model'].str.split().str[:2].str.join(' ')

# Select top 100 frequent models from model_base
top_models = train_df['model_base'].value_counts().nlargest(100).index

# Group model_base
train_df['mdl_grp'] = train_df['model_base'].apply(lambda x: x if x in top_models else 'other')
test_df['mdl_grp'] = test_df['model_base'].apply(lambda x: x if x in top_models else 'other')


# Verify the new cols
print(train_df['mdl_grp'].value_counts())
print("------------------")
print(test_df['mdl_grp'].value_counts())


print("Top values (engine):")
print(train_df['engine'].value_counts().head(20))

print("\nRare or weird values (engine):")
print(train_df['engine'].value_counts().tail(20))


train_df[
    train_df['engine'].str.contains(r'\d+\.?\d*\s*l', case=False, na=False) &
    ~train_df['engine'].str.contains(r'hp', case=False, na=False)
]['engine'].unique()


# Function that will standardize engine column a
def clean_engine_column(df):
    return (
        df['engine']
        .astype(str)
        .str.lower()
        .str.replace('litre', 'l', regex=False) # convert litre to l
        .str.replace('liter', 'l', regex=False) # convert liter to l
        .str.replace(r'(\d+\.?\d*)\s*l', r'\1l', regex=True)  # normalize space before 'l'
        .str.strip()
    )


train_df['engine_clean'] = clean_engine_column(train_df)
test_df['engine_clean'] = clean_engine_column(test_df)


# Subset 1: Engine strings that contain 'hp'
df_with_hp = train_df[train_df['engine_clean'].str.contains('hp', case=False, na=False)]

# Subset 2: Engine strings that contain 'l' (liter) but NOT 'hp'
df_with_l_only = train_df[
    train_df['engine_clean'].str.contains(r'\d+\.?\d*l', case=False, na=False) &
    ~train_df['engine_clean'].str.contains('hp', case=False, na=False)
]

# Name the subsets
subset_hp = df_with_hp.copy()
subset_l = df_with_l_only.copy()


# Print total recs
print("No. of records in subset_hp", len(subset_hp))

# Print common values
print("Most common values:\n")
print(subset_hp['engine_clean'].value_counts().head(20))
print()
print("Least common values:\n")
print(subset_hp['engine_clean'].value_counts().tail(20))


# Print total recs
print("No. of records in subset_l", len(subset_l))

# Print common values
print("Most common values:\n")
print(subset_l['engine_clean'].value_counts().head(20))
print()
print("Least common values:\n")
print(subset_l['engine_clean'].value_counts().tail(20))


def get_text_after_engine_keyword(text):
    if pd.isna(text) or not isinstance(text, str):
        return np.nan
    text_lower = text.lower()

    # Prioritize "electric motor"
    match_electric = re.search(r'electric motor\s*(.*)', text_lower)
    if match_electric:
        return match_electric.group(1).strip()

    # Fallback to "engine"
    match_engine = re.search(r'engine\s*(.*)', text_lower)
    if match_engine:
        return match_engine.group(1).strip()

    return np.nan # If neither keyword is found


eng_desc_hp = subset_hp['engine_clean'].apply(get_text_after_engine_keyword)
eng_desc_l = subset_l['engine_clean'].apply(get_text_after_engine_keyword)
print(eng_desc_hp.value_counts(dropna=False))
print()
print(eng_desc_l.value_counts(dropna=False))


def extract_hp_features(text):
    """
    Extracts structured engine features from a raw engine description string,
    with enhanced parsing for fuel and engine types.

    Parameters:
        text (str): Raw engine description (e.g., '355.0hp 5.3l 8 cylinder engine gasoline fuel')

    Returns:
        pd.Series: A pandas Series with the following extracted fields:
            - engine_hp (float): Horsepower value
            - engine_size (float): Engine displacement in liters
            - engine_cyl (int): Number of cylinders
            - engine_layout (str): Layout type (e.g., 'v6', 'straight 6', 'i4')
            - engine_fuel_type_raw (str): The raw fuel description captured
            - engine_fuel_type_categorized (str): Categorized fuel type (e.g., 'Gasoline', 'Diesel', 'Electric', 'Hybrid', 'Other')
    """
    if pd.isna(text) or not isinstance(text, str):
        text_lower = "" # Handle NaN or non-string inputs gracefully
    else:
        text_lower = str(text).lower()

    # Extract basic features
    hp_match = re.search(r'(\d+\.?\d*)hp', text_lower)
    l_match = re.search(r'(\d+\.?\d*)l', text_lower)
    cyl_match = re.search(r'(\d+) cylinder', text_lower)
    layout_match = re.search(r'(v\d|straight \d|i\d)', text_lower)

    # Extract Fuel and Engine Type
    raw_fuel_description = np.nan
    categorized_fuel_type = np.nan

    # Define patterns
    fuel_phrase_map = {
        'plug-in electric/gas': 'hybrid',
        'gasoline/mild electric hybrid': 'hybrid',
        'gas/electric hybrid': 'hybrid',
        'electric fuel system': 'electric',
        'flex fuel capability': 'flex fuel',
        'diesel fuel': 'diesel',
        'gasoline fuel': 'gasoline',
        'hydrogen fuel': 'hydrogen'
    }

    sorted_phrases = sorted(fuel_phrase_map.keys(), key=len, reverse=True)

    for phrase in sorted_phrases:
        if phrase in text_lower:
            engine_fuel_type_raw = phrase
            categorized_fuel_type = fuel_phrase_map[phrase]
            break

    extracted_cyl = int(cyl_match.group(1)) if cyl_match else np.nan
    extracted_layout = layout_match.group(1) if layout_match else np.nan

    if pd.isna(extracted_cyl) and not pd.isna(extracted_layout):
        layout_num_match = re.search(r'\d+', extracted_layout)
        if layout_num_match:
            extracted_cyl = int(layout_num_match.group(0))

    return pd.Series({
        'engine_hp': float(hp_match.group(1)) if hp_match else np.nan,
        'engine_size': float(l_match.group(1)) if l_match else np.nan,
        'engine_cyl': extracted_cyl,
        'engine_layout': extracted_layout,
        'engine_fuel_type_raw': engine_fuel_type_raw,
        'engine_fuel_type_categorized': categorized_fuel_type
    })


def extract_engine_features(text):
    if pd.isna(text) or not isinstance(text, str):
        text_lower = ""
    else:
        text_lower = str(text).lower()

    # Extract Engine Features
    hp_match = re.search(r'(\d+\.?\d*)hp', text_lower)
    l_match = re.search(r'(\d+\.?\d*)l', text_lower)
    cyl_match = re.search(r'(\d+) cylinder', text_lower)
    # layout_match = re.search(r'(v\d|straight \d|i\d)', text_lower)
    layout_match = re.search(r'(v\d|straight \d|i\d|h\d|w\d|inline \d|boxer)', text_lower)
    valves_match = re.search(r'(\d{2})v', text_lower) # e.g., 16v, 24v, 32v
    inject_match = re.search(r'\b(gdi|pdi|mpfi|ddi)\b', text_lower) # e.g., gdi, mpfi
    valve_train_match = re.search(r'\b(dohc|sohc|ohv)\b', text_lower) # e.g., dohc, ohv
    aspiration_match = re.search(r'(twin turbo|turbo|supercharged|naturally aspirated)', text_lower) # Added 'naturally aspirated' for completeness

    # Initialize Fuel and Engine Type Variables
    engine_fuel_type_raw = np.nan
    categorized_fuel_type = np.nan

    # Define Fuel Phrase Mapping
    fuel_phrase_map = {
        'plug-in electric/gas': 'plug-in hybrid',
        'gasoline/mild electric hybrid': 'hybrid',
        'gas/electric hybrid': 'hybrid',
        'electric fuel system': 'electric',
        'flex fuel capability': 'flex fuel',
        'diesel fuel': 'diesel',
        'gasoline fuel': 'gasoline',
        'hydrogen fuel': 'hydrogen',
        'petrol fuel': 'gasoline',
        'ethanol fuel': 'flex fuel',
        'natural gas': 'cng/lpg',
        'cng': 'cng/lpg',
        'lpg': 'cng/lpg'
    }

    sorted_phrases = sorted(fuel_phrase_map.keys(), key=len, reverse=True)

    # Loop to Extract Categorized Fuel Type
    for phrase in sorted_phrases:
        if phrase in text_lower:
            engine_fuel_type_raw = phrase
            categorized_fuel_type = fuel_phrase_map[phrase]
            break

    # Refine engine_fuel_type_raw if still NaN after direct mapping
    if pd.isna(engine_fuel_type_raw) and not pd.isna(categorized_fuel_type) and categorized_fuel_type != 'Unknown':
        engine_fuel_type_raw = categorized_fuel_type.lower() + " type" if categorized_fuel_type not in ['Electric', 'Hydrogen', 'CNG/LPG'] else categorized_fuel_type.lower()


    # Handle engine_cyl inference from layout
    extracted_cyl = int(cyl_match.group(1)) if cyl_match else np.nan
    extracted_layout = layout_match.group(1) if layout_match else np.nan

    if pd.isna(extracted_cyl) and not pd.isna(extracted_layout):
        layout_num_match = re.search(r'\d+', extracted_layout)
        if layout_num_match:
            extracted_cyl = int(layout_num_match.group(0))

    # Return the Series with all extracted and inferred features
    return pd.Series({
        'engine_hp': float(hp_match.group(1)) if hp_match else np.nan,
        'engine_size': float(l_match.group(1)) if l_match else np.nan,
        'engine_cyl': extracted_cyl,
        'engine_layout': extracted_layout,
        'engine_fuel_type_raw': engine_fuel_type_raw,
        'engine_fuel_type_categorized': categorized_fuel_type,
        'engine_valves': int(valves_match.group(1)) if valves_match else np.nan,
        'engine_injection_type': inject_match.group(1) if inject_match else np.nan,
        'engine_valvetrain': valve_train_match.group(1) if valve_train_match else np.nan,
        'engine_aspiration': aspiration_match.group(1) if aspiration_match else np.nan
    })


# Apply function on subset_hp
extracted_features_hp = subset_hp['engine_clean'].apply(extract_engine_features)

# Print most common values across new features
for col in extracted_features_hp.columns:
    print(f"\nTop values for {col}:")
    print(extracted_features_hp[col].value_counts(dropna=False).head(20))


# Apply function on subset_l
extracted_features_l = subset_l['engine_clean'].apply(extract_engine_features)

# Print most common values across new features
for col in extracted_features_l.columns:
    print(f"\nTop values for {col}:")
    print(extracted_features_l[col].value_counts(dropna=False).head(20))


# Concatenate extracted features back to subset_hp
subset_hp_with_features = pd.concat([subset_hp, extracted_features_hp], axis=1)

# Compare extracted fuel type and fuel_type column
comparison_df = pd.concat([
    subset_hp['engine_clean'],
    subset_hp['fuel_type'],                              # Original fuel_type
    extracted_features_hp['engine_fuel_type_categorized'] # Extracted fuel type
], axis=1).dropna(subset=['engine_clean']) # dropna for 'engine_clean' just to focus on processed rows

# Create cross-tabulation
crosstb = pd.crosstab(comparison_df['fuel_type'], comparison_df['engine_fuel_type_categorized'], dropna=False)

# Visualize crosstb
plt.figure(figsize=(10, 8))
sns.heatmap(crosstb, annot=True, fmt='d', cmap='viridis', linewidths=.5, cbar_kws={'label': 'Count'})
plt.title('Original Fuel Type vs. Extracted Fuel Type Categories')
plt.xlabel('Extracted Fuel Type Category')
plt.ylabel('Original Fuel Type')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()


# Concatenate extracted features back to subset_l
subset_l_with_features = pd.concat([subset_l, extracted_features_l], axis=1)

# Compare extracted fuel type and fuel_type column
comparison_df = pd.concat([
    subset_l['engine_clean'],
    subset_l['fuel_type'],                              # Original fuel_type
    extracted_features_l['engine_fuel_type_categorized'] # Extracted fuel type
], axis=1).dropna(subset=['engine_clean']) # dropna for 'engine_clean' just to focus on processed rows

# Create cross-tabulation
crosstb = pd.crosstab(comparison_df['fuel_type'], comparison_df['engine_fuel_type_categorized'], dropna=False)

# Visualize crosstb
plt.figure(figsize=(10, 8))
sns.heatmap(crosstb, annot=True, fmt='d', cmap='viridis', linewidths=.5, cbar_kws={'label': 'Count'})
plt.title('Original Fuel Type vs. Extracted Fuel Type Categories')
plt.xlabel('Extracted Fuel Type Category')
plt.ylabel('Original Fuel Type')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()


def preprocess_engine_features(df, numerical_imputation_values=None):

    # Apply extract_engine_features to df
    extracted_features = df['engine_clean'].apply(extract_engine_features)

    # Concatenate the new features back to main DataFrame
    for col in extracted_features.columns:
        df[col] = extracted_features[col]

    # Define a mapping for original fuel types
    fuel_type_standardization_map = {
        'gasoline': 'gasoline',
        'diesel': 'diesel',
        'hybrid': 'hybrid',
        'e85 flex fuel': 'flex fuel',
        'electric': 'electric',
        'hydrogen': 'hydrogen',
        'natural gas': 'cng/lpg',
        'cng': 'cng/lpg',
        'lpg': 'cng/lpg',
        'plug-in hybrid': 'plug-in hybrid',
        'unknown': np.nan # Map 'unknown' in original to NaN so it can be filled
    }

    # Create a new 'final_fuel_type' column, using values from original 'fuel_type'
    df['final_fuel_type'] = df['fuel_type']

    # Map original fuel types to a standardized set
    df['final_fuel_type'] = df['final_fuel_type'].map(fuel_type_standardization_map)

    # Fill NaNs using extracted fuel type
    df['final_fuel_type'] = df['final_fuel_type'].fillna(df['engine_fuel_type_categorized'].squeeze())

    # Fill all missing values with unknown
    df['final_fuel_type'] = df['final_fuel_type'].fillna('unknown')
    df['engine_layout'] = df['engine_layout'].fillna('unknown')
    df['engine_injection_type'] = df['engine_injection_type'].fillna('unknown')
    df['engine_valvetrain'] = df['engine_valvetrain'].fillna('unknown')
    df['engine_aspiration'] = df['engine_aspiration'].fillna('unknown')

    return df


# Apply to both train_df and test_df
train_df_cleaned = preprocess_engine_features(train_df.copy())
test_df_cleaned = preprocess_engine_features(test_df.copy())


# Verify the columns
print(train_df_cleaned.columns)
print(test_df_cleaned.columns)


print("Top values (transmission):")
print(train_df['transmission'].value_counts().head(30))

print("\nRare or weird values (transmission):")
print(train_df['transmission'].value_counts().tail(10))


# Define function to group transmission
def simplify_transmission(val):
    val = val.lower()
    if 'manual' in val or 'm/t' in val:
        return 'manual'
    elif 'a/t' in val or 'automatic' in val or 'at' in val or 'dual shift' in val or 'overdrive' in val:
        return 'auto'
    elif 'cvt' in val:
        return 'cvt'
    else:
        return 'other'

# Apply function on both dataset
train_df_cleaned['trn_grp'] = train_df['transmission'].apply(simplify_transmission)
test_df_cleaned['trn_grp'] = test_df['transmission'].apply(simplify_transmission)


# Verify the new cols
print(train_df_cleaned['trn_grp'].value_counts())
print("------------------")
print(test_df_cleaned['trn_grp'].value_counts())


print("Top values (ext_col):")
print(train_df['ext_col'].value_counts().head(20))

print("\nRare or weird values (ext_col):")
print(train_df['ext_col'].value_counts().tail(10))


# Combine train and test color values
combined_colors = pd.concat([train_df['ext_col'], test_df['ext_col']], axis=0)

# Extract most frequent base color keywords
top_colors = combined_colors.value_counts().head(20).index.tolist()

# Flatten to base color groups from common names
ext_base_colors = []
for val in top_colors:
    for token in val.split():
        if token not in ext_base_colors:
            ext_base_colors.append(token)
    if len(ext_base_colors) >= 20:
        break

print("Auto-detected base colors:", ext_base_colors)


# Define base colors based on results above
ext_base_colors = [
    'black', 'white', 'gray', 'silver', 'blue', 'red', 'green',
    'gold', 'brown', 'orange'
]

# Create function to match into color groups
def match_col_group(val, base_colors):
    val = str(val).strip().lower()

    for base in base_colors:
        if base in val:
            return base

    return 'other'


# Apply function on both train_df and test_df
train_df_cleaned['ext_col_grp'] = train_df['ext_col'].apply(lambda x: match_col_group(x, ext_base_colors))
test_df_cleaned['ext_col_grp'] = test_df['ext_col'].apply(lambda x: match_col_group(x, ext_base_colors))


# Verify the new cols
print(train_df_cleaned['ext_col_grp'].value_counts())
print("------------------")
print(test_df_cleaned['ext_col_grp'].value_counts())


print("Top values (int_col):")
print(train_df['int_col'].value_counts().head(20))

print("\nRare or weird values (int_col):")
print(train_df['int_col'].value_counts().tail(10))


# Combine train and test color values
combined_colors = pd.concat([train_df['int_col'], test_df['int_col']], axis=0)

# Extract most frequent base color keywords
top_colors = combined_colors.value_counts().head(20).index.tolist()

# Flatten to base color groups from common names
int_base_colors = []
for val in top_colors:
    for token in val.split():
        if token not in int_base_colors:
            int_base_colors.append(token)
    if len(int_base_colors) >= 20:
        break

print("Auto-detected base colors:", int_base_colors)


# Re-define the base int_col
int_base_colors = [
    'black', 'beige', 'gray', 'brown', 'red', 'white',
    'orange', 'blue', 'silver', 'gold'
]

# Apply function on both train_df and test_df
train_df_cleaned['int_col_grp'] = train_df['int_col'].apply(lambda x: match_col_group(x, int_base_colors))
test_df_cleaned['int_col_grp'] = test_df['int_col'].apply(lambda x: match_col_group(x, int_base_colors))


# Verify the new cols
print(train_df_cleaned['int_col_grp'].value_counts())
print("------------------")
print(test_df_cleaned['int_col_grp'].value_counts())


print(train_df_cleaned.info())


print(test_df_cleaned.info())


train_df_cleaned.to_csv("train_df_cleaned.csv", index=False)
test_df_cleaned.to_csv("test_df_cleaned.csv", index=False)


# Load cleaned train and test sets
train_df_cleaned = pd.read_csv("train_df_cleaned.csv")
test_df_cleaned = pd.read_csv("test_df_cleaned.csv")


# Define target
target = 'price'

# Separate target and features
y = train_df_cleaned[target]
X = train_df_cleaned.drop(columns=[target])

# Split into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print(X_train.shape)
print(X_test.shape)


print(X_train.info())


print(X_train.isna().sum())


# Get list of cat cols
cat_cols_train = X_train.select_dtypes(include='object').columns.tolist()
cat_cols_train


unique_counts = X_train[cat_cols_train].nunique(dropna=False).sort_values(ascending=False)
print(unique_counts)


for col in cat_cols_train:
    unique_vals = X_train[col].unique()
    print(f"\nColumn: {col} ({len(unique_vals)} unique)")
    print(unique_vals)


from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Define columns to drop
cols_to_drop_before_ohe = [
    'brand', # Already grouped
    'model', # Raw high-cardinality col
    'model_base', # Already grouped
    'fuel_type', # Already transformed to final_fuel_type
    'transmission', # Already grouped
    'ext_col', # Already grouped
    'int_col', # Already grouped
    'engine', # Original raw engine string
    'engine_clean', # Cleaned engine string (features extracted from it)
    'engine_fuel_type_raw', # Raw types that are categorized later
    'engine_fuel_type_categorized' # Already transformed to final_fuel_type
]

# Identify remaining categorical columns
cat_cols = train_df.select_dtypes(include=['object', 'category']).columns
cat_cols_for_ohe = [col for col in cat_cols if col not in cols_to_drop_before_ohe]

def encode_ohe(train_df, test_df, cat_cols, drop_cols):
    """
    One-hot encodes categorical features in train_df and test_df.

    Parameters:
    - train_df (DataFrame): Training features
    - test_df (DataFrame): Test features (no target)
    - cat_cols (list): All candidate categorical columns
    - drop_cols (list): Categorical columns to exclude from encoding

    Returns:
    - train_encoded_df (DataFrame): OHE result on train_df
    - test_encoded_df (DataFrame): OHE result on test_df
    - ohe (OneHotEncoder): Fitted encoder
    - ohe_feature_names
    """

    # Prepare drop list for unknown/other categories
    drop_categories = []
    for col in cat_cols_for_ohe:
        categories = train_df[col].unique().tolist()
        if 'unknown' in categories:
            drop_categories.append(['unknown'])
        elif 'other' in categories:
            drop_categories.append(['other'])
        elif 'none' in categories:
            drop_categories.append(['none'])
        else:
            drop_categories.append(None)


    # Initialize encoder
    ohe = OneHotEncoder(
        drop=drop_categories,
        handle_unknown='ignore',
        sparse_output=False
    )

    # Fit and transform on train
    X_train_encoded = ohe.fit_transform(train_df[cat_cols_for_ohe])

    # Transform on test
    X_test_encoded = ohe.transform(test_df[cat_cols_for_ohe])

    # Get feature names
    ohe_feature_names = ohe.get_feature_names_out(cat_cols_for_ohe)

    return X_train_encoded, X_test_encoded, ohe, ohe_feature_names


# Apply one-hot encoding
X_train_encoded, X_test_encoded, ohe, ohe_feature_names = encode_ohe(
    train_df=X_train,
    test_df=X_test,
    cat_cols=cat_cols,
    drop_cols=cols_to_drop_before_ohe
)


ohe_feature_names


# Verify the shapes
print(X_train_encoded.shape)
print(X_test_encoded.shape)


# Select numeric columns
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# Use describe to check features by highest std
X_train[numeric_cols].describe().T.sort_values(by='std', ascending=False)


# Plot distributions of numeric features
X_train[numeric_cols].hist(bins=30, figsize=(12, 8))
plt.suptitle("Distributions of Numerical Columns (before transformation)", fontsize=18)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.boxplot(x=X_train[col], ax=axes[i])
    axes[i].set_title(col)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False) # set visible to false to hide extra plots

plt.suptitle("Boxplots of Numerical Columns", fontsize=18)
plt.tight_layout()
plt.show()


# Get list of numerical cols
num_cols_train = X_train.select_dtypes(include=[np.number]).columns.tolist()
num_cols_train


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Define numerical columns explicitly
numerical_cols = ['milage',
    'car_age',
    'mileage_per_year',
    'engine_hp',
    'engine_size',
    'engine_cyl',
    'engine_valves']

# Verify nan counts
print(X_train[numerical_cols].isna().sum())
print(X_test[numerical_cols].isna().sum())

def impute_and_scale_numerical(train_df, test_df, numerical_cols):
    """
    Imputes and scales selected numerical features using median and standard scaling.
    Returns numpy arrays of scaled values for train and test sets.
    """
    # Imputer and scaler
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    # Fit on train, transform both
    X_train_imputed = imputer.fit_transform(train_df[numerical_cols])
    X_test_imputed = imputer.transform(test_df[numerical_cols])

    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    return X_train_scaled, X_test_scaled, imputer, scaler


# Apply imputation and scaling
X_train_scaled, X_test_scaled, num_imputer, num_scaler = impute_and_scale_numerical(
    train_df=X_train,
    test_df=X_test,
    numerical_cols=numerical_cols
    )


# Verify the nan values
print("Train NaNs:", np.isnan(X_train_scaled).sum())
print("Test NaNs:", np.isnan(X_test_scaled).sum())


# Verify the shapes
print(X_train_scaled.shape)
print(X_test_scaled.shape)


def combine_encoded_and_scaled_features(X_train_encoded, X_test_encoded,
                                        X_train_scaled, X_test_scaled,
                                        ohe, numerical_cols):
    """
    Horizontally stacks encoded categorical features with scaled numerical features.

    Returns:
        X_train_final (np.ndarray): Combined features for training set
        X_test_final (np.ndarray): Combined features for test set
        feature_names (list): Combined feature names
    """
    # Stack features
    X_train_final = np.hstack([X_train_encoded, X_train_scaled])
    X_test_final = np.hstack([X_test_encoded, X_test_scaled])

    # Get feature names
    cat_feature_names = ohe.get_feature_names_out()
    feature_names = list(cat_feature_names) + numerical_cols

    return X_train_final, X_test_final, feature_names


# Combine encoded and scaled features
X_train_final, X_test_final, final_feature_names = combine_encoded_and_scaled_features(
    X_train_encoded, X_test_encoded,
    X_train_scaled, X_test_scaled,
    ohe, numerical_cols
)


# Wrap numpy array as dataframe
X_train_final_df = pd.DataFrame(X_train_final, columns=final_feature_names)
X_test_final_df = pd.DataFrame(X_test_final, columns=final_feature_names)

# Verify the final dataframe
print(X_train_final_df.columns)


# Check rows
X_train_final_df.head()


# Check rows
X_test_final_df.head()


print(y_train.shape)
print(y_train.value_counts())


# Plot Distribution before transform
plt.figure(figsize=(6, 5))
sns.histplot(y_train, bins=50, kde=True)
plt.title("Distribution of Target Variable (before transform)")
plt.xlabel("Price")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot boxplot to check outliers
sns.boxplot(x=train_df['price'])
plt.title("Boxplot of Target Variable (Price)")
plt.show()


# Check number of price outliers
q1 = train_df['price'].quantile(0.25)
q3 = train_df['price'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = train_df[train_df['price'] > upper_bound]
print(f"Number of high-end price outliers: {len(outliers)}")


# Apply log1p transformation to target
y_train_scaled = np.log1p(y_train)

# Plot Distribution after tranform
plt.figure(figsize=(6, 5))
sns.histplot(y_train_scaled, bins=50, kde=True)
plt.title("Distribution of Target Variable (after log1p transform)")
plt.xlabel("log(price + 1)")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()


# Apply log1p transformation to y_test
y_test_scaled = np.log1p(y_test)


# Convert y_train_scaled to series
y_train_scaled_series = pd.Series(y_train_scaled, name="target")

# Concatenate with features
X_train_final_with_target = pd.concat([X_train_final_df.reset_index(drop=True), y_train_scaled_series.reset_index(drop=True)], axis=1)

# Compute correlations
corr_matrix = X_train_final_with_target.corr().abs()

# Get correlation with target
target_corr = corr_matrix['target'].sort_values(ascending=False).drop('target')


# Print top positive and top negative correlations
print(target_corr.head(15))
print("--------------------")
print(target_corr.tail(15))


# Get top features
top_20_corr_values = target_corr.abs().sort_values(ascending=False).head(20)
top_20_features = top_20_corr_values.index.tolist()


# Plot top 20 features correlated with Price
plt.figure(figsize=(12, 6))
sns.barplot(x=top_20_corr_values,
            y=top_20_features)
plt.title("Top 20 Features Correlated with Price")
plt.xlabel("Absolute Correlation")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# Create a subset dataframe using top 20 features
top20_corr_df = X_train_final_with_target[top_20_features + ['target']]

# Plot heatmap
plt.figure(figsize=(15,10))
sns.heatmap(top20_corr_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix", fontsize=18)
plt.tight_layout()
plt.show()


def show_high_corr_pairs(df, threshold=0.8):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [
        (col, row, upper.loc[row, col])
        for col in upper.columns
        for row in upper.index
        if not pd.isna(upper.loc[row, col]) and upper.loc[row, col] > threshold
    ]
    return high_corr_pairs


high_corr_pairs = show_high_corr_pairs(X_train_final_df, threshold=0.9)
for col1, col2, corr in high_corr_pairs:
    print(f"{col1} ↔ {col2} = {corr:.2f}")


high_corr_features_to_drop = ["accident_reported"]


X_train_final_df = X_train_final_df.drop(columns=high_corr_features_to_drop)
X_test_final_df = X_test_final_df.drop(columns=high_corr_features_to_drop)


print("Number of columns:", X_train_final_df.shape[1])
print("Number of columns:", X_test_final_df.shape[1])


# Check key relationships between car_age_scaled and price_log
plt.figure(figsize=(8, 5))
sns.regplot(data=X_train_final_with_target, x='car_age', y='target', scatter_kws={'alpha': 0.3}, line_kws={"color": "red"})
plt.title("Car Age (scaled) vs. Price (log)")
plt.tight_layout()
plt.show()


# Check key relationships between milage_log and price_log
plt.figure(figsize=(8, 5))
sns.regplot(data=X_train_final_with_target, x='milage', y='target', scatter_kws={'alpha': 0.3}, line_kws={"color": "red"})
plt.title("Mileage (log) vs. Price (log)")
plt.tight_layout()
plt.show()


# Plot boxplot to check accident_reported vs price_log
plt.figure(figsize=(6, 4))
sns.boxplot(data=X_train_final_with_target, x='accident_reported', y='target')
plt.title("Accident Reported vs. Price (log)")
plt.tight_layout()
plt.show()


# Plot boxplot to check clean_title_yes vs. price_log
plt.figure(figsize=(6, 4))
sns.boxplot(data=X_train_final_with_target, x='clean_title_yes', y='target')
plt.title("Clean Title vs. Price (log)")
plt.tight_layout()
plt.show()


# Select numeric columns excluding binary/dummy features
numeric_cols =['milage',
 'car_age',
 'mileage_per_year',
 'engine_hp',
 'engine_size',
 'engine_cyl',
 'engine_valves']

# Set layout
n_cols = 2
n_rows = math.ceil(len(numeric_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
axes = axes.flatten()

# Plot histograms
for i, col in enumerate(numeric_cols):
    sns.histplot(data=X_train_final_with_target, x=col, bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")

# Hide unused axes
for ax in axes[len(numeric_cols):]:
    ax.set_visible(False)

plt.tight_layout()
plt.suptitle("Histograms of Numerical Features", fontsize=18, y=1.02)
plt.show()



# Plot scatterplot for engine_hp_scaled vs price_log
plt.figure(figsize=(8, 5))
sns.regplot(data=X_train_final_with_target, x='engine_hp', y='target',
            scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
plt.title("Engine Horsepower (scaled) vs. Price (log)")
plt.tight_layout()
plt.show()


# Plot scatterplot for engine_size vs price_log
plt.figure(figsize=(8, 5))
sns.regplot(data=X_train_final_with_target, x='engine_size', y='target',
            scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
plt.title("Engine Size (scaled) vs. Price (log)")
plt.tight_layout()
plt.show()


# Replace white spaces in column names
X_train_final_df.columns = X_train_final_df.columns.str.replace(' ', '_')
X_test_final_df.columns = X_test_final_df.columns.str.replace(' ', '_')


# Convert back to NumPy arrays
X_train_final = X_train_final_df.values
X_test_final = X_test_final_df.values


print(X_train_final.shape)
print(X_test_final.shape)
print(y_train_scaled.shape)
print(y_test_scaled.shape)


# Fit PCA without limiting components
pca = PCA(random_state=42)
pca.fit(X_train_final)

# Get cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Find the number of components that explain >= 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components explaining 95% variance: {n_components_95}")


# Reduce dimensionality coputed principal components
pca = PCA(n_components=n_components_95, random_state=42)
X_train_pca = pca.fit_transform(X_train_final)
X_test_pca = pca.transform(X_test_final)

print(X_train_pca.shape, X_test_pca.shape)


# Initialize LinearRegression
lr = LinearRegression()

# Train the model
lr.fit(X_train_final, y_train_scaled)


# Initialize LinearRegression
lr_pca = LinearRegression()

# Train the model
lr_pca.fit(X_train_pca, y_train_scaled)


# Initialize Ridge Regression
ridge = Ridge(alpha=1.0)

# Train the model
ridge.fit(X_train_final, y_train_scaled)


# Initialize Ridge Regression
ridge_pca = Ridge(alpha=1.0)

# Train the model
ridge_pca.fit(X_train_pca, y_train_scaled)


# Initialize XGBRegressor
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)

# Train the model
xgb.fit(X_train_final, y_train_scaled)


# Initialize XGBRegressor
xgb_pca = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)

# Train the model
xgb_pca.fit(X_train_pca, y_train_scaled)


# Initialize LightGBM Regressor
lgbm = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)

# Train the model
lgbm.fit(X_train_final, y_train_scaled)


# Initialize LightGBM Regressor
lgbm_pca = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)

# Train the model
lgbm_pca.fit(X_train_pca, y_train_scaled)


# Define function to evaluate models
def evaluate_model(y_true, y_pred, model_name, dataset):

    # Get r2 score
    r2 = r2_score(y_true, y_pred)

    # Get mae
    mae = mean_absolute_error(y_true, y_pred)

    # Get rmse
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Print model name, dataset, and corresponding metrics
    print(f"{model_name} ({dataset} set):")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}\n")


# Evaluate all models
models = {
    "Linear Regression (PCA)": lr_pca,
    "Linear Regression (No PCA)": lr,
    "Ridge Regression (PCA)": ridge_pca,
    "Ridge Regression (No PCA)": ridge,
    "XGBoost (PCA)": xgb_pca,
    "XGBoost (No PCA)": xgb,
    "LightGBM (PCA)": lgbm_pca,
    "LightGBM (No PCA)": lgbm
}


# Define mapping for PCA vs Non-PCA features
feature_sets = {
    "PCA": (X_train_pca, X_test_pca, None),  # no names in PCA
    "No PCA": (X_train_final_df, X_test_final_df, X_train_final_df.columns.tolist())
}

# Evaluate each model
results = []

for name, model in models.items():
    fs_key = "PCA" if "(PCA)" in name else "No PCA"
    X_train_current, X_test_current, feature_names = feature_sets[fs_key]

    # If model supports feature names and they're available, use DataFrame
    if feature_names is not None and hasattr(model, "feature_name_") is False:
        X_train_input = pd.DataFrame(X_train_current, columns=feature_names)
    else:
        X_train_input = X_train_current

    model.fit(X_train_input, y_train_scaled)

    y_train_pred = model.predict(X_train_current)
    y_test_pred = model.predict(X_test_current)

    r2_train = r2_score(y_train_scaled, y_train_pred)
    r2_test = r2_score(y_test_scaled, y_test_pred)
    mae_test = mean_absolute_error(y_test_scaled, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test_scaled, y_test_pred))

    results.append({
        "Model": name,
        "R2 Train": r2_train,
        "R2 Test": r2_test,
        "MAE Test": mae_test,
        "RMSE Test": rmse_test,
        "Overfit Gap": r2_train - r2_test
    })


# Convert to DataFrame and sort
results_df = pd.DataFrame(results).sort_values("RMSE Test", ascending=True)
display(results_df)


# Find the best model based on lowest RMSE
best_row_idx = results_df['RMSE Test'].idxmin()
best_model_name = results_df.loc[best_row_idx, 'Model']
best_model = models[best_model_name]

print("Best model based on RMSE Test:", best_model_name)
print(best_model)


from lightgbm import plot_importance
import matplotlib.pyplot as plt

if "No PCA" in best_model_name:
    plot_importance(best_model,
                    max_num_features=20,
                    importance_type='gain',
                    figsize=(12, 6))
    plt.title(f"Top 20 Feature Importances ({best_model_name})")
    plt.tight_layout()
    plt.show()
else:
    print(f"{best_model_name} used PCA — cannot plot feature importances.")


# Save model using pickle
with open(f'best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)


with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe, f)
print("ohe saved.")

with open('num_imputer.pkl', 'wb') as f:
    pickle.dump(num_imputer, f)
print("num_imputer saved.")

with open('num_scaler.pkl', 'wb') as f:
    pickle.dump(num_scaler, f)
print("num_scaler saved.")


final_feature_names


OHE_PATH = "ohe.pkl"
NUM_IMPUTER_PATH = "num_imputer.pkl"
NUM_SCALER_PATH = "num_scaler.pkl"
BEST_MODEL_PATH = "best_model.pkl"
TEST_DATA_PATH = "test_df_cleaned.csv"
SUBMISSION_FILE_NAME = "submission.csv"

# Create the same numerical values defined ealier
numerical_cols = ['milage',
    'car_age',
    'mileage_per_year',
    'engine_hp',
    'engine_size',
    'engine_cyl',
    'engine_valves']

# Load test data
test_df_cleaned = pd.read_csv(TEST_DATA_PATH)

# Load preprocessing artifacts
with open(OHE_PATH, "rb") as f:
    ohe = pickle.load(f)

with open(NUM_IMPUTER_PATH, "rb") as f:
    num_imputer = pickle.load(f)

with open(NUM_SCALER_PATH, "rb") as f:
    num_scaler = pickle.load(f)

with open(BEST_MODEL_PATH, "rb") as f:
    best_model = pickle.load(f)


# Define cols that were dropped in the final training/ test sets
high_corr_features_to_drop = ["accident_reported"]

# OHE
X_kaggle_encoded = ohe.transform(test_df_cleaned[ohe.feature_names_in_])
X_kaggle_encoded_df = pd.DataFrame(X_kaggle_encoded, columns=ohe.get_feature_names_out())
print("Encoded OHE shape:", X_kaggle_encoded_df.shape)

# Drop high-corr categorical features
dropped_ohe_cols = [col for col in high_corr_features_to_drop if col in X_kaggle_encoded_df.columns]
X_kaggle_encoded_df.drop(columns=dropped_ohe_cols, inplace=True)
print("Dropped OHE col:", dropped_ohe_cols)
print("Encoded OHE shape:", X_kaggle_encoded_df.shape)

# Store cleaned encoded matrix
X_kaggle_encoded_cleaned = X_kaggle_encoded_df.values

# Impute and scale numerics
numerical_cols_cleaned = [col for col in numerical_cols if col in test_df_cleaned.columns]
X_kaggle_imputed = num_imputer.transform(test_df_cleaned[numerical_cols_cleaned])
X_kaggle_scaled = num_scaler.transform(X_kaggle_imputed)
print("Scaled cleaned shape:", X_kaggle_scaled.shape)

# Combine final input
X_kaggle_final = np.hstack([X_kaggle_encoded_cleaned, X_kaggle_scaled])
print("Final input shape to model:", X_kaggle_final.shape)
print("Expected model input features:", best_model.n_features_)

# Predict and inverse log
y_kaggle_pred_log = best_model.predict(X_kaggle_final)
y_kaggle_pred = np.expm1(y_kaggle_pred_log)
y_kaggle_pred_rounded = np.round(y_kaggle_pred, 3)


submission['price'] = y_kaggle_pred_rounded  # Overwrite placeholder 0s
submission.to_csv('submission.csv', index=False)

