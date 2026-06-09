import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_columns", None)


print("ğŸ“¥ Loading datasets...")
properties_2016 = pd.read_csv('../input/zillow-prize-1/properties_2016.csv', low_memory=False)
properties_2017 = pd.read_csv('../input/zillow-prize-1/properties_2017.csv', low_memory=False)
train_2016 = pd.read_csv('../input/zillow-prize-1/train_2016_v2.csv', low_memory=False)
train_2017 = pd.read_csv('../input/zillow-prize-1/train_2017.csv', low_memory=False)
sample_submission = pd.read_csv('../input/zillow-prize-1/sample_submission.csv', low_memory=False)
print("âœ… Data loaded!")


properties_2016.head()


properties_2017.head()


# Display basic information about the datasets
print("Properties 2016 shape:", properties_2016.shape)
print("Properties 2017 shape:", properties_2017.shape)
print("Train 2016 shape:", train_2016.shape)
print("Train 2017 shape:", train_2017.shape)
print("Sample Submission shape:", sample_submission.shape)


def explore_data(df, name):
    """
    Function to explore a dataset.
    - Prints basic information
    - Shows missing values percentage
    - Displays numerical feature distributions
    """
    print(f"\nğŸ”� Exploring Dataset: {name}")
    print("-" * 50)
    
    # General info
    print(f"Shape: {df.shape}")
    print("\nColumns & Data Types:")
    print(df.dtypes.value_counts())
    
    # Missing values analysis
    missing_percent = df.isnull().mean() * 100
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
    

    if not missing_percent.empty:
        print("\nğŸ“‰ Missing Values (%):")
        print(missing_percent)  # Show columns with missing values
        
        plt.figure(figsize=(10, 4))
        sns.barplot(x=missing_percent.index, y=missing_percent.values)
        plt.xticks(rotation=90)
        plt.ylabel("Missing Values (%)")
        plt.title(f"Missing Data in {name}")
        plt.show()
    
    # Distribution of numerical features
    df.hist(figsize=(20, 20), bins=30)
    plt.suptitle(f"Numerical Feature Distributions - {name}", fontsize=14)
    plt.show()



explore_data(properties_2016, "Properties 2016")


# Function to drop columns with high missing values
def drop_high_missing_cols(df, name, threshold=0.8):
    """
    Drops columns with missing values above a certain threshold.
    :param df: DataFrame to process
    :param name: Name of the dataset for printing
    :param threshold: Proportion of missing values to decide column removal
    :return: Cleaned DataFrame
    """
    missing_percent = df.isnull().mean()
    cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
    
    print(f"ğŸ›‘ Dropping {len(cols_to_drop)} columns from {name} (>{threshold*100}% missing values)")
    
    return df.drop(columns=cols_to_drop)

# Apply function to both properties datasets
properties_2016 = drop_high_missing_cols(properties_2016, "Properties 2016")
properties_2017 = drop_high_missing_cols(properties_2017, "Properties 2017")

# Display shape of the new datasets
print(f"Properties 2016 shape: {properties_2016.shape}")
print(f"Properties 2017 shape: {properties_2017.shape}")




properties_2016.dtypes


def compare_datasets(df1, df2, name1='Dataset 1', name2='Dataset 2'):
    """
    Compares two datasets to find similarities and differences.
    - Checks if they have the same columns.
    - Identifies columns that exist in one but not the other.
    - Compares summary statistics.
    - Checks if they have the same parcelId values.
    """
    print(f"\nğŸ”� Comparing {name1} and {name2}")
    print("-" * 50)
    
    # Check if columns are the same
    cols1, cols2 = set(df1.columns), set(df2.columns)
    common_cols = cols1.intersection(cols2)
    diff_cols1 = cols1 - cols2
    diff_cols2 = cols2 - cols1
    
    print(f"âœ… Common columns: {len(common_cols)}")
    print(f"â�Œ Columns in {name1} but not in {name2}: {len(diff_cols1)} -> {diff_cols1}")
    print(f"â�Œ Columns in {name2} but not in {name1}: {len(diff_cols2)} -> {diff_cols2}")
    
    # Compare basic statistics of common columns
    common_df1 = df1[list(common_cols)].describe()
    common_df2 = df2[list(common_cols)].describe()
    
    print("\nğŸ“Š Summary Statistics of Common Columns:")
    diff_stats = common_df1.compare(common_df2)
    if diff_stats.empty:
        print("âœ… No statistical differences found between common columns!")
    else:
        print(diff_stats)
    
    # Check parcelId differences
    if 'parcelid' in common_cols:
        parcel_diff = set(df1['parcelid']) ^ set(df2['parcelid'])
        print(f"\nğŸ“Œ ParcelId Differences: {len(parcel_diff)} unique parcelIds found in one dataset but not the other.")
    else:
        print("âš  parcelid column not found in both datasets.")




compare_datasets(train_2016, train_2017, 'train 2016', 'train 2017')


compare_datasets(properties_2016, properties_2017, 'train 2016', 'train 2017')


def analyze_variable_types(df, name):
    """
    Analyzes the types of variables in the dataset.
    - Counts numerical, categorical, and other types of variables.
    """
    print(f"\nğŸ”� Analyzing Variable Types in {name}")
    print("-" * 50)
    
    var_types = df.dtypes.value_counts()
    print(df.dtypes)
    print(var_types)
    
    # Plot distribution of variable types
    plt.figure(figsize=(8, 5))
    var_types.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f"Variable Type Distribution in {name}")
    plt.xlabel("Data Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()
    
# Run analysis on both properties datasets
analyze_variable_types(properties_2016, "Properties 2016")
analyze_variable_types(properties_2017, "Properties 2017")



properties_2016.head(1)


properties_2017.head(10)


# Columns to drop
drop_cols = [
    'calculatedbathnbr', 'fullbathcnt', 'latitude', 'longitude', 
    'propertycountylandusecode', 'censustractandblock', 'regionidneighborhood', 
    'unitcnt', 'taxvaluedollarcnt', 'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'assessmentyear', 'airconditioningtypeid',
    'heatingorsystemtypeid', 'buildingqualitytypeid', 'propertyzoningdesc'
]

# Drop the columns
properties_2016_cleaned = properties_2016.drop(columns=drop_cols, errors="ignore")
properties_2017_cleaned = properties_2017.drop(columns=drop_cols, errors="ignore")


properties_2016_cleaned.head(10)


missing_percent_v2 = properties_2016_cleaned.isnull().mean() * 100
print(missing_percent_v2)


# Define replacement strategies
fill_mode = [
    'regionidcity', 'regionidzip', 'fips', 'propertylandusetypeid', 
    'rawcensustractandblock', 'regionidcounty'
]
fill_median = [
    'lotsizesquarefeet', 'finishedsquarefeet12', 'yearbuilt', 
    'calculatedfinishedsquarefeet', 'taxamount', 'roomcnt', 
    'bathroomcnt', 'bedroomcnt'
]
fill_zero = ['numberofstories', 'garagecarcnt', 'garagetotalsqft']

# Apply mode filling
for col in fill_mode:
    properties_2017_cleaned[col] = properties_2017_cleaned[col].fillna(properties_2016[col].mode()[0])
    properties_2016_cleaned[col] = properties_2016_cleaned[col].fillna(properties_2016[col].mode()[0])
    
# Apply median filling
for col in fill_median:
    properties_2017_cleaned[col] = properties_2017_cleaned[col].fillna(properties_2016[col].median())
    properties_2016_cleaned[col] = properties_2016_cleaned[col].fillna(properties_2016[col].median())
    
# Apply zero filling
for col in fill_zero:
    properties_2017_cleaned[col] = properties_2017_cleaned[col].fillna(0)
    properties_2016_cleaned[col] = properties_2016_cleaned[col].fillna(0)

print("âœ… Missing values handled. Cleaned dataset")


properties_2017_cleaned.head(10)


def list_columns(df, name):
    """
    Prints all columns in the dataset along with their data types.
    """
    print(f"\nğŸ“Š Columns in {name}:")
    print("-" * 50)
    print(df.dtypes)
    print("\nTotal columns:", df.shape[1])

# List columns for both properties datasets
list_columns(properties_2016_cleaned, "Properties 2016")
list_columns(properties_2017_cleaned, "Properties 2017")


def merge_train_properties(train_df, properties_df, name):
    """
    Merges train data with property data on parcelid.
    """
    print(f"\nğŸ”— Merging {name} train data with properties data")
    merged_df = train_df.merge(properties_df, on='parcelid', how='left')
    print(f"âœ… Merged dataset shape: {merged_df.shape}")
    return merged_df

# Merge train datasets with properties datasets
train_2016_merged = merge_train_properties(train_2016, properties_2016_cleaned, "2016")
train_2017_merged = merge_train_properties(train_2017, properties_2017_cleaned, "2017")



plt.figure(figsize=(10, 6))

sns.kdeplot(train_2016['logerror'], label='2016', fill=True, alpha=0.5)
sns.kdeplot(train_2017['logerror'], label='2017', fill=True, alpha=0.5)

plt.title("Logerror Distribution for 2016 vs 2017")
plt.xlabel("Logerror")
plt.ylabel("Density")
plt.legend()
plt.show()



# Combine train datasets from 2016 and 2017 into one
print("\nğŸ”— Combining 2016 and 2017 train datasets")
train_properties = pd.concat([train_2016_merged, train_2017_merged], axis=0).reset_index(drop=True)
print(f"âœ… Combined dataset shape: {train_properties.shape}")



train_properties.head()


# Convert 'transactiondate' to YYYYMM format
train_properties['transactiondate'] = pd.to_datetime(train_properties['transactiondate'])
train_properties['transactiondate'] = train_properties['transactiondate'].dt.strftime('%Y%m').astype(int)

print("âœ… Converted 'transactiondate' to YYYYMM format.")



train_properties.head()


# ğŸ�¯ Drop outliers outside -0.4 and 0.4
min_log, max_log = -0.4, 0.4

train_properties_v2 = train_properties[
    (train_properties["logerror"] >= min_log) &
    (train_properties["logerror"] <= max_log)
]



from sklearn.model_selection import train_test_split

# Extract year and month for filtering
year_month = train_properties_v2['transactiondate']

# Split early months (January - September) from both years
early_data = train_properties_v2[year_month % 100 <= 9]

# Split late months (October - December) from 2016 only
late_2016 = train_properties_v2[(year_month // 100 == 2016) & (year_month % 100 >= 10)]

# Randomly select 80% of early_data for training, 20% for testing
train_main, test_main = train_test_split(early_data, test_size=0.2, random_state=42)

# Randomly select 10% of late_2016 for training, rest 90% goes to testing
late_2016_train, late_2016_test = train_test_split(late_2016, test_size=0.9, random_state=42)

# Combine train_main with 10% of late_2016 to form the final training set
train_final = pd.concat([train_main, late_2016_train]).reset_index(drop=True)

# Combine test_main with 90% of late_2016 to form the final testing set
test_final = pd.concat([test_main, late_2016_test]).reset_index(drop=True)

# Remove 'parcelid' from both train_final and test_final
train_final = train_final.drop(columns=['parcelid'])
test_final = test_final.drop(columns=['parcelid'])


print(f"âœ… Training set shape: {train_final.shape}")
print(f"âœ… Testing set shape: {test_final.shape}")



train_final.head()


test_final.head()


from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# Define features and target
X_train = train_final.drop(columns=["logerror"])
y_train = train_final["logerror"].values

X_test = test_final.drop(columns=["logerror"])
y_test = test_final["logerror"].values

# 1ï¸�âƒ£ Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2ï¸�âƒ£ Train Linear (Default Parameters)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predict & Evaluate
lr_preds_train = lr_model.predict(X_train_scaled)
lr_preds_test = lr_model.predict(X_test_scaled)

lr_mae_train = (abs(lr_preds_train - y_train).mean() * 100)
lr_mae_test = (abs(lr_preds_test - y_test).mean() * 100)

# 3ï¸�âƒ£ Train XGBoost Regressor (Default Parameters)
xgb_model = XGBRegressor(random_state=42, n_jobs=-1)
xgb_model.fit(X_train_scaled, y_train)

# Predict & Evaluate
xgb_preds_train = xgb_model.predict(X_train_scaled)
xgb_preds_test = xgb_model.predict(X_test_scaled)

xgb_mae_train = (abs(xgb_preds_train - y_train).mean() * 100)
xgb_mae_test = (abs(xgb_preds_test - y_test).mean() * 100)

# Print results
print(f"âœ… Linear Regression Train MAE Score: {lr_mae_train:.4f}")
print(f"âœ… Linear Regression Test MAE Score: {lr_mae_test:.4f}")
print(f"âœ… XGBoost Train MAE Score: {xgb_mae_train:.4f}")
print(f"âœ… XGBoost Test MAE Score: {xgb_mae_test:.4f}")



from sklearn.model_selection import RandomizedSearchCV, KFold

# Define features and target
X = train_final.drop(columns=['logerror'])  # Drop target column
y = train_final['logerror']  # Target variable

# Define 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define models and hyperparameter grids
models = {
    "XGBoost": {
        "model": XGBRegressor(n_jobs=4, random_state=42),
        "param_grid": {
            'n_estimators': [100, 200], 
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
    }
},
    "LinearRegression": {
        "model": LinearRegression(),
        "param_grid": {
            'fit_intercept': [True, False],
            'copy_X': [True, False]
        }
    }
}

# Run Randomized Search for each model
best_params = {}
best_models = {}
for model_name, model_info in models.items():
    print(f"ğŸ”� Running RandomizedSearchCV for {model_name}...")
    RS = RandomizedSearchCV(
        model_info["model"], model_info["param_grid"], n_iter=8, cv=cv, 
        scoring='neg_mean_absolute_error', n_jobs=4, random_state=42, verbose=2
    )
    RS.fit(X, y)
    best_params[model_name] = RS.best_params_
    best_models[model_name] = RS.best_estimator_
    print(f"âœ… Best {model_name} Parameters: {RS.best_params_}")



# Prepare test data (keep 'target_month' as a feature)
X_test = test_final.drop(columns=['logerror'], errors='ignore')
y_test = test_final['logerror']  # True target values for evaluation

# Make predictions with the optimized models
predictions = {}
mae_scores = {}
for model_name, model in best_models.items():
    predictions[model_name] = model.predict(X_test)
    mae_scores[model_name] = (abs(predictions[model_name] - y_test).mean() * 100)
    print(f"âœ… Predictions completed for {model_name} | MAE: {mae_scores[model_name]:.4f}")


# Convert predictions to DataFrame3
predictions_df = pd.DataFrame(predictions)

print("âœ… Predictions DataFrame created with 'target_month' included!")
predictions_df.head()


plt.figure(figsize=(8,5))
plt.hist(predictions_df["XGBoost"], bins=50, alpha=0.5, label="XGBoost")
plt.hist(predictions_df["LinearRegression"], bins=50, alpha=0.5, label="LinearRegression")
plt.xlabel("Logerror Predictions")
plt.ylabel("Frequency")
plt.legend()
plt.show()



sample_submission.head()


properties_2016_cleaned.head()


best_models


# Select the best model (XGBoost)
best_model = best_models['XGBoost']

# Prepare test data for 2016 and 2017
X_test_2016 = properties_2016_cleaned.drop(columns=['parcelid'], errors='ignore')
X_test_2017 = properties_2017_cleaned.drop(columns=['parcelid'], errors='ignore')

# Insert empty transactiondate column as the second column
X_test_2016.insert(0, 'transactiondate', np.nan)
X_test_2017.insert(0, 'transactiondate', np.nan)

# Create a new sample submission DataFrame
submission = pd.DataFrame()

# Assign ParcelId from properties_2016_cleaned
submission['ParcelId'] = properties_2016_cleaned['parcelid'].astype(int)


X_test_2016.head()


submission.head()


# Predict for 2016 months using properties_2016_cleaned
for col in ['201610', '201611', '201612']:
    X_test_2016['transactiondate'] = int(col)  # Update transaction date
    predictions = best_model.predict(X_test_2016)  # Make predictions
    submission[col] = predictions  # Store predictions for the corresponding month


submission.head()


# Predict for 2017 months using properties_2017_cleaned
predictions_2017 = pd.DataFrame()
predictions_2017['ParcelId'] = properties_2017_cleaned['parcelid'].astype(int)

for col in ['201710', '201711', '201712']:
    X_test_2017['transactiondate'] = int(col)  # Update transaction date
    predictions_2017[col] = best_model.predict(X_test_2017)  # Make predictions

# Merge predictions for 2017 using left join
submission = submission.merge(predictions_2017, on='ParcelId', how='left')


submission.head()


# Save the submission file
submission.to_csv("submission.csv", index=False)

