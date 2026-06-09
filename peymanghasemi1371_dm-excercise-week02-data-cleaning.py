import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
print("DataFrame Info:")
print(df.info())


# Q1: Drop inappropriate features
drop_cols = [
    "PurchDate",   # Dates not directly useful
    "VehYear",     # VehicleAge already present
    "Model",       # Too many classes
    "Trim",        # Too many classes
    "SubModel",    # Too many classes
    "WheelTypeID", # WheelType is better
    "BYRNO",       # Just an ID
    "VNZIP1",      # Location info not directly useful
    "VNST"         # Location info not directly useful
]

df = df.drop(columns=drop_cols)



# Q2: Set RefId as index
df = df.set_index("RefId")



# Q3: Define target (y) and features (X), then set inputs as training features
from sklearn.model_selection import train_test_split

# Target
y = df["IsBadBuy"]
# Features
X = df.drop(columns=["IsBadBuy"])

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Inputs for training
inputs = X_train



# Separate categorical and continuous fields on inputs (X_train)
# Continuous (numeric) fields
continuous_fields = [
    "VehicleAge", 
    "VehOdo", 
    "MMRAcquisitionAuctionAveragePrice",
    "MMRAcquisitionAuctionCleanPrice",
    "MMRAcquisitionRetailAveragePrice",
    "MMRAcquisitonRetailCleanPrice",
    "MMRCurrentAuctionAveragePrice",
    "MMRCurrentAuctionCleanPrice",
    "MMRCurrentRetailAveragePrice",
    "MMRCurrentRetailCleanPrice",
    "VehBCost",
    "WarrantyCost"
]

# Categorical fields (including binary IsOnlineSale)
categorical_fields = [
    "Auction",
    "Make",
    "Color",
    "Transmission",
    "WheelType",
    "Nationality",
    "Size",
    "TopThreeAmericanName",
    "PRIMEUNIT",
    "AUCGUART",
    "IsOnlineSale"
]

# Separate inputs
numeric_inputs = inputs[continuous_fields]
categorical_inputs = inputs[categorical_fields]



# Q4: Set out-of-range continuous values to NaN on inputs (X_train)
import numpy as np

# Define ranges for continuous fields
ranges = {
    "VehicleAge": (0, 30),
    "VehOdo": (0, 120000),
    "MMRAcquisitionAuctionAveragePrice": (800, 46000),
    "MMRAcquisitionAuctionCleanPrice": (1000, 46000),
    "MMRAcquisitionRetailAveragePrice": (1000, 46000),
    "MMRAcquisitonRetailCleanPrice": (1000, 46000),
    "MMRCurrentAuctionAveragePrice": (300, 46000),
    "MMRCurrentAuctionCleanPrice": (400, 46000),
    "MMRCurrentRetailAveragePrice": (800, 46000),
    "MMRCurrentRetailCleanPrice": (1000, 46000),
    "VehBCost": (1000, 46000),
    "WarrantyCost": (400, 8000)
}

# Apply ranges on numeric_inputs of inputs (X_train)
for col, (low, high) in ranges.items():
    numeric_inputs[col] = numeric_inputs[col].where(
        (numeric_inputs[col] >= low) & (numeric_inputs[col] <= high),
        np.nan
    )





# Q5: Correct inconsistencies in categorical variables (inputs)
import numpy as np

# Convert 'NOT AVAIL' in 'Color' to NaN
categorical_inputs["Color"] = categorical_inputs["Color"].replace("NOT AVAIL", np.nan)
unique_colors = categorical_inputs["Color"].dropna().unique()
print("Unique color groups:", unique_colors)




# Q6: Group rare categories (<1%) in 'Color' and 'Make' as 'OTHER'
for col in ["Color", "Make"]:
    # Calculate frequency percentage
    freq = categorical_inputs[col].value_counts(normalize=True)
    # Categories less than 1%
    rare_categories = freq[freq < 0.01].index
    # Replace rare categories with 'OTHER'
    categorical_inputs[col] = categorical_inputs[col].replace(rare_categories, "OTHER")

# Check the updated groups and their percentages
for col in ["Color", "Make"]:
    print(f"\n{col} value counts (%):")
    print(categorical_inputs[col].value_counts(normalize=True) * 100)



# Q7: Feature screening with reporting removed columns (single output)

removed_columns = []

# 1. Continuous variables: remove features with CV < 0.1
min_cv = 0.1
cv_values = numeric_inputs.std() / numeric_inputs.mean()
low_cv_cols = cv_values[cv_values < min_cv].index.tolist()
removed_columns.extend(low_cv_cols)

# 2. Categorical variables: remove features where mode percentage > 99%
mode_threshold = 99
mode_percentage = categorical_inputs.apply(lambda x: x.value_counts().max() / len(x) * 100)
high_mode_cols = mode_percentage[mode_percentage > mode_threshold].index.tolist()
removed_columns.extend(high_mode_cols)

# 3. Categorical variables: remove features with >90% unique categories
unique_threshold = 90
distinct_percentage = categorical_inputs.drop(columns=high_mode_cols).apply(lambda x: x.dropna().nunique() / x.count() * 100)
high_unique_cols = distinct_percentage[distinct_percentage > unique_threshold].index.tolist()
removed_columns.extend(high_unique_cols)

print("Columns removed during feature screening:", removed_columns)



# Q8: Hypothesis test for 'PRIMEUNIT' and 'AUCGUART' vs target
from scipy.stats import chi2_contingency

variables_to_test = ["PRIMEUNIT", "AUCGUART"]

for col in variables_to_test:
    # Create contingency table (drop NaNs)
    contingency = pd.crosstab(categorical_inputs[col].dropna(), y_train.loc[categorical_inputs[col].dropna().index])
    
    # Only perform test if more than 1 category
    if contingency.shape[0] > 1:
        chi2, p, dof, expected = chi2_contingency(contingency)
        print(f"\n{col}: p-value = {p}")
        
        if p < 0.05:
            # Significant relationship: fill NaNs with 'unknown'
            categorical_inputs[col] = categorical_inputs[col].fillna("unknown")
            print(f"{col}: Null values replaced with 'unknown'")
        else:
            # Not significant: drop the variable
            categorical_inputs = categorical_inputs.drop(columns=[col])
            print(f"{col}: Dropped due to no significant relationship")
    else:
        # Only one category: drop
        categorical_inputs = categorical_inputs.drop(columns=[col])
        print(f"{col}: Dropped (only one category)")



# Q8b: Drop PRIMEUNIT and AUCGUART as they are considered low-quality
cols_to_drop = ["PRIMEUNIT", "AUCGUART"]
categorical_inputs = categorical_inputs.drop(columns=[col for col in cols_to_drop if col in categorical_inputs.columns])

print("Remaining categorical columns after dropping low-quality fields:")
print(list(categorical_inputs.columns))



# Check skewness and kurtosis on current inputs (before IsolationForest)
continuous_data = inputs[continuous_fields]

# Number of continuous columns
num_continuous = continuous_data.shape[1]
print("Number of continuous columns:", num_continuous)

# Calculate skewness and kurtosis
skewness = continuous_data.skew()
kurtosis = continuous_data.kurt()

# Combine into a summary DataFrame
summary_df = pd.DataFrame({
    "Skewness": skewness,
    "Kurtosis": kurtosis
})

print("\nSkewness and Kurtosis for each continuous column:")
print(summary_df)


