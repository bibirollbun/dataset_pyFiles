import joblib
import pandas as pd
import numpy as np

from collections import Counter


# Load the tuned LightGBM model from the Kaggle input path
best_lgbm_model = joblib.load("/kaggle/input/cs-home-credit-default-lgbm-model/best_lgbm_model.pkl")


print(type(best_lgbm_model))


# Load the application_test.csv file into df_application_test
df_application_test = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")

# Review the shape and header
print("Shape:", df_application_test.shape)
print("Columns:", df_application_test.columns.tolist())


# Display the first 5 rows
df_application_test.head()


# Replace infinite values with NaN in test dataset
df_application_test.replace([np.inf, -np.inf], np.nan, inplace=True)

print("Infinite values in df_application_test replaced with NaN.")


df_application_test.describe()


# Identify categorical columns in the test dataset
categorical_candidates_test = df_application_test.select_dtypes(include=['object']).columns.tolist()

print("Categorical columns in test dataset:")
print(categorical_candidates_test)


# Step 2: Identify numeric columns with low unique values (possible categorical) in test dataset
low_unique_counts_test = df_application_test.nunique()

numeric_categoricals_test = low_unique_counts_test[
    (low_unique_counts_test < 20) & (df_application_test.dtypes != 'object')
].index.tolist()

print("Numeric columns with low unique values in test dataset:")
print(numeric_categoricals_test)


# Step 3: Combine all categorical columns in test dataset
final_categorical_columns_test = categorical_candidates_test + numeric_categoricals_test

print("Final categorical columns in test dataset:")
print(final_categorical_columns_test)


# Step 4: Convert detected categorical columns in test dataset to 'category' dtype
if final_categorical_columns_test:
    df_application_test[final_categorical_columns_test] = df_application_test[final_categorical_columns_test].astype("category")
    print(f"Converted categorical columns in test dataset: {final_categorical_columns_test}")
else:
    print("No categorical columns detected in test dataset. No dtype conversion needed.")


# Step 5: Manual Verification
print("\nChecking final dtype distribution:")
print(df_application_test.dtypes.value_counts())


# Normalize dtype names to strings for test dataset
dtype_names_test = df_application_test.dtypes.apply(lambda x: str(x))

# Count how many columns of each dtype exist
print("Dtype distribution in test dataset:")
print(dtype_names_test.value_counts())


# Check for missing values in test dataset
pd.set_option('display.max_rows', None) 
print("Missing values in each column (test dataset):")
print(df_application_test.isnull().sum())
pd.reset_option('display.max_rows')


# Define missing value thresholds
low_threshold = 1     # Less than 1% missing
moderate_threshold = 20  # Between 1% and 20% missing
high_threshold = 50   # More than 50% missing 

# Calculate missing value percentage for df_application_test
missing_percent_test = (df_application_test.isnull().sum() / len(df_application_test)) * 100  

# Display missing percentages sorted from highest to lowest
print("Missing Value Percentages in Test Dataset:")
display(missing_percent_test[missing_percent_test > 0].sort_values(ascending=False).apply(lambda x: f"{x:.2f}%"))


# Define the consistent drop list based on training dataset
columns_to_drop = [
    'OWN_CAR_AGE', 'EXT_SOURCE_1', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BUILD_AVG',
    'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',
    'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG',
    'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE',
    'ENTRANCES_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
    'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
    'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMIN_MEDI',
    'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI',
    'NONLIVINGAREA_MEDI', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE'
]

# Drop consistently from both train and test
df_application_test.drop(columns=columns_to_drop, inplace=True)

print(f"Dropped {len(columns_to_drop)} columns consistently from both train and test datasets.")


# Fill numeric columns with median in test dataset
numeric_cols_test = df_application_test.select_dtypes(include=['int64', 'float64']).columns
df_application_test[numeric_cols_test] = df_application_test[numeric_cols_test].fillna(df_application_test[numeric_cols_test].median())

print("Filled numeric missing values with median in test dataset.")


# Fill categorical columns with mode in test dataset
categorical_cols_test = df_application_test.select_dtypes(include=['category']).columns

for col in categorical_cols_test:
    df_application_test[col] = df_application_test[col].fillna(df_application_test[col].mode()[0])

print("Filled categorical missing values with mode in test dataset.")


df_application_test.info()


for col in df_application_test.select_dtypes(include=['category']).columns:
    df_application_test[col] = df_application_test[col].astype(str)


from sklearn.preprocessing import LabelEncoder

for col in df_application_test.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_application_test[col] = le.fit_transform(df_application_test[col].astype(str))


print(df_application_test.dtypes.value_counts())


# Drop SK_ID_CURR from test dataset
X_test = df_application_test.drop(columns=['SK_ID_CURR'])


# Generate predictions
test_predictions = best_lgbm_model.predict(X_test)
print(test_predictions.shape)


# Probabilities (for Kaggle submission)
test_probabilities = best_lgbm_model.predict_proba(X_test)[:, 1]
print(test_probabilities.shape)
print("Test predictions generated successfully.")


print(type(best_lgbm_model))
print("Has predict_proba:", hasattr(best_lgbm_model, "predict_proba"))


print("First 10 probabilities:", test_probabilities[:10])


# Create submission DataFrame
submission = pd.DataFrame({
    "SK_ID_CURR": df_application_test["SK_ID_CURR"],
    "TARGET": test_probabilities
})


submission.head()


# Save to CSV
submission.to_csv("submission.csv", index=False)

# Preview first few rows
print(submission.head())


# Convert probabilities into hard class labels
test_hard_predictions = (test_probabilities >= 0.5).astype(int)

print("Shape:", test_hard_predictions.shape)
print("First 10 hard predictions:", test_hard_predictions[:20])

