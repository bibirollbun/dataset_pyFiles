import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')


features_to_exclude = [
    'PurchDate',
    'VehYear',
    'Model',
    'Trim',
    'SubModel', 
    'WheelTypeID',
    'BYRNO',
    'VNZIP1',
    'VNST'
]

df_cleaned = df.drop(columns=features_to_exclude, axis=1, errors='ignore')

print(f"Original number of columns: {df.shape[1]}")
print(f"Number of columns after removal: {df_cleaned.shape[1]}")
print("\nFirst 5 rows of the cleaned data:")
print(df_cleaned.head())


df_cleaned.set_index('RefId', inplace=True)
print("\nFirst 5 rows of the updated data:")
print(df_cleaned.head())


from sklearn.model_selection import train_test_split

y = df_cleaned['IsBadBuy']
X = df_cleaned.drop('IsBadBuy', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData partitioning complete.")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")


import numpy as np

ranges = {
    'VehicleAge': (0, 30),
    'VehOdo': (0, 120000),
    'MMRAcquisitionAuctionAveragePrice': (800, 46000),
    'MMRAcquisitionAuctionCleanPrice': (1000, 46000),
    'MMRAcquisitionRetailAveragePrice': (1000, 46000),
    'MMRAcquisitonRetailCleanPrice': (1000, 46000),
    'MMRCurrentAuctionAveragePrice': (300, 46000),
    'MMRCurrentAuctionCleanPrice': (400, 46000),
    'MMRCurrentRetailAveragePrice': (800, 46000),
    'MMRCurrentRetailCleanPrice': (1000, 46000),
    'VehBCost': (1000, 46000),
    'WarrantyCost': (400, 8000)
}

print("\n--- Processing X_train ---")
print("Null counts before applying ranges:")
columns_to_check = [col for col in ranges.keys() if col in X_train.columns]
print(X_train[columns_to_check].isnull().sum())

for col, (min_val, max_val) in ranges.items():
    if col in X_train.columns:
        X_train[col] = X_train[col].where((X_train[col] > min_val) & (X_train[col] < max_val), np.nan)

print("\nNull counts after applying ranges:")
print(X_train[columns_to_check].isnull().sum())


print("\n--- Processing X_train.csv ---")
print("Before correction:")
print(f"Occurrences of 'NOT AVAIL': {(X_train['Color'] == 'NOT AVAIL').sum()}")
print(f"Null values: {X_train['Color'].isnull().sum()}")

X_train['Color'].replace('NOT AVAIL', np.nan, inplace=True)

print("\nAfter correction:")
print(f"Occurrences of 'NOT AVAIL': {(X_train['Color'] == 'NOT AVAIL').sum()}")
print(f"Null values: {X_train['Color'].isnull().sum()}")


# --- Process 'Color' Column ---
color_threshold = len(X_train) * 0.01
color_value_counts = X_train['Color'].value_counts()
colors_to_replace = color_value_counts[color_value_counts < color_threshold].index
X_train['Color'].replace(colors_to_replace, 'OTHER', inplace=True)

# --- Process 'Make' Column ---
make_threshold = len(X_train) * 0.01
make_value_counts = X_train['Make'].value_counts()
makes_to_replace = make_value_counts[make_value_counts < make_threshold].index
X_train['Make'].replace(makes_to_replace, 'OTHER', inplace=True)

print("--- Proof for 'Color' Column ---")
print(X_train['Color'].value_counts())



features_to_remove = []
print("\n--- Screening Continuous Variables (Coefficient of Variation < 0.1) ---")
continuous_cols = X_train.select_dtypes(include=['float64', 'int64']).columns

for col in continuous_cols:
    if X_train[col].mean() != 0:
        cv = X_train[col].std() / X_train[col].mean()
        print(f"'{col}': CV = {cv:.4f}")
        if cv < 0.1:
            features_to_remove.append(col)
            print(f"  -> Flagged for removal")


print("\n--- Screening Categorical Variables ---")
categorical_cols = X_train.select_dtypes(include=['object']).columns

for col in categorical_cols:
    mode_perc = X_train[col].value_counts(normalize=True).iloc[0]
    print(f"'{col}': Mode Percentage = {mode_perc:.4f}")
    if mode_perc > 0.99:
        features_to_remove.append(col)
        print(f"  -> Flagged for removal (Mode > 99%)")

    unique_perc = X_train[col].nunique() / len(X_train.dropna(subset=[col]))
    print(f"'{col}': Unique Percentage = {unique_perc:.4f}")
    if unique_perc > 0.90:
        features_to_remove.append(col)
        print(f"  -> Flagged for removal (Unique > 90%)")


print("\n" + "="*50)
print("Feature Screening of Training Set Complete.")

if features_to_remove:
    features_to_remove = list(set(features_to_remove))
    print(f"\nThe following {len(features_to_remove)} feature(s) are recommended for removal:")
    for feature in features_to_remove:
        print(f"- {feature}")
else:
    print("\nNo features were flagged for removal based on the specified criteria.")


from scipy.stats import chi2_contingency

train_df = X_train.join(y_train)


print("\n--- Processing 'PRIMEUNIT' ---")

if 'PRIMEUNIT' in train_df.columns:
    primeunit_test_df = train_df.dropna(subset=['PRIMEUNIT'])
    primeunit_contingency_table = pd.crosstab(primeunit_test_df['PRIMEUNIT'], primeunit_test_df['IsBadBuy'])

    chi2, p_value, _, _ = chi2_contingency(primeunit_contingency_table)
    print(f"P-value for 'PRIMEUNIT': {p_value:.4f}")

    if p_value < 0.05:
        print("Decision: Significant relationship found. Filling nulls with 'unknown'.")
        X_train['PRIMEUNIT'].fillna('unknown', inplace=True)
    else:
        print("Decision: No significant relationship. Dropping the column.")
        X_train.drop(columns=['PRIMEUNIT'], inplace=True)
else:
    print("Info: 'PRIMEUNIT' column not found, skipping.")


print("\n--- Processing 'AUCGUART' ---")

if 'AUCGUART' in train_df.columns:
    aucguart_test_df = train_df.dropna(subset=['AUCGUART'])
    aucguart_contingency_table = pd.crosstab(aucguart_test_df['AUCGUART'], aucguart_test_df['IsBadBuy'])
    chi2, p_value, _, _ = chi2_contingency(aucguart_contingency_table)
    print(f"P-value for 'AUCGUART': {p_value:.4f}")

    if p_value < 0.05:
        print("Decision: Significant relationship found. Filling nulls with 'unknown'.")
        X_train['AUCGUART'].fillna('unknown', inplace=True)
    else:
        print("Decision: No significant relationship. Dropping the column.")
        X_train.drop(columns=['AUCGUART'], inplace=True)
else:
    print("Info: 'AUCGUART' column not found, skipping.")





from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

X_train_processed = X_train.copy()

numeric_cols = X_train_processed.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X_train_processed.select_dtypes(include=['object']).columns

numeric_imputer = SimpleImputer(strategy='mean')
X_train_processed[numeric_cols] = numeric_imputer.fit_transform(X_train_processed[numeric_cols])

categorical_imputer = SimpleImputer(strategy='most_frequent')
X_train_processed[categorical_cols] = categorical_imputer.fit_transform(X_train_processed[categorical_cols])

X_train_processed = pd.get_dummies(X_train_processed, columns=categorical_cols, drop_first=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)
feature_names = X_train_processed.columns.tolist()


iso_forest = IsolationForest(contamination=0.01, random_state=42)
outlier_predictions = iso_forest.fit_predict(X_train_scaled)

outlier_indices = X_train.index[outlier_predictions == -1]
num_outliers = len(outlier_indices)
print(f"\nIsolation Forest identified {num_outliers} outliers (1% of the data).")


X_train.drop(outlier_indices, inplace=True)
y_train.drop(outlier_indices, inplace=True)

print(f"Number of rows in training data after removing outliers: {len(X_train)}")

iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X_train_scaled)

plt.figure(figsize=(20, 10)) 

plot_tree(
    iso_forest.estimators_[0], 
    feature_names=feature_names,
    filled=True,
    rounded=True,
    impurity=False, 
    proportion=True, 
    max_depth=4 
)

plt.title("Visualization of a Single Tree from the Isolation Forest (First 4 Levels)")
plt.show()


from sklearn.impute import SimpleImputer

price_cols = [
    'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
    'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
    'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
    'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice'
]
price_cols_in_df = [col for col in price_cols if col in X_train.columns]

null_price_counts = X_train[price_cols_in_df].isnull().sum(axis=1)
rows_to_drop_prices = null_price_counts[null_price_counts >= 4].index

X_train.drop(rows_to_drop_prices, inplace=True)
y_train.drop(rows_to_drop_prices, inplace=True)
print(f"\nStep 1: Removed {len(rows_to_drop_prices)} rows with 4+ nulls in price columns.")
print(f"Rows remaining: {len(X_train)}")


total_cols = len(X_train.columns)
null_total_counts = X_train.isnull().sum(axis=1)
rows_to_drop_total = null_total_counts[null_total_counts >= (total_cols * 0.5)].index

X_train.drop(rows_to_drop_total, inplace=True)
y_train.drop(rows_to_drop_total, inplace=True)
print(f"\nStep 2: Removed {len(rows_to_drop_total)} rows with >= 50% total null values.")
print(f"Rows remaining: {len(X_train)}")



numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

median_imputer = SimpleImputer(strategy='median')
X_train[numeric_cols] = median_imputer.fit_transform(X_train[numeric_cols])

mode_imputer = SimpleImputer(strategy='most_frequent')
X_train[categorical_cols] = mode_imputer.fit_transform(X_train[categorical_cols])

print("\nStep 3: Imputed all remaining missing values.")
print(f"Total null values in X_train after imputation: {X_train.isnull().sum().sum()}")

