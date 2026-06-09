
!pip install ydata_profiling



import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')


# Ù†Ù…Ø§ÛŒØ´ Ø§Ø·Ù„Ø§Ø¹Ø§Øª Ú©Ù„ÛŒ Ø¯Ø±Ø¨Ø§Ø±Ù‡ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ØŒ Ù†ÙˆØ¹ Ø¯Ø§Ø¯Ù‡ Ùˆ ØªØ¹Ø¯Ø§Ø¯ Ù…Ù‚Ø§Ø¯ÛŒØ± Ù†Ø§Ù„
df.info()



import pandas as pd
from ydata_profiling import ProfileReport

# Ù�Ø±Ø¶: df Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ… Ø§ØµÙ„ÛŒØª Ù‡Ø³Øª

# Ø¬Ø¯Ø§ Ú©Ø±Ø¯Ù† Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ú©Ù…ÛŒ (Ø¹Ø¯Ø¯ÛŒ) Ùˆ Ú©ÛŒÙ�ÛŒ (ØºÛŒØ± Ø¹Ø¯Ø¯ÛŒ)
numeric_df = df.select_dtypes(include=['int64', 'float64'])
categorical_df = df.select_dtypes(include=['object', 'category'])

print("ğŸ“Š ØªØ¹Ø¯Ø§Ø¯ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¹Ø¯Ø¯ÛŒ:", numeric_df.shape[1])
print("ğŸ”¤ ØªØ¹Ø¯Ø§Ø¯ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ú©ÛŒÙ�ÛŒ:", categorical_df.shape[1])

# Ù¾Ø±ÙˆÙ�Ø§ÛŒÙ„ Ù�Ù‚Ø· Ø¨Ø±Ø§ÛŒ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ÛŒ Ú©Ù…ÛŒ
# profile_numeric = ProfileReport(numeric_df, title="Numeric Features EDA", explorative=True)
# profile_numeric.to_file("Numeric_EDA_Report.html")

# Ù¾Ø±ÙˆÙ�Ø§ÛŒÙ„ Ù�Ù‚Ø· Ø¨Ø±Ø§ÛŒ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ÛŒ Ú©ÛŒÙ�ÛŒ
# profile_categorical = ProfileReport(categorical_df, title="Categorical Features EDA", explorative=True)
# profile_categorical.to_file("Categorical_EDA_Report.html")



# Ø­Ø°Ù� Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ù†Ø§Ù…Ù†Ø§Ø³Ø¨
cols_to_drop = [
    'PurchDate', 'VehYear', 'Model', 'Trim', 'SubModel',
    'WheelTypeID', 'BYRNO', 'VNZIP1', 'VNST'
]

df.drop(columns=cols_to_drop, inplace=True, errors='ignore')


df.set_index('RefId', inplace=True)


y = df['IsBadBuy']
X = df.drop(columns=['IsBadBuy'])



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Ø¯ÛŒÚ©Ø´Ù†Ø±ÛŒ Ù…Ø­Ø¯ÙˆØ¯Ù‡ Ù…Ù†Ø·Ù‚ÛŒ Ø¨Ø±Ø§ÛŒ Ù‡Ø± Ø³ØªÙˆÙ†
valid_ranges = {
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

# Ø§Ø¹Ù…Ø§Ù„ Ù…Ø­Ø¯ÙˆØ¯ÛŒØªâ€ŒÙ‡Ø§ Ùˆ ØªØ¨Ø¯ÛŒÙ„ Ù…Ù‚Ø§Ø¯ÛŒØ± Ø®Ø§Ø±Ø¬ Ø§Ø² Ù…Ø­Ø¯ÙˆØ¯Ù‡ Ø¨Ù‡ NaN
for col, (min_val, max_val) in valid_ranges.items():
    X_train[col] = X_train[col].apply(lambda x: x if min_val <= x <= max_val else None)



for field, (min_val, max_val) in valid_ranges.items():
    out_of_range = X_train[(X_train[field] < min_val) | (X_train[field] > max_val)]
    print(f"{field}: {len(out_of_range)} Ù…Ù‚Ø§Ø¯ÛŒØ± Ø®Ø§Ø±Ø¬ Ø§Ø² Ù…Ø­Ø¯ÙˆØ¯Ù‡")



import pandas as pd

# 1ï¸�âƒ£ ØªØ¨Ø¯ÛŒÙ„ 'NOT AVAIL' Ø¨Ù‡ NaN
X_train['Color'] = X_train['Color'].replace('NOT AVAIL', pd.NA)

# 2ï¸�âƒ£ Ø§Ø¯ØºØ§Ù… Ú©Ù„Ø§Ø³â€ŒÙ‡Ø§ÛŒ Ú©Ù…â€ŒØªÚ©Ø±Ø§Ø± (<1%) Ø¯Ø± Ú¯Ø±ÙˆÙ‡ OTHER
def merge_rare_categories(series, threshold=0.01):
    freq = series.value_counts(normalize=True)
    rare_labels = freq[freq < threshold].index
    return series.apply(lambda x: 'OTHER' if x in rare_labels else x)

X_train['Color'] = merge_rare_categories(X_train['Color'])
X_train['Make']  = merge_rare_categories(X_train['Make'])



# Ø§Ù†ØªØ®Ø§Ø¨ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¹Ø¯Ø¯ÛŒ
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# Ø´Ù†Ø§Ø³Ø§ÛŒÛŒ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¨Ø§ Ø¶Ø±ÛŒØ¨ ØªØºÛŒÛŒØ± Ú©Ù… (<0.1)
low_variance_cols = [col for col in numeric_cols if X_train[col].std() / X_train[col].mean() < 0.1]

# Ø­Ø°Ù� Ø³ØªÙˆÙ†â€ŒÙ‡Ø§
X_train.drop(columns=low_variance_cols, inplace=True)

print("Continuous features dropped due to low variance:", low_variance_cols)



# Ø§Ù†ØªØ®Ø§Ø¨ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¯Ø³ØªÙ‡â€ŒØ§ÛŒ
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Ø´Ù†Ø§Ø³Ø§ÛŒÛŒ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¨Ø§ Ø¨ÛŒØ´ Ø§Ø² 99% Ù…Ù‚Ø¯Ø§Ø± ÛŒÚ©Ø³Ø§Ù†
highly_skewed_cols = [col for col in categorical_cols if (X_train[col].value_counts(normalize=True).max() > 0.99)]

# Ø­Ø°Ù� Ø³ØªÙˆÙ†â€ŒÙ‡Ø§
X_train.drop(columns=highly_skewed_cols, inplace=True)

print("Categorical features dropped (99% same):", highly_skewed_cols)



# Ø´Ù†Ø§Ø³Ø§ÛŒÛŒ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¯Ø³ØªÙ‡â€ŒØ§ÛŒ Ø¨Ø§ Ø¨ÛŒØ´ Ø§Ø² 90% Ù…Ù‚Ø§Ø¯ÛŒØ± ÛŒÚ©ØªØ§
high_unique_cols = [col for col in categorical_cols if (X_train[col].nunique() / len(X_train)) > 0.9]

# Ø­Ø°Ù� Ø³ØªÙˆÙ†â€ŒÙ‡Ø§
X_train.drop(columns=high_unique_cols, inplace=True)

print("Categorical features dropped (90% unique):", high_unique_cols)



from scipy.stats import fisher_exact
import pandas as pd

for col in ['PRIMEUNIT','AUCGUART']:
    if col in X_train.columns:
        # Ù�Ù‚Ø· Ø±Ø¯ÛŒÙ�â€ŒÙ‡Ø§ÛŒ ØºÛŒØ± Null
        temp_df = pd.concat([X_train[col], y_train], axis=1).dropna()
        
        if temp_df.empty:
            print(f"{col}: no non-null data, skipping")
            continue
        
        # Ø¬Ø¯ÙˆÙ„ 2x2
        contingency_table = pd.crosstab(temp_df[col], temp_df['IsBadBuy'])
        
        if contingency_table.shape == (2,2):
            # Ø¢Ø²Ù…ÙˆÙ† Fisher
            _, p = fisher_exact(contingency_table)
            print(f"{col} -> p-value (Fisher): {p:.4f}")
            
            if p < 0.05:
                X_train[col] = X_train[col].fillna('UNKNOWN')
                print(f"{col}: significant association, missing replaced with 'UNKNOWN'")
            else:
                X_train.drop(columns=[col], inplace=True)
                print(f"{col}: no significant association, column dropped")
        else:
            print(f"{col}: not a 2x2 table, skipping Fisher")



import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ… ÙˆØ±ÙˆØ¯ÛŒ
inputs_iso = X_train.copy()

# Ø­Ø°Ù� Ø±Ø¯ÛŒÙ�â€ŒÙ‡Ø§ÛŒÛŒ Ú©Ù‡ Ù…Ù‚Ø§Ø¯ÛŒØ± NaN Ø¯Ø§Ø±Ù†Ø¯
inputs_iso = inputs_iso.dropna()

# Ø´Ù†Ø§Ø³Ø§ÛŒÛŒ Ø®ÙˆØ¯Ú©Ø§Ø± Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¹Ø¯Ø¯ÛŒ Ùˆ Ø¯Ø³ØªÙ‡â€ŒØ§ÛŒ
continuous_fields = inputs_iso.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_fields = inputs_iso.select_dtypes(include=['object']).columns.tolist()

# Ù…Ù‚ÛŒØ§Ø³â€ŒØ¨Ù†Ø¯ÛŒ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¹Ø¯Ø¯ÛŒ
scaler = StandardScaler()
inputs_iso[continuous_fields] = scaler.fit_transform(inputs_iso[continuous_fields])

# Ú©Ø¯Ú¯Ø°Ø§Ø±ÛŒ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¯Ø³ØªÙ‡â€ŒØ§ÛŒ
for col in categorical_fields:
    inputs_iso[col] = LabelEncoder().fit_transform(inputs_iso[col])

# Ø§Ø¬Ø±Ø§ÛŒ IsolationForest
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(inputs_iso)

# Ù¾ÛŒØ´â€ŒØ¨ÛŒÙ†ÛŒ Outlier
outliers = clf.predict(inputs_iso)

# Ø§Ø¶Ø§Ù�Ù‡ Ú©Ø±Ø¯Ù† Ø³ØªÙˆÙ† Outlier
inputs_iso['outlier'] = outliers

# Ù†Ù…Ø§ÛŒØ´ Ø¯ÛŒØªØ§ Ø¨Ø§ Ø§Ø·Ù„Ø§Ø¹Ø§Øª Outlier
print(inputs_iso.head())

# Ù…Ø­Ø§Ø³Ø¨Ù‡ Ø¯Ø±ØµØ¯ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ÛŒ Ù¾Ø±Øª
percentage_outliers = (outliers[outliers == -1].shape[0] / len(outliers)) * 100
print(f"Percentage of outliers: {percentage_outliers:.2f}%")



# Ù¾ÛŒØ¯Ø§ Ú©Ø±Ø¯Ù† Ø§ÛŒÙ†Ø¯Ú©Ø³ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ÛŒ Ù¾Ø±Øª
outlier_index = inputs_iso[inputs_iso['outlier'] == -1].index

# Ø­Ø°Ù� Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ÛŒ Ù¾Ø±Øª Ø§Ø² Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ… ÙˆØ±ÙˆØ¯ÛŒ Ø§ØµÙ„ÛŒ
inputs_outprep = X_train.drop(outlier_index)

# Ø­Ø°Ù� Ù‡Ù…Ø§Ù† Ø§ÛŒÙ†Ø¯Ú©Ø³â€ŒÙ‡Ø§ Ø§Ø² y_train
y_train_outprep = y_train.drop(outlier_index)

# ØªØ±Ú©ÛŒØ¨ Ø¯ÙˆØ¨Ø§Ø±Ù‡ ÙˆØ±ÙˆØ¯ÛŒâ€ŒÙ‡Ø§ Ùˆ Ù‡Ø¯Ù� Ø¨Ø±Ø§ÛŒ Ø¯ÛŒØªØ§ Ø¢Ù…Ø§Ø¯Ù‡
train_outprep = pd.concat([inputs_outprep, y_train_outprep], axis=1)

# Ù†Ù…Ø§ÛŒØ´ Ú†Ù†Ø¯ Ø±Ø¯ÛŒÙ� Ø§ÙˆÙ„
print(train_outprep.head())



# Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ù…Ø±ØªØ¨Ø· Ø¨Ø§ Ù‚ÛŒÙ…Øª
price_columns = [
    'MMRAcquisitionAuctionAveragePrice',
    'MMRAcquisitionAuctionCleanPrice',
    'MMRAcquisitionRetailAveragePrice',
    'MMRAcquisitonRetailCleanPrice',
    'MMRCurrentAuctionAveragePrice',
    'MMRCurrentAuctionCleanPrice',
    'MMRCurrentRetailAveragePrice',
    'MMRCurrentRetailCleanPrice'
]

# Ø´Ù†Ø§Ø³Ø§ÛŒÛŒ Ø±Ø¯ÛŒÙ�â€ŒÙ‡Ø§ÛŒÛŒ Ø¨Ø§ Û´ ÛŒØ§ Ø¨ÛŒØ´ØªØ± Ù…Ù‚Ø¯Ø§Ø± null Ø¯Ø± Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ù‚ÛŒÙ…ØªÛŒ
rows_to_drop_price = train_outprep[price_columns].isnull().sum(axis=1) >= 4

# Ø­Ø°Ù� Ø§ÛŒÙ† Ø±Ø¯ÛŒÙ�â€ŒÙ‡Ø§
train_outprep = train_outprep[~rows_to_drop_price]

print(f"Remaining rows after dropping based on price nulls: {train_outprep.shape[0]}")



# Ø¯Ø±ØµØ¯ Ù…Ù‚Ø§Ø¯ÛŒØ± null Ø¯Ø± Ù‡Ø± Ø±Ø¯ÛŒÙ�
row_null_percent = train_outprep.isnull().mean(axis=1)

# Ø­Ø°Ù� Ø±Ø¯ÛŒÙ�â€ŒÙ‡Ø§ÛŒÛŒ Ú©Ù‡ Ø¨ÛŒØ´ Ø§Ø² 50% Ù…Ù‚Ø§Ø¯ÛŒØ±Ø´Ø§Ù† null Ø§Ø³Øª
train_outprep = train_outprep[row_null_percent <= 0.5]

print(f"Remaining rows after dropping rows with >50% nulls: {train_outprep.shape[0]}")



# Ø´Ù†Ø§Ø³Ø§ÛŒÛŒ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¹Ø¯Ø¯ÛŒ Ùˆ Ø¯Ø³ØªÙ‡â€ŒØ§ÛŒ
numeric_cols = train_outprep.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = train_outprep.select_dtypes(include=['object']).columns

# Ø¬Ø§ÛŒÚ¯Ø²ÛŒÙ†ÛŒ Ù…Ù‚Ø§Ø¯ÛŒØ± null Ø¯Ø± Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¹Ø¯Ø¯ÛŒ Ø¨Ø§ Ù…ÛŒØ§Ù†Ù‡
train_outprep[numeric_cols] = train_outprep[numeric_cols].fillna(train_outprep[numeric_cols].median())

# Ø¬Ø§ÛŒÚ¯Ø²ÛŒÙ†ÛŒ Ù…Ù‚Ø§Ø¯ÛŒØ± null Ø¯Ø± Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¯Ø³ØªÙ‡â€ŒØ§ÛŒ Ø¨Ø§ Ù…Ø¯
for col in categorical_cols:
    train_outprep[col] = train_outprep[col].fillna(train_outprep[col].mode()[0])

# Ø¨Ø±Ø±Ø³ÛŒ Ù†Ù‡Ø§ÛŒÛŒ
print(train_outprep.isnull().sum().sum(), "null values remaining")



# Ø§ÛŒØ¬Ø§Ø¯ Ù†Ø³Ø®Ù‡â€ŒØ§ÛŒ Ø§Ø² Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ Ø¨Ø±Ø§ÛŒ train_FS
train_FS = train_outprep.copy()

# Ù†Ù…Ø§ÛŒØ´ Ø´Ú©Ù„ Ø¯ÛŒØªØ§ Ùˆ Ú†Ù†Ø¯ Ù†Ù…ÙˆÙ†Ù‡ Ø§Ø² Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ Ø¨Ø±Ø§ÛŒ Ø§Ø·Ù…ÛŒÙ†Ø§Ù†
print(train_FS.shape)
train_FS.head()



# Ø§Ù†ØªØ®Ø§Ø¨ Ù�ÛŒÙ„Ø¯Ù‡Ø§ÛŒ Ø¹Ø¯Ø¯ÛŒ (Ù¾ÛŒÙˆØ³ØªÙ‡)
continuous_fields = train_FS.select_dtypes(include=['int64', 'float64']).columns.tolist()
continuous_fields.remove('IsBadBuy')  # Ù…ØªØºÛŒØ± Ù‡Ø¯Ù� Ø±Ùˆ Ø­Ø°Ù� Ù…ÛŒâ€ŒÚ©Ù†ÛŒÙ…

# Ù†Ù…Ø§ÛŒØ´ Ø¢Ù…Ø§Ø± ØªÙˆØµÛŒÙ�ÛŒ
desc_stats = train_FS[continuous_fields].describe().T
desc_stats['skewness'] = train_FS[continuous_fields].skew()
desc_stats['kurtosis'] = train_FS[continuous_fields].kurtosis()

desc_stats



pip install scorecardbundle


from scorecardbundle.feature_discretization import ChiMerge as cm
import numpy as np
import pandas as pd

chi_merge_list = ['VehBCost', 'WarrantyCost']

# Ø§Ø¬Ø±Ø§ÛŒ Chi-Merge
trans_cm = cm.ChiMerge(max_intervals=5, min_intervals=1, decimal=3, output_dataframe=True)
result_cm = trans_cm.fit_transform(train_FS[chi_merge_list], train_FS['IsBadBuy'].astype('int')) 

# Ø§Ø³ØªØ®Ø±Ø§Ø¬ Ù…Ø±Ø²Ù‡Ø§ÛŒ Ø¨Ø§ÛŒÙ†â€ŒÙ‡Ø§
boundaries_dict = {key: np.insert(boundaries, 0, -np.inf) for key, boundaries in trans_cm.boundaries_.items()}

# Ø§Ù�Ø²ÙˆØ¯Ù† Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¬Ø¯ÛŒØ¯ Ø¨Ù‡ train_FS
for key, boundaries in boundaries_dict.items():
    column_name = f"{key}_cat_cm"
    # Ø¯Ø³ØªÙ‡â€ŒØ¨Ù†Ø¯ÛŒ Ø¨Ø§ Ø¨Ø±Ú†Ø³Ø¨â€ŒÙ‡Ø§ÛŒ 1 ØªØ§ 5
    train_FS[column_name] = pd.cut(
        train_FS[key],
        bins=boundaries,
        labels=range(1, len(boundaries)),
        right=False
    )
    
    # Ú†Ø§Ù¾ Ù…Ø±Ø²Ù‡Ø§
    print(f'{column_name} bin edges:', boundaries)
    
    # Ú†Ø§Ù¾ Ø¬Ø¯ÙˆÙ„ Ù�Ø±Ø§ÙˆØ§Ù†ÛŒ
    print(train_FS[column_name].value_counts().sort_index())
    print("\n")

# Ø­Ø°Ù� Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø§ØµÙ„ÛŒ
train_FS = train_FS.drop(columns=chi_merge_list)

# Ù†Ù…Ø§ÛŒØ´ Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ… Ù†Ù‡Ø§ÛŒÛŒ
print(train_FS.head())

# Ø¢Ù…Ø§Ø± ØªÙˆØµÛŒÙ�ÛŒ Ø¨Ø¹Ø¯ Ø§Ø² Ø¯ÛŒØ³Ú©Ø±ØªØ§ÛŒØ²ÛŒØ´Ù†
train_FS.describe()



from sklearn.preprocessing import OneHotEncoder

# Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø§Ø³Ù…ÛŒ (nominal)
nominal_fields = ["Auction", "Make", "Color", "Transmission", "WheelType", "Nationality", "Size", "TopThreeAmericanName"]

# One-Hot Encoder
one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

# Fit & Transform
one_hot_encoded = one_hot_encoder.fit_transform(train_FS[nominal_fields])

# Ø³Ø§Ø®Øª Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ… Ø¬Ø¯ÛŒØ¯ Ø¨Ø±Ø§ÛŒ ÙˆÛŒÚ˜Ú¯ÛŒâ€ŒÙ‡Ø§ÛŒ One-Hot
one_hot_encoded_df = pd.DataFrame(
    one_hot_encoded,
    columns=one_hot_encoder.get_feature_names_out(nominal_fields),
    index=train_FS.index
)

# Ø§ØªØµØ§Ù„ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ÛŒ Ø§ØµÙ„ÛŒ + Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ÛŒ Ø¬Ø¯ÛŒØ¯
train_FS_encoded = pd.concat([train_FS.drop(columns=nominal_fields), one_hot_encoded_df], axis=1)

print("Ø´Ú©Ù„ Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ… Ø¨Ø¹Ø¯ Ø§Ø² One-Hot Encoding:", train_FS_encoded.shape)



from sklearn.preprocessing import MinMaxScaler

# Ù‡Ù…Ù‡ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ Ø¨Ù‡ Ø¬Ø² Ù…ØªØºÛŒØ± Ù‡Ø¯Ù�
features_to_scale = train_FS_encoded.drop(columns=['IsBadBuy']).columns

# Ø§ÛŒØ¬Ø§Ø¯ Ø´ÛŒØ¡ MinMaxScaler
scaler = MinMaxScaler()

# Ø§Ø¹Ù…Ø§Ù„ Ù…Ù‚ÛŒØ§Ø³â€ŒØ¨Ù†Ø¯ÛŒ
train_FS_encoded[features_to_scale] = scaler.fit_transform(train_FS_encoded[features_to_scale])

# Ù†Ù…Ø§ÛŒØ´ Ú†Ù†Ø¯ Ø³Ø·Ø± Ø§ÙˆÙ„ Ø¨Ø±Ø§ÛŒ Ø¨Ø±Ø±Ø³ÛŒ
train_FS_encoded.head()



# Ø§ÛŒØ¬Ø§Ø¯ Ù†Ø³Ø®Ù‡â€ŒØ§ÛŒ Ø§Ø² Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ Ø¨Ø±Ø§ÛŒ train_FE
train_FE = train_outprep.copy()

# Ù†Ù…Ø§ÛŒØ´ Ø´Ú©Ù„ Ø¯ÛŒØªØ§ Ùˆ Ú†Ù†Ø¯ Ù†Ù…ÙˆÙ†Ù‡ Ø§Ø² Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ Ø¨Ø±Ø§ÛŒ Ø§Ø·Ù…ÛŒÙ†Ø§Ù†
print(train_FE.shape)
train_FE.head()


from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt

# Ù„ÛŒØ³Øª Ù�ÛŒÚ†Ø±Ù‡Ø§ÛŒÛŒ Ú©Ù‡ Ø¨Ø§ÛŒØ¯ ØªØ¨Ø¯ÛŒÙ„ Ø¨Ø´Ù†
selected_features = ['VehBCost', 'WarrantyCost']

# Ø§Ø¹Ù…Ø§Ù„ Box-Cox Transformation Ù�Ù‚Ø· Ø±ÙˆÛŒ Ù…Ù‚Ø§Ø¯ÛŒØ± Ù…Ø«Ø¨Øª
for feature in selected_features:
    if (train_FE[feature] <= 0).any():
        raise ValueError(f"Feature {feature} has non-positive values, Box-Cox cannot be applied.")

    transformer = PowerTransformer(method='box-cox', standardize=False)
    train_FE[f"{feature}_transformed"] = transformer.fit_transform(train_FE[[feature]])

    # Ú†Ø§Ù¾ Ù„Ø§Ù…Ø¨Ø¯Ø§
    lambda_value = transformer.lambdas_[0]
    print(f"Lambda for {feature}: {lambda_value}")

    # Ø±Ø³Ù… Ù‡ÛŒØ³ØªÙˆÚ¯Ø±Ø§Ù… Ù‚Ø¨Ù„ Ùˆ Ø¨Ø¹Ø¯ Ø§Ø² ØªØ¨Ø¯ÛŒÙ„
    plt.figure(figsize=(7, 3))

    plt.subplot(1, 2, 1)
    plt.hist(train_FE[feature], bins=30, color='blue', alpha=0.7)
    plt.title(f'Original {feature} Histogram')

    plt.subplot(1, 2, 2)
    plt.hist(train_FE[f"{feature}_transformed"], bins=30, color='green', alpha=0.7)
    plt.title(f'Transformed {feature} Histogram')

    plt.tight_layout()
    plt.show()

# Ø­Ø°Ù� Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø§ØµÙ„ÛŒ
train_FE.drop(columns=selected_features, inplace=True)

# Ø¨Ø±Ø±Ø³ÛŒ ØªØ¹Ø¯Ø§Ø¯ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§
print("Final shape:", train_FE.shape)
train_FE.head()



from sklearn.preprocessing import OneHotEncoder

# Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø§Ø³Ù…ÛŒ (nominal)
nominal_fields = ["Auction", "Make", "Color", "Transmission", "WheelType", "Nationality", "Size", "TopThreeAmericanName"]

# One-Hot Encoder
one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

# Fit & Transform
one_hot_encoded = one_hot_encoder.fit_transform(train_FE[nominal_fields])

# Ø³Ø§Ø®Øª Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ… Ø¬Ø¯ÛŒØ¯ Ø¨Ø±Ø§ÛŒ ÙˆÛŒÚ˜Ú¯ÛŒâ€ŒÙ‡Ø§ÛŒ One-Hot
one_hot_encoded_df = pd.DataFrame(
    one_hot_encoded,
    columns=one_hot_encoder.get_feature_names_out(nominal_fields),
    index=train_FE.index
)

# Ø§ØªØµØ§Ù„ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ÛŒ Ø§ØµÙ„ÛŒ + Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ÛŒ Ø¬Ø¯ÛŒØ¯
train_FE_encoded = pd.concat([train_FE.drop(columns=nominal_fields), one_hot_encoded_df], axis=1)

print("Ø´Ú©Ù„ Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ… Ø¨Ø¹Ø¯ Ø§Ø² One-Hot Encoding:", train_FE_encoded.shape)



train_FE_encoded.info()


from sklearn.preprocessing import StandardScaler

# Ø§Ù†ØªØ®Ø§Ø¨ Ù‡Ù…Ù‡ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ Ø¨Ù‡ Ø¬Ø² IsBadBuy
features_to_scale = train_FE_encoded.drop(columns=['IsBadBuy'])
target = train_FE_encoded['IsBadBuy']

# Ø§Ø¹Ù…Ø§Ù„ Z-Score Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_to_scale)

# Ø³Ø§Ø®Øª Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ… Ø¬Ø¯ÛŒØ¯ Ø¨Ø§ Ù‡Ù…Ø§Ù† Ù†Ø§Ù… Ø³ØªÙˆÙ†â€ŒÙ‡Ø§
scaled_df = pd.DataFrame(scaled_features, columns=features_to_scale.columns, index=train_FE_encoded.index)

# Ø¯ÙˆØ¨Ø§Ø±Ù‡ Ø§Ù„Ø­Ø§Ù‚ Ø³ØªÙˆÙ† Ù‡Ø¯Ù�
train_FE_scaled = pd.concat([scaled_df, target], axis=1)

print("Ø´Ú©Ù„ Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ… Ø¨Ø¹Ø¯ Ø§Ø² Z-Score Scaling:", train_FE_scaled.shape)
train_FE_scaled.head()



Y_train_encoded = train_FS_encoded.IsBadBuy
X_train_encoded = train_FS_encoded.drop('IsBadBuy', axis=1)
X_train_encoded.info()


!pip install --upgrade scikit-learn


from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier 

# configure RFECV
selector = RFECV(
    estimator=DecisionTreeClassifier(random_state=29),  # Classifier Ø¨Ù‡ Ø¬Ø§ÛŒ Regressor
    step=1,
    min_features_to_select=10,
    cv=5,
    scoring='accuracy',  # Ú†ÙˆÙ† classification Ø§Ø³Øª
    n_jobs=-1
)

# fit on training data
selector.fit(X_train_encoded, Y_train_encoded)

# print optimal number of features
print(f"Optimal number of features: {selector.n_features_}")
print("="*50)

# get selected features
wrapper_fs = X_train_encoded.columns[selector.support_]  # Ù†Ø­ÙˆÙ‡ Ú¯Ø±Ù�ØªÙ† Ø§Ø³Ù… Ù�ÛŒÚ†Ø±Ù‡Ø§
print("Wrapper Optimal Feature List:")
print(wrapper_fs.tolist())

# subset dataset with selected features
X_train_wrapper_fs = X_train_encoded[wrapper_fs]



Y_train_scaled = train_FE_scaled.IsBadBuy
X_train_scaled = train_FE_scaled.drop('IsBadBuy', axis=1)
X_train_scaled.info()


# Define categorical features manually
categorical = [
    'IsOnlineSale',
    'Auction_MANHEIM', 'Auction_OTHER',
    'Make_CHRYSLER', 'Make_DODGE', 'Make_FORD', 'Make_HYUNDAI', 'Make_JEEP',
    'Make_KIA', 'Make_MAZDA', 'Make_MERCURY', 'Make_MITSUBISHI', 'Make_NISSAN',
    'Make_OTHER', 'Make_PONTIAC', 'Make_SATURN', 'Make_SUZUKI', 'Make_TOYOTA',
    'Color_BLACK', 'Color_BLUE', 'Color_GOLD', 'Color_GREEN', 'Color_GREY',
    'Color_MAROON', 'Color_OTHER', 'Color_RED', 'Color_SILVER', 'Color_WHITE',
    'Transmission_MANUAL', 'Transmission_Manual',
    'WheelType_Covers', 'WheelType_Special',
    'Nationality_OTHER', 'Nationality_OTHER ASIAN', 'Nationality_TOP LINE ASIAN',
    'Size_CROSSOVER', 'Size_LARGE', 'Size_LARGE SUV', 'Size_LARGE TRUCK',
    'Size_MEDIUM', 'Size_MEDIUM SUV', 'Size_SMALL SUV', 'Size_SMALL TRUCK',
    'Size_SPECIALTY', 'Size_SPORTS', 'Size_VAN',
    'TopThreeAmericanName_FORD', 'TopThreeAmericanName_GM', 'TopThreeAmericanName_OTHER'
]

# Convert to categorical type
X_train_scaled[categorical] = X_train_scaled[categorical].astype('category')

# Continuous features = all - categorical
continuous = X_train_scaled.drop(categorical, axis=1).columns.tolist()

# Check lengths
len(categorical), len(continuous)



# Create DataFrame with only continuous features
train_FE_scaled_continuous = X_train_scaled[continuous].copy()

# Ø¨Ø±Ø±Ø³ÛŒ Ø´Ú©Ù„ Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ…
train_FE_scaled_continuous.info()



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Compute correlation matrix
correlation_matrix = train_FE_scaled_continuous.corr()

# Step 2: Visualize correlation matrix using a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 7})
plt.title('Pearson Correlation Heatmap')
plt.show()


from sklearn.decomposition import PCA
import pandas as pd

# Step 1.2: Apply PCA to continuous features
pca = PCA(n_components=None)  # Ù†Ú¯Ù‡ Ø¯Ø§Ø´ØªÙ† Ù‡Ù…Ù‡ Ú©Ø§Ù…Ù¾ÙˆÙ†Ù†Øªâ€ŒÙ‡Ø§
pca.fit(train_FE_scaled_continuous)

# Ø³Ø§Ø®Øª Ù†Ø§Ù… Ú©Ø§Ù…Ù¾ÙˆÙ†Ù†Øªâ€ŒÙ‡Ø§
pc_names = [f'PC_{i+1}' for i in range(pca.n_components_)]

# Variance Ùˆ Explained Variance Ratio
variance = pd.DataFrame(pca.explained_variance_, index=pc_names, columns=['Variance'])
explained_variance_ratio = pd.DataFrame(pca.explained_variance_ratio_, index=pc_names, columns=['Explained_Variance_Ratio'])
cumulative_variance = pd.DataFrame(pca.explained_variance_ratio_.cumsum(), index=pc_names, columns=['Cumulative_Ratio'])

# Component Weights (Loadings)
component_weights = pd.DataFrame(pca.components_, columns=train_FE_scaled_continuous.columns, index=pc_names)

# ØªØ±Ú©ÛŒØ¨ Ù‡Ù…Ù‡ Ø§Ø·Ù„Ø§Ø¹Ø§Øª Ø¯Ø± ÛŒÚ© DataFrame
pca_report = pd.concat([variance, explained_variance_ratio, cumulative_variance, component_weights], axis=1)

# Ù†Ù…Ø§ÛŒØ´ Ú©Ø§Ù…Ù„
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
print(pca_report)



import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ØªØ¨Ø¯ÛŒÙ„ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ Ø¨Ø§ PCA
pca_X_train = pca.transform(train_FE_scaled_continuous)
pca_X_train = pd.DataFrame(pca_X_train, columns=pc_names)

# Ø§Ù†ØªØ®Ø§Ø¨ Ú©Ø§Ù…Ù¾ÙˆÙ†Ù†Øªâ€ŒÙ‡Ø§ÛŒ Ø¨Ø± Ø§Ø³Ø§Ø³ Eigenvalue > 1 (PC_1 Ùˆ PC_2)
pca_selected = pca_X_train[['PC_1', 'PC_2']]

# ØªØ±Ú©ÛŒØ¨ Ø¨Ø§ target variable
pca_selected['IsBadBuy'] = train_FE_scaled['IsBadBuy'].reset_index(drop=True)

# Ø±Ø³Ù… pairplot
sns.pairplot(pca_selected, hue='IsBadBuy', diag_kind='kde', palette='Set2')
plt.show()



# Ø§Ù†ØªØ®Ø§Ø¨ Ú©Ø§Ù…Ù¾ÙˆÙ†Ù†Øªâ€ŒÙ‡Ø§ÛŒ PCA Ú©Ù‡ Ù…Ø±Ø­Ù„Ù‡ Ù‚Ø¨Ù„ Ù…Ø´Ø®Øµ Ø´Ø¯ (PC_1 Ùˆ PC_2)
selected_pcs = ['PC_1', 'PC_2']
pca_components = pd.DataFrame(pca.transform(train_FE_scaled_continuous), 
                              columns=[f'PC_{i+1}' for i in range(pca.n_components_)])[selected_pcs]

# Ø§Ù†ØªØ®Ø§Ø¨ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ ØºÛŒØ± Ù¾ÛŒÙˆØ³ØªÙ‡ Ø§Ø² train_FE_scaled
categorical_columns = X_train_scaled.drop(train_FE_scaled_continuous.columns, axis=1).columns
categorical_data = train_FE_scaled[categorical_columns].reset_index(drop=True)

# ØªØ±Ú©ÛŒØ¨ Ú©Ø§Ù…Ù¾ÙˆÙ†Ù†Øªâ€ŒÙ‡Ø§ÛŒ PCA Ø¨Ø§ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ ØºÛŒØ± Ù¾ÛŒÙˆØ³ØªÙ‡
train_pca_fe = pd.concat([pca_components.reset_index(drop=True), categorical_data], axis=1)

# Ù†Ù…Ø§ÛŒØ´ Ø³Ø§Ø®ØªØ§Ø± Ù†Ù‡Ø§ÛŒÛŒ
train_pca_fe.info()
train_pca_fe.head()



!pip install --upgrade scikit-learn



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

# Step 2.1: Apply LDA

# Ù…ØªØºÛŒØ± Ù‡Ø¯Ù�
y_train_lda = train_FE_scaled['IsBadBuy']

# LDA model: n_components <= n_classes - 1
lda = LinearDiscriminantAnalysis(n_components=1)

# fit LDA on continuous features
lda.fit(train_FE_scaled_continuous, y_train_lda)

# transform the data to get the LDA component(s)
lda_components = lda.transform(train_FE_scaled_continuous)

# ØªØ¨Ø¯ÛŒÙ„ Ø¨Ù‡ DataFrame Ø¨Ø§ Ù†Ø§Ù… Ù…Ù†Ø§Ø³Ø¨
lda_components_df = pd.DataFrame(lda_components, columns=['LDA_1'])

# Ø§Ø¶Ø§Ù�Ù‡ Ú©Ø±Ø¯Ù† Ø³ØªÙˆÙ† Ù‡Ø¯Ù� Ø¨Ø±Ø§ÛŒ ØªØ­Ù„ÛŒÙ„ Ø¨Ø¹Ø¯ÛŒ
lda_components_df['IsBadBuy'] = y_train_lda.reset_index(drop=True)

# Ù†Ù…Ø§ÛŒØ´ Ú†Ù†Ø¯ Ø³Ø·Ø± Ø§ÙˆÙ„ Ø¨Ø±Ø§ÛŒ Ø¨Ø±Ø±Ø³ÛŒ
print(lda_components_df.head())



import seaborn as sns
import matplotlib.pyplot as plt

# Ù…Ø±Ø­Ù„Ù‡ 2.2: Ø¨ØµØ±ÛŒâ€ŒØ³Ø§Ø²ÛŒ LDA component Ø¨Ø§ target
plt.figure(figsize=(8,6))
sns.histplot(data=lda_components_df, x='LDA_1', hue='IsBadBuy', kde=True, palette=['blue','red'], bins=30)
plt.title('LDA Component Distribution by IsBadBuy')
plt.xlabel('LDA Component 1')
plt.ylabel('Count')
plt.show()



# Ø´Ù†Ø§Ø³Ø§ÛŒÛŒ Ù�ÛŒÙ„Ø¯Ù‡Ø§ÛŒ ØºÛŒØ± Ù¾ÛŒÙˆØ³ØªÙ‡
non_continuous_features = train_FE_scaled.drop(columns=train_FE_scaled_continuous.columns).columns.tolist()

# Ø§ÛŒØ¬Ø§Ø¯ DataFrame Ù†Ù‡Ø§ÛŒÛŒ Ø´Ø§Ù…Ù„ LDA component Ùˆ Ø¨Ù‚ÛŒÙ‡ Ù�ÛŒÙ„Ø¯Ù‡Ø§
train_lda_fe = pd.concat([lda_components_df[['LDA_1']], train_FE_scaled[non_continuous_features].reset_index(drop=True)], axis=1)

# Ù†Ù…Ø§ÛŒØ´ Ú†Ù†Ø¯ Ø³Ø·Ø± Ø§ÙˆÙ„ Ø¨Ø±Ø§ÛŒ Ø¨Ø±Ø±Ø³ÛŒ
print(train_lda_fe.head())


