# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv")


df.head()


df.size


import re

def split_plate(plate):
    """
    Splits a Russian license plate like 'P001AY199' into:
    prefix_letters, number_digits, region_code
    """
    match = re.match(r'^([A-Z]+)(\d{1,3})([A-Z]{2})(\d{2,3})$', plate)
    if not match:
        return pd.Series([None, None, None, None])
    letter1, digits, letter2, region = match.groups()
    return pd.Series([letter1, digits, letter2, region])



df[['plate_prefix', 'plate_number', 'plate_suffix', 'plate_region']] = df['plate'].apply(split_plate)



df.head(15)


def validate_plate_format(df):
    failed_plates = []

    pattern = re.compile(r'^([A-Z]+)(\d{1,3})([A-Z]{2})(\d{2,3})$')

    for plate in df['plate']:
        if not pattern.match(plate):
            failed_plates.append(plate)
    
    print(f"Total plates: {len(df)}")
    print(f"Failed to match pattern: {len(failed_plates)}")

    if failed_plates:
        print("Some failed examples:")
        print(failed_plates[:10])  # Print first 10 only to avoid clutter

    return failed_plates



failed = validate_plate_format(df)



import sys
sys.path.append("/kaggle/working")



from supplemental_english import REGION_CODES, GOVERNMENT_CODES


REGION_CODES


REGION_CODE_TO_NAME = {}

for region, codes in REGION_CODES.items():
    for code in codes:
        REGION_CODE_TO_NAME[code] = region



df["region_name"] = df["plate_region"].map(REGION_CODE_TO_NAME)



df[["plate", "plate_region", "region_name"]].head(10)



unmapped = df[df["region_name"].isna()]["plate_region"].unique()
print("Unmapped region codes:", unmapped)



df.head()


df["series"] = df["plate_prefix"] + df["plate_suffix"]



def match_gov_plate(row):
    letters = row["series"]
    try:
        number = int(row["plate_number"])
    except:
        return pd.Series([0, 0, 0, None])  # if plate_number is bad
    region = row["plate_region"]

    for (gov_letters, (start, end), gov_region), (desc, forbidden, advantage, significance) in GOVERNMENT_CODES.items():
        if letters == gov_letters and region == gov_region and start <= number <= end:
            return pd.Series([forbidden, advantage, significance, desc])
    
    return pd.Series([0, 0, 0, None])



df[["is_forbidden", "has_advantage", "significance", "gov_plate_type"]] = df.apply(match_gov_plate, axis=1)


df.head(20)


df["gov_plate_type"].isna().sum()



df[df["gov_plate_type"].notna()]



df["has_advantage"].value_counts()


print(df[df["significance"] != 0] )
df["significance"].value_counts()



df["gov_plate_type"].notna().sum()



df.groupby("gov_plate_type")["price"].mean().sort_values(ascending=False)



df[df["significance"] == 1][["gov_plate_type", "price"]].groupby("gov_plate_type").agg(["count", "mean", "std"]).sort_values(("price", "mean"), ascending=False)



df.groupby("significance").agg({
    "price": ["mean", "median", "count"],
    "is_forbidden": "sum",
    "has_advantage": "sum"
}).sort_index()



def categorize_significance(x):
    if x == 0: return "Normal"
    elif x in [1, 2, 3, 4]: return "Special"
    elif x == 5: return "Forbidden Special"
    elif x == 6: return "Elite"
    elif x == 8: return "Ultra Elite"
    else: return "Unknown"

df["sig_level"] = df["significance"].apply(categorize_significance)



df.head()


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x="sig_level", y="price", data=df[df["sig_level"] != "Normal"])
plt.xticks(rotation=45)
plt.yscale("log")
plt.title("Price distribution by Significance Level")
plt.show()



numeric_cols = df.select_dtypes(include='number')
corr_matrix = numeric_cols.corr()



numeric_cols = df.select_dtypes(include='number')
corr_matrix = numeric_cols.corr().round(2)
print(corr_matrix)



import seaborn as sns
import matplotlib.pyplot as plt

# Select only numeric columns
numeric_cols = df.select_dtypes(include='number')

# Compute correlation matrix
corr_matrix = numeric_cols.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title("Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.histplot(df['price'], bins=50, kde=True)

plt.title('Original Price Distribution')
plt.show()

sns.histplot(np.log1p(df['price']), bins=50, kde=True)
plt.title('Log-Transformed Price Distribution')
plt.show()



sns.boxplot(x='significance', y='price', data=df)
plt.yscale('log')  # helpful if there's price skew
plt.title('Price vs. Significance (log scale)')
plt.show()



df.groupby('region_name')['price'].agg(['count', 'mean', 'median']).sort_values('mean', ascending=False)



top_regions = df['plate_region'].value_counts().head(10).index
sns.boxplot(data=df[df['plate_region'].isin(top_regions)], x='plate_region', y='price')
plt.yscale('log')  # because we know the price is skewed
plt.xticks(rotation=45)
plt.title('Price by Top Regions (Log Scale)')
plt.show()



df[df['plate_region'] == '88']



df['log_price'] = np.log1p(df['price'])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['price'], bins=100, ax=axes[0])
axes[0].set_title('Original Price Distribution')
axes[0].set_yscale('log')

sns.histplot(df['log_price'], bins=100, ax=axes[1])
axes[1].set_title('Log-Transformed Price Distribution')

plt.tight_layout()
plt.show()



def extract_plate_patterns(df):
    def is_triple(num): return len(set(num)) == 1
    def is_double(num): return len(set(num)) <= 2
    def is_palindrome(num): return num == num[::-1]
    def is_round(num): return num.endswith("00") or num.endswith("000")
    def is_low(num): return int(num) <= 99
    def is_sequence(num): 
        return num in {"123", "234", "345", "456", "567", "678", "789",
                       "987", "876", "765", "654", "543", "432", "321", "210"}

    df["has_triple_number"] = df["plate_number"].apply(is_triple)
    df["has_double_number"] = df["plate_number"].apply(is_double)
    df["is_palindrome"] = df["plate_number"].apply(is_palindrome)
    df["is_round"] = df["plate_number"].apply(is_round)
    df["is_low_number"] = df["plate_number"].astype(int).apply(is_low)
    df["is_sequence"] = df["plate_number"].apply(is_sequence)

    # Combine prefix and suffix for series
    df["series"] = df["plate_prefix"] + df["plate_suffix"]

    df["series_palindrome"] = df["series"].apply(lambda x: x == x[::-1])
    df["series_repeating"] = df["series"].apply(lambda x: len(set(x)) <= 2)
    df["series_all_same"] = df["series"].apply(lambda x: len(set(x)) == 1)

    return df

# âœ… APPLY IT
df = extract_plate_patterns(df)



df.head()


pattern_cols = [
    "plate_number", "series",
    "has_triple_number", "has_double_number", "is_palindrome", 
    "is_round", "is_low_number", "is_sequence",
    "series_palindrome", "series_repeating", "series_all_same"
]

df[pattern_cols].head(10)



df[pattern_cols + ["price"]].corr(numeric_only=True)["price"].sort_values(ascending=False)



# Convert to datetime if not already
df['date'] = pd.to_datetime(df['date'])

# Extract granular time features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
df['is_weekend'] = df['weekday'].isin([5, 6])

# Check correlation with price
df[['price', 'year', 'month', 'day', 'weekday', 'is_weekend']].corr()['price'].sort_values(ascending=False)



# Check for missing values
print(df.isnull().sum())

# Check for duplicate rows or IDs
print(df.duplicated().sum())
print(df['id'].duplicated().sum())

# Sanity check on target variable
import matplotlib.pyplot as plt
import seaborn as sns

# Plot log(price)
df['log_price'] = np.log1p(df['price'])
sns.histplot(df['log_price'], bins=50, kde=True)
plt.title('Log Price Distribution')
plt.show()



# Remove top 0.1% price outliers
threshold = df['price'].quantile(0.999)
df = df[df['price'] < threshold]



df[['significance', 'has_advantage']].drop_duplicates()



df[(df['significance'] != 0) & (df['has_advantage'] == 0)]



df[(df['significance'] != 0) & (df['has_advantage'] == 0)][['price', 'significance', 'has_advantage', 'gov_plate_type', 'plate']]



df['adv_inferred'] = ((df['significance'] > 0) | (df['has_advantage'] == 1)).astype(int)



df[['price', 'significance', 'has_advantage', 'adv_inferred']].corr()['price']



df.columns


df['gov_plate_type'] = df['gov_plate_type'].fillna('Unknown')



df["gov_plate_type"].value_counts()


df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
df['is_weekend'] = df['weekday'].isin([5, 6])



df['log_price'] = np.log1p(df['price'])



df['series'] = df['plate_prefix'] + df['plate_suffix']
df['has_triple_number'] = df['plate_number'].str.match(r'^(\d)\1{2}$')
df['has_double_number'] = df['plate_number'].str.match(r'^(\d)\1{1}\d?$')
df['is_palindrome'] = df['plate_number'] == df['plate_number'].str[::-1]
df['is_round'] = df['plate_number'].isin(['000', '100', '500', '1000', '5000'])
df['is_low_number'] = df['plate_number'].astype(int) < 100
df['is_sequence'] = df['plate_number'].str.contains('123|234|345|456|567|678|789')
df['series_palindrome'] = df['series'] == df['series'].str[::-1]
df['series_repeating'] = df['series'].apply(lambda x: x[0] == x[1] == x[2])
df['series_all_same'] = df['series'].apply(lambda x: len(set(x)) == 1)



# ğŸ§¼ Fill NaNs and set categorical types
df['gov_plate_type'] = df['gov_plate_type'].fillna('Unknown').astype('category')
df['sig_level'] = df['sig_level'].astype('category')

# ğŸŒ� High-cardinality â€” one-hot encode ONLY region_name (skip if using CatBoost)
df = pd.get_dummies(df, columns=['region_name'], drop_first=True)



freq_map = df['plate_region'].value_counts(normalize=True).to_dict()
df['plate_region_freq'] = df['plate_region'].map(freq_map)



drop_cols = [
    'id', 'plate', 'plate_prefix', 'plate_number', 'plate_suffix', 
    'series', 'date',
    'plate_region'              # drop if you're using adv_inferred
]
df = df.drop(columns=drop_cols)



df.head()


from sklearn.preprocessing import LabelEncoder
encoders = {}
label_cols = ['gov_plate_type', 'sig_level']
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le 



def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))



non_numeric = df.select_dtypes(include=['object', 'category']).columns
print(non_numeric)



X = df.drop(columns=['price', 'log_price'])  # or 'price' if you're predicting log_price
y = df['log_price']

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)






from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

results = {}

def evaluate_model(name, model, X_train, X_valid, y_train, y_valid):
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    
    mae = mean_absolute_error(y_valid, preds)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    smape = 100 * np.mean(2 * np.abs(preds - y_valid) / (np.abs(preds) + np.abs(y_valid)))
    
    print(f"{name}:\n  MAE: {mae:.2f}, RMSE: {rmse:.2f}, SMAPE: {smape:.2f}%")
    
    return preds


# Random Forest
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_preds = evaluate_model("Random Forest", rf, X_train, X_valid, y_train, y_valid)



xgb = XGBRegressor(random_state=42, n_jobs=-1)
xgb_preds = evaluate_model("XGBoost", xgb, X_train, X_valid, y_train, y_valid)


import matplotlib.pyplot as plt
importances = xgb.feature_importances_
features = X_train.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Top 20 Feature Importances (XGBoost)")
plt.bar(range(20), importances[indices[:20]], align="center")
plt.xticks(range(20), features[indices[:20]], rotation=90)
plt.tight_layout()
plt.show()




dftest = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")
dftest[['plate_prefix', 'plate_number', 'plate_suffix', 'plate_region']] = dftest['plate'].apply(split_plate)
dftest["region_name"] = dftest["plate_region"].map(REGION_CODE_TO_NAME)
dftest["series"] = dftest["plate_prefix"] + dftest["plate_suffix"]
dftest[["is_forbidden", "has_advantage", "significance", "gov_plate_type"]] = dftest.apply(match_gov_plate, axis=1)
dftest["sig_level"] = dftest["significance"].apply(categorize_significance)
dftest = extract_plate_patterns(dftest)
dftest[['significance', 'has_advantage']].drop_duplicates()
dftest['adv_inferred'] = ((dftest['significance'] > 0) | (dftest['has_advantage'] == 1)).astype(int)
dftest['gov_plate_type'] = dftest['gov_plate_type'].fillna('Unknown')
dftest['date'] = pd.to_datetime(dftest['date'])
dftest['year'] = dftest['date'].dt.year
dftest['month'] = dftest['date'].dt.month
dftest['day'] = dftest['date'].dt.day
dftest['weekday'] = dftest['date'].dt.weekday
dftest['is_weekend'] = dftest['weekday'].isin([5, 6])
dftest['series'] = dftest['plate_prefix'] + dftest['plate_suffix']
dftest['has_triple_number'] = dftest['plate_number'].str.match(r'^(\d)\1{2}$')
dftest['has_double_number'] = dftest['plate_number'].str.match(r'^(\d)\1{1}\d?$')
dftest['is_palindrome'] = dftest['plate_number'] == dftest['plate_number'].str[::-1]
dftest['is_round'] = dftest['plate_number'].isin(['000', '100', '500', '1000', '5000'])
dftest['is_low_number'] = dftest['plate_number'].astype(int) < 100
dftest['is_sequence'] = dftest['plate_number'].str.contains('123|234|345|456|567|678|789')
dftest['series_palindrome'] = dftest['series'] == dftest['series'].str[::-1]
dftest['series_repeating'] = dftest['series'].apply(lambda x: x[0] == x[1] == x[2])
dftest['series_all_same'] = dftest['series'].apply(lambda x: len(set(x)) == 1)
# ğŸ§¼ Fill NaNs and set categorical types
dftest['gov_plate_type'] = dftest['gov_plate_type'].fillna('Unknown').astype('category')
dftest['sig_level'] = dftest['sig_level'].astype('category')

# ğŸŒ� High-cardinality â€” one-hot encode ONLY region_name (skip if using CatBoost)
dftest = pd.get_dummies(dftest, columns=['region_name'], drop_first=True)
dftest = dftest.drop(columns=[col for col in drop_cols if col in dftest.columns])
for col in ['gov_plate_type', 'sig_level']:
    dftest[col] = dftest[col].astype(str)
    le = encoders[col]
    known_classes = le.classes_.tolist()

    dftest[col] = dftest[col].apply(lambda x: x if x in known_classes else 'Unknown')

    # Rebuild encoder with 'Unknown' if needed
    if 'Unknown' not in known_classes:
        known_classes.append('Unknown')
    le_test = LabelEncoder()
    le_test.classes_ = np.array(known_classes)
    dftest[col] = le_test.transform(dftest[col])
non_numeric = dftest.select_dtypes(include=['object', 'category']).columns
print(non_numeric)


missing_cols = set(X_train.columns) - set(dftest.columns)
extra_cols = set(dftest.columns) - set(X_train.columns)

for col in missing_cols:
    dftest[col] = 0

dftest = dftest[X_train.columns]



y_pred = xgb.predict(dftest)



dftest = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")


list(df.columns)




# 2. Reverse log transformation (log1p was used â†’ now do expm1)
pred_prices = np.expm1(y_pred)

# 3. Read the test file again to get 'id' (since you dropped it from dftest)
original_test = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")

# 4. Create submission DataFrame
submission = pd.DataFrame({
    'id': original_test['id'],  # gets back the ID column
    'price': pred_prices
})

# 5. Save it to CSV
submission.to_csv('submission.csv', index=False)
print("âœ… Final submission.csv saved with correct prices and ids!")



import seaborn as sns
import matplotlib.pyplot as plt

# 1. Check price distribution
sns.histplot(df['price'], bins=100, kde=True)
plt.title("Price Distribution")
plt.show()

# 2. Correlation heatmap for numeric features
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(15, 10))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 3. Boxplots for outlier-prone features
for col in ['price', 'significance']:
    sns.boxplot(data=df, x='is_forbidden', y=col)
    plt.title(f"{col} vs is_forbidden")
    plt.show()

# 4. Countplots for categorical features
for col in ['gov_plate_type', 'sig_level']:
    sns.countplot(data=df, x=col)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.show()



plt.figure(figsize=(8,6))
sns.scatterplot(x=y_valid, y=xgb_preds, alpha=0.3)
plt.xlabel("Actual Log Price")
plt.ylabel("Predicted Log Price")
plt.title("Predicted vs Actual Log Price")
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')
plt.show()



plt.figure(figsize=(8,6))
sns.scatterplot(x=np.expm1(y_valid), y=np.expm1(xgb_preds), alpha=0.3)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual Price")
plt.plot([0, 1.2e7], [0, 1.2e7], 'r--')
plt.show()



combined = pd.concat([X_train, X_valid])
combined['target'] = pd.concat([y_train, y_valid])



import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you already have y_valid and y_pred
residuals = y_valid - xgb_preds

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.title("Distribution of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(xgb_preds, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()



from sklearn.linear_model import LinearRegression
import numpy as np

plt.figure(figsize=(8, 8))
plt.scatter(y_valid, xgb_preds, alpha=0.5)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')  # Perfect predictions line

# Add regression line
model = LinearRegression()
model.fit(np.array(y_valid).reshape(-1, 1), xgb_preds)
line = model.predict(np.array(y_valid).reshape(-1, 1))
plt.plot(y_valid, line, color='green', label='Fit Line')

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
plt.legend()
plt.show()





