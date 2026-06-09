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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


train_data = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
train_data.head()


test_data = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')



train_data.describe().T


test_data.head()


# Checking the number of rows and columns

num_train_rows, num_train_columns = train_data.shape

num_test_rows, num_test_columns = test_data.shape



print("Training Data:")
print(f"Number of Rows: {num_train_rows}")
print(f"Number of Columns: {num_train_columns}\n")

print("Test Data:")
print(f"Number of Rows: {num_test_rows}")
print(f"Number of Columns: {num_test_columns}\n")



# Creating a table for missing values, unique values and data types of the features

missing_values_train = pd.DataFrame({'Feature': train_data.columns,
                              '[TRAIN] No. of Missing Values': train_data.isnull().sum().values,
                              '[TRAIN] % of Missing Values': ((train_data.isnull().sum().values)/len(train_data)*100)})

missing_values_test = pd.DataFrame({'Feature': test_data.columns,
                             '[TEST] No.of Missing Values': test_data.isnull().sum().values,
                             '[TEST] % of Missing Values': ((test_data.isnull().sum().values)/len(test_data)*100)})



unique_values = pd.DataFrame({'Feature': train_data.columns,
                              'No. of Unique Values[FROM TRAIN]': train_data.nunique().values})

feature_types = pd.DataFrame({'Feature': train_data.columns,
                              'DataType': train_data.dtypes})

merged_df = pd.merge(missing_values_train, missing_values_test, on='Feature', how='left')

merged_df = pd.merge(merged_df, unique_values, on='Feature', how='left')
merged_df = pd.merge(merged_df, feature_types, on='Feature', how='left')

merged_df


# Count duplicate rows in train_data
train_duplicates = train_data.duplicated().sum()

# Count duplicate rows in test_data
test_duplicates = test_data.duplicated().sum()



# Print the results
print(f"Number of duplicate rows in train_data: {train_duplicates}")
print(f"Number of duplicate rows in test_data: {test_duplicates}")






 train_data.head



# Ø§Ø¨ØªØ¯Ø§ region_code Ø±Ø§ Ø§Ø² supplemental_english Ø§Ø³ØªØ®Ø±Ø§Ø¬ Ù…ÛŒâ€ŒÚ©Ù†ÛŒÙ…
import supplemental_english  # Ø§ÛŒÙ…Ù¾ÙˆØ±Øª Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§
region_code_mapping = supplemental_english.REGION_CODES  

# Ø¯ÛŒÚ©Ø´Ù†Ø±ÛŒ Ø¨Ø±Ø§ÛŒ Ù†Ú¯Ø§Ø´Øª Ú©Ø¯ Ù…Ù†Ø·Ù‚Ù‡ Ø¨Ù‡ Ù†Ø§Ù… Ù…Ù†Ø·Ù‚Ù‡
region_mapping = {code.zfill(3): region for region, codes in region_code_mapping.items() for code in codes}

# Ø§Ø³ØªØ®Ø±Ø§Ø¬ Ú©Ø¯ Ù…Ù†Ø·Ù‚Ù‡ Ø§Ø² Ù¾Ù„Ø§Ú© (Ø¢Ø®Ø±ÛŒÙ† Ø¨Ø®Ø´ Ù¾Ù„Ø§Ú©)
train_data["region_code"] = train_data["plate"].apply(lambda x: x[-2:].zfill(3))

# Ø§Ø¶Ø§Ù�Ù‡ Ú©Ø±Ø¯Ù† Ù†Ø§Ù… Ù…Ù†Ø·Ù‚Ù‡ Ø§Ø² Ø±ÙˆÛŒ region_code
train_data["region_name"] = train_data["region_code"].map(region_mapping)



train_data['date']=pd.to_datetime(train_data['date'])
train_data['month']=train_data['date'].dt.month
train_data['year']=train_data['date'].dt.year
train_data['dayofweek']=train_data['date'].dt.dayofweek
train_data['weekofyear']=train_data['date'].dt.isocalendar().week
   


train_data.head()  # Ø¨Ø±Ø±Ø³ÛŒ ØµØ­Øª Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§



train_data.head()











import supplemental_english  # Ø§ÛŒÙ…Ù¾ÙˆØ±Øª Ù�Ø§ÛŒÙ„ Ø¯Ø§Ø¯Ù‡â€ŒØ´Ø¯Ù‡

# Ø®ÙˆØ§Ù†Ø¯Ù† Ø¯ÛŒÚ©Ø´Ù†Ø±ÛŒ REGION_CODES Ø§Ø² Ù�Ø§ÛŒÙ„
region_code = supplemental_english.REGION_CODES  

# Ø§ÛŒØ¬Ø§Ø¯ ÛŒÚ© Ø¯ÛŒÚ©Ø´Ù†Ø±ÛŒ Ø¨Ø±Ø§ÛŒ Ù†Ú¯Ø§Ø´Øª Ú©Ø¯ Ù…Ù†Ø·Ù‚Ù‡ Ø¨Ù‡ Ù†Ø§Ù… Ù…Ù†Ø·Ù‚Ù‡
region_mapping = {code.zfill(3): region for region, codes in region_code.items() for code in codes}

# ØªØ¨Ø¯ÛŒÙ„ Ø¯ÛŒÚ©Ø´Ù†Ø±ÛŒ Ø¨Ù‡ DataFrame
import pandas as pd

df = pd.DataFrame(region_mapping.items(), columns=['region_code', 'region_name'])

# Ù†Ù…Ø§ÛŒØ´ Ø®Ø±ÙˆØ¬ÛŒ
print(df)



# Ø¨Ø±Ø±Ø³ÛŒ Ú©Ø¯Ù‡Ø§ÛŒ Ø¯Ø§Ø®Ù„ Ø¯ÛŒÚ©Ø´Ù†Ø±ÛŒ
print("Unique region codes in region_code dictionary:")
print([key for key in region_code.keys()])

# Ø¨Ø±Ø±Ø³ÛŒ Ú©Ø¯Ù‡Ø§ÛŒ Ù…ÙˆØ¬ÙˆØ¯ Ø¯Ø± train_data
print("Unique region codes in train_data:")
print(df['region_code'].unique())



print(region_mapping)



import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

# Ø´Ù…Ø§Ø±Ø´ ØªØ¹Ø¯Ø§Ø¯ Ù…Ù†Ø§Ø·Ù‚ Ùˆ Ø§Ù†ØªØ®Ø§Ø¨ Û±Û° ØªØ§ÛŒ Ø¨Ø±ØªØ±
top_regions = df['region_name'].value_counts().nlargest(10).sort_values(ascending = True )

# Ø±Ø³Ù… Ù†Ù…ÙˆØ¯Ø§Ø±
plt.figure(figsize=(10, 6))
sns.barplot(x=top_regions.index, y=top_regions.values, palette='PiYG')

# ØªÙ†Ø¸ÛŒÙ…Ø§Øª Ù†Ù…ÙˆØ¯Ø§Ø±
plt.title('Top 10 Regions by Count', fontsize=15)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=90)  # Ú†Ø±Ø®Ø´ Ù†Ø§Ù… Ù…Ù†Ø§Ø·Ù‚ Ø¨Ø±Ø§ÛŒ Ø®ÙˆØ§Ù†Ø§ÛŒÛŒ Ø¨Ù‡ØªØ±

# Ù†Ù…Ø§ÛŒØ´ Ù†Ù…ÙˆØ¯Ø§Ø±
plt.show()



df.head()


train_data.head()


df.head()


sns.set_style('darkgrid')

# Ø´Ù…Ø§Ø±Ø´ ØªØ¹Ø¯Ø§Ø¯ Ù…Ù†Ø§Ø·Ù‚ Ùˆ Ø§Ù†ØªØ®Ø§Ø¨ Û±Û° ØªØ§ÛŒ Ø¨Ø±ØªØ±
top_regions = train_data['plate'].value_counts().nlargest(10).sort_values(ascending = True )

# Ø±Ø³Ù… Ù†Ù…ÙˆØ¯Ø§Ø±
plt.figure(figsize=(10, 6))
sns.barplot(x=top_regions.index, y=top_regions.values, palette='PiYG')

# ØªÙ†Ø¸ÛŒÙ…Ø§Øª Ù†Ù…ÙˆØ¯Ø§Ø±
plt.title('Top 10 plate by Count', fontsize=15)
plt.xlabel('plate', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=90)  # Ú†Ø±Ø®Ø´ Ù†Ø§Ù… Ù…Ù†Ø§Ø·Ù‚ Ø¨Ø±Ø§ÛŒ Ø®ÙˆØ§Ù†Ø§ÛŒÛŒ Ø¨Ù‡ØªØ±

# Ù†Ù…Ø§ÛŒØ´ Ù†Ù…ÙˆØ¯Ø§Ø±
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns


fig, axes = plt.subplots(3, 1, figsize=(10, 18))


top_regions = train_data['plate'].value_counts().nlargest(20).sort_values(ascending=True)
sns.barplot(x=top_regions.index, y=top_regions.values, palette='PiYG', ax=axes[0])
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)
axes[0].set_title('Top 20 plate')


top_regions_name = train_data['region_name'].value_counts().nlargest(20).sort_values(ascending=True)
sns.barplot(x=top_regions_name.index, y=top_regions_name.values, palette='PiYG', ax=axes[1])
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)
axes[1].set_title('Top 20 region name')


top_regions_code = train_data['region_code'].value_counts().nlargest(20).sort_values(ascending=True)
sns.barplot(x=top_regions_code.index, y=top_regions_code.values, palette='PiYG', ax=axes[2])
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=90)
axes[2].set_title('Top 20 region code')


plt.tight_layout()


plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

# Ù¾ÛŒØ¯Ø§ Ú©Ø±Ø¯Ù† 20 Ù…Ù†Ø·Ù‚Ù‡â€ŒÛŒ Ø¨Ø±ØªØ± Ø¨Ø± Ø§Ø³Ø§Ø³ ØªØ¹Ø¯Ø§Ø¯ Ø±Ø®Ø¯Ø§Ø¯Ù‡Ø§ÛŒ Ù…Ù†Ø·Ù‚Ù‡ (ÛŒØ§ Ø³Ø§ÛŒØ± Ù…Ø¹ÛŒØ§Ø±Ù‡Ø§ÛŒ Ø¯Ù„Ø®ÙˆØ§Ù‡)
top_regions = train_data['region_name'].value_counts().nlargest(20).index

# Ù�ÛŒÙ„ØªØ± Ú©Ø±Ø¯Ù† Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ Ø¨Ø±Ø§ÛŒ Ù†Ù…Ø§ÛŒØ´ Ù�Ù‚Ø· Ø§ÛŒÙ† 20 Ù…Ù†Ø·Ù‚Ù‡
filtered_data = train_data[train_data['region_name'].isin(top_regions)]

# Ù¾ÛŒØ¯Ø§ Ú©Ø±Ø¯Ù† Ù…ÛŒØ§Ù†Ú¯ÛŒÙ† Ù‚ÛŒÙ…Øª Ø¨Ø±Ø§ÛŒ Ø§ÛŒÙ† Ù…Ù†Ø§Ø·Ù‚
top_regions_avg_price = filtered_data.groupby('region_name')['price'].mean().nlargest(20).index

# Ù�ÛŒÙ„ØªØ± Ú©Ø±Ø¯Ù† Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ Ø¨Ø± Ø§Ø³Ø§Ø³ 20 Ù…Ù†Ø·Ù‚Ù‡â€ŒÛŒ Ø¨Ø±ØªØ± Ø¨Ø§ Ù…ÛŒØ§Ù†Ú¯ÛŒÙ† Ù‚ÛŒÙ…Øª
final_filtered_data = filtered_data[filtered_data['region_name'].isin(top_regions_avg_price)]

# Ø±Ø³Ù… Ù†Ù…ÙˆØ¯Ø§Ø±
plt.figure(figsize=(12, 6))
sns.lineplot(data=final_filtered_data, x='year', y='price', hue='region_name', palette='pastel')

plt.title('Price comparison in the top 20 regions based on average price')
plt.xlabel('year')
plt.ylabel('price')
plt.legend(title='Top 20 Regions', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()



import pandas as pd
import numpy as np
import re
import sys

sys.path.append("/kaggle/input/russian-car-plates-prices-prediction")
from supplemental_english import GOVERNMENT_CODES

# ğŸš€ 1. Ø¨Ù‡ÛŒÙ†Ù‡â€ŒØ³Ø§Ø²ÛŒ `GOVERNMENT_CODES`
def preprocess_government_codes(gov_codes):
    processed_codes = {}
    for (letters, num_range, region), importance_values in gov_codes.items():
        key = (letters, region)
        if key not in processed_codes:
            processed_codes[key] = []
        processed_codes[key].append({"range": num_range, "values": importance_values})
    return processed_codes

GOVERNMENT_CODES = preprocess_government_codes(GOVERNMENT_CODES)

# ğŸš€ 2. ØªØ§Ø¨Ø¹ Ø³Ø±ÛŒØ¹â€ŒØªØ± Ø¨Ø±Ø§ÛŒ Ø§Ø³ØªØ®Ø±Ø§Ø¬ ÙˆÛŒÚ˜Ú¯ÛŒâ€ŒÙ‡Ø§ÛŒ Ù¾Ù„Ø§Ú©
def extract_plate_features(plate, gov_codes):
    match = re.match(r"([A-Z]+)(\d+)(\d{2,3})", plate)
    if not match:
        return (0, 0, 0, "unknown")

    letters, numbers, region = match.groups()
    numbers = int(numbers)
    key = (letters, region)

    if key in gov_codes:
        for entry in gov_codes[key]:
            if entry["range"][0] <= numbers <= entry["range"][1]:
                forbidden, advantage, significance = entry["values"][1], entry["values"][2], entry["values"][3]
                plate_type = "government" if forbidden else "semi-government" if advantage else "private"
                return (forbidden, advantage, significance, plate_type)

    return (0, 0, 0, "private")

# ğŸš€ 3. Ø§Ù�Ø²ÙˆØ¯Ù† ÙˆÛŒÚ˜Ú¯ÛŒâ€ŒÙ‡Ø§ÛŒ Ù¾Ù„Ø§Ú© Ø¨Ù‡ Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ…
def add_plate_features(data):
    features = data["plate"].apply(lambda plate: extract_plate_features(plate, GOVERNMENT_CODES))
    data[["forbidden_to_buy", "advantage_on_road", "significance", "plate_type"]] = pd.DataFrame(features.tolist(), index=data.index)
    return data

# ğŸš€ 4. Ù…Ø­Ø§Ø³Ø¨Ù‡ Ù…Ø­Ø¨ÙˆØ¨ÛŒØª Ú©Ø¯ Ù…Ù†Ø·Ù‚Ù‡ (`region_popularity`)
def add_region_popularity(data):
    region_counts = data["plate"].apply(lambda plate: plate[6:]).value_counts()
    data["region_popularity"] = data["plate"].apply(lambda plate: region_counts.get(plate[6:], 0))
    return data

# ğŸš€ 5. Ø¨Ø§Ø±Ú¯Ø°Ø§Ø±ÛŒ Ùˆ Ù¾Ø±Ø¯Ø§Ø²Ø´ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§
data = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv", dtype={"id": int, "plate": str, "price": float}, parse_dates=["date"])

data["year"] = data["date"].dt.year - 2021
data["plate"] = data["plate"].apply(lambda plate: plate if len(plate) == 9 else f"{plate[:6]}0{plate[6:]}")
data = add_plate_features(data)
data = add_region_popularity(data)

data.drop(columns=["id", "date"], inplace=True)

print(data.head())



import lightgbm as lgb
from sklearn.model_selection import train_test_split
from supplemental_english import * 

def smape(actual, forecast):
    denominator = (np.abs(actual) + np.abs(forecast)) / 2.0
    diff = np.abs(actual - forecast) / denominator
    return 100 * np.mean(diff)

# Custom SMAPE evaluation
def smape_eval(y_pred, y_true):
    y_true = y_true.get_label()
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    return 'smape', 100 * np.mean(diff), False
    
df = data
region_to_name = {code: region for region, codes in REGION_CODES.items() for code in codes}

# Feature engineering
def extract_features(plate):
    match = re.match(r"([A-Z])([0-9]{3})([A-Z]{2})([0-9]{1,3})", plate)
    if not match:
        return pd.Series(["Unknown", "U", "000", "UU", 0, 0, 0, 0, len(plate), 0])
    prefix, numbers, suffix, region = match.groups()
    num_val = int(numbers)
    region_name = region_to_name.get(region, "Unknown")
    is_repeating = len(set(numbers)) == 1
    is_round = num_val % 100 == 0
    is_low = num_val <= 10
    plate_len = len(plate)
    is_symmetric = plate[:3] == plate[-3:][::-1]
    return pd.Series([region_name, prefix, numbers, suffix, num_val, 
                      is_repeating, is_round, is_low, plate_len, is_symmetric])

feature_cols = ["region_name", "prefix", "numbers", "suffix", "num_val", 
                "is_repeating", "is_round", "is_low", "plate_len", "is_symmetric"]
df[feature_cols] = df["plate"].apply(extract_features)

# Preprocessing
le_region = LabelEncoder()
le_prefix = LabelEncoder()
le_suffix = LabelEncoder()

df["region_name"] = df["region_name"].fillna("Unknown")
df["prefix"] = df["prefix"].fillna("U")
df["suffix"] = df["suffix"].fillna("UU")

new_plate = "A007MP77"
new_features = extract_features(new_plate)
df_extended = pd.concat([df, pd.DataFrame([new_features], columns=feature_cols)], ignore_index=True)

le_region.fit(df_extended["region_name"])
le_prefix.fit(df_extended["prefix"])
le_suffix.fit(df_extended["suffix"])

df["region_name"] = le_region.transform(df["region_name"])
df["prefix"] = le_prefix.transform(df["prefix"])
df["suffix"] = le_suffix.transform(df["suffix"])
df.fillna(0, inplace=True)

# Features and target (log-transformed)
X = df[["region_name", "prefix", "num_val", "suffix", "is_repeating", "is_round", 
        "is_low", "forbidden_to_buy", "advantage_on_road", "significance", 
        "plate_len", "is_symmetric", "year"]]
y = np.log1p(df["price"])  # Log transform prices

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM Dataset
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Parameters
params = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1
}

# Train with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[val_data],
    feval=smape_eval,
    callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=1)]
)

# Final SMAPE (on original scale)
y_pred_log = model.predict(X_val)
y_pred = np.expm1(y_pred_log)
y_val_orig = np.expm1(y_val)
smape_score = smape(y_val_orig, y_pred)
print(f"\nFinal SMAPE on validation set: {smape_score:.2f}%")

# Prediction
new_df = pd.DataFrame([new_features], columns=feature_cols)
new_df["year"] = 2025
new_df["forbidden_to_buy"] = 0
new_df["advantage_on_road"] = 0
new_df["significance"] = 0
new_df["region_name"] = le_region.transform([new_df["region_name"][0]])[0]
new_df["prefix"] = le_prefix.transform([new_df["prefix"][0]])[0]
new_df["suffix"] = le_suffix.transform([new_df["suffix"][0]])[0]
new_df.fillna(0, inplace=True)
new_X = new_df[X.columns]
pred_price_log = model.predict(new_X)[0]
pred_price = np.expm1(pred_price_log)
print(f"\nPredicted price for {new_plate}: {pred_price:,.0f}")

# Feature importance
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importance(importance_type="gain")
}).sort_values("importance", ascending=False)
print("\nFeature Importance:")
print(feature_importance)

