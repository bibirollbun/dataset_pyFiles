# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


identity_df = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
identity_df.head()


trx_id1 = list(set(identity_df.TransactionID.unique()))
len(trx_id1)


transaction_df = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
transaction_df


trx_id2 = list(set(transaction_df.TransactionID.unique()))
len(trx_id2)


trx_id_diff = list(set(trx_id2) - set(trx_id1))
print(len(trx_id_diff))
len(trx_id2) == len(trx_id_diff) + len(trx_id1)


transaction_df['isFraud'].value_counts(1) * 100


for i in range(1, 7):
    transaction_df[f'card{i}'] = transaction_df[f'card{i}'].astype('object')

for i in range(1, 10):
    transaction_df[f'M{i}'] = transaction_df[f'M{i}'].astype('object')

for i in range(1, 3):
    transaction_df[f'addr{i}'] = transaction_df[f'addr{i}'].astype('object')


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == object: 
            continue

        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)  
        else:
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    return df


identity_df = optimize_memory(identity_df)
transaction_df = optimize_memory(transaction_df)


cat_cols_identity = [col for col in identity_df.columns if identity_df[col].dtype == 'object']
num_cols_identity = [col for col in identity_df.columns if identity_df[col].dtype != 'object']
cat_cols_transaction = [col for col in transaction_df.columns if transaction_df[col].dtype == 'object']
num_cols_transaction = [col for col in transaction_df.columns if transaction_df[col].dtype != 'object']


print(transaction_df.shape)
transaction_df['TransactionID'].nunique()


transaction_df[[col for col in transaction_df.columns if 'card' in col] + ['addr1', 'addr2'] + ['P_emaildomain', 'R_emaildomain']]


transaction_df[[col for col in transaction_df.columns if 'card' in col] + ['addr1', 'addr2'] + ['P_emaildomain', 'R_emaildomain']].info()


print(transaction_df['card2'].min(), transaction_df['card2'].max())
print(transaction_df['card3'].min(), transaction_df['card3'].max())
print(transaction_df['card5'].min(), transaction_df['card5'].max())
print(transaction_df['addr1'].min(), transaction_df['addr1'].max())
print(transaction_df['addr2'].min(), transaction_df['addr2'].max())


transaction_df['card2'].fillna(0, inplace=True)
transaction_df['card3'].fillna(0, inplace=True)
transaction_df['card5'].fillna(0, inplace=True)
transaction_df['addr1'].fillna(0, inplace=True)
transaction_df['addr2'].fillna(0, inplace=True)


print(transaction_df['card1'].nunique())
print(transaction_df['card2'].nunique())
print(transaction_df['card3'].nunique())
print(transaction_df['card4'].nunique())
print(transaction_df['card5'].nunique())
print(transaction_df['card6'].nunique())


transaction_df['id1'] = transaction_df['card1'].astype(str) + '_' + transaction_df['card2'].astype(str)
transaction_df['id2'] = transaction_df['id1'].astype(str) + '_' + transaction_df['card3'].astype(str) + '_' + transaction_df['card5'].astype(str)
transaction_df['id3'] = transaction_df['id2'].astype(str) + '_' + transaction_df['addr1'].astype(str) + '_' + transaction_df['addr2'].astype(str)
transaction_df['id4'] = transaction_df['id3'].astype(str) + '_' + transaction_df['card4'].astype(str) + '_' + transaction_df['card6'].astype(str)





print(transaction_df['id1'].nunique())
print(transaction_df['id2'].nunique())
print(transaction_df['id3'].nunique())
print(transaction_df['id4'].nunique())


def group_by_with_label(df, col, label_col):
    grouped_df = df.groupby(col)[label_col].value_counts().unstack(fill_value=0)
    grouped_df.columns = [f'{label_col}_0', f'{label_col}_1']
    grouped_df['total'] = (grouped_df[f'{label_col}_1'] + grouped_df[f'{label_col}_0'])
    grouped_df[f"{label_col}_rate"] = grouped_df[f'{label_col}_1'] / grouped_df['total']
    grouped_df.reset_index(inplace=True)
    return grouped_df


x = group_by_with_label(transaction_df, 'id1', 'isFraud')
x[x.total > 10].sort_values('isFraud_rate', ascending=False)[:8]


x = group_by_with_label(transaction_df, 'id2', 'isFraud')
x[x.total > 10].sort_values('isFraud_rate', ascending=False)[:8]


x = group_by_with_label(transaction_df, 'id3', 'isFraud')
x[x.total > 15].sort_values('isFraud_rate', ascending=False)[:8]


x = group_by_with_label(transaction_df, 'id4', 'isFraud')
x[x.total > 15].sort_values('isFraud_rate', ascending=False)[:8]


x = group_by_with_label(transaction_df, 'card4', 'isFraud')
x.sort_values('isFraud_rate', ascending=False)


x = group_by_with_label(transaction_df, 'card6', 'isFraud')
x.sort_values('isFraud_rate', ascending=False)





transaction_df["TransactionDT"]


from datetime import datetime, date, timedelta

init_date = datetime(2018, 1, 1, 0, 0, 0) # asumption init date is 2018-1-1 as the competition was held in 2019

transaction_df["datetime"] = transaction_df.TransactionDT.apply(lambda x: (init_date + timedelta(seconds=x)))
transaction_df['datetime']



transaction_df['_month'] = transaction_df['datetime'].dt.month
transaction_df['_weekday'] = transaction_df['datetime'].dt.dayofweek
transaction_df['_hour'] = transaction_df['datetime'].dt.hour
transaction_df['_day'] = transaction_df['datetime'].dt.day


fig, ax = plt.subplots(4, 1, figsize=(12, 15)) 
plt.subplots_adjust(hspace=0.4) 

# Data to plot and their corresponding titles
groupings = ['_weekday', '_hour', '_day', '_month']
titles = [
    "Average Fraud Rate by Day of the Week",
    "Average Fraud Rate by Hour of the Day",
    "Average Fraud Rate by Day of the Month",
    "Average Fraud Rate by Month",
]
y_labels = ["Fraud Rate"] * 4
x_labels = [""] * 4

# Iterate through groupings and create bar plots
for i, (group, title, y_label, x_label) in enumerate(zip(groupings, titles, y_labels, x_labels)):
    avg_fraud_rate = transaction_df.groupby(group)['isFraud'].mean()
    avg_fraud_rate.plot(ax=ax[i], kind='line', marker='o')
    ax[i].set_title(title, fontsize=12)
    ax[i].set_ylabel(y_label, fontsize=8)
    ax[i].set_ylim(0, None)

    x_values = avg_fraud_rate.index
    ax[i].set_xticks(range(len(x_values)))
    ax[i].set_xticklabels(x_values, fontsize=6)
    if x_label:
        ax[i].set_xlabel(x_label, fontsize=12)



group_by_with_label(transaction_df, '_month', 'isFraud')


sns.histplot(transaction_df["TransactionAmt"])


import pandas as pd
import matplotlib.pyplot as plt

# Example: Number of bins (you can adjust this as needed)
num_bins = 10

# Create bins using pd.qcut
transaction_df['TransactionAmt_bin'] = pd.qcut(
    transaction_df['TransactionAmt'], 
    q=num_bins, 
    duplicates='drop'  # Handles duplicate bin edges if necessary
)

# Calculate fraud rate for each bin
fraud_rate_by_bin = transaction_df.groupby('TransactionAmt_bin')['isFraud'].mean()

# Plot the results
fig, ax = plt.subplots(figsize=(8, 4))
fraud_rate_by_bin.plot(kind='line', marker='o', ax=ax)
ax.set_title("Fraud Rate by Transaction Amount (Binned)", fontsize=16)
ax.set_xlabel("Transaction Amount Bin", fontsize=12)
ax.set_ylabel("Fraud Rate", fontsize=12)
ax.set_xticks(range(len(fraud_rate_by_bin)))
ax.set_xticklabels([str(interval) for interval in fraud_rate_by_bin.index], rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()



transaction_df['TransactionAmt_log'] = np.log(transaction_df["TransactionAmt"])
sns.histplot(transaction_df["TransactionAmt_log"])


sns.displot(transaction_df, x='TransactionAmt_log', hue='isFraud', kind='kde')


import pandas as pd
import matplotlib.pyplot as plt

def plot_label_rate_by_bin(df: pd.DataFrame, column_to_bin: str, target_col: str, num_bins=10):
    df_copy = df[[column_to_bin, target_col]].copy()
    
    df_copy[f'{column_to_bin}_bin'] = pd.qcut(
        df_copy[column_to_bin], 
        q=num_bins, 
        duplicates='drop'
    )
    
    fraud_rate_by_bin = df_copy.groupby(f'{column_to_bin}_bin')[target_col].mean()
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(8, 4))
    fraud_rate_by_bin.plot(kind='line', marker='o', ax=ax)
    ax.set_title(f"Rate by {column_to_bin} (Binned)", fontsize=16)
    ax.set_xlabel(f"{column_to_bin} Bin", fontsize=12)
    ax.set_ylabel("Label Rate", fontsize=12)
    ax.set_xticks(range(len(fraud_rate_by_bin)))
    ax.set_xticklabels([str(interval) for interval in fraud_rate_by_bin.index], rotation=45, ha='right')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()



plot_label_rate_by_bin(transaction_df, 'TransactionAmt_log', 'isFraud')


transaction_df['ProductCD'].info()


transaction_df['ProductCD'].value_counts()


transaction_df.groupby('ProductCD')['isFraud'].mean()


sns.boxplot(data=transaction_df, x='ProductCD', y='TransactionAmt', hue='isFraud')


sns.boxplot(data=transaction_df, x='ProductCD', y='TransactionAmt_log', hue='isFraud')


transaction_df['P_emaildomain'].fillna('unknown', inplace=True)
transaction_df['R_emaildomain'].fillna('unknown', inplace=True)


transaction_df['P_emaildomain'].value_counts(1).sort_values(ascending=False)[:15]


transaction_df['R_emaildomain'].value_counts(1).sort_values(ascending=False)[:15]


group_by_with_label(transaction_df, 'P_emaildomain', 'isFraud').sort_values('isFraud_rate', ascending=False)[:10]


group_by_with_label(transaction_df, 'P_emaildomain', 'isFraud').sort_values('isFraud_1', ascending=False)[:10]


group_by_with_label(transaction_df, 'P_emaildomain', 'isFraud').sort_values('total', ascending=False)[:10]


group_by_with_label(transaction_df, 'R_emaildomain', 'isFraud').sort_values('isFraud_rate', ascending=False)[:10]


group_by_with_label(transaction_df, 'R_emaildomain', 'isFraud').sort_values('total', ascending=False)[:10]





def missing_values_percentage(df):
  missing_values_count = df.isnull().sum()
  total_rows = len(df)
  missing_values_percentage = (missing_values_count / total_rows) * 100

  return missing_values_percentage


missing_values_percentage(transaction_df[[col for col in transaction_df.columns if col.startswith('C')]])


# plt.figure(figsize=(15, 12))
sns.heatmap(transaction_df[ [f'C{i}' for i in range(1, 14 + 1)] + ['isFraud']].corr(), annot=False)


def calculate_iv(data: pd.DataFrame, feature: str, target: str):
    crosstab = pd.crosstab(data[feature], data[target], normalize=False)
    crosstab.columns = ['Good', 'Bad']
    crosstab['Total'] = crosstab['Good'] + crosstab['Bad']
    crosstab['Good%'] = crosstab['Good'] / crosstab['Good'].sum()
    crosstab['Bad%'] = crosstab['Bad'] / crosstab['Bad'].sum()
    crosstab = crosstab[(crosstab['Good%'] > 0) & (crosstab['Bad%'] > 0)]
    crosstab['WOE'] = np.log(crosstab['Good%'] / crosstab['Bad%'])
    crosstab['IV'] = (crosstab['Good%'] - crosstab['Bad%']) * crosstab['WOE']
    return crosstab['IV'].sum()


iv_values = {f'C{i}': calculate_iv(transaction_df, f'C{i}', 'isFraud') for i in range(1, 14 + 1)}

iv_df = pd.DataFrame(list(iv_values.items()), columns=['Feature', 'IV']).sort_values(by='IV', ascending=False)
iv_df


strong_predictor = iv_df[iv_df.IV >= 0.5]['Feature'].unique().tolist()
medium_predictor = iv_df[(iv_df.IV >= 0.1) & (iv_df.IV < 0.5)]['Feature'].unique().tolist()
weak_predictor = iv_df[(iv_df.IV >= 0.05) & (iv_df.IV < 0.1)]['Feature'].unique().tolist()
print(strong_predictor)
print(medium_predictor)
print(weak_predictor)


import seaborn as sns
import matplotlib.pyplot as plt

def plot_dist_with_hue(df, column, hue_column):
    plt.figure(figsize=(6, 4))
    sns.displot(data=df, x=column, hue=hue_column, kind='kde')
    plt.title(f"Count Plot of {column} with Hue {hue_column}", fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(title=hue_column)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

# Example usage
# plot_count_with_hue(df, column='x', hue_column='y')



def plot_density_rate_by_bin(df: pd.DataFrame, col: str, target_col: str, num_bins=10):
    df0 = df[df[target_col] == 0][[col, target_col]].copy()
    df1 = df[df[target_col] == 1][[col, target_col]].copy()
    with np.errstate(invalid='ignore'):
        plt.figure(figsize=(8, 3))
        plt.hist([df0[col], df1[col]], bins=num_bins, density=True, label=['Negative', 'Positif'])
        
        plt.title(f'Density of {col} for Positive and Negative')
        plt.xlabel(f'{col} values')
        plt.ylabel('Density')
        
        # Add legend to distinguish fraud and non-fraud
        plt.legend()
        plt.show()


plot_label_rate_by_bin(transaction_df[transaction_df.C4 < 100], 'C4', 'isFraud', num_bins=10)
plot_density_rate_by_bin(transaction_df[transaction_df.C4 < 100], 'C4', 'isFraud', num_bins=10)
plot_dist_with_hue(transaction_df[transaction_df.C4 < 100], 'C4', 'isFraud')


missing_values_percentage(transaction_df[[col for col in transaction_df.columns if col.startswith('V')]])


transaction_df[[col for col in transaction_df.columns if col.startswith('V')]].describe()





plt.figure(figsize=(15, 12))
sns.heatmap(transaction_df[ [f'V{i}' for i in range(1, 337 + 1)] + ['isFraud']].corr(), annot=False, cmap='coolwarm')


iv_values = {f'V{i}': calculate_iv(transaction_df, f'V{i}', 'isFraud') for i in range(1, 337 + 1)}

iv_df = pd.DataFrame(list(iv_values.items()), columns=['Feature', 'IV']).sort_values(by='IV', ascending=False)
iv_df[:20]


strong_predictor = iv_df[iv_df.IV >= 0.5]['Feature'].unique().tolist()
medium_predictor = iv_df[(iv_df.IV >= 0.1) & (iv_df.IV < 0.5)]['Feature'].unique().tolist()
weak_predictor = iv_df[(iv_df.IV >= 0.05) & (iv_df.IV < 0.1)]['Feature'].unique().tolist()
print(strong_predictor)
print(medium_predictor)
print(weak_predictor)


missing_values_percentage(transaction_df[[col for col in transaction_df.columns if col.startswith('D')]])


transaction_df[[col for col in transaction_df.columns if col.startswith('D')]].describe()


# plt.figure(figsize=(15, 12))
sns.heatmap(transaction_df[ [f'D{i}' for i in range(1, 15 + 1)] + ['isFraud']].corr(), annot=False, cmap='coolwarm')


iv_values = {f'D{i}': calculate_iv(transaction_df, f'D{i}', 'isFraud') for i in range(1, 15 + 1)}

iv_df = pd.DataFrame(list(iv_values.items()), columns=['Feature', 'IV']).sort_values(by='IV', ascending=False)
iv_df[:20]


strong_predictor = iv_df[iv_df.IV >= 0.5]['Feature'].unique().tolist()
medium_predictor = iv_df[(iv_df.IV >= 0.1) & (iv_df.IV < 0.5)]['Feature'].unique().tolist()
weak_predictor = iv_df[(iv_df.IV >= 0.05) & (iv_df.IV < 0.1)]['Feature'].unique().tolist()
print(strong_predictor)
print(medium_predictor)
print(weak_predictor)


identity_df = pd.merge(identity_df, transaction_df[['isFraud', 'TransactionAmt', 'TransactionID']])
identity_df.head()


identity_df['isFraud'].value_counts(1) * 100


num_cols_corr1 = [f"id_{str(x)[1:]}" for x in range(101, 138) if f"id_{str(x)[1:]}" in num_cols_identity]

corr_matrix = identity_df[num_cols_corr1 + ['isFraud', 'TransactionAmt']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, fmt='.2f', cmap='coolwarm')
plt.show()


iv_values = {f'id_{str(i)[1:]}': calculate_iv(identity_df, f'id_{str(i)[1:]}', 'isFraud') for i in range(101, 139)}

iv_df = pd.DataFrame(list(iv_values.items()), columns=['Feature', 'IV']).sort_values(by='IV', ascending=False)
iv_df



strong_predictor = iv_df[iv_df.IV >= 0.5]['Feature'].unique().tolist()
medium_predictor = iv_df[(iv_df.IV >= 0.1) & (iv_df.IV < 0.5)]['Feature'].unique().tolist()
weak_predictor = iv_df[(iv_df.IV >= 0.05) & (iv_df.IV < 0.1)]['Feature'].unique().tolist()
print(strong_predictor)
print(medium_predictor)
print(weak_predictor)





plot_label_rate_by_bin(identity_df, 'id_25', 'isFraud', num_bins=6)
plot_density_rate_by_bin(identity_df, 'id_25', 'isFraud', num_bins=6)


plot_label_rate_by_bin(identity_df, 'id_25', 'isFraud', num_bins=8)
plot_density_rate_by_bin(identity_df, 'id_25', 'isFraud', num_bins=8)
plot_count_with_hue(identity_df, 'id_25', 'isFraud')


plot_label_rate_by_bin(identity_df, 'id_21', 'isFraud', num_bins=10)
plot_density_rate_by_bin(identity_df, 'id_25', 'isFraud', num_bins=10)


plot_label_rate_by_bin(identity_df, 'id_19', 'isFraud', num_bins=10)
plot_density_rate_by_bin(identity_df, 'id_19', 'isFraud', num_bins=10)


plot_label_rate_by_bin(identity_df, 'id_01', 'isFraud', num_bins=10)
plot_density_rate_by_bin(identity_df, 'id_01', 'isFraud', num_bins=10)


plot_label_rate_by_bin(identity_df, 'id_02', 'isFraud', num_bins=10)
plot_density_rate_by_bin(identity_df, 'id_02', 'isFraud', num_bins=10)


identity_df[cat_cols_identity + ['isFraud', 'TransactionAmt']]


identity_df['DeviceInfo'] = identity_df['DeviceInfo'].astype(str).fillna('unknown_device').str.lower()
identity_df['device_name'] = identity_df['DeviceInfo'].str.split('/', expand=True)[0].str.strip()
identity_df['device_name'].value_counts()[:20]


brand_dict = {
    'sm': 'Samsung', 'samsung': 'Samsung', 'gt-': 'Samsung',
    'moto g': 'Motorola', 'moto': 'Motorola',
    'lg-': 'LG', 'rv:': 'RV', 'huawei': 'Huawei', 'ale-': 'Huawei', '-l': 'Huawei',
    'blade': 'ZTE', 'linux': 'Linux', 'xt': 'Sony', 'htc': 'HTC', 'asus': 'Asus'
}

def map_to_brand(device_info):
    for key, brand in brand_dict.items():
        if key in device_info.lower():
            return brand
    return device_info

identity_df['brand_name'] = identity_df['device_name'].apply(map_to_brand)
identity_df['brand_name'].value_counts()[:15]


brand_count = identity_df['brand_name'].value_counts()
small_counts = brand_count[brand_count < 400].index
small_counts


identity_df['brand_name'] = identity_df['brand_name'].replace(small_counts, 'Others')
identity_df['brand_name'].value_counts()


group_by_with_label(identity_df, 'brand_name', 'isFraud')


joblib.dump(small_counts, 'small_counts_brand.pkl')


identity_df['OS_name'] = identity_df['id_30'].str.split(' ', expand=True)[0]


group_by_with_label(identity_df, 'OS_name', 'isFraud')


identity_df['browser_name'] = identity_df['id_31'].str.split(' ', expand=True)[0].str.split('/', expand=True)[0]
group_by_with_label(identity_df, 'browser_name', 'isFraud').sort_values('total', ascending=False)


brand_count = identity_df['browser_name'].value_counts()
small_counts_browser = brand_count[brand_count < 20].index
small_counts_browser


identity_df['browser_name'] = identity_df['browser_name'].replace(small_counts_browser, 'Others')
group_by_with_label(identity_df, 'browser_name', 'isFraud')


joblib.dump(small_counts_browser, 'small_counts_browser.pkl')


group_by_with_label(identity_df, 'id_33', 'isFraud')


identity_df['screen_width'] = identity_df['id_33'].str.split('x', expand=True)[0].fillna(0).astype(int)
identity_df['screen_height'] = identity_df['id_33'].str.split('x', expand=True)[1].fillna(0).astype(int)








identity_df['id_34_new'] = identity_df['id_34'].str.split(':', expand=True)[1].fillna(0).astype(int)
identity_df['id_23_new'] = identity_df['id_23'].str.split(':', expand=True)[1].fillna('UNKNOWN')


group_by_with_label(identity_df, 'id_23_new', 'isFraud')


group_by_with_label(identity_df, 'id_35', 'isFraud')


group_by_with_label(identity_df, 'id_05', 'isFraud')


cat_cols_identity


group_by_with_label(identity_df, 'id_12', 'isFraud')


group_by_with_label(identity_df, 'id_15', 'isFraud')


group_by_with_label(identity_df, 'id_35', 'isFraud')








transaction_test = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")
identity_test = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")


identity_test = optimize_memory(identity_test)
transaction_test = optimize_memory(transaction_test)


from datetime import datetime, date, timedelta

init_date = datetime(2018, 1, 1, 0, 0, 0) # asumption init date is 2018-1-1 as the competition was held in 2019

transaction_test["datetime"] = transaction_test.TransactionDT.apply(lambda x: (init_date + timedelta(seconds=x)))
transaction_test['_month'] = transaction_test['datetime'].dt.month
transaction_test['_weekday'] = transaction_test['datetime'].dt.dayofweek
transaction_test['_hour'] = transaction_test['datetime'].dt.hour
transaction_test['_day'] = transaction_test['datetime'].dt.day



group_by_with_label(transaction_df, '_month', 'isFraud')


transaction_test.groupby('_month')['TransactionID'].count()


transaction_test[transaction_test['R_emaildomain'].fillna('unknown').str.contains('proton')]['P_emaildomain'].count()

