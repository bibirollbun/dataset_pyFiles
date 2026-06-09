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


import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
colors = ['blue','green','orange','red','purple','grey','pink','cyan'] #colors we will need in viualization


train_users = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip')
test_users = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/test_users.csv.zip')

# Merge train and test users
df = pd.concat((train_users, test_users), axis=0, ignore_index=True)

print("We have", train_users.shape[0], "users in the training set and", test_users.shape[0], "in the test set.")
print("So, we have in total", train_users.shape[0] + test_users.shape[0], "users.")


df


df.info()


df.describe()


df.describe(include=['object'])


counter = int(df.shape[0])
counternull = int(df.isna().sum().sum())
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.rcParams["figure.figsize"] = (6,5)


plt.figure(figsize=(9,6))
sns.distplot(x=df['age']);
plt.xlabel('Age')
plt.title('Age Distribution')
sns.despine();


plt.figure(figsize=(9,6))
sns.distplot(x=df['signup_flow'])
plt.xlabel('Signup Flow')
plt.title('Signup Flow Distribution')
sns.despine()


plt.figure(figsize=(9,6))
counts = df['gender'].fillna('NaN').value_counts(dropna=False)
counts_order = counts.index
sns.countplot(x=df['gender'].fillna('NaN'), order=counts_order)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')
for i in range(counts.shape[0]):
    plt.text(i, counts[i]+1200, f"{counts[i]/df.shape[0]*100:0.2f}%", ha='center', fontsize=10)
sns.despine()


plt.figure(figsize=(9,6))
counts = df['signup_method'].value_counts()
counts_order = counts.index
sns.countplot(x=df['signup_method'], order=counts_order)
plt.xlabel('Signup Method')
plt.ylabel('Count')
plt.title('Signup Method Distribution')
for i in range(counts.shape[0]):
    plt.text(i, counts[i]+1200, f"{counts[i]/df.shape[0]*100:0.2f}%", ha='center', fontsize=10)
sns.despine()


plt.figure(figsize=(10,7))
counts = df['affiliate_provider'].value_counts()
counts_order = counts.index
sns.countplot(y=df['affiliate_provider'], order=counts_order)
plt.ylabel('Affiliate Provider')
plt.xlabel('Count')
plt.title('Affiliate Provider Distribution')
for i in range(counts.shape[0]):
    plt.text(counts[i]+5200, i+0.17, f"{counts[i]/df.shape[0]*100:0.2f}%", ha='center', fontsize=9)
sns.despine()


plt.figure(figsize=(10,6))
counts = df['affiliate_channel'].value_counts()
counts_order = counts.index
sns.countplot(y=df['affiliate_channel'], order=counts_order)
plt.ylabel('Affiliate Channel')
plt.xlabel('Count')
plt.title('Affiliate Channel Distribution')
for i in range(counts.shape[0]):
    plt.text(counts[i]+5200, i+0.09, f"{counts[i]/df.shape[0]*100:0.2f}%", ha='center', fontsize=10)
sns.despine()


plt.figure(figsize=(12,6))
counts = df['language'].value_counts()
counts_order = counts.index
sns.countplot(x=df['language'], order=counts_order)
plt.xlabel('Language')
plt.ylabel('Count')
plt.title('Language Distribution')
for i in range(counts.shape[0]):
    plt.text(i, counts[i]+1000, f"{counts[i]/df.shape[0]*100:0.2f}%", ha='center', fontsize=6)
sns.despine()


plt.figure(figsize=(10,6))
counts = df['first_device_type'].value_counts()
counts_order = counts.index
sns.countplot(data=df, y='first_device_type', order=counts_order);
plt.ylabel('First Device Type')
plt.xlabel('Count')
plt.title('First Device Type Distribution')
for i in range(counts.shape[0]):
    plt.text(counts[i]+4000, i+0.09, f"{counts[i]/df.shape[0]*100:0.2f}%", ha='center', fontsize=10)
sns.despine()


# Ensure 'date_first_booking' is in datetime format
df['date_first_booking'] = pd.to_datetime(df['date_first_booking'], errors='coerce')
plt.figure(figsize=(10, 7))
# Extract month names
months_freq = df['date_first_booking'].dropna().dt.month_name().str[:3]
# Count the occurrences of each month
counts = months_freq.value_counts()
# Order the counts by index
counts_order = counts.index
# Plot the count of bookings by month
sns.countplot(x=months_freq, order=counts_order)
plt.xlabel('Booking Date Month')
plt.ylabel('Count')
plt.title('Booking Date Month Distribution')
# Annotate each bar with the percentage
for i, count in enumerate(counts):
    percentage = count / months_freq.shape[0] * 100
    plt.text(i, count + 10, f"{percentage:.2f}%", ha='center', fontsize=9)

sns.despine()
plt.show()


plt.figure(figsize=(10,7))
week_days_freq = df['date_first_booking'].dropna().dt.day_name()
counts = week_days_freq.value_counts()
counts_order = counts.index
sns.countplot(x=week_days_freq, order=counts_order)
plt.xlabel('Booking Date Week Day')
plt.ylabel('Count')
plt.title('Booking Date Week Day Distribution')
for i in range(counts.shape[0]):
    plt.text(i, counts[i]+200, f"{counts[i]/week_days_freq.shape[0]*100:0.2f}%", ha='center', fontsize=9.5)
sns.despine()


plt.figure(figsize=(10,7))
counts = df['country_destination'].value_counts()
counts_order = counts.index
sns.countplot(x=df['country_destination'], order=counts_order);
plt.xlabel('Destination Country')
plt.ylabel('Count')
plt.title('Destination Country Distribution')
for i in range(counts.shape[0]):
    plt.text(i, counts[i]+1000, f"{counts[i]/df.shape[0]*100:0.2f}%", ha='center', fontsize=10)
sns.despine();


plt.figure(figsize=(15,8))
sns.countplot(data=df, x='country_destination', hue='gender')
plt.xlabel('Destination Country')
plt.ylabel('Count')
plt.title('Destination Country Distribution Per Gender')
sns.despine()


plt.figure(figsize=(15,8))
non_en_users_other = df[np.logical_and(df['language']!='en', np.logical_not(df['country_destination'].isin(['NDF', 'US', 'AU', 'GB'])))]
g = sns.countplot(data=non_en_users_other, x='country_destination', hue='language')
plt.xlabel('Destination Country')
plt.ylabel('Count')
plt.title('Destination Country (Excluding \'NDF\', \'US\', \'GB\' and \'AU\') Distribution Per Language (Excluding \'en\')')
sns.move_legend(g, "upper right")
sns.despine()


fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(14, 8))
sns.boxplot(data=df, y='age', x='country_destination', ax=ax)
plt.xlabel('Destination Country')
plt.ylabel('Age')
plt.title('Destination Country Distribution Per Age')
sns.despine()


plt.figure(figsize=(10,7))
gender_ndf = pd.concat([df['gender'].fillna('NaN'), np.logical_not(df.date_first_booking.rename('Booked').isna())], axis=1)
g = sns.countplot(data=gender_ndf, x='Booked', hue='gender')
plt.xlabel('Status')
plt.ylabel('Count')
plt.title('Whether Members Booked Per Gender')
sns.despine()


plt.figure(figsize=(10,7))
signup_ndf = pd.concat([df['signup_method'], np.logical_not(df.date_first_booking.isna())], axis=1)
sns.countplot(data=signup_ndf, x='date_first_booking', hue='signup_method')
plt.xlabel('Status')
plt.ylabel('Count')
plt.title('Whether Members Booked Per Signup Method')
sns.despine()


for col in df.select_dtypes("object").columns:
    if len(df[col].unique()) < 26:
        print("-"*25, f"\n{col} Column\n", "-"*25)
        display(df[col].value_counts())
    elif col != 'id':
        print("-"*25, f"\n{col} Column (Top 10)\n", "-"*25)
        display(df[col].value_counts()[:10])


#Finding the number of null entries in each column.
for col in df.columns:
    num_null_values = df[col].isnull().sum()
    if num_null_values != 0:
        print(col + " has {} null values.".format(num_null_values))
        print()


print(f"---------------------------------------------\n Dataset [Total Records: {df.shape[0]:,}]\n---------------------------------------------")
nulls = round((df.isnull().sum(axis=0)/df.shape[0])*100, 2)
display(nulls[nulls > 0].sort_values(ascending=False))


def handle_unknown(df):
    """Reduce values '-unknown-' in both gender and first_browser columns to np.nan"""
    print(f"Operation affected {df.loc[df[df['gender'] == '-unknown-'].index].shape[0]:,} records in the gender column.")
    df.loc[df[df['gender'] == '-unknown-'].index, 'gender'] = np.nan
    display(df['gender'].value_counts(dropna=False))
    
    print(f"\nOperation affected {df.loc[df[df['first_browser'] == '-unknown-'].index].shape[0]:,} records in the first_browser column.")
    df.loc[df[df['first_browser'] == '-unknown-'].index, 'first_browser'] = np.nan
    display(df['first_browser'].value_counts(dropna=False))


# gender and first_browser column in user_data dataset
handle_unknown(df)


def calc_age_from_bdate(df, LB=1900, UB=2000):
    """Calculate the age by subtracting the presumed birth date from the account creation date."""
    # Ensure 'date_account_created' is in datetime format
    df['date_account_created'] = pd.to_datetime(df['date_account_created'], errors='coerce')

    # Filter the index based on the age range
    bdate_index = df[(df['age'] > LB) & (df['age'] < UB)].index
    print(f"Operation affected {len(bdate_index):,} records.")

    # Calculate the age
    df.loc[bdate_index, 'age'] = df.loc[bdate_index, 'date_account_created'].dt.year - df.loc[bdate_index, 'age']

def nullify_age_outliers(df, LB=18, UB=95):
    """Reduce records containing age values outside the interval [LB, UB] exclusive to np.nan."""
    # Nullify ages greater than the upper bound
    upper_outliers = df[df['age'] > UB].index
    print(f"Upperbound Operation affected {len(upper_outliers):,} records.")
    df.loc[upper_outliers, 'age'] = np.nan

    # Nullify ages less than the lower bound
    lower_outliers = df[df['age'] < LB].index
    print(f"Lowerbound Operation affected {len(lower_outliers):,} records.")
    df.loc[lower_outliers, 'age'] = np.nan


print("-"*6, "Records with above normal age values", "-"*6)
df[df['age'] > 95]['age'].describe()


print("-"*6, "Records with age values below legal age", "-"*6)
df[df['age'] < 18]['age'].describe()


df[df['age'] > 95]['age'].unique()


# Calculate the age by subtracting the presumed birth date from the account creation date 

calc_age_from_bdate(df)


# Reduce records containing age values outside the interval [18, 120] exclusive to np.nan
nullify_age_outliers(df)


df[np.logical_or(df['age'] < 18, df['age'] > 95)]['age'].unique()


def fill_age_nulls(df):
    """Fill the null values in both age column.
    \nMust set the Constant MEDIAN_AGE before using the function."""
    
    print(f"Operation affected {df.isna().sum().age:,} records in the users dataset.")
    df.age.fillna(MEDIAN_AGE, inplace=True)
    
    
def fill_fat_nulls(df):
    """Fill the null values in first_affiliate_tracked column.
    \nMust set the Constant FIRST_AFFILIATE_TRACKED before using the function."""
    
    print(f"Operation affected {df.isna().sum().first_affiliate_tracked:,} records in the users dataset.")
    df.first_affiliate_tracked.fillna(FIRST_AFFILIATE_TRACKED, inplace=True)


# Updated null values percentage 
(df.isna().sum() / df.shape[0])[df.isna().sum() > 0]


# Creating a copy to keep the original dataframe without filling the nulls
user_filled = df.copy()


# Age Column (Imputing with Central Tendency Measure)
# The median could be better since the age distribution is skewed
MEDIAN_AGE = df.age.median()

fill_age_nulls(user_filled)


# First Affiliate Tracked Column (Imputing with 'untracked')
# Since it makes since from the business point of view
FIRST_AFFILIATE_TRACKED = 'untracked'

fill_fat_nulls(user_filled)


def split_dates(df, drop_original=False, verbose=True):
    """Splits any column with dtype of datetime64[ns] into 3 columns namely _year, _month, _day.
        Keeps the original column if drop_original is not True"""
    date_cols = df.select_dtypes('datetime64[ns]').columns
    if len(date_cols) == 0:
        print("No columns with dtype of datetime[ns]!")
        return None
    for date_col in date_cols:
        if verbose:
            print(f"Splitting {date_col} Column...")
        df[date_col+'_year'] = df[date_col].dt.year
        df[date_col+'_month'] = df[date_col].dt.month
        df[date_col+'_day'] = df[date_col].dt.day
    if drop_original:
        df.drop(date_cols, axis=1, inplace=True)


# Extract year, month, day data from date columns keeping the original columns for visual analysis

split_dates(df)
split_dates(user_filled, verbose=False)


def group_browsers(df, verbose=True):
    """Groups minor browsers that appear infrequently. Output two new categories Mobile and Others"""
    minors = df.first_browser.value_counts().index[np.where((df.first_browser.value_counts() < 500) == True)]
    mobile = []
    others = []
    for browser in minors:
        if 'Mobile' in browser:
            mobile.append(browser)
            df.first_browser.replace(to_replace=browser, value='Other Mobile', inplace=True)
        else:
            others.append(browser)
            df.first_browser.replace(to_replace=browser, value='Others', inplace=True)
    if verbose:
        print("Mobile Browsers Grouped: ", mobile)
        print("Other Browsers Grouped: ", others)


group_browsers(df)
group_browsers(user_filled, False)


df.first_browser.value_counts()


# Preview current dataframe
user_filled.head()


# List categorical features
cat_cols = ['gender', 'signup_method', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']

# Initialize One Hot Encoder (Since the features are nominal)
ohe_encoder = OneHotEncoder(handle_unknown='ignore')

# Fit the encoder on the categorical columns
ohe_encoder.fit(user_filled[cat_cols])


# Get the encoded features
ohe_encoded_train = ohe_encoder.transform(user_filled[cat_cols])

# get the features names
feat_ohe_names = ohe_encoder.get_feature_names_out()

# Construct Dataframe
ohe_encoded_train_df = pd.DataFrame(ohe_encoded_train.toarray(), columns=feat_ohe_names)

# Preview Dataframe
ohe_encoded_train_df.head()


# Separate Ground Truth Variable
y_train = user_filled['country_destination']

# Initialize LabelEncoder
lbl_encoder = LabelEncoder()

# Encode y
y_train = lbl_encoder.fit_transform(y_train)
y_train


user_filled.head()


# ID column and date columns we already split into year, month, day
cols_to_drop = ['id', 'date_account_created', 'timestamp_first_active', 'date_first_booking',]

# Categorical columns already encoded
cols_to_drop.extend(cat_cols)

# Training-specific columns
ground_truth_cols = ['country_destination']


# Dropping general redundant columns
user_filled.drop(cols_to_drop, axis=1, inplace=True)

# Dropping training-specific redundant columns
user_filled.drop(ground_truth_cols, axis=1, inplace=True)


# Preview training dataframe
user_filled.head()


# Concatenate merged_train with ohe_encoded_train_df
x_train = pd.concat([user_filled, ohe_encoded_train_df], axis=1)

# Preview current dimensions
x_train.shape


print(f"Shape of X: {x_train.shape}")
print(f"Shape of y: {y_train.shape}")


X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
params = {
    'objective': 'multi:softmax',  # Use 'multi:softprob' if you want probabilities
    'num_class': 13,  
    'max_depth': 6,
    'eta': 0.3,
    'eval_metric': 'mlogloss'
}

# Train the model
bst = xgb.train(params, dtrain, num_boost_round=100)

# Make predictions
preds = bst.predict(dtest)
predictions = [round(value) for value in preds]

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

