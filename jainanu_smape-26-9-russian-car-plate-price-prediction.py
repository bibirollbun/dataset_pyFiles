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

# Load datasets
train = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')

# Quick preview
train.head()



import re

# Extract prefix, digits, and suffix
def split_plate(plate):
    match = re.match(r"([A-Z])(\d{3})([A-Z]{2})(\d{2,3})", plate)
    if match:
        prefix, digits, suffix, region = match.groups()
        return prefix, int(digits), suffix, int(region)
    else:
        return None, None, None, None

# Apply to train and test
train[['prefix', 'middle_digits', 'suffix', 'region_code']] = train['plate'].apply(
    lambda x: pd.Series(split_plate(x))
)
test[['prefix', 'middle_digits', 'suffix', 'region_code']] = test['plate'].apply(
    lambda x: pd.Series(split_plate(x))
)

train[['plate', 'prefix', 'middle_digits', 'suffix', 'region_code']].head()



# Step 2.1: Load REGION_CODES dictionary
import sys
sys.path.append('data')  # Add the folder where supplemental_english.py lives
from supplemental_english import REGION_CODES

# Step 2.2: Reverse mapping to create code -> name mapping
CODE_TO_REGION = {}
for region, codes in REGION_CODES.items():
    for code in codes:
        CODE_TO_REGION[int(code)] = region

# Step 2.3: Map region_code to region_name
train['region_name'] = train['region_code'].map(CODE_TO_REGION)
test['region_name'] = test['region_code'].map(CODE_TO_REGION)

train[['region_code', 'region_name']].drop_duplicates().head(10)



# Compute average price per region from train set
region_avg_price = train.groupby('region_name')['price'].mean().to_dict()

# Map to train/test sets
train['region_avg_price'] = train['region_name'].map(region_avg_price)
test['region_avg_price'] = test['region_name'].map(region_avg_price)

# Preview
train[['region_name', 'region_avg_price']].drop_duplicates().head(10)



# Convert to datetime format
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

# Extract features
for df in [train, test]:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# Preview
train[['date', 'year', 'month', 'day', 'dayofweek', 'is_weekend']].head()



# Define prestige scores for individual letters
letter_scores = {
    'A': 1.0, 'B': 0.3, 'C': 0.6, 'E': 0.5, 'H': 0.7, 'K': 0.7, 'M': 0.8, 'O': 0.9, 'P': 0.6,
    'T': 0.8, 'X': 1.0, 'Y': 0.9, 'U': 0.4, 'D': 0.4, 'N': 0.4, 'R': 0.4
}

# Map scores to prefix and suffix letters
for df in [train, test]:
    df['first_letter_score'] = df['prefix'].map(letter_scores).fillna(0)
    df['last_letter_score'] = df['suffix'].str[-1].map(letter_scores).fillna(0)



# Score function for middle digits
def score_digits(digits):
    s = str(int(digits)).zfill(3)  # Pad to 3 digits
    if s[0] == s[1] == s[2]:  # All same
        return 1.0
    elif s[0] == s[2]:  # Palindrome
        return 0.7
    elif s in {'123', '321', '456', '654'}:  # Sequential
        return 0.6
    elif int(s) % 100 == 0:  # Round
        return 0.5
    elif s[0] == s[1] or s[1] == s[2]:  # Partial repeat
        return 0.4
    else:
        return 0.2

# Apply to datasets
for df in [train, test]:
    df['middle_digits_score'] = df['middle_digits'].apply(score_digits)



# Score prefix (first letter)
def score_prefix(p):
    p = str(p).upper()
    if p in {'X', 'A', 'M'}:
        return 1.0
    elif p in {'P', 'Y'}:
        return 0.7
    else:
        return 0.4

# Score suffix (letter combo)
def score_suffix(s):
    s = str(s).upper()
    if s in {'AY', '777', 'MH'}:
        return 1.0
    elif s in {'CP', 'TX'}:
        return 0.7
    else:
        return 0.4

# Apply scores
for df in [train, test]:
    df['first_letter_score'] = df['prefix'].apply(score_prefix)
    df['last_letter_score'] = df['suffix'].apply(score_suffix)
    df['plate_prestige_score'] = (
        0.5 * df['middle_digits_score'] +
        0.25 * df['first_letter_score'] +
        0.25 * df['last_letter_score']
    )



# Insert after plate parsing but before defining feature_cols
def add_plate_features(df):
    df['plate_length'] = df['plate'].str.len()
    df['digit_count'] = df['plate'].str.count(r'\d')
    df['is_repeating_digits'] = df['middle_digits'].apply(lambda x: int(str(x)[0]*len(str(x)) == str(x)) if pd.notnull(x) else 0)
    df['is_palindrome'] = df['plate'].apply(lambda x: int(str(x) == str(x)[::-1]) if pd.notnull(x) else 0)
    return df

train = add_plate_features(train)
test = add_plate_features(test)

train[['plate', 'plate_length', 'digit_count', 'is_repeating_digits', 'is_palindrome']].head()



# Cleaned and focused feature list
# Cleaned and expanded feature list
feature_cols = [
    'region_avg_price',     # macro signal
    'region_code',          # numeric region code
    'year', 'month',        # seasonal trend
    'prefix',               # first letter of plate
    'middle_digits',        # numeric digits in middle
    'suffix',               # last letter of plate
    'plate_prestige_score', # composite plate prestige
    'plate_length',         # NEW
    'digit_count',          # NEW
    'is_repeating_digits',  # NEW
    'is_palindrome'         # NEW
]


# Prepare training data
X_train = train[feature_cols]
y_train = train['price']

# Prepare test data (no 'price' column)
X_test = test[feature_cols]

# Check shape and preview
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
X_train.head()



!pip install autogluon --quiet


from autogluon.tabular import TabularPredictor



# Log-transform the target
import numpy as np
y_train = np.log1p(train['price'])  # log1p handles log(0) safely



from autogluon.tabular import TabularPredictor

# Combine features and target
train_ag = X_train.copy()
train_ag['price'] = y_train  # target is now log(price)

# Fit model
predictor = TabularPredictor(
    label='price',
    eval_metric='root_mean_squared_error'  # Better with log targets
).fit(
    train_data=train_ag,
    time_limit=600,
    presets='best_quality'
)



# Predict log(price)
y_pred_log = predictor.predict(X_test)

# Convert back to actual price
y_pred = np.expm1(y_pred_log)  # inverse of log1p



from sklearn.metrics import mean_absolute_error

# Predict on training set to estimate SMAPE
y_pred_train_log = predictor.predict(X_train)
y_pred_train = np.expm1(y_pred_train_log)
y_true = train['price']

# SMAPE function
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    return np.mean(diff) * 100

# Evaluate
smape_score = smape(y_true, y_pred_train)
print(f"Validation SMAPE: {smape_score:.4f}%")



# Predict log(price)
y_pred_test_log = predictor.predict(X_test)

# Inverse log1p to get back original price
y_pred_test = np.expm1(y_pred_test_log)



# Load sample submission template
submission = pd.read_csv("data/sample_submission.csv")

# Assign predictions
submission['price'] = y_pred_test

# Quick check
submission.head()



# Save the file for Kaggle submission
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv is ready to upload!")


