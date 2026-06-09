import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV


train_df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv")
test_df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")
sample_submission = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv")


import sys
sys.path.append('/kaggle/input/russian-car-plates-prices-prediction')
import supplemental_english


# Example
def get_region_name(region_code):
    for region, codes in REGION_CODES.items():
        if region_code in codes:
            return region
    return 'Unknown'
def get_gov_info(letters, number, region_code):
    for (prefix, (start, end), reg), (desc, forbidden, advantage, level) in GOVERNMENT_CODES.items():
        if letters == prefix and start <= number <= end and region_code == reg:
            return desc, forbidden, advantage, level
    return 'Normal', 0, 0, 0


train_df.head()


test_df.head()


train_df.info()
train_df.describe()


test_df.info()
test_df.describe()


test_df['price'] = np.nan


test_df.head()


# Box Plot
import seaborn as sns
sns.boxplot(train_df['price'])


upper_limit = train_df['price'].quantile(0.95)
train_df['price'] = np.where(train_df['price'] > upper_limit, upper_limit, train_df['price'])

# Box Plot
import seaborn as sns
sns.boxplot(train_df['price'])


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from supplemental_english import REGION_CODES, GOVERNMENT_CODES  # Ensure this module exists

def feature_engineering(df):
    # Convert date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek    
    
    # Extract plate components
    df['letter1'] = df['plate'].str.extract(r'([A-Z])')        # First Latin letter
    df['digits']  = df['plate'].str.extract(r'(\d{3})')         # 3 digits in the middle
    df['letter2'] = df['plate'].str.extract(r'([A-Z]{2})')      # Two Latin letters
    df['region']  = df['plate'].str.extract(r'(\d{2,3})$')  
    
    df['digits'] = pd.to_numeric(df['digits'], errors='coerce').fillna(-1).astype(int)
    df['region'] = pd.to_numeric(df['region'], errors='coerce').fillna(-1).astype(int)

    # Palindromes and other numeric features
    df['is_palindrome_letters'] = (df['letter1'] + df['letter2']).apply(lambda x: x == x[::-1] if isinstance(x, str) else False)
    df['is_palindrome_numbers'] = df['digits'].apply(lambda x: str(x) == str(x)[::-1] if x >= 0 else False)
    df['unique_letters_count'] = (df['letter1'] + df['letter2']).apply(lambda x: len(set(x)) if isinstance(x, str) else 0)
    df['unique_numbers_count'] = df['digits'].apply(lambda x: len(set(str(x))) if x >= 0 else 0)
    df['is_repeating_digits'] = df['digits'].apply(lambda x: len(set(str(x))) == 1 if x >= 0 else False)
    df['sum_digits'] = df['digits'].apply(lambda x: sum(int(d) for d in str(x)) if pd.notnull(x) else 0)
    df['is_beautiful'] = df['plate'].str.contains(r'000|111|777|999') 
    df['plate_length'] = df['plate'].str.len()

    # Frequency features
    for col in ['region', 'letter1', 'letter2']:
        freq = df[col].value_counts(normalize=True)
        df[col + '_freq'] = df[col].map(freq)
    
    # Add government tags using GOVERNMENT_CODES
    def get_gov_tags(row):
        for (letters, num_range, region), (_, is_forbidden, has_advantage, level) in GOVERNMENT_CODES.items():
            if row['letter1'] + row['letter2'] == letters and str(row['region']) == region:
                number = int(row['digits'])
                if num_range[0] <= number <= num_range[1]:
                    return pd.Series([is_forbidden, has_advantage, level])
        return pd.Series([0, 0, 0])

    df[['is_forbidden', 'has_advantage', 'significance_level']] = df.apply(get_gov_tags, axis=1)

    # Encode categorical columns
    for col in ['letter1', 'letter2', 'region']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df



train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)


import matplotlib.pyplot as plt

# Count values
train_counts = train_df['is_beautiful'].value_counts()
test_counts = test_df['is_beautiful'].value_counts()

# Convert to same order and fill missing values
all_keys = set(train_counts.index).union(set(test_counts.index))
train_counts = train_counts.reindex(all_keys, fill_value=0)
test_counts = test_counts.reindex(all_keys, fill_value=0)

# Create the plot
labels = [str(k) for k in all_keys]
x = range(len(labels))

bar_width = 0.35

plt.figure(figsize=(8, 5))
plt.bar([i - bar_width/2 for i in x], train_counts, width=bar_width, label='Train')
plt.bar([i + bar_width/2 for i in x], test_counts, width=bar_width, label='Test')

plt.xlabel('is_beautiful')
plt.ylabel('Count')
plt.title('Count of is_beautiful in Train and Test Sets')
plt.xticks(ticks=x, labels=labels)
plt.legend()
plt.tight_layout()
plt.show()



print("train:  ",train_df['is_beautiful'].value_counts())
print("___________________________")
print("test:  ", test_df['is_beautiful'].value_counts())


train_df['log_price'] = np.log1p(train_df['price'])
# Drop rows where 'price' or 'log_price' is missing
train_df = train_df.dropna(subset=['price', 'log_price'])

# Define features and target
exclude_cols = ['price', 'log_price', 'id', 'plate', 'date']
features = [col for col in train_df.columns if col not in exclude_cols]
X = train_df.drop(columns=['price','log_price', 'id', 'plate', 'date'])
y = (train_df['log_price'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.07, random_state=42)


train_df.isna().sum()


X_train.columns


# SMAPE definition
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from xgboost import XGBRegressor

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff  = np.abs(y_true - y_pred)
    return np.mean(diff / denom) * 100

# Scorer (note: greater_is_better=False)
smape_scorer = make_scorer(lambda yt, yp: smape(np.expm1(yt), np.expm1(yp)), greater_is_better=False)


model = XGBRegressor(
        max_depth=9,
        colsample_bytree=0.9,
        subsample=0.9,
        n_estimators=100000,
        learning_rate=0.005,
        random_state=42,
        eval_metric="rmse")
model.fit(X_train, y_train, verbose=10)


# Predictions and inverse transform
y_val_pred_log = model.predict(X_val[features])
y_val_pred     = np.expm1(y_val_pred_log)
y_true         = np.expm1(y_val)

mae   = mean_absolute_error(y_true, y_val_pred)
rmse  = np.sqrt(mean_squared_error(y_true, y_val_pred))
smape_score = smape(y_true, y_val_pred)

print(f"SMAPE: {smape_score:.2f}%")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


exclude_cols = ['id', 'plate', 'date', 'price']
test_features = [col for col in test_df.columns if col not in exclude_cols]


# Predict and save
test_X = test_df[test_features]
test_pred_log = model.predict(test_X)
test_pred     = np.expm1(test_pred_log).astype(int)

submission = pd.DataFrame({
    'id': test_df['id'],
    'price': test_pred
})
submission.to_csv('submission.csv', index=False)


submission.head()


# Filter by plate
price_row = submission[test_df['plate'] == 'P700TT790']

# Get the predicted price
predicted_price = price_row['price'].values[0] if not price_row.empty else None
print(predicted_price)

