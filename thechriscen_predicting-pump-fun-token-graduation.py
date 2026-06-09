import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import os
print(os.listdir('/kaggle/input'))


DIR = '/kaggle/input/pump-fun-graduation-february-2025'
train = pd.read_csv(os.path.join(DIR, 'train.csv'))
test  = pd.read_csv(os.path.join(DIR, 'test_unlabeled.csv'))
token_info = pd.read_csv(os.path.join(DIR, 'token_info_onchain_divers.csv'))
print(train.shape, train.columns)
print(test.shape,  test.columns)
train.info()
test.info()


import glob

# Find all chunk files in the directory
chunk_files = glob.glob(os.path.join(DIR, 'chunk*.csv'))
print(f"Found {len(chunk_files)} chunk files.")

# Load and concatenate all chunk files into one DataFrame
data = pd.concat([pd.read_csv(file) for file in chunk_files], ignore_index=True)
print("Transaction data shape:", data.shape)  # Expected: large, e.g., (17033442, 15)
print("Transaction columns:", data.columns.tolist())  # Includes 'base_coin', 'quote_coin_amount', etc.
print(data.head())
print(data.info())


total_sol_volume = data.groupby('base_coin')['quote_coin_amount'].sum().reset_index(name = 'total_sol_volume')
total_sol_volume = total_sol_volume.sort_values(by = 'total_sol_volume', ascending = False)

total_sol_volume


# Counts transactions per base_coin; no creator info needed
total_transactions = data.groupby('base_coin').size().reset_index(name='total_transactions')

total_transactions


# Counts distinct signing_wallets per base_coin; no creator info needed
unique_wallets = data.groupby('base_coin')['signing_wallet'].nunique().reset_index(name='unique_wallets')

unique_wallets


# Computes buy-to-sell ratio per base_coin; no creator info needed
buy_sell_ratio = data.groupby('base_coin').apply(
    lambda x: (x['direction'] == 'buy').sum() / (x['direction'] == 'sell').sum() if (x['direction'] == 'sell').sum() > 0 else 0
).reset_index(name='buy_sell_ratio')

buy_sell_ratio


#A- Fetch data of creator for each transactions base coin 
creator_data = data.merge(
    token_info[['mint', 'creator']]
    , left_on = 'base_coin'
    , right_on = 'mint'
    , how = 'left'
)

#B - Keep only rows where signing wallet matches Creator
is_creator = creator_data[creator_data['signing_wallet'] == creator_data['creator']].copy()

#C Counts transactions per base_coin where signing_wallet is the creator
creator_tx = is_creator.groupby('base_coin').size().reset_index(name='creator_transactions')

creator_tx


# Computes mean and max quote_coin_amount per base_coin; no creator info needed
mean_swap_size = data.groupby('base_coin')['quote_coin_amount'].mean().reset_index(name='mean_swap_size')
max_swap_size = data.groupby('base_coin')['quote_coin_amount'].max().reset_index(name='max_swap_size')

print(mean_swap_size)
print(max_swap_size)


# List of feature DataFrames (adjust names as per your variables)
feature_dfs = [
    total_sol_volume,           # Sum of quote_coin_amount per token
    total_transactions,         # Count of transactions per token
    unique_wallets,             # Distinct signing_wallet count
    buy_sell_ratio,             # Ratio of buy to sell transactions
    creator_tx,                 # Number of creator transactions
    mean_swap_size,             # Average quote_coin_amount per swap
    max_swap_size,              # Maximum quote_coin_amount per swap
   # top3_share[['mint', 'top3_volume_share']],  # Top 3 wallet volume share
   # time_to_first[['mint', 'time_to_first_swap']],  # Time to first swap
   # perc_first_10[['mint', 'perc_volume_first_10']],  # % volume in first 10 blocks
   # onchain_info[['mint', 'num_tokens_minted_past_30']]  # Tokens minted by creator
]

# Merge all features on 'mint', starting with the first DataFrame
features = feature_dfs[0]
for df in feature_dfs[1:]:
    features = features.merge(df, on='base_coin', how='left')

print(features)


train_features = train.merge(
    features,
    left_on='mint',      # train’s key
    right_on='base_coin',# features’ key
    how='left'
) #.drop(columns=['mint'])  # drop the extra column if you like

train_features.info()
train_features.head()


train_features = train_features.fillna(0)
train_features.isna().sum().sort_values(ascending=False).head(10)
train_features['has_graduated'] = train_features['has_graduated'].astype(int)

train_features


# Separate features and target
X = train_features.drop(columns=['mint', 'base_coin', 'has_graduated', 'slot_min', 'slot_graduated'])
y = train_features['has_graduated']

X
y


from sklearn.model_selection import train_test_split

# 80/20 split, stratified on the binary target to keep class balance
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape, "Validation shape:", X_val.shape)


from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both sets
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val), columns=X_val.columns, index=X_val.index
)

X_train_scaled.mean()  # Should be close to 0 for each feature
X_train_scaled.std()   # Should be close to 1 for each feature


from sklearn.linear_model import LogisticRegression

# Initialize the model with balanced class weights
model = LogisticRegression(class_weight='balanced', random_state=42)

# Train the model on the scaled training data (X_train_scaled, y_train)
model.fit(X_train_scaled, y_train)


# Distribution of graduation for train data
print(y_train.value_counts(normalize=True))


from sklearn.metrics import log_loss

# Predict probabilities on the validation set
y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]

# Calculate log loss
logloss = log_loss(y_val, y_pred_proba)
print("Validation log loss:", logloss)


test_features = test.merge(
    features
    , left_on = 'mint'
    , right_on = 'base_coin'
    , how = 'left'
)

test_features.head()


test_features = test_features.fillna(0)

test_features


test_features['is_valid'] = test_features['is_valid'].astype(int)

test_features


X_test = test_features.drop( columns = ['mint', 'base_coin', 'slot_min'])

X_test



# Initialize the Scalar
scaler = StandardScaler()

X_test_scaled = pd.DataFrame(
    scaler.fit_transform(X_test), columns = X_test.columns, index = X_test.index
)


# Predict probabilities for the positive class
test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

test_pred_proba


# Create submission DataFrame
submission = pd.DataFrame({'mint': test['mint'], 'has_graduated': test_pred_proba})

submission

submission.to_csv('submission.csv', index=False)




