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


# %% [code]
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# %% [code]
# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

# Identify target column
target_col = 'Listening_Time_minutes'
print(f"Target variable: {target_col}")

# Drop id column
train = train.drop('id', axis=1)
test_ids = test['id']
test = test.drop('id', axis=1)

# %% [code]
# Feature engineering
def create_features(df):
    df = df.copy()
    
    # Convert categorical features
    categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Convert Episode_Sentiment to numerical
    if 'Episode_Sentiment' in df.columns:
        sentiment_mapping = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
        df['Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_mapping)
        df['Episode_Sentiment'] = df['Episode_Sentiment'].fillna(0)  # Handle any missing
    
    # Handle Publication_Time
    if 'Publication_Time' in df.columns:
        try:
            # Extract hour and minute
            df['Publication_Hour'] = pd.to_datetime(df['Publication_Time']).dt.hour
            df['Publication_Minute'] = pd.to_datetime(df['Publication_Time']).dt.minute
        except:
            # Default values if parsing fails
            df['Publication_Hour'] = 12
            df['Publication_Minute'] = 0
    
    # Create interaction features
    df['Host_Guest_Interaction'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']
    df['Ads_Per_Minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1)  # +1 to avoid division by zero
    
    return df

train = create_features(train)
test = create_features(test)

# Drop original time column if it exists
if 'Publication_Time' in train.columns:
    train = train.drop('Publication_Time', axis=1)
if 'Publication_Time' in test.columns:
    test = test.drop('Publication_Time', axis=1)

# %% [code]
# Define numerical columns for scaling (only truly numerical features)
numerical_cols = [
    'Episode_Length_minutes',
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage',
    'Number_of_Ads',
    'Publication_Hour',
    'Publication_Minute',
    'Host_Guest_Interaction',
    'Ads_Per_Minute'
]

# Verify all numerical columns exist and are numeric
for col in numerical_cols:
    if col not in train.columns:
        numerical_cols.remove(col)
    elif not pd.api.types.is_numeric_dtype(train[col]):
        numerical_cols.remove(col)

print("Columns to scale:", numerical_cols)

# Scale numerical features
scaler = RobustScaler()
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])

# %% [code]
# Prepare data
X = train.drop(target_col, axis=1)
y = train[target_col]
X_test = test.copy()

# %% [code]
# Model parameters
xgb_params = {
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'gamma': 0.1,
    'min_child_weight': 10,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'objective': 'reg:squarederror',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'random_state': 42,
    'eval_metric': 'rmse'
}

# %% [code]
# Cross-validation
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
rmse_scores = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\nFold {fold + 1}")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    model = XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=100,
        verbose=100
    )
    
    valid_preds = model.predict(X_valid)
    test_preds += model.predict(X_test) / N_SPLITS
    
    fold_rmse = np.sqrt(mean_squared_error(y_valid, valid_preds))
    rmse_scores.append(fold_rmse)
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.5f}")

print(f"\nOverall RMSE: {np.mean(rmse_scores):.5f}")

# %% [code]
# Create submission
submission['Listening_Time_minutes'] = test_preds
submission.to_csv('submission.csv', index=False)
print("\nSubmission head:")
submission.head()

