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


# preprocessing & model selection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OrdinalEncoder, OneHotEncoder, KBinsDiscretizer, FunctionTransformer
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb


import warnings
warnings.simplefilter('ignore')


from zipfile import ZipFile
zipObj = ZipFile('podcast_eda_viz.zip', 'w')

for filename in os.listdir("/kaggle/working"):
    if filename.endswith(".png"):
        zipObj.write(filename)
zipObj.close()


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')
original = pd.read_csv('/kaggle/input/original-dataset/podcast_dataset.csv')


# Concatenate original data with synthetics ones
train = pd.concat([train, original], axis=0, ignore_index=True)
train.drop_duplicates(inplace=True)


# Remove missing target values
train = train.dropna(subset=['Listening_Time_minutes'])

# Impute Episode_Length_minutes with Genre median
train['Episode_Length_minutes'] = train.groupby('Genre')['Episode_Length_minutes'].transform(lambda x: x.fillna(x.median()))
test['Episode_Length_minutes'] = test.groupby('Genre')['Episode_Length_minutes'].transform(lambda x: x.fillna(x.median()))

# Handle Guest Popularity
train['Guest_Present'] = train['Guest_Popularity_percentage'].notna().astype(int)
test['Guest_Present'] = test['Guest_Popularity_percentage'].notna().astype(int)
train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage'].fillna(0)
test['Guest_Popularity_percentage'] = test['Guest_Popularity_percentage'].fillna(0)

# Impute single missing ad value with mode
ads_mode = train['Number_of_Ads'].mode()[0]
train['Number_of_Ads'] = train['Number_of_Ads'].fillna(ads_mode)
test['Number_of_Ads'] = test['Number_of_Ads'].fillna(ads_mode)


# Frequency Encoding for Podcast_Name
train['Podcast_Name_Freq'] = train['Podcast_Name'].map(train['Podcast_Name'].value_counts(normalize=True))
test['Podcast_Name_Freq'] = test['Podcast_Name'].map(test['Podcast_Name'].value_counts(normalize=True))

# Extract Episode Number from Episode_Title
train['Episode_Number'] = (
    train['Episode_Title']
    .str.extract('(\d+)')
    .astype(float)
    .fillna(train['Episode_Title'].str.extract('(\d+)').astype(float).median())  # Compute median from extracted values
)

test['Episode_Number'] = (
    test['Episode_Title']
    .str.extract('(\d+)')
    .astype(float)
    .fillna(test['Episode_Title'].str.extract('(\d+)').astype(float).median())
)

# Ordinal Encoding for Low-Cardinality Features
ordinal_mappings = {
    'Publication_Day': {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7},
    'Publication_Time': {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4},
    'Episode_Sentiment': {'Negative': 0, 'Neutral': 1, 'Positive': 2}
}

for col, mapping in ordinal_mappings.items():
    train[col] = train[col].map(mapping)
    test[col] = test[col].map(mapping)

# Convert Genre to category dtype (for interaction features)
train['Genre'] = train['Genre'].astype('category')
test['Genre'] = test['Genre'].astype('category')

# Drop high-cardinality raw columns
train.drop(['Podcast_Name', 'Episode_Title'], axis=1, inplace=True)
test.drop(['Podcast_Name', 'Episode_Title'], axis=1, inplace=True)


from itertools import combinations
from tqdm import tqdm  # For progress bars
import gc

# Prioritize features with high correlation/domain relevance
columns_to_combine = [
    'Genre',
    'Publication_Time',
    'Episode_Length_minutes',
    'Host_Popularity_percentage',
    'Number_of_Ads',
    'Publication_Day'
]

# Generate 2-feature interactions (safe for memory)
pair_size = [2]
batch_size = 20

for r in pair_size:
    combo_list = list(combinations(columns_to_combine, r))
    total_batches = (len(combo_list) + batch_size - 1) // batch_size  # Calculate total batches
    
    # Initialize progress bar for batches
    with tqdm(total=len(combo_list), desc=f"Processing {r}-way interactions") as pbar:
        for i in range(0, len(combo_list), batch_size):
            batch = combo_list[i:i+batch_size]
            
            # Process entire batch at once (vectorized)
            for cols in batch:
                new_col = '_'.join(cols)
                
                # Faster string concatenation using vectorized operations
                train[new_col] = (
                    train[cols[0]].astype(str) 
                    + '_' 
                    + train[cols[1]].astype(str)
                ).astype('category')
                
                test[new_col] = (
                    test[cols[0]].astype(str)
                    + '_'
                    + test[cols[1]].astype(str)
                ).astype('category')
                
                pbar.update(1)  # Update progress after each combination
            
            # Memory management
            gc.collect()
            print(f"\nBatch {i//batch_size + 1}/{total_batches} completed")
            print(f"Current memory: {train.memory_usage(deep=True).sum()/1024**2:.1f} MB")

    print(f"\n✅ Completed all {r}-way interactions")


# ====================================================
# STEP 1: Prepare Features & Target
# ====================================================
# Define features (X) and target (y)
X = train.drop(columns=['Listening_Time_minutes'])
y = train['Listening_Time_minutes']

# Align test data columns with training data
test = test[X.columns]

# ====================================================
# STEP 2: LightGBM Model Setup
# ====================================================
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np

# Initialize 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(test))  # Store predictions for test data

# ====================================================
# STEP 3: Training Loop with Cross-Validation
# ====================================================
for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
    print(f"\n▶ Fold {fold+1}/5")
    
    # Split data
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    # ----------------------------------------
    # Critical Step: Specify Categorical Features
    # LightGBM needs to know which columns are categorical
    # ----------------------------------------
    cat_features = [
        'Genre',
        'Publication_Day',
        'Publication_Time',
        'Episode_Sentiment'
    ]
    
    # Convert to LightGBM Dataset format
    train_set = lgb.Dataset(
        X_train, 
        label=y_train,
        categorical_feature=cat_features,
        free_raw_data=False
    )
    
    valid_set = lgb.Dataset(
        X_valid, 
        label=y_valid,
        categorical_feature=cat_features,
        free_raw_data=False
    )
    
    # ----------------------------------------
    # Model Parameters (Optimized for Interaction Features)
    # ----------------------------------------
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 512,  # Reduced from 1024 to prevent overfitting
        'max_depth': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'verbosity': -1,
        'seed': 42
    }
    
    # ----------------------------------------
    # Train Model with Early Stopping
    # ----------------------------------------
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[train_set, valid_set],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.log_evaluation(100),
            lgb.early_stopping(stopping_rounds=100)
        ]
    )
    
    # ----------------------------------------
    # Generate Test Predictions
    # ----------------------------------------
    test_preds += model.predict(test) / cv.n_splits  # Average across folds
    print(f"Fold {fold+1} completed ✅\n")


# Create submission DataFrame with id and predictions
submission = pd.DataFrame({
    'id': test.index,
    'Listening_Time_minutes': test_preds
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

