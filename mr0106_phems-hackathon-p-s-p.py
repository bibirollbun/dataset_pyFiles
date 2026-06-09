# Import Libraries
import warnings
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from lightgbm import LGBMRegressor, LGBMClassifier, log_evaluation, early_stopping
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

seed_everything(seed=2025)


# Clone Yunbase repository
!git clone https://github.com/yunsuxiaozi/Yunbase.git

# Install dependencies
!pip install -r Yunbase/requirements.txt

# Copy baseline.py to the working directory
import shutil
shutil.copy('/kaggle/working/Yunbase/baseline.py', '/kaggle/working/baseline.py')

# Install the baseline module
!pip install -q --requirement /kaggle/working/Yunbase/requirements.txt --no-index --find-links file:.

# Import Yunbase after installation
from baseline import Yunbase


# Function to get the next day
def get_next_day(date_str):
    """Get the next day from the given date string."""
    if date_str == 'None':
        return 'None'
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    next_day = date_obj + timedelta(days=1)
    return next_day.strftime('%Y-%m-%d')

# Function to generate features
def get_feats(mode='train'):
    """Generate features for training or testing data."""
    path = f'/kaggle/input/phems-hackathon-early-sepsis-prediction/{mode}ing_data/'
    
    print("< feats >")
    feats = pd.read_csv(path + f"SepsisLabel_{mode}.csv").drop_duplicates()
    feats['measurement_datetime_day'] = feats['measurement_datetime'].fillna('None').apply(lambda x: x[:10])

    print("< drug >")
    drug = pd.read_csv(path + f"drugsexposure_{mode}.csv")
    drug['measurement_datetime_day'] = drug['drug_datetime_hourly'].fillna('None').apply(lambda x: x[:10])
    
    for col in ['drug_concept_id', 'route_concept_id']:
        drug[col] = drug[col].fillna('None').astype(str)
        group_df = drug.groupby(['person_id', 'measurement_datetime_day'])[col].apply(lambda x: " ".join(x)).reset_index()
        feats = feats.merge(group_df, on=['person_id', 'measurement_datetime_day'], how='left')
        feats[col] = feats[col].fillna('None')
    
    drug['measurement_datetime_day'] = drug['measurement_datetime_day'].apply(lambda x: get_next_day(x))
    
    for col in ['drug_concept_id', 'route_concept_id']:
        group_df = drug.groupby(['person_id', 'measurement_datetime_day'])[col].apply(lambda x: " ".join(x)).reset_index().rename(columns={col: col + "_previous_day"})
        feats = feats.merge(group_df, on=['person_id', 'measurement_datetime_day'], how='left')
        feats[col + "_previous_day"] = feats[col + "_previous_day"].fillna('None')
    
    feats['measurement_datetime'] = pd.to_datetime(feats['measurement_datetime'])
    feats['dow'] = feats['measurement_datetime'].dt.dayofweek
    feats['doy'] = feats['measurement_datetime'].dt.dayofyear
    feats['hour'] = feats['measurement_datetime'].dt.hour
    feats.drop(['measurement_datetime'], axis=1, inplace=True)

    print("-" * 30)
    return feats

# Generate features for train and test data
train = get_feats(mode='train')
test = get_feats(mode='test')

# Display shapes and first few rows of train data
print(train.shape, test.shape)
print(train.head())


print(train.columns)
print(test.columns)


def get_feats(mode='train'):
    """Generate features for training or testing data."""
    path = f'/kaggle/input/phems-hackathon-early-sepsis-prediction/{mode}ing_data/'
    
    print("< feats >")
    feats = pd.read_csv(path + f"SepsisLabel_{mode}.csv").drop_duplicates()
    feats['measurement_datetime_day'] = feats['measurement_datetime'].fillna('None').apply(lambda x: x[:10])

    print("< drug >")
    drug = pd.read_csv(path + f"drugsexposure_{mode}.csv")
    drug['measurement_datetime_day'] = drug['drug_datetime_hourly'].fillna('None').apply(lambda x: x[:10])
    
    # Fill missing values and convert to string
    for col in ['drug_concept_id', 'route_concept_id']:
        drug[col] = drug[col].fillna('None').astype(str)
    
    # Group by person_id and measurement_datetime_day
    for col in ['drug_concept_id', 'route_concept_id']:
        group_df = drug.groupby(['person_id', 'measurement_datetime_day'])[col].apply(lambda x: " ".join(x)).reset_index()
        feats = feats.merge(group_df, on=['person_id', 'measurement_datetime_day'], how='left')
        feats[col] = feats[col].fillna('None')
    
    # Get the next day for each measurement_datetime_day
    drug['measurement_datetime_day'] = drug['measurement_datetime_day'].apply(lambda x: get_next_day(x))
    
    # Group by person_id and next day's measurement_datetime_day
    for col in ['drug_concept_id', 'route_concept_id']:
        group_df = drug.groupby(['person_id', 'measurement_datetime_day'])[col].apply(lambda x: " ".join(x)).reset_index()
        group_df = group_df.rename(columns={col: col + "_previous_day"})
        feats = feats.merge(group_df, on=['person_id', 'measurement_datetime_day'], how='left')
        feats[col + "_previous_day"] = feats[col + "_previous_day"].fillna('None')
    
    # Extract datetime features
    feats['measurement_datetime'] = pd.to_datetime(feats['measurement_datetime'])
    feats['dow'] = feats['measurement_datetime'].dt.dayofweek
    feats['doy'] = feats['measurement_datetime'].dt.dayofyear
    feats['hour'] = feats['measurement_datetime'].dt.hour
    feats.drop(['measurement_datetime'], axis=1, inplace=True)

    print("-" * 30)
    return feats


# Import required libraries
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from baseline import Yunbase
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

seed_everything(seed=2025)

# Load data
train = pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/SepsisLabel_train.csv')
test = pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/SepsisLabel_test.csv')

# Check column names in train and test data
print("Train columns:", train.columns)
print("Test columns:", test.columns)

# Add 'measurement_datetime_day' column if it doesn't exist
if 'measurement_datetime_day' not in train.columns:
    train['measurement_datetime_day'] = train['measurement_datetime'].fillna('None').apply(lambda x: x[:10])
if 'measurement_datetime_day' not in test.columns:
    test['measurement_datetime_day'] = test['measurement_datetime'].fillna('None').apply(lambda x: x[:10])

# Create missing columns if they don't exist
if 'drug_concept_id' not in train.columns:
    train['drug_concept_id'] = 'None'
    train['drug_concept_id_previous_day'] = 'None'
if 'route_concept_id' not in train.columns:
    train['route_concept_id'] = 'None'
    train['route_concept_id_previous_day'] = 'None'

if 'drug_concept_id' not in test.columns:
    test['drug_concept_id'] = 'None'
    test['drug_concept_id_previous_day'] = 'None'
if 'route_concept_id' not in test.columns:
    test['route_concept_id'] = 'None'
    test['route_concept_id_previous_day'] = 'None'

# Define text columns for TF-IDF
text_cols = [
    'drug_concept_id', 'route_concept_id',
    'drug_concept_id_previous_day', 'route_concept_id_previous_day'
]

# Initialize TF-IDF models
word2vec_models = []
for t_col in text_cols:
    word2vec_models.append(
        (TfidfVectorizer(analyzer='word', max_features=50, ngram_range=(1, 1)), t_col, 'tfidf')
    )

# LightGBM parameters
lgb_params = {
    'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.7488447863498244,
    'importance_type': 'gain', 'learning_rate': 0.06893681524443476, 'max_depth': -1, 'min_child_samples': 15,
    'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 150, 'n_jobs': -1, 'num_leaves': 19,
    'objective': None, 'random_state': 2025, 'reg_alpha': 6.321244942789797, 'reg_lambda': 0.44170411438999,
    'subsample': 0.9748276894890832, 'subsample_for_bin': 200000, 'extra_trees': True, 'metric': 'auc',
    'verbose': -1
}

# Initialize Yunbase
yunbase = Yunbase(
    num_folds=5,
    n_repeats=2,
    objective='binary',
    models=[(LGBMClassifier(**lgb_params), 'lgb')],
    metric='pr_auc',
    num_classes=2,
    word2vec_models=word2vec_models,
    early_stop=200,
    group_col='person_id',
    target_col='SepsisLabel',
    use_high_corr_feat=True,
    use_eval_metric=False,
)

# Train the model
yunbase.fit(train)

# Make predictions on the test data
test_preds = yunbase.predict(test)

# Create 'person_id_datetime' column if it doesn't exist
if 'person_id_datetime' not in test.columns:
    test['person_id_datetime'] = test['person_id'].astype(str) + '_' + test['measurement_datetime_day'].astype(str)

# Create submission DataFrame
submission_df = pd.DataFrame({
    'person_id_datetime': test['person_id_datetime'],  # Use existing or created column
    'SepsisLabel': test_preds  # Predictions
})

# Check for duplicates in 'person_id_datetime'
if submission_df['person_id_datetime'].duplicated().any():
    print("Duplicate IDs found. Resolving duplicates...")
    # Add a unique counter to duplicates
    submission_df['person_id_datetime'] = submission_df.groupby('person_id_datetime').cumcount().astype(str) + '_' + submission_df['person_id_datetime']
    print("Duplicates resolved.")

# Save predictions to submission.csv
submission_df.to_csv("submission.csv", index=False)

print("Predictions saved to submission.csv")


# Final Notes
print("We can see that the CV fluctuates greatly and differs significantly from LB. I estimate that there will be a significant shakeup in this competition.")
print("Good luck to everyone.")

