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
from autogluon.tabular import TabularDataset, TabularPredictor

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

# Feature engineering
def create_features(df):
    # Numerical interactions
    df['balance_to_duration'] = df['balance'] / (df['duration'] + 1)
    df['age_times_balance'] = df['age'] * df['balance']
    df['campaign_per_contact'] = df['campaign'] / (df['previous'] + 1)
    
    # Binning
    df['age_bin'] = pd.cut(df['age'], bins=5, labels=False)
    df['balance_bin'] = pd.qcut(df['balance'], q=5, labels=False)
    return df

train = create_features(train)
test = create_features(test)

# Prepare data
label = 'y'
test_ids = test['id']
train_data = TabularDataset(train.drop(columns=['id']))
test_data = TabularDataset(test.drop(columns=['id']))

# Train AutoGluon with compatible settings
predictor = TabularPredictor(
    label=label,
    eval_metric='roc_auc',
    problem_type='binary',
).fit(
    train_data=train_data,
    presets='good_quality',  # More stable than best_quality with v0.8.1
    time_limit=3600*2,  # 2 hours training time
    verbosity=3,
    hyperparameters={
        'GBM': [
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
            {},
        ],
        'CAT': {},
        'XGB': {},
    }
)

# Evaluate models
performance = predictor.leaderboard(train_data, silent=False)
print(performance)

# Generate predictions
probs = predictor.predict_proba(test_data)
submission = pd.DataFrame({'id': test_ids, 'y': probs[1]})

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission saved successfully!")
print(submission.head())


!pip install autogluon




