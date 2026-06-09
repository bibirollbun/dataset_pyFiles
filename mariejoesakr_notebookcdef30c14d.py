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


import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for clean output

# Show files in the folder
for dirname, _, filenames in os.walk('/kaggle/input/competition-april'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Set correct folder
folder = '/kaggle/input/competition-april'

# Load data
train = pd.read_csv(f'{folder}/train.csv')
test = pd.read_csv(f'{folder}/test.csv')
submission_format = pd.read_csv(f'{folder}/sample_submission.csv')

# Preview first 5 rows
train.head()



# ğŸ�¯ Feature Engineering
train['Length_Ads'] = train['Episode_Length_minutes'] * train['Number_of_Ads']
test['Length_Ads'] = test['Episode_Length_minutes'] * test['Number_of_Ads']

train['Avg_Popularity'] = (train['Host_Popularity_percentage'] + train['Guest_Popularity_percentage']) / 2
test['Avg_Popularity'] = (test['Host_Popularity_percentage'] + test['Guest_Popularity_percentage']) / 2

# âœ… Keep all columns (no dropping Podcast_Name or Episode_Title)
train_cleaned = train.drop(columns=['Listening_Time_minutes'])
test_cleaned = test.copy()

# ğŸ§  Combine for consistent encoding
full_data = pd.concat([train_cleaned, test_cleaned], axis=0)

# One-hot encode categorical features
full_encoded = pd.get_dummies(full_data)

# ğŸ”„ Split back into train/test
X = full_encoded.iloc[:len(train), :]
X_test = full_encoded.iloc[len(train):, :]

# ğŸ�¯ Target log-transform
y = np.log1p(train['Listening_Time_minutes'])

# ğŸ”� Preview
X.head()





# ğŸ”� Preview and confirm dataset structure
print("Train feature preview:")
display(X.head())

print("Shapes (X, X_test, y):")
X.shape, X_test.shape, y.shape



import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Datasets
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val)

# Define parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.02,
    'max_depth': 8,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'verbosity': -1,
    'random_state': 42
}

# Train using callbacks for early stopping
model = lgb.train(
    params,
    dtrain,
    num_boost_round=3000,
    valid_sets=[dtrain, dval],
    valid_names=['train', 'val'],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
)




# Predict on the test set using the best iteration
predictions = model.predict(X_test, num_iteration=model.best_iteration)

# Reverse the log transformation
predictions = np.expm1(predictions)

# Create the submission file
submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': predictions
})

# Save to CSV
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("ğŸš€ Submission file created: submission.csv")




submission.head()


