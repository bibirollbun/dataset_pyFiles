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


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import os


print("--- Loading Data and Preparing Full Feature Set ---")
# This section re-creates our most complex feature set
COMP_DIR = '/kaggle/input/playground-series-s5e7/'
EXTERNAL_DIR = '/kaggle/input/extrovert-vs-introvert-behavior-data-backup/'
train_df = pd.read_csv(os.path.join(COMP_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(COMP_DIR, 'test.csv'))
datasert_df = pd.read_csv(os.path.join(EXTERNAL_DIR, 'personality_dataset.csv'))
datasert_df_prep = datasert_df.rename(columns={'Personality': 'match_p'}).drop_duplicates()
merge_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance','Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']
train_df = train_df.merge(datasert_df_prep, how='left', on=merge_cols)
test_df = test_df.merge(datasert_df_prep, how='left', on=merge_cols)
y_train_series = train_df['Personality']
all_data = pd.concat([train_df.drop(['id', 'Personality'], axis=1), test_df.drop('id', axis=1)], ignore_index=True)
def fill_missing_by_quantile_group(df, group_source_col, target_col):
    temp_bin_col = f'{group_source_col}_bin'; df[temp_bin_col] = pd.qcut(df[group_source_col], q=[0, 0.25, 0.5, 0.75, 1.0], labels=[1,2,3,4], duplicates='drop'); df[target_col] = df[target_col].fillna(df.groupby(temp_bin_col)[target_col].transform('median')); df.drop(columns=[temp_bin_col], inplace=True); return df
all_data = fill_missing_by_quantile_group(all_data, 'Social_event_attendance', 'Time_spent_Alone'); all_data['Time_spent_Alone'].fillna(all_data['Time_spent_Alone'].median(), inplace=True)
all_data = fill_missing_by_quantile_group(all_data, 'Going_outside', 'Social_event_attendance'); all_data['Social_event_attendance'].fillna(all_data['Social_event_attendance'].median(), inplace=True)
all_data = fill_missing_by_quantile_group(all_data, 'Social_event_attendance', 'Going_outside'); all_data['Going_outside'].fillna(all_data['Going_outside'].median(), inplace=True)
all_data = fill_missing_by_quantile_group(all_data, 'Post_frequency', 'Friends_circle_size'); all_data['Friends_circle_size'].fillna(all_data['Friends_circle_size'].median(), inplace=True)
all_data = fill_missing_by_quantile_group(all_data, 'Friends_circle_size', 'Post_frequency'); all_data['Post_frequency'].fillna(all_data['Post_frequency'].median(), inplace=True)
all_data.fillna({'Stage_fear': 'Unknown', 'Drained_after_socializing': 'Unknown', 'match_p': 'Unknown'}, inplace=True)
all_data = pd.get_dummies(all_data, columns=['Stage_fear', 'Drained_after_socializing', 'match_p'], prefix=['Stage', 'Drained', 'match'])
X = all_data[:len(train_df)]; X_test = all_data[len(train_df):]
y_encoded = LabelEncoder().fit_transform(y_train_series)


print("\n--- Starting Recursive Feature Elimination ---")
# We use a fast and powerful model like LightGBM as the estimator for RFECV
estimator = LGBMClassifier(random_state=42)
# The selector will automatically find the best number of features
selector = RFECV(
    estimator, 
    step=1, 
    cv=StratifiedKFold(3), # Use 3 folds for speed
    scoring='accuracy',
    n_jobs=-1
)

# Fit the selector on the data
selector.fit(X, y_encoded)

# Get the list of the best features
best_features = X.columns[selector.support_].tolist()

print(f"\n--- Feature Selection Complete ---")
print(f"Optimal number of features found: {selector.n_features_}")
print("Best features:", best_features)

# Save the list of best features to a file
pd.Series(best_features).to_csv('best_features.csv', index=False)
print("\n✅ List of best features saved to 'best_features.csv'")

