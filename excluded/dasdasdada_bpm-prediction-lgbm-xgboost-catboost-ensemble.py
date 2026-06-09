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


!pip install lightgbm xgboost catboost


import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the data
try:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
    sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
except FileNotFoundError:
    print("Running outside of Kaggle. Please adjust file paths.")
    # Add your local file paths here if needed, e.g.:
    # train_df = pd.read_csv("train.csv")
    # test_df = pd.read_csv("test.csv")
    # sample_submission_df = pd.read_csv("sample_submission.csv")
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


def feature_engineer(df):
    # Corrected to use column names from your image
    df['duration_energy_ratio'] = df['TrackDurationMs'] / (df['Energy'] + 1e-6)
    df['liveness_speechiness_ratio'] = df['LivePerformanceLikelihood'] / (df['VocalContent'] + 1e-6)
    df['acousticness_instrumentalness_sum'] = df['AcousticQuality'] + df['InstrumentalScore'] 
    return df

train_df = feature_engineer(train_df)
test_df = feature_engineer(test_df)

from sklearn.preprocessing import PolynomialFeatures

# 1. Select the best features to combine based on domain knowledge
# These features seem central to the audio's characteristics.
important_features = ['Energy', 'AudioLoudness', 'AcousticQuality', 'RhythmScore', 'VocalContent']

# 2. Create the feature interaction transformer
# We'll create 2nd-degree interactions (feature_A * feature_B)
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

# 3. Fit on the training data and transform both train and test data
# This learns the combinations from the training set
poly_train = poly.fit_transform(train_df[important_features])
poly_test = poly.transform(test_df[important_features])

# 4. Create new DataFrames with meaningful column names
poly_train_df = pd.DataFrame(poly_train, columns=poly.get_feature_names_out(important_features))
poly_test_df = pd.DataFrame(poly_test, columns=poly.get_feature_names_out(important_features))

# 5. Add the new features back to the original dataframes
# We reset the index to ensure a clean join
train_df = pd.concat([train_df.drop(columns=important_features).reset_index(drop=True), poly_train_df], axis=1)
test_df = pd.concat([test_df.drop(columns=important_features).reset_index(drop=True), poly_test_df], axis=1)

print(f"Added {poly_train_df.shape[1]} new polynomial features.")
print("New shape of training data:", train_df.shape)

# Define features (X) and target (y) 
features = [col for col in train_df.columns if col not in ['id', 'BeatsPerMinute']]
X_train = train_df[features]
y_train = train_df['BeatsPerMinute']
X_test = test_df[features]


# Initialize models
lgb_model = lgb.LGBMRegressor(random_state=42)
xgb_model = xgb.XGBRegressor(random_state=42)
cat_model = cb.CatBoostRegressor(random_state=42, verbose=0)

# Train the models
lgb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
cat_model.fit(X_train, y_train)

# Generate predictions
lgb_preds = lgb_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)
cat_preds = cat_model.predict(X_test)


# Create a weighted ensemble
# These weights can be tuned for better performance
weighted_preds = 0.4 * lgb_preds + 0.4 * xgb_preds + 0.2 * cat_preds

# Create the submission file
submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': weighted_preds})
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
print(submission_df.head())




