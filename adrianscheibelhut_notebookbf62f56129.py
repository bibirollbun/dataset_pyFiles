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
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("\n--- Starting Baseline SVM Model ---")

# --- 1. Load Data ---
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

# --- 2. Define Original Features and Target ---
# No feature engineering for this baseline
original_features = [
    'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
    'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
    'TrackDurationMs', 'Energy'
]
target = 'BeatsPerMinute'

# Prepare the data
X_train = train[original_features]
y_train = train[target]
X_test = test[original_features]

# --- 3. Create a Pipeline with Scaling and SVR ---
# This is the standard and correct way to use an SVM.
# The SVR's default 'rbf' kernel is a good starting point.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR())
])

# --- 4. Train the Model ---
print("Training the SVM model on the entire training dataset...")
# The pipeline handles scaling and training in one step
pipeline.fit(X_train, y_train)

# --- 5. Create Submission ---
print("Making predictions on the test set...")
# The pipeline handles scaling the test data before prediction
test_predictions = pipeline.predict(X_test)

# Build the submission file
submission = pd.DataFrame({'id': test['id'], 'BeatsPerMinute': test_predictions})

# Use a different name to avoid overwriting your Random Forest submission
submission.to_csv('submission.csv', index=False)

print("\nSaved submission.csv successfully!")
print("Submission file head:")
print(submission.head())

