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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

# 1. Load Data
train = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
test = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
train_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
test_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')

# 2. Merge Demographics
train = train.merge(train_demo, on='subject', how='left')
test = test.merge(test_demo, on='subject', how='left')

# 3. Aggregate Features for Each Sequence
# We'll aggregate per sequence_id (basic statistics over time)
sensor_cols = [col for col in train.columns if (
    col.startswith('acc_') or col.startswith('rot_') or col.startswith('thm_') or col.startswith('tof_'))]
agg_funcs = ['mean', 'std', 'min', 'max']
train_agg = train.groupby('sequence_id')[sensor_cols].agg(agg_funcs)
train_agg.columns = ['_'.join(col) for col in train_agg.columns]
train_agg.reset_index(inplace=True)
train_meta = train.groupby('sequence_id')[['gesture', 'sequence_type', 'subject', 'orientation']].first().reset_index()
train_agg = train_agg.merge(train_meta, on='sequence_id', how='left')

test_agg = test.groupby('sequence_id')[sensor_cols].agg(agg_funcs)
test_agg.columns = ['_'.join(col) for col in test_agg.columns]
test_agg.reset_index(inplace=True)
test_meta = test.groupby('sequence_id')[['subject']].first().reset_index()
test_agg = test_agg.merge(test_meta, on='sequence_id', how='left')

# Merge demographics for aggregated test/train
train_agg = train_agg.merge(train_demo, on='subject', how='left')
test_agg = test_agg.merge(test_demo, on='subject', how='left')

# 4. Encode Target and Features
gesture_le = LabelEncoder()
train_agg['gesture_enc'] = gesture_le.fit_transform(train_agg['gesture'])

# 5. Train Classifier
features = [col for col in train_agg.columns if col not in [
    'sequence_id', 'gesture', 'gesture_enc', 'sequence_type', 'subject', 'orientation']]

X_train = train_agg[features].fillna(-1)  # Fill missing with -1 (could be improved)
y_train = train_agg['gesture_enc']
X_test = test_agg[features].fillna(-1)

clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# 6. Predict
test_preds = clf.predict(X_test)
test_agg['gesture_pred'] = gesture_le.inverse_transform(test_preds)

# 7. Create Submission
submission = pd.DataFrame({
    'sequence_id': test_agg['sequence_id'],
    'gesture': test_agg['gesture_pred']
})
submission.to_csv('submission.csv', index=False)
print(submission.head())




