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


#  Imports
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ğŸ›  Config
pd.set_option('display.max_columns', 100)
sns.set(style="whitegrid")

#  File paths
BASE_DIR = '/kaggle/input/cmi-detect-behavior-with-sensor-data'

train = pd.read_csv(f'{BASE_DIR}/train.csv')
test = pd.read_csv(f'{BASE_DIR}/test.csv')
train_demo = pd.read_csv(f'{BASE_DIR}/train_demographics.csv')
test_demo = pd.read_csv(f'{BASE_DIR}/test_demographics.csv')

# ğŸ”� Preview
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Train Demographics:", train_demo.shape)
print("Test Demographics:", test_demo.shape)

display(train.head())
display(train_demo.head())




seqs = train['sequence_id'].unique()
print("Unique sequences:", len(seqs))

# Evisualize acc_x of one sequence
sample_seq = train[train['sequence_id'] == seqs[0]]
sample_seq[['acc_x', 'acc_y', 'acc_z']].plot(figsize=(12, 4), title='Acceleration over time')
plt.show()




def extract_features(df):
    feats = []
    for seq_id, seq_df in df.groupby('sequence_id'):
        feat = {'sequence_id': seq_id, 'subject': seq_df['subject'].iloc[0]}
        
        # IMU features
        for col in ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']:
            feat[f'{col}_mean'] = seq_df[col].mean()
            feat[f'{col}_std'] = seq_df[col].std()
            feat[f'{col}_min'] = seq_df[col].min()
            feat[f'{col}_max'] = seq_df[col].max()
        
        feats.append(feat)
    
    return pd.DataFrame(feats)

train_feats = extract_features(train)
test_feats = extract_features(test)

# Merge with demographics
train_feats = train_feats.merge(train_demo, on='subject', how='left')
test_feats = test_feats.merge(test_demo, on='subject', how='left')

# Merge label
target_df = train[['sequence_id', 'gesture']].drop_duplicates()
train_feats = train_feats.merge(target_df, on='sequence_id', how='left')

display(train_feats.head())



from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier

# Preprocessing
X = train_feats.drop(columns=['sequence_id', 'subject', 'gesture'])
y = train_feats['gesture']

# train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

#  Train model
clf = LGBMClassifier()
clf.fit(X_train, y_train)

#  Evaluate
y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))




X_test = test_feats.drop(columns=['sequence_id', 'subject'])


X_test = X_test[X_train.columns]

# Predict
test_feats['gesture'] = clf.predict(X_test)

# Only keep required columns
submission = test_feats[['sequence_id', 'gesture']]

# Save as parquet file for Kaggle submission
submission.to_parquet("submission.parquet", index=False)


