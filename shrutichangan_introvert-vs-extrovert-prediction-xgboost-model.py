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


# ðŸ“Œ Step 1: Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# ðŸ“Œ Step 2: Load Data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

# ðŸ“Œ Step 3: Handle Missing Values (Numerical)
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
            'Friends_circle_size', 'Post_frequency']
for col in num_cols:
    median_val = train_df[col].median()
    train_df[col].fillna(median_val, inplace=True)
    test_df[col].fillna(median_val, inplace=True)

# ðŸ“Œ Step 4: Handle Missing Values (Categorical)
train_df['Stage_fear'].fillna(train_df['Stage_fear'].mode()[0], inplace=True)
test_df['Stage_fear'].fillna(train_df['Stage_fear'].mode()[0], inplace=True)
train_df['Drained_after_socializing'].fillna(train_df['Drained_after_socializing'].mode()[0], inplace=True)
test_df['Drained_after_socializing'].fillna(train_df['Drained_after_socializing'].mode()[0], inplace=True)

# ðŸ“Œ Step 5: Encode Categorical Columns
le_stage = LabelEncoder()
le_stage.fit(train_df['Stage_fear'])
test_df['Stage_fear'] = test_df['Stage_fear'].apply(lambda x: x if x in le_stage.classes_ else 'Unknown')
le_stage_classes = list(le_stage.classes_)
if 'Unknown' not in le_stage_classes:
    le_stage_classes.append('Unknown')
le_stage.classes_ = np.array(le_stage_classes)
train_df['Stage_fear'] = le_stage.transform(train_df['Stage_fear'])
test_df['Stage_fear'] = le_stage.transform(test_df['Stage_fear'])

le_drain = LabelEncoder()
le_drain.fit(train_df['Drained_after_socializing'])
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].apply(lambda x: x if x in le_drain.classes_ else 'Unknown')
le_drain_classes = list(le_drain.classes_)
if 'Unknown' not in le_drain_classes:
    le_drain_classes.append('Unknown')
le_drain.classes_ = np.array(le_drain_classes)
train_df['Drained_after_socializing'] = le_drain.transform(train_df['Drained_after_socializing'])
test_df['Drained_after_socializing'] = le_drain.transform(test_df['Drained_after_socializing'])

le_target = LabelEncoder()
train_df['Personality'] = le_target.fit_transform(train_df['Personality'])

# ðŸ“Œ Step 6: Train-Validation Split
X = train_df.drop(['id', 'Personality'], axis=1)
y = train_df['Personality']
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ðŸ“Œ Step 7: Train XGBoost with Better Parameters
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)

# ðŸ“Œ Step 8: Validation Report
y_val_pred = model.predict(X_val)
print(classification_report(y_val, y_val_pred))

# ðŸ“Œ Step 9: Make Predictions on Test Set
X_test = test_df.drop(['id'], axis=1)
test_pred = model.predict(X_test)
test_pred_labels = le_target.inverse_transform(test_pred)

# ðŸ“Œ Step 10: Submission File
submission = sample_submission.copy()
submission['Personality'] = test_pred_labels
submission.to_csv("submission.csv", index=False)
submission.head()


