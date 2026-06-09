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


import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


train_df.head()


print(train_df.dtypes)
print(train_df.isnull().sum())



# Encode categorical features
label_encoders = {}
for col in ['Soil Type', 'Crop Type', 'Fertilizer Name']:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    label_encoders[col] = le


train_df


# Prepare test set encoders
## We can't make a pipeline because we have already encoded the label too

for col in ['Soil Type', 'Crop Type']:
    test_df[col] = label_encoders[col].transform(test_df[col])


test_df


# Define features and target
X = train_df.drop(['id', 'Fertilizer Name'], axis=1)
y = train_df['Fertilizer Name']
X_test = test_df.drop(['id'], axis=1)
test_ids = test_df['id']


# Train a classifier (Random Forest)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)


# Predict probabilities on test set
probs = clf.predict_proba(X_test)


probs


# Get top 3 predicted class indices for each observation
top3_preds = np.argsort(probs, axis=1)[:, -3:][:, ::-1]


# Convert class indices back to original fertilizer names
fertilizer_names = label_encoders['Fertilizer Name'].inverse_transform(np.arange(len(label_encoders['Fertilizer Name'].classes_)))
predicted_labels = [
    " ".join([fertilizer_names[i] for i in row])
    for row in top3_preds
]


# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': predicted_labels
})


# Save to CSV
submission.to_csv("submission.csv", index=False)


# Convert to DMatrix (optional, but improves speed)
dtrain = xgb.DMatrix(X, label=y)
dtest = xgb.DMatrix(X_test)


# Define parameters for multi-class classification
num_classes = len(np.unique(y))
params = {
    'objective': 'multi:softprob',
    'num_class': num_classes,
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'eta': 0.1,
    'seed': 42
}


# Train model
xgb_model = xgb.train(params, dtrain, num_boost_round=100)


# Predict probabilities for test set
pred_probs = xgb_model.predict(dtest)


# Get top 3 class indices per row
top3_preds = np.argsort(pred_probs, axis=1)[:, -3:][:, ::-1]


# Decode predicted indices to actual fertilizer names
fertilizer_names = label_encoders['Fertilizer Name'].inverse_transform(np.arange(num_classes))
predicted_labels = [
    " ".join([fertilizer_names[i] for i in row])
    for row in top3_preds
]


# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': predicted_labels
})


# Save to CSV
submission.to_csv('xgboost_submission.csv', index=False)

