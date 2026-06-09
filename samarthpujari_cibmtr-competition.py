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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer


# Load data
train_path = '/kaggle/input/equity-post-HCT-survival-predictions/train.csv'
test_path = '/kaggle/input/equity-post-HCT-survival-predictions/test.csv'
data_dict_path = '/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv'
sample_submission_path = '/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
data_dict = pd.read_csv(data_dict_path)
sample_submission = pd.read_csv(sample_submission_path)


# Display dataset overview
print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
data_dict.head()


# Extract features and target
target = 'efs_time'  # Updated target column
censor_col = 'efs'  # Updated censor column

X = train.drop(columns=[target, censor_col, 'ID'])
y = train[target]
event = train[censor_col]


# Ensure race_group column exists
if 'race_group' not in train.columns:
    raise KeyError("Column 'race_group' not found in the training data")

race_group = train['race_group']


# Handle categorical and numeric features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# Define a stratified concordance index scorer
def stratified_c_index(y_true, y_pred, groups):
    group_indices = np.unique(groups)
    c_indices = []
    for group in group_indices:
        mask = (groups == group).values if isinstance(groups, pd.Series) else (groups == group)
        if mask.sum() > 1:
            c_indices.append(concordance_index(np.array(y_true)[mask], np.array(y_pred)[mask]))
    return np.mean(c_indices) - np.std(c_indices)



def concordance_index(y_true, y_pred):
    pairs = 0
    concordant = 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] != y_true[j]:
                pairs += 1
                if (y_pred[i] - y_pred[j]) * (y_true[i] - y_true[j]) > 0:
                    concordant += 1
    return concordant / pairs if pairs > 0 else 0.5

scorer = make_scorer(stratified_c_index, greater_is_better=True, groups=race_group)



# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])


# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in cv.split(X, event):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    race_train, race_val = race_group.iloc[train_idx], race_group.iloc[val_idx]
    
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    score = stratified_c_index(y_val, preds, race_val)
    scores.append(score)

print("CV Stratified C-Index:", np.mean(scores))



# Train on full data and predict
test_features = test.drop(columns=['ID'])
model.fit(X, y)
test_predictions = model.predict(test_features)



# Submission
submission = pd.DataFrame({'ID': test['ID'], 'prediction': test_predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")


