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


!pip install xgboost


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
import numpy as np




# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')




# Target encoding
target_le = LabelEncoder()
train['Fertilizer Name'] = target_le.fit_transform(train['Fertilizer Name'])



# Drop IDs
X = train.drop(columns=['id', 'Fertilizer Name'])
y = train['Fertilizer Name']
X_test = test.drop(columns=['id'])


# Feature engineering
X['NPK_ratio'] = (X['Nitrogen'] + 1) / (X['Phosphorous'] + X['Potassium'] + 1)
X_test['NPK_ratio'] = (X_test['Nitrogen'] + 1) / (X_test['Phosphorous'] + X_test['Potassium'] + 1)


# Columns to one-hot encode
categorical_features = ['Soil Type', 'Crop Type']
numeric_features = [col for col in X.columns if col not in categorical_features]


# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ])



# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
])


# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"New XGBoost Accuracy with One-Hot + Feature Engineering: {scores.mean():.4f}")











# Fit and Predict
model.fit(X, y)
preds = model.predict(X_test)
preds_decoded = target_le.inverse_transform(preds)

# Submission
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': preds_decoded
})
submission.to_csv("submission.csv", index=False)
print("ğŸ“� New submission.csv created with improved pipeline.")















from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np



# Label encode target
from sklearn.preprocessing import LabelEncoder
target_le = LabelEncoder()
train['Fertilizer Name'] = target_le.fit_transform(train['Fertilizer Name'])

# Feature engineering
X = train.drop(columns=['id', 'Fertilizer Name'])
y = train['Fertilizer Name']
X_test = test.drop(columns=['id'])

X['NPK_ratio'] = (X['Nitrogen'] + 1) / (X['Phosphorous'] + X['Potassium'] + 1)
X_test['NPK_ratio'] = (X_test['Nitrogen'] + 1) / (X_test['Phosphorous'] + X_test['Potassium'] + 1)

# Columns
categorical_features = ['Soil Type', 'Crop Type']
numeric_features = [col for col in X.columns if col not in categorical_features]

# Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', 'passthrough', numeric_features)
])

# XGBoost base model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', xgb)
])

# Hyperparameter grid
param_dist = {
    'xgb__n_estimators': [100, 200, 300],
    'xgb__max_depth': [3, 4, 5, 6, 7],
    'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'xgb__subsample': [0.6, 0.8, 1.0],
    'xgb__colsample_bytree': [0.6, 0.8, 1.0]
}

# RandomizedSearchCV
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=25,            # you can increase for better tuning
    cv=3,
    scoring='accuracy',
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the search
search.fit(X_train, y_train)

# Evaluate
best_model = search.best_estimator_
val_preds = best_model.predict(X_val)
print(f"\nğŸ”� Tuned XGBoost Validation Accuracy: {accuracy_score(y_val, val_preds):.4f}")
print(f"Best Params: {search.best_params_}")

# Final train on full data and predict on test set
best_model.fit(X, y)
final_preds = best_model.predict(X_test)
decoded_preds = target_le.inverse_transform(final_preds)

# Submission
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': decoded_preds
})
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv generated with tuned XGBoost model.")





