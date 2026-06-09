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
import json

from itertools import chain
from collections import Counter

from datetime import datetime

from catboost import CatBoostRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


train_path = "/kaggle/input/sparta-2024-data-science-competition/train.csv"
train_df = pd.read_csv(train_path)

test_path = "/kaggle/input/sparta-2024-data-science-competition/test.csv"
test_df = pd.read_csv(test_path)


train_df.dropna(subset=['beds', 'bedrooms', 'bathrooms', 'bathrooms_text', 'host_identity_verified', 'host_since', 'host_is_superhost'], inplace=True)


all_amenities = list(chain.from_iterable(train_df['amenities']))
print(f"Total unique amenities: {len(set(all_amenities))}")


train_df['host_since'] = pd.to_datetime(train_df['host_since'], errors='coerce')
train_df['first_review'] = pd.to_datetime(train_df['first_review'], errors='coerce')

today = pd.Timestamp.today()

train_df['host_duration_days'] = (today - train_df['host_since']).dt.days
train_df['host_duration_years'] = train_df['host_duration_days'] / 365

train_df['first_review_days'] = (today - train_df['first_review']).dt.days
train_df['first_review_years'] = train_df['first_review_days'] / 365


train_df['host_is_superhost'] = train_df['host_is_superhost'].map({'t': 1, 'f': 0})
train_df['host_identity_verified'] = train_df['host_identity_verified'].map({'t': 1, 'f': 0})


train_df['num_amenities'] = train_df['amenities'].apply(len)

review_scores = [
    'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
    'review_scores_communication', 'review_scores_location', 'review_scores_value'
]

train_df['review_scores_avg'] = train_df[review_scores].mean(axis=1)


train_df['bathrooms_type'] = train_df['bathrooms_text'].str.extract(r'[\d\.]+\s+(.*)')

numerical_cols = [
    'beds', 'bedrooms', 'bathrooms', 'accommodates', 'availability_30', 'host_is_superhost', 'host_identity_verified',
    'number_of_reviews', 'host_duration_years', 'review_scores_rating', 'review_scores_avg', 'first_review_years', 'num_amenities'
]

categorical_cols = ['room_type', 'bathrooms_type', 'property_type', 'city']

for col in categorical_cols:
    train_df[col] = train_df[col].astype('category')


def safe_json(x):
    try:
        return json.loads(x) if isinstance(x, str) else []
    except:
        return []

train_df['amenities'] = train_df['amenities'].apply(safe_json)

all_amenities = list(chain.from_iterable(train_df['amenities']))
top_amenities = set([a for a, _ in Counter(all_amenities).most_common(100)])

amenity_features = {
    f"amenity_{a}": train_df['amenities'].apply(lambda x: a in x)
    for a in top_amenities
}

amenities_df = pd.DataFrame(amenity_features, index=train_df.index)
train_df = pd.concat([train_df, amenities_df], axis=1)

amenity_cols = list(amenities_df.columns)


# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
    ],
    remainder='passthrough'
)

X = train_df[numerical_cols + categorical_cols + amenity_cols].copy()
y = train_df['price'].copy()

for col in ['room_type', 'bathrooms_type', 'property_type', 'city']:
    X[col] = X[col].astype('category')


'''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for col in ['room_type', 'bathrooms_type', 'property_type', 'city']:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')
    
# XGBoost
# Test using split dataset
xgb = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    enable_categorical=True
)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# RMSE
rmse_xgb = mean_squared_error(y_test, y_pred_xgb, squared=False)

print("XGBoost RMSE:", rmse_xgb)'''


'''X_train[categorical_cols] = X_train[categorical_cols].astype(str).fillna("missing")
X_test[categorical_cols] = X_test[categorical_cols].astype(str).fillna("missing")

cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    cat_features=categorical_cols,
    verbose=100
)

cat_model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50,
    cat_features=categorical_cols,
    verbose=100
)
cat_preds = cat_model.predict(X_test)

rmse_cat = mean_squared_error(y_test, cat_preds, squared=False)
print("CatBoost RMSE:", rmse_cat)'''


'''cat_model.get_feature_importance(prettified=True)'''


'''# Train final model
xgb_final = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    enable_categorical=True
)

xgb_final.fit(X, y)'''


X[categorical_cols] = X[categorical_cols].astype(str).fillna("missing")

# Train final model
catboost_final = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.1,
    depth=6,
    cat_features=categorical_cols,
    verbose=100
)

catboost_final.fit(X, y)


test_df['amenities'] = test_df['amenities'].apply(safe_json)
test_df['amenities_filtered'] = test_df['amenities'].apply(
    lambda items: [a for a in items if a in top_amenities]
)


amenity_features = {
    f'amenity_{amenity}': test_df['amenities'].apply(lambda x: amenity in x)
    for amenity in top_amenities
}
amenity_df = pd.DataFrame(amenity_features)

test_df['bathrooms_type'] = test_df['bathrooms_text'].str.extract(r'[\d\.]+\s+(.*)')

test_df['host_since'] = pd.to_datetime(test_df['host_since'], errors='coerce')
test_df['first_review'] = pd.to_datetime(test_df['first_review'], errors='coerce')

test_df['host_duration_years'] = (today - test_df['host_since']) / pd.Timedelta(days=365)
test_df['first_review_years'] = (today - test_df['first_review']) / pd.Timedelta(days=365)

test_df['host_is_superhost'] = test_df['host_is_superhost'].map({'t': 1, 'f': 0})
test_df['host_identity_verified'] = test_df['host_identity_verified'].map({'t': 1, 'f': 0})

test_df['num_amenities'] = train_df['amenities'].apply(len)

test_df['review_scores_avg'] = test_df[review_scores].mean(axis=1)

for col in categorical_cols:
    test_df[col] = test_df[col].astype('category')


'''
missing_amenities = [col for col in amenity_cols if col not in test_df.columns]
test_df = pd.concat([test_df, pd.DataFrame(0, index=test_df.index, columns=missing_amenities)], axis=1)

X_test_final = test_df[numerical_cols + categorical_cols + amenity_cols].copy()

for col in categorical_cols:
    X_test_final[col] = X_test_final[col].astype('category')

test_preds = xgb_final.predict(X_test_final)

# Save submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'price': test_preds
})
submission.to_csv('submission.csv', index=False)
'''

missing_amenities = [col for col in amenity_cols if col not in test_df.columns]
test_df = pd.concat([test_df, pd.DataFrame(0, index=test_df.index, columns=missing_amenities)], axis=1)

X_test_final = test_df[numerical_cols + categorical_cols + amenity_cols].copy()
X_test_final[categorical_cols] = X_test_final[categorical_cols].astype(str).fillna("missing")


for col in categorical_cols:
    X_test_final[col] = X_test_final[col].astype('category')

test_preds = catboost_final.predict(X_test_final)

# Save submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'price': test_preds
})
submission.to_csv('submission.csv', index=False)

