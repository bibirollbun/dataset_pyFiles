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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import joblib

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

print('Has LightGBM:', HAS_LGB)


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

print('Train shape:', train.shape)
print('Test shape:', test.shape)

train.head()

plt.figure(figsize=(8,4))
sns.histplot(train['accident_risk'], bins=50, kde=True)
plt.title('accident_risk appearance')
plt.show()

print('\ Number of missing values (train):')
print(train.isnull().sum())

print('\nNumber of unique values ​​of category variables:')
cat_cols = ['road_type','lighting','weather','time_of_day']
for c in cat_cols:
    if c in train.columns:
        print(c, ':', train[c].nunique())




def basic_clean_and_features(df):
    df = df.copy()
    for c in ['num_lanes','curvature','speed_limit','num_reported_accidents']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    bool_cols = ['road_signs_present','public_road','holiday','school_season']
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].map({True:1, False:0, 'True':1, 'False':0, 'true':1, 'false':0}).astype('float')

    if 'lighting' in df.columns:
        df['is_night'] = df['lighting'].astype(str).str.contains('night', case=False, na=False).astype(float)
    else:
        df['is_night'] = 0.0

    if 'time_of_day' in df.columns:
        df['is_rush_hour'] = df['time_of_day'].astype(str).isin(['morning','evening']).astype(float)
    else:
        df['is_rush_hour'] = 0.0

    if 'num_lanes' in df.columns and 'speed_limit' in df.columns:
        df['lanes_x_speed'] = df['num_lanes'] * df['speed_limit']
    else:
        df['lanes_x_speed'] = 0.0

    if 'curvature' in df.columns:
        df['curvature_sq'] = df['curvature'] ** 2
    else:
        df['curvature_sq'] = 0.0

    return df

train = basic_clean_and_features(train)
test = basic_clean_and_features(test)

id_col = 'id'
target_col = 'accident_risk'

features = [c for c in train.columns if c not in [id_col, target_col]]
print('Number of features to be used:', len(features))
print(features)


available_cols = train.columns.tolist()

numeric_features = [c for c in ['num_lanes','curvature','speed_limit','num_reported_accidents','lanes_x_speed','curvature_sq'] if c in available_cols]
bool_features = [c for c in ['road_signs_present','public_road','holiday','school_season','is_night','is_rush_hour'] if c in available_cols]
cat_features = [c for c in ['road_type','lighting','weather','time_of_day'] if c in available_cols]

print("Numeric features:", numeric_features)
print("Boolean features:", bool_features)
print("Categorical features:", cat_features)



from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

bool_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('bool', bool_pipeline, bool_features),
    ('cat', cat_pipeline, cat_features)
])

print('Numeric:', numeric_features)
print('Bool:', bool_features)
print('Cat:', cat_features)



target_col = 'accident_risk'
id_col = 'id'  

features = numeric_features + bool_features + cat_features



from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

if HAS_LGB:
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.2,
        min_child_samples=25,
        random_state=42,
        n_jobs=-1
    )
else:
    model_lgb = RandomForestRegressor(
        n_estimators=400,
        max_depth=14,
        random_state=42,
        n_jobs=-1
    )

model_rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

pipe_lgb = Pipeline([
    ('preproc', preprocessor),
    ('model', model_lgb)
])

pipe_rf = Pipeline([
    ('preproc', preprocessor),
    ('model', model_rf)
])

print("Training LightGBM model...")
pipe_lgb.fit(train[features], train[target_col])

print("Training Random Forest model...")
pipe_rf.fit(train[features], train[target_col])

print("Predicting...")
preds_lgb = pipe_lgb.predict(test[features])
preds_rf = pipe_rf.predict(test[features])

preds = 0.65 * preds_lgb + 0.35 * preds_rf
preds = np.clip(preds, 0, 1)

submission = pd.DataFrame({
    id_col: test[id_col].astype(int),
    'accident_risk': preds
})
submission.to_csv('submission.csv', index=False)

print("✅ Submission file created:")
print(submission.head())





