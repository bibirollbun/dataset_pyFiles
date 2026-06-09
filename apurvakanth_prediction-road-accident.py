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


import numpy as np, pandas as pd,seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor



train_path = "/kaggle/input/playground-series-s5e10/train.csv"
test_path ="/kaggle/input/playground-series-s5e10/test.csv"



train = pd.read_csv(train_path)
train.head()


display(train.dtypes.to_frame('dtype').T)
display(train.describe(include='all').T)


plt.figure(figsize=(8,4))
sns.histplot(train['accident_risk'],bins=40  , kde = True)
plt.title('Accident Risk distribution')
plt.xlabel('accident_risk')
plt.show()


train.isnull().values.any()


train.isnull().sum().sum()



df=train.copy()
df['high_speed']=(df['speed_limit'] >= 60).astype(int)
df['curvature_sq']=(df['curvature'])**2
df['road_signs_present']=(df['road_signs_present']).astype(int)
df['public_road']=(df['public_road']).astype(int)
df['holiday']=(df['holiday']).astype(int)
df['school_season']=(df['school_season']).astype(int)
display(df[['num_lanes','speed_limit','high_speed','curvature','curvature_sq']].describe().T)


df.head


features = [
    'road_type','num_lanes','curvature','speed_limit','lighting','weather',
    'road_signs_present','public_road','time_of_day','holiday',
    'school_season','num_reported_accidents','high_speed','curvature_sq'
]
target='accident_risk'
X= df[features]
y=df[target].values
cat_cols=['road_type','lighting','time_of_day','weather']
num_cols=[c for c in features if c not in cat_cols]
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols),
    ('num', StandardScaler(), num_cols)
])
# Split data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)



rf_pipeline = Pipeline([
    ('pre', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_valid)
rf_rmse = mean_squared_error(y_valid, rf_preds, squared=False)
print("RandomForest Validation RMSE:", round(rf_rmse,5))


try:
    import xgboost as xgb
    has_xgb = True
except ImportError:
    has_xgb = False



if has_xgb:
    xgb_pipeline = Pipeline([
        ('pre', preprocessor),
        ('xgb', xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        ))
    ])
    xgb_pipeline.fit(X_train, y_train)
    xgb_preds = xgb_pipeline.predict(X_valid)
    xgb_rmse = mean_squared_error(y_valid, xgb_preds, squared=False)
    print("XGBoost Validation RMSE:", round(xgb_rmse,5))
else:
    print('XGBoost not available in this environment.')


if has_xgb:
    ensemble_preds = (rf_preds + xgb_preds) / 2
    ensemble_rmse = mean_squared_error(y_valid, ensemble_preds, squared=False)
    print('Ensemble RMSE:', round(ensemble_rmse,5))
else:
    print('Ensemble not created (XGBoost not available).')

# Cross-validated RF score
cv_scores = cross_val_score(rf_pipeline, X, y, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
print('RF CV RMSE (3-fold avg):', round(-cv_scores.mean(),5))


print("RF RMSE:", rf_rmse)
print("XGB RMSE:", xgb_rmse)
print("Ensemble RMSE:", ensemble_rmse)





