# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory




train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
print("Train shape:", train.shape )
train.head()




test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
print("Test shape:", test.shape )
test.head()


print(train.info())
print("train datasets describtion: \n", train.describe())


print("train missing values:", train.isnull().sum().sum())

print("test missing values:", test.isnull().sum().sum())


plt.figure()
sns.histplot(train['accident_risk'], kde=True, bins=50)
plt.title('Distribution of Accident Risk')
plt.xlabel('Accident Risk')
plt.show()


plt.figure()
numeric_cols = train.select_dtypes(include=[np.number]).columns
correlation_matrix = train[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# Feature engineering
df = train.copy()
df['high_speed'] = (df['speed_limit'] >= 60).astype(int)
df['curvature_sq'] = df['curvature']**2
df['road_signs_present'] = df['road_signs_present'].astype(int)
df['public_road'] = df['public_road'].astype(int)
df['holiday'] = df['holiday'].astype(int)
df['school_season'] = df['school_season'].astype(int)

display(df[['num_lanes','speed_limit','high_speed','curvature','curvature_sq','num_reported_accidents']].describe().T) 


# Prepare features and modeling
features = ['road_type','num_lanes','curvature','speed_limit','lighting','weather','road_signs_present','public_road','time_of_day','holiday','school_season','num_reported_accidents','high_speed','curvature_sq']
target = 'accident_risk'
X = df[features].copy(); y = df[target].values
cat_cols = ['road_type','lighting','weather','time_of_day']
num_cols = [c for c in features if c not in cat_cols]

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols),
    ('num', StandardScaler(), num_cols)
])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)



xgb_pipeline = Pipeline([('pre', preprocessor), ('xgb', xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0))])
xgb_pipeline.fit(X_train, y_train)
xgb_val_preds = xgb_pipeline.predict(X_valid)
xgb_val_rmse = mean_squared_error(y_valid, xgb_val_preds, squared=False)
print('XGBoost validation RMSE:', round(xgb_val_rmse,5))




# Feature engineering
df_test = test.copy()
df_test['high_speed'] = (df_test['speed_limit'] >= 60).astype(int)
df_test['curvature_sq'] = df_test['curvature']**2
df_test['road_signs_present'] = df_test['road_signs_present'].astype(int)
df_test['public_road'] = df_test['public_road'].astype(int)
df_test['holiday'] = df_test['holiday'].astype(int)
df_test['school_season'] = df_test['school_season'].astype(int)
df_test = df_test.drop('id', axis=1)
display(df_test[['num_lanes','speed_limit','high_speed','curvature','curvature_sq','num_reported_accidents']].describe().T) 


# Prepare features and modeling
features = ['road_type','num_lanes','curvature','speed_limit','lighting','weather','road_signs_present','public_road','time_of_day','holiday','school_season','num_reported_accidents','high_speed','curvature_sq']
cat_cols = ['road_type','lighting','weather','time_of_day']
num_cols = [c for c in features if c not in cat_cols]





# prediction on test dataset
xgb_test_preds = xgb_pipeline.predict(df_test)


# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': xgb_test_preds
})

# Save submission file
submission.to_csv('submission.csv', index=False)

print(f"Submission shape: {submission.shape}")
print("\nFirst 5 rows of submission:")
display(submission.head())

