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


# import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# Read Data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


# Train Data
df_test.head()


# Test Data
df_test.head()


# Sample Submission Data
submission.head()


# Prepare Features and target
X = df_train.drop(columns=['id','accident_risk'], axis=1)
y = df_train['accident_risk']
X_test = df_test.drop(columns=['id'], axis=1)


# Define feature types
categorical = ['road_type', 'lighting', 'weather', 'time_of_day']
boolean = ['road_signs_present', 'public_road', 'holiday', 'school_season']
numeric = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']



#Preprocessor
preprocessor = ColumnTransformer([
    ('categ', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ('bool', 'passthrough', boolean),
    ('num', 'passthrough', numeric)
])


# Define Model
model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=.1,
                    max_depth=5,
                    random_state=42))
])



# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model.fit(X_train, y_train)



# Make predictions
y_pred = model.predict(X_val)


# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
print("RMSE: ", rmse)
print("R2: ", r2)


y_pred2 = model.predict(X_test)
submission['accident_risk'] = y_pred2
submission.to_csv('submission.csv', index=False)
print(submission.head())

