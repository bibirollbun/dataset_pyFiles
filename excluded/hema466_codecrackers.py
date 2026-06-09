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


# STEP 1: Imports
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# STEP 2: Load data
train = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')
test = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/test.csv')
val = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/val.csv')
sample = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/sample_submission.csv')

# STEP 3: Use only available + meaningful features
features = [
    'Grid_Position',
    'Avg_Speed_kmh',
    'Track_Condition',
    'Humidity_%',
    'Championship_Position',
    'Corners_per_Lap',
    'Tire_Degradation_Factor_per_Lap',
    'Pit_Stop_Duration_Seconds',
    'Ambient_Temperature_Celsius',
    'Track_Temperature_Celsius'
]

# STEP 4: Clean data
train = train.dropna(subset=features + ['Lap_Time_Seconds'])
test = test.fillna(0)

# Convert categorical feature (Track_Condition) to numeric
for df in [train, test, val]:
    df['Track_Condition'] = df['Track_Condition'].astype('category').cat.codes

# STEP 5: Train model
X_train = train[features]
y_train = train['Lap_Time_Seconds']

model = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# STEP 6: Predict on test
X_test = test[features]
y_pred = model.predict(X_test)

# STEP 7: Export solution.csv
solution = sample.copy()
solution['Lap_Time_Seconds'] = y_pred
solution.to_csv('solution.csv', index=False)
print("✅ Final solution.csv created using sample_submission!")


# STEP 8: Optional validation
val = val.dropna(subset=features + ['Lap_Time_Seconds'])
X_val = val[features]
y_val = val['Lap_Time_Seconds']
val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"✅ Validation RMSE: {rmse:.4f}")


