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


# ✅ Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# ✅ Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# ✅ Encode categorical variable 'Sex'
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

# ✅ Feature engineering function
def engineer_features(df):
    df['Duration_per_kg'] = df['Duration'] / df['Weight']
    df['HRxDuration'] = df['Heart_Rate'] * df['Duration']
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['BMI_squared'] = df['BMI'] ** 2
    df['Duration_squared'] = df['Duration'] ** 2
    df['WeightxDuration'] = df['Weight'] * df['Duration']
    df['AgexHeartRate'] = df['Age'] * df['Heart_Rate']
    df['Log_Duration'] = np.log1p(df['Duration'])
    df['Log_Weight'] = np.log1p(df['Weight'])
    df['Log_HeartRate'] = np.log1p(df['Heart_Rate'])
    df['Log_BodyTemp'] = np.log1p(df['Body_Temp'])
    df['Log_Age'] = np.log1p(df['Age'])
    return df

# ✅ Apply feature engineering
train = engineer_features(train)
test = engineer_features(test)

# ✅ Use log of target variable to optimize for RMSLE
train['log_Calories'] = np.log1p(train['Calories'])

# ✅ Select features (exclude target and ID)
features = [col for col in train.columns if col not in ['id', 'Calories', 'log_Calories']]

# ✅ Split features and target
X = train[features]
y = train['log_Calories']
X_test = test[features]

# ✅ Train XGBoost model
model = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=9,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# ✅ Make predictions and convert back from log scale
y_pred = model.predict(X_test)
test['Calories'] = np.expm1(y_pred)

# ✅ Create submission file
submission = test[['id', 'Calories']]
submission.to_csv('submission.csv', index=False)


