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


import numpy as np
import pandas as pd
import ydata_profiling as pp
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error

import warnings 
warnings.filterwarnings("ignore")


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


profile_train = pp.ProfileReport(train)
profile_train


# --- Solving duplicates
train = train.groupby(['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']).agg({'Calories':'mean'}).reset_index()
len(train)


def feature_engineering(df):
    # -- BMI
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2

    # -- BMR (Harris–Benedict equations revised by Mifflin and St Jeor in 1990) 
    df.loc[df['Sex']==('male'), 'BMR'] = 10 * df['Weight'] + 6.25 * df['Height'] - 5 * df['Age'] + 5
    df.loc[df['Sex']==('female'), 'BMR'] = 10 * df['Weight'] + 6.25 * df['Height'] - 5 * df['Age'] - 161

    # -- Age Segment
    bin_age = [0, 18, 30, 60, float('inf')]
    label_age = ['Teen', 'Young Adult', 'Adult', 'Senior']
    df['Age_Segment'] = pd.cut(df['Age'], bins=bin_age, labels=label_age, right=False)

    # -- BMI Segment
    bin_BMI = [0, 16.5, 18.5, 25, 30, 35, 40, float('inf')]
    label_BMI = ['Severly underweight', 'Underweight', 'Normal', 'Overweight', 'Obesity class I', 'Obesity class II','Obesity class III']
    df['BMI_Segment'] = pd.cut(df['Age'], bins=bin_BMI, labels=label_BMI, right=False)

    # -- Exercise level
    df['Exercise_level'] = df['Heart_Rate'] * df['Duration']

    # -- Max heart rate, Heart Rate Ratio & Exercise Segment
    df['Max_heart_rate'] = 220 - df['Age'] # Haskell & Fox formula
    df['Heart_rate_ratio'] = df['Heart_Rate'] / df['Max_heart_rate']
    bin_exercise = [0, 0.5, 0.6, 0.7, 0.85, float('inf')]
    label_exercise = ['Resting', 'Moderate', 'Aerobic', 'Anaerobic', 'Redline']
    df['Exercise_Segment'] = pd.cut(df['Heart_rate_ratio'], bins=bin_exercise, labels=label_exercise, right=False)
    
    return df


feature_engineering(train)
feature_engineering(test)


numerical_feature = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI', 'BMR', 'Exercise_level', 'Max_heart_rate', 'Heart_rate_ratio']
categorical_feature = ['Sex', 'Age_Segment', 'BMI_Segment', 'Exercise_Segment']

# -- StandardScaler
sc = StandardScaler()
train[numerical_feature] = sc.fit_transform(train[numerical_feature])
test[numerical_feature] = sc.transform(test[numerical_feature])

# -- Label Encoder
le = LabelEncoder()
for col in categorical_feature:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


X_train = train.drop('Calories', axis = 1)
y_true = train['Calories']
y_train = np.log1p(train['Calories'])
X_test = test.copy()

# -- XGBoost
xgb_model = XGBRegressor(objective = 'reg:squarederror',
                         n_estimators = 5000,
                         learning_rate = 0.1,
                         random_state = 42)
xgb_model.fit(X_train, y_train)

# Predict
y_train_predict_log = xgb_model.predict(X_train)
y_test_predict_log = xgb_model.predict(X_test)

y_train_predict = np.expm1(y_train_predict_log)
y_test_predict = np.expm1(y_test_predict_log)

# Evaluate
RMSLE_train = np.sqrt(mean_squared_log_error(y_true, y_train_predict))
print('RMSLE: ', RMSLE_train)


# -- Save test prediction
submission['Calories'] = y_test_predict
submission.to_csv('submission.csv', index = False)
print('Save file successfully')

