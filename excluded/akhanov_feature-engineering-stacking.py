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


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor


def data_processing(df):
    df = df.copy()

    # Encode 'Sex'
    # df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Drop missing values
    df = df.dropna()
    
    # Core engineered features
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['hr_ratio'] = df['Heart_Rate'] / (220 - df['Age'])

    # Interaction features
    df['duration_x_hr'] = df['Duration'] * df['Heart_Rate']
    df['temp_x_hr'] = df['Body_Temp'] * df['Heart_Rate']
    df['weight_x_hr_ratio'] = df['Weight'] * df['hr_ratio']

    #Nonlinear Transforms
    df['log_duration'] = np.log1p(df['Duration'])
    df['sqrt_hr'] = np.sqrt(df['Heart_Rate'])
    
    df['hr_per_kg'] = df['Heart_Rate'] / df['Weight']
    df['effort_temp_adj'] = df['duration_x_hr'] / df['Body_Temp']
    df['cardio_load'] = df['hr_ratio'] * df['Age']
    df['duration_bmi'] = df['Duration'] * df['BMI']

    #Ratio
    df['duration_to_temp'] = df['Duration'] / df['Body_Temp']
    df['hr_temp_ratio'] = df['Heart_Rate'] / df['Body_Temp']
    df['duration_per_age'] = df['Duration'] / df['Age']

    df['intensity_score'] = df['Heart_Rate'] * df['Duration'] / (220 - df['Age'])
    df['effort_per_kg'] = (df['Heart_Rate'] * df['Duration']) / df['Weight']
    df['cardio_efficiency'] = df['Heart_Rate'] / (0.7 * (220 - df['Age']))  # 70% of max HR
    df['power_output'] = df['Weight'] * df['Duration'] / df['BMI']

    #Drop
    df.drop(columns=['Height', 'Weight', 'Sex', 'BMI'], inplace=True)


    return df



train = data_processing(train)
test = data_processing(test)


import seaborn as sns
import matplotlib.pyplot as plt

corr = train.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr[['Calories']].sort_values(by='Calories', ascending=False), annot=True, cmap='coolwarm')
plt.title("Feature Correlation with Calories Burned")
plt.show()



train.head()


X = train.drop(columns=['Calories', 'id']).astype(np.float32).values
y = train['Calories'].astype(np.float32).values
y = np.log1p(y)


scaler = StandardScaler()
X = scaler.fit_transform(X)


stacked_model = StackingRegressor(
    estimators=[
        ('xgb', XGBRegressor(n_estimators=800, max_depth=5, learning_rate=0.03, random_state=42)),
        ('cat', CatBoostRegressor(iterations=400, depth=4, learning_rate=0.05, random_seed=42, verbose=0)),
        ('lgb', LGBMRegressor(n_estimators=600, learning_rate=0.03, min_gain_to_split=0.0, min_data_in_leaf=5, random_state=42))
    ],
    final_estimator=Ridge(alpha=0.3),
    cv=5  # internal stacking CV
)

stacked_model.fit(X, y)


# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# rmsle_scores = []

# for train_idx, val_idx in kf.split(X):
#     X_train, X_val = X[train_idx], X[val_idx]
#     y_train, y_val = y[train_idx], y[val_idx]
    
#     model = stacked_model
#     model.fit(X_train, y_train)
    
#     y_val_pred_log = model.predict(X_val)
#     score = mean_squared_log_error(np.expm1(y_val), np.expm1(y_val_pred_log)) ** 0.5
#     rmsle_scores.append(score)


X_test = test.drop(columns=['id']).astype(np.float32).values
X_test = scaler.transform(X_test)


y_test = stacked_model.predict(X_test)


y_pred = np.expm1(y_test)


submission = pd.DataFrame({
    'id': test['id'],
    'Calories': y_pred
})


submission.to_csv('submission.csv', index=False)


submission.head()


submission.head()

