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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')


submission_path = '/kaggle/input/playground-series-s5e5/sample_submission.csv'
train_path = '/kaggle/input/playground-series-s5e5/train.csv'
test_path = '/kaggle/input/playground-series-s5e5/test.csv'


submission_data = pd.read_csv(submission_path)
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


train_data.shape, test_data.shape, submission_data.shape


df = train_data.copy()


col = ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp','Calories']

plt.figure(figsize=(12,12))

for i in range(len(col)):
    plt.subplot(3, 3, i + 1)  # 3x3 grid for 7 plots
    sns.histplot(np.log1p(df[col[i]]), kde=True, bins=30, color='skyblue')
    plt.title(f'Log Distribution of {col[i]}')

plt.tight_layout()
plt.show()


sex = {'male' : 1 , 'female' : 0}

# train data
df['Sex'] = df['Sex'].map(sex)

# test data
test_data['Sex'] = test_data['Sex'].map(sex)


'''# train datatset
df['height_m'] = df['Height'] / 100
df['BMI'] = df['Weight'] / (df['height_m'] ** 2)
df['Cardio_Load'] = df['Heart_Rate'] * df['Duration']
# df['Fever_Flag'] = (df['Body_Temp'] > 37.5).astype(int)

# test dataset
test_data['height_m'] = test_data['Height'] / 100
test_data['BMI'] = test_data['Weight'] / (test_data['height_m'] ** 2)
test_data['Cardio_Load'] = test_data['Heart_Rate'] * test_data['Duration']
# test_data['Fever_Flag'] = (test_data['Body_Temp'] > 37.5).astype(int)

# train dataset
df2 = df.drop(['height_m','Height','Weight','Heart_Rate','Duration','Body_Temp'], axis=1)

# test dataset
test_data = test_data.drop(['height_m','Height','Weight','Heart_Rate','Duration','Body_Temp'], axis=1)'''


# train dataset
df3 = df.drop(['id'], axis=1)

# test dataset
test_data = test_data.drop(['id'], axis=1)

corr = df3.corr()
sns.heatmap(corr, linewidth = 0.5, annot= True)


from sklearn.preprocessing import StandardScaler


X = df3.drop(['Calories'], axis=1)
y = df3['Calories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Initialize scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


xgb_model = XGBRegressor(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print(f"XGBoostRegressor → R²: {r2_score(y_test, xgb_pred):.4f}, RMSE: {mean_squared_error(y_test, xgb_pred, squared=False):.2f}")


# Initialize scaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
test_data = scaler.transform(test_data)


xgb_model.fit(X, y)

y_pred = xgb_model.predict(test_data)
y_pred = np.clip(y_pred, 1, 314) # Fix negative predictions (clamp to 0)

pred_df = pd.DataFrame({
    'id': range(750000, 750000 + len(y_pred)),
    'Calories': y_pred
})

# Save to CSV
pred_df.to_csv('submission.csv', index=False)


submission = pd.read_csv('/kaggle/working/submission.csv')
submission.head()




