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

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold


submission_path = '/kaggle/input/playground-series-s5e5/sample_submission.csv'
train_path = '/kaggle/input/playground-series-s5e5/train.csv'
test_path = '/kaggle/input/playground-series-s5e5/test.csv'


submission_data = pd.read_csv(submission_path)
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


train_data.shape, test_data.shape, submission_data.shape


df = train_data.copy()


def age_group(age):
    if age < 13:
        return 'child'
    elif 13 <= age < 20:
        return 'teen'
    elif 20 <= age < 40:
        return 'young_adult'
    elif 40 <= age < 60:
        return 'middle_aged'
    else:
        return 'senior'

df['age_group'] = df['Age'].apply(age_group)
test_data['age_group'] = test_data['Age'].apply(age_group)


test_data


# test data
age_dummies = pd.get_dummies(test_data['age_group'], drop_first = True)
test_data = pd.concat([test_data, age_dummies], axis=1)

test_data.drop(['age_group'], axis=1, inplace=True)


# train data
age_dummies = pd.get_dummies(df['age_group'], drop_first = True)
df = pd.concat([df, age_dummies], axis=1)

df.drop(['age_group'], axis=1, inplace=True)


df.tail()


sex = {'male' : 1 , 'female' : 0}


df['Sex'] = df['Sex'].map(sex)
test_data['Sex'] = test_data['Sex'].map(sex)


test_data.tail()


# train datatset
df['height_m'] = df['Height'] / 100
df['BMI'] = df['Weight'] / (df['height_m'] ** 2)
df['Cardio_Load'] = df['Heart_Rate'] * df['Duration']

# test dataset
test_data['height_m'] = test_data['Height'] / 100
test_data['BMI'] = test_data['Weight'] / (test_data['height_m'] ** 2)
test_data['Cardio_Load'] = test_data['Heart_Rate'] * test_data['Duration']


# train dataset
df2 = df.drop(['height_m','Height','Weight','Heart_Rate','Duration','Body_Temp'], axis=1)

# test dataset
test_data = test_data.drop(['height_m','Height','Weight','Heart_Rate','Duration','Body_Temp'], axis=1)


corr = df2.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr,linewidth=0.5, annot=True)


# train dataset
df3 = df2.drop(['id'], axis=1)

# test dataset
test_data = test_data.drop(['id'], axis=1)


corr = df3.corr()
sns.heatmap(corr,linewidth=0.5, annot= True)


X = df3.drop(['Calories'], axis=1)
y = df3['Calories']


corr = X.corr()
sns.heatmap(corr,linewidth=0.5, annot=True)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestRegressor()

rf_model.fit(X_train, y_train)
rf_model.score(X_test, y_test)


rf_model = RandomForestRegressor()
rf_model.fit(X, y)

y_pred = rf_model.predict(test_data)
y_pred = np.clip(y_pred, 1, 314) # Fix negative predictions (clamp to 1)

pred_df = pd.DataFrame({
    'id': range(750000, 750000 + len(y_pred)),
    'Calories': y_pred
})

# Save to CSV
pred_df.to_csv('submission.csv', index=False)


submission = pd.read_csv('/kaggle/working/submission.csv')
submission.tail()

