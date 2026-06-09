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



train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
train_df.head()



test_df.head()


submission_df.head()


train_df['Age'] = train_df['Age'].astype(float)


train_df = pd.get_dummies(train_df, columns=['Sex'], drop_first=True)


train_df


X = train_df.drop(['id', 'Calories','Sex_male'], axis=1)
y = train_df['Calories']


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = test_df.drop(['id','Sex'], axis=1)




model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)






predictions = model.predict(X_test)

# Save to sample submission format
submission_df['Calories'] = predictions
submission_df.to_csv('submission.csv', index=False)

# To preview the result
print(submission_df.head())


val_predictions = model.predict(X_val)

# Calculate Mean Squared Error
mse = mean_squared_error(y_val, val_predictions)
print("Mean Squared Error (MSE) on validation set:", mse)

