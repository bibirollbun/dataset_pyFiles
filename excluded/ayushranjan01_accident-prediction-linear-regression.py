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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")


df.head()


categorical_col = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

for col in categorical_col:
    df[col] = encoder.fit_transform(df[col])

df.head()


df = df.dropna()


X = df.drop(['id', 'accident_risk'], axis=1)
y = df[['accident_risk']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_test_pred = model.predict(X_test)



mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)


test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


for col in categorical_col:
    test_df[col] = encoder.fit_transform(test_df[col])

ids = test_df['id']
test_df = test_df.drop(['id'], axis=1)


test_pred = model.predict(test_df)


test_pred = pd.DataFrame(test_pred, columns=['accident_risk'])
combined = pd.concat([ids, test_pred], axis=1)




combined.to_csv("submission.csv", index=False)




