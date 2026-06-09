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


import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test =pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


print(train.head())
print(train.info())


train.isnull().sum()


plt.hist(x="Calories", data = train)
plt.title("Calories Distribution")
plt.show()


from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()
train["Sex"] = encoder.fit_transform(train["Sex"])
test["Sex"] = encoder.fit_transform(test["Sex"])


X=train.drop(columns=["id","Calories"])
y=train[["Calories"]]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb


model = lgb.LGBMRegressor()
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)

# Evaluate model
rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R² Score:", r2)


#test dataset
test_df=test.drop(columns=["id"])
predictions = model.predict(test_df)
predictions


#submission
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': predictions
})

# Save
submission.to_csv('submission.csv', index=False)

